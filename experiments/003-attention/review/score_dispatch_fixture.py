"""Compare QK dispatch using saved activations only; never load model weights."""
import argparse
import ctypes as c
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('capture',type=Path)
p.add_argument('cpu_library',type=Path)
p.add_argument('attention_library',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()
meta=json.loads((a.capture/'tensors.json').read_text())
consumed={}
def get(name,occ=0):
    m=[m for m in meta if m['name']==name][occ]
    consumed[f'{name}/occurrence-{occ}']={'file':m['file'],'sha256':digest(a.capture/m['file']),
        'shape':m['shape'],'strides':m['strides'],'type':m['type']}
    return np.ndarray(tuple(reversed(m['shape'])),dtype='<f4',
        buffer=(a.capture/m['file']).read_bytes(),strides=tuple(reversed(m['strides']))).copy().squeeze()
def metric(x,y):
    d=np.abs(x-y)
    return {'max_abs':float(d.max()),'different_values':int(np.count_nonzero(d)),
        'relative_rms':float(np.sqrt(np.mean(d.astype(float)**2))/np.sqrt(np.mean(y.astype(float)**2)))}
q=get('Qcur-3');k=get('Kcur-3',1);v=get('Vcur-3',1)
t,h,d=q.shape;hk=k.shape[1];expected=get('kq-3');slots=expected.shape[-1]
lib=c.CDLL(str(a.cpu_library))
dot=lib.ggml_vec_dot_f32
dot.argtypes=[c.c_int,c.c_void_p,c.c_size_t,c.c_void_p,c.c_size_t,c.c_void_p,c.c_size_t,c.c_int]
actual=np.zeros((h,t,slots),np.float32)
for head in range(h):
    for row in range(t):
        for col in range(t):
            dst=actual.ctypes.data+((head*t+row)*slots+col)*4
            dot(d,dst,0,k[col,head//(h//hk)].ctypes.data,0,q[row,head].ctypes.data,0,1)
bridge=c.CDLL(str(a.attention_library))
mm=bridge.ngramma_batched_matmul
mm.argtypes=[c.c_void_p]*3+[c.c_int64]*5+[c.c_int]
mm.restype=c.c_int
kp=np.zeros((hk,slots,d),np.float32);kp[:,:t]=k.transpose(1,0,2)
qc=np.ascontiguousarray(q.transpose(1,0,2));out=np.empty((h,t,slots),np.float32)
assert mm(kp.ctypes.data,qc.ctypes.data,out.ctypes.data,d,slots,t,hk,h,4)==0
result={'scope':'Saved Q/K/V activations and native CPU functions only; no model execution.',
    'native_vec_dot_f32_exact_rotary':metric(actual[:,:,:t],expected[:,:,:t]),
    'contiguous_batched_matmul_exact_rotary':metric(out[:,:,:t],expected[:,:,:t])}
if hasattr(bridge,'ngramma_attention_scores'):
    scores_fn=bridge.ngramma_attention_scores
    scores_fn.argtypes=[c.c_void_p]*3+[c.c_int64]*5+[c.c_int]
    scores_fn.restype=c.c_int
    preserved=np.empty_like(out)
    q_original=np.ascontiguousarray(q)
    assert scores_fn(kp.ctypes.data,q_original.ctypes.data,preserved.ctypes.data,d,slots,t,hk,h,4)==0
    result['preserved_query_strides_exact_rotary']=metric(preserved[:,:,:t],expected[:,:,:t])
soft=bridge.ngramma_softmax_ext
soft.argtypes=[c.c_void_p]*3+[c.c_int64]*3+[c.c_float,c.c_int]
soft.restype=c.c_int
mask=np.full((t,slots),-np.inf,np.float32)
mask[:,:t]=np.where(np.arange(t)[None,:]<=np.arange(t)[:,None],0,-np.inf)
probs=np.empty_like(actual)
assert soft(actual.ctypes.data,mask.ctypes.data,probs.ctypes.data,h,t,slots,np.float32(1/np.sqrt(d)),4)==0
result['native_softmax_after_exact_dots']=metric(probs,get('kq_soft_max-3'))
vp=np.zeros((hk,d,slots),np.float32);vp[:,:,:t]=v.transpose(1,2,0)
vo=np.empty((h,t,d),np.float32)
assert mm(vp.ctypes.data,probs.ctypes.data,vo.ctypes.data,slots,d,t,hk,h,4)==0
result['native_values_after_exact_dots']=metric(vo.transpose(1,0,2).reshape(t,-1),get('attn_pregate-3').reshape(t,-1))
result['provenance']={
    'script_sha256':digest(Path(__file__)),
    'cpu_library_sha256':digest(a.cpu_library),
    'attention_library_sha256':digest(a.attention_library),
    'reference_metadata_sha256':digest(a.capture/'tensors.json'),
    'reference_tokens_sha256':digest(a.capture/'tokens.json'),
    'consumed_tensors':consumed,
    'threads':4,
}
build_record=a.attention_library.with_suffix(a.attention_library.suffix+'.build.json')
if build_record.is_file():
    record=json.loads(build_record.read_text())
    if record['library_sha256'] != result['provenance']['attention_library_sha256']:
        raise ValueError('Attention library does not match adjacent build identity')
    result['provenance']['attention_build_record_sha256']=digest(build_record)
    result['provenance']['attention_build_record']=record
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
