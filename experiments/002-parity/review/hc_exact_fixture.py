"""Exact-input HC fixture arithmetic and native SiLU comparison; CPU only."""
import argparse, ctypes, ctypes.util, json
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser()
p.add_argument('fixtures',type=Path)
p.add_argument('cpu_library',type=Path)
p.add_argument('--output',required=True,type=Path)
a=p.parse_args()
torch.set_num_threads(1)
meta=json.loads((a.fixtures/'tensors.json').read_text())
def get(name,occ=0):
    m=[m for m in meta if m['name']==name][occ]
    return np.ndarray(tuple(reversed(m['shape'])),dtype='<f4',buffer=(a.fixtures/m['file']).read_bytes(),strides=tuple(reversed(m['strides']))).copy().squeeze()
def metric(a,b):
    d=np.abs(a-b)
    return {'max_abs':float(d.max()),'unequal_elements':int(np.count_nonzero(d)),'relative_rms':float(np.sqrt(np.mean(d.astype(np.float64)**2))/np.sqrt(np.mean(b.astype(np.float64)**2)))}
libm=ctypes.CDLL(ctypes.util.find_library('m'))
libm.expf.argtypes=[ctypes.c_float]
libm.expf.restype=ctypes.c_float
def sigmoid_scalar(x):
    return np.array([np.float32(1)/np.float32(np.float32(1)+np.float32(libm.expf(float(-v)))) for v in x.ravel()],np.float32).reshape(x.shape)
result={}
for i in (0,1):
    norm=get('hc_norm-0',i).reshape(10,4,2560)
    gate=get('hc_gate-0',i).reshape(10,4,2560)
    gated=norm*gate
    expected=get('hc_mixed-0',i)
    explicit=(((gated[:,0]+gated[:,1])+gated[:,2])+gated[:,3])*np.float32(.25)
    mean=torch.from_numpy(gated).mean(1).numpy()
    result[f'mixer_{i}_explicit_vs_engine']=metric(explicit,expected)
    result[f'mixer_{i}_torch_mean_vs_engine']=metric(mean,expected)
    inject=get('hc_inject-0',i)
    residual=get('hc_init') if i==0 else get('hc_combine-0')
    block=get('linear_attn_out-0') if i==0 else get('ffn_out-0')
    target=get('hc_combine-0') if i==0 else get('l_last-0')
    wtorch=torch.sigmoid(torch.from_numpy(inject)/4).numpy()*np.float32(2)
    wscalar=sigmoid_scalar(inject*np.float32(.25))*np.float32(2)
    result[f'combine_{i}_torch_sigmoid_vs_engine']=metric(residual+block[:,None,:]*wtorch[:,:,None],target)
    result[f'combine_{i}_libm_sigmoid_vs_engine']=metric(residual+block[:,None,:]*wscalar[:,:,None],target)
cpu=ctypes.CDLL(str(a.cpu_library))
silu=cpu.ggml_vec_silu_f32
silu.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
silu.restype=None
synthetic=np.linspace(-16,16,4096,dtype=np.float32)
native=np.empty_like(synthetic)
silu(synthetic.size,native.ctypes.data,synthetic.ctypes.data)
result['synthetic_native_silu_vs_torch']=metric(native,torch.nn.functional.silu(torch.from_numpy(synthetic)).numpy())
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
