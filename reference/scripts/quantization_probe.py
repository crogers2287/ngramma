from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ctypes
import json
from flash_memory.environment import ROOT,worker_imports
worker_imports()
import numpy as np
import torch
from engraft.replica.hparams import Hparams
from engraft.replica.layers import rmsnorm_grouped,hc_mix
from engraft.replica.model import load_hc
from flash_memory.weights import EngineWeights
from flash_memory.artifacts import atomic_json
torch.set_num_threads(4)
m=json.loads((ROOT/'artifacts/model-identity.json').read_text());paths=[s['path'] for s in m['shards']]
w=EngineWeights(paths,ram_cache_bytes=1<<30);hp=Hparams.from_gguf_paths(paths[0],paths[1])
f=w.lib.flash_memory_activation_quant;f.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64];f.restype=ctypes.c_int
directory=ROOT/'artifacts/layer0-reference';meta=json.loads((directory/'tensors.json').read_text())
def reference(name):
    item=next(t for t in meta if t['name']==name);raw=(directory/item['file']).read_bytes()
    return torch.from_numpy(np.ndarray(tuple(reversed(item['shape'])),dtype='<f4',buffer=raw,strides=tuple(reversed(item['strides']))).copy().squeeze())
def linear(x,name):
    raw=x.detach().contiguous().numpy();rounded=np.empty_like(raw)
    rc=f(int(w.tensor_type(name)),raw.ctypes.data,rounded.ctypes.data,raw.shape[-1],raw.size//raw.shape[-1])
    if rc<0:raise ValueError('Activation quantization unavailable: '+str(rc))
    return torch.from_numpy(rounded)@torch.from_numpy(w.tensor(name)).T
x=reference('hc_init');expected=reference('hc_mixed-0')
norm,down,up,inject=load_hc(w,0,'attn')
plain,_=hc_mix(x,norm,down,up,inject,hp.f_norm_rms_eps,hp.hc_mult)
xn=rmsnorm_grouped(x,norm,hp.f_norm_rms_eps,hp.hc_mult)
lo=torch.nn.functional.silu(linear(xn.reshape(len(x),-1),'blk.0.hc_attn_down.weight')/hp.hc_mult)
gate=torch.sigmoid(linear(lo,'blk.0.hc_attn_up.weight')).reshape(x.shape)
quantized=(xn*gate).mean(1)
def metric(a):
    d=(a-expected).abs()
    return {'max_abs':float(d.max()),'relative_rms':float(d.square().mean().sqrt()/expected.square().mean().sqrt())}
result={'fp32_replica':metric(plain),'with_engine_activation_quantization':metric(quantized),'down_weight_type':int(w.tensor_type('blk.0.hc_attn_down.weight')),'up_weight_type':int(w.tensor_type('blk.0.hc_attn_up.weight')),'note':'Forward diagnostic only. Quantization introduces discontinuities; no straight-through gradient is accepted by this test.'}
atomic_json(ROOT/'artifacts/activation-quantization-diagnosis.json',result);print(json.dumps(result),flush=True)
