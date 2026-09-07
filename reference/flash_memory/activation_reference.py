"""CPU forward diagnostic matching ggml activation quantization.

Deliberately refuses gradient-enabled use: a detached quantizer is not a trainer,
and a straight-through estimate would need separate finite-difference evidence.
"""
import ctypes
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

class ActivationReference(TorchDispatchMode):
    def __init__(self,weights):
        super().__init__();self.weights=weights
        self.quantize=weights.lib.flash_memory_activation_quant
        self.quantize.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64]
        self.quantize.restype=ctypes.c_int
        self.calls=0

    def __enter__(self):
        if torch.is_grad_enabled():
            raise RuntimeError('CPU activation reference is forward-only; enter torch.no_grad first')
        return super().__enter__()

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        if func in (torch.ops.aten.mm.default,torch.ops.aten.mv.default):
            if func==torch.ops.aten.mm.default:
                value,weight=args[:2]
            else:
                weight,value=args[:2]  # vector @ weight.T lowers to mv(weight, vector)
            record=self.weights.quantized_weight_views.get(weight.data_ptr())
            if record is not None and record[0]() is not None:
                if value.requires_grad:raise RuntimeError('Quantized CPU reference has no admitted gradient path')
                qtype=record[1]
                if qtype!=0:
                    raw=value.detach().contiguous().numpy();rounded=np.empty_like(raw)
                    rc=self.quantize(qtype,raw.ctypes.data,rounded.ctypes.data,raw.shape[-1],raw.size//raw.shape[-1])
                    if rc<0:raise ValueError(f'Unsupported CPU activation quantization: weight type {qtype}, rc {rc}')
                    converted=torch.from_numpy(rounded)
                    args=(converted,weight,*args[2:]) if func==torch.ops.aten.mm.default else (weight,converted,*args[2:])
                    self.calls+=1
        return func(*args,**kwargs)
