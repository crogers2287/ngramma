"""Explicit, unadmitted straight-through estimator for CPU activation quantization.

This is NOT the exact derivative of rounding. It exists only for directional
finite-difference experiments. The production trainer does not use this module.
"""
import ctypes
import numpy as np
import torch
from torch.overrides import TorchFunctionMode

class _StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,rounded):return rounded
    @staticmethod
    def backward(ctx,gradient):return gradient,None

class ExperimentalQuantizedGradient(TorchFunctionMode):
    def __init__(self,weights):
        super().__init__();self.weights=weights;self.calls=0
        self.quantize=weights.lib.flash_memory_activation_quant
        self.quantize.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64]
        self.quantize.restype=ctypes.c_int

    def __torch_function__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        if getattr(func,'__name__','') in ('matmul','__matmul__','mm','mv') and len(args)>=2:
            left,right=args[:2]
            if isinstance(left,torch.Tensor) and isinstance(right,torch.Tensor):
                record=self.weights.quantized_weight_views.get(right.data_ptr())
                reverse=False
                if (record is None or record[0]() is None) and getattr(func,'__name__','')=='mv':
                    record=self.weights.quantized_weight_views.get(left.data_ptr());reverse=True
                if record is not None and record[0]() is not None and record[1]!=0:
                    value=right if reverse else left
                    raw=value.detach().contiguous().numpy();rounded=np.empty_like(raw)
                    rc=self.quantize(record[1],raw.ctypes.data,rounded.ctypes.data,raw.shape[-1],raw.size//raw.shape[-1])
                    if rc<0:raise ValueError('Unsupported activation type for experimental estimator')
                    converted=_StraightThrough.apply(value,torch.from_numpy(rounded))
                    args=(left,converted,*args[2:]) if reverse else (converted,right,*args[2:])
                    self.calls+=1
        return func(*args,**kwargs)
