"""Optional exact-engine forward primitives for numerical diagnosis only.

There is no backward implementation. The context refuses gradient-enabled use.
Original ENGRAFT functions are restored when the context exits.
"""
import ctypes
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import weakref

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from .weights import EngineWeights


class NativeEngineWeights(EngineWeights):
    def __init__(self, *args, **kwargs):
        self.native_weight_views = {}
        super().__init__(*args, **kwargs)

    def decode(self, raw, qtype, shape):
        contiguous = np.ascontiguousarray(raw)
        decoded = super().decode(contiguous, qtype, shape)
        self.native_weight_views[decoded.ctypes.data] = (
            weakref.ref(decoded), contiguous, int(qtype), tuple(shape))
        return decoded


class NativeForward(TorchDispatchMode):
    def __init__(self, weights, library, *, primitives=True, matmul=True, recurrent=False, repack=False, threads=4):
        super().__init__()
        self.weights = weights
        self.primitives = primitives
        self.matmul = matmul
        self.recurrent = recurrent
        self.repack = repack
        self.threads = threads
        self.calls = Counter()
        library = Path(library).resolve()
        self.library_sha256 = hashlib.sha256(library.read_bytes()).hexdigest()
        self.lib = ctypes.CDLL(str(library))
        self.lib.ngramma_matmul.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
        self.lib.ngramma_matmul.restype = ctypes.c_int
        if repack:
            self.lib.ngramma_matmul_repack.argtypes = self.lib.ngramma_matmul.argtypes
            self.lib.ngramma_matmul_repack.restype = ctypes.c_int
            self.lib.ngramma_last_buffer_type.restype = ctypes.c_char_p
        record = library.with_suffix(library.suffix+'.build.json')
        self.build_record = json.loads(record.read_text()) if record.is_file() else None
        if self.build_record and self.build_record['library_sha256'] != self.library_sha256:
            raise ValueError('Native library does not match its adjacent build record')
        self.lib.ngramma_unary.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_float, ctypes.c_int]
        self.lib.ngramma_unary.restype = ctypes.c_int
        self.lib.ngramma_last_error.restype = ctypes.c_char_p
        if recurrent:
            self.lib.ngramma_ssm_conv.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int64]*3 + [ctypes.c_int]
            self.lib.ngramma_ssm_conv.restype = ctypes.c_int
            self.lib.ngramma_gdn.argtypes = [ctypes.c_void_p]*8 + [ctypes.c_int64]*4 + [ctypes.c_int]
            self.lib.ngramma_gdn.restype = ctypes.c_int
        self.stack = None

    def checked(self, status):
        if status:
            message = self.lib.ngramma_last_error()
            raise RuntimeError(message.decode() if message else f'Native graph failed: {status}')

    def unary(self, value, op, eps=0.0):
        if value.requires_grad or value.device.type != 'cpu' or value.dtype != torch.float32:
            raise ValueError('Native forward requires detached CPU float32 inputs')
        raw = value.detach().contiguous().numpy()
        output = np.empty_like(raw)
        self.checked(self.lib.ngramma_unary(op, raw.ctypes.data, output.ctypes.data,
                     raw.shape[-1], raw.size//raw.shape[-1], eps, self.threads))
        self.calls[f'unary_{op}'] += 1
        return torch.from_numpy(output)

    def __enter__(self):
        if torch.is_grad_enabled():
            raise RuntimeError('NativeForward is diagnostic-only and has no backward; enter torch.no_grad() first')
        self.stack = ExitStack()
        if self.primitives:
            import engraft.replica.layers as layers
            def rmsnorm(value, weight, eps):
                return self.unary(value, 0, eps) * weight
            def l2norm(value, eps, dim=-1):
                if dim not in (-1, value.ndim-1):
                    moved = value.movedim(dim, -1)
                    return self.unary(moved, 3, eps).movedim(-1, dim)
                return self.unary(value, 3, eps)
            self.stack.enter_context(patch.object(layers, 'rmsnorm', rmsnorm))
            self.stack.enter_context(patch.object(layers, 'l2norm', l2norm))
        if self.recurrent:
            import engraft.replica.layers as layers
            original_conv = layers.causal_depthwise_conv
            def conv(value, history, weight, dilation=1):
                if dilation != 1:
                    self.calls['fallback_dilated_conv'] += 1
                    return original_conv(value, history, weight, dilation)
                full = torch.cat([history, value]).T.contiguous().numpy()
                kernel = weight.contiguous().numpy()
                output = np.empty(value.shape, np.float32)
                self.checked(self.lib.ngramma_ssm_conv(full.ctypes.data, kernel.ctypes.data,
                    output.ctypes.data, len(value), value.shape[1], weight.shape[1], self.threads))
                self.calls['ssm_conv'] += 1
                return torch.from_numpy(output)
            def recurrence(q, k, v, g_log, beta, state):
                arrays = [x.contiguous().numpy() for x in (q,k,v,g_log,beta)]
                s = state.transpose(-1,-2).contiguous().numpy()
                output = np.empty(v.shape, np.float32)
                new_state = np.empty_like(s)
                self.checked(self.lib.ngramma_gdn(*[x.ctypes.data for x in arrays],
                    s.ctypes.data, output.ctypes.data, new_state.ctypes.data,
                    q.shape[0],q.shape[1],v.shape[1],q.shape[2],self.threads))
                self.calls['gated_delta_net'] += 1
                return torch.from_numpy(output),torch.from_numpy(new_state).transpose(-1,-2).contiguous()
            self.stack.enter_context(patch.object(layers,'causal_depthwise_conv',conv))
            self.stack.enter_context(patch.object(layers,'gated_delta_net_recurrence',recurrence))
        return super().__enter__()

    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            if self.stack is not None:
                self.stack.close()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self.primitives:
            unary_ops = {torch.ops.aten.silu.default:1, torch.ops.aten.sigmoid.default:2,
                         torch.ops.aten.exp.default:5}
            if func in unary_ops:
                return self.unary(args[0], unary_ops[func])
            if func == torch.ops.aten._softmax.default:
                value, dim, half_to_float = args
                if half_to_float or dim not in (-1, value.ndim-1):
                    raise ValueError('Native softmax supports only the last CPU float32 dimension')
                return self.unary(value, 4)
            if func == torch.ops.aten.softplus.default:
                # ggml softplus uses its native branch threshold. This is a
                # reference replacement only for ENGRAFT's default parameters.
                if (len(args)>1 and args[1] != 1) or (len(args)>2 and args[2] != 20):
                    raise ValueError('Native softplus supports default beta/threshold only')
                return self.unary(args[0], 6)
        if self.matmul and func in (torch.ops.aten.mm.default, torch.ops.aten.mv.default):
            if func == torch.ops.aten.mm.default:
                value, weight = args[:2]
            else:
                weight, value = args[:2]
            record = self.weights.native_weight_views.get(weight.data_ptr())
            if record is not None and record[0]() is not None:
                _, raw_weights, qtype, shape = record
                if len(shape)>2 or value.shape[-1] != shape[-1]:
                    raise ValueError('Unsupported transformed native weight view')
                raw = value.detach().contiguous().numpy()
                width = shape[-1]
                outputs = int(np.prod(shape[:-1])) if len(shape)>1 else 1
                rows = raw.size//width
                output = np.empty((rows, outputs), dtype=np.float32)
                operation = self.lib.ngramma_matmul_repack if self.repack else self.lib.ngramma_matmul
                self.checked(operation(qtype, raw_weights.ctypes.data,
                    raw.ctypes.data, output.ctypes.data, width, outputs, rows, self.threads))
                self.calls[f'matmul_type_{qtype}'] += 1
                if self.repack:
                    self.calls['buffer_'+self.lib.ngramma_last_buffer_type().decode()] += 1
                return torch.from_numpy(output if value.ndim>1 else output.reshape(-1))
            self.calls['unregistered_mm_or_mv'] += 1
        return func(*args, **kwargs)
