"""Forward-only CPU attention diagnostics, with explicit engine RoPE settings.

These operations have no backward implementation. They are deliberately not a
training backend or a replacement for the engine's sparse attention selector.
"""
from collections import Counter
import ctypes as ct
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class RopeConfig:
    mode: int
    original_context: int
    frequency_scale: float
    extension_factor: float
    attention_factor: float
    beta_fast: float
    beta_slow: float

    @classmethod
    def from_metadata(cls, metadata):
        """Resolve the audited engine defaults; refuse untested scaled models.

        The diagnostic lens supplies no RoPE overrides. In the pinned engine,
        absent scaling metadata resolves to linear with factor 1, extension 0,
        and the model training context (not the lens's small allocated context).
        See experiment 003's rope-config audit for the source trace.
        """
        prefix = 'qwen4exp.'
        allowed = {prefix+'rope.'+key for key in ('dimension_count', 'dimension_sections', 'freq_base')}
        if metadata.get('general.architecture') != 'qwen4exp' or any(key.startswith(prefix+'rope.') and key not in allowed for key in metadata):
            raise ValueError('Only audited qwen4exp RoPE without scaling overrides is supported')
        context = metadata.get(prefix+'context_length')
        if type(context) is not int or not 0 < context <= np.iinfo(np.int32).max:
            raise ValueError('A positive original training context is required')
        return cls(40, context, 1.0, 0.0, 1.0, 32.0, 1.0)


class AttentionOps:
    def __init__(self, library, *, rope_config, threads=4):
        if not isinstance(rope_config, RopeConfig) or type(threads) is not int or not 1 <= threads <= 256:
            raise ValueError('Explicit RopeConfig and positive thread count required')
        if type(rope_config.mode) is not int or rope_config.mode != 40 or type(rope_config.original_context) is not int or not 0 < rope_config.original_context <= np.iinfo(np.int32).max:
            raise ValueError('Require IMROPE mode 40 and positive int32 original context')
        self.rope_config = rope_config
        self.threads = threads
        self.calls = Counter()
        library = Path(library).resolve()
        self.library_sha256 = hashlib.sha256(library.read_bytes()).hexdigest()
        record = library.with_suffix(library.suffix + '.build.json')
        self.build_record = json.loads(record.read_text()) if record.is_file() else None
        if self.build_record and self.build_record['library_sha256'] != self.library_sha256:
            raise ValueError('Native attention library does not match its build record')
        self.lib = ct.CDLL(str(library))
        self.lib.ngramma_last_error.restype = ct.c_char_p
        self.lib.ngramma_rope_multi.argtypes = [ct.c_void_p]*4 + [ct.c_int64]*3 + [ct.c_int]*3 + [ct.c_float]*6 + [ct.c_int]
        self.lib.ngramma_rope_multi.restype = ct.c_int
        self.lib.ngramma_batched_matmul.argtypes = [ct.c_void_p]*3 + [ct.c_int64]*5 + [ct.c_int]
        self.lib.ngramma_batched_matmul.restype = ct.c_int
        self.lib.ngramma_softmax_ext.argtypes = [ct.c_void_p]*3 + [ct.c_int64]*3 + [ct.c_float, ct.c_int]
        self.lib.ngramma_softmax_ext.restype = ct.c_int

    def checked(self, status):
        if status:
            raise RuntimeError(self.lib.ngramma_last_error().decode())

    @staticmethod
    def arrays(*values):
        if torch.is_grad_enabled():
            raise RuntimeError('Native attention has no backward; enter torch.no_grad() first')
        for value in values:
            if value.requires_grad or value.device.type != 'cpu' or value.dtype != torch.float32 or not value.numel():
                raise ValueError('Native attention requires nonempty detached CPU float32 tensors')
        return [value.contiguous().numpy() for value in values]

    def rope(self, value, positions, hp):
        raw, = self.arrays(value)
        if raw.ndim != 3 or positions.device.type != 'cpu' or positions.requires_grad or tuple(positions.shape) != (raw.shape[0],):
            raise ValueError('Text RoPE expects values[T,H,D] and detached CPU positions[T]')
        pos = positions.numpy()
        if not np.isfinite(pos).all() or (pos < 0).any() or (pos > np.iinfo(np.int32).max).any() or not np.array_equal(pos, np.floor(pos)):
            raise ValueError('RoPE positions must be nonnegative int32 values')
        sections = np.asarray(hp.rope_sections)
        if sections.shape != (4,) or not np.issubdtype(sections.dtype, np.integer) or (sections < 0).any() or (sections > np.iinfo(np.int32).max//3).any():
            raise ValueError('RoPE sections must contain four bounded nonnegative integers')
        sections = sections.astype(np.int32)
        if type(hp.rope_dim) is not int or not 0 < hp.rope_dim <= raw.shape[-1] or hp.rope_dim % 2:
            raise ValueError('RoPE dimension must be a positive even integer no larger than head width')
        # Text positions repeat across the four IMROPE axes. Multimodal input
        # needs a separate interface and is not covered by this experiment.
        pos = np.tile(pos.astype(np.int32), (4, 1))
        result = np.empty_like(raw)
        cfg = self.rope_config
        self.checked(self.lib.ngramma_rope_multi(raw.ctypes.data, pos.ctypes.data,
            sections.ctypes.data, result.ctypes.data, *raw.shape, hp.rope_dim,
            cfg.mode, cfg.original_context, hp.rope_freq_base, cfg.frequency_scale,
            cfg.extension_factor, cfg.attention_factor, cfg.beta_fast, cfg.beta_slow, self.threads))
        self.calls['rope_multi'] += 1
        return torch.from_numpy(result)

    def matmul(self, weights, inputs):
        w, x = self.arrays(weights, inputs)
        if w.ndim != 3 or x.ndim != 3 or w.shape[2] != x.shape[2] or x.shape[0] % w.shape[0]:
            raise ValueError('Expected weights[Hw,N,K], inputs[Hx,M,K], with Hx divisible by Hw')
        hw, n, k = w.shape
        hx, m, _ = x.shape
        result = np.empty((hx, m, n), np.float32)
        self.checked(self.lib.ngramma_batched_matmul(w.ctypes.data, x.ctypes.data,
            result.ctypes.data, k, n, m, hw, hx, self.threads))
        self.calls['batched_matmul'] += 1
        return torch.from_numpy(result)

    def softmax(self, scores, mask, scale):
        raw, additive = self.arrays(scores, mask)
        if raw.ndim != 3 or additive.shape != raw.shape[1:]:
            raise ValueError('Expected scores[H,T,K] and additive mask[T,K]')
        if np.isnan(additive).any() or np.isposinf(additive).any() or not np.isfinite(additive).any(axis=-1).all():
            raise ValueError('Every mask row must allow at least one finite key and contain no NaN/+inf')
        result = np.empty_like(raw)
        self.checked(self.lib.ngramma_softmax_ext(raw.ctypes.data, additive.ctypes.data,
            result.ctypes.data, *raw.shape, scale, self.threads))
        self.calls['softmax_ext'] += 1
        return torch.from_numpy(result)

    def settings(self):
        return asdict(self.rope_config)
