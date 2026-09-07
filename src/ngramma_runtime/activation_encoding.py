"""Complete dispatched Q8_0 activation bytes for non-repacked CPU PLE weights.

This replays the encoder; it does not intercept a running graph's workspace.
No gradients, dequantization, checkpoint access, or repacked layouts.
"""
from __future__ import annotations
import ctypes
from pathlib import Path
import numpy as np


class ActivationEncoding:
    metadata = {
        "weight_type": 8, "weight_type_name": "Q8_0",
        "activation_type": 8, "activation_type_name": "Q8_0",
        "block_elements": 32, "block_bytes": 34,
        "scale_offset": 0, "scale_bytes": 2, "scale_dtype": "float16",
        "codes_offset": 2, "codes_count": 32, "codes_dtype": "int8",
        "auxiliary_sums": None, "layout": "ordinary non-repacked CPU blocks",
    }

    def __init__(self, library: str | Path):
        self.library = Path(library).resolve()
        self.lib = ctypes.CDLL(str(self.library))
        self._encode = self.lib.ngramma_activation_encode_q8_0
        self._encode.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_int64, ctypes.c_int64, ctypes.c_uint64]
        self._encode.restype = ctypes.c_int

    def encode(self, weight_type: int, value: np.ndarray) -> np.ndarray:
        if isinstance(weight_type, bool) or weight_type != 8:
            raise ValueError("Only weight type8 Q8_0 ordinary CPU encoding is supported")
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float32):
            raise TypeError("Input must be a native-endian numpy float32 array")
        if value.ndim != 2 or min(value.shape) <= 0 or value.shape[1] % 32:
            raise ValueError("Input must have positive shape [T,K] with K divisible by32")
        if not np.isfinite(value).all():
            raise ValueError("Input must contain only finite values")
        contiguous = np.ascontiguousarray(value)
        rows, width = contiguous.shape
        encoded = np.empty((rows, width // 32 * 34), dtype=np.uint8)
        result = self._encode(weight_type, contiguous.ctypes.data, encoded.ctypes.data,
                              width, rows, encoded.nbytes)
        if result:
            raise RuntimeError(f"Native activation encoding failed: {result}")
        return encoded
