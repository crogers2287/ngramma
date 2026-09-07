"""Bounded dequantization using the exact pinned ggml routines."""
from .environment import worker_imports, ROOT
worker_imports()
import ctypes
import time
import weakref
import numpy as np
import gguf
from engraft.replica.weights import GgufWeights

class EngineWeights(GgufWeights):
    def __init__(self, paths, ram_cache_bytes=1 << 30):
        super().__init__(paths, ram_cache_bytes=ram_cache_bytes)
        self.lib = ctypes.CDLL(str(ROOT / "runtime/libflash-memory-dequant.so"))
        self.lib.flash_memory_dequant.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        self.lib.flash_memory_dequant.restype = ctypes.c_int
        self.stats = {"decoded_bytes": 0, "quantized_bytes_read": 0, "dequant_seconds": 0.0, "cache_hits": 0, "cache_misses": 0}
        self.quantized_weight_views = {}

    def decode(self, raw, qtype, shape):
        raw = np.ascontiguousarray(raw)
        output = np.empty(shape, dtype=np.float32)
        start = time.monotonic()
        rc = self.lib.flash_memory_dequant(int(qtype), raw.ctypes.data, output.ctypes.data, output.size)
        if rc:
            raise ValueError(f"ggml cannot decode type {qtype}, shape {shape}: {rc}")
        self.stats["dequant_seconds"] += time.monotonic() - start
        self.stats["decoded_bytes"] += output.nbytes
        self.stats["quantized_bytes_read"] += raw.nbytes
        self.quantized_weight_views[output.ctypes.data] = (weakref.ref(output), int(qtype))
        return output

    def tensor(self, name):
        if name in self._ram_cache:
            self.stats["cache_hits"] += 1
            self._touch_ram(name)
            return self._ram_cache[name]
        self.stats["cache_misses"] += 1
        _, t = self._index[name]
        arr = self.decode(t.data, t.tensor_type, tuple(reversed(t.shape)))
        if self.ram_cache_bytes:
            self._store_ram(name, arr)
        return arr

    def _dequant_expert(self, t, e):
        n = int(t.shape[-1])
        if not 0 <= e < n:
            raise ValueError("Expert out of range")
        raw = np.asarray(t.data).reshape(n, -1)[e]
        return self.decode(raw, t.tensor_type, tuple(reversed(t.shape[:-1])))

    def embedding_rows(self, tokens):
        _, t = self._index["token_embd.weight"]
        raw = np.asarray(t.data).reshape(int(t.shape[1]), -1)[tokens]
        return self.decode(raw, t.tensor_type, (len(tokens), int(t.shape[0])))
