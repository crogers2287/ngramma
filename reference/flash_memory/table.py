"""Read-only joined/per-head GGUF table; addressing inherited from pinned ENGRAFT.

Labels are observed uses, never a claim that a hash row has one meaning.
"""
from .environment import worker_imports
worker_imports()
from pathlib import Path
import os
import gguf
import numpy as np
from engraft.table import PleTable, HeadInfo, dequant_iq4nl

def field_value(field):
    if field.types[-1] == gguf.GGUFValueType.STRING:
        values = [bytes(field.parts[d]).decode("utf-8") for d in field.data]
    else:
        values = [v.item() for d in field.data for v in field.parts[d]]
    return values if field.types[0] == gguf.GGUFValueType.ARRAY else values[0]

class ModelTable(PleTable):
    def __init__(self, paths):
        self.paths = [Path(p).resolve() for p in paths]
        self.readers = [gguf.GGUFReader(str(p)) for p in self.paths]
        self.metadata = {k: field_value(v) for k, v in self.readers[0].fields.items() if not k.startswith("GGUF.")}
        md = self.metadata
        if md.get("general.architecture") != "qwen4exp":
            raise ValueError("Only qwen4exp has a verified addressing implementation")
        for attr in ("eos_token_id", "image_token_id", "ngram_size", "heads_per_ngram", "layer_multipliers", "head_vocab_sizes", "head_offsets"):
            setattr(self, attr, md["qwen4exp.ple." + attr])
        self.dim = int(md["qwen4exp.embedding_length_per_layer_input"])
        self.n_heads = len(self.head_vocab_sizes)
        if self.dim != 160 or self.n_heads != (self.ngram_size - 1) * self.heads_per_ngram:
            raise ValueError("Unsupported PLE geometry")
        self.tensors = {t.name: (p, t) for p, r in zip(self.paths, self.readers) for t in r.tensors}
        self.heads, self.head_paths = [], []
        joined = self.tensors.get("per_layer_token_embd.weight")
        self.layout = "joined" if joined else "per_head"
        self.row_bytes = 90
        for h, (offset, size) in enumerate(zip(self.head_offsets, self.head_vocab_sizes)):
            p, t = joined or self.tensors[f"ple_ngram_embd.{h}.weight"]
            if t.tensor_type != gguf.GGMLQuantizationType.IQ4_NL or int(t.shape[0]) != self.dim:
                raise ValueError("Only original IQ4_NL memory rows are supported")
            if (offset + size if joined else size) > int(t.shape[1]):
                raise ValueError("Table smaller than declared head range")
            self.heads.append(HeadInfo(h, size, offset, int(t.data_offset) + (offset * self.row_bytes if joined else 0)))
            self.head_paths.append(p)
        self.path = self.head_paths[0]
        self.n_rows = int(joined[1].shape[1]) if joined else self.head_offsets[-1] + self.head_vocab_sizes[-1]
        self._fds = {p: os.open(p, os.O_RDONLY) for p in set(self.head_paths)}

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds = {}

    def read_rows_raw(self, h, start, n):
        if not (0 <= h < self.n_heads and n >= 0 and 0 <= start <= start + n <= self.heads[h].vocab_size):
            raise ValueError("Row outside head bounds")
        raw = os.pread(self._fds[self.head_paths[h]], n * self.row_bytes, self.heads[h].data_offset + start * self.row_bytes)
        if len(raw) != n * self.row_bytes:
            raise ValueError("Short memory-table read")
        return np.frombuffer(raw, np.uint8).reshape(n, self.row_bytes)

    def global_addresses(self, tokens):
        if any(not isinstance(t, int) or t < 0 or t >= len(self.metadata["tokenizer.ggml.tokens"]) for t in tokens):
            raise ValueError("Invalid token ID")
        return np.asarray(self.ngram_addresses(tokens), dtype=np.int64).reshape(-1, self.n_heads) + np.asarray(self.head_offsets)

    def read_global(self, rows):
        result = []
        for row in rows:
            h = int(np.searchsorted(self.head_offsets, row, side="right") - 1)
            if h < 0 or row >= self.head_offsets[h] + self.head_vocab_sizes[h]:
                raise ValueError("Row is out of range or in table padding")
            result.append(self.read_rows(h, int(row - self.head_offsets[h]), 1)[0])
        return np.asarray(result, dtype=np.float32).reshape(-1, self.dim)

