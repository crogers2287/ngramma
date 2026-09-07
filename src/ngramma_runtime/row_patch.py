"""One checked existing-row intervention; no training or model-file writes.

Export uses the archived flash-memory-overlay/v1 engine format. Its loader
authenticates the complete model shards before accepting the replacement.
The format binds local shard paths as well as content: export on the worker.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct

import numpy as np
from .artifacts import canonical, digest, atomic_json, file_hash


@dataclass(frozen=True)
class RowPatch:
    row_id: int
    anchor: np.ndarray
    replacement: np.ndarray

    def __post_init__(self):
        if type(self.row_id) is not int or not 0 <= self.row_id < 2**31:
            raise ValueError('Row ID must be a nonnegative int32 integer')
        for name in ('anchor', 'replacement'):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != (160,) or not np.isfinite(value).all():
                raise ValueError('Require 160 finite FP32 row values')
            # A bytes-backed array cannot be made writable by the caller.
            frozen = np.frombuffer(value.astype('<f4').tobytes(), dtype='<f4')
            object.__setattr__(self, name, frozen)

    @classmethod
    def from_direction(cls, table, row_id, direction, epsilon):
        direction = np.asarray(direction)
        if direction.dtype != np.float32 or direction.shape != (160,) or not np.isfinite(direction).all():
            raise ValueError('Direction must contain 160 finite FP32 values')
        if isinstance(epsilon, bool) or not np.isfinite(epsilon):
            raise ValueError('Epsilon must be finite')
        anchor = table.read_global([row_id])[0]
        with np.errstate(over='ignore', invalid='ignore'):
            replacement = anchor + np.float32(epsilon) * direction
        return cls(row_id, anchor, replacement)

    def apply(self, addresses, gathered):
        """Replace every matching row in a [tokens, heads, 160] gathered copy."""
        addresses, gathered = np.asarray(addresses), np.asarray(gathered)
        if addresses.ndim != 2 or addresses.dtype.kind not in 'iu' or gathered.dtype != np.float32 or gathered.shape != (*addresses.shape, 160):
            raise ValueError('Require integer [tokens,heads] addresses and FP32 rows')
        selected = addresses == self.row_id
        if not np.array_equal(gathered[selected], np.broadcast_to(self.anchor, gathered[selected].shape)):
            raise ValueError('Gathered row differs from patch anchor')
        result = gathered.copy()
        result[selected] = self.replacement
        return result

    def export(self, directory, identity, table, provenance):
        """Write one experimental replacement. Never overwrite a prior overlay."""
        if not np.array_equal(table.read_global([self.row_id])[0], self.anchor):
            raise ValueError('Patch anchor differs from original table')
        if identity.get('schema') != 'flash-memory-model/v1' or not identity.get('shards'):
            raise ValueError('Require a complete model identity manifest')
        unsigned = {k: v for k, v in identity.items() if k != 'identity_sha256'}
        if digest(unsigned) != identity.get('identity_sha256'):
            raise ValueError('Model identity checksum mismatch')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        payload = struct.pack('<i', self.row_id) + self.replacement.astype('<f4').tobytes()
        header = {'schema': 'flash-memory-overlay/v1', 'model_identity': identity,
                  'row_count': 1, 'row_dim': 160, 'status': 'experiment',
                  'payload_sha256': hashlib.sha256(payload).hexdigest(),
                  'provenance': provenance}
        encoded = canonical(header)
        path = directory/'rows.fml'
        path.write_bytes(b'FMLROW1\0' + struct.pack('<I', len(encoded)) + encoded + payload)
        summary = {'schema': 'ngramma.row-patch/v1', 'row_id': self.row_id,
                   'identity_sha256': identity['identity_sha256'],
                   'overlay_sha256': file_hash(path),
                   'anchor_sha256': hashlib.sha256(self.anchor.tobytes()).hexdigest(),
                   'replacement_sha256': hashlib.sha256(self.replacement.tobytes()).hexdigest(),
                   'provenance': provenance, 'status': 'experiment'}
        atomic_json(directory/'manifest.json', summary)
        return summary
