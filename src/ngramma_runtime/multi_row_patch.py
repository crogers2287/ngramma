"""Bounded immutable existing-row edits, exported in the archived FML format.

Table metadata and selected original rows are checked without hashing model
shards. The engine must authenticate the actual model before using an overlay.
"""
from bisect import bisect_right
from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct

import numpy as np

from .artifacts import atomic_json, canonical, file_hash
from .overlay_inspect import MAX_HEADER_BYTES, inspect_overlay, validate_identity
from .row_patch import RowPatch


@dataclass(frozen=True)
class MultiRowPatch:
    """One to eight RowPatch values, copied and sorted by existing global row ID."""

    patches: tuple[RowPatch, ...]

    def __post_init__(self):
        if not isinstance(self.patches, (tuple, list)) or not 1 <= len(self.patches) <= 8:
            raise ValueError('Require one to eight row patches')
        if any(not isinstance(patch, RowPatch) for patch in self.patches):
            raise ValueError('Require RowPatch values')
        copied = tuple(sorted((RowPatch(p.row_id, p.anchor, p.replacement)
                               for p in self.patches), key=lambda p: p.row_id))
        if len({p.row_id for p in copied}) != len(copied):
            raise ValueError('Duplicate row IDs')
        object.__setattr__(self, 'patches', copied)

    @property
    def row_ids(self):
        return tuple(p.row_id for p in self.patches)

    def apply(self, addresses, gathered):
        """Check original anchors, then replace all selected occurrences in a copy."""
        addresses, gathered = np.asarray(addresses), np.asarray(gathered)
        if (addresses.ndim != 2 or addresses.dtype.kind not in 'iu'
                or gathered.dtype != np.float32 or gathered.shape != (*addresses.shape, 160)):
            raise ValueError('Require integer [tokens,heads] addresses and FP32 rows')
        if np.any(addresses < 0) or np.any(addresses >= 2**31):
            raise ValueError('Addresses must be nonnegative int32 row IDs')
        if not np.isfinite(gathered).all():
            raise ValueError('Gathered values must be finite')
        for patch in self.patches:
            selected = gathered[addresses == patch.row_id]
            expected = np.broadcast_to(patch.anchor, selected.shape)
            if selected.astype('<f4').tobytes() != expected.tobytes():
                raise ValueError(f'Gathered row differs from original anchor: {patch.row_id}')
        result = gathered.copy()
        for patch in self.patches:
            result[addresses == patch.row_id] = patch.replacement
        return result

    def _check_table(self, identity, table):
        offsets, sizes, rows = validate_identity(identity)
        declared = identity['table'][0]
        try:
            path, tensor = table.tensors['per_layer_token_embd.weight']
            actual = {'path': str(path), 'offset': int(tensor.data_offset),
                      'bytes': int(tensor.n_bytes), 'shape': [int(x) for x in tensor.shape],
                      'type': int(tensor.tensor_type)}
            if any(actual[key] != declared[key] for key in actual):
                raise ValueError('Joined table identity geometry/path mismatch')
            if declared.get('name', 'per_layer_token_embd.weight') != 'per_layer_token_embd.weight':
                raise ValueError('Joined table tensor name mismatch')
            if (table.layout != 'joined' or table.dim != 160 or table.n_rows != rows
                    or list(table.head_offsets) != offsets or list(table.head_vocab_sizes) != sizes
                    or [str(p) for p in table.paths] != [s['path'] for s in identity['shards']]):
                raise ValueError('Table layout, shard paths or head metadata mismatch')
            if any(canonical(table.metadata.get(key)) != canonical(value)
                   for key, value in identity['architecture_metadata'].items()):
                raise ValueError('Table architecture metadata mismatch')
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError('Require a table with complete ModelTable metadata') from exc
        for patch in self.patches:
            head = bisect_right(offsets, patch.row_id) - 1
            if head < 0 or patch.row_id >= rows or patch.row_id >= offsets[head] + sizes[head]:
                raise ValueError('Patch row outside original table/head range')
        original = table.read_global(list(self.row_ids))
        if (not isinstance(original, np.ndarray) or original.dtype != np.float32
                or original.shape != (len(self.patches), 160) or not np.isfinite(original).all()):
            raise ValueError('Require finite original FP32 table rows')
        for patch, row in zip(self.patches, original):
            if row.astype('<f4').tobytes() != patch.anchor.tobytes():
                raise ValueError(f'Patch anchor differs from original table: {patch.row_id}')

    def export(self, directory, identity, table, provenance):
        """Validate before creating a new directory; never overwrite an export.

        Checks selected decoded anchors and table metadata, not complete shard or
        table byte hashes. FML stores all int32 IDs followed by all FP32 rows.
        """
        self._check_table(identity, table)
        payload = struct.pack('<' + 'i'*len(self.patches), *self.row_ids)
        payload += b''.join(p.replacement.tobytes() for p in self.patches)
        header = {'schema': 'flash-memory-overlay/v1', 'model_identity': identity,
                  'row_count': len(self.patches), 'row_dim': 160, 'status': 'experiment',
                  'payload_sha256': hashlib.sha256(payload).hexdigest(), 'provenance': provenance}
        encoded = canonical(header)
        if not 0 < len(encoded) <= MAX_HEADER_BYTES:
            raise ValueError('FML header exceeds bound')
        # Inspector's strict JSON rules also apply to caller-supplied provenance.
        from .overlay_inspect import _json
        _json(encoded)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / 'rows.fml'
        with path.open('xb') as stream:
            stream.write(b'FMLROW1\0' + struct.pack('<I', len(encoded)) + encoded + payload)
        inspection = inspect_overlay(path)
        summary = {'schema': 'ngramma.multi-row-patch/v1', 'row_ids': list(self.row_ids),
                   'row_count': len(self.patches), 'identity_sha256': identity['identity_sha256'],
                   'overlay_sha256': file_hash(path), 'payload_sha256': header['payload_sha256'],
                   'rows': [{'row_id': p.row_id,
                             'anchor_sha256': hashlib.sha256(p.anchor.tobytes()).hexdigest(),
                             'replacement_sha256': hashlib.sha256(p.replacement.tobytes()).hexdigest()}
                            for p in self.patches],
                   'provenance': provenance, 'status': 'experiment',
                   'inspection': inspection, 'engine_model_authentication_required': True}
        atomic_json(directory / 'manifest.json', summary)
        return summary
