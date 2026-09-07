"""Read-only FML overlay inspection without model files or native libraries.

An internally consistent identity is a claim, not authentication of model shards.
The serving engine must still authenticate the actual model before using rows.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct

from .artifacts import digest

MAX_HEADER_BYTES = 4 << 20
MAX_ROWS = 131072
ROW_DIM = 160


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value, label):
    _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), f'Invalid {label} SHA-256')


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, f'{label} must be an integer >= {minimum}')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f'Nonfinite JSON number: {value}')


def _finite_float(text):
    value = float(text)
    _require(math.isfinite(value), 'Nonfinite JSON number')
    return value


def _json(raw):
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant, parse_float=_finite_float)
    _require(isinstance(value, dict), 'Expected a JSON object')
    return value


def validate_identity(identity):
    """Check embedded manifest structure/self-digest; do not access its paths."""
    _require(isinstance(identity, dict) and identity.get('schema') == 'flash-memory-model/v1',
             'Require flash-memory-model/v1 identity')
    checksum = identity.get('identity_sha256')
    _sha(checksum, 'model identity')
    _require(digest({key:value for key,value in identity.items() if key != 'identity_sha256'}) == checksum,
             'Model identity self-digest mismatch')
    shards = identity.get('shards')
    _require(isinstance(shards, list) and bool(shards), 'Missing model shard identities')
    by_path = {}
    for shard in shards:
        _require(isinstance(shard, dict), 'Invalid shard identity')
        path = shard.get('path')
        _require(isinstance(path, str) and bool(path.strip()) and path not in by_path,
                 'Shard paths must be nonempty and unique')
        _integer(shard.get('bytes'), 'Shard bytes', 1)
        _sha(shard.get('sha256'), 'shard')
        by_path[path] = shard
    for key in ('tokenizer_sha256', 'chat_template_sha256'):
        _sha(identity.get(key), key)
    engine = identity.get('engine')
    _require(isinstance(engine, dict) and isinstance(engine.get('commit'), str) and
             re.fullmatch('[0-9a-f]{40}', engine['commit']), 'Missing engine commit identity')
    _sha(engine.get('patch_sha256'), 'engine patch')
    runtime = engine.get('runtime')
    _require(isinstance(runtime, dict) and bool(runtime), 'Missing engine runtime identities')
    for value in runtime.values():
        _sha(value, 'engine runtime')
    tables = identity.get('table')
    _require(isinstance(tables, list) and len(tables) == 1 and isinstance(tables[0], dict),
             'Archived FML loader requires one joined table identity')
    table = tables[0]
    _require(table.get('path') in by_path, 'Table path must identify a declared model shard')
    _integer(table.get('offset'), 'Table offset')
    _integer(table.get('bytes'), 'Table bytes', 1)
    _integer(table.get('type'), 'Table storage type')
    _sha(table.get('sha256'), 'table')
    shape = table.get('shape')
    _require(isinstance(shape, list) and len(shape) == 2 and type(shape[0]) is int and shape[0] == ROW_DIM,
             'Table must have GGML shape [160, rows]')
    _integer(shape[1], 'Table row count', 1)
    _require(table['offset'] + table['bytes'] <= by_path[table['path']]['bytes'],
             'Table byte range exceeds declared shard')
    metadata = identity.get('architecture_metadata')
    _require(isinstance(metadata, dict), 'Missing architecture/head metadata')
    offsets = metadata.get('qwen4exp.ple.head_offsets')
    sizes = metadata.get('qwen4exp.ple.head_vocab_sizes')
    _require(isinstance(offsets, list) and isinstance(sizes, list) and offsets and len(offsets) == len(sizes),
             'Invalid head offset/size metadata')
    end = 0
    for offset, size in zip(offsets, sizes):
        _integer(offset, 'Head offset')
        _integer(size, 'Head vocabulary size', 1)
        _require(offset >= end and offset + size <= shape[1], 'Overlapping or out-of-table head ranges')
        end = offset + size
    return offsets, sizes, shape[1]


def inspect_overlay(path, identity_path=None):
    """Validate an FML file and optional manifest, never declared model files."""
    overall = hashlib.sha256()
    payload_hash = hashlib.sha256()
    with Path(path).open('rb') as stream:
        prefix = stream.read(12)
        _require(len(prefix) == 12 and prefix[:8] == b'FMLROW1\0', 'Invalid/truncated FML magic or prefix')
        header_size = struct.unpack_from('<I', prefix, 8)[0]
        _require(0 < header_size <= MAX_HEADER_BYTES, 'FML header length outside 1 byte–4 MiB bound')
        encoded = stream.read(header_size)
        _require(len(encoded) == header_size, 'Truncated FML header')
        header = _json(encoded)
        _require(header.get('schema') == 'flash-memory-overlay/v1', 'Unsupported overlay schema')
        count, dim = header.get('row_count'), header.get('row_dim')
        _integer(count, 'Overlay row count')
        _require(count <= MAX_ROWS and type(dim) is int and dim == ROW_DIM, 'Invalid FML row geometry')
        expected_bytes = count*(4 + 4*ROW_DIM)
        _require(os.fstat(stream.fileno()).st_size == 12 + header_size + expected_bytes,
                 'Truncated or trailing FML payload')
        expected_hash = header.get('payload_sha256')
        _sha(expected_hash, 'payload')
        embedded = header.get('model_identity')
        offsets, sizes, total_rows = validate_identity(embedded)
        supplied_match = None
        if identity_path is not None:
            with Path(identity_path).open('rb') as supplied_stream:
                raw = supplied_stream.read(MAX_HEADER_BYTES + 1)
            _require(len(raw) <= MAX_HEADER_BYTES, 'Supplied identity exceeds 4 MiB bound')
            supplied = _json(raw)
            validate_identity(supplied)
            _require(supplied['identity_sha256'] == embedded['identity_sha256'],
                     'Supplied and embedded model identity checksums differ')
            supplied_match = True
        raw_rows = stream.read(count*4)
        _require(len(raw_rows) == count*4, 'Truncated FML row IDs')
        rows = [value[0] for value in struct.iter_unpack('<i', raw_rows)]
        previous = -1
        for row in rows:
            _require(row >= 0 and row > previous, 'FML rows must be nonnegative, sorted, and unique')
            head = bisect_right(offsets, row) - 1
            _require(head >= 0 and row < total_rows and row < offsets[head] + sizes[head],
                     'FML row outside table/head range or in padding')
            previous = row
        overall.update(prefix); overall.update(encoded); overall.update(raw_rows)
        payload_hash.update(raw_rows)
        remaining = count*ROW_DIM*4
        while remaining:
            block = stream.read(min(1 << 16, remaining))
            _require(len(block) > 0 and len(block) % 4 == 0, 'Truncated FML values')
            _require(all(math.isfinite(value[0]) for value in struct.iter_unpack('<f', block)),
                     'Nonfinite FML row value')
            remaining -= len(block)
            overall.update(block); payload_hash.update(block)
        _require(not stream.read(1), 'Trailing FML data')
        _require(payload_hash.hexdigest() == expected_hash, 'FML payload SHA-256 mismatch')
    return {'schema': 'ngramma.overlay-inspection/v1', 'status': 'validated_file_and_manifest_only',
            'row_count': count, 'row_dim': dim, 'rows': rows,
            'overlay_sha256': overall.hexdigest(), 'payload_sha256': payload_hash.hexdigest(),
            'identity_sha256': embedded['identity_sha256'], 'supplied_identity_matched': supplied_match,
            'model_files_read': False, 'engine_model_authentication_required': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('overlay', type=Path)
    parser.add_argument('--identity', type=Path)
    args = parser.parse_args(argv)
    try:
        result = inspect_overlay(args.overlay, args.identity)
    except (OSError, ValueError, KeyError, TypeError, OverflowError, RecursionError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
