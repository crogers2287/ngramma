#!/usr/bin/env python3
"""Compare an actual captured memory gather to original rows plus one overlay."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'capture', 'overlay', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    from ngramma_runtime._table import ModelTable
    from ngramma_runtime.artifacts import atomic_json, file_hash
    from ngramma_runtime.overlay_inspect import inspect_overlay
    if a.output.exists():
        raise ValueError('Use new lookup evidence path')
    identity = json.loads(a.manifest.read_text())
    inspection = inspect_overlay(a.overlay, a.manifest)
    generation = json.loads((a.capture/'generation.json').read_text())
    tokens = generation['prompt_tokens'] + generation['generated_tokens'][:-1]
    meta = json.loads((a.capture/'tensors.json').read_text())
    parts, files = [], {}
    for item in meta:
        if item['name'] != 'ple_embd':
            continue
        if item['type'] != 0:
            raise ValueError('Require captured F32 gather')
        path = (a.capture/item['file']).resolve()
        if not path.is_relative_to(a.capture.resolve()):
            raise ValueError('Capture path escaped directory')
        raw = path.read_bytes()
        files[item['file']] = hashlib.sha256(raw).hexdigest()
        values = np.ndarray(tuple(reversed(item['shape'])), dtype='<f4', buffer=raw,
                            strides=tuple(reversed(item['strides'])))
        parts.append(values.copy().reshape(-1, 16, 160))
    actual = np.concatenate(parts, axis=0)
    if actual.shape != (len(tokens), 16, 160):
        raise ValueError('Captured gather positions differ from actual decoded history')
    table = ModelTable([s['path'] for s in identity['shards']])
    try:
        addresses = table.global_addresses(tokens)
        original = table.read_global(addresses.ravel()).reshape(actual.shape)
        raw = a.overlay.read_bytes()
        header_length = int.from_bytes(raw[8:12], 'little')
        payload = raw[12+header_length:]
        count = inspection['row_count']
        ids = np.frombuffer(payload[:4*count], dtype='<i4')
        replacements = np.frombuffer(payload[4*count:], dtype='<f4').reshape(count, 160)
        expected = original.copy()
        occurrences = []
        for row, replacement in zip(ids, replacements):
            mask = addresses == row
            expected[mask] = replacement
            occurrences.append({'row_id': int(row), 'positions_and_heads': np.argwhere(mask).tolist()})
        exact = actual.tobytes() == expected.tobytes()
        if not exact:
            raise ValueError('Captured memory values differ from intended original/replacement rows')
        atomic_json(a.output, {'schema': 'ngramma.multi-row-lookup/v1',
                              'identity_sha256': identity['identity_sha256'],
                              'overlay_sha256': inspection['overlay_sha256'],
                              'positions': len(tokens), 'row_accesses': len(tokens)*16,
                              'actual_equals_intended_bytes': exact,
                              'changed_values_from_original': int(np.count_nonzero(actual != original)),
                              'occurrences': occurrences, 'capture_files_sha256': files,
                              'generation_sha256': file_hash(a.capture/'generation.json'),
                              'gather_sha256': hashlib.sha256(actual.tobytes()).hexdigest()})
    finally:
        table.close()


if __name__ == '__main__':
    main()
