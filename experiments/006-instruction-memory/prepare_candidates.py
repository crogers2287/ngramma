#!/usr/bin/env python3
"""Prepare fixed finite edits from original rows; never score or train a model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def locate(tokens, phrase):
    return [i for i in range(len(tokens)-len(phrase)+1) if tokens[i:i+len(phrase)] == phrase]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--local', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from ngramma_runtime._table import ModelTable
    from ngramma_runtime.artifacts import atomic_json, file_hash
    from ngramma_runtime.row_patch import RowPatch
    from ngramma_runtime.multi_row_patch import MultiRowPatch
    root = Path(__file__).resolve().parents[2]
    exp = Path(__file__).resolve().parent
    identity = json.loads(a.manifest.read_text())
    tokenization = json.loads((exp/'tokenization.json').read_text())
    baseline = json.loads((root/'experiments/005-behavior/discovery.json').read_text())
    guidance = json.loads((root/'experiments/005-behavior/followup.json').read_text())
    for record in (tokenization, baseline, guidance):
        if record['status'] != 'complete' or record['identity_sha256'] != identity['identity_sha256']:
            raise ValueError('Require complete records from the same model')
    if a.output.exists():
        raise ValueError('Use new candidate output')
    a.local.mkdir(parents=True, exist_ok=False)
    phrase = tokenization['jobs'][0]['response']['tokens']
    hint = tokenization['jobs'][1]['response']['tokens']
    original_jobs = [j for j in baseline['jobs'] if 'generate' in j['request']]
    token_sequences = {j['id']: j['response']['prompt_tokens'] for j in original_jobs}
    positions = {name: locate(tokens, phrase) for name, tokens in token_sequences.items()}
    if any(len(indices) != 1 for indices in positions.values()) or len(phrase) < 3:
        raise ValueError('Require one full instruction span per prompt')
    table = ModelTable([s['path'] for s in identity['shards']])
    try:
        if table.ngram_size != 3 or table.heads_per_ngram != 8 or table.n_heads != 16:
            raise ValueError('Unexpected n-gram geometry')
        addresses = {name: table.global_addresses(tokens) for name, tokens in token_sequences.items()}
        first_id = original_jobs[0]['id']
        at = positions[first_id][0] + 2  # Earliest wholly-contained trigram.
        row_ids = [int(x) for x in addresses[first_id][at, 8:16]]
        for name in addresses:
            if list(map(int, addresses[name][positions[name][0]+2, 8:16])) != row_ids:
                raise ValueError('Instruction rows differ across prompts')
        anchors = table.read_global(row_ids)
        rms = np.sqrt(np.mean(anchors.astype(np.float64)**2, axis=1))
        if not np.isfinite(anchors).all() or np.any(rms <= 0):
            raise ValueError('Invalid/zero original anchor')
        hint_jobs = [j for j in guidance['jobs'] if j['id'].startswith('format/')]
        donor_ids = [set() for _ in range(8)]
        hint_spans = []
        for job in hint_jobs:
            tokens = job['response']['prompt_tokens']
            occurrences = locate(tokens, hint)
            if len(occurrences) != 1:
                raise ValueError('Require exactly one full guidance span')
            start = occurrences[0]
            rows = table.global_addresses(tokens)
            hint_spans.append({'id': job['id'], 'start': start, 'end': start+len(hint)})
            for hi in range(8):
                donor_ids[hi].update(int(x) for x in rows[start+2:start+len(hint), hi+8] if int(x) not in row_ids)
        raw = {}
        raw['rademacher'] = np.array([[1.0 if hashlib.sha256(
            f'ngramma005:seed=5005:row={row}:coord={coord}'.encode()).digest()[0] & 1 else -1.0
            for coord in range(160)] for row in row_ids], dtype=np.float64)
        raw['alternating'] = np.tile(np.where(np.arange(160) % 2 == 0, 1.0, -1.0), (8, 1))
        unavailable = []
        if all(donor_ids):
            raw['hint_contrast'] = np.stack([table.read_global(sorted(ids)).astype(np.float64).mean(axis=0)
                                            for ids in donor_ids]) - anchors.astype(np.float64)
        else:
            unavailable.append('hint_contrast: no eligible donor rows')
        directions = {}
        for name, values in raw.items():
            scale = np.sqrt(np.mean(values**2, axis=1))
            if np.any(scale <= 0) or not np.isfinite(scale).all():
                unavailable.append(name + ': invalid direction RMS')
                continue
            directions[name] = (values / scale[:, None] * rms[:, None]).astype(np.float32)
        coverage = {name: [{'row_id': row, 'positions_and_heads': np.argwhere(rows == row).tolist()}
                           for row in row_ids] for name, rows in addresses.items()}
        recipe = {'schema': 'ngramma.instruction-candidates/v1',
                  'status': 'prepared_not_scored', 'identity_sha256': identity['identity_sha256'],
                  'instruction_tokens': phrase, 'selected_trigram': phrase[:3], 'row_ids': row_ids,
                  'row_heads': list(range(8, 16)), 'anchor_rms': rms.tolist(),
                  'anchor_sha256': [sha(x.tobytes()) for x in anchors],
                  'direction_sha256': {name: sha(value.tobytes()) for name, value in directions.items()},
                  'hint_tokens': hint, 'hint_spans': hint_spans,
                  'hint_donor_rows_per_head': [sorted(ids) for ids in donor_ids],
                  'coverage': coverage, 'unavailable_directions': unavailable,
                  'source_sha256': {'prepare_candidates.py': file_hash(Path(__file__)),
                                    'row_patch.py': file_hash(root/'src/ngramma_runtime/row_patch.py'),
                                    'multi_row_patch.py': file_hash(root/'src/ngramma_runtime/multi_row_patch.py')},
                  'inputs_sha256': {'discovery.json': file_hash(root/'experiments/005-behavior/discovery.json'),
                                    'followup.json': file_hash(root/'experiments/005-behavior/followup.json'),
                                    'tokenization.json': file_hash(exp/'tokenization.json')},
                  'candidates': []}
        np.savez(a.local/'anchors-and-directions.npz', anchors=anchors, row_ids=np.array(row_ids), **directions)
        base_recipe_hash = sha(json.dumps(recipe, sort_keys=True, separators=(',', ':')).encode())
        sequence = [('zero', np.zeros_like(anchors), 0.0)]
        for name, direction in directions.items():
            for exponent in (8, 6):
                for sign in (-1, 1):
                    sequence.append((f'{name}-{"minus" if sign < 0 else "plus"}-{exponent}', direction, sign*2.0**-exponent))
        for label, direction, epsilon in sequence:
            replacement = anchors + np.float32(epsilon)*direction
            edit = MultiRowPatch(tuple(RowPatch(row, anchors[i], replacement[i]) for i, row in enumerate(row_ids)))
            manifest = edit.export(a.local/label, identity, table,
                                   {'experiment': '006-instruction-memory', 'label': label,
                                    'epsilon': epsilon, 'base_recipe_sha256': base_recipe_hash,
                                    'method': 'fixed_finite_candidate_not_gradient_training'})
            item = {'label': label, 'epsilon': epsilon, 'overlay_sha256': manifest['overlay_sha256'],
                    'payload_sha256': manifest['payload_sha256'],
                    'changed_values': int(np.count_nonzero(replacement != anchors)),
                    'relative_row_displacement_rms': (np.sqrt(np.mean((replacement.astype(np.float64)-anchors)**2, axis=1))/rms).tolist(),
                    'replacement_sha256': [sha(x.tobytes()) for x in replacement]}
            recipe['candidates'].append(item)
        atomic_json(a.output, recipe)
        print(json.dumps({'row_ids': row_ids, 'overlays': len(sequence), 'recipe_sha256': file_hash(a.output)}))
    finally:
        table.close()


if __name__ == '__main__':
    main()
