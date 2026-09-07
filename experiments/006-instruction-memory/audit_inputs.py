#!/usr/bin/env python3
"""Independent artifact bindings for the frozen candidate search; no inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    exp = Path(__file__).resolve().parent
    sys.path.insert(0, str(root/'src'))
    from ngramma_runtime.overlay_inspect import inspect_overlay
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    recipes = json.loads((exp/'candidates.json').read_text())
    zero = json.loads((exp/'zero-validation.json').read_text())
    lookup = json.loads((exp/'first-candidate-lookup.json').read_text())
    baseline_path = root/'experiments/005-behavior/discovery.json'
    baseline = json.loads(baseline_path.read_text())
    ids = recipes['row_ids']
    if len(ids) != 8 or len(set(ids)) != 8 or ids != sorted(ids):
        raise ValueError('Require eight unique, ordered selected rows')
    for name, expected in recipes['inputs_sha256'].items():
        path = exp/name if name == 'tokenization.json' else root/'experiments/005-behavior'/name
        if sha(path) != expected:
            raise ValueError('Recipe input changed: '+name)
    zero_candidate = next(c for c in recipes['candidates'] if c['label'] == 'zero')
    if zero['baseline_sha256'] != sha(baseline_path) or zero['zero_overlay_sha256'] != zero_candidate['overlay_sha256']:
        raise ValueError('Zero control belongs to another baseline or recipe')
    if zero['zero_record_sha256'] != sha(exp/'zero.json') or zero['lookup_sha256'] != sha(exp/'zero-lookup.json'):
        raise ValueError('Zero evidence changed')
    first = next(c for c in recipes['candidates'] if c['label'] != 'zero')
    if lookup['overlay_sha256'] != first['overlay_sha256'] or lookup['changed_values_from_original'] != 1280 or lookup['actual_equals_intended_bytes'] is not True:
        raise ValueError('First finite edit lacks the intended native gather')
    job_ids = {j['id'] for j in baseline['jobs'] if 'generate' in j['request']}
    if len(job_ids) != 24 or set(recipes['coverage']) != job_ids:
        raise ValueError('Coverage does not match the frozen 24 tasks')
    if sum(name.startswith('dev-') for name in job_ids) != 16 or sum(name.startswith('controls-') for name in job_ids) != 8:
        raise ValueError('Unexpected task partition in search')
    inspected = []
    for candidate in recipes['candidates']:
        result = inspect_overlay(a.local/'candidates'/candidate['label']/'rows.fml')
        if result['overlay_sha256'] != candidate['overlay_sha256'] or result['rows'] != ids or result['row_count'] != 8 or result['identity_sha256'] != recipes['identity_sha256']:
            raise ValueError('Export differs from frozen recipe')
        inspected.append({'label': candidate['label'], 'overlay_sha256': result['overlay_sha256'],
                          'stored_row_format': 'absolute FP32', 'row_count': result['row_count']})
    result = {'schema': 'ngramma.instruction-input-audit/v1', 'passed': True,
              'recipe_sha256': sha(exp/'candidates.json'), 'baseline_bound': True,
              'zero_overlay_bound': True, 'unique_selected_rows': 8,
              'development_tasks': 16, 'control_tasks': 8, 'heldout_search_tasks': 0,
              'native_changed_values': 1280, 'native_lookup_bound': True,
              'inspected_overlays': inspected, 'model_files_read': False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'inspected_overlays'}))


if __name__ == '__main__':
    main()
