#!/usr/bin/env python3
"""Audit expected row uses across saved complete histories; no model needed."""
import argparse
import json
from pathlib import Path
import sys

from verify_results import EXP, ROOT, read, require, sha, verify
sys.path.insert(0, str(ROOT / 'src'))
from ngramma_runtime.addressing import Addressing


def audit(allow_partial=False):
    checked = verify(require_complete=not allow_partial)
    metadata_path = ROOT / 'data/model-identity.json'
    addressing = Addressing.from_metadata(read(metadata_path)['architecture_metadata'])
    native_path = EXP / 'reference-addresses.jsonl'
    reference = [json.loads(line) for line in native_path.read_text().splitlines()]
    cases = read(ROOT / 'data/compatibility-cases.json')
    offset = 0
    for case in cases:
        expected = reference[offset:offset + len(case['tokens'])]
        require([r['token'] for r in expected] == case['tokens'] and
                [r['position'] for r in expected] == list(range(len(case['tokens']))), 'Native reference sequence changed')
        require(addressing.addresses(case['tokens']) == [r['rows'] for r in expected], 'Portable addresses differ from historical native reference')
        offset += len(expected)
    require(offset == len(reference) == 39, 'Unexpected reference trace length')
    recipe = read(EXP / 'candidates.json')
    report = read(EXP / 'search-results.json')
    paths = [('original', ROOT / 'experiments/005-behavior/discovery.json'), ('zero', EXP / 'zero.json')]
    paths += [(r['label'], EXP / (r['label'] + '.json')) for r in report['results']]
    conditions = []
    for name, path in paths:
        batches = read(path)
        totals = {'tasks': 0, 'positions': 0, 'prompt_selected_hits': 0, 'decoded_selected_hits': 0}
        unexpected = []
        for job in batches['jobs']:
            if job['id'] not in recipe['coverage']:
                continue
            response = job['response']
            prompt = response['prompt_tokens']
            require(addressing.occurrences(prompt, recipe['row_ids']) == recipe['coverage'][job['id']], 'Prompt coverage differs from frozen recipe')
            # The final sampled token is not fed back before generation stops.
            tokens = prompt + response['generated_tokens'][:-1]
            occurrences = addressing.occurrences(tokens, recipe['row_ids'])
            totals['tasks'] += 1
            totals['positions'] += len(tokens)
            for occurrence in occurrences:
                for position, head in occurrence['positions_and_heads']:
                    totals['prompt_selected_hits' if position < len(prompt) else 'decoded_selected_hits'] += 1
                    if position >= len(prompt):
                        unexpected.append({'task': job['id'], 'row_id': occurrence['row_id'], 'position': position, 'head': head})
        require(totals['tasks'] == 24 and totals['prompt_selected_hits'] == 192, 'Unexpected selected-row prompt coverage')
        conditions.append({'condition': name, 'batch_sha256': sha(path), **totals, 'decoded_occurrences': unexpected})
    return {'schema': 'ngramma.instruction-address-audit/v1',
            'complete': checked['complete'], 'model_files_read': False,
            'metadata_sha256': sha(metadata_path), 'recipe_sha256': sha(EXP / 'candidates.json'),
            'search_sha256': sha(EXP / 'search-results.json'), 'native_reference_sha256': sha(native_path),
            'native_reference_tokens_matched': 39, 'native_reference_addresses_matched': 624,
            'conditions': conditions,
            'note': 'Complete-history addresses are calculated offline from saved actual token IDs. Only the historical reference and separately captured lookup witnesses are native comparisons; this is not an exhaustive map of row meaning.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    result = audit(args.allow_partial)
    if args.check and read(args.check) != result:
        raise ValueError('Saved address audit differs from replay')
    encoded = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(encoded)
    print(json.dumps({k: v for k, v in result.items() if k != 'conditions'}, indent=2) if args.output or args.check else encoded)


if __name__ == '__main__':
    main()
