#!/usr/bin/env python3
"""Replay zero-overlay comparisons and bind the actual lookup witness."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    exp = Path(__file__).resolve().parent
    original_path = root/'experiments/005-behavior/discovery.json'
    zero_path = exp/'zero.json'
    original = json.loads(original_path.read_text())
    zero = json.loads(zero_path.read_text())
    lookup = json.loads((exp/'zero-lookup.json').read_text())
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    if zero['status'] != 'complete' or zero.get('inputs_unchanged') is not True:
        raise ValueError('Incomplete zero execution')
    if lookup['overlay_sha256'] != zero['overlay_sha256'] or lookup['actual_equals_intended_bytes'] is not True or lookup['changed_values_from_original'] != 0:
        raise ValueError('Invalid zero lookup witness')
    keys = ('prompt_tokens', 'generated_tokens', 'text', 'text_bytes_hex', 'first_step',
            'stop_reason', 'stopped_on_eog', 'eog_token')
    by_id = {j['id']: j for j in zero['jobs']}
    matches = []
    for job in original['jobs']:
        current = by_id[job['id']]
        if current['request'] != job['request']:
            raise ValueError('Zero job differs from original')
        if job['request'].get('tokenize_only'):
            if current['response'] != job['response']:
                raise ValueError('Tokenizer changed')
            continue
        matches.append(all(current['response'][key] == job['response'][key] for key in keys))
    legacy_hash = sha(a.local/'zero-legacy/logits.f32')
    if not all(matches) or len(matches) != 24 or legacy_hash != 'a55a60e98074b4b6a98173ca05b65545b12c9855c7aff951e35c6bef40a7829b':
        raise ValueError('Zero changed baseline output')
    result = {'schema': 'ngramma.instruction-zero/v1', 'passed': True,
              'matching_generated_tasks': len(matches), 'matching_legacy_tokens': 17,
              'legacy_logits_sha256': legacy_hash, 'baseline_sha256': sha(original_path),
              'zero_record_sha256': sha(zero_path), 'lookup_sha256': sha(exp/'zero-lookup.json'),
              'verification_library_sha256': zero['profile']['verification_library_sha256'],
              'full_shard_verification': zero['parallel_verification'],
              'zero_overlay_sha256': zero['overlay_sha256']}
    (exp/'zero-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
