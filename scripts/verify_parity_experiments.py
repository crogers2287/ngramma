#!/usr/bin/env python3
"""Check saved experiment-002 metrics for consistency, without loading a model."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT/'experiments/002-parity'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metric(value):
    for key in ('max_abs', 'relative_rms'):
        require(math.isfinite(value[key]) and value[key] >= 0, f'Invalid {key}')
    if 'different_values' in value:
        require(0 <= value['different_values'] <= value['values'], 'Invalid difference count')
        require((value['different_values'] == 0) == (value['max_abs'] == 0), 'Exactness/count disagree')


def main():
    identity = json.loads((ROOT/'data/model-identity.json').read_text())['identity_sha256']
    paths = sorted(DIRECTORY.glob('full-native-*.json')) + sorted(DIRECTORY.glob('operator-probe*.json'))
    require(paths, 'No saved parity experiments')
    for path in paths:
        record = json.loads(path.read_text())
        require(record['identity_sha256'] == identity, f'{path.name}: unexpected checkpoint')
        require(record['diagnostic_only'] is True, 'Diagnostic scope missing')
        if record.get('native_build_record'):
            require(record['native_build_record']['library_sha256'] == record['native_library_sha256'], 'Native build identity mismatch')
        if path.name.startswith('full-'):
            rows = record['intermediates']
            require({row['name'] for row in rows} == {'ple_embd', *[f'l_last-{i}' for i in range(48)]}, 'Missing/extra layer evidence')
            require(len(rows) == 49, 'Duplicate layer evidence')
            for row in rows: metric(row)
            errors = record['selected_logprob_error_by_position']
            require(len(errors) == len(record['tokens']), 'Position count mismatch')
            require(all(math.isfinite(x) and x >= 0 for x in errors), 'Invalid selected log-probability error')
            require(max(errors) == record['selected_logprob_error_max'], 'Selected error maximum mismatch')
            require(record['logit_parity'] == (max(errors) < .02), 'Logit gate changed or inconsistent')
            require(record['intermediate_parity'] == (max(x['relative_rms'] for x in rows) < .01), 'Intermediate gate changed or inconsistent')
            require(0 <= record['top1_agreement'] <= 1, 'Invalid top-token agreement')
            count = record['top1_agreement']*len(errors)
            require(abs(count-round(count)) < 1e-9, 'Top-token agreement is not an integer position count')
            require(record['admitted_for_gradients'] is False, 'Unexpected gradient admission')
            require(not record.get('sources_changed_during_run'), 'Sources changed during snapshot-qualified run')
        else:
            require(record['training_qualified'] is False, 'Unexpected training qualification')
            for value in record['metrics'].values(): metric(value)
        print(f'{path.name}: internally consistent')
    print(f'PASS: {len(paths)} saved experiment-002 records checked. No model inference or tensor comparison rerun.')


if __name__ == '__main__': main()
