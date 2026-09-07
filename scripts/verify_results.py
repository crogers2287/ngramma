#!/usr/bin/env python3
"""Verify archive integrity and replay published mock actions; no model inference."""
import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
from mock_tools import MockDevices


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(root, path):
    return json.loads((root / path).read_text())


def replay(task, record):
    require(record['task_id'] == task['task_id'], 'Task ID mismatch')
    digest = hashlib.sha256(json.dumps(task, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    require(record['task_sha256'] == digest, 'Task content differs from recorded task hash')
    require(not record['teacher_used'] and not record['training_eligible'], 'Unexpected training/teacher record')
    env = MockDevices(task)
    pending = deque()
    for message in record['messages']:
        if message['role'] == 'assistant':
            require(not pending, 'Assistant turn before pending tool results')
            for call in message.get('tool_calls') or []:
                try:
                    arguments = json.loads(call['function']['arguments'])
                    if not isinstance(arguments, dict):
                        raise ValueError('tool arguments must be an object')
                    result = env.call(call['function']['name'], arguments)
                except (KeyError, ValueError, TypeError) as error:
                    result = env.error(str(error))
                pending.append((call['id'], result))
        elif message['role'] == 'tool':
            require(bool(pending), 'Tool result without a call')
            call_id, expected = pending.popleft()
            require(message['tool_call_id'] == call_id, 'Tool-call ID mismatch')
            require(json.loads(message['content']) == expected, 'Saved tool result differs from replay')
    require(not pending, 'Missing tool results')
    grade = env.grade(task)
    require(grade == record['grade'], 'Saved grade differs from replay')
    require(grade['success'] is record['success'], 'Saved success differs from state grade')
    return grade['success']


def layer_errors(record):
    return {int(item['name'].split('-')[1]): item['relative_rms']
            for item in record['intermediates'] if re.fullmatch(r'l_last-\d+', item['name'])}


def verify(root):
    manifest = read(root, 'data/SHA256SUMS.json')
    for name, expected in manifest['files'].items():
        path = root / name
        require(path.is_file() and not path.is_symlink(), f'Missing/invalid file: {name}')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Hash mismatch: {name}')
    for item in read(root, 'data/publication-provenance.json')['files']:
        require(manifest['files'][item['published_path']] == item['published_sha256'], 'Source-map hash mismatch')
    print(f'Integrity: {len(manifest["files"])} files match the published manifest.')

    tasks = {t['task_id']: t for t in read(root, 'data/pilot/tasks.json')}
    require(len(tasks) == 20 and len({t['family_id'] for t in tasks.values()}) == 5, 'Unexpected task population')
    require(all(t['partition'] != 'sealed_test' for t in tasks.values()), 'Excluded family included')
    total = 0
    for collection in ('pilot', 'confirmation'):
        counts = Counter()
        successes = Counter()
        pairs = Counter()
        for path in sorted((root / 'data' / collection / 'episodes').glob('*.json')):
            record = json.loads(path.read_text())
            task = tasks[record['task_id']]
            try:
                success = replay(task, record)
            except (KeyError, ValueError, TypeError) as error:
                raise ValueError(f'{path.name}: {error}') from error
            # Initial pilot records predate this explicit flag; the original
            # runner's default was cache-enabled, noncanonical replay.
            require(record.get('deterministic_replay', False) is (collection == 'confirmation'), 'Cache/replay protocol differs')
            condition = record['condition']
            counts[condition] += 1
            successes[condition] += success
            pairs[(task['task_id'], condition)] += 1
            total += 1
        summary = read(root, f'data/{collection}/summary.json')
        conditions = ('unassisted', 'hinted', 'corrected_retry') if collection == 'pilot' else ('unassisted', 'hinted')
        require(set(counts) == set(conditions), 'Unexpected conditions')
        for condition in conditions:
            denominator = 'tasks' if collection == 'pilot' else 'attempts'
            require(counts[condition] == summary[condition][denominator], 'Reported episode count differs')
            require(successes[condition] == summary[condition]['successes'], 'Reported success count differs')
            if 'rate' in summary[condition]:
                require(abs(summary[condition]['rate'] - successes[condition]/counts[condition]) < 1e-12, 'Rate differs')
            print(f'{collection}/{condition}: {successes[condition]}/{counts[condition]} replayed successes')
        if collection == 'pilot':
            require(all(pairs[(task_id, condition)] == 1 for task_id in tasks for condition in ('unassisted', 'hinted')), 'Incomplete/duplicated task pairs')
            require(summary['independent_families'] == 5 and summary['sealed_tasks_used'] == 0 and not summary['training_data_generated'], 'Population metadata differs')
        else:
            require(set(key[0] for key in pairs) == {'multi_target/2'}, 'Confirmation used a different task')
    require(total == 47, 'Unexpected total episode count')

    cases = read(root, 'data/compatibility-cases.json')
    compatibility = read(root, 'data/engine-compatibility.json')
    require([[c['name'], len(c['tokens']), c['chunk_size']] for c in cases] == compatibility['cases'], 'Compatibility case metadata differs')
    require(sum(len(c['tokens']) for c in cases) == 39, 'Unexpected compatibility coverage')
    for name in ('disk', 'empty', 'original-rows'):
        condition = compatibility['conditions'][name]
        require(condition['address_count'] == 39 * 16 and condition['passed'], 'Address/compatibility gate differs')
        require(len(condition['logit_comparisons']) == 3 and all(x['exact'] and x['max_abs'] == 0 for x in condition['logit_comparisons']), 'Recorded unchanged-output check failed')
    negative = read(root, 'data/overlay-negative-checks.json')
    require(len(negative['cases']) == 6 and all(c['rejected'] for c in negative['cases']), 'Recorded invalid-overlay rejection failed')
    perturb = read(root, 'data/perturbation-check.json')
    require(perturb['diagnostic_only'] and not perturb['improvement_claimed'], 'Perturbation mislabeled')
    require(perturb['gather_exactly_matches_overlay'] and perturb['unselected_gather_values_unchanged'] and perturb['logits_changed'], 'Recorded perturbation check failed')

    for name in ('replica-parity.json', 'replica-quantized-parity.json'):
        record = read(root, 'data/' + name)
        errors = layer_errors(record)
        require(set(errors) == set(range(48)), 'Missing full-layer measurements')
        require(record['tokens'] == cases[0]['tokens'], 'Replica input differs from short reference')
        require(record['logit_parity'] == (record['selected_logprob_error_max'] < 0.02), 'Logit gate inconsistent')
        require(record['intermediate_parity'] == (max(x['relative_rms'] for x in record['intermediates']) < 0.01), 'Intermediate gate inconsistent')
        require(not record['logit_parity'] and not record['intermediate_parity'], 'This snapshot must report failed forward agreement')
        print(f'{name}: top-1 {record["top1_agreement"]:.0%}; max selected log-prob error {record["selected_logprob_error_max"]:.6f} nats; last-layer RMS {100*errors[47]:.4f}%')
    gate = read(root, 'data/training-admission.json')
    require(all(gate[k] is False for k in ('training_enabled', 'logit_parity', 'intermediate_parity', 'directional_gradient', 'live_routing')), 'Unexpected training admission')
    print('PASS: archive verified, 47 episodes replayed; saved numerical checks are internally consistent. Training remains unqualified.')
    print('This checks published evidence; it does not rerun the model or recompute logits.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Archive root to verify')
    args = parser.parse_args()
    try:
        verify(args.root)
    except (ValueError, KeyError, OSError, TypeError) as error:
        raise SystemExit(f'FAIL: {error}')
