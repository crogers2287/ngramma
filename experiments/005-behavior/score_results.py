#!/usr/bin/env python3
"""Replay exact synthetic prompts and whole-completion grading; no inference."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import counting_tasks
from prepare_jobs import jobs, followup_jobs, state_confirmation_jobs
from run_batch import validate_response
import tasks


def expected_jobs(stage):
    if stage == 'discovery':
        return jobs(controls=True)
    if stage == 'followup':
        return followup_jobs()
    if stage == 'state-confirmation':
        return state_confirmation_jobs()
    raise ValueError('Unknown experiment stage')


def task_for(job_id):
    base = {t.id: t for t in tasks.generate_tasks() + tasks.generate_tasks('controls')}
    counting = {t.id: t for t in counting_tasks.generate_tasks() + counting_tasks.generate_tasks('controls')}
    if '/' not in job_id:
        return 'initial', base[job_id]
    condition, task_id = job_id.split('/', 1)
    return condition, (counting if condition.startswith('count-') else base)[task_id]


def grade_completion(task, response):
    value = tasks.grade(task, response['text'])
    # The cap must not make an unfinished prefix look like a complete answer.
    value['terminated'] = response['stop_reason'] == 'eog'
    value['success'] = value['correct'] and value['terminated']
    if value['correct'] and not value['terminated']:
        value['error_kind'] = 'unfinished_choice'
    return value


def score_batch(record, stage):
    expected = expected_jobs(stage)
    encoded = (json.dumps(expected, indent=2) + '\n').encode()
    if record.get('status') != 'complete' or record.get('inputs_unchanged') is not True:
        raise ValueError('Only complete source-stable batches can be scored')
    if record.get('jobs_sha256') != hashlib.sha256(encoded).hexdigest():
        raise ValueError('Batch does not bind the fixed jobs')
    if len(record['jobs']) != len(expected):
        raise ValueError('Missing or extra jobs')
    groups = defaultdict(Counter)
    graded = []
    for actual, planned in zip(record['jobs'], expected):
        if actual['id'] != planned['id'] or actual['request'] != {k: v for k, v in planned.items() if k != 'id'}:
            raise ValueError('Job or prompt changed')
        response = actual['response']
        validate_response(actual['request'], response)
        if actual['id'].startswith('tokenize/'):
            continue
        if actual['id'].startswith('token-'):
            if response['tokens'] != ([32] if actual['id'] == 'token-A' else [33]):
                raise ValueError('Requested score tokens do not encode A/B')
            continue
        condition, task = task_for(actual['id'])
        value = grade_completion(task, response)
        key = condition + '/' + task.split
        groups[key]['total'] += 1
        groups[key]['correct' if value['success'] else value['error_kind']] += 1
        groups[key]['well_formed_eog'] += int(value['well_formed'] and value['terminated'])
        groups[key]['cap_exhausted'] += int(response['stop_reason'] != 'eog')
        graded.append({'id': actual['id'], 'condition': condition, 'task_id': task.id,
                       'family': task.family, 'split': task.split, 'answer': task.answer,
                       'text': response['text'], 'stop_reason': response['stop_reason'],
                       'prompt_tokens': len(response['prompt_tokens']), **value})
    return {'groups': {k: dict(v) for k, v in groups.items()}, 'tasks': graded}


def main():
    root = Path(__file__).resolve().parent
    outputs = {}
    records = {}
    for stage in ('discovery', 'followup'):
        path = root / (stage + '.json')
        records[stage] = json.loads(path.read_text())
        outputs[stage] = score_batch(records[stage], stage)
    initial = {j['id']: j['response'] for j in records['discovery']['jobs']}
    repeated = [j for j in records['followup']['jobs'] if j['id'].startswith('repeat/')]
    keys = ('prompt_tokens', 'generated_tokens', 'text', 'text_bytes_hex', 'first_step',
            'stop_reason', 'stopped_on_eog', 'eog_token')
    matches = sum(all(j['response'][k] == initial[j['id'].split('/', 1)[1]][k] for k in keys) for j in repeated)
    format_groups = outputs['followup']['groups']
    formatted = sum(format_groups['format/' + split]['well_formed_eog'] for split in ('dev', 'controls'))
    summary = {stage: out['groups'] for stage, out in outputs.items()}
    summary.update({'repeat_generation_and_selected_logits_identical': matches,
                    'repeated_tasks': len(repeated),
                    'format_gate': {'well_formed_eog': formatted, 'total': 24,
                                    'required': 22, 'passed': formatted >= 22},
                    'note': 'Saved evidence replay only; no model inference rerun.'})
    path = root / 'state-confirmation.json'
    if path.exists():
        confirmation = json.loads(path.read_text())
        scored = score_batch(confirmation, 'state-confirmation')
        responses = {j['id']: j['response'] for j in confirmation['jobs']}
        previous = {j['id']: j['response'] for j in records['followup']['jobs']}
        baseline_ids = [j['id'] for j in confirmation['jobs'] if j['id'].startswith('baseline/')]
        baseline_matches = sum(all(responses[i][k] == previous['format/' + i.split('/', 1)[1]][k] for k in keys) for i in baseline_ids)
        ordinary_matches = sum(all(responses[f'baseline/dev-state-{i}'][k] == responses[f'repeat-ordinary/dev-state-{i}'][k] for k in keys) for i in range(4))
        hinted_matches = sum(all(responses[f'hinted/dev-state-{i}'][k] == responses[f'repeat-hinted/dev-state-{i}'][k] for k in keys) for i in range(4))
        by_id = {j['id']: j for j in scored['tasks']}
        rescued = [i for i in range(4) if
                   by_id[f'baseline/dev-state-{i}']['error_kind'] == 'wrong_choice' and
                   by_id[f'repeat-ordinary/dev-state-{i}']['error_kind'] == 'wrong_choice' and
                   all(by_id[f'{condition}/dev-state-{i}']['success'] for condition in ('hinted', 'repeat-hinted'))]
        retained = all(all(by_id[f'{condition}/dev-state-{i}']['success'] for condition in ('hinted', 'repeat-hinted'))
                       for i in range(4) if by_id[f'baseline/dev-state-{i}']['success'])
        summary.update({'state_confirmation': scored['groups'],
                          'baseline_identical': baseline_matches, 'baseline_total': 24,
                          'within_process_ordinary_identical': ordinary_matches,
                          'within_process_hinted_identical': hinted_matches,
                          'repeated_hint_rescues': rescued,
                          'baseline_correct_retained_by_hint': retained,
                          'content_target_gate_passed': baseline_matches == 24 and ordinary_matches == hinted_matches == 4 and len(rescued) >= 2 and retained})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
