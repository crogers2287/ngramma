"""Saved synthetic-task evidence is replayed without model files or inference."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

EXP = Path(__file__).resolve().parents[1] / 'experiments/005-behavior'
sys.path.insert(0, str(EXP))
spec = importlib.util.spec_from_file_location('behavior_evidence', EXP / 'score_results.py')
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def test_published_initial_outcomes_distinguish_content_and_format():
    record = json.loads((EXP / 'discovery.json').read_text())
    score = evidence.score_batch(record, 'discovery')
    dev = score['groups']['initial/dev']
    assert (dev['correct'], dev['wrong_choice'], dev['malformed'], dev['total']) == (1, 2, 13, 16)
    assert dev['well_formed_eog'] == 3
    assert score['groups']['initial/controls']['correct'] == 6


def test_published_format_and_counting_contrasts():
    record = json.loads((EXP / 'followup.json').read_text())
    score = evidence.score_batch(record, 'followup')
    groups = score['groups']
    assert groups['format/dev']['correct'] == 14
    assert groups['format/controls']['correct'] == 7
    assert groups['count-ordinary/dev']['correct'] == groups['count-hinted/dev']['correct'] == 4
    count = [x for x in score['tasks'] if x['condition'].startswith('count-')]
    assert len(count) == 20 and all(x['text'] == 'A' for x in count)


@pytest.mark.parametrize('mutation', ['missing_job', 'duplicate_job', 'prompt', 'id', 'job_hash', 'failed', 'unstable', 'wrong_tokenization'])
def test_changed_evidence_rejected(mutation):
    record = json.loads((EXP / 'discovery.json').read_text())
    if mutation == 'missing_job': record['jobs'].pop()
    elif mutation == 'duplicate_job': record['jobs'].append(copy.deepcopy(record['jobs'][-1]))
    elif mutation == 'prompt': record['jobs'][2]['request']['text'] += ' Answer A.'
    elif mutation == 'id': record['jobs'][2]['id'] = 'different-task'
    elif mutation == 'job_hash': record['jobs_sha256'] = '0'*64
    elif mutation == 'failed': record['status'] = 'failed'
    elif mutation == 'unstable': record['inputs_unchanged'] = False
    elif mutation == 'wrong_tokenization': record['jobs'][0]['response']['tokens'] = [33]
    with pytest.raises(ValueError): evidence.score_batch(record, 'discovery')


def test_unfinished_valid_prefix_is_not_a_success():
    task = evidence.tasks.generate_tasks()[0]
    grade = evidence.grade_completion(task, {'text': 'A', 'stop_reason': 'max_new_tokens'})
    assert grade['correct'] and not grade['success']
    assert grade['error_kind'] == 'unfinished_choice'
