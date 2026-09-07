"""Counterfactual synthetic completions test the predeclared admission rule."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT/'experiments/006-instruction-memory'
spec = importlib.util.spec_from_file_location('instruction_search', EXP/'search.py')
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)


def fixture():
    original = json.loads((ROOT/'experiments/005-behavior/discovery.json').read_text())
    record = copy.deepcopy(original)
    record['overlay_sha256'] = 'synthetic-selection-fixture'
    candidate = {'label': 'rademacher-minus-8', 'epsilon': -2**-8,
                 'overlay_sha256': record['overlay_sha256']}
    return record, original, candidate, search.jobs(controls=True)


def answer(record, name, letter):
    # Deliberately synthetic, logically consistent response; never published as
    # measured output or passed to an actual model.
    response = next(j['response'] for j in record['jobs'] if j['id'] == name)
    token, other = (32, 33) if letter == 'A' else (33, 32)
    response.update(text=letter, text_bytes_hex=letter.encode().hex(),
                    generated_tokens=[token, 248046], generated_count=2,
                    stop_reason='eog', stopped_on_eog=True, eog_token=248046)
    response['first_step'] = {'greedy_token_id': token,
                             'top_logits': [{'token_id': t, 'logit': v} for t, v in
                                            [(token, 10.0), (other, 9.0), (0, 8.0), (1, 7.0), (2, 6.0)]],
                             'requested_logits': [{'token_id': t, 'logit': 10.0 if t == token else 9.0} for t in (32, 33)]}


def test_unchanged_candidate_has_no_admission():
    record, baseline, candidate, jobs = fixture()
    result = search.assess(record, baseline, candidate, jobs)
    assert result['development_correct'] == 1 and result['controls_correct'] == 6
    assert not result['eligible'] and result['development_rescues'] == []


def test_two_real_answer_rescues_and_retention_are_required():
    record, baseline, candidate, jobs = fixture()
    answer(record, 'dev-arithmetic-0', 'A')
    assert not search.assess(record, baseline, candidate, jobs)['eligible']
    answer(record, 'dev-arithmetic-1', 'A')
    result = search.assess(record, baseline, candidate, jobs)
    assert result['eligible'] and result['development_correct'] == 3
    assert result['development_rescues'] == ['dev-arithmetic-0', 'dev-arithmetic-1']


def test_a_control_loss_rejects_an_otherwise_improved_candidate():
    record, baseline, candidate, jobs = fixture()
    answer(record, 'dev-arithmetic-0', 'A')
    answer(record, 'dev-arithmetic-1', 'A')
    answer(record, 'controls-3', 'A')  # Correct original answer is B.
    result = search.assess(record, baseline, candidate, jobs)
    assert not result['eligible'] and result['baseline_correct_losses'] == ['controls-3']


def test_always_a_cannot_pass_by_hiding_behind_format_gain():
    record, baseline, candidate, jobs = fixture()
    for task in search.generate_tasks()+search.generate_tasks('controls'):
        answer(record, task.id, 'A')
    result = search.assess(record, baseline, candidate, jobs)
    assert result['development_correct'] == 8
    assert len(result['baseline_correct_losses']) == 3
    assert not result['eligible']


@pytest.mark.parametrize('change', ['prompt_tokens', 'prompt_text', 'overlay', 'runtime', 'threads', 'failed', 'missing'])
def test_incomparable_or_incomplete_candidates_rejected(change):
    record, baseline, candidate, jobs = fixture()
    if change == 'prompt_tokens': record['jobs'][2]['response']['prompt_tokens'][0] = 123
    elif change == 'prompt_text': record['jobs'][2]['request']['text'] += ' A'
    elif change == 'overlay': record['overlay_sha256'] = 'another'
    elif change == 'runtime': record['lens_sha256'] = 'another'
    elif change == 'threads': record['profile']['threads'] = 1
    elif change == 'failed': record['status'] = 'failed'
    elif change == 'missing': record['jobs'].pop()
    with pytest.raises(ValueError): search.assess(record, baseline, candidate, jobs)
