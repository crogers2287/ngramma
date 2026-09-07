"""Reject misleading instruction-edit evidence without executing a model."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1] / 'experiments/006-instruction-memory'
spec = importlib.util.spec_from_file_location('instruction_evidence', EXP / 'verify_results.py')
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


@pytest.fixture
def records(tmp_path):
    # Snapshot only finished records named by one atomic search report. This
    # also works while the real experiment is still producing later records.
    report = evidence.read(EXP / 'search-results.json')
    names = ['candidates', 'tokenization', 'parallel-verify-build', 'verification-tests',
             'zero', 'zero-validation', 'zero-lookup', 'input-audit', 'first-candidate-lookup']
    names += [r['label'] for r in report['results']]
    for name in names:
        (tmp_path / (name + '.json')).write_bytes((EXP / (name + '.json')).read_bytes())
    (tmp_path / 'search-results.json').write_text(json.dumps(report))
    return tmp_path


def mutate(directory, name, fn):
    path = directory / (name + '.json')
    record = evidence.read(path)
    fn(record)
    path.write_text(json.dumps(record))


def test_published_records_replay(records):
    result = evidence.verify(records, require_complete=False)
    report = evidence.read(records / 'search-results.json')
    assert result['candidates_replayed'] == len(report['results'])
    assert result['scored_generations'] == 24 * len(report['results'])
    assert result['zero_matching_generations'] == 24


@pytest.mark.parametrize('change', [
    'invented_winner', 'invented_rescue', 'deleted_candidate', 'heldout_claim',
    'duplicate_row', 'changed_magnitude', 'stale_zero', 'changed_zero',
    'no_lookup_edit', 'wrong_lookup_overlay', 'cached_authentication',
    'partial_authentication', 'wrong_verifier', 'skipped_authentication_test',
    'added_hint', 'wrong_overlay', 'missing_task', 'changed_control_answer',
])
def test_changed_or_misleading_evidence_rejected(records, change):
    if change == 'invented_winner':
        mutate(records, 'search-results', lambda r: r.update(winner='rademacher-minus-8'))
    elif change == 'invented_rescue':
        mutate(records, 'search-results', lambda r: r['results'][0].update(development_correct=16, eligible=True))
    elif change == 'deleted_candidate':
        mutate(records, 'search-results', lambda r: r['results'].pop(0))
    elif change == 'heldout_claim':
        mutate(records, 'search-results', lambda r: r.update(heldout_evaluated=True))
    elif change == 'duplicate_row':
        mutate(records, 'candidates', lambda r: r['row_ids'].__setitem__(1, r['row_ids'][0]))
    elif change == 'changed_magnitude':
        mutate(records, 'candidates', lambda r: r['candidates'][1].update(epsilon=-1.0))
    elif change == 'stale_zero':
        mutate(records, 'zero-validation', lambda r: r.update(baseline_sha256='0' * 64))
    elif change == 'changed_zero':
        mutate(records, 'zero', lambda r: r['jobs'][2]['response'].update(text='A'))
    elif change == 'no_lookup_edit':
        mutate(records, 'first-candidate-lookup', lambda r: r.update(changed_values_from_original=0))
    elif change == 'wrong_lookup_overlay':
        mutate(records, 'first-candidate-lookup', lambda r: r.update(overlay_sha256='0' * 64))
    elif change == 'cached_authentication':
        mutate(records, 'rademacher-minus-8', lambda r: r['parallel_verification'][0].update(cache_reused=True))
    elif change == 'partial_authentication':
        mutate(records, 'rademacher-minus-8', lambda r: r['parallel_verification'][0].update(bytes=10946624, shards=1))
    elif change == 'wrong_verifier':
        mutate(records, 'rademacher-minus-8', lambda r: r['profile'].update(verification_library_sha256='0' * 64))
    elif change == 'skipped_authentication_test':
        mutate(records, 'verification-tests', lambda r: r.update(skipped=1))
    elif change == 'added_hint':
        mutate(records, 'rademacher-minus-8', lambda r: r['jobs'][2]['request'].update(text='Answer A.'))
    elif change == 'wrong_overlay':
        mutate(records, 'rademacher-minus-8', lambda r: r.update(overlay_sha256='0' * 64))
    elif change == 'missing_task':
        mutate(records, 'rademacher-minus-8', lambda r: r['jobs'].pop(2))
    elif change == 'changed_control_answer':
        mutate(records, 'rademacher-minus-8', lambda r: next(j for j in r['jobs'] if j['id'] == 'controls-1')['response'].update(text='A'))
    with pytest.raises(ValueError):
        evidence.verify(records, require_complete=False)


def test_prefix_cannot_claim_finished_search(records):
    mutate(records, 'search-results', lambda r: r.update(results=r['results'][:1], complete=False, winner=None))
    assert evidence.verify(records, require_complete=False)['candidates_replayed'] == 1
    with pytest.raises(ValueError, match='incomplete'):
        evidence.verify(records)


def test_fixed_ranking_is_independent_of_input_order():
    records = [{'label': f'{d}-{sign}-{power}', 'epsilon': multiplier * 2**-power,
                'development_correct': score, 'eligible': True}
               for d, sign, multiplier, power, score in [
                   ('hint_contrast', 'minus', -1, 8, 4),
                   ('rademacher', 'plus', 1, 8, 4),
                   ('rademacher', 'minus', -1, 8, 4),
                   ('alternating', 'minus', -1, 6, 4),
               ]]
    assert evidence.winner_for(records) == evidence.winner_for(list(reversed(records))) == 'rademacher-minus-8'
    better = copy.deepcopy(records[-1])
    better['development_correct'] = 5
    assert evidence.winner_for(records + [better]) == 'alternating-minus-6'
    better['eligible'] = False
    assert evidence.winner_for(records + [better]) == 'rademacher-minus-8'
