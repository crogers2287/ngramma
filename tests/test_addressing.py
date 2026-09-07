"""Portable address arithmetic against historical native traces, not itself."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ngramma_runtime.addressing import Addressing

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/006-instruction-memory'


def metadata():
    return json.loads((ROOT / 'data/model-identity.json').read_text())['architecture_metadata']


def test_all_624_published_native_addresses_match():
    address = Addressing.from_metadata(metadata())
    cases = json.loads((ROOT / 'data/compatibility-cases.json').read_text())
    records = [json.loads(line) for line in (EXP / 'reference-addresses.jsonl').read_text().splitlines()]
    offset = 0
    for case in cases:
        expected = records[offset:offset + len(case['tokens'])]
        assert [r['position'] for r in expected] == list(range(len(case['tokens'])))
        assert [r['token'] for r in expected] == case['tokens']
        assert address.addresses(case['tokens']) == [r['rows'] for r in expected]
        offset += len(expected)
    assert offset == len(records) == 39


def test_prepared_instruction_coverage_matches_for_all_24_prompts():
    address = Addressing.from_metadata(metadata())
    recipe = json.loads((EXP / 'candidates.json').read_text())
    baseline = json.loads((ROOT / 'experiments/005-behavior/discovery.json').read_text())
    actual = {j['id']: address.occurrences(j['response']['prompt_tokens'], recipe['row_ids'])
              for j in baseline['jobs'] if 'generate' in j['request']}
    assert actual == recipe['coverage']


def test_prepared_hint_donor_rows_match():
    address = Addressing.from_metadata(metadata())
    recipe = json.loads((EXP / 'candidates.json').read_text())
    records = json.loads((ROOT / 'experiments/005-behavior/followup.json').read_text())
    jobs = {j['id']: j['response'] for j in records['jobs']}
    donors = [set() for _ in range(8)]
    for span in recipe['hint_spans']:
        tokens = jobs[span['id']]['prompt_tokens']
        assert tokens[span['start']:span['end']] == recipe['hint_tokens']
        rows = address.addresses(tokens)
        for hi in range(8):
            donors[hi].update(rows[pos][hi + 8] for pos in range(span['start'] + 2, span['end'])
                              if rows[pos][hi + 8] not in recipe['row_ids'])
    assert [sorted(x) for x in donors] == recipe['hint_donor_rows_per_head']


def test_eos_cuts_predecessors_but_not_its_own_context():
    address = Addressing.from_metadata(metadata())
    eos = address.eos_token_id
    assert address.addresses([11, eos, 32])[-1] == address.addresses([32])[0]
    assert address.addresses([11, eos])[1] != address.addresses([eos])[0]
    # Actual chat end-of-message is a different ID, not a PLE EOS reset.
    assert address.addresses([11, 248046, 32])[-1] != address.addresses([32])[0]


def test_unsigned_64_bit_products_wrap():
    # 2*(2^63+3) wraps to 6; predecessor 1 contributes 5. Thus bigram
    # mixed = 6 XOR 5 = 3, and trigram adds EOS(0)*7 = 0.
    address = Addressing(0, (2**63 + 3, 5, 7), (17,) * 16, tuple(17*h for h in range(16)))
    assert address.addresses([1, 2])[-1] == [17*h + 3 for h in range(16)]


@pytest.mark.parametrize('token', [-1, True, 1.0, '1', None, 2**31])
def test_bad_tokens_rejected(token):
    with pytest.raises(ValueError):
        Addressing.from_metadata(metadata()).addresses([token])


@pytest.mark.parametrize('key,value', [
    ('ngram_size', 4), ('ngram_size', 3.0), ('heads_per_ngram', 4),
    ('eos_token_id', -1), ('layer_multipliers', [1, 2]),
    ('head_vocab_sizes', [0] * 16), ('head_offsets', [0] * 16),
    ('layer_multipliers', [1, 2, 2**64]),
])
def test_unqualified_or_invalid_metadata_rejected(key, value):
    values = metadata()
    values['qwen4exp.ple.' + key] = value
    with pytest.raises(ValueError):
        Addressing.from_metadata(values)


def test_configuration_does_not_alias_mutable_inputs():
    multipliers = [1, 2, 3]
    address = Addressing(0, multipliers, [17] * 16, [17*h for h in range(16)])
    before = address.addresses([1, 2, 3])
    multipliers[0] = 900
    assert address.addresses([1, 2, 3]) == before


def test_cli_runs_without_site_packages_or_model_paths(tmp_path):
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    result = subprocess.run([sys.executable, '-S', '-m', 'ngramma_runtime.addressing',
                             '--identity', str(ROOT / 'data/model-identity.json'),
                             '--batch', str(ROOT / 'experiments/005-behavior/discovery.json'),
                             '--job', 'dev-decimal-0', '--row', '170852069'],
                            env=env, cwd=tmp_path, text=True, capture_output=True, check=True)
    record = json.loads(result.stdout)
    assert record['occurrences'] == [{'row_id': 170852069, 'positions_and_heads': [[38, 8]]}]
    assert record['model_files_read'] is record['identity_authenticated'] is False


def test_selection_outside_head_ranges_rejected():
    address = Addressing.from_metadata(metadata())
    with pytest.raises(ValueError):
        address.occurrences([32], [2**31 - 1])
