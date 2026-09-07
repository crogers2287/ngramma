"""Saved-encoding evidence relationships; no model or numerical dependencies."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence():
    directory = ROOT/'experiments/004-row-response'
    spec = importlib.util.spec_from_file_location('row_response_verifier',ROOT/'scripts/verify_row_response.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_scale_transition, json.loads((directory/'review/scale-transition-final.json').read_text()), json.loads((directory/'response.json').read_text()), json.loads((directory/'activation-build.json').read_text())


def test_published_scale_replay_is_bound_to_local_response(evidence):
    verify,record,response,build = evidence
    verify(record,response,build)


@pytest.mark.parametrize('change',['missing_pair','different_overlay','wrong_scale_bits','wrong_maximum_branch','changed_code','different_cpu'])
def test_contradictory_scale_explanations_rejected(evidence,change):
    verify,record,response,build = evidence
    record = copy.deepcopy(record)
    transition = record['transitions']['plus-10']
    if change == 'missing_pair': record['transitions'].pop('minus-10')
    elif change == 'different_overlay': record['provenance']['overlay_sha256']['plus-10'] = '0'*64
    elif change == 'wrong_scale_bits': transition['changed_blocks'][0]['scale_after_bits_hex'] = '0778'
    elif change == 'wrong_maximum_branch': transition['changed_blocks'][1]['absolute_maximum_indices_after'] = [31]
    elif change == 'changed_code': transition['changed_code_bytes'] = 1
    else: record['provenance']['cpu_library_sha256'] = '0'*64
    with pytest.raises(ValueError): verify(record,response,build)
