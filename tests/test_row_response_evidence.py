"""Tampered-record controls for the model-free experiment004 verifier.

The optional engine fixture below is fabricated test data, not a measurement.
It exercises relationships between records without launching a native engine.
"""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'experiments/004-row-response'


@pytest.fixture(scope='module')
def verifier():
    spec=importlib.util.spec_from_file_location('ngramma_test_verify_row_response',ROOT/'scripts/verify_row_response.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.verify


@pytest.fixture
def evidence():
    return (json.loads((DIRECTORY/'response.json').read_text()),
            json.loads((DIRECTORY/'activation-build.json').read_text()))


def zero(report):return next(point for point in report['points'] if point['epsilon']==0)
def active(report):return next(point for point in report['points'] if point['key_changed_count']>0)


def test_actual_local_record_passes_without_model(verifier,evidence):
    report,build=evidence
    result=verifier(report,build)
    assert result['points']==19 and result['engine_checks']==0


@pytest.mark.parametrize('field',['activation_changed_bytes','activation_scale_changed_bytes','activation_code_changed_bytes'])
def test_corrupted_scale_code_accounting_rejected(verifier,evidence,field):
    report,build=evidence;active(report)[field]+=1
    with pytest.raises(ValueError):verifier(report,build)


def test_encoded_hash_cannot_match_baseline_when_bytes_change(verifier,evidence):
    report,build=evidence;active(report)['activation_sha256']=zero(report)['activation_sha256']
    with pytest.raises(ValueError,match='Encoded-byte hash/count'):verifier(report,build)


def test_unchanged_encoded_bytes_cannot_have_a_different_hash(verifier,evidence):
    report,build=evidence
    point=next(point for point in report['points'] if point['epsilon'] and point['activation_changed_bytes']==0)
    point['activation_sha256']='f'*64
    with pytest.raises(ValueError,match='Encoded-byte hash/count'):verifier(report,build)


@pytest.mark.parametrize('stage',['key','value'])
def test_projection_hash_count_contradiction_rejected(verifier,evidence,stage):
    report,build=evidence;active(report)['stage_sha256'][stage]=zero(report)['stage_sha256'][stage]
    with pytest.raises(ValueError,match='Projection hash/count'):verifier(report,build)


def test_nonzero_step_cannot_export_a_numerical_noop_row(verifier,evidence):
    report,build=evidence;active(report)['replacement_changed_values']=0
    with pytest.raises(ValueError,match='replacement coordinate count'):verifier(report,build)


def test_altered_engine_step_selection_rejected(verifier,evidence):
    report,build=evidence;report['selected_engine_labels']=list(reversed(report['selected_engine_labels']))
    with pytest.raises(ValueError,match='selection differs'):verifier(report,build)


@pytest.mark.parametrize('target',['library','cpu_libraries'])
def test_different_native_libraries_rejected(verifier,evidence,target):
    report,build=evidence
    if target=='library':build['library_sha256']='f'*64
    else:build['linked_libraries']['libggml-cpu.so']='f'*64
    with pytest.raises(ValueError,match='identity mismatch|different CPU libraries'):verifier(report,build)


@pytest.mark.parametrize('field',['ple_output_rms','local_scalar_delta','key_changed_count','activation_scale_changed_bytes'])
def test_zero_step_cannot_change_local_outputs(verifier,evidence,field):
    report,build=evidence;zero(report)[field]=1
    with pytest.raises(ValueError):verifier(report,build)


def test_realized_step_and_smooth_derivative_evidence_rejected_if_tampered(verifier,evidence):
    report,build=evidence;active(report)['realized_normalized_rms']*=2
    with pytest.raises(ValueError,match='Realized displacement'):verifier(report,build)
    report,_=evidence
    report=copy.deepcopy(json.loads((DIRECTORY/'response.json').read_text()))
    report['smooth_control']['finite_differences'][0]['relative_error']=1000
    with pytest.raises(ValueError,match='Smooth derivative error'):verifier(report,build)


@pytest.fixture
def synthetic_engine_evidence(evidence):
    report,build=evidence
    report['test_fixture_notice']='SYNTHETIC engine metadata for verifier tests; no inference performed'
    checks=[]
    by_label={point['overlay_label']:point for point in report['points']}
    for label in ['unmodified',*report['selected_engine_labels']]:
        point=by_label.get(label)
        capture=copy.deepcopy(report['capture_identity'])
        changed=point is not None and bool(point['key_changed_count'] or point['value_changed_count'])
        capture['overlay_sha256']=None if point is None else point['overlay_sha256']
        if changed:capture['logits_sha256']='f'*64
        margin_delta=.125 if changed else 0.
        if point is not None:point['engine_logit_margin_delta']=margin_delta
        checks.append({'label':label,'capture':capture,'fresh_process':True,
            'live_routing_and_attention':True,'ple_matches_local_native':True,
            'gathered_replacement_verified':True,'logits_bitwise_equal_to_original':not changed,
            'changed_logit_values':5 if changed else 0,
            'top1_agree_positions':len(report['experiment']['tokens'])-(1 if changed else 0),
            'max_abs_logit_change':.25 if changed else 0.,
            'logit_margin':report['experiment']['engine_margin']['baseline']+margin_delta,
            'logit_margin_delta':margin_delta,
            'ple_output_sha256':point['stage_sha256']['output'] if point else zero(report)['stage_sha256']['output']})
    report['engine_checks']=checks
    report['engine_runtime_sha256']=copy.deepcopy(build['linked_libraries'])
    return report,build


def test_synthetic_engine_record_relationships_pass(verifier,synthetic_engine_evidence):
    report,build=synthetic_engine_evidence
    result=verifier(report,build)
    assert result['engine_checks']==len(report['selected_engine_labels'])+1


@pytest.mark.parametrize('field,value',[('fresh_process',False),('live_routing_and_attention',False),
    ('ple_matches_local_native',False),('gathered_replacement_verified',False),
    ('logits_bitwise_equal_to_original',False),('changed_logit_values',1),('logit_margin_delta',1.)])
def test_synthetic_engine_invalid_baseline_relationships_rejected(verifier,synthetic_engine_evidence,field,value):
    report,build=synthetic_engine_evidence;report['engine_checks'][0][field]=value
    with pytest.raises(ValueError):verifier(report,build)


@pytest.mark.parametrize('field,value',[('identity_sha256','f'*64),('tokens',[1]),('device','CUDA'),
                                      ('cache_type','f16'),('repack',False)])
def test_synthetic_engine_profile_mismatch_rejected(verifier,synthetic_engine_evidence,field,value):
    report,build=synthetic_engine_evidence;report['engine_checks'][1]['capture'][field]=value
    with pytest.raises(ValueError,match='fixture identity/config mismatch'):verifier(report,build)


def test_synthetic_engine_wrong_overlay_rejected(verifier,synthetic_engine_evidence):
    report,build=synthetic_engine_evidence;report['engine_checks'][1]['capture']['overlay_sha256']='f'*64
    with pytest.raises(ValueError,match='different overlay'):verifier(report,build)


def test_synthetic_engine_wrong_ple_hash_rejected(verifier,synthetic_engine_evidence):
    report,build=synthetic_engine_evidence;report['engine_checks'][1]['ple_output_sha256']='f'*64
    with pytest.raises(ValueError,match='PLE hashes differ'):verifier(report,build)


def test_synthetic_engine_cpu_library_mismatch_rejected(verifier,synthetic_engine_evidence):
    report,build=synthetic_engine_evidence;report['engine_runtime_sha256']['libggml-cpu.so']='f'*64
    with pytest.raises(ValueError,match='Engine runtime differs'):verifier(report,build)


def test_optional_actual_engine_record(verifier,evidence):
    path=DIRECTORY/'response-with-engine.json'
    if not path.is_file():pytest.skip('Full-engine measured result not published yet')
    _,build=evidence
    result=verifier(json.loads(path.read_text()),build)
    assert result['engine_checks']>0


def test_distinct_row_cannot_reuse_zero_overlay_hash(verifier,evidence):
    report,build=evidence
    active(report)['overlay_sha256']=zero(report)['overlay_sha256']
    with pytest.raises(ValueError,match='distinct overlay hashes'):
        verifier(report,build)


@pytest.mark.parametrize('value',[-1,True,1.5])
def test_changed_logit_count_requires_nonnegative_integer(verifier,synthetic_engine_evidence,value):
    report,build=synthetic_engine_evidence
    check=next(check for check in report['engine_checks'] if check['changed_logit_values']>0)
    check['changed_logit_values']=value
    with pytest.raises(ValueError,match='Changed logit count'):
        verifier(report,build)


@pytest.mark.parametrize('value',[-1,True,1.5,1000000])
def test_top1_count_requires_integer_within_sequence(verifier,synthetic_engine_evidence,value):
    report,build=synthetic_engine_evidence
    report['engine_checks'][0]['top1_agree_positions']=value
    with pytest.raises(ValueError,match='Top1 agreement count'):
        verifier(report,build)


@pytest.mark.parametrize('value',[-1,True,float('nan'),float('inf')])
def test_maximum_logit_change_requires_finite_nonnegative_value(verifier,synthetic_engine_evidence,value):
    report,build=synthetic_engine_evidence
    report['engine_checks'][0]['max_abs_logit_change']=value
    with pytest.raises(ValueError,match='maximum logit change'):
        verifier(report,build)


def test_unmodified_engine_ple_must_match_local_baseline_hash(verifier,synthetic_engine_evidence):
    report,build=synthetic_engine_evidence
    report['engine_checks'][0]['ple_output_sha256']='f'*64
    with pytest.raises(ValueError,match='Unmodified'):
        verifier(report,build)
