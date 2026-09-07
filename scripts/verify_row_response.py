#!/usr/bin/env python3
"""Audit published row-response evidence without model weights or dependencies.

This checks saved-record consistency and provenance, not fresh model inference.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ngramma_runtime.response_report import validate_report


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(report, build):
    validate_report(report)
    require(report['evidence_kind'] == 'measured', 'Synthetic demo is not experimental evidence')
    require(report['training_admitted'] is False, 'Experiment004 does not qualify training')
    require(report['native_baseline_matches_engine'] is True, 'Missing baseline PLE parity')
    require(report['sources_changed_during_run'] == [], 'Source changed during the local run')
    require(build['library_sha256'] == report['activation_library_sha256'], 'Activation library identity mismatch')
    require(build['linked_libraries'] == report['native_build_record']['linked_libraries'], 'Activation and native math use different CPU libraries')
    points = {point['epsilon']:point for point in report['points']}
    magnitudes = [2.**exponent for exponent in range(-20,-3,2)]
    expected = {0.,*[sign*epsilon for epsilon in magnitudes for sign in (-1,1)]}
    require(set(points) == expected, 'Missing or extra preregistered steps')
    baseline = points[0.]
    require(len({point['overlay_sha256'] for point in points.values()}) == len(points), 'Distinct row replacements must have distinct overlay hashes')
    for epsilon, point in points.items():
        require(point['activation_changed_bytes'] == point['activation_scale_changed_bytes']+point['activation_code_changed_bytes'], 'Scale/code byte accounting differs')
        require(point['replacement_changed_values'] == (160 if epsilon else 0), 'Row replacement coordinate count differs')
        same_encoding = point['activation_sha256'] == baseline['activation_sha256']
        require(same_encoding == (point['activation_changed_bytes'] == 0), 'Encoded-byte hash/count contradiction')
        for stage,count in (('key','key_changed_count'),('value','value_changed_count')):
            require((point['stage_sha256'][stage] == baseline['stage_sha256'][stage]) == (point[count] == 0), 'Projection hash/count contradiction')
        if same_encoding:
            require(point['key_changed_count'] == point['value_changed_count'] == point['ple_output_rms'] == point['local_scalar_delta'] == 0, 'Identical encoded inputs have a changed local response')
        if epsilon:
            require(math.isclose(point['realized_normalized_rms'],abs(epsilon),rel_tol=.01), 'Realized displacement differs from intended step')
    def changed(epsilon):
        return any(points[sign*epsilon]['key_changed_count'] or points[sign*epsilon]['value_changed_count'] for sign in (-1,1))
    plateau = [e for e in magnitudes if not changed(e)]
    active = [e for e in magnitudes if changed(e)]
    chosen = [max(plateau)] if plateau else []
    if active:
        chosen.append(active[0])
        larger = [e for e in magnitudes if e>active[0]]
        if larger: chosen.append(larger[0])
        if not plateau: chosen.append(magnitudes[-1])
    labels = ['zero',*[points[sign*e]['overlay_label'] for e in sorted(set(chosen)) for sign in (-1,1)]]
    require(labels == report['selected_engine_labels'], 'Full-engine selection differs from preregistered rule')
    smooth = report['smooth_control']
    derivative = smooth['autograd_directional_derivative']
    for point in smooth['finite_differences']:
        relative = abs(point['central_difference']-derivative)/max(abs(derivative),1e-12)
        require(math.isclose(relative,point['relative_error'],rel_tol=1e-12,abs_tol=1e-15), 'Smooth derivative error does not recompute')
    require(min(point['relative_error'] for point in smooth['finite_differences']) < 1e-6, 'Smooth local control has no accurate finite-difference sample')
    checks = report.get('engine_checks')
    if checks is None:
        return {'points':len(points),'engine_checks':0,'status':'local response only'}
    require([check['label'] for check in checks] == ['unmodified',*labels], 'Missing full-engine checks')
    baseline_hash = report['capture_identity']['logits_sha256']
    point_labels = {point['overlay_label']:point for point in points.values()}
    spec = report['experiment']
    for check in checks:
        capture = check['capture']
        require(type(check['changed_logit_values']) is int and check['changed_logit_values'] >= 0, 'Changed logit count must be a nonnegative integer')
        require(type(check['top1_agree_positions']) is int and 0 <= check['top1_agree_positions'] <= len(spec['tokens']), 'Top1 agreement count outside sequence bounds')
        require(type(check['max_abs_logit_change']) in (int,float) and math.isfinite(check['max_abs_logit_change']) and check['max_abs_logit_change'] >= 0, 'Invalid maximum logit change')
        for key in ('lens_sha256','device','cache_type','repack','flash_attention','context','batch','microbatch','rope_overrides','chunk_size','threads','tokens','identity_sha256'):
            require(capture[key] == report['capture_identity'][key], 'Engine fixture identity/config mismatch: '+key)
        require(check['fresh_process'] is True and check['live_routing_and_attention'] is True, 'Missing fresh live-engine condition')
        require(check['ple_matches_local_native'] is True and check['gathered_replacement_verified'] is True, 'Missing gather/PLE verification')
        bitwise = capture['logits_sha256'] == baseline_hash
        require(bitwise == check['logits_bitwise_equal_to_original'], 'Byte hash does not support bitwise equality claim')
        require(bitwise == (check['changed_logit_values'] == 0), 'Logit count/hash contradiction')
        require(math.isclose(check['logit_margin']-spec['engine_margin']['baseline'],check['logit_margin_delta'],abs_tol=1e-14), 'Margin delta does not recompute')
        label = check['label']
        if label == 'unmodified':
            require(capture['overlay_sha256'] is None and bitwise, 'Unmodified engine control failed')
            require(check['ple_output_sha256'] == baseline['stage_sha256']['output'], 'Unmodified PLE hash differs from baseline')
            continue
        point = point_labels[label]
        require(capture['overlay_sha256'] == point['overlay_sha256'], 'Engine used a different overlay')
        require(check['ple_output_sha256'] == point['stage_sha256']['output'], 'Engine/local PLE hashes differ')
        require(check['logit_margin_delta'] == point['engine_logit_margin_delta'], 'Explorer/engine margin mismatch')
        if point['key_changed_count'] == point['value_changed_count'] == 0:
            require(bitwise, 'Unchanged projections changed full-engine logits')
    for name,value in build['linked_libraries'].items():
        require(report['engine_runtime_sha256'][name] == value, 'Engine runtime differs from activation replay runtime')
    return {'points':len(points),'engine_checks':len(checks),'status':'saved records consistent; no inference rerun'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-only',action='store_true')
    args = parser.parse_args()
    directory = ROOT/'experiments/004-row-response'
    result_path = directory/('response.json' if args.local_only else 'response-with-engine.json')
    try:
        report = json.loads(result_path.read_text())
        build = json.loads((directory/'activation-build.json').read_text())
        summary = verify(report,build)
        if not args.local_only:
            require(report['local_response_sha256'] == hashlib.sha256((directory/'response.json').read_bytes()).hexdigest(), 'Engine checks bind a different local response')
    except (KeyError,TypeError,ValueError,OSError) as error:
        parser.error(str(error))
    print(json.dumps(summary))


if __name__ == '__main__': main()
