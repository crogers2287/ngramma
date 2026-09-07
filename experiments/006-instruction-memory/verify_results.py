#!/usr/bin/env python3
"""Check published instruction-edit evidence without weights or model inference.

This replays saved generations and their artifact bindings. It cannot recreate
native captures, authenticate absent model files, or establish generalization.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments/005-behavior'))
spec = importlib.util.spec_from_file_location('instruction_search_replay', EXP / 'search.py')
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)

OUTPUT_KEYS = ('prompt_tokens', 'generated_tokens', 'text', 'text_bytes_hex',
               'first_step', 'stop_reason', 'stopped_on_eog', 'eog_token')
DIRECTIONS = ('rademacher', 'alternating', 'hint_contrast')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def fixed_candidates(recipes):
    expected = [('zero', 0.0)]
    for direction in DIRECTIONS:
        for exponent in (8, 6):
            for sign, multiplier in (('minus', -1), ('plus', 1)):
                expected.append((f'{direction}-{sign}-{exponent}', multiplier * 2**-exponent))
    require([(c['label'], c['epsilon']) for c in recipes['candidates']] == expected,
            'Candidate directions, magnitudes, or order changed')
    rows = recipes['row_ids']
    require(len(rows) == len(set(rows)) == 8 and rows == sorted(rows), 'Selected rows must be eight unique sorted IDs')
    require(recipes['row_heads'] == list(range(8, 16)), 'Expected eight trigram heads')
    require(recipes['selected_trigram'] == recipes['instruction_tokens'][:3], 'Selected trigram differs from earliest instruction triple')
    zero = recipes['candidates'][0]
    require(zero['changed_values'] == 0 and zero['replacement_sha256'] == recipes['anchor_sha256'], 'Zero candidate changes anchors')
    for candidate in recipes['candidates'][1:]:
        require(candidate['changed_values'] == 1280, 'Expected 1280 changed FP32 values')
        require(len(candidate['relative_row_displacement_rms']) == 8, 'Missing row displacement')
        require(all(math.isclose(v, abs(candidate['epsilon']), rel_tol=1e-4)
                    for v in candidate['relative_row_displacement_rms']), 'Realized row displacement differs from recipe')
    return recipes['candidates']


def verify_authentication(record, build, identity):
    require(record['profile'].get('verification_library_sha256') == build['binary_sha256'], 'Unqualified authentication binary')
    require(record['runtime_sha256']['libllama.so'] == build['runtime_llama_sha256'], 'Authentication ABI binding changed')
    markers = record.get('parallel_verification', [])
    require(len(markers) == 1, 'Missing or extra full-model authentication record')
    marker = markers[0]
    require(marker.get('full_file_sha256') is True and marker.get('cache_reused') is False,
            'Model authentication must hash complete files without a cache')
    require(marker.get('shards') == len(identity['shards']) and
            marker.get('bytes') == sum(s['bytes'] for s in identity['shards']), 'Incomplete model shard authentication')
    require(type(marker.get('workers')) is int and 1 <= marker['workers'] <= build['maximum_hash_workers'], 'Unbounded authentication workers')
    require(math.isfinite(marker['seconds']) and marker['seconds'] > 0, 'Invalid authentication duration')
    require(record['peak_rss_bytes'] <= 80 * 2**30 and record['minimum_available_bytes'] >= 24 * 2**30,
            'Recorded resource limit violated')
    require(0 < record['seconds'] <= 900, 'Recorded process exceeded time budget')


def fixed_jobs(record, witness=False, legacy=False):
    """Only the capture output directory may vary across reproduction machines."""
    requested = search.jobs(controls=True)
    if witness:
        require(len(record['jobs']) > len(requested), 'Missing lookup witness')
        actual = record['jobs'][len(requested)]
        directory = actual['request'].get('output_dir')
        require(isinstance(directory, str) and bool(directory), 'Missing capture directory')
        requested.append({**requested[2], 'id': 'lookup-witness',
                          'generate': {**requested[2]['generate'], 'compact': False},
                          'capture': ['ple_embd'], 'output_dir': directory})
    if legacy:
        actual = record['jobs'][-1]
        requested.append({'id': 'legacy-reference',
                          'tokens': [18374, 50203, 198, 220, 1330, 1838, 26453, 220,
                                     248045, 846, 198, 5834, 248046, 198, 248045, 74455, 198],
                          'chunk_size': 32, 'output_dir': actual['request'].get('output_dir'), 'capture': []})
    actual_jobs = [{'id': j['id'], **j['request']} for j in record['jobs']]
    require(actual_jobs == requested, 'Requests differ from the frozen task/witness protocol')
    # The runner hashes the input JSON's insertion order. Zero qualification
    # placed output_dir before capture; finite candidates did the reverse.
    # Equality above checks semantics before retaining that serialization order.
    return actual_jobs


def verify_lookup(lookup, record, candidate, recipes):
    require(lookup['identity_sha256'] == recipes['identity_sha256'] and
            lookup['overlay_sha256'] == candidate['overlay_sha256'], 'Lookup witness belongs to another model or overlay')
    require(lookup['actual_equals_intended_bytes'] is True and
            lookup['changed_values_from_original'] == candidate['changed_values'], 'Lookup witness does not establish intended edits')
    response = next(j['response'] for j in record['jobs'] if j['id'] == 'lookup-witness')
    positions = len(response['prompt_tokens']) + len(response['generated_tokens']) - 1
    require(lookup['positions'] == positions and lookup['row_accesses'] == 16 * positions,
            'Lookup witness does not cover evaluated sequence')
    require(lookup['occurrences'] == recipes['coverage']['dev-decimal-0'], 'Lookup accesses differ from selected instruction rows')
    require(len(lookup['capture_files_sha256']) == response['prefill_calls'] + response['decode_calls'], 'Missing captured gather blocks')


def winner_for(results):
    eligible = [r for r in results if r['eligible']]
    def rank(item):
        direction = next(i for i, name in enumerate(DIRECTIONS) if item['label'].startswith(name))
        return (-item['development_correct'], abs(item['epsilon']), direction, int(item['epsilon'] > 0))
    return min(eligible, key=rank)['label'] if eligible else None


def verify(exp=EXP, root=ROOT, require_complete=True):
    baseline_path = root / 'experiments/005-behavior/discovery.json'
    baseline = read(baseline_path)
    recipes = read(exp / 'candidates.json')
    candidates = fixed_candidates(recipes)
    for name, digest in recipes['inputs_sha256'].items():
        path = exp / name if name == 'tokenization.json' else baseline_path.parent / name
        require(sha(path) == digest, 'Changed recipe input: ' + name)
    require(recipes['identity_sha256'] == baseline['identity_sha256'], 'Recipe model differs from baseline')
    task_ids = {t.id for t in search.generate_tasks() + search.generate_tasks('controls')}
    require(len(task_ids) == 24 and sum(x.startswith('dev-') for x in task_ids) == 16 and
            sum(x.startswith('controls-') for x in task_ids) == 8, 'Search partition changed')
    require(set(recipes['coverage']) == task_ids, 'Coverage includes missing or held-out search tasks')
    build = read(exp / 'parallel-verify-build.json')
    identity = read(root / 'data/model-identity.json')
    auth_tests = read(exp / 'verification-tests.json')
    require(auth_tests['tests'] == 11 and all(auth_tests[k] == 0 for k in ('failures', 'errors', 'skipped')),
            'Authentication negative tests did not all pass')
    require(auth_tests['verification_source_sha256'] == build['source_sha256'] and
            auth_tests['test_source_sha256'] == sha(root / 'tests/test_parallel_verify.py'), 'Authentication tests belong to different source')
    require(build['source_sha256'] == sha(root / 'src/ngramma_runtime/native/parallel_verify.cpp') and
            build['archived_overlay_header_sha256'] == sha(root / 'reference/engine/memory-overlay.h'), 'Authentication source changed')

    zero = read(exp / 'zero.json')
    qualification = read(exp / 'zero-validation.json')
    require(qualification['passed'] is True and qualification['baseline_sha256'] == sha(baseline_path) and
            qualification['zero_record_sha256'] == sha(exp / 'zero.json') and
            qualification['lookup_sha256'] == sha(exp / 'zero-lookup.json') and
            qualification['zero_overlay_sha256'] == candidates[0]['overlay_sha256'], 'Stale zero-overlay qualification')
    verify_authentication(zero, build, identity)
    require(qualification['verification_library_sha256'] == build['binary_sha256'] and
            qualification['full_shard_verification'] == zero['parallel_verification'], 'Zero verifier binding changed')
    search.assess(zero, baseline, candidates[0], fixed_jobs(zero, witness=True, legacy=True))
    for original, repeated in zip(baseline['jobs'], zero['jobs']):
        if original['request'].get('tokenize_only'):
            require(repeated['response'] == original['response'], 'Zero overlay changed tokenization')
        else:
            require(all(original['response'][k] == repeated['response'][k] for k in OUTPUT_KEYS), 'Zero overlay changed generation or scores')
    require(qualification['matching_generated_tasks'] == 24 and qualification['matching_legacy_tokens'] == 17 and
            qualification['legacy_logits_sha256'] == 'a55a60e98074b4b6a98173ca05b65545b12c9855c7aff951e35c6bef40a7829b', 'Legacy zero control changed')
    verify_lookup(read(exp / 'zero-lookup.json'), zero, candidates[0], recipes)

    audit = read(exp / 'input-audit.json')
    require(audit['passed'] is True and audit['recipe_sha256'] == sha(exp / 'candidates.json'), 'Input audit is stale')
    expected_inspections = [{'label': c['label'], 'overlay_sha256': c['overlay_sha256'],
                             'stored_row_format': 'absolute FP32', 'row_count': 8} for c in candidates]
    require(audit['inspected_overlays'] == expected_inspections and audit['heldout_search_tasks'] == 0,
            'Overlay inspection or partition audit changed')
    report = read(exp / 'search-results.json')
    require(report['candidate_recipe_sha256'] == sha(exp / 'candidates.json'), 'Search recipe binding changed')
    count = len(report['results'])
    require(1 <= count <= 12 and report['complete'] is (count == 12), 'Invalid search completion state')
    if require_complete:
        require(report['complete'], 'Search is incomplete; use --allow-partial for progress checks')
    replayed = []
    for index, (candidate, saved) in enumerate(zip(candidates[1:], report['results'])):
        record_path = exp / (candidate['label'] + '.json')
        record = read(record_path)
        verify_authentication(record, build, identity)
        requested = fixed_jobs(record, witness=index == 0)
        result = search.assess(record, baseline, candidate, requested)
        result['record_sha256'] = sha(record_path)
        require(len(result['tasks']) == 24 and {t['id'] for t in result['tasks']} == task_ids,
                'Scored task partition changed')
        require(result == saved, 'Candidate summary differs from raw completions: ' + candidate['label'])
        if index == 0:
            verify_lookup(read(exp / 'first-candidate-lookup.json'), record, candidate, recipes)
        replayed.append(result)
    expected_winner = winner_for(replayed) if report['complete'] else None
    require(report['winner'] == expected_winner, 'Winner violates fixed admission or ranking')
    require(report['heldout_evaluated'] is False, 'Search record must not claim separate holdout evaluation')
    return {'schema': 'ngramma.instruction-evidence/v1', 'complete': report['complete'],
            'candidates_replayed': count, 'scored_generations': 24 * count,
            'zero_matching_generations': 24, 'selected_rows': 8, 'edited_values': 1280,
            'eligible_candidates': sum(r['eligible'] for r in replayed),
            'selected_winner': expected_winner, 'heldout_evaluated': False,
            'note': 'Saved evidence replay only; no inference, model authentication, or raw capture recomputation.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = verify(require_complete=not args.allow_partial)
    encoded = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
