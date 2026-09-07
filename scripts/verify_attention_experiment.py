#!/usr/bin/env python3
"""Check saved experiment-003 evidence consistency with the standard library.

This validates recorded summaries and recomputes their gate decisions, not raw
probability tensors, logits, native binaries, or model inference.
"""
import json
import math
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT/'experiments/003-attention'
COMPONENTS = {'baseline.json': (False, False), 'native-rope.json': (True, False),
              'native-attention.json': (False, True), 'combined.json': (True, True),
              'combined-strides.json': (True, True)}
FULL_RESULTS = ('full-native-attention.json', 'full-native-attention-provenance.json', 'full-unicode-chat.json', 'full-eos-repeat.json',
                'full-native-attention-ple-scale.json', 'full-unicode-chat-ple-scale.json', 'full-eos-repeat-ple-scale.json')
METRICS = {'hc_mixed', 'hc_inject', 'query_projection', 'query_split', 'query_norm',
           'key_projection', 'key_norm', 'value_projection', 'query_rope', 'key_rope',
           'gate_input', 'gate_sigmoid', 'gated_output', 'projected_output',
           *[f'{prefix}/{name}' for prefix in ('chained', 'exact_rotary_input')
             for name in ('scores', 'probabilities', 'values')]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonnegative(value, name):
    require(type(value) in (float, int) and math.isfinite(value) and value >= 0,
            f'Invalid nonnegative finite {name}')


def sha(value):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value), 'Invalid SHA-256 identity')


def metric(value):
    for key in ('max_abs', 'relative_rms'):
        nonnegative(value[key], key)
    count, total = value['different_values'], value['values']
    require(type(count) is int and type(total) is int and 0 <= count <= total and total > 0,
            'Invalid difference/element counts')
    require((count == 0) == (value['max_abs'] == 0), 'Exactness/count disagree')
    require((value['max_abs'] == 0) == (value['relative_rms'] == 0), 'Zero max/RMS disagree')


def common(record, identity):
    require(record['identity_sha256'] == identity, 'Unexpected checkpoint identity')
    tokens = record['tokens']
    require(isinstance(tokens, list) and 0 < len(tokens) <= 128 and
            all(type(token) is int and 0 <= token < 2**31 for token in tokens), 'Invalid short token sequence')
    require(record['diagnostic_only'] is True, 'Diagnostic scope missing')
    require(record['admitted_for_gradients'] is False, 'Unexpected gradient admission')
    sha(record['native_library_sha256'])
    build = record['native_build_record']
    require(build['schema_version'] == 1 and build['library_sha256'] == record['native_library_sha256'],
            'Native build/library identity mismatch')
    for group in ('sources', 'headers', 'linked_libraries'):
        require(bool(build[group]), f'Empty native build {group}')
        for digest in build[group].values():sha(digest)
    require(bool(record['source_sha256']), 'Missing run source identities')
    source_hashes = record['source_sha256']
    for digest in source_hashes.values() if isinstance(source_hashes, dict) else [source_hashes]:
        sha(digest)
    require(not record.get('sources_changed_during_run'), 'Sources changed during run')


def verify_component(record, identity, tokens, flags):
    common(record, identity)
    require(record['schema'] == 'ngramma-attention-stages/v1', 'Unexpected component schema')
    require(record['tokens'] == tokens, 'Component token sequence changed')
    require(record['native_rope'] is flags[0] and record['native_attention'] is flags[1], 'Component ablation flags disagree')
    require(record['engine_positive_probability_mask_matches_dense_causal'] is True, 'Dense causal support not established')
    require(record['layer'] == 3 and record['cache_slots'] == 256 and record['query_heads'] == 24 and
            record['kv_heads'] == 2 and record['head_dim'] == 256, 'Audited component geometry changed')
    require(METRICS <= set(record['metrics']) <= METRICS | {'complete_layer3'}, 'Missing/extra component metrics')
    for value in record['metrics'].values():metric(value)
    sha(record['reference_metadata_sha256'])
    nonnegative(record['seconds_including_initialization'], 'elapsed seconds')


def verify_full(record, identity):
    common(record, identity)
    require(record['schema'] == 'ngramma-full-parity/v1', 'Unexpected full-forward schema')
    require('sources_changed_during_run' in record and not record['sources_changed_during_run'], 'Source stability evidence missing')
    rows = record['intermediates']
    require(len(rows) == 49 and {row['name'] for row in rows} ==
            {'ple_embd', *[f'l_last-{i}' for i in range(48)]}, 'Missing/duplicate/extra layer evidence')
    for row in rows:metric(row)
    errors = record['selected_logprob_error_by_position']
    require(len(errors) == len(record['tokens']), 'Log-probability position count mismatch')
    for value in errors:nonnegative(value, 'selected log-probability error')
    require(max(errors) == record['selected_logprob_error_max'], 'Selected error maximum mismatch')
    for key in ('max_logit_error', 'mean_logit_error', 'forward_seconds', 'peak_rss_gib'):
        nonnegative(record[key], key)
    require(record['mean_logit_error'] <= record['max_logit_error'], 'Mean logit error exceeds maximum')
    require(record['thresholds'] == {'selected_logprob_nats_strictly_below': .02,
                                     'relative_rms_strictly_below': .01}, 'Qualification thresholds changed')
    require(record['logit_parity'] is (max(errors) < .02), 'Saved logit gate is inconsistent')
    require(record['intermediate_parity'] is (max(row['relative_rms'] for row in rows) < .01), 'Saved intermediate gate is inconsistent')
    agreement = record['top1_agreement']
    nonnegative(agreement, 'top-token agreement')
    count = agreement*len(errors)
    require(agreement <= 1 and abs(count-round(count)) < 1e-9, 'Invalid top-token match count')
    layouts = record['verified_attention_layouts']
    require(len(layouts) == 12 and {item['layer'] for item in layouts} == set(range(3,48,4)),
            'Missing/duplicate full-attention layout evidence')
    for item in layouts:
        require(type(item['cache_slots']) is int and len(errors) <= item['cache_slots'] <= 4096,
                'Invalid verified cache width')
        sha(item['probability_sha256'])
    require(record['rope_config']['mode'] == 40, 'Unexpected RoPE mode')
    require(all(record[key] is True for key in ('native_attention', 'native_recurrent', 'native_repack', 'native_reductions')),
            'Full native execution condition changed')
    require(record['attention_calls'] == {'rope_multi': 24, 'attention_scores_strided': 12,
                                          'softmax_ext': 12, 'batched_matmul': 12}, 'Unexpected attention call coverage')
    require(not record['native_misses'] and record['native_calls'].get('registered_vector_weight') == 48,
            'Incomplete native matrix/vector coverage')
    for digest in record['reference_sha256'].values():sha(digest)



def verify_provenance(record):
    """Validate embedded provenance relationships, without access to raw files."""
    names = {'capture_identity', 'external_source_sha256', 'compared_tensor_sha256'}
    present = names & record.keys()
    if not present:
        return  # Initial historical full run predates this additive evidence.
    require(present == names, 'Incomplete enhanced capture/source provenance')
    capture = record['capture_identity']
    profile = {'identity_sha256': record['identity_sha256'], 'device': 'CPU',
               'cache_type': 'f32', 'repack': True, 'flash_attention': False,
               'rope_overrides': False, 'context': 128, 'batch': 32, 'microbatch': 32}
    for key, expected in profile.items():
        require(type(capture.get(key)) is type(expected) and capture[key] == expected,
                f'Enhanced capture profile mismatch: {key}')
    require(capture['tokens'] == record['tokens'] and len(record['tokens']) <= 32,
            'Enhanced capture token identity mismatch')
    require(type(capture['chunk_size']) is int and len(record['tokens']) <= capture['chunk_size'] <= 32,
            'Enhanced capture must use one complete bounded prefill')
    for key in ('lens_sha256', 'capture_script_sha256', 'metadata_sha256', 'logits_sha256'):
        sha(capture[key])
    require(capture['metadata_sha256'] == record['reference_sha256']['tensors.json'] and
            capture['logits_sha256'] == record['reference_sha256']['logits.f32'],
            'Enhanced capture/reference content hashes disagree')
    compared = record['compared_tensor_sha256']
    require(set(compared) == {'ple_embd', *[f'l_last-{i}' for i in range(48)]},
            'Enhanced provenance must identify all 49 compared tensors')
    for digest in compared.values():sha(digest)
    external = record['external_source_sha256']
    require(isinstance(external, dict) and all(any(key.startswith(prefix + '/') for key in external)
            for prefix in ('engraft', 'gguf')), 'Missing external ENGRAFT/GGUF source identities')
    for digest in external.values():sha(digest)



def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def ple_coefficient(width):
    require(type(width) is int and width > 0, 'Invalid PLE embedding width')
    return f32(1.0 / f32(math.sqrt(f32(width))))


def verify_ple_scaled_full(record):
    require(record['native_ple_scale'] is True and type(record['ple_scale_calls']) is int and
            record['ple_scale_calls'] == 1, 'Incomplete PLE scalar substitution')
    require(record['ple_scale_coefficient'] == ple_coefficient(2560), 'PLE scalar coefficient changed')
    for row in record['intermediates']:
        for key in ('actual_logical_sha256', 'reference_logical_sha256'):sha(row[key])
        equal = row['actual_logical_sha256'] == row['reference_logical_sha256']
        require(type(row['bitwise_equal']) is bool and row['bitwise_equal'] is equal,
                'Logical tensor hash/exactness claim disagree')
        if equal:
            require(row['max_abs'] == 0 and row['relative_rms'] == 0 and row['different_values'] == 0,
                    'Bitwise-equal tensor has nonzero numerical differences')
    sha(record['actual_logits_sha256'])
    equal = record['actual_logits_sha256'] == record['reference_sha256']['logits.f32']
    require(type(record['logits_bitwise_equal']) is bool and record['logits_bitwise_equal'] is equal,
            'Logit hash/exactness claim disagree')
    if equal:
        require(record['max_logit_error'] == 0 and record['mean_logit_error'] == 0 and
                record['selected_logprob_error_max'] == 0 and record['top1_agreement'] == 1,
                'Bitwise-equal logits have nonzero numerical differences')


def verify_ple_probe(record, identity):
    common(record, identity)
    require(record['schema'] == 'ngramma-ple-scalar-probe/v1' and record['layer'] == 1,
            'Unexpected PLE probe schema/layer')
    require(record['original_transcription_matches_upstream'] is True, 'Unverified PLE transcription')
    require(not record['sources_changed_during_run'] and not record['external_sources_changed_during_run'],
            'PLE probe source changed during run')
    require(record['n_embd'] == 2560 and record['original_divisor'] == math.sqrt(2560), 'Unexpected PLE dimension/divisor')
    coefficient = ple_coefficient(record['n_embd'])
    require(record['corrected_float32_coefficient'] == coefficient and
            record['corrected_coefficient_ieee754_le_hex'] == struct.pack('<f', coefficient).hex(),
            'PLE probe corrected scalar/value bits disagree')
    conditions = record['conditions']
    require(set(conditions) == {'original_division', 'float32_reciprocal_multiply'}, 'Missing/extra PLE scalar condition')
    for condition in conditions.values():
        require(condition['fresh_layer_state'] is True, 'PLE probe reused layer state')
        require(set(condition['metrics']) == {'gate', 'gated_value', 'conv_out', 'full_layer1'}, 'Missing PLE stage')
        for value in condition['metrics'].values():
            metric(value)
            counts = value['different_values_by_token']
            require(len(counts) == len(record['tokens']) and all(type(n) is int and n >= 0 for n in counts),
                    'Invalid per-token PLE difference counts')
            require(value['values'] % len(counts) == 0 and all(n <= value['values']//len(counts) for n in counts) and
                    sum(counts) == value['different_values'], 'PLE per-token/total count mismatch')
            first = next((i for i,n in enumerate(counts) if n), None)
            require(value['first_differing_token'] == first and
                    (value['first_differing_token'] is None or type(value['first_differing_token']) is int),
                    'PLE first differing token mismatch')
    for group in ('reference_sha256', 'compared_tensor_sha256', 'external_source_sha256'):
        require(bool(record[group]), 'Missing PLE reference/source hashes')
        for digest in record[group].values():sha(digest)
    require(set(record['compared_tensor_sha256']) == {'l_last-0', 'ple_embd', 'ple_gate-1',
            'ple_gated_value-1', 'ple_conv_out-1', 'l_last-1'}, 'Missing compared PLE tensor identities')
    capture = record['capture_identity']
    require(capture['identity_sha256'] == identity and capture['tokens'] == record['tokens'],
            'PLE capture identity/token mismatch')
    require(capture['metadata_sha256'] == record['reference_sha256']['tensors.json'] and
            capture['logits_sha256'] == record['reference_sha256']['logits.f32'], 'PLE capture content hash mismatch')


def main():
    identity = json.loads((ROOT/'data/model-identity.json').read_text())['identity_sha256']
    tokens = json.loads((ROOT/'data/replica-quantized-parity.json').read_text())['tokens']
    count = 0
    for filename, flags in COMPONENTS.items():
        path = DIRECTORY/filename
        require(path.is_file(), f'Required component evidence missing: {filename}')
        verify_component(json.loads(path.read_text()), identity, tokens, flags)
        if filename == 'combined-strides.json':
            record = json.loads(path.read_text())
            require(record['native_strides'] is True and all(value['max_abs'] == 0 for value in record['metrics'].values()),
                    'Final component exactness claim no longer matches evidence')
        print(f'{filename}: saved component evidence internally consistent')
        count += 1
    for filename in FULL_RESULTS:
        path = DIRECTORY/filename
        if not path.is_file():
            print(f'{filename}: absent; no full-forward claim checked for this fixture')
            continue
        record = json.loads(path.read_text())
        if filename in ('full-native-attention.json', 'full-native-attention-provenance.json', 'full-native-attention-ple-scale.json'):
            require(record['tokens'] == tokens, 'Primary full-forward token sequence changed')
        else:
            case_name = filename.removeprefix('full-').removesuffix('.json').removesuffix('-ple-scale')
            cases = json.loads((ROOT/'data/compatibility-cases.json').read_text())
            require(record['tokens'] == next(case['tokens'] for case in cases if case['name'] == case_name),
                    'Additional fixture token sequence changed')
        verify_full(record, identity)
        verify_provenance(record)
        if filename.endswith('-ple-scale.json'):
            verify_ple_scaled_full(record)
        if filename != 'full-native-attention.json':
            require('capture_identity' in record, 'Enhanced run missing capture provenance')
        print(f'{filename}: saved metrics and fixed-threshold decisions internally consistent')
        count += 1
    ple_path = DIRECTORY/'ple-gate-scaling.json'
    if ple_path.is_file():
        verify_ple_probe(json.loads(ple_path.read_text()), identity)
        count += 1
        print('ple-gate-scaling.json: scalar ablation and saved stage evidence internally consistent')
    print(f'PASS: {count} saved experiment-003 records checked. No raw tensor/logit comparison or inference rerun.')


if __name__ == '__main__':
    main()
