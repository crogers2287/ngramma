#!/usr/bin/env python3
"""Check saved experiment-003 evidence consistency with the standard library.

This validates recorded summaries and recomputes their gate decisions, not raw
probability tensors, logits, native binaries, or model inference.
"""
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT/'experiments/003-attention'
COMPONENTS = {'baseline.json': (False, False), 'native-rope.json': (True, False),
              'native-attention.json': (False, True), 'combined.json': (True, True),
              'combined-strides.json': (True, True)}
FULL_RESULTS = ('full-native-attention.json', 'full-unicode-chat.json', 'full-eos-repeat.json')
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
        if filename == 'full-native-attention.json':
            require(record['tokens'] == tokens, 'Primary full-forward token sequence changed')
        else:
            case_name = filename.removeprefix('full-').removesuffix('.json')
            cases = json.loads((ROOT/'data/compatibility-cases.json').read_text())
            require(record['tokens'] == next(case['tokens'] for case in cases if case['name'] == case_name),
                    'Additional fixture token sequence changed')
        verify_full(record, identity)
        print(f'{filename}: saved metrics and fixed-threshold decisions internally consistent')
        count += 1
    print(f'PASS: {count} saved experiment-003 records checked. No raw tensor/logit comparison or inference rerun.')


if __name__ == '__main__':
    main()
