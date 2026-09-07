#!/usr/bin/env python3
"""Summarize paired outcomes, termination errors, and measured process cost."""
import argparse
from collections import Counter
import json
from pathlib import Path

from verify_results import EXP, ROOT, read, search, sha, verify


def analyze(allow_partial=False):
    verification = verify(require_complete=not allow_partial)
    tasks = search.generate_tasks() + search.generate_tasks('controls')
    baseline = read(ROOT / 'experiments/005-behavior/discovery.json')
    guidance = read(ROOT / 'experiments/005-behavior/followup.json')
    report = read(EXP / 'search-results.json')
    original = {j['id']: j['response'] for j in baseline['jobs']}
    records = [('original', baseline, ''), ('format_instruction', guidance, 'format/')]
    records += [(r['label'], read(EXP / (r['label'] + '.json')), '') for r in report['results']]
    outcomes = []
    for name, record, prefix in records:
        by_id = {j['id']: j['response'] for j in record['jobs']}
        groups = {split: Counter() for split in ('dev', 'controls')}
        families = {family: Counter() for family in {t.family for t in tasks}}
        misleading_first_tokens, gains, losses, changed = [], [], [], []
        choices, gain_labels, loss_labels = Counter(), Counter(), Counter()
        for task in tasks:
            response = by_id[prefix + task.id]
            value = search.grade_completion(task, response)
            before = search.grade_completion(task, original[task.id])
            if value['well_formed'] and value['terminated']:
                choices[value['parsed_answer']] += 1
            category = 'correct' if value['success'] else value['error_kind']
            for group in (groups[task.split], families[task.family]):
                group['total'] += 1
                group[category] += 1
                group['cap_exhausted'] += response['stop_reason'] != 'eog'
            if not value['success'] and response['first_step']['greedy_token_id'] == {'A': 32, 'B': 33}[task.answer]:
                misleading_first_tokens.append(task.id)
            if not before['success'] and value['success']:
                gains.append(task.id)
                gain_labels[task.answer] += 1
            if before['success'] and not value['success']:
                losses.append(task.id)
                loss_labels[task.answer] += 1
            if response['text_bytes_hex'] != original[task.id]['text_bytes_hex']:
                changed.append(task.id)
        outcomes.append({'condition': name,
                         'groups': {k: dict(sorted(v.items())) for k, v in groups.items()},
                         'families_including_controls': {k: dict(sorted(v.items())) for k, v in sorted(families.items())},
                         'gains_from_original': gains, 'losses_from_original': losses,
                         'terminated_answer_labels': dict(sorted(choices.items())),
                         'gains_by_correct_label': dict(sorted(gain_labels.items())),
                         'losses_by_correct_label': dict(sorted(loss_labels.items())),
                         'changed_completion_text': changed,
                         'correct_literal_first_token_but_failed_complete_answer': misleading_first_tokens})
    candidates = [r for _, r, _ in records[2:]]
    return {'schema': 'ngramma.instruction-analysis/v1', 'complete': verification['complete'],
            'search_sha256': sha(EXP / 'search-results.json'), 'outcomes': outcomes,
            'candidate_processes': len(candidates),
            'scored_candidate_generations': 24 * len(candidates),
            'extra_candidate_lookup_generations': 1,
            'candidate_process_seconds': sum(r['seconds'] for r in candidates),
            'candidate_authentication_seconds': sum(m['seconds'] for r in candidates for m in r['parallel_verification']),
            'candidate_model_bytes_hashed': sum(m['bytes'] for r in candidates for m in r['parallel_verification']),
            'maximum_candidate_rss_bytes': max(r['peak_rss_bytes'] for r in candidates),
            'minimum_candidate_available_bytes': min(r['minimum_available_bytes'] for r in candidates),
            'selected_winner': verification['selected_winner'], 'heldout_evaluated': False,
            'scope': 'Fixed response-contract development search; no general accuracy or trained-overlay claim.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--check', type=Path, help='Compare a previously saved analysis to recomputed values')
    args = p.parse_args()
    result = analyze(args.allow_partial)
    if args.check and read(args.check) != result:
        raise ValueError('Saved analysis differs from recomputed outcomes')
    encoded = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(encoded)
    if args.output or args.check:
        print(json.dumps({k: v for k, v in result.items() if k != 'outcomes'}, indent=2))
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
