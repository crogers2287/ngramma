#!/usr/bin/env python3
"""Plot checked, actual whole-answer outcomes; optional Matplotlib dependency."""
import argparse
import hashlib
import json
from pathlib import Path

from verify_results import EXP, ROOT, read, search, verify


def outcome(task, response):
    grade = search.grade_completion(task, response)
    if grade['success']:
        return 0
    if grade['well_formed'] and grade['terminated']:
        return 1
    if grade['well_formed']:
        return 2
    return 3


def measured_matrix(exp=EXP, allow_partial=False):
    verification = verify(exp, require_complete=not allow_partial)
    tasks = search.generate_tasks() + search.generate_tasks('controls')
    baseline = read(ROOT / 'experiments/005-behavior/discovery.json')
    guidance = read(ROOT / 'experiments/005-behavior/followup.json')
    report = read(exp / 'search-results.json')
    series = [('Original', baseline, ''),
              ('Original + format instruction', guidance, 'format/')]
    series += [(r['label'], read(exp / (r['label'] + '.json')), '') for r in report['results']]
    matrix, scores = [], []
    for label, record, prefix in series:
        responses = {j['id']: j['response'] for j in record['jobs']}
        row = [outcome(task, responses[prefix + task.id]) for task in tasks]
        matrix.append(row)
        scores.append((row[:16].count(0), row[16:].count(0)))
    return verification, tasks, [x[0] for x in series], matrix, scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--output-prefix', type=Path, required=True)
    args = p.parse_args()
    verification, tasks, names, matrix, scores = measured_matrix(allow_partial=args.allow_partial)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'ngramma006'})
    colors = ['#228B72', '#D99A26', '#7374BA', '#E4E7EB']
    fig, ax = plt.subplots(figsize=(13, 3.4 + len(names) * .35))
    ax.imshow(matrix, cmap=ListedColormap(colors), vmin=-.5, vmax=3.5, aspect='auto')
    ax.set_yticks(range(len(names)), labels=names)
    short = [t.id.replace('dev-', '').replace('controls-', 'control-') for t in tasks]
    ax.set_xticks(range(len(tasks)), labels=short, rotation=60, ha='right')
    ax.tick_params(axis='both', length=0)
    for y in range(len(names) + 1):
        ax.axhline(y - .5, color='white', lw=1.5)
    for x in range(len(tasks) + 1):
        ax.axvline(x - .5, color='white', lw=1.5)
    for x in (3.5, 7.5, 11.5, 15.5):
        ax.axvline(x, color='#555555', lw=1)
    ax.axhline(1.5, color='#222222', lw=1.5)
    ax.set_xlim(-.5, 28)
    ax.text(25.5, -.85, 'Correct\ndev / controls', ha='center', va='bottom', fontsize=10)
    for y, (dev, controls) in enumerate(scores):
        ax.text(25.5, y, f'{dev}/16   {controls}/8', ha='center', va='center')
    for spine in ax.spines.values():
        spine.set_visible(False)
    state = 'complete' if verification['complete'] else 'in progress'
    fig.suptitle(f'Eight memory rows: actual generated answers ({state})', x=.02, ha='left', fontsize=17)
    ax.set_title('Four-token cap; exact A/B choice and end-of-generation required for success', loc='left', pad=26)
    legend = [Patch(facecolor=c, label=label) for c, label in zip(colors, [
        'Correct + terminated', 'Wrong choice + terminated', 'Unfinished A/B prefix', 'Malformed answer'])]
    fig.legend(handles=legend, loc='lower left', bbox_to_anchor=(.01, .05), ncol=4, frameon=False)
    winner = verification['selected_winner']
    footer = ('No candidate passes the fixed admission rule.' if verification['complete'] and winner is None
              else f'Selected candidate: {winner}; confirmation and holdout are separate.' if winner
              else 'Partial search: no winner may be selected yet.')
    fig.text(.02, .015, footer + '  The format-instruction row is a prompt baseline, not a memory edit.', fontsize=10)
    fig.subplots_adjust(left=.25, right=.98, top=.83, bottom=.25)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for suffix in ('.svg', '.png'):
        path = args.output_prefix.with_suffix(suffix)
        metadata = {'Date': None} if suffix == '.svg' else {'Software': 'Ngramma experiment 006'}
        fig.savefig(path, dpi=160, metadata=metadata)
        if suffix == '.svg':
            # Matplotlib leaves spaces at the ends of SVG path-command lines.
            # Keep generated source clean without changing its geometry.
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)
    print(json.dumps({'candidates': verification['candidates_replayed'], 'artifacts': artifacts}, indent=2))


if __name__ == '__main__':
    main()
