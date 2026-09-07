#!/usr/bin/env python3
"""Plot saved CPU parity diagnostics; never run a model or alter evidence."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
GATE = .01
SOURCES = [
    ('Historical activation rounding', ROOT / 'data/replica-quantized-parity.json', '#65748B', '--'),
    ('Native primitives + recurrence', EXPERIMENT / 'full-native-recurrent.json', '#C96B25', '-.'),
    ('Native primitives + recurrence + repack', EXPERIMENT / 'full-native-repack.json', '#176F92', '-'),
]


def layer_errors(record):
    layers = {}
    for row in record['intermediates']:
        match = re.fullmatch(r'l_last-(\d+)', row['name'])
        if match:
            index = int(match.group(1))
            if index in layers:
                raise ValueError(f'Duplicate layer {index}')
            value = float(row['relative_rms'])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid relative RMS at layer {index}: {value}')
            layers[index] = value
    if set(layers) != set(range(48)):
        raise ValueError('Expected exactly l_last-0 through l_last-47; reshaped aliases are excluded')
    return [layers[index] for index in range(48)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=EXPERIMENT / 'figures')
    args = parser.parse_args()
    sources = list(SOURCES)
    sums_path = EXPERIMENT / 'full-native-sums.json'
    if sums_path.is_file():
        sources.append(('Native + repack + row sums + vector-gate fix', sums_path, '#735AA8', ':'))
    records = []
    for label, path, color, style in sources:
        if not path.is_file():
            parser.error(f'Required saved result is not available yet: {path.relative_to(ROOT)}')
        record = json.loads(path.read_text())
        records.append((label, record, layer_errors(record), color, style))
    tokens = records[0][1]['tokens']
    identity = records[0][1]['identity_sha256']
    if len(tokens) != 10:
        raise ValueError('This figure describes exactly ten tokens')
    for _, record, _, _, _ in records:
        if record['tokens'] != tokens or record['identity_sha256'] != identity:
            raise ValueError('Cannot compare differing model identities or token sequences')
        if record.get('admitted_for_gradients') is not False:
            raise ValueError('Figure assumes no gradient qualification; source record disagrees')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'ngramma-experiment-002-parity',
                         'savefig.facecolor': 'white'})
    figure = plt.figure(figsize=(12, 8.4), facecolor='white')
    grid = figure.add_gridspec(2, 1, height_ratios=[4, 1.45], hspace=.37,
                              left=.09, right=.97, top=.80, bottom=.16)
    axis = figure.add_subplot(grid[0])
    table_axis = figure.add_subplot(grid[1])
    positive = [value for _, _, series, _, _ in records for value in series if value > 0]
    floor = min(1e-6, min(positive)/2)
    has_zero = any(value == 0 for _, _, series, _, _ in records for value in series)
    for label, _, series, color, style in records:
        axis.plot(range(48), [max(value, floor) for value in series], label=label,
                  color=color, linestyle=style, linewidth=2.1,
                  marker='o', markersize=2.7, markevery=4)
    axis.axhline(GATE, color='#A12B3B', linestyle=':', linewidth=1.7)
    axis.text(46.5, GATE*1.17, 'Fixed intermediate gate: 1%',
              color='#A12B3B', ha='right', fontsize=10,
              bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .86, 'pad': 1})
    axis.set_yscale('log')
    axis.set_xlim(-.5, 47.5)
    axis.set_xticks([0, 8, 16, 24, 32, 40, 47])
    axis.set_xlabel('Layer index (0–47)')
    axis.set_ylabel('Relative RMS error vs engine reference\n(log scale; lower is better)')
    axis.yaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5]))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.grid(axis='y', which='major', color='#D8DCE2', linewidth=.7)
    axis.legend(loc='upper left', bbox_to_anchor=(0, 1.32), frameon=False,
                fontsize=10, ncol=1, handlelength=3, labelspacing=.4)

    def percent(value):
        return f'{value*100:.4g}%'

    cells = []
    for label, record, series, _, _ in records:
        crossed = next((i for i, value in enumerate(series) if value >= GATE), None)
        cells.append([label.replace('Native primitives + recurrence', 'Native + recurrence').replace('Native + repack + row sums + vector-gate fix', 'Native + sums + vector-gate fix'),
                      percent(series[0]), percent(series[-1]),
                      'None' if crossed is None else str(crossed),
                      f"{record['max_logit_error']:.4g}",
                      f"{record['top1_agreement']*100:.0f}%"])
    table_axis.axis('off')
    table = table_axis.table(cellText=cells,
                            colLabels=['Execution condition', 'Layer 0', 'Layer 47',
                                       'First ≥1% layer', 'Max logit error', 'Top-1 match'],
                            colWidths=[.34, .12, .12, .15, .15, .12],
                            cellLoc='center', colLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9.4)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_facecolor('#E9EDF2' if row == 0 else ('#F6F7F9' if row % 2 else 'white'))
        if row == 0:
            cell.set_text_props(weight='bold')
        if column == 0:
            cell.set_text_props(ha='left')
            if row:
                cell.set_text_props(color=records[row-1][3])

    figure.suptitle('CPU forward parity across 48 layers', x=.09, y=.982,
                     ha='left', fontsize=19, fontweight='bold')
    figure.text(.09, .939, 'Same 10-token sequence · frozen model · no training or gradient qualification',
                fontsize=11, color='#424B57')
    figure.text(.09, .105, 'Relative RMS = RMS(replica − reference) / max(RMS(reference), 10⁻¹²). Canonical layer outputs only.',
                fontsize=9, color='#424B57')
    zero_note = f' Zero errors displayed at {floor:.2g} on the log axis.' if has_zero else ''
    figure.text(.09, .080, 'Layer error reduction alone does not establish full forward parity or learned improvement.'+zero_note,
                fontsize=9, color='#424B57')
    figure.text(.09, .045, 'Sources: data/replica-quantized-parity.json; experiments/002-parity/full-native-{recurrent,repack' + (',sums' if len(records) == 4 else '') + '}.json',
                fontsize=8.2, color='#636B75')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'svg'):
        target = args.output_dir / f'cpu-forward-parity.{suffix}'
        figure.savefig(target, dpi=180, metadata={'Date': None} if suffix == 'svg' else {})
        print(target)
    plt.close(figure)


if __name__ == '__main__':
    main()
