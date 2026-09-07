#!/usr/bin/env python3
"""Plot saved forward agreement; this is not a model-quality benchmark."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
before = ['full-native-attention-provenance.json', 'full-unicode-chat.json', 'full-eos-repeat.json']
after = ['full-native-attention-ple-scale.json', 'full-unicode-chat-ple-scale.json', 'full-eos-repeat-ple-scale.json']
conditions = [(label, [json.loads((ROOT/name).read_text()) for name in files], color, offset)
              for label, files, color, offset in
              [('Attention corrections', before, '#b84f38', -.18),
               ('Also memory-gate correction', after, '#237d62', .18)]]
labels = ['Original\n10 tokens', 'Unicode/chat\n17 tokens', 'EOS/repeats\n12 tokens']
x = np.arange(3)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), layout='constrained')
for ax, threshold, title, ylabel, getter in [
    (axes[0], .02, 'Selected-token log-probability error', 'Maximum absolute error (nats)',
     lambda r: r['selected_logprob_error_max']),
    (axes[1], .01, 'Largest intermediate disagreement', 'Relative RMS error',
     lambda r: max(row['relative_rms'] for row in r['intermediates']))]:
    largest = max(getter(r) for _, records, _, _ in conditions for r in records)
    top = max(largest, threshold)*1.30
    for label, records, color, offset in conditions:
        values = [getter(r) for r in records]
        ax.bar(x+offset, values, color=color, width=.34, label=label)
        for i, value in enumerate(values):
            ax.text(i+offset, value+top*(.055 if value == 0 else .025), '0' if value == 0 else f'{value:.4f}',
                    ha='center', fontsize=9, color=color)
    ax.axhline(threshold, color='#666666', linewidth=1, linestyle='--')
    ax.text(2.5, threshold+top*.07, f'gate {threshold:g}', ha='right', fontsize=8, color='#555555')
    ax.set_ylim(0, top)
    ax.set_xticks(x, labels)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('CPU agreement with the unchanged model: three short fixtures', fontsize=13)
handles, names = axes[0].get_legend_handles_labels()
fig.legend(handles, names, loc='outside lower center', ncol=2, frameon=False, fontsize=10)
directory = ROOT/'figures'
directory.mkdir(exist_ok=True)
fig.savefig(directory/'forward-agreement.png', dpi=180)
fig.savefig(directory/'forward-agreement.svg')
