#!/usr/bin/env python3
"""Regenerate the research figure from published numerical summaries."""
from pathlib import Path
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'svg.hashsalt':'flash-next-ngram-research-v1'})
fig, ax = plt.subplots(figsize=(10, 4.8), layout='constrained')
for filename, label, color in [
    ('replica-parity.json', 'Initial reference', '#697386'),
    ('replica-quantized-parity.json', 'Reference after diagnostic fixes', '#006da8'),
]:
    record = json.loads((ROOT/'data'/filename).read_text())
    layers = {int(x['name'].split('-')[1]):x['relative_rms'] for x in record['intermediates'] if re.fullmatch(r'l_last-\d+', x['name'])}
    indices = sorted(layers)
    ax.plot(indices, [100*layers[i] for i in indices], label=label, color=color, linewidth=2)
ax.axhline(1, color='#aa4a3f', linestyle='--', linewidth=1.2, label='1% diagnostic intermediate-error gate')
ax.set(xlabel='Backbone layer (zero-based)', ylabel='Relative RMS error versus engine (%)',
       title='Component fixes did not establish full-model agreement', xlim=(0,47), ylim=(0,None))
ax.grid(axis='y', alpha=.2)
ax.spines[['top','right']].set_visible(False)
ax.legend(loc='upper left', frameon=False)
fig.supxlabel('One 10-token sequence · CPU inference reference · Original rows · No trained overlay', fontsize=9, color='#535c6b')
out = ROOT/'figures'
out.mkdir(exist_ok=True)
fig.savefig(out/'replica-drift.png', dpi=180, metadata={'Software':'Matplotlib'})
fig.savefig(out/'replica-drift.svg', metadata={'Date':None, 'Creator':'Matplotlib'})
print('Wrote figures/replica-drift.png and figures/replica-drift.svg')
