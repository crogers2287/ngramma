#!/usr/bin/env python3
"""Regenerate exportable figures from saved measured response JSON."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(__file__).with_name('response-with-engine.json'))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = json.loads(args.input.read_text())
    points = data['points']
    engine = any(p.get('engine_logit_margin_delta') is not None for p in points)
    fig, axes = plt.subplots(3 if engine else 2,1,figsize=(9,10 if engine else 7),layout='constrained')
    colors = {-1:'#a34825',1:'#18758b'}
    for sign in (-1,1):
        series = sorted([p for p in points if p['epsilon']*sign>0],key=lambda p:abs(p['epsilon']))
        x = [abs(p['epsilon'])*100 for p in series]
        label = 'Negative direction' if sign<0 else 'Positive direction'
        axes[0].plot(x,[p['ple_output_rms'] for p in series],marker='o',color=colors[sign],label=label)
        axes[1].plot(x,[p['activation_scale_changed_bytes'] for p in series],marker='o',color=colors[sign],label=label+', scales')
        axes[1].plot(x,[p['activation_code_changed_bytes'] for p in series],marker='s',linestyle='--',color=colors[sign],label=label+', codes')
        if engine:
            measured = [p for p in series if p.get('engine_logit_margin_delta') is not None]
            axes[2].plot([abs(p['epsilon'])*100 for p in measured],
                         [p['engine_logit_margin_delta'] for p in measured],marker='o',color=colors[sign],label=label)
    titles = ['Small row changes can disappear at the quantizer',
              'The first surviving sample changes scales, not integer codes',
              'Fresh full-engine response at three preselected magnitudes']
    labels = ['Memory output RMS change','Changed activation bytes','Final logit-margin change']
    for ax,title,ylabel in zip(axes,titles,labels):
        ax.set_xscale('log')
        ax.set_title(title,loc='left',fontsize=12,fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Intended row RMS change / original row RMS (%)')
        ax.axhline(0,color='#919ba6',linewidth=.7)
        ax.grid(alpha=.18)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(fontsize=8,loc='best',frameon=False)
    axes[0].set_yscale('symlog',linthresh=1e-8)
    axes[1].set_yscale('symlog',linthresh=1)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    fig.suptitle('Ngramma · one existing row, one fixed direction, 17 tokens\nMeasured CPU response; no training or capability claim',fontsize=14,fontweight='bold')
    directory = Path(__file__).with_name('figures')
    directory.mkdir(exist_ok=True)
    for extension in ('png','svg'):
        fig.savefig(directory/('row-response.'+extension),dpi=160)
    plt.close(fig)


if __name__ == '__main__': main()
