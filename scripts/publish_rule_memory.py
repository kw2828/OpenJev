"""Publish aggregate RuleTaker evidence and figures, without source passages."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from openjev.research.text_distillation import digest, file_hash, write_new

LABELS = {'question_only': 'Question\nonly', 'mean_context': 'Mean\ncontext', 'single_read': 'Single\nread',
          'recurrent_3': 'Recurrent\n3 reads', 'recurrent_6': 'Recurrent\n6 reads', 'coverage_3': 'Coverage\n3 reads'}
COLORS = {'question_only': '#8b919a', 'mean_context': '#ba904a', 'single_read': '#294f8d',
          'recurrent_3': '#4a9287', 'recurrent_6': '#9074ae', 'coverage_3': '#ca644c'}


def publish(summary_path, out, freeze_commit):
    summary = json.loads(summary_path.read_text())
    plan = json.loads((out/'plan.json').read_text())
    if (summary['status'] != 'development_only' or summary['confirmation_executed'] or
            digest(summary['plan']) != digest(plan) or (out/'receipt.json').exists()):
        raise ValueError('Need exact published frozen plan and fresh development results')
    fits = [f for m in summary['methods'].values() for f in m['fits']]
    if len(fits) != len(plan['protocol']['modes'])*len(plan['protocol']['seeds']):
        raise ValueError('Incomplete fit family')
    write_new(out/'summary.json', summary)
    write_new(out/'receipt.json', {'source_summary_sha256': file_hash(summary_path),
        'freeze_commit': freeze_commit, 'hardware': 'Apple M5 Max', 'training_device': 'cpu',
        'feature_device': summary['features']['device'], 'fits': len(fits),
        'updates': sum(h['updates'] for f in fits for h in f['history']),
        'training_seconds': sum(f['training_seconds'] for f in fits),
        'shared_encoding_seconds': summary['features']['encoding_seconds'],
        'feature_time_scope': 'Deduplicated frozen English-string forward passes; excludes loading and serialization',
        'teacher_requests': 0, 'official_test_opened': False, 'clinc_confirmation_scored': False})
    methods = list(summary['methods'])
    # JSON canonicalization sorts keys; retain preregistered display order.
    methods = plan['protocol']['modes']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, part, title in zip(axes[:2], ('dev_in', 'dev_shift'),
                             ('Depth 0-2: 2,947 development questions',
                              'Depth 3-5: 1,941 development questions'), strict=True):
        for i, mode in enumerate(methods):
            values = [f['parts'][part]['metrics']['accuracy']*100 for f in summary['methods'][mode]['fits']]
            ax.scatter(i+np.linspace(-.09, .09, len(values)), values, color=COLORS[mode], s=20, alpha=.6)
            ax.scatter(i, np.mean(values), color=COLORS[mode], marker='D', s=54, zorder=3)
            ax.text(i, max(values)+1.2, f'{np.mean(values):.1f}', ha='center', fontsize=9)
        ax.set(title=title, ylabel='Accuracy (%)', ylim=(40, 100))
        ax.set_xticks(range(len(methods)), [LABELS[m] for m in methods], fontsize=8)
        ax.axhline(50, linestyle=':', color='#9fa6b0', linewidth=1)
    for mode in methods:
        fits = summary['methods'][mode]['fits']
        x = np.mean([f['parts']['dev_shift']['latency']['median_us'] for f in fits])
        y = summary['methods'][mode]['means']['dev_shift']['accuracy']*100
        axes[2].scatter(x, y, color=COLORS[mode], s=58, label=LABELS[mode].replace('\n', ' '))
    axes[2].set(title='Shift accuracy and measured head cost', xlabel='Mean median CPU head latency (microseconds)',
                ylabel='Depth 3-5 accuracy (%)', ylim=(40, 100))
    axes[2].legend(loc='upper right', fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('English rule memory: development evidence, three fits per head', fontsize=14, fontweight='bold')
    fig.savefig(out/'development.png', dpi=160, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for mode in methods:
        fits = summary['methods'][mode]['fits']
        values = []
        for depth in range(6):
            part = 'dev_in' if depth < 3 else 'dev_shift'
            values.append(np.mean([f['parts'][part]['metrics']['by_depth'][str(depth)]['accuracy']*100 for f in fits]))
        # Do not draw a continuous line across the change in world distribution.
        ax.plot(range(3), values[:3], marker='o', color=COLORS[mode], label=LABELS[mode].replace('\n', ' '))
        ax.plot(range(3, 6), values[3:], marker='o', color=COLORS[mode])
    ax.axvline(2.5, linestyle=':', color='#888888')
    ax.set(title='Accuracy by annotated depth: source worlds also change at depth 3', xlabel='Annotated query depth',
           ylabel='Development accuracy (%), mean of three fits', ylim=(40, 100), xticks=range(6))
    ax.legend(loc='lower left', ncol=2, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.2)
    fig.savefig(out/'depth.png', dpi=160, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    write_new(out/'figures.json', {name: file_hash(out/name) for name in ('development.png', 'depth.png')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--freeze-commit', required=True)
    a = p.parse_args()
    publish(a.summary, a.out, a.freeze_commit)
