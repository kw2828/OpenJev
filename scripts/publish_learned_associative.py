"""Publish aggregate development results and a figure; keep model/data artifacts local."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from openjev.research.text_distillation import file_hash, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    if args.out.exists() or summary['status'] != 'development_only' or summary['confirmation_executed']:
        raise ValueError('Fresh output and development-only summary required')
    write_new(args.out/'summary.json', summary)
    write_new(args.out/'receipt.json', {'summary_sha256': file_hash(args.summary),
               'frozen_code_commit': 'eed5b97', 'hardware': 'Apple M5 Max', 'training_device': 'cpu',
               'fits': 12, 'training_updates_per_fit': 1770,
               'training_seconds_all_fits': sum(f['training_seconds'] for r in summary['methods'].values() for f in r['fits']),
               'tests': {'passed': 98, 'skipped_optional_encoder': 1},
               'teacher_requests': 0, 'confirmation_scored': False, 'calibration_scored': False,
               'claim': 'Development selection evidence only; no recurrent improvement established'})
    methods = list(summary['methods'])
    labels = ['Linear metric', 'Feedforward', 'Dense recurrence', 'Sparse recurrence']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    colors = ['#27977e', '#8392a9', '#ba6262', '#ca9c6e']
    for i, mode in enumerate(methods):
        fits = summary['methods'][mode]['fits']
        accuracy = [f['metrics']['id_accuracy']*100 for f in fits]
        latency = [f['latency']['median_ms']*1000 for f in fits]
        for ax, values in zip(axes, [accuracy, latency], strict=True):
            if ax is axes[0]:
                ax.scatter(i, np.mean(values), color=colors[i], marker='D', s=65)
            else:
                ax.bar(i, np.mean(values), color=colors[i], width=.65)
            ax.scatter(i+np.linspace(-.07, .07, len(values)), values, color='#152033', s=20)
            ax.text(i, max(values)+(.1 if ax is axes[0] else 1.4),
                    f'{np.mean(values):.2f}', ha='center', fontsize=9)
    for ax in axes:
        ax.set_xticks(range(len(methods)), labels, rotation=12)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set(ylim=(93.5, 96), ylabel='In-scope development accuracy (%)', title='Three fits per head; all 68,481 parameters')
    axes[1].set(ylim=(0, 60), ylabel='Mean per-fit median head time (microseconds)', title='CPU, one request; encoder and rejection excluded')
    fig.suptitle('Learning helps; tested recurrence does not beat the metric control', fontsize=14, fontweight='bold')
    fig.savefig(args.out/'development.png', dpi=160, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)


if __name__ == '__main__':
    main()
