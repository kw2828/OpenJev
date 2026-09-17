"""Export a completed head-development screen without opening confirmation data."""
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
    parser.add_argument('--freeze-commit', required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    if args.out.exists() or summary['status'] != 'development_only' or summary['confirmation_executed']:
        raise ValueError('Fresh output and development-only evidence required')
    fits = [f for r in summary['methods'].values() for f in r['fits']]
    if len(fits) != summary['verified_fits']:
        raise ValueError('Fit count mismatch')
    write_new(args.out/'summary.json', summary)
    write_new(args.out/'receipt.json', {'summary_sha256': file_hash(args.summary),
        'frozen_code_commit': args.freeze_commit, 'hardware': 'Apple M5 Max', 'training_device': 'cpu',
        'fit_count': len(fits), 'updates': sum(h['updates'] for f in fits for h in f['history']),
        'total_training_seconds': sum(f['training_seconds'] for f in fits),
        'confirmation_executed': False, 'calibration_scored': False, 'teacher_requests': 0})
    methods = list(summary['methods'])
    labels = [m.replace('_', '\n').capitalize() for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for i, mode in enumerate(methods):
        color = '#23947b' if mode == summary['selected_candidate'] else '#8491a7'
        for ax, metric, scale in zip(axes, ['id_accuracy', 'id_nll'], [100, 1], strict=True):
            values = [f['metrics'][metric]*scale for f in summary['methods'][mode]['fits']]
            ax.scatter(i, np.mean(values), marker='D', s=60, color=color)
            ax.scatter(i+np.linspace(-.07, .07, len(values)), values, s=19, color='#142036')
            ax.text(i, max(values)+(.08 if scale == 100 else .003),
                    f'{np.mean(values):.2f}' if scale == 100 else f'{np.mean(values):.3f}', ha='center', fontsize=9)
    required = (summary['methods'][summary['strongest_control']]['mean_metrics']['id_accuracy']+
                summary['plan']['protocol']['continuation']['minimum_accuracy_gain_over_best_control'])*100
    axes[0].axhline(required, color='#b64b4b', linestyle=':', label='Predeclared minimum gain')
    axes[0].legend(loc='lower left', fontsize=9)
    axes[0].set(ylabel='Development in-scope accuracy (%)', ylim=(94.3, 95.9), title='Three paired fits per variant')
    axes[1].set(ylabel='Development NLL (lower is better)', ylim=(.18, .27), title='Probability quality on known intents')
    for ax in axes:
        ax.set_xticks(range(len(methods)), labels)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Corrective recurrence: small development gain; continuation criteria not met',
                 fontsize=13, fontweight='bold')
    fig.savefig(args.out/'development.png', dpi=160, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)


if __name__ == '__main__':
    main()
