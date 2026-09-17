"""Publish aggregate bound-rule results and all compact learned operators."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from openjev.research.text_distillation import digest, file_hash, write_new


def publish(args):
    summary = json.loads(args.summary.read_text())
    plan = json.loads((args.out/'plan.json').read_text())
    if summary['plan'] != plan or (args.out/'receipt.json').exists():
        raise ValueError('Need matching frozen plan and fresh result export')
    write_new(args.out/'summary.json', summary)
    fits = [f for m in summary['methods'].values() for f in m['fits'] if 'seed' in f]
    write_new(args.out/'receipt.json', {'summary_sha256': file_hash(args.summary), 'frozen_code_commit': args.freeze_commit,
        'hardware': 'Apple M5 Max', 'training_device': 'cpu', 'fits': len(fits), 'fixed_controls': 3,
        'updates': sum(h['updates'] for f in fits for h in f['history']),
        'training_seconds': sum(f['training_seconds'] for f in fits),
        'preparation': json.loads((args.runs/'prepared.json').read_text()),
        'teacher_requests': 0, 'confirmation_executed': False, 'official_test_opened': False})
    for name in summary['plan']['protocol']['variants']:
        for seed in summary['plan']['protocol']['seeds']:
            path = args.runs/f'{name}-{seed}.pt'
            fit = next(f for f in summary['methods'][name]['fits'] if f['seed'] == seed)
            if file_hash(path) != fit['checkpoint_sha256']:
                raise ValueError('Checkpoint changed before export')
            weights = torch.load(path, weights_only=True, map_location='cpu')
            write_new(args.out/'weights'/f'{name}-{seed}.json', {
                'architecture': 'BoundRuleNet operator 5-32-1 with tanh then sigmoid',
                'variant': name, 'seed': seed, 'steps': plan['protocol']['variants'][name][0],
                'collapse_entities': plan['protocol']['variants'][name][1],
                'plan_sha256': digest(plan), 'source_checkpoint_sha256': file_hash(path),
                'weights': {k: v.tolist() for k, v in weights.items()}})
    order = ['facts_only', 'fixed_1', 'learned_1', 'collapsed_16', 'learned_6', 'learned_16', 'fixed_32']
    labels = ['Facts\nonly', 'Fixed\n1 step', 'Learned\n1 step', 'Collapsed\nentities',
              'Learned\n6 steps', 'Learned\n16 steps', 'Fixed\n32 steps']
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
    for i, name in enumerate(order):
        m = summary['methods'][name]
        color = '#248a76' if name == 'learned_16' else '#294f8d' if name == 'fixed_32' else '#8392a7'
        values = [[f['metrics']['dev_shift']['macro_accuracy']*100 for f in m['fits']],
                  [f['metrics']['challenge']['counterfactual_pair_success']*100 for f in m['fits']]]
        for ax, v in zip(axes, values, strict=True):
            offsets = np.linspace(-.08, .08, len(v)) if len(v) > 1 else [0]
            ax.scatter(i+np.array(offsets), v, s=22, color=color, alpha=.65)
            ax.scatter(i, np.mean(v), marker='D' if len(v) > 1 else 's', s=52, color=color)
            ax.text(i, max(v)+2, f'{np.mean(v):.1f}', ha='center', fontsize=9)
    axes[0].set(title='949 questions in 150 fresh shift worlds', ylabel='Four-group macro accuracy (%)')
    axes[1].set(title='240 constructed relation-reversal pairs', ylabel='All four answers correct per pair (%)')
    for ax in axes:
        ax.set_xticks(range(len(order)), labels, fontsize=9)
        ax.set_ylim(-3, 110)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('Explicit binding solves the task; the fixed logical solver also scores 100%',
                 fontsize=14, fontweight='bold')
    fig.savefig(args.out/'development.png', dpi=160, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    write_new(args.out/'figures.json', {'development.png': file_hash(args.out/'development.png')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('summary', 'runs', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--freeze-commit', required=True)
    publish(p.parse_args())
