"""Export full-context diagnostic results and compare development accuracy."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from openjev.research.text_distillation import digest, file_hash, write_new


def publish(args):
    summary = json.loads(args.summary.read_text())
    memory = json.loads(args.memory.read_text())
    plan = json.loads((args.out/'plan.json').read_text())
    if (digest(plan) != digest(summary['plan']) or summary['status'] != 'development_only' or
            summary['confirmation_executed'] or (args.out/'receipt.json').exists()):
        raise ValueError('Fresh export and exact frozen plan required')
    if memory['plan']['packet_sha256'] != plan['packet_sha256']:
        raise ValueError('Comparisons require identical audited packets')
    write_new(args.out/'summary.json', summary)
    frozen, tuned = summary['methods']['frozen']['fits'], summary['methods']['finetuned']['fits']
    write_new(args.out/'receipt.json', {'source_summary_sha256': file_hash(args.summary),
        'memory_summary_sha256': file_hash(args.memory), 'freeze_commit': args.freeze_commit,
        'hardware': 'Apple M5 Max', 'fine_tune_device': 'mps', 'head_warmup_device': 'cpu',
        'platform': platform.platform(), 'python': platform.python_version(),
        'environment': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy', 'matplotlib')},
        'frozen_head_fits': len(frozen), 'fine_tuned_fits': len(tuned),
        'head_warmup_seconds': sum(f['head_training_seconds'] for f in frozen),
        'additional_finetune_seconds': sum(f['additional_training_seconds'] for f in tuned),
        'shared_frozen_forward_seconds': summary['features']['forward_seconds'],
        'head_warmup_updates': sum(h['updates'] for f in frozen for h in f['history']),
        'finetune_updates': sum(h['updates'] for f in tuned for h in f['history']),
        'teacher_requests': 0, 'official_test_opened': False, 'clinc_confirmation_scored': False})
    names = memory['plan']['protocol']['modes']+['frozen_joint', 'finetuned_joint']
    labels = ['Question\nonly', 'Mean\ncontext', 'Single\nread', 'Recurrent\n3 reads',
              'Recurrent\n6 reads', 'Coverage\n3 reads', 'Frozen\nfull context', 'Fine-tuned\nfull context']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, part, title in zip(axes, ('dev_in', 'dev_shift'),
                             ('Depth 0-2: 2,947 development questions',
                              'Depth 3-5: 1,941 development questions'), strict=True):
        for i, mode in enumerate(names):
            if mode in memory['methods']:
                values = [f['parts'][part]['metrics']['accuracy']*100 for f in memory['methods'][mode]['fits']]
                color = '#8290a5'
            else:
                method = 'frozen' if mode == 'frozen_joint' else 'finetuned'
                values = [f['metrics'][part]['accuracy']*100 for f in summary['methods'][method]['fits']]
                color = '#cf9950' if method == 'frozen' else '#248a76'
            ax.scatter(i+np.linspace(-.10, .10, 3), values, s=20, color=color, alpha=.65)
            ax.scatter(i, np.mean(values), marker='D', s=56, color=color)
            ax.text(i, max(values)+1.1, f'{np.mean(values):.1f}', ha='center', fontsize=9)
        ax.axvline(5.5, color='#a3a3a3', linestyle=':', linewidth=1)
        ax.axhline(50, color='#aaaaaa', linestyle=':', linewidth=1)
        ax.set(title=title, ylabel='Accuracy (%)', ylim=(40, 101))
        ax.set_xticks(range(len(names)), labels, fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('RuleTaker development: memory reads and full-context representation', fontsize=14, fontweight='bold')
    fig.savefig(args.out/'development.png', dpi=160, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    write_new(args.out/'figures.json', {'development.png': file_hash(args.out/'development.png')})


def plot_audit(path, out):
    """Keep the post-hoc shortcut diagnosis visibly separate from frozen scores."""
    audit = json.loads(path.read_text())
    keys = list(audit['models'])
    labels = ['Question\nonly', 'Mean\ncontext', 'Single\nread', 'Recurrent\n3 reads',
              'Recurrent\n6 reads', 'Coverage\n3 reads', 'Frozen\nfull context', 'Fine-tuned\nfull context',
              'Word "not"\ncontrol']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, metric, title in zip(axes, ('accuracy', 'four_group_macro_accuracy'),
                                ('Raw accuracy on 1,941 depth 3-5 questions',
                                 'Equal weight for each label / negation group'), strict=True):
        for i, key in enumerate(keys):
            values = [f['parts']['dev_shift'][metric]*100 for f in audit['models'][key]['fits']]
            color = '#248a76' if key.endswith(':finetuned') else '#8290a5'
            ax.scatter(i+np.linspace(-.09, .09, 3), values, color=color, s=20, alpha=.65)
            ax.scatter(i, np.mean(values), color=color, s=54, marker='D')
            ax.text(i, max(values)+1.4, f'{np.mean(values):.1f}', ha='center', fontsize=9)
        value = audit['baseline']['dev_shift'][metric]*100
        ax.scatter(len(keys), value, s=60, color='#bd6441', marker='s')
        ax.text(len(keys), value+1.4, f'{value:.1f}', ha='center', fontsize=9)
        ax.axhline(50, color='#aaaaaa', linestyle=':', linewidth=1)
        ax.set(title=title, ylabel='Accuracy (%)', ylim=(40, 101))
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('Post-hoc shortcut audit: high raw accuracy can hide weak reasoning', fontsize=14, fontweight='bold')
    fig.savefig(out, dpi=160, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('summary', 'memory', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--freeze-commit', required=True)
    p.add_argument('--audit', type=Path)
    args = p.parse_args()
    publish(args)
    if args.audit:
        plot_audit(args.audit, args.out/'shortcut-audit.png')
        write_new(args.out/'shortcut-figure.json', {'audit_sha256': file_hash(args.audit),
                                                  'figure_sha256': file_hash(args.out/'shortcut-audit.png')})
