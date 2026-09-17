"""Publish aggregates from matching frozen fits, never source passages or paid response traces."""
import argparse
import json
from pathlib import Path

import numpy as np

from .text_distillation import PROTOCOL, file_hash, load_packet, teacher_labels, write_new
from .text_student import categorical_metrics


def paired_interval(a, b, gold, seed=916):
    """Exploratory paired bootstrap over fits and items, stratified by gold class."""
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(np.array(gold) == k) for k in sorted(set(gold))]
    differences = a - b
    draws = []
    for _ in range(4000):
        fits = rng.integers(0, len(differences), len(differences))
        items = np.concatenate([rng.choice(group, len(group), replace=True) for group in classes])
        draws.append(float(differences[np.ix_(fits, items)].mean()))
    return {'delta': float(differences.mean()),
            'exploratory_95_percent_interval': np.percentile(draws, [2.5, 97.5]).tolist()}


def report(packet_dir, runs, out, teacher_dir=None):
    packet, manifest = load_packet(packet_dir)
    if out.exists():
        raise ValueError('Report exists; do not overwrite recorded results')
    results, matrices, hashes = {}, {}, {}
    tasks = sorted({r['task'] for r in packet['evaluation']})
    for arm in PROTOCOL['arms']:
        directory = runs / arm
        if not (directory / 'completed.json').exists():
            continue
        completion = json.loads((directory / 'completed.json').read_text())
        if (completion['status'] != 'completed' or completion['packet_sha256'] != manifest['packet_sha256']
                or completion['fit_seeds'] != PROTOCOL['fit_seeds'] or completion['arm'] != arm):
            raise ValueError('Unexpected completed fit identity')
        fits = [json.loads((directory / f'result-{seed}.json').read_text()) for seed in PROTOCOL['fit_seeds']]
        for seed, fit in zip(PROTOCOL['fit_seeds'], fits, strict=True):
            if (fit['arm'] != arm or fit['seed'] != seed or fit['packet_sha256'] != manifest['packet_sha256']
                    or [(r['id'], r['task'], r['gold']) for r in fit['predictions']] !=
                       [(r['id'], r['task'], r['gold']) for r in packet['evaluation']]):
                raise ValueError('Mismatched fit/evaluation identity')
            hashes[f'{arm}/result-{seed}.json'] = file_hash(directory / f'result-{seed}.json')
        results[arm] = {'tasks': {}, 'total_training_seconds': sum(f['training_seconds'] for f in fits)}
        matrices[arm] = {}
        for task in tasks:
            metrics = [categorical_metrics([r for r in f['predictions'] if r['task'] == task]) for f in fits]
            results[arm]['tasks'][task] = {
                'per_fit': [{'seed': seed, **m} for seed, m in zip(PROTOCOL['fit_seeds'], metrics, strict=True)],
                'mean_across_fits': {k: float(np.mean([m[k] for m in metrics])) for k in
                                    ('accuracy', 'macro_class_accuracy', 'nll', 'multiclass_brier',
                                     'median_latency_ms', 'p95_latency_ms')}}
            matrices[arm][task] = np.array([[float(r['choice'] == r['gold']) for r in f['predictions']
                                            if r['task'] == task] for f in fits])
    contrasts = {}
    for task in tasks:
        gold = [r['gold'] for r in packet['evaluation'] if r['task'] == task]
        contrasts[task] = {}
        for a, b in [('gold', 'untrained'), ('astra', 'untrained'), ('astra', 'gold')]:
            if a in results and b in results:
                contrasts[task][f'{a}_minus_{b}'] = paired_interval(matrices[a][task], matrices[b][task], gold)
    status = 'not_assessed_missing_astra'
    if all(arm in results for arm in PROTOCOL['arms']):
        passes = all(contrasts[t]['astra_minus_untrained']['delta'] >= 0.10 and
                     contrasts[t]['astra_minus_gold']['delta'] >= -0.05 for t in tasks)
        status = 'screening_rule_met' if passes else 'screening_rule_not_met'
    teacher = {'status': 'not_run', 'note': 'No Astra efficacy result available'}
    if teacher_dir is not None:
        labels = teacher_labels(teacher_dir, packet, manifest)
        teacher = {'status': 'completed', 'training_label_agreement': {
            task: float(np.mean([labels[r['id']] == r['gold'] for r in packet['train'] if r['task'] == task]))
            for task in tasks}, 'costs': json.loads((teacher_dir / 'summary.json').read_text())}
    if 'astra' in results and teacher['status'] != 'completed':
        raise ValueError('Astra report requires the matching teacher receipts')
    summary = {'protocol': PROTOCOL, 'packet_sha256': manifest['packet_sha256'],
               'status': 'completed' if all(a in results for a in PROTOCOL['arms']) else 'controls_only',
               'arms': results, 'contrasts': contrasts, 'teacher': teacher, 'continuation': status,
               'evidence_sha256': hashes,
               'caveats': ['Balanced, filtered, public subsets; prior model exposure is possible.',
                           'Three fits reuse the same evaluation items; they are not independent new test sets.',
                           'Intervals are exploratory, class-stratified and paired; no population claim.',
                           'Training and inference costs exclude encoder pretraining and model loading.',
                           'No claim of RL, JEPA transfer, novelty, calibration or production readiness.']}
    write_new(out / 'summary.json', summary)
    write_new(out / 'manifest.json', manifest)
    write_new(out / 'selection-audit.json', packet['selection_audit'])
    plot(summary, out / 'controls.png')
    print(json.dumps({'status': summary['status'], 'continuation': status, 'out': str(out)}, indent=2))


def plot(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    names = {'untrained': 'Untrained head', 'gold': 'Gold labels', 'astra': 'Astra labels'}
    for ax, task, title, baseline in zip(axes, ['boolq', 'clinc_domains'],
                                       ['BoolQ · 84 examples', 'CLINC domain routing · 88 examples'],
                                       [50., 100/11], strict=True):
        arms = list(summary['arms'])
        values = [summary['arms'][a]['tasks'][task]['mean_across_fits']['accuracy']*100 for a in arms]
        ax.bar(range(len(arms)), values, color=['#8290a6', '#36a78b', '#5965d9'][:len(arms)], width=.6)
        for i, arm in enumerate(arms):
            fits = summary['arms'][arm]['tasks'][task]['per_fit']
            ax.scatter(np.arange(len(fits))*.06 + i - .06, [r['accuracy']*100 for r in fits], color='#122039', s=20)
            ax.text(i, max([r['accuracy']*100 for r in fits]) + 3, f'{values[i]:.1f}%', ha='center')
        ax.axhline(baseline, color='#aaaaaa', linestyle=':', label='Uniform-choice expected accuracy')
        ax.set(xticks=range(len(arms)), xticklabels=[names[a] for a in arms], ylim=(0, 105),
               title=title, ylabel='Accuracy (%)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='upper left', fontsize=7)
    fig.suptitle('Text student pilot' + (' · Astra training pending' if summary['status'] != 'completed' else ''),
                 fontsize=15, fontweight='bold')
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--teacher', type=Path)
    args = parser.parse_args()
    report(args.packet, args.runs, args.out, args.teacher)


if __name__ == '__main__':
    main()
