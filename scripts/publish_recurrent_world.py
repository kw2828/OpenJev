"""Plot completed MiniGrid memory study aggregates without training or evaluation."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ARMS = ('current_ppo', 'recurrent_ppo', 'reward_prediction', 'world_prediction')
LABELS = {'current_ppo': 'Current-only PPO', 'recurrent_ppo': 'Recurrent PPO',
          'reward_prediction': '+ reward / termination', 'world_prediction': '+ full prediction',
          'random': 'Uniform random'}
COLORS = {'current_ppo': '#8392a7', 'recurrent_ppo': '#416d9c',
          'reward_prediction': '#8d71aa', 'world_prediction': '#248a76', 'random': '#ba9058'}
FILENAMES = ('world-model-results.png', 'world-model-results.svg', 'world-model-figures.json')
CURVE_FILENAMES = ('world-model-learning.png', 'world-model-learning.svg')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, description):
    if not np.isfinite(a) or not np.isfinite(b) or abs(a - b) > 1e-10:
        raise ValueError(f'Inconsistent or nonfinite {description}')


def load_completed_report(report):
    """Validate identities and aggregate replay before constructing any chart."""
    source = report / 'summary.json' if report.is_dir() else report
    summary = json.loads(source.read_text())
    plan_path = source.parent / 'plan.json'
    plan = json.loads(plan_path.read_text())
    p = summary['protocol']
    if plan['protocol'] != p or summary['plan_sha256'] != sha(plan_path):
        raise ValueError('Summary must match its adjacent frozen plan')
    seeds, sizes = p['seeds'], p['eval_sizes']
    if (tuple(p['arms']) != ARMS or len(seeds) != 3 or len(set(seeds)) != 3
            or len(sizes) != 3 or len(set(sizes)) != 3 or p['train_size'] not in sizes):
        raise ValueError('Expected four arms, three fits each and three map sizes')
    training = {(r['arm'], r['seed']): r for r in summary['training']}
    expected = {(arm, seed) for arm in ARMS for seed in seeds}
    budget = p['environments'] * p['rollout'] * p['updates']
    if len(summary['training']) != len(expected) or set(training) != expected:
        raise ValueError('Missing or duplicated training fits')
    for record in training.values():
        if (record['status'] != 'completed' or record['interactions'] != budget
                or not np.isfinite(record['training_seconds']) or record['training_seconds'] <= 0):
            raise ValueError('Training receipt is incomplete or invalid')
    close(summary['training_seconds'], sum(r['training_seconds'] for r in training.values()),
          'total training time')
    close(summary['interactions'], budget * len(expected), 'total interaction budget')

    expected_eval = {(arm, seed, mode) for arm, seed in expected
                     for mode in (('intact',) if arm == 'current_ppo' else ('intact', 'reset'))}
    observed, scores, random_seen = set(), {}, False
    for record in summary['evaluation']:
        arm = record['arm']
        if arm == 'random':
            if random_seen:
                raise ValueError('Duplicate random baseline')
            random_seen = True
            seed, mode = None, 'intact'
        else:
            seed, mode = record['seed'], record['mode']
            identity = (arm, seed, mode)
            if identity not in expected_eval or identity in observed:
                raise ValueError('Missing or duplicated evaluation identities')
            observed.add(identity)
            if record['checkpoint_sha256'] != training[(arm, seed)]['checkpoint_sha256']:
                raise ValueError('Evaluation used a different checkpoint')
        results = record['results']
        if len(results) != len(sizes) or {r['size'] for r in results} != set(sizes):
            raise ValueError('Evaluation size coverage mismatch')
        for result in results:
            size = result['size']
            rows = result['episodes']
            first = p['evaluation_seed_start'] + size * 10000
            if [r['seed'] for r in rows] != list(range(first, first + p['evaluation_episodes'])):
                raise ValueError('Evaluation episode identities mismatch')
            for row in rows:
                if sum(bool(row[k]) for k in ('success', 'wrong_goal', 'timeout')) != 1:
                    raise ValueError('Invalid episode outcome partition')
            for key in ('success', 'wrong_goal', 'timeout'):
                close(result[key], np.mean([r[key] for r in rows]), f'{arm} {key}')
            scores[(arm, seed, mode, size)] = result['success']
    if observed != expected_eval or not random_seen:
        raise ValueError('Incomplete evaluation conditions')

    conditions = [*ARMS, 'random', *(arm + '_reset' for arm in ARMS[1:])]
    if set(summary['averages']) != set(conditions) or set(summary['per_fit']) != set(conditions):
        raise ValueError('Aggregate condition mismatch')
    for name in conditions:
        mode = 'reset' if name.endswith('_reset') else 'intact'
        arm = name.removesuffix('_reset')
        fit_seeds = [None] if arm == 'random' else seeds
        for size in sizes:
            values = [scores[(arm, seed, mode, size)] for seed in fit_seeds]
            close(summary['averages'][name][str(size)]['success'], np.mean(values), 'mean success')
            rows = summary['per_fit'][name][str(size)]
            if len(rows) != len(values) or not np.allclose(
                    sorted(r['success'] for r in rows), sorted(values), rtol=0, atol=1e-10):
                raise ValueError('Per-fit success mismatch')
    checks = {'same_size_success': np.mean([
        scores[('world_prediction', seed, 'intact', p['train_size'])] for seed in seeds])
        >= p['continuation']['same_size_success_minimum']}
    for arm in p['continuation']['controls']:
        for size in sizes:
            delta = (np.mean([scores[('world_prediction', seed, 'intact', size)] for seed in seeds])
                     - np.mean([scores[(arm, seed, 'intact', size)] for seed in seeds]))
            close(summary['differences'][arm][str(size)], delta, 'continuation contrast')
            checks[f'{arm}_size_{size}'] = bool(delta >= p['continuation'][
                'minimum_gain_each_size_vs_each_control'])
    if checks != summary['checks'] or bool(all(checks.values())) != summary['continuation_passed']:
        raise ValueError('Continuation rule does not replay')
    return source, plan_path, summary, training, scores


def load_learning_curves(run, summary, training):
    curves, hashes = {}, {}
    p = summary['protocol']
    expected_interactions = np.arange(1, p['updates'] + 1) * p['environments'] * p['rollout']
    for identity, receipt in training.items():
        arm, seed = identity
        name = f'{arm}-{seed}'
        path = run / 'fits' / name / 'learning.jsonl'
        digest = sha(path)
        if digest != receipt['learning_sha256']:
            raise ValueError(f'Learning log hash mismatch for {name}')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if (len(rows) != p['updates'] or [r['update'] for r in rows] != list(range(1, p['updates'] + 1))
                or not np.array_equal([r['interactions'] for r in rows], expected_interactions)):
            raise ValueError(f'Incomplete or reordered learning curve for {name}')
        values = [r['recent_success'] for r in rows]
        for row, value in zip(rows, values, strict=True):
            if value is None:
                if row['completed_episodes'] != 0:
                    raise ValueError(f'Missing training success after completed episodes for {name}')
            elif not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Invalid training success for {name}')
        curves[identity] = np.array([np.nan if v is None else 100 * v for v in values])
        hashes[name] = digest
    return expected_interactions, curves, hashes


def format_axis(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#a9b1bb')
    ax.tick_params(colors='#364354', length=3)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#e1e6ec', linewidth=.7)


def mark_fits(ax, x, values, color, width=.075):
    offsets = np.linspace(-width / 2, width / 2, len(values)) if len(values) > 1 else [0.]
    ax.scatter(x + np.array(offsets), values, s=25, marker='o', facecolors='white',
               edgecolors=color, linewidths=1.2, zorder=5)


def figure(summary, training, scores):
    p = summary['protocol']
    seeds, sizes = p['seeds'], p['eval_sizes']
    style = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
             'axes.labelsize': 10, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(13.2, 9.3), facecolor='white')
        grid = fig.add_gridspec(2, 2, height_ratios=[1.12, 1], hspace=.48, wspace=.26)
        success_ax = fig.add_subplot(grid[0, :])
        reset_ax, cost_ax = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
        fig.subplots_adjust(left=.075, right=.98, top=.82, bottom=.13)
        fig.suptitle('MiniGrid Memory: recurrent prediction pilot', x=.075, y=.976,
                     ha='left', fontsize=18, fontweight='bold', color='#233044')
        fig.text(.075, .938, 'Three training fits per arm  |  Final snapshots  |  Development only',
                 fontsize=11, color='#5c6878')
        fig.legend(handles=[Patch(facecolor=COLORS[a], label=LABELS[a]) for a in (*ARMS, 'random')],
                   loc='upper left', bbox_to_anchor=(.069, .91), ncol=5, frameon=False,
                   fontsize=9.5, handlelength=1.1, columnspacing=1.8)

        centers, width = np.arange(len(sizes)), .15
        for i, arm in enumerate((*ARMS, 'random')):
            fit_seeds = [None] if arm == 'random' else seeds
            x = centers + (i - 2) * width
            values = [[100 * scores[(arm, seed, 'intact', size)] for seed in fit_seeds] for size in sizes]
            means = [np.mean(v) for v in values]
            success_ax.bar(x, means, width * .83, color=COLORS[arm], alpha=.84, zorder=3)
            for position, v, mean in zip(x, values, means, strict=True):
                if arm == 'random':
                    success_ax.scatter(position, mean, s=24, marker='D', color=COLORS[arm], zorder=5)
                else:
                    mark_fits(success_ax, position, v, COLORS[arm])
                success_ax.text(position, max(v) + 3, f'{mean:.1f}', ha='center', va='bottom',
                                fontsize=9, color='#364354')
        size_labels = [f'{size} × {size}\n' + ('Training size' if size == p['train_size'] else 'Size shift')
                       for size in sizes]
        success_ax.set(title='A. Success on unseen episode seeds', ylabel='Episodes successful (%)',
                       xticks=centers, xticklabels=size_labels, ylim=(-2, 112), yticks=np.arange(0, 101, 25))
        success_ax.set_title(success_ax.get_title(), loc='left', pad=11, fontweight='bold')
        success_ax.set_title('', loc='center')

        reset_values = []
        width = .22
        for i, arm in enumerate(ARMS[1:]):
            x = centers + (i - 1) * width
            values = [[100 * (scores[(arm, seed, 'intact', size)] - scores[(arm, seed, 'reset', size)])
                       for seed in seeds] for size in sizes]
            reset_values.extend(v for row in values for v in row)
            reset_ax.bar(x, [np.mean(v) for v in values], width * .82, color=COLORS[arm], alpha=.84, zorder=3)
            for position, v in zip(x, values, strict=True):
                mark_fits(reset_ax, position, v, COLORS[arm], width=.095)
        bottom, top = min(0, min(reset_values)), max(0, max(reset_values))
        pad = max(6, (top - bottom) * .17)
        reset_ax.set(xticks=centers, xticklabels=[f'{s} × {s}' for s in sizes],
                     ylabel='Intact minus reset success (percentage points)', ylim=(bottom - pad, top + pad))
        reset_ax.set_title('B. Effect of retaining recurrent state', loc='left', pad=11, fontweight='bold')
        reset_ax.axhline(0, color='#657185', linewidth=.8, zorder=4)
        reset_ax.text(.01, .975, 'Positive values favor intact memory', transform=reset_ax.transAxes,
                      ha='left', va='top', fontsize=9, color='#5c6878')

        for i, arm in enumerate(ARMS):
            values = [training[(arm, seed)]['training_seconds'] for seed in seeds]
            cost_ax.bar(i, np.mean(values), .58, color=COLORS[arm], alpha=.84, zorder=3)
            mark_fits(cost_ax, i, values, COLORS[arm], width=.18)
            cost_ax.text(i, max(values) * 1.03, f'{np.mean(values):.1f}s', ha='center', va='bottom',
                         fontsize=9, color='#364354')
        cost_ax.set(xticks=np.arange(4), xticklabels=['Current-only', 'Recurrent', '+ reward', '+ full'],
                    ylabel='Measured training seconds per fit',
                    ylim=(0, max(r['training_seconds'] for r in training.values()) * 1.25))
        cost_ax.set_title('C. Training cost at equal interactions', loc='left', pad=11, fontweight='bold')
        for ax in (success_ax, reset_ax, cost_ax):
            format_axis(ax)
        per_fit = p['environments'] * p['rollout'] * p['updates']
        fig.text(.075, .073,
                 f"Bars: fit means. Circles: individual training fits. Random: one fixed evaluation set. "
                 f"{p['evaluation_episodes']} episodes per fit and size.",
                 fontsize=9, color='#5c6878')
        fig.text(.075, .048,
                 f"{per_fit:,} interactions per fit; {p['max_steps']}-step cap. Reset clears hidden state at every step. "
                 'Predictive heads train the policy; no imagined planning.', fontsize=9, color='#5c6878')
        status = 'met' if summary['continuation_passed'] else 'not met'
        fig.text(.075, .023, f'Predeclared continuation criteria: {status}. No independent confirmation or novelty claim.',
                 fontsize=9, color='#364354', fontweight='bold')
        return fig


def learning_figure(summary, interactions, curves):
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        fig.subplots_adjust(left=.09, right=.98, top=.84, bottom=.16, hspace=.27, wspace=.12)
        fig.suptitle('Training trajectories: MiniGrid Memory', x=.09, y=.977, ha='left',
                     fontsize=18, fontweight='bold', color='#233044')
        fig.text(.09, .928, 'Three training fits per arm  |  Development diagnostics  |  No checkpoint selection',
                 fontsize=11, color='#5c6878')
        for arm, ax in zip(ARMS, axes.flat, strict=True):
            array = np.array([curves[(arm, seed)] for seed in summary['protocol']['seeds']])
            for row, seed, linestyle in zip(array, summary['protocol']['seeds'], ['-', '--', ':'], strict=True):
                ax.plot(interactions / 1000, row, color=COLORS[arm], alpha=.58,
                        linewidth=1.15, linestyle=linestyle, label=f'Seed {seed}')
            # Require all fits at a checkpoint; never average a subset of available seeds.
            mean = np.mean(array, axis=0)
            ax.plot(interactions / 1000, mean, color='#233044', linewidth=2.1, label='Fit mean')
            ax.set_title(LABELS[arm], loc='left', fontweight='bold', color=COLORS[arm])
            ax.set(ylim=(-2, 103), yticks=np.arange(0, 101, 25), xlim=(0, interactions[-1] / 1000))
            format_axis(ax)
        axes[0, 0].legend(loc='lower left', frameon=True, framealpha=.86, fontsize=8,
                           facecolor='white', edgecolor='#e1e6ec', ncol=2)
        for ax in axes[:, 0]:
            ax.set_ylabel('Recent training success (%)')
        for ax in axes[-1]:
            ax.set_xlabel('Environment interactions (thousands)')
        fig.text(.09, .086, 'Success over the most recent 100 completed training episodes, not held-out evaluation.',
                 fontsize=10, color='#364354')
        fig.text(.09, .055,
                 'Thin lines: individual fits. Thick line: arithmetic mean at the same interaction count. '
                 'Episode windows differ across fits.', fontsize=9, color='#5c6878')
        fig.text(.09, .028, 'No additional smoothing. Missing values remain missing; only final snapshots are evaluated.',
                 fontsize=9, color='#5c6878')
        return fig


def publish(args):
    source, plan_path, summary, training, scores = load_completed_report(args.report)
    curves = load_learning_curves(args.run, summary, training) if args.run is not None else None
    args.out.mkdir(parents=True, exist_ok=True)
    expected_files = FILENAMES + (CURVE_FILENAMES if curves is not None else ())
    if any((args.out / filename).exists() for filename in expected_files):
        raise FileExistsError('Figure export already exists; use a fresh output directory')
    figures = [('world-model-results', figure(summary, training, scores))]
    if curves is not None:
        figures.append(('world-model-learning', learning_figure(summary, curves[0], curves[1])))
    rendered = {}
    try:
        for name, fig in figures:
            for suffix in ('png', 'svg'):
                buffer = io.BytesIO()
                fig.savefig(buffer, format=suffix, dpi=180, bbox_inches='tight', pad_inches=.18)
                rendered[f'{name}.{suffix}'] = buffer.getvalue()
    finally:
        for _, fig in figures:
            plt.close(fig)
    for filename, contents in rendered.items():
        with (args.out / filename).open('xb') as stream:
            stream.write(contents)
    receipt = {'summary_sha256': sha(source), 'plan_sha256': sha(plan_path),
               'publisher_sha256': sha(Path(__file__)),
               'figures': {name: hashlib.sha256(contents).hexdigest() for name, contents in rendered.items()},
               'matplotlib_version': matplotlib.__version__, 'numpy_version': np.__version__,
               'training_fits': len(training), 'training_seeds': summary['protocol']['seeds'],
               'evaluation_sizes': summary['protocol']['eval_sizes'],
               'scope': 'Development pilot; all fits shown; no uncertainty intervals or novelty claim'}
    if curves is not None:
        receipt['learning_sources_sha256'] = curves[2]
        receipt['learning_scope'] = 'Rolling last 100 completed training episodes; not held-out evaluation'
    with (args.out / FILENAMES[-1]).open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True,
                        help='Completed summary.json, or directory containing summary.json and plan.json')
    parser.add_argument('--out', type=Path, required=True,
                        help='Output directory; may contain the source report, but not existing figure files')
    parser.add_argument('--run', type=Path,
                        help='Optional execution directory with preserved fits/*/learning.jsonl for training curves')
    publish(parser.parse_args())
