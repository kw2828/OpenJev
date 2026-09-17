"""Plot completed forced-path cue-retention diagnostics without loading models."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import publish_recurrent_world as base
from matplotlib.lines import Line2D

VERSION = 'cue-retention-probe-v1'
METRICS = ('hidden_l2_distance', 'hidden_rms_distance', 'latent_rms_normalized_distance',
           'policy_probability_l1_distance', 'argmax_disagreement', 'cue_visible_fraction',
           'observations_identical', 'native_forward_probability', 'swapped_forward_probability')
FIELDS = ('latent_rms_normalized_distance', 'policy_probability_l1_distance')
STEM = 'cue-retention'
LINTHRESH = 1e-8


def digest(value):
    return hashlib.sha256(value).hexdigest()


def check_hash(value, description):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'Invalid {description} hash')


def metric_means(rows):
    return {name: float(np.mean([row[name] for row in rows])) for name in METRICS}


def check_means(actual, expected, description):
    if set(actual) != set(expected):
        raise ValueError(f'Wrong fields in {description}')
    for key, value in expected.items():
        base.close(actual[key], value, f'{description} {key}')


def load_completed(input_path, plan_path):
    report = json.loads(input_path.read_text())
    plan = json.loads(plan_path.read_text())
    p = plan['protocol']
    if (p['version'] != VERSION or p['study'] != 'cue-memory-v1' or report['status'] != 'completed'
            or report['probe_plan_sha256'] != base.sha(plan_path)):
        raise ValueError('Expected a completed cue-retention probe matching the frozen plan')
    if any(report.get(key) != value for key, value in plan.items()):
        raise ValueError('Completed probe protocol, sources or runtime differ from the frozen plan')
    if (tuple(p['arms']) != base.ARMS or p['training_seeds'] != [61, 73, 89]
            or p['sizes'] != [11, 17, 23] or p['pairs_per_size'] != 16
            or tuple(p['metrics']) != METRICS[:6] or p['new_training'] is not False
            or p['independent_confirmation'] is not False):
        raise ValueError('Unexpected frozen condition coverage or claim boundary')
    for key in ('study_plan_sha256', 'outer_completed_sha256', 'core_completed_sha256', 'progress_sha256'):
        check_hash(report[key], key)
    for value in plan['source_sha256'].values():
        check_hash(value, 'probe source')
    if (report['fits'] != 12 or report['paired_trajectories'] != 36 * p['pairs_per_size']
            or not np.isfinite(report['elapsed_seconds']) or report['elapsed_seconds'] <= 0):
        raise ValueError('Invalid completion counts or elapsed time')
    expected = {(arm, seed, size) for arm in p['arms'] for seed in p['training_seeds'] for size in p['sizes']}
    if len(report['results']) != len(expected):
        raise ValueError('Expected all 36 final-checkpoint and map-size conditions')
    observed, records, checkpoint_hashes, observation_hashes = set(), {}, {}, {}
    for record in report['results']:
        arm, seed, size = record['arm'], record['training_seed'], record['size']
        identity = (arm, seed, size)
        if identity not in expected or identity in observed:
            raise ValueError('Unexpected or duplicated probe condition')
        observed.add(identity)
        check_hash(record['checkpoint_sha256'], 'checkpoint')
        check_hash(record['paired_observations_sha256'], 'paired observations')
        checkpoint_hashes.setdefault((arm, seed), set()).add(record['checkpoint_sha256'])
        observation_hashes.setdefault(size, set()).add(record['paired_observations_sha256'])
        first = p['seed_base'] + size * p['size_seed_multiplier']
        seeds = list(range(first, first + p['pairs_per_size']))
        if (record['pairs'] != len(seeds) or record['seeds'] != seeds
                or record['forced_actions_per_pair_member'] != [2] * (size - 3)
                or record['all_post_initial_observations_identical'] is not True
                or record['current_only_equality_check_passed'] is not (True if arm == 'current_ppo' else None)):
            raise ValueError('Probe seeds, forced actions or input-equality audit mismatch')
        pairs = record['per_pair']
        if len(pairs) != len(seeds):
            raise ValueError('Missing per-pair probe data')
        for pair_seed, rows in zip(seeds, pairs, strict=True):
            if len(rows) != size - 2:
                raise ValueError('Probe trajectory must include the initial cue through the fork')
            for step, row in enumerate(rows):
                if row['seed'] != pair_seed or row['step'] != step or row['x'] != step + 1:
                    raise ValueError('Per-pair probe sequence identity mismatch')
                if (type(row['observations_identical']) is not bool
                        or row['observations_identical'] != (step > 0)
                        or row['cue_visible_fraction'] != (1. if step == 0 else 0.)):
                    raise ValueError('Cue visibility or post-initial observation equality mismatch')
                if (type(row['argmax_disagreement']) is not bool
                        or any(type(row[key]) is not int or not 0 <= row[key] < 7
                               for key in ('native_argmax', 'swapped_argmax'))
                        or row['argmax_disagreement'] != (row['native_argmax'] != row['swapped_argmax'])):
                    raise ValueError('Action argmax diagnostic mismatch')
                for field in METRICS[:4]:
                    if not np.isfinite(row[field]) or row[field] < 0:
                        raise ValueError('Invalid hidden-state or action-probability distance')
                for field in FIELDS:
                    if row[field] > 2 + 1e-6:
                        raise ValueError('Normalized distance or probability L1 exceeds its mathematical bound')
                for field in ('native_forward_probability', 'swapped_forward_probability'):
                    if not np.isfinite(row[field]) or not 0 <= row[field] <= 1:
                        raise ValueError('Invalid recorded forward-action probability')
                if (arm == 'current_ppo' and step > 0
                        and (any(row[field] != 0 for field in METRICS[:5])
                             or row['native_forward_probability'] != row['swapped_forward_probability'])):
                    raise ValueError('Current-only equality control must remain exactly zero after the cue')
        steps = record['per_step']
        if len(steps) != size - 2 or [row['step'] for row in steps] != list(range(size - 2)):
            raise ValueError('Missing or reordered step aggregates')
        for step, row in enumerate(steps):
            check_means({k: v for k, v in row.items() if k != 'step'},
                        metric_means([pair[step] for pair in pairs]), f'step {step}')
        check_means(record['initial'], metric_means([pair[0] for pair in pairs]), 'initial cue')
        check_means(record['fork'], metric_means([pair[-1] for pair in pairs]), 'fork')
        check_means(record['last_cue_visible'], metric_means([pair[0] for pair in pairs]), 'last visible cue')
        if record['last_cue_visible_steps'] != [0] * len(pairs):
            raise ValueError('Cue must be visible only at the initial step')
        ratios = [pair[-1]['hidden_rms_distance'] / pair[0]['hidden_rms_distance']
                  for pair in pairs if pair[0]['hidden_rms_distance'] > 0]
        if record['pairs_with_nonzero_initial_state_difference'] != len(ratios):
            raise ValueError('Initial nonzero-state pair count mismatch')
        if ratios:
            base.close(record['mean_fork_to_initial_hidden_rms_ratio'], np.mean(ratios), 'retention ratio')
        elif record['mean_fork_to_initial_hidden_rms_ratio'] is not None:
            raise ValueError('Undefined retention ratio must remain null')
        records[identity] = record
    if observed != expected or any(len(v) != 1 for v in checkpoint_hashes.values()):
        raise ValueError('Probe conditions or checkpoint identity coverage mismatch')
    if any(len(v) != 1 for v in observation_hashes.values()):
        raise ValueError('Paired observation streams differ between models at the same map size')
    return report, plan, records


def figure(plan, records):
    p = plan['protocol']
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9), sharey='row')
    fig.subplots_adjust(left=.08, right=.98, top=.76, bottom=.18, hspace=.26, wspace=.19)
    fig.suptitle('Forced-path cue retention: a diagnostic, not gameplay', x=.08, y=.975,
                 ha='left', fontsize=18, fontweight='bold', color='#233044')
    fig.text(.08, .925, 'Cue identity differs at step 0; all later observations and imposed forward actions match.',
             fontsize=11, color='#5c6878')
    fig.legend(handles=[Line2D([0], [0], color=base.COLORS[arm], linewidth=2.5, label=base.LABELS[arm])
                        for arm in base.ARMS], loc='upper left', bbox_to_anchor=(.074, .893),
               ncol=4, frameon=False, fontsize=10, columnspacing=1.8)
    fig.text(.08, .828, 'Thin lines: all three training fits, averaged over 16 matched cue pairs each. '
             'Thick lines: arithmetic fit means.', fontsize=10, color='#5c6878')
    for column, size in enumerate(p['sizes']):
        steps = np.arange(size - 2)
        for row, field in enumerate(FIELDS):
            ax = axes[row, column]
            ax.axvspan(-.25, .5, color='#f5e7b7', alpha=.65, zorder=0)
            ax.axvline(.5, color='#a99155', linestyle=':', linewidth=.8, zorder=1)
            for arm in base.ARMS:
                values = np.array([[item[field] for item in records[(arm, seed, size)]['per_step']]
                                   for seed in p['training_seeds']])
                for fit, linestyle in zip(values, ['-', '--', ':'], strict=True):
                    ax.plot(steps, fit, color=base.COLORS[arm], linewidth=.95, alpha=.48,
                            linestyle=linestyle, zorder=2)
                ax.plot(steps, np.mean(values, axis=0), color=base.COLORS[arm], linewidth=2.2,
                        alpha=.95, zorder=4 if arm == 'current_ppo' else 3)
            ax.set_yscale('symlog', linthresh=LINTHRESH, linscale=.6)
            ax.set(xlim=(-.25, size - 3 + .25), ylim=(-.25 * LINTHRESH, 2.3),
                   xticks=np.arange(0, size - 2, 2),
                   yticks=[0., 1e-8, 1e-6, 1e-4, 1e-2, 1.],
                   yticklabels=['0', '$10^{-8}$', '$10^{-6}$', '$10^{-4}$', '$10^{-2}$', '1'])
            base.format_axis(ax)
            if row == 0:
                ax.set_title(f'{size} × {size} map; fork at step {size - 3}',
                             loc='left', fontweight='bold', fontsize=11, pad=12)
            else:
                ax.set_xlabel('Forced forward steps since initial cue')
    axes[0, 0].set_ylabel('Normalized hidden-state RMS difference')
    axes[1, 0].set_ylabel('Action-probability L1 difference')
    fig.text(.08, .116, 'Shaded region: initial visible cue. Steps 1+: identical observations. '
             'Current-only control must be exactly zero after step 0.', fontsize=9, color='#5c6878')
    fig.text(.08, .084, 'Symmetric-log axes are linear below 1e-8; zero and small fits are retained. '
             'Normalized RMS divides state difference by pooled state RMS.', fontsize=9, color='#5c6878')
    fig.text(.08, .052, 'Forced actions bypass the learned policy. State differences do not establish usable '
             'or decodable memory, gameplay success, or novelty.', fontsize=9, color='#364354', fontweight='bold')
    return fig


def publish(args):
    report, plan, records = load_completed(args.input, args.plan)
    names = [f'{STEM}.png', f'{STEM}.svg', f'{STEM}-figures.json']
    if any((args.out / name).exists() for name in names):
        raise FileExistsError('Retention figure export already exists; use a fresh output directory')
    style = {'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
             'svg.hashsalt': VERSION, 'savefig.facecolor': 'white'}
    rendered = {}
    with plt.rc_context(style):
        fig = figure(plan, records)
        try:
            for suffix in ('png', 'svg'):
                stream = io.BytesIO()
                metadata = {'Date': None, 'Creator': 'OpenJev cue-retention diagnostic'} if suffix == 'svg' else {}
                fig.savefig(stream, format=suffix, dpi=180, bbox_inches='tight', pad_inches=.18, metadata=metadata)
                data = stream.getvalue()
                if suffix == 'svg':
                    data = ('\n'.join(line.rstrip() for line in data.decode().splitlines()) + '\n').encode()
                rendered[f'{STEM}.{suffix}'] = data
        finally:
            plt.close(fig)
    receipt = {
        'protocol_version': VERSION, 'input_sha256': base.sha(args.input), 'plan_sha256': base.sha(args.plan),
        'publisher_sha256': base.sha(Path(__file__)), 'plot_helper_sha256': base.sha(Path(base.__file__)),
        'probe_sources_sha256': plan['source_sha256'],
        'study_plan_sha256': report['study_plan_sha256'],
        'figures': {name: digest(data) for name, data in rendered.items()},
        'conditions': len(records), 'training_fits': report['fits'],
        'paired_trajectories': report['paired_trajectories'],
        'aggregation': 'Mean across all 16 pairs per step per fit; arithmetic mean of all three fits per arm',
        'zero_handling': 'No drops or positive floor; symmetric-log axes linear below 1e-8',
        'scope': 'Forced-path diagnostic of cue-dependent state and output sensitivity, not gameplay or decoded memory',
        'new_training': False, 'independent_confirmation': False, 'novelty_established': False,
        'matplotlib_version': matplotlib.__version__, 'numpy_version': np.__version__,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in rendered.items():
        with (args.out / name).open('xb') as stream:
            stream.write(data)
    with (args.out / names[-1]).open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Completed retention.json probe result')
    parser.add_argument('--plan', type=Path, required=True, help='Frozen retention-plan.json')
    parser.add_argument('--out', type=Path, required=True, help='Fresh plot export directory')
    publish(parser.parse_args())
