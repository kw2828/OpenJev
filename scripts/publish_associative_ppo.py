"""Audit and visualize a completed autonomous associative-memory PPO study.

Requires the frozen study report and all execution receipts before publishing.
Checks complete budgets, learning logs, checkpoint hashes, every episode and its
aggregates, paired interventions and the original continuation rule. This is a
receipt/aggregate audit, not a fresh replay of the scored policies. No training.
"""

import argparse
import hashlib
import importlib.util
import io
import json
import random
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
ARMS = ['gru', 'feedforward', 'fast_global', 'fast_selective']
SEEDS = [101, 113, 127]
SIZES = [11, 17, 23]
COLORS = dict(zip(ARMS, ['#53708f', '#cf8642', '#178b87', '#7856b6'], strict=True))
LABELS = dict(zip(ARMS, ['GRU', 'GRU + feedforward adapter', 'Global-write memory',
                        'Selective-write memory'], strict=True))
METRICS = ['success', 'wrong_goal', 'timeout', 'mean_return', 'mean_length']
TIMING_SCOPE = 'Model+argmax for full batch; includes background reset slots; telemetry excluded'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, label, atol=1e-10):
    if (isinstance(actual, bool) or not np.isfinite(actual) or not np.isfinite(expected)
            or abs(actual-expected) > atol):
        raise ValueError(f'{label}: {actual!r} does not agree with {expected!r}')


def same_tree(actual, expected, label):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f'{label}: mapping coverage changed')
        for key in expected:
            same_tree(actual[key], expected[key], f'{label}.{key}')
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f'{label}: sequence coverage changed')
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            same_tree(left, right, f'{label}[{index}]')
    elif isinstance(expected, float):
        close(actual, expected, label)
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f'{label}: value changed')


def check_nonnegative(value, label):
    if isinstance(value, bool) or not np.isfinite(value) or value < 0:
        raise ValueError(f'{label}: expected a finite nonnegative value')


def validate_training(execution, row, protocol, model):
    directory = execution/'fits'/f"{row['arm']}-{row['seed']}"
    if row != read(directory/'completed.json') or row['status'] != 'completed':
        raise ValueError('Training completion receipts disagree')
    start = read(directory/'started.json')
    expected_start = {'arm': row['arm'], 'seed': row['seed'], 'protocol': protocol['version'],
                      'updates': protocol['updates']}
    if any(start[key] != value for key, value in expected_start.items()):
        raise ValueError('Training start identity or budget differs')
    check_nonnegative(start['started_unix'], 'Fit start time')
    interactions = protocol['updates']*protocol['environments']*protocol['rollout']
    gradient_steps = protocol['updates']*protocol['epochs']*protocol['environments']//protocol['minibatch_envs']
    if row['interactions'] != interactions or row['gradient_steps'] != gradient_steps:
        raise ValueError('PPO training interaction or optimizer budget changed')
    for filename, key in [('model.pt', 'checkpoint_sha256'), ('learning.jsonl', 'learning_sha256')]:
        if sha(directory/filename) != row[key]:
            raise ValueError(f'Hashed training artifact changed: {filename}')
    logs = [json.loads(line) for line in (directory/'learning.jsonl').read_text().splitlines()]
    if len(logs) != protocol['updates'] or [r['update'] for r in logs] != list(range(1, protocol['updates']+1)):
        raise ValueError('Missing or duplicated learning updates')
    for entry in logs:
        if (entry['interactions'] != entry['update']*protocol['environments']*protocol['rollout']
                or type(entry['completed_episodes']) is not int or entry['completed_episodes'] < 0
                or entry['completed_episodes'] > entry['interactions']):
            raise ValueError('Learning-log interaction or episode count invalid')
        check_nonnegative(entry['elapsed_seconds'], 'Learning-log elapsed time')
        losses = np.asarray(entry['mean_losses'])
        if losses.shape != (4,) or not np.isfinite(losses).all() or losses[1] < 0 or losses[3] < 0:
            raise ValueError('Expected finite total/entropy/actor/value training losses')
        if entry['completed_episodes'] == 0:
            if entry['recent_success'] is not None or entry['recent_return'] is not None:
                raise ValueError('Training success reported without completed episodes')
        else:
            for key in ('recent_success', 'recent_return'):
                check_nonnegative(entry[key], key)
                if entry[key] > 1:
                    raise ValueError('Training recent success/return exceeds native bounds')
    if any(a['elapsed_seconds'] > b['elapsed_seconds'] or a['completed_episodes'] > b['completed_episodes']
           for a, b in pairwise(logs)):
        raise ValueError('Learning-log cumulative time or episode count decreased')
    for key in ('training_seconds', 'collection_seconds', 'optimization_seconds'):
        check_nonnegative(row[key], key)
    if (row['training_seconds'] < logs[-1]['elapsed_seconds']
            or row['training_seconds'] < row['collection_seconds']+row['optimization_seconds']
            or row['completed_episodes'] != logs[-1]['completed_episodes']):
        raise ValueError('Full training cost or episode count differs from its logs')
    parameters = dict(model.named_parameters())
    active = sorted(name for name, value in parameters.items() if value.requires_grad)
    expected = {'registered_parameters': sum(value.numel() for value in parameters.values()),
                'trainable_parameters': sum(value.numel() for value in parameters.values() if value.requires_grad),
                'state_values_per_episode': model.state_size,
                'gradient_parameter_names': active,
                'gradient_parameters': sum(parameters[name].numel() for name in active)}
    if any(row[key] != value for key, value in expected.items()):
        raise ValueError('Model parameter, active-gradient or recurrent-state counts changed')
    weights = torch.load(directory/'model.pt', map_location='cpu', weights_only=True)
    if any(not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError('Nonfinite trained checkpoint')
    model.load_state_dict(weights, strict=True)
    return logs


def validate_result(result, arm, protocol, model=None):
    """Validate native episode accounting without executing scored episodes."""
    size, rows = result['size'], result['episodes']
    if size not in protocol['eval_sizes']:
        raise ValueError('Unexpected evaluation grid size')
    start = protocol['evaluation_seed_start'] + size*10000
    count = protocol['evaluation_episodes']
    if [row['seed'] for row in rows] != list(range(start, start+count)):
        raise ValueError('Evaluation seed coverage or ordering changed')
    for row in rows:
        if (any(type(row[key]) is not bool for key in ('success', 'wrong_goal', 'timeout'))
                or sum(row[key] for key in ('success', 'wrong_goal', 'timeout')) != 1):
            raise ValueError('Native episode outcomes do not partition the assigned examples')
        if (type(row['length']) is not int or not 1 <= row['length'] <= protocol['max_steps']
                or len(row['action_counts']) != 7
                or any(type(value) is not int or value < 0 for value in row['action_counts'])
                or sum(row['action_counts']) != row['length']
                or (row['timeout'] and row['length'] != protocol['max_steps'])):
            raise ValueError('Episode action counts, length or timeout invalid')
        expected_return = 1-.9*row['length']/protocol['max_steps'] if row['success'] else 0.
        close(row['return'], expected_return, 'Native time-sensitive return')
    aggregate = {}
    for metric, field in [('success', 'success'), ('wrong_goal', 'wrong_goal'), ('timeout', 'timeout'),
                          ('mean_return', 'return'), ('mean_length', 'length')]:
        aggregate[metric] = float(np.mean([row[field] for row in rows]))
        close(result[metric], aggregate[metric], f'Episode aggregate {metric}')
    for key in ('model_forward_seconds', 'model_batch_p50_ms', 'model_batch_p95_ms', 'wall_seconds'):
        check_nonnegative(result[key], key)
    if (result['model_forward_seconds'] > result['wall_seconds']
            or result['model_batch_p50_ms'] > result['model_batch_p95_ms']
            or result['timing_scope'] != TIMING_SCOPE):
        raise ValueError('Evaluation timing scope or cost invalid')
    batch = protocol['environments']
    expected_steps = sum(len(rows[i:i+batch])*max(row['length'] for row in rows[i:i+batch])
                         for i in range(0, count, batch))
    if type(result['executed_environment_steps']) is not int or result['executed_environment_steps'] != expected_steps:
        raise ValueError('Batched compute does not account for every background reset slot')
    if arm in ('fast_global', 'fast_selective'):
        telemetry = result['write_audit']
        if not isinstance(telemetry, dict) or set(telemetry) != {'cue_visible', 'cue_hidden'}:
            raise ValueError('Missing active-step write audit')
        if sum(bucket['steps'] for bucket in telemetry.values()) != sum(row['length'] for row in rows):
            raise ValueError('Write audit does not cover every assigned active step')
        for name, bucket in telemetry.items():
            if set(bucket) != {'steps', 'beta_sum', 'update_norm_sum'} or type(bucket['steps']) is not int:
                raise ValueError('Unexpected write-audit fields')
            for key, value in bucket.items():
                check_nonnegative(value, f'{name}.{key}')
            if bucket['beta_sum'] > bucket['steps'] or (bucket['steps'] == 0 and bucket['update_norm_sum'] != 0):
                raise ValueError('Write strengths or updates inconsistent with active steps')
            if arm == 'fast_global' and model is not None:
                beta = float(model.write_gate.bias.sigmoid().detach().item())
                close(bucket['beta_sum'], beta*bucket['steps'], 'Global write-strength sum', 1e-7)
    elif result['write_audit'] is not None:
        raise ValueError('Control without matrix memory has write telemetry')
    return aggregate


def recompute_gates(protocol, averages, grouped):
    rule = protocol['continuation']
    arm, size = rule['candidate'], str(protocol['train_size'])
    checks = {
        'same_size_success': averages[arm]['intact'][size]['success'] >= rule['same_size_success_minimum'],
        'each_fit_same_size_success': all(row['success'] >= rule['each_fit_same_size_success_minimum']
                                         for row in grouped[arm]['intact'][size]),
        'store_reset_effect': (averages[arm]['intact'][size]['success']-averages[arm]['reset_store'][size]['success']
                               >= rule['same_size_store_reset_drop_minimum']),
    }
    for control in rule['controls']:
        for grid_size in protocol['eval_sizes']:
            grid = str(grid_size)
            checks[f'gain_vs_{control}_size_{grid}'] = (
                averages[arm]['intact'][grid]['success']-averages[control]['intact'][grid]['success']
                >= rule['minimum_gain_each_size_vs_each_control'])
    return checks


def verify(report, execution, plan_path):
    # Check both terminal receipts first. No partial or smoke run can be plotted.
    completed, executed = read(report/'completed.json'), read(execution/'completed.json')
    summary, plan = read(report/'summary.json'), read(plan_path)
    if (completed['status'] != 'completed' or executed['status'] != 'completed'
            or completed['plan_sha256'] != sha(plan_path) or executed['plan_sha256'] != sha(plan_path)
            or completed['summary_sha256'] != sha(report/'summary.json')
            or executed['fits'] != 12 or executed['evaluations'] != 55 or read(report/'plan.json') != plan
            or summary['plan_sha256'] != sha(plan_path) or summary['protocol'] != plan['protocol']):
        raise ValueError('Report, execution or frozen-plan terminal identity mismatch')
    spec = importlib.util.spec_from_file_location('_associative_ppo_publisher_frozen',
                                                  ROOT/'scripts/associative_ppo_study.py')
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    if plan != study.signature() or plan['protocol']['version'] != 'associative-ppo-v1':
        raise ValueError('Frozen protocol, source, environment or dependency signature changed')
    protocol = plan['protocol']
    if (summary['novelty_established'] is not False or summary['independent_confirmation'] is not False
            or protocol['supervised_warm_start'] is not False or protocol['auxiliary_losses'] is not False):
        raise ValueError('Incorrect training or claim scope')
    start = read(execution/'started.json')
    if start['plan_sha256'] != sha(plan_path) or executed['finished_unix'] < start['started_unix']:
        raise ValueError('Execution start/end mismatch')
    training = read(execution/'training-completed.json')
    if training['status'] != 'completed' or training['fits'] != summary['training']:
        raise ValueError('Training completion receipt differs from report')
    expected_fits = {(arm, seed) for arm in ARMS for seed in SEEDS}
    fits = {(row['arm'], row['seed']): row for row in training['fits']}
    if len(training['fits']) != 12 or set(fits) != expected_fits:
        raise ValueError('Training fits incomplete or duplicated')
    order = [(arm, seed) for arm in ARMS for seed in SEEDS]
    random.Random(protocol['fit_order_seed']).shuffle(order)
    if (read(execution/'fit_order.json') != [list(identity) for identity in order]
            or [(row['arm'], row['seed']) for row in training['fits']] != order):
        raise ValueError('Training fit order changed')
    torch.set_num_threads(1)
    models, logs = {}, {}
    for identity, row in fits.items():
        arm, seed = identity
        model = study.AssociativePolicy(arm, study.OBS_SIZE, protocol['hidden'])
        logs[identity] = validate_training(execution, row, protocol, model)
        models[identity] = model
    close(summary['training_seconds'], sum(row['training_seconds'] for row in fits.values()), 'Total training cost')
    if (summary['interactions'] != sum(row['interactions'] for row in fits.values())
            or summary['interactions'] != 12582912 or sum(len(rows) for rows in logs.values()) != 12288
            or sum(row['gradient_steps'] for row in fits.values()) != 49152):
        raise ValueError('Full twelve-fit training budget differs')
    expected = {(arm, seed, mode) for arm, seed in expected_fits for mode in study.modes(arm)} | {('random', None, 'intact')}
    paths = sorted((execution/'evaluation').glob('*.json'))
    records = [read(path) for path in paths]
    if (len(records) != 55 or records != summary['evaluation']
            or {(row['arm'], row.get('seed'), row['mode']) for row in records} != expected):
        raise ValueError('Evaluation file/report identities incomplete or duplicated')
    by_identity, grouped, results, episode_count, write_result_count = {}, {}, {}, 0, 0
    for record in records:
        arm, seed, mode = record['arm'], record.get('seed'), record['mode']
        by_identity[(arm, seed, mode)] = record
        if arm != 'random' and record['checkpoint_sha256'] != fits[(arm, seed)]['checkpoint_sha256']:
            raise ValueError('Evaluation checkpoint identity differs from completed fit')
        if len(record['results']) != 3 or {row['size'] for row in record['results']} != set(SIZES):
            raise ValueError('Missing or duplicated evaluation grid sizes')
        for row in record['results']:
            size = row['size']
            aggregate = validate_result(row, arm, protocol, models.get((arm, seed)))
            results[(arm, seed, mode, size)] = aggregate
            grouped.setdefault(arm, {}).setdefault(mode, {}).setdefault(str(size), []).append(aggregate)
            episode_count += len(row['episodes'])
            write_result_count += int(row['write_audit'] is not None)
    if len(results) != 165 or episode_count != 21120 or write_result_count != 90:
        raise ValueError('Unexpected final episode, size or write-audit coverage')
    averages = {arm: {mode: {size: {key: float(np.mean([row[key] for row in rows])) for key in METRICS}
                             for size, rows in sizes.items()} for mode, sizes in modes.items()}
                for arm, modes in grouped.items()}
    same_tree(summary['per_fit'], grouped, 'Per-fit evaluation aggregates')
    same_tree(summary['averages'], averages, 'Mean evaluation aggregates')
    paired = []
    for arm, seed in sorted(expected_fits):
        intact = by_identity[(arm, seed, 'intact')]
        for mode in study.modes(arm)[1:]:
            comparison = {row['size']: row for row in by_identity[(arm, seed, mode)]['results']}
            for before in intact['results']:
                after = comparison[before['size']]
                pairs = list(zip(before['episodes'], after['episodes'], strict=True))
                if any(a['seed'] != b['seed'] for a, b in pairs):
                    raise ValueError('Intervention pairs do not share environment seeds')
                both = sum(not a['timeout'] and not b['timeout'] for a, b in pairs)
                reversed_branch = sum(not a['timeout'] and not b['timeout'] and a['success'] != b['success']
                                      for a, b in pairs)
                paired.append({'arm': arm, 'seed': seed, 'mode': mode, 'size': before['size'],
                               'assigned_pairs': len(pairs), 'both_reach_branch': both,
                               'reversed_branch': reversed_branch,
                               'success_difference': before['success']-after['success']})
    if len(paired) != 126:
        raise ValueError('Paired intervention coverage incomplete')
    same_tree(summary['paired_interventions'], paired, 'Paired interventions')
    checks = recompute_gates(protocol, averages, grouped)
    if checks != summary['checks'] or summary['continuation_passed'] is not all(checks.values()):
        raise ValueError('Predeclared continuation rule does not reproduce')
    file_hashes = {str(path.relative_to(execution)): sha(path) for path in paths}
    return summary, results, logs, file_hashes


def style_axis(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#cdd4df')
    ax.tick_params(colors='#445266', labelsize=9)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)


def grid_axis(ax):
    ax.set(xticks=range(3), xticklabels=['11 (trained)', '17', '23'], xlabel='MiniGrid Memory grid size')
    style_axis(ax)


def success_panel(ax, results, mode='intact'):
    for i, arm in enumerate(ARMS):
        x = np.arange(3)+(i-1.5)*.075
        values = np.asarray([[100*results[(arm, seed, mode, size)]['success'] for size in SIZES] for seed in SEEDS])
        for offset, row in zip([-.022, 0, .022], values, strict=True):
            ax.scatter(x+offset, row, s=27, color=COLORS[arm], zorder=4)
        ax.plot(x, values.mean(0), color=COLORS[arm], linewidth=2)
    if mode == 'intact':
        ax.plot(range(3), [100*results[('random', None, 'intact', size)]['success'] for size in SIZES],
                linestyle=':', color='#8793a1', linewidth=1.5)
    ax.set(ylabel='Successful episodes (%)', ylim=(-4, 106), yticks=[0, 25, 50, 75, 100])
    grid_axis(ax)


def effect_panel(ax, results, mode, arms):
    for i, arm in enumerate(arms):
        x = np.arange(3)+(i-(len(arms)-1)/2)*.075
        values = np.asarray([[100*(results[(arm, seed, 'intact', size)]['success']
                                   - results[(arm, seed, mode, size)]['success']) for size in SIZES] for seed in SEEDS])
        for offset, row in zip([-.022, 0, .022], values, strict=True):
            ax.scatter(x+offset, row, s=27, color=COLORS[arm], zorder=4)
        ax.plot(x, values.mean(0), color=COLORS[arm], linewidth=2)
    ax.axhline(0, linestyle=':', color='#8793a1', linewidth=1)
    ax.set(ylabel='Intact minus intervention (percentage points)', ylim=(-105, 105), yticks=[-100, -50, 0, 50, 100])
    grid_axis(ax)


def evaluation_figure(summary, results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.7))
    fig.subplots_adjust(left=.075, right=.98, top=.78, bottom=.20, hspace=.43, wspace=.23)
    fig.suptitle('Autonomous PPO: associative-memory gameplay and interventions', x=.075, y=.975,
                 ha='left', fontsize=18, fontweight='bold', color='#233044')
    fig.text(.075, .933, '12 final fits | 1,048,576 training steps each | 128 paired evaluation seeds per size and condition',
             fontsize=10.8, color='#5c6878')
    fig.legend([Line2D([0], [0], color=COLORS[arm], linewidth=2) for arm in ARMS],
               [LABELS[arm] for arm in ARMS], loc='upper left', bbox_to_anchor=(.068, .906),
               ncol=2, frameon=False, fontsize=10.5)
    success_panel(axes[0, 0], results)
    axes[0, 0].set_title('Final greedy policy: cue-visible start', loc='left', fontweight='bold', pad=12)
    for ax, mode, title, arms in [
            (axes[0, 1], 'reset_all', 'Erase all recurrent state at every observation', ARMS),
            (axes[1, 0], 'reset_store', 'Erase associative store at every observation', ARMS[2:]),
            (axes[1, 1], 'cue_swapped', 'Swap the initial cue; keep reward goals fixed', ARMS)]:
        effect_panel(ax, results, mode, arms)
        ax.set_title(title, loc='left', fontweight='bold', pad=12)
    failed = [key for key, value in summary['checks'].items() if not value]
    status = 'PASSED' if summary['continuation_passed'] else f'FAILED ({len(failed)} of {len(summary["checks"])} checks)'
    fig.text(.075, .131, f'Predeclared development continuation gate: {status}. All final fits are included.',
             fontsize=11, fontweight='bold', color='#23785f' if not failed else '#8e3d35')
    fig.text(.075, .094, 'Points: individual fit seeds. Lines: means. Dotted success line: uniform random policy. '
             'Positive intervention effects mean lower success.', fontsize=9.3, color='#5c6878')
    fig.text(.075, .063, 'Native actions and rewards; the primary task starts with the cue visible. '
             'Counterfactual cue swaps intentionally conflict with stored goals.', fontsize=9.3, color='#5c6878')
    fig.text(.075, .033, 'One small structured environment and three training seeds. '
             'Development evidence about established memory mechanisms; no novelty or independent confirmation.',
             fontsize=9.3, color='#26364a')
    return fig


def training_figure(summary, results, logs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.7))
    fig.subplots_adjust(left=.075, right=.98, top=.78, bottom=.19, hspace=.47, wspace=.23)
    fig.suptitle('Full PPO training cost and native-start transfer', x=.075, y=.975,
                 ha='left', fontsize=19, fontweight='bold', color='#233044')
    total_hours = summary['training_seconds']/3600
    fig.text(.075, .933, f'12,582,912 training interactions | 49,152 optimizer steps | '
             f'{total_hours:.2f} total fit-hours on the recorded local runtime', fontsize=10.8, color='#5c6878')
    fig.legend([Line2D([0], [0], color=COLORS[arm], linewidth=2) for arm in ARMS],
               [LABELS[arm] for arm in ARMS], loc='upper left', bbox_to_anchor=(.068, .906),
               ncol=2, frameon=False, fontsize=10.5)
    for arm in ARMS:
        success = np.asarray([[np.nan if row['recent_success'] is None else row['recent_success']*100
                               for row in logs[(arm, seed)]] for seed in SEEDS])
        elapsed = np.asarray([[row['elapsed_seconds']/60 for row in logs[(arm, seed)]] for seed in SEEDS])
        interactions = np.asarray([row['interactions']/1e6 for row in logs[(arm, SEEDS[0])]])
        for i in range(3):
            axes[0, 0].plot(interactions, success[i], color=COLORS[arm], linewidth=.7, alpha=.3)
            axes[0, 1].plot(elapsed[i], success[i], color=COLORS[arm], linewidth=.7, alpha=.3)
        observed = np.isfinite(success).sum(axis=0)
        means = np.divide(np.nansum(success, axis=0), observed,
                          out=np.full(success.shape[1], np.nan), where=observed > 0)
        axes[0, 0].plot(interactions, means, color=COLORS[arm], linewidth=1.7)
        axes[0, 1].plot(elapsed.mean(0), means, color=COLORS[arm], linewidth=1.7)
    for ax, label, title in [(axes[0, 0], 'Training interactions (millions)', 'Training success by sample budget'),
                              (axes[0, 1], 'Elapsed training time (minutes)', 'Training success by elapsed cost')]:
        ax.set(xlabel=label, ylabel='Recent 100 training episodes: success (%)', ylim=(-4, 106))
        ax.set_title(title, loc='left', fontweight='bold', pad=12)
        style_axis(ax)
    fit_by_id = {(row['arm'], row['seed']): row for row in summary['training']}
    ax = axes[1, 0]
    for i, arm in enumerate(ARMS):
        for offset, seed in zip([-.22, 0, .22], SEEDS, strict=True):
            row = fit_by_id[(arm, seed)]
            collection, optimization = row['collection_seconds']/60, row['optimization_seconds']/60
            overhead = (row['training_seconds']-row['collection_seconds']-row['optimization_seconds'])/60
            ax.bar(i+offset, collection, width=.19, color=COLORS[arm], alpha=.45)
            ax.bar(i+offset, optimization, width=.19, bottom=collection, color=COLORS[arm])
            ax.bar(i+offset, overhead, width=.19, bottom=collection+optimization, color='#aeb7c2')
    ax.set(xticks=range(4), xticklabels=['GRU', 'GRU +\nadapter', 'Global\nwrite', 'Selective\nwrite'],
           ylabel='Recorded fit wall time (minutes)', ylim=(0, None))
    ax.set_title('All three fit costs per architecture', loc='left', fontweight='bold', pad=12)
    style_axis(ax)
    success_panel(axes[1, 1], results, 'native_start')
    axes[1, 1].set_title('Transfer to native random corridor starts', loc='left', fontweight='bold', pad=12)
    fig.text(.075, .121, 'Training curves: thin lines show each complete fit; thick lines average fits at the same update. '
             'These curves are not held-out evaluations.', fontsize=9.3, color='#5c6878')
    fig.text(.075, .089, 'Cost bars: light = rollout collection; solid = optimization; gray = setup / logging / other overhead. '
             'Bars run left to right: seeds 101, 113, 127.', fontsize=9.3, color='#5c6878')
    fig.text(.075, .057, 'All allocated updates and fits are counted. '
             'Recorded timers exclude model / optimizer initialization and evaluation; timings are specific to this machine.',
             fontsize=9.3, color='#26364a')
    fig.text(.075, .029, 'Native-start panel: greedy final policies on the same assigned evaluation seed IDs; '
             'the cue may require active information seeking.', fontsize=9.3, color='#26364a')
    return fig


def render(summary, results, logs):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.hashsalt': 'associative-ppo-v1'})
    content = {}
    for name, build in [('associative-ppo-evaluation', lambda: evaluation_figure(summary, results)),
                        ('associative-ppo-training', lambda: training_figure(summary, results, logs))]:
        figure = build()
        for suffix in ('png', 'svg'):
            buffer = io.BytesIO()
            metadata = {'Software': 'OpenJev verified PPO report'} if suffix == 'png' else {'Date': None}
            figure.savefig(buffer, format=suffix, dpi=170, facecolor='white', metadata=metadata)
            content[f'{name}.{suffix}'] = buffer.getvalue()
        plt.close(figure)
    return content


def publish(args):
    if args.out.exists():
        raise FileExistsError('PPO figure output directory must be fresh')
    summary, results, logs, evaluation_hashes = verify(args.report, args.execution, args.plan)
    content = render(summary, results, logs)
    rows = summary['training']
    receipt = {
        'kind': 'verified_autonomous_associative_ppo_figures',
        'publisher_sha256': sha(Path(__file__)), 'plan_sha256': sha(args.plan),
        'summary_sha256': sha(args.report/'summary.json'), 'report_completed_sha256': sha(args.report/'completed.json'),
        'execution_completed_sha256': sha(args.execution/'completed.json'),
        'training_completed_sha256': sha(args.execution/'training-completed.json'),
        'evaluation_file_sha256': evaluation_hashes,
        'verified_fits': 12, 'verified_training_log_rows': 12288, 'training_interactions': summary['interactions'],
        'optimizer_steps': sum(row['gradient_steps'] for row in rows),
        'training_seconds': summary['training_seconds'],
        'collection_seconds': sum(row['collection_seconds'] for row in rows),
        'optimization_seconds': sum(row['optimization_seconds'] for row in rows),
        'training_timing_scope': ('Recorded fit timers include environment setup, all rollouts and optimization, '
                                 'logging and checkpoint save; exclude model/optimizer initialization and evaluation'),
        'verified_evaluation_files': 55, 'verified_size_condition_results': 165, 'verified_episodes': 21120,
        'verified_paired_intervention_rows': 126, 'verified_active_step_write_audits': 90,
        'full_budget_cost_reported': True, 'background_reset_slot_cost_accounting_verified': True,
        'episode_aggregate_audit': True, 'fresh_scored_policy_replay': False,
        'continuation_passed': summary['continuation_passed'], 'continuation_checks': summary['checks'],
        'new_training': False, 'novelty_established': False, 'independent_confirmation': False,
        'scope': summary['protocol']['claim'],
        'artifacts': {name: {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}
                      for name, value in content.items()},
    }
    args.out.mkdir(parents=True, exist_ok=False)
    for name, value in content.items():
        with (args.out/name).open('xb') as stream:
            stream.write(value)
    with (args.out/'associative-ppo-figures.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'evaluation_file_sha256'}, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    publish(parser.parse_args())


if __name__ == '__main__':
    main()
