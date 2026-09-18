"""Render authenticated completed mapping audits without models or private assets.

Supply externally obtained plan and audit-receipt hashes. Every configured seed,
control and gate check is required. Synthetic data needs both an explicit tag
and flag, and receives a prominent watermark. Three-seed means are descriptive.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'chess-connectome-mapping-figure-v1'
STUDY = 'chess-connectome-mapping-v1'
VARIANTS = ('biological', 'rewire151', 'rewire163', 'rewire179', 'node_local')
POLICIES = ('fixed', 'learned')
SEEDS = (97, 109, 127)
SPLITS = ('dev', 'shift')
LABELS = ('Biological', 'Rewire 151', 'Rewire 163', 'Rewire 179', 'Node-local')
COLORS = {'fixed': '#2878b5', 'learned': '#ce7826'}
MARKERS = ('o', 's', '^')
CORE_SOURCES = {'scripts/chess_connectome_mapping_study.py',
                'src/openjev/research/chess_connectome_mapping_study.py',
                'src/openjev/research/chess_connectome_interface.py',
                'src/openjev/research/chess_connectome_study.py',
                'src/openjev/research/chess_connectome_adapter.py'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def valid_hash(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def number(value, low=-math.inf, high=math.inf):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def integer(value, expected):
    return type(value) is int and value == expected


def close(actual, expected):
    return number(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def parse(content):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')

    def floating(value):
        result = float(value)
        require(math.isfinite(result), 'Nonfinite JSON overflow')
        return result

    return json.loads(content, object_pairs_hook=pairs, parse_constant=constant, parse_float=floating)


def read_bytes(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f'Regular nonsymlink file required: {path}')
    return path.read_bytes()


def exact(actual, expected):
    """Compare arithmetic while keeping booleans, integer counts and identities strict."""
    if type(expected) is dict:
        return type(actual) is dict and set(actual) == set(expected) and all(
            exact(actual[key], value) for key, value in expected.items())
    if type(expected) is list:
        return type(actual) is list and len(actual) == len(expected) and all(
            exact(a, b) for a, b in zip(actual, expected, strict=True))
    if type(expected) is float:
        return close(actual, expected)
    return type(actual) is type(expected) and actual == expected


def configurations():
    return [{'name': f'{v}-{p}-{s}', 'variant': v, 'mapping_policy': p,
             'mode': 'node_local' if v == 'node_local' else 'sparse',
             'seed': s, 'mapping_seed': s, 'backbone_seed': s}
            for v in VARIANTS for p in POLICIES for s in SEEDS]


def baselines():
    return [{'name': f'direct-{s}', 'variant': 'direct', 'mode': 'direct',
             'seed': s, 'mapping_seed': s, 'backbone_seed': s} for s in SEEDS]


def configuration_members(actual, expected, label):
    require(type(actual) is list and len(actual) == len(expected)
            and all(type(row) is dict and type(row.get('name')) is str for row in actual),
            f'{label} membership differs')
    lookup = {row['name']: row for row in actual}
    require(len(lookup) == len(actual) and exact(lookup, {c['name']: c for c in expected}),
            f'{label} identity differs')
    return lookup


def recompute_comparison(losses, scope):
    arms = ['direct'] + [f'{v}-{p}' for v in VARIANTS for p in POLICIES]
    means = {arm: {split: math.fsum(losses[f'{arm}-{seed}'][split] for seed in SEEDS) / 3
                   for split in SPLITS} for arm in arms}
    checks, tolerance = [], 1e-12
    for split in SPLITS:
        biology = means['biological-learned'][split]
        for reference_name in ('biological-fixed', 'rewire151-learned', 'rewire163-learned', 'rewire179-learned'):
            reference = means[reference_name][split]
            reduction = (reference - biology) / reference if reference > 0 else None
            checks.append({'split': split, 'comparator': reference_name, 'criterion': 'mean_relative_reduction',
                           'reference': reference, 'treatment': biology, 'reduction': reduction,
                           'passed': reduction is not None and reduction + tolerance >= .10})
        for variant in VARIANTS[1:4]:
            for seed in SEEDS:
                difference = losses[f'biological-learned-{seed}'][split] - losses[f'{variant}-learned-{seed}'][split]
                checks.append({'split': split, 'comparator': f'{variant}-learned', 'seed': seed,
                               'criterion': 'strict_paired_improvement', 'difference': difference,
                               'passed': difference < -tolerance})
        for seed in SEEDS:
            difference = losses[f'biological-learned-{seed}'][split] - losses[f'direct-{seed}'][split]
            checks.append({'split': split, 'comparator': 'direct', 'seed': seed,
                           'criterion': 'no_paired_degradation', 'difference': difference,
                           'passed': difference <= tolerance})
        biological_gain = means['biological-fixed'][split] - biology
        for variant in VARIANTS[1:]:
            comparator_gain = means[f'{variant}-fixed'][split] - means[f'{variant}-learned'][split]
            difference = biological_gain - comparator_gain
            checks.append({'split': split, 'comparator': variant, 'criterion': 'mapping_difference_in_differences',
                           'biological_gain': biological_gain, 'comparator_gain': comparator_gain,
                           'difference': difference, 'passed': difference > tolerance})
    return {'means': means, 'checks': checks, 'continuation_passed': all(c['passed'] for c in checks), 'scope': scope}


def validate(summary, plan, *, expected_plan_sha256, synthetic_fixture=False):
    require(valid_hash(expected_plan_sha256), 'External plan SHA-256 required')
    require(type(summary.get('synthetic_fixture', False)) is bool
            and summary.get('synthetic_fixture', False) is synthetic_fixture,
            'Synthetic data requires a matching explicit tag and fixture flag')
    fields = {'status', 'version', 'plan_sha256', 'execution_receipt_sha256', 'metrics', 'engine_loss',
              'engine_metrics', 'mapping_comparison', 'latency', 'training_costs', 'grading_costs',
              'backbone_pretraining_costs', 'original_data_costs', 'original_inputs', 'wall_seconds',
              'limits', 'report_new_model_or_engine_calls'}
    require(set(summary) == fields | ({'synthetic_fixture'} if 'synthetic_fixture' in summary else set()),
            'Completed summary schema differs')
    protocol = plan['protocol']
    expected_protocol = {'version': STUDY, 'paired_seeds': list(SEEDS), 'variants': list(VARIANTS),
                         'mapping_policies': list(POLICIES), 'train_examples': 32768, 'epochs': 6,
                         'updates_per_fit': 1536, 'warmup_updates': 256, 'proposal_interval': 4,
                         'proposal_seed_base': 11600041, 'proposals_per_fit': 320,
                         'all_final_fits_before_neural_evaluation': True, 'regret_positions_per_split': 128,
                         'regret_nodes': 20000, 'relative_loss_reduction': .1, 'gate_absolute_tolerance': 1e-12}
    require(all(exact(protocol.get(key), value) for key, value in expected_protocol.items()),
            'Frozen protocol semantics differ')
    require(summary['status'] == 'completed' and summary['version'] == STUDY
            and summary['plan_sha256'] == expected_plan_sha256
            and integer(summary['report_new_model_or_engine_calls'], 0)
            and valid_hash(summary['execution_receipt_sha256']), 'Study completion or identity differs')
    configs = configuration_members(plan['configurations'], configurations(), 'Fit')
    direct = configuration_members(plan['baselines'], baselines(), 'Baseline')
    names = set(configs) | set(direct)
    require(type(summary['training_costs']) is dict and set(summary['training_costs']) == set(configs),
            'All 30 completed fits are required')
    for field in ('metrics', 'engine_loss', 'engine_metrics', 'latency'):
        require(type(summary[field]) is dict and set(summary[field]) == names,
                f'All 33 evaluated configurations required: {field}')
    sources = plan['sources']
    require(type(sources) is dict and CORE_SOURCES <= set(sources)
            and all(valid_hash(h) for h in sources.values()), 'Complete helper/runner source binding required')
    for name, fit in summary['training_costs'].items():
        require(fit['status'] == 'completed' and fit['version'] == 'openjev-connectome-hard-mapping-study-v1'
                and fit['plan_sha256'] == expected_plan_sha256 and exact(fit['configuration'], configs[name])
                and exact(fit['settings'], plan['training_settings']), 'Completed fit identity differs')
        expected_counts = {'updates': 1536, 'examples_seen': 196608, 'proposal_count': 320,
                           'proposal_forward_calls': 640, 'proposal_optimizer_updates': 0,
                           'proposal_examples_seen': 81920, 'training_forward_calls': 1536,
                           'backward_calls': 1536, 'total_forward_calls': 2176}
        require(all(integer(fit.get(k), v) for k, v in expected_counts.items()), 'Fit or proposal budget differs')
        require(type(fit['accepted_proposals']) is int and 0 <= fit['accepted_proposals'] <= 320,
                'Invalid accepted-proposal count')
        permutation = fit['final_permutation']
        require(type(permutation) is list and all(type(i) is int for i in permutation)
                and sorted(permutation) == list(range(64)), 'Invalid final square permutation')
        if configs[name]['mapping_policy'] == 'fixed':
            require(fit['accepted_proposals'] == 0 and permutation == list(range(64)), 'Fixed control changed mapping')
        require(all(number(fit[k], 0) for k in
                    ('proposal_wall_seconds', 'step_wall_seconds', 'training_seconds', 'fit_wall_seconds'))
                and fit['proposal_wall_seconds'] <= fit['step_wall_seconds'] <= fit['training_seconds']
                <= fit['fit_wall_seconds'] <= protocol['primary_wall_seconds'], 'Fit timing nesting differs')
        for key in ('initial_state_sha256', 'state_sha256', 'checkpoint_sha256', 'learning_sha256',
                    'proposals_sha256', 'backbone_sha256', 'immutable_buffers_sha256', 'graph_sha256'):
            require(valid_hash(fit[key]), 'Invalid fit/checkpoint identity')
        require(type(fit['source_sha256']) is dict and bool(fit['source_sha256'])
                and CORE_SOURCES - {'scripts/chess_connectome_mapping_study.py'} <= set(fit['source_sha256'])
                and all(path in sources and sources[path] == digest for path, digest in fit['source_sha256'].items()),
                'Fit helper source differs from frozen plan')
        require(fit['backbone_unchanged'] is True and fit['mps_fallback_environment'] in (None, '0'),
                'Frozen backbone or no-fallback identity differs')
    require(number(summary['wall_seconds'], 0, protocol['primary_wall_seconds'])
            and math.fsum(f['fit_wall_seconds'] for f in summary['training_costs'].values())
            <= summary['wall_seconds'] + 1e-9, 'Execution cost is smaller than sequential completed fits')
    for name in names:
        for field in ('metrics', 'engine_loss', 'engine_metrics'):
            require(type(summary[field][name]) is dict and set(summary[field][name]) == set(SPLITS),
                    f'Both panels required: {field}')
        for split in SPLITS:
            metric, engine = summary['metrics'][name][split], summary['engine_metrics'][name][split]
            require(integer(metric['examples'], 2048) and type(metric['correct']) is int
                    and 0 <= metric['correct'] <= 2048 and close(metric['agreement'], metric['correct'] / 2048)
                    and number(metric['target_nll'], 0), 'Evaluation membership or arithmetic differs')
            require(set(engine) == {'positions', 'mean_signed_bounded_loss', 'mean_cp_loss',
                                   'p95_cp_loss', 'max_cp_loss'}
                    and integer(engine['positions'], 128) and number(engine['mean_signed_bounded_loss'], -2, 2)
                    and number(summary['engine_loss'][name][split], -2, 2)
                    and close(engine['mean_signed_bounded_loss'], summary['engine_loss'][name][split]),
                    'Engine coverage or bounded loss differs')
            require(number(engine['mean_cp_loss']) and number(engine['p95_cp_loss'])
                    and type(engine['max_cp_loss']) is int
                    and engine['mean_cp_loss'] <= engine['max_cp_loss'] + 1e-9
                    and engine['p95_cp_loss'] <= engine['max_cp_loss'] + 1e-9, 'Invalid signed CP tail metrics')
        latency = summary['latency'][name]
        require(all(number(latency[k], 0) for k in ('mean_ms', 'total_wall_ms', 'warmup_wall_ms'))
                and close(latency['mean_ms'], latency['total_wall_ms'] / 128), 'Native latency arithmetic differs')
        for key in ('host_load_average_before_latency', 'host_load_average_after_latency'):
            require(type(latency[key]) is list and len(latency[key]) == 3
                    and all(number(v, 0) for v in latency[key]), 'Host-load coverage differs')
    comparison = recompute_comparison(summary['engine_loss'], protocol['scope'])
    require(exact(summary['mapping_comparison'], comparison) and len(comparison['checks']) == 40,
            'All 40 predeclared criteria and their arithmetic are required')
    grading = summary['grading_costs']
    require(set(grading) == {'calls', 'requested_nodes', 'reported_nodes', 'wall_seconds'}
            and type(grading['calls']) is int and 0 < grading['calls'] <= protocol['regret_call_ceiling']
            and integer(grading['requested_nodes'], grading['calls'] * 20000)
            and grading['requested_nodes'] <= protocol['regret_node_ceiling']
            and type(grading['reported_nodes']) is int and grading['reported_nodes'] >= 0
            and number(grading['wall_seconds'], 0), 'Grading computation/budget differs')
    require(exact(summary['original_inputs'], plan['original_inputs'])
            and exact(summary['backbone_pretraining_costs'], plan['binding']['backbone_pretraining_costs']),
            'Historical source/pretraining binding differs')
    require(summary['limits'] == [protocol[k] for k in ('matching', 'data', 'pretraining', 'selection',
            'timing', 'scope', 'arena', 'asset_policy', 'audit_scope')], 'Claim/audit scope differs')
    return summary


def load_completed(audit, plan_path, *, expected_plan_sha256, expected_receipt_sha256,
                   source_root=ROOT, synthetic_fixture=False):
    audit, plan_path, source_root = Path(audit), Path(plan_path), Path(source_root).resolve()
    require(valid_hash(expected_plan_sha256) and valid_hash(expected_receipt_sha256),
            'Externally expected plan and completed audit-receipt SHA-256 values are required')
    require(audit.is_dir() and not audit.is_symlink()
            and {p.name for p in audit.iterdir()} == {'started.json', 'summary.json', 'receipt.json'},
            'Exact completed audit membership required; no partial or failed audits')
    raw_plan = read_bytes(plan_path)
    require(sha256(raw_plan) == expected_plan_sha256, 'External frozen plan hash differs')
    plan = parse(raw_plan)
    raw = {name: read_bytes(audit / name) for name in ('started.json', 'summary.json', 'receipt.json')}
    require(sha256(raw['receipt.json']) == expected_receipt_sha256, 'External audit receipt hash differs')
    receipt, started, summary = (parse(raw[name]) for name in ('receipt.json', 'started.json', 'summary.json'))
    require(set(receipt) == {'status', 'plan_sha256', 'execution_receipt_sha256', 'audit_wall_seconds',
                            'completed_unix', 'new_model_calls', 'new_engine_calls', 'files'}
            and set(started) == {'status', 'plan_sha256', 'started_unix'}, 'Audit receipt schema differs')
    require(receipt['status'] == 'completed' and receipt['plan_sha256'] == expected_plan_sha256
            and valid_hash(receipt['execution_receipt_sha256'])
            and receipt['execution_receipt_sha256'] == summary['execution_receipt_sha256'],
            'Completed audit receipt identity differs')
    require(receipt['files'] == {name: sha256(raw[name]) for name in ('started.json', 'summary.json')},
            'Audit receipt member/summary binding differs')
    require(integer(receipt['new_model_calls'], 0) and integer(receipt['new_engine_calls'], 0)
            and started['status'] == 'started' and started['plan_sha256'] == expected_plan_sha256
            and number(started['started_unix'], 0) and number(receipt['completed_unix'], started['started_unix'])
            and number(receipt['audit_wall_seconds'], 0, plan['protocol']['audit_wall_seconds']),
            'Audit call count, timing or start identity differs')
    for path, digest in plan['sources'].items():
        relative = Path(path)
        require(type(path) is str and not relative.is_absolute() and '..' not in relative.parts
                and (source_root / relative).resolve().is_relative_to(source_root) and valid_hash(digest),
                'Unsafe or invalid source binding')
        require(sha256(read_bytes(source_root / relative)) == digest, f'Frozen source hash differs: {path}')
    validate(summary, plan, expected_plan_sha256=expected_plan_sha256, synthetic_fixture=synthetic_fixture)
    return summary, plan, {'audit_path': str(audit.resolve()), 'plan_path': str(plan_path.resolve()),
                          'plan_sha256': expected_plan_sha256, 'receipt_sha256': expected_receipt_sha256,
                          'summary_sha256': sha256(raw['summary.json']),
                          'execution_receipt_sha256': receipt['execution_receipt_sha256'],
                          'source_root': str(source_root), 'source_sha256': plan['sources']}


def chart_values(summary):
    loss_points, costs, tails = [], [], []
    for c in configurations() + baselines():
        name = c['name']
        for split in SPLITS:
            metrics = summary['engine_metrics'][name][split]
            row = {'configuration': name, 'variant': c['variant'], 'mapping_policy': c.get('mapping_policy', 'unchanged'),
                   'seed': c['seed'], 'split': split, **metrics}
            tails.append(row)
            loss_points.append({key: row[key] for key in ('configuration', 'variant', 'mapping_policy',
                                                         'seed', 'split', 'mean_signed_bounded_loss')})
        if c['variant'] != 'direct':
            f = summary['training_costs'][name]
            costs.append({'configuration': name, 'variant': c['variant'], 'mapping_policy': c['mapping_policy'],
                          'seed': c['seed'], 'fit_wall_seconds': f['fit_wall_seconds'],
                          'proposal_wall_seconds': f['proposal_wall_seconds'],
                          'nonproposal_fit_wall_seconds': f['fit_wall_seconds'] - f['proposal_wall_seconds'],
                          'accepted_proposals': f['accepted_proposals'], 'proposal_count': f['proposal_count']})
    means = {f'{v}-{p}': {k: math.fsum(r[k] for r in costs if r['variant'] == v and r['mapping_policy'] == p) / 3
                           for k in ('fit_wall_seconds', 'proposal_wall_seconds', 'nonproposal_fit_wall_seconds')}
             for v in VARIANTS for p in POLICIES}
    return {'loss_points': loss_points, 'fit_costs': costs, 'mean_fit_costs': means,
            'engine_tail_rows': tails, 'criteria': summary['mapping_comparison']['checks'],
            'mean_losses': summary['mapping_comparison']['means']}


def criterion_label(check):
    name = check['comparator'].replace('biological-fixed', 'fixed bio').replace('rewire', 'R').replace('-learned', '')
    criterion = check['criterion']
    if criterion == 'mean_relative_reduction':
        return f'Mean: 10% below {name}'
    if criterion == 'strict_paired_improvement':
        return f'Paired: below {name}, seed {check["seed"]}'
    if criterion == 'no_paired_degradation':
        return f'No loss vs direct, seed {check["seed"]}'
    return f'Mapping gain exceeds {name.replace("node_local", "node-local")}'


def figure(summary, *, synthetic_fixture=False):
    values = chart_values(summary)
    fig = plt.figure(figsize=(14.5, 11.8), facecolor='white')
    grid = fig.add_gridspec(2, 2, left=.075, right=.97, bottom=.16, top=.79,
                           width_ratios=[1, 1], wspace=.39, hspace=.45)
    axes = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(2)]
    tag = 'SYNTHETIC PREVIEW - NOT RESULTS\n' if synthetic_fixture else ''
    fig.suptitle(f'{tag}OpenJev: learned square interfaces on connectome graphs', x=.075, y=.98,
                 ha='left', fontsize=17, fontweight='bold', color='#24354a')
    checks = values['criteria']
    passed = sum(c['passed'] for c in checks)
    gate = summary['mapping_comparison']['continuation_passed']
    fig.text(.075, .89 if synthetic_fixture else .92,
             f'Predeclared mechanism gate: {"PASS" if gate else "DID NOT PASS"} ({passed}/40 criteria)',
             fontsize=12, fontweight='bold', color='#227149' if gate else '#a5412e')
    fig.text(.075, .859, 'All 30 final fits and 33 evaluated models. Three paired seeds; every control retained.',
             fontsize=10.5, color='#47566b')
    handles = [Line2D([0], [0], marker='o', color=COLORS[p], ls='', label=p.title()) for p in POLICIES]
    handles += [Line2D([0], [0], marker=m, color='#47566b', ls='', label=f'Seed {s}')
                for m, s in zip(MARKERS, SEEDS, strict=True)]
    handles += [Line2D([0], [0], color='#172d40', marker='_', ls='', markersize=12, label='Three-seed mean'),
                Line2D([0], [0], color='#7d8996', ls=':', label='Unchanged direct mean')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.07, .845), frameon=False,
               ncol=7, fontsize=8.7, columnspacing=1.3)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_color('#b6bfcc')
        ax.tick_params(colors='#40516b', length=0, pad=6)
    for ax, split, title in zip(axes[:2], SPLITS, ('A. Ordinary panel', 'B. Scenario-shifted panel'), strict=True):
        ax.grid(axis='y', alpha=.18)
        for vi, variant in enumerate(VARIANTS):
            for policy, offset in zip(POLICIES, (-.18, .18), strict=True):
                arm = f'{variant}-{policy}'
                for seed, marker, jitter in zip(SEEDS, MARKERS, (-.065, 0, .065), strict=True):
                    value = summary['engine_loss'][f'{arm}-{seed}'][split]
                    ax.scatter(vi + offset + jitter, value, color=COLORS[policy], marker=marker, s=29, zorder=3)
                ax.scatter(vi + offset, values['mean_losses'][arm][split], color='#172d40',
                           marker='_', s=135, linewidths=2, zorder=4)
        for seed, marker, jitter in zip(SEEDS, MARKERS, (-.10, 0, .10), strict=True):
            ax.scatter(5 + jitter, summary['engine_loss'][f'direct-{seed}'][split], marker=marker,
                       color='#7d8996', s=32, zorder=3)
        ax.axhline(values['mean_losses']['direct'][split], color='#7d8996', ls=':', lw=1.1)
        ax.scatter(5, values['mean_losses']['direct'][split], color='#172d40', marker='_', s=135, linewidths=2)
        ax.axhline(0, color='#b6bfcc', lw=.7)
        ax.set_xticks(range(6), list(LABELS) + ['Direct\nunchanged'], fontsize=8)
        ax.set_xlim(-.55, 5.5)
        ax.set_ylabel('Signed bounded engine loss (lower is better)', fontsize=9)
        ax.set_title(f'{title}: 128 positions', loc='left', fontsize=11, fontweight='bold', pad=12)
    # Shared ordinate makes ordinary/shift differences directly comparable.
    low = min(ax.get_ylim()[0] for ax in axes[:2])
    high = max(ax.get_ylim()[1] for ax in axes[:2])
    for ax in axes[:2]:
        ax.set_ylim(low, high)
    ax = axes[2]
    ax.grid(axis='y', alpha=.18)
    for vi, variant in enumerate(VARIANTS):
        for policy, offset in zip(POLICIES, (-.18, .18), strict=True):
            mean = values['mean_fit_costs'][f'{variant}-{policy}']
            remainder, proposal = mean['nonproposal_fit_wall_seconds'] / 60, mean['proposal_wall_seconds'] / 60
            ax.bar(vi + offset, remainder, width=.31, color=COLORS[policy], alpha=.65, zorder=2)
            ax.bar(vi + offset, proposal, bottom=remainder, width=.31, color=COLORS[policy],
                   edgecolor='white', hatch='////', lw=.6, zorder=2)
            for seed, marker, jitter in zip(SEEDS, MARKERS, (-.065, 0, .065), strict=True):
                row = next(r for r in values['fit_costs'] if r['configuration'] == f'{variant}-{policy}-{seed}')
                ax.scatter(vi + offset + jitter, row['fit_wall_seconds'] / 60, color='#24354a',
                           marker=marker, s=17, zorder=3)
    ax.set_xticks(range(5), LABELS, fontsize=8)
    ax.set_ylabel('Complete fit wall time (minutes)', fontsize=9)
    ax.set_ylim(0, max(r['fit_wall_seconds'] for r in values['fit_costs']) / 60 * 1.22)
    ax.set_title('C. Matched training and proposal costs', loc='left', fontsize=11, fontweight='bold', pad=12)
    ax.legend(handles=[Patch(facecolor='#8d9baa', alpha=.65, label='Other fit work'),
                       Patch(facecolor='#8d9baa', edgecolor='white', hatch='////', label='Proposal time within fit')],
              loc='upper left', frameon=False, fontsize=8)
    ax.text(0, -.20, 'Each fit: 1,536 updates + 320 proposals (640 extra forwards).\n'
            'Fixed controls pay the same proposal budget and accept none.',
            transform=ax.transAxes, fontsize=8.2, color='#47566b')
    ax = axes[3]
    per_split = [[c for c in checks if c['split'] == s] for s in SPLITS]
    for row in range(20):
        for column in range(2):
            passed = per_split[column][row]['passed']
            ax.add_patch(plt.Rectangle((column - .45, row - .43), .9, .86,
                                       color='#d4e9df' if passed else '#f4dfd9'))
            ax.text(column, row, 'P' if passed else 'F', ha='center', va='center', fontsize=7.6,
                    color='#227149' if passed else '#a5412e', fontweight='bold')
    ax.set_yticks(range(20), [criterion_label(c) for c in per_split[0]], fontsize=6.7)
    ax.set_xticks([0, 1], ['Ordinary', 'Shifted'], fontsize=8)
    ax.xaxis.tick_top()
    ax.set_xlim(-.5, 1.5)
    ax.set_ylim(19.5, -.5)
    ax.spines[:].set_visible(False)
    # Narrow cells leave room for full criterion labels, with no hidden checks.
    box = ax.get_position()
    ax.set_position([box.x0 + .20, box.y0, box.width - .20, box.height])
    fig.text(box.x0, box.y1 + .042, 'D. All 40 predeclared criteria (P/F)',
             fontsize=11, fontweight='bold')
    fig.text(.075, .091, 'Engine loss = mean[tanh(unrestricted cp/600) - tanh(chosen cp/600)], 20,000 nodes per call. '
             'Signed losses and all per-seed CP tails are retained.', fontsize=8.8, color='#47566b')
    fig.text(.075, .069, 'Cost bars show three-seed means; points show every fit. Proposal time is already within fit '
             'time. Shared host; original pretraining and evaluation excluded.', fontsize=8.8, color='#47566b')
    fig.text(.075, .047, 'All means are descriptive, without confidence intervals from three seeds. '
             'CP p95 values are per model/panel, never pooled or averaged into a quantile.', fontsize=8.8, color='#47566b')
    fig.text(.075, .021, 'Exposed development mechanism screen. No gameplay, Elo, intact-fly, world-model or novelty claim.',
             fontsize=10, fontweight='bold', color='#24354a')
    return fig


def render(audit, plan_path, out, *, expected_plan_sha256, expected_receipt_sha256,
           source_root=ROOT, synthetic_fixture=False):
    summary, plan, binding = load_completed(audit, plan_path, expected_plan_sha256=expected_plan_sha256,
        expected_receipt_sha256=expected_receipt_sha256, source_root=source_root, synthetic_fixture=synthetic_fixture)
    out = Path(out)
    require(not out.resolve().is_relative_to(Path(audit).resolve())
            and not out.resolve().is_relative_to(Path(plan_path).parent.resolve()),
            'Outputs must be outside authenticated evidence directories')
    if out.exists() or out.is_symlink():
        raise FileExistsError('Output directory already exists; use a new path')
    values, files = chart_values(summary), {}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9}):
        fig = figure(summary, synthetic_fixture=synthetic_fixture)
        try:
            for extension in ('png', 'pdf'):
                stream = io.BytesIO()
                fig.savefig(stream, format=extension, dpi=160, facecolor='white')
                files[f'figure.{extension}'] = stream.getvalue()
        finally:
            plt.close(fig)
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=list(values['engine_tail_rows'][0]))
    writer.writeheader()
    writer.writerows(values['engine_tail_rows'])
    files['engine_cp_tails.csv'] = table.getvalue().encode()
    manifest = {'version': VERSION, 'status': 'completed', 'synthetic_fixture': synthetic_fixture, **binding,
                'plot_source_sha256': sha256(Path(__file__).read_bytes()), 'matplotlib_version': matplotlib.__version__,
                'files': {name: sha256(content) for name, content in files.items()},
                'new_model_or_engine_calls': 0, 'fit_count': 30, 'evaluated_models': 33, 'criteria_count': 40,
                'source_fit_order': [c['name'] for c in plan['configurations']],
                'mapping_comparison': summary['mapping_comparison'], 'plotted_values': values,
                'claim_scope': plan['protocol']['scope'],
                'validation_scope': 'External plan/receipt hashes, complete audit membership, all frozen source bytes, '
                                    'full summary coverage and scalar arithmetic. No raw journal replay or model loads.',
                'aggregation': 'All three paired seeds; means only. Per-seed CP mean/p95/max preserved without '
                               'pooled-quantile claims. Proposal time is a subset of complete fit time.'}
    files['provenance.json'] = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
    out.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        with (out / name).open('xb') as destination:
            destination.write(content)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--expected-receipt-sha256', required=True)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--synthetic-fixture', action='store_true')
    args = parser.parse_args()
    result = render(args.audit, args.plan, args.out, expected_plan_sha256=args.expected_plan_sha256,
                    expected_receipt_sha256=args.expected_receipt_sha256, source_root=args.source_root,
                    synthetic_fixture=args.synthetic_fixture)
    print(json.dumps({'status': result['status'], 'synthetic_fixture': result['synthetic_fixture'],
                      'output': str(args.out), 'summary_sha256': result['summary_sha256']}))
