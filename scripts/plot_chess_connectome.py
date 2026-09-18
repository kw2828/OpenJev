"""Render all arms from an authenticated, completed connectome chess report.

This is a saved-summary renderer, not a replacement for the report's raw-output
audit. It imports no chess model, loads no weights or graph, and runs no inference.
Synthetic fixtures require both a source tag and an explicit rendering flag.
"""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

VERSION = 'chess-connectome-plot-v1'
SEEDS = (97, 109, 127)
VARIANTS = ('direct', 'biological', 'rewire151', 'rewire163', 'rewire179', 'dense', 'node_local')
SPLITS = ('dev', 'shift')
IDENTITIES = tuple(f'{variant}-{seed}' for variant in VARIANTS for seed in SEEDS)
LABELS = ('Direct\nfrozen', 'Biological\nsigned graph', 'Rewire\n151', 'Rewire\n163',
          'Rewire\n179', 'Dense', 'Node-local')
COLORS = ('#2878b5', '#e17c24', '#3b9563')
SUMMARY_FIELDS = {'status', 'plan_sha256', 'execution_receipt_sha256', 'metrics', 'engine_loss',
                  'topology_comparison', 'latency', 'training_costs', 'grading_costs',
                  'backbone_pretraining_costs', 'data_costs', 'wall_seconds', 'limits',
                  'report_new_model_or_engine_calls'}
METRIC_FIELDS = {'examples', 'correct', 'agreement', 'target_nll', 'value_mae', 'mean_confidence',
                 'mismatch_confidence', 'entropy', 'hidden_rms', 'logit_span'}
LATENCY_FIELDS = {'mean_ms', 'total_wall_ms', 'warmup_wall_ms', 'host_load_average_before_latency',
                  'host_load_average_after_latency'}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_hash(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(actual, expected):
    return number(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _json(data):
    return json.loads(data, object_pairs_hook=_unique)


def expected_comparison(losses):
    """Replay the fixed v1 gate, without importing the training runner."""
    means = {variant: {split: math.fsum(losses[f'{variant}-{seed}'][split] for seed in SEEDS)/3
                       for split in SPLITS} for variant in VARIANTS}
    checks = []
    tolerance = 1e-12
    for split in SPLITS:
        biological = means['biological'][split]
        for comparator in ('rewire151', 'rewire163', 'rewire179'):
            reference = means[comparator][split]
            reduction = (reference-biological)/reference if reference > 0 else None
            checks.append({'split': split, 'comparator': comparator, 'criterion': 'mean_relative_reduction',
                           'reduction': reduction, 'passed': reduction is not None and reduction+tolerance >= .10})
            for seed in SEEDS:
                difference = losses[f'biological-{seed}'][split]-losses[f'{comparator}-{seed}'][split]
                checks.append({'split': split, 'comparator': comparator, 'seed': seed,
                               'criterion': 'strict_paired_improvement', 'difference': difference,
                               'passed': difference < -tolerance})
        for seed in SEEDS:
            difference = losses[f'biological-{seed}'][split]-losses[f'direct-{seed}'][split]
            checks.append({'split': split, 'comparator': 'direct', 'seed': seed,
                           'criterion': 'no_paired_degradation', 'difference': difference,
                           'passed': difference <= tolerance})
    return {'means': means, 'checks': checks, 'continuation_passed': all(c['passed'] for c in checks)}


def validate(summary, *, synthetic_fixture=False):
    fields = SUMMARY_FIELDS | ({'synthetic_fixture'} if synthetic_fixture else set())
    require(type(summary) is dict and set(summary) == fields, 'Unexpected summary schema or fixture label')
    require(not synthetic_fixture or summary['synthetic_fixture'] is True, 'Fixture source must be explicitly tagged')
    require(summary['status'] == 'completed', 'Report is not completed')
    require(type(summary['report_new_model_or_engine_calls']) is int
            and summary['report_new_model_or_engine_calls'] == 0, 'Report must be a saved-output audit')
    for field in ('plan_sha256', 'execution_receipt_sha256'):
        require(valid_hash(summary[field]), f'Invalid {field}')
    require(number(summary['wall_seconds']) and summary['wall_seconds'] >= 0, 'Invalid study wall time')
    require(type(summary['limits']) is list and len(summary['limits']) == 7
            and all(type(v) is str and v.strip() for v in summary['limits']), 'Missing study limitations')
    require(set(summary['training_costs']) == set(IDENTITIES)-{f'direct-{s}' for s in SEEDS},
            'Expected all eighteen retained final fits')
    for field in ('metrics', 'engine_loss', 'latency'):
        require(type(summary[field]) is dict and set(summary[field]) == set(IDENTITIES),
                f'Expected all twenty-one models in {field}')
    for identity in IDENTITIES:
        require(set(summary['metrics'][identity]) == set(SPLITS)
                and set(summary['engine_loss'][identity]) == set(SPLITS), 'Missing or extra panel')
        for split in SPLITS:
            metric = summary['metrics'][identity][split]
            require(set(metric) == METRIC_FIELDS, 'Unexpected metric schema')
            require(type(metric['examples']) is int and metric['examples'] == 2048,
                    'Expected the complete 2048-position panel')
            require(type(metric['correct']) is int and 0 <= metric['correct'] <= metric['examples'],
                    'Invalid target agreement count')
            require(close(metric['agreement'], metric['correct']/metric['examples']),
                    'Target agreement arithmetic differs')
            for key in METRIC_FIELDS-{'examples', 'correct', 'agreement', 'mismatch_confidence'}:
                require(number(metric[key]) and metric[key] >= 0, f'Invalid metric: {key}')
            require(metric['mean_confidence'] <= 1, 'Confidence exceeds one')
            mismatch = metric['mismatch_confidence']
            require((mismatch is None and metric['correct'] == metric['examples']) or
                    (metric['correct'] < metric['examples'] and number(mismatch) and 0 <= mismatch <= 1),
                    'Invalid mismatch confidence')
            loss = summary['engine_loss'][identity][split]
            require(number(loss) and -2 <= loss <= 2, 'Invalid signed bounded engine-score loss')
        latency = summary['latency'][identity]
        require(set(latency) == LATENCY_FIELDS, 'Unexpected latency schema')
        for key in ('mean_ms', 'total_wall_ms', 'warmup_wall_ms'):
            require(number(latency[key]) and latency[key] >= 0, 'Invalid latency')
        require(close(latency['mean_ms'], latency['total_wall_ms']/128), 'Latency arithmetic differs')
        for key in LATENCY_FIELDS-{'mean_ms', 'total_wall_ms', 'warmup_wall_ms'}:
            require(type(latency[key]) is list and len(latency[key]) == 3
                    and all(number(v) and v >= 0 for v in latency[key]), 'Invalid shared-host load observation')
    expected = expected_comparison(summary['engine_loss'])
    actual = summary['topology_comparison']
    # Canonical JSON preserves bool-vs-int types that ordinary dictionary equality would conflate.
    require(json.dumps(actual, sort_keys=True, allow_nan=False) ==
            json.dumps(expected, sort_keys=True, allow_nan=False), 'Gate or equal-seed mean arithmetic differs')
    return summary


def load_completed(path, expected_sha256, *, synthetic_fixture=False):
    path = Path(path)
    require(path.name == 'summary.json' and path.is_file() and not path.is_symlink(),
            'Expected a regular report summary.json')
    require(valid_hash(expected_sha256), 'Expected summary SHA256 is required')
    raw = path.read_bytes()
    require(sha256(raw) == expected_sha256, 'Summary hash differs')
    receipt_path = path.parent/'completed.json'
    require(receipt_path.is_file() and not receipt_path.is_symlink(), 'Missing regular report completion receipt')
    require({p.name for p in path.parent.iterdir()} == {'summary.json', 'completed.json'},
            'Report directory is incomplete or contains extra artifacts')
    receipt_raw = receipt_path.read_bytes()
    receipt, summary = _json(receipt_raw), _json(raw)
    require(receipt == {'status': 'completed', 'plan_sha256': summary.get('plan_sha256'),
                        'files': {'summary.json': expected_sha256}}, 'Report completion binding differs')
    validate(summary, synthetic_fixture=synthetic_fixture)
    return summary, {'summary_sha256': expected_sha256, 'report_completed_sha256': sha256(receipt_raw)}


def chart_values(summary):
    """Every plotted scalar in fixed arm and seed order, with descriptive means."""
    values = []
    for variant in VARIANTS:
        record = {'variant': variant, 'seeds': []}
        for seed in SEEDS:
            identity = f'{variant}-{seed}'
            record['seeds'].append({'seed': seed, 'model': identity,
                'target_agreement_percent': {s: 100*summary['metrics'][identity][s]['agreement'] for s in SPLITS},
                'signed_bounded_engine_score_loss': summary['engine_loss'][identity],
                'cpu_full_decision_mean_ms': summary['latency'][identity]['mean_ms']})
        record['mean'] = {key: {s: math.fsum(v[key][s] for v in record['seeds'])/3 for s in SPLITS}
                          for key in ('target_agreement_percent', 'signed_bounded_engine_score_loss')}
        record['mean']['cpu_full_decision_mean_ms'] = math.fsum(
            v['cpu_full_decision_mean_ms'] for v in record['seeds'])/3
        values.append(record)
    return values


def figure(summary, *, synthetic_fixture=False):
    values = chart_values(summary)
    fig = plt.figure(figsize=(14.5, 12.7), facecolor='white')
    grid = fig.add_gridspec(3, 2, left=.075, right=.98, bottom=.145, top=.80,
                           wspace=.23, hspace=.63, height_ratios=(1, 1, 1))
    axes = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(2)]
    axes.append(fig.add_subplot(grid[2, :]))
    fixture = 'SYNTHETIC FIXTURE - NOT RESULTS\n' if synthetic_fixture else ''
    fig.suptitle(f'{fixture}OpenJev: biological wiring on a frozen chess backbone', x=.075, y=.977,
                 ha='left', fontsize=18, fontweight='bold', color='#203149')
    passed = summary['topology_comparison']['continuation_passed']
    gate = 'PASS' if passed else 'DID NOT PASS'
    fig.text(.075, .908 if synthetic_fixture else .927, f'Predeclared continuation gate: {gate}',
             fontsize=13, fontweight='bold', color='#227149' if passed else '#a5412e')
    fig.text(.075, .876 if synthetic_fixture else .89,
             'All 21 models shown: 7 arms x 3 paired seeds. Points are seeds; black ticks are equal-seed means.',
             fontsize=10.5, color='#47566b')
    handles = [Line2D([0], [0], color=color, marker='o', linestyle='', label=f'Seed {seed}', markersize=6)
               for seed, color in zip(SEEDS, COLORS, strict=True)]
    handles.append(Line2D([0], [0], color='#162335', marker='_', linestyle='', markersize=17,
                          markeredgewidth=2.5, label='Mean (not a confidence interval)'))
    fig.legend(handles=handles, bbox_to_anchor=(.067, .866 if not synthetic_fixture else .853),
               loc='upper left', frameon=False, ncol=4, fontsize=10)
    specs = [('target_agreement_percent', s) for s in SPLITS]
    specs += [('signed_bounded_engine_score_loss', s) for s in SPLITS]
    specs += [('cpu_full_decision_mean_ms', None)]
    all_agreement, all_loss = [], []
    for ax, (metric, split) in zip(axes, specs, strict=True):
        extract = lambda obj, metric=metric, split=split: obj[metric][split] if split else obj[metric]
        for si, color in enumerate(COLORS):
            ax.scatter([i+(-.17, 0, .17)[si] for i in range(len(VARIANTS))],
                       [extract(record['seeds'][si]) for record in values], color=color,
                       s=29, edgecolors='white', linewidths=.45, zorder=3)
        means = [extract(record['mean']) for record in values]
        ax.scatter(range(len(VARIANTS)), means, color='#162335', marker='_', s=185, linewidths=2.4, zorder=4)
        ax.set_xticks(range(len(VARIANTS)), LABELS, fontsize=9)
        ax.set_xlim(-.55, len(VARIANTS)-.45)
        ax.grid(axis='y', alpha=.18, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        for side in ('bottom', 'left'):
            ax.spines[side].set_color('#b6bfcc')
        ax.tick_params(colors='#40516b', length=0, pad=6)
        panel = 'Development panel' if split == 'dev' else 'Scenario-shift panel'
        if metric == 'target_agreement_percent':
            ax.set_title(f'{panel}: target agreement', loc='left', fontsize=11, fontweight='bold', pad=12)
            ax.set_ylabel('Agreement (%) - higher is better')
            all_agreement += [v['target_agreement_percent'][split] for r in values for v in r['seeds']]
        elif metric == 'signed_bounded_engine_score_loss':
            ax.set_title(f'{panel}: engine-score loss', loc='left', fontsize=11, fontweight='bold', pad=12)
            ax.set_ylabel('Signed bounded loss - lower is better')
            all_loss += [v[metric][split] for r in values for v in r['seeds']]
        else:
            ax.set_title('Full CPU decision: 128 pooled positions (64 per panel)',
                         loc='left', fontsize=11, fontweight='bold', pad=12)
            ax.set_ylabel('Milliseconds per decision')
            ax.set_ylim(bottom=0)
    upper = max(10, math.ceil(max(all_agreement)*1.08/10)*10)
    for ax in axes[:2]:
        ax.set_ylim(0, min(100, upper))
    low, high = min(0., min(all_loss)), max(0., max(all_loss))
    padding = max(.01, (high-low)*.12)
    for ax in axes[2:4]:
        ax.set_ylim(low-padding, high+padding)
        ax.axhline(0, color='#9ca8b6', linewidth=.8, zorder=1)
    fig.text(.075, .105, 'Agreement: 2,048 labeled positions per panel. Engine loss: 128 positions per panel, '
             '20,000 Stockfish nodes per call.', fontsize=9, color='#47566b')
    fig.text(.075, .081, 'Timing includes board/menu construction and the full choose call on CPU. '
             'Shared interactive host, not isolated latency; 3 warmups charged separately.', fontsize=9, color='#47566b')
    fig.text(.075, .061, 'Engine loss = mean[tanh(unrestricted cp/600) - tanh(chosen cp/600)]; '
             'finite search can yield negative values.', fontsize=9, color='#47566b')
    fig.text(.075, .041, 'Every mode uses dense matmuls. Dense/node-local parameter counts differ. '
             'The topology gate compares biological wiring with all three rewires and direct.',
             fontsize=9, color='#47566b')
    fig.text(.075, .017, 'Development mechanism screen only. This does not establish Elo, gameplay strength, '
             'a world model, or novelty.', fontsize=10, color='#203149', fontweight='bold')
    return fig


def render(summary_path, expected_sha256, out, *, synthetic_fixture=False):
    summary, binding = load_completed(summary_path, expected_sha256, synthetic_fixture=synthetic_fixture)
    out = Path(out)
    require(not out.resolve().is_relative_to(Path(summary_path).parent.resolve()),
            'Figure outputs must be outside the authenticated report directory')
    if out.exists() or out.is_symlink():
        raise FileExistsError('Output directory already exists; choose a new directory')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9.5}):
        fig = figure(summary, synthetic_fixture=synthetic_fixture)
        try:
            stream = io.BytesIO()
            fig.savefig(stream, format='png', dpi=160, facecolor='white', metadata={'Software': VERSION})
            png = stream.getvalue()
        finally:
            plt.close(fig)
    manifest = {
        'version': VERSION, 'status': 'completed', 'synthetic_fixture': synthetic_fixture,
        **binding, 'plan_sha256': summary['plan_sha256'],
        'execution_receipt_sha256': summary['execution_receipt_sha256'],
        'plot_source_sha256': sha256(Path(__file__).read_bytes()),
        'figure_sha256': {'connectome-chess.png': sha256(png)},
        'matplotlib_version': matplotlib.__version__, 'models': 21, 'final_adapter_fits': 18,
        'paired_seeds': list(SEEDS), 'arm_order': list(VARIANTS), 'panels': list(SPLITS),
        'target_positions_per_panel': 2048, 'engine_positions_per_panel': 128,
        'engine_requested_nodes_per_call': 20000, 'latency_positions_total': 128,
        'latency_positions_per_panel': 64, 'latency_warmups_per_model': 3,
        'aggregation': 'Every seed shown; arithmetic means across all three paired seeds. No confidence intervals.',
        'continuation_passed': summary['topology_comparison']['continuation_passed'],
        'gate_checks_passed': sum(c['passed'] for c in summary['topology_comparison']['checks']),
        'gate_checks_total': 30, 'gate_recomputed_from_all_losses': True,
        'gate_rule': 'Biological mean signed bounded loss at least10% below each rewire with positive reference '
                     'means, strictly better in every paired seed, and no worse than direct in every paired '
                     'seed, all on both panels. Absolute tolerance1e-12.',
        'selection': 'No winner selection, sorting by outcome, dropping arms, seeds, panels or negative losses.',
        'timing_scope': 'Full CPU choose, both panels pooled, shared interactive host; not isolated latency.',
        'validation_scope': 'Authenticated completed summary plus saved-scalar/schema/gate validation. '
                            'The study report, not this renderer, audits raw predictions and engine calls.',
        'new_model_or_engine_calls': 0, 'limits': summary['limits'], 'plotted_values': chart_values(summary),
    }
    encoded = (json.dumps(manifest, indent=2, allow_nan=False)+'\n').encode()
    out.mkdir(parents=True, exist_ok=False)
    for name, data in (('connectome-chess.png', png), ('source-manifest.json', encoded)):
        with (out/name).open('xb') as destination:
            destination.write(data)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--summary-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--synthetic-fixture', action='store_true', help='Watermark explicitly tagged test data')
    args = parser.parse_args()
    result = render(args.summary, args.summary_sha256, args.out, synthetic_fixture=args.synthetic_fixture)
    print(json.dumps({'status': result['status'], 'synthetic_fixture': result['synthetic_fixture'],
                      'output': str(args.out), 'summary_sha256': result['summary_sha256']}))
