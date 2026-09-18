"""Render a complete, authenticated continuation audit without model or engine calls.

The frozen plan is pinned here. Synthetic fixtures require both an explicit
summary tag and the CLI flag; their figures carry a prominent watermark.
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

VERSION = 'chess-continuation-figure-v1'
PLAN_SHA256 = '5f3a1d25fcc2da1c1f59a41b91bd4dbc6f1de29bf34e27b3c0c710e6a72cc9b8'
OBJECTIVES = ('policy', 'best_value', 'continuation')
LABELS = {'policy': 'Policy', 'best_value': 'Best-value', 'continuation': 'Continuation'}
SEEDS = (193, 211, 227)
SPLITS = ('dev', 'shift')
COLORS = dict(zip(OBJECTIVES, ('#2878b5', '#e17c24', '#3b9563'), strict=True))
MARKERS = ('o', 's', '^')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def number(value, low=-math.inf, high=math.inf):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def count(value, low=0):
    return type(value) is int and value >= low


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


def unique_rows(rows, fields, expected, name):
    require(isinstance(rows, list) and len(rows) == len(expected), f'{name} coverage differs')
    keys = [tuple(row[field] for field in fields) for row in rows]
    require(len(set(keys)) == len(keys) and set(keys) == set(expected), f'{name} identities differ')
    return dict(zip(keys, rows, strict=True))


def validate_metrics(metrics, examples):
    require(type(metrics['examples']) is int and metrics['examples'] == examples
            and count(metrics['correct']) and metrics['correct'] <= examples,
            'Evaluation example counts differ')
    require(close(metrics['agreement'], metrics['correct']/examples), 'Agreement arithmetic differs')
    require(number(metrics['target_nll'], 0), 'Invalid target NLL')


def arena_points(games):
    statuses = {key: sum(g['status'] == key for g in games) for key in ('completed', 'unfinished', 'failed')}
    wins = draws = losses = 0
    for game in games:
        if game['status'] != 'completed':
            continue
        if game['result'] == '1/2-1/2':
            draws += 1
        elif game['result'] == ('1-0' if game['white'].startswith('continuation-') else '0-1'):
            wins += 1
        else:
            losses += 1
    completed = wins+.5*draws
    return {'games': len(games), 'status_counts': statuses, 'continuation': {
        'wins': wins, 'draws': draws, 'losses': losses, 'completed_points': completed,
        'possible_points': len(games), 'score_lower_bound': completed/len(games),
        'score_upper_bound': (completed+statuses['unfinished']+statuses['failed'])/len(games)}}


def validate_arena(summary, plan):
    require(summary['status'] == 'completed' and type(summary['games']) is int and summary['games'] == 192
            and set(summary['by_opponent']) == {'policy', 'best_value'}
            and summary['elo_estimate'] is None, 'Arena scope differs')
    games, schedule = summary['game_results'], plan['arena']['games']
    require(len(games) == len(schedule) == 192, 'Full arena schedule required')
    for game, spec in zip(games, schedule, strict=True):
        require(game['game_id'] == spec['id'] and all(game[key] == spec[key] for key in
                ('opening_id', 'opponent', 'seed', 'white', 'black')), 'Arena game identity differs')
        require(type(game['seed']) is int and count(game['played_plies'])
                and game['status'] in ('completed', 'unfinished', 'failed'), 'Invalid arena game status')
        expected_results = ('1-0', '0-1', '1/2-1/2') if game['status'] == 'completed' else ('*',)
        require(game['result'] in expected_results, 'Arena result and status disagree')
    checks = []
    for opponent in ('policy', 'best_value', None):
        chosen = games if opponent is None else [g for g in games if g['opponent'] == opponent]
        expected = arena_points(chosen)
        panel = summary['aggregate'] if opponent is None else summary['by_opponent'][opponent]
        require(type(panel['games']) is int and panel['games'] == expected['games'], 'Arena denominator differs')
        require(set(panel['status_counts']) == set(expected['status_counts'])
                and all(count(v) for v in panel['status_counts'].values())
                and panel['status_counts'] == expected['status_counts'], 'Arena statuses differ from game rows')
        for key, value in expected['continuation'].items():
            require(close(panel['continuation'][key], value), 'Arena point bounds differ from full game rows')
        require(all(count(panel['continuation'][key]) for key in ('wins', 'draws', 'losses', 'possible_points')),
                'Arena result counts must be integers')
        if opponent is not None:
            lower = expected['continuation']['score_lower_bound']
            checks.append({'opponent': opponent, 'threshold': .60, 'lower_score_bound': lower, 'passed': lower >= .60})
    aggregate = arena_points(games)
    require(summary['status_counts'] == aggregate['status_counts'], 'Aggregate statuses differ')
    zero_failed = aggregate['status_counts']['failed'] == 0
    expected_gate = {'checks': checks, 'zero_failed_games': zero_failed,
                     'passed': zero_failed and all(c['passed'] for c in checks)}
    for key in ('gate', 'continuation_gate'):
        require(summary[key] == expected_gate and type(summary[key]['passed']) is bool,
                'Arena gate differs')
    require(type(summary['gate_passed']) is bool and summary['gate_passed'] == expected_gate['passed'],
            'Arena gate flag differs')
    return expected_gate


def validate_gate(summary, engine, arena_gate):
    gate, expected_checks = summary['continuation_gate'], []
    for control in OBJECTIVES[:2]:
        for split in SPLITS:
            reference = math.fsum(engine[f'{control}-{seed}', split]['mean_signed_bounded_loss'] for seed in SEEDS)/3
            treatment = math.fsum(engine[f'continuation-{seed}', split]['mean_signed_bounded_loss'] for seed in SEEDS)/3
            expected_checks.append({'control': control, 'split': split, 'metric': 'bounded_engine_loss',
                'reference': reference, 'continuation': treatment,
                'relative_reduction': (reference-treatment)/reference if reference > 0 else None,
                'passed': reference > 0 and treatment <= .8*reference+1e-12})
    expected_checks.append({'metric': 'paired_games', 'details': arena_gate, 'passed': arena_gate['passed']})
    require(len(gate['checks']) == len(expected_checks), 'Study gate check coverage differs')
    for actual, expected in zip(gate['checks'], expected_checks, strict=True):
        require(set(actual) == set(expected), 'Study gate fields differ')
        for key, value in expected.items():
            if type(value) is float:
                require(close(actual[key], value), 'Study gate arithmetic differs')
            else:
                require(actual[key] == value and (type(value) is not bool or type(actual[key]) is bool),
                        'Study gate differs')
    require(type(gate['passed']) is bool and gate['passed'] == all(c['passed'] for c in expected_checks)
            and gate['scope'] == summary['claim_scope']
            and gate['thresholds'] == {'relative_loss_reduction': .20, 'lower_game_points': .60},
            'Study gate identity differs')


def validate(summary, plan, *, synthetic_fixture=False):
    require(type(summary.get('synthetic_fixture', False)) is bool
            and summary.get('synthetic_fixture', False) is synthetic_fixture,
            'Synthetic data requires an explicit tag and matching fixture flag')
    require(summary['version'] == plan['protocol']['version'] == 'chess-continuation-v1'
            and summary['plan_sha256'] == PLAN_SHA256, 'Study identity differs')
    require(summary['claim_scope'] == plan['protocol']['scope']
            and summary['audit_scope'] == plan['protocol']['audit_scope'], 'Study claim or audit scope differs')
    configs = plan['configurations']
    names = {f'{o}-{s}' for o in OBJECTIVES for s in SEEDS}
    require(len(configs) == 9 and {c['name'] for c in configs} == names, 'Frozen fit membership differs')
    require(len(summary['training']) == 9, 'All nine completed fits are required')
    for fit, config in zip(summary['training'], configs, strict=True):
        require(all(fit[key] == config[key] for key in config) and type(fit['seed']) is int
                and fit['status'] == 'completed' and fit['plan_sha256'] == PLAN_SHA256,
                'Fit identity or completion differs')
        require(count(fit['updates'], 1) and fit['updates'] == plan['updates_per_fit']
                and count(fit['examples_seen'], 1) and fit['examples_seen'] == plan['examples_seen_per_fit']
                and number(fit['training_seconds'], 0, plan['protocol']['primary_wall_seconds'])
                and number(fit['completed_unix'], 0), 'Fit count or wall time differs')
        require(fit['parameters'] == plan['parameters'][config['objective']]
                and fit['initial_state_sha256'] == plan['initial_state_sha256'][str(config['seed'])]
                and valid_hash(fit['learning_sha256']), 'Fit parameter or initialization identity differs')
        require(set(fit['checkpoints']) == {f'epoch-{e:02d}.pt' for e in range(1, 9)}
                and all(valid_hash(h) for h in fit['checkpoints'].values()), 'All eight checkpoint bindings required')
    require(math.fsum(f['training_seconds'] for f in summary['training']) <= plan['protocol']['primary_wall_seconds'],
            'Training time exceeds the frozen primary wall budget')
    curves = unique_rows(summary['curves'], ('configuration', 'epoch', 'partition'),
                         {(n, e, p) for n in names for e in range(1, 9) for p in ('train', 'diagnostic')}, 'Curve')
    for (_, epoch, partition), row in curves.items():
        require(type(epoch) is int and number(row['evaluation_wall_seconds'], 0), 'Invalid checkpoint evaluation')
        validate_metrics(row['metrics'], plan['dataset']['membership']['fixed_diagnostic_subsets'][partition]['roots'])
    primary = unique_rows(summary['primary'], ('configuration', 'split'),
                          {(n, s) for n in names for s in SPLITS}, 'Primary')
    for row in primary.values():
        validate_metrics(row['metrics'], 4096)
        require(number(row['evaluation_wall_seconds'], 0), 'Invalid primary evaluation time')
    engine = unique_rows(summary['engine_metrics'], ('configuration', 'split'),
                         {(n, s) for n in names for s in SPLITS}, 'Engine metric')
    for row in engine.values():
        require(type(row['positions']) is int and row['positions'] == 128
                and number(row['mean_signed_bounded_loss'], -2, 2), 'Invalid engine coverage or bounded loss')
    unique_rows(summary['native_latency'], ('configuration',), {(n,) for n in names}, 'Native latency')
    arena_gate = validate_arena(summary['arena'], plan)
    validate_gate(summary, engine, arena_gate)
    return summary


def load_completed(audit, plan_path, *, synthetic_fixture=False):
    audit, plan_path = Path(audit), Path(plan_path)
    require(audit.is_dir() and not audit.is_symlink(), 'Regular audit directory required')
    require(not (audit/'failed.json').exists() and not (audit/'failed.json').is_symlink(), 'Failed audit cannot be rendered')
    require({p.name for p in audit.iterdir()} == {'started.json', 'summary.json', 'receipt.json'},
            'Final audit file coverage differs')
    raw_plan = read_bytes(plan_path)
    require(sha256(raw_plan) == PLAN_SHA256, 'Frozen plan hash differs')
    plan = parse(raw_plan)
    raw = {name: read_bytes(audit/name) for name in ('started.json', 'summary.json', 'receipt.json')}
    receipt, started, summary = (parse(raw[name]) for name in ('receipt.json', 'started.json', 'summary.json'))
    require(receipt['status'] == 'completed' and receipt['plan_sha256'] == PLAN_SHA256
            and valid_hash(receipt['execution_receipt_sha256']), 'Completed audit receipt identity differs')
    require(receipt['files'] == {name: sha256(raw[name]) for name in ('started.json', 'summary.json')},
            'Audit receipt manifest or summary hash differs')
    require(all(type(receipt[key]) is int and receipt[key] == 0 for key in ('new_model_calls', 'new_engine_calls')),
            'Audit unexpectedly made new calls')
    require(number(receipt['audit_wall_seconds'], 0, plan['protocol']['audit_wall_seconds'])
            and number(receipt['completed_unix'], 0) and started['status'] == 'started'
            and started['plan_sha256'] == PLAN_SHA256 and number(started['started_unix'], 0)
            and receipt['completed_unix'] >= started['started_unix'], 'Audit time or start identity differs')
    validate(summary, plan, synthetic_fixture=synthetic_fixture)
    binding = {'audit_path': str(audit.resolve()), 'plan_path': str(plan_path.resolve()),
               'plan_sha256': PLAN_SHA256, 'receipt_sha256': sha256(raw['receipt.json']),
               'summary_sha256': sha256(raw['summary.json']),
               'execution_receipt_sha256': receipt['execution_receipt_sha256']}
    return summary, plan, binding


def chart_values(summary):
    fits = {r['name']: r for r in summary['training']}
    engine = {(r['configuration'], r['split']): r for r in summary['engine_metrics']}
    curves = {(r['configuration'], r['epoch'], r['partition']): r for r in summary['curves']}
    values = {'fits': [], 'nll_curves': [], 'arena': []}
    for objective in OBJECTIVES:
        for seed in SEEDS:
            name = f'{objective}-{seed}'
            values['fits'].append({'configuration': name, 'objective': objective, 'seed': seed,
                'training_seconds': fits[name]['training_seconds'],
                'checkpoints': fits[name]['checkpoints'],
                'engine_loss': {s: engine[name, s]['mean_signed_bounded_loss'] for s in SPLITS}})
        for partition in ('diagnostic', 'train'):
            for epoch in range(1, 9):
                losses = [curves[f'{objective}-{seed}', epoch, partition]['metrics']['target_nll'] for seed in SEEDS]
                values['nll_curves'].append({'objective': objective, 'partition': partition, 'epoch': epoch,
                                           'seed_nll': dict(zip(map(str, SEEDS), losses, strict=True)),
                                           'mean_nll': math.fsum(losses)/3})
    for opponent in OBJECTIVES[:2]:
        panel = summary['arena']['by_opponent'][opponent]
        values['arena'].append({'opponent': opponent, 'games': panel['games'],
                                'status_counts': panel['status_counts'], **panel['continuation']})
    return values


def figure(summary, *, synthetic_fixture=False):
    values = chart_values(summary)
    fig = plt.figure(figsize=(14.5, 11.5), facecolor='white')
    grid = fig.add_gridspec(2, 2, left=.075, right=.975, bottom=.17, top=.79, wspace=.27, hspace=.55)
    axes = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(2)]
    fixture = 'SYNTHETIC FIXTURE - NOT RESULTS\n' if synthetic_fixture else ''
    fig.suptitle(f'{fixture}OpenJev: continuation supervision for a recurrent chess actor',
                 x=.075, y=.978, ha='left', fontsize=17, fontweight='bold', color='#203149')
    passed = summary['continuation_gate']['passed']
    fig.text(.075, .891 if synthetic_fixture else .918,
             f'Predeclared continuation gate: {"PASS" if passed else "DID NOT PASS"}',
             fontsize=12.5, fontweight='bold', color='#227149' if passed else '#a5412e')
    fig.text(.075, .862, 'All 9 fits, all 8 checkpoints, all 192 scheduled games. Epoch 8 is primary; no best-seed selection.',
             fontsize=10.5, color='#47566b')
    handles = [Line2D([0], [0], color=COLORS[o], lw=3, label=LABELS[o]) for o in OBJECTIVES]
    handles += [Line2D([0], [0], color='#47566b', marker=m, linestyle='', label=f'Seed {s}')
                for m, s in zip(MARKERS, SEEDS, strict=True)]
    handles.append(Line2D([0], [0], color='#162335', marker='_', linestyle='', label='Engine mean', markersize=12))
    fig.legend(handles=handles, bbox_to_anchor=(.069, .845), loc='upper left',
               frameon=False, ncol=7, fontsize=9, columnspacing=1.3)
    for ax in axes:
        ax.grid(axis='y', alpha=.18, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        for side in ('bottom', 'left'):
            ax.spines[side].set_color('#b6bfcc')
        ax.tick_params(colors='#40516b', length=0, pad=6)

    ax = axes[0]
    for oi, objective in enumerate(OBJECTIVES):
        for si, split in enumerate(SPLITS):
            x = 2*oi+si
            records = [r for r in values['fits'] if r['objective'] == objective]
            for offset, marker, record in zip((-.14, 0, .14), MARKERS, records, strict=True):
                ax.scatter(x+offset, record['engine_loss'][split], color=COLORS[objective], marker=marker, s=34, zorder=3)
            ax.scatter(x, math.fsum(r['engine_loss'][split] for r in records)/3,
                       color='#162335', marker='_', s=185, linewidths=2.2, zorder=4)
    ax.set_xticks(range(6), [f'{LABELS[o]}\n{s}' for o in OBJECTIVES for s in ('ordinary', 'shift')], fontsize=8.5)
    ax.set_ylabel('Signed bounded engine loss (lower is better)')
    ax.set_title('A. Final checkpoints: 128 positions per panel', loc='left', fontsize=11, fontweight='bold', pad=13)
    ax.axhline(0, color='#9ca8b6', linewidth=.7)
    ax = axes[1]
    for objective in OBJECTIVES:
        for partition, style in (('diagnostic', '-'), ('train', '--')):
            rows = [r for r in values['nll_curves'] if r['objective'] == objective and r['partition'] == partition]
            ax.plot([r['epoch'] for r in rows], [r['mean_nll'] for r in rows], style,
                    color=COLORS[objective], lw=2, marker='o' if partition == 'diagnostic' else None, markersize=3)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel('Saved checkpoint epoch (all eight shown)')
    ax.set_ylabel('Target NLL (mean of three paired seeds)')
    ax.set_title('B. Fixed subsets: 2,048 roots per partition', loc='left', fontsize=11, fontweight='bold', pad=13)
    ax.legend(handles=[Line2D([0], [0], color='#47566b', linestyle=s, label=l)
                       for s, l in (('-', 'Diagnostic subset'), ('--', 'Training subset'))],
              loc='best', frameon=False, fontsize=8.5)

    ax = axes[2]
    ax.grid(False)
    for y, record in zip((1, 0), values['arena'], strict=True):
        lower, upper = record['score_lower_bound']*100, record['score_upper_bound']*100
        ax.plot([lower, upper], [y, y], color=COLORS['continuation'], lw=5, solid_capstyle='round', zorder=2)
        ax.scatter([lower, upper], [y, y], color=COLORS['continuation'], edgecolors='white', s=45, zorder=3)
        ax.text(2, y+.23, f'{lower:.1f}% to {upper:.1f}% of all {record["games"]} points', fontsize=9, color='#203149')
        counts = record['status_counts']
        ax.text(2, y-.33, f'W {record["wins"]}  /  D {record["draws"]}  /  L {record["losses"]}  /  '
                f'unfinished {counts["unfinished"]}  /  failed {counts["failed"]}', fontsize=8.5,
                color='#a5412e' if counts['failed'] else '#47566b')
    ax.axvline(60, color='#a5412e', linestyle='--', lw=1.2, label='60% lower-bound gate')
    ax.set_yticks([1, 0], ['vs Policy', 'vs Best-value'])
    ax.set_xlim(0, 100)
    ax.set_ylim(-.65, 1.65)
    ax.set_xlabel('Continuation points: unresolved outcome bounds, not confidence intervals', fontsize=8.5)
    ax.set_title('C. Paired games: all 192, actor-only inference', loc='left', fontsize=11, fontweight='bold', pad=13)
    ax.legend(loc='upper right', frameon=False, fontsize=8)

    ax = axes[3]
    records = values['fits']
    bars = ax.bar(range(9), [r['training_seconds']/60 for r in records],
                  color=[COLORS[r['objective']] for r in records], width=.7, zorder=3)
    for bar, record in zip(bars, records, strict=True):
        ax.annotate(f'{record["training_seconds"]:.1f}s', (bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords='offset points', ha='center', fontsize=7.5, color='#40516b')
    ax.set_xticks(range(9), [str(r['seed']) for r in records], fontsize=8)
    for i, objective in enumerate(OBJECTIVES):
        ax.text(3*i+1, -.12, LABELS[objective], transform=ax.get_xaxis_transform(),
                ha='center', fontsize=9, color=COLORS[objective])
    ax.set_ylabel('Training wall time (minutes)')
    ax.set_ylim(0, max(r['training_seconds']/60 for r in records)*1.2+1e-6)
    ax.set_title('D. Every completed fit: recorded MPS wall time', loc='left', fontsize=11, fontweight='bold', pad=13)
    fig.text(.075, .10, 'Engine loss = mean[tanh(unrestricted cp/600) - tanh(chosen cp/600)]; '
             '20,000 Stockfish nodes per call. Signed losses are retained.', fontsize=9, color='#47566b')
    fig.text(.075, .077, 'Timing includes fit updates, journal writes and checkpoint saves; excludes model/optimizer '
             'setup and all evaluation. Shared host, not isolated throughput.', fontsize=9, color='#47566b')
    fig.text(.075, .054, 'Means use every paired seed. Arena bounds keep unfinished and failed games in the denominator; '
             'no imputed draws or discarded games.', fontsize=9, color='#47566b')
    fig.text(.075, .025, 'Exposed historical development evidence. No Elo, broad playing-strength, architectural novelty '
             'or world-model claim.', fontsize=10, color='#203149', fontweight='bold')
    return fig


def render(audit, plan_path, out, *, synthetic_fixture=False):
    summary, plan, binding = load_completed(audit, plan_path, synthetic_fixture=synthetic_fixture)
    out = Path(out)
    require(not out.resolve().is_relative_to(Path(audit).resolve())
            and not out.resolve().is_relative_to(Path(plan_path).parent.resolve()),
            'Outputs must be outside the authenticated evidence directories')
    if out.exists() or out.is_symlink():
        raise FileExistsError('Output directory already exists; choose a new directory')
    images = {}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9.5}):
        fig = figure(summary, synthetic_fixture=synthetic_fixture)
        try:
            for extension in ('png', 'pdf'):
                stream = io.BytesIO()
                fig.savefig(stream, format=extension, dpi=160, facecolor='white')
                images[f'figure.{extension}'] = stream.getvalue()
        finally:
            plt.close(fig)
    manifest = {'version': VERSION, 'status': 'completed', 'synthetic_fixture': synthetic_fixture,
        **binding, 'plot_source_sha256': sha256(Path(__file__).read_bytes()),
        'files': {name: sha256(content) for name, content in images.items()},
        'matplotlib_version': matplotlib.__version__, 'objectives': list(OBJECTIVES), 'seeds': list(SEEDS),
        'source_fit_order': [c['name'] for c in plan['configurations']], 'primary_epoch': 8,
        'fit_count': 9, 'checkpoint_count': 72, 'scheduled_games': 192,
        'continuation_gate': summary['continuation_gate'], 'claim_scope': summary['claim_scope'],
        'new_model_or_engine_calls': 0,
        'validation_scope': 'Authenticated completed audit, full summary coverage and scalar arithmetic. '
                            'This renderer does not repeat the raw-evidence audit or load checkpoints.',
        'aggregation': 'All three paired seeds, arithmetic mean NLL and engine loss. All fits and signed losses '
                       'retained; all scheduled games enter unresolved outcome bounds. No confidence intervals.',
        'timing_scope': 'Recorded MPS training_seconds from each exact fit receipt, including journals/checkpoints, '
                        'excluding setup and evaluation. Shared interactive host.',
        'plotted_values': chart_values(summary)}
    encoded = (json.dumps(manifest, indent=2, allow_nan=False)+'\n').encode()
    out.mkdir(parents=True, exist_ok=False)
    for name, content in {**images, 'provenance.json': encoded}.items():
        with (out/name).open('xb') as destination:
            destination.write(content)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--synthetic-fixture', action='store_true', help='Render explicitly tagged invented test data')
    args = parser.parse_args()
    result = render(args.audit, args.plan, args.out, synthetic_fixture=args.synthetic_fixture)
    print(json.dumps({'status': result['status'], 'synthetic_fixture': result['synthetic_fixture'],
                      'output': str(args.out), 'summary_sha256': result['summary_sha256']}))
