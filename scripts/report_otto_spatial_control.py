"""Present only completed, independently audited autonomous spatial evidence.

This renderer hashes checkpoints and trajectories opaquely. It reads summaries
and streams episode membership only after original completion and independent
audit authentication. No neural, simulator, training or audit execution occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import resource
import signal
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = 'scripts/report_otto_spatial_study.py'
HELPER_PIN = '192e0b8e3f7eb630848aa4df0881c7153d3180a1ca479b1a30ade25af0bface2'
if hashlib.sha256((ROOT/HELPER).read_bytes()).hexdigest() != HELPER_PIN:
    raise ValueError('unchanged report authentication helpers')
_spec = importlib.util.spec_from_file_location('_spatial_control_report_helpers', ROOT/HELPER)
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)
read, write, digest, require, closed, agree = R.read, R.write, R.digest, R.require, R.closed, R.agree
VERSION = 'otto-spatial-control-report-v1'
PLAN_PIN = 'fdbcd35a5a8ac990f1221ceb5e6a661f62e04b5d5cf264df6aca2b9224880c51'
PRODUCER = 'scripts/study_otto_spatial_control.py'
PRODUCER_PIN = 'cf87432697852209d323a3f809758fef5382d229a85092ca8f49380d9d275797'
AUDITOR = 'scripts/audit_otto_spatial_control.py'
AUDITOR_PIN = '85fc1d059407a3e95a9365cb7cfc254ff60b8230dccc7629c427b9615b108baa'
CLOCK, CLOCK_PIN, THREADS = R.CLOCK, R.CLOCK_PIN, R.THREADS
KINDS, SEEDS = R.KINDS, R.SEEDS
ARMS = (*R.ORDER, 'analytic_inbounds')
FIRST = {'lambda3': 16100001, 'lambda4': 16200001, 'lambda5': 16300001}
TIMES = ('init_seconds', 'choose_seconds', 'update_seconds', 'setup_allocation_seconds')
METRICS = ('found', 'steps', *TIMES, 'controller_seconds', 'environment_seconds', 'state_bytes')
LIMITS = {'seconds': 300, 'rss_bytes': 2*1024**3, 'output_bytes': 64*1024**2}
OUTPUTS = {'spatial-control-comparison.png', 'spatial-control-comparison.svg',
           'cells.csv', 'conditions.csv', 'derivation.json'}
GROUPS = {'competence_checks': 18, 'improvement_checks': 48, 'control_competence_checks': 72}
AGGREGATE_KEYS = ('version', 'episodes', 'paired_cases', 'regimes', *GROUPS,
                  'pilot_continuation', 'learned_architecture_advantage_established')
SCOPE = ('Saved autonomous-evidence presentation only. Completed worker and independent audit '
         'supply numerical, neural, native-randomness and timing evidence. This renderer verifies '
         'closed joins and complete cohort membership, but performs no posterior or neural replay, '
         'training, simulator call, statistical test, selection or alteration of any gate.')


def worker_payloads():
    return {'started.json', 'runtime.json', 'inference-setup.json', 'work-contexts.jsonl', 'work.jsonl',
            'summary.json', 'serialization-projection.json', 'native-setup.json',
            'eval-transitions.jsonl', 'eval-episodes.jsonl', 'evaluation.jsonl'}


def authenticate(args, check):
    require(all(getattr(args, k).is_absolute() for k in ('plan', 'run', 'audit', 'terminal')), 'absolute evidence paths')
    require(args.plan_sha256 == PLAN_PIN and digest(args.plan, check)['sha256'] == args.plan_sha256,
            'exact external plan pin before decode')
    plan = read(args.plan)
    require(plan['version'] == 'otto-spatial-control-v1' and plan['mode'] == 'study'
            and plan['status'] == 'frozen_before_execution' and len(plan['sources']) == 219
            and plan['sources'][PRODUCER] == PRODUCER_PIN and plan['sources'][AUDITOR] == AUDITOR_PIN
            and plan['sources'][CLOCK] == CLOCK_PIN and plan['configuration']['arms'] == list(ARMS)
            and plan['configuration']['candidate'] == 'spatial' and plan['configuration']['horizon'] == 2188
            and plan['configuration']['evaluation_first_seeds'] == FIRST, 'fixed all-head autonomous study')
    for name, pin in plan['sources'].items():
        rel = Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts
                and digest(ROOT/rel, check)['sha256'] == pin, 'unchanged source: '+name)
    require(set(plan['qualification']) == {'plan', 'receipt', 'terminal', 'audit'}, 'qualified deployment lineage')
    for item in (*plan['inputs'].values(), *plan['qualification'].values()):
        rel = Path(item['path'])
        require(not rel.is_absolute() and '..' not in rel.parts
                and digest(ROOT/rel, check) == {k: item[k] for k in ('bytes', 'sha256')}, 'unchanged bound input')
    worker = closed(args.run, args.receipt_sha256, worker_payloads(), check)
    audit = closed(args.audit, args.audit_sha256, {'started.json', 'readouts.jsonl', 'summary.json'}, check)
    require(worker['version'] == plan['version'] and worker['mode'] == 'study'
            and worker['plan_sha256'] == args.plan_sha256 and worker['sources'] == plan['sources']
            and worker['inputs'] == plan['inputs'] and worker['limits'] == plan['limits']
            and worker['completed_episodes'] == 1152 and worker['completed_fits'] == 0
            and worker['pending'] == [] and worker['parity_passed'] is True
            and worker['external_model_calls'] == 0, 'original complete autonomous cohort')
    caps = {'model_load': 15, 'native_reset': 1155, 'native_step': 2520576,
            'value_forward': 2363040, 'analytic_choose': 157536}
    require(plan['expected_calls'] == caps and set(worker['calls']) == set(caps), 'fixed operation coverage')
    calls = {}
    for channel, cap in caps.items():
        call = worker['calls'][channel]
        require(type(call['returned']) is int and 0 < call['returned'] <= cap
                and call['attempted'] == call['returned'], 'all attempted operations returned')
        calls[channel] = call['returned']
    require(calls['model_load'] == 15 and calls['native_reset'] == 1155
            and calls['native_step'] == calls['value_forward']+calls['analytic_choose'], 'complete calls and reset count')
    require(audit['version'] == 'otto-spatial-control-saved-audit-v1' and audit['mode'] == 'study'
            and audit['agreement'] is True and audit['source']['sha256'] == AUDITOR_PIN
            and audit['producer_source_sha256'] == PRODUCER_PIN and audit['plan_sha256'] == args.plan_sha256
            and audit['worker_sha256'] == args.receipt_sha256 and audit['terminal_sha256'] == args.terminal_sha256
            and audit['pending'] is None and audit['readout_attempts'] == audit['readout_returns']
            == audit['saved_checkpoint_readout_calls'] == calls['value_forward']
            and audit['saved_network_rows'] == 16*calls['value_forward']
            and audit['training_calls'] == audit['simulator_calls'] == audit['remote_model_calls'] == 0,
            'complete independent audit linked to the original run')
    require(digest(args.terminal, check)['sha256'] == args.terminal_sha256, 'external original supervisor pin')
    terminal = read(args.terminal)
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
            and terminal['group_absent'] is True and terminal['error'] is None and terminal['clock_error'] is None
            and terminal['timing_available'] is True and terminal['cleanup']['errors'] == []
            and terminal['cleanup']['reaped'] is True and terminal['cleanup']['group_absent'] is True,
            'successful original process and cleanup')
    started = read(args.run/'started.json')
    launch, request = started['launch'], started['request']
    require(all(terminal[k] == v for k, v in launch.items()), 'same original launch and terminal')
    require(request == {'plan': str(args.plan), 'plan_sha256': args.plan_sha256,
                        'output': str(args.run), 'supervision': request['supervision']}, 'original worker request')
    supervision = Path(request['supervision'])
    require(digest(supervision, check)['sha256'] == worker['supervision_sha256'] and read(supervision) == launch,
            'actual original supervision file')
    command = list(launch['command'])
    if command[1:2] == ['-u']:
        command.pop(1)
    require(command == [plan['python_executable'], str(ROOT/PRODUCER), '--plan', str(args.plan),
                        '--plan-sha256', args.plan_sha256, '--output', str(args.run), '--supervision', str(supervision)]
            and launch['cwd'] == str(ROOT) and launch['pid'] == launch['pgid'] != launch['parent_pid'],
            'exact original executable and process')
    require(launch['cap_seconds'] == plan['limits']['native_seconds'] == 21600
            and launch['deadline_ns'] == launch['started_ns']+21600*10**9
            and launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns']
            and worker['started_ns'] == started['started_ns'] and worker['clock_backend'] == launch['clock_backend']
            and launch['clock_source_sha256'] == CLOCK_PIN
            and launch['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py']
            and worker['wall_seconds'] == (worker['finished_ns']-worker['started_ns'])/1e9
            and terminal['elapsed_ns'] == terminal['finished_ns']-terminal['started_ns']
            and terminal['wall_seconds'] == terminal['elapsed_ns']/1e9, 'original native timing enclosure')
    audit_request = read(args.audit/'started.json')['request']
    require(all(audit_request[k] == str(getattr(args, k)) for k in
                ('plan', 'plan_sha256', 'run', 'receipt_sha256', 'terminal', 'terminal_sha256'))
            and audit_request['output'] == str(args.audit) and audit_request['mode'] == 'study', 'original audit request')
    return worker, audit, terminal


def expected_cohort():
    for ri, (regime, first) in enumerate(FIRST.items()):
        for case in range(24):
            offset = (ri*24+case) % 16
            for arm in ARMS[offset:]+ARMS[:offset]:
                yield regime, first+case, case//3, 1+case % 3, arm


def condition_names(group):
    if group == 'improvement_checks':
        return [f'{r}.{k}.{suffix}' for r in FIRST for k in KINDS[1:]
                for suffix in ('success', 'moves', 'positive_blocks', 'cost')]
    kinds = ('spatial',) if group == 'competence_checks' else KINDS[1:]
    return [f'{r}.{k}.{s}.{suffix}' for r in FIRST for k in kinds for s in SEEDS for suffix in ('success', 'moves')]


def condition_counts(result):
    counts = {}
    for group, count in GROUPS.items():
        records = result[group]
        require(len(records) == count and [r['name'] for r in records] == condition_names(group)
                and all(type(r['passes']) is bool for r in records), 'exact complete criterion identities')
        counts[group] = {'passed': sum(r['passes'] for r in records), 'total': count}
    decision = all(r['passes'] for key in ('competence_checks', 'improvement_checks') for r in result[key])
    require(type(result['pilot_continuation']) is bool and result['pilot_continuation'] == decision,
            'unchanged exact frozen screen')
    return counts


def membership(path, calls, check):
    expected, raw, strata, steps = iter(expected_cohort()), Counter(), Counter(), Counter()
    n = 0
    with path.open() as stream:
        for line in stream:
            check()
            row = json.loads(line)
            identity = tuple(row[k] for k in ('regime', 'seed', 'block', 'initial_hit', 'arm'))
            require(identity == next(expected, None), 'complete ordered episode membership')
            require(type(row['steps']) is int and 1 <= row['steps'] <= 2188 and type(row['found']) is bool
                    and (row['found'] or row['steps'] == 2188) and row['updates'] == row['steps']
                    and row['blocked_steps'] == 0 and row['final_update_assimilated'] is True, 'retained full found/censored episode')
            raw[row['regime'], row['arm']] += int(row['found'])
            strata[row['regime'], row['arm'], row['initial_hit']] += 1
            steps['analytic_choose' if row['arm'] == 'analytic_inbounds' else 'value_forward'] += row['steps']
            n += 1
    require(n == 1152 and next(expected, None) is None and all(steps[k] == calls[k] for k in steps), 'complete episode/call coverage')
    return raw, strata


def values(args, check):
    worker, audit, terminal = authenticate(args, check)
    saved, result = read(args.run/'summary.json'), read(args.audit/'summary.json')
    require(result['agreement'] is True and result['audit_version'] == audit['version'], 'audited summary identity')
    for key in AGGREGATE_KEYS:
        require(agree(saved[key], result[key]), 'independently audited aggregate: '+key)
    require(result['episodes'] == 1152 and result['paired_cases'] == 72 and set(result['regimes']) == set(FIRST)
            and result['learned_architecture_advantage_established'] is False, 'all settings without architecture promotion')
    counts = condition_counts(result)
    require(worker['pilot_continuation'] == result['pilot_continuation'], 'same original candidate decision')
    require(agree(saved['amortization']['scenarios'], result['amortization'])
            and agree(saved['amortization']['training_reference'], result['costs']['training_reference'])
            and agree(saved['inference_setup'], result['costs']['inference_setup']), 'audited complete cost scopes')
    calls = {k: v['returned'] for k, v in worker['calls'].items()}
    require(result['actual_work'] == calls and saved['calls'] == worker['calls'], 'same actual operation totals')
    raw, strata = membership(args.run/'evaluation.jsonl', calls, check)
    cells = []
    for regime, panel in result['regimes'].items():
        require(set(panel['means']) == set(panel['raw_counts']) == set(ARMS)
                and set(panel['family_means']) == set(KINDS) and len(panel['blocks']) == 8
                and set(panel['strata']) == {'1', '2', '3'}, 'all families, seeds, blocks and strata')
        for arm in ARMS:
            metrics = panel['means'][arm]
            require(all(type(metrics[k]) in (float, int) and math.isfinite(metrics[k]) and metrics[k] >= 0 for k in METRICS)
                    and 0 <= metrics['found'] <= 1 and 1 <= metrics['steps'] <= 2188 and metrics['controller_seconds'] > 0
                    and math.isclose(metrics['controller_seconds'], math.fsum(metrics[k] for k in TIMES),
                                     rel_tol=1e-10, abs_tol=1e-10), 'finite complete reported means')
            require(panel['raw_counts'][arm] == {'found': raw[regime, arm], 'episodes': 24}
                    and all(strata[regime, arm, h] == 8 for h in (1, 2, 3)), 'raw counts and weighted denominators distinct')
            kind, seed = arm.split('@') if '@' in arm else ('analytic_inbounds', '')
            cells.append({'regime': regime, 'arm': arm, 'family': kind, 'fit_seed': seed,
                          'raw_found': raw[regime, arm], 'episodes': 24, 'cases_per_hit': 8, **metrics})
    return result, cells, counts, worker, audit, terminal


def tables(out, cells, result):
    with (out/'cells.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    with (out/'conditions.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['group', 'candidate_gate', 'name', 'value', 'threshold', 'passes'])
        writer.writeheader()
        for group in GROUPS:
            for row in result[group]:
                writer.writerow({'group': group, 'candidate_gate': group != 'control_competence_checks', **row})


def render(out, result, cells, counts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({'font.size': 11, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(3, 3, figsize=(20, 14), layout='constrained')
    labels = ('Spatial\nsums', 'Neighbor-\nfree', 'Ordinary\nCNN', 'Dense128', 'Statistics', 'Analytic\ninbounds')
    specs = (('found', 'Weighted success', 'linear'), ('steps', 'Mean capped moves', 'linear'),
             ('controller_seconds', 'Complete controller seconds/search', 'log'))
    bounds = []
    for ri, (metric, ylabel, scale) in enumerate(specs):
        values_all = [r[metric] for r in cells]
        lower, upper = ((0., 1.05) if metric == 'found' else (0., max(values_all)*1.12)
                        if scale == 'linear' else (min(values_all)/1.6, max(values_all)*1.6))
        for ci, regime in enumerate(FIRST):
            axis = axes[ri, ci]
            panel = result['regimes'][regime]
            for si, (seed, color, marker) in enumerate(zip(SEEDS, ('#2468a2', '#c06c22', '#448b50'), ('o', 's', '^'), strict=True)):
                axis.scatter([i+(si-1)*.16 for i in range(5)],
                             [panel['means'][f'{kind}@{seed}'][metric] for kind in KINDS],
                             color=color, marker=marker, s=61, label=f'Fit seed {seed}', zorder=3)
            for ki, kind in enumerate(KINDS):
                value = panel['family_means'][kind][metric]
                axis.plot([ki-.27, ki+.27], [value, value], color='#252525', linewidth=1.8,
                          label='Equal-fit-seed mean' if ki == 0 else None, zorder=2)
            axis.scatter([5], [panel['means']['analytic_inbounds'][metric]], marker='D', s=66,
                         color='#252525', label='Analytic control', zorder=3)
            axis.set(xticks=range(6), xticklabels=labels, xlim=(-.5, 5.5), yscale=scale, ylim=(lower, upper))
            axis.set_ylabel(ylabel+(' (log scale)' if scale == 'log' else ''))
            if metric == 'found':
                axis.yaxis.set_major_formatter(PercentFormatter(1.))
                axis.set_title(f'{regime}: '+('training-supported setting' if regime != 'lambda5' else 'supplied-model parameter shift'), fontsize=12)
            axis.grid(axis='y', which='both', alpha=.18)
            axis.tick_params(axis='x', labelsize=10)
            bounds.append({'regime': regime, 'metric': metric, 'scale': scale, 'limits': [lower, upper]})
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=5, frameon=False, fontsize=11)
    competence, improvement, descriptive = (counts[k]['passed'] for k in GROUPS)
    status = 'PASS' if result['pilot_continuation'] else 'NOT PASSED'
    fig.suptitle('OTTO spatial controllers: all 15 saved heads and analytic control\n'
                 f'Prespecified spatial screen: {status} ({competence+improvement}/66) | '
                 f'Competence {competence}/18; paired controls {improvement}/48 | '
                 f'Other-head competence {descriptive}/72 (descriptive)', fontsize=15)
    fig.supxlabel('72 paired cases; 24 per setting, 8 per initial-hit stratum. Mixture-weighted means; failures retain all 2,188 moves.\n'
                  'Cost includes initialization, branches/features/readout/selection, every update and allocated head/module load; '
                  'excludes measured artifact I/O.\n'
                  'Native execution, prior fitting, qualification and audit are separate. One timing pass; no architectural advantage claim.', fontsize=10)
    for suffix in ('png', 'svg'):
        fig.savefig(out/f'spatial-control-comparison.{suffix}', dpi=180)
    plt.close(fig)
    return bounds


def execute(args):
    out = args.output
    require(out.is_absolute() and not out.exists() and not any(p.is_symlink() for p in out.parents), 'exclusive nonsymlink output')
    out.mkdir(parents=True, exist_ok=False)
    start = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    try:
        require(digest(ROOT/CLOCK)['sha256'] == CLOCK_PIN, 'qualified suspend-inclusive clock')
        spec = importlib.util.spec_from_file_location('_spatial_control_report_clock', ROOT/CLOCK)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        clock = module.SuspendClock()
        start = clock.now_ns()
        def check():
            require(clock.now_ns()-start < LIMITS['seconds']*10**9, 'native report deadline')
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024) <= LIMITS['rss_bytes'], 'report RSS cap')
            require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'report output cap')
        def alarm(*_):
            raise TimeoutError('saved spatial control report deadline')
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, LIMITS['seconds'])
        require(all(os.environ.get(k) == v for k, v in THREADS.items()), 'single CPU report settings')
        source = digest(Path(__file__), check)
        result, cells, counts, worker, audit, terminal = values(args, check)
        tables(out, cells, result)
        bounds = render(out, result, cells, counts)
        check()
        manifest = {'version': VERSION, 'scope': SCOPE, 'source': source, 'helper_sha256': HELPER_PIN,
            'request': {k: str(v) for k, v in vars(args).items()}, 'cells': cells, 'regimes': result['regimes'],
            'criteria': {k: result[k] for k in GROUPS}, 'criterion_counts': counts, 'figure_panels': bounds,
            'amortization': result['amortization'], 'pilot_continuation': result['pilot_continuation'],
            'learned_architecture_advantage_established': False, 'episode_count': 1152, 'paired_cases': 72,
            'derivations': {'cells': 'Audited regime.means per arm; raw found counts independently joined to all1152 episode identities.',
                            'family_marks': 'Audited equal mean of three fixed fitting seeds; no outcome sorting.',
                            'success': 'Eight cases per initial-hit stratum, then authenticated positive-hit mixture.',
                            'moves': 'Same weighting, full2188 cap for every failed search.',
                            'conditions': 'Exact audited138 records; only18+48 define candidate screen, other72 descriptive.'},
            'cost_scope': {'audit_cost_record': result['costs'], 'worker_seconds': worker['wall_seconds'],
                           'parent_seconds': terminal['wall_seconds'], 'separate_audit_seconds': audit['wall_seconds'],
                           'scope': 'Parent encloses worker; do not sum them. Plotted controller cost includes head load/72 and module setup/1080, '
                                    'excluding measured artifact I/O. Prior fit/preparation are separately amortized at H1/100/10000. '
                                    'Final training diagnostics, qualification, environment and audit are separate scopes.'},
            'audit_scope': {'saved_readouts': audit['saved_checkpoint_readout_calls'], 'saved_branch_rows': audit['saved_network_rows'],
                            'scope': 'Audit shares qualified NumPy neural algebra; filtering, branches, choices and metrics are independently checked. '
                                     'Original Torch execution, native randomness and measured time remain authenticated witnesses.'},
            'input_payloads': {'worker': worker['files'], 'audit': audit['files']},
            'model_calls': 0, 'training_calls': 0, 'simulator_calls': 0}
        write(out/'derivation.json', manifest)
        authenticate(args, check)
        require(digest(Path(__file__), check) == source and digest(ROOT/HELPER, check)['sha256'] == HELPER_PIN, 'unchanged presentation source')
        require({p.name for p in out.iterdir()} == OUTPUTS, 'exact report payload closure')
        files = {name: digest(out/name, check) for name in sorted(OUTPUTS)}
        finish = clock.now_ns()
        require(finish-start < LIMITS['seconds']*10**9, 'strict final deadline')
        write(out/'receipt.json', {'version': VERSION, 'status': 'completed', 'source': source,
              'request': manifest['request'], 'files': files, 'scope': SCOPE, 'limits': LIMITS,
              'plan_sha256': args.plan_sha256, 'worker_sha256': args.receipt_sha256,
              'terminal_sha256': args.terminal_sha256, 'audit_sha256': args.audit_sha256,
              'clock_backend': clock.backend, 'started_ns': start, 'finished_ns': finish, 'wall_seconds': (finish-start)/1e9,
              'model_calls': 0, 'training_calls': 0, 'simulator_calls': 0})
        check()
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/'receipt.json').exists():
                (out/'receipt.json').rename(out/'invalid-completed-receipt.json')
            write(out/'failed.json', {'version': VERSION, 'status': 'failed', 'error': repr(error),
                  'started_ns': start, 'scope': SCOPE})
        except BaseException as secondary:  # noqa: BLE001 - Preserve the original error if failure publication also fails.
            error.add_note(f'Failure publication: {secondary!r}')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'run', 'audit', 'terminal', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('plan-sha256', 'receipt-sha256', 'audit-sha256', 'terminal-sha256'):
        parser.add_argument('--'+name, required=True)
    execute(parser.parse_args())
