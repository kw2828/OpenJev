"""Plot only completed, independently audited spatial-study evidence.

No scientific arrays, checkpoints or models are decoded. Payloads are hashed
opaquely; numeric JSON summaries are read only after all completion joins pass.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'otto-spatial-study-report-v1'
PLAN_PIN = '2f204b5f3aa98f7aa54e8537ed6df16110596944f54f4b7f63fd39053d9a9a62'
PRODUCER = 'scripts/study_otto_spatial.py'
PRODUCER_PIN = '548c21ab961400313e8ff77697ea8a4ee78bffd0be29227506d0236c0cbbd212'
AUDITOR = 'scripts/audit_otto_spatial_study.py'
AUDITOR_PIN = '55abea140c9273fba11f2a100021a0b4d0cf824c79e74bdffd4d00b248ebccf0'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
KINDS = ('spatial', 'neighbor_free', 'cnn', 'dense128', 'statistics')
SEEDS = (10101, 10102, 10103)
ORDER = tuple(f'{kind}@{seed}' for seed in SEEDS for kind in KINDS)
LABELS = ('Spatial sums', 'Neighbor-free', 'Ordinary CNN', 'Dense128', 'Statistics')
THREADS = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
COUNTS = {'model_initialization': 15, 'checkpoint_export': 30, 'optimizer_initialization': 15,
          'training_forward': 52800, 'backward': 52800, 'optimizer_update': 52800,
          'deployment_construction': 15, 'parity_restore': 15, 'numpy_prediction': 6300, 'torch_prediction': 6300}
LIMITS = {'seconds': 300, 'rss_bytes': 2*1024**3, 'output_bytes': 64*1024**2}
OUTPUTS = {'spatial-study-comparison.png', 'spatial-study-comparison.svg', 'derivation.json'}
SCOPE = ('Saved scalar-results presentation only. Numerical and training evidence is inherited from the '
         'pinned worker and completed saved-model audit. This renderer performs no model readout, optimization, '
         'simulator call, new statistical test, selection, scalar admission or autonomous evaluation.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def digest(path, check=lambda: None):
    path = Path(path)
    require(path.is_absolute() and path.is_file()
            and not any(p.is_symlink() for p in (path, *path.parents)), 'absolute regular nonsymlink input')
    result, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while block := stream.read(1024**2):
            check()
            result.update(block)
            size += len(block)
    return {'sha256': result.hexdigest(), 'bytes': size}


def worker_payloads():
    names = {'started.json', 'runtime.json', 'preparation.json', 'summary.json', 'work.jsonl',
             'initializations.jsonl', 'epoch-orders.jsonl', 'updates.jsonl', 'fit-curves.jsonl',
             'fits.jsonl', 'parity.jsonl'}
    names.update(f'{phase}-{kind}-{seed}.npz' for seed in SEEDS for kind in KINDS
                 for phase in ('initial', 'final', 'predictions'))
    return names


def closed(directory, pin, names, check):
    require(directory.is_absolute() and directory.is_dir(), 'absolute evidence directory')
    require(digest(directory/'receipt.json', check)['sha256'] == pin, 'external receipt pin')
    receipt = read(directory/'receipt.json')
    require(receipt['status'] == 'completed' and set(receipt['files']) == names
            and {p.name for p in directory.iterdir()} == names | {'receipt.json'}, 'exact completed evidence closure')
    for name in sorted(names):
        require(digest(directory/name, check) == receipt['files'][name], 'unchanged closed payload: '+name)
    return receipt


def authenticate(args, check):
    require(all(getattr(args, key).is_absolute() for key in ('plan', 'run', 'audit', 'terminal')), 'absolute input paths')
    require(args.plan_sha256 == PLAN_PIN and digest(args.plan, check)['sha256'] == args.plan_sha256,
            'external frozen spatial study plan pin')
    plan = read(args.plan)
    require(plan['version'] == 'otto-spatial-study-v1' and plan['mode'] == 'study'
            and plan['status'] == 'frozen_before_execution' and plan['sources'][PRODUCER] == PRODUCER_PIN
            and plan['sources'][AUDITOR] == AUDITOR_PIN and plan['sources'][CLOCK] == CLOCK_PIN
            and plan['expected_calls'] == COUNTS, 'fixed study and qualified source identities')
    for name, pin in plan['sources'].items():
        rel = Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts
                and digest(ROOT/rel, check)['sha256'] == pin, 'unchanged scientific source: '+name)
    for item in (*plan['inputs'].values(), *plan['qualification'].values()):
        rel = Path(item['path'])
        require(not rel.is_absolute() and '..' not in rel.parts
                and digest(ROOT/rel, check) == {k: item[k] for k in ('bytes', 'sha256')}, 'unchanged input descriptor')
    worker = closed(args.run, args.receipt_sha256, worker_payloads(), check)
    audit = closed(args.audit, args.audit_sha256, {'started.json', 'readouts.jsonl', 'summary.json'}, check)
    require(worker['version'] == plan['version'] and worker['mode'] == 'study'
            and worker['plan_sha256'] == args.plan_sha256 and worker['sources'] == plan['sources']
            and worker['inputs'] == plan['inputs'] and worker['qualification'] == plan['qualification']
            and worker['limits'] == plan['limits'] and worker['completed_fits'] == 15
            and worker['pending'] == [] and worker['parity_passed'] is True and worker['initial_pairing_passed'] is True
            and set(worker['calls']) == set(COUNTS)
            and all(worker['calls'][k]['attempted'] == worker['calls'][k]['returned'] == n for k, n in COUNTS.items())
            and worker['native_steps'] == worker['native_resets'] == worker['external_model_calls'] == 0,
            'all fifteen technically completed final fits')
    require(audit['version'] == 'otto-spatial-study-saved-audit-v1' and audit['agreement'] is True
            and audit['source']['sha256'] == AUDITOR_PIN and audit['producer_source_sha256'] == PRODUCER_PIN
            and audit['plan_sha256'] == args.plan_sha256 and audit['worker_sha256'] == args.receipt_sha256
            and audit['terminal_sha256'] == args.terminal_sha256 and audit['pending'] is None
            and audit['independent_readout_attempts'] == audit['independent_readout_returns'] == 6300
            and audit['independent_readout_rows'] == 100470
            and audit['optimizer_calls'] == audit['native_steps'] == audit['external_model_calls'] == 0,
            'completed linked independent saved-model audit')
    require(digest(args.terminal, check)['sha256'] == args.terminal_sha256, 'external original terminal pin')
    terminal = read(args.terminal)
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
            and terminal['group_absent'] is True and terminal['error'] is None and terminal['clock_error'] is None
            and terminal['timing_available'] is True and terminal['cleanup']['errors'] == []
            and terminal['cleanup']['reaped'] is True and terminal['cleanup']['group_absent'] is True,
            'successful original worker supervisor')
    started = read(args.run/'started.json')
    launch, request = started['launch'], started['request']
    require(all(terminal[key] == value for key, value in launch.items()), 'same parent launch and terminal')
    require(request == {'plan': str(args.plan), 'plan_sha256': args.plan_sha256,
                        'output': str(args.run), 'supervision': request['supervision']}, 'original worker request')
    supervision = Path(request['supervision'])
    require(digest(supervision, check)['sha256'] == worker['supervision_sha256'] and read(supervision) == launch,
            'actual supervision file identity')
    command = list(launch['command'])
    if command[1:2] == ['-u']:
        command.pop(1)
    require(command == [plan['python_executable'], str(ROOT/PRODUCER), '--plan', str(args.plan),
                        '--plan-sha256', args.plan_sha256, '--output', str(args.run), '--supervision', str(supervision)]
            and launch['cwd'] == str(ROOT) and launch['pid'] == launch['pgid'] != launch['parent_pid'],
            'exact original process')
    require(launch['cap_seconds'] == plan['limits']['native_seconds'] == 7200
            and launch['deadline_ns'] == launch['started_ns']+7200*10**9
            and launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns']
            and worker['started_ns'] == started['started_ns'] and worker['clock_backend'] == launch['clock_backend']
            and worker['wall_seconds'] == (worker['finished_ns']-worker['started_ns'])/1e9
            and terminal['elapsed_ns'] == terminal['finished_ns']-terminal['started_ns']
            and terminal['wall_seconds'] == terminal['elapsed_ns']/1e9, 'native process timing enclosure')
    audit_request = read(args.audit/'started.json')['request']
    require(all(audit_request[k] == str(getattr(args, k)) for k in
                ('plan', 'plan_sha256', 'run', 'receipt_sha256', 'terminal', 'terminal_sha256'))
            and audit_request['output'] == str(args.audit), 'audit paths and external pins')
    return worker, audit, terminal


def agree(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(agree(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(agree(a, b) for a, b in zip(actual, expected, strict=True))
    if type(expected) is float:
        return type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)
    return type(actual) is type(expected) and actual == expected


def values(args, check):
    worker, audit, terminal = authenticate(args, check)
    # No outcome JSON is decoded before successful original completion and audit.
    summary, published = read(args.audit/'summary.json'), read(args.run/'summary.json')
    require(summary['agreement'] is True and summary['audit_version'] == audit['version'], 'audited summary identity')
    for key in published.keys()-{'scope'}:
        require(agree(published[key], summary[key]), 'audited producer value: '+key)
    require(summary['fits'] == list(ORDER) and set(summary['metrics']) == set(summary['training_costs'])
            == set(summary['diagnostic_costs']) == set(ORDER) and set(summary['family_means']) == set(KINDS)
            and summary['rows'] == {'train': 5589, 'valid': 1109} and summary['training_updates'] == 52800
            and summary['scalar_admission_gate'] is None and summary['alias_floor'] is None
            and summary['learned_architecture_advantage_established'] is False, 'all models retained without scalar admission')
    with (args.run/'fits.jsonl').open() as stream:
        fits = [json.loads(line) for line in stream]
    require([f['fit_id'] for f in fits] == list(ORDER), 'complete fit records')
    rows = []
    for index, fit in enumerate(fits):
        fit_id = fit['fit_id']
        require(agree(fit['metrics'], summary['metrics'][fit_id])
                and fit['fit_seconds'] == summary['training_costs'][fit_id]
                and fit['diagnostic_seconds'] == summary['diagnostic_costs'][fit_id], 'audited metrics and observed intervals')
        for split in ('train', 'valid'):
            cell = fit['metrics'][split]
            require(cell['rows'] == summary['rows'][split] and type(cell['negative_predictions']) is int
                    and 0 <= cell['negative_predictions'] <= cell['rows']
                    and all(type(cell[k]) in (float, int) and math.isfinite(cell[k]) for k in
                            ('mse_normalized', 'mae_physical', 'minimum_normalized', 'maximum_normalized'))
                    and cell['mse_normalized'] >= 0 and cell['mae_physical'] >= 0, 'finite scalar metrics')
        costs = {key: fit[key] for key in ('training_seconds', 'fit_seconds', 'diagnostic_seconds',
                                         'deployment_setup_seconds', 'prediction_seconds')}
        require(all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in costs.values()), 'finite observed costs')
        require(costs['training_seconds'] <= costs['fit_seconds']
                and math.isclose(costs['diagnostic_seconds'], costs['deployment_setup_seconds']+costs['prediction_seconds'],
                                 rel_tol=1e-10, abs_tol=1e-10), 'nested and disjoint cost scopes')
        rows.append({'fit_id': fit_id, 'kind': fit['kind'], 'seed': fit['seed'], 'fit_row_index': index,
                     'train': summary['metrics'][fit_id]['train'], 'valid': summary['metrics'][fit_id]['valid'], **costs})
    return summary, rows, worker, audit, terminal


def render(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    specs = (('TRAIN mean squared error', 'Normalized return squared', lambda r: r['train']['mse_normalized']),
             ('Exposed VALID mean squared error', 'Normalized return squared', lambda r: r['valid']['mse_normalized']),
             ('Fit interval, including training and checkpointing', 'Seconds per fit', lambda r: r['fit_seconds']),
             ('Final diagnostics, including setup and both backends', 'Seconds per fit', lambda r: r['diagnostic_seconds']))
    bounds = []
    for axis, (title, unit, getter) in zip(axes.flat, specs, strict=True):
        largest = max(getter(row) for row in rows)
        for j, (seed, color, marker) in enumerate(zip(SEEDS, ('#2468a2', '#c06c22', '#448b50'), ('o', 's', '^'), strict=True)):
            selected = [next(row for row in rows if row['kind'] == kind and row['seed'] == seed) for kind in KINDS]
            axis.scatter([i+(j-1)*.16 for i in range(5)], [getter(row) for row in selected],
                         color=color, marker=marker, s=65, label=f'Seed {seed}', zorder=3)
        for i, kind in enumerate(KINDS):
            mean = math.fsum(getter(row) for row in rows if row['kind'] == kind)/3
            axis.plot([i-.27, i+.27], [mean, mean], color='#252525', linewidth=1.8,
                      label='Equal-seed mean' if i == 0 else None, zorder=2)
        axis.set(title=title, ylabel=unit, xticks=range(5), xticklabels=LABELS, xlim=(-.55, 4.55),
                 ylim=(0, largest*1.2 if largest > 0 else 1.0))
        axis.grid(axis='y', alpha=.2)
        axis.tick_params(labelsize=10)
        axis.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4), useOffset=False)
        bounds.append({'title': title, 'units': unit, 'scale': 'linear', 'y_limits': list(axis.get_ylim())})
    axes[0, 0].legend(loc='best', fontsize=9, framealpha=.85)
    fig.suptitle('OTTO spatial value learning: all 15 fixed final fits\nScalar regression only; autonomous utility is untested for these heads', fontsize=15)
    fig.supxlabel('Paired seeds; common data, MC targets and 80 epochs. VALID is descriptive, with no winner selection.\n'
                  'Fit and diagnostic intervals are disjoint; shared preparation is separate. One CPU timing pass, not controller latency.', fontsize=10)
    for suffix in ('png', 'svg'):
        fig.savefig(out/f'spatial-study-comparison.{suffix}', dpi=180)
    plt.close(fig)
    return bounds


def alarm(*_):
    raise TimeoutError('saved spatial report deadline')


def execute(args):
    out = args.output
    require(out.is_absolute() and not out.exists() and not any(p.is_symlink() for p in out.parents), 'exclusive report output')
    out.mkdir(parents=True, exist_ok=False)
    start = None
    try:
        require(digest(ROOT/CLOCK)['sha256'] == CLOCK_PIN, 'qualified suspend-inclusive clock')
        spec = importlib.util.spec_from_file_location('_spatial_report_clock', ROOT/CLOCK)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        clock = module.SuspendClock()
        start = clock.now_ns()
        def check():
            require(clock.now_ns()-start < LIMITS['seconds']*10**9, 'native report deadline')
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024) <= LIMITS['rss_bytes'], 'report RSS cap')
            require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'report output cap')
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, LIMITS['seconds'])
        require(all(os.environ.get(k) == v for k, v in THREADS.items()), 'single CPU report thread settings')
        source = digest(Path(__file__), check)
        summary, rows, worker, audit, terminal = values(args, check)
        bounds = render(out, rows)
        check()
        manifest = {'version': VERSION, 'scope': SCOPE, 'source': source,
            'request': {k: str(v) for k, v in vars(args).items()}, 'ordered_fit_ids': list(ORDER), 'fits': rows,
            'family_means': summary['family_means'], 'figure_panels': bounds,
            'derivations': {'train_mse': 'audit summary.metrics[fit_id].train.mse_normalized',
                            'valid_mse': 'audit summary.metrics[fit_id].valid.mse_normalized',
                            'fit_seconds': 'audit summary.training_costs[fit_id]',
                            'diagnostic_seconds': 'audit summary.diagnostic_costs[fit_id]',
                            'mean_marks': 'math.fsum(three paired fit values)/3; no outcome-based sorting'},
            'cost_scope': {'fit': 'Initialization, optimizer construction, all training, journals and initial/final checkpoints; training_seconds is a subset.',
                           'diagnostics': 'Deployment construction and Torch64 restore plus both full final scalar passes, metric computation and saved predictions.',
                           'preparation_seconds': summary['preparation_seconds'],
                           'worker_seconds': worker['wall_seconds'], 'parent_seconds': terminal['wall_seconds'],
                           'separate_audit_seconds': audit['wall_seconds'], 'nested_parent_worker': True,
                           'scope': 'Parent encloses worker; do not add them. Renderer time is separate. These are observed one-pass costs, not controller latency.'},
            'worker_peak_rss_bytes': worker['peak_rss_bytes'], 'audit_peak_rss_bytes': audit['peak_rss_bytes'],
            'audit_readout_returns': audit['independent_readout_returns'], 'audit_readout_rows': audit['independent_readout_rows'],
            'input_payloads': {'worker': worker['files'], 'audit': audit['files']},
            'scalar_admission_gate': None, 'alias_floor': None, 'autonomous_evaluation_performed': False,
            'model_calls': 0, 'training_calls': 0, 'simulator_calls': 0}
        write(out/'derivation.json', manifest)
        authenticate(args, check)
        require(digest(Path(__file__), check) == source, 'unchanged report source')
        require({p.name for p in out.iterdir()} == OUTPUTS, 'exact report payload closure')
        files = {name: digest(out/name, check) for name in sorted(OUTPUTS)}
        finish = clock.now_ns()
        require(finish-start < LIMITS['seconds']*10**9, 'strict final report deadline')
        write(out/'receipt.json', {'version': VERSION, 'status': 'completed', 'source': source,
              'request': manifest['request'], 'files': files, 'scope': SCOPE, 'limits': LIMITS,
              'clock_backend': clock.backend, 'started_ns': start, 'finished_ns': finish, 'wall_seconds': (finish-start)/1e9,
              'model_calls': 0, 'training_calls': 0, 'simulator_calls': 0})
        check()
        signal.setitimer(signal.ITIMER_REAL, 0)
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/'receipt.json').exists():
                (out/'receipt.json').rename(out/'invalid-completed-receipt.json')
            write(out/'failed.json', {'version': VERSION, 'status': 'failed', 'error': repr(error),
                  'started_ns': start, 'wall_seconds': None, 'scope': SCOPE})
        except BaseException as secondary:  # noqa: BLE001 - Preserve the original error if evidence publication fails.
            error.add_note(f'Failure publication: {secondary!r}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'run', 'audit', 'terminal', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    for key in ('plan-sha256', 'receipt-sha256', 'audit-sha256', 'terminal-sha256'):
        parser.add_argument('--'+key, required=True)
    execute(parser.parse_args())
