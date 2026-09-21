"""Publish a complete, audited scalar-conditioning screen without model calls."""
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
AUDITOR = 'scripts/audit_otto_conditioning.py'
AUDITOR_PIN = '28259542e97940c3dd9482dae4089a878dd946ca22c8adf352ed70aff6a1e51f'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
VERSION = 'otto-conditioning-report-v1'
SEEDS, KINDS = (10101, 10102, 10103), ('gain1', 'gain53')
LIMITS = {'seconds': 120, 'rss_bytes': 2*1024**3, 'output_bytes': 32*1024**2}
SCOPE = ('Saved aggregate presentation only. No new numerical model replay, training, simulator or '
         'statistical test. Independent numerical agreement is inherited from the pinned completed audit.')


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
    require(path.is_file() and not path.is_symlink() and not any(p.is_symlink() for p in path.parents),
            'regular input with no symlink ancestors')
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024**2):
            hasher.update(block)
            check()
    return {'bytes': path.stat().st_size, 'sha256': hasher.hexdigest()}


def closed(directory, pin, names, check):
    require(digest(directory/'receipt.json', check)['sha256'] == pin, 'external receipt pin')
    receipt = read(directory/'receipt.json')
    require(receipt['status'] == 'completed' and set(receipt['files']) == names
            and {p.name for p in directory.iterdir()} == names | {'receipt.json'}, 'complete exact output closure')
    for name in names:
        require(digest(directory/name, check) == receipt['files'][name], 'closed payload hash: '+name)
    return receipt


def authenticate(args, check):
    require(all(getattr(args, key).is_absolute() for key in ('plan', 'run', 'audit', 'terminal')), 'absolute evidence paths')
    require(digest(args.plan, check)['sha256'] == args.plan_sha256, 'external full-study plan pin')
    plan = read(args.plan)
    require(plan['version'] == 'otto-conditioning-v1' and plan['mode'] == 'study'
            and plan['status'] == 'frozen_before_execution' and plan['sources'][AUDITOR] == AUDITOR_PIN
            and plan['sources'][CLOCK] == CLOCK_PIN, 'fixed study and independent auditor identity')
    for name, pin in plan['sources'].items():
        path = ROOT/name
        require(path.resolve().is_relative_to(ROOT) and digest(path, check)['sha256'] == pin,
                'unchanged study source: '+name)
    names = {'started.json', 'runtime.json', 'preparation.json', 'work.jsonl', 'initializations.jsonl',
             'epoch-orders.jsonl', 'updates.jsonl', 'fit-curves.jsonl', 'fits.jsonl', 'parity.jsonl',
             'initial-pairing.jsonl', 'summary.json'}
    names.update(f'{phase}-{kind}-{seed}.npz' for seed in SEEDS for kind in KINDS
                 for phase in ('initial', 'final', 'predictions'))
    worker = closed(args.run, args.receipt_sha256, names, check)
    audit = closed(args.audit, args.audit_sha256, {'started.json', 'summary.json'}, check)
    require(worker['version'] == 'otto-conditioning-v1' and worker['mode'] == 'study'
            and worker['plan_sha256'] == args.plan_sha256 and worker['sources'] == plan['sources']
            and worker['inputs'] == plan['inputs'] and worker['limits'] == plan['limits']
            and worker['completed_fits'] == 6 and worker['pending'] == []
            and worker['parity_passed'] is True and worker['initial_pairing_passed'] is True,
            'all six technically completed fits')
    require(audit['version'] == 'otto-conditioning-saved-audit-v1' and audit['agreement'] is True
            and audit['source']['sha256'] == AUDITOR_PIN and audit['producer_source_sha256'] == plan['sources']['scripts/study_otto_conditioning.py']
            and audit['plan_sha256'] == args.plan_sha256 and audit['worker_sha256'] == args.receipt_sha256
            and audit['terminal_sha256'] == args.terminal_sha256
            and audit['independent_readout_calls'] == 300 and audit['independent_readout_rows'] == 73842,
            'completed independent audit of this full worker')
    require(digest(args.terminal, check)['sha256'] == args.terminal_sha256, 'external parent terminal pin')
    terminal = read(args.terminal)
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
            and terminal['group_absent'] is True and terminal['error'] is None and terminal['clock_error'] is None
            and terminal['cleanup']['errors'] == [] and terminal['cleanup']['reaped'] is True
            and terminal['cleanup']['group_absent'] is True, 'successful original supervisor')
    started = read(args.run/'started.json')
    launch = started['launch']
    require(all(launch[k] == terminal[k] for k in ('pid', 'pgid', 'parent_pid', 'command', 'cwd',
            'started_ns', 'deadline_ns', 'clock_backend', 'cap_seconds', 'watchdog_sha256', 'clock_source_sha256')),
            'same original parent and worker launch')
    require(launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns']
            and launch['cap_seconds'] == 600 and launch['deadline_ns'] == launch['started_ns']+600*10**9,
            'strict native timing enclosure')
    request = read(args.audit/'started.json')['request']
    require(all(request[k] == str(getattr(args, k)) for k in ('plan', 'plan_sha256', 'run', 'receipt_sha256',
            'terminal', 'terminal_sha256')) and request['output'] == str(args.audit), 'audited original paths and pins')
    return worker, audit, terminal


def agree(actual, expected):
    """Use the audit's comparison tolerance; retain exact discrete decisions."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(agree(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(agree(a, b) for a, b in zip(actual, expected, strict=True))
    if type(expected) is float:
        return type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)
    return type(actual) is type(expected) and actual == expected


def values(args, check):
    # Numerical summaries are decoded only after both complete closures authenticate.
    worker, audit, terminal = authenticate(args, check)
    summary, saved = read(args.audit/'summary.json'), read(args.run/'summary.json')
    require(summary['agreement'] is True and summary['audit_version'] == 'otto-conditioning-saved-audit-v1',
            'independent summary agreement')
    for key in saved.keys()-{'scope'}:
        require(agree(saved[key], summary[key]), 'audited producer summary: '+key)
    expected = {f'{kind}@{seed}' for seed in SEEDS for kind in KINDS}
    require(set(summary['metrics']) == set(summary['training_costs']) == expected
            and len(summary['checks']) == 6 and summary['rows'] == {'train': 5589, 'valid': 1109},
            'all six fits, conditions and original data rows')
    floor = summary['alias_floor']['irreducible_mse_normalized']
    rows = []
    for seed in SEEDS:
        for kind in KINDS:
            fit = f'{kind}@{seed}'
            metric = summary['metrics'][fit]
            rows.append({'fit_id': fit, 'seed': seed, 'kind': kind, **metric,
                         'train_excess': metric['train']['mse_normalized']-floor,
                         'fit_seconds': summary['training_costs'][fit]})
    return summary, rows, worker, audit, terminal


def render(out, summary, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    passed = sum(row['passes'] for row in summary['checks'])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), layout='constrained')
    specs = (('TRAIN excess above common alias floor', 'Normalized squared error', lambda r: r['train_excess']),
             ('VALID mean squared error', 'Normalized squared error', lambda r: r['valid']['mse_normalized']),
             ('Complete fit interval', 'Seconds per fit', lambda r: r['fit_seconds']))
    for axis, (title, label, getter) in zip(axes, specs, strict=True):
        for seed, color, marker in zip(SEEDS, ('#2166ac', '#b35806', '#4d9221'), ('o', 's', '^'), strict=True):
            pair = [next(r for r in rows if r['seed'] == seed and r['kind'] == kind) for kind in KINDS]
            axis.plot([0, 1], [getter(r) for r in pair], marker=marker, color=color, label=str(seed), linewidth=1.5)
        axis.set(title=title, ylabel=label, xticks=[0, 1], xticklabels=['Gain 1', 'Gain 53'], xlim=(-.18, 1.18))
        axis.margins(y=.18)
        axis.grid(axis='y', alpha=.2)
        axis.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4), useOffset=False)
    axes[0].legend(title='Paired seed', fontsize=8)
    fig.suptitle(f'Ordinary input conditioning: {passed}/6 scalar conditions passed; autonomous utility untested', fontsize=12)
    fig.supxlabel('All six fixed final fits. TRAIN excess is signed; fit intervals exclude preparation, pairing and final scoring.', fontsize=9)
    for suffix in ('png', 'svg'):
        fig.savefig(out/f'conditioning-comparison.{suffix}', dpi=180)
    plt.close(fig)


def markdown(summary, rows, worker, audit, terminal, args):
    passed = sum(c['passes'] for c in summary['checks'])
    link = lambda path: os.path.relpath(path, ROOT/'research')
    text = ['# OTTO: fixed spatial-input conditioning', '',
            (f'The fixed scalar screen passed **{passed}/6 conditions**; its overall flag is **{str(summary["conditioning_screen_passed"]).lower()}**. '
            'Both gains completed all three seeds. This is an ordinary optimization control with the same width-eight model class, not an architectural advance. '
            'Fresh autonomous evaluation is required regardless of this scalar result.'), '',
            ('Each fresh model used 5,589 cached TRAIN prefixes, uniform Monte Carlo targets `(T-t)/64`, 80 epochs and 3,520 Adam updates. '
            'All 1,109 exposed VALID prefixes were evaluated at the fixed final checkpoint. No new policy or simulator was run. '
            'The gain-53 spatial weights were inversely scaled at initialization; all three approximate initial-function checks passed before any optimizer update. '
            'The raw mass baseline and context features were unchanged.'), '',
            (f'The common empirical TRAIN alias floor is **{summary["alias_floor"]["irreducible_mse_normalized"]:.10g}**. '
            'Excess below is signed MSE minus that floor. MSE uses normalized return units; physical MAE restores the factor of 64. '
            'Negative counts refer to signed final predictions and are not clipped.'), '',
            '| Gain | Seed | TRAIN MSE | TRAIN excess | VALID MSE | TRAIN MAE | VALID MAE | Negative TRAIN / VALID | Fit seconds |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        a, b = row['train'], row['valid']
        text.append(f'| {row["kind"]} | {row["seed"]} | {a["mse_normalized"]:.8g} | {row["train_excess"]:.8g} | '
                    f'{b["mse_normalized"]:.8g} | {a["mae_physical"]:.6g} | {b["mae_physical"]:.6g} | '
                    f'{a["negative_predictions"]} / {b["negative_predictions"]} | {row["fit_seconds"]:.3f} |')
    text += ['', f'![All paired seeds]({link(args.output/"conditioning-comparison.png")})', '',
             '| Arm | Total fit seconds | Mean fit seconds |', '|---|---:|---:|']
    for kind in KINDS:
        total = math.fsum(r['fit_seconds'] for r in rows if r['kind'] == kind)
        text.append(f'| {kind} | {total:.3f} | {total/3:.3f} |')
    text += ['', (f'Preparation took {summary["preparation_seconds"]:.3f} s and initial pairing {summary["initial_pairing_seconds"]:.3f} s. '
             f'The worker interval was {worker["wall_seconds"]:.3f} s; its enclosing supervisor interval was {terminal["wall_seconds"]:.3f} s. '
             f'The separate independent audit took {audit["wall_seconds"]:.3f} s. These nested worker/parent intervals must not be summed. '
             f'Worker peak RSS was {worker["peak_rss_bytes"]/1024**2:.1f} MiB; audit peak RSS was {audit["peak_rss_bytes"]/1024**2:.1f} MiB. '
             'Per-arm fit intervals include initialization, optimizer work, journals and checkpoint publication, but exclude preparation, initial pairing and final scalar diagnostics. '
             'Timing is one CPU-thread pass, not a replicated latency benchmark.'), '',
             ('The audit independently replayed 300 local saved-model readouts covering 73,842 rows, including initial and final parity witnesses. '
             'It reconstructed cached features, targets, the alias floor and all six decisions. Torch initialization, optimization and timing truth remain authenticated execution evidence. '
             'Scalar fitting cannot establish action quality or autonomous success. Gain changes Adam steps, clipping geometry and rounding together; no isolated causal mechanism is established. '
             'All previous failed study decisions remain unchanged.'), '',
             '## Complete fixed conditions', '',
             '| Condition | Value | Maximum allowed | Passed |', '|---|---:|---:|:---:|']
    for row in summary['checks']:
        text.append(f'| {row["name"]} | {row["value"]:.10g} | {row["threshold"]:.10g} | {row["passes"]} |')
    text += ['', 'Decisions use unrounded values, without epsilon or clipping. All six are required for the scalar flag.', '',
             (f'[Protocol](otto-conditioning-protocol.md) | [Plan]({link(args.plan)}) | '
             f'[Worker receipt]({link(args.run/"receipt.json")}) | [Parent terminal]({link(args.terminal)}) | '
             f'[Independent audit]({link(args.audit/"receipt.json")}) | [Complete audit summary]({link(args.audit/"summary.json")}) | '
             f'[Exact plotted values]({link(args.output/"plotted-values.json")}) | [Publication receipt]({link(args.output/"receipt.json")})'), '']
    return '\n'.join(text)


def execute(args):
    out = args.output
    require(out.is_absolute() and not out.exists() and not any(p.is_symlink() for p in out.parents), 'exclusive report output')
    out.mkdir(parents=True, exist_ok=False)
    start = None
    try:
        require(digest(ROOT/CLOCK)['sha256'] == CLOCK_PIN, 'qualified suspend-inclusive clock')
        spec = importlib.util.spec_from_file_location('_conditioning_report_clock', ROOT/CLOCK)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        clock = module.SuspendClock()
        start = clock.now_ns()
        def check():
            require(clock.now_ns()-start < LIMITS['seconds']*10**9, 'report native deadline')
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024) <= LIMITS['rss_bytes'], 'report RSS cap')
            require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'report output cap')
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('report alarm')))
        signal.setitimer(signal.ITIMER_REAL, LIMITS['seconds'])
        summary, rows, worker, audit, terminal = values(args, check)
        render(out, summary, rows)
        check()
        write(out/'plotted-values.json', {'fits': rows, 'alias_floor': summary['alias_floor'], 'checks': summary['checks'],
                                        'conditioning_screen_passed': summary['conditioning_screen_passed']})
        with (out/'report.md').open('x') as stream:
            stream.write(markdown(summary, rows, worker, audit, terminal, args))
        authenticate(args, check)
        check()
        files = {p.name: digest(p, check) for p in out.iterdir()}
        require(set(files) == {'report.md', 'plotted-values.json', 'conditioning-comparison.png', 'conditioning-comparison.svg'}, 'exact report files')
        finished = clock.now_ns()
        write(out/'receipt.json', {'version': VERSION, 'status': 'completed', 'source': digest(Path(__file__), check),
              'request': {k: str(v) for k, v in vars(args).items()}, 'files': files, 'scope': SCOPE, 'limits': LIMITS,
              'clock_backend': clock.backend, 'started_ns': start, 'finished_ns': finished,
              'wall_seconds': (finished-start)/1e9, 'model_calls': 0, 'training_calls': 0, 'simulator_calls': 0})
        check()
        signal.setitimer(signal.ITIMER_REAL, 0)
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/'receipt.json').exists():
                (out/'receipt.json').rename(out/'invalid-completed-receipt.json')
            write(out/'failed.json', {'version': VERSION, 'status': 'failed', 'error': repr(error),
                                     'started_ns': start, 'wall_seconds': None, 'scope': SCOPE})
        except BaseException as secondary:  # noqa: BLE001 - Preserve the original failure if publication fails.
            error.add_note(repr(secondary))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'run', 'audit', 'terminal', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('plan-sha256', 'receipt-sha256', 'audit-sha256', 'terminal-sha256'):
        parser.add_argument('--'+name, required=True)
    execute(parser.parse_args())
