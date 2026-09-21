"""Stage a report and SVG from completed, independently audited saved evidence only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import resource
import signal
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts/render_otto_return_value.py'
HELPER_PIN = 'f390c890014fa97d269c13c7bd3621a982d23b3da987dbb32cb51dfaeabdf0f0'
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_PIN:
    raise ValueError('pinned saved presentation helper')
_spec = importlib.util.spec_from_file_location('_bellman_report_helpers', HELPER)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
require, read, write, digest, regular = H.require, H.read, H.write, H.digest, H.regular
VERSION = 'otto-bellman-control-report-v1'
AUDITOR = 'scripts/audit_otto_bellman_control.py'
AUDITOR_PIN = '37490345ad60d0c4a18228d65040d704179d77ab6e3586c77bf144cc96dee49d'
RUNNER = 'scripts/study_otto_bellman_control.py'
SEEDS, FAMILIES, REGIMES = (10101, 10102, 10103), ('backup', 'mc', 'reference'), ('lambda3', 'lambda4', 'lambda5')
ARMS = tuple(f'{f}@{s}' for f in FAMILIES for s in SEEDS) + ('analytic_inbounds',)
LABELS = {'backup': 'Bellman', 'mc': 'Monte Carlo', 'reference': 'Unchanged', 'analytic_inbounds': 'Analytic'}
COLORS = {'backup': '#7550a3', 'mc': '#286caa', 'reference': '#677583', 'analytic_inbounds': '#087f6d'}
REPORT, FIGURE = 'research/otto-bellman-control-results.md', 'docs/assets/otto-bellman-control.svg'
LIMITS = {'seconds': 180, 'rss_bytes': 2 * 1024**3, 'output_bytes': 64 * 1024**2}


def payload_names():
    names = {'started', 'runtime', 'preparation', 'target-audit', 'training-costs', 'native-setup', 'inference-setup', 'summary'}
    names = {n + '.json' for n in names} | {n + '.jsonl' for n in (
        'work-contexts', 'work', 'initializations', 'updates', 'epoch-orders', 'fit-curves', 'fits', 'target-checkpoints',
        'target-refreshes', 'parity', 'prediction-diagnostics', 'eval-transitions', 'eval-episodes', 'evaluation')}
    for seed in SEEDS:
        names.add(f'mc-targets-{seed}.npz')
        names.update(f'{p}-{m}-{seed}.npz' for p in ('initial', 'final', 'predictions') for m in ('mc', 'backup'))
        names.update(f'{p}-backup-{seed}-{i:02d}.npz' for p in ('target-checkpoint', 'targets') for i in range(8))
    return names


def rules(summary):
    expected = [f'{r}.{s}.{k}' for r in REGIMES for s in SEEDS for k in ('success', 'moves')]
    expected += [f'{r}.{c}.{k}' for r in REGIMES for c in ('mc', 'reference')
                 for k in ('success', 'moves', 'positive_blocks', 'controller_cost')]
    checks = summary['competence_checks'] + summary['improvement_checks']
    require([c['name'] for c in checks] == expected and len(summary['competence_checks']) == 18, 'all 42 fixed rules')
    require(all(type(c['passes']) is bool for c in checks), 'boolean rule decisions')
    require(summary['pilot_continuation'] is all(c['passes'] for c in checks), 'overall gate identity')
    return sum(c['passes'] for c in checks)


class Report:
    manifest = H.Render.manifest

    def __init__(self, args):
        self.args, self.out = args, args.output
        self.receipt = {'version': VERSION, 'status': 'started', 'model_calls': 0, 'simulator_calls': 0,
                        'training_calls': 0, 'scope': 'Saved audited aggregates only; no scientific rescoring.'}

    def check(self):
        require(self.clock.now_ns() - self.start < LIMITS['seconds'] * 10**9, 'report deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        require(rss <= LIMITS['rss_bytes'], 'report RSS cap')
        require(sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file()) <= LIMITS['output_bytes'], 'report output cap')

    def authenticate(self):
        a = self.args
        worker = self.manifest(a.run, a.receipt_sha256, payload_names())
        audit = self.manifest(a.audit, a.audit_receipt_sha256, {'started.json', 'summary.json'})
        for path, pin in ((a.plan, a.plan_sha256), (a.terminal, a.terminal_sha256)):
            require(digest(regular(path), self.check)['sha256'] == pin, 'external plan/terminal pin')
        plan, terminal = read(a.plan), read(a.terminal)
        require(plan['version'] == worker['version'] == 'otto-bellman-control-v1' and plan['mode'] == worker['mode'] == 'study'
                and plan['status'] == 'frozen_before_execution' and worker['plan_sha256'] == a.plan_sha256, 'frozen study identity')
        require(worker['completed_episodes'] == 1440 and worker['completed_fits'] == 6 and worker['pending'] == []
                and worker['external_model_calls'] == 0 and worker['target_refreshes'] == 24
                and worker['target_audit_passed'] is True and worker['parity_passed'] is True
                and all(v['attempted'] == v['returned'] for v in worker['calls'].values()), 'complete scientific work')
        require(audit['version'] == 'otto-bellman-control-saved-audit-v1' and audit['agreement'] is True
                and audit['worker_sha256'] == a.receipt_sha256 and audit['plan_sha256'] == a.plan_sha256
                and audit['terminal_sha256'] == a.terminal_sha256, 'matching independent audit')
        request = read(a.audit / 'started.json')['request']
        for k in ('plan', 'run', 'terminal', 'plan_sha256', 'receipt_sha256', 'terminal_sha256'):
            require(request[k] == str(getattr(a, k)), 'audit request identity')
        require(request['output'] == str(a.audit), 'audit directory identity')
        for k in ('sources', 'inputs', 'limits'):
            require(worker[k] == plan[k], 'worker prospective plan join')
        require(plan['sources'][AUDITOR] == AUDITOR_PIN == audit['source']['sha256']
                and audit['producer_source_sha256'] == plan['sources'][RUNNER], 'audited source identity')
        for name, pin in plan['sources'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'relative source path')
            require(digest(regular(ROOT / name), self.check)['sha256'] == pin, 'unchanged source closure')
        started = read(a.run / 'started.json')
        launch = started['launch']
        require(started['request']['plan'] == str(a.plan) and started['request']['plan_sha256'] == a.plan_sha256
                and started['request']['output'] == str(a.run), 'worker request identity')
        require(digest(regular(Path(started['request']['supervision'])), self.check)['sha256'] == worker['supervision_sha256'], 'launch pin')
        require(read(Path(started['request']['supervision'])) == launch, 'launch contents')
        for k in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend', 'cap_seconds'):
            require(terminal[k] == launch[k], 'parent launch join')
        require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
                and terminal['group_absent'] is True and terminal['cleanup']['reaped'] is True
                and terminal['cleanup']['errors'] == [] and terminal['error'] is None and terminal['clock_error'] is None
                and launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns'], 'successful original parent')
        result, producer = read(a.audit / 'summary.json'), read(a.run / 'summary.json')
        require(result['agreement'] is True and result['version'] == audit['version'] and result['episodes'] == 1440
                and result['paired_cases'] == 144 and result['learned_architecture_advantage_established'] is False, 'complete audited result')
        rules(result)
        require(worker['pilot_continuation'] is result['pilot_continuation'], 'worker/audit gate join')
        for k in ('regimes', 'competence_checks', 'improvement_checks', 'pilot_continuation'):
            H.Render.agree(producer[k], result[k])
        H.Render.agree(producer['amortization']['scenarios'], result['amortization'])
        self.receipt['inputs'] = {k: {'path': str(p), **digest(p, self.check)} for k, p in {
            'plan': a.plan, 'terminal': a.terminal, 'worker': a.run / 'receipt.json', 'audit': a.audit / 'receipt.json',
            'audit_summary': a.audit / 'summary.json', 'worker_summary': a.run / 'summary.json'}.items()}
        return result


def present(summary, args):
    passed = rules(summary)
    status = 'PASS' if summary['pilot_continuation'] else 'FAIL'
    rows = []
    for regime in REGIMES:
        panel = summary['regimes'][regime]
        require(set(panel['means']) == set(ARMS), 'all ten arms')
        for arm in ARMS:
            m = panel['means'][arm]
            require(all(type(m[k]) in (int, float) and math.isfinite(m[k]) and m[k] >= 0
                        for k in ('found', 'steps', 'controller_seconds')) and m['found'] <= 1, 'finite display values')
            rows.append({'regime': regime, 'arm': arm, **{k: m[k] for k in ('found', 'steps', 'controller_seconds')}})
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1260" height="1180" viewBox="0 0 1260 1180" role="img">',
           '<title>Fixed-architecture target comparison</title><rect width="1260" height="1180" fill="#fff"/>',
           '<g font-family="sans-serif" fill="#172431">']
    def text(x, y, value, size=14):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}">{escape(str(value))}</text>')
    text(30, 35, 'Same MLP architecture, different continuation targets', 25)
    text(30, 64, f'{status}: {passed}/42 conditions passed | 1,440 episodes | all three fit seeds retained', 17)
    metrics = (('found', 'Weighted success (%)', 100), ('steps', 'Capped moves (lower is better)', 1),
               ('controller_seconds', 'Paid controller seconds / search', 1))
    for ri, regime in enumerate(REGIMES):
        y0 = 110 + ri * 340
        text(30, y0, f'{regime}: ' + ('unseen supplied kernel' if ri == 2 else 'training-supported setting'), 18)
        group = [r for r in rows if r['regime'] == regime]
        for mi, (key, title, scale) in enumerate(metrics):
            x0 = 30 + mi * 410
            text(x0, y0 + 27, title)
            maximum = 100 if key == 'found' else max(r[key] for r in group) * 1.07 or 1
            for ai, r in enumerate(group):
                family, _, seed = r['arm'].partition('@')
                y, value = y0 + 51 + ai * 25, r[key] * scale
                text(x0, y, LABELS[family] + (' ' + seed if seed else ''), 12)
                svg.append(f'<rect x="{x0+125}" y="{y-11}" width="{190*value/maximum:.3f}" height="13" fill="{COLORS[family]}"/>')
                text(x0 + 322, y, f'{value:.3g}', 12)
    text(30, 1150, 'Initial-hit mixture weighted within each setting; no confidence intervals. Controller cost includes deployment allocation.', 13)
    svg.append('</g></svg>')
    rel = lambda p: '../' + str(p.relative_to(ROOT))
    lines = ['# Bellman versus Monte Carlo continuation', '', (f'**{status}: {passed}/42 frozen conditions passed.** '
             f'Competence: {sum(c["passes"] for c in summary["competence_checks"])}/18; paired improvement: '
             f'{sum(c["passes"] for c in summary["improvement_checks"])}/24.'), '',
             ('This compares training targets in the same MLP architecture. It does not establish an architecture, '
             'connectome or recurrent-model advantage. The original scalar study remains a 0/54 failure.'), '',
             '![All arms and settings](../docs/assets/otto-bellman-control.svg)', '',
             ('Six continuations restore three fixed MLP checkpoints: Monte Carlo returns versus delayed Bellman backups. '
             'Three unchanged checkpoints and an analytic controller complete the ten arms. All use the public full posterior. '
             'Each arm has 48 paired searches per setting, capped at 2,188 moves. Lambda 3 and 4 are training-supported; '
             'lambda 5 supplies an unseen kernel. These are fresh local evaluation cases.'), '',
             ('Each table uses the setting-specific initial-hit mixture, not pooled episode means. A failed search contributes '
             '2,188 moves. Controller seconds include feature/branch construction, readout, selection, public updates, '
             'initialization and allocated deployment setup; recorded artifact I/O is excluded. All fit seeds are shown.'), '']
    for regime in REGIMES:
        lines += [f'## {regime}', '', '| Arm | Success | Moves | Controller s | H=1 s | H=100 s | H=10,000 s |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for r in (v for v in rows if v['regime'] == regime):
            am = summary['amortization'][regime][r['arm']]['seconds_per_search']
            lines.append(f'| {r["arm"]} | {r["found"]:.2%} | {r["steps"]:.3f} | {r["controller_seconds"]:.6f} | '
                         + ' | '.join(f'{am[str(h)]:.6f}' for h in (1, 100, 10000)) + ' |')
        lines.append('')
        lines.append('Equal-seed family means (success / moves / controller s): ' + '; '.join(
            f'{LABELS[f]} {m["found"]:.2%} / {m["steps"]:.3f} / {m["controller_seconds"]:.6f}'
            for f, m in summary['regimes'][regime]['family_means'].items()) + '.')
        lines.append('')
    training = summary['costs']['training']
    lines += ['## Cost and interpretation', '', (f'Preparation: {training["preparation_seconds"]:.3f} s. '
              f'Full worker duration: {summary["costs"]["worker_seconds"]:.3f} s.'), '', '| Continuation | Paid fit seconds |', '|---|---:|']
    lines += [f'| {arm} | {seconds:.3f} |' for arm, seconds in training['fits'].items()]
    lines += ['', ('The two arms receive equal optimizer updates, not equal total compute. Backup target construction is paid in fit time. '
              'Separate parity, scalar diagnostics and independent target auditing are excluded from fit time. H columns are accounting scenarios '
              'U + (L + P/6 + D)/H, not additional searches: U excludes deployment setup, L is continuation time, P is preparation, '
              'and D is head loading plus shared module setup. Unchanged references have L=P=0; original training is a common prior investment. '
              'Analytic initialization is paid anew per search. These single-run timings are hardware-specific.'), '',
              ('Monte Carlo returns estimate the analytic teacher policy, not optimal costs. Bellman targets are generated by each arm\'s '
              'own delayed checkpoint on the same teacher-visited TRAIN states, with no exploratory collection. The comparison therefore retains '
              'target noise, coverage limits and undiscounted bootstrapping risk. A gate pass would remain a pilot result, not a novelty claim.'), '',
              (f'Independent audit: {summary["comparisons"]:,} comparisons. Saved targets, choices and aggregates were checked; '
              'optimizer trajectories, actual Torch execution, RNG and timing truth remain source-bound evidence.'), '',
              (f'[Protocol](otto-bellman-control-protocol.md) | [Worker summary]({rel(args.run / "summary.json")}) | '
              f'[Independent audit]({rel(args.audit / "summary.json")}) | [Worker receipt]({rel(args.run / "receipt.json")}) | '
              f'[Audit receipt]({rel(args.audit / "receipt.json")})'), '']
    return '\n'.join(lines), '\n'.join(svg), {'rows': rows, 'gate_passed': passed, 'gate_total': 42,
        'pilot_continuation': summary['pilot_continuation'], 'weights': {r: summary['regimes'][r]['weights'] for r in REGIMES},
        'family_means': {r: summary['regimes'][r]['family_means'] for r in REGIMES},
        'amortization': summary['amortization'], 'training_costs': training}


def execute(args):
    r = Report(args)
    require(args.output.is_absolute() and not args.output.exists() and not any(p.is_symlink() for p in args.output.parents), 'exclusive output')
    args.output.mkdir(parents=True, exist_ok=False)
    old = signal.getsignal(signal.SIGALRM)
    try:
        require(digest(regular(ROOT / H.CLOCK))['sha256'] == H.CLOCK_PIN, 'qualified clock')
        r.clock = H.load(ROOT / H.CLOCK, '_bellman_report_clock').SuspendClock()
        r.start = r.clock.now_ns()
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('report deadline')))
        signal.setitimer(signal.ITIMER_REAL, LIMITS['seconds'])
        summary = r.authenticate()
        markdown, svg, values = present(summary, args)
        for name, content in ((REPORT, markdown), (FIGURE, svg)):
            path = args.output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x') as stream:
                stream.write(content)
        write(args.output / 'plotted-values.json', values)
        r.check()
        r.receipt.update(status='completed', source={'path': str(Path(__file__)), **digest(Path(__file__))}, helper_sha256=HELPER_PIN,
                         files={str(p.relative_to(args.output)): digest(p, r.check) for p in args.output.rglob('*') if p.is_file()})
        write(args.output / 'receipt.json', r.receipt)
        r.check()
        return r.receipt
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (args.output / 'receipt.json').exists():
                (args.output / 'receipt.json').rename(args.output / 'invalid-completed-receipt.json')
            write(args.output / 'failed.json', {**r.receipt, 'status': 'failed', 'error': repr(error)})
        except BaseException as secondary:  # noqa: BLE001 - Retain the original failure.
            error.add_note(f'Failure receipt: {secondary!r}')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'run', 'terminal', 'audit', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    for flag in ('plan-sha256', 'receipt-sha256', 'terminal-sha256', 'audit-receipt-sha256'):
        parser.add_argument('--' + flag, required=True)
    execute(parser.parse_args())
