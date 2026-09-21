"""Stage PNG/SVG figures and a report from completed, independently audited saved evidence."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import resource
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts/render_otto_return_value.py'
HELPER_PIN = 'f390c890014fa97d269c13c7bd3621a982d23b3da987dbb32cb51dfaeabdf0f0'
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_PIN:
    raise ValueError('pinned saved presentation helper')
_spec = importlib.util.spec_from_file_location('_conditioning_report_helpers', HELPER)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
require, read, write, digest, regular = H.require, H.read, H.write, H.digest, H.regular
VERSION, STUDY = 'otto-conditioning-control-report-v1', 'otto-conditioning-control-v1'
AUDITOR = 'scripts/audit_otto_conditioning_control.py'
AUDITOR_PIN = '094fd41e7b9c34d0df580c1d99aef0e2f1bc2c83997afad36cd8c15f14794e1b'
RUNNER = 'scripts/study_otto_conditioning_control.py'
SEEDS, FAMILIES, REGIMES = (10101, 10102, 10103), ('gain1', 'gain53'), ('lambda3', 'lambda4', 'lambda5')
ARMS = tuple(f'{f}@{s}' for s in SEEDS for f in FAMILIES) + ('analytic_inbounds',)
METRICS = ('found', 'steps', 'controller_seconds')
COLORS = {'gain1': '#296ca6', 'gain53': '#9750a0', 'analytic_inbounds': '#087e70'}
REPORT, FIGURE = 'report.md', 'otto-conditioning-control'
LIMITS = {'seconds': 180, 'rss_bytes': 2 * 1024**3, 'output_bytes': 64 * 1024**2}
PAYLOADS = {n + '.json' for n in ('started', 'runtime', 'inference-setup', 'native-setup', 'summary')} | {
    n + '.jsonl' for n in ('work-contexts', 'work', 'eval-transitions', 'eval-episodes', 'evaluation')}


def rules(summary):
    """Validate the saved direct decisions; rounded display never decides a gate."""
    groups = {}
    for family, key in (('gain53', 'competence_checks'), ('gain1', 'control_competence_checks')):
        expected = [f'{r}.{family}.{s}.{k}' for r in REGIMES for s in SEEDS for k in ('success', 'moves')]
        require([c['name'] for c in summary[key]] == expected, 'all 18 fixed competence records')
        groups[key] = summary[key]
    key = 'improvement_checks'
    require([c['name'] for c in summary[key]] == [f'{r}.{k}' for r in REGIMES for k in ('success', 'moves', 'positive_blocks', 'cost')],
            'all 12 fixed improvement records')
    groups[key] = summary[key]
    for records in groups.values():
        for c in records:
            require(type(c['passes']) is bool and all(type(c[k]) in (float, int) and math.isfinite(c[k]) for k in ('value', 'threshold')),
                    'finite unrounded condition record')
            greater = c['name'].endswith(('.success', '.positive_blocks'))
            require(c['passes'] is (c['value'] >= c['threshold'] if greater else c['value'] <= c['threshold']), 'exact saved inequality')
    decisions = groups['competence_checks'] + groups['improvement_checks']
    require(summary['pilot_continuation'] is all(c['passes'] for c in decisions), 'all 30 conditions determine continuation')
    return {k: sum(c['passes'] for c in v) for k, v in groups.items()}


class Report:
    manifest = H.Render.manifest

    def __init__(self, args):
        self.args, self.out = args, args.output
        self.receipt = {'version': VERSION, 'status': 'started', 'limits': LIMITS,
                        'model_calls': 0, 'simulator_calls': 0, 'training_calls': 0,
                        'scope': 'Saved audited aggregates only. No arrays decoded, policies called or scientific rules changed.'}

    def check(self):
        require(self.clock.now_ns() - self.start < LIMITS['seconds'] * 10**9, 'report deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= LIMITS['rss_bytes'], 'report RSS cap')
        require(sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file()) <= LIMITS['output_bytes'], 'report output cap')

    def authenticate(self):
        a = self.args
        worker = self.manifest(a.run, a.receipt_sha256, PAYLOADS)
        audit = self.manifest(a.audit, a.audit_receipt_sha256, {'started.json', 'summary.json'})
        for path, pin in ((a.plan, a.plan_sha256), (a.terminal, a.terminal_sha256)):
            require(digest(regular(path), self.check)['sha256'] == pin, 'external plan/terminal pin')
        plan, terminal = read(a.plan), read(a.terminal)
        require(plan['version'] == worker['version'] == STUDY and plan['mode'] == worker['mode'] == 'study'
                and plan['status'] == 'frozen_before_execution' and worker['plan_sha256'] == a.plan_sha256, 'frozen study identity')
        require(worker['completed_episodes'] == 504 and worker['completed_fits'] == 0 and worker['pending'] == []
                and worker['external_model_calls'] == 0 and worker['parity_passed'] is True
                and set(worker['calls']) == {'model_load', 'native_reset', 'native_step', 'value_forward', 'analytic_choose'}
                and worker['calls']['model_load']['returned'] == 6 and worker['calls']['native_reset']['returned'] == 507
                and all(v['attempted'] == v['returned'] for v in worker['calls'].values()), 'complete no-training autonomous work')
        require(audit['version'] == 'otto-conditioning-control-saved-audit-v1' and audit['mode'] == 'study'
                and audit['agreement'] is True and audit['worker_sha256'] == a.receipt_sha256
                and audit['plan_sha256'] == a.plan_sha256 and audit['terminal_sha256'] == a.terminal_sha256, 'matching independent full audit')
        request = read(a.audit / 'started.json')['request']
        require(request['mode'] == 'study' and request['output'] == str(a.audit), 'full audit output identity')
        for k in ('plan', 'run', 'terminal', 'plan_sha256', 'receipt_sha256', 'terminal_sha256'):
            require(request[k] == str(getattr(a, k)), 'audit request identity')
        for k in ('sources', 'inputs', 'limits'):
            require(worker[k] == plan[k], 'worker prospective plan join')
        require(plan['sources'][AUDITOR] == AUDITOR_PIN == audit['source']['sha256']
                and audit['producer_source_sha256'] == plan['sources'][RUNNER], 'independent audit source identity')
        for name, pin in plan['sources'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'relative source path')
            require(digest(regular(ROOT / name), self.check)['sha256'] == pin, 'unchanged source closure')
        started = read(a.run / 'started.json')
        request, launch = started['request'], started['launch']
        require(request == {'plan': str(a.plan), 'plan_sha256': a.plan_sha256, 'output': str(a.run), 'supervision': request['supervision']},
                'worker request identity')
        require(digest(regular(Path(request['supervision'])), self.check)['sha256'] == worker['supervision_sha256']
                and read(Path(request['supervision'])) == launch, 'original launch pin and contents')
        for k in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend', 'cap_seconds',
                  'clock_source_sha256', 'watchdog_sha256'):
            require(terminal[k] == launch[k], 'original parent launch join')
        command = list(launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        require(len(command) == 10 and command[:2] == [plan['python_executable'], str(ROOT / RUNNER)]
                and dict(zip(command[2::2], command[3::2], strict=True)) == {f'--{k.replace("_", "-")}': v for k, v in request.items()},
                'actual original command')
        require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
                and terminal['group_absent'] is True and terminal['cleanup']['group_absent'] is True
                and terminal['cleanup']['reaped'] is True and terminal['cleanup']['errors'] == []
                and terminal['error'] is None and terminal['clock_error'] is None and terminal['timing_available'] is True
                and launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns'],
                'successful original parent before outcome decoding')
        require(launch['cwd'] == str(ROOT) and launch['pid'] == launch['pgid'] and launch['parent_pid'] != launch['pid']
                and launch['cap_seconds'] == plan['limits']['native_seconds']
                and launch['deadline_ns'] == launch['started_ns'] + launch['cap_seconds'] * 10**9
                and launch['clock_backend'] == worker['clock_backend']
                and launch['clock_source_sha256'] == H.CLOCK_PIN
                and launch['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py'], 'supervised timing/source contract')
        require(worker['wall_seconds'] == (worker['finished_ns'] - worker['started_ns']) / 1e9
                and terminal['wall_seconds'] == (terminal['finished_ns'] - terminal['started_ns']) / 1e9
                and 0 < worker['peak_rss_bytes'] <= plan['limits']['rss_bytes']
                and sum(p.stat().st_size for p in a.run.iterdir()) <= plan['limits']['output_bytes'], 'closed runtime and resource bounds')
        result, producer = read(a.audit / 'summary.json'), read(a.run / 'summary.json')
        require(result['agreement'] is True and result['version'] == STUDY and result['audit_version'] == audit['version']
                and result['episodes'] == 504 and result['paired_cases'] == 72
                and result['learned_architecture_advantage_established'] is False, 'complete audited result')
        rules(result)
        require(worker['pilot_continuation'] is result['pilot_continuation'], 'worker/audit gate join')
        for k in ('regimes', 'competence_checks', 'improvement_checks', 'control_competence_checks', 'pilot_continuation'):
            H.Render.agree(producer[k], result[k])
        H.Render.agree(producer['amortization']['scenarios'], result['amortization'])
        costs = {'autonomous_parent_seconds': terminal['wall_seconds'], 'saved_audit_seconds': audit['wall_seconds']}
        # Prior qualification costs are read only after the full audit identity is accepted.
        for role in ('receipt', 'terminal', 'audit'):
            descriptor = plan['qualification'][role]
            path = Path(descriptor['path'])
            if not path.is_absolute():
                path = ROOT / path
            require(digest(regular(path), self.check) == {k: descriptor[k] for k in ('sha256', 'bytes')}, 'qualified cost metadata pin')
            record = read(path)
            require(record['status'] == 'completed', 'completed qualification cost scope')
            costs['qualification_' + role + '_seconds'] = record['wall_seconds']
        self.receipt['inputs'] = {k: {'path': str(p), **digest(p, self.check)} for k, p in {
            'plan': a.plan, 'terminal': a.terminal, 'worker': a.run / 'receipt.json', 'audit': a.audit / 'receipt.json',
            'audit_summary': a.audit / 'summary.json', 'worker_summary': a.run / 'summary.json'}.items()}
        return result, costs


def display_rows(summary):
    rows = []
    require(set(summary['regimes']) == set(REGIMES), 'all three settings')
    for regime in REGIMES:
        panel = summary['regimes'][regime]
        require(set(panel['means']) == set(ARMS), 'all seven arms')
        for arm in ARMS:
            means, counts = panel['means'][arm], panel['raw_counts'][arm]
            require(counts['episodes'] == 24 and type(counts['found']) is int and 0 <= counts['found'] <= 24, 'complete raw success count')
            require(all(type(means[k]) in (int, float) and math.isfinite(means[k]) and means[k] >= 0 for k in METRICS)
                    and means['found'] <= 1 and 1 <= means['steps'] <= 2188, 'finite audited display values')
            raw = {k: math.fsum(panel['strata'][str(h)][arm][k] for h in (1, 2, 3)) / 3 for k in METRICS}
            require(abs(raw['found'] - counts['found'] / 24) <= 1e-15, 'balanced raw success identity')
            rows.append({'regime': regime, 'arm': arm, 'found_count': counts['found'], 'episodes': 24,
                         'weighted': {k: means[k] for k in METRICS}, 'unweighted': raw})
    return rows


def figure(rows, passed, output):
    """Standard scientific plot; same scale across settings for each metric."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    titles = ('Weighted success (%)', 'Weighted capped moves', 'Weighted controller seconds / search')
    maxima = {key: 100 if key == 'found' else max(r['weighted'][key] for r in rows) * 1.08 or 1 for key in METRICS}
    markers = ('o', 's', '^')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none'}):
        fig, axes = plt.subplots(3, 3, figsize=(12.9, 10.8), sharey='row')
        fig.subplots_adjust(left=.10, right=.985, bottom=.11, top=.81, wspace=.12, hspace=.34)
        fig.suptitle('Same model class, different input conditioning', x=.10, y=.98, ha='left', fontsize=21, fontweight='bold')
        fig.text(.10, .94, f'{"PASS" if passed == 30 else "FAIL"}: {passed}/30 conditions passed | 504 searches | all three fitting seeds', fontsize=13)
        legend = [Line2D([], [], color='#374151', marker=m, linestyle='none', markersize=7, label=str(seed))
                  for m, seed in zip(markers, SEEDS, strict=True)]
        legend += [Line2D([], [], color='#374151', linewidth=2.5, label='Equal-seed mean'),
                   Line2D([], [], color=COLORS['analytic_inbounds'], marker='D', linestyle='none', markersize=7, label='Analytic')]
        fig.legend(handles=legend, loc='upper left', bbox_to_anchor=(.09, .91), frameon=False, ncol=5, fontsize=11)
        for mi, (metric, title) in enumerate(zip(METRICS, titles, strict=True)):
            for ri, regime in enumerate(REGIMES):
                ax = axes[mi, ri]
                panel = [r for r in rows if r['regime'] == regime]
                scale = 100 if metric == 'found' else 1
                for fi, family in enumerate((*FAMILIES, 'analytic_inbounds')):
                    group = [r for r in panel if r['arm'].split('@')[0] == family]
                    values = [r['weighted'][metric] * scale for r in group]
                    if len(group) == 3:
                        for si, value in enumerate(values):
                            ax.plot(fi + (si - 1) * .10, value, marker=markers[si], markersize=7,
                                    linestyle='none', color=COLORS[family], markeredgecolor='white', markeredgewidth=.7, zorder=3)
                        ax.hlines(math.fsum(values) / 3, fi - .22, fi + .22, color=COLORS[family], linewidth=2.5, zorder=2)
                    else:
                        ax.plot(fi, values[0], marker='D', markersize=7, color=COLORS[family], linestyle='none', zorder=3)
                ax.set(xlim=(-.4, 2.4), ylim=(0, maxima[metric]), xticks=(0, 1, 2), xticklabels=('Gain 1', 'Gain 53', 'Analytic'))
                ax.set_axisbelow(True)
                ax.grid(axis='y', color='#dce3e9', linewidth=.8)
                ax.spines[['top', 'right']].set_visible(False)
                ax.spines[['left', 'bottom']].set_color('#9ca3af')
                ax.tick_params(length=0, pad=7)
                if ri == 0:
                    ax.set_ylabel(title, labelpad=10)
                if mi == 0:
                    ax.set_title(regime + ('\nUnseen supplied kernel' if ri == 2 else '\nTraining-supported'), fontsize=12, pad=14)
        fig.text(.10, .055, 'Same y scale across settings in each row. Hit-mixture weighted; markers are seeds, not confidence intervals.', fontsize=10)
        fig.text(.10, .030, 'Controller time includes deployment setup allocation. One rotated timing run; no architecture or novelty claim.', fontsize=10)
        for extension in ('png', 'svg'):
            path = output / f'{FIGURE}.{extension}'
            require(not path.exists(), 'exclusive figure file')
            fig.savefig(path, dpi=180, facecolor='white')
        plt.close(fig)
    return {'library': 'matplotlib', 'version': matplotlib.__version__, 'pixel_dimensions': [2322, 1944],
            'metric_axis_maxima': maxima, 'seed_markers': dict(zip(map(str, SEEDS), markers, strict=True)),
            'family_mean': 'equal weight for all three fitting seeds', 'confidence_intervals': False}


def present(summary, costs, args):
    counts, rows = rules(summary), display_rows(summary)
    passed = counts['competence_checks'] + counts['improvement_checks']
    rel = lambda p: '../' + str(p.relative_to(ROOT))
    lines = ['# Autonomous search after input conditioning', '',
             (f'**{"PASS" if passed == 30 else "FAIL"}: {passed}/30 frozen conditions passed.** Gain53 competence: '
             f'{counts["competence_checks"]}/18; paired conditioning improvement: {counts["improvement_checks"]}/12. '
             f'Gain1 descriptive competence: {counts["control_competence_checks"]}/18.'), '',
             ('This compares two training parameterizations of the same width-eight MLP, using six unchanged checkpoints and an analytic control. '
             'It establishes no new architecture, recurrence or connectome advantage. The earlier scalar screen remains **FAIL, 3/6 conditions passed**.'), '',
             '![Every seed and setting](../docs/assets/otto-conditioning-control.png)', '',
             ('All 504 searches are retained: seven arms on 24 paired cases per setting, with a 2,188-move cap. '
             'Lambda3/4 are training-supported; lambda5 has an unseen sensing length with the exact kernel supplied on the same grid. '
             'This is supplied-model parameter transfer, not unknown-model adaptation. Multiple fitting seeds share cases and are not independent environmental samples.'), '',
             ('Raw means weight the three balanced initial-hit strata equally. Weighted means use the setting-specific initial-hit mixture; '
             'family means equally average all three fitting seeds. Failures contribute 2,188 moves. '
             'Controller seconds include initialization, branches/features, gain scaling, readout, selection, updates and allocated deployment setup. '
             'Only measured nested artifact I/O is excluded. No confidence intervals are claimed.'), '']
    for regime in REGIMES:
        lines += [f'## {regime}', '', '| Arm | Raw found / 24 | Weighted success | Raw moves | Weighted moves | Raw controller s | Weighted controller s |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for r in (v for v in rows if v['regime'] == regime):
            u, w = r['unweighted'], r['weighted']
            lines.append(f'| {r["arm"]} | {r["found_count"]}/24 ({u["found"]:.1%}) | {w["found"]:.2%} | {u["steps"]:.3f} | '
                         f'{w["steps"]:.3f} | {u["controller_seconds"]:.6f} | {w["controller_seconds"]:.6f} |')
        panel = summary['regimes'][regime]
        lines += ['', 'Equal-seed weighted family means (success / capped moves / controller s): ' + '; '.join(
            f'{f}: {panel["family_means"][f]["found"]:.2%} / {panel["family_means"][f]["steps"]:.3f} / '
            f'{panel["family_means"][f]["controller_seconds"]:.6f}' for f in FAMILIES) + '.', '',
            '| Arm | H=1 s/search | H=100 s/search | H=10,000 s/search |', '|---|---:|---:|---:|']
        for arm in ARMS:
            am = summary['amortization'][regime][arm]['seconds_per_search']
            lines.append(f'| {arm} | ' + ' | '.join(f'{am[str(h)]:.6f}' for h in (1, 100, 10000)) + ' |')
        lines.append('')
    prior = summary['costs']['training_reference']
    lines += ['## Paid costs and scope', '', (f'Autonomous worker: {summary["costs"]["worker_seconds"]:.3f} s; original parent: '
              f'{costs["autonomous_parent_seconds"]:.3f} s; independent saved audit: {costs["saved_audit_seconds"]:.3f} s. '
              'Worker time is nested inside parent time, not additional to it.'), '',
              (f'Previously paid scalar fitting worker: {prior["worker_seconds"]:.3f} s, including preparation '
              f'{prior["preparation_seconds"]:.3f} s and initial function pairing {prior["initial_pairing_seconds"]:.3f} s. '
              'No fitting occurred in the autonomous study.'), '',
              '| Fixed checkpoint | Original fit s | Deployment load/validation s |', '|---|---:|---:|']
    lines += [f'| {a} | {prior["training_costs"][a]:.6f} | {summary["costs"]["inference_setup"][a]["seconds"]:.6f} |' for a in ARMS[:-1]]
    runtime = summary['costs']['runtime']
    lines += ['', (f'Autonomous shared setup: {runtime["shared_setup_seconds"]:.6f} s; learned module setup: '
              f'{runtime["model_module_setup_seconds"]:.6f} s. Each head load is allocated over 72 searches; learned module setup over 432.'), '',
              (f'Separate predeployment qualification: worker {costs["qualification_receipt_seconds"]:.3f} s, parent '
              f'{costs["qualification_terminal_seconds"]:.3f} s, saved audit {costs["qualification_audit_seconds"]:.3f} s. '
              'It used 96 head/state comparisons and zero native calls; full evaluation did not repeat parity.'), '',
              ('H columns are accounting scenarios U + (L + P/6 + D)/H, not extra searches. U removes deployment allocation, '
              'L is that head\'s original fit time, P is shared fitting preparation, and D is its load plus one sixth of learned module setup. '
              'Qualification, initial pairing and other diagnostics are reported separately, not silently added to this formula. '
              'Analytic initialization is paid per search. Timing is specific to one rotated run and this runtime.'), '',
              '## Every frozen condition', '', 'Values below are rounded for reading. Decisions use the saved unrounded inequalities; no epsilon or rule is changed.', '']
    for key, title in (('competence_checks', 'Gain53 competence (18)'), ('improvement_checks', 'Paired improvement (12)'),
                       ('control_competence_checks', 'Gain1 competence (18 descriptive controls, outside the 30-condition screen)')):
        lines += [f'### {title}', '', '| Rule | Value | Required bound | Result |', '|---|---:|---:|---|']
        for c in summary[key]:
            operator = '>=' if c['name'].endswith(('.success', '.positive_blocks')) else '<='
            lines.append(f'| {c["name"]} | {c["value"]:.9g} | {operator} {c["threshold"]:.9g} | {"PASS" if c["passes"] else "FAIL"} |')
        lines.append('')
    lines += ['## Interpretation and provenance', '',
              ('A pass supports separately frozen confirmation of this optimization recipe, not novelty or optimality. A failure does not rule out '
              'other conditioning, coverage, spatial or recurrent mechanisms. Full public belief is supplied to every arm; this study does not test learned memory. '
              'The independent audit reconstructs public trajectories, saved learned readouts and aggregates. Actual native randomness, historical optimization '
              'and timing truth remain authenticated execution evidence.'), '',
              (f'Independent audit: {summary["comparisons"]:,} comparisons and {summary["saved_checkpoint_readout_calls"]:,} saved-checkpoint readouts '
              f'({summary["saved_network_rows"]:,} branch rows). The reporter performed no model or simulator calls.'), '',
              (f'[Protocol](otto-conditioning-control-protocol.md) | [Prior scalar result](otto-conditioning-results.md) | '
              f'[Frozen plan]({rel(args.plan)}) | [Worker summary]({rel(args.run / "summary.json")}) | '
              f'[Independent audit]({rel(args.audit / "summary.json")}) | [Worker receipt]({rel(args.run / "receipt.json")}) | '
              f'[Audit receipt]({rel(args.audit / "receipt.json")}) | [Original parent]({rel(args.terminal)})'), '']
    values = {'rows': rows, 'condition_counts': counts, 'gate_passed': passed, 'gate_total': 30,
              'pilot_continuation': summary['pilot_continuation'], 'regimes': summary['regimes'],
              'conditions': {k: summary[k] for k in counts}, 'amortization': summary['amortization'],
              'costs': summary['costs'], 'separate_cost_intervals': costs}
    return '\n'.join(lines), values


def execute(args):
    r = Report(args)
    require(args.output.is_absolute() and not args.output.exists() and not any(p.is_symlink() for p in args.output.parents), 'exclusive output')
    args.output.mkdir(parents=True, exist_ok=False)
    old_alarm = signal.getsignal(signal.SIGALRM)
    try:
        require(digest(regular(ROOT / H.CLOCK))['sha256'] == H.CLOCK_PIN, 'qualified clock')
        r.clock = H.load(ROOT / H.CLOCK, '_conditioning_report_clock').SuspendClock()
        r.start = r.clock.now_ns()
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('report deadline')))
        signal.setitimer(signal.ITIMER_REAL, LIMITS['seconds'])
        summary, costs = r.authenticate()
        markdown, values = present(summary, costs, args)
        with (args.output / REPORT).open('x') as stream:
            stream.write(markdown)
        values['figure'] = figure(values['rows'], values['gate_passed'], args.output)
        write(args.output / 'plotted-values.json', values)
        r.check()
        finished = r.clock.now_ns()
        r.receipt.update(status='completed', source={'path': str(Path(__file__)), **digest(Path(__file__), r.check)},
                         helper_sha256=HELPER_PIN, clock_backend=r.clock.backend, started_ns=r.start, finished_ns=finished,
                         wall_seconds=(finished - r.start) / 1e9,
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
        except BaseException as secondary:  # noqa: BLE001 - Preserve the original failure.
            error.add_note(f'Failure receipt: {secondary!r}')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'run', 'terminal', 'audit', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    for flag in ('plan-sha256', 'receipt-sha256', 'terminal-sha256', 'audit-receipt-sha256'):
        parser.add_argument('--' + flag, required=True)
    execute(parser.parse_args())
