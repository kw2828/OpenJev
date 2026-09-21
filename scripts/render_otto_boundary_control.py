"""Render the audited boundary control and two fixed four-arm case-zero replays.

No simulator, actor, model, posterior computation, fitting or outcome selection.
The independently rebuilt audit aggregates supply the chart. Replays show saved
public positions only, with the source explicitly labeled evaluator-only truth.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import signal
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'otto-boundary-control-presentation-v1'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
AUDITOR = 'scripts/audit_otto_boundary_control.py'
RUNNER = 'scripts/study_otto_boundary_control.py'
ARMS = ('released_tf', 'released_inbounds', 'analytic_all4', 'analytic_inbounds')
REGIMES = {'base': {'seed': 870001, 'lambda': 3}, 'shift': {'seed': 880001, 'lambda': 4}}
LABELS = {'released_tf': 'Original TF: all 4', 'released_inbounds': 'TF: in bounds (candidate)',
          'analytic_all4': 'Analytic: all 4', 'analytic_inbounds': 'Analytic: in bounds'}
COLORS = {'released_tf': '#2563eb', 'released_inbounds': '#8b47b8',
          'analytic_all4': '#0f8b73', 'analytic_inbounds': '#c76a15'}
RUN_FILES = {'started.json', 'runtime.json', 'setup.json', 'weights.jsonl', 'work.jsonl',
             'forwards.jsonl', 'transitions.jsonl', 'episodes.jsonl', 'summary.json', 'paired-prefixes.jsonl'}
LIMITS = {'seconds': 300, 'rss_bytes': 4 * 1024**3, 'output_bytes': 128 * 1024**2}
SCOPE = ('Saved, independently audited aggregates and prospectively selected case zero in each regime. '
         'Four arms and both regimes retained. GIF playback follows evenly spaced decision steps, '
         'not measured wall time. Source markers expose evaluator truth for the viewer only. '
         'All four reported rule families apply to the released_inbounds candidate. '
         'No model, simulator, actor or new scientific scoring calls; no earlier failure is revised.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def digest(path, check=lambda: None):
    value, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            check()
            value.update(chunk)
            size += len(chunk)
    return {'sha256': value.hexdigest(), 'bytes': size}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def regular(path):
    require(path.is_absolute() and path.is_file()
            and not any(p.is_symlink() for p in (path, *path.parents)), 'absolute nonsymlink input file')
    return path


class Render:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = None
        self.receipt = {'version': VERSION, 'status': 'started', 'scope': SCOPE, 'limits': LIMITS,
                        'model_calls': 0, 'simulator_calls': 0, 'policy_calls': 0}

    def check(self):
        require(self.clock.now_ns() - self.start < LIMITS['seconds'] * 10**9, 'presentation suspend-inclusive deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= LIMITS['rss_bytes'], 'presentation RSS limit')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'presentation output limit')

    def manifest(self, directory, pin, expected):
        require(directory.is_absolute() and directory.is_dir() and not directory.is_symlink(), 'absolute complete evidence directory')
        receipt_path = regular(directory / 'receipt.json')
        require(digest(receipt_path, self.check)['sha256'] == pin, 'external completed receipt hash')
        receipt = read(receipt_path)
        require(receipt['status'] == 'completed' and set(receipt['files']) == expected
                and {p.name for p in directory.iterdir()} == expected | {'receipt.json'}, 'complete undemoted evidence closure')
        for name, descriptor in receipt['files'].items():
            require(digest(regular(directory / name), self.check) == descriptor, f'exact payload: {name}')
        return receipt

    def authenticate(self):
        args = self.args
        worker = self.manifest(args.run, args.receipt_sha256, RUN_FILES)
        audit = self.manifest(args.audit, args.audit_receipt_sha256, {'started.json', 'summary.json', 'report.md'})
        require(worker['version'] == 'otto-boundary-control-v1' and worker['completed_episodes'] == 768
                and worker['training_updates'] == worker['external_model_calls'] == 0
                and worker['numpy_port_admitted'] is False and worker['inherited_gate_revised'] is False
                and worker['candidate_arm'] == 'released_inbounds' and worker['paired_neural_prefixes'] == 192,
                'complete nontraining boundary run')
        require(audit['version'] == 'otto-boundary-control-saved-audit-v1' and audit['agreement'] is True
                and audit['worker_sha256'] == args.receipt_sha256 and audit['plan_sha256'] == worker['plan_sha256']
                and audit['model_calls'] == audit['simulator_calls'] == audit['policy_calls'] == 0,
                'completed independent audit of this exact run')
        request = read(args.audit / 'started.json')['request']
        require(request['run'] == str(args.run) and request['receipt_sha256'] == args.receipt_sha256
                and request['plan_sha256'] == audit['plan_sha256'] and request['terminal_sha256'] == audit['terminal_sha256'],
                'audit request joins')
        plan_path, terminal_path = regular(Path(request['plan'])), regular(Path(request['terminal']))
        require(digest(plan_path, self.check)['sha256'] == audit['plan_sha256']
                and digest(terminal_path, self.check)['sha256'] == audit['terminal_sha256'], 'audited plan and actual terminal hashes')
        plan, terminal = read(plan_path), read(terminal_path)
        require(plan['version'] == 'otto-boundary-control-v1' and plan['status'] == 'frozen_before_native_run'
                and plan['sources'] == worker['sources'] and plan['inputs'] == worker['inputs']
                and plan['sources'][AUDITOR] == audit['source']['sha256']
                and plan['sources'][RUNNER] == audit['producer_source_sha256'], 'prospective producer and independent reader source joins')
        for name, pin in plan['sources'].items():
            relative = Path(name)
            require(not relative.is_absolute() and '..' not in relative.parts
                    and digest(regular(ROOT / relative), self.check)['sha256'] == pin, f'unchanged source: {name}')
        for name, record in plan['inputs'].items():
            relative = Path(record['path'])
            require(not relative.is_absolute() and '..' not in relative.parts
                    and digest(regular(ROOT / relative), self.check) == {'sha256': record['sha256'], 'bytes': record['bytes']},
                    f'unchanged input: {name}')
        require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
                and terminal['group_absent'] is True and terminal['cleanup']['errors'] == []
                and terminal['cleanup']['group_absent'] is True and terminal['cleanup']['reaped'] is True
                and terminal['error'] is None and terminal['clock_error'] is None and terminal['timing_available'] is True
                and terminal['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < terminal['deadline_ns'],
                'successful cleaned native supervisor')
        result = read(args.audit / 'summary.json')
        require(result['version'] == 'otto-boundary-control-saved-audit-v1' and result['agreement'] is True
                and result['episodes'] == 768 and set(result['cohorts']) == set(REGIMES)
                and len(result['paired_neural_prefixes']) == 192
                and result['condition_counts'] == {'restriction_benefit': 5, 'competence': 6,
                    'promising_teacher_candidate': 12, 'utility_compute': 16}, 'complete independently reconstructed aggregates')
        rules = self.rules(result)
        require(all(worker[key] is record['passes'] for key, record in rules.items()), 'audited rule flags match completed worker')
        self.receipt['inputs'] = {'run_receipt': {'path': str(args.run / 'receipt.json'), 'sha256': args.receipt_sha256},
            'audit_receipt': {'path': str(args.audit / 'receipt.json'), 'sha256': args.audit_receipt_sha256},
            'plan': {'path': str(plan_path), **digest(plan_path, self.check)},
            'terminal': {'path': str(terminal_path), **digest(terminal_path, self.check)},
            'audited_summary': {'path': str(args.audit / 'summary.json'), **digest(args.audit / 'summary.json', self.check)},
            'producer_summary': {'path': str(args.run / 'summary.json'), **digest(args.run / 'summary.json', self.check)},
            'transitions': {'path': str(args.run / 'transitions.jsonl'), **worker['files']['transitions.jsonl']},
            'episodes': {'path': str(args.run / 'episodes.jsonl'), **worker['files']['episodes.jsonl']}}
        return result

    @staticmethod
    def rules(summary):
        groups = {'restriction_benefit': (5, summary['restriction_benefit_checks'])}
        for flag, key, total in (('competent_reference', 'competence_checks', 6),
                                 ('stronger_value_teacher', 'stronger_teacher_checks', 12),
                                 ('utility_compute_advantage', 'utility_compute_checks', 16)):
            groups[flag] = (total, [{**c, 'regime': regime} for regime in REGIMES
                                   for c in summary['cohorts'][regime][key]])
        result = {}
        for flag, (total, conditions) in groups.items():
            require(len(conditions) == total and all(type(c['passes']) is bool for c in conditions), 'complete audited condition statuses')
            passed = sum(c['passes'] for c in conditions)
            require(type(summary[flag]) is bool and summary[flag] == (passed == total), 'audited rule status consistency')
            result[flag] = {'candidate': 'released_inbounds', 'passed': passed, 'total': total,
                            'passes': summary[flag], 'conditions': conditions}
        return result

    def chart(self, summary):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False,
                             'axes.spines.right': False, 'svg.hashsalt': VERSION})
        metrics = (('found', 'Success (%)', 100.), ('capped_time', 'Capped moves', 1.),
                   ('controller_seconds', 'Controller seconds (log scale)', 1.))
        fig, axes = plt.subplots(2, 3, figsize=(17, 9.2))
        points = []
        for row, (regime, specification) in enumerate(REGIMES.items()):
            data = summary['cohorts'][regime]
            require(set(data['means']) == set(ARMS) and [b['block'] for b in data['blocks']] == list(range(8))
                    and all(set(b['means']) == set(ARMS) for b in data['blocks']), 'complete plotted arm and block coverage')
            for col, (metric, title, scale) in enumerate(metrics):
                ax = axes[row, col]
                values = [float(data['means'][a][metric]) * scale for a in ARMS]
                require(all(math.isfinite(v) and v >= 0 for v in values), 'finite plotted means')
                ax.bar(range(4), values, width=.62, color=[COLORS[a] for a in ARMS], alpha=.9, zorder=2)
                for x, arm in enumerate(ARMS):
                    block_values = [float(b['means'][arm][metric]) * scale for b in data['blocks']]
                    require(all(math.isfinite(v) and v >= 0 for v in block_values), 'finite plotted block means')
                    for block, value in enumerate(block_values):
                        ax.scatter(x + (block - 3.5) * .042, value, s=21, facecolors='white',
                                   edgecolors='#27364b', linewidths=.6, zorder=3)
                    label = f'{values[x]:.1f}' if metric != 'controller_seconds' else f'{values[x]:.3g}'
                    ax.annotate(label, (x, values[x]), xytext=(0, 8), textcoords='offset points',
                                ha='center', fontsize=12, weight='bold', zorder=4)
                    points.append({'regime': regime, 'arm': arm, 'metric': metric,
                                   'mean': float(data['means'][arm][metric]), 'display_scale': scale,
                                   'block_values': [float(b['means'][arm][metric]) for b in data['blocks']]})
                ax.set_xticks(range(4), ['Original TF\nall 4', 'TF in bounds\n(candidate)',
                                        'Analytic\nall 4', 'Analytic\nin bounds'], fontsize=10)
                ax.set_title(title, loc='left', fontsize=14, pad=13)
                ax.grid(axis='y', alpha=.18, zorder=0)
                if metric == 'found':
                    ax.set_ylim(0, 116)
                    ax.set_yticks((0, 25, 50, 75, 100))
                elif metric == 'controller_seconds':
                    require(all(v > 0 for v in values) and all(v > 0 for a in ARMS for b in data['blocks']
                                                             for v in [float(b['means'][a][metric])]), 'positive logged controller durations')
                    ax.set_yscale('log')
                    ax.margins(y=.35)
                else:
                    ax.set_ylim(0, max([*values, *(float(b['means'][a][metric]) for b in data['blocks'] for a in ARMS)]) * 1.22)
                if col == 0:
                    ax.set_ylabel(f"{'Baseline' if regime == 'base' else 'Known-kernel shift'}\nlambda = {specification['lambda']}", fontsize=13)
        rules = self.rules(summary)
        display = [('restriction_benefit', 'Restriction benefit'), ('competent_reference', 'Competence'),
                   ('stronger_value_teacher', 'Teacher candidate'), ('utility_compute_advantage', 'Utility / compute')]
        statuses = [f"{label}: {'PASS' if rules[key]['passes'] else 'FAIL'} {rules[key]['passed']}/{rules[key]['total']}"
                    for key, label in display]
        fig.suptitle('OTTO: restrict neural selection to moving directions?', x=.065, y=.99, ha='left', fontsize=22, weight='bold')
        fig.text(.065, .946, 'Candidate: TF in bounds. Unchanged four raw costs; selection restriction only. 768 autonomous episodes.', fontsize=13)
        fig.text(.065, .915, ' | '.join(statuses), fontsize=12)
        fig.legend(handles=[Patch(facecolor=COLORS[a], label=LABELS[a]) for a in ARMS], loc='lower center',
                   bbox_to_anchor=(.5, .047), ncol=4, frameon=False, fontsize=10)
        fig.text(.065, .031, 'Bars: initial-hit-weighted means. Dots: eight fixed paired blocks; no confidence intervals.', fontsize=10, color='#405168')
        fig.text(.065, .010, 'Controller cost includes full choices, updates and setup allocation; measured evidence I/O excluded. One CPU run. No new architecture claim.',
                 fontsize=10, color='#405168')
        fig.subplots_adjust(left=.085, right=.985, bottom=.15, top=.835, hspace=.42, wspace=.28)
        fig.savefig(self.out / 'boundary-comparison.png', dpi=200, facecolor='white')
        fig.savefig(self.out / 'boundary-comparison.svg', facecolor='white', metadata={'Date': None})
        plt.close(fig)
        values = {'scope': SCOPE, 'points': points, 'candidate_arm': 'released_inbounds', 'rules': rules,
                  **{key: record['passes'] for key, record in rules.items()},
                  'prior_numpy_port_qualified': False, 'block_dots_are_confidence_intervals': False}
        write(self.out / 'plotted-values.json', values)
        self.receipt['matplotlib_version'] = matplotlib.__version__
        self.check()

    def replay_cases(self):
        selected = {(name, spec['seed'], arm): {'packets': [], 'actions': []}
                    for name, spec in REGIMES.items() for arm in ARMS}
        episode_count = 0
        with (self.args.run / 'episodes.jsonl').open() as stream:
            for line in stream:
                self.check()
                row = json.loads(line)
                episode_count += 1
                key = (row['cohort'], row['seed'], row['arm'])
                if key in selected:
                    require('episode' not in selected[key] and row['block'] == 0 and row['initial_hit'] == 1, 'unique preselected case zero')
                    selected[key]['episode'] = row
        require(episode_count == 768 and all('episode' in r for r in selected.values()), 'all fixed replay episodes present')
        with (self.args.run / 'transitions.jsonl').open() as stream:
            for line in stream:
                self.check()
                row = json.loads(line)
                key = (row['cohort'], row['seed'], row['arm'])
                if key not in selected:
                    continue
                replay = selected[key]
                step = len(replay['packets'])
                require(row['public']['step'] == step and row['kind'] == ('reset' if step == 0 else 'step'), 'selected contiguous saved public path')
                replay['packets'].append(row['public'])
                if step:
                    replay['actions'].append({'step': step, 'action': row['action'], 'costs': row['costs'],
                                              'allowed_actions': row['allowed_actions']})
                else:
                    require(row['source_evaluation_only'] == replay['episode']['source_evaluation_only'], 'selected source join')
        result = {}
        for regime, spec in REGIMES.items():
            arms = {}
            for arm in ARMS:
                r = selected[regime, spec['seed'], arm]
                episode = r['episode']
                require(len(r['packets']) == episode['steps'] + 1 and len(r['actions']) == episode['steps']
                        and r['packets'][-1]['done'] == episode['found'], 'complete selected success or censored path')
                arms[arm] = {'steps': episode['steps'], 'found': episode['found'], 'packets': r['packets'],
                             'actions': r['actions'], 'source_evaluation_only': episode['source_evaluation_only']}
            require(len({tuple(r['source_evaluation_only']) for r in arms.values()}) == 1, 'same selected source across arms')
            longest = max(r['steps'] for r in arms.values())
            frame_steps = sorted({round(i * longest / min(80, longest)) for i in range(min(80, longest) + 1)})
            result[regime] = {'seed': spec['seed'], 'case_index': 0, 'initial_hit': 1, 'lambda': spec['lambda'],
                              'selection': 'Prospectively fixed first case, never selected using outcomes.',
                              'frame_steps': frame_steps, 'arms': arms}
        write(self.out / 'replay-cases.json', {'scope': SCOPE, 'cases': result})
        return result

    def gif(self, regime, record):
        import PIL
        from matplotlib import font_manager
        from PIL import Image, ImageDraw, ImageFont

        font_path = Path(font_manager.findfont('DejaVu Sans'))
        font = ImageFont.truetype(str(font_path), 17)
        title = ImageFont.truetype(str(font_path), 24)
        small = ImageFont.truetype(str(font_path), 14)
        frames = []
        for global_step in record['frame_steps']:
            self.check()
            frame = Image.new('RGB', (1060, 1040), '#f7f9fc')
            draw = ImageDraw.Draw(frame)
            draw.text((24, 14), f"{'Baseline' if regime == 'base' else 'Known-kernel shift'} | fixed case 0 | seed {record['seed']} | lambda={record['lambda']}", fill='#14243d', font=title)
            draw.text((24, 47), 'Saved public paths. Red target = source (evaluator-only truth). Playback timing is illustrative.', fill='#405168', font=small)
            for index, arm in enumerate(ARMS):
                data = record['arms'][arm]
                step = min(global_step, data['steps'])
                packet = data['packets'][step]
                column, row = index % 2, index // 2
                x0, y0, scale = 30 + column * 530, 116 + row * 484, 7.

                def xy(position, x0=x0, y0=y0, scale=scale):
                    return x0 + position[1] * scale, y0 + position[0] * scale

                draw.text((x0, y0 - 32), LABELS[arm], fill=COLORS[arm], font=font)
                draw.rectangle((x0 - 3, y0 - 3, x0 + 52 * scale + 3, y0 + 52 * scale + 3), fill='white', outline='#b7c4d5', width=1)
                for index in range(0, 53, 10):
                    draw.line((x0 + index * scale, y0, x0 + index * scale, y0 + 52 * scale), fill='#e5ebf3')
                    draw.line((x0, y0 + index * scale, x0 + 52 * scale, y0 + index * scale), fill='#e5ebf3')
                points = [xy(p['position']) for p in data['packets'][:step + 1]]
                if len(points) > 1:
                    draw.line(points, fill=COLORS[arm], width=3)
                sx, sy = xy(data['source_evaluation_only'])
                draw.ellipse((sx - 6, sy - 6, sx + 6, sy + 6), outline='#d12838', width=2)
                draw.line((sx - 9, sy, sx + 9, sy), fill='#d12838', width=2)
                draw.line((sx, sy - 9, sx, sy + 9), fill='#d12838', width=2)
                ox, oy = points[0]
                draw.ellipse((ox - 4, oy - 4, ox + 4, oy + 4), fill='#14243d')
                ax, ay = points[-1]
                draw.ellipse((ax - 5, ay - 5, ax + 5, ay + 5), fill=COLORS[arm], outline='white', width=1)
                status = 'found' if packet['done'] else 'censored' if step == data['steps'] else 'searching'
                draw.text((x0, y0 + 379), f"Move {step}/{data['steps']} | {status}", fill='#14243d', font=font)
                draw.text((x0, y0 + 404), f"Last public hit: {packet['hit']} | row {packet['position'][0]}, col {packet['position'][1]}", fill='#405168', font=small)
            frames.append(frame.convert('P', palette=Image.Palette.ADAPTIVE, colors=128))
        durations = [120] * len(frames)
        durations[-1] = 1400
        frames[0].save(self.out / f'replay-{regime}-case0.gif', save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, optimize=False, disposal=2)
        self.receipt.update(pillow_version=PIL.__version__, replay_font={'path': str(font_path), **digest(font_path, self.check)},
                            replay_frame_rule='At most 81 equally spaced decision indices, both endpoints retained; 120 ms frames, final 1400 ms.')
        self.check()

    def execute(self):
        require(self.out.is_absolute() and not any(p.is_symlink() for p in (self.out, *self.out.parents)),
                'absolute nonsymlink exclusive presentation output')
        self.out.mkdir(parents=True, exist_ok=False)
        previous = signal.getsignal(signal.SIGALRM)
        try:
            require(digest(ROOT / CLOCK)['sha256'] == CLOCK_PIN, 'registered clock before evidence reads')
            self.clock = load(ROOT / CLOCK, '_boundary_presentation_clock').SuspendClock()
            self.start = self.clock.now_ns()
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('presentation wall cap')))
            signal.setitimer(signal.ITIMER_REAL, LIMITS['seconds'])
            self.receipt['source'] = {'path': str(Path(__file__).resolve()), **digest(Path(__file__), self.check)}
            write(self.out / 'started.json', {'request': {k: str(v) for k, v in vars(self.args).items()},
                                            'limits': LIMITS, 'started_ns': self.start, 'clock_backend': self.clock.backend})
            summary = self.authenticate()
            self.chart(summary)
            replays = self.replay_cases()
            for regime in REGIMES:
                self.gif(regime, replays[regime])
            self.authenticate()
            self.check()
            self.receipt.update(status='completed', clock_backend=self.clock.backend, started_ns=self.start,
                                finished_ns=self.clock.now_ns(), replay_seeds=[r['seed'] for r in REGIMES.values()],
                                files={p.name: digest(p, self.check) for p in self.out.iterdir() if p.is_file()})
            self.receipt['wall_seconds'] = (self.receipt['finished_ns'] - self.start) / 1e9
            write(self.out / 'receipt.json', self.receipt)
            self.check()
            print(json.dumps({'status': 'completed', 'receipt_sha256': digest(self.out / 'receipt.json', self.check)['sha256']}))
            return self.receipt
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            self.receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            try:
                if (self.out / 'receipt.json').exists():
                    (self.out / 'receipt.json').rename(self.out / 'invalid-completed-receipt.json')
                write(self.out / 'receipt.json', self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - preserve the primary presentation failure.
                error.add_note(f'Failure receipt publication: {secondary!r}')
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('run', 'audit', 'output'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    for flag in ('receipt-sha256', 'audit-receipt-sha256'):
        parser.add_argument(f'--{flag}', required=True)
    Render(parser.parse_args()).execute()
