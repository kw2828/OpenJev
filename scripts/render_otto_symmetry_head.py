"""Present completed, independently audited symmetry-head outcomes and fixed paths.

Hashes checkpoints and training payloads without decoding them. No readout,
posterior, actor, training, simulator or new scientific scoring is performed.
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
VERSION = 'otto-symmetry-head-presentation-v1'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
RUNNER = 'scripts/study_otto_symmetry_head.py'
AUDITOR = 'scripts/audit_otto_symmetry_head.py'
SEEDS = (9101, 9102, 9103)
FAMILIES = ('shared', 'dense', 'dense_ensemble')
ARMS = tuple(f'{f}@{s}' for f in FAMILIES for s in SEEDS) + ('analytic_inbounds',)
REGIMES = {'lambda3': {'seed': 970001, 'lambda': 3, 'scope': 'Training-supported'},
           'lambda4': {'seed': 980001, 'lambda': 4, 'scope': 'Training-supported'},
           'lambda5': {'seed': 990001, 'lambda': 5, 'scope': 'Unseen supplied kernel'}}
LABELS = {'shared': 'Shared', 'dense': 'Dense', 'dense_ensemble': 'Dense ensemble', 'analytic_inbounds': 'Analytic'}
COLORS = {'shared': '#8b47b8', 'dense': '#2563eb', 'dense_ensemble': '#0f8b73', 'analytic_inbounds': '#c76a15'}
LIMITS = {'seconds': 600, 'rss_bytes': 4 * 1024**3, 'output_bytes': 128 * 1024**2}
SCOPE = ('Saved independently audited aggregates, all ten arms and all three settings, with every fit seed retained. '
         'All 54 conditions apply to the shared candidate. Fixed first-case replays use saved public positions; '
         'source markers are evaluator-only truth, and playback timing is illustrative. No model, simulator, '
         'actor, training, posterior or new scientific scoring calls. No previous failure is revised.')


def payload_names():
    names = {'started.json', 'runtime.json', 'native-setup.json', 'qualification.json', 'qualification.jsonl',
             'collection-transitions.jsonl', 'collection-episodes.jsonl', 'work-contexts.jsonl', 'work.jsonl',
             'fits.jsonl', 'fit-curves.jsonl', 'epoch-orders.jsonl', 'inference-setup.json',
             'eval-transitions.jsonl', 'eval-episodes.jsonl', 'evaluation.jsonl', 'summary.json',
             'pooled-weights.npz', 'pooling.json', 'training-costs.json'}
    names.update(f'kernel-{r}.npz' for r in REGIMES)
    for split in ('train', 'valid', 'dagger'):
        names.update((f'{split}-data.npz', f'{split}-rows.jsonl'))
    names.update(f'{phase}-{family}-{seed}.npz' for phase in ('initial', 'final')
                 for seed in SEEDS for family in ('shared', 'dense'))
    return names


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
                        'model_calls': 0, 'simulator_calls': 0, 'policy_calls': 0, 'training_calls': 0}

    def check(self):
        require(self.clock.now_ns() - self.start < LIMITS['seconds'] * 10**9, 'presentation native deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= LIMITS['rss_bytes'], 'presentation RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'presentation output cap')

    def manifest(self, directory, pin, expected):
        require(directory.is_absolute() and directory.is_dir()
                and not any(p.is_symlink() for p in (directory, *directory.parents)), 'absolute nonsymlink evidence directory')
        receipt_path = regular(directory / 'receipt.json')
        require(digest(receipt_path, self.check)['sha256'] == pin, 'external completed receipt pin')
        receipt = read(receipt_path)
        require(receipt['status'] == 'completed' and set(receipt['files']) == expected
                and {p.name for p in directory.iterdir()} == expected | {'receipt.json'}, 'exact completed payload closure')
        for name, descriptor in receipt['files'].items():
            require(digest(regular(directory / name), self.check) == descriptor, f'payload bytes: {name}')
        return receipt

    def authenticate(self):
        args = self.args
        worker = self.manifest(args.run, args.receipt_sha256, payload_names())
        audit = self.manifest(args.audit, args.audit_receipt_sha256, {'started.json', 'summary.json'})
        require(worker['version'] == 'otto-symmetry-head-v1' and worker['completed_episodes'] == 720
                and worker['completed_stage_fits'] == 12 and worker['collection_episodes'] == 384
                and worker['external_model_calls'] == 0 and worker['pending'] == []
                and 0 < worker['training_updates'] <= 63360
                and all(v['attempted'] == v['returned'] for v in worker['calls'].values()), 'complete scientific work')
        require(audit['version'] == 'otto-symmetry-head-saved-audit-v1' and audit['agreement'] is True
                and audit['worker_sha256'] == args.receipt_sha256 and audit['plan_sha256'] == worker['plan_sha256']
                and audit['training_calls'] == audit['simulator_calls'] == audit['remote_model_calls'] == 0,
                'independent successful saved audit')
        request = read(args.audit / 'started.json')['request']
        require(request['run'] == str(args.run) and request['receipt_sha256'] == args.receipt_sha256
                and request['plan_sha256'] == audit['plan_sha256']
                and request['terminal_sha256'] == audit['terminal_sha256'], 'audit input joins')
        plan_path, terminal_path = regular(Path(request['plan'])), regular(Path(request['terminal']))
        require(digest(plan_path, self.check)['sha256'] == audit['plan_sha256']
                and digest(terminal_path, self.check)['sha256'] == audit['terminal_sha256'], 'audited plan and parent pins')
        plan, terminal = read(plan_path), read(terminal_path)
        require(plan['version'] == worker['version'] and plan['status'] == 'frozen_before_native_run'
                and plan['sources'] == worker['sources'] and plan['inputs'] == worker['inputs']
                and plan['limits'] == worker['limits']
                and plan['sources'][RUNNER] == audit['producer_source_sha256']
                and plan['sources'][AUDITOR] == audit['source']['sha256'], 'prospective source and producer identity')
        for name, pin in plan['sources'].items():
            relative = Path(name)
            require(not relative.is_absolute() and '..' not in relative.parts
                    and digest(regular(ROOT / relative), self.check)['sha256'] == pin, f'unchanged source: {name}')
        for role, record in plan['inputs'].items():
            relative = Path(record['path'])
            require(not relative.is_absolute() and '..' not in relative.parts
                    and digest(regular(ROOT / relative), self.check) == {'sha256': record['sha256'], 'bytes': record['bytes']},
                    f'unchanged input: {role}')
        started = read(args.run / 'started.json')
        command_request, launch = started['request'], started['launch']
        launch_path = regular(Path(command_request['supervision']))
        require(digest(launch_path, self.check)['sha256'] == worker['supervision_sha256']
                and read(launch_path) == launch, 'exact worker launch witness')
        require(command_request == {'plan': str(plan_path), 'plan_sha256': audit['plan_sha256'],
                                    'output': str(args.run), 'supervision': str(launch_path)}, 'exact worker request')
        for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns',
                    'clock_backend', 'cap_seconds', 'clock_source_sha256', 'watchdog_sha256'):
            require(terminal[key] == launch[key], 'parent terminal/launch identity')
        command = list(launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        require(command[:2] == [plan['python_executable'], str(ROOT / RUNNER)] and len(command) == 10
                and dict(zip(command[2::2], command[3::2], strict=True))
                == {f'--{key.replace("_", "-")}': value for key, value in command_request.items()}, 'actual study command')
        require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
                and terminal['group_absent'] is True and terminal['cleanup']['group_absent'] is True
                and terminal['cleanup']['reaped'] is True and terminal['cleanup']['errors'] == []
                and terminal['error'] is None and terminal['clock_error'] is None and terminal['timing_available'] is True
                and launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
                and worker['clock_backend'] == launch['clock_backend'] and launch['clock_source_sha256'] == CLOCK_PIN
                and launch['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py']
                and launch['pid'] == launch['pgid'] and launch['pid'] != launch['parent_pid'] and launch['cwd'] == str(ROOT)
                and launch['cap_seconds'] == 5400 and launch['deadline_ns'] == launch['started_ns'] + 5400 * 10**9
                and launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns'],
                'complete parent and strict native timing enclosure')
        require(started['started_ns'] == worker['started_ns']
                and worker['wall_seconds'] == (worker['finished_ns'] - worker['started_ns']) / 1e9
                and terminal['elapsed_ns'] == terminal['finished_ns'] - terminal['started_ns']
                and terminal['wall_seconds'] == terminal['elapsed_ns'] / 1e9, 'native elapsed identities')
        require(0 < worker['peak_rss_bytes'] <= plan['limits']['rss_bytes']
                and sum(p.stat().st_size for p in args.run.iterdir()) <= plan['limits']['output_bytes'], 'saved resource limits')
        result = read(args.audit / 'summary.json')
        require(result['version'] == 'otto-symmetry-head-saved-audit-v1' and result['agreement'] is True,
                'successful independent audit summary')
        aggregate = result
        require(aggregate['episodes'] == 720 and set(aggregate['regimes']) == set(REGIMES)
                and aggregate['learned_architecture_advantage_established'] is False
                and aggregate['inherited_gate_revised'] is False, 'complete bounded aggregate scope')
        rules = self.rules(aggregate)
        require(worker['pilot_continuation'] is rules['overall']['passes'], 'worker/audit decision identity')
        # Compare all displayed values and exact decisions to the authenticated
        # producer summary. This is a presentation join, not a second scoring run.
        producer = read(args.run / 'summary.json')
        for key in ('competence_checks', 'compression_checks', 'architecture_checks', 'pilot_continuation'):
            self.agree(producer[key], aggregate[key])
        for regime in REGIMES:
            for key in ('weights', 'means', 'family_means', 'blocks', 'strata', 'raw_counts'):
                self.agree(producer['regimes'][regime][key], aggregate['regimes'][regime][key])
        self.receipt['inputs'] = {
            'worker': {'path': str(args.run / 'receipt.json'), 'sha256': args.receipt_sha256},
            'audit': {'path': str(args.audit / 'receipt.json'), 'sha256': args.audit_receipt_sha256},
            'plan': {'path': str(plan_path), **digest(plan_path, self.check)},
            'terminal': {'path': str(terminal_path), **digest(terminal_path, self.check)},
            'audit_summary': {'path': str(args.audit / 'summary.json'), **digest(args.audit / 'summary.json', self.check)},
            'producer_summary': {'path': str(args.run / 'summary.json'), **worker['files']['summary.json']},
            'evaluation': {'path': str(args.run / 'evaluation.jsonl'), **worker['files']['evaluation.jsonl']},
            'transitions': {'path': str(args.run / 'eval-transitions.jsonl'), **worker['files']['eval-transitions.jsonl']}}
        return aggregate

    @staticmethod
    def agree(actual, expected):
        if isinstance(expected, dict):
            require(isinstance(actual, dict) and set(actual) == set(expected), 'display mapping keys')
            for key in expected:
                Render.agree(actual[key], expected[key])
        elif isinstance(expected, list):
            require(isinstance(actual, list) and len(actual) == len(expected), 'display list length')
            for left, right in zip(actual, expected, strict=True):
                Render.agree(left, right)
        elif type(expected) is float:
            require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
                    and abs(actual - expected) <= 1e-9 + 2e-12 * abs(expected), 'audited displayed numeric agreement')
        else:
            require(type(actual) is type(expected) and actual == expected, 'exact display value/status agreement')

    @staticmethod
    def rules(summary):
        result = {}
        names = set()
        for group, total in (('competence', 18), ('compression', 12), ('architecture', 24)):
            conditions = summary[f'{group}_checks']
            require(len(conditions) == total and all(type(c['passes']) is bool for c in conditions), 'complete criterion statuses')
            for condition in conditions:
                require(condition['name'] not in names, 'unique criterion identity')
                names.add(condition['name'])
            passed = sum(c['passes'] for c in conditions)
            result[group] = {'candidate': 'shared', 'passed': passed, 'total': total,
                             'passes': passed == total, 'conditions': conditions}
        passed = sum(g['passed'] for g in result.values())
        require(type(summary['pilot_continuation']) is bool and summary['pilot_continuation'] == (passed == 54), 'all54 continuation identity')
        result['overall'] = {'candidate': 'shared', 'passed': passed, 'total': 54, 'passes': summary['pilot_continuation']}
        return result

    def chart(self, summary):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                             'axes.spines.right': False, 'svg.hashsalt': VERSION})
        metrics = (('found', 'Success (%)', 100.), ('steps', 'Capped moves', 1.),
                   ('controller_seconds', 'Complete controller seconds (log)', 1.))
        fig, axes = plt.subplots(3, 3, figsize=(19, 13))
        points = []
        for row, (regime, spec) in enumerate(REGIMES.items()):
            data = summary['regimes'][regime]
            require(set(data['means']) == set(ARMS) and set(data['family_means']) == set(FAMILIES)
                    and len(data['blocks']) == 8 and all(set(b) == set(ARMS) for b in data['blocks']), 'all arms/seeds/block coverage')
            for col, (metric, title, scale) in enumerate(metrics):
                self.check()
                ax = axes[row, col]
                values = [float(data['means'][a][metric]) * scale for a in ARMS]
                require(all(math.isfinite(v) and v >= 0 for v in values), 'finite displayed means')
                colors = [COLORS[a.split('@')[0]] for a in ARMS]
                ax.bar(range(10), values, width=.7, color=colors, alpha=.9, zorder=2)
                for x, arm in enumerate(ARMS):
                    block_values = [float(b[arm]) for b in data['blocks']] if metric == 'steps' else []
                    for block, value in enumerate(block_values):
                        ax.scatter(x + (block - 3.5) * .05, value, s=13, facecolors='white',
                                   edgecolors='#27364b', linewidths=.5, zorder=3)
                    label = f'{values[x]:.1f}' if metric != 'controller_seconds' else f'{values[x]:.2g}'
                    ax.annotate(label, (x, values[x]), xytext=(0, 5), textcoords='offset points',
                                ha='center', fontsize=8, rotation=90, zorder=4)
                    points.append({'regime': regime, 'arm': arm, 'metric': metric,
                                   'mean': float(data['means'][arm][metric]), 'display_scale': scale,
                                   'move_block_values': block_values})
                for index, family in enumerate(FAMILIES):
                    mean = float(data['family_means'][family][metric]) * scale
                    ax.plot([index * 3 - .4, index * 3 + 2.4], [mean, mean], color='#17283f', linewidth=1.4, zorder=4)
                ax.set_xticks(range(10), [str(s) for _ in FAMILIES for s in SEEDS] + ['Ref'], fontsize=8)
                ax.set_title(title, loc='left', fontsize=13, pad=14)
                ax.grid(axis='y', alpha=.18, zorder=0)
                if metric == 'found':
                    ax.set_ylim(0, 119)
                    ax.set_yticks((0, 25, 50, 75, 100))
                elif metric == 'controller_seconds':
                    require(all(v > 0 for v in values), 'positive logarithmic controller costs')
                    ax.set_yscale('log')
                    ax.margins(y=.42)
                else:
                    top = max([*values, *(float(v) for b in data['blocks'] for v in b.values())])
                    ax.set_ylim(0, max(1., top) * 1.24)
                if col == 0:
                    ax.set_ylabel(f"{spec['scope']}\nsensing length {spec['lambda']}", fontsize=11)
        rules = self.rules(summary)
        status = [f"{key.title()}: {'PASS' if record['passes'] else 'FAIL'} {record['passed']}/{record['total']}"
                  for key, record in rules.items()]
        fig.suptitle('OTTO: full-belief learned readouts with symmetry', x=.06, y=.99, ha='left', fontsize=23, weight='bold')
        fig.text(.06, .952, 'All 720 autonomous evaluations. Shared candidate vs dense augmentation, dense ensemble and analytic reference.', fontsize=13)
        fig.text(.06, .925, ' | '.join(status), fontsize=12)
        fig.legend(handles=[Patch(facecolor=COLORS[f], label=LABELS[f]) for f in (*FAMILIES, 'analytic_inbounds')],
                   loc='lower center', bbox_to_anchor=(.5, .05), ncol=4, frameon=False, fontsize=12)
        fig.text(.06, .032, 'Bars: each fixed fit seed and reference, using initial-hit-weighted means. Short lines: three-seed family means.', fontsize=11, color='#405168')
        fig.text(.06, .014, 'Move dots: eight paired blocks, no confidence intervals. Complete controller cost includes setup, features and updates. One CPU run; no novelty claim.',
                 fontsize=10, color='#405168')
        fig.subplots_adjust(left=.075, right=.987, bottom=.13, top=.865, hspace=.45, wspace=.29)
        fig.savefig(self.out / 'symmetry-comparison.png', dpi=180, facecolor='white')
        fig.savefig(self.out / 'symmetry-comparison.svg', facecolor='white', metadata={'Date': None})
        plt.close(fig)
        write(self.out / 'plotted-values.json', {'scope': SCOPE, 'points': points, 'rules': rules,
              'family_means': {r: summary['regimes'][r]['family_means'] for r in REGIMES},
              'candidate': 'shared', 'fit_seeds': SEEDS, 'pilot_continuation': summary['pilot_continuation'],
              'confidence_intervals_shown': False})
        self.receipt.update(matplotlib_version=matplotlib.__version__, displayed_arm_metric_means=len(points), rules=rules)
        self.check()

    def replay_cases(self):
        selected = {(regime, spec['seed'], arm): {'packets': [], 'actions': []}
                    for regime, spec in REGIMES.items() for arm in ARMS}
        by_id = {}
        count = 0
        with (self.args.run / 'evaluation.jsonl').open() as stream:
            for line in stream:
                self.check()
                row = json.loads(line)
                count += 1
                key = (row['regime'], row['seed'], row['arm'])
                if key in selected:
                    require('episode' not in selected[key] and row['initial_hit'] == 1 and row['block'] == 0
                            and row['episode_id'] == f"eval:{key[0]}:{key[1]}:{key[2]}", 'unique prospectively fixed first case')
                    selected[key]['episode'] = row
                    by_id[row['episode_id']] = selected[key]
        require(count == 720 and len(by_id) == 30 and all('episode' in v for v in selected.values()), 'all30 fixed replay paths')
        with (self.args.run / 'eval-transitions.jsonl').open() as stream:
            for line in stream:
                self.check()
                row = json.loads(line)
                if row['episode_id'] not in by_id:
                    continue
                replay = by_id[row['episode_id']]
                step = len(replay['packets'])
                require(row['public']['step'] == step and row['kind'] == ('reset' if step == 0 else 'step'), 'contiguous saved public replay')
                replay['packets'].append(row['public'])
                if step:
                    replay['actions'].append({'step': step, 'action': row['action'], 'costs': row['costs'],
                                              'allowed_actions': row['allowed_actions']})
                else:
                    require(row['source_evaluation_only'] == replay['episode']['source_evaluation_only'], 'replay evaluator-source join')
        result = {}
        for regime, spec in REGIMES.items():
            arms = {}
            for arm in ARMS:
                row = selected[regime, spec['seed'], arm]
                episode = row['episode']
                require(len(row['packets']) == episode['steps'] + 1 and len(row['actions']) == episode['steps']
                        and row['packets'][-1] == episode['final_public']
                        and row['packets'][-1]['done'] == episode['found'], 'complete replay found/censored endpoint')
                arms[arm] = {'steps': episode['steps'], 'found': episode['found'], 'packets': row['packets'],
                             'actions': row['actions'], 'source_evaluation_only': episode['source_evaluation_only']}
            require(len({tuple(a['source_evaluation_only']) for a in arms.values()}) == 1, 'paired case source equality')
            longest = max(a['steps'] for a in arms.values())
            n = min(80, longest)
            frame_steps = sorted({round(i * longest / n) for i in range(n + 1)})
            result[regime] = {'seed': spec['seed'], 'case_index': 0, 'initial_hit': 1, 'lambda': spec['lambda'],
                              'selection': 'Protocol-fixed first case, no outcome-based selection.',
                              'frame_steps': frame_steps, 'arms': arms}
        write(self.out / 'replay-cases.json', {'scope': SCOPE, 'cases': result})
        return result

    def gif(self, regime, record):
        import PIL
        from matplotlib import font_manager
        from PIL import Image, ImageDraw, ImageFont

        font_path = Path(font_manager.findfont('DejaVu Sans'))
        font = ImageFont.truetype(str(font_path), 17)
        title = ImageFont.truetype(str(font_path), 23)
        small = ImageFont.truetype(str(font_path), 14)
        frames = []
        for global_step in record['frame_steps']:
            self.check()
            frame = Image.new('RGB', (1480, 775), '#f7f9fc')
            draw = ImageDraw.Draw(frame)
            draw.text((20, 12), f"{REGIMES[regime]['scope']} | length {record['lambda']} | fixed case 0 | seed {record['seed']}", fill='#14243d', font=title)
            draw.text((20, 44), 'All ten saved public paths. Red target = source (evaluator-only truth). Step playback is illustrative, not runtime.', fill='#405168', font=small)
            for index, arm in enumerate(ARMS):
                data = record['arms'][arm]
                step = min(global_step, data['steps'])
                packet = data['packets'][step]
                column, row, family = index % 5, index // 5, arm.split('@')[0]
                x0, y0, scale = 18 + column * 294, 110 + row * 326, 4.7

                def xy(position, x0=x0, y0=y0, scale=scale):
                    return x0 + position[1] * scale, y0 + position[0] * scale

                label = LABELS[family] + (f" {arm.split('@')[1]}" if '@' in arm else '')
                draw.text((x0, y0 - 27), label, fill=COLORS[family], font=font)
                draw.rectangle((x0 - 2, y0 - 2, x0 + 52 * scale + 2, y0 + 52 * scale + 2), fill='white', outline='#b7c4d5')
                for grid in range(0, 53, 13):
                    draw.line((x0 + grid * scale, y0, x0 + grid * scale, y0 + 52 * scale), fill='#e5ebf3')
                    draw.line((x0, y0 + grid * scale, x0 + 52 * scale, y0 + grid * scale), fill='#e5ebf3')
                points = [xy(p['position']) for p in data['packets'][:step + 1]]
                if len(points) > 1:
                    draw.line(points, fill=COLORS[family], width=2)
                sx, sy = xy(data['source_evaluation_only'])
                draw.ellipse((sx - 4, sy - 4, sx + 4, sy + 4), outline='#d12838', width=2)
                draw.line((sx - 7, sy, sx + 7, sy), fill='#d12838', width=2)
                draw.line((sx, sy - 7, sx, sy + 7), fill='#d12838', width=2)
                ox, oy = points[0]
                draw.ellipse((ox - 3, oy - 3, ox + 3, oy + 3), fill='#14243d')
                ax, ay = points[-1]
                draw.ellipse((ax - 4, ay - 4, ax + 4, ay + 4), fill=COLORS[family], outline='white')
                status = 'found' if packet['done'] else 'censored' if step == data['steps'] else 'searching'
                draw.text((x0, y0 + 254), f"Move {step}/{data['steps']} | {status}", fill='#14243d', font=small)
                draw.text((x0, y0 + 274), f"Hit {packet['hit']} | row {packet['position'][0]}, col {packet['position'][1]}", fill='#405168', font=small)
            frames.append(frame.convert('P', palette=Image.Palette.ADAPTIVE, colors=128))
        durations = [120] * len(frames)
        durations[-1] = 1400
        frames[0].save(self.out / f'replay-{regime}-case0.gif', save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, optimize=False, disposal=2)
        self.receipt.update(pillow_version=PIL.__version__, replay_font={'path': str(font_path), **digest(font_path, self.check)},
                            replay_frame_rule='At most 81 evenly spaced decision indices, both endpoints; 120 ms per frame, final 1400 ms.')
        self.check()

    def execute(self):
        require(self.out.is_absolute() and not any(p.is_symlink() for p in (self.out, *self.out.parents)), 'exclusive absolute nonsymlink output')
        self.out.mkdir(parents=True, exist_ok=False)
        previous = signal.getsignal(signal.SIGALRM)
        try:
            require(digest(ROOT / CLOCK)['sha256'] == CLOCK_PIN, 'clock source before import')
            self.clock = load(ROOT / CLOCK, '_symmetry_presentation_clock').SuspendClock()
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
                finished_ns=self.clock.now_ns(), replay_seeds=[v['seed'] for v in REGIMES.values()],
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
            except BaseException as secondary:  # noqa: BLE001 - preserve primary presentation failure.
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
