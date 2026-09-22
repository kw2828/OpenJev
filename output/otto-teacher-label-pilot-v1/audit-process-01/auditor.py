"""Independent saved-only audit of the fixed six-anchor teacher-cost pilot.

No collector, sampler, actor, cost reducer, simulator or old audit is imported.
Only NumPy saved-array/PCG64 operations and the pinned nonscientific clock run.
Teacher actions are checked against saved scores, not freshly computed scores.
Continuation posteriors are reconstructed, but the producer saved no posterior
witnesses for those updates; only anchor/filter-prefix witnesses can be compared.
This is evidence validation, never new rollouts, training or efficacy admission.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
import os
import platform
import resource
import signal
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'otto-teacher-label-saved-audit-v1'
PRODUCER = 'scripts/collect_otto_teacher_labels.py'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
SUPERVISOR = 'scripts/supervise_dialogue_observation_v2.py'
HORIZON, SEED, N, EPSILON = 2188, 19000001, 53, 1e-10
CELLS = (('lambda3', 1, 'shared@9101'), ('lambda3', 2, 'dense@9102'),
         ('lambda3', 3, 'shared@9103'), ('lambda4', 1, 'dense@9101'),
         ('lambda4', 2, 'shared@9102'), ('lambda4', 3, 'dense@9103'))
PUBLIC = {'position', 'step', 'hit', 'done', 'valid_actions'}
PAYLOADS = {'started.json', 'reconstruction.jsonl', 'anchors.json', 'anchors.npz',
            'events.jsonl', 'summary.json'} | {f'panel-{i}.json' for i in range(6)}
LIMITATIONS = [
    'No fresh teacher, sampler, simulator, model or training calls.',
    'Teacher choice matches saved scores and the near-tie rule; analytic scores are not recomputed.',
    'Continuation public posteriors are independently derived; the producer saved no continuation posterior witnesses.',
    'Frozen selected identities are authenticated, not reselected from historical TRAIN metadata.',
    'Historical audits and preparation qualifications are authenticated as inputs, not rerun.',
    'Original timing, code execution and fsync completion remain authenticated process evidence.',
    'Technical agreement does not establish useful supervision, policy efficacy or an architecture advantage.',
]


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')) + '\n').encode()


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f'nonfinite JSON constant: {value}')


def decode(value):
    return json.loads(value, object_pairs_hook=strict_pairs, parse_constant=reject_constant)


def regular(path):
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('absolute nonsymlink path required')
    return path


def relative(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('unsafe manifest path')
    return regular(ROOT / path)


def digest(path, check=lambda: None):
    regular(path)
    hashed, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            check()
            hashed.update(block)
            size += len(block)
    return {'sha256': hashed.hexdigest(), 'bytes': size}


def write(path, value):
    with path.open('xb') as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


class Audit:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock, self.start = None, None
        self.counts = collections.Counter()
        self.context, self.primary, self.descriptors = {}, {}, {}
        self.calls, self.events = collections.Counter(), collections.Counter()
        self.receipt = {'version': VERSION, 'status': 'started', 'agreement': False,
                        'limitations': LIMITATIONS, 'failures': [], 'fresh_teacher_calls': 0,
                        'sampler_calls': 0, 'native_calls': 0, 'model_calls': 0, 'training_calls': 0}

    def require(self, condition, label):
        self.counts['checks'] += 1
        if not condition:
            raise ValueError(label)

    def equal(self, actual, expected, label):
        self.require(encoded(actual) == encoded(expected), label)

    def finite(self, value, label):
        self.require(type(value) in (int, float) and math.isfinite(value) and value >= 0, label)

    def check(self):
        self.require(self.clock.now_ns() - self.start < self.args.max_seconds * 10**9, 'saved-audit deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        self.require(rss <= self.args.max_rss_bytes, 'saved-audit RSS bound')
        self.require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
                     <= self.args.max_output_bytes, 'saved-audit output bound')

    def read(self, path):
        self.check()
        self.require(path.stat().st_size <= 32 * 1024**2, 'bounded JSON metadata')
        return decode(path.read_bytes())

    def bind(self, path, expected=None):
        actual = digest(path, self.check)
        if expected is not None:
            self.equal(actual, expected, f'file identity: {path}')
        self.descriptors[str(path)] = actual
        return actual

    def authenticate(self):
        args = self.args
        for role, path, pin in (('plan', args.plan, args.plan_sha256),
                                ('worker', args.run / 'receipt.json', args.receipt_sha256),
                                ('terminal', args.terminal, args.terminal_sha256)):
            desc = self.bind(path)
            self.require(desc['sha256'] == pin, f'external {role} pin')
            self.primary[role] = {'path': str(path), **desc}
        self.plan, self.worker, terminal = (self.read(args.plan), self.read(args.run / 'receipt.json'),
                                           self.read(args.terminal))
        plan, worker = self.plan, self.worker
        self.require(plan['version'] == worker['version'] == 'otto-teacher-label-pilot-v1'
                     and plan['status'] == 'frozen_before_collection' and worker['status'] == 'completed'
                     and worker['technical_completion'] is True, 'completed frozen pilot')
        self.require(worker['plan_sha256'] == args.plan_sha256 and worker['requires_successful_original_supervisor'] is True
                     and worker['pending'] == [] and worker.get('pending_emission') is None
                     and worker.get('active_panel') is None and not worker.get('cleanup_errors'), 'closed worker lifecycle')
        for name in ('sources', 'inputs', 'limits'):
            self.equal(worker[name], plan[name], f'worker/plan {name}')
        for name in ('native_calls', 'learned_model_calls', 'training_calls'):
            self.require(worker[name] == plan['configuration'][name] == 0, f'zero {name}')
        config = plan['configuration']
        self.require(config['seed'] == SEED and config['horizon'] == HORIZON
                     and config['anchor_ids'] == list(range(6)) and config['replicate_ids'] == list(range(16))
                     and config['familywise_alpha'] == .05 and config['no_retries'] is True
                     and config['first_actions'] == 'all geometric inbounds, ascending', 'fixed sampling allocation')
        self.equal(plan['limits'], {'native_seconds': 1800, 'rss_bytes': 4 * 1024**3,
                                   'output_bytes': 4 * 1024**3}, 'fixed producer limits')
        projection = plan['serialization_projection']
        self.require(projection['hard_event_bytes'] == 512 and projection['maximum_sampler_events'] == 6723474
                     and projection['non_event_reserve_bytes'] == 256 * 1024**2
                     and projection['projected_bytes'] == 6723474 * 512 + 256 * 1024**2
                     and projection['projected_bytes'] <= plan['limits']['output_bytes'], 'frozen serialization allocation')
        self.require(set(worker['files']) == PAYLOADS
                     and {p.name for p in args.run.iterdir()} == PAYLOADS | {'receipt.json'}, 'exact completed payload closure')
        for name, desc in worker['files'].items():
            self.bind(args.run / name, desc)
        self.counts['worker_payloads_authenticated'] = len(PAYLOADS)
        for name, pin in plan['sources'].items():
            self.require(self.bind(relative(name))['sha256'] == pin, f'current source: {name}')
        for name, desc in plan['inputs'].items():
            self.bind(relative(desc.get('path', name)), {k: desc[k] for k in ('sha256', 'bytes')})
        self.counts['sources_authenticated'], self.counts['inputs_authenticated'] = len(plan['sources']), len(plan['inputs'])
        started = self.read(args.run / 'started.json')
        launch, request = started['launch'], started['request']
        launch_path = regular(Path(request['supervision']))
        self.require(self.bind(launch_path)['sha256'] == worker['supervision_sha256'], 'original launch hash')
        self.equal(self.read(launch_path), launch, 'original embedded launch')
        self.require(all(terminal[k] == v for k, v in launch.items()), 'launch/terminal identity')
        self.require(terminal['version'] == 'dialogue-observation-supervision-v2'
                     and terminal['status'] == 'completed' and terminal['returncode'] == 0
                     and terminal['timed_out'] is False and terminal['error'] is None
                     and terminal['clock_error'] is None and terminal['timing_available'] is True
                     and terminal['group_absent'] is True and terminal['cleanup']['reaped'] is True
                     and terminal['cleanup']['group_absent'] is True and terminal['cleanup']['errors'] == [],
                     'original successful parent and cleanup')
        self.require(request['plan'] == str(args.plan) and request['plan_sha256'] == args.plan_sha256
                     and request['output'] == str(args.run), 'original request identity')
        command = list(launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        self.equal(command, [plan['python_executable'], str(relative(PRODUCER)), 'run', '--plan', str(args.plan),
                             '--plan-sha256', args.plan_sha256, '--supervision', str(launch_path), '--output', str(args.run)],
                   'exact original command')
        self.require(launch['cwd'] == str(ROOT) and launch['pid'] == launch['pgid'] != launch['parent_pid']
                     and launch['clock_source_sha256'] == CLOCK_PIN
                     and launch['watchdog_sha256'] == plan['sources'][SUPERVISOR]
                     and launch['clock_backend'] == worker['clock_backend']
                     and launch['cap_seconds'] == 1800 and launch['deadline_ns'] == launch['started_ns'] + 1800 * 10**9,
                     'original process and clock binding')
        self.require(launch['started_ns'] <= worker['started_ns'] == started['started_ns']
                     <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns'], 'complete elapsed ordering')
        self.equal(worker['wall_seconds'], (worker['finished_ns'] - worker['started_ns']) / 1e9, 'worker elapsed arithmetic')
        self.equal(terminal['elapsed_ns'], terminal['finished_ns'] - terminal['started_ns'], 'parent elapsed arithmetic')
        self.equal(terminal['wall_seconds'], terminal['elapsed_ns'] / 1e9, 'parent seconds')
        self.require(worker['peak_rss_bytes'] <= plan['limits']['rss_bytes'], 'producer RSS bound')
        total = sum(p.stat().st_size for p in args.run.iterdir())
        self.require(total <= plan['limits']['output_bytes'], 'actual producer output bound')
        self.equal(worker['output_bytes_before_receipt'], sum(d['bytes'] for d in worker['files'].values()), 'producer output accounting')
        self.require(sys.executable == plan['python_executable'] and platform.python_version() == plan['python_version'],
                     'same Python for exact saved numerical replay')
        self.receipt.update(primary_files=self.primary, producer_sources=plan['sources'], producer_inputs=plan['inputs'],
                            producer_parent_wall_seconds=terminal['wall_seconds'], producer_output_bytes=total)

    def packet(self, value, step):
        self.require(isinstance(value, dict) and set(value) == PUBLIC, 'exact public packet fields')
        position, done, hit = value['position'], value['done'], value['hit']
        self.require(isinstance(position, list) and len(position) == 2
                     and all(type(x) is int and 0 <= x < N for x in position), 'public grid coordinates')
        self.require(type(value['step']) is int and value['step'] == step and type(done) is bool
                     and type(hit) is int and (hit == -2 if done else 0 <= hit < 4), 'public step and found sentinel')
        actions = [] if done else [a for a in range(4) if 0 <= position[a // 2] + 2 * (a % 2) - 1 < N]
        self.equal(value['valid_actions'], actions, 'exact geometric eligibility')
        return value

    def posterior(self, belief, packet, kernel):
        np = self.np
        x, y = packet['position']
        if packet['done']:
            result = np.zeros((N, N), dtype=np.float64)
            result[x, y] = 1.
        else:
            result = belief.copy()
            result[x, y] = 0.
            result *= kernel[packet['hit'], N-x:2*N-x, N-y:2*N-y]
            self.require(np.isfinite(result).all() and (result >= 0).all(), 'finite nonnegative public posterior')
            mass = np.sum(result)
            if mass > EPSILON:
                result /= mass
        self.counts['public_posterior_updates'] += 1
        return result

    def witness(self, belief):
        return {'sha256': hashlib.sha256(belief.tobytes(order='C')).hexdigest(),
                'mass': float(self.np.sum(belief))}

    def arrays(self):
        import numpy as np
        self.np = np
        self.require(importlib.metadata.version('numpy') == self.plan['all_distributions']['numpy'], 'qualified NumPy version')
        self.require((self.args.run / 'anchors.npz').stat().st_size <= 2 * 1024**2, 'bounded anchor array container')
        with np.load(self.args.run / 'anchors.npz', allow_pickle=False) as data:
            self.require(set(data.files) == {'beliefs', 'kernel_lambda3', 'kernel_lambda4'}, 'exact saved array names')
            self.beliefs = data['beliefs'].copy()
            self.kernels = {name: data['kernel_' + name].copy() for name in ('lambda3', 'lambda4')}
        self.require(self.beliefs.dtype == np.float64 and self.beliefs.shape == (6, N, N)
                     and np.isfinite(self.beliefs).all() and (self.beliefs >= 0).all(), 'six public float64 beliefs')
        for regime, kernel in self.kernels.items():
            self.require(kernel.dtype == np.float64 and kernel.shape == (4, 107, 107)
                         and np.isfinite(kernel).all() and (kernel >= 0).all() and (kernel <= 1).all()
                         and (kernel[:, N, N] == 0).all(), 'public kernel shape/range/origin')
            source = relative(f'output/otto-symmetry-head-v1/run-01/kernel-{regime}.npz')
            self.require(str(source) in self.descriptors and source.stat().st_size <= 2 * 1024**2,
                         'authenticated bounded original public kernel')
            with np.load(source, allow_pickle=False) as data:
                self.require(set(data.files) == {'likelihood', 'initial_hit_weights'}, 'original kernel schema')
                original = data['likelihood']
                self.require(original.dtype == kernel.dtype and original.shape == kernel.shape
                             and original.tobytes(order='C') == kernel.tobytes(order='C'), 'unchanged original kernel entries')
        self.selected = self.plan['selected_anchors']
        self.equal(self.read(self.args.run / 'anchors.json')['selected'], self.selected, 'saved/frozen anchor identities')
        self.require(len(self.selected) == 6 and len({r['row_index'] for r in self.selected}) == 6, 'six unique frozen anchors')
        for i, row in enumerate(self.selected):
            self.context = {'phase': 'anchor', 'anchor_id': i}
            regime, hit, arm = CELLS[i]
            first = 950001 if regime == 'lambda3' else 960001
            self.require(set(row) == {'anchor_id', 'row_index', 'episode_id', 'stage', 'regime', 'seed',
                                      'initial_hit', 'arm', 'prefix_index', 'public', 'posterior'}
                         and type(row['row_index']) is int and row['row_index'] >= 0
                         and type(row['anchor_id']) is int and row['anchor_id'] == i
                         and type(row['initial_hit']) is int
                         and (row['regime'], row['initial_hit'], row['arm']) == CELLS[i]
                         and row['stage'] == 'dagger' and type(row['seed']) is int and first <= row['seed'] <= first + 11
                         and hit == 1 + (row['seed'] - first) % 3
                         and row['episode_id'] == f'dagger:{regime}:{row["seed"]}:{arm}'
                         and type(row['prefix_index']) is int and 1 <= row['prefix_index'] <= 2187,
                         'fixed nonreset learner-TRAIN cell')
            packet = self.packet(row['public'], row['prefix_index'])
            self.require(not packet['done'] and self.beliefs[i][tuple(packet['position'])] == 0., 'nonterminal anchor support')
            actual = self.witness(self.beliefs[i])
            self.equal(actual, row['posterior'], 'exact frozen anchor posterior witness')
            self.require(abs(actual['mass'] - 1.) <= EPSILON, 'normalized anchor, without repair')
        self.counts['anchors_verified'] = 6

    def lines(self, name, maximum):
        path = self.args.run / name
        hashed, size = hashlib.sha256(), 0
        with path.open('rb') as stream:
            for number, line in enumerate(stream, 1):
                if number % 512 == 1:
                    self.check()
                self.context.update(file=name, line=number)
                self.require(len(line) <= maximum and line.endswith(b'\n'), 'bounded complete journal record')
                hashed.update(line)
                size += len(line)
                row = decode(line)
                self.require(isinstance(row, dict), 'journal object required')
                self.equal(line.decode(), encoded(row).decode(), 'canonical journal serialization')
                yield row
        self.equal({'sha256': hashed.hexdigest(), 'bytes': size}, self.worker['files'][name], 'decoded journal identity')

    def reconstruction(self):
        states, validated = {}, []
        records = iter(self.lines('reconstruction.jsonl', 4096))
        for ordinal, attempt in enumerate(records):
            self.require(set(attempt) == {'operation_id', 'event', 'operation', 'anchor_id', 'step'}
                         and attempt['event'] == 'attempt' and attempt['operation_id'] == ordinal,
                         'reconstruction attempt sequence')
            i, operation, step = attempt['anchor_id'], attempt['operation'], attempt['step']
            self.require(type(i) is int and 0 <= i < 6, 'reconstruction anchor identity')
            row, kernel = self.selected[i], self.kernels[self.selected[i]['regime']]
            returned = next(records)
            self.require(set(returned) == set(attempt) | {'public', 'posterior'}, 'reconstruction return schema')
            self.equal({k: returned[k] for k in attempt}, {**attempt, 'event': 'return'}, 'matching reconstruction return')
            public = self.packet(returned['public'], step)
            if operation == 'public_reset':
                self.require(not validated and i not in states and step == 0 and public['position'] == [26, 26]
                             and public['hit'] == row['initial_hit'] and not public['done'], 'original public center reset')
                prior = self.np.ones((N, N), dtype=self.np.float64) / (N * N - 1)
                prior[26, 26] = 0.
                belief = self.posterior(prior, public, kernel)
            elif operation == 'public_update':
                self.require(not validated and i in states, 'public update before validation')
                prior, previous = states[i]
                self.require(step == previous['step'] + 1 <= row['prefix_index'] and not previous['done'], 'public prefix continuity')
                delta = [a - b for a, b in zip(public['position'], previous['position'], strict=True)]
                self.require(sum(abs(x) for x in delta) == 1, 'one inbounds prefix movement')
                belief = self.posterior(prior, public, kernel)
            elif operation == 'anchor_validation':
                self.require(len(states) == 6 and i == len(validated), 'all reconstructed before ordered validation')
                belief, previous = states[i]
                self.equal(public, previous, 'validation does not reset or update')
                self.require(step == row['prefix_index'] and belief.tobytes() == self.beliefs[i].tobytes(), 'validated saved anchor equality')
                validated.append(i)
            else:
                raise ValueError('undeclared reconstruction operation')
            self.equal(self.witness(belief), returned['posterior'], 'independent prefix posterior witness')
            states[i] = belief, public
            self.calls['reconstruction:' + operation] += 1
        self.require(validated == list(range(6)), 'six completed validation snapshots')
        for i, (_, packet) in states.items():
            self.equal(packet, self.selected[i]['public'], 'exact selected pre-action public state')

    def categorical(self, probability, uniform):
        self.require(type(uniform) is float and math.isfinite(uniform) and 0 <= uniform < 1., 'uniform range/type')
        values = [float(x) for x in probability]
        self.require(values and all(math.isfinite(x) and x >= 0 for x in values)
                     and abs(float(self.np.sum(probability, dtype=self.np.float64)) - 1.) <= EPSILON,
                     'qualified categorical mass acceptance')
        running, cumulative = 0., []
        for value in values:  # Independent scalar left-to-right float64 CDF.
            running += value
            cumulative.append(running)
        self.require(math.isfinite(running) and running > 0., 'positive CDF mass')
        selected = next((i for i, value in enumerate(cumulative) if uniform < value / running), None)
        self.require(selected is not None and values[selected] > 0., 'right-boundary positive support choice')
        self.counts['categorical_choices_recomputed'] += 1
        return {'uniform': uniform, 'cdf_mass': running, 'selected_index': selected}

    def generator(self, anchor, replica, channel):
        return self.np.random.Generator(self.np.random.PCG64(
            self.np.random.SeedSequence([0x4F54544F, SEED, anchor, replica, channel])))

    def sampler(self):
        stream = iter(self.lines('events.jsonl', 512))
        derived = []
        for i, row in enumerate(self.selected):
            self.context = {'phase': 'sampler', 'anchor_id': i}
            base = {'version': 'otto-teacher-rollouts-v1', 'seed': SEED, 'anchor_id': i}
            ordinal, completed = 0, []

            def event(kind, fields, base=base):
                value = next(stream)
                self.equal(value, {**base, 'event': kind, **fields}, 'exact scheduled sampler event')
                self.events[kind] += 1
                self.counts['sampler_events'] += 1

            def call(name, context, fields=(), base=base):
                nonlocal ordinal
                ordinal += 1
                identity = {**context, 'operation_id': ordinal, 'operation': name}
                event('attempt', identity)
                value = next(stream)
                self.require(set(value) == set(base) | {'event'} | set(identity) | set(fields), 'sampler return schema')
                self.equal({k: v for k, v in value.items() if k not in fields},
                           {**base, 'event': 'return', **identity}, 'matching sampler return identity')
                self.events['return'] += 1
                self.counts['sampler_events'] += 1
                self.calls['sampler:' + name] += 1
                return {k: value[k] for k in fields}

            self.equal(call('anchor_snapshot', {}, ('public',)), {'public': row['public']}, 'sampler anchor snapshot')
            for replica in range(16):
                common = {'replicate_id': replica}
                call('source_generator', common)
                uniform = float(self.generator(i, replica, 0).random())
                draw = self.categorical(self.beliefs[i].reshape(-1, order='C'), uniform)
                source = list(divmod(draw['selected_index'], N))
                self.equal(call('source_draw', common, (*draw, 'source')), {**draw, 'source': source}, 'seeded anchor source draw')
                self.counts['source_draws_verified'] += 1
                for first in row['public']['valid_actions']:
                    context = {**common, 'first_action': first}
                    self.context.update(replicate_id=replica, first_action=first)
                    self.equal(call('teacher_snapshot', context, ('public',)), {'public': row['public']}, 'fresh action snapshot')
                    call('hit_generator', context)
                    rng, current, belief = self.generator(i, replica, 1), row['public'], self.beliefs[i].copy()
                    for local in range(1, HORIZON + 1):
                        step_context = {**context, 'local_step': local, 'from_step': current['step']}
                        if local == 1:
                            action = first
                            event('forced_action', {**step_context, 'action': action})
                        else:
                            choice = call('teacher_choose', step_context, ('action', 'scores'))
                            scores = choice['scores']
                            allowed = current['valid_actions']
                            self.require(isinstance(scores, list) and len(scores) == 4 and all(
                                type(scores[a]) is float and math.isfinite(scores[a]) if a in allowed else scores[a] is None
                                for a in range(4)), 'four saved eligible/blocked teacher scores')
                            minimum = min(scores[a] for a in allowed)
                            action = next(a for a in allowed if abs(scores[a] - minimum) < EPSILON)
                            self.equal(choice['action'], action, 'first near-tie argmin of saved scores')
                            self.counts['saved_teacher_choices_checked'] += 1
                        self.require(action in current['valid_actions'], 'inbounds continuation action')
                        position = list(current['position'])
                        position[action // 2] += 2 * (action % 2) - 1
                        found = position == source
                        self.equal(call('movement', step_context, ('position', 'found')),
                                   {'position': position, 'found': found}, 'source-relative movement/found event')
                        hit = -2
                        if not found:
                            probability = self.kernels[row['regime']][:, N+source[0]-position[0], N+source[1]-position[1]]
                            uniform = float(rng.random())
                            draw = self.categorical(probability, uniform)
                            expected = {**draw, 'probabilities': probability.tolist(), 'draw_index': local - 1}
                            self.equal(call('hit_draw', step_context, tuple(expected)), expected, 'seeded common-uniform source-conditioned hit')
                            hit = draw['selected_index']
                            self.counts['hit_draws_verified'] += 1
                        packet = {'position': position, 'hit': hit, 'done': found, 'step': current['step'] + 1,
                                  'valid_actions': [] if found else [a for a in range(4)
                                      if 0 <= position[a // 2] + 2 * (a % 2) - 1 < N]}
                        self.equal(call('teacher_update', step_context, ('public',)), {'public': packet}, 'complete public update packet')
                        belief = self.posterior(belief, packet, self.kernels[row['regime']])
                        current = packet
                        self.counts['continuation_moves_verified'] += 1
                        if found or local == HORIZON:
                            record = {'replicate_id': replica, 'first_action': first, 'steps': local, 'found': found}
                            event('record', record)
                            completed.append(record)
                            derived.append({'anchor_id': i, **record, 'final_public': packet,
                                            'independent_final_posterior': self.witness(belief)})
                            break
            event('panel_complete', {'record_count': len(completed), 'operation_count': ordinal})
            panel = self.read(self.args.run / f'panel-{i}.json')
            self.require(panel['anchor_id'] == i and panel['episode_id'] == row['episode_id'], 'panel identity')
            self.equal(panel['records'], completed, 'complete saved sixteen-replicate/action panel')
            self.reduce(panel, row['public']['valid_actions'])
        self.require(next(stream, None) is None, 'no extra sampler events or panels')
        self.derived = derived

    def reduce(self, panel, actions):
        records = {(r['replicate_id'], r['first_action']): r for r in panel['records']}
        self.require(len(records) == 16 * len(actions), 'full action/replicate coverage')
        totals = {a: sum(records[r, a]['steps'] for r in range(16)) for a in actions}
        grand = sum(totals.values())
        expected_actions = []
        for action in actions:
            found = sum(records[r, action]['found'] for r in range(16))
            expected_actions.append({'action': action, 'mean_cost': totals[action] / 16,
                'centered_mean_cost': (len(actions)*totals[action]-grand)/(16*len(actions)),
                'success_fraction': found/16, 'censored_fraction': (16-found)/16})
        reduced = panel['reduction']
        self.equal(reduced['actions'], expected_actions, 'independent capped action costs and censoring')
        pairs, details = [], []
        count = len(actions) * (len(actions) - 1) // 2
        radius = 2 * (HORIZON - 1) * math.sqrt((math.log(2 * count) - math.log(.05)) / 32)
        for first, second in itertools.combinations(actions, 2):
            differences = [records[r, first]['steps'] - records[r, second]['steps'] for r in range(16)]
            total, mean = sum(differences), sum(differences) / 16
            # Integer sufficient statistics independently avoid cancellation.
            se = math.sqrt((16 * sum(x*x for x in differences) - total*total) / (16*16*15))
            lower, upper = max(1-HORIZON, mean-radius), min(HORIZON-1, mean+radius)
            pairs.append({'first_action': first, 'second_action': second, 'mean_difference': mean,
                          'paired_standard_error': se, 'hoeffding_radius': radius,
                          'hoeffding_interval': [lower, upper],
                          'resolved_direction': 'first_lower' if upper < 0 else 'second_lower' if lower > 0 else 'unresolved'})
            details.append({'first_action': first, 'second_action': second, 'differences': differences,
                            'exact_replicate_ties': sum(x == 0 for x in differences),
                            'all_censored_replicate_ties': sum(not records[r, first]['found'] and not records[r, second]['found'] for r in range(16)),
                            'equal_means': total == 0})
        self.equal(reduced['pairs'], pairs, 'independent paired means/SE/conditional Hoeffding summaries')
        self.equal(panel['paired_details'], details, 'complete paired differences and censored ties')
        for field, value in {'version': 'otto-teacher-cost-summary-v1', 'horizon': HORIZON,
                             'replicate_count': 16, 'action_count': len(actions), 'pair_count': count,
                             'familywise_alpha': .05}.items():
            self.equal(reduced[field], value, f'reduction declaration: {field}')
        self.counts['panels_reduced_independently'] += 1

    def totals(self):
        records = self.derived
        c, s, f = len(records), sum(r['steps'] for r in records), sum(r['found'] for r in records)
        expected = {'anchor_snapshot': 6, 'source_generator': 96, 'source_draw': 96,
                    'teacher_snapshot': c, 'hit_generator': c, 'teacher_choose': s-c,
                    'movement': s, 'hit_draw': s-f, 'teacher_update': s}
        calls = {'sampler:' + k: {'attempted': v, 'returned': v} for k, v in expected.items()}
        calls.update({'reconstruction:' + k: {'attempted': v, 'returned': v} for k, v in {
            'public_reset': 6, 'public_update': sum(r['prefix_index'] for r in self.selected), 'anchor_validation': 6}.items()})
        self.equal({k: {'attempted': n, 'returned': n} for k, n in self.calls.items()}, calls, 'complete independently counted operations')
        self.equal(self.worker['calls'], calls, 'worker operation accounting')
        self.equal(dict(self.events), {'attempt': sum(expected.values()), 'return': sum(expected.values()),
                                      'forced_action': c, 'record': c, 'panel_complete': 6}, 'sampler event coverage')
        self.require(self.counts['sampler_events'] == self.worker['sampler_events'] == 402 + 4*c + 8*s - 2*f,
                     'exact sampler event formula')
        self.require(c == 16 * sum(len(r['public']['valid_actions']) for r in self.selected)
                     and c <= 384 and s <= c * HORIZON, 'fixed full-horizon allocation')
        summary = self.read(self.args.run / 'summary.json')
        for key, value in {'version': 'otto-teacher-label-pilot-v1', 'technical_completion': True,
                           'anchors': 6, 'continuations': c, 'moves': s, 'found': f,
                           'calls': calls, 'sampler_events': self.worker['sampler_events']}.items():
            self.equal(summary[key], value, f'complete summary: {key}')
        panels = [self.read(self.args.run / f'panel-{i}.json') for i in range(6)]
        self.equal(summary['panels'], panels, 'summary/panel byte-value agreement')
        timings = self.worker['timings']
        self.equal(summary['timings'], {k: v for k, v in timings.items() if k != 'closing_authentication_seconds'}, 'summary/receipt timing join')
        for name, value in timings.items():
            self.finite(value, f'nonnegative timing: {name}')
        for panel in panels:
            for name in ('sampler_seconds_including_io', 'reduction_seconds'):
                self.finite(panel[name], f'nonnegative panel timing: {name}')
        disjoint = sum(timings[k] for k in ('authentication_seconds', 'extraction_reconstruction_validation_seconds',
                                          'closing_authentication_seconds'))
        disjoint += sum(p['sampler_seconds_including_io'] + p['reduction_seconds'] for p in panels)
        self.require(disjoint <= self.worker['wall_seconds'] + 1e-9, 'disjoint recorded intervals fit complete worker elapsed')
        nested = sum(timings[k] for k in ('public_reset_instrumented_seconds', 'public_update_instrumented_seconds',
                                        'anchor_validation_instrumented_seconds'))
        self.require(nested <= timings['extraction_reconstruction_validation_seconds'] + 1e-9, 'reconstruction nested intervals fit enclosing time')
        self.finite(self.worker['artifact_serialization_io_seconds_nested'], 'nested I/O time')
        self.require(self.worker['artifact_serialization_io_seconds_nested'] <= self.worker['wall_seconds'], 'I/O remains inside complete cost')
        self.require(self.worker['files']['events.jsonl']['bytes'] <= self.counts['sampler_events'] * 512
                     and self.worker['output_bytes_before_receipt'] - self.worker['files']['events.jsonl']['bytes'] <= 256 * 1024**2,
                     'actual journal and non-event serialization bounds')
        self.result = {'version': VERSION, 'agreement': True, 'anchors': 6, 'continuations': c,
                       'moves': s, 'found': f, 'counts': dict(self.counts), 'calls': calls,
                       'count_scope': 'Through scientific replay and summary checks; final authentication follows.',
                       'panels': panels, 'derived_continuation_end_states': records,
                       'limitations': LIMITATIONS, 'technical_completion_only': True}

    def execute(self):
        regular(self.out).mkdir(parents=True, exist_ok=False)
        try:
            self.require(digest(relative(CLOCK))['sha256'] == CLOCK_PIN, 'qualified audit clock before import')
            spec = importlib.util.spec_from_file_location('_teacher_saved_audit_clock', relative(CLOCK))
            clock_module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = clock_module
            spec.loader.exec_module(clock_module)
            self.clock = clock_module.SuspendClock()
            self.start = self.clock.now_ns()
            self.require(self.clock.backend in ('mach_continuous_time', 'CLOCK_BOOTTIME'), 'native suspend-inclusive audit clock')
            self.self_source = digest(Path(__file__).resolve(), self.check)
            write(self.out / 'started.json', {'version': VERSION, 'request': {k: str(v) for k, v in vars(self.args).items()},
                'command': [sys.executable, *sys.argv], 'source': self.self_source, 'clock_sha256': CLOCK_PIN,
                'started_ns': self.start, 'clock_backend': self.clock.backend,
                'budget': {k: getattr(self.args, k) for k in ('max_seconds', 'max_rss_bytes', 'max_output_bytes')}})
            self.authenticate()
            self.arrays()
            self.reconstruction()
            self.sampler()
            self.totals()
            self.context = {'phase': 'closing authentication'}
            for name, desc in self.descriptors.items():
                self.equal(digest(Path(name), self.check), desc, f'unchanged input at closure: {name}')
            self.equal(digest(Path(__file__).resolve(), self.check), self.self_source, 'unchanged auditor source')
            self.require(digest(relative(CLOCK), self.check)['sha256'] == CLOCK_PIN, 'unchanged audit clock source')
            write(self.out / 'audit.json', self.result)
            self.check()
            files = {p.name: digest(p, self.check) for p in self.out.iterdir()}
            finished = self.clock.now_ns()
            self.receipt.update(status='completed', agreement=True, counts=dict(self.counts),
                started_ns=self.start, finished_ns=finished, wall_seconds=(finished-self.start)/1e9,
                clock_backend=self.clock.backend, source=self.self_source, clock_sha256=CLOCK_PIN,
                count_scope='Through closing authentication and audit payload hashing; post-publication resource checks excluded.',
                timing_scope='Worker through audit payload hashing; final receipt publication and process exit need the original parent terminal.',
                files=files)
            write(self.out / 'receipt.json', self.receipt)
            self.check()
            print(json.dumps({'status': 'completed', 'receipt_sha256': digest(self.out / 'receipt.json')['sha256']}), flush=True)
            return 0
        except BaseException as error:
            self.receipt.update(status='failed', agreement=False, counts=dict(self.counts),
                failures=[{'error': repr(error), 'context': self.context, 'traceback': traceback.format_exc()}])
            try:
                self.receipt['wall_seconds'] = (self.clock.now_ns()-self.start)/1e9 if self.clock and self.start is not None else None
            except BaseException as secondary:  # noqa: BLE001 - preserve the primary failure
                self.receipt['clock_error'] = repr(secondary)
            try:
                path = self.out / 'receipt.json'
                if path.exists():
                    path.rename(self.out / 'receipt.invalid.json')
                self.receipt['files'] = {p.name: digest(p) for p in self.out.iterdir() if p.is_file()}
                write(path, self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - preserve failure even if publication fails
                error.add_note(f'Failure receipt publication error: {secondary!r}')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'plan', 'terminal', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('receipt-sha256', 'plan-sha256', 'terminal-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    parser.add_argument('--max-rss-bytes', type=int, default=4 * 1024**3)
    parser.add_argument('--max-output-bytes', type=int, default=64 * 1024**2)
    args = parser.parse_args()
    for name in ('run', 'plan', 'terminal', 'output'):
        regular(getattr(args, name))
    if min(args.max_seconds, args.max_rss_bytes, args.max_output_bytes) <= 0:
        raise ValueError('positive independently frozen saved-audit budget required')

    def interrupted(signum, _frame):
        raise InterruptedError(f'saved audit received signal {signum}')

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return Audit(args).execute()
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
