"""Autonomous evaluation of the qualified released TF policy and two analytic controls.

No training, NumPy policy substitution, reference-score tuning, or old gate revision.
All numerical/native imports follow external plan, lineage and supervision checks.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import resource
import sys
import time
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'otto-released-reference-v1'
ARMS = ('released_tf', 'analytic_all4', 'analytic_inbounds')
CONTROLS = ARMS[1:]
CASES, HORIZON, EPISODES = 96, 2188, 576
COHORTS = {name: {'first_seed': seed, 'lambda_over_dx': lam}
           for name, seed, lam in (('base', 850001, 3.), ('shift', 860001, 4.))}
LIMITS = {'native_seconds': 1800, 'rss_bytes': 4 * 1024**3, 'output_bytes': 512 * 1024**2,
          'native_steps': EPISODES * HORIZON, 'native_resets': EPISODES,
          'tensorflow_value_calls': 2 * CASES * HORIZON}
CONFIGURATION = {'arms': list(ARMS), 'cohorts': COHORTS, 'cases_per_cohort': CASES,
                 'blocks': 8, 'cases_per_hit_per_block': 4, 'horizon': HORIZON,
                 'episodes': EPISODES, 'Ngrid': 53, 'Nhits': 4, 'Ndim': 2, 'R_dt': 2.,
                 'norm_Poisson': 'Euclidean', 'rotation': 'global_case_index modulo 3',
                 'symmetry_average': True, 'public_native_belief_parity': 'exact',
                 'new_native_qualification_steps': 0, 'numpy_port_admitted': False,
                 'learned_pilot_admission': False}
QUALIFIER = 'scripts/qualify_otto_released_native.py'
QUALIFIER_PIN = '96c68e94498c6b83eb2983c1183c444c5d1080eee06ff961562aeb56a845c65e'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
AUDITOR = 'output/otto-released-native-v1/audit-01/audit.py'
AUDITOR_PIN = '7b405bccfcc6d6c4bf7412729801e37e8a2a80b027b0ad3d6d367ebb5a8559d1'
CONTROL = 'src/openjev/research/otto_reference_control.py'
PREFLIGHT = 'scripts/qualify_otto_reference_control.py'
PROTOCOL = 'research/otto-released-reference-protocol.md'
REQUIRED = {QUALIFIER, CLOCK, AUDITOR, CONTROL, PREFLIGHT, PROTOCOL, 'scripts/study_otto_released_reference.py',
            'tests/test_otto_released_reference.py', 'tests/test_otto_reference_control.py'}
ROLES = {'qualification_plan', 'qualification_receipt', 'qualification_terminal', 'audit_receipt',
         'analytic_preflight_receipt'}
TIMES = ('actor_initialization_seconds', 'choose_seconds', 'update_seconds', 'model_setup_allocation_seconds')
METRICS = ('capped_time', 'found', 'stuck_steps', 'blocked_steps', *TIMES, 'controller_seconds',
           'model_forward_seconds', 'environment_initialization_seconds', 'environment_seconds',
           'episode_seconds', 'state_array_bytes')
CHANNELS = ('native_reset', 'native_step', 'actor_construction', 'actor_choose', 'actor_update',
            'tensorflow_construction', 'tensorflow_build', 'tensorflow_load', 'tensorflow_value')
SCOPE = ('Fresh autonomous reference comparison with exact public-only beliefs and paired sampled sources. '
         'Released TF and analytic_all4 can stay at boundaries; analytic_inbounds restricts boundary actions. '
         'No model training, architectural novelty, prior gate revision, or NumPy policy admission.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


def descriptor(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def regular(name):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'relative contained source/input')
    path = ROOT / relative
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'regular nonsymlink input')
    return path


def payloads(receipt_path):
    record = read(receipt_path)
    directory = receipt_path.parent
    require(record['status'] == 'completed', 'completed upstream artifact')
    require({p.name for p in directory.iterdir()} == set(record['files']) | {receipt_path.name}, 'exact upstream payload closure')
    for name, pin in record['files'].items():
        require(Path(name).name == name and (directory / name).is_file() and not (directory / name).is_symlink()
                and descriptor(directory / name) == pin, f'upstream payload: {name}')
    return record


def authenticate(args):
    require(args.plan.is_absolute() and not args.plan.is_symlink() and sha(args.plan) == args.plan_sha256, 'external study plan pin')
    plan = read(args.plan)
    require(plan['version'] == VERSION and plan['status'] == 'frozen_before_native_run'
            and plan['configuration'] == CONFIGURATION and plan['limits'] == LIMITS, 'fixed reference study')
    require(REQUIRED <= plan['sources'].keys() and plan['sources'][QUALIFIER] == QUALIFIER_PIN
            and plan['sources'][CLOCK] == CLOCK_PIN and plan['sources'][AUDITOR] == AUDITOR_PIN, 'required source closure')
    for name, pin in plan['sources'].items():
        require(sha(regular(name)) == pin, f'current source identity: {name}')
    require(set(plan['inputs']) == ROLES, 'exact qualification input roles')
    paths = {}
    for role, record in plan['inputs'].items():
        paths[role] = regular(record['path'])
        require(descriptor(paths[role]) == {'sha256': record['sha256'], 'bytes': record['bytes']}, f'input pin: {role}')
    prior_module = load(ROOT / QUALIFIER, '_released_reference_auth_only')
    prior, inherited = prior_module.authenticate(types.SimpleNamespace(
        plan=paths['qualification_plan'], plan_sha256=plan['inputs']['qualification_plan']['sha256']))
    require(plan['runtime'] == prior['runtime'] and all(plan['sources'].get(n) == h for n, h in prior['sources'].items()),
            'unchanged complete native runtime and inherited source closure')
    receipt = payloads(paths['qualification_receipt'])
    require(receipt['version'] == prior['version'] and receipt['qualified'] is True
            and receipt['plan_sha256'] == sha(paths['qualification_plan']) and receipt['sources'] == prior['sources']
            and receipt['inputs'] == prior['inputs'] and receipt['numpy_port_admitted'] is False
            and receipt['prior_numpy_action_failure_preserved'] is True, 'successful native qualification identity')
    terminal = read(paths['qualification_terminal'])
    started = read(paths['qualification_receipt'].parent / 'started.json')
    launch_path = Path(started['request']['supervision'])
    require(sha(launch_path) == receipt['supervision_sha256'] and read(launch_path) == started['launch'], 'native launch identity')
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
            and terminal['group_absent'] is True and terminal['cleanup']['errors'] == []
            and terminal['started_ns'] <= receipt['started_ns'] <= receipt['finished_ns'] <= terminal['finished_ns'] < terminal['deadline_ns'],
            'successful native qualification supervisor')
    for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend'):
        require(terminal[key] == started['launch'][key], 'qualification parent identity')
    audit = payloads(paths['audit_receipt'])
    require(audit['version'] == 'otto-released-native-saved-audit-v1' and audit['agreement'] is True
            and audit['plan_sha256'] == sha(paths['qualification_plan'])
            and audit['worker_sha256'] == sha(paths['qualification_receipt'])
            and audit['terminal_sha256'] == sha(paths['qualification_terminal'])
            and audit['source']['sha256'] == AUDITOR_PIN and audit['model_calls'] == audit['simulator_calls'] == 0,
            'successful saved-only native audit joins')
    preflight = payloads(paths['analytic_preflight_receipt'])
    require(preflight['version'] == 'otto-reference-control-qualification-v1'
            and preflight['qualified'] is True and preflight['agreement'] is True
            and preflight['model_calls'] == preflight['simulator_calls'] == preflight['native_calls'] == 0
            and preflight['original_heuristic_calls'] == preflight['new_analytic_calls'] == 16
            and preflight['self_source']['sha256'] == plan['sources'][PREFLIGHT]
            and all(plan['sources'].get(n) == h for n, h in preflight['sources'].items()), 'analytic original-policy preflight')
    for key, expected in {'plan': str(paths['qualification_plan']), 'plan_sha256': sha(paths['qualification_plan']),
                          'run': str(paths['qualification_receipt'].parent), 'receipt_sha256': sha(paths['qualification_receipt']),
                          'terminal': str(paths['qualification_terminal']), 'terminal_sha256': sha(paths['qualification_terminal'])}.items():
        require(preflight['request'][key] == expected, 'analytic preflight joins same successful native evidence')
    return plan, inherited


def case_order():
    for regime_index, (name, config) in enumerate(COHORTS.items()):
        for i in range(CASES):
            offset = (regime_index * CASES + i) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                yield name, config['first_seed'] + i, i // 12, 1 + (i % 12) // 4, arm


def criteria(means, blocks):
    competence, teacher, pareto = [], [], []
    released = means['released_tf']
    competence.append({'name': 'released_success_at_least_95pct', 'passes': released['found'] >= .95})
    for control in CONTROLS:
        baseline = means[control]
        competence.append({'name': f'moves_at_most_105pct_{control}',
                           'passes': released['capped_time'] <= 1.05 * baseline['capped_time']})
        positive = sum(b['means'][control]['capped_time'] > b['means']['released_tf']['capped_time'] for b in blocks)
        teacher.extend([{'name': f'no_success_loss_vs_{control}', 'passes': released['found'] >= baseline['found']},
                        {'name': f'moves_at_most_95pct_{control}', 'passes': released['capped_time'] <= .95 * baseline['capped_time']},
                        {'name': f'positive_blocks_vs_{control}', 'value': positive, 'threshold': 6, 'passes': positive >= 6}])
        pareto.extend([{'name': f'no_success_loss_vs_{control}', 'passes': released['found'] >= baseline['found']},
                       {'name': f'no_more_moves_vs_{control}', 'passes': released['capped_time'] <= baseline['capped_time']},
                       {'name': f'no_more_controller_cost_vs_{control}', 'passes': released['controller_seconds'] <= baseline['controller_seconds']},
                       {'name': f'strict_move_or_cost_gain_vs_{control}',
                        'passes': released['capped_time'] < baseline['capped_time'] or released['controller_seconds'] < baseline['controller_seconds']}])
    return competence, teacher, pareto


def summarize(rows, weights):
    expected = list(case_order())
    require(len(rows) == EPISODES and [(r['cohort'], r['seed'], r['block'], r['initial_hit'], r['arm']) for r in rows] == expected,
            'complete exact 576-episode rotated cohort')
    require(set(weights) == set(COHORTS), 'both regime mixtures')
    for row in rows:
        require(type(row['steps']) is int and 1 <= row['steps'] <= HORIZON and row['capped_time'] == row['steps']
                and type(row['found']) is bool and (row['found'] or row['steps'] == HORIZON), 'found/censored horizon semantics')
        require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0 for k in METRICS if k != 'found'), 'finite nonnegative metrics')
        require(all(type(row[k]) is int and 0 <= row[k] <= row['steps'] for k in ('stuck_steps', 'blocked_steps')), 'integer movement counts')
        require(row['choose_calls'] == row['update_calls'] == row['steps']
                and row['model_forward_calls'] == (row['steps'] if row['arm'] == 'released_tf' else 0), 'actual operation count joins')
        require(abs(row['controller_seconds'] - math.fsum(row[k] for k in TIMES)) <= 1e-9, 'complete controller cost sum')
        require(type(row['controller_instrumented_seconds']) in (int, float)
                and type(row['controller_excluded_io_seconds']) in (int, float)
                and math.isfinite(row['controller_instrumented_seconds']) and math.isfinite(row['controller_excluded_io_seconds'])
                and row['controller_excluded_io_seconds'] >= 0
                and abs(row['controller_instrumented_seconds'] - row['controller_seconds'] - row['controller_excluded_io_seconds']) <= 1e-9,
                'raw controller time and excluded I/O reconstruction')
    result = {}
    for name in COHORTS:
        mixture = weights[name]
        require(set(mixture) == {1, 2, 3} and all(math.isfinite(w) and 0 < w < 1 for w in mixture.values())
                and abs(math.fsum(mixture.values()) - 1) <= 1e-12, 'fixed positive initial-hit mixture')
        selected = [r for r in rows if r['cohort'] == name]

        def group(subset, mixture=mixture):
            return {arm: {metric: math.fsum(mixture[h] * math.fsum(float(r[metric]) for r in subset if r['arm'] == arm and r['initial_hit'] == h)
                                          / sum(r['arm'] == arm and r['initial_hit'] == h for r in subset) for h in (1, 2, 3))
                          for metric in METRICS} for arm in ARMS}

        means = group(selected)
        strata = {h: {a: {m: math.fsum(float(r[m]) for r in selected if r['initial_hit'] == h and r['arm'] == a) / 32
                         for m in METRICS} for a in ARMS} for h in (1, 2, 3)}
        blocks = [{'block': b, 'means': group([r for r in selected if r['block'] == b])} for b in range(8)]
        gates = criteria(means, blocks)
        counts = {a: {'episodes': 96, 'found': sum(r['found'] for r in selected if r['arm'] == a),
                      'censored': sum(not r['found'] for r in selected if r['arm'] == a),
                      'total_steps': sum(r['steps'] for r in selected if r['arm'] == a)} for a in ARMS}
        result[name] = {'initial_hit_weights': mixture, 'means': means, 'strata': strata, 'blocks': blocks,
                        'unweighted_counts': counts,
                        'competence_checks': gates[0], 'stronger_teacher_checks': gates[1], 'utility_compute_checks': gates[2]}
    return {'version': VERSION, 'scope': SCOPE, 'episodes': EPISODES, 'cohorts': result,
            'competent_reference': all(c['passes'] for r in result.values() for c in r['competence_checks']),
            'stronger_value_teacher': all(c['passes'] for r in result.values() for c in r['stronger_teacher_checks']),
            'utility_compute_advantage': all(c['passes'] for r in result.values() for c in r['utility_compute_checks']),
            'learned_pilot_admission': False, 'inherited_gate_revised': False, 'numpy_port_admitted': False}


class Ledger:
    """Nested calls retain durable attempts and uncertain pending work on failure."""
    def __init__(self, out, check):
        self.out, self.check = out, check
        self.context = {'phase': 'authentication'}
        self.calls = {name: {'attempted': 0, 'returned': 0, 'seconds': 0., 'instrumented_seconds': 0.,
                             'excluded_io_seconds': 0.} for name in CHANNELS}
        self.pending, self.sequence = [], 0
        self.last_seconds, self.last_instrumented, self.last_io = {}, {}, {}
        self.io_seconds = 0.

    def emit(self, name, value):
        tick = time.perf_counter()
        try:
            append(self.out / name, value)
        finally:
            self.io_seconds += time.perf_counter() - tick

    def snapshot(self):
        return {'context': self.context, 'calls': self.calls, 'pending': self.pending,
                'sequence': self.sequence, 'artifact_io_seconds': self.io_seconds}

    def call(self, channel, operation):
        self.check()
        require(channel in self.calls, 'declared call channel')
        count = self.calls[channel]
        count['attempted'] += 1
        self.sequence += 1
        record = {'call_id': self.sequence, 'channel': channel, 'ordinal': count['attempted'],
                  'parent_call_id': self.pending[-1]['call_id'] if self.pending else None, 'context': dict(self.context)}
        self.pending.append(record)
        self.emit('work.jsonl', {'event': 'attempt', **record})
        io_before = self.io_seconds
        tick = time.perf_counter()
        result = operation()
        raw = time.perf_counter() - tick
        excluded = self.io_seconds - io_before
        elapsed = raw - excluded
        require(elapsed >= 0., 'nonnegative compute time after measured nested artifact I/O')
        count['returned'] += 1
        count['seconds'] += elapsed
        count['instrumented_seconds'] += raw
        count['excluded_io_seconds'] += excluded
        require(self.pending[-1] is record, 'nested operation stack')
        self.pending.pop()
        self.last_seconds[channel] = elapsed
        self.last_instrumented[channel], self.last_io[channel] = raw, excluded
        self.emit('work.jsonl', {'event': 'return', **record, 'seconds': elapsed,
                               'instrumented_seconds': raw, 'excluded_io_seconds': excluded})
        self.check()
        return result


class ForwardRecorder:
    """Records real original-TF calls and compact branch outputs, never substitutes values."""
    def __init__(self, model, policy_code, ledger, np):
        self.model, self.policy_code, self.ledger, self.np = model, policy_code, ledger, np

    def __call__(self, inputs, *, sym_avg):
        frame = inspect.currentframe().f_back
        require(frame.f_code is self.policy_code and sym_avg is True, 'only exact original policy with symmetry averaging')
        masses = self.np.asarray(frame.f_locals['probs']).copy()
        del frame
        require(tuple(inputs.shape) == (16, 105, 105) and masses.shape == (4, 4)
                and masses.dtype == self.np.float32 and self.np.isfinite(masses).all(), 'fixed policy forward geometry')
        result = self.ledger.call('tensorflow_value', lambda: self.model(inputs, training=False, sym_avg=True))
        values = result.numpy()
        require(values.shape == (16, 1) and values.dtype == self.np.float32 and self.np.isfinite(values).all(), 'actual finite original model output')
        self.ledger.emit('forwards.jsonl', {'context': dict(self.ledger.context),
               'ordinal': self.ledger.calls['tensorflow_value']['returned'], 'symmetry_average': True,
               'input_shape': [16, 105, 105], 'branch_masses': masses.tolist(), 'values': values[:, 0].tolist(),
               'seconds': self.ledger.last_seconds['tensorflow_value']})
        return result


def packet(observation):
    return {'position': tuple(observation.position), 'hit': int(observation.hit), 'done': bool(observation.done),
            'step': int(observation.step), 'valid_actions': tuple(observation.valid_actions)}


def belief_witness(actor, env, public, np):
    require(actor.public == public, 'exact actor public metadata')
    probability = actor.belief
    require(probability.dtype == env.p_source.dtype == np.float64 and probability.shape == env.p_source.shape == (53, 53)
            and np.isfinite(probability).all() and (probability >= 0).all()
            and probability.tobytes() == env.p_source.tobytes(), 'exact native/public posterior after every update')
    if public['done']:
        require(probability[public['position']] == 1. and np.count_nonzero(probability) == 1, 'terminal found point mass')
    return {'mass': float(probability.sum()), 'sha256': hashlib.sha256(probability.tobytes()).hexdigest(), 'exact': True}


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = self.launch = self.plan = self.paths = None
        self.ledger = Ledger(self.out, self.check)
        self.receipt = {'version': VERSION, 'status': 'started', 'scope': SCOPE, 'limits': LIMITS,
                        'completed_episodes': 0, 'training_updates': 0, 'external_model_calls': 0,
                        'numpy_port_admitted': False, 'prior_numpy_action_failure_preserved': True}

    def check(self):
        require(self.clock.now_ns() < self.launch['deadline_ns'], 'reference study shared suspend-inclusive deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= LIMITS['rss_bytes'], 'reference RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'reference output cap')
        for channel, cap in (('native_step', LIMITS['native_steps']), ('native_reset', EPISODES),
                             ('tensorflow_value', LIMITS['tensorflow_value_calls'])):
            require(self.ledger.calls[channel]['attempted'] <= cap, f'{channel} complete-work cap')

    def bind(self):
        require(sha(ROOT / CLOCK) == CLOCK_PIN, 'registered clock source')
        self.clock = load(ROOT / CLOCK, '_reference_clock').SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, 'missing actual supervisor launch')
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        command = list(self.launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and self.launch['pid'] == os.getpid()
                and self.launch['pgid'] == os.getpgrp() and self.launch['parent_pid'] == os.getppid()
                and self.launch['cap_seconds'] == LIMITS['native_seconds'] and self.launch['clock_backend'] == self.clock.backend
                and self.launch['started_ns'] <= self.start < self.launch['deadline_ns']
                and self.launch['deadline_ns'] == self.launch['started_ns'] + LIMITS['native_seconds'] * 10**9
                and Path(self.launch['cwd']).resolve() == ROOT == Path.cwd().resolve(), 'actual bounded reference process')
        self.plan, self.paths = authenticate(self.args)
        require(self.launch['watchdog_sha256'] == self.plan['sources']['scripts/supervise_dialogue_observation_v2.py']
                and self.launch['clock_source_sha256'] == CLOCK_PIN, 'actual supervisor source pins')
        self.receipt.update(plan_sha256=self.args.plan_sha256, supervision_sha256=sha(self.args.supervision),
                            sources=self.plan['sources'], inputs=self.plan['inputs'])
        write(self.out / 'started.json', {'request': {k: str(v) for k, v in vars(self.args).items()},
                                        'clock_backend': self.clock.backend, 'started_ns': self.start, 'launch': self.launch})
        self.check()

    def setup(self):
        require(not any(n in sys.modules for n in ('numpy', 'scipy', 'tensorflow', 'tf_keras', 'h5py')), 'clean numerical import boundary')
        common_tick = time.perf_counter()
        prior = load(ROOT / QUALIFIER, '_reference_qualified_helpers')
        helper = load(ROOT / prior.ENGINE, '_reference_original_helpers')
        os.environ.update(helper.ENVIRONMENT)
        import numpy as np

        common_import_seconds = time.perf_counter() - common_tick
        tick, io_before = time.perf_counter(), self.ledger.io_seconds
        self.ledger.context = {'phase': 'model_setup'}
        import tensorflow as tf

        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        keras = {'floatx': tf.keras.backend.floatx(), 'image_data_format': tf.keras.backend.image_data_format(),
                 'mixed_precision_policy': tf.keras.mixed_precision.global_policy().name}
        require(not tf.config.get_visible_devices('GPU') and tf.keras.Model.__module__.startswith('tf_keras.')
                and keras == {'floatx': 'float32', 'image_data_format': 'channels_last', 'mixed_precision_policy': 'float32'},
                'unchanged original float32 legacy-Keras CPU policy')
        package_name = '_reference_upstream'
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT / prior.UPSTREAM)]
        sys.modules[package_name] = package
        load(ROOT / prior.UPSTREAM / 'policy.py', f'{package_name}.policy')
        policy = load(ROOT / prior.UPSTREAM / 'rlpolicy.py', f'{package_name}.rlpolicy').RLPolicy
        value = load(ROOT / prior.UPSTREAM / 'valuemodel.py', f'{package_name}.valuemodel').ValueModel
        model = self.ledger.call('tensorflow_construction', lambda: value(**helper.MODEL_CONFIGURATION))
        self.ledger.call('tensorflow_build', lambda: model.build_graph(input_shape_nobatch=(105, 105)))
        self.ledger.call('tensorflow_load', lambda: model.load_weights(str(self.paths['weights'])))
        metadata = read(self.paths['tensor_metadata'])
        require(metadata['configuration'] == helper.MODEL_CONFIGURATION and metadata['parameters'] == 13390849
                and len(model.FC_block) == 3 and len(model.weights) == 8, 'unchanged released checkpoint geometry')
        keys = [f'{kind}_{i}' for i in range(4) for kind in ('kernel', 'bias')]
        with np.load(self.paths['tensors'], allow_pickle=False) as archive:
            require(set(archive.files) == set(keys), 'all eight original tensors')
            for i, layer in enumerate([*model.FC_block, model.densefinal]):
                arrays = layer.get_weights()
                require(len(arrays) == 2, 'exact ordered layer weights')
                for kind, actual in zip(('kernel', 'bias'), arrays, strict=True):
                    key = f'{kind}_{i}'
                    expected, record = archive[key], metadata['datasets'][key]
                    require(hashlib.sha256(expected.tobytes()).hexdigest() == record['sha256_c_order']
                            and list(expected.shape) == record['shape'] and expected.dtype.str == record['dtype']
                            and actual.shape == expected.shape and actual.dtype == expected.dtype
                            and actual.tobytes() == expected.tobytes() and np.isfinite(actual).all(), 'exact loaded published tensor')
                    self.ledger.emit('weights.jsonl', {'id': key, 'shape': list(actual.shape),
                                                      'sha256': hashlib.sha256(actual.tobytes()).hexdigest(), 'exact': True})
        raw = time.perf_counter() - tick
        excluded = self.ledger.io_seconds - io_before
        evaluator_tick = time.perf_counter()
        source = load(ROOT / prior.UPSTREAM / 'sourcetracking.py', f'{package_name}.sourcetracking').SourceTracking
        sys.path.insert(0, str(ROOT / 'src'))
        actor = load(ROOT / prior.ACTOR, '_reference_released_actor').ReleasedPolicyActor
        analytic = load(ROOT / CONTROL, '_reference_analytic_actor').SpaceAwareActor
        public = load(ROOT / prior.PUBLIC, '_reference_public_adapter')
        kernels, weights = {}, {}
        for name in COHORTS:
            with np.load(self.paths[f'{name}_kernel'], allow_pickle=False) as archive:
                require(set(archive.files) == {'likelihood', 'initial_hit_weights'}, 'qualified kernel archive')
                kernels[name] = archive['likelihood']
                w = archive['initial_hit_weights']
                require(w.shape == (4,) and w[0] == 0 and np.isfinite(w).all(), 'initial-hit weights geometry')
                weights[name] = {h: float(w[h]) for h in (1, 2, 3)}
        setup = {'model_setup_seconds': raw - excluded, 'model_setup_instrumented_seconds': raw,
                 'model_setup_excluded_io_seconds': excluded, 'allocated_over_released_episodes': 192,
                 'common_import_seconds': common_import_seconds,
                 'shared_evaluator_setup_seconds': time.perf_counter() - evaluator_tick,
                 'model_parameters': 13390849, 'model_tensor_bytes': 53563396,
                 'scope': 'One complete numerical/framework import, source import, model construction/build/load, '
                          'and exact tensor validation. Evaluator/actor-class imports and inherited kernel loading '
                          'are separately timed. Measured journal I/O excluded; no warm-up forward. '
                          'First forward remains charged to its actual episode.'}
        require(setup['model_setup_seconds'] >= 0, 'nonnegative full model setup cost')
        write(self.out / 'runtime.json', {'python': sys.version, 'executable': sys.executable,
              'all_distributions': self.plan['runtime']['all_distributions'], 'environment': helper.ENVIRONMENT,
              'keras_configuration': keras, 'keras_module': tf.keras.Model.__module__,
              'visible_devices': [str(d) for d in tf.config.get_visible_devices()]})
        write(self.out / 'setup.json', setup)
        return types.SimpleNamespace(np=np, source=source, released=actor, analytic=analytic,
                                     public=public, policy=policy, model=model, kernels=kernels, weights=weights, setup=setup)

    def episode(self, runtime, identity):
        np = runtime.np
        cohort, seed, block, hit, arm = identity
        context = {'cohort': cohort, 'seed': seed, 'block': block, 'initial_hit': hit, 'arm': arm}
        episode_tick = time.perf_counter()
        self.ledger.context = {'phase': 'episode_reset', **context}
        config = {'Ndim': 2, 'Ngrid': 53, 'Nhits': 4, 'lambda_over_dx': COHORTS[cohort]['lambda_over_dx'],
                  'R_dt': 2., 'norm_Poisson': 'Euclidean'}
        env = self.ledger.call('native_reset', lambda: runtime.public.seeded_environment(runtime.source, seed, config, initial_hit=hit))
        environment_init = self.ledger.last_seconds['native_reset']
        require(env.N == 53 and env.Nhits == env.Nactions == 4 and env.draw_source is True
                and env.p_Poisson.dtype == runtime.kernels[cohort].dtype
                and env.p_Poisson.tobytes() == runtime.kernels[cohort].tobytes(), 'native environment matches known public kernel')
        current = packet(runtime.public.observation(env, 0))
        initial_public = current
        recorder = ForwardRecorder(runtime.model, runtime.policy._value_policy.__code__, self.ledger, np)

        def construct():
            if arm == 'released_tf':
                return runtime.released(current, runtime.kernels[cohort], recorder, runtime.policy, sym_avg=True)
            return runtime.analytic(current, runtime.kernels[cohort], allow_stay=arm == 'analytic_all4')

        actor = self.ledger.call('actor_construction', construct)
        times = {'actor_initialization_seconds': self.ledger.last_seconds['actor_construction'],
                 'choose_seconds': 0., 'update_seconds': 0.,
                 'model_setup_allocation_seconds': runtime.setup['model_setup_seconds'] / 192 if arm == 'released_tf' else 0.}
        raw_times = {'actor_initialization_seconds': self.ledger.last_instrumented['actor_construction'],
                     'choose_seconds': 0., 'update_seconds': 0.,
                     'model_setup_allocation_seconds': runtime.setup['model_setup_instrumented_seconds'] / 192 if arm == 'released_tf' else 0.}
        storage = actor.storage_bytes()
        state = belief_witness(actor, env, current, np)
        self.ledger.emit('transitions.jsonl', {'kind': 'reset', **context, 'public': current, 'posterior_after': state,
                                             'source_evaluation_only': env.source.tolist()})
        native_seconds = model_seconds = 0.
        stuck = blocked = 0
        for step in range(1, HORIZON + 1):
            self.ledger.context = {'phase': 'decision', **context, 'step': step}
            require(current['done'] is False, 'no action after found')
            before_calls = self.ledger.calls['tensorflow_value']['returned']
            action, costs = self.ledger.call('actor_choose', actor.choose)
            choose = self.ledger.last_seconds['actor_choose']
            choose_raw, choose_io = self.ledger.last_instrumented['actor_choose'], self.ledger.last_io['actor_choose']
            allowed = tuple(range(4)) if arm != 'analytic_inbounds' else current['valid_actions']
            require(action in allowed and costs.shape == (4,)
                    and costs.dtype == (np.float32 if arm == 'released_tf' else np.float64), 'declared action and score contract')
            require(all(np.isfinite(costs[a]) if a in allowed else np.isposinf(costs[a]) for a in range(4)), 'only excluded moves have infinite scores')
            require(int(np.flatnonzero(np.abs(costs - costs.min()) < 1e-10)[0]) == action, 'exact first near-tie selection')
            forwards = self.ledger.calls['tensorflow_value']['returned'] - before_calls
            require(forwards == int(arm == 'released_tf'), 'one actual model call per released decision and zero per analytic decision')
            model_cost = self.ledger.last_seconds['tensorflow_value'] if forwards else 0.
            result = self.ledger.call('native_step', lambda action=action: env.step(action, quiet=True))
            native_cost = self.ledger.last_seconds['native_step']
            after = packet(runtime.public.observation(env, step))
            require((int(result[0]), bool(result[2])) == (after['hit'], after['done']), 'native returned public event')
            self.ledger.call('actor_update', lambda action=action, after=after: actor.update(action, after))
            update = self.ledger.last_seconds['actor_update']
            update_raw = self.ledger.last_instrumented['actor_update']
            after_state = belief_witness(actor, env, after, np)
            is_blocked = current['position'] == after['position']
            stuck += int(env.agent_stuck)
            blocked += int(is_blocked)
            times['choose_seconds'] += choose
            times['update_seconds'] += update
            raw_times['choose_seconds'] += choose_raw
            raw_times['update_seconds'] += update_raw
            native_seconds += native_cost
            model_seconds += model_cost
            self.ledger.emit('transitions.jsonl', {'kind': 'step', **context, 'step': step, 'action': int(action),
                'costs': [float(v) if np.isfinite(v) else None for v in costs], 'allowed_actions': list(allowed),
                'public': after, 'posterior_before': state, 'posterior_after': after_state,
                'native_p_end': float(result[1]), 'blocked': is_blocked, 'stuck': bool(env.agent_stuck),
                'choose_seconds': choose, 'choose_instrumented_seconds': choose_raw, 'choose_excluded_io_seconds': choose_io,
                'update_seconds': update, 'update_instrumented_seconds': update_raw,
                'environment_seconds': native_cost, 'model_forward_seconds': model_cost, 'model_forward_calls': forwards})
            current, state = after, after_state
            if current['done']:
                break
        require(actor.storage_bytes() == storage and current['step'] == step, 'complete final update and fixed actor state storage')
        draws = [{k: r[k] for k in ('channel', 'index', 'uniform', 'selected_index', 'cdf_mass')} for r in env.draw_log]
        require([r['index'] for r in draws if r['channel'] == 'source'] == [0]
                and [r['index'] for r in draws if r['channel'] == 'hit'] == list(range(step - int(current['done'])))
                and all(r['channel'] in ('source', 'hit') and 0 <= r['uniform'] < 1 and math.isfinite(r['cdf_mass']) and r['cdf_mass'] > 0 for r in draws),
                'complete paired categorical source and nonterminal hit draws')
        require(next(r['selected_index'] for r in draws if r['channel'] == 'source') == int(env.source[0]) * 53 + int(env.source[1]),
                'saved source draw agrees with evaluator source')
        row = {**context, 'steps': step, 'capped_time': step, 'found': current['done'], 'stuck_steps': stuck, 'blocked_steps': blocked,
               **times, 'controller_seconds': math.fsum(times.values()), 'controller_instrumented_seconds': math.fsum(raw_times.values()),
               'controller_excluded_io_seconds': math.fsum(raw_times.values()) - math.fsum(times.values()),
               'instrumented_component_seconds': raw_times, 'model_forward_seconds': model_seconds,
               'environment_initialization_seconds': environment_init, 'environment_seconds': native_seconds,
               'episode_seconds': time.perf_counter() - episode_tick, 'state_array_bytes': storage['mutable_array_bytes'],
               'storage': storage, 'choose_calls': step, 'update_calls': step, 'model_forward_calls': step if arm == 'released_tf' else 0,
               'initial_public': initial_public,
               'source_evaluation_only': env.source.tolist(), 'draws_evaluation_only': draws, 'final_update_assimilated': True}
        self.ledger.emit('episodes.jsonl', row)
        return row

    def body(self):
        runtime = self.setup()
        rows, source_pairs, initial_pairs, hit_pairs = [], {}, {}, {}
        for identity in case_order():
            row = self.episode(runtime, identity)
            case = (row['cohort'], row['seed'])
            if case in source_pairs:
                require(row['source_evaluation_only'] == source_pairs[case] and row['initial_public'] == initial_pairs[case],
                        'paired sampled source and initial public state across all arms')
            else:
                source_pairs[case] = row['source_evaluation_only']
                initial_pairs[case] = row['initial_public']
            for draw in row['draws_evaluation_only']:
                key = (*case, draw['channel'], draw['index'])
                if key in hit_pairs:
                    require(hit_pairs[key] == draw['uniform'], 'paired source/hit uniform stream at each shared draw index')
                else:
                    hit_pairs[key] = draw['uniform']
            rows.append(row)
            self.receipt['completed_episodes'] = len(rows)
            print(json.dumps({'completed_episodes': len(rows), 'native_steps': self.ledger.calls['native_step']['returned'],
                              'model_forwards': self.ledger.calls['tensorflow_value']['returned']}), flush=True)
        require(not self.ledger.pending and all(r['attempted'] == r['returned'] for r in self.ledger.calls.values()), 'all actual work returned')
        total_steps = sum(r['steps'] for r in rows)
        expected = {'native_reset': EPISODES, 'native_step': total_steps, 'actor_construction': EPISODES,
                    'actor_choose': total_steps, 'actor_update': total_steps,
                    'tensorflow_value': sum(r['steps'] for r in rows if r['arm'] == 'released_tf'),
                    'tensorflow_construction': 1, 'tensorflow_build': 1, 'tensorflow_load': 1}
        require({k: r['returned'] for k, r in self.ledger.calls.items()} == expected, 'complete actual operation count closure')
        summary = summarize(rows, runtime.weights)
        summary.update(setup=runtime.setup, actual_work=expected, paired_source_cases=len(source_pairs),
                       paired_uniform_checks=len(hit_pairs), all_public_beliefs_exact=True,
                       sum_controller_seconds=math.fsum(r['controller_seconds'] for r in rows),
                       sum_controller_instrumented_seconds=math.fsum(r['controller_instrumented_seconds'] for r in rows),
                       sum_excluded_controller_io_seconds=math.fsum(r['controller_excluded_io_seconds'] for r in rows),
                       artifact_io_seconds=self.ledger.io_seconds,
                       timing_scope='Primary controller cost includes actor initialization, all decisions and all updates, '
                       'plus one complete model/runtime setup allocated over 192 released episodes. Nested measured journal '
                       'and telemetry serialization/fsync are subtracted; raw instrumented costs are retained. '
                       'Forward cost is a subset of choose cost, not added twice. Environment and evaluator parity checks '
                       'are separate. Whole worker/supervisor cost includes all I/O, validation and process overhead. '
                       'Episode time excludes its final row serialization; first cold forward is included.')
        write(self.out / 'summary.json', summary)
        return summary

    def execute(self):
        require(self.out.is_absolute(), 'absolute exclusive output')
        self.out.mkdir(parents=True, exist_ok=False)
        try:
            self.bind()
            result = self.body()
            plan, paths = authenticate(self.args)
            require(plan == self.plan and paths == self.paths and sha(self.args.supervision) == self.receipt['supervision_sha256'], 'unchanged study lineage and launch')
            self.check()
            self.receipt.update(status='completed', work=self.ledger.snapshot(), competent_reference=result['competent_reference'],
                stronger_value_teacher=result['stronger_value_teacher'], utility_compute_advantage=result['utility_compute_advantage'],
                clock_backend=self.clock.backend, started_ns=self.start, finished_ns=self.clock.now_ns(),
                files={p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()})
            self.receipt['wall_seconds'] = (self.receipt['finished_ns'] - self.start) / 1e9
            write(self.out / 'receipt.json', self.receipt)
            self.check()
            return self.receipt
        except BaseException as error:
            self.receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc(), work=self.ledger.snapshot())
            if self.clock is not None and self.start is not None:
                try:
                    self.receipt['failure_elapsed_seconds'] = (self.clock.now_ns() - self.start) / 1e9
                except BaseException as secondary:  # noqa: BLE001 - retain the scientific failure.
                    error.add_note(f'Failure clock unavailable: {secondary!r}')
            try:
                write(self.out / ('failed.json' if (self.out / 'receipt.json').exists() else 'receipt.json'), self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - retain the scientific failure.
                error.add_note(f'Failure receipt publication failed: {secondary!r}')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'supervision', 'output'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    Run(parser.parse_args()).execute()
