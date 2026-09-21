"""Independent saved-only reconstruction of the fixed native/public qualification.

Imports the pinned runner's authentication function only; no Run/body/actor,
policy, simulator or TensorFlow instantiation. Numerical reconstruction uses
NumPy and saved public observations, likelihood kernels, inputs and outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = 'scripts/qualify_otto_released_native.py'
RUNNER_PIN = '96c68e94498c6b83eb2983c1183c444c5d1080eee06ff961562aeb56a845c65e'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
VERSION = 'otto-released-native-saved-audit-v1'
LIMITS = {'seconds': 60, 'rss_bytes': 4 * 1024**3, 'output_bytes': 64 * 1024**2}
CHANNELS = ('native_reset', 'native_step', 'actor_construction', 'actor_update',
            'tensorflow_construction', 'tensorflow_build', 'tensorflow_load', 'tensorflow_value')
EXPECTED_CALLS = dict(zip(CHANNELS, (8, 446, 16, 510, 1, 1, 1, 16), strict=True))
SCOPE = ('Independent reconstruction of all saved public beliefs, eight paired policy input/mass batches, '
         'recorded cost/action consistency and complete work counts. Original native/TF execution, hidden '
         'native-posterior equality assertions and timing truth remain authenticated producer/supervisor evidence. '
         'No model values, simulator episodes, RNG draws or actual performance are independently regenerated. '
         'The failed NumPy policy qualification remains failed.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def read(path):
    return json.loads(path.read_text())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path, check=lambda: None):
    h, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            check()
            h.update(block)
            size += len(block)
    return {'bytes': size, 'sha256': h.hexdigest()}


def cases():
    result = []
    for regime in ('base', 'shift'):
        for hit in (1, 2, 3):
            result.append({'id': f'{regime}.hit{hit}', 'regime': regime, 'initial_hit': hit,
                           'actions': (0, 2, 1, 3), 'hits': (0, 1, 2, 3), 'source': [52, 52],
                           'prefix': (0, 1, 4)[hit - 1], 'censored': True, 'blocked': []})
        boundary = (*([0] * 27), *([2] * 27), *([1] * 53), *([3] * 53), *([0] * 26), *([2] * 25))
        result.append({'id': f'{regime}.boundary', 'regime': regime, 'initial_hit': 1,
                       'actions': boundary, 'hits': (0,) * 211, 'source': [26, 27],
                       'prefix': 27, 'censored': False, 'blocked': [27, 54, 107, 160]})
    for i, row in enumerate(result):
        row['seed'] = 840001 + i
    return result


def move(position, action):
    p = list(position)
    axis, delta = action // 2, 2 * (action % 2) - 1
    p[axis] = max(0, min(52, p[axis] + delta))
    return tuple(p)


def public_packet(position, hit, done, step):
    return {'position': list(position), 'hit': hit, 'done': done, 'step': step,
            'valid_actions': [] if done else [a for a in range(4) if move(position, a) != tuple(position)]}


def update(probability, position, hit, done, kernel, np):
    if done:
        probability = np.zeros((53, 53), dtype=np.float64)
        probability[position] = 1.
        return probability
    probability = probability.copy()
    probability[position] = 0.
    x, y = 53 - position[0], 53 - position[1]
    probability *= kernel[hit, x:x + 53, y:y + 53]
    probability[(probability < 0) & (probability > -1e-15)] = 0
    mass = probability.sum()
    if mass > 1e-10:
        probability /= mass
    require(np.isfinite(probability).all() and (probability >= 0).all(), 'finite reconstructed belief')
    return probability


def inputs_and_masses(probability, position, kernel, np):
    inputs, masses = [], []
    for action in range(4):
        target = move(position, action)
        x, y = 53 - target[0], 53 - target[1]
        joint = probability[None] * kernel[:, x:x + 53, y:y + 53]
        current = []
        for h in range(4):
            mass = max(1e-10, float(joint[h].sum()))
            centered = np.pad(joint[h] / mass, ((52 - target[0], target[0]), (52 - target[1], target[1])))
            inputs.append(centered)
            current.append(mass)
        masses.append(current)
    return np.asarray(inputs, dtype=np.float32), np.asarray(masses, dtype=np.float32)


class Audit:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = None
        self.comparisons = 0
        self.receipt = {'version': VERSION, 'status': 'started', 'scope': SCOPE, 'limits': LIMITS,
                        'model_calls': 0, 'simulator_calls': 0, 'policy_instantiations': 0}

    def check(self):
        require(self.clock.now_ns() - self.start < 60 * 10**9, 'saved audit suspend-inclusive deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        require(rss <= LIMITS['rss_bytes'], 'saved audit RSS cap')
        self.receipt['peak_rss_bytes'] = rss
        require(sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file()) <= LIMITS['output_bytes'], 'saved audit output cap')

    def exact(self, actual, expected, message):
        self.comparisons += 1
        require(actual == expected, message)

    def arrays(self, actual, expected, np, message):
        self.check()
        self.comparisons += int(expected.size)
        require(actual.shape == expected.shape and actual.dtype == expected.dtype
                and np.isfinite(actual).all() and np.isfinite(expected).all()
                and actual.tobytes(order='C') == expected.tobytes(order='C'), message)

    def lines(self, name):
        result = []
        with (self.args.run / name).open() as stream:
            for line in stream:
                self.check()
                result.append(json.loads(line))
        return result

    def authenticate(self):
        a = self.args
        require(all(p.is_absolute() for p in (a.plan, a.run, a.terminal, a.output)), 'absolute external paths')
        for p in (a.plan, a.run, a.terminal):
            require(not any(x.is_symlink() for x in (p, *p.parents)), 'no symbolic input path')
        self.exact(digest(a.plan, self.check)['sha256'], a.plan_sha256, 'external plan pin')
        self.exact(digest(a.run / 'receipt.json', self.check)['sha256'], a.receipt_sha256, 'external worker receipt pin')
        self.exact(digest(a.terminal, self.check)['sha256'], a.terminal_sha256, 'external supervisor terminal pin')
        plan, worker, terminal = read(a.plan), read(a.run / 'receipt.json'), read(a.terminal)
        self.exact(plan['sources'][RUNNER], RUNNER_PIN, 'fixed producer authentication source')
        self.exact(digest(ROOT / RUNNER, self.check)['sha256'], RUNNER_PIN, 'producer source bytes before auth-only import')
        auth = load(ROOT / RUNNER, '_otto_native_audit_auth_only')
        validated, paths = auth.authenticate(a)
        self.check()
        self.exact(validated, plan, 'complete producer source/input/runtime authentication')
        require(worker['version'] == 'otto-released-native-qualification-v1' and worker['status'] == 'completed'
                and worker['qualified'] is True and worker['numpy_port_admitted'] is False
                and worker['prior_numpy_action_failure_preserved'] is True, 'completed qualified native worker only')
        self.exact(worker['plan_sha256'], a.plan_sha256, 'worker plan join')
        self.exact(worker['sources'], plan['sources'], 'worker source closure')
        self.exact(worker['inputs'], plan['inputs'], 'worker input closure')
        self.exact(worker['limits'], plan['limits'], 'worker budget identity')
        require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
                and terminal['group_absent'] is True and terminal['cleanup']['group_absent'] is True
                and terminal['cleanup']['errors'] == [] and terminal['cleanup']['reaped'] is True
                and terminal['clock_error'] is None and terminal['error'] is None
                and terminal['timing_available'] is True, 'successful cleaned native parent')
        expected_members = {'started.json', 'runtime.json', 'weights.jsonl', 'work.jsonl', 'transitions.jsonl',
                            'cases.jsonl', 'policy-checks.jsonl', 'summary.json', *(f"prefix-{c['id']}.npz" for c in cases())}
        self.exact(set(worker['files']), expected_members, 'all sixteen saved payloads')
        self.exact({p.name for p in a.run.iterdir()}, expected_members | {'receipt.json'}, 'complete undemoted seventeen-file run')
        for name, record in worker['files'].items():
            p = a.run / name
            require(p.is_file() and not p.is_symlink(), 'regular saved payload')
            self.exact(digest(p, self.check), record, f'payload bytes: {name}')
        require(type(worker['peak_rss_bytes']) is int
                and 0 < worker['peak_rss_bytes'] <= plan['limits']['rss_bytes'], 'recorded worker RSS within frozen cap')
        require(sum(p.stat().st_size for p in a.run.iterdir()) <= plan['limits']['output_bytes'],
                'complete run including receipt within frozen output cap')
        started = read(a.run / 'started.json')
        launch = started['launch']
        request = started['request']
        self.exact(request['plan'], str(a.plan), 'request plan path')
        self.exact(request['plan_sha256'], a.plan_sha256, 'request plan pin')
        self.exact(request['output'], str(a.run), 'request run path')
        launch_path = Path(request['supervision'])
        self.exact(digest(launch_path, self.check)['sha256'], worker['supervision_sha256'], 'actual launch pin')
        self.exact(read(launch_path), launch, 'actual embedded launch')
        for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns',
                    'clock_backend', 'watchdog_sha256', 'clock_source_sha256', 'cap_seconds'):
            self.exact(terminal[key], launch[key], f'parent join: {key}')
        command = list(terminal['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        self.exact(command[:2], [plan['runtime']['python_executable'], str(ROOT / RUNNER)], 'exact producer/interpreter route')
        require(len(command) == 10, 'exact four producer arguments')
        arguments = dict(zip(command[2::2], command[3::2], strict=True))
        self.exact(arguments, {'--plan': str(a.plan), '--plan-sha256': a.plan_sha256,
                               '--supervision': str(launch_path), '--output': str(a.run)}, 'actual producer command arguments')
        require(terminal['cap_seconds'] == 600 and terminal['clock_backend'] == worker['clock_backend']
                and terminal['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
                and terminal['deadline_ns'] == terminal['started_ns'] + 600 * 10**9
                and terminal['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < terminal['deadline_ns'],
                'native parent and worker timing bounds')
        self.exact(terminal['elapsed_ns'], terminal['finished_ns'] - terminal['started_ns'], 'parent elapsed arithmetic')
        self.exact(terminal['wall_seconds'], terminal['elapsed_ns'] / 1e9, 'parent seconds arithmetic')
        self.exact(worker['wall_seconds'], (worker['finished_ns'] - worker['started_ns']) / 1e9, 'worker seconds arithmetic')
        self.exact(terminal['watchdog_sha256'], plan['sources']['scripts/supervise_dialogue_observation_v2.py'], 'supervisor source')
        self.exact(terminal['clock_source_sha256'], CLOCK_PIN, 'native clock source')
        self.exact(Path(terminal['cwd']).resolve(), ROOT, 'actual producer cwd')
        runtime = read(a.run / 'runtime.json')
        self.exact(runtime['executable'], plan['runtime']['python_executable'], 'recorded producer interpreter')
        self.exact(runtime['all_distributions'], plan['runtime']['all_distributions'], 'recorded complete runtime')
        self.exact(runtime['keras_configuration'], {'floatx': 'float32', 'image_data_format': 'channels_last',
                                                   'mixed_precision_policy': 'float32'}, 'recorded float32 Keras contract')
        require(runtime['keras_module'].startswith('tf_keras.') and runtime['visible_devices']
                and all('GPU' not in device for device in runtime['visible_devices']), 'recorded legacy CPU runtime')
        self.receipt.update(plan_sha256=a.plan_sha256, worker_sha256=a.receipt_sha256,
                            terminal_sha256=a.terminal_sha256, producer_source_sha256=RUNNER_PIN,
                            authentication_reuse='Only pinned runner.authenticate; no producer numerical helpers used.')
        return plan, paths, worker

    def ledger(self, worker):
        counts = {name: {'attempted': 0, 'returned': 0} for name in CHANNELS}
        pending, phase, case = None, 'authentication', None
        native_steps, native_resets, forwards = [], [], []
        for row in self.lines('work.jsonl'):
            self.exact(set(row), {'event', 'phase', 'case', 'pending_call', 'calls'}, 'ledger event shape')
            event = row['event']
            if event == 'context':
                require(pending is None, 'context outside pending call')
                phase, case = row['phase'], row['case']
            elif event == 'attempt':
                require(pending is None, 'no overlapping or silently abandoned call')
                pending = row['pending_call']
                channel = pending['channel']
                require(channel in counts, 'declared operation channel')
                counts[channel]['attempted'] += 1
                self.exact(pending, {'channel': channel, 'ordinal': counts[channel]['attempted'],
                                      'phase': phase, 'case': case}, 'call identity before execution')
                if channel == 'native_step':
                    require(phase == 'prescribed_transition', 'all native steps mechanical')
                    native_steps.append(case)
                elif channel == 'native_reset':
                    require(phase == 'reset', 'all native resets explicit')
                    native_resets.append(case)
                elif channel == 'tensorflow_value':
                    require(phase == 'policy_prefix', 'zero primary actor policy forwards')
                    forwards.append(case)
            elif event == 'return':
                require(pending is not None, 'return has a preceding attempt')
                counts[pending['channel']]['returned'] += 1
                pending = None
            else:
                raise ValueError('unknown work event')
            self.exact(row['phase'], phase, 'ledger phase continuity')
            self.exact(row['case'], case, 'ledger case continuity')
            self.exact(row['pending_call'], pending, 'ledger pending state')
            self.exact(row['calls'], counts, 'ledger count reconstruction')
        self.exact(counts, {name: {'attempted': n, 'returned': n} for name, n in EXPECTED_CALLS.items()}, 'all complete expected work')
        self.exact(worker['work'], {'phase': phase, 'case': case, 'pending_call': None, 'calls': counts}, 'receipt final work join')
        self.exact(native_resets, [c['id'] for c in cases()], 'eight ordered native resets')
        self.exact(native_steps, [{'fixture': c['id'], 'step': t} for c in cases() for t in range(1, len(c['actions']) + 1)], '446ordered native transitions')
        self.exact(forwards, [c['id'] for c in cases() for _ in range(2)], 'sixteen paired original-TF calls')
        return counts

    def numerics(self, paths, np):
        kernels = {}
        for regime in ('base', 'shift'):
            with np.load(paths[f'{regime}_kernel'], allow_pickle=False) as saved:
                self.exact(set(saved.files), {'likelihood', 'initial_hit_weights'}, 'closed kernel archive keys')
                kernel = saved['likelihood']
            require(kernel.shape == (4, 107, 107) and kernel.dtype == np.float64
                    and np.isfinite(kernel).all() and (kernel >= 0).all() and (kernel <= 1).all(), 'finite known observation kernel')
            kernels[regime] = kernel
        metadata = read(paths['tensor_metadata'])
        weight_rows = self.lines('weights.jsonl')
        self.exact([r['id'] for r in weight_rows], [f'{kind}_{i}' for i in range(4) for kind in ('kernel', 'bias')], 'eight loaded ordered tensors')
        for row in weight_rows:
            require(row['exact'] is True, 'producer reported exact loaded tensor')
            self.exact(row['shape'], metadata['datasets'][row['id']]['shape'], 'loaded tensor shape record')
            self.exact(row['sha256'], metadata['datasets'][row['id']]['sha256_c_order'], 'loaded tensor hash record')
        transitions, prefixes, reconstructed_cases, i = self.lines('transitions.jsonl'), {}, [], 0
        self.exact(len(transitions), 454, 'eight resets plus446steps')
        for case in cases():
            kernel = kernels[case['regime']]
            position = (26, 26)
            probability = np.ones((53, 53), dtype=np.float64) / 2808
            probability[26, 26] = 0.
            probability = update(probability, position, case['initial_hit'], False, kernel, np)
            blocked = []
            for step in range(len(case['actions']) + 1):
                row = transitions[i]
                i += 1
                if step == 0:
                    expected_public = public_packet(position, case['initial_hit'], False, 0)
                    self.exact(row['kind'], 'reset', 'reset trace record')
                    self.exact(row['seed_evaluation_only'], case['seed'], 'fixed qualification seed')
                    self.exact(row['source_override_evaluation_only'], case['source'], 'mechanical source explicitly evaluator-only')
                else:
                    action, prescribed_hit = case['actions'][step - 1], case['hits'][step - 1]
                    previous_position = position
                    position = move(position, action)
                    found = list(position) == case['source']
                    hit = -2 if found else prescribed_hit
                    probability = update(probability, position, hit, found, kernel, np)
                    expected_public = public_packet(position, hit, found, step)
                    self.exact(row['kind'], 'step', 'primitive transition trace')
                    self.exact(row['action'], action, 'prescribed action')
                    self.exact(row['prescribed_hit_mechanical'], prescribed_hit, 'prescribed mechanical reading')
                    self.exact(row['blocked'], previous_position == position, 'all blocked transitions included')
                    self.exact(row['native_p_end'], float(found), 'sampled-source terminal value')
                    if previous_position == position:
                        blocked.append(step)
                    require(not found or step == len(case['actions']), 'no action after found')
                self.exact(row['case'], case['id'], 'trace case order')
                self.exact(row['public'], expected_public, 'exact public fields and movement')
                expected_state = {'step': step, 'done': expected_public['done'], 'mass': float(probability.sum()),
                                  'belief_sha256': hashlib.sha256(probability.tobytes(order='C')).hexdigest(), 'exact': True}
                self.exact(row['state'], expected_state, 'independently reconstructed public belief hash/mass')
                if step == case['prefix']:
                    prefixes[case['id']] = (probability.copy(), position)
            self.exact(blocked, case['blocked'], 'all four blocked directions or untouched square')
            self.exact(expected_public['done'], not case['censored'], 'found versus censor distinction')
            if case['censored']:
                require(step == 4 and expected_public['hit'] == 3, 'final nonterminal observation assimilated')
            else:
                require(step == 211 and probability[position] == 1. and np.count_nonzero(probability) == 1, 'found terminal point mass')
            reconstructed_cases.append({'case': case['id'], 'regime': case['regime'], 'initial_hit': case['initial_hit'],
                'native_steps': len(case['actions']), 'blocked_steps': blocked, 'censored': case['censored'],
                'found': not case['censored'], 'final_update_assimilated': True, 'all_beliefs_exact': True})
        self.exact(self.lines('cases.jsonl'), reconstructed_cases, 'saved eight fixture summaries')
        policy_rows = self.lines('policy-checks.jsonl')
        self.exact([r['case'] for r in policy_rows], [c['id'] for c in cases()], 'all eight paired policy prefixes')
        maximum_cost_error = maximum_cost_bound_ratio = 0.
        for case, record in zip(cases(), policy_rows, strict=True):
            self.check()
            probability, position = prefixes[case['id']]
            expected_inputs, expected_masses = inputs_and_masses(probability, position, kernels[case['regime']], np)
            with np.load(self.args.run / f"prefix-{case['id']}.npz", allow_pickle=False) as saved:
                names = {f'{view}_{n}' for view in ('native', 'public') for n in ('inputs', 'masses', 'values', 'scores')}
                self.exact(set(saved.files), names, 'exact policy array archive keys')
                arrays = {n: saved[n] for n in names}
            for view in ('native', 'public'):
                self.arrays(arrays[f'{view}_inputs'], expected_inputs, np, f'{view} public-history policy inputs')
                self.arrays(arrays[f'{view}_masses'], expected_masses, np, f'{view} public-history branch masses')
            self.arrays(arrays['native_values'], arrays['public_values'], np, 'saved same-TF values byte parity')
            self.arrays(arrays['native_scores'], arrays['public_scores'], np, 'saved same-TF costs byte parity')
            values, costs = arrays['native_values'], arrays['native_scores']
            require(values.shape == (16, 1) and values.dtype == np.float32 and costs.shape == (4,)
                    and costs.dtype == np.float32, 'recorded original TF output geometry')
            products = expected_masses.astype(np.float64) * values.reshape(4, 4).astype(np.float64)
            exact_sum = 1. + products.sum(axis=1)
            # Product + three sums + final addition. This conservative bound only
            # checks saved cost arithmetic, never relaxes the exact pair/action gate.
            bound = np.finfo(np.float32).eps * (6 * np.abs(products).sum(axis=1) + np.abs(exact_sum)) + np.finfo(np.float32).tiny
            error = np.abs(costs.astype(np.float64) - exact_sum)
            require((error <= bound).all(), 'saved costs consistent with float32 product/reduction rounding')
            maximum_cost_error = max(maximum_cost_error, float(error.max()))
            maximum_cost_bound_ratio = max(maximum_cost_bound_ratio, float((error / bound).max()))
            chosen = int(np.flatnonzero(np.abs(costs - costs.min()) < 1e-10)[0])
            self.exact(record['native_action'], chosen, 'native first near-tie action')
            self.exact(record['public_action'], chosen, 'public first near-tie action')
            self.exact(record['native_scores'], costs.tolist(), 'native JSON costs')
            self.exact(record['public_scores'], costs.tolist(), 'public JSON costs')
            self.exact(record['step'], case['prefix'], 'fixed before-action prefix')
            self.exact(record['comparisons'], dict.fromkeys(('inputs', 'masses', 'values', 'scores', 'action'), True), 'all exact pair decisions')
            require(record['passed'] is True and record['symmetry_average'] is True, 'complete fixed policy agreement')
        summary = read(self.args.run / 'summary.json')
        self.exact(summary['cases'], reconstructed_cases, 'published case summaries')
        self.exact(summary['policy_prefixes'], policy_rows, 'published policy records')
        for key, expected in {'status': 'completed', 'qualified': True, 'native_steps': 446, 'native_resets': 8,
                              'tensorflow_value_calls': 16, 'symmetry_average': True, 'numpy_port_admitted': False,
                              'prior_numpy_action_failure_preserved': True, 'autonomous_episodes': 0,
                              'training_updates': 0, 'weight_identity': True}.items():
            self.exact(summary[key], expected, f'complete qualification summary: {key}')
        return {'belief_snapshots_reconstructed': 454, 'policy_prefixes_reconstructed': 8,
                'loaded_weight_hash_records_checked': 8, 'maximum_float64_cost_reconstruction_error': maximum_cost_error,
                'maximum_float32_roundoff_bound_ratio': maximum_cost_bound_ratio,
                'exact_pair_gate_unchanged': True, 'cases': reconstructed_cases}

    def execute(self):
        require(self.out.is_absolute(), 'absolute exclusive audit output')
        self.out.mkdir(parents=True, exist_ok=False)
        old_alarm = signal.getsignal(signal.SIGALRM)

        def timeout(_signal, _frame):
            raise TimeoutError('saved audit 60-second alarm')

        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, 60)
        try:
            require(digest(ROOT / CLOCK)['sha256'] == CLOCK_PIN, 'registered clock source before evidence reads')
            self.clock = load(ROOT / CLOCK, '_otto_native_saved_audit_clock').SuspendClock()
            self.start = self.clock.now_ns()
            require(self.clock.backend in ('mach_continuous_time', 'CLOCK_BOOTTIME'), 'native suspend-inclusive audit clock')
            write(self.out / 'started.json', {'clock_backend': self.clock.backend, 'started_ns': self.start,
                  'request': {k: str(v) for k, v in vars(self.args).items()}, 'limits': LIMITS})
            self.receipt['source'] = {'path': str(Path(__file__).resolve()), **digest(Path(__file__).resolve(), self.check)}
            plan, paths, worker = self.authenticate()
            self.ledger(worker)
            os.environ.update({name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                                     'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')})
            import numpy as np
            result = self.numerics(paths, np)
            require(not any(name in sys.modules for name in ('tensorflow', 'tf_keras', '_otto_released_native_actor',
                                                             '_otto_released_native_upstream.sourcetracking')),
                    'no framework, actor or simulator execution')
            self.authenticate()  # Recheck every pinned source/payload/runtime after saved reconstruction.
            self.check()
            result.update(status='completed', agreement=True, comparisons=self.comparisons, scope=SCOPE,
                          work=EXPECTED_CALLS, numpy_version=np.__version__, plan_version=plan['version'])
            write(self.out / 'summary.json', result)
            self.receipt.update(status='completed', agreement=True, comparisons=self.comparisons,
                                clock_backend=self.clock.backend, started_ns=self.start, finished_ns=self.clock.now_ns(),
                                files={p.name: digest(p, self.check) for p in self.out.iterdir() if p.is_file()})
            self.receipt['wall_seconds'] = (self.receipt['finished_ns'] - self.start) / 1e9
            write(self.out / 'receipt.json', self.receipt)
            self.check()
            print(json.dumps({'status': 'completed', 'agreement': True, 'comparisons': self.comparisons,
                              'receipt_sha256': digest(self.out / 'receipt.json', self.check)['sha256']}))
            return self.receipt
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            self.receipt.update(status='failed', agreement=False, error=repr(error), traceback=traceback.format_exc(), comparisons=self.comparisons)
            try:
                write(self.out / ('failed.json' if (self.out / 'receipt.json').exists() else 'receipt.json'), self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - retain primary audit failure.
                error.add_note(f'Audit failure receipt publication failed: {secondary!r}')
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'run', 'terminal', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    for name in ('plan-sha256', 'receipt-sha256', 'terminal-sha256'):
        parser.add_argument(f'--{name}', required=True)
    return Audit(parser.parse_args()).execute()


if __name__ == '__main__':
    main()
