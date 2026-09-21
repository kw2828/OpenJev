"""Mechanical native/public integration qualification of the released TF policy.

No NumPy policy port, autonomous evaluation, training, or performance claim.
All source overrides and prescribed observations are evaluator-only fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import resource
import sys
import time
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'otto-released-native-qualification-v1'
ENGINE = 'scripts/qualify_otto_pretrained.py'
ENGINE_PIN = '7c6514831ea204938905bf36575d50bc9abb5365be19bcc4c396fdad672c035a'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
ACTOR = 'src/openjev/research/otto_released_policy.py'
PUBLIC = 'src/openjev/research/otto_public.py'
UPSTREAM = 'tmp/otto-source-review-01/isotropic/classes'
INTERPRETER = '.venv-otto-released-native/bin/python'
PROTOCOL = 'research/otto-released-native-qualification-protocol.md'
LIMITS = {'native_seconds': 600, 'rss_bytes': 4 * 1024**3, 'output_bytes': 64 * 1024**2,
          'native_steps': 446, 'native_resets': 8, 'tensorflow_value_calls': 16}
CONFIGURATION = {'regimes': {'base': 3., 'shift': 4.}, 'R_dt': 2., 'Ngrid': 53, 'Nhits': 4,
                 'Ndim': 2, 'norm_Poisson': 'Euclidean', 'initial_hits': [1, 2, 3],
                 'seeds': list(range(840001, 840009)), 'symmetry_average': True,
                 'square_actions': [0, 2, 1, 3], 'square_hits': [0, 1, 2, 3],
                 'square_source_evaluation_only': [52, 52], 'boundary_source_evaluation_only': [26, 27],
                 'prefix_steps': {'hit1': 0, 'hit2': 1, 'hit3': 4, 'boundary': 27},
                 'belief_input_mass_score_action_parity': 'exact', 'policy_tie_epsilon': 1e-10}
REQUIRED = {ENGINE, CLOCK, ACTOR, PUBLIC, PROTOCOL, 'scripts/qualify_otto_released_native.py',
            'tests/test_qualify_otto_released_native.py', 'tests/test_otto_released_policy.py',
            'scripts/supervise_dialogue_observation_v2.py',
            *(f'{UPSTREAM}/{n}.py' for n in ('sourcetracking', 'valuemodel', 'rlpolicy', 'policy'))}
ROLES = {'previous_plan', 'previous_receipt', 'previous_terminal', 'weights', 'tensors',
         'tensor_metadata', 'base_kernel', 'shift_kernel'}
SCOPE = ('Prescribed mechanical native/public integration only. Six fixed hit-conditioned square paths and two '
         'boundary/found paths. The actor receives public packets, known kernels, and the original TensorFlow policy; '
         'it never receives the simulator, hidden source, native posterior, seed, reward, or draw log. '
         'The prior NumPy port remains unqualified because its exact-action comparison failed.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


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


def descriptor(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def regular(name):
    relative = Path(name)
    require(not relative.is_absolute() and relative.parts and '..' not in relative.parts, 'contained source/input path')
    p = ROOT / relative
    require(p.is_file() and not any(x.is_symlink() for x in (p, *p.parents))
            and p.resolve().is_relative_to(ROOT), 'regular contained file')
    return p


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize_distribution(name):
    return name.lower().replace('_', '-').replace('.', '-')


def authenticate(args):
    require(args.plan.is_absolute() and not args.plan.is_symlink() and sha(args.plan) == args.plan_sha256, 'external native plan pin')
    plan = read(args.plan)
    require(plan['version'] == VERSION and plan['status'] == 'frozen_before_run'
            and plan['configuration'] == CONFIGURATION and plan['limits'] == LIMITS, 'fixed native qualification plan')
    require(REQUIRED <= plan['sources'].keys() and plan['sources'][ENGINE] == ENGINE_PIN
            and plan['sources'][CLOCK] == CLOCK_PIN, 'complete native qualification source closure')
    for name, pin in plan['sources'].items():
        require(sha(regular(name)) == pin, f'current source hash: {name}')
    require(set(plan['inputs']) == ROLES, 'exact native input roles')
    paths = {}
    for name, record in plan['inputs'].items():
        paths[name] = regular(record['path'])
        require(descriptor(paths[name]) == {'bytes': record['bytes'], 'sha256': record['sha256']}, f'input bytes: {name}')
    previous, receipt, terminal = (read(paths[name]) for name in ('previous_plan', 'previous_receipt', 'previous_terminal'))
    prior_dir = paths['previous_receipt'].parent
    require(previous['version'] == 'otto-pretrained-qualification-runtime-v2'
            and previous['status'] == 'frozen_before_run' and receipt['version'] == previous['version']
            and receipt['status'] == 'completed' and receipt['plan_sha256'] == sha(paths['previous_plan'])
            and receipt['sources'] == previous['sources'] and receipt['inputs'] == previous['inputs'], 'closed v2 numerical qualification')
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0
            and terminal['timed_out'] is False and terminal['group_absent'] is True
            and terminal['started_ns'] <= receipt['started_ns'] <= receipt['finished_ns'] <= terminal['finished_ns'], 'successful prior parent terminal')
    command = terminal['command']
    for flag, value in (('--plan', str(paths['previous_plan'])), ('--plan-sha256', receipt['plan_sha256']), ('--output', str(prior_dir))):
        require(command.count(flag) == 1 and command[command.index(flag) + 1] == value, 'prior parent exact command join')
    started = read(prior_dir / 'started.json')
    launch_path = Path(started['request']['supervision'])
    require(sha(launch_path) == receipt['supervision_sha256'] and read(launch_path) == started['launch'], 'prior actual launch join')
    for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend'):
        require(terminal[key] == started['launch'][key], f'prior parent launch/terminal identity: {key}')
    require({p.name for p in prior_dir.iterdir()} == set(receipt['files']) | {'receipt.json'}, 'complete undemoted prior closure')
    for name, record in receipt['files'].items():
        require(Path(name).name == name and not (prior_dir / name).is_symlink()
                and descriptor(prior_dir / name) == record, f'closed prior payload: {name}')
    for name, pin in previous['sources'].items():
        require(sha(regular(name)) == pin, f'unchanged previous source: {name}')
    summary = read(prior_dir / 'summary.json')
    require(all(summary[name] is True for name in ('weight_identity', 'policy_input_identity', 'value_parity', 'policy_cost_parity'))
            and summary['qualified'] is receipt['qualified'] is False and summary['policy_action_parity'] is False,
            'retain the prior exact-action failure while reusing only the original TF model')
    require(summary['tensorflow_value_calls'] == summary['numpy_value_calls'] == 82
            and receipt['work']['pending_call'] is None
            and all(receipt['work']['calls'][name] == {'attempted': 82, 'returned': 82}
                    for name in ('tensorflow_value', 'numpy_value')), 'complete previous calls, no unresolved forward')
    require(paths['weights'] == prior_dir / 'original.weights-legacy.h5'
            and descriptor(paths['weights']) == receipt['files']['original.weights-legacy.h5']
            and sha(paths['weights']) == previous['inputs']['original_weights']['sha256'], 'original byte-identical HDF5 input')
    for role in ('tensors', 'tensor_metadata', 'base_kernel', 'shift_kernel'):
        require(plan['inputs'][role] == previous['inputs'][role], f'unchanged inherited input: {role}')
    runtime = plan['runtime']
    require(runtime['python_executable'] == str(ROOT / INTERPRETER) == str(Path(sys.executable).absolute())
            and runtime['python_version'] == previous['runtime']['python_version'] == sys.version.split()[0], 'separate native reference interpreter')
    expected = {normalize_distribution(k): v for k, v in previous['runtime']['all_distributions'].items()}
    require('scipy' not in expected, 'prior runtime remains unmodified')
    expected['scipy'] = '1.18.1'
    declared = {normalize_distribution(k): v for k, v in runtime['all_distributions'].items()}
    actual = {normalize_distribution(d.metadata['Name']): d.version for d in importlib.metadata.distributions()}
    require(declared == actual == expected, 'native runtime adds only pinned SciPy to the complete prior distribution set')
    return plan, paths


def fixtures():
    cases = []
    for regime in ('base', 'shift'):
        for hit in (1, 2, 3):
            cases.append({'id': f'{regime}.hit{hit}', 'regime': regime, 'initial_hit': hit,
                          'source_evaluation_only': [52, 52], 'actions': [0, 2, 1, 3], 'hits': [0, 1, 2, 3],
                          'prefix_step': {1: 0, 2: 1, 3: 4}[hit], 'censored': True})
        actions = [0] * 27 + [2] * 27 + [1] * 53 + [3] * 53 + [0] * 26 + [2] * 25
        cases.append({'id': f'{regime}.boundary', 'regime': regime, 'initial_hit': 1,
                      'source_evaluation_only': [26, 27], 'actions': actions, 'hits': [0] * len(actions),
                      'prefix_step': 27, 'censored': False, 'blocked_steps': [27, 54, 107, 160]})
    for seed, case in zip(CONFIGURATION['seeds'], cases, strict=True):
        case['seed_evaluation_only'] = seed
    return cases


class Ledger:
    CHANNELS = ('native_reset', 'native_step', 'actor_construction', 'actor_update',
                'tensorflow_construction', 'tensorflow_build', 'tensorflow_load', 'tensorflow_value')

    def __init__(self, out, check):
        self.out, self.check = out, check
        self.state = {'phase': 'authentication', 'case': None, 'pending_call': None,
                      'calls': {k: {'attempted': 0, 'returned': 0} for k in self.CHANNELS}}

    def context(self, phase, case=None):
        require(self.state['pending_call'] is None, 'no context change during uncertain call')
        self.state.update(phase=phase, case=case)
        append(self.out / 'work.jsonl', {'event': 'context', **self.state})

    def call(self, channel, operation):
        self.check()
        require(self.state['pending_call'] is None and channel in self.CHANNELS, 'one declared work call')
        row = self.state['calls'][channel]
        row['attempted'] += 1
        self.state['pending_call'] = {'channel': channel, 'ordinal': row['attempted'], 'phase': self.state['phase'], 'case': self.state['case']}
        append(self.out / 'work.jsonl', {'event': 'attempt', **self.state})
        value = operation()
        row['returned'] += 1
        self.state['pending_call'] = None
        append(self.out / 'work.jsonl', {'event': 'return', **self.state})
        self.check()
        return value


def array_identity(first, second, np):
    return (first.shape == second.shape and first.dtype == second.dtype and np.isfinite(first).all()
            and np.isfinite(second).all() and first.tobytes(order='C') == second.tobytes(order='C'))


def packet(value):
    return {'position': tuple(value.position), 'hit': int(value.hit), 'done': bool(value.done),
            'step': int(value.step), 'valid_actions': tuple(value.valid_actions)}


def check_state(actor, env, public, np):
    require(actor.public == public, 'exact native/public packet parity')
    probability = actor.belief
    require(array_identity(probability, env.p_source, np), 'exact float64 native/public posterior parity')
    require(probability.dtype == np.float64 and probability.shape == (53, 53), 'native posterior geometry')
    return {'step': public['step'], 'done': public['done'], 'mass': float(probability.sum()),
            'belief_sha256': hashlib.sha256(probability.tobytes()).hexdigest(), 'exact': True}


def policy_pair(case, env, initial, history, kernel, Actor, RLPolicy, model, helper, ledger, out, np):
    """Reconstruct only public history into a separate actor, then discard its pending choice."""
    ledger.context('policy_prefix', case['id'])
    native_recorder = helper.RecordingModel(model, RLPolicy._value_policy.__code__, np, ledger)
    public_recorder = helper.RecordingModel(model, RLPolicy._value_policy.__code__, np, ledger)
    actor = ledger.call('actor_construction', lambda: Actor(initial, kernel, public_recorder, RLPolicy, sym_avg=True))
    for action, public in history:
        ledger.call('actor_update', lambda action=action, public=public: actor.update(action, public))
    check_state(actor, env, history[-1][1] if history else initial, np)
    native_action, native_scores = RLPolicy(env, native_recorder, sym_avg=True)._value_policy()
    public_action, public_scores = actor.choose()
    require(len(native_recorder.records) == len(public_recorder.records) == 1, 'one forward per view at each prefix')
    first, second = native_recorder.records[0], public_recorder.records[0]
    comparisons = {name: bool(array_identity(first[name], second[name], np)) for name in ('inputs', 'masses', 'values')}
    comparisons.update(scores=bool(array_identity(native_scores, public_scores, np)), action=int(native_action) == int(public_action))
    arrays = {f'{view}_{name}': record[name] for view, record in (('native', first), ('public', second))
              for name in ('inputs', 'masses', 'values')}
    arrays.update(native_scores=native_scores, public_scores=public_scores)
    with (out / f"prefix-{case['id']}.npz").open('xb') as stream:
        np.savez(stream, **arrays)
    row = {'case': case['id'], 'step': case['prefix_step'], 'comparisons': comparisons,
           'native_action': int(native_action), 'public_action': int(public_action),
           'native_scores': native_scores.tolist(), 'public_scores': public_scores.tolist(),
           'passed': all(comparisons.values()), 'symmetry_average': True,
           'scope': 'Native view is evaluator-only; public actor is reconstructed exclusively from public packets/actions.'}
    append(out / 'policy-checks.jsonl', row)
    require(row['passed'], 'exact original-TF native/public policy parity')
    return row


def mechanical_run(SourceTracking, seeded, observe, Actor, RLPolicy, model, helper, kernels, ledger, out, np):
    rows, pairs = [], []
    for case in fixtures():
        ledger.context('reset', case['id'])
        config = {'Ndim': 2, 'Ngrid': 53, 'Nhits': 4, 'lambda_over_dx': CONFIGURATION['regimes'][case['regime']],
                  'R_dt': 2., 'norm_Poisson': 'Euclidean'}
        env = ledger.call('native_reset', lambda case=case, config=config:
                          seeded(SourceTracking, case['seed_evaluation_only'], config, initial_hit=case['initial_hit']))
        require(env.N == 53 and env.Nhits == 4 and env.Nactions == 4, 'exact native geometry')
        kernel = kernels[case['regime']]
        require(array_identity(env.p_Poisson, kernel, np), 'actual native kernel exactly equals closed public kernel')
        # Explicit qualification-only override, never an actor input or policy choice.
        env.source = np.asarray(case['source_evaluation_only'], dtype=int)
        initial = packet(observe(env, 0))
        primary_recorder = helper.RecordingModel(model, RLPolicy._value_policy.__code__, np, ledger)
        actor = ledger.call('actor_construction', lambda initial=initial, kernel=kernel, primary_recorder=primary_recorder:
                            Actor(initial, kernel, primary_recorder, RLPolicy, sym_avg=True))
        state = check_state(actor, env, initial, np)
        append(out / 'transitions.jsonl', {'kind': 'reset', 'case': case['id'], 'public': initial, 'state': state,
                                         'seed_evaluation_only': case['seed_evaluation_only'],
                                         'source_override_evaluation_only': case['source_evaluation_only']})
        history, blocked = [], []
        if case['prefix_step'] == 0:
            pairs.append(policy_pair(case, env, initial, history, kernel, Actor, RLPolicy, model, helper, ledger, out, np))
        for step, (action, forced_hit) in enumerate(zip(case['actions'], case['hits'], strict=True), 1):
            ledger.context('prescribed_transition', {'fixture': case['id'], 'step': step})
            require(not env.obs['done'], 'no action after source found')
            before = tuple(env.agent)
            result = ledger.call('native_step', lambda action=action, forced_hit=forced_hit, env=env:
                                 env.step(action, hit=forced_hit, quiet=True))
            public = packet(observe(env, step))
            require((int(result[0]), bool(result[2])) == (public['hit'], public['done']), 'native result/public join')
            ledger.call('actor_update', lambda action=action, public=public, actor=actor: actor.update(action, public))
            state = check_state(actor, env, public, np)
            is_blocked = before == public['position']
            if is_blocked:
                blocked.append(step)
            history.append((action, public))
            append(out / 'transitions.jsonl', {'kind': 'step', 'case': case['id'], 'action': action,
                  'prescribed_hit_mechanical': forced_hit, 'public': public, 'state': state,
                  'blocked': is_blocked, 'native_p_end': float(result[1])})
            if case['prefix_step'] == step:
                pairs.append(policy_pair(case, env, initial, history, kernel, Actor, RLPolicy, model, helper, ledger, out, np))
        final = actor.public
        if case['censored']:
            require(final['step'] == 4 and final['done'] is False and final['hit'] == 3,
                    'censor after assimilating final nonterminal reading without fabricating done')
            require(blocked == [], 'square has no blocked moves')
        else:
            require(blocked == case['blocked_steps'] and final['step'] == 211 and final['done'] is True
                    and final['hit'] == -2 and final['position'] == tuple(case['source_evaluation_only']),
                    'four blocked directions followed by exact found sentinel')
            require(actor.belief[final['position']] == 1. and np.count_nonzero(actor.belief) == 1, 'found point mass')
            before_counts = json.dumps(ledger.state['calls'], sort_keys=True)
            for operation in (actor.choose, lambda actor=actor, final=final: actor.update(0, final)):
                try:
                    operation()
                except RuntimeError:
                    pass
                else:
                    raise ValueError('post-found choice/update must reject before model or environment work')
            require(json.dumps(ledger.state['calls'], sort_keys=True) == before_counts, 'post-found guards invoke no work')
        require(not primary_recorder.records, 'prescribed primary actor performs zero policy forwards')
        row = {'case': case['id'], 'regime': case['regime'], 'initial_hit': case['initial_hit'],
               'native_steps': len(history), 'blocked_steps': blocked, 'censored': case['censored'],
               'found': final['done'], 'final_update_assimilated': True, 'all_beliefs_exact': True}
        rows.append(row)
        append(out / 'cases.jsonl', row)
    for name, expected in (('native_reset', 8), ('native_step', 446), ('tensorflow_value', 16),
                           ('actor_construction', 16), ('actor_update', 510)):
        require(ledger.state['calls'][name] == {'attempted': expected, 'returned': expected}, f'complete exact work: {name}')
    require(len(rows) == len(pairs) == 8 and ledger.state['pending_call'] is None, 'complete native fixture/prefix coverage')
    return {'status': 'completed', 'qualified': True, 'version': VERSION, 'scope': SCOPE,
            'cases': rows, 'policy_prefixes': pairs, 'native_steps': 446, 'native_resets': 8,
            'tensorflow_value_calls': 16, 'symmetry_average': True, 'numpy_port_admitted': False,
            'prior_numpy_action_failure_preserved': True, 'autonomous_episodes': 0, 'training_updates': 0}


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = self.launch = self.plan = self.paths = None
        self.ledger = Ledger(self.out, self.check)
        self.receipt = {'version': VERSION, 'status': 'started', 'scope': SCOPE, 'limits': LIMITS,
                        'work': self.ledger.state, 'autonomous_episodes': 0, 'training_updates': 0,
                        'numpy_port_admitted': False, 'prior_numpy_action_failure_preserved': True}

    def check(self):
        require(self.clock.now_ns() < self.launch['deadline_ns'], 'shared suspend-inclusive native deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        require(rss <= LIMITS['rss_bytes'], 'native qualification RSS cap')
        self.receipt['peak_rss_bytes'] = rss
        require(sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file()) <= LIMITS['output_bytes'], 'native qualification output cap')

    def bind(self):
        require(sha(ROOT / CLOCK) == CLOCK_PIN, 'immutable suspend clock')
        self.clock = load(ROOT / CLOCK, '_otto_released_native_clock').SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, 'native launch receipt missing')
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        command = list(self.launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and self.launch['pid'] == os.getpid()
                and self.launch['pgid'] == os.getpgrp() and self.launch['parent_pid'] == os.getppid(), 'actual native process identity')
        require(self.launch['version'] == 'dialogue-observation-supervision-v2' and self.launch['cap_seconds'] == 600
                and self.launch['clock_backend'] == self.clock.backend
                and self.launch['started_ns'] <= self.start < self.launch['deadline_ns']
                and self.launch['deadline_ns'] == self.launch['started_ns'] + 600 * 10**9
                and Path(self.launch['cwd']).resolve() == ROOT == Path.cwd().resolve(), 'actual bounded native parent')
        self.plan, self.paths = authenticate(self.args)
        require(self.launch['watchdog_sha256'] == self.plan['sources']['scripts/supervise_dialogue_observation_v2.py']
                and self.launch['clock_source_sha256'] == CLOCK_PIN, 'supervisor source pins')
        self.receipt.update(plan_sha256=self.args.plan_sha256, supervision_sha256=sha(self.args.supervision),
                            sources=self.plan['sources'], inputs=self.plan['inputs'])
        write(self.out / 'started.json', {'request': {k: str(v) for k, v in vars(self.args).items()},
                                        'clock_backend': self.clock.backend, 'started_ns': self.start, 'launch': self.launch})
        self.check()

    def body(self):
        require(not any(n in sys.modules for n in ('numpy', 'scipy', 'tensorflow', 'tf_keras', 'h5py')), 'clean native numerical import boundary')
        helper = load(ROOT / ENGINE, '_otto_native_qualification_helpers')
        os.environ.update(helper.ENVIRONMENT)
        self.ledger.context('numerical_imports')
        import numpy as np
        import tensorflow as tf

        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        keras = {'floatx': tf.keras.backend.floatx(), 'image_data_format': tf.keras.backend.image_data_format(),
                 'mixed_precision_policy': tf.keras.mixed_precision.global_policy().name}
        require(not tf.config.get_visible_devices('GPU') and tf.keras.Model.__module__.startswith('tf_keras.')
                and keras == {'floatx': 'float32', 'image_data_format': 'channels_last', 'mixed_precision_policy': 'float32'},
                'original CPU float32 legacy-Keras configuration')
        package_name = '_otto_released_native_upstream'
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT / UPSTREAM)]
        sys.modules[package_name] = package
        load(ROOT / UPSTREAM / 'policy.py', f'{package_name}.policy')
        RLPolicy = load(ROOT / UPSTREAM / 'rlpolicy.py', f'{package_name}.rlpolicy').RLPolicy
        ValueModel = load(ROOT / UPSTREAM / 'valuemodel.py', f'{package_name}.valuemodel').ValueModel
        SourceTracking = load(ROOT / UPSTREAM / 'sourcetracking.py', f'{package_name}.sourcetracking').SourceTracking
        Actor = load(ROOT / ACTOR, '_otto_released_native_actor').ReleasedPolicyActor
        public = load(ROOT / PUBLIC, '_otto_released_native_public')
        write(self.out / 'runtime.json', {'python': sys.version, 'executable': sys.executable,
              'all_distributions': self.plan['runtime']['all_distributions'], 'environment': helper.ENVIRONMENT,
              'keras_configuration': keras, 'keras_module': tf.keras.Model.__module__,
              'visible_devices': [str(d) for d in tf.config.get_visible_devices()]})
        self.ledger.context('original_model_load')
        model = self.ledger.call('tensorflow_construction', lambda: ValueModel(**helper.MODEL_CONFIGURATION))
        self.ledger.call('tensorflow_build', lambda: model.build_graph(input_shape_nobatch=(105, 105)))
        self.ledger.call('tensorflow_load', lambda: model.load_weights(str(self.paths['weights'])))
        metadata = read(self.paths['tensor_metadata'])
        require(metadata['configuration'] == helper.MODEL_CONFIGURATION and metadata['parameters'] == 13390849,
                'unchanged extracted model configuration')
        keys = [f'{kind}_{i}' for i in range(4) for kind in ('kernel', 'bias')]
        require(len(model.FC_block) == 3 and len(model.weights) == 8, 'exact original layer/variable count')
        weights = []
        with np.load(self.paths['tensors'], allow_pickle=False) as archive:
            require(set(archive.files) == set(keys), 'all eight extracted tensors')
            for i, layer in enumerate([*model.FC_block, model.densefinal]):
                arrays = layer.get_weights()
                require(len(arrays) == 2, 'ordered original kernel/bias')
                for kind, actual in zip(('kernel', 'bias'), arrays, strict=True):
                    key = f'{kind}_{i}'
                    expected, record = archive[key], metadata['datasets'][key]
                    require(hashlib.sha256(expected.tobytes()).hexdigest() == record['sha256_c_order']
                            and list(expected.shape) == record['shape'] and expected.dtype.str == record['dtype'], 'extracted tensor metadata bytes')
                    row = {'id': key, 'exact': bool(array_identity(actual, expected, np)), 'shape': list(actual.shape),
                           'sha256': hashlib.sha256(actual.tobytes()).hexdigest()}
                    append(self.out / 'weights.jsonl', row)
                    weights.append(row)
        require(all(r['exact'] for r in weights), 'exact original HDF5/tensor identity before first forward')
        kernels = {}
        for regime in ('base', 'shift'):
            with np.load(self.paths[f'{regime}_kernel'], allow_pickle=False) as archive:
                require(set(archive.files) == {'likelihood', 'initial_hit_weights'}, 'inherited kernel archive identity')
                kernels[regime] = archive['likelihood']
        result = mechanical_run(SourceTracking, public.seeded_environment, public.observation,
                                Actor, RLPolicy, model, helper, kernels, self.ledger, self.out, np)
        for name in ('tensorflow_construction', 'tensorflow_build', 'tensorflow_load'):
            require(self.ledger.state['calls'][name] == {'attempted': 1, 'returned': 1}, 'single unchanged original model load')
        result['weight_identity'] = True
        write(self.out / 'summary.json', result)
        return result

    def execute(self):
        require(self.out.is_absolute(), 'absolute exclusive native output')
        self.out.mkdir(parents=True, exist_ok=False)
        try:
            self.bind()
            result = self.body()
            plan, paths = authenticate(self.args)
            require(plan == self.plan and paths == self.paths and sha(self.args.supervision) == self.receipt['supervision_sha256'],
                    'unchanged complete native qualification inputs')
            self.check()
            self.receipt.update(status='completed', qualified=result['qualified'], clock_backend=self.clock.backend,
                                started_ns=self.start, finished_ns=self.clock.now_ns(),
                                files={p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()})
            self.receipt['wall_seconds'] = (self.receipt['finished_ns'] - self.start) / 1e9
            write(self.out / 'receipt.json', self.receipt)
            self.check()
            print(json.dumps({'status': 'completed', 'qualified': result['qualified'], 'receipt_sha256': sha(self.out / 'receipt.json')}))
            return self.receipt
        except BaseException as error:
            self.receipt.update(status='failed', qualified=False, error=repr(error), traceback=traceback.format_exc())
            if self.clock is not None and self.start is not None:
                try:
                    self.receipt['failure_elapsed_seconds'] = (self.clock.now_ns() - self.start) / 1e9
                except BaseException as secondary:  # noqa: BLE001 - preserve the primary error.
                    self.receipt['failure_clock_error'] = repr(secondary)
            try:
                write(self.out / ('failed.json' if (self.out / 'receipt.json').exists() else 'receipt.json'), self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - preserve the primary error.
                error.add_note(f'Failure receipt publication failed: {secondary!r}')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'supervision', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    return Run(parser.parse_args()).execute()


if __name__ == '__main__':
    main()
