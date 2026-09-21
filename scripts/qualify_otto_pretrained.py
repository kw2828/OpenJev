"""Prospective synthetic parity of the released OTTO value model and policy.

Uses the original benchmark ValueModel/RLPolicy classes under a separately
qualified modern legacy-Keras runtime. No SourceTracking object, environment
step, training, autonomous episode or current scientific outcome is accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import resource
import shutil
import sys
import time
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'otto-pretrained-qualification-v1'
CLOCK_PATH = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
UPSTREAM = 'tmp/otto-source-review-01/isotropic/classes'
PORT = 'src/openjev/research/otto_pretrained_value.py'
MODEL_CONFIGURATION = {'Ndim': 2, 'FC_layers': 3, 'FC_units': 1024,
                       'regularization_factor': 0.0, 'loss_function': 'mean_squared_error'}
POSITIONS = ((26, 26), (0, 0), (0, 26), (52, 52))
PATTERNS = ('uniform', 'adjacent_point', 'asymmetric_pair', 'seeded_dense')
FLOOR_MASSES = (0., .5e-10, 1e-10, 2e-10)
CONFIGURATION = {'atol': 1e-4, 'rtol': 1e-5, 'raw_value_inputs': 16,
                 'raw_symmetry_settings': [False, True], 'raw_batch_sizes': [1, 3, 16],
                 'regimes': ['base', 'shift'], 'positions': [list(p) for p in POSITIONS],
                 'belief_patterns': list(PATTERNS), 'dense_seed_base': 830001,
                 'physical_policy_fixtures': 32, 'mechanical_floor_masses': list(FLOOR_MASSES),
                 'policy_symmetry_average': True, 'exact_action_agreement': True,
                 'policy_tie_epsilon': 1e-10, 'model_configuration': MODEL_CONFIGURATION}
LIMITS = {'native_seconds': 600, 'rss_bytes': 4 * 1024**3, 'output_bytes': 256 * 1024**2, 'simulator_calls': 0}
VERSIONS = {'tensorflow': '2.20.0', 'tf-keras': '2.20.1', 'numpy': '2.2.6', 'h5py': '3.14.0'}
ENVIRONMENT = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
                                    'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')}
ENVIRONMENT.update(TF_USE_LEGACY_KERAS='1', CUDA_VISIBLE_DEVICES='-1')
ROLES = {'original_weights', 'original_config', 'extraction_receipt', 'tensors', 'tensor_metadata',
         'base_kernel', 'shift_kernel', 'kernel_receipt'}
REQUIRED_SOURCES = {'scripts/qualify_otto_pretrained.py', 'tests/test_qualify_otto_pretrained.py', PORT,
                    'research/otto-pretrained-reference-protocol.md',
                    'tests/test_otto_pretrained_value.py', CLOCK_PATH, 'scripts/supervise_dialogue_observation_v2.py',
                    *(f'{UPSTREAM}/{name}.py' for name in ('valuemodel', 'rlpolicy', 'policy', 'sourcetracking'))}
SCOPE = ('Synthetic value and published-policy algebra parity under the pinned modern TensorFlow/legacy-Keras runtime. '
         'No original TensorFlow2.8 reproduction, public-adapter/native episode qualification, training, or efficacy claim. '
         'Original-policy model inputs and local branch masses are observed with a direct-caller-code-checked wrapper.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


class WorkLedger:
    """Journal every expensive call before entry and after synchronized return."""
    CHANNELS = ('tensorflow_construction', 'numpy_construction', 'tensorflow_build',
                'tensorflow_load', 'tensorflow_value', 'numpy_value')

    def __init__(self, out):
        self.path = out / 'work-ledger.jsonl'
        self.state = {'phase': 'authentication', 'case': None, 'pending_call': None,
                      'calls': {name: {'attempted': 0, 'returned': 0} for name in self.CHANNELS},
                      'completed_comparisons': {'weight': 0, 'value': 0, 'policy': 0}}

    def context(self, phase, case=None):
        require(self.state['pending_call'] is None, 'no context change across a pending call')
        self.state.update(phase=phase, case=case)
        append(self.path, {'event': 'context', **self.state})

    def call(self, channel, operation):
        require(channel in self.CHANNELS and self.state['pending_call'] is None, 'one declared pending call')
        counts = self.state['calls'][channel]
        counts['attempted'] += 1
        self.state['pending_call'] = {'channel': channel, 'ordinal': counts['attempted'],
                                      'phase': self.state['phase'], 'case': self.state['case']}
        append(self.path, {'event': 'call_attempt', **self.state})
        result = operation()
        counts['returned'] += 1
        self.state['pending_call'] = None
        append(self.path, {'event': 'call_return', **self.state})
        return result

    def completed(self, kind, record):
        require(kind in self.state['completed_comparisons'], 'declared comparison kind')
        append(self.path.parent / f'{kind}-checks.jsonl', record)
        self.state['completed_comparisons'][kind] += 1
        append(self.path, {'event': 'comparison_saved', **self.state})


def descriptor(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def regular(relative):
    path = Path(relative)
    require(not path.is_absolute() and path.parts and '..' not in path.parts, 'contained relative input path')
    path = ROOT / path
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
            and path.resolve().is_relative_to(ROOT), 'regular local input')
    return path


def authenticate(args):
    """All source/artifact bytes authenticate before numerical/framework imports."""
    require(args.plan.is_absolute() and not args.plan.is_symlink() and sha(args.plan) == args.plan_sha256, 'external plan pin')
    plan = read(args.plan)
    require(plan['version'] == VERSION and plan['status'] == 'frozen_before_run'
            and plan['configuration'] == CONFIGURATION and plan['limits'] == LIMITS, 'frozen qualification definition')
    require(REQUIRED_SOURCES <= plan['sources'].keys() and plan['sources'][CLOCK_PATH] == CLOCK_PIN, 'complete source membership')
    for name, pin in plan['sources'].items():
        require(sha(regular(name)) == pin, f'source pin {name}')
    require(set(plan['inputs']) == ROLES, 'exact qualification artifact roles')
    paths = {}
    for role, item in plan['inputs'].items():
        path = regular(item['path'])
        require(descriptor(path) == {'bytes': item['bytes'], 'sha256': item['sha256']}, f'artifact pin {role}')
        paths[role] = path
    require(plan['inputs']['original_weights']['sha256'] == '1efb73aa38e0fd8b08d6d03059c0db4664da3afb8eaff1d7ab9b363f8e7ad37d'
            and plan['inputs']['original_config']['sha256'] == 'ca12567f2e0333192a7a0b0c177d519bdae60749ddb8005a1456608fd6782b4c',
            'official zoo weights and statically inspected configuration')
    extraction = read(paths['extraction_receipt'])
    require(extraction['version'] == 'otto-pretrained-extraction-v1' and extraction['status'] == 'completed'
            and extraction['model_calls'] == extraction['simulator_calls'] == 0
            and extraction['inference_qualified'] is False, 'completed extraction only')
    require(not (paths['extraction_receipt'].parent / 'failed.json').exists(), 'no demoted extraction')
    for name, record in extraction['files'].items():
        path = paths['extraction_receipt'].parent / name
        require(Path(name).name == name and not path.is_symlink() and descriptor(path) == record, 'complete extraction payload closure')
    for role, name in (('tensors', 'tensors.npz'), ('tensor_metadata', 'tensor-metadata.json')):
        require(paths[role] == paths['extraction_receipt'].parent / name and descriptor(paths[role]) == extraction['files'][name], 'extraction artifact join')
    for role, name in (('original_weights', 'zoo_model_2_3_2'), ('original_config', 'zoo_model_2_3_2.config')):
        require(extraction['inputs']['assets'][name]['sha256'] == plan['inputs'][role]['sha256'], 'extracted original artifact join')
    closed = read(paths['kernel_receipt'])
    require(closed['status'] == 'completed' and closed['completed_episodes'] == 1152, 'completed prior native kernel producer')
    for regime in ('base', 'shift'):
        path = paths[f'{regime}_kernel']
        require(path == paths['kernel_receipt'].parent / f'public-kernel-{regime}.npz'
                and descriptor(path) == closed['files'][path.name], 'closed public-kernel bytes')
    runtime = plan['runtime']
    require(runtime['versions'] == VERSIONS and sys.version.split()[0] == runtime['python_version']
            and str(Path(sys.executable).absolute()) == runtime['python_executable'], 'isolated reference Python identity')
    for name, version in VERSIONS.items():
        require(importlib.metadata.version(name) == version, f'isolated package version {name}')
    return plan, paths


def moved(position, action):
    result = list(position)
    axis, direction = action // 2, 2 * (action % 2) - 1
    possible = 0 <= result[axis] + direction < 53
    if possible:
        result[axis] += direction
    return result, possible


class SyntheticEnvironment:
    """Only public belief algebra; intentionally has no source, reward or step."""
    Nactions = 4
    Nhits = 4
    NN_input_shape = (105, 105)

    def __init__(self, belief, position, kernel, np):
        self.p_source, self.agent, self.p_Poisson, self.np = belief, list(position), kernel, np

    def _move(self, action, position):
        return moved(position, action)

    def _extract_N_from_2N(self, input, origin):
        start = [53 - coordinate for coordinate in origin]
        return input[..., start[0]:start[0] + 53, start[1]:start[1] + 53]

    def _centeragent(self, p, agent):
        return self.np.pad(p, ((52 - agent[0], agent[0]), (52 - agent[1], agent[1])), mode='constant')


def d4(array, np):
    """Independent explicit dihedral order, without the port's helper."""
    return np.stack((array, array.T, array[::-1, :], array.T[::-1, :],
                     array[::-1, ::-1], array.T[::-1, ::-1], array[:, ::-1], array.T[:, ::-1]))


def raw_fixtures(np):
    zero = np.zeros((105, 105), dtype=np.float32)
    sub = zero.copy()
    sub[11, 72] = .5
    point = zero.copy()
    point[52, 52] = 1
    uniform = np.full((105, 105), 1 / 11025, dtype=np.float32)
    pair = zero.copy()
    pair[4, 23], pair[70, 91] = .3, .7
    dense = np.random.default_rng(830099).random((105, 105)).astype(np.float32)
    dense /= dense.sum(dtype=np.float64)
    dense *= .25
    asymmetric = zero.copy()
    asymmetric[4, 8], asymmetric[15, 34], asymmetric[39, 22] = .1, .2, .7
    edge = zero.copy()
    edge[0, 104] = 1
    checker = (np.indices((105, 105)).sum(axis=0) % 2).astype(np.float32)
    checker /= checker.sum(dtype=np.float64)
    names = ['zero', 'subnormalized_point', 'center_point', 'uniform', 'asymmetric_pair', 'subnormalized_dense']
    names += [f'd4_asymmetric_{i}' for i in range(8)] + ['corner_point', 'checkerboard']
    return names, np.stack([zero, sub, point, uniform, pair, dense, *d4(asymmetric, np), edge, checker])


def policy_fixtures(kernels, np):
    cases = []
    for regime in ('base', 'shift'):
        for index, position in enumerate(POSITIONS):
            for pattern in PATTERNS:
                belief = np.zeros((53, 53), dtype=np.float64)
                if pattern == 'uniform':
                    belief.fill(1)
                    belief[position] = 0
                    belief /= belief.sum()
                elif pattern == 'adjacent_point':
                    target = next(moved(position, a)[0] for a in range(4) if moved(position, a)[1])
                    belief[tuple(target)] = 1
                elif pattern == 'asymmetric_pair':
                    belief[5, 11], belief[45, 39] = .3, .7
                else:
                    belief = np.random.default_rng(830001 + index).random((53, 53))
                    belief[position] = 0
                    belief /= belief.sum()
                cases.append({'id': f'{regime}.position{index}.{pattern}', 'kind': 'physical', 'regime': regime,
                              'position': position, 'belief': belief, 'kernel': kernels[regime]})
    for index, mass in enumerate(FLOOR_MASSES):
        kernel = np.empty((4, 107, 107), dtype=np.float64)
        for category, value in enumerate((mass, .2, .3, .5 - mass)):
            kernel[category].fill(value)
        kernel[:, 53, 53] = 0
        belief = np.zeros((53, 53), dtype=np.float64)
        belief[40, 40] = 1
        cases.append({'id': f'mechanical.mass{index}', 'kind': 'mechanical_floor', 'regime': None,
                      'position': (26, 26), 'belief': belief, 'kernel': kernel, 'mass': mass})
    return cases


def compare_arrays(actual, reference, np, *, exact=False):
    """Return all mismatch evidence; ordinary numeric failures do not abort cases."""
    shape_ok = actual.shape == reference.shape
    dtype_ok = actual.dtype == reference.dtype
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    if not shape_ok or not finite:
        return {'passed': False, 'shape_equal': shape_ok, 'dtype_equal': dtype_ok, 'finite': finite,
                'compared_scalars': 0, 'mismatched_scalars': None, 'maximum_absolute_error': None}
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    bound = np.zeros_like(delta) if exact else CONFIGURATION['atol'] + CONFIGURATION['rtol'] * np.abs(reference.astype(np.float64))
    mismatch = int(np.count_nonzero(delta > bound))
    byte_equal = dtype_ok and actual.tobytes(order='C') == reference.tobytes(order='C')
    return {'passed': dtype_ok and mismatch == 0 and (not exact or byte_equal),
            'byte_equal': byte_equal if exact else None, 'shape_equal': True, 'dtype_equal': dtype_ok, 'finite': True,
            'compared_scalars': int(delta.size), 'mismatched_scalars': mismatch,
            'maximum_absolute_error': float(delta.max(initial=0)),
            'maximum_tolerance_ratio': float(np.max(np.divide(delta, bound, out=np.zeros_like(delta), where=bound > 0), initial=0))}


def import_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RecordingModel:
    def __init__(self, model, expected_code, np, work=None):
        self.model, self.expected_code, self.np = model, expected_code, np
        self.work = work
        self.records = []

    def __call__(self, inputs, *, sym_avg):
        frame = inspect.currentframe()
        try:
            caller = frame.f_back
            require(caller.f_code is self.expected_code, 'original RLPolicy direct caller identity')
            masses = caller.f_locals['probs']
            require(isinstance(masses, self.np.ndarray) and masses.shape == (4, 4) and masses.dtype == self.np.float32,
                    'observed original policy branch masses')
            observed_inputs = self.np.asarray(inputs.numpy()).copy()
            def forward():
                values = self.model(inputs, training=False, sym_avg=sym_avg)
                return values, self.np.asarray(values.numpy()).copy()
            values, observed_values = forward() if self.work is None else self.work.call('tensorflow_value', forward)
            self.records.append({'inputs': observed_inputs, 'masses': masses.copy(), 'values': observed_values})
            return values
        finally:
            del frame


class RecordingNumpyModel:
    def __init__(self, model, work=None):
        self.model, self.records = model, []
        self.work = work

    def predict_centered(self, inputs, *, sym_avg):
        def forward():
            return self.model.predict_centered(inputs, sym_avg=sym_avg)
        values = forward() if self.work is None else self.work.call('numpy_value', forward)
        self.records.append({'inputs': inputs.copy(), 'values': values.copy()})
        return values


def run_comparisons(out, paths, check, work):
    work.context('numerical_imports')
    import numpy as np
    import tensorflow as tf

    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    require(not tf.config.get_visible_devices('GPU') and tf.keras.Model.__module__.startswith('tf_keras.'),
            'CPU-only legacy Keras actually active')
    keras_configuration = {'floatx': tf.keras.backend.floatx(),
                           'image_data_format': tf.keras.backend.image_data_format(),
                           'mixed_precision_policy': tf.keras.mixed_precision.global_policy().name}
    require(keras_configuration == {'floatx': 'float32', 'image_data_format': 'channels_last',
                                     'mixed_precision_policy': 'float32'},
            'unchanged float32 row-major Keras configuration')
    write(out / 'runtime.json', {'python': sys.version, 'executable': sys.executable,
          'versions': {k: importlib.metadata.version(k) for k in VERSIONS}, 'environment': dict(ENVIRONMENT),
          'keras_model_module': tf.keras.Model.__module__, 'physical_devices': [str(d) for d in tf.config.list_physical_devices()],
          'keras_configuration': keras_configuration,
          'visible_devices': [str(d) for d in tf.config.get_visible_devices()],
          'intra_threads': tf.config.threading.get_intra_op_parallelism_threads(),
          'inter_threads': tf.config.threading.get_inter_op_parallelism_threads()})
    port = import_source(ROOT / PORT, '_otto_qualified_numpy_port')
    package_name = '_otto_qualification_upstream'
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / UPSTREAM)]
    sys.modules[package_name] = package
    import_source(ROOT / UPSTREAM / 'policy.py', f'{package_name}.policy')
    reference_policy = import_source(ROOT / UPSTREAM / 'rlpolicy.py', f'{package_name}.rlpolicy')
    reference_value = import_source(ROOT / UPSTREAM / 'valuemodel.py', f'{package_name}.valuemodel')
    with np.load(paths['tensors'], allow_pickle=False) as archive:
        require(set(archive.files) == set(port.KEYS), 'complete extracted tensor archive')
        tensors = {name: archive[name] for name in archive.files}
    metadata = read(paths['tensor_metadata'])
    require(metadata['configuration'] == MODEL_CONFIGURATION and metadata['parameters'] == 13390849
            and metadata['tensor_bytes'] == 53563396, 'actual extraction geometry')
    for name, tensor in tensors.items():
        item = metadata['datasets'][name]
        require(list(tensor.shape) == item['shape'] and tensor.dtype.str == item['dtype']
                and tensor.nbytes == item['bytes']
                and hashlib.sha256(tensor.tobytes(order='C')).hexdigest() == item['sha256_c_order'],
                f'exact extracted array {name}')
    work.context('model_construction')
    numpy_model = work.call('numpy_construction', lambda: port.PretrainedValue(tensors))
    check()
    # Only a byte-identical filename adaptation, never model conversion.
    h5_copy = out / 'original.weights-legacy.h5'
    with paths['original_weights'].open('rb') as source, h5_copy.open('xb') as destination:
        shutil.copyfileobj(source, destination)
    require(sha(h5_copy) == sha(paths['original_weights']), 'unchanged original HDF5 copy')
    model = work.call('tensorflow_construction', lambda: reference_value.ValueModel(**MODEL_CONFIGURATION))
    work.call('tensorflow_build', lambda: model.build_graph(input_shape_nobatch=(105, 105)))
    work.call('tensorflow_load', lambda: model.load_weights(str(h5_copy)))
    require(len(model.FC_block) == 3 and len(model.weights) == 8, 'all original Keras variables actually built/loaded')
    checks = []
    for index, layer in enumerate([*model.FC_block, model.densefinal]):
        weights = layer.get_weights()
        require(len(weights) == 2, 'one kernel and bias per original Dense layer')
        for kind, actual in zip(('kernel', 'bias'), weights, strict=True):
            key = f'{kind}_{index}'
            work.context('weight_identity', key)
            record = compare_arrays(actual, tensors[key], np, exact=True)
            record.update(kind='weight', id=key, keras_layer_name=layer.name,
                          keras_array_sha256=hashlib.sha256(actual.tobytes(order='C')).hexdigest(),
                          extracted_array_sha256=hashlib.sha256(tensors[key].tobytes(order='C')).hexdigest())
            checks.append(record)
            work.completed('weight', record)
    write(out / 'weight-checks.json', checks)
    names, raw = raw_fixtures(np)
    with (out / 'value-inputs.npz').open('xb') as stream:
        np.savez(stream, inputs=raw)
    value_records = []
    for sym_avg in (False, True):
        for batch_size in (1, 3, 16):
            for offset in range(0, len(raw), batch_size):
                check()
                inputs = raw[offset:offset + batch_size]
                work.context('raw_value', {'sym_avg': sym_avg, 'batch_size': batch_size, 'offset': offset})
                expected = work.call('tensorflow_value', lambda inputs=inputs, sym_avg=sym_avg: np.asarray(
                    model(tf.convert_to_tensor(inputs), training=False, sym_avg=sym_avg).numpy()))
                actual = work.call('numpy_value', lambda inputs=inputs, sym_avg=sym_avg:
                                   numpy_model.predict_centered(inputs, sym_avg=sym_avg))
                record = compare_arrays(actual, expected, np)
                record.update(kind='value', sym_avg=sym_avg, batch_size=batch_size, offset=offset,
                              ids=names[offset:offset + batch_size], tensorflow=expected.reshape(-1).tolist(),
                              numpy=actual.reshape(-1).tolist())
                value_records.append(record)
                work.completed('value', record)
    write(out / 'value-checks.json', value_records)
    kernels = {}
    for regime in ('base', 'shift'):
        with np.load(paths[f'{regime}_kernel'], allow_pickle=False) as archive:
            require(set(archive.files) == {'likelihood', 'initial_hit_weights'}, 'public frozen likelihood archive')
            kernels[regime] = archive['likelihood']
        require(kernels[regime].dtype == np.float64 and kernels[regime].shape == (4, 107, 107)
                and np.isfinite(kernels[regime]).all() and (kernels[regime] >= 0).all(), 'frozen physical kernel geometry')
    policy_records, saved = [], {}
    cases = policy_fixtures(kernels, np)
    require(len(cases) == 36, 'all predefined physical and mechanical policy fixtures')
    for index, case in enumerate(cases):
        check()
        work.context('policy', case['id'])
        env = SyntheticEnvironment(case['belief'], case['position'], case['kernel'], np)
        recorder = RecordingModel(model, reference_policy.RLPolicy._value_policy.__code__, np, work)
        policy = reference_policy.RLPolicy(env, recorder, sym_avg=True)
        expected_action, expected_scores = policy._value_policy()
        require(len(recorder.records) == 1, 'exactly one original policy model call')
        observed = recorder.records[0]
        inputs, masses = port.policy_inputs(case['belief'], case['position'], case['kernel'])
        np_recorder = RecordingNumpyModel(numpy_model, work)
        actual_action, actual_scores = port.value_policy(np_recorder, case['belief'], case['position'], case['kernel'], sym_avg=True)
        require(len(np_recorder.records) == 1, 'exactly one NumPy policy model call')
        generated = np_recorder.records[0]
        pieces = {'inputs': compare_arrays(inputs, observed['inputs'], np, exact=True),
                  'masses': compare_arrays(masses, observed['masses'], np, exact=True),
                  'numpy_policy_inputs': compare_arrays(generated['inputs'], inputs, np, exact=True),
                  'values': compare_arrays(generated['values'], observed['values'], np),
                  'scores': compare_arrays(actual_scores, expected_scores, np)}
        scores = sorted(float(v) for v in expected_scores)
        record = {'id': case['id'], 'kind': case['kind'], 'regime': case['regime'], 'position': list(case['position']),
                  'mechanical_mass': case.get('mass'), 'comparisons': pieces,
                  'action_equal': int(actual_action) == int(expected_action),
                  'tensorflow_action': int(expected_action), 'numpy_action': int(actual_action),
                  'tensorflow_scores': expected_scores.tolist(), 'numpy_scores': actual_scores.tolist(),
                  'tensorflow_values': observed['values'].reshape(-1).tolist(), 'numpy_values': generated['values'].reshape(-1).tolist(),
                  'tensorflow_best_second_gap': scores[1] - scores[0],
                  'tensorflow_tie_actions': np.flatnonzero(np.abs(expected_scores - expected_scores.min()) < 1e-10).tolist(),
                  'numpy_tie_actions': np.flatnonzero(np.abs(actual_scores - actual_scores.min()) < 1e-10).tolist()}
        record['passed'] = record['action_equal'] and all(piece['passed'] for piece in pieces.values())
        policy_records.append(record)
        work.completed('policy', record)
        prefix = f'case_{index:02d}'
        saved.update({f'{prefix}_belief': case['belief'], f'{prefix}_position': np.array(case['position'], dtype=np.int64),
                      f'{prefix}_tensorflow_inputs': observed['inputs'], f'{prefix}_numpy_inputs': inputs,
                      f'{prefix}_tensorflow_masses': observed['masses'], f'{prefix}_numpy_masses': masses})
        if case['kind'] == 'mechanical_floor':
            saved[f'{prefix}_kernel'] = case['kernel']
    with (out / 'policy-arrays.npz').open('xb') as stream:
        np.savez(stream, **saved)
    write(out / 'policy-checks.json', policy_records)
    work.context('summarization')
    result = summarize(checks, value_records, policy_records)
    result.update(version=VERSION, scope=SCOPE, raw_input_ids=names, simulator_calls=0, training_updates=0,
                  model_build_calls=1, tensorflow_value_calls=len(value_records) + len(policy_records),
                  numpy_value_calls=len(value_records) + len(policy_records),
                  source_tracking_objects_constructed=0, native_actor_integration_qualified=False,
                  original_tensorflow_2_8_reproduced=False, numpy_storage=numpy_model.storage_bytes())
    write(out / 'summary.json', result)
    check()
    return result


def summarize(weights, values, policies):
    require(len(weights) == 8 and len(values) == 46 and len(policies) == 36, 'complete qualification comparison counts')
    require([row['id'] for row in weights] == [f'{kind}_{i}' for i in range(4) for kind in ('kernel', 'bias')],
            'complete ordered original weight identities')
    value_ids = ['zero', 'subnormalized_point', 'center_point', 'uniform', 'asymmetric_pair', 'subnormalized_dense']
    value_ids += [f'd4_asymmetric_{i}' for i in range(8)] + ['corner_point', 'checkerboard']
    expected_routes = [(sym, size, offset, value_ids[offset:offset + size]) for sym in (False, True)
                       for size in (1, 3, 16) for offset in range(0, 16, size)]
    require([(r['sym_avg'], r['batch_size'], r['offset'], r['ids']) for r in values] == expected_routes,
            'all raw inputs in every fixed symmetry/batch route')
    policy_ids = [f'{regime}.position{i}.{pattern}' for regime in ('base', 'shift')
                  for i in range(4) for pattern in PATTERNS] + [f'mechanical.mass{i}' for i in range(4)]
    require([row['id'] for row in policies] == policy_ids, 'all fixed ordered policy identities')
    require(sum(len(row['ids']) for row in values) == 96, 'all sixteen values in every batch/symmetry route')
    require(sum(row['kind'] == 'physical' for row in policies) == 32
            and sum(row['kind'] == 'mechanical_floor' for row in policies) == 4, 'physical/mechanical case coverage')
    weight_ok = all(row['passed'] for row in weights)
    input_ok = all(row['comparisons'][name]['passed'] for row in policies for name in ('inputs', 'masses', 'numpy_policy_inputs'))
    value_ok = all(row['passed'] for row in values) and all(row['comparisons']['values']['passed'] for row in policies)
    cost_ok = all(row['comparisons']['scores']['passed'] for row in policies)
    action_ok = all(row['action_equal'] for row in policies)
    return {'status': 'completed', 'weight_identity': weight_ok, 'policy_input_identity': input_ok,
            'value_parity': value_ok, 'policy_cost_parity': cost_ok, 'policy_action_parity': action_ok,
            'qualified': weight_ok and input_ok and value_ok and cost_ok and action_ok,
            'weights_compared': len(weights), 'value_batches_compared': len(values),
            'raw_value_predictions_compared': 96, 'policy_fixtures_compared': len(policies),
            'action_disagreements': [row['id'] for row in policies if not row['action_equal']],
            'failed_value_batches': [{'sym_avg': row['sym_avg'], 'batch_size': row['batch_size'], 'offset': row['offset']}
                                     for row in values if not row['passed']],
            'failed_policy_fixtures': [row['id'] for row in policies if not row['passed']],
            'comparison_rule': 'abs(actual-reference)<=1e-4+1e-5*abs(reference); exact first action under<1e-10 ties',
            'near_tie_exemptions': False}


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = self.launch = None
        self.plan = self.paths = None
        self.work = WorkLedger(self.out)
        self.receipt = {'status': 'started', 'version': VERSION, 'scope': SCOPE, 'limits': LIMITS,
                        'simulator_calls': 0, 'training_updates': 0, 'native_actor_integration_qualified': False,
                        'work': self.work.state}

    def check(self):
        require(self.clock.now_ns() < self.launch['deadline_ns'], 'shared suspend-inclusive deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        require(rss <= LIMITS['rss_bytes'], 'qualification RSS cap')
        self.receipt['peak_rss_bytes'] = rss
        require(sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file()) <= LIMITS['output_bytes'], 'qualification output cap')

    def bind(self):
        require(sha(ROOT / CLOCK_PATH) == CLOCK_PIN, 'frozen suspend clock')
        clock_module = import_source(ROOT / CLOCK_PATH, '_otto_pretrained_clock')
        self.clock = clock_module.SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, 'supervisor launch receipt missing')
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        command = list(self.launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and self.launch['pid'] == os.getpid()
                and self.launch['pgid'] == os.getpgrp() and self.launch['parent_pid'] == os.getppid(), 'actual supervised process identity')
        require(self.launch['version'] == 'dialogue-observation-supervision-v2'
                and self.launch['clock_backend'] == self.clock.backend and self.launch['cap_seconds'] == 600
                and self.launch['started_ns'] <= self.start < self.launch['deadline_ns']
                and self.launch['deadline_ns'] == self.launch['started_ns'] + 600 * 10**9
                and Path(self.launch['cwd']).resolve() == ROOT == Path.cwd().resolve(), 'bounded native parent clock and working directory')
        self.plan, self.paths = authenticate(self.args)
        require(self.launch['watchdog_sha256'] == self.plan['sources']['scripts/supervise_dialogue_observation_v2.py']
                and self.launch['clock_source_sha256'] == CLOCK_PIN, 'supervisor source identity')
        self.receipt.update(plan_sha256=self.args.plan_sha256, supervision_sha256=sha(self.args.supervision),
                            sources=self.plan['sources'], inputs=self.plan['inputs'])
        write(self.out / 'started.json', {'request': {k: str(v) for k, v in vars(self.args).items()},
              'clock_backend': self.clock.backend, 'started_ns': self.start, 'launch': self.launch})
        self.check()

    def execute(self):
        require(self.out.is_absolute(), 'absolute exclusive output path')
        self.out.mkdir(parents=True, exist_ok=False)
        try:
            self.bind()
            require(not any(name in sys.modules for name in ('numpy', 'tensorflow', 'tf_keras', 'h5py')), 'clean numerical import boundary')
            os.environ.update(ENVIRONMENT)
            result = run_comparisons(self.out, self.paths, self.check, self.work)
            plan, paths = authenticate(self.args)
            require(plan == self.plan and paths == self.paths
                    and sha(self.args.supervision) == self.receipt['supervision_sha256'], 'unchanged complete qualification inputs')
            self.check()
            self.receipt.update(status='completed', qualified=result['qualified'],
                                clock_backend=self.clock.backend, started_ns=self.start, finished_ns=self.clock.now_ns(),
                                files={p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()})
            self.receipt['wall_seconds'] = (self.receipt['finished_ns'] - self.start) / 1e9
            write(self.out / 'receipt.json', self.receipt)
            self.check()
            print(json.dumps({'status': 'completed', 'qualified': result['qualified'],
                              'receipt_sha256': sha(self.out / 'receipt.json')}))
            return self.receipt
        except BaseException as error:
            self.receipt.update(status='failed', qualified=False, error=repr(error), traceback=traceback.format_exc())
            if self.clock is not None and self.start is not None:
                try:
                    self.receipt['failure_elapsed_seconds'] = (self.clock.now_ns() - self.start) / 1e9
                except BaseException as timing_error:  # noqa: BLE001 - preserve failure evidence if the clock breaks.
                    self.receipt['failure_clock_error'] = repr(timing_error)
            try:
                write(self.out / ('failed.json' if (self.out / 'receipt.json').exists() else 'receipt.json'), self.receipt)
            except BaseException as publication_error:  # noqa: BLE001 - keep the original qualification failure.
                error.add_note(f'Failure receipt publication failed: {publication_error!r}')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'supervision', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    return Run(parser.parse_args()).execute()


if __name__ == '__main__':
    main()
