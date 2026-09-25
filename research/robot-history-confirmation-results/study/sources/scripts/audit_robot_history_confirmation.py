"""Independent fixed-checkpoint confirmation audit, with qualified model replay.

Only the two registered confirmation MAT recordings may be decoded, and only
after the original producer closes. No selection, fitting or optimizer calls.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'robot-history-confirmation-audit-v1'
PRIMARY = ('last_two', 'local_affine', 'temporal_affine')
ARMS = (*PRIMARY, 'dense_bounded', 'dense_unbounded', 'gru32', 'legacy_instant', 'gru10')
REFS = ('causal_ridge_1', 'causal_ridge_100', 'linear_frozen', 'persistence')
SEEDS = (8101, 8102, 8103)
FIXED_RATES = dict(zip(ARMS, (.001, .003, .003, .001, .003, .003, .001, .003), strict=True))
CONFIRM = ('recording_2021_12_15_22H_41M.mat', 'recording_2021_12_15_22H_50M.mat')
PARAMETERS = dict(zip(ARMS, (590, 962, 962, 806, 806, 5916, 1014, 1296), strict=True))
CONDITIONS = ('primary_recipes_complete', 'equal_file_mean_5pct_vs_both_locals',
              'each_file_within_2pct_best_local', 'latency_within_125pct_last_two',
              'complete_frontier_not_dominated')
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
TIMING_SCOPE = ('batch1 normalization, casting, context32, future128, denormalization and finite check; '
                'no model/disk load; outer validation and operator preparation included; eager CPU')
STUDY_VERSION = 'robot-history-confirmation-v1'
PARENT_SHA = '628b038c910bf37246e939b104d70530f6783a7fbc67f79afeb79b8813738994'
ARCHIVE_SHA = '9011509cf901dcb50a8de1e4a5a0fcf6cb8bed2f64945ba0fbbfb8e0b54886fe'
SOURCES = ('src/openjev/research/joint_coupling.py', 'src/openjev/research/industrial_robot_data.py',
           'scripts/robot_coupling_study.py', 'tests/test_joint_coupling.py', 'tests/test_industrial_robot_data.py',
           'tests/test_robot_coupling_study.py', 'research/robot-coupling-protocol.md',
           'src/openjev/research/bounded_robot_transition.py', 'src/openjev/research/causal_robot_ridge.py',
           'scripts/robot_transition_study.py', 'tests/test_bounded_robot_transition.py', 'tests/test_causal_robot_ridge.py',
           'tests/test_robot_transition_study.py', 'research/robot-transition-protocol.md',
           'src/openjev/research/compact_robot_gate.py', 'tests/test_compact_robot_gate.py',
           'src/openjev/research/structured_robot_transition.py', 'tests/test_structured_robot_transition.py',
           'scripts/robot_structured_study.py', 'tests/test_robot_structured_study.py', 'research/robot-structured-protocol.md',
           'src/openjev/research/robot_history_initializer.py', 'tests/test_robot_history_initializer.py',
           'scripts/robot_history_initialization_study.py', 'tests/test_robot_history_initialization_study.py',
           'research/robot-history-initialization-protocol.md', 'scripts/plot_robot_structured.py',
           'scripts/audit_robot_structured.py', 'src/openjev/research/suspend_clock.py',
           'scripts/plot_robot_history_initialization.py', 'scripts/audit_robot_history_initialization.py',
           'scripts/robot_history_confirmation.py', 'tests/test_robot_history_confirmation.py',
           'research/robot-history-confirmation-protocol.md', 'scripts/audit_robot_history_confirmation.py',
           'tests/test_audit_robot_history_confirmation.py')
QUALIFICATION_SOURCES = ('scripts/robot_history_confirmation.py', 'scripts/audit_robot_history_confirmation.py',
                         'scripts/launch_robot_history_confirmation.py', 'tests/test_robot_history_confirmation.py',
                         'tests/test_audit_robot_history_confirmation.py')
COMMAND = ['.venv/bin/python', '-u', 'scripts/robot_history_confirmation.py', '--registration',
           'research/robot-history-confirmation-registration.json', '--output', 'output/robot-history-confirmation-v1']
COMMON_PAYLOADS = ('normalizers.npz', 'linear.npz', 'causal_ridge_1.npz', 'causal_ridge_1.json',
                   'causal_ridge_100.npz', 'causal_ridge_100.json')
NUMERIC_ERRORS = ('nonfinite LPV output; no rollout clipping or repair',
                  'nonfinite structured transition output; no repair',
                  'finite CPU tensor with exact shape/dtype: initializer features')
PREPROCESSING = {'version': 'industrial-robot-data-v1', 'raw_samples': 90881, 'raw_hz': 250.,
                 'filter': 'Butterworth lowpass SOS, scipy.signal.butter/sosfilt', 'filter_order': 4, 'cutoff_hz': 4.,
                 'initial_state': "sosfilt_zi scaled by each channel's first sample", 'stride': 25, 'first_raw_index': 0,
                 'q': 'q_se_meas first three rows; q_mot_meas last three rows; degrees',
                 'torque': 'tau_meas all six rows; total measured motor torque; Nm', 'time_atol_seconds': 1e-10}


def model_key(arm, seed):
    require(arm in ARMS and seed in SEEDS, 'fixed model identity')
    return f'{arm}-{seed}-lr{0 if FIXED_RATES[arm] == .001 else 1}'


def config():
    return {'version': STUDY_VERSION, 'arms': list(ARMS), 'primary_arms': list(PRIMARY), 'references': list(REFS),
            'seeds': list(SEEDS), 'fixed_rates': dict(FIXED_RATES), 'partitions': {'confirm': list(CONFIRM)},
            'context': 32, 'horizon': 128, 'horizons': [64, 128], 'skip': 64, 'window_stride': 160,
            'samples_per_recording': 3636, 'windows_per_recording': 22, 'timing_warmups': 3, 'timing_repeats': 20,
            'wall_cap_seconds': 900., 'mean_reduction': .05, 'file_harm_ratio': 1.02, 'latency_ratio': 1.25,
            'preprocessing_sha256': hashlib.sha256(json.dumps(PREPROCESSING, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            'confirmation_access': True, 'official_test_access': False, 'fits': 0, 'normalizer_refits': 0,
            'reference_refits': 0, 'recipe_selection_calls': 0}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular nonsymlink file: ' + str(path))
    raw = path.read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def pin(path):
    return {'path': str(Path(path).resolve()), **descriptor(path)}


def checked_pin(item, label):
    require(isinstance(item, dict) and set(item) == {'path', 'sha256', 'bytes'}
            and Path(item['path']).is_absolute(), label + ': descriptor')
    require(descriptor(item['path']) == {k: item[k] for k in ('sha256', 'bytes')}, label + ': bytes')
    return item


def close(actual, expected, label):
    """Exact schemas, identities and gates; fixed scalar roundoff tolerance."""
    if isinstance(actual, dict):
        require(isinstance(expected, dict) and set(actual) == set(expected), label + ': keys')
        for key in actual:
            close(actual[key], expected[key], label + '/' + key)
    elif isinstance(actual, list):
        require(isinstance(expected, list) and len(actual) == len(expected), label + ': list')
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            close(a, b, label + '/' + str(i))
    elif type(actual) is float:
        require(type(expected) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), label + ': scalar')
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': exact')


def authenticate(study, run_receipt):
    """Closed original process and opaque provenance admission before any decode."""
    study, run_receipt = Path(study).resolve(), Path(run_receipt).resolve()
    engineering = ROOT / 'output/robot-history-confirmation-engineering-v1'
    registration = ROOT / 'research/robot-history-confirmation-registration.json'
    launcher = ROOT / 'scripts/launch_robot_history_confirmation.py'
    require(study == ROOT / 'output/robot-history-confirmation-v1'
            and run_receipt == engineering / 'run-process-01.json', 'exact original confirmation paths')
    # The auditor itself is preregistered, so the original committed plan binds it
    # without a self-referential plan digest embedded in its own source.
    process, launch = read(run_receipt), read(engineering / 'run-launch-01.json')
    launch_keys = {'command', 'pre_access_commit', 'started_utc', 'registration_sha256', 'launcher', 'thread_env', 'scope'}
    require(set(launch) == launch_keys and set(process) == launch_keys | {'returncode', 'elapsed_seconds', 'external_timeout', 'log'},
            'original launch/terminal schema')
    require({k: process[k] for k in launch_keys} == launch and process['command'] == COMMAND
            and type(process['returncode']) is int and process['returncode'] == 0 and process['external_timeout'] is False
            and type(process['elapsed_seconds']) in (int, float) and math.isfinite(process['elapsed_seconds'])
            and 0 < process['elapsed_seconds'] <= 960 and descriptor(run_receipt.with_suffix('.log')) == process['log'],
            'original successful closed process required before reserved inputs')
    plan = read(registration)
    sha = descriptor(registration)['sha256']
    require(process['registration_sha256'] == sha and plan['version'] == STUDY_VERSION and plan['config'] == config(),
            'registered fixed confirmation config')
    require(set(plan['sources']) == set(SOURCES) and len(SOURCES) == 36, 'all36 frozen source files')
    require(process['thread_env'] == dict.fromkeys(THREADS, '1') and all(os.environ.get(k) == '1' for k in THREADS),
            'qualified single-thread environment')
    require(process['launcher'] == plan['launcher'] == descriptor(launcher), 'qualified original launcher')
    require(descriptor(study / 'registration.json') == descriptor(registration), 'saved original registration')
    for name, expected in plan['sources'].items():
        require(descriptor(ROOT / name) == expected == descriptor(study / 'sources' / name), 'source/snapshot drift: ' + name)
    commit = process['pre_access_commit']
    require(type(commit) is str and len(commit) == 40 and all(c in '0123456789abcdef' for c in commit), 'original commit identity')
    for name in (*SOURCES, 'scripts/launch_robot_history_confirmation.py', str(registration.relative_to(ROOT))):
        committed = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
        require(committed == (ROOT / name).read_bytes(), 'pre-access committed source: ' + name)
    inputs = {key: pin(path) for key, path in {
        'manifest': study / 'manifest.json', 'producer_receipt': study / 'receipt.json', 'registration': registration,
        'run_receipt': run_receipt, 'run_log': run_receipt.with_suffix('.log'), 'run_launch': engineering / 'run-launch-01.json',
        'launcher': launcher, 'runtime': study / 'runtime.json'}.items()}
    q = read(checked_pin(plan['qualification'], 'qualification')['path'])
    require(q['status'] == 'PASS' and q['sources_unchanged'] is True and q['sources'] == plan['sources']
            and q['launcher'] == plan['launcher'] and q['thread_env'] == dict.fromkeys(THREADS, '1'), 'exact qualified sources')
    require(datetime.fromisoformat(q['created_utc']) <= datetime.fromisoformat(plan['created_utc'])
            <= datetime.fromisoformat(launch['started_utc']), 'qualification/registration before original access')
    commands = [['.venv/bin/ruff', 'check', *QUALIFICATION_SOURCES], ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q',
                 'tests/test_robot_history_confirmation.py', 'tests/test_audit_robot_history_confirmation.py']]
    require(len(q['commands']) == 2, 'two original qualification commands')
    qlogs = []
    for row, command in zip(q['commands'], commands, strict=True):
        require(row['command'] == command and row['returncode'] == 0 and math.isfinite(row['seconds']) and row['seconds'] > 0
                and descriptor(row['log'])['sha256'] == row['sha256'], 'qualification argv/log closure')
        qlogs.append(pin(row['log']))
    # This qualified parent's admission is metadata-only; it neither opens raw
    # recordings numerically nor calls a model, metric, or selection function.
    from plot_robot_history_initialization import authenticate as admit_parent
    parent_folder = ROOT / 'output/robot-history-initialization-study-v1'
    parent_audit = ROOT / 'output/robot-history-initialization-audit-v1/audit.json'
    parent_engineering = ROOT / 'output/robot-history-initialization-engineering-v1'
    parent = admit_parent(parent_folder, parent_audit, parent_engineering)
    require(plan['parent_registration_sha256'] == PARENT_SHA and all(parent['plan']['sources'][n] == plan['sources'][n]
            for n in SOURCES[:29]), 'inherited source/registration identity')
    result = parent['audit']['results']
    require(result['selection']['selected_rates'] == FIXED_RATES and result['result']['passed'] == result['result']['total'] == 5
            and result['result']['status'] == 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL', 'fixed audited development choices')
    paths = {'manifest': parent_folder / 'manifest.json', 'receipt': parent_folder / 'receipt.json',
             'process': parent_engineering / 'run-process-01.json', 'audit': parent_audit,
             'audit_process': parent_engineering / 'audit-process-01.json'}
    require(set(plan['parent_closure']) == set(paths), 'complete parent closure')
    for name, path in paths.items():
        item = checked_pin(plan['parent_closure'][name], 'parent closure/' + name)
        require(Path(item['path']) == path and {k: item[k] for k in ('sha256', 'bytes')} == parent['inputs'][str(path)],
                'parent original closure join: ' + name)
    payloads = (*COMMON_PAYLOADS, *(model_key(a, s) + '/final.npz' for a in ARMS for s in SEEDS))
    require(set(plan['parent_payloads']) == set(payloads) and len(payloads) == 30, '24 selected finals and6 common payloads')
    inventory = read(parent_folder / 'manifest.json')['files']
    for name, item in plan['parent_payloads'].items():
        checked_pin(item, 'parent payload/' + name)
        require(Path(item['path']) == parent_folder / name and {k: item[k] for k in ('sha256', 'bytes')}
                == inventory[name] == descriptor(study / name), 'exact original/copy checkpoint or reference: ' + name)
    # All original process/source/parent checks above precede reading reserved
    # member bytes, even opaquely. No FIT/DEV/TEST MAT is decoded by this audit.
    archive = ROOT / 'output/robot-data-engineering-v1/raw-bundle-attempt-01.rar'
    extraction_path = engineering / 'raw-extraction-01.json'
    raw_folder = ROOT / 'output/robot-history-confirmation-inputs-v1'
    require(Path(plan['archive']['path']) == archive and plan['archive']['sha256'] == ARCHIVE_SHA
            and Path(plan['extraction']['path']) == extraction_path and set(plan['raw_recordings']) == set(CONFIRM),
            'exact opaque archive/extraction/two CONFIRM identities')
    extraction = read(checked_pin(plan['extraction'], 'original opaque extraction')['path'])
    extraction_command = ['/usr/bin/bsdtar', '-xf', str(archive), '-C', str(raw_folder), *['raw_data/' + n for n in CONFIRM]]
    require(extraction['command'] == extraction_command and extraction['returncode'] == 0
            and extraction['scope'] == 'Opaque extraction and hashing only; no numeric decoding'
            and extraction['archive'] == plan['archive'] and extraction['raw_recordings'] == plan['raw_recordings']
            and type(extraction['elapsed_seconds']) in (int, float) and math.isfinite(extraction['elapsed_seconds'])
            and extraction['elapsed_seconds'] > 0, 'original opaque extraction closure')
    checked_pin(plan['archive'], 'archive')
    for name, item in plan['raw_recordings'].items():
        require(Path(item['path']) == raw_folder / 'raw_data' / name, 'exact registered reserved member')
        checked_pin(item, 'confirmation opaque member/' + name)
    manifest, paths = read(study / 'manifest.json'), list(study.rglob('*'))
    require(set(manifest) == {'files'} and not any(p.is_symlink() for p in paths), 'regular complete manifest')
    require({str(p.relative_to(study)) for p in paths if p.is_file()} == set(manifest['files']) | {'manifest.json', 'receipt.json'},
            'no omitted producer evidence')
    for name, value in manifest['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and descriptor(study / name) == value,
                'producer payload pin: ' + name)
    receipt = read(study / 'receipt.json')
    require(receipt['status'] == 'PASS' and receipt['registration_sha256'] == sha
            and receipt['scientific_result'] in ('CONFIRMED_HISTORY_INITIALIZATION', 'DO_NOT_CONFIRM_HISTORY_INITIALIZATION')
            and type(receipt['seconds']) in (int, float) and math.isfinite(receipt['seconds']) and 0 < receipt['seconds'] <= 900
            and type(receipt['monotonic_seconds']) in (int, float) and math.isfinite(receipt['monotonic_seconds'])
            and 0 < receipt['monotonic_seconds'] <= process['elapsed_seconds'], 'complete producer receipt')
    counters = ('models', 'prediction_attempts', 'rows', 'resource_attempts', 'raw_decodes', 'fits', 'optimizer_updates',
                'normalizer_refits', 'reference_refits', 'recipe_selection_calls')
    require([receipt[k] for k in counters] == [24, 56, 112, 28, 2, 0, 0, 0, 0, 0]
            and receipt['confirmation_access'] is True and receipt['official_test_access'] is False, 'fixed counts and zero new fitting')
    runtime = read(study / 'runtime.json')
    require(runtime['python'] == sys.version and runtime['platform'] == platform.platform() and runtime['machine'] == platform.machine()
            and runtime['thread_env'] == dict.fromkeys(THREADS, '1') and runtime['torch_threads'] == 1, 'same inference runtime/platform')
    for package in ('numpy', 'torch', 'scipy'):
        require(importlib.metadata.version(package) == runtime[package], 'same numeric package: ' + package)
    require(receipt['clock'] == runtime['clock'] == ('mach_continuous_time' if sys.platform == 'darwin' else 'CLOCK_BOOTTIME')
            and runtime['timing_scope'] == 'complete physical request plus uniform native deadline callback', 'native clock/timing scope')
    inputs.update(qualification=plan['qualification'], qualification_logs=qlogs, parent_admission=parent['inputs'],
                  parent_closure=plan['parent_closure'], parent_payloads=plan['parent_payloads'], extraction=plan['extraction'],
                  archive=plan['archive'], raw_recordings=plan['raw_recordings'])
    return plan, inputs


def independent_recording(blob, name, np):
    """Literal raw schema and causal SOS reconstruction, without loader imports."""
    require(name in CONFIRM and type(blob) is bytes, 'only registered confirmation bytes')
    from scipy.io import loadmat, whosmat
    from scipy.signal import butter, sosfilt, sosfilt_zi
    matrices = ('q_mot_meas', 'q_se_meas', 'qd_mot_meas', 'qd_se_meas', 'tau_meas',
                'tau_fb_meas', 'q_ref', 'qd_ref', 'tau_ref_ff')
    expected = {key: ((6, 90881), 'double') for key in matrices}
    expected.update(recording_ok=((1, 1), 'logical'), time=((1, 90881), 'double'), READ_ME=((17, 2), 'cell'))
    headers = whosmat(io.BytesIO(blob))
    require(len(headers) == len(expected) and {k: (tuple(s), t) for k, s, t in headers} == expected,
            'independent MAT header schema')
    fields = ('q_se_meas', 'q_mot_meas', 'tau_meas', 'time', 'recording_ok')
    data = loadmat(io.BytesIO(blob), variable_names=list(fields), mat_dtype=False,
                   squeeze_me=False, verify_compressed_data_integrity=True)
    for key in fields[:-1]:
        value = data[key]
        require(value.dtype.kind == 'f' and value.dtype.itemsize == 8 and value.shape == expected[key][0]
                and np.isfinite(value).all(), 'independent finite raw field: ' + key)
    flag = data['recording_ok']
    require(flag.shape == (1, 1) and flag.dtype.kind in 'bu' and flag.item() == 1, 'recording marked valid')
    require(np.allclose(np.diff(data['time'][0]), 1 / 250., rtol=0., atol=1e-10), 'uniform raw 250Hz clock')
    positions = np.concatenate((data['q_se_meas'][:3], data['q_mot_meas'][3:]), axis=0).T
    channels = np.concatenate((positions, data['tau_meas'].T), axis=1)
    sos = butter(4, 4., btype='lowpass', fs=250., output='sos')
    initial = sosfilt_zi(sos)[:, :, None] * channels[0][None, None, :]
    filtered, _ = sosfilt(sos, channels, axis=0, zi=initial)
    require(np.isfinite(filtered).all(), 'finite independent causal filter')
    indices = np.arange(0, 90881, 25, dtype=np.int64)
    return {'q': filtered[indices, :6], 'u': filtered[indices, 6:], 'raw_indices': indices}


def confirmation_windows(record, norm, np):
    """Independent half-open ranges: first future torque31 predicts position32."""
    require(set(record) == {'q', 'u', 'raw_indices'} and record['q'].shape == record['u'].shape == (3636, 6),
            'complete independent decimated recording')
    q = (record['q'] - norm['q_mean']) / norm['q_std']
    u = (record['u'] - norm['u_mean']) / norm['u_std']
    starts = np.arange(64, 3636-32-128+1, 160, dtype=np.int64)
    require(len(starts) == 22 and int(starts[-1]) == 3424, 'fixed22 disjoint windows')
    batch = {'q_context': np.stack([q[s:s+32] for s in starts]),
             'u_context': np.stack([u[s:s+32] for s in starts]),
             'future_u': np.stack([u[s+31:s+159] for s in starts])}
    return starts, batch, np.stack([q[s+32:s+160] for s in starts])


def scored(prediction, target, scale, horizon, np):
    require(horizon in (64, 128) and prediction.shape == target.shape and target.ndim == 3
            and target.shape[1:] == (128, 6) and len(target) > 0 and scale.shape == (6,)
            and np.isfinite(target).all() and np.isfinite(scale).all() and (scale > 0).all(), 'score geometry/target')
    if not np.isfinite(prediction).all():
        return None
    with np.errstate(over='ignore', invalid='ignore'):
        delta = prediction[:, :horizon] - target[:, :horizon]
        sse = float(np.sum(delta * delta))
        joint = np.mean((delta * scale) ** 2, axis=(0, 1))
    if not math.isfinite(sse) or not np.isfinite(joint).all():
        return None
    return {'standardized_rmse': math.sqrt(sse / delta.size), 'standardized_sse': sse,
            'scalars': int(delta.size), 'physical_rmse_deg': math.sqrt(float(np.mean(joint))),
            'per_joint_rmse_deg': np.sqrt(joint).tolist(), 'windows': len(target), 'horizon': horizon}


def metric_rows(common, prediction, target, scale, error, np):
    rows = []
    for horizon in (64, 128):
        local, value = error, None
        if local is None:
            require(prediction.dtype == np.float64 and prediction.shape == target.shape, 'full float64 forecast bank')
            value = scored(prediction, target, scale, horizon, np)
            if value is None:
                local = {'type': 'NonfiniteEvaluation', 'message': 'nonfinite prediction/target'
                         if not np.isfinite(prediction).all() else 'nonfinite metric arithmetic'}
        rows.append({**common, 'horizon': horizon, 'status': 'PASS' if local is None else 'FAILED',
                     'error': local, 'metrics': value})
    return rows


def storage(arm):
    require(arm in (*ARMS, *REFS), 'declared storage family')
    neural = arm in ARMS
    count = PARAMETERS[arm] if neural else 445440 if arm in REFS[:2] else 150 if arm == 'linear_frozen' else 0
    state = (50 if arm == 'gru32' else 28 if arm == 'gru10' else 12) if neural else 192 if arm in REFS[:2] else 18 if arm == 'linear_frozen' else 6
    result = {'parameters': count, 'parameter_bytes': count * (4 if neural else 8),
              'buffer_bytes': 16 if arm in REFS[:2] else 0, 'state_scalars': state,
              'state_bytes': state * (4 if neural else 8), 'normalizer_bytes': 192,
              'dtype': 'float32' if neural else 'float64', 'input_bytes': 9216, 'output_bytes': 6144,
              'temporary_workspace': 'not measured'}
    if neural:
        result['inactive_parameters'] = 192 if arm == 'legacy_instant' else 0
    return result


def validate_resources(resources, np):
    """Attest saved raw timings and logical bytes; never rerun a timing sample."""
    roster = [(a, s, FIXED_RATES[a], model_key(a, s)) for a in ARMS for s in SEEDS]
    roster += [(a, None, None, a) for a in REFS]
    require(len(resources) == 28, 'all28 timing attempts')
    for row, (arm, seed, rate, key) in zip(resources, roster, strict=True):
        expected = storage(arm)
        require(set(row) == {'key', 'arm', 'seed', 'learning_rate', 'status', 'error', 'timing'} | set(expected), 'resource schema')
        require((row['arm'], row['seed'], row['learning_rate'], row['key']) == (arm, seed, rate, key), 'fixed ordered cost identity')
        close({k: row[k] for k in expected}, expected, 'full persistent numeric accounting')
        if row['status'] == 'FAILED':
            require(row['timing'] is None and row['error'] in [
                {'type': 'NonfiniteTiming', 'message': message} for message in (*NUMERIC_ERRORS, 'finite complete timed request', 'nonfinite ridge prediction')],
                'only retained known numeric timing failure')
            continue
        require(row['status'] == 'PASS' and row['error'] is None and isinstance(row['timing'], dict), 'successful timing status')
        timing = row['timing']
        require(set(timing) == {'seconds', 'median_seconds', 'p95_seconds', 'scope'} and timing['scope'] == TIMING_SCOPE,
                'complete physical request timing scope')
        samples = timing['seconds']
        require(len(samples) == 20 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples),
                'twenty positive finite original durations')
        close(float(np.median(samples)), timing['median_seconds'], 'timing median')
        close(float(np.percentile(samples, 95)), timing['p95_seconds'], 'timing p95')


def reference_prediction(arm, coefficients, batch, np):
    """Independent causal ridge, linear AR2 and persistence inference."""
    q, u, future = (batch[key] for key in ('q_context', 'u_context', 'future_u'))
    if arm in REFS[:2]:
        design = np.concatenate((q[:, 16:].reshape(len(q), 96), u[:, 15:31].reshape(len(q), 96),
                                 future.reshape(len(q), 768), np.ones((len(q), 1))), axis=1)
        result = np.empty((len(q), 128, 6), np.float64)
        for h in range(1, 129):
            indices = np.r_[np.arange(192+6*h), 960]
            result[:, h-1] = design[:, indices] @ coefficients[arm][f'h{h:03d}'].T
        return result
    if arm == 'persistence':
        return np.repeat(q[:, -1:, :], 128, axis=1)
    require(arm == 'linear_frozen', 'declared reference')
    current, previous, prior_u = q[:, -1].copy(), q[:, -2].copy(), u[:, -2].copy()
    output = []
    with np.errstate(over='ignore', invalid='ignore'):
        for torque in future.transpose(1, 0, 2):
            feature = np.concatenate((current, previous, torque, prior_u, np.ones((len(q), 1))), axis=1)
            prediction = feature @ coefficients[arm].T
            output.append(prediction)
            previous, current, prior_u = current, prediction, torque
    return np.stack(output, axis=1)


def decisions(rows, resources, cfg, fixed_rates):
    """Five rules on immutable DEV-selected recipes, with no selection function."""
    require(fixed_rates == FIXED_RATES and cfg == config(), 'immutable config and DEV-selected rates')
    names = cfg['partitions']['confirm']
    require(tuple(names) == CONFIRM, 'fixed confirmation roster')
    keys = {(r['recording'], r['arm'], r['seed'], r['learning_rate'], r['horizon']): r for r in rows}
    expected = {(f, a, s, fixed_rates[a], h) for f in names for a in ARMS for s in SEEDS for h in (64, 128)}
    expected |= {(f, a, None, None, h) for f in names for a in REFS for h in (64, 128)}
    require(len(rows) == len(keys) == 112 and set(keys) == expected, 'all112 fixed-checkpoint metric identities')
    def metric(f, a, s):
        row = keys[f, a, s, fixed_rates.get(a), 128]
        m = row['metrics']
        if row['status'] != 'PASS' or not isinstance(m, dict):
            return None
        values = [m[k] for k in ('standardized_rmse', 'standardized_sse', 'physical_rmse_deg')] + m['per_joint_rmse_deg']
        return m if (len(m['per_joint_rmse_deg']) == 6 and type(m['scalars']) is int and m['scalars'] > 0
                     and type(m['windows']) is int and m['windows'] > 0 and m['horizon'] == 128
                     and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values)) else None
    details, equal_means, costs = {}, {}, {}
    for name in names:
        means, seeds = {}, {}
        for arm in (*ARMS, *REFS):
            roster = SEEDS if arm in ARMS else (None,)
            values = [metric(name, arm, seed) for seed in roster]
            if all(v is not None for v in values):
                means[arm] = sum(v['standardized_rmse'] for v in values) / len(values)
                seeds[arm] = {seed: m['standardized_rmse'] for seed, m in zip(roster, values, strict=True)}
        paired = {a: {str(s): seeds['temporal_affine'][s] - seeds[a][s] for s in SEEDS}
                  for a in PRIMARY[:2] if 'temporal_affine' in seeds and a in seeds}
        details[name] = {'means': means, 'paired_primary_differences': paired}
    cost_ids = [(r['arm'], r['seed'], r.get('learning_rate')) for r in resources]
    expected_costs = {(a, s, fixed_rates[a]) for a in ARMS for s in SEEDS} | {(a, None, None) for a in REFS}
    require(len(cost_ids) == 28 and len(set(cost_ids)) == 28 and set(cost_ids) == expected_costs,
            'all28 fixed resource identities')
    for arm in (*ARMS, *REFS):
        if all(arm in d['means'] for d in details.values()):
            equal_means[arm] = sum(d['means'][arm] for d in details.values()) / len(details)
        subset = [r for r in resources if r['arm'] == arm]
        expected_seeds = set(SEEDS) if arm in ARMS else {None}
        valid = len(subset) == len(expected_seeds) and {r['seed'] for r in subset} == expected_seeds
        latencies, sizes = [], []
        for row in subset:
            t = row.get('timing')
            t = t.get('median_seconds') if isinstance(t, dict) else None
            parts = [row.get(k) for k in ('parameter_bytes', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')]
            valid = valid and row.get('learning_rate') == fixed_rates.get(arm)
            valid = valid and row.get('status') == 'PASS' and row.get('error') is None
            valid = valid and type(t) in (int, float) and math.isfinite(t) and t > 0
            valid = valid and all(type(v) is int and v >= 0 for v in parts)
            if valid:
                latencies.append(t); sizes.append(sum(parts))
        costs[arm] = {'latency': statistics.median(latencies), 'bytes': max(sizes)} if valid else None
    candidate = equal_means.get('temporal_affine')
    complete = all(a in equal_means for a in PRIMARY)
    improvement = candidate is not None and all(equal_means.get(a, 0) > 0
                     and candidate <= .95 * equal_means[a] for a in PRIMARY[:2])
    no_harm = all(all(a in d['means'] for a in PRIMARY) and d['means']['temporal_affine']
                  <= 1.02 * min(d['means'][a] for a in PRIMARY[:2]) for d in details.values())
    left, base = costs['temporal_affine'], costs['last_two']
    latency = left is not None and base is not None and left['latency'] <= 1.25 * base['latency']
    frontier = all(a in equal_means and costs[a] is not None for a in (*ARMS, *REFS))
    dominators = []
    if frontier:
        primary = (candidate, left['latency'], left['bytes'])
        for arm in (*ARMS, *REFS):
            if arm == 'temporal_affine':
                continue
            control = (equal_means[arm], costs[arm]['latency'], costs[arm]['bytes'])
            if all(x <= y for x, y in zip(control, primary, strict=True)) and any(x < y for x, y in zip(control, primary, strict=True)):
                dominators.append(arm)
    conditions = [{'name': name, 'passed': bool(value)} for name, value in zip(CONDITIONS,
                  (complete, improvement, no_harm, latency, frontier and not dominators), strict=True)]
    passed = sum(r['passed'] for r in conditions)
    return {'status': 'CONFIRMED_HISTORY_INITIALIZATION' if passed == 5 else 'DO_NOT_CONFIRM_HISTORY_INITIALIZATION',
            'passed': passed, 'total': 5, 'conditions': conditions, 'fixed_rates': dict(FIXED_RATES), 'details': details,
            'equal_file_means': equal_means, 'costs': costs, 'frontier_complete': frontier, 'dominators': dominators,
            'scope': 'fixed-checkpoint internal recording confirmation; not official TEST, independent robots, novelty or control'}


def audit(study, run_receipt):
    started = time.monotonic()
    study = Path(study).resolve()
    plan, inputs = authenticate(study, run_receipt)
    # No numerical library, loader or model import precedes original closure.
    import numpy as np
    import robot_history_initialization_study as replay
    import torch
    torch.set_num_threads(1)
    counts = {'npz_decodes': 0, 'array_decodes': 0, 'raw_mat_decodes': 0, 'selected_mat_variables': 0,
              'confirmation_decodes': 0, 'fit_mat_decodes': 0, 'dev_mat_decodes': 0, 'official_test_decodes': 0,
              'checkpoint_loads': 0, 'checkpoint_replays': 0, 'reference_replays': 0, 'prediction_files': 0,
              'metric_rows': 0, 'prediction_attempts': 0, 'resource_attempts': 28, 'timing_replays': 0,
              'fits': 0, 'optimizer_updates': 0, 'reference_refits': 0, 'normalizer_refits': 0, 'recipe_selection_calls': 0}
    expected = {'registration.json', 'runtime.json', 'models.json', 'checkpoint-barrier.json', 'parameter-checks.json',
                'prediction-attempts.json', 'results.json', 'resources.json', *COMMON_PAYLOADS}
    expected |= {'sources/' + name for name in SOURCES}

    def load(name):
        expected.add(name)
        with np.load(study / name, allow_pickle=False) as source:
            arrays = {key: source[key].copy(order='K') for key in source.files}
        counts['npz_decodes'] += 1
        counts['array_decodes'] += len(arrays)
        return arrays

    def equal(left, right, label):
        require(left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right, equal_nan=True), label)

    def state_hash(model):
        digest = hashlib.sha256()
        for name, value in model.state_dict().items():
            a = value.detach().cpu().numpy()
            digest.update(name.encode() + b'\0' + str(a.dtype).encode() + repr(a.shape).encode() + a.tobytes())
        return digest.hexdigest()

    norm = load('normalizers.npz')
    require(set(norm) == {'q_mean', 'q_std', 'u_mean', 'u_std'} and all(a.dtype == np.float64 and a.shape == (6,)
            and np.isfinite(a).all() for a in norm.values()) and (norm['q_std'] > 0).all() and (norm['u_std'] > 0).all(),
            'fixed finite FIT normalization schema')
    linear_data = load('linear.npz')
    require(set(linear_data) == {'coefficient'}, 'linear coefficient field')
    linear = linear_data['coefficient']
    require(linear.dtype == np.float64 and linear.shape == (6, 25) and np.isfinite(linear).all(), 'linear coefficient schema')
    coefficients = {'linear_frozen': linear}
    for arm, penalty in zip(REFS[:2], (1., 100.), strict=True):
        bank = load(arm + '.npz')
        require(set(bank) == {f'h{h:03d}' for h in range(1, 129)}, 'all128 causal coefficient matrices')
        for h in range(1, 129):
            a = bank[f'h{h:03d}']
            require(a.dtype == np.float64 and a.shape == (6, 193 + 6*h) and np.isfinite(a).all(), 'finite causal ridge shape')
        metadata = read(study / (arm + '.json'))
        require(metadata['version'] == 'causal-robot-ridge-v1' and metadata['penalty'] == penalty
                and metadata['horizons'] == 128 and metadata['outputs'] == 6 and metadata['dtype'] == 'float64'
                and metadata['coefficient_count'] == sum(a.size for a in bank.values()) == 445440
                and metadata['coefficient_bytes'] == sum(a.nbytes for a in bank.values()) == 3563520
                and metadata['scalar_metadata_bytes'] == 16 and metadata['retained_numeric_bytes'] == 3563536
                and type(metadata['fit_rows']) is int and metadata['fit_rows'] > 0, 'original fixed causal reference metadata')
        coefficients[arm] = bank
    saved_models, models, records, before = read(study / 'models.json'), {}, [], {}
    require(len(saved_models) == 24, 'all24 selected models before raw decode')
    parent_folder = ROOT / 'output/robot-history-initialization-study-v1'
    ledger_path = parent_folder / 'fits.json'
    require(descriptor(ledger_path) == read(parent_folder / 'manifest.json')['files']['fits.json'], 'unchanged historical fit ledger')
    ledger_rows = read(ledger_path)
    ledger = {r['key']: r for r in ledger_rows}
    require(len(ledger_rows) == len(ledger) == 48, 'all historical fit identities')
    for arm in ARMS:
        for seed in SEEDS:
            key = model_key(arm, seed)
            checkpoint = load(key + '/final.npz')
            model = replay.model_for(arm, seed, linear)
            template = model.state_dict()
            require(set(checkpoint) == set(template) and all(a.dtype == np.float32 and a.shape == tuple(template[k].shape)
                    and np.isfinite(a).all() for k, a in checkpoint.items()), 'qualified finite checkpoint schema: ' + key)
            model.load_state_dict({k: torch.from_numpy(a) for k, a in checkpoint.items()}, strict=True)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            require(all(p.grad is None for p in model.parameters()), 'no stored gradients')
            spec = storage(arm)
            require(sum(p.numel() for p in model.parameters()) == spec['parameters']
                    and sum(p.numel()*p.element_size() for p in model.parameters()) == spec['parameter_bytes']
                    and sum(b.numel()*b.element_size() for b in model.buffers()) == spec['buffer_bytes'], 'actual checkpoint numeric storage')
            parent_fit = ledger[key]
            require((parent_fit['arm'], parent_fit['seed'], parent_fit['learning_rate']) == (arm, seed, FIXED_RATES[arm])
                    and parent_fit['effective_status'] == parent_fit['fit']['status'] == 'PASS'
                    and parent_fit['fit']['files']['final.npz'] == descriptor(study / key / 'final.npz'), 'successful fixed historical checkpoint')
            digest = state_hash(model)
            records.append({'key': key, 'arm': arm, 'seed': seed, 'learning_rate': FIXED_RATES[arm],
                            'checkpoint': plan['parent_payloads'][key + '/final.npz'], 'state_sha256': digest,
                            'resources': spec, 'historical_fit': parent_fit['fit'], 'parent_fit_ledger': pin(ledger_path),
                            'training_scope': 'historical parent fit only; zero new updates'})
            models[key], before[key] = model, digest
            counts['checkpoint_loads'] += 1
    close(records, saved_models, '24 original selected model/fit/resource identities')
    barrier = {'checkpoint_count': 24, 'raw_decodes': 0, 'fits': 0, 'fixed_rates': dict(FIXED_RATES),
               'checkpoints': {r['key']: descriptor(study / r['key'] / 'final.npz') for r in records}, 'state_sha256': before}
    close(barrier, read(study / 'checkpoint-barrier.json'), 'all checkpoint barrier before CONFIRM decode')
    identities = [{k: r[k] for k in ('key', 'arm', 'seed', 'learning_rate')} for r in records]
    identities += [{'key': arm, 'arm': arm, 'seed': None, 'learning_rate': None} for arm in REFS]
    rows, attempts = [], []
    for name in CONFIRM:
        item = plan['raw_recordings'][name]
        checked_pin(item, 'reserved member before decode')
        record = independent_recording(Path(item['path']).read_bytes(), name, np)
        counts['raw_mat_decodes'] += 1
        counts['confirmation_decodes'] += 1
        counts['selected_mat_variables'] += 5
        saved_record = load('confirm-data-' + name + '.npz')
        require(set(saved_record) == set(record), 'saved causal recording fields')
        for field in record:
            equal(saved_record[field], record[field], 'independent causal filtering/decimation: ' + name + '/' + field)
        starts, batch, target = confirmation_windows(record, norm, np)
        windows = load('confirm-windows-' + name + '.npz')
        require(set(windows) == {'starts', 'target'}, 'saved target window fields')
        equal(windows['starts'], starts, 'independent window starts')
        equal(windows['target'], target, 'independent normalized targets')
        for identity in identities:
            key, arm = identity['key'], identity['arm']
            prediction, error = None, None
            filename = 'prediction-' + name + '-' + key + '.npz'
            if arm in ARMS:
                counts['checkpoint_replays'] += 1
                try:
                    with torch.no_grad():
                        prediction = replay.old.infer(models[key], batch).numpy().astype(np.float64)
                except replay.old.FitFailure as exc:
                    require(str(exc) in NUMERIC_ERRORS, 'known qualified numerical forecast failure')
                    error = {'type': 'NonfiniteEvaluation', 'message': str(exc)}
            else:
                counts['reference_replays'] += 1
                prediction = reference_prediction(arm, coefficients, batch, np)
                if arm in REFS[:2] and not np.isfinite(prediction).all():
                    prediction = None
                    error = {'type': 'NonfiniteEvaluation', 'message': 'nonfinite ridge prediction'}
            if prediction is None:
                require(error is not None and not (study / filename).exists(), 'numeric exception has no fabricated forecast bank')
            else:
                saved = load(filename)
                require(set(saved) == {'prediction'}, 'single prediction field')
                equal(saved['prediction'], prediction, 'exact qualified neural or independent reference replay')
                counts['prediction_files'] += 1
            common = {'recording': name, **identity}
            scores = metric_rows(common, prediction, target, norm['q_std'], error, np)
            rows.extend(scores)
            attempt = {**common, 'status': 'PASS' if all(r['status'] == 'PASS' for r in scores) else 'FAILED',
                       'errors': [r['error'] for r in scores], 'prediction_file': filename if prediction is not None else None,
                       'prediction_pin': descriptor(study / filename) if prediction is not None else None}
            attempts.append(attempt)
            completed = f'completed-prediction-{len(attempts):02d}.json'
            expected.add(completed)
            close(attempt, read(study / completed), 'original ordered forecast completion')
    close(attempts, read(study / 'prediction-attempts.json'), 'all56 fixed attempts')
    counts['metric_rows'], counts['prediction_attempts'] = len(rows), len(attempts)
    require(len(rows) == 112 and len(attempts) == 56 and counts['checkpoint_replays'] == 48 and counts['reference_replays'] == 8,
            'complete fixed forecast roster')
    resources = read(study / 'resources.json')
    validate_resources(resources, np)
    for i, resource in enumerate(resources, 1):
        completed = f'completed-timing-{i:02d}.json'
        expected.add(completed)
        close(resource, read(study / completed), 'original ordered timing completion')
    after = {key: state_hash(model) for key, model in models.items()}
    require(after == before and all(not p.requires_grad and p.grad is None for m in models.values() for p in m.parameters()),
            'all24 checkpoint state hashes unchanged without gradients')
    close({'before': before, 'after': after, 'unchanged': True, 'optimizer_updates': 0},
          read(study / 'parameter-checks.json'), 'producer before/after immutability')
    result = decisions(rows, resources, plan['config'], FIXED_RATES)
    results = {'version': STUDY_VERSION, 'config': plan['config'], 'fixed_rates': dict(FIXED_RATES), 'rows': rows, 'result': result}
    close(results, read(study / 'results.json'), 'independent112 scores and all5 frozen criteria')
    require(read(study / 'receipt.json')['scientific_result'] == result['status'], 'producer scientific status join')
    require(set(read(study / 'manifest.json')['files']) == expected and len(expected) == 162 + counts['prediction_files'],
            'complete declared output roster; no undeclared data or omitted failures')
    counts['producer_payloads'] = len(expected)
    counts['timing_samples_checked'] = sum(len(r['timing']['seconds']) for r in resources if r['status'] == 'PASS')
    final_plan, final_inputs = authenticate(study, run_receipt)
    require(final_plan == plan and final_inputs == inputs, 'all original source/input/output evidence unchanged after audit')
    return {'version': VERSION, 'status': 'PASS', 'agreement': True, 'study': str(study),
            'registration_sha256': inputs['registration']['sha256'], 'inputs': inputs, 'source_pins': plan['sources'],
            'auditor': pin(Path(__file__)), 'results': results, 'resources': resources, 'prediction_attempts': attempts,
            'counts': counts, 'seconds': time.monotonic() - started,
            'tolerance': {'arrays': 'exact same-runtime values with matching dtype/shape; identical nonfinite positions',
                          'scalar_rtol': 1e-10, 'scalar_atol': 1e-12},
            'scope': 'Independent two-recording raw preprocessing, windows, reference forecasts, scores and fixed rules; '
                     'qualified neural implementation replay only. Historical fitting and physical latency are not replayed. '
                     'Zero optimization, rate selection, FIT/DEV/official TEST MAT decoding or new reference fits.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--run-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    require(args.study.resolve() not in target.parents and target != args.study.resolve(), 'audit outside immutable producer output')
    require(not target.parent.exists(), 'exclusive audit output folder required')
    target.parent.mkdir(parents=True, exist_ok=False)
    try:
        result = audit(args.study, args.run_receipt)
        with target.open('x') as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
        with (target.parent / 'manifest.json').open('x') as handle:
            json.dump({'files': {target.name: descriptor(target)}}, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
        print(json.dumps({'status': result['status'], 'agreement': result['agreement'],
                          'result': result['results']['result']['status'], 'counts': result['counts']}), flush=True)
    except BaseException as exc:
        with (target.parent / 'failure.json').open('x') as handle:
            json.dump({'status': 'FAILED', 'type': type(exc).__name__, 'message': str(exc)}, handle, indent=2, allow_nan=False)
            handle.write('\n')
        raise


if __name__ == '__main__':
    main()
