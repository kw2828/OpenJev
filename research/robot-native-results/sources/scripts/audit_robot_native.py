"""Independent saved-output native audit. No model, raw data or producer imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'robot-native-audit-v1'
ARMS = ('householder', 'dense_bounded', 'dense_unbounded', 'dense_mlp', 'gru32')
SEEDS = (8101, 8102, 8103)
PARAMETERS = dict(zip(ARMS, (630, 806, 806, 590, 5916), strict=True))
PACKED = dict(zip(ARMS, (638, 806, 806, 590, 5916), strict=True))
RTOL = ATOL = 1e-5
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
CHECKS = (('physical', 'native_physical', 'torch_physical'),
          ('standardized', 'native_standardized', 'torch_standardized'),
          ('physical_final_state', 'native_final', 'torch_final'),
          ('standardized_final_state', 'native_standard_final', 'torch_final'))
BACKENDS = {'torch': ('torch_standardized', 'torch_physical', 'torch_final'),
            'native_physical': ('native_physical', 'native_final'),
            'native_standardized': ('native_standardized', 'native_standard_final')}


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    raw = Path(path).read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON field')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def close(actual, expected):
    """Exact identities/types/rosters; scalar recomputation allows only roundoff."""
    if isinstance(expected, dict):
        require(type(actual) is dict and set(actual) == set(expected), 'scalar field roster')
        for key in expected:
            close(actual[key], expected[key])
    elif isinstance(expected, list):
        require(type(actual) is list and len(actual) == len(expected), 'scalar list roster')
        for left, right in zip(actual, expected, strict=True):
            close(left, right)
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), 'scalar arithmetic')
    else:
        require(type(actual) is type(expected) and actual == expected, 'exact identity/status/counter')


def median(values):
    values = sorted(values)
    require(len(values) > 0, 'nonempty median')
    n = len(values)
    return float(values[n // 2]) if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


def authenticate(folder, process_path, registration_path):
    """Opaque closure/source joins complete before any retained-array decoding."""
    folder, process_path, registration_path = (Path(p).resolve() for p in (folder, process_path, registration_path))
    inputs = {}
    def bind(path, expected=None):
        path = Path(path).resolve()
        actual = pin(path)
        require(expected is None or actual == expected, 'changed evidence: ' + str(path))
        inputs[str(path)] = actual
        return actual
    def descriptor(value):
        require(set(value) == {'path', 'sha256', 'bytes'} and Path(value['path']).is_absolute(), 'absolute descriptor')
        return bind(value['path'], {key: value[key] for key in ('sha256', 'bytes')})
    bind(registration_path)
    bind(process_path)
    registration, process = read(registration_path), read(process_path)
    require(registration['version'] == 'robot-native-benchmark-v1', 'benchmark registration version')
    close(registration['config'], {'arms': list(ARMS), 'seeds': list(SEEDS), 'rtol': RTOL, 'atol': ATOL,
                                  'cap_seconds': 900, 'warmups': 3, 'repeats': 30,
                                  'parity_records': 30, 'batch_sizes': [22, 1]})
    require(process['command'] == registration['command'] and type(process['returncode']) is int
            and process['returncode'] in (0, 1) and process['inputs_unchanged'] is True
            and type(process['elapsed_seconds']) in (int, float) and 0 < process['elapsed_seconds'] < 900,
            'closed original benchmark process')
    bind(registration_path, process['registration'])
    bind(process_path.with_suffix('.log'), process['log'])
    bind(folder / 'manifest.json', process['manifest'])
    bind(folder / 'receipt.json', process['receipt'])
    command = registration['command']
    require(len(command) == 12 and command[:2] == ['.venv/bin/python', 'scripts/benchmark_robot_native.py']
            and command[2::2] == ['--study', '--audit', '--engineering', '--qualification', '--output'], 'benchmark argv')
    absolute = lambda value: (ROOT / value).resolve()
    require(absolute(command[-1]) == folder and absolute(command[3]) == Path(registration['original_study']['path'])
            and absolute(command[5]) == Path(registration['original_audit']['path'])
            and absolute(command[9]) == Path(registration['native_qualification']['path']), 'registered command paths')
    for name, expected in registration['sources'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'source relative path')
        bind(ROOT / name, expected)
    for key in ('original_audit', 'original_manifest', 'original_receipt', 'original_run_receipt',
                'native_qualification', 'protocol'):
        descriptor(registration[key])
    for item in registration['helper_qualification'].values():
        descriptor(item)
    original = Path(registration['original_study']['path']).resolve()
    audit = read(registration['original_audit']['path'])
    require(audit['status'] == 'PASS' and audit['agreement'] is True and audit['study'] == str(original), 'original audit admission')
    for name, key in (('manifest', 'original_manifest'), ('producer_receipt', 'original_receipt'),
                      ('run_receipt', 'original_run_receipt')):
        require(audit['inputs'][name] == registration[key], 'original audit provenance join')
    require(read(registration['original_receipt']['path'])['status'] == 'PASS'
            and read(registration['original_run_receipt']['path'])['returncode'] == 0,
            'original successful study closure')
    original_manifest = read(registration['original_manifest']['path'])['files']
    for name in ('fits.json', 'registration.json'):
        bind(original / name, original_manifest[name])
    original_plan, fits = read(original / 'registration.json'), read(original / 'fits.json')
    recordings = original_plan['config']['partitions']['dev']
    require(len(recordings) == len(set(recordings)) == 2, 'two original DEV recordings')
    rates = audit['results']['selection']['selected_rates']
    chosen = []
    require(len(fits) == 36 and len({f['key'] for f in fits}) == 36, 'original36fit metadata')
    for arm in ARMS:
        require(rates[arm] in (.001, .003), 'original selected rate eligible')
        for seed in SEEDS:
            selected = [f for f in fits if (f['arm'], f['seed'], f['learning_rate']) == (arm, seed, rates[arm])]
            require(len(selected) == 1, 'selected checkpoint roster')
            fit = selected[0]
            require(fit['origin'] == 'fresh' and fit['fit']['status'] == 'PASS'
                    and fit['fit']['completed_updates'] == 4096, 'qualified selected checkpoint')
            bind(original / fit['key'] / 'final.npz', original_manifest[fit['key'] + '/final.npz'])
            chosen.append(fit)
    manifest = read(folder / 'manifest.json')
    require(set(manifest) == {'files'}, 'benchmark manifest schema')
    files = list(folder.rglob('*'))
    require(not any(p.is_symlink() for p in files), 'no evidence symlinks')
    require({str(p.relative_to(folder)) for p in files if p.is_file()}
            == set(manifest['files']) | {'manifest.json', 'receipt.json'}, 'complete benchmark inventory')
    for name, expected in manifest['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'safe inventory path')
        bind(folder / name, expected)
    definition, receipt = read(folder / 'definition.json'), read(folder / 'receipt.json')
    require(definition['version'] == receipt['version'] == registration['version']
            and definition['study'] == str(original) and definition['inputs'] == receipt['inputs'], 'benchmark definition joins')
    for path, expected in definition['inputs'].items():
        require(Path(path).is_absolute(), 'absolute admitted input')
        bind(path, expected)
    for key in ('original_audit', 'original_manifest', 'original_receipt', 'original_run_receipt', 'native_qualification'):
        item = registration[key]
        require(definition['inputs'][item['path']] == {k: item[k] for k in ('bytes', 'sha256')}, 'definition registered input join')
    for name, expected in registration['sources'].items():
        require(definition['inputs'][str(ROOT / name)] == expected, 'registered source captured before execution')
    require(definition['native']['qualification'] == registration['native_qualification']['path'], 'native original qualification')
    native = read(registration['native_qualification']['path'])
    require(native['status'] == 'PASS' and native['counts'] == {'tests': 110, 'failed': 0, 'errors': 0, 'skipped': 0}
            and native['sources_before'] == native['sources_after'], 'qualified compiled library')
    descriptor(native['library'])
    require(definition['native']['library'] == native['library']['path']
            and definition['inputs'][native['library']['path']] == {k: native['library'][k] for k in ('sha256', 'bytes')},
            'executed library identity')
    for key, expected in {'arms': list(ARMS), 'seeds': list(SEEDS), 'context': 32, 'horizon': 128,
                          'parity_comparisons': 30, 'parity_batch_sizes': [22, 1], 'rtol': RTOL, 'atol': ATOL,
                          'warmup_pairs': 3, 'timed_pairs': 30, 'cap_seconds': 900}.items():
        close(definition[key], expected)
    require(receipt['manifest'] == pin(folder / 'manifest.json'), 'benchmark receipt manifest join')
    return {'inputs': inputs, 'manifest': manifest['files'], 'receipt': receipt, 'process': process,
            'definition': definition, 'selected': chosen, 'recordings': recordings, 'original_audit': audit}


def array_comparison(actual, expected):
    import numpy as np
    require(actual.shape == expected.shape, 'comparison shape')
    finite = np.isfinite(actual) & np.isfinite(expected)
    with np.errstate(over='ignore', invalid='ignore'):
        difference = np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64))
        allowed = ATOL + RTOL * np.abs(np.asarray(expected, dtype=np.float64))
    passed = finite & (difference <= allowed)
    errors = difference[finite]
    maximum = float(np.max(errors)) if errors.size else None
    return {'passed': bool(np.all(passed)), 'values': actual.size,
            'nonfinite': int(np.count_nonzero(~finite)), 'violations': int(np.count_nonzero(~passed)),
            'max_absolute_error': maximum if maximum is None or math.isfinite(maximum) else None}


def validate_work(work, arm, batch):
    gru = arm == 'gru32'
    expected = {'parameter_validations': 1 if gru else 2, 'parameter_preparations': 0 if gru else 1,
                'ffi_calls': 1, 'batch': batch, 'horizon': 128,
                'packed_parameter_bytes': PACKED[arm]*4, 'parameter_export_piece_bytes': PACKED[arm]*4,
                'input_copy_bytes': (4608 if gru else 3120)*batch,
                'output_buffer_bytes': (3272 if gru else 3120)*batch,
                'normalized_input_bytes': 9216*batch, 'cast_input_bytes': 4608*batch,
                'physical_output_bytes': 6144*batch, 'retained_prepared_bytes': 0}
    if gru:
        expected.update(context=32, context_gru_steps=30*batch, rollout_gru_steps=128*batch)
    else:
        expected.update(steps=128*batch, condition_state_bytes=48*batch,
                        prepared_tensor_bytes={'householder': 528, 'dense_bounded': 1208,
                                               'dense_unbounded': 1200, 'dense_mlp': 1208}[arm])
    require(set(work) == set(expected) | {'scope'} and isinstance(work['scope'], str), 'work field roster')
    close({key: work[key] for key in expected}, expected)


def validate_parity(record, arrays, arm):
    import numpy as np
    state = 50 if arm == 'gru32' else 12
    errors = record['errors']
    require(type(errors) is list, 'backend errors list')
    error_keys = set()
    for error in errors:
        require(set(error) == {'batch', 'backend', 'type', 'message'}
                and error['batch'] in ('batch22', 'batch1') and error['backend'] in BACKENDS
                and error['type'] in ('ValueError', 'RuntimeError', 'FitFailure') and type(error['message']) is str,
                'declared backend failure')
        key = error['batch'], error['backend']
        require(key not in error_keys, 'duplicate backend error')
        error_keys.add(key)
    expected_keys, expected_work, checks = set(), set(), {}
    for label, batch in (('batch22', 22), ('batch1', 1)):
        for backend, names in BACKENDS.items():
            if (label, backend) in error_keys:
                continue
            for name in names:
                key = label + '/' + name
                expected_keys.add(key)
                require(key in arrays, 'missing successful backend output')
                value = arrays[key]
                final = name.endswith('final')
                require(value.dtype == np.dtype('float32' if final else 'float64')
                        and value.shape == ((batch, state) if final else (batch, 128, 6)), 'retained array geometry')
            if backend != 'torch':
                key = label + ('/physical' if backend == 'native_physical' else '/standardized')
                expected_work.add(key)
                validate_work(record['work'][key], arm, batch)
        for name, left, right in CHECKS:
            a, b = label + '/' + left, label + '/' + right
            if a in arrays and b in arrays:
                checks[label + '/' + name] = array_comparison(arrays[a], arrays[b])
    require(set(arrays) == expected_keys and set(record['work']) == expected_work, 'complete retained backend roster')
    close(record['checks'], checks)
    passed = not errors and len(checks) == 8 and all(value['passed'] for value in checks.values())
    close(record['passed'], passed)
    return {'passed': passed, 'checks': len(checks), 'arrays': len(arrays), 'backend_errors': len(errors)}


def timing_summary(pairs):
    require(type(pairs) is list and len(pairs) == 33, 'three warmup and30 timed pairs')
    measured = {'torch': [], 'native': []}
    ratios = []
    for index, row in enumerate(pairs):
        phase = 'warmup' if index < 3 else 'timed'
        repetition = index if index < 3 else index - 3
        order = ['torch', 'native'] if repetition % 2 == 0 else ['native', 'torch']
        require(set(row) == {'phase', 'repetition', 'order', 'seconds'}, 'timing pair schema')
        close({key: row[key] for key in ('phase', 'repetition', 'order')},
              {'phase': phase, 'repetition': repetition, 'order': order})
        require(set(row['seconds']) == {'torch', 'native'} and all(type(v) in (float, int)
                and math.isfinite(v) and v > 0 for v in row['seconds'].values()), 'positive finite durations')
        if phase == 'timed':
            for backend, values in measured.items():
                values.append(row['seconds'][backend])
            ratios.append(row['seconds']['torch'] / row['seconds']['native'])
    medians = {key: median(value) for key, value in measured.items()}
    return {'median_seconds': medians, 'torch_over_native': medians['torch']/medians['native'], 'paired_ratios': ratios}


def aggregate(timings, recordings):
    families, ratios = [], []
    for recording in recordings:
        costs = {}
        for arm in ARMS:
            subset = [row for row in timings if row['recording'] == recording and row['arm'] == arm]
            require(len(subset) == 3 and {r['seed'] for r in subset} == set(SEEDS), 'three-seed family timing')
            values = {backend: median([row['median_seconds'][backend] for row in subset])
                      for backend in ('torch', 'native')}
            costs[arm] = values['native']
            families.append({'recording': recording, 'arm': arm, 'fit_count': 3,
                             'median_of_fit_medians_seconds': values,
                             'torch_over_native': values['torch']/values['native']})
        ratios.extend({'recording': recording, 'numerator': 'householder', 'denominator': arm,
                       'native_latency_ratio': costs['householder']/costs[arm]} for arm in ARMS[1:])
    return families, ratios


def audit(folder, process_path, registration_path):
    started = time.perf_counter()
    folder = Path(folder).resolve()
    proof = authenticate(folder, process_path, registration_path)
    import numpy as np
    summary = read(folder / 'summary.json')
    require(summary['version'] == 'robot-native-benchmark-v1' and summary['status'] in ('PASS', 'FAILED')
            and summary['inputs_unchanged'] is True and summary['clock_error'] is None
            and summary['confirmation_access'] is summary['raw_measurement_access'] is summary['target_scoring'] is False,
            'benchmark terminal scope')
    require(type(summary['elapsed_seconds']) in (int, float) and 0 < summary['elapsed_seconds'] < 900,
            'internal deadline closed')
    close(proof['receipt']['elapsed_seconds'], summary['elapsed_seconds'])
    close(proof['receipt']['status'], summary['status'])
    close(proof['receipt']['error'], summary['error'])
    require(proof['process']['returncode'] == (0 if summary['status'] == 'PASS' else 1), 'process/status agreement')
    host = read(folder / 'host.json')
    require(host['threads'] == dict.fromkeys(THREADS, '1') and host['torch_threads'] == host['torch_interop_threads'] == 1,
            'single-thread runtime')
    fits, recordings = proof['selected'], proof['recordings']
    expected = [(fit, recording) for fit in fits for recording in recordings]
    parity = summary['parity']
    require(len(parity) == 30, 'all30 parity slots retained')
    close(read(folder / 'parity.json'), parity)
    roster = {'definition.json', 'host.json', 'parity.json', 'summary.json'}
    counts = {'parity_records': 30, 'npz_decodes': 0, 'array_loads': 0, 'parity_checks': 0,
              'backend_errors': 0, 'timing_records': 0, 'timed_pairs': 0, 'warmup_pairs': 0,
              'model_calls': 0, 'raw_measurement_decodes': 0, 'training_calls': 0}
    for index, (row, (fit, recording)) in enumerate(zip(parity, expected, strict=True), 1):
        identity = {'fit_key': fit['key'], 'arm': fit['arm'], 'seed': fit['seed'],
                    'learning_rate': fit['learning_rate'], 'recording': recording}
        close({key: row[key] for key in identity}, identity)
        require(set(row) == set(identity) | {'file', 'checks', 'errors', 'work', 'passed'}, 'parity row schema')
        name = f'parity-{recording}-{fit["key"]}.npz'
        require(row['file'] == name, 'parity filename')
        close(read(folder / f'parity-{index:02d}.json'), row)
        roster.update((name, f'parity-{index:02d}.json'))
        with np.load(folder / name, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        result = validate_parity(row, arrays, fit['arm'])
        counts['npz_decodes'] += 1
        counts['array_loads'] += result['arrays']
        counts['parity_checks'] += result['checks']
        counts['backend_errors'] += result['backend_errors']
    resources = summary['resources']
    require(len(resources) == 15, 'all15 selected resource rows')
    for row, fit in zip(resources, fits, strict=True):
        identity = {'fit_key': fit['key'], 'arm': fit['arm'], 'seed': fit['seed'], 'learning_rate': fit['learning_rate']}
        close({key: row[key] for key in identity}, identity)
        require(set(row) == set(identity) | {'storage', 'retained_gradient_tensors'}, 'resource row schema')
        close(row['retained_gradient_tensors'], 0)
        count, state = PARAMETERS[fit['arm']], 50 if fit['arm'] == 'gru32' else 12
        storage = {'parameter_count': count, 'parameter_bytes': count*4, 'buffer_bytes': 0,
                   'retained_gradient_bytes': 0, 'state_bytes_per_stream': state*4,
                   'normalizer_bytes': 192, 'retained_prepared_bytes': 0}
        require(set(row['storage']) == set(storage) | {'scope'}, 'storage fields')
        close({key: row['storage'][key] for key in storage}, storage)
        original = [r for r in proof['original_audit']['resources'] if
                    (r['arm'], r.get('seed'), r.get('learning_rate')) == (fit['arm'], fit['seed'], fit['learning_rate'])]
        require(len(original) == 1 and [original[0][key] for key in ('parameters', 'parameter_bytes', 'state_bytes',
                    'buffer_bytes', 'normalizer_bytes')] == [count, count*4, state*4, 0, 192], 'original audited resource join')
    passed = all(row['passed'] for row in parity)
    timings = summary['timings']
    if passed:
        require(summary['status'] == 'PASS' and summary['error'] is None and len(timings) == 30, 'complete passing timing outcome')
        close(read(folder / 'parity-barrier.json'), {'passed': True, 'comparisons': 30, 'timed_requests_started': 0})
        roster.add('parity-barrier.json')
        for index, (row, (fit, recording)) in enumerate(zip(timings, expected, strict=True), 1):
            identity = {key: fit[key] for key in ('arm', 'seed', 'key', 'learning_rate')}
            identity.update(recording=recording, status='PASS')
            close({key: row[key] for key in identity}, identity)
            values = timing_summary(row['pairs'])
            require(set(row) == set(identity) | {'pairs'} | set(values), 'timing row fields')
            close({key: row[key] for key in values}, values)
            close(read(folder / f'timing-{index:02d}.json'), row)
            roster.add(f'timing-{index:02d}.json')
        families, ratios = aggregate(timings, recordings)
        require(set(summary['result']) == {'family_timings', 'native_family_ratios', 'scope'}, 'descriptive result fields')
        close(summary['result']['family_timings'], families)
        close(summary['result']['native_family_ratios'], ratios)
        counts.update(timing_records=30, timed_pairs=900, warmup_pairs=90)
    else:
        require(summary['status'] == 'FAILED' and timings == [] and summary['result'] is None,
                'failed parity must retain zero timing and no performance result')
        require(summary['error'] == {'type': 'ValueError', 'message': 'all30 parity comparisons must pass before timing'},
                'completed original parity gate failure')
    close(proof['receipt']['parity_comparisons'], 30)
    close(proof['receipt']['timed_slots'], len(timings))
    require(set(proof['manifest']) == roster, 'exact full terminal artifact roster')
    require(all(pin(path) == value for path, value in proof['inputs'].items()), 'all admitted bytes unchanged after audit')
    return {'version': VERSION, 'status': 'PASS', 'agreement': True, 'benchmark_status': summary['status'],
            'study': str(folder), 'auditor': {'path': str(Path(__file__).resolve()), **pin(__file__)},
            'inputs': proof['inputs'], 'counts': counts, 'parity_passed': sum(row['passed'] for row in parity),
            'parity_total': 30, 'results': summary['result'], 'resources': resources,
            'seconds': time.perf_counter()-started,
            'scope': 'Independent saved-array comparisons and duration arithmetic only. No model inference or training, '
                     'raw data, targets, quality rescoring or timing rerun. Backend failures and resource/timing samples '
                     'are recorded evidence, not independently re-executed. Audit PASS is agreement, not benchmark promotion.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--process', required=True, type=Path)
    parser.add_argument('--registration', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.resolve().is_relative_to(args.study.resolve()), 'audit output outside original benchmark')
    args.output.parent.mkdir(parents=True, exist_ok=False)
    result = audit(args.study, args.process, args.registration)
    with args.output.open('x') as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
    with (args.output.parent / 'manifest.json').open('x') as handle:
        json.dump({'files': {args.output.name: pin(args.output)}}, handle, sort_keys=True, indent=2)
        handle.write('\n')
    print(json.dumps({'status': result['status'], 'benchmark_status': result['benchmark_status'], 'counts': result['counts']}))


if __name__ == '__main__':
    main()
