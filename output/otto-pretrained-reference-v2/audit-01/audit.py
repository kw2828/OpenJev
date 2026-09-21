"""Independent saved-array qualification readback. No inference or HDF5 loading."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import resource
import signal
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'output/otto-pretrained-reference-v2'
RUN = BASE / 'qualification-01'
PLAN = BASE / 'qualification-plan-01.json'
TERMINAL = BASE / 'qualification-process-01.terminal.json'
PINS = {'plan': '40d90544fd1d9a49f12275140f704d96c5dfae2c3ac1becc8aebf55ac19364f5',
        'receipt': '30c1b7321787053820a5ebdd1194a2ff085d7c8d21f7e9a62d4e31ab15ac9457',
        'terminal': 'ff7747fa81fbc0ec8e264d2eebe6702b71c7f93f8fbdaf3467d8b0558f87df24'}
CAPS = {'native_seconds': 60, 'rss_bytes': 4 * 1024**3, 'output_bytes': 128 * 1024**2}
CLOCK = ROOT / 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
SCOPE = ('Independent saved prediction/input/mass/cost/first-tie arithmetic and fixed coverage; '
         'authenticates source/input/payload and saved process metadata. Inherits actual model-call, '
         'TensorFlow tensor-extraction, the successful actual-NumPy-call-versus-helper input witness '
         '(the actual-call array was not separately saved), and runtime/clock truth. Does not load HDF5, instantiate a '
         'model, replay matrix multiplication, call an actor/simulator, or change qualification gates.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def local(name):
    p = ROOT / name
    require(not Path(name).is_absolute() and '..' not in Path(name).parts and p.is_file()
            and not any(x.is_symlink() for x in (p, *p.parents)), 'regular contained input')
    return p


def descriptor(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def authenticate(check):
    for name, path in [('plan', PLAN), ('receipt', RUN / 'receipt.json'), ('terminal', TERMINAL)]:
        require(sha(path) == PINS[name], f'external {name} pin')
    plan, done, terminal = read(PLAN), read(RUN / 'receipt.json'), read(TERMINAL)
    require(plan['version'] == done['version'] == 'otto-pretrained-qualification-runtime-v2'
            and plan['status'] == 'frozen_before_run' and done['status'] == 'completed'
            and done['plan_sha256'] == PINS['plan'], 'complete fixed plan/run')
    require(done['inputs'] == plan['inputs'] and done['sources'] == plan['sources'], 'source/input joins')
    for name, pin in plan['sources'].items():
        check()
        require(sha(local(name)) == pin, f'source {name}')
    for name, item in plan['inputs'].items():
        require(descriptor(local(item['path'])) == {'sha256': item['sha256'], 'bytes': item['bytes']}, f'input {name}')
    files = {'original.weights-legacy.h5', 'policy-arrays.npz', 'policy-checks.json', 'policy-checks.jsonl',
             'runtime.json', 'started.json', 'summary.json', 'value-checks.json', 'value-checks.jsonl',
             'value-inputs.npz', 'weight-checks.json', 'weight-checks.jsonl', 'work-ledger.jsonl'}
    require(set(done['files']) == files and {p.name for p in RUN.iterdir()} == files | {'receipt.json'}, 'exact 13 payload closure')
    for name, record in done['files'].items():
        check()
        require(not (RUN / name).is_symlink() and descriptor(RUN / name) == record, f'payload {name}')
    require(done['files']['original.weights-legacy.h5']['sha256'] == plan['inputs']['original_weights']['sha256'], 'HDF5 identity')
    started = read(RUN / 'started.json')
    launch = started['launch']
    require(sha(BASE / 'qualification-process-01.launch.json') == done['supervision_sha256']
            and read(BASE / 'qualification-process-01.launch.json') == launch, 'launch identity')
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and not terminal['timed_out']
            and terminal['group_absent'] and terminal['cleanup']['reaped'] and terminal['cleanup']['errors'] == []
            and terminal['error'] is None and terminal['clock_error'] is None and terminal['timing_available'], 'successful parent')
    for key in ['command', 'pid', 'pgid', 'parent_pid', 'clock_backend', 'started_ns', 'deadline_ns',
                'cap_seconds', 'watchdog_sha256', 'clock_source_sha256']:
        require(terminal[key] == launch[key], f'parent join {key}')
    command = terminal['command']
    for flag, value in [('plan', str(PLAN)), ('plan-sha256', PINS['plan']), ('output', str(RUN)),
                        ('supervision', str(BASE / 'qualification-process-01.launch.json'))]:
        require(command.count('--' + flag) == 1 and command[command.index('--' + flag) + 1] == value, f'actual command {flag}')
    require(terminal['clock_backend'] == done['clock_backend'] == 'mach_continuous_time'
            and terminal['started_ns'] <= done['started_ns'] <= done['finished_ns'] <= terminal['finished_ns'] < terminal['deadline_ns']
            and terminal['deadline_ns'] == terminal['started_ns'] + 600 * 10**9
            and terminal['elapsed_ns'] == terminal['finished_ns'] - terminal['started_ns'], 'strict native enclosure')
    require(done['wall_seconds'] == (done['finished_ns'] - done['started_ns']) / 1e9
            and terminal['wall_seconds'] == terminal['elapsed_ns'] / 1e9, 'saved clock arithmetic')
    runtime = read(RUN / 'runtime.json')
    require(runtime['versions'] == plan['runtime']['versions'] and runtime['executable'] == plan['runtime']['python_executable']
            and runtime['keras_configuration'] == {'floatx': 'float32', 'image_data_format': 'channels_last', 'mixed_precision_policy': 'float32'}, 'saved runtime identity')
    require(done['simulator_calls'] == done['training_updates'] == 0, 'no simulator/training')
    return plan, done, terminal


def compare(a, b, exact=False):
    require(a.shape == b.shape and a.dtype == b.dtype and np.isfinite(a).all() and np.isfinite(b).all(), 'finite equal array geometry')
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    limit = np.zeros_like(delta) if exact else 1e-4 + 1e-5 * np.abs(b.astype(np.float64))
    mismatch = int(np.count_nonzero(delta > limit))
    byte_equal = a.tobytes() == b.tobytes() if exact else None
    return {'shape_equal': True, 'dtype_equal': True, 'finite': True, 'byte_equal': byte_equal,
            'compared_scalars': int(delta.size), 'mismatched_scalars': mismatch,
            'maximum_absolute_error': float(delta.max(initial=0)),
            'maximum_tolerance_ratio': 0.0 if exact else float((delta / limit).max(initial=0)),
            'passed': mismatch == 0 and (byte_equal if exact else True)}


def selected(scores):
    minimum = min(float(x) for x in scores)
    ties = [i for i, x in enumerate(scores) if abs(float(x) - minimum) < 1e-10]
    return ties[0], ties


def validate_record(record, recomputed):
    for key, value in recomputed.items():
        if isinstance(value, float):
            require(math.isclose(record[key], value, rel_tol=1e-14, abs_tol=1e-15), f'comparison value {key}')
        else:
            require(record[key] == value, f'comparison field {key}')


def audit(plan, done, check):
    records = {}
    for kind in ['weight', 'value', 'policy']:
        records[kind] = read(RUN / f'{kind}-checks.json')
        require(records[kind] == [json.loads(x) for x in (RUN / f'{kind}-checks.jsonl').read_text().splitlines()], f'{kind} durable journal agreement')
    weights, values, policies = (records[k] for k in ['weight', 'value', 'policy'])
    metadata = read(local(plan['inputs']['tensor_metadata']['path']))
    require([x['id'] for x in weights] == [f'{kind}_{i}' for i in range(4) for kind in ['kernel', 'bias']], 'all eight ordered weight identities')
    for r in weights:
        item = metadata['datasets'][r['id']]
        require(r['keras_array_sha256'] == r['extracted_array_sha256'] == item['sha256_c_order']
                and r['compared_scalars'] == math.prod(item['shape']) and r['byte_equal'] and r['passed']
                and r['maximum_absolute_error'] == r['mismatched_scalars'] == 0, 'bound Keras/extraction identity witness')
    names = ['zero', 'subnormalized_point', 'center_point', 'uniform', 'asymmetric_pair', 'subnormalized_dense']
    names += [f'd4_asymmetric_{i}' for i in range(8)] + ['corner_point', 'checkerboard']
    routes = [(sym, size, offset, names[offset:offset + size]) for sym in [False, True]
              for size in [1, 3, 16] for offset in range(0, 16, size)]
    require([(r['sym_avg'], r['batch_size'], r['offset'], r['ids']) for r in values] == routes, 'all 46 raw routes')
    with np.load(RUN / 'value-inputs.npz', allow_pickle=False) as z:
        require(z.files == ['inputs'] and z['inputs'].shape == (16, 105, 105)
                and z['inputs'].dtype == np.float32 and np.isfinite(z['inputs']).all(), 'sixteen finite saved raw inputs')
    maximum_raw, maximum_policy, maximum_cost, maximum_cost_reconstruction = 0.0, 0.0, 0.0, 0.0
    for r in values:
        a, b = (np.asarray(r[k], dtype=np.float32) for k in ['numpy', 'tensorflow'])
        c = compare(a, b)
        validate_record(r, c)
        maximum_raw = max(maximum_raw, c['maximum_absolute_error'])
    case_ids = [f'{regime}.position{i}.{pattern}' for regime in ['base', 'shift'] for i in range(4)
                for pattern in ['uniform', 'adjacent_point', 'asymmetric_pair', 'seeded_dense']]
    case_ids += [f'mechanical.mass{i}' for i in range(4)]
    require([r['id'] for r in policies] == case_ids, 'all 32 physical and four mechanical policies')
    disagreements, identity_scalars = [], 0
    with np.load(RUN / 'policy-arrays.npz', allow_pickle=False) as z:
        expected = {f'case_{i:02d}_{suffix}' for i in range(36) for suffix in
                    ['belief', 'position', 'tensorflow_inputs', 'numpy_inputs', 'tensorflow_masses', 'numpy_masses']}
        expected |= {f'case_{i:02d}_kernel' for i in range(32, 36)}
        require(set(z.files) == expected, 'exact saved policy array membership')
        for i, r in enumerate(policies):
            check()
            prefix = f'case_{i:02d}'
            require(z[prefix + '_position'].tolist() == r['position'], 'case position join')
            require(r['kind'] == ('physical' if i < 32 else 'mechanical_floor'), 'fixed fixture kind')
            for kind, shape in [('inputs', (16, 105, 105)), ('masses', (4, 4))]:
                a, b = (z[prefix + '_' + backend + '_' + kind] for backend in ['numpy', 'tensorflow'])
                require(a.shape == shape and a.dtype == np.float32, 'policy saved geometry')
                c = compare(a, b, exact=True)
                validate_record(r['comparisons'][kind], c)
                require(c['passed'], 'byte-identical policy inputs and masses')
                identity_scalars += c['compared_scalars']
            require(r['comparisons']['numpy_policy_inputs'] == r['comparisons']['inputs'], 'secondary input witness agrees')
            for kind in ['values', 'scores']:
                a, b = (np.asarray(r[backend + '_' + kind], dtype=np.float32) for backend in ['numpy', 'tensorflow'])
                require(a.shape == ((16,) if kind == 'values' else (4,)), 'fixed saved value/cost shape')
                c = compare(a, b)
                validate_record(r['comparisons'][kind], c)
                if kind == 'values':
                    maximum_policy = max(maximum_policy, c['maximum_absolute_error'])
                else:
                    maximum_cost = max(maximum_cost, c['maximum_absolute_error'])
            for backend in ['numpy', 'tensorflow']:
                mass = z[prefix + '_' + backend + '_masses']
                saved_values = np.asarray(r[backend + '_values'], dtype=np.float32).reshape(4, 4)
                # Four-term reduction order may differ across TF and NumPy.
                # This verifies the formula within float32 arithmetic error;
                # the qualification/action gates below use the saved scores exactly.
                ideal = np.asarray([1.0 + math.fsum(float(mass[a, h]) * float(saved_values[a, h])
                                   for h in range(4)) for a in range(4)])
                cost = np.asarray(r[backend + '_scores'], dtype=np.float32)
                error = np.abs(cost.astype(np.float64) - ideal)
                absolute_sum = np.asarray([1.0 + math.fsum(abs(float(mass[a, h]) * float(saved_values[a, h]))
                                          for h in range(4)) for a in range(4)])
                bound = 8 * np.finfo(np.float32).eps * absolute_sum
                require(np.all(error <= bound), 'cost formula within float32 product/reduction rounding')
                maximum_cost_reconstruction = max(maximum_cost_reconstruction, float(error.max()))
                choice, ties = selected(cost)
                require(choice == r[backend + '_action'] and ties == r[backend + '_tie_actions'], 'exact first tie choice')
            action_equal = r['numpy_action'] == r['tensorflow_action']
            require(r['action_equal'] == action_equal and r['passed'] ==
                    (action_equal and all(x['passed'] for x in r['comparisons'].values())), 'complete per-policy decision')
            if not action_equal:
                disagreements.append({k: r[k] for k in ['id', 'kind', 'position', 'tensorflow_action', 'numpy_action',
                                      'tensorflow_scores', 'numpy_scores', 'tensorflow_tie_actions', 'numpy_tie_actions']})
    calls = {name: {'attempted': 0, 'returned': 0} for name in done['work']['calls']}
    counts = {k: 0 for k in ['weight', 'value', 'policy']}
    pending = None
    for line in (RUN / 'work-ledger.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['event'] == 'call_attempt':
            require(pending is None, 'no overlapping recorded calls')
            pending = r['pending_call']
            calls[pending['channel']]['attempted'] += 1
            require(pending['ordinal'] == calls[pending['channel']]['attempted'], 'contiguous attempt ordinal')
        elif r['event'] == 'call_return':
            require(pending is not None, 'return requires an attempt')
            calls[pending['channel']]['returned'] += 1
            pending = None
        elif r['event'] == 'comparison_saved':
            changed = [k for k in counts if r['completed_comparisons'][k] != counts[k]]
            require(len(changed) == 1, 'one saved comparison at a time')
            counts[changed[0]] += 1
        else:
            require(r['event'] == 'context' and pending is None, 'valid context transition')
        require(r['calls'] == calls and r['pending_call'] == pending and r['completed_comparisons'] == counts, 'durable work state')
    expected_calls = {k: {'attempted': (82 if k.endswith('_value') else 1), 'returned': (82 if k.endswith('_value') else 1)} for k in calls}
    require(calls == expected_calls == done['work']['calls'] and counts == {'weight': 8, 'value': 46, 'policy': 36}
            and pending is None and done['work']['pending_call'] is None, 'all declared calls returned')
    published = read(RUN / 'summary.json')
    for key, expected_count in [('weights_compared', 8), ('value_batches_compared', 46),
                                ('raw_value_predictions_compared', 96), ('policy_fixtures_compared', 36),
                                ('tensorflow_value_calls', 82), ('numpy_value_calls', 82), ('model_build_calls', 1)]:
        require(published[key] == expected_count, f'published complete count {key}')
    gates = {'weight_identity': True, 'policy_input_identity': True,
             'value_parity': all(r['passed'] for r in values) and all(r['comparisons']['values']['passed'] for r in policies),
             'policy_cost_parity': all(r['comparisons']['scores']['passed'] for r in policies),
             'policy_action_parity': not disagreements}
    require(all(published[k] == value for k, value in gates.items()) and published['qualified'] == done['qualified'] == all(gates.values()), 'independent qualification gate')
    require(published['action_disagreements'] == [r['id'] for r in disagreements]
            and published['failed_policy_fixtures'] == [r['id'] for r in policies if not r['passed']]
            and published['failed_value_batches'] == [] and published['near_tie_exemptions'] is False, 'no omitted failures or exemptions')
    return {'agreement': True, 'gates': gates, 'qualified': all(gates.values()), 'weights_witnessed': 8,
            'raw_value_batches': 46, 'raw_values': sum(len(r['ids']) for r in values), 'policy_fixtures': 36,
            'physical_actions_equal': 32 - sum(r['kind'] == 'physical' for r in disagreements),
            'mechanical_actions_equal': 4 - sum(r['kind'] == 'mechanical_floor' for r in disagreements),
            'saved_input_mass_scalars_byte_compared': identity_scalars,
            'maximum_raw_value_error': maximum_raw, 'maximum_policy_value_error': maximum_policy,
            'maximum_policy_cost_error': maximum_cost,
            'maximum_cost_formula_reconstruction_error': maximum_cost_reconstruction,
            'cost_formula_check_scope': 'Independent float64 four-term formula with float32 rounding bound; qualification cost and exact first-tie gates use saved float32 scores without relaxation.',
            'disagreements': disagreements,
            'calls': calls, 'scope': SCOPE}


def main():
    require({p.name for p in OUT.iterdir()} == {'audit.py'}, 'exclusive unused audit output')
    receipt = {'status': 'started', 'pins': PINS, 'source_sha256': sha(Path(__file__)), 'limits': CAPS,
               'model_calls': 0, 'simulator_calls': 0, 'scope': SCOPE}
    clock = start = None
    def check():
        require(clock.now_ns() - start < 60 * 10**9, 'native audit cap')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024) <= CAPS['rss_bytes'], 'audit RSS cap')
        require(sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) <= CAPS['output_bytes'], 'audit output cap')
    try:
        write(OUT / 'started.json', {**receipt, 'utc': datetime.now(timezone.utc).isoformat()})
        require(sha(CLOCK) == CLOCK_PIN, 'qualified clock source')
        spec = importlib.util.spec_from_file_location('_readback_clock', CLOCK)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        clock = module.SuspendClock()
        start = clock.now_ns()
        def alarm(*_):
            raise TimeoutError('audit awake emergency cap')
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, 60)
        plan, done, terminal = authenticate(check)
        global np
        import numpy as np
        result = audit(plan, done, check)
        authenticate(check)
        require(receipt['source_sha256'] == sha(Path(__file__)), 'held audit source')
        write(OUT / 'summary.json', result)
        check()
        receipt.update(status='completed', agreement=True, qualified=result['qualified'], numpy=np.__version__,
                       wall_seconds=(clock.now_ns() - start) / 1e9, clock_backend=clock.backend,
                       peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                       source_files_verified=len(plan['sources']), producer_wall_seconds=done['wall_seconds'],
                       supervisor_wall_seconds=terminal['wall_seconds'],
                       files={p.name: descriptor(p) for p in OUT.iterdir() if p.is_file()})
        write(OUT / 'receipt.json', receipt)
        check()
        print(json.dumps({'status': 'completed', 'agreement': True, 'qualified': result['qualified'], 'receipt_sha256': sha(OUT / 'receipt.json')}))
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        try:
            if (OUT / 'receipt.json').exists():
                (OUT / 'receipt.json').rename(OUT / 'receipt.invalid.json')
            write(OUT / 'receipt.json', receipt)
            write(OUT / 'failed.json', receipt)
        except BaseException as secondary:
            error.add_note(f'Failure publication also failed: {secondary!r}')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
