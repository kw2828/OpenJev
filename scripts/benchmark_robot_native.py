"""Closed-weight native engineering comparison. Never fit or score targets.

All measured parity records precede timing. The 900s suspend-aware limit is
checked around every request; an external process supervisor must also enforce
the cap because a Python check cannot interrupt a blocked native call.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import plot_robot_structured as proof

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'robot-native-benchmark-v1'
ARMS = ('householder', 'dense_bounded', 'dense_unbounded', 'dense_mlp', 'gru32')
SEEDS = (8101, 8102, 8103)
RTOL = ATOL = 1e-5
CAP_SECONDS = 900
WARMUPS, REPEATS = 3, 30
QUALIFICATION_SHA = '7bcdd4a41316b086ac494e50832266fb4aae461f9f6a132b4afce1af38338113'
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
NEW_SOURCES = ('scripts/benchmark_robot_native.py', 'tests/test_benchmark_robot_native.py',
               'research/robot-native-protocol.md', 'scripts/plot_robot_structured.py',
               'src/openjev/research/suspend_clock.py')
require = proof.require
read = proof.read_json
pin = proof.pin


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def authenticate_native(path):
    """Exact original fabricated closure, compiled bytes and sources, no imports."""
    path = Path(path).resolve()
    inputs = {}
    def bind(p, expected=None):
        p = Path(p).resolve()
        value = pin(p)
        require(expected is None or value == expected, 'native evidence changed: ' + str(p))
        inputs[str(p)] = value
        return value
    def desc(item):
        require(set(item) == {'path', 'bytes', 'sha256'}, 'native descriptor schema')
        return bind(item['path'], {k: item[k] for k in ('bytes', 'sha256')})
    require(bind(path)['sha256'] == QUALIFICATION_SHA, 'original native qualification hash')
    q = read(path)
    require(q['status'] == 'PASS' and q['counts'] == {'tests': 110, 'failed': 0, 'errors': 0, 'skipped': 0}
            and q['evidence_unchanged'] is True and q['sources_before'] == q['sources_after']
            and q['tolerance'] == {'rtol': RTOL, 'atol': ATOL}, 'complete native qualification')
    desc(q['preflight'])
    preflight = read(q['preflight']['path'])
    require(preflight['sources_before'] == q['sources_before'] and preflight['expected_tests'] == 110
            and preflight['thread_env'] == dict.fromkeys(THREADS, '1'), 'qualified preflight join')
    for name, item in q['sources_before'].items():
        require(Path(item['path']) == ROOT / name, 'native source location')
        desc(item)
        bind(path.parent / 'source' / name, {k: item[k] for k in ('bytes', 'sha256')})
    require(len(q['sources_before']) == 17, 'native17 source closure')
    for key in ('prior_failed_attempt', 'source_derived_reduction_diagnostic'):
        for item in preflight[key].values():
            desc(item)
    for key in ('compiler', 'python', 'qualification_script'):
        desc(preflight[key])
    require(Path(preflight['python']['path']).resolve() == Path(sys.executable).resolve(), 'qualified Python runtime')
    require(set(q['process_receipts']) == {'lint-01', 'build-01', 'test-01'}, 'native phase roster')
    for name, item in q['process_receipts'].items():
        desc(item)
        process = read(item['path'])
        command_key = {'lint-01': 'lint', 'build-01': 'build', 'test-01': 'tests'}[name]
        require(process['state'] == 'EXITED' and type(process['returncode']) is int
                and process['returncode'] == 0 and process['error'] is None
                and process['command'] == preflight['commands'][command_key]
                and 0 < process['elapsed_seconds'] < 120, 'closed original native process')
        desc(process['log'])
    library = Path(q['library']['path'])
    desc(q['library'])
    admission_path = path.parent / 'test-admission.json'
    bind(admission_path)
    admission = read(admission_path)
    require(admission['library'] == q['library'] and admission['preflight'] == q['preflight'], 'tested library join')
    desc(admission['build_receipt'])
    build = read(admission['build_receipt']['path'])
    require(build['status'] == 'PASS' and build['sources_unchanged'] is True
            and build['library'] == q['library'] and len(build['processes']) == 2, 'qualified library build')
    definition_path = library.parent / 'definition.json'
    bind(definition_path, build['definition'])
    definition = read(definition_path)
    for name, value in definition['sources'].items():
        require(value == {k: q['sources_before'][name][k] for k in ('bytes', 'sha256')}, 'compiled source pin')
        bind(library.parent / 'source' / Path(name).name, value)
    for process, command in zip(build['processes'], definition['commands'], strict=True):
        require(process['returncode'] == 0 and process['command'] == command, 'original compiler command')
        desc(process['log'])
    require(definition['compiler'] == preflight['compiler'], 'compiler identity')
    bind(ROOT / 'scripts/build_robot_native.py', definition['builder'])
    for name in NEW_SOURCES:
        bind(ROOT / name)
    return {'library': str(library), 'inputs': inputs, 'qualification': str(path),
            'compiler': definition['compiler'], 'commands': definition['commands']}


def selected_fits(fits, selected_rates):
    expected = {(arm, seed, rate) for arm in (*ARMS, 'legacy_instant')
                for seed in SEEDS for rate in (.001, .003)}
    keyed = {(f['arm'], f['seed'], f['learning_rate']): f for f in fits}
    require(len(keyed) == len(fits) == 36 and set(keyed) == expected, 'complete unique original fit roster')
    result = []
    for arm in ARMS:
        rate = selected_rates[arm]
        require(rate in (.001, .003), 'every fresh family needs an eligible original selected rate')
        for seed in SEEDS:
            fit = keyed[(arm, seed, rate)]
            require(fit['origin'] == 'fresh' and fit['fit']['status'] == 'PASS'
                    and fit['fit']['completed_updates'] == 4096
                    and fit['key'] == f'{arm}-{seed}-lr{(.001, .003).index(rate)}', 'complete selected fit')
            result.append(fit)
    return result


def physical_contexts(record, starts):
    """Reconstruct inputs only. Never slice or expose future position targets."""
    import numpy as np
    q, u = record['q'], record['u']
    require(q.dtype == u.dtype == np.float64 and q.ndim == 2 and q.shape == u.shape
            and q.shape[1] == 6 and np.isfinite(q).all() and np.isfinite(u).all(), 'physical recording schema')
    require(starts.dtype == np.int64 and starts.ndim == 1 and len(starts) > 0
            and np.all(starts >= 0) and np.all(starts + 160 <= len(q)), 'input window bounds')
    return {'q_context': np.stack([q[s:s+32] for s in starts]),
            'u_context': np.stack([u[s:s+32] for s in starts]),
            'future_u': np.stack([u[s+31:s+159] for s in starts])}


def load_pinned_arrays(path, inputs, keys=None):
    import numpy as np
    path = Path(path).resolve()
    require(str(path) in inputs and pin(path) == inputs[str(path)], 'decode only authenticated arrays')
    with np.load(path, allow_pickle=False) as data:
        wanted = data.files if keys is None else keys
        return {key: data[key].copy(order='K') for key in wanted}


def normalized_inputs(physical, norm):
    return ((physical['q_context'] - norm['q_mean']) / norm['q_std'],
            (physical['u_context'] - norm['u_mean']) / norm['u_std'],
            (physical['future_u'] - norm['u_mean']) / norm['u_std'])


def torch_request(model, physical, norm):
    """Whole physical request using the original qualified inference methods."""
    import numpy as np
    import torch
    with torch.no_grad():
        q, u, future = normalized_inputs(physical, norm)
        require(all(np.isfinite(a).all() for a in (q, u, future)), 'finite normalized inputs')
        q, u, future = (torch.from_numpy(a.astype(np.float32)) for a in (q, u, future))
        state = model.condition(q, u)
        prediction, final = model(future, state)
        final = torch.cat(tuple(final), dim=1) if isinstance(final, tuple) else final
        standard = prediction.numpy().astype(np.float64)
        final = final.numpy().copy(order='K')
        physical_prediction = standard * norm['q_std'] + norm['q_mean']
        require(all(np.isfinite(a).all() for a in (standard, physical_prediction, final)), 'finite Torch outputs')
        return {'standardized': standard, 'prediction': physical_prediction, 'final_state': final}


def comparison(actual, expected):
    import numpy as np
    require(actual.shape == expected.shape, 'parity shape')
    finite = np.isfinite(actual) & np.isfinite(expected)
    with np.errstate(over='ignore', invalid='ignore'):
        delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    within = finite & (delta <= ATOL + RTOL * np.abs(expected.astype(np.float64)))
    maximum = float(delta[finite].max()) if finite.any() else None
    return {'passed': bool(within.all()), 'values': actual.size,
            'nonfinite': int((~finite).sum()), 'violations': int((~within).sum()),
            'max_absolute_error': maximum if maximum is None or math.isfinite(maximum) else None}


def require_all_parity(records):
    require(len(records) == 30 and len({(r['fit_key'], r['recording']) for r in records}) == 30
            and all(r['passed'] for r in records), 'all30 parity comparisons must pass before timing')


def parity_pair(model, native, physical, norm, check):
    """Compare full DEV batch and the actual batch1 timing request separately."""
    import numpy as np
    arrays, checks, errors, work = {}, {}, [], {}
    for batch_name in ('batch22', 'batch1'):
        batch = physical if batch_name == 'batch22' else {key: value[:1].copy(order='K') for key, value in physical.items()}
        current = {}
        for backend in ('torch', 'native_physical', 'native_standardized'):
            check()
            try:
                if backend == 'torch':
                    values = torch_request(model, batch, norm)
                    current.update(torch_standardized=values['standardized'], torch_physical=values['prediction'], torch_final=values['final_state'])
                elif backend == 'native_physical':
                    values = native.request(batch['q_context'], batch['u_context'], batch['future_u'], norm)
                    current.update(native_physical=values['prediction'], native_final=values['final_state'])
                    work[batch_name + '/physical'] = values['work']
                else:
                    identity = {k: np.ones(6) if k.endswith('std') else np.zeros(6) for k in norm}
                    values = native.request(*normalized_inputs(batch, norm), identity)
                    current.update(native_standardized=values['prediction'], native_standard_final=values['final_state'])
                    work[batch_name + '/standardized'] = values['work']
            except (ValueError, RuntimeError) as error:
                errors.append({'batch': batch_name, 'backend': backend, 'type': type(error).__name__, 'message': str(error)})
            check()
        for label, a, b in (('physical', 'native_physical', 'torch_physical'),
                            ('standardized', 'native_standardized', 'torch_standardized'),
                            ('physical_final_state', 'native_final', 'torch_final'),
                            ('standardized_final_state', 'native_standard_final', 'torch_final')):
            if a in current and b in current:
                checks[batch_name + '/' + label] = comparison(current[a], current[b])
        arrays.update({batch_name + '/' + key: value for key, value in current.items()})
    return arrays, {'checks': checks, 'errors': errors, 'work': work,
                    'passed': not errors and len(checks) == 8 and all(c['passed'] for c in checks.values())}


def timed_pairs(torch_call, native_call, check, *, clock=time.perf_counter):
    records = []
    for phase, count in (('warmup', WARMUPS), ('timed', REPEATS)):
        for repetition in range(count):
            order = ('torch', 'native') if repetition % 2 == 0 else ('native', 'torch')
            row = {'phase': phase, 'repetition': repetition, 'order': list(order), 'seconds': {}}
            # Append before either call so an interrupted pair is preserved.
            records.append(row)
            try:
                for backend in order:
                    check()
                    start = clock()
                    (torch_call if backend == 'torch' else native_call)()
                    duration = clock() - start
                    require(math.isfinite(duration) and duration > 0, 'positive finite request duration')
                    row['seconds'][backend] = duration
                    check()
            except BaseException as error:
                error.timing_records = records
                raise
    return records


def summarize_timings(records):
    timed = [r for r in records if r['phase'] == 'timed']
    require(len(timed) == REPEATS and all(set(r['seconds']) == {'torch', 'native'} for r in timed), 'complete paired timing')
    medians = {backend: statistics.median(r['seconds'][backend] for r in timed) for backend in ('torch', 'native')}
    return {'median_seconds': medians, 'torch_over_native': medians['torch'] / medians['native'],
            'paired_ratios': [r['seconds']['torch'] / r['seconds']['native'] for r in timed]}


def benchmark(study, audit, engineering, qualification, output):
    from openjev.research.suspend_clock import SuspendClock
    output = Path(output).resolve()
    require(not output.is_relative_to(Path(study).resolve()), 'output must be outside the original study')
    output.mkdir(parents=True, exist_ok=False)
    clock = deadline = None
    inputs, parity, timings, resources = {}, [], [], []
    status, failure, result = 'FAILED', None, None
    native_proof = original = None
    try:
        clock = SuspendClock()
        deadline = clock.deadline_after(CAP_SECONDS)
        original = proof.authenticate(study, audit, engineering)
        native_proof = authenticate_native(qualification)
        inputs = {**original['inputs'], **native_proof['inputs']}
        require(all(os.environ.get(name) == '1' for name in THREADS), 'single-thread environment required')
        write(output / 'definition.json', {
            'version': VERSION, 'study': original['study'], 'inputs': inputs,
            'native': native_proof, 'arms': ARMS, 'seeds': SEEDS, 'context': 32, 'horizon': 128,
            'parity_comparisons': 30, 'parity_batch_sizes': [22, 1], 'rtol': RTOL, 'atol': ATOL,
            'warmup_pairs': WARMUPS, 'timed_pairs': REPEATS, 'cap_seconds': CAP_SECONDS,
            'clock': clock.backend, 'started_ns': deadline.started_ns, 'deadline_ns': deadline.expires_ns,
            'scope': 'No fitting or target scoring. Parity uses an additional native identity-normalizer request '
                     'to expose standardized forecasts; timing uses ordinary physical requests only. '
                     'Structured models initialize from the last two observed positions, with older context '
                     'and torques validated but unused. GRU conditions on the observed prefix. '
                     'This is functional implementation parity, not equal-history or architecture-quality evidence.'})
        deadline.check()
        import numpy as np
        import robot_structured_study as producer
        import torch

        from openjev.research.native_robot_gru import NativeRobotGRU
        from openjev.research.native_robot_transition import NativeRobotTransition
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        write(output / 'host.json', {'platform': platform.platform(), 'machine': platform.machine(),
            'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
            'threads': {name: os.environ.get(name) for name in THREADS},
            'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
            'host_scope': 'Shared host; unrelated workloads not stopped; no dedicated-host claim.'})
        study = Path(original['study'])
        cfg = original['plan']['config']
        require(cfg['context'] == 32 and cfg['dev_horizon'] == 128 and len(cfg['partitions']['dev']) == 2, 'registered request scope')
        selected = selected_fits(read(study / 'fits.json'), original['audit']['results']['selection']['selected_rates'])
        def load(path, keys=None):
            return load_pinned_arrays(path, inputs, keys)
        norm = load(study / 'normalizers.npz')
        require(set(norm) == {'q_mean', 'q_std', 'u_mean', 'u_std'} and all(
            a.dtype == np.float64 and a.shape == (6,) and np.isfinite(a).all() for a in norm.values())
            and np.all(norm['q_std'] > 0) and np.all(norm['u_std'] > 0), 'physical normalizers')
        linear = load(study / 'linear.npz')['coefficient']
        contexts = {}
        for name in cfg['partitions']['dev']:
            desc = original['plan']['data']['dev-data-' + name + '.npz']
            record = load(desc['path'], ('q', 'u'))
            starts = load(study / ('dev-windows-' + name + '.npz'), ('starts',))['starts']
            require(np.array_equal(starts, np.arange(64, len(record['q'])-159, 160, dtype=np.int64))
                    and len(starts) == 22, 'original22 window starts')
            contexts[name] = physical_contexts(record, starts)
        models = {}
        for fit in selected:
            deadline.check()
            arm, key = fit['arm'], fit['key']
            model = producer.model_for(arm, fit['seed'], linear)
            saved = load(study / key / 'final.npz')
            model.load_state_dict({name: torch.from_numpy(value) for name, value in saved.items()}, strict=True)
            model.zero_grad(set_to_none=True)
            model.requires_grad_(False)
            model.eval()
            native = NativeRobotGRU(model, native_proof['library']) if arm == 'gru32' else NativeRobotTransition(model.cell, native_proof['library'])
            models[key] = (model, native)
            resources.append({'fit_key': key, 'arm': arm, 'seed': fit['seed'], 'learning_rate': fit['learning_rate'],
                              'storage': native.storage(), 'retained_gradient_tensors': sum(p.grad is not None for p in model.parameters())})
            for name, physical in contexts.items():
                deadline.check()
                common = {'fit_key': key, 'arm': arm, 'seed': fit['seed'], 'learning_rate': fit['learning_rate'], 'recording': name}
                arrays, record = parity_pair(model, native, physical, norm, deadline.check)
                filename = f'parity-{name}-{key}.npz'
                np.savez_compressed(output / filename, **arrays)
                parity.append({**common, **record, 'file': filename})
                write(output / f'parity-{len(parity):02d}.json', parity[-1])
        write(output / 'parity.json', parity)
        require_all_parity(parity)
        write(output / 'parity-barrier.json', {'passed': True, 'comparisons': 30, 'timed_requests_started': 0})
        for fit in selected:
            model, native = models[fit['key']]
            for name, batch in contexts.items():
                physical = {key: value[:1].copy(order='K') for key, value in batch.items()}
                torch_call = lambda model=model, physical=physical: torch_request(model, physical, norm)
                native_call = lambda native=native, physical=physical: native.request(physical['q_context'], physical['u_context'], physical['future_u'], norm)
                row = {k: fit[k] for k in ('arm', 'seed', 'key', 'learning_rate')}
                row['recording'] = name
                try:
                    pairs = timed_pairs(torch_call, native_call, deadline.check)
                except BaseException as error:
                    row.update(pairs=getattr(error, 'timing_records', []), status='FAILED')
                    timings.append(row)
                    write(output / f'timing-{len(timings):02d}.json', row)
                    raise
                row.update(pairs=pairs, status='PASS', **summarize_timings(pairs))
                timings.append(row)
                write(output / f'timing-{len(timings):02d}.json', row)
        families = []
        for name in contexts:
            for arm in ARMS:
                rows = [r for r in timings if r['recording'] == name and r['arm'] == arm]
                medians = {b: statistics.median(r['median_seconds'][b] for r in rows) for b in ('torch', 'native')}
                families.append({'recording': name, 'arm': arm, 'fit_count': len(rows),
                                 'median_of_fit_medians_seconds': medians, 'torch_over_native': medians['torch']/medians['native']})
        ratios = []
        for name in contexts:
            values = {r['arm']: r['median_of_fit_medians_seconds']['native'] for r in families if r['recording'] == name}
            ratios.extend({'recording': name, 'numerator': 'householder', 'denominator': arm,
                           'native_latency_ratio': values['householder']/values[arm]} for arm in ARMS[1:])
        result = {'family_timings': families, 'native_family_ratios': ratios,
                  'scope': 'Descriptive complete-request local latency only; original61rule unchanged; no accuracy, novelty or confirmation claim. Structured context32 uses lasttwo positions; GRU conditions the prefix, so history computation is not matched.'}
        deadline.check()
        require(proof.authenticate(study, audit, engineering)['inputs'] == original['inputs'], 'original evidence unchanged after requests')
        require(authenticate_native(qualification) == native_proof, 'native evidence unchanged after requests')
        status = 'PASS'
    except BaseException as error:  # noqa: BLE001 - preserve original failed attempt
        failure = {'type': type(error).__name__, 'message': str(error)}
    finally:
        try:
            unchanged = all(pin(path) == expected for path, expected in inputs.items()) if inputs else False
        except Exception:  # noqa: BLE001 - a missing input must not erase the receipt
            unchanged = False
        if not unchanged:
            status = 'FAILED'
            failure = failure or {'type': 'InputChanged', 'message': 'Input bytes changed or admission incomplete'}
        elapsed, clock_error = None, None
        try:
            require(deadline is not None, 'deadline initialization failed')
            elapsed = deadline.elapsed_ns()/1e9
            if elapsed >= CAP_SECONDS:
                status = 'FAILED'
                failure = failure or {'type': 'TimeoutError', 'message': '900s cap exceeded'}
        except Exception as error:  # noqa: BLE001 - preserve original error if clock fails
            status, clock_error = 'FAILED', repr(error)
        write(output / 'summary.json', {'version': VERSION, 'status': status, 'error': failure,
            'parity': parity, 'timings': timings, 'resources': resources, 'result': result,
            'elapsed_seconds': elapsed, 'clock_error': clock_error, 'inputs_unchanged': unchanged,
            'confirmation_access': False, 'raw_measurement_access': False, 'target_scoring': False})
        files = {str(p.relative_to(output)): pin(p) for p in sorted(output.rglob('*')) if p.is_file()}
        write(output / 'manifest.json', {'files': files})
        write(output / 'receipt.json', {'version': VERSION, 'status': status, 'inputs': inputs,
              'manifest': pin(output / 'manifest.json'), 'elapsed_seconds': elapsed,
              'parity_comparisons': len(parity), 'timed_slots': len(timings), 'error': failure})
    return {'status': status, 'output': str(output), 'error': failure}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--engineering', required=True, type=Path)
    parser.add_argument('--qualification', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = benchmark(args.study, args.audit, args.engineering, args.qualification, args.output)
    print(json.dumps(result))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
