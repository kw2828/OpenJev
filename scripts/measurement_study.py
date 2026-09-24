"""Frozen comparison of online linear measurements and packed raw memory."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import signal
import sys
import time
import tracemalloc
from dataclasses import fields
from pathlib import Path

import numpy as np
import scipy

from openjev.research import measurement_memory as memory
from openjev.research import retention_data as data_api

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('spectral118', 'dct118', 'bins118', 'coverage118', 'recent98', 'coverage98', 'full')
RAW = ('coverage118', 'recent98', 'coverage98')
HYBRID = {'spectral118': 'spectral', 'dct118': 'dct', 'bins118': 'bins'}
METRICS = ('regret', 'nll', 'brier', 'mse', 'coverage90', 'defer', 'always_defer_regret', 'risk_mae')
CONFIG = {
    'version': 'measurement-v1', 'namespace': 553260924, 'cohorts': 3,
    'evaluation_contexts': 128, 'queries': 4, 'state_budget_bytes': 1024,
    'populations': {'base': {'observations': 64, 'geometry': 'axial', 'offset': 100},
                    'shift': {'observations': 64, 'geometry': 'diagonal', 'offset': 200},
                    'long': {'observations': 192, 'geometry': 'axial', 'offset': 300},
                    'long_shift': {'observations': 192, 'geometry': 'diagonal', 'offset': 400}},
    'methods': list(METHODS), 'warmups': 3, 'timing_repeats': 10,
    'bootstrap_repeats': 1000, 'bootstrap_offsets': {'long': 900, 'long_shift': 901},
    'wall_cap_seconds': 1200,
}
SOURCES = ('src/openjev/research/measurement_memory.py',
           'src/openjev/research/retention_data.py',
           'scripts/measurement_study.py', 'scripts/audit_measurement_study.py',
           'tests/test_measurement_memory.py', 'tests/test_measurement_study.py',
           'tests/test_audit_measurement_study.py', 'research/measurement-protocol.md')


def descriptor(path):
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def register(out):
    out.mkdir(parents=True, exist_ok=False)
    pins = {}
    for name in SOURCES:
        target = out/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
        pins[name] = descriptor(target)
    write(out/'registration.json', {'config': CONFIG, 'sources': pins,
        'parent_result': descriptor(ROOT/'research/retention-results/summary.json'),
        'environment': {'python': sys.version, 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'platform': platform.platform(), 'threads': 1}})
    print(json.dumps({'registered': str(out), **descriptor(out/'registration.json')}), flush=True)


def verify(out):
    registration = json.loads((out/'registration.json').read_text())
    if registration['config'] != CONFIG or set(registration['sources']) != set(SOURCES):
        raise ValueError('frozen configuration/source roster changed')
    for name, pin in registration['sources'].items():
        if descriptor(ROOT/name) != pin or descriptor(out/'source'/name) != pin:
            raise ValueError('frozen source changed: '+name)
    if descriptor(ROOT/'research/retention-results/summary.json') != registration['parent_result']:
        raise ValueError('parent result changed')


def costs(risk):
    return np.concatenate((.02+risk, np.full((*risk.shape[:-1], 1), .20)), axis=-1)


def field_regrets(prediction, reference):
    selected = costs(prediction['risk']).argmin(-1)
    true_cost = costs(reference['risk'])
    return (np.take_along_axis(true_cost, selected[..., None], -1)[..., 0]-true_cost.min(-1)).mean(-1)


def metrics(prediction, reference, exposure):
    variance = prediction['variance']
    squared = (exposure-prediction['mean'])**2
    return {'regret': float(field_regrets(prediction, reference).mean()),
        'nll': float((.5*(math.log(2*math.pi)+np.log(variance)+squared/variance)).mean()),
        'brier': float(((prediction['risk']-(exposure > .5))**2).mean()),
        'mse': float(squared.mean()),
        'coverage90': float((squared <= 1.6448536269514722**2*variance).mean()),
        'defer': float((costs(prediction['risk']).argmin(-1) == 4).mean()),
        'always_defer_regret': float((.20-costs(reference['risk']).min(-1)).mean()),
        'risk_mae': float(np.abs(prediction['risk']-reference['risk']).mean())}


def retain(x, y, method):
    """Only public stream observations can enter a memory write."""
    if x.dtype != np.float64 or y.dtype != np.float64 or x.shape != (*y.shape, 2):
        raise ValueError('consistent float64 stream shapes required')
    for coordinates in x:
        if len(np.unique(coordinates, axis=0)) != len(coordinates):
            raise ValueError('unique grid locations per stream required')
    batch = y.shape[0]
    if method in HYBRID:
        state = memory.initial_hybrid(batch, kind=HYBRID[method])
        write_one = memory.write_hybrid
    elif method == 'coverage118':
        state = memory.initial_packed(batch)
        write_one = memory.write_packed
    elif method == 'recent98':
        state = memory.initial_recent(batch)
        write_one = memory.write_recent
    elif method == 'coverage98':
        state = memory.initial_recent(batch)
        write_one = memory.write_coverage98
    else:
        raise ValueError('unknown bounded memory')
    for step in range(y.shape[1]):
        state = write_one(state, x[:, step], y[:, step])
    return state


def predict(state, paths, method):
    if method in HYBRID:
        return memory.predict_hybrid(state, paths)
    if method == 'coverage118':
        return memory.predict_packed(state, paths)
    if method == 'recent98':
        return memory.predict_recent(state, paths)
    return memory.predict_recent(state, paths)


def evaluate(public, method):
    if method == 'full':
        pred = data_api.exact_reference(public['x'], public['y'], public['paths'])
        return {key: pred[key] for key in ('mean', 'variance', 'risk')}, None
    state = retain(public['x'], public['y'], method)
    return predict(state, public['paths'], method), state


def state_arrays(state):
    arrays = {field.name: getattr(state, field.name) for field in fields(state)
              if isinstance(getattr(state, field.name), np.ndarray)}
    arrays['step'] = np.array([state.step], dtype=np.int64)
    if hasattr(state, 'kind'):
        arrays['kind'] = np.array([('spectral', 'dct', 'bins').index(state.kind)], np.uint8)
    return arrays


def logical_bytes(state, public):
    if state is None:
        return (public['x'].nbytes+public['y'].nbytes)//len(public['y'])+40
    size = state.resident_bytes_per_context
    counted = sum(getattr(state, field.name).nbytes for field in fields(state)
                  if isinstance(getattr(state, field.name), np.ndarray))//len(public['y'])
    counted += 40+int(hasattr(state, 'kind'))
    if counted != size or size > CONFIG['state_budget_bytes']:
        raise ValueError('persistent array accounting or budget mismatch')
    return size


def bootstrap(differences, seed):
    values = np.asarray(differences, dtype=np.float64)
    if values.shape != (CONFIG['cohorts']*CONFIG['evaluation_contexts'],) or not np.isfinite(values).all():
        raise ValueError('complete finite paired field differences required')
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(values), size=(CONFIG['bootstrap_repeats'], len(values)))
    lower, upper = np.quantile(values[indices].mean(-1), [.025, .975], method='linear')
    return {'difference_mean': float(values.mean()), 'ci_low': float(lower), 'ci_high': float(upper),
            'context_differences': values.tolist(), 'seed': seed, 'repetitions': CONFIG['bootstrap_repeats']}


def conditions(rows, intervals):
    expected = {(p, c, m) for p in CONFIG['populations'] for c in range(CONFIG['cohorts']) for m in METHODS}
    keyed = {(r['population'], r['cohort'], r['method']): r['metrics'] for r in rows}
    if len(rows) != len(expected) or set(keyed) != expected:
        raise ValueError('complete unique metric roster required')
    if any(set(m) != set(METRICS) or not all(math.isfinite(v) for v in m.values()) for m in keyed.values()):
        raise ValueError('complete finite metrics required')
    if set(intervals) != set(CONFIG['bootstrap_offsets']):
        raise ValueError('complete interval roster required')
    result = []
    def add(name, actual, limit, strict=False):
        if not math.isfinite(actual) or not math.isfinite(limit):
            raise ValueError('finite condition inputs required')
        result.append({'name': name, 'actual': actual, 'limit': limit,
                       'comparison': '<' if strict else '<=', 'pass': actual < limit if strict else actual <= limit})
    def mean(p, m, metric):
        return float(np.mean([keyed[p, c, m][metric] for c in range(CONFIG['cohorts'])]))
    for p in ('base', 'shift'):
        for c in range(CONFIG['cohorts']):
            add(f'{p}/{c}/exact-prefix', keyed[p, c, 'spectral118']['regret'], 1e-8)
    for p in ('long', 'long_shift'):
        for m in RAW:
            add(f'{p}/mean-versus-{m}', mean(p, 'spectral118', 'regret'), .7*mean(p, m, 'regret')+1e-6)
    for p in CONFIG['populations']:
        add(f'{p}/nll', mean(p, 'spectral118', 'nll'), min(mean(p, m, 'nll') for m in RAW)+.02)
    for p in ('long', 'long_shift'):
        for c in range(CONFIG['cohorts']):
            add(f'{p}/{c}/best-raw', keyed[p, c, 'spectral118']['regret'],
                .9*min(keyed[p, c, m]['regret'] for m in RAW)+1e-6)
        add(f'{p}/defer', mean(p, 'spectral118', 'regret'), mean(p, 'spectral118', 'always_defer_regret'), True)
        add(f'{p}/paired-upper', intervals[p]['ci_high'], 0., True)
    return result


def resource(public, method):
    for _ in range(CONFIG['warmups']):
        evaluate(public, method)
    times = []
    for _ in range(CONFIG['timing_repeats']):
        start = time.perf_counter()
        _, state = evaluate(public, method)
        times.append((time.perf_counter()-start)*1000)
    size = logical_bytes(state, public)
    tracemalloc.start()
    try:
        evaluate(public, method)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {'logical_bytes': size, 'median_ms': float(np.median(times)), 'all_ms': times,
            'peak_python_bytes': peak}


def save_npz(path, arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def run(out):
    verify(out)
    with (out/'started.json').open('x') as stream:
        json.dump({'version': CONFIG['version'], 'registration': descriptor(out/'registration.json')}, stream)
    for directory in ('data', 'pred', 'state'):
        (out/directory).mkdir(exist_ok=False)
    start = time.monotonic()
    rows, resources, paired = [], [], {p: [] for p in CONFIG['bootstrap_offsets']}
    def timeout(*_):
        raise TimeoutError('Frozen measurement run cap')
    original = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, CONFIG['wall_cap_seconds'])
    try:
        for population, config in CONFIG['populations'].items():
            for cohort in range(CONFIG['cohorts']):
                generated = data_api.generate(CONFIG['namespace']+config['offset']+cohort,
                    CONFIG['evaluation_contexts'], config['observations'], config['geometry'], CONFIG['queries'])
                stem = f'{population}-{cohort}'
                save_npz(out/'data'/f'{stem}.npz', generated)
                public = {k: generated[k] for k in ('x', 'y', 'paths')}
                reference, _ = evaluate(public, 'full')
                field_scores = {}
                for method in METHODS:
                    prediction, state = (reference, None) if method == 'full' else evaluate(public, method)
                    save_npz(out/'pred'/f'{stem}-{method}.npz', prediction)
                    if state is not None:
                        save_npz(out/'state'/f'{stem}-{method}.npz', state_arrays(state))
                    logical_bytes(state, public)
                    rows.append({'population': population, 'cohort': cohort, 'method': method,
                                 'metrics': metrics(prediction, reference, generated['exposure'])})
                    field_scores[method] = field_regrets(prediction, reference)
                if population in paired:
                    paired[population].extend((field_scores['spectral118']-field_scores['coverage98']).tolist())
                print(json.dumps({'population': population, 'cohort': cohort, 'completed_methods': list(METHODS)}), flush=True)
                if cohort == 0:
                    first = {k: v[:1].copy() for k, v in public.items()}
                    for method in METHODS:
                        resources.append({'population': population, 'method': method, **resource(first, method)})
        intervals = {p: bootstrap(values, CONFIG['namespace']+CONFIG['bootstrap_offsets'][p]) for p, values in paired.items()}
        tests = conditions(rows, intervals)
        write(out/'bootstrap.json', intervals)
        write(out/'metrics.json', rows)
        write(out/'resources.json', resources)
        write(out/'summary.json', {'version': CONFIG['version'], 'rows': rows, 'resources': resources,
            'conditions': tests, 'passed': sum(c['pass'] for c in tests), 'total': len(tests),
            'gate': 'ADVANCE_CONSOLIDATION_BASELINE' if all(c['pass'] for c in tests) else 'DO_NOT_ADVANCE_CONSOLIDATION'})
        verify(out)
        write(out/'run-status.json', {'state': 'COMPLETE', 'elapsed_seconds': time.monotonic()-start,
                                     'metric_groups': len(rows), 'resources': len(resources)})
    except BaseException as error:
        write(out/'partial.json', {'rows': rows, 'resources': resources, 'paired': paired})
        write(out/'run-status.json', {'state': 'FAILED', 'elapsed_seconds': time.monotonic()-start,
                                     'error_type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, original)
        files = {str(path.relative_to(out)): descriptor(path) for path in sorted(out.rglob('*')) if path.is_file()}
        write(out/'manifest.json', {'files': files})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('register', 'run'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    if args.mode == 'run' and args.registration_sha256 != descriptor(args.out/'registration.json')['sha256']:
        raise ValueError('registration SHA256 required')
    (register if args.mode == 'register' else run)(args.out.resolve())


if __name__ == '__main__':
    main()
