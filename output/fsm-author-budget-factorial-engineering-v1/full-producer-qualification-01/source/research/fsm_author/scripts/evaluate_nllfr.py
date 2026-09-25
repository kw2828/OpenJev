# SPDX-License-Identifier: GPL-3.0-or-later
"""Frozen exposed-DEV evaluation of one closed FIT-only NL-LFR attempt."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
ENV = {**{k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                          'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')},
       'JAX_PLATFORMS': 'cpu', 'JAX_ENABLE_X64': 'True', 'PYTHONHASHSEED': '0',
       'PYTHONDONTWRITEBYTECODE': '1'}

import numpy as np  # noqa: E402
from openjev.research import fsm_data  # noqa: E402

from openjev_fsm_author.benchmark import DEV_IDS, requests, require, score  # noqa: E402
from openjev_fsm_author.nllfr import physical_rollout, validate  # noqa: E402
from openjev_fsm_author.nllfr_context import condition  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_value, allow_nan=False)+'\n')


def event(output, phase, **extra):
    row = {'phase': phase, 'time_ns': time.time_ns(), **extra}
    with (output/'events.jsonl').open('a') as handle:
        handle.write(json.dumps(row, default=json_value, allow_nan=False)+'\n')
    print(json.dumps(row, default=json_value), flush=True)


def identities(cfg):
    require(all(os.environ.get(k) == v for k, v in ENV.items()), 'fixed evaluation environment required')
    for path, digest in cfg['source_sha256'].items():
        require(sha(ROOT/path) == digest, 'source drift: '+path)
    for item in cfg['prerequisites'].values():
        require(sha(ROOT/item['path']) == item['sha256'], 'prerequisite drift: '+item['path'])
    preflight = json.loads((ROOT/cfg['prerequisites']['runtime_preflight']['path']).read_text())
    require(preflight['status'] == 'PASS' and platform.python_version() == preflight['python']
            and sys.executable == preflight['executable'], 'qualified Python required')
    versions = {name: importlib.metadata.version(name) for name in preflight['versions']}
    require(versions == preflight['versions'], 'runtime versions changed')
    dist = importlib.metadata.distribution('freq-statespace')
    direct = json.loads(dist.read_text('direct_url.json'))
    require(direct == preflight['upstream_direct_url'], 'author revision changed')
    installed = Path(dist.locate_file('freq_statespace'))
    sources = {str(p.relative_to(installed)): sha(p) for p in installed.rglob('*.py')}
    require(sources == {k: v['sha256'] for k, v in preflight['installed_source_matches'].items()},
            'installed author source changed')
    return {'source_sha256': cfg['source_sha256'], 'prerequisites': cfg['prerequisites'],
            'python': platform.python_version(), 'executable': sys.executable,
            'versions': versions, 'installed_author_sha256': sources, 'direct_url': direct,
            'thread_environment': {k: os.environ.get(k) for k in (*ENV, 'XLA_FLAGS')}}


def authenticate(cfg, registration):
    """Reject an unclosed or altered original fit before opening model arrays."""
    identities(cfg)
    source = ROOT/cfg['output']
    process_path = ROOT/cfg['process_directory']/'process.json'
    process = json.loads(process_path.read_text())
    require(process['phase'] == 'fit' and process['status'] == 'completed' and process['observed_exit_code'] == 0,
            'original fit process must be closed with exit zero')
    require(process['registration_sha256'] == sha(registration), 'fit registration mismatch')
    require(process['end_identity_matches'], 'fit identity closure failed')
    require(process['log_sha256'] == sha(process_path.with_name('process.log')), 'fit log drift')
    command = [str(ROOT/'research/fsm_author/.venv/bin/python'),
               str(ROOT/'research/fsm_author/scripts/fit_nllfr.py'), '--registration',
               str(registration.resolve()), '--output', str(source.resolve())]
    require(process['command'] == command, 'original fit command mismatch')
    require(process['timeout_seconds'] == cfg['experiment']['outer_timeout_seconds']
            and process['rss_cap_bytes'] == cfg['experiment']['rss_cap_bytes'], 'fit resource policy mismatch')
    require(process['environment'] == ENV, 'original fit environment mismatch')
    require(process['producer'] == pin(ROOT/'research/fsm_author/scripts/fit_nllfr.py')
            and process['supervisor'] == pin(ROOT/'research/fsm_author/scripts/run_nllfr_study.py'),
            'original executable source mismatch')
    inventory = {}
    for path in sorted(source.rglob('*')):
        require(not path.is_symlink(), 'symlink fit evidence')
        if path.is_file():
            inventory[str(path.relative_to(source))] = {k: pin(path)[k] for k in ('sha256', 'bytes')}
    require(inventory == process['artifacts'], 'original full fit inventory mismatch')
    require({'final.npz', 'final.zip', 'fit.json', 'summary.json'} <= inventory.keys(), 'fit artifacts missing')
    fit = json.loads((source/'fit.json').read_text())
    require(fit['status'] in ('complete', 'iteration_cap_reached'), 'no finite fit to evaluate')
    return source, fit, {'process_sha256': sha(process_path), 'process': process,
                        'fit_sha256': sha(source/'fit.json'), 'registration_sha256': sha(registration)}


def request(arrays, record, start):
    """One uncached call; all context work and diagnostics are inside the timer."""
    before = time.perf_counter()
    inputs = requests(record, np.array([start], dtype=np.int64))
    sliced = time.perf_counter()
    forecast_state, diagnostics = condition(arrays, inputs[0], inputs[1])
    conditioned = time.perf_counter()
    prediction, final_state = physical_rollout(arrays, inputs[2], forecast_state)
    ended = time.perf_counter()
    timing = {'request_ms': (ended-before)*1000, 'slice_ms': (sliced-before)*1000,
              'initializer_ms': (conditioned-sliced)*1000, 'rollout_ms': (ended-conditioned)*1000}
    return prediction, final_state, forecast_state, diagnostics, inputs, timing


def numerical_failure(error):
    if isinstance(error, (FloatingPointError, np.linalg.LinAlgError)):
        return True
    return type(error) is ValueError and str(error) in {
        'nonfinite linear seed system', 'nonfinite linear seed', 'nonfinite scaled gradient',
        'nonfinite context direction', 'nonfinite final Jacobian diagnostics',
        'nonfinite physical outputs', 'nonfinite score inputs', 'score overflow'}


def evaluate(arrays, records, norm, output):
    require(tuple(r.record_id for r in records) == DEV_IDS, 'exact DEV roster required')
    nx = validate(arrays)[0]
    require(set(norm) == {'y_mean', 'y_scale'} and all(isinstance(v, np.ndarray)
            and v.shape == (3,) and v.dtype == np.float64 and np.isfinite(v).all()
            for v in norm.values()) and (norm['y_scale'] > 0).all(), 'invalid common normalizer')
    before_arrays = {k: v.copy() for k, v in arrays.items()}
    starts = np.arange(0, 7937, 256, dtype=np.int64)
    directory = output/'evaluation'
    directory.mkdir()
    rows, request_rows, status_counts = [], [], Counter()
    for record in records:
        own_dir = directory/record.record_id
        own_dir.mkdir()
        predictions, targets, states, inputs_list = [], [], [], []
        for start in starts:
            s = int(start)
            label = f'{s:04d}'
            # Retain the legal request even if numerical inference subsequently fails.
            # Evidence preparation is outside the fresh full-request timer below.
            attempted_inputs = requests(record, np.array([s], dtype=np.int64))
            np.savez_compressed(own_dir/(label+'.inputs.npz'),
                starts=np.array([s], dtype=np.int64), y_context=attempted_inputs[0],
                u_context=attempted_inputs[1], future_u=attempted_inputs[2])
            try:
                pred, final, forecast_state, diagnostic, inputs, timing = request(arrays, record, s)
                # Future target is read only after the causal request has returned.
                target = record.y[s+100:s+228][None].copy()
                metrics = score(pred, target, norm['y_scale'])
                np.savez_compressed(own_dir/(label+'.npz'),
                    prediction=(pred-norm['y_mean'])/norm['y_scale'],
                    target=(target-norm['y_mean'])/norm['y_scale'],
                    starts=np.array([s], dtype=np.int64), final_state=final,
                    forecast_state=forecast_state, linear_seed_states=diagnostic['linear_seed_states'],
                    solved_context_start_states=diagnostic['solved_context_start_states'],
                    y_context=inputs[0], u_context=inputs[1], future_u=inputs[2])
                retained = (diagnostic['linear_seed_states'], diagnostic['solved_context_start_states'],
                            forecast_state, final)
                row = {'record_id': record.record_id, 'start': s, 'status': 'complete',
                       **metrics, **timing, 'context': diagnostic,
                       'retained_state_max_abs': max(float(np.abs(v).max()) for v in retained)}
                predictions.append(pred)
                targets.append(target)
                states.append(final)
                inputs_list.append(inputs)
                status_counts[diagnostic['requests'][0]['status']] += 1
            except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
                if not numerical_failure(exc):
                    raise
                row = {'record_id': record.record_id, 'start': s, 'status': 'failed',
                       'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()}
            write(own_dir/(label+'.json'), row)
            request_rows.append({k: row[k] for k in ('record_id', 'start', 'status')})
        complete = len(predictions) == 32
        if complete:
            pred, target = np.concatenate(predictions), np.concatenate(targets)
            metrics = score(pred, target, norm['y_scale'])
            np.savez_compressed(directory/(record.record_id+'.npz'),
                prediction=(pred-norm['y_mean'])/norm['y_scale'],
                target=(target-norm['y_mean'])/norm['y_scale'], starts=starts,
                final_state=np.concatenate(states),
                y_context=np.concatenate([v[0] for v in inputs_list]),
                u_context=np.concatenate([v[1] for v in inputs_list]),
                future_u=np.concatenate([v[2] for v in inputs_list]))
            row = {'record_id': record.record_id, 'status': 'complete', **metrics}
        else:
            row = {'record_id': record.record_id, 'status': 'incomplete',
                   'completed_requests': len(predictions), 'expected_requests': 32}
        rows.append(row)
        require(all(np.array_equal(arrays[k], before_arrays[k]) for k in arrays), 'model mutation')
        event(output, 'record_closed', **row)
    timings, timing_error = [], None
    try:
        request(arrays, records[0], 0)  # One declared untimed warmup, no retained workspace.
        for record in records:
            for start in (0, 7936):
                values = request(arrays, record, start)
                timings.append({'record_id': record.record_id, 'start': start, **values[-1]})
    except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
        if not numerical_failure(exc):
            raise
        timing_error = f'{type(exc).__name__}: {exc}'
    require(all(np.array_equal(arrays[k], before_arrays[k]) for k in arrays), 'timing mutated model')
    result = {'rows': rows, 'requests': request_rows, 'context_status_counts': dict(status_counts),
              'timings': timings, 'timing_error': timing_error,
              'median_request_ms': float(np.median([r['request_ms'] for r in timings]))
                  if len(timings) == 24 and timing_error is None else None,
              'persistent_numeric_bytes': sum(v.nbytes for v in arrays.values())+nx*8+9*8,
              'storage_scope': 'all 19 final numeric arrays, 28 retained state scalars, nine C/H/solver-policy scalars; no original BLA or cached factorization',
              'transient_scope': 'per-request seed/trajectory/Jacobian/SVD and line-search workspace; Python container overhead excluded',
              'timing_scope': 'CPU NumPy slicing, validation, normalization, linear seed, every nonlinear and Jacobian trial, final diagnostic SVD, H128 rollout and denormalization; target reads/artifact writes excluded',
              'retained_state_max_scope': 'seed, fitted context start, forecast start and horizon end; not maximum over every internal trajectory state'}
    write(output/'evaluation.json', result)
    return result


def run(registration, output):
    cfg = json.loads(registration.read_text())
    require(cfg['version'] == 'fsm-author-nllfr-study-v1', 'wrong registration')
    require(output.resolve() == (ROOT/cfg['evaluation_output']).resolve(), 'unregistered evaluation output')
    require(cfg['evaluation'] == {'context': 100, 'horizon': 128, 'stride': 256,
            'requests_per_record': 32, 'records': 12, 'warmup_requests': 1, 'timed_requests': 24,
            'outer_timeout_seconds': 3600, 'rss_cap_bytes': 32*1024**3}, 'evaluation policy drift')
    source, fit, provenance = authenticate(cfg, registration)
    write(output/'fit-provenance.json', provenance)
    write(output/'identity-before.json', identities(cfg))
    with np.load(source/'final.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    require(validate(arrays) == (28, 16, 8, 64), 'wrong final architecture')
    item = cfg['prerequisites']['common_normalizer']
    with np.load(ROOT/item['path'], allow_pickle=False) as archive:
        norm = {k: archive[k].copy() for k in ('y_mean', 'y_scale')}
    require(sha(ROOT/cfg['data_path']) == cfg['data_sha256'], 'measurement archive drift')
    data = fsm_data.read_npz_estimation(ROOT/cfg['data_path'])
    require(data.source_sha256 == cfg['data_sha256'], 'decoded snapshot drift')
    write(output/'admission.json', {'decoded_keys': list(data.decoded_keys),
          'dev_ids': list(DEV_IDS), 'scope': 'previously exposed 100/200mV DEV; no new weight updates; 300mV and official-test members and headers remain unopened'})
    result = evaluate(arrays, data.partition('dev'), norm, output)
    complete = len(result['rows']) == 12 and all(r['status'] == 'complete' for r in result['rows'])
    summary = {'status': 'REFERENCE_COMPLETE' if complete and fit['status'] == 'complete'
               and result['median_request_ms'] is not None else 'REFERENCE_INCOMPLETE',
               'fit_status': fit['status'],
               'mean_rmse': float(np.mean([r['rmse'] for r in result['rows']])) if complete else None,
               'median_request_ms': result['median_request_ms'],
               'persistent_numeric_bytes': result['persistent_numeric_bytes'],
               'context_status_counts': result['context_status_counts'],
               'scope': 'single-seed author-method adaptation on exposed DEV; no untouched generalization or architecture-superiority claim'}
    write(output/'summary.json', summary)
    _, _, after_provenance = authenticate(cfg, registration)
    require(after_provenance == provenance, 'fit changed during evaluation')
    write(output/'identity-after.json', identities(cfg))
    event(output, 'evaluation_closed', **summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT, 'run from OpenJev root')
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'started.json', {'registration_sha256': sha(args.registration), 'pid': os.getpid()})
    try:
        run(args.registration, args.output)
    except Exception as exc:  # noqa: BLE001
        write(args.output/'failure.json', {'type': type(exc).__name__, 'message': str(exc),
                                          'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
