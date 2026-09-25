# SPDX-License-Identifier: GPL-3.0-or-later
"""One registered measured BLA fit. Run only through supervise_bla.py."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'true'
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))

import jax  # noqa: E402

jax.config.update('jax_enable_x64', True)
import freq_statespace as fss  # noqa: E402
import numpy as np  # noqa: E402
import optimistix as optx  # noqa: E402
from openjev.research import fsm_data  # noqa: E402

from openjev_fsm_author.benchmark import (  # noqa: E402
    DEV_IDS,
    FIT_IDS,
    assemble_fit,
    export_model,
    physical_request,
    requests,
    require,
    score,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def event(output, phase, **extra):
    row = {'phase': phase, 'time_ns': time.time_ns(), **extra}
    with (output/'events.jsonl').open('a') as handle:
        handle.write(json.dumps(row, allow_nan=False)+'\n')
    print(json.dumps(row), flush=True)


def identities(cfg):
    for path, digest in cfg['source_sha256'].items():
        require(sha(ROOT/path) == digest, 'source drift: '+path)
    for item in cfg['prerequisites'].values():
        require(sha(ROOT/item['path']) == item['sha256'], 'prerequisite drift: '+item['path'])
    preflight = json.loads((ROOT/cfg['prerequisites']['runtime_preflight']['path']).read_text())
    require(preflight['status'] == 'PASS', 'runtime prerequisite failed')
    installed = Path(fss.__file__).parent
    actual = {str(path.relative_to(installed)): sha(path) for path in installed.rglob('*.py')}
    require(actual == {name: item['sha256'] for name, item in preflight['installed_source_matches'].items()},
            'installed author source differs from qualified source')
    dist = importlib.metadata.distribution('freq-statespace')
    require(json.loads(dist.read_text('direct_url.json')) == preflight['upstream_direct_url'],
            'installed author revision changed')
    versions = {name: importlib.metadata.version(name) for name in preflight['versions']}
    require(versions == preflight['versions'], 'installed runtime versions changed')
    return {'status': 'PASS', 'source_sha256': cfg['source_sha256'],
            'installed_author_sha256': actual, 'versions': versions,
            'prerequisites': cfg['prerequisites'],
            'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'XLA_FLAGS', 'JAX_PLATFORMS', 'JAX_ENABLE_X64')}}


def frequency_error(arrays, target):
    z = np.exp(2j*np.pi*np.arange(1, 3840)/8192)
    a, b, c, d = (arrays[n] for n in ('A', 'B_u', 'C_y', 'D_yu'))
    response = c @ np.linalg.solve(z[:, None, None]*np.eye(len(a))-a,
                                  np.broadcast_to(b, (len(z), *b.shape)))+d
    value = float(np.mean(np.abs(response-target)**2))
    require(np.isfinite(value), 'nonfinite independent frequency loss')
    return value


def evaluate(arrays, records, norm, output):
    directory = output/'evaluation'
    directory.mkdir()
    require(tuple(r.record_id for r in records) == DEV_IDS, 'DEV roster mismatch')
    starts = np.arange(0, 7937, 256, dtype=np.int64)
    rows = []
    for record in records:
        try:
            inputs = requests(record, starts)
            prediction, state, diagnostic = physical_request(arrays, *inputs)
            target = np.stack([record.y[s+100:s+228] for s in starts])
            metrics = score(prediction, target, norm['y_scale'])
            # Archive scores in the parent common normalized coordinate system.
            np.savez_compressed(directory/(record.record_id+'.npz'),
                prediction=(prediction-norm['y_mean'])/norm['y_scale'],
                target=(target-norm['y_mean'])/norm['y_scale'], starts=starts,
                final_state=state, singular_values=diagnostic['singular_values'],
                context_residual_norm=diagnostic['residual_norm'],
                y_context=inputs[0], u_context=inputs[1], future_u=inputs[2])
            rows.append({'record_id': record.record_id, 'status': 'complete', **metrics,
                         'rank': diagnostic['rank'], 'rank_deficient': diagnostic['rank_deficient']})
        except Exception as exc:  # noqa: BLE001
            rows.append({'record_id': record.record_id, 'status': 'failed',
                         'error': f'{type(exc).__name__}: {exc}'})
        event(output, 'record_closed', **rows[-1])
    costs, timing_error = [], None
    try:
        physical_request(arrays, *requests(records[0], starts[:1]))
        for record in records:
            for start in (0, 7936):
                before = time.perf_counter()
                value = physical_request(arrays, *requests(record, np.array([start], dtype=np.int64)))[0]
                costs.append((time.perf_counter()-before)*1000)
                require(np.isfinite(value).all(), 'nonfinite timed output')
    except Exception as exc:  # noqa: BLE001
        timing_error = f'{type(exc).__name__}: {exc}'
    result = {'rows': rows, 'request_ms': costs, 'timing_error': timing_error,
              'median_request_ms': float(np.median(costs)) if len(costs) == 24 and timing_error is None else None,
              'persistent_numeric_bytes': sum(a.nbytes for a in arrays.values())+28*8+3*8,
              'storage_scope': 'all nine numeric arrays including normalization/ts, 28-state request, C/H/rcond scalars; no cached observability matrix',
              'timing_scope': 'NumPy CPU request slicing, validation, normalization, fresh observability least-squares solve, state propagation, H128 rollout, diagnostics, denormalization; no target reads'}
    write(output/'evaluation.json', result)
    return result


def run(registration, output):
    cfg = json.loads(registration.read_text())
    require(cfg['version'] == 'fsm-author-bla-study-v1', 'wrong registration')
    require(cfg['experiment'] == {'order': 28, 'nq': 29, 'max_iter': 5000, 'rtol': 0.001,
            'atol': 0.00001, 'frequency_weighting': False, 'context': 100, 'horizon': 128,
            'stride': 256, 'outer_timeout_seconds': 1800}, 'experiment drift')
    write(output/'identity-before.json', identities(cfg))
    require(all(device.platform == 'cpu' for device in jax.devices())
            and jax.config.jax_enable_x64, 'CPU float64 required')
    data_path = ROOT/cfg['data_path']
    require(sha(data_path) == cfg['data_sha256'], 'measurement archive drift')
    event(output, 'admitting_exposed_fit_dev')
    data = fsm_data.read_npz_estimation(data_path)
    require(data.source_sha256 == cfg['data_sha256'], 'decoded snapshot drift')
    u, y = assemble_fit(data.partition('fit'))
    require(u.shape == y.shape == (8192, 3, 6, 2), 'native pooled shape required')
    write(output/'admission.json', {'source_sha256': data.source_sha256,
          'decoded_keys': list(data.decoded_keys), 'fit_ids': list(FIT_IDS), 'dev_ids': list(DEV_IDS),
          'author_preprocessing': 'FIT channel normalization, period averaging, unweighted pooled triplets',
          'reserved_data': '300mV and all official test members and headers remain undecoded'})
    fit_start = time.perf_counter()
    author_data = fss.create_data_object(u, y, np.arange(1, 3840), 6400.0)
    event(output, 'subspace_started')
    initial = fss.lin.subspace_id(author_data, nx=28, nq=29, freq_weighting=False,
                                 input_output_mode=False, logging_enabled=False)
    initial_arrays = export_model(initial)
    fss.save_model(initial, output/'initial.zip')
    np.savez_compressed(output/'initial.npz', **initial_arrays)
    target = np.asarray(author_data.freq.G_bla.G)
    np.savez_compressed(output/'frequency-target.npz', target=target, indices=np.arange(1, 3840))
    initial_loss = frequency_error(initial_arrays, target)
    event(output, 'refinement_started', initial_frequency_mse=initial_loss)
    model, details = fss.lin.optimize(initial, author_data,
        solver=optx.BFGS(rtol=1e-3, atol=1e-5), freq_weighting=False,
        input_output_mode=False, max_iter=5000, print_every=-1,
        return_solve_details=True, device='cpu')
    # Preserve returned values before finiteness/quality validation can fail.
    fss.save_model(model, output/'final.zip')
    raw = {k: np.asarray(getattr(model, k)) for k in ('A', 'B_u', 'C_y', 'D_yu', 'ts')}
    raw.update({k: np.asarray(getattr(model.norm, k)) for k in ('u_mean', 'u_std', 'y_mean', 'y_std')})
    np.savez_compressed(output/'final.npz', **raw)
    history = np.asarray(details.loss_history)
    iteration_times = np.asarray(details.iter_times)
    np.savez_compressed(output/'solver-trace.npz', loss_history=history, iter_times=iteration_times)
    arrays = export_model(model)
    final_loss = frequency_error(arrays, target)
    require(history.shape == iteration_times.shape == (int(details.iter_count),) and 1 <= len(history) <= 5000
            and np.isfinite(history).all() and np.isfinite(iteration_times).all(), 'invalid optimizer trace')
    require(final_loss <= initial_loss, 'refinement worsened frequency fit')
    fit = {'status': 'complete' if bool(details.converged) else 'iteration_cap_reached',
           'author_stop_flag': bool(details.converged), 'iterations': int(details.iter_count),
           'stop_meaning': 'BFGS Cauchy small-change flag; not a stationarity or global-optimum certificate',
           'initial_frequency_mse': initial_loss, 'final_frequency_mse': final_loss,
           'outer_fit_seconds': time.perf_counter()-fit_start,
           'author_reported_seconds': float(details.wall_time),
           'spectral_radius': float(np.max(np.abs(np.linalg.eigvals(arrays['A'])))),
           'final_sha256': sha(output/'final.npz'), 'initial_sha256': sha(output/'initial.npz')}
    write(output/'fit.json', fit)
    event(output, 'fit_closed', **fit)
    # A capped but finite fit may be scored diagnostically, never passed as a completed reference.
    norm_item = cfg['prerequisites']['common_normalizer']
    with np.load(ROOT/norm_item['path'], allow_pickle=False) as src:
        norm = {key: src[key].copy() for key in ('u_mean', 'u_scale', 'y_mean', 'y_scale')}
    result = evaluate(arrays, data.partition('dev'), norm, output)
    complete = len(result['rows']) == 12 and all(r['status'] == 'complete' for r in result['rows'])
    summary = {'status': 'REFERENCE_COMPLETE' if fit['status'] == 'complete' and complete
               and result['median_request_ms'] is not None else 'REFERENCE_INCOMPLETE',
               'fit': fit, 'mean_rmse': float(np.mean([r['rmse'] for r in result['rows']])) if complete else None,
               'median_request_ms': result['median_request_ms'],
               'persistent_numeric_bytes': result['persistent_numeric_bytes'],
               'scope': 'single deterministic pooled FIT-only author BLA28 adaptation, exposed DEV; no new neural updates or reserve access'}
    write(output/'summary.json', summary)
    write(output/'identity-after.json', identities(cfg))
    event(output, 'study_closed', **summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
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
