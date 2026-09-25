# SPDX-License-Identifier: GPL-3.0-or-later
"""One registered FIT-only NL-LFR fit, launched through run_nllfr_study.py.

This producer performs no DEV evaluation, context estimation or recipe selection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VERSION = 'fsm-author-nllfr-study-v1'
EXPERIMENT = {'order': 28, 'nz': 16, 'nw': 8, 'hidden_layers': 2, 'width': 64,
              'activation': 'relu', 'seed': 42, 'sigma': .0001, 'samples': 8192,
              'realizations': 6, 'periods': 2, 'fs': 6400., 'offset': 820,
              'frequency_bins': 4097, 'max_iter': 10000, 'rtol': .001, 'atol': .00001,
              'frequency_weighting': False, 'print_every': -1, 'outer_timeout_seconds': 18000,
              'rss_cap_bytes': 32*1024**3, 'pythonhashseed': '0'}
ENV = {**{k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                          'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')},
       'JAX_PLATFORMS': 'cpu', 'JAX_ENABLE_X64': 'True', 'PYTHONHASHSEED': '0',
       'PYTHONDONTWRITEBYTECODE': '1'}
REQUIRED = {'runtime_preflight', 'nllfr_runtime_receipt', 'nllfr_runtime_process',
            'nllfr_runtime_observation', 'nllfr_runtime_definition', 'nllfr_runtime_initialization',
            'bla_final_zip', 'bla_final_npz', 'bla_fit', 'bla_summary', 'bla_registration',
            'bla_process', 'bla_audit', 'bla_audit_process', 'producer_qualification', 'source_review'}
MIN_SOURCES = {'research/fsm_author/scripts/fit_nllfr.py',
               'research/fsm_author/scripts/run_nllfr_study.py',
               'research/fsm_author/src/openjev_fsm_author/nllfr.py',
               'research/fsm_author/src/openjev_fsm_author/benchmark.py',
               'research/fsm_author/src/openjev_fsm_author/linear_context.py',
               'src/openjev/research/fsm_data.py', 'research/fsm_author/uv.lock',
               'research/fsm_author/pyproject.toml'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError('nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    blob = Path(path).read_bytes()
    return {'path': str(Path(path).resolve()), 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def relative(value):
    require(isinstance(value, str) and not Path(value).is_absolute()
            and '\\' not in value and all(p not in ('', '.', '..') for p in value.split('/')),
            'canonical repository-relative path required')
    return ROOT/value


def event(output, phase, **fields):
    row = {'phase': phase, 'time_ns': time.time_ns(), **fields}
    with (output/'events.jsonl').open('a') as handle:
        handle.write(json.dumps(row, allow_nan=False)+'\n')
    print(json.dumps(row), flush=True)


def prerequisite(cfg, name):
    return relative(cfg['prerequisites'][name]['path'])


def metadata_admission(registration, output=None):
    cfg = read(registration)
    require(cfg['version'] == VERSION and cfg['experiment'] == EXPERIMENT, 'registration recipe drift')
    require(MIN_SOURCES <= set(cfg['source_sha256']), 'required source pins missing')
    require(REQUIRED <= set(cfg['prerequisites']), 'required prerequisite pins missing')
    for name, digest in cfg['source_sha256'].items():
        require(sha(relative(name)) == digest, 'source drift: '+name)
    for item in cfg['prerequisites'].values():
        require(sha(relative(item['path'])) == item['sha256'], 'prerequisite drift: '+item['path'])
    if output is not None:
        require(output.resolve() == relative(cfg['output']).resolve(), 'unregistered output')
    require(relative(cfg['output']).resolve() != relative(cfg['process_directory']).resolve(),
            'process and study directories must differ')
    for name in ('runtime_preflight', 'nllfr_runtime_receipt', 'nllfr_runtime_process',
                 'producer_qualification', 'source_review'):
        require(read(prerequisite(cfg, name))['status'] == 'PASS', 'failed prerequisite: '+name)
    probe = read(prerequisite(cfg, 'nllfr_runtime_process'))
    receipt = read(prerequisite(cfg, 'nllfr_runtime_receipt'))
    observation = read(prerequisite(cfg, 'nllfr_runtime_observation'))
    definition = read(prerequisite(cfg, 'nllfr_runtime_definition'))
    require(probe['sources_unchanged'] and len(probe['stages']) == 2
            and all(s['returncode'] == 0 for s in probe['stages']), 'synthetic original process failed')
    require(observation['observed_exit_code'] == 0
            and observation['process'] == pin(prerequisite(cfg, 'nllfr_runtime_process'))
            and observation['execution_receipt'] == pin(prerequisite(cfg, 'nllfr_runtime_receipt')),
            'synthetic terminal observation mismatch')
    require(probe['definition'] == receipt['definition'] == pin(prerequisite(cfg, 'nllfr_runtime_definition')),
            'synthetic definition mismatch')
    require(receipt['details']['fresh_process_initialization_equal'] is True
            and definition['fixture']['trainable_scalars'] == 7473, 'synthetic scope mismatch')
    for name, descriptor in definition['sources'].items():
        require(pin(relative(name)) == descriptor, 'synthetic qualified source changed: '+name)
    original_initial = probe['files']['fresh-initialization.json']
    require(original_initial == pin(prerequisite(cfg, 'nllfr_runtime_initialization')),
            'initialization qualification mismatch')
    audit = read(prerequisite(cfg, 'bla_audit'))
    process = read(prerequisite(cfg, 'bla_process'))
    audit_process = read(prerequisite(cfg, 'bla_audit_process'))
    require(audit['status'] == 'PASS' and audit['agreement'] is True
            and audit['results']['reference_status'] == 'REFERENCE_COMPLETE', 'BLA audit incomplete')
    require(process['status'] == 'completed' and process['observed_exit_code'] == 0
            and process['end_identity_matches'] is True, 'BLA original process incomplete')
    require(audit_process['observed_exit_code'] == 0
            and audit_process['audit_sha256'] == sha(prerequisite(cfg, 'bla_audit')),
            'BLA audit original process mismatch')
    require(read(prerequisite(cfg, 'bla_fit'))['status'] == 'complete'
            and read(prerequisite(cfg, 'bla_summary'))['status'] == 'REFERENCE_COMPLETE', 'BLA fit incomplete')
    require(process['registration_sha256'] == sha(prerequisite(cfg, 'bla_registration'))
            and process['fit.json_sha256'] == sha(prerequisite(cfg, 'bla_fit'))
            and process['summary.json_sha256'] == sha(prerequisite(cfg, 'bla_summary')), 'BLA receipt join')
    require(audit['inputs']['process'] == pin(prerequisite(cfg, 'bla_process')), 'BLA audited process join')
    require(audit['inputs']['registration'] == pin(prerequisite(cfg, 'bla_registration')), 'BLA registration join')
    for key, filename in (('bla_final_zip', 'final.zip'), ('bla_final_npz', 'final.npz'),
                          ('bla_fit', 'fit.json'), ('bla_summary', 'summary.json')):
        path = prerequisite(cfg, key)
        require(path.resolve() == (Path(audit['study'])/filename).resolve(), 'BLA artifact path join')
        require({k: pin(path)[k] for k in ('sha256', 'bytes')} == audit['inputs']['files'][filename],
                'BLA audited artifact join')
    require(sha(relative(cfg['data_path'])) == cfg['data_sha256'], 'data archive drift')
    return cfg


def runtime_identity(cfg, fss, jax):
    require(all(os.environ.get(k) == v for k, v in ENV.items()), 'fixed environment required')
    preflight = read(prerequisite(cfg, 'runtime_preflight'))
    require(platform.python_version() == preflight['python'] and sys.executable == preflight['executable'],
            'qualified Python required')
    versions = {k: importlib.metadata.version(k) for k in preflight['versions']}
    direct = read_text_json(importlib.metadata.distribution('freq-statespace').read_text('direct_url.json'))
    require(versions == preflight['versions'] and direct == preflight['upstream_direct_url'], 'runtime drift')
    folder = Path(fss.__file__).parent
    installed = {str(p.relative_to(folder)): {k: pin(p)[k] for k in ('sha256', 'bytes')}
                 for p in sorted(folder.rglob('*.py'))}
    require(installed == preflight['installed_source_matches'], 'installed vendor source drift')
    require(jax.config.jax_enable_x64 and all(d.platform == 'cpu' for d in jax.devices()), 'CPU float64 required')
    return {'status': 'PASS', 'python': platform.python_version(), 'executable': sys.executable,
            'versions': versions, 'direct_url': direct, 'installed_source_matches': installed,
            'environment': {k: os.environ.get(k) for k in ENV}, 'sources': cfg['source_sha256'],
            'prerequisites': cfg['prerequisites']}


def identities(cfg):
    """Public source/prerequisite/installed-runtime identity, also used by evaluation."""
    for name, digest in cfg['source_sha256'].items():
        require(sha(relative(name)) == digest, 'source drift: '+name)
    for item in cfg['prerequisites'].values():
        require(sha(relative(item['path'])) == item['sha256'], 'prerequisite drift: '+item['path'])
    import freq_statespace as fss
    import jax
    return runtime_identity(cfg, fss, jax)


def read_text_json(value):
    return json.loads(value)


def raw_arrays(model, np):
    arrays = {k: np.array(getattr(model, k), copy=True)
              for k in ('A', 'B_u', 'C_y', 'D_yu', 'B_w', 'C_z', 'D_yw', 'D_zu', 'ts')}
    arrays.update({k: np.array(getattr(model.norm, k), copy=True)
                   for k in ('u_mean', 'u_std', 'y_mean', 'y_std')})
    for i, layer in enumerate(model.func_static.model.layers):
        arrays[f'W{i}'] = np.array(layer.weight, copy=True)
        arrays[f'b{i}'] = np.array(layer.bias, copy=True)
    return arrays


def leaf_hash(value):
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest()}


def independent_x0(bla, U, np):
    z = np.exp(2j*np.pi*np.arange(4097)/8192)
    A, B = bla['A'], bla['B_u']
    state_spectrum = np.linalg.solve(z[:, None, None]*np.eye(28)-A,
                                    np.broadcast_to(B, (4097, 28, 3))) @ U
    return np.fft.irfft(state_spectrum, n=8192, axis=0)[-820]


def independent_loss(arrays, u, target_spectrum, x0, nllfr, np):
    extended = np.concatenate((u[-820:], u), axis=0)
    prediction, _, _ = nllfr.rollout(arrays, extended.transpose(2, 0, 1).copy(), x0.T.copy())
    spectrum = np.fft.rfft(prediction[:, 820:].transpose(1, 2, 0), axis=0)
    loss = float(np.sum(abs(target_spectrum-spectrum)**2)/(4097*6))
    require(np.isfinite(loss), 'nonfinite independently reconstructed FIT loss')
    return loss


def run(registration, output):
    cfg = metadata_admission(registration, output)
    import equinox as eqx
    import freq_statespace as fss
    import jax
    import numpy as np
    import optimistix as optx
    from freq_statespace import _nonlin_lfr

    from openjev_fsm_author import nllfr
    from openjev_fsm_author.benchmark import FIT_IDS, assemble_fit, export_model

    sys.path.insert(0, str(ROOT/'src'))
    from openjev.research import fsm_data

    before = runtime_identity(cfg, fss, jax)
    write(output/'identity-before.json', before)
    allowed = {relative(cfg['data_path']).resolve(), prerequisite(cfg, 'bla_final_zip').resolve(),
               prerequisite(cfg, 'bla_final_npz').resolve(),
               *(relative(item['path']).resolve() for item in cfg['prerequisites'].values())}

    def guard(event_name, args):
        if event_name == 'socket.connect':
            raise RuntimeError('network forbidden during registered FIT-only run')
        if event_name == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.suffix.lower() in ('.npz', '.npy', '.mat', '.zip'):
                require(path in allowed or path.is_relative_to(output), 'unregistered numerical file access')
    sys.addaudithook(guard)
    # All metadata admission completes before any numerical archive or model is decoded.
    model_bla = fss.load_model(prerequisite(cfg, 'bla_final_zip'))
    require(type(model_bla.ts) is float, 'sample interval must be Pythonfloat, not a trainable array')
    original = export_model(model_bla)
    require(original['A'].shape == (28, 28), 'BLA28 required')
    with np.load(prerequisite(cfg, 'bla_final_npz'), allow_pickle=False) as src:
        require(set(src.files) == set(original), 'BLA numeric roster')
        for key, value in original.items():
            np.testing.assert_array_equal(src[key], value)
    data = fsm_data.read_npz_estimation(relative(cfg['data_path']))
    require(data.source_sha256 == cfg['data_sha256'], 'decoded snapshot drift')
    records = data.partition('fit')
    require(tuple(r.record_id for r in records) == FIT_IDS, 'exact FIT roster required')
    u, y = assemble_fit(records)
    require(u.shape == y.shape == (8192, 3, 6, 2), 'native pooled FIT geometry')
    write(output/'admission.json', {'source_sha256': data.source_sha256, 'decoded_keys': list(data.decoded_keys),
          'fit_ids': list(FIT_IDS), 'optimization_records': 12, 'dev_evaluation_calls': 0,
          'decode_scope': 'Four100/200mV TRAIN members contain exposed FIT+DEV slices; only12FIT records enter normalization/objective',
          'reserved_scope': 'No300mV or official test member/header decoded',
          'original_bla_zip': pin(prerequisite(cfg, 'bla_final_zip'))})
    del data, records
    author_data = fss.create_data_object(u, y, np.arange(1, 3840), 6400.)
    for key in ('u_mean', 'u_std', 'y_mean', 'y_std'):
        np.testing.assert_array_equal(getattr(author_data.norm, key), original[key])
    network = fss.static.NeuralNetwork(nz=16, nw=8, layers=2, neurons_per_layer=64,
                                       activation=jax.nn.relu, seed=42, bias=True)
    initial = fss.nonlin.connect(model_bla, network, sigma=1e-4)
    fss.save_model(initial, output/'initial.zip')
    np.savez_compressed(output/'initial.npz', **raw_arrays(initial, np))
    initial_arrays = nllfr.export(initial)
    np.savez_compressed(output/'original-bla.npz', **original)
    synthetic = read(prerequisite(cfg, 'nllfr_runtime_initialization'))
    random_names = ('B_w', 'C_z', 'D_yw', 'D_zu', 'W0', 'b0', 'W1', 'b1', 'W2', 'b2')
    require({s: hash(s)&0xFFFFFFFF for s in ('neural_network', 'connect')} == synthetic['tag_hashes'],
            'random key tag identity mismatch')
    require(all(leaf_hash(initial_arrays[k]) == synthetic['arrays'][k] for k in random_names),
            'random initial components differ from qualification')
    theta, args = _nonlin_lfr._prepare_nonlin_optimization(author_data, initial, 820, False)
    leaves = jax.tree_util.tree_leaves(eqx.filter(theta, eqx.is_inexact_array))
    require(sum(v.size for v in leaves) == 7473 and all(v.dtype == np.float64 for v in leaves),
            'exact7473 CPUfloat64 trainable scalars required')
    fit_u, fit_y, U, Y = map(np.asarray, (author_data.time.u, author_data.time.y,
                                        author_data.freq.U, author_data.freq.Y))
    actual_x0 = np.asarray(args.x0)
    own_x0 = independent_x0(original, U, np)
    np.savez_compressed(output/'fit-data.npz', raw_fit_u=u, raw_fit_y=y, u=fit_u, y=fit_y, U=U, Y=Y,
                        f_idx=np.arange(1, 3840), training_x0=actual_x0, independent_x0=own_x0)
    np.testing.assert_allclose(actual_x0, own_x0, rtol=1e-10, atol=1e-10)
    require(args.u.shape == (9012, 3, 6) and Y.shape == (4097, 3, 6), 'full training geometry')
    initial_loss = independent_loss(initial_arrays, fit_u, Y, own_x0, nllfr, np)
    write(output/'initialization.json', {'random_component_hashes': {k: leaf_hash(initial_arrays[k]) for k in random_names},
          'trainable_scalars': 7473, 'trainable_leaves': len(leaves), 'initial_fit_loss': initial_loss,
          'training_x0_max_absolute_error': float(np.max(abs(actual_x0-own_x0))),
          'pythonhashseed': os.environ['PYTHONHASHSEED'], 'sample_interval_type': type(initial.ts).__name__})
    del theta, args, leaves, u, y
    event(output, 'optimization_started', initial_fit_loss=initial_loss, max_iter=10000)
    start = time.perf_counter()
    final, solve = fss.nonlin.optimize(initial, author_data, solver=optx.BFGS(rtol=1e-3, atol=1e-5),
        freq_weighting=False, max_iter=10000, print_every=-1, return_solve_details=True,
        offset=None, device='cpu')
    # Preserve returned state and raw traces before any validation can reject them.
    fss.save_model(final, output/'final.zip')
    np.savez_compressed(output/'final.npz', **raw_arrays(final, np))
    np.savez_compressed(output/'solver-trace.npz', loss_history=np.asarray(solve.loss_history),
        iter_times=np.asarray(solve.iter_times), iter_count=np.asarray(solve.iter_count),
        author_stop_flag=np.asarray(solve.converged), wall_time=np.asarray(solve.wall_time))
    elapsed = time.perf_counter()-start
    arrays = nllfr.export(final)
    require(solve.loss_history.shape == solve.iter_times.shape == (solve.iter_count,)
            and 1 <= solve.iter_count <= 10000 and np.isfinite(solve.loss_history).all()
            and np.isfinite(solve.iter_times).all(), 'invalid raw solver trace')
    for key, expected in original.items():
        np.testing.assert_array_equal(export_model(final._bla)[key], expected)
    for key in ('u_mean', 'u_std', 'y_mean', 'y_std', 'ts'):
        np.testing.assert_array_equal(arrays[key], original[key])
    reloaded = fss.load_model(output/'final.zip')
    np.savez_compressed(output/'roundtrip.npz', **raw_arrays(reloaded, np))
    np.savez_compressed(output/'roundtrip-original-bla.npz', **export_model(reloaded._bla))
    for actual, expected in ((nllfr.export(reloaded), arrays), (export_model(reloaded._bla), original)):
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
    write(output/'serialization.json', {'all_live_arrays_bitwise_equal': True,
          'original_bla_arrays_bitwise_equal': True, 'files': {name: pin(output/name) for name in
              ('final.zip', 'final.npz', 'original-bla.npz', 'roundtrip.npz', 'roundtrip-original-bla.npz')},
          'scope': 'Producer used pinned author ZIP loader; independent auditor can compare retained numeric exports'})
    final_loss = independent_loss(arrays, fit_u, Y, own_x0, nllfr, np)
    require(final_loss <= initial_loss, 'independent FIT objective worsened')
    fit = {'status': 'complete' if bool(solve.converged) else 'iteration_cap_reached',
           'author_stop_flag': bool(solve.converged), 'iterations': int(solve.iter_count),
           'initial_fit_loss': initial_loss, 'final_fit_loss': final_loss,
           'optimization_and_preservation_compile_inclusive_seconds': elapsed,
           'author_reported_seconds': float(solve.wall_time), 'trainable_scalars': 7473,
           'zip_roundtrip_all_numeric_arrays_bitwise_equal': True,
           'stop_meaning': 'Author BFGS small-change boolean only; no stationarity/global-optimum certificate',
           'discarded_vendor_warmup_steps': 1, 'peak_ru_maxrss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           'initial': pin(output/'initial.npz'), 'final': pin(output/'final.npz')}
    write(output/'fit.json', fit)
    after = runtime_identity(cfg, fss, jax)
    metadata_admission(registration, output)
    require(before == after, 'runtime/source identity drift')
    write(output/'identity-after.json', after)
    summary = {'status': 'FIT_COMPLETE' if fit['status'] == 'complete' else 'FIT_INCOMPLETE', 'fit': fit,
               'dev_evaluation_calls': 0, 'model_selection_calls': 0,
               'scope': 'One FIT-only author NL-LFR adaptation from own completed BLA28; evaluation remains separate'}
    write(output/'summary.json', summary)
    event(output, 'fit_closed', status=summary['status'], iterations=fit['iterations'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    metadata_admission(args.registration, output)
    require(all(os.environ.get(k) == v for k, v in ENV.items()), 'launch through fixed-environment supervisor')
    output.mkdir(parents=True, exist_ok=False)
    write(output/'started.json', {'registration': pin(args.registration), 'pid': os.getpid(), 'time_ns': time.time_ns()})
    cfg = read(args.registration)
    for name in cfg['source_sha256']:
        destination = output/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(relative(name).read_bytes())
    try:
        run(args.registration, output)
    except Exception as exc:
        write(output/'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        if not (output/'summary.json').exists():
            write(output/'summary.json', {'status': 'FIT_INCOMPLETE', 'error': f'{type(exc).__name__}: {exc}',
                                         'dev_evaluation_calls': 0, 'model_selection_calls': 0})
        raise


if __name__ == '__main__':
    main()
