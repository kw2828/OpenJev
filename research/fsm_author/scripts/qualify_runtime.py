# SPDX-License-Identifier: GPL-3.0-or-later
"""One retained, fabricated-only qualification of the pinned author runtime."""
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
from datetime import datetime, timezone
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[1]
ROOT = COMPONENT.parents[1]
COMMIT = 'a79e8c567b018a6c9462528fc1e10b77fd19b3e2'
PREFLIGHT = ROOT / 'output/fsm-author-engineering-v1/runtime-preflight-01/receipt.json'
SOURCES = (
    Path(__file__).resolve(), COMPONENT / 'pyproject.toml', COMPONENT / 'uv.lock',
    COMPONENT / 'src/openjev_fsm_author/linear_context.py',
    ROOT / 'research/fsm-author-runtime-contract.md',
)
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def pin(path):
    path = Path(path)
    payload = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest()}


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / 'source'
    snapshots.mkdir()
    sources = {}
    for source in SOURCES:
        relative = source.relative_to(ROOT)
        target = snapshots / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        sources[str(relative)] = pin(source)
    definition = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'sources': sources, 'runtime_preflight': pin(PREFLIGHT),
        'command': [str(COMPONENT / '.venv/bin/python'), str(SOURCES[0]),
                    '--execute', '--output', str(output)],
        'environment': {**{key: '1' for key in THREADS},
                        'JAX_PLATFORMS': 'cpu', 'JAX_ENABLE_X64': 'True',
                        'PYTHONDONTWRITEBYTECODE': '1'},
        'scope': 'Fabricated order4 only; no measurements, pretrained weights or network',
        'fixture': {'samples': 256, 'channels': 3, 'realizations': 6, 'periods': 2,
                    'frequency_indices': [1, 63], 'fs': 256.0, 'order': 4, 'nq': 5,
                    'direct_term_perturbation': .02, 'BFGS_max_iter': 25,
                    'BFGS_rtol': 1e-3, 'BFGS_atol': 1e-5,
                    'frequency_weighting': False, 'input_output_mode': False,
                    'subspace_atol_rtol': 1e-6, 'simulation_atol_rtol': 1e-10},
        'outer_timeout_seconds': 600,
    }
    write_json(output / 'definition.json', definition)
    print(json.dumps({'status': 'PREPARED', 'definition': pin(output / 'definition.json')}))


def check_pins(definition, output):
    for relative, expected in definition['sources'].items():
        if pin(ROOT / relative) != expected:
            raise ValueError(f'changed source: {relative}')
        snapshot = pin(output / 'source' / relative)
        if (snapshot['bytes'], snapshot['sha256']) != (expected['bytes'], expected['sha256']):
            raise ValueError(f'changed snapshot: {relative}')
    if pin(PREFLIGHT) != definition['runtime_preflight']:
        raise ValueError('changed runtime preflight')


def fixture(np):
    A = np.diag([1/5, 2/5, 3/5, 4/5])
    B = np.array([[1, 0, 1/4], [0, 1, -1/4], [1/2, 1/4, 1], [1/4, -1/2, 1/2]])
    C = np.array([[1, 1/5, 0, 1/4], [0, 1, 3/10, -1/4], [1/4, 0, 1, 1/2]])
    D = np.diag([1/10, 1/5, 3/10])
    indices = np.arange(1, 64)
    U = np.zeros((129, 3, 6), dtype=np.complex128)
    G = np.empty((63, 3, 3), dtype=np.complex128)
    Y = np.zeros_like(U)
    for index, k in enumerate(indices):
        for m in range(2):
            for c in range(3):
                for r in range(3):
                    U[k, c, 3*m+r] = ((m+1) * np.exp(2j*np.pi*k*(c+1)*(m+1)/101)
                                      * np.exp(-2j*np.pi*c*r/3))
        G[index] = C @ np.linalg.solve(np.exp(2j*np.pi*k/256)*np.eye(4)-A, B) + D
        Y[k] = G[index] @ U[k]
    u = np.repeat(np.fft.irfft(U, n=256, axis=0)[..., None], 2, axis=3)
    y = np.repeat(np.fft.irfft(Y, n=256, axis=0)[..., None], 2, axis=3)
    return u, y, indices, G, (A, B, C, D)


def fields(model, np):
    return {**{key: np.array(getattr(model, key), dtype=np.float64, copy=True)
               for key in ('A', 'B_u', 'C_y', 'D_yu')},
            **{key: np.array(getattr(model.norm, key), dtype=np.float64, copy=True)
               for key in ('u_mean', 'u_std', 'y_mean', 'y_std')},
            'ts': np.asarray(model.ts, dtype=np.float64)}


def transfer(model, indices, np):
    values = fields(model, np)
    A, B, C, D = (values[key] for key in ('A', 'B_u', 'C_y', 'D_yu'))
    return np.stack([C @ np.linalg.solve(np.exp(2j*np.pi*k/256)*np.eye(4)-A, B)+D
                     for k in indices])


def own_loop(values, physical_u, x0, np):
    A, B, C, D = (values[key] for key in ('A', 'B_u', 'C_y', 'D_yu'))
    normalized_u = ((physical_u-values['u_mean'][None, :, None])
                    / values['u_std'][None, :, None])
    state = x0.copy()
    ys, xs = [], []
    for u in normalized_u:
        xs.append(state.copy())
        ys.append(C @ state+D @ u)
        state = A @ state+B @ u
    raw_y = (np.stack(ys)*values['y_std'][None, :, None]
             + values['y_mean'][None, :, None])
    return raw_y, np.stack(xs), state


def qualify(output, details):
    import equinox as eqx
    import freq_statespace as fss
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optimistix as optx

    from openjev_fsm_author import linear_context

    jax.config.update('jax_enable_x64', True)
    assert jax.config.jax_enable_x64
    assert all(device.platform == 'cpu' for device in jax.devices())
    assert jnp.ones(1, dtype=jnp.float64).dtype == np.float64
    direct = json.loads(importlib.metadata.distribution('freq-statespace').read_text('direct_url.json'))
    assert direct['vcs_info']['commit_id'] == COMMIT
    preflight = json.loads(PREFLIGHT.read_text())
    assert preflight['status'] == 'PASS'
    for relative, expected in preflight['installed_source_matches'].items():
        actual = pin(Path(fss.__file__).parent / relative)
        assert {k: actual[k] for k in ('sha256', 'bytes')} == expected
    details['runtime'] = {'python': platform.python_version(), 'platform': platform.platform(),
                          'jax': jax.__version__, 'optimistix': optx.__version__,
                          'devices': [str(device) for device in jax.devices()],
                          'x64': bool(jax.config.jax_enable_x64), 'source_commit': COMMIT}

    def guard(event, args):
        if event == 'socket.connect':
            raise RuntimeError('network forbidden during fabricated qualification')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.suffix.lower() in ('.npz', '.npy', '.mat', '.zip') and not path.is_relative_to(output):
                raise RuntimeError('external numerical/archive file forbidden')

    sys.addaudithook(guard)
    try:
        guard('open', (str(ROOT / 'fabricated-forbidden-probe.npz'),))
    except RuntimeError:
        details['external_array_guard'] = 'PASS'
    else:
        raise AssertionError('array guard ineffective')

    with jax.default_device(jax.devices('cpu')[0]):
        u, y, indices, G, generative = fixture(np)
        before = (u.copy(), y.copy())
        u.flags.writeable = y.flags.writeable = False
        np.savez(output / 'fabricated-data.npz', u=u, y=y, f_idx=indices, G=G,
                 **dict(zip(('A', 'B', 'C', 'D'), generative, strict=True)))
        data = fss.create_data_object(u, y, indices, 256.0)
        assert data.time.u.shape == data.time.y.shape == (256, 3, 6)
        assert data.freq.U.shape == data.freq.Y.shape == (129, 3, 6)
        assert data.freq.G_bla.G.shape == (63, 3, 3)
        assert np.asarray(data.time.u).dtype == np.float64
        assert np.asarray(data.freq.U).dtype == np.complex128
        for name, raw in (('u', u), ('y', y)):
            for statistic in ('mean', 'std'):
                np.testing.assert_array_equal(getattr(data.norm, f'{name}_{statistic}'),
                                              getattr(raw, statistic)(axis=(0, 2, 3)))
        normalized_G = G*data.norm.u_std[None, None, :]/data.norm.y_std[None, :, None]
        np.testing.assert_allclose(data.freq.G_bla.G, normalized_G, rtol=1e-10, atol=1e-10)
        for actual, expected in zip((u, y), before, strict=True):
            np.testing.assert_array_equal(actual, expected)
        try:
            fss.create_data_object(u[..., 0], y, indices, 256.0)
        except ValueError:
            details['shape_guard'] = 'PASS'
        else:
            raise AssertionError('author accepted malformed input axes')
        details['data_contract'] = {'shape': list(u.shape), 'f_idx_count': len(indices),
                                    'max_bla_error': float(np.max(abs(data.freq.G_bla.G-normalized_G)))}
        print('Fabricated data and normalized BLA contract PASS', flush=True)

        initial = fss.lin.subspace_id(data, nx=4, nq=5, freq_weighting=False,
                                     input_output_mode=False, logging_enabled=False)
        initial_fields = fields(initial, np)
        fss.save_model(initial, output / 'initial.zip')
        np.savez(output / 'initial.npz', **initial_fields)
        np.testing.assert_allclose(transfer(initial, indices, np), normalized_G, rtol=1e-6, atol=1e-6)
        perturbed = eqx.tree_at(lambda model: model.D_yu, initial, initial.D_yu+.02*jnp.eye(3))
        fss.save_model(perturbed, output / 'perturbed.zip')
        initial_loss = float(np.mean(abs(transfer(perturbed, indices, np)-normalized_G)**2))
        assert initial_loss > 0
        details['subspace'] = {'max_fr_error': float(np.max(abs(transfer(initial, indices, np)-normalized_G))),
                               'perturbed_frequency_mse': initial_loss}
        print('Order4 subspace transfer-function qualification PASS; starting bounded BFGS', flush=True)
        model, solve = fss.lin.optimize(
            perturbed, data, solver=optx.BFGS(rtol=1e-3, atol=1e-5),
            freq_weighting=False, input_output_mode=False, max_iter=25, print_every=1,
            return_solve_details=True, device='cpu')
        final_fields = fields(model, np)
        # Materialization above synchronizes the result before recording outer completion.
        np.savez(output / 'final.npz', **final_fields)
        np.savez(output / 'solver-trace.npz', loss_history=np.asarray(solve.loss_history),
                 iter_times=np.asarray(solve.iter_times), iter_count=np.asarray(solve.iter_count),
                 author_stop_flag=np.asarray(solve.converged), wall_time=np.asarray(solve.wall_time))
        fss.save_model(model, output / 'final.zip')
        for value in final_fields.values():
            assert np.isfinite(value).all()
        assert 1 <= solve.iter_count <= 25
        assert solve.loss_history.shape == solve.iter_times.shape == (solve.iter_count,)
        assert np.isfinite(solve.loss_history).all() and np.isfinite(solve.iter_times).all()
        final_loss = float(np.mean(abs(transfer(model, indices, np)-normalized_G)**2))
        details['refinement'] = {
            'iterations': int(solve.iter_count), 'max_iter': 25,
            'author_stop_flag': bool(solve.converged),
            'stop_flag_scope': 'BFGS Cauchy small-change flag; discarded result code not recoverable',
            'initial_frequency_mse': initial_loss, 'final_frequency_mse': final_loss,
            'loss_history': np.asarray(solve.loss_history).tolist(),
            'iter_times': np.asarray(solve.iter_times).tolist(), 'author_wall_seconds': float(solve.wall_time),
            'optimizer_success_certified': False,
        }
        assert final_loss <= initial_loss
        for key, expected in initial_fields.items():
            np.testing.assert_array_equal(fields(initial, np)[key], expected)
        print('Bounded refinement finite/nonworsening qualification PASS', flush=True)

        checks = []
        for batch in (1, 3):
            t = np.arange(228, dtype=np.float64)[:, None, None]
            c = np.arange(3, dtype=np.float64)[None, :, None]
            b = np.arange(batch, dtype=np.float64)[None, None, :]
            request = .3*np.sin((t+1)*(c+1)/17+b/3)+.2*np.cos((t+3)/29+c/5+b)
            request = request*final_fields['u_std'][None, :, None]+final_fields['u_mean'][None, :, None]
            x0 = (np.arange(4*batch, dtype=np.float64).reshape(4, batch)+1)/10
            request.flags.writeable = x0.flags.writeable = False
            author_y, author_t, author_x = model.simulate(request, x0=x0, offset=None)
            expected_y, expected_x, expected_final = own_loop(final_fields, request, x0, np)
            np.testing.assert_allclose(author_y, expected_y, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(author_x, expected_x, rtol=1e-10, atol=1e-10)
            np.testing.assert_array_equal(author_t, np.arange(228)*float(model.ts))
            matrices = tuple(final_fields[key] for key in ('A', 'B_u', 'C_y', 'D_yu'))
            normalized_u = ((request-final_fields['u_mean'][None, :, None])
                            / final_fields['u_std'][None, :, None]).transpose(2, 0, 1).copy()
            normalized_y = ((author_y-final_fields['y_mean'][None, :, None])
                            / final_fields['y_std'][None, :, None]).transpose(2, 0, 1).copy()
            context_y, context_u = normalized_y[:, :100].copy(), normalized_u[:, 1:100].copy()
            future = normalized_u[:, 100:].copy()
            for array in (context_y, context_u, future):
                array.flags.writeable = False
            forecast, final, diagnostic = linear_context.predict(*matrices, context_y, context_u, future)
            assert diagnostic['rank'] == 4 and diagnostic['context_pairs'] == 99
            np.testing.assert_allclose(forecast, normalized_y[:, 100:], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(final, expected_final.T, rtol=1e-10, atol=1e-10)
            state, _ = linear_context.condition(*matrices, context_y, context_u)
            np.testing.assert_allclose(state, author_x[100].T, rtol=1e-10, atol=1e-10)
            public_future, _, public_states = model.simulate(request[100:], x0=state.T, offset=None)
            np.testing.assert_allclose(public_future, author_y[100:], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(public_states, author_x[100:], rtol=1e-10, atol=1e-10)
            assert not np.shares_memory(forecast, future) and not np.shares_memory(final, context_y)
            bad = context_y.copy()
            bad[0, 1, 0] = np.nan
            try:
                linear_context.predict(*matrices, bad, context_u, future)
            except ValueError:
                pass
            else:
                raise AssertionError('causal adapter accepted nonfinite context')
            np.savez(output / f'parity-batch{batch}.npz', physical_u=request, x0=x0,
                     author_y=author_y, author_x=author_x, own_y=expected_y, own_x=expected_x,
                     adapter_forecast=forecast, adapter_final=final, expected_final=expected_final)
            checks.append({'batch': batch, 'context': 100, 'horizon': 128, 'rank': diagnostic['rank'],
                           'public_y_max_error': float(np.max(abs(author_y-expected_y))),
                           'public_x_max_error': float(np.max(abs(author_x-expected_x))),
                           'adapter_max_error': float(np.max(abs(forecast-normalized_y[:, 100:])))})
        details['simulation'] = checks

        # A separate normalization witness avoids relying on nearly zero fixture means.
        shifted = fss.ModelBLA(
            A=model.A, B_u=model.B_u, C_y=model.C_y, D_yu=model.D_yu, ts=model.ts,
            norm=fss.Normalizer(u_mean=np.array([2., -3., .5]), u_std=np.array([.5, 2., 3.]),
                                y_mean=np.array([-4., 1., 7.]), y_std=np.array([3., .25, 2.])))
        shifted_fields = fields(shifted, np)
        shifted_checks = []
        for batch in (1, 3):
            shifted_input = np.arange(37*3*batch, dtype=np.float64).reshape(37, 3, batch)/101
            shifted_x0 = (np.arange(4*batch, dtype=np.float64).reshape(4, batch)+2)/7
            shifted_input.flags.writeable = shifted_x0.flags.writeable = False
            shifted_y, _, shifted_x = shifted.simulate(shifted_input, x0=shifted_x0, offset=None)
            expected_y, expected_x, _ = own_loop(shifted_fields, shifted_input, shifted_x0, np)
            np.testing.assert_allclose(shifted_y, expected_y, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(shifted_x, expected_x, rtol=1e-10, atol=1e-10)
            np.savez(output / f'nontrivial-normalization-batch{batch}.npz',
                     physical_u=shifted_input, x0=shifted_x0, author_y=shifted_y,
                     author_x=shifted_x, own_y=expected_y, own_x=expected_x, **shifted_fields)
            shifted_checks.append({'batch': batch, 'output_max_error': float(np.max(abs(shifted_y-expected_y))),
                                   'state_max_error': float(np.max(abs(shifted_x-expected_x)))})
        details['nontrivial_normalization'] = shifted_checks

        reloaded = fss.load_model(output / 'final.zip')
        for key, expected in final_fields.items():
            np.testing.assert_array_equal(fields(reloaded, np)[key], expected)
        roundtrip_y, roundtrip_t, roundtrip_x = reloaded.simulate(request, x0=x0, offset=None)
        for actual, expected in ((roundtrip_y, author_y), (roundtrip_t, author_t), (roundtrip_x, author_x)):
            np.testing.assert_array_equal(actual, expected)
        details['serialization'] = {'fields_bitwise_equal': True, 'forecasts_bitwise_equal': True}
        print('Public simulation, causal adapter and checkpoint roundtrip PASS', flush=True)


def execute(output):
    definition = json.loads((output / 'definition.json').read_text())
    if (output / 'receipt.json').exists() or (output / 'execution-start.json').exists():
        raise ValueError('qualification execution already attempted')
    check_pins(definition, output)
    for key, value in definition['environment'].items():
        if os.environ.get(key) != value:
            raise ValueError(f'wrong fixed environment: {key}')
    write_json(output / 'execution-start.json', {'utc': datetime.now(timezone.utc).isoformat(),
                                               'pid': os.getpid(), 'definition': pin(output / 'definition.json')})
    start = time.monotonic()
    details, error = {}, None
    try:
        qualify(output, details)
        check_pins(definition, output)
    except Exception as exc:  # preserve the original failed scientific-free qualification
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        traceback.print_exc()
    receipt = {'status': 'PASS' if error is None else 'FAIL', 'error': error,
               'elapsed_seconds': time.monotonic()-start, 'definition': pin(output / 'definition.json'),
               'details': details, 'measurement_access': False, 'pretrained_weight_access': False,
               'empirical_fit': False,
               'files': {str(p.relative_to(output)): pin(p) for p in sorted(output.rglob('*'))
                         if p.is_file() and p.name not in ('process.log', 'process.json')}}
    write_json(output / 'receipt.json', receipt)
    print(json.dumps({'status': receipt['status'], 'error': error, 'elapsed_seconds': receipt['elapsed_seconds']}), flush=True)
    return 0 if error is None else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    folder = args.output.resolve()
    if args.prepare:
        prepare(folder)
    else:
        raise SystemExit(execute(folder))
