# SPDX-License-Identifier: GPL-3.0-or-later
"""One full-size, synthetic-only qualification of pinned author NL-LFR BFGS."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[1]
ROOT = COMPONENT.parents[1]
COMMIT = 'a79e8c567b018a6c9462528fc1e10b77fd19b3e2'
PREFLIGHT = ROOT / 'output/fsm-author-engineering-v1/runtime-preflight-01/receipt.json'
SOURCES = (Path(__file__).resolve(), COMPONENT/'pyproject.toml', COMPONENT/'uv.lock',
           COMPONENT/'src/openjev_fsm_author/nllfr.py',
           COMPONENT/'src/openjev_fsm_author/benchmark.py',
           COMPONENT/'src/openjev_fsm_author/linear_context.py',
           ROOT/'research/fsm-nllfr-runtime-contract.md')
ENV = {**{k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                          'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')},
       'JAX_PLATFORMS': 'cpu', 'JAX_ENABLE_X64': 'True', 'PYTHONHASHSEED': '0',
       'PYTHONDONTWRITEBYTECODE': '1'}
N, OFFSET, MAX_ITER, TIME_CAP, RSS_CAP = 8192, 820, 2, 600, 32*1024**3


def pin(path):
    path = Path(path)
    blob = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for path in SOURCES:
        relative = path.relative_to(ROOT)
        destination = output/'source'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        sources[str(relative)] = pin(path)
    command = [str(COMPONENT/'.venv/bin/python'), str(Path(__file__).resolve())]
    definition = {'created_utc': now(), 'sources': sources, 'runtime_preflight': pin(PREFLIGHT),
                  'environment': ENV, 'commands': {
                      mode: command+[f'--{mode}', '--output', str(output)]
                      for mode in ('init-only', 'execute')},
                  'fixture': {'N': N, 'R': 6, 'P': 2, 'nu': 3, 'ny': 3, 'nx': 28,
                              'nz': 16, 'nw': 8, 'width': 64, 'hidden_layers': 2,
                              'activation': 'relu', 'seed': 42, 'sigma': 1e-4,
                              'frequency_indices': [1, 3839], 'fs': 6400.,
                              'offset': OFFSET, 'rollout_length': N+OFFSET,
                              'BFGS_max_iter': MAX_ITER, 'BFGS_rtol': 1e-3, 'BFGS_atol': 1e-5,
                              'frequency_weighting': False, 'parity_atol_rtol': 1e-10,
                              'trainable_scalars': 7473},
                  'outer_timeout_seconds': TIME_CAP, 'child_rss_cap_bytes': RSS_CAP,
                  'scope': 'Synthetic full-size runtime only; no measurements/pretrained weights; '
                           'one discarded vendor warmup plus at most two recorded BFGS steps; '
                           'one separate initialization-only process; no empirical fit'}
    write(output/'definition.json', definition)
    print(json.dumps({'status': 'PREPARED', 'definition': pin(output/'definition.json')}), flush=True)


def check(output):
    definition = json.loads((output/'definition.json').read_text())
    for relative, expected in definition['sources'].items():
        if pin(ROOT/relative) != expected:
            raise ValueError('changed source: '+relative)
        snapshot = pin(output/'source'/relative)
        if any(snapshot[k] != expected[k] for k in ('bytes', 'sha256')):
            raise ValueError('changed snapshot: '+relative)
    if pin(PREFLIGHT) != definition['runtime_preflight']:
        raise ValueError('changed runtime preflight')
    return definition


def fixture(np):
    poles = .15+.7*np.arange(28)/27
    A = np.diag(poles)
    B = np.sin((np.arange(28)[:, None]+1)*(np.arange(3)[None]+1))/np.sqrt(28)
    C = np.cos((np.arange(3)[:, None]+1)*(np.arange(28)[None]+1))/np.sqrt(28)
    D = np.diag([.1, .2, .3])
    indices = np.arange(1, 3840)
    U = np.zeros((N//2+1, 3, 6), dtype=np.complex128)
    for m in range(2):
        for c in range(3):
            for r in range(3):
                U[indices, c, 3*m+r] = ((m+1)*np.exp(2j*np.pi*indices*(c+1)*(m+1)/101)
                                        * np.exp(-2j*np.pi*c*r/3))
    G = np.einsum('ci,fi,id->fcd', C, 1/(np.exp(2j*np.pi*indices/N)[:, None]-poles), B)
    G += D+.02*np.eye(3)
    Y = np.zeros_like(U)
    Y[indices] = G @ U[indices]
    u = np.repeat(np.fft.irfft(U, n=N, axis=0)[..., None], 2, axis=3)
    y = np.repeat(np.fft.irfft(Y, n=N, axis=0)[..., None], 2, axis=3)
    return u, y, indices, (A, B, C, D)


def leaf_hashes(arrays):
    return {k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                'sha256': hashlib.sha256(v.tobytes(order='C')).hexdigest()}
            for k, v in sorted(arrays.items())}


def runtime_identity(fss):
    preflight = json.loads(PREFLIGHT.read_text())
    direct = json.loads(importlib.metadata.distribution('freq-statespace').read_text('direct_url.json'))
    versions = {name: importlib.metadata.version(name) for name in preflight['versions']}
    assert preflight['status'] == 'PASS'
    assert platform.python_version() == preflight['python']
    assert sys.executable == preflight['executable']
    assert versions == preflight['versions'] and direct == preflight['upstream_direct_url']
    assert direct['vcs_info']['commit_id'] == COMMIT
    installed = {}
    for relative, expected in preflight['installed_source_matches'].items():
        actual = pin(Path(fss.__file__).parent/relative)
        installed[relative] = {k: actual[k] for k in ('sha256', 'bytes')}
        assert installed[relative] == expected
    return {'python': platform.python_version(), 'executable': sys.executable,
            'versions': versions, 'direct_url': direct, 'installed_source_matches': installed}


def load_runtime(output):
    import equinox as eqx
    import freq_statespace as fss
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optimistix as optx

    from openjev_fsm_author import nllfr
    from openjev_fsm_author.benchmark import export_model

    if any(os.environ.get(k) != v for k, v in ENV.items()):
        raise ValueError('fixed process environment mismatch')
    assert jax.config.jax_enable_x64 and all(d.platform == 'cpu' for d in jax.devices())
    runtime_identity(fss)

    def guard(event, args):
        if event == 'socket.connect':
            raise RuntimeError('network forbidden in synthetic qualification')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.suffix.lower() in ('.npz', '.npy', '.mat', '.zip') and not path.is_relative_to(output):
                raise RuntimeError('external numerical/archive file forbidden')
    sys.addaudithook(guard)
    return eqx, fss, jax, jnp, np, optx, nllfr, export_model


def initialize(runtime):
    _, fss, jax, jnp, np, _, nllfr, export_model = runtime
    u, y, indices, (A, B, C, D) = fixture(np)
    data = fss.create_data_object(u, y, indices, 6400.)
    bla = fss.ModelBLA(A=A, B_u=B*np.asarray(data.norm.u_std)[None],
                       C_y=C/np.asarray(data.norm.y_std)[:, None],
                       D_yu=D*np.asarray(data.norm.u_std)[None]/np.asarray(data.norm.y_std)[:, None],
                       ts=1/6400, norm=data.norm)
    network = fss.static.NeuralNetwork(nz=16, nw=8, layers=2, neurons_per_layer=64,
                                       activation=jax.nn.relu, seed=42, bias=True)
    model = fss.nonlin.connect(bla, network, sigma=1e-4)
    arrays = nllfr.export(model)
    assert nllfr.validate(arrays) == (28, 16, 8, 64)
    assert data.time.u.shape == data.time.y.shape == (N, 3, 6)
    assert np.asarray(data.time.u).dtype == np.float64
    # Match the vendor's exact partition, including its explicit exclusions.
    eqx = runtime[0]
    trainable, _ = eqx.partition(model, eqx.is_inexact_array)
    trainable = eqx.tree_at(lambda t: (t.norm, t._bla), trainable, replace=(None, None))
    leaves = jax.tree_util.tree_leaves(trainable)
    assert sum(v.size for v in leaves) == 7473
    assert all(v.dtype == jnp.float64 for v in leaves)
    identity = {'arrays': leaf_hashes(arrays), 'original_bla': leaf_hashes(export_model(model._bla)),
                'pythonhashseed': os.environ['PYTHONHASHSEED'],
                'tag_hashes': {s: hash(s)&0xFFFFFFFF for s in ('neural_network', 'connect')},
                'trainable_scalars': sum(v.size for v in leaves), 'trainable_leaves': len(leaves)}
    return model, data, arrays, identity, (u, y, indices, A, B, C, D)


def own_loop(arrays, physical_u, x0, np):
    U = (physical_u-arrays['u_mean'][None, :, None])/arrays['u_std'][None, :, None]
    state = x0.copy()
    ys, xs, ws, zs = [], [], [], []
    for u in U:
        z = arrays['C_z'] @ state + arrays['D_zu'] @ u
        h0 = np.maximum(arrays['W0'] @ z + arrays['b0'][:, None], 0)
        h1 = np.maximum(arrays['W1'] @ h0 + arrays['b1'][:, None], 0)
        w = arrays['W2'] @ h1 + arrays['b2'][:, None]
        y = arrays['C_y'] @ state + arrays['D_yu'] @ u + arrays['D_yw'] @ w
        ys.append(y)
        xs.append(state.copy())
        zs.append(z)
        ws.append(w)
        state = arrays['A'] @ state + arrays['B_u'] @ u + arrays['B_w'] @ w
    physical = np.stack(ys)*arrays['y_std'][None, :, None]+arrays['y_mean'][None, :, None]
    return physical, np.stack(xs), np.stack(ws), np.stack(zs), state


def parity(model, output, label, runtime):
    _, _, _, _, np, _, nllfr, _ = runtime
    arrays = nllfr.export(model)
    checks = []
    for batch in (1, 3):
        t = np.arange(128)[:, None, None]
        c = np.arange(3)[None, :, None]
        b = np.arange(batch)[None, None, :]
        inputs = .3*np.sin((t+1)*(c+1)/17+b/3)+.2*np.cos((t+3)/29+c/5+b)
        inputs = inputs*arrays['u_std'][None, :, None]+arrays['u_mean'][None, :, None]
        x0 = (np.arange(28*batch).reshape(28, batch)+1)/100
        y, times, x, w, z = map(np.asarray, model.simulate(inputs, x0=x0, offset=None))
        ey, ex, ew, ez, final = own_loop(arrays, inputs, x0, np)
        exported_y, exported_final = nllfr.physical_rollout(arrays, inputs.transpose(2, 0, 1).copy(), x0.T.copy())
        np.savez(output/f'{label}-parity-B{batch}.npz', inputs=inputs, x0=x0, author_y=y,
                 author_x=x, author_w=w, author_z=z, own_y=ey, own_x=ex, own_w=ew,
                 own_z=ez, own_final=final, exported_y=exported_y, exported_final=exported_final)
        errors = {}
        for key, actual, expected in (('y', y, ey), ('x', x, ex), ('w', w, ew), ('z', z, ez),
                                      ('export_y', exported_y, y.transpose(2, 0, 1)),
                                      ('export_final', exported_final, final.T)):
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
            errors[key] = float(np.max(abs(actual-expected)))
        np.testing.assert_array_equal(times, np.arange(128)*float(model.ts))
        checks.append({'batch': batch, 'max_errors': errors})
    return checks


def spectral_loss(model, original, data, np):
    from openjev_fsm_author import nllfr

    arrays = nllfr.export(model)
    U = np.asarray(data.freq.U)
    poles = np.diag(original['A'])
    resolvent = 1/(np.exp(2j*np.pi*np.arange(N//2+1)/N)[:, None]-poles)
    X = np.einsum('fi,ic,fcr->fir', resolvent, original['B_u'], U)
    x0 = np.fft.irfft(X, n=N, axis=0)[-OFFSET]
    normalized = np.asarray(data.time.u)
    extended = np.concatenate((normalized[-OFFSET:], normalized), axis=0)
    values, _, _ = nllfr.rollout(arrays, extended.transpose(2, 0, 1).copy(), x0.T.copy())
    Y_hat = np.fft.rfft(values[:, OFFSET:].transpose(1, 2, 0), axis=0)
    return float(np.sum(abs(np.asarray(data.freq.Y)-Y_hat)**2)/((N//2+1)*6))


def qualify(output, detail):
    runtime = load_runtime(output)
    eqx, fss, jax, jnp, np, optx, nllfr, export_model = runtime
    detail['runtime_identity_before'] = runtime_identity(fss)
    detail['runtime'] = {'python': platform.python_version(), 'jax': jax.__version__,
                         'optimistix': optx.__version__, 'cpu': [str(d) for d in jax.devices()],
                         'x64': bool(jax.config.jax_enable_x64), 'commit': COMMIT}
    detail['baseline_ru_maxrss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    model, data, arrays, identity, raw = initialize(runtime)
    write(output/'main-initialization.json', identity)
    assert identity == json.loads((output/'fresh-initialization.json').read_text())
    detail['fresh_process_initialization_equal'] = True
    np.savez(output/'synthetic-data.npz', **dict(zip(('u', 'y', 'indices', 'A', 'B', 'C', 'D'), raw, strict=True)))
    np.savez(output/'initial.npz', **arrays)
    fss.save_model(model, output/'initial.zip')
    original = export_model(model._bla)
    np.savez(output/'original-bla.npz', **original)
    detail['initial_parity'] = parity(model, output, 'initial', runtime)
    shifted_norm = fss.Normalizer(u_mean=np.array([2., -3., .5]), u_std=np.array([.5, 2., 3.]),
                                  y_mean=np.array([-4., 1., 7.]), y_std=np.array([3., .25, 2.]))
    shifted = eqx.tree_at(lambda m: m.norm, model, shifted_norm)
    detail['shifted_normalizer_parity'] = parity(shifted, output, 'shifted', runtime)
    initial_loss = spectral_loss(model, original, data, np)
    assert np.isfinite(initial_loss) and initial_loss > 0
    detail['initial_loss'] = initial_loss
    write(output/'before-optimization.json', detail)
    print('Full-size synthetic 7473-parameter admission/parity PASS; starting BFGS cap2', flush=True)
    start = time.perf_counter()
    final, solve = fss.nonlin.optimize(model, data, solver=optx.BFGS(rtol=1e-3, atol=1e-5),
                                      freq_weighting=False, max_iter=MAX_ITER, print_every=1,
                                      return_solve_details=True, offset=None, device='cpu')
    # Preserve returned evidence before export performs finite/schema validation.
    fss.save_model(final, output/'final.zip')
    raw_final = {name: np.array(getattr(final, name), copy=True)
                 for name in ('A', 'B_u', 'C_y', 'D_yu', 'B_w', 'C_z', 'D_yw', 'D_zu', 'ts')}
    raw_final.update({name: np.array(getattr(final.norm, name), copy=True)
                      for name in ('u_mean', 'u_std', 'y_mean', 'y_std')})
    for i, layer in enumerate(final.func_static.model.layers):
        raw_final[f'W{i}'] = np.array(layer.weight, copy=True)
        raw_final[f'b{i}'] = np.array(layer.bias, copy=True)
    np.savez(output/'final.npz', **raw_final)
    np.savez(output/'solver-trace.npz', loss_history=np.asarray(solve.loss_history),
             iter_times=np.asarray(solve.iter_times), iter_count=np.asarray(solve.iter_count),
             author_stop_flag=np.asarray(solve.converged), wall_time=np.asarray(solve.wall_time))
    detail['optimization_and_preservation_compile_inclusive_seconds'] = time.perf_counter()-start
    final_arrays = nllfr.export(final)
    assert 1 <= solve.iter_count <= MAX_ITER
    assert solve.loss_history.shape == solve.iter_times.shape == (solve.iter_count,)
    assert np.isfinite(solve.loss_history).all() and np.isfinite(solve.iter_times).all()
    for key, value in original.items():
        np.testing.assert_array_equal(export_model(final._bla)[key], value)
    for key in ('u_mean', 'u_std', 'y_mean', 'y_std', 'ts'):
        np.testing.assert_array_equal(final_arrays[key], arrays[key])
    final_loss = spectral_loss(final, original, data, np)
    detail['optimization'] = {'iterations': int(solve.iter_count), 'max_iter': MAX_ITER,
                              'author_stop_flag': bool(solve.converged), 'initial_loss': initial_loss,
                              'final_loss': final_loss, 'loss_history': solve.loss_history.tolist(),
                              'iter_times': solve.iter_times.tolist(), 'author_wall_seconds': float(solve.wall_time),
                              'optimizer_success_certified': False,
                              'scope': 'Two recorded updates at most plus one discarded vendor warmup'}
    write(output/'after-optimization.json', detail)
    assert np.isfinite(final_loss) and final_loss <= initial_loss
    detail['final_parity'] = parity(final, output, 'final', runtime)
    reloaded = fss.load_model(output/'final.zip')
    for actual, expected in ((nllfr.export(reloaded), final_arrays),
                             (export_model(reloaded._bla), original)):
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])
    # Compare all five public results bitwise after serialization.
    request = np.arange(128*3, dtype=np.float64).reshape(128, 3, 1)/1000
    x0 = np.arange(28, dtype=np.float64)[:, None]/100
    for actual, expected in zip(reloaded.simulate(request, x0=x0), final.simulate(request, x0=x0), strict=True):
        np.testing.assert_array_equal(actual, expected)
    detail['serialization'] = {'all_numeric_arrays_bitwise_equal': True, 'public_results_bitwise_equal': True}
    detail['runtime_identity_after'] = runtime_identity(fss)
    assert detail['runtime_identity_before'] == detail['runtime_identity_after']
    print('Full-size synthetic refinement, public/export and ZIP roundtrip PASS', flush=True)


def execute(output, initialization_only):
    check(output)
    marker = 'fresh-init' if initialization_only else 'execution'
    write(output/f'{marker}-start.json', {'utc': now(), 'pid': os.getpid()})
    start, detail, error = time.perf_counter(), {}, None
    try:
        if initialization_only:
            runtime = load_runtime(output)
            _, _, _, identity, _ = initialize(runtime)
            write(output/'fresh-initialization.json', identity)
            detail['initialization_only'] = True
            detail['runtime_identity_after'] = runtime_identity(runtime[1])
        else:
            qualify(output, detail)
        check(output)
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        traceback.print_exc()
    detail['peak_ru_maxrss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt = {'status': 'PASS' if error is None else 'FAIL', 'error': error, 'details': detail,
               'seconds': time.perf_counter()-start, 'definition': pin(output/'definition.json'),
               'measurement_access': False, 'pretrained_weight_access': False, 'empirical_fit': False}
    write(output/f'{marker}-receipt.json', receipt)
    print(json.dumps({'status': receipt['status'], 'seconds': receipt['seconds'], 'error': error}), flush=True)
    return 0 if error is None else 1


def supervise(output):
    definition = check(output)
    write(output/'supervisor-start.json', {'utc': now(), 'pid': os.getpid(), 'definition': pin(output/'definition.json')})
    start, stages, peak = time.monotonic(), [], 0
    status = 'PASS'
    for mode, command in definition['commands'].items():
        with (output/f'{mode}.log').open('xb') as log:
            child = subprocess.Popen(command, env={**os.environ, **ENV}, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            stage = {'mode': mode, 'command': command, 'pid': child.pid, 'started_utc': now()}
            write(output/f'{mode}-launch.json', stage)
            stage_start = time.monotonic()
            while child.poll() is None:
                if time.monotonic()-start >= TIME_CAP:
                    status = 'TIMEOUT'
                rss_text = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(child.pid)],
                                          capture_output=True, text=True, check=False).stdout.strip()
                rss = int(rss_text)*1024 if rss_text else 0
                peak = max(peak, rss)
                if rss > RSS_CAP:
                    status = 'MEMORY_LIMIT'
                if status != 'PASS':
                    os.killpg(child.pid, signal.SIGKILL)
                    break
                time.sleep(.5)
            code = child.wait()
            stage.update(returncode=code, seconds=time.monotonic()-stage_start,
                         log=pin(output/f'{mode}.log'))
            stages.append(stage)
            write(output/f'{mode}-process.json', stage)
            if code != 0:
                status = status if status != 'PASS' else 'FAIL'
                break
    unchanged = True
    try:
        check(output)
    except Exception:
        unchanged = False
        status = 'IDENTITY_FAILED'
    receipt = {'status': status, 'stages': stages, 'elapsed_seconds': time.monotonic()-start,
               'peak_polled_child_rss_bytes': peak, 'timeout_seconds': TIME_CAP,
               'rss_cap_bytes': RSS_CAP, 'sources_unchanged': unchanged,
               'definition': pin(output/'definition.json'),
               'files': {str(p.relative_to(output)): pin(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    write(output/'process.json', receipt)
    print(json.dumps({'status': status, 'elapsed_seconds': receipt['elapsed_seconds'],
                      'peak_polled_child_rss_bytes': peak}), flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    for value in ('prepare', 'init-only', 'execute', 'supervise'):
        modes.add_argument('--'+value, action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare:
        prepare(output)
    elif args.supervise:
        raise SystemExit(supervise(output))
    else:
        raise SystemExit(execute(output, args.init_only))
