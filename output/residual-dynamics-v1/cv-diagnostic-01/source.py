"""Post hoc CV16 reference forecasts. No neural/native calls or parameter fit.

The fixed formula uses context indices16 and31 and extrapolates all nine
normalized coordinates in float64, without projecting orientation coordinates.
This executes a NEW deterministic reference, not merely a saved-output audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANELS = ('test_sin', 'test_zigzag')
FAMILIES = ('none', 'bias', 'public', 'latent', 'history16')
SEEDS = (411, 512, 613)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def cv16(context):
    require(context.ndim == 3 and context.shape[1:] == (32, 9)
            and context.dtype in (np.float32, np.float64) and np.isfinite(context).all(), 'Invalid context')
    values = context.astype(np.float64)
    velocity = (values[:, 31] - values[:, 16]) / 15.0
    return values[:, 31:32] + np.arange(1, 26, dtype=np.float64)[None, :, None] * velocity[:, None]


def synthetic_causality_test():
    values = np.arange(2 * 57 * 9, dtype=np.float64).reshape(2, 57, 9) / 10
    original = cv16(values[:, :32])
    changed = values.copy()
    changed[:, 32:] = -1e9
    require(np.array_equal(original, cv16(changed[:, :32])), 'Future targets influenced reference')
    expected = values[:, 31:32] + np.arange(1, 26)[None, :, None] * .9
    require(np.allclose(original, expected, rtol=0, atol=1e-12), 'Hand linear extrapolation mismatch')
    return {'status': 'passed', 'tests': 1, 'scope': 'synthetic future-target perturbation plus analytic linear fixture'}


def metric(squared, ids, scale):
    require(squared.shape == (160, 25, 9) and np.isfinite(squared).all(), 'Invalid squared errors')
    position = squared[:, :, :3] * scale[None, None, :3] ** 2
    return {'mse': float(squared.mean()), 'horizon_mse': squared.mean(axis=(0, 2)).tolist(),
            'position_rmse_m': float(np.sqrt(position.sum(-1).mean())),
            'position_horizon_rmse_m': np.sqrt(position.sum(-1).mean(0)).tolist(),
            'per_parent': [{'source_id': int(i), 'windows': int((ids == i).sum()),
                            'mse': float(squared[ids == i].mean()),
                            'horizon_mse': squared[ids == i].mean(axis=(0, 2)).tolist(),
                            'position_rmse_m': float(np.sqrt(position[ids == i].sum(-1).mean()))}
                           for i in range(10)]}


def run(experiment, report, out, protocol_sha256, completed_sha256, report_sha256):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    inputs, new_forecasts = {}, 0
    try:
        write(out / 'started.json', {'scope': 'post hoc new analytic reference; original 49-check gate unchanged',
                                    'source_sha256': sha(__file__), 'protocol_sha256': protocol_sha256,
                                    'completed_sha256': completed_sha256, 'report_sha256': report_sha256})
        (out / 'source.py').write_bytes(Path(__file__).read_bytes())
        write(out / 'synthetic-test.json', synthetic_causality_test())
        for path, digest in ((experiment / 'protocol.json', protocol_sha256),
                             (experiment / 'run-01/completed.json', completed_sha256),
                             (report / 'receipt.json', report_sha256)):
            require(sha(path) == digest, 'External artifact hash mismatch: ' + str(path))
            inputs[str(path)] = digest
        protocol, done, receipt = read(experiment / 'protocol.json'), read(experiment / 'run-01/completed.json'), read(report / 'receipt.json')
        require(done['status'] == receipt['status'] == 'completed', 'Incomplete study')
        require(done['protocol_sha256'] == receipt['protocol_sha256'] == protocol_sha256
                and receipt['execution_completed_sha256'] == completed_sha256, 'Study lineage mismatch')
        require(protocol['context'] == 32 and protocol['horizon'] == 25, 'Unsupported protocol')
        for name, digest in protocol['sources'].items():
            require(sha(ROOT / name) == digest, 'Frozen source mismatch: ' + name)
        require(sha(report / 'summary.json') == receipt['files']['summary.json']['sha256'], 'Report summary hash')
        inputs[str(report / 'summary.json')] = sha(report / 'summary.json')
        old_gate = read(report / 'summary.json')['continuation_gate']
        require(old_gate['total_checks'] == 49, 'Original gate schema')
        data = Path(protocol['data_path'])
        for name in ('completed.json', 'manifest.json', 'normalization.npz', *[p + '.npz' for p in PANELS]):
            require(sha(data / name) == protocol['data_hashes'][name], 'Data hash mismatch: ' + name)
            inputs[str(data / name)] = sha(data / name)
        with np.load(data / 'normalization.npz', allow_pickle=False) as arrays:
            scale = arrays['obs_scale'].astype(np.float64)
        require(scale.shape == (9,) and np.isfinite(scale).all() and (scale > 0).all(), 'Observation scales')
        results = {}
        for panel in PANELS:
            with np.load(data / (panel + '.npz'), allow_pickle=False) as arrays:
                obs, ids = arrays['obs'], arrays['source_ids']
            require(obs.shape == (160, 57, 9) and np.isfinite(obs).all()
                    and np.array_equal(ids, np.repeat(np.arange(10), 16)), 'Panel shape/parents')
            target = obs[:, 32:].astype(np.float64)
            t = time.perf_counter()
            predicted = cv16(obs[:, :32])
            forecast_seconds = time.perf_counter() - t
            new_forecasts += len(obs)
            np.save(out / (panel + '-cv16-predictions.npy'), predicted, allow_pickle=False)
            squared = (predicted - target) ** 2
            current = {'cv16': metric(squared, ids, scale), 'forecast_seconds': forecast_seconds,
                       'saved_per_fit': {}, 'saved_families': {}, 'saved_references': {}}
            for family in (*FAMILIES, 'ridge16', 'hold_last'):
                errors = []
                names = [f'{family}-{seed}' for seed in SEEDS] if family in FAMILIES else [family]
                for name in names:
                    path = experiment / 'run-01' / f'{panel}-{name}-predictions.npy'
                    binding = done['files'][path.name]
                    require(path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256'], 'Saved prediction identity')
                    inputs[str(path)] = binding['sha256']
                    saved = np.load(path, allow_pickle=False)
                    require(saved.shape == target.shape and np.isfinite(saved).all(), 'Saved prediction schema')
                    error = (saved.astype(np.float64) - target) ** 2
                    errors.append(error)
                    current['saved_per_fit'][name] = metric(error, ids, scale)
                pooled = metric(np.mean(errors, axis=0), ids, scale)
                dest = 'saved_families' if family in FAMILIES else 'saved_references'
                current[dest][family] = pooled
            results[panel] = current
        write(out / 'summary.json', {'status': 'completed', 'scope': 'posthoc_cv16_diagnostic',
              'formula': 'obs31 + h*(obs31-obs16)/15 for h1..25; all9 coordinates; float64; no orientation projection',
              'panels': results, 'original_gate_unchanged': old_gate,
              'new_deterministic_reference_forecasts': new_forecasts, 'new_neural_calls': 0,
              'new_native_calls': 0, 'new_fits': 0, 'new_random_draws': 0,
              'limits': ['Post hoc reference selected after primary results; not predeclared and not a rescue.',
                         '10 parent trajectories per panel,16 dependent windows each; family errors pool all3 saved fits.',
                         'Position RMSE is sqrt(mean squared Euclidean position error) over every window/horizon.',
                         'Prediction arrays remain local because upstream dataset license is unresolved.']})
        lines = ['# Post hoc CV16 diagnostic', '',
                 '320 new deterministic reference forecasts; zero neural/native calls or fitting. The original49-check gate is unchanged.', '',
                 '| Panel | Predictor | Normalized MSE | Pooled3D position RMSE(m) |', '|---|---|---:|---:|']
        for panel, values in results.items():
            for name, row in {'CV16': values['cv16'], **values['saved_families'], **values['saved_references']}.items():
                lines.append(f"| {panel} | {name} | {row['mse']:.9f} | {row['position_rmse_m']:.9f} |")
        lines += ['', 'CV16 uses only context endpoints16/31. No fitted parameters, future actions, target feedback, or orientation projection.',
                  'New post hoc forecasts are not a saved-output-only audit. No architecture or primary-gate claim follows from this diagnostic.']
        (out / 'report.md').write_text('\n'.join(lines) + '\n')
        for path, digest in inputs.items():
            require(sha(path) == digest, 'Input changed during diagnostic')
        files = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
        write(out / 'completed.json', {'status': 'completed', 'source_sha256': sha(__file__), 'inputs': inputs,
              'files': files, 'new_deterministic_reference_forecasts': new_forecasts, 'new_neural_calls': 0,
              'new_native_calls': 0, 'new_fits': 0, 'new_random_draws': 0,
              'wall_seconds_before_receipt': time.perf_counter() - start})
    except BaseException as error:
        write(out / 'failed.json', {'status': 'failed', 'error': repr(error),
              'new_deterministic_reference_forecasts': new_forecasts, 'wall_seconds': time.perf_counter() - start})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    for name in ('experiment', 'report', 'out'):
        parser.add_argument('--' + name, type=Path)
    for name in ('protocol-sha256', 'completed-sha256', 'report-sha256'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(synthetic_causality_test()))
    else:
        require(all(value is not None for key, value in vars(args).items() if key != 'self_test'), 'All artifact arguments required')
        run(args.experiment, args.report, args.out, args.protocol_sha256, args.completed_sha256, args.report_sha256)
