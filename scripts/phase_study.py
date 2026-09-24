"""Precommitted Silverbox development comparison of a tiny nonlinear recurrence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from openjev.research.phase_memory import PhaseMemory
from openjev.research.silverbox_data import load_partition

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'phase-study-v1'
CSV_SHA = 'ae62d5a91230c10f76e6dd02c8a4fac3c9d4d8a95fbf50e87cb0c4885003e0f1'
SEEDS = (7301, 7302, 7303)
ARMS = ('fixed_phase', 'energy_phase', 'nonlinear_readout', 'gru16', 'cubic_ar2')
REFERENCES = ('static_cubic', 'fir128', 'fir512', 'cubic_ar2_frozen')
SOURCES = ('src/openjev/research/phase_memory.py', 'src/openjev/research/silverbox_data.py',
           'scripts/phase_study.py', 'scripts/audit_phase_study.py',
           'tests/test_phase_memory.py', 'tests/test_silverbox_data.py',
           'tests/test_phase_study.py', 'tests/test_audit_phase_study.py', 'research/phase-protocol.md')
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
QUAL_ARGV = ['.venv/bin/python', '-m', 'pytest', '-q', 'tests/test_phase_memory.py',
             'tests/test_silverbox_data.py', 'tests/test_phase_study.py', 'tests/test_audit_phase_study.py']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def config():
    return {'version': VERSION, 'seeds': list(SEEDS), 'arms': list(ARMS), 'references': list(REFERENCES),
            'updates': 2048, 'batch_size': 32, 'sequence_length': 256, 'train_burn': 64,
            'evaluation_burn': 512, 'window_seed_offset': 410000, 'ridge': 1e-6,
            'adam_lr': .003, 'ar2_adam_lr': .0001, 'adam_betas': [.9, .999], 'adam_eps': 1e-8,
            'gradient_clip': 1., 'wall_cap_seconds': 2400, 'fit_cap_seconds': 600,
            'partitions': {'fit': [40650, 83946], 'dev_a': [84446, 92638], 'dev_b': [93138, 101330]},
            'official_test_access': False, 'simulation_dtype': 'float32', 'ridge_dtype': 'float64',
            'prediction_storage_dtype': 'float64', 'scored_units': 'mV',
            'min_control_reduction': .1, 'baseline_noninferiority_ratio': 1.05,
            'competence_fit_std_fraction': .1}


def pin(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


class GRUModel(nn.Module):
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.gru = nn.GRU(1, 16, batch_first=True)
            self.readout = nn.Linear(16, 1)

    def forward(self, sequence):
        states, final = self.gru(sequence)
        return self.readout(states).squeeze(-1), final


class CubicAR2(nn.Module):
    def __init__(self, coefficient):
        super().__init__()
        self.coefficient = nn.Parameter(torch.as_tensor(coefficient, dtype=torch.float32).clone())

    def forward(self, sequence):
        previous = sequence.new_zeros(sequence.shape[0])
        older, previous_u = previous.clone(), previous.clone()
        outputs = []
        for index in range(sequence.shape[1]):
            u = sequence[:, index, 0]
            features = torch.stack((previous, older, u, previous_u,
                                    previous.square(), previous**3, torch.ones_like(previous)), dim=-1)
            value = (features * self.coefficient).sum(dim=-1)
            outputs.append(value)
            older, previous, previous_u = previous, value, u
        return torch.stack(outputs, dim=1), torch.stack((previous, older, previous_u), dim=-1)


def model_for(arm, seed, ar2_coefficient):
    if arm in ('fixed_phase', 'energy_phase', 'nonlinear_readout'):
        return PhaseMemory(arm, seed)
    if arm == 'gru16':
        return GRUModel(seed)
    require(arm == 'cubic_ar2', 'unknown trainable model')
    return CubicAR2(ar2_coefficient)


def weights(model):
    return {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}


def normalizers(u, y):
    require(len(u) == len(y) and np.isfinite(u).all() and np.isfinite(y).all(), 'finite FIT rows')
    values = {'u_mean': float(np.mean(u)), 'u_std': float(np.std(u)),
              'y_mean': float(np.mean(y)), 'y_std': float(np.std(y))}
    require(values['u_std'] > 0 and values['y_std'] > 0, 'nonconstant FIT signals')
    return values


def lag_design(u, length):
    # Column zero is intercept, then present input and descending input lags.
    padded = np.pad(u, (length-1, 0))
    lags = np.lib.stride_tricks.sliding_window_view(padded, length)[:, ::-1]
    return np.column_stack((np.ones(len(u)), lags))


def solve_ridge(phi, y):
    return np.linalg.solve(phi.T @ phi + 1e-6*np.eye(phi.shape[1]), phi.T @ y)


def fit_references(u, y):
    coefficients = {'static_cubic': solve_ridge(np.column_stack((np.ones(len(u)), u, u*u, u**3)), y)}
    for length in (128, 512):
        phi = lag_design(u, length)
        coefficients[f'fir{length}'] = solve_ridge(phi[length-1:], y[length-1:])
    previous, older = y[1:-1], y[:-2]
    phi = np.column_stack((previous, older, u[2:], u[1:-1], previous**2, previous**3, np.ones(len(previous))))
    coefficients['cubic_ar2_frozen'] = solve_ridge(phi, y[2:])
    return coefficients


def reference_predict(name, coefficient, u):
    if name == 'static_cubic':
        return np.column_stack((np.ones(len(u)), u, u*u, u**3)) @ coefficient
    if name.startswith('fir'):
        return lag_design(u, int(name[3:])) @ coefficient
    require(name == 'cubic_ar2_frozen', 'unknown reference')
    with torch.no_grad():
        prediction, _ = CubicAR2(coefficient)(torch.from_numpy(u.astype(np.float32))[None, :, None])
    return prediction[0].numpy().astype(np.float64)


def train_one(model, u, y, windows, *, cfg, arm, check=lambda: None, failure_folder=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['ar2_adam_lr'] if arm == 'cubic_ar2' else cfg['adam_lr'],
                                 betas=tuple(cfg['adam_betas']), eps=cfg['adam_eps'])
    losses, norms = [], []
    u_tensor, y_tensor = torch.from_numpy(u.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
    offset = torch.arange(cfg['sequence_length'])
    started = time.monotonic()
    try:
        for starts in windows:
            check()
            require(time.monotonic()-started <= cfg['fit_cap_seconds'], 'single fit exceeded frozen cap')
            indices = torch.from_numpy(starts)[:, None] + offset[None]
            batch_x, batch_y = u_tensor[indices][..., None], y_tensor[indices]
            optimizer.zero_grad(set_to_none=True)
            predictions, _ = model(batch_x)
            loss = (predictions[:, cfg['train_burn']:] - batch_y[:, cfg['train_burn']:]).square().mean()
            require(bool(torch.isfinite(loss)), 'nonfinite simulation loss; original study stops')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'nonfinite optimizer state')
            losses.append(float(loss.detach()))
            norms.append(float(norm.detach()))
            check()
            require(time.monotonic()-started <= cfg['fit_cap_seconds'], 'single fit exceeded frozen cap')
    except Exception as error:
        if failure_folder is not None:
            np.savez_compressed(failure_folder/'failed.npz', **weights(model))
            np.savez_compressed(failure_folder/'training-partial.npz', loss=np.asarray(losses), gradnorm=np.asarray(norms))
            write_json(failure_folder/'failure.json', {'type': type(error).__name__, 'message': str(error),
                'completed_updates': len(losses), 'elapsed_seconds': time.monotonic()-started,
                'outcome': 'ORIGINAL_STUDY_EXECUTION_FAILED_NO_RETRY'})
        raise
    saved_optimizer = {}
    for name, parameter in model.named_parameters():
        for field in ('exp_avg', 'exp_avg_sq', 'step'):
            saved_optimizer[f'{name}.{field}'] = optimizer.state[parameter][field].detach().numpy().copy()
    seconds = time.monotonic()-started
    require(seconds <= cfg['fit_cap_seconds'], 'single fit exceeded frozen cap')
    check()
    return {'loss': np.asarray(losses), 'gradnorm': np.asarray(norms)}, saved_optimizer, seconds


def metrics(prediction, normalized_target, y_std):
    require(prediction.shape == normalized_target.shape and np.isfinite(prediction).all(), 'finite full rollout required')
    delta = (prediction[512:] - normalized_target[512:]) * y_std * 1000
    return {'rmse_mv': float(np.sqrt(np.mean(delta**2))), 'mae_mv': float(np.mean(np.abs(delta))),
            'scored_rows': len(delta)}


def evaluate_rule(scores, y_std):
    conditions, summary = [{'name': 'all_completed_predictions_finite', 'passed': True}], {}
    for partition in ('dev_a', 'dev_b'):
        row = scores[partition]
        means = {arm: float(np.mean([row[f'{arm}/{seed}']['rmse_mv'] for seed in SEEDS])) for arm in ARMS}
        reference_values = {name: row[name]['rmse_mv'] for name in REFERENCES}
        best = min(means['gru16'], means['cubic_ar2'], *reference_values.values())
        summary[partition] = {'family_mean_rmse_mv': means, 'reference_rmse_mv': reference_values,
                              'best_conventional_rmse_mv': best}
        for control in ('fixed_phase', 'nonlinear_readout'):
            conditions.append({'name': f'{partition}_mean_energy_improves_{control}_10pct',
                               'passed': means['energy_phase'] <= .9 * means[control]})
            for seed in SEEDS:
                conditions.append({'name': f'{partition}_seed{seed}_energy_not_worse_{control}',
                    'passed': row[f'energy_phase/{seed}']['rmse_mv'] <= row[f'{control}/{seed}']['rmse_mv']})
        conditions.append({'name': f'{partition}_within_5pct_best_conventional',
                           'passed': means['energy_phase'] <= 1.05 * best})
        conditions.append({'name': f'{partition}_competent_below_10pct_fit_std',
                           'passed': means['energy_phase'] <= .1 * y_std * 1000})
    passed = sum(bool(c['passed']) for c in conditions)
    return {'version': VERSION, 'scope': 'Silverbox internal development simulation; official TEST unused',
            'scores': scores, 'summary': summary, 'conditions': conditions, 'passed': passed,
            'total': len(conditions), 'outcome': 'ADVANCE_PHASE_MECHANISM' if passed == len(conditions)
            else 'DO_NOT_ADVANCE_PHASE_MECHANISM'}


def authenticate_qualification(path):
    raw = Path(path).read_bytes()
    receipt = json.loads(raw)
    expected = {s: pin((ROOT/s).read_bytes()) for s in SOURCES}
    require(receipt['state'] == 'EXITED' and receipt['returncode'] == 0 and receipt['argv'] == QUAL_ARGV,
            'closed original passing qualification with exact test command')
    require(receipt['sources_before'] == receipt['sources_after'] == expected, 'qualified exact source set')
    require(receipt['thread_env'] == {k: '1' for k in THREADS}, 'single thread qualification')
    require(Path(receipt['log_path']).name == receipt['log_path'], 'sibling qualification log')
    require(pin((Path(path).parent/receipt['log_path']).read_bytes()) == receipt['log'], 'original test log')
    return raw


def register(args):
    raw = Path(args.csv).read_bytes()
    require(pin(raw)['sha256'] == CSV_SHA, 'pinned Silverbox CSV')
    qualification = authenticate_qualification(args.qualification)
    require(all(os.environ.get(k) == '1' for k in THREADS), 'single-thread execution required')
    write_json(args.registration, {'config': config(), 'csv': pin(raw), 'qualification': pin(qualification),
        'sources': {s: pin((ROOT/s).read_bytes()) for s in SOURCES},
        'environment': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
                        'threads': {k: '1' for k in THREADS}, 'device': 'cpu'}})


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(all(os.environ.get(k) == '1' for k in THREADS), 'single-thread execution required')
    cfg = config()
    registration = Path(args.registration).read_bytes()
    reg = json.loads(registration)
    require(reg['config'] == cfg and set(reg['sources']) == set(SOURCES), 'exact registration')
    require(reg['environment'] == {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
            'threads': {k: '1' for k in THREADS}, 'device': 'cpu'}, 'frozen numerical environment')
    qualification = authenticate_qualification(args.qualification)
    require(pin(qualification) == reg['qualification'], 'qualified original process')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for source, expected in {Path(args.registration).resolve().relative_to(ROOT).as_posix(): pin(registration), **reg['sources']}.items():
        require(pin((ROOT/source).read_bytes()) == expected == pin(subprocess.check_output(['git', 'show', f'{commit}:{source}'], cwd=ROOT)),
                'sources and registration committed before scientific fitting')
    raw = Path(args.csv).read_bytes()
    require(pin(raw) == reg['csv'] and reg['csv']['sha256'] == CSV_SHA, 'original source CSV')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def check():
        if time.monotonic()-started > cfg['wall_cap_seconds']:
            raise TimeoutError('original phase study exceeded registered cap')
    (output/'registration.json').write_bytes(registration)
    (output/'qualification.json').write_bytes(qualification)
    for source in SOURCES:
        dest = output/'sources'/source
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/source).read_bytes())
    fit = load_partition(raw, 'fit')
    np.savez_compressed(output/'data-fit.npz', raw_indices=fit.raw_indices, u=fit.u, y=fit.y)
    norm = normalizers(fit.u, fit.y)
    write_json(output/'normalization.json', norm)
    u, y = (fit.u-norm['u_mean'])/norm['u_std'], (fit.y-norm['y_mean'])/norm['y_std']
    references = fit_references(u, y)
    np.savez_compressed(output/'references.npz', **references)
    resources, fit_times, retained_models = {}, {}, {}
    for seed in SEEDS:
        windows = np.random.default_rng(seed+cfg['window_seed_offset']).integers(
            0, len(u)-cfg['sequence_length']+1, size=(cfg['updates'], cfg['batch_size']), dtype=np.int64)
        np.save(output/f'windows-seed{seed}.npy', windows, allow_pickle=False)
        for arm in ARMS:
            check()
            key = f'{arm}/{seed}'
            folder = output/'models'/arm/str(seed)
            folder.mkdir(parents=True, exist_ok=False)
            model = model_for(arm, seed, references['cubic_ar2_frozen'])
            np.savez_compressed(folder/'initial.npz', **weights(model))
            trace, optimizer, seconds = train_one(model, u, y, windows, cfg=cfg, arm=arm, check=check, failure_folder=folder)
            np.savez_compressed(folder/'training.npz', **trace)
            np.savez_compressed(folder/'optimizer.npz', **optimizer)
            np.savez_compressed(folder/'final.npz', **weights(model))
            fit_times[key] = seconds
            count = sum(p.numel() for p in model.parameters())
            state = 4 if arm in ('energy_phase', 'fixed_phase', 'nonlinear_readout') else (16 if arm == 'gru16' else 3)
            resources[key] = {'parameters': count, 'parameter_bytes': count*4, 'state_scalars': state,
                              'state_bytes': state*4, 'normalization_bytes': 32,
                              'logical_total_bytes': count*4+state*4+32}
            retained_models[key] = model.eval()
            write_json(folder/'fit.json', {'arm': arm, 'seed': seed, 'updates': cfg['updates'],
                       'elapsed_seconds': seconds, 'loss_initial_batch': float(trace['loss'][0]),
                       'loss_final_batch': float(trace['loss'][-1]), 'state_initialization': 'zero every training sequence',
                       'dev_or_test_access': False})
            print(json.dumps({'finished': key, 'seconds': round(seconds, 3)}), flush=True)
    scores = {}
    for partition in ('dev_a', 'dev_b'):
        check()
        data = load_partition(raw, partition)
        np.savez_compressed(output/f'data-{partition}.npz', raw_indices=data.raw_indices, u=data.u, y=data.y)
        dev_u, dev_y = (data.u-norm['u_mean'])/norm['u_std'], (data.y-norm['y_mean'])/norm['y_std']
        scores[partition] = {}
        for key, model in retained_models.items():
            with torch.no_grad():
                prediction, _ = model(torch.from_numpy(dev_u.astype(np.float32))[None, :, None])
            prediction = prediction[0].numpy().astype(np.float64)
            require(np.isfinite(prediction).all(), 'divergent final simulation; original study stops')
            np.save(output/f'prediction-{key.replace("/", "-")}-{partition}.npy', prediction, allow_pickle=False)
            scores[partition][key] = metrics(prediction, dev_y, norm['y_std'])
        for name, coefficient in references.items():
            prediction = reference_predict(name, coefficient, dev_u)
            require(np.isfinite(prediction).all(), 'divergent classical simulation; original study stops')
            np.save(output/f'prediction-{name}-{partition}.npy', prediction, allow_pickle=False)
            scores[partition][name] = metrics(prediction, dev_y, norm['y_std'])
    result = evaluate_rule(scores, norm['y_std'])
    write_json(output/'results.json', result)
    write_json(output/'resources.json', {'trained_models': resources,
        'references': {name: {'coefficients': len(v), 'saved_coefficient_bytes': len(v)*8,
            'parameter_bytes': len(v)*(4 if name == 'cubic_ar2_frozen' else 8),
            'state_bytes': 12 if name == 'cubic_ar2_frozen' else ((int(name[3:])-1)*8 if name.startswith('fir') else 0),
            'normalization_bytes': 32,
            'logical_total_bytes': len(v)*(4 if name == 'cubic_ar2_frozen' else 8) + 32 +
                (12 if name == 'cubic_ar2_frozen' else ((int(name[3:])-1)*8 if name.startswith('fir') else 0))}
                for name, v in references.items()},
        'scope': 'Parameter/state footprint; no wall-time speedup or matched-training-compute claim',
        'excluded': 'Python/native workspace, optimizer and training/audit data; no hidden measurement archive at inference'})
    require(all(pin((ROOT/s).read_bytes()) == expected for s, expected in reg['sources'].items()), 'frozen sources after execution')
    check()
    write_json(output/'run.json', {'version': VERSION, 'registration': pin(registration), 'registration_commit': commit,
               'csv': pin(raw), 'models_fitted': len(retained_models), 'ridge_fits': 4,
               'optimizer_updates': len(retained_models)*cfg['updates'], 'fit_seconds': fit_times,
               'elapsed_seconds': time.monotonic()-started, 'numeric_partitions_loaded': ['fit', 'dev_a', 'dev_b'],
               'official_test_values_read': 0, 'source_pins_verified_before_and_after': True})
    write_json(output/'manifest.json', {'files': {p.relative_to(output).as_posix(): pin(p.read_bytes())
               for p in sorted(output.rglob('*')) if p.is_file()}})
    check()
    print(json.dumps({'outcome': result['outcome'], 'passed': result['passed'], 'total': result['total']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run'))
    parser.add_argument('--csv', required=True)
    parser.add_argument('--registration', required=True)
    parser.add_argument('--qualification', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.action == 'register':
        register(args)
    else:
        require(bool(args.output), 'new exclusive output directory required')
        run(args)


if __name__ == '__main__':
    main()
