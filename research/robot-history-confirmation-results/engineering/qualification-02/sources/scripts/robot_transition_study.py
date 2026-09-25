"""Bounded switching dynamics on exposed development data, with causal controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import robot_coupling_study as old
import torch
from torch import nn

from openjev.research.bounded_robot_transition import BoundedLPV
from openjev.research.causal_robot_ridge import fit as fit_ridge
from openjev.research.causal_robot_ridge import predict as predict_ridge

ROOT = old.ROOT
VERSION = 'robot-transition-study-v1'
ARMS = ('lpv_recurrent', 'lpv_instant', 'lpv_constant', 'gru_residual')
REFERENCES = ('causal_ridge_1', 'causal_ridge_100', 'linear_frozen', 'persistence')
SOURCES = (*old.SOURCES, 'src/openjev/research/bounded_robot_transition.py',
           'src/openjev/research/causal_robot_ridge.py', 'scripts/robot_transition_study.py',
           'tests/test_bounded_robot_transition.py', 'tests/test_causal_robot_ridge.py',
           'tests/test_robot_transition_study.py', 'research/robot-transition-protocol.md')


def config():
    cfg = old.config()
    cfg.update(version=VERSION, arms=list(ARMS), updates=4096, train_horizon=128,
               learning_rates=[.001, .003], window_seed_offset=520000,
               fit_cap_seconds=600., wall_cap_seconds=7200., mean_reduction=.05,
               paired_reduction=.02, latency_ratio=2., exposed_dev=True)
    return cfg


def authenticate(registration):
    path = Path(registration).resolve()
    raw = path.read_bytes()
    plan = json.loads(raw)
    old.require(plan['version'] == VERSION and plan['config'] == config(), 'exact transition config required')
    old.require(set(plan['sources']) == set(SOURCES), 'exact transition source roster required')
    for name, pin in plan['sources'].items():
        old.require(old.descriptor(ROOT / name) == pin, 'transition source changed: ' + name)
    _prior, prior_sha = old.authenticate(ROOT / 'research/robot-coupling-registration.json')
    old.require(prior_sha == plan['parent_registration_sha256'], 'parent registration changed')
    closure = plan['parent_closure']
    old.require(set(closure) == {'manifest', 'receipt', 'process', 'audit', 'audit_process'}, 'complete parent closure required')
    parent = {}
    for key, item in closure.items():
        old.require(old.descriptor(item['path']) == {k: item[k] for k in ('sha256', 'bytes')}, 'parent closure pin: ' + key)
        parent[key] = json.loads(Path(item['path']).read_text())
    old.require(parent['receipt']['status'] == 'PASS' and parent['receipt']['registration_sha256'] == prior_sha
                and parent['process']['returncode'] == 0 and parent['audit_process']['returncode'] == 0
                and parent['audit']['status'] == 'PASS' and parent['audit']['agreement'], 'closed successful parent execution/audit required')
    old.require(parent['audit']['registration_sha256'] == prior_sha
                and parent['audit']['inputs']['manifest'] == closure['manifest']
                and parent['audit']['inputs']['producer_receipt'] == closure['receipt']
                and parent['audit']['inputs']['run_receipt'] == closure['process'], 'parent audit closure join required')
    expected = {f'{split}-data-{name}.npz' for split in ('fit', 'dev') for name in old.PARTITIONS[split]}
    expected |= {'normalizers.npz', 'references.npz'}
    old.require(set(plan['data']) == expected, 'exact saved FIT/DEV data roster required')
    old.require(all(os.environ.get(key) == '1' for key in old.THREADS), 'single-thread environment required')
    for name, item in plan['data'].items():
        old.require(Path(item['path']).is_absolute(), 'absolute data path required')
        old.require(parent['manifest']['files'][name] == {k: item[k] for k in ('sha256', 'bytes')}, 'saved data must join parent manifest')
        old.require(old.descriptor(item['path']) == {k: item[k] for k in ('sha256', 'bytes')}, 'saved data pin changed')
    return plan, hashlib.sha256(raw).hexdigest()


def load_arrays(plan, filename):
    """Only the exact registered saved-array roster is accessible; preserve layout."""
    old.require(filename in plan['data'], 'unregistered saved array')
    item = plan['data'][filename]
    old.require(old.descriptor(item['path']) == {k: item[k] for k in ('sha256', 'bytes')}, 'saved data changed before load')
    with np.load(item['path'], allow_pickle=False) as data:
        return {key: data[key].copy(order='K') for key in data.files}


class LPVAdapter(nn.Module):
    """Reuse the frozen training/checkpoint loop without changing its interface."""

    def __init__(self, kind, seed):
        super().__init__()
        self.cell = BoundedLPV(kind=kind, seed=seed)

    def condition(self, q, u):
        return self._numeric_call(self.cell.condition, q, u)

    def forward(self, future, state):
        return self._numeric_call(self.cell.rollout, future, state)

    @staticmethod
    def _numeric_call(function, *args):
        try:
            return function(*args)
        except ValueError as exc:
            if str(exc) != 'nonfinite LPV output; no rollout clipping or repair':
                raise
            raise old.FitFailure(str(exc)) from exc


def model_for(arm, seed, linear):
    old.require(arm in ARMS, 'declared transition family required')
    return old.GRUResidual(seed, linear) if arm == 'gru_residual' else LPVAdapter(arm.removeprefix('lpv_'), seed)


def resource_model(model, arm):
    count = 28 if arm == 'gru_residual' else (20 if arm == 'lpv_recurrent' else 12)
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
            'buffer_bytes': sum(p.numel() * p.element_size() for p in model.buffers()),
            'state_scalars': count, 'state_bytes': 4 * count, 'normalizer_bytes': 192,
            'inactive_parameters': 192 if arm == 'lpv_instant' else 0, 'dtype': 'float32',
            'input_bytes': 9216, 'output_bytes': 6144, 'temporary_workspace': 'not measured'}


def ridge_batch(records, cfg):
    batches = []
    for index, data in enumerate(records):
        starts = np.arange(cfg['skip'], len(data['q']) - cfg['context'] - 128 + 1,
                           cfg['direct_fit_stride'], dtype=np.int64)
        batches.append(old.window_batch(records, {'record': np.full(len(starts), index, dtype=np.int64), 'start': starts}, 32, 128))
    return {key: np.concatenate([b[key] for b in batches]) for key in batches[0]}


def select(rows, cfg):
    options, selected = {}, {}
    for arm in ARMS:
        options[arm] = []
        for rate in cfg['learning_rates']:
            subset = [r for r in rows if r['arm'] == arm and r['learning_rate'] == rate and r['horizon'] == 128]
            expected = {(name, seed) for name in cfg['partitions']['dev'] for seed in cfg['seeds']}
            valid = len(subset) == len(expected) and {(r['recording'], r['seed']) for r in subset} == expected and all(old.valid_metric_row(r) for r in subset)
            score = float(np.sqrt(sum(r['metrics']['standardized_sse'] for r in subset) / sum(r['metrics']['scalars'] for r in subset))) if valid else None
            options[arm].append({'rate': rate, 'eligible': valid, 'pooled_rmse': score})
        eligible = [o for o in options[arm] if o['eligible']]
        selected[arm] = min(eligible, key=lambda o: (o['pooled_rmse'], o['rate']))['rate'] if eligible else None
    ridge_options = []
    for arm in REFERENCES[:2]:
        subset = [r for r in rows if r['arm'] == arm and r['horizon'] == 128]
        eligible = len(subset) == 2 and {r['recording'] for r in subset} == set(cfg['partitions']['dev']) and all(old.valid_metric_row(r) for r in subset)
        score = float(np.sqrt(sum(r['metrics']['standardized_sse'] for r in subset) / sum(r['metrics']['scalars'] for r in subset))) if eligible else None
        ridge_options.append({'arm': arm, 'eligible': eligible, 'pooled_rmse': score})
    eligible = [o for o in ridge_options if o['eligible']]
    return {'selected_rates': selected, 'options': options, 'ridge_options': ridge_options,
            'selected_ridge': min(eligible, key=lambda o: (o['pooled_rmse'], o['arm']))['arm'] if eligible else None,
            'scope': 'same already-exposed DEV2; three individual fits, no seed selection or ensemble'}


def evaluate_rule(rows, selection, resources, cfg):
    conditions, details = [], {}
    def add(name, passed):
        conditions.append({'name': name, 'passed': bool(passed)})
    rates = selection['selected_rates']
    add('all_selected_families_and_causal_ridge_eligible', all(rates[a] is not None for a in ARMS) and selection['selected_ridge'] is not None)
    for name in cfg['partitions']['dev']:
        current = [r for r in rows if r['recording'] == name and r['horizon'] == 128 and old.valid_metric_row(r)]
        neural = {a: {r['seed']: r['metrics'] for r in current if r['arm'] == a and r['learning_rate'] == rates[a]} for a in ARMS}
        means = {a: float(np.mean([v['standardized_rmse'] for v in m.values()])) for a, m in neural.items() if set(m) == set(cfg['seeds'])}
        refs = {r['arm']: r['metrics']['standardized_rmse'] for r in current if r['arm'] in REFERENCES}
        details[name] = {'means': means, 'references': refs}
        candidate = means.get('lpv_recurrent')
        for other in ARMS[1:]:
            add(name + '/mean_5pct/' + other, candidate is not None and other in means and candidate <= .95 * means[other])
            for seed in cfg['seeds']:
                left, right = neural['lpv_recurrent'].get(seed), neural[other].get(seed)
                add(f'{name}/seed{seed}_2pct/{other}', left is not None and right is not None and left['standardized_rmse'] <= .98 * right['standardized_rmse'])
        for other in REFERENCES[2:]:
            add(name + '/mean_5pct/' + other, candidate is not None and other in refs and candidate <= .95 * refs[other])
        ridge = refs.get(selection['selected_ridge'])
        add(name + '/within_5pct_causal_ridge', candidate is not None and ridge is not None and candidate <= 1.05 * ridge)
        for joint in range(6):
            complete = set(neural['lpv_recurrent']) == set(neural['gru_residual']) == set(cfg['seeds'])
            left = np.mean([m['per_joint_rmse_deg'][joint] for m in neural['lpv_recurrent'].values()]) if complete else None
            right = np.mean([m['per_joint_rmse_deg'][joint] for m in neural['gru_residual'].values()]) if complete else None
            add(f'{name}/joint{joint}_no_10pct_harm', complete and left <= 1.1 * right)
    def chosen_cost(arm):
        subset = [r for r in resources if r['arm'] == arm]
        if len(subset) != 3 or {r['seed'] for r in subset} != set(cfg['seeds']):
            return None
        return {'latency': float(np.median([r['timing']['median_seconds'] for r in subset])),
                'bytes': max(sum(r[k] for k in ('parameter_bytes', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')) for r in subset)}
    left, right = chosen_cost('lpv_recurrent'), chosen_cost('gru_residual')
    add('at_most_twice_gru_latency', left is not None and right is not None and left['latency'] <= cfg['latency_ratio'] * right['latency'])
    add('at_most_gru_numeric_storage', left is not None and right is not None and left['bytes'] <= right['bytes'])
    passed = sum(c['passed'] for c in conditions)
    return {'status': 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL' if passed == len(conditions) else 'DO_NOT_ADVANCE_TRANSITION',
            'passed': passed, 'total': len(conditions), 'conditions': conditions, 'details': details,
            'scope': 'adaptive development result, not independent confirmation, official benchmark score or novel architecture evidence'}


def timed_request(predict, physical, norm, cfg):
    def request():
        batch = {'q_context': (physical['q_context'] - norm['q_mean']) / norm['q_std'],
                 'u_context': (physical['u_context'] - norm['u_mean']) / norm['u_std'],
                 'future_u': (physical['future_u'] - norm['u_mean']) / norm['u_std']}
        with torch.no_grad():
            result = predict(batch) * norm['q_std'] + norm['q_mean']
        old.require(np.isfinite(result).all(), 'finite complete timed request')
        return result
    for _ in range(cfg['timing_warmups']):
        request()
    durations = []
    for _ in range(cfg['timing_repeats']):
        started = time.perf_counter()
        request()
        durations.append(time.perf_counter() - started)
    return {'seconds': durations, 'median_seconds': float(np.median(durations)), 'p95_seconds': float(np.percentile(durations, 95)),
            'scope': 'batch1 normalization, casting, context32, future128, denormalization and finite check; no model/disk load; spectral norms included'}


def run(registration, output):
    plan, sha = authenticate(registration)
    cfg = config()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    models, fits, rows, resources = {}, [], [], []
    def check():
        if time.monotonic() - started > cfg['wall_cap_seconds']:
            raise old.WholeStudyTimeout('transition study deadline exceeded')
    try:
        old.write_json(output / 'registration.json', plan)
        for name in SOURCES:
            path = output / 'sources' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
        physical_fit = [{'name': name, **load_arrays(plan, 'fit-data-' + name + '.npz')} for name in cfg['partitions']['fit']]
        norm = old.normalizers(physical_fit, cfg['skip'])
        prior_norm = load_arrays(plan, 'normalizers.npz')
        old.require(all(np.array_equal(norm[k], prior_norm[k]) for k in norm), 'same FIT-only normalization')
        np.savez_compressed(output / 'normalizers.npz', **norm)
        fit_data = [old.normalized_record(r, norm) for r in physical_fit]
        linear = load_arrays(plan, 'references.npz')['linear_frozen']
        np.savez_compressed(output / 'linear.npz', coefficient=linear)
        rb = ridge_batch(fit_data, cfg)
        ridges = {}
        for penalty in cfg['direct_ridges']:
            before = time.monotonic()
            arm = f'causal_ridge_{int(penalty)}'
            ridges[arm] = fit_ridge(**rb, penalty=penalty)
            np.savez_compressed(output / (arm + '.npz'), **{f'h{h:03d}': w for h, w in enumerate(ridges[arm].coefficients, 1)})
            old.write_json(output / (arm + '.json'), {**ridges[arm].metadata(), 'fit_seconds': time.monotonic() - before})
            check()
        for seed in cfg['seeds']:
            batches = old.make_batches([len(r['q']) for r in fit_data], seed, cfg)
            np.savez_compressed(output / f'batches-{seed}.npz', **batches)
            for ri, rate in enumerate(cfg['learning_rates']):
                for arm in ARMS:
                    check()
                    key = f'{arm}-{seed}-lr{ri}'
                    before = time.monotonic()
                    model = model_for(arm, seed, linear)
                    fit = old.train_one(model, fit_data, batches, cfg=cfg, lr=rate, folder=output / key, check=check, fit_started=before)
                    model.zero_grad(set_to_none=True)
                    models[key] = model
                    fits.append({'key': key, 'arm': arm, 'seed': seed, 'learning_rate': rate, 'fit': fit,
                                 'resources': resource_model(model, arm)})
                    old.write_json(output / f'completed-fit-{len(fits):02d}.json', fits[-1])
                    print(json.dumps({'completed': len(fits), 'key': key, 'status': fit['status'], 'updates': fit['completed_updates']}), flush=True)
        old.require(len(fits) == 24, 'all24 attempts before this study DEV load')
        old.write_json(output / 'checkpoint-barrier.json', {'fit_attempts': len(fits), 'dev_loads_this_run': 0, 'dev_exposed_prior': True,
                       'checkpoints': {f['key']: old.descriptor(output / f['key'] / 'final.npz') for f in fits}})
        physical_timing = None
        for name in cfg['partitions']['dev']:
            physical = {'name': name, **load_arrays(plan, 'dev-data-' + name + '.npz')}
            data = old.normalized_record(physical, norm)
            starts = old.dev_windows(len(data['q']), cfg)
            choices = {'record': np.zeros(len(starts), dtype=np.int64), 'start': starts}
            batch = old.window_batch([data], choices, 32, 128)
            np.savez_compressed(output / ('dev-windows-' + name + '.npz'), starts=starts, target=batch['target'])
            if physical_timing is None:
                physical_timing = old.window_batch([physical], {'record': np.zeros(1, dtype=np.int64), 'start': starts[:1]}, 32, 128)
            for f in fits:
                prediction, error = None, None
                if f['fit']['status'] == 'PASS':
                    try:
                        with torch.no_grad():
                            prediction = old.infer(models[f['key']], batch).numpy().astype(np.float64)
                    except old.FitFailure as exc:
                        error = {'type': 'NonfiniteEvaluation', 'message': str(exc)}
                    if prediction is not None:
                        np.savez_compressed(output / ('prediction-' + name + '-' + f['key'] + '.npz'), prediction=prediction)
                else:
                    error = {'type': 'FailedTrainingAttempt'}
                common = {'recording': name, 'arm': f['arm'], 'seed': f['seed'], 'learning_rate': f['learning_rate'], 'fit_key': f['key']}
                rows.extend(old.scored_rows(common, prediction, batch['target'], norm['q_std'], cfg, error))
            for arm in REFERENCES:
                prediction = predict_ridge(ridges[arm], batch['q_context'], batch['u_context'], batch['future_u']) if arm in ridges else old.reference_predict(arm, {'linear_frozen': linear}, batch)
                np.savez_compressed(output / ('prediction-' + name + '-' + arm + '.npz'), prediction=prediction)
                rows.extend(old.scored_rows({'recording': name, 'arm': arm, 'seed': None, 'learning_rate': None}, prediction, batch['target'], norm['q_std'], cfg))
            check()
        selection = select(rows, cfg)
        for f in fits:
            if selection['selected_rates'][f['arm']] == f['learning_rate']:
                def pred(batch, key=f['key']):
                    return old.infer(models[key], batch).numpy().astype(np.float64)
                resources.append({'arm': f['arm'], 'seed': f['seed'], 'learning_rate': f['learning_rate'], **f['resources'],
                                  'timing': timed_request(pred, physical_timing, norm, cfg)})
        for arm in REFERENCES:
            def pred(batch, name=arm):
                return predict_ridge(ridges[name], batch['q_context'], batch['u_context'], batch['future_u']) if name in ridges else old.reference_predict(name, {'linear_frozen': linear}, batch)
            count = ridges[arm].metadata()['coefficient_count'] if arm in ridges else (150 if arm == 'linear_frozen' else 0)
            state = 192 if arm in ridges else (18 if arm == 'linear_frozen' else 6)
            resources.append({'arm': arm, 'seed': None, 'parameters': count, 'parameter_bytes': count * 8,
                              'state_scalars': state, 'state_bytes': state * 8, 'normalizer_bytes': 192,
                              'buffer_bytes': 16 if arm in ridges else 0, 'dtype': 'float64',
                              'input_bytes': 9216, 'output_bytes': 6144, 'temporary_workspace': 'not measured',
                              'timing': timed_request(pred, physical_timing, norm, cfg)})
        result = evaluate_rule(rows, selection, resources, cfg)
        old.write_json(output / 'results.json', {'version': VERSION, 'config': cfg, 'rows': rows, 'selection': selection, 'result': result})
        old.write_json(output / 'resources.json', resources)
        old.write_json(output / 'fits.json', fits)
        after, after_sha = authenticate(registration)
        old.require(after == plan and after_sha == sha, 'registration unchanged')
        old.write_json(output / 'manifest.json', {'files': {str(p.relative_to(output)): old.descriptor(p) for p in sorted(output.rglob('*')) if p.is_file()}})
        check()
        old.write_json(output / 'receipt.json', {'status': 'PASS', 'registration_sha256': sha, 'seconds': time.monotonic() - started,
                       'fits': len(fits), 'rows': len(rows), 'scientific_result': result['status'],
                       'raw_decodes': 0, 'saved_fit_loads': 7, 'saved_dev_loads': 2, 'confirmation_access': False, 'official_test_access': False})
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        old.write_json(output / 'failure.json', {'status': 'FAILED', 'type': type(exc).__name__, 'message': str(exc),
                       'seconds': time.monotonic() - started, 'completed_fits': len(fits), 'metric_rows': len(rows)})
        raise
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.registration, args.output)
