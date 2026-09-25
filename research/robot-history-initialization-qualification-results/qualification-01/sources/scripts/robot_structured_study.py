"""Structured current-state transitions on exposed DEV, with cached strong controls.

No parent fit is rerun. All new attempts close before this campaign loads DEV.
Established matrix/gating structures; no architecture or Rust speed claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import robot_coupling_study as old
import robot_transition_study as parent
import torch
from torch import nn

from openjev.research.causal_robot_ridge import CausalRobotRidge
from openjev.research.causal_robot_ridge import predict as predict_ridge
from openjev.research.structured_robot_transition import StructuredRobotTransition

ROOT = old.ROOT
VERSION = 'robot-structured-study-v1'
ARMS = ('householder', 'dense_bounded', 'dense_unbounded', 'dense_mlp', 'gru32')
ALL_ARMS = (*ARMS, 'legacy_instant')
REFERENCES = parent.REFERENCES
LEGACY_FILES = ('initial.npz', 'final.npz', 'optimizer.npz', 'trace.json', 'fit-receipt.json')
PARENT_REGISTRATION = 'research/robot-transition-registration.json'
PARENT_FOLDER = ROOT / 'output/robot-transition-study-v1'
PAYLOADS = ('normalizers.npz', 'linear.npz', 'causal_ridge_1.npz', 'causal_ridge_1.json',
            'causal_ridge_100.npz', 'causal_ridge_100.json', 'fits.json',
            *(f'batches-{seed}.npz' for seed in old.SEEDS),
            *(f'lpv_instant-{seed}-lr{ri}/{name}' for seed in old.SEEDS for ri in range(2) for name in LEGACY_FILES))
SOURCES = (*parent.SOURCES, 'src/openjev/research/compact_robot_gate.py',
           'tests/test_compact_robot_gate.py', 'src/openjev/research/structured_robot_transition.py',
           'tests/test_structured_robot_transition.py', 'scripts/robot_structured_study.py',
           'tests/test_robot_structured_study.py', 'research/robot-structured-protocol.md')


def config():
    cfg = parent.config()
    cfg.update(version=VERSION, arms=list(ARMS), comparison_arms=list(ALL_ARMS),
               fit_cap_seconds=900., wall_cap_seconds=10800.,
               mean_ratio=1.02, paired_ratio=1.05, latency_ratio=.8, storage_ratio=.8,
               legacy_refit=False, causal_ridge_refit=False, batch_reuse=True)
    # Drop parent gate settings that no longer describe this prospective rule.
    cfg.pop('mean_reduction')
    cfg.pop('paired_reduction')
    return cfg


def _pin(item):
    old.require(set(item) == {'path', 'sha256', 'bytes'} and Path(item['path']).is_absolute(), 'absolute complete descriptor required')
    value = {k: item[k] for k in ('sha256', 'bytes')}
    old.require(old.descriptor(item['path']) == value, 'pinned file changed: ' + item['path'])
    return value


def authenticate(registration):
    path = Path(registration).resolve()
    raw = path.read_bytes()
    plan = json.loads(raw)
    old.require(plan['version'] == VERSION and plan['config'] == config(), 'exact structured config required')
    old.require(set(plan['sources']) == set(SOURCES), 'exact structured source roster required')
    for name, pin in plan['sources'].items():
        old.require(old.descriptor(ROOT / name) == pin, 'structured source changed: ' + name)
    prior, prior_sha = parent.authenticate(ROOT / PARENT_REGISTRATION)
    old.require(prior_sha == plan['parent_registration_sha256'], 'parent registration changed')
    old.require(plan['data'] == prior['data'], 'unaltered grandparent FIT/DEV descriptors required')
    closure = plan['parent_closure']
    old.require(set(closure) == {'manifest', 'receipt', 'process', 'audit', 'audit_process'}, 'complete parent closure required')
    closed = {}
    for key, item in closure.items():
        _pin(item)
        closed[key] = json.loads(Path(item['path']).read_text())
    old.require(Path(closure['manifest']['path']) == PARENT_FOLDER / 'manifest.json'
                and Path(closure['receipt']['path']) == PARENT_FOLDER / 'receipt.json', 'exact parent study closure paths required')
    old.require(closed['receipt']['status'] == 'PASS' and closed['receipt']['registration_sha256'] == prior_sha
                and closed['process']['returncode'] == closed['audit_process']['returncode'] == 0
                and closed['process']['registration_sha256'] == prior_sha
                and closed['audit']['status'] == 'PASS' and closed['audit']['agreement'] is True,
                'closed successful parent execution/audit required')
    old.require(closed['audit']['registration_sha256'] == prior_sha
                and closed['audit']['inputs']['manifest'] == closure['manifest']
                and closed['audit']['inputs']['producer_receipt'] == closure['receipt']
                and closed['audit']['inputs']['run_receipt'] == closure['process']
                and closed['audit']['inputs']['parent_data'] == plan['data']
                and closed['audit_process']['audit_output'] == {k: closure['audit'][k] for k in ('sha256', 'bytes')},
                'parent original audit joins required')
    old.require(set(plan['parent_payloads']) == set(PAYLOADS), 'complete exact cached parent payload roster required')
    for name, item in plan['parent_payloads'].items():
        old.require(Path(item['path']) == PARENT_FOLDER / name, 'canonical parent payload path required')
        old.require(_pin(item) == closed['manifest']['files'][name], 'parent manifest payload join: ' + name)
    return plan, hashlib.sha256(raw).hexdigest()


def load_arrays(plan, filename):
    return parent.load_arrays(plan, filename)


def load_parent_arrays(plan, filename):
    old.require(filename in PAYLOADS and filename.endswith('.npz'), 'registered parent NPZ only')
    item = plan['parent_payloads'][filename]
    _pin(item)
    with np.load(item['path'], allow_pickle=False) as data:
        return {key: data[key].copy(order='K') for key in data.files}


def copy_parent(plan, filename, destination):
    old.require(filename in PAYLOADS, 'registered parent payload only')
    item = plan['parent_payloads'][filename]
    _pin(item)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as handle:
        handle.write(Path(item['path']).read_bytes())
    old.require(old.descriptor(destination) == {k: item[k] for k in ('sha256', 'bytes')}, 'exact copied parent bytes')


class StructuredAdapter(nn.Module):
    def __init__(self, kind, seed):
        super().__init__()
        self.cell = StructuredRobotTransition(kind, seed)

    @staticmethod
    def _numeric_call(function, *args):
        try:
            return function(*args)
        except ValueError as exc:
            if str(exc) != 'nonfinite structured transition output; no repair':
                raise
            raise old.FitFailure(str(exc)) from exc

    def condition(self, q, u):
        return self._numeric_call(self.cell.condition, q, u)

    def forward(self, future, state):
        return self._numeric_call(self.cell, future, state)


class GRU32(old.GRUResidual):
    """The qualified residual GRU equations with hidden width32, no other change."""
    def __init__(self, seed, coefficient):
        nn.Module.__init__(self)
        self.base_weight = nn.Parameter(torch.as_tensor(coefficient, dtype=torch.float32).clone())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.gru = nn.GRUCell(24, 32, dtype=torch.float32)
            self.head = nn.Linear(32, 6, dtype=torch.float32)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()

    def condition(self, q_context, u_context):
        old.require(q_context.shape == u_context.shape and q_context.ndim == 3 and q_context.shape[1] >= 2
                    and q_context.shape[-1] == 6, 'GRU context shape')
        hidden = q_context.new_zeros(len(q_context), 32)
        for t in range(1, q_context.shape[1] - 1):
            x = torch.cat((q_context[:, t], q_context[:, t-1], u_context[:, t], u_context[:, t-1]), dim=-1)
            hidden = self.gru(x, hidden)
        return old.GRUState(q_context[:, -1].clone(), q_context[:, -2].clone(), u_context[:, -2].clone(), hidden)


def model_for(arm, seed, linear):
    old.require(arm in ALL_ARMS, 'declared structured family required')
    if arm == 'legacy_instant':
        return parent.model_for('lpv_instant', seed, linear)
    return GRU32(seed, linear) if arm == 'gru32' else StructuredAdapter(arm, seed)


def resource_model(model, arm):
    count = 50 if arm == 'gru32' else 12
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
            'buffer_bytes': sum(p.numel() * p.element_size() for p in model.buffers()),
            'state_scalars': count, 'state_bytes': 4 * count, 'normalizer_bytes': 192,
            'inactive_parameters': 192 if arm == 'legacy_instant' else 0, 'dtype': 'float32',
            'input_bytes': 9216, 'output_bytes': 6144, 'temporary_workspace': 'not measured'}


def select(rows, cfg):
    options, selected = {}, {}
    for arm in ALL_ARMS:
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
            'scope': 'same already-exposed DEV2; all three seeds per recipe; legacy rates reselected from six frozen fits, no refit'}


def evaluate_rule(rows, selection, resources, cfg):
    conditions, details = [], {}
    def add(name, passed):
        conditions.append({'name': name, 'passed': bool(passed)})
    rates = selection['selected_rates']
    add('all_selected_families_and_causal_ridge_eligible', all(rates[a] is not None for a in ALL_ARMS) and selection['selected_ridge'] is not None)
    for name in cfg['partitions']['dev']:
        current = [r for r in rows if r['recording'] == name and r['horizon'] == 128 and old.valid_metric_row(r)]
        neural = {a: {r['seed']: r['metrics'] for r in current if r['arm'] == a and r['learning_rate'] == rates[a]} for a in ALL_ARMS}
        means = {a: float(np.mean([v['standardized_rmse'] for v in m.values()])) for a, m in neural.items() if set(m) == set(cfg['seeds'])}
        refs = {r['arm']: r['metrics']['standardized_rmse'] for r in current if r['arm'] in REFERENCES}
        details[name] = {'means': means, 'references': refs}
        candidate = means.get('householder')
        for other in ALL_ARMS[1:]:
            add(name + '/mean_within_2pct/' + other, candidate is not None and other in means and candidate <= cfg['mean_ratio'] * means[other])
            for seed in cfg['seeds']:
                left, right = neural['householder'].get(seed), neural[other].get(seed)
                add(f'{name}/seed{seed}_within_5pct/{other}', left is not None and right is not None and left['standardized_rmse'] <= cfg['paired_ratio'] * right['standardized_rmse'])
        for other in REFERENCES[2:]:
            add(name + '/mean_5pct/' + other, candidate is not None and other in refs and candidate <= .95 * refs[other])
        ridge = refs.get(selection['selected_ridge'])
        add(name + '/within_5pct_causal_ridge', candidate is not None and ridge is not None and candidate <= 1.05 * ridge)
        for joint in range(6):
            complete = set(neural['householder']) == set(neural['gru32']) == set(cfg['seeds'])
            left = np.mean([m['per_joint_rmse_deg'][joint] for m in neural['householder'].values()]) if complete else None
            right = np.mean([m['per_joint_rmse_deg'][joint] for m in neural['gru32'].values()]) if complete else None
            add(f'{name}/joint{joint}_no_10pct_harm', complete and left <= 1.1 * right)
    def chosen_cost(arm):
        subset = [r for r in resources if r['arm'] == arm]
        if len(subset) != 3 or {r['seed'] for r in subset} != set(cfg['seeds']):
            return None
        return {'latency': float(np.median([r['timing']['median_seconds'] for r in subset])),
                'bytes': max(sum(r[k] for k in ('parameter_bytes', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')) for r in subset)}
    left, right = chosen_cost('householder'), chosen_cost('dense_bounded')
    add('at_most_80pct_dense_bounded_latency', left is not None and right is not None and left['latency'] <= cfg['latency_ratio'] * right['latency'])
    add('at_most_80pct_dense_bounded_numeric_storage', left is not None and right is not None and left['bytes'] <= cfg['storage_ratio'] * right['bytes'])
    passed = sum(c['passed'] for c in conditions)
    return {'status': 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL' if passed == len(conditions) else 'DO_NOT_ADVANCE_STRUCTURED_TRANSITION',
            'passed': passed, 'total': len(conditions), 'conditions': conditions, 'details': details,
            'scope': 'adaptive development utility screen; not independent confirmation, official benchmark score, Rust speedup or novel architecture evidence'}


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
            'scope': 'batch1 normalization, casting, context32, future128, denormalization and finite check; no model/disk load; outer validation and operator preparation included; eager CPU'}


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
            raise old.WholeStudyTimeout('structured study deadline exceeded')
    try:
        old.write_json(output / 'registration.json', plan)
        for name in SOURCES:
            path = output / 'sources' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
        physical_fit = [{'name': name, **load_arrays(plan, 'fit-data-' + name + '.npz')} for name in cfg['partitions']['fit']]
        norm = old.normalizers(physical_fit, cfg['skip'])
        prior_norm = load_arrays(plan, 'normalizers.npz')
        parent_norm = load_parent_arrays(plan, 'normalizers.npz')
        old.require(set(norm) == set(prior_norm) == set(parent_norm)
                    and all(np.array_equal(norm[k], prior_norm[k])
                            and np.array_equal(norm[k], parent_norm[k]) for k in norm), 'same FIT-only normalization')
        np.savez_compressed(output / 'normalizers.npz', **norm)
        fit_data = [old.normalized_record(r, norm) for r in physical_fit]
        linear = load_parent_arrays(plan, 'linear.npz')['coefficient']
        copy_parent(plan, 'normalizers.npz', output / 'parent-normalizers.npz')
        copy_parent(plan, 'linear.npz', output / 'linear.npz')
        ridges = {}
        for penalty in cfg['direct_ridges']:
            arm = f'causal_ridge_{int(penalty)}'
            saved = load_parent_arrays(plan, arm + '.npz')
            item = plan['parent_payloads'][arm + '.json']
            _pin(item)
            metadata = json.loads(Path(item['path']).read_text())
            old.require(set(saved) == {f'h{h:03d}' for h in range(1,129)}, 'complete cached ridge bank')
            ridges[arm] = CausalRobotRidge(tuple(saved[f'h{h:03d}'] for h in range(1,129)), penalty, metadata['fit_rows'])
            old.require(ridges[arm].metadata() == {k: v for k,v in metadata.items() if k != 'fit_seconds'}, 'cached ridge metadata')
            copy_parent(plan, arm + '.npz', output / (arm + '.npz'))
            copy_parent(plan, arm + '.json', output / (arm + '.json'))
            check()
        for seed in cfg['seeds']:
            batches = load_parent_arrays(plan, f'batches-{seed}.npz')
            expected_batches = old.make_batches([len(r['q']) for r in fit_data], seed, cfg)
            old.require(set(batches) == set(expected_batches) and all(np.array_equal(batches[k], expected_batches[k]) for k in batches), 'exact parent paired batches')
            copy_parent(plan, f'batches-{seed}.npz', output / f'batches-{seed}.npz')
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
                                 'resources': resource_model(model, arm), 'origin': 'fresh'})
                    old.write_json(output / f'completed-fit-{len(fits):02d}.json', fits[-1])
                    print(json.dumps({'completed': len(fits), 'key': key, 'status': fit['status'], 'updates': fit['completed_updates']}), flush=True)
        old.require(len(fits) == 30, 'all30 fresh attempts before this study DEV load')
        copy_parent(plan, 'fits.json', output / 'parent-fits.json')
        cached = json.loads((output / 'parent-fits.json').read_text())
        for seed in cfg['seeds']:
            for ri, rate in enumerate(cfg['learning_rates']):
                prior_key = f'lpv_instant-{seed}-lr{ri}'
                source_rows = [f for f in cached if f['key'] == prior_key]
                old.require(len(source_rows) == 1, 'unique cached legacy fit')
                original = source_rows[0]
                old.require(original['arm'] == 'lpv_instant' and original['seed'] == seed and original['learning_rate'] == rate
                            and original['fit']['status'] == 'PASS' and original['fit']['completed_updates'] == cfg['updates'], 'qualified cached legacy fit')
                key = f'legacy_instant-{seed}-lr{ri}'
                for name in LEGACY_FILES:
                    copy_parent(plan, prior_key + '/' + name, output / key / name)
                old.require(json.loads((output / key / 'fit-receipt.json').read_text()) == original['fit'], 'legacy fit receipt join')
                model = model_for('legacy_instant', seed, linear)
                state = load_parent_arrays(plan, prior_key + '/final.npz')
                model.load_state_dict({k: torch.from_numpy(v) for k,v in state.items()}, strict=True)
                model.zero_grad(set_to_none=True)
                models[key] = model
                fits.append({'key': key, 'arm': 'legacy_instant', 'seed': seed, 'learning_rate': rate,
                             'fit': original['fit'], 'resources': resource_model(model, 'legacy_instant'),
                             'origin': 'cached_parent', 'parent_key': prior_key,
                             'parent_files': {name: plan['parent_payloads'][prior_key + '/' + name] for name in LEGACY_FILES}})
                old.write_json(output / f'completed-fit-{len(fits):02d}.json', fits[-1])
        old.require(len(fits) == 36, 'complete fresh and cached fit roster')
        old.write_json(output / 'checkpoint-barrier.json', {'fresh_fit_attempts': 30, 'cached_fit_records': 6, 'fit_records': 36,
                       'dev_loads_this_run': 0, 'dev_exposed_prior': True,
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
                old.require(all(p.grad is None for p in models[f['key']].parameters()), 'no retained gradients in inference storage')
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
                       'fits': len(fits), 'fresh_fit_attempts': 30, 'cached_fit_records': 6, 'rows': len(rows), 'scientific_result': result['status'],
                       'raw_decodes': 0, 'reference_refits': 0, 'legacy_refits': 0, 'saved_fit_loads': 7, 'saved_dev_loads': 2, 'confirmation_access': False, 'official_test_access': False})
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
