"""Registered learned-deletion pilot, with bounded analytic GP controls."""
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
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.special import ndtr

from openjev.research import retention_controls as controls
from openjev.research import retention_data as data_api
from openjev.research import retention_memory as memory
from openjev.research.retention_policy import RetentionPolicy, leave_one_out_advantage, select

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 23, 37)
CONTROLS = ('kl8', 'kl9', 'fifo9', 'fic9', 'coverage41', 'recent41')
METHODS = tuple(f'learned-{seed}' for seed in SEEDS)+CONTROLS+('full', 'diagonal')
METRICS = ('regret', 'nll', 'brier', 'mse', 'coverage90', 'defer', 'always_defer_regret', 'risk_mae')
CONFIG = {
    'version': 'retention-v1', 'namespace': 443260924, 'fit_seeds': list(SEEDS),
    'train_contexts': 256, 'train_observations': 64, 'queries': 4,
    'epochs': 16, 'batch_contexts': 8, 'group_size': 4,
    'learning_rate': .01, 'entropy_coefficient': .002, 'gradient_clip': 1.,
    'cohorts': 3, 'evaluation_contexts': 128, 'state_budget_bytes': 1024,
    'populations': {'base': {'observations': 64, 'geometry': 'axial', 'offset': 100},
                    'shift': {'observations': 64, 'geometry': 'diagonal', 'offset': 200},
                    'long': {'observations': 192, 'geometry': 'axial', 'offset': 300}},
    'methods': list(METHODS), 'warmups': 3, 'timing_repeats': 10, 'wall_cap_seconds': 1800,
}
SOURCES = (
    'src/openjev/research/retention_memory.py', 'src/openjev/research/retention_data.py',
    'src/openjev/research/retention_controls.py', 'src/openjev/research/retention_policy.py',
    'scripts/retention_study.py', 'scripts/audit_retention_study.py',
    'tests/test_retention_memory.py', 'tests/test_retention_data.py',
    'tests/test_retention_controls.py', 'tests/test_retention_policy.py',
    'tests/test_retention_study.py', 'tests/test_audit_retention_study.py',
    'research/retention-protocol.md',
)


def descriptor(path):
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def register(out):
    out.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name in SOURCES:
        target = out/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
        sources[name] = descriptor(target)
    write(out/'registration.json', {'config': CONFIG, 'sources': sources,
        'parent_result': descriptor(ROOT/'research/residual-memory-results/summary.json'),
        'environment': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                        'torch': torch.__version__, 'platform': platform.platform(), 'threads': 1}})
    print(json.dumps({'registered': str(out), **descriptor(out/'registration.json')}), flush=True)


def verify(out):
    record = json.loads((out/'registration.json').read_text())
    if record['config'] != CONFIG or set(record['sources']) != set(SOURCES):
        raise ValueError('frozen configuration and source roster')
    for name, pin in record['sources'].items():
        if descriptor(ROOT/name) != pin or descriptor(out/'source'/name) != pin:
            raise ValueError('frozen source changed: '+name)
    if descriptor(ROOT/'research/residual-memory-results/summary.json') != record['parent_result']:
        raise ValueError('previous result changed')


def path_prediction(state, paths):
    batch, queries = paths.shape[:2]
    joint = memory.predict(state, paths.reshape(batch, -1, 2))
    means = joint['mean'].reshape(batch, queries, 4, 4).mean(-1)
    covariance = joint['cov'].reshape(batch, queries*4, 4, queries*4, 4)
    variance = np.stack([covariance[:, j, :, j, :].sum((1, 2))/16 for j in range(queries*4)], -1)
    variance = variance.reshape(batch, queries, 4)
    if not (np.isfinite(variance).all() and (variance > 0).all()):
        raise ValueError('positive latent path-average variance')
    return {'mean': means, 'variance': variance, 'risk': ndtr((means-.5)/np.sqrt(variance))}


def costs(risk):
    return np.concatenate((.02+risk, np.full((*risk.shape[:-1], 1), .20)), -1)


def regret(prediction, reference):
    selected = costs(prediction['risk']).argmin(-1)
    true_cost = costs(reference['risk'])
    return np.take_along_axis(true_cost, selected[..., None], -1)[..., 0]-true_cost.min(-1)


def metrics(prediction, reference, exposure):
    variance = prediction['variance']
    squared = (exposure-prediction['mean'])**2
    true_cost = costs(reference['risk'])
    return {
        'regret': float(regret(prediction, reference).mean()),
        'nll': float((.5*(math.log(2*math.pi)+np.log(variance)+squared/variance)).mean()),
        'brier': float(((prediction['risk']-(exposure > .5))**2).mean()),
        'mse': float(squared.mean()),
        'coverage90': float((squared <= 1.6448536269514722**2*variance).mean()),
        'defer': float((costs(prediction['risk']).argmin(-1) == 4).mean()),
        'always_defer_regret': float((.2-true_cost.min(-1)).mean()),
        'risk_mae': float(np.abs(prediction['risk']-reference['risk']).mean())}


def retain(x, y, method, policy=None, stochastic=False):
    """Writes have no request coordinates, outcome labels or teacher predictions."""
    batch, steps = y.shape
    traces = np.full((batch, steps), -1, np.int16)
    log_probabilities, entropies = [], []
    floor_count = 0
    if method in ('coverage41', 'recent41'):
        state = controls.initial_raw(batch)
        for step in range(steps):
            state = controls.write_raw(state, x[:, step], y[:, step],
                                       mode='coverage' if method == 'coverage41' else 'recent')
    elif method == 'fic9':
        state = controls.initial_fic(batch)
        for step in range(steps):
            state = controls.write_fic(state, x[:, step], y[:, step])
    else:
        capacity = 8 if method in ('learned', 'kl8') else 9
        state = memory.initial(batch, capacity)
        for step in range(steps):
            state = memory.expand(state, x[:, step], y[:, step])
            if state.Z.shape[1] <= capacity:
                continue
            if method == 'fifo9':
                drop = np.zeros(batch, np.int64)
            else:
                statistics = memory.deletion_statistics(state)
                floor_count += statistics['negative_kl_floor_count']
                if method == 'learned':
                    if policy is None:
                        raise ValueError('learned retention needs a policy')
                    drop, logp, entropy = select(policy, statistics, stochastic)
                    if stochastic:
                        log_probabilities.append(logp)
                        entropies.append(entropy)
                else:
                    drop = statistics['kl'].argmin(-1)
            traces[:, step] = drop
            state = memory.compress(state, np.asarray(drop, np.int64))
    return state, traces, log_probabilities, entropies, floor_count


def predict(state, paths, method):
    if method in ('coverage41', 'recent41'):
        return controls.predict_raw(state, paths)
    if method == 'fic9':
        return controls.predict_fic(state, paths)
    return path_prediction(state, paths)


def evaluate(data, method, policy=None):
    if policy is not None and any(p.grad is not None for p in policy.parameters()):
        raise ValueError('inference policy retains gradient tensors')
    if method in ('full', 'diagonal'):
        reference = data_api.exact_reference(data['x'], data['y'], data['paths'])
        prediction = {name: reference[name] for name in ('mean', 'variance', 'risk')}
        if method == 'diagonal':
            prediction.update(variance=reference['diag_variance'], risk=reference['diag_risk'])
        return prediction, None, None, 0
    with torch.no_grad():
        state, traces, _, _, floors = retain(data['x'], data['y'], method, policy)
        prediction = predict(state, data['paths'], method)
    return prediction, state, traces, floors


def _updates(policy, optimizer, data, reference, seed, rows, action_records, reward_records, field_records):
    generator = np.random.default_rng(seed)
    for epoch in range(CONFIG['epochs']):
        order = generator.permutation(CONFIG['train_contexts'])
        for offset in range(0, len(order), CONFIG['batch_contexts']):
            ids = order[offset:offset+CONFIG['batch_contexts']]
            repeated = np.repeat(ids, CONFIG['group_size'])
            state, actions, logps, entropies, floors = retain(data['x'][repeated], data['y'][repeated],
                                                             'learned', policy, stochastic=True)
            prediction = path_prediction(state, data['paths'][repeated])
            paired_reference = {'risk': reference['risk'][repeated]}
            rewards = -regret(prediction, paired_reference).mean(-1).reshape(len(ids), CONFIG['group_size'])
            advantage = leave_one_out_advantage(torch.from_numpy(rewards)).reshape(-1)
            logp = torch.stack(logps).mean(0)
            entropy = torch.stack(entropies).mean()
            loss = -(advantage*logp).mean()-CONFIG['entropy_coefficient']*entropy
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), CONFIG['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in policy.parameters()):
                raise ValueError('finite learned policy parameters')
            rows.append({'epoch': epoch, 'update': len(rows)+1, 'loss': float(loss.detach()),
                         'mean_reward': float(rewards.mean()), 'group_reward_std': float(rewards.std(1).mean()),
                         'entropy': float(entropy.detach()), 'gradient_norm': float(grad_norm),
                         'kl_roundoff_floors': floors})
            action_records.append(actions); reward_records.append(rewards); field_records.append(ids)
        print(json.dumps({'fit_seed': seed, 'epoch': epoch+1, 'updates': len(rows)}), flush=True)


def train(out, data, reference, seed):
    torch.manual_seed(seed)
    policy = RetentionPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=CONFIG['learning_rate'])
    np.savez_compressed(out/f'policy-{seed}-initial.npz', **policy.arrays())
    rows, action_records, reward_records, field_records = [], [], [], []
    start = time.perf_counter()
    try:
        _updates(policy, optimizer, data, reference, seed, rows, action_records, reward_records, field_records)
    except BaseException as error:
        np.savez_compressed(out/f'policy-{seed}-partial.npz', **policy.arrays())
        torch.save(optimizer.state_dict(), out/f'optimizer-{seed}-partial.pt')
        if action_records:
            np.savez_compressed(out/f'training-{seed}-partial.npz', actions=np.stack(action_records),
                                rewards=np.stack(reward_records), fields=np.stack(field_records))
        write(out/f'training-{seed}-partial.json', {
            'state': 'FAILED', 'completed_update_records': len(rows), 'trace': rows,
            'error_type': type(error).__name__, 'error': str(error),
            'seconds': time.perf_counter()-start,
            'scope': 'Current policy/Adam may include an interrupted update; never treated as final.'})
        raise
    np.savez_compressed(out/f'policy-{seed}-final.npz', **policy.arrays())
    torch.save(optimizer.state_dict(), out/f'optimizer-{seed}-final.pt')
    np.savez_compressed(out/f'training-{seed}.npz', actions=np.stack(action_records),
                        rewards=np.stack(reward_records), fields=np.stack(field_records))
    write(out/f'training-{seed}.json', {'fit_seed': seed, 'updates': len(rows), 'trace': rows,
          'seconds': time.perf_counter()-start, 'parameter_bytes': policy.parameter_bytes,
          'group_trajectories': len(rows)*CONFIG['batch_contexts']*CONFIG['group_size']})
    optimizer.zero_grad(set_to_none=True)
    policy.requires_grad_(False)
    policy.eval()
    if any(p.grad is not None for p in policy.parameters()):
        raise ValueError('training gradients retained by inference policy')
    return policy


def saved_state(state, traces):
    result = {name: getattr(state, name) for name in ('Z', 'mean', 'cov', 'y') if hasattr(state, name)}
    result.update(step=np.array(state.step, np.int64), traces=traces,
                  resident_bytes=np.array(state.resident_bytes_per_context, np.int64))
    return result


def resources(data, method, policy):
    one = {name: value[:1].copy() for name, value in data.items()}
    values = []
    state = None
    for repetition in range(CONFIG['warmups']+CONFIG['timing_repeats']):
        start = time.perf_counter()
        _, state, _, _ = evaluate(one, method, policy)
        elapsed = time.perf_counter()-start
        if repetition >= CONFIG['warmups']:
            values.append(elapsed)
    # Python/NumPy allocation tracing is a limited workspace measurement.
    # It omits native Torch/BLAS allocation and is never labeled peak RSS.
    tracemalloc.start()
    tracemalloc.reset_peak()
    evaluate(one, method, policy)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    retained = (one['x'].nbytes+one['y'].nbytes+40 if state is None else state.resident_bytes_per_context)
    if policy is not None:
        retained += policy.parameter_bytes
    return {'timings_seconds': values, 'median_seconds': float(np.median(values)),
            'resident_bytes': retained, 'python_numpy_traced_peak_bytes': peak,
            'traced_current_bytes': current, 'native_workspace_measured': False,
            'retained_gradient_tensors': 0,
            'scope': 'One context, complete stream then four requests; native allocations and peak RSS unmeasured.'}


def summarize(rows):
    means, checks = {}, []
    learned = [f'learned-{seed}' for seed in SEEDS]
    for phase in CONFIG['populations']:
        means[phase] = {method: {metric: float(np.mean([r[metric] for r in rows
            if r['phase'] == phase and r['method'] == method])) for metric in METRICS} for method in METHODS}
        candidate = {metric: float(np.mean([means[phase][m][metric] for m in learned])) for metric in METRICS}
        means[phase]['learned_mean'] = candidate
        factor = 1.05 if phase == 'shift' else .90
        for control in CONTROLS:
            checks.append({'name': f'{phase}_regret_vs_{control}',
                'passed': candidate['regret'] <= factor*means[phase][control]['regret']+1e-6})
        checks.append({'name': f'{phase}_nll', 'passed': candidate['nll'] <= min(
            means[phase][control]['nll'] for control in CONTROLS)+.02})
        checks.append({'name': f'{phase}_beats_defer', 'passed': candidate['regret'] < candidate['always_defer_regret']})
        for cohort in range(3):
            group = {r['method']: r for r in rows if r['phase'] == phase and r['cohort'] == cohort}
            bound = min(group[control]['regret'] for control in CONTROLS)
            value = np.mean([group[method]['regret'] for method in learned])
            checks.append({'name': f'{phase}_cohort_{cohort}', 'passed': bool(value <= (1.05 if phase == 'shift' else 1.)*bound+1e-6)})
        bound = min(means[phase][control]['regret'] for control in CONTROLS)
        for method in learned:
            checks.append({'name': f'{phase}_{method}', 'passed': means[phase][method]['regret'] <=
                (1.05 if phase == 'shift' else 1.)*bound+1e-6})
    return {'means': means, 'checks': checks,
            'gate': 'ADVANCE_RETENTION' if all(r['passed'] for r in checks) else 'DO_NOT_ADVANCE_RETENTION',
            'scope': 'Learned deletion pilot on supplied known-law GP fields; no general architecture claim.'}


def run(out):
    verify(out)
    write(out/'started.json', {'unix_time': time.time()})
    start = time.perf_counter()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('frozen wall cap')))
    signal.alarm(CONFIG['wall_cap_seconds'])
    receipt = {'state': 'RUNNING'}
    try:
        training = data_api.generate(CONFIG['namespace'], CONFIG['train_contexts'], CONFIG['train_observations'])
        reference = data_api.exact_reference(training['x'], training['y'], training['paths'])
        np.savez_compressed(out/'train-data.npz', **training)
        np.savez_compressed(out/'train-reference.npz', **reference)
        policies = {seed: train(out, training, reference, seed) for seed in SEEDS}
        rows, resource_rows, diagnostics = [], [], []
        for phase, population in CONFIG['populations'].items():
            for cohort in range(CONFIG['cohorts']):
                data = data_api.generate(CONFIG['namespace']+population['offset']+cohort,
                    CONFIG['evaluation_contexts'], population['observations'], population['geometry'])
                stem = f'{phase}-{cohort}'
                np.savez_compressed(out/f'{stem}-data.npz', **data)
                reference = data_api.exact_reference(data['x'], data['y'], data['paths'])
                for name in METHODS:
                    method = 'learned' if name.startswith('learned-') else name
                    policy = policies[int(name.split('-')[1])] if method == 'learned' else None
                    prediction, state, traces, floors = evaluate(data, method, policy)
                    np.savez_compressed(out/f'{stem}-{name}-prediction.npz', **prediction)
                    if state is not None:
                        np.savez_compressed(out/f'{stem}-{name}-state.npz', **saved_state(state, traces))
                        retained = state.resident_bytes_per_context+(policy.parameter_bytes if policy else 0)
                        if retained > CONFIG['state_budget_bytes']:
                            raise ValueError('retained model exceeds registered storage budget')
                    rows.append({'phase': phase, 'cohort': cohort, 'method': name,
                                 **metrics(prediction, reference, data['exposure'])})
                    diagnostics.append({'phase': phase, 'cohort': cohort, 'method': name,
                                        'kl_roundoff_floors': floors})
                    if cohort == 0:
                        resource_rows.append({'phase': phase, 'method': name, **resources(data, method, policy)})
                print(json.dumps({'evaluated': stem, 'rows': len(rows)}), flush=True)
        result = summarize(rows)
        write(out/'metrics.json', rows); write(out/'resources.json', resource_rows)
        write(out/'diagnostics.json', diagnostics); write(out/'summary.json', result)
        verify(out)
        receipt.update(state='EXITED', exit_code=0, gate=result['gate'], training_updates=1536)
    except BaseException as exc:
        receipt.update(state='FAILED', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0)
        receipt['elapsed_seconds'] = time.perf_counter()-start
        write(out/'run-receipt.json', receipt)
        write(out/'manifest.json', {str(p.relative_to(out)): descriptor(p) for p in sorted(out.rglob('*'))
                                  if p.is_file() and p.name != 'manifest.json'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('register', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'run' and args.registration_sha256 != descriptor(args.out/'registration.json')['sha256']:
        raise ValueError('registered SHA256 required')
    (register if args.mode == 'register' else run)(args.out.resolve())


if __name__ == '__main__':
    main()
