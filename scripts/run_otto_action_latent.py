"""One fixed nine-fit action-gap pilot, then fresh held-out saved predictions."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import otto_action_latent_common as c

FAMILIES = ('action_recurrent', 'action_blind', 'direct_horizon')
THRESHOLDS = {'long_log_relative_gain': .01, 'long_gap_relative_gain': .05,
              'normal_log_relative_tolerance': .01, 'normal_gap_relative_tolerance': .01,
              'minimum_supported_cases': 16}


def load_data(path, np, horizon):
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    n = len(data['case_ids'])
    expected = {'prefix': (n, 9, 31), 'prefix_lengths': (n,), 'actions': (n, horizon),
                'continuation': (n, horizon, 31), 'outcomes': (n, horizon),
                'raw_costs': (n, horizon, 4), 'legal': (n, horizon, 4), 'case_ids': (n,), 'regimes': (n,)}
    c.require(set(data) == set(expected) and all(data[k].shape == v for k, v in expected.items()), 'complete dataset schema')
    c.require(len(set(data['case_ids'].tolist())) == n and np.all(data['prefix_lengths'] == 9), 'one block per surviving originating case')
    for key in ('prefix', 'continuation', 'raw_costs'):
        c.require(data[key].dtype == np.float32 and np.isfinite(data[key]).all(), 'finite float32 ' + key)
    for key in ('prefix_lengths', 'actions', 'outcomes'):
        c.require(data[key].dtype == np.int64, 'int64 ' + key)
    c.require(data['legal'].dtype == np.bool_ and np.all((data['outcomes'] >= 0) & (data['outcomes'] <= 4))
              and np.all((data['actions'] >= 0) & (data['actions'] < 4)), 'declared categorical data')
    found = data['outcomes'] == 4
    c.require(np.array_equal(found, np.maximum.accumulate(found, axis=1))
              and np.array_equal(data['legal'].any(-1), ~found), 'absorbing outcome and exact decision support')
    return data


def train_cost_scale(data, np, *, return_variance=False):
    normalized = data['raw_costs'].astype(np.float64) / 64.
    centered = normalized - normalized.mean(-1, keepdims=True)
    alive = data['outcomes'] != 4
    case_mse = (np.square(centered).mean(-1) * alive).sum(-1) / np.maximum(alive.sum(-1), 1)
    variance = float(case_mse.mean())
    scale = float(np.float32(np.sqrt(max(variance, 1e-6))))
    return (scale, variance) if return_variance else scale


def loss_for(prediction, targets, torch, cost_scale=1.):
    outcomes, costs, future = targets['outcomes'], targets['raw_costs'] / 64., targets['continuation']
    alive = outcomes != 4
    centered = costs - costs.mean(-1, keepdim=True)
    ce = torch.nn.functional.cross_entropy(prediction['outcome_logits'].flatten(0, 1), outcomes.flatten())
    q = ((prediction['cost_contrasts'] - centered).square().mean(-1) * alive).sum(-1) / alive.sum(-1).clamp_min(1)
    aux = ((prediction['aux_features'] - future[..., 19:21]).square().mean(-1) * alive).sum(-1) / alive.sum(-1).clamp_min(1)
    return ce + c.CONFIG['cost_weight'] * q.mean() / cost_scale**2 + c.CONFIG['aux_weight'] * aux.mean()


def save_npz(path, np, **values):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **values)


class Pilot(c.Run):
    def body(self):
        collection = c.closed(c.OUT / 'collection-01', c.OUT / 'collection-native-01.terminal.json')
        c.require(collection['plan_sha256'] == self.args.plan_sha256, 'collection shares registration')
        import numpy as np
        import torch
        sys.path.insert(0, str(c.ROOT / 'src'))
        from openjev.research import otto_action_latent_metrics as metrics
        from openjev.research import otto_action_latent_model as models
        from openjev.research import otto_action_latent_ridge as ridge
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        train = load_data(c.OUT / 'collection-01/train.npz', np, 4)
        n = len(train['case_ids'])
        c.require(n >= c.CONFIG['min_train'] and set(train['regimes']) == {'lambda3'}, 'only registered TRAIN')
        tensors = {k: torch.from_numpy(v.copy()) for k, v in train.items() if k not in ('case_ids', 'regimes')}
        cost_scale, cost_variance = train_cost_scale(train, np, return_variance=True)
        c.write(self.out / 'normalization.json', {'cost_scale': cost_scale, 'rms_squared_before_floor': cost_variance,
                'variance_floor': 1e-6, 'source': 'TRAIN only, all-four centered, equal cases and surviving rows',
                'teacher_units_divisor': 64., 'dev_decodes': 0})
        fits, checkpoints, calls = [], {}, {'optimizer_steps': 0, 'fit_count': 0, 'dev_array_decodes': 0}
        for seed_index, seed in enumerate(c.CONFIG['fit_seeds']):
            # Rotate execution order; matched arm permutations reset the same RNG.
            order = FAMILIES[seed_index:] + FAMILIES[:seed_index]
            for family in order:
                self.check()
                tick = time.perf_counter()
                model = models.make_model(family, seed, cost_scale=cost_scale)
                initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                optimizer = torch.optim.Adam(model.parameters(), lr=c.CONFIG['lr'])
                rng = np.random.Generator(np.random.PCG64(seed))
                updates = 0
                for epoch in range(c.CONFIG['epochs']):
                    indices = rng.permutation(n)
                    c.append(self.out / 'training-orders.jsonl', {'family': family, 'seed': seed, 'epoch': epoch, 'indices': indices.tolist()})
                    for start in range(0, n, c.CONFIG['batch']):
                        self.check()
                        ix = indices[start:start + c.CONFIG['batch']]
                        batch = {k: v[ix] for k, v in tensors.items()}
                        optimizer.zero_grad(set_to_none=True)
                        blind = model.blind_rollout(batch['prefix'], batch['prefix_lengths'], batch['actions'])
                        normal = model.normal_rollout(batch['prefix'], batch['prefix_lengths'], batch['actions'],
                                                      batch['continuation'], found=batch['outcomes'] == 4)
                        loss = .5 * (loss_for(blind, batch, torch, cost_scale) + loss_for(normal, batch, torch, cost_scale))
                        c.require(bool(torch.isfinite(loss)), 'finite fixed objective')
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.CONFIG['clip'], error_if_nonfinite=True)
                        optimizer.step()
                        updates += 1
                        calls['optimizer_steps'] += 1
                    if (epoch + 1) % 20 == 0:
                        print(f'{family} seed={seed} epoch={epoch + 1}/80', flush=True)
                path = self.out / f'{family}-{seed}.npz'
                state = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
                c.require(all(np.isfinite(v).all() for v in state.values()), 'finite final checkpoint')
                save_npz(path, np, **state)
                row = {'family': family, 'seed': seed, 'epochs': c.CONFIG['epochs'], 'updates': updates,
                       'cases': n, 'exposures': n * c.CONFIG['epochs'], 'seconds': time.perf_counter() - tick,
                       'last_batch_loss': float(loss.detach()), 'last_gradient_norm': float(norm),
                       'parameters': model.parameter_metadata(), 'checkpoint': c.desc(path),
                       'changed_tensors': [k for k, v in model.state_dict().items() if not torch.equal(v, initial_state[k])],
                       'training_horizons': [1, 2, 3, 4], 'evaluation_decodes_so_far': 0}
                fits.append(row)
                checkpoints[family, seed] = state
                calls['fit_count'] += 1
                c.append(self.out / 'fits.jsonl', row)
        # A separately fitted deterministic linear control, using the same cases/views.
        self.check()
        tick = time.perf_counter()
        dx = [ridge.design(train['prefix'], train['prefix_lengths'], train['actions']),
              ridge.design(train['prefix'], train['prefix_lengths'], train['actions'], train['continuation'], train['outcomes'] == 4)]
        x = np.concatenate(dx).reshape(-1, 336)
        target = np.tile(train['outcomes'], (2, 1)).reshape(-1)
        probability_beta = ridge.solve(x, np.eye(5)[target], np.full(len(x), 1 / len(x)), c.CONFIG['ridge_lambda'])
        raw = train['raw_costs'].astype(np.float64) / 64
        centered = raw - raw.mean(-1, keepdims=True)
        alive = train['outcomes'] != 4
        weights = alive / np.maximum(alive.sum(-1, keepdims=True), 1) / (2 * n)
        cost_beta = ridge.solve(x, np.tile(centered, (2, 1, 1)).reshape(-1, 4), np.tile(weights, (2, 1)).reshape(-1), c.CONFIG['ridge_lambda'])
        save_npz(self.out / 'ridge.npz', np, probability_beta=probability_beta, cost_beta=cost_beta)
        ridge_fit = {'fits': 1, 'parameters': int(probability_beta.size + cost_beta.size),
                     'seconds': time.perf_counter() - tick, 'training_horizons': [1, 2, 3, 4],
                     'evaluation_decodes_so_far': 0, 'normal_equation_rows': len(x)}
        c.write(self.out / 'ridge-fit.json', ridge_fit)
        # Only final fits exist before first DEV decoding. No early stopping or selection.
        c.require(len(fits) == 9 and all(r['epochs'] == 80 for r in fits), 'all nine prescribed fits before DEV')
        self.check()
        dev = load_data(c.OUT / 'collection-01/dev.npz', np, 8)
        calls['dev_array_decodes'] += 1
        c.require(not set(dev['case_ids']) & set(train['case_ids']), 'originating case split')
        predictions, reports, timings = {}, [], []
        for family in (*FAMILIES, 'ridge'):
            for seed in c.CONFIG['fit_seeds']:
                model = None
                if family != 'ridge':
                    model = models.make_model(family, seed)
                    model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in checkpoints[family, seed].items()})
                    model.eval()
                for condition in ('gap', 'normal'):
                    self.check()
                    tick = time.perf_counter()
                    outputs, cost_outputs, work = [], [], {}
                    for i in range(len(dev['case_ids'])):
                        ix = slice(i, i + 1)
                        if family == 'ridge':
                            d = ridge.design(dev['prefix'][ix], dev['prefix_lengths'][ix], dev['actions'][ix],
                                dev['continuation'][ix] if condition == 'normal' else None,
                                dev['outcomes'][ix] == 4 if condition == 'normal' else None)
                            outcome = ridge.project_probabilities(d @ probability_beta, c.CONFIG['ridge_probability_smoothing'])
                            q = d @ cost_beta
                        else:
                            batch = {k: torch.from_numpy(dev[k][ix].copy()) for k in ('prefix', 'prefix_lengths', 'actions', 'continuation', 'outcomes')}
                            with torch.no_grad():
                                if condition == 'gap':
                                    result = model.blind_rollout(batch['prefix'], batch['prefix_lengths'], batch['actions'])
                                else:
                                    result = model.normal_rollout(batch['prefix'], batch['prefix_lengths'], batch['actions'], batch['continuation'], found=batch['outcomes'] == 4)
                            outcome, q = result['outcome_logits'].numpy(), result['cost_contrasts'].numpy()
                            for k, v in result['work'].items():
                                work[k] = work.get(k, 0) + v
                        outputs.append(outcome)
                        cost_outputs.append(q)
                    outcome, q = np.concatenate(outputs), np.concatenate(cost_outputs)
                    timings.append({'family': family, 'seed': seed, 'condition': condition,
                                    'seconds': time.perf_counter() - tick, 'cases': len(outcome), 'work': work})
                    key = f'{family}__{seed}__{condition}'
                    predictions[key + '__outcome'], predictions[key + '__cost'] = outcome, q
                    reports.append(metrics.score(dev['outcomes'], outcome, q, dev['raw_costs'], dev['legal'],
                        case_ids=dev['case_ids'].tolist(), regimes=dev['regimes'].tolist(), family=family, fit_seed=seed,
                        condition=condition, prediction_kind='probabilities' if family == 'ridge' else 'logits'))
        gate = metrics.evaluate_reports(reports, candidate=c.CONFIG['candidate'], controls=c.CONFIG['controls'],
            fit_seeds=c.CONFIG['fit_seeds'], regimes=['lambda3', 'lambda4'], thresholds=THRESHOLDS)
        save_npz(self.out / 'predictions.npz', np, **predictions)
        c.write(self.out / 'reports.json', {'reports': reports, 'gate': gate})
        c.write(self.out / 'summary.json', {'status': 'DEV_PASS' if gate['passed'] else 'DEV_FAIL',
                'fits': fits, 'ridge_fit': ridge_fit, 'prediction_times': timings, 'calls': calls,
                'collection_receipt': c.desc(c.OUT / 'collection-01/receipt.json'),
                'train_data': c.desc(c.OUT / 'collection-01/train.npz'), 'dev_data': c.desc(c.OUT / 'collection-01/dev.npz'),
                'data_cases': {'train': n, 'dev': len(dev['case_ids'])},
                'scope': 'fixed-path forecast imitation, not autonomous control or novel architecture proof'})
        self.receipt['calls'] = calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'supervision', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    c.require(Path(sys.executable).absolute() == c.NUMERICAL, 'declared numerical interpreter')
    run = Pilot(args, 'fit')
    try:
        run.body()
        run.finish()
    except BaseException as error:
        run.finish(error)
        raise


if __name__ == '__main__':
    main()
