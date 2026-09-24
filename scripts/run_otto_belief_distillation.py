"""Twelve fixed fits with sampled or full-belief targets, then fresh DEV forecasts."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import otto_belief_distillation_common as c

FAMILIES = ('recurrent_soft', 'recurrent_sampled', 'action_blind_soft', 'direct_soft')
MODEL_KINDS = dict(zip(FAMILIES, ('action_recurrent', 'action_recurrent', 'action_blind', 'direct_horizon'), strict=True))
THRESHOLDS = {'long_log_relative_gain': .01, 'long_gap_relative_gain': .05,
              'normal_log_relative_tolerance': .01, 'normal_gap_relative_tolerance': .01,
              'minimum_supported_cases': 48}
# Explicit projection: privileged beliefs and costs never enter either rollout.
INPUT_KEYS = ('prefix', 'prefix_lengths', 'actions')
TARGET_KEYS = ('continuation', 'outcomes', 'raw_costs', 'gap_oracle', 'normal_oracle')


def load_data(path, np, horizon):
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    n = len(data['case_ids'])
    expected = {'prefix': (n, 9, 31), 'prefix_lengths': (n,), 'actions': (n, horizon),
                'continuation': (n, horizon, 31), 'outcomes': (n, horizon),
                'raw_costs': (n, horizon, 4), 'legal': (n, horizon, 4), 'case_ids': (n,), 'regimes': (n,),
                'initial_belief': (n, 2809), 'prefix_actions': (n, 8), 'prefix_outcomes': (n, 8),
                'prefix_position': (n, 2), 'gap_oracle': (n, horizon, 5),
                'normal_oracle': (n, horizon, 5), 'opposite_oracle': (n, horizon, 5)}
    c.require(set(data) == set(expected) and all(data[k].shape == v for k, v in expected.items()), 'complete dataset schema')
    c.require(n > 0 and len(set(data['case_ids'].tolist())) == n and np.all(data['prefix_lengths'] == 9), 'unique surviving cases')
    c.require(set(data['regimes']) <= {'lambda3', 'lambda4'}, 'declared regimes')
    for key in ('prefix', 'continuation', 'raw_costs'):
        c.require(data[key].dtype == np.float32 and np.isfinite(data[key]).all(), 'finite float32 ' + key)
    for key in ('prefix_lengths', 'actions', 'outcomes', 'prefix_actions', 'prefix_outcomes', 'prefix_position'):
        c.require(data[key].dtype == np.int64, 'int64 ' + key)
    c.require(data['legal'].dtype == np.bool_ and np.all((data['outcomes'] >= 0) & (data['outcomes'] <= 4))
              and all(np.all((data[k] >= 0) & (data[k] < 4)) for k in ('actions', 'prefix_actions', 'prefix_outcomes')),
              'declared categorical data')
    c.require(np.all((data['prefix_position'] >= 8) & (data['prefix_position'] <= 44)), 'inbounds entire block')
    found = data['outcomes'] == 4
    c.require(np.array_equal(found, np.maximum.accumulate(found, axis=1))
              and np.array_equal(data['legal'].any(-1), ~found), 'absorbing outcome and exact decision support')
    for key in ('gap_oracle', 'normal_oracle', 'opposite_oracle', 'initial_belief'):
        value = data[key]
        c.require(value.dtype == np.float64 and np.isfinite(value).all() and np.all(value >= 0)
                  and np.all(value <= 1) and np.allclose(value.sum(-1), 1., rtol=0, atol=1e-12), 'valid probability ' + key)
    for key in ('gap_oracle', 'normal_oracle'):
        c.require(np.all(np.take_along_axis(data[key], data['outcomes'][..., None], -1) > 0), 'positive actual outcome probability')
    resolved = np.zeros_like(found)
    resolved[:, 1:] = found[:, :-1]
    terminal = np.zeros((int(resolved.sum()), 5), np.float64)
    terminal[:, 4] = 1
    c.require(np.array_equal(data['normal_oracle'][resolved], terminal), 'normal resolved targets exact')
    c.require(np.allclose(data['gap_oracle'][:, 0], data['normal_oracle'][:, 0], rtol=0, atol=1e-12), 'shared first forecast')
    return data


def train_cost_scale(data, np, *, return_variance=False):
    normalized = data['raw_costs'].astype(np.float64) / 64.
    centered = normalized - normalized.mean(-1, keepdims=True)
    alive = data['outcomes'] != 4
    case_mse = (np.square(centered).mean(-1) * alive).sum(-1) / np.maximum(alive.sum(-1), 1)
    variance = float(case_mse.mean())
    scale = float(np.float32(np.sqrt(max(variance, 1e-6))))
    return (scale, variance) if return_variance else scale


def loss_for(prediction, targets, torch, cost_scale=1., *, condition, soft_targets):
    c.require(condition in ('gap', 'normal') and type(soft_targets) is bool, 'explicit fixed target condition')
    outcomes, costs, future = targets['outcomes'], targets['raw_costs'] / 64., targets['continuation']
    alive = outcomes != 4
    centered = costs - costs.mean(-1, keepdim=True)
    logp = torch.log_softmax(prediction['outcome_logits'], dim=-1)
    if soft_targets:
        per_row = -(targets[condition + '_oracle'] * logp).sum(-1)
    else:
        per_row = -logp.gather(-1, outcomes[..., None]).squeeze(-1)
    if condition == 'normal':
        resolved = torch.zeros_like(alive)
        resolved[:, 1:] = (outcomes[:, :-1] == 4).cummax(dim=1).values
        per_row = per_row.masked_fill(resolved, 0.)
    ce = per_row.mean()  # Same all-case/all-horizon denominator for every arm.
    q = ((prediction['cost_contrasts'] - centered).square().mean(-1) * alive).sum(-1) / alive.sum(-1).clamp_min(1)
    aux = ((prediction['aux_features'] - future[..., 19:21]).square().mean(-1) * alive).sum(-1) / alive.sum(-1).clamp_min(1)
    return ce + c.CONFIG['cost_weight'] * q.mean() / cost_scale**2 + c.CONFIG['aux_weight'] * aux.mean()


def probabilities(logits, torch, *, condition, outcomes=None):
    """Only previously observed found is eligible for the analytical shortcut."""
    c.require(condition in ('gap', 'normal'), 'prediction condition')
    result = torch.softmax(logits.to(torch.float64), dim=-1)
    c.require(bool(torch.isfinite(result).all()) and bool((result > 0).all()), 'finite positive unresolved probabilities')
    if condition == 'normal':
        c.require(outcomes is not None, 'normal observations supplied')
        resolved = torch.zeros_like(outcomes, dtype=torch.bool)
        resolved[:, 1:] = (outcomes[:, :-1] == 4).cummax(dim=1).values
        result[resolved] = torch.tensor([0., 0., 0., 0., 1.], dtype=torch.float64)
    else:
        c.require(outcomes is None, 'blind predictions do not accept realized outcomes')
    return result


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
        from openjev.research import otto_action_latent_model as models
        from openjev.research import otto_belief_distillation_metrics as metrics
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        train = load_data(c.OUT / 'collection-01/train.npz', np, 4)
        n = len(train['case_ids'])
        c.require(n >= c.CONFIG['min_train'] and set(train['regimes']) == {'lambda3'}, 'only registered TRAIN')
        tensors = {k: torch.from_numpy(train[k].copy()) for k in (*INPUT_KEYS, *TARGET_KEYS)}
        cost_scale, cost_variance = train_cost_scale(train, np, return_variance=True)
        c.write(self.out / 'normalization.json', {'cost_scale': cost_scale, 'rms_squared_before_floor': cost_variance,
                'variance_floor': 1e-6, 'source': 'TRAIN only, all-four centered, equal cases and surviving rows',
                'teacher_units_divisor': 64., 'dev_decodes': 0})
        fits, checkpoints, calls = [], {}, {'optimizer_steps': 0, 'fit_count': 0, 'dev_array_decodes': 0}
        self.receipt['calls'] = calls
        for seed_index, seed in enumerate(c.CONFIG['fit_seeds']):
            order = FAMILIES[seed_index:] + FAMILIES[:seed_index]
            for family in order:
                self.check()
                tick = time.perf_counter()
                model = models.make_model(MODEL_KINDS[family], seed, cost_scale=cost_scale)
                initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                optimizer = torch.optim.Adam(model.parameters(), lr=c.CONFIG['lr'])
                rng = np.random.Generator(np.random.PCG64(seed))
                updates = 0
                for epoch in range(c.CONFIG['epochs']):
                    indices = rng.permutation(n)
                    c.append(self.out / 'training-orders.jsonl', {'family': family, 'seed': seed, 'epoch': epoch, 'indices': indices.tolist()})
                    for start in range(0, n, c.CONFIG['batch']):
                        self.check()
                        self.receipt['pending_fit'] = {'family': family, 'seed': seed, 'epoch': epoch, 'batch_start': start}
                        ix = indices[start:start + c.CONFIG['batch']]
                        batch = {k: v[ix] for k, v in tensors.items()}
                        optimizer.zero_grad(set_to_none=True)
                        blind = model.blind_rollout(batch['prefix'], batch['prefix_lengths'], batch['actions'])
                        normal = model.normal_rollout(batch['prefix'], batch['prefix_lengths'], batch['actions'],
                                                      batch['continuation'], found=batch['outcomes'] == 4)
                        soft = family != 'recurrent_sampled'
                        loss = .5 * (loss_for(blind, batch, torch, cost_scale, condition='gap', soft_targets=soft)
                                     + loss_for(normal, batch, torch, cost_scale, condition='normal', soft_targets=soft))
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
                row = {'family': family, 'model_kind': MODEL_KINDS[family], 'seed': seed, 'epochs': c.CONFIG['epochs'], 'updates': updates,
                       'cases': n, 'exposures': n * c.CONFIG['epochs'], 'seconds': time.perf_counter() - tick,
                       'soft_targets': soft, 'last_batch_loss': float(loss.detach()), 'last_gradient_norm': float(norm),
                       'parameters': model.parameter_metadata(), 'checkpoint': c.desc(path),
                       'changed_tensors': [k for k, v in model.state_dict().items() if not torch.equal(v, initial_state[k])],
                       'training_horizons': [1, 2, 3, 4], 'evaluation_decodes_so_far': 0,
                       'blind_input_keys': list(INPUT_KEYS), 'privileged_targets_only': ['raw_costs', 'gap_oracle', 'normal_oracle']}
                fits.append(row)
                checkpoints[family, seed] = state
                calls['fit_count'] += 1
                c.append(self.out / 'fits.jsonl', row)
                self.receipt['pending_fit'] = None
        c.require(len(fits) == 12 and all(r['epochs'] == 80 for r in fits), 'all twelve prescribed fits before DEV')
        self.check()
        dev = load_data(c.OUT / 'collection-01/dev.npz', np, 8)
        calls['dev_array_decodes'] += 1
        c.require(not set(dev['case_ids']) & set(train['case_ids']), 'originating case split')
        predictions, reports, timings, sensitivity = {}, [], [], []
        for family in FAMILIES:
            for seed in c.CONFIG['fit_seeds']:
                model = models.make_model(MODEL_KINDS[family], seed)
                model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in checkpoints[family, seed].items()})
                model.eval()
                for condition in ('gap', 'normal', 'opposite'):
                    self.check()
                    tick = time.perf_counter()
                    outputs, cost_outputs, work = [], [], {}
                    for i in range(len(dev['case_ids'])):
                        ix = slice(i, i + 1)
                        batch = {k: torch.from_numpy(dev[k][ix].copy()) for k in INPUT_KEYS}
                        if condition == 'opposite':
                            batch['actions'] ^= 1
                        with torch.no_grad():
                            if condition == 'normal':
                                outcomes = torch.from_numpy(dev['outcomes'][ix].copy())
                                future = torch.from_numpy(dev['continuation'][ix].copy())
                                result = model.normal_rollout(batch['prefix'], batch['prefix_lengths'], batch['actions'], future, found=outcomes == 4)
                                outcome = probabilities(result['outcome_logits'], torch, condition='normal', outcomes=outcomes)
                            else:
                                result = model.blind_rollout(batch['prefix'], batch['prefix_lengths'], batch['actions'])
                                outcome = probabilities(result['outcome_logits'], torch, condition='gap')
                        outputs.append(outcome.numpy())
                        cost_outputs.append(result['cost_contrasts'].numpy())
                        for k, v in result['work'].items():
                            work[k] = work.get(k, 0) + v
                    outcome, q = np.concatenate(outputs), np.concatenate(cost_outputs)
                    timings.append({'family': family, 'seed': seed, 'condition': condition,
                                    'seconds': time.perf_counter() - tick, 'cases': len(outcome), 'work': work})
                    key = f'{family}__{seed}__{condition}'
                    predictions[key + '__outcome'] = outcome
                    if condition != 'opposite':
                        predictions[key + '__cost'] = q
                        reports.append(metrics.score(dev['outcomes'], outcome, q, dev['raw_costs'], dev['legal'],
                            oracle_probabilities=dev[condition + '_oracle'], case_ids=dev['case_ids'].tolist(), regimes=dev['regimes'].tolist(),
                            family=family, fit_seed=seed, condition=condition, prediction_kind='probabilities'))
                sensitivity.append({'family': family, 'fit_seed': seed, 'report': metrics.action_sensitivity(
                    dev['gap_oracle'], dev['opposite_oracle'], case_ids=dev['case_ids'].tolist(), regimes=dev['regimes'].tolist(),
                    actions=dev['actions'], alternate_actions=dev['actions'] ^ 1,
                    predicted_original=predictions[f'{family}__{seed}__gap__outcome'],
                    predicted_alternate=predictions[f'{family}__{seed}__opposite__outcome'])})
        gate = metrics.evaluate_reports(reports, candidate=c.CONFIG['candidate'], controls=c.CONFIG['controls'],
            fit_seeds=c.CONFIG['fit_seeds'], regimes=['lambda3', 'lambda4'], thresholds=THRESHOLDS)
        save_npz(self.out / 'predictions.npz', np, **predictions)
        c.write(self.out / 'reports.json', {'reports': reports, 'gate': gate, 'sensitivity': sensitivity})
        c.write(self.out / 'summary.json', {'status': 'DEV_PASS' if gate['passed'] else 'DEV_FAIL',
                'fits': fits, 'prediction_times': timings, 'calls': calls,
                'collection_receipt': c.desc(c.OUT / 'collection-01/receipt.json'),
                'train_data': c.desc(c.OUT / 'collection-01/train.npz'), 'dev_data': c.desc(c.OUT / 'collection-01/dev.npz'),
                'data_cases': {'train': n, 'dev': len(dev['case_ids'])},
                'scope': 'full-belief TRAIN distillation into fixed-path forecast imitation, not autonomous control'})
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
