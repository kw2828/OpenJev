"""Matched paired-branch fits with prospective action-effect supervision."""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import otto_action_effect_common as c
from run_otto_belief_distillation import load_data, probabilities, save_npz, train_cost_scale

FAMILIES = ('effect_recurrent', 'paired_recurrent', 'paired_blind', 'paired_direct')
MODEL_KINDS = dict(zip(FAMILIES, ('action_recurrent', 'action_recurrent', 'action_blind', 'direct_horizon'), strict=True))
THRESHOLDS = {'long_effect_relative_gain': .10, 'long_gap_relative_gain': .05,
              'long_log_relative_tolerance': .01, 'normal_log_relative_tolerance': .01,
              'normal_gap_relative_tolerance': .01, 'minimum_supported_cases': 48}
LEGACY_THRESHOLDS = {'long_log_relative_gain': .01, 'long_gap_relative_gain': .05,
                     'normal_log_relative_tolerance': .01, 'normal_gap_relative_tolerance': .01,
                     'minimum_supported_cases': 48}
INPUT_KEYS = ('prefix', 'prefix_lengths', 'actions')
TARGET_KEYS = ('continuation', 'outcomes', 'raw_costs', 'gap_oracle', 'normal_oracle', 'opposite_oracle')
EFFECT_SOURCE = 'TRAIN only, mean squared signed oracle effect, equal cases and all four horizons'


def train_effect_variance(data, np, *, return_variance=False):
    delta = data['gap_oracle'] - data['opposite_oracle']
    variance = float(np.square(delta).sum(-1).mean())
    c.require(math.isfinite(variance) and variance >= 0, 'finite TRAIN effect variance')
    scale = max(variance, c.CONFIG['effect_variance_floor'])
    return (scale, variance) if return_variance else scale


def paired_loss(gap, normal, opposite, batch, torch, cost_scale, *, effect_variance, family):
    """Every arm sees both branches; only the registered candidate weights E."""
    c.require(family in FAMILIES and math.isfinite(cost_scale) and cost_scale > 0
              and math.isfinite(effect_variance) and effect_variance > 0, 'fixed paired loss inputs')
    outcomes = batch['outcomes']
    alive = outcomes != 4
    centered = batch['raw_costs'] / 64.
    centered = centered - centered.mean(-1, keepdim=True)

    def ce(prediction, target, observed=False):
        row = -(target * torch.log_softmax(prediction['outcome_logits'], dim=-1)).sum(-1)
        if observed:
            resolved = torch.zeros_like(alive)
            resolved[:, 1:] = (outcomes[:, :-1] == 4).cummax(dim=1).values
            row = row.masked_fill(resolved, 0.)
        return row.mean()

    def cost_aux(prediction):
        q = ((prediction['cost_contrasts'] - centered).square().mean(-1) * alive).sum(-1) / alive.sum(-1).clamp_min(1)
        aux = ((prediction['aux_features'] - batch['continuation'][..., 19:21]).square().mean(-1) * alive).sum(-1) / alive.sum(-1).clamp_min(1)
        return c.CONFIG['cost_weight'] * q.mean() / cost_scale**2 + c.CONFIG['aux_weight'] * aux.mean()

    paired_ce = .5 * (ce(gap, batch['gap_oracle']) + ce(opposite, batch['opposite_oracle']))
    outcome = .5 * (paired_ce + ce(normal, batch['normal_oracle'], observed=True))
    supervision = .5 * (cost_aux(gap) + cost_aux(normal))
    predicted_effect = torch.softmax(gap['outcome_logits'].to(torch.float64), -1) - torch.softmax(opposite['outcome_logits'].to(torch.float64), -1)
    oracle_effect = batch['gap_oracle'] - batch['opposite_oracle']
    effect = (predicted_effect - oracle_effect).square().sum(-1).mean()
    weight = c.CONFIG['effect_weight'] if family == c.CONFIG['candidate'] else 0.
    # Keep the same forward and backward graph in controls, with zero weight.
    return outcome + supervision + weight * effect / effect_variance


class Pilot(c.Run):
    def body(self):
        collection = c.closed(c.OUT / 'collection-01', c.OUT / 'collection-native-01.terminal.json')
        c.require(collection['plan_sha256'] == self.args.plan_sha256, 'collection shares registration')
        import numpy as np
        import torch
        sys.path.insert(0, str(c.ROOT / 'src'))
        from openjev.research import otto_action_effect_metrics as metrics
        from openjev.research import otto_action_latent_model as models
        from openjev.research import otto_belief_distillation_metrics as legacy_metrics
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        reference = c.training_reference()
        c.require(reference == c.read(c.OUT / 'collection-01/training-reference.json'), 'unchanged TRAIN lineage')
        train = load_data(c.PARENT_TRAIN, np, 4)
        n = len(train['case_ids'])
        c.require(n == c.CONFIG['train_cases'] == reference['cases'] and set(train['regimes']) == {'lambda3'}, 'only registered TRAIN')
        tensors = {k: torch.from_numpy(train[k].copy()) for k in (*INPUT_KEYS, *TARGET_KEYS)}
        cost_scale, cost_variance = train_cost_scale(train, np, return_variance=True)
        effect_variance, effect_before_floor = train_effect_variance(train, np, return_variance=True)
        c.write(self.out / 'normalization.json', {'cost_scale': cost_scale, 'rms_squared_before_floor': cost_variance,
                'variance_floor': 1e-6, 'source': 'TRAIN only, all-four centered, equal cases and surviving rows',
                'teacher_units_divisor': 64., 'dev_decodes': 0, 'effect_variance': effect_variance,
                'effect_variance_before_floor': effect_before_floor, 'effect_variance_floor': c.CONFIG['effect_variance_floor'],
                'effect_source': EFFECT_SOURCE})
        fits, checkpoints, calls = [], {}, {'optimizer_steps': 0, 'fit_count': 0, 'dev_array_decodes': 0, 'train_array_decodes': 1, 'training_rollouts': 0}
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
                        opposite = model.blind_rollout(batch['prefix'], batch['prefix_lengths'], batch['actions'] ^ 1)
                        calls['training_rollouts'] += 3
                        loss = paired_loss(blind, normal, opposite, batch, torch, cost_scale,
                                           effect_variance=effect_variance, family=family)
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
                       'soft_targets': True, 'training_rollouts_per_batch': 3,
                       'effect_weight': c.CONFIG['effect_weight'] if family == c.CONFIG['candidate'] else 0., 'last_batch_loss': float(loss.detach()), 'last_gradient_norm': float(norm),
                       'parameters': model.parameter_metadata(), 'checkpoint': c.desc(path),
                       'changed_tensors': [k for k, v in model.state_dict().items() if not torch.equal(v, initial_state[k])],
                       'training_horizons': [1, 2, 3, 4], 'evaluation_decodes_so_far': 0,
                       'blind_input_keys': list(INPUT_KEYS), 'privileged_targets_only': ['raw_costs', 'gap_oracle', 'normal_oracle', 'opposite_oracle']}
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
        gate = metrics.evaluate_reports(reports, sensitivity, candidate=c.CONFIG['candidate'], controls=c.CONFIG['controls'],
            fit_seeds=c.CONFIG['fit_seeds'], regimes=['lambda3', 'lambda4'], thresholds=THRESHOLDS)
        legacy_gate = legacy_metrics.evaluate_reports(reports, candidate=c.CONFIG['candidate'], controls=c.CONFIG['controls'],
            fit_seeds=c.CONFIG['fit_seeds'], regimes=['lambda3', 'lambda4'], thresholds=LEGACY_THRESHOLDS)
        save_npz(self.out / 'predictions.npz', np, **predictions)
        c.write(self.out / 'reports.json', {'reports': reports, 'gate': gate, 'sensitivity': sensitivity, 'legacy_gate': legacy_gate})
        c.write(self.out / 'summary.json', {'status': 'DEV_PASS' if gate['passed'] else 'DEV_FAIL',
                'fits': fits, 'prediction_times': timings, 'calls': calls,
                'collection_receipt': c.desc(c.OUT / 'collection-01/receipt.json'),
                'training_reference': reference, 'train_data': c.desc(c.PARENT_TRAIN), 'dev_data': c.desc(c.OUT / 'collection-01/dev.npz'),
                'data_cases': {'train': n, 'dev': len(dev['case_ids'])},
                'scope': 'paired action-effect TRAIN supervision for fixed-path forecast imitation, not autonomous control'})
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
