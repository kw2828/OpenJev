"""Same-GRU conditional-label comparison, admitted only by its own registration."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import otto_conditional_label_common as c

FAMILIES = ('sampled', 'mean32')
INPUT_KEYS = ('prefix', 'prefix_lengths', 'actions')
UNUSED_PREFIXES = ('outcome_head.', 'aux_head.')


def validate_data(d, np, *, split):
    c.require(split in ('train', 'dev'), 'declared data split')
    n, draws = len(d['case_ids']), c.CONFIG[split + '_draws']
    shapes = {'prefix': (n, 9, 31), 'prefix_lengths': (n,), 'prefix_actions': (n, 8),
        'prefix_outcomes': (n, 8), 'prefix_position': (n, 2), 'actions': (n, 8),
        **dict.fromkeys(('initial_belief', 'root_strict', 'root_grid', 'legacy_root'), (n, 2809)),
        'mc_draws': (n, draws, 9), 'mc_source_indices': (n, draws), 'mc_outcomes': (n, draws, 8),
        'alive': (n, draws), 'costs': (n, draws, 4), 'case_ids': (n,), 'regimes': (n,)}
    c.require(n > 0 and set(d) == set(shapes) and all(d[k].shape == s for k, s in shapes.items()), 'exact bank data schema')
    c.require(all(d[k].dtype.kind == 'U' for k in ('case_ids', 'regimes'))
              and len(set(d['case_ids'])) == n and all(len(v) > 0 for v in d['case_ids']), 'unique Unicode case identities')
    c.require(set(d['regimes']) <= ({'lambda3'} if split == 'train' else {'lambda3', 'lambda4'}), 'declared split regimes')
    for key in ('prefix', 'costs'):
        c.require(d[key].dtype == np.float32 and np.isfinite(d[key]).all(), 'finite float32 ' + key)
    for key in ('initial_belief', 'root_strict', 'root_grid', 'legacy_root'):
        c.require(d[key].dtype == np.float64 and np.isfinite(d[key]).all() and (d[key] >= 0).all(), 'finite nonnegative belief ' + key)
    for key in ('initial_belief', 'root_strict', 'root_grid'):
        c.require(np.allclose(d[key].sum(-1), 1., rtol=0, atol=1e-12), 'normalized ' + key)
    for key in ('prefix_lengths', 'prefix_actions', 'prefix_outcomes', 'prefix_position', 'actions', 'mc_source_indices', 'mc_outcomes'):
        c.require(d[key].dtype == np.int64, 'int64 ' + key)
    c.require(np.all(d['prefix_lengths'] == 9) and all(np.all((d[k] >= 0) & (d[k] < 4))
              for k in ('prefix_actions', 'prefix_outcomes', 'actions')), 'complete nonterminal public prefixes and actions')
    position = np.full((n, 2), 26, np.int64)
    for step in range(8):
        action = d['prefix_actions'][:, step]
        position[np.arange(n), action // 2] += np.where(action % 2 == 0, -1, 1)
    c.require(np.array_equal(position, d['prefix_position']) and np.all((position >= 18) & (position <= 34)), 'center-origin interior geometry')
    c.require(d['mc_draws'].dtype == np.uint64 and (d['mc_draws'] < 2**53).all()
              and np.all((d['mc_source_indices'] >= 0) & (d['mc_source_indices'] < 2809)), 'finite-law saved draws')
    outcomes = d['mc_outcomes']
    c.require(np.all((outcomes >= 0) & (outcomes <= 4)), 'five outcome classes')
    found = outcomes == 4
    c.require(d['alive'].dtype == np.bool_ and np.array_equal(found, np.maximum.accumulate(found, -1))
              and np.array_equal(d['alive'], ~found[..., 7]) and (d['costs'] >= 0).all()
              and (d['costs'][~d['alive']] == 0).all(), 'absorbing found and literal zero unconditional costs')
    return d


def load_data(path, np, *, split):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    return validate_data(data, np, split=split)


def validate_predictions(data, predictions, np):
    keys = {f'{family}__{seed}__cost' for family in FAMILIES for seed in c.CONFIG['fit_seeds']}
    c.require(set(predictions) == keys and all(p.shape == (len(data['case_ids']), 8, 4)
              and p.dtype == np.float32 and np.isfinite(p).all() for p in predictions.values()), 'six complete finite float32 prediction views')


def primary_gate(data, predictions, np, *, check=lambda: None):
    validate_predictions(data, predictions, np)
    costs = data['costs'].astype(np.float64)
    regrets = costs - costs.min(-1, keepdims=True)
    n, draws = costs.shape[:2]
    ix, dx = np.arange(n)[:, None], np.arange(draws)[None, :]
    per_fit = {family: np.stack([regrets[ix, dx, predictions[f'{family}__{seed}__cost'][:, 7].argmin(-1)[:, None]]
                                for seed in c.CONFIG['fit_seeds']]) for family in FAMILIES}
    sampled, mean32 = (per_fit[family].mean(0) for family in FAMILIES)
    gain = sampled - mean32
    reps, fraction = c.CONFIG['bootstrap_replicates'], c.CONFIG['required_fraction']
    groups = []
    for regime_index, regime in enumerate(('lambda3', 'lambda4')):
        check()
        selected = np.flatnonzero(data['regimes'] == regime)
        count = len(selected)
        c.require(count > 0, 'each declared evaluation regime present')
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([c.CONFIG['bootstrap_seed'], regime_index])))
        ci = rng.integers(0, count, size=(reps, count), dtype=np.int64)
        di = rng.integers(0, draws, size=(reps, count, draws), dtype=np.int64)
        digest = hashlib.sha256(ci.tobytes())
        digest.update(di.tobytes())
        records = []
        for start in range(0, reps, 100):
            check()
            sl = slice(start, start + 100)
            a = sampled[selected][ci[sl, :, None], di[sl]].mean((1, 2))
            b = mean32[selected][ci[sl, :, None], di[sl]].mean((1, 2))
            g = a - b
            records.extend(np.stack((a, b, g, g - fraction * a), -1).tolist())
        intervals = np.quantile(np.asarray(records), [c.CONFIG['quantile'], 1 - c.CONFIG['quantile']], axis=0, method='linear').T
        paired = [{'fit_seed': seed, 'sampled_gap': float(per_fit['sampled'][i, selected].mean()),
                   'mean32_gap': float(per_fit['mean32'][i, selected].mean()),
                   'gain': float((per_fit['sampled'][i, selected] - per_fit['mean32'][i, selected]).mean())}
                  for i, seed in enumerate(c.CONFIG['fit_seeds'])]
        a, b, g = float(sampled[selected].mean()), float(mean32[selected].mean()), float(gain[selected].mean())
        conditions = {'minimum_cases': count >= c.CONFIG['min_dev_per_regime'], 'positive_sampled_gap': a > 0,
                      'resolved_gain': float(intervals[3, 0]) > 0, 'all_paired_seeds_positive': all(row['gain'] > 0 for row in paired)}
        groups.append({'regime': regime, 'cases': count, 'sampled_gap': a, 'mean32_gap': b, 'gain': g,
            'required_fraction': fraction, 'gain_minus_required': g - fraction * a, 'paired_seed_gains': paired,
            'interval_columns': ['sampled_gap', 'mean32_gap', 'gain', 'gain_minus_required'],
            'approximate_95_percent_interval': intervals.tolist(), 'bootstrap_replicates': records,
            'bootstrap_index_sha256': digest.hexdigest(), 'conditions': conditions, 'passed': all(conditions.values())})
    passed = all(row['passed'] for row in groups)
    return {'version': 'otto-conditional-label-gate-v1', 'passed': passed, 'status': 'DEV_PASS' if passed else 'DEV_FAIL',
        'groups': groups, 'scope': 'Approximate hierarchical percentile intervals conditional on fitted policies and TRAIN banks; not independent fit uncertainty.',
        'aggregation': 'Equal originating prefixes including found zero; average three paired policies as separate decisions, not averaged logits.',
        'architecture_claim': False}


def analyze(data, predictions, np, *, check=lambda: None):
    validate_predictions(data, predictions, np)
    costs = data['costs'].astype(np.float64)
    normalized = costs / 64.
    target = normalized - normalized.mean(-1, keepdims=True)
    mean_target = target.mean(1)
    variance = ((target - mean_target[:, None])**2).mean((1, 2))
    regrets = costs - costs.min(-1, keepdims=True)
    n, draws = costs.shape[:2]
    rows, cases = [], []
    for family in FAMILIES:
        for seed in c.CONFIG['fit_seeds']:
            check()
            prediction = predictions[f'{family}__{seed}__cost'][:, 7].astype(np.float64)
            actions = prediction.argmin(-1)
            gap = regrets[np.arange(n)[:, None], np.arange(draws)[None, :], actions[:, None]].mean(1)
            sample_mse = ((prediction[:, None] - target)**2).mean((1, 2))
            mean_mse = ((prediction - mean_target)**2).mean(1)
            values = {'teacher_regret': gap, 'sampled_target_mse': sample_mse,
                      'mean_target_mse': mean_mse, 'within_bank_variance': variance}
            for i, identity in enumerate(data['case_ids']):
                if i % 32 == 0:
                    check()
                cases.append({'family': family, 'fit_seed': seed, 'case_id': str(identity), 'regime': str(data['regimes'][i]),
                    'action': int(actions[i]), **{key: float(value[i]) for key, value in values.items()},
                    'alive_draws': int(data['alive'][i].sum()), 'draws': draws})
            for regime in ('lambda3', 'lambda4'):
                selected = data['regimes'] == regime
                c.require(bool(selected.any()), 'both evaluation regimes')
                rows.append({'family': family, 'fit_seed': seed, 'regime': regime, 'cases': int(selected.sum()),
                    **{key: float(value[selected].mean()) for key, value in values.items()}})
    return {'version': c.VERSION, 'rows': rows, 'cases': cases, 'gate': primary_gate(data, predictions, np, check=check),
            'horizon': 8, 'normal_observation_claim': False, 'architecture_claim': False}


def save_npz(path, np, **values):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **values)


def state_digest(state):
    """Canonical tensor identity without serialization/container variability."""
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().numpy()
        header = json.dumps([key, value.dtype.str, list(value.shape)], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'little'))
        digest.update(header)
        digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def draw_permutations(seed, cases, np):
    return np.stack([np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, i, 717]))).permutation(32)
                     for i in range(cases)]).astype(np.int64)


def case_order(seed, epoch, cases, np):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch, 919]))).permutation(cases).astype(np.int64)


class Pilot(c.Run):
    def body(self):
        collection = c.closed(c.OUT / 'collection-01', c.OUT / 'collection-native-01.terminal.json')
        c.require(collection['plan_sha256'] == self.args.plan_sha256, 'same closed collection registration')
        import numpy as np
        import torch
        sys.path.insert(0, str(c.ROOT / 'src'))
        from openjev.research import otto_action_latent_model as models
        from openjev.research import otto_conditional_label as loss_functions

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        calls = {'train_array_decodes': 0, 'dev_array_decodes': 0, 'checkpoint_decodes': 0,
                 'model_constructions': 0, 'fit_count': 0, 'optimizer_attempts': 0, 'optimizer_steps': 0,
                 'training_rollouts': 0, 'training_prefix_exposures': 0, 'target_draw_uses': 0,
                 'mean_target_aggregation_calls': 0, 'mean_target_aggregation_draws': 0,
                 'evaluation_rollouts': 0, 'bootstrap_replicates': 0}
        self.receipt['calls'] = calls
        calls['train_array_decodes'] += 1
        train = load_data(c.OUT / 'collection-01/train.npz', np, split='train')
        n = len(train['case_ids'])
        c.require(n >= c.CONFIG['min_train'] and set(train['regimes']) == {'lambda3'}, 'registered retained TRAIN support')
        tensors = {key: torch.from_numpy(train[key].copy()) for key in INPUT_KEYS}
        bank = loss_functions.centered_bank(torch.from_numpy(train['costs'].copy()), torch.from_numpy(train['alive'].copy()))
        scaling = loss_functions.train_scale(bank, variance_floor=c.CONFIG['variance_floor'])
        scale = scaling['cost_scale']
        aggregation_started = self.clock.now_ns()
        mean_targets = loss_functions.select_target(bank, 'mean32')
        aggregation_seconds = (self.clock.now_ns() - aggregation_started) / 1e9
        calls['mean_target_aggregation_calls'] = 1
        calls['mean_target_aggregation_draws'] = n * 32
        c.write(self.out / 'normalization.json', {**scaling, 'source': 'TRAIN only, equal prefixes and all32 draws and four centered costs, found zero',
            'teacher_units_divisor': 64., 'train_cases': n, 'draws_per_case': 32, 'dev_decodes': 0,
            'mean_target_aggregation_seconds': aggregation_seconds,
            'mean_target_aggregation_draws': n * 32,
            'target_draw_uses_scope': 'Conceptual label contributions per optimizer target; mean32 targets are actually aggregated once.'})
        fits, checkpoints, initial_hashes = [], {}, {}
        for seed_index, seed in enumerate(c.CONFIG['fit_seeds']):
            permutations = draw_permutations(seed, n, np)
            permutation_sha = hashlib.sha256(permutations.tobytes()).hexdigest()
            c.append(self.out / 'draw-orders.jsonl', {'seed': seed, 'permutations': permutations.tolist(),
                'sha256': permutation_sha, 'cycles': 3, 'index_seed_suffix': 717})
            order = FAMILIES[seed_index % 2:] + FAMILIES[:seed_index % 2]
            for family in order:
                self.check()
                tick = self.clock.now_ns()
                calls['model_constructions'] += 1
                model = models.make_model('action_recurrent', seed, cost_scale=scale)
                for name, parameter in model.named_parameters():
                    if name.startswith(UNUSED_PREFIXES):
                        parameter.requires_grad_(False)
                initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
                initial_sha = state_digest(initial)
                c.require(seed not in initial_hashes or initial_hashes[seed] == initial_sha, 'bitwise identical paired initialization')
                initial_hashes[seed] = initial_sha
                parameters = [p for p in model.parameters() if p.requires_grad]
                c.require(sum(p.numel() for p in parameters) == 8096, 'effective cost-only parameter count')
                optimizer = torch.optim.Adam(parameters, lr=c.CONFIG['learning_rate'])
                updates, work = 0, {}
                for epoch in range(c.CONFIG['epochs']):
                    indices = case_order(seed, epoch, n, np)
                    draws = permutations[:, epoch % 32]
                    c.append(self.out / 'training-orders.jsonl', {'family': family, 'seed': seed, 'epoch': epoch,
                        'indices': indices.tolist(), 'draw_indices': draws.tolist(),
                        'index_sha256': hashlib.sha256(indices.tobytes()).hexdigest(),
                        'draw_sha256': hashlib.sha256(draws.tobytes()).hexdigest()})
                    for start in range(0, n, c.CONFIG['batch_size']):
                        self.check()
                        ix = indices[start:start + c.CONFIG['batch_size']]
                        pending = {'family': family, 'seed': seed, 'epoch': epoch, 'batch_start': start,
                                   'cases': len(ix), 'update': updates}
                        self.receipt['pending_fit'] = pending
                        c.append(self.out / 'updates.jsonl', {'event': 'attempt', **pending})
                        calls['optimizer_attempts'] += 1
                        optimizer.zero_grad(set_to_none=True)
                        result = model.blind_rollout(*(tensors[key][ix] for key in INPUT_KEYS))
                        calls['training_rollouts'] += 1
                        calls['training_prefix_exposures'] += len(ix)
                        for key, value in result['work'].items():
                            work[key] = work.get(key, 0) + value
                        selected = torch.from_numpy(draws[ix].copy()) if family == 'sampled' else None
                        target = mean_targets[ix] if family == 'mean32' else loss_functions.select_target(bank[ix], family, indices=selected)
                        calls['target_draw_uses'] += len(ix) * (32 if family == 'mean32' else 1)
                        loss = loss_functions.horizon8_loss(result['cost_contrasts'], target, cost_scale=scale)
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(parameters, c.CONFIG['clip_norm'], error_if_nonfinite=True)
                        optimizer.step()
                        updates += 1
                        calls['optimizer_steps'] += 1
                        c.require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'finite updated state')
                        c.append(self.out / 'updates.jsonl', {'event': 'return', **pending,
                            'loss': float(loss.detach()), 'gradient_norm': float(norm), 'work': result['work']})
                        self.receipt['pending_fit'] = None
                    if (epoch + 1) % 32 == 0:
                        print(f'{family} seed={seed} epoch={epoch + 1}/96', flush=True)
                unused = {key: value for key, value in model.state_dict().items() if key.startswith(UNUSED_PREFIXES)}
                c.require(all(torch.equal(value, initial[key]) for key, value in unused.items())
                          and torch.equal(model.cost_scale, initial['cost_scale']), 'unused heads and scale unchanged')
                state = {key: value.detach().numpy().copy() for key, value in model.state_dict().items()}
                path = self.out / f'{family}-{seed}.npz'
                save_npz(path, np, **state)
                row = {'family': family, 'seed': seed, 'model_kind': 'action_recurrent', 'epochs': c.CONFIG['epochs'],
                    'cases': n, 'updates': updates, 'exposures': n * c.CONFIG['epochs'],
                    'target_draw_uses': n * c.CONFIG['epochs'] * (32 if family == 'mean32' else 1),
                    'unique_teacher_labels': n * 32, 'draw_permutation_sha256': permutation_sha,
                    'sampled_cycles': 3 if family == 'sampled' else None, 'initial_state_sha256': initial_sha,
                    'final_state_sha256': state_digest(model.state_dict()),
                    'unused_heads_before_sha256': state_digest({key: initial[key] for key in unused}),
                    'unused_heads_after_sha256': state_digest(unused), 'checkpoint': c.desc(path),
                    'parameters': model.parameter_metadata(), 'effective_parameter_names': [key for key, p in model.named_parameters() if p.requires_grad],
                    'seconds': (self.clock.now_ns() - tick) / 1e9, 'last_batch_loss': float(loss.detach()), 'last_gradient_norm': float(norm),
                    'training_horizons': [8], 'executed_horizons': list(range(1, 9)), 'work': work,
                    'evaluation_decodes_so_far': 0, 'blind_input_keys': list(INPUT_KEYS), 'privileged_targets_only': ['costs', 'alive']}
                fits.append(row)
                checkpoints[family, seed] = state
                calls['fit_count'] += 1
                c.append(self.out / 'fits.jsonl', row)
        c.require(len(fits) == 6 and calls['dev_array_decodes'] == 0, 'six final checkpoint barrier before DEV')
        c.write(self.out / 'fit-barrier.json', {'fits_completed': 6, 'checkpoint_files': {f'{f}-{s}.npz': c.desc(self.out / f'{f}-{s}.npz')
            for f in FAMILIES for s in c.CONFIG['fit_seeds']}, 'train_data': c.desc(c.OUT / 'collection-01/train.npz'),
            'dev_array_decodes': 0, 'created_ns': self.clock.now_ns(), 'fits_sha256': c.desc(self.out / 'fits.jsonl')['sha256']})
        self.check()
        calls['dev_array_decodes'] += 1
        dev = load_data(c.OUT / 'collection-01/dev.npz', np, split='dev')
        c.require(not set(dev['case_ids']) & set(train['case_ids']), 'disjoint originating TRAIN and DEV cases')
        predictions, timings = {}, []
        for family in FAMILIES:
            for seed in c.CONFIG['fit_seeds']:
                self.check()
                calls['model_constructions'] += 1
                model = models.make_model('action_recurrent', seed, cost_scale=scale)
                model.load_state_dict({key: torch.from_numpy(value.copy()) for key, value in checkpoints[family, seed].items()})
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                model.eval()
                before = state_digest(model.state_dict())
                tick, outputs, work = self.clock.now_ns(), [], {}
                for i in range(len(dev['case_ids'])):
                    self.check()
                    batch = {key: torch.from_numpy(dev[key][i:i + 1].copy()) for key in INPUT_KEYS}
                    with torch.no_grad():
                        result = model.blind_rollout(*(batch[key] for key in INPUT_KEYS))
                    calls['evaluation_rollouts'] += 1
                    outputs.append(result['cost_contrasts'].numpy().copy())
                    for key, value in result['work'].items():
                        work[key] = work.get(key, 0) + value
                c.require(state_digest(model.state_dict()) == before, 'inference preserves all checkpoint state')
                predictions[f'{family}__{seed}__cost'] = np.concatenate(outputs)
                timings.append({'family': family, 'seed': seed, 'seconds': (self.clock.now_ns() - tick) / 1e9,
                    'cases': len(dev['case_ids']), 'work': work, 'state_before': before, 'state_after': before})
        save_npz(self.out / 'predictions.npz', np, **predictions)
        self.receipt['pending_analysis'] = True
        report = analyze(dev, predictions, np, check=self.check)
        calls['bootstrap_replicates'] = 2 * c.CONFIG['bootstrap_replicates']
        self.receipt['pending_analysis'] = None
        c.write(self.out / 'reports.json', report)
        c.write(self.out / 'summary.json', {'status': report['gate']['status'], 'fits': fits, 'prediction_times': timings,
            'calls': calls, 'collection_receipt': c.desc(c.OUT / 'collection-01/receipt.json'),
            'train_data': c.desc(c.OUT / 'collection-01/train.npz'), 'dev_data': c.desc(c.OUT / 'collection-01/dev.npz'),
            'data_cases': {'train': n, 'dev': len(dev['case_ids'])}, 'normalization': scaling,
            'scope': 'Same initial GRU, fixed bank and model-update budget; target aggregation differs. H8 teacher-cost imitation only, no normal-observation or architecture claim.'})


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
