"""Frozen-policy decisions and fixed-horizon conditional-cost diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import otto_cost_information_common as c

FAMILIES = c.FAMILIES
MODEL_KINDS = dict(zip(FAMILIES, ('action_recurrent', 'action_recurrent', 'action_blind', 'direct_horizon'), strict=True))
INPUT_KEYS = ('prefix', 'prefix_lengths', 'actions')


def validate_data(d, np):
    n, m = len(d['case_ids']), c.CONFIG['mc_draws']
    shapes = {'prefix': (n, 9, 31), 'prefix_lengths': (n,), 'prefix_actions': (n, 8),
        'prefix_outcomes': (n, 8), 'prefix_position': (n, 2), 'actions': (n, 8),
        **dict.fromkeys(('initial_belief', 'root_strict', 'root_grid', 'legacy_root'), (n, 2809)),
        'exact_weights': (n, 256), 'exact_costs': (n, 256, 4), 'exact_alive': (n, 256),
        'mc_draws': (n, 2, m, 9), 'mc_source_indices': (n, 2, m), 'mc_outcomes': (n, 2, m, 8),
        'mc_alive4': (n, 2, m), 'mc_alive8': (n, 2, m),
        'mc_costs4': (n, 2, m, 4), 'mc_costs8': (n, 2, m, 4), 'case_ids': (n,), 'regimes': (n,)}
    c.require(set(d) == set(shapes) and all(d[k].shape == s for k, s in shapes.items()), 'exact diagnostic array schema')
    c.require(n > 0 and len(set(d['case_ids'])) == n and np.all(d['prefix_lengths'] == 9), 'unique complete prefixes')
    for k in ('prefix', 'exact_costs', 'mc_costs4', 'mc_costs8'):
        c.require(d[k].dtype == np.float32 and np.isfinite(d[k]).all(), 'finite float32 ' + k)
    for k in ('initial_belief', 'root_strict', 'root_grid', 'legacy_root', 'exact_weights'):
        c.require(d[k].dtype == np.float64 and np.isfinite(d[k]).all() and (d[k] >= 0).all(), 'finite float64 ' + k)
    for k in ('prefix_lengths', 'prefix_actions', 'prefix_outcomes', 'prefix_position', 'actions', 'mc_source_indices', 'mc_outcomes'):
        c.require(d[k].dtype == np.int64, 'int64 ' + k)
    for k in ('exact_alive', 'mc_alive4', 'mc_alive8'):
        c.require(d[k].dtype == np.bool_, 'Boolean ' + k)
    c.require(d['mc_draws'].dtype == np.uint64 and (d['mc_draws'] < 2**53).all(), 'saved 53-bit integer draws')
    c.require(all(np.all((d[k] >= 0) & (d[k] < 4)) for k in ('prefix_actions', 'prefix_outcomes', 'actions'))
              and np.all((d['mc_source_indices'] >= 0) & (d['mc_source_indices'] < 2809))
              and np.all((d['mc_outcomes'] >= 0) & (d['mc_outcomes'] <= 4)), 'categorical arrays')
    for k in ('initial_belief', 'root_strict', 'root_grid'):
        c.require(np.allclose(d[k].sum(-1), 1., rtol=0, atol=1e-12), 'normalized ' + k)
    c.require(np.all((d['prefix_position'] >= 8) & (d['prefix_position'] <= 44)), 'inbounds committed path')
    found = d['mc_outcomes'] == 4
    c.require(np.array_equal(found, np.maximum.accumulate(found, -1)), 'absorbing found in Monte Carlo histories')
    c.require(np.array_equal(d['exact_alive'], d['exact_weights'] > 0), 'exact positive leaf support')
    for h in (4, 8):
        alive, costs = d[f'mc_alive{h}'], d[f'mc_costs{h}']
        c.require(np.array_equal(alive, ~found[..., h - 1]) and (costs >= 0).all()
                  and (costs[~alive] == 0).all(), 'terminal and nonnegative MC costs')
    c.require((d['exact_costs'] >= 0).all() and (d['exact_costs'][~d['exact_alive']] == 0).all(), 'exact cost support')
    c.require(np.all(d['exact_weights'].sum(-1) <= 1 + 1e-12), 'exact subprobability')
    return d


def load_data(path, np):
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    return validate_data(data, np)


def survival(data, np, horizon):
    position = data['prefix_position'].copy()
    visited = [set() for _ in position]
    for h in range(horizon):
        for i, action in enumerate(data['actions'][:, h]):
            position[i, action // 2] += -1 if action % 2 == 0 else 1
            visited[i].add(int(position[i, 0] * 53 + position[i, 1]))
    return np.asarray([float(np.delete(data['root_grid'][i], sorted(v)).sum()) for i, v in enumerate(visited)])


def validate_predictions(data, predictions, np):
    expected = {f'{f}__{s}__cost' for f in FAMILIES for s in c.FIT_SEEDS}
    c.require(set(predictions) == expected and all(p.shape == (len(data['case_ids']), 8, 4)
              and p.dtype == np.float32 and np.isfinite(p).all() for p in predictions.values()), 'all frozen decision forecasts')


def primary_gate(data, predictions, np, *, check=lambda: None):
    """Achievable gain using separate selection/evaluation draws, no oracle claim."""
    validate_predictions(data, predictions, np)
    m, reps = c.CONFIG['mc_draws'], c.CONFIG['bootstrap_replicates']
    select, evaluation = data['mc_costs8'][:, 0].astype(np.float64), data['mc_costs8'][:, 1].astype(np.float64)
    reference_actions = select.mean(1).argmin(-1)
    regrets = evaluation - evaluation.min(-1, keepdims=True)
    indices = np.arange(len(evaluation))[:, None]
    draw_indices = np.arange(m)[None, :]
    selected = regrets[indices, draw_indices, reference_actions[:, None]]
    actions = np.stack([predictions[f"{c.CONFIG['primary_family']}__{seed}__cost"][:, 7].argmin(-1)
                        for seed in c.FIT_SEEDS], axis=-1)
    control = np.stack([regrets[indices, draw_indices, actions[:, j, None]] for j in range(3)]).mean(0)
    delta = control - selected
    exact_survival = survival(data, np, 8)
    case_rows = [{'case_id': str(data['case_ids'][i]), 'regime': str(data['regimes'][i]),
        'reference_action': int(reference_actions[i]), 'control_actions': actions[i].tolist(),
        'control_gap': float(control[i].mean()), 'reference_gap': float(selected[i].mean()),
        'gain': float(delta[i].mean()), 'exact_survival': float(exact_survival[i]),
        'select_survivors': int(data['mc_alive8'][i, 0].sum()),
        'eval_survivors': int(data['mc_alive8'][i, 1].sum())} for i in range(len(evaluation))]
    groups = []
    for j, regime in enumerate(('lambda3', 'lambda4')):
        check()
        ix = np.flatnonzero(data['regimes'] == regime)
        n = len(ix)
        c.require(n > 0, 'each declared regime present')
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([c.CONFIG['bootstrap_seed'], j])))
        ci = rng.integers(0, n, size=(reps, n), dtype=np.int64)
        di = rng.integers(0, m, size=(reps, n, m), dtype=np.int64)
        records = []
        fraction = c.CONFIG['base_headroom_fraction'] if j == 0 else c.CONFIG['shift_headroom_fraction']
        for start in range(0, reps, 100):
            check()
            sl = slice(start, start + 100)
            bcontrol = control[ix][ci[sl, :, None], di[sl]].mean((1, 2))
            bgain = delta[ix][ci[sl, :, None], di[sl]].mean((1, 2))
            records.extend(np.stack([bcontrol, bgain, bgain - fraction * bcontrol], -1).tolist())
        replicates = np.asarray(records, np.float64)
        bounds = np.quantile(replicates, [c.CONFIG['quantile'], 1 - c.CONFIG['quantile']], axis=0, method='linear')
        support = all(row['exact_survival'] == 0 or min(row['select_survivors'], row['eval_survivors']) >= c.CONFIG['min_mc_survivors']
                      for row in case_rows if row['regime'] == regime)
        baseline, gain = float(control[ix].mean()), float(delta[ix].mean())
        conditions = {'minimum_cases': n >= c.CONFIG['min_cases'], 'mc_support': support,
                      'positive_control_gap': baseline > 0, 'resolved_gain': float(bounds[0, 2]) > 0}
        groups.append({'regime': regime, 'cases': n, 'control_gap': baseline, 'reference_gap': float(selected[ix].mean()),
            'gain': gain, 'required_fraction': fraction, 'gain_minus_required': gain - fraction * baseline,
            'interval_columns': ['control_gap', 'gain', 'gain_minus_required'], 'approximate_95_percent_interval': bounds.T.tolist(),
            'bootstrap_replicates': records, 'bootstrap_index_sha256': hashlib.sha256(ci.tobytes() + di.tobytes()).hexdigest(),
            'conditions': conditions, 'passed': all(conditions.values())})
    passed = all(g['passed'] for g in groups)
    return {'version': 'otto-cost-information-headroom-v1', 'passed': passed,
        'status': 'HEADROOM_RESOLVED' if passed else 'HEADROOM_NOT_RESOLVED', 'groups': groups, 'cases': case_rows,
        'scope': 'Approximate hierarchical percentile intervals, conditional on realized selection draws and fixed models; not a rigorous population bound.',
        'selection': '128 separate draws choose a full-public-history reference action; 128 independent draws evaluate it.',
        'aggregation': 'Mean of three frozen policies as separate decisions; equal originating prefixes including found draws as zero.',
        'admits_model_training': False, 'architecture_claim': False}


def analyze(data, predictions, np, *, check=lambda: None):
    from openjev.research.otto_conditional_cost_metrics import decompose_cases
    validate_predictions(data, predictions, np)
    rows, full = [], []
    for h in (4, 8):
        check()
        weights = data['exact_weights'] if h == 4 else data['mc_alive8'][:, 1].astype(np.float64) / c.CONFIG['mc_draws']
        costs = data['exact_costs'] if h == 4 else data['mc_costs8'][:, 1]
        cases = [{'case_id': str(identity), 'branches': [{'weight': float(w), 'costs': q.tolist(), 'legal': [True] * 4}
                 for w, q in zip(weights[i], costs[i], strict=True)]} for i, identity in enumerate(data['case_ids'])]
        for family in FAMILIES:
            for seed in c.FIT_SEEDS:
                check()
                action = predictions[f'{family}__{seed}__cost'][:, h - 1].argmin(-1)
                record = decompose_cases(cases, dict(zip(data['case_ids'].tolist(), map(int, action), strict=True)), horizon=h)
                full.append({'family': family, 'fit_seed': seed, 'horizon': h, 'report': record})
                for regime in ('lambda3', 'lambda4'):
                    selected = [r for i, r in enumerate(record['cases']) if data['regimes'][i] == regime]
                    values = {k: float(np.mean([r['support_mass'] * r[k] if r['defined'] else 0. for r in selected]))
                              for k in ('total_regret', 'information_advantage', 'approximation_regret')}
                    rows.append({'family': family, 'fit_seed': seed, 'regime': regime, 'horizon': h,
                        'cases': len(selected), 'measure': 'exact_grid' if h == 4 else 'evaluation_MC_plugin_descriptive', **values})
    calibration = []
    for i, identity in enumerate(data['case_ids']):
        q = data['exact_costs'][i].astype(np.float64)
        exact = (data['exact_weights'][i, :, None] * (q - q.min(-1, keepdims=True))).sum(0)
        for stream in range(2):
            q = data['mc_costs4'][i, stream].astype(np.float64)
            regrets = q - q.min(-1, keepdims=True)
            mean = regrets.mean(0)
            se = regrets.std(0, ddof=1) / np.sqrt(c.CONFIG['mc_draws'])
            calibration.append({'case_id': str(identity), 'regime': str(data['regimes'][i]), 'stream': stream,
                'exact_unconditional_regrets': exact.tolist(), 'sample_unconditional_regrets': mean.tolist(),
                'error': (mean - exact).tolist(), 'estimated_standard_error': se.tolist(),
                'scope': 'Sampling diagnostic only; estimated standard errors are not bounded guarantees.'})
    return {'version': c.VERSION, 'rows': rows, 'decompositions': full,
            'h4_sampling_validation': calibration, 'gate': primary_gate(data, predictions, np, check=check),
            'original_action_effect_gate_unchanged': 'DEV_FAIL 6/18', 'new_training': False}


class Diagnostic(c.Run):
    def body(self):
        receipt = c.closed(c.OUT / 'collection-01', c.OUT / 'collection-native-01.terminal.json')
        c.require(receipt['plan_sha256'] == self.args.plan_sha256, 'registered collection')
        reference = c.parent_reference()
        import numpy as np
        import torch
        sys.path.insert(0, str(c.ROOT / 'src'))
        from openjev.research.otto_action_latent_model import make_model
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        data = load_data(c.OUT / 'collection-01/cases.npz', np)
        wanted = {r['id']: r['regime'] for r in c.roster()}
        c.require(all(wanted.get(k) == r for k, r in zip(data['case_ids'], data['regimes'], strict=True)), 'fresh registered cases')
        predictions, timings = {}, []
        calls = {'data_array_decodes': 1, 'checkpoint_decodes': 0, 'model_constructions': 0,
                 'blind_rollouts': 0, 'training_updates': 0, 'bootstrap_replicates': 0}
        self.receipt['calls'] = calls
        for family in FAMILIES:
            for seed in c.FIT_SEEDS:
                self.check()
                tick = time.perf_counter()
                key = f'{family}__{seed}'
                path = c.ROOT / reference['checkpoints'][key]['path']
                with np.load(path, allow_pickle=False) as archive:
                    state = {k: torch.from_numpy(archive[k].copy()) for k in archive.files}
                calls['checkpoint_decodes'] += 1
                model = make_model(MODEL_KINDS[family], seed)
                calls['model_constructions'] += 1
                model.load_state_dict(state, strict=True)
                model.eval()
                outputs, work = [], {}
                for i in range(len(data['case_ids'])):
                    self.check()
                    batch = [torch.from_numpy(data[k][i:i + 1].copy()) for k in INPUT_KEYS]
                    with torch.no_grad():
                        result = model.blind_rollout(*batch)
                    calls['blind_rollouts'] += 1
                    outputs.append(result['cost_contrasts'].numpy())
                    for k, v in result['work'].items():
                        work[k] = work.get(k, 0) + v
                predictions[key + '__cost'] = np.concatenate(outputs)
                timings.append({'family': family, 'fit_seed': seed, 'seconds': time.perf_counter() - tick,
                                'cases': len(data['case_ids']), 'work': work})
        self.check()
        report = analyze(data, predictions, np, check=self.check)
        calls['bootstrap_replicates'] = 2 * c.CONFIG['bootstrap_replicates']
        with (self.out / 'predictions.npz').open('xb') as stream:
            np.savez_compressed(stream, **predictions)
        c.write(self.out / 'report.json', report)
        c.write(self.out / 'summary.json', {'status': report['gate']['status'], 'calls': calls, 'prediction_times': timings,
            'parent_reference': reference, 'collection_receipt': c.desc(c.OUT / 'collection-01/receipt.json'),
            'cases': {r: int((data['regimes'] == r).sum()) for r in ('lambda3', 'lambda4')},
            'data': c.desc(c.OUT / 'collection-01/cases.npz'), 'scope': 'Frozen-model diagnostic only; no training or architecture claim.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'supervision', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    c.require(Path(sys.executable).absolute() == c.NUMERICAL, 'declared numerical interpreter')
    run = Diagnostic(args, 'predict')
    try:
        run.body()
        run.finish()
    except BaseException as error:
        run.finish(error)
        raise


if __name__ == '__main__':
    main()
