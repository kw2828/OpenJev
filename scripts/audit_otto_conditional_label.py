"""Independent saved-output audit of sampled versus mean-bank supervision.

No learner, optimizer, checkpoint, native runtime or teacher is executed.
All numerical checks concern registered saved arrays or fabricated fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from bisect import bisect_right
from pathlib import Path

import otto_conditional_label_common as c

VERSION = 'otto-conditional-label-saved-audit-v1'
REGIMES = ('lambda3', 'lambda4')


def validate_data(data, np, *, split):
    c.require(split in ('train', 'dev') and type(data) is dict and 'case_ids' in data, 'declared split')
    n, draws = len(data['case_ids']), c.CONFIG[split + '_draws']
    shapes = {'prefix': ((n, 9, 31), np.float32), 'prefix_lengths': ((n,), np.int64),
        'prefix_actions': ((n, 8), np.int64), 'prefix_outcomes': ((n, 8), np.int64),
        'prefix_position': ((n, 2), np.int64), 'actions': ((n, 8), np.int64),
        **{k: ((n, 2809), np.float64) for k in ('initial_belief', 'root_strict', 'root_grid', 'legacy_root')},
        'mc_draws': ((n, draws, 9), np.uint64), 'mc_source_indices': ((n, draws), np.int64),
        'mc_outcomes': ((n, draws, 8), np.int64), 'alive': ((n, draws), np.bool_),
        'costs': ((n, draws, 4), np.float32)}
    c.require(n > 0 and set(data) == set(shapes) | {'case_ids', 'regimes'}, 'exact nonempty saved bank fields')
    for name, (shape, dtype) in shapes.items():
        value = data[name]
        c.require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == dtype
                  and np.isfinite(value).all(), 'finite exact saved array ' + name)
    for key in ('case_ids', 'regimes'):
        c.require(isinstance(data[key], np.ndarray) and data[key].shape == (n,) and data[key].dtype.kind == 'U', 'Unicode identities')
    c.require(len(set(data['case_ids'])) == n and all(str(x).strip() for x in data['case_ids']), 'unique nonempty identities')
    c.require(set(data['regimes']) <= ({'lambda3'} if split == 'train' else set(REGIMES)), 'registered split regimes')
    c.require((data['prefix_lengths'] == 9).all(), 'nine rows from reset')
    for name in ('prefix_actions', 'prefix_outcomes', 'actions'):
        c.require(((data[name] >= 0) & (data[name] < 4)).all(), 'public action/odor categories')
    c.require(((data['prefix_position'] >= 18) & (data['prefix_position'] <= 34)).all(), 'eight prefix moves')
    for name in ('initial_belief', 'root_strict', 'root_grid', 'legacy_root'):
        c.require(((data[name] >= 0) & (data[name] <= 1)).all(), 'belief range')
        if name != 'legacy_root':
            c.require((abs(data[name].sum(-1) - 1.) <= 1e-12).all(), 'normalized shadow law')
    c.require((data['mc_draws'] < 2**53).all(), '53-bit draw integers')
    c.require(((data['mc_source_indices'] >= 0) & (data['mc_source_indices'] < 2809)).all(), 'source cells')
    c.require(((data['mc_outcomes'] >= 0) & (data['mc_outcomes'] <= 4)).all(), 'five outcome classes')
    found = data['mc_outcomes'] == 4
    c.require(np.array_equal(found, np.maximum.accumulate(found, axis=-1)), 'absorbing found suffix')
    c.require(np.array_equal(data['alive'], ~found[..., 7]) and (data['costs'] >= 0).all()
              and (data['costs'][~data['alive']] == 0).all(), 'unconditional nonnegative costs with found zero')
    return n


def validate_predictions(data, predictions, np):
    wanted = {f'{family}__{seed}__cost' for family in c.FAMILIES for seed in c.FIT_SEEDS}
    c.require(type(predictions) is dict and set(predictions) == wanted, 'six complete model forecasts')
    for value in predictions.values():
        c.require(isinstance(value, np.ndarray) and value.dtype == np.float32
                  and value.shape == (len(data['case_ids']), 8, 4) and np.isfinite(value).all(), 'finite forecasts')


def centered_bank(data, np):
    """Independent scalar centering of every recorded draw, including found."""
    rows = []
    for case in data['costs']:
        draws = []
        for value in case:
            q = [float(x) / 64. for x in value]
            mean = math.fsum(q) / 4
            draws.append([x - mean for x in q])
        rows.append(draws)
    return np.asarray(rows, np.float64)


def primary_gate(data, predictions, np, *, check=lambda: None):
    validate_predictions(data, predictions, np)
    n, draws, reps = len(data['case_ids']), c.CONFIG['dev_draws'], c.CONFIG['bootstrap_replicates']
    costs = data['costs']
    c.require(costs.shape == (n, draws, 4) and np.isfinite(costs).all(), 'complete evaluation bank')
    by_seed = {}
    for family in c.FAMILIES:
        for seed in c.FIT_SEEDS:
            values = []
            for i, q in enumerate(costs):
                check()
                action = min(range(4), key=lambda a: (float(predictions[f'{family}__{seed}__cost'][i, 7, a]), a))
                values.append([float(row[action]) - min(map(float, row)) for row in q])
            by_seed[family, seed] = np.asarray(values, np.float64)
    sampled = sum(by_seed['sampled', seed] for seed in c.FIT_SEEDS) / len(c.FIT_SEEDS)
    mean32 = sum(by_seed['mean32', seed] for seed in c.FIT_SEEDS) / len(c.FIT_SEEDS)
    gain = sampled - mean32
    groups = []
    for regime_index, regime in enumerate(REGIMES):
        indices = [i for i, label in enumerate(data['regimes']) if label == regime]
        count = len(indices)
        c.require(count > 0, 'both evaluation regimes present')
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([c.CONFIG['bootstrap_seed'], regime_index])))
        case_draws = rng.integers(0, count, size=(reps, count), dtype=np.int64)
        inner_draws = rng.integers(0, draws, size=(reps, count, draws), dtype=np.int64)
        local_sampled, local_mean32, local_gain = sampled[indices], mean32[indices], gain[indices]
        records = []
        for r in range(reps):
            check()
            ci, di = case_draws[r, :, None], inner_draws[r]
            baseline = _mean(local_sampled[ci, di].reshape(-1))
            candidate = _mean(local_mean32[ci, di].reshape(-1))
            delta = _mean(local_gain[ci, di].reshape(-1))
            records.append([baseline, candidate, delta, delta - c.CONFIG['required_fraction'] * baseline])
        intervals = []
        for column in range(4):
            ordered = sorted(row[column] for row in records)
            endpoints = []
            for q in (c.CONFIG['quantile'], 1 - c.CONFIG['quantile']):
                rank = (reps - 1) * q
                lo, hi = math.floor(rank), math.ceil(rank)
                endpoints.append(ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo))
            intervals.append(endpoints)
        baseline = _mean(local_sampled.reshape(-1))
        candidate = _mean(local_mean32.reshape(-1))
        delta = _mean(local_gain.reshape(-1))
        paired = []
        for seed in c.FIT_SEEDS:
            left = _mean(by_seed['sampled', seed][indices].reshape(-1))
            right = _mean(by_seed['mean32', seed][indices].reshape(-1))
            paired.append({'fit_seed': seed, 'sampled_gap': left, 'mean32_gap': right, 'gain': left - right})
        conditions = {'minimum_cases': count >= c.CONFIG['min_dev_per_regime'],
            'positive_sampled_gap': baseline > 0, 'resolved_gain': intervals[3][0] > 0,
            'all_paired_seeds_positive': all(row['gain'] > 0 for row in paired)}
        groups.append({'regime': regime, 'cases': count, 'sampled_gap': baseline, 'mean32_gap': candidate,
            'gain': delta, 'required_fraction': c.CONFIG['required_fraction'],
            'gain_minus_required': delta - c.CONFIG['required_fraction'] * baseline,
            'paired_seed_gains': paired, 'interval_columns': ['sampled_gap', 'mean32_gap', 'gain', 'gain_minus_required'],
            'approximate_95_percent_interval': intervals, 'bootstrap_replicates': records,
            'bootstrap_index_sha256': hashlib.sha256(case_draws.tobytes() + inner_draws.tobytes()).hexdigest(),
            'conditions': conditions, 'passed': all(conditions.values())})
    passed = all(row['passed'] for row in groups)
    return {'version': 'otto-conditional-label-gate-v1', 'passed': passed,
            'status': 'DEV_PASS' if passed else 'DEV_FAIL', 'groups': groups, 'architecture_claim': False}


def analyze(data, predictions, np, *, check=lambda: None):
    validate_predictions(data, predictions, np)
    bank = centered_bank(data, np)
    cases, rows = [], []
    for family in c.FAMILIES:
        for seed in c.FIT_SEEDS:
            forecast = predictions[f'{family}__{seed}__cost'][:, 7].astype(np.float64)
            for i, identity in enumerate(data['case_ids']):
                check()
                target = [_mean(bank[i, :, a]) for a in range(4)]
                action = min(range(4), key=lambda a: (float(forecast[i, a]), a))
                sample_loss = _mean([(float(pred) - float(y)) ** 2 for draw in bank[i]
                                     for pred, y in zip(forecast[i], draw, strict=True)])
                mean_loss = _mean([(float(pred) - y) ** 2 for pred, y in zip(forecast[i], target, strict=True)])
                variance = _mean([(float(y) - average) ** 2 for draw in bank[i]
                                  for y, average in zip(draw, target, strict=True)])
                c.require(abs(sample_loss - mean_loss - variance) <= 1e-11 * (1 + sample_loss), 'independent finite-bank MSE identity')
                cases.append({'family': family, 'fit_seed': seed, 'case_id': str(identity),
                    'regime': str(data['regimes'][i]), 'action': action,
                    'teacher_regret': _mean([float(q[action]) - min(map(float, q)) for q in data['costs'][i]]),
                    'sampled_target_mse': sample_loss, 'mean_target_mse': mean_loss, 'within_bank_variance': variance,
                    'alive_draws': int(data['alive'][i].sum()), 'draws': len(bank[i])})
            for regime in REGIMES:
                chosen = [r for r in cases if r['family'] == family and r['fit_seed'] == seed and r['regime'] == regime]
                rows.append({'family': family, 'fit_seed': seed, 'regime': regime, 'cases': len(chosen),
                    **{key: _mean([r[key] for r in chosen]) for key in ('teacher_regret', 'sampled_target_mse', 'mean_target_mse', 'within_bank_variance')}})
    return {'version': c.VERSION, 'rows': rows, 'cases': cases, 'gate': primary_gate(data, predictions, np, check=check),
            'horizon': 8, 'normal_observation_claim': False, 'architecture_claim': False}


def verify_tables(tables, np, *, check=lambda: None):
    c.require(set(tables) == {name for r in REGIMES for name in (r, r + '_raw', 'legacy_' + r)}, 'six sensor/kernel tables')
    for regime in REGIMES:
        effective, raw, kernel = tables[regime], tables[regime + '_raw'], tables['legacy_' + regime]
        for value in (effective, raw):
            c.require(isinstance(value, np.ndarray) and value.dtype == np.float64 and value.shape == (105, 105, 4)
                      and np.isfinite(value).all() and ((value >= 0) & (value <= 1)).all()
                      and (value[52, 52] == 0).all(), 'finite categorical sensor table')
        c.require(isinstance(kernel, np.ndarray) and kernel.dtype == np.float64 and kernel.shape == (4, 107, 107)
                  and np.isfinite(kernel).all() and ((kernel >= 0) & (kernel <= 1)).all()
                  and (kernel[:, 53, 53] == 0).all(), 'legacy kernel contract')
        for x in range(105):
            check()
            for y in range(105):
                if x != 52 or y != 52:
                    c.require(np.array_equal(effective[x, y], _cdf_law(raw[x, y], np)), 'independent finite-grid sensor law')


def legacy_update(state, position, odor, kernel, np):
    value = state.reshape(53, 53).copy()
    value[tuple(position)] = 0.
    x, y = 53 - position[0], 53 - position[1]
    value *= kernel[odor, x:x + 53, y:y + 53]
    mass = float(value.sum(dtype=np.float64))
    if mass > 1e-10:
        value /= mass
    c.require(np.isfinite(value).all() and (value >= 0).all(), 'finite unrepaired legacy filter')
    return value.reshape(-1)


def prefix_row(saved, position, odor, last_action, step, sensing, state, np):
    """Check lossless public-history channels and cheap derived statistics."""
    expected = [position[0] / 52, position[1] / 52, 1., 1., 1., 1., sensing / 5,
                *[float(odor == h) for h in range(4)], *[float(last_action == a) for a in range(4)],
                step / 2188, (step % 4) / 2188, 1.]
    c.require(np.array_equal(saved[:18], np.asarray(expected, np.float32)), 'complete public prefix channels')
    positive = state > 0
    entropy = float(-(state[positive] * np.log2(state[positive])).sum() / math.log2(2809))
    coordinates = np.indices((53, 53)).reshape(2, -1).T
    distance = float((state * abs(coordinates - np.asarray(position)).sum(-1)).sum() / 104.)
    target = np.asarray([state.sum(), entropy, distance], np.float32)
    c.require(np.allclose(saved[18:21], target, rtol=2e-7, atol=1e-8), 'public belief summary features')


def verify_probabilities(data, tables, roster, np, *, split, check=lambda: None, forward_check=None):
    """Replay public laws and preallocated counterfactual draws, never teacher Q."""
    n = validate_data(data, np, split=split)
    by = {row['id']: row for row in roster}
    c.require(len(by) == len(roster), 'unique registered roster')
    coordinates = np.indices((53, 53)).reshape(2, -1).T
    draws = c.CONFIG[split + '_draws']
    rows = []
    for i, identity in enumerate(data['case_ids']):
        check()
        c.require(identity in by and by[identity]['split'] == split
                  and by[identity]['regime'] == data['regimes'][i], 'registered split identity')
        entry, regime = by[identity], str(data['regimes'][i])
        kernel = tables['legacy_' + regime]

        def likelihood(position, regime=regime):
            offsets = coordinates - np.asarray(position) + 52
            return tables[regime][offsets[:, 0], offsets[:, 1]]

        initial = np.full(2809, 1. / 2808, np.float64)
        initial[26 * 53 + 26] = 0.
        legacy = legacy_update(initial, (26, 26), entry['initial_hit'], kernel, np)
        initial = _cdf_law(legacy, np)
        initial /= initial.sum(dtype=np.float64)
        c.require(np.allclose(initial, data['initial_belief'][i], rtol=1e-12, atol=0.), 'initial conditional source law')
        strict = initial.copy()
        prefix_row(data['prefix'][i, 0], (26, 26), entry['initial_hit'], None, 0, float(regime[-1]), legacy, np)
        prefix_positions = _path((26, 26), data['prefix_actions'][i])
        for t, (position, odor) in enumerate(zip(prefix_positions, data['prefix_outcomes'][i], strict=True), 1):
            check()
            index = position[0] * 53 + position[1]
            law = likelihood(position)[:, int(odor)]
            previous = strict.copy()
            previous[index] = 0.
            joint = previous * law
            c.require(not ((previous > 0) & (law > 0) & (joint == 0)).any(), 'no positive shadow product underflow')
            mass = float(joint.sum(dtype=np.float64))
            c.require(mass > 0, 'positive observed-prefix evidence')
            strict = joint / mass
            legacy = legacy_update(legacy, position, int(odor), kernel, np)
            prefix_row(data['prefix'][i, t], position, int(odor), int(data['prefix_actions'][i, t - 1]),
                       t, float(regime[-1]), legacy, np)
        c.require(tuple(data['prefix_position'][i]) == prefix_positions[-1], 'reconstructed prefix position')
        for value, name in ((strict, 'root_strict'), (legacy, 'legacy_root')):
            c.require(np.allclose(value, data[name][i], rtol=1e-12, atol=0.), 'independent public belief ' + name)
        grid = _cdf_law(strict, np)
        c.require(np.array_equal(grid, data['root_grid'][i]), 'exact declared root-grid conversion')
        tv = .5 * float(abs(grid - strict).sum(dtype=np.float64))
        c.require(tv <= c.CONFIG['root_grid_max_tv'], 'root-grid variation limit')
        actions = np.random.Generator(np.random.PCG64(np.random.SeedSequence([entry['seed'], 911]))).integers(0, 4, size=8, dtype=np.int64)
        c.require(np.array_equal(actions, data['actions'][i]), 'precommitted complete action sequence')
        positions = _path(prefix_positions[-1], actions)
        indices = [x * 53 + y for x, y in positions]
        laws = [likelihood(p) for p in positions]
        expected_draws = np.random.PCG64(entry['mc_seed']).random_raw((draws, 9)) >> np.uint64(11)
        c.require(np.array_equal(expected_draws, data['mc_draws'][i]), 'all source/odor draw integers allocated before outcomes')
        root_cdf = _integer_cdf(grid)
        odor_draws = alive = 0
        for sample in range(draws):
            check()
            integers = data['mc_draws'][i, sample]
            source = bisect_right(root_cdf, int(integers[0]))
            c.require(source == int(data['mc_source_indices'][i, sample]) and grid[source] > 0, 'hypothetical source identity')
            outcomes, state = [4] * 8, legacy.copy()
            for h, index in enumerate(indices):
                if source == index:
                    break
                odor = bisect_right(_integer_cdf(laws[h][source]), int(integers[h + 1]))
                c.require(odor < 4 and laws[h][source, odor] > 0, 'positive categorical draw')
                outcomes[h] = odor
                state = legacy_update(state, positions[h], odor, kernel, np)
                odor_draws += 1
            c.require(outcomes == data['mc_outcomes'][i, sample].tolist(), 'complete absorbing hypothetical history')
            if outcomes[7] != 4:
                alive += 1
                if forward_check is not None:
                    forward_check(entry, sample, outcomes, state, positions[-1], kernel, data['costs'][i, sample])
        rows.append({'case_id': str(identity), 'root_grid_total_variation': tv,
            'work': {'allocated_draw_integers': draws * 9, 'source_draws': draws,
                'odor_draws': odor_draws, 'unused_odor_draws': draws * 8 - odor_draws,
                'legacy_updates': odor_draws, 'teacher_calls': alive}})
    return {'cases': rows, 'prefixes': n, 'teacher_recomputed': False, 'native_calls': 0}


def verify_forward(event, ordinal, entry, sample, history, state, position, kernel, saved, np):
    c.require(event['ordinal'] == ordinal and event['input_shape'] == [16, 105, 105]
              and event['symmetry_average'] is True, 'ordered original teacher forward')
    context = event['context']
    c.require(all(context[k] == v for k, v in entry.items()) and context['phase'] == 'counterfactual'
              and context['stream'] == entry['split'] and context['mode'] == 'mc'
              and context['horizon'] == 8 and context['sample'] == sample and context['history'] == history,
              'H8 endpoint to original-forward identity')
    expected = []
    for action in range(4):
        moved = list(position)
        moved[action // 2] += 2 * (action % 2) - 1
        x, y = moved
        joint = state.reshape(1, 53, 53) * kernel[:, 53 - x:106 - x, 53 - y:106 - y]
        expected.append([max(1e-10, float(joint[h].sum())) for h in range(4)])
    masses, values = np.asarray(event['branch_masses'], np.float32), np.asarray(event['values'], np.float32)
    c.require(masses.shape == (4, 4) and np.array_equal(masses, np.asarray(expected, np.float32)), 'independent endpoint teacher branch masses')
    c.require(values.shape == (16,) and np.isfinite(values).all(), 'finite recorded original forward values')
    products = masses * values.reshape(4, 4)
    reconstructed = np.float32(1.) + products.sum(-1, dtype=np.float32)
    error = abs(saved.astype(np.float64) - reconstructed.astype(np.float64))
    bound = 8 * np.finfo(np.float32).eps * (1 + abs(products.astype(np.float64)).sum(-1))
    c.require(np.isfinite(reconstructed).all() and (error <= bound).all(), 'saved teacher Q matches recorded float32 reduction')
    return float(error.max())


def verify_case_records(datasets, rows, summary, probabilities, np):
    c.require(len(rows) == len(c.roster()) and [r['identity'] for r in rows] == c.roster(), 'all attempted prefixes in roster order')
    teacher_calls = legacy_updates = total_retained = 0
    for split in ('train', 'dev'):
        data = datasets[split]
        retained = [r for r in rows if r['excluded'] is None and r['identity']['split'] == split]
        c.require([r['identity']['id'] for r in retained] == data['case_ids'].tolist(), 'complete split retained roster')
        total_retained += len(retained)
        for i, (row, probability) in enumerate(zip(retained, probabilities[split]['cases'], strict=True)):
            c.require(row['native_steps'] == 8 and set(row['array_sha256']) == set(data) - {'case_ids', 'regimes'}, 'case array descriptors')
            for name, pin in row['array_sha256'].items():
                c.require(hashlib.sha256(data[name][i].tobytes()).hexdigest() == pin, 'case array identity ' + name)
            compare_report(row['root_grid']['total_variation'], probability['root_grid_total_variation'])
            c.require(row['mc']['stream'] == split and row['mc']['seed'] == row['identity']['mc_seed']
                      and row['mc']['draws'] == c.CONFIG[split + '_draws']
                      and row['mc']['alive'] == int(data['alive'][i].sum()), 'bank support metadata')
            compare_report(row['mc']['work'], probability['work'])
            teacher_calls += probability['work']['teacher_calls']
            legacy_updates += probability['work']['legacy_updates']
    excluded = [r for r in rows if r['excluded'] is not None]
    c.require(all(r['excluded'] == 'found_during_observed_prefix' and type(r['native_steps']) is int
                  and 1 <= r['native_steps'] <= 8 for r in excluded), 'unreplaced prefix exclusions')
    counts = {'train': len(datasets['train']['case_ids']),
              **{'dev_' + r: int((datasets['dev']['regimes'] == r).sum()) for r in REGIMES}}
    c.require(summary['counts'] == counts and counts['train'] >= c.CONFIG['min_train']
              and all(counts['dev_' + r] >= c.CONFIG['min_dev_per_regime'] for r in REGIMES)
              and summary['cases'] == len(rows) and summary['retained'] == total_retained
              and summary['prefix_exclusions'] == len(excluded) and summary['replacement_cases'] == 0
              and summary['parent_data_array_decodes'] == summary['learner_calls'] == summary['native_continuation_steps'] == 0,
              'complete fresh dataset and no forbidden collection calls')
    steps = sum(r['native_steps'] for r in rows)
    expected = {'tensorflow_construction': 1, 'tensorflow_build': 1, 'tensorflow_load': 1,
        'tensorflow_value': teacher_calls, 'teacher_score': teacher_calls,
        'native_reset': len(rows), 'native_step': steps, 'actor_construction': len(rows), 'actor_update': steps,
        'analytic_score': steps + total_retained, 'feature_build': steps + total_retained,
        'backend_binding': total_retained, 'branch_view_construction': total_retained,
        'sampler_table': 2, 'sampler_lookup': steps + 8 * total_retained,
        'initial_source_law': len(rows), 'initial_normalization': len(rows), 'shadow_update': steps,
        'sampler_draw_validation': len(rows) + steps - len(excluded), 'committed_positions': total_retained,
        'root_grid_law': total_retained, 'mc_draw_allocation': total_retained,
        'mc_rollout': total_retained, 'legacy_mc_update': legacy_updates}
    c.require(set(summary['calls']) == set(expected), 'exact collection operation channels')
    for name, count in expected.items():
        row = summary['calls'][name]
        c.require(type(row['attempted']) is int and type(row['returned']) is int
                  and row['attempted'] == row['returned'] == count
                  and type(row['seconds']) in (float, int) and math.isfinite(row['seconds']) and row['seconds'] >= 0,
                  'independent collection count ' + name)
    return {'attempted_prefixes_checked': len(rows), 'retained_prefixes_checked': total_retained,
            'teacher_endpoint_records': teacher_calls, 'legacy_counterfactual_updates': legacy_updates}


def expected_work(cases, batches=1):
    return {'prefix_assimilation_calls': 9 * batches, 'prefix_assimilation_rows': 9 * cases,
        'transition_calls': 8 * batches, 'transition_rows': 8 * cases,
        'direct_token_calls': 0, 'direct_token_rows': 0, 'direct_prior_calls': 0, 'direct_prior_rows': 0,
        'observation_assimilation_calls': 0, 'observation_assimilation_rows': 0,
        'outcome_head_calls': 8 * batches, 'cost_head_calls': 8 * batches, 'aux_head_calls': 8 * batches,
        'readout_rows': 8 * cases, 'terminal_frozen_rows': 0}


def parameter_shapes():
    return {'assimilation.weight_ih': [84, 31], 'assimilation.weight_hh': [84, 28],
        'assimilation.bias_ih': [84], 'assimilation.bias_hh': [84],
        'outcome_head.weight': [5, 28], 'outcome_head.bias': [5],
        'cost_head.weight': [4, 28], 'cost_head.bias': [4],
        'aux_head.weight': [2, 28], 'aux_head.bias': [2],
        'transition.weight_ih': [84, 4], 'transition.weight_hh': [84, 28],
        'transition.bias_ih': [84], 'transition.bias_hh': [84]}


def state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        header = json.dumps([key, value.dtype.str, list(value.shape)], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'little'))
        digest.update(header)
        digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def verify_checkpoint(state, row, scale, np):
    shapes = {**parameter_shapes(), 'cost_scale': []}
    c.require(set(state) == set(shapes), 'complete action-recurrent checkpoint')
    for name, shape in shapes.items():
        value = state[name]
        c.require(isinstance(value, np.ndarray) and value.dtype == np.float32
                  and list(value.shape) == shape and np.isfinite(value).all(), 'finite checkpoint tensor ' + name)
    unused = {k: v for k, v in state.items() if k.startswith(('outcome_head.', 'aux_head.'))}
    c.require(float(state['cost_scale']) == scale and state_digest(state) == row['final_state_sha256']
              and state_digest(unused) == row['unused_heads_after_sha256'] == row['unused_heads_before_sha256'],
              'saved checkpoint and frozen heads agree with producer identities')


def verify_training(train, summary, normalization, draws, orders, events, fits, barrier, directory, receipt, np, *, check=lambda: None):
    """Independent TRAIN target moment, random schedules, masks and work geometry."""
    n, epochs, batch = len(train['case_ids']), c.CONFIG['epochs'], c.CONFIG['batch_size']
    bank = centered_bank(train, np)
    moment = _mean([float(v) ** 2 for v in bank.reshape(-1)])
    scaling = {'cost_scale': float(np.float32(math.sqrt(max(moment, c.CONFIG['variance_floor'])))),
               'second_moment_before_floor': moment, 'variance_floor': c.CONFIG['variance_floor']}
    compare_report(normalization, scaling)
    compare_report(summary['normalization'], scaling)
    c.require(normalization['cost_scale'] == scaling['cost_scale'] and normalization['teacher_units_divisor'] == 64.
              and normalization['train_cases'] == n and normalization['draws_per_case'] == 32
              and normalization['dev_decodes'] == 0, 'same TRAIN-only sampled-bank scale')
    expected_pairs = [(family, seed) for i, seed in enumerate(c.FIT_SEEDS)
                      for family in c.FAMILIES[i % 2:] + c.FAMILIES[:i % 2]]
    c.require(len(fits) == 6 and [(r['family'], r['seed']) for r in fits] == expected_pairs
              and fits == summary['fits'], 'all six final fits in registered order')
    c.require(len(draws) == 3 and [r['seed'] for r in draws] == list(c.FIT_SEEDS), 'three shared label permutations')
    permutations = {}
    for row in draws:
        check()
        seed = row['seed']
        p = np.stack([np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, i, 717]))).permutation(32) for i in range(n)]).astype(np.int64)
        pin = hashlib.sha256(p.tobytes()).hexdigest()
        c.require(row['permutations'] == p.tolist() and row['sha256'] == pin
                  and row['cycles'] == 3 and row['index_seed_suffix'] == 717, 'precommitted shared complete label bank permutations')
        permutations[seed] = p
    order_iter, event_iter = iter(orders), iter(events)
    updates_per_fit = epochs * math.ceil(n / batch)
    initial = {}
    shapes = parameter_shapes()
    effective = [name for name in shapes if not name.startswith(('outcome_head.', 'aux_head.'))]
    metadata = {'kind': 'action_recurrent', 'hidden_dim': 28, 'cost_scale': scaling['cost_scale'],
        'count': 8299, 'trainable_count': 8096,
        'parameters': {k: {'shape': shape, 'count': math.prod(shape)} for k, shape in shapes.items()},
        'zero_proposed_action_weight_parameters': 0, 'direct_descriptor': None}
    for row in fits:
        family, seed = row['family'], row['seed']
        p = permutations[seed]
        c.require(row['initial_state_sha256'] == initial.setdefault(seed, row['initial_state_sha256']), 'paired initialization attestation')
        c.require(row['model_kind'] == 'action_recurrent' and row['epochs'] == epochs and row['cases'] == n
                  and row['updates'] == updates_per_fit and row['exposures'] == epochs * n
                  and row['target_draw_uses'] == n * epochs * (32 if family == 'mean32' else 1)
                  and row['unique_teacher_labels'] == n * 32
                  and row['draw_permutation_sha256'] == hashlib.sha256(p.tobytes()).hexdigest()
                  and row['sampled_cycles'] == (3 if family == 'sampled' else None)
                  and row['training_horizons'] == [8] and row['executed_horizons'] == list(range(1, 9))
                  and row['evaluation_decodes_so_far'] == 0 and row['blind_input_keys'] == ['prefix', 'prefix_lengths', 'actions']
                  and row['privileged_targets_only'] == ['costs', 'alive'] and row['effective_parameter_names'] == effective,
                  'fixed fit recipe, exposure and parameter masks')
        compare_report(row['parameters'], metadata)
        c.require(row['work'] == expected_work(n * epochs, updates_per_fit), 'whole-fit recurrent geometry')
        c.require(row['checkpoint'] == c.desc(directory / f'{family}-{seed}.npz'), 'saved final checkpoint descriptor')
        for key in ('seconds', 'last_batch_loss', 'last_gradient_norm'):
            c.require(type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] >= 0, 'finite fit metadata')
        step = 0
        last = None
        for epoch in range(epochs):
            check()
            indices = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch, 919]))).permutation(n).astype(np.int64)
            chosen = p[:, epoch % 32]
            expected = {'family': family, 'seed': seed, 'epoch': epoch, 'indices': indices.tolist(),
                'draw_indices': chosen.tolist(), 'index_sha256': hashlib.sha256(indices.tobytes()).hexdigest(),
                'draw_sha256': hashlib.sha256(chosen.tobytes()).hexdigest()}
            c.require(next(order_iter, None) == expected, 'exact common case order and sampled target schedule')
            for start in range(0, n, batch):
                check()
                size = min(batch, n - start)
                pending = {'family': family, 'seed': seed, 'epoch': epoch, 'batch_start': start, 'cases': size, 'update': step}
                c.require(next(event_iter, None) == {'event': 'attempt', **pending}, 'durable original optimizer attempt')
                last = next(event_iter, None)
                c.require(last is not None and all(last[k] == v for k, v in pending.items())
                          and last['event'] == 'return' and last['work'] == expected_work(size), 'matching optimizer return and work')
                for key in ('loss', 'gradient_norm'):
                    c.require(type(last[key]) in (int, float) and math.isfinite(last[key]) and last[key] >= 0, 'finite saved update scalar')
                step += 1
        c.require(last['loss'] == row['last_batch_loss'] and last['gradient_norm'] == row['last_gradient_norm'], 'last update closes fit')
    c.require(next(order_iter, None) is None and next(event_iter, None) is None, 'no extra schedules or optimizer events')
    checkpoints = {f'{f}-{s}.npz': c.desc(directory / f'{f}-{s}.npz') for f in c.FAMILIES for s in c.FIT_SEEDS}
    c.require(barrier['fits_completed'] == 6 and barrier['checkpoint_files'] == checkpoints
              and barrier['fits_sha256'] == c.desc(directory / 'fits.jsonl')['sha256']
              and barrier['train_data'] == summary['train_data'] and barrier['dev_array_decodes'] == 0
              and receipt['started_ns'] <= barrier['created_ns'] <= receipt['finished_ns'], 'pre-DEV completed-fit barrier')
    return {'fits_checked': 6, 'optimizer_updates_checked': 6 * updates_per_fit,
            'case_orders_checked': 6 * epochs, 'shared_draw_permutations_checked': 3,
            'cost_scale': scaling['cost_scale']}


def admit(run):
    collection, fit = c.OUT / 'collection-01', c.OUT / 'fit-01'
    receipts, previous = {}, None
    for phase, directory, terminal_path in (
            ('collect', collection, c.OUT / 'collection-native-01.terminal.json'),
            ('fit', fit, c.OUT / 'fit-native-01.terminal.json')):
        receipt, terminal = c.closed(directory, terminal_path), c.read(terminal_path)
        c.require(receipt['version'] == c.VERSION and receipt['phase'] == phase
                  and receipt['plan_sha256'] == run.args.plan_sha256
                  and receipt['old_test_decodes'] == receipt['astra_calls'] == 0,
                  'same completed registration and allowed predecessor')
        c.require(previous is None or previous <= terminal['started_ns'], 'collection closes before fitting')
        previous = terminal['finished_ns']
        receipts[phase] = receipt
    c.require(previous <= run.launch['started_ns'], 'original fits close before independent audit')
    collection_files = {'started.json', 'parent-reference.json', 'runtime.json', 'setup.json', 'sensor-laws.npz',
        'sensor-laws.json', 'train.npz', 'dev.npz', 'cases.jsonl', 'commitments.jsonl', 'summary.json',
        'work.jsonl.gz', 'weights.jsonl.gz', 'forwards.jsonl.gz', 'native-draws.jsonl.gz',
        'shadow.jsonl.gz', 'source-laws.jsonl.gz', 'transitions.jsonl.gz'}
    fit_files = {'started.json', 'normalization.json', 'draw-orders.jsonl', 'training-orders.jsonl', 'updates.jsonl',
        'fits.jsonl', 'fit-barrier.json', 'predictions.npz', 'reports.json', 'summary.json'}
    fit_files.update(f'{f}-{s}.npz' for f in c.FAMILIES for s in c.FIT_SEEDS)
    c.require(set(receipts['collect']['files']) == collection_files and set(receipts['fit']['files']) == fit_files,
              'exact complete collection and fit inventories')
    c.require(not any(name in sys.modules for name in ('torch', 'tensorflow', 'jax', 'mlx')), 'no neural framework in audit')
    reference = c.parent_reference()
    c.require(c.read(collection / 'parent-reference.json') == reference and run.plan['parent_reference'] == reference,
              'published diagnostic lineage only, no old data')
    return collection, fit, receipts, reference


def run_audit(run):
    collection, directory, receipts, reference = admit(run)
    import numpy as np

    counts = {'array_decodes': 0, 'checkpoint_decodes': 0, 'model_calls': 0, 'optimizer_calls': 0,
              'teacher_calls': 0, 'native_calls': 0, 'parent_data_decodes': 0, 'old_test_decodes': 0}
    run.receipt['audit_counts'] = counts

    def arrays(folder, name, phase, checkpoint=False):
        run.check()
        c.require(c.desc(folder / name) == receipts[phase]['files'][name], 'authenticated saved array')
        run.receipt['pending'] = {'decode': name}
        with np.load(folder / name, allow_pickle=False) as source:
            value = {key: source[key] for key in source.files}
        counts['array_decodes'] += 1
        counts['checkpoint_decodes'] += int(checkpoint)
        run.receipt['pending'] = None
        return value

    data = {split: arrays(collection, split + '.npz', 'collect') for split in ('train', 'dev')}
    tables = arrays(collection, 'sensor-laws.npz', 'collect')
    predictions = arrays(directory, 'predictions.npz', 'fit')
    n = validate_data(data['train'], np, split='train')
    nd = validate_data(data['dev'], np, split='dev')
    c.require(not set(data['train']['case_ids']) & set(data['dev']['case_ids']), 'disjoint originating TRAIN and DEV')
    validate_predictions(data['dev'], predictions, np)
    verify_tables(tables, np, check=run.check)
    saved, summary = c.read(directory / 'reports.json'), c.read(directory / 'summary.json')
    acquisition = c.read(collection / 'summary.json')
    c.require(acquisition['calls'] == receipts['collect']['calls'] and acquisition['parent_reference'] == reference,
              'collection calls and diagnostic reference')
    for split in ('train', 'dev'):
        expected = {'path': split + '.npz', **receipts['collect']['files'][split + '.npz'],
                    'cases': len(data[split]['case_ids']), 'draws': c.CONFIG[split + '_draws']}
        c.require(acquisition['datasets'][split] == expected, 'saved split inventory')
    stream = iter(lines(collection / 'forwards.jsonl.gz', run.check))
    forward_count, maximum_error = 0, 0.

    def forward(entry, sample, history, state, position, kernel, costs):
        nonlocal forward_count, maximum_error
        run.check()
        event = next(stream, None)
        c.require(event is not None, 'all endpoint forward witnesses present')
        forward_count += 1
        error = verify_forward(event, forward_count, entry, sample, history, state, position, kernel, costs, np)
        maximum_error = max(maximum_error, error)

    probability = {split: verify_probabilities(data[split], tables, run.plan['roster'], np, split=split,
                       check=run.check, forward_check=forward) for split in ('train', 'dev')}
    c.require(next(stream, None) is None, 'no extra teacher forward witnesses')
    counts.update(verify_case_records(data, list(lines(collection / 'cases.jsonl', run.check)), acquisition, probability, np))
    c.require(forward_count == counts['teacher_endpoint_records'], 'one original forward per surviving endpoint')
    counts.update(verify_work(lines(collection / 'work.jsonl.gz', run.check), acquisition['calls'], check=run.check))
    normalization = c.read(directory / 'normalization.json')
    fits = list(lines(directory / 'fits.jsonl', run.check))
    counts.update(verify_training(data['train'], summary, normalization,
        list(lines(directory / 'draw-orders.jsonl', run.check)), lines(directory / 'training-orders.jsonl', run.check),
        lines(directory / 'updates.jsonl', run.check), fits, c.read(directory / 'fit-barrier.json'),
        directory, receipts['fit'], np, check=run.check))
    for row in fits:
        state = arrays(directory, f"{row['family']}-{row['seed']}.npz", 'fit', checkpoint=True)
        verify_checkpoint(state, row, normalization['cost_scale'], np)
    steps = 6 * c.CONFIG['epochs'] * math.ceil(n / c.CONFIG['batch_size'])
    expected_calls = {'train_array_decodes': 1, 'dev_array_decodes': 1, 'checkpoint_decodes': 0,
        'model_constructions': 12, 'fit_count': 6, 'optimizer_attempts': steps, 'optimizer_steps': steps,
        'training_rollouts': steps, 'training_prefix_exposures': 6 * n * c.CONFIG['epochs'],
        'target_draw_uses': 3 * n * c.CONFIG['epochs'] * 33, 'evaluation_rollouts': 6 * nd,
        'bootstrap_replicates': 2 * c.CONFIG['bootstrap_replicates'],
        'mean_target_aggregation_calls': 1, 'mean_target_aggregation_draws': 32 * n}
    c.require(summary['calls'] == receipts['fit']['calls'] == expected_calls, 'exact completed producer counters')
    c.require(normalization['mean_target_aggregation_draws'] == 32 * n
              and math.isfinite(normalization['mean_target_aggregation_seconds'])
              and normalization['mean_target_aggregation_seconds'] >= 0, 'single charged mean-label aggregation')
    c.require(summary['collection_receipt'] == c.desc(collection / 'receipt.json')
              and summary['train_data'] == receipts['collect']['files']['train.npz']
              and summary['dev_data'] == receipts['collect']['files']['dev.npz']
              and summary['data_cases'] == {'train': n, 'dev': nd}, 'exact producer data lineage')
    timing = summary['prediction_times']
    c.require(len(timing) == 6 and [(r['family'], r['seed']) for r in timing] == [(f, s) for f in c.FAMILIES for s in c.FIT_SEEDS], 'all six forecast costs')
    for row in timing:
        fit = next(f for f in fits if (f['family'], f['seed']) == (row['family'], row['seed']))
        c.require(row['cases'] == nd and row['work'] == expected_work(nd, nd)
                  and row['state_before'] == row['state_after'] == fit['final_state_sha256']
                  and type(row['seconds']) in (int, float) and math.isfinite(row['seconds']) and row['seconds'] >= 0,
                  'unchanged final checkpoint during complete inference')
    independent = analyze(data['dev'], predictions, np, check=run.check)
    c.require(set(saved) == set(independent), 'complete saved metric panels')
    compare_report(saved, independent)
    c.require(summary['status'] == independent['gate']['status'], 'unchanged prespecified outcome')
    for phase, folder in (('collect', collection), ('fit', directory)):
        for name, expected in receipts[phase]['files'].items():
            run.check()
            c.require(c.desc(folder / name) == expected, 'audit input bytes unchanged')
    counts.update(summary_rows_checked=12, case_rows_checked=6 * nd,
                  bootstrap_replicates_checked=2 * c.CONFIG['bootstrap_replicates'], forward_q_reductions_checked=forward_count)
    c.require(counts['array_decodes'] == 10 and counts['checkpoint_decodes'] == 6, 'four new datasets plus six new final checkpoints only')
    result = {'version': VERSION, 'agreement': True, 'technical_complete': False,
        'requires_original_supervisor_closure': True, 'counts': counts, 'report': independent, 'gate': independent['gate'],
        'probability_reconstruction': probability, 'maximum_forward_reduction_error': maximum_error,
        'inputs': {'collection_receipt': c.desc(collection / 'receipt.json'), 'fit_receipt': c.desc(directory / 'receipt.json')},
        'limitations': ['Teacher Q is linked to independently reconstructed branch masses and recorded forward values, not replayed.',
            'Native scalar sensor functions remain authenticated provenance, not native re-evaluation.',
            'Final tensor schemas and state hashes are decoded; initial paired equality remains a producer attestation checked against source and fabricated invariants.',
            'Schedules, counters and recorded updates are audited; gradients and optimizer updates are not replayed.',
            'Bootstrap is conditional on trained policies and fixed TRAIN banks, not full training uncertainty.',
            'H8 fixed-path teacher-cost supervision does not establish autonomous, normal-observation or architectural gains.']}
    run.check()
    c.write(run.out / 'audit.json', result)
    run.receipt['agreement'] = True


class Audit(c.Run):
    def body(self):
        run_audit(self)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'supervision', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    c.require(Path(sys.executable).absolute() == c.NUMERICAL, 'qualified numerical interpreter')
    run = Audit(args, 'audit')
    try:
        run.body()
        run.finish()
    except BaseException as error:
        run.finish(error)
        raise


def compare_report(saved, independent, path='report'):
    """Compare all independent fields; omit producer-specific roundoff metadata."""
    if isinstance(independent, dict):
        c.require(type(saved) is dict and set(independent) <= set(saved), path + ': fields')
        for key, value in independent.items():
            compare_report(saved[key], value, path + '.' + key)
    elif isinstance(independent, list):
        c.require(type(saved) is list and len(saved) == len(independent), path + ': complete list')
        for i, (left, right) in enumerate(zip(saved, independent, strict=True)):
            compare_report(left, right, f'{path}[{i}]')
    elif type(independent) is float:
        c.require(type(saved) in (int, float) and math.isfinite(saved) and math.isfinite(independent)
                  and abs(saved - independent) <= 1e-10 * (1 + abs(independent)), path + ': scalar agreement')
    else:
        c.require(type(saved) is type(independent) and saved == independent, path + ': exact identity')

def _mean(values):
    c.require(len(values) > 0, 'nonempty averaging denominator')
    result = math.fsum(float(v) / len(values) for v in values)
    c.require(math.isfinite(result), 'finite average')
    return result

def _path(position, actions):
    current, positions = list(map(int, position)), []
    for value in actions:
        action = int(value)
        c.require(0 <= action < 4, 'four movement actions')
        current[action // 2] += 2 * (action % 2) - 1
        c.require(all(0 < p < 52 for p in current), 'interior path supports all four decisions')
        positions.append(tuple(current))
    return positions

def _cdf_law(raw, np):
    """Independent scalar sequential CDF, exactly the declared finite grid."""
    values, cumulative, running = list(map(float, raw)), [], 0.
    c.require(all(math.isfinite(v) and v >= 0 for v in values)
              and abs(math.fsum(values) - 1) <= 1e-10, 'normalized raw categorical law')
    for value in values:
        running += value
        cumulative.append(running)
    c.require(math.isfinite(running) and running > 0, 'positive finite cumulative mass')
    result, previous = [], 0
    for value in cumulative:
        boundary = math.ceil((value / running) * 2**53)
        result.append((boundary - previous) / 2**53)
        previous = boundary
    c.require(previous == 2**53 and all(v >= 0 for v in result), 'complete integer law')
    return np.asarray(result, np.float64)

def _integer_cdf(law):
    result, total = [], 0
    for value in law:
        scaled = float(value) * 2**53
        c.require(math.isfinite(scaled) and 0 <= scaled <= 2**53 and scaled == int(scaled), 'exact finite-grid probability bins')
        total += int(scaled)
        result.append(total)
    c.require(total == 2**53, 'integer categorical bins sum to 2**53')
    return result

def lines(path, check):
    import gzip
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        for line in stream:
            check()
            c.require(line.endswith('\n'), 'complete durable JSON line')
            yield json.loads(line)

def verify_work(events, calls, *, check=lambda: None):
    stack, sequence, counts = [], 0, {name: 0 for name in calls}
    teacher_forwards = {}
    total = 0
    for event in events:
        check()
        total += 1
        c.require(event['channel'] in calls, 'declared original work channel')
        if event['event'] == 'attempt':
            sequence += 1
            c.require(event['id'] == sequence, 'monotone physical operation starts')
            if event['channel'] == 'tensorflow_value':
                c.require(stack and stack[-1]['channel'] == 'teacher_score', 'physical forward belongs to teacher annotation')
                parent = stack[-1]['id']
                teacher_forwards[parent] = teacher_forwards.get(parent, 0) + 1
            stack.append(event)
        else:
            c.require(event['event'] == 'return' and stack, 'closed nested work event')
            previous = stack.pop()
            c.require(all(event[k] == previous[k] for k in ('id', 'channel', 'context')), 'matching physical operation return')
            elapsed = event['seconds_including_nested_io']
            c.require(type(elapsed) in (float, int) and math.isfinite(elapsed) and elapsed >= 0, 'finite recorded operation time')
            if event['channel'] == 'teacher_score':
                c.require(teacher_forwards.pop(event['id'], 0) == 1, 'one physical forward per teacher score')
            counts[event['channel']] += 1
    c.require(not stack and not teacher_forwards and total == 2 * sequence, 'all work operations returned')
    c.require(all(counts[k] == value['attempted'] == value['returned'] for k, value in calls.items()), 'work journal reconciles all call counts')
    return {'work_events_checked': total, 'work_operations_checked': sequence}


if __name__ == '__main__':
    main()
