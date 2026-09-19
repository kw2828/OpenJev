"""Pure scalar calibration and proper-score arithmetic for saved card beliefs.

No environment, model, random generator, optimizer, or file operations. Callers
authenticate episode provenance and reconstruct the causal public target masks.
Current public overrides and action selection remain outside this module.
"""
from __future__ import annotations

import math

import numpy as np

VERSION = 'card-probability-calibration-v1'
MODES = ('baseline', 'temperature', 'hard')
BETA_BOUNDS = (0.05, 20.0)
BISECTION_ITERATIONS = 64
MAX_ORACLE_EVALUATIONS = 68
FIELDS = {'raw_probabilities', 'targets', 'target_mask', 'ages'}
WEIGHTING = 'queries within boundary; nonempty boundaries within episode; eligible episodes equally'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _raw(raw):
    _require(isinstance(raw, np.ndarray) and raw.ndim >= 2 and raw.shape[-2:] == (52, 13)
             and all(n > 0 for n in raw.shape) and raw.dtype in (np.dtype('float32'), np.dtype('float64')),
             'Expected nonempty float32/64 [...,52,13] probabilities')
    _require(np.isfinite(raw).all() and np.all(raw >= 0) and np.all(raw <= 1),
             'Raw probabilities must be finite and in [0,1]')
    result = raw.astype(np.float64, copy=True)
    mass = result.sum(axis=-1)
    _require(np.all(mass > 0) and np.all(np.abs(mass - 1) <= 1e-5),
             'Raw rows must have positive mass within 1e-5 of one')
    return result


def _beta(value):
    _require(type(value) in (int, float) and math.isfinite(value)
             and BETA_BOUNDS[0] <= value <= BETA_BOUNDS[1], 'beta must be finite in [0.05,20]')
    return float(value)


def _centered_logs(values):
    logs = np.full(values.shape, -np.inf, dtype=np.float64)
    np.log(values, out=logs, where=values > 0)
    return logs - logs.max(axis=-1, keepdims=True)


def _power(values, beta):
    scaled = beta * _centered_logs(values)
    result = np.exp(scaled)
    result /= result.sum(axis=-1, keepdims=True)
    return result


def transform_probabilities(raw, mode='baseline', beta=1.0):
    """Return owned float64 beliefs, before the unchanged C public overrides.

    Baseline and temperature beta=1 are exact value-preserving copies. The C
    picker performs its existing normalization afterward. Other temperatures
    use log(saved probabilities), not unavailable pre-softmax logits. Hard
    beliefs use the smallest rank index at an exact maximum; they are not
    claimed to be calibrated probabilities. Zeros are never epsilon-smoothed.
    """
    _require(type(mode) is str and mode in MODES, 'Unknown confidence mode')
    beta = _beta(beta)
    _require(mode == 'temperature' or beta == 1.0, 'Only temperature mode accepts a nonidentity beta')
    values = _raw(raw)
    if mode == 'hard':
        indices = values.argmax(axis=-1)
        values.fill(0.0)
        np.put_along_axis(values, indices[..., None], 1.0, axis=-1)
    elif mode == 'temperature' and beta != 1.0:
        values = _power(values, beta)
    values.flags.writeable = False
    return values


def _episodes(episodes):
    _require(type(episodes) in (list, tuple) and len(episodes) > 0, 'Nonempty explicit episode sequence required')
    validated = []
    for episode in episodes:
        _require(type(episode) is dict and set(episode) == FIELDS, 'Exact four episode fields required')
        raw = _raw(episode['raw_probabilities'])
        _require(raw.ndim == 3 and 1 <= len(raw) <= 104, 'Episode must have one to 104 actual boundaries')
        fields = {}
        for name, dtype in (('targets', np.int64), ('target_mask', np.bool_), ('ages', np.int32)):
            value = episode[name]
            _require(isinstance(value, np.ndarray) and value.shape == raw.shape[:2]
                     and value.dtype == np.dtype(dtype), f'Invalid {name} shape or dtype')
            fields[name] = value.copy()
        targets, mask, ages = (fields[name] for name in ('targets', 'target_mask', 'ages'))
        _require(np.all(targets[~mask] == -1) and np.all(ages[~mask] == -1),
                 'Unqueried targets and ages must be -1')
        _require(np.all((targets[mask] >= 0) & (targets[mask] < 13)), 'Queried target rank out of range')
        clock = np.arange(len(raw))[:, None]
        _require(np.all((ages >= 1)[mask]) and np.all((ages <= clock)[mask]),
                 'Queried age must precede this boundary; startup has no hidden-history targets')
        validated.append({'raw_probabilities': raw, **fields})
    return validated


def _sum(values):
    return math.fsum(float(value) for value in values)


def _weighted_queries(episodes):
    eligible = sum(bool(ep['target_mask'].any()) for ep in episodes)
    _require(eligible > 0, 'No eligible public targets for scalar fitting')
    logs, targets, weights = [], [], []
    boundaries, cards = 0, 0
    for ep in episodes:
        mask = ep['target_mask']
        per_boundary = mask.sum(axis=1)
        nonempty = int(np.count_nonzero(per_boundary))
        if nonempty == 0:
            continue
        selected = ep['raw_probabilities'][mask]
        y = ep['targets'][mask]
        _require(np.all(selected[np.arange(len(y)), y] > 0),
                 'Zero probability at a calibration target: infinite NLL for every finite beta; no floor allowed')
        logs.append(_centered_logs(selected))
        targets.append(y)
        per_query = 1.0 / (eligible * nonempty * np.maximum(per_boundary, 1))
        weights.append(np.broadcast_to(per_query[:, None], mask.shape)[mask])
        boundaries += nonempty
        cards += len(y)
    return (np.concatenate(logs), np.concatenate(targets), np.concatenate(weights),
            {'episodes': len(episodes), 'eligible_episodes': eligible,
             'query_boundaries': boundaries, 'query_cards': cards})


def fit_temperature(episodes):
    """Fit one beta=1/T by bounded convex weighted NLL, with no model updates.

    Two endpoint derivatives and identity are evaluated first. A strictly
    interior root receives exactly 64 bisection iterations and one final
    midpoint evaluation. Thus at most 68 loss/derivative passes occur per fit.
    Boundary optima need three passes. Exact loss ties prefer identity. The
    finite-precision candidate is replaced by identity if its measured NLL is
    worse. This safeguard and all bounds are fixed, not outcome-selected.
    """
    validated = _episodes(episodes)
    logs, target, weights, counts = _weighted_queries(validated)
    row = np.arange(len(target))
    selected_logs = logs[row, target]
    finite_logs = np.where(np.isfinite(logs), logs, 0.0)
    trace = []

    def oracle(beta, point):
        exponent = np.exp(beta * logs)
        mass = exponent.sum(axis=1)
        probabilities = exponent / mass[:, None]
        nll = _sum(weights * (np.log(mass) - beta * selected_logs))
        derivative = _sum(weights * ((probabilities * finite_logs).sum(axis=1) - selected_logs))
        _require(math.isfinite(nll) and math.isfinite(derivative), 'Nonfinite scalar-fit objective/derivative')
        result = {'point': point, 'beta': float(beta), 'nll': nll, 'derivative': derivative}
        trace.append(result)
        return result

    low, high = BETA_BOUNDS
    left, right = oracle(low, 'lower'), oracle(high, 'upper')
    identity = oracle(1.0, 'identity')
    if left['derivative'] >= 0:
        candidate = left
    elif right['derivative'] <= 0:
        candidate = right
    else:
        for iteration in range(BISECTION_ITERATIONS):
            middle = (low + high) / 2.0
            result = oracle(middle, f'bisection_{iteration:02d}')
            if result['derivative'] > 0:
                high = middle
            elif result['derivative'] < 0:
                low = middle
            else:
                low = high = middle
        candidate = oracle((low + high) / 2.0, 'final')
    chosen = candidate if candidate['nll'] < identity['nll'] else identity
    beta = chosen['beta']
    _require(len(trace) <= MAX_ORACLE_EVALUATIONS, 'Scalar-fit work exceeded frozen pass cap')
    return {'version': VERSION, 'status': 'fitted', 'beta': beta, 'temperature': 1.0 / beta,
            'beta_bounds': list(BETA_BOUNDS), 'bisection_iterations': BISECTION_ITERATIONS,
            'oracle_evaluations': len(trace), 'baseline_nll': identity['nll'],
            'calibrated_nll': chosen['nll'], 'chosen_derivative': chosen['derivative'],
            'at_bound': beta in BETA_BOUNDS, 'counts': counts, 'weighting': WEIGHTING,
            'scope': 'scalar fit on supplied calibration targets only; caller authenticates provenance and split',
            'new_model_calls': 0, 'new_memory_weight_updates': 0, 'trace': trace}


def _episode_scores(raw, targets, selected, mode, beta):
    probabilities = transform_probabilities(raw, mode=mode, beta=beta).copy()
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    cards = selected.sum(axis=1)
    active = np.flatnonzero(cards)
    nll, brier, accuracy, infinite_queries, underflow_queries = [], [], [], 0, 0
    for boundary in active:
        mask = selected[boundary]
        p, y = probabilities[boundary, mask], targets[boundary, mask]
        true = p[np.arange(len(y)), y]
        if mode == 'hard':
            losses = np.where(true == 0, math.inf, 0.0)
        else:
            logs = _centered_logs(raw[boundary, mask])
            scaled = logs * beta
            losses = np.log(np.exp(scaled).sum(axis=1)) - scaled[np.arange(len(y)), y]
            underflow_queries += int(np.count_nonzero((true == 0) & np.isfinite(losses)))
        zeros = int(np.count_nonzero(np.isinf(losses)))
        infinite_queries += zeros
        nll.append(math.inf if zeros else _sum(losses) / len(y))
        error = p.copy()
        error[np.arange(len(y)), y] -= 1
        brier.append(_sum(np.square(error).sum(axis=1)) / len(y))
        accuracy.append(float(np.count_nonzero(p.argmax(axis=1) == y)) / len(y))
    count = len(active)
    return {'query_boundaries': count, 'query_cards': int(cards.sum()),
            'nll': None if not count or infinite_queries else _sum(nll) / count,
            'nll_is_infinite': bool(infinite_queries), 'infinite_nll_queries': infinite_queries,
            'underflowed_target_probabilities': underflow_queries,
            'brier': _sum(brier) / count if count else None,
            'accuracy': _sum(accuracy) / count if count else None}


def score_episodes(episodes, mode='baseline', beta=1.0):
    """Proper scores on caller-supplied identical prefixes and public masks.

    Hard wrong beliefs have mathematically infinite NLL: JSON-safe nll=None
    and nll_is_infinite=True, never smoothing or averaging only finite errors.
    Empty strata also have nll=None but flag False and eligible_episodes=0.
    Finite-beta NLL uses the stable log-power distribution, even if its exposed
    floating probability underflows to zero; those targets are counted apart.
    Multiclass Brier is the sum of 13 squared probability errors.
    """
    _require(type(mode) is str and mode in MODES, 'Unknown confidence mode')
    beta = _beta(beta)
    _require(mode == 'temperature' or beta == 1.0, 'Only temperature mode accepts a nonidentity beta')
    validated, rows = _episodes(episodes), []
    for index, ep in enumerate(validated):
        row = {'episode_index': index}
        for group, mask in (('all', ep['target_mask']),
                            ('age_gt32', ep['target_mask'] & (ep['ages'] > 32))):
            row[group] = _episode_scores(ep['raw_probabilities'], ep['targets'], mask, mode, beta)
        rows.append(row)
    result = {'version': VERSION, 'scope': 'saved-prefix arithmetic only; no authenticity or native-utility claim',
              'mode': mode, 'beta': beta, 'weighting': WEIGHTING, 'episodes': len(rows), 'per_episode': rows,
              'brier_definition': 'sum over 13 classes of squared probability error'}
    for group in ('all', 'age_gt32'):
        selected = [row[group] for row in rows if row[group]['query_boundaries']]
        infinite = any(row['nll_is_infinite'] for row in selected)
        result[group] = {'eligible_episodes': len(selected),
                         'query_boundaries': sum(row['query_boundaries'] for row in selected),
                         'query_cards': sum(row['query_cards'] for row in selected),
                         'infinite_nll_queries': sum(row['infinite_nll_queries'] for row in selected),
                         'underflowed_target_probabilities': sum(row['underflowed_target_probabilities'] for row in selected),
                         'nll_is_infinite': infinite,
                         'nll': None if infinite or not selected else _sum(row['nll'] for row in selected) / len(selected),
                         **{key: _sum(row[key] for row in selected) / len(selected) if selected else None
                            for key in ('brier', 'accuracy')}}
    return result
