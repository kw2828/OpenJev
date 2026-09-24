"""Scaled expected-count learning for a finite action/observation filter.

T[a,next,current] is column stochastic. O[odor,next] is shared by the
uniform-prior reset and ordinary post-transition events. Hazard h[a,next]
acts after transition; found=4 terminates the sequence without an emission.
There is no hidden-state input, RNG, Torch dependency, file IO or fitting loop.

Forward/backward scaling avoids multiplying whole-sequence likelihoods. A
terminal row's hidden-state posterior describes the destination just before
absorption, solely for expected counts. It is not a live state after found.
Filtered rows use observations through that row; smoothed rows may use later
TRAIN observations. Padding has no likelihood or count contribution.

MLE permits exact boundary probabilities. Exactly unexposed columns/outcomes
retain their old probabilities. MAP adds a positive pseudocount to every T/O
category and each hazard outcome: equivalently a coefficient ``pseudocount``
on sum(log(T))+sum(log(O))+sum(log(h))+sum(log(1-h)). This is concentration
1+pseudocount, not an empirical prior fit. No probability clipping, pseudocount
insertion in MLE, or conversion to finite Torch logits is performed here.

All arithmetic is NumPy float64. Validation accepts normalized columns within
1e-12 but does not repair them. Positive-product underflow and impossible
observations fail explicitly. Posterior normalization removes only ordinary
floating reduction error after checking mass agreement. Work records count
structural operations, not validation, allocations or complete FLOPs.
"""
from __future__ import annotations

import math

import numpy as np

VERSION = 'finite-expected-count-v1'
TOLERANCE = 1e-12
COUNT_KEYS = ('transition', 'emission', 'survive', 'found', 'reset')
WORK_KEYS = ('sequences', 'reset_events', 'action_events', 'ordinary_events', 'found_events',
             'forward_matvecs', 'backward_matvecs', 'transition_posterior_matrices',
             'state_posterior_rows', 'check_calls')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, dtype, shape, name):
    _require(isinstance(value, np.ndarray) and value.dtype == dtype
             and value.shape == shape, name + ' exact array shape/dtype')
    _require(np.isfinite(value).all(), name + ' finite including padding')
    return value.copy()


def _parameters(transition, emission, hazard):
    _require(isinstance(transition, np.ndarray) and transition.ndim == 3,
             'transition [actions,next,current]')
    actions, states, current = transition.shape
    _require(actions > 0 and states > 0 and current == states, 'positive action/state dimensions')
    t = _array(transition, np.dtype('float64'), (actions, states, states), 'transition')
    o = _array(emission, np.dtype('float64'), (4, states), 'emission')
    h = _array(hazard, np.dtype('float64'), (actions, states), 'hazard')
    for value, name in ((t, 'transition'), (o, 'emission'), (h, 'hazard')):
        _require(((value >= 0) & (value <= 1)).all(), name + ' probability bounds')
    _require(np.all(np.abs(t.sum(axis=1) - 1) <= TOLERANCE), 'transition columns sum to one')
    _require(np.all(np.abs(o.sum(axis=0) - 1) <= TOLERANCE), 'emission columns sum to one')
    return t, o, h


def _sequence(actions, observations, action_count):
    _require(isinstance(actions, np.ndarray) and actions.ndim == 1, 'one-dimensional actions')
    a = _array(actions, np.dtype('int64'), actions.shape, 'actions')
    y = _array(observations, np.dtype('int64'), (len(a) + 1,), 'observations')
    _require(((a >= 0) & (a < action_count)).all(), 'declared actions only')
    _require(0 <= y[0] < 4 and ((y >= 0) & (y <= 4)).all(), 'ordinary reset and valid events')
    _require(not np.any(y[:-1] == 4), 'no event after first found')
    return a, y


def _empty_counts(action_count, states):
    return {'transition': np.zeros((action_count, states, states), dtype=np.float64),
            'emission': np.zeros((4, states), dtype=np.float64),
            'survive': np.zeros((action_count, states), dtype=np.float64),
            'found': np.zeros((action_count, states), dtype=np.float64),
            'reset': np.zeros(states, dtype=np.float64)}


def _product(first, second):
    result = first * second
    _require(np.isfinite(result).all(), 'finite probability product')
    _require(not np.any((first > 0) & (second > 0) & (result == 0)),
             'positive probability product underflow')
    return result


def _posterior(value, name):
    total = float(value.sum(dtype=np.float64))
    _require(np.isfinite(value).all() and (value >= 0).all()
             and math.isfinite(total) and total > 0
             and abs(total - 1) <= 1e-10, name + ' normalized posterior mass')
    return value / total


def _forward_backward(parameters, actions, observations, check):
    t, o, h = parameters
    action_count, states, _ = t.shape
    count = len(actions)
    work = dict.fromkeys(WORK_KEYS, 0)

    def checked():
        work['check_calls'] += 1
        check()

    checked()
    scales = np.empty(count + 1, dtype=np.float64)
    filtered = np.empty((count + 1, states), dtype=np.float64)
    backward = np.ones_like(filtered)
    kernels = np.empty((count, states, states), dtype=np.float64)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        reset_mass = _product(o[observations[0]], 1.0 / states)
        scales[0] = reset_mass.sum(dtype=np.float64)
        _require(0 < scales[0] <= 1 + TOLERANCE, 'positive reset evidence; no clipping')
        filtered[0] = reset_mass / scales[0]
        for step, (action, observation) in enumerate(zip(actions, observations[1:], strict=True)):
            checked()
            weights = h[action] if observation == 4 else _product(1 - h[action], o[observation])
            kernels[step] = _product(t[action], weights[:, None])
            mass = kernels[step] @ filtered[step]
            scales[step + 1] = mass.sum(dtype=np.float64)
            _require(np.isfinite(mass).all() and 0 < scales[step + 1] <= 1 + TOLERANCE,
                     'positive observed-event evidence; no clipping')
            filtered[step + 1] = mass / scales[step + 1]
            work['forward_matvecs'] += 1
        for step in range(count - 1, -1, -1):
            checked()
            backward[step] = (kernels[step].T @ backward[step + 1]) / scales[step + 1]
            _require(np.isfinite(backward[step]).all() and (backward[step] >= 0).all(),
                     'finite scaled backward message')
            work['backward_matvecs'] += 1
        smoothed = np.empty_like(filtered)
        for step in range(count + 1):
            checked()
            smoothed[step] = _posterior(_product(filtered[step], backward[step]), 'smoothed state')
        transitions = np.empty_like(kernels)
        counts = _empty_counts(action_count, states)
        counts['reset'][:] = smoothed[0]
        counts['emission'][observations[0]] += smoothed[0]
        for step, (action, observation) in enumerate(zip(actions, observations[1:], strict=True)):
            checked()
            joint = _product(_product(kernels[step], filtered[step][None]), backward[step + 1][:, None])
            transitions[step] = _posterior(joint / scales[step + 1], 'transition posterior')
            _require(np.allclose(transitions[step].sum(axis=0), smoothed[step], rtol=1e-10, atol=1e-12)
                     and np.allclose(transitions[step].sum(axis=1), smoothed[step + 1], rtol=1e-10, atol=1e-12),
                     'smoothed transition/state marginals agree')
            counts['transition'][action] += transitions[step]
            outcome = 'found' if observation == 4 else 'survive'
            counts[outcome][action] += smoothed[step + 1]
            if observation != 4:
                counts['emission'][observation] += smoothed[step + 1]
        likelihood = math.fsum(math.log(float(value)) for value in scales)
    _require(math.isfinite(likelihood), 'finite scaled log likelihood')
    work.update({'sequences': 1, 'reset_events': 1, 'action_events': count,
                 'ordinary_events': int(np.count_nonzero(observations[1:] < 4)),
                 'found_events': int(observations[-1] == 4),
                 'transition_posterior_matrices': count, 'state_posterior_rows': count + 1})
    checked()
    return {'version': VERSION, 'log_likelihood': likelihood, 'scales': scales,
            'filtered': filtered, 'smoothed': smoothed, 'transition_posteriors': transitions,
            'counts': counts, 'terminal_found': bool(observations[-1] == 4), 'work': work}


def forward_backward(transition, emission, hazard, actions, observations, *, check=lambda: None):
    """One reset plus K action events; terminal hidden destination is diagnostic.

    Probability arrays must be float64, labels/actions int64. K may be zero.
    All returned arrays own their storage and never alias caller arrays.
    """
    _require(callable(check), 'callable deadline/check hook')
    parameters = _parameters(transition, emission, hazard)
    a, y = _sequence(actions, observations, len(parameters[0]))
    return _forward_backward(parameters, a, y, check)


def expected_counts(transition, emission, hazard, actions, observations, lengths, *, check=lambda: None):
    """Aggregate equally weighted sequence likelihood/counts, not event means.

    actions[N,K], observations[N,K+1], lengths[N] count reset plus valid action
    events. Every inactive action/observation must be -1. Empty batches return
    zero counts/log likelihood. Only active evidence appears in scales.
    """
    _require(callable(check), 'callable deadline/check hook')
    parameters = _parameters(transition, emission, hazard)
    _require(isinstance(actions, np.ndarray) and actions.ndim == 2, 'batched actions [N,K]')
    batch, horizon = actions.shape
    a = _array(actions, np.dtype('int64'), (batch, horizon), 'batched actions')
    y = _array(observations, np.dtype('int64'), (batch, horizon + 1), 'batched observations')
    sizes = _array(lengths, np.dtype('int64'), (batch,), 'lengths')
    _require(((sizes >= 1) & (sizes <= horizon + 1)).all(), 'lengths count reset and action events')
    # Preflight the entire batch before any smoothing or user callback.
    for row, length in enumerate(sizes):
        _require((a[row, length - 1:] == -1).all() and (y[row, length:] == -1).all(), 'exact -1 token padding')
        _sequence(a[row, :length - 1], y[row, :length], len(parameters[0]))
    counts = _empty_counts(len(parameters[0]), parameters[0].shape[1])
    work = dict.fromkeys(WORK_KEYS, 0)
    log_likelihoods = np.empty(batch, dtype=np.float64)
    scales = np.zeros((batch, horizon + 1), dtype=np.float64)
    for row, length in enumerate(sizes):
        result = _forward_backward(parameters, a[row, :length - 1], y[row, :length], check)
        log_likelihoods[row] = result['log_likelihood']
        scales[row, :length] = result['scales']
        for name in COUNT_KEYS:
            counts[name] += result['counts'][name]
        for name in WORK_KEYS:
            work[name] += result['work'][name]
    total = math.fsum(map(float, log_likelihoods))
    _require(math.isfinite(total) and all(np.isfinite(value).all() for value in counts.values()),
             'finite aggregate likelihood and expected counts')
    return {'version': VERSION, 'counts': counts, 'sequence_log_likelihood': log_likelihoods,
            'log_likelihood': total, 'scales': scales, 'work': work}


def _counts(values, parameters):
    t, _, _ = parameters
    template = _empty_counts(len(t), t.shape[1])
    _require(type(values) is dict and set(values) == set(COUNT_KEYS), 'exact expected-count roster')
    result = {name: _array(values[name], np.dtype('float64'), shape.shape, name + ' counts')
              for name, shape in template.items()}
    _require(all((value >= 0).all() for value in result.values()), 'nonnegative expected counts')
    _require(np.allclose(result['transition'].sum(axis=2), result['survive'] + result['found'],
                        rtol=1e-12, atol=1e-10), 'transition destination counts match hazard exposures')
    _require(np.allclose(result['emission'].sum(axis=0), result['reset'] + result['survive'].sum(axis=0),
                        rtol=1e-12, atol=1e-10), 'shared emission counts include reset and ordinary events once')
    return result


def _update(transition, emission, hazard, values, pseudocount):
    old = _parameters(transition, emission, hazard)
    raw = _counts(values, old)
    t, o, h = (value.copy() for value in old)
    t_raw = raw['transition'].sum(axis=1)
    o_raw = raw['emission'].sum(axis=0)
    h_raw = raw['survive'] + raw['found']
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        t_counts = raw['transition'] + pseudocount
        o_counts = raw['emission'] + pseudocount
        survive, found = raw['survive'] + pseudocount, raw['found'] + pseudocount
        t_total, o_total, h_total = t_counts.sum(axis=1), o_counts.sum(axis=0), survive + found
        _require(all(np.isfinite(value).all() for value in (t_total, o_total, h_total)), 'finite M-step exposure')
        for action in range(len(t)):
            active = t_total[action] > 0
            t[action][:, active] = t_counts[action][:, active] / t_total[action, active][None]
        active_o, active_h = o_total > 0, h_total > 0
        o[:, active_o] = o_counts[:, active_o] / o_total[active_o][None]
        h[active_h] = found[active_h] / h_total[active_h]
    _parameters(t, o, h)
    return {'version': VERSION, 'transition': t, 'emission': o, 'hazard': h,
            'raw_counts': raw, 'pseudocount': float(pseudocount),
            'zero_exposure': {'transition': t_raw == 0, 'emission': o_raw == 0, 'hazard': h_raw == 0},
            'work': {'transition_columns_updated': int(np.count_nonzero(t_total > 0)),
                     'emission_columns_updated': int(np.count_nonzero(active_o)),
                     'hazard_entries_updated': int(np.count_nonzero(active_h))}}


def mle_update(transition, emission, hazard, counts):
    """Plain MLE, including boundaries; retain old values at exact zero exposure."""
    return _update(transition, emission, hazard, counts, 0.0)


def map_update(transition, emission, hazard, counts, *, pseudocount):
    """MAP update with explicit positive symmetric prior on each probability."""
    _require(isinstance(pseudocount, (int, float, np.integer, np.floating))
             and not isinstance(pseudocount, (bool, np.bool_))
             and math.isfinite(float(pseudocount)) and pseudocount > 0, 'positive finite explicit pseudocount')
    return _update(transition, emission, hazard, counts, float(pseudocount))


def tokens_from_prefix(prefix, lengths):
    """Strict frozen all-attempt public grammar, without beliefs or hidden state.

    Input float32[N,9,31] has reset marker9 and ordinary odor4:8 in row0;
    later rows carry one action0:4 and one event4:9. Only first found may
    shorten the eight-action prefix. Inactive rows must be entirely zero.
    Returned integer arrays use -1 padding and lengths include reset.
    """
    _require(isinstance(prefix, np.ndarray) and prefix.ndim == 3, 'public prefix [N,9,31]')
    batch = len(prefix)
    p = _array(prefix, np.dtype('float32'), (batch, 9, 31), 'public prefix')
    sizes = _array(lengths, np.dtype('int64'), (batch,), 'public lengths')
    _require(((sizes >= 2) & (sizes <= 9)).all(), 'reset plus one to eight public action events')
    active = np.arange(9)[None] < sizes[:, None]
    rows = p[active]
    _require((p[~active] == 0).all() and ((rows == 0) | (rows == 1)).all()
             and (rows[:, 10:] == 0).all() and (rows[:, 4:9].sum(axis=1) == 1).all(),
             'binary public labels and zero padding only')
    _require((p[:, 0, :4] == 0).all() and (p[:, 0, 8] == 0).all() and (p[:, 0, 9] == 1).all(),
             'ordinary reset odor without action or hazard')
    later = p[:, 1:][active[:, 1:]]
    _require((later[:, :4].sum(axis=1) == 1).all() and (later[:, 9] == 0).all(),
             'one action per later event without reset')
    found = p[:, :, 8] == 1
    last = np.arange(9)[None] == sizes[:, None] - 1
    _require(not np.any(found & ~last) and np.all(found.any(axis=1) | (sizes == 9)),
             'only first found may terminate a shorter public prefix')
    actions = np.full((batch, 8), -1, dtype=np.int64)
    observations = np.full((batch, 9), -1, dtype=np.int64)
    action_mask = active[:, 1:]
    actions[action_mask] = p[:, 1:, :4].argmax(axis=-1)[action_mask]
    observations[active] = p[:, :, 4:9].argmax(axis=-1)[active]
    return {'actions': actions, 'observations': observations, 'lengths': sizes, 'event_mask': active}
