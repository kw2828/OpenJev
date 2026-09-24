"""Passive float64 geometry of the eight-state factorized learned model.

No data, hidden states, world constants, fitting or model construction enter
this module. T transports columns; S=diag(1-h)T transports surviving mass.
The action average is uniform over four actions at each step, hence all 4**h
action sequences at horizon h, not an empirical policy. Its cost/state Grams propagate in the FULL eight
dimensional space. The zero-sum basis is used only when measuring them.
Consequently hazard-dependent mass changes can become cost differences even
when the initial contrast Gram is zero.

These are model-geometry diagnostics, not causal estimates of information
loss, observability of the true process, or evidence of task improvement.
Ranks use declared absolute thresholds. Raw signed eigenvalues, including
small negative roundoff, are retained. Nothing is clipped or aligned to a
true-world state basis. Work records count named numerical operations, not
validation operations, scalar FLOPs or wall-clock cost.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

VERSION = 'finite-transport-geometry-v1'
HORIZONS = (0, 1, 2, 4, 8)
SINGULAR_THRESHOLDS = (('1e-8', 1e-8), ('1e-6', 1e-6), ('1e-4', 1e-4), ('1e-2', 1e-2))
EIGEN_THRESHOLDS = (('1e-12', 1e-12), ('1e-10', 1e-10), ('1e-8', 1e-8), ('1e-6', 1e-6))
TOLERANCE = 1e-12
PARAMETER_SHAPES = {
    'transition_logits': (4, 8, 8),
    'emission_logits': (4, 8),
    'hazard_logits': (4, 8),
    'cost_logits': (4, 8),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, shape, name):
    _require(isinstance(value, np.ndarray) and value.dtype == np.dtype('float64')
             and value.shape == shape, name + ': exact float64 array and shape required')
    _require(bool(np.isfinite(value).all()), name + ': finite entries required')
    return np.array(value, dtype=np.float64, copy=True, order='C')


def _finite(value, name):
    _require(bool(np.isfinite(value).all()), name + ': nonfinite numerical result')
    return value


def _softmax(value, axis):
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        exponential = np.exp(value - np.max(value, axis=axis, keepdims=True))
        result = exponential / exponential.sum(axis=axis, keepdims=True)
    _finite(result, 'softmax')
    _require(bool(((result > 0) & (result < 1)).all()),
             'softmax saturation or underflow; no probability clipping')
    return result


def probability_fields(parameters):
    """Reconstruct owned T/O/h/C/S arrays from the exact four parameter names.

    The parameterization and strict numerical probability guards match the
    learned factorized model. Extreme finite logits can fail these guards;
    they are never repaired or silently replaced. C can round to an endpoint
    of [-.75,.25] even when the underlying head probabilities are interior.
    """
    _require(isinstance(parameters, Mapping) and set(parameters) == set(PARAMETER_SHAPES),
             'exact four-parameter roster required')
    values = {name: _array(parameters[name], shape, name)
              for name, shape in PARAMETER_SHAPES.items()}
    try:
        transition = _softmax(values['transition_logits'], 1)
        emission = _softmax(values['emission_logits'], 0)
        cost_probabilities = _softmax(values['cost_logits'], 0)
        logits = values['hazard_logits']
        hazard = np.empty_like(logits)
        positive = logits >= 0
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            hazard[positive] = 1 / (1 + np.exp(-logits[positive]))
            exponentials = np.exp(logits[~positive])
            hazard[~positive] = exponentials / (1 + exponentials)
        _finite(hazard, 'hazard')
        _require(bool(((hazard > 0) & (hazard < 1)).all()),
                 'hazard saturation or underflow; no probability clipping')
        costs = .25 - cost_probabilities
        surviving = (1 - hazard[:, :, None]) * transition
        _finite(surviving, 'surviving transport')
        _require(bool((surviving > 0).all()), 'surviving transport product underflow')
        _require(bool(np.max(np.abs(costs.sum(axis=0))) <= TOLERANCE), 'centered head')
    except FloatingPointError as error:
        raise ValueError('probability reconstruction numerical failure') from error
    return {'transition': transition, 'emission': emission, 'hazard': hazard,
            'costs': costs, 'surviving': surviving}


def zero_sum_basis():
    """Deterministic Helmert columns: k positive entries then one negative."""
    basis = np.zeros((8, 7), dtype=np.float64)
    for column in range(7):
        k = column + 1
        denominator = np.sqrt(float(k * (k + 1)))
        basis[:k, column] = 1 / denominator
        basis[k, column] = -k / denominator
    return basis


def _basis(value):
    basis = zero_sum_basis() if value is None else _array(value, (8, 7), 'basis')
    _require(bool(np.max(np.abs(basis.sum(axis=0))) <= TOLERANCE), 'zero-sum basis')
    _require(bool(np.max(np.abs(basis.T @ basis - np.eye(7))) <= TOLERANCE),
             'orthonormal basis')
    return basis


def _ranks(values, thresholds):
    return {name: int(np.count_nonzero(values > threshold)) for name, threshold in thresholds}


def transport_grams(surviving, costs, *, basis=None, check=lambda: None):
    """Full-space action-word Grams, measured at fixed horizons 0/1/2/4/8.

    Accepts nonnegative column-substochastic S[4,8,8], including boundary
    identity/permutation/zero matrices, and any finite action-centered C[4,8].
    Cost G0=C.T@C; state G0=I. Each step is sum_a S_a.T@G@S_a / 4.
    No intermediate centering, projection, state renormalization or hazard
    cancellation occurs. eigvalsh uses the lower triangle of the measured
    matrix after checking symmetry; no symmetrization or eigenvalue repair.
    """
    _require(callable(check), 'callable check required')
    work = {'check_calls': 0, 'cost_initial_gram_matrix_products': 0,
            'gram_recurrence_steps': 0, 'gram_channel_updates': 0,
            'gram_action_conjugations': 0, 'gram_recurrence_matrix_products': 0,
            'gram_measurements': 0, 'gram_projection_matrix_products': 0,
            'eigenvalue_decompositions': 0}

    def tick():
        work['check_calls'] += 1
        check()

    tick()
    matrices = _array(surviving, (4, 8, 8), 'surviving')
    readout = _array(costs, (4, 8), 'costs')
    q = _basis(basis)
    _require(bool((matrices >= 0).all()) and
             bool((matrices.sum(axis=1) <= 1 + TOLERANCE).all()),
             'nonnegative column-substochastic surviving transport')
    _require(bool(np.max(np.abs(readout.sum(axis=0))) <= TOLERANCE), 'action-centered costs')
    result = {'horizons': list(HORIZONS), 'cost': [], 'state': [], 'work': work}
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            current = {'cost': readout.T @ readout, 'state': np.eye(8, dtype=np.float64)}
            work['cost_initial_gram_matrix_products'] = 1
            for horizon in range(max(HORIZONS) + 1):
                if horizon in HORIZONS:
                    for name, gram in current.items():
                        tick()
                        _finite(gram, 'Gram')
                        projected = q.T @ gram @ q
                        _finite(projected, 'projected Gram')
                        _require(bool(np.max(np.abs(projected - projected.T)) <= TOLERANCE),
                                 'projected Gram symmetry')
                        eigenvalues = np.linalg.eigvalsh(projected)
                        _finite(eigenvalues, 'Gram eigenvalues')
                        _require(float(eigenvalues[0]) >= -TOLERANCE,
                                 'materially negative Gram eigenvalue')
                        result[name].append({
                            'horizon': horizon, 'gram': gram.tolist(),
                            'projected_gram': projected.tolist(),
                            'eigenvalues': eigenvalues.tolist(),
                            'trace': float(np.trace(projected)),
                            'ranks': _ranks(eigenvalues, EIGEN_THRESHOLDS),
                        })
                        work['gram_measurements'] += 1
                        work['gram_projection_matrix_products'] += 2
                        work['eigenvalue_decompositions'] += 1
                if horizon == max(HORIZONS):
                    break
                tick()
                following = {}
                for name, gram in current.items():
                    accumulated = np.zeros((8, 8), dtype=np.float64)
                    for action in range(4):
                        accumulated += matrices[action].T @ gram @ matrices[action]
                        work['gram_action_conjugations'] += 1
                        work['gram_recurrence_matrix_products'] += 2
                    following[name] = _finite(.25 * accumulated, 'propagated Gram')
                    work['gram_channel_updates'] += 1
                current = following
                work['gram_recurrence_steps'] += 1
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError('transport Gram numerical failure') from error
    return result


def _summary(values):
    _finite(values, 'summary values')
    return {'minimum': float(np.min(values)), 'mean': float(np.mean(values)),
            'maximum': float(np.max(values))}


def _column_geometry(matrix, basis):
    pairs = [(left, right) for left in range(8) for right in range(left + 1, 8)]
    distances = np.array([np.linalg.norm(matrix[:, left] - matrix[:, right])
                          for left, right in pairs], dtype=np.float64)
    singular_values = np.linalg.svd(matrix @ basis, compute_uv=False)
    _finite(singular_values, 'column contrast singular values')
    return {'pairs': [list(pair) for pair in pairs], 'column_distances': distances.tolist(),
            'column_distance_summary': _summary(distances),
            'contrast_singular_values': singular_values.tolist(),
            'contrast_ranks': _ranks(singular_values, SINGULAR_THRESHOLDS),
            'structural_rank_upper_bound': 3}


def analyze_parameters(parameters, check=lambda: None):
    """Return JSON-finite diagnostics for one exact four-array parameter state.

    Dobrushin delta is half the largest L1 distance between transition columns.
    Column entropies use natural logarithms. OQ and CQ have algebraic rank at
    most three because their four output rows sum to zero. Transport spectra
    use Q.T@T@Q, not the full T spectrum. Gram traces/eigenvalues measure
    unnormalized survival-weighted action-word propagation, not conditional
    state retention. Hazard values and S are retained to expose this limit.
    """
    _require(callable(check), 'callable check required')
    calls = 0

    def tick():
        nonlocal calls
        calls += 1
        check()

    tick()
    fields = probability_fields(parameters)
    q = zero_sum_basis()
    transitions = []
    uniform = np.full(8, 1 / 8, dtype=np.float64)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for action, matrix in enumerate(fields['transition']):
                tick()
                distances = [float(np.abs(matrix[:, left] - matrix[:, right]).sum())
                             for left in range(8) for right in range(left + 1, 8)]
                entropies = -(matrix * np.log(matrix)).sum(axis=0)
                singular_values = np.linalg.svd(q.T @ matrix @ q, compute_uv=False)
                _finite(singular_values, 'transition contrast singular values')
                transitions.append({
                    'action': action, 'dobrushin_delta': .5 * max(distances),
                    'column_entropies': entropies.tolist(),
                    'column_entropy_summary': _summary(entropies),
                    'contrast_singular_values': singular_values.tolist(),
                    'contrast_ranks': _ranks(singular_values, SINGULAR_THRESHOLDS),
                    'row_stochastic_deviation_max': float(np.max(np.abs(matrix.sum(axis=1) - 1))),
                    'uniform_image_l1': float(np.abs(matrix @ uniform - uniform).sum()),
                })
            tick()
            emission = _column_geometry(fields['emission'], q)
            tick()
            cost_head = _column_geometry(fields['costs'], q)
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError('parameter geometry numerical failure') from error
    grams = transport_grams(fields['surviving'], fields['costs'], basis=q, check=tick)
    work = {
        'parameter_arrays': 4, 'parameter_entries': 352,
        'softmax_calls': 3, 'softmax_columns': 48,
        'sigmoid_calls': 1, 'sigmoid_entries': 32,
        'surviving_transport_matrices': 4, 'surviving_product_entries': 256,
        'transition_matrices_analyzed': 4, 'transition_column_pair_l1_distances': 112,
        'column_entropies': 32, 'column_pair_euclidean_distances': 56,
        'spectral_matrix_products': 10, 'singular_value_decompositions': 6,
        'uniform_matrix_vector_products': 4,
        **{name: value for name, value in grams['work'].items() if name != 'check_calls'},
        'check_calls': calls,
    }
    return {
        'version': VERSION, 'dtype': 'float64', 'state_dimension': 8,
        'action_count': 4, 'basis': q.tolist(),
        'singular_rank_thresholds': dict(SINGULAR_THRESHOLDS),
        'eigenvalue_rank_thresholds': dict(EIGEN_THRESHOLDS),
        'negative_eigenvalue_tolerance': TOLERANCE,
        'fields': {name: value.tolist() for name, value in fields.items()},
        'transition': transitions, 'emission': emission, 'cost_head': cost_head,
        'grams': grams, 'work': work,
        'interpretation': (
            'Passive learned-model geometry only. Uniform action-word averaging is not a learned policy. '
            'Survival Grams combine transport and hazard mass loss without conditioning. '
            'No causal information-loss estimate, true-world alignment, fitted probe, '
            'scientific success gate or architecture-improvement claim.'),
    }
