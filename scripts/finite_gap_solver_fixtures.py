"""Deterministic, model-independent qualification fixtures and scalar oracle.

No random stream, model, optimizer, file, or empirical input is used. The 16
factorial fixtures cross four state geometries, two generating heads, and exact
or deliberately misspecified centered targets. Only exact targets supply a
known optimum. The two additional fixtures test absent support and tiny mass.
Their certificates do not establish coefficient recovery or unique solutions.

The scalar oracle computes blind MSE plus observed MSE with the same complete
4*N*H denominator for each route. Zero rows are retained. It does not use a
Gram matrix, the producer objective, or either solver implementation. Its
Frank-Wolfe bound is a floating-point numerical check, not an interval proof.
"""
from __future__ import annotations

import math

import numpy as np

VERSION = 'finite-gap-solver-fixtures-v1'
GEOMETRIES = ('basis', 'correlated95', 'correlated999', 'rank_deficient')
OPTIMA = ('interior', 'boundary')
TARGET_TYPES = ('exact', 'misspecified')
NAMES = tuple(f'{geometry}__{optimum}__{target}'
              for geometry in GEOMETRIES for optimum in OPTIMA for target in TARGET_TYPES) + ('zero_support', 'small_mass')
SIMPLEX_TOLERANCE = 1e-12
GAP_TOLERANCE = 1e-8
NONINCREASE_TOLERANCE = 1e-12
CERTIFICATE_KEYS = {'objective', 'gradient', 'fw_gap', 'simplex_violation',
                    'minimum_probability', 'maximum_probability', 'passed'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _head(kind):
    p = np.full((4, 8), .125 if kind == 'interior' else 0., np.float64)
    p[np.arange(8) % 4, np.arange(8)] = .625 if kind == 'interior' else 1.
    return p


def _states(geometry):
    base = np.eye(8, dtype=np.float64)
    if geometry.startswith('correlated'):
        epsilon = .05 if geometry == 'correlated95' else .001
        base *= epsilon
        base[:, 0] += 1. - epsilon
    elif geometry == 'rank_deficient':
        base[:] = 0.
        for state in range(8):
            base[state, 2 * (state // 2):2 * (state // 2) + 2] = .5
    indices = np.concatenate((np.arange(8), (np.arange(8) + 3) % 8))
    xb = np.repeat(base[indices, None, :], 2, axis=1)
    xb[:, 1] *= np.asarray([0., .001, .5] * 6, np.float64)[:16, None]
    xo = xb[(np.arange(16) + 5) % 16].copy()
    return xb, xo


def _misspecify(target, route):
    result = target.copy()
    pattern = np.asarray([.03, -.03, .01, -.01], np.float64)
    for case in range(16):
        for horizon in range(2):
            sign = 1. if (case // 4 + horizon + route) % 2 == 0 else -1.
            result[case, horizon] += sign * np.roll(pattern, case + 2 * horizon + route)
    return result


def fixtures():
    """Return 18 fresh, owned float64 fixtures in the registered fixed order.

    Exact targets use Y=X@(.25-Pstar).T, so Pstar attains nonnegative objective
    zero, including singular designs. Misspecification adds fixed row-dependent
    centered errors even on zero-state rows and has no asserted known optimum.
    """
    result = []
    for geometry in GEOMETRIES:
        for optimum in OPTIMA:
            for target_type in TARGET_TYPES:
                xb, xo = _states(geometry)
                p = _head(optimum)
                yb, yo = xb @ (.25 - p).T, xo @ (.25 - p).T
                if target_type == 'misspecified':
                    yb, yo = _misspecify(yb, 0), _misspecify(yo, 1)
                result.append({'name': f'{geometry}__{optimum}__{target_type}',
                    'xb': xb, 'xo': xo, 'yb': yb, 'yo': yo,
                    'initial': np.full((4, 8), .25, np.float64),
                    'known_optimum': p.copy() if target_type == 'exact' else None,
                    'description': f'{geometry} states; {optimum} generating head; {target_type} targets. '
                        + ('Known zero-loss feasible optimum, not necessarily unique.' if target_type == 'exact'
                           else 'Row-dependent centered perturbations; the generating head is not claimed optimal.')})
    zeros = np.zeros((16, 2, 8), np.float64)
    constant = np.broadcast_to(np.asarray([.125, -.125, 0., 0.], np.float64), (16, 2, 4)).copy()
    result.append({'name': 'zero_support', 'xb': zeros.copy(), 'xo': zeros.copy(),
        'yb': constant.copy(), 'yo': -constant, 'initial': np.full((4, 8), .25, np.float64),
        'known_optimum': np.full((4, 8), .25, np.float64),
        'description': 'All states zero, nonzero centered targets. Objective is the constant 1/64; every feasible head is optimal, no coefficient identification.'})
    xb, xo = _states('basis')
    xb, xo = xb * 1e-8, xo * 1e-8
    p = _head('interior')
    result.append({'name': 'small_mass', 'xb': xb, 'xo': xo,
        'yb': xb @ (.25 - p).T, 'yo': xo @ (.25 - p).T,
        'initial': np.full((4, 8), .25, np.float64), 'known_optimum': p.copy(),
        'description': 'Basis states scaled 1e-8 with exact targets. Absolute-tolerance scale coverage only: initial absolute gap can pass without coefficient recovery.'})
    _require(tuple(row['name'] for row in result) == NAMES, 'fixed 18-fixture roster')
    return result


def _array(value, name, shape=None):
    result = np.asarray(value)
    _require(result.dtype == np.float64 and (shape is None or result.shape == shape)
             and np.isfinite(result).all(), name + ': finite float64 with required shape')
    return result


def scalar_oracle(xb, xo, yb, yo, probabilities):
    """Independent raw residual objective, gradient and signed simplex gap."""
    xb = _array(xb, 'blind states')
    _require(xb.ndim == 3 and xb.shape[0] > 0 and xb.shape[1] > 0 and xb.shape[2] == 8,
             'nonempty N by H by eight states')
    n, h, _ = xb.shape
    xo = _array(xo, 'observed states', xb.shape)
    yb, yo = (_array(y, name, (n, h, 4)) for y, name in ((yb, 'blind targets'), (yo, 'observed targets')))
    p = _array(probabilities, 'probabilities', (4, 8))
    for x in (xb, xo):
        _require(np.all(x >= 0) and np.all(x.sum(-1) <= 1 + SIMPLEX_TOLERANCE), 'states are subprobabilities')
    for y in (yb, yo):
        _require(np.all(np.abs(y.sum(-1)) <= SIMPLEX_TOLERANCE), 'targets are centered costs')
    terms = []
    products = [[[] for _ in range(8)] for _ in range(4)]
    denominator = 4 * n * h
    for x, y in ((xb, yb), (xo, yo)):
        for i in range(n):
            for t in range(h):
                for a in range(4):
                    predicted = math.fsum((.25 - float(p[a, s])) * float(x[i, t, s]) for s in range(8))
                    error = predicted - float(y[i, t, a])
                    terms.append(error * error)
                    for s in range(8):
                        products[a][s].append(error * float(x[i, t, s]))
    objective = math.fsum(terms) / denominator
    gradient = np.asarray([[-2 * math.fsum(products[a][s]) / denominator for s in range(8)] for a in range(4)], np.float64)
    gap = math.fsum(float(p[a, s]) * float(gradient[a, s]) for a in range(4) for s in range(8)) \
        - math.fsum(min(float(gradient[a, s]) for a in range(4)) for s in range(8))
    violation = max(max(0., -float(p.min())),
                    max(abs(math.fsum(float(p[a, s]) for a in range(4)) - 1.) for s in range(8)))
    _require(math.isfinite(objective) and np.isfinite(gradient).all() and math.isfinite(gap), 'finite scalar oracle arithmetic')
    passed = violation <= SIMPLEX_TOLERANCE and -SIMPLEX_TOLERANCE <= gap <= GAP_TOLERANCE
    return {'objective': objective, 'gradient': gradient, 'fw_gap': gap,
            'simplex_violation': violation, 'minimum_probability': float(p.min()),
            'maximum_probability': float(p.max()), 'passed': bool(passed)}


def check_certificate(xb, xo, yb, yo, probabilities, claimed, *, require_pass=False):
    """Check an honest success or failure report; never replace it with success.

    Numeric agreement uses absolute 1e-12 plus relative 1e-10 tolerance. Acceptance
    thresholds are applied to the independent scalar values, not to the claim.
    A failed optimizer can still have a valid certificate, and vice versa; this
    function does not infer or modify the optimizer's separate exit status.
    """
    _require(type(claimed) is dict and set(claimed) == CERTIFICATE_KEYS, 'exact claimed certificate fields')
    _require(type(require_pass) is bool, 'boolean require_pass')
    independent = scalar_oracle(xb, xo, yb, yo, probabilities)
    declared_gradient = np.asarray(claimed['gradient'])
    _require(declared_gradient.shape == (4, 8) and declared_gradient.dtype.kind == 'f'
             and np.isfinite(declared_gradient).all(), 'finite claimed gradient with exact shape')
    for a in range(4):
        for s in range(8):
            _require(math.isclose(float(declared_gradient[a, s]), float(independent['gradient'][a, s]),
                                  abs_tol=1e-12, rel_tol=1e-10), 'independent gradient disagreement')
    for name in CERTIFICATE_KEYS - {'gradient', 'passed'}:
        value = claimed[name]
        _require(type(value) in (float, int) and math.isfinite(value)
                 and math.isclose(float(value), independent[name], abs_tol=1e-12, rel_tol=1e-10),
                 'independent certificate disagreement: ' + name)
    _require(type(claimed['passed']) is bool and claimed['passed'] == independent['passed'], 'independent certificate status disagreement')
    _require(not require_pass or independent['passed'], 'independent certificate did not pass')
    return independent


def check_nonincrease(initial_certificate, final_certificate):
    """Return the fixed objective comparison without treating it as solver success."""
    initial, final = initial_certificate['objective'], final_certificate['objective']
    _require(type(initial) in (float, int) and type(final) in (float, int)
             and math.isfinite(initial) and math.isfinite(final), 'finite independently checked objectives')
    return bool(final <= initial + NONINCREASE_TOLERANCE)
