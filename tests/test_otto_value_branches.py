"""Synthetic branch arithmetic only: no weights, simulator or fitted model."""

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from openjev.research.otto_value_branches import (
    EPSILON,
    explicit_scores,
    min_linear_scores,
    rl_branches,
    select_action,
)


def kernel():
    x, y = np.indices((107, 107))
    result = np.stack([1 + ((x * (h + 1) + y * (5 - h)) % 17) for h in range(4)]).astype(np.float64)
    result /= result.sum(axis=0)
    result[:, 53, 53] = 0
    return result


def sparse_belief():
    result = np.zeros((53, 53), dtype=np.float64)
    for position, mass in (((0, 0), .07), ((52, 52), .13), ((20, 17), .21), ((26, 26), .29)):
        result[position] = mass
    return result


def constant_kernel(values):
    result = np.stack([np.full((107, 107), v, dtype=np.float64) for v in values])
    result[:, 53, 53] = 0
    return result


def scalar_branches(belief, position, likelihood):
    """Independent cell-loop oracle, with coordinate arithmetic rather than crops."""
    raw = np.zeros((16, 105, 105))
    normalized = np.zeros_like(raw)
    masses, weights, successors = np.zeros((4, 4)), np.zeros((4, 4)), []
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for action, (dx, dy) in enumerate(directions):
        x = max(0, min(52, position[0] + dx))
        y = max(0, min(52, position[1] + dy))
        for hit in range(4):
            cells = []
            for i in range(53):
                for j in range(53):
                    value = float(belief[i, j]) * float(likelihood[hit, 53 + i - x, 53 + j - y])
                    raw[action * 4 + hit, 52 + i - x, 52 + j - y] = value
                    cells.append(value)
            mass = math.fsum(cells)
            masses[action, hit] = mass
            weight = max(1e-10, mass)
            weights[action, hit] = weight
            normalized[action * 4 + hit] = raw[action * 4 + hit] / weight
            successors.append((x, y))
    return raw, normalized, masses, weights, np.array(successors)


@pytest.mark.parametrize("position", [(26, 26), (0, 0), (0, 52), (52, 0), (52, 52),
                                     (0, 26), (52, 26), (26, 0), (26, 52)])
def test_all_sixteen_branches_match_independent_scalar_oracle(position):
    belief, likelihood = sparse_belief(), kernel()
    actual = rl_branches(belief, position, likelihood, [3, 1])
    expected = scalar_branches(belief, position, likelihood)
    for left, right in zip((actual.centered_u, actual.centered_z, actual.raw_masses,
                            actual.weights, actual.successors), expected, strict=True):
        np.testing.assert_allclose(left, right, atol=1e-12, rtol=1e-12)
    assert actual.eligible_actions == (1, 3)
    assert np.all(actual.centered_u[:, 52, 52] == 0)


def test_floor_boundaries_preserve_zero_subnormalized_inputs_and_weight_sum():
    belief = np.zeros((53, 53))
    belief[10, 10] = 1
    actual = rl_branches(belief, (26, 26), constant_kernel([0, .5e-10, 1e-10, 2e-10]), [0, 1, 2, 3])
    np.testing.assert_array_equal(actual.raw_masses, np.tile([0, .5e-10, 1e-10, 2e-10], (4, 1)))
    np.testing.assert_array_equal(actual.weights, np.tile([1e-10, 1e-10, 1e-10, 2e-10], (4, 1)))
    np.testing.assert_array_equal(actual.centered_z.sum(axis=(1, 2)), np.tile([0, .5, 1, 1], 4))
    assert np.all(actual.weights.sum(axis=1) == 5e-10)
    assert np.count_nonzero(actual.centered_z[::4]) == 0


def test_corner_blocked_actions_are_stay_and_observe_not_removed():
    branches = rl_branches(sparse_belief(), (0, 0), kernel(), [1, 3])
    np.testing.assert_array_equal(branches.successors[[0, 4, 8, 12]], [[0, 0], [1, 0], [0, 0], [0, 1]])
    np.testing.assert_array_equal(branches.centered_u[:4], branches.centered_u[8:12])
    assert not np.array_equal(branches.raw_masses[0], branches.raw_masses[1])


def test_generic_explicit_score_matches_independent_scalar_value_loop():
    belief, likelihood = sparse_belief(), kernel()
    branches = rl_branches(belief, (0, 52), likelihood, [3, 2])
    raw, normalized, _, weights, successors = scalar_branches(belief, (0, 52), likelihood)
    alpha = (np.arange(105 * 105).reshape(105, 105) % 23) / 100.
    calls = []

    def value(inputs, positions, known_kernel):
        calls.append(inputs.shape)
        assert known_kernel is branches.kernel and not inputs.flags.writeable
        return .4 + (inputs * alpha).sum(axis=(1, 2)) + positions[:, 0] / 1000.

    actual = explicit_scores(branches, value)
    expected = []
    for a in range(4):
        terms = []
        for h in range(4):
            i = 4 * a + h
            cost = .4 + math.fsum(float(p) * float(c) for p, c in zip(normalized[i].flat, alpha.flat, strict=True))
            cost += int(successors[i, 0]) / 1000.
            terms.append(float(weights[a, h]) * cost)
        expected.append(1 + math.fsum(terms))
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert calls == [(16, 105, 105)]
    assert not actual.flags.writeable
    assert raw.shape == branches.centered_u.shape


def coefficients():
    x, y = np.indices((105, 105))
    common = np.stack([((x + 2 * y) % 13) / 8., ((3 * x + y) % 17) / 9. - .4])
    return np.stack([common * (a + 1) + a / 7. for a in range(4)])


@pytest.mark.parametrize("scale", [0., .5e-10, 1e-10, 2e-10, .25, 1.])
def test_float64_explicit_fused_and_scalar_values_agree_with_exact_actions(scale):
    branches = rl_branches(sparse_belief() * scale, (26, 26), kernel(), [0, 1, 2, 3])
    bank = coefficients()
    before = bank.copy()
    explicit = min_linear_scores(branches, bank)
    fused = min_linear_scores(branches, bank, route="fused")
    expected = []
    for a in range(4):
        branches_values = []
        for h in range(4):
            values = [math.fsum(float(p) * float(c) for p, c in zip(
                branches.centered_u[4 * a + h].flat, template.flat, strict=True)) for template in bank[a]]
            branches_values.append(min(values))
        expected.append(1 + math.fsum(branches_values))
    np.testing.assert_allclose(explicit, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(fused, expected, atol=1e-12, rtol=1e-12)
    assert select_action(explicit, branches.eligible_actions) == select_action(fused, branches.eligible_actions)
    np.testing.assert_array_equal(bank, before)


def test_shared_bank_matches_equal_position_banks_and_generic_explicit_route():
    branches = rl_branches(sparse_belief(), (26, 26), kernel(), [1, 3])
    bank = coefficients()[0]
    four_banks = np.repeat(bank[None], 4, axis=0)
    a = min_linear_scores(branches, bank)
    b = min_linear_scores(branches, four_banks)
    c = explicit_scores(branches, lambda z, _x, _k: np.min(
        (z[:, None] * bank[None]).sum(axis=(2, 3)), axis=1))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)


def test_dense_asymmetric_belief_scalar_branches_and_float64_routes():
    x, y = np.indices((53, 53))
    belief = (1 + (13 * x + 7 * y) % 43).astype(np.float64)
    belief /= belief.sum()
    likelihood = kernel()
    branches = rl_branches(belief, (26, 26), likelihood, [0, 1, 2, 3])
    raw, normalized, masses, weights, successors = scalar_branches(belief, (26, 26), likelihood)
    for actual, expected in zip((branches.centered_u, branches.centered_z, branches.raw_masses,
                                 branches.weights, branches.successors),
                                (raw, normalized, masses, weights, successors), strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    bank = coefficients()
    expected = np.array([1 + math.fsum(min(math.fsum(float(p) * float(c)
        for p, c in zip(raw[4 * a + h].flat, alpha.flat, strict=True)) for alpha in bank[a])
        for h in range(4)) for a in range(4)])
    explicit = min_linear_scores(branches, bank)
    fused = min_linear_scores(branches, bank, route="fused")
    np.testing.assert_allclose(explicit, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(fused, expected, atol=1e-12, rtol=1e-12)
    assert select_action(explicit, branches.eligible_actions) == select_action(fused, branches.eligible_actions)
    assert select_action(explicit, branches.eligible_actions) == select_action(expected, branches.eligible_actions)


def test_zero_input_and_immediate_finding_do_not_repair_biased_value():
    belief = np.zeros((53, 53))
    belief[27, 26] = 1  # Action1 reaches the sole source hypothesis.
    branches = rl_branches(belief, (26, 26), constant_kernel([.25] * 4), [0, 1, 2, 3])
    assert not branches.centered_u[4:8].any()
    assert np.all(branches.weights[1] == EPSILON)
    generic = explicit_scores(branches, lambda z, _x, _k: np.ones(len(z)))
    homogeneous = min_linear_scores(branches, np.ones((1, 105, 105)))
    assert generic[1] == 1 + 4 * EPSILON
    assert homogeneous[1] == 1
    assert select_action(homogeneous, branches.eligible_actions) == 1
    # C(delta_origin)=7 is allowed; this is not a terminal-state-value API.
    alpha = np.ones((1, 105, 105))
    alpha[:, 52, 52] = 7
    delta_origin = np.zeros((105, 105))
    delta_origin[52, 52] = 1
    assert (alpha[0] * delta_origin).sum() == 7
    np.testing.assert_array_equal(min_linear_scores(branches, alpha), homogeneous)


def test_zero_belief_generic_bias_and_homogeneous_zero_are_distinct():
    branches = rl_branches(np.zeros((53, 53)), (26, 26), kernel(), [0, 1, 2, 3])
    generic = explicit_scores(branches, lambda z, _x, _k: np.full(len(z), 3.))
    homogeneous = min_linear_scores(branches, coefficients(), route="fused")
    np.testing.assert_array_equal(homogeneous, np.ones(4))
    np.testing.assert_array_equal(generic, np.full(4, 1 + 12 * EPSILON))


def test_biased_value_is_a_counterexample_to_raw_branch_fusion():
    branches = rl_branches(sparse_belief(), (26, 26), kernel(), [0, 1, 2, 3])
    explicit = explicit_scores(branches, lambda z, _x, _k: 2. + z.sum(axis=(1, 2)))
    wrongly_fused = 1 + (2. + branches.centered_u.sum(axis=(1, 2))).reshape(4, 4).sum(axis=1)
    assert np.max(np.abs(explicit - wrongly_fused)) > 1.


def test_strict_ties_respect_dtype_and_numeric_id_not_eligible_input_order():
    values = np.array([EPSILON, 0., EPSILON / 2, 4.])
    assert select_action(values, [2, 1, 0]) == 1  # Equality to epsilon is excluded.
    assert select_action(np.zeros(4), [3, 1]) == 1
    assert select_action(np.array([0., 4., 3., 2.]), [1, 3]) == 3
    before = values.copy()
    select_action(values, [1, 2])
    np.testing.assert_array_equal(values, before)


def test_named_float32_scores_follow_cast_then_reduce_and_can_change_tie():
    branches = rl_branches(sparse_belief(), (26, 26), kernel(), [0, 1, 2, 3])
    seen = []

    def value(z, _x, _k):
        seen.append(z.dtype)
        return np.sum(z, axis=(1, 2), dtype=np.float32)

    actual = explicit_scores(branches, value, arithmetic="float32")
    expected = np.float32(1) + np.sum(branches.weights.astype(np.float32)
        * np.sum(branches.centered_z.astype(np.float32), axis=(1, 2), dtype=np.float32).reshape(4, 4),
        axis=1, dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32 and seen == [np.dtype("float32")]
    # A declared dtype change can create a new exact tie. It is not qualified
    # deployment parity and must not be hidden behind a widened tolerance.
    costs = np.array([1 + 2e-8, 1., 3., 4.])
    assert select_action(costs, [0, 1]) == 1
    assert select_action(costs.astype(np.float32), [0, 1]) == 0


def test_float32_explicit_fused_underflow_difference_is_not_repaired():
    belief = np.zeros((53, 53))
    belief[10, 10] = 1e-35
    branches = rl_branches(belief, (26, 26), constant_kernel([.5e-10] * 4), [0, 1, 2, 3])
    bank = np.full((1, 105, 105), 1e38)
    explicit = min_linear_scores(branches, bank, arithmetic="float32")
    fused = min_linear_scores(branches, bank, route="fused", arithmetic="float32")
    assert np.all(np.isfinite(explicit)) and np.all(np.isfinite(fused))
    # Range-stress fixture: raw u casts to zero; normalized z does not.
    assert not branches.centered_u.astype(np.float32).any()
    assert branches.centered_z.astype(np.float32).any()
    assert np.all(fused == 1)
    assert np.all(explicit > fused)


def test_all_arrays_are_owned_bytes_backed_immutable_and_cannot_be_rebound():
    belief, likelihood = sparse_belief(), kernel()
    old_belief, old_kernel = belief.copy(), likelihood.copy()
    branches = rl_branches(belief, [26, 26], likelihood, [0, 1, 2, 3])
    np.testing.assert_array_equal(belief, old_belief)
    np.testing.assert_array_equal(likelihood, old_kernel)
    for array in (branches.centered_u, branches.centered_z, branches.raw_masses, branches.weights,
                  branches.successors, branches.kernel):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        branches.eligible_actions = (0,)
    likelihood[:] = 0
    belief[:] = 0
    np.testing.assert_array_equal(branches.kernel, old_kernel)
    assert branches.raw_masses.sum() > 0


@pytest.mark.parametrize("position", [[True, 0], [-1, 0], [53, 0], [0., 1], [0], [[0], [1]]])
def test_invalid_positions_rejected(position):
    with pytest.raises(ValueError):
        rl_branches(sparse_belief(), position, kernel(), [0])


@pytest.mark.parametrize("eligible", [[], [0, 0], [True], [-1], [4], [1.], "01"])
def test_invalid_action_metadata_rejected(eligible):
    error = TypeError if isinstance(eligible, str) else ValueError
    with pytest.raises(error):
        rl_branches(sparse_belief(), (26, 26), kernel(), eligible)
    with pytest.raises(error):
        select_action(np.zeros(4), eligible)


@pytest.mark.parametrize("problem", ["dtype", "shape", "negative", "nan", "mass"])
def test_invalid_beliefs_rejected(problem):
    b = sparse_belief()
    if problem == "dtype":
        b = b.astype(np.float32)
    elif problem == "shape":
        b = b[:-1]
    else:
        b[0, 0] = {"negative": -1., "nan": np.nan, "mass": 2.}[problem]
    with pytest.raises(ValueError):
        rl_branches(b, (26, 26), kernel(), [0])


@pytest.mark.parametrize("problem", ["dtype", "shape", "negative", "nan", "above_one", "origin"])
def test_invalid_kernels_rejected(problem):
    k = kernel()
    if problem == "dtype":
        k = k.astype(np.float32)
    elif problem == "shape":
        k = k[:, :-1]
    elif problem == "origin":
        k[0, 53, 53] = 1e-300
    else:
        k[0, 0, 0] = {"negative": -1., "nan": np.nan, "above_one": 1.01}[problem]
    with pytest.raises(ValueError):
        rl_branches(sparse_belief(), (26, 26), k, [0])


@pytest.mark.parametrize("values", [np.zeros((16, 1)), np.zeros(16, dtype=np.float32),
                                   np.full(16, np.nan), np.full(16, np.inf), [0.] * 16])
def test_invalid_value_outputs_rejected(values):
    branches = rl_branches(sparse_belief(), (26, 26), kernel(), [0])
    with pytest.raises(ValueError):
        explicit_scores(branches, lambda _z, _x, _k: values)


def test_invalid_routes_banks_and_score_inputs_fail_without_repair():
    branches = rl_branches(sparse_belief(), (26, 26), kernel(), [0])
    bank = coefficients()
    with pytest.raises(ValueError):
        explicit_scores(branches, None, arithmetic="native")
    with pytest.raises(ValueError):
        min_linear_scores(branches, bank, route="automatic")
    for invalid in (bank.astype(np.float32), bank[:3], np.zeros((0, 105, 105)), np.zeros((65, 105, 105)),
                    np.zeros((2, 104, 105)), np.full((1, 105, 105), np.nan)):
        with pytest.raises(ValueError):
            min_linear_scores(branches, invalid)
    for invalid in (np.zeros(4, dtype=int), np.zeros(3), np.full(4, np.nan), np.full(4, np.inf)):
        with pytest.raises(ValueError):
            select_action(invalid, [0, 1])
    with pytest.raises(TypeError):
        explicit_scores({}, None)
