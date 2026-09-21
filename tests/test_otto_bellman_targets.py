"""Synthetic arrays/exported heads only; no Torch, checkpoint or simulator."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from openjev.research.otto_bellman_targets import MAX_STATES, build_targets
from openjev.research.otto_return_value import INPUT_DIM, FrozenValue


def exported(kind="mlp8", c0=2., bias=0., x_coefficient=0.):
    first = np.zeros((8, INPUT_DIM), dtype=np.float32)
    first[:, -3] = x_coefficient
    result = {"version": "otto-return-value-v1", "kind": kind, "input_dim": INPUT_DIM,
              "c0": np.asarray(c0, dtype=np.float32), "first_weight": first}
    if kind != "min8":
        result["final_weight"] = np.full(8, .125, dtype=np.float32)
    if kind == "mlp8":
        result.update(hidden_bias=np.zeros(8, dtype=np.float32), output_bias=np.asarray(bias, dtype=np.float32))
    return result


def kernel(mass=.25):
    result = np.full((4, 107, 107), mass, dtype=np.float64)
    result[:, 53, 53] = 0
    return result


def inputs(count=1):
    beliefs = np.zeros((count, 53, 53), dtype=np.float64)
    beliefs[:, 5, 5] = 1
    return [beliefs, np.full((count, 2), 26, dtype=np.int64), [kernel()] * count,
            [3.] * count, [(0, 1, 2, 3)] * count]


def test_physical_and_normalized_units_with_exact_counts():
    result = build_targets(FrozenValue(exported()), *inputs())
    np.testing.assert_array_equal(result.branch_values, np.full((1, 16), 128.))
    np.testing.assert_array_equal(result.costs, np.full((1, 4), 129.))
    np.testing.assert_array_equal(result.targets_float64, [129. / 64])
    assert result.targets_float64.dtype == np.float64 and result.targets_float32.dtype == np.float32
    assert result.counts() == {"states": 1, "branch_constructions": 1, "checkpoint_call_invocations": 1,
        "evaluated_branch_rows": 16, "action_costs": 4, "training_target_casts": 1, "optimizer_steps": 0}
    assert result.diagnostics()["negative_targets"] == 0


def test_target_uses_true_minimum_not_deployed_near_tie_choice():
    result = build_targets(FrozenValue(exported("min8", c0=0., x_coefficient=-1e-12)), *inputs())
    assert result.minimum_actions[0] == 1 and result.deployed_actions[0] == 0
    assert 0 < result.deployed_cost_gaps[0] < 1e-10
    assert result.minimum_costs[0] == result.costs[0, 1] < result.costs[0, 0]
    assert result.targets_float64[0] == result.costs[0, 1] / 64
    assert result.targets_float64[0] != result.costs[0, 0] / 64
    assert result.diagnostics()["deployed_action_differs_from_first_minimizer"] == 1


def test_eligibility_changes_minimum_only_all_sixteen_values_remain():
    args = inputs()
    head = FrozenValue(exported("min8", c0=0., x_coefficient=-.25))
    all_actions = build_targets(head, *args)
    args[-1] = [(2,)]
    restricted = build_targets(head, *args)
    np.testing.assert_array_equal(restricted.branch_values, all_actions.branch_values)
    np.testing.assert_array_equal(restricted.costs, all_actions.costs)
    assert restricted.minimum_actions[0] == restricted.deployed_actions[0] == 2
    assert restricted.minimum_costs[0] == all_actions.costs[0, 2] > all_actions.minimum_costs[0]
    assert restricted.eligible_masks.tolist() == [[False, False, True, False]]


def test_subfloor_biased_zero_values_are_not_suppressed():
    args = inputs()
    sensor = np.zeros((4, 107, 107), dtype=np.float64)
    sensor[0] = .5e-10
    sensor[:, 53, 53] = 0
    args[2] = [sensor]
    result = build_targets(FrozenValue(exported(c0=2., bias=.25)), *args)
    assert result.raw_masses[0].tolist() == [[.5e-10, 0., 0., 0.]] * 4
    np.testing.assert_array_equal(result.weights, np.full((1, 4, 4), 1e-10))
    np.testing.assert_array_equal(result.branch_values, np.tile([80., 16., 16., 16.], (1, 4)))
    np.testing.assert_allclose(result.costs, [[1 + 1.28e-8] * 4], rtol=0, atol=2e-16)
    assert result.diagnostics()["zero_raw_mass_branches"] == 12
    assert result.diagnostics()["positive_subfloor_branches"] == 4


@pytest.mark.parametrize("kind", ["min8", "homogeneous8", "mlp8"])
def test_deterministic_found_branch_still_evaluates_value_at_zero(kind):
    args = inputs()
    args[0].fill(0)
    args[0][0, 27, 26] = 1
    result = build_targets(FrozenValue(exported(kind, bias=.25)), *args)
    assert result.raw_masses[0, 1].tolist() == [0.] * 4
    expected_value = 16. if kind == "mlp8" else 0.
    assert result.branch_values[0, 4:8].tolist() == [expected_value] * 4
    assert result.costs[0, 1] == 1 + 4e-10 * expected_value
    assert result.minimum_actions[0] == 1


def test_signed_targets_are_retained():
    result = build_targets(FrozenValue(exported("min8", c0=-2.)), *inputs())
    assert result.targets_float64.tolist() == [-127 / 64]
    assert result.targets_float32.tolist() == [-127 / 64]
    assert result.diagnostics()["negative_branch_values"] == 16
    assert result.diagnostics()["negative_action_costs"] == 4
    assert result.diagnostics()["negative_targets"] == 1


def test_corner_blocked_stays_and_scalar_public_context():
    args = inputs()
    args[1][:] = 0
    params = exported("min8", c0=0.)
    params["first_weight"][:, -1] = 1
    head = FrozenValue(params, sensing_length=99.)
    result = build_targets(head, *args)
    assert result.successors[0, ::4].tolist() == [[0, 0], [1, 0], [0, 0], [0, 1]]
    assert result.costs[0, 0] == result.costs[0, 2]
    np.testing.assert_allclose(result.branch_values, 64 * 3 / 5, rtol=0, atol=1e-14)
    args[3] = [4.]
    shifted = build_targets(head, *args)
    np.testing.assert_allclose(shifted.branch_values, 64 * 4 / 5, rtol=0, atol=1e-14)


def test_dense_independent_scalar_reduction_oracle():
    args = inputs()
    dense = np.arange(1, 53 * 53 + 1, dtype=np.float64).reshape(53, 53)
    dense /= dense.sum()
    args[0][0] = dense
    args[1][0] = [12, 33]
    args[3] = [4.]
    params = exported("mlp8", c0=1.5, bias=-.125, x_coefficient=.25)
    result = build_targets(FrozenValue(params), *args)
    expected_values, expected_costs = [], []
    for action in range(4):
        position = [12, 33]
        position[action // 2] += 2 * (action % 2) - 1
        x, y = position
        # Uniform four-category mechanical sensor: only the found cell is removed.
        raw = .25 * (1 - dense[x, y])
        normalized_mass = raw / max(raw, 1e-10)
        normalized_value = 1.5 * normalized_mass + .25 * normalized_mass * x / 52 - .125
        value = 64 * normalized_value
        expected_values.extend([value] * 4)
        expected_costs.append(1 + 4 * max(raw, 1e-10) * value)
    np.testing.assert_allclose(result.branch_values[0], expected_values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.costs[0], expected_costs, rtol=1e-12, atol=1e-12)
    assert result.minimum_actions[0] == int(np.argmin(expected_costs))
    assert result.targets_float64[0] == result.minimum_costs[0] / 64


def test_batch_is_exactly_single_state_computation_with_sixteen_row_calls(monkeypatch):
    args = inputs(3)
    args[0][1] *= .125
    args[0][2].fill(0)
    args[1][1] = [0, 52]
    args[3] = [3., 4., 5.]
    args[4] = [(0, 1, 2, 3), (1, 2), (3, 1)]
    head = FrozenValue(exported("mlp8", c0=.375, bias=.0625, x_coefficient=.25))
    calls = []
    original = FrozenValue.normalized
    def record(self, features):
        calls.append((features.shape, features.dtype))
        return original(self, features)
    monkeypatch.setattr(FrozenValue, "normalized", record)
    combined = build_targets(head, *args)
    assert calls == [((16, INPUT_DIM), np.dtype("float64"))] * 3
    for index in range(3):
        single_args = [args[0][index:index + 1], args[1][index:index + 1], [args[2][index]],
                       [args[3][index]], [args[4][index]]]
        single = build_targets(head, *single_args)
        for field in combined.__slots__:
            np.testing.assert_array_equal(getattr(combined, field)[index:index + 1], getattr(single, field))
    assert combined.counts()["checkpoint_call_invocations"] == 3
    assert combined.counts()["evaluated_branch_rows"] == 48


def test_owned_immutable_witnesses_and_no_input_mutation():
    args = inputs()
    before_belief, before_kernel, before_position = args[0].copy(), args[2][0].copy(), args[1].copy()
    params = exported()
    head = FrozenValue(params)
    weight_bytes = head.first_weight.tobytes()
    result = build_targets(head, *args)
    np.testing.assert_array_equal(args[0], before_belief)
    np.testing.assert_array_equal(args[2][0], before_kernel)
    np.testing.assert_array_equal(args[1], before_position)
    assert head.first_weight.tobytes() == weight_bytes
    for field in result.__slots__:
        array = getattr(result, field)
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        result.costs = np.zeros((1, 4))
    retained = result.raw_masses.copy()
    args[0].fill(0)
    args[2][0].fill(0)
    params["c0"][...] = -1
    np.testing.assert_array_equal(result.raw_masses, retained)
    assert not hasattr(result, "__dict__") and "centered_z" not in result.__slots__


@pytest.mark.parametrize("mutation", ["empty", "large", "f32", "nan", "negative", "excess_mass", "position",
    "bool_position", "kernel_count", "bad_kernel", "kernel_origin", "sensing", "sensing_count", "eligible", "duplicate"])
def test_malformed_public_inputs_rejected(mutation):
    args = inputs()
    if mutation == "empty":
        args[0] = args[0][:0]
    elif mutation == "large":
        args = inputs(MAX_STATES + 1)
    elif mutation == "f32":
        args[0] = args[0].astype(np.float32)
    elif mutation == "nan":
        args[0][0, 1, 1] = np.nan
    elif mutation == "negative":
        args[0][0, 1, 1] = -.1
    elif mutation == "excess_mass":
        args[0] *= 2
    elif mutation == "position":
        args[1][0, 0] = 53
    elif mutation == "bool_position":
        args[1] = args[1].astype(bool)
    elif mutation == "kernel_count":
        args[2] = []
    elif mutation == "bad_kernel":
        args[2][0][0, 1, 1] = np.inf
    elif mutation == "kernel_origin":
        args[2][0][:, 53, 53] = .25
    elif mutation == "sensing":
        args[3] = [True]
    elif mutation == "sensing_count":
        args[3] = []
    elif mutation == "eligible":
        args[4] = [()]
    else:
        args[4] = [(0, 0)]
    with pytest.raises(ValueError):
        build_targets(FrozenValue(exported()), *args)


def test_unfrozen_target_and_float32_overflow_rejected():
    with pytest.raises(TypeError):
        build_targets(exported(), *inputs())
    params = exported("mlp8", c0=0.)
    params["first_weight"].fill(np.finfo(np.float32).max)
    params["final_weight"].fill(np.finfo(np.float32).max)
    with pytest.raises(FloatingPointError):
        build_targets(FrozenValue(params), *inputs())


def test_nonfinite_frozen_readout_rejected(monkeypatch):
    monkeypatch.setattr(FrozenValue, "normalized", lambda _self, _features: np.full(16, np.nan))
    with pytest.raises(ValueError, match="physical branch values"):
        build_targets(FrozenValue(exported()), *inputs())


def test_later_state_failure_reports_completed_and_pending_readouts(monkeypatch):
    events, calls = [], []
    original = FrozenValue.normalized
    def failing(self, features):
        calls.append(features.shape)
        if len(calls) == 2:
            raise RuntimeError("synthetic second readout failure")
        return original(self, features)
    monkeypatch.setattr(FrozenValue, "normalized", failing)
    with pytest.raises(RuntimeError, match="second readout"):
        build_targets(FrozenValue(exported()), *inputs(3),
                      observe=lambda event, index, rows: events.append((event, index, rows)))
    assert calls == [(16, INPUT_DIM)] * 2
    assert events == [("attempt", 0, 16), ("return", 0, 16), ("attempt", 1, 16)]


def test_observer_return_cannot_replace_values_and_invalid_observer_fails():
    head, args = FrozenValue(exported()), inputs()
    plain = build_targets(head, *args)
    observed = build_targets(head, *args, observe=lambda _event, _index, _rows: float("nan"))
    for field in plain.__slots__:
        np.testing.assert_array_equal(getattr(plain, field), getattr(observed, field))
    with pytest.raises(TypeError, match="observe"):
        build_targets(head, *args, observe=True)
