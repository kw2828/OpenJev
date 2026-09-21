"""Invented arrays only: no released tensors, TensorFlow, OTTO or episodes."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from openjev.research import otto_pretrained_value as M


@pytest.fixture(scope="module")
def tensors():
    # One sparse path has a hand-computable scalar transfer function:
    # u=ReLU(x[0,0]-.25), v=ReLU(2*u-.125), w=ReLU(v+.0625), y=-2*w-.5.
    values = {key: np.zeros(shape, dtype=np.float32) for key, shape in M.EXPECTED_SHAPES.items()}
    values["kernel_0"][0, 0] = 1
    values["bias_0"][0] = -.25
    values["kernel_1"][0, 0] = 2
    values["bias_1"][0] = -.125
    values["kernel_2"][0, 0] = 1
    values["bias_2"][0] = .0625
    values["kernel_3"][0, 0] = -2
    values["bias_3"][0] = -.5
    for value in values.values():
        value.flags.writeable = False
    return values


@pytest.fixture(scope="module")
def model(tensors):
    return M.PretrainedValue(tensors)


def scalar(x):
    return -2 * max(0, max(0, 2 * max(0, x - .25) - .125) + .0625) - .5


def centered(batch=1):
    return np.zeros((batch, 105, 105), dtype=np.float64)


def belief():
    result = np.zeros((53, 53), dtype=np.float64)
    result[4, 7] = 1
    return result


def kernel():
    result = np.zeros((4, 107, 107), dtype=np.float64)
    result[0] = .25
    result[1] = .75
    return result


@pytest.mark.parametrize("position", [(0, 0), (0, 52), (52, 0), (52, 52), (26, 26)])
def test_centering_maps_every_cell_by_exact_offset(position):
    source = np.arange(1, 2810, dtype=np.float64).reshape(1, 53, 53)
    source /= source.sum()
    before = source.copy()
    actual = M.center_beliefs(source, [position])
    expected = np.zeros((1, 105, 105), dtype=np.float32)
    for i in range(53):
        for j in range(53):
            expected[0, 52 + i - position[0], 52 + j - position[1]] = source[0, i, j]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, before)
    assert actual.dtype == np.float32


@pytest.mark.parametrize("mass", [0., .5e-10, 1e-10, 2e-10, .5, 1.])
def test_centering_never_normalizes_or_floors_subnormalized_mass(mass):
    source = belief()[np.newaxis] * mass
    actual = M.center_beliefs(source, [(4, 7)])
    assert actual[0, 52, 52] == np.float32(mass)
    assert np.count_nonzero(actual) == int(mass > 0)


def test_exact_symmetry_order_is_transform_major_with_batch_inside():
    x = np.arange(2 * 105 * 105, dtype=np.float64).reshape(2, 105, 105)
    x /= x.sum(axis=(1, 2), keepdims=True)
    actual = M.symmetry_inputs(x)
    # Independent coordinate oracle, without transpose/flip/concatenation.
    expected = np.empty((8, 2, 105, 105), dtype=np.float32)
    for i in range(105):
        for j in range(105):
            locations = [(i, j), (j, i), (104 - i, j), (j, 104 - i),
                         (104 - i, 104 - j), (104 - j, 104 - i), (i, 104 - j), (104 - j, i)]
            for k, (u, v) in enumerate(locations):
                expected[k, :, i, j] = x[:, u, v]
    np.testing.assert_array_equal(actual, expected)
    assert M.D4_ORDER == ("identity", "transpose", "reverse_x", "reverse_x_transpose",
                          "reverse_xy", "reverse_xy_transpose", "reverse_y", "reverse_y_transpose")


def test_three_relu_layers_then_unclamped_negative_linear_output(model):
    x = centered(3)
    x[:, 0, 0] = [.125, .375, .75]
    before = x.copy()
    actual = model.predict_centered(x)
    np.testing.assert_array_equal(actual[:, 0], np.array([scalar(v) for v in x[:, 0, 0]], dtype=np.float32))
    assert actual.dtype == np.float32 and (actual < 0).all()
    np.testing.assert_array_equal(x, before)


def test_zero_branch_keeps_model_bias_output(model):
    actual = model.predict_centered(centered())
    np.testing.assert_array_equal(actual, [[scalar(0)]])
    assert actual[0, 0] != 0


def test_symmetry_average_matches_hand_oracle_and_does_not_mix_batch(model):
    x = centered(2)
    x[0, 0, 0], x[0, -1, -1] = .75, .25
    x[1, 0, -1], x[1, -1, 0] = .5, .5
    actual = model.predict_centered(x, sym_avg=True)
    expected = [sum(scalar(v) for v in [.75, .25, 0, 0]) / 4,
                sum(scalar(v) for v in [.5, .5, 0, 0]) / 4]
    np.testing.assert_array_equal(actual[:, 0], np.asarray(expected, dtype=np.float32))
    reflected = model.predict_centered(x.transpose(0, 2, 1)[:, ::-1], sym_avg=True)
    np.testing.assert_array_equal(actual, reflected)


def test_belief_entry_point_is_exact_composition(model):
    source = belief()[np.newaxis]
    a = model.predict_beliefs(source, [(52, 52)])
    b = model.predict_centered(M.center_beliefs(source, [(52, 52)]))
    np.testing.assert_array_equal(a, b)


def test_weights_are_bytes_backed_immutable_and_storage_complete(model, tensors):
    assert model.storage_bytes()["immutable_array_bytes"] == 53_563_396
    assert model.storage_bytes()["parameters"] == 13_390_849
    assert model.storage_bytes()["mutable_array_bytes"] == 0
    assert model.storage_bytes()["max_symmetry_batch"] == 128
    for name in M.KEYS:
        assert not np.shares_memory(tensors[name], model.tensors[name])
        np.testing.assert_array_equal(tensors[name], model.tensors[name])
        with pytest.raises(ValueError):
            model.tensors[name].flags.writeable = True
    with pytest.raises(TypeError):
        model.tensors["bias_3"] = np.zeros(1)
    with pytest.raises(FrozenInstanceError):
        model.max_batch = 100


def test_caller_mutation_cannot_change_model(tensors):
    values = dict(tensors)
    values["bias_3"] = np.array([-7], dtype=np.float32)
    copy = M.PretrainedValue(values, max_batch=1)
    values["bias_3"][0] = 100
    assert copy.tensors["bias_3"][0] == -7
    with pytest.raises(ValueError, match="batch"):
        copy.predict_centered(centered(2))


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong_shape", "float64", "nan", "inf", "list"])
def test_strict_eight_tensor_contract(tensors, mutation):
    values = dict(tensors)
    if mutation == "missing":
        values.pop("bias_3")
    elif mutation == "extra":
        values["other"] = np.zeros(1)
    elif mutation == "wrong_shape":
        values["kernel_0"] = np.zeros((1024, 11025), dtype=np.float32)
    elif mutation == "float64":
        values["kernel_0"] = np.zeros((11025, 1024), dtype=np.float64)
    elif mutation in {"nan", "inf"}:
        values["kernel_0"] = np.broadcast_to(np.float32(np.nan if mutation == "nan" else np.inf), (11025, 1024))
    else:
        values["kernel_0"] = []
    with pytest.raises(ValueError):
        M.PretrainedValue(values)


@pytest.mark.parametrize("bound", [0, 17, -1, True, 1.5, "16", None])
def test_batch_bound_is_explicit_before_allocation(tensors, bound):
    with pytest.raises((TypeError, ValueError), match="max_batch"):
        M.PretrainedValue(tensors, max_batch=bound)


@pytest.mark.parametrize("batch", [0, 17])
def test_no_silent_large_batch_chunking(model, batch):
    with pytest.raises(ValueError, match="batch"):
        model.predict_centered(centered(batch), sym_avg=True)


@pytest.mark.parametrize("position", [[True, 1], [-1, 0], [53, 0], [0.5, 1], [1], [1, 2, 3]])
def test_positions_are_public_bounded_integer_pairs(position):
    with pytest.raises(ValueError):
        M.center_beliefs(belief()[np.newaxis], [position])


@pytest.mark.parametrize("bad", ["negative", "nan", "inf", "mass", "integer", "shape"])
def test_invalid_probability_inputs_rejected_before_network(model, bad):
    x = centered()
    if bad == "integer":
        x = x.astype(np.int32)
    elif bad == "shape":
        x = x[:, :-1]
    else:
        x[0, 0, 0] = {"negative": -.1, "nan": np.nan, "inf": np.inf, "mass": 1.1}[bad]
    with pytest.raises(ValueError):
        model.predict_centered(x)


def test_symmetry_flag_is_strict(model):
    with pytest.raises(TypeError, match="sym_avg"):
        model.predict_centered(centered(), sym_avg=1)


def test_nonfinite_dense_overflow_fails_without_repair(tensors):
    values = dict(tensors)
    values["bias_0"] = np.full(1024, np.finfo(np.float32).max, dtype=np.float32)
    big = M.PretrainedValue(values)
    with pytest.raises(FloatingPointError):
        big.predict_centered(centered())


@pytest.mark.parametrize("mass", [0., .5e-10, 1e-10, 2e-10])
def test_policy_branch_floor_before_normalization_without_reweighting(mass):
    k = np.zeros((4, 107, 107), dtype=np.float64)
    k[2] = mass
    inputs, masses = M.policy_inputs(belief(), (26, 26), k)
    assert inputs.shape == (16, 105, 105) and inputs.dtype == np.float32
    assert masses.shape == (4, 4) and masses.dtype == np.float32
    expected_mass = max(1e-10, mass)
    np.testing.assert_array_equal(masses[:, 2], np.full(4, expected_mass, dtype=np.float32))
    np.testing.assert_array_equal(masses[:, [0, 1, 3]], np.full((4, 3), 1e-10, dtype=np.float32))
    np.testing.assert_array_equal(inputs.reshape(4, 4, 105, 105).sum(axis=(2, 3))[:, 2],
                                  np.full(4, mass / expected_mass, dtype=np.float32))
    assert masses.sum() < 1e-8  # Floored branch weights are not renormalized.


def test_policy_blocked_moves_stay_and_all_four_actions_retained():
    p = belief()
    p_before = p.copy()
    k = kernel()
    before = k.copy()
    inputs, masses = M.policy_inputs(p, (0, 0), k)
    # Actions0/2 are blocked,1 increments x,3 increments y.
    for action, moved in enumerate([(0, 0), (1, 0), (0, 0), (0, 1)]):
        expected = (52 + 4 - moved[0], 52 + 7 - moved[1])
        for hit in [0, 1]:
            assert inputs[4 * action + hit][expected] == 1
    np.testing.assert_array_equal(inputs[0], inputs[8])
    np.testing.assert_array_equal(masses[:, :2], np.tile(np.array([.25, .75], dtype=np.float32), (4, 1)))
    np.testing.assert_array_equal(p, p_before)
    np.testing.assert_array_equal(k, before)


def test_policy_extracts_offset_likelihood_without_repair_or_extra_exclusion():
    p = np.zeros((53, 53), dtype=np.float64)
    p[26, 26], p[25, 26] = .25, .75
    k = np.zeros((4, 107, 107), dtype=np.float64)
    k[0, 54, 53] = .5  # Relative to action0's moved position25,26.
    k[0, 53, 53] = .25  # Deliberately nonphysical origin proves no implicit masking.
    inputs, masses = M.policy_inputs(p, (26, 26), k)
    expected_mass = .25 * .5 + .75 * .25
    assert masses[0, 0] == np.float32(expected_mass)
    assert inputs[0, 53, 52] == np.float32(.125 / expected_mass)
    assert inputs[0, 52, 52] == np.float32(.1875 / expected_mass)


class FakeValue:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32).reshape(16, 1)
        self.calls = []

    def predict_centered(self, inputs, *, sym_avg):
        self.calls.append((inputs.copy(), sym_avg))
        return self.values.copy()


def test_policy_float32_expectation_and_first_tie_including_blocked_action():
    values = np.array([[2, 6, -100, 10], [7, 7, -100, 10],
                       [2, 6, -100, 10], [8, 8, -100, 10]], dtype=np.float32)
    fake = FakeValue(values)
    action, scores = M.value_policy(fake, belief(), (0, 0), kernel())
    assert action == 0  # blocked0 beats legal1/3; identical blocked2 loses first tie.
    expected = np.float32(1) + (values * np.array([.25, .75, 1e-10, 1e-10], dtype=np.float32)).sum(axis=1)
    np.testing.assert_array_equal(scores, expected)
    assert scores.dtype == np.float32 and fake.calls[0][1] is True


def test_policy_raw_negative_values_not_clamped_and_flag_forwarded():
    fake = FakeValue(np.tile([-3, -3, 0, 0], (4, 1)))
    action, scores = M.value_policy(fake, belief(), (26, 26), kernel(), sym_avg=False)
    assert action == 0 and (scores == -2).all()
    assert fake.calls[0][1] is False


@pytest.mark.parametrize("bad", ["belief32", "kernel32", "negative", "nan", "above_one", "shape"])
def test_policy_contract_rejects_invalid_sources(bad):
    p, k = belief(), kernel()
    if bad == "belief32":
        p = p.astype(np.float32)
    elif bad == "kernel32":
        k = k.astype(np.float32)
    elif bad == "shape":
        k = k[:, :-1]
    else:
        k[0, 0, 0] = {"negative": -.01, "nan": np.nan, "above_one": 1.01}[bad]
    with pytest.raises(ValueError):
        M.policy_inputs(p, (26, 26), k)


def test_mechanical_kernel_need_not_normalize_channels():
    k = np.ones((4, 107, 107), dtype=np.float64)
    _, masses = M.policy_inputs(belief(), (26, 26), k)
    np.testing.assert_array_equal(masses, np.ones((4, 4), dtype=np.float32))
