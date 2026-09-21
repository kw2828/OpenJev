"""Synthetic scalar readouts only, without science inputs or native episodes."""

import copy
import dataclasses
import io

import numpy as np
import pytest

from openjev.research import otto_return_value as m
from openjev.research import otto_value_branches as branches


def centered_inputs():
    p = np.zeros((4, 105, 105), dtype=np.float64)
    p[0, 9, 17], p[0, 57, 81] = 0.25, 0.75
    p[1, 13, 80] = 5e-11
    dense = np.arange(1, 105**2 + 1, dtype=np.float64).reshape(105, 105)
    p[2] = dense / dense.sum()
    return p, np.array([[0, 52], [52, 13], [26, 39], [4, 17]], dtype=np.int64)


def checkpoint(kind, c0=0.3):
    head = {"version": m.VERSION, "kind": kind, "input_dim": m.INPUT_DIM,
            "c0": np.array(c0, dtype=np.float32),
            "first_weight": np.random.default_rng(105).normal(0, 0.01, (8, m.INPUT_DIM)).astype(np.float32)}
    if kind != "min8":
        head["final_weight"] = np.linspace(-0.03, 0.04, 8, dtype=np.float32)
    if kind == "mlp8":
        head["hidden_bias"] = np.linspace(-0.04, 0.03, 8, dtype=np.float32)
        head["output_bias"] = np.array(-0.02, dtype=np.float32)
    return head


def test_features_independent_dense_sparse_zero_and_subfloor_oracle():
    p, pos = centered_inputs()
    before = p.copy()
    actual = m.value_features(p, pos, 4)
    for i in range(4):
        flat = list(p[i].ravel())
        mass = sum(flat)
        expected = np.array([*flat, mass * int(pos[i, 0]) / 52,
                             mass * int(pos[i, 1]) / 52, mass * 4 / 5], dtype=np.float64)
        np.testing.assert_allclose(actual[i], expected, atol=2e-15, rtol=2e-15)
    np.testing.assert_array_equal(actual[:, :11025], p.reshape(4, 11025))
    np.testing.assert_array_equal(p, before)
    assert not actual[-1].any() and actual[1, 13 * 105 + 80] == 5e-11
    assert actual.shape == (4, 11028) and actual.dtype == np.float64


def test_training_features_are_final_cast_not_intermediate_rounding():
    p, pos = centered_inputs()
    f64 = m.value_features(p, pos, 3.7)
    f32 = m.value_features(p, pos, 3.7, dtype="float32")
    assert f32.dtype == np.float32
    np.testing.assert_array_equal(f32, f64.astype(np.float32))
    assert not np.shares_memory(f64, p)


@pytest.mark.parametrize("mutation", ["dtype", "shape", "nan", "negative", "mass", "empty", "batch"])
def test_feature_belief_contract_rejects_invalid_inputs(mutation):
    p, pos = centered_inputs()
    if mutation == "dtype":
        p = p.astype(np.float32)
    elif mutation == "shape":
        p = p[:, :104]
    elif mutation == "nan":
        p[0, 0, 0] = np.nan
    elif mutation == "negative":
        p[0, 0, 0] = -1e-20
    elif mutation == "mass":
        p[0] *= 1.01
    elif mutation == "empty":
        p, pos = p[:0], pos[:0]
    else:
        p = np.broadcast_to(p[:1], (m.MAX_BATCH + 1, 105, 105))
        pos = np.tile(pos[:1], (m.MAX_BATCH + 1, 1))
    with pytest.raises(ValueError):
        m.value_features(p, pos, 3)


@pytest.mark.parametrize("position", [np.ones((4, 2), dtype=bool), np.ones((4, 2)),
                                     [[-1, 0]] * 4, [[53, 0]] * 4, [[0, 0]]])
def test_feature_positions_are_nonboolean_integers(position):
    with pytest.raises(ValueError):
        m.value_features(centered_inputs()[0], position, 3)


@pytest.mark.parametrize("sensing", [True, 0, -1, np.inf, np.nan, "3"])
def test_sensing_validation(sensing):
    with pytest.raises(ValueError):
        m.value_features(*centered_inputs(), sensing)


@pytest.mark.parametrize("kind", m.KINDS)
def test_numpy_formula_matches_independent_scalar_loops(kind):
    h = checkpoint(kind)
    x = m.value_features(*centered_inputs(), 3)
    frozen = m.FrozenValue(h, 3)
    expected = []
    for row in x:
        linear = [sum(float(a) * float(b) for a, b in zip(row, w, strict=True)) for w in h["first_weight"]]
        if kind == "min8":
            residual = min(linear)
        else:
            bias = h["hidden_bias"] if kind == "mlp8" else np.zeros(8)
            residual = sum(max(v + float(b), 0) * float(w)
                           for v, b, w in zip(linear, bias, h["final_weight"], strict=True))
            if kind == "mlp8":
                residual += float(h["output_bias"])
        expected.append(float(h["c0"]) * sum(row[:11025]) + residual)
    np.testing.assert_allclose(frozen.normalized(x), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("kind", ("min8", "homogeneous8"))
def test_positive_homogeneity_and_exact_zero(kind):
    p, pos = centered_inputs()
    frozen = m.FrozenValue(checkpoint(kind), 4)
    base = frozen.normalized(m.value_features(p, pos, 4))
    for scale in (0.0, 1e-12, 0.25, 0.9):
        actual = frozen.normalized(m.value_features(p * scale, pos, 4))
        np.testing.assert_allclose(actual, base * scale, atol=1e-12, rtol=1e-12)
    assert frozen.normalized(np.zeros((1, m.INPUT_DIM)))[0] == 0


def test_ordinary_mlp_preserves_bias_at_zero_and_signed_values():
    h = checkpoint("mlp8", c0=0)
    h["first_weight"].fill(0)
    h["hidden_bias"].fill(0)
    h["output_bias"][...] = -0.25
    frozen = m.FrozenValue(h, 3)
    x = np.zeros((2, m.INPUT_DIM), dtype=np.float64)
    np.testing.assert_array_equal(frozen.normalized(x), [-0.25, -0.25])
    assert frozen.normalized(x[:1] * 0.5)[0] != frozen.normalized(x[:1])[0] * 0.5


def test_biased_zero_branch_is_retained_by_explicit_policy_callback():
    h = checkpoint("mlp8", c0=0)
    h["first_weight"].fill(0)
    h["hidden_bias"].fill(0)
    h["output_bias"][...] = 1 / 64
    p = np.zeros((53, 53), dtype=np.float64)
    p[27, 26] = 1
    kernel = np.full((4, 107, 107), 0.25, dtype=np.float64)
    kernel[:, 53, 53] = 0
    b = branches.rl_branches(p, (26, 26), kernel, [0, 1, 2, 3])
    values = m.FrozenValue(h, 3)
    costs = branches.explicit_scores(b, values)
    assert costs[1] == 1 + 4e-10
    assert costs[0] == costs[2] == costs[3] == 2
    np.testing.assert_array_equal(values(b.centered_z, b.successors, kernel), np.ones(16))


@pytest.mark.parametrize("kind", m.KINDS)
def test_npz_roundtrip_c0_and_immutable_owned_upcasts(kind):
    h = checkpoint(kind, c0=np.float32(0.123456789))
    file = io.BytesIO()
    np.savez(file, **h)
    file.seek(0)
    with np.load(file, allow_pickle=False) as archive:
        restored = {name: archive[name] for name in archive.files}
    assert m.validate_head(restored) == kind
    frozen = m.FrozenValue(restored, 4)
    assert frozen.c0.dtype == np.float64 and float(frozen.c0) == float(h["c0"])
    before = frozen.first_weight.copy()
    restored["first_weight"].fill(123)
    np.testing.assert_array_equal(frozen.first_weight, before)
    for name in ("c0", "first_weight", "final_weight", "hidden_bias", "output_bias"):
        a = getattr(frozen, name)
        if a is None:
            continue
        assert a.dtype == np.float64 and not a.flags.writeable
        with pytest.raises(ValueError):
            a.setflags(write=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        frozen.sensing_length = 7
    assert not hasattr(frozen, "__dict__")
    storage = frozen.storage_bytes()
    assert storage["parameter_array_bytes"] == m.parameter_count(kind) * 8
    assert storage["baseline_array_bytes"] == 8 and storage["mutable_array_bytes"] == 0


@pytest.mark.parametrize("mutation", ["extra", "missing", "kind", "version", "dim", "booldim",
                                     "object", "f64", "shape", "nan", "c0dtype", "c0shape", "bias"])
def test_checkpoint_validation_has_no_silent_repair(mutation):
    h = checkpoint("homogeneous8")
    if mutation == "extra":
        h["seed"] = 5
    elif mutation == "missing":
        del h["final_weight"]
    elif mutation == "kind":
        h["kind"] = "min64"
    elif mutation == "version":
        h["version"] = "other"
    elif mutation == "dim":
        h["input_dim"] = 11027
    elif mutation == "booldim":
        h["input_dim"] = True
    elif mutation == "object":
        h["version"] = np.array(m.VERSION, dtype=object)
    elif mutation == "f64":
        h["first_weight"] = h["first_weight"].astype(np.float64)
    elif mutation == "shape":
        h["first_weight"] = h["first_weight"].T
    elif mutation == "nan":
        h["first_weight"][0, 0] = np.nan
    elif mutation == "c0dtype":
        h["c0"] = np.array(0.2, dtype=np.float64)
    elif mutation == "c0shape":
        h["c0"] = np.array([0.2], dtype=np.float32)
    else:
        h["hidden_bias"] = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError):
        m.FrozenValue(h, 3)


def test_plane_usage_first_tie_and_callback_has_no_sensor_reference():
    h = checkpoint("min8", c0=0)
    h["first_weight"].fill(0)
    h["first_weight"][3, 0] = -1
    h["first_weight"][5, 0] = -1
    frozen = m.FrozenValue(h, 5)
    p = np.zeros((2, 105, 105), dtype=np.float64)
    p[1, 0, 0] = 1
    pos = np.array([[1, 2], [3, 4]])
    x = m.value_features(p, pos, 5)
    np.testing.assert_array_equal(frozen.plane_indices(x), [0, 3])

    class ForbiddenKernel:
        def __array__(self, *args, **kwargs):
            raise AssertionError("readout must not inspect sensor state")

    actual = frozen(p, pos, ForbiddenKernel())
    np.testing.assert_array_equal(actual, 64 * frozen.normalized(x))
    assert actual.dtype == np.float64
    with pytest.raises(ValueError):
        m.FrozenValue(checkpoint("mlp8"), 3).plane_indices(x)


def test_deployment_rejects_nonfinite_wrong_dtype_and_unbounded_batch():
    frozen = m.FrozenValue(checkpoint("min8"), 3)
    x = np.zeros((1, m.INPUT_DIM), dtype=np.float64)
    for invalid in (x.astype(np.float32), x[0], x[:, :-1], x[:0],
                    np.broadcast_to(x, (m.MAX_BATCH + 1, m.INPUT_DIM)), x + np.nan):
        with pytest.raises(ValueError):
            frozen.normalized(invalid)


def test_context_free_frozen_weights_accept_explicit_regime_features_only():
    frozen = m.FrozenValue(checkpoint("min8"))
    assert frozen.sensing_length is None
    p, pos = centered_inputs()
    for sensing in (3, 4, 5):
        x = m.value_features(p, pos, sensing)
        bound = m.FrozenValue(checkpoint("min8"), sensing)
        np.testing.assert_array_equal(frozen.normalized(x), bound.normalized(x))
    with pytest.raises(ValueError, match="bound sensing"):
        frozen(p, pos, None)


def test_common_seed_weights_exact_draw_order_and_no_global_rng_change():
    import torch

    before = torch.random.get_rng_state().clone()
    heads = [m.make_head(kind, 10101, np.float32(0.25)) for kind in m.KINDS]
    assert torch.equal(before, torch.random.get_rng_state())
    generator = torch.Generator(device="cpu").manual_seed(10101)
    expected_first = torch.randn((8, m.INPUT_DIM), generator=generator, dtype=torch.float32) * 0.01
    expected_final = torch.randn((8,), generator=generator, dtype=torch.float32) * 0.01
    for head in heads:
        assert torch.equal(head.first_weight, expected_first)
        assert sum(p.numel() for p in head.parameters()) == m.parameter_count(head.kind)
        assert set(dict(head.named_buffers())) == {"c0"}
        assert head.c0.dtype == torch.float32 and head.c0.item() == 0.25
        if head.kind != "min8":
            assert torch.equal(head.final_weight, expected_final)
        if head.kind == "mlp8":
            assert not head.hidden_bias.any() and head.output_bias.item() == 0
    assert not torch.equal(m.make_head("min8", 10102, 0.25).first_weight, expected_first)


@pytest.mark.parametrize("kind", m.KINDS)
def test_torch_double_of_learned_f32_export_matches_numpy_and_branch_actions(kind):
    import torch

    head = m.make_head(kind, 10101, np.float32(0.123456789))
    # A tiny synthetic parameter update exercises export after autograd.
    p, pos = centered_inputs()
    x = m.value_features(p, pos, 4)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss = ((head(torch.from_numpy(x.astype(np.float32))) - torch.tensor([0.2, 0.0, 0.3, 0.0])) ** 2).mean()
    loss.backward()
    for parameter in head.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert head.first_weight.grad.abs().sum().item() > 0
    assert head.c0.grad is None
    optimizer.step()
    exported = m.export_head(head)
    frozen = m.FrozenValue(exported, 4)
    reference = copy.deepcopy(head).double()
    with torch.no_grad():
        expected = reference(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(frozen.normalized(x), expected, atol=1e-12, rtol=1e-12)
    assert float(frozen.c0) == float(np.float32(0.123456789))
    native_belief = np.arange(1, 53**2 + 1, dtype=np.float64).reshape(53, 53)
    native_belief /= native_belief.sum()
    axis = np.arange(107, dtype=np.float64) - 53
    distance = np.abs(axis[:, None]) + np.abs(axis[None, :])
    k = np.stack([np.exp(-distance / scale) / 4 for scale in (2, 7, 23, 91)])
    k[:, 53, 53] = 0
    b = branches.rl_branches(native_belief, (0, 17), k, [1, 2, 3])

    def callback(z, successors, kernel):
        features = m.value_features(z, successors, 4)
        with torch.no_grad():
            return 64 * reference(torch.from_numpy(features)).numpy()

    actual_scores = branches.explicit_scores(b, frozen)
    expected_scores = branches.explicit_scores(b, callback)
    np.testing.assert_allclose(actual_scores, expected_scores, atol=1e-12, rtol=1e-12)
    assert branches.select_action(actual_scores, [1, 2, 3]) == branches.select_action(expected_scores, [1, 2, 3])
    with pytest.raises(ValueError):
        m.export_head(reference)


@pytest.mark.parametrize("kind", m.KINDS)
def test_synthetic_gradient_step_changes_residual_without_changing_baseline(kind):
    import torch

    head = m.make_head(kind, 12, 0.25)
    x = torch.zeros((3, m.INPUT_DIM), dtype=torch.float32)
    x[:, 0] = 1
    x[:, -3:] = torch.tensor([[0.2, 0.3, 0.6], [0.4, 0.5, 0.8], [0.8, 0.9, 1.0]])
    target = torch.tensor([0.5, 0.55, 0.6])
    optimizer = torch.optim.SGD(head.parameters(), lr=0.01)
    before = ((head(x) - target) ** 2).mean()
    optimizer.zero_grad()
    before.backward()
    optimizer.step()
    after = ((head(x) - target) ** 2).mean()
    assert after.item() < before.item()
    assert head.c0.item() == 0.25


@pytest.mark.parametrize("seed", [-1, 2**63, True, 1.5])
def test_seed_contract(seed):
    with pytest.raises(ValueError):
        m.make_head("min8", seed, 0.2)


@pytest.mark.parametrize("baseline", [True, np.nan, np.inf, "0.2"])
def test_baseline_contract(baseline):
    with pytest.raises(ValueError):
        m.make_head("min8", 5, baseline)
