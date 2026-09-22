"""Independent synthetic geometry and readout qualification, without scientific data."""

import copy
import io
import math

import numpy as np
import pytest

from openjev.research import otto_spatial_value as m
from openjev.research import otto_value_branches as branches


def center(physical, positions):
    """Independent physical-to-centered placement, never a production helper."""
    result = np.zeros((len(physical), 105, 105), np.float64)
    for index, (x, y) in enumerate(positions):
        result[index, 52 - x:105 - x, 52 - y:105 - y] = physical[index]
    return result


def examples():
    physical = np.zeros((5, 53, 53), np.float64)
    physical[0, 0, 0], physical[0, 1, 2], physical[0, 52, 52] = .25, .5, .25
    dense = np.arange(1, 2810, dtype=np.float64).reshape(53, 53)
    physical[1] = dense / dense.sum()
    physical[2, 4, 51], physical[2, 48, 3] = .125, .375
    physical[3, 7, 9] = 5e-11
    positions = np.array([[0, 0], [26, 26], [52, 52], [0, 52], [52, 0]], np.int64)
    return center(physical, positions), physical, positions, np.array([3., 4., 5., 3., 4.])


def direct_boxes(physical, size):
    """Cell-by-cell clipped sums; no prefix sums, convolution, or padding helper."""
    half = size // 2
    result = np.empty_like(physical)
    for batch in range(len(physical)):
        for x in range(53):
            for y in range(53):
                result[batch, x, y] = math.fsum(
                    float(physical[batch, i, j])
                    for i in range(max(0, x - half), min(53, x + half + 1))
                    for j in range(max(0, y - half), min(53, y + half + 1))
                )
    return result


def test_unpack_lossless_all_corners_and_canonical_centered_mass():
    centered, expected, pos, _ = examples()
    before = centered.copy()
    actual, mass = m.unpack_centered(centered, pos)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(center(actual, pos), centered)
    np.testing.assert_array_equal(mass, centered.reshape(5, -1).sum(axis=1))
    np.testing.assert_array_equal(centered, before)
    assert mass[2] == .5 and mass[3] == 5e-11 and mass[4] == 0


@pytest.mark.parametrize("size", [3, 9, 27])
def test_boxes_match_independent_dense_and_corner_sums(size):
    physical = examples()[1][:2]
    before = physical.copy()
    actual = m.box_sum(physical, size)
    np.testing.assert_allclose(actual, direct_boxes(physical, size), atol=3e-15, rtol=3e-15)
    np.testing.assert_array_equal(physical, before)
    assert actual[0, 0, 0] < 1  # Zero padding neither wraps nor rescales the boundary.
    assert actual[0, 52, 52] == .25


def test_spatial_channel_order_raw_subfloor_geometry_and_neighbor_free():
    centered, physical, pos, sensing = examples()
    mass = centered.reshape(5, -1).sum(axis=1)
    feature = m.spatial_features(physical, mass, pos, sensing)
    bare = m.spatial_features(physical, mass, pos, sensing, neighbor_free=True)
    assert feature.shape == bare.shape == (5, 53, 53, 12)
    assert feature.dtype == bare.dtype == np.float64
    np.testing.assert_array_equal(feature[..., 0], 53 * physical)
    np.testing.assert_array_equal(bare[..., 1:4], np.repeat(physical[..., None], 3, axis=-1))
    np.testing.assert_array_equal(feature[..., 4:], bare[..., 4:])
    for batch, x, y in ((0, 0, 0), (1, 13, 47), (2, 52, 52), (3, 7, 9), (4, 25, 25)):
        expected = [53 * physical[batch, x, y]]
        for size in (3, 9, 27):
            half = size // 2
            expected.append(physical[batch, max(0, x-half):min(53, x+half+1),
                                     max(0, y-half):min(53, y+half+1)].sum())
        qx, qy = pos[batch]
        expected += [(x-qx)/52, (y-qy)/52, x/52, (52-x)/52, y/52, (52-y)/52,
                     sensing[batch]/5, mass[batch]]
        np.testing.assert_allclose(feature[batch, x, y], expected, atol=3e-15, rtol=3e-15)
    assert feature[3, 7, 9, 1] == 5e-11
    assert feature[4, :, :, :4].sum() == 0  # Geometry persists, evidence is not invented.


def test_statistics_independent_scalar_oracle_zero_and_entropy_threshold():
    centered, physical, pos, sensing = examples()
    physical[3, 8, 9] = 2e-10
    centered = center(physical, pos)
    mass = centered.reshape(5, -1).sum(axis=1)
    actual = m.statistics_features(physical, mass, pos, sensing)
    expected = []
    for b in range(5):
        terms = [[] for _ in range(8)]
        qx, qy = pos[b]
        for x in range(53):
            for y in range(53):
                z = float(physical[b, x, y])
                dx, dy = x-qx, y-qy
                values = [(-z*math.log2(z) if z > 1e-10 else 0), z*abs(dx), z*abs(dy),
                          z*dx*dx, z*dy*dy, z*dx*dy, z*z, z]
                for bucket, value in zip(terms, values, strict=True):
                    bucket.append(value)
        expected.append([mass[b], mass[b]*sensing[b]/5, mass[b]*qx/52, mass[b]*qy/52,
                         math.fsum(terms[0])/math.log2(2809), math.fsum(terms[1])/52,
                         math.fsum(terms[2])/52, math.fsum(terms[3])/52**2,
                         math.fsum(terms[4])/52**2, math.fsum(terms[5])/52**2,
                         math.fsum(terms[6]), max(terms[7])])
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-15)
    np.testing.assert_array_equal(actual[-1], np.zeros(12))


@pytest.mark.parametrize("fault", ["padding", "negative", "nan", "mass", "dtype", "position", "empty", "batch"])
def test_unpack_rejects_discarded_padding_or_invalid_public_arrays(fault):
    centered, _, pos, _ = examples()
    if fault == "padding":
        centered[0, 0, 0] = 1e-30
    elif fault == "negative":
        centered[0, 52, 52] = -1e-30
    elif fault == "nan":
        centered[0, 52, 52] = np.nan
    elif fault == "mass":
        centered[0] *= 1.01
    elif fault == "dtype":
        centered = centered.astype(np.float32)
    elif fault == "position":
        pos = pos.astype(np.float64)
    elif fault == "empty":
        centered, pos = centered[:0], pos[:0]
    else:
        centered = np.repeat(centered[:1], 129, axis=0)
        pos = np.repeat(pos[:1], 129, axis=0)
    with pytest.raises((ValueError, TypeError)):
        m.unpack_centered(centered, pos)


COUNTS = {"spatial": 737, "neighbor_free": 737, "cnn": 801, "dense128": 1411841, "statistics": 225}


def nonzero_head(kind):
    """Deterministic artificial weights; no fitting or task data."""
    import torch

    model = m.make_head(kind, 419, .375)
    rng = np.random.default_rng(701)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            values = rng.normal(0, .035, tuple(parameter.shape)).astype(np.float32)
            if "bias" in name:
                values += np.float32(.06)
            parameter.copy_(torch.from_numpy(values))
        model.readout_bias_1.fill_(-.125)
    return model


def test_local_rng_parameter_counts_paired_initialization_and_common_function():
    import torch

    centered, _, pos, sensing = examples()
    t, q, length = torch.from_numpy(centered.astype(np.float32)), torch.from_numpy(pos), torch.from_numpy(
        sensing.astype(np.float32))
    original_rng = torch.random.get_rng_state().clone()
    exports = {}
    predictions = []
    for kind, expected_count in COUNTS.items():
        head = m.make_head(kind, 419, .375)
        assert m.parameter_count(kind) == expected_count == sum(p.numel() for p in head.parameters())
        assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in head.parameters())
        assert not head.c0.requires_grad
        assert torch.count_nonzero(head.readout_weight_1) == torch.count_nonzero(head.readout_bias_1) == 0
        with torch.no_grad():
            predictions.append(head(t, q, length))
        exports[kind] = m.export_head(head)
        again = m.export_head(m.make_head(kind, 419, .375))
        for name in exports[kind]:
            np.testing.assert_array_equal(exports[kind][name], again[name])
    assert torch.equal(original_rng, torch.random.get_rng_state())
    for prediction in predictions:
        torch.testing.assert_close(prediction, .375*t.reshape(5, -1).sum(dim=1), atol=0, rtol=0)
    for name, value in exports["spatial"].items():
        if name != "kind":
            np.testing.assert_array_equal(value, exports["neighbor_free"][name])


@pytest.mark.parametrize("kind", tuple(COUNTS))
def test_nonzero_torch64_numpy64_export_parity_and_batch_independence(kind):
    import torch

    centered, _, pos, sensing = examples()
    model = nonzero_head(kind)
    exported = m.export_head(model)
    stream = io.BytesIO()
    np.savez(stream, **exported)
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as archive:
        restored = {key: archive[key] for key in archive.files}
    frozen = m.FrozenValue(restored)
    double = copy.deepcopy(model).double()
    with torch.no_grad():
        expected = double(torch.from_numpy(centered), torch.from_numpy(pos), torch.from_numpy(sensing)).numpy()
    actual = frozen.normalized(centered, pos, sensing)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert actual.shape == (5,) and actual.dtype == np.float64
    assert np.max(np.abs(actual - .375*centered.reshape(5, -1).sum(axis=1))) > .01
    separate = [frozen.normalized(centered[i:i+1], pos[i:i+1], sensing[i:i+1])[0] for i in range(5)]
    np.testing.assert_allclose(actual, separate, atol=1e-10, rtol=1e-10)
    assert frozen.storage_bytes()["parameter_array_bytes"] == 8*COUNTS[kind]
    assert frozen.storage_bytes()["baseline_array_bytes"] == 8
    assert frozen.storage_bytes()["mutable_array_bytes"] == 0
    for value in frozen.arrays.values():
        assert value.dtype == np.float64 and not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(TypeError):
        frozen.arrays["hidden_history"] = np.zeros(1)
    for value in restored.values():
        if isinstance(value, np.ndarray) and value.dtype == np.float32:
            value.fill(77)
    np.testing.assert_array_equal(actual, frozen.normalized(centered, pos, sensing))


@pytest.mark.parametrize("kind", tuple(COUNTS))
def test_two_synthetic_updates_unlock_hidden_gradients(kind):
    import torch

    centered, _, pos, sensing = examples()
    inputs = (torch.from_numpy(centered[:2].astype(np.float32)), torch.from_numpy(pos[:2]),
              torch.from_numpy(sensing[:2].astype(np.float32)))
    head = m.make_head(kind, 419, .375)
    optimizer = torch.optim.SGD(head.parameters(), lr=.01)
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        (head(*inputs) - 2.).square().mean().backward()
        for name, parameter in head.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if update == 0 and name not in ("readout_weight_1", "readout_bias_1"):
                assert torch.count_nonzero(parameter.grad) == 0, name
            else:
                assert torch.count_nonzero(parameter.grad) > 0, (kind, update, name)
        assert head.c0.grad is None
        optimizer.step()


def test_spatial_shared_cell_head_equals_ordinary_one_by_one_convolutions():
    import torch
    from torch.nn import functional

    centered, physical, pos, sensing = examples()
    model = nonzero_head("spatial").double()
    mass = centered.reshape(5, -1).sum(axis=1)
    feature = m.spatial_features(physical, mass, pos, sensing)
    with torch.no_grad():
        x = torch.from_numpy(feature).permute(0, 3, 1, 2)
        for index in (0, 1):
            weight, bias = getattr(model, f"cell_weight_{index}"), getattr(model, f"cell_bias_{index}")
            x = functional.conv2d(x, weight[:, :, None, None], bias).relu()
        pooled = (x * torch.from_numpy(physical[:, None])).sum(dim=(2, 3))
        context = torch.from_numpy(np.column_stack((mass, mass*pos[:, 0]/52, mass*pos[:, 1]/52,
                                                   mass*sensing/5)))
        rho = torch.cat((pooled, context), dim=1)
        rho = functional.linear(rho, model.readout_weight_0, model.readout_bias_0).relu()
        expected = functional.linear(rho, model.readout_weight_1, model.readout_bias_1)[:, 0] + .375*torch.from_numpy(mass)
        actual = model(torch.from_numpy(centered), torch.from_numpy(pos), torch.from_numpy(sensing))
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_cnn_discards_outside_activations_before_next_dilated_layer():
    """An outside intermediate could return to occupied cell2 if incorrectly retained."""
    import torch

    physical = np.zeros((1, 53, 53), np.float64)
    physical[0, 0, 0], physical[0, 2, 0], physical[0, 4, 0] = .5, .25, .25
    pos, sensing = np.array([[0, 0]], np.int64), np.array([3.])
    centered = center(physical, pos)
    model = m.make_head("cnn", 419, 0.)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.conv_weight_0[0, 0, 2, 1] = 1
        model.conv_weight_1[0, 0, 0, 1] = 1
        model.conv_weight_2[0, 0, 1, 1] = 1
        model.cell_weight_0[0, 8] = 1  # Only third-level first channel reaches pooling.
        model.cell_weight_1[0, 0] = model.readout_weight_0[0, 0] = model.readout_weight_1[0, 0] = 1
    frozen = m.FrozenValue(m.export_head(model))
    with torch.no_grad():
        actual_torch = model.double()(torch.from_numpy(centered), torch.from_numpy(pos),
                                      torch.from_numpy(sensing)).numpy()
    expected = np.array([53 * .25 * .25])
    np.testing.assert_array_equal(actual_torch, expected)
    np.testing.assert_array_equal(frozen.normalized(centered, pos, sensing), expected)
    assert expected[0] != 53 * (.25*.25 + .5*.25)  # Erroneous out-of-board path contribution.


def test_constructed_local_neighbor_capability_is_not_task_benefit():
    """A hand-picked local statistic separates clustered and dispersed equal masses."""
    import torch

    physical = np.zeros((2, 53, 53), np.float64)
    physical[0, 20, 20], physical[0, 21, 20] = .5, .5
    physical[1, 20, 20], physical[1, 32, 20] = .5, .5
    pos = np.full((2, 2), 26, np.int64)
    sensing = np.full(2, 3.)
    centered = center(physical, pos)
    predictions = {}
    for kind in ("spatial", "neighbor_free"):
        head = m.make_head(kind, 419, 0.)
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.zero_()
            head.cell_weight_0[0, 1] = 1  # B3 versus the specified point-mass replacement.
            head.cell_bias_0[0] = -.75
            head.cell_weight_1[0, 0] = head.readout_weight_0[0, 0] = head.readout_weight_1[0, 0] = 1
        predictions[kind] = m.FrozenValue(m.export_head(head)).normalized(centered, pos, sensing)
    np.testing.assert_array_equal(predictions["spatial"], [.25, 0.])
    np.testing.assert_array_equal(predictions["neighbor_free"], [0., 0.])


@pytest.mark.parametrize("kind", tuple(COUNTS))
def test_explicit_sixteen_branch_units_biased_zero_subfloor_and_selection(kind):
    import torch

    physical = np.zeros((53, 53), np.float64)
    physical[0, 0], physical[1, 0], physical[40, 42] = .2, .3, .5
    kernel = np.empty((4, 107, 107), np.float64)
    for hit, probability in enumerate((0., 5e-11, .125, .75)):
        kernel[hit].fill(probability)
    kernel[:, 53, 53] = 0
    branch = branches.rl_branches(physical, (0, 0), kernel, (1, 3))
    model = m.make_head(kind, 419, .5)
    with torch.no_grad():
        model.readout_bias_1.fill_(-.25)
    frozen = m.FrozenValue(m.export_head(model), sensing_length=4.)
    called = []

    def callback(z, q, known_kernel):
        called.append((z.shape, q.shape))
        return frozen(z, q, known_kernel)

    actual = branches.explicit_scores(branch, callback)
    expected = []
    for action in range(4):
        terms = []
        for hit in range(4):
            i = 4*action+hit
            value = 64*(.5*branch.centered_z[i].sum()-.25)
            terms.append(float(branch.weights[action, hit])*value)
        expected.append(1+math.fsum(terms))
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert called == [((16, 105, 105), (16, 2))]
    assert frozen(branch.centered_z, branch.successors, kernel)[0] == -16
    assert branches.select_action(actual, (3, 1)) in (1, 3)
    with pytest.raises(ValueError):
        m.FrozenValue(m.export_head(model))(branch.centered_z, branch.successors, kernel)


def test_action_ties_are_strict_and_respect_eligible_numeric_order():
    scores = np.array([-5., 1e-10, 7., 0.])
    assert branches.select_action(scores, (3, 1)) == 3
    scores[1] = np.nextafter(1e-10, 0)
    assert branches.select_action(scores, (3, 1)) == 1
    np.testing.assert_array_equal(scores[[0, 2]], [-5., 7.])


@pytest.mark.parametrize("fault", ["dtype", "shape", "nonfinite", "extra", "missing", "metadata"])
def test_checkpoint_rejects_invalid_arrays_and_metadata(fault):
    exported = m.export_head(m.make_head("spatial", 419, .375))
    if fault == "dtype":
        exported["cell_weight_0"] = exported["cell_weight_0"].astype(np.float64)
    elif fault == "shape":
        exported["cell_weight_0"] = exported["cell_weight_0"][:, :-1]
    elif fault == "nonfinite":
        exported["readout_bias_1"][0] = np.nan
    elif fault == "extra":
        exported["hidden_source"] = np.array([1])
    elif fault == "missing":
        del exported["cell_bias_0"]
    else:
        exported["input_dim"] = 11029
    with pytest.raises((ValueError, TypeError)):
        m.FrozenValue(exported)
