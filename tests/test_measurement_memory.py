"""Hand-constructed measurement-memory fixtures, without scientific draws."""
from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest
from scipy.linalg import svd
from scipy.special import ndtr

from openjev.research import measurement_memory as memory


def grid():
    return np.array([(i / 4 - 2, j / 4 - 2) for i in range(17) for j in range(17)])


def law(x, z):
    delta = x[:, None] - z[None]
    return np.exp(-np.sum(delta * delta, axis=-1) / 2) + 1e-5 * np.all(delta == 0, axis=-1)


def labels(ids):
    points = grid()[ids]
    return np.sin(points[:, 0]) + .3 * np.cos(2 * points[:, 1])


def requests(batch=1):
    indices = np.array([[[0, 2, 4, 6], [17, 51, 85, 119], [70, 72, 74, 76], [210, 212, 214, 216]],
                        [[0, 36, 72, 108], [36, 68, 100, 132], [85, 121, 157, 193], [180, 182, 184, 186]]])
    return np.broadcast_to(grid()[indices], (batch, 2, 4, 4, 2)).copy()


def dense_prediction(ids, y, paths, operator=None):
    x = grid()[ids]
    a = np.eye(len(ids)) if operator is None else operator
    observed = a @ (law(x, x) + .09 * np.eye(len(ids))) @ a.T
    target = a @ y
    means, variances = [], []
    for path in paths.reshape(-1, 4, 2):
        cross = law(path, x).mean(0) @ a.T
        means.append(cross @ np.linalg.solve(observed, target))
        variances.append(law(path, path).mean() - cross @ np.linalg.solve(observed, cross))
    shape = paths.shape[:2]
    return np.array(means).reshape(shape), np.array(variances).reshape(shape)


def fill(initial, write, ids, batch=1):
    state = initial(batch)
    for index, value in zip(ids, labels(ids), strict=True):
        state = write(state, np.broadcast_to(grid()[index], (batch, 2)).copy(), np.full(batch, value))
    return state


def selected(mask):
    return np.flatnonzero(np.unpackbits(mask, bitorder="little", count=289))


@pytest.mark.parametrize("kind", memory.KINDS)
def test_basis_is_deterministic_owned_and_has_declared_row_geometry(kind):
    actual = memory.basis(kind)
    np.testing.assert_array_equal(actual, memory.basis(kind))
    assert actual.shape == (118, 289) and actual.dtype == np.float64
    with pytest.raises(ValueError):
        actual.setflags(write=True)
    if kind == "bins":
        assert set(actual.sum(axis=1)) == {2., 3.}
        np.testing.assert_array_equal(actual.sum(axis=0), np.ones(289))
        np.testing.assert_array_equal(actual.argmax(axis=0), np.arange(289) * 118 // 289)
    else:
        np.testing.assert_allclose(actual @ actual.T, np.eye(118), atol=3e-14)
    if kind == "spectral":
        diagonal = actual @ law(grid(), grid()) @ actual.T
        np.testing.assert_allclose(diagonal, np.diag(diagonal.diagonal()), atol=2e-13)
        assert np.all(np.diff(diagonal.diagonal()) <= 2e-13)
    elif kind == "dct":
        np.testing.assert_allclose(actual[0], 1 / 17, atol=2e-17)
        axis = np.cos(np.pi * (np.arange(17) + .5) / 17) * np.sqrt(2 / 17)
        np.testing.assert_allclose(actual[1].reshape(17, 17), np.tile(axis / np.sqrt(17), (17, 1)), atol=2e-17)


@pytest.mark.parametrize("kind", memory.KINDS)
def test_first_64_events_are_exact_raw_gp_not_an_early_rank_approximation(kind):
    ids = (np.arange(64) * 73) % 289
    state = fill(lambda batch: memory.initial_hybrid(batch, kind), memory.write_hybrid, ids)
    np.testing.assert_array_equal(selected(state.mask[0]), np.sort(ids))
    np.testing.assert_array_equal(state.values[0, :64], labels(np.sort(ids)))
    assert not np.any(state.values[:, 64:])
    got = memory.predict_hybrid(state, requests())
    mean, variance = dense_prediction(ids, labels(ids), requests()[0])
    np.testing.assert_allclose(got["mean"][0], mean, atol=2e-13)
    np.testing.assert_allclose(got["variance"][0], variance, atol=2e-13)
    np.testing.assert_array_equal(got["rank"], [64])
    np.testing.assert_array_equal(got["rank_cutoff"], [0.])


@pytest.mark.parametrize("kind", memory.KINDS)
def test_transition_once_at_119_and_later_additions_preserve_all_measurement_sums(kind):
    ids = (np.arange(125) * 73) % 289
    state = fill(lambda batch: memory.initial_hybrid(batch, kind), memory.write_hybrid, ids[:118])
    np.testing.assert_array_equal(state.values[0], labels(np.sort(ids[:118])))
    phi = memory.basis(kind)
    for end in range(119, 126):
        state = memory.write_hybrid(state, grid()[ids[end - 1]][None], labels(ids[end - 1:end]))
        np.testing.assert_allclose(state.values[0], phi[:, ids[:end]] @ labels(ids[:end]), atol=8e-14, rtol=2e-14)
        np.testing.assert_array_equal(selected(state.mask[0]), np.sort(ids[:end]))
        assert state.step == end


@pytest.mark.parametrize("kind", memory.KINDS)
def test_full_grid_linear_measurement_conditioning_matches_dense_gaussian(kind):
    # Full-grid rows are orthogonal, so the independent dense Gaussian is well-conditioned.
    ids = np.arange(289)
    phi = memory.basis(kind)
    mask = np.packbits(np.ones((1, 289), dtype=np.uint8), axis=1, bitorder="little")
    state = memory.HybridMemory(mask, (phi @ labels(ids))[None], kind, 289)
    got = memory.predict_hybrid(state, requests())
    mean, variance = dense_prediction(ids, labels(ids), requests()[0], phi)
    np.testing.assert_allclose(got["mean"][0], mean, atol=4e-12, rtol=4e-12)
    np.testing.assert_allclose(got["variance"][0], variance, atol=4e-12, rtol=4e-12)
    np.testing.assert_allclose(got["risk"][0], ndtr((mean - .5) / np.sqrt(variance)), atol=4e-12)
    assert got["rank"].tolist() == [118]
    singular = np.linalg.svd(phi, compute_uv=False)
    assert got["rank_cutoff"][0] == pytest.approx(np.finfo(float).eps * 289 * singular[0], rel=1e-14)


def test_bins_rank_deficiency_is_reported_and_conditions_on_nonempty_exact_sums():
    ids = np.arange(119)
    phi = memory.basis("bins")
    state = fill(lambda batch: memory.initial_hybrid(batch, "bins"), memory.write_hybrid, ids)
    nonempty = phi[:, ids].sum(1) > 0
    assert nonempty.sum() < 118
    mean, variance = dense_prediction(ids, labels(ids), requests()[0], phi[nonempty][:, ids])
    got = memory.predict_hybrid(state, requests())
    assert got["rank"].tolist() == [int(nonempty.sum())]
    assert got["rank_cutoff"][0] > 0
    np.testing.assert_allclose(got["mean"][0], mean, atol=2e-12)
    np.testing.assert_allclose(got["variance"][0], variance, atol=2e-12)


@pytest.mark.parametrize("kind", ("spectral", "dct"))
@pytest.mark.parametrize("count", (119, 192))
@pytest.mark.parametrize("layout", ("dispersed", "clustered"))
def test_compressed_restricted_basis_prediction_matches_independent_row_space(kind, count, layout):
    ids = np.arange(count) if layout == "clustered" else (np.arange(count) * 73) % 289
    target = labels(ids) + .3 * np.cos(ids * np.sqrt(2))
    state = memory.initial_hybrid(1, kind)
    for index, value in zip(ids, target, strict=True):
        state = memory.write_hybrid(state, grid()[index][None], np.array([value]))
    got = memory.predict_hybrid(state, requests())
    ordered = np.sort(ids)
    operator = memory.basis(kind)[:, ordered]
    # Different LAPACK SVD driver; use raw projected labels, never invert saved sums.
    _, singular, row_space = svd(operator, full_matrices=False, lapack_driver="gesvd")
    cutoff = np.finfo(float).eps * max(operator.shape) * singular[0]
    selected_rows = row_space[singular > cutoff]
    mean, variance = dense_prediction(ordered, target[np.argsort(ids)], requests()[0], selected_rows)
    assert got["rank"].tolist() == [len(selected_rows)]
    np.testing.assert_allclose(got["mean"][0], mean, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(got["variance"][0], variance, rtol=1e-8, atol=1e-8)


def test_shared_coordinate_path_variance_and_noise_assimilated_once():
    state = memory.initial_hybrid(1)
    state = memory.write_hybrid(state, np.zeros((1, 2)), np.array([.7]))
    got = memory.predict_hybrid(state, np.zeros((1, 1, 4, 4, 2)))
    k = 1.00001
    np.testing.assert_allclose(got["mean"], .7 * k / (k + .09), atol=2e-15)
    np.testing.assert_allclose(got["variance"], k * .09 / (k + .09), atol=2e-15)
    # Four identical latent points do not average away the spatial white component.
    assert got["variance"][0, 0, 0] > k * .09 / (k + .09) / 4


def test_packed_coverage_ties_delete_lowest_id_and_preserve_sorted_labels():
    ids = np.arange(118, -1, -1)
    state = fill(memory.initial_packed, memory.write_packed, ids)
    # All points have a nearest neighbour at distance .25; ID0 arrived last and is removed.
    np.testing.assert_array_equal(selected(state.mask[0]), np.arange(1, 119))
    np.testing.assert_array_equal(state.values[0], labels(np.arange(1, 119)))
    got = memory.predict_packed(state, requests())
    mean, variance = dense_prediction(np.arange(1, 119), labels(np.arange(1, 119)), requests()[0])
    np.testing.assert_allclose(got["mean"][0], mean, atol=2e-13)
    np.testing.assert_allclose(got["variance"][0], variance, atol=2e-13)
    assert got["rank"].tolist() == [118]
    assert got["rank_cutoff"].tolist() == [0.]


def test_recent98_keeps_actual_last_observations_and_exact_retained_row_gp():
    ids = (np.arange(120) * 73) % 289
    state = fill(memory.initial_recent, memory.write_recent, ids)
    np.testing.assert_array_equal(state.ids[0], ids[-98:])
    np.testing.assert_array_equal(state.values[0], labels(ids[-98:]))
    got = memory.predict_recent(state, requests())
    mean, variance = dense_prediction(ids[-98:], labels(ids[-98:]), requests()[0])
    np.testing.assert_allclose(got["mean"][0], mean, atol=2e-13)
    np.testing.assert_allclose(got["variance"][0], variance, atol=2e-13)
    assert got["rank"].tolist() == [98]


def test_coverage98_ties_delete_oldest_not_lowest_grid_id():
    ids = np.arange(98, -1, -1)
    state = fill(memory.initial_recent, memory.write_coverage98, ids)
    # Uniform lattice ties: remove oldest ID98, retaining the newest ID0.
    np.testing.assert_array_equal(state.ids[0], np.arange(97, -1, -1))
    np.testing.assert_array_equal(state.values[0], labels(np.arange(97, -1, -1)))
    assert state.resident_bytes_per_context == 1020
    got = memory.predict_recent(state, requests())
    mean, variance = dense_prediction(ids[1:], labels(ids[1:]), requests()[0])
    np.testing.assert_allclose(got["mean"][0], mean, atol=2e-13)
    np.testing.assert_allclose(got["variance"][0], variance, atol=2e-13)


@pytest.mark.parametrize("initial,write,predict,expected_bytes", [
    (memory.initial_hybrid, memory.write_hybrid, memory.predict_hybrid, 1022),
    (memory.initial_packed, memory.write_packed, memory.predict_packed, 1021),
    (memory.initial_recent, memory.write_recent, memory.predict_recent, 1020),
    (memory.initial_recent, memory.write_coverage98, memory.predict_recent, 1020)])
def test_bytes_ownership_causal_public_boundary_and_batch_partition(initial, write, predict, expected_bytes):
    ids = np.arange(10)
    state = fill(initial, write, ids, batch=2)
    assert state.resident_bytes_per_context == expected_bytes <= 1024
    declared_arrays = [getattr(state, field.name) for field in fields(state)
                       if isinstance(getattr(state, field.name), np.ndarray)]
    assert len(declared_arrays) == 2
    assert state.array_bytes == sum(value.nbytes for value in declared_arrays)
    assert not hasattr(state, "__dict__")
    for value in declared_arrays:
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        state.step = 3
    assert set(inspect.signature(write).parameters) == {"state", "x", "y"}
    before = [a.copy() for a in declared_arrays]
    prediction = predict(state, requests(2))
    separate = predict(fill(initial, write, ids), requests())
    for key in prediction:
        np.testing.assert_array_equal(prediction[key][0], separate[key][0])
        np.testing.assert_array_equal(prediction[key][1], separate[key][0])
    for original, after in zip(before, declared_arrays, strict=True):
        np.testing.assert_array_equal(original, after)
    with pytest.raises(TypeError):
        write(state, np.zeros((2, 2)), np.zeros(2), future_paths=requests(2))


def test_constructor_owns_arrays_and_kind_is_bound():
    mask = np.zeros((1, 37), np.uint8)
    values = np.zeros((1, 118))
    state = memory.HybridMemory(mask, values, "bins")
    mask[0, 0] = 255
    values[0, 0] = 77
    assert not state.mask.any() and not state.values.any()
    with pytest.raises(FrozenInstanceError):
        state.kind = "spectral"
    with pytest.raises(TypeError):
        memory.predict_hybrid(state, requests(), kind="dct")


@pytest.mark.parametrize("initial,write", [(memory.initial_hybrid, memory.write_hybrid),
                                          (memory.initial_packed, memory.write_packed),
                                          (memory.initial_recent, memory.write_recent)])
def test_duplicate_retained_event_and_invalid_public_values_fail_without_mutation(initial, write):
    state = write(initial(1), grid()[3:4], np.array([.2]))
    old = state.values.copy()
    with pytest.raises(ValueError, match="repeated"):
        write(state, grid()[3:4], np.array([.8]))
    for x, y in ((np.array([[2.25, 0.]]), np.array([.1])),
                 (np.array([[.1, 0.]]), np.array([.1])),
                 (grid()[5:6], np.array([np.nan])),
                 (grid()[5:6].astype(np.float32), np.array([.1]))):
        with pytest.raises(ValueError):
            write(state, x, y)
    np.testing.assert_array_equal(state.values, old)


def test_post_conversion_mask_keeps_preconversion_history_for_duplicate_rejection():
    ids = (np.arange(120) * 73) % 289
    state = fill(lambda batch: memory.initial_hybrid(batch, "bins"), memory.write_hybrid, ids)
    with pytest.raises(ValueError, match="repeated observed"):
        memory.write_hybrid(state, grid()[ids[:1]], np.array([.4]))


@pytest.mark.parametrize("damage", ["bits", "population", "padding", "kind", "hyper", "counter"])
def test_malformed_state_is_rejected(damage):
    mask, values = np.zeros((1, 37), np.uint8), np.zeros((1, 118))
    kwargs = {}
    if damage == "bits":
        mask[0, -1] = 2
    elif damage == "population":
        mask[0, 0] = 1
    elif damage == "padding":
        values[0, 1] = 1
    elif damage == "kind":
        kwargs["kind"] = "adaptive"
    elif damage == "hyper":
        kwargs["noise_variance"] = .1
    else:
        kwargs["step"] = True
    with pytest.raises(ValueError):
        memory.HybridMemory(mask, values, **kwargs)


def test_empty_prediction_and_wrong_path_dtype_fail():
    with pytest.raises(ValueError, match="nonempty state"):
        memory.predict_hybrid(memory.initial_hybrid(1), requests())
    state = memory.write_hybrid(memory.initial_hybrid(1), grid()[:1], np.zeros(1))
    with pytest.raises(ValueError, match="declared array"):
        memory.predict_hybrid(state, requests().astype(np.float32))
    wrong = requests()
    wrong[0, 0, 0, 0] = [.1, 0.]
    with pytest.raises(ValueError, match="exactly on grid"):
        memory.predict_hybrid(state, wrong)
