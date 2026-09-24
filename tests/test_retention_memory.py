"""Fabricated independent Gaussian, projection and storage witnesses only."""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from openjev.research import retention_memory as memory


def covariance(left, right):
    """Scalar kernel reference, with a shared nugget at equal coordinates."""
    return np.array([[math.exp(-sum((float(a)-float(b))**2 for a, b in zip(x, z, strict=True))/2.)
                      + (1e-5 if all(a == b for a, b in zip(x, z, strict=True)) else 0.)
                      for z in right] for x in left])


def fixture(batch=2, count=8):
    points = np.array([[-1.5, -1.], [-.5, .5], [.5, -.5], [1.5, 1.],
                       [-1., 1.5], [0., 0.], [1., -1.5], [1.5, -.5], [-1.5, 1.5]])[:count]
    x = np.stack([points + np.array([b/8, -b/16]) for b in range(batch)])
    y = np.stack([np.array([.5, -.25, .75, 1., -.5, .125, .25, -.75, .3])[:count]+b/5
                  for b in range(batch)])
    return x, y


def assimilate(x, y):
    state = memory.initial(x.shape[0], x.shape[1])
    for index in range(x.shape[1]):
        state = memory.expand(state, x[:, index], y[:, index])
    return state


def dense(x, y, points):
    cross = covariance(points, x)
    observed = covariance(x, x) + .09*np.eye(len(x))
    return cross @ np.linalg.solve(observed, y), covariance(points, points)-cross@np.linalg.solve(observed, cross.T)


def test_kernel_exact_coordinate_nugget_and_batched_partition():
    x = np.array([[0., 0.], [0., 0.], [1., 0.], [np.nextafter(0., 1.), 0.]])
    result = memory.kernel(x, x)
    np.testing.assert_array_equal(result, covariance(x, x))
    assert result[0, 1] == 1.00001
    assert result[0, 3] == 1.
    batched = memory.kernel(np.stack([x, x+.5]), np.stack([x, x+.5]))
    np.testing.assert_array_equal(batched[0], result)
    np.testing.assert_array_equal(batched[1], memory.kernel(x+.5, x+.5))


def test_empty_prior_has_no_observations_or_reserved_unused_arrays():
    state = memory.initial(2, 8)
    assert state.step == 0 and state.array_bytes == 0 and state.resident_bytes_per_context == 40
    assert set(vars(state)) == {"Z", "mean", "cov", "step", "length", "amplitude", "nugget", "noise_variance"}
    points, _ = fixture(2, 3)
    result = memory.predict(state, points)
    np.testing.assert_array_equal(result["mean"], np.zeros((2, 3)))
    np.testing.assert_array_equal(result["cov"], memory.kernel(points, points))


def test_one_observation_exact_scalar_conditioning():
    state = memory.expand(memory.initial(1, 8), np.array([[0., 0.]]), np.array([2.]))
    k, noise = 1.00001, .09
    assert state.mean[0, 0] == pytest.approx(2*k/(k+noise), rel=1e-14)
    assert state.cov[0, 0, 0] == pytest.approx(k*noise/(k+noise), rel=1e-14)
    assert state.step == 1
    predicted = memory.predict(state, np.array([[[0., 0.], [1., 0.]]]))
    assert predicted["cov"][0, 0, 0] == pytest.approx(state.cov[0, 0, 0], abs=1e-14)


@pytest.mark.parametrize("count", [1, 2, 5, 8])
def test_no_compression_equals_full_batch_gp_at_every_query(count):
    x, y = fixture(2, count)
    state = assimilate(x, y)
    queries = np.array([[[-.75, -.25], [.3, .7], [1.5, -1.]],
                        [[0., 0.], [.5, .5], [-1., -1.]]])
    actual = memory.predict(state, queries)
    for b in range(2):
        mean, cov = dense(x[b], y[b], queries[b])
        np.testing.assert_allclose(actual["mean"][b], mean, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(actual["cov"][b], cov, rtol=1e-11, atol=1e-12)
        mean, cov = dense(x[b], y[b], x[b])
        np.testing.assert_allclose(state.mean[b], mean, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(state.cov[b], cov, rtol=1e-11, atol=1e-12)


def test_compression_preserves_marginal_and_restores_correlated_prior_residual():
    state = assimilate(*fixture(2, 4))
    drop = np.array([1, 3], dtype=np.int64)
    compressed = memory.compress(state, drop)
    for b in range(2):
        keep = [i for i in range(4) if i != drop[b]]
        np.testing.assert_array_equal(compressed.Z[b], state.Z[b, keep])
        np.testing.assert_array_equal(compressed.mean[b], state.mean[b, keep])
        np.testing.assert_array_equal(compressed.cov[b], state.cov[b][np.ix_(keep, keep)])
    assert compressed.step == state.step
    query = memory.predict(compressed, state.Z)
    for b in range(2):
        prior = covariance(compressed.Z[b], compressed.Z[b])
        a = np.linalg.solve(prior, covariance(compressed.Z[b], state.Z[b])).T
        expected = covariance(state.Z[b], state.Z[b])-a@prior@a.T+a@compressed.cov[b]@a.T
        np.testing.assert_allclose(query["mean"][b], a@compressed.mean[b], atol=1e-12)
        np.testing.assert_allclose(query["cov"][b], expected, atol=1e-12)
    np.testing.assert_allclose(memory.predict(compressed, compressed.Z)["cov"], compressed.cov, atol=1e-12)


def independent_deletion_kl(z, mean, cov, removed):
    """Chain-rule KL, using scalar conditionals rather than inverse diagonals."""
    prior = covariance(z, z)
    keep = [j for j in range(len(z)) if j != removed]
    if keep:
        rr = np.ix_(keep, keep)
        prior_slope = np.linalg.solve(prior[rr], prior[keep, removed])
        posterior_slope = np.linalg.solve(cov[rr], cov[keep, removed])
        r = prior[removed, removed]-prior[removed, keep]@prior_slope
        v = cov[removed, removed]-cov[removed, keep]@posterior_slope
        error_mean = mean[removed]-prior_slope@mean[keep]
        slope_difference = posterior_slope-prior_slope
        error_variance = slope_difference@cov[rr]@slope_difference
    else:
        r, v, error_mean, error_variance = prior[0, 0], cov[0, 0], mean[0], 0.
    result = .5*(math.log(r/v)+(v+error_mean**2+error_variance)/r-1)
    features = [z[removed, 0]/2, z[removed, 1]/2, error_mean/math.sqrt(r), math.log(v/r),
                math.log1p(result), cov[removed, removed]/prior[removed, removed]]
    return result, features


@pytest.mark.parametrize("count", [1, 2, 5])
def test_forward_kl_and_all_six_features_match_conditional_gaussians(count):
    state = assimilate(*fixture(2, count))
    result = memory.deletion_statistics(state)
    assert set(result) == {"kl", "features", "negative_kl_floor_count"}
    assert type(result["negative_kl_floor_count"]) is int
    for b in range(2):
        for removed in range(count):
            kl, features = independent_deletion_kl(state.Z[b], state.mean[b], state.cov[b], removed)
            assert result["kl"][b, removed] == pytest.approx(kl, rel=1e-10, abs=1e-12)
            np.testing.assert_allclose(result["features"][b, removed], features, rtol=1e-10, atol=1e-12)


def test_prior_state_has_zero_deletion_cost_with_declared_roundoff_count():
    x, _ = fixture(2, 4)
    state = memory.State(x, np.zeros((2, 4)), memory.kernel(x, x), step=4)
    result = memory.deletion_statistics(state)
    np.testing.assert_allclose(result["kl"], 0., atol=1e-12)
    assert np.all(result["kl"] >= 0)
    assert 0 <= result["negative_kl_floor_count"] <= 8


def test_projection_does_not_reapply_an_old_likelihood():
    x, y = fixture(1, 4)
    before = assimilate(x[:, :3], y[:, :3])
    compressed = memory.compress(before, np.array([0], dtype=np.int64))
    old_joint = memory.predict(compressed, np.concatenate((compressed.Z, x[:, 3:4]), axis=1))
    mean, cov = old_joint["mean"][0], old_joint["cov"][0]
    c, d = cov[:, -1], cov[-1, -1]+.09
    expected_mean = mean+c*(y[0, 3]-mean[-1])/d
    expected_cov = cov-np.outer(c, c)/d
    after = memory.expand(compressed, x[:, 3], y[:, 3])
    assert after.step == 4 and after.Z.shape[1] == 3
    np.testing.assert_allclose(after.mean[0], expected_mean, atol=1e-12)
    np.testing.assert_allclose(after.cov[0], expected_cov, atol=1e-12)


def test_joint_psd_path_variance_uses_offdiagonal_and_equal_location_covariance():
    state = memory.compress(assimilate(*fixture(2, 5)), np.array([0, 2], dtype=np.int64))
    queries = np.array([[[.2, .1], [.3, .1], [.2, .1], [.4, .2]],
                        [[-.7, .8], [-.6, .9], [-.7, .8], [-.5, .7]]])
    result = memory.predict(state, queries)
    weights = np.array([.25, .25, .25, .25])
    for b in range(2):
        cov = result["cov"][b]
        assert np.linalg.eigvalsh(cov).min() >= -1e-12
        assert weights@cov@weights > (weights**2)@np.diag(cov)
        assert cov[0, 2] == pytest.approx(cov[0, 0], abs=1e-14)
        np.testing.assert_allclose(cov[0], cov[2], atol=1e-14)


def test_batch_partition_and_dictionary_permutation_equivariance():
    x, y = fixture(2, 5)
    state = assimilate(x, y)
    stats = memory.deletion_statistics(state)
    order = np.array([3, 0, 4, 2, 1])
    permuted = memory.State(state.Z[:, order], state.mean[:, order],
                            state.cov[:, order][:, :, order], step=state.step)
    permuted_stats = memory.deletion_statistics(permuted)
    np.testing.assert_allclose(permuted_stats["kl"], stats["kl"][:, order], atol=1e-12)
    query = x[:, :3]+.13
    whole = memory.predict(state, query)
    shuffled = memory.predict(permuted, query)
    for name in whole:
        np.testing.assert_allclose(shuffled[name], whole[name], atol=1e-12)
    for b in range(2):
        one = assimilate(x[b:b+1], y[b:b+1])
        for name in ("Z", "mean", "cov"):
            np.testing.assert_allclose(getattr(one, name)[0], getattr(state, name)[b], atol=1e-12)


def test_state_owned_immutable_query_no_write_and_no_future_target_api():
    x, y = fixture(2, 3)
    source_x, source_y = x.copy(), y.copy()
    state = assimilate(x, y)
    snapshot = {name: getattr(state, name).copy() for name in ("Z", "mean", "cov")}
    query = x[:, :2]+.1
    first = memory.predict(state, query)
    first["mean"][:] = 123.
    again = memory.predict(state, query)
    assert not np.any(again["mean"] == 123.)
    for name, value in snapshot.items():
        np.testing.assert_array_equal(getattr(state, name), value)
        with pytest.raises(ValueError):
            getattr(state, name).setflags(write=True)
    np.testing.assert_array_equal(x, source_x)
    np.testing.assert_array_equal(y, source_y)
    with pytest.raises(TypeError, match="target"):
        memory.predict(state, query, target=np.zeros(2))
    assert list(inspect.signature(memory.predict).parameters) == ["state", "points"]


@pytest.mark.parametrize("size,bytes_per_context", [(8, 744), (9, 904)])
def test_actual_state_storage_and_policy_budget(size, bytes_per_context):
    state = assimilate(*fixture(3, size))
    assert state.array_bytes == 3*8*(size*2+size+size*size)
    assert state.resident_bytes_per_context == bytes_per_context
    if size == 8:
        assert state.resident_bytes_per_context+33*8 == 1008
        assert state.resident_bytes_per_context+33*8 < 1024


def test_repeated_retained_point_rejected_without_input_mutation():
    state = assimilate(*fixture(2, 3))
    before = state.cov.tobytes()
    with pytest.raises(ValueError, match="repeats"):
        memory.expand(state, state.Z[:, 1].copy(), np.array([0., 1.]))
    assert state.cov.tobytes() == before and state.step == 3


@pytest.mark.parametrize("batch,capacity", [(0, 8), (1, 0), (True, 8), (1, 8.), (1, -1)])
def test_invalid_initial_dimensions(batch, capacity):
    with pytest.raises(ValueError):
        memory.initial(batch, capacity)


@pytest.mark.parametrize("drop", [np.array([0, 1], dtype=np.int32), np.array([-1, 0], dtype=np.int64),
                                  np.array([3, 0], dtype=np.int64), np.array([0], dtype=np.int64)])
def test_invalid_drop_rejected(drop):
    state = assimilate(*fixture(2, 3))
    with pytest.raises(ValueError, match="drop"):
        memory.compress(state, drop)


@pytest.mark.parametrize("bad", [np.array([[np.nan, 0.]]), np.array([[0., np.inf]]),
                                 np.zeros((1, 2), dtype=np.float32), np.zeros((2, 2))])
def test_invalid_new_coordinates_rejected(bad):
    with pytest.raises(ValueError):
        memory.expand(memory.initial(1, 8), bad, np.zeros(1))


def test_invalid_state_covariance_scalars_and_empty_deletion_rejected():
    x, _ = fixture(1, 2)
    with pytest.raises(ValueError, match="positive definite"):
        memory.State(x, np.zeros((1, 2)), np.zeros((1, 2, 2)), step=2)
    with pytest.raises(ValueError, match="symmetric"):
        memory.State(x, np.zeros((1, 2)), np.array([[[1., .2], [.3, 1.]]]), step=2)
    with pytest.raises(ValueError, match="fixed"):
        memory.State(x, np.zeros((1, 2)), np.eye(2)[None], step=2, noise_variance=.1)
    with pytest.raises(ValueError, match="nonempty"):
        memory.deletion_statistics(memory.initial(1, 8))


def test_compress_single_point_to_empty_preserves_event_count():
    state = memory.expand(memory.initial(1, 1), np.zeros((1, 2)), np.ones(1))
    empty = memory.compress(state, np.zeros(1, dtype=np.int64))
    assert empty.step == 1 and empty.Z.shape == (1, 0, 2) and empty.array_bytes == 0
    np.testing.assert_array_equal(memory.predict(empty, np.zeros((1, 1, 2)))["mean"], [[0.]])
