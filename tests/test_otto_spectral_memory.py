"""Toy public-kernel qualification; no OTTO imports, episodes or saved data."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.special import gammainc

from openjev.research import otto_spectral_memory as M


def kernel(size=5, *, tail_zeros=True):
    grid = np.indices((2 * size + 1, 2 * size + 1)) - size
    distance = np.sqrt(np.sum(grid**2, axis=0))
    mu = 1.4 * np.exp(-distance / 2)
    zero = np.exp(-mu)
    one = mu * zero
    two = mu * one / 2
    tail = gammainc(3, mu)
    result = np.stack((zero, one, two, tail))
    if tail_zeros:
        excluded = distance >= size - 1
        result[0, excluded] += result[3, excluded]
        result[3, excluded] = 0
    result[:, size, size] = 0
    return result


def packet(size, position, hit, step=0, done=False):
    return {"position": list(position), "hit": hit, "done": done, "step": step,
            "valid_actions": [] if done else [a for a in range(4)
                if 0 <= position[a // 2] + (-1 if a % 2 == 0 else 1) < size]}


def initial(model, hit=1):
    return packet(model.N, model.center, hit)


def product_prior(model, hit):
    # Independent probability-domain initialization under a uniform noncenter prior.
    p = model.kernel[hit, model.N - model.center[0]:2 * model.N - model.center[0],
                     model.N - model.center[1]:2 * model.N - model.center[1]].copy()
    p /= p.sum()
    return p


def product_update(model, p, position, hit):
    p = p.copy()
    p[position] = 0
    x, y = position
    p *= model.kernel[hit, model.N - x:2 * model.N - x, model.N - y:2 * model.N - y]
    assert p.sum() > 0
    p /= p.sum()
    return p


@pytest.mark.parametrize("size", [3, 5])
@pytest.mark.parametrize("hit", [1, 2, 3])
def test_shared_initial_prior_once_without_observation_evidence(size, hit):
    model = M.SpectralModel(kernel(size))
    expected = product_prior(model, hit)
    for actor in (model.start_exact(initial(model, hit)), model.start(initial(model, hit), 1),
                  model.start(initial(model, hit), size, extension="nearest")):
        logs, p = actor.decode()
        np.testing.assert_allclose(p, expected, atol=2e-16, rtol=1e-14)
        np.testing.assert_allclose(np.exp(logs), p, atol=0, rtol=0)
        assert np.all(actor._values == 0)
        assert not np.isfinite(logs[model.center])


@pytest.mark.parametrize("initial_hit", [1, 2, 3])
def test_full_rank_both_fills_and_exact_match_probability_products_every_prefix(initial_hit):
    model = M.SpectralModel(kernel())
    actors = [model.start_exact(initial(model, initial_hit))]
    actors += [model.start(initial(model, initial_hit), 5, extension=fill) for fill in ("neutral", "nearest")]
    p = product_prior(model, initial_hit)
    history = [((1, 2), 0), ((1, 1), 3), ((2, 1), 2), ((3, 1), 1), ((3, 2), 0)]
    for step, (position, hit) in enumerate(history, 1):
        p = product_update(model, p, position, hit)
        for actor in actors:
            actor.update(packet(5, position, hit, step))
            logs, actual = actor.decode()
            np.testing.assert_allclose(actual, p, atol=2e-14, rtol=1e-12)
            assert np.all(actual[actor.excluded] == 0)
            assert np.all(np.isneginf(logs[actor.excluded]))
            assert np.all(np.isfinite(logs[~actor.excluded]))
        np.testing.assert_array_equal(actors[0].excluded, actors[1].excluded)
        np.testing.assert_array_equal(actors[0].excluded, actors[2].excluded)


def test_zero_reading_updates_and_initial_hit_is_not_replayed():
    model = M.SpectralModel(kernel())
    actor = model.start(initial(model, 2), 5)
    actor.update(packet(5, (1, 2), 0, 1))
    expected = product_update(model, product_prior(model, 2), (1, 2), 0)
    np.testing.assert_allclose(actor.belief(), expected, atol=1e-14, rtol=0)
    assert np.any(actor.coefficients != 0)
    assert actor.initial_hit == 2 and actor.step == 1


def test_numerical_tail_zeros_and_visited_cells_persist_in_single_mask():
    model = M.SpectralModel(kernel())
    actor = model.start(initial(model), 2)
    actor.update(packet(5, (1, 2), 3, 1))
    expected = ~np.isfinite(model.initial_log_priors[0]) | (model.likelihood((1, 2), 3) == 0)
    expected[1, 2] = True
    np.testing.assert_array_equal(actor.excluded, expected)
    actor.update(packet(5, (1, 1), 0, 2))
    expected[1, 1] = True
    np.testing.assert_array_equal(actor.excluded, expected)
    assert np.all(actor.belief()[expected] == 0)


def test_low_rank_extension_sensitivity_is_preserved_not_selected_away():
    model = M.SpectralModel(kernel())
    neutral = model.start(initial(model), 2, extension="neutral")
    nearest = model.start(initial(model), 2, extension="nearest")
    for actor in (neutral, nearest):
        actor.update(packet(5, (1, 2), 3, 1))
    np.testing.assert_array_equal(neutral.excluded, nearest.excluded)
    assert np.max(np.abs(neutral.belief() - nearest.belief())) > 1e-6


@pytest.mark.parametrize("fill", ["neutral", "nearest"])
def test_transform_matches_explicit_orthonormal_cosine_basis_and_truncation(fill):
    model = M.SpectralModel(kernel())
    position, hit = (1, 2), 3
    probability = model.likelihood(position, hit)
    increment = np.full((5, 5), 0.0 if fill == "neutral" else np.log(model.kernel[hit, 6, 5]), dtype=np.float64)
    increment[probability > 0] = np.log(probability[probability > 0])
    ks, xs = np.arange(5)[:, None], np.arange(5)[None, :]
    basis = np.cos(np.pi * ks * (xs + .5) / 5) * np.sqrt(2 / 5)
    basis[0] /= np.sqrt(2)
    explicit = basis @ increment @ basis.T
    full = model.start(initial(model), 5, extension=fill)
    small = model.start(initial(model), 2, extension=fill)
    for actor in (full, small):
        actor.update(packet(5, position, hit, 1))
    np.testing.assert_allclose(full.coefficients, explicit, atol=2e-14, rtol=1e-14)
    np.testing.assert_allclose(small.coefficients, explicit[:2, :2], atol=2e-14, rtol=1e-14)
    assert np.sum(full.coefficients**2) == pytest.approx(np.sum(increment**2), rel=1e-14)
    assert np.sum(small.coefficients**2) <= np.sum(full.coefficients**2)


def test_adding_a_constant_field_changes_no_normalized_probability():
    model = M.SpectralModel(kernel())
    for actor in (model.start_exact(initial(model)), model.start(initial(model), 2)):
        actor.update(packet(5, (1, 2), 1, 1))
        before = actor.belief()
        values = actor._values.copy()
        if isinstance(actor, M.SpectralMemory):
            values[0, 0] += 7 * model.N
        else:
            values += 7
        actor._values = M._immutable(values, np.float64)
        np.testing.assert_allclose(actor.belief(), before, atol=1e-15, rtol=1e-13)


def test_transform_calls_force_one_worker_and_exact_uses_none(monkeypatch):
    recorded = []
    original_dct, original_idct = M.dctn, M.idctn

    def dct(*args, **kwargs):
        recorded.append(("dct", kwargs))
        return original_dct(*args, **kwargs)

    def idct(*args, **kwargs):
        recorded.append(("idct", kwargs))
        return original_idct(*args, **kwargs)

    monkeypatch.setattr(M, "dctn", dct)
    monkeypatch.setattr(M, "idctn", idct)
    model = M.SpectralModel(kernel())
    spectral = model.start(initial(model), 2)
    spectral.update(packet(5, (1, 2), 0, 1))
    spectral.decode()
    assert recorded == [("dct", {"type": 2, "norm": "ortho", "workers": 1}),
                        ("idct", {"type": 2, "norm": "ortho", "workers": 1})]
    recorded.clear()
    exact = model.start_exact(initial(model))
    exact.update(packet(5, (1, 2), 0, 1))
    exact.decode()
    assert recorded == []


def test_terminal_is_not_encoded_and_fresh_start_resets_every_state():
    model = M.SpectralModel(kernel())
    actor = model.start(initial(model), 2)
    actor.update(packet(5, (1, 2), 0, 1))
    coefficients, excluded = actor.coefficients.copy(), actor.excluded.copy()
    actor.update(packet(5, (1, 1), -2, 2, True))
    assert actor.done is True and actor.step == 2 and actor.position == (1, 1)
    np.testing.assert_array_equal(actor.coefficients, coefficients)
    np.testing.assert_array_equal(actor.excluded, excluded)
    with pytest.raises(ValueError, match="termination"):
        actor.update(packet(5, (2, 1), 0, 3))
    with pytest.raises(ValueError, match="termination"):
        actor.decode()
    fresh = model.start(initial(model, 3), 2)
    assert not fresh.done and fresh.step == 0 and np.all(fresh.coefficients == 0)
    np.testing.assert_array_equal(fresh.excluded, ~np.isfinite(model.initial_log_priors[2]))


@pytest.mark.parametrize("change", ["source", "seed", "hit_bool", "step_bool", "step_repeat", "jump", "interior_stay", "sentinel", "actions", "found_excluded"])
def test_invalid_packets_do_not_mutate_state(change):
    model = M.SpectralModel(kernel())
    actor = model.start(initial(model), 2)
    value = packet(5, (1, 2), 0, 1)
    if change in ("source", "seed"):
        value[change] = 2
    elif change == "hit_bool":
        value["hit"] = True
    elif change == "step_bool":
        value["step"] = True
    elif change == "step_repeat":
        value["step"] = 0
    elif change == "jump":
        value = packet(5, (0, 0), 0, 1)
    elif change == "interior_stay":
        value = packet(5, (2, 2), 0, 1)
    elif change == "sentinel":
        value["hit"] = -2
    elif change == "actions":
        value["valid_actions"] = [1, 2, 3]
    else:
        actor.update(packet(5, (1, 2), 0, 1))
        value = packet(5, (2, 2), -2, 2, True)
    before = (actor.coefficients.copy(), actor.excluded.copy(), actor.position, actor.step, actor.done)
    with pytest.raises((ValueError, TypeError)):
        actor.update(value)
    np.testing.assert_array_equal(actor.coefficients, before[0])
    np.testing.assert_array_equal(actor.excluded, before[1])
    assert (actor.position, actor.step, actor.done) == before[2:]


def test_boundary_noop_is_valid_public_evidence_not_a_reset():
    model = M.SpectralModel(kernel(3))
    actor = model.start(initial(model), 1)
    actor.update(packet(3, (0, 1), 0, 1))
    first = actor.coefficients.copy()
    actor.update(packet(3, (0, 1), 0, 2))
    np.testing.assert_allclose(actor.coefficients, first * 2, atol=0, rtol=0)


@pytest.mark.parametrize("rank", [0, -1, True, 1.5, 6])
def test_rank_validation(rank):
    model = M.SpectralModel(kernel())
    with pytest.raises(ValueError):
        model.start(initial(model), rank)


@pytest.mark.parametrize("defect", ["negative", "nonfinite", "origin", "mass", "empty_initial", "shape"])
def test_kernel_validation(defect):
    value = kernel()
    if defect == "negative":
        value[0, 0, 0] = -1
    elif defect == "nonfinite":
        value[0, 0, 0] = np.nan
    elif defect == "origin":
        value[:, 5, 5] = .25
    elif defect == "mass":
        value[0, 0, 0] = 0
    elif defect == "empty_initial":
        value[0] += value[2]
        value[2] = 0
    else:
        value = value[:, :-1, :-1]
    with pytest.raises(ValueError):
        M.SpectralModel(value)


def test_empty_support_rejected_without_mutating_state():
    value = kernel(3)
    # At the next arrival, make category3 possible only at the already visited initial center.
    value[0] += value[3]
    value[3] = 0
    for coordinate in ((2, 3), (4, 3)):
        value[3][coordinate] = .001
        value[0][coordinate] -= .001
    model = M.SpectralModel(value)
    actor = model.start(initial(model), 1)
    # Both one-step vertical support locations from arrival (0,1) are center or off-grid.
    before = actor.coefficients.copy()
    with pytest.raises(ValueError, match="every source"):
        actor.update(packet(3, (0, 1), 3, 1))
    np.testing.assert_array_equal(actor.coefficients, before)
    assert actor.step == 0


def test_shared_arrays_and_state_are_immutable_and_exactly_accounted():
    input_kernel = kernel()
    model = M.SpectralModel(input_kernel)
    actor = model.start(initial(model), 2)
    input_kernel[:] = 0
    assert model.kernel.sum() > 0
    for array in (model.kernel, model.initial_log_priors, actor.coefficients, actor.excluded):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        model.N = 7
    storage = actor.storage_bytes()
    assert storage["evidence_array_bytes"] == 2 * 2 * 8
    assert storage["support_mask_bytes"] == 25
    assert storage["state_array_bytes"] == 57
    assert storage["shared_array_bytes"] == 4 * 11 * 11 * 8 + 3 * 5 * 5 * 8
    assert not hasattr(actor, "__dict__")
    assert set(M._Memory.__slots__) == {"model", "q", "extension", "_values", "_excluded", "initial_hit", "position", "step", "done"}
    before = (id(actor.coefficients), id(actor.excluded))
    logs, p = actor.decode()
    p[:] = 0
    logs[:] = 0
    assert before == (id(actor.coefficients), id(actor.excluded))
    assert actor.belief().sum() == pytest.approx(1)


def test_underflowed_probability_has_finite_supported_log_without_floor():
    model = M.SpectralModel(kernel())
    actor = model.start_exact(initial(model))
    values = np.zeros((5, 5), dtype=np.float64)
    values[0, 0] = -1000
    actor._values = M._immutable(values, np.float64)
    logs, p = actor.decode()
    assert not actor.excluded[0, 0] and np.isfinite(logs[0, 0]) and p[0, 0] == 0
    assert p.sum() == pytest.approx(1)
