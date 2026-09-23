"""Fabricated public arrays and callbacks only; no TF/model/native calls."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research.otto_exact_value_cache import INPUT_SHAPE, ExactValueCache, build_inputs


class View:
    Nactions, Nhits, NN_input_shape = 4, 4, (105, 105)

    def __init__(self, belief, kernel, position):
        self.p_source, self.p_Poisson, self.agent = belief, kernel, list(position)

    def _move(self, action, agent):
        result = list(agent)
        axis, delta = action // 2, (-1, 1)[action % 2]
        valid = 0 <= result[axis] + delta < 53
        result[axis] += delta if valid else 0
        return result, valid

    def _extract_N_from_2N(self, input, origin):
        x, y = origin
        return input[:, 53-x:106-x, 53-y:106-y]

    def _centeragent(self, probability, agent):
        x, y = agent
        return np.pad(probability, ((52-x, x), (52-y, y)))


def public_arrays():
    x, y = np.indices((53, 53))
    belief = (1 + (17*x + 29*y + 7*x*y) % 97).astype(np.float64)
    belief /= belief.sum()
    a, b = np.indices((107, 107))
    factor = (1 + (a + 3*b) % 7).astype(np.float64) / 7
    kernel = np.stack([factor * ((h + 1) / 10) for h in range(4)])
    kernel[:, 53, 53] = 0
    return belief, kernel


def oracle_inputs(belief, kernel, position):
    """Independent scalar crop/product and explicit embedding, no view helpers."""
    result = np.zeros(INPUT_SHAPE, dtype=np.float32)
    masses = np.zeros((4, 4), dtype=np.float32)
    x, y = position
    successors = ((max(0, x-1), y), (min(52, x+1), y),
                  (x, max(0, y-1)), (x, min(52, y+1)))
    for action, (qx, qy) in enumerate(successors):
        for hit in range(4):
            joint = np.empty((53, 53), dtype=np.float64)
            for sx in range(53):
                for sy in range(53):
                    joint[sx, sy] = belief[sx, sy] * kernel[hit, 53+sx-qx, 53+sy-qy]
            mass = max(np.float64(1e-10), np.sum(joint))
            result[action*4+hit, 52-qx:105-qx, 52-qy:105-qy] = joint / mass
            masses[action, hit] = mass
    return result, masses


@pytest.mark.parametrize("position", [(26, 26), (0, 0), (52, 52), (0, 26)])
def test_source_exact_branch_geometry_order_and_nonmutation(position):
    belief, kernel = public_arrays()
    view = View(belief, kernel, position)
    before = belief.tobytes(), kernel.tobytes(), list(view.agent)
    inputs, masses = build_inputs(view)
    expected, expected_masses = oracle_inputs(belief, kernel, position)
    assert inputs.dtype == masses.dtype == np.float32
    assert inputs.shape == (16, 105, 105) and masses.shape == (4, 4)
    assert inputs.tobytes() == expected.tobytes()
    assert masses.tobytes() == expected_masses.tobytes()
    assert before == (belief.tobytes(), kernel.tobytes(), view.agent)
    inputs.fill(0)
    masses.fill(0)
    assert before == (belief.tobytes(), kernel.tobytes(), view.agent)


def test_zero_subfloor_and_exact_floor_preserve_original_mass_without_repair():
    kernel = np.full((4, 107, 107), .25, dtype=np.float64)
    kernel[:, 53, 53] = 0
    for branch_mass, expected_sum in ((0., 0.), (.5e-10, .5), (1e-10, 1.), (2e-10, 1.)):
        belief = np.zeros((53, 53), dtype=np.float64)
        belief[7, 9] = 4 * branch_mass
        inputs, masses = build_inputs(View(belief, kernel, (26, 26)))
        assert np.all(masses == np.float32(max(1e-10, branch_mass)))
        assert np.all(inputs.sum(axis=(1, 2)) == np.float32(expected_sum))
        assert inputs.tobytes() == oracle_inputs(belief, kernel, (26, 26))[0].tobytes()


def test_input_builder_rejects_bad_public_shapes_dtypes_and_domain():
    belief, kernel = public_arrays()
    for bad in (belief.astype(np.float32), belief[:52], np.full_like(belief, np.nan), -belief):
        with pytest.raises(ValueError, match="posterior"):
            build_inputs(View(bad, kernel, (26, 26)))
    for bad in (kernel.astype(np.float32), kernel[:3], np.full_like(kernel, np.inf), -kernel):
        with pytest.raises(ValueError, match="kernel"):
            build_inputs(View(belief, bad, (26, 26)))
    for position in ((-1, 0), (53, 0), (True, 0), (.5, 0)):
        with pytest.raises(ValueError, match="position"):
            build_inputs(View(belief, kernel, position))
    with pytest.raises(ValueError, match="geometry"):
        build_inputs(SimpleNamespace(Nactions=5, Nhits=4, NN_input_shape=(105, 105)))


def test_identical_model_inputs_keep_different_branch_masses_outside_cache():
    kernel = np.full((4, 107, 107), .25, dtype=np.float64)
    kernel[:, 53, 53] = 0
    belief = np.zeros((53, 53), dtype=np.float64)
    belief[7, 9] = .5
    first, masses = build_inputs(View(belief, kernel, (26, 26)))
    second, other_masses = build_inputs(View(belief * 2, kernel, (26, 26)))
    assert first.tobytes() == second.tobytes()
    assert np.all(other_masses == masses * 2)
    cache = ExactValueCache()

    def only_once(_):
        assert cache.statistics()["misses"] == 1
        return values(4)

    saved = cache.get_or_compute(first, only_once)
    assert saved.tobytes() == cache.get_or_compute(second, only_once).tobytes()
    assert cache.statistics()["hits"] == 1 and cache.statistics()["misses"] == 1
    assert np.all(other_masses == masses * 2)


def inputs(value=0.):
    result = np.zeros(INPUT_SHAPE, dtype=np.float32)
    result[0, 0, 0] = value
    return result


def values(value=0.):
    return np.full((16, 1), value, dtype=np.float32)


def test_complete_byte_keys_distinguish_nextafter_signed_zero_and_accept_noncontiguous():
    cache = ExactValueCache(4)
    calls = []

    def compute(x):
        calls.append(x.tobytes())
        return values(len(calls))

    zero = inputs()
    negative_zero = zero.copy()
    negative_zero[0, 0, 1] = np.float32(-0.)
    adjacent = inputs(np.nextafter(np.float32(0), np.float32(1)))
    for x in (zero, negative_zero, adjacent):
        cache.get_or_compute(x, compute)
    noncontiguous = np.zeros((16, 105, 210), dtype=np.float32)[:, :, ::2]
    assert not noncontiguous.flags.c_contiguous
    assert cache.get_or_compute(noncontiguous, compute).tobytes() == values(1).tobytes()
    assert len(calls) == 3
    assert cache.statistics()["hits"] == 1 and cache.statistics()["misses"] == 3


def test_lru_order_eviction_and_clear_preserve_counters_but_end_episode_reuse():
    cache = ExactValueCache(2)
    calls = []

    def compute(x):
        calls.append(float(x[0, 0, 0]))
        return values(x[0, 0, 0])

    for value in (1, 2, 1, 3, 1, 2):
        assert np.all(cache.get_or_compute(inputs(value), compute) == value)
    assert calls == [1., 2., 3., 2.]
    before = cache.statistics()
    assert {k: before[k] for k in ("requests", "hits", "misses", "evictions", "entries")} == {
        "requests": 6, "hits": 2, "misses": 4, "evictions": 2, "entries": 2}
    assert before["key_bytes"] == 2 * 16 * 105 * 105 * 4 and before["value_bytes"] == 128
    cache.clear()
    empty = cache.statistics()
    assert empty["entries"] == empty["key_bytes"] == empty["value_bytes"] == 0
    assert empty["clears"] == 1 and empty["requests"] == before["requests"]
    cache.get_or_compute(inputs(2), compute)
    assert calls == [1., 2., 3., 2., 2.]
    empty["misses"] = -1
    assert cache.statistics()["misses"] == 5


def test_cache_owns_immutable_input_and_value_snapshots_without_aliases():
    cache = ExactValueCache()
    original, output = inputs(1), values(7)

    def compute(x):
        assert x.tobytes() == original.tobytes() and not x.flags.writeable
        assert not np.shares_memory(x, original)
        with pytest.raises(ValueError):
            x.flags.writeable = True
        return output

    first = cache.get_or_compute(original, compute)
    first.fill(99)
    output.fill(-8)
    original.fill(3)
    replay = cache.get_or_compute(inputs(1), compute)
    assert np.all(replay == 7) and replay.flags.writeable
    replay.fill(42)
    assert np.all(cache.get_or_compute(inputs(1), compute) == 7)
    assert cache.statistics()["misses"] == 1


def test_callback_or_invalid_output_failure_never_inserts_or_evicts():
    cache = ExactValueCache(1)
    cache.get_or_compute(inputs(1), lambda _: values(1))
    error = RuntimeError("fabricated interrupted computation")

    def fail(_):
        raise error

    with pytest.raises(RuntimeError) as caught:
        cache.get_or_compute(inputs(2), fail)
    assert caught.value is error
    bad_values = (np.zeros((16, 1), dtype=np.float64), values(np.nan), values(np.inf),
                  np.zeros(16, dtype=np.float32), [[0.]] * 16)
    for bad in bad_values:
        with pytest.raises(ValueError, match="model values"):
            cache.get_or_compute(inputs(2), lambda _, bad=bad: bad)
    assert np.all(cache.get_or_compute(inputs(1), fail) == 1)
    stats = cache.statistics()
    assert stats["entries"] == 1 and stats["evictions"] == 0
    assert stats["compute_failures"] == 1 and stats["invalid_outputs"] == 5 and stats["failures"] == 6
    assert stats["requests"] == 8 and stats["hits"] == 1 and stats["misses"] == 7
    assert np.all(cache.get_or_compute(inputs(2), lambda _: values(2)) == 2)
    assert cache.statistics()["evictions"] == 1


def test_strict_inputs_capacity_and_callback_contract():
    for capacity in (0, -1, True, 1.5, np.int64(2)):
        with pytest.raises(ValueError, match="capacity"):
            ExactValueCache(capacity)
    cache = ExactValueCache()

    def forbidden(_):
        raise AssertionError("invalid input must never compute")

    for bad in (np.zeros((16, 105, 105), dtype=np.float64), np.zeros((15, 105, 105), dtype=np.float32),
                np.full(INPUT_SHAPE, np.nan, dtype=np.float32),
                np.full(INPUT_SHAPE, np.inf, dtype=np.float32), [0.]):
        with pytest.raises(ValueError, match="model inputs"):
            cache.get_or_compute(bad, forbidden)
    with pytest.raises(ValueError, match="callable"):
        cache.get_or_compute(inputs(), None)
    stats = cache.statistics()
    assert stats["requests"] == stats["invalid_requests"] == stats["failures"] == 6
    assert stats["misses"] == stats["hits"] == stats["entries"] == 0


def test_reentrant_callback_cannot_clear_or_reuse_active_cache():
    cache = ExactValueCache()
    with pytest.raises(ValueError, match="clear"):
        cache.get_or_compute(inputs(), lambda _: cache.clear())
    with pytest.raises(ValueError, match="reentrant"):
        cache.get_or_compute(inputs(), lambda x: cache.get_or_compute(x, lambda _: values()))
    assert cache.statistics()["compute_failures"] == 2 and cache.statistics()["entries"] == 0
