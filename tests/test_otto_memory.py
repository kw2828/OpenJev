"""Synthetic public-memory tests; no OTTO imports, environments or native steps."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from openjev.research.otto_memory import OdorMemory, normalized


@dataclass(frozen=True)
class Packet:
    position: tuple[int, ...]
    hit: int
    done: bool
    step: int
    valid_actions: tuple[int, ...] = (0, 1)

    @property
    def source(self):
        raise AssertionError("hidden source must not be inspected")

    @property
    def seed(self):
        raise AssertionError("simulator seed must not be inspected")

    @property
    def p_source(self):
        raise AssertionError("live simulator posterior must not be inspected")


class FakeModel:
    """A one-dimensional tabulated likelihood model, with no simulator state."""

    draw_source = False
    Ndim, N, Nhits = 1, 6, 3
    initial_hit = 2

    def __init__(self):
        self.agent = [0]
        # Already conditioned on the publicly disclosed initial hit. It is
        # intentionally nonuniform to distinguish it from a fresh flat prior.
        self.p_source = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64) / 15
        self.entropy = self._entropy(self.p_source)
        self.obs = {"hit": self.initial_hit, "done": False}
        self.p_Poisson = object()
        self.evidence = {}
        for position in range(self.N):
            zero = np.roll(np.array([.05, .15, .25, .35, .45, .55]), position)
            table = np.array([zero, .7 - zero, np.full(self.N, .3)])
            table[:, position] = 0
            self.evidence[(position,)] = table
        self.updates, self.extractions = [], []

    @staticmethod
    def _entropy(p):
        return -math.fsum(float(x) * math.log(float(x)) for x in p if x > 0)

    def _extract_N_from_2N(self, input, origin):
        assert input is self.p_Poisson
        self.extractions.append(tuple(origin))
        return self.evidence[tuple(origin)].copy()

    def _update_after_hit(self, *, hit, done):
        self.updates.append((tuple(self.agent), hit, done))
        if done:
            self.p_source = np.eye(self.N)[self.agent[0]].copy()
        else:
            p = self.p_source.copy()
            p[self.agent[0]] = 0
            p *= self.evidence[tuple(self.agent)][hit]
            self.p_source = p / math.fsum(p)
        self.entropy = self._entropy(self.p_source)
        self.obs = {"hit": hit, "done": done}


class FakePolicy:
    def __init__(self, model, *, policy, steps_ahead):
        assert model.draw_source is False and not hasattr(model, "source")
        assert steps_ahead == 1
        self.model, self.index = model, policy
        self.calls = []
        # The upstream tolerance-based choice need not be exact argmax. The
        # adapter must preserve both that action and inaccessible-action scores.
        self.scores = np.array([1.0, 1.0 + 1e-12, -np.inf])

    def _infotaxis(self):
        self.calls.append("infotaxis")
        return np.int64(0), self.scores

    def _space_aware_infotaxis(self):
        self.calls.append("space_aware")
        return np.int64(0), self.scores


def make(window=None, policy=0):
    model = FakeModel()
    return OdorMemory(model, FakePolicy, Packet((0,), 2, False, 0), window, policy)


def posterior_oracle(prior, model, events, window):
    """Enumerate possible static source cells using scalar products and sums."""
    visited = {(0,), *(event.position for event in events)}
    retained = events if window is None else events[-window:] if window else []
    weights = []
    for cell, initial_mass in enumerate(prior):
        factors = [model.evidence[event.position][event.hit, cell] for event in retained]
        weights.append(0.0 if (cell,) in visited else float(initial_mass) * math.prod(factors))
    total = math.fsum(weights)
    return np.array([value / total for value in weights])


@pytest.mark.parametrize("window", [None, 0, 1, 2, 128])
def test_initial_hit_prior_is_preserved_and_owned(window):
    memory = make(window)
    expected = np.array([0, 1, 2, 3, 4, 5], dtype=float) / 15
    np.testing.assert_array_equal(memory.prior, expected)
    assert not np.shares_memory(memory.prior, memory.model.p_source)
    assert memory.visited == {(0,)} and memory.step == 0 and not memory.done
    assert memory.model.updates == memory.model.extractions == []
    memory.model.p_source[:] = 0
    np.testing.assert_array_equal(memory.prior, expected)


@pytest.mark.parametrize("window", [None, 1, 128])
def test_zero_hits_are_evidence_with_closed_form_posterior(window):
    memory = make(window)
    memory.observe(Packet((1,), 0, False, 1))
    # Weights are [0,0,.30,.75,1.40,2.25], totaling 4.70.
    expected = np.array([0, 0, 6, 15, 28, 45], dtype=float) / 94
    np.testing.assert_allclose(memory.model.p_source, expected, rtol=0, atol=1e-15)
    assert memory.model.obs == {"hit": 0, "done": False}
    assert memory.model.entropy == pytest.approx(FakeModel._entropy(expected), abs=1e-14)


def test_full_matches_unexpired_window_and_expiry_discards_only_old_odor():
    full, long, short = make(None), make(128), make(2)
    events = [Packet((1,), 0, False, 1), Packet((2,), 2, False, 2),
              Packet((3,), 0, False, 3), Packet((2,), 1, False, 4)]
    for index, event in enumerate(events, 1):
        for memory in (full, long, short):
            memory.observe(event)
            expected = posterior_oracle(memory.prior, memory.model, events[:index], memory.window)
            np.testing.assert_allclose(memory.model.p_source, expected, rtol=0, atol=1e-14)
            assert all(memory.model.p_source[pos] == 0 for pos in memory.visited)
        np.testing.assert_allclose(full.model.p_source, long.model.p_source, rtol=0, atol=1e-14)
    assert list(short.recent) == [((3,), 0), ((2,), 1)]
    assert short.model.extractions == [(1,), (1,), (2,), (2,), (3,), (3,), (2,)]
    assert short.visited == {(0,), (1,), (2,), (3,)}
    assert not np.allclose(full.model.p_source, short.model.p_source)
    assert full.model.updates == [(e.position, e.hit, False) for e in events]
    assert short.model.updates == []


def test_zero_window_retains_initial_hit_and_every_visited_cell():
    memory = make(0)
    for step, position in enumerate(((1,), (2,), (1,)), 1):
        memory.observe(Packet(position, step % 3, False, step))
    np.testing.assert_allclose(memory.model.p_source, np.array([0, 0, 0, 3, 4, 5]) / 12)
    assert list(memory.recent) == [] and memory.model.extractions == []
    assert memory.visited == {(0,), (1,), (2,)}


@pytest.mark.parametrize("window", [None, 0, 1, 128])
def test_found_terminal_delegates_once_and_prohibits_further_actions(window):
    memory = make(window)
    memory.observe(Packet((1,), 0, False, 1))
    memory.observe(Packet((4,), -2, True, 2, ()))
    np.testing.assert_array_equal(memory.model.p_source, np.eye(6)[4])
    assert memory.model.updates[-1] == ((4,), -2, True)
    assert memory.model.obs == {"hit": -2, "done": True}
    assert memory.model.entropy == 0 and memory.done
    before = list(memory.model.updates)
    with pytest.raises(ValueError, match="no action"):
        memory.choose()
    with pytest.raises(ValueError, match="consecutive"):
        memory.observe(Packet((5,), 0, False, 3))
    assert memory.model.updates == before


@pytest.mark.parametrize("policy,method", [(0, "infotaxis"), (1, "space_aware")])
def test_routes_policy_and_preserves_upstream_near_tie_choice_and_score_copy(policy, method):
    memory = make(policy=policy)
    action, scores = memory.choose()
    assert action == 0 and type(action) is int
    assert np.argmax(scores) == 1  # Adapter must not impose its own tie rule.
    assert memory.policy.calls == [method]
    np.testing.assert_array_equal(scores, memory.policy.scores)
    scores[:] = 88
    assert memory.policy.scores[0] == 1 and np.isneginf(memory.policy.scores[2])


@pytest.mark.parametrize("draw_source,has_source", [(True, False), (False, True), (True, True)])
def test_live_or_source_carrying_model_rejected_before_policy_factory(draw_source, has_source):
    model = FakeModel()
    model.draw_source = draw_source
    if has_source:
        model.source = (4,)

    def forbidden_policy(*args, **kwargs):
        raise AssertionError("policy construction preceded public-model validation")

    with pytest.raises(ValueError, match="public-only"):
        OdorMemory(model, forbidden_policy, Packet((0,), 2, False, 0), None, 0)


@pytest.mark.parametrize("packet", [Packet((0,), 2, False, 1), Packet((0,), 2, True, 0),
                                    Packet((0,), 1, False, 0), Packet((1,), 2, False, 0)])
def test_initial_packet_must_match_the_separate_public_initialization(packet):
    with pytest.raises(ValueError, match="initial public"):
        OdorMemory(FakeModel(), FakePolicy, packet, None, 0)


@pytest.mark.parametrize("window", [-1, True, 1.5, "2"])
def test_invalid_window_rejected(window):
    with pytest.raises(ValueError, match="window"):
        make(window)


@pytest.mark.parametrize("policy", [-1, 2, "0"])
def test_unqualified_policy_rejected(policy):
    with pytest.raises(ValueError, match="policies"):
        make(policy=policy)


@pytest.mark.parametrize("packet", [Packet((1,), 0, False, 2), Packet((6,), 0, False, 1),
                                    Packet((-1,), 0, False, 1), Packet((True,), 0, False, 1),
                                    Packet((1, 2), 0, False, 1), Packet((1,), 3, False, 1),
                                    Packet((1,), -2, False, 1), Packet((1,), 0, True, 1)])
def test_invalid_public_transition_rejected_without_mutation(packet):
    memory = make(2)
    before = memory.model.p_source.copy()
    with pytest.raises(ValueError):
        memory.observe(packet)
    np.testing.assert_array_equal(memory.model.p_source, before)
    assert memory.step == 0 and memory.model.agent == [0]
    assert memory.visited == {(0,)} and not memory.recent and not memory.model.updates


@pytest.mark.parametrize("p", [[0, 0], [-1, 2], [np.nan, 1], [np.inf, 1], [1e308, 1e308]])
def test_normalization_rejects_invalid_or_overflowed_mass(p):
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="posterior"):
        normalized(p)


def test_normalization_preserves_zero_support_tiny_mass_and_input():
    p = np.array([0.0, 1e-300, 2e-300])
    before = p.copy()
    actual = normalized(p)
    np.testing.assert_array_equal(p, before)
    np.testing.assert_allclose(actual, [0, 1 / 3, 2 / 3], rtol=0, atol=1e-16)
    assert actual[0] == 0 and not np.shares_memory(actual, p)


@pytest.mark.parametrize("p", [[0, 0, 0, 0, 0, 0], [0, .1, .2, .3, .4, .1],
                               [.1, .1, .2, .2, .2, .2], [0, -1, 1, 0, 0, 1],
                               [0, np.nan, 0, 0, 0, 1]])
def test_posterior_and_permanent_nonfound_support_are_checked(p):
    memory = make()
    memory.model.p_source = np.array(p)
    with pytest.raises(ValueError):
        memory.validate()
