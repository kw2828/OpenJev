"""Fake-helper qualification only. Never import or execute OTTO."""
from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from openjev.research.otto_public import CHANNELS, PublicObservation, observation, seeded_environment


class FakeSourceTracking:
    """Small deterministic helper surface, with no upstream simulator imports."""

    def __init__(self, initial_hit=None, **config):
        self.config = dict(config)
        self.Ndim, self.N = config["Ndim"], config["Ngrid"] or 3
        self.Nhits, self.Nactions = config["Nhits"] or 3, 2 * self.Ndim
        self.lambda_over_dx = config["lambda_over_dx"]
        self.draw_source, self.norm_Poisson = config["draw_source"], config["norm_Poisson"]
        self.executed = []
        self.restart(initial_hit=initial_hit)

    def restart(self, initial_hit=None):
        self.initial_hit = self._initial_hit() if initial_hit is None else initial_hit
        self.agent = [self.N // 2] * self.Ndim
        # An intentionally simple hit-dependent conditional source distribution.
        self.p_source = np.full([self.N] * self.Ndim, 1.0)
        self.p_source[tuple(self.agent)] = 0
        self.p_source.flat[-1] += self.initial_hit
        self.p_source /= self.p_source.sum()
        self._draw_a_source()
        self.obs = {"hit": self.initial_hit, "done": False}

    def _volume_ball(self, radius):
        return 2 * np.asarray(radius)

    def _mean_number_of_hits(self, distance):
        return np.ones_like(np.asarray(distance), dtype=float)

    def _Poisson(self, mu, hit):
        mu = np.asarray(mu)
        if hit == self.Nhits - 1:
            return 1 - sum(np.exp(-mu) * mu**h / math.factorial(h) for h in range(hit))
        return np.exp(-mu) * mu**hit / math.factorial(hit)

    def _move(self, action, agent):
        result = copy.deepcopy(agent)
        axis, direction = action // 2, 2 * (action % 2) - 1
        if 0 <= result[axis] + direction < self.N:
            result[axis] += direction
            return result, True
        return result, False

    def step(self, action, hit=None, quiet=False):
        hit, p_end, done = self._execute_action(action, hit, quiet)
        self.obs = {"hit": hit, "done": done}
        return hit, p_end, done

    def _execute_action(self, action, hit=None, quiet=False):
        # The adapter must prevent every fallback to an unseeded upstream draw.
        assert hit is not None
        self.executed.append((action, hit, quiet))
        self.agent, _ = self._move(action, self.agent)
        found = tuple(self.agent) == tuple(self.source)
        return (-2, 1, True) if found else (hit, 0, False)


def environment(seed=31, **config):
    return seeded_environment(FakeSourceTracking, seed, config)


def assert_global_state_equal(left, right):
    assert left[0] == right[0] and left[2:] == right[2:]
    np.testing.assert_array_equal(left[1], right[1])


def test_factory_defaults_no_global_rng_mutation_and_same_seed_replay():
    original = np.random.get_state()
    left, right = environment(), environment()
    assert left.config == {"Ndim": 2, "lambda_over_dx": 1., "R_dt": 1., "Ngrid": None,
                           "Nhits": None, "draw_source": True, "norm_Poisson": "Euclidean"}
    assert left.draw_log == right.draw_log
    np.testing.assert_array_equal(left.source, right.source)
    assert left.initial_hit > 0
    for env in (left, right):
        env.source = np.array([2, 2])  # synthetic evaluator-only source fixture
    for action in (0, 2, 1, 3):
        assert left.step(action) == right.step(action)
        assert observation(left, len(left.executed)) == observation(right, len(right.executed))
    assert left.draw_log == right.draw_log
    assert_global_state_equal(original, np.random.get_state())
    assert isinstance(left, FakeSourceTracking)


def test_initial_hit_distribution_and_source_conditional_are_preserved():
    env = environment()
    initial, source = env.draw_log
    p_one = math.exp(-1)
    p_tail = 1 - 2 * math.exp(-1)
    np.testing.assert_allclose(initial["probabilities"], [0, p_one / (p_one + p_tail),
                                                        p_tail / (p_one + p_tail)], atol=1e-15)
    assert initial["selected_index"] == env.initial_hit
    np.testing.assert_array_equal(source["probabilities"], env.p_source.ravel())
    assert source["probabilities"][4] == 0
    assert np.unravel_index(source["selected_index"], env.p_source.shape) == tuple(env.source)
    assert [row["channel"] for row in env.draw_log] == ["initial", "source"]


def test_draw_logs_reconstruct_channel_streams_without_exposing_mutable_records():
    env = environment(7)
    env.source = np.array([2, 2])
    env.step(0)
    env.step(2)
    env.restart()
    streams = {name: np.random.Generator(np.random.PCG64(np.random.SeedSequence([7, code])))
               for name, code in CHANNELS.items()}
    counts = dict.fromkeys(CHANNELS, 0)
    for record in env.draw_log:
        channel = record["channel"]
        assert record["index"] == counts[channel]
        counts[channel] += 1
        assert record["uniform"] == streams[channel].random()
        cumulative = np.cumsum(record["probabilities"])
        assert record["cdf_mass"] == cumulative[-1]
        expected = int(np.searchsorted(cumulative / cumulative[-1], record["uniform"], side="right"))
        assert record["selected_index"] == expected
        assert record["probabilities"][expected] > 0
    audit_copy = env.draw_log
    audit_copy[0]["probabilities"][0] = 999
    assert env.draw_log[0]["probabilities"][0] == 0
    json.dumps(env.draw_log, allow_nan=False)


def test_extra_hit_draw_does_not_change_next_initial_or_source_draws():
    left, right = environment(), environment()
    left.source = np.array([2, 2])
    left.step(0)
    for env in (left, right):
        env.restart()
    assert left.initial_hit == right.initial_hit
    np.testing.assert_array_equal(left.source, right.source)
    left_init = [r for r in left.draw_log if r["channel"] != "hit"]
    right_init = [r for r in right.draw_log if r["channel"] != "hit"]
    assert left_init == right_init


@pytest.mark.parametrize("initial_hit", [1, 2, np.int64(2)])
def test_explicit_initial_hit_is_conditioned_without_initial_draw(initial_hit):
    env = seeded_environment(FakeSourceTracking, 3, initial_hit=initial_hit)
    assert env.initial_hit == initial_hit
    assert [record["channel"] for record in env.draw_log] == ["source"]
    expected = np.ones((3, 3))
    expected[1, 1] = 0
    expected[2, 2] += int(initial_hit)
    expected /= expected.sum()
    np.testing.assert_array_equal(env.p_source, expected)
    np.testing.assert_array_equal(env.draw_log[0]["probabilities"], expected.ravel())


@pytest.mark.parametrize("initial_hit", [0, -1, True, np.bool_(False), 1.0, "1", 3, 4])
def test_invalid_explicit_initial_hit_rejected_before_any_draw(initial_hit):
    class NoDraw(FakeSourceTracking):
        def _initial_hit(self, hit=None):
            raise AssertionError("invalid initialization must not draw an initial hit")

        def _draw_a_source(self):
            raise AssertionError("invalid initialization must not draw a source")

        def restart(self, initial_hit=None):
            raise AssertionError("invalid initialization must fail before the original restart")

    with pytest.raises(ValueError, match="initial_hit"):
        seeded_environment(NoDraw, 0, initial_hit=initial_hit)


def test_explicit_initial_hit_range_uses_resolved_not_default_bin_count():
    with pytest.raises(ValueError, match="resolved Nhits"):
        seeded_environment(FakeSourceTracking, 0, {"Nhits": 2}, initial_hit=2)
    env = seeded_environment(FakeSourceTracking, 0, {"Nhits": 4}, initial_hit=3)
    assert observation(env, 0).hit == 3
    before = env.draw_log
    with pytest.raises(ValueError, match="resolved Nhits"):
        env.restart(initial_hit=4)
    assert env.draw_log == before


def test_whole_draw_logs_do_not_depend_on_environment_construction_or_step_order():
    """Interleaving another episode cannot advance any stream in this episode."""
    def make(seed):
        return seeded_environment(FakeSourceTracking, seed, initial_hit=1 + seed % 2)

    def advance(env):
        candidates = [a for a in observation(env, len(env.executed)).valid_actions
                      if tuple(env._move(a, env.agent)[0]) != tuple(env.source)]
        env.step(candidates[0])

    sequential = {}
    for seed in (5, 17):
        env = make(seed)
        for _ in range(4):
            advance(env)
        env.restart()
        for _ in range(3):
            advance(env)
        sequential[seed] = (env.draw_log, observation(env, len(env.executed)))
    interleaved = {seed: make(seed) for seed in (17, 5)}
    for step in range(7):
        if step == 4:
            for env in interleaved.values():
                env.restart()
        for env in interleaved.values():
            advance(env)
    for seed, env in interleaved.items():
        assert (env.draw_log, observation(env, len(env.executed))) == sequential[seed]


@pytest.mark.parametrize("norm, expected_distance", [("Euclidean", math.sqrt(5)),
                                                    ("Manhattan", 3), ("Chebyshev", 2)])
def test_post_move_distance_tail_and_original_execution_are_used(norm, expected_distance):
    env = environment(norm_Poisson=norm)
    env.source = np.array([2, 2])
    seen = []
    original_mean = env._mean_number_of_hits

    def mean(distance):
        seen.append(float(distance))
        return original_mean(distance)

    env._mean_number_of_hits = mean
    returned = env.step(0, quiet=True)  # [1,1] -> [0,1]
    assert seen == pytest.approx([expected_distance])
    record = env.draw_log[-1]
    np.testing.assert_allclose(record["probabilities"], [math.exp(-1), math.exp(-1),
                                                        1 - 2 * math.exp(-1)], atol=1e-15)
    assert returned == (record["selected_index"], 0, False)
    assert env.executed == [(0, record["selected_index"], True)]


def test_found_consumes_no_hit_draw_and_preserves_upstream_sentinel():
    env = environment()
    env.source = np.array([0, 1])
    before = env.draw_log
    assert env.step(0) == (-2, 1, True)
    assert env.draw_log == before
    assert env.executed == [(0, 0, False)]
    assert observation(env, 1) == PublicObservation((0, 1), -2, True, 1, ())


def test_forced_hit_is_delegated_without_random_draw_and_found_still_wins():
    env = environment()
    env.source = np.array([2, 2])
    before = env.draw_log
    assert env.step(0, hit=2) == (2, 0, False)
    assert env.draw_log == before
    env.source = np.array([0, 0])
    assert env.step(2, hit=1) == (-2, 1, True)
    assert env.draw_log == before


def test_blocked_move_stays_put_but_still_draws_a_nonterminal_hit():
    env = environment()
    env.agent, env.source = [0, 1], np.array([2, 2])
    before = len(env.draw_log)
    assert env.step(0)[2] is False
    assert env.agent == [0, 1] and len(env.draw_log) == before + 1
    assert observation(env, 1).valid_actions == (1, 2, 3)


def test_public_observation_reads_no_hidden_source_or_belief_and_is_immutable():
    class PublicOnly:
        Ndim, N, Nhits, Nactions = 2, 3, 3, 4
        _move = FakeSourceTracking._move

        def __init__(self):
            self.agent = [1, 1]
            self.obs = {"hit": 1, "done": False}

        def __getattr__(self, name):
            raise AssertionError(f"forbidden hidden field: {name}")

    actor = observation(PublicOnly(), 0)
    assert actor == ((1, 1), 1, False, 0, (0, 1, 2, 3))
    with pytest.raises(AttributeError):
        actor.hit = 2
    assert set(actor._fields) == {"position", "hit", "done", "step", "valid_actions"}


@pytest.mark.parametrize("state, exception", [({"hit": 1, "done": 0}, TypeError),
                                             ({"hit": True, "done": False}, TypeError),
                                             ({"hit": 1.0, "done": False}, TypeError),
                                             ({"hit": 1, "done": True}, ValueError),
                                             ({"hit": -2, "done": False}, ValueError)])
def test_public_hit_and_boundary_contract_rejects_inconsistent_state(state, exception):
    env = environment()
    env.obs = state
    with pytest.raises(exception):
        observation(env, 1)


@pytest.mark.parametrize("seed", [-1, True, np.bool_(False), 1.0, "1", None])
def test_invalid_seed_rejected_before_construction(seed):
    with pytest.raises(ValueError, match="seed"):
        environment(seed)


@pytest.mark.parametrize("config", [{"draw_source": False}, {"draw_source": 1}, {"dummy": True},
                                    {"initial_hit": 1}, {"seed": 2}, {"Ndim": True},
                                    {"Ngrid": 2}, {"Nhits": 1}, {"R_dt": np.nan},
                                    {"R_dt": 0}, {"lambda_over_dx": 0.5}, {"norm_Poisson": "L2"}])
def test_invalid_configuration_or_runtime_override_rejected(config):
    with pytest.raises(ValueError):
        seeded_environment(FakeSourceTracking, 0, config)


@pytest.mark.parametrize("action", [-1, True, np.bool_(False), 4, 0.0, "0"])
def test_invalid_action_has_no_draw_or_transition(action):
    env = environment()
    before = env.draw_log
    with pytest.raises(ValueError):
        env.step(action)
    assert env.draw_log == before and env.executed == []


@pytest.mark.parametrize("step", [-1, True, np.bool_(False), 0.0, "0"])
def test_invalid_step_rejected(step):
    with pytest.raises(ValueError, match="step"):
        observation(environment(), step)


@pytest.mark.parametrize("probability", [[0, 0], [-0.1, 1.1], [np.nan, 1], [0.1, 0.2], [[1.]], []])
def test_invalid_categorical_vector_fails_before_rng_or_log_mutation(probability):
    env = environment()
    before = env.draw_log
    state = copy.deepcopy(env._public_rngs["hit"].bit_generator.state)
    with pytest.raises(ValueError, match="categorical"):
        env._draw("hit", probability)
    assert env.draw_log == before and env._public_rngs["hit"].bit_generator.state == state


def test_zero_probability_bins_are_not_floored_and_explicit_hit_passthrough():
    env = environment()
    assert env._draw("hit", [0, 1, 0]) == 1
    assert env.draw_log[-1]["probabilities"] == [0, 1, 0]
    before = env.draw_log
    assert env._initial_hit(2) == 2 and env.draw_log == before
