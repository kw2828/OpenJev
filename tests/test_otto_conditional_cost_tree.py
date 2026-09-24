"""Independent fabricated short-tree and finite-grid sampling oracles."""
from __future__ import annotations

import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research import otto_conditional_cost_tree as tree


def fixture():
    root = np.array([.25, .25, .25, .125, .125], np.float64)
    legacy = np.array([0., 0., .125, .0625, .0625], np.float64)
    positions = np.array([0, 1] * 4, np.int64)
    laws = np.empty((8, 5, 4), np.float64)
    for h, pos in enumerate(positions):
        for s in range(5):
            laws[h, s] = [.5, .25, .125, .125] if s % 2 == 0 else [.125, .125, .25, .5]
        laws[h, pos] = 0.
    calls = {"update": 0, "score": 0}

    def update(state, pos, odor):
        calls["update"] += 1
        after = state.copy()
        after[pos] = 0.
        after *= laws[pos, :, odor]
        mass = float(after.sum())
        # Deliberately retain a legacy subnormalization floor, independently
        # of exact source-history probability weights.
        return after / mass if mass > .1 else after

    def score(state, pos):
        calls["score"] += 1
        return np.array([state.sum(), state[2], 2 * state[3], 3 * state[4]], np.float32)

    return root, legacy, laws, positions, update, score, calls


def independent_state(legacy, laws, positions, history):
    state = legacy.copy()
    for h, odor in enumerate(history):
        state[int(positions[h])] = 0.
        state = state * laws[h, :, odor]
        total = math.fsum(float(v) for v in state)
        if total > .1:
            state = state / total
    return state


def independent_scores(state):
    return np.array([state.sum(), state[2], 2 * state[3], 3 * state[4]], np.float32)


def test_exact_tree_matches_independent_enumerated_source_mixture_and_legacy_floor():
    root, legacy, laws, pos, update, score, calls = fixture()
    original = [v.copy() for v in (root, legacy, laws, pos)]
    result = tree.exact_h4(root, legacy, laws[:4], pos[:4], update, score)
    assert calls == {"update": 340, "score": 256}
    expected_histories = np.array(list(itertools.product(range(4), repeat=4)), np.int64)
    np.testing.assert_array_equal(result["histories"], expected_histories)
    for i, history in enumerate(expected_histories):
        weight = math.fsum(float(root[s]) * math.prod(float(laws[h, s, odor]) for h, odor in enumerate(history))
                           for s in range(5) if s not in pos[:4])
        assert result["weights"][i] == pytest.approx(weight, rel=1e-14, abs=1e-18)
        expected = independent_scores(independent_state(legacy, laws, pos, history))
        np.testing.assert_array_equal(result["costs"][i], expected)
    assert result["found_mass"] == .5
    assert result["survival_mass"] == .5
    assert result["work"] == {"internal_nodes": 85, "positive_children": 340,
                              "legacy_updates": 340, "teacher_calls": 256, "found_positive_branches": 5}
    for actual, before in zip((root, legacy, laws, pos), original, strict=True):
        np.testing.assert_array_equal(actual, before)
    assert not any(np.shares_memory(result[k], root) for k in ("weights", "costs", "histories"))


def test_exact_weights_do_not_use_legacy_mass_or_floor():
    root, legacy, laws, pos, update, score, _ = fixture()
    small = tree.exact_h4(root, legacy, laws[:4], pos[:4], update, score)
    empty = tree.exact_h4(root, np.zeros_like(legacy), laws[:4], pos[:4], update, score)
    np.testing.assert_array_equal(small["weights"], empty["weights"])
    assert np.any(small["costs"] != empty["costs"])
    assert not np.any(empty["costs"])


def test_zero_probability_branches_and_repeated_found_mass_are_not_repaired():
    root, legacy, laws, pos, update, score, calls = fixture()
    laws[:] = [1., 0., 0., 0.]
    for h, p in enumerate(pos):
        laws[h, p] = 0.
    result = tree.exact_h4(root, legacy, laws[:4], pos[:4], update, score)
    assert calls == {"update": 4, "score": 1}
    assert result["weights"][0] == .5 and result["found_mass"] == .5
    assert result["supported"].sum() == 1
    assert not result["costs"][1:].any()
    assert result["work"]["found_positive_branches"] == 2


def test_all_found_tree_never_calls_legacy_or_teacher():
    root, legacy, laws, pos, update, score, calls = fixture()
    root[:] = [1., 0., 0., 0., 0.]
    result = tree.exact_h4(root, legacy, laws[:4], pos[:4], update, score)
    assert result["found_mass"] == 1. and result["survival_mass"] == 0.
    assert calls == {"update": 0, "score": 0}
    assert not result["costs"].any() and not result["weights"].any()


def test_integer_cdf_boundary_and_zero_support_without_a_second_normalization():
    p = np.array([.25, 0., .125, .625], np.float64)
    cdf = tree.integer_cdf(p)
    np.testing.assert_array_equal(cdf, [tree.BINS // 4, tree.BINS // 4, 3 * tree.BINS // 8, tree.BINS])
    for draw, expected in ((0, 0), (tree.BINS // 4 - 1, 0), (tree.BINS // 4, 2),
                           (3 * tree.BINS // 8 - 1, 2), (3 * tree.BINS // 8, 3), (tree.BINS - 1, 3)):
        assert np.searchsorted(cdf, np.uint64(draw), side="right") == expected
    with pytest.raises(ValueError, match="53-bit bins"):
        tree.integer_cdf(np.array([.1, .9], np.float64))
    with pytest.raises(ValueError, match="integer mass"):
        tree.integer_cdf(np.array([.25, .25], np.float64))


def independent_category(probabilities, draw):
    cumulative = 0
    for i, p in enumerate(probabilities):
        cumulative += int(float(p) * 2**53)
        if int(draw) < cumulative:
            return i
    raise AssertionError("exhausted fabricated categorical law")


def test_monte_carlo_matches_independent_full_draw_replay_and_looks_up_h4_only():
    root, legacy, laws, pos, update, score, calls = fixture()
    exact = tree.exact_h4(root, legacy, laws[:4], pos[:4], update, score)
    calls.update(update=0, score=0)
    draws = np.empty((128, 9), np.uint64)
    for i in range(128):
        draws[i, 0] = (i % 8) * (tree.BINS // 8)
        for h in range(8):
            draws[i, h + 1] = ((i + h) % 8) * (tree.BINS // 8)
    before = draws.copy()
    actual = tree.sample_h8(root, legacy, laws, pos, draws, exact, update, score)
    expected_updates = expected_scores = 0
    for i in range(128):
        source = independent_category(root, draws[i, 0])
        assert actual["source_indices"][i] == source
        history = []
        for h, p in enumerate(pos):
            if p == source:
                break
            history.append(independent_category(laws[h, source], draws[i, h + 1]))
        expected_updates += len(history)
        np.testing.assert_array_equal(actual["outcomes"][i], history + [4] * (8 - len(history)))
        assert bool(actual["alive4"][i]) == (len(history) >= 4)
        assert bool(actual["alive8"][i]) == (len(history) == 8)
        if len(history) >= 4:
            code = sum(v * (4**(3 - h)) for h, v in enumerate(history[:4]))
            np.testing.assert_array_equal(actual["costs4"][i], exact["costs"][code])
        else:
            assert not actual["costs4"][i].any()
        if len(history) == 8:
            expected_scores += 1
            state = independent_state(legacy, laws, pos, history)
            np.testing.assert_array_equal(actual["costs8"][i], independent_scores(state))
        else:
            assert not actual["costs8"][i].any()
    assert calls == {"update": expected_updates, "score": expected_scores}
    assert actual["work"]["allocated_draw_integers"] == 128 * 9
    assert actual["work"]["unused_odor_draws"] == 128 * 8 - expected_updates
    assert actual["work"]["teacher_calls"] == int(actual["alive8"].sum())
    np.testing.assert_array_equal(draws, before)


@pytest.mark.parametrize("kind", ("root_dtype", "root_mass", "root_nan", "legacy_negative", "legacy_shape",
                                 "law_mass", "law_negative", "law_nan", "position_shape", "position_oob"))
def test_invalid_inputs_fail_before_callbacks(kind):
    root, legacy, laws, pos, update, score, calls = fixture()
    if kind == "root_dtype":
        root = root.astype(np.float32)
    elif kind == "root_mass":
        root[0] = 0
    elif kind == "root_nan":
        root[0] = np.nan
    elif kind == "legacy_negative":
        legacy[0] = -1
    elif kind == "legacy_shape":
        legacy = legacy[:, None]
    elif kind == "law_mass":
        laws[0, 2] = 0
    elif kind == "law_negative":
        laws[0, 2, 0] = -1
    elif kind == "law_nan":
        laws[0, 2, 0] = np.nan
    elif kind == "position_shape":
        pos = pos.astype(np.float64)
    else:
        pos[0] = len(root)
    with pytest.raises(ValueError):
        tree.exact_h4(root, legacy, laws[:4], pos[:4], update, score)
    assert calls == {"update": 0, "score": 0}


@pytest.mark.parametrize("kind", ("dtype", "shape", "upper", "missing", "too_many"))
def test_bad_draws_fail_before_calls(kind):
    root, legacy, laws, pos, update, score, calls = fixture()
    exact = tree.exact_h4(root, legacy, laws[:4], pos[:4], update, score)
    calls.update(update=0, score=0)
    draws = np.zeros((2, 9), np.uint64)
    if kind == "dtype":
        draws = draws.astype(np.int64)
    elif kind == "shape":
        draws = draws[:, :8]
    elif kind == "upper":
        draws[0, 0] = tree.BINS
    elif kind == "missing":
        draws = draws[:0]
    else:
        draws = np.zeros((129, 9), np.uint64)
    with pytest.raises(ValueError, match="complete uint64"):
        tree.sample_h8(root, legacy, laws, pos, draws, exact, update, score)
    assert calls == {"update": 0, "score": 0}


def test_callbacks_cannot_mutate_input_root_or_a_sibling_and_failures_propagate():
    root, legacy, laws, pos, _, _, _ = fixture()
    originals = [a.copy() for a in (root, legacy, laws, pos)]

    def update(state, _position, odor):
        state[:] = odor
        return state

    def score(state, _position):
        result = np.full(4, state[0], np.float32)
        state[:] = 999
        return result

    exact = tree.exact_h4(root, legacy, laws[:4], pos[:4], update, score)
    np.testing.assert_array_equal(exact["costs"][:, 0], exact["histories"][:, -1])
    for actual, expected in zip((root, legacy, laws, pos), originals, strict=True):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="updated legacy"):
        tree.exact_h4(root, legacy, laws[:4], pos[:4], lambda *_: np.zeros(5, np.float32), score)
    with pytest.raises(ValueError, match="teacher scores"):
        tree.exact_h4(root, legacy, laws[:4], pos[:4], update, lambda *_: np.full(4, np.nan, np.float32))


def test_bounded_check_precedes_first_tree_callback():
    root, legacy, laws, pos, update, score, calls = fixture()

    def stop():
        raise TimeoutError("fabricated cap")

    with pytest.raises(TimeoutError, match="fabricated cap"):
        tree.exact_h4(root, legacy, laws[:4], pos[:4], update, score, check=stop)
    assert calls == {"update": 0, "score": 0}


def collector_module():
    root = Path(__file__).resolve().parents[1]
    for path, name in (("scripts/otto_cost_information_common.py", "otto_cost_information_common"),
                       ("scripts/collect_otto_cost_information.py", "conditional_cost_collector_tests")):
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, root / path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    return sys.modules["conditional_cost_collector_tests"]


def test_collector_registration_caps_and_independent_complete_draw_allocations():
    module = collector_module()
    roster = module.c.roster()
    assert len(roster) == 64 and len({r["id"] for r in roster}) == 64
    assert len({r[key] for r in roster for key in ("seed", "select_seed", "eval_seed")}) == 192
    assert module.CALL_CAPS["teacher_score"] == module.CALL_CAPS["tensorflow_value"] == 64 * (256 + 2 * 128)
    assert module.CALL_CAPS["native_step"] == 64 * 8
    assert module.CALL_CAPS["legacy_tree_update"] == 64 * (4 + 16 + 64 + 256)
    assert module.CALL_CAPS["legacy_mc_update"] == 64 * 2 * 128 * 8
    first = module.draw_integers(np, roster[0]["select_seed"])
    expected = np.random.PCG64(roster[0]["select_seed"]).random_raw((128, 9)) >> np.uint64(11)
    np.testing.assert_array_equal(first, expected)
    assert first.dtype == np.uint64 and first.shape == (128, 9) and (first < 2**53).all()
    assert not np.array_equal(first, module.draw_integers(np, roster[0]["eval_seed"]))
    expected_actions = np.random.Generator(np.random.PCG64(np.random.SeedSequence([roster[0]["seed"], 911]))).integers(
        0, 4, size=8, dtype=np.int64)
    np.testing.assert_array_equal(module.committed_actions(np, roster[0]["seed"]), expected_actions)
    with pytest.raises(ValueError):
        module.draw_integers(np, True)


def fabricated_collector(tmp_path, *, found_at=None):
    from openjev.research import otto_predictive_belief as bayes
    from openjev.research import otto_sampler_law as sampler
    from openjev.research.otto_query_gate import ReadOnlyBeliefView, _analytic, _features
    from openjev.research.otto_released_policy import PublicBeliefView, ReleasedPolicyActor, _move

    module = collector_module()
    run = module.Collector.__new__(module.Collector)
    run.out, run.check = tmp_path, lambda: None
    run.ledger = module.Ledger(run)
    kernel = np.full((4, 107, 107), .25, np.float64)
    kernel[:, 53, 53] = 0.
    table = np.full((105, 105, 4), .25, np.float64)
    table[52, 52] = 0.
    run.sensor_laws = {r: table.copy() for r in ("lambda3", "lambda4")}
    run.sensor_raw = {r: table.copy() for r in ("lambda3", "lambda4")}
    run.view_class, run.public_view_class = ReadOnlyBeliefView, PublicBeliefView
    run.analytic, run.features, run.bayes, run.sampler, run.tree = _analytic, _features, bayes, sampler, tree
    constructed = []

    class Analytic:
        def __init__(self, *, env, model, sym_avg):
            assert model is None and not sym_avg
            self.env = env

        def _value_policy(self):
            return 0, np.arange(4, dtype=np.float64)

    class Teacher:
        def __init__(self, *, env, model, sym_avg):
            self.env, self.model = env, model
            assert sym_avg

        def _value_policy(self):
            self.model(None, sym_avg=True)
            p = self.env.p_source
            return 0, np.array([p.sum(), p[0].sum(), p[:, 0].sum(), p[-1].sum()], np.float32)

    class Recorder:
        def __init__(self, model, code, ledger, numpy):
            assert model is None and code is Teacher._value_policy.__code__ and numpy is np
            self.ledger = ledger

        def __call__(self, inputs, *, sym_avg):
            assert inputs is None and sym_avg
            self.ledger.call("tensorflow_value", lambda: None)

    class Environment:
        def __init__(self, hit):
            self.agent, self.obs, self.steps = [26, 26], {"hit": hit, "done": False}, 0
            self.source, self.p_Poisson = np.array([40, 40]), kernel
            self.view = PublicBeliefView(kernel)
            self.view._reset(hit)
            prior = self.view.p_source.reshape(-1)
            self.draw_log = [{"channel": "source", "probabilities": prior.tolist(),
                              "cdf_mass": float(np.cumsum(prior)[-1]), "selected_index": 2160}]

        def step(self, action, *, quiet):
            assert quiet and self.steps < 8, "no native continuation allowed"
            self.agent = _move(action, self.agent)[0]
            self.steps += 1
            done = self.steps == found_at
            self.obs = {"hit": -2 if done else 1, "done": done}
            packet = observation(self, self.steps).__dict__
            self.view._observe(packet)
            if not done:
                self.draw_log.append({"channel": "hit", "probabilities": [.25] * 4,
                                      "cdf_mass": 1., "selected_index": 1})
            return self.obs["hit"], 0., done

    def environment(_source, _seed, _config, *, initial_hit):
        env = Environment(initial_hit)
        constructed.append(env)
        return env

    def observation(env, step):
        return SimpleNamespace(position=tuple(env.agent), hit=env.obs["hit"], done=env.obs["done"], step=step,
                               valid_actions=() if env.obs["done"] else tuple(a for a in range(4) if _move(a, env.agent)[1]))

    def witness(actor, env, packet, _np):
        assert actor.public == packet
        np.testing.assert_array_equal(actor.belief, env.view.p_source)
        return {"exact": True}

    run.reference = SimpleNamespace(packet=lambda obs: obs.__dict__.copy(), belief_witness=witness, ForwardRecorder=Recorder)
    run.runtime = SimpleNamespace(np=np, kernels={"base": kernel, "shift": kernel}, source=None, model=None,
        analytic=lambda packet, kernel, allow_stay: ReleasedPolicyActor(packet, kernel, None, Analytic, sym_avg=False),
        policy=Teacher, public=SimpleNamespace(seeded_environment=environment, observation=observation))
    return module, run, constructed


def test_fabricated_collector_no_native_continuation_exact_and_mc_share_root(tmp_path):
    module, run, environments = fabricated_collector(tmp_path)
    identity = module.c.roster()[0]
    row, arrays = run.case(identity)
    run.ledger.close()
    assert row["excluded"] is None and row["native_steps"] == environments[0].steps == 8
    assert arrays["prefix"].shape == (9, 31) and arrays["prefix_lengths"] == 9
    assert arrays["mc_draws"].shape == (2, 128, 9)
    assert arrays["exact_weights"].shape == (256,) and arrays["exact_costs"].shape == (256, 4)
    assert row["root_grid"]["total_variation"] <= 1e-10
    tree.integer_cdf(arrays["root_grid"])
    assert abs(arrays["exact_weights"].sum() + row["exact"]["found_mass"] - 1.) < 1e-12
    assert run.ledger.calls["native_step"]["returned"] == 8
    expected = int(arrays["exact_alive"].sum() + arrays["mc_alive8"].sum())
    assert run.ledger.calls["teacher_score"]["returned"] == expected
    assert run.ledger.calls["tensorflow_value"]["returned"] == expected
    assert run.ledger.calls["mc_draw_allocation"]["returned"] == 2
    for s, key in enumerate(("select_seed", "eval_seed")):
        np.testing.assert_array_equal(arrays["mc_draws"][s], module.draw_integers(np, identity[key]))
        for i in np.flatnonzero(arrays["mc_alive4"][s]):
            code = sum(int(x) * 4**(3 - h) for h, x in enumerate(arrays["mc_outcomes"][s, i, :4]))
            np.testing.assert_array_equal(arrays["mc_costs4"][s, i], arrays["exact_costs"][code])
    assert set(json.loads((tmp_path / "commitments.jsonl").read_text())) == {
        "identity", "prefix_public", "actions", "prefix_actions", "native_steps_before_counterfactuals"}


def test_prefix_termination_excludes_without_draws_or_teacher(tmp_path):
    module, run, envs = fabricated_collector(tmp_path, found_at=3)
    row, arrays = run.case(module.c.roster()[0])
    run.ledger.close()
    assert row["excluded"] == "found_during_observed_prefix" and arrays is None and envs[0].steps == 3
    for channel in ("teacher_score", "tensorflow_value", "mc_draw_allocation", "exact_tree", "mc_rollout"):
        assert run.ledger.calls[channel]["attempted"] == 0


def test_unregistered_case_rejected_before_native_or_branch_work(tmp_path):
    module, run, envs = fabricated_collector(tmp_path)
    identity = {**module.c.roster()[0], "seed": 1}
    with pytest.raises(ValueError, match="registered fresh"):
        run.case(identity)
    assert not envs and not any(r["attempted"] for r in run.ledger.calls.values())
