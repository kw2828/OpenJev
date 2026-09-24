"""Fabricated independent finite-grid endpoint and collector checks."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research import otto_conditional_label_sampling as sampling


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
        return after / mass if mass > .1 else after

    def score(state, _pos):
        calls["score"] += 1
        return np.array([state.sum(), state[2], 2 * state[3], 3 * state[4]], np.float32)

    return root, legacy, laws, positions, update, score, calls


def independent_category(probabilities, draw):
    cumulative = 0
    for i, p in enumerate(probabilities):
        cumulative += int(float(p) * 2**53)
        if int(draw) < cumulative:
            return i
    raise AssertionError("exhausted fabricated categorical law")


def independent_state(legacy, laws, positions, history):
    state = legacy.copy()
    for h, odor in enumerate(history):
        state[int(positions[h])] = 0.
        state = state * laws[h, :, odor]
        total = math.fsum(float(v) for v in state)
        if total > .1:
            state = state / total
    return state


def draw_fixture(count):
    draws = np.empty((count, 9), np.uint64)
    for i in range(count):
        draws[i, 0] = (i % 8) * (sampling.BINS // 8)
        for h in range(8):
            draws[i, h + 1] = ((i + h) % 8) * (sampling.BINS // 8)
    return draws


@pytest.mark.parametrize("count", (32, 128))
def test_independent_integer_replay_legacy_floor_and_only_endpoint_annotations(count):
    root, legacy, laws, pos, update, score, calls = fixture()
    draws = draw_fixture(count)
    originals = [v.copy() for v in (root, legacy, laws, pos, draws)]
    events = []
    actual = sampling.sample_endpoint(root, legacy, laws, pos, draws, update, score, emit=events.append)
    expected_updates = expected_scores = 0
    for i in range(count):
        source = independent_category(root, draws[i, 0])
        assert actual["source_indices"][i] == source
        history = []
        for h, p in enumerate(pos):
            if p == source:
                break
            history.append(independent_category(laws[h, source], draws[i, h + 1]))
        expected_updates += len(history)
        np.testing.assert_array_equal(actual["outcomes"][i], history + [4] * (8 - len(history)))
        assert bool(actual["alive"][i]) == (len(history) == 8)
        if len(history) == 8:
            expected_scores += 1
            state = independent_state(legacy, laws, pos, history)
            expected = np.array([state.sum(), state[2], 2 * state[3], 3 * state[4]], np.float32)
            np.testing.assert_array_equal(actual["costs"][i], expected)
        else:
            assert actual["costs"][i].tobytes() == np.zeros(4, np.float32).tobytes()
    assert calls == {"update": expected_updates, "score": expected_scores}
    assert actual["work"] == {"allocated_draw_integers": count * 9, "source_draws": count,
        "odor_draws": expected_updates, "unused_odor_draws": count * 8 - expected_updates,
        "legacy_updates": expected_updates, "teacher_calls": expected_scores}
    assert len(events) == expected_updates
    assert events[-1]["horizon"] == 8 and events[-1]["mode"] == "mc"
    for before, after in zip(originals, (root, legacy, laws, pos, draws), strict=True):
        np.testing.assert_array_equal(before, after)
    assert not np.shares_memory(actual["costs"], legacy)


def test_grid_boundary_zero_support_and_no_probability_repair():
    p = np.array([.25, 0., .125, .625], np.float64)
    actual = sampling.integer_cdf(p)
    np.testing.assert_array_equal(actual, [2**51, 2**51, 3 * 2**50, 2**53])
    for draw, expected in ((0, 0), (2**51 - 1, 0), (2**51, 2), (3 * 2**50, 3), (2**53 - 1, 3)):
        assert np.searchsorted(actual, np.uint64(draw), side="right") == expected
    with pytest.raises(ValueError, match="53-bit bins"):
        sampling.integer_cdf(np.array([.1, .9], np.float64))
    with pytest.raises(ValueError, match="integer mass"):
        sampling.integer_cdf(np.array([.25, .25], np.float64))


def test_found_absorbs_at_first_visit_without_teacher_or_legacy_update():
    root, legacy, laws, pos, update, score, calls = fixture()
    root[:] = [1., 0., 0., 0., 0.]
    result = sampling.sample_endpoint(root, legacy, laws, pos, draw_fixture(32), update, score)
    assert calls == {"update": 0, "score": 0}
    assert (result["outcomes"] == 4).all() and not result["alive"].any()
    assert result["costs"].tobytes() == np.zeros((32, 4), np.float32).tobytes()
    assert result["work"]["unused_odor_draws"] == 256


@pytest.mark.parametrize("kind", ("root_dtype", "root_mass", "root_nan", "legacy_negative", "legacy_shape",
                                 "law_mass", "law_negative", "law_nan", "position_dtype", "position_oob",
                                 "draw_dtype", "draw_shape", "draw_upper", "draw_count"))
def test_invalid_inputs_rejected_before_any_callbacks(kind):
    root, legacy, laws, pos, update, score, calls = fixture()
    draws = draw_fixture(32)
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
    elif kind == "position_dtype":
        pos = pos.astype(np.float64)
    elif kind == "position_oob":
        pos[0] = len(root)
    elif kind == "draw_dtype":
        draws = draws.astype(np.int64)
    elif kind == "draw_shape":
        draws = draws[:, :8]
    elif kind == "draw_upper":
        draws[0, 0] = sampling.BINS
    else:
        draws = draws[:31]
    with pytest.raises(ValueError):
        sampling.sample_endpoint(root, legacy, laws, pos, draws, update, score)
    assert calls == {"update": 0, "score": 0}


def test_mutating_callbacks_cannot_touch_caller_or_other_histories():
    root, legacy, laws, pos, _, _, _ = fixture()
    originals = [a.copy() for a in (root, legacy, laws, pos)]
    draws = draw_fixture(32)

    def update(state, _position, odor):
        state[:] = odor
        return state

    def score(state, _position):
        result = np.full(4, state[0], np.float32)
        state[:] = 999
        return result

    result = sampling.sample_endpoint(root, legacy, laws, pos, draws, update, score)
    np.testing.assert_array_equal(result["costs"][result["alive"], 0], result["outcomes"][result["alive"], -1])
    for actual, expected in zip((root, legacy, laws, pos), originals, strict=True):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="updated legacy"):
        sampling.sample_endpoint(root, legacy, laws, pos, draws, lambda *_: np.zeros(5, np.float32), score)
    with pytest.raises(ValueError, match="teacher endpoint"):
        sampling.sample_endpoint(root, legacy, laws, pos, draws, update, lambda *_: np.full(4, np.nan, np.float32))
    with pytest.raises(ValueError, match="nonnegative teacher"):
        sampling.sample_endpoint(root, legacy, laws, pos, draws, update, lambda *_: np.full(4, -1, np.float32))


def test_bound_check_precedes_first_callback():
    root, legacy, laws, pos, update, score, calls = fixture()

    def stop():
        raise TimeoutError("fabricated cap")

    with pytest.raises(TimeoutError, match="fabricated cap"):
        sampling.sample_endpoint(root, legacy, laws, pos, draw_fixture(32), update, score, check=stop)
    assert calls == {"update": 0, "score": 0}


def collector_module():
    root = Path(__file__).resolve().parents[1]
    for path, name in (("scripts/otto_conditional_label_common.py", "otto_conditional_label_common"),
                       ("scripts/collect_otto_conditional_label.py", "conditional_label_collector_tests")):
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, root / path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    return sys.modules["conditional_label_collector_tests"]


def test_fresh_roster_fixed_caps_and_complete_bank_allocations():
    module = collector_module()
    roster = module.c.roster()
    assert len(roster) == 768 and len({r["id"] for r in roster}) == 768
    assert [r["seed"] for r in roster[:512]] == list(range(336000001, 336000513))
    assert [r["seed"] for r in roster[512:640]] == list(range(337000001, 337000129))
    assert [r["seed"] for r in roster[640:]] == list(range(338000001, 338000129))
    assert [r["mc_seed"] for r in roster[:512]] == list(range(339000001, 339000513))
    assert [r["mc_seed"] for r in roster[512:]] == list(range(340000001, 340000257))
    assert len({r[k] for r in roster for k in ("seed", "mc_seed")}) == 1536
    assert module.CALL_CAPS["teacher_score"] == module.CALL_CAPS["tensorflow_value"] == 512 * 32 + 256 * 128
    assert module.CALL_CAPS["native_step"] == 768 * 8
    assert module.CALL_CAPS["legacy_mc_update"] == (512 * 32 + 256 * 128) * 8
    assert not {"exact_tree", "legacy_tree_update"} & set(module.CALL_CAPS)
    for identity, count in ((roster[0], 32), (roster[512], 128)):
        actual = module.draw_integers(np, identity["mc_seed"], count)
        expected = np.random.PCG64(identity["mc_seed"]).random_raw((count, 9)) >> np.uint64(11)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.uint64 and actual.shape == (count, 9) and (actual < 2**53).all()
        expected_actions = np.random.Generator(np.random.PCG64(np.random.SeedSequence([identity["seed"], 911]))).integers(
            0, 4, size=8, dtype=np.int64)
        np.testing.assert_array_equal(module.committed_actions(np, identity["seed"]), expected_actions)
    with pytest.raises(ValueError):
        module.draw_integers(np, True, 32)
    with pytest.raises(ValueError):
        module.draw_integers(np, 1, 64)


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
    run.analytic, run.features, run.bayes, run.sampler, run.sampling = _analytic, _features, bayes, sampler, sampling
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


@pytest.mark.parametrize("index,count", ((0, 32), (512, 128), (640, 128)))
def test_fabricated_collector_endpoint_only_and_no_native_continuation(tmp_path, index, count):
    module, run, environments = fabricated_collector(tmp_path)
    identity = module.c.roster()[index]
    row, arrays = run.case(identity)
    run.ledger.close()
    assert row["excluded"] is None and row["native_steps"] == environments[0].steps == 8
    assert arrays["prefix"].shape == (9, 31) and arrays["prefix_lengths"] == 9
    assert arrays["mc_draws"].shape == (count, 9) and arrays["costs"].shape == (count, 4)
    assert arrays["alive"].dtype == np.bool_ and arrays["mc_outcomes"].shape == (count, 8)
    assert set(arrays) == {"prefix", "prefix_lengths", "actions", "prefix_actions", "prefix_outcomes",
        "prefix_position", "initial_belief", "root_strict", "root_grid", "legacy_root", "mc_draws",
        "mc_source_indices", "mc_outcomes", "alive", "costs"}
    assert row["root_grid"]["total_variation"] <= 1e-10
    sampling.integer_cdf(arrays["root_grid"])
    assert run.ledger.calls["native_step"]["returned"] == 8
    expected = int(arrays["alive"].sum())
    assert run.ledger.calls["teacher_score"]["returned"] == expected
    assert run.ledger.calls["tensorflow_value"]["returned"] == expected
    assert run.ledger.calls["mc_draw_allocation"]["returned"] == 1
    assert row["mc"]["draws"] == count and row["mc"]["stream"] == identity["split"]
    assert row["mc"]["work"]["teacher_calls"] == expected
    np.testing.assert_array_equal(arrays["mc_draws"], module.draw_integers(np, identity["mc_seed"], count))
    assert arrays["costs"][~arrays["alive"]].tobytes() == np.zeros((count - expected, 4), np.float32).tobytes()
    assert set(json.loads((tmp_path / "commitments.jsonl").read_text())) == {
        "identity", "prefix_public", "actions", "prefix_actions", "native_steps_before_counterfactuals"}


def test_prefix_termination_excludes_without_draws_or_teacher(tmp_path):
    module, run, envs = fabricated_collector(tmp_path, found_at=3)
    row, arrays = run.case(module.c.roster()[0])
    run.ledger.close()
    assert row["excluded"] == "found_during_observed_prefix" and arrays is None and envs[0].steps == 3
    for channel in ("teacher_score", "tensorflow_value", "mc_draw_allocation", "mc_rollout"):
        assert run.ledger.calls[channel]["attempted"] == 0


def test_unregistered_case_rejected_before_native_or_branch_work(tmp_path):
    module, run, envs = fabricated_collector(tmp_path)
    identity = {**module.c.roster()[0], "seed": 1}
    with pytest.raises(ValueError, match="registered fresh"):
        run.case(identity)
    assert not envs and not any(r["attempted"] for r in run.ledger.calls.values())
