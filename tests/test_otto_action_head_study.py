"""Tiny artificial policy-distillation fixtures; no OTTO imports or episodes."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from openjev.research import otto_action_head as head

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("otto_action_study_test", ROOT / "scripts/study_otto_action_head.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
WEIGHTS = {r: {1: .5, 2: .25, 3: .25} for r in ("base", "shift")}


def test_complete_rotated_cohort_and_disjoint_seed_namespaces():
    order = list(M.evaluation_order())
    assert len(order) == 1536 == 2 * 48 * 16
    assert len(set(order)) == len(order)
    assert Counter((r, a, h) for r, _, _, h, a in order) == Counter(
        {(r, a, h): 16 for r in ("base", "shift") for a in M.ARMS for h in (1, 2, 3)})
    assert Counter((r, b, a) for r, _, b, _, a in order) == Counter(
        {(r, b, a): 6 for r in ("base", "shift") for b in range(8) for a in M.ARMS})
    for i in range(96):
        arms = tuple(x[-1] for x in order[i * 16:(i + 1) * 16])
        rotation = i % 16
        assert arms == M.ARMS[rotation:] + M.ARMS[:rotation]
    train = set(range(670001, 670001 + 384))
    valid = set(range(680001, 680001 + 96))
    evaluation = {r[1] for r in order}
    assert train.isdisjoint(valid | evaluation) and valid.isdisjoint(evaluation)
    assert M.LIMITS["native_steps"] == 480 * 256 + 1536 * 2188 + 2048


def test_teacher_targets_have_correct_sign_mask_and_fixed_temperature():
    target, legal = M.teacher_target([10., None, 12., 11.], np)
    expected = np.exp([0., -4., -2.])
    expected /= expected.sum()
    np.testing.assert_allclose(target[[0, 2, 3]], expected, rtol=1e-7)
    np.testing.assert_array_equal(legal, [True, False, True, True])
    assert target[1] == 0 and target.argmax() == 0
    shifted, _ = M.teacher_target([130., None, 136., 133.], np)
    np.testing.assert_array_equal(shifted, target)


def test_teacher_ties_single_action_and_tiny_gap_are_defined():
    target, _ = M.teacher_target([None, 4., 4., 4.], np)
    np.testing.assert_array_equal(target, np.array([0, 1 / 3, 1 / 3, 1 / 3], dtype=np.float32))
    target, _ = M.teacher_target([None, None, 3., None], np)
    np.testing.assert_array_equal(target, [0, 0, 1, 0])
    target, _ = M.teacher_target([0., 1e-10, None, None], np)
    expected = np.exp([0., -.04])
    expected /= expected.sum()
    np.testing.assert_allclose(target[:2], expected, rtol=1e-7)


@pytest.mark.parametrize("scores", [[None] * 4, [0, 1, 2], [0, 1, 2, 3, 4],
                                    [0, np.nan, None, 1], [0, None, np.inf, 1]])
def test_invalid_teacher_scores_rejected(scores):
    with pytest.raises(ValueError):
        M.teacher_target(scores, np)


def passing_rows():
    rows = []
    for regime, seed, block, hit, arm in M.evaluation_order():
        family, fit = arm.split("@")
        learned = fit != "planner"
        steps = 90 if learned and family != "recent32_hard" else 120 if learned else 100
        cost = .5 if family.startswith("dct") and learned else .6 if learned else 1.
        state = 4857 if family.startswith("dct") else 3577 if family == "recent32_hard" else 25281
        rows.append({"regime": regime, "seed": seed, "block": block, "initial_hit": hit, "arm": arm,
                     "steps": steps, "found": True, "updates": steps - 1, "state_bytes": state,
                     "init_seconds": 0., "update_seconds": 0., "decision_seconds": cost,
                     "setup_allocation_seconds": 0., "controller_seconds": cost, "environment_seconds": .1})
    return rows


def test_complete_summary_means_all_forty_rules_and_no_novelty_claim():
    summary = M.aggregate(passing_rows(), WEIGHTS)
    assert summary["readout_pilot_passes"] is True
    assert summary["learned_architecture_advantage_established"] is False
    assert summary["inherited_gate_revised"] is False
    assert summary["episodes"] == 1536
    for regime in summary["regimes"].values():
        assert regime["full_head_competent"] is True
        assert len(regime["blocks"]) == 8
        assert regime["family_means"]["dct16_neutral"]["steps"] == 90
        for panel in regime["criteria"].values():
            assert len(panel["checks"]) == 10 and all(panel["checks"].values())
            assert panel["block_gains_vs_recent_head"] == [30] * 8


def test_stratum_weighting_is_not_pooled_and_seed_means_are_equal():
    rows = passing_rows()
    for r in rows:
        if r["arm"] == "dct16_neutral@7901":
            r["steps"] = {1: 20, 2: 40, 3: 80}[r["initial_hit"]]
            r["updates"] = r["steps"] - 1
    result = M.aggregate(rows, WEIGHTS)["regimes"]["base"]
    assert result["means"]["dct16_neutral@7901"]["steps"] == 40
    assert result["family_means"]["dct16_neutral"]["steps"] == pytest.approx((40 + 90 + 90) / 3)


def test_each_fill_and_regime_must_pass_without_pooling():
    rows = passing_rows()
    for r in rows:
        if r["regime"] == "shift" and r["arm"].startswith("dct16_nearest@") and "planner" not in r["arm"]:
            r["decision_seconds"] = r["controller_seconds"] = .8000001
    summary = M.aggregate(rows, WEIGHTS)
    assert not summary["readout_pilot_passes"]
    assert summary["regimes"]["base"]["criteria"]["dct16_nearest"]["passes"]
    assert not summary["regimes"]["shift"]["criteria"]["dct16_nearest"]["checks"]["mean_controller_at_most_80pct_full_planner"]


def test_block_sign_and_full_head_competence_are_separate_requirements():
    rows = passing_rows()
    for r in rows:
        if r["arm"].startswith("dct16_neutral@") and "planner" not in r["arm"]:
            r["steps"] = 70 if r["block"] < 5 else 120
            r["updates"] = r["steps"] - 1
        if r["arm"] == "full_bayes@7903":
            r["steps"], r["updates"] = 106, 105
    summary = M.aggregate(rows, WEIGHTS)
    for result in summary["regimes"].values():
        assert result["criteria"]["dct16_neutral"]["checks"]["mean_moves_at_most_95pct_recent_head"]
        assert not result["criteria"]["dct16_neutral"]["checks"]["positive_blocks_vs_recent_head_at_least_six"]
        assert not result["full_head_competent"]
    assert not summary["readout_pilot_passes"]


def test_unsuccessful_episode_keeps_full_horizon_and_all_updates():
    rows = passing_rows()
    r = rows[0]
    r.update(found=False, steps=M.HORIZON, updates=M.HORIZON)
    result = M.aggregate(rows, WEIGHTS)["regimes"][r["regime"]]["means"][r["arm"]]
    assert result["found"] == 1 - .5 / 16
    assert result["steps"] == pytest.approx(90 + .5 * (M.HORIZON - 90) / 16)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "order", "early_failure", "updates", "cost", "nan"])
def test_summary_rejects_incomplete_or_inconsistent_evidence(defect):
    rows = passing_rows()
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows[1] = rows[0].copy()
    elif defect == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif defect == "early_failure":
        rows[0]["found"] = False
    elif defect == "updates":
        rows[0]["updates"] += 1
    elif defect == "cost":
        rows[0]["controller_seconds"] += 1
    else:
        rows[0]["environment_seconds"] = math.nan
    with pytest.raises(ValueError):
        M.aggregate(rows, WEIGHTS)


def bare_run(tmp_path):
    run = M.Run.__new__(M.Run)
    run.out, run.np, run.torch = tmp_path, np, torch
    run.receipt = {"training_updates": 0, "completed_fits": 0, "completed_episodes": 0,
                   "native_steps_attempted": 0, "native_steps_returned": 0, "status": "started"}
    run.check = lambda: None
    return run


@pytest.mark.parametrize("learning_rate", [0., .01])
def test_tiny_real_fit_train_only_moments_selection_sign_and_serialized_replay(tmp_path, monkeypatch, learning_rate):
    run = bare_run(tmp_path)
    monkeypatch.setattr(M, "FAMILIES", ("dct16_neutral",))
    monkeypatch.setattr(M, "FIT_SEEDS", (11,))
    monkeypatch.setattr(M, "CONFIGURATION", {**M.CONFIGURATION, "epochs": 2, "batch_size": 2,
                                            "checkpoint_epochs": [1, 2], "learning_rate": learning_rate})
    calls = []
    def moments(x):
        calls.append(x.copy())
        return head.fit_standardizer(x)
    def zero_head(dimension, seed):
        model = head.make_head(dimension, seed)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        return model
    run.head = SimpleNamespace(**{name: getattr(head, name) for name in
                                 ("standardize", "export_head", "predict", "parameter_count", "storage_bytes", "validate_head", "FrozenHead")},
                               fit_standardizer=moments, make_head=zero_head)
    tx = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]], dtype=np.float32)
    vx = tx[:2] + 100
    target = np.tile(np.array([.9, .1, 0., 0.], dtype=np.float32), (4, 1))
    legal = np.tile(np.array([1, 1, 0, 0], dtype=bool), (4, 1))
    heads = run.fit(({"dct16_neutral": tx}, target, legal, None),
                    ({"dct16_neutral": vx}, target[:2], legal[:2], None))
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], tx)
    runtime = heads["dct16_neutral@11"]
    restored = M.restore_head(tmp_path / "head-dct16_neutral-11.npz", np)
    assert type(restored["version"]) is str and type(restored["input_dim"]) is int
    assert restored["input_dim"] == 3
    np.testing.assert_array_equal(restored["mean"], [.5, .5, 0])
    record = json.loads((tmp_path / "fit-dct16_neutral-11.json").read_text())
    assert run.receipt["training_updates"] == 4 and run.receipt["completed_fits"] == 1
    assert len(record["curve"]) == 2
    assert record["checkpoint_sha256"] == M.sha(tmp_path / record["checkpoint"])
    assert run.head_setup_seconds["dct16_neutral@11"] >= 0
    assert runtime.predict(vx[0], [0, 1, 2, 3]) == head.predict(restored, vx[0], [0, 1, 2, 3])
    if learning_rate == 0:
        assert record["selected_epoch"] == 1  # first minimum wins exact ties
        assert record["validation_ce"] == pytest.approx(math.log(2), abs=1e-7)
    else:
        assert record["selected_epoch"] == 2
        assert record["curve"][-1]["validation_ce"] < record["curve"][0]["validation_ce"]
        assert restored["bias2"][0] < restored["bias2"][1]  # lower means better
        np.testing.assert_array_equal(restored["bias2"][2:], [0, 0])


def test_restore_rejects_non_scalar_metadata(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, version=np.array(["not-scalar"]), input_dim=3)
    with pytest.raises(ValueError, match="scalar checkpoint metadata"):
        M.restore_head(path, np)


def test_attempted_native_call_is_counted_when_fake_step_raises(tmp_path):
    run = bare_run(tmp_path)
    run.receipt["phase"] = "collection"
    def fail(*args, **kwargs):
        raise RuntimeError("fake-step-failure")
    with pytest.raises(RuntimeError, match="fake-step-failure"):
        run.step(SimpleNamespace(step=fail), 0)
    assert run.receipt["native_steps_attempted"] == 1
    assert run.receipt["native_steps_returned"] == 0


def test_qualification_native_cap_prevents_fake_call(tmp_path):
    run = bare_run(tmp_path)
    run.receipt.update(phase="qualification", native_steps_attempted=2048)
    def forbidden(*args, **kwargs):
        raise AssertionError("step must not be called")
    with pytest.raises(ValueError, match="qualification allocation"):
        run.step(SimpleNamespace(step=forbidden), 0)


def test_failed_auth_original_error_survives_broken_final_clock(tmp_path, monkeypatch):
    run = bare_run(tmp_path)
    run.start = None
    def broken_clock():
        raise RuntimeError("fake-clock-failure")
    def failed_auth():
        raise ValueError("primary-auth-failure")
    run.clock = SimpleNamespace(now_ns=broken_clock, backend="fake")
    monkeypatch.setattr(M, "SuspendClock", lambda: run.clock)
    run.bind = failed_auth
    with pytest.raises(ValueError, match="primary-auth-failure"):
        run.execute()
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["status"] == "failed" and "primary-auth-failure" in receipt["error"]
    assert receipt["finished_ns"] is None and receipt["wall_seconds"] is None
    assert "fake-clock-failure" in receipt["finalization_errors"][0]


def test_failed_auth_original_error_survives_failed_receipt_write(tmp_path, monkeypatch):
    run = bare_run(tmp_path)
    run.start, run.clock = None, SimpleNamespace(now_ns=lambda: 100, backend="fake")
    monkeypatch.setattr(M, "SuspendClock", lambda: run.clock)
    def failed_auth():
        raise ValueError("primary-auth-failure")
    def failed_write(*args, **kwargs):
        raise OSError("fake-write-failure")
    run.bind = failed_auth
    monkeypatch.setattr(M, "write", failed_write)
    with pytest.raises(ValueError, match="primary-auth-failure"):
        run.execute()


def test_clock_construction_failure_retains_failed_receipt_without_timing(tmp_path, monkeypatch):
    run = bare_run(tmp_path)
    run.clock, run.start = None, None
    def broken_constructor():
        raise RuntimeError("initial-clock-failure")
    def forbidden_bind():
        raise AssertionError("clock initialization must happen before binding")
    monkeypatch.setattr(M, "SuspendClock", broken_constructor)
    run.bind = forbidden_bind
    with pytest.raises(RuntimeError, match="initial-clock-failure"):
        run.execute()
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["status"] == "failed" and "initial-clock-failure" in receipt["error"]
    assert receipt["finished_ns"] is None and receipt["wall_seconds"] is None


def test_fake_complete_route_late_publication_failure_is_explicit(tmp_path, monkeypatch):
    """Exercise terminal lifecycle without importing any actual OTTO module."""
    run = bare_run(tmp_path)
    run.start = None
    clock = SimpleNamespace(now_ns=lambda: 100, backend="fake")
    monkeypatch.setattr(M, "SuspendClock", lambda: clock)
    run.clock = clock
    plan, launch = tmp_path / "input-plan.json", tmp_path / "input-launch.json"
    plan.write_text("{}")
    launch.write_text("{}")
    run.args = SimpleNamespace(plan=plan, plan_sha256=M.sha(plan), supervision=launch)
    def bind():
        run.start = 90
        run.plan = {"runtime_versions": {}}
        run.receipt["supervision_sha256"] = M.sha(launch)
    run.bind, run.authenticate = bind, lambda: None
    for suffix, name in (("sourcetracking", "SourceTracking"), ("heuristicpolicy", "HeuristicPolicy"), ("policy", None)):
        module = ModuleType(f"isotropic.classes.{suffix}")
        module.__file__ = str(ROOT / f"tmp/otto-source-review-01/isotropic/classes/{suffix}.py")
        if name:
            setattr(module, name, type(name, (), {}))
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, "isotropic", ModuleType("isotropic"))
    monkeypatch.setitem(sys.modules, "isotropic.classes", ModuleType("isotropic.classes"))
    def load(path, name):
        if path.endswith("study_otto_spectral_control.py"):
            return SimpleNamespace(Run=SimpleNamespace(qualify=lambda *args: ({}, {}, {})))
        return SimpleNamespace()
    monkeypatch.setattr(M, "load", load)
    monkeypatch.setattr(M, "THREAD_ENV", {})
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda b: None)
    def collect(split, first, count):
        return {}, np.ones((1, 4)), None, np.array([[first, 1, 0]])
    def fit(*args):
        run.receipt["completed_fits"] = 12
        return {}
    def evaluate(*args):
        run.receipt["completed_episodes"] = 2 * M.CASES * len(M.ARMS)
    run.collect, run.fit, run.evaluate = collect, fit, evaluate
    def limits():
        if (tmp_path / "receipt.json").exists():
            raise ValueError("late-published-cap-failure")
    run.check = limits
    with pytest.raises(ValueError, match="late-published-cap-failure"):
        run.execute()
    late = json.loads((tmp_path / "late-failure.json").read_text())
    assert late["status"] == "failed" and "late-published-cap-failure" in late["error"]
    assert run.receipt["native_steps_attempted"] == 0


class FakeActor:
    def __init__(self, public):
        self.step, self.position = public["step"], tuple(public["position"])
        self.probabilities = np.ones((1, 1))
        self.updates = []

    def update(self, public):
        assert not public["done"]
        self.updates.append(copy.deepcopy(public))
        self.step, self.position = public["step"], tuple(public["position"])

    def decode(self):
        raise AssertionError("learned actor must not decode or plan")

    def storage_bytes(self):
        return {"mutable_array_bytes": 4857}


def fake_public(env, step):
    return {"position": [26, 26], "step": step, "hit": -2 if env.done else 1,
            "done": env.done, "valid_actions": [] if env.done else [0, 1, 2, 3]}


def fake_environment(seed, *, ending=False):
    env = SimpleNamespace(source=np.array([0, 0]), p_source=np.ones((1, 1)), done=False, draw_log=[])
    def step(action, **kwargs):
        assert 0 <= action < 4
        env.done = ending
    env.step = step
    return env


def test_fake_collection_uses_shared_teacher_public_inputs_and_updates_censored_step(tmp_path, monkeypatch):
    run = bare_run(tmp_path)
    monkeypatch.setattr(M, "COLLECTION_HORIZON", 2)
    actors, feature_calls = [], []
    def actor_factory(family, public, *args):
        actor = FakeActor(public)
        actors.append(actor)
        return actor
    def features(family, actor, public, initial_hit, **kwargs):
        assert set(public) == head.PUBLIC_FIELDS
        assert actor.step == public["step"]
        feature_calls.append((family, public["step"]))
        return np.array([public["step"], initial_hit, 0], dtype=np.float32)
    run.models, run.kernels, run.SourceTracking = {"base": object()}, {"base": object()}, object()
    run.control = SimpleNamespace(COHORTS={"base": {"config": {}}}, public=lambda p: p)
    run.seeded = lambda cls, seed, config, **kw: fake_environment(seed)
    run.observation = fake_public
    run.saved = SimpleNamespace(make_actor=actor_factory, choose=lambda scores: (0, 1))
    run.analytic = SimpleNamespace(action_scores=lambda *args: [0., 1., 2., 3.])
    run.head = SimpleNamespace(features=features)
    run.receipt["phase"] = "collection"
    arrays, targets, masks, metadata = run.collect("train", 1001, 1)
    assert set(arrays) == set(M.FAMILIES)
    assert all(x.shape == (2, 3) for x in arrays.values())
    np.testing.assert_array_equal(metadata, [[1001, 1, 0], [1001, 1, 1]])
    np.testing.assert_array_equal(targets[0], targets[1])
    assert masks.all() and len(feature_calls) == 8
    assert all(len(a.updates) == 2 for a in actors)
    assert run.receipt["native_steps_returned"] == 2
    episodes = json.loads((tmp_path / "train-episodes.json").read_text())
    assert episodes == [{"seed": 1001, "initial_hit": 1, "steps": 2, "found": False}]


def test_fake_evaluation_skips_terminal_update_charges_censored_update_and_setup(tmp_path, monkeypatch):
    run = bare_run(tmp_path)
    monkeypatch.setattr(M, "HORIZON", 2)
    cases = [("base", 10, 0, 1, "dct16_neutral@7901"), ("shift", 11, 0, 1, "dct16_neutral@7901")]
    monkeypatch.setattr(M, "evaluation_order", lambda: iter(cases))
    actors = []
    def factory(family, public, *args):
        actor = FakeActor(public)
        actors.append(actor)
        return actor
    run.control = SimpleNamespace(COHORTS={r: {"config": {"lambda_over_dx": v}} for r, v in (("base", 3), ("shift", 4))}, public=lambda p: p)
    run.seeded = lambda cls, seed, config, **kw: fake_environment(seed, ending=seed == 10)
    run.SourceTracking, run.observation = object(), fake_public
    run.saved = SimpleNamespace(make_actor=factory, choose=lambda scores: (0, 1))
    def forbidden(*args):
        raise AssertionError("learned evaluation must not invoke analytic planner")
    run.analytic = SimpleNamespace(action_scores=forbidden)
    run.models = {r: SimpleNamespace(storage_bytes=lambda: {"shared": 1}) for r in ("base", "shift")}
    run.kernels = {r: object() for r in ("base", "shift")}
    seen = []
    def features(family, actor, public, initial_hit, **kwargs):
        assert set(public) == head.PUBLIC_FIELDS and actor.step == public["step"]
        seen.append(kwargs["sensing_length"])
        return np.array([0.], dtype=np.float32)
    run.head = SimpleNamespace(features=features)
    run.head_setup_seconds = {"dct16_neutral@7901": .48}
    run.qualified = {"initial_hit_weights": WEIGHTS, "shared_model_initialization_seconds": {"base": .96, "shift": 1.44}}
    saved = []
    def aggregate(rows, weights):
        saved.extend(rows)
        return {"readout_pilot_passes": False}
    monkeypatch.setattr(M, "aggregate", aggregate)
    run.receipt["phase"] = "evaluation"
    runtime = SimpleNamespace(predict=lambda *args: [0., 1., 2., 3.], storage_bytes=lambda: {"head": 1})
    run.evaluate({"dct16_neutral@7901": runtime})
    assert [r["steps"] for r in saved] == [1, 2]
    assert [r["updates"] for r in saved] == [0, 2]
    assert [len(a.updates) for a in actors] == [0, 2]
    assert all(r["environment_initialization_seconds"] >= 0 and r["draws"] == [] for r in saved)
    assert [r["setup_allocation_seconds"] for r in saved] == pytest.approx([.03, .04])
    for r in saved:
        assert r["controller_seconds"] == pytest.approx(sum(r[k] for k in
            ("init_seconds", "update_seconds", "decision_seconds", "setup_allocation_seconds")))
    assert seen == [3, 4, 4] and run.receipt["native_steps_returned"] == 3
    assert run.receipt["completed_episodes"] == 2
