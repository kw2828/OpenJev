"""Artificial scoring/lifecycle and fake public-actor tests; no OTTO execution."""
from __future__ import annotations

import builtins
import copy
import importlib.util
import json
import math
import os
import sys
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/study_otto_spectral_control.py"


def load():
    spec = importlib.util.spec_from_file_location("synthetic_spectral_control", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def m():
    return load()


def qualified():
    return {"status": "completed", "checks": [{"name": "synthetic", "passes": True}], "native_steps": 100,
            "shared_model_initialization_seconds": {"base": .096, "shift": .192}}


def weights():
    return {"base": {1: .5, 2: .25, 3: .25}, "shift": {1: .25, 2: .5, 3: .25}}


def rows(m):
    result = []
    for name, seed, block, hit, arm in m.case_order():
        candidate = arm in m.CANDIDATES
        steps = 101 if candidate else 100 if arm in ("full_bayes", "exact_log") else 140
        allocation = qualified()["shared_model_initialization_seconds"][name] / 96 if candidate or arm == "exact_log" else 0
        result.append({"cohort": name, "seed": seed, "block": block, "initial_hit": hit, "arm": arm,
                       "steps": steps, "capped_time": steps, "found": True, "stuck_steps": 0,
                       "actor_initialization_seconds": .1, "update_seconds": .2, "decode_seconds": .2,
                       "planner_seconds": .7 if candidate else .5, "shared_initialization_allocation_seconds": allocation,
                       "controller_seconds": (1.2 if candidate else 1) + allocation,
                       "environment_initialization_seconds": .1, "environment_seconds": .2, "episode_seconds": 3,
                       "state_array_bytes": 4857 if candidate else 25281, "update_calls": steps - 1,
                       "decode_calls": steps, "planner_calls": steps})
    return result


def score(m, cohort=None):
    return m.summarize(rows(m) if cohort is None else cohort, weights(), qualification=qualified())


def test_import_has_no_numeric_or_environment_work(monkeypatch):
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        assert name.split(".")[0] not in {"numpy", "scipy", "isotropic", "torch", "jax", "tensorflow"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    load()


def test_exact_cohort_rotation_and_worst_case_cap(m):
    order = list(m.case_order())
    assert len(order) == len(set(order)) == 1152
    assert m.LIMITS["native_steps"] == 1152 * 2188 + 2048
    assert [r[4] for r in order[:6]] == list(m.ARMS)
    assert [r[4] for r in order[6:12]] == list(m.ARMS[1:] + m.ARMS[:1])
    for name, first in (("base", 630001), ("shift", 640001)):
        chosen = [r for r in order if r[0] == name]
        assert {r[1] for r in chosen} == set(range(first, first + 96))
        assert all(sum(r[3] == h and r[4] == arm for r in chosen) == 32 for h in (1, 2, 3) for arm in m.ARMS)
        assert m.COHORTS[name]["config"]["Ngrid"] == 53 and m.COHORTS[name]["config"]["Nhits"] == 4


def test_all_four_cells_and_separate_stricter_utility(m):
    result = score(m)
    assert result["compact_control_viable"] is True
    assert result["utility_compute_advantage"] is False
    assert result["learned_pilot_admission"] is result["inherited_gate_revised"] is False
    for cohort in result["cohorts"].values():
        for arm in m.CANDIDATES:
            assert len(cohort["criteria_by_candidate"][arm]) == 10
            assert len(cohort["utility_criteria_by_candidate"][arm]) == 4
            assert all(c["passes"] for c in cohort["criteria_by_candidate"][arm])


@pytest.mark.parametrize("defect", ["missing", "duplicate", "order", "seed", "stratum", "block", "early_failure", "calls", "cost", "nonfinite"])
def test_cohort_rejects_incomplete_or_inconsistent_rows(m, defect):
    cohort = rows(m)
    if defect == "missing":
        cohort.pop()
    elif defect == "duplicate":
        cohort[-1] = copy.deepcopy(cohort[0])
    elif defect == "order":
        cohort[0], cohort[1] = cohort[1], cohort[0]
    elif defect in ("seed", "block", "stratum"):
        cohort[0]["initial_hit" if defect == "stratum" else defect] += 1
    elif defect == "early_failure":
        cohort[0]["found"] = False
    elif defect == "calls":
        cohort[0]["update_calls"] += 1
    elif defect == "cost":
        cohort[0]["controller_seconds"] += .1
    else:
        cohort[0]["episode_seconds"] = math.nan
    with pytest.raises(ValueError):
        score(m, cohort)


def test_regime_specific_weights_and_no_pooling_rescue(m):
    cohort = rows(m)
    for r in cohort:
        if r["arm"] == "dct16_neutral":
            r["steps"] = r["capped_time"] = 90 if r["initial_hit"] == 1 else 110
            r["decode_calls"] = r["planner_calls"] = r["steps"]
            r["update_calls"] = r["steps"] - 1
    result = score(m, cohort)
    assert result["cohorts"]["base"]["means"]["dct16_neutral"]["capped_time"] == 100
    assert result["cohorts"]["shift"]["means"]["dct16_neutral"]["capped_time"] == 105
    for r in cohort:
        if r["cohort"] == "shift" and r["arm"] == "dct16_neutral" and r["initial_hit"] == 2:
            r["steps"] += 1
            r["capped_time"] += 1
            r["decode_calls"] += 1
            r["planner_calls"] += 1
            r["update_calls"] += 1
    result = score(m, cohort)
    assert result["cohorts"]["base"]["compact_by_candidate"]["dct16_neutral"] is True
    assert result["compact_control_viable"] is False


def baseline_means(m):
    return score(m)["cohorts"]["base"]["means"]


@pytest.mark.parametrize("change,failed", [
    ("full_success", "full_success_at_least_95pct"),
    ("found_full", "no_failure_regression_vs_full_bayes"),
    ("found_exact", "no_failure_regression_vs_exact_log"),
    ("time_full", "time_at_most_105pct_full_bayes"),
    ("time_exact", "time_at_most_105pct_exact_log"),
    ("recent", "strict_time_gain_vs_recent32"),
    ("hard", "strict_time_gain_vs_recent32_hard"),
    ("blocks", "positive_blocks_vs_recent32_hard_at_least_six"),
    ("state", "state_at_most_20pct_full"),
    ("cost", "controller_at_most_150pct_full"),
])
def test_each_frozen_compact_condition(m, change, failed):
    means = baseline_means(m)
    c = means["dct16_neutral"]
    blocks = [{"means": copy.deepcopy(means)} for _ in range(8)]
    if change == "full_success":
        means["full_bayes"]["found"] = .949
    elif change == "found_full":
        c["found"] = .99
        means["exact_log"]["found"] = .98
    elif change == "found_exact":
        c["found"] = .99
        means["full_bayes"]["found"] = .98
    elif change == "time_full":
        c["capped_time"] = 105.000001
    elif change == "time_exact":
        means["exact_log"]["capped_time"] = 90
    elif change in ("recent", "hard"):
        means["recent32" if change == "recent" else "recent32_hard"]["capped_time"] = c["capped_time"]
    elif change == "blocks":
        for block in blocks[:3]:
            block["means"]["recent32_hard"]["capped_time"] = c["capped_time"]
    elif change == "state":
        c["state_array_bytes"] = .2 * means["full_bayes"]["state_array_bytes"] + 1
    else:
        c["controller_seconds"] = 1.5 * means["full_bayes"]["controller_seconds"] + 1e-8
    conditions, _ = m.criteria(means, blocks, "dct16_neutral")
    assert next(item for item in conditions if item["name"] == failed)["passes"] is False


def test_utility_requires_strict_gain_no_compensation(m):
    means = baseline_means(m)
    means["dct16_neutral"] = copy.deepcopy(means["full_bayes"])
    blocks = [{"means": means} for _ in range(8)]
    _, conditions = m.criteria(means, blocks, "dct16_neutral")
    assert [c["passes"] for c in conditions] == [True, True, True, False]
    means["dct16_neutral"]["capped_time"] -= 1
    assert all(c["passes"] for c in m.criteria(means, blocks, "dct16_neutral")[1])
    means["dct16_neutral"]["controller_seconds"] += .01
    assert not all(c["passes"] for c in m.criteria(means, blocks, "dct16_neutral")[1])


@pytest.mark.parametrize("bad", [0, 2049])
def test_qualification_bound_before_scoring(m, bad):
    q = qualified()
    q["native_steps"] = bad
    with pytest.raises(ValueError, match="qualification"):
        m.summarize(rows(m), weights(), qualification=q)


def test_attempted_call_survives_native_exception(m, tmp_path):
    run = m.Run(SimpleNamespace(output=tmp_path / "run"))
    run.check = lambda: None
    run.receipt["phase"] = "qualification"

    class Broken:
        def step(self, *_args, **_kwargs):
            raise RuntimeError("fake native failure")

    with pytest.raises(RuntimeError, match="fake native"):
        run.step(Broken(), 0)
    assert run.receipt["native_steps_attempted"] == 1
    assert run.receipt["native_steps_returned"] == 0
    run.receipt["native_steps_attempted"] = 2048
    with pytest.raises(ValueError, match="qualification allocation"):
        run.step(Broken(), 0)
    assert run.receipt["native_steps_attempted"] == 2048


@pytest.mark.parametrize("done", [True, False])
def test_fake_autonomous_route_terminal_and_censored_update(m, tmp_path, monkeypatch, done):
    run = m.Run(SimpleNamespace(output=tmp_path / "run"))
    run.check = lambda: None
    monkeypatch.setattr(m, "HORIZON", 2)
    monkeypatch.setattr(m, "case_order", lambda: iter([("base", 630001, 0, 1, "dct16_neutral")]))
    captured = {}
    monkeypatch.setattr(m, "summarize", lambda rows, *_args, **_kwargs: captured.update(rows=rows) or {
        "compact_control_viable": False, "utility_compute_advantage": False})
    Public = namedtuple("Public", "position hit done step valid_actions")
    calls = []

    class Env:
        source = np.array([1, 1])
        agent_stuck = False
        draw_log = ()
        t = 0

        def step(self, action, *, hit, quiet):
            assert action == 1 and hit is None and quiet
            self.t += 1
            calls.append(("env", self.t))
            return (-2 if done and self.t == 2 else 0), 0, done and self.t == 2

    class Actor:
        def storage_bytes(self):
            return {"state_array_bytes": 4857}

        def decode(self):
            calls.append(("decode",))
            p = np.full((53, 53), 1 / 2809)
            return np.log(p), p

        def update(self, value):
            assert set(value) == {"position", "hit", "done", "step", "valid_actions"}
            assert not value["done"]
            calls.append(("update", value["step"]))

    def make_actor(arm, packet, model, kernel, analytic, numeric):
        assert arm == "dct16_neutral" and set(packet) == {"position", "hit", "done", "step", "valid_actions"}
        assert not hasattr(model, "source") and numeric is np
        return Actor()

    saved = SimpleNamespace(make_actor=make_actor, validate_decoded=lambda *_: None, choose=lambda _: (1, 1))
    analytic = SimpleNamespace(action_scores=lambda *_: [1., 0., 1., 1.])
    obs = lambda env, step: Public((26 + env.t, 26), -2 if done and env.t == 2 else 1 if env.t == 0 else 0,
                                  done and env.t == 2, step, () if done and env.t == 2 else (0, 1, 2, 3))
    model = SimpleNamespace(storage_bytes=lambda: {"shared_array_bytes": 100})
    q = qualified()
    q["initial_hit_weights"] = weights()
    run.cohort(object, lambda *_args, **_kwargs: Env(), obs, saved, analytic, np, q, {"base": model}, {"base": object()})
    row = captured["rows"][0]
    assert row["steps"] == 2 and row["found"] is done and row["update_calls"] == 2 - int(done)
    assert row["shared_initialization_allocation_seconds"] == .001
    assert row["controller_seconds"] == pytest.approx(sum(row[k] for k in m.TIMES))
    assert calls == [("decode",), ("env", 1), ("update", 1), ("decode",), ("env", 2)] + ([] if done else [("update", 2)])
    records = [json.loads(line) for line in (run.out / "transitions.jsonl").read_text().splitlines()]
    assert records[-1]["update_seconds"] == 0 if done else records[-1]["update_seconds"] >= 0
    assert "source_evaluation_only" not in records[-1]["public"]


def test_failure_before_import_preserves_receipt(m, tmp_path, monkeypatch):
    run = m.Run(SimpleNamespace(output=tmp_path / "run"))
    monkeypatch.setattr(m, "SuspendClock", lambda: SimpleNamespace(backend="synthetic", now_ns=lambda: 10))

    def denied():
        raise ValueError("plan hash mismatch")

    run.bind = denied
    with pytest.raises(ValueError, match="plan hash"):
        run.execute()
    receipt = json.loads((run.out / "receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["native_steps_attempted"] == 0
    assert "plan hash mismatch" in receipt["error"] and not (run.out / "imports.json").exists()


def test_clock_initialization_failure_is_null_not_zero(m, tmp_path, monkeypatch):
    run = m.Run(SimpleNamespace(output=tmp_path / "run"))

    def broken():
        raise RuntimeError("clock unavailable")

    monkeypatch.setattr(m, "SuspendClock", broken)
    with pytest.raises(RuntimeError, match="clock unavailable"):
        run.execute()
    receipt = json.loads((run.out / "receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["native_elapsed_seconds"] is None


@pytest.mark.parametrize("field", ["command", "pid", "cap_seconds", "deadline_ns", "clock_backend", "watchdog_sha256", "clock_source_sha256"])
def test_live_supervision_identity_rejects_mismatches(m, tmp_path, monkeypatch, field):
    monkeypatch.setattr(m, "ROOT", Path.cwd())
    args = SimpleNamespace(output=tmp_path / "run", supervision=tmp_path / "launch.json", plan=tmp_path / "plan.json", plan_sha256="wrong")
    run = m.Run(args)
    run.start = 20
    run.clock = SimpleNamespace(backend="synthetic", now_ns=lambda: 25)
    launch = {"command": [sys.executable, *sys.argv], "pid": os.getpid(), "pgid": os.getpgrp(),
              "parent_pid": os.getppid(), "cap_seconds": 900, "clock_backend": "synthetic", "cwd": str(Path.cwd()),
              "started_ns": 10, "deadline_ns": 900 * 10**9 + 10,
              "watchdog_sha256": m.PINNED["scripts/supervise_dialogue_observation_v2.py"],
              "clock_source_sha256": m.PINNED["src/openjev/research/suspend_clock.py"]}
    launch[field] = [] if field == "command" else -1 if field in {"pid", "cap_seconds", "deadline_ns"} else "bad"
    args.supervision.write_text(json.dumps(launch))
    with pytest.raises(ValueError, match="supervisor"):
        run.bind()
