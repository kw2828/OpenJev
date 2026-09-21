"""Synthetic representation-screen qualification; no native environment imports."""
from __future__ import annotations

import builtins
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_rocksample_policy_value import (
    FakeRandom,
    OpaqueState,
    ScriptedEnvironment,
    exit_outputs,
    macro,
    observation,
)

from openjev.research.rocksample_factorized_planning import FactorizedRockBelief
from openjev.research.rocksample_particle_belief import ParticleRockBelief

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/study_rocksample_representations.py"
ARMS = ("exit", "factorized_full", "factorized_recent128", "factorized_latest",
        "particle_full", "quality", "known_map", "privileged")
ROLES = ("exit", "full", "recent128", "latest", "full", "quality", "full", "privileged")
CONTROLS = ("factorized_recent128", "factorized_latest", "quality", "particle_full")


def load_runner(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("synthetic_representations", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner(monkeypatch):
    return load_runner(monkeypatch)


def test_import_does_not_load_native_or_neural_dependencies(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"jax", "jaxlib", "pobax", "gymnax", "torch", "brax", "navix"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = load_runner(monkeypatch)
    assert module.ARMS == ARMS
    assert module.MAP_SEEDS == tuple(range(13001, 13009))
    assert module.RESET_SEEDS == tuple(range(23001, 23005))


@pytest.mark.parametrize(("arm", "role"), list(zip(ARMS, ROLES, strict=True)))
def test_role_mapping_and_construction_cost_are_explicit(runner, monkeypatch, arm, role):
    prior = ParticleRockBelief(particles=2)
    env = ScriptedEnvironment([], privileged=arm in {"known_map", "privileged"})
    env._state = OpaqueState()  # The adapter must never request qualities.
    captured = []
    clock = iter((100.0, 100.25))
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(clock))

    def fake_episode(env_arg, params, actual_role, reset, transition, actual_prior, jax, emit, check, counters):
        captured.append(actual_prior)
        assert env_arg is env and actual_role == role
        assert reset == 23001 and transition == 440000
        return {"arm": actual_role, "filter_seconds": 2.0, "planning_seconds": 3.0,
                "controller_seconds": 5.0, "information": "public history only"}

    monkeypatch.setattr(runner.baseline, "play_episode", fake_episode)
    row = runner.play_episode(env, None, arm, 23001, 440000, prior, None, None, None, {})
    assert row["arm"] == arm
    assert row["representation_construction_seconds"] == 0.25
    assert row["filter_seconds"] == 2.25 and row["planning_seconds"] == 3.0
    assert row["controller_seconds"] == 5.25
    if arm.startswith("factorized_"):
        assert isinstance(captured[0], FactorizedRockBelief)
        assert captured[0].posterior.shape == (11, 110, 2)
    elif arm == "known_map":
        assert captured[0] is not prior
        np.testing.assert_array_equal(captured[0].maps, env._maps[None])
        np.testing.assert_array_equal(captured[0].quality_probabilities(), np.full(11, 0.5))
        assert row["information"] == "supplied true map; qualities inferred from public history"
    else:
        assert captured[0] is prior
    assert env.hidden_reads == (["map"] if arm == "known_map" else [])


@pytest.mark.parametrize("arm", ARMS)
def test_actual_loop_enforces_information_roles_and_common_public_ledger(runner, monkeypatch, arm):
    outputs, plans = exit_outputs(), []
    if arm != "exit":
        outputs = [(5, observation(rock=0), 0, False, False),
                   (4, observation(), 10, False, False), *outputs]
        plans = [macro("sense", [5]), macro("exploit", [4]), macro("exit", [1] * 10)]
    env = ScriptedEnvironment(outputs, privileged=arm in {"known_map", "privileged"})
    quality_reads = []

    class HiddenQualities:
        @property
        def rock_morality(self):
            assert arm == "privileged", "initial qualities leaked to another arm"
            quality_reads.append(True)
            return np.ones(11)

    env._state = (SimpleNamespace(env_state=HiddenQualities()) if arm == "privileged" else OpaqueState())
    prior = ParticleRockBelief(particles=8)
    original_quality = prior.conditional_quality_probabilities
    decisions, emitted = [], []
    counters = {"steps_attempted": 0, "steps_returned": 0}

    def fixed_plan(belief, position, ledger, remaining_steps, checks_remaining):
        decisions.append({"quality": belief.quality_probabilities().copy(), "ledger": ledger.copy(),
                          "position": position, "remaining": remaining_steps, "checks": checks_remaining})
        return copy.deepcopy(plans[len(decisions) - 1])

    monkeypatch.setattr(runner.baseline, "plan_action", fixed_plan)
    row = runner.play_episode(env, object(), arm, 23001, 440000, prior,
                              SimpleNamespace(random=FakeRandom),
                              lambda kind, value: emitted.append((kind, copy.deepcopy(value))),
                              lambda: None, counters)
    assert row["arm"] == arm and env.reset_keys == [23001]
    assert env.hidden_reads == (["map"] if arm in {"known_map", "privileged"} else [])
    assert len(quality_reads) == int(arm == "privileged")
    assert row["steps"] == (10 if arm == "exit" else 12)
    assert counters["steps_attempted"] == counters["steps_returned"] == row["steps"]
    assert row["raw_return"] == (10 if arm == "exit" else 20)
    assert row["exited"] and not row["truncated"] and not env.outputs
    assert row["controller_seconds"] == pytest.approx(row["filter_seconds"] + row["planning_seconds"])
    assert all("arm" not in payload for _, payload in emitted)  # Outer writer supplies the eight-arm identity.
    np.testing.assert_array_equal(prior.conditional_quality_probabilities, original_quality)
    if arm != "exit":
        assert [d["checks"] for d in decisions] == [64, 63, 63]
        assert [d["ledger"].sum() for d in decisions] == [0, 0, 1]
        assert decisions[-1]["ledger"][0, 0]
    if arm == "known_map":
        assert [d["quality"][0] for d in decisions] == [0.5, 1.0, 0.0]
        assert row["information"] == "supplied true map; qualities inferred from public history"
    elif arm == "privileged":
        assert [d["quality"][0] for d in decisions] == [1.0, 1.0, 0.0]
        assert row["information"] == "true map and qualities"
    else:
        assert row["information"] == "public history only"
    if arm.startswith("factorized_"):
        assert row["representation_diagnostics"]["planning_decisions"] == row["plans"] == 3
        assert row["ess_evaluated_plans"] == 0  # A factorized table does not acquire a fictitious ESS.
    else:
        assert row["representation_diagnostics"]["planning_decisions"] == 0


def test_unknown_role_rejected_before_any_construction(runner, monkeypatch):
    monkeypatch.setattr(runner, "FactorizedRockBelief", lambda: pytest.fail("unexpected construction"))
    with pytest.raises(ValueError, match="unknown representation arm"):
        runner.play_episode(None, None, "full", None, None, None, None, None, None, None)


def test_factorized_diagnostics_aggregate_all_decisions_without_clipping(runner, monkeypatch):
    inputs = [(0.4, 0, -2.0, 3.0), (1.25, 2, -12.0, 14.0), (1.1, 1, -11.0, 11.0)]
    emitted = []

    def fake_episode(env, params, arm, reset, transition, prior, jax, emit, check, counters):
        for step, (occupancy, overfull, low, high) in enumerate(inputs):
            emit("decisions", {"step": step, "belief_diagnostics": {
                "maximum_cell_occupancy": occupancy, "overfull_cells": overfull,
                "minimum_expected_reward": low, "maximum_expected_reward": high}})
        return {"filter_seconds": 0.0, "controller_seconds": 0.0}

    monkeypatch.setattr(runner.baseline, "play_episode", fake_episode)
    row = runner.play_episode(None, None, "factorized_full", None, None, None, None,
                              lambda kind, payload: emitted.append((kind, copy.deepcopy(payload))), None, {})
    assert len(emitted) == 3
    assert row["representation_diagnostics"] == {
        "planning_decisions": 3, "maximum_cell_occupancy": 1.25, "overfull_cell_decisions": 2,
        "overfull_cells_total": 3, "minimum_expected_reward": -12.0, "maximum_expected_reward": 14.0}


def cohort(runner):
    rows = []
    returns = {arm: 20.0 for arm in ARMS}
    returns.update(exit=10.0, factorized_full=30.0, known_map=40.0, privileged=50.0)
    for mi, map_seed in enumerate(runner.MAP_SEEDS):
        for ei, reset_seed in enumerate(runner.RESET_SEEDS):
            for arm in ARMS:
                plans = 1 + mi + ei
                rows.append({"map_seed": map_seed, "reset_seed": reset_seed, "arm": arm,
                             "raw_return": returns[arm], "discounted_return_099": returns[arm] / 2,
                             "steps": 10 + mi + ei, "checks": 1, "samples": 1, "good_samples": 1,
                             "bad_samples": 0, "empty_samples": 0, "controller_seconds": 0.75,
                             "filter_seconds": 0.25, "planning_seconds": 0.5, "environment_seconds": 0.125,
                             "representation_construction_seconds": 0.125, "plans": plans,
                             "exited": True, "truncated": False,
                             "representation_diagnostics": {
                                 "planning_decisions": plans, "maximum_cell_occupancy": 0.1,
                                 "overfull_cell_decisions": 0, "overfull_cells_total": 0,
                                 "minimum_expected_reward": -1.0, "maximum_expected_reward": 1.0}})
    return rows


def test_all_256_cases_and_thirteen_conditions_are_retained(runner):
    rows = cohort(runner)
    result = runner.summarize(rows)
    assert result["episodes"] == 256 and result["learned_pilot_admitted"] is True
    expected = [f"{prefix}_vs_{control}" for control in CONTROLS for prefix in ("raw_gain", "positive_maps")]
    expected += ["factorized_full_gain_vs_exit", "known_map_gain_vs_exit", "privileged_gain_vs_exit",
                 "complete_cohort", "factorized_observed_forecast_bounds"]
    assert [c["name"] for c in result["criteria"]] == expected
    assert len(result["criteria"]) == 13 and all(c["passes"] for c in result["criteria"])
    for row in rows:
        if row["arm"] == "factorized_full":
            row["raw_return"] = [10, 20, 30, 40][runner.RESET_SEEDS.index(row["reset_seed"])]
            row["steps"] = [10, 100, 200, 1000][runner.RESET_SEEDS.index(row["reset_seed"])]
    result = runner.summarize(rows)
    assert result["scores_equal_map_means"]["factorized_full"]["raw_return"] == 25
    assert result["scores_equal_map_means"]["factorized_full"]["steps"] == 327.5
    assert all(m["scores"]["factorized_full"]["raw_return"] == 25 for m in result["by_map"])
    assert result["representation_diagnostics"]["factorized_full"]["planning_decisions"] == 192


NUMERICAL_GATES = [f"{prefix}_vs_{control}" for control in CONTROLS
                   for prefix in ("raw_gain", "positive_maps")]
NUMERICAL_GATES += ["factorized_full_gain_vs_exit", "known_map_gain_vs_exit", "privileged_gain_vs_exit",
                    "factorized_observed_forecast_bounds"]


@pytest.mark.parametrize("name", NUMERICAL_GATES)
def test_each_numerical_condition_can_fail_alone_and_blocks_admission(runner, name):
    rows = cohort(runner)
    for row in rows:
        if name.startswith("raw_gain") and row["arm"] == name.split("_vs_")[1]:
            row["raw_return"] = 20.0 if row["reset_seed"] == runner.RESET_SEEDS[0] else 30.0
        elif name.startswith("positive_maps") and row["arm"] == name.split("_vs_")[1]:
            if row["map_seed"] in runner.MAP_SEEDS[:3]:
                row["raw_return"] = 30.0
        elif name == "factorized_full_gain_vs_exit":
            if row["arm"] == "exit":
                row["raw_return"] = 30.0
            elif row["arm"] == "known_map":
                row["raw_return"] = 50.0
        elif name in ("known_map_gain_vs_exit", "privileged_gain_vs_exit"):
            if row["arm"] == name.removesuffix("_gain_vs_exit"):
                row["raw_return"] = 20.0
        elif name == "factorized_observed_forecast_bounds" and row["arm"] == "factorized_full":
            row["representation_diagnostics"]["maximum_cell_occupancy"] = 1.01
            row["representation_diagnostics"]["overfull_cells_total"] = 1
            row["representation_diagnostics"]["overfull_cell_decisions"] = 1
    summary = runner.summarize(rows)
    assert summary["learned_pilot_admitted"] is False
    assert [c["name"] for c in summary["criteria"] if not c["passes"]] == [name]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "extra", "wrong_map", "wrong_reset", "wrong_arm"])
def test_cohort_closure_rejects_missing_duplicate_and_unplanned_cases(runner, defect):
    rows = cohort(runner)
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif defect == "extra":
        rows.append(copy.deepcopy(rows[0]))
    else:
        key, value = {"wrong_map": ("map_seed", 12001), "wrong_reset": ("reset_seed", 22001),
                      "wrong_arm": ("arm", "full")}[defect]
        rows[-1][key] = value
    with pytest.raises(ValueError, match="fixed representation cohort"):
        runner.summarize(rows)


def test_exact_gain_thresholds_and_strictly_positive_map_counts(runner):
    rows = cohort(runner)
    for row in rows:
        if row["arm"] == "factorized_full":
            row["raw_return"] = 10.0 if row["reset_seed"] in runner.RESET_SEEDS[:2] else 20.0
        elif row["arm"] in CONTROLS:
            row["raw_return"] = 10.0
        elif row["arm"] in ("known_map", "privileged"):
            row["raw_return"] = 30.0
    assert runner.summarize(rows)["learned_pilot_admitted"] is True
    rows = cohort(runner)
    for row in rows:
        if row["map_seed"] in runner.MAP_SEEDS[:2] and row["arm"] in CONTROLS:
            row["raw_return"] = 30.0
    summary = runner.summarize(rows)
    assert summary["learned_pilot_admitted"] is True
    for criterion in summary["criteria"]:
        if criterion["name"].startswith("positive_maps"):
            assert criterion["value"] == 6
            assert criterion["map_gains"] == [0, 0, 10, 10, 10, 10, 10, 10]


@pytest.mark.parametrize(("field", "bad"), [
    ("maximum_cell_occupancy", np.nextafter(1.0, np.inf)),
    ("minimum_expected_reward", np.nextafter(-10.0, -np.inf)),
    ("maximum_expected_reward", np.nextafter(10.0, np.inf)),
    ("overfull_cells_total", 1),
])
def test_forecast_bounds_accept_exact_limits_and_reject_any_excursion(runner, field, bad):
    rows = cohort(runner)
    full = [r["representation_diagnostics"] for r in rows if r["arm"] == "factorized_full"]
    for row in full:
        row.update(maximum_cell_occupancy=1.0, minimum_expected_reward=-10.0, maximum_expected_reward=10.0)
    assert runner.summarize(rows)["learned_pilot_admitted"] is True
    full[0][field] = bad
    if field == "overfull_cells_total":
        full[0]["overfull_cell_decisions"] = 1
    result = runner.summarize(rows)
    assert [c["name"] for c in result["criteria"] if not c["passes"]] == ["factorized_observed_forecast_bounds"]


@pytest.mark.parametrize("arm", ARMS[1:4])
@pytest.mark.parametrize("field", ["planning_decisions", "maximum_cell_occupancy", "minimum_expected_reward", "maximum_expected_reward"])
def test_missing_or_nonfinite_diagnostics_cannot_silently_pass(runner, arm, field):
    rows = cohort(runner)
    target = next(r["representation_diagnostics"] for r in rows if r["arm"] == arm)
    target[field] = 0 if field == "planning_decisions" else np.nan
    with pytest.raises(ValueError):
        runner.summarize(rows)


@pytest.mark.parametrize("changes", [
    {"planning_decisions": 2},  # First episode has exactly one actual decision.
    {"planning_decisions": True},
    {"planning_decisions": 1.0},
    {"overfull_cell_decisions": -1},
    {"overfull_cell_decisions": 0.5},
    {"overfull_cell_decisions": True},
    {"overfull_cells_total": -1},
    {"overfull_cells_total": 0.5},
    {"overfull_cells_total": True},
    {"overfull_cell_decisions": 2, "overfull_cells_total": 2},
    {"overfull_cell_decisions": 1, "overfull_cells_total": 0},
    {"overfull_cell_decisions": 0, "overfull_cells_total": 1},
    {"overfull_cell_decisions": 1, "overfull_cells_total": 111},
])
def test_diagnostic_counts_must_cover_actual_plans_and_possible_cells(runner, changes):
    rows = cohort(runner)
    first = next(r for r in rows if r["arm"] == "factorized_full")
    assert first["plans"] == 1
    first["representation_diagnostics"].update(changes)
    with pytest.raises(ValueError, match="coverage or counts disagree"):
        runner.summarize(rows)


@pytest.mark.parametrize("field", ["maximum_cell_occupancy", "minimum_expected_reward", "maximum_expected_reward"])
def test_nonfinite_decision_diagnostic_fails_before_forwarding_record(runner, monkeypatch, field):
    emitted = []

    def fake_episode(env, params, arm, reset, transition, prior, jax, emit, check, counters):
        diagnostic = {"maximum_cell_occupancy": 0.1, "overfull_cells": 0,
                      "minimum_expected_reward": -1.0, "maximum_expected_reward": 1.0}
        diagnostic[field] = np.nan
        emit("decisions", {"belief_diagnostics": diagnostic})
        pytest.fail("nonfinite diagnostic escaped validation")

    monkeypatch.setattr(runner.baseline, "play_episode", fake_episode)
    with pytest.raises(ValueError, match="nonfinite factorized diagnostics"):
        runner.play_episode(None, None, "factorized_full", None, None, None, None,
                            lambda kind, payload: emitted.append((kind, payload)), None, {})
    assert not emitted
