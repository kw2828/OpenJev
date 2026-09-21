"""Synthetic three-stratum OTTO cohort scoring checks; never import or run its environment."""
from __future__ import annotations

import builtins
import copy
import importlib.util
import math
from collections import Counter
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/study_otto_large_memory.py"
ARMS = ("space_full", "space_recent32", "space_recent8", "space_initial",
        "info_full", "info_recent32", "info_recent8", "info_initial")
CRITERIA = {
    "full_success_at_least_95pct",
    "gain_vs_recent32_at_least_10pct",
    "gain_vs_recent32_at_least_two_steps",
    "positive_blocks_vs_recent32_at_least_six",
    "no_failure_regression_vs_space_recent32",
    "no_failure_regression_vs_space_recent8",
    "strict_time_gain_vs_space_recent8",
    "strict_time_gain_vs_space_initial",
    "complete_cohort",
    "structural_qualification",
}


def load_runner():
    spec = importlib.util.spec_from_file_location("synthetic_otto_large_study", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner():
    return load_runner()


def qualification():
    return {"status": "completed", "native_steps": 100,
            "checks": [{"name": "synthetic public-interface check", "passes": True}]}


def cohort():
    times = dict(zip(ARMS, (18, 20, 24, 30, 16, 22, 28, 36), strict=True))
    return [
        {"seed": 610001 + case, "arm": arm, "block": case // 12,
         "initial_hit": 1 + (case % 12) // 4,
         "steps": times[arm], "capped_time": times[arm], "found": True,
         "stuck_steps": 0, "initialization_seconds": 0.25,
         "filter_seconds": 0.5, "planning_seconds": 0.75,
         "controller_seconds": 1.5, "environment_seconds": 0.125}
        for case in range(96) for arm in ARMS
    ]


def score(runner, rows, weights=None, qualified=None):
    return runner.summarize(rows, {1: 0.5, 2: 0.25, 3: 0.25} if weights is None else weights,
                            qualification=qualification() if qualified is None else qualified)


def conditions(summary):
    return {item["name"]: item for item in summary["criteria"]}


def set_time(rows, arm, steps, *, block=None):
    for row in rows:
        if row["arm"] == arm and (block is None or row["block"] == block):
            row["steps"] = row["capped_time"] = steps


def fail(rows, arm, seeds):
    for row in rows:
        if row["arm"] == arm and row["seed"] in seeds:
            row.update(found=False, steps=2188, capped_time=2188)


def test_import_cannot_load_simulator_or_neural_dependencies(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {
            "isotropic", "otto", "scipy", "numpy", "torch", "jax", "tensorflow",
        }, f"unexpected execution dependency: {name}"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = load_runner()
    assert module.ARMS == ARMS
    assert module.HORIZON == 2188
    assert module.GRID_SIZE == 53 and module.NHITS == 4
    assert module.STRATA == (1, 2, 3)
    assert module.EPISODES == 768 and module.CASES == 96
    assert module.FIRST_SEED == 610001 and module.CASES_PER_BLOCK == 12
    assert module.QUALIFICATION_STEP_CAP == 2000
    assert module.NATIVE_STEP_CAP == 1690000
    assert module.CONFIG == {"Ndim": 2, "lambda_over_dx": 3.0, "R_dt": 2.0,
                             "norm_Poisson": "Euclidean", "Ngrid": None, "Nhits": None}


def test_complete_passing_cohort_has_all_ten_conditions_and_no_input_mutation(runner):
    rows, receipt = cohort(), qualification()
    before = copy.deepcopy((rows, receipt))
    result = score(runner, rows, qualified=receipt)
    assert (rows, receipt) == before
    assert result["episodes"] == 768
    assert set(conditions(result)) == CRITERIA
    assert all(c["passes"] is True for c in result["criteria"])
    assert result["learned_pilot_opportunity"] is True
    assert result["means"]["space_full"]["capped_time"] == 18
    assert conditions(result)["complete_cohort"]["value"] == 768
    assert Counter((r["arm"], r["initial_hit"]) for r in rows) == {
        (arm, hit): 32 for arm in ARMS for hit in (1, 2, 3)
    }
    assert Counter((r["arm"], r["initial_hit"], r["block"]) for r in rows) == {
        (arm, hit, block): 4 for arm in ARMS for hit in (1, 2, 3) for block in range(8)
    }


def test_stratum_weights_apply_before_population_and_block_means(runner):
    rows = cohort()
    for row in rows:
        # Equal sampled counts must not turn the declared mixture into equal thirds.
        steps = 10 + 40 * (row["initial_hit"] - 1) + row["block"]
        row["steps"] = row["capped_time"] = steps
    result = score(runner, rows, weights={1: 0.5, 2: 0.25, 3: 0.25})
    assert result["strata"][1]["space_full"]["capped_time"] == 13.5
    assert result["strata"][2]["space_full"]["capped_time"] == 53.5
    assert result["strata"][3]["space_full"]["capped_time"] == 93.5
    assert result["means"]["space_full"]["capped_time"] == 43.5
    assert result["means"]["space_full"]["capped_time"] != 53.5
    assert [b["means"]["space_full"]["capped_time"] for b in result["blocks"]] == [
        40 + b for b in range(8)
    ]
    for arm in ARMS:
        assert result["means"][arm]["controller_seconds"] == 1.5


@pytest.mark.parametrize("failed", [False, True])
def test_step_2188_found_and_censored_episodes_have_same_cost_different_success(runner, failed):
    rows = cohort()
    selected = next(r for r in rows if r["arm"] == "space_full" and r["seed"] == 610001)
    selected.update(steps=2188, capped_time=2188, found=not failed)
    result = score(runner, rows, weights={1: 0.5, 2: 0.25, 3: 0.25})
    assert result["means"]["space_full"]["capped_time"] == (63 * 18 + 2188) / 64
    assert result["means"]["space_full"]["found"] == (63 / 64 if failed else 1)


def test_success_at_95pct_is_inclusive_and_one_additional_failure_rejects(runner):
    rows = cohort()
    # Two of 32 hit-1 cases fail: 0.8 * (30/32) + 0.2 = 0.95.
    fail(rows, "space_full", {610001, 610002})
    condition = conditions(score(runner, rows, weights={1: 0.8, 2: 0.1, 3: 0.1}))["full_success_at_least_95pct"]
    assert condition["value"] == 0.95 and condition["passes"] is True
    fail(rows, "space_full", {610003})
    result = score(runner, rows, weights={1: 0.8, 2: 0.1, 3: 0.1})
    assert conditions(result)["full_success_at_least_95pct"]["passes"] is False
    assert result["learned_pilot_opportunity"] is False


def test_relative_gain_uses_control_denominator_and_ten_percent_is_inclusive(runner):
    rows = cohort()
    result = score(runner, rows)
    criterion = conditions(result)["gain_vs_recent32_at_least_10pct"]
    assert criterion["value"] == 0.1 and criterion["passes"] is True
    set_time(rows, "space_full", 19)
    result = score(runner, rows)
    assert conditions(result)["gain_vs_recent32_at_least_10pct"]["value"] == 0.05
    assert conditions(result)["gain_vs_recent32_at_least_10pct"]["passes"] is False
    assert result["learned_pilot_opportunity"] is False


def test_absolute_two_step_gate_cannot_be_replaced_by_relative_gain(runner):
    rows = cohort()
    assert conditions(score(runner, rows))["gain_vs_recent32_at_least_two_steps"]["passes"] is True
    set_time(rows, "space_full", 8)
    set_time(rows, "space_recent32", 9)
    result = score(runner, rows)
    assert conditions(result)["gain_vs_recent32_at_least_10pct"]["passes"] is True
    assert conditions(result)["gain_vs_recent32_at_least_two_steps"]["value"] == 1
    assert conditions(result)["gain_vs_recent32_at_least_two_steps"]["passes"] is False
    assert result["learned_pilot_opportunity"] is False


@pytest.mark.parametrize(("wins", "passes"), [(6, True), (5, False)])
def test_at_least_six_weighted_block_gains_are_strictly_positive(runner, wins, passes):
    rows = cohort()
    set_time(rows, "space_full", 10)
    for block in range(8):
        set_time(rows, "space_recent32", 20 if block < wins else 10, block=block)
    result = score(runner, rows)
    criterion = conditions(result)["positive_blocks_vs_recent32_at_least_six"]
    assert criterion["value"] == wins and criterion["passes"] is passes
    assert criterion["block_gains"] == [10] * wins + [0] * (8 - wins)
    assert result["learned_pilot_opportunity"] is passes


def test_block_win_sign_uses_the_three_stratum_mixture_not_raw_case_average(runner):
    rows = cohort()
    set_time(rows, "space_full", 50)
    for row in rows:
        if row["arm"] == "space_recent32":
            steps = (70 if row["initial_hit"] == 1 else 35) if row["block"] < 6 else 50
            row["steps"] = row["capped_time"] = steps
    # In six blocks the raw balanced mean gain is -10/3; declared weights
    # instead give .75*20 + .125*(-15) + .125*(-15) = 11.25.
    result = score(runner, rows, weights={1: 0.75, 2: 0.125, 3: 0.125})
    condition = conditions(result)["positive_blocks_vs_recent32_at_least_six"]
    assert condition["block_gains"] == [11.25] * 6 + [0] * 2
    assert condition["value"] == 6 and condition["passes"] is True
    other = score(runner, rows, weights={1: 0.25, 2: 0.375, 3: 0.375})
    assert conditions(other)["positive_blocks_vs_recent32_at_least_six"]["value"] == 0


@pytest.mark.parametrize("control", ["space_recent32", "space_recent8"])
def test_failure_regression_to_either_recent_control_blocks_admission(runner, control):
    rows = cohort()
    for arm in ("space_recent32", "space_recent8", "space_initial"):
        set_time(rows, arm, 300)
    fail(rows, "space_full", {610001})
    other = "space_recent8" if control == "space_recent32" else "space_recent32"
    fail(rows, other, {610001})
    result = score(runner, rows)
    failing = [c["name"] for c in result["criteria"] if not c["passes"]]
    assert failing == [f"no_failure_regression_vs_{control}"]
    assert result["learned_pilot_opportunity"] is False
    fail(rows, control, {610001})
    equal = conditions(score(runner, rows))[f"no_failure_regression_vs_{control}"]
    assert equal["value"] == 0 and equal["passes"] is True


@pytest.mark.parametrize("control", ["space_recent8", "space_initial"])
def test_equal_or_worse_time_to_shortest_controls_is_not_a_gain(runner, control):
    rows = cohort()
    set_time(rows, control, 18)
    result = score(runner, rows)
    assert [c["name"] for c in result["criteria"] if not c["passes"]] == [f"strict_time_gain_vs_{control}"]
    assert conditions(result)[f"strict_time_gain_vs_{control}"]["value"] == 0
    assert result["learned_pilot_opportunity"] is False
    set_time(rows, control, 17)
    assert conditions(score(runner, rows))[f"strict_time_gain_vs_{control}"]["passes"] is False


def test_infotaxis_outcomes_are_descriptive_and_cannot_rescue_space_failure(runner):
    rows = cohort()
    set_time(rows, "space_full", 20)
    for arm in ARMS:
        if arm.startswith("info_"):
            set_time(rows, arm, 1 if arm == "info_full" else 2188)
    result = score(runner, rows)
    assert result["means"]["info_full"]["capped_time"] == 1
    assert result["learned_pilot_opportunity"] is False
    assert len(result["criteria"]) == 10


@pytest.mark.parametrize("corruption", ["missing", "extra", "duplicate", "wrong_seed", "wrong_arm", "wrong_block", "wrong_hit"])
def test_exact_case_ids_and_strata_are_required(runner, corruption):
    rows = cohort()
    if corruption == "missing":
        rows.pop()
    elif corruption == "extra":
        rows.append(copy.deepcopy(rows[0]))
    elif corruption == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    else:
        key, value = {"wrong_seed": ("seed", 610097), "wrong_arm": ("arm", "space_latest"),
                      "wrong_block": ("block", 1), "wrong_hit": ("initial_hit", 2)}[corruption]
        rows[0][key] = value
    with pytest.raises(AssertionError):
        score(runner, rows)


@pytest.mark.parametrize("weights", [
    {}, {1: 1.0}, {1: 0.5, 2: 0.25, 3: 0.25, 4: 0.0}, {1: 0.8, 2: 0.1},
    {1: 0.5, 2: 0.2, 3: 0.2}, {1: 0.8, 2: 0.2, 3: 0.0},
    {1: -0.2, 2: 1.0, 3: 0.2}, {1: math.nan, 2: 0.25, 3: 0.25},
    {1: 0.5, 2: 0.25, 3: math.inf},
])
def test_invalid_or_nonfinite_mixture_weights_are_rejected(runner, weights):
    with pytest.raises(AssertionError, match="mixture"):
        score(runner, cohort(), weights=weights)


@pytest.mark.parametrize("update", [
    {"steps": 0, "capped_time": 0}, {"steps": 2189, "capped_time": 2189},
    {"steps": True, "capped_time": True}, {"steps": 18.0}, {"capped_time": 17},
    {"found": 1}, {"found": False},
    {"stuck_steps": -1}, {"stuck_steps": 19}, {"stuck_steps": True},
    {"stuck_steps": math.nan}, {"stuck_steps": 1.5},
])
def test_invalid_horizon_termination_and_stuck_counts_are_rejected(runner, update):
    rows = cohort()
    rows[0].update(update)
    with pytest.raises(AssertionError):
        score(runner, rows)


@pytest.mark.parametrize("field", ["controller_seconds", "initialization_seconds", "filter_seconds", "planning_seconds", "environment_seconds"])
@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_each_cost_component_must_be_finite_and_nonnegative(runner, field, value):
    rows = cohort()
    rows[0][field] = value
    with pytest.raises(AssertionError, match="time"):
        score(runner, rows)


def test_public_inference_cost_includes_initialization_filter_and_planning(runner):
    rows = cohort()
    rows[0]["controller_seconds"] = rows[0]["filter_seconds"] + rows[0]["planning_seconds"]
    with pytest.raises(AssertionError, match="timing sum"):
        score(runner, rows)


def test_structural_qualification_is_required_explicitly(runner):
    with pytest.raises(TypeError, match="qualification"):
        runner.summarize(cohort(), {1: 0.5, 2: 0.25, 3: 0.25})


def test_exact_qualification_budget_is_allowed(runner):
    receipt = qualification()
    receipt["native_steps"] = 2000
    result = score(runner, cohort(), qualified=receipt)
    assert conditions(result)["structural_qualification"]["passes"] is True


@pytest.mark.parametrize("update", [
    {"status": "failed"}, {"status": "started"}, {"checks": []},
    {"checks": [{"name": "broken", "passes": False}]},
    {"native_steps": 0}, {"native_steps": 2001}, {"native_steps": True},
    {"native_steps": 100.0},
])
def test_failed_or_incomplete_qualification_never_admits_a_complete_cohort(runner, update):
    receipt = qualification()
    receipt.update(update)
    result = score(runner, cohort(), qualified=receipt)
    assert conditions(result)["structural_qualification"]["passes"] is False
    assert result["learned_pilot_opportunity"] is False
    assert all(c["passes"] for c in result["criteria"] if c["name"] != "structural_qualification")
