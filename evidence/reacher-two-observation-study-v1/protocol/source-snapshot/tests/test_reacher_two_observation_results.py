"""Hand-computable synthetic arrays only; no RNG, production artifacts or models."""

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from openjev.research import reacher_two_observation_protocol as protocol
from openjev.research import reacher_two_observation_results as results


def fixture():
    plan = protocol.settings()
    case_ids = [f"synthetic-case-{index:02d}" for index in range(64)]
    rows = []
    for row in protocol.execution_order(plan):
        cost = 80. if row.get("arm") == "residual_gru" else 100.
        if row.get("reference") in ("particle", "public_kinematic"):
            cost = 175.  # These remain descriptive and must not silently enter the gate.
        if row.get("reference") in ("zero", "uniform"):
            cost = 200.
        rewards = np.zeros((64, 50), np.float64)
        rewards[:, 0] = -cost
        rows.append({"panel": row["panel"], "label": row["label"],
                     "case_ids": case_ids.copy(), "native_rewards": rewards})
    return plan, rows


def row(rows, panel, label):
    return next(item for item in rows if (item["panel"], item["label"]) == (panel, label))


def set_cost(rows, panel, label, cost):
    target = row(rows, panel, label)["native_rewards"]
    target.fill(0)
    target[:, 0] = -cost


def named_check(output, name):
    return next(item for item in output["continuation_gate"]["checks"] if item["name"] == name)


def test_all25_pass_with_all42_rows_all_cases_and_all_references_preserved():
    plan, rows = fixture()
    result = results.evaluate_rows(plan, rows)
    assert result["continuation_gate"]["passed"] is True
    assert result["continuation_gate"]["checks_passed"] == result["continuation_gate"]["total_checks"] == 25
    assert result["control_rows"] == 42 and result["learned_rows"] == 27 and result["reference_rows"] == 15
    for panel in protocol.PANELS:
        assert len(result["controls"][panel]) == 14
        assert result["families"][panel]["residual_gru"]["fit_mean_costs"] == [80.] * 3
        assert result["families"][panel]["residual_gru"]["mean_cost"] == 80.
        for reference in protocol.REFERENCES:
            saved = result["controls"][panel][reference]
            assert len(saved["episode_costs"]) == len(saved["case_ids"]) == 64
            assert saved["case_ids"] == result["case_ids"]
        assert result["controls"][panel]["public_kinematic"]["mean_cost"] == 175.
    assert result["status"] == "arithmetic_completed"
    assert "no authenticity, freeze or effectiveness certification" in result["scope"]
    assert result["new_rng_draws"] == result["new_model_calls"] == result["new_native_calls"] == 0


def test_bad_pair_cannot_be_rescued_by_good_family_mean_or_secondary():
    plan, rows = fixture()
    set_cost(rows, "ordinary", "residual_gru-pair0", 110.)
    set_cost(rows, "ordinary", "residual_gru-pair1", 50.)
    set_cost(rows, "ordinary", "residual_gru-pair2", 50.)
    result = results.evaluate_rows(plan, rows)
    assert result["families"]["ordinary"]["residual_gru"]["mean_cost"] == 70.
    assert named_check(result, "gap_mean/ordinary/two_observation_gru")["passed"]
    assert named_check(result, "gap_mean/ordinary/cached_gru")["passed"]
    assert not named_check(result, "gap_pair/ordinary/two_observation_gru/pair0")["passed"]
    assert not named_check(result, "gap_pair/ordinary/cached_gru/pair0")["passed"]
    assert result["continuation_gate"]["checks_passed"] == 23
    assert result["continuation_gate"]["passed"] is False
    assert result["controls"]["ordinary"]["residual_gru-pair0"]["episode_costs"] == [110.] * 64


@pytest.mark.parametrize("panel,bound,check_prefix", [("ordinary", 97., "gap_mean/ordinary"),
                                                     ("shift", 97., "gap_mean/shift"),
                                                     ("full", 102., "full_mean")])
def test_exact3percent_and2percent_boundaries_are_inclusive_without_epsilon(panel, bound, check_prefix):
    plan, rows = fixture()
    for pair in protocol.PAIRS:
        set_cost(rows, panel, f"residual_gru-{pair}", bound)
    result = results.evaluate_rows(plan, rows)
    for comparator in ("two_observation_gru", "cached_gru"):
        check = named_check(result, f"{check_prefix}/{comparator}")
        assert check["left"] == check["right"] == bound and check["passed"]
    for pair in protocol.PAIRS:
        set_cost(rows, panel, f"residual_gru-{pair}", np.nextafter(bound, np.inf))
    result = results.evaluate_rows(plan, rows)
    for comparator in ("two_observation_gru", "cached_gru"):
        check = named_check(result, f"{check_prefix}/{comparator}")
        assert check["left"] > check["right"] and not check["passed"]


@pytest.mark.parametrize("label", ["known_state", "residual_gru-pair2"])
def test_one_competence_failure_survives_other_successes(label):
    plan, rows = fixture()
    if label.startswith("residual"):
        for pair in protocol.PAIRS:
            for comparator in ("two_observation_gru", "cached_gru"):
                set_cost(rows, "ordinary", f"{comparator}-{pair}", 300.)
    set_cost(rows, "ordinary", label, 181.)
    result = results.evaluate_rows(plan, rows)
    check = named_check(result, f"competence/ordinary/{label}")
    assert check["left"] == 181. and check["right"] == 180. and not check["passed"]
    assert result["continuation_gate"]["checks_passed"] == 24 and not result["continuation_gate"]["passed"]


def test_every_step_and_case_contributes_to_fit_mean_and_every_fit_to_family_mean():
    plan, rows = fixture()
    changed = row(rows, "full", "two_observation_gru-pair0")["native_rewards"]
    changed[0, 49] = -64.
    result = results.evaluate_rows(plan, rows)
    saved = result["controls"]["full"]["two_observation_gru-pair0"]
    assert saved["episode_costs"] == [164.] + [100.] * 63
    assert saved["mean_cost"] == 101.
    family = result["families"]["full"]["two_observation_gru"]
    assert family["fit_mean_costs"] == [101., 100., 100.]
    assert family["mean_cost"] == 301. / 3.


def test_row_order_can_change_but_case_order_cannot_and_output_does_not_alias():
    plan, rows = fixture()
    before = copy.deepcopy(rows)
    result = results.evaluate_rows(plan, list(reversed(rows)))
    normal = results.evaluate_rows(plan, rows)
    assert result == normal
    result["case_ids"][0] = "changed"
    result["controls"]["full"]["residual_gru-pair0"]["episode_costs"][0] = -1.
    result["criterion"]["mean_improvement"] = 0
    for original, current in zip(before, rows, strict=True):
        assert original["case_ids"] == current["case_ids"]
        np.testing.assert_array_equal(original["native_rewards"], current["native_rewards"])
    assert plan["criterion"]["mean_improvement"] == .03


@pytest.mark.parametrize("mutation", ["missing_row", "extra_row", "duplicate_row", "unknown_label", "unknown_panel",
    "missing_case", "duplicate_case", "reordered_case", "different_case", "empty_case", "bool_case", "future_step",
    "missing_step", "nan", "infinite", "complex", "bool_rewards", "object_rewards", "list_rewards",
    "negative_cost", "sum_overflow", "extra_field"])
def test_incomplete_mismatched_or_invalid_evidence_rejected(mutation):
    plan, rows = fixture()
    target = rows[-1]
    if mutation == "missing_row":
        rows.pop()
    elif mutation == "extra_row":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "duplicate_row":
        rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "unknown_label":
        target["label"] = "best_selected_fit"
    elif mutation == "unknown_panel":
        target["panel"] = "shift2"
    elif mutation == "missing_case":
        target["case_ids"].pop()
    elif mutation == "duplicate_case":
        target["case_ids"][1] = target["case_ids"][0]
    elif mutation == "reordered_case":
        target["case_ids"].reverse()
    elif mutation == "different_case":
        target["case_ids"][0] = "different-physical-case"
    elif mutation == "empty_case":
        target["case_ids"][0] = " "
    elif mutation == "bool_case":
        target["case_ids"][0] = True
    elif mutation == "future_step":
        target["native_rewards"] = np.zeros((64, 51))
    elif mutation == "missing_step":
        target["native_rewards"] = target["native_rewards"][:, :-1]
    elif mutation == "nan":
        target["native_rewards"][0, 25] = np.nan
    elif mutation == "infinite":
        target["native_rewards"][0, 25] = -np.inf
    elif mutation == "complex":
        target["native_rewards"] = target["native_rewards"].astype(np.complex128)
    elif mutation == "bool_rewards":
        target["native_rewards"] = target["native_rewards"].astype(bool)
    elif mutation == "object_rewards":
        target["native_rewards"] = target["native_rewards"].astype(object)
    elif mutation == "list_rewards":
        target["native_rewards"] = target["native_rewards"].tolist()
    elif mutation == "negative_cost":
        target["native_rewards"][0] = 1.
    elif mutation == "sum_overflow":
        target["native_rewards"][:] = -np.finfo(np.float64).max
    else:
        target["winner"] = True
    with pytest.raises(ValueError):
        results.evaluate_rows(plan, rows)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_finite_real_numeric_reward_dtypes_use_declared_float64_arithmetic(dtype):
    plan, rows = fixture()
    for item in rows:
        item["native_rewards"] = item["native_rewards"].astype(dtype)
    result = results.evaluate_rows(plan, rows)
    assert result["continuation_gate"]["checks_passed"] == 25
    assert result["controls"]["full"]["residual_gru-pair0"]["mean_cost"] == 80.


def test_prospective_criterion_cannot_be_weakened_by_caller():
    plan, rows = fixture()
    plan["criterion"]["mean_improvement"] = 0
    with pytest.raises(ValueError, match="setting changed"):
        results.evaluate_rows(plan, rows)


def test_static_dependency_boundary_has_no_rng_io_model_or_native_surface():
    source = Path(results.__file__).read_text()
    tree = ast.parse(source)
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module)
    assert set(modules) <= {"__future__", "copy", "math", "collections.abc", "numpy", "openjev.research"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in {"open", "eval", "exec", "__import__"} for node in ast.walk(tree))
    assert "np.random" not in source and "torch" not in source
