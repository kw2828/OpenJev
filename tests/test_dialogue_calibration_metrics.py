"""Synthetic calibration and complete-factorial arithmetic, with no task files."""
from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from openjev.research import dialogue_calibration_metrics as metrics
from openjev.research import dialogue_observation_metrics as original
from openjev.research.dialogue_temperature import output_logs


def row(index, *, unseen=True, label=2, bin_name="first_assignment", values=("True", "False"), calibration=False):
    service = "unseen" if unseen else "seen"
    slot = f"slot-{len(values)}"
    ids = [original.NONE, original.DONTCARE, *("value:" + value for value in values)]
    result = {"row_index": index, "split": "train" if calibration else "dev",
              "dialogue_id": ("calibration" if calibration else "dev") + f"-{index}",
              "source_row_index": 0, "time": 0, "turn_index": 0,
              "query_index": 10 * int(unseen) + len(values), "query_position": 0,
              "query_id": json.dumps([service, slot]), "service": service, "slot": slot,
              "candidate_ids": ids, "candidate_values": [None, None, *values],
              "label_index": label, "label_id": ids[label], "bin": bin_name,
              "stratum": bin_name if bin_name in original.STRATA[:2] else "changed",
              "dontcare": label == 1, "unseen": unseen}
    if calibration:
        result.update(source_split="train", analysis_role="calibration")
    return result


def logs(rows, probabilities):
    result = np.full((len(rows), original.MAX_CANDIDATES), -np.inf, dtype=np.float32)
    for i, (record, values) in enumerate(zip(rows, probabilities, strict=True)):
        assert len(values) == len(record["candidate_ids"])
        result[i, :len(values)] = np.log(np.asarray(values, dtype=np.float64)).astype(np.float32)
    return result


def fit_fixture():
    calibration_rows = [row(i, unseen=False, label=label, values=("red",), calibration=True,
                            bin_name="unmentioned_retention" if label == 0 else "first_assignment")
                        for i, label in enumerate((0, 0, 0, 2))]
    calibration_logs = logs(calibration_rows, [[.8, .1, .1]] * 4)
    dev_rows = [row(i, unseen=i >= 3, bin_name=original.STRATA[i % 3] if i % 3 < 2 else "revision",
                    label=0 if i % 3 == 0 else 2) for i in range(6)]
    dev_logs = logs(dev_rows, [[.1, .1, .7, .1]] * 6)
    return (dev_logs, np.arange(6, dtype=np.int64), dev_rows,
            calibration_logs, np.arange(4, dtype=np.int64), calibration_rows)


def test_closed_form_three_candidate_fit_keeps_raw_scoring_and_source_roles_exact():
    fixture = fit_fixture()
    before = copy.deepcopy(fixture)
    result = metrics.evaluate_fit(*fixture)
    dev_logs, dev_indices, dev_rows, calibration_logs, _, calibration_rows = fixture
    expected_beta = math.log(6) / (float(calibration_logs[0, 0]) - float(calibration_logs[0, 1]))
    assert result["temperature"]["beta"] == pytest.approx(expected_beta, abs=1e-12)
    assert result["temperature"]["normalized_nll_after"] == pytest.approx(-.75 * math.log(.75) - .25 * math.log(.125))
    assert result["temperature"]["bisection_steps"] == 64
    assert result["raw"] == original.score_panels(dev_logs, dev_rows, row_indices=dev_indices)
    assert result["calibration_validation"]["rows"] == result["calibration_validation"]["dialogues"] == 4
    assert result["calibration_validation"]["source_split"] == "train"
    assert result["calibration_validation"]["analysis_role"] == "calibration"
    assert all(r["split"] == r["source_split"] == "train" for r in calibration_rows)
    for old, new in zip(before, fixture, strict=True):
        if isinstance(old, np.ndarray):
            np.testing.assert_array_equal(new, old)
        else:
            assert new == old
    for name in ("normalized", "calibrated"):
        assert result[name]["validation"]["dtype"] == "float64"
        assert result[name]["validation"]["choices_unchanged"]
        assert result[name]["validation"]["top_tie_masks_unchanged"]


def test_fitting_is_independent_of_dev_labels_and_dev_probabilities():
    fixture = fit_fixture()
    before = metrics.evaluate_fit(*fixture)["temperature"]
    changed = copy.deepcopy(fixture)
    changed[0][:, :4] = np.log(np.array([.6, .1, .2, .1])).astype(np.float32)
    for record in changed[2]:
        record["label_index"] = 1
        record["label_id"] = original.DONTCARE
        record["dontcare"] = True
    assert metrics.evaluate_fit(*changed)["temperature"] == before


def test_float64_scores_do_not_round_back_to_saved_float32():
    dev_logs, indices, rows, *_ = fit_fixture()
    transformed = output_logs(dev_logs, .71312341)
    scored = metrics.score_transformed(transformed, rows, row_indices=indices, raw_logs=dev_logs)
    labels = np.asarray([record["label_index"] for record in rows])
    expected = math.fsum(-float(transformed[i, label]) for i, label in enumerate(labels)) / len(rows)
    rounded = math.fsum(-float(transformed.astype(np.float32)[i, label]) for i, label in enumerate(labels)) / len(rows)
    assert scored["panels"]["all"]["micro"]["nll"] == expected and expected != rounded
    residual = np.exp(transformed)
    residual[np.arange(len(rows)), labels] -= 1
    assert scored["panels"]["all"]["micro"]["brier"] == math.fsum(np.square(residual).sum(1)) / len(rows)
    assert scored["panels"]["unseen"]["bins"]["clear"]["count"] == 0
    assert scored["panels"]["unseen"]["bins"]["clear"]["nll"] is None
    assert scored["services"]["unseen"]["metrics"] == scored["panels"]["unseen"]


def test_macro_weighting_candidate_types_and_variable_support_reuse_original_definitions():
    rows = [row(i, label=0, bin_name="unmentioned_retention") for i in range(9)]
    rows += [row(9, label=3, bin_name="assigned_retention"), row(10, label=2),
             row(11, unseen=False, label=2, values=("True", "False", "None"))]
    saved = logs(rows, [[.7, .1, .1, .1]] * 11 + [[.1, .1, .6, .1, .1]])
    scored = metrics.score_transformed(output_logs(saved, .7), rows,
                                       row_indices=np.arange(len(rows), dtype=np.int64), raw_logs=saved)
    raw = original.score_panels(saved, rows)
    assert metrics._decisions(scored) == metrics._decisions(raw)
    unseen = scored["panels"]["unseen"]
    assert unseen["micro"]["accuracy"] == 9 / 11 and unseen["macro_three"]["accuracy"] == 1 / 3
    assert unseen["types"]["false"]["count"] == 1
    assert scored["panels"]["all"]["types"]["other"]["count"] == 1
    assert scored["validation"]["supported_candidate_positions"] == 11 * 4 + 5


def test_uniform_fit_reports_normalization_drift_and_preserves_entire_tied_top_set():
    fixture = list(fit_fixture())
    fixture[3][:, :3] = np.float32(-math.log(3)) + np.float32(1e-7)
    fixture[0][:, :4] = np.float32(-math.log(4)) + np.float32(1e-7)
    result = metrics.evaluate_fit(*fixture)
    assert result["temperature"]["beta"] == 1.0
    assert result["temperature"]["location"] == "unidentified_uniform_identity"
    assert result["temperature"]["normalization_only_nll_drift"] != 0
    assert result["normalized"] == result["calibrated"]
    assert result["raw"]["panels"]["all"]["micro"]["nll"] != result["normalized"]["panels"]["all"]["micro"]["nll"]
    assert result["calibrated"]["validation"]["exact_top1_tie_rows"] == len(fixture[2])


def test_direct_finite_tail_nll_survives_exponent_underflow():
    rows = [row(0)]
    raw = np.full((1, 12), -np.inf, dtype=np.float32)
    raw[0, :4] = [0, -1000, -1000, -1000]
    scored = metrics.score_transformed(output_logs(raw, 2), rows, row_indices=np.array([0], np.int64), raw_logs=raw)
    assert scored["panels"]["all"]["micro"]["nll"] == 2000
    assert scored["panels"]["all"]["micro"]["brier"] == 2
    assert scored["validation"]["float64_exponent_underflow_positions"] == 3


@pytest.mark.parametrize("defect", ["dtype", "shape", "supported_neginf", "nan", "positive_inf", "padding", "mass", "indices", "argmax"])
def test_transformed_outputs_fail_closed(defect):
    raw, indices, rows, *_ = fit_fixture()
    transformed = output_logs(raw, 1.0)
    if defect == "dtype":
        transformed = transformed.astype(np.float32)
    elif defect == "shape":
        transformed = transformed[:, :4]
    elif defect == "supported_neginf":
        transformed[0, 0] = -np.inf
    elif defect == "nan":
        transformed[0, 0] = np.nan
    elif defect == "positive_inf":
        transformed[0, 0] = np.inf
    elif defect == "padding":
        transformed[0, 4] = -1000
    elif defect == "mass":
        transformed[:, :4] -= 1
    elif defect == "indices":
        indices = indices[::-1]
    else:
        transformed[0, [0, 2]] = transformed[0, [2, 0]]
    with pytest.raises(ValueError):
        metrics.score_transformed(transformed, rows, row_indices=indices, raw_logs=raw)


@pytest.mark.parametrize("change", ["remove_tie", "introduce_tie"])
def test_full_top_tie_mask_guard_is_stronger_than_first_argmax(change):
    rows = [row(0, label=0)]
    raw = logs(rows, [[.4, .4, .1, .1]] if change == "remove_tie" else [[.5, .3, .1, .1]])
    transformed = np.full((1, 12), -np.inf, np.float64)
    transformed[0, :4] = np.log([.5, .3, .1, .1] if change == "remove_tie" else [.4, .4, .1, .1])
    assert transformed.argmax(1).tolist() == raw.argmax(1).tolist() == [0]
    with pytest.raises(ValueError, match="Complete top-tie masks"):
        metrics.score_transformed(transformed, rows, row_indices=np.array([0], np.int64), raw_logs=raw)


@pytest.mark.parametrize("defect", ["split", "source_split", "role", "unseen", "dtype", "indices", "row_order",
                                    "label_id", "candidate_value", "supported_neginf", "padding", "mass",
                                    "duplicate_endpoint", "query_identity", "query_schema"])
def test_calibration_requires_actual_train_role_and_canonical_support(defect):
    fixture = list(fit_fixture())
    saved, rows = fixture[3], fixture[5]
    if defect == "split":
        rows[0]["split"] = "dev"
    elif defect == "source_split":
        rows[0]["source_split"] = "dev"
    elif defect == "role":
        rows[0]["analysis_role"] = "evaluation"
    elif defect == "unseen":
        rows[0]["unseen"] = True
    elif defect == "dtype":
        fixture[3] = saved.astype(np.float64)
    elif defect == "indices":
        fixture[4] = fixture[4][::-1]
    elif defect == "row_order":
        rows[0]["row_index"] = 1
    elif defect == "label_id":
        rows[0]["label_id"] = "value:missing"
    elif defect == "candidate_value":
        rows[0]["candidate_values"][2] = "blue"
    elif defect == "supported_neginf":
        saved[0, 0] = -np.inf
    elif defect == "padding":
        saved[0, 3] = -1000
    elif defect == "mass":
        saved[:, :3] -= 1
    elif defect == "duplicate_endpoint":
        rows[1]["dialogue_id"] = rows[0]["dialogue_id"]
    elif defect == "query_identity":
        rows[0]["query_id"] = json.dumps(["another", "slot"])
    else:
        rows[1]["candidate_values"][2] = "blue"
        rows[1]["candidate_ids"][2] = "value:blue"
    with pytest.raises(ValueError):
        metrics.evaluate_fit(*fixture)


def gate_fixture():
    def panel(count, hits, nll, brier):
        cell = {"count": count, "correct": hits, "incorrect": count - hits,
                "accuracy": hits / count, "error": (count - hits) / count, "nll": nll, "brier": brier}
        return {"micro": {**cell, "count": count * 3, "correct": hits * 3, "incorrect": (count - hits) * 3},
                "strata": {name: dict(cell) for name in original.STRATA},
                "macro_three": {"accuracy": hits / count, "nll": nll, "brier": brier}}

    routes = {route: {} for route in metrics.ROUTES}
    for arm in original.ARMS:
        for seed in original.SEEDS:
            hits = 102 if arm == "trainable_numbers" else 100
            raw_nll = .8 if arm == "trainable_numbers" else .7
            for route in metrics.ROUTES:
                nll = raw_nll if route in ("raw", "normalized") else .6 if arm == "trainable_numbers" else .65
                brier = .4 if route in ("raw", "normalized") else .3 if arm == "trainable_numbers" else .35
                routes[route][f"{arm}-{seed}"] = {
                    "panels": {"seen": panel(200, 100, .5, .4), "unseen": panel(200, hits, nll, brier),
                               "all": panel(400, 100 + hits, (.5 + nll) / 2, (.4 + brier) / 2)},
                    "validation": {"rows": 1200}}
    return routes


def gate(routes):
    return metrics.calibration_criteria(*(routes[name] for name in metrics.ROUTES))


def check(result, name):
    return next(value for value in result["checks"] if value["name"] == name)


def test_new_eleven_condition_pass_retains_original_six_of_seven_failure():
    routes = gate_fixture()
    result = gate(routes)
    assert result["passed"] and result["checks_passed"] == result["checks_total"] == 11
    raw = result["original_conditions"]["raw"]
    assert raw["passed"] is False and raw["checks_passed"] == 6 and raw["checks_total"] == 7
    assert result["raw_result_revised"] is False and result["decision_conditions_unchanged"] is True
    for name in metrics.DECISION_CHECKS:
        assert check(raw, name) == check(result["original_conditions"]["normalized"], name) == check(result, name)


@pytest.mark.parametrize("arm,metric,strict", [("trainable_numbers", "nll", True), ("trainable_numbers", "brier", False),
                                             ("frozen_numbers", "nll", False), ("frozen_numbers", "brier", False)])
def test_own_raw_guards_use_all_three_means_with_no_epsilon(arm, metric, strict):
    routes = gate_fixture()
    name = f"{arm}_unseen_micro_{metric}_" + ("improves_own_raw" if strict else "nonworse_own_raw")
    # Use a binary-exact center whose adjacent values stay distinct after the
    # declared math.fsum(values) / 3 arm-mean operation.
    before = .5
    for seed in original.SEEDS:
        for route in metrics.ROUTES:
            routes[route][f"{arm}-{seed}"]["panels"]["unseen"]["micro"][metric] = before
    equal = check(gate(routes), name)
    assert equal["passed"] is not strict and equal["mean"] == 0
    above, below = math.nextafter(before, math.inf), math.nextafter(before, -math.inf)
    assert original._mean([above] * 3) > original._mean([before] * 3) > original._mean([below] * 3)
    for seed in original.SEEDS:
        routes["calibrated"][f"{arm}-{seed}"]["panels"]["unseen"]["micro"][metric] = above
    worse = check(gate(routes), name)
    assert worse["passed"] is False and worse["mean"] > 0
    for seed in original.SEEDS:
        routes["calibrated"][f"{arm}-{seed}"]["panels"]["unseen"]["micro"][metric] = below
    assert check(gate(routes), name)["passed"]


@pytest.mark.parametrize("metric,before,strict", [("nll", .8, True), ("brier", .4, False)])
def test_rounded_equal_arm_means_retain_positive_paired_differences(metric, before, strict):
    routes = gate_fixture()
    after = math.nextafter(before, math.inf)
    for seed in original.SEEDS:
        routes["raw"][f"trainable_numbers-{seed}"]["panels"]["unseen"]["micro"][metric] = before
        routes["calibrated"][f"trainable_numbers-{seed}"]["panels"]["unseen"]["micro"][metric] = after
    expected_before, expected_after = original._mean([before] * 3), original._mean([after] * 3)
    assert after > before and expected_after == expected_before
    name = f"trainable_numbers_unseen_micro_{metric}_" + ("improves_own_raw" if strict else "nonworse_own_raw")
    result = check(gate(routes), name)
    assert result["raw_mean"] == expected_before and result["calibrated_mean"] == expected_after
    assert result["mean"] == 0 and result["paired_seed_values"] == [after - before] * 3
    assert all(value > 0 for value in result["paired_seed_values"])
    expected_pass = expected_after < expected_before if strict else expected_after <= expected_before
    assert result["passed"] is expected_pass


def test_all_seed_proper_score_changes_include_regression_and_direct_mean_comparison():
    routes = gate_fixture()
    changes = [.1, -.2, -.2]
    for seed, delta in zip(original.SEEDS, changes, strict=True):
        routes["calibrated"][f"trainable_numbers-{seed}"]["panels"]["unseen"]["micro"]["nll"] = .8 + delta
    result = check(gate(routes), "trainable_numbers_unseen_micro_nll_improves_own_raw")
    assert result["passed"] and result["paired_seed_values"] == pytest.approx(changes)
    assert result["paired_seed_values"][0] > 0 and len(result["paired_seed_values"]) == 3
    assert result["calibrated_mean"] == math.fsum(.8 + change for change in changes) / 3
    assert result["mean"] == result["calibrated_mean"] - result["raw_mean"]


@pytest.mark.parametrize("route", metrics.ROUTES)
def test_no_route_may_omit_a_fit(route):
    routes = gate_fixture()
    routes[route].pop("frozen_original-6901")
    with pytest.raises(ValueError, match="Exactly all four arms"):
        gate(routes)


@pytest.mark.parametrize("route", ["normalized", "calibrated"])
def test_gate_rejects_changed_decisions_even_on_nonprimary_arm(route):
    routes = gate_fixture()
    routes[route]["trainable_original-6901"]["panels"]["seen"]["micro"]["accuracy"] += .01
    with pytest.raises(ValueError, match="decision metrics"):
        gate(routes)


def test_missing_proper_score_is_unevaluable_instead_of_dropped_seed():
    routes = gate_fixture()
    routes["calibrated"]["frozen_numbers-6902"]["panels"]["unseen"]["micro"]["brier"] = None
    result = gate(routes)
    guard = check(result, "frozen_numbers_unseen_micro_brier_nonworse_own_raw")
    assert not result["passed"] and not guard["passed"] and guard["mean"] is None
    assert guard["paired_seed_values"] is None


def test_complete_summary_exposes_every_fit_group_route_and_seed_change():
    fixture = fit_fixture()
    results = {}
    for arm in original.ARMS:
        for seed in original.SEEDS:
            results[f"{arm}-{seed}"] = metrics.evaluate_fit(*fixture)
    summary = metrics.summarize(results)
    assert set(summary["fits"]) == original._fit_names()
    assert set(summary["factorial"]) == set(metrics.ROUTES)
    assert len(summary["paired_changes"]) == 3 and summary["continuation"]["checks_total"] == 11
    for comparison in summary["paired_changes"].values():
        assert set(comparison["arms"]) == set(original.ARMS)
        for arm in comparison["arms"].values():
            assert set(arm["seeds"]) == {str(seed) for seed in original.SEEDS}
            assert arm["mean"]["panels"]["all"]["micro"]["accuracy"] == 0
            assert arm["mean"]["panels"]["seen"]["bins"]["clear"]["nll"] is None
            assert "types" in arm["mean"]["panels"]["unseen"] and "services" in arm["mean"]
    assert summary["continuation"]["original_conditions"]["raw"]["checks_passed"] == 5
    results.pop("trainable_numbers-6903")
    with pytest.raises(ValueError, match="Exactly all four arms"):
        metrics.summarize(results)
