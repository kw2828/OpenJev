"""Synthetic histories only: never import or execute the benchmark environment."""
from __future__ import annotations

import builtins
import copy
import importlib.util
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from openjev.research.rocksample_public_belief import PublicRockBelief

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnose_rocksample_public_memory.py"


def load_diagnostic():
    spec = importlib.util.spec_from_file_location("synthetic_rocksample_diagnostic", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def diagnostic():
    return load_diagnostic()


def public_transition(action=0, sign=1, coordinate=(0, 0)):
    """Stationary checks/sampling or a north move clipped at the northern edge."""
    observation = np.zeros(33)
    observation[coordinate[0]] = 1
    observation[11 + coordinate[1]] = 1
    if action >= 5:
        observation[22 + action - 5] = sign
    return np.array(coordinate), action, observation, False


def replay(history):
    belief = PublicRockBelief()
    for transition in history:
        belief.update(*transition)
    return belief


def test_import_is_independent_of_benchmark_and_neural_libraries(monkeypatch):
    original_import = builtins.__import__
    blocked = {"jax", "jaxlib", "gymnax", "pobax", "brax", "navix", "torch", "transformers"}

    def public_only_import(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise AssertionError(f"unexpected benchmark/model import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", public_only_import)
    assert load_diagnostic().ARMS == ("prior", "full", "recent32", "recent128", "latest_check")


def test_forecast_has_only_prefix_inputs_and_does_not_assimilate_or_mutate_them(diagnostic):
    assert list(inspect.signature(diagnostic.forecast).parameters) == ["history", "full", "coordinate", "rock"]
    history = [public_transition(5, 1), public_transition(0), public_transition(6, -1)]
    copied = copy.deepcopy(history)
    full = replay(history)
    saved = full._log_belief.copy()
    predictions, metadata = diagnostic.forecast(history, full, (0, 0), 0)
    assert predictions["full"] > 0.5
    assert metadata == {"age": "le32", "prior_reading_age": 3, "prior_checks": 1,
                        "samples_since_last_check": 0}
    positive = diagnostic.losses(predictions, True)
    negative = diagnostic.losses(predictions, False)
    assert positive["full"]["nll"] < negative["full"]["nll"]
    assert diagnostic.forecast(history, full, (0, 0), 0) == (predictions, metadata)
    np.testing.assert_array_equal(full._log_belief, saved)
    for before, after in zip(copied, history, strict=True):
        for left, right in zip(before, after, strict=True):
            np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("length,age,include32,include128", [(1, "le32", True, True),
                                                           (32, "le32", True, True),
                                                           (33, "33to128", False, True),
                                                           (128, "33to128", False, True),
                                                           (129, "gt128", False, False)])
def test_recent_history_includes_exact_boundary_and_excludes_one_older(diagnostic, length, age,
                                                                     include32, include128):
    history = [public_transition(5, 1)] + [public_transition() for _ in range(length - 1)]
    predictions, metadata = diagnostic.forecast(history, replay(history), (0, 0), 0)
    assert predictions["full"] > 0.5
    for arm, included in (("recent32", include32), ("recent128", include128)):
        assert predictions[arm] == pytest.approx(predictions["full"] if included else 0.5)
    assert predictions["latest_check"] == pytest.approx(predictions["full"])
    assert metadata == {"age": age, "prior_reading_age": length, "prior_checks": 1,
                        "samples_since_last_check": 0}


def test_latest_check_drops_older_checks_but_keeps_subsequent_sampling(diagnostic):
    history = [public_transition(5, -1), public_transition(5, -1), public_transition(5, 1),
               public_transition(4), public_transition(6, 1), public_transition()]
    predictions, metadata = diagnostic.forecast(history, replay(history), (0, 0), 0)
    expected = replay(history[2:]).probabilities((0, 0))[0]
    without_sample = replay([history[2], history[4], history[5]]).probabilities((0, 0))[0]
    assert predictions["latest_check"] == pytest.approx(expected)
    assert expected < without_sample
    assert metadata["samples_since_last_check"] == 1
    assert metadata["prior_checks"] == 3 and metadata["prior_reading_age"] == 4
    changed_old_checks = [public_transition(5, 1), public_transition(5, 1), *history[2:]]
    changed, _ = diagnostic.forecast(changed_old_checks, replay(changed_old_checks), (0, 0), 0)
    assert changed["latest_check"] == predictions["latest_check"]
    assert changed["full"] != predictions["full"]


def test_never_checked_control_still_retains_public_sample_history(diagnostic):
    history = [public_transition(6, 1), public_transition(4), public_transition()]
    predictions, metadata = diagnostic.forecast(history, replay(history), (0, 0), 0)
    assert metadata == {"age": "never", "prior_reading_age": None, "prior_checks": 0,
                        "samples_since_last_check": 1}
    assert predictions["latest_check"] == predictions["full"] < 0.5
    assert predictions["prior"] == 0.5
    empty, empty_metadata = diagnostic.forecast([], PublicRockBelief(), (0, 0), 0)
    assert all(probability == 0.5 for probability in empty.values())
    assert empty_metadata["age"] == "never"


def score_record(diagnostic, seed, full=0.8, control=1.0, age="never"):
    return {"map_seed": seed, "age": age,
            "losses": {arm: {"nll": full if arm == "full" else control,
                             "brier": (full if arm == "full" else control) / 10}
                       for arm in diagnostic.ARMS}}


def admitted_records(diagnostic):
    return [score_record(diagnostic, seed, age="gt128" if index < 4 else "never")
            for index, seed in enumerate(diagnostic.MAP_SEEDS)
            for _ in range(8 if index < 4 else 1)]


def test_summary_weights_maps_equally_despite_different_endpoint_counts(diagnostic):
    records = [score_record(diagnostic, seed, full=0.2 if index == 0 else 2,
                            control=1 if index == 0 else 3)
               for index, seed in enumerate(diagnostic.MAP_SEEDS)
               for _ in range(9 if index == 0 else 1)]
    summary = diagnostic.summarize(records)
    equal_map = (0.2 + 7 * 2) / 8
    pooled = (9 * 0.2 + 7 * 2) / 16
    assert equal_map != pooled
    assert summary["scores_equal_map_means"]["full"]["nll"] == pytest.approx(equal_map)
    assert summary["scores_equal_map_means"]["full"]["brier"] == pytest.approx(equal_map / 10)
    assert summary["pooled_age_breakdown"]["never"]["scores"]["full"]["nll"] == pytest.approx(pooled)
    assert [item["checks"] for item in summary["by_map"]] == [9, 1, 1, 1, 1, 1, 1, 1]
    assert summary["check_endpoints"] == 16
    assert summary["pooled_age_breakdown"]["gt128"] == {"count": 0, "scores": None}
    assert summary["delayed_equal_map_means"] is None
    assert summary["delayed_contributing_maps"] == 0


def test_delayed_summary_averages_nonempty_maps_equally_and_identifies_support(diagnostic):
    records = [score_record(diagnostic, seed) for seed in diagnostic.MAP_SEEDS]
    records.extend(score_record(diagnostic, diagnostic.MAP_SEEDS[0], full=0.2, age="gt128")
                   for _ in range(9))
    records.append(score_record(diagnostic, diagnostic.MAP_SEEDS[1], full=2, age="gt128"))
    summary = diagnostic.summarize(records)
    assert summary["delayed_contributing_maps"] == 2
    assert summary["delayed_equal_map_means"]["full"]["nll"] == pytest.approx((0.2 + 2) / 2)
    assert summary["delayed_equal_map_means"]["full"]["brier"] == pytest.approx((0.02 + 0.2) / 2)
    assert summary["pooled_age_breakdown"]["gt128"]["count"] == 10
    assert summary["pooled_age_breakdown"]["gt128"]["scores"]["full"]["nll"] == pytest.approx(
        (9 * 0.2 + 2) / 10)


def test_all_six_continuation_criteria_are_reported_and_jointly_required(diagnostic):
    summary = diagnostic.summarize(admitted_records(diagnostic))
    assert summary["memory_pilot_admitted"] is True
    assert [criterion["name"] for criterion in summary["criteria"]] == [
        "relative_nll_gain_vs_recent128", "positive_maps_vs_recent128",
        "relative_nll_gain_vs_latest_check", "positive_maps_vs_latest_check",
        "delayed_endpoints", "delayed_maps"]
    assert all(criterion["passes"] for criterion in summary["criteria"])
    assert summary["criteria"][-2]["value"] == 32
    assert summary["criteria"][-1]["value"] == 4


@pytest.mark.parametrize("failed_criterion", ["relative_nll_gain_vs_recent128", "positive_maps_vs_recent128",
                                              "relative_nll_gain_vs_latest_check", "positive_maps_vs_latest_check",
                                              "delayed_endpoints", "delayed_maps"])
def test_each_individual_gate_failure_blocks_admission_with_other_five_passing(diagnostic, failed_criterion):
    records = admitted_records(diagnostic)
    if failed_criterion.startswith("relative"):
        control = failed_criterion.split("_vs_")[1]
        for row in records:
            row["losses"][control]["nll"] = 0.8 / 0.91  # 9% gain, with positive gains on all maps
    elif failed_criterion.startswith("positive"):
        control = failed_criterion.split("_vs_")[1]
        for row in records:
            if row["map_seed"] in diagnostic.MAP_SEEDS[:3]:
                row["losses"][control]["nll"] = 0.8  # ties are not positive gains
    elif failed_criterion == "delayed_endpoints":
        records.pop(0)  # 31 delayed checks still span all four maps
    else:
        for row in records:
            if row["map_seed"] == diagnostic.MAP_SEEDS[3]:
                row["age"] = "le32"
        records.extend(copy.deepcopy(records[:8]))  # 32 delayed checks now span only three maps
    summary = diagnostic.summarize(records)
    assert summary["memory_pilot_admitted"] is False
    assert [criterion["name"] for criterion in summary["criteria"] if not criterion["passes"]] == [
        failed_criterion]


def test_ten_percent_gain_includes_boundary_without_rounding_reported_percentage(diagnostic):
    records = admitted_records(diagnostic)
    for row in records:
        row["losses"]["full"]["nll"] = 0.9
    summary = diagnostic.summarize(records)
    relative = [criterion for criterion in summary["criteria"] if criterion["name"].startswith("relative")]
    assert all(criterion["passes"] for criterion in relative)
    assert all(criterion["value"] == 1 - 0.9 for criterion in relative)
    for row in records:
        row["losses"]["full"]["nll"] = float(np.nextafter(0.9, math.inf))
    summary = diagnostic.summarize(records)
    assert summary["scores_equal_map_means"]["full"]["nll"] > 0.9
    assert not any(criterion["passes"] for criterion in summary["criteria"]
                   if criterion["name"].startswith("relative"))


def test_six_positive_maps_includes_boundary_and_preserves_paired_map_gains(diagnostic):
    records = admitted_records(diagnostic)
    for row in records:
        if row["map_seed"] in diagnostic.MAP_SEEDS[:2]:
            for arm in ("recent128", "latest_check"):
                row["losses"][arm]["nll"] = 0.8
    summary = diagnostic.summarize(records)
    assert summary["memory_pilot_admitted"] is True
    for criterion in summary["criteria"]:
        if criterion["name"].startswith("positive"):
            assert criterion["value"] == 6 and criterion["passes"]
            np.testing.assert_allclose(criterion["map_gains"], [0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])


def test_every_fixed_map_must_have_scored_endpoints(diagnostic):
    records = admitted_records(diagnostic)
    with pytest.raises(ValueError, match="each fixed map"):
        diagnostic.summarize([row for row in records if row["map_seed"] != diagnostic.MAP_SEEDS[-1]])


def test_nll_and_binary_brier_use_true_outcome_without_floor(diagnostic):
    for positive, target in ((True, 1), (False, 0)):
        actual = diagnostic.losses({"synthetic": 0.25}, positive)["synthetic"]
        assert actual["nll"] == pytest.approx(-math.log(0.25 if positive else 0.75))
        assert actual["brier"] == (0.25 - target) ** 2
    assert diagnostic.losses({"certain": 1}, True)["certain"] == {"nll": 0, "brier": 0}
    assert diagnostic.losses({"certain": 0}, False)["certain"] == {"nll": 0, "brier": 0}
    with pytest.raises(ValueError, match="impossible event"):
        diagnostic.losses({"certain": 1}, False)
    with pytest.raises(ValueError, match="impossible event"):
        diagnostic.losses({"certain": 0}, True)


@pytest.mark.parametrize("probability", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_invalid_probabilities_are_rejected(diagnostic, probability):
    with pytest.raises(ValueError, match="invalid probability"):
        diagnostic.losses({"invalid": probability}, True)
