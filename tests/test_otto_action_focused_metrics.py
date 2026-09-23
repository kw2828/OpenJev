"""Fabricated action-focused gates, paired seeds and inherited metric domains."""
import copy
import itertools
import math

import numpy as np
import pytest

from openjev.research import otto_action_focused_metrics as M
from openjev.research import otto_prequery_metrics as prior_base
from openjev.research import otto_separate_prior_metrics as forecast_base
from openjev.research.otto_score_forecast_data import build_windows


def fixture(specifications):
    episodes, identities = [], []
    for regime, case, arm, length in specifications:
        episode_id = f"{regime}:{case}:{arm}"
        features = np.zeros((length, 31), np.float32)
        steps = np.arange(length)
        features[:, 15] = steps / 2188
        features[:, 16] = steps % 4 / 2188
        features[:, 17] = 1
        episodes.append({"id": episode_id, "regime": regime, "features": features,
                         "teacher_scores": np.zeros((length, 4), np.float32),
                         "legal": np.ones((length, 4), np.bool_)})
        identities.append({"episode_id": episode_id, "regime": regime, "case": case, "arm": arm})
    offsets = np.concatenate((np.zeros(1, np.int64), np.cumsum([len(e["features"]) for e in episodes], dtype=np.int64)))
    query = np.full((int(offsets[-1]), 4), np.nan, np.float32)
    mask, prior_mask = np.zeros(int(offsets[-1]), np.bool_), np.zeros(int(offsets[-1]), np.bool_)
    for low, high in itertools.pairwise(offsets):
        mask[low:high:4] = True
        prior_mask[low + 4:high:4] = True
        query[low + 4:high:4] = 0
    history = {"episode_ids": tuple(e["id"] for e in episodes), "episode_regimes": tuple(e["regime"] for e in episodes),
               "episode_offsets": offsets, "query_scores": query, "query_mask": mask}
    prior = np.full_like(query, np.nan)
    prior[prior_mask] = 0
    return history, prior, prior_mask, identities, episodes


def gate_fixture(active_cases=6):
    history, prior, _, ids, episodes = fixture([
        (regime, case, arm, 8 if case < active_cases else 1)
        for regime in M.REGIMES for case in range(6) for arm in M.ARMS])
    windows = build_windows(episodes)
    template = M.forecast_metrics(windows, windows["targets"].copy(), ids)
    calibration = M.prior_metrics(history, prior, ids)
    hold = copy.deepcopy(template)
    for regime in M.REGIMES:
        hold["postcorrection"]["by_regime"][regime][M.GAP] = 12.
    models = []
    for family in M.FAMILIES:
        for seed in M.SEEDS:
            report = copy.deepcopy(template)
            for scope in M.SCOPES:
                for regime in M.REGIMES:
                    group = report[scope]["by_regime"][regime]
                    group[M.AGREEMENT] = .5
                    group[M.GAP] = {"innovation_aux": 10., "innovation_spo": 7.2,
                                    "gru_aux": 10., "gru_spo": 8.}[family]
            models.append({"family": family, "seed": seed, "metrics": report,
                           "prior_metrics": copy.deepcopy(calibration)})
    return models, hold


def selected(rows, name):
    return next(row for row in rows if row["name"] == name)


def test_metric_functions_are_unchanged_and_preserve_near_ties_zero_support_and_case_denominators():
    assert M.forecast_metrics is forecast_base.forecast_metrics
    assert M.prior_metrics is prior_base.prior_metrics
    assert M.validate_prediction_support is prior_base.validate_prediction_support
    history, prior, mask, ids, episodes = fixture([
        ("lambda3", 0, "analytic", 8), ("lambda3", 0, "neural", 2), ("lambda3", 1, "analytic", 1)])
    for episode in episodes:
        episode["teacher_scores"][:] = [0., 5e-11, 999., 999.]
        episode["legal"][:, 2:] = False
    windows = build_windows(episodes)
    prediction = np.tile(np.array([2e-10, 0., -999., -999.], np.float32), (len(windows["targets"]), 4, 1))
    metrics = M.forecast_metrics(windows, prediction, ids)
    assert metrics["version"] == forecast_base.VERSION
    assert metrics["initial"]["overall"][M.AGREEMENT] == 2 / 3
    assert metrics["postcorrection"]["overall"][M.AGREEMENT] == 1 / 3
    assert metrics["initial"]["overall"]["episode_weighted_first_argmin_match"] == 0
    assert metrics["postcorrection"]["overall"]["supported_case_count"] == 1
    assert metrics["postcorrection"]["overall"]["declared_case_count"] == 2
    assert metrics["initial"]["overall"]["by_age"]["1"]["weight_mass"] == 2 / 3
    assert metrics["initial"]["overall"]["by_age"]["2"]["weight_mass"] == 1 / 3
    assert np.array_equal(M.validate_prediction_support(history, prior, mask), mask)
    calibration = M.prior_metrics(history, prior, ids)
    assert calibration["version"] == prior_base.VERSION
    assert calibration["overall"]["episodes"] == 3 and calibration["overall"]["supported_episodes"] == 1
    assert calibration["overall"]["weight_mass"] == 1 / 3


def test_exact_41_inventory_and_all_three_gate_memberships_have_no_omnibus_decision():
    models, hold = gate_fixture()
    rows = M.criteria(models, hold, technical_complete=True)
    assert tuple(row["name"] for row in rows) == M.condition_names() and len(rows) == 41
    assert all(row["passes"] for row in rows)
    gates = M.gate_decisions(rows)
    assert {gate: value["total"] for gate, value in gates.items()} == {
        "innovation_objective": 23, "gru_objective": 23, "architecture": 29}
    for failed in range(41):
        changed = copy.deepcopy(rows)
        changed[failed]["passes"] = False
        outcomes = M.gate_decisions(changed)
        for gate, value in outcomes.items():
            member = rows[failed]["name"] in gates[gate]["conditions"]
            assert value["passes"] is (not member)
            assert value["passed"] == value["total"] - int(member)
    pending = M.gate_decisions(M.criteria(models, hold, technical_complete=False))
    assert all(not value["passes"] for value in pending.values())


def test_exact_ten_percent_boundary_and_zero_reference_strict_guard():
    models, hold = gate_fixture()
    name = "objective.innovation.lambda3.postcorrection.gap"
    for model in models:
        if model["family"] == "innovation_spo":
            model["metrics"]["postcorrection"]["by_regime"]["lambda3"][M.GAP] = 9.
    row = selected(M.criteria(models, hold, technical_complete=True), name)
    assert row["threshold"] == 9. and row["strict_upper_bound"] == 10. and row["passes"]
    for model in models:
        if model["family"] == "innovation_spo":
            model["metrics"]["postcorrection"]["by_regime"]["lambda3"][M.GAP] = math.nextafter(9., math.inf)
    assert not selected(M.criteria(models, hold, technical_complete=True), name)["passes"]
    for model in models:
        for regime in M.REGIMES:
            model["metrics"]["postcorrection"]["by_regime"][regime][M.GAP] = 0.
    failed = [row for row in M.criteria(models, hold, technical_complete=True) if not row["passes"]]
    assert len(failed) == 6 and all(row["relation"] == "<= and <" for row in failed)


def test_one_worse_seed_fails_even_when_mean_and_architecture_gap_pass():
    models, hold = gate_fixture()
    for model in models:
        if model["family"] == "innovation_spo":
            model["metrics"]["postcorrection"]["by_regime"]["lambda3"][M.GAP] = 11. if model["seed"] == M.SEEDS[0] else 5.
    rows = M.criteria(models, hold, technical_complete=True)
    assert selected(rows, "objective.innovation.lambda3.postcorrection.gap")["passes"]
    assert selected(rows, "architecture.lambda3.postcorrection.gap")["passes"]
    failed = [row["name"] for row in rows if not row["passes"]]
    assert failed == [f"objective.innovation.lambda3.{M.SEEDS[0]}.postcorrection.gap_nonregression"]
    gates = M.gate_decisions(rows)
    assert not gates["innovation_objective"]["passes"] and not gates["architecture"]["passes"]
    assert gates["gru_objective"]["passes"]


def test_architecture_requires_better_of_both_gru_cells_for_each_metric():
    models, hold = gate_fixture()
    for model in models:
        if model["family"] == "gru_aux":
            for regime in M.REGIMES:
                model["metrics"]["postcorrection"]["by_regime"][regime][M.GAP] = 6.
                for scope in ("full", "initial"):
                    model["metrics"][scope]["by_regime"][regime][M.AGREEMENT] = .6
    rows = M.criteria(models, hold, technical_complete=True)
    for regime in M.REGIMES:
        gap = selected(rows, f"architecture.{regime}.postcorrection.gap")
        assert gap["threshold"] == .9 * 6 and gap["strict_upper_bound"] == 6 and not gap["passes"]
        for scope in ("full", "initial"):
            row = selected(rows, f"architecture.{regime}.{scope}.agreement")
            assert row["threshold"] == .6 and not row["passes"]


@pytest.mark.parametrize("scope", ["full", "initial"])
def test_agreement_equality_passes_but_any_representable_drop_fails(scope):
    models, hold = gate_fixture()
    name = f"objective.innovation.lambda3.{scope}.agreement"
    assert selected(M.criteria(models, hold, technical_complete=True), name)["passes"]
    for model in models:
        if model["family"] == "innovation_spo":
            model["metrics"][scope]["by_regime"]["lambda3"][M.AGREEMENT] = math.nextafter(.5, 0.)
    assert not selected(M.criteria(models, hold, technical_complete=True), name)["passes"]


def test_support_counts_originating_cases_and_hold_gap_must_be_positive():
    models, hold = gate_fixture(active_cases=3)
    rows = M.criteria(models, hold, technical_complete=True)
    failed = [row for row in rows if not row["passes"]]
    assert len(failed) == 8 and all(row["name"].endswith("case_support") and row["value"] == 3 for row in failed)
    models, hold = gate_fixture(active_cases=4)
    assert all(row["passes"] for row in M.criteria(models, hold, technical_complete=True))
    hold["postcorrection"]["by_regime"]["lambda3"][M.GAP] = 0.
    failed = [row for row in M.criteria(models, hold, technical_complete=True) if not row["passes"]]
    assert len(failed) == 1 and failed[0]["name"] == "common.lambda3.hold.postcorrection.positive_gap"
    assert failed[0]["relation"] == ">"


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_seed", "support", "nan", "prior_nan", "closure_int"])
def test_incomplete_nonfinite_or_changed_scored_domain_rejected(fault):
    models, hold = gate_fixture()
    complete = True
    if fault == "missing":
        models.pop()
    elif fault == "duplicate":
        models[-1] = models[0]
    elif fault == "wrong_seed":
        models[-1]["seed"] = -1
    elif fault == "support":
        models[-1]["metrics"]["initial"]["by_regime"]["lambda3"]["nonquery_rows"] -= 1
    elif fault == "nan":
        models[-1]["metrics"]["full"]["by_regime"]["lambda3"][M.GAP] = float("nan")
    elif fault == "prior_nan":
        models[-1]["prior_metrics"]["by_regime"]["lambda3"][M.PRIOR] = float("nan")
    elif fault == "closure_int":
        complete = 1
    with pytest.raises(ValueError):
        M.criteria(models, hold, technical_complete=complete)


@pytest.mark.parametrize("fault", ["renamed", "duplicate", "missing", "nonboolean"])
def test_gate_memberships_reject_corrupt_records(fault):
    models, hold = gate_fixture()
    rows = M.criteria(models, hold, technical_complete=True)
    if fault == "renamed":
        rows[0]["name"] = "common.other"
    elif fault == "duplicate":
        rows[-1] = rows[0]
    elif fault == "missing":
        rows.pop()
    else:
        rows[0]["passes"] = 1
    with pytest.raises(ValueError):
        M.gate_decisions(rows)
