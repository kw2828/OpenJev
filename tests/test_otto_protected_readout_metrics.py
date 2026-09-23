"""Fabricated numerical gates only; no model, optimizer or empirical inputs."""
import copy
import itertools
import math

import numpy as np
import pytest

from openjev.research import otto_prequery_metrics as prior_base
from openjev.research import otto_protected_readout_metrics as M
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
    offsets = np.concatenate((np.zeros(1, np.int64),
                              np.cumsum([len(e["features"]) for e in episodes], dtype=np.int64)))
    query = np.full((int(offsets[-1]), 4), np.nan, np.float32)
    query_mask, prior_mask = np.zeros(int(offsets[-1]), np.bool_), np.zeros(int(offsets[-1]), np.bool_)
    for low, high in itertools.pairwise(offsets):
        query_mask[low:high:4] = True
        prior_mask[low + 4:high:4] = True
        query[low + 4:high:4] = 0
    history = {"episode_ids": tuple(e["id"] for e in episodes),
               "episode_regimes": tuple(e["regime"] for e in episodes), "episode_offsets": offsets,
               "query_scores": query, "query_mask": query_mask}
    prior = np.full_like(query, np.nan)
    prior[prior_mask] = 0
    return history, prior, prior_mask, identities, episodes


def gate_fixture(active_cases=6, declared_cases=6):
    history, prior, _, identities, episodes = fixture([
        (regime, case, arm, 8 if case < active_cases else 1)
        for regime in M.REGIMES for case in range(declared_cases) for arm in M.ARMS])
    windows = build_windows(episodes)
    template = M.forecast_metrics(windows, windows["targets"].copy(), identities)
    calibration = M.prior_metrics(history, prior, identities)
    hold = copy.deepcopy(template)
    for regime in M.REGIMES:
        hold["postcorrection"]["by_regime"][regime][M.GAP] = 20.
    models = []
    gaps = {"pretrained": 10., "frozen_aux": 12., "frozen_spo": 9., "joint_aux": 11., "joint_spo": 13.}
    for family in M.FAMILIES:
        for seed in M.SEEDS:
            report = copy.deepcopy(template)
            for scope in M.SCOPES:
                for regime in M.REGIMES:
                    report[scope]["by_regime"][regime][M.AGREEMENT] = .5
                    report[scope]["by_regime"][regime][M.GAP] = gaps[family]
            models.append({"family": family, "seed": seed, "metrics": report,
                           "prior_metrics": copy.deepcopy(calibration),
                           "evaluation": {"parameter_mode": "frozen", "grad_enabled": False,
                                          "backbone_requires_grad": False,
                                          "residual_requires_grad": None if family == "pretrained" else True}})
    return models, hold


def selected(rows, name):
    return next(row for row in rows if row["name"] == name)


def change(models, family, *, scope="postcorrection", regime="lambda3", metric=None, value=0., seed=None):
    for model in models:
        if model["family"] == family and (seed is None or model["seed"] == seed):
            model["metrics"][scope]["by_regime"][regime][M.GAP if metric is None else metric] = value


def test_unchanged_metrics_keep_near_ties_case_counts_and_zero_support_denominators():
    assert M.forecast_metrics is forecast_base.forecast_metrics
    assert M.prior_metrics is prior_base.prior_metrics
    assert M.validate_prediction_support is prior_base.validate_prediction_support
    history, prior, mask, identities, episodes = fixture([
        ("lambda3", 0, "analytic", 8), ("lambda3", 0, "neural", 2), ("lambda3", 1, "analytic", 1)])
    for episode in episodes:
        episode["teacher_scores"][:] = [0., 5e-11, 999., 999.]
        episode["legal"][:, 2:] = False
    windows = build_windows(episodes)
    prediction = np.tile(np.array([2e-10, 0., -999., -999.], np.float32), (len(windows["targets"]), 4, 1))
    report = M.forecast_metrics(windows, prediction, identities)
    assert report["version"] == forecast_base.VERSION
    assert report["initial"]["overall"][M.AGREEMENT] == 2 / 3
    assert report["postcorrection"]["overall"][M.AGREEMENT] == 1 / 3
    assert report["initial"]["overall"]["episode_weighted_first_argmin_match"] == 0
    assert report["postcorrection"]["overall"]["supported_case_count"] == 1
    assert report["postcorrection"]["overall"]["declared_case_count"] == 2
    assert report["initial"]["overall"]["by_age"]["1"]["weight_mass"] == 2 / 3
    assert report["initial"]["overall"]["by_age"]["2"]["weight_mass"] == 1 / 3
    assert np.array_equal(M.validate_prediction_support(history, prior, mask), mask)
    prior_report = M.prior_metrics(history, prior, identities)
    assert prior_report["version"] == prior_base.VERSION
    assert prior_report["overall"]["episodes"] == 3
    assert prior_report["overall"]["supported_episodes"] == 1
    assert prior_report["overall"]["weight_mass"] == 1 / 3


def test_exact_29_conditions_hand_calculated_thresholds_and_one_gate():
    models, hold = gate_fixture()
    rows = M.criteria(models, hold, technical_complete=True)
    assert len(rows) == len({row["name"] for row in rows}) == 29
    assert sum(row["name"].startswith("common.") for row in rows) == 11
    assert sum(row["name"].startswith("candidate.") for row in rows) == 18
    assert tuple(row["name"] for row in rows) == M.condition_names()
    assert all(row["passes"] for row in rows)
    for regime in M.REGIMES:
        for control, value in (("pretrained", 10.), ("frozen_aux", 12.), ("joint_aux", 11.), ("joint_spo", 13.)):
            row = selected(rows, f"candidate.{regime}.postcorrection.gap_vs_{control}")
            assert row["value"] == 9. and row["threshold"] == .9 * value
            assert row["strict_upper_bound"] == value and row["relation"] == "<= and <"
    gates = M.gate_decisions(rows)
    assert set(gates) == {"protected_readout"}
    assert gates["protected_readout"] == {"conditions": list(M.condition_names()), "passed": 29,
                                         "total": 29, "passes": True}
    for failed in range(29):
        changed = copy.deepcopy(rows)
        changed[failed]["passes"] = False
        decision = M.gate_decisions(changed)["protected_readout"]
        assert not decision["passes"] and decision["passed"] == 28 and decision["total"] == 29
    pending = M.gate_decisions(M.criteria(models, hold, technical_complete=False))["protected_readout"]
    assert not pending["passes"] and pending["passed"] == 28


@pytest.mark.parametrize("control", M.COMPARATORS)
def test_every_control_can_independently_block_a_mean_gap_claim(control):
    models, hold = gate_fixture()
    change(models, control, value=6.)
    rows = M.criteria(models, hold, technical_complete=True)
    for other in M.COMPARATORS:
        row = selected(rows, f"candidate.lambda3.postcorrection.gap_vs_{other}")
        assert row["passes"] is (other != control)
    assert not M.gate_decisions(rows)["protected_readout"]["passes"]


def test_exact_ten_percent_boundary_and_zero_reference_cannot_pass_on_a_tie():
    models, hold = gate_fixture()
    name = "candidate.lambda3.postcorrection.gap_vs_pretrained"
    assert selected(M.criteria(models, hold, technical_complete=True), name)["passes"]
    change(models, "frozen_spo", value=math.nextafter(9., math.inf))
    assert not selected(M.criteria(models, hold, technical_complete=True), name)["passes"]
    for family in M.FAMILIES:
        for regime in M.REGIMES:
            change(models, family, regime=regime, value=0.)
    failed = [row for row in M.criteria(models, hold, technical_complete=True) if not row["passes"]]
    assert len(failed) == 8 and all(row["relation"] == "<= and <" for row in failed)


def test_equal_seed_mean_cannot_hide_one_paired_seed_regression():
    models, hold = gate_fixture()
    for seed, gap in zip(M.SEEDS, (13., 5., 5.), strict=True):
        change(models, "frozen_spo", seed=seed, value=gap)
    rows = M.criteria(models, hold, technical_complete=True)
    for control in M.COMPARATORS:
        row = selected(rows, f"candidate.lambda3.postcorrection.gap_vs_{control}")
        assert row["value"] == 23 / 3 and row["passes"]
    failed = [row["name"] for row in rows if not row["passes"]]
    assert failed == [f"candidate.lambda3.{M.SEEDS[0]}.postcorrection.gap_vs_frozen_aux"]
    assert not M.gate_decisions(rows)["protected_readout"]["passes"]


@pytest.mark.parametrize("scope", ["full", "initial"])
@pytest.mark.parametrize("control", M.COMPARATORS)
def test_agreement_uses_largest_mean_across_all_four_controls(scope, control):
    models, hold = gate_fixture()
    change(models, control, scope=scope, metric=M.AGREEMENT, value=.6)
    rows = M.criteria(models, hold, technical_complete=True)
    row = selected(rows, f"candidate.lambda3.{scope}.agreement")
    assert row["threshold"] == .6 and row["value"] == .5 and not row["passes"]
    other_scope = "initial" if scope == "full" else "full"
    assert selected(rows, f"candidate.lambda3.{other_scope}.agreement")["passes"]


@pytest.mark.parametrize("scope", ["full", "initial"])
def test_agreement_equality_passes_and_smallest_representable_drop_fails(scope):
    models, hold = gate_fixture()
    name = f"candidate.lambda3.{scope}.agreement"
    assert selected(M.criteria(models, hold, technical_complete=True), name)["passes"]
    change(models, "frozen_spo", scope=scope, metric=M.AGREEMENT, value=math.nextafter(.5, 0.))
    assert not selected(M.criteria(models, hold, technical_complete=True), name)["passes"]


def test_support_counts_cases_not_collector_paths_and_hold_gap_is_strictly_positive():
    models, hold = gate_fixture(active_cases=3)
    failed = [row for row in M.criteria(models, hold, technical_complete=True) if not row["passes"]]
    assert len(failed) == 8
    assert all(row["name"].endswith("case_support") and row["value"] == 3 for row in failed)
    models, hold = gate_fixture(active_cases=4)
    assert all(row["passes"] for row in M.criteria(models, hold, technical_complete=True))
    hold["postcorrection"]["by_regime"]["lambda3"][M.GAP] = 0.
    failed = [row for row in M.criteria(models, hold, technical_complete=True) if not row["passes"]]
    assert len(failed) == 1 and failed[0]["name"] == "common.lambda3.hold.postcorrection.positive_gap"
    assert failed[0]["relation"] == ">"


def test_criteria_cannot_replace_complete_cohort_by_a_declared_subset():
    models, hold = gate_fixture(declared_cases=5)
    with pytest.raises(ValueError, match="complete fixed VALID"):
        M.criteria(models, hold, technical_complete=True)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_seed", "boolean_seed", "wrong_family",
                                 "nonquery_support", "age_support", "prior_support", "nan", "prior_nan", "closure_int"])
def test_incomplete_nonfinite_or_changed_scored_domain_rejected(fault):
    models, hold = gate_fixture()
    complete = True
    if fault == "missing":
        models.pop()
    elif fault == "duplicate":
        models[-1] = models[0]
    elif fault == "wrong_seed":
        models[-1]["seed"] = -1
    elif fault == "boolean_seed":
        models[-1]["seed"] = True
    elif fault == "wrong_family":
        models[-1]["family"] = "selected_control"
    elif fault == "nonquery_support":
        models[-1]["metrics"]["initial"]["by_regime"]["lambda3"]["nonquery_rows"] -= 1
    elif fault == "age_support":
        models[-1]["metrics"]["full"]["by_regime"]["lambda3"]["by_age"]["2"]["nonquery_rows"] -= 1
    elif fault == "prior_support":
        models[-1]["prior_metrics"]["by_regime"]["lambda3"]["prior_rows"] -= 1
    elif fault == "nan":
        models[-1]["metrics"]["full"]["by_regime"]["lambda3"][M.GAP] = float("nan")
    elif fault == "prior_nan":
        models[-1]["prior_metrics"]["by_regime"]["lambda3"][M.PRIOR] = float("nan")
    else:
        complete = 1
    with pytest.raises(ValueError):
        M.criteria(models, hold, technical_complete=complete)


@pytest.mark.parametrize("fault", ["missing", "extra", "grad_enabled", "integer_false", "backbone", "mode",
                                 "pretrained_residual", "adaptation_residual"])
def test_canonical_evaluation_assertion_has_exact_family_specific_flags(fault):
    models, hold = gate_fixture()
    target = models[0] if fault == "pretrained_residual" else models[-1]
    metadata = target["evaluation"]
    if fault == "missing":
        metadata.pop("grad_enabled")
    elif fault == "extra":
        metadata["unverified_override"] = True
    elif fault == "grad_enabled":
        metadata["grad_enabled"] = True
    elif fault == "integer_false":
        metadata["grad_enabled"] = 0
    elif fault == "backbone":
        metadata["backbone_requires_grad"] = True
    elif fault == "mode":
        metadata["parameter_mode"] = "joint"
    elif fault == "pretrained_residual":
        metadata["residual_requires_grad"] = True
    else:
        metadata["residual_requires_grad"] = False
    with pytest.raises(ValueError, match="canonical frozen no_grad"):
        M.criteria(models, hold, technical_complete=True)


@pytest.mark.parametrize("fault", ["renamed", "duplicate", "missing", "nonboolean", "reordered"])
def test_one_gate_rejects_changed_membership_or_nonboolean_records(fault):
    models, hold = gate_fixture()
    rows = M.criteria(models, hold, technical_complete=True)
    if fault == "renamed":
        rows[0]["name"] = "common.other"
    elif fault == "duplicate":
        rows[-1] = rows[0]
    elif fault == "missing":
        rows.pop()
    elif fault == "nonboolean":
        rows[0]["passes"] = 1
    else:
        rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError):
        M.gate_decisions(rows)


def test_owned_summary_retains_all_fifteen_models_hold_and_false_closure():
    models, hold = gate_fixture()
    before = copy.deepcopy((models, hold))
    summary = M.summarize(list(reversed(models)), hold, technical_complete=False)
    assert (models, hold) == before
    assert [(record["family"], record["seed"]) for record in summary["models"]] == [
        (family, seed) for seed in M.SEEDS for family in M.FAMILIES]
    assert summary["hold"] == hold and summary["hold"] is not hold
    assert summary["required_conditions"] == 29 and summary["required_passed"] == 28
    assert not summary["technical_complete"] and not summary["gates"]["protected_readout"]["passes"]
    assert summary["candidate"] == "frozen_spo"
    assert summary["comparators"] == ["pretrained", "frozen_aux", "joint_aux", "joint_spo"]
    summary["models"][0]["metrics"]["initial"]["by_regime"]["lambda3"][M.GAP] = 123.
    summary["hold"]["initial"]["by_regime"]["lambda3"][M.GAP] = 456.
    assert (models, hold) == before
    for family in M.FAMILIES:
        flags = M.evaluation_metadata(family)
        assert flags == {"parameter_mode": "frozen", "grad_enabled": False, "backbone_requires_grad": False,
                         "residual_requires_grad": None if family == "pretrained" else True}
    with pytest.raises(ValueError, match="canonical-evaluation family"):
        M.evaluation_metadata("unknown")
