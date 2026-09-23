"""Raw-unit prior calibration and all fixed prospective comparison conditions."""
import copy
import itertools
import json

import numpy as np
import pytest

from openjev.research import otto_cross_query_metrics as original
from openjev.research import otto_prequery_metrics as M
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
    lengths = [len(e["features"]) for e in episodes]
    offsets = np.concatenate((np.zeros(1, np.int64), np.cumsum(lengths, dtype=np.int64)))
    total = int(offsets[-1])
    query = np.full((total, 4), np.nan, np.float32)
    mask, prior_mask = np.zeros(total, np.bool_), np.zeros(total, np.bool_)
    for low, high in itertools.pairwise(offsets):
        mask[low:high:4] = True
        prior_mask[low + 4:high:4] = True
        query[low + 4:high:4] = 0
    history = {"episode_ids": tuple(e["id"] for e in episodes),
               "episode_regimes": tuple(e["regime"] for e in episodes),
               "episode_offsets": offsets, "query_scores": query, "query_mask": mask}
    prior = np.full_like(query, np.nan)
    prior[prior_mask] = 0
    return history, prior, prior_mask, identities, episodes


def test_raw_all_four_mse_equal_episodes_then_queries_retains_unsupported_episodes():
    history, prior, mask, identities, _ = fixture([
        ("lambda3", 0, "analytic", 9), ("lambda3", 0, "neural", 5), ("lambda4", 0, "analytic", 4)])
    prior[4] = [0, 0, 0, 4]  # centered MSE3
    prior[8] = [0, 0, 0, 8]  # centered MSE12, episode mean7.5
    prior[13] = [0, 0, 0, 4]  # second episode mean3
    before = {k: v.tobytes() for k, v in history.items() if isinstance(v, np.ndarray)}
    result = M.prior_metrics(history, prior, identities)
    group = result["overall"]
    assert group["episode_weighted_centered_mse"] == 3.5
    assert group["supported_episode_centered_mse"] == 5.25
    assert group["prior_rows"] == 3 and group["episodes"] == 3
    assert group["weight_mass"] == pytest.approx(2 / 3)
    assert group["supported_case_count"] == 1 and group["declared_case_count"] == 2
    assert result["by_regime"]["lambda3"]["episode_weighted_centered_mse"] == 5.25
    assert result["by_case"][0]["episodes"] == 2
    assert [e["episode_weighted_centered_mse"] for e in result["by_episode"]] == [7.5, 3., 0.]
    assert result["by_regime"]["lambda4"]["supported_episode_centered_mse"] is None
    assert M.validate_prediction_support(history, prior, mask).tobytes() == mask.tobytes()
    assert all(history[k].tobytes() == b for k, b in before.items())
    shifted_history = {**history, "query_scores": history["query_scores"] - np.float32(128)}
    assert M.prior_metrics(shifted_history, prior + np.float32(256), identities) == result


def test_first_query_and_nonqueries_are_inert_even_when_poisoned_and_zero_support_explicit():
    history, prior, mask, identities, _ = fixture([
        ("lambda3", 0, "analytic", 1), ("lambda4", 0, "neural", 4)])
    result = M.prior_metrics(history, prior, identities)
    assert result["overall"]["prior_rows"] == result["overall"]["supported_episodes"] == 0
    assert result["overall"]["episode_weighted_centered_mse"] == 0
    assert result["overall"]["supported_episode_centered_mse"] is None
    assert result["overall"]["zero_support_episode_ids"] == list(history["episode_ids"])
    checked = M.validate_prediction_support(history, prior, mask)
    with pytest.raises(ValueError):
        checked.flags.writeable = True
    json.dumps(result, allow_nan=False)
    assert M.forecast_metrics is original.forecast_metrics


@pytest.mark.parametrize("fault", ["drop_later", "include_first", "active_nan", "teacher_nan", "wrong_dtype", "wrong_case"])
def test_prior_contract_rejects_missing_support_or_active_corruption(fault):
    history, prior, mask, identities, _ = fixture([("lambda3", 0, "analytic", 5)])
    if fault == "drop_later":
        mask[4] = False
    elif fault == "include_first":
        mask[0] = True
    elif fault == "active_nan":
        prior[4, 0] = np.nan
    elif fault == "teacher_nan":
        history["query_scores"][4, 0] = np.nan
    elif fault == "wrong_dtype":
        prior = prior.astype(np.float64)
    else:
        identities[0]["case"] = False
    with pytest.raises(ValueError):
        M.validate_prediction_support(history, prior, mask)
        M.prior_metrics(history, prior, identities)


def gate_fixture():
    history, prior, _, identities, episodes = fixture([
        (regime, case, arm, 8) for regime in M.REGIMES for case in range(4) for arm in M.ARMS])
    windows = build_windows(episodes)
    template = original.forecast_metrics(windows, windows["targets"].copy(), identities)
    prior_template = M.prior_metrics(history, prior, identities)
    hold = copy.deepcopy(template)
    for regime in M.REGIMES:
        held = hold["postcorrection"]["by_regime"][regime]
        held["episode_weighted_agreement"] = .5
        held["episode_weighted_raw_gap"] = 11.25
        for age in M.AGES:
            held["by_age"][age]["episode_weighted_raw_gap"] = 1.
    models = []
    for kind in M.FAMILIES:
        for seed in M.SEEDS:
            report, prior_report = copy.deepcopy(template), copy.deepcopy(prior_template)
            for regime in M.REGIMES:
                for scope in ("full", "postcorrection"):
                    group = report[scope]["by_regime"][regime]
                    group["episode_weighted_agreement"] = .5
                    group["episode_weighted_raw_gap"] = 1. if scope == "full" else 9. if kind == "innovation_aux" else 10.
                    for age in M.AGES:
                        group["by_age"][age]["episode_weighted_raw_gap"] = 1.
                prior_report["by_regime"][regime]["episode_weighted_centered_mse"] = 8. if kind == "innovation_aux" else 10.
            models.append({"family": kind, "seed": seed, "metrics": report, "prior_metrics": prior_report})
    return models, hold


def test_all_55_prospective_rules_and_explicit_strict_ratio_schema():
    models, hold = gate_fixture()
    rows = M.criteria(models, hold, technical_complete=True)
    assert len(rows) == len({r["name"] for r in rows}) == 55
    assert all(r["passes"] for r in rows)
    strict = [r for r in rows if r["relation"] == "<= and <"]
    assert len(strict) == 6
    assert all(r["value"] == r["threshold"] == 9. and r["strict_upper_bound"] == 10. for r in strict)
    assert sum("mean_prior_mse" in r["name"] for r in rows) == 2
    assert sum(".full." in r["name"] for r in rows) == 4
    assert not M.criteria(models, hold, technical_complete=False)[0]["passes"]
    json.dumps(rows, allow_nan=False)


def test_zero_vs_zero_learned_gaps_fail_all_six_strict_comparisons():
    models, hold = gate_fixture()
    for model in models:
        for regime in M.REGIMES:
            model["metrics"]["postcorrection"]["by_regime"][regime]["episode_weighted_raw_gap"] = 0.
    failed = [r for r in M.criteria(models, hold, technical_complete=True) if not r["passes"]]
    assert len(failed) == 6 and all(r["relation"] == "<= and <" for r in failed)


@pytest.mark.parametrize("regime", M.REGIMES)
def test_each_added_prior_condition_can_fail_independently(regime):
    models, hold = gate_fixture()
    for model in models:
        if model["family"] == "innovation_aux":
            model["prior_metrics"]["by_regime"][regime]["episode_weighted_centered_mse"] = 8.01
    failed = [r["name"] for r in M.criteria(models, hold, technical_complete=True) if not r["passes"]]
    assert failed == [f"{regime}.mean_prior_mse_vs_innovation_mse"]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "support", "nan_prior"])
def test_gate_rejects_different_domains_or_incomplete_cells(fault):
    models, hold = gate_fixture()
    if fault == "missing":
        models.pop()
    elif fault == "duplicate":
        models[-1] = models[0]
    elif fault == "support":
        models[-1]["prior_metrics"]["by_regime"]["lambda3"]["prior_rows"] -= 1
    else:
        models[-1]["prior_metrics"]["by_regime"]["lambda3"]["episode_weighted_centered_mse"] = float("nan")
    with pytest.raises(ValueError):
        M.criteria(models, hold, technical_complete=True)
