"""Fabricated initial/support/factorial boundaries, without scientific data."""
import copy
import itertools
import math

import numpy as np
import pytest

from openjev.research import otto_separate_prior_metrics as M
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


def test_initial_scope_uses_original_episode_denominator_and_separate_age_weights():
    _, _, _, ids, episodes = fixture([("lambda3", 0, "analytic", 8),
        ("lambda3", 0, "neural", 2), ("lambda3", 1, "analytic", 1)])
    for episode in episodes:
        episode["teacher_scores"][:] = [0, 2, 9, 10]
    windows = build_windows(episodes)
    predictions = windows["targets"].copy()
    predictions[0, 1:4, 0] = 3  # first path's initial window all wrong, later all correct.
    result = M.forecast_metrics(windows, predictions, ids)
    initial, post, full = (result[s]["overall"] for s in ("initial", "postcorrection", "full"))
    assert initial["episode_weighted_agreement"] == 1/3
    assert initial["episode_weighted_raw_gap"] == 2/3
    assert post["episode_weighted_agreement"] == 1/3 and post["nonquery_rows"] == 3
    assert full["episode_weighted_agreement"] == .5
    assert initial["supported_case_count"] == 1 and initial["declared_case_count"] == 2
    assert initial["by_age"]["1"]["weight_mass"] == 2/3
    assert initial["by_age"]["2"]["weight_mass"] == 1/3
    assert result["initial"]["by_case"][1]["supported_episode_agreement"] is None


def test_qualified_float32_near_tie_and_common_offset():
    _, _, _, ids, episodes = fixture([("lambda3", 0, "analytic", 8)])
    episodes[0]["teacher_scores"][:] = [0, 5e-11, 999, 999]
    episodes[0]["legal"][:, 2:] = False
    windows = build_windows(episodes)
    predictions = np.tile(np.array([2e-10, 0, -999, -999], np.float32), (2, 4, 1))
    result = M.forecast_metrics(windows, predictions, ids)
    for scope in M.SCOPES:
        assert result[scope]["overall"][M.AGREEMENT] == 1
        assert result[scope]["overall"]["episode_weighted_first_argmin_match"] == 0
        assert result[scope]["overall"][M.GAP] == float(np.float32(5e-11))


def gate_fixture():
    history, prior, _, ids, episodes = fixture([
        (r, c, a, 8) for r in M.REGIMES for c in range(6) for a in M.ARMS])
    windows = build_windows(episodes)
    template = M.forecast_metrics(windows, windows["targets"].copy(), ids)
    prior_template = M.prior_metrics(history, prior, ids)
    models = []
    for family in M.FAMILIES:
        for seed in M.SEEDS:
            metric, calibration = copy.deepcopy(template), copy.deepcopy(prior_template)
            separate_aux = family.endswith("separate_aux")
            explicit = family.startswith("innovation")
            for regime in M.REGIMES:
                for scope in M.SCOPES:
                    group = metric[scope]["by_regime"][regime]
                    group[M.AGREEMENT] = .48 if family.endswith("shared_aux") else .5
                    group[M.GAP] = (7. if explicit else 8.) if separate_aux else 8. if family.endswith("shared_aux") else 10.
                calibration["by_regime"][regime][M.PRIOR] = 6. if separate_aux else 8. if family.endswith("shared_aux") else 10.
            models.append({"family": family, "seed": seed, "metrics": metric, "prior_metrics": calibration})
    return models, template


def test_39_inventory_three_gates_and_no_omnibus_promotion():
    models, hold = gate_fixture()
    rows = M.criteria(models, hold, technical_complete=True)
    assert len(rows) == len({r["name"] for r in rows}) == 39 and all(r["passes"] for r in rows)
    gates = M.gate_decisions(rows)
    assert {k: v["total"] for k, v in gates.items()} == {
        "innovation_mechanism": 19, "gru_mechanism": 19, "architecture": 29}
    for failed in range(39):
        changed = copy.deepcopy(rows)
        changed[failed]["passes"] = False
        outcomes = M.gate_decisions(changed)
        for gate, value in outcomes.items():
            member = rows[failed]["name"] in gates[gate]["conditions"]
            assert value["passes"] is (not member)
            assert value["passed"] == value["total"] - int(member)
    pending = M.gate_decisions(M.criteria(models, hold, technical_complete=False))
    assert all(not v["passes"] for v in pending.values())


@pytest.mark.parametrize("scope", ["full", "initial"])
def test_one_percentage_point_recovery_is_exact_and_not_a_tolerance(scope):
    models, hold = gate_fixture()
    for row in models:
        if row["family"] == "innovation_shared_aux":
            row["metrics"][scope]["by_regime"]["lambda3"][M.AGREEMENT] = .6
        elif row["family"] == "innovation_separate_aux":
            row["metrics"][scope]["by_regime"]["lambda3"][M.AGREEMENT] = .61
    name = f"mechanism.innovation.lambda3.{scope}.agreement_recovery"
    check = next(r for r in M.criteria(models, hold, technical_complete=True) if r["name"] == name)
    assert check["threshold"] == .61 and check["passes"]
    for row in models:
        if row["family"] == "innovation_separate_aux":
            row["metrics"][scope]["by_regime"]["lambda3"][M.AGREEMENT] = math.nextafter(.61, -math.inf)
    assert not next(r for r in M.criteria(models, hold, technical_complete=True) if r["name"] == name)["passes"]


def test_zero_gap_and_prior_ties_do_not_establish_strict_gains():
    models, hold = gate_fixture()
    for row in models:
        for regime in M.REGIMES:
            row["metrics"]["postcorrection"]["by_regime"][regime][M.GAP] = 0.
            row["prior_metrics"]["by_regime"][regime][M.PRIOR] = 0.
    failed = [r for r in M.criteria(models, hold, technical_complete=True) if not r["passes"]]
    assert len(failed) == 10 and all(r["relation"] == "<= and <" for r in failed)


def test_factorial_interactions_have_paired_seed_and_equal_seed_mean_units():
    models, _ = gate_fixture()
    result = M.interactions(models)
    assert len(result["per_seed"]) == 84 and len(result["means"]) == 28
    one = next(r for r in result["means"] if r["architecture"] == "innovation"
               and r["regime"] == "lambda3" and r["scope"] == "postcorrection" and r["metric"] == M.GAP)
    assert one["shared_aux_minus_mse"] == -2.
    assert one["separate_aux_minus_mse"] == -3.
    assert one["interaction"] == -1.


@pytest.mark.parametrize("fault", ["missing", "duplicate", "support", "nan", "initial_nan", "bad_prior"])
def test_rejects_incomplete_domains_or_nonfinite_metrics(fault):
    models, hold = gate_fixture()
    if fault == "missing":
        models.pop()
    elif fault == "duplicate":
        models[-1] = models[0]
    elif fault == "support":
        models[-1]["metrics"]["initial"]["by_regime"]["lambda3"]["nonquery_rows"] -= 1
    elif fault == "bad_prior":
        models[-1]["prior_metrics"]["by_regime"]["lambda3"][M.PRIOR] = float("nan")
    else:
        models[-1]["metrics"]["initial" if fault == "initial_nan" else "full"]["by_regime"]["lambda3"][M.GAP] = float("nan")
    with pytest.raises(ValueError):
        M.criteria(models, hold, technical_complete=True)
