"""Fabricated decision ties and episode-balanced additive decompositions only."""
from __future__ import annotations

import copy
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_prequery_decision_diagnostic_fixture", ROOT / "scripts/diagnose_otto_prequery_decisions.py")
diagnostic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnostic
SPEC.loader.exec_module(diagnostic)


def test_strict_float32_near_sets_use_first_legal_action_and_preserve_positive_correct_gap():
    epsilon = np.float32(1e-10)
    below = np.nextafter(epsilon, np.float32(0))
    q = np.array([[0, below, 2, -999], [0, epsilon, 2, -999],
                  [0, 0, 1, 2], [-100, 2, 2, 4]], np.float32)
    p = np.array([[2, 0, 3, -1000], [epsilon, 0, 3, -1000],
                  [below, 0, 3, 4], [-200, 10, 10, 12]], np.float32)
    legal = np.array([[True, True, True, False], [True, True, True, False],
                      [True, True, True, True], [False, True, True, True]])
    before = [a.tobytes() for a in (q, p, legal)]
    result = diagnostic.decisions(q, p, legal)
    assert set(result) == {"action", "correct", "gap"}
    assert result["action"].dtype == np.int64 and result["action"].tolist() == [1, 1, 0, 1]
    assert result["correct"].dtype == np.bool_ and result["correct"].tolist() == [True, False, True, True]
    assert result["gap"].dtype == np.float64
    np.testing.assert_array_equal(result["gap"], [float(below), float(epsilon), 0., 0.])
    assert result["correct"][0] and result["gap"][0] > 0
    assert [a.tobytes() for a in (q, p, legal)] == before


def test_extreme_finite_float32_subtraction_overflow_is_not_a_tie_or_infinite_raw_gap():
    largest = np.finfo(np.float32).max
    q = np.array([[-largest, largest, 0, 1], [-largest, -largest, 0, largest]], np.float32)
    p = np.array([[largest, -largest, 0, 1], [-largest, -largest, 0, largest]], np.float32)
    with np.errstate(over="raise", invalid="raise"):
        result = diagnostic.decisions(q, p, np.ones((2, 4), np.bool_))
    assert result["action"].tolist() == [1, 0]
    assert result["correct"].tolist() == [False, True]
    assert result["gap"].tolist() == [2 * float(largest), 0.]
    assert np.isfinite(result["gap"]).all()


@pytest.mark.parametrize("fault", ["score_dtype", "legal_dtype", "shape", "nonfinite", "no_legal"])
def test_invalid_decision_contract_raises_value_error(fault):
    q, p = np.zeros((2, 4), np.float32), np.zeros((2, 4), np.float32)
    legal = np.ones((2, 4), np.bool_)
    if fault == "score_dtype":
        q = q.astype(np.float64)
    elif fault == "legal_dtype":
        legal = legal.astype(np.int64)
    elif fault == "shape":
        p = p[:, :3]
    elif fault == "nonfinite":
        q[0, 3] = np.inf
        legal[0, 3] = False  # Invalid scores are rejected even on a blocked action.
    else:
        legal[1] = False
    with pytest.raises(ValueError):
        diagnostic.decisions(q, p, legal)


def decomposition_fixture():
    delta = 2.**-35
    base = {"action": np.array([0, 1, 1, 1, 1, 0], np.int64),
            "correct": np.array([True, False, False, True, False, True]),
            "gap": np.array([0., 4., 6., delta, 2., 0.], np.float64)}
    candidate = {"action": np.array([1, 0, 0, 2, 2, 1], np.int64),
                 "correct": np.array([False, True, True, True, False, False]),
                 "gap": np.array([3., 0., 0., 2 * delta, 1., 5.], np.float64)}
    steps = np.array([1, 2, 5, 6, 1, 5], np.int64)
    indices = np.array([0, 0, 0, 0, 1, 1], np.int64)
    identities = [{"episode_id": "a", "regime": "lambda3", "case": 0, "arm": "analytic"},
                  {"episode_id": "b", "regime": "lambda3", "case": 0, "arm": "neural"},
                  {"episode_id": "c", "regime": "lambda4", "case": 0, "arm": "analytic"}]
    return base, candidate, steps, indices, identities


def test_unequal_episode_lengths_use_full_episode_contributions_and_exact_reweight_identity():
    args = decomposition_fixture()
    original = copy.deepcopy(args)
    result = diagnostic.decompose(*args)
    episodes = result["episodes"]
    assert [r["episode_id"] for r in episodes] == ["a", "b", "c"]
    a, b, zero = episodes
    delta = 2.**-35
    assert (a["rows"], a["initial_rows"], a["post_rows"]) == (4, 2, 2)
    assert (b["rows"], b["initial_rows"], b["post_rows"]) == (2, 1, 1)
    assert (a["baseline_full_agreement"], a["candidate_full_agreement"]) == (.5, .75)
    assert (a["baseline_primary_agreement"], a["candidate_primary_agreement"]) == (.5, 1.)
    assert a["initial_agreement_contribution"] == 0 and a["post_agreement_contribution"] == .25
    assert a["reweight_agreement"] == -.25
    assert a["initial_gap_contribution"] == -.25
    assert a["post_gap_contribution"] == (-6 + delta) / 4
    assert a["reweight_gap"] == (6 - delta) / 4
    assert b["initial_gap_contribution"] == -.5 and b["post_gap_contribution"] == 2.5
    assert b["reweight_gap"] == -2.5 and b["reweight_agreement"] == .5
    for row in episodes:
        for metric in ("agreement", "gap"):
            full = row[f"candidate_full_{metric}"] - row[f"baseline_full_{metric}"]
            primary = row[f"candidate_primary_{metric}"] - row[f"baseline_primary_{metric}"]
            initial = row[f"initial_{metric}_contribution"]
            post = row[f"post_{metric}_contribution"]
            reweight = row[f"reweight_{metric}"]
            assert full == pytest.approx(initial + post, abs=1e-15)
            assert full == pytest.approx(primary + initial + reweight, abs=1e-15)
    mean_delta = math.fsum(r["candidate_full_agreement"] - r["baseline_full_agreement"] for r in episodes) / 3
    pooled_delta = (int(args[1]["correct"].sum()) - int(args[0]["correct"].sum())) / 6
    assert mean_delta == -1 / 12 and pooled_delta == 0
    identity_keys = {"episode_id", "regime", "case", "arm"}
    assert all(value == 0 for key, value in zero.items() if key not in identity_keys)
    for left, right in zip(args[:4], original[:4], strict=True):
        if isinstance(left, dict):
            assert all(left[k].tobytes() == right[k].tobytes() for k in left)
        else:
            assert left.tobytes() == right.tobytes()
    assert args[4] == original[4]


def test_all_eight_transition_cells_retained_and_correct_correct_can_have_positive_gap_delta():
    args = decomposition_fixture()
    result = diagnostic.decompose(*args)
    rows = result["transitions"]
    by = {(r["episode_id"], r["phase"], r["cell"]): r for r in rows}
    assert len(rows) == len(by) == 24
    assert set(by) == {(i["episode_id"], phase, cell) for i in args[-1]
                       for phase in ("initial", "post") for cell in ("CC", "CW", "WC", "WW")}
    cc = by["a", "post", "CC"]
    assert cc["count"] == 1 and cc["mass"] == .25
    assert cc["baseline_gap"] == 2.**-37 and cc["candidate_gap"] == 2.**-36
    assert cc["gap_delta"] == 2.**-37 > 0
    for episode in result["episodes"]:
        selected = [r for r in rows if r["episode_id"] == episode["episode_id"]]
        assert sum(r["count"] for r in selected) == episode["rows"]
        assert math.fsum(r["mass"] for r in selected) == (1. if episode["rows"] else 0.)
        assert math.fsum(r["gap_delta"] for r in selected) == pytest.approx(
            episode["candidate_full_gap"] - episode["baseline_full_gap"], abs=1e-15)
        if not episode["rows"]:
            assert all(r[k] == 0 for r in selected for k in ("count", "mass", "baseline_gap", "candidate_gap", "gap_delta"))


def test_episode_without_post_rows_has_zero_primary_but_retains_initial_full_change():
    base, candidate, steps, indices, identities = decomposition_fixture()
    chosen = slice(0, 2)
    result = diagnostic.decompose({k: v[chosen] for k, v in base.items()},
        {k: v[chosen] for k, v in candidate.items()}, steps[chosen], indices[chosen], identities)
    row = result["episodes"][0]
    assert row["rows"] == row["initial_rows"] == 2 and row["post_rows"] == 0
    assert row["baseline_primary_gap"] == row["candidate_primary_gap"] == row["reweight_gap"] == 0
    assert row["initial_gap_contribution"] == -.5
    assert row["post_gap_contribution"] == row["reweight_agreement"] == 0
    assert len(result["episodes"]) == 3


@pytest.mark.parametrize("fault", ["duplicate", "query_step"])
def test_duplicate_episode_step_or_query_rows_rejected(fault):
    args = list(decomposition_fixture())
    args[2] = args[2].copy()
    args[2][1] = 1 if fault == "duplicate" else 4
    with pytest.raises(ValueError):
        diagnostic.decompose(*args)
