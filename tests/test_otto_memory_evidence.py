"""Synthetic-only public-history, alignment and conditional-summary fixtures."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "otto_evidence_tested", Path(__file__).resolve().parents[1] / "scripts/diagnose_otto_memory_evidence.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


@pytest.fixture
def analytic():
    def moved(position, action):
        p = list(position)
        axis = action // 2
        p[axis] = min(2, max(0, p[axis] + (-1 if action % 2 == 0 else 1)))
        return tuple(p)

    def normalized(p):
        if not np.isfinite(p).all() or np.any(p < 0) or not p.sum() > 0:
            raise ValueError("invalid fixture posterior")
        return p / p.sum()

    def likelihood_at(kernel, position):
        x, y = position
        return kernel[:, 3 - x:6 - x, 3 - y:6 - y]

    def prior(kernel, hit):
        p = np.arange(1, 10, dtype=float).reshape(3, 3)
        p[1, 1] = 0
        return normalized(p * likelihood_at(kernel, (1, 1))[hit])

    def action_scores(p, position, _kernel, space):
        assert space is True
        grid = np.indices(p.shape)
        return [None if moved(position, a) == position else
                float(np.sum(p * (np.abs(grid[0] - moved(position, a)[0]) + np.abs(grid[1] - moved(position, a)[1]))))
                for a in range(4)]

    return SimpleNamespace(N=3, NHITS=4, CENTER=(1, 1), moved=moved, normalized=normalized,
                           likelihood_at=likelihood_at, prior=prior, action_scores=action_scores)


@pytest.fixture
def kernel():
    k = np.arange(1, 197, dtype=float).reshape(4, 7, 7)
    k /= k.sum(axis=0, keepdims=True)
    k[:, 3, 3] = 0
    return k


def packet(analytic, step, position, hit):
    return {"position": list(position), "hit": hit, "done": False, "step": step,
            "valid_actions": [a for a in range(4) if analytic.moved(position, a) != position]}


def history(analytic, hits):
    positions = ((0, 1), (0, 0), (1, 0), (0, 0))
    initial = packet(analytic, 0, analytic.CENTER, 2)
    return initial, [packet(analytic, i, positions[(i - 1) % 4], hit) for i, hit in enumerate(hits, 1)]


@pytest.mark.parametrize("length", [0, 1, 31, 32])
def test_no_old_evidence_before_33_completed_moves(analytic, kernel, length):
    initial, records = history(analytic, [0, 1, 2, 3] * 8)
    beliefs, counts = M.evidence_beliefs(initial, records[:length], kernel, analytic)
    assert counts["old_positive_count"] == counts["old_zero_count"] == 0
    for p in beliefs.values():
        np.testing.assert_array_equal(p, beliefs["full"])


@pytest.mark.parametrize("old_hit", [0, 1, 2, 3])
def test_exact_window_boundary_and_saturated_positive(analytic, kernel, old_hit):
    initial, records = history(analytic, [old_hit] + [0] * 32)
    beliefs, counts = M.evidence_beliefs(initial, records, kernel, analytic)
    assert counts["old_positive_count"] == int(old_hit > 0)
    assert counts["old_zero_count"] == int(old_hit == 0)
    matching = "old_positive" if old_hit > 0 else "old_zero"
    absent = "old_zero" if old_hit > 0 else "old_positive"
    np.testing.assert_array_equal(beliefs[matching], beliefs["full"])
    np.testing.assert_array_equal(beliefs[absent], beliefs["recent32"])
    assert np.max(np.abs(beliefs["full"] - beliefs["recent32"])) > 0


def test_four_factorial_beliefs_match_independent_log_product(analytic, kernel):
    initial, records = history(analytic, [3, 0] + [1, 0] * 16)
    beliefs, counts = M.evidence_beliefs(initial, records, kernel, analytic)
    assert counts["old_positive_count"] == counts["old_zero_count"] == 1
    visited = {tuple(initial["position"]), *(tuple(r["position"]) for r in records)}
    for mode in M.MODES:
        prior = analytic.prior(kernel, initial["hit"])
        allowed = prior > 0
        for cell in visited:
            allowed[cell] = False
        log_weight = np.log(prior[allowed])
        for index, row in enumerate(records):
            keep = index >= 2 or mode == "full" or (mode == "old_positive" and row["hit"] > 0) or (mode == "old_zero" and row["hit"] == 0)
            if keep:
                log_weight += np.log(analytic.likelihood_at(kernel, row["position"])[row["hit"]][allowed])
        expected = np.exp(log_weight - np.max(log_weight))
        expected /= expected.sum()
        np.testing.assert_allclose(beliefs[mode][allowed], expected, atol=1e-14, rtol=0)
        assert all(beliefs[mode][cell] == 0 for cell in visited)


def test_initial_hit_once_and_never_an_old_positive(analytic, kernel):
    initial, records = history(analytic, [0] * 40)
    beliefs, counts = M.evidence_beliefs(initial, records, kernel, analytic)
    assert counts["old_positive_count"] == 0
    assert counts["moves_since_last_positive"] == 40
    np.testing.assert_array_equal(beliefs["old_positive"], beliefs["recent32"])
    p = analytic.prior(kernel, initial["hit"])
    no_history, zero_counts = M.evidence_beliefs(initial, [], kernel, analytic)
    np.testing.assert_allclose(no_history["full"], p, atol=1e-16, rtol=0)
    assert zero_counts["moves_since_last_positive"] == 0


@pytest.mark.parametrize("mutation", ["terminal", "sentinel", "source", "time", "duplicate", "jump", "nonfinite"])
def test_invalid_or_nonpublic_history_fails(analytic, kernel, mutation):
    initial, records = history(analytic, [1, 0])
    if mutation == "terminal":
        records[0]["done"] = True
        records[0]["hit"] = -2
    elif mutation == "sentinel":
        records[0]["hit"] = -2
    elif mutation == "source":
        initial["source"] = [0, 0]
    elif mutation == "time":
        records[0]["step"] = 2
    elif mutation == "duplicate":
        records.append(copy.deepcopy(records[-1]))
    elif mutation == "jump":
        records[0] = packet(analytic, 1, (2, 2), 1)
    else:
        kernel[1, 1, 1] = np.nan
    with pytest.raises(ValueError):
        M.evidence_beliefs(initial, records, kernel, analytic)


def test_empty_support_fails_without_floor(analytic, kernel):
    initial, records = history(analytic, [1])
    kernel[1] = 0
    with pytest.raises(ValueError, match="posterior"):
        M.evidence_beliefs(initial, records, kernel, analytic)


def test_tie_rule_uses_first_within_tolerance_not_exact_argmin():
    action, count = M.choose([1 + 5e-11, 1.0, None, 2.0])
    assert action == 0 and count == 2
    assert M.choose([1 + 2e-10, 1.0, None, 2.0]) == (1, 1)


def test_factorial_contrasts_are_not_assumed_additive():
    assert M.factorial(dict(zip(M.MODES, [8.0, 5.0, 6.0, 1.0], strict=True))) == {
        "positive_without_old_zero": -3.0, "positive_with_old_zero": -5.0,
        "zero_without_old_positive": -2.0, "zero_with_old_positive": -4.0, "interaction": -2.0}


def test_descriptions_compare_chosen_actions_under_full_objective(analytic, kernel):
    initial, records = history(analytic, [3, 0] + [0, 1] * 16)
    beliefs, meta = M.evidence_beliefs(initial, records, kernel, analytic)
    result = M.describe(beliefs, meta, tuple(records[-1]["position"]), kernel, analytic)
    assert set(result["modes"]) == set(M.MODES)
    full = result["modes"]["full"]
    for mode, row in result["modes"].items():
        assert row["tv_to_full"] == pytest.approx(np.abs(beliefs[mode] - beliefs["full"]).sum() / 2)
        assert row["full_objective_excess"] == full["scores"][row["action"]] - min(x for x in full["scores"] if x is not None)
    assert full["tv_to_full"] == 0 and full["matches_full_action"] is True


def trace(analytic, actions, hits):
    position = analytic.CENTER
    result = [{"public": packet(analytic, 0, position, 1)}]
    for i, (action, hit) in enumerate(zip(actions, hits, strict=True), 1):
        position = analytic.moved(position, action)
        result.append({"action": action, "scores": [0.0, 1.0, 2.0, 3.0], "public": packet(analytic, i, position, hit)})
    return result


def test_first_disagreement_excludes_current_resulting_observations(analytic):
    left = trace(analytic, [0, 2, 1], [0, 1, 3])
    right = trace(analytic, [0, 2, 3], [0, 1, 0])
    assert M.first_disagreement(left, right) == 2
    assert M.first_disagreement(left, copy.deepcopy(left)) is None


def test_disagreeing_public_prefix_or_unmatched_termination_fails(analytic):
    left = trace(analytic, [0, 2, 1], [0, 1, 3])
    right = copy.deepcopy(left)
    right[1]["public"]["hit"] = 2
    with pytest.raises(ValueError, match="identical public"):
        M.first_disagreement(left, right)
    with pytest.raises(ValueError, match="termination"):
        M.first_disagreement(left, left[:-1])


def test_score_parity_requires_exact_action_even_within_numeric_tolerance():
    result = {"modes": {"full": {"scores": [0.0, 1.0, None, 2.0], "action": 0}}}
    assert M.parity(result, {"scores": [1e-9, 1.0, None, 2.0], "action": 0}) == 1e-9
    with pytest.raises(ValueError, match="selected-action"):
        M.parity(result, {"scores": [0.0, 1.0, None, 2.0], "action": 1})
    with pytest.raises(ValueError, match="score parity"):
        M.parity(result, {"scores": [2e-8, 1.0, None, 2.0], "action": 0})


def test_population_keeps_conditional_case_and_prefix_weights_distinct():
    cases = [
        {"initial_hit": 1, "prefix_means": {"x": 1.0}, "eligible_prefixes": 1},
        {"initial_hit": 1, "prefix_means": {"x": 3.0}, "eligible_prefixes": 3},
        {"initial_hit": 2, "prefix_means": {"x": 10.0}, "eligible_prefixes": 2},
        {"initial_hit": 3, "prefix_means": None, "eligible_prefixes": 0},
    ]
    result = M.population(cases, {1: .5, 2: .25, 3: .25})
    assert result["all_cases"] == 4 and result["eligible_cases"] == 3 and result["eligible_prefixes"] == 6
    assert result["unweighted_eligible_case_means"]["x"] == pytest.approx(14 / 3)
    assert result["stratum_weighted_eligible_case_means"]["x"] == pytest.approx(3.6)
    assert result["stratum_weighted_eligible_prefix_means"]["x"] == pytest.approx(4.0)
    assert result["eligible_case_mixture_mass"] == pytest.approx(1.25 / 32)
    assert result["by_initial_hit"]["3"]["eligible_cases"] == 0


def test_empty_selected_population_is_null_not_success():
    case = {"initial_hit": 1, "prefix_means": None, "eligible_prefixes": 0}
    result = M.population([case], {1: .5, 2: .25, 3: .25})
    assert result["eligible_cases"] == 0
    assert result["stratum_weighted_eligible_case_means"] is None
    assert result["stratum_weighted_eligible_prefix_means"] is None
