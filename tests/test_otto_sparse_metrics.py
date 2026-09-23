"""Complete fabricated scalar cohorts; no files, policies or native execution."""
from __future__ import annotations

import copy
import math

import pytest

from openjev.research.otto_sparse_metrics import aggregate, select_threshold


def calibration():
    rows = [{"stage": "valid", "episode_index": i, "steps": 1 if i < 12 else 3} for i in range(24)]
    decisions = [{"episode_index": row["episode_index"], "step": step,
                  "entropy": .25 if row["episode_index"] < 12 else .75}
                 for row in rows for step in range(row["steps"])]
    return decisions, rows


def evaluation():
    # Integer steps vary with hit and block; weighted means differ from raw means.
    specs = (("analytic", 100, 2.), ("neural", 80, 10.), ("period2", 80, 5.5),
             ("random_pair", 80, 4.), ("entropy", 80, 4.5))
    rows = []
    for regime in ("lambda3", "lambda4", "lambda5"):
        for arm, base_steps, base_cost in specs:
            for case in range(24):
                hit, block = 1 + case % 3, case // 3
                steps = base_steps + hit + block
                online = base_cost + hit / 10 + block / 100
                queries = 0 if arm == "analytic" else steps if arm == "neural" else (steps + 1) // 2
                rows.append({"stage": "eval", "regime": regime, "arm": arm, "case": case,
                    "initial_hit": hit, "block": block, "steps": steps, "found": True,
                    "updates": steps, "blocked_steps": 0, "final_update_assimilated": True,
                    "queries": queries, "query_quota_valid": True, "init_seconds": .1,
                    "choose_seconds": online - .4, "update_seconds": .2, "setup_allocation_seconds": .1,
                    "controller_seconds": online, "environment_seconds": .3})
    mixtures = {r: {1: .5, 2: .3, 3: .2} for r in ("lambda3", "lambda4", "lambda5")}
    return rows, mixtures


def censor(row):
    row.update(found=False, steps=2188, updates=2188,
               queries=0 if row["arm"] == "analytic" else 2188 if row["arm"] == "neural" else 1094)


def test_exact_episode_balanced_lower_median_with_unequal_lengths_and_ties():
    decisions, rows = calibration()
    original = copy.deepcopy((decisions, rows))
    result = select_threshold(decisions, rows)
    # Twelve one-row episodes carry exactly half the total mass. A row-weighted
    # median would be .75; the required lower episode-balanced median is .25.
    assert result["threshold"] == result["value"] == .25
    assert result["selection_cumulative_weight_exact"] == [1, 2]
    assert result["selection_cumulative_weight"] == .5
    assert result["episodes"] == 24 and result["decisions"] == 48
    assert result["total_weight"] == 1. and result["direction"] == "entropy >= threshold"
    assert select_threshold(list(reversed(decisions)), list(reversed(rows))) == result
    assert (decisions, rows) == original


def test_calibration_requires_every_unique_finite_valid_step_and_all_24_episodes():
    decisions, rows = calibration()
    bad_inputs = [(decisions, rows[:-1]), (decisions, [rows[0], *rows[:-1]]),
                  (decisions[:-1], rows), ([decisions[0], *decisions[:-1]], rows)]
    for bad_decisions, bad_rows in bad_inputs:
        with pytest.raises(ValueError):
            select_threshold(bad_decisions, bad_rows)
    for field, value in (("entropy", math.nan), ("entropy", math.inf), ("entropy", -math.inf),
                         ("entropy", 1), ("episode_index", 24), ("step", 1)):
        altered = copy.deepcopy(decisions)
        altered[0][field] = value
        with pytest.raises(ValueError):
            select_threshold(altered, rows)
    for field, value in (("stage", "eval"), ("episode_index", True), ("steps", 0), ("steps", 2189)):
        altered = copy.deepcopy(rows)
        altered[0][field] = value
        with pytest.raises(ValueError):
            select_threshold(decisions, altered)


def test_all_360_rows_hit_mixtures_eight_blocks_and_single_calibration_allocation():
    rows, mixtures = evaluation()
    original = copy.deepcopy(rows)
    result = aggregate(rows, mixtures, 72.)
    assert result["episodes"] == 360 and result["paired_cases"] == 72
    assert result["primary_arm"] == "period2" and result["pilot_continuation"] is True
    assert len(result["required"]) == result["required_conditions"] == result["required_passed"] == 16
    assert len(result["diagnostic"]) == result["diagnostic_conditions"] == 54
    assert len({r["name"] for r in result["required"]}) == 16
    assert len({r["name"] for r in result["diagnostic"]}) == 54
    assert all("lambda5" not in c["name"] and "random_pair" not in c["name"]
               and "entropy" not in c["name"] for c in result["required"])
    per_row_bill = result["calibration_per_entropy_episode_seconds"]
    assert per_row_bill == 1. and per_row_bill * sum(r["arm"] == "entropy" for r in rows) == 72.
    for regime in ("lambda3", "lambda4", "lambda5"):
        local = result["regimes"][regime]
        assert len(local["blocks"]) == 8
        for arm, base_steps, base_cost in (("analytic", 100, 2.), ("neural", 80, 10.),
                ("period2", 80, 5.5), ("random_pair", 80, 4.), ("entropy", 80, 4.5)):
            means = local["means"][arm]
            # E[hit]=1.7, E[block]=3.5 under the declared positive-hit mixture.
            assert means["steps"] == pytest.approx(base_steps + 5.2)
            assert means["controller_seconds"] == pytest.approx(base_cost + .205)
            assert means["paid_controller_seconds"] == pytest.approx(base_cost + .205 + (arm == "entropy"))
            assert local["raw_counts"][arm]["episodes"] == local["raw_counts"][arm]["found"] == 24
            assert local["raw_counts"][arm]["steps"] == 24 * base_steps + 132
            expected_queries = 0 if arm == "analytic" else 24 * base_steps + 132 if arm == "neural" else 12 * base_steps + 72
            assert local["raw_counts"][arm]["queries"] == expected_queries
            for block, reduced in enumerate(local["blocks"]):
                assert reduced[arm]["steps"] == pytest.approx(base_steps + block + 1.7)
    assert rows == original and all("paid_controller_seconds" not in row for row in rows)


def test_only_prespecified_period2_can_pass_primary_and_bad_transfer_stays_visible():
    rows, mixtures = evaluation()
    censor(next(r for r in rows if r["regime"] == "lambda3" and r["arm"] == "period2" and r["case"] == 0))
    result = aggregate(rows, mixtures, 0.)
    assert result["primary_arm"] == "period2" and result["pilot_continuation"] is False
    assert any(not r["passes"] for r in result["required"] if r["name"].startswith("lambda3.period2"))
    assert all(r["passes"] for r in result["diagnostic"] if ".random_pair." in r["name"] or ".entropy." in r["name"])
    fresh, mixtures = evaluation()
    for row in fresh:
        if row["regime"] == "lambda5" and row["arm"] == "period2":
            censor(row)
    transfer = aggregate(fresh, mixtures, 0.)
    assert transfer["pilot_continuation"] is True and transfer["required_passed"] == 16
    assert any(not r["passes"] for r in transfer["diagnostic"] if r["name"].startswith("lambda5.period2"))
    assert transfer["regimes"]["lambda5"]["raw_counts"]["period2"]["found"] == 0


def test_censored_final_tails_weighted_success_and_false_prefix_quota_witness():
    rows, mixtures = evaluation()
    mixtures["lambda4"] = {"1": .2, "2": .3, "3": .5}
    for row in rows:
        if row["initial_hit"] == 3:
            censor(row)
    result = aggregate(rows, mixtures, 0.)
    for regime, expected in (("lambda3", .8), ("lambda4", .5), ("lambda5", .8)):
        local = result["regimes"][regime]
        for arm in ("analytic", "neural", "period2", "random_pair", "entropy"):
            assert local["means"][arm]["found"] == pytest.approx(expected)
            assert local["raw_counts"][arm]["found"] == 16
            assert all(block[arm]["found"] == pytest.approx(expected) for block in local["blocks"])
    fresh, mixtures = evaluation()
    next(r for r in fresh if r["arm"] == "random_pair")["query_quota_valid"] = False
    invalid_prefix = aggregate(fresh, mixtures, 0.)
    assert invalid_prefix["required_passed"] == 15 and invalid_prefix["pilot_continuation"] is False
    assert next(r for r in invalid_prefix["required"] if r["name"] == "causal_quotas")["passes"] is False


def test_rejects_dropped_duplicate_nonfinite_incomplete_tail_quota_and_cost_records():
    rows, mixtures = evaluation()
    for invalid in (rows[:-1], [rows[0], *rows[:-1]]):
        with pytest.raises(ValueError):
            aggregate(invalid, mixtures, 0.)
    for arm, field, value in (("period2", "found", False), ("period2", "steps", 0),
            ("period2", "steps", 2189), ("period2", "updates", 1), ("period2", "blocked_steps", 1),
            ("period2", "final_update_assimilated", False), ("period2", "queries", 42),
            ("period2", "query_quota_valid", 1), ("analytic", "queries", 1),
            ("neural", "queries", 0), ("period2", "controller_seconds", math.nan),
            ("period2", "controller_seconds", 20.), ("period2", "environment_seconds", math.inf),
            ("period2", "setup_allocation_seconds", -1.), ("period2", "initial_hit", 3),
            ("period2", "block", 7), ("period2", "stage", "valid")):
        altered = copy.deepcopy(rows)
        next(r for r in altered if r["arm"] == arm)[field] = value
        with pytest.raises(ValueError):
            aggregate(altered, mixtures, 0.)
    for calibration_cost in (-1., math.inf, math.nan, True):
        with pytest.raises(ValueError):
            aggregate(rows, mixtures, calibration_cost)
