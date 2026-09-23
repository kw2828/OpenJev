"""Synthetic public-state metadata only; no saved journal or numerical imports."""
import importlib.util
import json
from pathlib import Path

import pytest

P = Path(__file__).resolve().parents[1] / "scripts/diagnose_otto_exact_cache.py"
SPEC = importlib.util.spec_from_file_location("_test_exact_cache", P)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def identity(name, **values):
    return {"episode_id": name, "stage": "train", "regime": "lambda3", "arm": "neural", **values}


def sample(name, step, **values):
    return {**identity(name), "step": step, "position": (26, 26), "posterior_sha256": "a" * 64, **values}


def test_exact_key_regime_position_posterior_and_true_lru_eviction():
    cache = module.Cache()
    def key(i):
        return "lambda3", (26, 26), f"{i:064x}"
    for i in range(64):
        assert cache.query(key(i)) == (False, False)
    assert cache.query(key(0)) == (True, False)
    assert cache.query(key(64)) == (False, True)
    assert key(0) in cache.entries and key(1) not in cache.entries
    assert cache.query(("lambda4", (26, 26), f"{0:064x}")) == (False, True)
    assert cache.query(("lambda3", (26, 25), f"{0:064x}")) == (False, True)


def test_episode_reset_and_partial_episode_rows_are_retained():
    cohort = [identity("a"), identity("b"), identity("unstarted", stage="valid", regime="lambda4", arm="analytic")]
    result = module.scan([sample("a", 0), sample("a", 1), sample("b", 0), sample("b", 1)],
                         cohort, [{**identity("a"), "steps": 2}], identity("b"), 4)
    assert result["totals"] == {"queries": 4, "hits": 2, "misses": 2, "evictions": 0}
    assert [row["partial_episode"] for row in result["episodes"]] == [False, True]
    assert result["episodes"][1]["misses"] == 1
    assert result["by_stage_regime_arm"][0]["queries"] == 4
    assert result["by_stage_regime_arm"][1] == {"stage": "valid", "regime": "lambda4", "arm": "analytic",
                                               "queries": 0, "hits": 0, "misses": 0, "evictions": 0}


@pytest.mark.parametrize("rows", [[sample("a", 1)], [sample("a", 0), sample("a", 0)],
                                  [sample("b", 0)], [sample("a", 0), sample("a", 2)]])
def test_nonsequential_duplicate_or_reordered_metadata_rejected(rows):
    with pytest.raises(ValueError):
        module.scan(rows, [identity("a")], [], identity("a"), len(rows))


def test_projection_ignores_scores_features_actions_and_truth_fields():
    raw = {**identity("a"), "step": 2, "public": {"position": [26, 25]}, "posterior": {"sha256": "f" * 64},
           "raw_q": [1e308, -1e308, 0, 4], "features": [1.25] * 31, "action": "unused",
           "source_evaluation_only": [9, 9], "held_q": {"not": "a score array"}}
    result = module.project(json.dumps(raw))
    assert result == sample("a", 2, position=(26, 25), posterior_sha256="f" * 64)
    raw["raw_q"] = {"arbitrary ignored labels": [None, "unknown"]}
    assert module.project(json.dumps(raw)) == result
    raw["step"] = "2"
    with pytest.raises(ValueError, match="integer metadata"):
        module.project(json.dumps(raw))


def test_complete_lengths_and_total_count_cannot_silently_drop_partial_rows():
    rows = [sample("a", 0), sample("a", 1)]
    with pytest.raises(ValueError, match="all returned"):
        module.scan(rows, [identity("a")], [], identity("a"), 1)
    with pytest.raises(ValueError, match="complete episode"):
        module.scan(rows, [identity("a")], [{**identity("a"), "steps": 1}], None, 2)


def test_all_twelve_declared_groups_show_unstarted_zero_exposure():
    cohort = [identity(f"{stage}-{regime}-{arm}", stage=stage, regime=regime, arm=arm)
              for stage in ("train", "valid") for regime in ("lambda3", "lambda4")
              for arm in ("analytic", "neural", "period4_hold")]
    first = cohort[0]
    one = {**first, "step": 0, "position": (26, 26), "posterior_sha256": "a" * 64}
    result = module.scan([one], cohort, [], first, 1)
    assert len(result["by_stage_regime_arm"]) == 12
    assert result["by_stage_regime_arm"][0]["misses"] == 1
    assert all(row[key] == 0 for row in result["by_stage_regime_arm"][1:]
               for key in ("queries", "hits", "misses", "evictions"))
