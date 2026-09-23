"""Fabricated selection/provenance boundaries; no saved studies or model calls."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research import otto_sampled_forecast_data as sampled

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_test_sampled_train", ROOT / "scripts/train_otto_sampled_forecasts.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture():
    lengths = [37] + [1] * 53
    offsets = np.asarray([0, *np.cumsum(lengths)], np.int64)
    n = int(offsets[-1])
    features = np.zeros((n, 31), np.float32)
    correction = np.zeros(n, np.bool_)
    identities, selections = [], []
    for i, (lo, hi) in enumerate(pairwise(offsets)):
        steps = np.arange(hi - lo)
        features[lo:hi, 15] = steps / 2188
        features[lo:hi, 16] = steps % 4 / 2188
        features[lo:hi, 17] = 1
        correction[lo:hi] = steps % 4 == 0
        identities.append({"stage": "train", "episode_index": i, "episode_id": f"episode-{i}", "regime": "lambda3"})
        selections.append({"episode_id": f"episode-{i}", **sampled.select_windows(int(hi - lo), runner.SELECTION_START + i)})
    flat = {"features": features, "raw_q": np.arange(n * 4, dtype=np.float32).reshape(n, 4),
            "legal": np.ones((n, 4), np.bool_), "actions": np.zeros(n, np.int64),
            "correction": correction, "episode_offsets": offsets, "label_mask": np.ones(n, np.bool_)}
    return flat, identities, selections


def test_only_selected_scores_reach_builder_and_ipw_uses_full_episode():
    flat, identities, selections = fixture()
    seen = {}

    def build(episodes, records, scores):
        assert all(set(e) == {"id", "regime", "split", "features", "legal", "actions"} for e in episodes)
        assert set(scores[0]) == set(selections[0]["start_offsets"])
        seen["scores"] = scores
        return sampled.build_sampled_windows(episodes, records, scores)

    proxy = SimpleNamespace(select_windows=sampled.select_windows, build_sampled_windows=build)
    _, windows = runner.selected_training(np, flat, identities, selections, proxy)
    selected = {step for start in selections[0]["start_offsets"] for step in range(start, min(start + 4, 37))}
    skipped = sorted(set(range(37)) - selected)
    assert skipped and seen["scores"]
    flat["raw_q"][skipped] = -123456.
    _, after = runner.selected_training(np, flat, identities, selections, sampled)
    assert after["targets"].tobytes() == windows["targets"].tobytes()
    assert windows["episode_nonquery_counts"][0] == 27
    assert windows["counts"]["zero_support_episodes"] == 53
    positive = windows["nonquery_weights"][windows["nonquery_mask"]]
    assert np.all(positive == (10 / 8) / (54 * 27))
    assert windows["counts"]["windows"] == 61
    assert not windows["sampling"]["renormalized"]


@pytest.mark.parametrize("defect", ["missing_selected_label", "wrong_seed", "nonzero_unscored"])
def test_selected_labels_and_predeclared_selection_cannot_be_substituted(defect):
    flat, identities, selections = fixture()
    row = selections[0]["start_offsets"][0]
    if defect == "missing_selected_label":
        flat["label_mask"][row] = False
        flat["raw_q"][row] = 0
    elif defect == "wrong_seed":
        selections[0] = {"episode_id": identities[0]["episode_id"], **sampled.select_windows(37, 99)}
    else:
        selected = {s for start in selections[0]["start_offsets"] for s in range(start, min(start + 4, 37))}
        row = next(s for s in range(37) if s not in selected)
        flat["label_mask"][row] = False
    with pytest.raises(ValueError):
        runner.selected_training(np, flat, identities, selections, sampled)


def test_final_checkpoint_barrier_is_inherited_before_any_valid_decode(tmp_path):
    run = runner.Run(SimpleNamespace(output=tmp_path))
    run.valid_allowed = False
    for completed in (0, 11, 12):
        run.receipt["fits_completed"] = completed
        with pytest.raises(ValueError, match="before VALID"):
            run.episodes("valid")
    assert "fit" not in runner.Run.__dict__ and "body" not in runner.Run.__dict__
    assert hashlib.sha256((ROOT / runner.BASE_PATH).read_bytes()).hexdigest() == runner.BASE_PIN


def test_configuration_keeps_all_45_rules_with_new_fixed_fit_seeds():
    group = {"episode_weighted_agreement": .8, "episode_weighted_raw_gap": 2.,
             "by_age": {str(a): {"episode_weighted_raw_gap": 2.} for a in (1, 2, 3)}}
    baseline = {"by_regime": {r: copy.deepcopy(group) for r in ("lambda3", "lambda4")}}
    models = []
    for seed in runner.SEEDS:
        for family in runner.KINDS:
            metric = copy.deepcopy(baseline)
            if family == "residual_gru":
                for value in metric["by_regime"].values():
                    value["episode_weighted_raw_gap"] = 1.
            models.append({"family": family, "seed": seed, "metrics": metric})
    support = {r: {str(a): 6 for a in (1, 2, 3)} for r in ("lambda3", "lambda4")}
    rules = runner.criteria(models, baseline, support)
    assert len(rules) == 45 and all(r["passes"] for r in rules)
    assert any("235003" in r["name"] for r in rules)
    assert not any("22500" in r["name"] for r in rules)
