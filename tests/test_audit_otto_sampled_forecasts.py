"""Independent fabricated sampled-window reconstruction, no scientific calls."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_test_sampled_audit", ROOT / "scripts/audit_otto_sampled_forecasts.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture(tmp_path):
    obj = audit.Audit(SimpleNamespace(output=tmp_path))
    obj.np = np
    obj.collection = tmp_path
    obj.run = tmp_path
    obj.check = lambda: None
    lengths = [37] + [1] * 53
    offsets = np.asarray([0, *np.cumsum(lengths)], np.int64)
    n = int(offsets[-1])
    x = np.zeros((n, 31), np.float32)
    correction = np.zeros(n, np.bool_)
    records, episodes = [], []
    for index, (identity, length) in enumerate(zip(audit.cohort()[:54], lengths, strict=True)):
        lo, hi = int(offsets[index]), int(offsets[index + 1])
        steps = np.arange(length)
        x[lo:hi, 15] = steps / 2188
        x[lo:hi, 16] = steps % 4 / 2188
        x[lo:hi, 17] = 1
        correction[lo:hi] = steps % 4 == 0
        w, seed = (length + 3) // 4, audit.SELECTION_START + index
        k = min(w, 8)
        starts = sorted(int(v) * 4 for v in np.random.Generator(np.random.PCG64(seed)).choice(w, k, replace=False))
        records.append({"episode_id": identity["episode_id"], "version": "otto-sampled-forecast-data-v1",
            "length": length, "seed": seed, "period": 4, "population_windows": w, "selected_windows": k,
            "start_offsets": starts, "inclusion_numerator": k, "inclusion_denominator": w,
            "inclusion_probability": k / w, "algorithm": "numpy.Generator(PCG64(seed)).choice(W,size=k,replace=False); sorted"})
        episodes.append({**identity, "start_row": lo, "end_row": hi, "steps": length, "teacher_calls": length})
    flat = {"features": x, "raw_q": np.arange(n * 4, dtype=np.float32).reshape(n, 4),
            "legal": np.ones((n, 4), np.bool_), "actions": np.zeros(n, np.int64),
            "correction": correction, "episode_offsets": offsets, "label_mask": np.ones(n, np.bool_)}
    obj.episodes = episodes
    obj.arrays = lambda path: flat
    obj.read = lambda path: records
    return obj, flat, records


def test_independent_selected_bytes_full_denominator_and_zero_support(tmp_path):
    obj, flat, records = fixture(tmp_path)
    windows, meta, _ = obj.reconstruct("train")
    assert windows["episode_nonquery_counts"].tolist() == [27] + [0] * 53
    assert meta["counts"]["windows"] == 61 and meta["counts"]["population_windows"] == 63
    assert meta["counts"]["full_rows"] == 90
    assert meta["sampling"]["episode_denominator"] == 54
    for index, start in enumerate(records[0]["start_offsets"]):
        size = min(4, 37 - start)
        assert windows["targets"][index, :size].tobytes() == flat["raw_q"][start:start + size].tobytes()
        assert np.all(windows["nonquery_weights"][index, 1:size] == (10 / 8) / (54 * 27))
        assert windows["nonquery_weights"][index, 0] == 0
    assert not meta["sampling"]["renormalized"]


@pytest.mark.parametrize("defect", ["selection", "label", "chronology"])
def test_corrupt_selection_missing_label_or_full_history_is_rejected(tmp_path, defect):
    obj, flat, records = fixture(tmp_path)
    if defect == "selection":
        records[0]["start_offsets"][0] += 4
    elif defect == "label":
        index = records[0]["start_offsets"][0]
        flat["label_mask"][index] = False
        flat["raw_q"][index] = 0
        obj.episodes[0]["teacher_calls"] -= 1
    else:
        flat["features"][30, 15] = 0
    with pytest.raises(ValueError):
        obj.reconstruct("train")


def test_persisted_ipw_cannot_be_renormalized(tmp_path):
    obj, _, _ = fixture(tmp_path)
    windows, meta, _ = obj.reconstruct("train")
    saved = {key: value.copy() for key, value in windows.items()}
    saved["nonquery_weights"] /= saved["nonquery_weights"].sum()
    obj.arrays = lambda path: saved
    obj.read = lambda path: copy.deepcopy(meta)
    with pytest.raises(ValueError, match="window bytes nonquery_weights"):
        obj.saved_windows("training", windows, meta)


def test_fresh_cohort_order_and_all_validation_rules_remain_fixed():
    rows = audit.cohort()
    assert len(rows) == len({r["episode_id"] for r in rows}) == 90
    assert [r["episode_index"] for r in rows] == list(range(90))
    assert sum(r["stage"] == "train" for r in rows) == 54
    assert {r["seed"] for r in rows if r["stage"] == "valid"} == set(range(23300001, 23300007)) | set(range(23400001, 23400007))
    assert [r["arm"] for r in rows[:6]] == ["analytic", "neural", "period4_hold", "neural", "period4_hold", "analytic"]
    assert "training" not in audit.Audit.__dict__ and "metrics" not in audit.Audit.__dict__
    assert audit.base.SEEDS == (235001, 235002, 235003)
