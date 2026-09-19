"""Saved first-fit normalization failure audit. No model or efficacy computation."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
REPLAY = ROOT / "runs/dialogue-evidence-qualification-v1/replay-01"
STUDY = ROOT / "runs/dialogue-copy-v1/study-01"
PLAN = ROOT / "output/dialogue-evidence-qualification-v1/protocol/plan.json"
OLD_PLAN_SHA = "609a0758f2a4ae4722052bb27740c74b3d1bdac396d8134eb39f5118f1429a2b"
OLD_COMPLETED_SHA = "8317abad5ef8235f31f5381591944ad152c42fddb98743fa64fd1f5cd8c3f300"
QUALIFIER_SHA = "6cdf0ba0d1d01ad7af048a84350eff149db2fe958ef20c2333620400e0680c71"
MODEL_SHA = "182c71a944c6d787533bf129bca1631ff3b366b8559e3db7567e7fd9b19a8717"
FIELDS = {"probabilities", "labels", "choice", "bin", "unseen", "dialogue", "time", "query"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def arrays(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


def audit():
    require(not (OUT / "summary.json").exists() and not (OUT / "receipt.json").exists(), "Exclusive audit outputs")
    start = time.perf_counter()
    bound = {}

    def bind(path, expected=None):
        actual = sha(path)
        require(not path.is_symlink() and (expected is None or actual == expected), "Changed input: " + str(path))
        bound[str(path.relative_to(ROOT))] = actual

    bind(STUDY / "plan.json", OLD_PLAN_SHA)
    bind(STUDY / "completed.json", OLD_COMPLETED_SHA)
    bind(REPLAY / "failed.json")
    bind(REPLAY / "started.json")
    failed, started = read(REPLAY / "failed.json"), read(REPLAY / "started.json")
    bind(PLAN, started["plan_sha256"])
    plan = read(PLAN)
    require(plan["source_sha256"] == started["source_sha256"] and plan["runtime"] == started["runtime"],
            "Started/prospective identity")
    require(plan["old_plan_sha256"] == OLD_PLAN_SHA and plan["old_completed_sha256"] == OLD_COMPLETED_SHA,
            "Original identity")
    require(plan["source_sha256"]["scripts/qualify_dialogue_evidence.py"] == QUALIFIER_SHA
            and plan["source_sha256"]["src/openjev/research/dialogue_copy_memory.py"] == MODEL_SHA,
            "Reviewed source identity")
    for name, expected in plan["source_sha256"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe source path")
        bind(ROOT / name, expected)
    require(failed["status"] == "failed" and failed["error"] == "Invalid saved factors"
            and failed["completed_fits"] == [] and not (REPLAY / "completed.json").exists(), "Failure state")
    work = failed["active_progress"]
    require(work["fit"] == "scalar-4101" and work["matched_rows"] == work["returned_rows"] == 62329
            and work["model_forward_calls"] == work["model_forward_returned"] == 74,
            "First-fit-only progress")
    require({p.name for p in (REPLAY / "fits").iterdir()} == {"scalar-4101"}, "Foreign replay fit")
    folder = REPLAY / "fits/scalar-4101"
    require({p.name for p in folder.iterdir()} == {"operators.npz"}, "No completed operator diagnostic expected")
    bind(folder / "operators.npz")
    original = read(STUDY / "completed.json")
    fit = next(v for v in original["fits"] if v["method"] == "scalar" and v["seed"] == 4101)
    old_folder = STUDY / "fits/scalar-4101"
    require(read(old_folder / "completed.json") == fit, "Original fit record")
    bind(old_folder / "completed.json")
    bind(old_folder / "dev-predictions.npz", fit["predictions_sha256"])
    a, p = arrays(folder / "operators.npz"), arrays(old_folder / "dev-predictions.npz")
    require(set(p) == FIELDS and set(a) == FIELDS | {
        "old_b", "writer", "departure_mass", "result", "candidate_count", "user_unique", "system_unique", "boolean_slot"},
        "Saved factor schema")
    parity = {}
    for key in FIELDS:
        require(a[key].shape == p[key].shape and a[key].dtype == p[key].dtype
                and np.array_equal(a[key], p[key]), "Original prediction/metadata parity: " + key)
        parity[key] = {"array_equal": True, "bitwise_equal": a[key].tobytes() == p[key].tobytes()}
        require(parity[key]["bitwise_equal"], "Bitwise parity: " + key)
    require(a["probabilities"].dtype == np.float32 and a["old_b"].dtype == a["writer"].dtype == np.float32,
            "Factor dtype")
    require(np.array_equal(a["result"], a["probabilities"]), "Captured normalized output differs")
    b, w, m = a["old_b"].astype(np.float64), a["writer"].astype(np.float64), a["departure_mass"].astype(np.float64)
    require(b.shape == w.shape == (62329, 12) and m.shape == (62329,) and np.isfinite(b).all()
            and np.isfinite(w).all() and np.isfinite(m).all(), "Finite factor schema")
    n = a["candidate_count"]
    mask = np.arange(12)[None, :] < n[:, None]
    require((b[~mask] == 0).all() and (w[~mask] == 0).all() and (b >= 0).all() and (w >= 0).all(), "Masked factors")
    s, ws = b.sum(1), w.sum(1)
    tol = plan["factor_roundoff"]
    require(tol == 2e-6, "Qualification tolerance")
    bp = np.abs(s - 1) > tol
    wp = np.abs(ws - 1) > tol
    mp = (m < 0) | (m > 1 + tol)
    by_time = []
    for t in sorted(np.unique(a["time"]).tolist()):
        selected = a["time"] == t
        by_time.append({"public_user_time_index": t, "rows": int(selected.sum()),
                       "old_b_sum_min": float(s[selected].min()), "old_b_sum_max": float(s[selected].max()),
                       "old_b_sum_mean_absolute_error": float(np.abs(s[selected] - 1).mean()),
                       "old_b_normalization_violations": int(bp[selected].sum()),
                       "old_b_sum_below_one_minus_tolerance": int((s[selected] < 1 - tol).sum()),
                       "old_b_sum_above_one_plus_tolerance": int((s[selected] > 1 + tol).sum()),
                       "writer_normalization_violations": int(wp[selected].sum()),
                       "mass_min": float(m[selected].min()), "mass_max": float(m[selected].max()),
                       "mass_outside_tolerance": int(mp[selected].sum())})
    # Adjacent scored same-query rows expose the next old state without another forward call.
    previous, left, right = {}, [], []
    for i, (dialog, query, t) in enumerate(zip(a["dialogue"], a["query"], a["time"], strict=True)):
        key = str(dialog), int(query)
        if key in previous:
            j = previous[key]
            require(t > a["time"][j], "Query chronology")
            if t == a["time"][j] + 1:
                left.append(j)
                right.append(i)
        previous[key] = i
    left, right = np.asarray(left), np.asarray(right)
    projected_sum = s[left] * (s[left] - m[left]) + m[left]
    writer_adjusted = s[left] * (s[left] - m[left]) + m[left] * ws[left]
    residual, adjusted_residual = s[right] - projected_sum, s[right] - writer_adjusted
    identity_error = projected_sum - 1 - (s[left] - 1) * (s[left] + 1 - m[left])
    require(np.max(np.abs(identity_error)) < 1e-12, "Scalar identity arithmetic")
    normal_output_sum = a["probabilities"].astype(np.float64).sum(1)
    result = {
        "status": "completed", "audited_execution_status": "failed", "study": "dialogue-evidence-qualification-v1",
        "fit": "scalar-4101", "scored_rows": 62329, "source_sha256": sha(__file__),
        "input_sha256": bound, "parity": parity,
        "normalization": {"tolerance": tol, "old_b_sum_min": float(s.min()), "old_b_sum_max": float(s.max()),
            "old_b_normalization_violations": int(bp.sum()), "old_b_below_one_minus_tolerance": int((s < 1 - tol).sum()),
            "old_b_above_one_plus_tolerance": int((s > 1 + tol).sum()),
            "old_b_component_max": float(b.max()), "old_b_rows_with_component_above_one": int((b.max(1) > 1).sum()),
            "writer_sum_min": float(ws.min()), "writer_sum_max": float(ws.max()),
            "writer_normalization_violations": int(wp.sum()), "mass_min": float(m.min()), "mass_max": float(m.max()),
            "mass_above_one": int((m > 1).sum()), "mass_outside_tolerance": int(mp.sum()),
            "reported_output_sum_max_absolute_error": float(np.abs(normal_output_sum - 1).max()),
            "rows_with_any_failed_factor_check": int((bp | wp | mp).sum()), "by_public_time": by_time},
        "scalar_amplification": {"adjacent_same_query_pairs": len(left),
            "derived_exact_real_arithmetic": "S_next = S*(S-m)+m when writer sums to1; delta_next = delta*(S+1-m).",
            "derivation": "The scalar retained multiplier is sum(b*(1-r))=S-m, not1-m for an off-normalized state.",
            "actual_next_sum_minus_formula_max_absolute": float(np.abs(residual).max()),
            "actual_next_sum_minus_formula_mean_absolute": float(np.abs(residual).mean()),
            "actual_next_sum_minus_formula_rmse": float(np.sqrt(np.mean(residual**2))),
            "formula_with_saved_writer_sum_max_absolute": float(np.abs(adjusted_residual).max()),
            "no_amplification_next_minus_current_rmse": float(np.sqrt(np.mean((s[right] - s[left])**2))),
            "saved_formula_amplification_factor_min": float((s[left] + 1 - m[left]).min()),
            "saved_formula_amplification_factor_max": float((s[left] + 1 - m[left]).max()),
            "pairs_amplification_factor_above_one": int((s[left] + 1 - m[left] > 1).sum()),
            "scope": "Independent arithmetic on adjacent saved factor rows; float32 logsumexp/logaddexp/exp rounding prevents exact algebraic equality."},
        "failure": {"recorded_wall_seconds": failed["wall_seconds"], "completed_fit_count": 0,
                    "work": work, "original_error": failed["error"]},
        "interpretation": [
            "Replay outputs and metadata exactly reproduce the existing scalar-4101 predictions; normalized output softmax masks internal mass drift.",
            "The captured b values are materially non-normalized. The probability-factor qualification correctly failed before writer-gold or intervention efficacy diagnostics.",
            "The scalar transition can amplify small normalization error. Its raw belief and entropy features also consume that drifting state.",
            "This establishes a defect in the intended normalized-state invariant for this one existing scalar fit. Other fits were not replayed or audited here.",
            "Selective algebra uses S-m+m=S in ideal arithmetic and lacks the scalar quadratic factor, but its actual finite-precision normalization and prediction impact remain unmeasured.",
            "Existing reported prediction metrics remain reproducible numerical outcomes. They do not validate the intended probability-preserving scalar/selective mechanism comparison.",
            "No model/encoder/optimizer/environment calls, counterfactual efficacy partitions, added fits, or changed historical files."],
    }
    for name, expected in bound.items():
        require(sha(ROOT / name) == expected, "Input changed during audit")
    write(OUT / "summary.json", result)
    write(OUT / "receipt.json", {"status": "completed", "scope": "First scalar-4101 saved-factor failure only",
        "source_sha256": sha(__file__), "input_sha256": bound,
        "files": {name: {"sha256": sha(OUT / name), "bytes": (OUT / name).stat().st_size}
                  for name in ("analyze.py", "summary.json")},
        "input_pinning": "Original study and reviewed code externally pinned; failed replay artifact digests observed and recorded by this read-only audit.",
        "wall_seconds": time.perf_counter() - start, "new_model_calls": 0, "new_optimizer_calls": 0,
        "new_encoder_calls": 0, "counterfactual_efficacy_computed": False})
    print(json.dumps({"status": "completed", "old_b_violations": int(bp.sum()), "mass_violations": int(mp.sum()),
                      "formula_max_error": float(np.abs(residual).max()), "summary_sha256": sha(OUT / "summary.json"),
                      "receipt_sha256": sha(OUT / "receipt.json")}))


if __name__ == "__main__":
    audit()
