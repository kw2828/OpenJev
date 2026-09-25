"""One descriptive pass over authenticated, already evaluated robot outputs."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STUDY = ROOT / "output/robot-coupling-study-v1"
AUDIT = ROOT / "output/robot-coupling-audit-v1/audit.json"
REGISTRATION_SHA = "7bf116569bc141bf949b04b5bd79d465b2cd0acda098ffcf63cc26553725c4de"
BINS = ((1, 8), (9, 32), (33, 64), (65, 128))
TRACE_BLOCK = 128
FIXED = ("linear_frozen", "quadratic_frozen", "persistence", "velocity")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    require(path.is_file() and not path.is_symlink(), f"regular file: {path}")
    b = path.read_bytes()
    return {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def read(path):
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def close(a, b):
    """The original auditor's scalar tolerance, with exact structure/identities."""
    if isinstance(a, dict):
        require(isinstance(b, dict) and set(a) == set(b), "scalar object roster")
        for key in a:
            close(a[key], b[key])
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), "scalar list roster")
        for left, right in zip(a, b, strict=True):
            close(left, right)
    elif isinstance(a, float):
        require(type(b) in (float, int) and math.isfinite(a) and math.isfinite(b)
                and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), "audited scalar join")
    else:
        require(type(a) is type(b) and a == b, "exact scalar join")


def authenticate():
    # Current source pins are checked before any saved scientific values.
    receipt = read(STUDY / "receipt.json")
    require(receipt["status"] == "PASS" and receipt["registration_sha256"] == REGISTRATION_SHA, "study closure")
    require(receipt["sources_before"] == receipt["sources_after"], "source closure")
    inputs = {}
    for name, expected in receipt["sources_before"].items():
        require(pin(ROOT / name) == expected, f"current source {name}")
        require(pin(STUDY / "sources" / name) == expected, f"source snapshot {name}")
        inputs[str(ROOT / name)] = expected
    require(receipt["confirmation_decodes"] == receipt["official_test_decodes"] == 0, "closed partitions")
    manifest = read(STUDY / "manifest.json")["files"]
    actual = {str(p.relative_to(STUDY)) for p in STUDY.rglob("*") if p.is_file()}
    require(actual == set(manifest) | {"receipt.json", "manifest.json"}, "study roster")
    for name, expected in manifest.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "safe name")
        require(pin(STUDY / name) == expected, f"saved bytes {name}")
        inputs[str(STUDY / name)] = expected
    audit = read(AUDIT)
    require(audit["status"] == "PASS" and audit["agreement"] is True, "independent audit")
    require(audit["registration_sha256"] == REGISTRATION_SHA and audit["study"] == str(STUDY), "audit identity")
    require(audit["source_pins"] == receipt["sources_before"], "audit source pins")
    require(read(AUDIT.parent / "manifest.json")["files"]["audit.json"] == pin(AUDIT), "audit manifest")
    for key, item in audit["inputs"].items():
        for entry in item if key == "qualification_logs" else [item]:
            p = Path(entry["path"])
            require(pin(p) == {k: entry[k] for k in ("sha256", "bytes")}, f"audit input {p}")
            inputs[str(p)] = pin(p)
    require(pin(Path(audit["auditor"]["path"])) == {k: audit["auditor"][k] for k in ("sha256", "bytes")}, "auditor source")
    for p in (AUDIT, AUDIT.parent / "manifest.json", STUDY / "receipt.json", STUDY / "manifest.json"):
        inputs[str(p)] = pin(p)
    return inputs, audit


def error_summary(delta, scales):
    count = int(delta.size)
    finite = int(np.isfinite(delta).sum())
    if finite != count:
        return {"status": "NONFINITE", "scalars": count, "finite_scalars": finite,
                "standardized_rmse": None, "physical_rmse_deg": None}
    with np.errstate(over="ignore", invalid="ignore"):
        mse = float(np.mean(delta * delta))
        physical_mse = float(np.mean((delta * scales) ** 2))
    valid = math.isfinite(mse) and math.isfinite(physical_mse)
    return {"status": "FINITE" if valid else "OVERFLOW", "scalars": count,
            "finite_scalars": finite, "standardized_rmse": math.sqrt(mse) if valid else None,
            "physical_rmse_deg": math.sqrt(physical_mse) if valid else None}


def main():
    require(not (HERE / "diagnostic.json").exists() and not (HERE / "receipt.json").exists(), "one original analysis only")
    started = time.perf_counter()
    inputs, audit = authenticate()
    results = read(STUDY / "results.json")
    fits = read(STUDY / "fits.json")
    close(audit["results"], {k: results[k] for k in ("rows", "selection", "result")})
    cfg, selection = results["config"], results["selection"]
    require(cfg["horizons"] == [64, 128] and len(cfg["partitions"]["dev"]) == 2 and len(fits) == 30, "study scope")
    reads = []

    def array(name, key):
        with np.load(STUDY / name, allow_pickle=False) as archive:
            result = archive[key].copy()
        reads.append({"file": name, "array": key})
        return result

    scales = array("normalizers.npz", "q_std")
    require(scales.shape == (6,) and np.isfinite(scales).all() and (scales > 0).all(), "six scales")
    selected = [fit for fit in fits if fit["learning_rate"] == selection["selected_rates"][fit["arm"]]]
    references = [*FIXED, selection["selected_direct"]]
    bin_rows, joint_rows, unselected = [], [], []
    for arm, rate in selection["selected_rates"].items():
        if rate is None:
            unselected.append({"arm": arm, "status": "INELIGIBLE", "reason": "no eligible original rate; original failures preserved"})
    model_roster = [{"arm": fit["arm"], "seed": fit["seed"], "learning_rate": fit["learning_rate"], "key": fit["key"]} for fit in selected]
    model_roster += [{"arm": arm, "seed": None, "learning_rate": None, "key": arm} for arm in references if arm is not None]
    for recording in cfg["partitions"]["dev"]:
        target = array("dev-windows-" + recording + ".npz", "target")
        require(target.ndim == 3 and target.shape[1:] == (128, 6) and np.isfinite(target).all(), "complete DEV target")
        for model in model_roster:
            name = "prediction-" + recording + "-" + model["key"] + ".npz"
            require(name in read(STUDY / "manifest.json")["files"], "saved prediction required")
            pred = array(name, "prediction")
            require(pred.shape == target.shape, "prediction shape")
            original_rows = [r for r in results["rows"] if r["recording"] == recording
                             and r["arm"] == model["arm"] and r["seed"] == model["seed"]
                             and r["learning_rate"] == model["learning_rate"] and r["horizon"] == 128]
            require(len(original_rows) == 1, "unique original outcome")
            for first, last in BINS:
                delta = pred[:, first-1:last, :] - target[:, first-1:last, :]
                common = {**model, "recording": recording, "first_step": first, "last_step": last,
                          "windows": len(target), "original_h128_status": original_rows[0]["status"]}
                bin_rows.append({**common, **error_summary(delta, scales)})
                for joint in range(6):
                    joint_rows.append({**common, "joint": joint + 1, **error_summary(delta[:, :, joint], scales[joint])})
    trends = []
    for fit in fits:
        trace = read(STUDY / fit["key"] / "trace.json")
        require(len(trace) == fit["fit"]["completed_updates"], "trace receipt length")
        require([r["update"] for r in trace] == list(range(1, len(trace)+1)), "trace update roster")
        blocks = []
        for start in range(0, len(trace), TRACE_BLOCK):
            block = trace[start:start+TRACE_BLOCK]
            losses = [r["loss"] for r in block]
            gradients = [r["gradient_norm_before_clip"] for r in block]
            require(all(math.isfinite(v) for v in losses + gradients), "recorded finite trace")
            blocks.append({"first_update": start + 1, "last_update": start + len(block),
                           "mean_loss": float(np.mean(losses)), "median_loss": float(np.median(losses)),
                           "mean_gradient_norm_before_clip": float(np.mean(gradients)),
                           "fraction_gradient_norm_gt_1": sum(v > 1 for v in gradients) / len(block)})
        trends.append({"key": fit["key"], "arm": fit["arm"], "seed": fit["seed"],
                       "learning_rate": fit["learning_rate"], "selected": fit in selected,
                       "status": fit["fit"]["status"], "error": fit["fit"]["error"],
                       "completed_updates": len(trace), "blocks": blocks,
                       "last_over_first_block_mean_loss": blocks[-1]["mean_loss"] / blocks[0]["mean_loss"] if blocks and blocks[0]["mean_loss"] > 0 else None})
    means = []
    for recording in cfg["partitions"]["dev"]:
        for arm in cfg["arms"] + references:
            if arm is None:
                continue
            for first, last in BINS:
                group = [r for r in bin_rows if r["recording"] == recording and r["arm"] == arm and r["first_step"] == first]
                expected = 3 if arm in cfg["arms"] else 1
                complete = len(group) == expected and all(r["status"] == "FINITE" for r in group)
                means.append({"recording": recording, "arm": arm, "first_step": first, "last_step": last,
                              "fits": len(group), "status": "FINITE" if complete else "UNAVAILABLE",
                              "mean_standardized_rmse": float(np.mean([r["standardized_rmse"] for r in group])) if complete else None})
    output = {"version": "robot-transition-diagnostic-v1", "bins_inclusive": BINS,
              "training_block_updates": TRACE_BLOCK, "original_result": results["result"],
              "original_selection": selection, "unselected_families": unselected,
              "bin_rows": bin_rows, "joint_rows": joint_rows, "family_means": means, "training_trends": trends,
              "array_reads": reads, "counts": {"npz_opens": len(reads), "array_decodes": len(reads),
                  "model_calls": 0, "training_calls": 0, "raw_mat_reads": 0, "confirmation_reads": 0, "official_test_reads": 0},
              "scope": "Retrospective descriptive decomposition of already evaluated outputs, not new quality evaluation, selection or gate rescue.",
              "limits": ["Early-bin error cannot isolate initialization from one-step model error.",
                         "Minibatches change across updates; trace block means are optimization diagnostics, not a fixed training validation curve.",
                         "Direct ridge reads all future measured torques jointly; autoregressive models use stepwise inputs. The reference is not causally matched.",
                         "All six joints and both DEV recordings retained. Nonfinite cells are unavailable, never filtered to favorable finite subsets.",
                         "Family means average saved-fit RMSE values, not ensemble predictions."]}
    after, _ = authenticate()
    require(after == inputs, "closed evidence changed")
    with (HERE / "diagnostic.json").open("x") as f:
        json.dump(output, f, indent=2, sort_keys=True, allow_nan=False)
    receipt = {"status": "PASS", "seconds": time.perf_counter()-started, "script": pin(Path(__file__)),
               "inputs": inputs, "output": pin(HERE / "diagnostic.json"), "counts": output["counts"]}
    with (HERE / "receipt.json").open("x") as f:
        json.dump(receipt, f, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({"status": "PASS", "seconds": receipt["seconds"], "counts": output["counts"],
                      "bin_rows": len(bin_rows), "joint_rows": len(joint_rows), "traces": len(trends)}))


if __name__ == "__main__":
    main()
