#!/usr/bin/env python3
"""Post-hoc arithmetic on saved chess decisions; no model or engine imports.

Usage: python runs/chess-compute-v1/confidence-diagnostic/analyze.py
Outputs are exclusive-create, preserving previous analysis receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
from pathlib import Path

SEEDS = (17, 29, 43)
DEPTHS = (2, 4, 8, 16)
SPLITS = ("dev", "shift")
PANEL_SIZE = 256


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def describe(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def metrics(rows: list[dict]) -> dict:
    result = {"decisions": len(rows)}
    result["teacher_agreement"] = sum(row["correct"] for row in rows) / len(rows)
    for label, subset in (
        ("all", rows),
        ("teacher_matches", [row for row in rows if row["correct"]]),
        ("teacher_mismatches", [row for row in rows if not row["correct"]]),
    ):
        result[label] = {
            "count": len(subset),
            "max_legal_softmax_probability": describe(
                [row["policy_max_probability"] for row in subset]
            ),
            "policy_entropy_nats": describe([row["policy_entropy"] for row in subset]),
            "confidence_at_least_0_9_count": sum(
                row["policy_max_probability"] >= 0.9 for row in subset
            ),
        }
    confident = [row for row in rows if row["policy_max_probability"] >= 0.9]
    result["confidence_at_least_0_9_teacher_agreement"] = (
        sum(row["correct"] for row in confident) / len(confident) if confident else None
    )
    return result


def paired(base: list[dict], new: list[dict]) -> dict:
    require(len(base) == len(new), "Paired row count differs")
    buckets: dict[str, list[tuple[dict, dict]]] = {
        "corrected": [],
        "broken": [],
        "unchanged_correct": [],
        "unchanged_wrong": [],
    }
    for earlier, later in zip(base, new, strict=True):
        require(
            (earlier["id"], earlier["panel_index"]) == (later["id"], later["panel_index"]),
            "Paired position identities differ",
        )
        label = {
            (False, True): "corrected",
            (True, False): "broken",
            (True, True): "unchanged_correct",
            (False, False): "unchanged_wrong",
        }[(earlier["correct"], later["correct"])]
        buckets[label].append((earlier, later))
    result = {}
    for label, pairs in buckets.items():
        result[label] = {
            "count": len(pairs),
            "depth4_max_probability": describe([a["policy_max_probability"] for a, _ in pairs]),
            "new_depth_max_probability": describe([b["policy_max_probability"] for _, b in pairs]),
            "depth4_entropy_nats": describe([a["policy_entropy"] for a, _ in pairs]),
            "new_depth_entropy_nats": describe([b["policy_entropy"] for _, b in pairs]),
        }
    result["net_agreement_change"] = (
        len(buckets["corrected"]) - len(buckets["broken"])
    ) / len(base)
    return result


def analyze(study: Path) -> tuple[dict, dict]:
    source_paths = {
        "decisions": study / "execution/decisions.jsonl",
        "execution_receipt": study / "execution/completed.json",
        "report_summary": study / "report/summary.json",
        "report_receipt": study / "report/completed.json",
    }
    hashes = {name: sha256(path) for name, path in source_paths.items()}
    execution = read_json(source_paths["execution_receipt"])
    report = read_json(source_paths["report_summary"])
    receipt = read_json(source_paths["report_receipt"])
    require(execution["status"] == report["status"] == receipt["status"] == "completed", "Incomplete source")
    require(execution["files"]["decisions.jsonl"] == hashes["decisions"], "Decision hash mismatch")
    require(receipt["summary_sha256"] == hashes["report_summary"], "Report summary hash mismatch")
    require(
        report["execution_receipt_sha256"] == receipt["execution_receipt_sha256"] == hashes["execution_receipt"],
        "Execution receipt binding mismatch",
    )
    require(
        execution["plan_sha256"] == report["plan_sha256"] == receipt["plan_sha256"],
        "Plan identity mismatch",
    )
    wanted = {
        f"predict-{seed}-policy_d{depth}": (seed, depth)
        for seed in SEEDS for depth in DEPTHS
    }
    panels: dict[tuple[int, int, str], dict[int, dict]] = {
        (seed, depth, split): {} for seed in SEEDS for depth in DEPTHS for split in SPLITS
    }
    line_count = 0
    with source_paths["decisions"].open() as stream:
        for line in stream:
            row = json.loads(line)
            line_count += 1
            if row["configuration"] not in wanted:
                continue
            seed, depth = wanted[row["configuration"]]
            key = (seed, depth, row["split"])
            require(key in panels, "Unexpected selected split")
            index = row["panel_index"]
            require(type(index) is int and 0 <= index < PANEL_SIZE, "Invalid panel index")
            require(index not in panels[key], "Duplicate selected decision")
            require(type(row["correct"]) is bool, "Correctness must be Boolean")
            probability, entropy = row["policy_max_probability"], row["policy_entropy"]
            require(math.isfinite(probability) and 0 <= probability <= 1, "Invalid confidence")
            require(
                math.isfinite(entropy) and 0 <= entropy <= math.log(row["compute"]["legal_count"]) + 1e-5,
                "Invalid entropy",
            )
            require(row["compute"]["depth"] == depth, "Depth metadata mismatch")
            panels[key][index] = row
    require(line_count == execution["decisions"], "Complete decision count mismatch")
    ordered = {}
    for key, panel in panels.items():
        require(set(panel) == set(range(PANEL_SIZE)), f"Incomplete selected panel: {key}")
        ordered[key] = [panel[index] for index in range(PANEL_SIZE)]
    for split in SPLITS:
        reference_ids = [(row["id"], row["panel_index"]) for row in ordered[(SEEDS[0], 4, split)]]
        require(len({item[0] for item in reference_ids}) == PANEL_SIZE, "Duplicate position identity")
        for seed in SEEDS:
            for depth in DEPTHS:
                require(
                    [(row["id"], row["panel_index"]) for row in ordered[(seed, depth, split)]] == reference_ids,
                    "Position identities not shared across seeds/depths",
                )
    reported = {}
    for row in report["configurations"]:
        key = (row["configuration"], row["split"])
        require(key not in reported, "Duplicate report row")
        reported[key] = row
    per_seed, pooled, transitions = [], [], []
    for split in SPLITS:
        for depth in DEPTHS:
            all_rows = []
            all_base = []
            for seed in SEEDS:
                rows = ordered[(seed, depth, split)]
                base = ordered[(seed, 4, split)]
                measured = metrics(rows)
                transition = paired(base, rows)
                saved = reported[(f"predict-{seed}-policy_d{depth}", split)]
                require(saved["examples"] == PANEL_SIZE, "Report panel count mismatch")
                require(math.isclose(saved["agreement"], measured["teacher_agreement"], abs_tol=1e-12), "Agreement mismatch")
                for name in ("corrected", "broken"):
                    require(saved["paired_vs_depth4"][name] == transition[name]["count"], "Paired transition mismatch")
                per_seed.append({"split": split, "seed": seed, "depth": depth, **measured})
                all_rows.extend(rows)
                all_base.extend(base)
            pooled.append({"split": split, "depth": depth, **metrics(all_rows)})
            transitions.append({"split": split, "new_depth": depth, **paired(all_base, all_rows)})
    result = {
        "status": "completed",
        "analysis_kind": "posthoc_saved_decision_confidence_diagnostic",
        "plan_sha256": execution["plan_sha256"],
        "selection": {"mode": "predict", "seeds": list(SEEDS), "depths": list(DEPTHS), "splits": list(SPLITS)},
        "source_decisions": line_count,
        "selected_decisions": len(ordered) * PANEL_SIZE,
        "positions_per_split": PANEL_SIZE,
        "pooled_events_per_split_and_depth": len(SEEDS) * PANEL_SIZE,
        "per_seed": per_seed,
        "pooled": pooled,
        "paired_vs_depth4": transitions,
        "scope": [
            "Previously scored development panels; this is a post-hoc descriptive analysis, not a new test.",
            "Each split has 256 positions repeated across three training seeds; 768 events are not 768 independent positions or games.",
            "Wrong and corrected refer only to disagreement/agreement with the recorded finite-budget teacher move, not verified chess optimality.",
            "Confidence is uncalibrated maximum softmax probability over legal moves; entropy is in nats.",
            "The models were trained at four recurrent steps. Depth eight and sixteen are untrained computation extrapolations.",
            "No model inference, engine analysis, training, or parameter changes were performed.",
            "Saved decisions establish confidence saturation and accuracy changes, not a causal latent-state instability mechanism.",
        ],
    }
    bindings = {
        name: {"path": str(path.resolve()), "sha256": hashes[name], "bytes": path.stat().st_size}
        for name, path in source_paths.items()
    }
    return result, bindings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    require(not (args.out / "summary.json").exists(), "Summary exists; use a fresh output directory")
    require(not (args.out / "receipt.json").exists(), "Receipt exists; use a fresh output directory")
    result, sources = analyze(args.study)
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "summary.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    receipt = {
        "status": "completed",
        "analysis_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "python_version": platform.python_version(),
        "sources": sources,
        "output": {"path": str(output.resolve()), "sha256": sha256(output), "bytes": output.stat().st_size},
        "source_receipt_bindings_verified": True,
        "selected_panel_completeness_verified": True,
        "agreement_and_paired_counts_match_completed_report": True,
        "model_or_engine_calls": 0,
    }
    with (args.out / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "completed", "summary": str(output), "selected_decisions": result["selected_decisions"]}))


if __name__ == "__main__":
    main()
