"""Report completed Mystery Path qualification artifacts without environment calls.

The saved audit replays actions with the same official environment and checks
public parsing against native fields; it is not an alternative simulator. This reporter checks local
completion bindings, every saved episode payload, pairing, and exact arithmetic.
The four priority orders reuse the same 64 layouts and are not independent cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

MODES = ("full", "last16", "last32", "erase_on_failure", "reference")
ROOT = Path(__file__).resolve().parents[1]
BINDING_FILES = {
    "protocol_sha256": "evidence/mystery-path-qualification-v1/protocol.json",
    "bindings_sha256": "evidence/mystery-path-qualification-v1/bindings.json",
    "inputs_sha256": "evidence/mystery-path-qualification-v1/inputs.json",
}
COMPARATORS = ("last16", "last32", "erase_on_failure")
LAYOUTS = 64
ORDERS = 4
EPISODES = LAYOUTS * ORDERS * len(MODES)
MAX_STEPS = 128
TIME_FIELDS = (
    "elapsed_seconds", "environment_seconds", "policy_seconds", "parsing_seconds",
    "update_seconds", "serialization_seconds", "reset_seconds", "evaluator_seconds",
)
BYTE_FIELDS = ("peak_serialized_state_bytes", "peak_retained_python_bytes")
RECORD_FIELDS = {
    "layout_index", "seed", "order_index", "mode", "success", "steps", "falls",
    "npz_path", "npz_sha256", "layout_sha256", *TIME_FIELDS, *BYTE_FIELDS,
}
LABELS = {
    "full": "Full public map", "last16": "Last 16 transitions",
    "last32": "Last 32 transitions", "erase_on_failure": "Erase on failure",
    "reference": "Known-route reference",
}
SCOPE = (
    "Completed saved-output qualification arithmetic; no model, environment, or RNG calls. "
    "This is not an architecture result, a pixels-only benchmark score, or proof of novelty."
)
LIMITS = [
    "The four priority orders reuse 64 layouts; 256 layout/order pairs are not 256 independent layouts.",
    "All public policies share an explicit public-observation parser; the reference alone receives the route.",
    "Whole-episode time includes reset through NPZ writing/hashing, but excludes the outer JSONL append.",
    "Per-step wall time amortizes whole-episode work; it is not isolated policy latency.",
    "State figures are recorded retained-state estimates and serialized payload sizes, not process peak RSS.",
    "A failed gate closes this recipe without shrinking windows, replacing layouts, or changing thresholds.",
    "Prior Reacher and Pendulum failures remain unchanged; qualification cannot establish biological superiority.",
    ("The completed audit replays the same official environment and checks the parser against native fields; "
     "it is not an alternative simulator or independent policy implementation."),
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: object, name: str) -> str:
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"Invalid {name}")
    return value


def _int(value: object, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    require(type(value) is int and value >= minimum, f"Invalid {name}")
    if maximum is not None:
        require(value <= maximum, f"Invalid {name}")
    return value


def _number(value: object, name: str) -> float:
    require(type(value) in (int, float), f"Invalid {name}")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"Invalid {name}") from error
    require(math.isfinite(result) and result >= 0, f"Invalid {name}")
    return result


def _json_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode(text: str) -> object:
    def invalid_constant(value: str) -> None:
        raise ValueError(f"Invalid JSON constant: {value}")
    return json.loads(text, object_pairs_hook=_json_pairs, parse_constant=invalid_constant)


def _read_json(path: Path) -> dict:
    value = _decode(path.read_text())
    require(type(value) is dict, f"Expected JSON object: {path}")
    return value


def _payload_path(record: dict) -> str:
    expected = f"episodes/{record['layout_index']:03d}-{record['mode']}-order{record['order_index']}.npz"
    require(record["npz_path"] == expected, "Unexpected or unsafe NPZ path")
    return expected


def validate_records(records: list[dict]) -> dict[tuple[int, int, str], dict]:
    require(type(records) is list and len(records) == EPISODES, "Expected exactly 1280 episode records")
    indexed = {}
    layout_bindings = {}
    unique_seeds = {}
    for record in records:
        require(type(record) is dict and set(record) == RECORD_FIELDS, "Unexpected episode record fields")
        layout = _int(record["layout_index"], "layout_index", maximum=LAYOUTS - 1)
        order = _int(record["order_index"], "order_index", maximum=ORDERS - 1)
        mode = record["mode"]
        require(type(mode) is str and mode in MODES, "Unknown mode")
        key = (layout, order, mode)
        require(key not in indexed, f"Duplicate episode identity: {key}")
        seed = _int(record["seed"], "seed")
        require(type(record["success"]) is bool, "success must be a bool")
        steps = _int(record["steps"], "steps", minimum=1, maximum=MAX_STEPS)
        _int(record["falls"], "falls", maximum=steps)
        for field in TIME_FIELDS:
            value = _number(record[field], field)
            if field == "elapsed_seconds":
                require(value > 0, "elapsed_seconds must be positive")
        for field in BYTE_FIELDS:
            _int(record[field], field)
        _sha(record["npz_sha256"], "npz_sha256")
        layout_sha = _sha(record["layout_sha256"], "layout_sha256")
        _payload_path(record)
        binding = (seed, layout_sha)
        require(layout not in layout_bindings or layout_bindings[layout] == binding,
                f"Mismatched paired layout/seed: {layout}")
        require(seed not in unique_seeds or unique_seeds[seed] == layout,
                "Different layout indices reuse the same seed")
        layout_bindings[layout] = binding
        unique_seeds[seed] = layout
        indexed[key] = record
    expected = {(layout, order, mode) for layout in range(LAYOUTS) for order in range(ORDERS) for mode in MODES}
    require(set(indexed) == expected, "Incomplete or foreign episode coverage")
    return indexed


def _aggregate(rows: list[dict]) -> dict:
    count = len(rows)
    total_steps = sum(row["steps"] for row in rows)
    timing = {field: math.fsum(float(row[field]) for row in rows) for field in TIME_FIELDS}
    return {
        "episodes": count,
        "successes": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / count,
        "total_steps": total_steps,
        "mean_steps_all_episodes": total_steps / count,
        "total_falls": sum(row["falls"] for row in rows),
        "mean_falls_all_episodes": sum(row["falls"] for row in rows) / count,
        "timing_seconds": timing,
        "amortized_milliseconds_per_environment_step": 1000 * timing["elapsed_seconds"] / total_steps,
        "state_bytes": {
            field: {"max_episode_peak": max(row[field] for row in rows),
                    "mean_episode_peak": math.fsum(row[field] for row in rows) / count}
            for field in BYTE_FIELDS
        },
    }


def _paired(indexed: dict, comparator: str, orders: tuple[int, ...]) -> dict:
    counts = {"both_succeed": 0, "full_only": 0, "comparator_only": 0, "neither_succeeds": 0}
    for layout in range(LAYOUTS):
        for order in orders:
            full = indexed[layout, order, "full"]["success"]
            other = indexed[layout, order, comparator]["success"]
            key = "both_succeed" if full and other else (
                "full_only" if full else ("comparator_only" if other else "neither_succeeds")
            )
            counts[key] += 1
    return {"pairs": LAYOUTS * len(orders), **counts,
            "full_minus_comparator_successes": counts["full_only"] - counts["comparator_only"]}


def evaluate_records(records: list[dict]) -> dict:
    """Pure paired arithmetic. This function does not establish artifact authenticity."""
    indexed = validate_records(records)
    modes = {}
    for mode in MODES:
        rows = [indexed[layout, order, mode] for layout in range(LAYOUTS) for order in range(ORDERS)]
        modes[mode] = {
            **_aggregate(rows),
            "orders": {
                str(order): _aggregate([indexed[layout, order, mode] for layout in range(LAYOUTS)])
                for order in range(ORDERS)
            },
        }
    pairs = {
        mode: {"pooled": _paired(indexed, mode, tuple(range(ORDERS))),
               "orders": {str(order): _paired(indexed, mode, (order,)) for order in range(ORDERS)}}
        for mode in (*COMPARATORS, "reference")
    }
    checks = []

    def check(name: str, actual: int, minimum: int, denominator: int) -> None:
        checks.append({"name": name, "actual_success_count": actual, "minimum_success_count": minimum,
                       "denominator": denominator, "comparison": ">=", "passed": actual >= minimum})

    check("reference_95_percent", modes["reference"]["successes"], 244, 256)
    check("full_80_percent", modes["full"]["successes"], 205, 256)
    for mode in COMPARATORS:
        check(f"full_minus_{mode}_15_percentage_points_pooled",
              pairs[mode]["pooled"]["full_minus_comparator_successes"], 39, 256)
        for order in range(ORDERS):
            check(f"full_minus_{mode}_5_percentage_points_order{order}",
                  pairs[mode]["orders"][str(order)]["full_minus_comparator_successes"], 4, 64)
    passed = all(item["passed"] for item in checks)
    return {
        "scope": SCOPE, "layout_count": LAYOUTS, "priority_orders": ORDERS,
        "distinct_layout_sha256_count": len({row["layout_sha256"] for row in records}),
        "episodes": EPISODES, "maximum_episode_steps": MAX_STEPS,
        "total_steps": sum(row["steps"] for row in records),
        "modes": modes, "paired_success_tables": pairs,
        "gate": {"passed": passed, "checks_passed": sum(item["passed"] for item in checks),
                 "checks_total": len(checks), "checks": checks,
                 "decision": "CONTINUE_MEMORY_QUALIFICATION" if passed else "CLOSE_THIS_RECIPE"},
        "limits": LIMITS,
        "records": [dict(indexed[layout, order, mode]) for layout in range(LAYOUTS)
                    for order in range(ORDERS) for mode in MODES],
    }


def authenticate_attempt(attempt: Path, *, root: Path = ROOT) -> tuple[list[dict], dict]:
    attempt = attempt.resolve(strict=True)
    for directory in (attempt, attempt / "audit"):
        for name in ("failed.json", "late-completion.json", "cleanup-error.json",
                     "completion-before-cleanup-error.json"):
            require(not (directory / name).exists(), f"Retained failure marker: {directory / name}")
    completed_path = attempt / "completed.json"
    audit_path = attempt / "audit/completed.json"
    completed, audit = _read_json(completed_path), _read_json(audit_path)
    require(completed.get("status") == "complete", "Execution is not complete")
    require(audit.get("status") == "passed", "Saved replay audit has not passed")
    for value in (completed, audit):
        require(type(value.get("episodes")) is int and value["episodes"] == EPISODES,
                "Completion/audit episode count mismatch")
    for name in ("protocol_sha256", "bindings_sha256", "inputs_sha256", "episodes_sha256"):
        _sha(completed.get(name), name)
    for field, relative in BINDING_FILES.items():
        require(sha256(root / relative) == completed[field], f"Bound {field} mismatch")
    episodes_path = attempt / "episodes.jsonl"
    episode_sha = sha256(episodes_path)
    require(episode_sha == completed["episodes_sha256"] == audit.get("episodes_sha256"),
            "Episode JSONL hash mismatch")
    records = []
    for number, line in enumerate(episodes_path.read_text().splitlines(), 1):
        require(bool(line.strip()), f"Blank episode record at line {number}")
        records.append(_decode(line))
    validate_records(records)
    total_steps = sum(row["steps"] for row in records)
    require(type(completed.get("total_steps")) is int and completed["total_steps"] == total_steps,
            "Completed total_steps mismatch")
    require(type(audit.get("steps")) is int and audit["steps"] == total_steps,
            "Saved replay audit steps mismatch")
    require(_number(completed.get("elapsed_seconds"), "completed elapsed_seconds") > 0,
            "Completed elapsed_seconds must be positive")
    for row in records:
        relative = _payload_path(row)
        path = attempt / relative
        require(not path.is_symlink() and not path.parent.is_symlink(), "Symlink episode payload is forbidden")
        require(path.resolve(strict=True).is_relative_to(attempt), "Episode payload escapes attempt")
        require(sha256(path) == row["npz_sha256"], f"Episode NPZ hash mismatch: {relative}")
    return records, {
        "attempt": str(attempt), "completed_sha256": sha256(completed_path),
        "audit_completed_sha256": sha256(audit_path), "episodes_sha256": episode_sha,
        "protocol_sha256": completed["protocol_sha256"],
        "bindings_sha256": completed["bindings_sha256"],
        "inputs_sha256": completed["inputs_sha256"],
        "verified_episode_payloads": EPISODES,
        "execution_elapsed_seconds": completed["elapsed_seconds"],
        "binding_scope": "Exact protocol/bindings/inputs bytes, completed/audit receipts and episode payloads; "
        "source admission and official-environment replay belong to upstream execution/audit.",
    }


def markdown(summary: dict) -> str:
    gate = summary["gate"]
    verdict = "PASS" if gate["passed"] else "FAIL"
    lines = ["# Mystery Path public-memory qualification", "",
             (f"**{verdict}: {gate['checks_passed']}/{gate['checks_total']} checks passed.** "
              f"Decision: `{gate['decision']}`."), "", SCOPE, "",
             "All 64 layouts, four paired priority orders, and five controllers are retained.", "",
             (f"Distinct saved layout hashes: {summary['distinct_layout_sha256_count']}/64; "
              "coincident generated layouts are retained."), "",
             "| Controller | Successes / 256 | Actions / episode | Falls / episode | Whole ms / step |",
             "|---|---:|---:|---:|---:|"]
    for mode in MODES:
        row = summary["modes"][mode]
        lines.append(f"| {LABELS[mode]} | {row['successes']} | {row['mean_steps_all_episodes']:.3f} | "
                     f"{row['mean_falls_all_episodes']:.3f} | "
                     f"{row['amortized_milliseconds_per_environment_step']:.3f} |")
    lines += ["", ("Actions and falls include failed episodes. Whole ms/step is total episode wall time "
                   "divided by all executed environment steps."), "",
              "| Full map versus | Both succeed | Full only | Comparator only | Neither |",
              "|---|---:|---:|---:|---:|"]
    for mode, value in summary["paired_success_tables"].items():
        row = value["pooled"]
        lines.append(f"| {LABELS[mode]} | {row['both_succeed']} | {row['full_only']} | "
                     f"{row['comparator_only']} | {row['neither_succeeds']} |")
    lines += ["", "| Controller | Peak serialized state bytes | Peak retained Python bytes |",
              "|---|---:|---:|"]
    for mode in MODES:
        row = summary["modes"][mode]["state_bytes"]
        lines.append(f"| {LABELS[mode]} | {row[BYTE_FIELDS[0]]['max_episode_peak']} | "
                     f"{row[BYTE_FIELDS[1]]['max_episode_peak']} |")
    lines += ["", "| Controller | Reset s | Environment s | Act s | Parse s | Observe s | Storage s | Evaluator s |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        row = summary["modes"][mode]["timing_seconds"]
        fields = ("reset_seconds", "environment_seconds", "policy_seconds", "parsing_seconds",
                  "update_seconds", "serialization_seconds", "evaluator_seconds")
        lines.append(f"| {LABELS[mode]} | " + " | ".join(f"{row[field]:.4f}" for field in fields) + " |")
    lines += ["", "| Exact check | Count | Required | Result |", "|---|---:|---:|---|"]
    for row in gate["checks"]:
        lines.append(f"| {row['name']} | {row['actual_success_count']}/{row['denominator']} | "
                     f">= {row['minimum_success_count']}/{row['denominator']} | "
                     f"{'PASS' if row['passed'] else 'FAIL'} |")
    lines += ["", ("Per-order paired tables, all original records, and separate timing components "
                   "are retained in [summary.json](summary.json)."), ""]
    lines += [f"- {item}" for item in LIMITS]
    return "\n".join(lines) + "\n"


def figure(summary: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = ("#166a8f", "#7593a7", "#a0abb3", "#a56846", "#604b84")
    labels = ("Full map", "Last 16", "Last 32", "Erase on\nfailure", "Route ref.")
    for order, ax in enumerate(axes.flat):
        values = [summary["modes"][mode]["orders"][str(order)]["successes"] for mode in MODES]
        bars = ax.bar(range(len(MODES)), values, color=colors)
        ax.bar_label(bars, labels=[f"{value}/64" for value in values], padding=3, fontsize=9)
        ax.set_xticks(range(len(MODES)), labels)
        ax.set_ylim(0, 72)
        ax.set_yticks([0, 16, 32, 48, 64])
        ax.set_ylabel("Successful layouts")
        ax.set_title(f"Priority order {order}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.18)
    gate = summary["gate"]
    fig.suptitle("Mystery Path: public-memory qualification\n"
                 f"{'PASS' if gate['passed'] else 'FAIL'}: {gate['checks_passed']}/{gate['checks_total']} checks; "
                 "same 64 layouts in every panel, not independent repetitions", fontsize=13)
    try:
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def _write_json(path: Path, value: dict) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def report_saved(attempt: Path, out: Path, *, root: Path = ROOT) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        records, inputs = authenticate_attempt(Path(attempt), root=root)
        summary = evaluate_records(records)
        summary.update({"status": "complete", "inputs": inputs,
                        "reporter_source_sha256": sha256(Path(__file__)),
                        "new_environment_calls": 0, "new_model_calls": 0, "new_rng_calls": 0})
        figure(summary, out / "figure.png")
        with (out / "report.md").open("x") as handle:
            handle.write(markdown(summary))
        summary["report_files"] = {name: {"sha256": sha256(out / name), "bytes": (out / name).stat().st_size}
                                   for name in ("figure.png", "report.md")}
        summary["reporting_seconds_before_final_json_write"] = time.perf_counter() - start
        _write_json(out / "summary.json", summary)
        return summary
    except BaseException as error:
        try:
            _write_json(out / "failed.json", {"status": "failed", "error_type": type(error).__name__,
                                               "error": str(error), "elapsed_seconds": time.perf_counter() - start})
        except BaseException as preservation_error:  # noqa: BLE001 - preserve the original failure.
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note(f"Failed to preserve reporting failure: {preservation_error}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = report_saved(args.attempt, args.out)
    print(json.dumps({"status": summary["status"], "gate": summary["gate"]["passed"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
