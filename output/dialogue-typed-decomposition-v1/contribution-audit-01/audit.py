"""Independent, bounded saved-log contribution and TRUE-branch check."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "output/dialogue-typed-v1/training-01"
ANALYSIS = ROOT / "output/dialogue-typed-decomposition-v1/diagnostic-01/summary.json"
RUN_PIN = "65bd57084dbc7f7b99b782e140bae2e18f25b9e237e89a0132cead5fbfe2a095"
ANALYSIS_PIN = "7de563a2a3903707ff694efdfa5e50f753bd28cf237b5fd1ad5290bcd11b0cef"
ARMS = ("flat_stratum", "flat_balanced", "typed_stratum", "typed_balanced")
SEEDS = (6101, 6102, 6103)
TYPES = ("none", "dontcare", "true", "false", "other")
COMPONENTS = ("total", "branch", "value")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(actual, expected, label):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label)


def weights(services):
    counts = {s: services.count(s) for s in set(services)}
    return {"row": [1 / len(services)] * len(services),
            "equal_service": [1 / (len(counts) * counts[s]) for s in services]}


def contribution(delta, weight, selected):
    return math.fsum(float(delta[i]) * weight[i] for i in selected)


def self_check():
    w = weights(["a", "a", "b"])
    require(w["equal_service"] == [.25, .25, .5], "Synthetic equal-service denominator")
    d = [1., 3., -2.]
    close(contribution(d, w["equal_service"], range(3)), 0., "Synthetic service cancellation")
    close(contribution(d, w["row"], range(3)), 2 / 3, "Synthetic row cancellation")
    require(contribution(d, w["equal_service"], []) == 0, "Empty contribution is zero")
    groups = ([0], [1], [], [2])
    for key in w:
        close(math.fsum(contribution(d, w[key], g) for g in groups),
              contribution(d, w[key], range(3)), "Synthetic partition denominator")


def quantities(rows, saved_logs, saved_ids):
    require(saved_logs.dtype == np.float32 and saved_logs.shape == (len(rows), 12)
            and saved_ids.dtype == np.int64 and saved_ids.tolist() == [r["row_index"] for r in rows],
            "Canonical prediction array identity")
    output = []
    for row, raw in zip(rows, saved_logs, strict=True):
        n, target = row["candidate_count"], row["current_label_index"]
        kinds = row["candidate_types"]
        logs = raw.astype(np.float64)
        require(np.isfinite(logs[:n]).all() and np.all(logs[:n] <= 0)
                and np.isneginf(logs[n:]).all(), "Raw support/mask")
        require(abs(math.fsum(math.exp(float(x)) for x in logs[:n]) - 1) <= 2e-6,
                "Raw probability normalization, without repair")
        if not (row["heldout_service"] and target != row["previous_current_index"]):
            continue
        branch = min(kinds[target], 2)
        branch_candidates = [j for j in range(n) if min(kinds[j], 2) == branch]
        maximum = max(float(logs[j]) for j in branch_candidates)
        mass_log = maximum + math.log(math.fsum(math.exp(float(logs[j]) - maximum)
                                                for j in branch_candidates))
        total, branch_loss = -float(logs[target]), -mass_log
        value = mass_log - float(logs[target])
        require(abs(total - branch_loss - value) <= 1e-10, "Raw additive identity")
        choice = max(range(n), key=lambda j: logs[j])
        conditional = max(branch_candidates, key=lambda j: logs[j])
        output.append({"total": total, "branch": branch_loss, "value": value,
                       "correct": choice == target, "conditional_correct": conditional == target,
                       "wrong_branch": min(kinds[choice], 2) != branch,
                       "wrong_value": choice != target and min(kinds[choice], 2) == branch})
    return output


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    require({p.name for p in OUT.iterdir()} == {"audit.py"}, "Exclusive unexecuted audit directory")
    start = time.monotonic()
    source = sha(__file__)
    inputs = {}
    try:
        self_check()

        def bind(path, pin, size=None):
            require(sha(path) == pin and (size is None or Path(path).stat().st_size == size),
                    "Authenticated input: " + str(path))
            inputs[str(Path(path).relative_to(ROOT))] = {"sha256": pin, "bytes": Path(path).stat().st_size}

        bind(RUN / "completed.json", RUN_PIN)
        bind(ANALYSIS, ANALYSIS_PIN)
        done, published = read(RUN / "completed.json"), read(ANALYSIS)
        names = {f"{arm}-{seed}" for arm in ARMS for seed in SEEDS}
        require(done["status"] == published["status"] == "completed"
                and len(done["completed_fits"]) == 12 and set(done["completed_fits"]) == names
                and set(published["fits"]) == names and published["execution_completed_sha256"] == RUN_PIN
                and published["original_continuation_allowed"] is False, "Complete, closed study binding")
        files = ["evaluation-rows.jsonl"] + [f"fits/{name}/predictions.npz" for name in sorted(names)]
        for name in files:
            entry = done["files"][name]
            bind(RUN / name, entry["sha256"], entry["bytes"])
        rows = [json.loads(line) for line in (RUN / "evaluation-rows.jsonl").read_text().splitlines()]
        primary = [r for r in rows if r["heldout_service"] and r["current_label_index"] != r["previous_current_index"]]
        require(len(primary) == 578, "Fixed primary panel")
        for r in primary:
            require(r["current_value_group"] == TYPES[r["candidate_types"][r["current_label_index"]]],
                    "Target type from public schema")
        true_indices = [i for i, r in enumerate(primary) if r["current_value_group"] == "true"]
        require(len(true_indices) == 29 and all(tuple(primary[i]["candidate_types"]) == (0, 1, 2, 3)
                                              for i in true_indices), "TRUE support: NONE/DC/TRUE/FALSE")
        w = weights([r["service"] for r in primary])
        q, true_checks = {}, {}
        for name in sorted(names):
            with np.load(RUN / "fits" / name / "predictions.npz", allow_pickle=False) as archive:
                require(set(archive.files) == {"log_probs", "row_indices"}, "Exact saved prediction fields")
                q[name] = quantities(rows, archive["log_probs"], archive["row_indices"])
            require(len(q[name]) == len(primary), "Primary row join")
            counts = {key: sum(bool(q[name][i][key]) for i in true_indices)
                      for key in ("correct", "conditional_correct", "wrong_branch", "wrong_value")}
            cell = published["fits"][name]["groups"]["primary/value/true"]
            require(cell["rows"] == 29, "Published TRUE support")
            for a, b in (("correct", "correct"), ("conditional_correct", "conditional_value_correct"),
                         ("wrong_branch", "wrong_selected_branch"), ("wrong_value", "wrong_value")):
                require(cell["decisions"][b] == counts[a], "Published TRUE decision partition")
            require(counts["correct"] + counts["wrong_branch"] + counts["wrong_value"] == 29,
                    "TRUE actual decisions partition")
            true_checks[name] = {"rows": 29, **counts}

        results = {}
        for seed in SEEDS:
            before, after = q[f"flat_balanced-{seed}"], q[f"typed_balanced-{seed}"]
            partitions = {"by_correctness": {key: [] for key in ("CC", "CW", "WC", "WW")},
                          "by_target_type": {key: [] for key in TYPES}}
            for i, (a, b) in enumerate(zip(before, after, strict=True)):
                category = ("C" if a["correct"] else "W") + ("C" if b["correct"] else "W")
                partitions["by_correctness"][category].append(i)
                partitions["by_target_type"][primary[i]["current_value_group"]].append(i)
            pair = published["pairs"]["typing_balanced"][str(seed)]
            require(pair["base"] == f"flat_balanced-{seed}" and pair["candidate"] == f"typed_balanced-{seed}",
                    "Fixed primary contrast orientation")
            changes = {c: [b[c] - a[c] for a, b in zip(before, after, strict=True)] for c in COMPONENTS}
            results[str(seed)] = {}
            for partition, groups in partitions.items():
                require(sorted(i for group in groups.values() for i in group) == list(range(578)),
                        "Exhaustive disjoint contribution partition")
                results[str(seed)][partition] = {}
                for label, selected in groups.items():
                    cell = pair["primary_contributions"][partition][label]
                    require(cell["rows"] == len(selected), "Contribution support")
                    losses = {c: {weight: contribution(delta, values, selected) for weight, values in w.items()}
                              for c, delta in changes.items()}
                    for c in COMPONENTS:
                        for weight in w:
                            close(cell["loss_difference"][c][weight], losses[c][weight], "Contribution arithmetic")
                    results[str(seed)][partition][label] = {"rows": len(selected), "loss_difference": losses}
                for c, delta in changes.items():
                    for weight, values in w.items():
                        total = contribution(delta, values, range(578))
                        close(math.fsum(cell["loss_difference"][c][weight]
                                        for cell in results[str(seed)][partition].values()), total, "Contribution sum")
                        close(pair["groups"]["heldout_service/changed"]["loss_difference"][c][weight],
                              total, "Full primary denominator join")

        mean = {partition: {label: {c: {weight: math.fsum(results[str(seed)][partition][label]["loss_difference"][c][weight]
                                                         for seed in SEEDS) / 3 for weight in w} for c in COMPONENTS}
                            for label in results[str(SEEDS[0])][partition]}
                for partition in ("by_correctness", "by_target_type")}
        for path, entry in inputs.items():
            require(sha(ROOT / path) == entry["sha256"], "End input stability")
        require(sha(__file__) == source, "Audit source stability")
        summary = {"status": "completed", "all_scoped_quantities_match": True, "primary_rows": 578,
                   "primary_services": len({r["service"] for r in primary}),
                   "contribution_scalars_checked": 162, "contribution_supports_checked": 27,
                   "true_decision_counts_checked": 48, "per_seed": results,
                   "mean_across_three_seeds": mean, "true_target": {"candidate_type_order": [0, 1, 2, 3], "fits": true_checks},
                   "scope": "Independent saved-log arithmetic for typing_balanced only: correctness and target-type contributions use full primary row/equal-service denominators. All twelve TRUE conditional and actual decision counts are checked. No producer/report/model imports, probability repair, new forecasts, significance or causal inference. WW denotes both actual candidate decisions wrong, not the same wrong candidate. Seed-row events repeat the same 578 rows. Original continuation remains failed."}
        write(OUT / "summary.json", summary)
        write(OUT / "receipt.json", {"status": "completed", "source_sha256": source,
              "run_completed_sha256": RUN_PIN, "diagnostic_summary_sha256": ANALYSIS_PIN,
              "inputs": inputs, "files": {"audit.py": {"sha256": source, "bytes": Path(__file__).stat().st_size},
              "summary.json": {"sha256": sha(OUT / "summary.json"), "bytes": (OUT / "summary.json").stat().st_size}},
              "wall_seconds": time.monotonic() - start, "model_calls": 0, "training_calls": 0, "rng_calls": 0,
              "scope": summary["scope"]})
        print(json.dumps({"status": "completed", "source_sha256": source, "summary_sha256": sha(OUT / "summary.json"),
                          "receipt_sha256": sha(OUT / "receipt.json")}))
    except BaseException as error:
        try:
            write(OUT / "failed.json", {"status": "failed", "error": repr(error), "source_sha256": source,
                  "run_completed_sha256": RUN_PIN, "diagnostic_summary_sha256": ANALYSIS_PIN,
                  "wall_seconds": time.monotonic() - start, "model_calls": 0})
        except BaseException as save_error:  # noqa: BLE001 - preserve original failure
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failure preservation error: " + repr(save_error))
        raise


if __name__ == "__main__":
    main()
