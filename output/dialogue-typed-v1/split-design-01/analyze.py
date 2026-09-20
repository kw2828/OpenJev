"""One fixed TRAIN-only metadata split; no feature/model/prediction imports."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "runs/sgd-state-v1/data/completed.json"
PREP = ROOT / "runs/dialogue-conditional-v1/preparation-01/completed.json"
PINS = {
    str(DATA.relative_to(ROOT)): "677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12",
    str(PREP.relative_to(ROOT)): "960afa60172056134bc4d3cc523338b5fdb8886b55e2ef41e8ce125c72dffbb3",
}
CLASSES = ("NONE", "DONTCARE", "TRUE", "FALSE", "OTHER")
RULE = {
    "universe": "Union of services in every original official TRAIN dialogue.services list, including noncategorical services",
    "ordering": "Ascending (SHA256(UTF8('openjev-typed-v1:'+exact_service_name)).hexdigest(), exact_service_name)",
    "heldout_count": "ceil(0.20 * number of services)",
    "fit": "Existing admitted TRAIN rows in dialogues whose full original services list contains no heldout service",
    "evaluation": "Existing admitted TRAIN rows in dialogues containing any heldout service",
    "primary": "Evaluation rows whose supplied query service belongs to heldout services",
    "secondary": "Other evaluation rows, partitioned by whether their service has any admitted fit support",
    "searches": 1,
    "alternate_split": False,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    with (OUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def bind(path, digest, records, size=None):
    require(path.is_file() and not path.is_symlink(), "Missing or symlink input")
    require(sha(path) == digest, f"Input SHA mismatch: {path}")
    require(size is None or path.stat().st_size == size, "Input size mismatch")
    records[str(path.relative_to(ROOT))] = {"sha256": digest, "bytes": path.stat().st_size}


def cell(rows):
    return {"rows": len(rows), "dialogues": len({r["dialogue_id"] for r in rows}),
            "services": len({r["service"] for r in rows}),
            "schemas": len({(r["service"], r["slot"]) for r in rows})}


def support(rows):
    result = {"all": cell(rows), "by_transition": {}, "by_value": {}, "transition_by_value": {}}
    for transition in ("changed", "retained"):
        selected = [r for r in rows if (r["current_candidate_id"] != r["previous_candidate_id"]) == (transition == "changed")]
        result["by_transition"][transition] = cell(selected)
        result["transition_by_value"][transition] = {
            kind: cell([r for r in selected if r["current_value_group"].upper() == kind]) for kind in CLASSES
        }
    result["by_value"] = {kind: cell([r for r in rows if r["current_value_group"].upper() == kind]) for kind in CLASSES}
    result["five_transitions"] = {name: cell([r for r in rows if r["derived_bin"] == name]) for name in
                                  ("unmentioned_retention", "assigned_retention", "first_assignment", "revision", "clear")}
    result["schema_ids"] = sorted({r["query_id"] for r in rows})
    result["absent_values"] = [kind for kind in CLASSES if result["by_value"][kind]["rows"] == 0]
    return result


def main():
    require({p.name for p in OUT.iterdir()} == {"analyze.py"}, "Exclusive unused attempt directory required")
    started = time.monotonic()
    write("started.json", {"version": "dialogue-typed-split-design-v1", "rule": RULE, "input_pins": PINS,
                           "script_sha256": sha(Path(__file__)), "python": platform.python_version(),
                           "allowed_raw_fields": ["dialogue_id", "services"], "no_retry": True})
    inputs = {}
    try:
        for path in (DATA, PREP):
            bind(path, PINS[str(path.relative_to(ROOT))], inputs)
        data, prep = read(DATA), read(PREP)
        require(data["status"] == prep["status"] == "completed", "Incomplete source")
        require(data["test_contents_accessed"] is False, "Source test boundary")
        require(not (DATA.parent / "failed.json").exists() and not (PREP.parent / "failed.json").exists(), "Failed source")
        for name in ("catalog.json", "rows.jsonl"):
            entry = prep["files"][name]
            bind(PREP.parent / name, entry["sha256"], inputs, entry["bytes"])
        catalog = {q["query_index"]: q for q in read(PREP.parent / "catalog.json")["queries"] if q["split"] == "train"}
        original = {}
        source = Path(data["source"])
        require(source.resolve().is_relative_to(ROOT.resolve()), "Raw TRAIN source path")
        raw_members = sorted(n for n in data["source_files"] if n.startswith("train/dialogues_") and n.endswith(".json"))
        require(raw_members, "Missing original TRAIN shards")
        for name in raw_members:
            entry, path = data["source_files"][name], source / name
            bind(path, entry["sha256"], inputs, entry["bytes"])
            # Mixed containers are decoded, but no turns, text, frames or labels are accessed.
            for raw in read(path):
                identifier, services = raw["dialogue_id"], raw["services"]
                require(type(identifier) is str and identifier and identifier not in original, "Original dialogue identity")
                require(type(services) is list and services and all(type(s) is str and s for s in services)
                        and len(services) == len(set(services)), "Original public services list")
                original[identifier] = sorted(services)
        universe = sorted({s for values in original.values() for s in values},
                          key=lambda s: (hashlib.sha256(("openjev-typed-v1:" + s).encode()).hexdigest(), s))
        heldout = set(universe[:math.ceil(len(universe) / 5)])
        admitted, all_train_dialogues = [], set()
        with (PREP.parent / "rows.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                # Original ledger contains both splits; dev rows are not retained or inspected beyond split.
                if row["split"] != "train":
                    continue
                all_train_dialogues.add(row["dialogue_id"])
                require(row["dialogue_id"] in original and row["service"] in original[row["dialogue_id"]], "Full-service membership")
                if row["admission"] != "admitted":
                    continue
                q = catalog[row["query_index"]]
                require((row["service"], row["slot"], row["query_id"]) == (q["service"], q["slot"], q["query_id"]), "Admitted schema identity")
                require(q["candidate_ids"][row["current_label_index"]] == row["current_candidate_id"]
                        and q["candidate_ids"][row["previous_current_index"]] == row["previous_candidate_id"], "Admitted current/previous identity")
                label, current = row["current_label_index"], row["current_candidate_id"]
                kind = ("NONE" if current == "reserved:NOT_MENTIONED" else "DONTCARE" if current == "reserved:DONTCARE"
                        else q["candidate_values"][label].strip().upper() if q["boolean_slot"] else "OTHER")
                require(kind == row["current_value_group"].upper() and kind in CLASSES, "Target metadata classification")
                admitted.append(row)
        require(len({r["row_index"] for r in admitted}) == len(admitted), "Duplicate admitted TRAIN row")
        eval_dialogues = {d for d, services in original.items() if heldout.intersection(services)}
        fit = [r for r in admitted if r["dialogue_id"] not in eval_dialogues]
        evaluation = [r for r in admitted if r["dialogue_id"] in eval_dialogues]
        primary = [r for r in evaluation if r["service"] in heldout]
        secondary = [r for r in evaluation if r["service"] not in heldout]
        fit_services = {r["service"] for r in fit}
        partitions = {"fit": fit, "evaluation": evaluation, "primary_heldout_service": primary,
                      "secondary_nonheldout_service": secondary,
                      "secondary_seen_in_fit": [r for r in secondary if r["service"] in fit_services],
                      "secondary_absent_from_fit": [r for r in secondary if r["service"] not in fit_services]}
        require(len(fit) + len(evaluation) == len(admitted) and len(primary) + len(secondary) == len(evaluation), "Partition closure")
        require(not {r["dialogue_id"] for r in fit}.intersection(r["dialogue_id"] for r in evaluation), "Dialogue leakage")
        require(not heldout.intersection(s for r in fit for s in original[r["dialogue_id"]]), "Full-service fit leakage")
        write("services.json", {"rule": RULE, "service_count": len(universe), "heldout_count": len(heldout),
              "services_in_hash_order": [{"service": s, "digest": hashlib.sha256(("openjev-typed-v1:" + s).encode()).hexdigest(),
                                           "heldout": s in heldout} for s in universe]})
        write("membership.json", {"original_train_dialogues": [{"dialogue_id": d, "services": original[d],
              "partition": "evaluation" if d in eval_dialogues else "fit", "in_existing_scored_cohort": d in all_train_dialogues}
              for d in sorted(original)], "admitted_row_indices": {k: [r["row_index"] for r in rows] for k, rows in partitions.items()}})
        summary = {"rule": RULE, "service_count": len(universe), "heldout_services": sorted(heldout),
                   "raw_train_shards": len(raw_members), "original_train_dialogues": len(original),
                   "original_fit_dialogues": len(original) - len(eval_dialogues), "original_eval_dialogues": len(eval_dialogues),
                   "existing_scored_cohort_dialogues": len(all_train_dialogues), "admitted_train_rows": len(admitted),
                   "partitions": {k: support(v) for k, v in partitions.items()},
                   "scope": "Historically fitted official TRAIN only. New pseudo-unseen development split, not fresh confirmation. No official DEV/TEST predictions, encoder, features or model calls.",
                   "mixed_container_boundary": "Raw TRAIN JSON decoded; only dialogue_id/services used. Mixed conditional catalog/ledger decoded; only TRAIN metadata retained. Text and official DEV labels are not used.",
                   "model_calls": 0, "encoder_calls": 0, "optimizer_steps": 0,
                   "feature_arrays_loaded": 0, "predictions_loaded": 0}
        write("summary.json", summary)
        sources = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
        for name in ("scripts/prepare_dialogue_conditional.py", "src/openjev/research/dialogue_state_data.py", "scripts/prepare_sgd_state.py"):
            sources[name] = sha(ROOT / name)
        files = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in OUT.iterdir() if p.is_file()}
        write("receipt.json", {"status": "completed", "version": "dialogue-typed-split-design-v1", "rule": RULE,
              "files": files, "inputs": inputs, "sources": sources, "wall_seconds": time.monotonic() - started,
              "scope": summary["scope"], "no_retry": True, "alternate_splits_evaluated": 0})
        print(json.dumps({"heldout_services": summary["heldout_services"], "services": len(universe),
                          "partitions": {k: v["all"] for k, v in summary["partitions"].items()},
                          "receipt_sha256": sha(OUT / "receipt.json")}, indent=2))
    except BaseException as error:
        write("failed.json", {"status": "failed", "error": type(error).__name__, "message": str(error),
                              "inputs": inputs, "wall_seconds": time.monotonic() - started, "rule": RULE, "no_retry": True})
        raise


if __name__ == "__main__":
    main()
