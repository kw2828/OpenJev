"""Narrow independent saved-output audit. No producer/reporter/model imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (6101, 6102, 6103)
ARMS = ("flat_stratum", "flat_balanced", "typed_stratum", "typed_balanced")
NAMES = {f"{arm}-{seed}" for arm in ARMS for seed in SEEDS}
VALUES = ("none", "dontcare", "true", "false", "other")
BASE, PRIMARY = "flat_balanced", "typed_balanced"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1024*1024), b""):
            h.update(piece)
    return h.hexdigest()


def decode(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON value: "+value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def read(path):
    return decode(Path(path).read_text())


def compare(actual, expected, location):
    if type(expected) is dict:
        require(type(actual) is dict and set(actual) == set(expected), "Fields: "+location)
        for k, v in expected.items(): compare(actual[k], v, location+"/"+k)
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), "List: "+location)
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)): compare(a, e, location+"/"+str(i))
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), "Arithmetic: "+location)
    else:
        require(type(actual) is type(expected) and actual == expected, "Identity: "+location)


def authenticate(args):
    run, summary_path = Path(args.run).resolve(), Path(args.summary).resolve()
    bindings = {}
    def bind(path, pin, size=None):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), "Input boundary")
        actual = {"sha256": sha(path), "bytes": path.stat().st_size}
        require(actual["sha256"] == pin and (size is None or actual["bytes"] == size), "Hash/size: "+str(path))
        bindings[str(path.relative_to(ROOT))] = actual
    def closure(path, files, names):
        require(set(files) == names, "Manifest member names")
        require({p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()} == names | {"completed.json"}, "Exact file tree")
        for name, entry in files.items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts and not (path/name).is_symlink(), "Unsafe file")
            bind(path/name, entry["sha256"], entry["bytes"])
    bind(run/"completed.json", args.run_sha256)
    bind(summary_path, args.summary_sha256)
    done, summary = read(run/"completed.json"), read(summary_path)
    require(done["status"] == summary["status"] == "completed" and done["version"] == summary["version"] == "dialogue-typed-v1", "Completed study identity")
    require(summary["execution_completed_sha256"] == args.run_sha256 and summary["technical_validity_passed"] is True, "Reporter run binding")
    order = [f"{arm}-{seed}" for i, seed in enumerate(SEEDS) for arm in ARMS[i:]+ARMS[:i]]
    require(done["completed_fits"] == done["expected_fits"] == order and set(summary["fits"]) == NAMES, "Exactly twelve final fits")
    members = {"started.json", "plan.json", "evaluation-rows.jsonl", "references.npz"} | {f"orders-{s}.npy" for s in SEEDS}
    members |= {f"fits/{fit}/{name}" for fit in NAMES for name in ("completed.json", "weights.pt", "updates.jsonl", "predictions.npz")}
    closure(run, done["files"], members)
    require(len(members) == 55, "56 execution files including completion")
    plan, started = read(run/"plan.json"), read(run/"started.json")
    require(sha(run/"plan.json") == done["plan_sha256"] == summary["plan_sha256"] == started["request"]["plan_sha256"], "Plan digest joins")
    require(plan["config"]["seeds"] == list(SEEDS) and plan["config"]["methods"] == list(ARMS)
            and plan["config"]["epochs"] == 20 and plan["expected_fits"] == order, "Paired recipe")
    source_map = plan["source_sha256"]
    require(len(source_map) == 25 and source_map == done["source_sha256"] == summary["source_sha256"], "25-source closure")
    frozen = Path(started["request"]["plan"]).resolve().parent
    bind(frozen/"plan.json", done["plan_sha256"])
    freeze = read(frozen/"completed.json")
    bind(frozen/"completed.json", sha(frozen/"completed.json"))
    require(freeze["status"] == "completed" and freeze["phase"] == "freeze" and freeze["plan_sha256"] == done["plan_sha256"], "Frozen plan completion")
    closure(frozen, freeze["files"], {"started.json", "plan.json"} | {f"orders-{s}.npy" for s in SEEDS} | {"sources/"+s for s in source_map})
    for name, pin in source_map.items():
        bind(ROOT/name, pin)
        require(freeze["files"]["sources/"+name]["sha256"] == pin, "Source snapshot identity")
    prepared = Path(plan["prepared_path"]).resolve()
    bind(prepared/"completed.json", plan["prepared_completed_sha256"])
    prepared_done = read(prepared/"completed.json")
    closure(prepared, prepared_done["files"], {"started.json", "summary.json", "catalog.json", "rows.jsonl"})
    canonical = {r["row_index"]: r for r in (decode(line) for line in (prepared/"rows.jsonl").read_text().splitlines())}
    catalog = read(prepared/"catalog.json")["queries"]
    rows = [decode(line) for line in (run/"evaluation-rows.jsonl").read_text().splitlines()]
    require([r["row_index"] for r in rows] == plan["evaluation_row_indices"] == sorted(set(plan["evaluation_row_indices"])), "Canonical evaluation coverage")
    held_services = set(plan["split"]["heldout_services"])
    for row in rows:
        q = catalog[row["query_index"]]
        types = []
        for cid, value in zip(q["candidate_ids"], q["candidate_values"], strict=True):
            types.append(0 if cid == "reserved:NOT_MENTIONED" else 1 if cid == "reserved:DONTCARE" else
                         2 if q["boolean_slot"] and value.strip().casefold() == "true" else 3 if q["boolean_slot"] else 4)
        expected = {**canonical[row["row_index"]], "candidate_types": types, "heldout_service": row["service"] in held_services}
        require(row == expected and row["split"] == "train" and row["admission"] == "admitted", "Exact canonical metadata/schema join")
        target, previous = row["current_label_index"], row["previous_current_index"]
        require(len(types) == row["candidate_count"] and 3 <= len(types) <= 12 and type(target) is int and type(previous) is int
                and 0 <= target < len(types) and 0 <= previous < len(types), "Target/prior support")
        require(types.count(0) == types.count(1) == 1 and types.count(2) <= 1 and types.count(3) <= 1, "Candidate type support")
        require(q["candidate_ids"][target] == row["current_candidate_id"] and q["candidate_ids"][previous] == row["previous_candidate_id"], "Canonical previous remapping")
        require(VALUES[types[target]] == row["current_value_group"], "Current type label")
    fit_ids = plan["fit_row_indices"]
    require(fit_ids == sorted(set(fit_ids)) and not set(fit_ids) & set(plan["evaluation_row_indices"]), "Disjoint fit/evaluation rows")
    fit_rows = [canonical[i] for i in fit_ids]
    require(not {r["dialogue_id"] for r in fit_rows} & {r["dialogue_id"] for r in rows}, "Disjoint dialogues")
    require(all(r["split"] == "train" and r["admission"] == "admitted" and r["service"] not in held_services for r in fit_rows), "Fit service exclusion")
    initial = {}
    for name in NAMES:
        record = read(run/"fits"/name/"completed.json")
        arm, seed = name.rsplit("-", 1)
        require(record["status"] == "completed" and record["method"] == arm and record["seed"] == int(seed)
                and record["plan_sha256"] == done["plan_sha256"], "Fit receipt identity")
        for payload in ("weights.pt", "updates.jsonl", "predictions.npz"):
            require(record["files"][payload] == done["files"][f"fits/{name}/{payload}"], "Fit/root payload identity")
        initial[name] = record["initial_state_sha256"]
    for seed in SEEDS:
        require(len({initial[f"{arm}-{seed}"] for arm in ARMS}) == 1, "Four-arm paired initializer witness")
    return run, rows, summary, bindings


def calculate(rows, log, indices):
    require(indices.dtype == np.int64 and indices.tolist() == [r["row_index"] for r in rows], "Prediction row alignment")
    require(log.dtype == np.float32 and log.shape == (len(rows), 12), "Saved distribution shape")
    mask = np.arange(12)[None, :] < np.asarray([r["candidate_count"] for r in rows])[:, None]
    require(np.isfinite(log[mask]).all() and (log[mask] <= 0).all() and np.isneginf(log[~mask]).all(), "Finite supported logs and masked infinity")
    log = log.astype(np.float64)
    mass = np.exp(log).sum(axis=1)
    require(np.all(np.abs(mass-1) <= 2e-6), "Unmodified probability mass")
    choices = np.argmax(log, axis=1)
    target = np.asarray([r["current_label_index"] for r in rows]); previous = np.asarray([r["previous_current_index"] for r in rows])
    changed = target != previous
    nll = -log[np.arange(len(rows)), target]
    correct = choices == target
    held = np.asarray([r["heldout_service"] for r in rows])
    cells = {}
    for panel, population in (("all", np.ones(len(rows), bool)), ("heldout_service", held), ("seen_service", ~held)):
        for subset, selected in (("all", population), ("changed", population & changed), ("retained", population & ~changed)):
            ids = np.flatnonzero(selected).tolist()
            by_service = {}
            for i in ids: by_service.setdefault(rows[i]["service"], []).append(i)
            loss = math.fsum(float(nll[i]) for i in ids)/len(ids) if ids else None
            service_loss = (math.fsum(math.fsum(float(nll[i]) for i in part)/len(part) for part in by_service.values())/len(by_service)) if ids else None
            cells[panel+"/"+subset] = {"rows": len(ids), "services": len(by_service), "correct_count": sum(int(correct[i]) for i in ids),
                "row_accuracy": sum(int(correct[i]) for i in ids)/len(ids) if ids else None,
                "row_nll": loss, "equal_service_nll": service_loss}
    decisions = {}
    for value, code in (("true", 2), ("dontcare", 1)):
        recall_indices = [i for i, r in enumerate(rows) if held[i] and changed[i] and r["candidate_types"][target[i]] == code]
        fp_indices = [i for i, r in enumerate(rows) if held[i] and code in r["candidate_types"] and r["candidate_types"][target[i]] != code]
        decisions[value+"_changed_recall"] = {"numerator": sum(int(correct[i]) for i in recall_indices), "denominator": len(recall_indices)}
        decisions[value+"_false_positive_rate"] = {"numerator": sum(rows[i]["candidate_types"][int(choices[i])] == code for i in fp_indices), "denominator": len(fp_indices)}
    retained = np.flatnonzero(held & ~changed).tolist()
    decisions["retained_error"] = {"numerator": sum(not bool(correct[i]) for i in retained), "denominator": len(retained)}
    return {"cells": cells, "decisions": decisions}


def qualify(fits):
    paired = []
    for seed in SEEDS:
        b, t = (fits[f"{arm}-{seed}"]["cells"]["heldout_service/changed"]["equal_service_nll"] for arm in (BASE, PRIMARY))
        paired.append({"seed": seed, "flat": b, "typed": t, "difference": t-b if t is not None and b is not None else None})
    available = all(p["difference"] is not None for p in paired)
    b = math.fsum(p["flat"] for p in paired)/3 if available else None
    t = math.fsum(p["typed"] for p in paired)/3 if available else None
    checks = [{"name": "equal_service_changed_nll_reduction", "flat": b, "typed": t,
               "relative_reduction": (b-t)/b if available and b > 0 else None, "threshold": .05,
               "passed": bool(available and b > 0 and t <= .95*b)}]
    for p in paired:
        checks.append({"name": f"seed_{p['seed']}_nll_nonworse", "difference": p["difference"], "passed": p["difference"] is not None and p["difference"] <= 0})
    rates = {}
    for name, bound, lower in (("true_changed_recall", Fraction(1,20), True), ("dontcare_changed_recall", Fraction(1,20), True),
                              ("true_false_positive_rate", Fraction(1,200), False), ("dontcare_false_positive_rate", Fraction(1,200), False),
                              ("retained_error", Fraction(1,200), False)):
        entries = [{"seed": seed, **{arm: fits[f"{arm}-{seed}"]["decisions"][name] for arm in (BASE, PRIMARY)}} for seed in SEEDS]
        available = all(e[arm]["denominator"] for e in entries for arm in (BASE, PRIMARY))
        if available:
            require(all(e[BASE]["denominator"] == e[PRIMARY]["denominator"] for e in entries), "Paired decision denominator")
            means = {arm: sum((Fraction(e[arm]["numerator"], e[arm]["denominator"]) for e in entries), Fraction())/3 for arm in (BASE, PRIMARY)}
            difference = means[PRIMARY]-means[BASE]
            passed = difference >= bound if lower else difference <= bound
        else:
            means, difference, passed = {BASE: None, PRIMARY: None}, None, False
        rates[name] = {"paired": entries, "mean_rates": {k: None if v is None else float(v) for k, v in means.items()}, "difference": None if difference is None else float(difference)}
        checks.append({"name": name, "difference": None if difference is None else float(difference), "threshold": float(bound), "relation": ">=" if lower else "<=", "passed": bool(passed)})
    return {"paired_nll": paired, "decision_counts": rates, "checks": checks, "checks_passed": sum(c["passed"] for c in checks), "total_checks": 9, "passed": all(c["passed"] for c in checks)}


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); source = sha(__file__)
    try:
        run, rows, summary, inputs = authenticate(args)
        fits = {}
        for name in sorted(NAMES):
            with np.load(run/"fits"/name/"predictions.npz", allow_pickle=False) as archive:
                require(set(archive.files) == {"row_indices", "log_probs"}, "Exact prediction keys")
                fits[name] = calculate(rows, archive["log_probs"], archive["row_indices"])
            published = summary["fits"][name]
            compare(published["decisions"], fits[name]["decisions"], name+"/decisions")
            for group, metrics in fits[name]["cells"].items():
                cell = published["cells"][group]
                compare(cell["rows"], metrics["rows"], name+"/"+group+"/rows")
                compare(cell["services"], metrics["services"], name+"/"+group+"/services")
                compare(cell["accuracy"]["row"], metrics["row_accuracy"], name+"/"+group+"/accuracy")
                compare(cell["nll"]["row"], metrics["row_nll"], name+"/"+group+"/row_nll")
                compare(cell["nll"]["equal_service"], metrics["equal_service_nll"], name+"/"+group+"/service_nll")
        rule = qualify(fits)
        for key, value in rule.items(): compare(summary["continuation"][key], value, "continuation/"+key)
        compare(summary["continuation_allowed"], rule["passed"], "Overall continuation")
        for name, entry in inputs.items():
            path = ROOT/name
            require(sha(path) == entry["sha256"] and path.stat().st_size == entry["bytes"], "End input stability")
        result = {"status": "completed", "all_scoped_quantities_match": True, "fits": fits, "continuation": rule,
                  "fit_count": 12, "metric_cells_verified": 108, "decision_count_cells_verified": 60,
                  "run_completed_sha256": args.run_sha256, "report_summary_sha256": args.summary_sha256,
                  "scope": ["Independent direct saved-log NLL, top-1 accuracy, equal-service NLL for all/changed/retained across three panels and twelve fits; five decision counts per fit; all nine continuation checks.",
                            "Exact execution manifests, live/frozen sources, paired initializer witnesses and canonical original prepared row/schema joins verified. No checkpoint deserialization, model calls, RNG or probability repair.",
                            "Not an independent recomputation of Brier, equal-dialogue means, all 144 transition/value groups, references, training journals, optimizer behavior, clocks or RSS. Those remain authenticated producer/reporter witnesses.",
                            "Prepared semantic admission and public split generation are inherited from authenticated preparation/source-bound lineage. Historically exposed TRAIN development data; gold prior supplied; no significance or fresh-confirmation claim."]}
        write(out/"summary.json", result)
        write(out/"receipt.json", {"status": "completed", "source_sha256": source, "source_path": str(Path(__file__).resolve().relative_to(ROOT)),
              "run_completed_sha256": args.run_sha256, "report_summary_sha256": args.summary_sha256,
              "files": {"summary.json": {"sha256": sha(out/"summary.json"), "bytes": (out/"summary.json").stat().st_size}},
              "authenticated_inputs": inputs, "wall_seconds": time.monotonic()-started,
              "continuation_allowed": rule["passed"], "model_calls": 0, "rng_calls": 0, "training_calls": 0})
        print(json.dumps({"status": "completed", "continuation_allowed": rule["passed"], "summary_sha256": sha(out/"summary.json"), "receipt_sha256": sha(out/"receipt.json"), "source_sha256": source}))
    except BaseException as error:
        try:
            write(out/"failed.json", {"status": "failed", "error": repr(error), "source_sha256": source,
                  "run_completed_sha256": args.run_sha256, "report_summary_sha256": args.summary_sha256,
                  "wall_seconds": time.monotonic()-started, "model_calls": 0})
        except BaseException as secondary:
            if callable(getattr(error, "add_note", None)): error.add_note("Failure receipt error: "+repr(secondary))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--run-sha256", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
