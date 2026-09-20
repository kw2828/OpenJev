"""Independent, saved-only primary arithmetic for the commitment diagnostic.

No producer, model, encoder or training imports. This is a posthoc metric audit,
not a replay of training or a new continuation decision. Real inputs may be read
only after the externally pinned diagnostic has completed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-commitment-independent-primary-audit-v1"
SEEDS = (6201, 6202, 6203)
METHODS = ("flat_stratum", "token_mean", "token_aligned")
CONSTRUCTIONS = ("flat", "MM", "AA", "MA", "AM")
STRATA = ("all", "changed", "retained", "unmentioned_retention", "assigned_retention")
LIMITS = {"wall_seconds": 60, "rss_bytes": 1024**3, "output_bytes": 64*1024**2}
RUN_PIN = "e7222147935b1f1604d31da83c940ed428ae3745dfddec374cfa64f7ddd1ce7c"
PLAN_PIN = "e700080ee2dd28c83c0c13a0dad2640cdc83fb1992efb4bfd3777a90111877c1"
ROWS_PIN = "56e572f6df34cf81ccc38db137d82a699eb3f4f84509af39682d7d963e62a4d0"
PREDICTION_PINS = {
    "flat_stratum-6201": "dbccd26acbaa4622de3b2ecd66a752f7a06d67cc9598c880dc933ee1c8ba27ec",
    "flat_stratum-6202": "1ff1ae1b6a8f746fd086a63e7ecfa355751e2c1bcb5a6f7cdeaa8cba42444537",
    "flat_stratum-6203": "773ee404db257cfa81c5de0fe75bc6e495de10759c2094c9ac06857e256caf40",
    "token_mean-6201": "1979aa6c082415d208d4ea5ab8dfff738a76c0f09efa569199a083f573e16dc3",
    "token_mean-6202": "ec1534af8240f36585c5c6512414cf2eba9c13d27ffac7083c0e7411738ea72d",
    "token_mean-6203": "2b20d456f367bf0fd2cff57ee0684faba454f0aeea54042cef956620b5098c9b",
    "token_aligned-6201": "bae0149549750430358c52ef0ecff2b2e237704ff442fc232d6ea2a912d4a170",
    "token_aligned-6202": "0067cbf5e9fb38a24a221873d27addea3f1cd3a5e9bda8125806afbb9bbe42d6",
    "token_aligned-6203": "066b953cd9073fc4a927feeb77e0a6c44225f255957a4a57f038fd3745bcfb0a",
}
PARENT_PINS = {
    "report-01/receipt.json": "7591182a5c07c9cd4ee4035128b6c0900d7d9177ba89efccfe92bfd528c8a6c5",
    "report-01/summary.json": "b4b8c1d3f6e9d1e98096c2684a5bca84b011496f89044e3f3eb31260f2fbbdc3",
    "audit-01/result-01/receipt.json": "8773cf48d0b07b10f6bb920549d3628a4716beb814c98185df021081046b19bd",
}
SCOPE = (
    "Independent primary held-out-service metrics for all three seeds and five distributions, "
    "including normalized NLL/Brier, decision partitions and paired repairs/harms. "
    "Other panels, per-service tables, producer means and training/source-execution witnesses "
    "are not independently reproduced. Authentication covers the structural training manifest "
    "and selected plan/row/prediction bytes, without opening checkpoints, journals or orders. "
    "Original frozen continuation remains FAIL (18/22). "
    "No model, encoder, checkpoint deserialization or training calls; no new continuation rule."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def read(path):
    return decode(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def item(path):
    return {"sha256": sha(path), "bytes": Path(path).stat().st_size}


def safe(parent, name):
    parent = Path(parent).resolve()
    path = (parent/name).resolve()
    require(path != parent and path.is_relative_to(parent), "Unsafe manifest path")
    return path


def manifest(parent, members, *, terminal):
    require(type(members) is dict and bool(members), "Payload manifest required")
    for name, record in members.items():
        require(set(record) == {"sha256", "bytes"}, "Manifest record schema")
        require(item(safe(parent, name)) == record, "Payload identity: " + name)
    actual = {p.relative_to(parent).as_posix() for p in Path(parent).rglob("*") if p.is_file()}
    require(actual == set(members) | {terminal}, "Exact artifact membership")


def metadata(rows, width=12):
    ids, labels, previous, sizes, kinds, held = [], [], [], [], [], []
    for row in rows:
        n, y, old = row["candidate_count"], row["current_label_index"], row["previous_current_index"]
        require(type(n) is int and 2 <= n <= width and type(y) is int and type(old) is int
                and 0 <= y < n and 0 <= old < n, "Candidate and previous/target support")
        types = row["candidate_types"]
        require(len(types) == n and all(type(t) is int and 0 <= t <= 4 for t in types)
                and types.count(0) == types.count(1) == 1, "Canonical candidate types")
        require(type(row["heldout_service"]) is bool and type(row["row_index"]) is int,
                "Canonical panel and row identity")
        require(row["split"] == "train" and row["admission"] == "admitted", "Admitted TRAIN rows")
        require((y == old) == (row["current_candidate_id"] == row["previous_candidate_id"]),
                "Current/previous identity mapping")
        transition = ("unmentioned_retention" if y == old and types[y] == 0 else
                      "assigned_retention" if y == old else "first_assignment" if types[old] == 0 else
                      "clear" if types[y] == 0 else "revision")
        require(row["derived_bin"] == transition, "Derived transition identity")
        require(row["current_value_group"] == ("none", "dontcare", "true", "false", "other")[types[y]],
                "Target value identity")
        ids.append(row["row_index"]); labels.append(y); previous.append(old)
        sizes.append(n); kinds.append(types+[-1]*(width-n)); held.append(row["heldout_service"])
    require(ids == sorted(set(ids)) and all(i >= 0 for i in ids), "Unique sorted row identity")
    result = {"ids": np.asarray(ids, np.int64), "labels": np.asarray(labels, np.int64),
              "previous": np.asarray(previous, np.int64), "types": np.asarray(kinds, np.int64),
              "held": np.asarray(held, bool)}
    result["mask"] = np.arange(width)[None, :] < np.asarray(sizes)[:, None]
    result["target_types"] = result["types"][np.arange(len(rows)), result["labels"]]
    return result


def normalize(raw, mask):
    require(raw.dtype == np.float32 and raw.shape == mask.shape and len(raw), "Float32 source shape")
    require(np.isfinite(raw[mask]).all() and np.all(raw[mask] <= 0)
            and np.isneginf(raw[~mask]).all(), "Supported finite logs and padded negative infinity")
    logs = raw.astype(np.float64)
    z = np.logaddexp.reduce(logs, axis=1)
    error = np.abs(np.expm1(z))
    require(float(error.max()) <= 2e-6, "Source probability mass")
    q = logs-z[:, None]
    require(np.array_equal(np.argmax(raw, axis=1), np.argmax(q, axis=1)), "Normalization choice identity")
    return q, z, {"maximum_raw_mass_error": float(error.max()), "max_abs_log_z": float(np.abs(z).max()),
                  "mean_log_z": math.fsum(map(float, z))/len(z),
                  "min_log_z": float(z.min()), "max_log_z": float(z.max())}


def hybrid(mass_logs, rank_logs, previous, mask, *, self_cell=False):
    require(mass_logs.shape == rank_logs.shape == mask.shape, "Hybrid shape")
    require(np.all(mask.sum(axis=1) >= 2), "An alternative is required")
    alternate = mask.copy()
    alternate[np.arange(len(previous)), previous] = False
    mass_change = np.logaddexp.reduce(np.where(alternate, mass_logs, -np.inf), axis=1)
    rank_change = np.logaddexp.reduce(np.where(alternate, rank_logs, -np.inf), axis=1)
    result = np.where(alternate, mass_change[:, None] + (rank_logs-rank_change[:, None]), -np.inf)
    result[np.arange(len(previous)), previous] = mass_logs[np.arange(len(previous)), previous]
    require(np.isfinite(result[mask]).all() and np.isneginf(result[~mask]).all(), "Hybrid support")
    drift = float(np.abs(np.expm1(np.logaddexp.reduce(result, axis=1))).max())
    require(drift <= 1e-12, "Hybrid normalization")
    if self_cell:
        require(np.allclose(result[mask], mass_logs[mask], atol=1e-10, rtol=1e-12), "Self reconstruction")
        result = mass_logs.copy()
    return result, drift


def primary_masks(meta):
    held, changed = meta["held"], meta["labels"] != meta["previous"]
    return {"all": held, "changed": held & changed, "retained": held & ~changed,
            "unmentioned_retention": held & ~changed & (meta["target_types"] == 0),
            "assigned_retention": held & ~changed & (meta["target_types"] != 0)}


def means(values, rows, mask):
    selected = np.flatnonzero(mask)
    if not len(selected):
        return {"row": None, "equal_service": None, "equal_dialogue": None}
    result = {"row": math.fsum(float(values[i]) for i in selected)/len(selected)}
    for name, key in (("equal_service", "service"), ("equal_dialogue", "dialogue_id")):
        buckets = {}
        for i in selected:
            buckets.setdefault(rows[i][key], []).append(float(values[i]))
        result[name] = math.fsum(math.fsum(v)/len(v) for v in buckets.values())/len(buckets)
    return result


def score(logs, meta):
    choice = np.argmax(logs, axis=1)
    idx = np.arange(len(choice))
    correct = choice == meta["labels"]
    selected_type = meta["types"][idx, choice]
    branch_wrong = np.minimum(selected_type, 2) != np.minimum(meta["target_types"], 2)
    value_wrong = ~correct & ~branch_wrong
    probability = np.exp(logs)
    probability[idx, meta["labels"]] -= 1
    brier = np.sum(probability*probability, axis=1)
    vectors = {"accuracy": correct.astype(float), "error": (~correct).astype(float),
               "wrong_selected_branch": branch_wrong.astype(float), "wrong_value": value_wrong.astype(float),
               "nll": -logs[idx, meta["labels"]], "brier": brier}
    ties = (logs == logs[idx, choice, None]).sum(axis=1) > 1
    require(np.array_equal(correct.astype(int)+branch_wrong+value_wrong, np.ones(len(choice))), "Decision partition")
    return choice, vectors, ties


def cell(rows, meta, choice, vectors, ties, mask):
    count = int(mask.sum())
    counts = {key: int(vectors[metric][mask].sum()) for key, metric in
              (("correct", "accuracy"), ("error", "error"), ("wrong_selected_branch", "wrong_selected_branch"),
               ("wrong_value", "wrong_value"))}
    require(counts["correct"]+counts["error"] == count
            and counts["wrong_selected_branch"]+counts["wrong_value"] == counts["error"], "Cell partition")
    rare = {}
    for label, code in (("true", 2), ("dontcare", 1)):
        for kind in ("recall", "false_positive"):
            eligible = mask & ((meta["target_types"] == code) if kind == "recall" else
                               ((meta["types"] == code).any(axis=1) & (meta["target_types"] != code)))
            event = choice == meta["labels"] if kind == "recall" else meta["types"][np.arange(len(rows)), choice] == code
            n, d = int((event & eligible).sum()), int(eligible.sum())
            rare[label+"_"+kind] = {"numerator": n, "denominator": d, "rate": n/d if d else None}
    return {"rows": count, "counts": counts, "rare": rare,
            "metrics": {key: means(value, rows, mask) for key, value in vectors.items()},
            "exact_tie_rows": int((ties & mask).sum())}


def paired(rows, target, candidate, control, av, bv, mask):
    ac, bc = candidate == target, control == target
    counts = {"rows": int(mask.sum()), "wrong_to_correct": int((mask & ~bc & ac).sum()),
              "correct_to_wrong": int((mask & bc & ~ac).sum()), "both_correct": int((mask & bc & ac).sum()),
              "both_wrong": int((mask & ~bc & ~ac).sum())}
    require(sum(v for k, v in counts.items() if k != "rows") == counts["rows"], "Paired partition")
    return {"paired": counts, "metric_differences": {key: means(av[key]-bv[key], rows, mask) for key in av}}


def compare(actual, expected, path="root"):
    """Compare requested fields only; other producer panels are outside scope."""
    if type(expected) is dict:
        require(type(actual) is dict and set(expected) <= set(actual), "Missing fields: " + path)
        return sum(compare(actual[k], v, path+"/"+k) for k, v in expected.items())
    if expected is None:
        require(actual is None, "Undefined support: " + path)
    elif type(expected) in (int, bool, str):
        require(type(actual) is type(expected) and actual == expected, "Exact mismatch: " + path)
    else:
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(actual, expected, abs_tol=1e-11, rel_tol=1e-10), "Numeric mismatch: " + path)
    return 1


def aggregate(rows, packets, producer):
    require(set(packets) == set(PREDICTION_PINS), "All nine input distributions")
    require(set(producer["fits"]) == {f"{c}-{s}" for c in CONSTRUCTIONS for s in SEEDS}, "All fifteen constructions")
    meta = metadata(rows)
    masks = primary_masks(meta)
    fits, comparisons, normalizations, validations = {}, {}, {}, {}
    checked = 0
    for seed in SEEDS:
        normalized = {}
        for method in METHODS:
            name = f"{method}-{seed}"
            packet = packets[name]
            require(set(packet) == {"row_indices", "log_probs"}, "Prediction NPZ schema")
            require(packet["row_indices"].dtype == np.int64
                    and np.array_equal(packet["row_indices"], meta["ids"]), "Prediction row identity")
            q, z, norm = normalize(packet["log_probs"], meta["mask"])
            normalized[method] = q
            norm["nll_normalized_minus_original_raw"] = {
                "heldout_service/"+key: means(z, rows, mask) for key, mask in masks.items()}
            normalizations[name] = norm
            checked += compare(producer["source_normalization"][name], norm, "normalization/"+name)
        output = {"flat": normalized["flat_stratum"]}
        sources = {"M": normalized["token_mean"], "A": normalized["token_aligned"]}
        for construction in ("MM", "AA", "MA", "AM"):
            output[construction], drift = hybrid(sources[construction[0]], sources[construction[1]],
                meta["previous"], meta["mask"], self_cell=construction[0] == construction[1])
            validations[f"{construction}-{seed}"] = {"maximum_mass_error": drift}
        choices, vectors, cells = {}, {}, {}
        for construction, logs in output.items():
            name = f"{construction}-{seed}"
            choice, values, ties = score(logs, meta)
            choices[construction], vectors[construction] = choice, values
            cells[construction] = {"heldout_service/"+key: cell(rows, meta, choice, values, ties, mask)
                                   for key, mask in masks.items()}
            fits[name] = {"cells": cells[construction]}
            checked += compare(producer["fits"][name], fits[name], "fits/"+name)
            maximum_mass_error = float(np.abs(np.expm1(np.logaddexp.reduce(logs, axis=1))).max())
            require(maximum_mass_error <= 1e-12, "Final normalized construction")
            checked += compare(producer["fits"][name]["validation"],
                               {"maximum_mass_error": maximum_mass_error, "exact_tie_rows": int(ties.sum())},
                               "validation/"+name)
        for construction in CONSTRUCTIONS:
            for control in ("flat", "MM", "AA"):
                pairs = {}
                for stratum, mask in masks.items():
                    key = "heldout_service/"+stratum
                    record = paired(rows, meta["labels"], choices[construction], choices[control],
                                    vectors[construction], vectors[control], mask)
                    a, b = cells[construction][key], cells[control][key]
                    record["rows"] = int(mask.sum())
                    record["count_differences"] = {k: a["counts"][k]-b["counts"][k] for k in a["counts"]}
                    record["rare_rate_differences"] = {
                        k: None if a["rare"][k]["rate"] is None else a["rare"][k]["rate"]-b["rare"][k]["rate"]
                        for k in a["rare"]}
                    pairs[key] = record
                comparisons.setdefault(construction, {}).setdefault(control, {"seeds": {}})["seeds"][str(seed)] = {"cells": pairs}
                checked += compare(producer["comparisons"][construction][control]["seeds"][str(seed)],
                                   {"cells": pairs}, f"comparisons/{construction}/{control}/{seed}")
    return {"fits": fits, "comparisons": comparisons, "source_normalization": normalizations,
            "independent_hybrid_validation": validations, "scalar_comparisons": checked,
            "counts": {"original_fits": 9, "constructions": 15, "primary_cells": 75, "paired_cells": 225,
                       "rows_per_fit": len(rows), "primary_support": {k: int(v.sum()) for k, v in masks.items()}}}


def authenticate(args):
    """Hash receipts and selected metric payloads before decoding arrays."""
    run, diagnostic = Path(args.run).resolve(), Path(args.diagnostic).resolve()
    require(args.run_sha256 == RUN_PIN and sha(run/"completed.json") == RUN_PIN, "External fixed run digest")
    require(sha(diagnostic/"summary.json") == args.summary_sha256, "External diagnostic summary digest")
    require(sha(diagnostic/"receipt.json") == args.receipt_sha256, "External diagnostic receipt digest")
    done, plan = read(run/"completed.json"), read(run/"plan.json")
    require(sha(run/"plan.json") == PLAN_PIN and done["plan_sha256"] == PLAN_PIN
            and sha(run/"evaluation-rows.jsonl") == ROWS_PIN, "Frozen plan/row pins")
    require(done["status"] == "completed" and set(done["completed_fits"]) == set(PREDICTION_PINS)
            and len(done["completed_fits"]) == 9, "All nine completed fits")
    names = {"started.json", "plan.json", "evaluation-rows.jsonl", "references.npz"}
    names |= {f"orders-{seed}.npy" for seed in SEEDS}
    names |= {f"fits/{name}/{file}" for name in PREDICTION_PINS
              for file in ("completed.json", "updates.jsonl", "weights.pt", "predictions.npz")}
    require(set(done["files"]) == names and len(names) == 43, "Fixed 44-file training closure")
    for name, record in done["files"].items():
        safe(run, name)
        require(set(record) == {"sha256", "bytes"} and type(record["bytes"]) is int and record["bytes"] >= 0
                and type(record["sha256"]) is str and len(record["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in record["sha256"]), "Training manifest record")
    require(plan["source_sha256"] == done["source_sha256"], "Bound training source identities")
    selected = {"plan.json", "evaluation-rows.jsonl"} | {f"fits/{name}/predictions.npz" for name in PREDICTION_PINS}
    bindings = {run/name: done["files"][name] for name in selected}
    for path, record in bindings.items():
        require(item(path) == record, "Selected metric payload: "+str(path))
    bindings[run/"completed.json"] = item(run/"completed.json")
    for name, pin in PREDICTION_PINS.items():
        require(done["files"][f"fits/{name}/predictions.npz"]["sha256"] == pin, "Fixed prediction identity")
    parent = ROOT/"output/dialogue-token-alignment-scientific-v1"
    for name, pin in PARENT_PINS.items():
        path = parent/name
        require(sha(path) == pin, "Previous sealed report/audit: "+name)
        bindings[path] = item(path)
    prior = read(parent/"audit-01/result-01/receipt.json")
    require(prior["status"] == "completed" and prior["agreement"] is True
            and prior["continuation_passed"] is False, "Previous primary audit outcome")
    receipt, producer = read(diagnostic/"receipt.json"), read(diagnostic/"summary.json")
    require(receipt["status"] == producer["status"] == "completed"
            and receipt["version"] == producer["version"] == "dialogue-commitment-diagnostic-v1"
            and receipt["diagnostic_has_gate"] is producer["diagnostic_has_gate"] is False,
            "Completed descriptive diagnostic without a gate")
    require(producer["original_campaign"] == {"passed": False, "checks_passed": 18, "total_checks": 22},
            "Original campaign remains failed")
    require(receipt["prediction_sha256"] == producer["prediction_sha256"] == PREDICTION_PINS,
            "Diagnostic prediction bindings")
    expected_pins = {"completed": RUN_PIN, "plan": PLAN_PIN, "rows": ROWS_PIN,
                     "report_receipt": PARENT_PINS["report-01/receipt.json"],
                     "report_summary": PARENT_PINS["report-01/summary.json"],
                     "audit_receipt": PARENT_PINS["audit-01/result-01/receipt.json"]}
    require(receipt["input_sha256"] == producer["input_sha256"] == expected_pins, "Diagnostic fixed inputs")
    require(set(receipt["files"]) == {"started.json", "summary.json"}, "Diagnostic payload names")
    manifest(diagnostic, receipt["files"], terminal="receipt.json")
    bindings.update({diagnostic/name: record for name, record in receipt["files"].items()})
    bindings[diagnostic/"receipt.json"] = item(diagnostic/"receipt.json")
    started = read(diagnostic/"started.json")
    require(receipt["source_sha256"] == producer["source_sha256"] == started["source_sha256"], "Diagnostic source maps")
    for name, pin in receipt["source_sha256"].items():
        source = safe(ROOT, name)
        require(sha(source) == pin, "Diagnostic live source: "+name)
        bindings[source] = item(source)
    require(receipt["model_calls"] == receipt["encoder_calls"] == receipt["checkpoint_deserializations"] == 0,
            "Recorded saved-only diagnostic scope")
    # Metadata is decoded only after every input member has been authenticated.
    rows = [decode(line) for line in (run/"evaluation-rows.jsonl").read_text().splitlines()]
    meta = metadata(rows)
    require(len(rows) == 13599 and meta["ids"].tolist() == plan["evaluation_row_indices"], "Canonical evaluation cohort")
    require({k: int(m.sum()) for k, m in primary_masks(meta).items()} ==
            {"all": 7819, "changed": 578, "retained": 7241,
             "unmentioned_retention": 4032, "assigned_retention": 3209}, "Fixed primary support")
    return rows, producer, bindings


def peak_rss():
    amount = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(amount if sys.platform == "darwin" else amount*1024)


def execute(args):
    out = Path(args.out).resolve()
    for source in (args.run, args.diagnostic):
        parent = Path(source).resolve()
        require(not out.is_relative_to(parent) and not parent.is_relative_to(out), "Separate audit output")
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    request = {k: str(v) for k, v in vars(args).items()}
    sources = {}
    prior_handler = None

    def check():
        require(time.monotonic()-start <= LIMITS["wall_seconds"], "Audit wall limit")
        require(peak_rss() <= LIMITS["rss_bytes"], "Audit process-lifetime RSS limit")
        require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= LIMITS["output_bytes"], "Audit output limit")

    def timeout(_sig, _frame):
        raise TimeoutError("Independent audit 60-second wall limit")
    try:
        sources.update({"scripts/audit_dialogue_commitment.py": sha(__file__),
                        "tests/test_audit_dialogue_commitment.py": sha(ROOT/"tests/test_audit_dialogue_commitment.py")})
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "No overlapping timer")
        prior_handler = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        write(out/"started.json", {"version": VERSION, "request": request, "source_sha256": sources,
                                  "limits": LIMITS, "scope": SCOPE})
        check()
        rows, producer, bindings = authenticate(args)
        check()
        packets = {}
        for name in PREDICTION_PINS:
            with np.load(Path(args.run)/"fits"/name/"predictions.npz", allow_pickle=False) as archive:
                require(len(archive.files) == 2 and set(archive.files) == {"row_indices", "log_probs"}, "NPZ exact fields")
                packets[name] = {key: archive[key] for key in archive.files}
            check()
        result = aggregate(rows, packets, producer)
        check()
        result.update(status="completed", agreement=True, version=VERSION, scope=SCOPE,
                      execution_completed_sha256=RUN_PIN, producer_summary_sha256=args.summary_sha256,
                      producer_receipt_sha256=args.receipt_sha256, original_campaign_passed=False,
                      model_calls=0, encoder_calls=0, checkpoint_deserializations=0)
        write(out/"summary.json", result)
        for path, bound in bindings.items():
            require(item(path) == bound, "End input stability: "+str(path))
        for name, pin in sources.items():
            require(sha(ROOT/name) == pin, "End audit source stability")
        check()
        files = {name: item(out/name) for name in ("started.json", "summary.json")}
        write(out/"receipt.json", {"status": "completed", "agreement": True, "version": VERSION,
            "execution_completed_sha256": RUN_PIN, "producer_summary_sha256": args.summary_sha256,
            "producer_receipt_sha256": args.receipt_sha256, "source_sha256": sources["scripts/audit_dialogue_commitment.py"],
            "test_source_sha256": sources["tests/test_audit_dialogue_commitment.py"],
            "request": request, "files": files, "input_members": {str(p): v for p, v in bindings.items()},
            "scalar_comparisons": result["scalar_comparisons"], "scope": SCOPE, "limits": LIMITS,
            "wall_seconds": time.monotonic()-start, "process_lifetime_peak_rss_bytes": peak_rss(),
            "wall_scope": "Through input stability and payload hashing; terminal write remains cap-checked",
            "model_calls": 0, "encoder_calls": 0, "checkpoint_deserializations": 0})
        check()
        return result
    except BaseException as error:
        if prior_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/"receipt.json").exists():
                (out/"receipt.json").rename(out/"receipt-before-error.json")
            write(out/"failed.json", {"status": "failed", "version": VERSION, "request": request,
                "source_sha256": sources, "error": repr(error), "scope": SCOPE,
                "wall_seconds": time.monotonic()-start, "model_calls": 0, "no_retry": True})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original audit failure
            if callable(getattr(error, "add_note", None)):
                error.add_note("Audit failure preservation: "+repr(secondary))
        raise
    finally:
        if prior_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prior_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "run-sha256", "diagnostic", "summary-sha256", "receipt-sha256", "out"):
        parser.add_argument("--"+name, required=True)
    execute(parser.parse_args())
