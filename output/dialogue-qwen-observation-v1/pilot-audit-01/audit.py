"""Single saved-only pilot audit. No model/tokenizer/label/quality access."""
import hashlib
import json
import math
import resource
import signal
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PILOT = ROOT / "output/dialogue-qwen-observation-v1/pilot-01"
PREP = ROOT / "output/dialogue-qwen-observation-v1/preparation-02"
PILOT_SHA = "3b8bcf6a4e1fb27322e6260e4cdc10b0514b9db72c98760e5a7d04284137f54d"
PLAN_SHA = "2d5f7e6b512ae7260cc01685ae03220891ad4ca5236092645197d4074be90111"
PREP_SHA = "6a7a8283efa612866a0a9f0c2bcce92bb54e9ce8982bde26baaa8d8aeb24eb9f"
PREP_AUDIT_SHA = "ab3fdc632e68ea739a8c9ecb0109f1abe48fecbfb4302599327c15a5173df1ab"
ARMS = ("current", "history4")
START = time.monotonic()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check():
    require(time.monotonic() - START <= 60, "Audit wall cap")
    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    require(memory * (1 if sys.platform == "darwin" else 1024) <= 2 * 1024**3, "Audit RSS cap")
    require(sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) <= 128 * 1024**2, "Audit output cap")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            h.update(block)
            check()
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    with (OUT / name).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def same(a, b, message):
    require(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-12), message)


def run():
    require(sha(PILOT / "completed.json") == PILOT_SHA, "External pilot completion")
    require(sha(PREP / "plan.json") == PLAN_SHA and sha(PREP / "completed.json") == PREP_SHA, "External preparation pins")
    prior_path = ROOT / "output/dialogue-qwen-observation-v1/preparation-audit-01/receipt.json"
    require(sha(prior_path) == PREP_AUDIT_SHA, "Independent actor audit receipt")
    prior = read(prior_path)
    require(prior["agreement"] is True and prior["status"] == "completed"
            and prior["plan_sha256"] == PLAN_SHA and prior["prepared_completed_sha256"] == PREP_SHA, "Actor audit scope")
    plan, done = read(PREP / "plan.json"), read(PILOT / "completed.json")
    require(set(done["files"]) == {"started.json", "plan.json", "timings.jsonl"}
            and {p.name for p in PILOT.iterdir()} == set(done["files"]) | {"completed.json"}, "Pilot exact closure")
    for name, entry in done["files"].items():
        path = PILOT / name
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], "Pilot payload hash: " + name)
    require(sha(PILOT / "plan.json") == PLAN_SHA, "Copied plan")
    start = read(PILOT / "started.json")
    require(done["status"] == "completed" and done["phase"] == start["phase"] == "pilot"
            and done["version"] == start["version"] == plan["version"] == "dialogue-qwen-observation-v1"
            and done["plan_sha256"] == start["request"]["plan_sha256"] == PLAN_SHA
            and done["source_sha256"] == plan["source_sha256"] and len(done["source_sha256"]) == 9
            and done["model"] == plan["model"] and done["runtime"] == plan["runtime"], "Phase/source/model lineage")
    require(done["labels_accessed"] is start["labels_accessed"] is False
            and done["quality_outputs_saved"] is False and done["no_retry"] is start["no_retry"] is True
            and done["generated_tokens"] == done["optimizer_updates"] == 0, "Pilot no-quality scope")
    limits = {"wall_seconds": 300, "rss_bytes": 12 * 1024**3, "output_bytes": 512 * 1024**2}
    require(done["limits"] == start["limits"] == plan["limits"]["pilot"] == limits
            and 0 < done["wall_seconds"] <= 300 and 0 < done["process_lifetime_peak_rss_bytes"] <= limits["rss_bytes"]
            and sum(p.stat().st_size for p in PILOT.iterdir()) <= limits["output_bytes"], "Pilot recorded caps")
    for name, digest in plan["source_sha256"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and sha(ROOT / name) == digest, "Frozen source identity")
    request_path = PREP / "requests.jsonl"
    require(request_path.stat().st_size == plan["files"]["requests.jsonl"]["bytes"]
            and sha(request_path) == plan["files"]["requests.jsonl"]["sha256"], "Authenticated requests")
    metadata, counts, slots = [], Counter(), Counter()
    with request_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            lengths = [len(t) for t in row["tokens"]]
            n, longest = len(lengths), max(lengths)
            work = {"calls": 1, "prefill_calls": 0, "branch_calls": 1, "input_token_slots": n * longest,
                    "prefix_tokens": 0, "padding_token_slots": n * longest - sum(lengths)}
            metadata.append({"request_id": row["request_id"], "arm": row["arm"], "dialogue_id": row["dialogue_id"],
                             "time": row["time"], "questions": n, "max_tokens": longest, "work": work})
            counts[row["arm"]] += 1
            slots[row["arm"]] += n * longest
    require(dict(counts) == plan["request_counts"] and dict(slots) == plan["input_token_slots"], "Full request work")
    groups = set(sorted({(r["dialogue_id"], r["time"]) for r in metadata})[:12])
    for arm in ARMS:
        longest = min((r for r in metadata if r["arm"] == arm),
                      key=lambda r: (-r["max_tokens"], r["dialogue_id"], r["time"], r["request_id"]))
        groups.add((longest["dialogue_id"], longest["time"]))
    selected = [r for r in metadata if (r["dialogue_id"], r["time"]) in groups]
    require(len(selected) == 28 and [r["request_id"] for r in selected] == plan["pilot_request_ids"], "All 28 fixed pilot requests")
    timings = [json.loads(line) for line in (PILOT / "timings.jsonl").read_text().splitlines()]
    require(len(timings) == 28, "Timing count")
    numerical_max = 0.0
    for recorded, expected in zip(timings, selected, strict=True):
        require(all(recorded[k] == expected[k] for k in ("request_id", "arm", "questions", "work")), "Exact selected work/order")
        require(recorded["finite_logits"] is True and math.isfinite(recorded["seconds"]) and recorded["seconds"] > 0
                and 0 <= recorded["maximum_probability_mass_error"] <= 2e-12, "Recorded numerical validity")
        require(type(recorded["serialized_score_bytes"]) is int and recorded["serialized_score_bytes"] > 0
                and all(type(recorded[k]) is int and recorded[k] >= 0
                        for k in ("mlx_peak_active_bytes", "mlx_cached_allocator_bytes")), "Output/memory witnesses")
        numerical_max = max(numerical_max, recorded["maximum_probability_mass_error"])
    totals = {k: sum(r["work"][k] for r in selected) for k in selected[0]["work"]}
    questions = sum(r["questions"] for r in selected)
    require(done["model_calls"] == 28 and done["work_totals"] == totals and done["progress"] == {
        "requests_completed": 28, "questions_completed": questions, "forward_calls_attempted": 28,
        "forward_calls_returned": 28, "active_request_id": None}, "Complete paid work")
    same(done["request_seconds"], math.fsum(t["seconds"] for t in timings), "Request timing sum")
    require(0 <= done["model_load_seconds"] and done["request_seconds"] + done["model_load_seconds"] <= done["wall_seconds"], "Timing scopes")
    components, token_total, request_total = {}, 0., 0.
    for arm in ARMS:
        records = [r for r in timings if r["arm"] == arm]
        rate = max(r["seconds"] / r["work"]["input_token_slots"] for r in records)
        per_request = max(r["seconds"] for r in records)
        token_value, request_value = rate * slots[arm], per_request * counts[arm]
        components[arm] = {"max_seconds_per_charged_slot": rate, "max_seconds_per_request": per_request,
                           "token_scaled_seconds": token_value, "request_scaled_seconds": request_value,
                           "variable_seconds": token_value}
        for key, value in components[arm].items():
            same(done["projection"]["arms"][arm][key], value, "Projection component")
        token_total += token_value
        request_total += request_value
    expected = math.ceil(2 * token_total + done["wall_seconds"] + 60)
    descriptive = math.ceil(2 * request_total + done["wall_seconds"] + 60)
    projection = done["projection"]
    require(projection["factor"] == 2 and projection["fixed_seconds"] == 60
            and projection["pilot_whole_wall_seconds"] == done["wall_seconds"]
            and projection["admission_limit_seconds"] == plan["limits"]["run"]["wall_seconds"] == 7200
            and projection["projected_full_seconds"] == expected
            and projection["request_scaled_projection_seconds_descriptive_only"] == descriptive
            and projection["admitted"] is (expected <= 7200), "Frozen projection/admission rule")
    require(sha(PILOT / "completed.json") == PILOT_SHA and sha(PREP / "plan.json") == PLAN_SHA, "Terminal pins unchanged")
    return {"status": "completed", "agreement": True, "pilot_completed_sha256": PILOT_SHA, "plan_sha256": PLAN_SHA,
            "preparation_audit_receipt_sha256": PREP_AUDIT_SHA, "requests": 28, "questions": questions,
            "pilot_groups": len(groups), "work_totals": totals, "pilot_wall_seconds": done["wall_seconds"],
            "pilot_request_seconds": done["request_seconds"], "model_load_seconds": done["model_load_seconds"],
            "process_lifetime_peak_rss_bytes": done["process_lifetime_peak_rss_bytes"],
            "maximum_recorded_probability_mass_error": numerical_max, "projected_full_seconds": expected,
            "request_scaled_seconds_descriptive_only": descriptive, "admitted": expected <= 7200,
            "components": components, "model_calls": 0, "tokenizer_calls": 0, "labels_accessed": False,
            "quality_scored": False, "scope": "Independent saved-file identities, request/work coverage and projection arithmetic. "
            "Finite logits, measured time/RSS and model execution are authenticated producer witnesses, not replayed. "
            "The token-rate projection is heuristic; the higher request-rate diagnostic does not change the frozen admission rule."}


if __name__ == "__main__":
    require(not any((OUT / n).exists() for n in ("started.json", "receipt.json", "failed.json")), "Exclusive audit invocation")
    def alarm(*_):
        raise TimeoutError("Audit wall cap")
    old = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        source_hash = sha(Path(__file__))
        write("started.json", {"audit_source_sha256": source_hash, "pilot_completed_sha256": PILOT_SHA,
              "plan_sha256": PLAN_SHA, "limits": {"wall_seconds": 60, "rss_bytes": 2 * 1024**3, "output_bytes": 128 * 1024**2}})
        summary = run()
        write("summary.json", summary)
        check()
        write("receipt.json", {"status": "completed", "agreement": True, "audit_source_sha256": source_hash,
              "pilot_completed_sha256": PILOT_SHA, "summary_sha256": sha(OUT / "summary.json"),
              "wall_seconds": time.monotonic() - START, "model_calls": 0, "tokenizer_calls": 0,
              "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file()}})
        check()
        print(json.dumps({"receipt_sha256": sha(OUT / "receipt.json"), "projected_full_seconds": summary["projected_full_seconds"],
                          "admitted": summary["admitted"]}))
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (OUT / "receipt.json").exists():
                (OUT / "receipt.json").rename(OUT / "late-receipt.json")
            write("failed.json", {"status": "failed", "error": repr(exc), "wall_seconds": time.monotonic() - START})
        except BaseException as secondary:  # noqa: BLE001 - preserve original terminal error
            if callable(getattr(exc, "add_note", None)):
                exc.add_note("Failure receipt: " + repr(secondary))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
