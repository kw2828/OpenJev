"""Descriptive V2 saved-choice diagnosis, outside the frozen thirteen checks.

Historical extraction/count definitions are reused from exact pinned local
source. No model, encoder, training module, RNG or checkpoint loader is imported.
Execution requires terminal input hashes supplied by the caller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import time
import traceback
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OLD_PATH = "output/dialogue-copy-error-diagnostic-v1/analyze.py"
OLD_SHA = "7f23872436fde9049ebb181a26c55627aa10cb4b26f88e41b194e486fdfe2642"
METHODS = ("readout", "scalar", "selective", "selective_no_lexical", "candidate_gru")
SEEDS = (4101, 4102, 4103)
PANELS = ("seen", "unseen")
GROUPS = ("all", "assigned_boolean", "assigned_boolean/true", "assigned_boolean/false",
          "label/dontcare", "bin/revision")
SOURCES = {
    "scripts/study_dialogue_copy.py", "scripts/prepare_dialogue_copy.py", "scripts/report_dialogue_copy.py",
    "src/openjev/research/dialogue_copy_features.py", "src/openjev/research/dialogue_copy_memory.py",
    "tests/test_dialogue_copy_features.py", "tests/test_dialogue_copy_memory.py", "tests/test_study_dialogue_copy.py",
    "tests/test_report_dialogue_copy.py", "research/dialogue-copy-protocol.md", "scripts/study_dialogue_memory.py",
    "scripts/report_dialogue_memory.py", "src/openjev/research/dialogue_carry.py",
    "src/openjev/research/dialogue_copy_memory_v2.py", "tests/test_dialogue_copy_memory_v2.py",
    "scripts/study_dialogue_copy_v2.py", "tests/test_study_dialogue_copy_v2.py",
    "scripts/report_dialogue_copy_v2.py", "tests/test_report_dialogue_copy_v2.py", "research/dialogue-copy-v2-protocol.md",
}
GENERATED = ("started.json", "diagnostic.json", "README.md", "receipt.json", "failed.json")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)

    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def safe(root, name):
    require(type(name) is str and name and not Path(name).is_absolute() and ".." not in Path(name).parts,
            "Unsafe relative member")
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), "Member escapes root")
    return path


def bind(path, expected, inventory):
    path = Path(path)
    require(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected), "Invalid external digest")
    require(path.is_file() and not path.is_symlink() and sha(path) == expected, "Input hash mismatch: " + str(path))
    inventory[str(path.resolve())] = expected


def historical(root=ROOT):
    path = root / OLD_PATH
    source = path.read_bytes()
    require(hashlib.sha256(source).hexdigest() == OLD_SHA, "Historical diagnostic source changed")
    module = types.ModuleType("pinned_historical_dialogue_diagnostic")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102 - exact pinned local source, main guard inactive
    return module


def execution_members(run):
    expected = {"plan.json", "started.json", "completed.json", "references.npz"} | {
        f"fits/{method}-{seed}/{name}" for method in METHODS for seed in SEEDS
        for name in ("completed.json", "weights.pt", "dev-predictions.npz", "batches.jsonl")}
    paths = list(run.rglob("*"))
    require(not any(p.is_symlink() for p in paths)
            and {p.relative_to(run).as_posix() for p in paths if p.is_file()} == expected,
            "Expected exact 64-file completed execution")
    return expected


def bind_files(directory, files, inventory):
    require(type(files) is dict and files, "Missing payload manifest")
    for name, entry in files.items():
        path = safe(directory, name)
        if isinstance(entry, dict):
            require(set(entry) == {"sha256", "bytes"} and type(entry["bytes"]) is int
                    and path.stat().st_size == entry["bytes"], "Payload size/schema")
            entry = entry["sha256"]
        bind(path, entry, inventory)


def authenticate(args, inventory):
    run, report = Path(args.run), Path(args.report)
    packet, lexical = Path(args.packet), Path(args.lexical)
    for path, digest in ((run / "plan.json", args.plan_sha256), (run / "completed.json", args.completed_sha256),
                         (report / "receipt.json", args.report_receipt_sha256),
                         (packet / "completed.json", args.packet_completed_sha256),
                         (lexical / "completed.json", args.lexical_completed_sha256)):
        bind(path, digest, inventory)
    plan, done, receipt = read(run / "plan.json"), read(run / "completed.json"), read(report / "receipt.json")
    require(plan["study"] == done["study"] == receipt["study"] == "dialogue-copy-v2"
            and done["status"] == receipt["status"] == "completed", "Incomplete or wrong study")
    require(done["plan_sha256"] == receipt["plan_sha256"] == args.plan_sha256
            and receipt["execution_completed_sha256"] == args.completed_sha256, "Report/execution identity")
    require(plan["implementation_version"] == "dialogue-copy-memory-v2-normalized"
            and done["fit_count"] == receipt["fit_count"] == 15 and done["resume_authorized"] is False
            and receipt["technical_validity_passed"] is True, "Incomplete normalized replication")
    require(plan["packet_completed_sha256"] == receipt["feature_packet_completed_sha256"] == args.packet_completed_sha256
            and plan["lexical_completed_sha256"] == receipt["lexical_completed_sha256"] == args.lexical_completed_sha256,
            "Packet/lexical identity")
    require(safe(ROOT, plan["paths"]["packet"]).resolve() == packet.resolve()
            and safe(ROOT, plan["paths"]["lexical"]).resolve() == lexical.resolve(), "Input paths differ from plan")
    require(set(plan["source_sha256"]) == SOURCES and receipt["source_sha256"] == plan["source_sha256"], "Source closure")
    for name, digest in plan["source_sha256"].items():
        bind(safe(ROOT, name), digest, inventory)
    require(set(done["files"]) == execution_members(run) - {"completed.json"}, "Execution manifest closure")
    bind_files(run, done["files"], inventory)
    require(set(receipt["files"]) == {"started.json", "summary.json", "report.md", "comparison.png"}, "Report manifest closure")
    require(not any((report / name).exists() for name in ("failed.json", "late-completion.json", "cleanup-error.json")),
            "Failed report marker")
    bind_files(report, receipt["files"], inventory)
    summary = read(report / "summary.json")
    require(summary["source_sha256"] == plan["source_sha256"] and summary["technical_validity"]["passed"] is True,
            "Summary identity/technical validity")
    gate = summary["continuation_gate"]
    require(gate["checks_total"] == receipt["checks_total"] == 13
            and gate["checks_passed"] == receipt["checks_passed"]
            and gate["passed"] == receipt["continuation_passed"], "Original gate receipt differs")
    inputs = {}
    for directory, name, members in ((packet, "packet", {"encoder-plan.json", "encoder-source.py", "features.npy", "packet.json"}),
                                     (lexical, "lexical", {"started.json", "index.json", "lexical.npy"})):
        require(not (directory / "failed.json").exists(), "Failed input packet")
        item = read(directory / "completed.json")
        require(item["status"] == "completed" and set(item["files"]) == members, "Input packet membership")
        bind_files(directory, item["files"], inventory)
        inputs[name] = item
    require(inputs["lexical"]["packet_completed_sha256"] == args.packet_completed_sha256, "Lexical source packet")
    return plan, done, summary, inputs


def diagnostic_groups(packet, truth, evidence, old):
    """Keep the old subgroup definitions; split assigned booleans by ontology value."""
    all_masks = old.groups(truth, evidence)
    canonical = np.asarray(["" if y < 2 else packet["queries"][q]["candidate_values"][y].strip().casefold()
                            for q, y in zip(truth["query"], truth["labels"], strict=True)])
    result = {}
    for panel in PANELS:
        for name in GROUPS:
            key = panel + "/" + name
            if name in ("assigned_boolean/true", "assigned_boolean/false"):
                result[key] = all_masks[panel + "/assigned_boolean"] & (canonical == name.rsplit("/", 1)[1])
            else:
                result[key] = all_masks[key]
        require(np.array_equal(result[panel + "/assigned_boolean/true"] | result[panel + "/assigned_boolean/false"],
                               result[panel + "/assigned_boolean"]), "Boolean true/false partition")
    return result


def measure(choice, mask, truth, evidence, old):
    result = old.measure(choice, mask, truth, evidence)
    n = result["count"]
    result["literal_correct"] = result["both_correct"] + result["literal_only"]
    result["literal_accuracy"] = result["literal_correct"] / n if n else None
    result["accuracy_difference_from_literal_pp"] = 100 * (result["neural_only"] - result["literal_only"]) / n if n else None
    return result


def validate_saved(path, truth, evidence, *, reference=False):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    extras = {"pred_none", "pred_literal"} if reference else {"choice", "probabilities"}
    require(set(arrays) == set(truth) | extras, "Saved prediction field closure")
    n, counts = len(truth["labels"]), evidence["candidate_count"]
    for name, expected in truth.items():
        kind = "b" if name == "unseen" else "iu" if name in ("labels", "time", "query") else "U"
        require(arrays[name].dtype.kind in kind and arrays[name].shape == (n,)
                and np.array_equal(arrays[name], expected), "Saved row identity: " + name)
    if reference:
        require(arrays["pred_none"].shape == arrays["pred_literal"].shape == (n,)
                and arrays["pred_none"].dtype.kind in "iu" and arrays["pred_literal"].dtype.kind in "iu"
                and (arrays["pred_none"] == 0).all()
                and np.array_equal(arrays["pred_literal"], evidence["literal"]), "Literal/reference alignment")
        return arrays["pred_literal"]
    p, choice = arrays["probabilities"], arrays["choice"]
    require(p.shape == (n, 12) and p.dtype.kind == "f" and np.isfinite(p).all()
            and ((p >= 0) & (p <= 1)).all() and np.all(np.abs(p.astype(np.float64).sum(1) - 1) <= 1e-5)
            and (p[np.arange(12)[None] >= counts[:, None]] == 0).all(), "Invalid saved probabilities")
    require(choice.shape == (n,) and choice.dtype.kind in "iu" and ((choice >= 0) & (choice < counts)).all()
            and np.array_equal(choice, p.argmax(1)), "Saved choice mismatch")
    return choice


def pool(rows, masks):
    families = {}
    for method in METHODS:
        families[method] = {}
        for name in masks:
            values = [rows[f"{method}-{seed}"][name] for seed in SEEDS]
            require(len({v["count"] for v in values}) == 1, "Unpaired subgroup support")
            families[method][name] = {"count_per_fit": values[0]["count"], "fits": 3,
                **{key: None if any(v[key] is None for v in values) else math.fsum(v[key] for v in values) / 3
                   for key in values[0] if key != "count"}}
    return families


def render(result):
    lines = ["# Normalized dialogue-copy error diagnosis", "",
             ("Descriptive saved-output analysis on already exposed development data. No model or encoder calls. "
             "The frozen thirteen-check rule is unchanged; this diagnostic selects no new winner."), "",
             ("Accuracy and count summaries below average all three fits per method on the same examples. "
             "Literal carry is one deterministic reference. Exact per-seed counts and literal-only/neural-only "
             "partitions are in diagnostic.json. A dash means zero denominator."), "",
             "| Panel | Method | Boolean n | Boolean % | True % | False % | DONTCARE n | DONTCARE % | Boolean vs literal pp |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]

    def number(value, scale=1):
        return "-" if value is None else f"{scale * value:.2f}"

    for panel in PANELS:
        for method in (*METHODS, "literal"):
            group = result["literal"] if method == "literal" else result["families"][method]
            b, t, f, d = [group[panel + "/" + name] for name in
                          ("assigned_boolean", "assigned_boolean/true", "assigned_boolean/false", "label/dontcare")]
            count_key = "count" if method == "literal" else "count_per_fit"
            lines.append(f"| {panel} | {method} | {b[count_key]} | {number(b['accuracy'], 100)} | "
                         f"{number(t['accuracy'], 100)} | {number(f['accuracy'], 100)} | {d[count_key]} | "
                         f"{number(d['accuracy'], 100)} | {number(b['accuracy_difference_from_literal_pp'])} |")
    lines += ["", "Stale means the wrong choice equals previous scored gold, not proof of an internal carry operation.", "",
              "| Panel | Method | Revision n | Correct | Stale | Other wrong | Stale after correct prior | Revision vs literal pp |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for panel in PANELS:
        for method in (*METHODS, "literal"):
            group = result["literal"] if method == "literal" else result["families"][method]
            r = group[panel + "/bin/revision"]
            lines.append(f"| {panel} | {method} | {r['revision_count']:.0f} | {r['revision_correct']:.2f} | "
                         f"{r['revision_stale_prior_gold']:.2f} | {r['revision_other_wrong']:.2f} | "
                         f"{r['revision_stale_after_prior_correct']:.2f} | {number(r['accuracy_difference_from_literal_pp'])} |")
    lines += ["", "## Limits", ""] + ["- " + item for item in result["limits"]]
    return "\n".join(lines) + "\n"


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    require(not any((out / name).exists() for name in GENERATED), "Diagnostic attempt already exists; no retry")
    started, inventory, progress = time.perf_counter(), {}, {"stage": "authentication", "fits_checked": []}
    try:
        write(out / "started.json", {"status": "started", "source_sha256": args.source_sha256,
            "tests_sha256": args.tests_sha256, "historical_definition_source_sha256": OLD_SHA,
            "plan_sha256": args.plan_sha256, "execution_completed_sha256": args.completed_sha256,
            "report_receipt_sha256": args.report_receipt_sha256,
            "packet_completed_sha256": args.packet_completed_sha256, "lexical_completed_sha256": args.lexical_completed_sha256,
            "scope": "Descriptive saved-output only; no new scientific criterion"})
        bind(Path(__file__), args.source_sha256, inventory)
        bind(ROOT / OLD_PATH, OLD_SHA, inventory)
        tests = Path(__file__).with_name("test_analyze.py")
        bind(tests, args.tests_sha256, inventory)
        plan, done, summary, inputs = authenticate(args, inventory)
        old = historical()
        packet = read(Path(args.packet) / "packet.json")
        index = read(Path(args.lexical) / "index.json")
        lexical = np.load(Path(args.lexical) / "lexical.npy", allow_pickle=False, mmap_mode="r")
        require(lexical.dtype == np.float32 and lexical.ndim == 1
                and lexical.size == inputs["lexical"]["float32_scalars"], "Lexical array schema")
        truth, evidence = old.extract(packet, index, lexical)
        n = len(truth["labels"])
        require(n == 62329 == inputs["packet"]["cohorts"]["dev"]["queries"], "Incomplete scored cohort")
        run_path = Path(args.run)
        require(done["references"]["sha256"] == done["files"]["references.npz"]["sha256"], "Reference binding")
        literal_choice = validate_saved(run_path / "references.npz", truth, evidence, reference=True)
        masks = diagnostic_groups(packet, truth, evidence, old)
        rows = {}
        names = [f"{method}-{seed}" for seed in SEEDS for method in METHODS]
        require([f"{fit['method']}-{fit['seed']}" for fit in done["fits"]] == names, "Fit membership/order")
        progress["stage"] = "saved_choices"
        for name, record in zip(names, done["fits"], strict=True):
            folder = run_path / "fits" / name
            require(record["status"] == "completed" and read(folder / "completed.json") == record
                    and record["predictions_sha256"] == done["files"][f"fits/{name}/dev-predictions.npz"]["sha256"],
                    "Per-fit binding")
            choice = validate_saved(folder / "dev-predictions.npz", truth, evidence)
            rows[name] = {key: measure(choice, mask, truth, evidence, old) for key, mask in masks.items()}
            for panel in PANELS:
                current = rows[name][panel + "/all"]
                reported = summary["rows"][name]["metrics"][panel]["micro"]
                require(current["count"] == reported["count"] and current["correct"] == reported["correct"]
                        and current["accuracy"] == reported["accuracy"], "Reported micro accuracy differs")
            progress["fits_checked"].append(name)
        result = {"status": "completed", "study": "dialogue-copy-v2-error-diagnostic", "posthoc": True,
            "scientific_gate_changed": False, "historical_definitions_sha256": OLD_SHA,
            "source_sha256": args.source_sha256, "tests_sha256": args.tests_sha256,
            "input_sha256": inventory, "counts": {"fits": 15, "queries_per_fit": n, "saved_prediction_rows": 15*n,
                "scored_rows_with_intervening_unscored_steps": int((evidence["gap"] > 1).sum()),
                "new_model_calls": 0, "new_encoder_calls": 0, "checkpoint_deserializations": 0},
            "group_support": {key: int(mask.sum()) for key, mask in masks.items()},
            "per_fit": rows, "families": pool(rows, masks),
            "literal": {key: measure(literal_choice, mask, truth, evidence, old) for key, mask in masks.items()},
            "unchanged_continuation_gate": {key: summary["continuation_gate"][key] for key in ("passed", "checks_passed", "checks_total")},
            "definitions": {
                "boolean": "Ontology values after strip/casefold form exactly {true,false}; assigned means gold index>=2. True/false use ontology strings, not indices or cues.",
                "dontcare": "Reserved gold index1 for all schemas, including boolean schemas; disjoint from assigned true/false.",
                "revision": "Current non-NONE gold differs from previous non-NONE scored gold for the same dialogue/question. DONTCARE is an assigned value here.",
                "stale": "Wrong current choice equals previous scored gold, irrespective of whether prior prediction was correct.",
                "prior_correct": "Previous saved prediction for the same question equals its previous scored gold; absence of scored frames is not an internal state observation.",
                "literal_difference": "100*(neural-only correct minus literal-only correct)/subgroup count; equal-three-seed mean for family rows.",
                "family_counts": "Mean per fit across three seeds on identical examples, not independent repeated data; exact per-fit counts retained."},
            "limits": [
                "Exposed development only; this is descriptive subgroup analysis outside the unchanged thirteen-check rule, not a new winner or continuation test.",
                "Historical extraction and count functions are reused by exact source hash. This is comparable arithmetic, not an independent reimplementation of those definitions.",
                "No neural execution or hidden-state reconstruction. Internal normalization is authenticated by the completed V2 report, not reverified from saved choices.",
                "Stale predictions and poor boolean/DONTCARE accuracy are error patterns, not causal proof of memory inertia, lost encoder information, or failed semantic understanding.",
                "Lexical USER/SYSTEM matches are pinned saved features, not intent labels. SYSTEM text may ask about or reject a value. Raw dialogue text is not reprocessed.",
                "The prior scored frame can be separated by public steps. No intermediate predictions are invented, and the three seeds share the same examples.",
                "The original V1 normalization defect compromises operator comparisons. No V1-to-V2 subgroup improvement is attributed to a representation or architecture change here."],
            "runtime": {"python": platform.python_version(), "numpy": str(np.__version__)}}
        progress["stage"] = "final_authentication"
        for path, digest in tuple(inventory.items()):
            bind(Path(path), digest, inventory)
        execution_members(run_path)
        write(out / "diagnostic.json", result)
        with (out / "README.md").open("x") as stream:
            stream.write(render(result))
        write(out / "receipt.json", {"status": "completed", "posthoc": True, "scientific_gate_changed": False,
            "source_sha256": args.source_sha256, "tests_sha256": args.tests_sha256, "input_sha256": inventory,
            "plan_sha256": args.plan_sha256, "execution_completed_sha256": args.completed_sha256,
            "report_receipt_sha256": args.report_receipt_sha256, "packet_completed_sha256": plan["packet_completed_sha256"],
            "lexical_completed_sha256": plan["lexical_completed_sha256"],
            "files": {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size}
                      for name in ("started.json", "diagnostic.json", "README.md")},
            "wall_seconds": time.perf_counter() - started, "new_model_calls": 0, "new_encoder_calls": 0,
            "checkpoint_deserializations": 0})
        return result
    except BaseException as error:
        try:
            write(out / "failed.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(), "progress": progress, "authenticated_inputs": inventory,
                "wall_seconds": time.perf_counter() - started})
        except BaseException as secondary:  # noqa: BLE001 - retain original mismatch
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failure receipt also failed: " + repr(secondary))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("run", "report", "packet", "lexical", "plan-sha256", "completed-sha256", "report-receipt-sha256",
                  "packet-completed-sha256", "lexical-completed-sha256", "source-sha256", "tests-sha256"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--out", default=str(Path(__file__).parent))
    result = run(parser.parse_args())
    print(json.dumps({"status": result["status"], "counts": result["counts"], "scientific_gate_changed": False}))
