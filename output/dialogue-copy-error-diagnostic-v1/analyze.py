"""Posthoc public-lexical/saved-choice diagnosis; no model or encoder calls."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
STUDY = ROOT / "runs/dialogue-copy-v1/study-01"
FEATURES = ROOT / "runs/sgd-state-v1/features-02"
LEXICAL = ROOT / "runs/dialogue-copy-v1/lexical-01"
METHODS = ("readout", "scalar", "selective", "selective_no_lexical", "candidate_gru")
SEEDS = (4101, 4102, 4103)
BINS = ("unmentioned_retention", "assigned_retention", "first_assignment", "revision", "clear")
PINS = {
    "runs/dialogue-copy-v1/study-01/plan.json": "609a0758f2a4ae4722052bb27740c74b3d1bdac396d8134eb39f5118f1429a2b",
    "runs/dialogue-copy-v1/study-01/completed.json": "8317abad5ef8235f31f5381591944ad152c42fddb98743fa64fd1f5cd8c3f300",
    "output/dialogue-copy-v1/report-01/summary.json": "4f895654bd5b046dee8a3c6e0a2ef7156b15a6299341450ebcae25398578b2f9",
    "output/dialogue-copy-v1/report-01/receipt.json": "cbeecb5f158b8a4f1cf9be46751348d1de718d493a7b47eafee6ee8ce230da88",
    "output/dialogue-copy-v1/independent-audit-01/receipt.json": "f143291635d7b895a82c9bf39a56b95d6b9e1946d5ad1d9e2c1548368304abe4",
}
LEXICAL_FEATURES = ["user_match", "system_match", "unique_longest_user", "unique_longest_system",
    "literal_current", "literal_previous", "is_none", "is_dontcare", "affirmative_cue_for_true",
    "negative_cue_for_false"]


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def bind(path, expected, bound):
    require(not path.is_symlink() and sha(path) == expected, "Changed input: " + str(path))
    bound[str(path.relative_to(ROOT))] = expected


def receipt_files(folder, receipt, bound):
    require(receipt["status"] == "completed", "Incomplete input")
    for name, value in receipt["files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe input member")
        bind(folder / name, value["sha256"] if isinstance(value, dict) else value, bound)


def extract(packet, index, lexical):
    """Flatten only scored rows; public lexical streams were computed on every turn."""
    expected = {k: [] for k in ("labels", "bin", "unseen", "dialogue", "time", "query")}
    fields = {k: [] for k in ("prior_gold", "previous_record", "boolean_slot", "user_unique", "system_unique",
                              "literal", "gold_boolean_cue", "candidate_count", "gap")}
    require(index["features"] == LEXICAL_FEATURES, "Lexical feature order")
    offset = 0
    for split in ("train", "dev"):
        cohort = packet["cohorts"][split]
        entries = index["cohorts"][split]
        require(len(entries) == len(cohort), "Lexical cohort")
        for dialog, entry in zip(cohort, entries, strict=True):
            qids = sorted({r["query"] for r in dialog["queries"]})
            c = max(len(packet["queries"][q]["candidate_ids"]) for q in qids)
            shape = [len(dialog["turns"]), len(qids), c, 10]
            require(entry["id"] == dialog["id"] and entry["query_ids"] == qids
                    and entry["shape"] == shape and entry["offset"] == offset, "Lexical alignment")
            block = lexical[offset:offset + math.prod(shape)].reshape(shape)
            offset += math.prod(shape)
            if split != "dev":
                continue
            positions = {q: i for i, q in enumerate(qids)}
            previous = {}
            for row in dialog["queries"]:
                q, t, y = row["query"], row["time"], row["label"]
                catalog = packet["queries"][q]
                values = catalog["candidate_values"]
                size = len(values)
                x = block[t, positions[q], :size]
                require(np.isfinite(x).all() and np.isin(x, [0, 1]).all(), "Nonbinary lexical values")
                require(catalog["split"] == "dev" and values[:2] == [None, None]
                        and catalog["candidate_ids"][:2] == ["reserved:NOT_MENTIONED", "reserved:DONTCARE"]
                        and 0 <= y < size, "Candidate labels")
                old_index, old_y, old_t = previous.get(q, (-1, 0, -1))
                require(t > old_t, "Noncausal scored query")
                b = ("unmentioned_retention" if y == old_y == 0 else "assigned_retention" if y == old_y
                     else "first_assignment" if old_y == 0 else "clear" if y == 0 else "revision")
                require(b == row["bin"], "Gold transition definition")
                users, systems, lit = (np.flatnonzero(x[:, j]) for j in (2, 3, 4))
                require(len(users) <= 1 and len(systems) <= 1 and len(lit) == 1, "Lexical one-hot indicators")
                u, s = int(users[0]) if len(users) else -1, int(systems[0]) if len(systems) else -1
                boolean = {v.strip().casefold() for v in values[2:]} == {"true", "false"}
                payload = (old_y, old_index, boolean, u, s, int(lit[0]), bool(x[y, 8] or x[y, 9]),
                           size, t - old_t if old_index >= 0 else 0)
                for key, value in zip(fields, payload, strict=True):
                    fields[key].append(value)
                for key, value in (("labels", y), ("bin", b), ("unseen", row["unseen"]),
                                   ("dialogue", dialog["id"]), ("time", t), ("query", q)):
                    expected[key].append(value)
                previous[q] = len(expected["labels"]) - 1, y, t
    require(offset == lexical.size, "Unaccounted lexical payload")
    return ({k: np.asarray(v) for k, v in expected.items()}, {k: np.asarray(v) for k, v in fields.items()})


def groups(truth, evidence):
    y, b = truth["labels"], truth["bin"]
    u, s = evidence["user_unique"], evidence["system_unique"]
    ug, sg = u == y, s == y
    ret = np.isin(b, BINS[:2])
    partitions = {
        "all": np.ones(len(y), bool),
        **{"bin/" + key: b == key for key in BINS},
        "label/not_mentioned": y == 0, "label/dontcare": y == 1, "label/ontology": y >= 2,
        "assigned_boolean": evidence["boolean_slot"] & (y >= 2),
        "assigned_nonboolean": ~evidence["boolean_slot"] & (y >= 2),
        "assigned_boolean/gold_cue": evidence["boolean_slot"] & (y >= 2) & evidence["gold_boolean_cue"],
        "assigned_boolean/no_gold_cue": evidence["boolean_slot"] & (y >= 2) & ~evidence["gold_boolean_cue"],
        "retention": ret,
        "retention/user_unique_gold": ret & ug,
        "retention/user_unique_other": ret & (u >= 0) & ~ug,
        "retention/no_user_unique": ret & (u < 0),
    }
    for key, condition in (("both_gold", ug & sg), ("user_gold_only", ug & ~sg),
                            ("system_gold_only", ~ug & sg), ("neither_gold", ~ug & ~sg)):
        partitions["revision/evidence/" + key] = (b == "revision") & condition
    partitions["revision/user_unique_old_label"] = (b == "revision") & (u == evidence["prior_gold"])
    partitions["revision/system_unique_old_label"] = (b == "revision") & (s == evidence["prior_gold"])
    partitions["revision/dontcare"] = (b == "revision") & (y == 1)
    partitions["revision/assigned_boolean"] = (b == "revision") & evidence["boolean_slot"] & (y >= 2)
    result = {}
    for panel, mask in (("all", np.ones(len(y), bool)), ("seen", ~truth["unseen"]), ("unseen", truth["unseen"])):
        result.update({panel + "/" + key: mask & condition for key, condition in partitions.items()})
    return result


def measure(choice, mask, truth, evidence):
    y = truth["labels"]
    correct, literal_correct = choice == y, evidence["literal"] == y
    prior_index = evidence["previous_record"]
    has_prior = prior_index >= 0
    previous = np.zeros_like(choice)
    previous[has_prior] = choice[prior_index[has_prior]]
    retained = np.isin(truth["bin"], BINS[:2])
    revisions = truth["bin"] == "revision"
    prior_correct = has_prior & (previous == evidence["prior_gold"])
    stale = ~correct & (choice == evidence["prior_gold"])
    n = int(mask.sum())
    count = lambda condition: int((mask & condition).sum())
    r = {
        "count": n, "correct": count(correct), "wrong": count(~correct),
        "accuracy": count(correct) / n if n else None,
        "both_correct": count(correct & literal_correct), "literal_only": count(~correct & literal_correct),
        "neural_only": count(correct & ~literal_correct), "both_wrong": count(~correct & ~literal_correct),
        "revision_count": count(revisions), "revision_correct": count(revisions & correct),
        "revision_stale_prior_gold": count(revisions & ~correct & (choice == evidence["prior_gold"])),
        "revision_other_wrong": count(revisions & ~correct & (choice != evidence["prior_gold"])),
        "revision_prior_prediction_correct": count(revisions & prior_correct),
        "revision_correct_after_prior_correct": count(revisions & prior_correct & correct),
        "revision_stale_after_prior_correct": count(revisions & prior_correct & stale),
        "revision_other_wrong_after_prior_correct": count(revisions & prior_correct & ~correct & ~stale),
        "revision_stale_after_prior_wrong": count(revisions & ~prior_correct & stale),
        "revision_adjacent_public_step": count(revisions & (evidence["gap"] == 1)),
        "revision_stale_adjacent_public_step": count(revisions & stale & (evidence["gap"] == 1)),
        "revision_stale_nonadjacent_public_step": count(revisions & stale & (evidence["gap"] > 1)),
        "revision_prior_correct_adjacent_public_step": count(revisions & prior_correct & (evidence["gap"] == 1)),
        "revision_stale_after_prior_correct_adjacent_public_step": count(revisions & prior_correct & stale & (evidence["gap"] == 1)),
        "wrong_despite_unique_user_gold": count(~correct & (evidence["user_unique"] == y)),
        "wrong_despite_unique_system_gold": count(~correct & (evidence["system_unique"] == y)),
        "wrong_no_unique_gold_in_either": count(~correct & (evidence["user_unique"] != y) & (evidence["system_unique"] != y)),
        "retention_count": count(retained), "retention_wrong": count(retained & ~correct),
        "retention_with_prior_scored_prediction": count(retained & has_prior),
        "retention_prior_prediction_correct": count(retained & has_prior & (previous == y)),
        "retention_new_error_after_correct_prediction": count(retained & has_prior & (previous == y) & ~correct),
        "retention_error_after_wrong_prediction": count(retained & has_prior & (previous != y) & ~correct),
        "retention_error_without_previous_scored_prediction": count(retained & ~has_prior & ~correct),
        "retention_scored_choice_changed": count(retained & has_prior & (choice != previous)),
        "retention_new_error_with_other_user_match": count(retained & has_prior & (previous == y) & ~correct
                                                         & (evidence["user_unique"] >= 0) & (evidence["user_unique"] != y)),
        "retention_new_error_with_gold_user_match": count(retained & has_prior & (previous == y) & ~correct
                                                        & (evidence["user_unique"] == y)),
        "retention_new_error_without_user_match": count(retained & has_prior & (previous == y) & ~correct
                                                       & (evidence["user_unique"] < 0)),
    }
    require(r["both_correct"] + r["literal_only"] + r["neural_only"] + r["both_wrong"] == n, "Complementarity partition")
    require(r["revision_correct"] + r["revision_stale_prior_gold"] + r["revision_other_wrong"] == r["revision_count"],
            "Revision partition")
    require(r["revision_stale_after_prior_correct"] + r["revision_stale_after_prior_wrong"]
            == r["revision_stale_prior_gold"], "Previous prediction partition")
    require(r["revision_correct_after_prior_correct"] + r["revision_stale_after_prior_correct"]
            + r["revision_other_wrong_after_prior_correct"] == r["revision_prior_prediction_correct"],
            "Previously correct revision partition")
    require(sum(r[k] for k in ("retention_new_error_after_correct_prediction", "retention_error_after_wrong_prediction",
                               "retention_error_without_previous_scored_prediction")) == r["retention_wrong"], "Retention partition")
    return r


def run():
    require(not any((OUT / n).exists() for n in ("diagnostic.json", "receipt.json", "README.md")), "Output already exists")
    start, bound = time.perf_counter(), {}
    for name, value in PINS.items():
        bind(ROOT / name, value, bound)
    plan, done = read(STUDY / "plan.json"), read(STUDY / "completed.json")
    summary = read(ROOT / "output/dialogue-copy-v1/report-01/summary.json")
    require(done["status"] == "completed" and done["fit_count"] == 15
            and done["plan_sha256"] == PINS["runs/dialogue-copy-v1/study-01/plan.json"], "Study completion")
    for name, value in plan["source_sha256"].items():
        bind(ROOT / name, value, bound)
    for folder, key in ((FEATURES, "packet_completed_sha256"), (LEXICAL, "lexical_completed_sha256")):
        bind(folder / "completed.json", plan[key], bound)
        receipt_files(folder, read(folder / "completed.json"), bound)
    lexical_receipt = read(LEXICAL / "completed.json")
    require(lexical_receipt["packet_completed_sha256"] == plan["packet_completed_sha256"], "Lexical lineage")
    packet, index = read(FEATURES / "packet.json"), read(LEXICAL / "index.json")
    lexical = np.load(LEXICAL / "lexical.npy", allow_pickle=False, mmap_mode="r")
    require(lexical.dtype == np.float32 and lexical.ndim == 1
            and lexical.size == lexical_receipt["float32_scalars"], "Lexical array")
    truth, evidence = extract(packet, index, lexical)
    require(len(truth["labels"]) == 62329, "Incomplete cohort")
    bind(STUDY / "references.npz", done["references"]["sha256"], bound)
    with np.load(STUDY / "references.npz", allow_pickle=False) as saved:
        require(set(saved.files) == set(truth) | {"pred_none", "pred_literal"}, "Reference fields")
        for key, value in truth.items():
            require(np.array_equal(saved[key], value), "Reference record mismatch")
        require(np.array_equal(saved["pred_literal"], evidence["literal"]), "Literal feature/reference mismatch")
    masks = groups(truth, evidence)
    support = {name: int(mask.sum()) for name, mask in masks.items()}
    rows = {}
    names = [f"{m}-{s}" for s in SEEDS for m in METHODS]
    require([f"{r['method']}-{r['seed']}" for r in done["fits"]] == names, "Fit membership")
    expected_files = {"plan.json", "completed.json", "references.npz"} | {
        f"fits/{n}/{file}" for n in names for file in ("completed.json", "weights.pt", "dev-predictions.npz")}
    actual_files = {p.relative_to(STUDY).as_posix() for p in STUDY.rglob("*") if p.is_file()}
    require(actual_files == expected_files, "Execution members")
    for record, name in zip(done["fits"], names, strict=True):
        folder = STUDY / "fits" / name
        require(read(folder / "completed.json") == record, "Fit receipt")
        for file, value in (("completed.json", sha(folder / "completed.json")), ("weights.pt", record["weights_sha256"]),
                            ("dev-predictions.npz", record["predictions_sha256"])):
            bind(folder / file, value, bound)
        with np.load(folder / "dev-predictions.npz", allow_pickle=False) as saved:
            require(set(saved.files) == set(truth) | {"choice", "probabilities"}, "Prediction fields")
            for key, value in truth.items():
                require(np.array_equal(saved[key], value), "Prediction cohort mismatch")
            choice = saved["choice"]
            require(np.array_equal(choice, saved["probabilities"].argmax(1))
                    and ((choice >= 0) & (choice < evidence["candidate_count"])).all(), "Choice mismatch")
        rows[name] = {group: measure(choice, mask, truth, evidence) for group, mask in masks.items()}
        for panel in ("seen", "unseen"):
            require(math.isclose(rows[name][panel + "/all"]["accuracy"],
                                summary["rows"][name]["metrics"][panel]["micro"]["accuracy"], abs_tol=1e-15),
                    "Existing summary micro metric")
    families = {}
    for method in METHODS:
        families[method] = {}
        for group in masks:
            members = [rows[f"{method}-{s}"][group] for s in SEEDS]
            families[method][group] = {"count_per_fit": members[0]["count"], "fits": 3,
                **{k: None if any(v[k] is None for v in members) else float(np.mean([v[k] for v in members]))
                   for k in members[0] if k != "count"}}
    literal = {group: measure(evidence["literal"], mask, truth, evidence) for group, mask in masks.items()}
    result = {
        "status": "completed", "study": "dialogue-copy-error-diagnostic-v1", "posthoc": True,
        "input_sha256": bound, "source_sha256": sha(__file__), "counts": {
            "fits": 15, "seeds": list(SEEDS), "queries_per_fit": len(truth["labels"]),
            "saved_prediction_rows": len(truth["labels"]) * 15, "literal_feature_reference_mismatches": 0,
            "scored_rows_with_intervening_unscored_steps": int((evidence["gap"] > 1).sum()),
            "new_model_calls": 0, "new_encoder_calls": 0, "test_data_accessed": False},
        "group_support": support, "families": families, "per_fit": rows, "literal": literal,
        "definitions": {
            "revision": "Current assigned gold differs from previous assigned gold for the same question, using only scored frames.",
            "stale_prior_gold": "Wrong current prediction equals previous scored gold. This is an error phenotype, not proof of an internal carry operation.",
            "unique_user": "Saved unique-longest bounded ontology match in current USER text; sentinel labels never match.",
            "unique_system": "Saved unique-longest match in immediately preceding SYSTEM text. It can be a question or rejected option, not affirmative evidence.",
            "false_update_proxy": "On gold retention, previous scored prediction was correct but current prediction is wrong. Intervening hidden-state steps are not saved.",
            "boolean": "Ontology set after strip/casefold is exactly {true,false}; only labels>=2 counted as assigned boolean.",
            "boolean_cue": "Saved current USER affirmative/negative cue attached to gold True/False candidate; evaluator-only label match.",
            "count_aggregation": "Per-fit exact counts retained. Family counts are arithmetic mean across three seeds on the same query set, not independent examples.",
            "complementarity": "Literal-only, neural-only, both-correct and both-wrong partitions. No union oracle is reported as achieved performance.",
        },
        "limits": ["Already exposed official development only; no test content, fitting, inference, encoding or threshold search.",
                   "A lexical match does not establish intent/relevance. Missing literal evidence does not establish missing semantic evidence.",
                   "No saved gate/departure probabilities or intermediate belief states: internal inertia and observation error cannot be causally separated.",
                   "The frozen 7/13 continuation failure is unchanged; subgroup diagnostics are not new success criteria.",
                   "Lexical extraction is pinned-source bound, not regenerated from raw utterances in this diagnostic."]}
    for name, value in bound.items():
        require(sha(ROOT / name) == value, "Input changed during diagnostic")
    write(OUT / "diagnostic.json", result)
    (OUT / "README.md").write_text(render(result))
    write(OUT / "receipt.json", {"status": "completed", "posthoc": True, "source_sha256": sha(__file__),
        "input_sha256": bound, "files": {name: {"sha256": sha(OUT / name), "bytes": (OUT / name).stat().st_size}
                                        for name in ("analyze.py", "diagnostic.json", "README.md")},
        "wall_seconds": time.perf_counter() - start, "new_model_calls": 0, "new_encoder_calls": 0,
        "scientific_gate_changed": False})
    print(json.dumps({"status": "completed", "diagnostic_sha256": sha(OUT / "diagnostic.json"),
                      "receipt_sha256": sha(OUT / "receipt.json")}))


def render(result):
    f = result["families"]
    lines = ["# Dialogue copy error diagnosis", "", ("Posthoc saved-output analysis of exposed development data. "
             "All 15 fits and three seeds are retained; no model or encoder calls. The frozen continuation rule remains failed (7/13)."), "",
             ("Counts below are mean per fit across the same three seeds, except the single literal control. "
             "Exact per-seed counts and all subgroups are in diagnostic.json."), "",
             "## Revisions", "", "A stale error selects the previous scored gold label; other wrong predictions select a different wrong candidate.", "",
             "| Panel | Method | Revisions | Correct | Stale previous gold | Other wrong |",
             "|---|---|---:|---:|---:|---:|"]
    for panel in ("seen", "unseen"):
        for method in (*METHODS, "literal"):
            r = result["literal"][panel + "/bin/revision"] if method == "literal" else f[method][panel + "/bin/revision"]
            lines.append(f"| {panel} | {method} | {r['revision_count']:.0f} | {r['revision_correct']:.2f} | "
                         f"{r['revision_stale_prior_gold']:.2f} | {r['revision_other_wrong']:.2f} |")
    lines += ["", "## Current evidence", "", "These are lexical matches, not verified intent. SYSTEM mentions may propose or question a value.", "",
              "| Panel | Gold-matching unique mention | Queries | Scalar correct | Selective correct |",
              "|---|---|---:|---:|---:|"]
    for panel in ("seen", "unseen"):
        for evidence in ("both_gold", "user_gold_only", "system_gold_only", "neither_gold"):
            key = panel + "/revision/evidence/" + evidence
            lines.append(f"| {panel} | {evidence} | {result['group_support'][key]} | "
                         f"{f['scalar'][key]['correct']:.2f} | {f['selective'][key]['correct']:.2f} |")
    lines += ["", "## Prior correctness", "",
              "| Panel | Method | All revision errors | Stale errors | Stale after correct prior | Stale on adjacent USER step |",
              "|---|---|---:|---:|---:|---:|"]
    for panel in ("seen", "unseen"):
        for method in ("scalar", "selective"):
            r = f[method][panel + "/bin/revision"]
            lines.append(f"| {panel} | {method} | {r['wrong']:.2f} | {r['revision_stale_prior_gold']:.2f} | "
                         f"{r['revision_stale_after_prior_correct']:.2f} | {r['revision_stale_adjacent_public_step']:.2f} |")
    lines += ["", "## Retention errors and new observed errors", "",
              ("A new error means the previous scored prediction was correct on the same retained gold label. "
              "This observes scored endpoints, not the internal update that produced the error."), "",
              "| Panel | Method | Retention queries | Wrong | New after correct | Already wrong | No previous prediction |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for panel in ("seen", "unseen"):
        for method in (*METHODS, "literal"):
            r = result["literal"][panel + "/retention"] if method == "literal" else f[method][panel + "/retention"]
            lines.append(f"| {panel} | {method} | {r['retention_count']:.0f} | {r['retention_wrong']:.2f} | "
                         f"{r['retention_new_error_after_correct_prediction']:.2f} | "
                         f"{r['retention_error_after_wrong_prediction']:.2f} | "
                         f"{r['retention_error_without_previous_scored_prediction']:.2f} |")
    lines += ["", "## Assigned boolean and DONTCARE targets", "",
              "| Panel | Subgroup | Queries | Method | Correct | Literal only | Neural only | Both wrong |",
              "|---|---|---:|---|---:|---:|---:|---:|"]
    for panel in ("seen", "unseen"):
        for group in ("assigned_boolean", "label/dontcare"):
            for method in ("scalar", "selective"):
                r = f[method][panel + "/" + group]
                lines.append(f"| {panel} | {group} | {r['count_per_fit']} | {method} | {r['correct']:.2f} | "
                             f"{r['literal_only']:.2f} | {r['neural_only']:.2f} | {r['both_wrong']:.2f} |")
    lines += ["", "## What the counts identify", ""]
    for panel in ("seen", "unseen"):
        r = f["selective"][panel + "/bin/revision"]
        supported = sum(result["group_support"][panel + "/revision/evidence/" + key]
                        for key in ("both_gold", "user_gold_only", "system_gold_only"))
        lines.append(f"On {panel} revisions, {r['revision_stale_prior_gold']:.2f} of {r['wrong']:.2f} mean selective errors "
                     f"select the old gold value. {r['revision_stale_after_prior_correct']:.2f} of those followed a correct "
                     f"previous scored prediction. A unique current USER or preceding SYSTEM mention matches the new gold "
                     f"on {supported}/{r['count_per_fit']} revisions. These counts distinguish error patterns, not their internal cause.")
    lines += ["", "## Interpretation limits", ""] + ["- " + s for s in result["limits"]]
    lines += ["", ("The aggregate files contain no raw dialogue examples or individual saved predictions. "
              "Subgroups describe where current methods fail; they do not identify a causal mechanism or validate a proposed replacement.")]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    run()
