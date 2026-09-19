"""Frozen-fit replay and one-step operator diagnosis, never trajectory evaluation."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import study_dialogue_copy as old
import torch

from openjev.research.dialogue_copy_memory import DialogueCopyMemory

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-evidence-qualification-v1"
OLD_PLAN = "609a0758f2a4ae4722052bb27740c74b3d1bdac396d8134eb39f5118f1429a2b"
OLD_COMPLETED = "8317abad5ef8235f31f5381591944ad152c42fddb98743fa64fd1f5cd8c3f300"
NEW_SOURCES = ["scripts/qualify_dialogue_evidence.py", "tests/test_qualify_dialogue_evidence.py",
               "research/dialogue-evidence-qualification-protocol.md"]
FITS = [f"{method}-{seed}" for seed in (4101, 4102, 4103) for method in ("scalar", "selective")]
FIELDS = ("probabilities", "labels", "choice", "bin", "unseen", "dialogue", "time", "query")
ROWS = 62329
QUANTILES = [0., .25, .5, .75, 1.]
TOLERANCE = 1e-6
FACTOR_ROUNDOFF = 2e-6
WALL_CAP = 300.
sha, write = old.sha, old.write


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def bound_file(path, digest):
    path = Path(path)
    require(not path.is_symlink() and path.is_file() and sha(path) == digest, "Changed file: " + str(path))


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def receipt_tree(folder, digest, check=lambda: None):
    bound_file(folder / "completed.json", digest)
    require(not (folder / "failed.json").exists(), "Failed input")
    receipt = read(folder / "completed.json")
    require(receipt["status"] == "completed", "Incomplete input")
    for name, value in receipt["files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe member")
        bound_file(folder / name, value["sha256"] if isinstance(value, dict) else value)
        check()
    return receipt


def authenticate(plan, check=lambda: None):
    require(plan["version"] == VERSION and plan["fits"] == FITS and plan["rows_per_fit"] == ROWS
            and plan["probability_atol"] == TOLERANCE and plan["quantiles"] == QUANTILES
            and plan["factor_roundoff"] == FACTOR_ROUNDOFF
            and plan["runtime"] == old.runtime(), "Frozen settings/runtime changed")
    require(type(plan["wall_cap_seconds"]) in (int, float)
            and plan["wall_cap_seconds"] == WALL_CAP, "Invalid cap")
    require(set(plan["source_sha256"]) == set(old.SOURCES + NEW_SOURCES), "Source closure")
    for name, digest in plan["source_sha256"].items():
        bound_file(ROOT / name, digest)
        check()
    paths = {}
    for key in ("old_study", "packet", "lexical"):
        value = plan["paths"][key]
        require(not Path(value).is_absolute() and ".." not in Path(value).parts, "Unsafe path")
        paths[key] = ROOT / value
    study = paths["old_study"]
    require(plan["old_plan_sha256"] == OLD_PLAN and plan["old_completed_sha256"] == OLD_COMPLETED,
            "Original study pins")
    bound_file(study / "plan.json", OLD_PLAN)
    bound_file(study / "completed.json", OLD_COMPLETED)
    original, done = read(study / "plan.json"), read(study / "completed.json")
    require(original["runtime"] == plan["runtime"] and original["config"] == old.CONFIG
            and original["practical_checks"] == old.PRACTICAL_CHECKS, "Original recipe/runtime")
    require(set(original["source_sha256"]) == set(old.SOURCES), "Original sources")
    require(all(plan["source_sha256"][k] == v for k, v in original["source_sha256"].items()), "Old source rebinding")
    require(done["status"] == "completed" and done["fit_count"] == 15
            and done["plan_sha256"] == OLD_PLAN, "Original completion")
    names = [f"{m}-{s}" for s in old.SEEDS for m in old.METHODS]
    require([f"{r['method']}-{r['seed']}" for r in done["fits"]] == names, "Original fit membership")
    expected = {"plan.json", "completed.json", "references.npz"}
    for record, name in zip(done["fits"], names, strict=True):
        folder = study / "fits" / name
        require(read(folder / "completed.json") == record and record["status"] == "completed", "Fit receipt")
        for filename, key in (("weights.pt", "weights_sha256"), ("dev-predictions.npz", "predictions_sha256")):
            bound_file(folder / filename, record[key])
            check()
        expected.update(f"fits/{name}/{n}" for n in ("completed.json", "weights.pt", "dev-predictions.npz"))
    bound_file(study / "references.npz", done["references"]["sha256"])
    require({p.relative_to(study).as_posix() for p in study.rglob("*") if p.is_file()} == expected, "Original file closure")
    for key in ("packet", "lexical"):
        pin = key + "_completed_sha256"
        require(plan[pin] == original[pin], "Original preparation binding")
        receipt_tree(paths[key], plan[pin], check)
    return paths, {f"{r['method']}-{r['seed']}": r for r in done["fits"]}


def freeze(args):
    original = read(Path(args.old_study) / "plan.json")
    plan = {"version": VERSION, "paths": {k: relative(getattr(args, k)) for k in ("old_study", "packet", "lexical")},
            "old_plan_sha256": OLD_PLAN, "old_completed_sha256": OLD_COMPLETED,
            "packet_completed_sha256": original["packet_completed_sha256"],
            "lexical_completed_sha256": original["lexical_completed_sha256"],
            "source_sha256": {p: sha(ROOT / p) for p in old.SOURCES + NEW_SOURCES},
            "runtime": old.runtime(), "fits": FITS, "rows_per_fit": ROWS,
            "probability_atol": TOLERANCE, "quantiles": QUANTILES,
            "factor_roundoff": FACTOR_ROUNDOFF,
            "wall_cap_seconds": args.wall_cap_seconds,
            "scope": "Replay six fixed fits; source-bound one-step operator diagnostic only; no training or trajectory counterfactual"}
    authenticate(plan)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        write(out / "plan.json", plan)
    except BaseException as error:
        old.failure(out, error, {"phase": "freeze"})
        raise
    return {"plan_sha256": sha(out / "plan.json")}


class TraceMemory(DialogueCopyMemory):
    """Parent forward/update untouched; capture returned public operator factors."""

    def reset_trace(self):
        self.trace = []
        self.advance_attempted = self.advance_returned = 0

    def _advance(self, state, *args):
        self.advance_attempted += 1
        before = state["log_b"].detach().exp().clone()
        proposed, diagnostic = super()._advance(state, *args)
        self.advance_returned += 1
        self.trace.append({"old_b": before.cpu().numpy(),
                           "writer": diagnostic["write_log_probs"].detach().exp().cpu().numpy().copy(),
                           "departure_mass": diagnostic["departure_mass"].detach().cpu().numpy().copy(),
                           "result": proposed["log_b"].detach().softmax(-1).cpu().numpy().copy()})
        return proposed, diagnostic


def restore_model(folder, record):
    bound_file(folder / "weights.pt", record["weights_sha256"])
    # Construction draws are discarded and global RNG is restored; no scientific seed is allocated.
    with torch.random.fork_rng(devices=[]):
        model = TraceMemory(record["method"], projection_dim=old.CONFIG["projection_dim"],
                            hidden_dim=old.CONFIG["hidden_dim"], gru_width=old.CONFIG["gru_width"])
    weights = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
    require(set(weights) == set(model.state_dict()) and all(
        isinstance(v, torch.Tensor) and v.dtype == torch.float32 and torch.isfinite(v).all()
        and v.shape == model.state_dict()[k].shape for k, v in weights.items()), "Weights schema")
    model.load_state_dict(weights, strict=True)
    # configuration's class name deliberately differs only because of instrumentation.
    config = model.configuration()
    config["class"] = "DialogueCopyMemory"
    require(config == record["configuration"], "Model configuration")
    require(all(torch.equal(v, model.state_dict()[k]) for k, v in weights.items()), "Weight restore")
    bound_file(folder / "weights.pt", record["weights_sha256"])
    model.eval()
    model.reset_trace()
    return model


def save_arrays(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def compare_replay(arrays, expected):
    require(set(expected) == set(FIELDS), "Saved fields")
    for key in FIELDS:
        if key == "probabilities":
            continue
        compatible = arrays[key].dtype == expected[key].dtype or (
            arrays[key].dtype.kind == expected[key].dtype.kind == "U")
        require(compatible and np.array_equal(arrays[key], expected[key]),
                "Replay metadata/choice mismatch: " + key)
    p, old_p = arrays["probabilities"], expected["probabilities"]
    require(p.dtype == old_p.dtype == np.float32 and p.shape == old_p.shape
            and np.isfinite(p).all() and np.isfinite(old_p).all(), "Replay probability schema")
    difference = float(np.max(np.abs(p.astype(np.float64) - old_p.astype(np.float64)), initial=0))
    require(difference <= TOLERANCE, "Replay probability mismatch")
    return {"bitwise_equal": p.tobytes() == old_p.tobytes(), "max_probability_abs_difference": difference,
            "exact_metadata_and_choices": True}


def replay_batches(model, ds, queries, features, lexical, expected, progress, check):
    pieces = progress["pieces"]
    cursor = 0
    with torch.inference_mode():
        for start in range(0, len(ds), old.CONFIG["batch_size"]):
            check()
            subset = ds[start:start + old.CONFIG["batch_size"]]
            actor, _, _ = old.make_batch(subset, queries, features, lexical)
            progress["active_dialogues"] = [d["id"] for d in subset]
            progress["active_valid"] = actor[1].numpy().copy()
            progress["active_mask"] = actor[4].numpy().copy()
            progress["actor_work"].update(old.work_counts(subset, actor))
            model.reset_trace()
            progress["model_forward_calls"] += 1
            try:
                probabilities = model(*actor).softmax(-1).numpy()
                progress["model_forward_returned"] += 1
            finally:
                progress["advance_attempted"] += model.advance_attempted
                progress["advance_returned"] += model.advance_returned
            check()
            batch = {k: [] for k in (*FIELDS, "old_b", "writer", "departure_mass", "result", "candidate_count", "user_unique", "system_unique", "boolean_slot")}
            for i, d in enumerate(subset):
                positions = {q: j for j, q in enumerate(d["layout"]["query_ids"])}
                for row in d["queries"]:
                    t, j = row["time"], positions[row["query"]]
                    n = len(queries[row["query"]]["candidates"])
                    require(n <= 12, "Original candidate width")
                    for key in ("old_b", "writer", "result"):
                        value = np.zeros(12, np.float32)
                        value[:n] = model.trace[t][key][i, j, :n]
                        batch[key].append(value)
                    value = np.zeros(12, np.float32)
                    value[:n] = probabilities[i, t, j, :n]
                    for key, item in (("probabilities", value), ("choice", int(value[:n].argmax())),
                                      ("labels", row["label"]), ("bin", row["bin"]), ("unseen", row["unseen"]),
                                      ("dialogue", d["id"]), ("time", t), ("query", row["query"]),
                                      ("candidate_count", n), ("departure_mass", float(model.trace[t]["departure_mass"][i, j, 0]))):
                        batch[key].append(item)
                    for key, field in (("user_unique", 2), ("system_unique", 3)):
                        found = np.flatnonzero(actor[5][i, t, j, :n, field].numpy())
                        require(len(found) <= 1, "Nonunique public lexical indicator")
                        batch[key].append(int(found[0]) if len(found) else -1)
                    values = queries[row["query"]]["candidate_values"]
                    batch["boolean_slot"].append({v.strip().casefold() for v in values if v is not None}
                                                 == {"true", "false"})
            batch = {k: np.asarray(v) for k, v in batch.items()}
            # Preserve actual returned rows even when a comparison fails.
            pieces.append(batch)
            progress["returned_rows"] += len(batch["labels"])
            require(np.array_equal(batch["probabilities"], batch["result"]), "Capture changed result")
            size = len(batch["labels"])
            compare_replay(batch, {k: v[cursor:cursor + size] for k, v in expected.items()})
            cursor += size
            progress["matched_rows"] = cursor
            model.reset_trace()
            progress["active_dialogues"] = []
            progress.pop("active_valid", None)
            progress.pop("active_mask", None)
    arrays = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    require(cursor == len(expected["labels"]) == ROWS, "Replay row coverage")
    return arrays, compare_replay(arrays, expected)


def factors(arrays):
    b, w = (arrays[k].astype(np.float64) for k in ("old_b", "writer"))
    n, mass = arrays["candidate_count"], arrays["departure_mass"].astype(np.float64)
    mask = np.arange(12)[None] < n[:, None]
    require(b.shape == w.shape == mask.shape and mass.shape == n.shape
            and np.isfinite(b).all() and np.isfinite(w).all() and np.isfinite(mass).all()
            and (b >= 0).all() and (w >= 0).all() and ((mass >= 0) & (mass <= 1 + FACTOR_ROUNDOFF)).all()
            and ((n >= 1) & (n <= 12)).all() and (b[~mask] == 0).all() and (w[~mask] == 0).all()
            and np.allclose(b.sum(1), 1, atol=FACTOR_ROUNDOFF, rtol=0)
            and np.allclose(w.sum(1), 1, atol=FACTOR_ROUNDOFF, rtol=0), "Invalid saved factors")
    pi = (1 - mass[:, None]) * b + mass[:, None] * mask / n[:, None]
    require(np.isfinite(pi).all() and (pi >= 0).all(), "Negative/nonfinite product prior")
    product = pi * w
    normalizer = product.sum(1, keepdims=True)
    require((normalizer > 0).all() and np.isfinite(normalizer).all(), "Zero product support")
    return w, product / normalizer


def diagnostic(arrays):
    w, product = factors(arrays)
    y, choice = arrays["labels"], arrays["choice"]
    prior_gold, previous, gap = np.zeros(len(y), np.int64), np.full(len(y), -1, np.int64), np.zeros(len(y), np.int64)
    seen = {}
    for i, (d, q, t) in enumerate(zip(arrays["dialogue"], arrays["query"], arrays["time"], strict=True)):
        key = str(d), int(q)
        if key in seen:
            j = seen[key]
            require(t > arrays["time"][j], "Scored chronology")
            previous[i], prior_gold[i], gap[i] = j, y[j], t - arrays["time"][j]
        seen[key] = i
    correct = choice == y
    has_prior = previous >= 0
    prior_correct = has_prior & (choice[np.maximum(previous, 0)] == prior_gold)
    revision = arrays["bin"] == "revision"
    retention = np.isin(arrays["bin"], ["unmentioned_retention", "assigned_retention"])
    primary = revision & prior_correct & (gap == 1) & (arrays["user_unique"] == y)
    masks = {"all": np.ones(len(y), bool), "revision": revision, "retention": retention,
             **{"bin/" + name: arrays["bin"] == name for name in (
                 "unmentioned_retention", "assigned_retention", "first_assignment", "revision", "clear")},
             "revision/prior_correct_adjacent_user_unique_gold": primary,
             "revision/user_unique_gold": revision & (arrays["user_unique"] == y),
             "revision/system_unique_gold": revision & (arrays["system_unique"] == y),
             "revision/neither_unique_gold": revision & (arrays["user_unique"] != y) & (arrays["system_unique"] != y),
             "retention/prior_correct": retention & prior_correct,
             "retention/prior_wrong": retention & has_prior & ~prior_correct,
             "retention/no_prior": retention & ~has_prior,
             "retention/new_wrong": retention & prior_correct & ~correct,
             "retention/already_wrong": retention & has_prior & ~prior_correct & ~correct,
             "retention/wrong_no_prior": retention & ~has_prior & ~correct,
             "assigned_boolean": arrays["boolean_slot"] & (y >= 2), "dontcare": y == 1}
    stale = ~correct & (choice == prior_gold)
    for prior_name, prior_mask in (("any_prior", np.ones(len(y), bool)), ("prior_correct", prior_correct),
                                   ("prior_wrong", has_prior & ~prior_correct), ("no_prior", ~has_prior)):
        base = revision & prior_mask
        masks["revision/" + prior_name] = base
        for outcome, outcome_mask in (("stale", stale), ("other_wrong", ~correct & ~stale)):
            masks[f"revision/{prior_name}/{outcome}"] = base & outcome_mask
        for gap_name, gap_mask in (("adjacent", gap == 1), ("nonadjacent", gap > 1)):
            masks[f"revision/{prior_name}_{gap_name}"] = base & gap_mask
    ug, sg = arrays["user_unique"] == y, arrays["system_unique"] == y
    evidence_bases = {"revision": revision, "primary": primary}
    evidence_bases.update({name: mask for name, mask in masks.items()
                          if name in ("revision/prior_correct", "revision/prior_wrong")
                          or (name.startswith("revision/") and name.endswith(("/stale", "/other_wrong")))})
    for name, base in evidence_bases.items():
        for evidence, mask in (("both_gold", ug & sg), ("user_gold_only", ug & ~sg),
                               ("system_gold_only", ~ug & sg), ("neither_gold", ~ug & ~sg)):
            masks[f"{name}/evidence/{evidence}"] = base & mask
    rows = {}
    writer_choice, product_choice = w.argmax(1), product.argmax(1)
    indices = np.arange(len(y))
    b = arrays["old_b"].astype(np.float64)
    scalars = {"writer_max": w.max(1), "writer_gold": w[indices, y], "writer_previous_gold": w[indices, prior_gold],
               "old_b_max": b.max(1), "old_b_squared_mass": (b*b).sum(1),
               "old_b_previous_gold": b[indices, prior_gold], "departure_mass": arrays["departure_mass"]}

    def stats(values):
        return {"count": len(values), "sum": float(values.sum()),
                "quantiles": np.quantile(values, QUANTILES).tolist() if len(values) else None}

    for panel, panel_mask in (("all", np.ones(len(y), bool)), ("seen", ~arrays["unseen"]), ("unseen", arrays["unseen"])):
        for name, subset in masks.items():
            mask = panel_mask & subset
            wrong = mask & ~correct
            categories = {"writer_gold": wrong & (writer_choice == y),
                          "writer_previous_gold": wrong & (writer_choice == prior_gold) & (writer_choice != y),
                          "writer_other": wrong & (writer_choice != prior_gold) & (writer_choice != y)}
            require(sum(int(v.sum()) for v in categories.values()) == int(wrong.sum()), "Writer partition")
            row = {"count": int(mask.sum()), "correct": int((mask & correct).sum()), "wrong": int(wrong.sum()),
                   "wrong_stale_previous_gold": int((wrong & stale).sum()),
                   "wrong_other": int((wrong & ~stale).sum()),
                   "wrong_writer_partition": {k: int(v.sum()) for k, v in categories.items()}}
            row["one_step_counterfactual"] = {}
            for key, picked in (("force_write", writer_choice), ("normalized_product", product_choice)):
                row["one_step_counterfactual"][key] = {
                    "repair_wrong": int((wrong & (picked == y)).sum()),
                    "break_correct": int((mask & correct & (picked != y)).sum()),
                    "unchanged_correct": int((mask & correct & (picked == y)).sum()),
                    "still_wrong": int((wrong & (picked != y)).sum())}
            groups = {"all": mask, "wrong": wrong, **categories}
            row["factor_statistics"] = {group: {k: stats(v[selection]) for k, v in scalars.items()}
                                         for group, selection in groups.items()}
            rows[panel + "/" + name] = row
    mass = arrays["departure_mass"].astype(np.float64)
    roundoff = {"departure_mass_above_one_count": int((mass > 1).sum()),
                "departure_mass_max_overshoot": float(np.maximum(mass - 1, 0).max(initial=0)),
                "old_b_max_sum_error": float(np.abs(b.sum(1) - 1).max(initial=0)),
                "writer_max_sum_error": float(np.abs(w.sum(1) - 1).max(initial=0)),
                "permitted_roundoff": FACTOR_ROUNDOFF,
                "convention": "Literal saved factors cast float64; no b/w normalization, m clipping, or floor"}
    return {"strata": rows, "quantile_probabilities": QUANTILES, "factor_roundoff": roundoff,
            "scope": "One-step substitutions at actual old-model states. Not rolled-out performance; no gate counterfactual identifies causation.",
            "definitions": {"previous_gold": "Evaluator-only previous scored label, initially NONE",
                            "prior_correct": "Previous scored prediction equals previous scored gold; intervening public updates may exist",
                            "writer_gold": "Actual old writer argmax is gold while actual resulting choice is wrong",
                            "purity": "old_b_max and old_b_squared_mass are descriptive, with no selected threshold",
                            "counterfactual": "Float64 arithmetic on saved float32 factors; smallest-index argmax; no floor or learned replacement"}}


def public_progress(progress):
    return {k: dict(v) if isinstance(v, Counter) else v for k, v in progress.items()
            if k not in ("pieces", "active_valid", "active_mask")}


def preserve_failure(out, error, progress, model, records, start):
    def attempt(fn, label):
        try:
            fn()
        except BaseException as secondary:  # noqa: BLE001 - preserve original failure
            error.add_note(label + ": " + str(secondary))
    completed = out / "completed.json"
    if completed.exists():
        attempt(lambda: completed.rename(out / "invalid-completion.json"), "Completion demotion")
    if progress.get("pieces"):
        attempt(lambda: save_arrays(out / "partial.npz", {k: np.concatenate([p[k] for p in progress["pieces"]])
                                                         for k in progress["pieces"][0]}), "Partial rows")
    if model is not None and getattr(model, "trace", None):
        def partial_batch():
            batch = {k: np.stack([t[k] for t in model.trace], 1) for k in model.trace[0]}
            batch.update({k: progress[k] for k in ("active_valid", "active_mask") if k in progress})
            save_arrays(out / "partial-batch.npz", batch)
        attempt(partial_batch, "Partial public batch")
    old.failure(out, error, {"completed_fits": records, "active_progress": public_progress(progress),
                             "wall_seconds": time.perf_counter() - start})


def replay(args):
    start = time.perf_counter()
    bound_file(Path(args.plan), args.plan_sha256)
    plan = read(args.plan)
    cap = plan["wall_cap_seconds"]
    require(type(cap) in (int, float) and math.isfinite(cap) and cap > 0, "Invalid cap")
    def check():
        if time.perf_counter() - start > cap:
            raise TimeoutError("Qualification wall cap exceeded")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    records, progress, model = [], {}, None
    try:
        write(out / "started.json", {"version": VERSION, "plan_sha256": args.plan_sha256,
                                      "runtime": old.runtime(), "source_sha256": plan["source_sha256"]})
        check()
        paths, old_records = authenticate(plan, check)
        cohorts, queries, features, lexical = old.load_inputs(paths["packet"], paths["lexical"])
        check()
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        (out / "fits").mkdir()
        for name in FITS:
            check()
            model = None
            progress = {"fit": name, "pieces": [], "model_forward_calls": 0, "model_forward_returned": 0,
                        "advance_attempted": 0, "advance_returned": 0, "returned_rows": 0, "matched_rows": 0,
                        "actor_work": Counter(), "active_dialogues": []}
            fit_start = time.perf_counter()
            record, folder = old_records[name], paths["old_study"] / "fits" / name
            destination = out / "fits" / name
            destination.mkdir()
            model = restore_model(folder, record)
            with np.load(folder / "dev-predictions.npz", allow_pickle=False) as saved:
                expected = {k: saved[k] for k in saved.files}
            arrays, agreement = replay_batches(model, cohorts["dev"], queries, features, lexical, expected, progress, check)
            check()
            save_arrays(destination / "operators.npz", arrays)
            aggregate = diagnostic(arrays)
            write(destination / "diagnostic.json", aggregate)
            check()
            item = {"status": "completed", "fit": name, "source_weights_sha256": record["weights_sha256"],
                    "source_predictions_sha256": record["predictions_sha256"], "agreement": agreement,
                    "work": public_progress(progress), "wall_seconds": time.perf_counter() - fit_start,
                    "files": {f: {"sha256": sha(destination / f), "bytes": (destination / f).stat().st_size}
                              for f in ("operators.npz", "diagnostic.json")}}
            check()
            write(destination / "completed.json", item)
            records.append(item)
            # Completed costs and active partial work are disjoint.
            progress, model = {}, None
            check()
        authenticate(plan, check)
        bound_file(Path(args.plan), args.plan_sha256)
        files = {p.relative_to(out).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
                 for p in sorted(out.rglob("*")) if p.is_file()}
        check()
        write(out / "completed.json", {"status": "completed", "version": VERSION,
              "plan_sha256": args.plan_sha256, "fits": records, "files": files,
              "model_forward_calls": sum(r["work"]["model_forward_calls"] for r in records),
              "matched_rows": sum(r["work"]["matched_rows"] for r in records),
              "wall_seconds": time.perf_counter() - start, "new_fits": 0,
              "scope": "Frozen-model factors and one-step counterfactuals only; no new trajectory performance"})
        check()
        digest = sha(out / "completed.json")
        check()
        return {"completed_sha256": digest, "wall_seconds": time.perf_counter() - start}
    except BaseException as error:
        preserve_failure(out, error, progress, model, records, start)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    frozen = sub.add_parser("freeze")
    for name in ("old-study", "packet", "lexical", "out"):
        frozen.add_argument("--" + name, required=True)
    frozen.add_argument("--wall-cap-seconds", required=True, type=float)
    run = sub.add_parser("replay")
    for name in ("plan", "plan-sha256", "out"):
        run.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "replay": replay}[args.mode](args)), flush=True)


if __name__ == "__main__":
    main()
