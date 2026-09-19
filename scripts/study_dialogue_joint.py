"""Paired independent/joint observations with fixed normalized readout/scalar heads.

No encoder calls occur here. New features must already have a completed,
externally pinned cache receipt. The V2 monitor is reused without mutation.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import study_dialogue_copy as old
import study_dialogue_copy_v2 as v2
import torch
from torch import nn

from openjev.research.dialogue_joint_memory import VERSION, DialogueJointMemory

ROOT = Path(__file__).resolve().parents[1]
STUDY = "dialogue-joint-v1"
OLD_PLAN = "9c39c3b27c7219190bc3f45fc342bc4da4eb408e622402a92ce51efb68ed3905"
OLD_COMPLETED = "768376e24824523b9a5c26923c14206e567fda4f3578ed1babdf7cbdfb1d89f4"
ARMS = ("independent_readout", "independent_scalar", "joint_readout", "joint_scalar")
SEEDS = (4101, 4102, 4103)
CONFIG = {**v2.CONFIG, "methods": list(ARMS), "heads": ["readout", "scalar"], "observations": ["independent", "joint"]}
CHECKS = {"heads": ["readout", "scalar"], "required_complete_fits": 12, "unseen_true_gain": .10,
          "unseen_dontcare_gain": .10, "unseen_macro_gain": .02, "maximum_seen_macro_deficit": .01,
          "unseen_micro_nll_nonworse": True, "strict_unseen_macro_wins": 2,
          "maximum_unseen_revision_deficit": .01, "total_head_checks": 14}
CAP = 3600.
TOLERANCE = 2e-6
SOURCES = v2.SOURCES + [
    "src/openjev/research/dialogue_joint_memory.py", "tests/test_dialogue_joint_memory.py",
    "scripts/prepare_dialogue_joint.py", "tests/test_prepare_dialogue_joint.py",
    "scripts/study_dialogue_joint.py", "tests/test_study_dialogue_joint.py",
    "scripts/report_dialogue_joint.py", "tests/test_report_dialogue_joint.py",
    "research/dialogue-joint-protocol.md"]
sha, write, runtime = old.sha, old.write, old.runtime
COUNT_KEYS, MAX_KEYS = v2.COUNT_KEYS, v2.MAX_KEYS
empty_invariants, merge_invariants = v2.empty_invariants, v2.merge_invariants
save_npz, save_weights = v2.save_npz, v2.save_weights


class MonitoredJointMemory(v2.MonitoredCopyMemoryV2, DialogueJointMemory):
    """Immutable V2 hooks surround the new shape-dispatched public update."""


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def file_check(path, expected):
    require(not Path(path).is_symlink() and sha(path) == expected, "Changed file: " + str(path))


def old_identity(path, check=lambda: None):
    path = Path(path)
    file_check(path / "plan.json", OLD_PLAN)
    file_check(path / "completed.json", OLD_COMPLETED)
    plan, done = read(path / "plan.json"), read(path / "completed.json")
    require(done["status"] == "completed" and done["fit_count"] == 15 and done["plan_sha256"] == OLD_PLAN,
            "Old study incomplete")
    require(plan["config"] == v2.CONFIG and plan["practical_checks"] == v2.CHECKS and plan["runtime"] == runtime(),
            "Original recipe/runtime changed")
    require(set(plan["source_sha256"]) == set(v2.SOURCES), "Original source closure")
    for name, digest in plan["source_sha256"].items():
        file_check(ROOT / name, digest)
        check()
    names = [f"{m}-{s}" for s in old.SEEDS for m in old.METHODS]
    require([f"{r['method']}-{r['seed']}" for r in done["fits"]] == names, "Old fit membership")
    for name, record in zip(names, done["fits"], strict=True):
        folder = path / "fits" / name
        require(read(folder / "completed.json") == record and record["status"] == "completed", "Old fit receipt")
        require(not (folder / "completed.json").is_symlink(), "Symlinked old fit receipt")
        check()
    # Only authenticated provenance and initialization digests are used. The
    # public receipt copy suffices; old trained weights/predictions are not read.
    return plan, {name: r for name, r in zip(names, done["fits"], strict=True)}


def path_map(args):
    return {k: str(Path(getattr(args, k)).resolve().relative_to(ROOT.resolve()))
            for k in ("packet", "lexical", "joint", "old_study")}


def cache_identity(args, parent, check=lambda: None):
    """Authenticate a completed offline cache, never instantiate its encoder."""
    from prepare_dialogue_joint import authenticate_cache

    directory = Path(args.joint)
    file_check(directory / "completed.json", args.joint_completed_sha256)
    receipt = read(directory / "completed.json")
    require(receipt["status"] == "completed" and receipt["test_contents_accessed"] is False,
            "Joint cache incomplete or wrong data scope")
    for key in ("packet_completed_sha256", "lexical_completed_sha256"):
        require(receipt[key] == parent[key], "Joint cache preparation lineage differs")
    result = authenticate_cache(directory, args.joint_completed_sha256)
    check()
    return result


def align_joint(cohorts, queries, index, embeddings, indices, check=lambda: None):
    """Bind every supplied question's full public prefix, independent of targets."""
    require(embeddings.ndim == 2 and embeddings.shape[1] == 384 and embeddings.dtype == np.float32,
            "Joint embedding shape/dtype")
    for start in range(0, len(embeddings), 4096):
        require(np.isfinite(embeddings[start:start+4096]).all(), "Nonfinite joint embedding")
        check()
    require(indices.ndim == 1 and indices.dtype == np.int64 and set(index["cohorts"]) == {"train", "dev"},
            "Joint index shape/dtype/cohorts")
    offset = 0
    for split in ("train", "dev"):
        require(len(cohorts[split]) == len(index["cohorts"][split]), "Joint cohort membership")
        for d, entry in zip(cohorts[split], index["cohorts"][split], strict=True):
            shape = d["layout"]["shape"][:3]
            qids = d["layout"]["query_ids"]
            require(entry == {"id": d["id"], "query_ids": qids, "offset": offset, "shape": shape},
                    "Joint public-prefix layout differs")
            size = math.prod(shape)
            require(offset+size <= len(indices), "Truncated joint index")
            ix = indices[offset:offset+size].reshape(shape)
            mask = np.arange(shape[2])[None, :] < np.array([len(queries[q]["candidates"]) for q in qids])[:, None]
            valid = np.broadcast_to(mask, shape)
            require((ix[~valid] == -1).all() and ((ix[valid] >= 0) & (ix[valid] < len(embeddings))).all(),
                    "Joint candidate support/index bounds")
            d["joint_layout"] = entry
            offset += size
            check()
    require(offset == len(indices), "Unaccounted joint index entries")


def load_inputs(args, plan, check=lambda: None):
    cohorts, queries, features, lexical = old.load_inputs(args.packet, args.lexical)
    check()
    index, embeddings, indices = cache_identity(args, plan, check)
    align_joint(cohorts, queries, index, embeddings, indices, check)
    return cohorts, queries, features, lexical, (embeddings, indices)


def make_batch(ds, queries, features, lexical, joint, mode):
    require(mode in ("independent", "joint"), "Unknown observation mode")
    actor, labels, bins = old.make_batch(ds, queries, features, lexical)
    if mode == "joint":
        b, t = actor[1].shape
        q, c = actor[4].shape[1:]
        observations = np.zeros((b, t, q, c, 384), np.float32)
        embeddings, indices = joint
        for i, d in enumerate(ds):
            layout = d["joint_layout"]
            dt, dq, dc = layout["shape"]
            ix = indices[layout["offset"]:layout["offset"]+dt*dq*dc].reshape(dt, dq, dc)
            valid = ix >= 0
            block = observations[i, :dt, :dq, :dc]
            block[valid] = embeddings[ix[valid]]
        actor[0] = torch.from_numpy(observations)
    return actor, labels, bins


def observation_work(ds, queries, actor):
    b, _ = actor[1].shape
    q, c = actor[4].shape[1:]
    return {"emitted_float32_scalars": actor[0].numel(), "emitted_bytes": actor[0].numel()*4,
            "turn_projection_positions": actor[0].numel()//384, "schema_query_projection_positions": b*q,
            "schema_candidate_projection_positions": b*q*c,
            "real_public_turns": sum(len(d["turns"]) for d in ds),
            "real_question_updates": sum(len(d["turns"])*len(d["layout"]["query_ids"]) for d in ds),
            "real_candidate_updates": sum(len(d["turns"])*sum(len(queries[qi]["candidates"])
                for qi in d["layout"]["query_ids"]) for d in ds)}


def expected_configurations():
    """Schema-only construction, isolated engineering seed; no forward/data call."""
    result = {}
    with torch.random.fork_rng(devices=[]):
        for arm in ARMS:
            torch.manual_seed(410)
            model = MonitoredJointMemory(arm.rsplit("_", 1)[1], projection_dim=CONFIG["projection_dim"],
                                        hidden_dim=CONFIG["hidden_dim"], gru_width=CONFIG["gru_width"])
            result[arm] = model.configuration()
    return result


def expected_members():
    return {"plan.json", "started.json", "references.npz", "completed.json"} | {
        f"fits/{arm}-{seed}/{name}" for seed in SEEDS for arm in ARMS
        for name in ("completed.json", "batches.jsonl", "dev-predictions.npz", "weights.pt")}


def validate(args, check=lambda: None):
    out = Path(args.out)
    file_check(out / "plan.json", args.plan_sha256)
    plan = read(out / "plan.json")
    require(plan["study"] == STUDY and plan["implementation_version"] == VERSION
            and plan["config"] == CONFIG and plan["practical_checks"] == CHECKS
            and plan["runtime"] == runtime() and plan["wall_cap_seconds"] == CAP
            and plan["normalization_tolerance"] == TOLERANCE, "Frozen configuration/runtime")
    require(set(plan["source_sha256"]) == set(SOURCES) and plan["paths"] == path_map(args), "Source/path closure")
    for name, digest in plan["source_sha256"].items():
        file_check(ROOT / name, digest)
        check()
    require(plan["old_plan_sha256"] == OLD_PLAN and plan["old_completed_sha256"] == OLD_COMPLETED, "Old lineage pins")
    prior, original = old_identity(args.old_study, check)
    require(all(plan[key] == prior[key] for key in (
        "loss_counts", "loss_weights", "optimizer_updates_per_fit")), "Original objective/update budget changed")
    for key in ("packet", "lexical"):
        pin = key + "_completed_sha256"
        require(plan[pin] == prior[pin], "Preparation lineage")
        old.authenticate(Path(getattr(args, key)), plan[pin])
        check()
    require(plan["joint_completed_sha256"] == args.joint_completed_sha256, "Joint cache external pin differs")
    require(plan["execution_files"] == 52 and plan["expected_fits"] == 12
            and set(plan["expected_configurations"]) == set(ARMS), "Frozen artifact/configuration contract")
    cache_identity(args, plan, check)
    return plan, original


def freeze(args):
    started = time.perf_counter()
    prior, _ = old_identity(args.old_study)
    cohorts, _, _, _, _ = load_inputs(args, prior)
    for key in ("packet", "lexical"):
        require(sha(Path(getattr(args, key)) / "completed.json") == prior[key + "_completed_sha256"], "Preparation lineage")
    counts = Counter(old.loss_bin(r["bin"]) for d in cohorts["train"] for r in d["queries"])
    require(set(counts) == {0, 1, 2}, "Missing training stratum")
    total = sum(counts.values())
    weights = [total/(3*counts[i]) for i in range(3)]
    require(weights == prior["loss_weights"] and {str(k): v for k, v in counts.items()} == prior["loss_counts"],
            "Original loss weights/counts")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        plan = {"study": STUDY, "implementation_version": VERSION, "config": CONFIG, "practical_checks": CHECKS,
                "runtime": runtime(), "wall_cap_seconds": CAP, "normalization_tolerance": TOLERANCE,
                "source_sha256": {name: sha(ROOT / name) for name in SOURCES}, "paths": path_map(args),
                "old_plan_sha256": OLD_PLAN, "old_completed_sha256": OLD_COMPLETED,
                "packet_completed_sha256": prior["packet_completed_sha256"],
                "lexical_completed_sha256": prior["lexical_completed_sha256"],
                "joint_completed_sha256": args.joint_completed_sha256,
                "expected_configurations": expected_configurations(),
                "loss_counts": dict(counts), "loss_weights": weights,
                "optimizer_updates_per_fit": math.ceil(len(cohorts["train"])/CONFIG["batch_size"])*CONFIG["epochs"],
                "execution_files": 52, "expected_fits": 12,
                "scope": "Paired2x2 fixed observation control; exposed development only; no new encoder or old-weight warm start"}
        write(out / "plan.json", plan)
        file_check(Path(args.joint) / "completed.json", args.joint_completed_sha256)
        for name, digest in plan["source_sha256"].items():
            file_check(ROOT / name, digest)
    except BaseException as error:
        old.failure(out, error, {"phase": "freeze"})
        raise
    return {"plan_sha256": sha(out / "plan.json"), "preparation_wall_seconds": time.perf_counter()-started}


def evaluate(model, ds, queries, features, lexical, joint, mode, destination, check, progress):
    started = time.perf_counter()
    model.eval()
    arrays = {k: [] for k in ("probabilities", "labels", "choice", "bin", "unseen", "dialogue", "time", "query")}
    progress["_evaluation_arrays"] = arrays
    total, shapes, observations, batches = empty_invariants(), Counter(), Counter(), []
    progress["evaluation_invariants"] = total
    progress["evaluation_actor_shapes"] = shapes
    progress["evaluation_observation_work"] = observations
    progress["evaluation_rows"] = 0
    with torch.inference_mode():
        for start in range(0, len(ds), CONFIG["batch_size"]):
            check()
            indices = list(range(start, min(start+CONFIG["batch_size"], len(ds))))
            subset = [ds[i] for i in indices]
            actor, _, _ = make_batch(subset, queries, features, lexical, joint, mode)
            work = old.work_counts(subset, actor)
            emitted = observation_work(subset, queries, actor)
            shapes.update(work)
            observations.update(emitted)
            model.begin_batch(actor, subset)
            progress["active_batch"] = {"phase": "evaluation", "indices": indices, "stage": "forward_started"}
            try:
                probabilities = model(*actor).softmax(-1).numpy()
            finally:
                merge_invariants(total, model.audit)
            progress["active_batch"]["stage"] = "forward_returned"
            check()
            for i, d in enumerate(subset):
                positions = {qi: j for j, qi in enumerate(d["layout"]["query_ids"])}
                for row in d["queries"]:
                    n = len(queries[row["query"]]["candidates"])
                    p = probabilities[i, row["time"], positions[row["query"]], :n]
                    require(n <= 12 and np.isfinite(p).all() and np.isclose(p.sum(), 1, atol=1e-5), "Invalid output probability")
                    padded = np.zeros(12, np.float32)
                    padded[:n] = p
                    for key, value in (("probabilities", padded), ("labels", row["label"]), ("choice", int(p.argmax())),
                                      ("bin", row["bin"]), ("unseen", row["unseen"]), ("dialogue", d["id"]),
                                      ("time", row["time"]), ("query", row["query"])):
                        arrays[key].append(value)
            progress["evaluation_rows"] = len(arrays["labels"])
            batches.append({"indices": indices, "actor_shapes": work, "observation_work": emitted, "invariants": dict(model.audit)})
            progress["active_batch"]["stage"] = "rows_extracted"
    save_npz(destination, arrays)
    check()
    progress.pop("_evaluation_arrays")
    return {"queries": len(arrays["labels"]), "wall_seconds": time.perf_counter()-started,
            "actor_shapes": dict(shapes), "observation_work": dict(observations), "invariants": total, "batches": batches,
            "scope": "Includes assembly, monitored forward, extraction and saved prediction I/O; encoder excluded"}


def preserve_failure(root, error, records, progress, model, active, start):
    def preserve(fn, label):
        try:
            fn()
        except BaseException as secondary:  # noqa: BLE001 - retain original failure
            error.add_note(label + ": " + str(secondary))
    if (root / "completed.json").exists():
        preserve(lambda: (root / "completed.json").rename(root / "invalid-completion.json"), "Terminal demotion")
    if model is not None and active is not None:
        preserve(lambda: save_weights(active / "partial-weights.pt", model), "Partial weights")
    if active is not None and progress.get("_evaluation_arrays", {}).get("labels"):
        preserve(lambda: save_npz(active / "partial-predictions.npz", progress["_evaluation_arrays"]), "Partial predictions")
    public = {k: v for k, v in progress.items() if not k.startswith("_")}
    old.failure(root, error, {"completed_fits": records, "active_progress": public,
                             "wall_seconds": time.perf_counter()-start, "resume_authorized": False})


def train(args):
    started = time.perf_counter()
    root = Path(args.out)
    file_check(root / "plan.json", args.plan_sha256)
    require({p.name for p in root.iterdir()} == {"plan.json"}, "Execution already attempted; no retry")
    records, progress, model, active = [], {}, None, None
    def check():
        if time.perf_counter()-started > CAP:
            raise TimeoutError("Whole joint-observation study wall cap exceeded")
    try:
        write(root / "started.json", {"status": "started", "plan_sha256": args.plan_sha256,
                                        "runtime": runtime(), "wall_cap_seconds": CAP})
        check()
        plan, originals = validate(args, check)
        cohorts, queries, features, lexical, joint = load_inputs(args, plan, check)
        check()
        torch.set_num_threads(CONFIG["threads"])
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        (root / "fits").mkdir(exist_ok=False)
        weight = torch.tensor(plan["loss_weights"], dtype=torch.float32)
        data = cohorts["train"]
        references = old.reference_predictions(cohorts["dev"], queries, root / "references.npz")
        check()
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            orders = [rng.permutation(len(data)) for _ in range(CONFIG["epochs"])]
            paired = None
            for arm in ARMS:
                mode, method = arm.rsplit("_", 1)
                check()
                model, active = None, None
                name = f"{arm}-{seed}"
                progress = {"fit": name, "phase": "construction", "optimizer_steps_attempted": 0,
                            "completed_optimizer_steps": 0, "completed_supervised_queries": 0,
                            "completed_epoch_losses": [], "training_actor_shapes": Counter(),
                            "training_observation_work": Counter(),
                            "training_invariants": empty_invariants(), "active_batch": None}
                active = root / "fits" / name
                active.mkdir(exist_ok=False)
                fit_start = time.perf_counter()
                torch.manual_seed(seed)
                model = MonitoredJointMemory(method, projection_dim=CONFIG["projection_dim"],
                                              hidden_dim=CONFIG["hidden_dim"], gru_width=CONFIG["gru_width"])
                initial = old.tensor_digest(model)
                original_initial = originals[f"{method}-{seed}"]["initial_tensors_sha256"]
                require(initial == original_initial, "Original V2 initialization changed")
                paired = initial if paired is None else paired
                require(initial == paired, "All four arms require identical full initialization")
                require(model.configuration() == plan["expected_configurations"][arm], "Frozen model configuration changed")
                optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
                losses, count_updates, count_queries = [], 0, 0
                model.train()
                with (active / "batches.jsonl").open("x") as ledger:
                    for epoch, order in enumerate(orders):
                        loss_sum, count = 0., 0
                        for begin in range(0, len(order), CONFIG["batch_size"]):
                            check()
                            indices = [int(i) for i in order[begin:begin+CONFIG["batch_size"]]]
                            subset = [data[i] for i in indices]
                            actor, labels, bins = make_batch(subset, queries, features, lexical, joint, mode)
                            work = old.work_counts(subset, actor)
                            emitted = observation_work(subset, queries, actor)
                            progress["training_actor_shapes"].update(work)
                            progress["training_observation_work"].update(emitted)
                            progress["phase"] = "training"
                            progress["active_batch"] = {"epoch": epoch, "indices": indices, "stage": "forward_started"}
                            model.begin_batch(actor, subset)
                            optimizer.zero_grad(set_to_none=True)
                            try:
                                scores = model(*actor)
                            finally:
                                merge_invariants(progress["training_invariants"], model.audit)
                            progress["active_batch"]["stage"] = "forward_returned"
                            check()
                            eligible = labels != -100
                            ce = nn.functional.cross_entropy(scores[eligible], labels[eligible], reduction="none")
                            loss = (ce*weight[bins[eligible]]).mean()
                            require(torch.isfinite(loss).item(), "Nonfinite loss")
                            loss.backward()
                            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"], error_if_nonfinite=True)
                            progress["active_batch"]["stage"] = "backward_returned"
                            check()
                            progress["optimizer_steps_attempted"] += 1
                            progress["active_batch"]["stage"] = "optimizer_started"
                            optimizer.step()
                            count_updates += 1
                            n = int(eligible.sum())
                            count_queries += n
                            progress["completed_optimizer_steps"] = count_updates
                            progress["completed_supervised_queries"] = count_queries
                            progress["active_batch"]["stage"] = "optimizer_returned"
                            value = float(loss.detach())
                            count += n
                            loss_sum += value*n
                            ledger.write(json.dumps({"epoch": epoch, "update": count_updates, "indices": indices,
                                                     "loss": value, "supervised_queries": n, "actor_shapes": work,
                                                     "observation_work": emitted,
                                                     "invariants": model.audit}, sort_keys=True, allow_nan=False)+"\n")
                            ledger.flush()
                            progress["active_batch"]["stage"] = "ledger_flushed"
                            check()
                        losses.append(loss_sum/count)
                        progress["completed_epoch_losses"] = list(losses)
                        print(json.dumps({"method": arm, "head": method, "observation": mode, "seed": seed, "epoch": epoch+1,
                                          "training_loss": losses[-1], "seconds": time.perf_counter()-fit_start}), flush=True)
                require(count_updates == plan["optimizer_updates_per_fit"], "Update coverage")
                train_seconds = time.perf_counter()-fit_start
                progress["phase"] = "final_serialization"
                save_weights(active / "weights.pt", model)
                check()
                progress["phase"] = "evaluation"
                evaluation = evaluate(model, cohorts["dev"], queries, features, lexical, joint, mode,
                                      active / "dev-predictions.npz", check, progress)
                require(model.configuration() == plan["expected_configurations"][arm], "Final model configuration changed")
                row = {"method": arm, "observation": mode, "head": method, "seed": seed, "status": "completed", "epochs": len(losses),
                       "updates": count_updates, "training_queries": count_queries, "training_losses": losses,
                       "training_actor_shapes": dict(progress["training_actor_shapes"]),
                       "training_observation_work": dict(progress["training_observation_work"]),
                       "training_invariants": dict(progress["training_invariants"]),
                       "initial_tensors_sha256": initial, "original_initial_tensors_sha256": original_initial,
                       "common_initial_tensors_sha256": initial,
                       "parameters": sum(p.numel() for p in model.parameters()),
                       "parameters_with_final_gradient": sum(p.numel() for p in model.parameters() if p.grad is not None),
                       "configuration": model.configuration(), "train_wall_seconds": train_seconds,
                       "evaluation": evaluation, "weights_sha256": sha(active / "weights.pt"),
                       "predictions_sha256": sha(active / "dev-predictions.npz"), "batches_sha256": sha(active / "batches.jsonl"),
                       "fit_wall_seconds_before_receipt": time.perf_counter()-fit_start}
                check()
                write(active / "completed.json", row)
                records.append(row)
                progress, model, active = {}, None, None
                check()
        validate(args, check)
        files = {p.relative_to(root).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
                 for p in sorted(root.rglob("*")) if p.is_file()}
        require(set(files) == expected_members()-{"completed.json"}, "Exact execution payload closure")
        check()
        write(root / "completed.json", {"status": "completed", "study": STUDY, "fit_count": len(records), "fits": records,
              "plan_sha256": args.plan_sha256, "references": references, "files": files,
              "wall_seconds": time.perf_counter()-started, "external_model_api_calls": 0,
              "normalization_tolerance": TOLERANCE, "resume_authorized": False,
              "joint_completed_sha256": plan["joint_completed_sha256"], "encoder_calls": 0,
              "scope": "Paired2x2 fixed observation control; exposed development only; no architecture claim"})
        check()
        digest = sha(root / "completed.json")
        check()
        return {"completed_sha256": digest, "wall_seconds": time.perf_counter()-started}
    except BaseException as error:
        preserve_failure(root, error, records, progress, model, active, started)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "train"))
    for name in ("packet", "lexical", "joint", "old-study", "out"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--joint-completed-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "train": train}[args.mode](args)), flush=True)


if __name__ == "__main__":
    main()
