"""Fresh, normalized replication of the fixed candidate-copy development study."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import study_dialogue_copy as old
import torch
from torch import nn

from openjev.research.dialogue_copy_memory_v2 import VERSION, DialogueCopyMemoryV2

ROOT = Path(__file__).resolve().parents[1]
STUDY = "dialogue-copy-v2"
OLD_PLAN = "609a0758f2a4ae4722052bb27740c74b3d1bdac396d8134eb39f5118f1429a2b"
OLD_COMPLETED = "8317abad5ef8235f31f5381591944ad152c42fddb98743fa64fd1f5cd8c3f300"
CONFIG = {**old.CONFIG}
CHECKS = old.PRACTICAL_CHECKS
CAP = 3600.
TOLERANCE = 2e-6
SOURCES = old.SOURCES + [
    "src/openjev/research/dialogue_copy_memory_v2.py", "tests/test_dialogue_copy_memory_v2.py",
    "scripts/study_dialogue_copy_v2.py", "tests/test_study_dialogue_copy_v2.py",
    "scripts/report_dialogue_copy_v2.py", "tests/test_report_dialogue_copy_v2.py",
    "research/dialogue-copy-v2-protocol.md"]
sha, write, runtime = old.sha, old.write, old.runtime
COUNT_KEYS = ("forward_calls", "forward_returned", "advance_calls", "advance_returned", "valid_turns",
              "executed_valid_question_slots", "real_question_updates", "incoming_checks", "feature_checks",
              "result_checks", "mass_checks", "mass_above_one_count")
MAX_KEYS = ("incoming_max_sum_error", "feature_max_sum_error", "result_max_sum_error", "mass_max_overshoot")


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
    require(plan["config"] == CONFIG and plan["practical_checks"] == CHECKS and plan["runtime"] == runtime(),
            "Original recipe/runtime changed")
    require(set(plan["source_sha256"]) == set(old.SOURCES), "Original source closure")
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
            for k in ("packet", "lexical", "old_study")}


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
    return plan, original


def freeze(args):
    prior, _ = old_identity(args.old_study)
    cohorts, _, _, _ = old.load_inputs(args.packet, args.lexical)
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
                "loss_counts": dict(counts), "loss_weights": weights,
                "optimizer_updates_per_fit": math.ceil(len(cohorts["train"])/CONFIG["batch_size"])*CONFIG["epochs"],
                "scope": "Fresh fits under corrected numerical implementation; exposed development only; no old-weight warm start"}
        write(out / "plan.json", plan)
    except BaseException as error:
        old.failure(out, error, {"phase": "freeze"})
        raise
    return {"plan_sha256": sha(out / "plan.json")}


def empty_invariants():
    return {**dict.fromkeys(COUNT_KEYS, 0), **dict.fromkeys(MAX_KEYS, 0.),
            "mass_min": None, "mass_max": None, "tolerance": TOLERANCE}


def merge_invariants(total, batch):
    for key in COUNT_KEYS:
        total[key] += batch[key]
    for key in MAX_KEYS:
        total[key] = max(total[key], batch[key])
    if batch["mass_checks"]:
        total["mass_min"] = batch["mass_min"] if total["mass_min"] is None else min(total["mass_min"], batch["mass_min"])
        total["mass_max"] = batch["mass_max"] if total["mass_max"] is None else max(total["mass_max"], batch["mass_max"])


class MonitoredCopyMemoryV2(DialogueCopyMemoryV2):
    """Read-only checks on actual feature tensors and recurrent state, not labels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit = empty_invariants()
        self.audit_query_mask = None
        self.active_check = None
        self.feature.register_forward_pre_hook(self._feature_check)

    def begin_batch(self, actor, dialogues):
        self.audit = empty_invariants()
        self.audit_query_mask = torch.arange(actor[4].shape[1])[None] < torch.tensor(
            [d["layout"]["shape"][1] for d in dialogues])[:, None]
        require(self.audit_query_mask.shape == actor[4].shape[:2], "Audit query layout")

    def _probabilities(self, value, candidate_mask, selected, name):
        value = value.detach()
        chosen = value[selected]
        supported = candidate_mask[selected]
        require(torch.isfinite(chosen).all().item() and (chosen >= 0).all().item()
                and (chosen[~supported] == 0).all().item(), "Invalid " + name + " probabilities/support")
        error = (chosen.to(torch.float64).sum(-1)-1).abs().max().item() if chosen.shape[0] else 0.
        self.audit[name + "_max_sum_error"] = max(self.audit[name + "_max_sum_error"], error)
        require(error <= TOLERANCE, name + " normalization defect")
        self.audit[name + "_checks"] += int(selected.sum())

    def _feature_check(self, module, arguments):
        require(self.active_check is not None, "Unscoped feature audit")
        mask, selected = self.active_check
        self._probabilities(arguments[0][..., -2], mask, selected, "feature")

    def _advance(self, state, turn, valid, query, candidates, mask, lexical):
        require(self.audit_query_mask is not None and self.audit_query_mask.shape == mask.shape[:2], "Missing audit layout")
        selected = valid[:, None].expand_as(mask[..., 0])
        self.audit["advance_calls"] += 1
        self.audit["valid_turns"] += int(valid.sum())
        self.audit["executed_valid_question_slots"] += int(selected.sum())
        self.audit["real_question_updates"] += int((selected & self.audit_query_mask).sum())
        self._probabilities(state["log_b"].detach().exp(), mask, selected, "incoming")
        self.active_check = mask, selected
        try:
            proposed, diagnostic = super()._advance(state, turn, valid, query, candidates, mask, lexical)
            self.audit["advance_returned"] += 1
        finally:
            self.active_check = None
        self._probabilities(proposed["log_b"].detach().exp(), mask, selected, "result")
        if "departure_mass" in diagnostic:
            mass = diagnostic["departure_mass"].detach()[..., 0][selected].to(torch.float64)
            if mass.numel():
                require(torch.isfinite(mass).all().item(), "Nonfinite released mass")
                minimum, maximum = mass.min().item(), mass.max().item()
                self.audit["mass_min"] = minimum if self.audit["mass_min"] is None else min(self.audit["mass_min"], minimum)
                self.audit["mass_max"] = maximum if self.audit["mass_max"] is None else max(self.audit["mass_max"], maximum)
                self.audit["mass_max_overshoot"] = max(self.audit["mass_max_overshoot"], maximum-1)
                self.audit["mass_above_one_count"] += int((mass > 1).sum())
                require(minimum >= 0 and maximum <= 1+TOLERANCE, "Invalid released mass")
                self.audit["mass_checks"] += int(selected.sum())
        return proposed, diagnostic

    def forward(self, *args, **kwargs):
        self.audit["forward_calls"] += 1
        output = super().forward(*args, **kwargs)
        self.audit["forward_returned"] += 1
        return output


def save_npz(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **{k: np.asarray(v) for k, v in arrays.items()})


def save_weights(path, model):
    with Path(path).open("xb") as stream:
        torch.save(model.state_dict(), stream)


def evaluate(model, ds, queries, features, lexical, destination, check, progress):
    started = time.perf_counter()
    model.eval()
    arrays = {k: [] for k in ("probabilities", "labels", "choice", "bin", "unseen", "dialogue", "time", "query")}
    progress["_evaluation_arrays"] = arrays
    total, shapes, batches = empty_invariants(), Counter(), []
    progress["evaluation_invariants"] = total
    progress["evaluation_actor_shapes"] = shapes
    progress["evaluation_rows"] = 0
    with torch.inference_mode():
        for start in range(0, len(ds), CONFIG["batch_size"]):
            check()
            indices = list(range(start, min(start+CONFIG["batch_size"], len(ds))))
            subset = [ds[i] for i in indices]
            actor, _, _ = old.make_batch(subset, queries, features, lexical)
            work = old.work_counts(subset, actor)
            shapes.update(work)
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
            batches.append({"indices": indices, "actor_shapes": work, "invariants": dict(model.audit)})
            progress["active_batch"]["stage"] = "rows_extracted"
    save_npz(destination, arrays)
    check()
    progress.pop("_evaluation_arrays")
    return {"queries": len(arrays["labels"]), "wall_seconds": time.perf_counter()-started,
            "actor_shapes": dict(shapes), "invariants": total, "batches": batches,
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
            raise TimeoutError("Whole V2 training wall cap exceeded")
    try:
        write(root / "started.json", {"status": "started", "plan_sha256": args.plan_sha256,
                                        "runtime": runtime(), "wall_cap_seconds": CAP})
        check()
        plan, originals = validate(args, check)
        cohorts, queries, features, lexical = old.load_inputs(args.packet, args.lexical)
        check()
        torch.set_num_threads(CONFIG["threads"])
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        (root / "fits").mkdir(exist_ok=False)
        weight = torch.tensor(plan["loss_weights"], dtype=torch.float32)
        data = cohorts["train"]
        references = old.reference_predictions(cohorts["dev"], queries, root / "references.npz")
        check()
        for seed in old.SEEDS:
            rng = np.random.default_rng(seed)
            orders = [rng.permutation(len(data)) for _ in range(CONFIG["epochs"])]
            paired = None
            paired_common = None
            for method in old.METHODS:
                check()
                model, active = None, None
                name = f"{method}-{seed}"
                progress = {"fit": name, "phase": "construction", "optimizer_steps_attempted": 0,
                            "completed_optimizer_steps": 0, "completed_supervised_queries": 0,
                            "completed_epoch_losses": [], "training_actor_shapes": Counter(),
                            "training_invariants": empty_invariants(), "active_batch": None}
                active = root / "fits" / name
                active.mkdir(exist_ok=False)
                fit_start = time.perf_counter()
                torch.manual_seed(seed)
                model = MonitoredCopyMemoryV2(method, projection_dim=CONFIG["projection_dim"],
                                              hidden_dim=CONFIG["hidden_dim"], gru_width=CONFIG["gru_width"])
                initial = old.tensor_digest(model)
                require(initial == originals[name]["initial_tensors_sha256"], "Original initialization changed")
                common = {k: v for k, v in model.state_dict().items() if not k.startswith(("gru.", "gru_head."))}
                common_initial = old.tensor_digest(SimpleNamespace(state_dict=lambda common=common: common))
                paired_common = common_initial if paired_common is None else paired_common
                require(common_initial == paired_common, "Unpaired common initialization")
                if method in ("scalar", "selective", "selective_no_lexical"):
                    paired = initial if paired is None else paired
                    require(initial == paired, "Unpaired transport initialization")
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
                            actor, labels, bins = old.make_batch(subset, queries, features, lexical)
                            work = old.work_counts(subset, actor)
                            progress["training_actor_shapes"].update(work)
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
                                                     "invariants": model.audit}, sort_keys=True, allow_nan=False)+"\n")
                            ledger.flush()
                            progress["active_batch"]["stage"] = "ledger_flushed"
                            check()
                        losses.append(loss_sum/count)
                        progress["completed_epoch_losses"] = list(losses)
                        print(json.dumps({"method": method, "seed": seed, "epoch": epoch+1,
                                          "training_loss": losses[-1], "seconds": time.perf_counter()-fit_start}), flush=True)
                require(count_updates == plan["optimizer_updates_per_fit"], "Update coverage")
                train_seconds = time.perf_counter()-fit_start
                progress["phase"] = "final_serialization"
                save_weights(active / "weights.pt", model)
                check()
                progress["phase"] = "evaluation"
                evaluation = evaluate(model, cohorts["dev"], queries, features, lexical, active / "dev-predictions.npz", check, progress)
                row = {"method": method, "seed": seed, "status": "completed", "epochs": len(losses),
                       "updates": count_updates, "training_queries": count_queries, "training_losses": losses,
                       "training_actor_shapes": dict(progress["training_actor_shapes"]),
                       "training_invariants": dict(progress["training_invariants"]),
                       "initial_tensors_sha256": initial, "original_initial_tensors_sha256": originals[name]["initial_tensors_sha256"],
                       "common_initial_tensors_sha256": common_initial,
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
        check()
        write(root / "completed.json", {"status": "completed", "study": STUDY, "fit_count": len(records), "fits": records,
              "plan_sha256": args.plan_sha256, "references": references, "files": files,
              "wall_seconds": time.perf_counter()-started, "external_model_api_calls": 0,
              "normalization_tolerance": TOLERANCE, "resume_authorized": False,
              "scope": "Fresh normalized fits; unchanged13 criteria; exposed development only"})
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
    for name in ("packet", "lexical", "old-study", "out"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "train": train}[args.mode](args)), flush=True)


if __name__ == "__main__":
    main()
