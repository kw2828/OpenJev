"""Independent saved-output checks for a fixed protected-readout comparison.

The immutable action-focused auditor supplies scalar loss/metric arithmetic and
chronological input reconstruction, never producer loss or metric code. This
audit reads complete saved predictions/checkpoints and journals without running
a model, optimizer, teacher or simulator. Original process closure is required
for both fresh collection and training. A successful worker audit remains
conditional on its own original supervisor subsequently closing successfully.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/audit_otto_protected_readout.py"
TEST = "tests/test_audit_otto_protected_readout.py"
HELPER = "scripts/audit_otto_action_focused.py"
HELPER_PIN = "4ee295fc8d77b81b6df869811b471770f47af2d6f5a763e040a21ad7d0578729"
VERSION = "otto-protected-readout-saved-audit-v1"
PRODUCER = "scripts/train_otto_protected_readout.py"
PRODUCER_VERSION = "otto-protected-readout-training-v1"
COLLECTOR = "scripts/collect_otto_protected_readout.py"
COLLECTOR_VERSION = "otto-protected-readout-collection-v1"
FAMILIES = ("pretrained", "frozen_aux", "frozen_spo", "joint_aux", "joint_spo")
SEEDS = (301000001, 301000002, 301000003)
SELECTION_START = 302000001
REGIMES, ARMS = ("lambda3", "lambda4"), ("analytic", "neural", "period4_hold")
EPOCHS = {kind: 80 if kind == "pretrained" else 40 for kind in FAMILIES}
PARAMETERS = {kind: 5996 if kind == "pretrained" else 6112 for kind in FAMILIES}
CELLS = {kind: ("innovation_gru", "shared" if kind == "pretrained" else "protected",
                "aux" if kind == "pretrained" else kind.rsplit("_", 1)[1]) for kind in FAMILIES}
LIMITS = {"seconds": 300, "rss_bytes": 2 * 1024**3, "output_bytes": 256 * 1024**2}
CAPACITY_SCOPE = "Inherited original backbone evidence only; protected residual, staged fitting and current total runtime were not measured"
CONFIG = {"families": list(FAMILIES), "fit_seeds": list(SEEDS), "epochs": EPOCHS,
          "batch_episodes": 6, "chunk": 32, "learning_rate": .003, "gradient_clip": 5.,
          "weight_decay": 0., "scale": 64., "training_episodes": 54,
          "validation_episodes": 36, "required_conditions": 29,
          "checkpoint": "last", "device": "cpu", "dtype": "float32",
          "selection_seed_start": SELECTION_START,
          "training_weight": "(W/k)/(54*full_episode_nonquery_rows)",
          "batch_scale": "54/actual_batch_episodes", "tbptt": "detach every32; accumulate then one Adam step",
          "loss": "eligible-centered nonquery MSE plus all-four prequery MSE; spo adds nonquery SPO+; scores /64",
          "cells": {k: {"architecture": v[0], "readout": v[1], "objective": v[2]} for k, v in CELLS.items()},
          "prior_coefficient": 1., "spo_coefficient": 1., "residual_output_scale": 64.,
          "prior_weight": "1/(54*full_episode_later_query_rows)",
          "fork": "each seed pretrained80 then four fresh Adam40 branches from identical checkpoint and zero residual",
          "branch_order": "PCG64(fit_seed) restarted for each stage2 branch; no optimizer reuse",
          "compute_matching": "same branch exposure and optimizer schedule; actual differentiable chunks and costs counted",
          "evaluation": "fresh frozen-backbone clone; residual retains required True flag; outer no_grad",
          "capacity_evidence_scope": CAPACITY_SCOPE}
LIMITATIONS = [
    "Saved model outputs, gradients and physical timings are authenticated, never replayed through a model.",
    "Original teacher/filter/native numerical truth remains inherited from the closed collection.",
    "Frozen backbone parameters and complete TRAIN/VALID base outputs/prior are compared bitwise with pretraining.",
    "Canonical parameter flags and no_grad use are recorded execution claims; saved numeric parity is checked independently.",
    "Only the one prospective 29-condition protected_readout gate exists; no architecture or autonomous-policy claim.",
    "Shared-architecture capacity evidence does not measure this new residual adaptation or current objective costs.",
]
WORK_KEYS = ("forward_chunks", "backward_chunks", "no_grad_chunks", "forward_rows", "loss_chunks",
             "differentiable_chunks", "skipped_backward_chunks", "spo_loss_calls", "spo_weighted_rows")
FIT_KEYS = {"family", "architecture", "readout", "objective", "training_mode", "stage", "seed", "parameter_count",
    "trainable_parameter_count", "optimizer_parameter_names", "optimizer_initial_state_entries", "residual_output_scale",
    "checkpoint_path", "checkpoint", "initial_tensors", "initial_sha256", "initial_backbone_sha256", "initial_residual_zero",
    "final_backbone_sha256", "frozen_backbone_unchanged", "pretrained_checkpoint_path", "pretrained_checkpoint",
    "train_base_equals_pretrained", "evaluation", "epochs", "steps", "episode_exposures", "episodes_per_epoch",
    "episode_orders", "permutation_sha256", "final_train_loss", "final_nonquery_loss", "final_prior_loss",
    "final_objective_prior_loss", "final_prior_loss_scope", "final_spo_loss", "final_objective_spo_loss", "final_spo_loss_scope",
    "final_rescore_spo_loss_calls", "train_rescore", "fit_seconds", "evaluation_clone_seconds", "train_rescore_seconds",
    "checkpoint_seconds", "wall_seconds", *WORK_KEYS}


def load_helper():
    path = ROOT / HELPER
    if hashlib.sha256(path.read_bytes()).hexdigest() != HELPER_PIN:
        raise ValueError("immutable independent scalar audit helper")
    spec = importlib.util.spec_from_file_location("_protected_readout_independent_helper", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


old = load_helper()
require, write = old.require, old.write
scalar_metrics, scalar_prior_metrics = old.scalar_metrics, old.scalar_prior_metrics
scalar_training_loss, scalar_spo_loss = old.scalar_training_loss, old.scalar_spo_loss
CLOCK, CLOCK_PIN, SUPERVISOR, SUPERVISOR_PIN = old.CLOCK, old.CLOCK_PIN, old.SUPERVISOR, old.SUPERVISOR_PIN


def cohort():
    result, position = [], 0
    for stage, size, starts in (("train", 9, (297000001, 298000001)), ("valid", 6, (299000001, 300000001))):
        for regime, first in zip(REGIMES, starts, strict=True):
            for case in range(size):
                shift = position % 3
                for arm in ARMS[shift:] + ARMS[:shift]:
                    result.append({"stage": stage, "episode_index": len(result), "regime": regime,
                        "seed": first + case, "case": case, "initial_hit": 1 + case % 3,
                        "arm": arm, "episode_id": f"{stage}:{regime}:{first + case}:{arm}"})
                position += 1
    return result


# Only configuration in this private imported helper instance changes. Its
# source, independent scalar routines and historical capacity checks do not.
old.SELECTION_START = SELECTION_START
old.cohort = cohort
old.base.cohort = cohort
old.base.LIMITS = LIMITS


def evaluation_metadata(kind):
    require(kind in FAMILIES, "declared canonical evaluation family")
    return {"parameter_mode": "frozen", "grad_enabled": False, "backbone_requires_grad": False,
            "residual_requires_grad": None if kind == "pretrained" else True}


def condition_names():
    names = ["common.technical_complete"]
    for regime in REGIMES:
        names += [f"common.{regime}.initial.case_support"]
        names += [f"common.{regime}.age{age}.case_support" for age in (1, 2, 3)]
        names += [f"common.{regime}.hold.postcorrection.positive_gap"]
    for regime in REGIMES:
        names += [f"candidate.{regime}.postcorrection.gap_vs_{control}"
                  for control in ("pretrained", "frozen_aux", "joint_aux", "joint_spo")]
        names += [f"candidate.{regime}.{scope}.agreement" for scope in ("full", "initial")]
        names += [f"candidate.{regime}.{seed}.postcorrection.gap_vs_frozen_aux" for seed in SEEDS]
    return names


def continuation_rules(models, hold, *, technical_complete):
    require(type(technical_complete) is bool, "strict technical completion Boolean")
    require([(r["family"], r["seed"]) for r in models]
            == [(kind, seed) for seed in SEEDS for kind in FAMILIES], "all fifteen ordered model records")
    by = {(r["family"], r["seed"]): r for r in models}
    for row in models:
        expected = evaluation_metadata(row["family"])
        require(set(row["evaluation"]) == set(expected) and all(
            type(row["evaluation"][k]) is type(v) and row["evaluation"][k] == v for k, v in expected.items()),
            "canonical evaluation assertions")
    result = [{"name": "common.technical_complete", "value": technical_complete,
               "relation": "==", "threshold": True, "passes": technical_complete}]

    def emit(name, value, relation, threshold, strict=None):
        require(math.isfinite(value) and math.isfinite(threshold), "finite independent criterion")
        passes = value >= threshold if relation == ">=" else value > threshold if relation == ">" else value <= threshold
        row = {"name": name, "value": value, "relation": relation, "threshold": threshold, "passes": bool(passes)}
        if strict is not None:
            require(math.isfinite(strict), "finite strict comparator")
            row.update(relation="<= and <", strict_upper_bound=strict, passes=bool(passes and value < strict))
        result.append(row)

    def metric(kind, seed, scope, regime, key):
        return by[kind, seed]["metrics"][scope]["by_regime"][regime][key]

    def mean(kind, scope, regime, key):
        return math.fsum(metric(kind, seed, scope, regime, key) for seed in SEEDS) / 3

    gap_key, agree_key = "episode_weighted_raw_gap", "episode_weighted_agreement"
    for regime in REGIMES:
        emit(f"common.{regime}.initial.case_support", hold["initial"]["by_regime"][regime]["supported_case_count"], ">=", 4)
        later = hold["postcorrection"]["by_regime"][regime]
        for age in (1, 2, 3):
            emit(f"common.{regime}.age{age}.case_support", later["by_age"][str(age)]["supported_case_count"], ">=", 4)
        emit(f"common.{regime}.hold.postcorrection.positive_gap", later[gap_key], ">", 0.)
    for regime in REGIMES:
        candidate = mean("frozen_spo", "postcorrection", regime, gap_key)
        controls = ("pretrained", "frozen_aux", "joint_aux", "joint_spo")
        for control in controls:
            threshold = mean(control, "postcorrection", regime, gap_key)
            emit(f"candidate.{regime}.postcorrection.gap_vs_{control}", candidate, "<=", .9 * threshold, threshold)
        for scope in ("full", "initial"):
            emit(f"candidate.{regime}.{scope}.agreement", mean("frozen_spo", scope, regime, agree_key), ">=",
                 max(mean(control, scope, regime, agree_key) for control in controls))
        for seed in SEEDS:
            emit(f"candidate.{regime}.{seed}.postcorrection.gap_vs_frozen_aux",
                 metric("frozen_spo", seed, "postcorrection", regime, gap_key), "<=",
                 metric("frozen_aux", seed, "postcorrection", regime, gap_key))
    require([r["name"] for r in result] == condition_names(), "complete independent 29 conditions")
    return result


def gate_decisions(rows):
    require(len(rows) == 29 and [r["name"] for r in rows] == condition_names()
            and all(type(r["passes"]) is bool for r in rows), "exact ordered Boolean gate")
    return {"protected_readout": {"conditions": condition_names(), "passed": sum(r["passes"] for r in rows),
                                  "total": 29, "passes": all(r["passes"] for r in rows)}}


class Audit(old.Audit):
    def __init__(self, args):
        super().__init__(args)
        self.receipt.update(version=VERSION, limits=LIMITS, limitations=LIMITATIONS, teacher_calls=0)

    def execute(self):
        self.require(self.out.is_absolute() and self.out.is_relative_to(ROOT) and ".." not in self.out.parts
                     and not any(p.is_symlink() for p in self.out.parents), "exclusive canonical audit output")
        return super().execute()

    def admit(self):
        self.require(self.sha(self.path(CLOCK)) == CLOCK_PIN, "qualified native clock before import")
        spec = importlib.util.spec_from_file_location("_protected_saved_audit_clock", ROOT / CLOCK)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.clock = module.SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            self.require(self.clock.now_ns() - self.start < 5 * 10**9, "original audit launch available")
            time.sleep(.01)
        self.launch = self.read(self.args.supervision)
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        self.require(command == [sys.executable, *sys.argv] and self.launch["pid"] == os.getpid()
            and self.launch["pgid"] == os.getpgrp() and self.launch["parent_pid"] == os.getppid()
            and self.launch["cwd"] == str(ROOT) == str(Path.cwd()) and self.launch["cap_seconds"] == 300
            and self.launch["clock_backend"] == self.clock.backend
            and self.launch["clock_source_sha256"] == CLOCK_PIN and self.launch["watchdog_sha256"] == SUPERVISOR_PIN
            and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
            and self.launch["deadline_ns"] == self.launch["started_ns"] + 300 * 10**9,
            "actual bounded original saved-audit process")
        self.inputs = {}
        for role in ("plan", "worker", "terminal"):
            path = self.path(getattr(self.args, role))
            item = self.descriptor(path)
            self.require(item["sha256"] == getattr(self.args, role + "_sha256"), "external " + role)
            self.inputs[role] = item
        self.plan = self.read(self.args.plan)
        self.require(self.plan["version"] == PRODUCER_VERSION and self.plan["status"] == "frozen_before_fitting"
            and self.plan["config"] == CONFIG and self.plan["limits"] ==
            {"seconds": 14400, "rss_bytes": 4 * 1024**3, "output_bytes": 2 * 1024**3}, "fixed training recipe")
        self.require({SELF, TEST, CLOCK, SUPERVISOR, HELPER} <= self.plan["sources"].keys()
            and self.plan["sources"][HELPER] == HELPER_PIN, "independent auditor frozen before fitting")
        for name, pin in self.plan["sources"].items():
            self.require(self.sha(self.path(name)) == pin, "all unchanged frozen source " + name)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.require(os.environ.get(name) == "1", "single numerical thread")
        self.receipt.update(plan_sha256=self.args.plan_sha256, producer_inputs=self.inputs,
            sources={name: self.plan["sources"][name] for name in (SELF, TEST, CLOCK, SUPERVISOR, HELPER)},
            supervision_sha256=self.sha(self.args.supervision))
        write(self.out / "started.json", {"started_ns": self.start, "launch": self.launch, "producer_inputs": self.inputs})

    def authenticate(self):
        _, self.worker, self.parent, self.run = self.phase(
            self.args.plan, self.args.worker, self.args.terminal, PRODUCER, ".venv/bin/python", 14400)
        self.require(self.worker["version"] == PRODUCER_VERSION and self.worker["fits_completed"] == 15
            and self.worker["optimizer_steps"] == 6480
            and self.worker["teacher_calls"] == self.worker["native_calls"] == 0, "all fixed fits and updates")
        roles = self.plan["inputs"]
        self.require(set(roles) == {"collection_plan", "collection_receipt", "collection_terminal", "engineering",
                                   "capacity_plan", "capacity_receipt", "capacity_terminal"}, "exact training input roles")
        self.collection_plan, self.collection_worker, self.collection_parent, self.collection = self.phase(
            self.path(roles["collection_plan"]["path"]), self.path(roles["collection_receipt"]["path"]),
            self.path(roles["collection_terminal"]["path"]), COLLECTOR, ".venv-otto-released-native/bin/python", 7200)
        c = self.collection_worker
        payloads = {name + ".jsonl.gz" for name in ("work", "weights", "forwards", "transitions", "samples", "annotations")}
        payloads.update({"started.json", "runtime.json", "setup.json", "deployment.json", "cohort.json",
            "episode-boundaries.jsonl", "episodes.jsonl", "train.npz", "valid.npz", "train-selection.json", "costs.json", "summary.json"})
        self.require(self.collection_plan["version"] == COLLECTOR_VERSION
            and self.collection_plan["status"] == "frozen_before_collection"
            and self.collection_plan["limits"] == {"native_seconds": 7200, "rss_bytes": 4 * 1024**3, "output_bytes": 2 * 1024**3}
            and set(c["files"]) == set(self.collection_plan["payloads"]) == payloads, "fresh complete collection allocation")
        self.require(c["version"] == COLLECTOR_VERSION and c["completed_episodes"] == 90
            and c["train_episodes"] == 54 and c["valid_episodes"] == 36 and c["training_updates"] == 0,
            "all ninety fresh paths")
        self.equal(self.collection_plan["cohort"], cohort(), "exact prospectively selected fresh cohort")
        self.episodes = list(self.rows(self.collection / "episodes.jsonl"))
        self.require(len(self.episodes) == 90, "all complete episode records")
        for row, identity in zip(self.episodes, cohort(), strict=True):
            self.equal({key: row[key] for key in identity}, identity, "ordered episode identity")
            steps = row["steps"]
            self.require(type(steps) is int and 1 <= steps <= 2188 and row["rows"] == row["updates"] == steps
                and row["final_update_assimilated"] is True and type(row["found"]) is bool
                and row["censored"] is (not row["found"]) and (row["found"] or steps == 2188)
                and row["corrections"] == (steps + 3) // 4, "complete tails and query chronology")
            self.require(all(type(row[k]) is int and row[k] >= 0 for k in
                ("teacher_calls", "deployed_queries", "annotation_only", "deferred_annotations", "unlabeled_rows"))
                and row["teacher_calls"] == row["deployed_queries"] + row["annotation_only"]
                and row["teacher_calls"] + row["unlabeled_rows"] == steps
                and row["deferred_annotations"] <= row["annotation_only"], "all labeled-state calls accounted")
            if row["stage"] == "valid":
                self.require(row["teacher_calls"] == steps and row["unlabeled_rows"] == row["deferred_annotations"] == 0,
                             "full VALID teacher census")
        self.capacity()
        engineering = self.read(roles["engineering"]["path"])
        self.require(engineering["status"] == "passed" and engineering["sources_before"] == engineering["sources_after"]
            and engineering["results"] and all(type(r["exit_code"]) is int and r["exit_code"] == 0
            for r in engineering["results"]), "qualified fabricated engineering")

    def expected_batches(self, train, orders, family):
        np = self.np
        self.require(family in FAMILIES, "fixed batch family")
        offsets = train["episode_offsets"]
        lengths = [int(offsets[i+1] - offsets[i]) for i in range(54)]
        selected, prior = np.zeros((54, 69), np.bool_), np.zeros((54, 69), np.bool_)
        for i, length in enumerate(lengths):
            low = int(offsets[i])
            for start in range(0, length, 32):
                span = slice(low+start, low+min(length, start+32))
                selected[i, start//32] = bool((train["weights"][span] > 0).any())
                prior[i, start//32] = bool(train["prior_mask"][span].any())
        result = []
        frozen, objective = family.startswith("frozen_"), CELLS[family][2]
        for epoch, order in enumerate(orders, start=1):
            for batch in range(9):
                indices = order[batch*6:(batch+1)*6]
                chunks = (max(lengths[i] for i in indices)+31)//32
                active = selected[indices, :chunks].any(axis=0)
                prior_active = prior[indices, :chunks].any(axis=0)
                loss_chunks = int((active | prior_active).sum())
                backward = int((active if frozen else active | prior_active).sum())
                result.append({"epoch": epoch, "batch": batch, "episode_indices": indices,
                    "optimizer_step": len(result)+1, "objective": objective,
                    "forward_chunks": chunks, "loss_chunks": loss_chunks, "differentiable_chunks": backward,
                    "backward_chunks": backward, "skipped_backward_chunks": loss_chunks-backward,
                    "no_grad_chunks": chunks-backward, "forward_rows": sum(lengths[i] for i in indices),
                    "prior_rows": sum((lengths[i]+3)//4-1 for i in indices),
                    "spo_loss_calls": loss_chunks if objective == "spo" else 0,
                    "spo_weighted_rows": sum(int((train["weights"][int(offsets[i]):int(offsets[i+1])] > 0).sum())
                                             for i in indices) if objective == "spo" else 0,
                    "trainable_parameter_count": 116 if frozen else PARAMETERS[family],
                    "frozen_backbone_unchanged": True if frozen else None})
        return result

    def checkpoint(self, family, filename):
        shapes = {"recurrent.weight_ih_l0": (84, 40), "recurrent.weight_hh_l0": (84, 28),
                  "recurrent.bias_ih_l0": (84,), "recurrent.bias_hh_l0": (84,),
                  "output.weight": (4, 28), "output.bias": (4,)}
        if family != "pretrained":
            shapes.update({"action_residual.weight": (4, 28), "action_residual.bias": (4,)})
        values = self.arrays(self.run / filename)
        self.require(set(values) == set(shapes), "exact final checkpoint tensor membership")
        for key, shape in shapes.items():
            self.require(values[key].dtype == self.np.float32 and values[key].shape == shape
                and bool(self.np.isfinite(values[key]).all()), "finite final checkpoint tensor " + key)
        self.require(sum(v.size for v in values.values()) == PARAMETERS[family], "exact checkpoint parameter count")
        return values

    def witness(self, arrays, *, backbone_only=False):
        names = ["recurrent.weight_ih_l0", "recurrent.weight_hh_l0", "recurrent.bias_ih_l0",
                 "recurrent.bias_hh_l0", "output.weight", "output.bias"]
        if not backbone_only and "action_residual.weight" in arrays:
            names += ["action_residual.weight", "action_residual.bias"]
        digest, tensors = hashlib.sha256(), {}
        for name in names:
            raw = arrays[name].tobytes()
            tensors[name] = hashlib.sha256(raw).hexdigest()
            digest.update(name.encode() + raw)
        return digest.hexdigest(), tensors

    def same_frozen(self, saved, reference, label):
        for name, ref in (("base_predictions", "predictions"), ("prior", "prior"), ("prior_mask", "prior_mask")):
            a, b = saved[name], reference[ref]
            self.require(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
                         label + " frozen base equals pretrained: " + name)

    def training(self, train, meta):
        np = self.np
        record = self.read(self.run / "fits.json")
        self.equal(record["train_counts"], meta["counts"], "all chronological TRAIN exposure")
        self.require(set(record) == {"fits", "train_counts"}, "exact fit container")
        fits = record["fits"]
        self.require([(r["family"], r["seed"]) for r in fits]
            == [(family, seed) for seed in SEEDS for family in FAMILIES], "all fifteen stages in fixed order")
        progress, work = iter(self.rows(self.run / "progress.jsonl")), iter(self.rows(self.run / "work.jsonl"))
        work_keys = WORK_KEYS
        totals = {k: 0 for k in work_keys if k != "no_grad_chunks"}
        serial, parents, checkpoints = 0, {}, {}
        self.train_references = {}
        for fit in fits:
            self.require(set(fit) == FIT_KEYS, "exact complete fit schema")
            kind, seed = fit["family"], fit["seed"]
            frozen, pretrained = kind.startswith("frozen_"), kind == "pretrained"
            architecture, readout, objective = CELLS[kind]
            mode = "pretrain" if pretrained else kind.split("_", 1)[0]
            self.equal(next(progress), {"event": "fit_start", "family": kind, "seed": seed}, "ordered stage entry")
            rng, digest, orders = np.random.default_rng(seed), hashlib.sha256(), []
            for _ in range(EPOCHS[kind]):
                order = rng.permutation(54).astype(np.int64)
                digest.update(order.tobytes()); orders.append(order.tolist())
            self.equal(fit["episode_orders"], orders, "complete independent stage episode orders")
            self.require(fit["permutation_sha256"] == digest.hexdigest(), "exact ordered exposure digest")
            times, sums = [], {k: 0 for k in work_keys}
            for expected in self.expected_batches(train, orders, kind):
                serial += 1
                identity = {"call_id": serial, "family": kind, "seed": seed, "epoch": expected["epoch"]-1,
                            "batch": expected["batch"], "episode_indices": expected["episode_indices"]}
                self.equal(next(work), {"event": "attempt", **identity}, "every actual batch attempt")
                returned = next(work)
                self.require(set(returned) == {"event", *identity, "seconds", "result"}, "exact returned batch fields")
                self.equal({k: returned[k] for k in identity}, identity, "paired batch return identity")
                self.require(returned["event"] == "return" and type(returned["seconds"]) in (int, float)
                    and math.isfinite(returned["seconds"]) and returned["seconds"] >= 0, "finite returned batch interval")
                result = returned["result"]
                expected_result = {k: v for k, v in expected.items() if k not in ("epoch", "batch")}
                for key in ("loss", "nonquery_loss", "prior_loss", "spo_loss", "gradient_norm_before_clip"):
                    self.require(type(result[key]) in (int, float) and math.isfinite(result[key]) and result[key] >= 0,
                                 "finite batch scalar " + key)
                    expected_result[key] = result[key]
                expected_result["all_parameter_gradients_finite"] = True
                self.equal(result, expected_result, "complete independent batch work and lock geometry")
                component_sum = result["nonquery_loss"] + result["prior_loss"] + result["spo_loss"]
                self.require(abs(result["loss"]-component_sum) <= 2**-22 * component_sum
                    + 2 * expected["loss_chunks"] * 2**-149, "batch sum within original f32 bound")
                if objective == "aux":
                    self.require(result["spo_loss"] == 0, "AUX has no trained SPO component")
                for key in sums:
                    sums[key] += result[key]
                times.append(returned["seconds"])
            for epoch in range(20, EPOCHS[kind]+1, 20):
                self.equal(next(progress), {"event": "epoch", "family": kind, "seed": seed, "epoch": epoch,
                    "optimizer_steps": epoch*9, "episode_exposures": epoch*54}, "every fixed epoch checkpoint")
            self.equal(next(progress), {"event": "fit_complete", **fit}, "durable stage completion")
            name = f"{kind}-{seed}.npz"
            self.require(fit["checkpoint_path"] == name and fit["epochs"] == EPOCHS[kind]
                and fit["steps"] == 9*EPOCHS[kind] and fit["episode_exposures"] == 54*EPOCHS[kind]
                and fit["episodes_per_epoch"] == 54 and fit["parameter_count"] == PARAMETERS[kind]
                and fit["trainable_parameter_count"] == (116 if frozen else PARAMETERS[kind])
                and fit["architecture"] == architecture and fit["readout"] == readout and fit["objective"] == objective
                and fit["training_mode"] == mode and fit["stage"] == ("pretrain" if pretrained else "adaptation")
                and fit["optimizer_initial_state_entries"] == 0
                and fit["residual_output_scale"] == (None if pretrained else 64.), "complete fixed model/stage recipe")
            self.equal(fit["evaluation"], evaluation_metadata(kind), "canonical frozen evaluation metadata")
            self.equal(fit["checkpoint"], self.worker["files"][name], "published final checkpoint bytes")
            values = self.checkpoint(kind, name)
            checkpoints[kind, seed] = values
            final_backbone, _ = self.witness(values, backbone_only=True)
            self.require(fit["final_backbone_sha256"] == final_backbone, "actual saved final backbone hash")
            ordered_names = ["recurrent.weight_ih_l0", "recurrent.weight_hh_l0", "recurrent.bias_ih_l0",
                             "recurrent.bias_hh_l0", "output.weight", "output.bias"]
            if not pretrained:
                ordered_names += ["action_residual.weight", "action_residual.bias"]
            self.equal(fit["optimizer_parameter_names"], ordered_names[-2:] if frozen else ordered_names,
                       "exact ordered optimizer parameter ownership")
            self.require(set(fit["initial_tensors"]) == set(ordered_names) and all(
                type(v) is str and len(v) == 64 and all(c in "0123456789abcdef" for c in v)
                for v in [*fit["initial_tensors"].values(), fit["initial_sha256"], fit["initial_backbone_sha256"]]),
                "complete initial parameter hash witnesses")
            if pretrained:
                self.require(fit["initial_sha256"] == fit["initial_backbone_sha256"]
                    and fit["initial_residual_zero"] is None and fit["pretrained_checkpoint_path"] is None
                    and fit["pretrained_checkpoint"] is None and fit["frozen_backbone_unchanged"] is None
                    and fit["train_base_equals_pretrained"] is None, "ordinary pretraining has no fork claim")
                for key, shape in (("output.weight", (4, 28)), ("output.bias", (4,))):
                    self.require(fit["initial_tensors"][key] == hashlib.sha256(np.zeros(shape, np.float32).tobytes()).hexdigest(),
                                 "original pretraining starts with zero score readout")
                parents[seed] = fit
            else:
                parent, parent_values = parents[seed], checkpoints["pretrained", seed]
                initial = {**parent_values, "action_residual.weight": np.zeros((4, 28), np.float32),
                           "action_residual.bias": np.zeros(4, np.float32)}
                initial_hash, tensor_hashes = self.witness(initial)
                self.equal(fit["initial_tensors"], tensor_hashes, "actual pretrained tensors and zero residual fork")
                self.require(fit["initial_sha256"] == initial_hash and fit["initial_residual_zero"] is True
                    and fit["initial_backbone_sha256"] == parent["final_backbone_sha256"]
                    and fit["pretrained_checkpoint_path"] == parent["checkpoint_path"]
                    and fit["pretrained_checkpoint"] == parent["checkpoint"], "same complete pretraining checkpoint fork")
                self.require(fit["frozen_backbone_unchanged"] is (True if frozen else None)
                    and fit["train_base_equals_pretrained"] is (True if frozen else None), "exact declared frozen parity")
                if frozen:
                    for key in parent_values:
                        self.require(values[key].tobytes() == parent_values[key].tobytes(), "frozen saved backbone " + key)
            for key, value in sums.items():
                self.require(fit[key] == value, "complete stage work " + key)
                if key in totals:
                    totals[key] += value
            intervals = ("fit_seconds", "evaluation_clone_seconds", "train_rescore_seconds", "checkpoint_seconds")
            for key in (*intervals, "wall_seconds", "final_train_loss", "final_nonquery_loss", "final_prior_loss", "final_spo_loss"):
                self.require(type(fit[key]) in (int, float) and math.isfinite(fit[key]) and fit[key] >= 0,
                             "finite fit scalar " + key)
            self.require(math.fsum(times) <= fit["fit_seconds"] + 1e-9
                and math.fsum(fit[k] for k in intervals) <= fit["wall_seconds"] + 1e-9, "complete paid fit intervals")
            offsets = train["episode_offsets"]
            chunks = sum((max(int(offsets[i+1]-offsets[i]) for i in range(first, first+6))+31)//32
                         for first in range(0, 54, 6))
            self.equal(fit["train_rescore"], {"forward_chunks": chunks, "forward_rows": meta["counts"]["rows"],
                "prior_rows": meta["counts"]["prior_rows"]}, "complete canonical TRAIN rescore")
            self.training_predictions(fit, train, meta)
        self.equal(next(progress), {"event": "all_checkpoints_closed_before_VALID", "fits_completed": 15,
            "checkpoints": {f["checkpoint_path"]: f["checkpoint"] for f in fits}}, "all fifteen checkpoints before VALID")
        self.require(old.journal_exhausted(progress) and old.journal_exhausted(work), "no extra or missing stage events")
        self.require(serial == self.worker["optimizer_steps"] == 6480, "all 6480 fixed optimizer updates")
        for key, value in totals.items():
            self.require(self.worker["training_" + key] == value, "complete worker training accounting " + key)
        self.require(self.worker["training_rescore_chunks"] == sum(f["train_rescore"]["forward_chunks"] for f in fits)
            and self.worker["training_rescore_rows"] == 15*meta["counts"]["rows"], "all fifteen canonical TRAIN rescans")
        self.counts.update(fits=15, optimizer_steps=6480, training_events=67, work_events=12960)
        return fits

    def training_predictions(self, fit, train, meta):
        np = self.np
        saved = self.arrays(self.run / ("training-prediction-" + fit["checkpoint_path"]))
        self.require(set(saved) == {"predictions", "base_predictions", "prior", "prior_mask"}, "exact saved TRAIN fields")
        for name in ("predictions", "base_predictions"):
            values = saved[name]
            self.require(values.dtype == np.float32 and values.shape == train["targets"].shape
                and bool(np.isfinite(values).all()), "complete finite TRAIN " + name)
            mask = train["query_mask"]
            self.require(values[mask].tobytes() == train["query_scores"][mask].tobytes(), "TRAIN genuine query outputs")
        prior = self.checked_prior(saved, train)
        kind, seed = fit["family"], fit["seed"]
        if kind == "pretrained":
            self.require(saved["base_predictions"].tobytes() == saved["predictions"].tobytes(), "pretrained TRAIN action equals base")
            self.train_references[seed] = saved
        elif kind.startswith("frozen_"):
            self.same_frozen(saved, self.train_references[seed], "TRAIN")
        nonquery, nq_bound = scalar_training_loss(saved["predictions"], train["targets"], train["weights"], train["legal"], check=self.check)
        auxiliary, prior_bound = scalar_training_loss(prior, train["prior_targets"], train["prior_weights"], check=self.check)
        spo, spo_bound = scalar_spo_loss(saved["predictions"], train["targets"], train["weights"], train["legal"], check=self.check)
        for name, value, bound in (("final_nonquery_loss", nonquery, nq_bound), ("final_prior_loss", auxiliary, prior_bound),
                                   ("final_spo_loss", spo, spo_bound)):
            self.require(abs(fit[name]-value) <= bound, "independent final TRAIN scalar with derived f32 bound " + name)
        self.require(fit["final_prior_loss_scope"] == "constant diagnostic in frozen mode; trained in pretrain and joint modes"
            and fit["final_objective_prior_loss"] == fit["final_prior_loss"]
            and fit["final_spo_loss_scope"] == "post-fit diagnostic for every cell; trained only by spo cells"
            and fit["final_objective_spo_loss"] == (fit["final_spo_loss"] if fit["objective"] == "spo" else 0.)
            and fit["final_rescore_spo_loss_calls"] == 1, "trained versus descriptive objective components")
        self.equal(fit["final_train_loss"], fit["final_nonquery_loss"] + fit["final_prior_loss"]
                   + fit["final_objective_spo_loss"], "fixed final objective arithmetic")
        identities = [row for row in self.episodes if row["stage"] == "train"]
        history = {**train, **{key: meta[key] for key in ("episode_ids", "episode_regimes")}}
        self.train_reports.append({"family": kind, "seed": seed,
            "prior_metrics": scalar_prior_metrics(history, prior, identities, check=self.check),
            "base_equals_pretrained": True if kind.startswith("frozen_") else None,
            "loss_check": {"nonquery_float64": nonquery, "nonquery_roundoff_bound": nq_bound,
                           "prior_float64": auxiliary, "prior_roundoff_bound": prior_bound,
                           "spo_float64": spo, "spo_roundoff_bound": spo_bound,
                           "scope": "independent saved-f32 scalar arithmetic; no Torch or model replay"}})
        self.counts["training_prediction_files"] += 1

    def predictions(self, windows, meta, identities, fits, history):
        np = self.np
        models, hold, references = [], None, {}
        for kind, seed in [("hold", None), *[(r["family"], r["seed"]) for r in fits]]:
            path = "prediction-hold.npz" if kind == "hold" else f"prediction-{kind}-{seed}.npz"
            saved = self.arrays(self.run / path)
            self.require(set(saved) == ({"predictions"} if kind == "hold" else
                {"predictions", "base_predictions", "prior", "prior_mask"}), "exact saved VALID fields")
            for name in (("predictions",) if kind == "hold" else ("predictions", "base_predictions")):
                values = saved[name]
                self.require(values.dtype == np.float32 and values.shape == windows["targets"].shape
                    and bool(np.isfinite(values).all()), "complete finite VALID " + name)
                self.require(values[:, 0].tobytes() == windows["query_scores"].tobytes(), "exact observed VALID query outputs")
                padding = values[~windows["valid_mask"]]
                self.require(padding.tobytes() == np.zeros(padding.shape, np.float32).tobytes(), "exact positive-zero VALID padding")
            metrics = self.metrics(saved["predictions"], windows, meta, identities)
            if kind == "hold":
                expected = np.repeat(windows["query_scores"][:, None, :], 4, axis=1)
                expected[~windows["valid_mask"]] = 0
                self.require(saved["predictions"].tobytes() == expected.tobytes(), "exact whole hold-Q reference")
                hold = metrics
            else:
                prior = self.checked_prior(saved, history)
                if kind == "pretrained":
                    self.require(saved["base_predictions"].tobytes() == saved["predictions"].tobytes(), "pretrained VALID action equals base")
                    references[seed] = saved
                elif kind.startswith("frozen_"):
                    self.same_frozen(saved, references[seed], "VALID")
                models.append({"family": kind, "architecture": CELLS[kind][0], "readout": CELLS[kind][1],
                    "objective": CELLS[kind][2], "seed": seed, "evaluation": evaluation_metadata(kind),
                    "base_equals_pretrained": True if kind.startswith("frozen_") else None,
                    "metrics": metrics, "prior_metrics": scalar_prior_metrics(history, prior, identities, check=self.check)})
            self.counts["predictions"] += 1
            self.counts["forecast_metric_rows"] += meta["counts"]["nonquery_rows"]
        return models, hold

    def body(self):
        self.authenticate()
        import numpy as np
        self.np = np
        runtime = self.read(self.run / "runtime.json")
        self.require(runtime["threads"] == runtime["interop_threads"] == 1 and runtime["deterministic"] is True
            and runtime["cuda_used"] is runtime["mps_used"] is False, "deterministic one-thread CPU runtime")
        train, train_meta, _ = self.reconstruct("train")
        self.saved_windows("training-history", train, train_meta)
        self.train_reports = []
        fits = self.training(train, train_meta)
        del train
        windows, meta, identities = self.reconstruct("valid")
        self.saved_windows("validation-windows", windows, meta)
        history, history_meta = self.validation_history(windows, meta)
        history.update({key: meta[key] for key in ("episode_ids", "episode_regimes")})
        models, hold = self.predictions(windows, meta, identities, fits, history)
        lengths = windows["episode_lengths"]
        chunks = sum((max(int(v) for v in lengths[first:first+6])+31)//32 for first in range(0, 36, 6))
        for row in models:
            row["prediction_work"] = {"forward_chunks": chunks, "forward_rows": meta["counts"]["rows"],
                                      "prior_rows": history_meta["counts"]["prior_rows"]}
        self.require(self.worker["validation_forward_chunks"] == 15*chunks
            and self.worker["validation_forward_rows"] == 15*meta["counts"]["rows"], "all fifteen census evaluations")
        rules = continuation_rules(models, hold, technical_complete=False)
        expected = {"version": PRODUCER_VERSION, "models": models, "hold": hold, "required": rules,
            "required_passed": sum(r["passes"] for r in rules), "required_conditions": 29,
            "gates": gate_decisions(rules), "technical_complete_pending_saved_audit": True,
            "scientific_conditions": 28, "scientific_passed": sum(r["passes"] for r in rules[1:]),
            "requires_successful_original_supervisor_and_saved_audit": True,
            "train_counts": train_meta["counts"], "validation_counts": meta["counts"],
            "validation_history_counts": history_meta["counts"], "capacity_evidence_scope": CAPACITY_SCOPE,
            "scope": "Forced-path forecasts only; protected adaptation must beat all four controls. No autonomous utility, scenario shift, or architecture superiority claim."}
        summary = self.read(self.run / "summary.json")
        for name in ("setup_seconds", "fitting_seconds", "validation_seconds"):
            self.require(type(summary[name]) in (int, float) and math.isfinite(summary[name]) and summary[name] >= 0,
                         "finite paid training stage")
            expected[name] = summary[name]
        self.require(math.fsum(f["wall_seconds"] for f in fits) <= summary["fitting_seconds"]+1e-9
            and math.fsum(summary[k] for k in ("setup_seconds", "fitting_seconds", "validation_seconds"))
            <= self.worker["wall_seconds"]+1e-9, "whole original training interval containment")
        self.equal(summary, expected, "independent full provisional summary and all 28 scientific criteria")
        payloads = {"started.json", "runtime.json", "progress.jsonl", "work.jsonl", "fits.json", "summary.json",
            "training-history.npz", "training-history.json", "validation-history.npz", "validation-history.json",
            "validation-windows.npz", "validation-windows.json", "prediction-hold.npz"}
        for fit in fits:
            payloads.update((fit["checkpoint_path"], "prediction-"+fit["checkpoint_path"],
                             "training-prediction-"+fit["checkpoint_path"]))
        self.require(len(payloads) == 58 and set(self.worker["files"]) == payloads, "exact final 58-payload closure")
        for directory, worker in ((self.run, self.worker), (self.collection, self.collection_worker)):
            for name, pin in worker["files"].items():
                self.pinned(directory/name, pin)
        for item in self.plan["inputs"].values():
            self.pinned(self.path(item["path"]), item)
        final_rules = continuation_rules(models, hold, technical_complete=True)
        final = {**expected, "required": final_rules, "required_passed": sum(r["passes"] for r in final_rules),
                 "gates": gate_decisions(final_rules), "technical_complete_pending_saved_audit": False}
        self.counts.update(required_conditions=29, required_passed=final["required_passed"], collection_episodes=90,
            prediction_files=31, source_files=len(self.plan["sources"]), training_payloads=58, collection_payloads=18)
        self.result = {"version": VERSION, "agreement": True, "producer_summary": expected, "summary": final,
            "fits": fits, "training_forecast_checks": self.train_reports, "counts": dict(self.counts),
            "limitations": LIMITATIONS, "technical_condition_requires_successful_original_audit_supervisor": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "worker", "terminal"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    Audit(parser.parse_args()).execute()


if __name__ == "__main__":
    main()
