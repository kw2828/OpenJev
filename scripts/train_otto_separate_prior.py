"""Matched shared/separate prior-readout fitting; no teacher or environment calls.

Metadata admission precedes numerical imports. The complete twenty-four-checkpoint
barrier precedes VALID decoding. All model inputs exclude skipped labels; only
selected nonquery rows and objective-specific query priors supply loss. Parameters stay fixed throughout
each chronological episode batch, with detached state at 32-step boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/train_otto_separate_prior.py"
TEST = "tests/test_train_otto_separate_prior.py"
MODELS = "src/openjev/research/otto_prequery_scores.py"
SEPARATE_MODELS = "src/openjev/research/otto_separate_prior_scores.py"
DATA = "src/openjev/research/otto_prequery_data.py"
LOSS = "src/openjev/research/otto_prequery_loss.py"
METRICS = "src/openjev/research/otto_separate_prior_metrics.py"
WINDOWS = "src/openjev/research/otto_score_forecast_data.py"
SAMPLING = "src/openjev/research/otto_sampled_forecast_data.py"
PROTOCOL = "research/otto-separate-prior-protocol.md"
AUDIT = "scripts/audit_otto_separate_prior.py"
AUDIT_TEST = "tests/test_audit_otto_separate_prior.py"
COLLECTOR = "scripts/collect_otto_separate_prior.py"
CAPACITY = "scripts/qualify_otto_separate_prior_capacity.py"
BASE_MODELS = "src/openjev/research/otto_cross_query_scores.py"
BASE_MODELS_PIN = "799979c0c60460352df076750d15b2cb9a5db6b6976707605dcca7c90fa76e08"
CLOCK = "src/openjev/research/suspend_clock.py"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SUPERVISOR_PIN = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
VERSION = "otto-separate-prior-training-v1"
COLLECTION_VERSION = "otto-separate-prior-collection-v1"
KINDS = ("innovation_shared_mse", "innovation_shared_aux", "innovation_separate_mse", "innovation_separate_aux",
         "gru_shared_mse", "gru_shared_aux", "gru_separate_mse", "gru_separate_aux")
CELLS = {"innovation_shared_mse": ("innovation", "shared", "mse"),
         "innovation_shared_aux": ("innovation", "shared", "query_aux"),
         "innovation_separate_mse": ("innovation", "separate", "mse"),
         "innovation_separate_aux": ("innovation", "separate", "query_aux"),
         "gru_shared_mse": ("innovation_gru", "shared", "mse"),
         "gru_shared_aux": ("innovation_gru", "shared", "query_aux"),
         "gru_separate_mse": ("innovation_gru", "separate", "mse"),
         "gru_separate_aux": ("innovation_gru", "separate", "query_aux")}
SEEDS = (285000001, 285000002, 285000003)
SELECTION_START = 286000001
LIMITS = {"seconds": 21600, "rss_bytes": 4 * 1024**3, "output_bytes": 2 * 1024**3}
CONFIG = {"families": list(KINDS), "fit_seeds": list(SEEDS), "epochs": 80,
          "batch_episodes": 6, "chunk": 32, "learning_rate": .003, "gradient_clip": 5.,
          "weight_decay": 0., "scale": 64., "training_episodes": 54,
          "validation_episodes": 36, "required_conditions": 39,
          "checkpoint": "last", "device": "cpu", "dtype": "float32",
          "selection_seed_start": SELECTION_START,
          "training_weight": "(W/k)/(54*full_episode_nonquery_rows)",
          "batch_scale": "54/actual_batch_episodes", "tbptt": "detach every32; accumulate then one Adam step",
          "loss": "eligible-centered nonquery MSE plus objective-specific all-four prequery MSE; each /64",
          "cells": {k: {"architecture": v[0], "readout": v[1], "objective": v[2]} for k, v in CELLS.items()},
          "prior_coefficient": 1., "prior_weight": "1/(54*full_episode_later_query_rows)",
          "compute_matching": "same episode exposure and updates; auxiliary backprop costs separately counted"}
COMPONENTS = {SELF, TEST, MODELS, SEPARATE_MODELS, DATA, LOSS, METRICS, AUDIT, AUDIT_TEST,
    "tests/test_otto_prequery_scores.py", "tests/test_otto_separate_prior_scores.py", "tests/test_otto_prequery_data.py",
    "tests/test_otto_prequery_loss.py", "tests/test_otto_separate_prior_metrics.py"}
NEW = COMPONENTS | {PROTOCOL, WINDOWS, SAMPLING, CAPACITY, CLOCK, SUPERVISOR,
                   "scripts/train_otto_cross_query_forecasts.py", "scripts/train_otto_prequery_calibration.py",
                   "scripts/qualify_otto_prequery_capacity.py",
                   "tests/test_qualify_otto_separate_prior_capacity.py", BASE_MODELS,
                   "src/openjev/research/otto_cross_query_data.py",
                   "src/openjev/research/otto_cross_query_metrics.py",
                   "src/openjev/research/otto_prequery_metrics.py",
                   "scripts/audit_otto_score_forecasts.py",
                   "src/openjev/__init__.py", "src/openjev/research/__init__.py"}
ROLES = ("collection_plan", "collection_receipt", "collection_terminal", "engineering",
         "capacity_plan", "capacity_receipt", "capacity_terminal")
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")
ARRAYS = {"features", "raw_q", "legal", "actions", "correction", "episode_offsets"}
COLLECTION_PAYLOADS = {f"{name}.jsonl.gz" for name in
    ("work", "weights", "forwards", "transitions", "samples", "annotations")} | {
    "started.json", "runtime.json", "setup.json", "deployment.json", "cohort.json",
    "episode-boundaries.jsonl", "episodes.jsonl", "train.npz", "valid.npz", "costs.json",
    "summary.json", "train-selection.json"}
PAYLOADS = {"started.json", "runtime.json", "progress.jsonl", "work.jsonl", "fits.json", "summary.json",
    "training-history.npz", "training-history.json", "validation-history.npz", "validation-history.json",
    "validation-windows.npz", "validation-windows.json", "prediction-hold.npz"} | {
    f"{kind}-{seed}.npz" for kind in KINDS for seed in SEEDS} | {
    f"prediction-{kind}-{seed}.npz" for kind in KINDS for seed in SEEDS}
PAYLOADS |= {f"training-prediction-{kind}-{seed}.npz" for kind in KINDS for seed in SEEDS}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def model_provider(kind, shared, separate):
    """Explicit frozen readout dispatch; no model inputs determine the route."""
    require(kind in CELLS, "declared logical cell")
    return shared if CELLS[kind][1] == "shared" else separate


def check_paired_initialization(rows):
    """Two objectives pair exactly; extra prior tensors do not alter old tensors."""
    require(len(rows) == 8 and {r["family"] for r in rows} == set(KINDS)
            and len({r["seed"] for r in rows}) == 1, "complete single-seed initialization panel")
    for architecture in ("innovation", "innovation_gru"):
        members = [r for r in rows if r["architecture"] == architecture]
        require(len(members) == 4, "four cells per architecture")
        for readout in ("shared", "separate"):
            paired = [r for r in members if r["readout"] == readout]
            require(len(paired) == 2 and {r["objective"] for r in paired} == {"mse", "query_aux"}
                    and paired[0]["initial_tensors"] == paired[1]["initial_tensors"]
                    and paired[0]["initial_sha256"] == paired[1]["initial_sha256"],
                    "same architecture/readout has exact paired initialization across objectives")
        tensors = [{k: v for k, v in r["initial_tensors"].items() if not k.startswith("prior_output.")}
                   for r in members]
        require(all(value == tensors[0] for value in tensors)
                and len({r["initial_core_sha256"] for r in members}) == 1,
                "shared and separate existing tensors start identically")
        for row in members:
            extra = {k for k in row["initial_tensors"] if k.startswith("prior_output.")}
            require(extra == ({"prior_output.weight", "prior_output.bias"}
                             if row["readout"] == "separate" else set()), "exact added prior readout")


def regular(value):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    require(p.is_file() and p.is_relative_to(ROOT) and ".." not in p.parts
            and not any(x.is_symlink() for x in (p, *p.parents)), "regular contained evidence")
    return p


def descriptor(value):
    p = regular(value)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": p.stat().st_size}


def read(value):
    return json.loads(regular(value).read_text())


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime_record():
    return {"python": sys.version, "executable": sys.executable,
            "distributions": dict(sorted((x.metadata["Name"], x.version)
                                          for x in importlib.metadata.distributions()))}


def cohort():
    rows, global_case = [], 0
    arms = ("analytic", "neural", "period4_hold")
    for stage, starts, count in (("train", (281000001, 282000001), 9),
                                 ("valid", (283000001, 284000001), 6)):
        for regime, first in zip(("lambda3", "lambda4"), starts, strict=True):
            for case in range(count):
                rotation = global_case % 3
                for arm in arms[rotation:] + arms[:rotation]:
                    rows.append({"stage": stage, "episode_index": len(rows), "regime": regime,
                        "seed": first + case, "case": case, "initial_hit": 1 + case % 3,
                        "arm": arm, "episode_id": f"{stage}:{regime}:{first + case}:{arm}"})
                global_case += 1
    return rows


def closed_files(path, receipt, expected=None):
    directory = regular(path).parent
    require(expected is None or set(receipt["files"]) == expected, "exact phase payload names")
    require({p.name for p in directory.iterdir()} == set(receipt["files"]) | {"receipt.json"}, "closed phase inventory")
    for name, pin in receipt["files"].items():
        require(Path(name).name == name and descriptor(directory / name) == pin, "closed payload hash")
    return directory


def successful_process(inputs, prefix, receipt, script, interpreter, seconds):
    """Bind the original successful parent, its launch, command and paid bounds."""
    parent = read(inputs[prefix + "_terminal"]["path"])
    require(parent["status"] == "completed" and parent["returncode"] == 0 and not parent["timed_out"]
        and parent["group_absent"] and parent["cleanup"]["reaped"] and parent["cleanup"]["errors"] == []
        and parent["error"] is parent["clock_error"] is None and parent["cap_seconds"] == seconds
        and parent["clock_source_sha256"] == CLOCK_PIN and parent["watchdog_sha256"] == SUPERVISOR_PIN
        and parent["deadline_ns"] == parent["started_ns"] + seconds * 10**9
        and parent["started_ns"] <= receipt["started_ns"] < receipt["finished_ns"]
        <= parent["finished_ns"] <= parent["deadline_ns"], "original successful " + prefix + " parent")
    command = list(parent["command"])
    if command[1:2] == ["-u"]:
        command.pop(1)
    require(command[:3] == [str(ROOT / interpreter), str(ROOT / script), "run"]
            and len(command) == 11 and len(set(command[3::2])) == 4, "original phase command")
    options = dict(zip(command[3::2], command[4::2], strict=True))
    require(set(options) == {"--plan", "--plan-sha256", "--supervision", "--output"}
        and options["--plan"] == inputs[prefix + "_plan"]["path"]
        and options["--plan-sha256"] == inputs[prefix + "_plan"]["sha256"]
        and options["--output"] == str(regular(inputs[prefix + "_receipt"]["path"]).parent),
        "original phase argument joins")
    launch = read(options["--supervision"])
    require(descriptor(options["--supervision"])["sha256"] == receipt["supervision_sha256"]
            and all(parent[k] == v for k, v in launch.items()), "original launch identity")
    return parent


def authenticate_inputs(inputs):
    require(set(inputs) == set(ROLES), "exact training input roles")
    for item in inputs.values():
        require(descriptor(item["path"]) == {k: item[k] for k in ("sha256", "bytes")}, "input pin before decode")
    collection = read(inputs["collection_plan"]["path"])
    receipt = read(inputs["collection_receipt"]["path"])
    require(collection["version"] == receipt["version"] == COLLECTION_VERSION
        and collection["status"] == "frozen_before_collection" and collection["cohort"] == cohort()
        and receipt["status"] == "completed" and receipt["complete"] is True
        and receipt["plan_sha256"] == inputs["collection_plan"]["sha256"]
        and receipt["sources"] == collection["sources"] and receipt["inputs"] == collection["inputs"]
        and receipt["native_inputs"] == collection["native_inputs"]
        and receipt["limits"] == collection["limits"] == {
            "native_seconds": 7200, "rss_bytes": 4 * 1024**3, "output_bytes": 2 * 1024**3},
        "complete fixed fresh collection")
    require(receipt["completed_episodes"] == 90 and receipt["train_episodes"] == 54
        and receipt["valid_episodes"] == 36 and receipt["training_updates"] == 0
        and not receipt.get("pending") and receipt.get("pending_episode") is None
        and receipt.get("pending_action") is None and receipt.get("pending_emission") is None
        and not receipt.get("cleanup_errors") and receipt["peak_rss_bytes"] <= 4 * 1024**3,
        "all complete collection paths")
    successful_process(inputs, "collection", receipt, COLLECTOR, ".venv-otto-released-native/bin/python", 7200)
    directory = closed_files(inputs["collection_receipt"]["path"], receipt, COLLECTION_PAYLOADS)
    require(sum(d["bytes"] for d in receipt["files"].values()) <= 2 * 1024**3, "collection output bound")
    for section in ("inputs", "native_inputs"):
        for item in collection[section].values():
            require(descriptor(item["path"]) == {k: item[k] for k in ("sha256", "bytes")}, "inherited collection input")
    capacity = read(inputs["capacity_plan"]["path"])
    measured = read(inputs["capacity_receipt"]["path"])
    require(capacity["version"] == measured["version"] == "otto-separate-prior-capacity-v1"
        and capacity["status"] == "frozen_before_synthetic_work"
        and measured["status"] == "completed" and measured["complete"] is True and measured["admitted"] is True
        and measured["plan_sha256"] == inputs["capacity_plan"]["sha256"]
        and measured["sources"] == capacity["sources"] and measured["pending"] is measured["pending_emission"] is None
        and measured["completed_families"] == list(KINDS) and measured["optimizer_updates"] == 8
        and measured["peak_rss_bytes"] <= 4 * 1024**3
        and capacity["sources"][CAPACITY] == descriptor(CAPACITY)["sha256"]
        and capacity["sources"][SELF] == descriptor(SELF)["sha256"]
        and capacity["limits"] == measured["limits"] == {
            "seconds": 180, "rss_bytes": 4 * 1024**3, "output_bytes": 128 * 1024**2},
        "admitted qualified capacity")
    successful_process(inputs, "capacity", measured, CAPACITY, ".venv/bin/python", 180)
    capdir = closed_files(inputs["capacity_receipt"]["path"], measured,
        {"started.json", "runtime.json", "synthetic.json", "work.jsonl", "summary.json"})
    capsummary = read(capdir / "summary.json")
    seconds = math.fsum(r["batch_seconds"] for r in capsummary["families"])
    require(capsummary["admitted"] is True and capsummary["technical_complete"] is True
        and [r["family"] for r in capsummary["families"]] == list(KINDS)
        and all(math.isfinite(r["batch_seconds"]) and r["batch_seconds"] > 0 for r in capsummary["families"])
        and capsummary["sum_batch_seconds"] == seconds
        and capsummary["projected_seconds"] == 1.5 * 3 * 720 * seconds + 180
        and capsummary["projected_seconds"] <= capsummary["threshold_seconds"] == 16200.
        and capacity["configuration"]["training_cap_seconds"] == LIMITS["seconds"]
        and capacity["configuration"]["admission_seconds"] == 16200., "fixed capacity gate")
    engineering = read(inputs["engineering"]["path"])
    require(engineering["status"] == "passed" and engineering["sources_before"] == engineering["sources_after"]
        and COMPONENTS <= set(engineering["sources_after"]) and engineering["results"]
        and all(type(r["exit_code"]) is int and r["exit_code"] == 0 for r in engineering["results"]),
        "all new components qualified unchanged")
    closed_files(inputs["engineering"]["path"], engineering)
    sources = {}
    for mapping in (collection["sources"], capacity["sources"], engineering["sources_after"]):
        for name, pin in mapping.items():
            require(descriptor(name)["sha256"] == pin and (name not in sources or sources[name] == pin),
                    "unchanged source and consistent inherited pin")
            sources[name] = pin
    for name in NEW:
        pin = descriptor(name)["sha256"]
        require(name not in sources or sources[name] == pin, "new source cannot replace old pin")
        sources[name] = pin
    require(sources[CLOCK] == CLOCK_PIN and sources[SUPERVISOR] == SUPERVISOR_PIN
            and sources[BASE_MODELS] == BASE_MODELS_PIN, "fixed process guards and model arithmetic")
    return collection, receipt, directory, sources


def freeze(args):
    inputs = {}
    for role in ROLES:
        path = regular(getattr(args, role)); item = descriptor(path)
        require(item["sha256"] == getattr(args, role + "_sha256"), "external planning pin")
        inputs[role] = {"path": str(path), **item}
    _, _, _, sources = authenticate_inputs(inputs)
    write(args.output, {"version": VERSION, "status": "frozen_before_fitting", "config": CONFIG,
        "limits": LIMITS, "inputs": inputs, "sources": sources, "runtime": runtime_record()})
    print(json.dumps({"status": "frozen_before_fitting", "plan": descriptor(args.output)}), flush=True)


def centered_rows(torch, prediction, targets, legal):
    """Exact capacity expression; only empty padding uses the neutral divisor1."""
    allowed = legal.to(torch.float32)
    count = allowed.sum(dim=-1, keepdim=True).clamp(min=1)
    pred = prediction / 64; target = targets / 64
    difference = (pred - (pred * allowed).sum(dim=-1, keepdim=True) / count
                  - target + (target * allowed).sum(dim=-1, keepdim=True) / count)
    return (difference.square() * allowed).sum(dim=-1) / count[:, :, 0]


def rescore_losses(torch, losses, np, saved, history, objective):
    """Final whole-cohort component sums; no minibatch54/B multiplier."""
    require(objective in ("mse", "query_aux"), "declared rescore objective")
    tensor = lambda name: torch.from_numpy(np.array(history[name], copy=True))
    with torch.no_grad():
        nq = losses.nonquery_rows(torch.from_numpy(saved["predictions"])[None],
            tensor("targets")[None], tensor("legal")[None])[0]
        nonquery = float((nq * tensor("weights").to(torch.float32)).sum())
        mask = torch.from_numpy(saved["prior_mask"])
        if bool(mask.any()):
            rows = losses.prior_rows(torch.from_numpy(saved["prior"])[mask, None],
                tensor("prior_targets")[mask, None])[:, 0]
            prior = float((rows * tensor("prior_weights")[mask].to(torch.float32)).sum())
        else:
            prior = 0.
    total = nonquery + (prior if objective == "query_aux" else 0.)
    require(all(math.isfinite(x) for x in (total, nonquery, prior)), "finite final component sums")
    return {"total": total, "nonquery": nonquery, "prior": prior}


def batch_update(torch, models, data_module, losses, model, optimizer, data, indices, *, objective, check, stage):
    """One complete chronological batch, no parameter update between chunks."""
    require(bool(indices) and len(indices) <= 6 and len(set(indices)) == len(indices), "episode batch")
    lengths = [int(data["episode_offsets"][i + 1] - data["episode_offsets"][i]) for i in indices]
    stage("zero_grad", None); optimizer.zero_grad(set_to_none=True)
    carry = model.initial_carry(len(indices))
    chunks = backwards = rows = 0
    require(objective in ("mse", "query_aux"), "declared batch objective")
    loss_total = nonquery_total = prior_total = 0.
    prior_rows = 0
    for start in range(0, max(lengths), 32):
        check(); stage("chunk_forward", start)
        packet = data_module.batch_chunk(data, indices, start)
        inputs = {k: torch.from_numpy(v) for k, v in packet["model_inputs"].items()}
        target = torch.from_numpy(packet["targets"])
        legal = torch.from_numpy(packet["legal"])
        weight = torch.from_numpy(packet["weights"]).to(torch.float32)
        prior_target = torch.from_numpy(packet["prior_targets"])
        prior_weight = torch.from_numpy(packet["prior_weights"])
        prior_mask = torch.from_numpy(packet["prior_mask"])
        nonzero = bool((weight > 0).any()) or (objective == "query_aux" and bool(prior_mask.any()))
        with torch.set_grad_enabled(nonzero):
            forecast = model(**inputs, carry=carry)
            require(torch.equal(forecast.prior_mask, prior_mask), "model and projected later-query masks agree")
            if nonzero:
                terms = losses.weighted_loss(forecast.prediction, forecast.prior, target, legal, weight,
                    prior_target, prior_weight, prior_mask, objective=objective)
                loss = terms["total"]
                require(bool(torch.isfinite(loss)), "finite batch loss")
                loss_total += float(loss.detach())
                nonquery_total += float(terms["nonquery"].detach())
                prior_total += float(terms["prior"].detach())
                stage("chunk_backward", start); loss.backward(); backwards += 1
        carry = models.detach_carry(forecast.carry)
        prior_rows += int(prior_mask.sum())
        chunks += 1; rows += int(inputs["lengths"].sum())
    require(bool(carry.ended.all()) and carry.absolute_step.tolist() == lengths, "every actual tail consumed")
    stage("complete_gradients", None)
    for parameter in model.parameters():
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        require(bool(torch.isfinite(parameter.grad).all()), "finite complete gradient set")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    stage("optimizer_update", None); optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "finite updated parameters")
    steps = [float(optimizer.state[p]["step"]) for p in model.parameters()]
    require(len(set(steps)) == 1 and steps[0].is_integer(), "one Adam step for every parameter")
    return {"loss": loss_total, "nonquery_loss": nonquery_total, "prior_loss": prior_total,
        "objective": objective, "prior_rows": prior_rows, "gradient_norm_before_clip": float(norm),
        "forward_chunks": chunks, "backward_chunks": backwards, "no_grad_chunks": chunks - backwards,
        "forward_rows": rows, "episode_indices": list(indices), "optimizer_step": int(steps[0]),
        "all_parameter_gradients_finite": True}


def validation_history(np, data_module, episodes):
    """Called only after census validation; nonquery scores stay out of inputs."""
    lengths = [len(e["features"]) for e in episodes]
    offsets = np.concatenate((np.zeros(1, np.int64), np.cumsum(lengths, dtype=np.int64)))
    raw = np.concatenate([e["teacher_scores"] for e in episodes])
    mask = np.concatenate([np.arange(n) % 4 == 0 for n in lengths])
    query = np.zeros_like(raw); query[mask] = raw[mask]
    prior_mask = np.concatenate([(np.arange(n) % 4 == 0) & (np.arange(n) >= 4) for n in lengths])
    prior_targets = np.zeros_like(raw); prior_targets[prior_mask] = raw[prior_mask]
    arrays = {"features": np.concatenate([e["features"] for e in episodes]), "query_scores": query,
        "targets": raw, "legal": np.concatenate([e["legal"] for e in episodes]),
        "actions": np.concatenate([e["actions"] for e in episodes]), "query_mask": mask,
        "weights": np.zeros(len(raw), np.float64), "episode_offsets": offsets,
        "prior_targets": prior_targets, "prior_weights": np.zeros(len(raw), np.float64), "prior_mask": prior_mask}
    return {"version": data_module.VERSION, **arrays, "episode_ids": tuple(e["id"] for e in episodes),
        "episode_regimes": tuple(e["regime"] for e in episodes),
        "episode_splits": tuple("valid" for _ in episodes),
        "counts": {"episodes": len(episodes), "rows": len(raw), "query_rows": int(mask.sum()),
                   "prior_rows": int(prior_mask.sum())}}


def window_predictions(np, windows, history, flat):
    output = np.zeros_like(windows["targets"])
    for row, (episode, start, length) in enumerate(zip(windows["episode_index"], windows["step_offsets"],
                                                      windows["lengths"], strict=True)):
        low = int(history["episode_offsets"][int(episode)]) + int(start)
        output[row, :int(length)] = flat[low:low + int(length)]
    return output


def metric_report(metrics, data, np, windows, episodes, identities, predictions):
    report = metrics.forecast_metrics(windows, predictions, identities)
    report["by_collector"] = {}
    for arm in ("analytic", "neural", "period4_hold"):
        selected = [i for i, row in enumerate(identities) if row["arm"] == arm]
        require(bool(selected), "all declared collectors")
        subset = data.build_windows([episodes[i] for i in selected])
        mask = np.isin(windows["episode_index"], selected)
        report["by_collector"][arm] = metrics.forecast_metrics(subset, predictions[mask], [identities[i] for i in selected])
    return report


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.start = self.clock = self.launch = None
        self.valid_allowed = False
        self.sequence = 0
        self.receipt = {"version": VERSION, "status": "started", "teacher_calls": 0, "native_calls": 0,
            "fits_completed": 0, "optimizer_steps": 0, "pending": None, "pending_emission": None,
            "training_forward_chunks": 0, "training_backward_chunks": 0, "training_forward_rows": 0,
            "training_rescore_chunks": 0, "training_rescore_rows": 0,
            "validation_forward_chunks": 0, "validation_forward_rows": 0}

    def check(self):
        require(self.clock.now_ns() < self.launch["deadline_ns"], "original training deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"] and sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
                < LIMITS["output_bytes"] - 1024**2, "training RSS/output cap with failure reserve")

    def event(self, record, filename="progress.jsonl"):
        self.check()
        self.receipt["pending_emission"] = {"file": filename, "record": record}
        with (self.out / filename).open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        self.receipt["pending_emission"] = None

    def publish(self, name, value):
        self.check(); self.receipt["pending_emission"] = {"file": name}
        write(self.out / name, value)
        self.receipt["pending_emission"] = None

    def npz(self, name, arrays):
        self.check(); self.receipt["pending_emission"] = {"file": name}
        with (self.out / name).open("xb") as stream:
            self.np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
        self.receipt["pending_emission"] = None

    def save_data(self, stem, data):
        self.npz(stem + ".npz", {k: v for k, v in data.items() if isinstance(v, self.np.ndarray)})
        self.publish(stem + ".json", {k: v for k, v in data.items() if not isinstance(v, self.np.ndarray)})

    def admit(self):
        require(descriptor(CLOCK)["sha256"] == CLOCK_PIN, "clock before import")
        self.clock = load(ROOT / CLOCK, "_separate_prior_training_clock").SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, "original launch available")
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and self.launch["pid"] == os.getpid()
            and self.launch["pgid"] == os.getpgrp() and self.launch["parent_pid"] == os.getppid()
            and self.launch["cap_seconds"] == LIMITS["seconds"]
            and self.launch["watchdog_sha256"] == SUPERVISOR_PIN and self.launch["clock_source_sha256"] == CLOCK_PIN
            and self.launch["deadline_ns"] == self.launch["started_ns"] + LIMITS["seconds"] * 10**9,
            "original bounded training process")
        require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256, "external training plan")
        self.plan = read(self.args.plan)
        require(self.plan["version"] == VERSION and self.plan["status"] == "frozen_before_fitting"
            and self.plan["config"] == CONFIG and self.plan["limits"] == LIMITS
            and self.plan["runtime"] == runtime_record(), "fixed protocol and runtime")
        self.collection_plan, self.collection_receipt, self.collection_dir, sources = authenticate_inputs(self.plan["inputs"])
        require(self.plan["sources"] == sources, "full source closure")
        for name in THREADS:
            require(os.environ.get(name) == "1", "one numerical CPU thread")
        self.receipt.update(plan_sha256=self.args.plan_sha256, sources=sources, inputs=self.plan["inputs"],
            supervision_sha256=descriptor(self.args.supervision)["sha256"], limits=LIMITS)
        self.publish("started.json", {"launch": self.launch, "started_ns": self.start})

    def training(self):
        np = self.np
        with np.load(self.collection_dir / "train.npz", allow_pickle=False) as archive:
            require(len(archive.files) == 7 and set(archive.files) == ARRAYS | {"label_mask"}, "seven TRAIN arrays")
            flat = {key: archive[key] for key in archive.files}
        identities = [r for r in self.collection_plan["cohort"] if r["stage"] == "train"]
        history = self.data.project_training(flat, identities, read(self.collection_dir / "train-selection.json"),
                                             selection_start=SELECTION_START)
        self.save_data("training-history", history)
        return history

    def validation(self):
        require(self.valid_allowed and self.receipt["fits_completed"] == 24, "all final checkpoints before VALID decode")
        np = self.np
        with np.load(self.collection_dir / "valid.npz", allow_pickle=False) as archive:
            require(len(archive.files) == len(ARRAYS) and set(archive.files) == ARRAYS, "six VALID arrays")
            flat = {key: archive[key] for key in archive.files}
        identities = [r for r in self.collection_plan["cohort"] if r["stage"] == "valid"]
        offsets = flat["episode_offsets"]
        require(offsets.dtype == np.int64 and offsets.shape == (37,) and offsets[0] == 0
            and bool(((np.diff(offsets) >= 1) & (np.diff(offsets) <= 2188)).all()), "complete VALID offsets")
        total = int(offsets[-1])
        for name, dtype, shape in (("features", np.float32, (total, 31)), ("raw_q", np.float32, (total, 4)),
            ("legal", np.bool_, (total, 4)), ("actions", np.int64, (total,)), ("correction", np.bool_, (total,))):
            require(flat[name].dtype == dtype and flat[name].shape == shape, "VALID array " + name)
        episodes = []
        for index, identity in enumerate(identities):
            low, high = int(offsets[index]), int(offsets[index + 1]); action = flat["actions"][low:high]
            require(bool(((action >= 0) & (action < 4)).all())
                and bool(flat["legal"][low:high][np.arange(high - low), action].all())
                and np.array_equal(flat["correction"][low:high], np.arange(high - low) % 4 == 0), "VALID public actions/schedule")
            episodes.append({"id": identity["episode_id"], "regime": identity["regime"], "split": "valid",
                "features": flat["features"][low:high], "teacher_scores": flat["raw_q"][low:high],
                "legal": flat["legal"][low:high], "actions": action})
        # The qualified census helper rejects extra schema keys, so pass its exact projection.
        census = [{k: v for k, v in e.items() if k != "actions"} for e in episodes]
        windows = self.windows.build_windows(census)
        history = validation_history(np, self.data, episodes)
        self.save_data("validation-windows", windows); self.save_data("validation-history", history)
        return history, windows, census, identities

    def batch(self, model, optimizer, data, indices, kind, seed, epoch, batch):
        self.check(); self.sequence += 1
        identity = {"call_id": self.sequence, "family": kind, "seed": seed, "epoch": epoch,
                    "batch": batch, "episode_indices": indices}
        self.receipt["pending"] = dict(identity)
        self.event({"event": "attempt", **identity}, "work.jsonl")
        tick = self.clock.now_ns()

        def stage(name, chunk):
            self.receipt["pending"] = {**identity, "stage": name, "chunk_start": chunk}

        result = batch_update(self.torch, self.models, self.data, self.losses, model, optimizer, data, indices,
                              objective=CELLS[kind][2], check=self.check, stage=stage)
        self.event({"event": "return", **identity, "result": result,
                    "seconds": (self.clock.now_ns() - tick) / 1e9}, "work.jsonl")
        self.receipt["pending"] = None
        self.receipt["optimizer_steps"] += 1
        for key in ("forward_chunks", "backward_chunks", "forward_rows"):
            self.receipt["training_" + key] += result[key]
        return result

    def predict(self, model, data, phase):
        np, torch = self.np, self.torch
        output = np.zeros_like(data["query_scores"])
        priors = np.zeros_like(output)
        prior_mask = np.zeros(len(output), dtype=np.bool_)
        chunks = rows = 0
        episodes = len(data["episode_ids"])
        with torch.no_grad():
            for first in range(0, episodes, 6):
                indices = list(range(first, min(first + 6, episodes)))
                lengths = [int(data["episode_offsets"][i + 1] - data["episode_offsets"][i]) for i in indices]
                carry = model.initial_carry(len(indices))
                for start in range(0, max(lengths), 32):
                    self.check()
                    self.receipt["pending"] = {"phase": phase, "episode_indices": indices, "chunk_start": start}
                    packet = self.data.batch_chunk(data, indices, start)
                    inputs = {k: torch.from_numpy(v) for k, v in packet["model_inputs"].items()}
                    forecast = model(**inputs, carry=carry)
                    require(torch.equal(forecast.prior_mask, torch.from_numpy(packet["prior_mask"])),
                            "prediction mask equals complete public chronology")
                    carry = self.models.detach_carry(forecast.carry)
                    for lane, index in enumerate(indices):
                        length = int(inputs["lengths"][lane])
                        if length:
                            low = int(data["episode_offsets"][index]) + start
                            output[low:low + length] = forecast.prediction[lane, :length].numpy()
                            priors[low:low + length] = forecast.prior[lane, :length].numpy()
                            prior_mask[low:low + length] = forecast.prior_mask[lane, :length].numpy()
                            rows += length
                    chunks += 1
                require(bool(carry.ended.all()) and carry.absolute_step.tolist() == lengths, "prediction consumes all true tails")
        self.receipt["pending"] = None
        self.receipt[phase + "_chunks"] += chunks; self.receipt[phase + "_rows"] += rows
        self.metrics.validate_prediction_support(data, priors, prior_mask)
        return {"predictions": output, "prior": priors, "prior_mask": prior_mask}, {
            "forward_chunks": chunks, "forward_rows": rows, "prior_rows": int(prior_mask.sum())}

    def checkpoint(self, name, model):
        arrays = {key: value.detach().numpy().copy() for key, value in model.state_dict().items()}
        self.npz(name, arrays)
        with self.np.load(self.out / name, allow_pickle=False) as saved:
            require(len(saved.files) == len(arrays) and set(saved.files) == set(arrays), "checkpoint key equality")
            for key, value in arrays.items():
                restored = saved[key]
                require(restored.dtype == value.dtype and restored.shape == value.shape
                    and restored.tobytes() == value.tobytes(), "checkpoint exact published bytes")
        return descriptor(self.out / name)

    def fit(self, kind, seed, history):
        np, torch = self.np, self.torch
        tick = self.clock.now_ns()
        architecture, readout, objective = CELLS[kind]
        provider = model_provider(kind, self.models, self.separate_models)
        model = provider.make_head(architecture, seed)
        initial = {k: hashlib.sha256(v.detach().numpy().tobytes()).hexdigest() for k, v in model.state_dict().items()}
        full_hash, core_hash = hashlib.sha256(), hashlib.sha256()
        for key, value in model.state_dict().items():
            encoded = key.encode() + value.detach().numpy().tobytes()
            full_hash.update(encoded)
            if not key.startswith(("correction.", "prior_output.")):
                core_hash.update(encoded)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=0.)
        rng = np.random.default_rng(seed)
        count = len(history["episode_ids"])
        require(count == 54, "fixed full TRAIN episode count")
        orders, steps, exposure, forward_chunks, backward_chunks, forward_rows = [], 0, 0, 0, 0, 0
        permutation = hashlib.sha256()
        for epoch in range(80):
            order = rng.permutation(count).astype(np.int64)
            orders.append(order.tolist()); permutation.update(order.tobytes())
            for start in range(0, count, 6):
                indices = order[start:start + 6].tolist()
                result = self.batch(model, optimizer, history, indices, kind, seed, epoch, start // 6)
                steps += 1; exposure += len(indices)
                require(result["optimizer_step"] == steps, "chronological fixed update count")
                forward_chunks += result["forward_chunks"]; backward_chunks += result["backward_chunks"]
                forward_rows += result["forward_rows"]
            if (epoch + 1) % 20 == 0:
                self.event({"event": "epoch", "family": kind, "seed": seed, "epoch": epoch + 1,
                            "optimizer_steps": steps, "episode_exposures": exposure})
        fit_only = (self.clock.now_ns() - tick) / 1e9
        rescore_tick = self.clock.now_ns()
        saved, rescore = self.predict(model, history, "training_rescore")
        terms = rescore_losses(torch, self.losses, np, saved, history, objective)
        final_loss, nonquery_loss, prior_loss = (terms[key] for key in ("total", "nonquery", "prior"))
        require(math.isfinite(final_loss), "finite final full-history selected TRAIN rescore")
        self.npz(f"training-prediction-{kind}-{seed}.npz", saved)
        rescore_seconds = (self.clock.now_ns() - rescore_tick) / 1e9
        name = f"{kind}-{seed}.npz"
        checkpoint_tick = self.clock.now_ns(); pin = self.checkpoint(name, model)
        checkpoint_seconds = (self.clock.now_ns() - checkpoint_tick) / 1e9
        fit = {"family": kind, "architecture": architecture, "readout": readout, "objective": objective,
            "seed": seed, "parameter_count": provider.parameter_count(architecture),
            "checkpoint_path": name, "checkpoint": pin, "initial_tensors": initial,
            "initial_sha256": full_hash.hexdigest(), "initial_core_sha256": core_hash.hexdigest(),
            "epochs": 80, "steps": steps, "episode_exposures": exposure, "episodes_per_epoch": count,
            "episode_orders": orders, "permutation_sha256": permutation.hexdigest(),
            "forward_chunks": forward_chunks, "backward_chunks": backward_chunks,
            "no_grad_chunks": forward_chunks - backward_chunks, "forward_rows": forward_rows,
            "final_train_loss": final_loss, "final_nonquery_loss": nonquery_loss,
            "final_prior_loss": prior_loss,
            "final_prior_loss_scope": "post-fit diagnostic for every cell; not trained by mse controls",
            "final_objective_prior_loss": prior_loss if objective == "query_aux" else 0.,
            "train_rescore": rescore,
            "fit_seconds": fit_only, "train_rescore_seconds": rescore_seconds,
            "checkpoint_seconds": checkpoint_seconds, "wall_seconds": (self.clock.now_ns() - tick) / 1e9}
        require(steps == 720 and exposure == 4320 and forward_rows == 80 * history["counts"]["rows"], "complete fit exposure")
        self.event({"event": "fit_complete", **fit})
        self.receipt["fits_completed"] += 1
        return model, fit

    def body(self):
        tick = self.clock.now_ns()
        import numpy as np
        import torch

        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
        self.np, self.torch = np, torch
        sys.path.insert(0, str(ROOT / "src"))
        self.models = load(ROOT / MODELS, "_separate_prior_training_models")
        self.separate_models = load(ROOT / SEPARATE_MODELS, "_separate_prior_training_separate_models")
        self.data = load(ROOT / DATA, "_separate_prior_training_data")
        self.losses = load(ROOT / LOSS, "_separate_prior_training_loss")
        self.windows = load(ROOT / WINDOWS, "_separate_prior_census_windows")
        self.metrics = load(ROOT / METRICS, "_separate_prior_metrics")
        require(self.models.KINDS == self.separate_models.KINDS == ("innovation", "innovation_gru")
                and self.metrics.FAMILIES == KINDS and self.metrics.SEEDS == SEEDS
                and torch.get_num_threads() == torch.get_num_interop_threads() == 1, "fixed CPU models and metrics")
        self.publish("runtime.json", {**runtime_record(), "torch": torch.__version__, "numpy": np.__version__,
            "threads": 1, "interop_threads": 1, "deterministic": True, "cuda_used": False, "mps_used": False})
        train = self.training()
        setup_seconds = (self.clock.now_ns() - tick) / 1e9
        tick = self.clock.now_ns(); fits, fitted = [], []
        for seed in SEEDS:
            for kind in KINDS:
                self.event({"event": "fit_start", "family": kind, "seed": seed})
                model, fit = self.fit(kind, seed, train)
                fits.append(fit); fitted.append((model, fit))
        fit_seconds = (self.clock.now_ns() - tick) / 1e9
        self.publish("fits.json", {"fits": fits, "train_counts": train["counts"]})
        for seed in SEEDS:
            subset = [r for r in fits if r["seed"] == seed]
            require(len({r["permutation_sha256"] for r in subset}) == 1, "same paired episode orders")
            check_paired_initialization(subset)
        self.event({"event": "all_checkpoints_closed_before_VALID", "fits_completed": 24,
                    "checkpoints": {r["checkpoint_path"]: r["checkpoint"] for r in fits}})
        self.valid_allowed = True
        tick = self.clock.now_ns()
        valid, windows, episodes, identities = self.validation()
        held = np.repeat(windows["query_scores"][:, None, :], 4, axis=1)
        held[~windows["valid_mask"]] = 0.
        self.npz("prediction-hold.npz", {"predictions": held})
        hold = metric_report(self.metrics, self.windows, np, windows, episodes, identities, held)
        records = []
        for model, fit in fitted:
            saved, work = self.predict(model, valid, "validation_forward")
            forecast = window_predictions(np, windows, valid, saved["predictions"])
            self.npz(f"prediction-{fit['family']}-{fit['seed']}.npz", {
                "predictions": forecast, "prior": saved["prior"], "prior_mask": saved["prior_mask"]})
            records.append({"family": fit["family"], "architecture": fit["architecture"],
                "readout": fit["readout"], "objective": fit["objective"], "seed": fit["seed"], "prediction_work": work,
                "metrics": metric_report(self.metrics, self.windows, np, windows, episodes, identities, forecast),
                "prior_metrics": self.metrics.prior_metrics(valid, saved["prior"], identities)})
        conditions = self.metrics.criteria(records, hold, technical_complete=False)
        summary = {"version": VERSION, "models": records, "hold": hold,
            "required": conditions, "required_passed": sum(c["passes"] for c in conditions),
            "required_conditions": 39, "gates": self.metrics.gate_decisions(conditions),
            "interactions": self.metrics.interactions(records),
            "technical_complete_pending_saved_audit": True,
            "scientific_conditions": 38, "scientific_passed": sum(c["passes"] for c in conditions[1:]),
            "requires_successful_original_supervisor_and_saved_audit": True,
            "train_counts": train["counts"], "validation_counts": windows["counts"],
            "validation_history_counts": valid["counts"], "setup_seconds": setup_seconds,
            "fitting_seconds": fit_seconds, "validation_seconds": (self.clock.now_ns() - tick) / 1e9,
            "scope": "Forced-path forecasts only, with separately gated mechanisms and architecture. No omnibus all-39 gate, autonomous performance, new-shift evidence or true action regret."}
        self.publish("summary.json", summary)

    def execute(self):
        self.out.mkdir(exist_ok=False)

        def interrupted(_signum, _frame):
            raise InterruptedError("original cross-query supervisor stopped worker")

        signal.signal(signal.SIGTERM, interrupted)
        try:
            self.admit(); self.body()
            for name, pin in self.plan["sources"].items():
                self.check(); require(descriptor(name)["sha256"] == pin, "unchanged final source")
            for item in self.plan["inputs"].values():
                require(descriptor(item["path"]) == {k: item[k] for k in ("sha256", "bytes")}, "unchanged final input")
            for name, pin in self.collection_receipt["files"].items():
                self.check(); require(descriptor(self.collection_dir / name) == pin, "unchanged collection payload")
            require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256
                and descriptor(self.args.supervision)["sha256"] == self.receipt["supervision_sha256"], "unchanged plan/launch")
            require(self.receipt["fits_completed"] == 24 and self.receipt["optimizer_steps"] == 17280
                and self.receipt["pending"] is self.receipt["pending_emission"] is None, "complete fixed training")
            require({p.name for p in self.out.iterdir()} == PAYLOADS, "exact 85 payload closure")
            files = {name: descriptor(self.out / name) for name in sorted(PAYLOADS)}
            self.check(); finished = self.clock.now_ns()
            self.receipt.update(status="completed", complete=True, files=files, started_ns=self.start,
                finished_ns=finished, wall_seconds=(finished - self.start) / 1e9,
                requires_successful_original_supervisor=True)
            write(self.out / "receipt.json", self.receipt); self.check()
            print(json.dumps({"status": "completed", "receipt": descriptor(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            try:
                self.receipt.update(status="failed", complete=False, error=repr(error), traceback=traceback.format_exc())
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - never replace the original failure
                error.add_note("Failure receipt publication also failed: " + repr(secondary))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    plan = sub.add_parser("plan")
    for role in ROLES:
        plan.add_argument("--" + role.replace("_", "-"), type=Path, required=True)
        plan.add_argument("--" + role.replace("_", "-") + "-sha256", required=True)
    plan.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    for name in ("plan", "supervision", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path(sys.executable).absolute() == ROOT / ".venv/bin/python" and Path.cwd() == ROOT,
            "original general CPU interpreter and checkout")
    require(args.output.is_absolute() and args.output.is_relative_to(ROOT) and ".." not in args.output.parts
            and not any(p.is_symlink() for p in args.output.parents), "contained exclusive absolute output")
    if args.mode == "plan":
        freeze(args)
    else:
        require(args.plan.is_absolute() and args.supervision.is_absolute(), "absolute execution inputs")
        Run(args).execute()


if __name__ == "__main__":
    main()
