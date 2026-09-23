"""Shared pretraining then protected or joint readout adaptation; no new scientific collection.

Metadata admission precedes numerical imports. The complete fifteen-checkpoint
barrier precedes VALID decoding. All model inputs exclude skipped labels; only
selected nonquery rows and all later-query priors supply loss. Parameters stay fixed throughout
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
SELF = "scripts/train_otto_protected_readout.py"
TEST = "tests/test_train_otto_protected_readout.py"
MODELS = "src/openjev/research/otto_protected_training_model.py"
PRETRAIN_MODELS = "src/openjev/research/otto_prequery_scores.py"
PROTECTED_PARENT = "src/openjev/research/otto_protected_readout.py"
DATA = "src/openjev/research/otto_prequery_data.py"
LOSS = "src/openjev/research/otto_action_focused_loss.py"
BASE_LOSS = "src/openjev/research/otto_prequery_loss.py"
SPO_LOSS = "src/openjev/research/otto_spo_plus_loss.py"
METRICS = "src/openjev/research/otto_protected_readout_metrics.py"
WINDOWS = "src/openjev/research/otto_score_forecast_data.py"
SAMPLING = "src/openjev/research/otto_sampled_forecast_data.py"
PROTOCOL = "research/otto-protected-readout-protocol.md"
AUDIT = "scripts/audit_otto_protected_readout.py"
AUDIT_TEST = "tests/test_audit_otto_protected_readout.py"
COLLECTOR = "scripts/collect_otto_protected_readout.py"
CAPACITY = "scripts/qualify_otto_prequery_capacity.py"
CAPACITY_TRAINER = "scripts/train_otto_prequery_calibration.py"
CAPACITY_KINDS = ("innovation_aux", "innovation_mse", "innovation_gru_mse", "innovation_gru_aux")
CAPACITY_SCOPE = "Inherited original backbone evidence only; protected residual, staged fitting and current total runtime were not measured"
BASE_MODELS = "src/openjev/research/otto_cross_query_scores.py"
BASE_MODELS_PIN = "799979c0c60460352df076750d15b2cb9a5db6b6976707605dcca7c90fa76e08"
CLOCK = "src/openjev/research/suspend_clock.py"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SUPERVISOR_PIN = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
VERSION = "otto-protected-readout-training-v1"
COLLECTION_VERSION = "otto-protected-readout-collection-v1"
KINDS = ("pretrained", "frozen_aux", "frozen_spo", "joint_aux", "joint_spo")
BRANCHES = KINDS[1:]
CELLS = {kind: ("innovation_gru", "shared" if kind == "pretrained" else "protected",
                "aux" if kind == "pretrained" else kind.rsplit("_", 1)[1]) for kind in KINDS}
SEEDS = (301000001, 301000002, 301000003)
SELECTION_START = 302000001
EPOCHS = {kind: 80 if kind == "pretrained" else 40 for kind in KINDS}
LIMITS = {"seconds": 14400, "rss_bytes": 4 * 1024**3, "output_bytes": 2 * 1024**3}
CONFIG = {"families": list(KINDS), "fit_seeds": list(SEEDS), "epochs": EPOCHS,
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

COMPONENTS = {SELF, TEST, MODELS, PRETRAIN_MODELS, PROTECTED_PARENT, DATA, LOSS, BASE_LOSS, SPO_LOSS, METRICS, AUDIT, AUDIT_TEST,
    "tests/test_otto_protected_training_model.py", "tests/test_otto_protected_readout.py",
    "tests/test_otto_prequery_scores.py", "tests/test_otto_prequery_data.py",
    "tests/test_otto_prequery_loss.py", "tests/test_otto_spo_plus_loss.py",
    "tests/test_otto_action_focused_loss.py", "tests/test_otto_protected_readout_metrics.py"}
NEW = COMPONENTS | {PROTOCOL, WINDOWS, SAMPLING, CAPACITY, CLOCK, SUPERVISOR,
                   "scripts/train_otto_cross_query_forecasts.py", "scripts/train_otto_prequery_calibration.py",
                   "scripts/train_otto_action_focused.py",
                   BASE_MODELS, "src/openjev/research/otto_cross_query_data.py",
                   "src/openjev/research/otto_cross_query_metrics.py",
                   "src/openjev/research/otto_prequery_metrics.py",
                   "src/openjev/research/otto_separate_prior_metrics.py",
                   "scripts/audit_otto_score_forecasts.py", "scripts/audit_otto_action_focused.py",
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


def evaluation_metadata(kind):
    require(kind in KINDS, "declared evaluation family")
    return {"parameter_mode": "frozen", "grad_enabled": False,
            "backbone_requires_grad": False, "residual_requires_grad": None if kind == "pretrained" else True}


def state_witness(model, *, backbone_only=False):
    states = {k: v for k, v in model.state_dict().items()
              if not backbone_only or not k.startswith("action_residual.")}
    digest, tensors = hashlib.sha256(), {}
    for name, value in states.items():
        raw = value.detach().cpu().numpy().tobytes()
        tensors[name] = hashlib.sha256(raw).hexdigest()
        digest.update(name.encode() + raw)
    return digest.hexdigest(), tensors


def trainable_parameters(model):
    values = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    require(bool(values), "nonempty trainable parameter set")
    return values


def canonical_clone(pretrain_provider, protected_provider, model, kind, seed):
    """Fresh model ownership and exact weights, under matched frozen backbone flags."""
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    clone = (pretrain_provider.make_head("innovation_gru", seed) if kind == "pretrained"
             else protected_provider.make_head("frozen", seed))
    clone.load_state_dict(state, strict=True)
    if kind == "pretrained":
        for p in clone.parameters():
            p.requires_grad_(False)
    clone.eval()
    expected = evaluation_metadata(kind)
    require(all(p.requires_grad == (name.startswith("action_residual.") and kind != "pretrained")
                for name, p in clone.named_parameters()), "canonical frozen backbone and permitted residual locks")
    require(state_witness(clone) == state_witness(model), "canonical clone exact model weights")
    require(all(clone.state_dict()[name].data_ptr() != tensor.data_ptr() for name, tensor in model.state_dict().items()),
            "canonical clone owns independent weights")
    return clone, expected


def check_paired_initialization(rows):
    require(len(rows) == 5 and [r["family"] for r in rows] == list(KINDS)
            and len({r["seed"] for r in rows}) == 1, "complete ordered single-seed pretraining and adaptation panel")
    parent, branches = rows[0], rows[1:]
    require(parent["epochs"] == 80 and parent["steps"] == 720 and parent["parameter_count"] == 5996,
            "complete ordinary GRU pretraining")
    for row in branches:
        require(row["epochs"] == 40 and row["steps"] == 360 and row["parameter_count"] == 6112
                and row["pretrained_checkpoint"] == parent["checkpoint"]
                and row["pretrained_checkpoint_path"] == parent["checkpoint_path"]
                and row["initial_backbone_sha256"] == parent["final_backbone_sha256"]
                and row["initial_residual_zero"] is True
                and row["optimizer_initial_state_entries"] == 0, "exact fresh checkpoint fork and Adam reset")
        require(row["initial_tensors"] == branches[0]["initial_tensors"]
                and row["initial_sha256"] == branches[0]["initial_sha256"]
                and row["permutation_sha256"] == branches[0]["permutation_sha256"],
                "identical four-way initial weights and stage2 episode order")
        if row["family"].startswith("frozen_"):
            require(row["frozen_backbone_unchanged"] is True
                    and row["final_backbone_sha256"] == parent["final_backbone_sha256"]
                    and row["train_base_equals_pretrained"] is True, "frozen fork exact backbone and TRAIN outputs")


def same_frozen_predictions(np, saved, pretrained):
    for key, reference in (("base_predictions", "predictions"), ("prior", "prior"), ("prior_mask", "prior_mask")):
        value, target = saved[key], pretrained[reference]
        require(value.dtype == target.dtype and value.shape == target.shape and value.tobytes() == target.tobytes(),
                "frozen canonical pretrained equality: " + key)
    return True


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
    for stage, starts, count in (("train", (297000001, 298000001), 9),
                                 ("valid", (299000001, 300000001), 6)):
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
    require(capacity["version"] == measured["version"] == "otto-prequery-capacity-v1"
        and capacity["status"] == "frozen_before_synthetic_work"
        and measured["status"] == "completed" and measured["complete"] is True and measured["admitted"] is True
        and measured["plan_sha256"] == inputs["capacity_plan"]["sha256"]
        and measured["sources"] == capacity["sources"] and measured["pending"] is measured["pending_emission"] is None
        and measured["completed_families"] == list(CAPACITY_KINDS) and measured["optimizer_updates"] == 4
        and measured["peak_rss_bytes"] <= 4 * 1024**3
        and capacity["sources"][CAPACITY] == descriptor(CAPACITY)["sha256"]
        and capacity["sources"][CAPACITY_TRAINER] == descriptor(CAPACITY_TRAINER)["sha256"]
        and capacity["limits"] == measured["limits"] == {
            "seconds": 120, "rss_bytes": 4 * 1024**3, "output_bytes": 128 * 1024**2},
        "admitted qualified capacity")
    successful_process(inputs, "capacity", measured, CAPACITY, ".venv/bin/python", 120)
    capdir = closed_files(inputs["capacity_receipt"]["path"], measured,
        {"started.json", "runtime.json", "synthetic.json", "work.jsonl", "summary.json"})
    capsummary = read(capdir / "summary.json")
    seconds = math.fsum(r["batch_seconds"] for r in capsummary["families"])
    require(capsummary["admitted"] is True and capsummary["technical_complete"] is True
        and [r["family"] for r in capsummary["families"]] == list(CAPACITY_KINDS)
        and all(math.isfinite(r["batch_seconds"]) and r["batch_seconds"] > 0 for r in capsummary["families"])
        and capsummary["sum_batch_seconds"] == seconds
        and capsummary["projected_seconds"] == 1.5 * 3 * 720 * seconds + 120
        and capsummary["projected_seconds"] <= capsummary["threshold_seconds"] == 8100.
        and capacity["configuration"]["training_cap_seconds"] == 10800
        and capacity["configuration"]["admission_seconds"] == 8100., "closed historical capacity only, not a new-loss runtime gate")
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
    """Final whole-cohort component sums; SPO is a paid diagnostic even for AUX."""
    require(objective in ("aux", "spo"), "declared rescore objective")
    tensor = lambda name: torch.from_numpy(np.array(history[name], copy=True))
    with torch.no_grad():
        prediction = torch.from_numpy(saved["predictions"])[None]
        targets, legal = tensor("targets")[None], tensor("legal")[None]
        weights = tensor("weights").to(torch.float32)
        nq = losses.nonquery_rows(prediction, targets, legal)[0]
        nonquery = float((nq * weights).sum())
        spo = float((losses.spo_plus_rows(prediction, targets, legal)[0] * weights).sum())
        mask = torch.from_numpy(saved["prior_mask"])
        if bool(mask.any()):
            rows = losses.prior_rows(torch.from_numpy(saved["prior"])[mask, None],
                tensor("prior_targets")[mask, None])[:, 0]
            prior = float((rows * tensor("prior_weights")[mask].to(torch.float32)).sum())
        else:
            prior = 0.
    total = nonquery + prior + (spo if objective == "spo" else 0.)
    require(all(math.isfinite(x) for x in (total, nonquery, prior, spo)), "finite final component sums")
    return {"total": total, "nonquery": nonquery, "prior": prior, "spo": spo}


def batch_update(torch, models, data_module, losses, model, optimizer, data, indices, *, objective, check, stage):
    """A whole episode batch; frozen prior terms are recorded without backward work."""
    require(bool(indices) and len(indices) <= 6 and len(set(indices)) == len(indices), "episode batch")
    require(objective in ("aux", "spo"), "declared batch objective")
    lengths = [int(data["episode_offsets"][i+1] - data["episode_offsets"][i]) for i in indices]
    parameters = trainable_parameters(model)
    expected_parameters = [p for _, p in parameters]
    require([id(p) for group in optimizer.param_groups for p in group["params"]] == [id(p) for p in expected_parameters],
            "optimizer contains exactly the trainable parameters in declared order")
    frozen = getattr(model, "mode", None) == "frozen"
    initial_backbone = state_witness(model, backbone_only=True)[0] if frozen else None
    stage("zero_grad", None); optimizer.zero_grad(set_to_none=True)
    carry = model.initial_carry(len(indices))
    chunks = backwards = rows = loss_chunks = differentiable = skipped_backward = 0
    loss_total = nonquery_total = prior_total = spo_total = 0.
    spo_calls = spo_weighted_rows = prior_rows = 0
    for start in range(0, max(lengths), 32):
        check(); stage("chunk_forward", start)
        packet = data_module.batch_chunk(data, indices, start)
        inputs = {k: torch.from_numpy(v) for k, v in packet["model_inputs"].items()}
        target, legal = (torch.from_numpy(packet[k]) for k in ("targets", "legal"))
        weight = torch.from_numpy(packet["weights"]).to(torch.float32)
        prior_target = torch.from_numpy(packet["prior_targets"])
        prior_weight = torch.from_numpy(packet["prior_weights"])
        prior_mask = torch.from_numpy(packet["prior_mask"])
        nonquery_support, prior_support = bool((weight > 0).any()), bool(prior_mask.any())
        has_loss = nonquery_support or prior_support
        has_differentiable_loss = nonquery_support or (prior_support and not frozen)
        with torch.set_grad_enabled(has_differentiable_loss):
            forecast = model(**inputs, carry=carry)
            require(torch.equal(forecast.prior_mask, prior_mask), "complete later-query masks agree")
            if has_loss:
                prediction = getattr(forecast, "action_prediction", forecast.prediction)
                terms = losses.weighted_loss(prediction, forecast.prior, target, legal, weight,
                    prior_target, prior_weight, prior_mask, objective=objective)
                loss = terms["total"]
                require(bool(torch.isfinite(loss)), "finite batch loss")
                require(loss.requires_grad is has_differentiable_loss, "loss differentiability matches trained support")
                loss_total += float(loss.detach()); nonquery_total += float(terms["nonquery"].detach())
                prior_total += float(terms["prior"].detach()); spo_total += float(terms["spo"].detach())
                loss_chunks += 1
                if objective == "spo":
                    spo_calls += 1; spo_weighted_rows += int((weight > 0).sum())
                if has_differentiable_loss:
                    stage("chunk_backward", start); loss.backward(); backwards += 1; differentiable += 1
                else:
                    skipped_backward += 1
        carry = models.detach_carry(forecast.carry)
        prior_rows += int(prior_mask.sum()); chunks += 1; rows += int(inputs["lengths"].sum())
    require(bool(carry.ended.all()) and carry.absolute_step.tolist() == lengths, "all true batch tails consumed")
    stage("complete_gradients", None)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            require(parameter.grad is None, "locked backbone has no gradient")
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        require(bool(torch.isfinite(parameter.grad).all()), "finite trainable gradient set: " + name)
    norm = torch.nn.utils.clip_grad_norm_(expected_parameters, 5., error_if_nonfinite=True)
    stage("optimizer_update", None); optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "finite updated parameters")
    steps = [float(optimizer.state[p]["step"]) for p in expected_parameters]
    require(len(set(steps)) == 1 and steps[0].is_integer(), "one Adam step for every trainable parameter")
    require(all(p not in optimizer.state for p in model.parameters() if not p.requires_grad), "no frozen Adam state")
    if frozen:
        require(state_witness(model, backbone_only=True)[0] == initial_backbone, "frozen backbone unchanged after batch")
    return {"loss": loss_total, "nonquery_loss": nonquery_total, "prior_loss": prior_total,
        "spo_loss": spo_total, "spo_loss_calls": spo_calls, "spo_weighted_rows": spo_weighted_rows,
        "objective": objective, "prior_rows": prior_rows, "gradient_norm_before_clip": float(norm),
        "forward_chunks": chunks, "loss_chunks": loss_chunks, "differentiable_chunks": differentiable,
        "backward_chunks": backwards, "skipped_backward_chunks": skipped_backward,
        "no_grad_chunks": chunks - differentiable, "forward_rows": rows, "episode_indices": list(indices),
        "optimizer_step": int(steps[0]), "trainable_parameter_count": sum(p.numel() for p in expected_parameters),
        "all_parameter_gradients_finite": True, "frozen_backbone_unchanged": True if frozen else None}


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
            "training_loss_chunks": 0, "training_differentiable_chunks": 0, "training_skipped_backward_chunks": 0,
            "training_spo_loss_calls": 0, "training_spo_weighted_rows": 0,
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
        self.clock = load(ROOT / CLOCK, "_protected_readout_training_clock").SuspendClock()
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
        require(self.valid_allowed and self.receipt["fits_completed"] == 15, "all fifteen final checkpoints before VALID decode")
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
        for key in ("forward_chunks", "backward_chunks", "forward_rows", "spo_loss_calls", "spo_weighted_rows",
                    "loss_chunks", "differentiable_chunks", "skipped_backward_chunks"):
            self.receipt["training_" + key] += result[key]
        return result

    def predict(self, model, data, phase):
        np, torch = self.np, self.torch
        output = np.zeros_like(data["query_scores"])
        base_output = np.zeros_like(output)
        priors = np.zeros_like(output)
        prior_mask = np.zeros(len(output), dtype=np.bool_)
        chunks = rows = 0
        episodes = len(data["episode_ids"])
        require(all(p.requires_grad == name.startswith("action_residual.") for name, p in model.named_parameters()),
                "canonical frozen-backbone evaluation flags")
        with torch.no_grad():
            require(torch.is_grad_enabled() is False, "canonical evaluation gradient context")
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
                            output[low:low + length] = getattr(forecast, "action_prediction", forecast.prediction)[lane, :length].numpy()
                            base_output[low:low + length] = forecast.prediction[lane, :length].numpy()
                            priors[low:low + length] = forecast.prior[lane, :length].numpy()
                            prior_mask[low:low + length] = forecast.prior_mask[lane, :length].numpy()
                            rows += length
                    chunks += 1
                require(bool(carry.ended.all()) and carry.absolute_step.tolist() == lengths, "prediction consumes all true tails")
        self.receipt["pending"] = None
        self.receipt[phase + "_chunks"] += chunks; self.receipt[phase + "_rows"] += rows
        self.metrics.validate_prediction_support(data, priors, prior_mask)
        return {"predictions": output, "base_predictions": base_output, "prior": priors, "prior_mask": prior_mask}, {
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

    def fit(self, kind, seed, history, *, backbone_state=None, pretrained_fit=None, pretrained_saved=None):
        np, torch = self.np, self.torch
        tick = self.clock.now_ns()
        architecture, readout, objective = CELLS[kind]
        if kind == "pretrained":
            require(backbone_state is pretrained_fit is pretrained_saved is None, "ordinary pretraining has no fork input")
            model = self.pretrain_models.make_head("innovation_gru", seed)
            mode = "pretrain"
        else:
            require(pretrained_fit is not None and pretrained_saved is not None and backbone_state is not None
                    and pretrained_fit["family"] == "pretrained" and pretrained_fit["seed"] == seed,
                    "same-seed completed pretraining required before branch construction")
            mode = kind.split("_", 1)[0]
            model = self.models.from_pretrained(mode, seed, backbone_state)
        initial_hash, initial = state_witness(model)
        initial_backbone, _ = state_witness(model, backbone_only=True)
        initial_zero = None if kind == "pretrained" else all(
            bool((p == 0).all()) for name, p in model.named_parameters() if name.startswith("action_residual."))
        if kind != "pretrained":
            require(initial_backbone == pretrained_fit["final_backbone_sha256"] and initial_zero is True,
                    "exact completed pretrain backbone and zero new residual")
        selected = trainable_parameters(model)
        optimizer = torch.optim.Adam([p for _, p in selected], lr=.003, weight_decay=0.)
        require(len(optimizer.state) == 0, "new Adam state for every stage")
        rng = np.random.default_rng(seed)
        count = len(history["episode_ids"])
        require(count == 54, "fixed full TRAIN episode count")
        orders, steps, exposure = [], 0, 0
        totals = {key: 0 for key in ("forward_chunks", "backward_chunks", "no_grad_chunks", "forward_rows",
                  "loss_chunks", "differentiable_chunks", "skipped_backward_chunks", "spo_loss_calls", "spo_weighted_rows")}
        permutation = hashlib.sha256()
        for epoch in range(EPOCHS[kind]):
            order = rng.permutation(count).astype(np.int64)
            orders.append(order.tolist()); permutation.update(order.tobytes())
            for start in range(0, count, 6):
                indices = order[start:start + 6].tolist()
                result = self.batch(model, optimizer, history, indices, kind, seed, epoch, start // 6)
                steps += 1; exposure += len(indices)
                require(result["optimizer_step"] == steps, "chronological fixed stage update count")
                for key in totals:
                    totals[key] += result[key]
            if (epoch + 1) % 20 == 0:
                self.event({"event": "epoch", "family": kind, "seed": seed, "epoch": epoch + 1,
                            "optimizer_steps": steps, "episode_exposures": exposure})
        fit_only = (self.clock.now_ns() - tick) / 1e9
        final_backbone, _ = state_witness(model, backbone_only=True)
        if mode == "frozen":
            require(final_backbone == initial_backbone, "whole frozen branch preserves original checkpoint")
        clone_tick = self.clock.now_ns()
        evaluated, evaluation = canonical_clone(self.pretrain_models, self.models, model, kind, seed)
        clone_seconds = (self.clock.now_ns() - clone_tick) / 1e9
        rescore_tick = self.clock.now_ns()
        saved, rescore = self.predict(evaluated, history, "training_rescore")
        train_equality = same_frozen_predictions(np, saved, pretrained_saved) if mode == "frozen" else None
        terms = rescore_losses(torch, self.losses, np, saved, history, objective)
        self.npz(f"training-prediction-{kind}-{seed}.npz", saved)
        rescore_seconds = (self.clock.now_ns() - rescore_tick) / 1e9
        name = f"{kind}-{seed}.npz"
        checkpoint_tick = self.clock.now_ns(); pin = self.checkpoint(name, model)
        checkpoint_seconds = (self.clock.now_ns() - checkpoint_tick) / 1e9
        require(state_witness(evaluated) == state_witness(model), "evaluation cannot mutate final training weights")
        fit = {"family": kind, "architecture": architecture, "readout": readout, "objective": objective,
            "training_mode": mode, "stage": "pretrain" if kind == "pretrained" else "adaptation",
            "seed": seed, "parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count": sum(p.numel() for _, p in selected),
            "optimizer_parameter_names": [name for name, _ in selected], "optimizer_initial_state_entries": 0,
            "residual_output_scale": None if kind == "pretrained" else 64.,
            "checkpoint_path": name, "checkpoint": pin, "initial_tensors": initial,
            "initial_sha256": initial_hash, "initial_backbone_sha256": initial_backbone,
            "initial_residual_zero": initial_zero, "final_backbone_sha256": final_backbone,
            "frozen_backbone_unchanged": True if mode == "frozen" else None,
            "pretrained_checkpoint_path": None if pretrained_fit is None else pretrained_fit["checkpoint_path"],
            "pretrained_checkpoint": None if pretrained_fit is None else pretrained_fit["checkpoint"],
            "train_base_equals_pretrained": train_equality, "evaluation": evaluation,
            "epochs": EPOCHS[kind], "steps": steps, "episode_exposures": exposure, "episodes_per_epoch": count,
            "episode_orders": orders, "permutation_sha256": permutation.hexdigest(), **totals,
            "final_train_loss": terms["total"], "final_nonquery_loss": terms["nonquery"],
            "final_prior_loss": terms["prior"], "final_objective_prior_loss": terms["prior"],
            "final_prior_loss_scope": "constant diagnostic in frozen mode; trained in pretrain and joint modes",
            "final_spo_loss": terms["spo"], "final_objective_spo_loss": terms["spo"] if objective == "spo" else 0.,
            "final_spo_loss_scope": "post-fit diagnostic for every cell; trained only by spo cells",
            "final_rescore_spo_loss_calls": 1, "train_rescore": rescore,
            "fit_seconds": fit_only, "evaluation_clone_seconds": clone_seconds, "train_rescore_seconds": rescore_seconds,
            "checkpoint_seconds": checkpoint_seconds, "wall_seconds": (self.clock.now_ns() - tick) / 1e9}
        require(steps == EPOCHS[kind] * 9 and exposure == EPOCHS[kind] * 54
                and totals["forward_rows"] == EPOCHS[kind] * history["counts"]["rows"], "complete fixed stage exposure")
        self.event({"event": "fit_complete", **fit})
        self.receipt["fits_completed"] += 1
        return evaluated, fit, saved

    def close_training(self, fits):
        require([(r["family"], r["seed"]) for r in fits] == [(k, s) for s in SEEDS for k in KINDS]
                and self.receipt["fits_completed"] == 15 and self.receipt["optimizer_steps"] == 6480,
                "all fifteen completed stages before VALID")
        for seed in SEEDS:
            check_paired_initialization([r for r in fits if r["seed"] == seed])
        for row in fits:
            require(descriptor(self.out / row["checkpoint_path"]) == row["checkpoint"], "durable final checkpoint barrier")
        self.event({"event": "all_checkpoints_closed_before_VALID", "fits_completed": 15,
                    "checkpoints": {r["checkpoint_path"]: r["checkpoint"] for r in fits}})
        self.valid_allowed = True

    def body(self):
        tick = self.clock.now_ns()
        import numpy as np
        import torch

        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
        self.np, self.torch = np, torch
        sys.path.insert(0, str(ROOT / "src"))
        self.models = load(ROOT / MODELS, "_protected_training_models")
        self.pretrain_models = load(ROOT / PRETRAIN_MODELS, "_protected_pretrain_models")
        self.data = load(ROOT / DATA, "_protected_training_data")
        self.losses = load(ROOT / LOSS, "_protected_training_loss")
        self.windows = load(ROOT / WINDOWS, "_protected_census_windows")
        self.metrics = load(ROOT / METRICS, "_protected_metrics")
        require(self.models.MODES == ("frozen", "joint") and self.models.PARAMETERS == 6112
                and self.metrics.FAMILIES == KINDS and self.metrics.SEEDS == SEEDS
                and torch.get_num_threads() == torch.get_num_interop_threads() == 1, "fixed protected CPU models and metrics")
        self.publish("runtime.json", {**runtime_record(), "torch": torch.__version__, "numpy": np.__version__,
            "threads": 1, "interop_threads": 1, "deterministic": True, "cuda_used": False, "mps_used": False})
        train = self.training()
        setup_seconds = (self.clock.now_ns() - tick) / 1e9
        tick = self.clock.now_ns(); fits, fitted = [], []
        for seed in SEEDS:
            self.event({"event": "fit_start", "family": "pretrained", "seed": seed})
            pretrained, parent, parent_saved = self.fit("pretrained", seed, train)
            fits.append(parent); fitted.append((pretrained, parent))
            backbone = {k: v.detach().clone() for k, v in pretrained.state_dict().items()}
            for kind in BRANCHES:
                self.event({"event": "fit_start", "family": kind, "seed": seed})
                model, fit, _ = self.fit(kind, seed, train, backbone_state=backbone,
                    pretrained_fit=parent, pretrained_saved=parent_saved)
                fits.append(fit); fitted.append((model, fit))
            require(state_witness(pretrained, backbone_only=True)[0] == parent["final_backbone_sha256"],
                    "all branches leave their pretrained reference untouched")
        fit_seconds = (self.clock.now_ns() - tick) / 1e9
        self.publish("fits.json", {"fits": fits, "train_counts": train["counts"]})
        self.close_training(fits)
        tick = self.clock.now_ns()
        valid, windows, episodes, identities = self.validation()
        held = np.repeat(windows["query_scores"][:, None, :], 4, axis=1)
        held[~windows["valid_mask"]] = 0.
        self.npz("prediction-hold.npz", {"predictions": held})
        hold = metric_report(self.metrics, self.windows, np, windows, episodes, identities, held)
        records, references = [], {}
        for model, fit in fitted:
            saved, work = self.predict(model, valid, "validation_forward")
            kind, seed = fit["family"], fit["seed"]
            equality = None
            if kind == "pretrained":
                references[seed] = saved
            elif kind.startswith("frozen_"):
                equality = same_frozen_predictions(np, saved, references[seed])
            forecast = window_predictions(np, windows, valid, saved["predictions"])
            base_forecast = window_predictions(np, windows, valid, saved["base_predictions"])
            self.npz(f"prediction-{kind}-{seed}.npz", {"predictions": forecast, "base_predictions": base_forecast,
                "prior": saved["prior"], "prior_mask": saved["prior_mask"]})
            records.append({"family": kind, "architecture": fit["architecture"],
                "readout": fit["readout"], "objective": fit["objective"], "seed": seed, "prediction_work": work,
                "evaluation": fit["evaluation"], "base_equals_pretrained": equality,
                "metrics": metric_report(self.metrics, self.windows, np, windows, episodes, identities, forecast),
                "prior_metrics": self.metrics.prior_metrics(valid, saved["prior"], identities)})
        conditions = self.metrics.criteria(records, hold, technical_complete=False)
        summary = {"version": VERSION, "models": records, "hold": hold,
            "required": conditions, "required_passed": sum(c["passes"] for c in conditions),
            "required_conditions": 29, "gates": self.metrics.gate_decisions(conditions),
            "technical_complete_pending_saved_audit": True,
            "scientific_conditions": 28, "scientific_passed": sum(c["passes"] for c in conditions[1:]),
            "requires_successful_original_supervisor_and_saved_audit": True,
            "train_counts": train["counts"], "validation_counts": windows["counts"],
            "validation_history_counts": valid["counts"], "setup_seconds": setup_seconds,
            "capacity_evidence_scope": CAPACITY_SCOPE,
            "fitting_seconds": fit_seconds, "validation_seconds": (self.clock.now_ns() - tick) / 1e9,
            "scope": "Forced-path forecasts only; protected adaptation must beat all four controls. No autonomous utility, scenario shift, or architecture superiority claim."}
        self.publish("summary.json", summary)

    def execute(self):
        self.out.mkdir(exist_ok=False)

        def interrupted(_signum, _frame):
            raise InterruptedError("original protected-readout supervisor stopped worker")

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
            require(self.receipt["fits_completed"] == 15 and self.receipt["optimizer_steps"] == 6480
                and self.receipt["pending"] is self.receipt["pending_emission"] is None, "complete fixed training")
            require({p.name for p in self.out.iterdir()} == PAYLOADS, "exact 58 payload closure")
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
