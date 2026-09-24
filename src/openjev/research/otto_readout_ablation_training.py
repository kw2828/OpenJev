"""Source-pinned training fork with only the admitted model type changed.

The complete optimizer, loss, chronological32-step chunks, carry detachment,
gradient completion/clipping and whole-batch update bodies are unchanged from
the qualified query-memory trainer. This module accepts only the new no-memory
readout masks. It adds no sampling, dataset IO, evaluation or study admission.
The caller binds TRAIN provenance and witnesses frozen weights outside batches.
"""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch

from openjev.research import otto_query_memory as memory
from openjev.research import otto_query_memory_data as data_module
from openjev.research import otto_query_memory_training as original
from openjev.research import otto_readout_ablation_model as models

VERSION = "otto-readout-ablation-training-v1"
ORIGINAL_SOURCE_SHA256 = "ab03857158f10b03466fad8543e939385478df0f5d28bbf489c9b49bde439704"
DATA_SOURCE_SHA256 = "a41ebb635c946dd1861590bc173259a7d96de947d5ec6dfa53acb064de19ebf1"
LEARNING_RATE = .003
GRADIENT_CLIP = 5.0
require = memory.require


def _effective(model):
    for module, expected in ((original, ORIGINAL_SOURCE_SHA256), (data_module, DATA_SOURCE_SHA256)):
        require(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected,
                "unchanged qualified training source: " + module.__name__)
    require(type(model) is models.ReadoutAblationModel, "qualified readout ablation training model")
    require(model.slow.query_period == 4, "training uses actual period-four observations")
    require(model.config == memory.Config(model.config.mode, key_dim=8), "fixed eight-key memory constants")
    expected = model.effective_named_parameters()
    require(bool(expected), "model has effective optimization parameters")
    require(len({name for name, _ in expected}) == len(expected)
            and len({id(p) for _, p in expected}) == len(expected), "distinct effective parameter names and objects")
    all_named = dict(model.named_parameters())
    for name, parameter in expected:
        require(name in all_named and all_named[name] is parameter and parameter.requires_grad
                and parameter.device.type == "cpu" and parameter.dtype == torch.float32,
                "owned CPU float32 effective parameter: " + name)
    for name, parameter in all_named.items():
        require(parameter.device.type == "cpu" and parameter.dtype == torch.float32
                and bool(torch.isfinite(parameter).all()), "finite CPU float32 model parameter: " + name)
    if model.projection is not None:
        require(bool((model.projection.weight != 0).any()), "nonzero key projection; no dead-start replacement")
    return expected


def _step_number(optimizer, parameters):
    allowed = {id(p) for p in parameters}
    require(all(id(p) in allowed for p in optimizer.state), "no optimizer state outside effective parameters")
    if not optimizer.state:
        return 0
    require(len(optimizer.state) == len(parameters), "complete effective Adam state")
    steps = []
    for parameter in parameters:
        state = optimizer.state[parameter]
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "default Adam state fields")
        step = state["step"]
        require(isinstance(step, torch.Tensor) and step.device.type == "cpu" and step.ndim == 0
                and not step.requires_grad and bool(torch.isfinite(step)), "finite CPU Adam step")
        number = float(step)
        require(number >= 1 and number.is_integer(), "positive integer Adam step")
        steps.append(int(number))
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                    and value.dtype == torch.float32 and value.shape == parameter.shape
                    and not value.requires_grad and bool(torch.isfinite(value).all()), "finite effective Adam moment")
        require(bool((state["exp_avg_sq"] >= 0).all()), "nonnegative second Adam moment")
    require(len(set(steps)) == 1, "same completed Adam step for every effective parameter")
    return steps[0]


def _validate_optimizer(model, optimizer):
    named = _effective(model)
    parameters = [p for _, p in named]
    require(type(optimizer) is torch.optim.Adam and len(optimizer.param_groups) == 1, "one ordinary Adam parameter group")
    group = optimizer.param_groups[0]
    require([id(p) for p in group["params"]] == [id(p) for p in parameters],
            "optimizer owns exactly the effective parameters in declared order")
    require(group["lr"] == LEARNING_RATE and group["weight_decay"] == 0
            and group["betas"] == (.9, .999) and group["eps"] == 1e-8
            and not group["amsgrad"] and not group["maximize"] and not group["capturable"]
            and not group["differentiable"] and group.get("foreach") is None
            and group.get("fused") is None, "fixed default Adam settings")
    effective_ids = {id(p) for p in parameters}
    for name, parameter in model.named_parameters():
        if id(parameter) not in effective_ids:
            require(parameter.grad is None, "no frozen parameter gradient: " + name)
    return named, _step_number(optimizer, parameters)


def construct_optimizer(model):
    """Create a fresh fixed Adam over actual gradient paths, without RNG draws."""
    named = _effective(model)
    optimizer = torch.optim.Adam([p for _, p in named], lr=LEARNING_RATE, weight_decay=0.0)
    _validate_optimizer(model, optimizer)
    require(not optimizer.state, "fresh Adam state")
    return optimizer


def _owned_tensor(value):
    require(isinstance(value, np.ndarray), "chunk payload is a NumPy array")
    # torch.tensor owns its storage even when the NumPy producer returns a view.
    return torch.tensor(value, device="cpu")


def _add_counts(destination, values):
    require(isinstance(values, dict) and all(type(name) is str and type(number) is int and number >= 0
                                            for name, number in values.items()), "nonnegative operation counters")
    for name, number in values.items():
        destination[name] = destination.get(name, 0) + number


def batch_update(model, optimizer, data, indices, *, check=lambda: None, stage=lambda name, start: None):
    """Update one explicit batch of complete TRAIN episodes exactly once.

    ``check()`` is the caller's resource/deadline guard. ``stage(name, start)``
    records pending work, with a chunk start or None for batch operations.
    Exceptions propagate; this function never retries or synthesizes completion.
    Returned values are detached scalars/counters only. Loss sums retain the
    data helper's fixed episode_count / actual_batch_size multiplier.
    """
    wall_start = time.perf_counter_ns()
    require(callable(check) and callable(stage), "guard and stage callbacks")
    check()
    require(isinstance(data, dict) and data.get("version") == data_module.VERSION
            and data.get("stage") == "train" and type(data.get("query_period")) is int
            and data["query_period"] == 4, "authenticated TRAIN period-four census input")
    count = data.get("episode_count")
    require(type(count) is int and count > 0, "positive complete TRAIN episode count")
    require(isinstance(indices, (list, tuple)) and 1 <= len(indices) <= 6
            and all(type(i) is int and 0 <= i < count for i in indices)
            and len(set(indices)) == len(indices), "one to six distinct declared TRAIN episode indices")
    offsets = data.get("episode_offsets")
    require(isinstance(offsets, np.ndarray) and offsets.dtype == np.int64 and offsets.shape == (count + 1,)
            and offsets[0] == 0, "complete chronological episode offsets")
    all_lengths = np.diff(offsets)
    require(bool(((all_lengths >= 1) & (all_lengths <= data_module.HORIZON)).all()), "bounded complete episodes")
    lengths = [int(all_lengths[i]) for i in indices]
    named, previous_step = _validate_optimizer(model, optimizer)
    parameters = [p for _, p in named]
    effective_ids = {id(p) for p in parameters}
    stage("zero_grad", None)
    optimizer.zero_grad(set_to_none=True)
    carry = model.initial_carry(len(indices))
    timings = {name: 0.0 for name in ("materialize", "forward", "loss", "backward", "carry_detach",
                                     "gradient_completion", "gradient_clip", "optimizer")}
    setup_seconds = (time.perf_counter_ns() - wall_start) / 1e9
    counts, units = {}, {}
    chunks = backward_chunks = loss_chunks = rows = nonquery_rows = prior_rows = 0
    total_loss = nonquery_loss = prior_loss = 0.0
    for start in range(0, max(lengths), data_module.CHUNK):
        check()
        stage("chunk_materialize", start)
        before = time.perf_counter_ns()
        packet = data_module.batch_chunk(data, indices, start)
        inputs = {name: _owned_tensor(value) for name, value in packet["model_inputs"].items()}
        targets = _owned_tensor(packet["targets"])
        legal = _owned_tensor(packet["legal"])
        weights = _owned_tensor(packet["nonquery_weights"])
        prior_weights = _owned_tensor(packet["prior_weights"])
        prior_mask = _owned_tensor(packet["prior_mask"])
        timings["materialize"] += (time.perf_counter_ns() - before) / 1e9
        check()
        stage("chunk_forward", start)
        with torch.enable_grad():
            before = time.perf_counter_ns()
            forecast = model(**inputs, carry=carry)
            timings["forward"] += (time.perf_counter_ns() - before) / 1e9
            require(torch.equal(forecast.prior_mask, prior_mask), "actual later-query prior masks agree")
            check()
            stage("chunk_loss", start)
            before = time.perf_counter_ns()
            terms = data_module.weighted_loss(
                forecast.action_prediction, forecast.corrected_shadow_prior, targets, legal,
                weights, prior_weights, inputs["query_mask"], prior_mask, episode_count=count)
            timings["loss"] += (time.perf_counter_ns() - before) / 1e9
            loss = terms["total"]
            require(bool(torch.isfinite(loss)), "finite complete-forecast AUX loss")
            total_loss += float(loss.detach())
            nonquery_loss += float(terms["nonquery"].detach())
            prior_loss += float(terms["prior"].detach())
            has_support = bool((weights > 0).any()) or bool(prior_mask.any())
            loss_chunks += int(has_support)
            if loss.requires_grad:
                require(has_support, "differentiable loss requires scored support")
                check()
                stage("chunk_backward", start)
                before = time.perf_counter_ns()
                loss.backward()
                timings["backward"] += (time.perf_counter_ns() - before) / 1e9
                backward_chunks += 1
                check()
        stage("chunk_detach", start)
        before = time.perf_counter_ns()
        carry = models.detach_carry(forecast.carry)
        timings["carry_detach"] += (time.perf_counter_ns() - before) / 1e9
        _add_counts(counts, forecast.work_counts)
        _add_counts(units, forecast.memory_work_units)
        rows += int(inputs["lengths"].sum())
        nonquery_rows += int((weights > 0).sum())
        prior_rows += int(prior_mask.sum())
        chunks += 1
    require(bool(carry.fast.ended.all()) and bool(carry.slow.base.ended.all())
            and carry.fast.absolute_step.tolist() == lengths
            and carry.slow.base.absolute_step.tolist() == lengths, "all complete episode tails consumed")
    check()
    stage("complete_gradients", None)
    before = time.perf_counter_ns()
    missing = []
    for name, parameter in model.named_parameters():
        if id(parameter) not in effective_ids:
            require(parameter.grad is None, "frozen slow parameter has no gradient: " + name)
    for name, parameter in named:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
            missing.append(name)
        require(parameter.grad.device.type == "cpu" and parameter.grad.dtype == torch.float32
                and parameter.grad.shape == parameter.shape and bool(torch.isfinite(parameter.grad).all()),
                "finite effective gradient: " + name)
    timings["gradient_completion"] += (time.perf_counter_ns() - before) / 1e9
    require(_step_number(optimizer, parameters) == previous_step, "no optimizer update before all episode chunks")
    stage("clip_gradients", None)
    before = time.perf_counter_ns()
    norm = torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP, error_if_nonfinite=True)
    timings["gradient_clip"] += (time.perf_counter_ns() - before) / 1e9
    require(bool(torch.isfinite(norm)), "finite total gradient norm")
    check()
    stage("optimizer_update", None)
    before = time.perf_counter_ns()
    optimizer.step()
    timings["optimizer"] += (time.perf_counter_ns() - before) / 1e9
    check()
    final_named, final_step = _validate_optimizer(model, optimizer)
    require([id(p) for _, p in final_named] == [id(p) for p in parameters]
            and final_step == previous_step + 1, "exactly one Adam step per whole episode batch")
    require(all(math.isfinite(value) for value in (total_loss, nonquery_loss, prior_loss)), "finite summed batch losses")
    elapsed = (time.perf_counter_ns() - wall_start) / 1e9
    accounted = setup_seconds + math.fsum(timings.values())
    return {"version": VERSION, "objective": "full_forecast_aux", "loss": total_loss,
            "nonquery_loss": nonquery_loss, "prior_loss": prior_loss,
            "episode_indices": list(indices), "episode_exposures": len(indices), "forward_rows": rows,
            "nonquery_rows": nonquery_rows, "prior_rows": prior_rows, "forward_chunks": chunks,
            "loss_chunks": loss_chunks, "backward_chunks": backward_chunks,
            "differentiable_chunks": backward_chunks, "no_gradient_chunks": chunks - backward_chunks,
            "skipped_backward_chunks": chunks - backward_chunks, "optimizer_updates": 1,
            "optimizer_step": final_step, "gradient_norm_before_clip": float(norm),
            "zero_filled_gradient_names": missing, "effective_parameter_names": [name for name, _ in named],
            "effective_parameter_count": sum(p.numel() for p in parameters),
            "frozen_optimizer_state_entries": 0, "work_counts": counts, "memory_work_units": units,
            "timing_seconds": {"wall": elapsed, "setup": setup_seconds, **timings,
                               "overhead": max(0.0, elapsed - accounted)}}
