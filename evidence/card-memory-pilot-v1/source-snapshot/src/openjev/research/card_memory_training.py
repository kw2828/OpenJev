"""Small public-card learning core; no environment, seed allocation or selection.

The caller authenticates the corpus, layout splits, initial weights and orders.
Exact visibility and visibility-age replay belongs to the collector/auditor:
one selected reveal cannot reconstruct other cards' continued public visibility.
Loss and evaluation average queries within a boundary, nonempty boundaries
within an episode, then eligible episodes. A layout is one supplied episode.
This module never consults future reveals when constructing model inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

VERSION = "card-memory-training-v1"
FIELDS = {"positions", "ranks", "valid", "targets", "target_mask", "ages"}
ADAM = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0,
        "amsgrad": False, "foreach": False, "maximize": False,
        "capturable": False, "differentiable": False, "fused": False}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_data(data):
    """Return owned CPU tensors, rejecting noncausal labels and padding masks."""
    _require(isinstance(data, dict) and set(data) == FIELDS, "Exact six data fields required")
    result = {}
    dtypes = {"positions": torch.int64, "ranks": torch.int64, "valid": torch.bool,
              "targets": torch.int64, "target_mask": torch.bool, "ages": torch.int32}
    for name in sorted(FIELDS):
        value = torch.as_tensor(data[name])
        _require(value.device.type == "cpu" and value.dtype == dtypes[name]
                 and not value.requires_grad, f"CPU {dtypes[name]} required: {name}")
        result[name] = value.detach().clone()
    pos, rank, valid = (result[k] for k in ("positions", "ranks", "valid"))
    _require(pos.ndim == 2 and pos.shape[0] > 0 and 1 <= pos.shape[1] <= 104,
             "Positive episode count and one to 104 boundaries required")
    n, steps = pos.shape
    _require(rank.shape == valid.shape == pos.shape, "Reveal shapes must agree")
    _require(all(result[k].shape == (n, steps, 52) for k in ("targets", "target_mask", "ages")),
             "Target fields must have shape [N,T,52]")
    _require(not bool((valid[:, 1:] & ~valid[:, :-1]).any()), "valid must be a prefix")
    _require(bool(((pos[valid] >= 0) & (pos[valid] < 52)).all())
             and bool(((rank[valid] >= 0) & (rank[valid] < 13)).all()), "Valid reveal out of range")
    targets, mask, ages = (result[k] for k in ("targets", "target_mask", "ages"))
    _require(not bool((mask & ~valid[..., None]).any()), "Padding cannot carry query targets")
    _require(bool((targets[~mask] == -1).all()) and bool((ages[~mask] == -1).all()),
             "Unqueried target and age must be -1")
    known = torch.full((n, 52), -1, dtype=torch.int64)
    last_reveal_clock = torch.full((n, 52), -1, dtype=torch.int64)
    rows = torch.arange(n)
    for step in range(steps):
        active = mask[:, step]
        _require(bool((known[active] >= 0).all())
                 and torch.equal(targets[:, step][active], known[active]),
                 "Targets must equal previously revealed public ranks")
        elapsed = step - last_reveal_clock
        _require(bool((ages[:, step][active] >= 1).all())
                 and bool((ages[:, step][active] <= elapsed[active]).all()),
                 "Target age must fit earlier public reveal timing")
        chosen = valid[:, step]
        r, p = rows[chosen], pos[:, step][chosen]
        previous = known[r, p]
        _require(bool(((previous == -1) | (previous == rank[:, step][chosen])).all()),
                 "A revealed card rank cannot change within an episode")
        known[r, p] = rank[:, step][chosen]
        last_reveal_clock[r, p] = step + 1
    return result


def _model_device(model):
    parameters = list(model.parameters())
    _require(bool(parameters) and all(p.device.type == "cpu" and p.dtype == torch.float32
                                    and bool(torch.isfinite(p).all()) for p in parameters),
             "Finite CPU float32 model parameters required")
    return parameters[0].device


def _finite(value, name):
    if isinstance(value, torch.Tensor):
        _require(bool(torch.isfinite(value).all()), f"Nonfinite {name}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _finite(child, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite(child, name)
    elif isinstance(value, float):
        _require(math.isfinite(value), f"Nonfinite {name}")


def _padding_copy(state, padding):
    _require(isinstance(state, dict), "Model state must be a tensor dictionary")
    result = {}
    for key, value in state.items():
        _require(value is None or (isinstance(value, torch.Tensor) and value.ndim >= 1
                                  and value.shape[0] == len(padding)), "Batched tensor or None state required")
        result[key] = value[padding].detach().clone() if value is not None else None
    return result


def _padding_check(saved, state, padding):
    _require(isinstance(state, dict) and set(saved) == set(state), "Write changed state fields")
    for key, value in saved.items():
        actual = state[key]
        _require((value is None and actual is None) or
                 (value is not None and isinstance(actual, torch.Tensor)
                  and actual.ndim >= 1 and actual.shape[0] == len(padding)
                  and torch.equal(value, actual[padding])), "Padding write changed state")


def _counts():
    return {k: 0 for k in ("predict_calls_attempted", "predict_calls_completed",
                           "write_calls_attempted", "write_calls_completed",
                           "predicted_card_logits", "valid_reveals", "padded_write_examples",
                           "query_cards", "query_boundaries", "eligible_episodes")}


def _increment(work, key, amount=1):
    work[key] = work.get(key, 0) + amount


def _check(deadline):
    if deadline is not None:
        _require(type(deadline) in (int, float) and math.isfinite(deadline), "Finite monotonic deadline required")
        if time.monotonic() >= deadline:
            raise TimeoutError("Card memory deadline exceeded")


def _forward(model, data, *, deadline=None, work=None):
    device = _model_device(model)
    batch, steps = data["valid"].shape
    state = model.init_state(batch, device=device)
    _finite(state, "initial state")
    queries = torch.arange(52, device=device).expand(batch, -1)
    counts = _counts() if work is None else work
    sums = {group: {metric: torch.zeros(batch, device=device) for metric in ("ce", "brier", "accuracy")}
            for group in ("all", "age_gt32")}
    boundaries = {group: torch.zeros(batch, dtype=torch.int64) for group in sums}
    cards = {group: torch.zeros(batch, dtype=torch.int64) for group in sums}
    for step in range(steps):
        _check(deadline)
        valid = data["valid"][:, step]
        if not bool(valid.any()):
            break
        _increment(counts, "predict_calls_attempted")
        logits = model.predict(state, queries)
        _increment(counts, "predict_calls_completed")
        _require(isinstance(logits, torch.Tensor) and logits.shape == (batch, 52, 13)
                 and logits.dtype == torch.float32 and logits.device.type == "cpu", "Prediction must be CPU float32 [B,52,13]")
        _finite(logits, "prediction")
        _increment(counts, "predicted_card_logits", batch * 52)
        mask = data["target_mask"][:, step]
        target = torch.where(mask, data["targets"][:, step], 0)
        ce = F.cross_entropy(logits.reshape(-1, 13), target.reshape(-1), reduction="none").reshape(batch, 52)
        probabilities = logits.softmax(-1)
        brier = (probabilities - F.one_hot(target, 13)).square().sum(-1)
        accuracy = (logits.argmax(-1) == target).float()
        for group, selected in (("all", mask), ("age_gt32", mask & (data["ages"][:, step] > 32))):
            n = selected.sum(1)
            boundaries[group] += n > 0
            cards[group] += n
            for name, values in (("ce", ce), ("brier", brier), ("accuracy", accuracy)):
                sums[group][name] = sums[group][name] + (values * selected).sum(1) / n.clamp_min(1)
        padding = ~valid
        unchanged = _padding_copy(state, padding)
        pos = torch.where(valid, data["positions"][:, step], 0)
        rank = torch.where(valid, data["ranks"][:, step], 0)
        _increment(counts, "write_calls_attempted")
        returned = model.write(state, pos, rank, valid)
        _increment(counts, "write_calls_completed")
        _require(isinstance(returned, tuple) and len(returned) == 2, "write must return (state, diagnostics)")
        state, diagnostics = returned
        _finite(state, "written state")
        _finite(diagnostics, "write diagnostics")
        _padding_check(unchanged, state, padding)
        _increment(counts, "valid_reveals", int(valid.sum()))
        _increment(counts, "padded_write_examples", int(padding.sum()))
        _check(deadline)
    _increment(counts, "query_cards", int(cards["all"].sum()))
    _increment(counts, "query_boundaries", int(boundaries["all"].sum()))
    _increment(counts, "eligible_episodes", int((boundaries["all"] > 0).sum()))
    return sums, boundaries, cards, counts


def _indices(indices, n):
    result = torch.as_tensor(indices)
    _require(result.dtype == torch.int64 and result.device.type == "cpu" and result.ndim == 1
             and len(result) > 0 and bool(((result >= 0) & (result < n)).all())
             and len(result.unique()) == len(result), "Unique nonempty int64 episode indices required")
    return result


def _loss(model, data, *, deadline=None, work=None):
    sums, boundaries, _, counts = _forward(model, data, deadline=deadline, work=work)
    eligible = boundaries["all"] > 0
    if bool(eligible.any()):
        loss = (sums["all"]["ce"][eligible] / boundaries["all"][eligible]).mean()
    else:
        loss = sum(p.sum() * 0 for p in model.parameters())
    _finite(loss, "loss")
    return loss, counts


def batch_loss(model, data, indices):
    """Differentiable episode-balanced CE and actual completed forward counts."""
    validated = validate_data(data)
    selected = _indices(indices, len(validated["valid"]))
    return _loss(model, {key: value[selected] for key, value in validated.items()})


def evaluate(model, data, *, batch_size=16, return_episode_metrics=True, wall_deadline=None):
    """No optimizer, model selection or pooled-query weighting; empty strata are None."""
    _require(type(batch_size) is int and batch_size > 0, "Positive integer batch size")
    _require(type(return_episode_metrics) is bool, "Explicit boolean episode reporting")
    begin = time.monotonic()
    _check(wall_deadline)
    validated = validate_data(data)
    previous_training = model.training
    work, episodes = _counts(), []
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(validated["valid"]), batch_size):
                batch = {key: value[start:start + batch_size] for key, value in validated.items()}
                sums, boundaries, cards, _ = _forward(model, batch, deadline=wall_deadline, work=work)
                for i in range(len(batch["valid"])):
                    episode = {"episode_index": start + i}
                    for group in sums:
                        n = int(boundaries[group][i])
                        numerators = {name: float(value[i]) for name, value in sums[group].items()}
                        episode[group] = {"query_boundaries": n, "query_cards": int(cards[group][i]),
                                          "boundary_mean_sums": numerators,
                                          **{name: value / n if n else None for name, value in numerators.items()}}
                    episodes.append(episode)
    finally:
        model.train(previous_training)
    result = {"version": VERSION, "scope": "public-target prediction; caller authenticates layouts and visibility",
              "weighting": "queries within boundary, nonempty boundaries within episode, eligible episodes equally",
              "brier_definition": "sum over 13 classes of squared probability error",
              "episodes": len(episodes), "work": work}
    for group in ("all", "age_gt32"):
        selected = [ep[group] for ep in episodes if ep[group]["query_boundaries"] > 0]
        result[group] = {"eligible_episodes": len(selected),
                         "query_boundaries": sum(ep["query_boundaries"] for ep in selected),
                         "query_cards": sum(ep["query_cards"] for ep in selected),
                         **{name: math.fsum(ep[name] for ep in selected) / len(selected) if selected else None
                            for name in ("ce", "brier", "accuracy")}}
    if return_episode_metrics:
        result["per_episode"] = episodes
    _check(wall_deadline)
    result["wall_seconds"] = time.monotonic() - begin
    return result


def _tensor_hash(mapping):
    digest = hashlib.sha256()
    for name, value in sorted(mapping.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _configuration(model):
    return model.configuration() if callable(getattr(model, "configuration", None)) else {
        "class": type(model).__name__, "mode": getattr(model, "mode", None),
        "parameter_counts": model.parameter_counts() if callable(getattr(model, "parameter_counts", None)) else None}


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_fit(model, data, orders, out_path, lr=0.003, batch_size=16, wall_deadline=None):
    """Train supplied model in-place once; out_path is an exclusive directory.

    Orders are explicit whole-episode permutations. No generators are created.
    Only the final successful update schedule produces a checkpoint. Failures
    preserve completed batch/epoch logs and never produce a resumable fit.
    """
    begin = time.monotonic()
    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=False)
    work, phase = _counts(), "validation"
    counters = {name: 0 for name in ("backward_attempted", "backward_completed", "optimizer_attempted",
                                    "optimizer_returned", "successful_updates", "completed_epochs")}
    timings = {"forward_seconds": 0.0, "backward_seconds": 0.0, "optimizer_seconds": 0.0}
    previous_training = model.training
    try:
        _json(out / "started.json", {"version": VERSION, "status": "started", "automatic_retry": False,
                                     "model_class": type(model).__name__, "mode": getattr(model, "mode", None)})
        _check(wall_deadline)
        _require(type(lr) in (int, float) and math.isfinite(lr) and lr > 0, "Positive finite learning rate")
        _require(type(batch_size) is int and batch_size > 0, "Positive integer batch size")
        validated = validate_data(data)
        _model_device(model)
        orders = torch.as_tensor(orders)
        n = len(validated["valid"])
        _require(orders.device.type == "cpu" and orders.dtype == torch.int64 and orders.ndim == 2
                 and orders.shape[0] > 0 and orders.shape[1] == n, "Explicit int64 [epochs,N] orders required")
        _require(all(torch.equal(row.sort().values, torch.arange(n)) for row in orders), "Each order must be a permutation")
        orders = orders.clone()
        identity = {"initial_weights_sha256": _tensor_hash(model.state_dict()),
                    "data_sha256": _tensor_hash(validated), "orders_sha256": _tensor_hash({"orders": orders})}
        recipe = {"epochs": len(orders), "episodes": n, "batch_size": batch_size, "learning_rate": lr,
                  "adam": ADAM, "gradient_norm_clip": 1.0, "final_checkpoint_only": True,
                  "expected_updates": len(orders) * math.ceil(n / batch_size)}
        configuration = _configuration(model)
        _json(out / "fit-settings.json", {"identity": identity, "recipe": recipe,
              "model_class": type(model).__name__, "configuration": configuration})
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, **ADAM)
        model.train()
        with (out / "batches.jsonl").open("x", encoding="utf-8") as log:
            for epoch, order in enumerate(orders):
                epoch_begin, epoch_weighted, eligible_sum = time.monotonic(), [], 0
                for start in range(0, n, batch_size):
                    _check(wall_deadline)
                    phase = "forward"
                    indices = order[start:start + batch_size]
                    batch = {key: value[indices] for key, value in validated.items()}
                    optimizer.zero_grad(set_to_none=True)
                    before_eligible = work["eligible_episodes"]
                    tick = time.monotonic()
                    try:
                        loss, _ = _loss(model, batch, deadline=wall_deadline, work=work)
                    finally:
                        timings["forward_seconds"] += time.monotonic() - tick
                    eligible = work["eligible_episodes"] - before_eligible
                    _require(eligible > 0, "Training batch has no public query targets")
                    phase = "backward"
                    _check(wall_deadline)
                    counters["backward_attempted"] += 1
                    tick = time.monotonic()
                    try:
                        loss.backward()
                        counters["backward_completed"] += 1
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    finally:
                        timings["backward_seconds"] += time.monotonic() - tick
                    phase = "optimizer"
                    _check(wall_deadline)
                    counters["optimizer_attempted"] += 1
                    tick = time.monotonic()
                    try:
                        optimizer.step()
                        counters["optimizer_returned"] += 1
                        _model_device(model)
                        _finite(optimizer.state_dict(), "Adam state")
                    finally:
                        timings["optimizer_seconds"] += time.monotonic() - tick
                    counters["successful_updates"] += 1
                    value = float(loss.detach())
                    epoch_weighted.append(value * eligible)
                    eligible_sum += eligible
                    log.write(json.dumps({"epoch": epoch, "batch_start": start, "indices": indices.tolist(),
                                          "loss": value, "eligible_episodes": eligible,
                                          "gradient_norm_before_clip": float(norm),
                                          "successful_updates": counters["successful_updates"]}, allow_nan=False) + "\n")
                    log.flush()
                    _check(wall_deadline)
                counters["completed_epochs"] += 1
                _json(out / f"epoch-{epoch:03d}.json", {"epoch": epoch, "status": "complete",
                      "episode_balanced_training_loss": math.fsum(epoch_weighted) / eligible_sum,
                      "eligible_episodes": eligible_sum, "successful_updates": counters["successful_updates"],
                      "wall_seconds": time.monotonic() - epoch_begin})
        phase = "final_serialization"
        _check(wall_deadline)
        _require(_configuration(model) == configuration, "Model configuration changed during training")
        checkpoint = {"version": VERSION, "model_class": type(model).__name__, "configuration": configuration,
                      "weights": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                      "optimizer": optimizer.state_dict(), "orders": orders, "recipe": recipe,
                      "identity": identity, "counts": dict(counters), "resume_authorized": False}
        with (out / "final-checkpoint.pt").open("xb") as handle:
            torch.save(checkpoint, handle)
        _check(wall_deadline)
        files = {p.name: {"sha256": _sha(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir())}
        _check(wall_deadline)
        completed = {"version": VERSION, "status": "complete", "identity": identity, "recipe": recipe,
                     "final_weights_sha256": _tensor_hash(model.state_dict()), "counts": counters,
                     "work": work, "timings": timings, "files": files,
                     "wall_seconds": time.monotonic() - begin, "evaluation_calls": 0,
                     "scope": "caller-supplied public data and orders; final fit only; no external authenticity certification"}
        _json(out / "completed.json", completed)
        _check(wall_deadline)
        return completed
    except BaseException as error:
        original_error = error
        actions = [lambda: (out / "completed.json").rename(out / "invalid-completion.json")
                   if (out / "completed.json").exists() else None,
                   lambda: (out / "final-checkpoint.pt").rename(out / "invalid-final-checkpoint.pt")
                   if (out / "final-checkpoint.pt").exists() else None,
                   lambda: _json(out / "failed.json", {"version": VERSION, "status": "failed", "phase": phase,
                            "error": repr(original_error), "exception_type": type(original_error).__name__, "counts": counters,
                            "work": work, "timings": timings, "wall_seconds": time.monotonic() - begin,
                            "automatic_retry": False, "resume_authorized": False})]
        for action in actions:
            try:
                action()
            except BaseException as preservation_error:  # noqa: BLE001 - preserve the original failure
                note = getattr(error, "add_note", None)
                if note:
                    note(f"Failure preservation also failed: {preservation_error!r}")
        raise
    finally:
        model.train(previous_training)
