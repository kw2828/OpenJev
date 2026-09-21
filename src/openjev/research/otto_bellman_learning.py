"""Caller-supervised continuation of a frozen mlp8 export, without data loading.

Default recipe implements the proposed matched MC/observation-backup control.
Torch imports are lazy. Hooks own durable records, budgets and artifact writes;
this module performs no simulator, validation, evaluation or checkpoint reads.
"""
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from types import MappingProxyType

import numpy as np

from openjev.research.otto_return_value import (
    INPUT_DIM,
    MAX_BATCH,
    FrozenValue,
    export_head,
    make_head,
    validate_head,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _owned(array):
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _seed(seed):
    _require(type(seed) is int and 0 <= seed < 2**63 - 30000, "bounded nonboolean seed required")
    return seed


@dataclass(frozen=True, slots=True)
class Recipe:
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = .001
    gradient_clip: float = 5.
    refresh_every: int = 5

    def __post_init__(self):
        for name in ("epochs", "batch_size", "refresh_every"):
            value = getattr(self, name)
            _require(type(value) is int and value > 0, f"{name} must be a positive integer")
        _require(self.batch_size <= MAX_BATCH, "batch exceeds value-model bound")
        for name in ("learning_rate", "gradient_clip"):
            value = getattr(self, name)
            _require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                     f"{name} must be finite and positive")


DEFAULT_RECIPE = Recipe()


def _snapshot(exported):
    _require(validate_head(exported) == "mlp8", "continuation requires the biased mlp8 family")
    # Normalize only metadata representation, never parameter dtype or values.
    return MappingProxyType({"version": "otto-return-value-v1", "kind": "mlp8", "input_dim": INPUT_DIM,
        **{name: _owned(value) for name, value in exported.items() if name not in ("version", "kind", "input_dim")}})


def checkpoint_identity(exported):
    """Canonical array-content identity, not the bytes of an NPZ container."""
    snapshot = _snapshot(exported)
    record = {"version": snapshot["version"], "kind": snapshot["kind"], "input_dim": snapshot["input_dim"],
              "arrays": {name: {"dtype": str(value.dtype), "shape": list(value.shape),
                                   "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
                         for name, value in snapshot.items() if isinstance(value, np.ndarray)}}
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {**record, "content_sha256": hashlib.sha256(encoded).hexdigest()}


def restore_mlp(initial_export, seed):
    """Create a fresh CPU model and copy every float32 parameter and c0 exactly."""
    import torch

    snapshot = _snapshot(initial_export)
    model = make_head("mlp8", _seed(seed), float(snapshot["c0"]))
    with torch.no_grad():
        for name, tensor in {**dict(model.named_parameters()), **dict(model.named_buffers())}.items():
            tensor.copy_(torch.from_numpy(np.array(snapshot[name], copy=True)))
    _require(checkpoint_identity(export_head(model)) == checkpoint_identity(snapshot), "exact restored checkpoint")
    return model


@dataclass(frozen=True, slots=True)
class LearningResult:
    final_export: object
    initial_identity: dict
    final_identity: dict
    epochs: tuple
    refreshes: tuple
    progress: dict
    optimizer_steps: tuple


class LearningFailure(RuntimeError):
    """A failed attempt with its original cause and last counter/context snapshot."""

    def __init__(self, original, progress):
        super().__init__(f"{type(original).__name__}: {original}")
        self.original = original
        self.progress = progress


def _targets(values, rows):
    _require(isinstance(values, np.ndarray) and values.dtype == np.float64 and values.shape == (rows,)
             and np.isfinite(values).all(), "one finite float64 normalized target per row required")
    with np.errstate(over="raise", invalid="raise"):
        cast = values.astype(np.float32)
    _require(np.isfinite(cast).all(), "nonfinite float32 target cast")
    return _owned(values), _owned(cast)


def _statistics(values, cast):
    return {"rows": len(values), "minimum": float(values.min()), "maximum": float(values.max()),
            "negative_count": int(np.count_nonzero(values < 0)),
            "mean": float(np.mean(values, dtype=np.float64)),
            "maximum_float32_cast_error": float(np.max(np.abs(cast.astype(np.float64) - values)))}


def continue_learning(initial_export, features, mc_targets, *, seed, mode, recipe=DEFAULT_RECIPE,
                      target_builder=None, hook=None):
    """Continue one fixed final-checkpoint arm; no validation or early stopping.

    features is finite float32[N,11028], mc_targets finite float64[N]. Both
    represent caller-authenticated TRAIN rows only. The module does not infer
    dataset provenance. It owns a CPU feature copy and never changes inputs.

    In backup mode target_builder(frozen_value, refresh_metadata) returns N
    float64 normalized targets, using externally qualified explicit branches.
    The immutable target network is copied before epochs1,6,... by default;
    one target array is reused for the subsequent five epochs. The callback is
    called under no_grad and receives no mutable learner or optimizer.

    hook(event, payload) receives operation_attempt/operation_return records,
    then initial, targets (MC), target_checkpoint/refresh (backup), order, epoch
    and final records. target_checkpoint precedes every target readout so the
    caller can durably write the exact float32 snapshot before using it.
    An attempt event precedes every optimizer/refresh operation, so hooks can
    enforce caller-owned budgets. Large immutable arrays are separate mapping
    entries (checkpoint, targets_float64, targets_float32, order), suitable for
    caller NPZ serialization; other entries are scalar/JSON metadata. No hook
    return value changes learning. A hook error fails this attempt without
    retry. LearningFailure chains the original error and retains counters.

    optimizer_update is atomic accounting for forward/loss/backward/clip/step
    and finite checks. Attempted but unreturned updates may have partially
    executed. Counters never imply rollback or permission to resume a failure.
    Epoch loss is the row-weighted mean of pre-update minibatch losses, not
    a separate evaluation of the final epoch model.
    """
    _require(isinstance(recipe, Recipe), "an explicit Recipe is required")
    seed = _seed(seed)
    _require(mode in ("mc", "backup"), "mode must be mc or backup")
    _require((target_builder is None if mode == "mc" else callable(target_builder)), "mode/target callback contract")
    _require(hook is None or callable(hook), "hook must be callable or None")
    _require(isinstance(features, np.ndarray) and features.dtype == np.float32 and features.ndim == 2
             and features.shape[1] == INPUT_DIM and len(features) > 0 and np.isfinite(features).all(),
             "finite float32[N,11028] TRAIN features required")
    initial = _snapshot(initial_export)
    mc64, mc32 = _targets(mc_targets, len(features))
    rows = len(features)
    progress = {"mode": mode, "seed": seed, "epochs_completed": 0, "updates_completed": 0,
                "refreshes_completed": 0, "context": {}, "calls": {name: {"attempted": 0, "returned": 0}
                for name in ("restore", "optimizer_initialization", "checkpoint_export", "target_refresh", "optimizer_update")}}

    def current_progress():
        return {**progress, "context": dict(progress["context"]),
                "calls": {name: dict(counts) for name, counts in progress["calls"].items()}}

    def emit(event, payload):
        if hook is not None:
            hook(event, payload)

    def call(name, operation, context):
        progress["context"] = dict(context)
        progress["calls"][name]["attempted"] += 1
        emit("operation_attempt", {"operation": name, "progress": current_progress()})
        result = operation()
        progress["calls"][name]["returned"] += 1
        if name == "optimizer_update":
            progress["updates_completed"] += 1
        elif name == "target_refresh":
            progress["refreshes_completed"] += 1
        emit("operation_return", {"operation": name, "progress": current_progress(),
                                  "result": result if name == "optimizer_update" else None})
        return result

    try:
        import torch

        model = call("restore", lambda: restore_mlp(initial, seed), {"phase": "restore"})
        initial_id = checkpoint_identity(initial)
        optimizer = call("optimizer_initialization", lambda: torch.optim.Adam(model.parameters(), lr=recipe.learning_rate),
                         {"phase": "optimizer_initialization"})
        _require(len(optimizer.state) == 0, "fresh Adam state required")
        emit("initial", {"checkpoint": initial, "checkpoint_identity": initial_id,
                         "recipe": asdict(recipe), "mode": mode, "seed": seed, "rows": rows,
                         "optimizer_states_before": 0, "feature_copy_bytes": features.nbytes,
                         "progress": current_progress()})
        tx = torch.from_numpy(np.array(features, copy=True, order="C"))
        rng = np.random.default_rng(seed + 30000)
        epochs, refreshes = [], []
        target64, target32, target_epoch = mc64, mc32, None
        if mode == "mc":
            emit("targets", {"targets_float64": target64, "targets_float32": target32,
                             "statistics": _statistics(target64, target32), "mode": mode})
        ty = torch.from_numpy(np.array(target32, copy=True))
        for epoch in range(1, recipe.epochs + 1):
            if mode == "backup" and (epoch - 1) % recipe.refresh_every == 0:
                snapshot = call("checkpoint_export", lambda: _snapshot(export_head(model)),
                                {"phase": "target_checkpoint", "epoch": epoch})
                identity = checkpoint_identity(snapshot)
                metadata = {"epoch": epoch, "refresh_index": len(refreshes), "source_epochs_completed": epoch - 1,
                            "rows": rows, "checkpoint_identity": identity}
                emit("target_checkpoint", {**metadata, "checkpoint": snapshot, "progress": current_progress()})
                def refresh(snapshot=snapshot, metadata=metadata):
                    frozen = FrozenValue(snapshot)
                    with torch.no_grad():
                        raw = target_builder(frozen, {**metadata, "checkpoint_identity": checkpoint_identity(snapshot)})
                    return _targets(raw, rows)
                target64, target32 = call("target_refresh", refresh, {"phase": "target_refresh", "epoch": epoch,
                                                   "refresh_index": len(refreshes)})
                target_epoch = epoch
                record = {**metadata, "statistics": _statistics(target64, target32)}
                refreshes.append(record)
                emit("refresh", {**record, "checkpoint": snapshot, "targets_float64": target64,
                                 "targets_float32": target32, "progress": current_progress()})
                ty = torch.from_numpy(np.array(target32, copy=True))
            model.train()
            order = rng.permutation(rows)
            order_sha = hashlib.sha256(order.tobytes()).hexdigest()
            emit("order", {"epoch": epoch, "order": _owned(order), "rows": rows, "sha256": order_sha})
            weighted_losses, gradient_norms = [], []
            for offset in range(0, rows, recipe.batch_size):
                indices = order[offset:offset + recipe.batch_size]
                def update(indices=indices, ty=ty):
                    optimizer.zero_grad(set_to_none=True)
                    predictions = model(tx[indices])
                    loss = ((predictions - ty[indices])**2).mean()
                    _require(bool(torch.isfinite(loss)), "finite mean squared loss required")
                    loss.backward()
                    _require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                             "finite present gradients required")
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip,
                                                                  error_if_nonfinite=True)
                    optimizer.step()
                    _require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "finite post-update parameters required")
                    _require(model.c0.item() == float(initial["c0"]), "fixed c0 baseline changed")
                    return {"loss": float(loss.detach()), "rows": len(indices),
                            "gradient_norm_before_clip": float(gradient_norm)}
                result = call("optimizer_update", update, {"phase": "optimizer_update", "epoch": epoch,
                               "batch_index": offset // recipe.batch_size, "offset": offset, "rows": len(indices)})
                weighted_losses.append(result["loss"] * len(indices))
                gradient_norms.append(result["gradient_norm_before_clip"])
            progress["epochs_completed"] = epoch
            record = {"epoch": epoch, "rows": rows, "updates": len(gradient_norms), "order_sha256": order_sha,
                      "training_mse_normalized": math.fsum(weighted_losses) / rows,
                      "mean_gradient_norm_before_clip": math.fsum(gradient_norms) / len(gradient_norms),
                      "maximum_gradient_norm_before_clip": max(gradient_norms), "target_refresh_epoch": target_epoch}
            epochs.append(record)
            emit("epoch", {**record, "progress": current_progress()})
        final = call("checkpoint_export", lambda: _snapshot(export_head(model)), {"phase": "final_export"})
        final_id = checkpoint_identity(final)
        steps = tuple(int(state["step"].item()) for state in optimizer.state.values())
        expected = recipe.epochs * ((rows + recipe.batch_size - 1) // recipe.batch_size)
        _require(steps and set(steps) == {expected} and progress["updates_completed"] == expected,
                 "complete fresh-Adam update count")
        _require(final["c0"].tobytes() == initial["c0"].tobytes(), "exact unchanged c0 bytes")
        emit("final", {"checkpoint": final, "checkpoint_identity": final_id, "optimizer_steps": steps,
                       "progress": current_progress()})
        return LearningResult(final, initial_id, final_id, tuple(epochs), tuple(refreshes), current_progress(), steps)
    except BaseException as error:  # Preserve interrupted and partial attempts without retry.
        failure = LearningFailure(error, current_progress())
        try:
            emit("failure", {"error": f"{type(error).__name__}: {error}", "progress": current_progress()})
        except BaseException as hook_error:  # noqa: BLE001 - A failure hook must not replace the primary error.
            failure.add_note(f"Failure hook also failed: {type(hook_error).__name__}: {hook_error}")
        raise failure from error
