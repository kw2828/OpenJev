"""Matched R16/R64 continuation learning with the original R16 target scale.

No data or checkpoint files are read or written here. The caller authenticates
the common cohort, supplies targets from otto_cost_regression and owns durable
records, resource checks and complete elapsed-time accounting. There is no
validation, early stopping, checkpoint selection, simulator or rollout sampler.

The learning body is inherited from otto_teacher_learning.py at SHA256
9ae311cd39f42fa727245149ca5daaae168ed542264afeeb9b0b1803b57b54be.
Only arm/seed/bundle identity changes; optimizer, loss and callback behavior
remain the same. Neither arm uses analytic-preference range normalization.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

import numpy as np

VERSION = "otto-teacher-precision-v1"
ARMS = ("r16", "r64")
SEEDS = (20101, 20102, 20103)
INPUT_DIM = 2836
TRAIN_ROWS = 558
LEARNING_RATE = 3e-4
GRADIENT_CLIP = 5.0
PARITY_ATOL = PARITY_RTOL = 2e-5
CHANNELS = ("model_initialization", "checkpoint_export", "checkpoint_publication",
            "optimizer_initialization", "optimizer_update", "inference_setup", "parity_numpy", "parity_torch")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _owned(array):
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def build_precision_targets(r16_costs, r64_costs, allowed, episode_ids, *, reference_continuation):
    """Own two continuation bundles, both divided by the saved R16 global RMS.

    The reference is the complete original continuation-arm bundle, with kind
    'continuation'. Its centered targets, both weight dtypes, mask, scale and
    float32 casts must match freshly reconstructed R16 targets byte-for-byte.
    R64 is centered by the same qualified continuation recipe, but its own RMS
    is discarded. Both arms use the original R16 RMS and episode weights.

    The caller authenticates unchanged raw R16 costs, common feature bytes,
    masks/episode identities and complete integer continuation records for all
    64 replicates. This constructor verifies arithmetic, not that provenance.
    All returned arrays are fresh immutable copies; callers are not mutated.
    """
    from openjev.research.otto_cost_regression import build_targets

    fields = {"kind", "centered_float64", "scale", "scaled_float32",
              "weights_float64", "weights_float32", "allowed"}
    reference = reference_continuation
    _require(isinstance(reference, dict) and set(reference) == fields
             and reference["kind"] == "continuation", "exact original continuation reference required")
    original = build_targets(r16_costs, allowed, episode_ids, kind="continuation")
    for key in fields - {"kind", "scale"}:
        value, expected = reference[key], original[key]
        _require(isinstance(value, np.ndarray) and value.dtype == expected.dtype
                 and value.shape == expected.shape and value.tobytes() == expected.tobytes(),
                 f"original R16 {key} bytes differ")
    _require(type(reference["scale"]) in (int, float) and math.isfinite(reference["scale"])
             and float(reference["scale"]).hex() == original["scale"].hex(), "original R16 scale differs")
    combined = build_targets(r64_costs, allowed, episode_ids, kind="continuation")
    scale = float(reference["scale"])
    bundle = {"version": VERSION}
    for arm, source in (("r16", original), ("r64", combined)):
        bundle[arm] = {"kind": arm, "scale": scale,
                       **{key: _owned(source[key]) for key in fields - {"kind", "scale", "scaled_float32"}},
                       "scaled_float32": _owned((source["centered_float64"] / scale).astype(np.float32))}
    return _bundle(bundle, len(r16_costs))


@dataclass(frozen=True, slots=True)
class Recipe:
    """Production defaults; smaller explicit recipes are for fabricated tests only."""
    epochs: int = 80
    batch_size: int = 128

    def __post_init__(self):
        _require(type(self.epochs) is int and 1 <= self.epochs <= 80, "epochs must be an integer in [1,80]")
        _require(type(self.batch_size) is int and 1 <= self.batch_size <= 128,
                 "batch_size must be an integer in [1,128]")


DEFAULT_RECIPE = Recipe()


class LearningFailure(RuntimeError):
    """Original failure and truthful attempted/returned work, without retry."""
    def __init__(self, original, progress):
        super().__init__(f"{type(original).__name__}: {original}")
        self.original, self.progress = original, progress


def _bundle(bundle, rows):
    TARGET_VERSION = VERSION

    _require(isinstance(bundle, dict) and set(bundle) == {"version", *ARMS}
             and bundle["version"] == TARGET_VERSION, "qualified two-arm target bundle required")
    result = {"version": TARGET_VERSION}
    fields = {"kind", "centered_float64", "scale", "scaled_float32",
              "weights_float64", "weights_float32", "allowed"}
    for arm in ARMS:
        source = bundle[arm]
        _require(isinstance(source, dict) and set(source) == fields and source["kind"] == arm,
                 "exact target-arm schema required")
        expected = {"centered_float64": (np.float64, (rows, 4)),
                    "scaled_float32": (np.float32, (rows, 4)),
                    "weights_float64": (np.float64, (rows,)),
                    "weights_float32": (np.float32, (rows,)), "allowed": (np.bool_, (rows, 4))}
        for key, (dtype, shape) in expected.items():
            value = source[key]
            _require(isinstance(value, np.ndarray) and value.dtype == dtype and value.shape == shape
                     and np.isfinite(value).all(), f"invalid {key}")
        mask = source["allowed"]
        _require(mask.any(axis=1).all() and (source["centered_float64"][~mask] == 0).all()
                 and (source["scaled_float32"][~mask] == 0).all(), "nonempty masks with zero blocked targets")
        scale = source["scale"]
        _require(type(scale) in (int, float) and math.isfinite(scale) and scale >= 1e-8,
                 "finite qualified target scale required")
        weights = source["weights_float64"]
        _require((weights > 0).all() and abs(float(weights.mean()) - 1.0) <= 1e-12
                 and np.array_equal(weights.astype(np.float32), source["weights_float32"])
                 and (source["weights_float32"] > 0).all(), "positive common mean-one row weights required")
        _require(np.array_equal((source["centered_float64"] / scale).astype(np.float32), source["scaled_float32"]),
                 "original targets, scale and float32 training values must agree")
        result[arm] = {"kind": arm, "scale": float(scale),
                       **{key: _owned(source[key]) for key in expected}}
    _require(all(np.array_equal(result["r16"][key], result["r64"][key])
                 for key in ("allowed", "weights_float64", "weights_float32")),
             "both arms require exactly the same masks and row weights")
    _require(result["r16"]["scale"].hex() == result["r64"]["scale"].hex(),
             "both precision arms require the original shared R16 scale")
    return result


def _snapshot(model, head):
    exported = head.export_head(model)
    _require(head.validate_head(exported) == "dense_augmented", "ordinary dense head required")
    return {key: _owned(value) if isinstance(value, np.ndarray) else value for key, value in exported.items()}


def _same_weights(left, right):
    return set(left) == set(right) and all(
        (isinstance(right[k], np.ndarray) and value.dtype == right[k].dtype
         and value.shape == right[k].shape and value.tobytes() == right[k].tobytes())
        if isinstance(value, np.ndarray) else value == right[k] for k, value in left.items())


def _descriptor(value):
    _require(isinstance(value, dict) and set(value) == {"path", "sha256", "bytes"}
             and type(value["path"]) is str and value["path"]
             and type(value["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
             and type(value["bytes"]) is int and value["bytes"] > 0, "durable checkpoint descriptor required")
    return dict(value)


def train_pair(features, target_bundle, seed, *, check, emit, checkpoint, recipe=DEFAULT_RECIPE):
    """Train both matched targets with fresh identical heads and epoch orders.

    Production: 558 finite float32 TRAIN feature rows; seeds 20101..20103; 80
    epochs, batch128, Adam3e-4, clip5. Explicit smaller Recipe values permit only
    fabricated engineering fixtures and are recorded in the return value.

    check() runs before and after every counted operation. emit(event,payload)
    receives attempt/return events, targets, initial_pair, fit_start/fit_end,
    order (immutable int64 array and SHA), epoch, parity and failure records.
    Hooks own timing and durable journals. Atomic optimizer_update includes
    eight-view forward, weighted row-mean loss, backward, clipping and Adam step.
    Epoch weighted_mse is the row-count-weighted mean of pre-update minibatch
    losses, not a new final-model evaluation. No batch weight renormalization.

    checkpoint(arm,seed,phase,exported) must durably write the immutable arrays,
    verify saved array bytes against the export and return {path,sha256,bytes}.
    Saved-file equality is inherited from that caller hook, not reloaded here.
    Both initial checkpoints are compared
    bytewise and published before either arm updates. Only final trained weights
    are exported, with publication preceding first16 TRAIN NumPy/Torch parity.
    Parity compares raw four-cost single-view predictions at atol=rtol=2e-5;
    it is a diagnostic numerical check, not action or autonomous qualification.
    No callback return except the checkpoint descriptor affects optimization.
    A failure retains pending/returned operations and never retries or resumes.
    Returned/completed counters acknowledge successfully emitted return records.
    A pending operation may have executed if its return publication failed;
    no counter claims rollback or authorizes resuming that attempt.
    """
    _require(isinstance(recipe, Recipe) and type(seed) is int and seed in SEEDS, "fixed recipe and paired seed required")
    _require(all(callable(callback) for callback in (check, emit, checkpoint)), "all caller hooks required")
    _require(isinstance(features, np.ndarray) and features.dtype == np.float32 and features.ndim == 2
             and features.shape[1] == INPUT_DIM and len(features) > 0 and np.isfinite(features).all(),
             "finite float32[N,2836] common TRAIN features required")
    _require(recipe != DEFAULT_RECIPE or len(features) == TRAIN_ROWS, "production requires exactly 558 rows")
    targets = _bundle(target_bundle, len(features))
    progress = {"calls": {channel: {"attempted": 0, "returned": 0} for channel in CHANNELS},
                "completed_updates": 0, "completed_fits": 0, "pending": {}}
    operation_id = 0

    def state():
        return {"seed": seed, "completed_updates": progress["completed_updates"],
                "completed_fits": progress["completed_fits"],
                "calls": {k: dict(v) for k, v in progress["calls"].items()},
                "pending": {k: dict(v) for k, v in progress["pending"].items()}}

    def call(channel, operation, **context):
        nonlocal operation_id
        check()
        record = {"operation_id": operation_id, "channel": channel, "seed": seed, **context}
        operation_id += 1
        progress["calls"][channel]["attempted"] += 1
        progress["pending"][record["operation_id"]] = dict(record)
        emit("attempt", dict(record))
        value = operation()
        emit("return", {**record, "result": dict(value) if channel == "optimizer_update" else None})
        progress["calls"][channel]["returned"] += 1
        del progress["pending"][record["operation_id"]]
        if channel == "optimizer_update":
            progress["completed_updates"] += 1
        check()
        return value

    def publish(arm, phase, exported):
        return call("checkpoint_publication", lambda: _descriptor(checkpoint(arm, seed, phase, dict(exported))),
                    arm=arm, phase=phase)

    try:
        import torch

        from openjev.research import otto_symmetry_head as head
        from openjev.research.otto_cost_regression import training_losses

        models, initial, initial_files = {}, {}, {}
        for arm in ARMS:
            emit("targets", {"seed": seed, "arm": arm, **targets[arm]})
            models[arm] = call("model_initialization", lambda: head.make_head("dense_augmented", seed), arm=arm)
            initial[arm] = call("checkpoint_export", lambda arm=arm: _snapshot(models[arm], head), arm=arm, phase="initial")
        _require(_same_weights(initial["r16"], initial["r64"]), "paired initial weights differ")
        for arm in ARMS:
            initial_files[arm] = publish(arm, "initial", initial[arm])
        pairing = {"seed": seed, "identical_weights": True, "initial_checkpoints": initial_files}
        emit("initial_pair", pairing)
        tx = torch.from_numpy(np.array(features, copy=True, order="C"))
        fits = []
        for arm in ARMS:
            emit("fit_start", {"seed": seed, "arm": arm, "rows": len(features)})
            model = models[arm]
            optimizer = call("optimizer_initialization", lambda model=model: torch.optim.Adam(model.parameters(), lr=LEARNING_RATE), arm=arm)
            _require(not optimizer.state, "fresh Adam state required")
            ty = torch.from_numpy(np.array(targets[arm]["scaled_float32"], copy=True))
            tw = torch.from_numpy(np.array(targets[arm]["weights_float32"], copy=True))
            tm = torch.from_numpy(np.array(targets[arm]["allowed"], copy=True))
            rng = np.random.default_rng(seed + 20000)
            epochs = []
            model.train()
            for epoch in range(1, recipe.epochs + 1):
                order = rng.permutation(len(features)).astype(np.int64, copy=False)
                order_sha = hashlib.sha256(order.tobytes(order="C")).hexdigest()
                emit("order", {"seed": seed, "arm": arm, "epoch": epoch,
                               "order": _owned(order), "sha256": order_sha})
                losses, norms = [], []
                for offset in range(0, len(features), recipe.batch_size):
                    indices = order[offset:offset + recipe.batch_size]

                    def update(indices=indices, optimizer=optimizer, model=model, ty=ty, tm=tm, tw=tw):
                        optimizer.zero_grad(set_to_none=True)
                        loss = (training_losses(model, tx[indices], ty[indices], tm[indices]) * tw[indices]).mean()
                        _require(bool(torch.isfinite(loss)), "finite weighted regression loss required")
                        loss.backward()
                        _require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                                 "finite present gradients required")
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True)
                        optimizer.step()
                        _require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "finite post-update weights required")
                        return {"loss": float(loss.detach()), "gradient_norm_before_clip": float(norm), "rows": len(indices)}

                    result = call("optimizer_update", update, arm=arm, epoch=epoch,
                                  batch=offset // recipe.batch_size, offset=offset, rows=len(indices))
                    losses.append(result["loss"] * len(indices))
                    norms.append(result["gradient_norm_before_clip"])
                record = {"seed": seed, "arm": arm, "epoch": epoch, "rows": len(features),
                          "updates": len(norms), "weighted_mse": math.fsum(losses) / len(features),
                          "order_sha256": order_sha, "gradient_norm_mean": math.fsum(norms) / len(norms),
                          "gradient_norm_max": max(norms)}
                epochs.append(record)
                emit("epoch", dict(record))
            expected = recipe.epochs * math.ceil(len(features) / recipe.batch_size)
            steps = {name: int(optimizer.state[parameter]["step"].item()) for name, parameter in model.named_parameters()}
            _require(steps and all(value == expected for value in steps.values()), "complete Adam step witness required")
            final = call("checkpoint_export", lambda model=model: _snapshot(model, head), arm=arm, phase="final")
            final_file = publish(arm, "final", final)
            frozen = call("inference_setup", lambda final=final: head.FrozenHead(final), arm=arm)
            count = min(16, len(features))
            probe = np.array(features[:count], copy=True, order="C")
            numpy_values = call("parity_numpy", lambda frozen=frozen, probe=probe: frozen.scores(probe), arm=arm, rows=count)
            model.eval()

            def torch_prediction(model=model, probe=probe):
                with torch.no_grad():
                    return model(torch.from_numpy(probe.copy())).numpy().copy()

            torch_values = call("parity_torch", torch_prediction, arm=arm, rows=count)
            passed = bool(np.isfinite(numpy_values).all() and np.isfinite(torch_values).all()
                          and np.allclose(numpy_values, torch_values, atol=PARITY_ATOL, rtol=PARITY_RTOL))
            parity = {"seed": seed, "arm": arm, "rows": count, "checkpoint": dict(final_file),
                      "features_sha256": hashlib.sha256(probe.tobytes()).hexdigest(),
                      "numpy": _owned(numpy_values), "torch": _owned(torch_values),
                      "maximum_difference": float(np.max(np.abs(numpy_values.astype(np.float64) - torch_values))),
                      "absolute_tolerance": PARITY_ATOL, "relative_tolerance": PARITY_RTOL, "passed": passed}
            emit("parity", parity)
            _require(passed, "final exported-weight NumPy/Torch parity failed")
            fit = {"arm": arm, "seed": seed, "rows": len(features), "epochs": tuple(epochs),
                   "initial_checkpoint": dict(initial_files[arm]), "final_checkpoint": dict(final_file),
                   "final_export": final, "optimizer_steps": steps, "parity": parity}
            fits.append(fit)
            progress["completed_fits"] += 1
            emit("fit_end", {"seed": seed, "arm": arm, "updates": expected,
                             "final_checkpoint": dict(final_file), "parity_passed": True})
        return {"version": VERSION, "seed": seed, "recipe": {"epochs": recipe.epochs, "batch_size": recipe.batch_size,
                "learning_rate": LEARNING_RATE, "gradient_clip": GRADIENT_CLIP}, "targets": targets,
                "initial_pair": pairing, "fits": tuple(fits), "progress": state()}
    except BaseException as error:
        failure = LearningFailure(error, state())
        try:
            emit("failure", {"error": repr(error), "progress": state()})
        except BaseException as hook_error:  # noqa: BLE001 - Preserve the original failure.
            failure.add_note(f"Failure hook also failed: {hook_error!r}")
        raise failure from error
