"""Matched short/long optimization on byte-identical original R64 targets.

No data or checkpoint files are read or written here. The caller authenticates
the common cohort, supplies authenticated cached targets and owns durable
records, resource checks and complete elapsed-time accounting. There is no
validation, early stopping, checkpoint selection, simulator or rollout sampler.

The update body is inherited from otto_teacher_precision.py at SHA256
e8cebaafb105433c1b916ad2ffaa0da9fba6492e256854745c240a4a28fa8a6d.
The long arm continues its same optimizer after an exact short-prefix witness.
No label transformation or unrecorded TRAIN inference is introduced.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

import numpy as np

VERSION = "otto-training-budget-v1"
ARMS = ("short", "long")
SEEDS = (30101, 30102, 30103)
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


def build_training_targets(reference_r64, *, expected):
    """Own identical copies of an authenticated R64 target bundle.

    expected is {'scale': float, 'arrays': {array_name: sha256}} obtained by
    the caller from its authenticated original R64 cache and metadata. No raw
    costs, label construction, normalization, sampling or confidence filtering
    occurs here. The caller separately binds common feature and row bytes.
    """
    fields = {"kind", "centered_float64", "scale", "scaled_float32",
              "weights_float64", "weights_float32", "allowed"}
    _require(isinstance(reference_r64, dict) and set(reference_r64) == fields
             and reference_r64["kind"] == "r64", "exact original R64 target reference required")
    values = reference_r64["centered_float64"]
    _require(isinstance(values, np.ndarray) and values.ndim == 2 and len(values) > 0,
             "nonempty original R64 array required")
    bundle = {"version": VERSION, "reference": expected}
    for arm in ARMS:
        bundle[arm] = {**reference_r64, "kind": arm}
    return _bundle(bundle, len(values))


@dataclass(frozen=True, slots=True)
class Recipe:
    """Production defaults; smaller explicit recipes are for fabricated tests only."""
    short_epochs: int = 80
    long_epochs: int = 320
    batch_size: int = 128

    def __post_init__(self):
        _require(type(self.short_epochs) is int and 1 <= self.short_epochs <= 80,
                 "short_epochs must be an integer in [1,80]")
        _require(type(self.long_epochs) is int and self.short_epochs < self.long_epochs <= 320,
                 "long_epochs must exceed short_epochs and be at most 320")
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

    _require(isinstance(bundle, dict) and set(bundle) == {"version", "reference", *ARMS}
             and bundle["version"] == TARGET_VERSION, "qualified two-arm target bundle required")
    reference = bundle["reference"]
    array_keys = {"centered_float64", "scaled_float32", "weights_float64", "weights_float32", "allowed"}
    _require(isinstance(reference, dict) and set(reference) == {"scale", "arrays"}
             and type(reference["scale"]) in (int, float) and math.isfinite(reference["scale"])
             and reference["scale"] >= 1e-8 and isinstance(reference["arrays"], dict)
             and set(reference["arrays"]) == array_keys
             and all(type(pin) is str and re.fullmatch(r"[0-9a-f]{64}", pin) for pin in reference["arrays"].values()),
             "original R64 scale and exact array hash descriptor required")
    result = {"version": TARGET_VERSION,
              "reference": {"scale": float(reference["scale"]), "arrays": dict(reference["arrays"])}}
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
            _require(hashlib.sha256(value.tobytes()).hexdigest() == reference["arrays"][key],
                     f"original R64 {key} bytes differ")
        mask = source["allowed"]
        _require(mask.any(axis=1).all() and (source["centered_float64"][~mask] == 0).all()
                 and (source["scaled_float32"][~mask] == 0).all(), "nonempty masks with zero blocked targets")
        scale = source["scale"]
        _require(type(scale) in (int, float) and math.isfinite(scale) and scale >= 1e-8,
                 "finite qualified target scale required")
        _require(float(scale).hex() == float(reference["scale"]).hex(), "original R64 scale differs")
        weights = source["weights_float64"]
        _require((weights > 0).all() and abs(float(weights.mean()) - 1.0) <= 1e-12
                 and np.array_equal(weights.astype(np.float32), source["weights_float32"])
                 and (source["weights_float32"] > 0).all(), "positive common mean-one row weights required")
        _require(np.array_equal((source["centered_float64"] / scale).astype(np.float32), source["scaled_float32"]),
                 "original targets, scale and float32 training values must agree")
        result[arm] = {"kind": arm, "scale": float(scale),
                       **{key: _owned(source[key]) for key in expected}}
    _require(all(result["short"][key].tobytes() == result["long"][key].tobytes() for key in array_keys),
             "both arms require byte-identical original targets, masks and weights")
    _require(result["short"]["scale"].hex() == result["long"]["scale"].hex(),
             "both training budgets require the original shared R64 scale")
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

    Production: 558 finite float32 TRAIN feature rows; seeds 30101..30103; 80
    short epochs and 320 long epochs, batch128, Adam3e-4, clip5. Smaller recipes permit only
    fabricated engineering fixtures and are recorded in the return value.

    check() runs before and after every counted operation. emit(event,payload)
    receives attempt/return events, targets, initial_pair, fit_start/fit_end,
    order (immutable int64 array and SHA), epoch, matched_prefix, parity and failure records.
    Hooks own timing and durable journals. Atomic optimizer_update includes
    eight-view forward, weighted row-mean loss, backward, clipping and Adam step.
    Epoch weighted_mse is the row-count-weighted mean of pre-update minibatch
    losses, not a new final-model evaluation. No batch weight renormalization.

    checkpoint(arm,seed,phase,exported) must durably write the immutable arrays,
    verify saved array bytes against the export and return {path,sha256,bytes}.
    Saved-file equality is inherited from that caller hook, not reloaded here.
    Both initial checkpoints are compared bytewise and published before updates.
    The long arm publishes a 'prefix' checkpoint at short_epochs. Its weights,
    all prior row-order bytes, batch loss/norm records and epoch summaries must
    exactly match the short arm before matched_prefix is emitted. The same long
    optimizer then continues. This prefix is diagnostic, never selected for use.
    Each final checkpoint precedes first16 TRAIN NumPy/Torch parity.
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
        _require(_same_weights(initial["short"], initial["long"]), "paired initial weights differ")
        for arm in ARMS:
            initial_files[arm] = publish(arm, "initial", initial[arm])
        pairing = {"seed": seed, "identical_weights": True, "initial_checkpoints": initial_files}
        emit("initial_pair", pairing)
        tx = torch.from_numpy(np.array(features, copy=True, order="C"))
        fits = []
        short_trace = matched_prefix = None
        for arm in ARMS:
            arm_epochs = recipe.short_epochs if arm == "short" else recipe.long_epochs
            emit("fit_start", {"seed": seed, "arm": arm, "rows": len(features)})
            model = models[arm]
            optimizer = call("optimizer_initialization", lambda model=model: torch.optim.Adam(model.parameters(), lr=LEARNING_RATE), arm=arm)
            _require(not optimizer.state, "fresh Adam state required")
            ty = torch.from_numpy(np.array(targets[arm]["scaled_float32"], copy=True))
            tw = torch.from_numpy(np.array(targets[arm]["weights_float32"], copy=True))
            tm = torch.from_numpy(np.array(targets[arm]["allowed"], copy=True))
            rng = np.random.default_rng(seed + 20000)
            epochs, orders, batches = [], [], []
            model.train()
            for epoch in range(1, arm_epochs + 1):
                order = rng.permutation(len(features)).astype(np.int64, copy=False)
                order_sha = hashlib.sha256(order.tobytes(order="C")).hexdigest()
                orders.append(order.tobytes(order="C"))
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
                    batches.append({"epoch": epoch, "batch": offset // recipe.batch_size,
                                    "offset": offset, **result})
                    losses.append(result["loss"] * len(indices))
                    norms.append(result["gradient_norm_before_clip"])
                record = {"seed": seed, "arm": arm, "epoch": epoch, "rows": len(features),
                          "updates": len(norms), "weighted_mse": math.fsum(losses) / len(features),
                          "order_sha256": order_sha, "gradient_norm_mean": math.fsum(norms) / len(norms),
                          "gradient_norm_max": max(norms)}
                epochs.append(record)
                emit("epoch", dict(record))
                if arm == "long" and epoch == recipe.short_epochs:
                    _require(short_trace is not None and orders == short_trace["orders"]
                             and batches == short_trace["batches"]
                             and [{k: v for k, v in row.items() if k != "arm"} for row in epochs]
                             == short_trace["epochs"], "long prefix order/loss/norm records differ from short fit")
                    prefix = call("checkpoint_export", lambda model=model: _snapshot(model, head),
                                  arm=arm, phase="prefix")
                    _require(_same_weights(prefix, short_trace["final"]), "long prefix weights differ from short final")
                    prefix_file = publish(arm, "prefix", prefix)
                    prefix_updates = recipe.short_epochs * math.ceil(len(features) / recipe.batch_size)
                    prefix_steps = {name: int(optimizer.state[p]["step"].item()) for name, p in model.named_parameters()}
                    _require(prefix_steps and all(value == prefix_updates for value in prefix_steps.values()),
                             "long prefix Adam steps differ from short budget")
                    witness = {"seed": seed, "arm": arm, "epoch": epoch, "updates": prefix_updates,
                               "checkpoint": dict(prefix_file), "short_final_checkpoint": dict(short_trace["file"]),
                               "identical_weights": True, "identical_order_records": True,
                               "identical_update_records": True, "identical_epoch_records": True,
                               "compared_epochs": len(epochs), "compared_updates": len(batches),
                               "optimizer_steps": prefix_steps}
                    emit("matched_prefix", dict(witness))
                    matched_prefix = witness
            expected = arm_epochs * math.ceil(len(features) / recipe.batch_size)
            steps = {name: int(optimizer.state[parameter]["step"].item()) for name, parameter in model.named_parameters()}
            _require(steps and all(value == expected for value in steps.values()), "complete Adam step witness required")
            final = call("checkpoint_export", lambda model=model: _snapshot(model, head), arm=arm, phase="final")
            final_file = publish(arm, "final", final)
            if arm == "short":
                short_trace = {"orders": orders, "batches": batches,
                               "epochs": [{k: v for k, v in row.items() if k != "arm"} for row in epochs],
                               "final": final, "file": dict(final_file)}
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
        _require(matched_prefix is not None, "matched long prefix must complete before return")
        return {"version": VERSION, "seed": seed,
                "recipe": {"short_epochs": recipe.short_epochs, "long_epochs": recipe.long_epochs, "batch_size": recipe.batch_size,
                "learning_rate": LEARNING_RATE, "gradient_clip": GRADIENT_CLIP}, "targets": targets,
                "initial_pair": pairing, "matched_prefix": matched_prefix, "fits": tuple(fits), "progress": state()}
    except BaseException as error:
        failure = LearningFailure(error, state())
        try:
            emit("failure", {"error": repr(error), "progress": state()})
        except BaseException as hook_error:  # noqa: BLE001 - Preserve the original failure.
            failure.add_note(f"Failure hook also failed: {hook_error!r}")
        raise failure from error
