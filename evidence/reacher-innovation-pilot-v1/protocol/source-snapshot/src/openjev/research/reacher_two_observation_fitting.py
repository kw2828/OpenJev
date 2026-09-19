"""Exclusive fit serialization for the reviewed two-observation trainer.

Caller supplies authenticated original tensors, complete sealed orders, public
training data and every provenance binding. No initialization/order generation,
protocol, study admission, scientific stream allocation or automatic retry is
owned here. An enclosing runner must authenticate those inputs before calling.

Exactly five files define a successful fit. Every trainer log/checkpoint field
is retained unchanged. Failures preserve the actual partial trainer state when
exportable, including an optimizer step completed before a subsequent guard or
I/O failure. The outer failed receipt remains terminal even if that partial
checkpoint itself describes a technically resumable trainer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import torch

from openjev.research import reacher_two_observation_training as training
from openjev.research.reacher_objective_training import canonical_state_hash, canonical_tensor_hash

VERSION = "reacher-two-observation-fitting-v1"
PAYLOAD_FILES = frozenset({"initial-weights.pt", "training.jsonl", "weights.pt", "checkpoint.pt"})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _cap(deadline):
    _require(type(deadline) in (int, float) and not math.isnan(deadline), "Explicit monotonic deadline required")
    if time.monotonic() >= deadline:
        raise TimeoutError("Cooperative two-observation fit cap")


def _save_torch(path, payload):
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sha(path, deadline):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            _cap(deadline)
            digest.update(chunk)
    return digest.hexdigest()


def _row(row, *, update, settings, orders, previous):
    _require(isinstance(row, dict), "Complete trainer log dict required")
    epoch, batch = divmod(update - 1, settings.batches_per_epoch)
    indices = orders["orders"][epoch, batch * settings.batch_size:(batch + 1) * settings.batch_size]
    _require(type(row.get("update")) is int and row["update"] == update
             and type(row.get("optimizer_steps")) is int and row["optimizer_steps"] == update
             and row.get("kind") == training.KIND and row.get("epoch") == epoch and row.get("batch") == batch,
             "Trainer log update/cursor mismatch")
    _require(row.get("indices") == indices.tolist()
             and row.get("indices_sha256") == canonical_tensor_hash({"indices": indices}),
             "Trainer log does not match externally sealed original order")
    _require(row.get("previous_log_sha256") == previous, "Broken trainer log chain")
    body = {key: value for key, value in row.items() if key != "log_sha256"}
    _require(row.get("log_sha256") == canonical_state_hash(body), "Invalid complete trainer log digest")
    # No allowlist/filter: new work, RNG, gradient and timing fields are retained.
    return json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"


def _checkpoint(payload, trainer, *, settings, initial_hash, order_hash, data_hash,
                provenance, sources, runtime, updates, chain, final_weights):
    _require(isinstance(payload, dict) and payload.get("integrity_sha256") == canonical_state_hash(
        {key: value for key, value in payload.items() if key != "integrity_sha256"}), "Full checkpoint seal")
    _require(payload.get("version") == training.VERSION and payload.get("kind") == training.KIND
             and payload.get("settings") == settings.configuration()
             and payload.get("model_configuration") == trainer.model_configuration
             and trainer.model_configuration.get("model_class") == "TwoObservationHistoryGRUWorldModel",
             "Actual checkpoint class/settings binding")
    _require(canonical_tensor_hash(payload["initialization"]["weights"]) == initial_hash
             and payload["initialization"]["tensor_sha256"] == initial_hash
             and canonical_state_hash(payload["orders"]) == order_hash and payload["data_sha256"] == data_hash,
             "Checkpoint original initialization/full sealed order/data binding")
    _require(canonical_state_hash(payload["provenance"]) == canonical_state_hash(provenance)
             and canonical_state_hash(payload["source_sha256"]) == canonical_state_hash(sources)
             and canonical_state_hash(payload["runtime"]) == canonical_state_hash(runtime),
             "Checkpoint external provenance/source/runtime binding")
    _require(payload.get("failed") is False and payload.get("failure") is None
             and payload.get("successful_updates") == payload.get("optimizer_steps") == updates
             and payload.get("cursor") == {"epoch": settings.epochs, "batch": 0}
             and payload.get("log_chain_sha256") == chain == trainer.log_chain_sha256,
             "Complete trainer counters/cursor/log chain required")
    _require(canonical_tensor_hash(payload["student_state"]) == canonical_tensor_hash(final_weights),
             "Final deployment tensors differ from resumable checkpoint")


def fit_one(initial_weights, orders, public_data, out, *, settings,
            expected_initial_sha256, expected_orders_sha256, expected_data_sha256,
            provenance, source_sha256, runtime, pair, name, deadline, progress=None):
    """Serialize one complete fresh fit with externally supplied identities.

    ``expected_orders_sha256`` is canonical_state_hash(complete sealed payload),
    NOT its embedded integrity_sha256. Returned receipt equals completed.json.
    ``wall_seconds`` includes construction, all updates, flushing/fsync, payload
    exports and hashes, excluding only that receipt's own write. The cap is also
    checked AFTER its write. An over-cap/failed completion marker is renamed to
    invalid-completion.json before failed.json, never left as completed.json.
    """
    begin = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    trainer, checkpoint, active_row = None, None, None
    phase, flushed, log_chain = "validation", 0, canonical_state_hash([])
    constructor_seconds = update_seconds = log_io_seconds = payload_io_seconds = hash_seconds = 0.0
    active_update_seconds = 0.0

    def phase_set(value):
        nonlocal phase
        phase = value
        if progress is not None:
            progress.update(phase=value, name=name, pair=pair, flushed_updates=flushed,
                successful_updates=0 if trainer is None else trainer.successful_updates,
                optimizer_steps=0 if trainer is None else trainer.optimizer_steps)
            if trainer is not None:
                progress.update(trainer.cursor)

    try:
        _cap(deadline)
        _require(type(settings) is training.TrainingSettings, "Exact two-observation TrainingSettings required")
        _require(progress is None or type(progress) is dict, "Progress dict or None required")
        _require(all(type(value) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)
                     for value in (pair, name)), "Explicit safe pair and fit names required")
        _require(canonical_tensor_hash(initial_weights) == expected_initial_sha256, "Externally bound initial tensors")
        _require(canonical_state_hash(orders) == expected_orders_sha256, "Externally bound full sealed orders")
        _require(canonical_tensor_hash(public_data) == expected_data_sha256, "Externally bound public data")
        phase_set("constructor")
        tick = time.perf_counter()
        try:
            trainer = training.TwoObservationTrainer(initial_weights, orders, public_data, settings=settings,
                expected_initial_sha256=expected_initial_sha256, expected_orders_sha256=expected_orders_sha256,
                expected_data_sha256=expected_data_sha256, provenance=provenance, source_sha256=source_sha256,
                runtime=runtime)
        finally:
            constructor_seconds = time.perf_counter() - tick
        _cap(deadline)
        phase_set("initial_payload")
        tick = time.perf_counter()
        try:
            _require(canonical_tensor_hash(trainer.student.state_dict()) == expected_initial_sha256,
                     "Constructed student differs from original paired tensors")
            _save_torch(out / "initial-weights.pt", trainer.student.state_dict())
        finally:
            payload_io_seconds += time.perf_counter() - tick
        expected_updates = settings.epochs * settings.batches_per_epoch
        with (out / "training.jsonl").open("x") as log:
            for update in range(1, expected_updates + 1):
                _cap(deadline)
                phase_set("update")
                active_row = None
                tick = time.perf_counter()
                try:
                    active_row = trainer.train_next(deadline_check=lambda: _cap(deadline))
                finally:
                    active_update_seconds = time.perf_counter() - tick
                    update_seconds += active_update_seconds
                phase_set("log_flush")
                tick = time.perf_counter()
                try:
                    serialized = _row(active_row, update=update, settings=settings, orders=orders, previous=log_chain)
                    _require(active_row["log_sha256"] == trainer.log_chain_sha256, "Returned row differs from live log chain")
                    log.write(serialized)
                    log.flush()
                    os.fsync(log.fileno())
                    flushed += 1
                    log_chain = active_row["log_sha256"]
                finally:
                    log_io_seconds += time.perf_counter() - tick
                _cap(deadline)
        phase_set("final_payload")
        _require(trainer.successful_updates == trainer.optimizer_steps == flushed == expected_updates,
                 "Complete successful/optimizer/flushed update counts required")
        tick = time.perf_counter()
        try:
            final_weights = trainer.student.state_dict()
            checkpoint = trainer.export_checkpoint()
            _checkpoint(checkpoint, trainer, settings=settings, initial_hash=expected_initial_sha256,
                        order_hash=expected_orders_sha256, data_hash=expected_data_sha256, provenance=provenance,
                        sources=source_sha256, runtime=runtime, updates=expected_updates, chain=log_chain,
                        final_weights=final_weights)
            _save_torch(out / "weights.pt", final_weights)
            _save_torch(out / "checkpoint.pt", checkpoint)
        finally:
            payload_io_seconds += time.perf_counter() - tick
        _cap(deadline)
        phase_set("manifest")
        _require({str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()} == PAYLOAD_FILES,
                 "Exact four fit payload files required before completion")
        tick = time.perf_counter()
        try:
            files = {path: _sha(out / path, deadline) for path in sorted(PAYLOAD_FILES)}
        finally:
            hash_seconds = time.perf_counter() - tick
        receipt = {"version": VERSION, "status": "completed", "name": name, "pair": pair, "arm": training.KIND,
            "model_configuration": trainer.model_configuration, "settings": settings.configuration(),
            "provenance": provenance, "source_sha256": source_sha256, "runtime": runtime,
            "updates": trainer.successful_updates, "optimizer_steps": trainer.optimizer_steps,
            "flushed_updates": flushed, "cursor": trainer.cursor,
            "initialization_sha256": expected_initial_sha256, "orders_sha256": expected_orders_sha256,
            "orders_hash_scope": "canonical_state_hash of entire sealed payload including integrity field",
            "data_sha256": expected_data_sha256, "student_tensor_sha256": canonical_tensor_hash(final_weights),
            "checkpoint_integrity_sha256": checkpoint["integrity_sha256"], "log_chain_sha256": log_chain,
            "parameters": sum(p.numel() for p in trainer.student.parameters()),
            "constructor_seconds": constructor_seconds, "trainer_setup_seconds": trainer.setup_wall_seconds,
            "training_wall_seconds": trainer.training_wall_seconds, "update_call_seconds": update_seconds,
            "log_validation_flush_seconds": log_io_seconds, "payload_export_write_seconds": payload_io_seconds,
            "file_hash_seconds": hash_seconds, "wall_seconds": time.perf_counter() - begin, "files": files,
            "timing_scope": "Full fit through file hashes, excluding only completed.json write; nested timing scopes are not additive. Cap also checked after terminal write.",
            "scope": "One externally authenticated fresh fit; not a whole-study audit or authorization."}
        _cap(deadline)
        phase_set("completion_write")
        _json(out / "completed.json", receipt)
        _cap(deadline)
        phase_set("completed")
        return receipt
    except BaseException as error:
        if (out / "completed.json").exists():
            try:
                (out / "completed.json").rename(out / "invalid-completion.json")
            except BaseException as rename_error:  # noqa: BLE001
                error.add_note(f"Completion invalidation failed: {type(rename_error).__name__}: {rename_error}")
        partial_error = None
        if trainer is not None:
            try:
                _save_torch(out / "partial-checkpoint.pt", trainer.export_checkpoint())
            except BaseException as export_error:  # noqa: BLE001
                partial_error = {"type": type(export_error).__name__, "message": str(export_error)}
                error.add_note(f"Partial checkpoint failed: {type(export_error).__name__}: {export_error}")
        # Preserve a returned but not durably flushed row separately, without
        # inventing a log entry for a trainer call that never returned one.
        if active_row is not None and active_row.get("update", 0) > flushed:
            try:
                _json(out / "unflushed-update.json", active_row)
            except BaseException as row_error:  # noqa: BLE001
                error.add_note(f"Unflushed row preservation failed: {type(row_error).__name__}: {row_error}")
        failure = {"version": VERSION, "status": "failed", "phase": phase, "name": name, "pair": pair,
            "exception_type": type(error).__name__, "message": str(error), "notes": list(getattr(error, "__notes__", [])),
            "successful_updates": 0 if trainer is None else trainer.successful_updates,
            "optimizer_steps": 0 if trainer is None else trainer.optimizer_steps, "flushed_updates": flushed,
            "cursor": None if trainer is None else trainer.cursor,
            "trainer_failed": None if trainer is None else trainer.failed,
            "trainer_failure": None if trainer is None else trainer.failure,
            "last_attempt": None if trainer is None else trainer.last_attempt,
            "partial_checkpoint_error": partial_error, "log_chain_of_flushed_rows": log_chain,
            "constructor_seconds": constructor_seconds, "update_call_seconds": update_seconds,
            "active_update_seconds": active_update_seconds, "log_validation_flush_seconds": log_io_seconds,
            "payload_export_write_seconds": payload_io_seconds, "file_hash_seconds": hash_seconds,
            "wall_seconds": time.perf_counter() - begin,
            "bindings": {"initialization_sha256": expected_initial_sha256, "orders_sha256": expected_orders_sha256,
                         "data_sha256": expected_data_sha256},
            "scope": "Terminal fit failure; partial trainer state does not authorize resuming this attempt."}
        try:
            _json(out / "failed.json", failure)
        except BaseException as receipt_error:  # noqa: BLE001
            error.add_note(f"Failure receipt failed: {type(receipt_error).__name__}: {receipt_error}")
        raise
