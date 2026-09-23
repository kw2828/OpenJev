"""Fabricated audit orchestration and work accounting; no empirical reads."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openjev.research import otto_query_memory as memory
from openjev.research import otto_query_memory_model as composed

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_test_independent_query_memory_audit", ROOT / "scripts/audit_otto_query_memory.py")
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def synthetic_chunk(lengths, start):
    """Physical batch and 32-step padding match canonical saved inference."""
    features = torch.full((len(lengths), 32, 31), float("nan"))
    scores = torch.full((len(lengths), 32, 4), float("nan"))
    mask = torch.zeros(len(lengths), 32, dtype=torch.bool)
    local = [max(0, min(32, length - start)) for length in lengths]
    for lane, size in enumerate(local):
        steps = torch.arange(start, start + size)
        features[lane, :size] = torch.sin(
            steps[:, None] * .1 + torch.arange(31)[None, :] * .03 + lane)
        features[lane, :size, 15] = steps / 2188
        features[lane, :size, 16] = (steps % 4) / 2188
        features[lane, :size, 17] = 1
        mask[lane, :size] = steps % 4 == 0
        query_steps = steps[steps % 4 == 0]
        scores[lane, :size][mask[lane, :size]] = (
            torch.tensor([4., 7., 2., 6.]) + query_steps[:, None] * torch.tensor([.125, -.25, .5, .25]))
    return {"features": features, "query_scores": scores, "query_mask": mask,
            "lengths": torch.tensor(local, dtype=torch.int64),
            "episode_ends": torch.tensor([size > 0 and start + size == length
                                          for size, length in zip(local, lengths, strict=True)])}


@pytest.mark.parametrize("view", audit.VIEWS)
def test_independent_schedule_matches_qualified_model_with_mixed_lengths_and_padded_tail(view):
    # Two physical batches, including a partial batch and lanes ending at a
    # first query, before/at the next query, and beyond a chunk boundary.
    lengths = (1, 4, 5, 33, 65, 32, 65)
    mode = "none" if view in ("pretrained", "joint_aux") else (
        "trace_delta" if view == "trace_no_write" else view)
    model = composed.make_model(memory.Config(mode, key_dim=8), seed=1001, query_period=4,
                                slow_mode="frozen")
    model.eval()
    accumulated, accumulated_units = {}, {}
    scheduled, scheduled_units = {}, {}
    for first in range(0, len(lengths), 6):
        batch = lengths[first:first + 6]
        carry = None
        for start in range(0, max(batch), 32):
            packet = synthetic_chunk(batch, start)
            with torch.no_grad():
                result = model(**packet, carry=carry, no_write=view == "trace_no_write")
            expected, units = audit.scheduled_work(packet["lengths"].tolist(), start, view)
            audit.close_equal(result.work_counts, expected, f"{view} batch {first} chunk {start}")
            audit.close_equal(result.memory_work_units, units, f"{view} units {first} chunk {start}")
            audit.add_counts(accumulated, result.work_counts)
            audit.add_counts(accumulated_units, result.memory_work_units)
            audit.add_counts(scheduled, expected)
            audit.add_counts(scheduled_units, units)
            carry = composed.detach_carry(result.carry)
        assert carry.fast.absolute_step.tolist() == list(batch)
        assert carry.fast.ended.tolist() == [True] * len(batch)
    assert accumulated == scheduled and accumulated_units == scheduled_units
    assert accumulated["slow_active_rows"] == sum(lengths)
    assert accumulated["slow_query_rows"] == sum((length + 3) // 4 for length in lengths)
    assert accumulated["memory_eligible_write_steps"] == sum((length - 1) // 4 for length in lengths)
    if view == "trace_no_write":
        assert accumulated["memory_matrix_reads"] > 0
        assert accumulated["memory_matrix_writes"] == 0
    if model.projection is not None:
        assert accumulated["projection_rows"] > accumulated["projection_key_rows"]


def first_training_work(item):
    """One fake first batch, stopped before any full training journal exists."""
    lengths = [1, 4, 5, 33, 65, 8] + [1] * 48
    item.histories_by_stage = {"train": {"episode_offsets": np.array([0, *np.cumsum(lengths)], np.int64)}}
    fit = {"family": "trace_delta", "seed": audit.SEEDS[0], "episode_orders": [list(range(54))],
           "optimizer_parameter_names": ["projection.weight"], "parameters": {"effective_count": 224}}
    item.fits = [fit]
    work, units = {}, {}
    for start in range(0, max(lengths[:6]), 32):
        counts, coordinates = audit.scheduled_work(
            [max(0, min(32, length - start)) for length in lengths[:6]], start, "trace_delta")
        audit.add_counts(work, counts)
        audit.add_counts(units, coordinates)
    timings = {name: .1 for name in ("wall", "setup", "materialize", "forward", "loss", "backward",
        "carry_detach", "gradient_completion", "gradient_clip", "optimizer", "overhead")}
    value = {"version": "otto-query-memory-training-v1", "objective": "full_forecast_aux",
        "episode_indices": list(range(6)), "episode_exposures": 6, "optimizer_updates": 1,
        "optimizer_step": 1, "forward_rows": sum(lengths[:6]), "forward_chunks": 3,
        "nonquery_rows": sum(length - (length + 3) // 4 for length in lengths[:6]),
        "prior_rows": sum((length - 1) // 4 for length in lengths[:6]), "loss_chunks": 3,
        "backward_chunks": 3, "differentiable_chunks": 3, "no_gradient_chunks": 0,
        "skipped_backward_chunks": 0, "effective_parameter_names": ["projection.weight"],
        "effective_parameter_count": 224, "frozen_optimizer_state_entries": 0,
        "zero_filled_gradient_names": [], "loss": 0., "nonquery_loss": 0.,
        "prior_loss": 0., "gradient_norm_before_clip": 0., "work_counts": work,
        "memory_work_units": units, "timing_seconds": timings}
    identity = {"call_id": 1, "family": "trace_delta", "seed": audit.SEEDS[0], "epoch": 1,
                "batch": 0, "episode_indices": list(range(6))}
    return [{"event": "attempt", **identity}, {"event": "return", **identity, "result": value}]


@pytest.mark.parametrize("tamper", (None, "operations", "units", "call_identity"))
def test_work_journal_rejects_schedule_counter_tampering_before_advancing(tmp_path, monkeypatch, tamper):
    item = audit.Audit(SimpleNamespace(output=tmp_path / "audit"))
    item.np, item.run = np, tmp_path
    records = first_training_work(item)
    if tamper == "operations":
        records[1]["result"]["work_counts"]["slow_query_rows"] += 1
    elif tamper == "units":
        records[1]["result"]["memory_work_units"]["matrix_read_terms"] += 1
    elif tamper == "call_identity":
        records[1]["call_id"] = 2
    monkeypatch.setattr(item, "rows", lambda _path: iter(records))
    expected = {None: "complete original work attempt/return pair",
                "operations": "independent scheduled training operations",
                "units": "independent scheduled training units",
                "call_identity": "same-call successful work return"}[tamper]
    # The intact first batch reaches the deliberately absent second batch.
    # Altered counters must instead fail while checking the first batch.
    with pytest.raises(ValueError, match=expected):
        item.journal()


def metadata_fixture(item):
    """All 18 fit / 24 view metadata records, with opaque fake payload hashes."""
    files = {}

    def pin(name):
        files[name] = {"sha256": hashlib.sha256(name.encode()).hexdigest(), "bytes": 1}
        return files[name]

    fits = []
    for seed in audit.SEEDS:
        for family in audit.FITS:
            name = f"checkpoint-{family}-{seed}.npz"
            fits.append({"family": family, "seed": seed, "checkpoint_path": name, "checkpoint": pin(name)})
    views = {}
    for stage in ("train", "dev"):
        views[stage] = []
        for seed in audit.SEEDS:
            for view in audit.VIEWS:
                name = f"{stage}-prediction-{view}-{seed}.npz"
                views[stage].append({"view": view, "seed": seed, "prediction_path": name, "prediction": pin(name)})
    for name in ("fits.json", "forks.json", "train-views.json", "train-history.json", "train-history.npz"):
        pin(name)
    barrier_names = {row["checkpoint_path"] for row in fits} | {
        row["prediction_path"] for row in views["train"]} | {
        "fits.json", "forks.json", "train-views.json", "train-history.json", "train-history.npz"}
    barrier = {"event": "all18_checkpoints_and24_TRAIN_views_closed_before_DEV", "fits": 18,
        "train_views": 24, "optimizer_steps": 7560, "episode_exposures": 45360, "work_sequence": 8000,
        "files": {name: copy.deepcopy(files[name]) for name in barrier_names}}
    barrier_pin = pin("dev-barrier.json")
    progress = [{"event": "fit_complete", **copy.deepcopy(row)} for row in fits]
    progress.append({**copy.deepcopy(barrier), "barrier": barrier_pin})
    progress.append({"event": "dev_decode_after_barrier", "work_sequence": 8000, "barrier": barrier_pin})
    config, runtime = {"fabricated_recipe": True}, {"fabricated_runtime": "unit test"}
    values = {"fits.json": {"fits": fits}, "forks.json": {"forks": [{"seed": seed} for seed in audit.SEEDS]},
        "train-views.json": {"stage": "train", "technical_complete": False, "views": views["train"]},
        "dev-views.json": {"stage": "dev", "technical_complete": False, "views": views["dev"]},
        "dev-barrier.json": barrier,
        "summary.json": {"version": audit.PRODUCER_VERSION, "configuration": config, "fits": copy.deepcopy(fits),
            "train_views": copy.deepcopy(views["train"]), "dev_views": copy.deepcopy(views["dev"]),
            "technical_complete": False, "test_array_decodes": 0, "test_evaluation_admitted": False},
        "runtime.json": {**runtime, "torch_threads": 1, "interop_threads": 1, "deterministic": True,
                         "cuda_used": False, "mps_used": False}}
    item.worker = {"files": files}
    item.plan = {"configuration": config, "runtime": runtime}
    return values, progress


@pytest.mark.parametrize("defect", (None, "early_dev", "duplicate_dev", "late_fit", "missing_checkpoint",
                                    "changed_barrier_hash", "promoted_views", "summary_drift"))
def test_metadata_barrier_requires_all_fits_and_train_outputs_before_single_dev_decode(
        tmp_path, monkeypatch, defect):
    item = audit.Audit(SimpleNamespace(output=tmp_path / "audit"))
    item.run = tmp_path
    values, progress = metadata_fixture(item)
    if defect == "early_dev":
        progress.insert(0, progress.pop())
    elif defect == "duplicate_dev":
        progress.append(copy.deepcopy(progress[-1]))
    elif defect == "late_fit":
        progress.append(copy.deepcopy(progress[0]))
    elif defect == "missing_checkpoint":
        del values["dev-barrier.json"]["files"]["checkpoint-trace_delta-309000001.npz"]
    elif defect == "changed_barrier_hash":
        values["dev-barrier.json"]["files"]["train-history.npz"]["sha256"] = "0" * 64
    elif defect == "promoted_views":
        values["dev-views.json"]["technical_complete"] = True
    elif defect == "summary_drift":
        values["summary.json"]["train_views"][0]["seed"] += 1
    monkeypatch.setattr(audit, "read", lambda path: values[path.name])
    monkeypatch.setattr(item, "rows", lambda _path: iter(progress))
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("metadata audit decoded arrays"))
    if defect is None:
        item.authenticate_metadata()
        assert len(item.fits) == 18 and len(item.views["train"]) == len(item.views["dev"]) == 24
        assert len(item.barrier["files"]) == 47
    else:
        with pytest.raises(ValueError):
            item.authenticate_metadata()
    assert item.authenticated is False and item.receipt["array_decodes"] == 0


def test_checkpoint_witness_records_names_shapes_and_raw_float32_bits():
    arrays = {"slow." + name: np.zeros(shape, np.float32) for name, shape in audit.SLOW_SHAPES.items()}
    arrays["projection.weight"] = np.ones((8, 28), np.float32)
    result = audit.tensor_witness(arrays)
    raw = b"".join(name.encode() + b"\0" + arrays[name].tobytes() for name in sorted(arrays))
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["tensors"]["projection.weight"]["bytes"] == 896
    assert result["tensors"]["projection.weight"]["shape"] == [8, 28]
    assert set(audit.tensor_witness(arrays, slow_only=True)["tensors"]) == set(audit.SLOW_SHAPES)
    arrays["slow.output.bias"][0] = np.float32(-0.)
    assert audit.tensor_witness(arrays)["sha256"] != result["sha256"]


@pytest.mark.parametrize("filename", ("test.npz", "test-prediction-trace_delta-309000001.npz"))
def test_numerical_decode_requires_metadata_admission_and_never_opens_test(tmp_path, monkeypatch, filename):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("forbidden numerical decode"))
    item = audit.Audit(SimpleNamespace(output=tmp_path / "audit"))
    item.np = np
    with pytest.raises(ValueError, match="metadata/closure/barrier"):
        item.arrays(tmp_path / "dev.npz", {})
    item.authenticated = True
    with pytest.raises(ValueError, match="cannot decode TEST"):
        item.arrays(tmp_path / filename, {})
    assert item.receipt["array_decodes"] == item.receipt["test_array_decodes"] == 0


def test_failed_admission_preserves_failure_receipt_without_numerical_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit.signal, "signal", lambda *_args: None)
    item = audit.Audit(SimpleNamespace(output=tmp_path / "audit"))

    def fail():
        raise ValueError("fabricated unclosed original producer")

    monkeypatch.setattr(item, "admit", fail)
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("array decode before closed admission"))
    with pytest.raises(ValueError, match="unclosed original"):
        item.execute()
    receipt = json.loads((tmp_path / "audit/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["complete"] is False
    assert receipt["array_decodes"] == receipt["model_calls"] == receipt["optimizer_calls"] == 0
    assert not (tmp_path / "audit/audit.json").exists()
