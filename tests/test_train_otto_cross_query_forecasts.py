"""Fabricated chronological training contracts; no empirical files or teachers."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openjev.research import otto_cross_query_data as data
from openjev.research import otto_cross_query_scores as models
from openjev.research import otto_score_forecast_data as windows

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_cross_query_training_fixture", ROOT / "scripts/train_otto_cross_query_forecasts.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def history(lengths=(37, 5), *, positive=True):
    offsets = np.array([0, *np.cumsum(lengths)], np.int64)
    total = int(offsets[-1])
    features = np.zeros((total, 31), np.float32)
    query_mask = np.zeros(total, np.bool_)
    queries = np.full((total, 4), np.nan, np.float32)
    targets = np.zeros((total, 4), np.float32)
    weights = np.zeros(total, np.float64)
    legal = np.ones((total, 4), np.bool_)
    for index, length in enumerate(lengths):
        low = int(offsets[index]); steps = np.arange(length)
        features[low:low + length, 2:6] = 1
        features[low:low + length, 15] = steps / 2188
        features[low:low + length, 16] = steps % 4 / 2188
        features[low:low + length, 17] = 1
        features[low:low + length, 0] = (index + 1) / 10
        query_mask[low:low + length] = steps % 4 == 0
        queries[low + steps[steps % 4 == 0]] = np.array([31., 35., 33., 32.], np.float32)
        if positive:
            selected = [step for step in range(length) if step >= 33 and step % 4 != 0]
            targets[low + np.array(selected, np.int64)] = np.array([34., 30., 32., 37.], np.float32)
            weights[low + np.array(selected, np.int64)] = .025
    return {"version": data.VERSION, "features": features, "query_scores": queries,
        "targets": targets, "legal": legal, "actions": np.zeros(total, np.int64),
        "query_mask": query_mask, "weights": weights, "episode_offsets": offsets,
        "episode_ids": tuple(f"e{i}" for i in range(len(lengths))),
        "episode_regimes": tuple("lambda3" for _ in lengths),
        "episode_splits": tuple("train" for _ in lengths),
        "counts": {"rows": total, "episodes": len(lengths)}}


def test_loss_is_capacity_expression_with_neutral_empty_padding():
    generator = torch.Generator().manual_seed(31)
    prediction = torch.rand((2, 4, 4), generator=generator, dtype=torch.float32) * 64
    target = torch.rand((2, 4, 4), generator=generator, dtype=torch.float32) * 64
    legal = torch.tensor([[[True, False, True, True]] * 4, [[False, True, True, False]] * 4])
    allowed = legal.to(torch.float32)
    count = allowed.sum(-1, keepdim=True)
    p, t = prediction / 64, target / 64
    difference = p - (p * allowed).sum(-1, keepdim=True) / count - t + (t * allowed).sum(-1, keepdim=True) / count
    expected = (difference.square() * allowed).sum(-1) / count[:, :, 0]
    actual = runner.centered_rows(torch, prediction, target, legal)
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual).all()
    legal[1, 3] = False
    padded = runner.centered_rows(torch, prediction, target, legal)
    assert padded[1, 3].item() == 0 and torch.isfinite(padded).all()


@pytest.mark.parametrize("kind", models.KINDS)
def test_full_batch_preserves_history_and_detaches_before_single_update(kind):
    fixture = history()
    originals = {k: v.tobytes() for k, v in fixture.items() if isinstance(v, np.ndarray)}
    model = models.make_head(kind, 91)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    initial = {k: p.detach().clone() for k, p in model.named_parameters()}
    calls, stages = [], []
    original_forward = model.forward

    def record_forward(*args, **kwargs):
        carry = kwargs["carry"]
        assert not carry.hidden.requires_grad and carry.hidden.grad_fn is None
        assert all(torch.equal(p, initial[k]) for k, p in model.named_parameters())
        query = kwargs["query_scores"]
        assert torch.isnan(query[~kwargs["query_mask"]]).all()
        calls.append((torch.is_grad_enabled(), carry.absolute_step.tolist(), kwargs["lengths"].tolist(),
                      kwargs["episode_ends"].tolist()))
        return original_forward(*args, **kwargs)

    model.forward = record_forward
    result = runner.batch_update(torch, models, data, model, optimizer, fixture, [0, 1],
                                 check=lambda: None, stage=lambda name, step: stages.append((name, step)))
    assert calls == [(False, [0, 0], [32, 5], [False, True]), (True, [32, 5], [5, 0], [True, False])]
    assert result["forward_chunks"] == 2 and result["backward_chunks"] == 1
    assert result["no_grad_chunks"] == 1 and result["forward_rows"] == 42
    assert result["optimizer_step"] == 1 and result["loss"] > 0
    assert stages[-1] == ("optimizer_update", None)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and optimizer.state[p]["step"] == 1
               for p in model.parameters())
    assert not torch.equal(model.output.weight, initial["output.weight"])
    assert all(fixture[k].tobytes() == value for k, value in originals.items())


def test_zero_target_batch_after_learning_advances_adam_with_explicit_zero_gradients():
    model = models.make_head("innovation", 41)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    runner.batch_update(torch, models, data, model, optimizer, history(), [0, 1],
                        check=lambda: None, stage=lambda *_: None)
    before = {key: p.detach().clone() for key, p in model.named_parameters()}
    result = runner.batch_update(torch, models, data, model, optimizer, history(positive=False), [0, 1],
                                 check=lambda: None, stage=lambda *_: None)
    assert result["loss"] == 0 and result["backward_chunks"] == 0 and result["no_grad_chunks"] == 2
    assert result["optimizer_step"] == 2
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 and optimizer.state[p]["step"] == 2
               for p in model.parameters())
    # A real prior moment makes this distinguish a declared zero-gradient Adam
    # step from silently skipping the batch.
    assert not torch.equal(model.output.weight, before["output.weight"])


def test_full_valid_projection_query_isolation_and_flat_window_mapping(tmp_path):
    identities = [r for r in runner.cohort() if r["stage"] == "valid"]
    lengths = [9, 5, 1] * 12
    fixture = history(lengths, positive=False)
    raw = np.arange(len(fixture["features"]) * 4, dtype=np.float32).reshape(-1, 4) / 4 + 20
    flat = {"features": fixture["features"], "raw_q": raw, "legal": fixture["legal"],
        "actions": fixture["actions"], "correction": fixture["query_mask"], "episode_offsets": fixture["episode_offsets"]}
    np.savez(tmp_path / "valid.npz", **flat)
    run = runner.Run(SimpleNamespace(output=tmp_path))
    # Barrier executes before accessing np, collection_dir or any actual file.
    with pytest.raises(ValueError, match="before VALID"):
        run.validation()
    run.np, run.data, run.windows = np, data, windows
    run.collection_dir = tmp_path
    run.collection_plan = {"cohort": identities}
    run.receipt["fits_completed"] = 12
    with pytest.raises(ValueError, match="before VALID"):
        run.validation()
    run.valid_allowed = True; run.check = lambda: None
    chronological, grouped, episodes, found_ids = run.validation()
    assert found_ids == identities and all("actions" not in e for e in episodes)
    assert chronological["targets"].tobytes() == raw.tobytes()
    mask = chronological["query_mask"]
    assert chronological["query_scores"][mask].tobytes() == raw[mask].tobytes()
    assert not chronological["query_scores"][~mask].any()
    assert not chronological["weights"].any()
    rebuilt = runner.window_predictions(np, grouped, chronological, raw)
    assert rebuilt.tobytes() == grouped["targets"].tobytes()
    for index in range(len(identities)):
        packet = data.batch_chunk(chronological, [index], 0)
        inputs = packet["model_inputs"]
        assert set(inputs) == {"features", "query_scores", "lengths", "query_mask", "episode_ends"}
        assert np.isnan(inputs["query_scores"][~inputs["query_mask"]]).all()


def test_return_publication_failure_preserves_uncertain_update(tmp_path, monkeypatch):
    run = runner.Run(SimpleNamespace(output=tmp_path))
    run.check = lambda: None
    run.clock = SimpleNamespace(now_ns=lambda: 100)
    run.torch = run.models = run.data = object()
    seen = []

    def event(record, filename):
        seen.append(record)
        if record["event"] == "return":
            raise OSError("durable return failed")

    def update(*args, stage, **kwargs):
        stage("optimizer_update", None)
        return {"forward_chunks": 2, "backward_chunks": 1, "forward_rows": 42, "optimizer_step": 1}

    run.event = event
    monkeypatch.setattr(runner, "batch_update", update)
    with pytest.raises(OSError, match="durable return"):
        run.batch(None, None, None, [0, 1], "innovation", 11, 0, 0)
    assert len(seen) == 2 and run.receipt["optimizer_steps"] == 0
    assert run.receipt["pending"]["stage"] == "optimizer_update"
    assert run.receipt["pending"]["episode_indices"] == [0, 1]


def test_checkpoint_requires_exact_saved_tensor_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    run = runner.Run(SimpleNamespace(output=tmp_path))
    run.np = np; run.check = lambda: None
    model = models.make_head("persistent_direct", 19)
    good = run.checkpoint("good.npz", model)
    assert good == runner.descriptor(tmp_path / "good.npz")

    def corrupt(name, arrays):
        changed = copy.deepcopy(arrays)
        changed["output.bias"][0] = np.float32(-0.)
        np.savez(tmp_path / name, **changed)

    run.npz = corrupt
    with pytest.raises(ValueError, match="exact published bytes"):
        run.checkpoint("bad.npz", model)


def test_late_publication_failure_demotes_completed_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.signal, "signal", lambda *_: None)
    output = tmp_path / "run"
    run = runner.Run(SimpleNamespace(output=output, plan=tmp_path / "plan", plan_sha256="p", supervision=tmp_path / "launch"))
    run.admit = lambda: None; run.body = lambda: None
    run.plan = {"sources": {}, "inputs": {}}
    run.collection_receipt = {"files": {}}
    run.collection_dir = tmp_path
    run.receipt.update(fits_completed=12, optimizer_steps=8640, supervision_sha256="s")
    run.clock = SimpleNamespace(now_ns=lambda: 2)
    run.start = 1
    monkeypatch.setattr(runner, "PAYLOADS", set())
    original_descriptor = runner.descriptor

    def descriptor(path):
        if path == run.args.plan:
            return {"sha256": "p", "bytes": 0}
        if path == run.args.supervision:
            return {"sha256": "s", "bytes": 0}
        return original_descriptor(path)

    monkeypatch.setattr(runner, "descriptor", descriptor)

    def check():
        if (output / "receipt.json").exists():
            raise TimeoutError("late deadline")

    run.check = check
    with pytest.raises(TimeoutError, match="late deadline"):
        run.execute()
    assert json.loads((output / "receipt.invalid.json").read_text())["status"] == "completed"
    failed = json.loads((output / "receipt.json").read_text())
    assert failed["status"] == "failed" and failed["complete"] is False
    assert "late deadline" in failed["error"]
