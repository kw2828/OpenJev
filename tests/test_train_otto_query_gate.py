import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_test_query_trainer", ROOT / "scripts/train_otto_query_gate.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def fabricated(lengths):
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    n = int(offsets[-1])
    x = np.zeros((n, 31), np.float32)
    x[:, 2:6] = 1
    scheduled = np.zeros(n, np.bool_)
    for start, end in itertools.pairwise(offsets):
        t = np.arange(end - start)
        x[start:end, 15:17] = (t[:, None] / 2188).astype(np.float32)
    return {"features": x, "labels": np.ones(n, np.bool_),
            "neural_costs": np.tile(np.asarray([1, 2, 3, 4], np.float32), (n, 1)),
            "analytic_action": np.ones(n, np.int64), "masks": np.ones((n, 4), np.bool_),
            "episode_offsets": offsets, "scheduled_query": scheduled, "neural_gap": np.ones(n, np.float32)}


def test_episode_equal_weights_and_exact_tail_window_coverage():
    data = fabricated([1, 33, 2])
    weights = runner.validate_data(data, np, episodes=3)
    assert weights[0] == 12 and weights[-1] == 6
    for begin, end in zip(data["episode_offsets"][:-1], data["episode_offsets"][1:], strict=True):
        assert float(weights[begin:end].astype(np.float64).sum()) == pytest.approx(12, abs=1e-6)
    chunks = list(runner.windows(np.asarray([1, 0, 2]), data["episode_offsets"], np, batch_size=2, window=32))
    assert [(b, t) for b, t, _ in chunks] == [(0, 0), (0, 32), (1, 0)]
    assert chunks[1][2][0, 0] == 33 and (chunks[1][2][1] == -1).all()
    actual = [int(x) for _, _, a in chunks for x in a.flat if x >= 0]
    assert sorted(actual) == list(range(36)) and len(actual) == len(set(actual))


def test_fixed_256_loss_divider_retains_tail_weight_and_zero_padding_gradient():
    logits = torch.zeros(8, 32, requires_grad=True)
    labels, weights = torch.zeros(8, 32), torch.zeros(8, 32)
    labels[0, 0] = 1
    weights[0, :2] = torch.tensor([1., 3.])
    loss, numerator = runner.window_loss(logits, labels, weights, torch)
    assert numerator.item() == pytest.approx(4 * math.log(2), abs=2e-7)
    assert loss.item() == pytest.approx(math.log(2) / 64, abs=1e-8)
    loss.backward()
    assert logits.grad[0, 0].item() == -.5 / 256
    assert logits.grad[0, 1].item() == 1.5 / 256
    assert torch.count_nonzero(logits.grad) == 2


@pytest.mark.parametrize("bad", ["age", "annotation_label", "gap", "offset", "dtype"])
def test_rejects_rewritten_query_history_or_labels(bad):
    data = fabricated([2, 2])
    if bad == "age":
        data["features"][1, 17] = 1
    elif bad == "annotation_label":
        data["labels"][0] = False
    elif bad == "gap":
        data["neural_gap"][0] = 2
    elif bad == "offset":
        data["episode_offsets"][1] = 0
    else:
        data["features"] = data["features"].astype(np.float64)
    with pytest.raises(ValueError):
        runner.validate_data(data, np, episodes=2)


def test_failed_return_publication_keeps_executed_update_pending(tmp_path):
    run = runner.Run(SimpleNamespace(output=tmp_path))
    run.check = lambda: None
    events, executed = [], []
    original = OSError("durable return failed")

    def emit(_name, record):
        events.append(record)
        if record["event"] == "return":
            raise original

    run.emit = emit
    with pytest.raises(OSError) as caught:
        run.call("optimizer_update", lambda: executed.append(True))
    assert caught.value is original and executed == [True]
    assert [x["event"] for x in events] == ["attempt", "return"]
    assert run.calls["optimizer_update"]["attempted"] == 1
    assert run.calls["optimizer_update"]["returned"] == 0 and len(run.pending) == 1


def test_incomplete_collection_refused_before_payload_access(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    roles = {name: {"path": str(tmp_path / name), "bytes": 1, "sha256": "a" * 64} for name in runner.ROLES}
    monkeypatch.setattr(runner, "digest", lambda *_: {"bytes": 1, "sha256": "a" * 64})
    accessed = []
    records = {
        "collection_plan": {"version": "otto-query-gate-collection-v1"},
        "collection_receipt": {"version": "otto-query-gate-collection-v1", "status": "failed"},
        "collection_terminal": {},
    }

    def read(p):
        accessed.append(p.name)
        return records[p.name]

    monkeypatch.setattr(runner, "read", read)
    with pytest.raises(ValueError, match="completed full collection"):
        runner.metadata_closure(roles)
    assert accessed == ["collection_plan", "collection_receipt", "collection_terminal"]


def test_late_failure_demotes_completed_receipt_preserving_original(tmp_path, monkeypatch):
    output = tmp_path / "run"
    run = runner.Run(SimpleNamespace(output=output))
    original = OSError("late write")
    run.bind = lambda: None

    def body():
        runner.write(output / "receipt.json", {"status": "completed", "qualified": True})
        raise original

    run.body = body
    monkeypatch.setattr(runner.signal, "signal", lambda *_: None)
    with pytest.raises(OSError) as caught:
        run.execute()
    assert caught.value is original
    assert json.loads((output / "receipt.invalid.json").read_text())["status"] == "completed"
    failed = json.loads((output / "receipt.json").read_text())
    assert failed["status"] == "failed" and failed["qualified"] is False
