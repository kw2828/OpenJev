"""Synthetic integration and receipt failures, without pretrained assets."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from openjev.research.dialogue_finetune_inputs import synthetic_targets, workload_profile
from openjev.research.dialogue_trainable_encoder import build_actor, encode_token_lists

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("qualification_under_test", SCRIPTS / "qualify_dialogue_finetune.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
from study_dialogue_copy_v2 import MonitoredCopyMemoryV2

torch.set_num_threads(1)


def payload():
    return {"tokens": [[1, 2, 3], [4, 5], [6], [7], [8], [9], [10], [11], [12]],
        "original_feature_ids": list(range(9)), "turn_text_ids": [0, 1, 0],
        "query_text_ids": [2, 3], "candidate_text_ids": [[4, 5, 6], [7, 4, 5, 8]],
        "candidate_ids": [["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:X"],
                          ["value:Y", "reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:Z"]],
        "lexical": np.zeros((3, 2, 4, 10), dtype=np.float32)}


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.attention = nn.Linear(8, 8, bias=False)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        scores = self.attention(x) @ x.transpose(-1, -2)
        weights = scores.masked_fill(~attention_mask.bool()[:, None, :], -torch.inf).softmax(-1)
        return SimpleNamespace(last_hidden_state=x + weights @ x)


@pytest.mark.parametrize("trainable", [False, True])
def test_serialized_payload_through_actual_monitor_loss_and_optimizer(trainable):
    p = payload()
    p["lexical"] = p["lexical"].tolist()
    restored = pilot.restore_payload(json.loads(json.dumps(p)))
    assert restored["lexical"].dtype == np.float32
    assert synthetic_targets(restored).shape == (3, 2)
    work = workload_profile(restored)
    torch.manual_seed(73)
    encoder = TinyEncoder().requires_grad_(trainable)
    memory = MonitoredCopyMemoryV2("scalar", input_dim=8, projection_dim=4, hidden_dim=6)
    before_encoder, before_memory = pilot.tensor_digest(encoder), pilot.tensor_digest(memory)
    parameters = list(memory.parameters()) + (list(encoder.parameters()) if trainable else [])
    optimizer = torch.optim.AdamW(parameters, lr=.001)
    for offset in range(pilot.CONFIG["accumulation"]):
        vectors, actual_work = encode_token_lists(encoder, restored["tokens"], cls_id=30, sep_id=31,
                                                  pad_id=0, trainable=trainable)
        assert all(v == work[k] for k, v in actual_work.items())
        actor, none = build_actor(vectors, **{k: restored[k] for k in
            ("turn_text_ids", "query_text_ids", "candidate_text_ids", "candidate_ids", "lexical")},
            memory_device="cpu")
        memory.begin_batch(actor, [{"layout": {"shape": [3, 2, 4, 10]}}])
        logs = memory(*actor, none_index=none)
        value = pilot.synthetic_loss(logs, restored, offset)
        targets = synthetic_targets(restored, offset=offset)
        explicit = -sum(logs[0, t, q, targets[t, q]] for t in range(3) for q in range(2)) / 24
        torch.testing.assert_close(value, explicit)
        value.backward()
        for name in ("incoming", "feature", "result", "mass"):
            assert memory.audit[name + "_checks"] == 6
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in memory.parameters())
    for parameter in encoder.parameters():
        if trainable:
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0
        else:
            assert parameter.grad is None
    torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True, foreach=False)
    optimizer.step()
    assert (pilot.tensor_digest(encoder) != before_encoder) is trainable
    assert pilot.tensor_digest(memory) != before_memory


def test_loss_rejects_partial_public_stream():
    p = payload()
    with pytest.raises(ValueError, match="Full public"):
        pilot.synthetic_loss(torch.zeros(1, 2, 2, 4), p, 0)


def test_failed_attempt_preserves_progress_and_demotes_completion(tmp_path):
    out = tmp_path / "attempt"
    with pytest.raises(ValueError, match="deliberate"), pilot.attempt(out, "fake", {}) as (_, progress, _):
        progress["encoder_calls"] = 7
        pilot.write(out / "completed.json", {"status": "premature"})
        raise ValueError("deliberate")
    assert not (out / "completed.json").exists()
    assert (out / "incomplete-completion.json").exists()
    receipt = pilot.read(out / "failed.json")
    assert receipt["progress"]["encoder_calls"] == 7
    assert receipt["error_type"] == "ValueError"
    with pytest.raises(FileExistsError), pilot.attempt(out, "fake", {}):
        pass


def test_final_cap_failure_never_leaves_success_marker(tmp_path, monkeypatch):
    peak = [0]
    monkeypatch.setattr(pilot, "rss", lambda: peak[0])
    out = tmp_path / "attempt"
    with pytest.raises(ValueError, match="RSS cap"), pilot.attempt(out, "fake", {}):
        pilot.write(out / "completed.json", {"status": "premature"})
        peak[0] = pilot.CONFIG["rss_bytes"] + 1
    assert (out / "failed.json").exists() and not (out / "completed.json").exists()


def test_file_identity_and_tensor_digests_reject_mutation(tmp_path):
    path = tmp_path / "input"
    path.write_text("original")
    pins = {path.name: pilot.sha(path)}
    pilot.bind(pins, root=tmp_path)
    path.write_text("tampered")
    with pytest.raises(ValueError, match="Changed"):
        pilot.bind(pins, root=tmp_path)
    model = TinyEncoder()
    twin = copy.deepcopy(model)
    assert pilot.tensor_digest(model) == pilot.tensor_digest(twin)
    with torch.no_grad():
        twin.embedding.weight[0, 0] += 1
    assert pilot.tensor_digest(model) != pilot.tensor_digest(twin)


def test_external_completion_pin_rejects_before_loading_plan(tmp_path):
    pilot.write(tmp_path / "completed.json", {"status": "completed"})
    with pytest.raises(ValueError, match="Preparation completion"):
        pilot.authenticate_preparation(tmp_path, "0" * 64)


def test_preparation_closure_rejects_unlisted_file(tmp_path):
    pilot.write(tmp_path / "completed.json", {"status": "completed", "files": {}})
    (tmp_path / "unexpected").write_text("extra")
    with pytest.raises(ValueError, match="Preparation closure"):
        pilot.authenticate_preparation(tmp_path, pilot.sha(tmp_path / "completed.json"))
