"""Artificial-only pilot lifecycle, pooling, identity and loss fixtures."""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("belief_pilot_tested", ROOT / "scripts/pilot_dialogue_belief_pooling.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
torch.set_num_threads(1)


class Budget:
    def __init__(self):
        self.ticks = 0

    def check(self):
        self.ticks += 1

    def storage(self):
        self.check()

    def elapsed(self):
        self.check()
        return self.ticks / 1000

    def sync(self, _torch):
        self.check()


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(384), requires_grad=False)
        self.calls = 0

    def forward(self, *, input_ids, attention_mask):
        self.calls += 1
        assert not torch.is_grad_enabled() and not self.training
        value = input_ids[..., None].float() + torch.arange(384)[None, None]/1000
        return SimpleNamespace(last_hidden_state=value*self.scale)


def test_exact_hash_selection_ignores_targets_and_input_order():
    import hashlib

    profiles = [{"split": split, "dialogue_id": f"d{i}", "work": {"anything": i}, "label": "unused"}
                for split, n in (("train", 2017), ("dev", 2363)) for i in range(n)]
    selected = pilot.select_ids(profiles)
    assert selected == pilot.select_ids(list(reversed(profiles)))
    for split in selected:
        expected = sorted((p["dialogue_id"] for p in profiles if p["split"] == split),
                          key=lambda did: (hashlib.sha256(f"openjev-belief-pooling-pilot-v1:{split}:{did}".encode()).hexdigest(), did))[:128]
        assert selected[split] == expected
    profiles[-1]["dialogue_id"] = profiles[-2]["dialogue_id"]
    with pytest.raises(ValueError, match="inventory"):
        pilot.select_ids(profiles)


def test_orders_are_complete_paired_and_epoch_specific():
    ids = [f"d{i}" for i in range(128)]
    result = pilot.epoch_orders(ids)
    assert result == pilot.epoch_orders(list(reversed(ids)))
    assert set(result) == {"7101", "7102"}
    assert all(sorted(epoch) == sorted(ids) for epochs in result.values() for epoch in epochs)
    assert result["7101"] != result["7102"]
    assert len({tuple(epoch) for epoch in result["7101"]}) == 3


def test_global_text_identity_allows_cross_split_same_dialogue_id():
    a = {"original_feature_ids": [7, 9], "tokens": [[1], [2, 3]]}
    b = {"original_feature_ids": [11, 9], "tokens": [[4], [2, 3]]}
    ids, tokens = pilot.unique_texts({("train", "same"): a, ("dev", "same"): b})
    assert ids == [7, 9, 11] and tokens == [[1], [2, 3], [4]]
    b["tokens"][1] = [99]
    with pytest.raises(ValueError, match="token identity"):
        pilot.unique_texts({("train", "same"): a, ("dev", "same"): b})


def test_cache_exact_chunk_pool_and_raw_specials_without_padding(tmp_path):
    from openjev.research.dialogue_trainable_encoder import encode_token_lists

    ids, tokens = [3, 8], [[7]*260, [13]*2]
    encoder = Encoder().eval()
    progress = {}
    pooled, hidden, offsets, work = pilot.cache_tokens(encoder, tokens, ids,
        {"cls_id": 101, "sep_id": 102, "pad_id": 0}, tmp_path, Budget(), progress)
    assert offsets.tolist() == [0, 264, 268]
    assert hidden.shape == (268, 384)
    assert hidden[[0, 255, 256, 263, 264, 267], 0].tolist() == [101, 102, 101, 102, 101, 102]
    assert (hidden[:, 0] != 0).all()
    expected, expected_work = encode_token_lists(Encoder().eval(), tokens, cls_id=101, sep_id=102, pad_id=0, trainable=False)
    assert np.array_equal(pooled, expected.numpy()) and work == expected_work
    assert encoder.calls == progress["encoder_forward_attempts"] == progress["encoder_forward_returns"] == 1
    assert not encoder._forward_hooks and not encoder._forward_pre_hooks


def fixture():
    ids = [f"d{i}" for i in range(128)]
    candidates = ["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:x"]
    actor = {"original_feature_ids": list(range(6)), "turn_text_ids": [0, 1], "query_text_ids": [2],
             "candidate_text_ids": [[3, 4, 5]], "candidate_ids": [candidates], "query_ids": [0],
             "lexical_offset": 0, "lexical_shape": [2, 1, 3, 10]}
    actors, targets, rows = {}, {}, []
    for split in ("train", "dev"):
        for did in ids:
            actors[split, did] = actor
            targets[split, did] = [{"split": split, "dialogue_id": did, "source_row_index": t, "time": t,
                "query_position": 0, "query_index": 0, "label_index": 2, "label_id": candidates[2], "stratum_index": 2} for t in range(2)]
            if split == "dev":
                base = len(rows)
                rows.extend({**r, "row_index": base+i, "candidate_ids": candidates} for i, r in enumerate(targets[split, did]))
    pooled = np.ones((6, 384), np.float32)/np.sqrt(np.float32(384))
    cache = pooled, np.ones((12, 384), np.float32), np.arange(0, 13, 2, dtype=np.int64), {i: i for i in range(6)}
    plan = {"selected": {"train": ids, "dev": ids}, "orders": pilot.epoch_orders(ids)}
    return plan, actors, targets, np.zeros(60, np.float32), rows, cache


class TinyHead(nn.Module):
    method = "pooled"

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([.1, .2, .3]))
        self.unused_attention = nn.Parameter(torch.ones(2))
        self.last_audit = {}

    def configuration(self):
        return {"method": self.method, "parameters": 5, "active_parameters": 3}

    def forward(self, token_states, pooled_turns, query, candidates, candidate_mask, lexical, none_index):
        assert len(token_states) == 2 and pooled_turns.shape == (2, 384)
        assert query.shape == (1, 384) and candidates.shape == (1, 3, 384)
        assert candidate_mask.tolist() == [[True, True, True]] and none_index.tolist() == [0]
        self.last_audit = {"forward_attempts": 1, "forward_returns": 1, "observation_attempts": 2, "observation_returns": 2,
            "step_attempts": 2, "step_returns": 2, "state_checks": 2, "real_question_updates": 2, "attention_positions": 0}
        return self.logits.log_softmax(-1).reshape(1, 1, 1, 3).expand(1, 2, 1, 3)


def test_full_tiny_fit_exact_counts_unused_parameters_and_saved_endpoint_rows(tmp_path):
    plan, actors, targets, lexical, rows, cache = fixture()
    model, progress = TinyHead(), {}
    original_unused = model.unused_attention.detach().clone()
    result = pilot.fit(model, "pooled-7101", plan, actors, targets, lexical, rows, cache, tmp_path, Budget(), progress)
    assert result["counts"]["optimizer_updates"] == 48
    assert result["counts"]["training_forwards"] == 384 and result["counts"]["evaluation_forwards"] == 128
    assert result["counts"]["training_endpoints"] == 768
    assert result["training_work"]["step_returns"] == 768 and result["evaluation_work"]["step_returns"] == 256
    assert result["gradient_present_names"] == result["nonzero_gradient_names"] == ["logits"]
    assert result["registered_parameters"] == 5 and result["gradient_present_parameters"] == 3
    assert torch.equal(original_unused, model.unused_attention)
    assert {p.name for p in tmp_path.iterdir()} == {"weights.pt", "updates.jsonl", "predictions.npz"}
    records = list(pilot.lines(tmp_path / "updates.jsonl"))
    assert len(records) == 48 and all(r["endpoint_count"] == 16 and len(r["dialogue_ids"]) == 8 for r in records)
    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as saved:
        assert saved["log_probs"].shape == (256, 12) and saved["log_probs"].dtype == np.float32
        assert np.array_equal(saved["row_indices"], np.arange(256)) and np.isneginf(saved["log_probs"][:, 3:]).all()


def test_actor_assembly_uses_numbers_without_method_names_or_target_fields():
    _, actors, _, lexical, _, cache = fixture()
    record = {**actors["train", "d0"], "label_index": 999, "gold_previous": "must-not-enter"}
    inputs = pilot.actor_inputs(record, lexical, *cache)
    assert len(inputs) == 7 and inputs[-1].tolist() == [0]
    assert isinstance(inputs[0], list) and all(v.dtype == torch.float32 for v in inputs[0])
    assert all(not v.requires_grad for v in inputs[0])


def test_evaluator_values_come_from_exact_authenticated_packet_join(tmp_path, monkeypatch):
    _, actors, targets, _, _, _ = fixture()
    queries = []
    records = {}
    for qi, split in enumerate(("train", "dev")):
        actor = {**actors[split, "d0"], "split": split, "dialogue_id": "d0", "query_ids": [qi]}
        endpoint = {**targets[split, "d0"][0], "query_index": qi, "query_id": f"query-{split}", "service": "s", "slot": "k"}
        records[f"actors-{split}.jsonl"] = [actor]
        records[f"targets-{split}.jsonl"] = [{"split": split, "dialogue_id": "d0", "rows": [endpoint]}]
        queries.append({"split": split, "id": endpoint["query_id"], "service": "s", "slot": "k",
                        "candidate_ids": actor["candidate_ids"][0], "candidate_values": [None, None, "exact source value"]})
    monkeypatch.setattr(pilot, "lines", lambda path: iter(records[path.name]))
    monkeypatch.setattr(pilot, "read", lambda path: {"queries": queries})
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: np.zeros(60, np.float32))
    plan = {"prepared": str(tmp_path), "selected": {"train": ["d0"], "dev": ["d0"]}}
    _, _, _, rows = pilot.load_inputs(plan, Budget())
    assert rows["dev"][0]["candidate_values"] == [None, None, "exact source value"]
    queries[1]["service"] = "wrong service"
    with pytest.raises(ValueError, match="canonical query"):
        pilot.load_inputs(plan, Budget())


def test_failed_forward_retains_partial_internal_attempt_counts():
    class Failing(TinyHead):
        def forward(self, *_):
            self.last_audit = {"forward_attempts": 1, "forward_returns": 0, "step_attempts": 1, "step_returns": 0}
            raise RuntimeError("forward failed")

    _, actors, _, lexical, _, cache = fixture()
    counts, audit = Counter(), Counter()
    with pytest.raises(RuntimeError, match="forward failed"):
        pilot.forward(Failing(), pilot.actor_inputs(actors["train", "d0"], lexical, *cache), Budget(), counts, training=True, audit=audit)
    assert counts["forward_attempts"] == 1 and counts["forward_returns"] == 0 and audit["step_attempts"] == 1


def test_uniform_endpoint_loss_is_whole_batch_not_dialogue_average():
    from openjev.research.dialogue_finetune_training import supervised_loss

    a = torch.tensor([[[[-.2, -2., -3.]]]], requires_grad=True)
    b = torch.tensor([[[[-.9, -1., -3.]], [[-.3, -2., -3.]]]], requires_grad=True)
    one = [{"time": 0, "query_position": 0, "label_index": 0, "stratum_index": 0}]
    two = one + [{"time": 1, "query_position": 0, "label_index": 0, "stratum_index": 2}]
    result = supervised_loss(a, one, [1., 1., 1.], 3) + supervised_loss(b, two, [1., 1., 1.], 3)
    assert float(result.detach()) == pytest.approx((.2+.9+.3)/3)
    result.backward()
    assert a.grad[0, 0, 0, 0] == pytest.approx(-1/3)


def test_authentication_rejects_before_model_or_data_when_plan_pin_wrong(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="External pilot plan pin"):
        pilot.authenticate(SimpleNamespace(plan=path, plan_sha256="0"*64, out=tmp_path))


def test_exclusive_output_and_failure_preservation(tmp_path, monkeypatch):
    out = tmp_path / "attempt"
    args = SimpleNamespace(command="freeze", out=out, prepared=tmp_path, prepared_sha256="bad")
    def fail(*_):
        raise ValueError("synthetic authentication rejection")
    monkeypatch.setattr(pilot, "freeze", fail)
    with pytest.raises(ValueError, match="synthetic authentication rejection"):
        pilot.execute(args)
    failed = json.loads((out / "failed.json").read_text())
    assert failed["status"] == "failed" and failed["quality_metrics_computed"] is False
    assert not (out / "completed.json").exists()
    with pytest.raises(FileExistsError):
        pilot.execute(args)


def test_late_success_demoted_not_admitted(tmp_path, monkeypatch):
    args = SimpleNamespace(command="freeze", out=tmp_path / "attempt", prepared=tmp_path, prepared_sha256="bad")
    monkeypatch.setattr(pilot, "freeze", lambda *_: {"model_calls": 0})
    old = pilot.Budget.storage
    def late(self):
        if (self.out / "completed.json").exists():
            raise TimeoutError("late write")
        old(self)
    monkeypatch.setattr(pilot.Budget, "storage", late)
    with pytest.raises(TimeoutError, match="late write"):
        pilot.execute(args)
    assert (args.out / "invalid-completion.json").exists() and (args.out / "failed.json").exists()
    assert not (args.out / "completed.json").exists()


def test_cli_requires_supervised_fixed_run_without_threshold_override():
    with pytest.raises(SystemExit):
        pilot.parse_args(["run", "--plan", "p", "--plan-sha256", "s", "--out", "o"])
    args = pilot.parse_args(["run", "--plan", "p", "--plan-sha256", "s", "--supervision", "l", "--out", "o"])
    assert args.command == "run" and args.supervision == Path("l")
