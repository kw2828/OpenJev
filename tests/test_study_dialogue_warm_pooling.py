"""Tiny artificial warm-start, exact encoder-call and fail-closed fixtures."""
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
sys.path.insert(0, str(ROOT/"scripts"))
spec = importlib.util.spec_from_file_location("warm_runner_tested", ROOT/"scripts/study_dialogue_warm_pooling.py")
warm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(warm)


class Budget:
    def __init__(self, out):
        self.out, self.ticks = out, 0

    def check(self):
        self.ticks += 1

    def elapsed(self):
        self.check()
        return self.ticks/1000

    def storage(self):
        self.check()

    def sync(self, _torch):
        self.check()


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(384), requires_grad=False)
        self.calls = []

    def forward(self, *, input_ids, attention_mask):
        assert not torch.is_grad_enabled() and not self.training
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        values = input_ids[..., None].float() + torch.arange(384)[None, None]/1000
        return SimpleNamespace(last_hidden_state=values*self.scale)


def test_full_cohort_preserves_metadata_order_and_split_qualified_ids():
    profiles = [{"split": split, "dialogue_id": f"d{i}"} for split, n in (("train", 2017), ("dev", 2363)) for i in range(n)]
    selected = warm.full_ids(profiles)
    assert selected["train"][0] == selected["dev"][0] == "d0"
    assert len(selected["train"]) == 2017 and len(selected["dev"]) == 2363
    orders = warm.epoch_orders(selected["train"])
    assert set(orders) == {"6901", "6902", "6903"}
    assert all(len(epochs) == 5 and all(set(epoch) == set(selected["train"]) for epoch in epochs) for epochs in orders.values())
    assert len({tuple(epoch) for epochs in orders.values() for epoch in epochs}) == 15
    assert orders == warm.epoch_orders(list(reversed(selected["train"])))
    profiles[-1]["dialogue_id"] = "d0"
    with pytest.raises(ValueError, match="inventory"):
        warm.full_ids(profiles)


def test_cache_replays_each_original_dialogue_and_retains_only_turn_tokens(tmp_path):
    from openjev.research.dialogue_trainable_encoder import encode_token_lists

    actors = {("train", "same"): {"tokens": [[7]*260, [12, 13], [31]], "turn_text_ids": [0], "original_feature_ids": [8, 10, 12]},
              ("dev", "same"): {"tokens": [[7]*260, [31], [12, 13]], "turn_text_ids": [0, 2], "original_feature_ids": [8, 12, 10]}}
    special = {"cls_id": 101, "sep_id": 102, "pad_id": 0}
    encoder, expected_encoder = Encoder().eval(), Encoder().eval()
    expected = [encode_token_lists(expected_encoder, actor["tokens"], **special, trainable=False)[0].numpy() for actor in actors.values()]
    cache, index = warm.cache_dialogues(encoder, actors, special, tmp_path, Budget(tmp_path), {})
    pooled, hidden, offsets, lookup = cache
    assert np.array_equal(pooled, np.concatenate(expected))
    assert len(encoder.calls) == len(expected_encoder.calls) == 2
    assert all(torch.equal(a, b) and torch.equal(c, d) for (a, c), (b, d) in zip(encoder.calls, expected_encoder.calls, strict=True))
    assert offsets.tolist() == [0, 264, 264, 264, 528, 528, 532]
    assert hidden.shape == (532, 384) and hidden[[0, 255, 256, 263], 0].tolist() == [101, 102, 101, 102]
    assert not (hidden[:, 0] == 0).any()
    assert lookup["train", "same"][8] == 0 and lookup["dev", "same"][8] == 3
    assert index["encoder_work"]["input_texts"] == 6 and index["calls"]["encoder_forward_returns"] == 2
    assert not encoder._forward_hooks and not encoder._forward_pre_hooks
    reopened = warm.open_cache(tmp_path)
    assert np.array_equal(reopened[0], pooled) and reopened[-1] == lookup


def test_cache_cap_rejects_before_encoder_calls(tmp_path, monkeypatch):
    monkeypatch.setitem(warm.CAPS["run"], "output_bytes", 1)
    actor = {("train", "d"): {"tokens": [[5]], "turn_text_ids": [0], "original_feature_ids": [1]}}
    encoder = Encoder().eval()
    with pytest.raises(ValueError, match="before encoding"):
        warm.cache_dialogues(encoder, actor, {"cls_id": 1, "sep_id": 2, "pad_id": 0}, tmp_path, Budget(tmp_path), {})
    assert encoder.calls == [] and list(tmp_path.iterdir()) == []


def logs(values):
    p = np.asarray(values, np.float64)
    out = np.full((len(p), 12), -np.inf, np.float32)
    out[:, :p.shape[1]] = np.log(p).astype(np.float32)
    return out, np.arange(12)[None, :] < p.shape[1]


def test_raw_parity_rejects_top_tie_change_even_inside_numeric_tolerance():
    reference, mask = logs([[.4, .4, .2]])
    actual = reference.copy()
    actual[0, 1] = np.nextafter(actual[0, 1], np.float32(-np.inf))
    assert actual.argmax(1).tolist() == reference.argmax(1).tolist()
    with pytest.raises(ValueError, match="all top ties"):
        warm.parity(actual, reference, mask)
    assert warm.parity(reference, reference, mask)["passed"] is True


@pytest.mark.parametrize("corruption", ["support", "padding", "dtype", "mass", "choice"])
def test_raw_parity_failures(corruption):
    reference, mask = logs([[.4, .3, .3]])
    actual = reference.copy()
    if corruption == "support":
        actual[0, 1] = -np.inf
    elif corruption == "padding":
        actual[0, 3] = -100
    elif corruption == "dtype":
        actual = actual.astype(np.float64)
    elif corruption == "mass":
        actual[0, :3] += .1
    else:
        actual[0, 0], actual[0, 1] = actual[0, 1], actual[0, 0]
    with pytest.raises(ValueError):
        warm.parity(actual, reference, mask)


def test_strict_checkpoint_state_and_optimizer_groups():
    model = nn.Module()
    model.memory = nn.Linear(2, 3)
    model.read_output = nn.Linear(2, 3, bias=False)
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    warm.strict_state(model, snapshot)
    optimizer = warm.optimizer_for(model)
    assert optimizer.state == {}
    assert [g["lr"] for g in optimizer.param_groups] == [1e-4, 1e-3]
    assert all(g["weight_decay"] == 1e-4 for g in optimizer.param_groups)
    assert {id(p) for p in optimizer.param_groups[0]["params"]} == {id(p) for p in model.memory.parameters()}
    for broken in ({}, {**snapshot, "optimizer": {}}, {**snapshot, "memory.weight": snapshot["memory.weight"].double()},
                   {**snapshot, "memory.weight": torch.full_like(snapshot["memory.weight"], float("nan"))}):
        with pytest.raises(ValueError, match="restored"):
            warm.strict_state(model, broken)


class TinyHead(nn.Module):
    method = "pooled"

    def __init__(self):
        super().__init__()
        self.memory = nn.Linear(1, 3)
        self.read_output = nn.Linear(1, 3, bias=False)
        self.last_audit = {}

    def configuration(self):
        return {"method": self.method}

    def forward(self, token_states, pooled_turns, query, candidates, candidate_mask, lexical, none_index):
        assert len(token_states) == 2 and query.shape == (1, 384)
        self.last_audit = {"forward_attempts": 1, "forward_returns": 1, "observation_attempts": 2, "observation_returns": 2,
            "step_attempts": 2, "step_returns": 2, "state_checks": 2, "real_question_updates": 2, "attention_positions": 0}
        return self.memory.bias.log_softmax(-1).reshape(1, 1, 1, 3).expand(1, 2, 1, 3)


def tiny_fixture():
    candidate_ids = ["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:x"]
    actor = {"original_feature_ids": list(range(6)), "turn_text_ids": [0, 1], "query_text_ids": [2],
             "candidate_text_ids": [[3, 4, 5]], "candidate_ids": [candidate_ids], "query_ids": [0],
             "lexical_offset": 0, "lexical_shape": [2, 1, 3, 10]}
    actors, targets, rows = {}, {}, []
    ids = ["d0", "d1", "d2"]
    for split in ("train", "dev"):
        for i, did in enumerate(ids):
            actors[split, did] = actor
            targets[split, did] = [{"split": split, "dialogue_id": did, "source_row_index": 0,
                "time": 1, "query_position": 0, "query_index": 0, "label_index": i,
                "label_id": candidate_ids[i], "stratum_index": i}]
            if split == "dev":
                rows.append({**targets[split, did][0], "row_index": i, "candidate_ids": candidate_ids})
    vectors = np.ones((6, 384), np.float32)/np.sqrt(np.float32(384))
    cache = vectors, np.ones((12, 384), np.float32), np.arange(0, 13, 2, dtype=np.int64), {k: {i: i for i in range(6)} for k in actors}
    plan = {"selected": {s: ids for s in ("train", "dev")}, "orders": {"6901": [ids, list(reversed(ids))]},
            "loss_weights": [.5, 1., 3.]}
    return plan, actors, targets, np.zeros(60, np.float32), rows, cache


def test_tiny_adaptation_actual_batch_denominators_counts_and_unused_attention(tmp_path, monkeypatch):
    monkeypatch.setitem(warm.CONFIG, "epochs", 2)
    monkeypatch.setitem(warm.CONFIG, "batch_dialogues", 2)
    plan, actors, targets, lexical, rows, cache = tiny_fixture()
    model = TinyHead()
    untouched = model.read_output.weight.detach().clone()
    result = warm.adapt(model, "pooled-6901", plan, actors, targets, lexical, rows, cache, tmp_path, Budget(tmp_path), {})
    assert result["training_counts"]["optimizer_updates"] == 4
    assert result["training_counts"]["training_forwards"] == 6 and result["evaluation"]["counts"]["evaluation_forwards"] == 3
    assert result["training_counts"]["training_endpoints"] == 6
    assert torch.equal(untouched, model.read_output.weight)
    journal = list(warm.lines(tmp_path/"updates.jsonl"))
    assert [row["endpoint_count"] for row in journal] == [2, 1, 2, 1]
    with np.load(tmp_path/"predictions.npz", allow_pickle=False) as packet:
        assert packet["log_probs"].shape == (3, 12) and np.isneginf(packet["log_probs"][:, 3:]).all()
        assert packet["row_indices"].tolist() == [0, 1, 2]
    assert {p.name for p in tmp_path.iterdir()} == {"updates.jsonl", "weights.pt", "predictions.npz"}


def test_original_scalar_reference_keeps_original_batch_query_layout(tmp_path):
    class Memory:
        def __init__(self):
            self.audit = {}

        def __call__(self, turns, valid, query, candidates, mask, lexical, none):
            assert turns.shape == (1, 2, 384) and valid.shape == (1, 2)
            assert query.shape == (1, 1, 384) and mask.shape == (1, 1, 3)
            assert lexical.shape == (1, 2, 1, 3, 10) and none.shape == (1, 1)
            return torch.zeros(1, 2, 1, 3).log_softmax(-1)

    from study_dialogue_copy_v2 import empty_invariants

    _, actors, _, lexical, _, cache = tiny_fixture()
    inputs = warm.inputs_for(("train", "d0"), actors, lexical, cache)
    counts, audit = Counter(), empty_invariants()
    warm.untouched_forward(Memory(), inputs, Budget(tmp_path), counts, audit)
    assert counts["forward_attempts"] == counts["forward_returns"] == 1 and counts["question_updates"] == 2


def test_wrong_plan_hash_precedes_any_checkpoint_load(tmp_path, monkeypatch):
    path = tmp_path/"plan.json"
    path.write_text("{}")
    monkeypatch.setattr(warm, "checkpoint_lineage", lambda: pytest.fail("No checkpoint access before plan authentication"))
    with pytest.raises(ValueError, match="External warm plan"):
        warm.authenticate(SimpleNamespace(plan=path, plan_sha256="0"*64, out=tmp_path))


def test_live_launch_requires_exact_7200_native_parent_deadline(tmp_path, monkeypatch):
    launch = {"command": [sys.executable, "-u", str(Path(warm.__file__).resolve()), "run", "--plan", str(tmp_path/"plan.json"),
        "--plan-sha256", "a"*64, "--supervision", str(tmp_path/"launch.json"), "--out", str(tmp_path/"out")],
        "version": "dialogue-observation-supervision-v2", "cwd": str(warm.ROOT), "pid": 10, "pgid": 10, "parent_pid": 9,
        "clock_backend": "CLOCK_BOOTTIME", "cap_seconds": 7200, "started_ns": 100, "deadline_ns": 7200_000_000_100,
        "watchdog_sha256": "b"*64, "clock_source_sha256": "b"*64}
    path = tmp_path/"launch.json"
    path.write_text(json.dumps(launch))
    args = SimpleNamespace(plan=tmp_path/"plan.json", plan_sha256="a"*64, supervision=path, out=tmp_path/"out")
    monkeypatch.setattr(warm, "sha", lambda _: "b"*64)
    monkeypatch.setattr(warm.os, "getpid", lambda: 10)
    monkeypatch.setattr(warm.os, "getpgrp", lambda: 10)
    monkeypatch.setattr(warm.os, "getppid", lambda: 9)
    monkeypatch.setattr(sys, "argv", launch["command"][2:])
    clock = SimpleNamespace(backend="CLOCK_BOOTTIME", now_ns=lambda: 200)
    deadline, _ = warm.await_supervision(args, clock, 150)
    assert deadline.expires_ns == launch["deadline_ns"]
    launch["cap_seconds"] = 1800
    path.write_text(json.dumps(launch))
    with pytest.raises(ValueError, match="parent deadline"):
        warm.await_supervision(args, clock, 150)


def test_qualification_requires_all_fifteen_full_paths_before_adaptation():
    records = [{"fit_id": f"{m}-{s}", "status": "completed", "parity": {"passed": True, "endpoints": 62329},
                "rows": 62329, "counts": {"evaluation_forwards": 2363}} for s in warm.SEEDS for m in ("untouched", *warm.METHODS)]
    warm.require_qualification(records)
    for broken in (records[:-1], list(reversed(records)), [*records[:-1], records[0]]):
        with pytest.raises(ValueError, match="before any adaptation"):
            warm.require_qualification(broken)
    records[-1]["parity"]["passed"] = False
    with pytest.raises(ValueError, match="before any adaptation"):
        warm.require_qualification(records)


def test_one_parity_failure_in_run_never_constructs_optimizer(tmp_path, monkeypatch):
    plan, actors, targets, lexical, rows, cache = tiny_fixture()
    parent = {"tokenizer_ids": {"cls_id": 101, "sep_id": 102, "pad_id": 0}}
    old = tmp_path/"historical"
    path = old/"trainable_numbers-6901"
    path.mkdir(parents=True)
    expected, _ = logs([[.4, .3, .3]]*len(rows))
    np.savez(path/"predictions.npz", log_probs=expected, row_indices=np.arange(len(rows), dtype=np.int64))
    out = tmp_path/"new"
    out.mkdir()
    planfile = tmp_path/"plan.json"
    planfile.write_text("{}")
    monkeypatch.setattr(warm, "OLD_RUN", old)
    monkeypatch.setitem(warm.CONFIG, "dev_endpoints", len(rows))
    monkeypatch.setattr(warm, "authenticate", lambda _args: (plan, parent))
    monkeypatch.setattr(warm.shared, "load_inputs", lambda *_args: (actors, targets, lexical, {"train": [], "dev": rows}))
    monkeypatch.setattr(warm, "lines", lambda _: iter(rows))
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)
    monkeypatch.setattr(warm, "load_checkpoint", lambda *_args: (Encoder().eval(), TinyHead().eval().requires_grad_(False)))
    monkeypatch.setattr(warm, "cache_dialogues", lambda *_args: (cache, {"encoder_work": {}}))
    monkeypatch.setattr(warm, "paired_model", lambda *_args: TinyHead())
    monkeypatch.setattr(warm.inherited.qualified, "tensor_digest", lambda _: "a"*64)
    monkeypatch.setattr(warm, "evaluate", lambda *_args, **_kwargs: (expected.copy(), {"rows": len(rows), "counts": {"evaluation_forwards": 3}}))
    monkeypatch.setattr(warm, "optimizer_for", lambda _: pytest.fail("Optimizer created despite failed qualification"))
    calls = []

    def reject_second(*_args):
        calls.append(True)
        if len(calls) == 2:
            raise ValueError("deliberate zero-init parity failure")
        return {"passed": True, "endpoints": len(rows)}

    monkeypatch.setattr(warm, "parity", reject_second)
    with pytest.raises(ValueError, match="deliberate zero-init"):
        warm.run(SimpleNamespace(plan=planfile, plan_sha256="a"*64, out=out), Budget(out), {})
    assert len(calls) == 2 and not (out/"qualification.json").exists()
