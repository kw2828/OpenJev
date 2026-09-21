"""Artificial CPU modules/checkpoints only; never load pretrained assets."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from study_dialogue_copy_v2 import MonitoredCopyMemoryV2

from openjev.research.dialogue_calibration_inference import (
    collect_endpoints,
    file_digest,
    load_final_checkpoint,
    tensor_digest,
)
from openjev.research.dialogue_trainable_encoder import (
    DONTCARE,
    NONE,
    build_actor,
    encode_token_lists,
)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(40, 8)
        self.projection = nn.Linear(8, 8)
        self.dropout = nn.Dropout(0.8)
        self.calls = []
        self.fail_on_call = None

    def forward(self, input_ids, attention_mask):
        self.calls.append({"grad_enabled": torch.is_grad_enabled(), "training": self.training,
                           "shape": tuple(input_ids.shape)})
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("Artificial encoder failure")
        return SimpleNamespace(last_hidden_state=self.dropout(self.projection(self.embedding(input_ids))))


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def actor_payload(split="train"):
    return {"split": split, "source_split": split, "analysis_role": "calibration" if split == "train" else "qualification",
            "dialogue_id": "synthetic-dialogue", "query_ids": [4, 9], "user_turn_indices": [1, 3],
            "tokens": [[1, 2, 3], [4, 5], [6, 7], [8], [9], [10], [11], [12], [13]],
            "turn_text_ids": [0, 1], "query_text_ids": [2, 3],
            "candidate_text_ids": [[4, 5, 6], [7, 4, 5, 8]],
            "candidate_ids": [[NONE, DONTCARE, "value:X"], ["value:Y", NONE, DONTCARE, "value:Z"]],
            "lexical": torch.zeros(2, 2, 4, 10), "lexical_shape": [2, 2, 4, 10]}


def endpoints(payload):
    return [{"split": payload["split"], "dialogue_id": payload["dialogue_id"], "row_index": 8,
             "source_row_index": 1, "time": 1, "turn_index": 3, "query_position": 1, "query_index": 9,
             "label_index": 0, "label_id": "value:Y", "candidate_ids": payload["candidate_ids"][1]},
            {"split": payload["split"], "dialogue_id": payload["dialogue_id"], "row_index": 2,
             "source_row_index": 0, "time": 0, "turn_index": 1, "query_position": 0, "query_index": 4,
             "label_index": 2, "label_id": "value:X", "candidate_ids": payload["candidate_ids"][0]}]


def setup_restore(tmp_path, arm="frozen_original"):
    torch.manual_seed(521)
    encoder = TinyEncoder()
    memory = MonitoredCopyMemoryV2("scalar", input_dim=8, projection_dim=4, hidden_dim=6)
    final_encoder, final_memory = copy.deepcopy(encoder), copy.deepcopy(memory)
    with torch.no_grad():
        final_memory.head.weight.add_(0.2)
        if arm.startswith("trainable_"):
            final_encoder.embedding.weight.add_(0.3)
    state = {"memory": final_memory.state_dict()}
    if arm.startswith("trainable_"):
        state["encoder"] = final_encoder.state_dict()
    checkpoint = tmp_path / "weights.pt"
    torch.save(state, checkpoint)
    snapshot = tmp_path / "fake-snapshot"
    snapshot.mkdir()
    files = {}
    for name in ("config.json", "model.safetensors"):
        path = snapshot/name
        path.write_bytes(b"ARTIFICIAL - supplied module factory, never decoded")
        files[name] = {"path": str(path), "sha256": file_digest(path)}
    calls = []
    def encoder_factory(path, **kwargs):
        calls.append({"path": path, **kwargs})
        return copy.deepcopy(encoder)
    kwargs = {"arm": arm, "checkpoint_sha256": file_digest(checkpoint), "snapshot": snapshot,
              "model_files": files, "tokenizer_ids": {"cls_id": 30, "sep_id": 31, "pad_id": 0},
              "expected_encoder_sha256": tensor_digest(final_encoder), "encoder_factory": encoder_factory,
              "memory_factory": lambda: copy.deepcopy(memory), "encoder_device": "cpu"}
    return checkpoint, kwargs, state, final_encoder, final_memory, calls


@pytest.mark.parametrize("arm", ("frozen_original", "frozen_numbers", "trainable_original", "trainable_numbers"))
def test_strict_restore_matches_direct_v2_path_without_optimizer(tmp_path, monkeypatch, arm):
    checkpoint, kwargs, _, encoder, memory, calls = setup_restore(tmp_path, arm)
    load = torch.load
    load_calls = []
    def observed_load(path, **options):
        load_calls.append(options)
        return load(path, **options)
    def forbidden_optimizer(*_args, **_kwargs):
        raise AssertionError("Optimizer construction is forbidden")
    monkeypatch.setattr(torch, "load", observed_load)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden_optimizer)
    route = load_final_checkpoint(checkpoint, **kwargs)
    assert load_calls == [{"weights_only": True, "map_location": "cpu"}]
    assert calls == [{"path": str(kwargs["snapshot"]), "local_files_only": True, "trust_remote_code": False,
                      "use_safetensors": True, "attn_implementation": "eager", "dtype": torch.float32}]
    assert all(not module.training for module in (route.encoder, route.memory))
    assert all(not p.requires_grad and p.grad is None for m in (route.encoder, route.memory) for p in m.parameters())
    payload = actor_payload()
    with torch.no_grad():
        vectors, work = encode_token_lists(encoder.eval(), payload["tokens"], **kwargs["tokenizer_ids"],
                                           trainable=False, chunk_tokens=254, chunk_batch_size=32)
        actor, none = build_actor(vectors, **{k: payload[k] for k in ("turn_text_ids", "query_text_ids", "candidate_text_ids", "candidate_ids", "lexical")}, memory_device="cpu")
        memory.eval().begin_batch(actor, [{"layout": {"shape": payload["lexical_shape"]}}])
        expected = memory(*actor, none_index=none)
    result = route.forward_public(payload)
    assert torch.equal(result.log_probs, expected)
    assert result.encoder_work == work and result.invariants == memory.audit
    assert result.log_probs.shape == (1, 2, 2, 4)
    assert result.invariants["real_question_updates"] == 4
    assert result.invariants["advance_calls"] == 2
    assert result.invariants["forward_calls"] == result.invariants["forward_returned"] == 1
    assert all(c["grad_enabled"] is False and c["training"] is False for c in route.encoder.calls)
    observed = route.snapshot()
    assert observed["counts"]["encoder_forward_attempts"] == observed["counts"]["encoder_forward_returns"] == work["encoder_calls"]
    assert observed["encoder_attempted_work"] == observed["encoder_returned_work"]
    assert observed["synthetic_injection"] is True and observed["optimizer_created"] is False
    assert observed["restored_sha256"] == {"encoder": tensor_digest(encoder), "memory": tensor_digest(memory)}
    assert route.verify_parameters_unchanged() == observed["restored_sha256"]


@pytest.mark.parametrize("corruption", ("extra_outer", "missing_memory", "unexpected_encoder", "missing_tensor", "shape", "dtype", "nan"))
def test_checkpoint_schema_rejects_instead_of_partial_or_cast_loading(tmp_path, corruption):
    checkpoint, kwargs, state, _, _, _ = setup_restore(tmp_path)
    if corruption == "extra_outer":
        state["optimizer"] = {}
    elif corruption == "missing_memory":
        state = {}
    elif corruption == "unexpected_encoder":
        state["encoder"] = {}
    else:
        key = next(iter(state["memory"]))
        if corruption == "missing_tensor":
            del state["memory"][key]
        elif corruption == "shape":
            state["memory"][key] = state["memory"][key].flatten()[:1]
        elif corruption == "dtype":
            state["memory"][key] = state["memory"][key].double()
        else:
            state["memory"][key].fill_(float("nan"))
    torch.save(state, checkpoint)
    kwargs["checkpoint_sha256"] = file_digest(checkpoint)
    with pytest.raises(ValueError, match="Exact|Finite"):
        load_final_checkpoint(checkpoint, **kwargs)


def test_trained_checkpoint_requires_encoder_and_its_recorded_digest(tmp_path):
    checkpoint, kwargs, state, _, _, _ = setup_restore(tmp_path, "trainable_numbers")
    kwargs["expected_encoder_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Published final encoder digest"):
        load_final_checkpoint(checkpoint, **kwargs)
    del state["encoder"]
    torch.save(state, checkpoint)
    kwargs["checkpoint_sha256"] = file_digest(checkpoint)
    with pytest.raises(ValueError, match="Exact final checkpoint fields"):
        load_final_checkpoint(checkpoint, **kwargs)


@pytest.mark.parametrize("changed", ("checkpoint", "asset"))
def test_hash_authentication_precedes_deserialization_and_factory(tmp_path, monkeypatch, changed):
    checkpoint, kwargs, _, _, _, calls = setup_restore(tmp_path)
    if changed == "checkpoint":
        checkpoint.write_bytes(b"changed checkpoint")
    else:
        (kwargs["snapshot"]/"config.json").write_bytes(b"changed asset")
    def prohibited(*_args, **_kwargs):
        raise AssertionError("Unverified deserialization")
    monkeypatch.setattr(torch, "load", prohibited)
    with pytest.raises(ValueError, match="bytes|asset hash"):
        load_final_checkpoint(checkpoint, **kwargs)
    assert calls == []


def test_production_cpu_fallback_is_not_permitted(tmp_path):
    checkpoint, kwargs, *_ = setup_restore(tmp_path)
    kwargs["encoder_factory"] = kwargs["memory_factory"] = None
    with pytest.raises(ValueError, match="Production MPS without fallback"):
        load_final_checkpoint(checkpoint, **kwargs)


@pytest.mark.parametrize("field", ("label", "label_index", "targets", "stratum", "previous_gold", "policy_oracle", "temperature"))
def test_evaluator_and_temperature_fields_rejected_before_encoder(tmp_path, field):
    checkpoint, kwargs, *_ = setup_restore(tmp_path)
    route = load_final_checkpoint(checkpoint, **kwargs)
    payload = actor_payload()
    payload[field] = 1
    with pytest.raises(ValueError, match="Public actor whitelist"):
        route.forward_public(payload)
    assert route.snapshot()["counts"]["encoder_forward_attempts"] == 0


def test_roles_preserve_split_and_reject_cross_role_mix(tmp_path):
    checkpoint, kwargs, *_ = setup_restore(tmp_path)
    route = load_final_checkpoint(checkpoint, **kwargs)
    dev = actor_payload("dev")
    result = route.forward_public(dev)
    assert result.split == "dev" and result.analysis_role == "qualification"
    dev["analysis_role"] = "calibration"
    with pytest.raises(ValueError, match="Separate TRAIN"):
        route.forward_public(dev)
    dev["analysis_role"], dev["source_split"] = "qualification", "train"
    with pytest.raises(ValueError, match="Original source split preserved"):
        route.forward_public(dev)


def test_endpoint_targets_are_separate_and_never_change_full_trajectory(tmp_path):
    checkpoint, kwargs, *_ = setup_restore(tmp_path)
    route = load_final_checkpoint(checkpoint, **kwargs)
    payload = actor_payload()
    result = route.forward_public(payload)
    before = result.log_probs.clone()
    rows = endpoints(payload)
    first = collect_endpoints(result, rows)
    assert first["row_indices"].tolist() == [8, 2]
    assert first["label_indices"].tolist() == [0, 2]
    assert first["log_probs"].dtype == torch.float32 and first["log_probs"].shape == (2, 12)
    assert torch.equal(first["log_probs"][0, :4], before[0, 1, 1])
    assert torch.isneginf(first["log_probs"][0, 4:]).all()
    rows[0]["label_index"], rows[0]["label_id"] = 2, DONTCARE
    changed = collect_endpoints(result, rows)
    assert torch.equal(changed["log_probs"], first["log_probs"])
    assert changed["label_indices"].tolist() == [2, 2]
    assert torch.equal(result.log_probs, before)
    assert route.snapshot()["counts"]["public_forward_returns"] == 1


@pytest.mark.parametrize("field,value", (("split", "dev"), ("dialogue_id", "wrong"), ("query_index", 4),
                                         ("time", 2), ("turn_index", 1), ("label_id", DONTCARE),
                                         ("label_index", True), ("analysis_role", "qualification")))
def test_endpoint_identity_errors_rejected(tmp_path, field, value):
    checkpoint, kwargs, *_ = setup_restore(tmp_path)
    route = load_final_checkpoint(checkpoint, **kwargs)
    payload = actor_payload()
    result = route.forward_public(payload)
    rows = endpoints(payload)
    rows[0][field] = value
    with pytest.raises(ValueError):
        collect_endpoints(result, rows)


def test_duplicate_endpoint_and_candidate_reordering_rejected(tmp_path):
    checkpoint, kwargs, *_ = setup_restore(tmp_path)
    route = load_final_checkpoint(checkpoint, **kwargs)
    payload = actor_payload()
    result = route.forward_public(payload)
    rows = endpoints(payload)
    with pytest.raises(ValueError, match="Unique endpoint"):
        collect_endpoints(result, [rows[0], rows[0]])
    rows[0]["candidate_ids"] = list(reversed(rows[0]["candidate_ids"]))
    with pytest.raises(ValueError, match="Canonical candidate order"):
        collect_endpoints(result, rows)


def test_failed_second_encoder_chunk_preserves_attempt_and_return_work(tmp_path):
    checkpoint, kwargs, *_ = setup_restore(tmp_path)
    route = load_final_checkpoint(checkpoint, **kwargs)
    route.encoder.fail_on_call = 2
    payload = actor_payload()
    payload["tokens"][0] = [1] * (254 * 33)
    with pytest.raises(RuntimeError, match="Artificial encoder failure"):
        route.forward_public(payload)
    witness = route.snapshot()
    assert witness["counts"]["encoder_forward_attempts"] == 2
    assert witness["counts"]["encoder_forward_returns"] == 1
    assert witness["counts"]["encoding_returns"] == witness["counts"]["memory_forward_attempts"] == 0
    assert witness["encoder_returned_work"]["padded_attention_positions"] == 32 * 256**2
    assert witness["encoder_attempted_work"]["padded_attention_positions"] > 32 * 256**2
    assert witness["encoder_work"] == {} and witness["invariants"]["forward_calls"] == 0


def test_memory_failure_keeps_paid_encoder_and_partial_state_monitor(tmp_path):
    checkpoint, kwargs, *_ = setup_restore(tmp_path)
    route = load_final_checkpoint(checkpoint, **kwargs)
    calls = 0
    def fail_second(_module, _args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Artificial memory failure")
    route.memory.head.register_forward_pre_hook(fail_second)
    with pytest.raises(RuntimeError, match="Artificial memory failure"):
        route.forward_public(actor_payload())
    witness = route.snapshot()
    assert witness["counts"]["encoding_returns"] == witness["counts"]["memory_forward_attempts"] == 1
    assert witness["counts"]["memory_forward_returns"] == witness["counts"]["public_forward_returns"] == 0
    assert witness["encoder_work"]["encoder_calls"] == 1
    assert witness["invariants"]["advance_calls"] == 2 and witness["invariants"]["advance_returned"] == 1
    assert witness["invariants"]["forward_calls"] == 1 and witness["invariants"]["forward_returned"] == 0


def test_repeated_dialogues_reset_state_and_parameter_mutation_is_detected(tmp_path):
    checkpoint, kwargs, *_ = setup_restore(tmp_path)
    route = load_final_checkpoint(checkpoint, **kwargs)
    first = route.forward_public(actor_payload())
    second = route.forward_public(actor_payload())
    assert torch.equal(first.log_probs, second.log_probs)
    assert route.snapshot()["invariants"]["real_question_updates"] == 8
    with torch.no_grad():
        route.memory.head.weight.add_(1)
    with pytest.raises(ValueError, match="Inference changed restored state"):
        route.verify_parameters_unchanged()
