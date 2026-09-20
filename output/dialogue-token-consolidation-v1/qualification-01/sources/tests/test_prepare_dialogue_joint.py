"""Synthetic public layouts and fake encoders only; no corpus or model loads."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_dialogue_joint_test", ROOT / "scripts/prepare_dialogue_joint.py")
joint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(joint)


def layout_fixture():
    catalog = {"query_id": "q", "query_text": "Which color?", "candidates": [
        {"id": "reserved:NOT_MENTIONED", "text": "Not mentioned"},
        {"id": "reserved:DONTCARE", "text": "Any color"},
        {"id": "value:red", "text": "Color red", "value": "red"}]}
    query = {"split": "train", "id": "q", "text": 0, "candidates": [1, 2, 3],
        "candidate_ids": [x["id"] for x in catalog["candidates"]],
        "candidate_values": [x.get("value") for x in catalog["candidates"]]}
    d = {"id": "d", "turns": [4, 5], "user_text": ["red", "actually any"],
         "queries": [{"query": 0, "time": 1, "label": 1, "bin": "revision"}]}
    packet = {"queries": [query], "cohorts": {"train": [d], "dev": []}}
    index = {"cohorts": {"train": [{"id": "d", "query_ids": [0], "shape": [2, 1, 3, 10], "offset": 0}], "dev": []}}
    raw = {"dialogue_id": "d", "turns": [{"speaker": "SYSTEM", "utterance": "Which color?"},
        {"speaker": "USER", "utterance": "red"}, {"speaker": "USER", "utterance": "actually any"}],
        "user_turns": [{"turn_index": 1, "previous_system_turn_index": 0},
                       {"turn_index": 2, "previous_system_turn_index": 0}]}
    return packet, index, {"train": [catalog], "dev": []}, {"train": {"d": raw}, "dev": {}}


def test_template_all_candidates_all_public_steps_and_no_labels():
    args = layout_fixture()
    texts, indices, index = joint.build_layout(*args)
    assert texts[0] == "Not mentioned\nSystem: Which color?\nUser: red"
    assert texts[5] == "Color red\nSystem: Which color?\nUser: actually any"
    assert len(texts) == 6 and indices.tolist() == list(range(6))
    assert index["counts"]["train"]["real_candidate_steps"] == 6
    mutated = copy.deepcopy(args)
    mutated[0]["cohorts"]["train"][0]["queries"][0].update(label=999, time=999, bin="not-read")
    other = joint.build_layout(*mutated)
    assert other[0] == texts and np.array_equal(other[1], indices) and other[2] == index


def test_future_turn_mutation_cannot_change_earlier_strings():
    args = layout_fixture()
    before = joint.build_layout(*args)
    args[0]["cohorts"]["train"][0]["user_text"][1] = "different future"
    args[3]["train"]["d"]["turns"][2]["utterance"] = "different future"
    after = joint.build_layout(*args)
    assert after[0][:3] == before[0][:3]
    assert after[0][3:] != before[0][3:]


def test_candidate_order_follows_catalog_and_only_padding_is_minus_one():
    packet, index, catalogs, public = layout_fixture()
    q = copy.deepcopy(packet["queries"][0])
    q.update(id="q2", text=6, candidates=[7, 8, 9, 10],
        candidate_ids=["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:blue", "value:red"],
        candidate_values=[None, None, "blue", "red"])
    packet["queries"].append(q)
    packet["cohorts"]["train"][0]["queries"].append({"query": 1})
    catalogs["train"].append({"query_id": "q2", "query_text": "Which alternate?", "candidates": [
        {"id": k, "text": "alt " + k, "value": v} for k, v in zip(q["candidate_ids"], q["candidate_values"], strict=True)]})
    index["cohorts"]["train"][0].update(query_ids=[0, 1], shape=[2, 2, 4, 10])
    texts, indices, layout = joint.build_layout(packet, index, catalogs, public)
    block = indices.reshape(2, 2, 4)
    assert (block[:, 0, 3] == -1).all() and (block[:, 1] >= 0).all()
    assert texts[block[0, 1, 2]].startswith("alt value:blue\n")
    assert texts[block[0, 1, 3]].startswith("alt value:red\n")
    assert layout["counts"]["train"]["real_candidate_steps"] == 14


@pytest.mark.parametrize("mutation", ["missing", "future_system", "catalog", "text_index", "test_split", "empty"])
def test_invalid_public_alignment_rejected(mutation):
    packet, index, catalogs, public = layout_fixture()
    if mutation == "missing":
        public["train"] = {}
    elif mutation == "future_system":
        public["train"]["d"]["user_turns"][0]["previous_system_turn_index"] = 2
    elif mutation == "catalog":
        catalogs["train"][0]["candidates"][2]["id"] = "value:blue"
    elif mutation == "text_index":
        packet["queries"][0]["candidates"][0] = 0
    elif mutation == "test_split":
        packet["cohorts"]["test"] = []
    else:
        catalogs["train"][0]["candidates"][2]["text"] = ""
    with pytest.raises(ValueError):
        joint.build_layout(packet, index, catalogs, public)


class FakeEncoder:
    def __init__(self, plan=None):
        self.tokenizer = SimpleNamespace(cls_token_id=101, sep_token_id=102)
        self.calls = 0
        self.syncs = 0

    def tokenize(self, texts):
        return [list(range(1, int(text) + 1)) for text in texts]

    def sync(self):
        self.syncs += 1

    def memory(self):
        return {"mps_current_allocated_bytes": 0, "mps_driver_allocated_bytes": 0}

    def pooled(self, batch):
        self.calls += 1
        values = np.ones((len(batch), 384), np.float32)
        values[:, 0] = [sum(x) / len(x) for x in batch]
        return values, len(batch) * max(map(len, batch)), self.memory()


def progress():
    return {"encoder_calls_attempted": 0, "encoder_calls_returned": 0, "encoded_sequences": 0, "encoded_texts": 0}


def test_chunking_uses_every_token_specials_and_original_weighted_pooling(tmp_path):
    backend, work = FakeEncoder(), progress()
    profile, lengths = joint.token_profile(["1", "255", "510"], backend, lambda: None)
    assert backend.calls == 0
    assert lengths.tolist() == [1, 255, 510]
    assert profile["input_tokens"] == 766 and profile["encoder_sequences"] == 6
    assert profile["encoder_tokens_with_special"] == 778 and profile["padded_token_slots"] == 1536
    result = joint.encode_texts(["1", "255", "510"], backend, tmp_path / "v.npy", work, lambda: None)
    assert work["encoder_calls_attempted"] == work["encoder_calls_returned"] == 1
    assert work["encoded_texts"] == 3 and work["encoded_sequences"] == 6
    assert result["input_tokens"] == 766 and result["truncated_tokens"] == 0
    saved = np.load(tmp_path / "v.npy")
    first = (101 + sum(range(1, 255)) + 102) / 256
    second = (101 + 255 + 102) / 3
    vector = np.ones(384, np.float32)
    vector[0] = np.float32(254 * np.float32(first) + np.float32(second)) / 255
    vector /= np.linalg.norm(vector)
    np.testing.assert_allclose(saved[1], vector, atol=2e-7, rtol=0)
    assert backend.syncs >= 2


def test_sequential_batch_padding_profile_includes_partial_last_batch(tmp_path):
    texts = ["254"] * 128 + ["1"]
    backend, work = FakeEncoder(), progress()
    profile, _ = joint.token_profile(texts, backend, lambda: None)
    actual = joint.encode_texts(texts, backend, tmp_path / "v.npy", work, lambda: None)
    assert profile["padded_token_slots"] == actual["padded_token_slots"] == 128 * 256 + 3
    assert backend.calls == 2 and profile["encoder_calls"] == 2


@pytest.mark.parametrize("kind", ["nan", "zero", "empty_tokens"])
def test_bad_encoder_results_retained_with_actual_call_prefix(tmp_path, kind):
    class Bad(FakeEncoder):
        def tokenize(self, texts):
            return [[]] if kind == "empty_tokens" else super().tokenize(texts)
        def pooled(self, batch):
            v, slots, memory = super().pooled(batch)
            v[:] = np.nan if kind == "nan" else 0
            return v, slots, memory
    work = progress()
    with pytest.raises(ValueError):
        joint.encode_texts(["1"], Bad(), tmp_path / "v.npy", work, lambda: None)
    assert work["encoder_calls_returned"] == (0 if kind == "empty_tokens" else 1)
    assert (tmp_path / "v.npy").exists()


def test_fixed_projection_arithmetic_and_boundaries():
    work = {"encoder_seconds": 2., "padded_token_slots": 100}
    full = {"padded_token_slots": 35000}
    good = joint.capacity_projection(work, full, 20., joint.DISK_CAP)
    assert good["projected_total_seconds"] == 720. and good["encoding_permitted"]
    assert not joint.capacity_projection(work, full, 20.001, joint.DISK_CAP)["encoding_permitted"]
    assert not joint.capacity_projection(work, full, 20., joint.DISK_CAP + 1)["encoding_permitted"]


def test_attempt_failure_exclusivity_and_late_completion_demotion(tmp_path):
    out = tmp_path / "attempt"
    with pytest.raises(TimeoutError, match="synthetic cap"), joint.attempt(out, "capacity", {"synthetic": True}) as (folder, _, work, _):
        work["encoder_calls_attempted"] = 1
        joint.write(folder / "completed.json", {"status": "completed"})
        raise TimeoutError("synthetic cap")
    assert (out / "late-completion.json").exists() and not (out / "completed.json").exists()
    assert joint.read(out / "failed.json")["progress"]["encoder_calls_attempted"] == 1
    with pytest.raises(FileExistsError), joint.attempt(out, "capacity", {}):
        pass


def test_source_hash_corruption_rejected_before_payload_use(tmp_path):
    path = tmp_path / "member"
    path.write_text("first")
    digest = joint.sha(path)
    joint.bind(path, digest)
    path.write_text("different")
    with pytest.raises(ValueError, match="Hash mismatch"):
        joint.bind(path, digest)


def test_existing_alarm_not_cancelled(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(joint.signal, "getitimer", lambda *args: (10., 0.))
    monkeypatch.setattr(joint.signal, "signal", lambda *args: calls.append(args))
    monkeypatch.setattr(joint.signal, "setitimer", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="Existing process timer"), joint.attempt(tmp_path / "attempt", "capacity", {}):
        pass
    assert calls == []


def test_empty_encoder_text_rejected_before_output(tmp_path):
    with pytest.raises(ValueError, match="Empty encoding input"):
        joint.encode_texts([""], FakeEncoder(), tmp_path / "v.npy", progress(), lambda: None)
    assert not (tmp_path / "v.npy").exists()


def test_capacity_refusal_prevents_encoder_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(joint, "ROOT", tmp_path)
    plan = {"source_sha256": {}, "runtime": {}, "unique_texts": 600}
    monkeypatch.setattr(joint, "validate_plan", lambda *args: plan)
    monkeypatch.setattr(joint, "sources", lambda *args: None)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")
    (tmp_path / "original-encoder-source.py").write_text("# synthetic")
    cap = {"phase": "capacity", "plan_sha256": "p", "runtime": {}, "source_sha256": {}, "no_retry": True, "wall_seconds": 1.,
           "unique_texts_encoded": 512, "projection": {"encoding_permitted": False}}
    monkeypatch.setattr(joint, "authenticate", lambda *args: cap)
    args = SimpleNamespace(phase="encode", plan=plan_path, plan_sha256="p", out=tmp_path / "encode",
                           capacity=tmp_path / "capacity", capacity_sha256="c")
    def forbidden(_):
        raise AssertionError("No model construction allowed")
    with pytest.raises(ValueError, match="Capacity denied"):
        joint.encode(args, backend_factory=forbidden)
    assert joint.read(args.out / "failed.json")["progress"]["encoder_calls_attempted"] == 0


def test_safe_members_reject_escape(tmp_path):
    with pytest.raises(ValueError):
        joint.safe(tmp_path, "../outside")


def test_runtime_metadata_json_safe(monkeypatch):
    monkeypatch.setattr(joint.importlib.metadata, "version", lambda _: "synthetic-version")
    value = joint.runtime()
    assert all(type(x) is str for x in value.values())
    json.dumps(value, allow_nan=False)
