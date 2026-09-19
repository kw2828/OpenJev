"""Synthetic public layouts and fake raw states only; no corpus/encoder calls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("token_prep_test", ROOT / "scripts/prepare_dialogue_tokens.py")
token = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(token)


def layout():
    packet = {"queries": [{"split": "train", "candidates": [1, 2, 3]}], "cohorts": {
        "train": [{"id": "d", "turns": [7, 8], "user_text": ["red", "blue"],
                   "queries": [{"label": 2, "time": 0}]}], "dev": []}}
    index = {"cohorts": {"train": [{"id": "d", "query_ids": [0], "shape": [2, 1, 3, 10]}], "dev": []}}
    public = {"train": {"d": {"turns": [{"speaker": "SYSTEM", "utterance": "Color?"},
        {"speaker": "USER", "utterance": "red"}, {"speaker": "USER", "utterance": "blue"}],
        "user_turns": [{"turn_index": 1, "previous_system_turn_index": 0},
                       {"turn_index": 2, "previous_system_turn_index": 0}]}}, "dev": {}}
    return packet, index, public


def test_public_context_once_and_no_labels_or_candidates_in_encoding():
    args = layout()
    a = token.build_contexts(*args)
    assert a[0] == ["System: Color?\nUser: red", "System: Color?\nUser: blue"]
    assert a[1].tolist() == [0, 1] and a[2]["original_feature_indices"] == [7, 8]
    args[0]["queries"][0]["candidates"] = [900, 800, 700]
    args[0]["cohorts"]["train"][0]["queries"] = [{"label": object(), "time": -1000}]
    b = token.build_contexts(*args)
    assert a[0] == b[0] and a[2] == b[2] and np.array_equal(a[1], b[1])


def test_future_text_changes_only_future_context():
    args = layout()
    before = token.build_contexts(*args)[0]
    args[0]["cohorts"]["train"][0]["user_text"][1] = "later"
    args[2]["train"]["d"]["turns"][2]["utterance"] = "later"
    after = token.build_contexts(*args)[0]
    assert before[0] == after[0] and before[1] != after[1]


def test_dedup_preserves_original_index_and_all_public_occurrences():
    args = layout()
    args[0]["cohorts"]["train"][0]["user_text"][1] = "red"
    args[0]["cohorts"]["train"][0]["turns"][1] = 7
    args[2]["train"]["d"]["turns"][2]["utterance"] = "red"
    texts, indices, index = token.build_contexts(*args)
    assert len(texts) == 1 and indices.tolist() == [0, 0]
    assert index["counts"]["train"] == {"dialogues": 1, "public_turns": 2, "public_question_steps": 2}


@pytest.mark.parametrize("kind", ["source_collision", "duplicate_text_index", "missing", "future_system", "bad_user", "test", "empty"])
def test_invalid_context_alignment(kind):
    packet, index, public = layout()
    if kind == "source_collision":
        packet["cohorts"]["train"][0]["turns"][1] = 7
    elif kind == "duplicate_text_index":
        packet["cohorts"]["train"][0]["user_text"][1] = "red"
        public["train"]["d"]["turns"][2]["utterance"] = "red"
    elif kind == "missing":
        public["train"] = {}
    elif kind == "future_system":
        public["train"]["d"]["user_turns"][0]["previous_system_turn_index"] = 2
    elif kind == "bad_user":
        packet["cohorts"]["train"][0]["user_text"][0] = "mismatch"
    elif kind == "test":
        packet["cohorts"]["test"] = []
    else:
        index["cohorts"]["train"][0]["query_ids"] = []
    with pytest.raises(ValueError):
        token.build_contexts(packet, index, public)


class FakeEncoder:
    def __init__(self, plan=None):
        self.tokenizer = SimpleNamespace(cls_token_id=101, sep_token_id=102)
        self.calls = 0

    def tokenize(self, texts):
        return [list(range(1, int(text) + 1)) for text in texts]

    def sync(self):
        pass

    def memory(self):
        return {"mps_current_allocated_bytes": 0, "mps_driver_allocated_bytes": 0}

    def tokens(self, batch, progress):
        self.calls += 1
        progress["encoder_calls_attempted"] += 1
        values = np.full((len(batch), max(map(len, batch)), 384), np.nan, np.float32)
        for j, ids in enumerate(batch):
            values[j, :len(ids)] = 1
            values[j, :len(ids), 0] = ids
            values[j, :len(ids), 1] = np.asarray(ids) % 7
        progress["encoder_calls_returned"] += 1
        return values, len(batch) * max(map(len, batch)), self.memory()


def progress():
    return {"encoder_calls_attempted": 0, "encoder_calls_returned": 0, "encoded_sequences": 0, "encoded_texts": 0}


def original_vectors(texts):
    # Independent old recipe: masked per-chunk mean, content-weighted, then L2.
    result = []
    for text in texts:
        ids = list(range(1, int(text) + 1))
        total = np.zeros(384, np.float32)
        for start in range(0, len(ids), 254):
            part = ids[start:start + 254]
            seq = [101, *part, 102]
            rows = np.ones((len(seq), 384), np.float32)
            rows[:, 0], rows[:, 1] = seq, np.asarray(seq) % 7
            total += rows.mean(0) * len(part)
        total /= len(ids)
        result.append(total / np.linalg.norm(total))
    return np.stack(result)


def test_all_tokens_uneven_chunks_priors_recover_original_vectors(tmp_path):
    texts = ["1", "255", "510"]
    backend, p = FakeEncoder(), progress()
    profile, lengths = token.old.token_profile(texts, backend, lambda: None)
    assert backend.calls == 0
    work, eq, files = token.encode_tokens(texts, lengths, backend, tmp_path, original_vectors(texts), p, lambda: None)
    token.validate_work({"work": work, "progress": p, "pooling_equivalence": eq}, lengths)
    assert all(profile[k] == v for k, v in token.profile_counts(lengths).items())
    assert p["encoded_sequences"] == 6 and p["encoder_calls_returned"] == 1
    assert work["input_tokens"] == 766 and work["retained_token_rows"] == 778
    priors, offsets = np.load(tmp_path / "priors.npy"), np.load(tmp_path / "offsets.npy")
    for i in range(3):
        assert abs(float(priors[offsets[i]:offsets[i + 1]].sum()) - 1) < 2e-7
    assert priors[offsets[1]] == np.float32(254 / (255 * 256))
    assert priors[offsets[1] + 256] == np.float32(1 / (255 * 3))
    assert eq["max_abs_error"] <= 2e-5 and set(files) == set(token.TOKEN_FILES)
    assert np.isfinite(np.load(tmp_path / "tokens.npy")).all()  # NaN padding was never retained.


def test_full_profile_includes_partial_last_batch():
    lengths = np.array([254] * 128 + [1], np.int64)
    p = token.profile_counts(lengths)
    assert p["encoder_calls"] == 2 and p["padded_token_slots"] == 128 * 256 + 3


@pytest.mark.parametrize("kind", ["nan", "zero", "changed_mean", "empty_tokens", "wrong_shape"])
def test_bad_encoding_retains_paid_prefix_and_payload(tmp_path, kind):
    class Bad(FakeEncoder):
        def tokenize(self, texts):
            return [[]] if kind == "empty_tokens" else super().tokenize(texts)
        def tokens(self, batch, p):
            v, slots, memory = super().tokens(batch, p)
            if kind == "nan":
                v[0, 0, 0] = np.nan
            elif kind == "zero":
                v[:] = 0
            elif kind == "changed_mean":
                v[:, :, 0] += 100
            elif kind == "wrong_shape":
                v = v[:, :, :5]
            return v, slots, memory
    p = progress()
    with pytest.raises(ValueError):
        token.encode_tokens(["1"], np.array([1], np.int64), Bad(), tmp_path, original_vectors(["1"]), p, lambda: None)
    assert p["encoder_calls_returned"] == (0 if kind == "empty_tokens" else 1)
    assert (tmp_path / "tokens.npy").exists()


def test_disjoint_projection_exact_admission_and_caps():
    w = {"forward_transfer_seconds": 2., "retention_seconds": 1., "padded_token_slots": 100, "retained_token_rows": 50}
    full = {"padded_token_slots": 30000, "encoder_tokens_with_special": 5000}
    result = token.projection(w, full, 20., token.CAP_BYTES)
    assert result["projected_forward_transfer_seconds"] == 600
    assert result["projected_retention_seconds"] == 100
    assert result["projected_total_seconds"] == 720 and result["encoding_permitted"]
    assert not token.projection(w, full, 20.0001, token.CAP_BYTES)["encoding_permitted"]
    assert not token.projection(w, full, 20, token.CAP_BYTES + 1)["encoding_permitted"]


@pytest.fixture
def fake_campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(token, "ROOT", tmp_path)
    prep, packet = tmp_path / "prep", tmp_path / "packet"
    prep.mkdir()
    packet.mkdir()
    texts = ["1", "255", "3"]
    np.save(packet / "features.npy", original_vectors(texts))
    np.save(prep / "context-indices.npy", np.array([0, 1, 2, 1], np.int64))
    counts = {"train": {"dialogues": 1, "public_turns": 4, "public_question_steps": 4}, "dev": {"dialogues": 0, "public_turns": 0, "public_question_steps": 0}}
    index = {"version": token.VERSION, "template": token.TEMPLATE, "width": 384, "unique_contexts": 3,
             "context_occurrences": 4, "original_feature_indices": [0, 1, 2], "counts": counts, "cohorts": {}}
    token.write(prep / "index.json", index)
    (prep / "texts.jsonl").write_text("".join(json.dumps(x) + "\n" for x in texts))
    (prep / "original-encoder-source.py").write_text("# synthetic source\n")
    plan = {"version": token.VERSION, "source_sha256": {token.HELPER: token.HELPER_SHA}, "runtime": {"synthetic": True},
        "model_files_sha256": {str(i): "a" * 64 for i in range(6)}, "device": "mps", "encoder": token.old.ENCODER,
        "revision": token.old.REVISION, "template": token.TEMPLATE, "dtype": "float32", "chunk_tokens": 254,
        "batch_size": 128, "width": 384, "capacity_unique_contexts": 512, "capacity_seconds": 120,
        "encoding_seconds": 900, "admission_seconds": 720, "disk_cap_bytes": token.CAP_BYTES, "pooling_max_abs_tolerance": token.TOLERANCE,
        "original_encoder_source_sha256": token.sha(prep / "original-encoder-source.py"),
        "inputs": {n: {"path": "packet", "completed_sha256": "b" * 64} for n in ("packet", "lexical", "data")},
        "layout_files": {n: token.sha(prep / n) for n in ("texts.jsonl", "index.json", "context-indices.npy")},
        "unique_contexts": 3, "context_occurrences": 4, "counts": counts}
    token.write(prep / "plan.json", plan)
    plan_sha = token.sha(prep / "plan.json")
    monkeypatch.setattr(token, "validate_plan", lambda *args: plan)
    monkeypatch.setattr(token.old, "model_paths", lambda *args: {})
    def args(phase):
        return SimpleNamespace(phase=phase, plan=prep / "plan.json", plan_sha256=plan_sha, out=tmp_path / phase,
            capacity=tmp_path / "capacity" if phase == "encode" else None, capacity_sha256=None)
    capargs = args("capacity")
    capsha = token.encode(capargs, backend_factory=FakeEncoder)
    fullargs = args("encode")
    fullargs.capacity_sha256 = capsha
    return plan, capargs, fullargs


def test_fake_capacity_full_encode_and_saved_only_loader(fake_campaign):
    _, capargs, args = fake_campaign
    digest = token.encode(args, backend_factory=FakeEncoder)
    index, states, offsets, priors, ids = token.authenticate_cache(args.out, digest)
    assert index["unique_contexts"] == 3 and states.shape == (267, 384)
    assert offsets.tolist() == [0, 3, 262, 267] and ids.tolist() == [0, 1, 2, 1]
    assert priors.shape == (267,)
    cap = token.read(capargs.out / "completed.json")
    assert cap["projection"]["encoding_permitted"] and cap["progress"]["encoder_calls_returned"] == 1
    with pytest.raises(FileExistsError):
        token.encode(args, backend_factory=FakeEncoder)


def reseal(path, mutation):
    receipt = token.read(path / "completed.json")
    mutation(receipt)
    (path / "completed.json").unlink()
    token.write(path / "completed.json", receipt)
    return token.sha(path / "completed.json")


@pytest.mark.parametrize("field", ["projection", "sample", "calls", "model", "parent", "pooling", "tokens", "overlength"])
def test_corrupted_capacity_rejected_before_full_encoder(fake_campaign, field):
    _, capargs, args = fake_campaign
    def mutate(r):
        if field == "projection":
            r["projection"]["projected_total_seconds"] = 1.
        elif field == "sample":
            r["unique_contexts_encoded"] -= 1
        elif field == "calls":
            r["progress"]["encoder_calls_returned"] += 1
        elif field == "model":
            r["model_files_sha256"]["0"] = "c" * 64
        elif field == "parent":
            r["lexical_completed_sha256"] = "c" * 64
        elif field == "pooling":
            r["pooling_equivalence"]["max_abs_error"] = 2.01e-5
        elif field == "tokens":
            r["work"]["input_tokens"] += 1
        else:
            r["work"]["overlength_contexts_chunked"] += 1
    args.capacity_sha256 = reseal(capargs.out, mutate)
    def forbidden(_):
        raise AssertionError("Encoder construction must not happen")
    with pytest.raises(ValueError):
        token.encode(args, backend_factory=forbidden)
    failure = token.read(args.out / "failed.json")
    assert failure["progress"]["encoder_calls_attempted"] == 0
    assert not (args.out / "completed.json").exists()


def test_actual_byte_cap_checked_before_success(fake_campaign, monkeypatch):
    _, _, args = fake_campaign
    # Lower only the terminal cap after the admitted synthetic forward has returned.
    original = token.encode_tokens
    def small_actual_cap(*a, **kw):
        result = original(*a, **kw)
        monkeypatch.setattr(token, "CAP_BYTES", 1)
        return result
    monkeypatch.setattr(token, "encode_tokens", small_actual_cap)
    with pytest.raises(ValueError, match="Actual cache"):
        token.encode(args, backend_factory=FakeEncoder)
    assert (args.out / "failed.json").exists() and not (args.out / "completed.json").exists()
    assert token.read(args.out / "failed.json")["progress"]["encoder_calls_returned"] == 1


@pytest.mark.parametrize("kind", ["parent", "model", "prior", "profile", "extra_capacity"])
def test_full_cache_rejects_resealed_corruption(fake_campaign, kind):
    _, _, args = fake_campaign
    token.encode(args, backend_factory=FakeEncoder)
    def mutate(r):
        if kind == "parent":
            r["packet_completed_sha256"] = "c" * 64
        elif kind == "model":
            r["model_files_sha256"]["0"] = "c" * 64
        elif kind == "extra_capacity":
            (args.capacity / "unbound.txt").write_text("extra")
        else:
            name = "priors.npy" if kind == "prior" else "token-profile.json"
            path = args.out / name
            if kind == "prior":
                p = np.load(path)
                p[0] *= 2
                np.save(path, p)
            else:
                p = token.read(path)
                p["padded_token_slots"] += 1
                path.unlink()
                token.write(path, p)
            r["files"][name] = {"sha256": token.sha(path), "bytes": path.stat().st_size}
    digest = reseal(args.out, mutate)
    with pytest.raises(ValueError):
        token.authenticate_cache(args.out, digest)


def test_freeze_rejects_cpu_before_input_authentication(tmp_path):
    args = SimpleNamespace(out=tmp_path / "freeze", device="cpu")
    with pytest.raises(ValueError, match="MPS"):
        token.freeze(args)
    assert token.read(args.out / "failed.json")["progress"]["encoder_calls_attempted"] == 0
