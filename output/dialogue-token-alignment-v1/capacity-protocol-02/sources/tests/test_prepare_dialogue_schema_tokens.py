"""Synthetic public schemas and fake raw tokens only; no corpus/model calls."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("schema_preparation_test", ROOT / "scripts/prepare_dialogue_schema_tokens.py")
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


def layout():
    prefix = "Service: Trips\nSlot: Seating"
    ids = ["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:None", "value:true"]
    values = [None, None, "None", "true"]
    texts = [prefix + "\nValue: NOT_MENTIONED (no constraint stated)", prefix + "\nValue: DONTCARE (no preference)",
             prefix + "\nValue: None", prefix + "\nValue: true"]
    catalog = [{"query_id": "q", "service": "Trips", "slot": "seat", "service_description": "Trips", "slot_description": "Seating",
                "query_text": prefix, "candidates": [{"id": i, "value": v, "text": t} for i, v, t in zip(ids, values, texts, strict=True)]}]
    packet = {"queries": [{"split": "train", "id": "q", "service": "Trips", "slot": "seat", "text": 0,
                           "candidates": [1, 2, 3, 4], "candidate_ids": ids, "candidate_values": values}],
              "cohorts": {"train": [{"label": 2}], "dev": [{"label": 3}]}}
    return packet, catalog


def test_exact_full_candidate_strings_and_query_retained_independent_of_labels():
    packet, catalog = layout()
    before = p.build_layout(packet, catalog, [0])
    assert before[0][0] == "Service: Trips\nSlot: Seating"
    assert before[0][3].endswith("Value: None")
    assert before[1]["queries"][0]["candidate_token_ids"] == [1, 2, 3, 4]
    assert before[1]["original_feature_indices"] == [0, 1, 2, 3, 4]
    packet["cohorts"] = {"poison": object()}
    packet["queries"][0]["current_label"] = object()
    assert p.build_layout(packet, catalog, [0]) == before


def test_text_dedup_preserves_every_schema_mapping():
    packet, catalog = layout()
    packet["queries"].append({**copy.deepcopy(packet["queries"][0]), "id": "q2"})
    catalog.append({**copy.deepcopy(catalog[0]), "query_id": "q2"})
    texts, index = p.build_layout(packet, catalog, [0, 1])
    assert len(texts) == 5 and index["candidate_occurrences"] == 8
    assert index["queries"][0]["candidate_token_ids"] == index["queries"][1]["candidate_token_ids"]


@pytest.mark.parametrize("bad", ["dev", "bool_id", "bool_feature", "changed_prefix", "changed_value", "candidate_reorder", "collision"])
def test_schema_identity_corruptions_rejected(bad):
    packet, catalog = layout(); ids = [0]
    if bad == "dev": packet["queries"][0]["split"] = "dev"
    elif bad == "bool_id": ids = [False]
    elif bad == "bool_feature": packet["queries"][0]["text"] = False
    elif bad == "changed_prefix": catalog[0]["query_text"] += " extra"
    elif bad == "changed_value": catalog[0]["candidates"][2]["text"] += " changed"
    elif bad == "candidate_reorder": packet["queries"][0]["candidate_ids"].reverse()
    else: packet["queries"][0]["candidates"][0] = 0
    with pytest.raises(ValueError): p.build_layout(packet, catalog, ids)


def test_scope_reads_no_targets_and_keeps_all_split_schemas():
    rows = [{"row_index": i, "split": "train", "admission": "admitted", "query_index": i % 2,
             "current_label_index": object()} for i in range(4)]
    assert p.query_scope(rows, [0, 2], [1, 3]) == [0, 1]
    rows[0]["current_label_index"] = -100
    assert p.query_scope(rows, [0, 2], [1, 3]) == [0, 1]
    with pytest.raises(ValueError, match="Uncovered"): p.query_scope(rows, [0, 2], [1])
    with pytest.raises(ValueError, match="Disjoint"): p.query_scope(rows, [0, 1], [1, 2, 3])
    rows[0]["query_index"] = False
    with pytest.raises(ValueError, match="Strict supplied"): p.query_scope(rows, [0, 2], [1, 3])


class FakeEncoder:
    def __init__(self):
        self.tokenizer = SimpleNamespace(cls_token_id=101, sep_token_id=102)
    def tokenize(self, texts): return [list(range(1, int(s) + 1)) for s in texts]
    def sync(self): pass
    def tokens(self, sequences, progress):
        progress["encoder_calls_attempted"] += 1
        output = np.full((len(sequences), max(map(len, sequences)), 384), np.nan, np.float32)
        for i, ids in enumerate(sequences):
            output[i, :len(ids)] = 1
            output[i, :len(ids), 0] = ids
        progress["encoder_calls_returned"] += 1
        return output, len(sequences) * max(map(len, sequences)), {"mps_current_allocated_bytes": 0, "mps_driver_allocated_bytes": 0}


def original_vectors(texts):
    vectors = []
    for text in texts:
        ids = list(range(1, int(text) + 1)); pooled = np.zeros(384, np.float32)
        for start in range(0, len(ids), 254):
            part = ids[start:start + 254]; sequence = [101, *part, 102]
            values = np.ones((len(sequence), 384), np.float32); values[:, 0] = sequence
            pooled += values.mean(0) * len(part)
        pooled /= len(ids)
        vectors.append(pooled / np.linalg.norm(pooled))
    return np.stack(vectors)


def test_unequal_chunk_prior_matches_original_pooling_and_excludes_padding(tmp_path):
    texts = ["1", "255", "510"]
    progress = {"encoder_calls_attempted": 0, "encoder_calls_returned": 0, "encoded_sequences": 0, "encoded_texts": 0}
    profile, lengths = p.old.token_profile(texts, FakeEncoder(), lambda: None)
    work, equivalence, _ = p.tokens.encode_tokens(texts, lengths, FakeEncoder(), tmp_path, original_vectors(texts), progress, lambda: None)
    p.tokens.validate_work({"work": work, "pooling_equivalence": equivalence, "progress": progress}, lengths)
    assert profile["encoder_sequences"] == 6 and progress["encoder_calls_returned"] == 1
    assert np.isfinite(np.load(tmp_path / "tokens.npy")).all()
    priors = np.load(tmp_path / "priors.npy"); offsets = np.load(tmp_path / "offsets.npy")
    assert priors[offsets[1]] == np.float32(254 / (255 * 256))
    assert priors[offsets[1] + 256] == np.float32(1 / (255 * 3))
    assert equivalence["max_abs_error"] <= 2e-5


def test_external_plan_mismatch_precedes_encoder_and_preserves_failure(tmp_path):
    plan = tmp_path / "plan.json"; plan.write_text("{}")
    calls = []
    def forbidden(_): calls.append(True); raise AssertionError("encoder")
    out = tmp_path / "failed"
    with pytest.raises(ValueError, match="Hash mismatch"):
        p.encode(SimpleNamespace(plan=plan, plan_sha256="0" * 64, out=out), backend_factory=forbidden)
    assert not calls and (out / "failed.json").exists()
    assert json.loads((out / "failed.json").read_text())["progress"]["encoder_calls_attempted"] == 0
    with pytest.raises(FileExistsError):
        p.encode(SimpleNamespace(plan=plan, plan_sha256="0" * 64, out=out), backend_factory=forbidden)


def test_partial_and_late_completion_are_preserved(tmp_path):
    out = tmp_path / "failure"
    with pytest.raises(RuntimeError, match="after return"), p.attempt(out, "encode", {"pin": "fixed"}) as (folder, _, progress, _):
        progress["encoder_calls_returned"] = 1
        (folder / "tokens.npy").write_bytes(b"partial")
        p.write(folder / "completed.json", {"status": "completed"})
        raise RuntimeError("after return")
    failed = p.read(out / "failed.json")
    assert failed["progress"]["encoder_calls_returned"] == 1
    assert (out / "tokens.npy").read_bytes() == b"partial"
    assert (out / "late-completion.json").exists() and not (out / "completed.json").exists()


def test_secondary_receipt_failure_does_not_replace_original(tmp_path, monkeypatch):
    original = p.write
    def broken(path, value):
        if Path(path).name == "failed.json": raise OSError("disk")
        return original(path, value)
    monkeypatch.setattr(p, "write", broken)
    with pytest.raises(RuntimeError, match="original") as caught, p.attempt(tmp_path / "failure", "encode", {}):
        raise RuntimeError("original")
    assert any("disk" in s for s in caught.value.__notes__)


def synthetic_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "ROOT", tmp_path)
    monkeypatch.setattr(p, "HELPER", "helper.py")
    (tmp_path / "helper.py").write_text("# synthetic source")
    helper_sha = p.sha(tmp_path / "helper.py")
    monkeypatch.setattr(p, "HELPER_SHA", helper_sha)
    monkeypatch.setattr(p, "SOURCES", ("helper.py",))
    monkeypatch.setattr(p, "QUERY_COUNT", 1)
    out = tmp_path / "cache"; out.mkdir()
    (out / "sources").mkdir(); shutil_path = out / "sources/helper.py"
    shutil_path.write_bytes((tmp_path / "helper.py").read_bytes())
    lengths = np.asarray([1, 255], np.int64)
    profile, _ = p.old.token_profile(["1", "255"], FakeEncoder(), lambda: None)
    progress = {"encoder_calls_attempted": 0, "encoder_calls_returned": 0, "encoded_sequences": 0, "encoded_texts": 0}
    work, equivalence, _ = p.tokens.encode_tokens(["1", "255"], lengths, FakeEncoder(), out, original_vectors(["1", "255"]), progress, lambda: None)
    index = {"query_count": 1, "unique_texts": 2, "candidate_occurrences": 1, "original_feature_indices": [0, 1], "queries": []}
    p.write(out / "index.json", index); p.write(out / "token-profile.json", profile)
    np.save(out / "text-token-counts.npy", lengths, allow_pickle=False)
    (out / "original-encoder-source.py").write_text("# original")
    p.write(out / "started.json", {"status": "started"})
    feature_dir = tmp_path / "features"; feature_dir.mkdir(); np.save(feature_dir / "features.npy", original_vectors(["1", "255"]))
    model_files = {name: "a" * 64 for name in p.old.MODEL_FILES}
    plan = {"version": p.VERSION, "encoder": p.old.ENCODER, "revision": p.old.REVISION,
            "device": "mps", "dtype": "float32", "width": 384, "chunk_tokens": 254, "batch_size": 128,
            "wall_cap_seconds": p.WALL_CAP, "disk_cap_bytes": p.DISK_CAP, "pooling_max_abs_tolerance": 2e-5,
            "no_retry": True, "source_sha256": {"helper.py": helper_sha}, "inputs": {"packet": {"path": "features"}},
            "model_files_sha256": model_files, "runtime": {}, "original_encoder_source_sha256": p.sha(out / "original-encoder-source.py"),
            "layout_files": {"index.json": p.sha(out / "index.json")}, "unique_texts": 2}
    p.write(out / "plan.json", plan)
    done = {"status": "completed", "phase": "encode", "version": p.VERSION, "plan_sha256": p.sha(out / "plan.json"),
            "wall_seconds": .1, "no_retry": True, "parameter_training_steps": 0, "test_contents_accessed": False,
            "source_sha256": plan["source_sha256"], "inputs": plan["inputs"], "model_files_sha256": model_files,
            "runtime": {}, "query_count": 1, "unique_texts": 2, "candidate_occurrences": 1,
            "work": work, "progress": progress, "pooling_equivalence": equivalence, "files": p.old.manifest(out)}
    done["payload_bytes"] = sum(v["bytes"] for v in done["files"].values()); p.write(out / "completed.json", done)
    monkeypatch.setattr(p, "parents", lambda _: ({}, {"source_sha256": {}}, {"model_files_sha256": model_files}))
    monkeypatch.setattr(p, "layout_from_inputs", lambda *_: ([], index))
    return out, p.sha(out / "completed.json")


def test_metadata_loader_does_not_decode_any_float_array(tmp_path, monkeypatch):
    out, pin = synthetic_cache(tmp_path, monkeypatch)
    load = np.load; names = []
    def integer_only(path, **kwargs):
        names.append(Path(path).name)
        assert Path(path).name not in {"tokens.npy", "priors.npy", "features.npy"}
        return load(path, **kwargs)
    monkeypatch.setattr(p.np, "load", integer_only)
    index, offsets = p.authenticate_cache_metadata(out, pin)
    assert index["unique_texts"] == 2 and offsets.tolist() == [0, 3, 262]
    assert set(names) == {"offsets.npy", "chunk-offsets.npy", "chunk-lengths.npy", "text-token-counts.npy"}


def test_full_loader_recomputes_exact_priors_and_pooled_parity(tmp_path, monkeypatch):
    out, pin = synthetic_cache(tmp_path, monkeypatch)
    index, raw, offsets, priors = p.authenticate_cache(out, pin)
    assert raw.shape == (262, 384) and index["unique_texts"] == 2
    assert abs(float(priors[offsets[1]:offsets[2]].sum()) - 1) < 2e-7


@pytest.mark.parametrize("tamper", ["byte", "extra", "prior", "offset"])
def test_authenticated_cache_corruption_rejected(tmp_path, monkeypatch, tamper):
    out, pin = synthetic_cache(tmp_path, monkeypatch)
    if tamper == "extra": (out / "unexpected.json").write_text("{}")
    elif tamper == "byte":
        with (out / "tokens.npy").open("ab") as stream: stream.write(b"x")
    elif tamper == "prior": np.save(out / "priors.npy", np.ones(262, np.float32))
    else: np.save(out / "offsets.npy", np.asarray([0, 4, 262], np.int64))
    with pytest.raises(ValueError): p.authenticate_cache_metadata(out, pin)
