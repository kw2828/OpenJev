"""Fake local tokenizer and opaque synthetic labels; no real preparation calls."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p = load("lexical_ablation_prep_test", ROOT/"scripts/prepare_dialogue_qwen_lexical_ablation.py")
runner = load("lexical_ablation_frozen_runner_test", ROOT/"scripts/run_dialogue_qwen_observation.py")
fixtures = load("lexical_ablation_public_fixture", ROOT/"tests/test_dialogue_qwen_lexical_ablation.py")


class FakeTokenizer:
    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        self.prompts.append(messages)
        return json.dumps(messages)

    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is False and truncation is False
        return [32+ord(text)-ord("A")] if len(text) == 1 and text in "ABCDEFGHIJKL" else [11]*(len(text)//20+1)


def overwrite(path, obj):
    Path(path).write_text(json.dumps(obj, sort_keys=True)+"\n")


@pytest.fixture
def tree(tmp_path, monkeypatch):
    root, baseline = tmp_path/"repo", tmp_path/"baseline"
    baseline.mkdir()
    for name in p.BASE_SOURCES | p.NEW_SOURCES:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic-source:"+name)
    monkeypatch.setattr(p, "ROOT", root)
    monkeypatch.setattr(p, "ROW_COUNT", 2)
    monkeypatch.setattr(p, "load_runner", lambda: runner)
    monkeypatch.setattr(runner, "runtime", lambda: {"synthetic": True})
    records = [fixtures.packet(i, arm) for i in range(2) for arm in runner.ARMS]
    for record in records:
        record["request"]["context"] = "Exact synthetic context"
    (baseline/"requests.jsonl").write_text("".join(json.dumps(v)+"\n" for v in records))
    (baseline/"labels.jsonl").write_bytes(b'\xff\xfeOPAQUE SYNTHETIC LABEL BYTES, not JSON\x00')
    snapshot = tmp_path/runner.REVISION
    snapshot.mkdir()
    for name in runner.MODEL_FILES:
        (snapshot/name).write_bytes(b"fake-model-file:"+name.encode())
    plan = {"version": runner.VERSION, "method": "batch", "limits": runner.LIMITS, "runtime": runner.runtime(),
        "source_sha256": {name: p.sha(root/name) for name in p.BASE_SOURCES},
        "model": {"id": runner.MODEL_ID, "revision": runner.REVISION, "snapshot_path": str(snapshot),
                  "files_sha256": {name: p.sha(snapshot/name) for name in runner.MODEL_FILES}},
        "files": {name: p.item(baseline/name) for name in ("requests.jsonl", "labels.jsonl")},
        "inputs": {"do-not-read-evaluator-or-corpus": {"sha256": "a"*64, "bytes": 987}},
        "row_count": 2, "decisions": 4, "strata": {"copied_without_decoding": 2},
        "input_token_slots": dict.fromkeys(runner.ARMS, 6), "request_counts": dict.fromkeys(runner.ARMS, 2),
        "pilot_request_ids": [x["request_id"] for x in records]}
    overwrite(baseline/"started.json", {"synthetic": True})
    overwrite(baseline/"plan.json", plan)
    plan_sha = p.sha(baseline/"plan.json")
    done = {"status": "completed", "plan_sha256": plan_sha, "model_calls": 0, "encoder_calls": 0,
            "tokenizer_only": True, "files": {name: p.item(baseline/name) for name in
                ("started.json", "plan.json", "requests.jsonl", "labels.jsonl")}}
    overwrite(baseline/"completed.json", done)
    monkeypatch.setattr(p, "BASE_PLAN", plan_sha)
    monkeypatch.setattr(p, "BASE_COMPLETED", p.sha(baseline/"completed.json"))
    tokenizer, loads = FakeTokenizer(), []
    def tokenizer_load(path):
        loads.append(path)
        return tokenizer
    monkeypatch.setattr(p, "load_tokenizer", tokenizer_load)
    args = SimpleNamespace(baseline=baseline, baseline_plan_sha256=p.BASE_PLAN,
        baseline_completed_sha256=p.BASE_COMPLETED, protocol_sha256=p.sha(root/p.PROTOCOL), out=tmp_path/"prepared")
    return SimpleNamespace(args=args, plan=plan, records=records, tokenizer=tokenizer, loads=loads, snapshot=snapshot)


def test_complete_fake_preparation_opaque_labels_and_original_runner_contract(tree):
    result = p.execute(tree.args)
    out = tree.args.out
    assert (out/"labels.jsonl").read_bytes() == (tree.args.baseline/"labels.jsonl").read_bytes()
    assert result["inputs"] == tree.plan["inputs"] and result["strata"] == tree.plan["strata"]
    assert result["model"] == tree.plan["model"] and result["runtime"] == tree.plan["runtime"]
    assert result["version"] == runner.VERSION and result["experiment_id"] == p.EXPERIMENT
    assert set(result["source_sha256"]) == p.BASE_SOURCES | p.NEW_SOURCES
    assert result["parent"]["files"] == tree.plan["files"]
    assert result["request_counts"] == tree.plan["request_counts"]
    actual = [json.loads(x) for x in (out/"requests.jsonl").read_text().splitlines()]
    runner.validate_requests(actual, result)
    assert [x["request_id"] for x in actual] == [x["request_id"] for x in tree.records]
    assert len(tree.tokenizer.prompts) == 4 and len(tree.loads) == 1
    for before, after in zip(tree.records, actual, strict=True):
        assert fixtures.verify_transform(before, after)
    done = p.read(out/"completed.json")
    assert done["labels_decoded"] is False and done["model_calls"] == 0 and done["tokenizer_only"] is True
    assert set(done["files"]) == {"started.json", "requests.jsonl", "labels.jsonl", "plan.json"}
    assert all(p.item(out/name) == value for name, value in done["files"].items())
    with pytest.raises(FileExistsError):
        p.execute(tree.args)


@pytest.mark.parametrize("damage", ["requests", "labels", "extra", "source", "protocol", "completion"])
def test_authentication_precedes_tokenizer_and_request_decode(tree, monkeypatch, damage):
    if damage in ("requests", "labels"):
        (tree.args.baseline/(damage+".jsonl")).write_bytes(b"corrupt bytes")
    elif damage == "extra":
        (tree.args.baseline/"failed.json").write_text("{}")
    elif damage == "source":
        (p.ROOT/next(iter(p.BASE_SOURCES))).write_text("corrupt source")
    elif damage == "protocol":
        tree.args.protocol_sha256 = "0"*64
    else:
        tree.args.baseline_completed_sha256 = "0"*64
    monkeypatch.setattr(p, "tokenize_record", lambda *args: pytest.fail("Prompt decoding reached"))
    with pytest.raises(ValueError):
        p.execute(tree.args)
    assert tree.loads == []
    failure = p.read(tree.args.out/"failed.json")
    assert failure["progress"]["tokenizer_load_attempted"] is False and failure["model_calls"] == 0


def test_model_file_corruption_rejected_before_local_tokenizer(tree):
    (tree.snapshot/"model.safetensors").write_bytes(b"corrupt fake weights")
    with pytest.raises(ValueError, match="Model/tokenizer file"):
        p.execute(tree.args)
    assert tree.loads == []


def test_runtime_mismatch_rejected_without_tokenizer(tree, monkeypatch):
    monkeypatch.setattr(runner, "runtime", lambda: {"different": True})
    with pytest.raises(ValueError, match="runtime"):
        p.execute(tree.args)
    assert tree.loads == []


def test_no_truncation_on_overflow_and_partial_progress_preserved(tree, monkeypatch):
    original = tree.tokenizer.encode
    count = 0
    def encode(text, **kwargs):
        nonlocal count
        if len(text) > 1:
            count += 1
        return [1]*4097 if count == 2 else original(text, **kwargs)
    monkeypatch.setattr(tree.tokenizer, "encode", encode)
    with pytest.raises(ValueError, match="no truncation"):
        p.execute(tree.args)
    failure = p.read(tree.args.out/"failed.json")
    assert failure["progress"]["requests_completed"] == 1
    assert failure["progress"]["tokenizer_loaded"] is True
    assert len((tree.args.out/"requests.jsonl").read_text().splitlines()) == 1
    assert not (tree.args.out/"completed.json").exists()


def test_terminal_output_failure_demotes_completion(tree, monkeypatch):
    original = p.write
    def write(path, value):
        original(path, value)
        if Path(path).name == "completed.json":
            monkeypatch.setitem(p.LIMITS, "output_bytes", 0)
    monkeypatch.setattr(p, "write", write)
    with pytest.raises(ValueError, match="output cap"):
        p.execute(tree.args)
    assert (tree.args.out/"late-completion.json").exists()
    assert (tree.args.out/"failed.json").exists()
    assert not (tree.args.out/"completed.json").exists()
