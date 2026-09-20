"""Fake-backend tests only: no tokenizer, MLX, weights, corpus, or device calls."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1]/"scripts/run_dialogue_qwen_observation.py"
SPEC = importlib.util.spec_from_file_location("qwen_observation_runner", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_underflow_preserves_direct_nll_and_vocabulary_mass():
    result = runner.summarize_distribution(np.array([-2000., -1000., 1000.], dtype=np.float32), [0, 1])
    assert result["probabilities"] == [0., 1.]
    assert result["log_probs"] == [-1000., 0.]
    assert result["candidate_token_mass"] == 0.
    assert result["log_candidate_token_mass"] == -2000.
    assert -result["log_probs"][0] == 1000.


def test_closed_form_distribution_and_mass_are_distinct():
    result = runner.summarize_distribution(np.log([1., 2., 7.]), [0, 1])
    assert result["probabilities"] == pytest.approx([1/3, 2/3])
    assert result["candidate_token_mass"] == pytest.approx(.3)
    assert result["log_probs"] == pytest.approx(np.log([1/3, 2/3]))
    assert result["log_candidate_token_mass"] == pytest.approx(math.log(.3))


def test_large_common_offset_does_not_destroy_log_normalization():
    result = runner.summarize_distribution(np.full(3, np.finfo(np.float32).max), [0, 1])
    assert result["log_probs"] == pytest.approx([-math.log(2)]*2)
    assert result["candidate_token_mass"] == pytest.approx(2/3)


@pytest.mark.parametrize("values,labels", [([0., float("nan")], [0, 1]), ([0., float("inf")], [0, 1]),
                                         ([0., -1.], [0, 0]), ([0., -1.], [0, True]), ([0., -1.], [0, 2])])
def test_invalid_logits_or_label_ids_fail(values, labels):
    with pytest.raises(ValueError):
        runner.summarize_distribution(values, labels)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path/"repo")
    monkeypatch.setattr(runner, "ROW_COUNT", 2)
    monkeypatch.setattr(runner, "VOCABULARY_SIZE", 6)
    monkeypatch.setattr(runner, "LABEL_IDS", [0, 1])
    monkeypatch.setattr(runner, "runtime", lambda: {"fixture": "no real packages"})
    root = runner.ROOT
    for name in runner.REQUIRED_SOURCES:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source "+name)
    snapshot = tmp_path/"model"/runner.REVISION
    snapshot.mkdir(parents=True)
    for name in runner.MODEL_FILES:
        (snapshot/name).write_text("synthetic model metadata "+name)
    directory = tmp_path/"prepared"
    directory.mkdir()
    rows = []
    for arm in runner.ARMS:
        questions = [{"id": f"q{i}", "question": "Which value?", "candidates": [
            {"id": "c00", "description": "Zebra NONE"}, {"id": "c01", "description": "Apple assigned"}]} for i in range(2)]
        rows.append({"request_id": arm+"-d0-0", "arm": arm, "dialogue_id": "d0", "time": 0,
                     "row_indices": [4, 9], "request": {"context": "Synthetic context", "questions": questions},
                     "tokens": [[1, 2], [1, 2, 3]], "ordered_candidate_ids": [["VALUE", "NONE"]]*2,
                     "canonical_id_maps": [{"c00": "NONE", "c01": "VALUE"}]*2, "label_ids": [0, 1]})
    (directory/"requests.jsonl").write_bytes(b"".join(runner.encoded(r) for r in rows))
    # Intentionally non-JSON, inaccessible task-label payload. The runner must not open it.
    (directory/"labels.jsonl").write_text("DO NOT READ OR HASH LABELS")
    files = {name: {"sha256": runner.sha(directory/name), "bytes": (directory/name).stat().st_size}
             for name in ("requests.jsonl", "labels.jsonl")}
    plan = {"version": runner.VERSION, "method": "batch", "runtime": runner.runtime(), "limits": runner.LIMITS,
            "model": {"id": runner.MODEL_ID, "revision": runner.REVISION, "snapshot_path": str(snapshot),
                      "files_sha256": {name: runner.sha(snapshot/name) for name in runner.MODEL_FILES}},
            "source_sha256": {name: runner.sha(root/name) for name in runner.REQUIRED_SOURCES}, "files": files,
            "row_count": 2, "decisions": 4, "pilot_request_ids": [r["request_id"] for r in rows],
            "input_token_slots": {arm: 6 for arm in runner.ARMS}, "request_counts": {arm: 1 for arm in runner.ARMS}}
    runner.write(directory/"plan.json", plan)
    runner.write(directory/"started.json", {"model_calls": 0})
    runner.write(directory/"completed.json", {"status": "completed", "plan_sha256": runner.sha(directory/"plan.json"),
                                             "model_calls": 0, "tokenizer_only": True,
                                             "files": runner.manifest(directory)})
    calls = []
    mx = SimpleNamespace(synchronize=lambda: None, reset_peak_memory=lambda: None,
                         get_peak_memory=lambda: 100, get_cache_memory=lambda: 20)
    class Experiment:
        def prepare(self, request):
            return [(q, sorted(q.candidates, key=lambda c: c.description), [1, 2] if i == 0 else [1, 2, 3])
                    for i, q in enumerate(request.questions)]

        def logits(self, tokens, method):
            calls.append((tokens, method))
            return np.tile(np.array([2., 0., 1., -1., -2., -3.], dtype=np.float32), (2, 1)), SimpleNamespace(
                **runner.work_for(rows[0]))
    experiment = Experiment()
    monkeypatch.setattr(runner, "load_backend", lambda _plan: (SimpleNamespace(mx=mx), experiment))
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path == directory/"labels.jsonl":
            raise AssertionError("Task labels were accessed")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    return SimpleNamespace(path=directory, plan=plan, rows=rows, calls=calls, experiment=experiment,
                           pin=runner.sha(directory/"plan.json"), tmp=tmp_path)


def arguments(prepared, phase="pilot", **overrides):
    return SimpleNamespace(phase=phase, prepared=prepared.path, plan_sha256=prepared.pin,
                           out=overrides.pop("out", prepared.tmp/phase), **overrides)


def test_canonical_mapping_and_ties_use_frozen_prompt_letter(prepared):
    result = runner.output_questions(prepared.rows[0], np.zeros((2, 6), dtype=np.float32))
    assert result[0]["candidate_ids"] == ["NONE", "VALUE"]
    assert result[0]["label_ids"] == [1, 0]
    assert result[0]["selected_id"] == "VALUE"
    assert result[0]["candidate_max_tie_count"] == 2
    result = runner.output_questions(prepared.rows[0], np.tile([2., 0., 1., -1., -2., -3.], (2, 1)))
    assert result[0]["candidate_logits"] == [0., 2.]
    assert result[0]["probabilities"][1] > result[0]["probabilities"][0]


def test_complete_fake_pilot_then_run_never_reads_labels(prepared):
    pin = runner.execute(arguments(prepared))
    pilot = prepared.tmp/"pilot"
    done = runner.read(pilot/"completed.json")
    assert done["projection"]["admitted"]
    assert done["progress"]["forward_calls_returned"] == 2
    assert not (pilot/"scores.jsonl").exists()
    assert set(done["files"]) == {"started.json", "plan.json", "timings.jsonl"}
    assert "selected_id" not in (pilot/"timings.jsonl").read_text()
    fullpin = runner.execute(arguments(prepared, "run", pilot=pilot, pilot_sha256=pin))
    full = prepared.tmp/"run"
    assert fullpin == runner.sha(full/"completed.json")
    full_done = runner.read(full/"completed.json")
    assert full_done["labels_accessed"] is False
    assert full_done["quality_outputs_saved"] is True
    assert full_done["work_totals"]["input_token_slots"] == 12
    assert len(prepared.calls) == 4
    scores = [json.loads(s) for s in (full/"scores.jsonl").read_text().splitlines()]
    assert [q["row_index"] for s in scores for q in s["questions"]] == [4, 9, 4, 9]
    with pytest.raises(FileExistsError):
        runner.execute(arguments(prepared, "run", pilot=pilot, pilot_sha256=pin))


def test_wrong_external_pin_fails_before_backend(prepared):
    args = arguments(prepared)
    args.plan_sha256 = "0"*64
    with pytest.raises(ValueError, match="External plan"):
        runner.execute(args)
    assert prepared.calls == []
    assert runner.read(args.out/"failed.json")["status"] == "failed"


def test_failed_preparation_cannot_reach_model_loading(prepared):
    runner.write(prepared.path/"failed.json", {"status": "failed"})
    args = arguments(prepared)
    with pytest.raises(ValueError, match="Preparation failed"):
        runner.execute(args)
    assert prepared.calls == []


@pytest.mark.parametrize("target", ["source", "model", "requests"])
def test_mutated_authenticated_bytes_fail_before_backend(prepared, target):
    if target == "source":
        path = runner.ROOT/next(iter(runner.REQUIRED_SOURCES))
    elif target == "model":
        path = Path(prepared.plan["model"]["snapshot_path"])/"tokenizer.json"
    else:
        path = prepared.path/"requests.jsonl"
    path.write_text("tampered")
    args = arguments(prepared)
    with pytest.raises(ValueError, match="drift|identity"):
        runner.execute(args)
    assert prepared.calls == []
    assert (args.out/"failed.json").exists()


def test_full_run_refuses_unadmitted_pilot_before_backend(prepared):
    pin = runner.execute(arguments(prepared))
    pilot = prepared.tmp/"pilot"
    done = runner.read(pilot/"completed.json")
    assert pin == runner.sha(pilot/"completed.json")
    done["projection"]["admitted"] = False
    (pilot/"completed.json").write_bytes(runner.encoded(done))
    prior_calls = len(prepared.calls)
    with pytest.raises(ValueError, match="did not admit"):
        runner.execute(arguments(prepared, "run", pilot=pilot, pilot_sha256=runner.sha(pilot/"completed.json")))
    assert len(prepared.calls) == prior_calls


def test_retokenization_drift_preserves_failure(prepared, monkeypatch):
    monkeypatch.setattr(prepared.experiment, "prepare", lambda request: [
        (q, sorted(q.candidates, key=lambda c: c.description), [5]) for q in request.questions])
    args = arguments(prepared)
    with pytest.raises(ValueError, match="Retokenized prompt"):
        runner.execute(args)
    assert prepared.calls == []
    assert runner.read(args.out/"failed.json")["progress"]["forward_calls_attempted"] == 0


def test_nonfinite_forward_preserves_attempted_and_returned_counts(prepared, monkeypatch):
    def invalid_logits(tokens, method):
        return np.full((2, 6), np.nan, dtype=np.float32), SimpleNamespace(**runner.work_for(prepared.rows[0]))
    monkeypatch.setattr(prepared.experiment, "logits", invalid_logits)
    args = arguments(prepared)
    with pytest.raises(ValueError, match="Nonfinite"):
        runner.execute(args)
    progress = runner.read(args.out/"failed.json")["progress"]
    assert progress["forward_calls_attempted"] == progress["forward_calls_returned"] == 1
    assert progress["requests_completed"] == 0


def test_pilot_covers_both_arms_of_each_longest_group(prepared):
    rows = []
    for t in range(14):
        for original in prepared.rows:
            row = copy.deepcopy(original)
            row.update(request_id=f"{original['arm']}:d:{t}:0", time=t, row_indices=[2*t, 2*t+1])
            if (t, row["arm"]) in ((12, "current"), (13, "history4")):
                row["tokens"] = [[1]*5, [1]*5]
            rows.append(row)
    plan = copy.deepcopy(prepared.plan)
    plan.update(row_count=28, pilot_request_ids=[r["request_id"] for r in rows],
                request_counts={arm: 14 for arm in runner.ARMS},
                input_token_slots={arm: sum(runner.work_for(r)["input_token_slots"] for r in rows if r["arm"] == arm)
                                   for arm in runner.ARMS})
    runner.validate_requests(rows, plan)
    plan["pilot_request_ids"].remove("history4:d:12:0")
    with pytest.raises(ValueError, match="Fixed first12/longest"):
        runner.validate_requests(rows, plan)


def test_later_failure_preserves_completed_request_and_original_exception(prepared, monkeypatch):
    original = prepared.experiment.logits
    def fail_second(tokens, method):
        if len(prepared.calls):
            raise RuntimeError("injected later failure")
        return original(tokens, method)
    monkeypatch.setattr(prepared.experiment, "logits", fail_second)
    args = arguments(prepared)
    with pytest.raises(RuntimeError, match="injected later failure"):
        runner.execute(args)
    failure = runner.read(args.out/"failed.json")
    assert failure["progress"]["requests_completed"] == 1
    assert failure["progress"]["forward_calls_attempted"] == 2
    assert failure["progress"]["forward_calls_returned"] == 1
    assert len((args.out/"timings.jsonl").read_text().splitlines()) == 1
    assert not (args.out/"completed.json").exists()


def test_projection_uses_frozen_token_rule_not_descriptive_request_estimate():
    plan = {"input_token_slots": {"current": 10, "history4": 20},
            "request_counts": {"current": 100000, "history4": 100000}}
    timings = [{"arm": "current", "seconds": .1, "work": {"input_token_slots": 2}},
               {"arm": "history4", "seconds": .4, "work": {"input_token_slots": 4}}]
    result = runner.projection(timings, plan, 1.)
    assert result["projected_full_seconds"] == 66
    assert result["request_scaled_projection_seconds_descriptive_only"] > 7200
    assert result["admitted"] is True


def test_request_duplicate_rows_or_bool_tokens_rejected(prepared):
    rows = copy.deepcopy(prepared.rows)
    rows[0]["row_indices"][1] = 4
    with pytest.raises(ValueError, match="row coverage"):
        runner.validate_requests(rows, prepared.plan)
    rows = copy.deepcopy(prepared.rows)
    rows[0]["tokens"][0][0] = True
    with pytest.raises(ValueError, match="prompt tokens"):
        runner.validate_requests(rows, prepared.plan)


def test_wall_and_storage_limits_are_terminal(tmp_path, monkeypatch):
    budget = runner.Budget(tmp_path, 0., "pilot")
    monkeypatch.setattr(runner.time, "monotonic", lambda: 301.)
    with pytest.raises(TimeoutError):
        budget.check()
    monkeypatch.setattr(runner.time, "monotonic", lambda: 1.)
    monkeypatch.setattr(runner, "peak_rss", lambda: 0)
    with pytest.raises(ValueError, match="storage"):
        budget.storage(512*1024**2+1)


def test_failure_receipt_error_does_not_replace_first_error(prepared, monkeypatch):
    old_write = runner.write
    def fail_receipt(path, value):
        if Path(path).name == "failed.json":
            raise OSError("secondary write problem")
        old_write(path, value)
    monkeypatch.setattr(runner, "write", fail_receipt)
    args = arguments(prepared)
    args.plan_sha256 = "0"*64
    with pytest.raises(ValueError, match="External plan") as captured:
        runner.execute(args)
    assert any("secondary write problem" in note for note in captured.value.__notes__)


def test_late_completion_failure_demotes_receipt(prepared, monkeypatch):
    old_sha = runner.sha
    def fail_final_hash(path, check=lambda: None):
        if Path(path) == prepared.tmp/"pilot"/"completed.json":
            raise OSError("final hash failed")
        return old_sha(path, check)
    monkeypatch.setattr(runner, "sha", fail_final_hash)
    with pytest.raises(OSError, match="final hash"):
        runner.execute(arguments(prepared))
    assert not (prepared.tmp/"pilot"/"completed.json").exists()
    assert (prepared.tmp/"pilot"/"late-completion.json").exists()
    assert (prepared.tmp/"pilot"/"failed.json").exists()
