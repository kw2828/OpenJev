"""Synthetic orchestration of the unchanged inference entry points only."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("runtime_inference_test", SCRIPTS / "infer_dialogue_runtime.py")
inference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inference)


class Budget:
    def __init__(self):
        self.ticks = 0
        self.checks = 0
        self.storage_checks = 0
        self.syncs = 0

    def check(self):
        self.checks += 1

    def elapsed(self):
        self.ticks += 1
        return self.ticks / 100

    def sync(self, _torch):
        self.syncs += 1

    def storage(self):
        self.storage_checks += 1


def public_work():
    return {"input_texts": 1, "content_tokens": 1, "encoder_sequences": 1, "encoder_calls": 1,
            "special_token_positions": 2, "valid_token_positions": 3, "padded_token_positions": 3,
            "padded_attention_positions": 9, "padding_token_positions": 0, "overlength_texts_chunked": 0,
            "truncated_tokens": 0, "public_user_turns": 1, "real_question_updates": 1}


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    selected = [f"synthetic-calibration-{i:04d}" for i in range(512)]
    profiles = [{"split": "train", "dialogue_id": did, "work": public_work()} for did in selected]
    totals = {key: sum(p["work"][key] for p in profiles) for key in public_work()}
    calibration_work = {"dialogue_count": 512, **{key: totals[key] for key in inference.replay.WORK_MEASURES}}
    actors = [{"dialogue_id": did, "split": "train", "source_split": "train", "analysis_role": "calibration",
               "tokens": [[4]], "query_ids": [0], "user_turn_indices": [0], "turn_text_ids": [0],
               "query_text_ids": [0], "candidate_text_ids": [[0, 0, 0]],
               "candidate_ids": [["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:red"]],
               "lexical_offset": 30 * i, "lexical_shape": [1, 1, 3, 10]}
              for i, did in enumerate(selected)]
    rows = [{"row_index": i, "dialogue_id": did, "split": "train", "source_split": "train",
             "analysis_role": "calibration", "label_id": "value:red", "label_index": 2}
            for i, did in enumerate(selected)]
    prepared = tmp_path / "prepared"
    data = {"selection.json": {"selected_ids": selected, "sample_size": 512,
                                "source_split": "train", "analysis_role": "calibration"},
            "workloads.json": {"profiles": profiles, "all": {"dialogues": 512, "totals": totals}},
            "actors.jsonl": actors, "evaluation-rows.jsonl": rows}
    calls = {"reads": [], "backend": [], "lexical": [], "loads": [], "infer": [], "cache": [], "snapshots": []}
    behavior = {}
    out = tmp_path / "out"
    out.mkdir()
    args = SimpleNamespace(command="infer", plan=tmp_path / "plan.json", plan_sha256="runtime-plan",
                           supervision=tmp_path / "launch.json", out=out, pilot_sha256="pilot-pin",
                           pilot_terminal=tmp_path / "pilot-terminal.json", pilot_terminal_sha256="pilot-terminal-pin")
    ctx = {"prepared": prepared, "preparation_receipt": {"counts": {"dialogues": 512, "scored_endpoints": 512}},
           "preparation_terminal": {"status": "completed", "returncode": 0},
           "pilot_terminal": {"status": "completed", "returncode": 0},
           "pilot_receipt": {"fit_order": list(inference.FIT_ORDER), "projection": {
               "admitted": True, "all_verifications_passed": True, "verifications_passed": 12, "verifications_total": 12},
               "parity_passed": True, "completed_forwards": 1560, "calibration_work": calibration_work},
           "science": {"fit_order": list(inference.FIT_ORDER)}, "runtime_plan": {"config": {"fit_order": list(inference.FIT_ORDER)}},
           "plan": {"unchanged": "legacy plan"}, "progress": {}}
    api = SimpleNamespace(torch=SimpleNamespace(mps=SimpleNamespace(empty_cache=lambda: calls["cache"].append(True))))
    lexical = {"original": object(), "numbers": object()}

    def read(path):
        assert path.parent == prepared, "Unexpected input source"
        calls["reads"].append(path.name)
        return data[path.name]

    def read_lines(path, budget):
        budget.check()
        return read(path)

    def load_lexical(path):
        assert path == prepared
        calls["lexical"].append(path)
        return lexical

    def backend():
        calls["backend"].append(True)
        return api

    class Route:
        def __init__(self, fit_id):
            self.fit_id = fit_id

        def snapshot(self):
            calls["snapshots"].append(self.fit_id)
            if behavior.get("snapshot_error") == self.fit_id:
                raise LookupError("synthetic cleanup snapshot error")
            return {"synthetic_fit": self.fit_id}

    def load_route(supplied_api, supplied_ctx, fit_id, budget):
        assert supplied_api is api and supplied_ctx is ctx
        calls["loads"].append(fit_id)
        return Route(fit_id), .1

    def infer_fit(supplied_api, route, directory, supplied_actors, grouped, supplied_lexical, arm,
                  supplied_profiles, row_count, budget, progress):
        assert supplied_api is api and supplied_actors is actors and supplied_lexical is lexical
        assert [a["dialogue_id"] for a in supplied_actors] == selected
        assert set(supplied_profiles) == set(selected) and row_count == 512
        assert [row["row_index"] for did in selected for row in grouped[did]] == list(range(512))
        assert arm == route.fit_id.rsplit("-", 1)[0]
        assert progress is ctx["progress"]
        assert not any("label_index" in actor for actor in supplied_actors)
        calls["infer"].append(route.fit_id)
        if behavior.get("failure_fit") == route.fit_id:
            (directory / "partial-log-probs.npy").write_bytes(b"synthetic partial artifact")
            raise RuntimeError("synthetic inference failure")
        (directory / "predictions.npz").write_bytes(b"synthetic packet, never decoded")
        (directory / "dialogues.jsonl").write_text('{"synthetic":true}\n')
        if behavior.get("extra_payload") == route.fit_id:
            (directory / "partial-row-indices.npy").write_bytes(b"unfinished")
        progress["completed_forwards"] += behavior.get("count_increment", 512)
        return {"dialogues": behavior.get("returned_dialogues", 512), "endpoint_rows": row_count,
                "dialogue_seconds": .2, "prediction_finalization_seconds": .01, "witness": {"synthetic": True}}

    monkeypatch.setattr(inference.common, "read", read)
    monkeypatch.setattr(inference.common, "PINS", {inference.common.PREPARED + "/completed.json": "prepared-pin",
                                                   inference.common.PREP_TERMINAL: "prepared-terminal-pin"})
    monkeypatch.setattr(inference.replay, "read_lines", read_lines)
    monkeypatch.setattr(inference.replay, "load_lexical", load_lexical)
    monkeypatch.setattr(inference.replay, "backend", backend)
    monkeypatch.setattr(inference.replay, "load_route", load_route)
    monkeypatch.setattr(inference.replay, "infer_fit", infer_fit)
    monkeypatch.setattr(inference.gc, "collect", lambda: None)
    return SimpleNamespace(args=args, ctx=ctx, budget=Budget(), data=data, calls=calls, behavior=behavior)


def test_cli_delegates_authentication_to_runtime_lifecycle_before_body(monkeypatch, tmp_path):
    calls = []

    def execute(args, body):
        calls.append((args, body))
        return "not-executed"

    monkeypatch.setattr(inference.common, "execute", execute)
    result = inference.main(["--plan", str(tmp_path / "plan.json"), "--plan-sha256", "plan-pin",
                             "--supervision", str(tmp_path / "launch.json"), "--out", str(tmp_path / "out"),
                             "--pilot-sha256", "pilot-pin", "--pilot-terminal", str(tmp_path / "pilot-terminal.json"),
                             "--pilot-terminal-sha256", "terminal-pin"])
    assert result == "not-executed" and calls[0][1] is inference.body
    args = calls[0][0]
    assert args.command == "infer" and args.pilot_sha256 == "pilot-pin" and args.pilot_terminal_sha256 == "terminal-pin"
    assert args.pilot_terminal == tmp_path / "pilot-terminal.json" and not args.out.exists()


def test_all_twelve_reuse_unchanged_inference_path_and_preserve_exact_cohort(fixture, capsys):
    before = copy.deepcopy(fixture.data)
    result = inference.body(fixture.args, fixture.ctx, fixture.budget)
    assert fixture.calls["backend"] == [True]
    assert fixture.calls["loads"] == fixture.calls["infer"] == fixture.calls["snapshots"] == inference.FIT_ORDER
    assert len(fixture.calls["cache"]) == fixture.budget.syncs == fixture.budget.storage_checks == 12
    assert result["completed_forwards"] == 6144 and result["endpoint_rows_per_fit"] == 512
    assert result["total_endpoint_rows"] == 6144 and result["fit_order"] == inference.FIT_ORDER
    assert result["prepared_sha256"] == "prepared-pin" and result["prepared_terminal_sha256"] == "prepared-terminal-pin"
    assert result["pilot_sha256"] == "pilot-pin" and result["pilot_terminal_sha256"] == "pilot-terminal-pin"
    assert result["source_split"] == "train" and result["analysis_role"] == "calibration"
    assert result["model_weight_updates"] == result["temperature_fits"] == 0
    for key in ("optimizer_created", "temperature_applied", "official_test_opened", "official_dev_inference",
                "task_metrics_computed", "legacy_v1_admission_revised", "scientific_conditions_changed"):
        assert result[key] is False
    assert fixture.data == before
    files = inference.common.members(fixture.args.out)
    assert set(files) == inference.common.expected_members("infer") - {"started.json"}
    assert len(files) == 36
    for fit in result["fits"]:
        path = fixture.args.out / fit["fit_id"] / "completed.json"
        assert fit["completed_sha256"] == inference.common.sha(path)
        assert json.loads(path.read_text()) == {key: value for key, value in fit.items() if key != "completed_sha256"}
        assert set(fit["files"]) == {"predictions.npz", "dialogues.jsonl"}
    assert len(capsys.readouterr().out.splitlines()) == 12


@pytest.mark.parametrize("defect", ["old_fit_order", "runtime_fit_order", "pilot_fit_order", "not_admitted",
                                    "verification_failed", "verification_missing", "parity_failed", "pilot_incomplete",
                                    "pilot_parent_failed", "preparation_parent_nonzero"])
def test_prerequisite_failure_precedes_prepared_decode_and_backend(fixture, defect):
    ctx = fixture.ctx
    if defect == "old_fit_order":
        ctx["science"]["fit_order"] = list(reversed(inference.FIT_ORDER))
    elif defect == "runtime_fit_order":
        ctx["runtime_plan"]["config"]["fit_order"] = inference.FIT_ORDER[:-1]
    elif defect == "pilot_fit_order":
        ctx["pilot_receipt"]["fit_order"] = inference.FIT_ORDER[:-1]
    elif defect == "not_admitted":
        ctx["pilot_receipt"]["projection"]["admitted"] = False
    elif defect == "verification_failed":
        ctx["pilot_receipt"]["projection"]["all_verifications_passed"] = False
    elif defect == "verification_missing":
        ctx["pilot_receipt"]["projection"]["verifications_total"] = 11
    elif defect == "parity_failed":
        ctx["pilot_receipt"]["parity_passed"] = False
    elif defect == "pilot_incomplete":
        ctx["pilot_receipt"]["completed_forwards"] = 1559
    elif defect == "pilot_parent_failed":
        ctx["pilot_terminal"]["status"] = "failed"
    else:
        ctx["preparation_terminal"]["returncode"] = 1
    with pytest.raises(ValueError):
        inference.body(fixture.args, ctx, fixture.budget)
    assert not fixture.calls["reads"] and not fixture.calls["backend"] and not fixture.calls["lexical"]
    assert not list(fixture.args.out.iterdir())


@pytest.mark.parametrize("defect", ["selection_short", "selection_duplicate", "selection_role", "profile_missing",
                                    "profile_duplicate", "profile_split", "work_sum", "work_type", "pilot_work",
                                    "actor_order", "actor_role", "actor_target", "row_order", "row_role", "row_count"])
def test_calibration_preparation_errors_fail_before_loading_models(fixture, defect):
    data = fixture.data
    if defect == "selection_short":
        data["selection.json"]["selected_ids"].pop()
    elif defect == "selection_duplicate":
        data["selection.json"]["selected_ids"][-1] = data["selection.json"]["selected_ids"][0]
    elif defect == "selection_role":
        data["selection.json"]["source_split"] = "dev"
    elif defect == "profile_missing":
        data["workloads.json"]["profiles"].pop()
    elif defect == "profile_duplicate":
        data["workloads.json"]["profiles"][-1] = data["workloads.json"]["profiles"][0]
    elif defect == "profile_split":
        data["workloads.json"]["profiles"][0]["split"] = "dev"
    elif defect == "work_sum":
        data["workloads.json"]["all"]["totals"]["content_tokens"] += 1
    elif defect == "work_type":
        data["workloads.json"]["profiles"][0]["work"]["encoder_calls"] = True
    elif defect == "pilot_work":
        fixture.ctx["pilot_receipt"]["calibration_work"]["encoder_calls"] += 1
    elif defect == "actor_order":
        data["actors.jsonl"].reverse()
    elif defect == "actor_role":
        data["actors.jsonl"][0]["source_split"] = "dev"
    elif defect == "actor_target":
        data["actors.jsonl"][0]["label_index"] = 2
    elif defect == "row_order":
        data["evaluation-rows.jsonl"].reverse()
    elif defect == "row_role":
        data["evaluation-rows.jsonl"][0]["analysis_role"] = "evaluation"
    else:
        fixture.ctx["preparation_receipt"]["counts"]["scored_endpoints"] += 1
    with pytest.raises(ValueError):
        inference.body(fixture.args, fixture.ctx, fixture.budget)
    assert not fixture.calls["backend"] and not fixture.calls["lexical"]
    assert not list(fixture.args.out.iterdir())


def test_partial_fit_failure_stops_without_retry_and_preserves_obtainable_artifacts(fixture):
    first, second = inference.FIT_ORDER[:2]
    fixture.behavior["failure_fit"] = second
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        inference.body(fixture.args, fixture.ctx, fixture.budget)
    assert fixture.calls["loads"] == fixture.calls["infer"] == [first, second]
    assert (fixture.args.out / first / "completed.json").exists()
    assert (fixture.args.out / second / "partial-log-probs.npy").exists()
    assert not (fixture.args.out / second / "completed.json").exists()
    assert fixture.ctx["progress"]["completed_fits"] == [first]


def test_cleanup_error_does_not_replace_primary_forward_failure(fixture):
    first = inference.FIT_ORDER[0]
    fixture.behavior.update(failure_fit=first, snapshot_error=first)
    with pytest.raises(RuntimeError, match="synthetic inference failure") as caught:
        inference.body(fixture.args, fixture.ctx, fixture.budget)
    assert any("cleanup snapshot error" in note for note in caught.value.__notes__)
    assert not (fixture.args.out / first / "completed.json").exists()


@pytest.mark.parametrize("defect", ["short_fit", "leftover_partial", "total_forward_count"])
def test_incomplete_return_or_payload_cannot_produce_success_metadata(fixture, defect):
    if defect == "short_fit":
        fixture.behavior["returned_dialogues"] = 511
    elif defect == "leftover_partial":
        fixture.behavior["extra_payload"] = inference.FIT_ORDER[0]
    else:
        fixture.behavior["count_increment"] = 511
    with pytest.raises(ValueError):
        inference.body(fixture.args, fixture.ctx, fixture.budget)
    if defect != "total_forward_count":
        assert not (fixture.args.out / inference.FIT_ORDER[0] / "completed.json").exists()
        assert len(fixture.calls["loads"]) == 1
    else:
        assert len(fixture.calls["loads"]) == 12 and fixture.ctx["progress"]["completed_forwards"] == 6132


def test_existing_fit_directory_is_not_reused_or_changed(fixture):
    directory = fixture.args.out / inference.FIT_ORDER[0]
    directory.mkdir()
    sentinel = directory / "keep.txt"
    sentinel.write_bytes(b"existing attempt")
    with pytest.raises(FileExistsError):
        inference.body(fixture.args, fixture.ctx, fixture.budget)
    assert not fixture.calls["loads"]
    assert sentinel.read_bytes() == b"existing attempt" and list(directory.iterdir()) == [sentinel]
