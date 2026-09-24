"""Fabricated routing, durability and census export; no real native/model calls."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "residual_collector_tests", Path(__file__).resolve().parents[1] / "scripts/collect_otto_residual.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def select(scores, allowed):
    subset = scores[list(allowed)]
    return allowed[int(np.flatnonzero(np.abs(subset - subset.min()) < 1e-10)[0])]


def test_master_roster_reserves54_but_executes18_with_continuing_rotation():
    rows = M.cohort()
    assert M.VERSION == "otto-residual-collection-v1"
    assert M.CASES == {"dev": 3, "confirm": 6}
    assert len(rows) == M.MASTER_EPISODES == 54 and M.EPISODES == 18
    assert [r["episode_index"] for r in rows] == list(range(54))
    assert len({r["episode_id"] for r in rows}) == 54
    assert M.execution_cohort("dev") == rows[:18]
    assert [r["episode_index"] for r in M.cohort("confirm")] == list(range(18, 54))
    seeds = []
    for stage, size in (("dev", 18), ("confirm", 36)):
        assert len(M.cohort(stage)) == M.STAGE_EPISODES[stage] == size
        for regime, first in M.FIRST[stage].items():
            expected = set(range(first, first + M.CASES[stage]))
            assert {r["seed"] for r in rows if r["stage"] == stage and r["regime"] == regime} == expected
            seeds.extend(expected)
    assert len(seeds) == len(set(seeds)) == 18
    assert not set(seeds) & set(M.FIT_SEEDS)
    assert sorted(seeds) == [*range(314000001, 314000004), *range(315000001, 315000004),
                             *range(316000001, 316000007), *range(317000001, 317000007)]
    for case_index in range(18):
        group = rows[3 * case_index:3 * case_index + 3]
        offset = case_index % 3
        assert [r["arm"] for r in group] == list(M.ARMS[offset:] + M.ARMS[:offset])
        assert len({(r["stage"], r["regime"], r["seed"], r["case"]) for r in group}) == 1
        assert all(r["initial_hit"] == 1 + r["case"] % 3 for r in group)


def test_collector_stays_period_four_and_sampling_has_no_payload():
    assert [M.virtual_query(n) for n in range(10)] == [0, 0, 0, 0, 4, 4, 4, 4, 8, 8]
    assert M.virtual_query(2187) == 2184
    for bad in (True, -1, 2188, 1.):
        with pytest.raises(ValueError, match="preaction step"):
            M.virtual_query(bad)
    assert M.CONFIGURATION["collector_period"] == 4
    assert M.CONFIGURATION["window_sampling"] is False
    assert M.CONFIGURATION["deferred_annotation"] is False
    assert "annotations.jsonl" not in M.JOURNALS
    assert "annotations.jsonl.gz" not in M.PAYLOADS
    assert "train-selection.json" not in M.PAYLOADS
    assert "valid.npz" not in M.PAYLOADS
    assert {name for name in M.PAYLOADS if name.endswith(".npz")} == {"dev.npz"}
    assert len(M.PAYLOADS) == 15
    assert not hasattr(M.Run, "annotate_train")


@pytest.mark.parametrize("stage", ("dev",))
@pytest.mark.parametrize("arm,expected_queries", (("analytic", 0), ("neural", 9), ("period4_hold", 3)))
def test_census_reuses_every_deployed_score_without_duplicate_calls(stage, arm, expected_queries):
    assert arm in {r["arm"] for r in M.cohort(stage)}
    cache, calls, queries, annotations = None, [], 0, 0
    for step in range(9):
        def teacher(deployed, step=step):
            calls.append((step, deployed))
            return np.asarray([4., 3., 2., 1.], np.float32)
        action, raw, cache, deployed, correction = M.route_scores(
            arm, step, (0, 1, 2, 3), 1, teacher, select, cache)
        assert raw.dtype == np.float32 and action == (1 if arm == "analytic" else 3)
        assert correction is (step % 4 == 0)
        queries += deployed
        annotations += not deployed
    assert len(calls) == 9 and [step for step, _ in calls] == list(range(9))
    assert queries == expected_queries and annotations == 9 - expected_queries


def test_held_action_is_fixed_before_annotation_and_memory_never_refreshes():
    original = np.asarray([0., 4., 2., 3.], np.float32)
    action, _, cache, deployed, _ = M.route_scores(
        "period4_hold", 0, (0, 1, 2, 3), 1, lambda _: original, select, None)
    assert action == 0 and deployed and not cache.flags.writeable
    original[:] = 99
    before, events = cache.tobytes(), []
    def held_select(scores, allowed):
        events.append("selected")
        return select(scores, allowed)
    def annotate(is_deployed):
        assert not is_deployed and events == ["selected"]
        events.append("annotated")
        return np.asarray([8., 0., 7., 6.], np.float32)
    action, raw, after, deployed, correction = M.route_scores(
        "period4_hold", 1, (1, 2, 3), 1, annotate, held_select, cache)
    assert action == 2 and raw[1] == 0 and not deployed and not correction
    assert after is cache and after.tobytes() == before
    assert events == ["selected", "annotated"]


def test_analytic_ignores_labels_and_annotation_errors_propagate():
    action, _, cache, deployed, _ = M.route_scores(
        "analytic", 0, (0, 1, 2, 3), 2, lambda _: np.asarray([8., 0., 7., 6.], np.float32), select, None)
    assert (action, cache, deployed) == (2, None, False)
    error = OSError("fabricated failed annotation")
    def fail(_):
        raise error
    with pytest.raises(OSError) as caught:
        M.route_scores("analytic", 1, (0, 1, 2, 3), 2, fail, select, None)
    assert caught.value is error


def test_missing_anchor_and_mutating_annotation_are_rejected():
    cache = np.arange(4, dtype=np.float32)
    def corrupt(_):
        cache[0] += 1
        return np.zeros(4, np.float32)
    with pytest.raises(ValueError, match="modify held memory"):
        M.route_scores("period4_hold", 1, (0, 1, 2, 3), 1, corrupt, select, cache)
    with pytest.raises(ValueError, match="earlier correction"):
        M.route_scores("period4_hold", 1, (0, 1, 2, 3), 1, corrupt, select, None)


def test_offsets_preserve_singletons_and_horizon_tails():
    assert M.episode_offsets([1, 3, 4, 5, 2188]) == [0, 1, 4, 8, 13, 2201]
    for values in ([], [0], [True], [2189]):
        with pytest.raises(ValueError, match="episode lengths"):
            M.episode_offsets(values)


def fill_stage(run, stage):
    for index, identity in enumerate(M.cohort(stage)):
        run.rows.append({**identity, "steps": 1, "start_row": index, "end_row": index + 1})
        run.buffer["features"].append(np.full(31, -0., np.float32))
        run.buffer["raw_q"].append(np.asarray([4., 3., 2., -0.], np.float32))
        run.buffer["legal"].append(np.asarray([True, True, True, False]))
        run.buffer["actions"].append(2)
        run.buffer["correction"].append(True)
        run.label_mask.append(True)


@pytest.mark.parametrize("stage", ("dev",))
def test_every_split_exports_same_six_arrays_and_exact_bytes(stage, tmp_path, monkeypatch):
    run = M.Run(SimpleNamespace(output=tmp_path))
    run.runtime, run.check = SimpleNamespace(np=np), lambda: None
    monkeypatch.setattr(M, "ROOT", tmp_path)
    fill_stage(run, stage)
    size = M.STAGE_EPISODES[stage]
    run.save_stage(stage)
    with np.load(tmp_path / f"{stage}.npz", allow_pickle=False) as saved:
        assert set(saved.files) == set(M.ARRAY_KEYS)
        assert saved["features"].shape == (size, 31) and np.signbit(saved["features"]).all()
        assert np.signbit(saved["raw_q"][:, 3]).all()
        assert saved["episode_offsets"].tolist() == list(range(size + 1))
        assert saved["actions"].dtype == np.int64 and saved["correction"].dtype == np.bool_
    assert run.datasets[stage]["rows"] == size
    assert run.datasets[stage]["episodes"] == size
    assert set(run.datasets[stage]["arrays"]) == set(M.ARRAY_KEYS)
    assert all(not values for values in run.buffer.values()) and run.label_mask == []


@pytest.mark.parametrize("stage", ("dev",))
@pytest.mark.parametrize("defect", ("missing_label", "nonfinite_label", "missing_episode", "wrong_offset"))
def test_incomplete_census_cannot_be_exported(stage, defect, tmp_path):
    run = M.Run(SimpleNamespace(output=tmp_path))
    run.runtime, run.check = SimpleNamespace(np=np), lambda: None
    fill_stage(run, stage)
    if defect == "missing_label":
        run.label_mask[-1] = False
    elif defect == "nonfinite_label":
        run.buffer["raw_q"][-1][0] = np.nan
    elif defect == "missing_episode":
        run.rows.pop()
    else:
        run.rows[-1]["start_row"] += 1
    with pytest.raises(ValueError):
        run.save_stage(stage)
    assert not (tmp_path / f"{stage}.npz").exists()


@pytest.mark.parametrize("phase", ("confirm", "test", "train", "", None, True))
def test_no_other_phase_can_enter_episode_or_serialization(phase, tmp_path):
    run = M.Run(SimpleNamespace(output=tmp_path))
    # Deliberately omit runtime: rejection must precede every numerical touch.
    with pytest.raises(ValueError, match="separately admitted future phase"):
        M.execution_cohort(phase)
    with pytest.raises(ValueError, match="separately admitted future phase"):
        run.episode({**M.cohort()[0], "stage": phase})
    with pytest.raises(ValueError, match="separately admitted future phase"):
        run.complete_episode({**M.cohort()[0], "stage": phase}, {})
    with pytest.raises(ValueError, match="separately admitted future phase"):
        run.save_stage(phase)
    assert list(tmp_path.iterdir()) == [] and run.ledger.sequence == 0


def test_unregistered_or_out_of_order_dev_identity_never_attempts_native(tmp_path):
    run = M.Run(SimpleNamespace(output=tmp_path))
    with pytest.raises(ValueError, match="registered DEV identity"):
        run.episode({**M.cohort()[0], "seed": 999})
    with pytest.raises(ValueError, match="next exact master DEV identity"):
        run.complete_episode(M.cohort()[1], {})
    assert run.rows == [] and run.pending_episode is None and run.ledger.sequence == 0


def test_episode_completion_requires_both_durable_return_records(tmp_path):
    run = M.Run(SimpleNamespace(output=tmp_path))
    identity = M.cohort()[0]
    events, error = [], OSError("completion fsync failed")
    def episode(_):
        run.pending_episode = dict(identity)
        return {**identity, "steps": 1, "start_row": 0, "end_row": 1,
                "source_evaluation_only": [1, 2], "draws_evaluation_only": []}
    def publish(name, record):
        events.append(name)
        if record.get("event") == "return":
            raise error
    run.episode, run.append_durable, run.check = episode, publish, lambda: None
    run.ledger.flush = lambda: events.append("flushed")
    with pytest.raises(OSError) as caught:
        run.complete_episode(identity, {})
    assert caught.value is error and run.rows == [] and run.pending_episode == identity
    assert events == ["flushed", "episodes.jsonl", "episode-boundaries.jsonl"]


def test_paired_draw_checks_are_split_specific_and_reject_mismatch():
    paired = {}
    common = {"regime": "lambda3", "case": 0,
              "source_evaluation_only": [1, 2],
              "draws_evaluation_only": [{"channel": "hit", "index": 0, "uniform": .25}]}
    M.paired_identity({**common, "stage": "confirm"}, paired)
    M.paired_identity({**common, "stage": "dev", "source_evaluation_only": [3, 4]}, paired)
    with pytest.raises(ValueError, match="paired original"):
        M.paired_identity({**common, "stage": "confirm", "source_evaluation_only": [3, 4]}, paired)


def test_full_census_caps_include_every_split_and_forbid_replay():
    assert M.CALL_CAPS["teacher_score"] == M.CALL_CAPS["tensorflow_value"] == 39384
    for name in ("native_step", "actor_update", "analytic_score", "feature_build"):
        assert M.CALL_CAPS[name] == 18 * 2188
    for name in ("native_reset", "actor_construction", "backend_binding"):
        assert M.CALL_CAPS[name] == 18
    for name in ("annotation_public_reset", "annotation_public_update"):
        assert M.CALL_CAPS[name] == 0
    for name in ("tensorflow_construction", "tensorflow_build", "tensorflow_load"):
        assert M.CALL_CAPS[name] == 1


def test_ledger_forbids_zero_cap_replay_before_callback(tmp_path):
    run = M.Run(SimpleNamespace(output=tmp_path))
    run.check = lambda: None
    with pytest.raises(ValueError, match="operation cap"):
        run.ledger.call("annotation_public_reset", lambda: pytest.fail("replay cannot execute"))
    assert run.ledger.pending == [] and run.ledger.sequence == 0
    assert list(tmp_path.iterdir()) == []


def test_failed_call_keeps_original_pending_operation_without_return(tmp_path):
    run = M.Run(SimpleNamespace(output=tmp_path))
    run.check = lambda: None
    error = OSError("fabricated failed scorer")
    def fail():
        raise error
    with pytest.raises(OSError) as caught:
        run.ledger.call("teacher_score", fail)
    run.ledger.close()
    assert caught.value is error
    assert run.ledger.calls["teacher_score"]["attempted"] == 1
    assert run.ledger.calls["teacher_score"]["returned"] == 0
    assert len(run.ledger.pending) == 1
    with gzip.open(tmp_path / "work.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 1 and rows[0]["event"] == "attempt"


def test_lazy_journal_closure_has_no_unwritten_annotation_member(tmp_path):
    run = M.Run(SimpleNamespace(output=tmp_path))
    for name in M.JOURNALS:
        run.ledger.emit(name, {"event": "fabricated"})
    run.ledger.flush()
    run.ledger.close()
    assert {p.name for p in tmp_path.iterdir()} == {name + ".gz" for name in M.JOURNALS}
    assert {p.name for p in tmp_path.iterdir()} <= M.PAYLOADS
    with pytest.raises(ValueError, match="declared bounded journal"):
        run.ledger.emit("annotations.jsonl", {"event": "forbidden"})


def seed_review():
    seeds = [first + case for stage, regimes in M.FIRST.items()
             for first in regimes.values() for case in range(M.CASES[stage])]
    allocations = {stage: {regime: list(range(first, first + M.CASES[stage]))
                          for regime, first in regimes.items()} for stage, regimes in M.FIRST.items()}
    hits = [{"path": f"source-{index}.py", "line": 1, "column": 1, "literal": "314966016",
             "context": "count = 314966016", "value": 314966016, "exact_seed": False} for index in range(5)]
    reviewed = [{**hit, "source_verified": True, "classification": "non-seed operation or sample count"} for hit in hits]
    common = {"admits_execution": False, "allocations": allocations, "fresh_environment_seeds": seeds,
              "reused_fit_seeds": list(M.FIT_SEEDS), "scope": "fabricated sources only", "scope_limits": ["fixture"],
              "checkpoint_decodes": 0, "empirical_array_decodes": 0, "raw_journal_reads": 0, "scientific_calls": 0}
    review = {**common, "version": "otto-residual-seed-review-v1", "status": "reserved_before_run",
              "exact_seed_hits": 0, "resolved_block_hits": reviewed}
    reservation = {**common, "version": "otto-residual-estimator-seed-reservation-v1", "attempt": 2,
                   "status": "candidate_blocks_require_review", "changed_files_during_scan": [],
                   "exact_seed_hit_count": 0, "files": {hit["path"]: {"sha256": "0" * 64, "bytes": 1} for hit in hits},
                   "files_scanned": 5, "array_decodes": 0, "model_calls": 0, "native_calls": 0,
                   "project_module_imports": 0, "block_hits": hits}
    return review, reservation


def test_seed_review_exact18_reserved_environment_seeds_excludes_reused_fit_seeds():
    review, reservation = seed_review()
    M.validate_seed_review(review, reservation)
    assert len(review["fresh_environment_seeds"]) == 18
    assert review["reused_fit_seeds"] == [309000001, 309000002, 309000003]


@pytest.mark.parametrize("defect", ("missing_seed", "reordered", "overlap", "wrong_range", "empty_inventory",
                                  "unresolved_hit", "changed_context", "false_verification", "wrong_classification",
                                  "changed_during_scan", "wrong_scope", "model_call", "claim_admission", "wrong_attempt"))
def test_seed_review_rejects_changed_allocation_or_uncleared_overlap(defect):
    review, reservation = copy.deepcopy(seed_review())
    if defect == "missing_seed":
        review["fresh_environment_seeds"].pop()
    elif defect == "reordered":
        review["fresh_environment_seeds"].reverse()
    elif defect == "overlap":
        review["exact_seed_hits"] = 1
    elif defect == "wrong_range":
        review["allocations"]["dev"]["lambda3"][1] += 1
    elif defect == "empty_inventory":
        reservation["files"] = {}
    elif defect == "unresolved_hit":
        review["resolved_block_hits"].pop()
    elif defect == "changed_context":
        review["resolved_block_hits"][0]["context"] = "changed"
    elif defect == "false_verification":
        review["resolved_block_hits"][0]["source_verified"] = False
    elif defect == "wrong_classification":
        review["resolved_block_hits"][0]["classification"] = "probably safe"
    elif defect == "changed_during_scan":
        reservation["changed_files_during_scan"] = ["changed.py"]
    elif defect == "wrong_scope":
        review["scope"] = "different"
    elif defect == "model_call":
        reservation["model_calls"] = 1
    elif defect == "claim_admission":
        review["admits_execution"] = True
    else:
        reservation["attempt"] = 1
    with pytest.raises(ValueError):
        M.validate_seed_review(review, reservation)


def engineering_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "ROOT", tmp_path)
    sources = {}
    for name in (*sorted(M.NEW_COMPONENTS), "extra-qualified.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fabricated source\n")
        sources[name] = M.descriptor(path)
    directory = tmp_path / "qualification"
    directory.mkdir()
    (directory / "pytest-temp").mkdir()
    (directory / "pytest-temp" / "fabricated.txt").write_text("fixture only")
    (directory / "command-1.log").write_text("fabricated pass\n")
    log = M.descriptor(directory / "command-1.log")
    record = {"status": "passed", "source_before": dict(sources), "source_after": dict(sources),
              "commands": [{"returncode": 0, "timed_out": False, "reaped": True,
                            "group_absent": True,
                            "log": "command-1.log", **log}],
              "files": {"command-1.log": log}}
    path = directory / "receipt.json"
    path.write_text(json.dumps(record))
    return path, record, sources


def test_engineering_pins_qualified_superset_and_only_named_fixture_exclusion(tmp_path, monkeypatch):
    path, _record, expected = engineering_fixture(tmp_path, monkeypatch)
    admitted = {}
    M.authenticate_engineering(path, admitted)
    assert admitted == {name: value["sha256"] for name, value in expected.items()}


@pytest.mark.parametrize("defect", ("failed", "timeout", "unreaped", "group_present", "exit", "changed_source", "wrong_bytes",
                                  "missing_component", "extra_file", "extra_directory", "fixture_symlink",
                                  "changed_log", "wrong_command_log"))
def test_engineering_failure_or_manifest_drift_cannot_admit(defect, tmp_path, monkeypatch):
    path, record, _expected = engineering_fixture(tmp_path, monkeypatch)
    if defect == "failed":
        record["status"] = "failed"
    elif defect in ("timeout", "unreaped", "group_present", "exit"):
        key, value = {"timeout": ("timed_out", True), "unreaped": ("reaped", False),
                      "group_present": ("group_absent", False), "exit": ("returncode", 1)}[defect]
        record["commands"][0][key] = value
    elif defect == "changed_source":
        (tmp_path / "extra-qualified.py").write_text("# changed bytes\n")
    elif defect == "wrong_bytes":
        for name in ("source_before", "source_after"):
            record[name] = {key: dict(value) for key, value in record[name].items()}
            record[name]["extra-qualified.py"]["bytes"] += 1
    elif defect == "missing_component":
        for name in ("source_before", "source_after"):
            record[name].pop(M.TEST)
    elif defect == "extra_file":
        (path.parent / "extra.log").write_text("undeclared")
    elif defect == "extra_directory":
        (path.parent / "untracked").mkdir()
    elif defect == "fixture_symlink":
        (path.parent / "pytest-temp").rename(path.parent / "old-fixture")
        (path.parent / "pytest-temp").symlink_to(path.parent / "old-fixture", target_is_directory=True)
    elif defect == "changed_log":
        (path.parent / "command-1.log").write_text("changed log")
    else:
        record["commands"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        M.authenticate_engineering(path, {})


def test_qualification_cannot_replace_inherited_source(tmp_path, monkeypatch):
    path, _record, _expected = engineering_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="replace inherited source"):
        M.authenticate_engineering(path, {"extra-qualified.py": "0" * 64})


def valid_plan():
    return {"version": M.VERSION, "status": "frozen_before_collection", "phase": "dev",
            "configuration": copy.deepcopy(M.CONFIGURATION), "limits": dict(M.LIMITS),
            "call_caps": dict(M.CALL_CAPS), "cohort": M.cohort(),
            "execution_cohort": M.execution_cohort("dev"), "payloads": sorted(M.PAYLOADS)}


@pytest.mark.parametrize("defect", ("phase", "confirm_roster", "confirm_payload", "cap", "native_cap",
                                  "partial_dev", "master_index", "reserved_seed", "period", "rotation"))
def test_plan_cannot_change_master_roster_or_dev_only_recipe(defect):
    plan = valid_plan()
    M.validate_plan(plan)
    if defect == "phase":
        plan["phase"] = "confirm"
    elif defect == "confirm_roster":
        plan["execution_cohort"] = M.cohort("confirm")
    elif defect == "confirm_payload":
        plan["payloads"].append("confirm.npz")
    elif defect == "cap":
        plan["limits"]["native_seconds"] = 3601
    elif defect == "native_cap":
        plan["call_caps"]["native_reset"] = 54
    elif defect == "partial_dev":
        plan["execution_cohort"].pop()
    elif defect == "master_index":
        plan["cohort"][18]["episode_index"] = 0
    elif defect == "reserved_seed":
        plan["cohort"][-1]["seed"] += 1
    elif defect == "period":
        plan["configuration"]["collector_period"] = 8
    else:
        plan["cohort"][18:21] = reversed(plan["cohort"][18:21])
    with pytest.raises(ValueError):
        M.validate_plan(plan)


@pytest.mark.parametrize("phase", ("confirm", "test", "train", None))
def test_body_and_setup_reject_non_dev_before_native_setup(phase, tmp_path):
    run = M.Run(SimpleNamespace(output=tmp_path))
    run.plan = {**valid_plan(), "phase": phase}
    # No reference/runtime objects exist. Both entry points must fail first.
    with pytest.raises(ValueError, match="separately admitted future phase"):
        run.body()
    with pytest.raises(ValueError, match="separately admitted future phase"):
        run.setup()
    assert run.ledger.sequence == 0 and list(tmp_path.iterdir()) == []


def test_metadata_freeze_contains_full_master_and_only_dev_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "ROOT", tmp_path)
    values = {"output": tmp_path / "plan.json"}
    for role in M.ROLES:
        path = tmp_path / (role + ".json")
        path.write_text('{"fabricated": true}\n')
        values[role], values[role + "_sha256"] = path, M.descriptor(path)["sha256"]
    seen = []
    def auth(inputs):
        seen.append(inputs)
        return {"fabricated.py": "0" * 64}, {"fabricated": True}, {}, None, None
    monkeypatch.setattr(M, "authenticate_inputs", auth)
    M.freeze(SimpleNamespace(**values))
    plan = json.loads(values["output"].read_text())
    M.validate_plan(plan)
    assert len(seen) == 1 and set(seen[0]) == M.ROLES
    assert len(plan["cohort"]) == 54 and len(plan["execution_cohort"]) == 18
    assert not (tmp_path / "dev.npz").exists() and not (tmp_path / "confirm.npz").exists()


def seed_files_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "ROOT", tmp_path)
    review, reservation = copy.deepcopy(seed_review())
    for hit in reservation["block_hits"]:
        path = tmp_path / hit["path"]
        path.write_text(hit["context"] + "\n")
        reservation["files"][hit["path"]] = M.descriptor(path)
    previous = tmp_path / M.PREVIOUS_RESERVATION
    previous.parent.mkdir(parents=True)
    previous.write_text('{"status": "candidate_blocks_require_review", "fabricated": true}\n')
    prior_pin = M.descriptor(previous)
    monkeypatch.setattr(M, "PREVIOUS_RESERVATION_PIN", prior_pin["sha256"])
    reservation["supersedes"] = {"path": M.PREVIOUS_RESERVATION, **prior_pin}
    reserved = tmp_path / M.RESERVATION
    reserved.write_text(json.dumps(reservation))
    pin = M.descriptor(reserved)
    monkeypatch.setattr(M, "RESERVATION_PIN", pin["sha256"])
    review["reservation"] = {"path": M.RESERVATION, **pin}
    reviewed = tmp_path / M.SEED_REVIEW
    reviewed.write_text(json.dumps(review))
    monkeypatch.setattr(M, "SEED_REVIEW_PIN", M.descriptor(reviewed)["sha256"])
    return reviewed, review, reservation


def test_seed_authentication_binds_both_originals_and_all_resolved_match_sources(tmp_path, monkeypatch):
    path, _review, reservation = seed_files_fixture(tmp_path, monkeypatch)
    sources = {}
    M.authenticate_seed_review(path, sources)
    assert set(sources) == {M.SEED_REVIEW, M.RESERVATION, M.PREVIOUS_RESERVATION,
                            *(hit["path"] for hit in reservation["block_hits"])}
    assert all(M.descriptor(name)["sha256"] == pin for name, pin in sources.items())


@pytest.mark.parametrize("defect", ("review", "reservation", "previous", "match_source", "conflict"))
def test_seed_authentication_rejects_changed_originals_or_reviewed_sources(defect, tmp_path, monkeypatch):
    path, _review, reservation = seed_files_fixture(tmp_path, monkeypatch)
    sources = {}
    if defect == "conflict":
        sources[M.RESERVATION] = "0" * 64
    else:
        name = {"review": M.SEED_REVIEW, "reservation": M.RESERVATION, "previous": M.PREVIOUS_RESERVATION,
                "match_source": reservation["block_hits"][0]["path"]}[defect]
        changed = tmp_path / name
        changed.write_text(changed.read_text() + " ")
    with pytest.raises(ValueError):
        M.authenticate_seed_review(path, sources)


def test_fake_native_complete_dev_body_has18_resets_no_confirmation_and_one_census_each(tmp_path, monkeypatch):
    """Exercise the actual episode/body code using one-step in-memory fakes."""
    monkeypatch.setattr(M, "ROOT", tmp_path)
    run = M.Run(SimpleNamespace(output=tmp_path))
    run.plan, run.check = valid_plan(), lambda: None
    events, reset_seeds = [], []
    kernel = np.zeros((1,), np.float64)

    class Environment:
        def __init__(self, seed):
            self.seed, self.p_Poisson = seed, kernel.copy()
            self.source = np.asarray([1, 2])
            self.position, self.done = [0, 0], False
            self.draw_log = [{"channel": "source", "index": 0, "uniform": .25,
                              "selected_index": 55, "cdf_mass": .5}]

        def step(self, action, quiet):
            assert quiet and action in range(4)
            self.position, self.done = [1, 0], True
            events.append("step")
            return 0, .5, True

    def reset(_source, seed, configuration, initial_hit):
        assert configuration["Ngrid"] == 53 and initial_hit in (1, 2, 3)
        reset_seeds.append(seed)
        return Environment(seed)

    def observation(env, _step):
        return {"position": list(env.position), "valid_actions": (0, 1, 2, 3), "hit": 0, "done": env.done}

    class Analytic:
        def __init__(self, current, _kernel, allow_stay):
            assert not allow_stay
            self.public, self.belief = current, np.zeros(1, np.float64)
            self._view = self
            self._pending_action = None
            self._policy = SimpleNamespace(_value_policy=lambda: (1, np.asarray([4., 0., 2., 3.])))

        def update(self, action, after):
            assert action in range(4)
            self.public = after
            events.append("update")

    class Teacher:
        def __init__(self, env, model, sym_avg):
            assert isinstance(env, Analytic) and sym_avg
            self.model = model

        def _value_policy(self):
            return 0, self.model()

    def recorder(_model, _code, ledger, _np):
        def forward():
            def result():
                ledger.emit("forwards.jsonl", {"event": "fabricated_forward"})
                return np.asarray([0., 4., 2., 3.], np.float32)
            return ledger.call("tensorflow_value", result)
        return forward

    def features(_view, _current, _regime, _analytic, last_action, last_query, previous):
        assert last_action is previous is None and last_query == 0
        values = np.zeros(31, np.float32)
        values[17] = 1
        return values, None

    runtime = SimpleNamespace(np=np, source=None, analytic=Analytic, policy=Teacher, model=None,
                              public=SimpleNamespace(seeded_environment=reset, observation=observation))
    run.reference = SimpleNamespace(packet=lambda value: value, ForwardRecorder=recorder,
                                    belief_witness=lambda *_args: {})
    run.kernels = {"lambda3": kernel, "lambda4": kernel}
    run.readonly, run.features, run.select = lambda value: value, features, select
    run.analytic_scores = lambda scores, allowed: (select(scores, allowed), scores)
    run.original_setup_wall = run.feature_module_seconds = 0.

    def setup():
        for name in ("tensorflow_construction", "tensorflow_build", "tensorflow_load"):
            run.ledger.call(name, lambda: None)
        run.ledger.emit("weights.jsonl", {"event": "fabricated_weights"})
        return runtime
    run.setup = setup
    run.body()
    run.ledger.close()
    assert reset_seeds == [row["seed"] for row in M.execution_cohort("dev")]
    assert len(run.rows) == 18 and set(run.datasets) == {"dev"}
    assert events == [value for _ in range(18) for value in ("step", "update")]
    for name in ("native_reset", "native_step", "teacher_score", "tensorflow_value", "actor_update"):
        assert run.ledger.calls[name]["attempted"] == run.ledger.calls[name]["returned"] == 18
    assert not run.ledger.pending and run.ledger.pending_emission is None
    assert not (tmp_path / "confirm.npz").exists() and not (tmp_path / "test.npz").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["phase"] == "dev" and summary["confirm_episodes"] == 0
    assert summary["teacher_calls"] == 18 and summary["deployed_queries"] == 12 and summary["annotation_only"] == 6
    with np.load(tmp_path / "dev.npz", allow_pickle=False) as archive:
        assert archive["episode_offsets"].tolist() == list(range(19))
    with (tmp_path / "episode-boundaries.jsonl").open() as stream:
        boundaries = [json.loads(line) for line in stream]
    assert [row["event"] for row in boundaries] == [value for _ in range(18) for value in ("attempt", "return")]


@pytest.mark.parametrize("defect", ("external", "relative", "dotdot", "existing_file", "existing_directory",
                                  "symlink_ancestor", "dangling_leaf", "missing_parent", "file_parent"))
def test_unsafe_output_rejected_before_plan_authentication_or_worker_bind(defect, tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(M, "ROOT", root)
    path = root / "output"
    if defect == "external":
        path = tmp_path / "external-output"
    elif defect == "relative":
        path = Path("relative-output")
    elif defect == "dotdot":
        parent = root / "parent"
        parent.mkdir()
        path = parent / ".." / "output"
    elif defect == "existing_file":
        path.write_text("preserve existing bytes")
    elif defect == "existing_directory":
        path.mkdir()
    elif defect == "symlink_ancestor":
        target = root / "target"
        target.mkdir()
        link = root / "link"
        link.symlink_to(target, target_is_directory=True)
        path = link / "output"
    elif defect == "dangling_leaf":
        path.symlink_to(root / "missing-target")
    elif defect == "missing_parent":
        path = root / "missing-parent" / "output"
    else:
        parent = root / "file-parent"
        parent.write_text("not a directory")
        path = parent / "output"
    events = []
    monkeypatch.setattr(M, "authenticate_inputs", lambda _inputs: events.append("authenticated"))
    args = SimpleNamespace(output=path)
    with pytest.raises(ValueError):
        M.freeze(args)
    run = M.Run(args)
    run.bind = lambda: events.append("bound")
    run.body = lambda: events.append("native")
    with pytest.raises(ValueError):
        run.execute()
    assert events == [] and run.rows == [] and run.ledger.sequence == 0
    if defect == "existing_file":
        assert path.read_text() == "preserve existing bytes"


def test_contained_absent_output_and_future_supervision_are_valid_metadata_locations(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "ROOT", tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n")
    args = SimpleNamespace(plan=plan, supervision=tmp_path / "future-launch.json")
    assert M.exclusive_output(tmp_path / "future-run") == tmp_path / "future-run"
    M.validate_worker_paths(args)
    assert not args.supervision.exists() and not (tmp_path / "future-run").exists()
    args.supervision.write_text("{}\n")
    M.validate_worker_paths(args)


@pytest.mark.parametrize("role,defect", [(role, defect) for role in ("plan", "supervision")
                                       for defect in ("external", "symlink", "directory", "missing_parent")])
def test_worker_input_locations_fail_before_output_creation_and_bind(role, defect, tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(M, "ROOT", root)
    plan, launch = root / "plan.json", root / "launch.json"
    plan.write_text("{}\n")
    launch.write_text("{}\n")
    bad = root / "bad-input"
    if defect == "external":
        bad = tmp_path / "outside.json"
        bad.write_text("{}\n")
    elif defect == "symlink":
        bad.symlink_to(plan)
    elif defect == "directory":
        bad.mkdir()
    else:
        bad = root / "missing-parent" / "input.json"
    args = SimpleNamespace(output=root / "output", plan=plan, supervision=launch)
    setattr(args, role, bad)
    events = []
    run = M.Run(args)
    run.bind = lambda: events.append("bound")
    run.body = lambda: events.append("native")
    with pytest.raises(ValueError):
        run.execute()
    assert events == [] and not args.output.exists()
