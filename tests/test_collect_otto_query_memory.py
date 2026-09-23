"""Fabricated routing, durability and census export only; no native/model calls."""
import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "query_memory_collector_tests", Path(__file__).resolve().parents[1] / "scripts/collect_otto_query_memory.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def select(scores, allowed):
    subset = scores[list(allowed)]
    return allowed[int(np.flatnonzero(np.abs(subset - subset.min()) < 1e-10)[0])]


def test_all_three_splits_are_disjoint_complete_paired_cases():
    rows = M.cohort()
    assert M.VERSION == "otto-query-memory-collection-v1"
    assert M.CASES == {"train": 9, "dev": 3, "test": 6}
    assert len(rows) == M.EPISODES == 108
    assert [r["episode_index"] for r in rows] == list(range(108))
    assert len({r["episode_id"] for r in rows}) == 108
    seeds = []
    for stage, size in (("train", 54), ("dev", 18), ("test", 36)):
        assert len(M.cohort(stage)) == M.STAGE_EPISODES[stage] == size
        for regime, first in M.FIRST[stage].items():
            expected = set(range(first, first + M.CASES[stage]))
            assert {r["seed"] for r in rows if r["stage"] == stage and r["regime"] == regime} == expected
            seeds.extend(expected)
    assert len(seeds) == len(set(seeds)) == 36
    assert not set(seeds) & set(M.FIT_SEEDS)
    for case_index in range(36):
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
    assert {"train.npz", "dev.npz", "test.npz"} <= M.PAYLOADS
    assert len(M.PAYLOADS) == 17
    assert not hasattr(M.Run, "annotate_train")


@pytest.mark.parametrize("stage", ("train", "dev", "test"))
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


@pytest.mark.parametrize("stage", ("train", "dev", "test"))
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


@pytest.mark.parametrize("stage", ("train", "dev", "test"))
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


def test_each_stage_offset_restarts_and_prior_stage_arrays_do_not_leak(tmp_path, monkeypatch):
    run = M.Run(SimpleNamespace(output=tmp_path))
    run.runtime, run.check = SimpleNamespace(np=np), lambda: None
    monkeypatch.setattr(M, "ROOT", tmp_path)
    for stage in M.FIRST:
        fill_stage(run, stage)
        run.save_stage(stage)
    assert len(run.rows) == 108 and set(run.datasets) == {"train", "dev", "test"}
    for stage, size in M.STAGE_EPISODES.items():
        with np.load(tmp_path / f"{stage}.npz", allow_pickle=False) as saved:
            assert saved["episode_offsets"].tolist() == list(range(size + 1))


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
    M.paired_identity({**common, "stage": "train"}, paired)
    M.paired_identity({**common, "stage": "dev", "source_evaluation_only": [3, 4]}, paired)
    with pytest.raises(ValueError, match="paired original"):
        M.paired_identity({**common, "stage": "train", "source_evaluation_only": [3, 4]}, paired)


def test_full_census_caps_include_every_split_and_forbid_replay():
    assert M.CALL_CAPS["teacher_score"] == M.CALL_CAPS["tensorflow_value"] == 236304
    for name in ("native_step", "actor_update", "analytic_score", "feature_build"):
        assert M.CALL_CAPS[name] == 108 * 2188
    for name in ("native_reset", "actor_construction", "backend_binding"):
        assert M.CALL_CAPS[name] == 108
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
    return {"status": "reserved_before_run", "hits": [], "block_hits": [],
            "files": {"fabricated-prior.json": {"sha256": "0" * 64, "bytes": 1}},
            "seeds": seeds + list(M.FIT_SEEDS),
            "allocations": {**{stage: {regime: [first, first + M.CASES[stage] - 1]
                                       for regime, first in regimes.items()} for stage, regimes in M.FIRST.items()},
                            "fits": list(M.FIT_SEEDS)}}


def test_seed_review_admits_exact39_reserved_fresh_seeds():
    record = seed_review()
    assert len(record["seeds"]) == 39
    assert record["seeds"][:9] == list(range(303000001, 303000010))
    assert record["seeds"][-3:] == [309000001, 309000002, 309000003]
    M.validate_seed_review(record)


@pytest.mark.parametrize("defect", ("missing_seed", "reordered", "overlap", "block_overlap", "wrong_range", "empty_inventory"))
def test_seed_review_rejects_changed_allocation_or_uncleared_overlap(defect):
    record = seed_review()
    if defect == "missing_seed":
        record["seeds"].pop()
    elif defect == "reordered":
        record["seeds"].reverse()
    elif defect == "overlap":
        record["hits"] = [303000001]
    elif defect == "block_overlap":
        record["block_hits"] = [303000099]
    elif defect == "wrong_range":
        record["allocations"]["dev"]["lambda3"][1] += 1
    else:
        record["files"] = {}
    with pytest.raises(ValueError, match="fresh seed reservation"):
        M.validate_seed_review(record)


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


@pytest.mark.parametrize("defect", ("failed", "timeout", "unreaped", "exit", "changed_source", "wrong_bytes",
                                  "missing_component", "extra_file", "extra_directory", "fixture_symlink",
                                  "changed_log", "wrong_command_log"))
def test_engineering_failure_or_manifest_drift_cannot_admit(defect, tmp_path, monkeypatch):
    path, record, _expected = engineering_fixture(tmp_path, monkeypatch)
    if defect == "failed":
        record["status"] = "failed"
    elif defect in ("timeout", "unreaped", "exit"):
        key, value = {"timeout": ("timed_out", True), "unreaped": ("reaped", False), "exit": ("returncode", 1)}[defect]
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
