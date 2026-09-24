"""Fabricated runner integration only; no empirical input, native or teacher IO."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("residual_reanalysis_runner_tests", Path(__file__).parents[1] / "scripts/run_otto_residual_reanalysis.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
SEEDS = (309000001, 309000002, 309000003)
SHAPES = {"recurrent.weight_ih_l0": (84, 40), "recurrent.weight_hh_l0": (84, 28),
          "recurrent.bias_ih_l0": (84,), "recurrent.bias_hh_l0": (84,), "output.weight": (4, 28),
          "output.bias": (4,), "action_residual.weight": (4, 28), "action_residual.bias": (4,)}


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def byte_pin(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def synthetic_states():
    rng = np.random.default_rng(4703)
    states = {}
    for seed in SEEDS:
        pretrained = {"slow." + name: (rng.standard_normal(shape) * .025).astype(np.float32)
                      for name, shape in SHAPES.items()}
        states["pretrained", seed] = pretrained
        states["joint_aux", seed] = {"slow." + name: (rng.standard_normal(shape) * .03).astype(np.float32)
                                     for name, shape in SHAPES.items()}
        states["trace_delta", seed] = {**{name: value.copy() for name, value in pretrained.items()},
                                       "projection.weight": (rng.standard_normal((8, 28)) * .04).astype(np.float32)}
    return states


def fabricated_census():
    identities = []
    arms = ("analytic", "neural", "period4_hold")
    for regime, first in (("lambda3", 314000001), ("lambda4", 315000001)):
        for case in range(3):
            global_case = len(identities) // 3
            for arm in arms[global_case % 3:] + arms[:global_case % 3]:
                identities.append({"stage": "dev", "regime": regime, "seed": first + case, "case": case,
                    "arm": arm, "episode_index": len(identities), "initial_hit": 1 + case % 3,
                    "episode_id": f"dev:{regime}:{first + case}:{arm}"})
    lengths = (1, 4, 5, 6, 9, 33) * 3
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    rows = int(offsets[-1])
    values = np.linspace(-.1, .2, rows * 31, dtype=np.float32).reshape(rows, 31)
    targets = np.empty((rows, 4), np.float32)
    legal = np.ones((rows, 4), np.bool_)
    corrections = np.zeros(rows, np.bool_)
    for episode, (low, high) in enumerate(pairwise(offsets)):
        steps = np.arange(high - low)
        values[low:high, 15] = (steps / 2188).astype(np.float32)
        values[low:high, 16] = ((steps % 4) / 2188).astype(np.float32)
        values[low:high, 17] = 1
        targets[low:high] = np.array([1., 2., 4., 3.], np.float32) + steps[:, None] * np.array([.125, -.25, .25, -.125], np.float32)
        targets[low, 0] = np.float32(-0.)
        corrections[low:high] = steps % 4 == 0
        legal[low:high, 3] = episode % 2 == 0
    flat = {"features": values, "raw_q": targets, "legal": legal,
            "actions": np.zeros(rows, np.int64), "correction": corrections, "episode_offsets": offsets}
    return flat, identities


class Clock:
    def __init__(self):
        self.tick = 1000

    def now_ns(self):
        self.tick += 1000
        return self.tick


def fabricated_run(root, phase, collection, lineage, *, producer=None):
    output = root / phase
    output.mkdir()
    run = M.Run(SimpleNamespace(output=output))
    run.np, run.clock, run.check = np, Clock(), lambda **_kwargs: None
    run.plan = {"phase": phase, "limits": M.LIMITS[phase]}
    run.bound = {"collection": collection, "lineage": lineage, "producer": producer}
    return run


def setup_evaluation(tmp_path, monkeypatch, states=None):
    monkeypatch.setattr(M, "ROOT", tmp_path.resolve())
    root = tmp_path.resolve()
    flat, identities = fabricated_census()
    collection_dir = root / "collection"
    collection_dir.mkdir()
    with (collection_dir / "dev.npz").open("wb") as stream:
        np.savez_compressed(stream, **flat)
    collection = ({"execution_cohort": identities}, {"files": {"dev.npz": byte_pin(collection_dir / "dev.npz")}}, collection_dir)
    checkpoint_dir = root / "fabricated-checkpoints"
    checkpoint_dir.mkdir()
    records = {family: {} for family in ("pretrained", "joint_aux", "trace_delta")}
    states = synthetic_states() if states is None else states
    for (family, seed), arrays in states.items():
        path = checkpoint_dir / f"{family}-{seed}.npz"
        with path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        records[family][str(seed)] = {"path": str(path), **byte_pin(path)}
    run = fabricated_run(root, "evaluate", collection, {"checkpoints": records})
    return run, flat, identities


def test_all_nine_checkpoint_files_validate_before_use_and_trace_slow_matches():
    states = synthetic_states()
    M.validate_checkpoints(np, states, SHAPES)
    assert len(states) == 9
    assert not np.array_equal(states["pretrained", SEEDS[0]]["slow.output.weight"],
                              states["joint_aux", SEEDS[0]]["slow.output.weight"])


@pytest.mark.parametrize("defect", ("missing", "extra", "wrong_seed", "extra_tensor", "missing_tensor",
                                   "dtype", "shape", "nonfinite", "projection_shape", "trace_slow_bits"))
def test_checkpoint_roster_dtype_shape_finiteness_and_same_seed_frozen_guard(defect):
    states = synthetic_states()
    final = states["trace_delta", SEEDS[-1]]
    if defect == "missing":
        states.pop(("joint_aux", SEEDS[-1]))
    elif defect == "extra":
        states["other", SEEDS[-1]] = final
    elif defect == "wrong_seed":
        states["trace_delta", 999] = states.pop(("trace_delta", SEEDS[-1]))
    elif defect == "extra_tensor":
        final["hidden_teacher"] = np.zeros(1, np.float32)
    elif defect == "missing_tensor":
        final.pop("projection.weight")
    elif defect == "dtype":
        final["projection.weight"] = final["projection.weight"].astype(np.float64)
    elif defect == "shape":
        final["slow.output.bias"] = np.zeros((1, 4), np.float32)
    elif defect == "nonfinite":
        final["projection.weight"][-1, -1] = np.nan
    elif defect == "projection_shape":
        final["projection.weight"] = final["projection.weight"].T.copy()
    else:
        final["slow.output.bias"][0] += .125
    with pytest.raises(ValueError):
        M.validate_checkpoints(np, states, SHAPES)


def test_last_checkpoint_defect_prevents_every_model_construction(tmp_path, monkeypatch):
    from openjev.research import otto_scheduled_predictor as predictor
    states = synthetic_states()
    states["trace_delta", SEEDS[-1]]["projection.weight"] = np.zeros((8, 28), np.float64)
    run, _flat, _identities = setup_evaluation(tmp_path, monkeypatch, states)
    history, identities = run.inputs()
    monkeypatch.setattr(predictor, "from_state", lambda *_a, **_k: pytest.fail("invalid last file reached a model constructor"))
    with pytest.raises(ValueError, match="dtype and shape"):
        run.evaluate(history, identities)
    assert run.receipt["counts"]["checkpoint_decodes"] == 9
    assert run.receipt["counts"]["model_constructions"] == 0
    assert not list(run.out.glob("cache-*.npz"))


def test_record_comparison_normalizes_relative_absolute_paths_and_checks_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "ROOT", tmp_path.resolve())
    absolute = json_file(tmp_path / "evidence.json", {"fabricated": True})
    relative = {**absolute, "path": "evidence.json"}
    assert M.same_record(absolute, relative)
    assert not M.same_record(absolute, {**relative, "bytes": relative["bytes"] + 1})
    assert not M.same_record(absolute, {**relative, "sha256": "0" * 64})
    M.verify_records({"one": absolute, "two": relative})


@pytest.mark.parametrize("defect", ("sha", "bytes", "extra", "symlink"))
def test_input_descriptors_fail_before_any_numerical_access(tmp_path, monkeypatch, defect):
    monkeypatch.setattr(M, "ROOT", tmp_path.resolve())
    record = json_file(tmp_path / "evidence.json", {"fabricated": True})
    if defect == "sha":
        record["sha256"] = "0" * 64
    elif defect == "bytes":
        record["bytes"] += 1
    elif defect == "extra":
        record["ignored"] = True
    else:
        alias = tmp_path / "alias.json"
        alias.symlink_to(record["path"])
        record["path"] = str(alias)
    with pytest.raises(ValueError):
        M.verify_records({"evidence": record})


@pytest.mark.parametrize("phase", ("confirm", "test", "train", None, True))
def test_unsupported_phase_rejected_before_evidence_or_module_access(phase, monkeypatch):
    monkeypatch.setattr(M, "verify_records", lambda *_a: pytest.fail("unsupported phase reached evidence"))
    monkeypatch.setattr(M, "load", lambda *_a: pytest.fail("unsupported phase imported a module"))
    with pytest.raises(ValueError, match="DEV evaluation or audit only"):
        M.authenticate(phase, {})


@pytest.mark.parametrize("defect", ("confirm", "old_test", "wrong_count"))
def test_inputs_reject_nondev_identity_roster_before_array_path_decode(tmp_path, monkeypatch, defect):
    run, _flat, _identities = setup_evaluation(tmp_path, monkeypatch)
    roster = run.bound["collection"][0]["execution_cohort"]
    if defect == "wrong_count":
        roster.pop()
    else:
        roster[-1]["stage"] = "confirm" if defect == "confirm" else "test"
    monkeypatch.setattr(run, "decode", lambda *_a, **_k: pytest.fail("wrong phase reached array decoder"))
    with pytest.raises(ValueError, match="DEV only"):
        run.inputs()


def test_decode_rechecks_bytes_before_np_load_and_preserves_parent_context(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "ROOT", tmp_path.resolve())
    run = fabricated_run(tmp_path, "evaluate", None, {})
    path = tmp_path / "fake.npz"
    path.write_bytes(b"not an archive")
    run.receipt["pending"] = {"operation": "view", "seed": SEEDS[0]}
    run.np = SimpleNamespace(load=lambda *_a, **_k: pytest.fail("wrong hash reached np.load"))
    with pytest.raises(ValueError, match="immediately before decode"):
        run.decode(path, {"bytes": 0, "sha256": "0" * 64})
    assert run.receipt["pending_io"] is None and run.receipt["pending"]["operation"] == "view"
    error = OSError("fabricated archive error")
    def broken(*_a, **_k):
        raise error
    run.np = SimpleNamespace(load=broken)
    with pytest.raises(OSError) as caught:
        run.decode(path, byte_pin(path))
    assert caught.value is error
    assert run.receipt["pending_io"]["operation"] == "decode"
    assert run.receipt["pending_io"]["parent"] == run.receipt["pending"]
    assert run.receipt["counts"]["array_decodes"] == 0


def test_failed_save_and_failed_log_keep_both_parent_and_child_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "ROOT", tmp_path.resolve())
    run = fabricated_run(tmp_path, "evaluate", None, {})
    run.receipt["pending"] = {"operation": "view", "seed": SEEDS[0], "method": "pretrained", "tau": None}
    error = OSError("fabricated fsync failure")
    def fail(_fd):
        raise error
    monkeypatch.setattr(M.os, "fsync", fail)
    with pytest.raises(OSError) as caught:
        run.save_arrays("partial.npz", {"x": np.arange(4, dtype=np.float32)})
    assert caught.value is error and run.receipt["pending"]["operation"] == "view"
    assert run.receipt["pending_emission"]["parent"] == run.receipt["pending"]
    assert run.receipt["counts"]["array_decodes"] == 0
    event = {"event": "view_complete", "seed": SEEDS[0]}
    with pytest.raises(OSError):
        run.event(event)
    assert run.receipt["pending_log"] == event
    assert run.receipt["views_completed"] == 0


def test_save_roundtrip_owns_arrays_and_does_not_clear_active_work(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "ROOT", tmp_path.resolve())
    run = fabricated_run(tmp_path, "evaluate", None, {})
    parent = {"operation": "cache", "seed": SEEDS[0]}
    run.receipt["pending"] = parent.copy()
    values = {"signed": np.array([-0., 1.], np.float32), "mask": np.array([True, False])}
    before = {key: value.tobytes() for key, value in values.items()}
    expected = run.save_arrays("cache.npz", values)
    assert expected == byte_pin(run.out / "cache.npz")
    assert run.receipt["pending"] == parent and run.receipt["pending_io"] is None
    assert run.receipt["pending_emission"] is run.receipt["pending_log"] is None
    assert run.receipt["counts"]["array_decodes"] == 1
    assert {key: value.tobytes() for key, value in values.items()} == before


def test_equal_nested_rejects_missing_fields_nonfinite_and_boolean_numeric_aliases():
    M.equal_nested({"items": [1., None, True]}, {"items": [1. + 1e-14, None, True]})
    for actual, expected in (({}, {"extra": 1}), ({"x": float("nan")}, {"x": 1.}),
                             ({"x": True}, {"x": 1}), ({"x": 1}, {"x": True}),
                             ({"x": []}, {"x": [1]}), ({"x": 2.}, {"x": 1.})):
        with pytest.raises(ValueError):
            M.equal_nested(actual, expected)


def test_cache_work_counts_match_independent_boundary_roster():
    counts = M.expected_cache_work(np.array([0, 1, 5, 10, 41, 73, 106], np.int64))
    assert counts["pretrained_forward_chunks"] == counts["joint_forward_chunks"] == 7
    assert counts["projection_calls"] == counts["projection_rows"] == counts["projection_key_rows"] == 100
    assert counts["projection_linear_terms"] == 22400
    assert counts["cue_normalized_coordinates"] == 1600 and counts["cue_trace_mixed_coordinates"] == 800
    expected = {"active_rows": 106, "query_rows": 29, "later_query_rows": 23, "key_rows": 100,
                "nonquery_rows": 77, "recurrent_calls": 78, "recurrent_token_transitions": 129,
                "base_readout_calls": 49, "base_readout_rows": 100, "action_readout_calls": 26,
                "action_readout_rows": 77, "shadow_readout_calls": 23, "shadow_readout_rows": 23}
    for prefix in ("pretrained_slow_", "joint_slow_"):
        assert {name: counts[prefix + name] for name in expected} == expected
    for name in ("cue_key_normalizations", "cue_normalizations", "cue_trace_updates"):
        assert counts[name] == 100
    horizon = M.expected_cache_work(np.array([0, 2188], np.int64))
    assert horizon["pretrained_forward_chunks"] == 69
    assert horizon["pretrained_slow_recurrent_calls"] == 1640
    for offsets in ([], [0], [0, 0], [0, 2189]):
        with pytest.raises(ValueError):
            M.expected_cache_work(offsets)


def exact_counts(phase):
    values = {"checkpoint_decodes": 0, "array_decodes": 76, "model_constructions": 0,
              "model_construction_attempts": 0, "cache_builds": 0, "replays": 72,
              "teacher_calls": 0, "native_calls": 0, "optimizer_steps": 0, "confirm_decodes": 0}
    if phase == "evaluate":
        values.update(checkpoint_decodes=9, array_decodes=85, model_constructions=6,
                      model_construction_attempts=6, cache_builds=3)
    return values


@pytest.mark.parametrize("phase", ("evaluate", "audit"))
def test_exact_operation_caps_reject_missing_extra_bool_and_any_changed_counter(phase):
    expected = exact_counts(phase)
    M.validate_counts(phase, expected)
    for name in expected:
        changed = dict(expected)
        changed[name] += 1
        with pytest.raises(ValueError):
            M.validate_counts(phase, changed)
    for changed in ({**expected, "undeclared": 0}, {**expected, "native_calls": False},
                    {key: value for key, value in expected.items() if key != "confirm_decodes"}):
        with pytest.raises(ValueError):
            M.validate_counts(phase, changed)
    with pytest.raises(ValueError):
        M.validate_counts("confirm", expected)


@pytest.mark.parametrize("defect", ("existing", "outside", "relative", "traversal", "parent_symlink", "missing_parent"))
def test_output_rejected_before_mkdir_bind_or_import(tmp_path, monkeypatch, defect):
    root = tmp_path.resolve()
    monkeypatch.setattr(M, "ROOT", root)
    if defect == "existing":
        output = root / "exists"
        output.mkdir()
    elif defect == "outside":
        output = root.parent / (root.name + "-outside")
    elif defect == "relative":
        output = Path("relative")
    elif defect == "traversal":
        output = root / "sub" / ".." / "escaped"
        (root / "sub").mkdir()
    elif defect == "missing_parent":
        output = root / "missing" / "nested"
    else:
        (root / "target").mkdir()
        (root / "alias").symlink_to(root / "target", target_is_directory=True)
        output = root / "alias" / "new"
    run = M.Run(SimpleNamespace(output=output))
    monkeypatch.setattr(run, "bind", lambda: pytest.fail("bad output reached source/runtime admission"))
    with pytest.raises(ValueError, match="exclusive contained output"):
        run.execute()
    if defect != "existing":
        assert not output.exists()


def test_deadline_checked_even_when_resource_poll_is_throttled(tmp_path, monkeypatch):
    run = M.Run(SimpleNamespace(output=tmp_path))
    ticks, scans = iter((100, 101, 1000)), []
    run.clock = SimpleNamespace(now_ns=lambda: next(ticks))
    run.launch = {"deadline_ns": 1000}
    run.plan = {"limits": {"rss_bytes": 2**20, "output_bytes": 2**22}}
    def usage(_kind):
        scans.append(True)
        return SimpleNamespace(ru_maxrss=1)
    monkeypatch.setattr(M.resource, "getrusage", usage)
    run.check(force=True)
    run.check()
    assert len(scans) == 1
    with pytest.raises(ValueError, match="fixed deadline"):
        run.check()
    assert len(scans) == 1


def producer_auth_fixture(tmp_path, monkeypatch, defect=None):
    root = tmp_path.resolve()
    monkeypatch.setattr(M, "ROOT", root)
    runtime = {"fabricated_runtime": True}
    sources = {"fabricated_source": "1" * 64}
    lineage = {"sources": sources.copy(), "checkpoints": {}, "runtime": runtime.copy(), "evidence": {}}
    underlying = {role: json_file(root / (role + ".json"), {"fabricated": role}) for role in M.ROLES["evaluate"]}
    underlying["collection_terminal"] = json_file(root / "collection_terminal.json",
        {"finished_ns": 2001 if defect == "reversed_collection_chronology" else 1000})
    underlying["bridge_terminal"] = json_file(root / "bridge_terminal.json",
        {"finished_ns": 2001 if defect == "reversed_bridge_chronology" else 1500})
    plan = {"version": M.VERSION, "phase": "evaluate", "status": "frozen_before_execution", "configuration": M.CONFIG,
            "limits": M.LIMITS["evaluate"], "payloads": sorted(M.PAYLOADS["evaluate"]), "inputs": underlying,
            "sources": sources, "lineage": lineage, "runtime": runtime}
    if defect == "wrong_phase":
        plan["phase"] = "confirm"
    elif defect == "wrong_configuration":
        plan["configuration"] = {**M.CONFIG, "views": 71}
    elif defect == "lineage_changed":
        plan["lineage"] = {**lineage, "checkpoints": {"changed": True}}
    plan_pin = json_file(root / "producer-plan.json", plan)
    receipt = {"version": M.VERSION, "phase": "evaluate", "status": "completed", "complete": True,
               "plan_sha256": plan_pin["sha256"], "sources": sources, "inputs": underlying, "limits": M.LIMITS["evaluate"],
               "pending": None, "pending_io": None, "pending_emission": None, "pending_log": None,
               "views_completed": 72, "caches_completed": 3, "counts": exact_counts("evaluate"),
               "peak_rss_bytes": 1, "requires_successful_original_supervisor": True}
    if defect in ("pending", "pending_io", "pending_emission", "pending_log"):
        receipt[defect] = {"unfinished": True}
    elif defect == "incomplete":
        receipt["complete"] = False
    elif defect == "failed":
        receipt["status"] = "failed"
    elif defect == "missing_view":
        receipt["views_completed"] = 71
    elif defect == "missing_cache":
        receipt["caches_completed"] = 2
    elif defect == "extra_decode":
        receipt["counts"]["array_decodes"] += 1
    elif defect == "unfinished_model":
        receipt["counts"]["model_constructions"] -= 1
    elif defect == "unrecorded_attempt":
        receipt["counts"]["model_construction_attempts"] -= 1
    elif defect == "native_call":
        receipt["counts"]["native_calls"] = 1
    elif defect == "confirmation_read":
        receipt["counts"]["confirm_decodes"] = 1
    elif defect == "rss":
        receipt["peak_rss_bytes"] = 2**40
    elif defect == "no_parent_requirement":
        receipt["requires_successful_original_supervisor"] = False
    elif defect == "wrong_plan_pin":
        receipt["plan_sha256"] = "0" * 64
    worker_pin = json_file(root / "producer-receipt.json", receipt)
    terminal_pin = json_file(root / "producer-terminal.json", {"fabricated_terminal": True})
    calls = []
    def closed(path, received, expected):
        assert path == worker_pin["path"] and received == receipt and expected == M.PAYLOADS["evaluate"]
        calls.append("closed")
        return root
    def process(*args):
        calls.append(args)
        if defect == "parent_failed":
            raise ValueError("fabricated original parent failed")
        return {"started_ns": 2000}
    helper = SimpleNamespace(closed_files=closed, successful_process=process)
    def loaded(path, _name):
        if path == M.LINEAGE:
            return SimpleNamespace(authenticate=lambda: copy.deepcopy(lineage))
        assert path == M.OLD_PRODUCER
        return helper
    monkeypatch.setattr(M, "load", loaded)
    monkeypatch.setattr(M, "runtime_record", lambda: copy.deepcopy(runtime))
    monkeypatch.setattr(M, "engineering", lambda _path: sources.copy())
    monkeypatch.setattr(M, "authenticate_collection", lambda *_a: ({}, {}, root))
    return {"producer_plan": plan_pin, "producer_receipt": worker_pin, "producer_terminal": terminal_pin}, calls


def test_audit_authentication_requires_original_producer_closure_without_scientific_promotion(tmp_path, monkeypatch):
    inputs, calls = producer_auth_fixture(tmp_path, monkeypatch)
    result = M.authenticate("audit", inputs)
    assert result["producer"][0]["phase"] == "evaluate"
    assert calls[0] == "closed" and len(calls) == 2
    assert calls[1][1:] == ("producer", result["producer"][1], M.SELF, ".venv/bin/python", 900)


@pytest.mark.parametrize("defect", ("wrong_phase", "wrong_configuration", "lineage_changed", "pending", "pending_io",
    "pending_emission", "pending_log", "incomplete", "failed", "missing_view", "missing_cache", "extra_decode",
    "unfinished_model", "unrecorded_attempt", "native_call", "confirmation_read", "rss", "no_parent_requirement",
    "wrong_plan_pin", "parent_failed", "reversed_collection_chronology", "reversed_bridge_chronology"))
def test_audit_cannot_accept_partial_or_unclosed_producer_evidence(tmp_path, monkeypatch, defect):
    inputs, _calls = producer_auth_fixture(tmp_path, monkeypatch, defect)
    with pytest.raises(ValueError):
        M.authenticate("audit", inputs)


@pytest.mark.parametrize("defect", (None, "capacity_failed", "empirical_decode", "wrong_roster", "over_cap",
                                   "wrong_projection", "qualification_failed", "source_changed", "extra_file"))
def test_engineering_requires_source_bound_complete_fabricated_capacity(tmp_path, monkeypatch, defect):
    root = tmp_path.resolve()
    monkeypatch.setattr(M, "ROOT", root)
    sources = {}
    for name in M.COMPONENTS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fabricated qualification source\n")
        sources[name] = byte_pin(path)
    directory = root / "qualification"
    directory.mkdir()
    (directory / "pytest-temp").mkdir()
    capacity = {"status": "passed", "empirical_array_decodes": 0, "checkpoint_decodes": 0,
                "teacher_calls": 0, "native_calls": 0, "evaluate_seconds": 90., "evaluate_projected_seconds": 660.,
                "audit_seconds": 90., "audit_projected_seconds": 660., "episodes": 18, "length": 2188,
                "views": 24, "fit_seeds_measured": 1}
    if defect == "capacity_failed":
        capacity["status"] = "failed"
    elif defect == "empirical_decode":
        capacity["empirical_array_decodes"] = 1
    elif defect == "wrong_roster":
        capacity["length"] = 2187
    elif defect == "over_cap":
        capacity.update(evaluate_seconds=100., evaluate_projected_seconds=720.)
    elif defect == "wrong_projection":
        capacity["audit_projected_seconds"] = 120.
    json_file(directory / "capacity.json", capacity)
    (directory / "command-1.log").write_text("fabricated pass\n")
    log_pin = byte_pin(directory / "command-1.log")
    record = {"status": "passed", "source_before": sources, "source_after": sources, "sources_unchanged": True,
              "commands": [{"returncode": 0, "timed_out": False, "reaped": True, "group_absent": True,
                            "log": "command-1.log", **log_pin}],
              "files": {"capacity.json": byte_pin(directory / "capacity.json"), "command-1.log": log_pin}}
    if defect == "qualification_failed":
        record["status"] = "failed"
    elif defect == "source_changed":
        (root / M.SELF).write_text("# changed after qualification\n")
    elif defect == "extra_file":
        (directory / "unlisted.log").write_text("unlisted\n")
    path = directory / "receipt.json"
    json_file(path, record)
    if defect is None:
        assert M.engineering(path) == {name: value["sha256"] for name, value in sources.items()}
    else:
        with pytest.raises(ValueError):
            M.engineering(path)


@pytest.mark.parametrize("passed,defect", ((False, None), (True, None), (True, "parent_failed"),
                                         (False, "bad_condition_count"), (True, "pending"),
                                         (True, "reversed_producer_chronology")))
def test_close_requires_original_audit_closure_and_never_admits_confirmation(tmp_path, monkeypatch, passed, defect):
    root = tmp_path.resolve()
    monkeypatch.setattr(M, "ROOT", root)
    runtime = {"fabricated": True}
    producer_inputs = {"producer_terminal": json_file(root / "prior-producer-terminal.json",
        {"finished_ns": 1001 if defect == "reversed_producer_chronology" else 500})}
    bound = {"sources": {}, "lineage": {"fabricated": True}}
    plan = {"version": M.VERSION, "phase": "audit", "status": "frozen_before_execution",
            "configuration": M.CONFIG, "limits": M.LIMITS["audit"], "payloads": sorted(M.PAYLOADS["audit"]),
            "runtime": runtime, "sources": bound["sources"], "lineage": bound["lineage"], "inputs": producer_inputs}
    plan_pin = json_file(root / "audit-plan.json", plan)
    directory = root / "audited"
    directory.mkdir()
    conditions = [{"name": f"fabricated-{index}", "passed": passed or index == 0} for index in range(13)]
    decision = {"stage": "dev", "candidate": "rls_full", "selected_tau": .1, "technical_complete": True,
                "reports": 72, "total_conditions": 13, "conditions": conditions,
                "passed_conditions": 13 if passed else 1, "passed": passed}
    if defect == "bad_condition_count":
        decision["passed_conditions"] += 1
    audit = {"version": M.VERSION, "decision": decision, "reports": [{} for _ in range(72)],
             "verified_replays": [{} for _ in range(72)], "requires_successful_original_supervisor": True}
    json_file(directory / "audit.json", audit)
    json_file(directory / "summary.json", {"version": M.VERSION, "phase": "audit", "decision": decision, "counts": exact_counts("audit")})
    receipt = {"version": M.VERSION, "phase": "audit", "status": "completed", "complete": True,
               "plan_sha256": plan_pin["sha256"], "inputs": producer_inputs, "sources": {}, "limits": M.LIMITS["audit"],
               "views_completed": 72, "caches_completed": 3, "counts": exact_counts("audit"),
               "pending": None, "pending_io": None, "pending_emission": None, "pending_log": None,
               "peak_rss_bytes": 1, "requires_successful_original_supervisor": True}
    if defect == "pending":
        receipt["pending_io"] = {"unfinished": True}
    worker_pin = json_file(directory / "receipt.json", receipt)
    terminal_pin = json_file(root / "audit-terminal.json", {"fabricated": True})
    seen = []
    def parent(*args):
        seen.append(args)
        if defect == "parent_failed":
            raise ValueError("fabricated original audit failed")
        return {"started_ns": 1000, "finished_ns": 100001000}
    helper = SimpleNamespace(closed_files=lambda *_args: directory, successful_process=parent)
    monkeypatch.setattr(M, "load", lambda *_args: helper)
    monkeypatch.setattr(M, "runtime_record", lambda: runtime)
    def authenticated(phase, inputs):
        assert phase == "audit" and inputs == producer_inputs
        return bound
    monkeypatch.setattr(M, "authenticate", authenticated)
    arguments = {"output": root / "closed.json"}
    for role, record in (("audit_plan", plan_pin), ("audit_receipt", worker_pin), ("audit_terminal", terminal_pin)):
        arguments[role], arguments[role + "_sha256"] = Path(record["path"]), record["sha256"]
    args = SimpleNamespace(**arguments)
    if defect:
        with pytest.raises(ValueError):
            M.close(args)
        assert not args.output.exists()
    else:
        M.close(args)
        result = json.loads(args.output.read_text())
        assert result["status"] == ("DEV_PASS" if passed else "DEV_FAIL")
        assert result["confirmation_eligible_for_separate_registration"] is passed
        assert result["confirmation_execution_admitted"] is result["old_test_access"] is False
        assert result["numerical_array_decodes"] == result["model_calls"] == 0
        assert len(seen) == 1 and seen[0][1] == "audit" and seen[0][-1] == 900


@pytest.mark.parametrize("event_name", ("cache_complete", "view_complete"))
def test_completion_counter_advances_only_after_durable_complete_record(tmp_path, monkeypatch, event_name):
    run, _flat, _ids = setup_evaluation(tmp_path, monkeypatch)
    history, identities = run.inputs()
    original = run.event
    error = OSError("fabricated complete-record publication failure")
    attempted = []
    def event(row):
        if row["event"] == event_name:
            attempted.append(copy.deepcopy(row))
            raise error
        original(row)
    monkeypatch.setattr(run, "event", event)
    with pytest.raises(OSError) as caught:
        run.evaluate(history, identities)
    assert caught.value is error and len(attempted) == 1
    assert run.receipt["views_completed"] == 0
    assert run.receipt["caches_completed"] == (0 if event_name == "cache_complete" else 1)
    assert run.receipt["pending"]["operation"] == ("cache" if event_name == "cache_complete" else "view")
    assert "record" in attempted[0]
    if event_name == "view_complete":
        assert "report" in attempted[0]["record"] and "file" in attempted[0]["record"]


def test_fabricated_dev_end_to_end_produces72_views_and_independent_audit_matches(tmp_path, monkeypatch):
    from openjev.research import otto_residual_audit as independent
    producer, flat, _ids = setup_evaluation(tmp_path, monkeypatch)
    history, identities = producer.inputs()
    assert np.isnan(history["query_scores"][~history["query_mask"]]).all()
    raw_before = {key: value.tobytes() for key, value in flat.items()}
    producer.evaluate(history, identities)
    assert producer.receipt["views_completed"] == 72 and producer.receipt["caches_completed"] == 3
    assert producer.receipt["counts"]["array_decodes"] == 85
    assert producer.receipt["counts"]["checkpoint_decodes"] == 9
    assert producer.receipt["counts"]["model_constructions"] == 6
    assert producer.receipt["counts"]["model_construction_attempts"] == 6
    assert producer.receipt["counts"]["cache_builds"] == 3 and producer.receipt["counts"]["replays"] == 72
    assert len(list(producer.out.glob("prediction-*.npz"))) == 72
    assert len(list(producer.out.glob("cache-*.npz"))) == 3
    assert all(producer.receipt[name] is None for name in ("pending", "pending_io", "pending_emission", "pending_log"))
    assert {key: value.tobytes() for key, value in flat.items()} == raw_before
    saved = json.loads((producer.out / "views.json").read_text())
    assert len(saved["views"]) == 72 and len(saved["caches"]) == 3
    assert {row["seed"] for row in saved["views"]} == set(SEEDS)
    assert all(len([row for row in saved["views"] if row["seed"] == seed]) == 24 for seed in SEEDS)
    producer_files = {path.name: byte_pin(path) for path in producer.out.iterdir()}
    auditor = fabricated_run(tmp_path, "audit", producer.bound["collection"], producer.bound["lineage"],
                             producer=(producer.plan, {"files": producer_files}, producer.out))
    audit_history, audit_ids = auditor.inputs()
    auditor.audit(audit_history, audit_ids)
    assert auditor.receipt["counts"]["array_decodes"] == 76
    assert auditor.receipt["counts"]["checkpoint_decodes"] == auditor.receipt["counts"]["model_constructions"] == 0
    assert auditor.receipt["counts"]["cache_builds"] == 0 and auditor.receipt["counts"]["replays"] == 72
    assert auditor.receipt["views_completed"] == 72
    assert auditor.receipt["caches_completed"] == 3
    assert all(auditor.receipt[name] is None for name in ("pending", "pending_io", "pending_emission", "pending_log"))
    audit = json.loads((auditor.out / "audit.json").read_text())
    provisional = json.loads((producer.out / "summary.json").read_text())["decision"]
    recomputed = independent.evaluate_reports(audit["reports"], identities, stage="dev", technical_complete=False)
    M.equal_nested(provisional, recomputed)
    assert provisional["technical_complete"] is False
    assert audit["decision"]["technical_complete"] is True
    assert audit["decision"]["conditions"][1:] == provisional["conditions"][1:]
    assert audit["decision"]["selected_tau"] == provisional["selected_tau"]
    assert len(audit["verified_replays"]) == 72
    for worker in (producer, auditor):
        assert all(worker.receipt["counts"][name] == 0 for name in ("native_calls", "teacher_calls", "optimizer_steps", "confirm_decodes"))
    events = [json.loads(line) for line in (producer.out / "progress.jsonl").read_text().splitlines()]
    assert sum(row["event"] == "view_complete" for row in events) == 72
    assert sum(row["event"] == "cache_complete" for row in events) == 3
    assert sum(row["event"] == "model_construction_attempt" for row in events) == 6
    assert sum(row["event"] == "model_construction_return" for row in events) == 6
    assert [row["record"] for row in events if row["event"] == "view_complete"] == saved["views"]
    M.validate_counts("evaluate", producer.receipt["counts"])
    M.validate_counts("audit", auditor.receipt["counts"])
    numerical = json.loads((producer.out / "numerical.json").read_text())
    assert numerical["threads"] == numerical["interop_threads"] == 1
    assert numerical["deterministic_algorithms"] is True and numerical["device"] == "cpu"
    assert json.loads((auditor.out / "numerical.json").read_text())["neural_runtime_used"] is False
    # A hash-rebound fabricated report still must match independently computed
    # metrics. This deliberately bypasses only the outer byte admission fixture.
    saved["views"][0]["report"]["scopes"]["full"]["overall"]["case_weighted_raw_gap"] += .5
    json_file(producer.out / "views.json", saved)
    producer_files["views.json"] = byte_pin(producer.out / "views.json")
    (tmp_path / "tampered").mkdir()
    rejected = fabricated_run(tmp_path / "tampered", "audit", producer.bound["collection"], producer.bound["lineage"],
                              producer=(producer.plan, {"files": producer_files}, producer.out))
    rejected_history, rejected_ids = rejected.inputs()
    with pytest.raises(ValueError, match="independent saved action metrics"):
        rejected.audit(rejected_history, rejected_ids)
    assert rejected.receipt["views_completed"] == 0
    assert rejected.receipt["pending"]["operation"] == "audit_view"
