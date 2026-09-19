"""Independent proposal reconstruction and tiny engineering410 native evidence."""
from __future__ import annotations

import ast
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("gymnasium.envs.mujoco.reacher_v5")

from openjev.research import reacher_proposal_memory_audit as audit
from openjev.research.reacher_adaptive_search import SearchInputs
from openjev.research.reacher_cem_proposal_memory import CEMProposalMemory
from openjev.research.reacher_search_protocol import save_inputs
from openjev.research.reacher_tracking_dynamics import nominal_model


def inputs(chunks=2):
    rng = np.random.default_rng(410)
    return SearchInputs(rng.normal(size=(1, 64, chunks, 2)), rng.normal(size=(1, 192, chunks, 2)),
                        tuple(rng.normal(size=(1, k, chunks, 2)) for k in (64, 64, 63)))


def mapping(value):
    return dict(zip(("initial", "random_extra", "cem/1", "cem/2", "cem/3"),
                    (value.initial, value.random_extra, *value.cem), strict=True))


def memory_fixture(mode="shift_plan"):
    source = inputs(); original = mapping(source)
    memory = CEMProposalMemory(mode, 7, 4, 3)
    initial = memory.snapshot()
    counts = audit.new_counts(); counts["snapshot_calls"] = 1
    prior_costs = initial["costs"]; previous = {"previous_target": None, "previous_sequence": None, "previous_command": None}
    records = []
    for step in range(7):
        target = np.array([.1 if step < 4 else -.1, .05], np.float32)
        transformed, trace = memory.prepare(source, target, step=step)
        kwargs = dict(mode=mode, step=step, steps=7, horizon=4, block=3, target=target,
                      prior_counts=counts.copy(), prior_costs=copy.deepcopy(prior_costs), **previous)
        expected, result = audit.audit_prepare(trace, original, **kwargs)
        for key, value in mapping(transformed).items():
            np.testing.assert_array_equal(expected[key], value)
        records.append((copy.deepcopy(trace), kwargs))
        counts = {key: counts[key]+result["counts"][key] for key in counts}
        prior_costs = trace["costs"]
        sequence = np.array([[.1, -.2], [.3, .05], [-.1, .4], [.2, -.3]], np.float32)[:min(4, 7-step)].copy()
        memory.commit(sequence, sequence[0].copy())
        previous = {"previous_target": target, "previous_sequence": sequence, "previous_command": sequence[0].copy()}
    final = memory.snapshot()
    kwargs = {"mode": mode, "steps": 7, "horizon": 4, "block": 3, "counts": counts,
                  "last_sequence": sequence, "last_command": sequence[0], "last_target": target,
                  "last_trace": trace, "prepare_seconds": trace["costs"]["prepare_wall_seconds"],
                  "enclosing_seconds": final["costs"]["prepare_wall_seconds"]+final["costs"]["commit_wall_seconds"]+1.}
    return original, initial, final, records, kwargs


@pytest.mark.parametrize("mode", audit.MODES)
def test_real_memory_traces_and_unique_payload_accounting(mode):
    _, initial, final, records, kwargs = memory_fixture(mode)
    audit.audit_memory_snapshots(initial, final, **kwargs)
    assert records[0][0]["costs"]["retained_numpy_payload_bytes"] == 40
    assert records[4][0]["reset_reason"] == "observed_target_change"
    assert records[4][0]["shifted_sequence"] is None
    assert final["costs"]["retained_numpy_payload_bytes"] == (64 if mode == "shift_plan" else 56)
    assert records[-1][0]["center"][1] == [0., 0.]


def test_shift_one_action_explicit_block_means_hold_tail_and_preserve_other_streams():
    original = mapping(inputs())
    previous = np.array([[.1, -.2], [.3, .05], [-.1, .4], [.2, -.3]], np.float32)
    target = np.array([.1, .05], np.float32)
    transformed, info = audit.derive_inputs(original, mode="shift_plan", step=1, steps=7, horizon=4, block=3,
        target=target, previous_target=target, previous_sequence=previous, previous_command=previous[0])
    np.testing.assert_array_equal(info["shifted_sequence"], previous[[1, 2, 3, 3]])
    np.testing.assert_array_equal(info["center"][0], previous[1:].mean(0, dtype=np.float64))
    np.testing.assert_array_equal(info["center"][1], previous[-1])
    np.testing.assert_array_equal(transformed["initial"][:, :7], original["initial"][:, :7])
    for key in ("random_extra", "cem/1", "cem/2", "cem/3"):
        assert transformed[key] is original[key]
    for slot, scale in ((7, .25), (31, .25), (32, .75), (63, .75)):
        np.testing.assert_array_equal(transformed["initial"][0, slot], original["initial"][0, slot]+info["center"]/scale)


def test_cold_and_observed_target_change_return_exact_inputs_without_hidden_gain_reset():
    original = mapping(inputs()); target = np.array([.1, .05], np.float32)
    sequence = np.full((4, 2), .1, np.float32)
    for mode, current in (("cold", target), ("repeat_last", -target), ("shift_plan", -target)):
        transformed, info = audit.derive_inputs(original, mode=mode, step=1, steps=7, horizon=4, block=3,
            target=current, previous_target=target, previous_sequence=sequence, previous_command=sequence[0])
        assert all(transformed[key] is value for key, value in original.items())
        assert info["input_object_reused"]
    with pytest.raises(ValueError, match="acknowledged"):
        audit.derive_inputs(original, mode="shift_plan", step=1, steps=7, horizon=4, block=3,
            target=target, previous_target=target, previous_sequence=sequence, previous_command=-sequence[0])


@pytest.fixture(scope="module")
def memory_evidence():
    return memory_fixture()


@pytest.mark.parametrize("kind", ["center", "shift", "target", "reset", "base_id", "derived_id", "reuse",
    "copy_bytes", "hash_bytes", "retained_bytes", "projected", "column_means", "tail", "commit",
    "extra_snapshot", "setup", "negative_time", "nan_time", "unknown_field"])
def test_resealed_prepare_trace_corruption_rejected(memory_evidence, kind):
    original, _, _, records, _ = memory_evidence
    trace, kwargs = copy.deepcopy(records[1])
    if kind == "center": trace["center"][0][0] += .01
    elif kind == "shift": trace["shifted_sequence"][0][0] += .01
    elif kind == "target": trace["public_target"][0] += .01
    elif kind == "reset": trace["reset_reason"] = "observed_target_change"
    elif kind == "base_id": trace["base_input_identities"]["initial"] = "0"*64
    elif kind == "derived_id": trace["transformed_input_identities"]["initial"] = "0"*64
    elif kind == "reuse": trace["input_object_reused"] = True
    elif kind == "setup": trace["costs"]["setup_wall_seconds"] += .01
    elif kind == "negative_time": trace["costs"]["prepare_wall_seconds"] = -1.
    elif kind == "nan_time": trace["costs"]["commit_wall_seconds"] = float("nan")
    elif kind == "unknown_field": trace["future_gain"] = 1.3
    else:
        key = {"copy_bytes": "input_payload_bytes_copied", "hash_bytes": "input_identity_bytes_hashed",
               "retained_bytes": "retained_numpy_payload_bytes", "projected": "projected_blocks",
               "column_means": "column_mean_calls", "tail": "tail_actions_repeated", "commit": "commit_completed",
               "extra_snapshot": "snapshot_calls"}[kind]
        trace["costs"][key] += 1
    with pytest.raises(ValueError):
        audit.audit_prepare(trace, original, **kwargs)


@pytest.mark.parametrize("kind", ["sequence", "command", "target", "pending", "cursor", "failed", "count",
    "bytes", "trace", "config", "startup_work", "startup_cache", "snapshot", "unpaid_time"])
def test_final_snapshot_and_startup_corruption_rejected(memory_evidence, kind):
    _, initial, final, _, kwargs = copy.deepcopy(memory_evidence)
    if kind == "sequence": final["last_selected_sequence"][0][0] += .01
    elif kind == "command": final["last_issued_command"][0] += .01
    elif kind == "target": final["last_target"][0] += .01
    elif kind == "pending": final["pending"] = {"step": 7}
    elif kind == "cursor": final["next_step"] -= 1
    elif kind == "failed": final["failed"] = True
    elif kind == "count": final["costs"]["commit_completed"] -= 1
    elif kind == "bytes": final["costs"]["retained_numpy_payload_bytes"] += 8
    elif kind == "trace": final["last_trace"]["step"] -= 1
    elif kind == "config": final["configuration"]["rng_draws"] = 1
    elif kind == "startup_work": initial["costs"]["commit_completed"] = 1
    elif kind == "startup_cache": initial["last_target"] = [0., 0.]
    elif kind == "snapshot": final["costs"]["snapshot_calls"] += 1
    elif kind == "unpaid_time": kwargs["enclosing_seconds"] = 0.
    with pytest.raises(ValueError):
        audit.audit_memory_snapshots(initial, final, **kwargs)


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    from openjev.research.reacher_proposal_memory_rollout import run_row
    from openjev.research.reacher_tracking_rollout import run_row as original_run
    folder = tmp_path_factory.mktemp("proposal-audit-engineering410")
    stems = []
    for step in range(3):
        stem = folder/"inputs"/f"{step:03d}"
        save_inputs(stem, inputs(), f"engineering410/{step}"); stems.append(stem)
    common = {"target_path": np.array([[.12, -.04], [.12, -.04], [-.06, .10], [-.06, .10]]),
        "gain_schedule": np.array([.5, 1.5, 1.5]), "noise": np.full((3, 2), .0001), "reset_seed": 410,
        "inputs_by_step": stems, "gain_grid": np.array([.5, 1., 1.5]), "window": 2, "freeze_after": 1,
        "noise_std": .05, "planning_horizon": 3, "action_block": 2, "deadline": time.monotonic()+120, "engineering": True}
    for arm in audit.ARMS:
        original_run(arm=arm, out=folder/f"old-{arm}", **common)
        for mode in audit.MODES:
            run_row(arm=arm, proposal_mode=mode, out=folder/f"{arm}-{mode}", **common)
    return folder, stems


@pytest.mark.parametrize("arm", audit.ARMS)
@pytest.mark.parametrize("mode", audit.MODES)
def test_actual_completed_writer_native_and_memory_audit(rows, arm, mode):
    folder, stems = rows
    report = audit.audit_row(folder/f"{arm}-{mode}", nominal_model=nominal_model(), inputs_by_step=stems)
    assert report["status"] == "completed"
    assert report["native_control"]["transitions"] == 3
    assert report["candidate_transitions_checked"] == 1536
    assert report["selected_transitions_checked"] == 3
    assert report["max_candidate_state_abs_error"] == 0
    assert report["new_model_calls"] == report["new_planner_calls"] == report["new_rng_draws"] == 0
    if mode == "cold":
        old = json.loads((folder/f"old-{arm}"/"episode.json").read_text())
        new = json.loads((folder/f"{arm}-{mode}"/"episode.json").read_text())
        assert old == new
        for step in range(3):
            with np.load(folder/f"old-{arm}"/f"decisions/{step:03d}.npz") as a, \
                    np.load(folder/f"{arm}-{mode}"/f"decisions/{step:03d}.npz") as b:
                assert set(a.files) == set(b.files)
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize("kind", ["center", "derived_ids", "root", "final_cache", "final_cost", "unpaid", "failure", "extra"])
def test_resealed_complete_row_rejects_corruption(rows, tmp_path, kind):
    folder, stems = rows; out = tmp_path/"row"
    shutil.copytree(folder/"public_gain-shift_plan", out)
    if kind in ("center", "derived_ids"):
        path = out/"decisions/001.json"; record = json.loads(path.read_text())
        if kind == "center": record["proposal_memory"]["center"][0][0] += .01
        else: record["input_identities"]["initial"] = "0"*64
        path.write_text(json.dumps(record))
    elif kind == "root":
        path = out/"decisions/001.npz"
        with np.load(path) as f: data = {k: f[k] for k in f.files}
        data["root_qvel"][0] += .01
        np.savez_compressed(path, **data)
    elif kind in ("final_cache", "final_cost"):
        path = out/"proposal-memory.json"; record = json.loads(path.read_text())
        if kind == "final_cache": record["last_issued_command"][0] += .01
        else: record["costs"]["retained_numpy_payload_bytes"] += 1
        path.write_text(json.dumps(record))
    elif kind == "failure": (out/"failed.json").write_text("{}")
    elif kind == "extra": (out/"extra.json").write_text("{}")
    receipt = json.loads((out/"completed.json").read_text())
    for rel in receipt["files"]:
        receipt["files"][rel] = audit.file_hash(out/rel)
    if kind == "unpaid": receipt["proposal_memory_seconds"] = 0.
    (out/"completed.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        audit.audit_row(out, nominal_model=nominal_model(), inputs_by_step=stems)


def test_expired_audit_and_unimplemented_roles_rejected_before_replay(rows, tmp_path):
    folder, stems = rows
    with pytest.raises(TimeoutError):
        audit.audit_row(folder/"nominal-cold", nominal_model=nominal_model(), inputs_by_step=stems,
                        deadline=time.perf_counter()-1)
    shutil.copytree(folder/"nominal-cold", tmp_path/"row")
    path = tmp_path/"row/started.json"; record = json.loads(path.read_text()); record["arm"] = "adaptive"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="role"):
        audit.audit_row(tmp_path/"row", nominal_model=nominal_model(), inputs_by_step=stems)


def test_auditor_has_no_mutable_proposal_or_controller_imports():
    tree = ast.parse(Path(audit.__file__).read_text())
    imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert imported == ["__future__", "pathlib", "openjev.research.reacher_tracking_audit"]
    assert not any(isinstance(node, ast.Global) for node in ast.walk(tree))
