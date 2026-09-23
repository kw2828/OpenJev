"""Fabricated conditional TEST admission and frozen P4/P8 evaluation checks.

No fixture accesses empirical arrays, native environments or trained checkpoints.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openjev.research import otto_query_memory_data as data
from openjev.research import otto_query_memory_metrics as metrics
from openjev.research import otto_query_memory_model as models

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


runner = module("_test_conditional_test_evaluator", "scripts/evaluate_otto_query_memory.py")
producer = module("_test_conditional_training_helpers", "scripts/train_otto_query_memory.py")
auditor = module("_test_conditional_dev_audit_helpers", "scripts/audit_otto_query_memory.py")


def put(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return {"path": str(path), **runner.descriptor(path)}


def fake_dev_reports():
    reports = []
    for seed in runner.SEEDS:
        for view in runner.VIEWS:
            gap = 8. if view == "trace_delta" else 10.
            scopes = {scope: {"by_regime": {regime: {"episodes": 9, "declared_case_count": 3,
                "supported_case_count": 3, "case_weighted_raw_gap": gap} for regime in ("lambda3", "lambda4")}}
                for scope in ("full", "later")}
            reports.append({"stage": "dev", "query_period": 4, "episodes": 18, "family": view, "seed": seed, "scopes": scopes})
    return reports


@pytest.fixture
def dev_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(producer, "ROOT", tmp_path)
    inputs = {"training_" + name: put(tmp_path / (name + ".json"), {"fabricated": name})
              for name in ("plan", "receipt", "terminal")}
    expected_inputs = {"plan": inputs["training_plan"], "worker": inputs["training_receipt"], "terminal": inputs["training_terminal"]}
    sources = {"fabricated-source.py": "0" * 64}
    directory = tmp_path / "audit"
    directory.mkdir()
    launch_path = tmp_path / "audit.launch.json"
    command = [str(tmp_path / ".venv/bin/python"), str(tmp_path / runner.DEV_AUDITOR)]
    for name, pin in expected_inputs.items():
        command += ["--" + name, pin["path"], "--" + name + "-sha256", pin["sha256"]]
    command += ["--supervision", str(launch_path), "--output", str(directory)]
    launch = {"command": command, "pid": 102, "pgid": 102, "parent_pid": 101, "cwd": str(tmp_path),
        "cap_seconds": 600, "started_ns": 1, "deadline_ns": 600 * 10**9 + 1,
        "watchdog_sha256": runner.SUPERVISOR_PIN, "clock_source_sha256": runner.CLOCK_PIN}
    launch_pin = put(launch_path, launch)
    started = {"version": auditor.VERSION, "launch": launch, "started_ns": 2, "inputs": expected_inputs,
               "authenticated_before_numerical_reads": True}
    put(directory / "started.json", started)
    reports = fake_dev_reports()
    producer_gate = auditor.independent_gate(reports, technical_complete=False)
    result = {"version": auditor.VERSION, "agreement": True, "stage": "dev", "inputs": expected_inputs,
        "test_evaluation_admitted": False, "conditional_test_admission_requires_successful_original_audit_supervisor": True,
        "zero_call_counts": {k: 0 for k in ("model_calls", "optimizer_calls", "teacher_calls", "simulator_calls", "test_array_decodes")},
        "metrics": reports, "gate": auditor.independent_gate(reports, technical_complete=True), "producer_gate": producer_gate}
    put(directory / "audit.json", result)
    receipt = {"version": auditor.VERSION, "status": "completed", "complete": True, "agreement": True, "pending": None,
        "sources": sources, "inputs": expected_inputs, "source_plan_sha256": inputs["training_plan"]["sha256"],
        "limits": auditor.LIMITS, "peak_rss_bytes": 1024, "requires_successful_original_supervisor": True,
        "supervision_sha256": launch_pin["sha256"], "started_ns": 2, "finished_ns": 100,
        "files": {name: runner.descriptor(directory / name) for name in ("started.json", "audit.json")},
        **dict.fromkeys(("model_calls", "optimizer_calls", "teacher_calls", "simulator_calls", "test_array_decodes"), 0)}
    inputs["dev_audit_receipt"] = put(directory / "receipt.json", receipt)
    terminal = {**launch, "status": "completed", "returncode": 0, "timed_out": False, "group_absent": True,
        "cleanup": {"reaped": True, "group_absent": True, "errors": []}, "timing_available": True,
        "error": None, "clock_error": None, "finished_ns": 101}
    inputs["dev_audit_terminal"] = put(tmp_path / "audit.terminal.json", terminal)
    training = {"summary": {"gate": producer_gate}}
    return SimpleNamespace(inputs=inputs, sources=sources, training=training, receipt=receipt, terminal=terminal,
        launch=launch, result=result, directory=directory, launch_path=launch_path)


def call_dev(fixture):
    return runner.authenticate_dev_audit(fixture.inputs, fixture.training, fixture.sources, producer, auditor)


def resave_audit(fixture):
    put(fixture.directory / "audit.json", fixture.result)
    fixture.receipt["files"]["audit.json"] = runner.descriptor(fixture.directory / "audit.json")
    fixture.inputs["dev_audit_receipt"] = put(fixture.directory / "receipt.json", fixture.receipt)


def test_fixed_test_roster_payloads_and_no_training():
    assert len(runner.PAYLOADS) == 58
    assert len([n for n in runner.PAYLOADS if "prediction" in n]) == 48
    assert runner.VIEWS == producer.VIEWS and runner.SEEDS == producer.SEEDS and runner.PERIODS == (4, 8)
    assert runner.LIMITS == {"seconds": 1800, "rss_bytes": 4 * 1024**3, "output_bytes": 2 * 1024**3}
    assert runner.CONFIG["optimizer_calls"] == runner.CONFIG["train_array_decodes"] == runner.CONFIG["dev_array_decodes"] == 0
    assert runner.NEW_COMPONENTS == {runner.SELF, runner.TEST, "scripts/audit_otto_query_memory_test.py", "tests/test_audit_otto_query_memory_test.py"}


def test_closed_independent_dev_admits_metadata_only(dev_evidence):
    admitted = call_dev(dev_evidence)
    assert admitted["result"]["gate"]["passed"] is True
    assert admitted["receipt"]["test_array_decodes"] == 0
    assert admitted["run"] == dev_evidence.directory


@pytest.mark.parametrize("field,value", [("status", "failed"), ("returncode", True), ("returncode", 1),
    ("timed_out", True), ("group_absent", False), ("cap_seconds", 601), ("timing_available", False),
    ("error", "failed"), ("clock_error", "failed"), ("deadline_ns", 10), ("finished_ns", 99)])
def test_original_dev_audit_parent_failure_blocks_test(dev_evidence, field, value):
    dev_evidence.terminal[field] = value
    dev_evidence.inputs["dev_audit_terminal"] = put(Path(dev_evidence.inputs["dev_audit_terminal"]["path"]), dev_evidence.terminal)
    with pytest.raises(ValueError, match="original successful DEV audit supervisor"):
        call_dev(dev_evidence)


@pytest.mark.parametrize("field,value", [("complete", False), ("agreement", False), ("pending", {"decode": True}),
    ("test_array_decodes", 1), ("model_calls", 1), ("optimizer_calls", 1), ("teacher_calls", 1),
    ("simulator_calls", 1), ("requires_successful_original_supervisor", False)])
def test_incomplete_or_nonindependent_dev_worker_blocks_test(dev_evidence, field, value):
    dev_evidence.receipt[field] = value
    resave_audit(dev_evidence)
    with pytest.raises(ValueError, match="complete independent DEV audit"):
        call_dev(dev_evidence)


def test_failed_dev_condition_cannot_be_hidden_by_claimed_pass(dev_evidence):
    next(r for r in dev_evidence.result["metrics"] if r["family"] == "trace_delta")["scopes"]["later"]["by_regime"]["lambda3"]["case_weighted_raw_gap"] = 11.
    resave_audit(dev_evidence)
    with pytest.raises(ValueError, match="all13 independently audited"):
        call_dev(dev_evidence)


def test_honest_failed_dev_gate_still_blocks_test(dev_evidence):
    for report in dev_evidence.result["metrics"]:
        if report["family"] == "trace_delta":
            report["scopes"]["later"]["by_regime"]["lambda4"]["case_weighted_raw_gap"] = 9.5
    dev_evidence.result["gate"] = auditor.independent_gate(dev_evidence.result["metrics"], technical_complete=True)
    resave_audit(dev_evidence)
    with pytest.raises(ValueError, match="all13 independently audited"):
        call_dev(dev_evidence)


def test_successful_audit_for_other_training_run_cannot_admit(dev_evidence):
    dev_evidence.receipt["inputs"]["worker"] = {"path": "elsewhere", "sha256": "0" * 64, "bytes": 1}
    resave_audit(dev_evidence)
    with pytest.raises(ValueError, match="complete independent DEV audit"):
        call_dev(dev_evidence)


def test_original_audit_command_and_payload_inventory_checked(dev_evidence):
    dev_evidence.terminal["command"][2] = "--unknown"
    dev_evidence.inputs["dev_audit_terminal"] = put(Path(dev_evidence.inputs["dev_audit_terminal"]["path"]), dev_evidence.terminal)
    with pytest.raises(ValueError, match="exact DEV audit options"):
        call_dev(dev_evidence)


def test_original_audit_launch_pin_is_required(dev_evidence):
    changed = copy.deepcopy(dev_evidence.launch)
    changed["pid"] = 103
    put(dev_evidence.launch_path, changed)
    with pytest.raises(ValueError, match="original DEV audit launch identity"):
        call_dev(dev_evidence)


@pytest.mark.parametrize("component", sorted(runner.NEW_COMPONENTS))
def test_missing_new_component_qualification_blocks_before_source_reads(component, monkeypatch):
    source = {name: {"sha256": "0" * 64, "bytes": 1} for name in runner.NEW_COMPONENTS - {component}}
    receipt = {"status": "passed", "source_before": source, "source_after": source,
               "commands": [{"returncode": 0, "timed_out": False, "reaped": True}]}
    monkeypatch.setattr(runner, "read", lambda path: receipt)
    monkeypatch.setattr(runner, "descriptor", lambda path: pytest.fail("must reject unqualified source first"))
    with pytest.raises(ValueError, match="qualified new TEST producer and auditor"):
        runner.authenticate_engineering("fabricated", {})


def fabricated_flat(lengths=(65, 9, 1, 33, 5, 1)):
    offsets = np.array([0, *np.cumsum(lengths)], np.int64)
    total = int(offsets[-1])
    features = np.linspace(-.2, .6, total * 31, dtype=np.float32).reshape(total, 31)
    target, correction, identities = np.zeros((total, 4), np.float32), np.zeros(total, np.bool_), []
    for index, (low, high) in enumerate(pairwise(offsets)):
        steps = np.arange(high - low)
        features[low:high, 15] = steps / 2188
        features[low:high, 16] = (steps % 4) / 2188
        features[low:high, 17] = 1
        target[low:high] = np.array([31, 34, 33, 32], np.float32) + index / 8 + steps[:, None] * np.array([.25, -.125, .5, -.25], np.float32)
        correction[low:high] = steps % 4 == 0
        regime = "lambda3" if index < 3 else "lambda4"
        identities.append({"stage": "test", "episode_id": f"fabricated:test:{index}", "episode_index": index,
            "seed": 307000001 if regime == "lambda3" else 308000001, "case": 0, "regime": regime,
            "arm": ("analytic", "neural", "period4_hold")[index % 3]})
    return {"features": features, "raw_q": target, "legal": np.ones((total, 4), np.bool_),
            "actions": np.zeros(total, np.int64), "correction": correction, "episode_offsets": offsets}, identities


def synthetic_checkpoints(seed=309000001):
    parent = producer.construct_fit(models, "pretrained", seed)
    with torch.no_grad():
        parent.slow.output.weight.copy_(torch.arange(112).sin().reshape(4, 28) * .01)
        parent.slow.action_residual.weight.copy_(torch.arange(112).cos().reshape(4, 28) * .002)
        parent.slow.action_residual.bias.copy_(torch.tensor([.1, -.2, .3, -.1]))
    fitted = {"pretrained": parent}
    for family in producer.BRANCHES:
        fitted[family] = producer.construct_fit(models, family, seed, parent.slow.state_dict())
    return {family: {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}
            for family, model in fitted.items()}


def numerical_run(tmp_path):
    run = runner.Run(SimpleNamespace(output=tmp_path))
    run.np, run.torch, run.models, run.data, run.metrics, run.producer = np, torch, models, data, metrics, producer
    clock = iter(range(1000, 100000000, 1000))
    run.clock = SimpleNamespace(now_ns=lambda: next(clock))
    run.check = lambda: None
    run.events = []
    run.event = run.events.append
    return run


@pytest.mark.parametrize("period", (4, 8))
@pytest.mark.parametrize("view", runner.VIEWS)
def test_canonical_checkpoint_copy_flags_ownership_and_rng(period, view):
    arrays = synthetic_checkpoints()[producer.view_parent(view)]
    rng = torch.random.get_rng_state().clone()
    model = runner.load_view(np, torch, models, producer, arrays, view, runner.SEEDS[0], period)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert producer.state_witness(model) == auditor.tensor_witness(arrays)
    assert model.slow.query_period == period and not model.training and model.slow.mode == "frozen"
    for name, value in model.state_dict().items():
        assert value.detach().numpy().tobytes() == arrays[name].tobytes()
        assert not np.shares_memory(value.detach().numpy(), arrays[name])
    assert all(p.requires_grad == name.startswith("action_residual.") for name, p in model.slow.named_parameters())


@pytest.mark.parametrize("period", (4, 8))
def test_every_view_exact_queries_tails_no_write_and_unchanged_weights(period, tmp_path):
    flat, identities = fabricated_flat()
    history = data.project_census(flat, identities, query_period=period, expected_stage="test")
    checkpoints = synthetic_checkpoints()
    run = numerical_run(tmp_path)
    reference = None
    for view in runner.VIEWS:
        model = runner.load_view(np, torch, models, producer, checkpoints[producer.view_parent(view)], view, runner.SEEDS[0], period)
        before = producer.state_witness(model)
        saved, row = run.predict(model, view, runner.SEEDS[0], period, history, identities)
        assert producer.state_witness(model) == before
        assert row["query_period"] == period and row["rows"] == flat["episode_offsets"][-1]
        assert row["chunks"] == 3 and row["metrics"]["query_period"] == period
        assert {"common_full", "common_initial", "common_later"} <= set(row["metrics"]["scopes"])
        for name in ("action_prediction", "base_prediction", "slow_action_prediction"):
            assert saved[name][history["query_mask"]].tobytes() == flat["raw_q"][history["query_mask"]].tobytes()
        if view == "pretrained":
            reference = saved
        elif view != "joint_aux":
            assert producer.same_frozen_predictions(saved, reference, no_write=view == "trace_no_write")
        assert not row["evaluation"]["grad_enabled"]
        assert all(value == 0 for name, value in run.receipt.items() if name in runner.ZERO_COUNTS)
    assert run.receipt["model_calls"] == 24
    assert len(run.events) == 48 and run.receipt["pending"] is None


def test_p8_unobserved_p4_answers_cannot_change_model_predictions(tmp_path):
    flat, identities = fabricated_flat()
    changed = {name: value.copy() for name, value in flat.items()}
    for low, high in pairwise(flat["episode_offsets"]):
        steps = np.arange(high - low)
        changed["raw_q"][low:high][steps % 8 != 0] += np.array([3, -2, 5, -4], np.float32)
    arrays = synthetic_checkpoints()["trace_delta"]
    saved = []
    for source in (flat, changed):
        run = numerical_run(tmp_path)
        history = data.project_census(source, identities, query_period=8, expected_stage="test")
        model = runner.load_view(np, torch, models, producer, arrays, "trace_delta", runner.SEEDS[0], 8)
        value, _ = run.predict(model, "trace_delta", runner.SEEDS[0], 8, history, identities)
        saved.append(value)
    assert all(saved[0][name].tobytes() == saved[1][name].tobytes() for name in runner.PREDICTION_FIELDS)


def test_numerical_work_requires_closed_admission_before_io(tmp_path):
    run = runner.Run(SimpleNamespace(output=tmp_path))
    with pytest.raises(ValueError, match="closed independent DEV admission"):
        run.decode_checkpoints()
    with pytest.raises(ValueError, match="one TEST decode only"):
        run.decode_test()
    with pytest.raises(ValueError, match="full conditional admission"):
        run.body()


def test_repeated_test_decode_is_rejected_before_path_lookup(tmp_path):
    run = runner.Run(SimpleNamespace(output=tmp_path))
    run.authenticated = True
    run.receipt.update(checkpoint_decodes=18, test_array_decodes=1)
    with pytest.raises(ValueError, match="one TEST decode only"):
        run.decode_test()
