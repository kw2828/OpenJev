"""Fabricated-only checks: arithmetic evidence must never repair process closure."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_test_interrupted_action_diagnostic", ROOT / "scripts/diagnose_otto_action_focused_interruption.py")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def write_json(path, value):
    path.write_text(json.dumps(value))


def pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def gate_rows():
    rows = [{"name": "common.technical_complete", "value": False,
             "passes": False, "relation": "==", "threshold": True}]
    rows += [{"name": f"common.support_{i}", "passes": True} for i in range(10)]
    for architecture in ("innovation", "gru"):
        rows += [{"name": f"objective.{architecture}.condition_{i}", "passes": True} for i in range(12)]
    rows += [{"name": f"architecture.condition_{i}", "passes": True} for i in range(6)]
    return rows


def test_all_scientific_conditions_passing_never_qualifies_original_study():
    gates = diagnostic.ineligible_gates(gate_rows())
    assert {k: (v["passed"], v["total"]) for k, v in gates.items()} == {
        "innovation_objective": (22, 23), "gru_objective": (22, 23), "architecture": (28, 29)}
    assert all(v["eligible"] is False and v["passes"] is False for v in gates.values())


@pytest.mark.parametrize("field", ["value", "passes"])
def test_gate_rejects_even_partial_technical_promotion(field):
    rows = gate_rows()
    rows[0][field] = True
    with pytest.raises(ValueError, match="always false"):
        diagnostic.ineligible_gates(rows)


def test_original_arithmetic_methods_are_inherited_without_runtime_source_rewriting():
    for name in ("training", "training_predictions", "reconstruct", "sampled_training", "predictions",
                 "capacity", "validation_history", "saved_windows", "authenticate"):
        assert getattr(diagnostic.Diagnostic, name) is getattr(diagnostic.original.Audit, name)
    assert diagnostic.ORIGINAL_PIN == hashlib.sha256((ROOT / diagnostic.ORIGINAL).read_bytes()).hexdigest()


def test_no_model_imports_or_technical_true_body():
    tree = ast.parse((ROOT / diagnostic.SELF).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert not any(name.startswith(("torch", "tensorflow", "mlx", "openjev")) for name in names)
        if isinstance(node, ast.keyword) and node.arg == "technical_complete":
            assert isinstance(node.value, ast.Constant) and node.value.value is False
    assert not any(isinstance(node, ast.Attribute) and node.attr == "closed_parent"
                   for node in ast.walk(tree))


def phase_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    directory = tmp_path / "training"
    directory.mkdir()
    plan_path, worker_path = tmp_path / "plan.json", directory / "receipt.json"
    launch_path = tmp_path / "training.launch.json"
    observation_path = tmp_path / "observation.json"
    runtime = {"python": "fabricated", "executable": str(tmp_path / ".venv/bin/python"),
               "distributions": {"numpy": "fixture_np", "torch": "fixture_torch"}}
    limits = {"seconds": 14400, "rss_bytes": 4 * 1024**3, "output_bytes": 2 * 1024**3}
    plan = {"sources": {}, "inputs": {}, "runtime": runtime, "limits": limits}
    write_json(plan_path, plan)
    launch = {"command": [str(tmp_path / ".venv/bin/python"), str(tmp_path / diagnostic.original.PRODUCER),
        "run", "--plan", str(plan_path), "--plan-sha256", pin(plan_path)["sha256"],
        "--supervision", str(launch_path), "--output", str(directory)],
        "cwd": str(tmp_path), "cap_seconds": 14400, "clock_source_sha256": diagnostic.CLOCK_PIN,
        "watchdog_sha256": diagnostic.SUPERVISOR_PIN, "started_ns": 1,
        "deadline_ns": 1 + 14400 * 10**9}
    write_json(launch_path, launch)
    write_json(directory / "started.json", {"launch": launch})
    write_json(directory / "runtime.json", {**runtime, "numpy": "fixture_np", "torch": "fixture_torch"})
    for index in range(47):
        (directory / f"saved-{index}.bin").write_bytes(b"fabricated payload")
    worker = {"sources": {}, "inputs": {}, "limits": limits,
        "plan_sha256": pin(plan_path)["sha256"], "supervision_sha256": pin(launch_path)["sha256"],
        "version": diagnostic.PRODUCER_VERSION, "status": "completed", "complete": True,
        "requires_successful_original_supervisor": True, "fits_completed": 12, "optimizer_steps": 8640,
        "native_calls": 0, "teacher_calls": 0, "started_ns": 2, "finished_ns": 1000000002,
        "wall_seconds": 1.0, "peak_rss_bytes": 1234, "pending": None, "pending_emission": None,
        "files": {p.name: {k: v for k, v in pin(p).items() if k != "path"} for p in directory.iterdir()}}
    write_json(worker_path, worker)
    observation = {"status": "original_parent_closure_unavailable", "original_terminal_exists": False,
        "process_group_present": False, "processes_present": {"supervisor": False, "worker": False},
        "scientific_retry_launched": False, "worker_receipt_sha256": pin(worker_path)["sha256"],
        "worker_complete": True, "worker_status": "completed", "fits_completed": 12,
        "optimizer_steps": 8640, "worker_reported_seconds": 1.0,
        "files": {launch_path.name: {k: v for k, v in pin(launch_path).items() if k != "path"}}}
    write_json(observation_path, observation)
    args = SimpleNamespace(plan=plan_path, worker=worker_path, terminal=None, output=tmp_path / "diagnostic")
    obj = diagnostic.Diagnostic(args)
    obj.inputs = {"worker": pin(worker_path), "launch": pin(launch_path), "interruption": pin(observation_path)}
    obj.diagnostic_plan = {"original_terminal_path": str(tmp_path / "training.terminal.json")}
    # Regular-evidence containment is tested separately; this fixture remains in pytest's temp root.
    obj.path = lambda value: Path(value)
    obj.check = lambda: None
    return obj, plan, worker, launch, observation


def phase(obj):
    return obj.phase(obj.args.plan, obj.args.worker, None, diagnostic.original.PRODUCER, ".venv/bin/python", 14400)


def refresh_worker(obj, worker, observation):
    write_json(obj.args.worker, worker)
    obj.inputs["worker"] = pin(obj.args.worker)
    observation["worker_receipt_sha256"] = obj.inputs["worker"]["sha256"]
    observation["worker_reported_seconds"] = worker["wall_seconds"]
    write_json(Path(obj.inputs["interruption"]["path"]), observation)


def test_saved_training_authentication_has_no_parent_and_does_not_count_closed_phase(tmp_path, monkeypatch):
    obj, plan, worker, _, _ = phase_fixture(tmp_path, monkeypatch)
    result = phase(obj)
    assert result == (plan, worker, None, obj.args.worker.parent)
    assert obj.counts["unverified_training_phases"] == 1
    assert obj.counts["closed_phases"] == 0
    assert obj.receipt["technical_complete"] is False


def test_worker_reported_time_does_not_become_parent_bound_claim(tmp_path, monkeypatch):
    obj, _, worker, _, observation = phase_fixture(tmp_path, monkeypatch)
    worker["finished_ns"] = worker["started_ns"] + 16000 * 10**9
    worker["wall_seconds"] = 16000.0
    # Diagnostic accepts internally consistent saved timings even beyond the intended cap,
    # because they cannot establish actual original parent enforcement either way.
    refresh_worker(obj, worker, observation)
    phase(obj)
    assert obj.receipt["original_process_closure"] is False


def test_original_terminal_appearance_is_not_silently_admitted(tmp_path, monkeypatch):
    obj, *_ = phase_fixture(tmp_path, monkeypatch)
    Path(obj.diagnostic_plan["original_terminal_path"]).write_text('{}')
    with pytest.raises(ValueError, match="remains unavailable"):
        phase(obj)


@pytest.mark.parametrize("key,value", [("pending", {"batch": 2}), ("pending_emission", {"file": "x"}),
    ("complete", False), ("fits_completed", 11), ("optimizer_steps", 8639), ("native_calls", 1),
    ("requires_successful_original_supervisor", False), ("wall_seconds", 2.0)])
def test_worker_integrity_defects_fail_instead_of_substituting(tmp_path, monkeypatch, key, value):
    obj, _, worker, _, observation = phase_fixture(tmp_path, monkeypatch)
    worker[key] = value
    refresh_worker(obj, worker, observation)
    with pytest.raises(ValueError):
        phase(obj)


def test_mutated_payload_is_rejected_before_any_array_decode(tmp_path, monkeypatch):
    obj, *_ = phase_fixture(tmp_path, monkeypatch)
    (obj.args.worker.parent / "saved-0.bin").write_bytes(b"different")
    obj.arrays = lambda _: pytest.fail("No array should be decoded before authentication")
    with pytest.raises(ValueError, match="pinned bytes"):
        phase(obj)


def test_runtime_metadata_must_match_original_frozen_training_runtime(tmp_path, monkeypatch):
    obj, _, worker, _, observation = phase_fixture(tmp_path, monkeypatch)
    path = obj.args.worker.parent / "runtime.json"
    value = json.loads(path.read_text())
    value["torch"] = "changed"
    write_json(path, value)
    worker["files"][path.name] = pin(path)
    refresh_worker(obj, worker, observation)
    with pytest.raises(ValueError, match="framework versions"):
        phase(obj)


def test_original_collection_still_requires_unmodified_parent_authentication(tmp_path, monkeypatch):
    obj, *_ = phase_fixture(tmp_path, monkeypatch)
    calls = []

    def closed_phase(self, *args):
        calls.append(args)
        raise ValueError("genuine closed parent required")

    monkeypatch.setattr(diagnostic.original.Audit, "phase", closed_phase)
    with pytest.raises(ValueError, match="genuine closed parent"):
        obj.phase(Path("cplan"), Path("cworker"), Path("cterminal"),
                  diagnostic.original.COLLECTOR, ".venv-otto-released-native/bin/python", 7200)
    assert len(calls) == 1 and calls[0][2] == Path("cterminal")


def test_regular_evidence_rejects_traversal_and_symlinks(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    value = tmp_path / "file.json"
    value.write_text('{}')
    symlink = tmp_path / "link.json"
    symlink.symlink_to(value)
    with pytest.raises(ValueError, match="contained regular"):
        diagnostic.regular(symlink)
    (tmp_path / "directory").mkdir()
    with pytest.raises(ValueError, match="contained regular"):
        diagnostic.regular(tmp_path / "directory" / ".." / "file.json")


def test_freeze_reads_metadata_only_and_adds_new_sources_without_replacing_old(tmp_path, monkeypatch):
    fake_pin = "a" * 64
    arguments = {role: tmp_path / (role + '.json') for role in diagnostic.ROLES}
    arguments.update({role + '_sha256': fake_pin for role in diagnostic.ROLES})
    arguments['launch'] = tmp_path / 'original.launch.json'
    arguments['output'] = tmp_path / 'frozen.json'
    args = SimpleNamespace(**arguments)
    runtime = {"fabricated": "runtime metadata"}
    original_plan = {"version": diagnostic.PRODUCER_VERSION, "status": "frozen_before_fitting",
        "config": diagnostic.original.CONFIG, "runtime": runtime,
        "sources": {diagnostic.ORIGINAL: diagnostic.ORIGINAL_PIN}, "inputs": {}}
    worker = {"sources": original_plan['sources'], "inputs": {}, "plan_sha256": fake_pin,
              "supervision_sha256": fake_pin}
    records = {str(args.training_plan): original_plan, str(args.worker): worker, str(args.interruption): {}}
    reads = []

    def meta_only(path):
        reads.append(str(path))
        assert str(path) in records, 'No saved outcomes may be decoded while freezing'
        return copy.deepcopy(records[str(path)])

    def fake_descriptor(path):
        return {"path": str(path), "bytes": 1,
                "sha256": diagnostic.ORIGINAL_PIN if str(path) == diagnostic.ORIGINAL else fake_pin}

    monkeypatch.setattr(diagnostic, 'read', meta_only)
    monkeypatch.setattr(diagnostic, 'descriptor', fake_descriptor)
    monkeypatch.setattr(diagnostic, 'runtime_record', lambda: runtime)
    monkeypatch.setattr(diagnostic, 'check_observation', lambda *_: None)
    qualifications = []
    monkeypatch.setattr(diagnostic, 'engineering', lambda inputs, role='engineering', components=None:
                        qualifications.append(role))
    diagnostic.freeze(args)
    result = json.loads(args.output.read_text())
    assert result['sources'][diagnostic.ORIGINAL] == diagnostic.ORIGINAL_PIN
    assert set(result['sources']) == {diagnostic.ORIGINAL, diagnostic.PROTOCOL} | diagnostic.COMPONENTS | diagnostic.LAUNCHER_COMPONENTS
    assert result['technical_complete'] is result['original_process_closure'] is False
    assert result['scientific_calls'] == {'model': 0, 'optimizer': 0, 'teacher': 0, 'native': 0}
    assert set(reads) == set(records)
    assert qualifications == ['engineering', 'launcher_engineering']
    with pytest.raises(FileExistsError):
        diagnostic.freeze(args)


def test_failure_preserves_distinct_receipt_and_never_publishes_original_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, 'ROOT', tmp_path)
    obj = diagnostic.Diagnostic(SimpleNamespace(output=tmp_path / 'failed'))

    def fail():
        raise ValueError('fabricated admission rejection')

    obj.admit = fail
    with pytest.raises(ValueError, match='fabricated admission'):
        obj.execute()
    receipt = json.loads((obj.out / 'receipt.json').read_text())
    assert receipt['status'] == 'failed' and receipt['arithmetic_agreement'] is False
    assert receipt['original_process_closure'] is receipt['technical_complete'] is False
    assert receipt['model_calls'] == receipt['optimizer_calls'] == receipt['teacher_calls'] == receipt['native_calls'] == 0
    assert not (obj.out / 'audit.json').exists() and not (obj.out / 'diagnostic.json').exists()
    with pytest.raises(FileExistsError):
        obj.execute()
