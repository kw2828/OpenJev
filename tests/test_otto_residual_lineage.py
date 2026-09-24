"""Fabricated byte/metadata provenance only; no empirical inputs or model calls."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("residual_lineage_under_test", Path(__file__).parents[1] / "scripts/otto_residual_lineage.py")
lineage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lineage)


def write(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def pin(record):
    return {key: record[key] for key in ("sha256", "bytes")}


def fabricated(root, monkeypatch, defect=None):
    """Construct independent fake audit/producer evidence around non-NPZ bytes.

    The anchors are replaced only in this tmp fixture. A deliberately rebound
    defect tests structural guards after its opaque hashes remain consistent.
    No real study receipt, journal, checkpoint or numerical array is read.
    """
    root = root.resolve()
    base = "output/otto-query-memory-v1"
    sources = {}
    names = [lineage.PRODUCER, lineage.AUDITOR, lineage.CLOCK, lineage.SUPERVISOR]
    names += [f"fabricated/source-{index:03}.txt" for index in range(141)]
    for name in names:
        sources[name] = write(root, name, ("synthetic source " + name).encode())["sha256"]
    inputs = {role: write(root, f"fabricated/{role}.json", {"synthetic": role}) for role in lineage.INPUT_ROLES}
    runtime = {"python": "fabricated", "executable": str(root / ".venv/bin/python"), "distributions": {"fabricated": "1"}}
    plan = {"version": lineage.TRAIN_VERSION, "status": "frozen_before_fitting", "sources": sources,
            "limits": lineage.TRAIN_LIMITS, "payloads": sorted(lineage.PAYLOADS), "inputs": inputs,
            "runtime": runtime, "configuration": {"synthetic": True}}
    plan_pin = write(root, base + "/training-plan-01.json", plan)
    payloads = {name: pin(write(root, base + "/training-01/" + name, ("opaque, not NumPy or JSON: " + name).encode()))
                for name in lineage.PAYLOADS}
    fits, forks = [], []
    for seed in (309000001, 309000002, 309000003):
        slow = {"sha256": str(seed).zfill(64), "tensors": {"slow.fabricated": {"sha256": str(seed).zfill(64)}}}
        projected = {"shape": [8, 28], "dtype": "torch.float32", "bytes": 896, "sha256": str(seed + 1).zfill(64)}
        parent_name = f"checkpoint-pretrained-{seed}.npz"
        branches = []
        for family in lineage.FITS:
            matrix = family not in ("pretrained", "joint_aux")
            name = f"checkpoint-{family}-{seed}.npz"
            initial = copy.deepcopy(slow)
            if matrix:
                initial["tensors"]["projection.weight"] = projected.copy()
            epochs = 80 if family == "pretrained" else 40
            row = {"family": family, "seed": seed, "checkpoint_path": name, "checkpoint": payloads[name],
                   "epochs": epochs, "steps": epochs * 9, "episode_exposures": epochs * 54,
                   "optimizer_initial_state_entries": 0, "stage": "pretrain" if family == "pretrained" else "branch",
                   "mode": family if matrix else "none", "slow_mode": "frozen" if matrix else "joint",
                   "pretrained_checkpoint": None if family == "pretrained" else {"path": parent_name, **payloads[parent_name]},
                   "initial_slow": copy.deepcopy(slow), "final_slow": copy.deepcopy(slow), "initial": initial,
                   "final": copy.deepcopy(initial), "frozen_slow_unchanged": True if matrix else None}
            fits.append(row)
            if family != "pretrained":
                branches.append({key: copy.deepcopy(row[key]) for key in ("family", "initial", "initial_slow", "checkpoint_path", "checkpoint")})
        forks.append({"seed": seed, "pretrained_checkpoint": payloads[parent_name], "pretrained_checkpoint_path": parent_name,
                      "memory_initial_projection": projected.copy(), "branches": branches})
    if defect == "missing_fit":
        fits.pop()
    elif defect == "wrong_parent_seed":
        fits[3]["pretrained_checkpoint"]["path"] = "checkpoint-pretrained-309000002.npz"
    elif defect == "changed_frozen_slow":
        fits[3]["final_slow"]["sha256"] = "f" * 64
    elif defect == "wrong_initial_slow":
        fits[1]["initial_slow"]["sha256"] = "f" * 64
    elif defect == "wrong_fork":
        forks[0]["branches"][0]["family"] = "trace_delta"
    elif defect == "nonfinal_fit":
        fits[0]["epochs"] = 79
    for name, value in (("fits.json", {"fits": fits}), ("forks.json", {"forks": forks})):
        payloads[name] = pin(write(root, base + "/training-01/" + name, value))
    barrier_names = {f"checkpoint-{family}-{seed}.npz" for seed in lineage.SEEDS for family in lineage.FITS}
    barrier_names |= {f"train-prediction-{view}-{seed}.npz" for seed in lineage.SEEDS for view in lineage.VIEWS}
    barrier_names |= {"fits.json", "forks.json", "train-views.json", "train-history.json", "train-history.npz"}
    barrier = {"event": "all18_checkpoints_and24_TRAIN_views_closed_before_DEV", "fits": 18, "train_views": 24,
               "optimizer_steps": 7560, "episode_exposures": 45360, "work_sequence": 22464,
               "files": {name: payloads[name] for name in barrier_names}}
    if defect == "incomplete_barrier":
        barrier["files"].pop("train-history.npz")
    elif defect == "wrong_barrier_pin":
        barrier["files"]["fits.json"] = {"sha256": "0" * 64, "bytes": 1}
    elif defect == "early_barrier":
        barrier["work_sequence"] = 7560
    payloads["dev-barrier.json"] = pin(write(root, base + "/training-01/dev-barrier.json", barrier))
    recorded_runtime = {**runtime, "torch_threads": 1, "interop_threads": 1, "deterministic": True, "cuda_used": False, "mps_used": False}
    if defect == "wrong_runtime":
        recorded_runtime["torch_threads"] = 2
    payloads["runtime.json"] = pin(write(root, base + "/training-01/runtime.json", recorded_runtime))

    def process(label, script, cap, tick, extra=None):
        output = root / base / (label + "-01")
        options = {"--plan": plan_pin["path"], "--plan-sha256": plan_pin["sha256"],
                   "--supervision": str(root / base / (label + "-native-01.launch.json")), "--output": str(output)}
        options.update(extra or {})
        command = [str(root / ".venv/bin/python"), "-u", str(root / script)]
        if label == "training":
            command.append("run")
        command += [item for pair in options.items() for item in pair]
        launch = {"command": command, "cwd": str(root), "cap_seconds": cap, "pid": tick, "pgid": tick, "parent_pid": tick - 1,
                  "started_ns": tick, "deadline_ns": tick + cap * 10**9, "watchdog_sha256": sources[lineage.SUPERVISOR],
                  "clock_source_sha256": sources[lineage.CLOCK]}
        launch_pin = write(root, base + "/" + label + "-native-01.launch.json", launch)
        terminal = {**launch, "status": "completed", "returncode": 0, "timed_out": False, "group_absent": True,
                    "cleanup": {"reaped": True, "group_absent": True, "errors": []}, "timing_available": True,
                    "error": None, "clock_error": None, "finished_ns": tick + 100}
        if defect == label + "_exit":
            terminal["returncode"] = 1
        elif defect == label + "_cleanup":
            terminal["cleanup"]["reaped"] = False
        elif defect == label + "_launch":
            terminal["pid"] += 1
        elif defect == label + "_timeout":
            terminal["timed_out"] = True
        elif defect == label + "_command":
            # A forged pair still has mutually consistent launch and terminal.
            launch["command"].extend(["--test", "forbidden"])
            launch_pin = write(root, base + "/" + label + "-native-01.launch.json", launch)
        terminal_pin = write(root, base + "/" + label + "-native-01.terminal.json", terminal)
        return launch, launch_pin, terminal_pin, options

    launch, launch_pin, terminal_pin, options = process("training", lineage.PRODUCER, 21600, 1000)
    started = {"launch": launch, "started_ns": 1001,
               "request": {"mode": "run", **{key[2:].replace("-", "_"): value for key, value in options.items()}}}
    if defect == "training_started":
        started["request"]["output"] = "wrong"
    payloads["started.json"] = pin(write(root, base + "/training-01/started.json", started))
    training = {"version": lineage.TRAIN_VERSION, "status": "completed", "complete": True, "plan_sha256": plan_pin["sha256"],
                "sources": sources, "inputs": inputs, "limits": lineage.TRAIN_LIMITS, "fits_completed": 18,
                "optimizer_steps": 7560, "episode_exposures": 45360, "teacher_calls": 0, "native_calls": 0,
                "test_array_decodes": 0, "technical_complete": False, "pending": None, "pending_emission": None,
                "requires_successful_original_supervisor": True, "requires_independent_saved_audit": True,
                "peak_rss_bytes": 1, "files": payloads, "supervision_sha256": launch_pin["sha256"], "started_ns": 1001, "finished_ns": 1090}
    if defect == "training_test_decode":
        training["test_array_decodes"] = 1
    elif defect == "producer_self_admits":
        training["technical_complete"] = True
    elif defect == "producer_pending":
        training["pending"] = "unfinished"
    training_pin = write(root, base + "/training-01/receipt.json", training)
    closure = {"status": "original_training_closure_authenticated", "frozen_source_count": 145, "payload_count": 80,
               "technical_complete": False, "requires_independent_saved_audit": True, "numerical_array_decodes": 0, "model_calls": 0,
               "inputs": {"training_plan": plan_pin, "training_receipt": training_pin, "training_terminal": terminal_pin},
               "payload_bytes": sum(value["bytes"] for value in payloads.values())}
    closure_pin = write(root, base + "/training-closure-01.json", closure)
    audit_inputs = {"plan": plan_pin, "worker": training_pin, "terminal": terminal_pin}
    extra = {"--worker": training_pin["path"], "--worker-sha256": training_pin["sha256"],
             "--terminal": terminal_pin["path"], "--terminal-sha256": terminal_pin["sha256"]}
    audit_launch, audit_launch_pin, audit_terminal_pin, _ = process("dev-audit", lineage.AUDITOR, 600, 2000, extra)
    audit_started = {"version": lineage.AUDIT_VERSION, "launch": audit_launch, "started_ns": 2001,
                     "inputs": audit_inputs, "authenticated_before_numerical_reads": True}
    if defect == "audit_started":
        audit_started["authenticated_before_numerical_reads"] = False
    audit_started_pin = write(root, base + "/dev-audit-01/started.json", audit_started)
    audit_result_pin = write(root, base + "/dev-audit-01/audit.json", b"opaque audit result; deliberately not JSON")
    audit = {"version": lineage.AUDIT_VERSION, "status": "completed", "complete": True, "agreement": True, "pending": None,
             "sources": sources, "inputs": audit_inputs, "source_plan_sha256": plan_pin["sha256"], "limits": lineage.AUDIT_LIMITS,
             "peak_rss_bytes": 1, "requires_successful_original_supervisor": True, "array_decodes": 70,
             "training_updates_checked": 7560, "episode_exposures_checked": 45360, "paired_work_calls_checked": 27432,
             "model_calls": 0, "optimizer_calls": 0, "teacher_calls": 0, "simulator_calls": 0, "test_array_decodes": 0,
             "files": {"started.json": pin(audit_started_pin), "audit.json": pin(audit_result_pin)},
             "supervision_sha256": audit_launch_pin["sha256"], "started_ns": 2001, "finished_ns": 2090}
    if defect == "audit_disagrees":
        audit["agreement"] = False
    elif defect == "audit_test_decode":
        audit["test_array_decodes"] = 1
    elif defect == "audit_wrong_worker":
        audit["inputs"] = {**audit_inputs, "worker": {**training_pin, "sha256": "0" * 64}}
    audit_pin = write(root, base + "/dev-audit-01/receipt.json", audit)
    audit_closure = {"status": "original_DEV_audit_closure_authenticated", "gate_passed": False,
                     "conditions_passed": 6, "conditions_total": 13, "test_evaluation_admitted": False,
                     "numerical_array_decodes_by_handoff": 0, "audit": audit_result_pin, "receipt": audit_pin, "terminal": audit_terminal_pin}
    if defect == "promote_failed_gate":
        audit_closure.update(gate_passed=True, conditions_passed=13)
    elif defect == "open_old_test":
        audit_closure["test_evaluation_admitted"] = True
    audit_closure_pin = write(root, base + "/dev-audit-closure-01.json", audit_closure)
    monkeypatch.setattr(lineage, "ROOT", root)
    monkeypatch.setattr(lineage, "PINS", {role: (record["path"], record["sha256"]) for role, record in
                                        (("training_plan", plan_pin), ("training_closure", closure_pin), ("dev_audit_closure", audit_closure_pin))})
    return root, base


def test_fabricated_opaque_evidence_authenticates_nine_checkpoints_and_all_bindings(tmp_path, monkeypatch):
    root, base = fabricated(tmp_path, monkeypatch)
    result = lineage.authenticate()
    assert set(result) == {"sources", "checkpoints", "evidence", "runtime"}
    assert len(result["sources"]) == 145
    assert set(result["checkpoints"]) == {"pretrained", "joint_aux", "trace_delta"}
    for family, seeds in result["checkpoints"].items():
        assert set(seeds) == {"309000001", "309000002", "309000003"}
        for seed, record in seeds.items():
            assert record["path"] == str(root / base / "training-01" / f"checkpoint-{family}-{seed}.npz")
            assert set(record) == {"path", "sha256", "bytes"}
            assert Path(record["path"]).read_bytes().startswith(b"opaque, not NumPy")
    assert sum(role.startswith("training_payload:") for role in result["evidence"]) == 80
    assert sum(role.startswith("dev_audit_payload:") for role in result["evidence"]) == 2
    assert sum(role.startswith("prior_input:") for role in result["evidence"]) == 7
    assert not any(Path(record["path"]).name == "test.npz" for record in result["evidence"].values())
    assert result["runtime"]["executable"] == str(root / ".venv/bin/python")
    # Repeated calls are independent; outputs cannot mutate retained provenance.
    result["sources"].clear()
    result["checkpoints"]["pretrained"]["309000001"]["sha256"] = "bad"
    assert len(lineage.authenticate()["sources"]) == 145


@pytest.mark.parametrize("defect", (
    "missing_fit", "wrong_parent_seed", "changed_frozen_slow", "wrong_initial_slow", "wrong_fork", "nonfinal_fit",
    "incomplete_barrier", "wrong_barrier_pin", "early_barrier", "wrong_runtime", "training_started",
    "training_test_decode", "producer_self_admits", "producer_pending", "audit_started", "audit_disagrees",
    "audit_test_decode", "audit_wrong_worker", "promote_failed_gate", "open_old_test",
    "training_exit", "training_cleanup", "training_launch", "training_timeout", "training_command",
    "dev-audit_exit", "dev-audit_cleanup", "dev-audit_launch", "dev-audit_timeout", "dev-audit_command",
))
def test_rebound_metadata_defects_still_fail_structural_authentication(tmp_path, monkeypatch, defect):
    fabricated(tmp_path, monkeypatch, defect)
    with pytest.raises(ValueError):
        lineage.authenticate()


@pytest.mark.parametrize("defect", ("anchor", "source", "checkpoint", "audit_bytes", "missing", "extra", "symlink"))
def test_opaque_tampering_missing_and_unsafe_files_fail_without_decode(tmp_path, monkeypatch, defect):
    root, base = fabricated(tmp_path, monkeypatch)
    checkpoint = root / base / "training-01/checkpoint-trace_delta-309000003.npz"
    if defect == "anchor":
        (root / base / "training-plan-01.json").write_bytes(b"{}")
    elif defect == "source":
        (root / "fabricated/source-005.txt").write_bytes(b"different source")
    elif defect == "checkpoint":
        checkpoint.write_bytes(b"different opaque checkpoint")
    elif defect == "audit_bytes":
        (root / base / "dev-audit-01/audit.json").write_bytes(b"different opaque audit")
    elif defect == "missing":
        checkpoint.unlink()
    elif defect == "extra":
        (checkpoint.parent / "unexpected.txt").write_bytes(b"unexpected")
    else:
        backup = root / "fabricated/checkpoint-backup.bin"
        backup.write_bytes(checkpoint.read_bytes())
        checkpoint.unlink()
        checkpoint.symlink_to(backup)
    with pytest.raises(ValueError):
        lineage.authenticate()


def test_return_boundary_rechecks_source_and_payload_bytes(tmp_path, monkeypatch):
    root, base = fabricated(tmp_path, monkeypatch)
    original = lineage._fit_lineage

    def mutate_after_initial_hashes(evidence, training):
        result = original(evidence, training)
        (root / base / "training-01/checkpoint-pretrained-309000001.npz").write_bytes(b"changed after metadata check")
        return result

    monkeypatch.setattr(lineage, "_fit_lineage", mutate_after_initial_hashes)
    with pytest.raises(ValueError, match="unchanged at return"):
        lineage.authenticate()
