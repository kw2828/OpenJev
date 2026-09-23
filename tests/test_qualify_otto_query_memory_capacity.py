"""Fabricated capacity admission, timing and updates; no empirical data or calls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("query_memory_capacity_tests", ROOT / "scripts/qualify_otto_query_memory_capacity.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
SPEC_C = importlib.util.spec_from_file_location("capacity_collector_fixture", ROOT / M.COLLECTOR)
C = importlib.util.module_from_spec(SPEC_C)
SPEC_C.loader.exec_module(C)


def test_fixed_synthetic_allocation_and_no_scientific_execution_interface():
    assert M.VERSION == "otto-query-memory-capacity-v1"
    assert M.FAMILIES == ("ordinary", "instant_delta", "trace_delta", "trace_additive", "trace_scrambled")
    assert M.CONFIGURATION["batch"] == 6 and M.CONFIGURATION["length"] == 256
    assert M.CONFIGURATION["chunk"] == 32 and M.CONFIGURATION["period"] == 4
    assert M.CONFIGURATION["synthetic_seed"] == M.CONFIGURATION["model_seed"] == 1001
    assert M.CONFIGURATION["threshold_seconds"] == 16200 and M.CONFIGURATION["training_cap_seconds"] == 21600
    assert M.LIMITS == {"seconds": 240, "rss_bytes": 4 * 1024**3, "output_bytes": 128 * 1024**2}
    assert M.PAYLOADS == {"started.json", "runtime.json", "synthetic.json", "work.jsonl", "summary.json"}
    assert "scripts/train_otto_query_memory.py" not in M.NEW_SOURCES


@pytest.mark.parametrize("length,chunks", ((1, 1), (32, 1), (33, 2), (256, 8), (2188, 69)))
def test_uniform_lengths_count_exact_pretrain_joint_and_four_branch_orders(length, chunks):
    guarded = []
    result = M.planned_chunks([length] * 54, check=lambda: guarded.append(True))
    assert result["ordinary"] == 3 * (80 + 40) * 9 * chunks
    assert all(result[name] == 3 * 40 * 9 * chunks for name in M.FAMILIES[1:])
    assert len(guarded) == 3 * (80 + 40)


def test_mixed_lengths_follow_pcg64_restarts_not_global_or_continued_stream():
    lengths = [1 + ((i * 677 + 79) % 2188) for i in range(54)]
    first80, first40, continued40 = 0, 0, 0
    for seed in M.SEEDS:
        generator = np.random.Generator(np.random.PCG64(seed))
        epoch_counts = []
        for _ in range(120):
            order = generator.permutation(54)
            epoch_counts.append(sum((max(lengths[int(i)] for i in order[j:j + 6]) + 31) // 32
                                    for j in range(0, 54, 6)))
        first80 += sum(epoch_counts[:80])
        first40 += sum(epoch_counts[:40])
        continued40 += sum(epoch_counts[80:])
    result = M.planned_chunks(lengths)
    assert result == {"ordinary": first80 + first40, **dict.fromkeys(M.FAMILIES[1:], first40)}
    assert first40 != continued40
    assert M.planned_chunks(lengths) == result


@pytest.mark.parametrize("bad", ([1] * 53, [1] * 55, [0] * 54, [2189] * 54, [True] * 54, (1,) * 54))
def test_invalid_metadata_length_allocation_fails_before_order_generation(bad):
    with pytest.raises(ValueError, match="exact54"):
        M.planned_chunks(bad)


def test_projection_formula_and_equal_threshold_admission():
    rows = [{"family": name, "batch_seconds": 8., "forward_chunks": 8, "work": {}}
            for name in M.FAMILIES]
    counts = dict.fromkeys(M.FAMILIES, 1560)
    summary = M.capacity_summary(rows, counts)
    assert summary["projected_seconds"] == 16200
    assert summary["admitted"] is summary["technical_ready"] is True
    counts["ordinary"] += 1
    failed = M.capacity_summary(rows, counts)
    assert failed["projected_seconds"] == 16202
    assert failed["technical_ready"] is True and failed["admitted"] is False
    assert all(failed[key] == 0 for key in ("empirical_array_decodes", "empirical_checkpoint_decodes",
                                          "teacher_calls", "native_calls"))


@pytest.mark.parametrize("defect", ("missing", "reorder", "zero", "negative", "nan", "inf", "wrong_chunks", "wrong_counts"))
def test_projection_never_promotes_incomplete_or_invalid_timings(defect):
    rows = [{"family": name, "batch_seconds": 1., "forward_chunks": 8, "work": {}}
            for name in M.FAMILIES]
    counts = dict.fromkeys(M.FAMILIES, 1)
    if defect == "missing":
        rows.pop()
    elif defect == "reorder":
        rows.reverse()
    elif defect == "wrong_chunks":
        rows[0]["forward_chunks"] = 7
    elif defect == "wrong_counts":
        counts["ordinary"] = True
    else:
        rows[0]["batch_seconds"] = {"zero": 0, "negative": -1, "nan": float("nan"), "inf": float("inf")}[defect]
    with pytest.raises(ValueError):
        M.capacity_summary(rows, counts)


def dump(path, value):
    path.write_text(json.dumps(value))
    return path


def collection_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "ROOT", tmp_path)
    directory = tmp_path / "collection"
    directory.mkdir()
    roster = C.cohort()
    (tmp_path / "fabricated-source.py").write_text("# fabricated source\n")
    sources = {"fabricated-source.py": M.descriptor("fabricated-source.py")["sha256"]}
    plan = {"version": C.VERSION, "status": "frozen_before_collection", "configuration": C.CONFIGURATION,
            "limits": C.LIMITS, "call_caps": C.CALL_CAPS, "cohort": roster, "payloads": sorted(C.PAYLOADS),
            "sources": sources, "runtime": {"python_executable": str(tmp_path / ".venv-otto-released-native/bin/python"),
                "python_version": "3.fabricated", "all_distributions": {"fabricated-native": "1"}},
            "inputs": {}, "native_inputs": {}}
    plan_path = dump(tmp_path / "collection-plan.json", plan)
    inputs = {"collection_plan": {"path": str(plan_path), **M.descriptor(plan_path)}}
    launch_path = tmp_path / "collection-launch.json"
    command = [str(tmp_path / ".venv-otto-released-native/bin/python"), str(tmp_path / M.COLLECTOR), "run",
               "--plan", str(plan_path), "--plan-sha256", inputs["collection_plan"]["sha256"],
               "--supervision", str(launch_path), "--output", str(directory)]
    launch = {"command": command, "cwd": str(tmp_path), "pid": 201, "pgid": 201, "parent_pid": 200,
              "started_ns": 100, "deadline_ns": 100 + 7200 * 10**9, "cap_seconds": 7200,
              "clock_backend": "fabricated", "clock_source_sha256": M.CLOCK_PIN,
              "watchdog_sha256": M.SUPERVISOR_PIN}
    dump(launch_path, launch)
    terminal = {**launch, "status": "completed", "returncode": 0, "timed_out": False,
                "group_absent": True, "cleanup": {"reaped": True, "group_absent": True, "errors": []},
                "error": None, "clock_error": None, "timing_available": True, "finished_ns": 400}
    terminal_path = dump(tmp_path / "collection-terminal.json", terminal)
    inputs["collection_terminal"] = {"path": str(terminal_path), **M.descriptor(terminal_path)}
    for name in C.PAYLOADS:
        (directory / name).write_bytes(b"fabricated opaque bytes, deliberately not a valid NPZ or GZIP")
    rows = [{**identity, "steps": 1 + identity["episode_index"] % 11} for identity in roster]
    (directory / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    request = {"mode": "run", "plan": str(plan_path), "plan_sha256": inputs["collection_plan"]["sha256"],
               "supervision": str(launch_path), "output": str(directory)}
    dump(directory / "started.json", {"launch": launch, "started_ns": 200, "request": request})
    dump(directory / "runtime.json", {"executable": plan["runtime"]["python_executable"],
        "python": "3.fabricated native", "all_distributions": plan["runtime"]["all_distributions"]})
    total = sum(row["steps"] for row in rows)
    numbers = {name: (0 if "annotation_public" in name else 1 if name in (
        "tensorflow_construction", "tensorflow_build", "tensorflow_load") else 108 if name in (
        "native_reset", "actor_construction", "backend_binding") else total) for name in C.CALL_CAPS}
    receipt = {"version": C.VERSION, "status": "completed", "complete": True,
               "plan_sha256": inputs["collection_plan"]["sha256"], "sources": sources, "inputs": {}, "native_inputs": {},
               "training_updates": 0, "completed_episodes": 108, "train_episodes": 54, "dev_episodes": 18,
               "test_episodes": 36, "pending": [], "pending_emission": None, "pending_episode": None,
               "pending_action": None, "requires_successful_original_supervisor": True, "limits": C.LIMITS,
               "peak_rss_bytes": 1024,
               "started_ns": 200, "finished_ns": 300, "supervision_sha256": M.descriptor(launch_path)["sha256"],
               "calls": {name: {"attempted": n, "returned": n, "seconds": 0.} for name, n in numbers.items()},
               "files": {name: M.descriptor(directory / name) for name in C.PAYLOADS}}
    receipt_path = dump(directory / "receipt.json", receipt)
    inputs["collection_receipt"] = {"path": str(receipt_path), **M.descriptor(receipt_path)}
    collector = SimpleNamespace(**{name: getattr(C, name) for name in (
        "VERSION", "CONFIGURATION", "LIMITS", "CALL_CAPS", "PAYLOADS", "cohort")},
        authenticate_inputs=lambda _: pytest.fail("never execute the live native authenticator"))
    return inputs, collector, plan, receipt, terminal, rows


def test_complete_collection_admission_reads_no_array_or_checkpoint(tmp_path, monkeypatch):
    inputs, collector, _, _, _, rows = collection_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("capacity must never decode empirical arrays"))
    sources, metadata = M.authenticate_collection(inputs, collector)
    assert sources == {"fabricated-source.py": M.descriptor("fabricated-source.py")["sha256"]}
    assert metadata["train_lengths"] == [row["steps"] for row in rows[:54]]
    assert set(metadata) == {"path", "sha256", "bytes", "train_lengths", "scope"}
    assert "features" not in str(metadata) and "raw_q" not in str(metadata)


@pytest.mark.parametrize("defect", ("failed_worker", "pending", "wrong_split", "wrong_parent", "unreaped",
    "live_group", "cleanup_group", "deadline", "wrong_launch", "wrong_started", "wrong_command", "bad_payload",
    "extra_payload", "bad_counts", "bad_identity", "bad_length", "missing_episode", "changed_source", "changed_runtime", "rss"))
def test_collection_defects_fail_before_capacity_models_or_arrays(defect, tmp_path, monkeypatch):
    inputs, collector, _, receipt, terminal, rows = collection_fixture(tmp_path, monkeypatch)
    directory = Path(inputs["collection_receipt"]["path"]).parent
    if defect == "failed_worker":
        receipt["status"] = "failed"
    elif defect == "pending":
        receipt["pending_action"] = {"action": 1}
    elif defect == "wrong_split":
        receipt["test_episodes"] = 35
    elif defect == "wrong_parent":
        terminal["returncode"] = 1
    elif defect in ("unreaped", "cleanup_group"):
        terminal["cleanup"]["reaped" if defect == "unreaped" else "group_absent"] = False
    elif defect == "live_group":
        terminal["group_absent"] = False
    elif defect == "deadline":
        terminal["finished_ns"] = terminal["deadline_ns"] + 1
    elif defect == "wrong_launch":
        receipt["supervision_sha256"] = "0" * 64
    elif defect == "wrong_started":
        record = json.loads((directory / "started.json").read_text())
        record["request"]["output"] = str(tmp_path / "wrong-output")
        dump(directory / "started.json", record)
        receipt["files"]["started.json"] = M.descriptor(directory / "started.json")
    elif defect == "wrong_command":
        terminal["command"] = terminal["command"][:-1] + [str(tmp_path / "wrong-output")]
    elif defect == "bad_payload":
        (directory / "train.npz").write_text("changed opaque bytes")
    elif defect == "extra_payload":
        (directory / "unexpected.txt").write_text("undeclared")
    elif defect == "bad_counts":
        receipt["calls"]["teacher_score"]["returned"] -= 1
    elif defect == "changed_source":
        (tmp_path / "fabricated-source.py").write_text("# changed bytes\n")
    elif defect == "changed_runtime":
        runtime_path = directory / "runtime.json"
        runtime = json.loads(runtime_path.read_text())
        runtime["all_distributions"] = {"different-native": "2"}
        dump(runtime_path, runtime)
        receipt["files"]["runtime.json"] = M.descriptor(runtime_path)
    elif defect == "rss":
        receipt["peak_rss_bytes"] = C.LIMITS["rss_bytes"] + 1
    else:
        if defect == "bad_identity":
            rows[0]["seed"] += 1
        elif defect == "bad_length":
            rows[0]["steps"] = 0
        else:
            rows.pop()
        (directory / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        receipt["files"]["episodes.jsonl"] = M.descriptor(directory / "episodes.jsonl")
    dump(Path(inputs["collection_receipt"]["path"]), receipt)
    dump(Path(inputs["collection_terminal"]["path"]), terminal)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("no empirical decode"))
    with pytest.raises(ValueError):
        M.authenticate_collection(inputs, collector)


def test_parent_failure_precedes_episode_metadata_parsing(tmp_path, monkeypatch):
    inputs, collector, _, receipt, terminal, _ = collection_fixture(tmp_path, monkeypatch)
    directory = Path(inputs["collection_receipt"]["path"]).parent
    (directory / "episodes.jsonl").write_text("not even JSON")
    receipt["files"]["episodes.jsonl"] = M.descriptor(directory / "episodes.jsonl")
    terminal["timed_out"] = True
    dump(Path(inputs["collection_receipt"]["path"]), receipt)
    dump(Path(inputs["collection_terminal"]["path"]), terminal)
    with pytest.raises(ValueError, match="original collection parent"):
        M.authenticate_collection(inputs, collector)


def test_dev_test_lengths_and_outcomes_never_enter_capacity_metadata(tmp_path, monkeypatch):
    inputs, collector, _, receipt, _, rows = collection_fixture(tmp_path, monkeypatch)
    directory = Path(inputs["collection_receipt"]["path"]).parent
    for row in rows[54:]:
        row["steps"] = "uninspected metadata fixture"
        row["raw_q"] = "not a capacity input"
        row["found"] = "not a capacity input"
    (directory / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt["files"]["episodes.jsonl"] = M.descriptor(directory / "episodes.jsonl")
    dump(Path(inputs["collection_receipt"]["path"]), receipt)
    _, metadata = M.authenticate_collection(inputs, collector)
    assert metadata["train_lengths"] == [row["steps"] for row in rows[:54]]


@pytest.mark.parametrize("missing_core", (False, True))
def test_current_descriptor_engineering_must_cover_capacity_and_training_core(missing_core, tmp_path, monkeypatch):
    monkeypatch.setattr(M, "ROOT", tmp_path)
    names = M.NEW_SOURCES | {"scripts/collect_otto_query_memory.py"}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fabricated source\n")
    monkeypatch.setattr(M, "COLLECTOR_PIN", M.descriptor(M.COLLECTOR)["sha256"])
    monkeypatch.setattr(M, "CLOCK_PIN", M.descriptor(M.CLOCK)["sha256"])
    monkeypatch.setattr(M, "SUPERVISOR_PIN", M.descriptor(M.SUPERVISOR)["sha256"])
    qualified = tmp_path / "qualification"
    qualified.mkdir()
    (qualified / "pytest-temp").mkdir()
    (qualified / "command-01.log").write_text("fabricated passed\n")
    log = M.descriptor(qualified / "command-01.log")
    descriptors = {name: M.descriptor(name) for name in M.COMPONENTS}
    qualification = {"status": "passed", "source_before": dict(descriptors), "source_after": dict(descriptors),
        "commands": [{"returncode": 0, "timed_out": False, "reaped": True, "log": "command-01.log", **log}],
        "files": {"command-01.log": log}}
    if missing_core:
        qualification["source_after"].pop(M.CORE)
        qualification["source_before"].pop(M.CORE)
    inputs = {}
    for role in M.ROLES:
        path = dump(qualified / "receipt.json" if role == "engineering" else tmp_path / f"{role}.json",
                    qualification if role == "engineering" else {})
        inputs[role] = {"path": str(path), **M.descriptor(path)}
    collector = SimpleNamespace()
    monkeypatch.setattr(M, "load", lambda *_: collector)
    monkeypatch.setattr(M, "authenticate_collection", lambda *_: ({}, {"train_lengths": [1] * 54}))
    if missing_core:
        with pytest.raises(ValueError, match="capacity/core qualifications"):
            M.authenticate_inputs(inputs)
    else:
        sources, metadata = M.authenticate_inputs(inputs)
        assert M.COMPONENTS <= sources.keys() and metadata["train_lengths"] == [1] * 54


class Clock:
    def __init__(self):
        self.now = 10

    def now_ns(self):
        return self.now


def run_fixture(tmp_path):
    run = M.Run(SimpleNamespace(output=tmp_path))
    run.clock = Clock()
    run.check = lambda: None
    return run


def test_original_timer_records_attempt_return_and_exact_positive_duration(tmp_path):
    run = run_fixture(tmp_path)
    def operation():
        assert run.receipt["pending"]["operation"] == "fabricated_batch"
        run.clock.now += 3_500_000_000
        return {"fabricated": True}
    value, seconds = run.call("ordinary", "fabricated_batch", operation)
    assert value == {"fabricated": True} and seconds == 3.5
    records = [json.loads(line) for line in (tmp_path / "work.jsonl").read_text().splitlines()]
    assert [row["event"] for row in records] == ["attempt", "return"]
    assert records[1]["seconds"] == 3.5 and records[0]["call_id"] == records[1]["call_id"] == 1
    assert run.receipt["pending"] is run.receipt["pending_emission"] is None


def test_failed_batch_remains_pending_without_synthetic_return_or_update(tmp_path):
    run = run_fixture(tmp_path)
    error = OSError("fabricated original failure")
    def fail():
        raise error
    with pytest.raises(OSError) as caught:
        run.call("trace_delta", "batch_update_and_validation", fail)
    assert caught.value is error
    assert run.receipt["pending"]["operation"] == "batch_update_and_validation"
    assert run.receipt["optimizer_updates"] == 0 and run.receipt["completed_families"] == []
    records = [json.loads(line) for line in (tmp_path / "work.jsonl").read_text().splitlines()]
    assert [row["event"] for row in records] == ["attempt"]


def test_synthetic_fixture_uses_local_rng_and_full_census_projection(tmp_path, monkeypatch):
    from openjev.research import otto_query_memory_data

    run = run_fixture(tmp_path)
    run.np, run.data = np, otto_query_memory_data
    before = np.random.get_state()
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("no saved data can seed synthetic work"))
    data = run.synthetic()
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert data["episode_count"] == 6 and data["stage"] == "train" and data["query_period"] == 4
    assert data["episode_offsets"].tolist() == list(range(0, 1537, 256))
    assert data["query_mask"].sum() == 384 and data["prior_mask"].sum() == 378
    assert (data["nonquery_weights"] > 0).sum() == 1152
    assert np.isnan(data["query_scores"][~data["query_mask"]]).all()
    assert np.isfinite(data["targets"]).all() and np.ptp(data["targets"]) > 0
    assert np.ptp(data["features"][:, 0]) > 0
    assert np.array_equal(data["features"][:, 16], ((np.tile(np.arange(256), 6) % 4) / 2188).astype(np.float32))
    record = json.loads((tmp_path / "synthetic.json").read_text())
    assert record["empirical_values_used"] is False
    assert set(record["arrays"]) == set(C.ARRAY_KEYS)


@pytest.mark.parametrize("family", M.FAMILIES)
def test_family_orchestration_uses_one_core_batch_and_only_effective_update(family, tmp_path):
    import torch

    run = run_fixture(tmp_path)
    run.torch = torch
    events = []
    ordinary = family == "ordinary"
    names = ["slow.fabricated"] if ordinary else ["slow.fabricated", "projection.weight"]
    parameters = [torch.nn.Parameter(torch.ones(1)) for _ in names]
    named = list(zip(names, parameters, strict=True))
    effective = named if ordinary else named[1:]
    fake_model = SimpleNamespace(named_parameters=lambda: named,
        effective_named_parameters=lambda: effective, parameter_metadata=lambda: {"fabricated": True})
    optimizer = SimpleNamespace(state={})

    def construct(config, *, seed, query_period, slow_mode):
        assert config.mode == ("none" if ordinary else family) and config.key_dim == 8
        assert seed == 1001 and query_period == 4 and slow_mode == ("joint" if ordinary else "frozen")
        events.append("model")
        return fake_model

    def construct_optimizer(model):
        assert model is fake_model
        events.append("optimizer")
        return optimizer

    supplied_data = {"fabricated": True}

    def batch(model, opt, data, indices, *, check, stage):
        assert model is fake_model and opt is optimizer and data is supplied_data and indices == list(range(6))
        events.append("one_whole_batch")
        stage("chunk_materialize", 0)
        check()
        with torch.no_grad():
            for _, parameter in effective:
                parameter.add_(.125)
                parameter.grad = torch.ones_like(parameter)
                optimizer.state[parameter] = {"step": torch.tensor(1.)}
        run.clock.now += 2_000_000_000
        return {"forward_chunks": 8, "forward_rows": 1536, "nonquery_rows": 1152, "prior_rows": 378,
                "optimizer_updates": 1, "optimizer_step": 1, "backward_chunks": 8,
                "frozen_optimizer_state_entries": 0, "effective_parameter_names": [name for name, _ in effective],
                "effective_parameter_count": 6112 if ordinary else 224,
                "work_counts": {"fabricated_rows": 1536}, "memory_work_units": {}}

    run.memory = SimpleNamespace(Config=lambda mode, key_dim: SimpleNamespace(mode=mode, key_dim=key_dim))
    run.models = SimpleNamespace(make_model=construct)
    run.core = SimpleNamespace(construct_optimizer=construct_optimizer, batch_update=batch)
    row = run.one_family(family, supplied_data)
    assert events == ["model", "optimizer", "one_whole_batch"]
    assert row["batch_seconds"] == 2 and row["forward_chunks"] == 8
    assert row["changed_parameter_names"] == [name for name, _ in effective]
    assert run.receipt["completed_families"] == [family] and run.receipt["completed_family_count"] == 1
    assert run.receipt["optimizer_updates"] == 1 and run.receipt["pending"] is None
    if not ordinary:
        assert torch.equal(parameters[0], torch.ones(1)) and parameters[0].grad is None


def test_real_fabricated_eight_chunk_trace_batch_with_test_clock(tmp_path):
    """Engineering-only real Adam; fabricated values, no timing/performance claim."""
    import torch

    from openjev.research import otto_query_memory as memory
    from openjev.research import otto_query_memory_data as data
    from openjev.research import otto_query_memory_model as models
    from openjev.research import otto_query_memory_training as core

    class TestClock:
        def __init__(self):
            self.value = 0

        def now_ns(self):
            self.value += 1_000_000
            return self.value

    run = run_fixture(tmp_path)
    run.clock = TestClock()
    run.np, run.torch, run.memory, run.data, run.models, run.core = np, torch, memory, data, models, core
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        synthetic = run.synthetic()
        row = run.one_family("trace_delta", synthetic)
    finally:
        torch.set_num_threads(threads)
    assert row["batch_seconds"] == .001  # Deterministic test clock, not measured throughput.
    assert row["forward_chunks"] == row["work"]["backward_chunks"] == 8
    assert row["work"]["forward_rows"] == 1536 and row["work"]["optimizer_updates"] == 1
    assert row["work"]["effective_parameter_names"] == ["projection.weight"]
    assert row["work"]["effective_parameter_count"] == 224
    assert row["changed_parameter_names"] == ["projection.weight"]
    assert row["work"]["work_counts"]["projection_key_rows"] == 6 * 255
    assert row["work"]["work_counts"]["memory_matrix_writes"] == 6 * 63
    assert run.receipt["completed_families"] == ["trace_delta"] and run.receipt["optimizer_updates"] == 1
    assert run.receipt["empirical_array_decodes"] == run.receipt["empirical_checkpoint_decodes"] == 0


def test_capacity_failure_receipt_is_exclusive_and_keeps_pending_work(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "ROOT", tmp_path)
    out = tmp_path / "failed"
    run = M.Run(SimpleNamespace(output=out))
    error = RuntimeError("fabricated admission stop before numerical work")
    def fail():
        run.receipt["pending"] = {"operation": "metadata_admission"}
        raise error
    run.admit = fail
    monkeypatch.setattr(M.signal, "signal", lambda *args: None)
    with pytest.raises(RuntimeError) as caught:
        run.execute()
    assert caught.value is error
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["complete"] is receipt["admitted"] is False
    assert receipt["pending"] == {"operation": "metadata_admission"}
    assert receipt["completed_families"] == [] and receipt["optimizer_updates"] == 0
    assert receipt["empirical_array_decodes"] == receipt["teacher_calls"] == receipt["native_calls"] == 0
    before = (out / "receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        run.execute()
    assert (out / "receipt.json").read_bytes() == before


def test_successful_execute_can_close_technical_work_but_deny_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "ROOT", tmp_path)
    monkeypatch.setattr(M.signal, "signal", lambda *args: None)
    plan_path = dump(tmp_path / "plan.json", {"fabricated": True})
    launch_path = dump(tmp_path / "launch.json", {"fabricated": True})
    args = SimpleNamespace(output=tmp_path / "closed", plan=plan_path,
                           plan_sha256=M.descriptor(plan_path)["sha256"], supervision=launch_path)
    run = M.Run(args)
    run.clock, run.start, run.check = Clock(), 1, lambda: None
    def admit():
        run.plan = {"sources": {}, "inputs": {}, "runtime": M.runtime_record(),
                    "collection_metadata": {"path": str(plan_path), **M.descriptor(plan_path)}}
        run.receipt["supervision_sha256"] = M.descriptor(launch_path)["sha256"]
    def body():
        for name in M.PAYLOADS:
            (args.output / name).write_text("fabricated\n")
        run.receipt.update(completed_families=list(M.FAMILIES), completed_family_count=5, optimizer_updates=5)
        return {"admitted": False}
    run.admit, run.body = admit, body
    run.execute()
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["status"] == "completed" and receipt["complete"] is True and receipt["admitted"] is False
    assert receipt["completed_families"] == list(M.FAMILIES) and receipt["completed_family_count"] == 5
    assert receipt["requires_successful_original_supervisor"] is True and set(receipt["files"]) == M.PAYLOADS
    assert {p.name for p in args.output.iterdir()} == M.PAYLOADS | {"receipt.json"}
