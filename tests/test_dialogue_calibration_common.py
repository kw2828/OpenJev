"""Synthetic manifests, launch identities and lifecycle failures; no real inputs."""
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
SPEC = importlib.util.spec_from_file_location("calibration_common_test", SCRIPTS / "dialogue_calibration_common.py")
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return common.sha(path)


class FakeClock:
    backend = "mach_continuous_time"

    def __init__(self, now=101_000_000_000):
        self.now = now

    def now_ns(self):
        return self.now


@pytest.fixture
def manifest_fixture(tmp_path):
    directory = tmp_path / "owned"
    directory.mkdir()
    (directory / "nested").mkdir()
    (directory / "a.txt").write_text("synthetic payload")
    (directory / "nested/b.txt").write_text("second payload")
    files = common.members(directory)
    put(directory / "completed.json", {"status": "completed"})
    return directory, files


def test_manifest_requires_complete_membership_and_ignores_only_receipt(manifest_fixture):
    directory, files = manifest_fixture
    common.manifest(directory, files)
    assert set(files) == {"a.txt", "nested/b.txt"}


@pytest.mark.parametrize("defect", ["missing", "extra", "bytes", "digest", "same_size_change", "receipt_member",
                                    "traversal", "absolute", "descriptor_extra", "descriptor_missing"])
def test_manifest_rejects_membership_hash_and_descriptor_defects(manifest_fixture, defect, tmp_path):
    directory, files = manifest_fixture
    if defect == "missing":
        (directory / "a.txt").unlink()
    elif defect == "extra":
        (directory / "extra.txt").write_text("unlisted")
    elif defect == "bytes":
        files["a.txt"]["bytes"] += 1
    elif defect == "digest":
        files["a.txt"]["sha256"] = "0" * 64
    elif defect == "same_size_change":
        original = (directory / "a.txt").read_bytes()
        (directory / "a.txt").write_bytes(b"x" * len(original))
    elif defect == "receipt_member":
        files["completed.json"] = {"bytes": 0, "sha256": "0" * 64}
    elif defect in {"traversal", "absolute"}:
        name = "../outside.txt" if defect == "traversal" else str(tmp_path / "absolute.txt")
        files[name] = {"bytes": 0, "sha256": "0" * 64}
    elif defect == "descriptor_extra":
        files["a.txt"]["extra"] = "unbound"
    else:
        files["a.txt"].pop("bytes")
    with pytest.raises(ValueError):
        common.manifest(directory, files)


@pytest.mark.parametrize("kind", ["file", "directory", "broken"])
def test_manifest_rejects_descendant_symlinks(manifest_fixture, tmp_path, kind):
    directory, files = manifest_fixture
    target = tmp_path / "external"
    if kind == "file":
        target.write_text("external file")
    elif kind == "directory":
        target.mkdir()
    (directory / "link").symlink_to(target, target_is_directory=kind == "directory")
    with pytest.raises(ValueError):
        common.manifest(directory, files)


def test_manifest_rejects_symlink_root_even_when_target_payload_matches(manifest_fixture, tmp_path):
    directory, files = manifest_fixture
    link = tmp_path / "root-link"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError):
        common.manifest(link, files)


def test_missing_root_is_not_an_authenticated_empty_manifest(tmp_path):
    with pytest.raises(ValueError):
        common.manifest(tmp_path / "missing", {})


@pytest.fixture
def plan_fixture(monkeypatch, tmp_path):
    sources = {"source/one.py", "source/two.py"}
    for name in sources:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic source " + name)
    source_pins = {name: common.sha(tmp_path / name) for name in sources}
    input_name = "inputs/old-receipt.json"
    input_pins = {input_name: put(tmp_path / input_name, {"synthetic": True})}
    runtime = {"python": "synthetic", "numpy": "synthetic"}
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "SOURCES", sources)
    monkeypatch.setattr(common, "PINS", input_pins)
    monkeypatch.setattr(common, "environment", lambda: runtime)
    qualified = {"status": "completed", "actual_exit_code": 0, "source_sha256": source_pins,
                 "model_calls": 0, "real_task_inputs_read": False}
    qualification_path = tmp_path / "synthetic/qualification.json"
    qualification_pin = put(qualification_path, qualified)
    plan = {"version": common.VERSION, "config": copy.deepcopy(common.CONFIG),
            "limits": copy.deepcopy(common.LIMITS), "outputs": copy.deepcopy(common.OUTPUTS),
            "sources": source_pins, "inputs": input_pins, "environment": runtime,
            "synthetic_qualification": {"path": "synthetic/qualification.json", "sha256": qualification_pin}}
    path = tmp_path / "plan.json"
    pin = put(path, plan)
    return SimpleNamespace(path=path, pin=pin, plan=plan, qualification=qualified,
                           qualification_path=qualification_path, root=tmp_path)


def test_valid_plan_binds_every_synthetic_source_input_and_qualification(plan_fixture):
    calls = []
    fixture = plan_fixture
    result = common.validate_plan(fixture.path, fixture.pin, lambda: calls.append("check"))
    assert result == fixture.plan
    assert len(calls) == len(fixture.plan["sources"]) + len(fixture.plan["inputs"])


def test_bad_plan_pin_fails_before_json_decode(plan_fixture, monkeypatch):
    monkeypatch.setattr(common, "read", lambda _: pytest.fail("Decode after bad plan pin"))
    with pytest.raises(ValueError, match="Prospective control plan hash"):
        common.validate_plan(plan_fixture.path, "0" * 64)


@pytest.mark.parametrize("defect", ["version", "config", "limits", "outputs", "missing_source", "extra_source",
                                    "missing_input", "extra_input", "changed_input_pin", "runtime",
                                    "qualification_descriptor", "qualification_pin"])
def test_changed_plan_contract_is_rejected(plan_fixture, defect):
    fixture, plan = plan_fixture, copy.deepcopy(plan_fixture.plan)
    if defect == "version":
        plan["version"] = "another-study"
    elif defect == "config":
        plan["config"]["dialogues"] = 511
    elif defect == "limits":
        plan["limits"]["prepare"]["wall_seconds"] += 1
    elif defect == "outputs":
        plan["outputs"]["prepare"] = "unfrozen-output"
    elif defect == "missing_source":
        plan["sources"].pop(next(iter(plan["sources"])))
    elif defect == "extra_source":
        plan["sources"]["extra.py"] = "0" * 64
    elif defect == "missing_input":
        plan["inputs"].clear()
    elif defect == "extra_input":
        plan["inputs"]["extra.json"] = "0" * 64
    elif defect == "changed_input_pin":
        plan["inputs"][next(iter(plan["inputs"]))] = "0" * 64
    elif defect == "runtime":
        plan["environment"]["python"] = "changed"
    elif defect == "qualification_descriptor":
        plan["synthetic_qualification"]["extra"] = "unbound"
    else:
        plan["synthetic_qualification"]["sha256"] = "0" * 64
    pin = put(fixture.path, plan)
    with pytest.raises(ValueError):
        common.validate_plan(fixture.path, pin)


@pytest.mark.parametrize("binding", ["sources", "inputs"])
def test_changed_bound_bytes_are_rejected_before_qualification_read(plan_fixture, monkeypatch, binding):
    fixture = plan_fixture
    name = next(iter(fixture.plan[binding]))
    (fixture.root / name).write_text("changed bytes")
    original_read = common.read

    def guarded_read(path):
        assert Path(path) != fixture.qualification_path, "Qualification decoded after invalid source/input"
        return original_read(path)

    monkeypatch.setattr(common, "read", guarded_read)
    with pytest.raises(ValueError, match="Pinned file changed"):
        common.validate_plan(fixture.path, fixture.pin)


@pytest.mark.parametrize("field,value", [("status", "failed"), ("actual_exit_code", 1),
                                        ("source_sha256", {}), ("model_calls", 1),
                                        ("real_task_inputs_read", True)])
def test_qualification_must_match_sources_and_have_no_real_execution(plan_fixture, field, value):
    fixture = plan_fixture
    fixture.qualification[field] = value
    fixture.plan["synthetic_qualification"]["sha256"] = put(fixture.qualification_path, fixture.qualification)
    pin = put(fixture.path, fixture.plan)
    with pytest.raises(ValueError, match="Matching completed synthetic qualification"):
        common.validate_plan(fixture.path, pin)


def test_plan_changed_during_authentication_is_rejected(plan_fixture):
    fixture = plan_fixture

    def mutate():
        fixture.path.write_text(fixture.path.read_text() + " ")

    with pytest.raises(ValueError, match="Plan remained stable"):
        common.validate_plan(fixture.path, fixture.pin, mutate)


@pytest.fixture
def prior_output_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    output = tmp_path / "synthetic-preparation"
    output.mkdir()
    names = {"started.json", "actors.jsonl", "targets.jsonl", "packet.json", "selection.json", "workloads.json",
             "evaluation-rows.jsonl", "replay-cases.json", "service-exposure.json",
             "lexical-original.npy", "lexical-numbers.npy"}
    for name in names:
        (output / name).write_text("synthetic payload, never decoded")
    ctx = {"plan_sha256": "synthetic-plan", "plan_path": tmp_path / "plan.json",
           "plan": {"outputs": {"prepare": "synthetic-preparation"}, "sources": {"a.py": "source-pin"},
                    "inputs": {"input.json": "input-pin"}, "environment": {"python": "synthetic"}}}
    supervision = {"path": str(tmp_path / "launch.json"), "sha256": "synthetic-launch"}
    request = {"command": "prepare", "plan": str(ctx["plan_path"]), "plan_sha256": ctx["plan_sha256"],
               "supervision": supervision["path"], "out": str(output)}
    started = {"version": common.VERSION, "phase": "prepare", "request": request,
               "limits": common.LIMITS["prepare"], "supervision": supervision, "worker_started_ns": 100}
    put(output / "started.json", started)
    receipt = {"version": common.VERSION, "status": "completed", "phase": "prepare",
               "plan_sha256": ctx["plan_sha256"], "sources": ctx["plan"]["sources"],
               "inputs": ctx["plan"]["inputs"], "files": common.members(output),
               "request": request, "limits": common.LIMITS["prepare"], "supervision": supervision,
               "timing": {"worker_started_ns": 100}, "environment": ctx["plan"]["environment"],
               "peak_rss_bytes": 1234, "sampled_mps_driver_max_bytes": 0, "sampled_mps_current_max_bytes": 0}
    pin = put(output / "completed.json", receipt)
    return SimpleNamespace(output=output, ctx=ctx, receipt=receipt, pin=pin)


def test_prior_output_binds_identity_receipt_closure_and_payload(prior_output_fixture):
    fixture = prior_output_fixture
    assert common.authenticate_output(fixture.output, fixture.pin, fixture.ctx, "prepare") == fixture.receipt


@pytest.mark.parametrize("field,value", [("version", "other"), ("status", "failed"), ("phase", "qualify"),
                                        ("plan_sha256", "other-plan"), ("sources", {}), ("inputs", {})])
def test_prior_output_rejects_mismatched_receipt(prior_output_fixture, field, value):
    fixture = prior_output_fixture
    fixture.receipt[field] = value
    pin = put(fixture.output / "completed.json", fixture.receipt)
    with pytest.raises(ValueError):
        common.authenticate_output(fixture.output, pin, fixture.ctx, "prepare")


@pytest.mark.parametrize("defect", ["path", "receipt_pin", "payload_change", "unlisted_file"])
def test_prior_output_rejects_identity_and_payload_changes(prior_output_fixture, defect):
    fixture = prior_output_fixture
    path, pin = fixture.output, fixture.pin
    if defect == "path":
        path = fixture.output.with_name("other")
    elif defect == "receipt_pin":
        pin = "0" * 64
    elif defect == "payload_change":
        (fixture.output / "packet.json").write_text("changed")
    else:
        (fixture.output / "unlisted.txt").write_text("unbound")
    with pytest.raises(ValueError):
        common.authenticate_output(path, pin, fixture.ctx, "prepare")


@pytest.mark.parametrize("defect", ["removed_member", "extra_member"])
def test_rewritten_manifest_cannot_shrink_or_extend_required_phase_closure(prior_output_fixture, defect):
    fixture = prior_output_fixture
    if defect == "removed_member":
        (fixture.output / "actors.jsonl").unlink()
    else:
        (fixture.output / "unplanned.txt").write_text("unexpected")
    fixture.receipt["files"] = common.members(fixture.output, exclude=("completed.json",))
    pin = put(fixture.output / "completed.json", fixture.receipt)
    with pytest.raises(ValueError):
        common.authenticate_output(fixture.output, pin, fixture.ctx, "prepare")


def test_all_phase_success_closures_are_fixed_and_cover_twelve_fits():
    assert len(common.expected_members("prepare")) == 11
    assert len(common.expected_members("qualify")) == 50
    assert len(common.expected_members("infer")) == 37
    expected_fits = {f"{arm}-{seed}" for arm in common.CONFIG["arms"] for seed in common.CONFIG["seeds"]}
    assert len(expected_fits) == 12
    for phase in ("qualify", "infer"):
        assert {name.split("/")[0] for name in common.expected_members(phase) if "/" in name} == expected_fits
    with pytest.raises(ValueError):
        common.expected_members("unknown")


@pytest.mark.parametrize("defect", ["started_request", "started_phase", "started_limits", "started_supervision",
                                    "worker_start", "request_plan", "request_output", "request_supervision",
                                    "runtime", "rss", "rss_type", "mps_driver", "mps_current"])
def test_prior_output_rejects_rehashed_start_request_and_resource_mismatches(prior_output_fixture, defect):
    fixture = prior_output_fixture
    receipt = copy.deepcopy(fixture.receipt)
    started = common.read(fixture.output / "started.json")
    if defect == "started_request":
        started["request"]["command"] = "qualify"
    elif defect == "started_phase":
        started["phase"] = "qualify"
    elif defect == "started_limits":
        started["limits"]["wall_seconds"] += 1
    elif defect == "started_supervision":
        started["supervision"]["sha256"] = "changed"
    elif defect == "worker_start":
        receipt["timing"]["worker_started_ns"] += 1
    elif defect in {"request_plan", "request_output", "request_supervision"}:
        key = {"request_plan": "plan", "request_output": "out", "request_supervision": "supervision"}[defect]
        started["request"][key] += "-other"
        receipt["request"][key] = started["request"][key]
    elif defect == "runtime":
        receipt["environment"] = {"python": "changed"}
    elif defect == "rss":
        receipt["peak_rss_bytes"] = common.LIMITS["prepare"]["rss_bytes"] + 1
    elif defect == "rss_type":
        receipt["peak_rss_bytes"] = True
    elif defect == "mps_driver":
        receipt["sampled_mps_driver_max_bytes"] = 1
    else:
        receipt["sampled_mps_current_max_bytes"] = 1
    put(fixture.output / "started.json", started)
    receipt["files"] = common.members(fixture.output, exclude=("completed.json",))
    pin = put(fixture.output / "completed.json", receipt)
    with pytest.raises(ValueError):
        common.authenticate_output(fixture.output, pin, fixture.ctx, "prepare")


@pytest.fixture
def launch_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(command="prepare", plan=tmp_path / "plan.json", plan_sha256="synthetic",
                           supervision=tmp_path / "launch.json", out=tmp_path / "out")
    script = tmp_path / "scripts/prepare_dialogue_calibration.py"
    argv = [str(script), "--plan", str(args.plan), "--plan-sha256", args.plan_sha256,
            "--supervision", str(args.supervision), "--out", str(args.out)]
    interpreter = str(tmp_path / "python")
    monkeypatch.setattr(common.sys, "argv", argv)
    monkeypatch.setattr(common.sys, "executable", interpreter)
    monkeypatch.setattr(common.os, "getpid", lambda: 222)
    monkeypatch.setattr(common.os, "getpgrp", lambda: 222)
    monkeypatch.setattr(common.os, "getppid", lambda: 111)
    pins = {}
    for name in (common.SUPERVISOR, common.CLOCK):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic " + name)
        pins[name] = common.sha(path)
    monkeypatch.setattr(common, "PINS", pins)
    clock = FakeClock()
    start = 100_000_000_000
    cap = common.LIMITS[args.command]["wall_seconds"]
    launch = {"version": "dialogue-observation-supervision-v2", "command": [interpreter, "-u", *argv],
              "cwd": str(tmp_path), "pid": 222, "pgid": 222, "parent_pid": 111,
              "clock_backend": clock.backend, "started_ns": start, "deadline_ns": start + cap * 10**9,
              "cap_seconds": cap, "watchdog_sha256": pins[common.SUPERVISOR],
              "clock_source_sha256": pins[common.CLOCK]}
    pin = put(args.supervision, launch)
    return SimpleNamespace(args=args, launch=launch, pin=pin, clock=clock, worker_start=clock.now)


def test_valid_launch_shares_exact_native_absolute_deadline(launch_fixture):
    fixture = launch_fixture
    deadline = common.validate_supervision(fixture.args, fixture.launch, fixture.clock, fixture.worker_start)
    assert deadline.started_ns == fixture.launch["started_ns"]
    assert deadline.expires_ns == fixture.launch["deadline_ns"]
    assert deadline.clock is fixture.clock
    without_u = copy.deepcopy(fixture.launch)
    without_u["command"].remove("-u")
    common.validate_supervision(fixture.args, without_u, fixture.clock, fixture.worker_start)


@pytest.mark.parametrize("defect", ["version", "argv", "interpreter", "cwd", "pid", "pgid", "parent",
                                    "boolean_pid", "backend", "nonnative", "cap", "cap_type", "duration",
                                    "worker_before_parent", "worker_at_deadline", "watchdog_pin", "clock_pin"])
def test_launch_identity_and_deadline_failures(launch_fixture, defect):
    fixture, launch = launch_fixture, copy.deepcopy(launch_fixture.launch)
    worker_start = fixture.worker_start
    if defect == "version":
        launch["version"] = "unqualified"
    elif defect == "argv":
        launch["command"].append("--extra")
    elif defect == "interpreter":
        launch["command"][0] += "-other"
    elif defect == "cwd":
        launch["cwd"] += "/other"
    elif defect == "pid":
        launch["pid"] += 1
    elif defect == "pgid":
        launch["pgid"] += 1
    elif defect == "parent":
        launch["parent_pid"] += 1
    elif defect == "boolean_pid":
        launch["pid"] = True
    elif defect == "backend":
        launch["clock_backend"] = "CLOCK_BOOTTIME"
    elif defect == "nonnative":
        launch["clock_backend"] = fixture.clock.backend = "injected"
    elif defect == "cap":
        launch["cap_seconds"] += 1
    elif defect == "cap_type":
        launch["cap_seconds"] = float(launch["cap_seconds"])
    elif defect == "duration":
        launch["deadline_ns"] += 1
    elif defect == "worker_before_parent":
        worker_start = launch["started_ns"] - 1
    elif defect == "worker_at_deadline":
        worker_start = launch["deadline_ns"]
    elif defect == "watchdog_pin":
        launch["watchdog_sha256"] = "0" * 64
    else:
        launch["clock_source_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        common.validate_supervision(fixture.args, launch, fixture.clock, worker_start)


def test_launch_rejects_wrong_phase_script_even_when_argv_matches(launch_fixture, monkeypatch):
    fixture = launch_fixture
    changed = list(common.sys.argv)
    changed[0] = str(common.ROOT / "scripts/another.py")
    monkeypatch.setattr(common.sys, "argv", changed)
    fixture.launch["command"] = [common.sys.executable, *changed]
    with pytest.raises(ValueError, match="Phase worker source"):
        common.validate_supervision(fixture.args, fixture.launch, fixture.clock, fixture.worker_start)


@pytest.mark.parametrize("defect", ["different_output", "duplicate_flag", "missing_flag", "unexpected_flag"])
def test_live_argv_must_match_parsed_request_exactly(launch_fixture, monkeypatch, defect):
    fixture = launch_fixture
    argv = list(common.sys.argv)
    if defect == "different_output":
        argv[-1] += "-other"
    elif defect == "duplicate_flag":
        argv.extend(["--out", str(fixture.args.out)])
    elif defect == "missing_flag":
        argv = argv[:-2]
    else:
        argv.extend(["--unknown", "value"])
    monkeypatch.setattr(common.sys, "argv", argv)
    fixture.launch["command"] = [common.sys.executable, *argv]
    with pytest.raises(ValueError, match="live parsed request"):
        common.validate_supervision(fixture.args, fixture.launch, fixture.clock, fixture.worker_start)


def test_none_optional_arguments_do_not_require_unpassed_flags(launch_fixture):
    fixture = launch_fixture
    fixture.args.unused_optional = None
    common.validate_supervision(fixture.args, fixture.launch, fixture.clock, fixture.worker_start)


@pytest.mark.parametrize("source", [common.SUPERVISOR, common.CLOCK])
def test_modified_supervisor_or_clock_bytes_prevent_launch(launch_fixture, source):
    fixture = launch_fixture
    (common.ROOT / source).write_text("changed synthetic source")
    with pytest.raises(ValueError, match="Supervisor/clock identity"):
        common.validate_supervision(fixture.args, fixture.launch, fixture.clock, fixture.worker_start)


def test_native_deadline_expired_during_launch_validation_fails(launch_fixture):
    fixture = launch_fixture
    fixture.clock.now = fixture.launch["deadline_ns"]
    with pytest.raises(TimeoutError):
        common.validate_supervision(fixture.args, fixture.launch, fixture.clock, fixture.worker_start)


def test_await_supervision_has_bounded_wait_and_no_native_sleep(launch_fixture, monkeypatch):
    fixture = launch_fixture
    fixture.args.supervision.unlink()
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        fixture.clock.now += 5_000_000_000

    monkeypatch.setattr(common.time, "sleep", sleep)
    with pytest.raises(TimeoutError):
        common.await_supervision(fixture.args, fixture.clock, fixture.worker_start)
    assert sleeps == [0.025]


def test_await_supervision_rejects_oversized_publication_before_decode(launch_fixture, monkeypatch):
    fixture = launch_fixture
    fixture.args.supervision.write_bytes(b"x" * 65537)
    monkeypatch.setattr(common, "read", lambda _: pytest.fail("Decoded oversized launch"))
    with pytest.raises(ValueError, match="Bounded parent launch"):
        common.await_supervision(fixture.args, fixture.clock, fixture.worker_start)


def test_launch_publication_must_remain_stable_during_decode(launch_fixture, monkeypatch):
    fixture, original_read = launch_fixture, common.read

    def changed_after_read(path):
        value = original_read(path)
        path.write_text(path.read_text() + " ")
        return value

    monkeypatch.setattr(common, "read", changed_after_read)
    with pytest.raises(ValueError, match="Stable parent launch publication"):
        common.await_supervision(fixture.args, fixture.clock, fixture.worker_start)


def terminal_fixture(fixture):
    terminal = {**fixture.launch, "status": "completed", "returncode": 0, "timing_available": True,
                "timed_out": False, "error": None, "clock_error": None, "group_absent": True,
                "cleanup": {"reaped": True, "group_absent": True, "errors": []},
                "finished_ns": fixture.worker_start + 2_000_000_000}
    terminal["elapsed_ns"] = terminal["finished_ns"] - terminal["started_ns"]
    terminal["wall_seconds"] = terminal["elapsed_ns"] / 1e9
    finished = fixture.worker_start + 1_000_000_000
    elapsed = finished - fixture.launch["started_ns"]
    request = {k: str(v) if isinstance(v, Path) else v for k, v in vars(fixture.args).items()}
    receipt = {"phase": "prepare", "wall_seconds": elapsed / 1e9, "request": request,
               "supervision": {"path": str(fixture.args.supervision), "sha256": fixture.pin},
               "timing": {"clock_backend": fixture.clock.backend, "parent_started_ns": fixture.launch["started_ns"],
                          "deadline_ns": fixture.launch["deadline_ns"], "started_ns": fixture.launch["started_ns"],
                          "worker_started_ns": fixture.worker_start, "finished_ns": finished,
                          "elapsed_ns": elapsed, "wall_seconds": elapsed / 1e9}}
    path = fixture.args.supervision.with_name("terminal.json")
    return path, terminal, receipt


def test_parent_terminal_requires_success_and_worker_interval_containment(launch_fixture):
    path, terminal, receipt = terminal_fixture(launch_fixture)
    assert common.authenticate_terminal(path, put(path, terminal), receipt) == terminal


@pytest.mark.parametrize("defect", ["failed", "exit", "timing_unknown", "timeout", "error", "clock_error", "group_alive",
                                    "unreaped", "cleanup_error", "late", "equal_deadline", "launch_mismatch",
                                    "worker_before_parent", "worker_after_parent", "worker_reversed",
                                    "worker_clock_mismatch", "worker_parent_start_mismatch", "worker_deadline_mismatch",
                                    "worker_start_before_parent", "worker_start_after_finish", "worker_elapsed",
                                    "worker_wall", "receipt_wall", "parent_elapsed", "parent_wall", "wrong_phase_cap"])
def test_failed_late_or_mismatched_parent_is_not_completion(launch_fixture, defect):
    path, terminal, receipt = terminal_fixture(launch_fixture)
    mutations = {"failed": ("status", "failed"), "exit": ("returncode", 1),
                 "timing_unknown": ("timing_available", False), "timeout": ("timed_out", True),
                 "error": ("error", "failed"), "clock_error": ("clock_error", "failed"),
                 "group_alive": ("group_absent", False)}
    if defect in mutations:
        key, value = mutations[defect]
        terminal[key] = value
    elif defect == "unreaped":
        terminal["cleanup"]["reaped"] = False
    elif defect == "cleanup_error":
        terminal["cleanup"]["errors"] = ["failed"]
    elif defect in {"late", "equal_deadline"}:
        terminal["finished_ns"] = terminal["deadline_ns"] + int(defect == "late")
    elif defect == "launch_mismatch":
        terminal["pid"] += 1
    elif defect == "worker_before_parent":
        receipt["timing"]["started_ns"] = terminal["started_ns"] - 1
    elif defect == "worker_after_parent":
        receipt["timing"]["finished_ns"] = terminal["finished_ns"] + 1
    elif defect == "worker_reversed":
        receipt["timing"]["finished_ns"] = receipt["timing"]["started_ns"] - 1
    elif defect == "worker_clock_mismatch":
        receipt["timing"]["clock_backend"] = "CLOCK_BOOTTIME"
    elif defect == "worker_parent_start_mismatch":
        receipt["timing"]["parent_started_ns"] += 1
    elif defect == "worker_deadline_mismatch":
        receipt["timing"]["deadline_ns"] += 1
    elif defect == "worker_start_before_parent":
        receipt["timing"]["worker_started_ns"] = terminal["started_ns"] - 1
    elif defect == "worker_start_after_finish":
        receipt["timing"]["worker_started_ns"] = receipt["timing"]["finished_ns"] + 1
    elif defect == "worker_elapsed":
        receipt["timing"]["elapsed_ns"] += 1
    elif defect == "worker_wall":
        receipt["timing"]["wall_seconds"] += 1
    elif defect == "receipt_wall":
        receipt["wall_seconds"] += 1
    elif defect == "parent_elapsed":
        terminal["elapsed_ns"] += 1
    elif defect == "parent_wall":
        terminal["wall_seconds"] += 1
    else:
        receipt["phase"] = "qualify"
    if defect in {"late", "equal_deadline"}:
        terminal["elapsed_ns"] = terminal["finished_ns"] - terminal["started_ns"]
        terminal["wall_seconds"] = terminal["elapsed_ns"] / 1e9
    with pytest.raises(ValueError):
        common.authenticate_terminal(path, put(path, terminal), receipt)


@pytest.mark.parametrize("target", ["terminal", "launch"])
def test_parent_evidence_changed_bytes_are_rejected(launch_fixture, target):
    path, terminal, receipt = terminal_fixture(launch_fixture)
    pin = put(path, terminal)
    changed = path if target == "terminal" else launch_fixture.args.supervision
    changed.write_text(changed.read_text() + " ")
    with pytest.raises(ValueError, match="pin"):
        common.authenticate_terminal(path, pin, receipt)


@pytest.fixture
def execution_fixture(launch_fixture, monkeypatch):
    fixture = launch_fixture
    monkeypatch.setattr(common, "SuspendClock", lambda: fixture.clock)
    monkeypatch.setattr(common, "peak_rss", lambda: 1234)
    timers, handlers, auth_calls = [], [], []
    monkeypatch.setattr(common.signal, "setitimer", lambda *args: timers.append(args))

    def signal(*args):
        handlers.append(args)
        return "previous-handler"

    monkeypatch.setattr(common.signal, "signal", signal)
    ctx = {"plan": {"sources": {"synthetic.py": "a"}, "inputs": {"synthetic.json": "b"},
                    "environment": {"python": "synthetic"}}}
    monkeypatch.setattr(common, "expected_members", lambda _: {"started.json", "payload.json"})

    def authenticate(args, budget):
        budget.check()
        auth_calls.append(args.command)
        return ctx

    monkeypatch.setattr(common, "authenticate", authenticate)
    fixture.timers, fixture.handlers, fixture.auth_calls, fixture.ctx = timers, handlers, auth_calls, ctx
    return fixture


def test_exclusive_output_collision_precedes_clock_auth_and_changes_nothing(execution_fixture, monkeypatch):
    fixture = execution_fixture
    fixture.args.out.mkdir()
    sentinel = fixture.args.out / "keep.txt"
    sentinel.write_bytes(b"existing attempt")
    monkeypatch.setattr(common, "SuspendClock", lambda: pytest.fail("Clock touched after output collision"))
    with pytest.raises(FileExistsError):
        common.execute(fixture.args, lambda *args: pytest.fail("Body after collision"))
    assert sentinel.read_bytes() == b"existing attempt"
    assert list(fixture.args.out.iterdir()) == [sentinel]
    assert not fixture.auth_calls and not fixture.timers


def test_authentication_failure_records_failed_attempt_before_body(execution_fixture, monkeypatch):
    fixture = execution_fixture

    def fail(*args):
        raise ValueError("synthetic authentication failure")

    monkeypatch.setattr(common, "authenticate", fail)
    with pytest.raises(ValueError, match="synthetic authentication failure"):
        common.execute(fixture.args, lambda *args: pytest.fail("Body after authentication failure"))
    failed = common.read(fixture.args.out / "failed.json")
    assert failed["status"] == "failed" and failed["progress"]["operation"] == "authentication"
    assert failed["error_type"] == "ValueError" and not (fixture.args.out / "completed.json").exists()
    assert fixture.timers[-1][1] == 0 and fixture.handlers[-1][1] == "previous-handler"


def test_partial_body_failure_preserves_artifacts_without_completion(execution_fixture):
    fixture = execution_fixture

    def fail(args, ctx, budget):
        assert ctx is fixture.ctx
        (args.out / "partial.txt").write_text("preserve me")
        budget.progress["completed_dialogues"] = 7
        raise RuntimeError("synthetic partial failure")

    with pytest.raises(RuntimeError, match="synthetic partial failure"):
        common.execute(fixture.args, fail)
    assert (fixture.args.out / "partial.txt").read_text() == "preserve me"
    failed = common.read(fixture.args.out / "failed.json")
    assert failed["progress"] == {"operation": "phase-body", "completed_dialogues": 7}
    assert not (fixture.args.out / "completed.json").exists() and fixture.auth_calls == ["prepare"]


def test_success_reauthenticates_and_manifests_every_body_artifact(execution_fixture, capsys):
    fixture = execution_fixture

    def body(args, ctx, budget):
        common.write(args.out / "payload.json", {"synthetic": True})
        fixture.clock.now += 1_000_000_000
        return {"model_calls": 0, "status": "cannot-override", "phase": "cannot-override"}

    receipt = common.execute(fixture.args, body)
    assert receipt["status"] == "completed" and receipt["phase"] == "prepare"
    assert fixture.auth_calls == ["prepare", "prepare"]
    assert receipt["sources"] == fixture.ctx["plan"]["sources"]
    assert receipt["inputs"] == fixture.ctx["plan"]["inputs"]
    assert set(receipt["files"]) == {"started.json", "payload.json"}
    assert receipt["timing"]["deadline_ns"] == fixture.launch["deadline_ns"]
    assert receipt["timing"]["parent_started_ns"] == fixture.launch["started_ns"]
    assert receipt["timing"]["started_ns"] == fixture.launch["started_ns"]
    assert receipt["timing"]["worker_started_ns"] == fixture.worker_start
    assert receipt["timing"]["elapsed_ns"] == 2_000_000_000
    assert receipt["wall_seconds"] == 2.0
    common.manifest(fixture.args.out, receipt["files"])
    assert common.read(fixture.args.out / "completed.json") == receipt
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert fixture.timers[-1][1] == 0 and fixture.handlers[-1][1] == "previous-handler"


@pytest.mark.parametrize("point", ["body", "receipt_write", "receipt_hash"])
def test_late_completion_is_failed_and_published_receipt_is_demoted(execution_fixture, monkeypatch, point):
    fixture = execution_fixture
    original_write, original_sha = common.write, common.sha

    def late_write(path, value):
        original_write(path, value)
        if path.name == "completed.json" and point == "receipt_write":
            fixture.clock.now = fixture.launch["deadline_ns"]

    def late_sha(path):
        result = original_sha(path)
        if Path(path).name == "completed.json" and point == "receipt_hash":
            fixture.clock.now = fixture.launch["deadline_ns"]
        return result

    monkeypatch.setattr(common, "write", late_write)
    monkeypatch.setattr(common, "sha", late_sha)

    def body(args, ctx, budget):
        (args.out / "payload.json").write_text("synthetic")
        if point == "body":
            fixture.clock.now = fixture.launch["deadline_ns"]
        return {"model_calls": 0}

    with pytest.raises(TimeoutError):
        common.execute(fixture.args, body)
    assert not (fixture.args.out / "completed.json").exists()
    assert (fixture.args.out / "invalid-completion.json").exists() is (point != "body")
    assert common.read(fixture.args.out / "failed.json")["error_type"] == "TimeoutError"
    assert (fixture.args.out / "payload.json").exists()


def test_source_reauthentication_failure_after_body_prevents_completion(execution_fixture, monkeypatch):
    fixture = execution_fixture
    calls = []

    def authenticate(args, budget):
        calls.append(True)
        if len(calls) == 2:
            raise ValueError("source changed during body")
        return fixture.ctx

    monkeypatch.setattr(common, "authenticate", authenticate)
    with pytest.raises(ValueError, match="source changed during body"):
        common.execute(fixture.args, lambda *a: {"model_calls": 0})
    assert len(calls) == 2 and not (fixture.args.out / "completed.json").exists()
    assert common.read(fixture.args.out / "failed.json")["status"] == "failed"


def test_changed_parent_publication_after_body_prevents_completion(execution_fixture):
    fixture = execution_fixture

    def body(args, ctx, budget):
        args.supervision.write_text(args.supervision.read_text() + " ")
        return {}

    with pytest.raises(ValueError, match="Parent launch remained stable"):
        common.execute(fixture.args, body)
    assert not (fixture.args.out / "completed.json").exists()


def test_post_receipt_storage_overflow_demotes_completion(execution_fixture, monkeypatch):
    fixture = execution_fixture
    original_storage = common.Budget.storage

    def storage(self):
        if (self.out / "completed.json").exists():
            raise ValueError("Owned output cap")
        original_storage(self)

    monkeypatch.setattr(common.Budget, "storage", storage)

    def body(args, ctx, budget):
        common.write(args.out / "payload.json", {"synthetic": True})
        return {}

    with pytest.raises(ValueError, match="Owned output cap"):
        common.execute(fixture.args, body)
    assert not (fixture.args.out / "completed.json").exists()
    assert (fixture.args.out / "invalid-completion.json").exists()
    assert common.read(fixture.args.out / "failed.json")["status"] == "failed"


def test_budget_checks_rss_storage_and_mps_caps_without_models(tmp_path, monkeypatch):
    clock = FakeClock()
    deadline = common.Deadline(clock, clock.now, clock.now + 10**9)
    budget = common.Budget(tmp_path, deadline, {"rss_bytes": 100, "output_bytes": 3, "mps_driver_bytes": 50})
    monkeypatch.setattr(common, "peak_rss", lambda: 100)
    budget.check()
    (tmp_path / "file.txt").write_text("1234")
    with pytest.raises(ValueError, match="Owned output cap"):
        budget.storage()
    synced = []
    torch = SimpleNamespace(mps=SimpleNamespace(synchronize=lambda: synced.append(True),
                                                driver_allocated_memory=lambda: 51,
                                                current_allocated_memory=lambda: 30))
    with pytest.raises(ValueError, match="Sampled MPS driver cap"):
        budget.sync(torch)
    assert synced == [True] and budget.max_driver == 51 and budget.max_current == 30
    monkeypatch.setattr(common, "peak_rss", lambda: 101)
    with pytest.raises(ValueError, match="Process RSS cap"):
        budget.check()
