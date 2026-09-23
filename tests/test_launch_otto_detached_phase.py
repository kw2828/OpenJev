"""Fabricated process-control tests only; no model, data or scientific calls."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/launch_otto_detached_phase.py"
SPEC = importlib.util.spec_from_file_location("otto_detached_launch_test", SOURCE)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture
def fabricated(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher, "native_clock_backend", lambda: "CLOCK_BOOTTIME")
    calls = []

    def spawn(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=1234567)

    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    prefix = tmp_path / "output" / "fabricated" / "phase"
    command = [sys.executable, "-c", "pass", "--supervision", f"{prefix}.launch.json"]
    return prefix, command, calls


def read(path):
    return json.loads(Path(path).read_text())


def test_spawn_is_detached_exclusive_and_only_reports_launch(fabricated, monkeypatch):
    prefix, command, calls = fabricated
    monkeypatch.setenv("OMP_NUM_THREADS", "23")
    result = launcher.launch(prefix, 7, command)
    paths = launcher.artifact_paths(prefix)
    intent = read(paths["intent"])
    assert result == read(paths["acknowledgement"])
    assert result["status"] == "launched" and result["worker_completion"] == "not_observed"
    assert result["observation_not_completion"]
    assert result["supervisor_pid"] == 1234567
    assert result["intent_sha256"] == launcher.digest(paths["intent"])
    assert not paths["launch"].exists() and not paths["terminal"].exists()
    assert len(calls) == 1
    command_actual, kwargs = calls[0]
    assert command_actual == intent["supervisor_command"]
    assert command_actual[-len(command):] == command
    assert kwargs["start_new_session"] and kwargs["close_fds"]
    assert kwargs["stdin"] == subprocess.DEVNULL and kwargs["stderr"] == subprocess.STDOUT
    assert kwargs["stdout"].name == str(paths["supervisor_log"])
    assert kwargs["stdout"].mode == "x"
    assert kwargs["cwd"] == launcher.ROOT
    assert all(kwargs["env"][name] == "1" for name in launcher.THREAD_ENV)
    assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert not intent["retry_allowed"] and not intent["resume_allowed"]
    before = {p.name: p.read_bytes() for p in prefix.parent.iterdir()}
    with pytest.raises(FileExistsError):
        launcher.launch(prefix, 7, command)
    assert len(calls) == 1
    assert before == {p.name: p.read_bytes() for p in prefix.parent.iterdir()}


@pytest.mark.parametrize("cap", [0, -1, 1.0, True, "3", None])
def test_invalid_caps_cannot_spawn(fabricated, cap):
    prefix, command, calls = fabricated
    with pytest.raises(ValueError):
        launcher.launch(prefix, cap, command)
    assert not calls and not prefix.parent.exists()


@pytest.mark.parametrize("command", [None, [], "python -c pass", ["python", "pass"],
                                      ["/missing/executable", "pass"], [sys.executable, "\0"]])
def test_invalid_commands_cannot_spawn(fabricated, command):
    prefix, _, calls = fabricated
    with pytest.raises(ValueError):
        launcher.launch(prefix, 7, command)
    assert not calls and not prefix.parent.exists()


def test_wrong_cwd_rejects_before_spawn(fabricated, monkeypatch):
    prefix, command, calls = fabricated
    elsewhere = launcher.ROOT / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with pytest.raises(ValueError, match="checkout root"):
        launcher.launch(prefix, 7, command)
    assert not calls and not prefix.parent.exists()


@pytest.mark.parametrize("change", ["missing", "duplicate", "equals", "other", "relative", "no_value"])
def test_supervision_must_be_exact_original_path(fabricated, change):
    prefix, command, calls = fabricated
    if change == "missing":
        command = command[:-2]
    elif change == "duplicate":
        command += command[-2:]
    elif change == "equals":
        command[-2:] = [f"--supervision={command[-1]}"]
    elif change == "other":
        command[-1] += ".other"
    elif change == "relative":
        command[-1] = "output/fabricated/phase.launch.json"
    else:
        command.pop()
    with pytest.raises(ValueError):
        launcher.launch(prefix, 7, command)
    assert not calls and not prefix.parent.exists()


@pytest.mark.parametrize("which", ["SUPERVISOR_SHA256", "CLOCK_SHA256"])
def test_immutable_source_mismatch_rejects_before_spawn(fabricated, monkeypatch, which):
    prefix, command, calls = fabricated
    monkeypatch.setattr(launcher, which, "0" * 64)
    with pytest.raises(ValueError, match="source pin"):
        launcher.launch(prefix, 7, command)
    assert not calls and not prefix.parent.exists()


def test_native_clock_failure_rejects_before_spawn(fabricated, monkeypatch):
    prefix, command, calls = fabricated

    def unavailable():
        raise RuntimeError("fabricated unavailable native clock")

    monkeypatch.setattr(launcher, "native_clock_backend", unavailable)
    with pytest.raises(RuntimeError, match="unavailable"):
        launcher.launch(prefix, 7, command)
    assert not calls and not prefix.parent.exists()


@pytest.mark.parametrize("which", ["relative", "outside", "symlink", "dotdot", "output_itself"])
def test_prefix_must_be_canonical_and_contained(fabricated, which):
    prefix, command, calls = fabricated
    if which == "relative":
        prefix = Path("output/fabricated/phase")
    elif which == "outside":
        prefix = launcher.ROOT / "elsewhere" / "phase"
    elif which == "symlink":
        target = launcher.ROOT / "target"
        target.mkdir()
        (launcher.ROOT / "output").symlink_to(target, target_is_directory=True)
    elif which == "dotdot":
        prefix = launcher.ROOT / "output" / "x" / ".." / "phase"
    else:
        prefix = launcher.ROOT / "output"
    command[-1] = f"{prefix}.launch.json"
    with pytest.raises(ValueError):
        launcher.launch(prefix, 7, command)
    assert not calls


@pytest.mark.parametrize("key", list(launcher.artifact_paths(Path("unused"))))
@pytest.mark.parametrize("temporary", [False, True])
def test_any_existing_artifact_or_staging_file_spends_prefix(fabricated, key, temporary):
    prefix, command, calls = fabricated
    prefix.parent.mkdir(parents=True)
    path = launcher.artifact_paths(prefix)[key]
    if temporary:
        path = Path(f"{path}.tmp")
    path.write_bytes(b"original attempt must survive")
    with pytest.raises(FileExistsError):
        launcher.launch(prefix, 7, command)
    assert not calls and path.read_bytes() == b"original attempt must survive"


def test_dangling_symlink_cannot_be_overwritten(fabricated):
    prefix, command, calls = fabricated
    prefix.parent.mkdir(parents=True)
    path = launcher.artifact_paths(prefix)["intent"]
    path.symlink_to(prefix.parent / "absent")
    with pytest.raises(FileExistsError):
        launcher.launch(prefix, 7, command)
    assert not calls and path.is_symlink()


def test_spawn_failure_is_retained_and_not_retried(fabricated, monkeypatch):
    prefix, command, calls = fabricated

    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise OSError("fabricated spawn failure")

    monkeypatch.setattr(launcher.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="spawn failure"):
        launcher.launch(prefix, 7, command)
    paths = launcher.artifact_paths(prefix)
    failure = read(paths["failure"])
    assert failure["status"] == "launch_or_acknowledgement_failed"
    assert failure["supervisor_pid"] is None and failure["worker_completion"] == "unknown"
    assert not paths["launch"].exists() and not paths["terminal"].exists()
    with pytest.raises(FileExistsError):
        launcher.launch(prefix, 7, command)
    assert len(calls) == 1


def test_failed_ack_does_not_invent_terminal_or_relaunch(fabricated, monkeypatch):
    prefix, command, calls = fabricated
    paths = launcher.artifact_paths(prefix)
    original = launcher.publish

    def publish(path, value):
        if path == paths["acknowledgement"]:
            raise OSError("fabricated acknowledgement failure")
        return original(path, value)

    monkeypatch.setattr(launcher, "publish", publish)
    with pytest.raises(OSError, match="acknowledgement failure"):
        launcher.launch(prefix, 7, command)
    assert read(paths["failure"])["supervisor_pid"] == 1234567
    assert not paths["terminal"].exists() and not paths["launch"].exists()
    with pytest.raises(FileExistsError):
        launcher.launch(prefix, 7, command)
    assert len(calls) == 1


@pytest.fixture
def live_dir(request):
    configured = os.environ.get("OPENJEV_DETACHED_TEST_ROOT")
    if configured:
        base = Path(configured)
        base.mkdir(parents=True, exist_ok=True)
    else:
        base = Path(tempfile.mkdtemp(prefix="otto-detached-fabricated-", dir=ROOT / "output"))
    directory = base / request.node.name
    directory.mkdir(exist_ok=False)
    return directory


def wait_for(path, timeout=12):
    end = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= end:
            raise TimeoutError(f"fabricated receipt was not published: {path}")
        time.sleep(0.02)
    return read(path)


def stop_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_originating_group_killed_detached_supervisor_reaps_original_worker(live_dir):
    prefix = live_dir / "phase"
    worker = live_dir / "worker.py"
    driver = live_dir / "origin.py"
    worker.write_text('''import json, os, pathlib, sys, time
base = pathlib.Path(sys.argv[1])
deadline = time.monotonic() + 6
launch = base / 'phase.launch.json'
while not launch.exists():
    if time.monotonic() >= deadline: raise TimeoutError('no original launch')
    time.sleep(.01)
record = json.loads(launch.read_text())
assert record['pid'] == record['pgid'] == os.getpid() == os.getpgrp()
assert record['parent_pid'] == os.getppid()
ready = {'pid': os.getpid(), 'ppid': os.getppid(), 'pgid': os.getpgrp(),
         'sid': os.getsid(0), 'threads': os.environ['OMP_NUM_THREADS']}
def marker(name, value):
    temporary = base / (name + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, base / name)
    temporary.unlink()
marker('worker-ready.json', ready)
while not (base / 'release').exists():
    if time.monotonic() >= deadline: raise TimeoutError('fixture release not received')
    time.sleep(.01)
marker('worker-finished.json',
       {'pid': os.getpid(), 'parent_still_original': os.getppid() == record['parent_pid']})
''')
    driver.write_text('''import importlib.util, json, pathlib, sys, time
spec = importlib.util.spec_from_file_location('detached_fixture', sys.argv[1])
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
base = pathlib.Path(sys.argv[2]); prefix = base / 'phase'
command = [sys.executable, str(base / 'worker.py'), str(base), '--supervision', f'{prefix}.launch.json']
module.launch(prefix, 8, command)
time.sleep(10)
''')
    original = subprocess.Popen([sys.executable, str(driver), str(SOURCE), str(live_dir)],
                                cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
    supervisor_pid = worker_pid = None
    completed = False
    try:
        acknowledgement = wait_for(Path(f"{prefix}.detached-launch.json"), 4)
        ready = wait_for(live_dir / "worker-ready.json", 4)
        supervisor_pid, worker_pid = acknowledgement["supervisor_pid"], ready["pid"]
        assert supervisor_pid != original.pid and worker_pid != supervisor_pid
        assert os.getpgid(supervisor_pid) == os.getsid(supervisor_pid) == supervisor_pid
        assert ready["ppid"] == supervisor_pid and ready["pgid"] == ready["sid"] == worker_pid
        assert ready["threads"] == "1"
        assert not Path(f"{prefix}.terminal.json").exists()
        os.killpg(original.pid, signal.SIGKILL)
        assert original.wait(timeout=2) == -signal.SIGKILL
        os.kill(supervisor_pid, 0)
        os.kill(worker_pid, 0)
        (live_dir / "origin-killed.json").write_text(json.dumps({
            "pid": original.pid, "returncode": original.returncode,
            "reaped": True, "supervisor_confirmed_live": supervisor_pid,
            "worker_confirmed_live": worker_pid,
        }))
        (live_dir / "release").touch(exist_ok=False)
        terminal = wait_for(Path(f"{prefix}.terminal.json"))
        launch = read(Path(f"{prefix}.launch.json"))
        assert terminal["status"] == "completed" and terminal["returncode"] == 0
        assert terminal["cleanup"]["reaped"] and terminal["group_absent"]
        assert terminal["parent_pid"] == launch["parent_pid"] == supervisor_pid
        assert terminal["pid"] == launch["pid"] == worker_pid
        assert terminal["watchdog_sha256"] == launcher.SUPERVISOR_SHA256
        assert terminal["clock_source_sha256"] == launcher.CLOCK_SHA256
        assert terminal["timing_available"] and not terminal["timed_out"]
        assert terminal["clock_backend"] in {"mach_continuous_time", "CLOCK_BOOTTIME"}
        assert read(live_dir / "worker-finished.json")["parent_still_original"]
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)
        completed = True
    finally:
        if original.poll() is None:
            stop_group(original.pid)
            original.wait(timeout=2)
        if supervisor_pid and not completed:
            stop_group(supervisor_pid)
        if worker_pid and not completed:
            stop_group(worker_pid)


@pytest.mark.parametrize("kind", ["timeout", "nonzero"])
def test_launch_ack_does_not_hide_genuine_original_worker_failure(live_dir, kind):
    prefix = live_dir / "phase"
    code = "import time; time.sleep(5)" if kind == "timeout" else "raise SystemExit(3)"
    args = [sys.executable, str(SOURCE), "--prefix", str(prefix), "--cap-seconds", "1", "--",
            sys.executable, "-c", code, "--supervision", f"{prefix}.launch.json"]
    outer = subprocess.run(args, cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True,
                           text=True, timeout=5, check=False)
    assert outer.returncode == 0
    acknowledgement = json.loads(outer.stdout)
    assert acknowledgement["status"] == "launched"
    assert acknowledgement["worker_completion"] == "not_observed"
    terminal = wait_for(Path(f"{prefix}.terminal.json"), 5)
    assert terminal["status"] == "failed"
    assert terminal["parent_pid"] == acknowledgement["supervisor_pid"]
    assert terminal["cleanup"]["reaped"] and terminal["group_absent"]
    if kind == "timeout":
        assert terminal["timed_out"] and terminal["returncode"] == -signal.SIGTERM
    else:
        assert not terminal["timed_out"] and terminal["returncode"] == 3
