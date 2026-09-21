"""Synthetic supervisor failures plus one native no-op child, without models."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/supervise_dialogue_observation_v2.py"
SPEC = importlib.util.spec_from_file_location("supervise_observation_v2_test", SOURCE)
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


class FakeExecution:
    def __init__(self):
        self.ns = 100_000_000_000
        self.clock_failure = False
        self.civil = 1_700_000_000.0
        self.alive = True
        self.returncode = None
        self.pid = 12345
        self.signals = []
        self.waits = []
        self.after_wait = lambda: None
        self.after_poll = lambda: None
        self.after_kill = lambda: None

    def read(self):
        if self.clock_failure:
            raise OSError("synthetic native timer failure")
        return self.ns

    def poll(self):
        self.after_poll()
        return self.returncode

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.returncode is None:
            self.returncode = 0
            self.alive = False
            self.after_wait()
        return self.returncode

    def killpg(self, _pid, signum):
        self.signals.append(signum)
        self.alive = False
        if self.returncode is None:
            self.returncode = -signum
        self.after_kill()


@pytest.fixture
def fake(monkeypatch):
    execution = FakeExecution()
    clock = supervisor.suspend_clock.SuspendClock(execution.read)
    clock.backend = "mach_continuous_time"  # Synthetic dependency only, never CLI admission.
    monkeypatch.setattr(supervisor.suspend_clock, "SuspendClock", lambda: clock)
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **kw: execution)
    monkeypatch.setattr(supervisor, "group_exists", lambda _: execution.alive)
    monkeypatch.setattr(supervisor.os, "killpg", execution.killpg)
    monkeypatch.setattr(supervisor.time, "time", lambda: execution.civil)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    return execution


def invoke(tmp_path, cap=2):
    prefix = tmp_path / "run"
    command = [sys.executable, "fake-worker.py", "--supervision", f"{prefix}.launch.json"]
    result = supervisor.supervise(prefix, cap, command)
    assert json.loads(Path(f"{prefix}.terminal.json").read_text()) == result
    return result


def test_complete_native_interval_schema_and_short_waits(fake, tmp_path):
    fake.after_wait = lambda: setattr(fake, "ns", fake.ns + 500_000_000)
    result = invoke(tmp_path)
    launch = json.loads((tmp_path / "run.launch.json").read_text())
    assert result["status"] == "completed"
    assert result["elapsed_ns"] == 500_000_000 and result["wall_seconds"] == 0.5
    assert result["deadline_ns"] == launch["started_ns"] + 2_000_000_000
    assert launch["parent_pid"] == supervisor.os.getpid()
    assert launch["pid"] == launch["pgid"] == fake.pid
    assert result["group_absent"] and result["cleanup"]["reaped"]
    assert fake.waits[0] <= 0.25
    assert {p.name for p in tmp_path.iterdir()} == {"run.log", "run.launch.json", "run.terminal.json"}


@pytest.mark.parametrize("jump", [2_000_000_000, 20_000_000_000])
def test_suspend_or_exact_deadline_rejects_late_zero_exit(fake, tmp_path, jump):
    fake.after_wait = lambda: setattr(fake, "ns", fake.ns + jump)
    result = invoke(tmp_path)
    assert result["returncode"] == 0
    assert result["timed_out"] and result["status"] == "failed"
    assert result["timing_available"] and result["group_absent"]


def test_clock_failure_still_kills_and_preserves_unknown_timing(fake, tmp_path):
    fake.after_poll = lambda: setattr(fake, "clock_failure", True)
    result = invoke(tmp_path)
    assert result["status"] == "failed" and result["clock_error"]
    assert not result["timing_available"]
    assert result["finished_ns"] is result["elapsed_ns"] is result["wall_seconds"] is None
    assert fake.signals == [signal.SIGTERM]
    assert result["returncode"] == -signal.SIGTERM
    assert result["group_absent"]


@pytest.mark.parametrize("jump", [10**10, -(10**10)])
def test_civil_jumps_are_provenance_only(fake, tmp_path, jump):
    fake.after_wait = lambda: setattr(fake, "civil", fake.civil + jump)
    result = invoke(tmp_path)
    assert result["status"] == "completed" and result["elapsed_ns"] == 0
    assert result["finished_unix"] - result["started_unix"] == jump


def test_timeout_after_cleanup_is_failure(fake, tmp_path):
    fake.returncode = 0  # A lingering descendant keeps the process group alive.
    fake.after_kill = lambda: setattr(fake, "ns", fake.ns + 3_000_000_000)
    result = invoke(tmp_path)
    assert result["returncode"] == 0 and result["group_absent"]
    assert result["timed_out"] and result["status"] == "failed"


def test_nonzero_child_exit_is_retained(fake, tmp_path):
    fake.returncode = -signal.SIGABRT
    fake.alive = False
    result = invoke(tmp_path)
    assert result["returncode"] == -signal.SIGABRT
    assert result["status"] == "failed" and not result["timed_out"]


def test_kill_escalation_and_unreaped_descendant_are_bounded(fake, tmp_path, monkeypatch):
    fake.after_poll = lambda: setattr(fake, "clock_failure", True)

    def resistant_group(_pid, signum):
        fake.signals.append(signum)
        if signum == signal.SIGKILL:
            fake.returncode = -signal.SIGKILL

    def bounded_wait(timeout):
        fake.waits.append(timeout)
        if fake.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return fake.returncode

    monkeypatch.setattr(supervisor.os, "killpg", resistant_group)
    fake.wait = bounded_wait
    result = invoke(tmp_path)
    assert fake.signals == [signal.SIGTERM, signal.SIGKILL]
    assert fake.waits == [1.0, 1.0]
    assert result["cleanup"]["group_poll_attempts"] == supervisor.GROUP_POLLS
    assert not result["group_absent"] and result["status"] == "failed"


def test_exclusive_artifacts_are_never_replaced(fake, tmp_path):
    invoke(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        invoke(tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_bad_supervision_path_fails_before_popen(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: pytest.fail("launched"))
    result = supervisor.supervise(tmp_path / "run", 2, ["worker", "--supervision", "elsewhere"])
    assert result["status"] == "failed" and result["pid"] is None
    assert not (tmp_path / "run.launch.json").exists()
    assert (tmp_path / "run.terminal.json").exists()


def test_initial_clock_failure_is_preserved_before_popen(fake, tmp_path):
    fake.clock_failure = True
    result = invoke(tmp_path)
    assert result["status"] == "failed" and result["pid"] is None
    assert not result["timing_available"] and result["clock_error"]


def test_native_noop_process_smoke(tmp_path):
    prefix = tmp_path / "native"
    launch = f"{prefix}.launch.json"
    # Bounded bootstrap only; validates complete launch publication and identity.
    code = """
import json, os, pathlib, sys, time
p=pathlib.Path(sys.argv[sys.argv.index('--supervision')+1])
end=time.monotonic()+5
while not p.exists():
    if time.monotonic()>=end: raise TimeoutError('launch not published')
    time.sleep(.005)
v=json.loads(p.read_text())
assert v['pid']==v['pgid']==os.getpid()==os.getpgrp()
assert v['parent_pid']==os.getppid()
assert v['deadline_ns']==v['started_ns']+v['cap_seconds']*1_000_000_000
"""
    command = [sys.executable, str(SOURCE), "--prefix", str(prefix), "--cap-seconds", "5", "--",
               sys.executable, "-c", code, "--supervision", launch]
    if sys.platform == "darwin":
        command = ["/usr/bin/caffeinate", "-i", *command]
    outer = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = outer.communicate(timeout=10)
    except BaseException:
        outer.kill()
        outer.communicate(timeout=2)
        raise
    assert outer.returncode == 0, (stdout, stderr)
    with pytest.raises(ProcessLookupError):
        os.kill(outer.pid, 0)
    print(json.dumps({"native_smoke_command": command, "wrapper_pid": outer.pid,
                      "wrapper_exit_code": outer.returncode, "wrapper_absent": True}))
    result = json.loads(Path(f"{prefix}.terminal.json").read_text())
    assert result["status"] == "completed", result
    assert result["returncode"] == 0 and result["group_absent"]
    assert result["clock_backend"] in supervisor.NATIVE_BACKENDS
    assert 0 <= result["elapsed_ns"] < 5_000_000_000
