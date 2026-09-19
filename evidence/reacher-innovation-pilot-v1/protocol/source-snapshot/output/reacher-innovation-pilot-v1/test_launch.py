"""Fake subprocess tests only; no actual launch, data, model or process kill."""

import importlib.util
import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).with_name("launch.py")


@pytest.fixture
def harness(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("innovation_launcher_fake_tests", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, suffix in {"ROOT": "repo", "PROCESS": "process", "ATTEMPT": "attempt",
                         "PLAN": "plan.json", "CHILD": "child.py", "PYTHON": "python"}.items():
        monkeypatch.setattr(module, name, tmp_path / suffix)
    module.PLAN.write_text('{"synthetic":true}\n')
    digest, commit = module.sha(module.PLAN), "1" * 40
    clock = SimpleNamespace(value=0.0)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.value))
    options = {"mode": "success", "exit": 0, "missing_completed": False,
               "failed_marker": False, "invalid_marker": False, "completion_override": {}, "spawn_seconds": 0}
    spawns, waits, kills = [], [], []
    children = []

    class FakeChild:
        pid = 424242
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            waits.append(timeout)
            if self.returncode is not None:
                return self.returncode
            if options["mode"] == "timeout":
                clock.value += timeout
                raise subprocess.TimeoutExpired("synthetic child", timeout)
            if options["mode"] == "interrupt":
                raise KeyboardInterrupt("synthetic user interruption")
            clock.value += 3.0
            self.returncode = options["exit"]
            module.ATTEMPT.mkdir()
            if not options["missing_completed"]:
                receipt = {"plan_sha256": digest, "fits": 12, "evaluations": 48,
                    "development_only": True, "native_control_measured": False,
                    "gate_status": "requires_saved_output_review", "wall_seconds": 2.0}
                receipt.update(options["completion_override"])
                (module.ATTEMPT / "completed.json").write_text(json.dumps(receipt))
            for flag, name in (("failed_marker", "failed.json"), ("invalid_marker", "invalid-completion.json")):
                if options[flag]:
                    (module.ATTEMPT / name).write_text('{"retained":true}')
            return self.returncode

    def popen(argv, **kwargs):
        spawns.append((argv, kwargs))
        assert kwargs["start_new_session"] is True
        assert kwargs["cwd"] == module.ROOT
        assert kwargs["env"]["PYTHONPATH"] == f"{module.ROOT / 'src'}:{module.ROOT / 'scripts'}"
        kwargs["stdout"].write(b"synthetic stdout\n")
        kwargs["stderr"].write(b"synthetic stderr\n")
        clock.value += options["spawn_seconds"]
        child = FakeChild()
        children.append(child)
        return child

    def killpg(pid, requested_signal):
        assert children and pid == children[-1].pid
        kills.append((pid, requested_signal))
        children[-1].returncode = -requested_signal

    monkeypatch.setattr(module, "subprocess", SimpleNamespace(Popen=popen, TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(module, "os", SimpleNamespace(environ={"PRIVATE_TEST_ENV": "do_not_log"}, pathsep=":", killpg=killpg))
    return SimpleNamespace(module=module, options=options, clock=clock, digest=digest, commit=commit,
        spawns=spawns, waits=waits, kills=kills, children=children, launch=lambda: module.launch(digest, commit))


def test_one_exact_child_logs_and_completed_receipt(harness):
    h = harness
    result = h.launch()
    assert len(h.spawns) == 1 and not h.kills
    argv, _ = h.spawns[0]
    assert argv == [str(h.module.PYTHON), "-u", str(h.module.CHILD), "run",
                    "--plan-sha256", h.digest, "--published-commit", h.commit]
    assert h.waits == [1800]
    assert result["status"] == "completed" and result["exit_code"] == 0
    assert result["child_completed_sha256"] == h.module.sha(h.module.ATTEMPT / "completed.json")
    assert result["logs"]["stdout.log"]["sha256"] == h.module.sha(h.module.PROCESS / "stdout.log")
    assert result["logs"]["stderr.log"]["sha256"] == h.module.sha(h.module.PROCESS / "stderr.log")
    assert result["wall_seconds"] == result["child_wait_seconds"] == 3.0
    assert result["gate_status"] == "requires_saved_output_review"
    stored = json.loads((h.module.PROCESS / "completed.json").read_text())
    assert stored == result and not (h.module.PROCESS / "failed.json").exists()
    assert "do_not_log" not in (h.module.PROCESS / "started.json").read_text()


def test_initial_bookkeeping_and_spawn_reduce_the_actual_child_timeout(harness, monkeypatch):
    h = harness
    original = h.module.write_json
    def write(path, value):
        original(path, value)
        if path.name == "started.json":
            h.clock.value += 12
    monkeypatch.setattr(h.module, "write_json", write)
    h.options["spawn_seconds"] = 4
    result = h.launch()
    assert h.waits == [1784]
    assert result["subprocess_timeout_seconds"] == 1784
    assert result["wall_seconds"] == 19


@pytest.mark.parametrize("mode,exception", [("timeout", subprocess.TimeoutExpired), ("interrupt", KeyboardInterrupt)])
def test_timeout_and_interrupt_kill_only_owned_process_group_once_and_preserve_logs(harness, mode, exception):
    h = harness
    h.options["mode"] = mode
    with pytest.raises(exception):
        h.launch()
    assert len(h.spawns) == 1 and h.kills == [(424242, signal.SIGKILL)]
    assert h.waits[-1] == 5
    assert not (h.module.PROCESS / "completed.json").exists()
    receipt = json.loads((h.module.PROCESS / "failed.json").read_text())
    assert receipt["timeout"] is (mode == "timeout")
    assert receipt["exit_code"] == -signal.SIGKILL and receipt["no_retry"]
    assert receipt["logs"]["stderr.log"]["bytes"] == len(b"synthetic stderr\n")


@pytest.mark.parametrize("defect", ["nonzero", "missing", "failed_marker", "invalid_marker",
                                  "wrong_plan", "partial_fits", "partial_evaluations", "nan_wall", "overcap_wall"])
def test_exit_zero_alone_never_establishes_success(harness, defect):
    h = harness
    if defect == "nonzero":
        h.options["exit"] = 7
    elif defect == "missing":
        h.options["missing_completed"] = True
    elif defect in {"failed_marker", "invalid_marker"}:
        h.options[defect] = True
    else:
        change = {"wrong_plan": {"plan_sha256": "f" * 64}, "partial_fits": {"fits": 11},
                  "partial_evaluations": {"evaluations": 47}, "nan_wall": {"wall_seconds": float("nan")},
                  "overcap_wall": {"wall_seconds": 1800}}[defect]
        h.options["completion_override"] = change
    with pytest.raises((ValueError, FileNotFoundError)):
        h.launch()
    assert len(h.spawns) == 1 and not h.kills
    assert not (h.module.PROCESS / "completed.json").exists()
    assert (h.module.PROCESS / "failed.json").exists()


@pytest.mark.parametrize("failed_write", [False, True])
def test_late_outer_receipt_write_demotes_success_and_keeps_original_error(harness, monkeypatch, failed_write):
    h = harness
    original = h.module.write_json
    def write(path, value):
        if path == h.module.PROCESS / "failed.json" and failed_write:
            raise OSError("secondary receipt failure")
        original(path, value)
        if path == h.module.PROCESS / "completed.json":
            h.clock.value = 1800
    monkeypatch.setattr(h.module, "write_json", write)
    with pytest.raises(ValueError, match="during completion write") as caught:
        h.launch()
    assert not (h.module.PROCESS / "completed.json").exists()
    assert (h.module.PROCESS / "invalid-completion.json").exists()
    # Do not modify the original child's own successful receipt.
    assert (h.module.ATTEMPT / "completed.json").exists()
    if failed_write:
        assert any("secondary receipt failure" in note for note in caught.value.__notes__)
    else:
        receipt = json.loads((h.module.PROCESS / "failed.json").read_text())
        assert "during completion write" in receipt["error"]
        assert receipt["wall_seconds_including_cleanup"] == 1800


def test_pre_spawn_budget_exhaustion_never_forks_child(harness, monkeypatch):
    h = harness
    original = h.module.write_json
    def write(path, value):
        original(path, value)
        if path.name == "started.json":
            h.clock.value = 1800
    monkeypatch.setattr(h.module, "write_json", write)
    with pytest.raises(ValueError, match="before spawn"):
        h.launch()
    assert not h.spawns and not h.kills
    assert (h.module.PROCESS / "failed.json").exists()


def test_spawn_exhausting_budget_is_killed_without_first_wait(harness):
    h = harness
    h.options["spawn_seconds"] = 1800
    with pytest.raises(ValueError, match="during spawn"):
        h.launch()
    assert len(h.spawns) == 1 and h.waits == [5]
    assert h.kills == [(424242, signal.SIGKILL)]


@pytest.mark.parametrize("existing", ["PROCESS", "ATTEMPT"])
def test_exclusive_process_and_no_resume_preserve_existing_bytes(harness, existing):
    h = harness
    path = getattr(h.module, existing)
    path.mkdir()
    (path / "original.bin").write_bytes(b"retained original")
    with pytest.raises((ValueError, FileExistsError)):
        h.launch()
    assert not h.spawns
    assert (path / "original.bin").read_bytes() == b"retained original"


def test_wrong_plan_file_leaves_failure_receipt_without_launch(harness):
    h = harness
    h.module.PLAN.write_text("tampered")
    with pytest.raises(ValueError, match="plan file digest mismatch"):
        h.launch()
    assert not h.spawns
    assert json.loads((h.module.PROCESS / "failed.json").read_text())["phase"] == "admission"


def test_malformed_explicit_identities_refuse_before_any_output(harness):
    h = harness
    with pytest.raises(ValueError, match="SHA-256"):
        h.module.launch("not-a-digest", h.commit)
    with pytest.raises(ValueError, match="published commit"):
        h.module.launch(h.digest, "main")
    assert not h.module.PROCESS.exists() and not h.spawns
