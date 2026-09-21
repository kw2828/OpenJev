"""Pinned input authentication and supervised lifecycle for calibration control.

This module performs no corpus decoding or model loading. Each phase uses an
exclusive output directory and the same native deadline as its parent process.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import signal
import sys
import time
from pathlib import Path

from openjev.research.suspend_clock import Deadline, SuspendClock

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-calibration-control-v1"
BASE = "output/dialogue-probability-diagnostic-v1"
DATA = "runs/sgd-state-v1/data"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
CLOCK = "src/openjev/research/suspend_clock.py"
NATIVE = {"mach_continuous_time", "CLOCK_BOOTTIME"}
LIMITS = {
    "prepare": {"wall_seconds": 300, "rss_bytes": 8 * 1024**3, "output_bytes": 512 * 1024**2},
    "qualify": {"wall_seconds": 600, "rss_bytes": 8 * 1024**3, "output_bytes": 256 * 1024**2,
                "mps_driver_bytes": 8 * 1024**3},
    "infer": {"wall_seconds": 3600, "rss_bytes": 8 * 1024**3, "output_bytes": 512 * 1024**2,
              "mps_driver_bytes": 8 * 1024**3},
}
CONFIG = {"dialogues": 512, "selection_salt": "openjev-calibration-v1:",
          "seeds": [6901, 6902, 6903],
          "arms": ["frozen_original", "frozen_numbers", "trainable_original", "trainable_numbers"],
          "replay_dialogues_per_fit": 2, "projection_admission_seconds": 1800,
          "supported_log_tolerance": 1e-5, "probability_tolerance": 1e-6,
          "mass_tolerance": 2e-6, "beta_bounds": [0.125, 8.0], "bisection_steps": 64,
          "model_weight_updates": 0, "official_test_opened": False}
SOURCES = {
    "research/dialogue-calibration-control-protocol.md",
    "scripts/dialogue_calibration_common.py", "tests/test_dialogue_calibration_common.py",
    "scripts/prepare_dialogue_calibration.py", "tests/test_prepare_dialogue_calibration.py",
    "scripts/replay_dialogue_calibration.py", "tests/test_replay_dialogue_calibration.py",
    "src/openjev/research/dialogue_calibration_data.py", "tests/test_dialogue_calibration_data.py",
    "src/openjev/research/dialogue_calibration_inference.py", "tests/test_dialogue_calibration_inference.py",
    "src/openjev/research/dialogue_temperature.py", "tests/test_dialogue_temperature.py",
}
PINS = {
    "scripts/diagnose_dialogue_probabilities.py": "a99c68b2101f299d1118bb9c69f568578d46519ec6f0387b18576263a7f8d78f",
    f"{BASE}/plan.json": "b251d8d63713f2908f79fc327b90aa3e52c1e5108c8fc44c07f32b90bff04d36",
    f"{BASE}/run-01/receipt.json": "a1d2e52a2c00a67e477378dee5e4e50b0279136b474841fcc5d4102f70d0b321",
    f"{BASE}/run-01/summary.json": "7cdd2533910826e95760e3d0144b2163060a85c05702b4413cafda857c13260c",
    f"{BASE}/actual-exit-01.json": "42d0fe4e863c7a6b0f435b20e1e7f26822015aedf9cde21f7c640c2fbc8a5b00",
    f"{BASE}/independent-audit-01/result-01/receipt.json": "3c2f26224726ab362334de70b082f2855ce134825236cc1aa4a6dc937788948f",
    f"{BASE}/independent-audit-01/result-01/summary.json": "721b2b70675061e21f0981a33e595588fd24ecec171196c48546739497e0f5ed",
    f"{BASE}/independent-audit-01/actual-exit.json": "182043e60ea9bbdfe94a30e77683ccfc06650c51c0c476ce6c794fefc9685df5",
    f"{DATA}/completed.json": "677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12",
    SUPERVISOR: "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144",
    CLOCK: "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124",
}
OUTPUTS = {phase: f"runs/{VERSION}/{name}-01" for phase, name in
           (("prepare", "preparation"), ("qualify", "qualification"), ("infer", "inference"))}
FIT_ORDER = [f"{arm}-{seed}" for seed in CONFIG["seeds"] for arm in CONFIG["arms"]]


def expected_members(phase):
    """Exact success payloads; partial arrays may survive only failed attempts."""
    if phase == "prepare":
        return {"started.json", "selection.json", "packet.json", "workloads.json", "replay-cases.json",
                "service-exposure.json", "actors.jsonl", "targets.jsonl", "evaluation-rows.jsonl",
                "lexical-original.npy", "lexical-numbers.npy"}
    require(phase in ("qualify", "infer"), "Known output phase")
    files = {"started.json"}
    if phase == "qualify":
        files.add("projection.json")
        fit_files = ("case-00.npz", "case-01.npz", "cases.jsonl", "completed.json")
    else:
        fit_files = ("predictions.npz", "dialogues.jsonl", "completed.json")
    return files | {f"{fit}/{name}" for fit in FIT_ORDER for name in fit_files}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_lines(path, rows):
    with Path(path).open("x") as stream:
        stream.writelines(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: importlib.metadata.version(name) for name in
               ("numpy", "torch", "transformers", "tokenizers")}}


def bind(mapping, check=lambda: None):
    for name, pin in mapping.items():
        check()
        require(sha(ROOT / name) == pin, "Pinned file changed: " + name)


def members(directory, *, exclude=()):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), "Real owned output directory")
    result = {}
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "No symbolic links in owned output")
        if path.is_file():
            name = path.relative_to(directory).as_posix()
            if name not in exclude:
                result[name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    return result


def manifest(directory, files, receipt="completed.json"):
    for name, value in files.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts and name != receipt,
                "Safe relative manifest member")
        require(set(value) == {"bytes", "sha256"}, "Exact file descriptor")
    require(members(directory, exclude=(receipt,)) == files, "Exact payload membership and hashes")


def validate_plan(path, pin, check=lambda: None):
    require(sha(path) == pin, "Prospective control plan hash")
    plan = read(path)
    require(plan["version"] == VERSION and plan["config"] == CONFIG
            and plan["limits"] == LIMITS and plan["outputs"] == OUTPUTS, "Fixed calibration recipe")
    require(set(plan["sources"]) == SOURCES and plan["inputs"] == PINS, "Exact source/input closure")
    require(plan["environment"] == environment(), "Frozen runtime versions")
    bind(plan["sources"], check)
    bind(plan["inputs"], check)
    qualification = plan["synthetic_qualification"]
    require(set(qualification) == {"path", "sha256"}, "Synthetic qualification binding")
    require(sha(ROOT / qualification["path"]) == qualification["sha256"], "Qualification receipt pin")
    qualified = read(ROOT / qualification["path"])
    require(qualified["status"] == "completed" and qualified["actual_exit_code"] == 0
            and qualified["source_sha256"] == plan["sources"] and qualified["model_calls"] == 0
            and qualified["real_task_inputs_read"] is False, "Matching completed synthetic qualification")
    require(sha(path) == pin, "Plan remained stable")
    return plan


def authenticate(args, budget):
    plan = validate_plan(args.plan, args.plan_sha256, budget.check)
    require(args.out.resolve() == (ROOT / plan["outputs"][args.command]).resolve(), "Fixed exclusive phase output")
    # The pinned qualified reader authenticates the entire frozen V2 run and
    # its independent audit before any corpus records are decoded here.
    import diagnose_dialogue_probabilities as diagnostic

    auditor, prior, science, _ = diagnostic.authenticate(read(ROOT / BASE / "plan.json"))
    for directory in (ROOT / BASE / "run-01", ROOT / BASE / "independent-audit-01/result-01"):
        receipt = read(directory / "receipt.json")
        require(receipt["status"] == "completed" and receipt["model_calls"] == 0
                and receipt["plan_sha256"] == PINS[f"{BASE}/plan.json"], "Complete saved diagnostic identity")
        manifest(directory, receipt["files"], "receipt.json")
    independent = read(ROOT / BASE / "independent-audit-01/result-01/receipt.json")
    require(independent["agreement"] is True
            and independent["producer_receipt_sha256"] == PINS[f"{BASE}/run-01/receipt.json"]
            and independent["producer_summary_sha256"] == PINS[f"{BASE}/run-01/summary.json"],
            "Independent diagnostic agreement binding")
    terminal = read(ROOT / BASE / "independent-audit-01/actual-exit.json")
    require(terminal["status"] == "completed" and terminal["actual_exit_code"] == 0
            and terminal["independent_agreement"] is True
            and terminal["receipt_sha256"] == PINS[f"{BASE}/independent-audit-01/result-01/receipt.json"],
            "Actual independent diagnostic exit")
    producer_exit = read(ROOT / BASE / "actual-exit-01.json")
    require(producer_exit["status"] == "completed" and producer_exit["actual_exit_code"] == 0
            and producer_exit["receipt_sha256"] == PINS[f"{BASE}/run-01/receipt.json"],
            "Actual producer diagnostic exit")
    import qualify_dialogue_observation_batches as original

    prepared = Path(science["prepared"])
    parent = original.authenticate_prepared(prepared, science["prepared_completed_sha256"])
    source = read(ROOT / DATA / "completed.json")
    require(source["status"] == "completed" and source["study"] == "sgd-state-v1"
            and source["test_contents_accessed"] is False and source["new_model_calls"] == 0,
            "Completed TRAIN/DEV source preparation")
    manifest(ROOT / DATA, source["files"])
    for name, pin in source["implementation_sha256"].items():
        require(source["files"]["implementation/" + name]["sha256"] == pin, "Source implementation binding")
    budget.check()
    return {"original_prepared": prepared, "parent": parent, "prior": prior, "science": science,
            "auditor": auditor, "plan": plan, "plan_sha256": args.plan_sha256,
            "plan_path": args.plan.resolve(), "progress": budget.progress}


def authenticate_output(path, pin, ctx, phase):
    path = Path(path)
    require(path.resolve() == (ROOT / ctx["plan"]["outputs"][phase]).resolve(), "Prior phase output identity")
    require(sha(path / "completed.json") == pin, "Prior phase completion pin")
    receipt = read(path / "completed.json")
    require(receipt["version"] == VERSION and receipt["status"] == "completed" and receipt["phase"] == phase
            and receipt["plan_sha256"] == ctx["plan_sha256"]
            and receipt["sources"] == ctx["plan"]["sources"]
            and receipt["inputs"] == ctx["plan"]["inputs"], "Successful matching prior phase")
    require(set(receipt["files"]) == expected_members(phase), "Exact required prior phase payloads")
    manifest(path, receipt["files"])
    started = read(path / "started.json")
    require(started["version"] == VERSION and started["phase"] == phase
            and started["request"] == receipt["request"]
            and started["limits"] == receipt["limits"] == LIMITS[phase]
            and started["supervision"] == receipt["supervision"]
            and started["worker_started_ns"] == receipt["timing"]["worker_started_ns"]
            and receipt["request"]["command"] == phase
            and receipt["request"]["plan_sha256"] == ctx["plan_sha256"]
            and Path(receipt["request"]["plan"]).resolve() == ctx["plan_path"]
            and Path(receipt["request"]["out"]).resolve() == path.resolve()
            and Path(receipt["request"]["supervision"]).resolve() == Path(receipt["supervision"]["path"]),
            "Original start/request/resource/supervision binding")
    require(receipt["environment"] == ctx["plan"]["environment"], "Prior phase runtime")
    limits = LIMITS[phase]
    require(type(receipt["peak_rss_bytes"]) is int and 0 < receipt["peak_rss_bytes"] <= limits["rss_bytes"]
            and sum(v["bytes"] for v in receipt["files"].values()) + (path / "completed.json").stat().st_size
            <= limits["output_bytes"], "Saved phase RSS and complete output cap")
    for key in ("sampled_mps_driver_max_bytes", "sampled_mps_current_max_bytes"):
        require(type(receipt[key]) is int and 0 <= receipt[key] <= limits.get("mps_driver_bytes", 0),
                "Saved sampled MPS cap")
    return receipt


def authenticate_terminal(path, pin, output_receipt):
    require(sha(path) == pin, "Prior parent terminal pin")
    terminal = read(path)
    launch_path = Path(output_receipt["supervision"]["path"])
    require(sha(launch_path) == output_receipt["supervision"]["sha256"], "Prior parent launch pin")
    launch = read(launch_path)
    require(all(terminal.get(k) == v for k, v in launch.items()), "Exact prior parent launch/terminal join")
    command_request(launch["command"], output_receipt["request"])
    require(Path(launch["cwd"]).resolve() == ROOT and launch["pid"] == launch["pgid"]
            and all(type(launch[k]) is int and launch[k] > 0 for k in ("pid", "pgid", "parent_pid")),
            "Prior repository and dedicated child group")
    timing = output_receipt["timing"]
    require(timing["clock_backend"] == launch["clock_backend"]
            and timing["parent_started_ns"] == launch["started_ns"]
            and timing["started_ns"] == launch["started_ns"]
            and timing["deadline_ns"] == launch["deadline_ns"]
            and launch["clock_backend"] in NATIVE
            and launch["cap_seconds"] == LIMITS[output_receipt["phase"]]["wall_seconds"]
            and launch["deadline_ns"] - launch["started_ns"] == launch["cap_seconds"] * 1_000_000_000,
            "Prior worker and parent clock/deadline identity")
    require(timing["elapsed_ns"] == timing["finished_ns"] - timing["started_ns"]
            and timing["wall_seconds"] == timing["elapsed_ns"] / 1e9 == output_receipt["wall_seconds"]
            and terminal["elapsed_ns"] == terminal["finished_ns"] - terminal["started_ns"]
            and terminal["wall_seconds"] == terminal["elapsed_ns"] / 1e9
            and launch["started_ns"] <= timing["worker_started_ns"] <= timing["finished_ns"]
            and launch["version"] == "dialogue-observation-supervision-v2"
            and launch["watchdog_sha256"] == PINS[SUPERVISOR]
            and launch["clock_source_sha256"] == PINS[CLOCK], "Prior elapsed arithmetic and source identity")
    require(terminal["status"] == "completed" and terminal["returncode"] == 0
            and terminal["timing_available"] is True and terminal["timed_out"] is False
            and terminal["error"] is None and terminal["clock_error"] is None
            and terminal["group_absent"] is True and terminal["cleanup"]["reaped"] is True
            and terminal["cleanup"]["group_absent"] is True
            and not terminal["cleanup"]["errors"]
            and terminal["started_ns"] <= output_receipt["timing"]["started_ns"]
            <= output_receipt["timing"]["finished_ns"] <= terminal["finished_ns"] < terminal["deadline_ns"],
            "Successful actual parent exit within shared deadline")
    return terminal


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


class Budget:
    def __init__(self, out, deadline, limits):
        self.out, self.deadline, self.limits = out, deadline, limits
        self.max_driver = self.max_current = 0
        self.progress = {"operation": "authentication"}

    def check(self):
        self.deadline.check()
        require(peak_rss() <= self.limits["rss_bytes"], "Process RSS cap")

    def elapsed(self):
        self.check()
        return self.deadline.elapsed_ns() / 1e9

    def storage(self):
        self.check()
        require(sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
                <= self.limits["output_bytes"], "Owned output cap")

    def sync(self, torch):
        torch.mps.synchronize()
        self.max_driver = max(self.max_driver, torch.mps.driver_allocated_memory())
        self.max_current = max(self.max_current, torch.mps.current_allocated_memory())
        require(self.max_driver <= self.limits["mps_driver_bytes"], "Sampled MPS driver cap")
        self.check()


def command_request(command, request):
    """Bind every parsed option to the live or authenticated saved command."""
    require(type(command) is list and command and all(type(v) is str for v in command), "Supervisor command")
    tail = command[1:]
    if tail and tail[0] == "-u":
        tail = tail[1:]
    require(Path(command[0]).resolve() == Path(sys.executable).resolve(), "Worker interpreter identity")
    phase = request["command"]
    require(phase in LIMITS and bool(tail), "Known worker phase")
    script = "prepare_dialogue_calibration.py" if phase == "prepare" else "replay_dialogue_calibration.py"
    require(Path(tail[0]).resolve() == ROOT / "scripts" / script, "Phase worker source")
    argv_flags = tail[1:] if phase == "prepare" else tail[2:]
    require(phase == "prepare" or len(tail) > 1 and tail[1] == phase, "Live phase argument")
    require(len(argv_flags) % 2 == 0, "Paired live CLI flags")
    flags = dict(zip(argv_flags[::2], argv_flags[1::2], strict=True))
    expected = {"--" + key.replace("_", "-"): str(value) for key, value in request.items()
                if key != "command" and value is not None}
    require(len(flags) * 2 == len(argv_flags) and flags == expected, "Exact live parsed request and output")
    return tail


def validate_supervision(args, launch, clock, worker_started_ns):
    require(launch["version"] == "dialogue-observation-supervision-v2", "Qualified supervisor version")
    tail = command_request(launch["command"], vars(args))
    require(tail == sys.argv, "Exact live worker interpreter and argv")
    require(Path(launch["cwd"]).resolve() == Path.cwd().resolve() == ROOT, "Supervisor/worker repository")
    require(all(type(launch[k]) is int and launch[k] > 0 for k in ("pid", "pgid", "parent_pid"))
            and launch["pid"] == os.getpid() and launch["pgid"] == os.getpgrp()
            and launch["pid"] == launch["pgid"] and launch["parent_pid"] == os.getppid(),
            "Live parent and dedicated worker group")
    require(clock.backend in NATIVE and launch["clock_backend"] == clock.backend, "Same native continuous clock")
    cap = LIMITS[args.command]["wall_seconds"]
    require(type(launch["started_ns"]) is int and type(launch["deadline_ns"]) is int
            and 0 <= launch["started_ns"] <= worker_started_ns < launch["deadline_ns"]
            and type(launch["cap_seconds"]) is int and launch["cap_seconds"] == cap
            and launch["deadline_ns"] - launch["started_ns"] == cap * 1_000_000_000,
            "Exact parent absolute deadline and allocation")
    require(launch["watchdog_sha256"] == PINS[SUPERVISOR] == sha(ROOT / SUPERVISOR)
            and launch["clock_source_sha256"] == PINS[CLOCK] == sha(ROOT / CLOCK), "Supervisor/clock identity")
    deadline = Deadline(clock, launch["started_ns"], launch["deadline_ns"])
    deadline.check()
    return deadline


def await_supervision(args, clock, worker_started_ns):
    wait = Deadline(clock, worker_started_ns, worker_started_ns + 5_000_000_000)
    while True:
        wait.check()
        if args.supervision.exists():
            break
        time.sleep(.025)
    require(args.supervision.stat().st_size <= 65536, "Bounded parent launch record")
    pin, launch = sha(args.supervision), read(args.supervision)
    require(sha(args.supervision) == pin, "Stable parent launch publication")
    return validate_supervision(args, launch, clock, worker_started_ns), pin


def execute(args, body):
    args.out.mkdir(parents=True, exist_ok=False)
    request = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    clock = deadline = budget = handler = None
    worker_start = None
    try:
        clock = SuspendClock()
        worker_start = clock.now_ns()
        deadline, supervision_pin = await_supervision(args, clock, worker_start)
        budget = Budget(args.out, deadline, LIMITS[args.command])

        def timeout(*_):
            raise TimeoutError("Supplementary worker alarm expired")

        handler = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, deadline.remaining_ns() / 1e9)
        write(args.out / "started.json", {"version": VERSION, "phase": args.command, "request": request,
              "worker_started_ns": worker_start, "limits": budget.limits,
              "supervision": {"path": str(args.supervision.resolve()), "sha256": supervision_pin}})
        ctx = authenticate(args, budget)
        budget.progress["operation"] = "phase-body"
        metadata = body(args, ctx, budget)
        # Authenticate the original complete closure again; body code cannot
        # quietly replace a checkpoint, frozen source, input or model asset.
        authenticate(args, budget)
        require(sha(args.supervision) == supervision_pin, "Parent launch remained stable")
        budget.storage()
        files = members(args.out)
        require(set(files) == expected_members(args.command), "Exact required phase payloads before completion")
        budget.check()
        finished = clock.now_ns()
        receipt = {**metadata, "version": VERSION, "status": "completed", "phase": args.command,
                   "request": request, "limits": budget.limits, "environment": ctx["plan"]["environment"],
                   "plan_sha256": args.plan_sha256, "sources": ctx["plan"]["sources"],
                   "inputs": ctx["plan"]["inputs"], "files": files,
                   "supervision": {"path": str(args.supervision.resolve()), "sha256": supervision_pin},
                   "timing": {"clock_backend": clock.backend, "started_ns": deadline.started_ns,
                              "worker_started_ns": worker_start, "elapsed_ns": finished - deadline.started_ns,
                              "parent_started_ns": deadline.started_ns, "deadline_ns": deadline.expires_ns,
                              "finished_ns": finished, "wall_seconds": (finished - deadline.started_ns) / 1e9},
                   "wall_seconds": (finished - deadline.started_ns) / 1e9,
                   "peak_rss_bytes": peak_rss(), "sampled_mps_driver_max_bytes": budget.max_driver,
                   "sampled_mps_current_max_bytes": budget.max_current,
                   "timing_scope": "Parent start through pre-receipt reading; publication rechecked before return. "
                                   "Parent terminal must independently witness exit and final elapsed cost."}
        write(args.out / "completed.json", receipt)
        budget.storage()
        print(json.dumps({"status": "completed", "phase": args.command,
                          "receipt_sha256": sha(args.out / "completed.json"),
                          "wall_seconds": budget.elapsed()}, allow_nan=False), flush=True)
        budget.check()
        return receipt
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (args.out / "completed.json").exists():
                (args.out / "completed.json").rename(args.out / "invalid-completion.json")
            write(args.out / "failed.json", {"version": VERSION, "status": "failed", "request": request,
                  "error_type": type(error).__name__, "error": str(error), "peak_rss_bytes": peak_rss(),
                  "progress": None if budget is None else budget.progress,
                  "scope": "Failed attempt preserved; no automatic retry or scientific result."})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original phase failure
            error.add_note("Failure receipt error: " + repr(secondary))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if handler is not None:
            signal.signal(signal.SIGALRM, handler)
