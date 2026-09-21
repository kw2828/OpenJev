"""Separate runtime V2 admission, preserving the failed V1 resource decision.

Model restoration and inference are imported from the immutable V1 control.
This module owns only the new prospective plan and bounded process lifecycle.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import dialogue_calibration_common as legacy

from openjev.research.suspend_clock import Deadline, SuspendClock

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-calibration-runtime-v2"
SUPERVISOR = legacy.SUPERVISOR
CLOCK = legacy.CLOCK
NATIVE = legacy.NATIVE
FIT_ORDER = list(legacy.FIT_ORDER)
LIMITS = {
    "pilot": {"wall_seconds": 600, "rss_bytes": 8 * 1024**3,
              "mps_driver_bytes": 8 * 1024**3, "output_bytes": 512 * 1024**2},
    "infer": {"wall_seconds": 3600, "rss_bytes": 8 * 1024**3,
              "mps_driver_bytes": 8 * 1024**3, "output_bytes": 512 * 1024**2},
}
CONFIG = {"dev_dialogues": 2363, "sample_dialogues": 128, "estimator_dialogues": 64,
          "verification_dialogues": 64, "warmup_dialogues": 2, "calibration_dialogues": 512,
          "selection_salt": "openjev-calibration-runtime-v2:", "safety_factor": 2,
          "projection_admission_seconds": 1800, "all_twelve_verifications_required": True,
          "fit_order": FIT_ORDER, "model_weight_updates": 0, "official_test_opened": False,
          "geometry_report_is_diagnostic": True, "scientific_criteria_unchanged": 11}
SOURCES = {"research/dialogue-calibration-runtime-v2-protocol.md",
           "scripts/dialogue_runtime_common.py", "tests/test_dialogue_runtime_common.py",
           "scripts/dialogue_runtime_geometry.py", "tests/test_dialogue_runtime_geometry.py",
           "scripts/pilot_dialogue_runtime.py", "tests/test_pilot_dialogue_runtime.py",
           "scripts/infer_dialogue_runtime.py", "tests/test_infer_dialogue_runtime.py",
           "src/openjev/research/dialogue_runtime_projection.py", "tests/test_dialogue_runtime_projection.py"}
CONTROL_PLAN = "output/dialogue-calibration-control-v1/plan.json"
PREPARED = "runs/dialogue-calibration-control-v1/preparation-01"
PREP_TERMINAL = "output/dialogue-calibration-control-v1/preparation-process-01.terminal.json"
QUALIFIED = "runs/dialogue-calibration-control-v1/qualification-01"
QUAL_TERMINAL = "output/dialogue-calibration-control-v1/qualification-process-01.terminal.json"
PINS = {
    CONTROL_PLAN: "e706e40cd4a0d95fb1536ce105074c071b57a6bc4652ec57b9243672542a413c",
    PREPARED + "/completed.json": "896099f38f7316f23e549d1271d81f90a2884028e6c0281dcf3c035b6ee286e9",
    PREP_TERMINAL: "9a3f54de5ee09259fdfafa9e569556fc51ff78c3bf7ea8b29842e9944aa05bd3",
    QUALIFIED + "/completed.json": "ca06666f1849a49c3e230f8a1fa6f485f5214dbdbe36c9f7fccde1a2cf2e2a94",
    QUAL_TERMINAL: "b9ef0d20799f1e8a1fda7b565d57378a9bac7d7b3c53de5a48425d9bb6cf644b",
    SUPERVISOR: legacy.PINS[SUPERVISOR], CLOCK: legacy.PINS[CLOCK],
}
OUTPUTS = {phase: f"runs/{VERSION}/{name}-01" for phase, name in (("pilot", "pilot"), ("infer", "inference"))}


def expected_members(phase):
    require(phase in LIMITS, "Known runtime phase")
    if phase == "pilot":
        return {"started.json", "sample.json", "projection.json"} | {
            f"{fit}/{name}" for fit in FIT_ORDER
            for name in ("predictions.npz", "dialogues.jsonl", "forecast.json", "completed.json")}
    return {"started.json"} | {f"{fit}/{name}" for fit in FIT_ORDER
                               for name in ("predictions.npz", "dialogues.jsonl", "completed.json")}


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
    require(args.command in LIMITS and args.out.resolve() == (ROOT / OUTPUTS[args.command]).resolve(),
            "Fixed exclusive runtime phase output")
    historical = SimpleNamespace(plan=ROOT / CONTROL_PLAN, plan_sha256=PINS[CONTROL_PLAN],
                                 command="prepare", out=ROOT / PREPARED)
    ctx = legacy.authenticate(historical, budget)
    prepared = legacy.authenticate_output(ROOT / PREPARED, PINS[PREPARED + "/completed.json"], ctx, "prepare")
    prep_terminal = legacy.authenticate_terminal(ROOT / PREP_TERMINAL, PINS[PREP_TERMINAL], prepared)
    old = legacy.authenticate_output(ROOT / QUALIFIED, PINS[QUALIFIED + "/completed.json"], ctx, "qualify")
    old_terminal = legacy.authenticate_terminal(ROOT / QUAL_TERMINAL, PINS[QUAL_TERMINAL], old)
    require(old["projection"]["admitted"] is False and old["parity_passed"] is True
            and old["fit_order"] == FIT_ORDER and old["completed_forwards"] == 24
            and old["prepared_sha256"] == PINS[PREPARED + "/completed.json"]
            and old["prepared_terminal_sha256"] == PINS[PREP_TERMINAL],
            "V1 exact replay and failed admission stay unchanged")
    ctx.update(runtime_plan=plan, runtime_plan_sha256=args.plan_sha256, runtime_plan_path=args.plan.resolve(),
               prepared=ROOT / PREPARED, preparation_receipt=prepared, preparation_terminal=prep_terminal,
               old_qualification=old, old_qualification_terminal=old_terminal)
    if args.command == "infer":
        pilot = authenticate_output(ROOT / OUTPUTS["pilot"], args.pilot_sha256, ctx, "pilot")
        terminal = authenticate_terminal(args.pilot_terminal, args.pilot_terminal_sha256, pilot)
        authenticate_admission(pilot, ctx)
        ctx.update(pilot_receipt=pilot, pilot_terminal=terminal)
    budget.check()
    return ctx


def authenticate_admission(pilot, ctx):
    """Recompute the new gate from all paid fit records, without changing V1."""
    from dialogue_runtime_geometry import report_geometry

    from openjev.research.dialogue_runtime_projection import project_calibration, select_timing_sample

    directory = ROOT / OUTPUTS["pilot"]
    require(pilot["fit_order"] == FIT_ORDER and pilot["completed_forwards"] == 12 * 130
            and pilot["parity_passed"] is True, "Complete fixed runtime pilot replay")
    fits = pilot["fits"]
    require([fit["fit_id"] for fit in fits] == FIT_ORDER, "All twelve ordered timing records")
    sample = read(directory / "sample.json")
    profiles = [p for p in read(ctx["original_prepared"] / "workloads.json")["profiles"] if p["split"] == "dev"]
    calibration_profiles = read(ctx["prepared"] / "workloads.json")["profiles"]
    replay_ids = [r["dialogue_id"] for r in read(ctx["prepared"] / "replay-cases.json")["cases"]]
    expected_sample = select_timing_sample(profiles, replay_ids)
    expected_sample["geometry"] = report_geometry(profiles, calibration_profiles, expected_sample)
    require(sample == expected_sample and sha(directory / "sample.json") == pilot["sample_sha256"],
            "Recomputed public-only fixed timing sample and geometry")
    calibration_work = {"dialogue_count": len(calibration_profiles),
                        **{key: sum(p["work"][key] for p in calibration_profiles) for key in
                           ("encoder_calls", "padded_attention_positions", "real_question_updates")}}
    require(calibration_work == pilot["calibration_work"], "Unchanged complete calibration work")
    previous_end = 0
    for fit in fits:
        fit_dir = directory / fit["fit_id"]
        require(sha(fit_dir / "completed.json") == fit["completed_sha256"], "Pilot nested completion binding")
        require(read(fit_dir / "completed.json") == {k: v for k, v in fit.items() if k != "completed_sha256"},
                "Pilot nested receipt identity")
        manifest(fit_dir, fit["files"])
        forecast = read(fit_dir / "forecast.json")
        require(sha(fit_dir / "forecast.json") == fit["forecast_sha256"]
                and forecast["sample_sha256"] == sha(directory / "sample.json")
                and forecast["fit_id"] == fit["fit_id"] and forecast["verification_forwards_started"] == 0
                and forecast["prediction"] == fit["verification_result"]["prediction"]
                and fit["estimator"]["finished_elapsed_seconds"] <= forecast["created_elapsed_seconds"]
                <= fit["forecast_published_elapsed_seconds"] <= fit["verification"]["started_elapsed_seconds"],
                "Forecast frozen before verification")
        require(forecast["prediction"]["estimator_work"] == sample["estimator"]["work"]
                and forecast["prediction"]["verification_work"] == sample["verification"]["work"]
                and forecast["prediction"]["estimator_seconds"] == fit["estimator"]["seconds"]
                and fit["verification_result"]["actual_seconds"] == fit["verification"]["seconds"],
                "Forecast and verification use actual complete block durations")
        require(fit["warmup"]["dialogue_ids"] == sample["warmup_ids"]
                and fit["witness"]["restored_sha256"] == fit["witness"]["final_sha256"],
                "Both paid warmups and unchanged model weights")
        for block in ("warmup", "estimator", "verification"):
            measured = fit[block]
            require(measured["seconds"] == measured["finished_elapsed_seconds"] - measured["started_elapsed_seconds"]
                    and 0 <= previous_end <= measured["started_elapsed_seconds"] < measured["finished_elapsed_seconds"]
                    <= pilot["projection_elapsed_seconds"], "Ordered disjoint complete timing intervals")
            previous_end = measured["finished_elapsed_seconds"]
        for block in ("estimator", "verification"):
            require(fit[block]["work"] == sample[block]["work"]
                    and fit[block]["dialogue_ids"] == sample[block]["dialogue_ids"], "Fixed timing cohort and geometry")
    disjoint = math.fsum(f["loading_seconds"] + math.fsum(f[b]["seconds"] for b in
                           ("warmup", "estimator", "verification")) for f in fits)
    require(disjoint == pilot["disjoint_measured_seconds"]
            and pilot["pilot_nonblock_overhead_seconds"] == pilot["projection_elapsed_seconds"] - disjoint
            and pilot["projection_elapsed_seconds"] <= pilot["wall_seconds"], "Paid complete overhead snapshot")
    result = project_calibration(
        {f["fit_id"]: f["verification_result"] for f in fits}, pilot["calibration_work"],
        loading_seconds_by_fit={f["fit_id"]: f["loading_seconds"] for f in fits},
        warmup_seconds_by_fit={f["fit_id"]: f["warmup"]["seconds"] for f in fits},
        preparation_parent_wall_seconds=ctx["preparation_terminal"]["wall_seconds"],
        pilot_nonblock_overhead_seconds=pilot["pilot_nonblock_overhead_seconds"])
    require(result == pilot["projection"] == read(directory / "projection.json") and result["admitted"] is True,
            "All twelve forecasts pass and new full projection fits 1800 seconds")
    require(pilot["model_weight_updates"] == 0 and pilot["optimizer_created"] is False
            and pilot["temperature_applied"] is False and pilot["official_test_opened"] is False
            and pilot["task_metrics_computed"] is False, "Pilot never substitutes scientific evidence")
    return result


def authenticate_output(path, pin, ctx, phase):
    path = Path(path)
    require(path.resolve() == (ROOT / ctx["runtime_plan"]["outputs"][phase]).resolve(), "Prior phase output identity")
    require(sha(path / "completed.json") == pin, "Prior phase completion pin")
    receipt = read(path / "completed.json")
    require(receipt["version"] == VERSION and receipt["status"] == "completed" and receipt["phase"] == phase
            and receipt["plan_sha256"] == ctx["runtime_plan_sha256"]
            and receipt["sources"] == ctx["runtime_plan"]["sources"]
            and receipt["inputs"] == ctx["runtime_plan"]["inputs"], "Successful matching prior phase")
    require(set(receipt["files"]) == expected_members(phase), "Exact required prior phase payloads")
    manifest(path, receipt["files"])
    started = read(path / "started.json")
    require(started["version"] == VERSION and started["phase"] == phase
            and started["request"] == receipt["request"]
            and started["limits"] == receipt["limits"] == LIMITS[phase]
            and started["supervision"] == receipt["supervision"]
            and started["worker_started_ns"] == receipt["timing"]["worker_started_ns"]
            and receipt["request"]["command"] == phase
            and receipt["request"]["plan_sha256"] == ctx["runtime_plan_sha256"]
            and Path(receipt["request"]["plan"]).resolve() == ctx["runtime_plan_path"]
            and Path(receipt["request"]["out"]).resolve() == path.resolve()
            and Path(receipt["request"]["supervision"]).resolve() == Path(receipt["supervision"]["path"]),
            "Original start/request/resource/supervision binding")
    require(receipt["environment"] == ctx["runtime_plan"]["environment"], "Prior phase runtime")
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
    script = "pilot_dialogue_runtime.py" if phase == "pilot" else "infer_dialogue_runtime.py"
    require(Path(tail[0]).resolve() == ROOT / "scripts" / script, "Phase worker source")
    argv_flags = tail[1:]
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
        members(args.out)  # Charge payload hashing before the optional final cost snapshot.
        if hasattr(body, "finalize"):
            metadata = body.finalize(args, ctx, budget, metadata)
        files = members(args.out)
        require(set(files) == expected_members(args.command), "Exact required phase payloads before completion")
        budget.check()
        finished = clock.now_ns()
        receipt = {**metadata, "version": VERSION, "status": "completed", "phase": args.command,
                   "request": request, "limits": budget.limits, "environment": ctx["runtime_plan"]["environment"],
                   "plan_sha256": args.plan_sha256, "sources": ctx["runtime_plan"]["sources"],
                   "inputs": ctx["runtime_plan"]["inputs"], "files": files,
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
