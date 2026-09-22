"""Fixed six-anchor teacher-cost feasibility collection, without model/native calls.

Plan mode authenticates closed historical evidence and selects public metadata.
Run mode reconstructs only the frozen learner-TRAIN prefixes, validates all six
anchors, and invokes the unchanged sampler. Completion is technical only and
requires the original successful supervisor plus a later independent saved audit.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import os
import platform
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-teacher-label-pilot-v1"
SCRIPT = "scripts/collect_otto_teacher_labels.py"
PROTOCOL = "research/otto-teacher-label-pilot-protocol.md"
ANCHORS = "src/openjev/research/otto_teacher_anchors.py"
CLOCK = "src/openjev/research/suspend_clock.py"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SYMM = "output/otto-symmetry-head-v1"
QUAL = "output/otto-teacher-sampler-qualification-v1"
PINS = {
    f"{SYMM}/plan-01.json": "ae4475170f3eb8211b49c480871d1b534f201b7240cb579f099e01efee2727be",
    f"{SYMM}/run-01/receipt.json": "afd3ed59c6fc907721ebebcff47c1342fc53e592ec14b190497c647cb8d26c62",
    f"{SYMM}/run-process-01.terminal.json": "a5bd0e90b12df04d1bb8cb0a677c072c6d67f70b67c2892c66d6158cd87ccf90",
    f"{SYMM}/audit-01/receipt.json": "43ec2e8dc4016a9ec857b7ee3cf9dfb89e1107042c9fbfbfb4d1026d0d7ba6bb",
    f"{QUAL}/plan-01.json": "7dd2f4a795a1772b68973d0fa2e507d97efe9ca4d7b18b2d663e7b889a909a4d",
    f"{QUAL}/run-01/receipt.json": "859bfb303daea27d187670b0e3e452855f2fd3be7ce5824c371a0526879268b6",
    f"{QUAL}/supervision-01.terminal.json": "785df84e5a9839903cd50f9b8b3e7d8a1a14ed21ec1c48e6230e41c723cb0c51",
    f"{QUAL}/independent-review-01.json": "e1a1974f4e58b7844bc11d4e90549b9b14a09cadb62df9e1c6ef9b7febdd35cd",
}
PILOT = "output/otto-teacher-label-pilot-v1"
BUDGET_PROGRAM = "scripts/budget_otto_teacher_labels.py"
PREPARED_SOURCES = {
    ANCHORS: "88d0793089111ac4aed0f3571a22486a0d6fbfd6b1354356632f93bdd756d39d",
    "tests/test_otto_teacher_anchors.py": "162b01b6e54fb048409af55ff0e179a88a650ed5991aec31fffc5c189157c816",
    BUDGET_PROGRAM: "950400bf4aa924da0239fd4365546d25545b8d8cf809107040b62df6b85c1dc9",
}
PREPARED_INPUTS = {
    f"{PILOT}/serialization-01.json": "cd31c90bb54ade51e7a5f149e50f6fc927dc93d1890470e6bd35f67118a4bf6d",
    f"{PILOT}/engineering-01/receipt.json": "6dea86b78940290bfc19fe80d3ab530899a85aec06c2e802892ea5ef7d477fdb",
    f"{PILOT}/engineering-02/receipt.json": "d6f4db2066ba0373b70acee9bee5d599a81df43e8458c807e7ae0fd88b008a57",
}
NEW_SOURCES = (SCRIPT, PROTOCOL, *PREPARED_SOURCES, "tests/test_collect_otto_teacher_labels.py")
ENVIRONMENT = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")
LIMITS = {"native_seconds": 1800, "rss_bytes": 4 * 1024**3, "output_bytes": 4 * 1024**3}
SEED, HORIZON, REPLICATES = 19000001, 2188, tuple(range(16))
EVENT_BYTES, MAX_EVENTS, NON_EVENT_RESERVE = 512, 6723474, 256 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(path):
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), "absolute nonsymlink path")
    return path


def relative(name):
    p = Path(name)
    require(not p.is_absolute() and ".." not in p.parts, "safe relative evidence path")
    return regular(ROOT / p)


def digest(path, check=None):
    regular(path)
    h, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            if check:
                check()
            h.update(chunk)
            size += len(chunk)
    return {"sha256": h.hexdigest(), "bytes": size}


def read(path):
    return json.loads(path.read_text())


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


def write(path, value):
    with path.open("xb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def projection():
    return {"maximum_continuations": 384, "maximum_moves": 840192,
            "maximum_operation_pairs": 3361350, "maximum_sampler_events": MAX_EVENTS,
            "hard_event_bytes": EVENT_BYTES, "non_event_reserve_bytes": NON_EVENT_RESERVE,
            "projected_bytes": MAX_EVENTS * EVENT_BYTES + NON_EVENT_RESERVE,
            "serializer": "json.dumps(sort_keys=True,allow_nan=False,separators=(',',':'))+'\\n'",
            "float_character_allowance": 32, "maximum_calculated_event_bytes": 475,
            "scope": "Unmodified sampler events, each charged512 bytes; no duplicated draw vectors."}


def configuration():
    return {"seed": SEED, "anchor_ids": list(range(6)), "replicate_ids": list(REPLICATES),
            "horizon": HORIZON, "first_actions": "all geometric inbounds, ascending",
            "familywise_alpha": .05, "no_retries": True, "selection": "fixed six protocol cells; lexicographic(seed,prefix_index,row_index)",
            "native_calls": 0, "learned_model_calls": 0, "training_calls": 0}


def authenticate_history(check=None):
    """Hash complete closures; decode only plans, receipts and launch metadata."""
    inputs, sources = {}, {}

    def bind(name, expected=None):
        actual = digest(relative(name), check)
        require(expected is None or actual == expected, f"evidence identity: {name}")
        inputs[name] = actual
        return actual

    def closed(folder, receipt, count):
        require(len(receipt["files"]) == count and {p.name for p in relative(folder).iterdir()}
                == set(receipt["files"]) | {"receipt.json"}, "exact historical payload closure")
        for name, desc in receipt["files"].items():
            require(Path(name).name == name, "flat historical payload")
            bind(f"{folder}/{name}", desc)

    for name, pin in PINS.items():
        require(bind(name)["sha256"] == pin, "external historical root pin")
    for stem, prefix, source_count, payload_count in ((SYMM, "run-process-01", 121, 41), (QUAL, "supervision-01", 17, 417)):
        plan, worker = read(relative(f"{stem}/plan-01.json")), read(relative(f"{stem}/run-01/receipt.json"))
        terminal = read(relative(f"{stem}/{prefix}.terminal.json"))
        launch_name = f"{stem}/{prefix}.launch.json"
        require(bind(launch_name)["sha256"] == worker["supervision_sha256"], "original launch pin")
        launch = read(relative(launch_name))
        require(all(terminal[k] == v for k, v in launch.items()), "original parent launch/terminal join")
        require(terminal["status"] == "completed" and terminal["returncode"] == 0 and not terminal["timed_out"]
                and terminal["group_absent"] is True and terminal["cleanup"]["reaped"] is True
                and terminal["cleanup"]["group_absent"] is True and terminal["cleanup"]["errors"] == []
                and terminal["error"] is None and terminal["clock_error"] is None and terminal["timing_available"] is True,
                "original successful supervisor")
        require(worker["status"] == "completed" and worker["sources"] == plan["sources"]
                and worker["inputs"] == plan["inputs"] and worker["limits"] == plan["limits"]
                and worker["plan_sha256"] == PINS[f"{stem}/plan-01.json"]
                and len(plan["sources"]) == source_count, "closed prospective source/input identity")
        require(launch["cwd"] == str(ROOT) and launch["cap_seconds"] == plan["limits"]["native_seconds"]
                and launch["deadline_ns"] == launch["started_ns"] + launch["cap_seconds"] * 10**9
                and launch["started_ns"] <= worker["started_ns"] <= worker["finished_ns"] <= terminal["finished_ns"] < launch["deadline_ns"]
                and worker["peak_rss_bytes"] <= plan["limits"]["rss_bytes"], "historical bounded elapsed resources")
        require(terminal["version"] == "dialogue-observation-supervision-v2"
                and launch["pid"] == launch["pgid"] != launch["parent_pid"]
                and launch["clock_source_sha256"] == CLOCK_PIN
                and launch["watchdog_sha256"] == plan["sources"][SUPERVISOR]
                and worker["clock_backend"] == launch["clock_backend"]
                and worker["wall_seconds"] == (worker["finished_ns"]-worker["started_ns"])/1e9
                and terminal["elapsed_ns"] == terminal["finished_ns"]-terminal["started_ns"]
                and terminal["wall_seconds"] == terminal["elapsed_ns"]/1e9,
                "historical timing and supervisor identity")
        for name, pin in plan["sources"].items():
            require(name not in sources or sources[name] == pin, "consistent shared source")
            require(digest(relative(name), check)["sha256"] == pin, "unchanged historical source")
            sources[name] = pin
        for role, desc in plan["inputs"].items():
            bind(desc.get("path", role), {k: desc[k] for k in ("sha256", "bytes")})
        closed(f"{stem}/run-01", worker, payload_count)
        require(sum(p.stat().st_size for p in relative(f"{stem}/run-01").iterdir()) <= plan["limits"]["output_bytes"],
                "historical actual output cap")
        started = read(relative(f"{stem}/run-01/started.json"))
        require(started["launch"] == launch and started["started_ns"] == worker["started_ns"], "original started witness")
        request = started["request"]
        require(request["plan"] == str(relative(f"{stem}/plan-01.json"))
                and request["plan_sha256"] == worker["plan_sha256"]
                and request["output"] == str(relative(f"{stem}/run-01"))
                and request["supervision"] == str(relative(launch_name)), "historical request paths")
        command = list(launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        script = "scripts/study_otto_symmetry_head.py" if stem == SYMM else "scripts/qualify_otto_teacher_sampler.py"
        flags = ["--plan", request["plan"], "--plan-sha256", request["plan_sha256"],
                 "--supervision", request["supervision"], "--output", request["output"]]
        require(command == [plan["python_executable"], str(relative(script)), *([] if stem == SYMM else ["run"]), *flags],
                "exact historical command")
        require(all(c["attempted"] == c["returned"] for c in worker["calls"].values()), "closed historical calls")
        if stem == SYMM:
            require(worker["version"] == "otto-symmetry-head-v1" and plan["status"] == "frozen_before_native_run"
                    and worker["pending"] == [] and worker["collection_episodes"] == 384
                    and worker["completed_episodes"] == 720 and worker["completed_stage_fits"] == 12, "completed symmetry study")
            audit = read(relative(f"{SYMM}/audit-01/receipt.json"))
            require(audit["status"] == "completed" and audit["agreement"] is True
                    and audit["worker_sha256"] == PINS[f"{SYMM}/run-01/receipt.json"]
                    and audit["plan_sha256"] == worker["plan_sha256"]
                    and audit["terminal_sha256"] == PINS[f"{SYMM}/{prefix}.terminal.json"]
                    and audit["producer_source_sha256"] == sources["scripts/study_otto_symmetry_head.py"]
                    and audit["source"]["sha256"] == sources["scripts/audit_otto_symmetry_head.py"], "completed independent symmetry audit")
            closed(f"{SYMM}/audit-01", audit, 2)
            audit_request = read(relative(f"{SYMM}/audit-01/started.json"))["request"]
            require(audit_request["run"] == str(relative(f"{SYMM}/run-01"))
                    and audit_request["receipt_sha256"] == audit["worker_sha256"]
                    and audit_request["plan"] == request["plan"] and audit_request["plan_sha256"] == worker["plan_sha256"]
                    and audit_request["terminal"] == str(relative(f"{SYMM}/{prefix}.terminal.json"))
                    and audit_request["terminal_sha256"] == audit["terminal_sha256"], "independent original audit request")
        else:
            review = read(relative(f"{QUAL}/independent-review-01.json"))
            require(worker["version"] == "otto-teacher-sampler-qualification-v1" and plan["status"] == "frozen"
                    and worker["qualified"] is True and worker["pending"] is None
                    and worker["requires_successful_original_supervisor"] is True
                    and worker["calls"] == {k: {"attempted": n, "returned": n} for k, n in plan["expected_calls"].items()}
                    and worker["native_draws"] == {"hit": 440, "source": 12, "initial": 0}
                    and all(worker[k] == 0 for k in ("teacher_choices", "learned_model_calls", "training_calls", "sample_teacher_panel_calls")),
                    "strict completed native qualification")
            require(review["status"] == "passed" and review["failures"] == [] and review["sources"] == plan["sources"]
                    and review["inputs"] == plan["inputs"] and review["counts"]["posterior_pairs_verified"] == 408,
                    "completed independent native review")
            for name, desc in review["primary_files"].items():
                require(inputs[f"{QUAL}/{name}"] == desc, "native independent review joins")
    for name in NEW_SOURCES:
        sources[name] = digest(relative(name), check)["sha256"]
    require(all(sources[name] == pin for name, pin in PREPARED_SOURCES.items()), "qualified extraction and budget sources")
    for name, pin in PREPARED_INPUTS.items():
        require(bind(name)["sha256"] == pin, "fixed preparation evidence")
    proof = read(relative(f"{PILOT}/serialization-01.json"))
    require(proof["fits"] is True and proof["program_sha256"] == sources[BUDGET_PROGRAM]
            and proof["source"]["sha256"] == sources["src/openjev/research/otto_teacher_rollouts.py"]
            and proof["event_count_maximum"] == MAX_EVENTS and proof["enforced_sampler_event_cap_bytes"] == EVENT_BYTES
            and proof["largest_padded_event_bytes"] == 475
            and proof["non_sampler_output_reserve_bytes"] == NON_EVENT_RESERVE
            and proof["conservative_total_upper_bytes"] == projection()["projected_bytes"] < LIMITS["output_bytes"],
            "fixed fabricated serialization admission")
    first, second = [read(relative(f"{PILOT}/engineering-{i:02d}/receipt.json")) for i in (1, 2)]
    require(first["status"] == "failed" and first["sources_before"] == first["sources_after"]
            and first["sources_after"][ANCHORS] == sources[ANCHORS]
            and [(r["name"], r["exit_code"]) for r in first["results"]] == [("pytest", 0), ("ruff", 1)]
            and all(r["status"] == "returned" for r in first["results"]), "preserved passing tests and initial lint failure")
    require(second["status"] == "passed" and type(second["exit_code"]) is int and second["exit_code"] == 0
            and second["previous_receipt_sha256"] == PREPARED_INPUTS[f"{PILOT}/engineering-01/receipt.json"]
            and second["anchor_source_sha256"] == sources[ANCHORS]
            and second["source_after_sha256"] == sources["tests/test_otto_teacher_anchors.py"]
            and second["unchanged_executable_ast"] is True
            and second["ast_after_sha256"] == "bb9551f86cb9fdedf3827c7bbfb2f216f73700dbb186fc71cfbb567dc92f5e54",
            "passing lint-only correction retains qualified executable tests")
    for index, attempt in enumerate((first, second), 1):
        require(attempt["native_calls"] == attempt["model_calls"] == 0, "fabricated engineering only")
        closed(f"{PILOT}/engineering-{index:02d}", attempt, len(attempt["files"]))
    return sources, inputs


def metadata(check=None):
    sources, inputs = authenticate_history(check)
    runtime = {"python_executable": sys.executable, "python_version": platform.python_version(),
               "all_distributions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    reference = read(relative(f"{QUAL}/plan-01.json"))
    require(all(runtime[k] == reference[k] for k in runtime), "unchanged qualified runtime")
    return {"version": VERSION, "configuration": configuration(), "limits": LIMITS, "environment": ENVIRONMENT,
            "serialization_projection": projection(), "sources": sources, "inputs": inputs, **runtime}


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = self.launch = self.plan = None
        self.streams, self.calls, self.pending, self.events = {}, {}, {}, collections.Counter()
        self.bytes_written, self.event_count, self.sampler_bytes = 0, 0, 0
        self.io_seconds, self.operation_starts = 0., {}
        self.timings, self.panels = {}, []
        self.receipt = {"version": VERSION, "status": "started", "limits": LIMITS,
                        "native_calls": 0, "learned_model_calls": 0, "training_calls": 0,
                        "requires_successful_original_supervisor": True}

    def check(self):
        deadline = self.launch["deadline_ns"] if self.launch else self.start + LIMITS["native_seconds"] * 10**9
        require(self.clock.now_ns() < deadline, "suspend-inclusive collection deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"] and self.bytes_written <= LIMITS["output_bytes"], "collection resource bound")

    def emit(self, name, event, sampler=False):
        tick = self.clock.now_ns()
        self.receipt["pending_emission"] = {"file": name, "event": event}
        value = encoded(event)
        require(not sampler or len(value) <= EVENT_BYTES, "sampler event exceeds frozen byte allowance")
        require(self.bytes_written + len(value) <= LIMITS["output_bytes"], "output cap before journal write")
        require(sampler or self.bytes_written-self.sampler_bytes+len(value) <= NON_EVENT_RESERVE,
                "frozen non-event reserve")
        channel = "sampler" if sampler else "reconstruction"
        key = (channel, event.get("anchor_id"), event.get("operation_id"))
        if event["event"] == "attempt":
            require(key not in self.pending, "unique pending operation")
            self.pending[key], self.operation_starts[key] = event, tick
            count = self.calls.setdefault(channel + ":" + event["operation"], {"attempted": 0, "returned": 0})
            count["attempted"] += 1
        elif event["event"] == "return":
            require(all(event[k] == v for k, v in self.pending[key].items() if k != "event"),
                    "matching operation return")
        if name not in self.streams:
            self.streams[name] = (self.out / name).open("xb", buffering=0)
        stream = self.streams[name]
        written = stream.write(value)
        self.bytes_written += written
        if sampler:
            self.sampler_bytes += written
        require(written == len(value), "complete journal write")
        os.fsync(stream.fileno())
        if event["event"] == "return":
            del self.pending[key]
            self.calls[channel + ":" + event["operation"]]["returned"] += 1
            started = self.operation_starts.pop(key)
            if not sampler:
                field = event["operation"] + "_instrumented_seconds"
                self.timings[field] = self.timings.get(field, 0.) + (self.clock.now_ns()-started)/1e9
        if sampler:
            self.event_count += 1
            self.events[event["event"]] += 1
            require(self.event_count <= MAX_EVENTS, "prospective sampler event cap")
        self.receipt["pending_emission"] = None
        self.io_seconds += (self.clock.now_ns()-tick)/1e9

    def artifact(self, name, value):
        tick = self.clock.now_ns()
        data = encoded(value)
        require(self.bytes_written-self.sampler_bytes+len(data) <= NON_EVENT_RESERVE, "frozen non-event reserve")
        require(self.bytes_written + len(data) <= LIMITS["output_bytes"], "output cap before artifact")
        write(self.out / name, value)
        self.bytes_written += len(data)
        self.io_seconds += (self.clock.now_ns()-tick)/1e9

    def bind(self):
        require(digest(relative(CLOCK))["sha256"] == CLOCK_PIN, "qualified clock identity")
        self.clock = load(relative(CLOCK), "_teacher_label_clock").SuspendClock()
        self.start = self.clock.now_ns()
        require(self.clock.backend in ("mach_continuous_time", "CLOCK_BOOTTIME"), "native suspend-inclusive clock")
        require(digest(self.args.plan, self.check)["sha256"] == self.args.plan_sha256, "external collection plan")
        self.plan = read(self.args.plan)
        expected = metadata(self.check)
        require(self.plan["status"] == "frozen_before_collection"
                and all(self.plan[k] == v for k, v in expected.items()), "current sources/runtime/input closure")
        require(all(os.environ.get(k) == v for k, v in ENVIRONMENT.items()), "single-thread pre-import environment")
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, "original supervisor launch missing")
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        argv = [sys.executable, str(relative(SCRIPT)), "run", "--plan", str(self.args.plan), "--plan-sha256",
                self.args.plan_sha256, "--supervision", str(self.args.supervision), "--output", str(self.out)]
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == argv == [sys.executable, *sys.argv] and self.launch["cwd"] == str(ROOT) == str(Path.cwd())
                and self.launch["pid"] == os.getpid() and self.launch["pgid"] == os.getpgrp()
                and self.launch["parent_pid"] == os.getppid(), "exact original process invocation")
        require(self.launch["cap_seconds"] == 1800 and self.launch["clock_source_sha256"] == CLOCK_PIN
                and self.launch["watchdog_sha256"] == self.plan["sources"][SUPERVISOR]
                and self.launch["clock_backend"] == self.clock.backend
                and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
                and self.launch["deadline_ns"] == self.launch["started_ns"] + 1800 * 10**9, "original finite supervisor")
        self.receipt.update(plan_sha256=self.args.plan_sha256, supervision_sha256=digest(self.args.supervision)["sha256"],
                            sources=self.plan["sources"], inputs=self.plan["inputs"])
        self.artifact("started.json", {"request": {k: str(v) for k, v in vars(self.args).items()},
                                      "started_ns": self.start, "launch": self.launch})
        self.timings["authentication_seconds"] = (self.clock.now_ns() - self.start) / 1e9

    def body(self):
        import numpy as np
        sys.path.insert(0, str(ROOT / "src"))
        from openjev.research import otto_teacher_anchors as helper
        from openjev.research.otto_teacher_costs import summarize_costs
        from openjev.research.otto_teacher_rollouts import sample_teacher_panel

        tick = self.clock.now_ns()
        selections = self.plan["selected_anchors"]
        require(len(selections) == 6 and [r["anchor_id"] for r in selections] == list(range(6)), "frozen six anchors")
        kernels = {}
        for regime in ("lambda3", "lambda4"):
            with np.load(relative(f"{SYMM}/run-01/kernel-{regime}.npz"), allow_pickle=False) as arrays:
                require(set(arrays.files) == {"likelihood", "initial_hit_weights"}, "historical public kernel schema")
                kernels[regime] = arrays["likelihood"].copy()
        with relative(f"{SYMM}/run-01/collection-transitions.jsonl").open("rb") as stream:
            def checked_lines():
                for line in stream:
                    self.check()
                    yield line
            projected = helper.iter_public_records(checked_lines(), selections)
            anchors = helper.reconstruct_anchors(selections, projected, kernels, check=self.check,
                emit=lambda event: self.emit("reconstruction.jsonl", event))
        self.timings["extraction_reconstruction_validation_seconds"] = (self.clock.now_ns() - tick) / 1e9
        require(self.calls.get("reconstruction:anchor_validation") == {"attempted": 6, "returned": 6}
                and self.calls.get("reconstruction:public_reset") == {"attempted": 6, "returned": 6}
                and self.calls.get("reconstruction:public_update")
                    == dict.fromkeys(("attempted", "returned"), sum(r["prefix_index"] for r in selections))
                and not self.pending, "all six anchors validated before source generation")
        self.check()
        require(self.bytes_written-self.sampler_bytes+2*1024**2 <= NON_EVENT_RESERVE, "room for public anchor arrays")
        with (self.out / "anchors.npz").open("xb") as stream:
            np.savez(stream, beliefs=np.stack([r["belief"] for r in anchors]),
                     kernel_lambda3=kernels["lambda3"], kernel_lambda4=kernels["lambda4"])
            stream.flush()
            os.fsync(stream.fileno())
        self.bytes_written += (self.out / "anchors.npz").stat().st_size
        self.artifact("anchors.json", {"selected": selections, "scope": "Copied exact public beliefs; no original source or rewards."})
        for row in anchors:
            self.check()
            tick = self.clock.now_ns()
            self.receipt["active_panel"] = row["anchor_id"]
            records = sample_teacher_panel(row["public"], row["belief"], kernels[row["regime"]],
                seed=SEED, anchor_id=row["anchor_id"], replicate_ids=REPLICATES, horizon=HORIZON,
                check=self.check, emit=lambda event: self.emit("events.jsonl", event, sampler=True))
            panel_seconds = (self.clock.now_ns() - tick) / 1e9
            tick = self.clock.now_ns()
            reduced = summarize_costs(records, eligible_actions=row["public"]["valid_actions"],
                                      replicate_ids=REPLICATES, horizon=HORIZON, familywise_alpha=.05)
            lookup = {(r.replicate_id, r.first_action): r for r in records}
            ties = []
            for a, b in itertools.combinations(row["public"]["valid_actions"], 2):
                diffs = [lookup[r, a].steps - lookup[r, b].steps for r in REPLICATES]
                ties.append({"first_action": a, "second_action": b, "differences": diffs,
                    "exact_replicate_ties": sum(x == 0 for x in diffs),
                    "all_censored_replicate_ties": sum(not lookup[r, a].found and not lookup[r, b].found for r in REPLICATES),
                    "equal_means": sum(diffs) == 0})
            panel = {"anchor_id": row["anchor_id"], "episode_id": row["episode_id"],
                     "records": [vars(r) for r in records], "reduction": reduced, "paired_details": ties,
                     "sampler_seconds_including_io": panel_seconds,
                     "reduction_seconds": (self.clock.now_ns() - tick) / 1e9}
            self.panels.append(panel)
            self.artifact(f"panel-{row['anchor_id']}.json", panel)
            self.receipt["active_panel"] = None
        c = sum(len(p["records"]) for p in self.panels)
        s = sum(r["steps"] for p in self.panels for r in p["records"])
        f = sum(r["found"] for p in self.panels for r in p["records"])
        expected = {"anchor_snapshot": 6, "source_generator": 96, "source_draw": 96,
                    "teacher_snapshot": c, "hit_generator": c, "teacher_choose": s-c,
                    "movement": s, "hit_draw": s-f, "teacher_update": s}
        require(c == 16*sum(len(r["public"]["valid_actions"]) for r in anchors)
                and not self.pending and all(self.calls.get("sampler:"+k) == {"attempted": n, "returned": n}
                for k, n in expected.items()) and self.event_count == 402 + 4*c + 8*s - 2*f
                and self.events == {"attempt": sum(expected.values()), "return": sum(expected.values()),
                                    "forced_action": c, "record": c, "panel_complete": 6}, "complete exact sampler ledger")
        result = {"version": VERSION, "technical_completion": True, "anchors": 6, "continuations": c,
                  "moves": s, "found": f, "calls": self.calls, "sampler_events": self.event_count,
                  "panels": self.panels, "timings": self.timings,
                  "scope": "Six fixed feasibility anchors; no efficacy, confidence or architecture admission. All panel time includes durable I/O."}
        self.artifact("summary.json", result)

    def execute(self):
        for path in (self.out, self.args.plan, self.args.supervision):
            regular(path)
        self.out.mkdir(parents=True, exist_ok=False)
        try:
            self.bind()
            self.body()
            self.close()
            self.check()
            tick = self.clock.now_ns()
            require(all(self.plan[k] == v for k, v in metadata(self.check).items())
                    and digest(self.args.plan, self.check)["sha256"] == self.args.plan_sha256
                    and digest(self.args.supervision, self.check)["sha256"] == self.receipt["supervision_sha256"],
                    "closing source/input/runtime/plan/launch identity")
            self.timings["closing_authentication_seconds"] = (self.clock.now_ns()-tick)/1e9
            files = {p.name: digest(p, self.check) for p in self.out.iterdir()}
            finished = self.clock.now_ns()
            self.receipt.update(status="completed", technical_completion=True, pending=[], calls=self.calls,
                started_ns=self.start, finished_ns=finished, wall_seconds=(finished-self.start)/1e9,
                clock_backend=self.clock.backend, timings=self.timings, sampler_events=self.event_count,
                files=files, artifact_serialization_io_seconds_nested=self.io_seconds, output_bytes_before_receipt=self.bytes_written,
                timing_scope="Worker elapsed through payload hashing, before final receipt publication; original parent covers complete process. All serialization/I/O retained, nested timings not additive.")
            self.artifact("receipt.json", self.receipt)
            require(sum(p.stat().st_size for p in self.out.iterdir()) <= LIMITS["output_bytes"], "final actual output bound")
            self.check()
            print(json.dumps({"status": "completed", "receipt_sha256": digest(self.out / "receipt.json")["sha256"]}), flush=True)
            return 0
        except BaseException as error:
            self.receipt.update(status="failed", technical_completion=False, error=repr(error), traceback=traceback.format_exc(),
                pending=[{"channel": k[0], **v} for k, v in self.pending.items()], calls=self.calls,
                sampler_events=self.event_count, timings=self.timings)
            try:
                self.receipt["failure_elapsed_seconds"] = (self.clock.now_ns()-self.start)/1e9 if self.clock else None
            except BaseException as secondary:  # noqa: BLE001 - clock failure must not block evidence
                self.receipt["failure_clock_error"] = repr(secondary)
            try:
                self.close(suppress=True)
            except BaseException as secondary:  # noqa: BLE001 - cleanup must not block failed receipt
                self.receipt.setdefault("cleanup_errors", []).append(repr(secondary))
                error.add_note(f"Cleanup error: {secondary!r}")
            try:
                path = self.out / "receipt.json"
                if path.exists():
                    path.rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {p.name: digest(p) for p in self.out.iterdir() if p.is_file()}
                write(path, self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - primary failure must survive publication errors
                error.add_note(f"Failure publication error: {secondary!r}")
            raise

    def close(self, suppress=False):
        errors = []
        for name, stream in self.streams.items():
            try:
                if not stream.closed:
                    stream.close()
            except BaseException as error:  # noqa: BLE001 - attempt every handle and preserve primary failure
                errors.append({"file": name, "error": repr(error)})
        if errors:
            self.receipt.setdefault("cleanup_errors", []).extend(errors)
            if not suppress:
                raise OSError("journal cleanup failed; per-handle errors preserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    plan = modes.add_parser("plan")
    plan.add_argument("--output", type=Path, required=True)
    run = modes.add_parser("run")
    for name in ("plan", "supervision", "output"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "plan":
        regular(args.output)
        require(digest(relative(CLOCK))["sha256"] == CLOCK_PIN, "plan clock pin")
        clock = load(relative(CLOCK), "_teacher_label_plan_clock").SuspendClock()
        start = clock.now_ns()

        def check():
            require(clock.now_ns()-start < 1800*10**9, "bounded plan preparation")
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
            require(rss <= LIMITS["rss_bytes"], "plan metadata memory bound")

        result = metadata(check)
        require(all(os.environ.get(k) == v for k, v in ENVIRONMENT.items()), "one-thread selector import")
        sys.path.insert(0, str(ROOT / "src"))
        helper = load(relative(ANCHORS), "_teacher_label_metadata_selector")
        count = 0
        with relative(f"{SYMM}/run-01/dagger-rows.jsonl").open("rb") as stream:
            def rows():
                nonlocal count
                for line in stream:
                    check()
                    lexical = helper._decode_record(line)
                    require(lexical["row_index"] == str(count), "complete ordered retained metadata")
                    count += 1
                    yield {**{k: helper._numbers(v) for k, v in lexical.items() if k != "teacher_costs"},
                           "teacher_costs": None}
            result["selected_anchors"] = list(helper.select_anchors(rows()))
        require(count == 4596, "complete authenticated original learner TRAIN pool")
        require(not any(name in sys.modules for name in ("torch", "tensorflow", "scipy")), "no model or native runtime")
        result.update(status="frozen_before_collection", selection_rows=count,
                      selection_wall_seconds=(clock.now_ns()-start)/1e9)
        check()
        write(args.output, result)
        print(json.dumps({"plan": str(args.output), **digest(args.output)}), flush=True)
        return 0

    def interrupted(signum, _frame):
        raise InterruptedError(f"collector received signal {signum}")

    old = signal.signal(signal.SIGTERM, interrupted)
    try:
        return Run(args).execute()
    finally:
        signal.signal(signal.SIGTERM, old)


if __name__ == "__main__":
    raise SystemExit(main())
