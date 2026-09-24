"""DEV-only full-census collector for the matched-compute readout study.

This explicit source fork preserves the qualified residual collector's
fixed actions, native setup, compressed ledger and episode durability. It never
imports or invokes that collector's Run/freeze or mutates its globals. Complete
source/runtime/input metadata admission precedes every numerical import.

The allocation contains exactly 36 fresh development paths. No confirmation
roster or execution is supported. Old query-memory TEST and the old residual
confirmation seeds remain closed. No learner, fitting or model selection.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import time
import traceback
import types
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-readout-compute-collection-v1"
SELF, TEST = "scripts/collect_otto_readout_compute.py", "tests/test_collect_otto_readout_compute.py"
PROTOCOL = "research/otto-readout-compute-protocol.md"
QUALIFIER = "scripts/qualify_otto_readout_compute.py"
FORK_SOURCE = "scripts/collect_otto_residual.py"
FORK_PIN = "62455365dd7a7f56662a2a2a02c10668ebe16bc4fbe173924a1af83a2a7f2815"
COLLECTOR = "scripts/collect_otto_query_gate.py"
COLLECTOR_PIN = "5e2b06fc0ff5b888122e733f12b20c881deab7f56b94908ed5a283f7a48c76df"
COLLECTION_PLAN_PIN = "f16f23f82d9eb271ac3c1a15df0c567f0a782b169abcdd4711de8fb9a7700be6"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
INTERPRETER = ".venv-otto-released-native/bin/python"
ARMS = ("analytic", "neural", "period4_hold")
FIRST = {"dev": {"lambda3": 318000001, "lambda4": 319000001}}
CASES = {"dev": 6}
STAGE_EPISODES = {"dev": 36}
PHASE = "dev"
HORIZON, EPISODES, MASTER_EPISODES = 2188, 36, 36
FIT_SEEDS = (309000001, 309000002, 309000003)
LIMITS = {"native_seconds": 3600, "rss_bytes": 4 * 1024**3, "output_bytes": 1024**3}
CALL_CAPS = {"tensorflow_construction": 1, "tensorflow_build": 1, "tensorflow_load": 1,
    "tensorflow_value": EPISODES * HORIZON, "native_reset": EPISODES,
    "native_step": EPISODES * HORIZON, "actor_construction": EPISODES,
    "backend_binding": EPISODES, "analytic_score": EPISODES * HORIZON,
    "feature_build": EPISODES * HORIZON, "teacher_score": EPISODES * HORIZON,
    "actor_update": EPISODES * HORIZON, "annotation_public_reset": 0,
    "annotation_public_update": 0}
CONFIGURATION = {"phase": PHASE, "arms": list(ARMS), "first_seeds": FIRST, "cases": CASES,
    "horizon": HORIZON, "master_episodes": MASTER_EPISODES, "dev_episodes": EPISODES,
    "confirm_reserved_episodes": 0, "confirm_collected_episodes": 0,
    "rotation": "global case index modulo three; DEV is the only roster",
    "initial_hit": "1 + case % 3", "Ngrid": 53, "Nhits": 4, "R_dt": 2., "norm_Poisson": "Euclidean",
    "collector_period": 4, "annotation": "complete DEV census; one score per retained state",
    "features": "virtual last_query=step-step%4, age=step%4, has_queried=1, including correction rows",
    "action_boundary": "skip action fixed before annotation; annotation never refreshes held cache",
    "confirmation_boundary": "not supported; no new confirmation allocation; old TEST and old confirmation remain closed",
    "old_test_access": False, "deferred_annotation": False, "window_sampling": False,
    "training_updates": 0, "warmup_forwards": 0}
ROLES = {"collection_plan", "seed_review", "engineering"}
NEW_COMPONENTS = {SELF, TEST, PROTOCOL, QUALIFIER}
NEW_SOURCES = NEW_COMPONENTS | {FORK_SOURCE}
SEED_REVIEW = "output/otto-readout-compute-v1/seed-review-01.json"
SEED_REVIEW_PIN = "ff963f141611c5f252237501d263391d4eefb12c3952781cd3ec19bbffd1320e"
RESERVATION = "output/otto-readout-compute-v1/seed-reservation-01.json"
RESERVATION_PIN = "a10bb81aba770cc54caa5397f9000e6c815f7d114c41491ac6cf17647852ac28"
# No annotations journal is declared: without deferred replay it is never emitted.
JOURNALS = {"work.jsonl", "weights.jsonl", "forwards.jsonl", "transitions.jsonl", "samples.jsonl"}
PAYLOADS = {f"{name}.gz" for name in JOURNALS} | {"started.json", "runtime.json", "setup.json",
    "deployment.json", "cohort.json", "episode-boundaries.jsonl", "episodes.jsonl",
    "dev.npz", "costs.json", "summary.json"}
ARRAY_KEYS = ("features", "raw_q", "legal", "actions", "correction", "episode_offsets")

def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_file() and path.is_relative_to(ROOT) and ".." not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents)), "contained regular input")
    return path


def absolute_location(value):
    """Check public input/output locations before authentication or work."""
    path = Path(value)
    require(path.is_absolute() and path.is_relative_to(ROOT) and ".." not in path.parts
            and not path.is_symlink()
            and all(parent.is_dir() and not parent.is_symlink() for parent in path.parents),
            "absolute contained path with existing nonsymlink directory parents")
    return path


def exclusive_output(value):
    path = absolute_location(value)
    require(not path.exists(), "exclusive output must not already exist")
    return path


def validate_worker_paths(args):
    regular(absolute_location(args.plan))
    launch = absolute_location(args.supervision)
    require(not launch.exists() or launch.is_file(), "supervision must be absent or a regular file")


def descriptor(path):
    path = regular(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def read(path):
    return json.loads(regular(path).read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cohort(stage=None):
    require(stage is None or stage in FIRST, "registered master cohort stage")
    result, global_case = [], 0
    for split, regimes in FIRST.items():
        for regime, first in regimes.items():
            for case in range(CASES[split]):
                offset = global_case % len(ARMS)
                for arm in ARMS[offset:] + ARMS[:offset]:
                    result.append({"stage": split, "episode_index": len(result), "regime": regime,
                        "seed": first + case, "case": case, "initial_hit": 1 + case % 3,
                        "arm": arm, "episode_id": f"{split}:{regime}:{first + case}:{arm}"})
                global_case += 1
    return result if stage is None else [r for r in result if r["stage"] == stage]


def execution_cohort(phase):
    require(phase == PHASE,
            "only DEV collection is implemented; confirmation needs a separately admitted future phase with new allocation")
    return cohort(PHASE)


def validate_plan(plan):
    """Reject any non-DEV phase before inherited authentication or setup."""
    expected = execution_cohort(plan.get("phase"))
    require(plan["version"] == VERSION and plan["status"] == "frozen_before_collection"
            and plan["configuration"] == CONFIGURATION and plan["limits"] == LIMITS
            and plan["call_caps"] == CALL_CAPS and plan["cohort"] == cohort()
            and plan["execution_cohort"] == expected and set(plan["payloads"]) == PAYLOADS,
            "fixed DEV-only execution and full master allocation")


def virtual_query(step):
    require(type(step) is int and 0 <= step < HORIZON, "preaction step")
    return step - step % 4


def route_scores(arm, step, allowed, analytic_action, teacher, select, cache):
    """Exactly one census score; undeployed annotations never change held scores."""
    require(arm in ARMS and callable(teacher) and callable(select), "fixed controller and scorers")
    correction = virtual_query(step) == step
    deployed = arm == "neural" or (arm == "period4_hold" and correction)
    before = None if cache is None else cache.tobytes()
    action = analytic_action if arm == "analytic" else None
    if arm == "period4_hold" and not correction:
        require(cache is not None, "held controller requires earlier correction")
        action = select(cache, allowed)
    raw = teacher(deployed)
    require(cache is None or cache.tobytes() == before, "annotation cannot modify held memory")
    if deployed:
        action = select(raw, allowed)
        if arm == "period4_hold":
            cache = raw.copy()
            cache.flags.writeable = False
    require(action in allowed, "inbounds actual action")
    return int(action), raw, cache, deployed, correction


def episode_offsets(lengths):
    require(lengths and all(type(n) is int and 1 <= n <= HORIZON for n in lengths), "all complete positive episode lengths")
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    return offsets



def authenticate_engineering(path, sources):
    """Bind all qualified bytes; only named pytest fixture storage is excluded.

    The qualifier may cover more components than this collector, but cannot
    replace any inherited source. Its direct regular files have exact closure;
    only a genuine nonsymlink pytest-temp directory is outside that manifest.
    """
    engineering = read(path)
    require(engineering["status"] == "passed"
            and isinstance(engineering["source_after"], dict)
            and engineering["source_before"] == engineering["source_after"]
            and NEW_COMPONENTS <= set(engineering["source_after"])
            and isinstance(engineering["commands"], list) and engineering["commands"]
            and all(type(command["returncode"]) is int and command["returncode"] == 0
                    and command["timed_out"] is False and command["reaped"] is True
                    and command["group_absent"] is True
                    for command in engineering["commands"]), "all current fabricated component qualifications")
    for name, expected in engineering["source_after"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and isinstance(expected, dict) and set(expected) == {"sha256", "bytes"}
                and descriptor(ROOT / name) == expected, "current qualified source descriptor")
        require(name not in sources or sources[name] == expected["sha256"],
                "qualified source cannot replace inherited source")
        sources[name] = expected["sha256"]
    directory = regular(path).parent
    names = set()
    for entry in directory.iterdir():
        if entry.name == "pytest-temp":
            require(entry.is_dir() and not entry.is_symlink(), "genuine excluded pytest-temp directory")
            continue
        require(entry.is_file() and not entry.is_symlink(), "only declared regular engineering files")
        names.add(entry.name)
    require(isinstance(engineering["files"], dict)
            and names == set(engineering["files"]) | {"receipt.json"}, "engineering file closure")
    for name, expected in engineering["files"].items():
        require(Path(name).name == name and set(expected) == {"sha256", "bytes"}
                and descriptor(directory / name) == expected, "engineering log descriptor")
    for command in engineering["commands"]:
        require(command["log"] in engineering["files"]
                and {key: command[key] for key in ("sha256", "bytes")}
                == engineering["files"][command["log"]], "command log equals authenticated file")


def validate_seed_review(seeds, reservation):
    """Check the full new allocation and the exact reviewed lexical matches.

    The bounded historical scan is evidence of scoped reservation only. Its
    files are not assumed unchanged forever. The saved inventory and reviewed
    matches are bound by the two immutable receipt hashes, independently of the
    current experiment's source/runtime closure. Whole blocks are not reserved.
    """
    expected_seeds = [first + case for split, regimes in FIRST.items()
                      for first in regimes.values() for case in range(CASES[split])]
    allocations = {split: {regime: list(range(first, first + CASES[split]))
                          for regime, first in regimes.items()} for split, regimes in FIRST.items()}
    require(seeds["version"] == "otto-readout-compute-seed-review-v1" and seeds["status"] == "reserved_before_run"
            and seeds["admits_execution"] is False and seeds["allocations"] == allocations
            and seeds["fresh_environment_seeds"] == expected_seeds and len(set(expected_seeds)) == 12
            and seeds["reused_fit_seeds"] == list(FIT_SEEDS) and not set(expected_seeds) & set(FIT_SEEDS)
            and type(seeds["exact_seed_hits"]) is int and seeds["exact_seed_hits"] == 0
            and all(type(seeds[name]) is int and seeds[name] == 0 for name in
                    ("checkpoint_decodes", "empirical_array_decodes", "raw_journal_reads", "scientific_calls")),
            "scoped exact fresh seed reservation review")
    require(reservation["version"] == "otto-readout-compute-seed-reservation-v1"
            and type(reservation["attempt"]) is int and reservation["attempt"] == 1
            and reservation["status"] == "candidate_blocks_require_review"
            and reservation["admits_execution"] is False and reservation["changed_files_during_scan"] == []
            and reservation["allocations"] == allocations and reservation["fresh_environment_seeds"] == expected_seeds
            and reservation["reused_fit_seeds"] == list(FIT_SEEDS)
            and type(reservation["exact_seed_hit_count"]) is int and reservation["exact_seed_hit_count"] == 0
            and isinstance(reservation["files"], dict) and bool(reservation["files"])
            and reservation["files_scanned"] == len(reservation["files"])
            and seeds["scope"] == reservation["scope"] and seeds["scope_limits"] == reservation["scope_limits"]
            and all(type(reservation[name]) is int and reservation[name] == 0 for name in
                    ("array_decodes", "checkpoint_decodes", "empirical_array_decodes", "model_calls",
                     "native_calls", "project_module_imports", "raw_journal_reads", "scientific_calls")),
            "original scoped seed reservation 01")
    hits, resolved = reservation["block_hits"], seeds["resolved_block_hits"]
    require(isinstance(hits, list) and isinstance(resolved, list) and len(hits) == len(resolved),
            "all conservative block matches resolved")
    for original, reviewed in zip(hits, resolved, strict=True):
        name = original["path"]
        require(type(name) is str and not Path(name).is_absolute() and ".." not in Path(name).parts
                and name in reservation["files"], "match belongs to saved scoped inventory")
        require(type(original["value"]) is int and type(original["literal"]) is str
                and int(original["literal"].replace("_", "")) == original["value"]
                and 318000000 <= original["value"] < 320000000
                and type(original["line"]) is int and original["line"] > 0
                and type(original["column"]) is int and original["column"] > 0
                and type(original["context"]) is str
                and original["context"][original["column"] - 1:
                    original["column"] - 1 + len(original["literal"])] == original["literal"],
                "exact saved lexical match")
        require(set(reviewed) == set(original) | {"classification", "source_verified"}
                and {name: reviewed[name] for name in original} == original
                and original["exact_seed"] is False and original["value"] not in expected_seeds
                and reviewed["source_verified"] is True
                and reviewed["classification"] ==
                "outside the exact selected environment seeds; whole-block freshness not claimed",
                "exact conservative match and explicit non-seed resolution")


def authenticate_seed_review(path, sources):
    require(regular(path) == ROOT / SEED_REVIEW and descriptor(path)["sha256"] == SEED_REVIEW_PIN,
            "original exact seed review 01")
    seeds = read(path)
    record = seeds["reservation"]
    require(set(record) == {"path", "sha256", "bytes"} and record["path"] == RESERVATION
            and record["sha256"] == RESERVATION_PIN
            and descriptor(record["path"]) == {key: record[key] for key in ("sha256", "bytes")},
            "review binds unchanged original reservation 01")
    reservation = read(RESERVATION)
    validate_seed_review(seeds, reservation)
    verified = {SEED_REVIEW: SEED_REVIEW_PIN, RESERVATION: RESERVATION_PIN}
    for name, pin in verified.items():
        require(name not in sources or sources[name] == pin, "seed evidence cannot replace inherited source")
        sources[name] = pin


def authenticate_inputs(inputs):
    require(set(inputs) == ROLES, "exact readout-compute collection roles")
    paths = {}
    for role, record in inputs.items():
        require(set(record) == {"path", "sha256", "bytes"}, "input descriptor")
        paths[role] = regular(record["path"])
        require(descriptor(paths[role]) == {k: record[k] for k in ("sha256", "bytes")}, "frozen input " + role)
    require(inputs["collection_plan"]["sha256"] == COLLECTION_PLAN_PIN
            and descriptor(ROOT / COLLECTOR)["sha256"] == COLLECTOR_PIN
            and descriptor(ROOT / FORK_SOURCE)["sha256"] == FORK_PIN,
            "original qualified collection authentication source and plan")
    collector = load(ROOT / COLLECTOR, "_readout_compute_qualified_collection")
    prior, native_paths, reference, native = collector.authenticate(types.SimpleNamespace(
        plan=paths["collection_plan"], plan_sha256=COLLECTION_PLAN_PIN))
    sources = dict(prior["sources"])
    for name in NEW_SOURCES:
        pin = descriptor(ROOT / name)["sha256"]
        require(name not in sources or sources[name] == pin, "new source cannot replace inherited source")
        sources[name] = pin
    authenticate_engineering(paths["engineering"], sources)
    authenticate_seed_review(paths["seed_review"], sources)
    return sources, prior["runtime"], native_paths, reference, native


def freeze(args):
    exclusive_output(args.output)
    inputs = {}
    for role in sorted(ROLES):
        path = regular(getattr(args, role))
        d = descriptor(path)
        require(d["sha256"] == getattr(args, role + "_sha256"), "external planning pin")
        inputs[role] = {"path": str(path.relative_to(ROOT)), **d}
    sources, runtime, native_paths, _, _ = authenticate_inputs(inputs)
    plan = {"version": VERSION, "status": "frozen_before_collection", "configuration": CONFIGURATION,
        "sources": sources, "inputs": inputs, "runtime": runtime, "limits": LIMITS, "call_caps": CALL_CAPS,
        "phase": PHASE, "payloads": sorted(PAYLOADS), "cohort": cohort(), "execution_cohort": execution_cohort(PHASE),
        "native_inputs": {k: {"path": str(p.relative_to(ROOT)), **descriptor(p)} for k, p in native_paths.items()}}
    write(args.output, plan)
    print(json.dumps({"status": plan["status"], "plan": descriptor(args.output)}), flush=True)


def paired_identity(row, paired):
    key = row["stage"], row["regime"], row["case"]
    draws = {(d["channel"], d["index"]): d["uniform"] for d in row["draws_evaluation_only"]}
    require(len(draws) == len(row["draws_evaluation_only"]), "unique paired draw identities")
    if key in paired:
        source, previous = paired[key]
        require(source == row["source_evaluation_only"] and all(previous[k] == draws[k]
                for k in previous.keys() & draws.keys()), "paired original source and uniform streams")
        previous.update(draws)
    else:
        paired[key] = row["source_evaluation_only"], draws

class Ledger:
    """Copied qualified compressed ledger; returns are provisional until episode fsync."""

    def __init__(self, run):
        self.run, self.context, self.sequence = run, {"phase": "setup"}, 0
        self.pending, self.pending_emission, self.streams = [], None, {}
        self.io_seconds = 0.
        self.last_seconds, self.last_instrumented, self.last_io = {}, {}, {}
        self.calls = {k: {"attempted": 0, "returned": 0, "seconds": 0.} for k in CALL_CAPS}

    def emit(self, name, value):
        require(name in JOURNALS, "declared bounded journal")
        self.pending_emission = {"file": name, "context": dict(self.context),
                                 "event": value.get("event"), "call_id": value.get("call_id")}
        tick = time.perf_counter()
        try:
            if name not in self.streams:
                raw = (self.run.out / f"{name}.gz").open("xb")
                self.streams[name] = (raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0))
            encoded = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            require(len(encoded) <= 64 * 1024 and self.streams[name][1].write(encoded) == len(encoded), "complete bounded event")
            self.pending_emission = None
        finally:
            self.io_seconds += time.perf_counter() - tick

    def call(self, channel, function):
        self.run.check()
        require(channel in CALL_CAPS and self.calls[channel]["attempted"] < CALL_CAPS[channel], "fixed operation cap")
        self.sequence += 1
        event = {"call_id": self.sequence, "channel": channel, "context": dict(self.context),
                 "parent_call_id": self.pending[-1]["call_id"] if self.pending else None}
        self.pending.append(event)
        self.calls[channel]["attempted"] += 1
        self.emit("work.jsonl", {"event": "attempt", **event})
        tick, io = time.perf_counter(), self.io_seconds
        result = function()
        raw, excluded = time.perf_counter() - tick, self.io_seconds - io
        elapsed = raw - excluded
        require(elapsed >= 0 and self.pending[-1] is event, "nested operation timing/order")
        self.emit("work.jsonl", {"event": "return", **event, "seconds": elapsed,
                               "instrumented_seconds": raw, "excluded_io_seconds": excluded})
        self.calls[channel]["returned"] += 1
        self.calls[channel]["seconds"] += elapsed
        self.last_seconds[channel], self.last_instrumented[channel], self.last_io[channel] = elapsed, raw, excluded
        self.pending.pop()
        self.run.check()
        return result

    def flush(self):
        tick = time.perf_counter()
        try:
            for raw, stream in self.streams.values():
                stream.flush()
                raw.flush()
                os.fsync(raw.fileno())
        finally:
            self.io_seconds += time.perf_counter() - tick

    def close(self, *, suppress=False):
        errors = []
        for name, (raw, stream) in self.streams.items():
            for handle in (stream, raw):
                try:
                    if not handle.closed:
                        if handle is raw:
                            handle.flush()
                            os.fsync(handle.fileno())
                        handle.close()
                except BaseException as error:  # noqa: BLE001 - preserve every cleanup failure and original work error
                    errors.append({"file": name, "error": repr(error)})
        if errors:
            self.run.receipt.setdefault("cleanup_errors", []).extend(errors)
            if not suppress:
                raise OSError(f"journal closure failed: {errors}")



class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = self.launch = self.plan = None
        self.ledger = Ledger(self)
        self.rows, self.datasets, self.label_mask = [], {}, []
        self.pending_episode = self.active_actor = self.pending_action = None
        self.buffer = {key: [] for key in ARRAY_KEYS if key != "episode_offsets"}
        self.receipt = {"version": VERSION, "status": "started", "training_updates": 0,
            "phase": PHASE, "confirm_episodes": 0, "old_test_array_decodes": 0,
            "scope": "DEV36 full fixed paths and teacher census. Old confirmation remains closed; no fitting or autonomous efficacy."}

    def check(self):
        require(self.clock.now_ns() < self.launch["deadline_ns"], "original shared readout-compute DEV collection deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"], "readout-compute DEV collection RSS cap")
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
                <= LIMITS["output_bytes"] - 1024**2, "output cap with failure reserve")

    def append_durable(self, name, value):
        require(name in ("episode-boundaries.jsonl", "episodes.jsonl"), "declared durable boundary")
        self.check()
        encoded = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        require(len(encoded) < 2 * 1024**2, "bounded episode record")
        with (self.out / name).open("ab") as stream:
            require(stream.write(encoded) == len(encoded), "complete boundary write")
            stream.flush()
            os.fsync(stream.fileno())

    def bind(self):
        require(descriptor(ROOT / CLOCK)["sha256"] == CLOCK_PIN, "clock pin before import")
        self.clock = load(ROOT / CLOCK, "_readout_compute_clock").SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, "original launch available")
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and command[:3] == [str(ROOT / INTERPRETER), str(ROOT / SELF), "run"]
            and self.launch["pid"] == os.getpid() and self.launch["pgid"] == os.getpgrp()
            and self.launch["parent_pid"] == os.getppid() and self.launch["pid"] != self.launch["parent_pid"]
            and Path(self.launch["cwd"]) == Path.cwd() == ROOT and self.launch["cap_seconds"] == LIMITS["native_seconds"]
            and self.launch["clock_backend"] == self.clock.backend
            and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
            and self.launch["deadline_ns"] == self.launch["started_ns"] + LIMITS["native_seconds"] * 10**9,
            "canonical original bounded parent")
        self.check()
        require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256, "external plan pin")
        self.plan = read(self.args.plan)
        validate_plan(self.plan)
        sources, runtime, self.paths, self.reference, self.native = authenticate_inputs(self.plan["inputs"])
        require(sources == self.plan["sources"] and runtime == self.plan["runtime"]
            and {k: {"path": str(p.relative_to(ROOT)), **descriptor(p)} for k, p in self.paths.items()}
            == self.plan["native_inputs"], "unchanged native/source/runtime admission")
        require(self.launch["watchdog_sha256"] == sources["scripts/supervise_dialogue_observation_v2.py"]
            and self.launch["clock_source_sha256"] == CLOCK_PIN, "original parent source")
        self.receipt.update(plan_sha256=self.args.plan_sha256, supervision_sha256=descriptor(self.args.supervision)["sha256"],
            sources=sources, inputs=self.plan["inputs"], native_inputs=self.plan["native_inputs"], limits=LIMITS)
        write(self.out / "started.json", {"started_ns": self.start, "launch": self.launch,
            "request": {k: str(v) for k, v in vars(self.args).items()}})

    def setup(self):
        validate_plan(self.plan)
        tick = time.perf_counter()
        runtime = self.reference.Run.setup(self)
        self.original_setup_wall = time.perf_counter() - tick
        tick = time.perf_counter()
        from openjev.research.otto_query_gate import FEATURE_NAMES, ReadOnlyBeliefView, _analytic, _features
        from openjev.research.otto_restricted_policy import select_inbounds_action
        self.feature_names, self.readonly = FEATURE_NAMES, ReadOnlyBeliefView
        self.analytic_scores, self.features, self.select = _analytic, _features, select_inbounds_action
        self.feature_module_seconds = time.perf_counter() - tick
        require("torch" not in sys.modules, "no learner imports or fitting")
        self.kernels = {"lambda3": runtime.kernels["base"], "lambda4": runtime.kernels["shift"]}
        self.mixtures = {"lambda3": runtime.weights["base"], "lambda4": runtime.weights["shift"]}
        write(self.out / "deployment.json", {"original_setup_wall_seconds": self.original_setup_wall,
            "feature_module_seconds": self.feature_module_seconds, "warmup_forwards": 0,
            "feature_names": list(self.feature_names), "mixtures": self.mixtures,
            "features": CONFIGURATION["features"], "teacher": "original restricted float32 four-action costs",
            "scope": "DEV36 full paths and every teacher score. No confirmation allocation or execution is admitted. Deployed and annotation-only scores are distinguished. Model-visible corrections remain period four; collector policies are distinct. No deployment-cost claim."})
        write(self.out / "cohort.json", {"episodes": cohort(), "execution_episodes": execution_cohort(PHASE),
            "phase": PHASE, "configuration": CONFIGURATION})
        return runtime

    def episode(self, identity):
        require(identity in execution_cohort(identity.get("stage")), "registered DEV identity before native reset")
        runtime, np = self.runtime, self.runtime.np
        arm, regime = identity["arm"], identity["regime"]
        start_row = len(self.buffer["features"])
        self.pending_episode = dict(identity)
        self.append_durable("episode-boundaries.jsonl", {"event": "attempt", **identity, "start_row": start_row})
        self.ledger.context = {"phase": "native_reset", **identity}
        env = self.ledger.call("native_reset", lambda: runtime.public.seeded_environment(runtime.source, identity["seed"],
            {"Ndim": 2, "Ngrid": 53, "Nhits": 4, "lambda_over_dx": float(regime[-1]),
             "R_dt": 2., "norm_Poisson": "Euclidean"}, initial_hit=identity["initial_hit"]))
        require(env.p_Poisson.tobytes() == self.kernels[regime].tobytes(), "reset equals authenticated kernel")
        current = self.reference.packet(runtime.public.observation(env, 0))
        self.ledger.context = {"phase": "actor_setup", **identity}
        actor = self.ledger.call("actor_construction", lambda: runtime.analytic(current, self.kernels[regime], allow_stay=False))
        self.active_actor = actor
        view = self.readonly(actor._view)
        recorder = self.reference.ForwardRecorder(runtime.model, runtime.policy._value_policy.__code__, self.ledger, np)
        teacher = self.ledger.call("backend_binding", lambda: runtime.policy(env=view, model=recorder, sym_avg=True))
        witness = self.reference.belief_witness(actor, env, current, np)
        self.ledger.emit("transitions.jsonl", {"kind": "reset", **identity, "public": current,
            "posterior_after": witness, "source_evaluation_only": env.source.tolist()})
        cache = previous = last_action = None
        deployed = annotations = corrections = 0
        for step in range(HORIZON):
            self.ledger.context = {"phase": "preaction", **identity, "step": step}
            allowed = current["valid_actions"]
            analytic_action, analytic = self.ledger.call("analytic_score", lambda allowed=allowed: self.analytic_scores(
                actor._policy._value_policy()[1], allowed))
            features, next_statistics = self.ledger.call("feature_build", lambda step=step, current=current, analytic=analytic, last_action=last_action, previous=previous: self.features(
                view, current, float(regime[-1]), analytic, last_action, virtual_query(step), previous))
            require(features[15] == np.float32(step / HORIZON) and features[16] == np.float32((step % 4) / HORIZON)
                and features[17] == np.float32(1), "declared virtual correction features")
            public_before, belief_before = actor.public, actor.belief.tobytes()
            tf_before = self.ledger.calls["tensorflow_value"]["returned"]

            def score(is_deployed, step=step, public_before=public_before, belief_before=belief_before):
                self.ledger.context = {"phase": "teacher_score", **identity, "step": step,
                    "deployed_query": is_deployed, "annotation_only": not is_deployed}
                raw = self.ledger.call("teacher_score", lambda: teacher._value_policy()[1])
                require(raw.shape == (4,) and raw.dtype == np.float32 and np.isfinite(raw).all(), "all four raw teacher costs")
                require(actor.public == public_before and actor.belief.tobytes() == belief_before
                    and actor._pending_action is None, "functional annotation cannot change actor state")
                return raw.copy()

            def select_and_record(scores, permitted, step=step):
                action = self.select(scores, permitted)
                self.pending_action = {**identity, "step": step, "action": action}
                return action
            if arm == "analytic":
                self.pending_action = {**identity, "step": step, "action": analytic_action}
            action, raw, cache, queried, correction = route_scores(
                arm, step, allowed, analytic_action, score, select_and_record, cache)
            require(self.ledger.calls["tensorflow_value"]["returned"] - tf_before == 1,
                    "one physical forward only for a deployed or census annotation request")
            deployed += int(queried)
            annotations += int(not queried)
            corrections += int(correction)
            legal = np.asarray([a in allowed for a in range(4)], dtype=np.bool_)
            row_index = len(self.buffer["features"])
            self.ledger.emit("samples.jsonl", {**identity, "row_index": row_index, "step": step,
                "public": current, "posterior": witness, "features": features.tolist(),
                "raw_q": raw.tolist(), "label_available": True,
                "legal": legal.tolist(), "action": action, "analytic_action": analytic_action,
                "deployed_query": queried, "annotation_only": not queried, "correction_scheduled": correction,
                "virtual_last_query": virtual_query(step),
                "held_q": None if cache is None else cache.tolist(),
                "features_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
                "raw_q_sha256": hashlib.sha256(raw.tobytes()).hexdigest()})
            self.ledger.context = {"phase": "transition", **identity, "step": step + 1}
            result = self.ledger.call("native_step", lambda action=action: env.step(action, quiet=True))
            after = self.reference.packet(runtime.public.observation(env, step + 1))
            require((int(result[0]), bool(result[2])) == (after["hit"], after["done"]), "native public observation")
            self.ledger.call("actor_update", lambda action=action, after=after: actor.update(action, after))
            next_witness = self.reference.belief_witness(actor, env, after, np)
            require(after["position"] != current["position"], "inbounds movement")
            self.ledger.emit("transitions.jsonl", {"kind": "step", **identity, "step": step + 1,
                "row_index": row_index, "action": action, "deployed_query": queried, "public": after,
                "posterior_before": witness, "posterior_after": next_witness, "native_p_end": float(result[1])})
            for key, value in zip(ARRAY_KEYS[:-1], (features.copy(), raw.copy(), legal, action, correction), strict=True):
                self.buffer[key].append(value)
            self.label_mask.append(True)
            self.pending_action = None
            current, witness, last_action, previous = after, next_witness, action, next_statistics
            if current["done"]:
                break
        steps = step + 1
        require(deployed == (steps if arm == "neural" else (steps + 3) // 4 if arm == "period4_hold" else 0)
            and deployed + annotations == steps
            and corrections == (steps + 3) // 4, "controller and annotation accounting")
        draws = [{k: r[k] for k in ("channel", "index", "uniform", "selected_index", "cdf_mass")} for r in env.draw_log]
        require([r["index"] for r in draws if r["channel"] == "source"] == [0]
            and [r["index"] for r in draws if r["channel"] == "hit"] == list(range(steps - int(current["done"]))),
            "all native draw identities and final no-hit semantics")
        return {**identity, "steps": steps, "rows": steps, "start_row": start_row,
            "end_row": len(self.buffer["features"]), "found": current["done"], "censored": not current["done"],
            "deployed_queries": deployed, "annotation_only": annotations, "teacher_calls": deployed + annotations,
            "deferred_annotations": 0, "backend_bindings": 1,
            "unlabeled_rows": sum(not v for v in self.label_mask[start_row:start_row + steps]),
            "corrections": corrections, "updates": steps, "final_public": current, "final_update_assimilated": True,
            "source_evaluation_only": env.source.tolist(), "draws_evaluation_only": draws}

    def complete_episode(self, identity, paired):
        expected = execution_cohort(identity.get("stage"))
        require(len(self.rows) < len(expected) and identity == expected[len(self.rows)],
                "next exact master DEV identity before episode attempt")
        row = self.episode(identity)
        paired_identity(row, paired)
        self.ledger.flush()
        self.append_durable("episodes.jsonl", row)
        self.append_durable("episode-boundaries.jsonl", {"event": "return", **identity, "steps": row["steps"],
            "start_row": row["start_row"], "end_row": row["end_row"]})
        self.rows.append(row)
        self.pending_episode, self.active_actor = None, None
        self.check()

    def save_stage(self, stage):
        expected = execution_cohort(stage)
        np = self.runtime.np
        rows = self.rows
        require([{k: r[k] for k in e} for r, e in zip(rows, expected, strict=True)] == expected, "all stage identities")
        offsets = episode_offsets([r["steps"] for r in rows])
        require([(r["start_row"], r["end_row"]) for r in rows] == list(pairwise(offsets)),
            "full chronological per-episode slices")
        arrays = {key: np.asarray(self.buffer[key], dtype=dtype) for key, dtype in zip(ARRAY_KEYS[:-1],
            (np.float32, np.float32, np.bool_, np.int64, np.bool_), strict=True)}
        arrays["episode_offsets"] = np.asarray(offsets, dtype=np.int64)
        n = offsets[-1]
        require([arrays[k].shape for k in ARRAY_KEYS] == [(n, 31), (n, 4), (n, 4), (n,), (n,), (len(rows) + 1,)],
            "exact flat stage schema")
        require(np.isfinite(arrays["features"]).all() and np.isfinite(arrays["raw_q"]).all(), "finite public dataset")
        mask = np.asarray(self.label_mask, dtype=np.bool_)
        require(mask.shape == (n,) and bool(mask.all()), "complete teacher census in every split")
        path = self.out / f"{stage}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        with np.load(path, allow_pickle=False) as saved:
            require(set(saved.files) == set(arrays) and all(saved[k].dtype == arrays[k].dtype
                and saved[k].shape == arrays[k].shape and saved[k].tobytes() == arrays[k].tobytes() for k in arrays),
                "saved flat dataset byte equality")
        self.datasets[stage] = {"path": path.name, **descriptor(path), "episodes": len(rows), "rows": n,
            "arrays": {k: {"dtype": str(v.dtype), "shape": list(v.shape), "bytes": v.nbytes,
                           "sha256": hashlib.sha256(v.tobytes()).hexdigest()} for k, v in arrays.items()}}
        self.buffer = {key: [] for key in ARRAY_KEYS[:-1]}
        self.label_mask = []
        self.check()

    def body(self):
        validate_plan(self.plan)
        self.runtime = self.setup()
        self.ledger.flush()
        timings, paired = {}, {}
        for stage in (PHASE,):
            tick = time.perf_counter()
            for identity in execution_cohort(stage):
                self.complete_episode(identity, paired)
                if len(self.rows) % 3 == 0:
                    print(json.dumps({"phase": stage, "completed_episodes": len(self.rows), "total": EPISODES}), flush=True)
            timings[stage + "_collection_seconds"] = time.perf_counter() - tick
            tick = time.perf_counter()
            self.save_stage(stage)
            timings[stage + "_serialization_seconds"] = time.perf_counter() - tick
        n = sum(r["steps"] for r in self.rows)
        require(len(self.rows) == EPISODES and not self.ledger.pending and self.ledger.pending_emission is None
            and all(x["attempted"] == x["returned"] for x in self.ledger.calls.values()), "all episodes and calls complete")
        for channel in ("native_step", "actor_update", "analytic_score", "feature_build"):
            require(self.ledger.calls[channel]["returned"] == n, "one operation per retained state: " + channel)
        teacher_calls = sum(r["teacher_calls"] for r in self.rows)
        require(teacher_calls == n and all(self.ledger.calls[k]["returned"] == n for k in ("teacher_score", "tensorflow_value")),
                "all deployed and annotation-only teacher calls")
        require(self.ledger.calls["backend_binding"]["returned"] == sum(r["backend_bindings"] for r in self.rows)
                == EPISODES
                and self.ledger.calls["annotation_public_reset"]["returned"] == 0
                and self.ledger.calls["annotation_public_update"]["returned"] == 0, "one binding per episode and no replay work")
        for channel in ("native_reset", "actor_construction"):
            require(self.ledger.calls[channel]["returned"] == EPISODES, "one operation per episode: " + channel)
        for channel in ("tensorflow_construction", "tensorflow_build", "tensorflow_load"):
            require(self.ledger.calls[channel]["returned"] == 1, "one physical native setup operation: " + channel)
        costs = {"original_setup_wall_seconds": self.original_setup_wall, "feature_module_seconds": self.feature_module_seconds,
            **timings, "journal_io_seconds": self.ledger.io_seconds, "operation_seconds": self.ledger.calls,
            "scope": "Setup and DEV collection and serialization are disjoint physical intervals. "
                     "Operation timers overlap; I/O timer is partial and overlaps those intervals. Source authentication, final "
                     "hashing and process overhead remain paid in original worker/parent time. Physical teacher annotations "
                     "are included; these are collection costs, not deployed-controller cost measurements."}
        summary = {"version": VERSION, "complete": True, "episodes": EPISODES, "rows": n,
            "phase": PHASE, "master_episodes": MASTER_EPISODES, "confirm_episodes": 0,
            "datasets": self.datasets, "teacher_calls": teacher_calls, "deployed_queries": sum(r["deployed_queries"] for r in self.rows),
            "annotation_only": sum(r["annotation_only"] for r in self.rows), "corrections": sum(r["corrections"] for r in self.rows),
            "found": sum(r["found"] for r in self.rows), "censored": sum(r["censored"] for r in self.rows),
            "by_stage_arm": [{"stage": stage, "arm": arm, "episodes": len(group),
                "rows": sum(r["rows"] for r in group), "deployed_queries": sum(r["deployed_queries"] for r in group),
                "annotation_only": sum(r["annotation_only"] for r in group)} for stage in (PHASE,) for arm in ARMS
                for group in [[r for r in self.rows if r["stage"] == stage and r["arm"] == arm]]],
            "costs": costs, "training_updates": 0, "efficacy_claim": False}
        write(self.out / "costs.json", costs)
        write(self.out / "summary.json", summary)

    def execute(self):
        exclusive_output(self.out)
        validate_worker_paths(self.args)
        self.out.mkdir(parents=False, exist_ok=False)
        def interrupt(_signum, _frame):
            raise InterruptedError("original readout-compute DEV collection supervisor stopped worker")
        signal.signal(signal.SIGTERM, interrupt)
        try:
            self.bind()
            self.body()
            self.ledger.close()
            for name, pin in self.plan["sources"].items():
                self.check()
                require(descriptor(ROOT / name)["sha256"] == pin, "unchanged source")
            for record in [*self.plan["inputs"].values(), *self.plan["native_inputs"].values()]:
                self.check()
                require(descriptor(ROOT / record["path"]) == {k: record[k] for k in ("sha256", "bytes")}, "unchanged input")
            require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256
                and descriptor(self.args.supervision)["sha256"] == self.receipt["supervision_sha256"], "unchanged plan/launch")
            require({p.name for p in self.out.iterdir()} == PAYLOADS, "exact completed payload closure")
            files = {name: descriptor(self.out / name) for name in sorted(PAYLOADS)}
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status="completed", complete=True, files=files, calls=self.ledger.calls,
                completed_episodes=len(self.rows), dev_episodes=EPISODES, confirm_episodes=0, master_episodes=MASTER_EPISODES,
                datasets=self.datasets, pending=self.ledger.pending, pending_emission=self.ledger.pending_emission,
                pending_episode=self.pending_episode, pending_action=self.pending_action, started_ns=self.start, finished_ns=finished,
                wall_seconds=(finished-self.start)/1e9, requires_successful_original_supervisor=True,
                journal_scope="Encoded returns are provisional within a pending episode; all journals flush/fsync before "
                              "durable episode row and completion boundary. Interrupted episodes are failures, never resumed.")
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "receipt": descriptor(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.receipt.update(status="failed", complete=False, error=repr(error), traceback=traceback.format_exc(),
                calls=self.ledger.calls, pending=self.ledger.pending, pending_emission=self.ledger.pending_emission,
                pending_episode=self.pending_episode, pending_action=self.pending_action, completed_episodes=len(self.rows),
                active_gate=getattr(self.active_actor, "progress", None))
            try:
                self.ledger.close(suppress=True)
            except BaseException as cleanup:  # noqa: BLE001 - preserve primary error
                self.receipt.setdefault("cleanup_errors", []).append({"error": repr(cleanup)})
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as publication:  # noqa: BLE001 - preserve primary error
                error.add_note(f"Failure publication: {publication!r}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    plan = sub.add_parser("plan")
    for role in sorted(ROLES):
        plan.add_argument("--" + role.replace("_", "-"), type=Path, required=True)
        plan.add_argument("--" + role.replace("_", "-") + "-sha256", required=True)
    plan.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    for name in ("plan", "supervision", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path(sys.executable).absolute() == ROOT / INTERPRETER and Path.cwd() == ROOT, "original native runtime/cwd")
    exclusive_output(args.output)
    if args.mode == "plan":
        freeze(args)
    else:
        validate_worker_paths(args)
        Run(args).execute()


if __name__ == "__main__":
    main()
