"""Bounded DEV-only frozen-feature evaluation and independent saved-output audit.

Only standard-library imports precede source, lineage, runtime and phase
authentication. Confirmation is deliberately unavailable in this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import resource
import signal
import sys
import time
import traceback
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/run_otto_residual.py"
VERSION = "otto-residual-execution-v1"
INTERPRETER = ".venv/bin/python"
COLLECTOR = "scripts/collect_otto_residual.py"
LINEAGE = "scripts/otto_residual_lineage.py"
OLD_PRODUCER = "scripts/train_otto_query_memory.py"
CLOCK = "src/openjev/research/suspend_clock.py"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SUPERVISOR_PIN = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
PROTOCOL = "research/otto-residual-estimator-protocol.md"
SEEDS = (309000001, 309000002, 309000003)
FAMILIES = ("pretrained", "joint_aux", "trace_delta")
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")
LIMITS = {"evaluate": {"seconds": 900, "rss_bytes": 4 * 1024**3, "output_bytes": 1024**3},
          "audit": {"seconds": 900, "rss_bytes": 4 * 1024**3, "output_bytes": 256 * 1024**2}}
ROLES = {"evaluate": ("collection_plan", "collection_receipt", "collection_terminal", "engineering"),
         "audit": ("producer_plan", "producer_receipt", "producer_terminal")}
COMPONENTS = {SELF, "tests/test_run_otto_residual.py", COLLECTOR, "tests/test_collect_otto_residual.py",
              LINEAGE, "tests/test_otto_residual_lineage.py", "scripts/qualify_otto_residual_capacity.py",
              "src/openjev/research/otto_residual_audit.py", "tests/test_otto_residual_audit.py", PROTOCOL}
COMPONENTS |= {f"{prefix}/otto_residual_{name}.py" for prefix in ("src/openjev/research",)
               for name in ("contract", "features", "replay", "gate")}
COMPONENTS |= {f"tests/test_otto_residual_{name}.py" for name in ("contract", "features", "replay", "gate")}
CONFIG = {"phase_scope": "dev only", "episodes": 18, "master_episodes": 54, "fit_seeds": list(SEEDS),
          "methods": 9, "views": 72, "taus": [.01, .1, 1., 10.], "query_period": 4,
          "cache_builds": 3, "model_constructions": 6, "checkpoint_decodes": 9,
          "batch": 1, "chunk": 32, "gradient_updates": 0, "confirm_admitted": False,
          "audit": "exact estimator replay plus independent scalar metrics and gate; no neural calls"}
COMMON_PAYLOADS = {"started.json", "runtime.json", "numerical.json", "progress.jsonl", "summary.json"}
PAYLOADS = {"evaluate": COMMON_PAYLOADS | {"views.json"}
            | {f"cache-{seed}.{suffix}" for seed in SEEDS for suffix in ("npz", "json")}
            | {f"prediction-{seed}-{index:02}.npz" for seed in SEEDS for index in range(24)},
            "audit": COMMON_PAYLOADS | {"audit.json"}}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def regular(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_file() and path.is_relative_to(ROOT) and ".." not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents)), "regular contained input")
    return path


def exclusive_output(value):
    path = Path(value)
    require(path.is_absolute() and path.is_relative_to(ROOT) and ".." not in path.parts
            and not path.exists() and not path.is_symlink() and path.parent.is_dir()
            and not any(p.is_symlink() for p in path.parents), "exclusive contained output with regular parents")
    return path


def descriptor(value):
    path = regular(value)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read(value):
    return json.loads(regular(value).read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime_record():
    return {"python": sys.version, "executable": sys.executable,
            "distributions": dict(sorted((x.metadata["Name"], x.version)
                                          for x in importlib.metadata.distributions()))}


def verify_records(records):
    for record in records.values():
        require(set(record) == {"path", "sha256", "bytes"}
                and descriptor(record["path"]) == {k: record[k] for k in ("sha256", "bytes")},
                "exact input descriptor before decode")


def same_record(left, right):
    return (regular(left["path"]) == regular(right["path"])
            and {k: left[k] for k in ("sha256", "bytes")} == {k: right[k] for k in ("sha256", "bytes")})


def verify_sources(sources):
    require(COMPONENTS <= set(sources), "complete new implementation source closure")
    for path, expected in sources.items():
        require(not Path(path).is_absolute() and descriptor(path)["sha256"] == expected,
                "unchanged bound source: " + path)


def engineering(path):
    record = read(path)
    require(record["status"] == "passed" and record["source_before"] == record["source_after"]
            and record["sources_unchanged"] is True and record["commands"]
            and all(row["returncode"] == 0 and row["timed_out"] is False
                    and row["reaped"] is True and row["group_absent"] is True for row in record["commands"]),
            "complete engineering qualification")
    sources = {}
    for name, expected in record["source_after"].items():
        require(descriptor(name) == expected, "qualified source descriptor")
        sources[name] = expected["sha256"]
    verify_sources(sources)
    directory = regular(path).parent
    actual = set()
    for entry in directory.iterdir():
        if entry.name == "pytest-temp":
            require(entry.is_dir() and not entry.is_symlink(), "bounded pytest fixture directory")
        else:
            require(entry.is_file() and not entry.is_symlink(), "regular engineering payload")
            actual.add(entry.name)
    require(actual == set(record["files"]) | {"receipt.json"}, "engineering file closure")
    for name, expected in record["files"].items():
        require(Path(name).name == name and descriptor(directory / name) == expected, "engineering payload pin")
    require("capacity.json" in record["files"], "recorded integrated capacity qualification")
    capacity = read(directory / "capacity.json")
    require(capacity["status"] == "passed" and capacity["empirical_array_decodes"] == 0
            and capacity["checkpoint_decodes"] == 0 and capacity["teacher_calls"] == capacity["native_calls"] == 0,
            "synthetic-only capacity qualification")
    for phase in LIMITS:
        seconds = capacity[phase + "_seconds"]
        require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0
                and capacity[phase + "_projected_seconds"] == 2 * 3 * seconds + 120 <= 675,
                "fixed prospective capacity threshold")
    require(capacity["episodes"] == 18 and capacity["length"] == 2188
            and capacity["views"] == 24 and capacity["fit_seeds_measured"] == 1,
            "complete worst-horizon fabricated capacity roster")
    return sources


def authenticate_collection(inputs, sources):
    require(set(inputs) == set(ROLES["evaluate"]), "exact evaluation evidence roles")
    verify_records(inputs)
    collector = load(COLLECTOR, "_residual_collector_metadata")
    plan, receipt = read(inputs["collection_plan"]["path"]), read(inputs["collection_receipt"]["path"])
    require(plan["version"] == receipt["version"] == collector.VERSION
            and plan["status"] == "frozen_before_collection" and plan["phase"] == "dev"
            and plan["cohort"] == collector.cohort() and plan["execution_cohort"] == collector.cohort("dev")
            and plan["configuration"] == collector.CONFIGURATION and plan["limits"] == collector.LIMITS
            and plan["call_caps"] == collector.CALL_CAPS and set(plan["payloads"]) == collector.PAYLOADS
            and same_record(plan["inputs"]["engineering"], inputs["engineering"]), "same qualified fresh DEV collection")
    authenticated, runtime, native_inputs, _, _ = collector.authenticate_inputs(plan["inputs"])
    require(plan["sources"] == authenticated and all(sources.get(k) == v for k, v in authenticated.items())
            and plan["runtime"] == runtime
            and plan["native_inputs"] == {k: {"path": str(p.relative_to(ROOT)), **descriptor(p)}
                                          for k, p in native_inputs.items()}, "native/source/runtime collection identity")
    require(receipt["status"] == "completed" and receipt["complete"] is True
            and receipt["phase"] == "dev" and receipt["training_updates"] == 0 and receipt["old_test_array_decodes"] == 0
            and receipt["plan_sha256"] == inputs["collection_plan"]["sha256"]
            and receipt["sources"] == plan["sources"] and receipt["inputs"] == plan["inputs"]
            and receipt["native_inputs"] == plan["native_inputs"] and receipt["limits"] == collector.LIMITS
            and receipt["completed_episodes"] == receipt["dev_episodes"] == 18
            and receipt["confirm_episodes"] == 0 and receipt["master_episodes"] == 54
            and receipt["pending"] == [] and receipt["pending_episode"] is None
            and receipt["pending_action"] is None and receipt["pending_emission"] is None
            and receipt["peak_rss_bytes"] <= collector.LIMITS["rss_bytes"]
            and receipt["requires_successful_original_supervisor"] is True, "complete DEV-only collector receipt")
    require(set(receipt["calls"]) == set(collector.CALL_CAPS)
            and all(type(row["returned"]) is int and row["attempted"] == row["returned"] and 0 <= row["returned"] <= collector.CALL_CAPS[key]
                    for key, row in receipt["calls"].items()), "complete bounded collection work")
    helper = load(OLD_PRODUCER, "_residual_original_process_helpers")
    directory = helper.closed_files(inputs["collection_receipt"]["path"], receipt, collector.PAYLOADS)
    helper.successful_process(inputs, "collection", receipt, COLLECTOR,
                              ".venv-otto-released-native/bin/python", 3600)
    return plan, receipt, directory


def authenticate(phase, inputs):
    require(phase in ROLES and set(inputs) == set(ROLES[phase]), "DEV evaluation or audit only")
    verify_records(inputs)
    if phase == "evaluate":
        sources = engineering(inputs["engineering"]["path"])
        lineage = load(LINEAGE, "_residual_checkpoint_lineage").authenticate()
        require(all(sources.get(k) == v for k, v in lineage["sources"].items()), "unchanged old source lineage")
        collection = authenticate_collection(inputs, sources)
        require(runtime_record() == lineage["runtime"], "original fixed numerical runtime")
        return {"sources": sources, "lineage": lineage, "collection": collection, "producer": None}
    producer_plan = read(inputs["producer_plan"]["path"])
    require(producer_plan["version"] == VERSION and producer_plan["phase"] == "evaluate"
            and producer_plan["status"] == "frozen_before_execution" and producer_plan["configuration"] == CONFIG
            and producer_plan["limits"] == LIMITS["evaluate"]
            and set(producer_plan["payloads"]) == PAYLOADS["evaluate"], "new DEV producer plan only")
    bound = authenticate("evaluate", producer_plan["inputs"])
    require(producer_plan["sources"] == bound["sources"] and producer_plan["lineage"] == bound["lineage"]
            and producer_plan["runtime"] == runtime_record(), "producer's exact frozen identities")
    receipt = read(inputs["producer_receipt"]["path"])
    require(receipt["version"] == VERSION and receipt["phase"] == "evaluate"
            and receipt["status"] == "completed" and receipt["complete"] is True
            and receipt["plan_sha256"] == inputs["producer_plan"]["sha256"]
            and receipt["sources"] == bound["sources"] and receipt["inputs"] == producer_plan["inputs"]
            and receipt["limits"] == LIMITS["evaluate"] and receipt["pending"] is None
            and receipt["pending_io"] is None and receipt["pending_emission"] is None and receipt["pending_log"] is None
            and receipt["views_completed"] == 72 and receipt["caches_completed"] == 3
            and receipt["counts"]["checkpoint_decodes"] == 9 and receipt["counts"]["model_constructions"] == 6
            and receipt["counts"]["model_construction_attempts"] == 6 and receipt["counts"]["array_decodes"] == 85
            and receipt["counts"]["cache_builds"] == 3 and receipt["counts"]["replays"] == 72
            and all(receipt["counts"][name] == 0 for name in ("teacher_calls", "native_calls", "optimizer_steps", "confirm_decodes"))
            and receipt["peak_rss_bytes"] <= LIMITS["evaluate"]["rss_bytes"]
            and receipt["requires_successful_original_supervisor"] is True, "complete original DEV producer")
    helper = load(OLD_PRODUCER, "_residual_producer_process_helpers")
    directory = helper.closed_files(inputs["producer_receipt"]["path"], receipt, PAYLOADS["evaluate"])
    parent = helper.successful_process(inputs, "producer", receipt, SELF, INTERPRETER, 900)
    require(read(producer_plan["inputs"]["collection_terminal"]["path"])["finished_ns"] <= parent["started_ns"],
            "evaluation starts after original DEV collection closure")
    bound["producer"] = (producer_plan, receipt, directory)
    return bound


def freeze(args):
    exclusive_output(args.output)
    require(args.phase in ROLES, "DEV-only phase")
    inputs = {}
    for role in ROLES[args.phase]:
        path = regular(getattr(args, role))
        pin = descriptor(path)
        require(pin["sha256"] == getattr(args, role + "_sha256"), "external planning pin")
        inputs[role] = {"path": str(path), **pin}
    bound = authenticate(args.phase, inputs)
    plan = {"version": VERSION, "phase": args.phase, "status": "frozen_before_execution",
            "configuration": CONFIG, "inputs": inputs, "sources": bound["sources"], "lineage": bound["lineage"],
            "runtime": runtime_record(), "limits": LIMITS[args.phase], "payloads": sorted(PAYLOADS[args.phase])}
    write(args.output, plan)
    print(json.dumps({"status": plan["status"], "phase": args.phase, "plan": descriptor(args.output)}), flush=True)


def close(args):
    """Publish a DEV decision only after the original saved-audit process closes."""
    exclusive_output(args.output)
    inputs = {}
    for role in ("audit_plan", "audit_receipt", "audit_terminal"):
        path = regular(getattr(args, role))
        pin = descriptor(path)
        require(pin["sha256"] == getattr(args, role + "_sha256"), "external closure pin")
        inputs[role] = {"path": str(path), **pin}
    plan, receipt = read(inputs["audit_plan"]["path"]), read(inputs["audit_receipt"]["path"])
    require(plan["version"] == VERSION and plan["phase"] == "audit" and plan["status"] == "frozen_before_execution"
            and plan["configuration"] == CONFIG and plan["limits"] == LIMITS["audit"]
            and set(plan["payloads"]) == PAYLOADS["audit"] and plan["runtime"] == runtime_record(), "original DEV audit plan")
    bound = authenticate("audit", plan["inputs"])
    require(plan["sources"] == bound["sources"] and plan["lineage"] == bound["lineage"], "audit's exact source and lineage identities")
    require(receipt["version"] == VERSION and receipt["phase"] == "audit"
            and receipt["status"] == "completed" and receipt["complete"] is True
            and receipt["plan_sha256"] == inputs["audit_plan"]["sha256"] and receipt["inputs"] == plan["inputs"]
            and receipt["sources"] == plan["sources"] and receipt["limits"] == LIMITS["audit"]
            and receipt["views_completed"] == 72 and receipt["caches_completed"] == 3
            and all(receipt[name] is None for name in ("pending", "pending_io", "pending_emission", "pending_log"))
            and receipt["peak_rss_bytes"] <= LIMITS["audit"]["rss_bytes"]
            and receipt["requires_successful_original_supervisor"] is True, "complete original independent DEV audit")
    validate_counts("audit", receipt["counts"])
    helper = load(OLD_PRODUCER, "_residual_audit_process_helpers")
    directory = helper.closed_files(inputs["audit_receipt"]["path"], receipt, PAYLOADS["audit"])
    parent = helper.successful_process(inputs, "audit", receipt, SELF, INTERPRETER, 900)
    require(read(plan["inputs"]["producer_terminal"]["path"])["finished_ns"] <= parent["started_ns"],
            "independent audit starts after original producer closure")
    summary, audit = read(directory / "summary.json"), read(directory / "audit.json")
    decision = audit["decision"]
    require(audit["version"] == summary["version"] == VERSION and len(audit["reports"]) == len(audit["verified_replays"]) == 72
            and summary["decision"] == decision and summary["counts"] == receipt["counts"] and summary["phase"] == "audit"
            and audit["requires_successful_original_supervisor"] is True and decision["stage"] == "dev"
            and decision["candidate"] == "rls_full" and decision["selected_tau"] in (.01, .1, 1., 10.)
            and decision["technical_complete"] is True and decision["reports"] == 72
            and decision["total_conditions"] == len(decision["conditions"]) == 13
            and all(type(row["passed"]) is bool for row in decision["conditions"])
            and decision["passed_conditions"] == sum(row["passed"] for row in decision["conditions"])
            and type(decision["passed"]) is bool and decision["passed"] == all(row["passed"] for row in decision["conditions"]),
            "complete independently audited provisional decision")
    result = {"version": VERSION, "status": "DEV_PASS" if decision["passed"] else "DEV_FAIL",
              "inputs": inputs, "decision": decision, "technical_complete": True,
              "original_audit_seconds": (parent["finished_ns"] - parent["started_ns"]) / 1e9,
              "audit_output": {"path": str(directory / "audit.json"), **descriptor(directory / "audit.json")},
              "confirmation_eligible_for_separate_registration": decision["passed"], "confirmation_execution_admitted": False,
              "old_test_access": False, "numerical_array_decodes": 0, "model_calls": 0,
              "scope": "fixed-path DEV selection result; no held-out, autonomous, calibration or novelty claim"}
    write(args.output, result)
    print(json.dumps({"status": result["status"], "passed_conditions": decision["passed_conditions"],
                      "closure": descriptor(args.output)}), flush=True)


def equal_nested(actual, expected, label="value"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + " fields")
        for key, value in expected.items():
            equal_nested(actual[key], value, label + "." + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + " list")
        for index, value in enumerate(expected):
            equal_nested(actual[index], value, label + f"[{index}]")
    elif type(expected) is float:
        require(type(actual) in (float, int) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label + " numeric parity")
    else:
        require(type(actual) is type(expected) and actual == expected, label + " exact parity")


def validate_checkpoints(np, arrays, shapes):
    """Validate all nine supplied files before any model is constructed."""
    require(set(arrays) == {(family, seed) for family in FAMILIES for seed in SEEDS}, "all nine checkpoint arrays")
    slow_names = {"slow." + name for name in shapes}
    for (family, _seed), state in arrays.items():
        expected = slow_names | ({"projection.weight"} if family == "trace_delta" else set())
        require(set(state) == expected, "exact selected checkpoint tensor fields")
        for name, value in state.items():
            shape = (8, 28) if name == "projection.weight" else shapes[name.removeprefix("slow.")]
            require(isinstance(value, np.ndarray) and value.dtype == np.float32 and value.shape == tuple(shape)
                    and bool(np.isfinite(value).all()), "finite checkpoint dtype and shape")
    for seed in SEEDS:
        for name in slow_names:
            require(arrays["pretrained", seed][name].tobytes() == arrays["trace_delta", seed][name].tobytes(),
                    "same-seed projection retains unchanged pretrained slow weights")


def expected_cache_work(offsets):
    lengths = [int(high - low) for low, high in pairwise(offsets)]
    require(lengths and all(1 <= n <= 2188 for n in lengths), "bounded complete cache episodes")
    rows, episodes = sum(lengths), len(lengths)
    queries = sum((n - 1) // 4 + 1 for n in lengths)
    later, keys = queries - episodes, rows - episodes
    groups = sum((n - 2) // 4 + 1 if n > 1 else 0 for n in lengths)
    chunks = sum((n + 31) // 32 for n in lengths)
    counts = {"pretrained_forward_chunks": chunks, "joint_forward_chunks": chunks,
              "projection_calls": keys, "projection_rows": keys, "projection_key_rows": keys,
              "cue_key_normalizations": keys, "cue_normalizations": keys, "cue_trace_updates": keys,
              "projection_linear_terms": 224 * keys, "cue_normalized_coordinates": 16 * keys,
              "cue_trace_mixed_coordinates": 8 * keys}
    slow = {"active_rows": rows, "query_rows": queries, "later_query_rows": later, "key_rows": keys,
            "nonquery_rows": rows - queries, "recurrent_calls": episodes + 2 * later + groups,
            "recurrent_token_transitions": rows + later, "base_readout_calls": later + groups,
            "base_readout_rows": keys, "action_readout_calls": groups, "action_readout_rows": rows - queries,
            "shadow_readout_calls": later, "shadow_readout_rows": later}
    for prefix in ("pretrained_slow_", "joint_slow_"):
        counts.update({prefix + key: value for key, value in slow.items()})
    return counts


def validate_counts(phase, counts):
    expected = {"checkpoint_decodes": 9 if phase == "evaluate" else 0,
                "array_decodes": 85 if phase == "evaluate" else 76,
                "model_constructions": 6 if phase == "evaluate" else 0,
                "model_construction_attempts": 6 if phase == "evaluate" else 0,
                "cache_builds": 3 if phase == "evaluate" else 0, "replays": 72,
                "teacher_calls": 0, "native_calls": 0, "optimizer_steps": 0, "confirm_decodes": 0}
    require(phase in LIMITS and counts == expected and all(type(v) is int for v in counts.values()),
            "exact complete DEV operation counts")


def report_view(metrics, history, identities, scores, method, tau, seed, contract, check):
    reports = []
    for identity, (low, high) in zip(identities, pairwise(history["episode_offsets"]), strict=True):
        check()
        reports.append(metrics.episode_metrics(identity, 4, history["targets"][low:high],
                                               history["legal"][low:high], scores[low:high]))
    return metrics.aggregate_episodes(reports, family=contract.view_name(method, tau), fit_seed=seed,
                                      expected_identities=identities)


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.launch = self.start = self.plan = self.bound = None
        self.next_resource_poll_ns = 0
        self.receipt = {"version": VERSION, "status": "started", "complete": False, "pending": None,
                        "pending_io": None, "pending_emission": None, "pending_log": None,
                        "views_completed": 0, "caches_completed": 0,
                        "counts": {name: 0 for name in ("checkpoint_decodes", "array_decodes", "model_constructions", "model_construction_attempts",
                                    "cache_builds", "replays", "teacher_calls", "native_calls", "optimizer_steps", "confirm_decodes")}}

    def check(self, *, force=False):
        now = self.clock.now_ns()
        require(now < self.launch["deadline_ns"], "original fixed deadline")
        # Keep every callback's deadline check. Filesystem/RSS scans are bounded
        # to four per second, plus explicit IO boundaries, rather than per row.
        if not force and now < self.next_resource_poll_ns:
            return
        self.next_resource_poll_ns = now + 250_000_000
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= self.plan["limits"]["rss_bytes"], "fixed RSS cap")
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
                <= self.plan["limits"]["output_bytes"] - 1024**2, "output cap with failure reserve")

    def event(self, row):
        self.check(force=True)
        self.receipt["pending_log"] = dict(row)
        with (self.out / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.receipt["pending_log"] = None

    def bind(self):
        require(descriptor(CLOCK)["sha256"] == CLOCK_PIN and descriptor(SUPERVISOR)["sha256"] == SUPERVISOR_PIN,
                "original supervisor and continuous clock")
        self.clock = load(CLOCK, "_residual_worker_clock").SuspendClock()
        self.start = self.clock.now_ns()
        regular(self.args.plan)
        require(self.args.supervision.is_absolute() and self.args.supervision.is_relative_to(ROOT)
                and ".." not in self.args.supervision.parts and self.args.supervision.parent.is_dir()
                and not any(p.is_symlink() for p in (self.args.supervision, *self.args.supervision.parents)),
                "contained original supervision path")
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, "original launch available")
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256, "external plan pin")
        self.plan = read(self.args.plan)
        phase = self.plan["phase"]
        require(phase in ROLES and self.plan["version"] == VERSION and self.plan["status"] == "frozen_before_execution"
                and self.plan["configuration"] == CONFIG and self.plan["limits"] == LIMITS[phase]
                and set(self.plan["payloads"]) == PAYLOADS[phase], "registered DEV-only configuration")
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and command[:3] == [str(ROOT / INTERPRETER), str(ROOT / SELF), "run"]
                and self.launch["pid"] == os.getpid() == self.launch["pgid"]
                and self.launch["parent_pid"] == os.getppid() != os.getpid()
                and Path(self.launch["cwd"]) == Path.cwd() == ROOT and self.launch["cap_seconds"] == 900
                and self.launch["clock_backend"] == self.clock.backend
                and self.launch["clock_source_sha256"] == CLOCK_PIN and self.launch["watchdog_sha256"] == SUPERVISOR_PIN
                and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
                and self.launch["deadline_ns"] == self.launch["started_ns"] + 900 * 10**9,
                "canonical original bounded parent")
        self.check()
        require(self.plan["runtime"] == runtime_record() and all(os.environ.get(name) == "1" for name in THREADS),
                "fixed numerical runtime and CPU threads")
        self.bound = authenticate(phase, self.plan["inputs"])
        require(self.bound["sources"] == self.plan["sources"] and self.bound["lineage"] == self.plan["lineage"],
                "all frozen source and lineage bindings")
        require(phase != "audit" or ("torch" not in sys.modules and "tensorflow" not in sys.modules),
                "audit process has no neural runtime")
        self.receipt.update(phase=phase, sources=self.plan["sources"], inputs=self.plan["inputs"],
                            limits=self.plan["limits"], plan_sha256=self.args.plan_sha256,
                            supervision_sha256=descriptor(self.args.supervision)["sha256"])
        write(self.out / "started.json", {"started_ns": self.start, "launch": self.launch,
              "request": {key: str(value) for key, value in vars(self.args).items()}})
        write(self.out / "runtime.json", runtime_record())
        self.event({"event": "admitted", "phase": phase, "before_numerical_imports": True})

    def decode(self, path, expected, *, checkpoint=False):
        self.check(force=True)
        require(descriptor(path) == expected, "authenticated array bytes immediately before decode")
        self.receipt["pending_io"] = {"operation": "decode", "path": str(path), "parent": self.receipt["pending"]}
        self.event({"event": "attempt", **self.receipt["pending_io"]})
        with self.np.load(path, allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), "unique archive fields")
            values = {name: archive[name] for name in archive.files}
        self.receipt["counts"]["array_decodes"] += 1
        self.receipt["counts"]["checkpoint_decodes"] += int(checkpoint)
        self.event({"event": "return", **self.receipt["pending_io"]})
        self.receipt["pending_io"] = None
        return values

    def save_arrays(self, name, values):
        self.check(force=True)
        path = self.out / name
        self.receipt["pending_emission"] = {"operation": "save_arrays", "path": str(path), "parent": self.receipt["pending"]}
        with path.open("xb") as stream:
            self.np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        pin = descriptor(path)
        saved = self.decode(path, pin)
        require(set(saved) == set(values) and all(saved[key].dtype == value.dtype
                and saved[key].shape == value.shape and saved[key].tobytes() == value.tobytes()
                for key, value in values.items()), "exact saved array roundtrip")
        self.receipt["pending_emission"] = None
        return pin

    def inputs(self):
        import numpy as np

        from openjev.research import otto_query_memory_data as data
        self.np = np
        collection_plan, receipt, directory = self.bound["collection"]
        identities = collection_plan["execution_cohort"]
        require(len(identities) == 18 and all(row["stage"] == "dev" for row in identities), "DEV only before path creation")
        flat = self.decode(directory / "dev.npz", receipt["files"]["dev.npz"])
        return data.project_census(flat, identities, query_period=4, expected_stage="dev"), identities

    def evaluate(self, history, identities):
        import torch

        from openjev.research import otto_query_memory_metrics as metrics
        from openjev.research import otto_residual_contract as contract
        from openjev.research import otto_residual_features as features
        from openjev.research import otto_residual_gate as gate
        from openjev.research import otto_residual_replay as replay
        from openjev.research import otto_scheduled_predictor as predictor
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        require(torch.get_num_threads() == torch.get_num_interop_threads() == 1
                and torch.are_deterministic_algorithms_enabled(), "fixed deterministic one-thread Torch runtime")
        write(self.out / "numerical.json", {"numpy": self.np.__version__, "torch": torch.__version__,
              "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "device": "cpu"})
        states = {}
        for family in FAMILIES:
            for seed in SEEDS:
                record = self.bound["lineage"]["checkpoints"][family][str(seed)]
                states[family, seed] = self.decode(record["path"], {k: record[k] for k in ("bytes", "sha256")}, checkpoint=True)
        validate_checkpoints(self.np, states, predictor.STATE_SHAPES)
        reports, views, cache_records = [], [], []
        for seed in SEEDS:
            self.receipt["pending"] = {"operation": "cache", "seed": seed}
            self.event({"event": "attempt", **self.receipt["pending"]})
            tick = self.clock.now_ns()
            models = []
            for family in ("pretrained", "joint_aux"):
                self.receipt["counts"]["model_construction_attempts"] += 1
                self.event({"event": "model_construction_attempt", "family": family, "seed": seed})
                models.append(predictor.from_state("frozen", seed, 4, {name.removeprefix("slow."): torch.from_numpy(value.copy())
                              for name, value in states[family, seed].items()}))
                self.receipt["counts"]["model_constructions"] += 1
                self.event({"event": "model_construction_return", "family": family, "seed": seed})
            projection = torch.from_numpy(states["trace_delta", seed]["projection.weight"].copy())
            cache = features.build_cache(*models, projection, history["features"], history["query_scores"], history["episode_offsets"])
            self.receipt["counts"]["cache_builds"] += 1
            require(cache["work_counts"] == expected_cache_work(cache["episode_offsets"]), "complete schedule-derived cache accounting")
            arrays = {k: v for k, v in cache.items() if isinstance(v, self.np.ndarray)}
            pin = self.save_arrays(f"cache-{seed}.npz", arrays)
            metadata = {k: v for k, v in cache.items() if k not in arrays}
            write(self.out / f"cache-{seed}.json", metadata)
            cache_records.append({"seed": seed, "array": pin, "work_counts": cache["work_counts"],
                                  "seconds": (self.clock.now_ns() - tick) / 1e9})
            self.event({"event": "cache_complete", "record": cache_records[-1]})
            self.receipt["caches_completed"] += 1
            self.receipt["pending"] = None
            for index, (method, tau, _seed) in enumerate(spec for spec in contract.view_specs("dev") if spec[2] == seed):
                self.receipt["pending"] = {"operation": "view", "seed": seed, "method": method, "tau": tau}
                self.event({"event": "attempt", **self.receipt["pending"]})
                tick = self.clock.now_ns()
                result = replay.replay(cache, method, tau=tau)
                self.receipt["counts"]["replays"] += 1
                self.event({"event": "replay_return", **self.receipt["pending"],
                            "work_counts": result["work_counts"], "state_bytes": result["state_bytes"]})
                name = f"prediction-{seed}-{index:02}.npz"
                arrays = {k: v for k, v in result.items() if isinstance(v, self.np.ndarray)}
                arrays["episode_offsets"] = cache["episode_offsets"].copy()
                pin = self.save_arrays(name, arrays)
                report = report_view(metrics, history, identities, result["action_scores"], method, tau, seed, contract, self.check)
                reports.append(report)
                views.append({"seed": seed, "method": method, "tau": tau, "path": name, "file": pin,
                              "work_counts": result["work_counts"], "state_bytes": result["state_bytes"],
                              "seconds": (self.clock.now_ns() - tick) / 1e9, "report": report})
                self.event({"event": "view_complete", "record": views[-1]})
                self.receipt["views_completed"] += 1
                self.receipt["pending"] = None
            print(json.dumps({"phase": "evaluate", "completed_views": len(views), "total": 72}), flush=True)
        decision = gate.evaluate(reports, identities, stage="dev", technical_complete=False)
        write(self.out / "views.json", {"version": VERSION, "views": views, "caches": cache_records})
        write(self.out / "summary.json", {"version": VERSION, "phase": "evaluate", "decision": decision,
              "provisional": True, "scope": "requires independent saved-output audit and original closures",
              "counts": self.receipt["counts"]})

    def audit(self, history, identities):
        from openjev.research import otto_residual_audit as independent
        from openjev.research import otto_residual_contract as contract
        from openjev.research import otto_residual_replay as replay
        write(self.out / "numerical.json", {"numpy": self.np.__version__, "neural_runtime_used": False})
        _plan, receipt, directory = self.bound["producer"]
        saved = read(directory / "views.json")
        expected = [(method, tau, seed) for seed in SEEDS for method, tau, s in contract.view_specs("dev") if s == seed]
        require(saved["version"] == VERSION and [(r["method"], r["tau"], r["seed"]) for r in saved["views"]] == expected,
                "all ordered 72 saved views")
        require([row["seed"] for row in saved["caches"]] == list(SEEDS), "all three ordered cache records")
        reports, verified = [], []
        for seed in SEEDS:
            cache = self.decode(directory / f"cache-{seed}.npz", receipt["files"][f"cache-{seed}.npz"])
            metadata = read(directory / f"cache-{seed}.json")
            require(not set(cache) & set(metadata), "disjoint cache arrays and metadata")
            cache.update(metadata)
            contract.validate_cache(cache)
            cache_record = saved["caches"][list(SEEDS).index(seed)]
            require(cache_record["array"] == receipt["files"][f"cache-{seed}.npz"]
                    and cache_record["work_counts"] == cache["work_counts"] == expected_cache_work(cache["episode_offsets"])
                    and type(cache_record["seconds"]) in (float, int) and math.isfinite(cache_record["seconds"])
                    and cache_record["seconds"] >= 0, "authenticated schedule-derived cache accounting")
            require(cache["episode_offsets"].tobytes() == history["episode_offsets"].tobytes()
                    and cache["query_scores"].tobytes() == history["query_scores"].tobytes(), "cache matches projected visible history")
            self.event({"event": "cache_verified", "seed": seed, "work_counts": cache["work_counts"]})
            self.receipt["caches_completed"] += 1
            for view in (r for r in saved["views"] if r["seed"] == seed):
                self.receipt["pending"] = {"operation": "audit_view", "seed": seed, "method": view["method"], "tau": view["tau"]}
                self.event({"event": "attempt", **self.receipt["pending"]})
                require(Path(view["path"]).name == view["path"] and view["file"] == receipt["files"][view["path"]], "saved view path pin")
                arrays = self.decode(directory / view["path"], view["file"])
                repeated = replay.replay(cache, view["method"], tau=view["tau"])
                self.receipt["counts"]["replays"] += 1
                expected_arrays = {k: v for k, v in repeated.items() if isinstance(v, self.np.ndarray)}
                expected_arrays["episode_offsets"] = cache["episode_offsets"]
                require(set(arrays) == set(expected_arrays) and all(arrays[k].dtype == v.dtype
                        and arrays[k].shape == v.shape and arrays[k].tobytes() == v.tobytes()
                        for k, v in expected_arrays.items()), "exact saved estimator replay")
                require(repeated["work_counts"] == view["work_counts"] and repeated["state_bytes"] == view["state_bytes"],
                        "saved estimator operation and state accounting")
                report = independent.report_view(identities, history["targets"], history["legal"], arrays["action_scores"],
                          history["episode_offsets"], view["method"], view["tau"], seed, check=self.check)
                equal_nested(view["report"], report, "independent saved action metrics")
                reports.append(report)
                verified.append({"seed": seed, "method": view["method"], "tau": view["tau"],
                                 "work_counts": repeated["work_counts"], "state_bytes": repeated["state_bytes"]})
                self.event({"event": "audit_view_complete", "record": verified[-1], "report": report})
                self.receipt["views_completed"] += 1
                self.receipt["pending"] = None
            print(json.dumps({"phase": "audit", "completed_views": len(reports), "total": 72}), flush=True)
        provisional = independent.evaluate_reports(reports, identities, stage="dev", technical_complete=False, check=self.check)
        equal_nested(read(directory / "summary.json")["decision"], provisional, "independent selection and conditions")
        decision = independent.evaluate_reports(reports, identities, stage="dev", technical_complete=True, check=self.check)
        write(self.out / "audit.json", {"version": VERSION, "reports": reports, "verified_replays": verified,
              "decision": decision, "requires_successful_original_supervisor": True,
              "limits": "Recurrent cache correctness and causality are source-tested; audit replays saved caches without neural calls."})
        write(self.out / "summary.json", {"version": VERSION, "phase": "audit", "decision": decision,
              "provisional": True, "scope": "decision valid only after genuine original audit supervisor closure",
              "counts": self.receipt["counts"]})

    def execute(self):
        exclusive_output(self.out)
        self.out.mkdir(parents=False, exist_ok=False)

        def interrupt(_signum, _frame):
            raise InterruptedError("original bounded supervisor stopped worker")

        signal.signal(signal.SIGTERM, interrupt)
        try:
            self.bind()
            history, identities = self.inputs()
            getattr(self, self.plan["phase"])(history, identities)
            self.check(force=True)
            require(all(self.receipt[name] is None for name in ("pending", "pending_io", "pending_emission", "pending_log"))
                    and self.receipt["views_completed"] == 72 and self.receipt["caches_completed"] == 3, "all 72 views complete")
            validate_counts(self.plan["phase"], self.receipt["counts"])
            verify_sources(self.plan["sources"])
            verify_records(self.plan["inputs"])
            require(self.bound["lineage"] == load(LINEAGE, "_residual_lineage_after").authenticate(), "unchanged checkpoint lineage")
            require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256
                    and descriptor(self.args.supervision)["sha256"] == self.receipt["supervision_sha256"], "unchanged plan and launch")
            require({p.name for p in self.out.iterdir()} == PAYLOADS[self.plan["phase"]], "exact completed payload closure")
            files = {name: descriptor(self.out / name) for name in sorted(PAYLOADS[self.plan["phase"]])}
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status="completed", complete=True, files=files, started_ns=self.start,
                                finished_ns=finished, wall_seconds=(finished - self.start) / 1e9,
                                requires_successful_original_supervisor=True)
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "phase": self.plan["phase"], "receipt": descriptor(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.receipt.update(status="failed", complete=False, error=repr(error), traceback=traceback.format_exc())
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as publication:  # noqa: BLE001 - preserve original failure
                error.add_note(f"Failure publication: {publication!r}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--phase", choices=tuple(ROLES), required=True)
    for role in sorted(set().union(*ROLES.values())):
        plan.add_argument("--" + role.replace("_", "-"), type=Path)
        plan.add_argument("--" + role.replace("_", "-") + "-sha256")
    plan.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    for name in ("plan", "supervision", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--plan-sha256", required=True)
    closing = sub.add_parser("close")
    for role in ("audit_plan", "audit_receipt", "audit_terminal"):
        closing.add_argument("--" + role.replace("_", "-"), type=Path, required=True)
        closing.add_argument("--" + role.replace("_", "-") + "-sha256", required=True)
    closing.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(Path(sys.executable).absolute() == ROOT / INTERPRETER and Path.cwd() == ROOT, "fixed interpreter and cwd")
    require(args.output.is_absolute(), "absolute exclusive output")
    if args.mode == "plan":
        freeze(args)
    elif args.mode == "close":
        close(args)
    else:
        require(args.plan.is_absolute() and args.supervision.is_absolute(), "absolute worker bindings")
        Run(args).execute()


if __name__ == "__main__":
    main()
