"""Bounded saved-record arithmetic and journal audit; never scientific replay.

No evaluator, model, sampler, optimizer, Torch, NumPy or native simulator imports.
The only imported project code is the pinned nonscientific suspend clock.
"""
from __future__ import annotations

import argparse
import ast
import collections
import gzip
import hashlib
import importlib.util
import json
import math
import os
import resource
import signal
import struct
import sys
import time
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/audit_otto_query_gate.py"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
SUPERVISOR_PIN = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
VERSION = "otto-query-gate-saved-audit-v1"
LIMITS = {"seconds": 120, "rss_bytes": 1024**3, "output_bytes": 64 * 1024**2}
SEEDS, KINDS = (40101, 40102, 40103), ("gru32", "mlp190")
LEARNED = tuple(f"{kind}@{seed}" for seed in SEEDS for kind in KINDS)
ARMS = LEARNED + ("neural", "analytic")
FIRST = {"lambda3": 17100001, "lambda4": 17200001, "lambda5": 17300001}
METRICS = ("found", "steps", "queries", "init_seconds", "choose_seconds", "update_seconds",
           "setup_allocation_seconds", "controller_seconds", "environment_seconds")
PAYLOADS = {f"{name}.jsonl.gz" for name in ("work", "weights", "forwards", "gate-operations", "gate-decisions", "transitions")} | {
    "started.json", "runtime.json", "setup.json", "deployment.json", "episodes.jsonl", "summary.json"}
LIMITATIONS = [
    "Original neural values and native/public posterior equality remain inherited numerical evidence. No model or filter replay.",
    "Saved-score action selection, episode reductions and cost/query accounting are recomputed; timing truth remains the original instrumentation.",
    "Immediate completed phase files and process roots are authenticated; ancestral experiments are not re-audited.",
    "Agreement verifies the fixed small screen, not algorithmic novelty or general robustness.",
]


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class Audit:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = self.launch = None
        self.recording_failure = False
        self.counts = collections.Counter()
        self.receipt = {"version": VERSION, "status": "started", "agreement": False, "limits": LIMITS,
                        "model_calls": 0, "native_calls": 0, "sampler_calls": 0, "optimizer_calls": 0,
                        "limitations": LIMITATIONS, "failures": []}

    def require(self, value, message):
        self.counts["checks"] += 1
        if not value:
            raise ValueError(message)
        if not self.recording_failure and self.clock is not None and self.launch is not None and self.counts["checks"] % 256 == 0:
            self.check()

    def check(self):
        if self.clock.now_ns() >= self.launch["deadline_ns"]:
            raise TimeoutError("original saved-audit deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        if rss > LIMITS["rss_bytes"] or sum(p.stat().st_size for p in self.out.iterdir()) > LIMITS["output_bytes"]:
            raise MemoryError("saved-audit resource cap")

    def path(self, value):
        p = Path(value)
        p = p if p.is_absolute() else ROOT / p
        self.require(p.is_file() and p.is_relative_to(ROOT) and ".." not in p.parts
                     and not any(q.is_symlink() for q in (p, *p.parents)), "contained regular input")
        return p

    def digest(self, value):
        p = self.path(value)
        h = hashlib.sha256()
        with p.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                h.update(block)
                if not self.recording_failure and self.clock is not None and self.launch is not None:
                    self.check()
        return {"sha256": h.hexdigest(), "bytes": p.stat().st_size}

    def read(self, value):
        return json.loads(self.path(value).read_text())

    def rows(self, path):
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(self.path(path), "rt") as stream:
            for line in stream:
                self.require(len(line) <= 2 * 1024**2, "bounded saved JSON row")
                yield json.loads(line)

    def same(self, actual, expected, name):
        if isinstance(expected, dict):
            self.require(isinstance(actual, dict) and set(actual) == set(expected), name + " keys")
            for k, v in expected.items():
                self.same(actual[k], v, f"{name}.{k}")
        elif isinstance(expected, list):
            self.require(isinstance(actual, list) and len(actual) == len(expected), name + " length")
            for i, value in enumerate(expected):
                self.same(actual[i], value, f"{name}[{i}]")
        elif isinstance(expected, float):
            self.require(type(actual) in (int, float) and math.isfinite(actual)
                         and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12), name)
        else:
            self.require(type(actual) is type(expected) and actual == expected, name)

    def admit(self):
        self.require(self.digest(CLOCK)["sha256"] == CLOCK_PIN, "qualified clock before import")
        spec = importlib.util.spec_from_file_location("_query_saved_audit_clock", ROOT / CLOCK)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.clock = module.SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            self.require(self.clock.now_ns() - self.start < 5 * 10**9, "original launch available")
            time.sleep(.01)
        self.launch = self.read(self.args.supervision)
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        self.require(command == [sys.executable, *sys.argv] and self.launch["pid"] == os.getpid()
                     and self.launch["pgid"] == os.getpgrp() and self.launch["parent_pid"] == os.getppid()
                     and Path.cwd() == ROOT == Path(self.launch["cwd"])
                     and self.launch["cap_seconds"] == 120 and self.launch["clock_backend"] == self.clock.backend
                     and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
                     and self.launch["deadline_ns"] == self.launch["started_ns"] + 120 * 10**9
                     and self.launch["clock_source_sha256"] == CLOCK_PIN
                     and self.launch["watchdog_sha256"] == SUPERVISOR_PIN, "original bounded audit process")
        self.require(self.digest(self.args.plan)["sha256"] == self.args.plan_sha256, "external frozen audit plan")
        self.plan = self.read(self.args.plan)
        self.require(self.plan["version"] == VERSION and self.plan["status"] == "frozen_before_saved_arithmetic"
                     and self.plan["limits"] == LIMITS and set(self.plan["inputs"]) == {"plan", "worker", "terminal"}
                     and set(self.plan["sources"]) == {SELF, CLOCK, SUPERVISOR}, "fixed saved-only audit scope")
        for name, pin in self.plan["sources"].items():
            self.require(self.digest(name)["sha256"] == pin, "audit source " + name)
        for record in self.plan["inputs"].values():
            self.require(Path(record["path"]).is_absolute() and self.digest(record["path"]) == {
                k: record[k] for k in ("sha256", "bytes")}, "external producer pin")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.require(os.environ.get(name) == "1", "single-thread audit environment")
        self.receipt.update(plan_sha256=self.args.plan_sha256, supervision_sha256=self.digest(self.args.supervision)["sha256"],
                            producer_inputs=self.plan["inputs"], sources=self.plan["sources"])
        write(self.out / "started.json", {"launch": self.launch, "started_ns": self.start, "plan_sha256": self.args.plan_sha256})

    def closure(self, plan_path, worker_path, terminal_path, expected_files=None):
        plan, worker, terminal = self.read(plan_path), self.read(worker_path), self.read(terminal_path)
        command = terminal["command"]
        launch_path = self.path(command[command.index("--supervision") + 1])
        launch = self.read(launch_path)
        self.require(worker["status"] == "completed" and worker["pending"] == []
                     and worker["plan_sha256"] == self.digest(plan_path)["sha256"]
                     and worker["supervision_sha256"] == self.digest(launch_path)["sha256"], "completed worker lineage")
        self.require(terminal["status"] == "completed" and terminal["returncode"] == 0 and terminal["timed_out"] is False
                     and terminal["error"] is None and terminal["clock_error"] is None
                     and terminal["group_absent"] is True and terminal["cleanup"]["reaped"] is True
                     and terminal["cleanup"]["errors"] == [], "successful original phase parent")
        for key, value in launch.items():
            self.require(terminal[key] == value, "same original launch " + key)
        self.require(Path(command[command.index("--plan") + 1]) == self.path(plan_path)
                     and command[command.index("--plan-sha256") + 1] == self.digest(plan_path)["sha256"]
                     and Path(command[command.index("--output") + 1]) == self.path(worker_path).parent,
                     "phase invocation joins")
        self.require(launch["started_ns"] <= worker["started_ns"] < worker["finished_ns"] <= terminal["finished_ns"]
                     <= launch["deadline_ns"] and terminal["elapsed_ns"] == terminal["finished_ns"] - terminal["started_ns"]
                     and terminal["wall_seconds"] == terminal["elapsed_ns"] / 1e9
                     and worker["wall_seconds"] == (worker["finished_ns"] - worker["started_ns"]) / 1e9, "phase physical time")
        directory = self.path(worker_path).parent
        self.require({p.name for p in directory.iterdir()} == set(worker["files"]) | {"receipt.json"}, "closed phase inventory")
        if expected_files is not None:
            self.require(set(worker["files"]) == expected_files, "exact evaluation inventory")
        for name, record in worker["files"].items():
            self.require(Path(name).name == name and self.digest(directory / name) == record, "phase payload " + name)
        self.counts["phase_payloads"] += len(worker["files"])
        self.counts["closed_phases"] += 1
        return plan, worker, terminal

    def authenticate_producer(self):
        self.producer, self.worker, self.parent = self.closure(*(self.plan["inputs"][k]["path"] for k in ("plan", "worker", "terminal")), PAYLOADS)
        self.run = self.path(self.plan["inputs"]["worker"]["path"]).parent
        self.require(self.producer["version"] == self.worker["version"] == "otto-query-gate-evaluation-v1"
                     and self.producer["configuration"]["reported_criteria"] == 62
                     and self.producer["configuration"]["required_criteria"] == 38
                     and self.worker["completed_episodes"] == 576 and self.worker["pending_episode"] is None,
                     "completed fixed scientific allocation")
        self.require(self.worker["sources"] == self.producer["sources"] and self.worker["inputs"] == self.producer["inputs"], "producer provenance")
        for name, pin in self.producer["sources"].items():
            self.require(self.digest(name)["sha256"] == pin, "producer source " + name)
        for records in (self.producer["inputs"], self.producer["native_inputs"], self.producer["checkpoints"]):
            for record in records.values():
                self.require(self.digest(record["path"]) == {k: record[k] for k in ("sha256", "bytes")}, "producer input")
        self.phases = {}
        for stage in ("native", "collection", "training"):
            self.phases[stage] = self.closure(*(self.producer["inputs"][f"{stage}_{part}"]["path"] for part in ("plan", "receipt", "terminal")))
            phase_plan, phase_worker, _ = self.phases[stage]
            self.require(set(phase_worker["files"]) == set(phase_plan["payloads"])
                         and len(phase_worker["files"]) == (27 if stage == "training" else 12), "exact prerequisite phase inventory")
        self.require(self.phases["native"][1]["qualified"] and self.phases["native"][1]["completed_episodes"] == 18
                     and self.phases["collection"][1]["complete"] and self.phases["collection"][1]["completed_episodes"] == 60
                     and self.phases["training"][1]["qualified"] and self.phases["training"][1]["completed_fits"] == 6
                     and self.phases["training"][1]["all_train_parity_passed"], "completed phase prerequisites")
        self.saved = self.read(self.run / "summary.json")
        self.deployment, self.setup = self.read(self.run / "deployment.json"), self.read(self.run / "setup.json")
        self.same(self.saved["deployment"], self.deployment, "deployment join")
        self.same(self.deployment["checkpoints"], self.producer["checkpoints"], "six checkpoint joins")
        self.mixtures = {}
        for regime, descriptor in (("lambda3", self.producer["native_inputs"]["base_kernel"]),
                                   ("lambda4", self.producer["native_inputs"]["shift_kernel"]),
                                   ("lambda5", self.producer["inputs"]["kernel_lambda5"])):
            # Decode only the four public mixture scalars, never the likelihood array.
            with zipfile.ZipFile(self.path(descriptor["path"])) as archive:
                info = archive.getinfo("initial_hit_weights.npy")
                self.require(info.file_size <= 4096, "bounded four-scalar mixture entry")
                raw = archive.read(info)
            self.require(raw[:6] == b"\x93NUMPY" and raw[6:8] in (b"\x01\x00", b"\x02\x00"), "declared NPY header")
            width = 2 if raw[6] == 1 else 4
            header_size = int.from_bytes(raw[8:8 + width], "little")
            offset = 8 + width + header_size
            header = ast.literal_eval(raw[8 + width:offset].decode("latin1"))
            self.require(header == {"descr": "<f8", "fortran_order": False, "shape": (4,)}
                         and len(raw) == offset + 32, "exact four float64 mixture values")
            values = struct.unpack("<4d", raw[offset:])
            self.require(values[0] == 0 and all(math.isfinite(v) and v > 0 for v in values[1:])
                         and abs(math.fsum(values) - 1) <= 1e-12, "original public initial-hit weights")
            self.mixtures[regime] = {str(h): values[h] for h in (1, 2, 3)}

    def episodes(self):
        expected = []
        for ri, (regime, first) in enumerate(FIRST.items()):
            for case in range(24):
                offset = (ri * 24 + case) % 8
                for arm in ARMS[offset:] + ARMS[:offset]:
                    expected.append({"regime": regime, "seed": first + case, "case": case, "initial_hit": case % 3 + 1,
                                     "block": case // 3, "arm": arm, "episode_id": f"eval:{regime}:{first + case}:{arm}"})
        self.same(self.producer["cohort"], expected, "frozen exact cohort")
        rows, paired = [], {}
        for row, identity in zip(self.rows(self.run / "episodes.jsonl"), expected, strict=True):
            self.same({k: row[k] for k in identity}, identity, "episode identity")
            steps, found, arm = row["steps"], row["found"], row["arm"]
            self.require(type(steps) is int and 1 <= steps <= 2188 and type(found) is bool and (found or steps == 2188)
                         and row["updates"] == steps and row["blocked_steps"] == 0 and row["final_update_assimilated"]
                         and row["final_public"]["step"] == steps and row["final_public"]["done"] == found, "complete terminal/censored episode")
            self.require(type(row["queries"]) is int and 0 <= row["queries"] <= steps
                         and (arm != "analytic" or row["queries"] == 0) and (arm != "neural" or row["queries"] == steps), "query endpoint counts")
            self.require(all(math.isfinite(row[k]) and row[k] >= 0 for k in METRICS), "finite episode metrics")
            self.same(row["controller_seconds"], sum(row[k] for k in ("init_seconds", "choose_seconds", "update_seconds", "setup_allocation_seconds")), "full controller sum")
            draws = row.pop("draws_evaluation_only")
            self.require([x["index"] for x in draws if x["channel"] == "source"] == [0]
                         and [x["index"] for x in draws if x["channel"] == "hit"] == list(range(steps - int(found)))
                         and all(x["channel"] in ("source", "hit") and 0 <= x["uniform"] < 1 for x in draws), "full draw-index tails")
            source = row["source_evaluation_only"]
            self.require(next(x["selected_index"] for x in draws if x["channel"] == "source") == source[0] * 53 + source[1], "source draw coordinate")
            case = row["regime"], row["case"]
            stream = {(x["channel"], x["index"]): x["uniform"] for x in draws}
            self.require(len(stream) == len(draws), "unique draw identities")
            if case in paired:
                old_source, old_stream = paired[case]
                self.require(old_source == source and all(old_stream[k] == stream[k] for k in old_stream.keys() & stream.keys()), "paired source/uniform witnesses")
                old_stream.update(stream)
            else:
                paired[case] = source, stream
            rows.append(row)
        self.counts["episodes"], self.counts["paired_cases"] = len(rows), len(paired)
        return rows

    def costs(self, rows):
        d, setup = self.deployment, self.setup
        self.same(d["query_threshold"], .05, "fixed threshold")
        self.require(d["warmup_forwards"] == 0 and set(d["gate_load_seconds"]) == set(LEARNED), "fresh six gate loads")
        for row in rows:
            arm = row["arm"]
            common = (setup["common_import_seconds"] + setup["shared_evaluator_setup_seconds"] + d["extra_common_seconds"]) / 576
            tf = 0. if arm == "analytic" else setup["model_setup_seconds"] / 504
            gate = d["gate_load_seconds"].get(arm, 0.) / 72 + (d["gate_module_seconds"] / 432 if arm in LEARNED else 0.)
            for name, value in (("common_setup_allocation_seconds", common), ("tf_setup_allocation_seconds", tf),
                                ("gate_setup_allocation_seconds", gate), ("setup_allocation_seconds", common + tf + gate)):
                self.same(row[name], value, name)
        training_dir = self.path(self.producer["inputs"]["training_receipt"]["path"]).parent
        train = self.read(training_dir / "summary.json")
        self.require([f["fit_id"] for f in train["fits"]] == list(LEARNED), "six fixed training identities")
        fit_cost = {f["fit_id"]: f["fit_seconds"] + f["parity_seconds"] for f in train["fits"]}
        paid_collection, paid_training = self.phases["collection"][2]["wall_seconds"], self.phases["training"][2]["wall_seconds"]
        self.same(d["training_costs"], {"collection_worker_seconds": self.phases["collection"][1]["wall_seconds"],
            "collection_parent_seconds": paid_collection, "training_worker_seconds": self.phases["training"][1]["wall_seconds"],
            "training_parent_seconds": paid_training}, "raw physical TRAIN costs")
        shared = paid_training - math.fsum(fit_cost.values())
        self.require(shared >= 0, "nonoverlapping paid training intervals")
        allocations = {}
        for arm in LEARNED:
            total = paid_collection / 6 + fit_cost[arm] + shared / 6
            allocations[arm] = {"collection_allocation_seconds": paid_collection / 6, "exclusive_fit_and_parity_seconds": fit_cost[arm],
                "training_common_allocation_seconds": shared / 6, "paid_training_seconds": total,
                "training_amortized_over_72_evaluations_seconds": total / 72}
        self.same(d["training_allocations"], allocations, "complete per-fit paid training allocation")
        self.same(d["training_family_allocations"], {k: math.fsum(allocations[f"{k}@{s}"]["paid_training_seconds"] for s in SEEDS) for k in KINDS}, "paid family costs")
        self.same(math.fsum(x["paid_training_seconds"] for x in allocations.values()), paid_collection + paid_training, "pay physical TRAIN totals once")

    def journals(self, rows):
        by_id = {r["episode_id"]: r for r in rows}
        calls, elapsed, per_case = collections.Counter(), collections.defaultdict(float), collections.defaultdict(collections.Counter)
        per_time, stack, sequence = collections.defaultdict(lambda: collections.defaultdict(float)), [], 0
        tf_work, restores = hashlib.sha256(), {}
        for event in self.rows(self.run / "work.jsonl.gz"):
            self.counts["work_events"] += 1
            if event["event"] == "attempt":
                sequence += 1
                self.require(event["call_id"] == sequence and event["parent_call_id"] == (stack[-1]["call_id"] if stack else None), "nested unique attempt ID")
                stack.append(event)
            else:
                self.require(event["event"] == "return" and bool(stack), "matched return exists")
                attempt = stack.pop()
                self.same({k: event[k] for k in attempt if k != "event"}, {k: v for k, v in attempt.items() if k != "event"}, "returned operation identity")
                channel, seconds = event["channel"], event["seconds"]
                self.require(math.isfinite(seconds) and seconds >= 0 and seconds == event["instrumented_seconds"] - event["excluded_io_seconds"], "recorded operation timer")
                calls[channel] += 1
                elapsed[channel] += seconds
                context = event["context"]
                if channel == "gate_restore":
                    self.require(context["arm"] in LEARNED and context["arm"] not in restores, "one fresh restore per gate")
                    restores[context["arm"]] = seconds
                if "episode_id" in context:
                    ep = context["episode_id"]
                    self.require(ep in by_id, "declared work case")
                    per_case[ep][channel] += 1
                    per_time[ep][channel] += seconds
                    if channel == "tensorflow_value":
                        tf_work.update(json.dumps([ep, context["step"]]).encode() + b"\n")
        self.require(not stack, "no pending work stack")
        self.same(restores, self.deployment["gate_load_seconds"], "actual gate load costs")
        self.require(all(calls[name] == 1 for name in ("tensorflow_construction", "tensorflow_build", "tensorflow_load"))
                     and calls["gate_restore"] == 6, "exact cold model and gate setup calls")
        for channel, record in self.worker["calls"].items():
            self.same(record, {"attempted": calls[channel], "returned": calls[channel], "seconds": elapsed[channel]}, "global operation counts/times")
            self.require(calls[channel] <= self.producer["call_caps"][channel], "declared operation maximum")
        self.same(self.saved["calls"], self.worker["calls"], "summary work counts")
        for row in rows:
            ep, steps, queries, arm = row["episode_id"], row["steps"], row["queries"], row["arm"]
            expected = {"native_reset": 1, "actor_construction": 1, "actor_choose": steps, "actor_update": steps,
                        "native_step": steps, "tensorflow_value": queries}
            if arm in LEARNED:
                expected.update(backend_binding=1, numpy_gate=steps)
            self.same(dict(per_case[ep]), {k: v for k, v in expected.items() if v}, "every case operation counts")
            for field, channels in {"init_seconds": ("actor_construction", "backend_binding"), "choose_seconds": ("actor_choose",),
                                    "update_seconds": ("actor_update",), "environment_seconds": ("native_step",),
                                    "reset_seconds": ("native_reset",), "model_forward_seconds": ("tensorflow_value",)}.items():
                self.same(row[field], sum(per_time[ep][c] for c in channels), "episode operation time " + field)
        tf_transition, tf_record = hashlib.sha256(), hashlib.sha256()
        transitions = iter(self.rows(self.run / "transitions.jsonl.gz"))
        for row in rows:
            reset = next(transitions)
            self.require(reset["kind"] == "reset" and reset["episode_id"] == row["episode_id"]
                         and reset["source_evaluation_only"] == row["source_evaluation_only"], "exact reset journal")
            current = reset["public"]
            state = reset["posterior_after"]
            queries = 0
            for step in range(1, row["steps"] + 1):
                event = next(transitions)
                self.require(event["kind"] == "step" and event["episode_id"] == row["episode_id"] and event["step"] == step
                             and event["public"]["step"] == step and event["allowed_actions"] == current["valid_actions"]
                             and event["action"] in current["valid_actions"] and not current["done"], "actual action/public sequence")
                scores, allowed = event["costs"], current["valid_actions"]
                self.require(all(type(scores[a]) in (int, float) and math.isfinite(scores[a]) for a in allowed), "finite saved eligible scores")
                minimum = min(scores[a] for a in allowed)
                rounded = (lambda x: struct.unpack("f", struct.pack("f", x))[0]) if event["queried"] else float
                selected = next(a for a in allowed if abs(rounded(scores[a] - minimum)) < rounded(1e-10))
                self.require(selected == event["action"] and event["model_forward_calls"] == int(event["queried"]), "saved-score endpoint choice/query")
                self.require(event["posterior_before"]["exact"] and event["posterior_after"]["exact"], "inherited original array comparison witness")
                self.same(event["posterior_before"], state, "saved posterior witness continuity")
                position = list(current["position"])
                position[event["action"] // 2] += -1 if event["action"] % 2 == 0 else 1
                self.same(event["public"]["position"], position, "actual selected in-bounds move")
                if event["queried"]:
                    tf_transition.update(json.dumps([row["episode_id"], step]).encode() + b"\n")
                    queries += 1
                current, state = event["public"], event["posterior_after"]
                self.counts["transitions"] += 1
            self.same(current, row["final_public"], "last updated packet")
            self.require(queries == row["queries"], "all actual queries")
        self.require(next(transitions, None) is None, "no extra transitions")
        for ordinal, event in enumerate(self.rows(self.run / "forwards.jsonl.gz"), 1):
            self.require(event["ordinal"] == ordinal and event["input_shape"] == [16, 105, 105]
                         and event["symmetry_average"] is True and len(event["values"]) == 16
                         and all(math.isfinite(x) for x in event["values"]), "bounded original forward witness")
            tf_record.update(json.dumps([event["context"]["episode_id"], event["context"]["step"]]).encode() + b"\n")
            self.counts["forwards"] += 1
        self.require(tf_record.digest() == tf_transition.digest() == tf_work.digest()
                     and self.counts["forwards"] == calls["tensorflow_value"], "exact query/forward/work chronology")
        gate_counts, gate_stack, ids = collections.defaultdict(collections.Counter), {}, collections.Counter()
        for event in self.rows(self.run / "gate-operations.jsonl.gz"):
            ep = event["episode_id"]
            self.require(ep in by_id and by_id[ep]["arm"] in LEARNED, "only learned gate operations")
            if event["event"] == "attempt":
                ids[ep] += 1
                self.require(ep not in gate_stack and event["id"] == ids[ep], "unique gate attempt")
                gate_stack[ep] = event
            else:
                self.require(event["event"] == "return" and ep in gate_stack, "gate return")
                previous = gate_stack.pop(ep)
                self.same({k: v for k, v in event.items() if k != "event"}, {k: v for k, v in previous.items() if k != "event"}, "same gate operation")
                gate_counts[ep][event["channel"]] += 1
            self.counts["gate_operation_events"] += 1
        self.require(not gate_stack, "closed gate journal")
        for row in rows:
            if row["arm"] in LEARNED:
                expected = {"view_initialization": 1, "public_reset": 1, "analytic_score": row["steps"], "feature_build": row["steps"],
                            "gate": row["steps"], "neural_score": row["queries"], "public_update": row["steps"]}
                self.same(dict(gate_counts[row["episode_id"]]), {k: v for k, v in expected.items() if v}, "full public gate operation counts")
                self.same(row["gate_progress"]["calls"], {k: {"attempted": v, "returned": v} for k, v in expected.items()}, "gate acknowledged counters")
        decisions, ages = collections.Counter(), collections.defaultdict(collections.Counter)
        last_query, queried, last_state = {}, collections.Counter(), {}
        for event in self.rows(self.run / "gate-decisions.jsonl.gz"):
            ep = event["episode_id"]
            self.require(ep in by_id and by_id[ep]["arm"] in LEARNED and event["step"] == decisions[ep], "chronological gate decisions")
            age = event["step"] - last_query[ep] if ep in last_query else event["step"]
            self.require(event["query_age"] == age and len(event["features"]) == 31
                         and all(math.isfinite(x) for x in [*event["features"], *event["state_before"], *event["state_after"], event["logit"], event["probability"]])
                         and event["queried"] == (event["probability"] >= .05), "saved gate threshold and public age")
            width = 32 if by_id[ep]["arm"].startswith("gru32@") else 0
            self.require(event["state_before"] == last_state.get(ep, [0.] * width)
                         and len(event["state_after"]) == width, "reset and carried-state continuity")
            last_state[ep] = event["state_after"]
            if event["queried"]:
                last_query[ep] = event["step"]
                queried[ep] += 1
            decisions[ep] += 1
            ages[ep][str(age)] += 1
        for row in rows:
            if row["arm"] in LEARNED:
                ep = row["episode_id"]
                self.require(decisions[ep] == row["steps"] and queried[ep] == row["queries"], "all learned decisions/queries")
                self.same(dict(ages[ep]), row["query_age_histogram"], "query-age distribution")
                self.require(max(map(int, ages[ep])) == row["maximum_query_age"], "maximum actual query age")
        self.counts["gate_decisions"] = sum(decisions.values())

    def reductions(self, rows):
        groups = {k: [] for k in ("gru_absolute", "mlp_absolute", "usefulness", "recurrence", "reference")}
        regimes = {}

        def test(group, name, value, threshold, relation):
            groups[group].append({"name": name, "value": value, "threshold": threshold, "relation": relation,
                                  "passes": value >= threshold if relation == ">=" else value <= threshold})

        for regime in FIRST:
            weights = self.mixtures[regime]
            self.require(set(weights) == {"1", "2", "3"} and all(math.isfinite(x) and x > 0 for x in weights.values())
                         and abs(math.fsum(weights.values()) - 1) <= 1e-12, "saved positive-hit weights")
            local = [r for r in rows if r["regime"] == regime]

            def mean(selected, metric, weights=weights):
                values = {h: [float(r[metric]) for r in selected if r["initial_hit"] == h] for h in (1, 2, 3)}
                return math.fsum(weights[str(h)] * math.fsum(values[h]) / len(values[h]) for h in (1, 2, 3))

            means = {arm: {m: mean([r for r in local if r["arm"] == arm], m) for m in METRICS} for arm in ARMS}
            families = {kind: {m: math.fsum(means[f"{kind}@{seed}"][m] for seed in SEEDS) / 3 for m in METRICS} for kind in KINDS}
            blocks = [{arm: {m: mean([r for r in local if r["arm"] == arm and r["block"] == block], m) for m in METRICS}
                       for arm in ARMS} for block in range(8)]
            if regime != "lambda5":
                for control in ("neural", "analytic"):
                    test("reference", f"{regime}.{control}.success", means[control]["found"], .95, ">=")
                for arm in LEARNED:
                    group = "gru_absolute" if arm.startswith("gru32@") else "mlp_absolute"
                    test(group, f"{regime}.{arm}.success", means[arm]["found"], .95, ">=")
                    for control in ("neural", "analytic"):
                        test(group, f"{regime}.{arm}.moves_vs_{control}", means[arm]["steps"], 1.05 * means[control]["steps"], "<=")
                    test(group, f"{regime}.{arm}.cost_vs_neural", means[arm]["controller_seconds"], .5 * means["neural"]["controller_seconds"], "<=")
                a, b, c = families["gru32"], families["mlp190"], means["analytic"]
                test("usefulness", f"{regime}.success_vs_analytic", a["found"], c["found"], ">=")
                test("usefulness", f"{regime}.moves_vs_analytic", a["steps"], .95 * c["steps"], "<=")
                test("recurrence", f"{regime}.success_vs_mlp", a["found"], b["found"], ">=")
                test("recurrence", f"{regime}.moves_vs_mlp", a["steps"], b["steps"], "<=")
                test("recurrence", f"{regime}.cost_vs_mlp", a["controller_seconds"], .9 * b["controller_seconds"], "<=")
            query_ages = {}
            for arm in LEARNED:
                counts = collections.Counter()
                for row in local:
                    if row["arm"] == arm:
                        counts.update(row["query_age_histogram"])
                query_ages[arm] = {"histogram": dict(counts), "maximum": max(map(int, counts), default=None)}
            regimes[regime] = {"weights": weights, "means": means, "family_means": families, "blocks": blocks,
                "strata": {str(h): {arm: {m: math.fsum(float(r[m]) for r in local if r["arm"] == arm and r["initial_hit"] == h) / 8
                                           for m in METRICS} for arm in ARMS} for h in (1, 2, 3)},
                "paired_family_block_differences": [{m: math.fsum(block[f"mlp190@{s}"][m] - block[f"gru32@{s}"][m] for s in SEEDS) / 3
                                                      for m in ("found", "steps", "controller_seconds", "queries")} for block in blocks],
                "query_ages": query_ages, "raw_counts": {arm: {"found": sum(r["found"] for r in local if r["arm"] == arm), "episodes": 24} for arm in ARMS}}
        self.require({k: len(v) for k, v in groups.items()} == {"gru_absolute": 24, "mlp_absolute": 24, "usefulness": 4, "recurrence": 6, "reference": 4}, "all 62 conditions")
        required = [r for k, v in groups.items() if k != "mlp_absolute" for r in v]
        result = {"regimes": regimes, "criteria": groups, "required_conditions": 38, "required_passed": sum(r["passes"] for r in required),
                  "reported_conditions": 62, "reported_passed": sum(r["passes"] for v in groups.values() for r in v),
                  "pilot_continuation": all(r["passes"] for r in required), "deployment": self.deployment}
        self.same({k: self.saved[k] for k in result}, result, "independent scientific reductions")
        self.counts["reported_criteria"], self.counts["required_criteria"] = 62, 38
        return result

    def execute(self):
        self.out.mkdir(parents=False, exist_ok=False)

        def interrupt(_signum, _frame):
            raise InterruptedError("saved audit terminated")

        signal.signal(signal.SIGTERM, interrupt)
        try:
            self.admit()
            self.authenticate_producer()
            rows = self.episodes()
            self.costs(rows)
            self.journals(rows)
            summary = self.reductions(rows)
            write(self.out / "audit.json", {"version": VERSION, "agreement": True, "summary": summary, "limitations": LIMITATIONS})
            for name, pin in self.plan["sources"].items():
                self.require(self.digest(name)["sha256"] == pin, "unchanged audit source")
            for record in self.plan["inputs"].values():
                self.require(self.digest(record["path"]) == {k: record[k] for k in ("sha256", "bytes")}, "unchanged producer root")
            self.require(self.digest(self.args.plan)["sha256"] == self.args.plan_sha256
                         and self.digest(self.args.supervision)["sha256"] == self.receipt["supervision_sha256"], "unchanged own plan/launch")
            files = {name: self.digest(self.out / name) for name in ("started.json", "audit.json")}
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status="completed", agreement=True, counts=dict(self.counts), files=files,
                started_ns=self.start, finished_ns=finished, wall_seconds=(finished - self.start) / 1e9,
                requires_successful_original_supervisor=True)
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "receipt": self.digest(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            # The original supervisor still bounds emergency failure publication.
            self.recording_failure = True
            self.receipt.update(status="failed", agreement=False, counts=dict(self.counts),
                                failures=[{"error": repr(error), "traceback": traceback.format_exc()}])
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {p.name: self.digest(p) for p in self.out.iterdir() if p.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - never replace the primary failure
                error.add_note(f"Failure receipt publication: {secondary!r}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "supervision", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if not all(getattr(args, k).is_absolute() for k in ("plan", "supervision", "output")):
        parser.error("absolute paths required")
    Audit(args).execute()


if __name__ == "__main__":
    main()
