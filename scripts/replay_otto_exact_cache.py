"""Bounded saved-public-input cache replay, with no model or simulator calls.

Original inputs were not hashed. Input identity is reconstructed from the pinned
public filter/branch algebra and exact posterior witnesses, not independently
observed historical input bytes. Saved values are answers, never new inference.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import resource
import signal
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SELF, TEST = "scripts/replay_otto_exact_cache.py", "tests/test_replay_otto_exact_cache.py"
PROXY = "scripts/diagnose_otto_exact_cache.py"
PROXY_PIN = "3aba301da062f0243d0e14738cb8a636cfd82ec80dbaa997a7f50067249dec18"
STOP = "scripts/summarize_otto_score_forecast_stop.py"
STOP_PIN = "19bc6d0170eb569afa5599a8408fa72c52243b310d9f93d671af2be81dd0d22e"
PUBLIC = "src/openjev/research/otto_released_policy.py"
PUBLIC_PIN = "8abac34fef2a6517d43209d5ca27514633363248f66a249e2f4d6615aa0a0d45"
HELPER, HELPER_TEST = "src/openjev/research/otto_exact_value_cache.py", "tests/test_otto_exact_value_cache.py"
HELPER_PIN = "b7ff822c3feb727cc61dd22741536e56727788cf8b8f36e8f7ce2fa1faab9574"
CLOCK, SUPERVISOR = "src/openjev/research/suspend_clock.py", "scripts/supervise_dialogue_observation_v2.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
PROTOCOL = "research/otto-exact-cache-protocol.md"
UPSTREAM = "tmp/otto-source-review-01/isotropic/classes/rlpolicy.py"
LICENSE = "third_party/otto/LICENSE"
INTERPRETER = ".venv-otto-released-native/bin/python"
VERSION = "otto-exact-cache-replay-v1"
LIMITS = {"seconds": 120, "rss_bytes": 1024**3, "output_bytes": 128 * 1024**2}
CAPACITY, EXPECTED_ROWS, EXPECTED_RESETS, EXPECTED_COMPLETE = 64, 22296, 68, 67
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")
REQUIRED_SOURCES = {SELF, TEST, PROXY, STOP, PUBLIC, HELPER, HELPER_TEST, CLOCK, SUPERVISOR,
                    PROTOCOL, UPSTREAM, LICENSE}
INPUT_ROLES = {"collection_plan", "receipt", "terminal", "stop_summary", "engineering", "runner_engineering"}
FORBIDDEN = ("tensorflow", "tf_keras", "torch", "scipy", "h5py")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


class Rows:
    """Read complete bounded records and consume the entire gzip trailer."""

    def __init__(self, stream, check=lambda: None, maximum=65536):
        self.stream, self.check, self.maximum = stream, check, maximum
        self.count, self.eof = 0, False

    def __iter__(self):
        return self

    def __next__(self):
        self.check()
        line = self.stream.readline(self.maximum + 1)
        if not line:
            self.eof = True
            raise StopIteration
        require(len(line) <= self.maximum and line.endswith(b"\n"), "whole bounded journal record")
        result = json.loads(line)
        require(type(result) is dict, "JSON object journal")
        self.count += 1
        return result

    def end(self):
        require(next(self, None) is None, "no omitted or extra journal records")


class Work:
    """Original attempt/return stack, including the final unreturned calls."""

    def __init__(self, records, expected, pending, cohort):
        self.records, self.expected, self.final_pending = records, expected, pending
        self.counts = {k: {"attempted": 0, "returned": 0, "seconds": 0.} for k in expected}
        self.stack, self.sequence = [], 0
        self.identities = {r["episode_id"]: r for r in cohort}

    def consume(self, row):
        channel, context = row["channel"], row["context"]
        require(channel in self.counts and type(context) is dict, "declared original work channel")
        if "episode_id" in context:
            identity = self.identities[context["episode_id"]]
            require(all(context[k] == v for k, v in identity.items()), "original work episode identity")
        else:
            require(channel in ("tensorflow_construction", "tensorflow_build", "tensorflow_load"),
                    "only model setup work lacks episode identity")
        base = {k: row[k] for k in ("call_id", "channel", "context", "parent_call_id")}
        if row["event"] == "attempt":
            self.sequence += 1
            require(row["call_id"] == self.sequence
                    and row["parent_call_id"] == (self.stack[-1]["call_id"] if self.stack else None),
                    "original ordered nested attempts")
            self.stack.append(base)
            self.counts[channel]["attempted"] += 1
        else:
            require(row["event"] == "return" and self.stack and self.stack[-1] == base,
                    "original return has exact pending attempt")
            require(all(finite(row[k]) for k in ("seconds", "instrumented_seconds", "excluded_io_seconds"))
                    and row["seconds"] == row["instrumented_seconds"] - row["excluded_io_seconds"],
                    "original operation timing arithmetic")
            self.stack.pop()
            self.counts[channel]["returned"] += 1
            self.counts[channel]["seconds"] += row["seconds"]
        for key in ("attempted", "returned"):
            require(self.counts[channel][key] <= self.expected[channel][key], "no excess original calls")

    def forward(self, context, ordinal, seconds):
        for row in self.records:
            self.consume(row)
            if row["channel"] == "tensorflow_value" and row["event"] == "return":
                require(row["context"] == context and self.counts["tensorflow_value"]["returned"] == ordinal
                        and row["seconds"] == seconds, "forward/work exact identity and recorded time")
                return
        raise ValueError("missing original returned TensorFlow operation")

    def finish(self):
        for row in self.records:
            require(not (row["channel"] == "tensorflow_value" and row["event"] == "return"),
                    "no unjoined returned forward")
            self.consume(row)
        require(self.counts == self.expected and self.stack == self.final_pending,
                "complete original operation accounting and preserved pending stack")


def witness(view):
    probability = view.p_source
    return {"mass": float(probability.sum()), "sha256": hashlib.sha256(probability.tobytes()).hexdigest(),
            "exact": True}


def identity_matches(row, identity):
    require(all(row[k] == v for k, v in identity.items()), "exact declared episode identity")


def saved_f32(value, shape, np):
    array = np.asarray(value, dtype=np.float32)
    require(array.shape == shape and np.isfinite(array).all() and array.tolist() == value,
            "saved finite float32 geometry and exact representation")
    return array


class Replay:
    """Pure stream orchestration, with injected qualified public/cache modules."""

    def __init__(self, np, public, helper, check=lambda: None, emit=lambda _: None):
        self.np, self.public, self.helper, self.check, self.emit = np, public, helper, check, emit
        self.counts = {"public_resets": 0, "public_updates": 0, "branch_builds": 0,
                       "historical_value_callbacks": 0, "requests": 0}
        self.pending = None
        self.episodes, self.maximum_cache_key_bytes, self.maximum_cache_value_bytes = [], 0, 0

    def run(self, samples, forwards, transitions, *, cohort, completed, pending, kernels,
            expected_rows, expected_resets, work=None):
        np, cache = self.np, self.helper.ExactValueCache(CAPACITY)
        require(len(completed) + 1 == expected_resets and pending == cohort[len(completed)],
                "complete prefix plus exactly one explicit interrupted episode")
        lengths = [r["steps"] for r in completed]
        remaining = expected_rows - sum(lengths)
        require(0 < remaining <= 2188, "nonempty bounded interrupted prefix")
        lengths.append(remaining)
        stage_offsets = {"train": 0, "valid": 0}
        for index, length in enumerate(lengths):
            identity = cohort[index]
            cache.clear()
            before = cache.statistics()
            partial = index == len(completed)
            self.pending = {**identity, "phase": "reset"}
            self.check()
            reset = next(transitions)
            identity_matches(reset, identity)
            current = self.public._packet(reset["public"], 0)
            require(reset["kind"] == "reset" and current["position"] == (26, 26)
                    and current["hit"] == identity["initial_hit"] and not current["done"], "public reset packet")
            view = self.public.PublicBeliefView(kernels[identity["regime"]])
            view._reset(identity["initial_hit"])
            self.counts["public_resets"] += 1
            state = witness(view)
            require(state == reset["posterior_after"], "exact reconstructed reset witness")
            record = {**identity, "partial_episode": partial, "requests": 0, "hits": 0, "misses": 0,
                      "evictions": 0, "original_forward_seconds": 0., "hit_forward_seconds": 0.}
            start_row = stage_offsets[identity["stage"]]
            if not partial:
                identity_matches(completed[index], identity)
                require(type(length) is int and 1 <= length <= 2188
                        and (completed[index]["found"] or length == 2188)
                        and completed[index]["censored"] is (not completed[index]["found"])
                        and completed[index]["start_row"] == start_row
                        and completed[index]["end_row"] == start_row + length, "complete episode extent")
            for step in range(length):
                self.pending = {**identity, "phase": "request", "step": step}
                self.check()
                sample, forward = next(samples), next(forwards)
                identity_matches(sample, identity)
                require(sample["step"] == step and sample["row_index"] == start_row + step
                        and self.public._packet(sample["public"], step) == current
                        and sample["posterior"] == state and not current["done"], "chronological preaction public sample")
                context = {"phase": "teacher_score", **identity, "step": step,
                           "deployed_query": sample["deployed_query"], "annotation_only": sample["annotation_only"]}
                ordinal = self.counts["requests"] + 1
                require(forward["context"] == context and forward["ordinal"] == ordinal
                        and forward["symmetry_average"] is True and forward["input_shape"] == [16, 105, 105]
                        and finite(forward["seconds"]), "exact original returned forward join")
                if work is not None:
                    work.forward(context, ordinal, forward["seconds"])
                inputs, masses = self.helper.build_inputs(view)
                self.counts["branch_builds"] += 1
                historical_masses = saved_f32(forward["branch_masses"], (4, 4), np)
                require(masses.tobytes() == historical_masses.tobytes(), "exact original float32 branch masses")
                values = saved_f32(forward["values"], (16,), np).reshape(16, 1)
                prior = cache.statistics()

                def historical(_inputs, values=values):
                    self.counts["historical_value_callbacks"] += 1
                    return values

                cached = cache.get_or_compute(inputs, historical)
                require(cached.tobytes() == values.tobytes(), "cache value byte agreement with every saved answer")
                after = cache.statistics()
                hit, evicted = after["hits"] > prior["hits"], after["evictions"] > prior["evictions"]
                self.maximum_cache_key_bytes = max(self.maximum_cache_key_bytes, after["key_bytes"])
                self.maximum_cache_value_bytes = max(self.maximum_cache_value_bytes, after["value_bytes"])
                action = sample["action"]
                require(type(action) is int and action in current["valid_actions"]
                        and sample["legal"] == [a in current["valid_actions"] for a in range(4)],
                        "recorded actual action eligibility")
                transition = next(transitions)
                identity_matches(transition, identity)
                require(transition["kind"] == "step" and transition["step"] == step + 1
                        and transition["row_index"] == sample["row_index"] and transition["action"] == action
                        and transition["deployed_query"] == sample["deployed_query"]
                        and transition["posterior_before"] == state, "actual public transition join")
                following = self.public._packet(transition["public"], step + 1)
                successor, possible = view._move(action, view.agent)
                require(possible and tuple(successor) == following["position"]
                        and transition["native_p_end"] == float(following["done"]), "public action/terminal geometry")
                self.check()
                view._observe(following)
                self.counts["public_updates"] += 1
                state = witness(view)
                require(state == transition["posterior_after"], "exact reconstructed posterior after every final update")
                self.emit({**identity, "ordinal": ordinal, "step": step, "row_index": sample["row_index"],
                    "partial_episode": partial, "hit": hit, "evicted": evicted,
                    "input_sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
                    "values_sha256": hashlib.sha256(cached.tobytes()).hexdigest(),
                    "masses_sha256": hashlib.sha256(masses.tobytes()).hexdigest(),
                    "original_forward_seconds": forward["seconds"], "cache_entries": after["entries"],
                    "cache_key_bytes": after["key_bytes"], "cache_value_bytes": after["value_bytes"],
                    "posterior_after": state, "exact_agreement": True})
                self.counts["requests"] += 1
                record["requests"] += 1
                record["hits"] += int(hit)
                record["misses"] += int(not hit)
                record["evictions"] += int(evicted)
                record["original_forward_seconds"] += forward["seconds"]
                record["hit_forward_seconds"] += forward["seconds"] if hit else 0.
                current = following
            require(all(record[k] == cache.statistics()[k] - before[k] for k in ("requests", "hits", "misses", "evictions")),
                    "per-episode cache counter conservation")
            if partial:
                require(not current["done"], "interrupted teacher prefix remains nonterminal")
            else:
                require(current["done"] == completed[index]["found"]
                        and self.public._packet(completed[index]["final_public"], length) == current
                        and completed[index]["updates"] == length
                        and completed[index]["final_update_assimilated"] is True, "complete final public update")
            stage_offsets[identity["stage"]] += length
            self.episodes.append(record)
            self.pending = None
        for iterator in (samples, forwards, transitions):
            require(next(iterator, None) is None, "all original returned rows consumed through full EOF")
        require(self.counts["requests"] == self.counts["public_updates"] == self.counts["branch_builds"] == expected_rows
                and self.counts["public_resets"] == expected_resets and cache.statistics()["failures"] == 0,
                "complete exact replay counts")
        return self.summary(cohort, cache.statistics())

    def summary(self, cohort, cache):
        fields = ("requests", "hits", "misses", "evictions", "original_forward_seconds", "hit_forward_seconds")

        def aggregate(rows):
            return {k: math.fsum(r[k] for r in rows) if k.endswith("seconds") else sum(r[k] for r in rows)
                    for k in fields}

        totals = aggregate(self.episodes)
        fraction = totals["hit_forward_seconds"] / totals["original_forward_seconds"] if totals["original_forward_seconds"] else 0.
        conditions = [{"name": "complete_exact_replay", "passed": True}]
        for regime in ("lambda3", "lambda4"):
            hits = sum(r["hits"] for r in self.episodes if r["regime"] == regime)
            conditions.append({"name": regime + "_at_least_one_hit", "hits": hits, "passed": hits > 0})
        conditions.append({"name": "at_least_40_percent_original_forward_time", "actual": fraction,
                           "required": .4, "passed": fraction >= .4})
        cells = []
        for stage, regime, arm in dict.fromkeys((r["stage"], r["regime"], r["arm"]) for r in cohort):
            declared = [r for r in cohort if (r["stage"], r["regime"], r["arm"]) == (stage, regime, arm)]
            rows = [r for r in self.episodes if (r["stage"], r["regime"], r["arm"]) == (stage, regime, arm)]
            cells.append({"stage": stage, "regime": regime, "arm": arm, "declared_episodes": len(declared),
                "complete_episodes": sum(not r["partial_episode"] for r in rows),
                "partial_episodes": sum(r["partial_episode"] for r in rows),
                "unstarted_episodes": len(declared) - len(rows), **aggregate(rows)})
        return {"version": VERSION, "agreement": True, "capacity": CAPACITY, "reset": "each episode",
                "counts": self.counts, "cache": cache, "maximum_cache_key_bytes": self.maximum_cache_key_bytes,
                "maximum_cache_value_bytes": self.maximum_cache_value_bytes, "episodes": self.episodes,
                "totals": totals, "by_stage_regime_arm": cells, "hit_forward_time_fraction": fraction,
                "conditions": conditions, "continuation": all(r["passed"] for r in conditions),
                "new_tensorflow_calls": 0, "native_environment_calls": 0, "optimizer_calls": 0,
                "new_model_calls": 0, "forecast_conditions": "not_evaluated",
                "limitations": ["Reconstructed inputs inherit pinned arithmetic and exact posterior witnesses; the original run saved no input hash.",
                    "Saved values are reused only for retrospective consistency, never fitting or new inference.",
                    "Hit-associated historical forward time is avoided-work potential, not measured speedup; future cache overhead is excluded.",
                    "Interrupted-episode observed records remain separate from originally acknowledged complete episodes.",
                    "No Q reduction, model algebra, optimizer, native trajectory, efficacy or forecast rule is regenerated."]}


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.launch = self.replay = self.stream = None
        self.receipt = {"version": VERSION, "status": "started", "agreement": False, "limits": LIMITS,
                        "new_model_calls": 0, "native_calls": 0, "optimizer_calls": 0}

    def check(self):
        require(self.clock.now_ns() < self.launch["deadline_ns"], "original replay deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"] and sum(p.stat().st_size for p in self.out.iterdir())
                < LIMITS["output_bytes"] - 1024**2, "bounded replay resources with failure reserve")

    def bind(self):
        require(hashlib.sha256((ROOT / PROXY).read_bytes()).hexdigest() == PROXY_PIN, "pinned stdlib proxy before import")
        self.p = load("_exact_replay_proxy", PROXY)
        require(self.p.descriptor(CLOCK)["sha256"] == CLOCK_PIN, "pinned clock before import")
        self.clock = load("_exact_replay_clock", CLOCK).SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, "original launch available")
            time.sleep(.01)
        self.launch = self.p.read(self.args.supervision)
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and command[:2] == [str(ROOT / INTERPRETER), str(ROOT / SELF)]
                and self.launch["pid"] == os.getpid() and self.launch["pgid"] == os.getpgrp()
                and self.launch["parent_pid"] == os.getppid() and self.launch["pid"] != self.launch["parent_pid"]
                and self.launch["cwd"] == str(ROOT) == str(Path.cwd()) and self.launch["cap_seconds"] == 120
                and self.launch["clock_backend"] == self.clock.backend
                and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
                and self.launch["deadline_ns"] == self.launch["started_ns"] + 120 * 10**9, "original bounded native process")
        require(all(os.environ.get(key) == "1" for key in THREADS), "one-thread environment before NumPy import")
        require(self.p.descriptor(self.args.plan, self.check)["sha256"] == self.args.plan_sha256, "external prospective plan")
        self.plan = self.p.read(self.args.plan)
        require(self.plan["version"] == VERSION and self.plan["status"] == "frozen_before_input_replay"
                and self.plan["limits"] == LIMITS and self.plan["capacity"] == CAPACITY
                and self.plan["expected_rows"] == EXPECTED_ROWS and self.plan["expected_resets"] == EXPECTED_RESETS
                and self.plan["expected_completed_episodes"] == EXPECTED_COMPLETE and self.plan["reset"] == "each episode"
                and REQUIRED_SOURCES <= self.plan["sources"].keys() and set(self.plan["inputs"]) == INPUT_ROLES,
                "fixed complete replay allocation")
        self.recheck()
        require(all(self.plan["sources"][name] == pin for name, pin in
                    ((PROXY, PROXY_PIN), (STOP, STOP_PIN), (PUBLIC, PUBLIC_PIN), (HELPER, HELPER_PIN), (CLOCK, CLOCK_PIN)))
                and self.launch["clock_source_sha256"] == CLOCK_PIN
                and self.launch["watchdog_sha256"] == self.plan["sources"][SUPERVISOR], "qualified helper and parent pins")
        for role, sources in (("engineering", (PROXY, HELPER, HELPER_TEST)), ("runner_engineering", (SELF, TEST))):
            record = self.plan["inputs"][role]
            engineering = self.p.read(record["path"])
            require(engineering["status"] == "passed" and engineering["sources_before"] == engineering["sources_after"]
                    and all(engineering["sources_after"][name] == self.plan["sources"][name] for name in sources)
                    and engineering["results"] and all(r["exit_code"] == 0 for r in engineering["results"]),
                    "passed fabricated qualification of held sources")
            directory = self.p.path(record["path"]).parent
            for name, d in engineering["files"].items():
                require(Path(name).name == name and self.p.descriptor(directory / name, self.check) == d,
                        "closed engineering evidence")
        self.receipt.update(plan_sha256=self.args.plan_sha256, sources=self.plan["sources"], inputs=self.plan["inputs"],
                            supervision_sha256=self.p.descriptor(self.args.supervision)["sha256"])
        self.p.write(self.out / "started.json", {"launch": self.launch, "started_ns": self.start})

    def recheck(self):
        for name, pin in self.plan["sources"].items():
            require(self.p.descriptor(name, self.check)["sha256"] == pin, "frozen replay source")
        for record in self.plan["inputs"].values():
            require(self.p.descriptor(record["path"], self.check) == {k: record[k] for k in ("sha256", "bytes")},
                    "pinned replay input")

    def emit(self, record):
        self.check()
        self.receipt["pending_publication"] = {k: record[k] for k in ("episode_id", "step", "ordinal")}
        encoded = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        require(len(encoded) <= 4096 and self.stream.write(encoded) == len(encoded), "bounded full replay record")
        self.receipt["pending_publication"] = None

    def body(self):
        inputs = self.plan["inputs"]
        old, worker, stop = (self.p.read(inputs[r]["path"]) for r in ("collection_plan", "receipt", "stop_summary"))
        self.check()
        report = load("_exact_replay_stop", STOP).summarize(SimpleNamespace(
            plan=self.p.path(inputs["collection_plan"]["path"]), receipt=self.p.path(inputs["receipt"]["path"]),
            terminal=self.p.path(inputs["terminal"]["path"])))
        require(report == stop and stop["status"] == "incomplete_collection_verified"
                and worker["status"] == "failed" and worker["complete"] is False
                and worker["completed_episodes"] == EXPECTED_COMPLETE
                and worker["calls"]["native_reset"]["returned"] == EXPECTED_RESETS
                and all(worker["calls"][k]["returned"] == EXPECTED_ROWS for k in
                        ("native_step", "actor_update", "teacher_score", "tensorflow_value")), "exact original stopped scope")
        self.check()
        directory = self.p.path(inputs["receipt"]["path"]).parent
        runtime = self.p.read(directory / "runtime.json")
        require(runtime["python"] == sys.version and runtime["executable"] == sys.executable,
                "unchanged original Python runtime")
        require(not any(k in sys.modules for k in ("numpy", *FORBIDDEN)), "clean pre-numerical admission")
        import numpy as np

        require(np.__version__ == runtime["all_distributions"]["numpy"], "original native NumPy version")
        public, helper = load("_exact_replay_public", PUBLIC), load("_exact_replay_cache", HELPER)
        kernels = {}
        for regime, role in (("lambda3", "base_kernel"), ("lambda4", "shift_kernel")):
            with np.load(self.p.path(old["native_inputs"][role]["path"]), allow_pickle=False) as archive:
                require(set(archive.files) == {"likelihood", "initial_hit_weights"}, "exact original kernel archive")
                kernels[regime] = archive["likelihood"]
        self.receipt["kernel_arrays_decoded"] = 2
        self.receipt["runtime"] = {"python": sys.version, "executable": sys.executable,
                                   "numpy": np.__version__, "threads": {k: os.environ[k] for k in THREADS}}
        with ExitStack() as stack:
            streams = {name: Rows(stack.enter_context(gzip.open(directory / f"{name}.jsonl.gz", "rb")), self.check)
                       for name in ("samples", "forwards", "transitions", "work", "weights")}
            completed = list(Rows(stack.enter_context((directory / "episodes.jsonl").open("rb")),
                                  self.check, 2 * 1024**2))
            boundaries = list(Rows(stack.enter_context((directory / "episode-boundaries.jsonl").open("rb")),
                                   self.check, 2 * 1024**2))
            require(len(completed) == EXPECTED_COMPLETE and len(boundaries) == 2 * EXPECTED_COMPLETE + 1,
                    "all observed complete boundaries and one interrupted attempt")
            expected_weights = [f"{kind}_{i}" for i in range(4) for kind in ("kernel", "bias")]
            weights = list(streams["weights"])
            require([r["id"] for r in weights] == expected_weights and all(r["exact"] is True for r in weights),
                    "all original weight-identity metadata, no tensor decoding")
            work = Work(streams["work"], worker["calls"], worker["pending"], old["cohort"])
            self.stream = (self.out / "requests.jsonl").open("xb")
            self.replay = Replay(np, public, helper, self.check, self.emit)
            summary = self.replay.run(streams["samples"], streams["forwards"], streams["transitions"],
                cohort=old["cohort"], completed=completed, pending=worker["pending_episode"], kernels=kernels,
                expected_rows=EXPECTED_ROWS, expected_resets=EXPECTED_RESETS, work=work)
            work.finish()
            require(all(s.eof for s in streams.values()), "all five original gzip trailers consumed")
            require(math.isclose(summary["totals"]["original_forward_seconds"],
                                 worker["calls"]["tensorflow_value"]["seconds"], rel_tol=1e-12, abs_tol=1e-9),
                    "complete original recorded forward time")
            summary["original_calls"] = worker["calls"]
            summary["original_pending_calls"] = worker["pending"]
            summary["journal_counts"] = {k: s.count for k, s in streams.items()}
            summary["all_gzip_eof_verified"] = True
        self.close()
        self.p.write(self.out / "summary.json", summary)
        self.receipt.update(counts=self.replay.counts, continuation=summary["continuation"],
                            original_payloads=worker["files"], all_gzip_eof_verified=True)
        for name, descriptor in worker["files"].items():
            require(self.p.descriptor(directory / name, self.check) == descriptor, "unchanged original payload")
        for name, pin in old["sources"].items():
            require(self.p.descriptor(name, self.check)["sha256"] == pin, "unchanged inherited source")
        for record in [*old["inputs"].values(), *old["native_inputs"].values()]:
            require(self.p.descriptor(record["path"], self.check) == {k: record[k] for k in ("sha256", "bytes")},
                    "unchanged inherited input")
        require(not any(k in sys.modules for k in FORBIDDEN), "no model or native runtime imports")

    def close(self):
        if self.stream is not None and not self.stream.closed:
            failure = None
            try:
                self.stream.flush()
                os.fsync(self.stream.fileno())
            except BaseException as error:  # noqa: BLE001 - still attempt close and preserve original I/O error
                failure = error
            try:
                self.stream.close()
            except BaseException as error:  # noqa: BLE001 - retain both failed flush and failed close
                if failure is None:
                    failure = error
                else:
                    failure.add_note(f"Additional close failure: {error!r}")
            if failure is not None:
                raise failure

    def execute(self):
        self.out.mkdir(exist_ok=False)

        def interrupted(_signal, _frame):
            raise InterruptedError("original exact-input replay stopped")

        signal.signal(signal.SIGTERM, interrupted)
        try:
            self.bind()
            self.body()
            self.recheck()
            require(self.p.descriptor(self.args.plan)["sha256"] == self.args.plan_sha256
                    and self.p.descriptor(self.args.supervision)["sha256"] == self.receipt["supervision_sha256"],
                    "unchanged plan and original launch")
            require({p.name for p in self.out.iterdir()} == {"started.json", "requests.jsonl", "summary.json"},
                    "exact completed replay payloads")
            files = {p.name: self.p.descriptor(p, self.check) for p in self.out.iterdir()}
            finished = self.clock.now_ns()
            self.receipt.update(status="completed", agreement=True, files=files, started_ns=self.start,
                finished_ns=finished, wall_seconds=(finished-self.start)/1e9, requires_successful_original_supervisor=True)
            self.p.write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "receipt": self.p.descriptor(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:  # noqa: BLE001 - preserve primary replay failure
                error.add_note(f"Replay journal cleanup: {cleanup!r}")
            self.receipt.update(status="failed", agreement=False, error=repr(error),
                progress=None if self.replay is None else self.replay.counts,
                pending=None if self.replay is None else self.replay.pending)
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                if hasattr(self, "p"):
                    self.receipt["files"] = {p.name: self.p.descriptor(p) for p in self.out.iterdir() if p.is_file()}
                with (self.out / "receipt.json").open("x") as stream:
                    json.dump(self.receipt, stream, sort_keys=True, indent=2, allow_nan=False)
                    stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            except BaseException as publication:  # noqa: BLE001 - preserve original failure
                error.add_note(f"Failure publication: {publication!r}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "supervision", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(all(p.is_absolute() and p.is_relative_to(ROOT) and ".." not in p.parts
                and not any(q.is_symlink() for q in (p, *p.parents))
                for p in (args.plan, args.supervision, args.output)), "contained absolute invocation paths")
    Run(args).execute()


if __name__ == "__main__":
    main()
