"""Saved public-state cache proxy, never teacher/model replay or a speed test."""
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
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF, TEST = "scripts/diagnose_otto_exact_cache.py", "tests/test_diagnose_otto_exact_cache.py"
CLOCK, SUPERVISOR = "src/openjev/research/suspend_clock.py", "scripts/supervise_dialogue_observation_v2.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
VERSION = "otto-exact-cache-proxy-v1"
LIMITS = {"seconds": 60, "rss_bytes": 256 * 1024**2, "output_bytes": 32 * 1024**2}
CAPACITY, EXPECTED_ROWS = 64, 22296


def require(ok, message):
    if not ok:
        raise ValueError(message)


def path(value):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    require(p.is_file() and p.is_relative_to(ROOT) and ".." not in p.parts
            and not any(q.is_symlink() for q in (p, *p.parents)), "contained regular evidence")
    return p


def descriptor(value, check=lambda: None):
    p, digest = path(value), hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block); check()
    return {"sha256": digest.hexdigest(), "bytes": p.stat().st_size}


def read(value):
    return json.loads(path(value).read_text())


def write(p, value):
    with p.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


class IntegerToken(str):
    """Inert JSON token; only whitelisted metadata is converted to integers."""


def integer(value):
    require(type(value) is IntegerToken, "integer metadata token")
    return int(value)


def project(line, *, episode=False):
    # Float/constant labels are discarded by the decoder; integer labels stay
    # inert. No scores, feature vectors, actions, hits or outcomes are accessed.
    raw = json.loads(line, parse_int=IntegerToken, parse_float=lambda _: None, parse_constant=lambda _: None)
    fields = ("episode_id", "stage", "regime", "arm")
    result = {key: raw[key] for key in fields}
    require(all(type(value) is str and value for value in result.values()), "plain metadata identities")
    if episode:
        return {**result, "steps": integer(raw["steps"])}
    position, fingerprint = raw["public"]["position"], raw["posterior"]["sha256"]
    require(isinstance(position, list) and len(position) == 2, "public 2D position")
    position = tuple(integer(value) for value in position)
    require(all(0 <= value < 53 for value in position) and type(fingerprint) is str and len(fingerprint) == 64
            and all(value in "0123456789abcdef" for value in fingerprint), "exact public-state key")
    return {**result, "step": integer(raw["step"]), "position": position, "posterior_sha256": fingerprint}


class Cache:
    def __init__(self):
        self.entries = OrderedDict()

    def query(self, key):
        hit, evicted = key in self.entries, False
        if hit:
            self.entries.move_to_end(key)
        else:
            self.entries[key] = None
            if len(self.entries) > CAPACITY:
                self.entries.popitem(last=False); evicted = True
        return hit, evicted


def scan(records, cohort, completed, pending, expected_rows, check=lambda: None):
    """Preserve every chronological row, including the declared interrupted path."""
    rows, current, cache = [], None, None
    for sample in records:
        if current is None or sample["episode_id"] != current["episode_id"]:
            require(len(rows) < len(cohort), "bounded available episode prefix")
            identity = {key: cohort[len(rows)][key] for key in ("episode_id", "stage", "regime", "arm")}
            require({key: sample[key] for key in identity} == identity, "declared ordered episode prefix")
            current = {**identity, "queries": 0, "hits": 0, "misses": 0, "evictions": 0,
                       "partial_episode": len(rows) >= len(completed)}
            rows.append(current); cache = Cache()
        require(all(sample[key] == current[key] for key in ("episode_id", "stage", "regime", "arm"))
                and type(sample["step"]) is int and sample["step"] == current["queries"] < 2188,
                "unique consecutive preaction metadata")
        hit, evicted = cache.query((sample["regime"], sample["position"], sample["posterior_sha256"]))
        current["queries"] += 1; current["hits"] += int(hit)
        current["misses"] += int(not hit); current["evictions"] += int(evicted)
        check()
    require(sum(row["queries"] for row in rows) == expected_rows, "all returned sample metadata retained")
    require(len(completed) <= len(rows) <= len(completed) + 1, "complete paths plus at most one partial path")
    for index, saved in enumerate(completed):
        require(all(rows[index][key] == saved[key] for key in ("episode_id", "stage", "regime", "arm"))
                and rows[index]["queries"] == saved["steps"], "complete episode row count")
    if len(rows) > len(completed):
        require(pending is not None and rows[-1]["episode_id"] == pending["episode_id"], "explicit partial episode")
    def totals(group):
        return {key: sum(row[key] for row in group) for key in ("queries", "hits", "misses", "evictions")}
    return {"episodes": rows, "totals": totals(rows),
            "by_stage_regime_arm": [{"stage": stage, "regime": regime, "arm": arm, **totals(group)}
                for stage, regime, arm in dict.fromkeys((r["stage"], r["regime"], r["arm"]) for r in cohort)
                for group in [[r for r in rows if (r["stage"], r["regime"], r["arm"]) == (stage, regime, arm)]]]}


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.launch = None
        self.receipt = {"version": VERSION, "status": "started", "limits": LIMITS, "scientific_calls": 0,
                        "scores_consumed": 0, "array_decodes": 0}

    def check(self):
        require(self.clock.now_ns() < self.launch["deadline_ns"], "original diagnostic deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"] and sum(p.stat().st_size for p in self.out.iterdir())
                < LIMITS["output_bytes"] - 1024**2, "bounded diagnostic resources")

    def bind(self):
        require(descriptor(CLOCK)["sha256"] == CLOCK_PIN, "qualified clock before import")
        spec = importlib.util.spec_from_file_location("_cache_proxy_clock", ROOT / CLOCK)
        module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
        self.clock = module.SuspendClock(); self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, "original launch available"); time.sleep(.01)
        self.launch = read(self.args.supervision)
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and self.launch["pid"] == os.getpid()
                and self.launch["pgid"] == os.getpgrp() and self.launch["parent_pid"] == os.getppid()
                and self.launch["cwd"] == str(ROOT) == str(Path.cwd()) and self.launch["cap_seconds"] == 60
                and self.launch["clock_backend"] == self.clock.backend
                and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
                and self.launch["deadline_ns"] == self.launch["started_ns"] + 60 * 10**9, "original bounded process")
        require(descriptor(self.args.plan, self.check)["sha256"] == self.args.plan_sha256, "external prospective plan")
        self.plan = read(self.args.plan)
        require(self.plan["version"] == VERSION and self.plan["status"] == "frozen_before_metadata_scan"
                and self.plan["limits"] == LIMITS and self.plan["capacity"] == CAPACITY
                and self.plan["expected_rows"] == EXPECTED_ROWS
                and {SELF, TEST, CLOCK, SUPERVISOR} <= self.plan["sources"].keys(), "fixed proxy allocation")
        for name, pin in self.plan["sources"].items():
            require(descriptor(name, self.check)["sha256"] == pin, "frozen diagnostic source")
        require(self.launch["clock_source_sha256"] == CLOCK_PIN
                and self.launch["watchdog_sha256"] == self.plan["sources"][SUPERVISOR], "original parent source")
        require(set(self.plan["inputs"]) == {"collection_plan", "receipt", "terminal", "stop_summary"}, "exact input roles")
        for record in self.plan["inputs"].values():
            require(descriptor(record["path"], self.check) == {k: record[k] for k in ("sha256", "bytes")}, "pinned input")
        self.receipt.update(plan_sha256=self.args.plan_sha256, sources=self.plan["sources"], inputs=self.plan["inputs"],
                            supervision_sha256=descriptor(self.args.supervision)["sha256"])
        write(self.out / "started.json", {"launch": self.launch, "started_ns": self.start})

    def body(self):
        inputs = self.plan["inputs"]
        old, worker, terminal, stop = (read(inputs[key]["path"]) for key in ("collection_plan", "receipt", "terminal", "stop_summary"))
        require(worker["status"] == "failed" and worker["complete"] is False
                and worker["plan_sha256"] == inputs["collection_plan"]["sha256"]
                and worker["sources"] == old["sources"] and worker["inputs"] == old["inputs"]
                and worker["native_inputs"] == old["native_inputs"], "original failed scientific allocation")
        require(terminal["group_absent"] is True and terminal["cleanup"]["reaped"] is True
                and terminal["cap_seconds"] == 900, "original process closed without extending allocation")
        command = list(terminal["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command[:3] == [str(ROOT / ".venv-otto-released-native/bin/python"),
                str(ROOT / "scripts/collect_otto_score_forecasts.py"), "run"] and len(command) == 11, "original native command")
        options = dict(zip(command[3::2], command[4::2], strict=True)); directory = path(inputs["receipt"]["path"]).parent
        require(set(options) == {"--plan", "--plan-sha256", "--supervision", "--output"}
                and options["--plan"] == str(path(inputs["collection_plan"]["path"]))
                and options["--plan-sha256"] == worker["plan_sha256"] and options["--output"] == str(directory), "original command joins")
        launch = read(options["--supervision"])
        require(all(terminal[k] == v for k, v in launch.items())
                and descriptor(options["--supervision"])["sha256"] == worker["supervision_sha256"], "original launch join")
        for name, pin in old["sources"].items():
            require(descriptor(name, self.check)["sha256"] == pin, "unchanged collection source")
        for d in [*old["inputs"].values(), *old["native_inputs"].values()]:
            require(descriptor(d["path"], self.check) == {k: d[k] for k in ("sha256", "bytes")}, "unchanged inherited input")
        require(stop["status"] == "incomplete_collection_verified" and stop["rule"] == "not_evaluated"
                and all(stop["inputs"][key]["sha256"] == inputs[role]["sha256"]
                        for key, role in (("plan", "collection_plan"), ("receipt", "receipt"), ("terminal", "terminal"))), "preserved stop summary")
        for name in ("samples.jsonl.gz", "episodes.jsonl"):
            require(descriptor(directory / name, self.check) == worker["files"][name], "authenticated metadata stream")
        completed = []
        with (directory / "episodes.jsonl").open("rb") as stream:
            for line in stream:
                require(len(line) <= 2 * 1024**2 and line.endswith(b"\n"), "whole original episode metadata")
                completed.append(project(line, episode=True))
        require(len(completed) == worker["completed_episodes"] == stop["acknowledged_complete_episodes"], "acknowledged complete prefix")
        def records():
            with gzip.open(directory / "samples.jsonl.gz", "rb") as stream:
                for line in stream:
                    require(len(line) <= 65536 and line.endswith(b"\n"), "whole bounded sample metadata")
                    yield project(line)
            self.receipt["gzip_full_eof_verified"] = True
        require(all(worker["calls"][channel]["returned"] == EXPECTED_ROWS
                    for channel in ("native_step", "teacher_score", "tensorflow_value")), "all original returned-row counts")
        result = scan(records(), old["cohort"], completed, worker["pending_episode"], EXPECTED_ROWS, self.check)
        result.update(version=VERSION, capacity=CAPACITY, reset="each episode", key=["regime", "position", "posterior_sha256"],
            scope="Conservative exact public-state proxy only. No TensorFlow-input equality, functional cache replay or speedup is established.",
            interpretation="Hits are exact public-state proxy repetitions, not measured saved computation. Low proxy reuse cannot rule out float32 inference-input reuse.",
            original_allocation_status="failed", forecast_rule="not_evaluated", sampled_rows_include_partial_episode=True,
            original_calls=worker["calls"], metadata_files={name: worker["files"][name]
                for name in ("samples.jsonl.gz", "episodes.jsonl")},
            key_equality="Full tuple equality; posterior identity inherits the saved SHA256 witness assumption.")
        for name in ("samples.jsonl.gz", "episodes.jsonl"):
            require(descriptor(directory / name, self.check) == worker["files"][name], "unchanged scanned metadata")
        write(self.out / "summary.json", result)

    def execute(self):
        self.out.mkdir(exist_ok=False)
        def interrupted(_signal, _frame):
            raise InterruptedError("original cache diagnostic stopped")
        signal.signal(signal.SIGTERM, interrupted)
        try:
            self.bind(); self.body(); self.check()
            for name, pin in self.plan["sources"].items():
                require(descriptor(name, self.check)["sha256"] == pin, "unchanged diagnostic source")
            for d in self.plan["inputs"].values():
                require(descriptor(d["path"], self.check) == {k: d[k] for k in ("sha256", "bytes")}, "unchanged input")
            require(descriptor(self.args.plan, self.check)["sha256"] == self.args.plan_sha256
                    and descriptor(self.args.supervision)["sha256"] == self.receipt["supervision_sha256"], "unchanged plan/launch")
            files = {p.name: descriptor(p, self.check) for p in self.out.iterdir() if p.is_file()}
            finished = self.clock.now_ns()
            self.receipt.update(status="completed", started_ns=self.start, finished_ns=finished,
                wall_seconds=(finished-self.start)/1e9, files=files)
            write(self.out / "receipt.json", self.receipt); self.check()
            print(json.dumps({"status": "completed", "receipt": descriptor(self.out / "receipt.json")}))
        except BaseException as error:
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt.update(status="failed", error=repr(error),
                    files={p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()})
                write(self.out / "receipt.json", self.receipt)
            except BaseException as publication:  # noqa: BLE001 - retain the primary failure
                error.add_note(f"Failure publication: {publication!r}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "supervision", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(args.output.is_absolute() and args.output.is_relative_to(ROOT) and ".." not in args.output.parts,
            "exclusive contained output")
    Run(args).execute()


if __name__ == "__main__":
    main()
