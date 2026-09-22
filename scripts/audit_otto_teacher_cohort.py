"""Independent saved-record audit of the fixed teacher-cohort support preflight.

No teachers, policies, simulators or producer helpers are imported. Saved
posterior bytes are checked against original public witnesses; intermediate
filter arithmetic and actual constructor execution remain source-bound evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import resource
import signal
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-teacher-cohort-saved-audit-v1"
WORKER_VERSION = "otto-teacher-cohort-v1"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
HISTORY = "output/otto-teacher-label-pilot-v1/plan-01.json"
HISTORY_PIN = "6956fd2658fc8920c277a6a4585ecb75d0b1639730ead5158af0cc7835269380"
SYMM = "output/otto-symmetry-head-v1/run-01"
NEW_SOURCES = {
    "scripts/prepare_otto_teacher_cohort.py", "src/openjev/research/otto_teacher_cohort.py",
    "tests/test_otto_teacher_cohort.py", "tests/test_prepare_otto_teacher_cohort.py",
    "research/otto-teacher-cohort-protocol.md",
}
PAYLOADS = {"started.json", "anchors.npz", "reconstruction.jsonl", "support.jsonl", "support.json"}
THREADS = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")
LIMITS = {"native_seconds": 180, "rss_bytes": 4 * 1024**3, "output_bytes": 64 * 1024**2}
WORKER_LIMITS = {"native_seconds": 600, "rss_bytes": 4 * 1024**3, "output_bytes": 1024**3}
ARMS = tuple(f"{kind}@{seed}" for kind in ("dense", "shared") for seed in (9101, 9102, 9103))
FIRST = {"lambda3": 950001, "lambda4": 960001}
IDENTITY = {"episode_id", "stage", "regime", "seed", "initial_hit", "arm"}
METADATA = IDENTITY | {"row_index", "prefix_index", "public", "posterior", "teacher_costs"}
HEADER = re.compile(r'^\{"kind":"(reset|step)","episode_id":"([^"\\]+)",')
SCOPE = (
    "Independent metadata selection, original public-record/ledger joins, captured float64 byte hashes, "
    "mass/support predicates and count reconstruction. No numerical posterior replay, constructor, "
    "policy, sampler, RNG, learned model, optimization or native environment calls. Original filtering, "
    "constructor execution and measured timings remain inherited authenticated evidence. Unsupported "
    "anchors are preserved; audit agreement is not sampling admission or efficacy evidence."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


class Number(str):
    """Numerical JSON token kept lexical until a public field is selected."""


def decode(raw, *, lexical=False):
    kwargs = {"parse_int": Number, "parse_float": Number} if lexical else {}
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, **kwargs)


def numbers(value):
    if isinstance(value, Number):
        return float(value) if any(c in value for c in ".eE") else int(value)
    if isinstance(value, list):
        return [numbers(v) for v in value]
    if isinstance(value, dict):
        return {k: numbers(v) for k, v in value.items()}
    return value


def integer(value, lower, upper):
    require(type(value) is int and lower <= value <= upper, "integer range")
    return value


def public(value, step):
    require(set(value) == {"position", "step", "hit", "done", "valid_actions"}, "public fields")
    require(type(value["done"]) is bool and value["step"] == integer(step, 0, 2188), "public step/done")
    require(type(value["step"]) is int and len(value["position"]) == 2, "public dimensions")
    position = [integer(v, 0, 52) for v in value["position"]]
    allowed = [a for a in range(4) if 0 <= position[a // 2] + 2 * (a % 2) - 1 < 53]
    require(value["valid_actions"] == ([] if value["done"] else allowed)
            and all(type(a) is int for a in value["valid_actions"]), "exact public eligibility")
    require(value["hit"] == -2 if value["done"] else 0 <= integer(value["hit"], 0, 3) <= 3,
            "public hit")
    return value


def witness(value):
    require(set(value) == {"sha256", "mass"} and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
            "posterior witness fields")
    require(type(value["mass"]) in (int, float) and math.isfinite(value["mass"])
            and value["mass"] >= 0, "posterior mass")
    return value


def select_rows(lines):
    """Independent floor-spaced selector; labels are discarded before conversion."""
    episodes = [(regime, seed, arm, f"dagger:{regime}:{seed}:{arm}")
                for regime, first in FIRST.items() for seed in range(first, first + 12) for arm in ARMS]
    groups = {episode: [] for _, _, _, episode in episodes}
    seen = set()
    count = 0
    for index, raw in enumerate(lines):
        item = decode(raw, lexical=True)
        require(set(item) == METADATA, "original metadata schema")
        del item["teacher_costs"]
        row = numbers(item)
        require(type(row["row_index"]) is int and row["row_index"] == index, "complete metadata order")
        require(row["regime"] in FIRST and row["arm"] in ARMS and row["stage"] == "dagger", "TRAIN identity")
        first = FIRST[row["regime"]]
        seed = integer(row["seed"], first, first + 11)
        require(type(row["initial_hit"]) is int and row["initial_hit"] == 1 + (seed - first) % 3
                and row["episode_id"] == f'dagger:{row["regime"]}:{seed}:{row["arm"]}', "episode identity")
        prefix = integer(row["prefix_index"], 0, 2187)
        public(row["public"], prefix)
        witness(row["posterior"])
        require(not row["public"]["done"], "pre-action nonterminal metadata")
        key = row["episode_id"], prefix
        require(key not in seen, "unique retained episode prefix")
        seen.add(key)
        groups[row["episode_id"]].append(row)
        count += 1
    require(count == 4596 and all(groups.values()), "complete 4596-row, 144-episode pool")
    selected, diagnostics = [], []
    for regime, seed, arm, episode in episodes:
        rows = sorted(groups[episode], key=lambda r: (r["prefix_index"], r["row_index"]))
        k = min(4, len(rows))
        indices = [0] if k == 1 else [j * (len(rows) - 1) // (k - 1) for j in range(k)]
        selected.extend(rows[i] for i in indices)
        diagnostics.append({"episode_id": episode, "regime": regime, "seed": seed, "arm": arm,
                            "retained_rows": len(rows), "selected_rows": k,
                            "selected_retained_indices": indices, "fewer_than_four": k < 4})
    selected.sort(key=lambda r: (r["regime"], r["seed"], r["arm"], r["prefix_index"], r["row_index"]))
    return {"selections": [{"anchor_id": i, **r} for i, r in enumerate(selected)], "episodes": diagnostics,
            "counts": {"metadata_rows": count, "episodes": 144, "selected_anchors": len(selected),
                       "fewer_than_four_episodes": sum(r["fewer_than_four"] for r in diagnostics)}}


class Audit:
    def __init__(self, args):
        self.args, self.clock, self.start = args, None, None
        self.comparisons = self.bytes_hashed = 0

    def agree(self, condition, message):
        self.comparisons += 1
        require(condition, message)

    def check(self):
        if self.clock is not None:
            require(self.clock.now_ns() - self.start < LIMITS["native_seconds"] * 10**9, "audit deadline")
        require(self.rss() <= LIMITS["rss_bytes"], "audit RSS limit")
        require(sum(p.stat().st_size for p in self.args.output.iterdir() if p.is_file())
                <= LIMITS["output_bytes"], "audit output limit")

    @staticmethod
    def rss():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)

    def digest(self, target):
        require(not any(p.is_symlink() for p in (target, *target.parents)) and target.is_file(),
                "regular nonsymlink evidence")
        h, size = hashlib.sha256(), 0
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                self.check()
                h.update(block)
                size += len(block)
        self.bytes_hashed += size
        return {"sha256": h.hexdigest(), "bytes": size}

    @staticmethod
    def path(name):
        p = Path(name)
        require(not p.is_absolute() and ".." not in p.parts and name, "relative manifest path")
        return ROOT / p

    def read(self, target):
        self.check()
        return decode(target.read_bytes())

    def lines(self, target):
        with target.open("rb") as stream:
            for line in stream:
                self.check()
                yield line

    def write(self, name, value):
        data = encoded(value)
        require(len(data) <= LIMITS["output_bytes"], "bounded audit artifact")
        with (self.args.output / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    def closed(self, directory, files, expected):
        self.agree(set(files) == expected and {p.name for p in directory.iterdir()} == expected | {"receipt.json"},
                   "exact worker payload closure")
        for name, desc in files.items():
            self.agree(Path(name).name == name and self.digest(directory / name) == desc, f"payload {name}")

    def authenticate(self):
        a = self.args
        for target, pin in ((a.plan, a.plan_sha256), (a.run / "receipt.json", a.receipt_sha256),
                            (a.terminal, a.terminal_sha256)):
            self.agree(self.digest(target)["sha256"] == pin, "external evidence pin")
        plan, worker, terminal = self.read(a.plan), self.read(a.run / "receipt.json"), self.read(a.terminal)
        self.agree(plan["version"] == worker["version"] == WORKER_VERSION
                   and plan["status"] == "frozen_before_reconstruction" and worker["status"] == "completed",
                   "completed fixed cohort worker")
        self.agree(plan["history"] == {"path": HISTORY, **self.digest(self.path(HISTORY))}
                   and plan["history"]["sha256"] == HISTORY_PIN, "frozen historical parent")
        history = self.read(self.path(HISTORY))
        self.agree(len(history["sources"]) == 138 and len(history["inputs"]) == 495
                   and plan["inputs"] == history["inputs"]
                   and set(plan["sources"]) == set(history["sources"]) | NEW_SOURCES
                   and all(plan["sources"][k] == v for k, v in history["sources"].items()), "full lineage joins")
        for name, pin in plan["sources"].items():
            self.agree(self.digest(self.path(name))["sha256"] == pin, f"source {name}")
        for name, desc in plan["inputs"].items():
            self.agree(self.digest(self.path(name)) == desc, f"input {name}")
        qualifications = {f"output/otto-teacher-cohort-v1/engineering-0{i}/receipt.json": test for i, test in
                          ((1, "tests/test_otto_teacher_cohort.py"), (2, "tests/test_prepare_otto_teacher_cohort.py"))}
        self.agree(set(plan["qualifications"]) == set(qualifications), "both engineering attempts bound")
        for name, test in qualifications.items():
            self.agree(self.digest(self.path(name)) == plan["qualifications"][name], "engineering receipt pin")
            engineering = self.read(self.path(name))
            self.agree(engineering["source_before"] == engineering["source_after"]
                       and engineering["commands"][0]["command"] == [plan["python_executable"], "-m", "pytest", "-q", test]
                       and type(engineering["commands"][0]["returncode"]) is int
                       and engineering["commands"][0]["returncode"] == 0, "successful original fabricated tests")
            if name.endswith("engineering-02/receipt.json"):
                self.agree(engineering["status"] == "passed", "final engineering passed")
            for source, pin in engineering["source_after"].items():
                self.agree(plan["sources"][source] == pin, "qualified source joins")
            for file, desc in engineering["files"].items():
                self.agree(Path(file).name == file and self.digest(self.path(name).parent / file) == desc,
                           "original engineering artifact")
        runtime = {"python_executable": sys.executable, "python_version": platform.python_version(),
                   "all_distributions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
        self.agree(all(runtime[k] == plan[k] == history[k] for k in runtime)
                   and plan["environment"] == THREADS, "qualified runtime and threads")
        self.agree(plan["limits"] == worker["limits"] == WORKER_LIMITS, "fixed worker caps")
        self.closed(a.run, worker["files"], PAYLOADS)
        self.agree(sum(p.stat().st_size for p in a.run.iterdir()) <= WORKER_LIMITS["output_bytes"]
                   and 0 < worker["peak_rss_bytes"] <= WORKER_LIMITS["rss_bytes"], "actual closed worker resources")
        started = self.read(a.run / "started.json")
        request = started["request"]
        supervision = Path(request["supervision"])
        self.agree(supervision.is_absolute() and supervision.name.endswith(".launch.json")
                   and a.terminal == supervision.with_name(supervision.name[:-12] + ".terminal.json"),
                   "original supervisor path join")
        self.agree(self.digest(supervision)["sha256"] == worker["supervision_sha256"], "worker launch hash")
        launch = self.read(supervision)
        self.agree(started["launch"] == launch and all(terminal[k] == v for k, v in launch.items()),
                   "parent launch and terminal identity")
        self.agree(request == {"mode": "run", "output": str(a.run), "plan": str(a.plan),
                              "plan_sha256": a.plan_sha256, "supervision": str(supervision)}, "exact worker request")
        command = list(launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        options = {"--plan": str(a.plan), "--plan-sha256": a.plan_sha256,
                   "--supervision": str(supervision), "--output": str(a.run)}
        self.agree(len(command) == 11 and command[:3] == [plan["python_executable"],
                   str(ROOT / "scripts/prepare_otto_teacher_cohort.py"), "run"]
                   and len(set(command[3::2])) == 4 and dict(zip(command[3::2], command[4::2], strict=True)) == options
                   and launch["cwd"] == str(ROOT) and launch["pid"] == launch["pgid"] > 0
                   and launch["parent_pid"] > 0 and launch["pid"] != launch["parent_pid"],
                   "original worker command and process")
        self.agree(terminal["version"] == "dialogue-observation-supervision-v2"
                   and terminal["status"] == "completed" and type(terminal["returncode"]) is int
                   and terminal["returncode"] == 0
                   and terminal["group_absent"] is True and terminal["timed_out"] is False
                   and terminal["timing_available"] is True and terminal["error"] is None
                   and terminal["clock_error"] is None and terminal["cleanup"]["reaped"] is True
                   and terminal["cleanup"]["group_absent"] is True
                   and terminal["cleanup"]["errors"] == [], "successful original supervisor")
        self.agree(launch["cap_seconds"] == 600 and launch["clock_backend"] == worker["clock_backend"]
                   == self.clock.backend and launch["clock_source_sha256"] == plan["sources"][CLOCK] == CLOCK_PIN
                   and launch["watchdog_sha256"] == plan["sources"]["scripts/supervise_dialogue_observation_v2.py"]
                   and launch["deadline_ns"] == launch["started_ns"] + 600 * 10**9
                   and launch["started_ns"] <= worker["started_ns"] == started["started_ns"]
                   <= worker["finished_ns"] <= terminal["finished_ns"] < launch["deadline_ns"]
                   and terminal["elapsed_ns"] == terminal["finished_ns"] - launch["started_ns"]
                   and terminal["wall_seconds"] == terminal["elapsed_ns"] / 1e9
                   and worker["wall_seconds"] == (worker["finished_ns"] - worker["started_ns"]) / 1e9,
                   "joined native elapsed intervals")
        self.agree(worker["plan_sha256"] == a.plan_sha256 and worker["pending"] == []
                   and worker["requires_successful_original_supervisor"] is True
                   and all(type(worker[k]) is int and worker[k] == 0 for k in
                           ("native_calls", "source_draws", "analytic_choices", "learned_model_calls", "optimizer_calls")),
                   "closed worker zero-action scope")
        return plan, worker

    def reconstruction(self, selected):
        wanted = {}
        for row in selected:
            wanted.setdefault(row["episode_id"], {})[row["prefix_index"]] = row
        saved = iter(self.lines(self.args.run / "reconstruction.jsonl"))
        states, captured = {}, set()
        counts = {"public_reset": 0, "public_update": 0}
        operation_id = 0
        for raw in self.lines(self.path(f"{SYMM}/collection-transitions.jsonl")):
            match = HEADER.match(raw.decode("utf-8"))
            require(match is not None, "original compact public header")
            kind, episode = match.groups()
            if episode not in wanted:
                continue
            token = decode(raw, lexical=True)
            step = 0 if kind == "reset" else integer(numbers(token["step"]), 1, 2188)
            if step > max(wanted[episode]):
                continue
            packet = public(numbers(token["public"]), step)
            after = witness(numbers(token["posterior_after"]))
            identity = next(iter(wanted[episode].values()))
            if kind == "reset":
                self.agree(episode not in states and all(numbers(token[k]) == identity[k] for k in IDENTITY)
                           and token["block"] is None and packet["position"] == [26, 26]
                           and packet["hit"] == identity["initial_hit"] and packet["done"] is False,
                           "original TRAIN reset identity")
            else:
                self.agree(episode in states, "original reset precedes updates")
                before, previous = states[episode]
                action = integer(numbers(token["action"]), 0, 3)
                self.agree(step == before["step"] + 1 and before["done"] is False
                           and numbers(token["allowed_actions"]) == before["valid_actions"]
                           and action in before["valid_actions"] and witness(numbers(token["posterior_before"])) == previous,
                           "public chronology and preceding witness")
                moved = list(before["position"])
                moved[action // 2] += 2 * (action % 2) - 1
                self.agree(packet["position"] == moved, "recorded public movement")
            operation = "public_reset" if kind == "reset" else "public_update"
            attempt = {"event": "attempt", "operation_id": operation_id, "operation": operation,
                       "episode_id": episode, "step": step}
            capture = wanted[episode].get(step)
            ids = [] if capture is None else [capture["anchor_id"]]
            returned = {**attempt, "event": "return", "public": packet, "posterior": after,
                        "captured_anchor_ids": ids}
            self.agree(decode(next(saved)) == attempt and decode(next(saved)) == returned,
                       "exact original-public reconstruction attempt/return")
            if capture is not None:
                self.agree(packet == capture["public"] and after == capture["posterior"], "selected capture witness")
                captured.update(ids)
            states[episode] = packet, after
            counts[operation] += 1
            operation_id += 1
        self.agree(next(saved, None) is None and captured == set(range(len(selected)))
                   and set(states) == set(wanted)
                   and all(states[e][0]["step"] == max(wanted[e]) for e in wanted), "complete reconstruction coverage")
        return {"anchors": len(selected), "episodes": len(states), "public_resets": counts["public_reset"],
                "public_updates": counts["public_update"]}, counts

    def compute(self, plan, worker):
        selected = select_rows(self.lines(self.path(f"{SYMM}/dagger-rows.jsonl")))
        self.agree(selected == plan["cohort"], "independently frozen cohort selection")
        rows = selected["selections"]
        reconstruction, calls = self.reconstruction(rows)
        import numpy as np

        with np.load(self.args.run / "anchors.npz", allow_pickle=False) as arrays:
            self.agree(set(arrays.files) == {"beliefs", "kernel_lambda3", "kernel_lambda4"}, "captured array schema")
            beliefs = arrays["beliefs"]
            self.agree(beliefs.dtype == np.float64 and beliefs.shape == (len(rows), 53, 53), "all float64 captures")
            for regime in FIRST:
                kernel = arrays[f"kernel_{regime}"]
                with np.load(self.path(f"{SYMM}/kernel-{regime}.npz"), allow_pickle=False) as original:
                    self.agree(set(original.files) == {"likelihood", "initial_hit_weights"}, "historical kernel keys")
                    source = original["likelihood"]
                self.agree(kernel.dtype == source.dtype == np.float64 and kernel.shape == source.shape == (4, 107, 107)
                           and hashlib.sha256(kernel.tobytes(order="C")).digest()
                           == hashlib.sha256(source.tobytes(order="C")).digest()
                           and np.isfinite(kernel).all() and (kernel >= 0).all() and (kernel <= 1).all()
                           and (kernel[:, 53, 53] == 0).all(), "exact owned public kernel bytes and validity")
        beliefs.setflags(write=False)
        assessments = []
        ledger = iter(self.lines(self.args.run / "support.jsonl"))
        for i, row in enumerate(rows):
            self.check()
            belief = beliefs[i]
            mass = float(np.sum(belief, dtype=np.float64))
            actual = {"sha256": hashlib.sha256(belief.tobytes(order="C")).hexdigest(), "mass": mass}
            self.agree(actual == row["posterior"], "exact captured posterior hash and mass")
            message = None
            if not np.isfinite(belief).all() or (belief < 0).any():
                message = "anchor belief must be finite and nonnegative"
            elif not math.isfinite(mass) or abs(mass - 1.0) > 1e-10:
                message = "anchor belief must already be normalized; no repair"
            elif belief[tuple(row["public"]["position"])] != 0:
                message = "nonterminal anchor must have exactly zero current-cell mass"
            error = None if message is None else {"type": "ValueError", "message": message}
            assessment = {"anchor_id": i, "episode_id": row["episode_id"], "prefix_index": row["prefix_index"],
                          "supported": error is None, "error": error}
            assessments.append(assessment)
            attempt = {"event": "attempt", "operation_id": i, "operation": "anchor_support", "anchor_id": i,
                       "episode_id": row["episode_id"], "step": row["prefix_index"]}
            returned = {**attempt, "event": "return", "supported": error is None,
                        "error": error, "snapshot_returned": error is None}
            self.agree(decode(next(ledger)) == attempt and decode(next(ledger)) == returned,
                       "independent support result and attempted/returned witness")
        self.agree(next(ledger, None) is None, "support ledger closure")
        supported = sum(r["supported"] for r in assessments)
        support = {"anchors": len(rows), "supported": supported, "unsupported": len(rows) - supported,
                   "snapshot_attempts": len(rows), "snapshot_returns": supported}
        self.agree(self.read(self.args.run / "support.json") == {"assessments": assessments, "counts": support},
                   "every independent support assessment retained")
        calls["anchor_support"] = len(rows)
        self.agree(worker["calls"] == {k: {"attempted": v, "returned": v} for k, v in calls.items()}
                   and worker["events"] == 2 * sum(calls.values())
                   and worker["events"] <= 631296
                   and worker["reconstruction"] == reconstruction and worker["support"] == support
                   and worker["eligible_for_sampling"] is (supported == len(rows)), "complete counters and eligibility")
        return {"version": VERSION, "agreement": True, "selection": selected["counts"],
                "reconstruction": reconstruction, "support": support, "assessments": assessments,
                "all_selected_supported": supported == len(rows), "events": worker["events"],
                "scope": SCOPE, "constructor_calls": 0, "posterior_filter_updates": 0}

    def execute(self):
        a = self.args
        a.output.mkdir(parents=False, exist_ok=False)
        receipt = {"version": VERSION, "status": "started", "agreement": False, "limits": LIMITS,
                   "plan_sha256": a.plan_sha256, "worker_sha256": a.receipt_sha256,
                   "terminal_sha256": a.terminal_sha256, "environment": THREADS, "scope": SCOPE,
                   "constructor_calls": 0, "model_calls": 0, "simulator_calls": 0, "optimizer_calls": 0}

        def interrupted(_number, _frame):
            raise TimeoutError("audit deadline or termination")

        old_alarm, old_term = signal.signal(signal.SIGALRM, interrupted), signal.signal(signal.SIGTERM, interrupted)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["native_seconds"])
        try:
            require(all(os.environ.get(k) == v for k, v in THREADS.items()), "one numerical thread before imports")
            require(self.digest(self.path(CLOCK))["sha256"] == CLOCK_PIN, "clock source pin")
            sys.path.insert(0, str(ROOT / "src"))
            from openjev.research.suspend_clock import SuspendClock

            self.clock = SuspendClock()
            require(self.clock.backend in ("mach_continuous_time", "CLOCK_BOOTTIME"), "native elapsed clock")
            self.start = self.clock.now_ns()
            own = self.digest(Path(__file__).resolve())
            receipt.update(source=own, started_ns=self.start, clock_backend=self.clock.backend)
            self.write("started.json", {"request": {k: str(v) for k, v in vars(a).items()},
                                        "source": own, "limits": LIMITS, "environment": THREADS})
            plan, worker = self.authenticate()
            summary = self.compute(plan, worker)
            for name, pin in plan["sources"].items():
                self.agree(self.digest(self.path(name))["sha256"] == pin, "source unchanged at close")
            self.closed(a.run, worker["files"], PAYLOADS)
            for target, pin in ((a.plan, a.plan_sha256), (a.run / "receipt.json", a.receipt_sha256),
                                (a.terminal, a.terminal_sha256), (Path(__file__).resolve(), own["sha256"])):
                self.agree(self.digest(target)["sha256"] == pin, "final immutable evidence pin")
            summary["comparisons"] = self.comparisons
            self.write("summary.json", summary)
            files = {name: self.digest(a.output / name) for name in ("started.json", "summary.json")}
            receipt.update(status="completed", agreement=True, comparisons=self.comparisons,
                           bytes_hashed=self.bytes_hashed, finished_ns=self.clock.now_ns(), peak_rss_bytes=self.rss(),
                           files=files,
                           timing_scope="Through audit payload hashing; receipt publication and process exit follow.")
            receipt["wall_seconds"] = (receipt["finished_ns"] - self.start) / 1e9
            self.check()
            self.write("receipt.json", receipt)
            self.check()
            print(json.dumps({"status": "completed", "agreement": True, "receipt": self.digest(a.output / "receipt.json")}))
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            receipt.update(status="failed", agreement=False, error=repr(error), traceback=traceback.format_exc(),
                           comparisons=self.comparisons)
            try:
                receipt["failure_elapsed_seconds"] = ((self.clock.now_ns() - self.start) / 1e9
                                                       if self.clock is not None and self.start is not None else None)
            except BaseException as secondary:  # noqa: BLE001 - retain primary failure
                receipt["failure_clock_error"] = repr(secondary)
            try:
                target = a.output / "receipt.json"
                if target.exists():
                    target.rename(a.output / "receipt.invalid.json")
                self.write("receipt.json", receipt)
            except BaseException as secondary:  # noqa: BLE001 - publishing failure must not replace original
                error.add_note(f"Failure receipt publication: {secondary!r}")
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)
            signal.signal(signal.SIGTERM, old_term)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "run", "terminal", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("plan-sha256", "receipt-sha256", "terminal-sha256"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    require(all(getattr(args, k).is_absolute() for k in ("plan", "run", "terminal", "output")), "absolute paths required")
    require(all(re.fullmatch(r"[0-9a-f]{64}", getattr(args, k)) for k in
                ("plan_sha256", "receipt_sha256", "terminal_sha256")), "external SHA-256 pins required")
    Audit(args).execute()


if __name__ == "__main__":
    main()
