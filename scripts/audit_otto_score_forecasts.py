"""Independent saved-score forecast audit; no model, optimizer or simulator calls.

NumPy is used only to decode saved arrays and compare their bytes. Forecast
metrics and continuation rules are independently reduced from saved predictions.
The saved training updates, model inference and collector/filter truth remain
inherited authenticated evidence, not replayed here.
"""
from __future__ import annotations

import argparse
import collections
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
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/audit_otto_score_forecasts.py"
TEST = "tests/test_audit_otto_score_forecasts.py"
VERSION = "otto-score-forecast-saved-audit-v1"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
SUPERVISOR_PIN = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
LIMITS = {"seconds": 120, "rss_bytes": 2 * 1024**3, "output_bytes": 128 * 1024**2}
FAMILIES = ("residual_gru", "direct_gru", "history_mlp", "current_mlp")
SEEDS = (225001, 225002, 225003)
REGIMES = ("lambda3", "lambda4")
PARAMETERS = {"residual_gru": 5862, "direct_gru": 5862, "current_mlp": 5826, "history_mlp": 5895}
PRODUCER = "scripts/train_otto_score_forecasts.py"
PRODUCER_VERSION = "otto-score-forecast-training-v1"
COLLECTOR = "scripts/collect_otto_score_forecasts.py"
COLLECTOR_VERSION = "otto-score-forecast-collection-v1"
ARMS = ("analytic", "neural", "period4_hold")
CONFIG = {"families": list(FAMILIES), "fit_seeds": list(SEEDS), "epochs": 80, "batch_windows": 32,
          "learning_rate": .003, "gradient_clip": 5., "weight_decay": 0., "scale": 64.,
          "training_episodes": 54, "validation_episodes": 36, "required_conditions": 45,
          "checkpoint": "last", "device": "cpu", "dtype": "float32"}
SHAPES = {"residual_gru": {"recurrent.weight_ih": (87, 35), "recurrent.weight_hh": (87, 29),
                          "recurrent.bias_ih": (87,), "recurrent.bias_hh": (87,),
                          "output.weight": (4, 29), "output.bias": (4,)},
          "history_mlp": {"hidden.weight": (43, 132), "hidden.bias": (43,),
                          "output.weight": (4, 43), "output.bias": (4,)},
          "current_mlp": {"hidden.weight": (82, 66), "hidden.bias": (82,),
                          "output.weight": (4, 82), "output.bias": (4,)}}
SHAPES["direct_gru"] = dict(SHAPES["residual_gru"])
LIMITATIONS = [
    "Saved predictions are not recomputed through neural models; optimizer updates, gradients and timings are inherited.",
    "Collected teacher values, native trajectories and public-filter numerical correctness are inherited authenticated evidence.",
    "Independent checks cover complete saved windows, prediction anchors/padding, scalar metrics, fixed rules and process/file joins.",
    "Forced-path teacher-score forecasting is not autonomous control, true action value, biological wiring or novelty evidence.",
    "Initial GRU equality is checked from saved initial hashes; initial tensors are not stored or regenerated.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def near_set(scores, legal):
    allowed = [i for i, value in enumerate(legal) if value]
    require(len(scores) == len(legal) == 4 and allowed and all(math.isfinite(float(x)) for x in scores),
            "four finite scores and legal support")
    minimum = min(float(scores[i]) for i in allowed)
    near = [i for i in allowed if abs(f32(float(scores[i]) - minimum)) < f32(1e-10)]
    return near, minimum


def scalar_metrics(predictions, targets, legal, lengths, episode_index, ids, regimes, *, check=None):
    """Different reduction route: per-episode scalar lists, then fixed denominators."""
    require(len(ids) == len(regimes) and ids and len(set(ids)) == len(ids), "unique episode metadata")
    require(len(predictions) == len(targets) == len(legal) == len(lengths) == len(episode_index),
            "aligned window arrays")
    samples = [[] for _ in ids]
    for window, (length, index) in enumerate(zip(lengths, episode_index, strict=True)):
        length, index = int(length), int(index)
        require(1 <= length <= 4 and 0 <= index < len(ids), "active window and episode index")
        for age in range(1, length):
            p, t, mask = predictions[window][age], targets[window][age], legal[window][age]
            p_near, _ = near_set(p, mask)
            t_near, t_min = near_set(t, mask)
            action = p_near[0]
            actions = [i for i in range(4) if mask[i]]
            p_mean = math.fsum(float(p[i]) for i in actions) / len(actions)
            t_mean = math.fsum(float(t[i]) for i in actions) / len(actions)
            error = math.fsum(((float(p[i]) - p_mean) - (float(t[i]) - t_mean)) ** 2
                              for i in actions) / len(actions)
            samples[index].append({"age": age, "agreement": float(action in t_near),
                                   "gap": float(t[action]) - t_min,
                                   "first": float(action == t_near[0]), "mse": error})
        if check is not None and window % 256 == 0:
            check()

    def reduce(indices, age=None):
        values = {name: [] for name in ("agreement", "gap", "first", "mse")}
        empty, count, support = [], 0, 0
        for index in indices:
            selected = [row for row in samples[index] if age is None or row["age"] == age]
            count += len(selected)
            if not selected:
                empty.append(ids[index])
                continue
            support += 1
            for name, value in values.items():
                value.append(math.fsum(row[name] for row in selected) / len(selected))
        totals = {name: math.fsum(value) for name, value in values.items()}
        return {"episodes": len(indices), "supported_episodes": support, "zero_support_episode_ids": empty,
                "nonquery_rows": count, "weight_mass": support / len(indices),
                "episode_weighted_agreement": totals["agreement"] / len(indices),
                "episode_weighted_raw_gap": totals["gap"] / len(indices),
                "episode_weighted_first_argmin_match": totals["first"] / len(indices),
                "episode_weighted_centered_mse": totals["mse"] / len(indices),
                "supported_episode_agreement": totals["agreement"] / support if support else None,
                "supported_episode_raw_gap": totals["gap"] / support if support else None,
                "supported_episode_centered_mse": totals["mse"] / support if support else None}

    def group(indices):
        return {**reduce(indices), "by_age": {str(age): reduce(indices, age) for age in (1, 2, 3)}}

    return {"version": "otto-score-forecast-data-v1",
            "scope": "saved teacher-score imitation; no search-return claim",
            "overall": group(list(range(len(ids)))),
            "by_regime": {regime: group([i for i, value in enumerate(regimes) if value == regime])
                          for regime in dict.fromkeys(regimes)}}


def continuation_rules(models, hold, support):
    """All 45 prospectively required comparisons, with exact zero-gap semantics."""
    indexed = {(row["family"], row["seed"]): row["metrics"] for row in models}
    require(len(models) == len(indexed) == 12 and set(indexed) == {(f, s) for f in FAMILIES for s in SEEDS},
            "all twelve fixed final model metrics")
    result = []

    def rule(name, value, relation, threshold):
        require(type(value) in (int, float) and type(threshold) in (int, float)
                and math.isfinite(value) and math.isfinite(threshold), "finite rule operands")
        result.append({"name": name, "value": value, "relation": relation, "threshold": threshold,
                       "passes": value >= threshold if relation == ">=" else value <= threshold})

    rule("technical_complete_requires_closed_saved_audit", 1, ">=", 1)
    for regime in REGIMES:
        for age in (1, 2, 3):
            rule(f"{regime}.age{age}.case_support", support[regime][str(age)], ">=", 4)
        for seed in SEEDS:
            candidate = indexed["residual_gru", seed]["by_regime"][regime]
            baseline = hold["by_regime"][regime]
            rule(f"{regime}.{seed}.agreement_vs_hold", candidate["episode_weighted_agreement"],
                 ">=", baseline["episode_weighted_agreement"])
            rule(f"{regime}.{seed}.gap_vs_hold", candidate["episode_weighted_raw_gap"],
                 "<=", .8 * baseline["episode_weighted_raw_gap"])
            for age in (1, 2, 3):
                rule(f"{regime}.{seed}.age{age}.gap_vs_hold",
                     candidate["by_age"][str(age)]["episode_weighted_raw_gap"], "<=",
                     baseline["by_age"][str(age)]["episode_weighted_raw_gap"])
        for control in ("history_mlp", "direct_gru"):
            for field, factor, relation in (("episode_weighted_agreement", 1., ">="),
                                            ("episode_weighted_raw_gap", .9, "<=")):
                own = math.fsum(indexed["residual_gru", seed]["by_regime"][regime][field]
                                for seed in SEEDS) / 3
                other = math.fsum(indexed[control, seed]["by_regime"][regime][field] for seed in SEEDS) / 3
                rule(f"{regime}.mean.{field}_vs_{control}", own, relation, factor * other)
    require(len(result) == 45 and len({r["name"] for r in result}) == 45, "exact 45-rule definition")
    return result


def cohort():
    result, global_case = [], 0
    for stage, size, starts in (("train", 9, (22100001, 22200001)), ("valid", 6, (22300001, 22400001))):
        for regime, start in zip(REGIMES, starts, strict=True):
            for case in range(size):
                shift = global_case % 3
                for arm in ARMS[shift:] + ARMS[:shift]:
                    result.append({"stage": stage, "episode_index": len(result), "regime": regime,
                                   "seed": start + case, "case": case, "initial_hit": 1 + case % 3,
                                   "arm": arm, "episode_id": f"{stage}:{regime}:{start + case}:{arm}"})
                global_case += 1
    return result


def closed_parent(worker, terminal, launch, cap):
    require(worker["status"] == "completed" and worker["complete"] is True, "completed original worker")
    require(terminal["status"] == "completed" and terminal["returncode"] == 0 and terminal["timed_out"] is False
            and terminal["error"] is terminal["clock_error"] is None and terminal["group_absent"] is True
            and terminal["cleanup"]["reaped"] is True and terminal["cleanup"]["errors"] == [],
            "successful fully closed original parent")
    require(all(terminal[k] == v for k, v in launch.items()), "terminal retains original launch")
    require(launch["cwd"] == str(ROOT) and launch["cap_seconds"] == cap
            and launch["clock_source_sha256"] == CLOCK_PIN and launch["watchdog_sha256"] == SUPERVISOR_PIN
            and launch["deadline_ns"] == launch["started_ns"] + cap * 10**9
            and launch["started_ns"] <= worker["started_ns"] < worker["finished_ns"]
            <= terminal["finished_ns"] <= launch["deadline_ns"]
            and terminal["elapsed_ns"] == terminal["finished_ns"] - launch["started_ns"]
            and terminal["wall_seconds"] == terminal["elapsed_ns"] / 1e9
            and worker["wall_seconds"] == (worker["finished_ns"] - worker["started_ns"]) / 1e9,
            "original allocation and physical elapsed arithmetic")


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
        self.failure = False
        self.counts = collections.Counter()
        self.receipt = {"version": VERSION, "status": "started", "agreement": False, "limits": LIMITS,
                        "native_calls": 0, "model_calls": 0, "optimizer_calls": 0,
                        "limitations": LIMITATIONS, "failures": []}

    def require(self, condition, message):
        self.counts["checks"] += 1
        require(condition, message)
        if self.clock is not None and self.launch is not None and not self.failure and self.counts["checks"] % 256 == 0:
            self.check()

    def check(self):
        if self.clock.now_ns() >= self.launch["deadline_ns"]:
            raise TimeoutError("original saved-audit deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        if rss > LIMITS["rss_bytes"]:
            raise MemoryError("saved-audit RSS bound")
        if self.out.exists() and sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()) > LIMITS["output_bytes"]:
            raise OSError("saved-audit output bound")

    def sha(self, path):
        value = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                value.update(chunk)
                if self.clock is not None and self.launch is not None and not self.failure:
                    self.check()
        return value.hexdigest()

    def descriptor(self, path):
        return {"path": str(path.resolve()), "sha256": self.sha(path), "bytes": path.stat().st_size}

    def pinned(self, path, descriptor):
        self.require(path.is_file() and self.sha(path) == descriptor["sha256"]
                     and path.stat().st_size == descriptor["bytes"], f"pinned bytes {path}")

    def equal(self, actual, expected, label):
        """Schema is exact; inherited arithmetic uses its unchanged tight tolerance."""
        if isinstance(expected, dict):
            self.require(isinstance(actual, dict) and set(actual) == set(expected), f"{label} keys")
            for key in expected:
                self.equal(actual[key], expected[key], f"{label}.{key}")
        elif isinstance(expected, (list, tuple)):
            self.require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), f"{label} length")
            for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
                self.equal(a, b, f"{label}[{i}]")
        elif type(expected) is float:
            self.require(type(actual) in (int, float) and math.isfinite(actual)
                         and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12), label)
        else:
            self.require(type(actual) is type(expected) and actual == expected, label)

    def arrays(self, path):
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            self.require(len({x.filename for x in infos}) == len(infos)
                         and sum(x.file_size for x in infos) <= 1024**3, "bounded unique NPZ members")
        with self.np.load(path, allow_pickle=False) as archive:
            result = {name: archive[name] for name in archive.files}
        self.counts["npz_decodes"] += 1
        self.check()
        return result

    def path(self, value):
        p = Path(value)
        p = p if p.is_absolute() else ROOT / p
        self.require(p.is_file() and p.is_relative_to(ROOT) and ".." not in p.parts
                     and not any(q.is_symlink() for q in (p, *p.parents)), "contained regular evidence")
        return p

    def read(self, value):
        return json.loads(self.path(value).read_text())

    def rows(self, value):
        with self.path(value).open() as stream:
            for line in stream:
                self.require(len(line) <= 2 * 1024**2, "bounded JSON evidence row")
                yield json.loads(line)

    def admit(self):
        self.require(self.sha(self.path(CLOCK)) == CLOCK_PIN, "qualified clock before import")
        spec = importlib.util.spec_from_file_location("_forecast_audit_clock", ROOT / CLOCK)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.clock = module.SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            self.require(self.clock.now_ns() - self.start < 5 * 10**9, "original audit launch available")
            time.sleep(.01)
        self.launch = self.read(self.args.supervision)
        cmd = list(self.launch["command"])
        if cmd[1:2] == ["-u"]:
            cmd.pop(1)
        self.require(cmd == [sys.executable, *sys.argv] and self.launch["pid"] == os.getpid()
                     and self.launch["pgid"] == os.getpgrp() and self.launch["parent_pid"] == os.getppid()
                     and self.launch["cwd"] == str(ROOT) == str(Path.cwd())
                     and self.launch["cap_seconds"] == LIMITS["seconds"]
                     and self.launch["clock_backend"] == self.clock.backend
                     and self.launch["clock_source_sha256"] == CLOCK_PIN
                     and self.launch["watchdog_sha256"] == SUPERVISOR_PIN
                     and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
                     and self.launch["deadline_ns"] == self.launch["started_ns"] + 120 * 10**9,
                     "original bounded saved-audit process")
        self.inputs = {}
        for role in ("plan", "worker", "terminal"):
            path = self.path(getattr(self.args, role))
            descriptor = self.descriptor(path)
            self.require(descriptor["sha256"] == getattr(self.args, role + "_sha256"), "external " + role)
            self.inputs[role] = descriptor
        self.plan = self.read(self.args.plan)
        self.require(self.plan["version"] == PRODUCER_VERSION and self.plan["status"] == "frozen_before_fitting"
                     and self.plan["config"] == CONFIG
                     and self.plan["limits"] == {"seconds": 600, "rss_bytes": 4 * 1024**3,
                                                 "output_bytes": 2 * 1024**3}, "fixed training allocation")
        self.require({SELF, TEST, CLOCK, SUPERVISOR} <= self.plan["sources"].keys(), "audit source frozen before fitting")
        for name, pin in self.plan["sources"].items():
            self.require(self.sha(self.path(name)) == pin, "frozen source " + name)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.require(os.environ.get(name) == "1", "single numerical thread")
        self.receipt.update(plan_sha256=self.args.plan_sha256, producer_inputs=self.inputs,
                            sources={name: self.plan["sources"][name] for name in (SELF, TEST, CLOCK, SUPERVISOR)},
                            supervision_sha256=self.sha(self.args.supervision))
        write(self.out / "started.json", {"started_ns": self.start, "launch": self.launch,
                                           "producer_inputs": self.inputs})

    def phase(self, plan_path, worker_path, terminal_path, script, interpreter, cap):
        plan, worker, terminal = self.read(plan_path), self.read(worker_path), self.read(terminal_path)
        command = list(terminal["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        self.require(command[:3] == [str(ROOT / interpreter), str(ROOT / script), "run"], "actual original absolute command")
        self.require(len(command[3:]) == 8 and len(set(command[3::2])) == 4, "unique original command options")
        options = dict(zip(command[3::2], command[4::2], strict=True))
        self.require(set(options) == {"--plan", "--plan-sha256", "--supervision", "--output"}
                     and options["--plan"] == str(plan_path)
                     and options["--plan-sha256"] == self.sha(plan_path) == worker["plan_sha256"]
                     and options["--output"] == str(worker_path.parent), "original phase input/output joins")
        launch_path = self.path(options["--supervision"])
        launch = self.read(launch_path)
        self.require(self.sha(launch_path) == worker["supervision_sha256"], "original launch hash")
        closed_parent(worker, terminal, launch, cap)
        self.equal(self.read(worker_path.parent / "started.json")["launch"], launch, "saved original launch")
        self.equal(worker["sources"], plan["sources"], "phase source map")
        self.equal(worker["inputs"], plan["inputs"], "phase input map")
        if "native_inputs" in plan:
            self.equal(worker["native_inputs"], plan["native_inputs"], "native input map")
        self.require(not worker.get("cleanup_errors") and not worker.get("pending")
                     and worker.get("pending_emission") is None and worker.get("pending_episode") is None
                     and worker.get("pending_action") is None,
                     "no unresolved phase operation")
        directory = worker_path.parent
        self.require({p.name for p in directory.iterdir()} == set(worker["files"]) | {"receipt.json"},
                     "closed phase inventory")
        for name, descriptor in worker["files"].items():
            self.require(Path(name).name == name, "flat payload name")
            self.pinned(self.path(directory / name), descriptor)
        self.require(worker["peak_rss_bytes"] <= plan["limits"]["rss_bytes"]
                     and sum(d["bytes"] for d in worker["files"].values()) <= plan["limits"]["output_bytes"],
                     "original phase resource accounting")
        for name, pin in plan["sources"].items():
            self.require(self.sha(self.path(name)) == pin, "phase source " + name)
        for section in ("inputs", "native_inputs"):
            for descriptor in plan.get(section, {}).values():
                self.pinned(self.path(descriptor["path"]), descriptor)
        self.counts["closed_phases"] += 1
        return plan, worker, terminal, directory

    def authenticate(self):
        _, self.worker, self.parent, self.run = self.phase(
            self.args.plan, self.args.worker, self.args.terminal, PRODUCER, ".venv/bin/python", 600)
        self.require(self.worker["version"] == PRODUCER_VERSION and self.worker["fits_completed"] == 12
                     and self.worker["teacher_calls"] == self.worker["native_calls"] == 0,
                     "complete forecast fit worker without scientific collection")
        roles = self.plan["inputs"]
        self.require(set(roles) == {"collection_plan", "collection_receipt", "collection_terminal", "engineering"},
                     "exact training source roles")
        self.collection_plan, self.collection_worker, self.collection_parent, self.collection = self.phase(
            self.path(roles["collection_plan"]["path"]), self.path(roles["collection_receipt"]["path"]),
            self.path(roles["collection_terminal"]["path"]), COLLECTOR, ".venv-otto-released-native/bin/python", 900)
        c = self.collection_worker
        collection_payloads = {name + ".jsonl.gz" for name in ("work", "weights", "forwards", "transitions", "samples")}
        collection_payloads.update({"started.json", "runtime.json", "setup.json", "deployment.json", "cohort.json",
                                    "episode-boundaries.jsonl", "episodes.jsonl", "train.npz", "valid.npz",
                                    "costs.json", "summary.json"})
        self.require(self.collection_plan["version"] == COLLECTOR_VERSION
                     and self.collection_plan["status"] == "frozen_before_collection"
                     and self.collection_plan["limits"] == {"native_seconds": 900, "rss_bytes": 4 * 1024**3,
                                                             "output_bytes": 2 * 1024**3}
                     and set(c["files"]) == set(self.collection_plan["payloads"]) == collection_payloads,
                     "original complete collection allocation")
        self.require(c["version"] == COLLECTOR_VERSION and c["completed_episodes"] == 90
                     and c["train_episodes"] == 54 and c["valid_episodes"] == 36
                     and c["training_updates"] == 0, "complete original 90-path collection")
        self.equal(self.collection_plan["cohort"], cohort(), "exact prospective collection cohort")
        self.episodes = list(self.rows(self.collection / "episodes.jsonl"))
        self.require(len(self.episodes) == 90, "all complete episode records")
        for row, identity in zip(self.episodes, cohort(), strict=True):
            self.equal({key: row[key] for key in identity}, identity, "ordered episode identity")
            steps = row["steps"]
            self.require(type(steps) is int and 1 <= steps <= 2188 and row["rows"] == steps
                         and row["updates"] == row["teacher_calls"] == steps
                         and row["final_update_assimilated"] is True
                         and type(row["found"]) is bool and row["censored"] is (not row["found"])
                         and (row["found"] or steps == 2188)
                         and row["corrections"] == (steps + 3) // 4,
                         "all path tails and final updates retained")
        engineering = self.read(roles["engineering"]["path"])
        self.require(engineering["status"] == "passed"
                     and engineering["sources_before"] == engineering["sources_after"]
                     and all(row["exit_code"] == 0 for row in engineering["results"]), "qualified engineering scope")

    def reconstruct(self, stage):
        np = self.np
        flat = self.arrays(self.collection / f"{stage}.npz")
        self.require(set(flat) == {"features", "raw_q", "legal", "actions", "correction", "episode_offsets"},
                     "exact saved flat array keys")
        identities = [row for row in self.episodes if row["stage"] == stage]
        offsets = flat["episode_offsets"]
        self.require(offsets.dtype == np.int64 and offsets.shape == (len(identities) + 1,)
                     and offsets[0] == 0 and bool((np.diff(offsets) >= 1).all())
                     and bool((np.diff(offsets) <= 2188).all()), "complete ordered flat episode offsets")
        total = int(offsets[-1])
        for name, dtype, shape in (("features", np.float32, (total, 31)), ("raw_q", np.float32, (total, 4)),
                                   ("legal", np.bool_, (total, 4)), ("actions", np.int64, (total,)),
                                   ("correction", np.bool_, (total,))):
            value = flat[name]
            self.require(value.dtype == dtype and value.shape == shape, "exact flat " + name)
        self.require(bool(np.isfinite(flat["features"]).all()) and bool(np.isfinite(flat["raw_q"]).all())
                     and bool(flat["legal"].any(axis=1).all()), "finite public rows and legal support")
        lengths = [int(hi - lo) for lo, hi in pairwise(offsets)]
        n = sum((length + 3) // 4 for length in lengths)
        output = {"features": np.zeros((n, 4, 31), np.float32), "query_scores": np.zeros((n, 4), np.float32),
                  "targets": np.zeros((n, 4, 4), np.float32), "legal": np.zeros((n, 4, 4), np.bool_),
                  "valid_mask": np.zeros((n, 4), np.bool_), "nonquery_mask": np.zeros((n, 4), np.bool_),
                  "lengths": np.zeros(n, np.int64), "episode_index": np.zeros(n, np.int64),
                  "step_offsets": np.zeros(n, np.int64), "nonquery_weights": np.zeros((n, 4), np.float64),
                  "episode_lengths": np.asarray(lengths, np.int64),
                  "episode_nonquery_counts": np.asarray([length - (length + 3) // 4 for length in lengths], np.int64)}
        window = 0
        for index, (identity, lo, hi, length) in enumerate(zip(identities, offsets[:-1], offsets[1:], lengths, strict=True)):
            lo, hi = int(lo), int(hi)
            self.require(identity["start_row"] == lo and identity["end_row"] == hi
                         and identity["steps"] == length, "exact saved per-episode rows")
            x, actions = flat["features"][lo:hi], flat["actions"][lo:hi]
            steps = np.arange(length)
            self.require(np.array_equal(x[:, 15], (steps / 2188).astype(np.float32))
                         and np.array_equal(x[:, 16], ((steps % 4) / 2188).astype(np.float32))
                         and bool((x[:, 17] == 1).all())
                         and np.array_equal(flat["correction"][lo:hi], steps % 4 == 0)
                         and bool(((actions >= 0) & (actions < 4)).all())
                         and bool(flat["legal"][lo:hi][steps, actions].all()), "public chronology and virtual corrections")
            count = int(output["episode_nonquery_counts"][index])
            for start in range(0, length, 4):
                size = min(4, length - start)
                for key, original in (("features", "features"), ("targets", "raw_q"), ("legal", "legal")):
                    output[key][window, :size] = flat[original][lo + start:lo + start + size]
                output["query_scores"][window] = flat["raw_q"][lo + start]
                output["valid_mask"][window, :size] = True
                output["nonquery_mask"][window, 1:size] = True
                output["lengths"][window] = size
                output["episode_index"][window] = index
                output["step_offsets"][window] = start
                if count:
                    output["nonquery_weights"][window, 1:size] = 1 / (len(identities) * count)
                window += 1
            self.check()
        counts = {"episodes": len(identities), "windows": n, "rows": total, "query_rows": n,
                  "nonquery_rows": total - n, "zero_support_episodes": sum(length == 1 for length in lengths),
                  "query_only_windows": int((output["lengths"] == 1).sum())}
        meta = {"version": "otto-score-forecast-data-v1", "episode_ids": [row["episode_id"] for row in identities],
                "episode_regimes": [row["regime"] for row in identities], "episode_splits": [stage] * len(identities),
                "counts": counts}
        self.counts[stage + "_rows"] = total
        self.counts[stage + "_windows"] = n
        return output, meta, identities

    def training(self, train_meta):
        np = self.np
        saved = self.read(self.run / "fits.json")
        self.equal(saved["train_counts"], train_meta["counts"], "complete TRAIN exposure")
        fits = saved["fits"]
        self.require([(row["family"], row["seed"]) for row in fits] == [(f, s) for s in SEEDS for f in FAMILIES],
                     "all twelve final fits in fixed order")
        windows = train_meta["counts"]["windows"]
        permutations = {}
        for seed in SEEDS:
            generator, digest = np.random.default_rng(seed), hashlib.sha256()
            for _ in range(80):
                digest.update(generator.permutation(windows).astype(np.int64).tobytes())
            permutations[seed] = digest.hexdigest()
        events = iter(self.rows(self.run / "progress.jsonl"))
        for fit in fits:
            kind, seed = fit["family"], fit["seed"]
            self.equal(next(events), {"event": "fit_start", "family": kind, "seed": seed}, "fit starts before epochs")
            for epoch in (20, 40, 60, 80):
                self.equal(next(events), {"event": "epoch", "family": kind, "seed": seed, "epoch": epoch,
                    "optimizer_steps": epoch * ((windows + 31) // 32), "window_exposures": epoch * windows},
                    "saved fixed epoch exposure")
            self.equal(next(events), {"event": "fit_complete", **fit}, "durable final fit record")
            name = f"{kind}-{seed}.npz"
            self.require(fit["checkpoint_path"] == name and fit["epochs"] == 80
                         and fit["parameter_count"] == PARAMETERS[kind]
                         and fit["steps"] == 80 * ((windows + 31) // 32)
                         and fit["window_exposures"] == 80 * windows and fit["windows_per_epoch"] == windows
                         and fit["permutation_sha256"] == permutations[seed]
                         and len(fit["initial_sha256"]) == 64
                         and all(c in "0123456789abcdef" for c in fit["initial_sha256"]), "fixed fit recipe metadata")
            for field in ("final_train_loss", "wall_seconds"):
                self.require(type(fit[field]) in (int, float) and math.isfinite(fit[field]) and fit[field] >= 0,
                             "finite fit scalar " + field)
            self.equal(fit["checkpoint"], self.worker["files"][name], "final checkpoint descriptor")
            arrays = self.arrays(self.run / name)
            self.require(set(arrays) == set(SHAPES[kind]), "exact checkpoint tensor names")
            for key, shape in SHAPES[kind].items():
                self.require(arrays[key].dtype == np.float32 and arrays[key].shape == shape
                             and bool(np.isfinite(arrays[key]).all()), "finite final checkpoint tensor")
        for seed in SEEDS:
            pair = [fit for fit in fits if fit["seed"] == seed and fit["family"] in ("residual_gru", "direct_gru")]
            self.require(pair[0]["initial_sha256"] == pair[1]["initial_sha256"], "paired saved GRU initial hashes")
        self.equal(next(events), {"event": "all_checkpoints_closed_before_VALID", "fits_completed": 12,
                                 "checkpoints": {fit["checkpoint_path"]: fit["checkpoint"] for fit in fits}},
                   "all final checkpoint publication before VALID barrier")
        self.require(next(events, None) is None, "no extra training lifecycle events")
        self.require(self.worker["optimizer_steps"] == sum(fit["steps"] for fit in fits), "all attempted updates acknowledged")
        self.counts.update(fits=12, optimizer_steps=sum(fit["steps"] for fit in fits), training_events=73)
        return fits

    def metrics(self, predictions, windows, meta, identities):
        args = (windows["targets"], windows["legal"], windows["lengths"], windows["episode_index"],
                meta["episode_ids"], meta["episode_regimes"])
        value = scalar_metrics(predictions, *args, check=self.check)
        value["by_collector"] = {}
        for arm in ARMS:
            episodes = [i for i, row in enumerate(identities) if row["arm"] == arm]
            remap = {old: new for new, old in enumerate(episodes)}
            chosen = [i for i, index in enumerate(windows["episode_index"]) if int(index) in remap]
            value["by_collector"][arm] = scalar_metrics(
                predictions[chosen], windows["targets"][chosen], windows["legal"][chosen],
                windows["lengths"][chosen], [remap[int(windows["episode_index"][i])] for i in chosen],
                [meta["episode_ids"][i] for i in episodes], [meta["episode_regimes"][i] for i in episodes],
                check=self.check)
        return value

    def predictions(self, windows, meta, identities, fits):
        np = self.np
        records, hold = [], None
        all_fits = [("hold", None), *[(fit["family"], fit["seed"]) for fit in fits]]
        for kind, seed in all_fits:
            filename = "prediction-hold.npz" if kind == "hold" else f"prediction-{kind}-{seed}.npz"
            saved = self.arrays(self.run / filename)
            self.require(set(saved) == {"predictions"}, "only saved predictions in forecast file")
            value = saved["predictions"]
            self.require(value.dtype == np.float32 and value.shape == windows["targets"].shape
                         and bool(np.isfinite(value).all()), "all finite float32 final forecasts")
            self.require(value[:, 0].tobytes() == windows["query_scores"].tobytes(), "exact unmodified query anchors")
            padding = value[~windows["valid_mask"]]
            self.require(padding.tobytes() == np.zeros(padding.shape, np.float32).tobytes(), "exact zero forecast padding")
            if kind == "hold":
                expected = np.repeat(windows["query_scores"][:, None, :], 4, axis=1)
                expected[~windows["valid_mask"]] = 0
                self.require(value.tobytes() == expected.tobytes(), "whole hold-Q baseline exact bytes")
            metrics = self.metrics(value, windows, meta, identities)
            if kind == "hold":
                hold = metrics
            else:
                records.append({"family": kind, "seed": seed, "metrics": metrics})
            self.counts["predictions"] += 1
            self.counts["forecast_metric_rows"] += meta["counts"]["nonquery_rows"]
        return records, hold

    def body(self):
        self.authenticate()
        import numpy as np
        self.np = np
        runtime = self.read(self.run / "runtime.json")
        self.require(runtime["threads"] == runtime["interop_threads"] == 1
                     and runtime["deterministic"] is True and runtime["cuda_used"] is runtime["mps_used"] is False,
                     "inherited deterministic CPU training runtime")
        train, train_meta, _ = self.reconstruct("train")
        del train
        fits = self.training(train_meta)
        windows, meta, identities = self.reconstruct("valid")
        saved_windows = self.arrays(self.run / "validation-windows.npz")
        self.require(set(saved_windows) == set(windows), "exact saved validation window schema")
        for key, value in windows.items():
            actual = saved_windows[key]
            self.require(actual.dtype == value.dtype and actual.shape == value.shape
                         and actual.tobytes() == value.tobytes(), "reconstructed validation bytes " + key)
        del saved_windows
        self.equal(self.read(self.run / "validation-windows.json"), meta, "validation identities and coverage")
        models, hold = self.predictions(windows, meta, identities, fits)
        support = {regime: {str(age): len({row["case"] for row, length in zip(identities, windows["episode_lengths"], strict=True)
                                          if row["regime"] == regime and int(length) > age})
                           for age in (1, 2, 3)} for regime in REGIMES}
        rules = continuation_rules(models, hold, support)
        summary = self.read(self.run / "summary.json")
        expected = {"version": PRODUCER_VERSION, "models": models, "hold": hold, "support": support,
                    "required": rules, "required_passed": sum(row["passes"] for row in rules),
                    "required_conditions": 45, "forecast_continuation": all(row["passes"] for row in rules),
                    "requires_successful_original_supervisor_and_saved_audit": True,
                    "train_counts": train_meta["counts"], "validation_counts": meta["counts"],
                    "scope": "Forced-path forecasts on fresh VALID, not autonomous performance or true action regret."}
        for field in ("setup_seconds", "fitting_seconds", "validation_seconds"):
            number = summary[field]
            self.require(type(number) in (int, float) and math.isfinite(number) and number >= 0,
                         "finite physical forecast stage " + field)
            expected[field] = number
        self.require(math.fsum(fit["wall_seconds"] for fit in fits) <= summary["fitting_seconds"] + 1e-9,
                     "fit intervals contained in fitting phase")
        self.require(math.fsum(summary[key] for key in ("setup_seconds", "fitting_seconds", "validation_seconds"))
                     <= self.worker["wall_seconds"] + 1e-9, "physical training stages bounded by original worker")
        self.equal(summary, expected, "all saved forecast reductions and criteria")
        payloads = {"started.json", "runtime.json", "progress.jsonl", "fits.json", "validation-windows.npz",
                    "validation-windows.json", "prediction-hold.npz", "summary.json"}
        for fit in fits:
            payloads.update((fit["checkpoint_path"], "prediction-" + fit["checkpoint_path"]))
        self.require(set(self.worker["files"]) == payloads, "exact complete forecast payload membership")
        self.counts.update(required_conditions=45, required_passed=expected["required_passed"],
                           collection_episodes=90, prediction_files=13, source_files=len(self.plan["sources"]),
                           training_payloads=len(payloads), collection_payloads=len(self.collection_worker["files"]))
        self.result = {"version": VERSION, "agreement": True, "summary": expected,
                       "fits": fits, "counts": dict(self.counts), "limitations": LIMITATIONS}

    def execute(self):
        self.require(self.out.is_absolute() and self.out.is_relative_to(ROOT) and ".." not in self.out.parts,
                     "exclusive contained audit output")
        self.out.mkdir(parents=False, exist_ok=False)
        old_handler = signal.getsignal(signal.SIGTERM)

        def interrupted(_signum, _frame):
            raise InterruptedError("original score-forecast saved-audit supervisor stopped worker")

        signal.signal(signal.SIGTERM, interrupted)
        try:
            self.admit()
            self.body()
            for name, pin in self.plan["sources"].items():
                self.require(self.sha(self.path(name)) == pin, "source unchanged after saved audit")
            for descriptor in self.inputs.values():
                self.pinned(self.path(descriptor["path"]), descriptor)
            self.require(self.sha(self.args.supervision) == self.receipt["supervision_sha256"], "unchanged original audit launch")
            write(self.out / "audit.json", self.result)
            self.check()
            files = {name: {k: v for k, v in self.descriptor(self.out / name).items() if k != "path"}
                     for name in ("started.json", "audit.json")}
            self.require({p.name for p in self.out.iterdir()} == set(files), "exact saved-audit payload closure")
            finished = self.clock.now_ns()
            self.receipt.update(status="completed", agreement=True, files=files, counts=dict(self.counts),
                                started_ns=self.start, finished_ns=finished, wall_seconds=(finished-self.start)/1e9,
                                requires_successful_original_supervisor=True)
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "agreement": True,
                              "receipt": self.descriptor(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.failure = True
            self.receipt.update(status="failed", agreement=False, error=repr(error), traceback=traceback.format_exc(),
                                counts=dict(self.counts))
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {p.name: {k: v for k, v in self.descriptor(p).items() if k != "path"}
                                         for p in self.out.iterdir() if p.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as publication:  # noqa: BLE001 - preserve original failure
                error.add_note(f"Failure publication: {publication!r}")
            raise
        finally:
            signal.signal(signal.SIGTERM, old_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "worker", "terminal"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    Audit(args).execute()


if __name__ == "__main__":
    main()
