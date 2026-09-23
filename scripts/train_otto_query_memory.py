"""Prospective query-memory fitting and DEV producer. Never decodes TEST arrays.

All original collection/capacity closures, source bytes and runtime are admitted
before numerical imports. Eighteen fixed final fits and all 24 canonical TRAIN
views must be durable before DEV decoding. Producer completion is not an audit
or a gate pass. A separate closed DEV audit owns any conditional TEST admission.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF, TEST = "scripts/train_otto_query_memory.py", "tests/test_train_otto_query_memory.py"
CORE = "src/openjev/research/otto_query_memory_training.py"
CORE_TEST = "tests/test_otto_query_memory_training.py"
MODEL = "src/openjev/research/otto_query_memory_model.py"
DATA = "src/openjev/research/otto_query_memory_data.py"
METRICS = "src/openjev/research/otto_query_memory_metrics.py"
GATE = "src/openjev/research/otto_query_memory_gate.py"
PROTOCOL = "research/otto-query-memory-protocol.md"
COLLECTOR = "scripts/collect_otto_query_memory.py"
COLLECTOR_PIN = "e1cd73b7e2389c208955ce5dd46e7f7238444482d5ae06d7ee73fc640a579c8f"
CLOCK = "src/openjev/research/suspend_clock.py"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SUPERVISOR_PIN = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
VERSION = "otto-query-memory-training-v1"
COLLECTION_VERSION = "otto-query-memory-collection-v1"
CAPACITY_SCRIPT = "scripts/qualify_otto_query_memory_capacity.py"
CAPACITY_LIMITS = {"seconds": 240, "rss_bytes": 4 * 1024**3, "output_bytes": 128 * 1024**2}
CAPACITY_VERSION = "otto-query-memory-capacity-v1"
CAPACITY_FAMILIES = ("ordinary", "instant_delta", "trace_delta", "trace_additive", "trace_scrambled")
SEEDS = (309000001, 309000002, 309000003)
FITS = ("pretrained", "joint_aux", "instant_delta", "trace_delta", "trace_additive", "trace_scrambled")
BRANCHES = FITS[1:]
VIEWS = ("pretrained", "joint_aux", "last_error", "instant_delta", "trace_delta", "trace_additive", "trace_scrambled", "trace_no_write")
EPOCHS = {family: 80 if family == "pretrained" else 40 for family in FITS}
LIMITS = {"seconds": 21600, "rss_bytes": 4 * 1024**3, "output_bytes": 2 * 1024**3}
CONFIG = {"fit_seeds": list(SEEDS), "fits": list(FITS), "views": list(VIEWS), "epochs": EPOCHS,
    "train_episodes": 54, "dev_episodes": 18, "test_decode": False, "query_period": 4,
    "batch_episodes": 6, "chunk": 32, "optimizer": "Adam", "learning_rate": .003,
    "weight_decay": 0., "gradient_clip": 5., "score_scale": 64., "prior_coefficient": 1.,
    "objective": "legal-centered nonquery MSE plus all-four corrected-shadow-prior MSE",
    "projection": {"input_dim": 28, "output_dim": 8, "bias": False, "parameters": 224},
    "memory": {"step_size": .25, "trace_decay": .75, "memory_decay": 1., "last_error_decay": .75, "epsilon": 1e-6},
    "optimizer_updates": 7560, "episode_exposures": 45360, "optimized_fits": 18, "evaluation_views": 24,
    "checkpoint": "final fixed epoch", "branch_orders": "PCG64(fit_seed) restarted in every branch",
    "frozen_evaluation": "independent cloned slow weights; required residual/projection flags True; outer no_grad",
    "shared_pretraining": "one80-epoch fit per seed; no charge duplication in branch costs",
    "test_admission": "separate evaluator after successful original DEV producer and saved-audit closures",
    "device": "cpu", "dtype": "float32"}
AUDIT_COMPONENTS = {"scripts/audit_otto_query_memory.py", "tests/test_audit_otto_query_memory.py",
                    "tests/test_otto_query_memory_audit_numerics.py"}
NEW_COMPONENTS = {SELF, TEST, CORE, CORE_TEST} | AUDIT_COMPONENTS
NEW_SOURCES = NEW_COMPONENTS | {MODEL, DATA, METRICS, GATE, "tests/test_otto_query_memory_gate.py", PROTOCOL, COLLECTOR, CLOCK, SUPERVISOR, CAPACITY_SCRIPT,
    "src/openjev/research/otto_query_memory.py", "src/openjev/research/otto_scheduled_predictor.py",
    "src/openjev/research/otto_cross_query_scores.py", "src/openjev/research/otto_prequery_scores.py",
    "src/openjev/research/otto_protected_readout.py", "src/openjev/research/otto_protected_training_model.py",
    "src/openjev/research/otto_score_forecast_data.py", "src/openjev/research/otto_prequery_loss.py",
    "tests/test_otto_query_memory_model.py", "tests/test_otto_query_memory_data.py",
    "tests/test_otto_query_memory_metrics.py", "src/openjev/__init__.py", "src/openjev/research/__init__.py"}
ROLES = ("collection_plan", "collection_receipt", "collection_terminal", "engineering",
         "capacity_plan", "capacity_receipt", "capacity_terminal")
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")
ARRAY_KEYS = {"features", "raw_q", "legal", "actions", "correction", "episode_offsets"}
COLLECTION_PAYLOADS = {f"{name}.jsonl.gz" for name in ("work", "weights", "forwards", "transitions", "samples")} | {
    "started.json", "runtime.json", "setup.json", "deployment.json", "cohort.json", "episode-boundaries.jsonl",
    "episodes.jsonl", "train.npz", "dev.npz", "test.npz", "costs.json", "summary.json"}
PREDICTION_FIELDS = ("action_prediction", "slow_action_prediction", "base_prediction", "shadow_prior",
                     "corrected_shadow_prior", "prior_mask", "prewrite_correction", "episode_offsets")
PAYLOADS = {"started.json", "runtime.json", "progress.jsonl", "work.jsonl", "fits.json", "forks.json",
            "train-views.json", "dev-views.json", "summary.json", "dev-barrier.json",
            "train-history.npz", "train-history.json", "dev-history.npz", "dev-history.json"}
PAYLOADS |= {f"checkpoint-{family}-{seed}.npz" for seed in SEEDS for family in FITS}
PAYLOADS |= {f"{stage}-prediction-{view}-{seed}.npz" for stage in ("train", "dev") for seed in SEEDS for view in VIEWS}


def require(condition, message):
    if not condition:
        raise ValueError(message)

def regular(value):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    require(p.is_file() and p.is_relative_to(ROOT) and ".." not in p.parts
            and not any(x.is_symlink() for x in (p, *p.parents)), "regular contained evidence")
    return p


def descriptor(value):
    p = regular(value)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": p.stat().st_size}


def read(value):
    return json.loads(regular(value).read_text())


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime_record():
    return {"python": sys.version, "executable": sys.executable,
            "distributions": dict(sorted((x.metadata["Name"], x.version)
                                          for x in importlib.metadata.distributions()))}



def closed_files(path, receipt, expected=None):
    directory = regular(path).parent
    require(expected is None or set(receipt["files"]) == expected, "exact phase payload names")
    require({p.name for p in directory.iterdir()} == set(receipt["files"]) | {"receipt.json"}, "closed phase inventory")
    for name, pin in receipt["files"].items():
        require(Path(name).name == name and descriptor(directory / name) == pin, "closed payload hash")
    return directory


def successful_process(inputs, prefix, receipt, script, interpreter, seconds):
    """Bind the original successful parent, its launch, command and paid bounds."""
    parent = read(inputs[prefix + "_terminal"]["path"])
    require(parent["status"] == "completed" and type(parent["returncode"]) is int and parent["returncode"] == 0
        and parent["timed_out"] is False and parent["group_absent"] is True
        and parent["cleanup"]["reaped"] is True and parent["cleanup"]["group_absent"] is True
        and parent["cleanup"]["errors"] == [] and parent["timing_available"] is True
        and parent["error"] is parent["clock_error"] is None and parent["cap_seconds"] == seconds
        and Path(parent["cwd"]) == ROOT and parent["pid"] == parent["pgid"] != parent["parent_pid"]
        and parent["clock_source_sha256"] == CLOCK_PIN and parent["watchdog_sha256"] == SUPERVISOR_PIN
        and parent["deadline_ns"] == parent["started_ns"] + seconds * 10**9
        and parent["started_ns"] <= receipt["started_ns"] < receipt["finished_ns"]
        <= parent["finished_ns"] <= parent["deadline_ns"], "original successful " + prefix + " parent")
    command = list(parent["command"])
    if command[1:2] == ["-u"]:
        command.pop(1)
    require(command[:3] == [str(ROOT / interpreter), str(ROOT / script), "run"]
            and len(command) == 11 and len(set(command[3::2])) == 4, "original phase command")
    options = dict(zip(command[3::2], command[4::2], strict=True))
    require(set(options) == {"--plan", "--plan-sha256", "--supervision", "--output"}
        and options["--plan"] == inputs[prefix + "_plan"]["path"]
        and options["--plan-sha256"] == inputs[prefix + "_plan"]["sha256"]
        and options["--output"] == str(regular(inputs[prefix + "_receipt"]["path"]).parent),
        "original phase argument joins")
    launch = read(options["--supervision"])
    require(descriptor(options["--supervision"])["sha256"] == receipt["supervision_sha256"]
            and all(parent[k] == v for k, v in launch.items()), "original launch identity")
    started = read(regular(inputs[prefix + "_receipt"]["path"]).parent / "started.json")
    require(started["launch"] == launch and started["started_ns"] == receipt["started_ns"],
            "original worker started/launch join")
    if prefix in ("collection", "capacity") or "request" in started:
        require(started["request"] == {"mode": "run", **{
            name[2:].replace("-", "_"): value for name, value in options.items()}},
            "original worker request joins")
    return parent



def authenticate_engineering(path, sources):
    """Bind all qualified bytes; only named pytest fixture storage is excluded.

    The qualifier may cover more components than this producer, but cannot
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


def authenticate_capacity(inputs, sources):
    plan = read(inputs["capacity_plan"]["path"])
    receipt = read(inputs["capacity_receipt"]["path"])
    require(plan["version"] == receipt["version"] == CAPACITY_VERSION
            and plan["status"] == "frozen_before_synthetic_work"
            and receipt["status"] == "completed" and receipt["complete"] is True
            and receipt["admitted"] is True and receipt["plan_sha256"] == inputs["capacity_plan"]["sha256"]
            and receipt["requires_successful_original_supervisor"] is True
            and receipt["sources"] == plan["sources"] and receipt["inputs"] == plan["inputs"]
            and receipt["limits"] == plan["limits"] == CAPACITY_LIMITS
            and receipt["pending"] is None and receipt.get("pending_emission") is None
            and receipt["completed_families"] == list(CAPACITY_FAMILIES)
            and receipt["completed_family_count"] == receipt["optimizer_updates"] == 5
            and receipt["peak_rss_bytes"] <= CAPACITY_LIMITS["rss_bytes"]
            and all(receipt[name] == 0 for name in ("empirical_array_decodes", "empirical_checkpoint_decodes",
                                                   "teacher_calls", "native_calls")),
            "complete admitted new memory capacity")
    for role in ("collection_plan", "collection_receipt", "collection_terminal"):
        require(plan["inputs"][role] == inputs[role], "capacity uses the same closed collection")
    for record in plan["inputs"].values():
        require(descriptor(record["path"]) == {key: record[key] for key in ("sha256", "bytes")},
                "capacity input remains unchanged")
    for name, pin in plan["sources"].items():
        require(descriptor(name)["sha256"] == pin and (name not in sources or sources[name] == pin),
                "capacity uses unchanged current source")
        sources[name] = pin
    require(CAPACITY_SCRIPT in plan["sources"], "capacity script authenticated before metadata import")
    capacity = load(ROOT / CAPACITY_SCRIPT, "_query_memory_capacity_metadata")
    require(plan["configuration"] == capacity.CONFIGURATION, "fixed qualified capacity configuration")
    successful_process(inputs, "capacity", receipt, CAPACITY_SCRIPT, ".venv/bin/python", 240)
    directory = closed_files(inputs["capacity_receipt"]["path"], receipt,
                             {"started.json", "runtime.json", "synthetic.json", "work.jsonl", "summary.json"})
    runtime = read(directory / "runtime.json")
    require(all(runtime[key] == value for key, value in plan["runtime"].items()),
            "capacity runtime equals original plan")
    summary = read(directory / "summary.json")
    require(summary["version"] == CAPACITY_VERSION and summary["admitted"] is True
            and summary["training_cap_seconds"] == LIMITS["seconds"]
            and all(summary[name] == 0 for name in ("empirical_array_decodes", "empirical_checkpoint_decodes",
                                                   "teacher_calls", "native_calls"))
            and [row["family"] for row in summary["families"]] == list(CAPACITY_FAMILIES)
            and set(summary["planned_chunks"]) == set(CAPACITY_FAMILIES), "all measured capacity families")
    require(summary["planned_chunks"] == plan["planned_chunks"], "capacity summary retains frozen chunk forecast")
    require(all(type(row["batch_seconds"]) in (int, float) and math.isfinite(row["batch_seconds"])
                and row["batch_seconds"] > 0 and row["forward_chunks"] == 8
                and type(summary["planned_chunks"][row["family"]]) is int
                and summary["planned_chunks"][row["family"]] > 0 for row in summary["families"]),
            "positive finite eight-chunk capacity measurements")
    projected = 2 * math.fsum(row["batch_seconds"] / 8 * summary["planned_chunks"][row["family"]]
                              for row in summary["families"]) + 600
    require(summary["projected_seconds"] == projected <= summary["threshold_seconds"] == 16200,
            "prospective capacity heuristic admits fixed training cap")
    require(sum(value["bytes"] for value in receipt["files"].values()) <= CAPACITY_LIMITS["output_bytes"],
            "capacity output bound")
    return summary


def authenticate_inputs(inputs):
    require(set(inputs) == set(ROLES), "exact required training evidence roles")
    for record in inputs.values():
        require(set(record) == {"path", "sha256", "bytes"}
                and descriptor(record["path"]) == {key: record[key] for key in ("sha256", "bytes")},
                "input pin before any numerical decoding")
    require(descriptor(COLLECTOR)["sha256"] == COLLECTOR_PIN, "immutable census collector before metadata import")
    collector = load(ROOT / COLLECTOR, "_query_memory_census_metadata")
    plan, receipt = read(inputs["collection_plan"]["path"]), read(inputs["collection_receipt"]["path"])
    require(plan["version"] == receipt["version"] == COLLECTION_VERSION
            and plan["status"] == "frozen_before_collection" and plan["cohort"] == collector.cohort()
            and plan["configuration"] == collector.CONFIGURATION and plan["call_caps"] == collector.CALL_CAPS
            and set(plan["payloads"]) == COLLECTION_PAYLOADS
            and receipt["status"] == "completed" and receipt["complete"] is True
            and receipt["plan_sha256"] == inputs["collection_plan"]["sha256"]
            and receipt["sources"] == plan["sources"] and receipt["inputs"] == plan["inputs"]
            and receipt["native_inputs"] == plan["native_inputs"]
            and receipt["requires_successful_original_supervisor"] is True
            and receipt["limits"] == plan["limits"] == collector.LIMITS,
            "complete current census collection")
    require(receipt["completed_episodes"] == 108 and receipt["train_episodes"] == 54
            and receipt["dev_episodes"] == 18 and receipt["test_episodes"] == 36
            and receipt["training_updates"] == 0 and not receipt.get("pending")
            and receipt.get("pending_episode") is receipt.get("pending_action") is receipt.get("pending_emission") is None
            and not receipt.get("cleanup_errors") and receipt["peak_rss_bytes"] <= 4 * 1024**3,
            "complete108 paths without pending collection work")
    successful_process(inputs, "collection", receipt, COLLECTOR, ".venv-otto-released-native/bin/python", 7200)
    directory = closed_files(inputs["collection_receipt"]["path"], receipt, COLLECTION_PAYLOADS)
    require(sum(value["bytes"] for value in receipt["files"].values()) <= 2 * 1024**3, "collection output cap")
    calls = receipt["calls"]
    require(set(calls) == set(collector.CALL_CAPS) and all(
        type(row["attempted"]) is int and type(row["returned"]) is int
        and row["attempted"] == row["returned"] and 0 <= row["attempted"] <= collector.CALL_CAPS[name]
        for name, row in calls.items()), "complete bounded collection operations")
    runtime = read(directory / "runtime.json")
    require(runtime["executable"] == plan["runtime"]["python_executable"]
            == str(ROOT / ".venv-otto-released-native/bin/python")
            and runtime["python"].split()[0] == plan["runtime"]["python_version"]
            and runtime["all_distributions"] == plan["runtime"]["all_distributions"],
            "saved native runtime joins without native imports")
    for section in ("inputs", "native_inputs"):
        for record in plan[section].values():
            require(descriptor(record["path"]) == {key: record[key] for key in ("sha256", "bytes")},
                    "inherited collection input pin")
    sources = dict(plan["sources"])
    for name, pin in sources.items():
        require(descriptor(name)["sha256"] == pin, "immutable inherited collection source")
    for name in NEW_SOURCES:
        pin = descriptor(name)["sha256"]
        require(name not in sources or sources[name] == pin, "new source cannot replace collection source")
        sources[name] = pin
    authenticate_engineering(inputs["engineering"]["path"], sources)
    capacity = authenticate_capacity(inputs, sources)
    return plan, receipt, directory, sources, capacity


def freeze(args):
    inputs = {}
    for role in ROLES:
        path = regular(getattr(args, role))
        pin = descriptor(path)
        require(pin["sha256"] == getattr(args, role + "_sha256"), "external planning input pin")
        inputs[role] = {"path": str(path), **pin}
    _, _, _, sources, capacity = authenticate_inputs(inputs)
    write(args.output, {"version": VERSION, "status": "frozen_before_fitting", "configuration": CONFIG,
                       "limits": LIMITS, "sources": sources, "inputs": inputs, "runtime": runtime_record(),
                       "capacity_projected_seconds": capacity["projected_seconds"], "payloads": sorted(PAYLOADS)})
    print(json.dumps({"status": "frozen_before_fitting", "plan": descriptor(args.output)}), flush=True)


def state_witness(model, *, slow_only=False):
    states = model.slow.state_dict() if slow_only else model.state_dict()
    digest, tensors = hashlib.sha256(), {}
    for name in sorted(states):
        value = states[name]
        raw = value.detach().cpu().numpy().tobytes()
        tensors[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "shape": list(value.shape),
                         "dtype": str(value.dtype), "bytes": len(raw)}
        digest.update(name.encode() + b"\0" + raw)
    return {"sha256": digest.hexdigest(), "tensors": tensors}


def view_parent(view):
    require(view in VIEWS, "declared canonical view")
    return "pretrained" if view == "last_error" else "trace_delta" if view == "trace_no_write" else view


def mode_for(family):
    require(family in set(FITS) | {"last_error", "trace_no_write"}, "declared model family")
    return "none" if family in ("pretrained", "joint_aux") else "trace_delta" if family == "trace_no_write" else family


def construct_fit(models, family, seed, slow_state=None):
    require(family in FITS and seed in SEEDS, "declared optimized fit")
    config = models.memory.Config(mode=mode_for(family), key_dim=8)
    if family == "pretrained":
        require(slow_state is None, "fresh pretraining has no inherited model")
        return models.make_model(config, seed, 4, slow_mode="joint")
    require(slow_state is not None, "branch requires a complete final same-seed slow state")
    mode = "joint" if family == "joint_aux" else "frozen"
    slow = models.predictor.from_state(mode, seed, 4, slow_state)
    return models.QueryMemoryModel(slow, config, seed)


def canonical_clone(models, trained, view, seed):
    config = models.memory.Config(mode=mode_for(view), key_dim=8)
    projection = None if trained.projection is None else trained.projection.weight.detach().clone()
    slow_state = {name: value.detach().clone() for name, value in trained.slow.state_dict().items()}
    clone = models.from_states(config, seed, 4, slow_state, projection, slow_mode="frozen")
    clone.eval()
    require(state_witness(clone, slow_only=True) == state_witness(trained, slow_only=True), "exact canonical slow copy")
    if projection is not None:
        require(clone.projection.weight.detach().numpy().tobytes() == projection.numpy().tobytes(),
                "exact canonical learned projection copy")
    require(clone.slow.mode == "frozen" and all(parameter.requires_grad == name.startswith("action_residual.")
                for name, parameter in clone.slow.named_parameters())
            and (clone.projection is None or clone.projection.weight.requires_grad),
            "canonical frozen backbone and enabled residual/projection flags")
    return clone


def add_counts(destination, values):
    for name, value in values.items():
        require(type(value) is int and value >= 0, "nonnegative integer work count")
        destination[name] = destination.get(name, 0) + value


def same_frozen_predictions(saved, reference, *, no_write=False):
    fields = ("base_prediction", "slow_action_prediction", "shadow_prior", "prior_mask", "episode_offsets")
    if no_write:
        fields += ("action_prediction", "corrected_shadow_prior", "prewrite_correction")
    for name in fields:
        left, right = saved[name], reference[name]
        require(left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes(),
                "canonical frozen prediction equality: " + name)
    return True


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = self.launch = self.plan = None
        self.dev_allowed = False
        self.sequence = 0
        self.receipt = {"version": VERSION, "status": "started", "fits_completed": 0, "optimizer_steps": 0,
                        "episode_exposures": 0, "pending": None, "pending_emission": None,
                        "teacher_calls": 0, "native_calls": 0, "test_array_decodes": 0,
                        "technical_complete": False, "requires_independent_saved_audit": True}

    def check(self):
        require(self.clock.now_ns() < self.launch["deadline_ns"], "original21600-second training deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"] and sum(path.stat().st_size for path in self.out.iterdir() if path.is_file())
                < LIMITS["output_bytes"] - 1024**2, "bounded RSS/output with failure reserve")

    def event(self, value, filename="progress.jsonl"):
        require(filename in ("progress.jsonl", "work.jsonl"), "declared phase journal")
        self.check()
        self.receipt["pending_emission"] = {"file": filename, "record": value}
        with (self.out / filename).open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.receipt["pending_emission"] = None

    def publish(self, name, value):
        require(name in PAYLOADS, "declared JSON payload")
        self.check()
        self.receipt["pending_emission"] = {"file": name}
        write(self.out / name, value)
        self.receipt["pending_emission"] = None

    def npz(self, name, arrays):
        require(name in PAYLOADS, "declared numerical payload")
        self.check()
        self.receipt["pending_emission"] = {"file": name}
        with (self.out / name).open("xb") as stream:
            self.np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        with self.np.load(self.out / name, allow_pickle=False) as saved:
            require(set(saved.files) == set(arrays) and all(saved[key].dtype == arrays[key].dtype
                    and saved[key].shape == arrays[key].shape and saved[key].tobytes() == arrays[key].tobytes()
                    for key in arrays), "saved numerical payload byte roundtrip")
        self.receipt["pending_emission"] = None
        return descriptor(self.out / name)

    def admit(self):
        require(descriptor(CLOCK)["sha256"] == CLOCK_PIN, "immutable clock before import")
        self.clock = load(ROOT / CLOCK, "_query_memory_training_clock").SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, "original launch available")
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == [sys.executable, *sys.argv]
                and self.launch["pid"] == os.getpid() and self.launch["pgid"] == os.getpgrp()
                and self.launch["parent_pid"] == os.getppid() != os.getpid()
                and Path(self.launch["cwd"]) == Path.cwd() == ROOT
                and self.launch["cap_seconds"] == LIMITS["seconds"]
                and self.launch["watchdog_sha256"] == SUPERVISOR_PIN and self.launch["clock_source_sha256"] == CLOCK_PIN
                and self.launch["clock_backend"] == self.clock.backend
                and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
                and self.launch["deadline_ns"] == self.launch["started_ns"] + LIMITS["seconds"] * 10**9,
                "original detached bounded training supervisor identity")
        self.check()
        require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256, "external training plan pin")
        self.plan = read(self.args.plan)
        require(self.plan["version"] == VERSION and self.plan["status"] == "frozen_before_fitting"
                and self.plan["configuration"] == CONFIG and self.plan["limits"] == LIMITS
                and self.plan["runtime"] == runtime_record() and set(self.plan["payloads"]) == PAYLOADS,
                "frozen configuration, payloads and current runtime")
        self.collection_plan, self.collection_receipt, self.collection_dir, sources, self.capacity = authenticate_inputs(self.plan["inputs"])
        require(sources == self.plan["sources"] and self.capacity["projected_seconds"] == self.plan["capacity_projected_seconds"],
                "unchanged complete source and capacity closure")
        require(all(os.environ.get(name) == "1" for name in THREADS), "one numerical CPU thread")
        self.receipt.update(plan_sha256=self.args.plan_sha256, sources=sources, inputs=self.plan["inputs"], limits=LIMITS,
                            supervision_sha256=descriptor(self.args.supervision)["sha256"])
        self.publish("started.json", {"launch": self.launch, "started_ns": self.start,
            "request": {key: str(value) for key, value in vars(self.args).items()}})

    def decode_split(self, stage):
        # This check precedes even constructing a path or asking NumPy to open it.
        require(stage in ("train", "dev"), "this producer never decodes TEST")
        if stage == "dev":
            require(self.dev_allowed and self.receipt["fits_completed"] == 18
                    and self.receipt["optimizer_steps"] == 7560, "closed checkpoint/TRAIN barrier before DEV")
            self.event({"event": "dev_decode_after_barrier", "work_sequence": self.sequence,
                        "barrier": descriptor(self.out / "dev-barrier.json")})
        self.check()
        path = self.collection_dir / (stage + ".npz")
        require(descriptor(path) == self.collection_receipt["files"][path.name], "split bytes before decode")
        with self.np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == ARRAY_KEYS, "exact six census arrays")
            flat = {name: archive[name] for name in archive.files}
        identities = [row for row in self.collection_plan["cohort"] if row["stage"] == stage]
        require(len(identities) == (54 if stage == "train" else 18), "complete declared split roster")
        history = self.data.project_census(flat, identities, query_period=4, expected_stage=stage)
        self.npz(stage + "-history.npz", {name: value for name, value in history.items() if isinstance(value, self.np.ndarray)})
        self.publish(stage + "-history.json", {name: value for name, value in history.items() if not isinstance(value, self.np.ndarray)})
        return history, identities

    def fit(self, family, seed, history, *, parent=None, parent_fit=None):
        self.check()
        tick = self.clock.now_ns()
        parent_state = None if parent is None else {name: value.detach().clone() for name, value in parent.slow.state_dict().items()}
        model = construct_fit(self.models, family, seed, parent_state)
        initial, initial_slow = state_witness(model), state_witness(model, slow_only=True)
        if parent is not None:
            require(initial_slow == parent_fit["final_slow"], "branch starts at exact final pretrained slow tensors")
        optimizer = self.core.construct_optimizer(model)
        require(len(optimizer.state) == 0, "fresh empty optimizer for every fit")
        initialization_seconds = (self.clock.now_ns() - tick) / 1e9
        order_rng = self.np.random.Generator(self.np.random.PCG64(seed))
        orders, permutation, steps, exposures = [], hashlib.sha256(), 0, 0
        totals, work, units, timings = {}, {}, {}, {}
        fit_tick = self.clock.now_ns()
        for epoch in range(EPOCHS[family]):
            order = order_rng.permutation(54).astype(self.np.int64)
            orders.append(order.tolist())
            permutation.update(order.tobytes())
            for offset in range(0, 54, 6):
                indices = order[offset:offset + 6].tolist()
                self.sequence += 1
                attempt = {"call_id": self.sequence, "family": family, "seed": seed,
                           "epoch": epoch + 1, "batch": offset // 6, "episode_indices": indices}
                self.receipt["pending"] = {**attempt, "stage": "batch_start", "start": None}
                self.event({"event": "attempt", **attempt}, "work.jsonl")
                def stage(name, start, attempt=attempt):
                    self.receipt["pending"] = {**attempt, "stage": name, "start": start}
                result = self.core.batch_update(model, optimizer, history, indices, check=self.check, stage=stage)
                steps += 1
                exposures += len(indices)
                require(result["optimizer_step"] == steps and result["optimizer_updates"] == 1
                        and result["episode_indices"] == indices and result["episode_exposures"] == len(indices),
                        "exact chronological batch update and exposure")
                self.event({"event": "return", **attempt, "result": result}, "work.jsonl")
                self.receipt["pending"] = None
                self.receipt["optimizer_steps"] += 1
                self.receipt["episode_exposures"] += len(indices)
                for key in ("forward_rows", "nonquery_rows", "prior_rows", "forward_chunks", "loss_chunks",
                            "backward_chunks", "differentiable_chunks", "no_gradient_chunks", "skipped_backward_chunks"):
                    totals[key] = totals.get(key, 0) + result[key]
                add_counts(work, result["work_counts"])
                add_counts(units, result["memory_work_units"])
                for name, seconds in result["timing_seconds"].items():
                    require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0, "finite batch timing")
                    timings[name] = timings.get(name, 0.) + seconds
            self.event({"event": "epoch_complete", "family": family, "seed": seed, "epoch": epoch + 1,
                        "optimizer_steps": steps, "episode_exposures": exposures})
        fit_seconds = (self.clock.now_ns() - fit_tick) / 1e9
        final, final_slow = state_witness(model), state_witness(model, slow_only=True)
        frozen = family not in ("pretrained", "joint_aux")
        require(not frozen or final_slow == initial_slow == parent_fit["final_slow"], "all eight frozen slow tensors unchanged")
        require(steps == EPOCHS[family] * 9 and exposures == EPOCHS[family] * 54
                and totals["forward_rows"] == EPOCHS[family] * int(history["episode_offsets"][-1]),
                "complete fixed epochs, updates and row exposure")
        checkpoint_name = f"checkpoint-{family}-{seed}.npz"
        checkpoint_tick = self.clock.now_ns()
        checkpoint = self.npz(checkpoint_name, {name: value.detach().cpu().numpy().copy()
                                              for name, value in model.state_dict().items()})
        row = {"family": family, "seed": seed, "stage": "pretrain" if family == "pretrained" else "branch",
               "mode": mode_for(family), "slow_mode": model.slow.mode, "parameters": model.parameter_metadata(),
               "epochs": EPOCHS[family], "steps": steps, "episode_exposures": exposures,
               "episode_orders": orders, "permutation_sha256": permutation.hexdigest(),
               "initial": initial, "initial_slow": initial_slow, "final": final, "final_slow": final_slow,
               "frozen_slow_unchanged": True if frozen else None,
               "pretrained_checkpoint": None if parent_fit is None else {"path": parent_fit["checkpoint_path"], **parent_fit["checkpoint"]},
               "checkpoint_path": checkpoint_name, "checkpoint": checkpoint,
               "optimizer_initial_state_entries": 0,
               "optimizer_parameter_names": [name for name, _ in model.effective_named_parameters()],
               "optimizer_final_steps": {name: int(optimizer.state[value]["step"].item())
                                         for name, value in model.effective_named_parameters()},
               "totals": totals, "work_counts": work, "memory_work_units": units, "batch_timing_seconds": timings,
               "initialization_seconds": initialization_seconds, "fit_seconds": fit_seconds,
               "checkpoint_seconds": (self.clock.now_ns() - checkpoint_tick) / 1e9}
        require(set(row["optimizer_final_steps"].values()) == {steps}, "all effective Adam counters match final fit")
        self.event({"event": "fit_complete", **row})
        self.receipt["fits_completed"] += 1
        return model, row

    def predict(self, model, view, seed, history, identities, stage):
        np, torch = self.np, self.torch
        total = int(history["episode_offsets"][-1])
        saved = {name: np.zeros((total, 4), np.float32) for name in PREDICTION_FIELDS
                 if name not in ("prior_mask", "episode_offsets")}
        saved["prior_mask"] = np.zeros(total, np.bool_)
        saved["episode_offsets"] = history["episode_offsets"].copy()
        work, units, chunks, rows = {}, {}, 0, 0
        terms = {name: [] for name in ("total", "nonquery", "prior")}
        tick = self.clock.now_ns()
        with torch.no_grad():
            for first in range(0, history["episode_count"], 6):
                indices = list(range(first, min(first + 6, history["episode_count"])))
                lengths = [int(history["episode_offsets"][i + 1] - history["episode_offsets"][i]) for i in indices]
                carry = model.initial_carry(len(indices))
                for start in range(0, max(lengths), 32):
                    self.check()
                    packet = self.data.batch_chunk(history, indices, start)
                    inputs = {name: torch.from_numpy(value) for name, value in packet["model_inputs"].items()}
                    before = {"phase": stage, "view": view, "seed": seed, "episode_indices": indices, "start": start}
                    self.sequence += 1
                    call_id = self.sequence
                    self.receipt["pending"] = {"call_id": call_id, **before}
                    self.event({"event": "attempt", "call_id": call_id, **before}, "work.jsonl")
                    forecast = model(**inputs, carry=carry, no_write=view == "trace_no_write")
                    require(torch.equal(forecast.prior_mask, torch.from_numpy(packet["prior_mask"])),
                            "canonical complete prior support")
                    loss = self.data.weighted_loss(forecast.action_prediction, forecast.corrected_shadow_prior,
                        torch.from_numpy(packet["targets"]), torch.from_numpy(packet["legal"]),
                        torch.from_numpy(packet["nonquery_weights"]), torch.from_numpy(packet["prior_weights"]),
                        inputs["query_mask"], forecast.prior_mask, episode_count=history["episode_count"])
                    for name, values in terms.items():
                        values.append(float(loss[name]) * len(indices) / history["episode_count"])
                    fields = {name: getattr(forecast, "prediction" if name == "base_prediction" else name)
                              for name in PREDICTION_FIELDS if name != "episode_offsets"}
                    for lane, index in enumerate(indices):
                        length = int(inputs["lengths"][lane])
                        low = int(history["episode_offsets"][index]) + start
                        for name, value in fields.items():
                            tail = value[lane, length:].detach().numpy()
                            require(tail.tobytes() == np.zeros_like(tail).tobytes(), "canonical positive-zero padding")
                            saved[name][low:low + length] = value[lane, :length].detach().numpy()
                    carry = self.models.detach_carry(forecast.carry)
                    add_counts(work, forecast.work_counts)
                    add_counts(units, forecast.memory_work_units)
                    chunk_rows = int(inputs["lengths"].sum())
                    rows += chunk_rows
                    chunks += 1
                    self.event({"event": "return", "call_id": call_id, **before, "rows": chunk_rows,
                                "work_counts": forecast.work_counts, "memory_work_units": forecast.memory_work_units}, "work.jsonl")
                    self.receipt["pending"] = None
                require(carry.slow.base.ended.tolist() == [True] * len(indices)
                        and carry.slow.base.absolute_step.tolist() == lengths
                        and carry.fast.absolute_step.tolist() == lengths, "all canonical episode tails consumed")
        require(rows == total and all(bool(np.isfinite(value).all()) for value in saved.values()), "finite complete canonical predictions")
        for name in ("action_prediction", "slow_action_prediction", "base_prediction"):
            require(saved[name][history["query_mask"]].tobytes() == history["query_scores"][history["query_mask"]].tobytes(),
                    "exact observed query prediction bytes")
        require(np.array_equal(saved["prior_mask"], history["prior_mask"]), "saved complete prior support")
        inference_seconds = (self.clock.now_ns() - tick) / 1e9
        metric_tick = self.clock.now_ns()
        episodes = []
        for index, identity in enumerate(identities):
            low, high = map(int, history["episode_offsets"][index:index + 2])
            episodes.append(self.metrics.episode_metrics(identity, 4, history["targets"][low:high],
                history["legal"][low:high], saved["action_prediction"][low:high],
                prequery_forecast=saved["corrected_shadow_prior"][low:high]))
        report = self.metrics.aggregate_episodes(episodes, family=view, fit_seed=seed, expected_identities=identities)
        return saved, {"view": view, "seed": seed, "stage": stage, "parent_fit": view_parent(view),
                       "work_counts": work, "memory_work_units": units, "chunks": chunks, "rows": rows,
                       "loss": {name: math.fsum(values) for name, values in terms.items()}, "metrics": report,
                       "inference_and_loss_seconds": inference_seconds,
                       "metric_seconds": (self.clock.now_ns() - metric_tick) / 1e9,
                       "evaluation": {"slow_mode": "frozen", "grad_enabled": False,
                                      "residual_requires_grad": True, "projection_requires_grad": model.projection is not None}}

    def evaluate_views(self, fitted, history, identities, stage):
        records = []
        for seed in SEEDS:
            reference = None
            for view in VIEWS:
                trained = fitted[seed, view_parent(view)]
                clone_tick = self.clock.now_ns()
                model = canonical_clone(self.models, trained, view, seed)
                clone_seconds = (self.clock.now_ns() - clone_tick) / 1e9
                before = state_witness(model)
                saved, record = self.predict(model, view, seed, history, identities, stage)
                require(state_witness(model) == before, "canonical evaluation cannot modify trained parameters")
                if view == "pretrained":
                    reference = saved
                elif view != "joint_aux":
                    record["frozen_predictions_equal_pretrained"] = same_frozen_predictions(saved, reference, no_write=view == "trace_no_write")
                name = f"{stage}-prediction-{view}-{seed}.npz"
                serialization = self.clock.now_ns()
                pin = self.npz(name, saved)
                records.append({**record, "clone_seconds": clone_seconds, "prediction_path": name, "prediction": pin,
                                "serialization_seconds": (self.clock.now_ns() - serialization) / 1e9})
        self.metrics.validate_matched_reports([row["metrics"] for row in records], families=VIEWS,
            fit_seeds=SEEDS, query_periods=(4,), expected_identities=identities)
        self.publish(stage + "-views.json", {"stage": stage, "views": records, "technical_complete": False})
        return records

    def close_training(self, fits, train_views):
        require([(row["family"], row["seed"]) for row in fits] == [(family, seed) for seed in SEEDS for family in FITS]
                and [(row["view"], row["seed"]) for row in train_views] == [(view, seed) for seed in SEEDS for view in VIEWS]
                and self.receipt["fits_completed"] == 18 and self.receipt["optimizer_steps"] == 7560
                and self.receipt["episode_exposures"] == 45360, "all fits and TRAIN views before DEV")
        pins = {row["checkpoint_path"]: row["checkpoint"] for row in fits}
        pins.update({row["prediction_path"]: row["prediction"] for row in train_views})
        expected = {f"checkpoint-{family}-{seed}.npz" for seed in SEEDS for family in FITS}
        expected |= {f"train-prediction-{view}-{seed}.npz" for seed in SEEDS for view in VIEWS}
        require(set(pins) == expected, "distinct complete checkpoint/TRAIN prediction paths")
        for name in ("fits.json", "forks.json", "train-views.json", "train-history.json", "train-history.npz"):
            pins[name] = descriptor(self.out / name)
        for name, pin in pins.items():
            require(descriptor(self.out / name) == pin, "complete durable DEV barrier payload")
        barrier = {"event": "all18_checkpoints_and24_TRAIN_views_closed_before_DEV", "fits": 18, "train_views": 24,
                   "optimizer_steps": 7560, "episode_exposures": 45360, "work_sequence": self.sequence, "files": pins}
        self.publish("dev-barrier.json", barrier)
        self.event({**barrier, "barrier": descriptor(self.out / "dev-barrier.json")})
        self.dev_allowed = True

    def body(self):
        setup_tick = self.clock.now_ns()
        import numpy as np
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        sys.path.insert(0, str(ROOT / "src"))
        from openjev.research import otto_query_memory_data as data
        from openjev.research import otto_query_memory_gate as gate
        from openjev.research import otto_query_memory_metrics as metrics
        from openjev.research import otto_query_memory_model as models
        from openjev.research import otto_query_memory_training as core

        self.np, self.torch, self.models, self.data, self.metrics, self.core = np, torch, models, data, metrics, core
        self.gate = gate
        require(torch.get_num_threads() == torch.get_num_interop_threads() == 1, "fixed numerical thread counts")
        self.publish("runtime.json", {**runtime_record(), "numpy": np.__version__, "torch": torch.__version__,
            "torch_threads": 1, "interop_threads": 1, "deterministic": True, "cuda_used": False, "mps_used": False})
        train, train_ids = self.decode_split("train")
        setup_seconds = (self.clock.now_ns() - setup_tick) / 1e9
        fits, fitted, forks = [], {}, []
        for seed in SEEDS:
            parent, parent_fit = self.fit("pretrained", seed, train)
            fitted[seed, "pretrained"] = parent
            fits.append(parent_fit)
            projection_witness = None
            branch_orders = None
            for family in BRANCHES:
                model, fit = self.fit(family, seed, train, parent=parent, parent_fit=parent_fit)
                if family != "joint_aux":
                    projection = fit["initial"]["tensors"]["projection.weight"]
                    require(projection_witness is None or projection == projection_witness, "identical seeded memory-branch projection")
                    projection_witness = projection
                require(branch_orders is None or fit["episode_orders"] == branch_orders, "identical stage-two episode orders")
                branch_orders = fit["episode_orders"]
                fits.append(fit)
                fitted[seed, family] = model
            require(state_witness(parent) == parent_fit["final"], "branches cannot mutate pretrained parent")
            forks.append({"seed": seed, "pretrained_checkpoint": parent_fit["checkpoint"],
                          "pretrained_checkpoint_path": parent_fit["checkpoint_path"],
                          "memory_initial_projection": projection_witness,
                          "branches": [{"family": row["family"], "initial": row["initial"], "initial_slow": row["initial_slow"],
                                        "checkpoint_path": row["checkpoint_path"], "checkpoint": row["checkpoint"]}
                                       for row in fits if row["seed"] == seed and row["family"] != "pretrained"]})
        self.publish("fits.json", {"fits": fits})
        self.publish("forks.json", {"forks": forks})
        train_views = self.evaluate_views(fitted, train, train_ids, "train")
        self.close_training(fits, train_views)
        dev, dev_ids = self.decode_split("dev")
        dev_views = self.evaluate_views(fitted, dev, dev_ids, "dev")
        prospective_gate = self.gate.continuation_gate([row["metrics"] for row in dev_views], dev_ids, technical_complete=False)
        self.publish("summary.json", {"version": VERSION, "configuration": CONFIG, "fits": fits,
            "train_views": train_views, "dev_views": dev_views, "technical_complete": False,
            "gate": prospective_gate, "gate_status": "pending independent saved-output audit and original producer closure",
            "test_array_decodes": 0, "test_evaluation_admitted": False, "setup_seconds": setup_seconds,
            "capacity_projected_seconds": self.capacity["projected_seconds"],
            "capacity_scope": "prospective synthetic heuristic only; fixed21600-second cap remains binding",
            "cost_scope": "shared pretraining once per seed; branch fitting, all24 TRAIN views and24 DEV views paid separately",
            "scope": "fixed-path teacher-score imitation only; no autonomous return or compute-saving claim"})

    def execute(self):
        self.out.mkdir(parents=False, exist_ok=False)
        def interrupt(_signum, _frame):
            raise InterruptedError("original query-memory training supervisor stopped worker")
        signal.signal(signal.SIGTERM, interrupt)
        try:
            self.admit()
            self.body()
            for name, pin in self.plan["sources"].items():
                self.check()
                require(descriptor(name)["sha256"] == pin, "unchanged final source")
            for record in self.plan["inputs"].values():
                require(descriptor(record["path"]) == {key: record[key] for key in ("sha256", "bytes")}, "unchanged final input")
            for name, pin in self.collection_receipt["files"].items():
                self.check()
                require(descriptor(self.collection_dir / name) == pin, "unchanged collection payload bytes")
            require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256
                    and descriptor(self.args.supervision)["sha256"] == self.receipt["supervision_sha256"], "unchanged plan and original launch")
            require(self.receipt["fits_completed"] == 18 and self.receipt["optimizer_steps"] == 7560
                    and self.receipt["episode_exposures"] == 45360 and self.receipt["test_array_decodes"] == 0
                    and self.receipt["pending"] is self.receipt["pending_emission"] is None, "complete TRAIN/DEV producer")
            require({path.name for path in self.out.iterdir()} == PAYLOADS, "exact80 training payloads")
            files = {name: descriptor(self.out / name) for name in sorted(PAYLOADS)}
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status="completed", complete=True, files=files, started_ns=self.start, finished_ns=finished,
                                wall_seconds=(finished - self.start) / 1e9, requires_successful_original_supervisor=True)
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "receipt": descriptor(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.receipt.update(status="failed", complete=False, error=repr(error), traceback=traceback.format_exc())
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {path.name: descriptor(path) for path in self.out.iterdir() if path.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - retain original failure
                error.add_note("Failure receipt publication also failed: " + repr(secondary))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    plan = sub.add_parser("plan")
    for role in ROLES:
        plan.add_argument("--" + role.replace("_", "-"), type=Path, required=True)
        plan.add_argument("--" + role.replace("_", "-") + "-sha256", required=True)
    plan.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    for name in ("plan", "supervision", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path(sys.executable).absolute() == ROOT / ".venv/bin/python" and Path.cwd() == ROOT,
            "original general CPU interpreter and checkout")
    require(args.output.is_absolute() and args.output.is_relative_to(ROOT) and ".." not in args.output.parts
            and not any(p.is_symlink() for p in args.output.parents), "contained exclusive absolute output")
    if args.mode == "plan":
        freeze(args)
    else:
        require(args.plan.is_absolute() and args.supervision.is_absolute(), "absolute execution inputs")
        Run(args).execute()


if __name__ == "__main__":
    main()
