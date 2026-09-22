"""Fixed six-fit query gates from a completed separately supervised collection.

Plan mode reads metadata/bytes only. Run authenticates before numerical imports;
no planner, native environment, new label or held-out input is invoked here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
import os
import platform
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-query-gate-training-v1"
SELF = "scripts/train_otto_query_gate.py"
MODEL = "src/openjev/research/otto_query_gate_models.py"
TEST = "tests/test_train_otto_query_gate.py"
MODEL_TEST = "tests/test_otto_query_gate_models.py"
PROTOCOL = "research/otto-query-gate-training-protocol.md"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
SUPERVISOR_PIN = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
SEEDS, KINDS = (40101, 40102, 40103), ("gru32", "mlp190")
FITS = tuple(f"{kind}@{seed}" for seed in SEEDS for kind in KINDS)
THREADS = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")
LIMITS = {"seconds": 300, "rss_bytes": 4 * 1024**3, "output_bytes": 2 * 1024**3}
NEW = {SELF, MODEL, TEST, MODEL_TEST, PROTOCOL, CLOCK, SUPERVISOR}
ROLES = {"collection_plan", "collection_receipt", "collection_terminal", "engineering", "model_engineering", "seed_review"}
COLLECTION_PAYLOADS = {"started.json", "runtime.json", "setup.json", "work.jsonl", "weights.jsonl", "forwards.jsonl",
                       "gate-operations.jsonl", "public-transitions.jsonl", "episodes.jsonl", "training-data.npz",
                       "collection-costs.json", "summary.json"}
ROOT_PAYLOADS = {"started.json", "runtime.json", "data.json", "work.jsonl", "orders.jsonl", "epochs.jsonl",
                 "parity.jsonl", "fits.jsonl", "summary.json"}
PAYLOADS = ROOT_PAYLOADS | {f"{k}-{s}{suffix}" for s in SEEDS for k in KINDS
                          for suffix in ("-initial.npz", ".npz", "-parity.npz")}
CONFIG = {"fit_seeds": list(SEEDS), "kinds": list(KINDS), "episodes": 60, "features": 31,
          "epochs": 80, "batch_episodes": 8, "window": 32, "loss_divider": 256,
          "learning_rate": .0003, "clip": 5., "shuffle_seed_offset": 20000, "threshold": .05,
          "parameters": {"gru32": 6273, "mlp190": 6271}, "parity_absolute_tolerance": 2e-5,
          "parity": "All TRAIN sequences, independently evolved NumPy/Torch states, exact query decisions.",
          "no_early_stopping": True, "native_calls": 0, "new_label_calls": 0}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


def path(value):
    p = Path(value)
    require(not p.is_absolute() and ".." not in p.parts, "relative manifest path")
    return ROOT / p


def digest(p, check=lambda: None):
    require(p.is_file() and not any(v.is_symlink() for v in (p, *p.parents)), "regular nonsymlink evidence")
    h, size = hashlib.sha256(), 0
    with p.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            check()
            h.update(chunk)
            size += len(chunk)
    return {"sha256": h.hexdigest(), "bytes": size}


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    with p.open("xb") as stream:
        stream.write(encode(value))
        stream.flush()
        os.fsync(stream.fileno())


def runtime():
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"), "torch": importlib.metadata.version("torch"),
            "environment": {k: os.environ.get(k) for k in THREADS}}


def metadata_closure(roles, check=lambda: None):
    require(set(roles) == ROLES, "exact training metadata roles")
    files = {}

    def bind(p, expected=None):
        require(p.is_absolute() and p.is_relative_to(ROOT), "contained absolute evidence")
        actual = digest(p, check)
        require(expected is None or actual == expected, "metadata/payload identity")
        name = str(p.relative_to(ROOT))
        require(name not in files or files[name] == actual, "consistent evidence identity")
        files[name] = actual
        return actual

    for record in roles.values():
        require(set(record) == {"path", "sha256", "bytes"}, "external role descriptor")
        bind(Path(record["path"]), {k: record[k] for k in ("sha256", "bytes")})
    prior = read(Path(roles["collection_plan"]["path"]))
    worker_path = Path(roles["collection_receipt"]["path"])
    worker, terminal = read(worker_path), read(Path(roles["collection_terminal"]["path"]))
    require(prior["version"] == worker["version"] == "otto-query-gate-collection-v1"
            and worker["status"] == "completed" and worker["complete"] is True
            and worker["completed_episodes"] == 60 and worker["pending"] == []
            and worker["plan_sha256"] == roles["collection_plan"]["sha256"]
            and worker["requires_successful_original_supervisor"] is True
            and set(worker["files"]) == COLLECTION_PAYLOADS, "completed full collection before fitting")
    require({p.name for p in worker_path.parent.iterdir()} == COLLECTION_PAYLOADS | {"receipt.json"}, "closed collection directory")
    for name, desc in worker["files"].items():
        bind(worker_path.parent / name, desc)
    started = read(worker_path.parent / "started.json")
    launch_path = Path(started["request"]["supervision"])
    require(bind(launch_path)["sha256"] == worker["supervision_sha256"], "collection launch pin")
    launch = read(launch_path)
    require(launch == started["launch"] and all(terminal[k] == v for k, v in launch.items())
            and terminal["status"] == "completed" and terminal["returncode"] == 0 and terminal["timed_out"] is False
            and terminal["error"] is terminal["clock_error"] is None
            and terminal["group_absent"] is terminal["cleanup"]["group_absent"] is terminal["cleanup"]["reaped"] is True
            and terminal["cleanup"]["errors"] == []
            and launch["started_ns"] <= worker["started_ns"] <= worker["finished_ns"] <= terminal["finished_ns"] < launch["deadline_ns"],
            "successful original collection process")
    sources = dict(prior["sources"])
    require(worker["sources"] == sources and worker["inputs"] == prior["inputs"]
            and worker["native_inputs"] == prior["native_inputs"], "collection source/input joins")
    for records in (prior["inputs"], prior["native_inputs"]):
        for record in records.values():
            p = Path(record["path"])
            bind(p if p.is_absolute() else path(record["path"]), {k: record[k] for k in ("sha256", "bytes")})
    for name, pin in sources.items():
        require(digest(path(name), check)["sha256"] == pin, "frozen collection source")
    for role, needed in (("model_engineering", {MODEL, MODEL_TEST}), ("engineering", {SELF, TEST})):
        p = Path(roles[role]["path"])
        e = read(p)
        require(e["status"] == "passed" and e["source_before"] == e["source_after"]
                and needed <= e["source_after"].keys() and e["commands"]
                and all(type(c["returncode"]) is int and c["returncode"] == 0 for c in e["commands"]), "passed source-bound engineering")
        for name, pin in e["source_after"].items():
            require(digest(path(name), check)["sha256"] == pin, "current qualified source bytes")
        require({q.name for q in p.parent.iterdir()} == set(e["files"]) | {p.name}, "closed engineering attempt")
        for name, desc in e["files"].items():
            require(Path(name).name == name, "flat engineering payload")
            bind(p.parent / name, desc)
    seed = read(Path(roles["seed_review"]["path"]))
    require(seed["status"] == "reserved_before_collection" and seed["fit_seeds"] == list(SEEDS)
            and seed["hits"] == [], "fixed fit seed reservation")
    for name in NEW:
        pin = digest(path(name), check)["sha256"]
        require(name not in sources or sources[name] == pin, "unchanged inherited source")
        sources[name] = pin
    require(sources[CLOCK] == CLOCK_PIN and sources[SUPERVISOR] == SUPERVISOR_PIN, "qualified supervision")
    return sources, files, {"worker_seconds": worker["wall_seconds"], "parent_seconds": terminal["wall_seconds"],
                            "rows": worker["rows"], "episodes": 60, "shared_once": True,
                            "scope": "One physical shared collection; per-fit allocations are amortization, not new work."}


def freeze(args):
    roles = {}
    for role in sorted(ROLES):
        p = getattr(args, role)
        actual = digest(p)
        require(actual["sha256"] == getattr(args, role + "_sha256"), "external planning pin")
        roles[role] = {"path": str(p), **actual}
    sources, inputs, collection = metadata_closure(roles)
    require(runtime()["environment"] == THREADS, "one numerical thread before plan")
    write(args.output, {"version": VERSION, "status": "frozen_before_training", "configuration": CONFIG,
          "limits": LIMITS, "sources": sources, "inputs": inputs, "roles": roles, "runtime": runtime(),
          "payloads": sorted(PAYLOADS), "collection": collection})
    print(json.dumps({"status": "frozen_before_training", "plan": digest(args.output)}), flush=True)


def validate_data(data, np, *, episodes=60):
    keys = {"features", "labels", "neural_costs", "analytic_action", "masks", "episode_offsets", "scheduled_query", "neural_gap"}
    require(set(data) == keys, "exact collected array schema")
    n = len(data["features"])
    require(episodes <= n <= episodes * 2188, "complete nonempty bounded episode rows")
    shapes = {"features": (np.float32, (n, 31)), "labels": (np.bool_, (n,)), "neural_costs": (np.float32, (n, 4)),
              "analytic_action": (np.int64, (n,)), "masks": (np.bool_, (n, 4)), "episode_offsets": (np.int64, (episodes + 1,)),
              "scheduled_query": (np.bool_, (n,)), "neural_gap": (np.float32, (n,))}
    for key, (dtype, shape) in shapes.items():
        a = data[key]
        require(isinstance(a, np.ndarray) and a.dtype == dtype and a.shape == shape and np.isfinite(a).all(), f"finite exact {key}")
    offsets = data["episode_offsets"]
    lengths = np.diff(offsets)
    require(offsets[0] == 0 and offsets[-1] == n and (lengths >= 1).all() and (lengths <= 2188).all(), "complete episode boundaries")
    mask, action, costs = data["masks"], data["analytic_action"], data["neural_costs"]
    require(mask.any(axis=1).all() and ((0 <= action) & (action <= 3)).all()
            and mask[np.arange(n), action].all() and np.array_equal(data["features"][:, 2:6], mask.astype(np.float32)), "public eligible analytic action")
    minimum = np.min(np.where(mask, costs, np.float32(np.inf)), axis=1)
    gap = costs[np.arange(n), action] - minimum
    require(np.array_equal(data["labels"], ~(np.abs(gap) < 1e-10)) and np.array_equal(data["neural_gap"], gap), "original float32 annotation label/gap")
    for start, end in itertools.pairwise(offsets):
        last = None
        for t, row in enumerate(range(int(start), int(end))):
            age = t if last is None else t - last
            require(data["features"][row, 15] == np.float32(t / 2188)
                    and data["features"][row, 16] == np.float32(age / 2188)
                    and data["features"][row, 17] == np.float32(last is not None), "schedule-only query history")
            if data["scheduled_query"][row]:
                last = t
    weights = np.concatenate([np.full(int(length), n / (episodes * int(length)), dtype=np.float32) for length in lengths])
    return weights


def windows(order, offsets, np, *, batch_size=8, window=32):
    """Whole chronological episode batches; padded slots are never training rows."""
    for batch, left in enumerate(range(0, len(order), batch_size)):
        ids = order[left:left + batch_size]
        maximum = max(int(offsets[e + 1] - offsets[e]) for e in ids)
        for offset in range(0, maximum, window):
            indices = np.full((batch_size, window), -1, dtype=np.int64)
            for slot, episode in enumerate(ids):
                begin, end = int(offsets[episode]) + offset, int(offsets[episode + 1])
                count = min(window, max(0, end - begin))
                if count:
                    indices[slot, :count] = np.arange(begin, begin + count)
            yield batch, offset, indices


def window_loss(logits, targets, weights, torch):
    """Padded weights are zero; denominator is always 8*32, including tails."""
    weighted = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weights
    numerator = weighted.sum()
    return numerator / 256, numerator


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = self.launch = self.plan = None
        self.calls, self.pending, self.sequence, self.context = {}, {}, 0, {}
        self.completed = []
        self.receipt = {"version": VERSION, "status": "started", "qualified": False,
                        "model_training": True, "native_calls": 0, "new_label_calls": 0}

    def check(self):
        require(self.clock.now_ns() < self.launch["deadline_ns"], "original 300-second deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"], "training RSS cap")
        require(sum(p.stat().st_size for p in self.out.iterdir()) <= LIMITS["output_bytes"], "training output cap")

    def emit(self, name, value):
        with (self.out / name).open("ab") as stream:
            stream.write(encode(value))
            stream.flush()
            os.fsync(stream.fileno())

    def call(self, channel, function, *, describe=lambda _: {}):
        self.check()
        self.sequence += 1
        record = {"operation_id": self.sequence, "channel": channel, "context": dict(self.context)}
        count = self.calls.setdefault(channel, {"attempted": 0, "returned": 0, "seconds": 0.})
        count["attempted"] += 1
        self.pending[self.sequence] = record
        self.emit("work.jsonl", {"event": "attempt", **record})
        tick = time.perf_counter()
        value = function()
        elapsed = time.perf_counter() - tick
        self.emit("work.jsonl", {"event": "return", **record, "seconds": elapsed, "result": describe(value)})
        count["returned"] += 1
        count["seconds"] += elapsed
        del self.pending[record["operation_id"]]
        self.check()
        return value

    def bind(self):
        require(digest(path(CLOCK))["sha256"] == CLOCK_PIN, "clock source before import")
        spec = importlib.util.spec_from_file_location("_gate_training_clock", path(CLOCK))
        clock = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = clock
        spec.loader.exec_module(clock)
        self.clock = clock.SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, "missing original parent launch")
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and self.launch["pid"] == os.getpid()
                and self.launch["pgid"] == os.getpgrp() == os.getpid() and self.launch["parent_pid"] == os.getppid()
                and self.launch["cap_seconds"] == 300 and self.launch["clock_backend"] == self.clock.backend
                and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
                and self.launch["deadline_ns"] == self.launch["started_ns"] + 300 * 10**9
                and Path(self.launch["cwd"]).resolve() == Path.cwd().resolve() == ROOT
                and self.launch["watchdog_sha256"] == SUPERVISOR_PIN and self.launch["clock_source_sha256"] == CLOCK_PIN,
                "original bounded training supervisor")
        self.check()
        require(digest(self.args.plan, self.check)["sha256"] == self.args.plan_sha256, "external frozen training plan")
        p = read(self.args.plan)
        require(p["version"] == VERSION and p["status"] == "frozen_before_training" and p["configuration"] == CONFIG
                and p["limits"] == LIMITS and p["payloads"] == sorted(PAYLOADS)
                and p["runtime"] == runtime() and p["runtime"]["environment"] == THREADS, "fixed training recipe/runtime")
        sources, inputs, collection = metadata_closure(p["roles"], self.check)
        require(sources == p["sources"] and inputs == p["inputs"] and collection == p["collection"], "complete frozen evidence")
        self.plan = p
        self.receipt.update(plan_sha256=self.args.plan_sha256, supervision_sha256=digest(self.args.supervision)["sha256"],
                            sources=p["sources"], roles=p["roles"], limits=LIMITS, collection=collection)
        write(self.out / "started.json", {"request": {k: str(v) for k, v in vars(self.args).items()},
                                         "launch": self.launch, "started_ns": self.start})

    def checkpoint(self, model, kind, seed, initial=False):
        export = self.call("checkpoint_export", lambda: self.models.export_gate(model))
        name = f"{kind}-{seed}{'-initial' if initial else ''}.npz"

        def save():
            with (self.out / name).open("xb") as stream:
                self.np.savez_compressed(stream, **export)
                stream.flush()
                os.fsync(stream.fileno())
            with self.np.load(self.out / name, allow_pickle=False) as archive:
                require(set(archive.files) == set(export), "saved checkpoint field equality")
                for key, expected in export.items():
                    actual = archive[key]
                    require(actual.dtype == expected.dtype and actual.shape == expected.shape
                            and actual.tobytes() == expected.tobytes(), "saved checkpoint exact byte equality")
            return {"path": name, **digest(self.out / name, self.check)}

        return self.call("checkpoint_publication", save, describe=lambda value: value)

    def fit(self, kind, seed, data, weights):
        np, torch = self.np, self.torch
        self.context = {"fit_id": f"{kind}@{seed}", "phase": "training"}
        tick = time.perf_counter()
        model = self.call("model_initialization", lambda: self.models.make_gate(kind, seed))
        initial = self.checkpoint(model, kind, seed, initial=True)
        require(bool(torch.all(model.output.weight == 0))
                and model.output.bias.detach().numpy().tobytes() == np.asarray([math.log(19)], np.float32).tobytes(), "shared initial constant function")
        optimizer = self.call("optimizer_initialization", lambda: torch.optim.Adam(model.parameters(), lr=.0003))
        rng = np.random.default_rng(seed + 20000)
        offsets, n = data["episode_offsets"], len(data["features"])
        orders, updates = [], 0
        for epoch in range(1, 81):
            order = rng.permutation(60).astype(np.int64)
            order_sha = hashlib.sha256(order.tobytes()).hexdigest()
            orders.append(order_sha)
            self.emit("orders.jsonl", {"fit_id": f"{kind}@{seed}", "epoch": epoch, "episode_order": order.tolist(), "sha256": order_sha})
            state, previous_batch, numerator, epoch_updates, real_rows = None, None, 0., 0, 0
            for batch, offset, indices in windows(order, offsets, np):
                if batch != previous_batch:
                    state = torch.zeros(8, model.state_size, dtype=torch.float32)
                    previous_batch = batch
                valid = indices >= 0
                safe = np.maximum(indices, 0)
                x = np.zeros((8, 32, 31), np.float32)
                x[valid] = data["features"][safe[valid]]
                labels = np.zeros((8, 32), np.float32)
                labels[valid] = data["labels"][safe[valid]]
                weight = np.zeros((8, 32), np.float32)
                weight[valid] = weights[safe[valid]]
                tx, ty, tw = (torch.from_numpy(a) for a in (x, labels, weight))
                self.context = {"fit_id": f"{kind}@{seed}", "phase": "training", "epoch": epoch,
                                "batch": batch, "offset": offset, "rows": int(valid.sum()),
                                "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()}

                def update(tx=tx, ty=ty, tw=tw, state=state):
                    optimizer.zero_grad(set_to_none=True)
                    logits, final_state = model(tx, state)
                    loss, total = window_loss(logits, ty, tw, torch)
                    require(bool(torch.isfinite(loss)), "finite weighted loss")
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    require(bool(torch.isfinite(gradient)), "finite gradient norm")
                    optimizer.step()
                    return final_state.detach(), float(total.detach()), float(gradient)

                state, subtotal, _gradient = self.call("optimizer_update", update,
                    describe=lambda value: {"weighted_numerator": value[1], "loss": value[1] / 256, "gradient_norm": value[2]})
                numerator += subtotal
                epoch_updates += 1
                updates += 1
                real_rows += int(valid.sum())
            require(real_rows == n, "every TRAIN row exactly once per epoch including tails")
            self.emit("epochs.jsonl", {"fit_id": f"{kind}@{seed}", "epoch": epoch, "rows": real_rows,
                      "updates": epoch_updates, "weighted_bce": numerator / n,
                      "scope": "Online pre-update window losses; fixed row weights, not a final-model evaluation."})
        self.context = {"fit_id": f"{kind}@{seed}", "phase": "final_checkpoint"}
        final = self.checkpoint(model, kind, seed)
        steps = {name: int(optimizer.state[p]["step"].item()) for name, p in model.named_parameters()}
        require(set(steps.values()) == {updates}, "all parameters share complete optimizer step count")
        fit_seconds = time.perf_counter() - tick
        # Parity restores the published file. Neither in-memory training weights
        # nor a teacher-forced NumPy hidden state can substitute for this check.
        parity_tick = time.perf_counter()
        parity = self.parity(final, data)
        parity_seconds = time.perf_counter() - parity_tick
        result = {"fit_id": f"{kind}@{seed}", "kind": kind, "seed": seed, "parameters": self.models.parameter_count(kind),
                  "initial_checkpoint": initial, "final_checkpoint": final, "epochs": 80, "updates": updates,
                  "optimizer_steps": steps, "order_hashes": orders, "fit_seconds": fit_seconds,
                  "parity_seconds": parity_seconds, "parity": parity}
        self.emit("fits.jsonl", result)
        self.completed.append(result)
        return result

    def parity(self, checkpoint, data):
        np, torch = self.np, self.torch
        self.context["phase"] = "saved_checkpoint_parity"
        require(digest(self.out / checkpoint["path"], self.check) == {k: checkpoint[k] for k in ("sha256", "bytes")}, "published parity input")
        with np.load(self.out / checkpoint["path"], allow_pickle=False) as archive:
            export = {name: archive[name] for name in archive.files}
        frozen = self.call("numpy_restore", lambda: self.models.FrozenGate(export))
        reference = self.call("torch_restore", lambda: self.models.restore_gate(export))
        n = len(data["features"])
        arrays = {name: np.empty(n, dtype=np.float32) for name in ("logits_numpy", "logits_torch", "probabilities_numpy", "probabilities_torch")}
        arrays.update({name: np.empty(n, dtype=np.bool_) for name in ("queries_numpy", "queries_torch")})
        arrays.update({name: np.empty((n, frozen.state_size), dtype=np.float32) for name in ("states_numpy", "states_torch")})
        maximum_logit = maximum_state = 0.
        for episode, (begin, end) in enumerate(zip(data["episode_offsets"][:-1], data["episode_offsets"][1:], strict=True)):
            state, reference_state = frozen.initial_state(), torch.zeros(1, frozen.state_size)
            for row in range(int(begin), int(end)):
                self.context = {"fit_id": f"{frozen.kind}@{frozen.seed}", "phase": "saved_checkpoint_parity", "episode": episode, "row": row}
                x = data["features"][row]
                query, state, logit, probability = self.call("parity_numpy", lambda x=x, state=state: frozen.step(x, state))

                def reference_step(x=x, reference_state=reference_state):
                    with torch.no_grad():
                        logits, h = reference(torch.from_numpy(x.copy()).reshape(1, 1, 31), reference_state)
                    return h, float(logits.item()), float(torch.sigmoid(logits).item())

                reference_state, other_logit, other_probability = self.call("parity_torch", reference_step)
                other_state = reference_state.numpy()[0]
                logit_error = abs(logit - other_logit)
                state_error = float(np.max(np.abs(state.astype(np.float64) - other_state.astype(np.float64)))) if frozen.state_size else 0.
                other_query = other_probability >= .05
                maximum_logit, maximum_state = max(maximum_logit, logit_error), max(maximum_state, state_error)
                passed = logit_error <= 2e-5 and state_error <= 2e-5 and query == other_query
                self.emit("parity.jsonl", {**self.context, "maximum_logit_error": logit_error,
                          "maximum_state_error": state_error, "logit_numpy": logit, "logit_torch": other_logit,
                          "probability_numpy": probability, "probability_torch": other_probability,
                          "query_numpy": query, "query_torch": other_query, "passed": passed})
                require(passed, "fixed all-sequence deployment parity")
                for key, value in (("logits_numpy", logit), ("logits_torch", other_logit), ("probabilities_numpy", probability),
                                   ("probabilities_torch", other_probability), ("queries_numpy", query), ("queries_torch", other_query),
                                   ("states_numpy", state), ("states_torch", other_state)):
                    arrays[key][row] = value
        name = f"{frozen.kind}-{frozen.seed}-parity.npz"
        with (self.out / name).open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        return {"rows": n, "episodes": 60, "max_logit_error": maximum_logit, "max_state_error": maximum_state,
                "identical_query_decisions": True, "passed": True, "absolute_tolerance": 2e-5,
                "witness": {"path": name, **digest(self.out / name, self.check)}}

    def body(self):
        tick = time.perf_counter()
        import numpy as np
        import torch
        sys.path.insert(0, str(ROOT / "src"))
        from openjev.research import otto_query_gate_models as models
        self.np, self.torch, self.models = np, torch, models
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        write(self.out / "runtime.json", {**runtime(), "torch_threads": torch.get_num_threads(),
              "torch_interop_threads": torch.get_num_interop_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()})
        collection = Path(self.plan["roles"]["collection_receipt"]["path"]).parent
        with np.load(collection / "training-data.npz", allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
        weights = validate_data(data, np)
        episodes = [json.loads(line) for line in (collection / "episodes.jsonl").read_text().splitlines()]
        require(len(episodes) == 60 and [r["episode_index"] for r in episodes] == list(range(60)), "all canonical collection episodes")
        for i, row in enumerate(episodes):
            require(row["start_row"] == int(data["episode_offsets"][i])
                    and row["end_row"] == int(data["episode_offsets"][i + 1])
                    and row["rows"] == row["end_row"] - row["start_row"] > 0, "complete public sequence ownership")
        require(len(weights) == self.plan["collection"]["rows"], "collected row count")
        write(self.out / "data.json", {"rows": len(weights), "episodes": 60,
              "arrays": {k: {"sha256": hashlib.sha256(a.tobytes()).hexdigest(), "shape": list(a.shape), "dtype": str(a.dtype)} for k, a in data.items()},
              "weights_float32": weights.tolist(), "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
              "scope": "All collected TRAIN rows, episode-equal weights; no held-out inputs or new labels."})
        setup_seconds = time.perf_counter() - tick
        for name in ("work.jsonl", "orders.jsonl", "epochs.jsonl", "parity.jsonl", "fits.jsonl"):
            (self.out / name).touch(exist_ok=False)
        for seed in SEEDS:
            pair = [self.fit(kind, seed, data, weights) for kind in KINDS]
            require(pair[0]["order_hashes"] == pair[1]["order_hashes"] and pair[0]["updates"] == pair[1]["updates"], "paired epoch/window exposure")
        summary = {"version": VERSION, "qualified": True, "rows": len(weights), "episodes": 60, "completed_fits": 6,
                   "all_train_parity_passed": True, "fits": self.completed, "training_setup_seconds": setup_seconds,
                   "training_setup_allocation_per_fit_seconds": setup_seconds / 6,
                   "fit_seconds": math.fsum(r["fit_seconds"] for r in self.completed),
                   "parity_seconds": math.fsum(r["parity_seconds"] for r in self.completed),
                   "collection": self.plan["collection"], "calls": self.calls,
                   "scope": "Fixed final six fits with saved-checkpoint all-TRAIN deployment parity; no autonomous efficacy claim."}
        write(self.out / "summary.json", summary)

    def execute(self):
        self.out.mkdir(parents=False, exist_ok=False)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError("training supervisor terminated")))
        try:
            self.bind()
            self.body()
            for name, pin in self.plan["sources"].items():
                require(digest(path(name), self.check)["sha256"] == pin, "unchanged training sources")
            for name, desc in self.plan["inputs"].items():
                require(digest(path(name), self.check) == desc, "unchanged training inputs")
            require(digest(self.args.plan, self.check)["sha256"] == self.args.plan_sha256
                    and digest(self.args.supervision, self.check)["sha256"] == self.receipt["supervision_sha256"], "unchanged plan/launch")
            require(len(self.completed) == 6 and not self.pending and all(c["attempted"] == c["returned"] for c in self.calls.values()), "complete training work")
            require({p.name for p in self.out.iterdir()} == PAYLOADS, "closed exact training output inventory")
            files = {name: digest(self.out / name, self.check) for name in sorted(PAYLOADS)}
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status="completed", qualified=True, completed_fits=6, all_train_parity_passed=True,
                files=files, calls=self.calls, pending=[], started_ns=self.start, finished_ns=finished,
                wall_seconds=(finished - self.start) / 1e9, requires_successful_original_supervisor=True)
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "receipt": digest(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.receipt.update(status="failed", qualified=False, error=repr(error), traceback=traceback.format_exc(),
                                completed_fits=len(self.completed), calls=self.calls, pending=list(self.pending.values()))
            try:
                target = self.out / "receipt.json"
                if target.exists():
                    target.rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {p.name: {"bytes": p.stat().st_size} for p in self.out.iterdir() if p.is_file()}
                write(target, self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - preserve the primary training/publication failure
                error.add_note(f"Failure receipt publication: {secondary!r}")
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
    require(args.output.is_absolute() and Path(sys.executable).absolute() == ROOT / ".venv/bin/python", "qualified absolute invocation")
    if args.mode == "plan":
        freeze(args)
    else:
        Run(args).execute()


if __name__ == "__main__":
    main()
