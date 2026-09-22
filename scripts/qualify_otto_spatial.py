"""Disposable synthetic runtime qualification, never scientific model fitting."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
VERSION = "otto-spatial-qualification-v1"
KINDS = ("spatial", "neighbor_free", "cnn", "dense128", "statistics")
THREADS = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
SOURCES = (
    "src/openjev/research/otto_spatial_value.py",
    "src/openjev/research/otto_value_branches.py",
    "src/openjev/research/suspend_clock.py",
    "scripts/supervise_dialogue_observation_v2.py",
    "scripts/qualify_otto_spatial.py",
    "tests/test_otto_spatial_value.py",
    "tests/test_qualify_otto_spatial.py",
    "research/otto-spatial-design-review.md",
    "research/otto-spatial-qualification-protocol.md",
)
LIMITS = {"seconds": 600, "rss_bytes": 4 * 1024**3, "output_bytes": 128 * 1024**2}
CONFIG = {
    "kinds": list(KINDS), "batch_sizes": [128, 85], "fixture_seed": 41001,
    "model_seed": 42001, "c0": 0.4, "update_warmup": 2, "update_measured": 3,
    "diagnostic_first": 1, "diagnostic_warm": 3, "deployment_batch": 16,
    "deployment_warmup": 2, "deployment_measured": 5, "learning_rate": 0.001,
    "gradient_clip": 5.0, "parity_atol": 1e-8, "scientific_data_rows": 0,
    "native_steps": 0, "native_resets": 0, "external_model_calls": 0,
    "projection": {"seeds": 3, "epochs": 80, "train_full": 43, "train_tail": 1,
                   "valid_full": 8, "valid_tail": 1, "training_budget_seconds": 7200,
                   "allowance": 2, "cases": 72, "episode_cap": 2188},
}
EXPECTED = {"model_construction": 10, "optimizer_construction": 5,
            "torch_forward": 95, "backward": 50, "optimizer_update": 50,
            "export": 5, "deployment_construction": 5, "numpy_forward": 40}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def descriptor(path):
    return {"sha256": sha(path), "bytes": Path(path).stat().st_size}


def runtime():
    return {"python_executable": sys.executable, "python_version": sys.version,
            "platform": platform.platform(),
            "distributions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}


def freeze(path):
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, {"version": VERSION, "status": "frozen_before_execution", "root": str(ROOT),
                 "configuration": CONFIG, "limits": LIMITS, "expected_calls": EXPECTED,
                 "threads": THREADS, "runtime": runtime(),
                 "sources": {name: sha(ROOT / name) for name in SOURCES},
                 "scope": "Synthetic engineering only; no empirical datasets or retained scientific checkpoints."})
    print(json.dumps({"plan": str(path), **descriptor(path)}), flush=True)


def synthetic_fixture():
    """Fixed mixed shapes; no filesystem, simulator, global RNG or model access."""
    import numpy as np

    rng = np.random.default_rng(CONFIG["fixture_seed"])
    centered = np.zeros((128, 105, 105), dtype=np.float32)
    positions = np.empty((128, 2), dtype=np.int64)
    sensing = np.array([3 + i % 3 for i in range(128)], dtype=np.float32)
    target = np.empty(128, dtype=np.float32)
    anchors = ((0, 0), (0, 52), (52, 0), (52, 52), (0, 26), (26, 52), (26, 26))
    masses = (0.0, 1e-12, 0.2, 1.0, 1.0)
    for i in range(128):
        q = anchors[i % len(anchors)]
        positions[i] = q
        field = np.zeros((53, 53), dtype=np.float64)
        case = i % 5
        if case in (1, 2):
            field[(3*i) % 53, (7*i+1) % 53] = 1
        elif case == 3:
            field[(3*i) % 53, (7*i+1) % 53] += .4
            field[(11*i+2) % 53, (13*i+3) % 53] += .6
        elif case == 4:
            field = rng.uniform(.1, 1, size=(53, 53))
            field /= field.sum()
        field *= masses[case]
        centered[i, 52-q[0]:105-q[0], 52-q[1]:105-q[1]] = field
        target[i] = .15 + .35*masses[case] + .05*(q[0]+q[1])/104 + .01*sensing[i]
    return centered, positions, sensing, target


def project(records):
    """Arithmetic projection from timings, not an efficacy or confidence bound."""
    def maximum(kind, stage, batch=None):
        rows = [r["seconds"] for r in records if r["kind"] == kind and r["stage"] == stage
                and (batch is None or r.get("batch") == batch)
                and not r.get("warmup", False) and not r.get("first", False)]
        require(bool(rows), f"missing projection records: {kind}/{stage}/{batch}")
        return max(rows)

    results = {}
    for kind in KINDS:
        setup = sum(r["seconds"] for r in records if r["kind"] == kind and r["stage"] in
                    ("model_init", "optimizer_init", "checkpoint_write"))
        update = 80 * (43 * maximum(kind, "update", 128) + maximum(kind, "update", 85))
        final_diagnostics = 51 * maximum(kind, "diagnostic", 128) + 2 * maximum(kind, "diagnostic", 85)
        latency = maximum(kind, "deployment", 16)
        results[kind] = {"one_fit_seconds": setup+update+final_diagnostics,
                         "training_updates_seconds": update, "final_diagnostics_seconds": final_diagnostics,
                         "head_only_seconds_per_max_length_episode": latency*2188,
                         "head_only_seconds_for_72_cases_three_seeds_at_cap": latency*2188*72*3}
    total = 3 * sum(r["one_fit_seconds"] for r in results.values())
    return {"families": results, "fifteen_fit_seconds": total, "two_times_allowance_seconds": 2*total,
            "training_budget_seconds": 7200, "within_planning_budget": 2*total <= 7200,
            "not_admission": True,
            "excluded": ["data authentication", "shuffle and copies", "per-epoch diagnostics",
                         "scientific journal I/O", "complete supervision", "branch construction",
                         "action selection", "posterior updates", "variable real-state workload"]}


def timing_distributions(records):
    result = []
    for kind in KINDS:
        for stage, sizes in (("update", (128, 85)), ("diagnostic", (128, 85)), ("deployment", (16,))):
            for batch in sizes:
                rows = [r for r in records if r["kind"] == kind and r["stage"] == stage and r["batch"] == batch]
                warm = [r["seconds"] for r in rows if not r.get("warmup", False) and not r.get("first", False)]
                require(bool(warm), "warmed timing samples required")
                result.append({"kind": kind, "stage": stage, "batch": batch,
                               "all_seconds": [r["seconds"] for r in rows], "warm_seconds": warm,
                               "warm_min": min(warm), "warm_median": statistics.median(warm), "warm_max": max(warm)})
    return result


class Run:
    def __init__(self, args):
        self.args = args
        self.out = args.output.absolute()
        self.out.mkdir(parents=True, exist_ok=False)
        self.calls = {name: {"attempted": 0, "returned": 0} for name in EXPECTED}
        self.records = []
        self.peak_rss = 0
        self.clock = None
        self.start = None
        self.plan = None
        self.launch = None

    def check(self):
        if self.clock is not None and self.launch is not None:
            require(self.clock.now_ns() < self.launch["deadline_ns"], "qualification deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.peak_rss = max(self.peak_rss, int(rss if sys.platform == "darwin" else rss * 1024))
        require(self.peak_rss < LIMITS["rss_bytes"], "qualification peak RSS limit")
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) < LIMITS["output_bytes"],
                "qualification output limit")

    def counted(self, name, function):
        self.check()
        require(self.calls[name]["attempted"] < EXPECTED[name], "operation allocation exceeded: " + name)
        self.calls[name]["attempted"] += 1
        self.call_event(name, "attempted")
        result = function()
        self.calls[name]["returned"] += 1
        self.call_event(name, "returned")
        self.check()
        return result

    def call_event(self, name, event):
        with (self.out / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps({"name": name, "event": event, "counts": self.calls[name],
                                     "time_ns": self.clock.now_ns()}, sort_keys=True)+"\n")
            stream.flush()

    def timed(self, kind, stage, function, **metadata):
        self.check()
        start = self.clock.now_ns()
        value = function()
        finish = self.clock.now_ns()
        row = {"kind": kind, "stage": stage, "seconds": (finish-start)/1e9, **metadata}
        self.records.append(row)
        with (self.out / "operations.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
        self.check()
        return value

    def bind(self):
        from openjev.research.suspend_clock import SuspendClock

        self.clock = SuspendClock()
        self.start = self.clock.now_ns()
        require(self.args.plan.is_absolute() and not self.args.plan.is_symlink(), "absolute regular plan")
        require(sha(self.args.plan) == self.args.plan_sha256, "external plan digest")
        self.plan = json.loads(self.args.plan.read_text())
        p = self.plan
        require(p["version"] == VERSION and p["status"] == "frozen_before_execution", "frozen plan version")
        require(p["root"] == str(ROOT) and Path.cwd() == ROOT, "qualification checkout")
        require(p["configuration"] == CONFIG and p["limits"] == LIMITS and p["expected_calls"] == EXPECTED,
                "exact qualification workload")
        require(p["runtime"] == runtime() and p["threads"] == THREADS, "frozen runtime")
        require(all(os.environ.get(k) == v for k, v in THREADS.items()), "one CPU runtime thread")
        self.sources()
        while not self.args.supervision.exists():
            require(self.clock.now_ns()-self.start < 5_000_000_000, "supervision launch absent")
            time.sleep(.01)
        self.launch = json.loads(self.args.supervision.read_text())
        launch = self.launch
        command = [sys.executable, *sys.argv]
        require(launch["command"] == command and launch["pid"] == os.getpid()
                and launch["pgid"] == os.getpgrp() and launch["parent_pid"] == os.getppid(),
                "exact supervised command and process")
        require(launch["cap_seconds"] == LIMITS["seconds"] and launch["clock_backend"] == self.clock.backend
                and launch["cwd"] == str(ROOT) and launch["started_ns"] <= self.start < launch["deadline_ns"]
                and launch["deadline_ns"] == launch["started_ns"]+LIMITS["seconds"]*10**9,
                "bounded original launch")
        require(launch["watchdog_sha256"] == p["sources"]["scripts/supervise_dialogue_observation_v2.py"]
                and launch["clock_source_sha256"] == p["sources"]["src/openjev/research/suspend_clock.py"],
                "qualified supervisor source")
        write(self.out/"started.json", {"launch": launch, "request": {k: str(v) for k, v in vars(self.args).items()},
                                       "started_ns": self.start, "plan_sha256": self.args.plan_sha256})
        write(self.out/"runtime.json", runtime())
        self.check()

    def sources(self):
        require(set(self.plan["sources"]) == set(SOURCES), "exact source closure")
        for name, pin in self.plan["sources"].items():
            path = ROOT/name
            require(not path.is_symlink() and sha(path) == pin, "source changed: " + name)

    def work(self):
        import numpy as np
        import torch

        from openjev.research import otto_spatial_value as model

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        require(tuple(model.KINDS) == KINDS, "all five model families required")
        arrays = self.timed("shared", "fixture", synthetic_fixture)
        centered, positions, sensing, target = arrays
        with (self.out/"fixtures.npz").open("xb") as stream:
            np.savez(stream, centered=centered, positions=positions, sensing=sensing, target=target)
        tensors = tuple(torch.from_numpy(a.copy()) for a in arrays)
        deployment = (centered[:16].astype(np.float64), positions[:16].copy(), sensing[:16].astype(np.float64))
        parity = {}
        for kind in KINDS:
            parity[kind] = self.family(kind, model, torch, np, tensors, deployment)
        require(all(v == {"attempted": EXPECTED[k], "returned": EXPECTED[k]} for k, v in self.calls.items()),
                "exact completed call allocation")
        self.sources()
        self.check()
        write(self.out/"summary.json", {"scope": "synthetic engineering only", "parity": parity,
                                       "projection": project(self.records), "calls": self.calls,
                                       "timings": timing_distributions(self.records),
                                       "learned_task_advantage_established": False})

    def family(self, kind, model, torch, np, tensors, deployment):
        head = self.timed(kind, "model_init", lambda: self.counted(
            "model_construction", lambda: model.make_head(kind, CONFIG["model_seed"], CONFIG["c0"])))
        require(sum(p.numel() for p in head.parameters()) == model.parameter_count(kind), "parameter count")
        optimizer = self.timed(kind, "optimizer_init", lambda: self.counted(
            "optimizer_construction", lambda: torch.optim.Adam(head.parameters(), lr=CONFIG["learning_rate"])))
        for batch in CONFIG["batch_sizes"]:
            x, q, length, y = (a[:batch] for a in tensors)

            def update(x=x, q=q, length=length, y=y):
                optimizer.zero_grad(set_to_none=True)
                prediction = self.counted("torch_forward", lambda: head(x, q, length))
                loss = torch.mean((prediction-y)**2)
                require(bool(torch.isfinite(loss)), "finite synthetic loss")
                self.counted("backward", loss.backward)
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), CONFIG["gradient_clip"],
                                                     error_if_nonfinite=True)
                self.counted("optimizer_update", optimizer.step)
                return float(loss.detach()), float(norm)

            for rep in range(5):
                loss, norm = self.timed(kind, "update", update, batch=batch, repeat=rep, warmup=rep < 2)
                with (self.out/"synthetic-losses.jsonl").open("a") as stream:
                    stream.write(json.dumps({"kind": kind, "batch": batch, "repeat": rep,
                                             "loss": loss, "gradient_norm": norm})+"\n")
            with torch.no_grad():
                for rep in range(4):
                    self.timed(kind, "diagnostic", lambda x=x, q=q, length=length: self.counted(
                        "torch_forward", lambda: head(x, q, length)), batch=batch, repeat=rep, first=rep == 0)

        def save_checkpoint():
            checkpoint = self.counted("export", lambda: model.export_head(head))
            with (self.out/f"disposable-{kind}.npz").open("xb") as stream:
                np.savez(stream, **checkpoint)

        self.timed(kind, "checkpoint_write", save_checkpoint)

        def load_checkpoint():
            with np.load(self.out/f"disposable-{kind}.npz", allow_pickle=False) as saved:
                result = {name: saved[name].copy() for name in saved.files}
            model.validate_head(result)
            return result

        saved = self.timed(kind, "checkpoint_read", load_checkpoint)
        frozen = self.timed(kind, "deployment_init", lambda: self.counted(
            "deployment_construction", lambda: model.FrozenValue(saved)))

        def reference():
            restored = self.counted("model_construction", lambda: model.make_head(kind, 42001, .4))
            with torch.no_grad():
                for name, parameter in restored.named_parameters():
                    parameter.copy_(torch.from_numpy(saved[name]))
                for name, buffer in restored.named_buffers():
                    buffer.copy_(torch.from_numpy(saved[name]))
            return restored.double().eval()

        restored = self.timed(kind, "torch64_restore", reference)
        with torch.no_grad():
            reference_values = self.timed(kind, "torch64_parity", lambda: self.counted(
                "torch_forward", lambda: restored(*(torch.from_numpy(a) for a in deployment)))).numpy()
        deployed_values = self.timed(kind, "numpy64_parity", lambda: self.counted(
            "numpy_forward", lambda: frozen.normalized(*deployment)))
        difference = float(np.max(np.abs(reference_values-deployed_values)))
        require(np.isfinite(reference_values).all() and np.isfinite(deployed_values).all()
                and difference <= CONFIG["parity_atol"], "independent deployment parity: " + kind)
        parity_result = {"maximum_absolute_error": difference, "passes": True}
        with (self.out/f"parity-{kind}.npz").open("xb") as stream:
            np.savez(stream, torch64=reference_values, numpy64=deployed_values)
        for rep in range(7):
            self.timed(kind, "deployment", lambda: self.counted(
                "numpy_forward", lambda: frozen.normalized(*deployment)), batch=16, repeat=rep, warmup=rep < 2)
        return parity_result

    def execute(self):
        errors = []
        try:
            self.bind()
            self.work()
            self.check()
        except BaseException as caught:  # noqa: BLE001 - preserve failure evidence, including interrupts
            errors.append(repr(caught))
            traceback.print_exc()
        finish = None
        try:
            finish = self.clock.now_ns() if self.clock else None
        except BaseException as caught:  # noqa: BLE001 - preserve failure evidence, including interrupts
            errors.append("terminal clock: " + repr(caught))
        files = {}
        try:
            files = {p.name: descriptor(p) for p in sorted(self.out.iterdir()) if p.is_file()}
        except BaseException as caught:  # noqa: BLE001 - preserve failure evidence, including interrupts
            errors.append("artifact closure: " + repr(caught))
        receipt = {"version": VERSION, "status": "failed" if errors else "completed", "errors": errors,
                   "plan_sha256": self.args.plan_sha256, "calls": self.calls,
                   "scientific_data_rows": 0, "native_steps": 0, "native_resets": 0, "external_model_calls": 0,
                   "started_ns": self.start, "finished_ns": finish,
                   "wall_seconds": (finish-self.start)/1e9 if finish and self.start else None,
                   "peak_rss_bytes": self.peak_rss,
                   "sources": self.plan["sources"] if self.plan else None,
                   "requires_successful_original_supervisor_terminal": True, "files": files}
        try:
            write(self.out/"receipt.json", receipt)
            self.check()
        except BaseException as caught:  # noqa: BLE001 - preserve failure evidence, including interrupts
            errors.append("receipt publication or late resource check: " + repr(caught))
            # A prior receipt alone never admits success. Preserve the late
            # failure separately and return nonzero to the original parent.
            try:
                write(self.out/"late-failure.json", {"status": "failed", "errors": errors})
            except BaseException as publication_error:  # noqa: BLE001 - preserve original failure in supervisor log
                errors.append("late failure publication: " + repr(publication_error))
        print(json.dumps({"status": "failed" if errors else "completed", "receipt": str(self.out/"receipt.json"),
                          "wall_seconds": receipt["wall_seconds"], "errors": errors}), flush=True)
        return int(bool(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "run"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--supervision", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.plan)
        return 0
    require(args.output is not None and args.supervision is not None and args.plan_sha256 is not None,
            "run requires explicit output, supervision and external plan digest")
    return Run(args).execute()


if __name__ == "__main__":
    raise SystemExit(main())
