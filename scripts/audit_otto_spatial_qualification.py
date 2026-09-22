"""Saved-only independent audit; never imports a model or repeats a forward."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import signal
import statistics
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLOCK = ROOT / "src/openjev/research/suspend_clock.py"
CLOCK_SHA = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
VERSION = "otto-spatial-qualification-saved-audit-v1"
KINDS = ("spatial", "neighbor_free", "cnn", "dense128", "statistics")
COUNTS = {"model_construction": 10, "optimizer_construction": 5, "torch_forward": 95,
              "backward": 50, "optimizer_update": 50, "export": 5, "deployment_construction": 5, "numpy_forward": 40}
LIMITS = {"seconds": 600, "rss_bytes": 4*1024**3, "output_bytes": 128*1024**2}
SCOPE = ("Saved file identities, journal order, tensor schemas/finiteness, saved parity pairs and timing arithmetic. "
         "No independent forward, gradient, optimizer, native environment or runtime-speed measurement. "
         "Execution truth and recorded durations remain inherited from the pinned producer and supervisor. "
         "Synthetic engineering does not establish scientific efficacy or admit a scientific run.")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def descriptor(path):
    return {"sha256": sha(path), "bytes": Path(path).stat().st_size}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


class Audit:
    def __init__(self, args):
        self.args, self.checks, self.peak_rss = args, 0, 0
        if sha(CLOCK) != CLOCK_SHA:
            raise ValueError("clock source changed")
        spec = importlib.util.spec_from_file_location("spatial_audit_clock", CLOCK)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.clock = module.SuspendClock()
        self.start = self.clock.now_ns()
        self.out = args.output
        self.out.mkdir(parents=True, exist_ok=False)
        signal.signal(signal.SIGALRM, self.timeout)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["seconds"])
        self.inputs = {}

    @staticmethod
    def timeout(*_):
        raise TimeoutError("saved audit deadline")

    def check(self, condition, message):
        self.checks += 1
        if not condition:
            raise ValueError(message)

    def budget(self):
        self.check(self.clock.now_ns()-self.start < LIMITS["seconds"]*10**9, "audit elapsed cap")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.peak_rss = max(self.peak_rss, int(rss if sys.platform == "darwin" else rss*1024))
        self.check(self.peak_rss <= LIMITS["rss_bytes"], "audit RSS cap")
        self.check(sum(p.stat().st_size for p in self.out.iterdir()) <= LIMITS["output_bytes"], "audit disk cap")

    def read(self, path, pin=None):
        self.budget()
        self.check(path.is_absolute() and path.is_file() and not path.is_symlink(), "regular absolute input")
        found = descriptor(path)
        self.check(pin is None or found["sha256"] == pin, "external input digest: " + str(path))
        self.inputs[str(path)] = found
        return json.loads(path.read_text())

    def close_number(self, actual, expected, name):
        self.check(type(actual) in (int, float) and math.isfinite(actual)
                   and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10), name)

    def authenticate(self):
        a = self.args
        plan = self.read(a.plan, a.plan_sha256)
        worker = self.read(a.run/"receipt.json", a.receipt_sha256)
        terminal = self.read(a.terminal, a.terminal_sha256)
        launch = self.read(a.launch, a.launch_sha256)
        self.check(plan["version"] == worker["version"] == "otto-spatial-qualification-v1"
                   and plan["status"] == "frozen_before_execution", "plan/worker versions")
        self.check(worker["status"] == terminal["status"] == "completed"
                   and worker["errors"] == [] and terminal["returncode"] == 0
                   and terminal["timed_out"] is False and terminal["error"] is None
                   and terminal["clock_error"] is None and terminal["timing_available"] is True
                   and terminal["group_absent"] is True and terminal["cleanup"]["reaped"] is True
                   and terminal["cleanup"]["errors"] == [] and terminal["cleanup"]["group_absent"] is True,
                   "successful original worker and parent")
        self.check(plan["root"] == str(ROOT) and worker["plan_sha256"] == a.plan_sha256
                   and worker["sources"] == plan["sources"] and len(plan["sources"]) == 9, "source/plan joins")
        for name, pin in plan["sources"].items():
            path = ROOT/name
            self.check(not path.is_symlink() and sha(path) == pin, "source identity: " + name)
        self.sources = plan["sources"]
        for key, value in launch.items():
            self.check(terminal[key] == value, "parent launch/terminal join: " + key)
        command = [plan["runtime"]["python_executable"], str(ROOT/"scripts/qualify_otto_spatial.py"), "run",
                   "--plan", str(a.plan), "--plan-sha256", a.plan_sha256,
                   "--output", str(a.run), "--supervision", str(a.launch)]
        self.check(launch["command"] == command and launch["cwd"] == str(ROOT)
                   and launch["version"] == "dialogue-observation-supervision-v2"
                   and launch["pid"] == launch["pgid"] and launch["parent_pid"] != launch["pid"], "exact launch")
        self.check(launch["clock_backend"] in ("mach_continuous_time", "CLOCK_BOOTTIME")
                   and launch["clock_source_sha256"] == CLOCK_SHA
                   and launch["watchdog_sha256"] == plan["sources"]["scripts/supervise_dialogue_observation_v2.py"],
                   "parent implementation pins")
        self.check(plan["limits"] == LIMITS and launch["cap_seconds"] == 600
                   and launch["deadline_ns"] == launch["started_ns"]+600*10**9
                   and launch["started_ns"] <= worker["started_ns"] <= worker["finished_ns"]
                   <= terminal["finished_ns"] < launch["deadline_ns"], "exact time boundaries")
        self.close_number(worker["wall_seconds"], (worker["finished_ns"]-worker["started_ns"])/1e9, "worker time")
        self.close_number(terminal["wall_seconds"], terminal["elapsed_ns"]/1e9, "parent time")
        self.check(terminal["elapsed_ns"] == terminal["finished_ns"]-terminal["started_ns"], "parent duration")
        self.check(worker["requires_successful_original_supervisor_terminal"] is True
                   and 0 < worker["peak_rss_bytes"] <= LIMITS["rss_bytes"], "recorded resource contract")
        files = {"started.json", "runtime.json", "fixtures.npz", "calls.jsonl", "operations.jsonl",
                 "synthetic-losses.jsonl", "summary.json"}
        files |= {f"{prefix}-{kind}.npz" for prefix in ("disposable", "parity") for kind in KINDS}
        self.check(set(worker["files"]) == files and {p.name for p in a.run.iterdir()} == files | {"receipt.json"},
                   "exact 17-payload closure")
        for name in sorted(files):
            self.budget()
            path = a.run/name
            self.check(path.is_file() and not path.is_symlink() and descriptor(path) == worker["files"][name],
                       "worker payload: " + name)
            self.inputs[str(path)] = worker["files"][name]
        self.check(sum(p.stat().st_size for p in a.run.iterdir()) <= LIMITS["output_bytes"], "actual worker output cap")
        summary = self.read(a.run/"summary.json", a.summary_sha256)
        started = self.read(a.run/"started.json")
        self.check(started == {"launch": launch, "started_ns": worker["started_ns"], "plan_sha256": a.plan_sha256,
                              "request": {"mode": "run", "output": str(a.run), "plan": str(a.plan),
                                              "plan_sha256": a.plan_sha256, "supervision": str(a.launch)}}, "started identity")
        self.check(self.read(a.run/"runtime.json") == plan["runtime"], "runtime identity")
        self.check(plan["expected_calls"] == COUNTS and tuple(plan["configuration"]["kinds"]) == KINDS, "workload")
        for key in ("scientific_data_rows", "native_steps", "native_resets", "external_model_calls"):
            self.check(worker[key] == plan["configuration"][key] == 0, "scope: " + key)
        self.check(summary["learned_task_advantage_established"] is False, "no efficacy admission")
        return plan, worker, terminal, summary

    def journal(self, worker, summary):
        calls = [json.loads(line) for line in (self.args.run/"calls.jsonl").read_text().splitlines()]
        expected = []
        for _ in KINDS:
            expected += ["model_construction", "optimizer_construction"]
            for _batch in (128, 85):
                expected += ["torch_forward", "backward", "optimizer_update"]*5 + ["torch_forward"]*4
            expected += ["export", "deployment_construction", "model_construction", "torch_forward"] + ["numpy_forward"]*8
        self.check(len(calls) == 2*len(expected) == 520, "260 actual call pairs")
        counts = {name: {"attempted": 0, "returned": 0} for name in COUNTS}
        previous = worker["started_ns"]
        for pair, name in enumerate(expected):
            for offset, event in enumerate(("attempted", "returned")):
                row = calls[2*pair+offset]
                counts[name][event] += 1
                self.check(row["name"] == name and row["event"] == event and row["counts"] == counts[name], "call sequence")
                self.check(previous <= row["time_ns"] <= worker["finished_ns"], "call chronology")
                previous = row["time_ns"]
        self.check(counts == worker["calls"] == summary["calls"]
                   == {k: {"attempted": v, "returned": v} for k, v in COUNTS.items()}, "exact final counters")
        operations = [json.loads(line) for line in (self.args.run/"operations.jsonl").read_text().splitlines()]
        identities = [{"kind": "shared", "stage": "fixture"}]
        loss_order = []
        for kind in KINDS:
            identities += [{"kind": kind, "stage": s} for s in ("model_init", "optimizer_init")]
            for batch in (128, 85):
                for rep in range(5):
                    identities.append({"kind": kind, "stage": "update", "batch": batch, "repeat": rep, "warmup": rep < 2})
                    loss_order.append({"kind": kind, "batch": batch, "repeat": rep})
                identities += [{"kind": kind, "stage": "diagnostic", "batch": batch, "repeat": r, "first": r == 0} for r in range(4)]
            identities += [{"kind": kind, "stage": s} for s in ("checkpoint_write", "checkpoint_read", "deployment_init",
                                                              "torch64_restore", "torch64_parity", "numpy64_parity")]
            identities += [{"kind": kind, "stage": "deployment", "batch": 16, "repeat": r, "warmup": r < 2} for r in range(7)]
        self.check(len(operations) == len(identities) == 166, "exact operation count")
        for actual, expected_row in zip(operations, identities, strict=True):
            self.check({k: v for k, v in actual.items() if k != "seconds"} == expected_row, "operation order")
            self.check(type(actual["seconds"]) is float and math.isfinite(actual["seconds"])
                       and 0 <= actual["seconds"] <= worker["wall_seconds"], "finite paid operation")
        self.check(math.fsum(r["seconds"] for r in operations) <= worker["wall_seconds"], "disjoint recorded intervals")
        losses = [json.loads(line) for line in (self.args.run/"synthetic-losses.jsonl").read_text().splitlines()]
        self.check(len(losses) == len(loss_order) == 50, "all synthetic update diagnostics")
        for row, identity in zip(losses, loss_order, strict=True):
            self.check({k: row[k] for k in identity} == identity, "loss chronology")
            self.check(all(type(row[k]) is float and math.isfinite(row[k]) and row[k] >= 0
                           for k in ("loss", "gradient_norm")), "finite saved loss/gradient norm")
        return operations

    def tensors(self, summary):
        import numpy as np

        def load(name):
            self.budget()
            with np.load(self.args.run/name, allow_pickle=False) as saved:
                return {key: saved[key].copy() for key in saved.files}

        fixture = load("fixtures.npz")
        self.check(set(fixture) == {"centered", "positions", "sensing", "target"}, "fixture arrays")
        for key, shape, dtype in (("centered", (128, 105, 105), np.float32), ("positions", (128, 2), np.int64),
                                  ("sensing", (128,), np.float32), ("target", (128,), np.float32)):
            self.check(fixture[key].shape == shape and fixture[key].dtype == dtype
                       and np.isfinite(fixture[key]).all(), "fixture shape/dtype/finite")
        anchors = ((0, 0), (0, 52), (52, 0), (52, 52), (0, 26), (26, 52), (26, 26))
        for i, grid in enumerate(fixture["centered"]):
            q = anchors[i % 7]
            self.check(tuple(fixture["positions"][i]) == q and fixture["sensing"][i] == 3+i % 3, "fixture public context")
            physical = grid[52-q[0]:105-q[0], 52-q[1]:105-q[1]]
            restored = np.zeros_like(grid)
            restored[52-q[0]:105-q[0], 52-q[1]:105-q[1]] = physical
            self.check(np.array_equal(restored, grid) and np.all(grid >= 0), "exact fixture board support")
            mass = (0.0, 1e-12, .2, 1.0, 1.0)[i % 5]
            self.check(math.isclose(float(grid.sum(dtype=np.float64)), mass, rel_tol=1e-6, abs_tol=1e-20), "fixture mass")
            # The producer's final term is a NumPy float32 scalar; its final
            # addition therefore also occurs in float32 under the pinned ABI.
            target = np.float32(.15+.35*mass+.05*sum(q)/104) + np.float32(.01)*fixture["sensing"][i]
            self.check(fixture["target"][i] == target, "fixed synthetic target")
        counts, parity = {}, {}
        for kind in KINDS:
            checkpoint = load(f"disposable-{kind}.npz")
            shapes = {}
            if kind == "cnn":
                for i, channels in enumerate((1, 4, 4)):
                    shapes[f"conv_weight_{i}"], shapes[f"conv_bias_{i}"] = (4, channels, 3, 3), (4,)
            if kind in ("spatial", "neighbor_free", "cnn"):
                sizes = (20, 8, 8) if kind == "cnn" else (12, 24, 8)
                for i in (0, 1):
                    shapes[f"cell_weight_{i}"], shapes[f"cell_bias_{i}"] = (sizes[i+1], sizes[i]), (sizes[i+1],)
            sizes = (11028, 128, 1) if kind == "dense128" else (12, 16, 1)
            for i in (0, 1):
                shapes[f"readout_weight_{i}"], shapes[f"readout_bias_{i}"] = (sizes[i+1], sizes[i]), (sizes[i+1],)
            self.check(set(checkpoint) == {*shapes, "c0", "version", "kind", "input_dim"}, "checkpoint exact schema")
            self.check(checkpoint["kind"].shape == () and checkpoint["kind"].item() == kind
                       and checkpoint["version"].item() == "otto-spatial-value-v1"
                       and checkpoint["input_dim"].item() == 11028, "checkpoint metadata")
            for key, shape in {**shapes, "c0": ()}.items():
                self.check(checkpoint[key].shape == shape and checkpoint[key].dtype == np.float32
                           and np.isfinite(checkpoint[key]).all(), "checkpoint tensor contract")
            self.check(checkpoint["c0"].item() == float(np.float32(.4)), "fixed baseline")
            counts[kind] = sum(math.prod(shape) for shape in shapes.values())
            self.check(counts[kind] == {"spatial": 737, "neighbor_free": 737, "cnn": 801, "dense128": 1411841, "statistics": 225}[kind],
                       "independent parameter count")
            pair = load(f"parity-{kind}.npz")
            self.check(set(pair) == {"torch64", "numpy64"}, "parity payload keys")
            for values in pair.values():
                self.check(values.shape == (16,) and values.dtype == np.float64 and np.isfinite(values).all(), "parity values")
            difference = float(np.max(np.abs(pair["torch64"]-pair["numpy64"])))
            self.check(difference <= 1e-8 and summary["parity"][kind]
                       == {"maximum_absolute_error": difference, "passes": True}, "saved parity arithmetic")
            parity[kind] = difference
        return counts, parity

    def timings(self, operations, summary):
        distributions, families, medians = [], {}, {}
        for kind in KINDS:
            maxima = {}
            for stage, sizes in (("update", (128, 85)), ("diagnostic", (128, 85)), ("deployment", (16,))):
                for batch in sizes:
                    rows = [r for r in operations if r["kind"] == kind and r["stage"] == stage and r["batch"] == batch]
                    warmed = [r["seconds"] for r in rows if not r.get("warmup", False) and not r.get("first", False)]
                    distributions.append({"kind": kind, "stage": stage, "batch": batch, "all_seconds": [r["seconds"] for r in rows],
                                              "warm_seconds": warmed, "warm_min": min(warmed), "warm_median": statistics.median(warmed),
                                              "warm_max": max(warmed)})
                    maxima[stage, batch] = max(warmed)
                    if stage == "deployment":
                        medians[kind] = statistics.median(warmed)*1000
            setup = math.fsum(r["seconds"] for r in operations if r["kind"] == kind
                              and r["stage"] in ("model_init", "optimizer_init", "checkpoint_write"))
            updates = 80*(43*maxima["update", 128]+maxima["update", 85])
            diagnostics = 51*maxima["diagnostic", 128]+2*maxima["diagnostic", 85]
            families[kind] = {"one_fit_seconds": setup+updates+diagnostics, "training_updates_seconds": updates,
                                  "final_diagnostics_seconds": diagnostics,
                                  "head_only_seconds_per_max_length_episode": maxima["deployment", 16]*2188,
                                  "head_only_seconds_for_72_cases_three_seeds_at_cap": maxima["deployment", 16]*2188*216}
        self.check(summary["timings"] == distributions and len(distributions) == 25, "all timing samples/statistics")
        total = 3*math.fsum(v["one_fit_seconds"] for v in families.values())
        projected = summary["projection"]
        self.check(set(projected["families"]) == set(KINDS), "all projection families")
        for kind, values in families.items():
            self.check(set(projected["families"][kind]) == set(values), "projection field set")
            for key, value in values.items():
                self.close_number(projected["families"][kind][key], value, "projection: " + kind + "/" + key)
        self.close_number(projected["fifteen_fit_seconds"], total, "15-fit projection")
        self.close_number(projected["two_times_allowance_seconds"], 2*total, "fixed 2x allowance")
        self.check(projected["training_budget_seconds"] == 7200
                   and projected["within_planning_budget"] == (2*total <= 7200)
                   and projected["not_admission"] is True, "exact planning predicate, no admission")
        return {"families": families, "fifteen_fit_seconds": total, "two_times_allowance_seconds": 2*total,
                    "within_planning_budget": 2*total <= 7200, "deployment_warm_median_ms": medians}

    def execute(self):
        error, result = None, None
        source = descriptor(Path(__file__))
        try:
            write(self.out/"started.json", {"version": VERSION, "inputs": {k: str(v) for k, v in vars(self.args).items()},
                                               "limits": LIMITS, "source": source, "scope": SCOPE})
            _, worker, terminal, summary = self.authenticate()
            operations = self.journal(worker, summary)
            counts, parity = self.tensors(summary)
            projection = self.timings(operations, summary)
            for name, pin in self.sources.items():
                self.check(sha(ROOT/name) == pin, "source unchanged after audit")
            for name, original in self.inputs.items():
                self.check(descriptor(Path(name)) == original, "input unchanged after audit")
            self.check(descriptor(Path(__file__)) == source, "auditor source unchanged")
            result = {"agreement": True, "checks": self.checks, "source_files": 9, "worker_payloads": 17,
                          "operation_records": 166, "call_journal_events": 520, "synthetic_updates": 50,
                          "saved_parity_pairs": 80, "parameter_counts": counts, "saved_parity_max_errors": parity,
                          "projection": projection, "worker_seconds": worker["wall_seconds"], "parent_seconds": terminal["wall_seconds"],
                          "scope": SCOPE, "model_forward_calls": 0, "optimizer_calls": 0, "native_calls": 0}
            write(self.out/"summary.json", result)
            self.budget()
        except BaseException as caught:  # noqa: BLE001 - preserve primary failure
            signal.setitimer(signal.ITIMER_REAL, 0)
            error = repr(caught)
            traceback.print_exc()
        signal.setitimer(signal.ITIMER_REAL, 0)
        finish = None
        try:
            finish = self.clock.now_ns()
        except BaseException as caught:  # noqa: BLE001 - preserve clock failure
            error = f"{error}; terminal clock: {caught!r}"
        files = {}
        try:
            files = {p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()}
        except BaseException as caught:  # noqa: BLE001 - preserve artifact publication failure
            error = f"{error}; artifact closure: {caught!r}"
        receipt = {"version": VERSION, "status": "failed" if error else "completed", "agreement": error is None,
                       "error": error, "checks": self.checks, "source": source, "inputs": self.inputs, "scope": SCOPE,
                       "wall_seconds": None if finish is None else (finish-self.start)/1e9,
                       "peak_rss_bytes": self.peak_rss, "model_forward_calls": 0, "optimizer_calls": 0, "native_calls": 0,
                       "files": files}
        try:
            write(self.out/"receipt.json", receipt)
            self.budget()
        except BaseException as caught:  # noqa: BLE001 - preserve original failure in command output
            error = f"{error}; receipt publication: {caught!r}"
            try:
                receipt_path = self.out/"receipt.json"
                invalid_path = self.out/"receipt.invalid.json"
                if receipt_path.exists():
                    if invalid_path.exists():
                        raise FileExistsError(invalid_path)
                    receipt_path.rename(invalid_path)
                write(self.out/"late-failure.json", {"status": "failed", "error": error})
                receipt.update(status="failed", agreement=False, error=error, checks=self.checks,
                               files={p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()})
                write(receipt_path, receipt)
            except BaseException as publication_error:  # noqa: BLE001 - retain both failures
                error += f"; late failure publication: {publication_error!r}"
        print(json.dumps({"status": "failed" if error else "completed", "receipt": str(self.out/"receipt.json"), "error": error}))
        return int(error is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "run", "launch", "terminal", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    for key in ("plan", "receipt", "summary", "launch", "terminal"):
        parser.add_argument("--"+key+"-sha256", required=True)
    args = parser.parse_args()
    if not all(getattr(args, key).is_absolute() for key in ("plan", "run", "launch", "terminal", "output")):
        parser.error("all paths must be absolute")
    return Audit(args).execute()


if __name__ == "__main__":
    raise SystemExit(main())
