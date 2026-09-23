"""Fixed twelve-fit score forecast screen; no teacher or environment calls.

Metadata admission precedes numerical imports. VALID arrays are opened only
after all final checkpoints have been saved and a durable barrier published.
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
SELF = "scripts/train_otto_score_forecasts.py"
TEST = "tests/test_train_otto_score_forecasts.py"
MODELS = "src/openjev/research/otto_recurrent_scores.py"
DATA = "src/openjev/research/otto_score_forecast_data.py"
PROTOCOL = "research/otto-score-forecast-protocol.md"
AUDIT = "scripts/audit_otto_score_forecasts.py"
AUDIT_TEST = "tests/test_audit_otto_score_forecasts.py"
CLOCK = "src/openjev/research/suspend_clock.py"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SUPERVISOR_PIN = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
VERSION = "otto-score-forecast-training-v1"
KINDS = ("residual_gru", "direct_gru", "history_mlp", "current_mlp")
SEEDS = (225001, 225002, 225003)
LIMITS = {"seconds": 600, "rss_bytes": 4 * 1024**3, "output_bytes": 2 * 1024**3}
CONFIG = {"families": list(KINDS), "fit_seeds": list(SEEDS), "epochs": 80,
          "batch_windows": 32, "learning_rate": .003, "gradient_clip": 5.,
          "weight_decay": 0., "scale": 64., "training_episodes": 54,
          "validation_episodes": 36, "required_conditions": 45,
          "checkpoint": "last", "device": "cpu", "dtype": "float32"}
NEW = {SELF, TEST, MODELS, DATA, PROTOCOL, AUDIT, AUDIT_TEST,
       "tests/test_otto_recurrent_scores.py", "tests/test_otto_score_forecast_data.py"}
ROLES = ("collection_plan", "collection_receipt", "collection_terminal", "engineering")
ARRAYS = {"features", "raw_q", "legal", "actions", "correction", "episode_offsets"}


def require(ok, message):
    if not ok:
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
        f.write("\n"); f.flush(); os.fsync(f.fileno())


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


def collection_inputs(inputs):
    require(set(inputs) == set(ROLES), "exact training inputs")
    for d in inputs.values():
        require(descriptor(d["path"]) == {k: d[k] for k in ("sha256", "bytes")}, "input hash")
    plan = read(inputs["collection_plan"]["path"])
    receipt = read(inputs["collection_receipt"]["path"])
    parent = read(inputs["collection_terminal"]["path"])
    require(receipt["status"] == "completed" and receipt["plan_sha256"] == inputs["collection_plan"]["sha256"]
            and receipt["sources"] == plan["sources"], "complete original collection")
    require(parent["status"] == "completed" and parent["returncode"] == 0 and not parent["timed_out"]
            and parent["group_absent"] and parent["cleanup"]["reaped"]
            and parent["cleanup"]["errors"] == [] and parent["error"] is parent["clock_error"] is None
            and parent["cap_seconds"] == 900 and parent["clock_source_sha256"] == CLOCK_PIN
            and parent["watchdog_sha256"] == SUPERVISOR_PIN
            and parent["started_ns"] <= receipt["started_ns"] < receipt["finished_ns"]
            <= parent["finished_ns"] <= parent["deadline_ns"], "original successful collection supervisor")
    command = list(parent["command"])
    if command[1:2] == ["-u"]:
        command.pop(1)
    require(command[:3] == [str(ROOT / ".venv-otto-released-native/bin/python"),
            str(ROOT / "scripts/collect_otto_score_forecasts.py"), "run"], "original native collection command")
    opts = dict(zip(command[3::2], command[4::2], strict=True))
    directory = regular(inputs["collection_receipt"]["path"]).parent
    require(set(opts) == {"--plan", "--plan-sha256", "--supervision", "--output"}
            and opts["--plan"] == str(regular(inputs["collection_plan"]["path"]))
            and opts["--plan-sha256"] == inputs["collection_plan"]["sha256"]
            and opts["--output"] == str(directory), "original collection input/output join")
    launch = read(opts["--supervision"])
    require(descriptor(opts["--supervision"])["sha256"] == receipt["supervision_sha256"]
            and all(parent[k] == v for k, v in launch.items()), "original collection launch")
    require({p.name for p in directory.iterdir()} == set(receipt["files"]) | {"receipt.json"}, "closed collection inventory")
    for name, d in receipt["files"].items():
        require(Path(name).name == name and descriptor(directory / name) == d, "saved collection payload")
    for name, pin in plan["sources"].items():
        require(descriptor(name)["sha256"] == pin, "unchanged collection source")
    engineering = read(inputs["engineering"]["path"])
    require(engineering["status"] == "passed" and engineering["sources_before"] == engineering["sources_after"]
            and all(x["exit_code"] == 0 for x in engineering["results"]), "qualified forecast components")
    for name, pin in engineering["sources_after"].items():
        require(descriptor(name)["sha256"] == pin, "qualified component unchanged")
    return plan, receipt, directory


def freeze(args):
    inputs = {}
    for role in ROLES:
        p = regular(getattr(args, role)); d = descriptor(p)
        require(d["sha256"] == getattr(args, role + "_sha256"), "external planning pin")
        inputs[role] = {"path": str(p), **d}
    old, _, _ = collection_inputs(inputs)
    sources = dict(old["sources"])
    for name in NEW:
        pin = descriptor(name)["sha256"]
        require(name not in sources or sources[name] == pin, "no replacement frozen source")
        sources[name] = pin
    value = {"version": VERSION, "status": "frozen_before_fitting", "config": CONFIG,
             "limits": LIMITS, "inputs": inputs, "sources": sources, "runtime": runtime_record()}
    write(args.output, value)
    print(json.dumps({"status": value["status"], "plan": descriptor(args.output)}), flush=True)


def criteria(models, hold, support):
    result = []
    def add(name, value, relation, threshold):
        require(math.isfinite(value) and math.isfinite(threshold), "finite criterion")
        result.append({"name": name, "value": value, "relation": relation, "threshold": threshold,
                       "passes": value >= threshold if relation == ">=" else value <= threshold})
    by = {(r["family"], r["seed"]): r["metrics"] for r in models}
    require(len(models) == len(by) == 12 and set(by) == {(k, seed) for k in KINDS for seed in SEEDS}, "all fixed fits")
    add("technical_complete_requires_closed_saved_audit", 1, ">=", 1)
    for regime in ("lambda3", "lambda4"):
        for age in ("1", "2", "3"):
            add(f"{regime}.age{age}.case_support", support[regime][age], ">=", 4)
        held = hold["by_regime"][regime]
        for seed in SEEDS:
            candidate = by["residual_gru", seed]["by_regime"][regime]
            add(f"{regime}.{seed}.agreement_vs_hold", candidate["episode_weighted_agreement"], ">=", held["episode_weighted_agreement"])
            add(f"{regime}.{seed}.gap_vs_hold", candidate["episode_weighted_raw_gap"], "<=", .8 * held["episode_weighted_raw_gap"])
            for age in ("1", "2", "3"):
                add(f"{regime}.{seed}.age{age}.gap_vs_hold", candidate["by_age"][age]["episode_weighted_raw_gap"],
                    "<=", held["by_age"][age]["episode_weighted_raw_gap"])
        for control in ("history_mlp", "direct_gru"):
            for key, factor, relation in (("episode_weighted_agreement", 1., ">="), ("episode_weighted_raw_gap", .9, "<=")):
                cand = math.fsum(by["residual_gru", seed]["by_regime"][regime][key] for seed in SEEDS) / 3
                other = math.fsum(by[control, seed]["by_regime"][regime][key] for seed in SEEDS) / 3
                add(f"{regime}.mean.{key}_vs_{control}", cand, relation, factor * other)
    require(len(result) == 45, "all forty-five fixed conditions")
    return result


def centered_loss(torch, prediction, targets, legal):
    """Per-row eligible-centered MSE in teacher units divided by64."""
    count = legal.sum(-1).clamp(min=1)
    difference = (prediction - targets) / 64.
    mean = (difference * legal).sum(-1) / count
    return (((difference - mean[..., None]) ** 2) * legal).sum(-1) / count


def batch_loss(torch, model, prediction, targets, legal, weights, scale):
    # Retained query-only windows have no model-dependent outputs. Explicit zero
    # gradients preserve the declared Adam step and exposure for these batches.
    zero = sum(parameter.sum() * 0. for parameter in model.parameters())
    return (centered_loss(torch, prediction, targets, legal) * weights).sum() * scale + zero


def metric_report(data, np, windows, episodes, identities, predictions):
    """Reweight each collector separately using the same saved predictions."""
    require(len(episodes) == len(identities) and all(e["id"] == i["episode_id"]
            for e, i in zip(episodes, identities, strict=True)), "ordered collector identities")
    report = data.forecast_metrics(windows, predictions)
    report["by_collector"] = {}
    for arm in ("analytic", "neural", "period4_hold"):
        selected = [index for index, row in enumerate(identities) if row["arm"] == arm]
        require(bool(selected), "every collector has episodes")
        subset = data.build_windows([episodes[index] for index in selected])
        mask = np.isin(windows["episode_index"], selected)
        report["by_collector"][arm] = data.forecast_metrics(subset, predictions[mask])
    return report


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.receipt = {"version": VERSION, "status": "started", "teacher_calls": 0, "native_calls": 0,
                        "fits_completed": 0, "optimizer_steps": 0, "pending": None}
        self.start = self.clock = self.launch = None

    def check(self):
        require(self.clock.now_ns() < self.launch["deadline_ns"], "original forecast allocation deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"] and sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
                < LIMITS["output_bytes"] - 1024**2, "forecast RSS/output cap")

    def event(self, value):
        self.check()
        with (self.out / "progress.jsonl").open("a") as f:
            f.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
            f.flush(); os.fsync(f.fileno())

    def admit(self):
        require(descriptor(CLOCK)["sha256"] == CLOCK_PIN, "original clock")
        self.clock = load(ROOT / CLOCK, "_score_fit_clock").SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, "original launch available")
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        cmd = list(self.launch["command"])
        if cmd[1:2] == ["-u"]:
            cmd.pop(1)
        require(cmd == [sys.executable, *sys.argv] and self.launch["pid"] == os.getpid()
                and self.launch["pgid"] == os.getpgrp() and self.launch["parent_pid"] == os.getppid()
                and self.launch["cap_seconds"] == 600 and self.launch["watchdog_sha256"] == SUPERVISOR_PIN
                and self.launch["clock_source_sha256"] == CLOCK_PIN and Path.cwd() == ROOT
                and self.launch["deadline_ns"] == self.launch["started_ns"] + 600 * 10**9,
                "original bounded fitting process")
        require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256, "external training plan")
        self.plan = read(self.args.plan)
        require(self.plan["version"] == VERSION and self.plan["status"] == "frozen_before_fitting"
                and self.plan["config"] == CONFIG and self.plan["limits"] == LIMITS
                and self.plan["runtime"] == runtime_record(), "fixed fit protocol and runtime")
        for name, pin in self.plan["sources"].items():
            require(descriptor(name)["sha256"] == pin, "frozen training source")
        self.collection_plan, self.collection_receipt, self.collection_dir = collection_inputs(self.plan["inputs"])
        require(self.plan["sources"] == {**self.collection_plan["sources"],
                **{name: descriptor(name)["sha256"] for name in NEW}}, "full original and new source closure")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            require(os.environ.get(name) == "1", "one numerical CPU thread")
        self.receipt.update(plan_sha256=self.args.plan_sha256, sources=self.plan["sources"], inputs=self.plan["inputs"],
                            supervision_sha256=descriptor(self.args.supervision)["sha256"], limits=LIMITS)
        write(self.out / "started.json", {"launch": self.launch, "started_ns": self.start})

    def episodes(self, stage):
        """Decode only the requested stage after its explicit lifecycle barrier."""
        require(stage in ("train", "valid"), "fixed episode stage")
        if stage == "valid":
            require(self.receipt["fits_completed"] == 12 and self.valid_allowed, "all final checkpoints before VALID decoding")
        np = self.np
        with np.load(self.collection_dir / f"{stage}.npz", allow_pickle=False) as z:
            require(set(z.files) == ARRAYS, "exact flat collection arrays")
            a = {k: z[k] for k in z.files}
        identities = [r for r in self.collection_plan["cohort"] if r["stage"] == stage]
        offsets = a["episode_offsets"]
        require(offsets.dtype == np.int64 and offsets.shape == (len(identities) + 1,)
                and offsets[0] == 0 and offsets[-1] == len(a["features"])
                and (np.diff(offsets) >= 1).all() and (np.diff(offsets) <= 2188).all(), "full ordered episode offsets")
        n = int(offsets[-1])
        require(a["actions"].dtype == np.int64 and a["actions"].shape == (n,)
                and a["correction"].dtype == np.bool_ and a["correction"].shape == (n,), "action and correction arrays")
        result = []
        for index, identity in enumerate(identities):
            lo, hi = int(offsets[index]), int(offsets[index + 1])
            action = a["actions"][lo:hi]
            require(((action >= 0) & (action < 4)).all()
                    and np.array_equal(a["correction"][lo:hi], np.arange(hi - lo) % 4 == 0)
                    and a["legal"][lo:hi][np.arange(hi - lo), action].all(), "legal actions and fixed virtual correction schedule")
            result.append({"id": identity["episode_id"], "regime": identity["regime"], "split": stage,
                           "features": a["features"][lo:hi], "teacher_scores": a["raw_q"][lo:hi],
                           "legal": a["legal"][lo:hi]})
        require(len(result) == (54 if stage == "train" else 36), "complete declared stage")
        return result, identities

    def npz(self, name, arrays):
        self.check()
        with (self.out / name).open("xb") as f:
            self.np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())

    def tensors(self, windows):
        torch, np = self.torch, self.np
        result = {key: torch.from_numpy(np.array(windows[key], copy=True))
                for key in ("features", "query_scores", "lengths", "targets", "legal", "nonquery_weights")}
        result["nonquery_weights"] = result["nonquery_weights"].to(torch.float32)
        return result

    def predict(self, model, tensors):
        torch, np = self.torch, self.np
        outputs = []
        with torch.no_grad():
            for start in range(0, len(tensors["lengths"]), 32):
                self.check(); sl = slice(start, start + 32)
                outputs.append(model(tensors["features"][sl], tensors["query_scores"][sl], tensors["lengths"][sl]).numpy())
        return np.concatenate(outputs)

    def fit(self, kind, seed, windows, tensors):
        np, torch = self.np, self.torch
        tick = time.perf_counter(); model = self.models.make_head(kind, seed)
        initial = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            initial.update(name.encode()); initial.update(tensor.detach().numpy().tobytes())
        optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=0.)
        rng = np.random.default_rng(seed)
        count, steps, exposure = len(windows["lengths"]), 0, 0
        permutation = hashlib.sha256()
        for epoch in range(80):
            order = rng.permutation(count).astype(np.int64)
            permutation.update(order.tobytes())
            for start in range(0, count, 32):
                self.check(); index = order[start:start + 32]
                self.receipt["pending"] = {"family": kind, "seed": seed, "epoch": epoch, "batch": start // 32}
                optimizer.zero_grad(set_to_none=True)
                prediction = model(tensors["features"][index], tensors["query_scores"][index], tensors["lengths"][index])
                loss = batch_loss(torch, model, prediction, tensors["targets"][index],
                                  tensors["legal"][index], tensors["nonquery_weights"][index], count / len(index))
                require(bool(torch.isfinite(loss)), "finite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                require(bool(torch.isfinite(norm)), "finite gradient norm")
                optimizer.step()
                require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "finite fitted parameters")
                steps += 1; exposure += len(index); self.receipt["optimizer_steps"] += 1
                self.receipt["pending"] = None
            if (epoch + 1) % 20 == 0:
                self.event({"event": "epoch", "family": kind, "seed": seed, "epoch": epoch + 1,
                            "optimizer_steps": steps, "window_exposures": exposure})
        pred = self.predict(model, tensors)
        with torch.no_grad():
            final_loss = float((centered_loss(torch, torch.from_numpy(pred), tensors["targets"], tensors["legal"])
                                * tensors["nonquery_weights"]).sum())
        name = f"{kind}-{seed}.npz"
        self.npz(name, {k: v.detach().numpy() for k, v in model.state_dict().items()})
        fit = {"family": kind, "seed": seed, "parameter_count": self.models.parameter_count(kind),
               "checkpoint_path": name, "checkpoint": descriptor(self.out / name), "initial_sha256": initial.hexdigest(),
               "epochs": 80, "steps": steps, "window_exposures": exposure, "windows_per_epoch": count,
               "permutation_sha256": permutation.hexdigest(), "final_train_loss": final_loss,
               "wall_seconds": time.perf_counter() - tick}
        self.receipt["fits_completed"] += 1
        self.event({"event": "fit_complete", **fit})
        return model, fit

    def body(self):
        tick = time.perf_counter()
        import numpy as np
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.np, self.torch = np, torch
        self.models = load(ROOT / MODELS, "_forecast_models")
        self.data = load(ROOT / DATA, "_forecast_data")
        require(self.models.KINDS == KINDS and torch.get_num_threads() == 1, "fixed CPU model families")
        write(self.out / "runtime.json", {**runtime_record(), "torch": torch.__version__, "numpy": np.__version__,
              "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
              "deterministic": torch.are_deterministic_algorithms_enabled(), "cuda_used": False, "mps_used": False})
        self.valid_allowed = False
        episodes, _ = self.episodes("train")
        train = self.data.build_windows(episodes)
        train_t = self.tensors(train)
        setup_seconds = time.perf_counter() - tick
        fits, fitted = [], []
        tick = time.perf_counter()
        for seed in SEEDS:
            for kind in KINDS:
                self.event({"event": "fit_start", "family": kind, "seed": seed})
                model, fit = self.fit(kind, seed, train, train_t)
                fits.append(fit); fitted.append((model, fit))
        fit_seconds = time.perf_counter() - tick
        write(self.out / "fits.json", {"fits": fits, "train_counts": train["counts"]})
        for seed in SEEDS:
            subset = [r for r in fits if r["seed"] == seed]
            require(len({r["permutation_sha256"] for r in subset}) == 1, "same paired window exposure")
            require(subset[0]["initial_sha256"] == subset[1]["initial_sha256"], "identical GRU initialization")
        self.event({"event": "all_checkpoints_closed_before_VALID", "fits_completed": 12,
                    "checkpoints": {r["checkpoint_path"]: r["checkpoint"] for r in fits}})
        self.valid_allowed = True
        tick = time.perf_counter()
        episodes, identities = self.episodes("valid")
        valid = self.data.build_windows(episodes)
        valid_t = self.tensors(valid)
        self.npz("validation-windows.npz", {k: v for k, v in valid.items() if isinstance(v, np.ndarray)})
        write(self.out / "validation-windows.json", {k: v for k, v in valid.items() if not isinstance(v, np.ndarray)})
        held = np.repeat(valid["query_scores"][:, None, :], 4, axis=1)
        held[~valid["valid_mask"]] = 0.
        self.npz("prediction-hold.npz", {"predictions": held})
        hold = metric_report(self.data, np, valid, episodes, identities, held)
        records = []
        for model, fit in fitted:
            self.check()
            prediction = self.predict(model, valid_t)
            self.npz(f"prediction-{fit['family']}-{fit['seed']}.npz", {"predictions": prediction})
            records.append({"family": fit["family"], "seed": fit["seed"],
                            "metrics": metric_report(self.data, np, valid, episodes, identities, prediction)})
        support = {regime: {str(age): len({identity["case"] for identity, episode in zip(identities, episodes, strict=True)
                            if identity["regime"] == regime and len(episode["features"]) > age})
                            for age in (1, 2, 3)} for regime in ("lambda3", "lambda4")}
        conditions = criteria(records, hold, support)
        summary = {"version": VERSION, "models": records, "hold": hold, "support": support,
                   "required": conditions, "required_passed": sum(c["passes"] for c in conditions),
                   "required_conditions": 45, "forecast_continuation": all(c["passes"] for c in conditions),
                   "requires_successful_original_supervisor_and_saved_audit": True,
                   "train_counts": train["counts"], "validation_counts": valid["counts"],
                   "setup_seconds": setup_seconds, "fitting_seconds": fit_seconds,
                   "validation_seconds": time.perf_counter() - tick,
                   "scope": "Forced-path forecasts on fresh VALID, not autonomous performance or true action regret."}
        write(self.out / "summary.json", summary)

    def execute(self):
        self.out.mkdir(exist_ok=False)
        def interrupted(_signum, _frame):
            raise InterruptedError("original forecast supervisor stopped worker")
        signal.signal(signal.SIGTERM, interrupted)
        try:
            self.admit(); self.body()
            for n, pin in self.plan["sources"].items():
                self.check(); require(descriptor(n)["sha256"] == pin, "unchanged source after fitting")
            for d in self.plan["inputs"].values():
                require(descriptor(d["path"]) == {k: d[k] for k in ("sha256", "bytes")}, "unchanged input after fitting")
            for n, d in self.collection_receipt["files"].items():
                require(descriptor(self.collection_dir / n) == d, "unchanged collection payload")
            require(self.receipt["fits_completed"] == 12 and self.receipt["pending"] is None, "complete fixed fitting")
            self.check(); finished = self.clock.now_ns()
            self.receipt.update(status="completed", complete=True,
                files={p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()},
                started_ns=self.start, finished_ns=finished, wall_seconds=(finished-self.start)/1e9,
                requires_successful_original_supervisor=True)
            write(self.out / "receipt.json", self.receipt); self.check()
            print(json.dumps({"status": "completed", "receipt": descriptor(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            try:
                self.receipt.update(status="failed", complete=False, error=repr(error), traceback=traceback.format_exc())
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as publication_error:  # noqa: BLE001 - preserve the original failure
                error.add_note("Failure receipt publication also failed: " + repr(publication_error))
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
