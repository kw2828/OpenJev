"""Saved-score qualification of restricted selection, with no policy inference.

Replays eight already-qualified public prefixes and both recorded score views.
The injected fake policy returns a saved action and all four float32 costs once.
This qualifies selection/lifecycle only, not new neural execution, an autonomous
trajectory, or a counterfactual recovery from the earlier observed failure.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-restricted-policy-qualification-v1"
AUTH = "scripts/qualify_otto_reference_control.py"
CLOCK = "src/openjev/research/suspend_clock.py"
PINS = {
    AUTH: "0b8591d2c92dd1c006e06d120b167eebfd350d483d2a61e6e933d2025a59bdb5",
    CLOCK: "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124",
    "src/openjev/research/otto_restricted_policy.py": "3e63240b3502b4d4633f9c1671b4ad7ce34466f21c74a4d470bed91d77514630",
    "tests/test_otto_restricted_policy.py": "49132b1c8cf3e7891a88cb387f40f01e10afb946952a58463c129743b275c0eb",
}
LIMITS = {"native_seconds": 60, "rss_bytes": 4 * 1024**3, "output_bytes": 64 * 1024**2}
CASES = [(f"{regime}.{name}", prefix) for regime in ("base", "shift")
         for name, prefix in (("hit1", 0), ("hit2", 1), ("hit3", 4), ("boundary", 27))]
SCOPE = ("Authenticated saved costs and public-prefix replay only. Original all-four costs remain unchanged; "
         "only final selection is restricted. No original policy, TensorFlow, simulator, checkpoint load, "
         "new trajectory or recovery inference. Inherited public update implementation is unchanged.")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def descriptor(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return {"sha256": value.hexdigest(), "bytes": path.stat().st_size}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def authenticate(args):
    for name, pin in PINS.items():
        path = ROOT / name
        require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
                and descriptor(path)["sha256"] == pin, f"fixed source: {name}")
    helper = load(ROOT / AUTH, "_restricted_saved_auth_only")
    paths = helper.authenticate(args)  # Pure metadata/runtime authentication only.
    return paths, {**helper.PINS, **PINS}


def scalar_action(scores, allowed, np):
    """Independent ordered scalar float32 near-minimum, without actor helpers."""
    lowest = min(scores[a] for a in allowed)
    for action in allowed:
        difference = np.float32(scores[action] - lowest)
        if abs(difference) < np.float32(1e-10):
            return action
    raise ValueError("no finite saved-score minimum")


def comparisons(args, paths, receipt, check):
    os.environ.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")})
    import numpy as np

    sys.path.insert(0, str(ROOT / "src"))
    from openjev.research.otto_restricted_policy import RestrictedPolicyActor

    histories = {name: [] for name, _ in CASES}
    with (args.run / "transitions.jsonl").open() as stream:
        for line in stream:
            check()
            row = json.loads(line)
            require(row["case"] in histories, "fixed native qualification case")
            histories[row["case"]].append(row)
    require(sum(map(len, histories.values())) == 454, "eight complete resets and 446 public steps")
    with (args.run / "policy-checks.jsonl").open() as stream:
        checks = [json.loads(line) for line in stream]
    require([(r["case"], r["step"]) for r in checks] == CASES
            and all(r["passed"] is True and r["symmetry_average"] is True for r in checks), "all eight saved score pairs")
    kernels = {}
    for regime in ("base", "shift"):
        with np.load(paths[f"{regime}_kernel"], allow_pickle=False) as archive:
            require(set(archive.files) == {"likelihood", "initial_hit_weights"}, "qualified kernel keys")
            kernels[regime] = archive["likelihood"]

    class SavedPolicy:
        def __init__(self, *, env, model, sym_avg):
            require(sym_avg is True and model is None, "no numerical model in fake policy")
            self.calls, self.saved_action, self.saved_scores = 0, None, None

        def _value_policy(self):
            self.calls += 1
            receipt["fake_policy_calls"] += 1
            require(self.calls == 1, "one saved-score policy call per actor")
            return self.saved_action, self.saved_scores

    for (name, prefix), saved in zip(CASES, checks, strict=True):
        history = histories[name]
        require(len(history) == (212 if name.endswith("boundary") else 5), "complete fixed mechanical history")
        for step, row in enumerate(history):
            require(row["public"]["step"] == step and row["kind"] == ("reset" if step == 0 else "step"), "ordered public history")
        with np.load(args.run / f"prefix-{name}.npz", allow_pickle=False) as archive:
            expected = {f"{view}_{field}" for view in ("native", "public") for field in ("inputs", "masses", "values", "scores")}
            require(set(archive.files) == expected, "complete prior saved prefix array names")
            scores_by_view = {view: archive[f"{view}_scores"] for view in ("native", "public")}
        require(scores_by_view["native"].tobytes() == scores_by_view["public"].tobytes(), "inherited exact paired raw costs")
        for view in ("native", "public"):
            check()
            scores = scores_by_view[view]
            require(scores.shape == (4,) and scores.dtype == np.float32 and np.isfinite(scores).all()
                    and scores.tolist() == saved[f"{view}_scores"], "unchanged four finite saved float32 costs")
            original_action = saved[f"{view}_action"]
            require(type(original_action) is int and original_action == scalar_action(scores, range(4), np), "saved original all-four action")
            actor = RestrictedPolicyActor(history[0]["public"], kernels[name.split(".")[0]], None, SavedPolicy)
            receipt["public_actor_constructions"] += 1
            for row in history[1:prefix + 1]:
                actor.update(row["action"], row["public"])
                receipt["public_updates"] += 1
            probability = actor.belief
            fingerprint = hashlib.sha256(probability.tobytes()).hexdigest()
            require(fingerprint == history[prefix]["state"]["belief_sha256"], "exact saved public-prefix posterior")
            before = actor.public
            allowed = tuple(before["valid_actions"])
            expected_action = scalar_action(scores, allowed, np)
            actor._policy.saved_action, actor._policy.saved_scores = original_action, scores
            chosen, raw = actor.choose()
            agreement = (chosen == expected_action and raw.dtype == np.float32 and raw.shape == (4,)
                         and raw.tobytes() == scores.tobytes() and actor._pending_action == chosen
                         and actor._policy.calls == 1 and actor.allowed_actions == allowed
                         and actor.selection_mask == tuple(a in allowed for a in range(4))
                         and actor.public == before and actor.belief.tobytes() == probability.tobytes())
            record = {"case": name, "step": prefix, "saved_view": view, "agreement": agreement,
                      "original_action": original_action, "restricted_action": chosen,
                      "independent_restricted_action": expected_action, "raw_scores": scores.tolist(),
                      "raw_costs_unchanged": raw.tobytes() == scores.tobytes(), "allowed_actions": list(allowed),
                      "selection_mask": list(actor.selection_mask), "pending_action": actor._pending_action,
                      "fake_policy_calls": actor._policy.calls, "public_belief_sha256": fingerprint}
            with (args.output / "comparisons.jsonl").open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            receipt["comparisons"] += 1
            require(agreement, "exact saved selection/raw-score/public-state qualification")
    require(receipt["comparisons"] == receipt["fake_policy_calls"] == receipt["public_actor_constructions"] == 16
            and receipt["public_updates"] == 128, "complete sixteen independent replayed actor calls")
    require(not any(name in ("tensorflow", "tf_keras") or name.endswith((".sourcetracking", ".rlpolicy", ".valuemodel"))
                    for name in sys.modules), "no original policy, model framework or simulator import")


def execute(args):
    require(args.output.is_absolute() and not any(p.is_symlink() for p in (args.output, *args.output.parents)), "absolute nonsymlink exclusive output")
    args.output.mkdir(parents=True, exist_ok=False)
    clock = start = None
    receipt = {"version": VERSION, "status": "started", "limits": LIMITS, "scope": SCOPE,
               "request": {k: str(v) for k, v in vars(args).items()}, "sources": PINS,
               "self_source": descriptor(Path(__file__)), "comparisons": 0, "fake_policy_calls": 0,
               "public_actor_constructions": 0, "public_updates": 0,
               "model_calls": 0, "simulator_calls": 0, "native_calls": 0}
    previous = signal.getsignal(signal.SIGALRM)

    def timeout(*_):
        raise TimeoutError("saved restricted-selection qualification 60-second cap")

    def check():
        require(clock.now_ns() - start < 60 * 10**9, "suspend-inclusive qualification deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"], "qualification RSS cap")
        require(sum(p.stat().st_size for p in args.output.iterdir() if p.is_file()) <= LIMITS["output_bytes"], "qualification output cap")

    try:
        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, 60)
        require(descriptor(ROOT / CLOCK)["sha256"] == PINS[CLOCK], "fixed native clock")
        clock = load(ROOT / CLOCK, "_restricted_selection_clock").SuspendClock()
        start = clock.now_ns()
        write(args.output / "started.json", receipt)
        paths, sources = authenticate(args)
        receipt["sources"] = sources
        receipt["inputs"] = {name: {"path": str(path), **descriptor(path)} for name, path in paths.items()}
        receipt["transitions"] = descriptor(args.run / "transitions.jsonl")
        receipt["saved_policy_checks"] = descriptor(args.run / "policy-checks.jsonl")
        check()
        comparisons(args, paths, receipt, check)
        require(authenticate(args) == (paths, sources), "unchanged final source/runtime/payload bindings")
        require(descriptor(Path(__file__)) == receipt["self_source"], "unchanged qualification source")
        check()
        finished = clock.now_ns()
        receipt.update(status="completed", qualified=True, agreement=True, clock_backend=clock.backend,
                       started_ns=start, finished_ns=finished, wall_seconds=(finished - start) / 1e9,
                       files={p.name: descriptor(p) for p in args.output.iterdir() if p.is_file()})
        write(args.output / "receipt.json", receipt)
        check()
        print(json.dumps({"status": "completed", "qualified": True, "receipt_sha256": descriptor(args.output / "receipt.json")["sha256"]}))
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt.update(status="failed", qualified=False, agreement=False, error=repr(error), traceback=traceback.format_exc())
        try:
            if (args.output / "receipt.json").exists():
                (args.output / "receipt.json").rename(args.output / "receipt.invalid.json")
            write(args.output / "receipt.json", receipt)
        except BaseException as secondary:  # noqa: BLE001 - Preserve original failure.
            error.add_note(f"Failure receipt publication: {secondary!r}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "run", "terminal", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("plan-sha256", "receipt-sha256", "terminal-sha256"):
        parser.add_argument(f"--{name}", required=True)
    execute(parser.parse_args())
