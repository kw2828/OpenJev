"""Native-free exact heuristic comparison on eight closed public prefixes.

Imports original Policy/HeuristicPolicy only, and compiles the pinned original
_entropy/_distance method ASTs into a public numerical view. No SourceTracking
class or TensorFlow import, simulator construction, sampling or model forward.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-reference-control-qualification-v1"
UPSTREAM = "tmp/otto-source-review-01/isotropic/classes"
CLOCK = "src/openjev/research/suspend_clock.py"
RUNNER = "scripts/qualify_otto_released_native.py"
PINS = {
    CLOCK: "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124",
    RUNNER: "96c68e94498c6b83eb2983c1183c444c5d1080eee06ff961562aeb56a845c65e",
    "src/openjev/research/otto_released_policy.py": "8abac34fef2a6517d43209d5ca27514633363248f66a249e2f4d6615aa0a0d45",
    "tests/test_otto_released_policy.py": "cc2e7e543a8e8b7139dbcb974680e8af0213e81cbd089b475b14710dc8f8b6b7",
    "src/openjev/research/otto_reference_control.py": "21e5f6ece09426d26c4a3214d95f0f4803a8efd4bacb9ca347c1d896407c5893",
    "tests/test_otto_reference_control.py": "e989a18ad56f80bb412b62570bba649978ea171eccfdb2b8c404566a543ab37c",
    f"{UPSTREAM}/policy.py": "8fc915c86972a8be0ccd0a98d9522be4cd9c751d862a351d60221f8c7902976d",
    f"{UPSTREAM}/heuristicpolicy.py": "e473c5c0e2908622e704012fc79dffc05172fe8b6f4036dcbf240a05c4e4b469",
    f"{UPSTREAM}/sourcetracking.py": "1057f129fa8a3249a7297a9ef2c15059d37430b12e42a2f1d43dfcdffe3e76f1",
}
LIMITS = {"native_seconds": 60, "rss_bytes": 4 * 1024**3, "output_bytes": 64 * 1024**2}
CASES = [(f"{regime}.{name}", prefix) for regime in ("base", "shift")
         for name, prefix in (("hit1", 0), ("hit2", 1), ("hit3", 4), ("boundary", 27))]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


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
    for path in (args.plan, args.run, args.terminal):
        require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), "regular absolute input path")
    for path, pin in ((args.plan, args.plan_sha256), (args.run / "receipt.json", args.receipt_sha256),
                      (args.terminal, args.terminal_sha256)):
        require(digest(path)["sha256"] == pin, "external source-result pin")
    for name, pin in PINS.items():
        require(digest(ROOT / name)["sha256"] == pin, f"fixed source: {name}")
    helper = load(ROOT / RUNNER, "_otto_reference_qualification_auth")
    plan, paths = helper.authenticate(args)  # stdlib-only transitive lineage/runtime authentication.
    done, terminal = read(args.run / "receipt.json"), read(args.terminal)
    require(done["version"] == plan["version"] and done["status"] == "completed" and done["qualified"] is True
            and done["plan_sha256"] == args.plan_sha256 and done["sources"] == plan["sources"]
            and done["inputs"] == plan["inputs"] and done["limits"] == plan["limits"], "closed qualified native result")
    expected = {"started.json", "runtime.json", "weights.jsonl", "work.jsonl", "transitions.jsonl",
                "cases.jsonl", "policy-checks.jsonl", "summary.json", *(f"prefix-{name}.npz" for name, _ in CASES)}
    require(set(done["files"]) == expected and {p.name for p in args.run.iterdir()} == expected | {"receipt.json"},
            "complete undemoted source qualification")
    for name, record in done["files"].items():
        path = args.run / name
        require(path.is_file() and not path.is_symlink() and digest(path) == record, "source result payload bytes")
    require(done["peak_rss_bytes"] <= plan["limits"]["rss_bytes"]
            and sum(p.stat().st_size for p in args.run.iterdir()) <= plan["limits"]["output_bytes"], "source resource caps")
    started = read(args.run / "started.json")
    launch, request = started["launch"], started["request"]
    launch_path = Path(request["supervision"])
    require(request["plan"] == str(args.plan) and request["plan_sha256"] == args.plan_sha256
            and request["output"] == str(args.run) and read(launch_path) == launch
            and digest(launch_path)["sha256"] == done["supervision_sha256"], "source launch binding")
    require(terminal["status"] == "completed" and terminal["returncode"] == 0 and terminal["timed_out"] is False
            and terminal["group_absent"] is True and terminal["error"] is None and terminal["clock_error"] is None
            and terminal["timing_available"] is True and terminal["cleanup"]["reaped"] is True
            and terminal["cleanup"]["group_absent"] is True and terminal["cleanup"]["errors"] == [], "source parent success")
    for key in ("pid", "pgid", "parent_pid", "command", "cwd", "started_ns", "deadline_ns", "clock_backend",
                "cap_seconds", "watchdog_sha256", "clock_source_sha256"):
        require(launch[key] == terminal[key], "source parent launch/terminal identity")
    command = list(terminal["command"])
    if command[1:2] == ["-u"]:
        command.pop(1)
    require(command[:2] == [plan["runtime"]["python_executable"], str(ROOT / RUNNER)] and len(command) == 10
            and dict(zip(command[2::2], command[3::2], strict=True)) == {
                "--plan": str(args.plan), "--plan-sha256": args.plan_sha256,
                "--supervision": str(launch_path), "--output": str(args.run)}, "actual source command")
    require(terminal["cap_seconds"] == 600 and terminal["clock_backend"] == done["clock_backend"]
            and terminal["deadline_ns"] == terminal["started_ns"] + 600 * 10**9
            and terminal["started_ns"] <= done["started_ns"] <= done["finished_ns"] <= terminal["finished_ns"] < terminal["deadline_ns"]
            and terminal["elapsed_ns"] == terminal["finished_ns"] - terminal["started_ns"]
            and terminal["wall_seconds"] == terminal["elapsed_ns"] / 1e9
            and done["wall_seconds"] == (done["finished_ns"] - done["started_ns"]) / 1e9, "source timing enclosure")
    require(terminal["watchdog_sha256"] == plan["sources"]["scripts/supervise_dialogue_observation_v2.py"]
            and terminal["clock_source_sha256"] == PINS[CLOCK] and Path(terminal["cwd"]).resolve() == ROOT,
            "source supervisor provenance")
    return paths


def original_view_class(np):
    source = ROOT / UPSTREAM / "sourcetracking.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SourceTracking")
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in ("_entropy", "_distance")]
    require(len(methods) == 2, "exact original pure method inventory")
    namespace = {"np": np, "EPSILON": 1e-10}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102 - Two hash-authenticated original methods only.

    class OriginalPublicView:
        N, Ndim, Nhits, Nactions = 53, 2, 4, 4
        _entropy, _distance = namespace["_entropy"], namespace["_distance"]

        def __init__(self, qualified_view, all_actions):
            self._public, self._all_actions = qualified_view, all_actions

        @property
        def p_source(self):
            return self._public.p_source

        @property
        def p_Poisson(self):
            return self._public.p_Poisson

        @property
        def agent(self):
            return self._public.agent

        def _move(self, action, agent):
            position, possible = self._public._move(action, agent)
            return position, True if self._all_actions else possible

        def _extract_N_from_2N(self, input, origin):
            return self._public._extract_N_from_2N(input, origin)

    return OriginalPublicView


def comparisons(args, paths, check, receipt):
    os.environ.update({n: "1" for n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")})
    import numpy as np

    sys.path.insert(0, str(ROOT / "src"))
    from openjev.research.otto_reference_control import SpaceAwareActor

    package_name = "_otto_reference_heuristic_upstream"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / UPSTREAM)]
    sys.modules[package_name] = package
    load(ROOT / UPSTREAM / "policy.py", f"{package_name}.policy")
    Policy = load(ROOT / UPSTREAM / "heuristicpolicy.py", f"{package_name}.heuristicpolicy").HeuristicPolicy
    View = original_view_class(np)
    histories = {name: [] for name, _ in CASES}
    with (args.run / "transitions.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            require(row["case"] in histories, "known prior case")
            histories[row["case"]].append(row)
    require(sum(map(len, histories.values())) == 454, "complete eight reset/446-step public trace")
    kernels = {}
    for regime in ("base", "shift"):
        with np.load(paths[f"{regime}_kernel"], allow_pickle=False) as saved:
            require(set(saved.files) == {"likelihood", "initial_hit_weights"}, "exact saved kernel keys")
            kernels[regime] = saved["likelihood"]
    for name, prefix in CASES:
        history = histories[name]
        require(history[0]["kind"] == "reset" and len(history) == (212 if name.endswith("boundary") else 5),
                "fixed mechanical history length")
        for step, row in enumerate(history):
            require(row["public"]["step"] == step and row["kind"] == ("reset" if step == 0 else "step"), "ordered public history")
        for allow_stay in (False, True):
            check()
            actor = SpaceAwareActor(history[0]["public"], kernels[name.split(".")[0]], allow_stay=allow_stay)
            for row in history[1:prefix + 1]:
                actor.update(row["action"], row["public"])
            p = actor.belief
            require(hashlib.sha256(p.tobytes()).hexdigest() == history[prefix]["state"]["belief_sha256"],
                    "exact previously qualified public prefix belief")
            original = Policy(View(actor._view, allow_stay), policy=1)
            expected_action, expected_scores = original._space_aware_infotaxis()
            receipt["original_heuristic_calls"] += 1
            actual_action, actual_scores = actor.choose()
            receipt["new_analytic_calls"] += 1
            same = (expected_scores.shape == actual_scores.shape == (4,)
                    and expected_scores.dtype == actual_scores.dtype == np.float64
                    and not np.isnan(expected_scores).any() and not np.isnan(actual_scores).any()
                    and expected_scores.tobytes() == actual_scores.tobytes() and actual_action == int(expected_action))
            record = {"case": name, "step": prefix, "allow_stay": allow_stay,
                      "allowed_actions": list(actor.allowed_actions), "exact_scores_and_action": bool(same),
                      "expected_action": int(expected_action), "actual_action": actual_action,
                      "expected_scores": [float(v) if np.isfinite(v) else None for v in expected_scores],
                      "actual_scores": [float(v) if np.isfinite(v) else None for v in actual_scores],
                      "blocked_score_encoding": "null represents positive infinity in in-bounds mode only"}
            with (args.output / "comparisons.jsonl").open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            require(same, "exact original heuristic arithmetic/action mismatch")
            check()
    require(receipt["original_heuristic_calls"] == receipt["new_analytic_calls"] == 16, "eight prefixes times two action variants")
    require(not any(name == "tensorflow" or name == "tf_keras" or name.endswith(".sourcetracking")
                    for name in sys.modules), "no model framework or simulator-module import")


def execute(args):
    require(args.output.is_absolute(), "absolute exclusive output")
    args.output.mkdir(parents=True, exist_ok=False)
    clock, start = None, None
    receipt = {"version": VERSION, "status": "started", "limits": LIMITS, "sources": PINS,
               "self_source": digest(Path(__file__)), "request": {k: str(v) for k, v in vars(args).items()},
               "native_calls": 0, "simulator_calls": 0, "model_calls": 0,
               "original_heuristic_calls": 0, "new_analytic_calls": 0,
               "scope": "Exact pure heuristic arithmetic on closed public prefixes; no native or learned-policy efficacy claim."}
    write(args.output / "started.json", receipt)

    def timeout(_signum, _frame):
        raise TimeoutError("reference control qualification 60-second alarm")

    def check():
        require(clock.now_ns() - start < 60 * 10**9, "suspend-inclusive qualification deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        require(rss <= LIMITS["rss_bytes"], "qualification RSS limit")
        receipt["peak_rss_bytes"] = rss
        require(sum(p.stat().st_size for p in args.output.iterdir() if p.is_file()) <= LIMITS["output_bytes"], "qualification output limit")

    old_alarm = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        require(digest(ROOT / CLOCK)["sha256"] == PINS[CLOCK], "fixed clock before import")
        clock = load(ROOT / CLOCK, "_otto_reference_qualification_clock").SuspendClock()
        start = clock.now_ns()
        paths = authenticate(args)
        receipt["inputs"] = {name: {"path": str(path), **digest(path)} for name, path in paths.items()}
        receipt["transitions"] = digest(args.run / "transitions.jsonl")
        check()
        comparisons(args, paths, check, receipt)
        require(authenticate(args) == paths, "unchanged end source/result/runtime bindings")
        check()
        finish = clock.now_ns()
        receipt.update(status="completed", qualified=True, agreement=True, clock_backend=clock.backend,
                       started_ns=start, finished_ns=finish, wall_seconds=(finish - start) / 1e9,
                       files={p.name: digest(p) for p in args.output.iterdir() if p.is_file()})
        write(args.output / "receipt.json", receipt)
        check()
        print(json.dumps({"status": "completed", "qualified": True, "receipt_sha256": digest(args.output / "receipt.json")["sha256"]}))
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt.update(status="failed", qualified=False, agreement=False, error=repr(error), traceback=traceback.format_exc())
        try:
            if (args.output / "receipt.json").exists():
                (args.output / "receipt.json").rename(args.output / "receipt.invalid.json")
            write(args.output / "receipt.json", receipt)
        except BaseException as secondary:  # noqa: BLE001 - retain the original failure.
            error.add_note(f"Failed to preserve qualification failure: {secondary!r}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "run", "terminal", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("plan-sha256", "receipt-sha256", "terminal-sha256"):
        parser.add_argument(f"--{name}", required=True)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
