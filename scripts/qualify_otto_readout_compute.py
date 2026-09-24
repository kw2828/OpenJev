"""Fabricated checks, bounded capacity and original-runtime metadata preflight."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/qualify_otto_readout_compute.py"


def load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, ROOT / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def native_preflight(output):
    bridge = load("scripts/otto_readout_compute_native_bridge.py", "_compute_preflight_bridge")
    bridge.require(Path.cwd() == ROOT and Path(sys.executable).absolute() == ROOT / bridge.INTERPRETER,
                   "original native interpreter")
    with bridge.metadata_imports_only():
        collector = load("scripts/collect_otto_readout_compute.py", "_compute_preflight_collector")
        prior_plan = bridge.read("output/otto-residual-estimator-v1/dev-collection-plan-01.json")
        inherited = prior_plan["inputs"]["collection_plan"]
        bridge.require(bridge.descriptor(inherited["path"])["sha256"] == collector.COLLECTION_PLAN_PIN,
                       "original native collection plan")
        old = load(collector.COLLECTOR, "_compute_preflight_original")
        plan, paths, _, _ = old.authenticate(SimpleNamespace(
            plan=ROOT / inherited["path"], plan_sha256=collector.COLLECTION_PLAN_PIN))
        sources = {}
        collector.authenticate_seed_review(ROOT / collector.SEED_REVIEW, sources)
        bridge.require(len(collector.cohort()) == 36, "complete fresh development roster")
        bridge.write(output, {"status": "passed", "native_runtime": plan["runtime"],
            "native_inputs": {k: bridge.record(p) for k, p in paths.items()},
            "seed_sources": sources, "counts": dict(bridge.ZERO_COUNTS), "metadata_import_guard": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-preflight", action="store_true")
    args = parser.parse_args()
    if args.native_preflight:
        native_preflight(args.output)
        return
    runner = load("scripts/run_otto_readout_compute.py", "_compute_qualification_runner")
    runner.require(Path.cwd() == ROOT and args.output.is_absolute() and args.output.is_relative_to(ROOT),
                   "contained qualification output")
    args.output.mkdir(exist_ok=False)
    auth = runner.authenticate_prior()
    before = {name: {k: runner.descriptor(name)[k] for k in ("sha256", "bytes")}
              for name in auth["sources"]}
    tests = sorted(name for name in runner.NEW if name.startswith("tests/"))
    lint = sorted(name for name in runner.NEW if name.endswith(".py"))
    commands = [(180, [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "-p", "no:cacheprovider",
                      "--basetemp", str(args.output / "pytest-temp"), *tests]),
                (60, [str(ROOT / ".venv/bin/ruff"), "check", "--no-cache", *lint]),
                (240, [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/qualify_otto_readout_compute_capacity.py"),
                       "--output", str(args.output / "capacity.json")]),
                (60, [str(ROOT / ".venv-otto-released-native/bin/python"), str(ROOT / SELF),
                      "--native-preflight", "--output", str(args.output / "native-preflight.json")])]
    env = {**os.environ, **{k: "1" for k in (*runner.THREADS, "PYTHONDONTWRITEBYTECODE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD")}}
    runner.write(args.output / "started.json", {"source_before": before, "commands": commands,
                                              "scope": "fabricated data and metadata only", "created_unix_ns": time.time_ns()})
    outcomes = []
    for i, (cap, command) in enumerate(commands):
        log = args.output / f"command-{i + 1}.log"
        started, timed_out = time.monotonic(), False
        with log.open("xb") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait(timeout=5)
        try:
            os.killpg(child.pid, 0)
            absent = False
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            absent = True
        outcomes.append({"command": command, "cap_seconds": cap, "returncode": code, "timed_out": timed_out,
                         "reaped": child.poll() is not None, "group_absent": absent,
                         "elapsed_seconds": time.monotonic() - started, "log": log.name,
                         **{k: runner.descriptor(log)[k] for k in ("sha256", "bytes")}})
        if code != 0 or timed_out or not absent:
            break
    after = {name: {k: runner.descriptor(name)[k] for k in ("sha256", "bytes")} for name in before}
    passed = len(outcomes) == len(commands) and all(x["returncode"] == 0 and not x["timed_out"]
                                                   and x["group_absent"] for x in outcomes) and before == after
    if passed:
        passed = runner.authenticate_prior() == auth and not any(x in sys.modules for x in ("torch", "numpy", "tensorflow", "jax", "mlx"))
    files = {p.name: {k: runner.descriptor(p)[k] for k in ("sha256", "bytes")}
             for p in args.output.iterdir() if p.is_file()}
    runner.write(args.output / "receipt.json", {"status": "passed" if passed else "failed",
        "source_before": before, "source_after": after, "sources_unchanged": before == after,
        "sources_before": {k: v["sha256"] for k, v in before.items()},
        "sources_after": {k: v["sha256"] for k, v in after.items()},
        "commands": outcomes, "files": files, "metadata_handoff_passed": passed,
        "empirical_array_decodes": 0, "checkpoint_decodes": 0, "scientific_calls": 0})
    print(json.dumps({"status": "passed" if passed else "failed", "receipt": runner.descriptor(args.output / "receipt.json")}), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
