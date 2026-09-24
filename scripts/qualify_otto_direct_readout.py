"""Fabricated direct-readout qualification and numerical capacity, no empirical decodes."""
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

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/qualify_otto_direct_readout.py"


def load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, ROOT / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def capacity(output):
    """Fixed fabricated 5000-row solve and long causal cache capacity probe."""
    import resource

    import numpy as np
    import torch

    from openjev.research import otto_direct_readout as solver
    from openjev.research import otto_direct_readout_cache as cache
    from openjev.research import otto_query_memory_data as data
    from openjev.research import otto_scheduled_predictor as predictor

    runner = load("scripts/run_otto_direct_readout.py", "_direct_capacity_runner")
    runner.require(all(os.environ.get(k) == "1" for k in runner.THREADS), "one numerical thread")
    clock = load(runner.CLOCK, "_direct_capacity_clock").SuspendClock()
    start = clock.now_ns()
    deadline = start + 240 * 10**9
    def check():
        runner.require(clock.now_ns() < deadline, "fabricated capacity deadline")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rng = np.random.default_rng(17)
    z = rng.normal(size=(5000,29)); z[:, -1] = 1
    errors = rng.normal(size=(5000,4))
    legal = np.ones((5000,4), np.bool_)
    tick = clock.now_ns()
    A,b = solver.build_design(z, errors, legal, np.full(5000,1/20000))
    design = (clock.now_ns()-tick)/1e9
    timings = {}
    for mode in ('ols','ridge'):
        check()
        tick = clock.now_ns()
        solution = solver.solve_design(A,b,mode=mode)
        timings[mode] = (clock.now_ns()-tick)/1e9
        runner.require(solution['diagnostics']['rank'] == 87, 'fabricated full-rank probe')
    sys.path.insert(0, str(ROOT/'tests'))
    from test_otto_query_memory_data import census
    lengths = (2188,2188,2188)
    flat,identities = census(lengths)
    history = data.project_census(flat,identities,query_period=4,expected_stage='train')
    state = {k:v.detach().clone() for k,v in predictor.make_head('frozen',17,4).state_dict().items()}
    extracted = cache.extract(state,17,history,check=check)
    # Worst 54 TRAIN and 36 DEV horizons, all 27 inference views, factor two.
    projection = 2*(3*(54*2188/5000)*(design+sum(timings.values()))
                   + (3*54+12*54+15*36)*2188/sum(lengths)*extracted['seconds']) + 120
    auditor = load('scripts/audit_otto_direct_readout.py', '_direct_audit_capacity')
    tick = clock.now_ns()
    auditor.projected_objective(np, z, errors, legal, np.full(5000, 1/20000),
                               np.zeros((4,29)), check=check)
    objective_seconds = (clock.now_ns()-tick)/1e9
    scalar = load('scripts/audit_otto_query_memory.py', '_direct_scalar_capacity')
    tick = clock.now_ns()
    scalar.scalar_report(np, identities, flat['raw_q'], flat['legal'], flat['raw_q'],
                         flat['raw_q'], flat['episode_offsets'], 'fabricated', 17, check=check)
    scalar_seconds = (clock.now_ns()-tick)/1e9
    # The bound uses authenticated length metadata only, never empirical arrays.
    auth = runner.authenticate()
    length_pin = auth['lineage']['evidence']['training_payload:train-views.json']
    runner.verify(length_pin)
    train_metadata = runner.read(length_pin['path'])['views'][0]['metrics']['identity_manifest']
    runner.require(len(train_metadata) == 54, 'complete TRAIN length metadata')
    runner.verify(auth['prior_views'])
    dev_metadata = runner.read(auth['prior_views']['path'])[0]['metrics']['identity_manifest']
    runner.require(len(dev_metadata) == 36, 'complete DEV length metadata')
    train_rows, dev_rows = (sum(r['length'] for r in m) for m in (train_metadata,dev_metadata))
    audit_projection = 2*(24*train_rows/5000*objective_seconds
                           + (12*train_rows+15*dev_rows)/sum(lengths)*scalar_seconds) + 60
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024)
    passed = projection <= 1350 and audit_projection <= 450 and rss <= 2*1024**3
    runner.write(output, {'status':'passed' if passed else 'failed','empirical_array_decodes':0,
        'checkpoint_decodes':0,'scientific_calls':0,'fabricated_rows':5000,'cache_rows':sum(lengths),
        'design_seconds':design,'solve_seconds':timings,'cache_seconds':extracted['seconds'],
        'projected_seconds':projection,'peak_rss_bytes':rss,'admission_threshold_seconds':1350,
        'audit_projected_seconds':audit_projection,'audit_admission_threshold_seconds':450,
        'audit_objective_probe_seconds':objective_seconds,'audit_scalar_probe_seconds':scalar_seconds,
        'authenticated_train_rows':train_rows,'authenticated_dev_rows':dev_rows,
        'projection_scope':'factor-two projection at maximum allowed complete episode lengths, plus120seconds'})
    runner.require(passed,'fabricated capacity within fixed admission threshold')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", action="store_true")
    args = parser.parse_args()
    if args.capacity:
        capacity(args.output)
        return
    runner = load("scripts/run_otto_direct_readout.py", "_compute_qualification_runner")
    runner.require(Path.cwd() == ROOT and args.output.is_absolute() and args.output.is_relative_to(ROOT),
                   "contained qualification output")
    args.output.mkdir(exist_ok=False)
    auth = runner.authenticate()
    before = {name: {k: runner.descriptor(name)[k] for k in ("sha256", "bytes")}
              for name in auth["sources"]}
    tests = sorted(name for name in runner.NEW if name.startswith("tests/"))
    lint = sorted(name for name in runner.NEW if name.endswith(".py"))
    commands = [(180, [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "-p", "no:cacheprovider",
                      "--basetemp", str(args.output / "pytest-temp"), *tests]),
                (60, [str(ROOT / ".venv/bin/ruff"), "check", "--no-cache", *lint]),
                (240, [str(ROOT / ".venv/bin/python"), str(ROOT / SELF),
                       "--capacity", "--output", str(args.output / "capacity.json")])]
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
        passed = runner.authenticate() == auth and not any(x in sys.modules for x in ("torch", "numpy", "tensorflow", "jax", "mlx"))
    files = {p.name: runner.descriptor(p)
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
