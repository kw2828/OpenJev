"""One bounded synthetic capacity probe, not an empirical performance result.

Runs one fit's complete worst-horizon DEV roster and all24 views. The original
engineering supervisor bounds this process and binds its sources and receipt.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location("_residual_capacity_runner", ROOT / "scripts/run_otto_residual.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temp", type=Path, required=True)
    args = parser.parse_args()
    runner = load_runner()
    runner.exclusive_output(args.output)
    runner.exclusive_output(args.temp)
    runner.require(Path.cwd() == ROOT and Path(sys.executable).absolute() == ROOT / ".venv/bin/python"
                   and all(os.environ.get(name) == "1" for name in runner.THREADS), "fixed synthetic runtime")
    args.temp.mkdir()
    start = time.monotonic()
    result = {"status": "started", "episodes": 18, "length": 2188, "views": 24, "fit_seeds_measured": 1,
              "synthetic_seed": 1001, "synthetic_joint_seed": 1002,
              "empirical_array_decodes": 0, "checkpoint_decodes": 0, "teacher_calls": 0, "native_calls": 0,
              "scope": "fabricated capacity only; synthetic weights and targets; no effectiveness or speedup claim"}
    try:
        import numpy as np
        import torch

        from openjev.research import otto_query_memory_data as data
        from openjev.research import otto_query_memory_metrics as metrics
        from openjev.research import otto_residual_audit as audit
        from openjev.research import otto_residual_contract as contract
        from openjev.research import otto_residual_features as features
        from openjev.research import otto_residual_replay as replay
        from openjev.research import otto_scheduled_predictor as predictor
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        identities = []
        for regime_index, regime in enumerate(("lambda3", "lambda4")):
            for case in range(3):
                for arm in ("analytic", "neural", "period4_hold"):
                    identities.append({"stage": "dev", "regime": regime, "case": case, "arm": arm,
                                       "seed": 1001 + regime_index * 3 + case, "episode_index": len(identities),
                                       "episode_id": f"capacity:{regime}:{case}:{arm}"})
        length, count = 2188, 18
        steps = np.tile(np.arange(length), count)
        rng = np.random.default_rng(1001)
        flat = {"features": rng.normal(0., .2, (count * length, 31)).astype(np.float32),
                "raw_q": rng.normal(4., 1., (count * length, 4)).astype(np.float32),
                "legal": np.ones((count * length, 4), dtype=np.bool_),
                "actions": np.zeros(count * length, dtype=np.int64), "correction": steps % 4 == 0,
                "episode_offsets": np.arange(count + 1, dtype=np.int64) * length}
        flat["features"][:, 15] = steps / 2188
        flat["features"][:, 16] = (steps % 4) / 2188
        flat["features"][:, 17] = 1.
        probe = runner.Run(SimpleNamespace(output=args.temp))
        probe.clock = runner.load(runner.CLOCK, "_residual_capacity_clock").SuspendClock()
        probe.launch = {"deadline_ns": probe.clock.now_ns() + 240 * 10**9}
        probe.plan = {"limits": runner.LIMITS["evaluate"]}

        def save(name, arrays):
            probe.check(force=True)
            path = args.temp / name
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            pin = runner.descriptor(path)
            with np.load(path, allow_pickle=False) as archive:
                runner.require(set(archive.files) == set(arrays)
                               and all(archive[k].dtype == v.dtype and archive[k].shape == v.shape
                                       and archive[k].tobytes() == v.tobytes() for k, v in arrays.items()), "capacity array roundtrip")
            return pin

        tick = time.monotonic()
        history = data.project_census(flat, identities, query_period=4, expected_stage="dev")
        models = [predictor.make_head("frozen", seed, 4) for seed in (1001, 1002)]
        projection = torch.tensor(rng.normal(0., .2, (8, 28)), dtype=torch.float32)
        cache = features.build_cache(*models, projection, history["features"], history["query_scores"], history["episode_offsets"])
        runner.require(cache["work_counts"] == runner.expected_cache_work(cache["episode_offsets"]), "worst-horizon cache accounting")
        cache_arrays = {k: v for k, v in cache.items() if isinstance(v, np.ndarray)}
        cache_pin = save("cache.npz", cache_arrays)
        views = []
        for index, (method, tau, seed) in enumerate(spec for spec in contract.view_specs("dev") if spec[2] == contract.FIT_SEEDS[0]):
            replayed = replay.replay(cache, method, tau=tau)
            arrays = {k: v for k, v in replayed.items() if isinstance(v, np.ndarray)}
            name = f"prediction-{index:02}.npz"
            pin = save(name, arrays)
            report = runner.report_view(metrics, history, identities, replayed["action_scores"], method, tau, seed, contract, probe.check)
            views.append({"path": name, "pin": pin, "method": method, "tau": tau, "seed": seed,
                          "report": report, "work_counts": replayed["work_counts"], "state_bytes": replayed["state_bytes"]})
        runner.write(args.temp / "views.json", {"views": views})
        result["evaluate_seconds"] = time.monotonic() - tick
        print(json.dumps({"phase": "synthetic_evaluate", "seconds": result["evaluate_seconds"]}), flush=True)
        tick = time.monotonic()
        probe.plan = {"limits": runner.LIMITS["audit"]}
        runner.require(runner.descriptor(args.temp / "cache.npz") == cache_pin, "synthetic cache bytes")
        with np.load(args.temp / "cache.npz", allow_pickle=False) as archive:
            loaded_cache = {k: archive[k] for k in archive.files}
        loaded_cache.update({k: v for k, v in cache.items() if k not in cache_arrays})
        saved = runner.read(args.temp / "views.json")
        for view in saved["views"]:
            runner.require(runner.descriptor(args.temp / view["path"]) == view["pin"], "synthetic prediction bytes")
            with np.load(args.temp / view["path"], allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
            repeated = replay.replay(loaded_cache, view["method"], tau=view["tau"])
            runner.require(all(v.dtype == repeated[k].dtype and v.shape == repeated[k].shape
                               and v.tobytes() == repeated[k].tobytes() for k, v in arrays.items())
                           and repeated["work_counts"] == view["work_counts"] and repeated["state_bytes"] == view["state_bytes"],
                           "synthetic replay identity")
            report = audit.report_view(identities, history["targets"], history["legal"], arrays["action_scores"],
                                       history["episode_offsets"], view["method"], view["tau"], view["seed"], check=probe.check)
            runner.equal_nested(view["report"], report, "synthetic independent metrics")
        result["audit_seconds"] = time.monotonic() - tick
        result.update({phase + "_projected_seconds": 2 * 3 * result[phase + "_seconds"] + 120 for phase in ("evaluate", "audit")})
        result.update(status="passed" if all(result[p + "_projected_seconds"] <= 675 for p in ("evaluate", "audit")) else "failed_capacity",
                      cache_work_counts=cache["work_counts"], numpy=np.__version__, torch=torch.__version__,
                      torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                      deterministic=torch.are_deterministic_algorithms_enabled())
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result["total_seconds"] = time.monotonic() - start
        result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        runner.write(args.output, result)
        print(json.dumps({k: v for k, v in result.items() if k != "cache_work_counts"}), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
