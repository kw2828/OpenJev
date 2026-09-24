"""One fabricated epoch per arm, with authenticated historical TRAIN lengths.

Only episode lengths are reused. Weights, features, targets and legal masks are
fabricated. This is a capacity screen, not a scientific update or effectiveness
result. The central qualifier owns the original240-second process supervisor.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINEAGE = "scripts/otto_residual_lineage.py"
LINEAGE_PIN = "07eedf024f3364e7d744f460009da7b1e4d9d0226b69e49656b1e7b5c3377667"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")
SEED = 309000001


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path, pin, name):
    source = ROOT / path
    require(not source.is_symlink() and hashlib.sha256(source.read_bytes()).hexdigest() == pin, "qualified source pin")
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    require(output.is_absolute() and output.is_relative_to(ROOT) and ".." not in output.parts
            and output.parent.is_dir() and not output.exists() and not output.is_symlink()
            and not any(p.is_symlink() for p in output.parents), "fresh contained capacity receipt")
    result = {"version": "otto-readout-capacity-v1", "status": "started", "episodes": 54, "seed": SEED,
        "epochs_per_arm": 1, "updates_per_arm": 9, "total_updates": 0, "episode_exposures": 0,
        "empirical_array_decodes": 0, "checkpoint_decodes": 0, "empirical_model_calls": 0,
        "teacher_calls": 0, "native_calls": 0, "fabricated_model_constructions": 0,
        "arms": [], "pending": "metadata_admission", "resource_cap_bytes": 4 * 1024**3,
        "scope": "fabricated capacity using only historical TRAIN length metadata; no effectiveness or speedup claim"}
    clock = start = None
    try:
        require(Path.cwd() == ROOT and Path(sys.executable).absolute() == ROOT / ".venv/bin/python"
                and all(os.environ.get(name) == "1" for name in THREADS), "fixed single-thread CPU runtime")
        clock = load(CLOCK, CLOCK_PIN, "_readout_capacity_clock").SuspendClock()
        start = clock.now_ns()
        deadline = start + 240 * 10**9

        def check():
            require(clock.now_ns() < deadline, "fixed capacity deadline")
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
            result["peak_rss_bytes"] = rss
            require(rss <= result["resource_cap_bytes"], "fixed capacity RSS cap")

        lineage = load(LINEAGE, LINEAGE_PIN, "_readout_capacity_lineage").authenticate()
        check()
        descriptor = lineage["evidence"]["training_payload:train-views.json"]
        raw = Path(descriptor["path"]).read_bytes()
        require(len(raw) == descriptor["bytes"] and hashlib.sha256(raw).hexdigest() == descriptor["sha256"],
                "authenticated TRAIN geometry metadata")
        metadata = json.loads(raw)
        require(metadata["stage"] == "train" and len(metadata["views"]) == 24, "complete historical TRAIN views")
        manifests = [row["metrics"]["identity_manifest"] for row in metadata["views"]]
        lengths = [row["length"] for row in manifests[0]]
        require(len(lengths) == 54 and all(type(n) is int and 1 <= n <= 2188 for n in lengths)
                and all([row["length"] for row in manifest] == lengths and len(manifest) == 54
                        and all(row["stage"] == "train" for row in manifest) for manifest in manifests),
                "same exact54 complete TRAIN episode lengths")
        result.update(length_metadata=descriptor, lengths=lengths, retained_rows=sum(lengths))
        del metadata, manifests, raw

        import numpy as np
        import torch

        from openjev.research import otto_query_memory_data as data
        from openjev.research import otto_readout_ablation_model as models
        from openjev.research import otto_readout_ablation_training as training

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        check()
        result["pending"] = "fabricated_geometry"
        steps = np.concatenate([np.arange(n, dtype=np.int64) for n in lengths])
        rng = np.random.Generator(np.random.PCG64(1001))
        features = rng.normal(0., .2, (len(steps), 31)).astype(np.float32)
        features[:, 15], features[:, 16], features[:, 17] = steps / 2188, steps % 4 / 2188, 1.
        flat = {"features": features, "raw_q": rng.normal(4., 1., (len(steps), 4)).astype(np.float32),
                "legal": np.ones((len(steps), 4), dtype=np.bool_), "actions": np.zeros(len(steps), dtype=np.int64),
                "correction": steps % 4 == 0, "episode_offsets": np.array([0, *np.cumsum(lengths)], dtype=np.int64)}
        identities = [{"stage": "train", "episode_id": f"fabricated-capacity:{i}", "episode_index": i,
                       "seed": 1001 + i // 3, "case": i // 3 % 9, "regime": "lambda3" if i < 27 else "lambda4",
                       "arm": ("analytic", "neural", "period4_hold")[i % 3]} for i in range(54)]
        fabricated = data.project_census(flat, identities, query_period=4, expected_stage="train")
        parent = models.make_model("full_joint", SEED)
        result["fabricated_model_constructions"] += 1
        parent_state = {name: value.detach().clone() for name, value in parent.slow.state_dict().items()}
        for arm in models.ARMS:
            check()
            tick = clock.now_ns()
            item = models.from_state(arm, SEED, parent_state)
            result["fabricated_model_constructions"] += 1
            optimizer = training.construct_optimizer(item)
            order = np.random.Generator(np.random.PCG64(SEED)).permutation(54).astype(np.int64)
            row = {"arm": arm, "order": order.tolist(), "updates": 0, "episode_exposures": 0,
                   "forward_rows": 0, "forward_chunks": 0, "backward_chunks": 0,
                   "work_counts": {}, "effective_parameter_count": item.parameter_metadata()["effective_count"]}
            result["arms"].append(row)
            for offset in range(0, 54, 6):
                indices = order[offset:offset + 6].tolist()
                result["pending"] = {"arm": arm, "batch": offset // 6, "indices": indices}
                recorded = training.batch_update(item, optimizer, fabricated, indices, check=check)
                require(recorded["optimizer_step"] == row["updates"] + 1 and recorded["optimizer_updates"] == 1,
                        "one complete fabricated update")
                row["updates"] += 1
                result["total_updates"] += 1
                for name in ("episode_exposures", "forward_rows", "forward_chunks", "backward_chunks"):
                    row[name] += recorded[name]
                result["episode_exposures"] += recorded["episode_exposures"]
                for name, count in recorded["work_counts"].items():
                    row["work_counts"][name] = row["work_counts"].get(name, 0) + count
            row["seconds"] = (clock.now_ns() - tick) / 1e9
            require(row["updates"] == 9 and row["episode_exposures"] == 54 and row["forward_rows"] == sum(lengths),
                    "complete exact-length fabricated epoch")
        result["projected_seconds"] = 2 * 40 * 3 * sum(row["seconds"] for row in result["arms"]) + 120
        require(result["total_updates"] == 27 and result["episode_exposures"] == 162, "complete capacity roster")
        result.update(status="passed" if result["projected_seconds"] <= 4050 else "failed_capacity", pending=None,
                      numpy=np.__version__, torch=torch.__version__, torch_threads=torch.get_num_threads(),
                      interop_threads=torch.get_num_interop_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
                      projection_rule="2 * 40 epochs * 3 fit seeds * sum(three measured arm epochs) + 120 <= 4050")
        check()
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        if clock is not None and start is not None:
            try:
                result["total_seconds"] = (clock.now_ns() - start) / 1e9
            except BaseException as error:  # noqa: BLE001 - preserve terminal clock failure
                result.update(status="failed", clock_error=repr(error))
        with output.open("x") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({key: result[key] for key in ("status", "total_updates", "projected_seconds")}), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
