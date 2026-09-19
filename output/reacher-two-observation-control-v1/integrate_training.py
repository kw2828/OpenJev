"""Retained engineering integration, never a scientific fit or effect estimate.

Three tiny fits intentionally reuse the existing seed410 fixture and identical
initial tensors/data/orders. This checks fit serialization and independent saved
audit together. It is not three independent training replicates.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import torch
from test_reacher_two_observation_training import inputs

from openjev.research import reacher_two_observation_fitting as fitting
from openjev.research import reacher_two_observation_training_audit as audit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/reacher-two-observation-control-v1/training-integration-attempt-01"
CAP_SECONDS = 120


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    begin = time.monotonic()
    deadline = begin + CAP_SECONDS
    OUT.mkdir(parents=True, exist_ok=False)
    original_threads = torch.get_num_threads()
    original_rng = torch.get_rng_state().clone()
    phase, progress, rows = "source_authentication", {}, []

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Retained training integration cap")

    try:
        frozen = json.loads((ROOT / "evidence/reacher-geometry-memory-v1/protocol/plan.json").read_text())["sources"]
        assert len(frozen) == 90
        for name, digest in frozen.items():
            assert sha(ROOT / name) == digest, name
        paths = [Path(__file__).relative_to(ROOT).as_posix(),
                 "tests/test_reacher_two_observation_training.py"]
        paths += [p.relative_to(ROOT).as_posix() for p in sorted(
            (ROOT / "src/openjev/research").glob("reacher_two_observation_*.py"))
            if p.name != "reacher_two_observation_experiment.py"]
        sources = {name: sha(ROOT / name) for name in paths}
        runtime = {"python": sys.version, "torch": str(torch.__version__),
                   "platform": platform.platform(), "machine": platform.machine(),
                   "device": "cpu", "threads": 2, "dtype": "torch.float32"}
        write(OUT / "started.json", {"scope": __doc__, "cap_seconds": CAP_SECONDS,
              "sources": sources, "frozen_sources": frozen, "runtime": runtime,
              "seed": 410, "native_calls": 0, "scientific_streams": 0,
              "planned_fits": 3, "planned_updates_per_fit": 4})
        torch.set_num_threads(2)
        phase = "fixture_inputs"
        initial, orders, data, kw = inputs()
        torch.save({"initial": initial, "orders": orders, "data": data}, OUT / "inputs.pt")
        input_file_sha = sha(OUT / "inputs.pt")
        kw["source_sha256"], kw["runtime"] = sources, runtime
        kw["provenance"] = {"scope": "synthetic_seed410_identical_three_fit_integration",
                            "initial_state_kind": "engineering_initialization",
                            "initial_source_sha256": input_file_sha,
                            "independent_replicates": False}
        write(OUT / "bindings.json", {**{k: v for k, v in kw.items() if k != "settings"},
                                    "settings": kw["settings"].configuration(),
                                    "input_file_sha256": input_file_sha})
        for pair in range(3):
            phase = f"fit_pair{pair}"
            folder = OUT / f"fits/two_observation_gru-pair{pair}"
            receipt = fitting.fit_one(initial, orders, data, folder, **kw,
                pair=f"pair{pair}", name=f"two_observation_gru-pair{pair}",
                deadline=deadline, progress=progress)
            phase = f"saved_audit_pair{pair}"
            checkpoint = torch.load(folder / "checkpoint.pt", weights_only=True)
            final = torch.load(folder / "weights.pt", weights_only=True)
            logs = [json.loads(line) for line in (folder / "training.jsonl").read_text().splitlines()]
            checked = audit.audit_training(checkpoint, logs, initial_weights=initial,
                orders=orders, public_data=data,
                **{k: v for k, v in kw.items() if k != "settings"},
                settings=kw["settings"].configuration(),
                expected_checkpoint_sha256=checkpoint["integrity_sha256"],
                final_weights=final, deadline_check=check)
            write(OUT / f"audit-pair{pair}.json", checked)
            rows.append({"pair": pair, "fit": receipt, "audit": checked})
        phase = "final_authentication"
        for name, digest in {**frozen, **sources}.items():
            check()
            assert sha(ROOT / name) == digest, name
        assert torch.equal(original_rng, torch.get_rng_state()), "Ambient RNG changed"
        members = {p.relative_to(OUT).as_posix(): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
        check()
        write(OUT / "completed.json", {"status": "completed_engineering_integration",
              "new_scientific_fits": 0, "new_engineering_fits": 3,
              "new_engineering_optimizer_updates": 12, "native_calls": 0,
              "scientific_streams": 0, "identical_fit_inputs_intentional": True,
              "ambient_rng_unchanged": True, "frozen_sources_unchanged": 90,
              "wall_seconds": time.monotonic() - begin, "rows": rows, "members": members,
              "limits": "Tiny synthetic fit/serialization/audit integration only; no full study, learned performance, native control or independent replicate result."})
        check()
        print(json.dumps({"status": "completed", "directory": str(OUT),
                          "wall_seconds": time.monotonic() - begin}))
    except BaseException as error:
        if (OUT / "completed.json").exists():
            (OUT / "completed.json").rename(OUT / "invalid-completion.json")
        write(OUT / "failed.json", {"status": "failed", "phase": phase,
              "error": repr(error), "traceback": traceback.format_exc(),
              "progress": progress, "completed_pairs": len(rows),
              "wall_seconds": time.monotonic() - begin, "automatic_retry": False})
        raise
    finally:
        torch.set_num_threads(original_threads)


if __name__ == "__main__":
    main()
