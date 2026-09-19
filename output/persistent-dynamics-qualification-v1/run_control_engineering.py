"""One complete six-role engineering run and replay, using only seed410.

No scientific gate or performance selection is made here. The single fixed
case measures integration, complete evidence, failure behavior and capacity.
"""

import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from openjev.research import reacher_search_protocol as artifacts
from openjev.research.reacher_adaptive_search import SearchInputs
from openjev.research.reacher_tracking_audit import audit_row
from openjev.research.reacher_tracking_dynamics import nominal_model
from openjev.research.reacher_tracking_policy import ARMS
from openjev.research.reacher_tracking_rollout import check_deadline, run_row, sha, write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/persistent-dynamics-qualification-v1/control-engineering-01"
SOURCES = ["src/openjev/research/" + name + ".py" for name in (
    "reacher_tracking_dynamics", "reacher_tracking_identification", "reacher_tracking_control",
    "reacher_tracking_policy", "reacher_tracking_rollout", "reacher_tracking_audit",
    "reacher_geometry_physics", "reacher_geometry_reward", "reacher_reward_residual",
    "reacher_physics_control", "reacher_adaptive_search", "reacher_search_protocol", "robotics_reacher")]
SOURCES += [Path(__file__).relative_to(ROOT).as_posix()]


def protected_sources():
    checks = []
    for name, key in [("evidence/reacher-two-observation-study-v1/protocol/plan.json", "sources"),
                      ("evidence/reacher-innovation-pilot-v1/protocol/plan.json", "source_sha256")]:
        sources = json.loads((ROOT / name).read_text())[key]
        assert all(sha(ROOT / path) == digest for path, digest in sources.items()), name
        checks.append({"plan": name, "unchanged_sources": len(sources)})
    return checks


def main():
    torch.set_num_threads(1)
    protected = protected_sources()
    source_hashes = {name: sha(ROOT / name) for name in SOURCES}
    OUT.mkdir(parents=True, exist_ok=False)
    execution_start = time.monotonic()
    execution_receipt_valid = False
    stage, rows, audits = "preparation", [], []
    try:
        write(OUT / "started.json", {"engineering": True, "status": "started", "seed": 410,
            "case_count": 1, "arms": ARMS, "steps": 200, "gain_change_step": 80,
            "gain_before": .7, "gain_after": 1.3, "noise_std": .05,
            "target_events": {0: [.12, -.04], 50: [-.12, .08], 100: [.08, .13], 150: [-.09, -.12]},
            "window": 20, "freeze_after": 40, "planning_horizon": 12, "action_block": 3,
            "execution_cap_seconds": 600, "audit_cap_seconds": 600,
            "gain_grid": np.linspace(.5, 1.5, 21), "all_rows_before_audits": True,
            "scientific_gate": None, "automatic_retry": False,
            "source_sha256": source_hashes, "protected_sources": protected,
            "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "mujoco", "gymnasium")},
                "torch_threads": torch.get_num_threads()},
            "limits": "Single repeated engineering initial condition and fixed target/gain recipe; no fresh scientific cohort, fitting or benchmark qualification."})
        for name in SOURCES:
            target = OUT / "source-snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write((ROOT / name).read_bytes())
        assert all(sha(OUT / "source-snapshot" / name) == digest for name, digest in source_hashes.items())
        check_deadline(execution_start+600)
        rng = np.random.default_rng(410)
        noise = rng.normal(0., .05, (200, 2))
        targets = np.empty((201, 2))
        for first, last, value in [(0, 50, [.12, -.04]), (50, 100, [-.12, .08]),
                                   (100, 150, [.08, .13]), (150, 201, [-.09, -.12])]:
            targets[first:last] = value
        grid = np.linspace(.5, 1.5, 21)
        assert grid[4] == .7 and grid[10] == 1. and grid[16] == 1.3
        gains = np.r_[np.full(80, grid[4]), np.full(120, grid[16])]
        artifacts.save_npz(OUT / "case.npz", targets=targets, gains=gains, noise=noise)
        stems = []
        for step in range(200):
            arrays = [rng.normal(size=(1, count, 4, 2)) for count in (64, 192, 64, 64, 63)]
            inputs = SearchInputs(arrays[0], arrays[1], tuple(arrays[2:]))
            stem = OUT / "inputs" / f"{step:03d}"
            artifacts.save_inputs(stem, inputs, prefix=f"engineering410/{step}")
            stems.append(stem)
        stage = "execution"
        for arm in ARMS:
            receipt = run_row(arm=arm, target_path=targets, gain_schedule=gains, noise=noise,
                reset_seed=410, inputs_by_step=stems, gain_grid=grid, window=20, freeze_after=40,
                noise_std=.05, planning_horizon=12, action_block=3,
                out=OUT / "rows" / arm, deadline=execution_start+600, engineering=True)
            rows.append({"arm": arm, "wall_seconds": receipt["wall_seconds"],
                         "completed_sha256": sha(OUT / "rows" / arm / "completed.json")})
            print(json.dumps({"phase": "engineering_row", "arm": arm, "wall_seconds": receipt["wall_seconds"]}), flush=True)
        check_deadline(execution_start+600)
        write(OUT / "execution-completed.json", {"status": "completed", "engineering": True,
            "rows": rows, "wall_seconds": time.monotonic()-execution_start,
            "native_decisions": 1200, "candidate_native_transitions": 2987520,
            "selected_native_transitions": 1000, "identifier_native_transitions": 5040})
        check_deadline(execution_start+600)
        execution_receipt_valid = True
        stage = "audit"
        audit_start = time.monotonic()
        for arm in ARMS:
            value = audit_row(OUT / "rows" / arm, nominal_model=nominal_model(),
                              inputs_by_step=stems, deadline=audit_start+600)
            write(OUT / ("audit-"+arm+".json"), value)
            audits.append({"arm": arm, "sha256": sha(OUT / ("audit-"+arm+".json"))})
            print(json.dumps({"phase": "engineering_audit", "arm": arm}), flush=True)
        assert all(sha(ROOT / name) == digest for name, digest in source_hashes.items())
        protected_sources()
        check_deadline(audit_start+600)
        write(OUT / "completed.json", {"status": "completed", "engineering": True,
            "rows": rows, "audits": audits, "audit_wall_seconds": time.monotonic()-audit_start,
            "total_wall_seconds": time.monotonic()-execution_start,
            "scientific_gate": None, "scientific_data_allocated": False,
            "execution_completed_sha256": sha(OUT / "execution-completed.json"),
            "source_sha256": source_hashes})
        check_deadline(audit_start+600)
    except BaseException as error:
        for name in ("completed.json", "execution-completed.json"):
            if name == "execution-completed.json" and execution_receipt_valid:
                continue
            if (OUT / name).exists():
                try:
                    (OUT / name).rename(OUT / ("partial-"+name))
                except BaseException as receipt_error:  # noqa: BLE001 - preserve the primary failure
                    error.add_note("Completion demotion failed: " + repr(receipt_error))
        try:
            write(OUT / "failed.json", {"status": "failed", "engineering": True, "stage": stage,
            "error": repr(error), "rows_completed": rows, "audits_completed": audits,
                "wall_seconds": time.monotonic()-execution_start, "automatic_retry": False})
        except BaseException as receipt_error:  # noqa: BLE001 - preserve the primary failure
            error.add_note("Failure receipt could not be written: " + repr(receipt_error))
        raise


if __name__ == "__main__":
    main()
