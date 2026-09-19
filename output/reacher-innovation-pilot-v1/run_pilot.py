"""Explicit preparation and no-retry execution of the four-gate development pilot."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from openjev.research.reacher_innovation_context import VARIANTS, InnovationContextWorldModel
from openjev.research.reacher_innovation_pilot import PilotConfig, evaluate_development, fit_one
from openjev.research.reacher_innovation_pilot_data import prepare_data
from openjev.research.reacher_objective_training import canonical_tensor_hash

BASE = Path("output/reacher-innovation-pilot-v1")
PROTOCOL = Path("evidence/reacher-innovation-pilot-v1/protocol")
ATTEMPT = Path("runs/reacher-innovation-pilot-v1/attempt")
PARENT = Path("evidence/reacher-two-observation-study-v1/protocol")
CORPUS = Path("runs/reacher-two-observation-study-v1/attempt/inherited")
EXTRA_SOURCES = [
    "src/openjev/research/reacher_innovation_context.py",
    "src/openjev/research/reacher_innovation_loss.py",
    "src/openjev/research/reacher_innovation_pilot.py",
    "src/openjev/research/reacher_innovation_pilot_data.py",
    "tests/test_reacher_innovation_context.py",
    "tests/test_reacher_innovation_loss.py",
    "tests/test_reacher_innovation_pilot.py",
    "tests/test_reacher_innovation_pilot_data.py",
    str(BASE / "run_pilot.py"), str(BASE / "design.md"),
    str(BASE / "test_run_pilot.py"),
    str(BASE / "capacity.py"), str(BASE / "capacity-review.json"),
    str(BASE / "launch.py"), str(BASE / "test_launch.py"),
    str(BASE / "audit_pilot.py"), str(BASE / "test_audit_pilot.py"),
    str(BASE / "capacity-process-01" / "terminal.json"),
    str(BASE / "capacity-process-02" / "terminal.json"),
    str(BASE / "capacity-attempt-02" / "completed.json"),
]


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as file:
        json.dump(value, file, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")


def save(path, value):
    with Path(path).open("xb") as file:
        torch.save(value, file)


def utc():
    return datetime.now(UTC).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def runtime():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": str(torch.__version__), "numpy": str(np.__version__),
            "dtype": "torch.float32", "device": "cpu", "torch_threads": 1}


def config(variant):
    return PilotConfig(variant=variant, hidden_size=64, context_size=16,
                       dt=0.02, noise_std=0.05, eta=0.1, variance_min=1e-4,
                       variance_max=4.0, epochs=8, batch_size=32, variance_score_weight=0.1)


def check_sources(sources):
    for path, digest in sources.items():
        require(sha(path) == digest, "Source changed: " + path)


def seed(role):
    return int.from_bytes(hashlib.sha256(f"OpenJev/reacher-innovation-pilot-v1/{role}".encode()).digest()[:4], "big")


def historical_seeds(value, selected=False):
    """Conservative exclusion of seed/registry integers in the bound parent plan."""
    if isinstance(value, dict):
        result = set()
        for key, item in value.items():
            result |= historical_seeds(item, selected or "seed" in key.lower() or "registry" in key.lower())
        return result
    if isinstance(value, list):
        return set().union(*(historical_seeds(item, selected) for item in value))
    return {value & 0xFFFFFFFF} if selected and type(value) is int else set()


def prepare():
    PROTOCOL.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(PROTOCOL / "started.json", {"utc": utc(), "purpose": "development-only preparation"})
    try:
        parent = json.loads((PARENT / "freeze.json").read_text())
        check_sources(parent["source_sha256"])
        sources = dict(parent["source_sha256"])
        sources.update({path: sha(path) for path in EXTRA_SOURCES})
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        for path, digest in sources.items():
            require(sha_bytes(subprocess.check_output(["git", "show", f"{source_commit}:{path}"])) == digest,
                    "Source must be committed before preparation: " + path)
        for path in sources:
            target = PROTOCOL / "source-snapshot" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(path).read_bytes())
        data = prepare_data(CORPUS / "train.npz", CORPUS / "train.json")
        write_json(PROTOCOL / "split.json", data.manifest)
        public = {"train": data.train, "dev6": data.dev6, "dev10": data.dev10}
        data_hashes = {name: canonical_tensor_hash(tensors) for name, tensors in public.items()}
        for name, tensors in public.items():
            save(PROTOCOL / f"{name}.pt", tensors)
        registry = {f"pair{pair}/{kind}": seed(f"pair{pair}/{kind}")
                    for pair in range(3) for kind in ("initialization", "orders")}
        require(len(set(registry.values())) == 6, "New pilot role collision")
        prior = historical_seeds(json.loads((PARENT / "plan.json").read_text())) | {410}
        require(not (set(registry.values()) & prior), "Pilot role overlaps inherited seed inventory")
        pairs = []
        for pair in range(3):
            init_seed, order_seed = (registry[f"pair{pair}/{k}"] for k in ("initialization", "orders"))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(init_seed)
                generator_before = sha_bytes(torch.get_rng_state().numpy().tobytes())
                model = InnovationContextWorldModel(**config("constant").model_kwargs()).cpu().float()
                weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
                generator_after = sha_bytes(torch.get_rng_state().numpy().tobytes())
            rng = np.random.default_rng(order_seed)
            order_before = rng.bit_generator.state
            orders = torch.from_numpy(np.stack([rng.permutation(640) for _ in range(8)]).astype(np.int64))
            save(PROTOCOL / f"initial-pair{pair}.pt", weights)
            save(PROTOCOL / f"orders-pair{pair}.pt", orders)
            pairs.append({"pair": pair, "initial_tensor_sha256": canonical_tensor_hash(weights),
                          "orders_tensor_sha256": canonical_tensor_hash({"orders": orders}),
                          "initialization_seed": init_seed, "orders_seed": order_seed,
                          "torch_initial_rng_sha256": generator_before, "torch_final_rng_sha256": generator_after,
                          "numpy_initial_rng": order_before, "numpy_final_rng": rng.bit_generator.state})
        require(len({p["initial_tensor_sha256"] for p in pairs}) == 3, "Distinct paired initializations required")
        members = {str(p.relative_to(PROTOCOL)): {"sha256": sha(p), "bytes": p.stat().st_size}
                   for p in sorted(PROTOCOL.rglob("*")) if p.is_file()}
        plan = {"schema": "reacher-innovation-development-pilot-v1", "status": "frozen_before_training",
                "prepared_utc": utc(), "development_only": True, "no_retry": True,
                "native_control": False, "untouched_test_performance": False,
                "fit_count": 12, "updates_per_fit": 160, "total_updates": 1920,
                "cap_seconds": 1800, "all_fits_before_evaluation": True,
                "engineering_qualification": {"failed_attempt": str(BASE / "capacity-process-01" / "terminal.json"),
                    "completed_attempt": str(BASE / "capacity-process-02" / "terminal.json"),
                    "independent_saved_output_review": str(BASE / "capacity-review.json"),
                    "synthetic_only": True, "speed_comparison": False},
                "evaluations": ["initial", "final"], "panels": ["dev6", "dev10"],
                "variants": list(VARIANTS), "pairs": pairs,
                "configurations": {v: dataclasses.asdict(config(v)) for v in VARIANTS},
                "source_commit": source_commit,
                "data_tensor_sha256": data_hashes, "runtime": runtime(), "source_sha256": sources,
                "prepared_members": members,
                "parent_plan_sha256": sha(PARENT / "plan.json"),
                "parent_freeze_sha256": sha(PARENT / "freeze.json"),
                "random_registry": registry, "historical_seed32_exclusion_count": len(prior),
                "selection": {"primary": "post_reacquisition_mean_mse", "normalized_recovery_ratio": 0.97,
                              "paired_nonworse": True, "constant_family_nonworse": True,
                              "angle_regression_ratio": 1.02, "reward_regression_ratio": 1.05,
                              "ordinary_initial_recovery_ratio": 0.9,
                              "confirm_age_table_before_learned_uncertainty_claim": True},
                "preparation_wall_seconds": time.perf_counter() - started}
        write_json(PROTOCOL / "plan.json", plan)
        print(json.dumps({"status": "prepared", "plan_sha256": sha(PROTOCOL / "plan.json")}))
    except BaseException as error:
        original_traceback = traceback.format_exc()
        try:
            write_json(PROTOCOL / "failed.json", {"utc": utc(), "error": repr(error), "traceback": original_traceback})
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note("Preparation failure preservation failed: " + repr(preservation_error))
        raise


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load(name, plan):
    item = plan["prepared_members"][name]
    path = PROTOCOL / name
    require(sha(path) == item["sha256"], "Prepared artifact changed: " + name)
    return torch.load(path, map_location="cpu", weights_only=True)


def run(expected_sha, published_commit):
    plan_path = PROTOCOL / "plan.json"
    require(sha(plan_path) == expected_sha, "External plan digest mismatch")
    plan = json.loads(plan_path.read_text())
    published = subprocess.check_output(["git", "show", f"{published_commit}:{plan_path}"])
    require(sha_bytes(published) == expected_sha, "Published plan bytes differ")
    remote = json.loads(subprocess.check_output(["gh", "api", f"repos/kw2828/OpenJev/commits/{published_commit}"]))
    require(remote["sha"] == published_commit, "Protocol commit not visible on GitHub")
    require(runtime() == plan["runtime"], "Runtime differs from prepared protocol")
    check_sources(plan["source_sha256"])
    for name, item in plan["prepared_members"].items():
        require(sha(PROTOCOL / name) == item["sha256"], "Prepared member changed: " + name)
    ATTEMPT.mkdir(parents=True, exist_ok=False)
    start, deadline = time.perf_counter(), time.monotonic() + plan["cap_seconds"]
    write_json(ATTEMPT / "started.json", {"utc": utc(), "plan_sha256": expected_sha,
                                        "published_commit": published_commit, "no_retry": True})
    phase, current = "loading", None
    try:
        public = {name: load(name + ".pt", plan) for name in ("train", "dev6", "dev10")}
        completed_fits, fit_receipts = [], {}
        for pair in plan["pairs"]:
            n = pair["pair"]
            weights, orders = load(f"initial-pair{n}.pt", plan), load(f"orders-pair{n}.pt", plan)
            for variant in plan["variants"]:
                current, phase = f"pair{n}-{variant}", "fit"
                fit_path = ATTEMPT / "fits" / current
                receipt = fit_one(weights, orders, public["train"], fit_path,
                        config=PilotConfig(**plan["configurations"][variant]),
                        source_sha256=plan["source_sha256"], runtime=plan["runtime"],
                        expected_initial_sha256=pair["initial_tensor_sha256"],
                        expected_orders_sha256=pair["orders_tensor_sha256"],
                        expected_data_sha256=plan["data_tensor_sha256"]["train"], deadline=deadline)
                require(receipt["status"] == "completed"
                        and all(value == plan["updates_per_fit"] for value in receipt["counts"].values())
                        and receipt["config"] == plan["configurations"][variant], "Exact completed fit required")
                require(json.loads((fit_path / "completed.json").read_text()) == json.loads(json.dumps(receipt)),
                        "Returned and stored fit receipt differ")
                for member, item in receipt["files"].items():
                    require(sha(fit_path / member) == item["sha256"], "Fit artifact changed: " + member)
                fit_receipts[current] = {"receipt": receipt,
                                         "completed_sha256": sha(fit_path / "completed.json")}
                completed_fits.append(current)
                print(json.dumps({"phase": "fit_complete", "name": current, "fits": len(completed_fits)}), flush=True)
        require(len(completed_fits) == plan["fit_count"], "All fits required before development")
        write_json(ATTEMPT / "all-fits-completed.json", {"utc": utc(), "fits": fit_receipts})
        write_json(ATTEMPT / "evaluation-started.json", {"utc": utc(),
                   "all_fits_completed_sha256": sha(ATTEMPT / "all-fits-completed.json")})
        evaluations = []
        for pair in plan["pairs"]:
            n = pair["pair"]
            for variant in plan["variants"]:
                name = f"pair{n}-{variant}"
                for stage in plan["evaluations"]:
                    if stage == "initial":
                        weights = load(f"initial-pair{n}.pt", plan)
                        expected_weights = pair["initial_tensor_sha256"]
                    else:
                        fit_path, bound = ATTEMPT / "fits" / name, fit_receipts[name]
                        require(sha(fit_path / "completed.json") == bound["completed_sha256"],
                                "Completed fit receipt changed")
                        require(sha(fit_path / "weights.pt") == bound["receipt"]["files"]["weights.pt"]["sha256"],
                                "Final weight file changed")
                        weights = torch.load(fit_path / "weights.pt", map_location="cpu", weights_only=True)
                        expected_weights = bound["receipt"]["final_weights_sha256"]
                    for panel in plan["panels"]:
                        phase, current = "development", f"{name}/{stage}/{panel}"
                        summary = evaluate_development(
                            weights, public[panel], ATTEMPT / "evaluation" / name / stage / panel,
                            config=PilotConfig(**plan["configurations"][variant]),
                            source_sha256=plan["source_sha256"], runtime=plan["runtime"],
                            expected_weights_sha256=expected_weights,
                            expected_data_sha256=plan["data_tensor_sha256"][panel], deadline=deadline)
                        evaluations.append({"pair": n, "variant": variant, "stage": stage,
                                            "panel": panel, "summary": summary})
                        print(json.dumps({"phase": "evaluation_complete", "name": current}), flush=True)
        check_sources(plan["source_sha256"])
        require(time.monotonic() < deadline, "Pilot cap exceeded")
        write_json(ATTEMPT / "evaluation-index.json", evaluations)
        members = {str(p.relative_to(ATTEMPT)): {"sha256": sha(p), "bytes": p.stat().st_size}
                   for p in sorted(ATTEMPT.rglob("*")) if p.is_file()}
        write_json(ATTEMPT / "completed.json", {"utc": utc(), "plan_sha256": expected_sha,
                    "fits": len(completed_fits), "evaluations": len(evaluations),
                    "wall_seconds": time.perf_counter() - start, "development_only": True,
                    "gate_status": "requires_saved_output_review", "native_control_measured": False,
                    "members": members})
        require(time.monotonic() < deadline, "Pilot cap exceeded during terminal serialization")
    except BaseException as error:
        original_traceback = traceback.format_exc()
        try:
            if (ATTEMPT / "completed.json").exists():
                (ATTEMPT / "completed.json").rename(ATTEMPT / "invalid-completion.json")
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note("Terminal demotion failed: " + repr(preservation_error))
        try:
            write_json(ATTEMPT / "failed.json", {"utc": utc(), "phase": phase, "current": current,
                        "wall_seconds": time.perf_counter() - start, "error": repr(error),
                        "traceback": original_traceback, "no_retry": True})
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note("Failure preservation failed: " + repr(preservation_error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--plan-sha256")
    parser.add_argument("--published-commit")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "prepare":
        prepare()
    else:
        require(args.plan_sha256 and args.published_commit, "Explicit published protocol required")
        run(args.plan_sha256, args.published_commit)
