"""Complete geometry-memory rehearsal with already completed tiny checkpoints.

No fitting or production checkpoints. The helper is separately hash-bound in
its engineering plan; it is not part of the production ninety-source contract.
Calling prepare_fixture does not run models or draw evaluation inputs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import reacher_geometry_memory_study as study

from openjev.research import reacher_geometry_memory_protocol as protocol

CACHE_PARENT = "output/reacher-cache-rehearsal-v1/attempt-01"
CACHE_PLAN_SHA = "c56f6b73c19b0c3ac17bf55a0f1ac95b2513e2688561d52907cb150ff57c1279"
CACHE_AUDIT_SHA = "77853eb8c8166651054f5e2fef185744809cf686a5c0fa8f7415b3782d9b6e34"
GEOMETRY_PARENT = "output/reacher-geometry-rehearsal-v1/attempt-02"
GEOMETRY_PLAN_SHA = "a72c5864d02147f354ec41fd03e888cb3995d864a1399ff78a6275c6dffb5a4a"
GEOMETRY_AUDIT_SHA = "fd53954a7247f07e54861a481f0c57c95c70d9f69320db048a416a4c6a5cd9ca"
NAMESPACE = "reacher-geometry-memory-engineering-whole-tree-v1"
SCOPE = "completed_tiny_engineering_weights_only_no_effectiveness_claim"


@dataclass(frozen=True)
class MemoryFixture:
    root: Path
    plan: dict
    path: Path
    digest: str
    execution: Path


def parents():
    cache_plan, cache_source = study.authenticate_evidence(
        f"{CACHE_PARENT}/plan.json", CACHE_PLAN_SHA, f"{CACHE_PARENT}/audit/receipt.json",
        CACHE_AUDIT_SHA, f"{CACHE_PARENT}/execution", kind="cache", engineering=True)
    geometry_plan, geometry_source = study.authenticate_evidence(
        f"{GEOMETRY_PARENT}/plan.json", GEOMETRY_PLAN_SHA,
        f"{GEOMETRY_PARENT}/audit/receipt.json", GEOMETRY_AUDIT_SHA,
        f"{GEOMETRY_PARENT}/execution", kind="geometry", engineering=True)
    study.require(geometry_plan["parent_source"]["plan_sha256"] == CACHE_PLAN_SHA
                  and geometry_plan["parent_source"]["audit_receipt_sha256"] == CACHE_AUDIT_SHA,
                  "Both completed engineering parents share the same tiny checkpoints")
    return cache_plan, cache_source, geometry_plan, geometry_source


def prepare_fixture(folder, *, cap_seconds=600, audit_cap_seconds=600):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    try:
        study.write(folder / "preparation-started.json", {"scope": SCOPE,
            "fixture_source_sha256": study.sha(Path(__file__)),
            "requested_execution_cap_seconds": repr(cap_seconds),
            "requested_audit_cap_seconds": repr(audit_cap_seconds), "new_model_calls": 0})
        study.require(all(type(v) is int and v > 0 for v in (cap_seconds, audit_cap_seconds)),
                      "Explicit positive engineering caps")
        case = _prepare_fixture(folder, cap_seconds=cap_seconds, audit_cap_seconds=audit_cap_seconds)
        study.write(folder / "preparation-completed.json", {"status": "prepared_not_run",
            "scope": SCOPE, "plan_sha256": case.digest,
            "wall_seconds": time.monotonic() - begin, "new_model_calls": 0, "new_native_calls": 0})
        return case
    except BaseException as error:
        study.write(folder / "preparation-failed.json", {"status": "failed", "scope": SCOPE,
            "error": repr(error), "wall_seconds": time.monotonic() - begin,
            "new_model_calls": 0, "new_native_calls": 0, "automatic_retry": False})
        raise


def _prepare_fixture(folder, *, cap_seconds, audit_cap_seconds):
    cache_plan, cache_source, geometry_plan, geometry_source = parents()
    plan = protocol.settings(engineering=True)
    plan.update(rng_namespace=NAMESPACE,
        engineering_rng_namespaces=[name for name in protocol.ENGINEERING_NAMESPACES if name != NAMESPACE],
        control_episodes=1, bootstrap_samples=8,
        hidden_size=cache_plan["hidden_size"], mlp_width=cache_plan["mlp_width"],
        cache_source=cache_source, geometry_source=geometry_source,
        cap_seconds=cap_seconds, audit_cap_seconds=audit_cap_seconds,
        sources=study.source_hashes(geometry_plan["sources"]), runtime=study.base.runtime(),
        engineering=True, fixture_source_sha256=study.sha(Path(__file__)))
    plan["random_stream_contract"] = study.stream_contract(plan)
    protocol.validate_settings(plan, engineering=True)
    snapshots = {**plan["sources"],
                 Path(__file__).resolve().relative_to(study.ROOT).as_posix(): plan["fixture_source_sha256"]}
    for name, digest in snapshots.items():
        study.search_study.copy_verified(study.ROOT / name, folder / "source-snapshot" / name,
                                         digest, float("inf"))
    path = folder / "plan.json"
    study.write(path, plan)
    digest = study.sha(path)
    study.write(folder / "fixture.json", {"scope": SCOPE, "plan_sha256": digest,
        "cache_parent_plan_sha256": CACHE_PLAN_SHA, "cache_parent_audit_sha256": CACHE_AUDIT_SHA,
        "geometry_parent_plan_sha256": GEOMETRY_PLAN_SHA,
        "geometry_parent_audit_sha256": GEOMETRY_AUDIT_SHA,
        "production_authentication": False, "new_fits": 0, "inherited_models": 12,
        "control_rows": 51, "diagnostic_roots": 0, "namespace": NAMESPACE,
        "fixture_source_sha256": plan["fixture_source_sha256"]})
    return MemoryFixture(folder, plan, path, digest, folder / "execution")


def validate_fixture(case):
    plan = case.plan
    study.require(all(type(plan[key]) is int and plan[key] > 0
                      for key in ("cap_seconds", "audit_cap_seconds")), "Positive integer fixture caps")
    study.require(plan["engineering"] is True and plan["rng_namespace"] == NAMESPACE,
                  "Explicit engineering fixture, never a scored namespace")
    study.require(study.sha(case.path) == case.digest and study.read(case.path) == plan,
                  "Fixture plan identity")
    study.require(study.sha(Path(__file__)) == plan["fixture_source_sha256"], "Fixture helper identity")
    protocol.validate_settings(plan, engineering=True)
    study.require(plan["runtime"] == study.base.runtime(), "Fixture runtime identity")
    _, cache_source, geometry_plan, geometry_source = parents()
    study.require(plan["sources"] == study.source_hashes(geometry_plan["sources"])
                  and plan["cache_source"] == cache_source and plan["geometry_source"] == geometry_source
                  and plan["random_stream_contract"] == study.stream_contract(plan),
                  "Fixture source, lineage and stream identity")
    return plan


def run_fixture(case):
    failed = case.root / "execution-preflight-failed.json"
    study.require(not case.execution.exists() and not failed.exists(), "Exclusive engineering execution")
    begin = time.monotonic()
    try:
        validate_fixture(case)
    except BaseException as error:
        study.write(failed, {"status": "failed", "phase": "engineering_execution_preflight",
            "scope": SCOPE, "error": repr(error), "plan_sha256": case.digest,
            "wall_seconds": time.monotonic() - begin, "new_model_calls": 0, "new_native_calls": 0})
        raise
    return study.run_validated(case.plan, case.digest, case.execution, begin=begin,
                               final_validate=lambda: validate_fixture(case))


def audit_fixture(case, out):
    failed = case.root / "audit-preflight-failed.json"
    study.require(not Path(out).exists() and not failed.exists(), "Exclusive engineering audit")
    try:
        validate_fixture(case)
        import audit_reacher_geometry_memory_study as auditor
    except BaseException as error:
        study.write(failed, {"status": "failed", "phase": "engineering_audit_preflight",
            "scope": SCOPE, "error": repr(error), "plan_sha256": case.digest,
            "new_model_calls": 0, "new_native_calls": 0})
        raise
    return auditor.audit_saved(case.plan, case.digest, case.execution, Path(out), engineering=True)
