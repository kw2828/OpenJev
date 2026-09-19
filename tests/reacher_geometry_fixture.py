"""Reuse completed tiny cache checkpoints for an explicit geometry rehearsal.

No new fits, scored histories, scored streams or efficacy claims. Every model,
score, panel and reference remains present; only case/root counts are smaller.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import reacher_geometry_study as study

from openjev.research import reacher_geometry_protocol as protocol

PARENT = "output/reacher-cache-rehearsal-v1/attempt-01"
PLAN_SHA = "c56f6b73c19b0c3ac17bf55a0f1ac95b2513e2688561d52907cb150ff57c1279"
AUDIT_SHA = "77853eb8c8166651054f5e2fef185744809cf686a5c0fa8f7415b3782d9b6e34"
NAMESPACE = "reacher-geometry-engineering-whole-tree-v1"
SCOPE = "reused_completed_engineering_cache_checkpoints_only_not_scored_efficacy"


@dataclass(frozen=True)
class GeometryFixture:
    root: Path
    plan: dict
    path: Path
    digest: str
    execution: Path


def parent():
    return study.authenticate_parent(f"{PARENT}/plan.json", PLAN_SHA, f"{PARENT}/audit/receipt.json",
        AUDIT_SHA, f"{PARENT}/execution", engineering=True)


def prepare_fixture(folder, *, cap_seconds=600, audit_cap_seconds=600):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    previous, source = parent()
    plan = protocol.settings(engineering=True)
    plan.update(rng_namespace=NAMESPACE,
        engineering_rng_namespaces=[name for name in protocol.ENGINEERING_NAMESPACES if name != NAMESPACE],
        control_episodes=1, diagnostic_cases=1, diagnostic_root_steps=[12], bootstrap_samples=8,
        hidden_size=previous["hidden_size"], mlp_width=previous["mlp_width"],
        parent_source=source, cap_seconds=cap_seconds, audit_cap_seconds=audit_cap_seconds,
        sources={name: study.sha(study.ROOT / name) for name in study.SOURCES}, runtime=study.base.runtime(),
        engineering=True, fixture_scope=SCOPE, fixture_source_sha256=study.sha(Path(__file__)),
        stop="Engineering rehearsal only; no scored namespace, weights, histories or efficacy claims.")
    plan["random_stream_contract"] = study.stream_contract(plan)
    protocol.validate_settings(plan, engineering=True)
    path = folder / "plan.json"
    study.write(path, plan)
    digest = study.sha(path)
    study.write(folder / "fixture.json", {"scope": SCOPE, "plan_sha256": digest,
        "parent_plan_sha256": PLAN_SHA, "parent_audit_sha256": AUDIT_SHA,
        "production_authentication": False, "new_fits": 0, "inherited_models": 6,
        "control_rows": 51, "diagnostic_roots": 3, "namespace": NAMESPACE})
    return GeometryFixture(folder, plan, path, digest, folder / "execution")


def validate_fixture(case):
    plan = case.plan
    study.require(plan["engineering"] is True and plan["fixture_scope"] == SCOPE
                  and plan["rng_namespace"] == NAMESPACE, "Explicit fixture scope")
    study.require(study.sha(case.path) == case.digest and study.read(case.path) == plan, "Fixture plan identity")
    study.require(study.sha(Path(__file__)) == plan["fixture_source_sha256"], "Fixture helper identity")
    protocol.validate_settings(plan, engineering=True)
    study.require(plan["runtime"] == study.base.runtime(), "Fixture runtime identity")
    for name, digest in plan["sources"].items():
        study.checked(study.ROOT / name, digest)
    _, source = parent()
    study.require(plan["parent_source"] == source and plan["random_stream_contract"] == study.stream_contract(plan),
                  "Fixture lineage and stream identity")
    return plan


def run_fixture(case):
    begin = time.monotonic()
    validate_fixture(case)
    return study.run_validated(case.plan, case.digest, case.execution, begin=begin,
                               final_validate=lambda: validate_fixture(case))


def audit_fixture(case, out):
    import audit_reacher_geometry_study as auditor

    validate_fixture(case)
    return auditor.audit_saved(case.plan, case.digest, case.execution, Path(out), engineering=True)
