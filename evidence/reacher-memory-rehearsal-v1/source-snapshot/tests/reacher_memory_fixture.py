"""Explicit whole-pipeline engineering fixture, never scored efficacy evidence.

Four fresh engineering episodes, twelve tiny fits and all 51 controller rows.
No historical checkpoint/performance is used. Only the historical lineage and
training-cohort identity boundaries are substituted during the fixture audit;
all saved-output, native replay, ordering and cost checks remain enabled.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reacher_memory_study as study

from openjev.research import reacher_memory_protocol as protocol
from openjev.research.robotics_reacher import collect_episode, native_replay

NAMESPACE = "reacher-memory-engineering-whole-tree-v1"
SOURCE_NAMESPACE = "reacher-memory-engineering-capacity-v1"
SCOPE = "synthetic_engineering_only_not_production_lineage_or_efficacy"


@dataclass(frozen=True)
class MemoryFixture:
    root: Path
    plan: dict
    path: Path
    digest: str
    execution: Path
    source_plan: dict
    objective_plan: dict


def prepare_fixture(folder, *, cap_seconds=600, audit_cap_seconds=600):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    source = folder / "synthetic-source"
    source.mkdir()
    np_prior, torch_prior, descriptors = study.prior_streams()
    source_plan = protocol.settings(engineering=True)
    source_plan.update(rng_namespace=SOURCE_NAMESPACE, prediction_episodes=4, control_episodes=1,
                      engineering_rng_namespaces=[name for name in protocol.ENGINEERING_NAMESPACES
                                                  if name != SOURCE_NAMESPACE])
    source_plan["random_stream_contract"] = protocol.stream_contract(source_plan, np_prior, torch_prior, descriptors)
    source_plan.update(engineering=True, fixture_scope=SCOPE)
    study.write(source / "collection-plan.json", source_plan)
    started, records = time.monotonic(), []
    for index, policy in enumerate(("ik_pd", "random_low", "random_high", "ik_pd")):
        record = collect_episode(protocol.seed(source_plan, f"prediction/reset/{index}"),
            protocol.schedule(source_plan, index, "ordinary", "prediction"), noise_std=.05,
            noise_seed=protocol.seed(source_plan, f"prediction/actuator_noise/{index}"),
            action_seed=protocol.seed(source_plan, f"prediction/exploration/{index}"), policy=policy)
        replay = native_replay(record)
        study.require(replay["transitions"] == 50 and replay["max_abs_error"] == 0, "Native fixture identity")
        records.append(record)
    study.base.save_records(source / "train", records)
    cohort = {"train_episodes": 4, "steps": 50, "noise_std": .05, "ordinary_gap": 6, "shift_gap": 10,
              "fixture_scope": SCOPE, "unique_native_records": 4, "collection_namespace": SOURCE_NAMESPACE}
    study.write(source / "training-plan.json", cohort)
    members = {name: study.sha(source / name) for name in ("train.npz", "train.json")}
    study.write(source / "training-receipt.json", {"status": "synthetic_fixture_only", "scope": SCOPE,
        "plan_sha256": study.sha(source / "training-plan.json"), "execution_members": members,
        "native_transitions": 200, "collection_and_replay_wall_seconds": time.monotonic() - started})
    # Bind historical metadata only, so prior generator exclusions remain real.
    objective = study.read(study.checked(study.ROOT / study.OBJECTIVE_PLAN, study.OBJECTIVE_PLAN_SHA))
    study.write(source / "objective-protocol-only.json", {"scope": SCOPE,
        "plan_path": study.OBJECTIVE_PLAN, "plan_sha256": study.OBJECTIVE_PLAN_SHA,
        "historical_weights_or_performance_loaded": False})
    plan = protocol.settings(engineering=True)
    plan.update(rng_namespace=NAMESPACE,
        engineering_rng_namespaces=[name for name in protocol.ENGINEERING_NAMESPACES if name != NAMESPACE],
        train_episodes=4, epochs=1, batch_size=2, hidden_size=4, mlp_width=7,
        control_episodes=1, prediction_episodes=2, bootstrap_samples=8,
        cap_seconds=cap_seconds, audit_cap_seconds=audit_cap_seconds,
        engineering=True, fixture_scope=SCOPE, fixture_source_sha256=study.sha(Path(__file__)),
        sources={name: study.sha(study.ROOT / name) for name in study.SOURCES}, runtime=study.base.runtime(),
        stop="Engineering fixture only; no scored namespace or efficacy claims.",
        training_source={"execution_path": str(source), "plan_path": str(source / "training-plan.json"),
            "plan_sha256": study.sha(source / "training-plan.json"),
            "audit_receipt_path": str(source / "training-receipt.json"),
            "audit_receipt_sha256": study.sha(source / "training-receipt.json"),
            "members": members, "cohort_plan": cohort},
        objective_source={"plan_path": study.OBJECTIVE_PLAN, "plan_sha256": study.OBJECTIVE_PLAN_SHA,
            "audit_path": str(source / "objective-protocol-only.json"),
            "audit_receipt_sha256": study.sha(source / "objective-protocol-only.json"),
            "prior_costs": {"cumulative_attempt_wall_seconds": 0.}, "training_weights_reused": False,
            "previous_scientific_gate_passed": False})
    plan["random_stream_contract"] = protocol.stream_contract(plan, np_prior, torch_prior, descriptors)
    protocol.validate_settings(plan, engineering=True)
    path = folder / "plan.json"
    study.write(path, plan)
    digest = study.sha(path)
    study.write(folder / "fixture.json", {"scope": SCOPE, "plan_sha256": digest,
        "production_authentication": False, "new_models": 12, "control_rows": 51,
        "training_rows": 4, "namespace": NAMESPACE, "source_namespace": SOURCE_NAMESPACE,
        "inherited_checkpoint_calls": 0})
    return MemoryFixture(folder, plan, path, digest, folder / "execution", source_plan, objective)


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
    source = plan["training_source"]
    study.checked(source["plan_path"], source["plan_sha256"])
    study.checked(source["audit_receipt_path"], source["audit_receipt_sha256"])
    for name, digest in source["members"].items():
        study.checked(Path(source["execution_path"]) / name, digest)
    prior = plan["objective_source"]
    study.checked(study.ROOT / prior["plan_path"], prior["plan_sha256"])
    study.checked(prior["audit_path"], prior["audit_receipt_sha256"])
    study.require(plan["random_stream_contract"] == study.stream_contract(plan), "Fixture stream identity")
    return plan


def run_fixture(case):
    begin = time.monotonic()
    validate_fixture(case)
    return study.run_validated(case.plan, case.digest, case.execution, begin=begin,
                               final_validate=lambda: validate_fixture(case))


def fixture_training_replay(cohort_plan, records):
    study.require(cohort_plan["fixture_scope"] == SCOPE and len(records) == 4, "Synthetic cohort scope")
    replays = [native_replay(record) for record in records]
    return {"transitions": sum(row["transitions"] for row in replays),
            "max_abs_error": max(row["max_abs_error"] for row in replays), "scope": SCOPE}


@contextmanager
def fixture_audit_boundary(case):
    import audit_reacher_memory_study as auditor

    validate_fixture(case)
    with patch.object(auditor, "validate_lineage", return_value=(case.plan["training_source"]["cohort_plan"],
                                                                case.objective_plan)), \
         patch.object(auditor, "audit_inherited_training", side_effect=fixture_training_replay):
        yield auditor


def audit_fixture(case, out):
    with fixture_audit_boundary(case) as auditor:
        return auditor.audit_saved(case.plan, case.digest, case.execution, Path(out), engineering=True)
