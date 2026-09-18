"""Callable whole-tree engineering fixture, never production evidence.

Importing this module performs no data collection, training or evaluation.
The caller explicitly invokes prepare_fixture(), run_fixture() and optionally
audit_fixture(). Four native-valid engineering records are repeated to 128
training rows. The nine new tiny models and their results are test artifacts,
not research efficacy observations or inherited learned checkpoints.
"""

from __future__ import annotations

import copy
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import reacher_objective_study as study

from openjev.research import reacher_objective_protocol as protocol
from openjev.research import reacher_search_protocol as search_protocol
from openjev.research.robotics_reacher import collect_episode, native_replay

NAMESPACE = "reacher-objective-engineering-whole-tree-v1"
SOURCE_NAMESPACE = "reacher-objective-engineering-capacity-v1"
SCOPE = "synthetic_engineering_only_not_production_lineage_or_efficacy"


@dataclass(frozen=True)
class ObjectiveFixture:
    root: Path
    plan: dict
    path: Path
    digest: str
    execution: Path
    baseline: dict
    source_plan: dict


def _write(path, value):
    search_protocol.write(path, value)
    return search_protocol.sha(path)


def prepare_fixture(folder, *, cap_seconds=600, audit_cap_seconds=600):
    """Collect only engineering records and prepare an exclusive synthetic plan.

    All native phases use 50-step episodes. No model is constructed here. The
    source namespace is distinct from the whole-tree evaluation namespace and
    both are included in the protocol's production engineering exclusions.
    Historical protocol metadata is read solely to retain real prior-stream
    exclusion checks; no historical corpus, checkpoint or efficacy is read.
    """
    require = search_protocol.require
    require(type(cap_seconds) is int and cap_seconds > 0 and type(audit_cap_seconds) is int
            and audit_cap_seconds > 0, "Positive fixture caps")
    require(NAMESPACE in protocol.ENGINEERING_NAMESPACES and SOURCE_NAMESPACE in protocol.ENGINEERING_NAMESPACES,
            "Fixture namespaces must be declared production exclusions")
    sources = {name: search_protocol.sha(ROOT / name) for name in study.SOURCES}
    fixture_source_sha256 = search_protocol.sha(Path(__file__))
    baseline = search_protocol.read(study.checked(ROOT / study.SEARCH_PLAN, study.SEARCH_PLAN_SHA))
    numpy_prior, torch_prior, descriptors = study.prior_streams()
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    source = folder / "synthetic-source"
    source.mkdir()

    source_plan = protocol.settings(engineering=True)
    source_plan.update(rng_namespace=SOURCE_NAMESPACE, prediction_episodes=4, control_episodes=1,
                       engineering_rng_namespaces=[name for name in protocol.ENGINEERING_NAMESPACES
                                                   if name != SOURCE_NAMESPACE])
    source_plan["random_stream_contract"] = protocol.stream_contract(source_plan, numpy_prior, torch_prior, descriptors)
    source_plan.update(engineering=True, fixture_scope=SCOPE)
    _write(source / "collection-plan.json", source_plan)
    started = time.monotonic()
    records = []
    for index, policy in enumerate(("ik_pd", "random_low", "random_high", "ik_pd")):
        record = collect_episode(protocol.seed(source_plan, f"prediction/reset/{index}"),
                                 protocol.schedule(source_plan, index, "ordinary", "prediction"),
                                 noise_std=.05,
                                 noise_seed=protocol.seed(source_plan, f"prediction/actuator_noise/{index}"),
                                 action_seed=protocol.seed(source_plan, f"prediction/exploration/{index}"),
                                 policy=policy)
        replay = native_replay(record)
        require(replay["transitions"] == 50 and replay["max_abs_error"] == 0,
                "Fixture source must replay exactly")
        records.append(record)
    rows = [copy.deepcopy(records[index % len(records)]) for index in range(128)]
    study.base.save_records(source / "train", rows)
    members = {name: search_protocol.sha(source / name) for name in ("train.npz", "train.json")}
    cohort = {"train_episodes": 128, "steps": 50, "noise_std": .05, "ordinary_gap": 6, "shift_gap": 10,
              "fixture_scope": SCOPE, "unique_native_records": 4, "copies_per_record": 32,
              "collection_namespace": SOURCE_NAMESPACE}
    cohort_digest = _write(source / "training-plan.json", cohort)
    source_receipt = {"status": "synthetic_fixture_only", "fixture_scope": SCOPE,
                      "plan_sha256": cohort_digest, "execution_members": members,
                      "unique_collection_native_transitions": 200,
                      "repeated_saved_training_transitions": 6400,
                      "collection_and_fixture_replay_wall_seconds": time.monotonic() - started}
    source_receipt_digest = _write(source / "training-receipt.json", source_receipt)
    baseline_fixture = {"status": "synthetic_fixture_only", "fixture_scope": SCOPE,
                        "baseline_protocol_path": study.SEARCH_PLAN,
                        "baseline_protocol_sha256": study.SEARCH_PLAN_SHA,
                        "historical_weights_or_performance_loaded": False}
    baseline_fixture_digest = _write(source / "baseline-protocol-only.json", baseline_fixture)

    plan = protocol.settings(engineering=True)
    plan.update(rng_namespace=NAMESPACE,
                engineering_rng_namespaces=[name for name in protocol.ENGINEERING_NAMESPACES if name != NAMESPACE],
                train_episodes=128, hidden_size=4, epochs=1, batch_size=32,
                control_episodes=1, prediction_episodes=2, bootstrap_samples=8,
                cap_seconds=cap_seconds, audit_cap_seconds=audit_cap_seconds,
                sources=sources, runtime=study.base.runtime(), engineering=True,
                fixture_scope=SCOPE, fixture_source_sha256=fixture_source_sha256,
                criterion={"objective_improvement": .05, "every_pair_nonworse": True,
                           "reset_increase": .05, "every_pair_reset_positive": True,
                           "versus_zero_improvement": .10, "physics_versus_zero_improvement": .10,
                           "panels": ["ordinary", "shift"], "expected_checks": 31},
                stop="Synthetic fixture only; exclusive artifacts; no production namespace, lineage or efficacy.",
                training_source={
                    "execution_path": str(source), "plan_path": str(source / "training-plan.json"),
                    "plan_sha256": cohort_digest, "audit_receipt_path": str(source / "training-receipt.json"),
                    "audit_receipt_sha256": source_receipt_digest, "members": members, "cohort_plan": cohort},
                search_source={
                    "plan_path": study.SEARCH_PLAN, "plan_sha256": study.SEARCH_PLAN_SHA,
                    "audit_path": str(source / "baseline-protocol-only.json"),
                    "audit_receipt_sha256": baseline_fixture_digest,
                    "prior_costs": {"cumulative_attempt_wall_seconds":
                                    source_receipt["collection_and_fixture_replay_wall_seconds"]},
                    "training_weights_reused": False})
    plan["random_stream_contract"] = protocol.stream_contract(plan, numpy_prior, torch_prior, descriptors)
    protocol.validate_settings(plan, engineering=True)
    path = folder / "plan.json"
    digest = _write(path, plan)
    _write(folder / "fixture.json", {"scope": SCOPE, "plan_sha256": digest,
                                     "production_authentication": False, "new_models": 9,
                                     "control_rows": 57, "training_rows": 128, "unique_training_records": 4,
                                     "namespace": NAMESPACE, "source_namespace": SOURCE_NAMESPACE,
                                     "inherited_checkpoint_calls": 0})
    return ObjectiveFixture(folder, plan, path, digest, folder / "execution", baseline, source_plan)


def validate_fixture(case):
    """Fixture byte/runtime checks; deliberately not production authentication."""
    require = search_protocol.require
    require(case.plan["engineering"] is True and case.plan["fixture_scope"] == SCOPE
            and case.plan["rng_namespace"] == NAMESPACE, "Explicit isolated engineering scope required")
    require(search_protocol.sha(case.path) == case.digest
            and search_protocol.read(case.path) == case.plan, "Fixture plan changed")
    require(search_protocol.sha(Path(__file__)) == case.plan["fixture_source_sha256"], "Fixture source changed")
    protocol.validate_settings(case.plan, engineering=True)
    require(case.plan["runtime"] == study.base.runtime(), "Fixture runtime changed")
    for name, digest in case.plan["sources"].items():
        study.checked(ROOT / name, digest)
    source = case.plan["training_source"]
    study.checked(source["plan_path"], source["plan_sha256"])
    study.checked(source["audit_receipt_path"], source["audit_receipt_sha256"])
    for name, digest in source["members"].items():
        study.checked(Path(source["execution_path"]) / name, digest)
    np_prior, torch_prior, descriptors = study.prior_streams()
    require(case.plan["random_stream_contract"] == protocol.stream_contract(case.plan, np_prior, torch_prior, descriptors),
            "Fixture named stream contract changed")
    return case.plan


def run_fixture(case):
    """Explicit caller-owned engineering execution; this function is never auto-run."""
    begin = time.monotonic()
    validate_fixture(case)
    return study.run_validated(case.plan, case.digest, case.execution, begin=begin,
                               final_validate=lambda: validate_fixture(case))


@contextmanager
def fixture_audit_boundary(case):
    """Substitute only synthetic lineage/old training-seed semantics.

    Source/runtime/streams, calibration, initialization, nine fits, restoration,
    every control and prediction, proposal traces, native replay and scientific
    checks remain the actual auditor implementations. The inherited-training
    hook replaces the old unique-seed formula with exact byte identity and
    native replay of every repeated fixture row, never a replay bypass.
    """
    import audit_reacher_objective_study as auditor

    def lineage(plan, execution):
        validate_fixture(case)
        search_protocol.require(plan == case.plan and Path(execution) == case.execution,
                                "Fixture audit cannot authorize another plan or execution")
        source = plan["training_source"]
        search_protocol.require(search_protocol.read(Path(execution) / "train-provenance.json") == source,
                                "Synthetic training provenance changed")
        for name, digest in source["members"].items():
            study.checked(Path(execution) / name, digest)
        source_receipt = search_protocol.read(source["audit_receipt_path"])
        search_protocol.require(source_receipt["status"] == "synthetic_fixture_only"
                                and source_receipt["fixture_scope"] == SCOPE,
                                "Synthetic receipt cannot stand in for a production audit")
        baseline = search_protocol.read(study.checked(ROOT / study.SEARCH_PLAN, study.SEARCH_PLAN_SHA))
        search_protocol.require(baseline == case.baseline, "Baseline protocol metadata changed")
        return source["cohort_plan"], baseline

    def inherited_training(cohort, records):
        search_protocol.require(cohort == case.plan["training_source"]["cohort_plan"]
                                and len(records) == 128, "Synthetic inherited corpus identity/count")
        maximum = 0.
        for record in records:
            auditor.check_budget()
            result = native_replay(record)
            search_protocol.require(result["transitions"] == 50 and result["new_policy_calls"] == 0
                                    and result["saved_output_only"] is True,
                                    "Synthetic inherited corpus must replay every row")
            maximum = max(maximum, result["max_abs_error"])
        return {"episodes": len(records), "transitions": len(records) * 50, "max_abs_error": maximum}

    with (patch.object(auditor, "validate_lineage", lineage),
          patch.object(auditor, "audit_inherited_training", inherited_training)):
        yield auditor


def audit_fixture(case, out=None):
    """Run the actual saved-output auditor with its explicit engineering switch."""
    out = case.root / "audit" if out is None else Path(out)
    with fixture_audit_boundary(case) as auditor:
        return auditor.audit_saved(case.plan, case.digest, case.execution, out, engineering=True)
