"""Retained full-study rehearsal using authenticated old engineering parents.

No work runs at import. Preparation authenticates metadata and generator
identities only. Execution fits three tiny history models and exercises all42
control rows; the independent audit verifies saved outputs separately. No
production checkpoint, new scored namespace, or scientific efficacy claim.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path

import reacher_geometry_memory_study as previous
import reacher_two_observation_study as study

from openjev.research import reacher_two_observation_protocol as protocol
from openjev.research import reacher_two_observation_streams as streams
from openjev.research.reacher_random_streams import generator_manifest

CACHE = "output/reacher-cache-rehearsal-v1/attempt-01"
CACHE_PLAN = "c56f6b73c19b0c3ac17bf55a0f1ac95b2513e2688561d52907cb150ff57c1279"
CACHE_AUDIT = "77853eb8c8166651054f5e2fef185744809cf686a5c0fa8f7415b3782d9b6e34"
PREREQUISITE = "output/reacher-geometry-memory-rehearsal-v1/attempt-01"
PREREQUISITE_PLAN = "5fdfe0061777e60337c7b4c17780f30c062a7172637ebbac8a2f6ed4dbd4931d"
PREREQUISITE_AUDIT = "4793f27b9839239abd48c063dfe0b56a2f63581fed9d14cb09ece7cab84779ee"
NAMESPACE = "reacher-two-observation-engineering-whole-tree-v1"
SCOPE = "tiny_engineering_full_study_not_scientific_performance"
EXTRA_SOURCE = "output/reacher-two-observation-control-v1/integrate_training.py"


@dataclass(frozen=True)
class HistoryFixture:
    root: Path
    path: Path
    digest: str
    execution: Path


def lineage_refs():
    return {
        "cache": {"plan_path": f"{CACHE}/plan.json", "plan_sha256": CACHE_PLAN,
                  "audit_path": f"{CACHE}/audit/receipt.json", "audit_receipt_sha256": CACHE_AUDIT,
                  "execution_path": f"{CACHE}/execution"},
        "prerequisite": {"plan_path": f"{PREREQUISITE}/plan.json", "plan_sha256": PREREQUISITE_PLAN,
                         "audit_path": f"{PREREQUISITE}/audit/receipt.json",
                         "audit_receipt_sha256": PREREQUISITE_AUDIT,
                         "execution_path": f"{PREREQUISITE}/execution",
                         "terminal_path": f"{PREREQUISITE}/audit/receipt.json",
                         "terminal_sha256": PREREQUISITE_AUDIT,
                         "terminal_kind": "engineering_completed_audit"},
    }


def historical_registries():
    numpy, torch, descriptors = previous.prior_streams()
    known = copy.deepcopy(streams.REQUIRED_LITERAL_CALLS)
    known["tests/test_reacher_two_observation_training_audit.py"] = {
        "numpy": {}, "torch": {"synthetic_historical_orders": 410}}
    known["tests/test_audit_reacher_two_observation_study.py"] = {
        "numpy": {}, "torch": {"delegated_synthetic_historical_orders": 410}}
    known["scripts/reacher_two_observation_study.py"] = {
        "numpy": {}, "torch": {"inherited_overwritten_constructor": 0,
                                  "history_overwritten_constructor": 410}}
    known[EXTRA_SOURCE] = {"numpy": {}, "torch": {
        "delegated_seed410_fixture": 410}}
    literals = []
    for name, roles in sorted(known.items()):
        literals.append({"source_path": name, "source_sha256": study.sha(study.ROOT / name),
            "numpy_registry": roles["numpy"], "numpy_generators": generator_manifest(roles["numpy"]),
            "torch_registry": roles["torch"],
            "torch_generators": streams.torch_generator_manifest(roles["torch"])})
    helper = Path(__file__).resolve().relative_to(study.ROOT).as_posix()
    return {"numpy": numpy, "torch": torch, "descriptors": descriptors,
            "literal_calls": literals,
            "engineering_sources": {helper: study.sha(__file__), EXTRA_SOURCE: study.sha(study.ROOT / EXTRA_SOURCE)}}


def prepare_fixture(folder, *, cap_seconds, audit_cap_seconds):
    """Explicit engineering caps; never reused as measured scientific limits."""
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    try:
        study.write(folder / "preparation-started.json", {"scope": SCOPE,
            "helper_sha256": study.sha(__file__), "requested_execution_cap_seconds": repr(cap_seconds),
            "requested_audit_cap_seconds": repr(audit_cap_seconds)})
        study.require(all(type(v) is int and v > 0 for v in (cap_seconds, audit_cap_seconds)),
                      "Explicit positive engineering caps")
        parent = study.read(study.checked(study.ROOT / CACHE / "plan.json", CACHE_PLAN))
        settings = protocol.settings(engineering=True)
        settings.update(rng_namespace=NAMESPACE, control_episodes=1, bootstrap_samples=8,
            **{key: parent[key] for key in ("hidden_size", "mlp_width", "train_episodes", "epochs", "batch_size")})
        protocol.validate_settings(settings, engineering=True)
        history = historical_registries()
        from openjev.research import reacher_two_observation_experiment as experiment
        plan = experiment.prepare(study.ROOT, settings, lineage_refs=lineage_refs(),
            historical_registries=history, runtime=study.base.runtime(), cap_seconds=cap_seconds,
            audit_cap_seconds=audit_cap_seconds, engineering_evidence={}, engineering=True)
        for name, digest in {**plan["sources"], **history["engineering_sources"]}.items():
            study.search_study.copy_verified(study.ROOT / name, folder / "source-snapshot" / name,
                                             digest, float("inf"))
        path = folder / "plan.json"
        study.write(path, plan)
        expected = study.sha(path)
        experiment.validate_plan(plan, expected, root=study.ROOT, runtime=study.base.runtime(),
                                 engineering=True, plan_bytes=path.read_bytes())
        study.write(folder / "preparation-completed.json", {"status": "prepared_not_run", "scope": SCOPE,
            "plan_sha256": expected, "namespace": NAMESPACE, "new_model_calls": 0, "new_native_calls": 0,
            "planned_new_fits": 3, "inherited_fits": 6, "control_rows": 42,
            "wall_seconds": time.monotonic() - begin})
        return HistoryFixture(folder, path, expected, folder / "execution")
    except BaseException as error:
        study.write(folder / "preparation-failed.json", {"status": "failed", "scope": SCOPE,
            "error": repr(error), "wall_seconds": time.monotonic() - begin,
            "new_model_calls": 0, "new_native_calls": 0, "automatic_retry": False})
        raise


def run_fixture(case):
    return study.run(case.path, case.digest, case.execution, engineering=True)


def audit_fixture(case, out):
    import audit_reacher_two_observation_study as auditor
    return auditor.audit_plan(case.path, case.digest, case.execution, Path(out), engineering=True)
