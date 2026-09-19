"""Prospective geometry-scored memory comparison; not a freeze or execution.

Settings and manifests perform no I/O, model/native calls or sampling. Stream
manifests allocate isolated generators without drawing. The future runner must
still authenticate every source, lineage artifact and supplied prior registry.
Only caller-triggered sampling functions consume the chosen named streams.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from openjev.research import reacher_cache_protocol as cache_protocol
from openjev.research import reacher_geometry_protocol as geometry_protocol
from openjev.research import reacher_search_protocol as search_protocol
from openjev.research.reacher_adaptive_search import SearchInputs
from openjev.research.reacher_geometry_reward import geometry_reward_configuration
from openjev.research.reacher_random_streams import generator_manifest

STUDY = "reacher-geometry-memory-v1"
SCORED_NAMESPACE = "reacher-geometry-memory-v1-scored"
ENGINEERING_NAMESPACES = (
    "reacher-geometry-memory-engineering-unit-v1",
    "reacher-geometry-memory-engineering-runner-v1",
    "reacher-geometry-memory-engineering-capacity-v1",
    "reacher-geometry-memory-engineering-whole-tree-v1",
)
ARMS = ("residual_gru", "encoded_current_gru", "cached_gru", "cached_mlp")
PAIRS = ("pair0", "pair1", "pair2")
PANELS = ("full", "ordinary", "shift")
REFERENCES = ("known_state", "particle", "public_kinematic", "zero", "uniform")
PHYSICS_REFERENCES = REFERENCES[:3]
SCORE_MODES = ("geometry",)
MODEL_CLASSES = {arm: cache_protocol.MODEL_CLASSES[arm] for arm in ARMS}
INPUT_NAMES, INPUT_COUNTS = search_protocol.INPUT_NAMES, search_protocol.INPUT_COUNTS
require = search_protocol.require
save_inputs, load_inputs = search_protocol.save_inputs, search_protocol.load_inputs

CACHE_PLAN_SHA256 = "7868daa12242df37f020946f9d3b279811a0e97547eef4d8b179da6e9394cefa"
CACHE_AUDIT_SHA256 = "d1a6e486fde8823f5a760af8036e79af4fc57b452bc5fa0c8d184093ce2a1790"
GEOMETRY_PLAN_SHA256 = "94c90f7585303f4edd88e1b0cb0dc8a488a2e2f07237aae4cfb489ade2068a2b"
GEOMETRY_AUDIT_SHA256 = "9cd5fdde29601ae5227afb9b3725be1fc855d4c7ba72aa7ca4c83248ae8de72e"
INHERITED_SOURCE_MANIFEST_SHA256 = "6900f21002dfc6dac3b86a6d8566fcf30303cab34a5b6fd8d88ed4163e20eb5a"
# Canonical JSON hashes of the completed geometry plan's ordered lists. These
# bind older numerical registries, not merely their descriptive labels/counts.
PRIOR_BINDINGS = {
    "numpy_count": 26,
    "numpy_hash_list_sha256": "d1af13a89c64eb02889427d81871381831580301c20109cfc73f9ca652efc47e",
    "torch_count": 18,
    "torch_hash_list_sha256": "c6fef8d2c5b5d84f95a9396a73ac6e184b1dc4797518c95740263683b4908a3f",
    "descriptors_sha256": "e0991438b61fb266193c5f8af649526c3747d0ffd2e7c0fd1e3b65aec55111e2",
}
NEW_SOURCES = (
    "src/openjev/research/reacher_geometry_memory_protocol.py",
    "tests/test_reacher_geometry_memory_protocol.py",
    "src/openjev/research/reacher_geometry_memory_control.py",
    "tests/test_reacher_geometry_memory_control.py",
    "src/openjev/research/reacher_geometry_physics.py",
    "tests/test_reacher_geometry_physics.py",
    "scripts/reacher_geometry_memory_study.py",
    "tests/test_reacher_geometry_memory_study.py",
    "scripts/audit_reacher_geometry_memory_study.py",
    "tests/test_audit_reacher_geometry_memory_study.py",
)
FIT_MEMBERS = ("initial-weights.pt", "training.jsonl", "weights.pt", "checkpoint.pt", "completed.json")
ENGINEERING_LITERALS = (
    {"source_path": "tests/test_reacher_geometry_memory_control.py",
     "numpy_registry": {"controller/innovations": 410, "controller/native_reset": 410,
                        "controller/native_actuator_noise": 410},
     "torch_registry": {"controller/global_fixture": 410, "controller/model_construction": 410}},
    {"source_path": "tests/test_reacher_geometry_physics.py",
     "numpy_registry": {"physics/innovations": 410, "physics/native_reset": 410},
     "torch_registry": {}},
    {"source_path": "tests/test_reacher_geometry_memory_study.py",
     "numpy_registry": {"runner/innovations": 410, "runner/native_reset": 410,
                        "runner/native_actuator_noise": 410},
     "torch_registry": {"runner/model_construction": 410}},
    {"source_path": "tests/test_audit_reacher_geometry_memory_study.py",
     "numpy_registry": {"auditor/innovations": 410, "auditor/native_reset": 410},
     "torch_registry": {}},
)
PLAN_METADATA_FIELDS = frozenset((
    "engineering", "sources", "runtime", "source_commit", "cap_seconds", "audit_cap_seconds",
    "cache_source", "geometry_source", "random_stream_contract", "fixture_source_sha256",
))
CRITERION = {
    "metric": "negative_total_native_reward_lower_is_better",
    "persistent_arm": "residual_gru",
    "score_mode": "geometry",
    "gap_panels": ["ordinary", "shift"],
    "persistent_vs_reset_gru": {
        "comparators": ["encoded_current_gru", "cached_gru"],
        "mean_improvement": .03, "every_pair_nonworse": True,
    },
    "full_sensing_vs_reset_gru": {
        "comparators": ["encoded_current_gru", "cached_gru"],
        "maximum_mean_degradation": .02,
    },
    "competence": {
        "persistent_vs_zero_improvement": .10, "every_pair_on_each_gap": True,
        "reference": "known_state", "reference_panel": "ordinary",
        "reference_vs_zero_improvement": .10,
        "particle_and_public_kinematic": "descriptive_only",
    },
    "expected_checks": 25, "all_checks_required": True,
    "secondary_cannot_rescue_primary": True,
    "pass_interpretation": "This trained persistent real-observation update policy improves control beyond both trained reset-GRU policies under the same geometry interface and candidate budget.",
    "fail_interpretation": "The required persistent-policy advantage was not established; failure is not evidence of equivalence or that all memory is unnecessary.",
    "limits": "Three inherited fit pairs and one task family; real-assimilation gating differs. No isolated velocity, Bayesian, connectome, independent-confirmation or architecture-novelty claim.",
}
SECONDARY_CONTRASTS = [
    {"name": "persistent_vs_cached_mlp", "treatment": "residual_gru", "control": "cached_mlp"},
    {"name": "gru_cache_information", "treatment": "cached_gru", "control": "encoded_current_gru"},
    {"name": "cached_gru_vs_cached_mlp", "treatment": "cached_gru", "control": "cached_mlp"},
]


def _copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def lineage_contract():
    """Pinned completed context, distinct from this future, unrun study."""
    return {
        "cache": {"study": "reacher-cache-ablation-v1",
            "plan_path": "evidence/reacher-cache-ablation-v1/protocol/plan.json", "plan_sha256": CACHE_PLAN_SHA256,
            "audit_path": "evidence/reacher-cache-ablation-v1/audit/receipt.json", "audit_receipt_sha256": CACHE_AUDIT_SHA256,
            "role": "All twelve original checkpoint directories, paired initializations/orders and training-cohort identities.",
            "historical_gate_passed": False},
        "geometry": {"study": "reacher-geometry-score-v1",
            "plan_path": "evidence/reacher-geometry-score-v1/protocol/plan.json", "plan_sha256": GEOMETRY_PLAN_SHA256,
            "audit_path": "evidence/reacher-geometry-score-v1/audit/receipt.json", "audit_receipt_sha256": GEOMETRY_AUDIT_SHA256,
            "role": "Completed score-intervention evidence, immutable80-source dependency contract and complete earlier RNG lineage.",
            "historical_gate_passed": True},
        "engineering_parent": "An explicit completed engineering parent may replace checkpoint provenance only in an engineering runner; it does not replace the pinned full-size RNG exclusions or establish efficacy.",
        "restore": "Authenticate all12 before any fresh cohort/innovation draw; exact classes and deployment tensors only; no trainer or optimizer construction. Snapshot tensors before/after execution.",
        "pairing": "Three GRU arms share original initial tensors, orders and updates within pair, but separately trained final weights and real-assimilation rules differ. MLP initialization is separately named.",
    }


def source_contract():
    """Required future membership, never a filesystem/readiness assertion."""
    return {"inherited_count": 80, "inherited_manifest_sha256": INHERITED_SOURCE_MANIFEST_SHA256,
            "inherited_plan_sha256": GEOMETRY_PLAN_SHA256, "new_sources": list(NEW_SOURCES),
            "expected_total": 90, "future_runner": NEW_SOURCES[6], "future_auditor": NEW_SOURCES[8],
            "validation": "Caller supplies current file hashes; exact old80 manifest plus all10 new paths required. Missing future files block freeze. No source I/O or readiness claim by this module."}


def source_membership(inherited_sources):
    require(isinstance(inherited_sources, dict) and len(inherited_sources) == 80
            and all(isinstance(name, str) and _digest(value) for name, value in inherited_sources.items())
            and _identity(inherited_sources) == INHERITED_SOURCE_MANIFEST_SHA256,
            "Exact completed geometry eighty-source manifest required")
    require(not set(inherited_sources) & set(NEW_SOURCES), "New sources must be additive")
    return tuple(sorted((*inherited_sources, *NEW_SOURCES)))


def validate_source_hashes(sources, inherited_sources):
    expected = source_membership(inherited_sources)
    require(isinstance(sources, dict) and set(sources) == set(expected), "Complete future ninety-source membership required")
    require(all(_digest(value) for value in sources.values()), "Lowercase SHA256 source digests required")
    require(all(sources[name] == digest for name, digest in inherited_sources.items()), "Frozen inherited source changed")
    return sources


def settings(*, engineering=False):
    """Return independent constants; no source reads, derived seeds or draws."""
    require(type(engineering) is bool, "Explicit engineering flag")
    namespace = ENGINEERING_NAMESPACES[0] if engineering else SCORED_NAMESPACE
    plan = {
        "study": STUDY, "version": 1, "rng_namespace": namespace,
        "engineering_rng_namespaces": [name for name in ENGINEERING_NAMESPACES if name != namespace],
        "arms": list(ARMS), "pairs": list(PAIRS), "model_classes": dict(MODEL_CLASSES),
        "panels": list(PANELS), "references": list(REFERENCES), "score_modes": list(SCORE_MODES),
        "new_fits": 0, "new_optimizer_updates": 0, "weights_unchanged": True,
        "control_episodes": 64, "steps": 50, "dt": .02, "noise_std": .05,
        "hidden_size": 64, "mlp_width": 107, "residual_reward": True,
        "ordinary_gap": 6, "shift_gap": 10, "threads": 2, "bootstrap_samples": 4096,
        "planner": "cem256", "physics_planner": "cem256", "candidates": 64,
        "planning_horizon": 12, "action_block": 3, "particles": 32, "filter_bandwidth": .02,
        "search_parameters": {"elites": 8, "minimum_std": .001, "reward_clip": [-2.5, 0.],
            "callback_sizes": {"cem256": [64, 64, 64, 64]}, "warm_start": False,
            "elite_carryover": False, "proposal_momentum": 0.},
        "score_contract": {
            "geometry": geometry_reward_configuration(.05),
            "accumulation": "Per-step clip[-2.5,0], sequential float32 summation; truncate at absolute50-step terminal. Expected actuator cost charged once.",
            "model": "Original transitions, observation decoder and learned reward head still execute; only geometry selects commands. Preserve original reward predictions.",
            "public_target": "Static target from actual public root packet only; never realized future disturbances or measurements.",
            "physics": "Nominal native transition from each reference's supplied state, decoded per-step angle features, same approximate geometry/expected-action score and CEM256 arithmetic. No aggregate native-cost clipping shortcut.",
            "approximation": "Final-qpos FK is not exact RK4 cached-body reward; projected mean-angle distance is not expected distance under actuator noise.",
        },
        "reference_contract": {
            "known_state": "Privileged current qpos/qvel; no future noise; nominal physics with CEM256 and geometry.",
            "particle": "Public packets/actions only for state estimate; supplied nominal physics with CEM256 and geometry.",
            "public_kinematic": "Last two valid public measurements, elapsed time and issued actions only; supplied nominal physics with CEM256 and geometry.",
            "zero": "Zero issued command; no search or geometry computation.",
            "uniform": "Shared named uniform-command floor; no search or geometry computation.",
            "comparison": "Same candidate budget and scoring objective, not equal CPU cost; native physics and model uncertainty assumptions still differ.",
        },
        "feature_contract": {key: _copy(value) for key, value in cache_protocol.FEATURE_CONTRACT.items()},
        "search_pairing": "All families/physics/panels share reset, actuator-noise, phase and proposal-innovation streams. Later CEM proposals are score-adaptive and can differ; initial64 proposals coincide.",
        "exposed_diagnostic_roots": 0, "new_prediction_episodes": 0, "reset_interventions": False,
        "stage_order": ["authenticate_and_restore_all_twelve", "fresh_closed_loop_control"],
        "criterion": _copy(CRITERION), "secondary_contrasts": _copy(SECONDARY_CONTRASTS),
        "lineage_contract": lineage_contract(), "source_contract": source_contract(),
        "engineering_literal_contract": _copy(ENGINEERING_LITERALS),
        "cost_reporting": {
            "comparison": "Candidate-budget matched, not total-compute matched. Report all points, whole rows/decisions and complete lifecycle wall time; no isolated-latency or FLOP claim.",
            "learned": "All model heads, geometry, cache validation/copying, selected re-advance, native stepping and trace I/O remain charged.",
            "physics": "All256 nominal candidate trajectories, geometry/filter/observer and reset costs charged; include integration substeps separately.",
            "inherited": "No new fitting; retain inherited training and failed-attempt costs separately without double counting prior cumulative lineages.",
        },
    }
    plan["fit_order"] = [row["name"] for row in fit_manifest(plan)]
    plan["execution_order"] = execution_order(plan)
    return plan


def fit_manifest(plan):
    require(plan["arms"] == list(ARMS) and plan["pairs"] == list(PAIRS), "All twelve inherited fits required")
    return [{"name": f"{arm}-{pair}", "arm": arm, "pair": pair, "model_class": MODEL_CLASSES[arm],
             "weights_reused": True} for pair in PAIRS for arm in ARMS]


def restore_manifest(plan):
    """Exact original checkpoint identities; no selected old control trajectories."""
    return [{**fit, "source_study": "reacher-cache-ablation-v1",
             "source_path": f"fits/{fit['name']}", "copy_path": f"inherited/fits/{fit['name']}",
             "members": [f"fits/{fit['name']}/{member}" for member in FIT_MEMBERS],
             "initialization_path": f"initializations/{fit['pair']}.pt",
             "order_path": f"orders/{fit['pair']}.pt", "new_optimizer_updates": 0}
            for fit in fit_manifest(plan)]


def inherited_members(plan):
    names = {"train.npz", "train.json"}
    for pair in PAIRS:
        names.update(f"{folder}/{pair}.{suffix}" for folder in ("initializations", "orders") for suffix in ("pt", "json"))
    for fit in restore_manifest(plan):
        names.update(fit["members"])
    return sorted(names)


def execution_order(plan):
    require(plan["panels"] == list(PANELS) and plan["references"] == list(REFERENCES)
            and plan["score_modes"] == list(SCORE_MODES), "Exact panels/references and geometry-only scoring")
    rows = []
    for panel in PANELS:
        for fit in fit_manifest(plan):
            rows.append({"panel": panel, "fit": fit["name"], "arm": fit["arm"], "pair": fit["pair"],
                         "score_mode": "geometry", "planner": "cem256", "label": fit["name"],
                         "path": f"control/{panel}/{fit['name']}"})
        for name in REFERENCES:
            rows.append({"panel": panel, "reference": name, "label": name,
                         "planner": "cem256" if name in PHYSICS_REFERENCES else None,
                         "score_mode": "geometry" if name in PHYSICS_REFERENCES else None,
                         "path": f"control/{panel}/{name}"})
    return rows


def criterion_manifest(plan):
    """Exact check identities and cost multipliers; auditor rederives arithmetic."""
    require(plan["criterion"] == CRITERION, "Exact proposed twenty-five-check gate")
    checks = []
    for comparator in ("encoded_current_gru", "cached_gru"):
        for panel in ("ordinary", "shift"):
            checks.append({"name": f"gap_mean/{panel}/{comparator}", "panel": panel,
                           "treatment": "residual_gru", "control": comparator,
                           "aggregation": "three_fit_mean", "maximum_cost_multiplier": .97})
            for pair in PAIRS:
                checks.append({"name": f"gap_pair/{panel}/{comparator}/{pair}", "panel": panel,
                               "treatment": f"residual_gru-{pair}", "control": f"{comparator}-{pair}",
                               "aggregation": "paired_fit", "maximum_cost_multiplier": 1.})
        checks.append({"name": f"full_mean/{comparator}", "panel": "full",
                       "treatment": "residual_gru", "control": comparator,
                       "aggregation": "three_fit_mean", "maximum_cost_multiplier": 1.02})
    for panel in ("ordinary", "shift"):
        for pair in PAIRS:
            checks.append({"name": f"competence/{panel}/residual_gru-{pair}", "panel": panel,
                           "treatment": f"residual_gru-{pair}", "control": "zero",
                           "aggregation": "fit_vs_reference", "maximum_cost_multiplier": .9})
    checks.append({"name": "competence/ordinary/known_state", "panel": "ordinary",
                   "treatment": "known_state", "control": "zero",
                   "aggregation": "reference_vs_reference", "maximum_cost_multiplier": .9})
    require(len(checks) == 25 and len({row["name"] for row in checks}) == 25, "Unique complete gate")
    return checks


def coverage(plan):
    fits, rows = fit_manifest(plan), execution_order(plan)
    cases, steps = plan["control_episodes"], plan["steps"]
    horizon_sum = sum(min(plan["planning_horizon"], steps - step) for step in range(steps))
    learned, physics, floors = 36, 9, 6
    return {"fit_names": [fit["name"] for fit in fits], "inherited_fits": 12,
            "new_fits": 0, "new_optimizer_updates": 0, "inherited_members": len(inherited_members(plan)),
            "control_paths": [row["path"] for row in rows], "control_rows": len(rows),
            "learned_control_rows": learned, "physics_reference_rows": physics, "floor_rows": floors,
            "reference_rows": physics + floors, "control_cases_per_row": cases,
            "native_control_transitions": len(rows) * cases * steps,
            "learned_candidate_evaluations": learned * cases * steps * 256,
            "physics_candidate_evaluations": physics * cases * steps * 256,
            "learned_imagined_transitions": learned * cases * 256 * horizon_sum,
            "physics_nominal_transitions": physics * cases * 256 * horizon_sum,
            "learned_selected_advances": learned * cases * steps,
            "physics_selected_advances": physics * cases * steps,
            "geometry_candidate_samples": (learned + physics) * cases * 256 * horizon_sum,
            "geometry_selected_samples": (learned + physics) * cases * steps,
            "physics_candidate_native_substeps": 2 * physics * cases * 256 * horizon_sum,
            "physics_selected_native_substeps": 2 * physics * cases * steps,
            "exposed_diagnostic_roots": 0, "new_prediction_episodes": 0,
            "criterion_checks": len(criterion_manifest(plan))}


def validate_settings(plan, *, engineering=False):
    expected = settings(engineering=engineering)
    require(isinstance(plan, dict) and set(expected) <= set(plan), "Complete geometry-memory settings required")
    require(set(plan) <= set(expected) | PLAN_METADATA_FIELDS, "Unknown or obsolete study setting")
    dimensions = {"control_episodes", "hidden_size", "mlp_width", "threads", "bootstrap_samples"}
    flexible = {"rng_namespace", "engineering_rng_namespaces"} | (dimensions if engineering else set())
    for key, value in expected.items():
        if key not in flexible:
            require(_identity(plan[key]) == _identity(value), f"Geometry-memory setting changed: {key}")
    for key in dimensions:
        require(type(plan[key]) is int and 0 < plan[key] <= expected[key], f"Bounded positive integer: {key}")
    namespace = plan["rng_namespace"]
    require(namespace in (ENGINEERING_NAMESPACES if engineering else (SCORED_NAMESPACE,)), "Declared engineering/scored namespace required")
    require(plan["engineering_rng_namespaces"] == [name for name in ENGINEERING_NAMESPACES if name != namespace], "Exact engineering namespace exclusions")
    if "engineering" in plan:
        require(plan["engineering"] is engineering, "Plan engineering flag differs from validation mode")
    require(plan["fit_order"] == [fit["name"] for fit in fit_manifest(plan)]
            and plan["execution_order"] == execution_order(plan), "Exact fit/restore/control order required")
    return plan


def _namespace(plan):
    value = plan["rng_namespace"]
    require(value in (SCORED_NAMESPACE, *ENGINEERING_NAMESPACES), "Declared geometry-memory RNG namespace")
    return value


def registry(plan):
    namespace, count = _namespace(plan), plan["control_episodes"]
    require(type(count) is int and 0 < count <= 64 and type(plan["steps"]) is int and plan["steps"] == 50,
            "Bounded fresh controls and fixed terminal boundary")
    roles = [f"control/{purpose}/{index}" for purpose in ("reset", "actuator_noise", "sensor_schedule") for index in range(count)]
    roles += [f"planner/particle_filter/{index}" for index in range(count)]
    roles += ["floor/uniform/0", "analysis/bootstrap/0"]
    roles += [f"planner/control/{step}/{name}" for step in range(50) for name in INPUT_NAMES]
    require(len(roles) == len(set(roles)), "Unique stream roles")
    return {role: search_protocol.named_seed(namespace, role) for role in roles}


def _registry(value, label):
    require(isinstance(value, dict) and all(isinstance(name, str) and name.strip()
            and type(seed_value) is int and 0 <= seed_value < 2**64 for name, seed_value in value.items()), f"{label} named uint64 registry")
    return dict(value)


def validate_prior_registries(prior_registries, prior_torch_registries, prior_plans):
    """Reject absent, altered, reordered or unknown historical registries."""
    require(isinstance(prior_registries, (list, tuple)) and isinstance(prior_torch_registries, (list, tuple))
            and isinstance(prior_plans, (list, tuple)), "Explicit ordered prior registries and descriptors required")
    previous = [_registry(value, "Prior NumPy") for value in prior_registries]
    torch_previous = [_registry(value, "Prior Torch") for value in prior_torch_registries]
    hashes = [_identity(value) for value in previous]
    torch_hashes = [_identity(value) for value in torch_previous]
    require(len(hashes) == PRIOR_BINDINGS["numpy_count"]
            and _identity(hashes) == PRIOR_BINDINGS["numpy_hash_list_sha256"], "Missing, changed or unknown prior NumPy registry")
    require(len(torch_hashes) == PRIOR_BINDINGS["torch_count"]
            and _identity(torch_hashes) == PRIOR_BINDINGS["torch_hash_list_sha256"], "Missing, changed or unknown prior Torch registry")
    require(_identity(list(prior_plans)) == PRIOR_BINDINGS["descriptors_sha256"], "Missing, changed or unknown prior descriptor")
    return previous, torch_previous


def _states(manifest):
    return {value["initial_state_sha256"] for value in manifest.values()}


def geometry_stream_exclusions():
    """All previous geometry roots, including full-size engineering diagnostics."""
    result = []
    for namespace in (geometry_protocol.SCORED_NAMESPACE, *geometry_protocol.ENGINEERING_NAMESPACES):
        previous = geometry_protocol.settings()
        previous["rng_namespace"] = namespace
        values = geometry_protocol.registry(previous)
        result.append({"namespace": namespace, "registry": values, "generators": generator_manifest(values),
                       "coverage": {"control_episodes": 64, "diagnostic_cases": 8, "diagnostic_root_steps": [12, 32],
                                    "diagnostic_branches": 4, "steps": 50}})
    return result


def engineering_literal_exclusions():
    """Known consumed test literals, separately attributed despite old reuse.

    Native reset/noise use the declared NumPy PCG64 initial states. Multiple
    engineering purposes deliberately reused410; this is not scientific pairing.
    The enclosing90-source manifest binds these exact test files by hash.
    """
    return [{**_copy(item), "generators": generator_manifest(item["numpy_registry"]),
             "torch_generators": cache_protocol.torch_generator_manifest(item["torch_registry"]),
             "source_hash_binding": "plan.sources[source_path]",
             "scope": "Known literal engineering generators only; no claim about unrelated ambient RNG."}
            for item in ENGINEERING_LITERALS]


def stream_contract(plan, prior_registries=(), prior_torch_registries=(), prior_plans=()):
    """No draws; complete pinned earlier registries are mandatory even in tests.

    The caller authenticates old artifacts before supplying their exact maps.
    Newly consumed literal seeds/undeclared namespaces require a prospective
    contract extension before freeze, never an unrecorded extra or omission.
    """
    validate_settings(plan, engineering=_namespace(plan) in ENGINEERING_NAMESPACES)
    previous, torch_previous = validate_prior_registries(prior_registries, prior_torch_registries, prior_plans)
    current = registry(plan)
    generators = generator_manifest(current)
    require(len(set(current.values())) == len(current) and len(_states(generators)) == len(generators), "Current root/generator-state collision")
    exclusions = []
    # Engineering tests also exclude the future full scored namespace without
    # consuming it. Production excludes all four full engineering namespaces.
    for namespace in (SCORED_NAMESPACE, *ENGINEERING_NAMESPACES):
        if namespace == _namespace(plan):
            continue
        full = settings()
        full["rng_namespace"] = namespace
        values = registry(full)
        exclusions.append({"namespace": namespace, "registry": values, "generators": generator_manifest(values),
                           "coverage": {"control_episodes": 64, "steps": 50}})
    geometry = geometry_stream_exclusions()
    cache = geometry_protocol.cache_stream_exclusions()
    memory = cache_protocol.memory_stream_exclusions()
    literals = engineering_literal_exclusions()
    for label, entries in (("Own namespace", exclusions), ("Geometry", geometry), ("Cache", cache), ("Memory", memory)):
        for item in entries:
            require(not set(current.values()) & set(item["registry"].values()), f"{label} root collision")
            require(not _states(generators) & _states(item["generators"]), f"{label} generator-state collision")
    for values in previous:
        require(not set(current.values()) & set(values.values()), "Prior NumPy root collision")
        require(not _states(generators) & _states(generator_manifest(values)), "Prior NumPy generator-state collision")
    for item in literals:
        require(not set(current.values()) & set(item["numpy_registry"].values()), "Engineering literal root collision")
        require(not _states(generators) & _states(item["generators"]), "Engineering literal generator-state collision")
    return {"namespace": _namespace(plan), "registry": current, "generators": generators,
            "torch_registry": {}, "torch_generators": {}, "new_torch_scored_streams": 0,
            "training_and_stochastic_inference_rng_draws": 0,
            "discarded_constructor_rng": {"calls": 12, "seed": 0, "outer_rng_restored": True,
                "all_tensors_overwritten_from_authenticated_checkpoints": True,
                "scope": "Existing excluded seed reused inside isolated actual-class constructors only; no training or stochastic inference."},
            "namespace_exclusions": exclusions, "geometry_study_exclusions": geometry,
            "cache_study_exclusions": cache, "memory_study_exclusions": memory,
            "engineering_literal_exclusions": literals,
            "prior_bindings": _copy(PRIOR_BINDINGS),
            "prior_numpy_registry_sha256": [_identity(value) for value in previous],
            "prior_torch_registry_sha256": [_identity(value) for value in torch_previous],
            "priors": _copy(list(prior_plans)), "draws_for_manifest": 0, "full_horizon_innovations": True,
            "unused_random_extra": "Generated and saved for immutable SearchInputs compatibility; never scored by CEM256.",
            "intentional_reuse": [
                "All learned/physics/floor/panel rows share fresh resets, actuator noise and sensing phases.",
                "All CEM256 searches share initial proposals and innovations, not later score-adaptive proposals.",
                "Particle children separate initial/process/resample roles; uniform commands and bootstrap cases are paired.",
            ],
            "provenance_limit": "Caller authenticates pinned lineage and each source file. Unknown additional consumed seeds require a prospective contract extension; these settings do not authorize or freeze execution."}


def seed(plan, role):
    expected = registry(plan)
    require(isinstance(role, str) and role in expected, "Known geometry-memory NumPy role required")
    bound = plan.get("random_stream_contract")
    require(isinstance(bound, dict) and bound.get("namespace") == _namespace(plan)
            and bound.get("registry") == expected, "Complete bound RNG registry differs")
    return expected[role]


def phase(plan, index, split="control"):
    require(split == "control" and type(index) is int and 0 <= index < plan["control_episodes"], "Fresh control index only")
    return int(np.random.default_rng(seed(plan, f"control/sensor_schedule/{index}")).integers(0, 4))


def schedule(plan, index, panel, split="control"):
    require(panel in PANELS and plan["steps"] == 50 and plan["ordinary_gap"] == 6 and plan["shift_gap"] == 10,
            "Fixed public sensing panels and terminal boundary")
    require(split == "control" and type(index) is int and 0 <= index < plan["control_episodes"], "Fresh control index only")
    valid = np.ones(51, dtype=bool)
    if panel != "full":
        offset = phase(plan, index, split)
        length = plan["shift_gap"] if panel == "shift" else plan["ordinary_gap"]
        for start in (8 + offset, 28 + offset):
            valid[start:start + length] = False
    return valid


def draw_control_inputs(plan, step):
    require(type(step) is int and 0 <= step < plan["steps"] == 50, "No innovation draw beyond terminal boundary")
    require(plan["planning_horizon"] == 12 and plan["action_block"] == 3, "Fixed search horizon/block")
    seeds = [seed(plan, f"planner/control/{step}/{name}") for name in INPUT_NAMES]
    values = [np.random.default_rng(value).normal(size=(plan["control_episodes"], count, 4, 2))
              for value, count in zip(seeds, INPUT_COUNTS, strict=True)]
    return SearchInputs(values[0], values[1], tuple(values[2:]))


def common_bank(inputs, step, plan):
    """Shared initial64 bank only; all model/physics decisions still require CEM256."""
    return cache_protocol.common_bank(inputs, step, plan)
