"""Prospective zero-fit Reacher score-intervention protocol.

No artifact reads, model construction, native calls or RNG draws occur while
settings/manifests are built. The enclosing runner authenticates all inherited
checkpoints, the exposed cache-study histories and complete prior registries.
Engineering fixtures are separate namespaces and may only shrink stated sizes.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from openjev.research import reacher_cache_protocol as cache_protocol
from openjev.research import reacher_search_protocol as search_protocol
from openjev.research.reacher_adaptive_search import SearchInputs
from openjev.research.reacher_geometry_reward import geometry_reward_configuration
from openjev.research.reacher_random_streams import generator_manifest

STUDY = "reacher-geometry-score-v1"
SCORED_NAMESPACE = "reacher-geometry-score-v1-scored"
ENGINEERING_NAMESPACES = (
    "reacher-geometry-engineering-unit-v1",
    "reacher-geometry-engineering-runner-v1",
    "reacher-geometry-engineering-capacity-v1",
    "reacher-geometry-engineering-whole-tree-v1",
)
BASE_STUDY = "reacher-cache-ablation-v1"
BASE_PLAN_SHA256 = "7868daa12242df37f020946f9d3b279811a0e97547eef4d8b179da6e9394cefa"
BASE_AUDIT_SHA256 = "d1a6e486fde8823f5a760af8036e79af4fc57b452bc5fa0c8d184093ce2a1790"
ARMS = ("residual_gru", "cached_mlp")
PAIRS = ("pair0", "pair1", "pair2")
SCORE_MODES = ("learned", "geometry")
PANELS = ("full", "ordinary", "shift")
REFERENCES = ("known_state", "particle", "zero", "uniform", "public_kinematic")
MODEL_CLASSES = {arm: cache_protocol.MODEL_CLASSES[arm] for arm in ARMS}
INPUT_NAMES, INPUT_COUNTS = search_protocol.INPUT_NAMES, search_protocol.INPUT_COUNTS
require = search_protocol.require
save_inputs, load_inputs = search_protocol.save_inputs, search_protocol.load_inputs

CRITERION = {
    "metric": "negative_total_native_reward_lower_is_better",
    "primary_arm": "residual_gru",
    "treatment_score": "geometry",
    "control_score": "learned",
    "gap_panels": ["ordinary", "shift"],
    "geometry_vs_own_learned": {"mean_improvement": .03, "every_pair_nonworse": True},
    "full_sensing_vs_own_learned": {"maximum_mean_degradation": .02},
    "competence": {
        "score_modes": list(SCORE_MODES), "every_pair_on_each_gap": True,
        "gru_vs_zero_improvement": .10,
        "references": ["known_state", "particle"], "reference_vs_zero_improvement": .10,
        "public_kinematic": "descriptive_only",
    },
    "expected_checks": 25,
    "all_checks_required": True,
    "cached_mlp_comparisons": "secondary_only; never rescue a failed primary gate",
    "diagnostic": "exposed_development_histories; descriptive_only; never rescue a failed primary gate",
    "pass_interpretation": "Approximate geometric scoring improves the same saved recurrent models on this task under the prespecified margins.",
    "fail_interpretation": "The required scoring improvement was not established; failure does not prove equivalent scorers or identify the remaining model error.",
    "limits": "This changes planning scores, not architecture or weights. It establishes no recurrent, biological, calibration or independent-confirmation novelty.",
}
SECONDARY_CONTRASTS = [
    {"name": "geometry_gru_vs_learned_cached_mlp", "treatment": ["residual_gru", "geometry"],
     "control": ["cached_mlp", "learned"]},
    {"name": "geometry_gru_vs_geometry_cached_mlp", "treatment": ["residual_gru", "geometry"],
     "control": ["cached_mlp", "geometry"]},
    {"name": "cached_mlp_geometry_vs_own_learned", "treatment": ["cached_mlp", "geometry"],
     "control": ["cached_mlp", "learned"]},
]


def _copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def settings(*, engineering=False):
    """Fresh proposed constants; no seed derivation, generators or artifact I/O."""
    require(type(engineering) is bool, "Explicit engineering flag")
    namespace = ENGINEERING_NAMESPACES[0] if engineering else SCORED_NAMESPACE
    plan = {
        "study": STUDY, "version": 1, "rng_namespace": namespace,
        "engineering_rng_namespaces": [name for name in ENGINEERING_NAMESPACES if name != namespace],
        "arms": list(ARMS), "pairs": list(PAIRS), "score_modes": list(SCORE_MODES),
        "panels": list(PANELS), "references": list(REFERENCES), "model_classes": dict(MODEL_CLASSES),
        "new_fits": 0, "new_optimizer_updates": 0, "weights_unchanged": True,
        "control_episodes": 64, "steps": 50, "dt": .02, "noise_std": .05,
        "hidden_size": 64, "mlp_width": 107, "residual_reward": True,
        "ordinary_gap": 6, "shift_gap": 10, "threads": 2, "bootstrap_samples": 4096,
        "planner": "cem256", "candidates": 64, "planning_horizon": 12, "action_block": 3,
        "particles": 32, "filter_bandwidth": .02,
        "search_parameters": {
            "elites": 8, "minimum_std": .001, "reward_clip": [-2.5, 0.],
            "callback_sizes": {"cem256": [64, 64, 64, 64]}, "warm_start": False,
            "elite_carryover": False, "proposal_momentum": 0.,
        },
        "score_contract": {
            "learned": "Unchanged model reward output, including its existing actuator-cost accounting; no geometry computation or replacement.",
            "geometry": geometry_reward_configuration(.05),
            "common": "Same saved model/class/public root and action-conditioned transitions; replace only per-step scalar score before identical clipping and float32 accumulation.",
            "public_target": "Static target from the actual public root packet only; never native state or future measurements.",
            "diagnostics": "Retain original predicted reward/angles; geometry mode additionally retains approximate distance, action cost, pair norms and joint-limit diagnostics. No joint-limit correction or penalty.",
            "approximation": "Final-qpos FK is not exact RK4 cached-body reward; projected mean-angle distance is not expected distance under actuator uncertainty.",
        },
        "search_pairing": "Initial proposal banks and innovation arrays are shared across scoring modes/fits/panels; later CEM proposals adapt to each mode's scores.",
        "diagnostic_cases": 8, "diagnostic_root_steps": [12, 32], "diagnostic_branches": 4,
        "diagnostic_contract": {
            "history_study": BASE_STUDY, "history_fit": "residual_gru-pair0",
            "history_scope": "Previous completed cache-study exposed development trajectories, first cases in each panel; never the fresh control rows of this study.",
            "copied_episode_files": "Copy and hash-bind complete parent episode files with all64 cases; select first diagnostic_cases and fixed root steps only.",
            "public_prefix": "Assimilate actual saved packets through root t and issued actions through t-1, under each exact inherited model class; no future packets or native state enter models.",
            "root_native_state": "Native qpos/qvel/integration/time only for physical replay, restored to the saved root; never passed to learned scorers.",
            "initial_candidates": 64, "selected_candidates": 12, "maximum_identity_slots": 76,
            "slot_order": "Common initial bank0..63, then fit_manifest order with learned then geometry selections.",
            "deduplication": "Exact float32 command bytes, first occurrence; retain all76 identity slots and slot-to-unique mapping, replay each unique sequence only.",
            "noise": "Four fresh independent actuator-noise branches per root, each shared by every unique sequence; never copied source or fresh-control disturbances.",
            "comparison": "All twelve model/scoring combinations score the common finite union; native means/ranks and selected-sequence regret are descriptive within this union, not global optima.",
            "terminal": "Absolute50-step TimeLimit state restored; fixed roots12/32 both have full12-step horizon.",
            "scope": "Exposed-history diagnostic only, outside the primary gate; no tuning between diagnostic and fresh control.",
        },
        "stage_order": ["authenticate_and_restore_all_six", "exposed_history_diagnostic", "fresh_closed_loop_control"],
        "reset_interventions": False, "criterion": _copy(CRITERION),
        "secondary_contrasts": _copy(SECONDARY_CONTRASTS),
        "cost_reporting": {
            "setup": "Source/checkpoint authentication, actual-class restoration and nominal-physics construction charged; inherited training costs reported separately.",
            "diagnostic": "Every model rollout, score component, union selection, native noise branch and trace/hash cost; actual unique-sequence counts plus maximum bounds.",
            "control": "All6savedfits x2scores x3panels plus15references. Charge all model.advance heads even when reward output is discarded; geometry arithmetic is additional work.",
            "timing": "Whole shared-host wall time plus components; equal candidate counts do not imply equal total compute or isolated latency.",
        },
    }
    plan["fit_order"] = [row["name"] for row in fit_manifest(plan)]
    plan["execution_order"] = execution_order(plan)
    return plan


def fit_manifest(plan):
    require(plan["arms"] == list(ARMS) and plan["pairs"] == list(PAIRS), "All six inherited fits required")
    return [{"name": f"{arm}-{pair}", "arm": arm, "pair": pair, "model_class": MODEL_CLASSES[arm],
             "weights_reused": True} for pair in PAIRS for arm in ARMS]


def policy_name(fit_name, score_mode):
    require(fit_name in {f"{arm}-{pair}" for arm in ARMS for pair in PAIRS}
            and score_mode in SCORE_MODES, "Known inherited fit and scoring mode")
    return f"{fit_name}--{score_mode}"


def execution_order(plan):
    require(plan["panels"] == list(PANELS) and plan["references"] == list(REFERENCES)
            and plan["score_modes"] == list(SCORE_MODES), "All scoring modes/panels/references required")
    rows = []
    for panel in PANELS:
        for fit in fit_manifest(plan):
            for mode in SCORE_MODES:
                label = policy_name(fit["name"], mode)
                rows.append({"panel": panel, "fit": fit["name"], "arm": fit["arm"], "pair": fit["pair"],
                             "score_mode": mode, "planner": "cem256", "label": label,
                             "path": f"control/{panel}/{label}"})
        rows.extend({"panel": panel, "reference": name, "label": name,
                     "path": f"control/{panel}/{name}"} for name in REFERENCES)
    return rows


def diagnostic_manifest(plan):
    require(plan["panels"] == list(PANELS), "All diagnostic panels required")
    count = plan["diagnostic_cases"]
    roots = plan["diagnostic_root_steps"]
    require(type(count) is int and 0 < count <= 8 and isinstance(roots, list)
            and all(type(step) is int for step in roots) and roots in ([12], [12, 32]),
            "First bounded diagnostic cases and fixed root prefix required")
    return [{"panel": panel, "case_index": case, "step": step,
             "history_fit": "residual_gru-pair0", "source_path": f"control/{panel}/residual_gru-pair0",
             "path": f"diagnostic/{panel}/{case:03d}/{step:03d}",
             "input_prefix": f"planner/diagnostic/{panel}/{case}/{step}", "horizon": 12}
            for panel in PANELS for case in range(count) for step in roots]


def coverage(plan):
    fit_names = [row["name"] for row in fit_manifest(plan)]
    rows, diagnostic = execution_order(plan), diagnostic_manifest(plan)
    maximum_slots = 64 + len(fit_names) * len(SCORE_MODES)
    searches = len(diagnostic) * len(fit_names) * len(SCORE_MODES)
    return {"fit_names": fit_names, "inherited_fits": len(fit_names), "new_fits": 0, "new_optimizer_updates": 0,
            "control_paths": [row["path"] for row in rows], "control_rows": len(rows),
            "learned_control_rows": len(fit_names) * len(SCORE_MODES) * len(PANELS),
            "reference_rows": len(REFERENCES) * len(PANELS), "control_cases_per_row": plan["control_episodes"],
            "native_control_transitions": len(rows) * plan["control_episodes"] * plan["steps"],
            "diagnostic_roots": len(diagnostic), "diagnostic_paths": [row["path"] for row in diagnostic],
            "diagnostic_identity_slots_per_root": maximum_slots,
            "diagnostic_unique_sequences_per_root_maximum": maximum_slots,
            "diagnostic_model_searches": searches,
            "diagnostic_search_candidate_evaluations": searches * 256,
            "diagnostic_search_imagined_transitions": searches * 256 * 12,
            "diagnostic_union_score_calls": searches,
            "diagnostic_union_candidate_evaluations_maximum": searches * maximum_slots,
            "diagnostic_union_imagined_transitions_maximum": searches * maximum_slots * 12,
            "diagnostic_native_branches_maximum": len(diagnostic) * maximum_slots * plan["diagnostic_branches"],
            "diagnostic_native_transitions_maximum": len(diagnostic) * maximum_slots * plan["diagnostic_branches"] * 12,
            "criterion_checks": 25}


def validate_settings(plan, *, engineering=False):
    expected = settings(engineering=engineering)
    require(isinstance(plan, dict) and set(expected) <= set(plan), "Complete geometry-study settings")
    dimensions = {"control_episodes", "diagnostic_cases", "hidden_size", "mlp_width", "threads", "bootstrap_samples"}
    flexible = {"rng_namespace", "engineering_rng_namespaces"} | (dimensions | {"diagnostic_root_steps"} if engineering else set())
    for key, value in expected.items():
        if key not in flexible:
            require(_identity(plan[key]) == _identity(value), f"Geometry setting changed: {key}")
    for key in dimensions:
        require(type(plan[key]) is int and 0 < plan[key] <= expected[key], f"Bounded positive integer: {key}")
    require(plan["diagnostic_root_steps"] in ([12], [12, 32]) if engineering
            else plan["diagnostic_root_steps"] == [12, 32], "Fixed diagnostic root prefix")
    namespace = plan["rng_namespace"]
    require(namespace in (ENGINEERING_NAMESPACES if engineering else (SCORED_NAMESPACE,)),
            "Declared engineering/scored namespace required")
    require(plan["engineering_rng_namespaces"] == [name for name in ENGINEERING_NAMESPACES if name != namespace],
            "Exact engineering namespace exclusions")
    require(plan["fit_order"] == [row["name"] for row in fit_manifest(plan)]
            and plan["execution_order"] == execution_order(plan), "Exact fit/control order")
    diagnostic_manifest(plan)
    return plan


def _namespace(plan):
    value = plan["rng_namespace"]
    require(value in (SCORED_NAMESPACE, *ENGINEERING_NAMESPACES), "Declared geometry RNG namespace")
    return value


def registry(plan):
    namespace = _namespace(plan)
    n = plan["control_episodes"]
    require(type(n) is int and 0 < n <= 64 and type(plan["steps"]) is int and plan["steps"] == 50,
            "Bounded fresh controls and fixed terminal boundary")
    roles = [f"control/{purpose}/{i}" for purpose in ("reset", "actuator_noise", "sensor_schedule") for i in range(n)]
    roles += [f"planner/particle_filter/{i}" for i in range(n)]
    roles += ["floor/uniform/0", "analysis/bootstrap/0", "diagnostic/native_template/0"]
    roles += [f"planner/control/{step}/{name}" for step in range(50) for name in INPUT_NAMES]
    for row in diagnostic_manifest(plan):
        roles += [f"{row['input_prefix']}/{name}" for name in INPUT_NAMES]
        roles += [f"diagnostic/branch_noise/{row['panel']}/{row['case_index']}/{row['step']}/{branch}"
                  for branch in range(plan["diagnostic_branches"])]
    require(plan["diagnostic_branches"] == 4 and len(roles) == len(set(roles)), "Four independent branches and unique roles")
    return {role: search_protocol.named_seed(namespace, role) for role in roles}


def _validated_registry(value, label):
    require(isinstance(value, dict) and all(isinstance(name, str) and name.strip()
            and type(seed) is int and 0 <= seed < 2**64 for name, seed in value.items()), f"{label} role/uint64 registry")
    return dict(value)


def _states(manifest):
    return {value["initial_state_sha256"] for value in manifest.values()}


def cache_stream_exclusions():
    """Automatic full-size cache exclusions; caller authenticates source lineage."""
    result = []
    for namespace in (cache_protocol.SCORED_NAMESPACE, *cache_protocol.ENGINEERING_NAMESPACES):
        old = cache_protocol.settings()
        old["rng_namespace"] = namespace
        numpy_registry, torch_registry = cache_protocol.registry(old), cache_protocol.torch_registry(old)
        result.append({"namespace": namespace,
            "coverage": {"control_episodes": 64, "prediction_episodes": 96, "steps": 50,
                         "arms": list(cache_protocol.ARMS), "pairs": list(PAIRS)},
            "registry": numpy_registry, "generators": generator_manifest(numpy_registry),
            "torch_registry": torch_registry, "torch_generators": cache_protocol.torch_generator_manifest(torch_registry)})
    return result


def stream_contract(plan, prior_registries=(), prior_torch_registries=(), prior_plans=()):
    """Manifest concrete NumPy states without sampling; no new Torch streams.

    Supplied registries/descriptors must cover all authenticated earlier studies
    and literal engineering seeds. Automatic exclusions additionally cover every
    full-size cache and memory namespace. This helper does not read provenance.
    """
    validate_settings(plan, engineering=_namespace(plan) in ENGINEERING_NAMESPACES)
    current = registry(plan)
    generators = generator_manifest(current)
    require(len(set(current.values())) == len(current) and len(_states(generators)) == len(generators),
            "Current root/generator-state collision")
    prior_np = [_validated_registry(value, "Prior NumPy") for value in prior_registries]
    prior_torch = [_validated_registry(value, "Prior Torch") for value in prior_torch_registries]
    exclusions = []
    for namespace in plan["engineering_rng_namespaces"]:
        full = settings()
        full["rng_namespace"] = namespace
        values = registry(full)
        exclusions.append({"namespace": namespace,
            "coverage": {"control_episodes": 64, "diagnostic_cases": 8, "diagnostic_root_steps": [12, 32],
                         "diagnostic_branches": 4, "steps": 50},
            "registry": values, "generators": generator_manifest(values)})
    cache_exclusions = cache_stream_exclusions()
    memory_exclusions = cache_protocol.memory_stream_exclusions()
    for label, entries in (("Engineering", exclusions), ("Cache-study", cache_exclusions), ("Memory-study", memory_exclusions)):
        for item in entries:
            require(not set(current.values()) & set(item["registry"].values()), f"{label} root collision")
            require(not _states(generators) & _states(item["generators"]), f"{label} generator-state collision")
    for values in prior_np:
        require(not set(current.values()) & set(values.values()), "Prior NumPy root collision")
        require(not _states(generators) & _states(generator_manifest(values)), "Prior NumPy generator-state collision")
    return {"namespace": _namespace(plan), "registry": current, "generators": generators,
            "torch_registry": {}, "torch_generators": {}, "new_torch_scored_streams": 0,
            "training_and_stochastic_inference_rng_draws": 0,
            "discarded_constructor_rng": {
                "calls": 6, "seed": 0, "outer_rng_restored": True,
                "all_tensors_overwritten_from_authenticated_checkpoints": True,
                "scope": "Existing excluded seed reused only inside isolated actual-class constructors; these discarded draws are not training or inference randomness.",
            },
            "engineering_exclusions": exclusions, "cache_study_exclusions": cache_exclusions,
            "memory_study_exclusions": memory_exclusions,
            "prior_numpy_registry_sha256": [_identity(value) for value in prior_np],
            "prior_torch_registry_sha256": [_identity(value) for value in prior_torch],
            "priors": _copy(list(prior_plans)), "draws_for_manifest": 0, "full_horizon_innovations": True,
            "unused_random_extra": "Generated and saved for immutable SearchInputs compatibility; never scored by CEM256.",
            "intentional_reuse": [
                "All controller/mode/panel rows share corresponding fresh resets, actuator noise and sensor phase.",
                "Each decision shares initial proposals and CEM innovations; adapted later proposals may differ.",
                "Every exposed diagnostic root shares its proposals across all6models x2modes.",
                "Each of4 diagnostic noise branches is shared by all unique sequences at that root; branches/roots/control disturbances are distinct.",
                "Public-history rows are copied from the completed exposed cache study, with no source disturbance replay as new branch noise.",
                "Particle-filter children remain independent by purpose; bootstrap/uniform commands are paired.",
            ],
            "provenance_limit": "Caller authenticates complete prior artifacts and literal engineering streams; no artifact I/O or scored samples here."}


def seed(plan, role):
    expected = registry(plan)
    require(isinstance(role, str) and role in expected, "Known geometry NumPy role required")
    bound = plan["random_stream_contract"]
    require(bound["namespace"] == _namespace(plan), "Bound RNG namespace differs")
    value = bound["registry"][role]
    require(type(value) is int and value == expected[role], "Bound seed differs from role-derived seed")
    return value


def phase(plan, index, split="control"):
    require(split == "control" and type(index) is int and 0 <= index < plan["control_episodes"],
            "Fresh control index only; diagnostic schedule is inherited")
    return int(np.random.default_rng(seed(plan, f"control/sensor_schedule/{index}")).integers(0, 4))


def schedule(plan, index, panel, split="control"):
    require(panel in PANELS and plan["steps"] == 50 and plan["ordinary_gap"] == 6 and plan["shift_gap"] == 10,
            "Fixed public sensing and terminal boundary")
    require(split == "control" and type(index) is int and 0 <= index < plan["control_episodes"],
            "Fresh control index only; diagnostic schedule is inherited")
    result = np.ones(51, dtype=bool)
    if panel != "full":
        offset = phase(plan, index, split)
        length = plan["shift_gap"] if panel == "shift" else plan["ordinary_gap"]
        for start in (8 + offset, 28 + offset):
            result[start:start + length] = False
    return result


def _draw_inputs(plan, prefix, cases):
    require(plan["planning_horizon"] == 12 and plan["action_block"] == 3, "Fixed full search horizon/block")
    seeds = [seed(plan, f"{prefix}/{name}") for name in INPUT_NAMES]
    values = [np.random.default_rng(value).normal(size=(cases, count, 4, 2))
              for value, count in zip(seeds, INPUT_COUNTS, strict=True)]
    return SearchInputs(values[0], values[1], tuple(values[2:]))


def draw_control_inputs(plan, step):
    require(type(step) is int and 0 <= step < plan["steps"] == 50, "No draw beyond terminal boundary")
    return _draw_inputs(plan, f"planner/control/{step}", plan["control_episodes"])


def _diagnostic_root(plan, panel, case_index, step):
    require(type(case_index) is int and type(step) is int, "Integer diagnostic root identity")
    rows = diagnostic_manifest(plan)
    return next((row for row in rows if row["panel"] == panel and row["case_index"] == case_index
                 and row["step"] == step), None)


def diagnostic_inputs(plan, panel, case_index, step):
    row = _diagnostic_root(plan, panel, case_index, step)
    require(row is not None, "Preselected exposed root required")
    return _draw_inputs(plan, row["input_prefix"], 1)


def diagnostic_noise(plan, panel, case_index, step):
    require(_diagnostic_root(plan, panel, case_index, step) is not None, "Preselected exposed root required")
    require(plan["diagnostic_branches"] == 4 and plan["planning_horizon"] == 12 and plan["noise_std"] == .05,
            "Four fixed12-step native noise branches")
    seeds = [seed(plan, f"diagnostic/branch_noise/{panel}/{case_index}/{step}/{branch}") for branch in range(4)]
    return np.stack([np.random.default_rng(value).normal(0., plan["noise_std"], size=(12, 2)) for value in seeds])


def common_bank(inputs, step, plan):
    """Same anchored arithmetic for control batches and one-case diagnostic roots."""
    require(isinstance(inputs, SearchInputs), "SearchInputs required")
    cases = inputs.initial.shape[0]
    require(cases in (1, plan["control_episodes"]), "Control or one-root proposal batch required")
    return cache_protocol.common_bank(inputs, step, {**plan, "control_episodes": cases})


def diagnostic_union(plan, bank, selected):
    """Deduplicate finite float32 command bytes in fixed identity-slot order.

    All slots are retained for provenance; only unique commands need native
    replay. Byte-exact dedup deliberately distinguishes signed zero. Inputs are
    copied; the returned arrays do not alias caller-owned buffers.
    """
    labels = [policy_name(row["name"], mode) for row in fit_manifest(plan) for mode in SCORE_MODES]
    require(isinstance(selected, dict) and set(selected) == set(labels), "All12 selected model/mode sequences")
    arrays = [np.asarray(bank), *[np.asarray(selected[label]) for label in labels]]
    require(arrays[0].shape == (64, 12, 2) and all(value.shape == (12, 2) for value in arrays[1:]),
            "Full12-step initial bank and selected sequences")
    require(all(value.dtype == np.float32 and np.isfinite(value).all() and (np.abs(value) <= 1).all()
                for value in arrays), "Finite float32 issued commands in [-1,1]")
    slots = np.concatenate((arrays[0], np.stack(arrays[1:])), axis=0)
    mapping, first, unique, lookup = [], [], [], {}
    for index, commands in enumerate(slots):
        key = commands.tobytes(order="C")
        if key not in lookup:
            lookup[key] = len(unique)
            first.append(index)
            unique.append(commands.copy())
        mapping.append(lookup[key])
    return {"slot_commands": slots, "commands": np.stack(unique),
            "slot_to_unique": np.asarray(mapping, dtype=np.int64),
            "unique_first_slots": np.asarray(first, dtype=np.int64),
            "slot_ids": [f"common/{index}" for index in range(64)] + [f"selected/{label}" for label in labels]}
