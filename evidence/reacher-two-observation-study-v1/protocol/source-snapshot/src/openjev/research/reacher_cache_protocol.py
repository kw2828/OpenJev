"""Prospective protocol helpers for the five-arm Reacher explicit-cache comparison.

This is an engineering component, not a freeze or scored execution. The runner
authenticates inherited artifacts and supplies prior stream registries. This
module never reads those artifacts, trains models or selects study outcomes.
RNG manifests allocate isolated generators but draw no samples. Sampling tests
use only the declared engineering namespaces.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager

import numpy as np
import torch

from openjev.research import reacher_memory_protocol as memory_protocol
from openjev.research import reacher_search_protocol as search_protocol
from openjev.research.reacher_adaptive_search import ANCHORS, SearchInputs
from openjev.research.reacher_random_streams import generator_manifest

STUDY = "reacher-cache-ablation-v1"
SCORED_NAMESPACE = "reacher-cache-ablation-v1-scored"
ENGINEERING_NAMESPACES = (
    "reacher-cache-engineering-unit-v1",
    "reacher-cache-engineering-runner-v1",
    "reacher-cache-engineering-capacity-v1",
    "reacher-cache-engineering-whole-tree-v1",
)
ARMS = ("residual_gru", "encoded_current_gru", "cached_gru", "packet_mlp", "cached_mlp")
PAIRS = ("pair0", "pair1", "pair2")
PANELS = ("full", "ordinary", "shift")
REFERENCES = ("known_state", "particle", "zero", "uniform", "public_kinematic")
MODEL_CLASSES = {
    "residual_gru": "GRUResidualRewardWorldModel",
    "encoded_current_gru": "EncodedCurrentGRUWorldModel",
    "cached_gru": "CachedObservationGRUWorldModel",
    "packet_mlp": "FeedForwardObservationWorldModel",
    "cached_mlp": "CachedObservationMLPWorldModel",
}
INPUT_NAMES = search_protocol.INPUT_NAMES
INPUT_COUNTS = search_protocol.INPUT_COUNTS
require = search_protocol.require

CRITERION = {
    "metric": "negative_total_native_reward_lower_is_better",
    "persistent_arm": "residual_gru",
    "gap_panels": ["ordinary", "shift"],
    "persistent_vs_cache": {
        "comparators": ["cached_gru", "cached_mlp"],
        "mean_improvement": .03, "every_pair_nonworse": True,
    },
    "full_sensing_vs_cache": {
        "comparators": ["cached_gru", "cached_mlp"], "maximum_mean_degradation": .02,
    },
    "competence": {
        "persistent_vs_zero_improvement": .10, "every_pair_on_each_gap": True,
        "references": ["known_state", "particle"], "reference_vs_zero_improvement": .10,
        "each_reference_on_each_gap": True, "public_kinematic": "descriptive_only",
    },
    "expected_checks": 28,
    "all_checks_required": True,
    "scope": "Development on fresh cases; all paired fits retained. No independent-confirmation or biological claim.",
    "pass_interpretation": "Persistent recurrence improves this task beyond both explicitly cached comparator families under the prespecified margins.",
    "fail_interpretation": "The required incremental advantage was not established; failure is not proof of equivalence or proof that recurrence is unnecessary.",
}

SECONDARY_CONTRASTS = [
    {"name": "gru_cache_information", "treatment": "cached_gru", "control": "encoded_current_gru"},
    {"name": "mlp_cache_information", "treatment": "cached_mlp", "control": "packet_mlp"},
]

FEATURE_CONTRACT = {
    "public_packet": "Original sanitized eight-field packet; missing angular placeholders are zero; targets and loss masks use actual validity.",
    "encoder_order": ["cos_q0", "cos_q1", "sin_q0", "sin_q1", "target_x", "target_y", "actual_validity", "actual_age_seconds"],
    "encoded_current_gru": "Unconditionally encode sanitized current packet from zero hidden at every real root, including missing packets.",
    "cached_gru": "Unconditionally encode last actual visible angles plus current goal/validity/age from zero hidden at every real root.",
    "cached_mlp": "First imagined advance encodes last actual visible angles plus current goal/validity/age and action; subsequent advances use predicted packets.",
    "cache_source": "Only actual visible public packets; no predicted angles, recurrent hidden state, privileged native state or realized disturbance.",
    "new_class_boundary": "All three new classes require a valid first packet, static public target, age checked against actual real-step indices, and exactly one selected-action advance before each later assimilation.",
    "age_tolerance": {"atol": 1e-6, "rtol": 1e-5, "visible_age_exact_zero": True},
    "cache_imagination": "Candidate expansion copies real caches. Imagined advances never update measurement caches. Predicted packets remain unobserved.",
    "no_motion_features": True,
    "gate_control": "Cached and encoded-current GRUs use the same unconditional root encoder. The historical gated current GRU is absent, so its isolated gate-removal effect is not estimated.",
}



def _copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def settings(*, engineering=False):
    """Return proposed settings without deriving seeds or allocating generators."""
    require(type(engineering) is bool, "Explicit engineering flag")
    namespace = ENGINEERING_NAMESPACES[0] if engineering else SCORED_NAMESPACE
    plan = {
        "study": STUDY, "version": 1, "rng_namespace": namespace,
        "engineering_rng_namespaces": [name for name in ENGINEERING_NAMESPACES if name != namespace],
        "arms": list(ARMS), "pairs": list(PAIRS), "panels": list(PANELS), "references": list(REFERENCES),
        "model_classes": dict(MODEL_CLASSES), "objective": "unchanged_sequence_loss",
        "train_episodes": 768, "prediction_episodes": 96, "control_episodes": 64,
        "steps": 50, "epochs": 48, "batch_size": 32, "learning_rate": .001,
        "hidden_size": 64, "mlp_width": 107, "dt": .02, "residual_reward": True, "noise_std": .05,
        "ordinary_gap": 6, "shift_gap": 10, "threads": 2,
        "rollout_horizon": 5, "rollout_weight": .5, "reward_scale": 4.,
        "kl_weight": .01, "kl_balance": .8, "free_nats": 1., "gradient_clip": 10.,
        "optimizer": {"name": "Adam", "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.,
                      "amsgrad": False, "foreach": False, "fused": False, "maximize": False,
                      "capturable": False, "differentiable": False, "decoupled_weight_decay": False},
        "prediction_horizons": [1, 3, 7], "auxiliary_horizons": [1, 3, 7],
        "planner": "cem256", "candidates": 64, "planning_horizon": 12, "action_block": 3,
        "particles": 32, "filter_bandwidth": .02, "bootstrap_samples": 4096,
        "search_parameters": {
            "elites": 8, "minimum_std": .001, "reward_clip": [-2.5, 0.],
            "callback_sizes": {"cem256": [64, 64, 64, 64]}, "warm_start": False,
            "elite_carryover": False, "proposal_momentum": 0.,
        },
        "search_pairing": "All arms share the initial proposal bank and Gaussian innovations; later CEM proposals adapt to each model's scores and need not be identical.",
        "reset_interventions": False, "criterion": _copy(CRITERION),
        "feature_contract": _copy(FEATURE_CONTRACT),
        "secondary_contrasts": _copy(SECONDARY_CONTRASTS),
        "secondary_interpretation": "Report all three panels and every paired fit descriptively; cache-information contrasts do not replace the 28-check primary gate. Fitted full-sensing differences can reflect different blackout-training gradients, not different cache information at that visible root.",
        "cost_reporting": {
            "training": "All fit wall times, actual-class operation/sample counts, updates and failed attempts.",
            "deployment": "Public assimilation/reconstruction, CEM proposals/scoring, selected advance and trace storage; native steps/setup separately.",
            "comparison": "Equal data and optimizer updates are not equal training compute; equal CEM candidate counts are not equal total work.",
            "timing": "Shared-host wall times and batch-amortized decisions, not isolated scalar latency or FLOPs.",
        },
    }
    plan["fit_order"] = [row["name"] for row in fit_manifest(plan)]
    plan["execution_order"] = execution_order(plan)
    return plan


def fit_manifest(plan):
    """Three paired five-arm groups, rotating arm order without dropping any fit."""
    require(plan["arms"] == list(ARMS) and plan["pairs"] == list(PAIRS), "All fifteen paired fits required")
    rows = []
    for index, pair in enumerate(PAIRS):
        for arm in (*ARMS[index:], *ARMS[:index]):
            family = "mlp" if arm in ("packet_mlp", "cached_mlp") else "gru"
            rows.append({"name": f"{arm}-{pair}", "arm": arm, "pair": pair,
                         "model_class": MODEL_CLASSES[arm], "initialization_family": family,
                         "initialization_role": f"fit/{family}/{pair}",
                         "minibatch_role": f"fit/minibatch/{pair}"})
    return rows


def execution_order(plan):
    """Exactly 45 learned rows plus 15 reference rows; no reset intervention."""
    require(plan["panels"] == list(PANELS) and plan["references"] == list(REFERENCES), "Exact panels and references required")
    rows = []
    for panel in PANELS:
        for fit in fit_manifest(plan):
            rows.append({"panel": panel, "fit": fit["name"], "arm": fit["arm"], "pair": fit["pair"],
                         "planner": "cem256", "reset": False, "label": fit["name"],
                         "path": f"control/{panel}/{fit['name']}"})
        rows.extend({"panel": panel, "reference": name, "reset": False, "label": name,
                     "path": f"control/{panel}/{name}"} for name in REFERENCES)
    return rows


def coverage(plan):
    """Expected complete membership/counts, not a claim that work has run."""
    fits = [row["name"] for row in fit_manifest(plan)]
    rows = execution_order(plan)
    updates = plan["epochs"] * (plan["train_episodes"] // plan["batch_size"])
    return {"fit_names": fits, "prediction_names": fits, "control_paths": [row["path"] for row in rows],
            "fits": len(fits), "learned_control_rows": len(fits)*len(PANELS),
            "reference_rows": len(REFERENCES)*len(PANELS), "control_rows": len(rows),
            "control_cases_per_row": plan["control_episodes"],
            "native_control_transitions": len(rows)*plan["control_episodes"]*plan["steps"],
            "native_prediction_transitions": plan["prediction_episodes"]*plan["steps"],
            "optimizer_updates_per_fit": updates, "total_optimizer_updates": len(fits)*updates,
            "criterion_checks": CRITERION["expected_checks"]}


def validate_settings(plan, *, engineering=False):
    """Bind study-owned fields; provenance and run authorization belong to runner.

    Engineering dimensions may only shrink. Thus full-size namespace exclusions
    cover every cohort/step/initialization role usable by the synthetic helpers.
    Scientific constants, membership, order and criterion never become flexible.
    """
    expected = settings(engineering=engineering)
    require(isinstance(plan, dict) and set(expected) <= set(plan), "Missing cache-study settings")
    dimensions = {"train_episodes", "prediction_episodes", "control_episodes", "epochs", "batch_size",
                  "hidden_size", "mlp_width", "threads", "bootstrap_samples"}
    flexible = {"rng_namespace", "engineering_rng_namespaces"} | (dimensions if engineering else set())
    for key, value in expected.items():
        if key not in flexible:
            require(_identity(plan[key]) == _identity(value), f"Cache setting changed: {key}")
    for key in dimensions:
        require(type(plan[key]) is int and 0 < plan[key] <= expected[key], f"Invalid bounded positive integer: {key}")
    require(plan["train_episodes"] % plan["batch_size"] == 0, "Complete equal-size training batches required")
    namespace = plan["rng_namespace"]
    require(namespace in (ENGINEERING_NAMESPACES if engineering else (SCORED_NAMESPACE,)), "Declared engineering/scored namespace required")
    exclusions = [name for name in ENGINEERING_NAMESPACES if name != namespace]
    require(plan["engineering_rng_namespaces"] == exclusions, "Exact full-size engineering exclusions required")
    require(plan["fit_order"] == [row["name"] for row in fit_manifest(plan)], "Fit ordering changed")
    require(plan["execution_order"] == execution_order(plan), "Control ordering changed")
    return plan


def _namespace(plan):
    value = plan["rng_namespace"]
    require(value in (SCORED_NAMESPACE, *ENGINEERING_NAMESPACES), "Declared cache RNG namespace required")
    return value


def registry(plan):
    """Fresh NumPy roots, using the existing search helper's role names."""
    namespace = _namespace(plan)
    roles = []
    for split in ("control", "prediction"):
        count = plan[f"{split}_episodes"]
        require(type(count) is int and 0 < count <= (64 if split == "control" else 96), "Bounded cohort count required")
        for purpose in ("reset", "actuator_noise", "sensor_schedule"):
            roles.extend(f"{split}/{purpose}/{i}" for i in range(count))
        if split == "prediction":
            roles.extend(f"prediction/exploration/{i}" for i in range(count))
    require(type(plan["steps"]) is int and plan["steps"] == 50, "Fixed real terminal boundary required")
    roles.extend(f"planner/particle_filter/{i}" for i in range(plan["control_episodes"]))
    roles.extend(("floor/uniform/0", "analysis/bootstrap/0"))
    for step in range(plan["steps"]):
        roles.extend(f"planner/control/{step}/{name}" for name in INPUT_NAMES)
    require(len(roles) == len(set(roles)), "Duplicate NumPy role")
    return {role: search_protocol.named_seed(namespace, role) for role in roles}


def seed32(namespace, role):
    """CPU Torch seeds restricted to 32 bits to avoid high-bit aliases."""
    return search_protocol.named_seed(namespace, role) & ((1 << 32) - 1)


def torch_registry(plan):
    namespace = _namespace(plan)
    require(plan["pairs"] == list(PAIRS), "All paired initialization identities required")
    return {f"fit/{purpose}/{pair}": seed32(namespace, f"fit/{purpose}/{pair}")
            for pair in PAIRS for purpose in ("gru", "mlp", "minibatch")}


def _validated_registry(values, label):
    require(isinstance(values, dict), f"{label} registry mapping required")
    require(all(isinstance(name, str) and name.strip() and type(value) is int and 0 <= value < (1 << 64)
                for name, value in values.items()), f"{label} registry role/uint64 seed")
    return dict(values)


def torch_generator_manifest(values):
    result = {}
    for role, value in _validated_registry(values, "Torch").items():
        normalized = value & ((1 << 32) - 1)
        generator = torch.Generator(device="cpu").manual_seed(normalized)
        result[role] = {"seed": value, "effective_seed32": normalized, "device": "cpu",
                        "initial_state_sha256": hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest(),
                        "draws_for_manifest": 0}
    return result


def _states(manifest):
    return {row["initial_state_sha256"] for row in manifest.values()}


def memory_stream_exclusions():
    """Exclude every full-size prior memory namespace without sampling or I/O.

    Historical artifact identity and older inherited streams still belong to
    the caller's authenticated lineage. This closed set prevents accidental
    omission of memory-study roles even in a reduced engineering fixture.
    """
    result = []
    for namespace in (memory_protocol.SCORED_NAMESPACE, *memory_protocol.ENGINEERING_NAMESPACES):
        previous_plan = memory_protocol.settings()
        previous_plan["rng_namespace"] = namespace
        numpy_previous = memory_protocol.registry(previous_plan)
        torch_previous = memory_protocol.torch_registry(previous_plan)
        result.append({"namespace": namespace,
                       "coverage": {"control_episodes": 64, "prediction_episodes": 96, "steps": 50,
                                    "pairs": list(memory_protocol.PAIRS), "arms": list(memory_protocol.ARMS)},
                       "registry": numpy_previous, "generators": generator_manifest(numpy_previous),
                       "torch_registry": torch_previous,
                       "torch_generators": torch_generator_manifest(torch_previous)})
    return result


def stream_contract(plan, prior_registries=(), prior_torch_registries=(), prior_plans=()):
    """Bind supplied authenticated priors and full engineering streams, no draws.

    The caller must include every relevant prior scored/engineering role and its
    authenticated provenance. Hashes here bind what was supplied; they do not
    independently authenticate files or prove a prior registry is complete.
    """
    validate_settings(plan, engineering=_namespace(plan) in ENGINEERING_NAMESPACES)
    previous_numpy = [_validated_registry(value, "Prior NumPy") for value in prior_registries]
    previous_torch = [_validated_registry(value, "Prior Torch") for value in prior_torch_registries]
    current, torch_current = registry(plan), torch_registry(plan)
    require(len(set(current.values())) == len(current), "NumPy root seed collision")
    require(len(set(torch_current.values())) == len(torch_current), "Torch root seed collision")
    require(not (set(current.values()) & set(torch_current.values())), "Cross-runtime root seed collision")
    numpy_states, torch_states = generator_manifest(current), torch_generator_manifest(torch_current)
    require(len(_states(numpy_states)) == len(numpy_states), "NumPy generator-state collision")
    require(len(_states(torch_states)) == len(torch_states), "Torch generator-state collision")
    exclusions = []
    for namespace in plan["engineering_rng_namespaces"]:
        full = settings()
        full["rng_namespace"] = namespace
        previous, torch_previous = registry(full), torch_registry(full)
        prior_np_states, prior_torch_states = generator_manifest(previous), torch_generator_manifest(torch_previous)
        require(not (set(current.values()) & set(previous.values())), "Engineering NumPy root collision")
        require(not (_states(numpy_states) & _states(prior_np_states)), "Engineering NumPy state collision")
        require(not (_states(torch_states) & _states(prior_torch_states)), "Engineering Torch state collision")
        exclusions.append({"namespace": namespace,
                           "coverage": {"control_episodes": 64, "prediction_episodes": 96, "steps": 50,
                                        "pairs": list(PAIRS), "arms": list(ARMS)},
                           "registry": previous, "generators": prior_np_states,
                           "torch_registry": torch_previous, "torch_generators": prior_torch_states})
    for previous in previous_numpy:
        require(not (set(current.values()) & set(previous.values())), "Prior NumPy root collision")
        require(not (_states(numpy_states) & _states(generator_manifest(previous))), "Prior NumPy state collision")
    for previous in previous_torch:
        require(not (_states(torch_states) & _states(torch_generator_manifest(previous))), "Prior Torch state collision")
    memory_exclusions = memory_stream_exclusions()
    for previous in memory_exclusions:
        require(not (set(current.values()) & set(previous["registry"].values())), "Memory-study NumPy root collision")
        require(not (_states(numpy_states) & _states(previous["generators"])), "Memory-study NumPy state collision")
        require(not (_states(torch_states) & _states(previous["torch_generators"])), "Memory-study Torch state collision")
    return {
        "namespace": plan["rng_namespace"], "registry": current, "generators": numpy_states,
        "torch_registry": torch_current, "torch_generators": torch_states,
        "engineering_exclusions": exclusions, "priors": _copy(list(prior_plans)),
        "memory_study_exclusions": memory_exclusions,
        "prior_numpy_registry_sha256": [_identity(value) for value in previous_numpy],
        "prior_torch_registry_sha256": [_identity(value) for value in previous_torch],
        "draws_for_manifest": 0, "full_horizon_innovations": True,
        "unused_random_extra": "Generated and saved for SearchInputs compatibility; never scored by CEM256.",
        "intentional_reuse": [
            "The three GRU classes share initial parameter tensors per pair; both MLP classes share a separate family initialization.",
            "All five arms in a pair share the same complete minibatch order; all inherited training episodes are shared.",
            "Every fit/panel/reference shares corresponding control resets, actuator noise and schedule phase.",
            "All control rows share full-horizon innovations at each decision; adapted CEM candidates may differ.",
            "Particle-filter children have distinct initialization/process/resampling purposes; public kinematics is deterministic.",
            "Uniform commands and bootstrap draws are paired; no prediction stream shares a control stream.",
        ],
        "provenance_limit": "Caller authenticates prior artifacts and completeness; no files are read by this helper.",
    }


def seed(plan, role):
    """Reject an unbound, aliased or renamed role before returning its seed."""
    require(isinstance(role, str), "String RNG role required")
    torch_role = role.startswith("fit/")
    expected = torch_registry(plan) if torch_role else registry(plan)
    require(role in expected, "Unknown RNG role")
    contract = plan["random_stream_contract"]
    require(contract["namespace"] == _namespace(plan), "Bound stream namespace differs")
    value = contract["torch_registry" if torch_role else "registry"][role]
    require(type(value) is int and value == expected[role], "Bound seed differs from role-derived seed")
    return value


def torch_generator(plan, role):
    require(role in torch_registry(plan), "Unknown Torch RNG role")
    return torch.Generator(device="cpu").manual_seed(seed(plan, role))


@contextmanager
def initialization_rng(plan, pair, component="gru"):
    """Temporarily install isolated CPU initialization state, including on failure."""
    require(pair in PAIRS and component in ("gru", "mlp"), "Unknown initialization role")
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch_generator(plan, f"fit/{component}/{pair}").get_state())
        yield


def minibatch_orders(plan, pair):
    require(pair in PAIRS, "Unknown minibatch pair")
    require(type(plan["train_episodes"]) is int and 0 < plan["train_episodes"] <= 768
            and type(plan["epochs"]) is int and 0 < plan["epochs"] <= 48, "Bounded minibatch dimensions")
    generator = torch_generator(plan, f"fit/minibatch/{pair}")
    return torch.stack([torch.randperm(plan["train_episodes"], generator=generator)
                        for _ in range(plan["epochs"])])


def _cohort(plan, index, split):
    require(split in ("control", "prediction") and type(index) is int
            and 0 <= index < plan[f"{split}_episodes"], "Invalid cohort index")


def phase(plan, index, split="control"):
    _cohort(plan, index, split)
    return int(np.random.default_rng(seed(plan, f"{split}/sensor_schedule/{index}")).integers(0, 4))


def schedule(plan, index, panel, split="control"):
    _cohort(plan, index, split)
    require(panel in PANELS and plan["steps"] == 50 and plan["ordinary_gap"] == 6 and plan["shift_gap"] == 10,
            "Fixed sensing panels and terminal boundary required")
    valid = np.ones(plan["steps"] + 1, dtype=bool)
    if panel != "full":
        offset = phase(plan, index, split)
        gap = plan["shift_gap"] if panel == "shift" else plan["ordinary_gap"]
        for start in (8 + offset, 28 + offset):
            valid[start:start + gap] = False
    return valid


def draw_control_inputs(plan, step):
    require(type(step) is int and 0 <= step < plan["steps"] == 50, "No innovation draw beyond terminal boundary")
    require(plan["planning_horizon"] == 12 and plan["action_block"] == 3, "Fixed search horizon/block required")
    chunks = (plan["planning_horizon"] + plan["action_block"] - 1) // plan["action_block"]
    values_seeds = [seed(plan, f"planner/control/{step}/{name}") for name in INPUT_NAMES]
    values = [np.random.default_rng(value).normal(
        size=(plan["control_episodes"], count, chunks, 2))
        for value, count in zip(values_seeds, INPUT_COUNTS, strict=True)]
    return SearchInputs(values[0], values[1], tuple(values[2:]))


def common_bank(inputs, step, plan):
    """Anchored reference proposals with the same arithmetic as prior studies."""
    require(type(step) is int and 0 <= step < plan["steps"] == 50, "No candidate bank beyond terminal boundary")
    require(plan["planning_horizon"] == 12 and plan["action_block"] == 3 and plan["candidates"] == 64,
            "Fixed reference candidate scope")
    require(isinstance(inputs, SearchInputs) and inputs.initial.shape == (plan["control_episodes"], 64, 4, 2),
            "Full-horizon initial proposal membership")
    require(inputs.initial.dtype == np.float64 and np.isfinite(inputs.initial).all(), "Finite float64 proposal innovations")
    horizon = min(plan["planning_horizon"], plan["steps"] - step)
    chunks = (horizon + plan["action_block"] - 1) // plan["action_block"]
    value = inputs.initial[:, :, :chunks] * np.r_[np.full(32, .25), np.full(32, .75)][None, :, None, None]
    for index, anchor in enumerate(ANCHORS):
        value[:, index] = anchor
    return np.repeat(np.clip(value, -1, 1).astype(np.float32), plan["action_block"], axis=2)[:, :, :horizon]
