"""Prospective two-observation study definitions, never a runnable freeze.

This stdlib-only module enumerates scope, symbolic random roles and proposed
criteria. It performs no I/O, seed derivation, generator allocation, sampling,
training, model restoration or native call. A future preparation/auditor pair
must authenticate bytes and actual RNG states, measure capacity, bind the full
source closure and freeze the resulting study before scientific execution.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import PurePosixPath

STUDY = "reacher-two-observation-study-v1"
SCORED_NAMESPACE = "reacher-two-observation-study-v1-scored"
ENGINEERING_NAMESPACES = (
    "reacher-two-observation-engineering-unit-v1",
    "reacher-two-observation-engineering-runner-v1",
    "reacher-two-observation-engineering-capacity-v1",
    "reacher-two-observation-engineering-whole-tree-v1",
)
ARMS = ("residual_gru", "two_observation_gru", "cached_gru")
INHERITED_ARMS = ("residual_gru", "cached_gru")
PAIRS = ("pair0", "pair1", "pair2")
PANELS = ("full", "ordinary", "shift")
REFERENCES = ("known_state", "particle", "public_kinematic", "zero", "uniform")
PHYSICS_REFERENCES = REFERENCES[:3]
MODEL_CLASSES = {"residual_gru": "GRUResidualRewardWorldModel",
                 "two_observation_gru": "TwoObservationHistoryGRUWorldModel",
                 "cached_gru": "CachedObservationGRUWorldModel"}
INPUT_NAMES = ("initial", "random_extra", "cem/1", "cem/2", "cem/3")
INPUT_COUNTS = (64, 192, 64, 64, 63)
FILTER_CHILDREN = ("initial", "process_noise", "resample")
FIT_MEMBERS = ("initial-weights.pt", "training.jsonl", "weights.pt", "checkpoint.pt", "completed.json")
CACHE_PLAN_SHA256 = "7868daa12242df37f020946f9d3b279811a0e97547eef4d8b179da6e9394cefa"
PREREQUISITE_PLAN_SHA256 = "23c93e4adfbb45cf224383ffa31d7323df2405807eb3f1c4c75bb27378bcd9ee"
PREREQUISITE_TERMINAL_SHA256 = "8c65730d6547b118eaeaca86ef61857ef0e4f52145f3eea214316f23f32dbae2"
ADAM = {"name": "Adam", "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.,
        "amsgrad": False, "foreach": False, "fused": False, "maximize": False,
        "capturable": False, "differentiable": False, "decoupled_weight_decay": False}
NEW_SOURCE_PATHS = (
    "src/openjev/research/reacher_two_observation_history.py",
    "tests/test_reacher_two_observation_history.py",
    "src/openjev/research/reacher_two_observation_training.py",
    "tests/test_reacher_two_observation_training.py",
    "src/openjev/research/reacher_two_observation_control.py",
    "tests/test_reacher_two_observation_control.py",
    "src/openjev/research/reacher_two_observation_protocol.py",
    "tests/test_reacher_two_observation_protocol.py",
    "src/openjev/research/reacher_two_observation_audit.py",
    "tests/test_reacher_two_observation_audit.py",
    "scripts/reacher_two_observation_study.py",
    "tests/test_reacher_two_observation_study.py",
    "scripts/audit_reacher_two_observation_study.py",
    "tests/test_audit_reacher_two_observation_study.py",
)
CRITERION = {
    "status": "proposed_until_complete_protocol_is_frozen",
    "metric": "negative_total_native_reward_lower_is_better",
    "persistent_arm": "residual_gru", "comparators": ["two_observation_gru", "cached_gru"],
    "gap_panels": ["ordinary", "shift"], "mean_improvement": .03,
    "every_pair_nonworse": True, "full_maximum_mean_degradation": .02,
    "persistent_vs_zero_improvement": .10, "every_persistent_pair_on_each_gap": True,
    "reference": "known_state", "reference_panel": "ordinary", "reference_vs_zero_improvement": .10,
    "particle_and_public_kinematic": "descriptive_only", "expected_checks": 25,
    "all_checks_required": True, "secondary_cannot_rescue_primary": True,
    "aggregation": "Same cases per fit; arithmetic mean of all three fit costs. No best-fit selection or averaging per-fit percentage improvements for a family check.",
    "pass_interpretation": "Persistent control improves beyond the trained bounded two-observation recipe and trained one-observation cache under this task, objective and candidate budget.",
    "fail_interpretation": "The specified persistent advantage was not established. Failure does not prove equivalence, memory irrelevance or a successful cache explanation.",
    "limits": "Three paired fits, shared historical training corpus/initializations/orders, one task family. Not independent training replication, isolated velocity inference, Bayesian inference, biological wiring or architecture novelty.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def _path(value):
    return (isinstance(value, str) and bool(value) and not PurePosixPath(value).is_absolute()
            and all(part not in (".", "..") for part in value.split("/")) and "\\" not in value)


def preparation_requirements():
    """Unfulfilled integration obligations, not a check that files exist."""
    return {
        "execution_authorized": False,
        "required_bindings": ["lineage", "paired_training", "inherited_checkpoints", "sources", "runtime",
                              "fresh_streams", "capacity_and_caps", "audit_contract"],
        "lineage": {"cache_plan_sha256": CACHE_PLAN_SHA256,
                    "prerequisite_plan_sha256": PREREQUISITE_PLAN_SHA256,
                    "prerequisite_terminal_sha256": PREREQUISITE_TERMINAL_SHA256,
                    "validation": "Authenticate completed execution/audit/terminal artifacts and positive prerequisite gate; metadata assertions alone are insufficient."},
        "sources": {"unchanged_inherited_count": 90, "required_new_paths": list(NEW_SOURCE_PATHS),
                    "minimum_total": 90 + len(NEW_SOURCE_PATHS), "complete_closure_pending": True,
                    "validation": "Supply authenticated inherited90 map and an independently reviewed complete new closure. Every actual file digest must match; any additional dependency must be explicitly bound."},
        "paired_training": "Authenticate all768 historical episodes and their unchanged public tensors, all3 original initial tensors, complete48-epoch orders and semantic identities; fitted weights cannot substitute for initial tensors.",
        "restore": "Authenticate all6 inherited actual-class checkpoint/configuration/tensor boundaries before fitting or fresh draws. Restore all9 final actual classes before new control. Verify every inherited final tensor unchanged.",
        "runtime": "Bind exact Python/NumPy/PyTorch/MuJoCo/Gymnasium versions, machine/CPU/threads, deterministic execution settings and runner/auditor runtime checks.",
        "streams": "Future preparation alone derives seeds and verifies actual root/child generator-state separation against every authenticated historical scored/engineering registry, own full-size engineering namespaces and known literal engineering calls. Namespace text alone proves no separation.",
        "capacity_and_caps": "Run retained synthetic whole-tree and capacity checks first; supply positive execution/audit caps with receipt hashes. No guessed default caps; no scientific retry on failure or timeout.",
        "audit": "Saved-output-only independent auditor must verify paired fitting and Adam/orders/log chains, actual classes, real suffix/action alignment, CEM/geometry, nominal physics replay, all42 rows, all25 checks, exact member boundaries and all costs. No model inference in saved-output audit.",
        "reporting": "Retain all fits, panels, failures and engineering attempts; failed qualification is still a completed result. Native steps and trace storage are additional to public-controller timing.",
    }


def settings(*, engineering=False):
    require(type(engineering) is bool, "Explicit engineering flag")
    value = {
        "study": STUDY, "version": 1, "status": "prospective_unfrozen_unrun", "engineering": engineering,
        "rng_namespace": ENGINEERING_NAMESPACES[0] if engineering else SCORED_NAMESPACE,
        "arms": list(ARMS), "pairs": list(PAIRS), "model_classes": dict(MODEL_CLASSES),
        "panels": list(PANELS), "references": list(REFERENCES), "score_modes": ["geometry"],
        "train_episodes": 768, "epochs": 48, "batch_size": 32, "hidden_size": 64,
        "mlp_width": 107, "mlp_width_usage": "unused inherited trainer metadata; no MLP arm",
        "steps": 50, "dt": .02, "noise_std": .05, "residual_reward": True,
        "learning_rate": .001, "gradient_clip": 10., "rollout_horizon": 5,
        "rollout_weight": .5, "reward_scale": 4., "kl_weight": .01, "kl_balance": .8, "free_nats": 1.,
        "optimizer": copy.deepcopy(ADAM), "objective": "unchanged_sequence_loss",
        "device": "cpu", "dtype": "torch.float32", "threads": 2,
        "control_episodes": 64, "ordinary_gap": 6, "shift_gap": 10, "bootstrap_samples": 4096,
        "planner": "cem256", "physics_planner": "cem256", "candidates": 64,
        "planning_horizon": 12, "action_block": 3, "particles": 32, "filter_bandwidth": .02,
        "search_parameters": {"elites": 8, "minimum_std": .001, "reward_clip": [-2.5, 0.],
            "callback_sizes": [64, 64, 64, 64], "final_stage": "63 fresh proposals plus one paid mean",
            "selection": "best globally evaluated sequence; earliest candidate ID breaks ties",
            "warm_start": False, "elite_carryover": False, "proposal_momentum": 0.,
            "accumulation": "sequential_float32_per_step_clip", "terminal": "shorten at absolute step50"},
        "history_contract": {"actual_class": MODEL_CLASSES["two_observation_gru"],
            "public_packet_width": 8, "command_width": 2, "max_real_packets": 12, "max_issued_commands": 11,
            "anchor": "older of the last two valid observations; first observation before second exists",
            "real_boundary": "Reset learned state and replay actual sanitized packets plus issued commands; accept startup or exactly one acknowledged selected advance.",
            "paid_reconstruction": "Every real assimilation executes12 observation updates and11 full transitions, including padding, both original heads and residual analytic actuator cost.",
            "imagined_boundary": "Private candidate state never appends real evidence; selected command re-advances from untouched real root.",
            "overflow": "Append new visible packet before moving anchor; reject spans above11 commands without truncation.",
            "training": "Reconstruction active throughout fitting, gradients attached; unchanged valid-root/endpoint masks and five-step rollout loss."},
        "scoring_contract": {"helper": "openjev.research.reacher_geometry_reward.geometry_reward_components",
            "geometry": "atan2 projection of decoded cosine/sine pairs; link offsets0.10/0.11m; norm>=1e-6; no joint-limit penalty",
            "score": "negative planar fingertip distance minus expected clipped noisy actuator squared cost once",
            "original_heads": "Retain original learned observation/reward heads and residual cost skip, though only geometry chooses commands.",
            "limits": "Final-qpos FK approximates RK4 cached-body reward; distance of projected mean is not expected distance. No measured native future/noise enters learned scoring."},
        "reference_contract": {"known_state": "Privileged current qpos/qvel, never future noise; nominal physics CEM256+geometry.",
            "particle": "Public packets and issued commands only; supplied nominal physics CEM256+geometry.",
            "public_kinematic": "Two valid public measurements, elapsed time and issued commands; supplied nominal physics CEM256+geometry.",
            "zero": "Zero issued command; no search.", "uniform": "Paired fresh named uniform command floor; no search."},
        "pairing": "Original initial tensors/data/full orders are shared within each GRU pair; inherited final weights stay frozen, new history weights are fitted from scratch. Fresh resets/noise/phases/initial proposals and innovations are paired across all rows; later CEM proposals adapt to each score.",
        "stage_order": ["authenticate_original_training_and_six_inherited_fits",
                        "train_three_history_fits_from_original_initial_tensors",
                        "authenticate_restore_all_nine_final_models", "fresh_closed_loop_control",
                        "saved_output_only_independent_audit"],
        "new_prediction_episodes": 0, "exposed_diagnostic_roots": 0, "reset_interventions": False,
        "criterion": copy.deepcopy(CRITERION),
        "secondary_contrasts": [{"name": "two_observation_vs_cached", "treatment": "two_observation_gru", "control": "cached_gru",
                                  "interpretation": "Longer explicit public information/reconstruction recipe; not isolated velocity estimation."}],
        "cost_reporting": {"historical": "Report inherited training and earlier failed-attempt costs separately using authenticated cumulative lineage; avoid double counting.",
            "new_training": "Three new fits, full padded reconstruction/heads/analytic skip, backward/Adam, setup, validation, resume checks and storage; equal updates do not equal compute.",
            "deployment": "All reconstruction, original heads, geometry, history validation/copies, candidate search, selected advance, snapshots, native steps, audit replay and trace storage.",
            "fairness": "Candidate-budget matched only; reconstruction, instrumentation and copying penalties are not information-theoretic or architectural advantages."},
        "preparation_requirements": preparation_requirements(),
    }
    value["fit_order"] = [row["name"] for row in fit_manifest(value)]
    value["execution_order"] = execution_order(value)
    return value


def _scope(plan):
    require(plan["arms"] == list(ARMS) and plan["pairs"] == list(PAIRS)
            and plan["model_classes"] == MODEL_CLASSES, "All nine actual-class models required")


def fit_manifest(plan):
    _scope(plan)
    updates = math.ceil(plan["train_episodes"] / plan["batch_size"]) * plan["epochs"]
    return [{"name": f"{arm}-{pair}", "arm": arm, "pair": pair, "model_class": MODEL_CLASSES[arm],
             "new_fit": arm == "two_observation_gru", "weights_reused": arm in INHERITED_ARMS,
             "new_optimizer_updates": updates if arm == "two_observation_gru" else 0,
             "initialization_path": f"initializations/{pair}.pt", "order_path": f"orders/{pair}.pt"}
            for pair in PAIRS for arm in ARMS]


def training_manifest(plan):
    return [{**row, "initial_state_kind": "original_initialization",
             "initial_source_fit": f"residual_gru-{row['pair']}",
             "optimizer_start": "fresh empty Adam; never inherit fitted optimizer state",
             "training_source": "reacher-cache-ablation-v1/train.npz+train.json"}
            for row in fit_manifest(plan) if row["new_fit"]]


def restore_manifest(plan):
    return [{**row, "source_study": "reacher-cache-ablation-v1",
             "source_path": f"fits/{row['name']}", "copy_path": f"inherited/fits/{row['name']}",
             "members": [f"fits/{row['name']}/{member}" for member in FIT_MEMBERS]}
            for row in fit_manifest(plan) if row["weights_reused"]]


def inherited_members(plan):
    names = {"train.npz", "train.json"}
    for pair in PAIRS:
        names.update(f"{folder}/{pair}.{suffix}" for folder in ("initializations", "orders") for suffix in ("pt", "json"))
    for row in restore_manifest(plan):
        names.update(row["members"])
    return sorted(names)


def execution_order(plan):
    require(plan["panels"] == list(PANELS) and plan["references"] == list(REFERENCES)
            and plan["score_modes"] == ["geometry"], "Exact sensing panels/references/geometry required")
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
    require(_same(plan["criterion"], CRITERION), "Exact proposed25-check criterion")
    checks = []
    for comparator in CRITERION["comparators"]:
        for panel in ("ordinary", "shift"):
            checks.append({"name": f"gap_mean/{panel}/{comparator}", "panel": panel,
                "treatment": "residual_gru", "control": comparator, "aggregation": "three_fit_mean",
                "maximum_cost_multiplier": .97})
            for pair in PAIRS:
                checks.append({"name": f"gap_pair/{panel}/{comparator}/{pair}", "panel": panel,
                    "treatment": f"residual_gru-{pair}", "control": f"{comparator}-{pair}",
                    "aggregation": "paired_fit", "maximum_cost_multiplier": 1.})
        checks.append({"name": f"full_mean/{comparator}", "panel": "full",
            "treatment": "residual_gru", "control": comparator, "aggregation": "three_fit_mean",
            "maximum_cost_multiplier": 1.02})
    for panel in ("ordinary", "shift"):
        for pair in PAIRS:
            checks.append({"name": f"competence/{panel}/residual_gru-{pair}", "panel": panel,
                "treatment": f"residual_gru-{pair}", "control": "zero", "aggregation": "fit_vs_reference",
                "maximum_cost_multiplier": .9})
    checks.append({"name": "competence/ordinary/known_state", "panel": "ordinary",
        "treatment": "known_state", "control": "zero", "aggregation": "reference_vs_reference", "maximum_cost_multiplier": .9})
    require(len(checks) == len({row["name"] for row in checks}) == 25, "Unique complete proposed gate")
    return checks


def coverage(plan):
    validate_settings(plan, engineering=plan.get("engineering") is True)
    cases, steps = plan["control_episodes"], plan["steps"]
    learned, physics, floors, history = 27, 9, 6, 9
    horizon_sum = sum(min(plan["planning_horizon"], steps - t) for t in range(steps))
    presentations = 3 * plan["train_episodes"] * plan["epochs"]
    assimilations = presentations * steps
    training_advances = presentations * (steps + plan["rollout_horizon"] * (steps - plan["rollout_horizon"] + 1))
    history_assimilations = history * cases * steps
    return {"evaluated_models": 9, "inherited_fits": 6, "new_fits": 3,
        "new_optimizer_updates_per_fit": math.ceil(plan["train_episodes"] / plan["batch_size"]) * plan["epochs"],
        "new_optimizer_updates": sum(row["new_optimizer_updates"] for row in fit_manifest(plan)),
        "new_native_training_episodes": 0, "inherited_training_episodes": plan["train_episodes"],
        "new_training_episode_presentations": presentations,
        "new_training_public_assimilate_samples": assimilations,
        "new_training_replayed_observation_update_samples": 12 * assimilations,
        "new_training_replayed_transition_samples": 11 * assimilations,
        "new_training_prefix_and_rollout_advance_samples": training_advances,
        "new_training_total_transition_samples": 11 * assimilations + training_advances,
        "new_training_gru_cell_samples": 23 * assimilations + training_advances,
        "new_training_linear_layer_samples": 4 * (11 * assimilations + training_advances),
        "new_training_analytic_reward_samples": 11 * assimilations + training_advances,
        "training_count_scope": "Scheduled successful forward work; not backward/Adam, failed work, measured wall time or total FLOPs.",
        "control_rows": learned + physics + floors, "learned_control_rows": learned,
        "history_control_rows": history, "physics_reference_rows": physics, "floor_rows": floors, "reference_rows": physics + floors,
        "native_control_transitions": (learned + physics + floors) * cases * steps,
        "learned_candidate_evaluations": learned * cases * steps * 256,
        "physics_candidate_evaluations": physics * cases * steps * 256,
        "learned_imagined_transitions": learned * cases * 256 * horizon_sum,
        "physics_nominal_transitions": physics * cases * 256 * horizon_sum,
        "learned_selected_advances": learned * cases * steps, "physics_selected_advances": physics * cases * steps,
        "physics_candidate_native_substeps": 2 * physics * cases * 256 * horizon_sum,
        "physics_selected_native_substeps": 2 * physics * cases * steps,
        "geometry_candidate_samples": (learned + physics) * cases * 256 * horizon_sum,
        "geometry_selected_samples": (learned + physics) * cases * steps,
        "history_public_assimilate_samples": history_assimilations,
        "history_replayed_observation_update_samples": 12 * history_assimilations,
        "history_replayed_transition_samples": 11 * history_assimilations,
        "sum_shortened_planning_horizons": horizon_sum, "new_prediction_episodes": 0,
        "exposed_diagnostic_roots": 0, "criterion_checks": 25}


def role_manifest(plan):
    """Symbolic identities only, including filter children; no numerical seeds."""
    validate_settings(plan, engineering=plan.get("engineering") is True)
    roots = [f"control/{purpose}/{case}" for purpose in ("reset", "actuator_noise", "sensor_schedule")
             for case in range(plan["control_episodes"])]
    roots += [f"planner/particle_filter/{case}" for case in range(plan["control_episodes"])]
    roots += ["floor/uniform/0", "analysis/bootstrap/0"]
    roots += [f"planner/control/{step}/{name}" for step in range(50) for name in INPUT_NAMES]
    generators = []
    for role in roots:
        if role.startswith("planner/particle_filter/"):
            generators.extend({"role": f"{role}/{child}", "parent_role": role, "spawn_key": [index]}
                              for index, child in enumerate(FILTER_CHILDREN))
        else:
            generators.append({"role": role, "parent_role": role, "spawn_key": []})
    return {"namespace": plan["rng_namespace"], "root_roles": roots, "generator_roles": generators,
        "root_role_count": len(roots), "generator_role_count": len(generators),
        "new_training_roles": [], "numerical_seed_allocation": "not_implemented_not_performed",
        "own_namespace_exclusions": [name for name in (SCORED_NAMESPACE, *ENGINEERING_NAMESPACES)
                                     if name != plan["rng_namespace"]],
        "exclusion_coverage": "Full64-case scope even when an engineering fixture uses fewer cases; all inherited scored/engineering root and spawned-generator states plus new known literal410 calls.",
        "inherited_torch_reuse": "Original initial tensors and minibatch orders are intentionally reused. Order validation replays historical Torch permutations; constructor410 is isolated and fully overwritten, with ambient RNG restored.",
        "innovation_shapes": {name: [plan["control_episodes"], count, 4, 2] for name, count in zip(INPUT_NAMES, INPUT_COUNTS, strict=True)},
        "unused_random_extra": "Generate/save later for complete SearchInputs provenance, never score these192 proposals in CEM256.",
        "schedule": {"draw": "one offset uniformly in{0,1,2,3} per named sensor_schedule role",
                     "base_gap_starts": [8, 28], "ordinary_length": 6, "shift_length": 10,
                     "packets": 51, "first_valid": True, "terminal_valid": True}}


def validate_settings(plan, *, engineering=False):
    require(type(engineering) is bool, "Explicit engineering validation mode")
    expected = settings(engineering=engineering)
    require(isinstance(plan, dict) and set(plan) == set(expected), "Exact prospective settings; preparation bindings are separate")
    flexible = {"rng_namespace"}
    dimensions = {"train_episodes", "epochs", "batch_size", "hidden_size", "control_episodes", "threads", "bootstrap_samples"}
    if engineering:
        flexible |= dimensions
    for key, value in expected.items():
        if key not in flexible:
            require(_same(plan[key], value), f"Prospective setting changed: {key}")
    for key in dimensions:
        require(type(plan[key]) is int and 0 < plan[key] <= expected[key], f"Bounded positive integer: {key}")
    require(plan["rng_namespace"] in (ENGINEERING_NAMESPACES if engineering else (SCORED_NAMESPACE,)), "Declared prospective namespace")
    require(plan["fit_order"] == [row["name"] for row in fit_manifest(plan)]
            and plan["execution_order"] == execution_order(plan), "Exact all-fit/all-row order")
    return plan


def validate_preparation_bindings(plan, bindings, *, expected_sources, inherited_sources):
    """Metadata schema checks only; does NOT authenticate bytes or allow a run.

    The future preparation layer supplies both independently reviewed source
    maps. This module never invents source/runtime/cap values or reads artifacts.
    Actual lineage, generator states, checkpoint tensors and files still need
    independent validation before a separate frozen protocol can be executable.
    """
    validate_settings(plan, engineering=plan.get("engineering") is True)
    require(isinstance(bindings, dict) and set(bindings) == set(preparation_requirements()["required_bindings"]),
            "Complete external preparation metadata required")
    require(isinstance(inherited_sources, dict) and len(inherited_sources) == 90
            and all(_path(name) and _sha(digest) for name, digest in inherited_sources.items()), "Explicit inherited90 source map")
    require(isinstance(expected_sources, dict) and not set(inherited_sources) & set(NEW_SOURCE_PATHS)
            and set(expected_sources) >= set(inherited_sources) | set(NEW_SOURCE_PATHS)
            and all(_path(name) and _sha(digest) for name, digest in expected_sources.items()), "Reviewed complete additive source closure")
    require(bindings["sources"] == expected_sources and all(expected_sources[name] == digest
            for name, digest in inherited_sources.items()), "Exact external sources and unchanged inherited hashes")
    lineage = bindings["lineage"]
    require(isinstance(lineage, dict) and lineage == {
        "cache_plan_sha256": CACHE_PLAN_SHA256, "prerequisite_plan_sha256": PREREQUISITE_PLAN_SHA256,
        "prerequisite_terminal_sha256": PREREQUISITE_TERMINAL_SHA256,
        "completed_positive_gate_required": True}, "Exact prerequisite and original-training lineage identifiers")
    paired = bindings["paired_training"]
    require(isinstance(paired, dict) and set(paired) == {"cohort_sha256", "public_tensor_sha256", "pairs"}
            and _sha(paired["cohort_sha256"]) and _sha(paired["public_tensor_sha256"])
            and isinstance(paired["pairs"], dict) and set(paired["pairs"]) == set(PAIRS), "All paired training identities")
    for row in paired["pairs"].values():
        require(isinstance(row, dict) and set(row) == {"initial_tensor_sha256", "orders_sha256", "initial_state_kind"}
                and row["initial_state_kind"] == "original_initialization"
                and _sha(row["initial_tensor_sha256"]) and _sha(row["orders_sha256"]), "Original initialization and complete order binding")
    checkpoints = bindings["inherited_checkpoints"]
    expected_fits = {row["name"]: row for row in restore_manifest(plan)}
    require(isinstance(checkpoints, dict) and set(checkpoints) == set(expected_fits), "All six inherited checkpoints")
    for name, row in checkpoints.items():
        require(isinstance(row, dict) and set(row) == {"model_class", "checkpoint_sha256", "weights_sha256", "configuration_sha256"}
                and row["model_class"] == expected_fits[name]["model_class"]
                and all(_sha(row[key]) for key in ("checkpoint_sha256", "weights_sha256", "configuration_sha256")), "Actual inherited class and tensor/configuration bindings")
    runtime = bindings["runtime"]
    require(isinstance(runtime, dict) and set(runtime) == {"python", "numpy", "torch", "mujoco", "gymnasium", "machine", "threads", "determinism"}
            and all(isinstance(runtime[key], str) and runtime[key].strip() for key in runtime if key != "threads")
            and type(runtime["threads"]) is int and runtime["threads"] == plan["threads"], "Explicit runtime bindings")
    streams = bindings["fresh_streams"]
    require(isinstance(streams, dict) and set(streams) == {"namespace", "role_manifest_sha256", "prior_lineage_sha256", "generator_separation_receipt_sha256"}
            and streams["namespace"] == plan["rng_namespace"]
            and all(_sha(streams[key]) for key in streams if key != "namespace"), "External freshness verification receipts required")
    role_sha = hashlib.sha256(json.dumps(role_manifest(plan), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(streams["role_manifest_sha256"] == role_sha, "Freshness receipt binds exact symbolic roles")
    caps = bindings["capacity_and_caps"]
    require(isinstance(caps, dict) and set(caps) == {"execution_seconds", "audit_seconds", "capacity_receipt_sha256", "rehearsal_receipt_sha256"}
            and all(type(caps[key]) is int and caps[key] > 0 for key in ("execution_seconds", "audit_seconds"))
            and all(_sha(caps[key]) for key in ("capacity_receipt_sha256", "rehearsal_receipt_sha256")), "Measured capacity receipts and positive explicit caps")
    audit = bindings["audit_contract"]
    require(isinstance(audit, dict) and set(audit) == {"member_schema_sha256", "cost_schema_sha256", "independent_review_sha256"}
            and all(_sha(value) for value in audit.values()), "External independent auditor/member/cost contract")
    return {"validated_scope": "metadata_structure_only", "execution_authorized": False,
            "remaining_checks": ["actual source and lineage bytes", "all original data/orders/tensors and inherited classes",
                "numerical RNG derivation and complete generator-state exclusions", "whole-tree auditor and measured capacity",
                "new frozen protocol and externally authenticated launch"]}
