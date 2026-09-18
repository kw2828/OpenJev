"""Saved-output audit of the prospective evaluation of inherited Reacher fits.

No learned model/controller is imported. Native replay and numerical summaries
reuse frozen arithmetic, while inheritance, role separation and cost accounting
are checked here. The caller authenticates the current source/runtime/plan.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import audit_reacher_reward_residual_study as prior
import torch

from openjev.research import reacher_random_streams as streams

base = prior.base
ROOT = Path(__file__).resolve().parents[1]
VERSION = "reacher-reward-residual-control-v2"
PLAN_CHANGES = {
    "version",
    "study",
    "sources",
    "runtime",
    "stop",
    "data",
    "models",
    "control",
    *streams.FIELDS,
}
SOURCE_KEYS = {
    "execution_path",
    "plan_path",
    "plan_sha256",
    "invalid_receipt_path",
    "invalid_receipt_sha256",
    "inheritance_receipt_path",
    "inheritance_receipt_sha256",
    "all_fits_boundary_sha256",
    "failure_receipt_sha256",
    "members",
    "inherited_fit_wall_seconds",
    "prior_invalid_attempt_wall_seconds",
}
COPIES = {
    "inherited-plan.json": ("plan_path", "plan_sha256"),
    "invalid-attempt.json": ("invalid_receipt_path", "invalid_receipt_sha256"),
    "fit-inheritance.json": ("inheritance_receipt_path", "inheritance_receipt_sha256"),
}
FIT_FILES = prior.FIT_FILES | {"completed.json"}


def expected_members(plan):
    return base.expected_members(plan) | {
        "train-provenance.json",
        "fit-provenance.json",
        "inherited-plan.json",
        "invalid-attempt.json",
        "fit-inheritance.json",
        "source-all-fits-completed.json",
        "source-failed.json",
        "random-streams.json",
        "inherited-fits-ready.json",
        "evaluation-started.json",
        *(f"fits/{name}/initial-weights.pt" for name in base.configurations(plan)),
    }


def validate_inheritance(plan, execution):
    """Authenticate the complete unselected fit set, without prior evaluation reads."""
    source = plan["fit_source"]
    base.require(set(source) == SOURCE_KEYS, "Fit source schema")
    base.require(base.read(execution / "fit-provenance.json") == source, "Copied fit provenance")
    for member, (path_key, hash_key) in COPIES.items():
        base.require(
            base.sha(ROOT / source[path_key]) == source[hash_key]
            and base.sha(execution / member) == source[hash_key],
            f"Inherited source hash: {member}",
        )
    parent = base.read(execution / "inherited-plan.json")
    base.require(
        set(plan) == set(parent) | {"fit_source", "random_stream_contract"}, "Inherited plan membership"
    )
    for key in set(parent) - PLAN_CHANGES:
        base.require(plan[key] == parent[key], f"Inherited plan field changed: {key}")
    base.require(
        plan["study"] == VERSION and type(plan["version"]) is int and plan["version"] == 2,
        "Evaluation study identity",
    )
    names = base.configurations(parent)
    base.require(
        parent["kinds"] == ["free", "residual"]
        and parent["steps"] == 50
        and bool(parent["fit_seeds"])
        and all(type(seed) is int and seed >= 0 for seed in parent["fit_seeds"])
        and len(names) == len(set(names))
        and len(parent["fit_order"]) == len(names)
        and set(parent["fit_order"]) == set(names),
        "Inherited fit seed/order membership",
    )
    members = {f"fits/{name}/{file}" for name in names for file in FIT_FILES}
    base.require(set(source["members"]) == members, "Inherited fit member set")
    original = ROOT / source["execution_path"]
    base.require(
        original.is_dir() and not original.is_symlink() and not (original / "completed.json").exists(),
        "Original attempt must remain stopped",
    )
    for member, digest in source["members"].items():
        base.require(
            base.sha(original / member) == digest and base.sha(execution / member) == digest,
            f"Inherited fit bytes changed: {member}",
        )
    for member, old_name, key in (
        ("source-all-fits-completed.json", "all-fits-completed.json", "all_fits_boundary_sha256"),
        ("source-failed.json", "failed.json", "failure_receipt_sha256"),
    ):
        base.require(
            base.sha(original / old_name) == source[key] and base.sha(execution / member) == source[key],
            f"Inherited terminal hash: {member}",
        )
    invalid = base.read(execution / "invalid-attempt.json")
    inheritance = base.read(execution / "fit-inheritance.json")
    boundary = base.read(execution / "source-all-fits-completed.json")
    failure = base.read(execution / "source-failed.json")
    base.require(
        invalid["status"] == "invalid_protocol_stopped"
        and invalid["plan_sha256"] == source["plan_sha256"]
        and invalid["process_exit_code"] == 130
        and invalid["resumable"] is False
        and invalid["failure_receipt_sha256"] == source["failure_receipt_sha256"]
        and invalid["all_six_fits_completed_sha256"] == source["all_fits_boundary_sha256"]
        and invalid["completed_fit_members"] == source["members"],
        "Invalid attempt inheritance binding",
    )
    fit_receipts = {name: source["members"][f"fits/{name}/completed.json"] for name in names}
    base.require(
        inheritance["status"] == "all_six_fits_authenticated_for_unselected_inheritance"
        and inheritance["source_plan_sha256"] == source["plan_sha256"]
        and inheritance["source_invalid_receipt_sha256"] == source["invalid_receipt_sha256"]
        and inheritance["fit_receipts_sha256"] == fit_receipts
        and set(inheritance["fits"]) == set(names)
        and type(inheritance["new_fits"]) is int
        and inheritance["new_fits"] == 0
        and type(inheritance["new_model_calls"]) is int
        and inheritance["new_model_calls"] == 0
        and inheritance["paired_initial_tensors_identical"] is True
        and inheritance["training_roles_have_no_seed_collisions"] is True
        and inheritance["efficacy_metrics_read"] is False,
        "Unselected inheritance receipt binding",
    )
    base.require(
        set(boundary) == {"plan_sha256", "fit_order", "files", "unix_time", "elapsed_seconds"}
        and boundary["plan_sha256"] == source["plan_sha256"]
        and boundary["fit_order"] == parent["fit_order"]
        and boundary["files"]
        == {f"fits/{name}/completed.json": digest for name, digest in fit_receipts.items()},
        "Source all-fits boundary binding",
    )
    inherited_wall = base.finite_number(
        source["inherited_fit_wall_seconds"], "Inherited fit timing", positive=True
    )
    old_wall = base.finite_number(
        source["prior_invalid_attempt_wall_seconds"], "Prior attempt timing", positive=True
    )
    source_elapsed = base.finite_number(
        boundary["elapsed_seconds"], "Source fit boundary timing", positive=True
    )
    base.finite_number(boundary["unix_time"], "Source boundary timestamp", positive=True)
    base.require(
        inheritance["inherited_fit_wall_seconds"] == inherited_wall
        and invalid["wall_seconds"] == old_wall
        and failure["wall_seconds"] == old_wall
        and inherited_wall <= source_elapsed <= old_wall,
        "Inherited/prior cost binding",
    )
    return parent, inheritance


def validate_streams(plan, execution):
    """Re-expand every NumPy stream and spawned filter state; draw no samples."""
    contract = plan["random_stream_contract"]
    base.require(
        set(contract)
        == {"namespace", "registry", "generators", "priors", "intentional_reuse", "draws_for_manifest"},
        "Random stream contract schema",
    )
    base.require(base.read(execution / "random-streams.json") == contract, "Copied random stream contract")
    base.require(
        type(contract["draws_for_manifest"]) is int
        and contract["draws_for_manifest"] == 0
        and isinstance(contract["intentional_reuse"], list)
        and all(isinstance(item, str) and bool(item.strip()) for item in contract["intentional_reuse"]),
        "Random stream declaration",
    )
    base.require(contract["namespace"] == VERSION, "Random stream namespace")
    bases = streams.domain_bases(contract["namespace"])
    base.require(all(plan[key] == value for key, value in bases.items()), "Domain-separated seed bases")
    wanted = {
        (str((ROOT / source["plan_path"]).resolve()), source["plan_sha256"], train)
        for source, train in ((plan["training_source"], True), (plan["fit_source"], False))
    }
    actual, priors = set(), []
    base.require(
        isinstance(contract["priors"], list) and len(contract["priors"]) == 2,
        "Both prior stream registries required",
    )
    for item in contract["priors"]:
        base.require(
            set(item) == {"plan_path", "plan_sha256", "include_train"}
            and type(item["include_train"]) is bool,
            "Prior stream schema",
        )
        path = ROOT / item["plan_path"]
        base.require(base.sha(path) == item["plan_sha256"], "Prior stream plan hash")
        actual.add((str(path.resolve()), item["plan_sha256"], item["include_train"]))
        priors.append(streams.concrete_streams(base.read(path), include_train=item["include_train"]))
    base.require(actual == wanted, "Prior stream identity/coverage")
    registry = streams.validate_separation(plan, prior_registries=priors)
    base.require(contract["registry"] == registry, "Expanded stream registry mismatch")
    manifest = streams.generator_manifest(registry)
    base.require(contract["generators"] == manifest, "Expanded generator state mismatch")
    # Inherited training is intentionally reused, but its own roles must be distinct.
    old_train = {
        key: value
        for key, value in streams.concrete_streams(
            base.read(ROOT / plan["training_source"]["plan_path"]), include_train=True
        ).items()
        if key.startswith("train/")
    }
    base.require(not streams.collisions(old_train), "Inherited training cross-role collision")
    return {
        "streams": len(registry),
        "generators": len(manifest),
        "priors_checked": 2,
        "draws_for_manifest": 0,
        "cross_role_seed_and_initial_state_disjoint": True,
    }


def audit_boundaries(plan, expected_hash, execution, completed):
    started = base.read(execution / "started.json")
    ready = base.read(execution / "inherited-fits-ready.json")
    evaluation = base.read(execution / "evaluation-started.json")
    base.require(
        set(ready)
        == {
            "plan_sha256",
            "fit_order",
            "files",
            "unix_time",
            "elapsed_seconds",
            "inherited_fit_wall_seconds",
            "new_fits",
        },
        "Inherited ready boundary schema",
    )
    base.require(
        set(evaluation)
        == {
            "plan_sha256",
            "inherited_fits_ready_sha256",
            "random_streams_sha256",
            "unix_time",
            "elapsed_seconds",
        },
        "Evaluation boundary schema",
    )
    for item in (started, ready, evaluation):
        base.require(item["plan_sha256"] == expected_hash, "Phase plan binding")
        base.finite_number(item["unix_time"], "Phase timestamp", positive=True)
    base.require(
        ready["fit_order"] == plan["fit_order"]
        and type(ready["new_fits"]) is int
        and ready["new_fits"] == 0
        and ready["inherited_fit_wall_seconds"] == plan["fit_source"]["inherited_fit_wall_seconds"],
        "Inherited boundary identity",
    )
    wanted = {
        f"fits/{name}/completed.json": plan["fit_source"]["members"][f"fits/{name}/completed.json"]
        for name in base.configurations(plan)
    }
    base.require(ready["files"] == wanted, "Inherited boundary fit membership/hash")
    base.require(
        evaluation["inherited_fits_ready_sha256"] == base.sha(execution / "inherited-fits-ready.json")
        and evaluation["random_streams_sha256"] == base.sha(execution / "random-streams.json"),
        "Evaluation boundary dependency hash",
    )
    ready_elapsed = base.finite_number(ready["elapsed_seconds"], "Ready boundary timing", positive=True)
    eval_elapsed = base.finite_number(
        evaluation["elapsed_seconds"], "Evaluation boundary timing", positive=True
    )
    base.require(
        ready_elapsed <= eval_elapsed <= completed["wall_seconds"]
        and started["unix_time"] <= ready["unix_time"] <= evaluation["unix_time"],
        "Inheritance/evaluation chronology",
    )
    base.require(
        completed["evaluation_started_elapsed_seconds"] == eval_elapsed,
        "Completion evaluation timing binding",
    )
    return {
        "fit_order": ready["fit_order"],
        "inherited_fits_ready_sha256": base.sha(execution / "inherited-fits-ready.json"),
        "evaluation_started_sha256": base.sha(execution / "evaluation-started.json"),
        "inherited_fits_ready_elapsed_seconds": ready_elapsed,
        "evaluation_started_elapsed_seconds": eval_elapsed,
        "scope": "Source-bound logged phase chronology, not an independent process observer.",
    }


def audit_saved(plan, expected_plan_sha256, execution: Path, out: Path):
    audit_start = time.monotonic()
    execution, out = Path(execution), Path(out)
    base.require(not out.exists() and not out.is_symlink(), "Audit output already exists")
    base.require(
        isinstance(expected_plan_sha256, str)
        and len(expected_plan_sha256) == 64
        and all(c in "0123456789abcdef" for c in expected_plan_sha256),
        "External plan SHA",
    )
    completed = base.read(execution / "completed.json")
    completion_hash = base.sha(execution / "completed.json")
    expected = expected_members(plan)
    actual = {str(path.relative_to(execution)) for path in execution.rglob("*") if path.is_file()}
    base.require(
        not execution.is_symlink() and not any(path.is_symlink() for path in execution.rglob("*")),
        "Execution symlinks forbidden",
    )
    base.require(
        actual == expected | {"completed.json"} and set(completed["files"]) == expected,
        "Execution member set",
    )
    base.require(
        set(completed)
        == {
            "status",
            "plan_sha256",
            "fits",
            "new_fits",
            "astra_calls",
            "wall_seconds",
            "evaluation_started_elapsed_seconds",
            "inherited_fit_wall_seconds",
            "prior_invalid_attempt_wall_seconds",
            "files",
        }
        and completed["status"] == "completed"
        and completed["plan_sha256"] == expected_plan_sha256
        and type(completed["fits"]) is int
        and completed["fits"] == len(base.configurations(plan))
        and type(completed["new_fits"]) is int
        and completed["new_fits"] == 0
        and type(completed["astra_calls"]) is int
        and completed["astra_calls"] == 0,
        "Execution completion identity/schema",
    )
    wall = base.finite_number(completed["wall_seconds"], "Execution timing", positive=True)
    base.require(wall <= plan["cap_seconds"], "Whole-run cap exceeded")
    for member in expected:
        base.require(
            base.sha(execution / member) == completed["files"][member], f"Execution member hash: {member}"
        )
    parent, inheritance = validate_inheritance(plan, execution)
    stream_summary = validate_streams(plan, execution)
    cohort_plan = prior.validate_training_source(plan, execution)
    train = base.load_records(execution / "train", plan["train_episodes"])
    prediction = base.load_records(execution / "prediction", plan["prediction_episodes"])
    replay = {
        "train": base.audit_cohort(cohort_plan, train, "train"),
        "prediction": base.audit_cohort(plan, prediction, "prediction"),
    }
    fits, initial = {}, {}
    for name in parent["fit_order"]:
        fits[name], initial[name] = prior.audit_fit(parent, execution, name, train)
        base.require(
            {key: value for key, value in fits[name].items() if key != "epochs"} == inheritance["fits"][name],
            "Inherited fit audit metadata mismatch",
        )
    pairs = {}
    for seed in plan["fit_seeds"]:
        left, right = initial[f"free-{seed}"], initial[f"residual-{seed}"]
        base.require(all(torch.equal(left[key], right[key]) for key in left), "Paired initial tensors differ")
        base.require(
            fits[f"free-{seed}"]["minibatch_order_sha256"]
            == fits[f"residual-{seed}"]["minibatch_order_sha256"],
            "Paired minibatch order differs",
        )
        pairs[str(seed)] = {
            "initial_tensors_identical": True,
            "parameters": fits[f"free-{seed}"]["parameters"],
            "minibatch_order_sha256": fits[f"free-{seed}"]["minibatch_order_sha256"],
        }
    del initial
    inherited_wall = sum(item["wall_seconds"] for item in fits.values())
    base.require(
        math.isclose(
            inherited_wall, plan["fit_source"]["inherited_fit_wall_seconds"], rel_tol=1e-12, abs_tol=1e-9
        ),
        "Inherited fit cost arithmetic",
    )
    for key in ("inherited_fit_wall_seconds", "prior_invalid_attempt_wall_seconds"):
        base.finite_number(completed[key], f"Completion {key}", positive=True)
        base.require(completed[key] == plan["fit_source"][key], "Completion inherited/prior timing binding")
    boundary = audit_boundaries(plan, expected_plan_sha256, execution, completed)
    predictions = {
        name: base.prediction_metrics(
            plan,
            prediction,
            base.load_npz(execution / "predictions" / f"{name}.npz", {"one", "reward", "multi"}),
        )
        for name in base.configurations(plan)
    }
    controls = {}
    for panel in base.PANELS:
        controls[panel] = {}
        for arm in base.arms(plan, panel):
            records = base.load_records(execution / "control" / panel / arm, plan["control_episodes"])
            replay[f"{panel}/{arm}"] = base.audit_cohort(plan, records, "control", panel)
            values = base.load_npz(execution / "control" / panel / f"{arm}-planning.npz")
            controls[panel][arm] = base.control_metrics(plan, records, values, arm)
    decision_and_setup = sum(
        item["setup_seconds"] + item["decision_wall_seconds"]
        for panel in controls.values()
        for item in panel.values()
    )
    base.require(
        decision_and_setup <= wall - boundary["evaluation_started_elapsed_seconds"] + 1e-6,
        "Control timings exceed new evaluation interval",
    )
    gate, comparisons = prior.qualify(plan, predictions, controls)
    costs = {
        "inherited_fit_wall_seconds": inherited_wall,
        "prior_invalid_attempt_wall_seconds": completed["prior_invalid_attempt_wall_seconds"],
        "new_evaluation_wall_seconds": wall,
        "new_control_setup_and_decision_seconds": decision_and_setup,
        "cumulative_attempt_wall_seconds": completed["prior_invalid_attempt_wall_seconds"] + wall,
        "fresh_evaluation_plus_inherited_fits_seconds": wall + inherited_wall,
        "audit_validation_wall_seconds": time.monotonic() - audit_start,
        "new_fits": 0,
        "accounting": "The prior invalid attempt already includes inherited fits. Cumulative cost adds prior attempt plus new evaluation, never all three. Fresh evaluation plus inherited fitting is a descriptive successful-pipeline equivalent, not actual cumulative cost. Audit validation is outside execution timing and excludes publication writes and final rehashing.",
    }
    summary = {
        "version": VERSION,
        "status": "completed",
        "plan_sha256": expected_plan_sha256,
        "execution_completed_sha256": completion_hash,
        "saved_output_only": True,
        "new_model_calls": 0,
        "new_policy_calls": 0,
        "new_mpc_calls": 0,
        "new_fits": 0,
        "native_transitions_checked": sum(item["transitions"] for item in replay.values()),
        "native_max_abs_error": max(item["max_abs_error"] for item in replay.values()),
        "wall_seconds": wall,
        "costs": costs,
        "training_source": plan["training_source"],
        "fit_source": plan["fit_source"],
        "random_streams": stream_summary,
        "paired_training": pairs,
        "phase_boundary": boundary,
        "fits": fits,
        "predictions": predictions,
        "control": controls,
        "cohorts": replay,
        "paired_descriptive_comparisons": comparisons,
        "continuation_gate": gate,
        "limits": [
            "All final fits are inherited byte-for-byte without selection or new optimization. The original attempt remains invalid and stopped; its partial evaluation is not a result of this study.",
            "Fresh concrete seeds and generator initial states are checked across every role and against both prior plans. Controller/panel pairing intentionally reuses named streams; no samples are drawn to build the manifest.",
            "Native replay checks all inherited training and fresh evaluation transitions; saved prediction/control arithmetic is recomputed without learned-model, policy or MPC calls. Recorded training checks do not rerun optimization.",
            "The treatment uses supplied actuator-cost and noise-distribution knowledge. Privileged raw state and realized actuator noise remain audit-only for learned models.",
            "Inherited-fit and new-evaluation costs are separate. Prior invalid-attempt cost already contains the inherited fits. Phase receipts provide source-bound logged chronology, not an independent process observer.",
            "All paired seeds and ordinary/shift panels retain the parent criteria. Reset interventions and case-paired intervals are descriptive; intervals are conditional on inherited fits, not architecture uncertainty.",
            "Setup and batch decision times are charged; per-case latency is amortized, not a deadline. The execution cap is cooperative. Physics references retain their supplied-model privilege; floor scores are placeholders.",
            "This is a known-reward-structure qualification, not a new RL algorithm, memory advantage, connectome result or ICLR claim.",
        ],
    }
    base.require(base.sha(execution / "completed.json") == completion_hash, "Completion changed during audit")
    for member in expected:
        base.require(
            base.sha(execution / member) == completed["files"][member], "Execution changed during audit"
        )
    out.mkdir(parents=True, exist_ok=False)
    base.write(out / "summary.json", summary)
    lines = [
        "# Reacher inherited-model evaluation",
        "",
        f"Continuation gate: **{'PASS' if gate['passed'] else 'FAIL'}** ({sum(check['passed'] for check in gate['checks'])}/{len(gate['checks'])} checks).",
        "",
        f"{len(fits)} unchanged inherited fits; 0 new fits; {summary['native_transitions_checked']:,} native transitions replayed.",
        "",
        f"Inherited fitting: {inherited_wall:.6f} s. Prior invalid attempt including fitting: {costs['prior_invalid_attempt_wall_seconds']:.6f} s. New evaluation: {wall:.6f} s.",
        "",
        "| Panel | Arm | Mean episode cost |",
        "|---|---|---:|",
    ]
    for panel, results in controls.items():
        lines.extend(f"| {panel} | {name} | {item['mean_cost']:.6f} |" for name, item in results.items())
    lines.extend(["", *(f"- {limit}" for limit in summary["limits"])])
    (out / "README.md").write_text("\n".join(lines) + "\n")
    base.write(
        out / "receipt.json",
        {
            "status": "completed",
            "version": VERSION,
            "plan_sha256": expected_plan_sha256,
            "source_sha256": plan["sources"],
            "runtime": plan["runtime"],
            "plan_canonical_sha256": hashlib.sha256(
                json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest(),
            "execution_completed_sha256": completion_hash,
            "execution_members": completed["files"],
            "training_source": plan["training_source"],
            "fit_source": plan["fit_source"],
            "random_stream_contract": plan["random_stream_contract"],
            "costs": costs,
            "saved_output_only": True,
            "files": {name: base.sha(out / name) for name in ("summary.json", "README.md")},
        },
    )
    return summary
