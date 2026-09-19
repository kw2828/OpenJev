"""Recovery-contract checks using synthetic files and no scored environment calls."""

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "reacher_residual_control_fixture", SCRIPTS / "reacher_reward_residual_control.py"
)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

FIT_ORDER = [
    "free-271", "residual-271", "residual-283", "free-283", "free-293", "residual-293",
]
CHANGED_FIELDS = {
    "version", "study", "sources", "runtime", "stop", "data", "models", "control",
    "prediction_seed", "control_seed", "schedule_seed", "noise_seed",
    "exploration_seed", "candidate_seed", "filter_seed", "bootstrap_seed",
}


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    s.write(path, value)
    return s.sha(path)


def fake_parent(tmp_path, monkeypatch, *, cap_seconds=1800):
    """Only versioned protocol metadata is read; every execution file is synthetic."""
    original = s.read(ROOT / "evidence/reacher-world-model-v1/protocol/plan.json")
    parent = s.read(ROOT / "evidence/reacher-reward-residual-v1/protocol/plan.json")
    root = tmp_path / "synthetic-root"
    root.mkdir()
    monkeypatch.setattr(s, "ROOT", root)
    monkeypatch.setattr(s.legacy, "ROOT", root)
    monkeypatch.setattr(s.base, "runtime", lambda: {"synthetic_fixture": True})
    for name in s.SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Synthetic source identity: " + name)

    train_dir = root / "original-training"
    train_dir.mkdir()
    for name in ("train.npz", "train.json"):
        (train_dir / name).write_bytes(b"synthetic training identity only: " + name.encode())
    training_members = {name: s.sha(train_dir / name) for name in ("train.npz", "train.json")}
    original_path = root / "original-plan.json"
    original_sha = put(original_path, original)
    training_audit_path = root / "original-audit.json"
    training_audit_sha = put(training_audit_path, {
        "status": "completed", "plan_sha256": original_sha,
        "execution_members": training_members, "synthetic_fixture": True,
    })
    training_source = {
        "execution_path": str(train_dir), "plan_path": str(original_path),
        "plan_sha256": original_sha, "audit_receipt_path": str(training_audit_path),
        "audit_receipt_sha256": training_audit_sha, "members": training_members,
        "cohort_plan": {key: original[key] for key in s.legacy.COHORT_KEYS},
    }
    parent.update(training_source=training_source, cap_seconds=cap_seconds,
                  runtime=s.base.runtime(), sources={name: s.sha(root / name) for name in s.legacy.SOURCES})
    parent_path = root / "residual-parent-plan.json"
    parent_sha = put(parent_path, parent)

    def validate_synthetic_parent(path, digest):
        # Only the old source/runtime gate is substituted. Its training-source
        # authentication still executes against the synthetic immutable files.
        assert s.sha(path) == digest
        result = s.read(path)
        s.legacy.authenticate_training(result["training_source"])
        return result

    monkeypatch.setattr(s.legacy, "validate", validate_synthetic_parent)
    execution = root / "stopped-parent"
    execution.mkdir()
    members, fit_receipts = {}, {}
    for name in FIT_ORDER:
        kind, seed = name.rsplit("-", 1)
        folder = execution / "fits" / name
        folder.mkdir(parents=True)
        # No PyTorch checkpoint or model inference is needed for these hash checks.
        (folder / "initial-weights.pt").write_bytes(b"paired initial " + seed.encode())
        (folder / "weights.pt").write_bytes(b"synthetic final " + name.encode())
        put(folder / "training.json", [{"synthetic_fixture": True}])
        files = {filename: s.sha(folder / filename) for filename in (
            "initial-weights.pt", "weights.pt", "training.json",
        )}
        receipt = {
            "kind": kind, "seed": int(seed), "updates": 1152, "parameters": 36805,
            "initial_weights_sha256": files["initial-weights.pt"],
            "minibatch_order_sha256": "synthetic paired order " + seed,
            "residual_reward": kind == "residual", "noise_std": parent["noise_std"],
            "wall_seconds": 1.0, "files": files,
        }
        fit_receipts[name] = put(folder / "completed.json", receipt)
        members.update({f"fits/{name}/{filename}": s.sha(folder / filename) for filename in (
            "initial-weights.pt", "weights.pt", "training.json", "completed.json",
        )})
    boundary_sha = put(execution / "all-fits-completed.json", {
        "plan_sha256": parent_sha, "fit_order": FIT_ORDER,
        "files": {f"fits/{name}/completed.json": fit_receipts[name] for name in FIT_ORDER},
        "elapsed_seconds": 7.0,
    })
    failure_sha = put(execution / "failed.json", {
        "error": "KeyboardInterrupt()", "wall_seconds": 9.0,
        "progress": {"phase": "control", "synthetic_fixture": True},
    })
    invalid_path = root / "invalid.json"
    invalid_sha = put(invalid_path, {
        "status": "invalid_protocol_stopped", "resumable": False, "process_exit_code": 130,
        "plan_sha256": parent_sha, "failure_receipt_sha256": failure_sha,
        "all_six_fits_completed_sha256": boundary_sha, "completed_fit_members": members,
        "wall_seconds": 9.0,
    })
    inheritance_path = root / "inheritance.json"
    inheritance_sha = put(inheritance_path, {
        "status": "all_six_fits_authenticated_for_unselected_inheritance",
        "source_plan_sha256": parent_sha, "source_invalid_receipt_sha256": invalid_sha,
        "new_fits": 0, "efficacy_metrics_read": False, "fit_receipts_sha256": fit_receipts,
        "new_model_calls": 0, "paired_initial_tensors_identical": True,
        "training_roles_have_no_seed_collisions": True,
        "fits": {name: {"synthetic_fixture": True} for name in FIT_ORDER},
        "inherited_fit_wall_seconds": 6.0,
    })
    source = {
        "execution_path": str(execution), "plan_path": str(parent_path), "plan_sha256": parent_sha,
        "invalid_receipt_path": str(invalid_path), "invalid_receipt_sha256": invalid_sha,
        "inheritance_receipt_path": str(inheritance_path), "inheritance_receipt_sha256": inheritance_sha,
        "all_fits_boundary_sha256": boundary_sha, "failure_receipt_sha256": failure_sha,
        "members": members, "inherited_fit_wall_seconds": 6.0,
        "prior_invalid_attempt_wall_seconds": 9.0,
    }
    return SimpleNamespace(root=root, parent=parent, original=original, source=source, execution=execution)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    fixture = fake_parent(tmp_path, monkeypatch)
    result = s.prepare(tmp_path / "protocol", fit_source=fixture.source)
    fixture.path = Path(result["path"])
    fixture.plan = s.read(fixture.path)
    fixture.digest = result["sha256"]
    return fixture


def test_prepare_preserves_all_parent_settings_and_all_six_fits(prepared):
    assert s.CHANGED_PARENT_FIELDS == CHANGED_FIELDS
    assert s.NEW_FIELDS == {"fit_source", "random_stream_contract"}
    assert prepared.plan["fit_order"] == FIT_ORDER
    assert prepared.plan["fit_seeds"] == [271, 283, 293]
    assert prepared.plan["kinds"] == ["free", "residual"]
    assert s.validate(prepared.path, prepared.digest) == prepared.plan
    for key, value in prepared.parent.items():
        if key not in CHANGED_FIELDS:
            assert prepared.plan[key] == value, key


@pytest.mark.parametrize("key", [
    "kinds", "fit_seeds", "fit_order", "epochs", "batch_size", "learning_rate",
    "hidden_size", "train_episodes", "prediction_episodes", "control_episodes",
    "steps", "noise_std", "ordinary_gap", "shift_gap", "candidates",
    "planning_horizon", "action_block", "particles", "filter_bandwidth",
    "rollout_horizon", "rollout_weight", "reward_scale", "bootstrap_samples",
    "threads", "cap_seconds", "criteria", "task", "training_source", "train_seed",
])
def test_frozen_parent_settings_reject_drift_even_with_rehashed_plan(prepared, key):
    plan = copy.deepcopy(prepared.plan)
    old = plan[key]
    if isinstance(old, list):
        plan[key] = old[:-1]
    elif isinstance(old, dict):
        plan[key] = {**old, "unauthorized": True}
    elif isinstance(old, str):
        plan[key] += " changed"
    else:
        plan[key] = old + 1
    digest = put(prepared.path, plan)
    with pytest.raises(ValueError, match="Frozen parent setting changed"):
        s.validate(prepared.path, digest)


@pytest.mark.parametrize("operation", ["drop", "add"])
def test_plan_cannot_change_field_membership(prepared, operation):
    plan = copy.deepcopy(prepared.plan)
    if operation == "drop":
        del plan["criteria"]
    else:
        plan["replacement_fit"] = "free-999"
    with pytest.raises(ValueError, match="field membership"):
        s.validate_parent_settings(plan, prepared.parent)


@pytest.mark.parametrize("mutation", ["omit_arm", "omit_member", "replace_fit", "extra_member"])
def test_inheritance_requires_every_original_fit_member(prepared, mutation):
    source = copy.deepcopy(prepared.source)
    if mutation == "omit_arm":
        source["members"] = {key: value for key, value in source["members"].items()
                             if not key.startswith("fits/residual-")}
    elif mutation == "omit_member":
        del source["members"]["fits/free-271/weights.pt"]
    elif mutation == "replace_fit":
        source["members"] = {key.replace("free-271", "free-999"): value
                             for key, value in source["members"].items()}
    else:
        source["members"]["fits/free-271/alternate-epoch.pt"] = "0" * 64
    with pytest.raises(ValueError, match="Every original final fit"):
        s.authenticate_fit_source(source)


@pytest.mark.parametrize("member", ["initial-weights.pt", "weights.pt", "training.json", "completed.json"])
def test_changed_inherited_file_rejected_before_output_or_evaluation(prepared, tmp_path, member):
    path = prepared.execution / "fits/free-271" / member
    path.write_bytes(path.read_bytes() + b"\nmodified")
    out = tmp_path / "must-not-start"
    with pytest.raises(ValueError, match="Inherited identity mismatch"):
        s.run(prepared.path, prepared.digest, out)
    assert not out.exists()


def test_inherited_symlink_and_completed_parent_are_rejected(prepared):
    weight = prepared.execution / "fits/free-271/weights.pt"
    target = weight.with_suffix(".original")
    weight.rename(target)
    weight.symlink_to(target)
    with pytest.raises(ValueError, match="Inherited identity mismatch"):
        s.authenticate_fit_source(prepared.source)
    weight.unlink()
    target.rename(weight)
    put(prepared.execution / "completed.json", {"status": "completed"})
    with pytest.raises(ValueError, match="must remain stopped"):
        s.authenticate_fit_source(prepared.source)


def test_training_files_remain_authenticated(prepared):
    training = prepared.parent["training_source"]
    path = Path(training["execution_path"]) / "train.npz"
    path.write_bytes(b"changed original training")
    with pytest.raises(ValueError, match="Training parent member changed"):
        s.validate(prepared.path, prepared.digest)


@pytest.mark.parametrize("field,value", [
    ("new_fits", 1), ("new_model_calls", 1), ("efficacy_metrics_read", True),
    ("paired_initial_tensors_identical", False), ("training_roles_have_no_seed_collisions", False),
])
def test_inheritance_provenance_requirements_are_semantic_not_only_hashes(prepared, field, value):
    source = copy.deepcopy(prepared.source)
    path = Path(source["inheritance_receipt_path"])
    receipt = s.read(path)
    receipt[field] = value
    source["inheritance_receipt_sha256"] = put(path, receipt)
    with pytest.raises(ValueError, match="Fit inheritance identity/scope"):
        s.authenticate_fit_source(source)


@pytest.mark.parametrize("field", ["inherited_fit_wall_seconds", "prior_invalid_attempt_wall_seconds"])
def test_historical_costs_cannot_be_relabelled(prepared, field):
    source = copy.deepcopy(prepared.source)
    source[field] += 1
    with pytest.raises(ValueError, match="cost"):
        s.authenticate_fit_source(source)


def test_changed_bound_source_runtime_and_external_plan_hash_are_rejected(prepared, monkeypatch):
    with pytest.raises(ValueError, match="Inherited identity mismatch"):
        s.validate(prepared.path, "0" * 64)
    monkeypatch.setattr(s.base, "runtime", lambda: {"synthetic_fixture": "changed"})
    with pytest.raises(ValueError, match="Frozen runtime changed"):
        s.validate(prepared.path, prepared.digest)
    monkeypatch.setattr(s.base, "runtime", lambda: {"synthetic_fixture": True})
    source_path = prepared.root / s.SOURCES[-1]
    source_path.write_text("changed evaluation implementation")
    with pytest.raises(ValueError, match="Inherited identity mismatch"):
        s.validate(prepared.path, prepared.digest)


def test_contract_covers_both_old_evaluations_and_actual_original_training(prepared, monkeypatch):
    observed = []
    original_validate = s.validate_separation

    def capture(plan, *, prior_registries):
        observed.extend(prior_registries)
        return original_validate(plan, prior_registries=prior_registries)

    monkeypatch.setattr(s, "validate_separation", capture)
    contract = s.stream_contract(prepared.plan)
    assert len(observed) == 2
    assert observed[0] == s.concrete_streams(prepared.original, include_train=True)
    assert observed[1] == s.concrete_streams(prepared.parent, include_train=False)
    assert "train/actuator_noise/767" in observed[0]
    assert observed[0]["train/actuator_noise/767"] == prepared.original["noise_seed"] + 767
    assert observed[0]["train/actuator_noise/767"] != prepared.parent["noise_seed"] + 767
    assert not any(role.startswith("train/") for role in observed[1])
    assert contract["registry"] == s.concrete_streams(prepared.plan)
    assert contract["generators"] == s.generator_manifest(contract["registry"])
    assert len(contract["registry"]) == 692
    assert len(contract["generators"]) == 820
    assert contract["draws_for_manifest"] == 0
    assert all(item["draws_for_manifest"] == 0 for item in contract["generators"].values())


@pytest.mark.parametrize("prior_role", [
    "original_training_last", "original_evaluation", "residual_evaluation",
])
def test_stream_contract_rejects_overlap_with_every_prior_cohort(prepared, prior_role):
    plan = copy.deepcopy(prepared.plan)
    prior = prepared.original if prior_role != "residual_evaluation" else prepared.parent
    if prior_role == "original_training_last":
        collision = prior["noise_seed"] + prior["train_episodes"] - 1
    else:
        collision = prior["noise_seed"] + 200000
    plan["candidate_seed"] = collision
    with pytest.raises(ValueError, match="Prior.*overlap|prior.*overlap|overlap.*prior"):
        s.stream_contract(plan)


@pytest.mark.parametrize("mutation", ["registry", "generators", "priors", "pairing", "draws"])
def test_manifest_is_exactly_reconstructed_not_trusted(prepared, mutation):
    plan = copy.deepcopy(prepared.plan)
    contract = plan["random_stream_contract"]
    if mutation == "registry":
        contract["registry"]["planner/candidate_bank/0"] += 1
    elif mutation == "generators":
        key = next(iter(contract["generators"]))
        contract["generators"][key]["initial_state_sha256"] = "0" * 64
    elif mutation == "priors":
        contract["priors"][0]["include_train"] = False
    elif mutation == "pairing":
        contract["intentional_reuse"] = []
    else:
        contract["draws_for_manifest"] = 1
    digest = put(prepared.path, plan)
    with pytest.raises(ValueError, match="Random stream manifest changed"):
        s.validate(prepared.path, digest)


def test_domain_namespace_is_bound_even_if_manifest_is_regenerated(prepared):
    plan = copy.deepcopy(prepared.plan)
    plan.update(s.domain_bases("unapproved-replacement-evaluation"))
    plan["random_stream_contract"] = s.stream_contract(plan)
    digest = put(prepared.path, plan)
    with pytest.raises(ValueError, match="Prospective domain seed binding"):
        s.validate(prepared.path, digest)


def test_mocked_full_run_inherits_unchanged_files_and_never_trains(prepared, tmp_path, monkeypatch):
    out = tmp_path / "mock-execution"
    before = {str(path): s.sha(path) for path in prepared.root.rglob("*") if path.is_file()}
    loaded, predicted, controlled, referenced, collected = [], [], [], [], []

    def forbidden(*args, **kwargs):
        pytest.fail("Evaluation-only recovery attempted training or an unmocked native call")

    for module, name in ((s.legacy, "fit"), (s.legacy, "sequence_loss"),
                         (s.base, "sequence_loss"), (s.base, "learning_tensors"),
                         (s.base, "make_env"), (s.torch.optim, "Adam"), (s.torch.Tensor, "backward")):
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(s.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(s.torch, "use_deterministic_algorithms", lambda _: None)
    monkeypatch.setattr(s.torch, "load", lambda path, **kwargs: {"synthetic": Path(path).read_bytes()})

    class FakeModel:
        def __init__(self, name):
            self.name = name
            self.frozen = False

        def load_state_dict(self, state, *, strict):
            assert strict and state["synthetic"] == b"synthetic final " + self.name.encode()
            loaded.append(self.name)

        def requires_grad_(self, enabled):
            assert enabled is False
            self.frozen = True
            return self

        def eval(self):
            assert self.frozen
            return self

    monkeypatch.setattr(s.legacy, "model_for", lambda plan, kind, seed: FakeModel(f"{kind}-{seed}"))
    monkeypatch.setattr(s.base, "schedule", lambda plan, index: ("synthetic_schedule", index))

    def collect(seed, schedule, **kwargs):
        assert loaded == FIT_ORDER
        assert (out / "inherited-fits-ready.json").exists()
        assert (out / "evaluation-started.json").exists()
        collected.append((seed, schedule, kwargs))
        return {"synthetic_fixture": True}

    def save_records(path, records):
        path.with_suffix(".npz").write_bytes(b"synthetic records, not performance")
        put(path.with_suffix(".json"), {"synthetic_fixture": True, "count": len(records)})

    def prediction(plan, model, records, deadline):
        predicted.append(model.name)
        return {"synthetic_fixture": np.asarray([1])}

    def learned(plan, model, panel, reset, deadline, progress):
        controlled.append((model.name, panel, reset))
        return [{"synthetic_fixture": True}], {"synthetic_fixture": np.asarray([1])}

    def reference(plan, panel, arm, deadline, progress):
        referenced.append((panel, arm))
        return [{"synthetic_fixture": True}], {"synthetic_fixture": np.asarray([1])}

    monkeypatch.setattr(s, "collect_episode", collect)
    monkeypatch.setattr(s.base, "save_records", save_records)
    monkeypatch.setattr(s.base, "prediction_record", prediction)
    monkeypatch.setattr(s.base, "learned_control", learned)
    monkeypatch.setattr(s.base, "reference_control", reference)
    s.run(prepared.path, prepared.digest, out)
    assert loaded == predicted == FIT_ORDER
    assert len(collected) == prepared.plan["prediction_episodes"]
    assert controlled == [(name, panel, reset) for panel in ("full", "ordinary", "shift")
                          for name in FIT_ORDER for reset in ((False,) if panel == "full" else (False, True))]
    assert referenced == [(panel, arm) for panel in ("full", "ordinary", "shift")
                          for arm in ("known_state", "particle", "zero", "uniform")]
    for i, (seed, schedule, kwargs) in enumerate(collected):
        assert seed == prepared.plan["prediction_seed"] + i
        assert schedule == ("synthetic_schedule", 100000 + i)
        assert kwargs["noise_seed"] == prepared.plan["noise_seed"] + 100000 + i
        assert kwargs["action_seed"] == prepared.plan["exploration_seed"] + 100000 + i
    assert before == {str(path): s.sha(path) for path in prepared.root.rglob("*") if path.is_file()}
    for name, digest in prepared.source["members"].items():
        assert s.sha(out / name) == digest
    for name, digest in prepared.parent["training_source"]["members"].items():
        assert s.sha(out / name) == digest
    ready = s.read(out / "inherited-fits-ready.json")
    started = s.read(out / "evaluation-started.json")
    completed = s.read(out / "completed.json")
    assert ready["new_fits"] == completed["new_fits"] == 0
    assert ready["fit_order"] == FIT_ORDER and completed["fits"] == 6
    assert started["inherited_fits_ready_sha256"] == s.sha(out / "inherited-fits-ready.json")
    assert started["random_streams_sha256"] == s.sha(out / "random-streams.json")
    assert ready["elapsed_seconds"] <= started["elapsed_seconds"] <= completed["wall_seconds"]
    assert completed["inherited_fit_wall_seconds"] == 6.0
    assert completed["prior_invalid_attempt_wall_seconds"] == 9.0
    assert not (out / "failed.json").exists()
    assert completed["files"] == {str(path.relative_to(out)): s.sha(path)
                                 for path in out.rglob("*")
                                 if path.is_file() and path != out / "completed.json"}
    with pytest.raises(FileExistsError):
        s.run(prepared.path, prepared.digest, out)


def test_expired_cap_stops_before_copy_loading_or_evaluation(tmp_path, monkeypatch):
    fixture = fake_parent(tmp_path, monkeypatch, cap_seconds=0)
    result = s.prepare(tmp_path / "protocol", fit_source=fixture.source)

    def forbidden(*args, **kwargs):
        pytest.fail("Expired evaluation cap allowed model/environment work")

    monkeypatch.setattr(s.legacy, "model_for", forbidden)
    monkeypatch.setattr(s, "collect_episode", forbidden)
    out = tmp_path / "expired"
    with pytest.raises(TimeoutError):
        s.run(Path(result["path"]), result["sha256"], out)
    failure = s.read(out / "failed.json")
    assert failure["new_fits"] == 0
    assert failure["progress"]["phase"] == "inherit-training-and-fits"
    assert not (out / "fits").exists()
    assert not (out / "completed.json").exists()
