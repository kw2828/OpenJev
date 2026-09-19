"""Engineering checks of actual-class training/restoration and run boundaries."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reacher_cache_study as study

from openjev.research import reacher_cache_protocol as protocol
from openjev.research import reacher_cache_training as training
from openjev.research.robotics_reacher import collect_episode, native_replay


@pytest.fixture(autouse=True)
def isolate():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


def engineering_plan():
    p = protocol.settings(engineering=True)
    namespace = "reacher-cache-engineering-runner-v1"
    p.update(rng_namespace=namespace,
             engineering_rng_namespaces=[n for n in protocol.ENGINEERING_NAMESPACES if n != namespace],
             train_episodes=4, epochs=1, batch_size=2, hidden_size=4, mlp_width=7,
             control_episodes=1, prediction_episodes=2, bootstrap_samples=8,
             sources={"synthetic": "a" * 64}, runtime={"engineering": True},
             engineering=True, cap_seconds=300, audit_cap_seconds=300)
    p["random_stream_contract"] = protocol.stream_contract(p)
    return p


def synthetic_data(plan):
    gen = torch.Generator().manual_seed(410)
    n, t = plan["train_episodes"], plan["steps"]
    angles = torch.randn(n, t + 1, 2, generator=gen) * .05
    packets = torch.zeros(n, t + 1, 8)
    packets[..., :2], packets[..., 2:4] = angles.cos(), angles.sin()
    packets[..., 4:6], packets[..., 6] = .1, 1.
    packets[:, 8:14, :4], packets[:, 8:14, 6] = 0., 0.
    packets[:, 8:14, 7] = torch.arange(1, 7) * .02
    commands = torch.randn(n, t, 2, generator=gen).clamp(-1, 1) * .1
    return {"packets": packets, "commands": commands, "rewards": -commands.square().sum(-1) - .1}


@pytest.mark.parametrize("kind", protocol.ARMS)
def test_fit_restores_actual_architecture_and_exact_optimizer_boundary(kind, tmp_path):
    plan = engineering_plan()
    cfg = study.settings_from_plan(plan)
    initial = training.make_initialization(410, 410, cfg)
    orders = training.make_orders(410, cfg)
    row = next(row for row in protocol.fit_manifest(plan) if row["arm"] == kind)
    data = synthetic_data(plan)
    done = study.fit_one(plan, row, data, initial, orders, tmp_path / kind, float("inf"), {})
    assert done["updates"] == done["optimizer_steps"] == 2
    checkpoint = torch.load(tmp_path / kind / "checkpoint.pt", weights_only=True)
    restored = training.restore_checkpoint(checkpoint, data, expected_kind=kind, expected_settings=cfg,
        expected_initialization_sha256=initial["integrity_sha256"],
        expected_orders_sha256=orders["integrity_sha256"],
        expected_data_sha256=training.canonical_tensor_hash(data),
        expected_source_sha256=plan["sources"], expected_runtime=plan["runtime"])
    assert type(restored.student) is training.REGISTRY[kind]
    assert restored.successful_updates == 2
    assert restored.cursor == {"epoch": 1, "batch": 0}
    assert training.canonical_tensor_hash(restored.student.state_dict()) == done["student_tensor_sha256"]
    assert done["model_configuration"]["model_class"] == protocol.MODEL_CLASSES[kind]


def test_no_evaluation_when_training_fails_and_partial_attempt_retained(tmp_path, monkeypatch):
    plan = engineering_plan()
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.npz").write_bytes(b"synthetic")
    plan["training_source"] = {"execution_path": str(source),
                              "members": {"train.npz": study.sha(source / "train.npz")}}
    monkeypatch.setattr(study.base, "load_records", lambda _: [None] * 4)
    monkeypatch.setattr(study, "public_training_data", lambda _: synthetic_data(plan))
    monkeypatch.setattr(study, "prepare_training", lambda *args: ({p: {} for p in protocol.PAIRS},
                                                               {p: {} for p in protocol.PAIRS}))

    def fail(*args, **kwargs):
        raise RuntimeError("engineering training failure")

    monkeypatch.setattr(study, "fit_one", fail)
    monkeypatch.setattr(study, "collect_predictions", lambda *args: pytest.fail("Evaluation leaked"))
    target = tmp_path / "execution"
    with pytest.raises(RuntimeError, match="training failure"):
        study.run_validated(plan, "engineering", target)
    failure = study.read(target / "failed.json")
    assert failure["progress"]["phase"] == "fit"
    assert not (target / "evaluation-started.json").exists()
    with pytest.raises(FileExistsError):
        study.run_validated(plan, "engineering", target)


def test_fit_deadline_preserves_partial_checkpoint_and_failure(tmp_path):
    plan = engineering_plan()
    cfg = study.settings_from_plan(plan)
    initial = training.make_initialization(410, 410, cfg)
    order = training.make_orders(410, cfg)
    row = protocol.fit_manifest(plan)[0]
    folder = tmp_path / "fit"
    with pytest.raises(TimeoutError):
        study.fit_one(plan, row, synthetic_data(plan), initial, order, folder, -1, {})
    assert study.read(folder / "failed.json")["successful_updates"] == 0
    checkpoint = torch.load(folder / "partial-checkpoint.pt", weights_only=True)
    assert checkpoint["failed"] is True
    assert not (folder / "completed.json").exists()


def test_missing_fit_stops_before_restore_or_evaluation(tmp_path):
    plan = engineering_plan()
    study.write(tmp_path / "all-fits-completed.json", {"plan_sha256": "test", "fit_order": plan["fit_order"],
                                                      "files": {}})
    with pytest.raises(ValueError, match="Every fit"):
        study.restore_students(plan, tmp_path, synthetic_data(plan), float("inf"), "test")
    assert not (tmp_path / "restored-models.json").exists()


@pytest.fixture
def fit_tree(tmp_path):
    plan = engineering_plan()
    data = synthetic_data(plan)
    initial, orders = study.prepare_training(plan, tmp_path, float("inf"))
    for row in protocol.fit_manifest(plan):
        study.fit_one(plan, row, data, initial[row["pair"]], orders[row["pair"]],
                      tmp_path / "fits" / row["name"], float("inf"), {})
    study.write(tmp_path / "all-fits-completed.json", {"plan_sha256": "engineering", "fit_order": plan["fit_order"],
        "files": {f"fits/{name}/completed.json": study.sha(tmp_path / "fits" / name / "completed.json")
                  for name in plan["fit_order"]},
        "training_prepared_sha256": study.sha(tmp_path / "training-prepared.json")})
    return plan, data, tmp_path


def test_all_fifteen_restore_before_evaluation(fit_tree):
    plan, data, folder = fit_tree
    models = study.restore_students(plan, folder, data, float("inf"), "engineering")
    assert list(models) == plan["fit_order"]
    assert all(type(models[row["name"]]) is training.REGISTRY[row["arm"]]
               for row in protocol.fit_manifest(plan))
    assert len(study.read(folder / "restored-models.json")["models"]) == 15


@pytest.mark.parametrize("change", ["prepared_member", "fit_member", "initial_seed", "order_seed", "completion"])
def test_malformed_bound_payload_stops_before_any_restoration_result(fit_tree, change):
    plan, data, folder = fit_tree
    prepared = study.read(folder / "training-prepared.json")
    boundary = study.read(folder / "all-fits-completed.json")
    row = protocol.fit_manifest(plan)[0]
    done_path = folder / "fits" / row["name"] / "completed.json"
    done = study.read(done_path)

    def replace_json(path, value):
        path.write_text(json.dumps(value, allow_nan=False) + "\n")

    if change == "prepared_member":
        prepared["files"].pop("initializations/pair0.pt")
    elif change == "fit_member":
        done["files"].pop("weights.pt")
    elif change in ("initial_seed", "order_seed"):
        cfg = study.settings_from_plan(plan)
        category = "initializations" if change == "initial_seed" else "orders"
        path = folder / category / "pair0.pt"
        if change == "initial_seed":
            payload = training.make_initialization(410, 410, cfg,
                role_names={"gru": "fit/gru/pair0", "mlp": "fit/mlp/pair0"})
        else:
            payload = training.make_orders(410, cfg, role_name="fit/minibatch/pair0")
        torch.save(payload, path)
        replace_json(path.with_suffix(".json"), {"pair": "pair0", "file_sha256": study.sha(path),
                                                "integrity_sha256": payload["integrity_sha256"]})
        for suffix in ("pt", "json"):
            name = f"{category}/pair0.{suffix}"
            prepared["files"][name] = study.sha(folder / name)
    else:
        done["checkpoint_integrity_sha256"] = "f" * 64
    replace_json(done_path, done)
    boundary["files"][f"fits/{row['name']}/completed.json"] = study.sha(done_path)
    replace_json(folder / "training-prepared.json", prepared)
    boundary["training_prepared_sha256"] = study.sha(folder / "training-prepared.json")
    replace_json(folder / "all-fits-completed.json", boundary)
    with pytest.raises(ValueError, match="membership|planned stream|completion metadata"):
        study.restore_students(plan, folder, data, float("inf"), "engineering")
    assert not (folder / "restored-models.json").exists()


def test_scored_namespace_rejected_by_engineering_core_before_artifacts(tmp_path):
    plan = protocol.settings()
    plan.update(engineering=True, cap_seconds=100)
    with pytest.raises(ValueError, match="namespace"):
        study.run_validated(plan, "not-a-frozen-protocol", tmp_path / "execution")
    assert not (tmp_path / "execution").exists()


def test_unvalidated_scored_execution_rejected(tmp_path):
    plan = protocol.settings()
    plan.update(engineering=False, cap_seconds=100)
    with pytest.raises(ValueError, match="requires validation"):
        study.run_validated(plan, "not-a-frozen-protocol", tmp_path / "execution")
    assert not (tmp_path / "execution").exists()


def test_failed_initial_validation_is_preserved(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("engineering invalid identity")

    monkeypatch.setattr(study, "validate", fail)
    out = tmp_path / "attempt"
    with pytest.raises(ValueError, match="identity"):
        study.run(tmp_path / "plan.json", "expected", out)
    failure = study.read(out / "failed.json")
    assert failure["plan_sha256"] == "expected"
    assert failure["progress"]["phase"] == "validate-or-setup"


def test_prediction_collection_failure_keeps_completed_native_episode(tmp_path, monkeypatch):
    plan = engineering_plan()
    record = collect_episode(410, np.ones(51, dtype=bool), noise_seed=410, action_seed=410, policy="mixed")
    count = 0

    def partial(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("engineering second episode failed")
        return copy.deepcopy(record)

    monkeypatch.setattr(study, "collect_episode", partial)
    with pytest.raises(RuntimeError, match="second episode"):
        study.collect_predictions(plan, tmp_path, float("inf"), {})
    records = study.base.load_records(tmp_path / "partial-prediction")
    assert len(records) == 1
    assert native_replay(records[0])["max_abs_error"] == 0
    assert study.read(tmp_path / "partial-prediction-state.json")["completed_episodes"] == 1


def test_kinematic_native_control_reference_and_snapshot_serialization(tmp_path):
    plan = engineering_plan()
    stems = []
    for step in range(50):
        stem = tmp_path / "inputs" / f"{step:03d}"
        study.serialization.save_inputs(stem, protocol.draw_control_inputs(plan, step), f"planner/control/{step}")
        stems.append(stem)
    out = tmp_path / "control"
    timing = study.kinematic_control(plan, "shift", stems, out, float("inf"), {})
    records = study.base.load_records(out / "episodes")
    assert len(records) == 1
    assert native_replay(records[0])["max_abs_error"] == 0
    assert len(timing["decision_seconds"]) == 50
    assert len(study.read(out / "observer-final.json")) == 1
    with np.load(out / "planning.npz", allow_pickle=False) as saved:
        assert saved["public_estimates"].shape == (1, 50, 8)
        assert saved["candidate_scores"].shape == (1, 50, 64)
    assert timing["information"].endswith("no true state")
