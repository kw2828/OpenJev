"""Causal/native engineering checks, never scored objective-study cases."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reacher_objective_study as study

from openjev.research import reacher_objective_protocol as protocol
from openjev.research import reacher_search_protocol as search_protocol
from openjev.research.reacher_reward_residual import GRUResidualRewardWorldModel
from openjev.research.robotics_reacher import collect_episode, native_replay


@pytest.fixture(scope="module")
def engineering():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    plan = protocol.settings(engineering=True)
    plan.update(rng_namespace="reacher-objective-engineering-runner-v1",
                engineering_rng_namespaces=[], control_episodes=2, prediction_episodes=2, hidden_size=4)
    plan["random_stream_contract"] = protocol.stream_contract(plan)
    with protocol.initialization_rng(plan, "pair0", "student"):
        model = GRUResidualRewardWorldModel(hidden_size=4, noise_std=.05, residual_reward=True).eval()
    yield plan, model
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def prediction_cases(engineering):
    plan, _ = engineering
    return [collect_episode(protocol.seed(plan, f"prediction/reset/{index}"),
                            protocol.schedule(plan, index, "ordinary", "prediction"),
                            noise_std=plan["noise_std"],
                            noise_seed=protocol.seed(plan, f"prediction/actuator_noise/{index}"),
                            action_seed=protocol.seed(plan, f"prediction/exploration/{index}"), policy="mixed")
            for index in range(plan["prediction_episodes"])]


def test_native_decoder_endpoints_and_compute(engineering, prediction_cases):
    plan, model = engineering
    values, cost = study.prediction_record(plan, model, prediction_cases, float("inf"))
    assert np.array_equal(values["one_step_angles"], values["h1_angles"])
    observed = np.stack([row["policy"]["packets"][:, 6] > .5 for row in prediction_cases])
    for horizon in (1, 3, 7):
        assert values[f"h{horizon}_angles"].shape == (2, 51 - horizon, 4)
        assert np.array_equal(values[f"h{horizon}_valid"], observed[:, :-horizon] & observed[:, horizon:])
    assert cost["student_assimilations"] == cost["prefix_advances"] == 100
    assert cost["imagined_advances"] == 2 * sum(min(7, 50 - root) for root in range(50))
    assert cost["teacher_or_auxiliary_calls"] == 0
    assert cost["wall_seconds"] > 0


def test_future_packets_and_privileged_fields_cannot_change_prior_prediction(engineering, prediction_cases):
    plan, model = engineering
    expected, _ = study.prediction_record(plan, model, prediction_cases, float("inf"))
    changed = copy.deepcopy(prediction_cases)
    for record in changed:
        record["policy"]["packets"][20:, :6] += 5
        for name, value in record["audit"].items():
            record["audit"][name] = np.full_like(value, 1234)
    actual, _ = study.prediction_record(plan, model, changed, float("inf"))
    for name in ("one_step_angles", "one_step_rewards", "h1_angles", "h3_angles", "h7_angles"):
        assert np.array_equal(actual[name][:, :20], expected[name][:, :20]), name


def test_prediction_does_not_mutate_weights_or_records(engineering, prediction_cases):
    plan, model = engineering
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    originals = copy.deepcopy(prediction_cases)
    study.prediction_record(plan, model, prediction_cases, float("inf"))
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in weights.items())
    for first, second in zip(prediction_cases, originals, strict=True):
        for category in ("policy", "audit"):
            assert all(np.array_equal(value, second[category][name]) for name, value in first[category].items())


def test_prediction_deadline_is_enforced(engineering, prediction_cases):
    with pytest.raises(TimeoutError):
        study.prediction_record(engineering[0], engineering[1], prediction_cases, -1)


def test_control_parity_reset_packets_and_native_replay(engineering, tmp_path):
    plan, model = engineering
    stems = []
    for step in range(50):
        stem = tmp_path / "inputs" / f"{step:03d}"
        search_protocol.save_inputs(stem, protocol.draw_control_inputs(plan, step), f"planner/control/{step}")
        stems.append(stem)
    intact = tmp_path / "intact"
    baseline = tmp_path / "baseline"
    reset = tmp_path / "reset"
    study.learned_control(plan, model, "ordinary", False, stems, intact, float("inf"), {})
    study.search_study.learned_control(plan, model, "ordinary", "cem256", stems, baseline, float("inf"), {})
    study.learned_control(plan, model, "ordinary", True, stems, reset, float("inf"), {})
    intact_records = study.base.load_records(intact / "episodes")
    baseline_records = study.base.load_records(baseline / "episodes")
    reset_records = study.base.load_records(reset / "episodes")
    for first, second in zip(intact_records, baseline_records, strict=True):
        for category in ("policy", "audit"):
            assert all(np.array_equal(value, second[category][name]) for name, value in first[category].items())
    with np.load(intact / "reset-events.npz", allow_pickle=False) as events:
        assert not events["mask"].any()
        assert not events["packets"].any()
    with np.load(reset / "reset-events.npz", allow_pickle=False) as events:
        assert events["mask"].dtype == np.bool_
        assert events["packets"].dtype == np.float32
        assert int(events["mask"].sum()) == 4
        assert not events["packets"][~events["mask"]].any()
        for index, record in enumerate(reset_records):
            expected = protocol.reset_steps(np.asarray(record["metadata"]["sensor_schedule"], dtype=bool))
            assert tuple(np.flatnonzero(events["mask"][index])) == expected
            for step in expected:
                assert np.array_equal(events["packets"][index, step], record["policy"]["packets"][step])
                assert events["packets"][index, step, 6] == 1
    for record in intact_records + reset_records:
        assert native_replay(record)["max_abs_error"] == 0


def test_torch_outputs_require_exclusive_paths(tmp_path):
    target = tmp_path / "weights.pt"
    study.save_torch(target, {"a": torch.ones(2)})
    with pytest.raises(FileExistsError):
        study.save_torch(target, {"a": torch.zeros(2)})
    assert torch.equal(torch.load(target, weights_only=True)["a"], torch.ones(2))


def test_collection_failure_preserves_completed_episodes(engineering, prediction_cases, tmp_path, monkeypatch):
    count = 0

    def fail_second(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("engineering collection failure")
        return prediction_cases[0]

    monkeypatch.setattr(study, "collect_episode", fail_second)
    with pytest.raises(RuntimeError, match="engineering collection failure"):
        study.collect_predictions(engineering[0], tmp_path, float("inf"), {"phase": "prediction-collection"})
    saved = study.base.load_records(tmp_path / "partial-prediction")
    assert len(saved) == 1
    assert native_replay(saved[0])["max_abs_error"] == 0
    state = study.read(tmp_path / "partial-prediction-state.json")
    assert state["completed_episodes"] == 1
    assert state["progress"]["episode"] == 1
    assert not (tmp_path / "prediction.npz").exists()


def test_validation_failure_is_a_preserved_attempt(tmp_path, monkeypatch):
    def reject(*args):
        raise ValueError("engineering source mismatch")

    monkeypatch.setattr(study, "validate", reject)
    out = tmp_path / "attempt"
    with pytest.raises(ValueError, match="engineering source mismatch"):
        study.run(tmp_path / "plan.json", "expected-external-hash", out)
    receipt = study.read(out / "failed.json")
    assert receipt["status"] == "failed"
    assert receipt["plan_sha256"] == "expected-external-hash"
    assert receipt["progress"]["phase"] == "validate-or-setup"
    assert list(out.iterdir()) == [out / "failed.json"]
    with pytest.raises(FileExistsError):
        study.run(tmp_path / "plan.json", "expected-external-hash", out)


def test_unvalidated_scored_namespace_cannot_enter_core(tmp_path):
    plan = protocol.settings()
    plan.update(engineering=False, cap_seconds=100)
    with pytest.raises(ValueError, match="frozen validation"):
        study.run_validated(plan, "not-frozen", tmp_path / "attempt")
    assert not (tmp_path / "attempt").exists()


def test_engineering_flag_cannot_authorize_scored_streams(tmp_path):
    plan = protocol.settings()
    plan.update(engineering=True, cap_seconds=100)
    with pytest.raises(ValueError, match="Declared engineering namespace"):
        study.run_validated(plan, "not-frozen", tmp_path / "attempt")
    assert not (tmp_path / "attempt").exists()


def test_member_hashing_obeys_deadline_and_rejects_symlinks(tmp_path):
    (tmp_path / "record").write_text("engineering fixture")
    with pytest.raises(TimeoutError):
        study.file_members(tmp_path, -1)
    (tmp_path / "link").symlink_to(tmp_path / "record")
    with pytest.raises(ValueError, match="symlink"):
        study.file_members(tmp_path)


def test_prior_streams_bind_literal_training_test_generators_without_drawing():
    state = torch.get_rng_state().clone()
    numpy_registries, registries, descriptors = study.prior_streams()
    source = "tests/test_reacher_objective_training.py"
    descriptor = next(row for row in descriptors if row.get("artifact_path") == source)
    expected = {
        "engineering/training-tests/public_data": 410,
        "engineering/training-tests/student/initial": 1121,
        "engineering/training-tests/predictor/initial": 1123,
        "engineering/training-tests/predictor/alternate": 1151,
        "engineering/training-tests/minibatch": 1129,
        **{f"engineering/training-tests/student/calibration/{i}": 1201 + i for i in range(3)},
        **{f"engineering/training-tests/predictor/calibration/{i}": 1301 + i for i in range(3)},
    }
    assert descriptor == {"artifact_path": source, "artifact_sha256": study.sha(study.ROOT / source),
                          "registry_kind": "additional-engineering-torch",
                          "roles": sorted(expected), "seeds": dict(sorted(expected.items()))}
    assert all(registries[0][role] == seed for role, seed in expected.items())
    source = "tests/test_audit_reacher_objective_study.py"
    numpy_fixture = {"engineering/auditor-tests/public_data": 716}
    torch_fixture = {"engineering/auditor-tests/corrupted-minibatch-state": 72}
    assert numpy_registries[-1] == numpy_fixture
    assert registries[0]["engineering/auditor-tests/corrupted-minibatch-state"] == 72
    for descriptor, kind, seeds in zip(descriptors[-2:], ("numpy", "torch"),
                                       (numpy_fixture, torch_fixture), strict=True):
        assert descriptor == {"artifact_path": source, "artifact_sha256": study.sha(study.ROOT / source),
                              "registry_kind": f"additional-engineering-{kind}",
                              "roles": sorted(seeds), "seeds": seeds}
    assert torch.equal(state, torch.get_rng_state())


def _stubbed_attempt(tmp_path, monkeypatch, *, corrupt_ninth=False):
    """Exercise actual artifact authentication and orchestration, without fitting.

    The trainer and environment work are stubs. Files, hashes, all-nine restore
    ordering, paired order checks, and terminal failure handling remain real.
    """
    plan = protocol.settings(engineering=True)
    plan.update(rng_namespace="reacher-objective-engineering-runner-v1", engineering_rng_namespaces=[],
                engineering=True, control_episodes=1, prediction_episodes=2, hidden_size=4,
                epochs=1, train_episodes=128, cap_seconds=60, sources={}, runtime={},
                training_source={"members": {}, "execution_path": str(tmp_path)},
                search_source={"prior_costs": {"cumulative_attempt_wall_seconds": 1.}})
    plan["random_stream_contract"] = protocol.stream_contract(plan)
    counts = {"fits": 0, "restored": 0, "collection": 0, "prediction": 0, "control": 0}

    def prepare(_plan, _data, out, _deadline):
        (out / "initializations").mkdir()
        (out / "orders").mkdir()
        initials, orders = {}, {}
        for pair in protocol.PAIRS:
            initial = {"hashes": {"student_state": pair},
                       "seeds": {"student": protocol.seed(plan, f"fit/student/{pair}"),
                                 "predictor": protocol.seed(plan, f"fit/predictor/{pair}")}}
            initials[pair] = initial
            study.save_torch(out / "initializations" / f"{pair}.pt", initial)
            order = protocol.minibatch_orders(plan, pair)
            orders[pair] = order
            path = out / "orders" / f"{pair}.npy"
            study.save_npy(path, order.numpy())
            study.write(path.with_suffix(".json"), {"file_sha256": study.sha(path)})
        calibration = {"multipliers": {arm: {"value": .5} for arm in ("raw", "latent")}}
        study.write(out / "calibration.json", calibration)
        study.write(out / "calibration-completed.json", {
            "files": {"calibration.json": study.sha(out / "calibration.json")},
            "initialization_files": {f"initializations/{pair}.pt": study.sha(out / "initializations" / f"{pair}.pt")
                                     for pair in protocol.PAIRS}})
        return None, initials, orders, calibration, {"preparation_wall_seconds": 0.}

    def fit(_plan, row, _data, initial, order, multiplier, folder, _deadline, _progress):
        counts["fits"] += 1
        folder.mkdir(parents=True)
        updates = plan["train_episodes"] // plan["batch_size"]
        weights = {"synthetic": torch.tensor([counts["fits"]], dtype=torch.float32)}
        checkpoint = {"arm": row["arm"], "minibatch_seed": protocol.seed(plan, row["minibatch_role"]),
                      "initialization": initial, "successful_updates": updates, "optimizer_steps": updates,
                      "ema_updates": updates if row["arm"] == "latent" else 0,
                      "loss_multiplier": multiplier, "student": weights}
        study.save_torch(folder / "checkpoint.pt", checkpoint)
        study.save_torch(folder / "weights.pt", weights)
        receipt = {"name": row["name"], "arm": row["arm"], "pair": row["pair"],
                   "initial_hashes": initial["hashes"],
                   "order_tensor_sha256": study.training.canonical_tensor_hash({"orders": order}),
                   "updates": updates, "loss_multiplier": multiplier, "wall_seconds": 0.,
                   "final_weights": {"canonical_tensor_sha256": study.training.canonical_tensor_hash(weights)},
                   "files": study.file_members(folder)}
        study.write(folder / "completed.json", receipt)
        if corrupt_ninth and counts["fits"] == 9:
            with (folder / "checkpoint.pt").open("ab") as handle:
                handle.write(b"post-receipt corruption")
        return receipt

    def restore(checkpoint, **_kwargs):
        counts["restored"] += 1
        student = SimpleNamespace(state_dict=lambda: checkpoint["student"])
        student.eval = lambda: student
        return SimpleNamespace(student=student, successful_updates=checkpoint["successful_updates"])

    def collect(*_args):
        counts["collection"] += 1
        return [{} for _ in range(plan["prediction_episodes"])], 0.

    def predict(*_args):
        counts["prediction"] += 1
        return {"synthetic": np.zeros(1)}, {"wall_seconds": 0.}

    def control(*_args):
        counts["control"] += 1
        return {"row_wall_seconds": 0., "setup_seconds": 0.,
                "decision_seconds": [], "native_step_seconds": []}

    monkeypatch.setattr(study.base, "load_records", lambda _path: [{}] * plan["train_episodes"])
    monkeypatch.setattr(study.base, "learning_tensors", lambda _records: ())
    monkeypatch.setattr(study, "prepare_training", prepare)
    monkeypatch.setattr(study, "fit_one", fit)
    monkeypatch.setattr(study.training, "restore_checkpoint", restore)
    monkeypatch.setattr(study, "collect_predictions", collect)
    monkeypatch.setattr(study, "prediction_record", predict)
    monkeypatch.setattr(study, "learned_control", control)
    monkeypatch.setattr(study.search_study, "reference_control", control)
    monkeypatch.setattr(study.torch, "use_deterministic_algorithms", lambda _value: None)
    return plan, counts


def test_ninth_checkpoint_corruption_prevents_fresh_collection(tmp_path, monkeypatch):
    plan, counts = _stubbed_attempt(tmp_path, monkeypatch, corrupt_ninth=True)
    out = tmp_path / "attempt"
    ninth = plan["fit_order"][-1]
    with pytest.raises(ValueError, match=f"Artifact identity mismatch: .*{ninth}/checkpoint.pt"):
        study.run_validated(plan, "synthetic-plan", out)
    assert counts == {"fits": 9, "restored": 8, "collection": 0, "prediction": 0, "control": 0}
    assert (out / "all-fits-completed.json").exists()
    assert not (out / "restored-models.json").exists()
    assert not (out / "evaluation-started.json").exists()
    assert not (out / "completed.json").exists()
    failure = study.read(out / "failed.json")
    assert failure["progress"] == {"phase": "restore-all-students"}
    assert f"fits/{ninth}/checkpoint.pt" in failure["files"]


def test_postcompletion_cap_failure_preserves_failed_and_overcap_receipts(tmp_path, monkeypatch):
    plan, counts = _stubbed_attempt(tmp_path, monkeypatch)
    out = tmp_path / "attempt"
    check_cap = study.check_cap

    def expire_after_completion(deadline):
        check_cap(-1 if (out / "completed.json").exists() else deadline)

    monkeypatch.setattr(study, "check_cap", expire_after_completion)
    with pytest.raises(TimeoutError):
        study.run_validated(plan, "synthetic-plan", out)
    assert counts == {"fits": 9, "restored": 9, "collection": 1, "prediction": 9, "control": 57}
    assert not (out / "completed.json").exists()
    overcap = study.read(out / "over-cap-completion.json")
    assert overcap["fits"] == 9 and overcap["control_rows"] == 57
    failure = study.read(out / "failed.json")
    assert failure["status"] == "failed" and "TimeoutError" in failure["error"]
    assert failure["progress"] == {"phase": "final-validation"}
    assert failure["files"]["over-cap-completion.json"] == study.sha(out / "over-cap-completion.json")
