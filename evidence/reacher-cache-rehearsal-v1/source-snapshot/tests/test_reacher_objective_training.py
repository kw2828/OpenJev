"""Synthetic CPU-only tests; no real corpus, fitted weights or scored streams."""

import copy
import dataclasses
import io

import numpy as np
import pytest
import torch

from openjev.research import reacher_objective_training as training
from openjev.research.reacher_reward_residual import GRUResidualRewardWorldModel
from openjev.research.reacher_world_models import sequence_loss


@pytest.fixture(autouse=True)
def cpu_threads():
    threads = torch.get_num_threads()
    state = torch.get_rng_state()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.set_rng_state(state)


def settings(**kw):
    return training.TrainingSettings(hidden_size=4, steps=8, train_episodes=128, epochs=2, **kw)


def data(n=3, steps=8):
    generator = torch.Generator().manual_seed(410)
    angles = torch.randn(n, steps + 1, 2, generator=generator) * 0.1
    packets = torch.zeros(n, steps + 1, 8)
    packets[..., :2], packets[..., 2:4] = angles.cos(), angles.sin()
    packets[..., 4:6] = torch.rand(n, 1, 2, generator=generator) * 0.1
    packets[..., 6] = 1
    packets[:, 2:5, :4] = 0
    packets[:, 2:5, 6] = 0
    packets[:, 2:5, 7] = torch.tensor([0.02, 0.04, 0.06])
    commands = torch.randn(n, steps, 2, generator=generator).clamp(-1, 1) * 0.1
    rewards = -commands.square().sum(-1) - 0.1
    return packets, commands, rewards


def trainer(arm="anchor", multiplier=0.0, cfg=None):
    cfg = cfg or settings()
    initial = training.make_initialization(1121, 1123, cfg)
    return training.make_trainer(
        arm, initial, 1129, multiplier, cfg, source_sha256={"synthetic": "a" * 64}, runtime={"test": True}
    )


def assert_states(left, right):
    assert set(left) == set(right)
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_paired_initialization_is_exact_independent_and_does_not_change_rng():
    state = torch.get_rng_state().clone()
    cfg = settings()
    initial = training.make_initialization(1121, 1123, cfg)
    assert torch.equal(state, torch.get_rng_state())
    same = training.make_initialization(1121, 1123, cfg)
    changed = training.make_initialization(1121, 1151, cfg)
    assert initial["hashes"] == same["hashes"]
    assert initial["hashes"]["student_state"] == changed["hashes"]["student_state"]
    assert initial["hashes"]["predictor_state"] != changed["hashes"]["predictor_state"]
    arms = [training.make_trainer(arm, initial, 1129, 0.0, cfg) for arm in training.ARMS]
    for item in arms:
        assert_states(item.student.state_dict(), initial["student_state"])
    assert_states(arms[1].auxiliary.predictor.state_dict(), arms[2].auxiliary.predictor.state_dict())
    assert_states(arms[2].auxiliary.teacher.state_dict(), initial["student_state"])
    assert torch.equal(state, torch.get_rng_state())
    with torch.no_grad():
        next(arms[0].student.parameters()).add_(1)
    assert_states(arms[1].student.state_dict(), initial["student_state"])


def test_canonical_hash_binds_dtype_shape_name_bytes_and_order_is_irrelevant():
    values = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor(3, dtype=torch.int64)}
    canonical = training.canonical_tensor_hash(values)
    assert canonical == training.canonical_tensor_hash(dict(reversed(list(values.items()))))
    for changed in (
        {"a": values["a"].double(), "b": values["b"]},
        {"a": values["a"].reshape(1, 2), "b": values["b"]},
        {"x": values["a"], "b": values["b"]},
        {"a": values["a"] + 1, "b": values["b"]},
    ):
        assert training.canonical_tensor_hash(changed) != canonical
    with pytest.raises(ValueError):
        training.canonical_tensor_hash({"x": torch.tensor(float("nan"))})


@pytest.mark.parametrize("arm", training.ARMS)
def test_zero_weight_matches_existing_anchor_loss_adam_and_weights_bitwise(arm):
    item = trainer(arm)
    reference = copy.deepcopy(item.student)
    optimizer = torch.optim.Adam(reference.parameters(), lr=0.001)
    batch = data()
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, values = sequence_loss(reference, *batch, **item.settings.anchor_kwargs())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
        metrics = item.train_batch(*batch, epoch=0, batch=update, indices=torch.arange(3))
        assert metrics["anchor"] == values
        assert metrics["total_loss"] == float(loss.detach())
        assert metrics["combined_gradient_norm"] == float(norm)
        assert_states(item.student.state_dict(), reference.state_dict())
        if arm != "anchor":
            assert_states(item.auxiliary.predictor.state_dict(), item.initialization["predictor_state"])
    assert item.ema_updates == (2 if arm == "latent" else 0)


def test_optimizer_owns_each_student_and_predictor_once_never_teacher():
    item = trainer("latent", 0.5)
    actual = [id(p) for group in item.optimizer.param_groups for p in group["params"]]
    expected = [id(p) for p in item.student.parameters()] + [
        id(p) for p in item.auxiliary.predictor.parameters()
    ]
    assert actual == expected and len(actual) == len(set(actual))
    assert not set(actual).intersection(id(p) for p in item.auxiliary.teacher.parameters())
    assert item.optimizer_group_names == [
        ["student." + name for name, _ in item.student.named_parameters()]
        + ["predictor." + name for name, _ in item.auxiliary.predictor.named_parameters()]
    ]
    assert not item.auxiliary.teacher.training
    assert not any(p.requires_grad for p in item.auxiliary.teacher.parameters())


def test_successful_adam_precedes_exact_one_ema_and_witness_matches_arithmetic(monkeypatch):
    item = trainer("latent", 0.5)
    seen = []
    step, update = item.optimizer.step, item.auxiliary.update_teacher

    def optimizer_step(*args, **kw):
        seen.append("optimizer")
        return step(*args, **kw)

    def teacher_update(student):
        seen.append("ema")
        assert item.optimizer_steps == 1
        return update(student)

    monkeypatch.setattr(item.optimizer, "step", optimizer_step)
    monkeypatch.setattr(item.auxiliary, "update_teacher", teacher_update)
    metrics = item.train_batch(*data(), epoch=0, batch=0, capture_ema_witness=True)
    assert seen == ["optimizer", "ema"] and item.successful_updates == item.ema_updates == 1
    witness = item.ema_witness
    for name, before in witness["teacher_before"]["parameters"].items():
        expected = before * 0.99
        expected.add_(witness["student_after_optimizer"]["parameters"][name], alpha=0.01)
        torch.testing.assert_close(expected, witness["teacher_after"]["parameters"][name], rtol=0, atol=2e-8)
    assert witness["sha256"] == training.canonical_state_hash(
        {k: v for k, v in witness.items() if k != "sha256"}
    )
    assert metrics["auxiliary_work"]["collapse_diagnostics_included_in_auxiliary_wall"]
    assert all(p.grad is None for p in item.auxiliary.teacher.parameters())


@pytest.mark.parametrize("exception", [ValueError("failed optimizer"), KeyboardInterrupt()])
def test_failed_step_never_advances_teacher_or_swallows_baseexception(monkeypatch, exception):
    item = trainer("latent", 0.5)
    before = training.canonical_tensor_hash(item.auxiliary.teacher.state_dict())

    def fail():
        raise exception

    monkeypatch.setattr(item.optimizer, "step", fail)
    with pytest.raises(type(exception)):
        item.train_batch(*data(), epoch=0, batch=0)
    assert item.failed and item.ema_updates == item.optimizer_steps == item.successful_updates == 0
    assert training.canonical_tensor_hash(item.auxiliary.teacher.state_dict()) == before
    assert all(p.grad is None for p in item.named_trainable.values())
    with pytest.raises(ValueError, match="terminal"):
        item.train_batch(*data(), epoch=0, batch=0)
    payload = item.export_checkpoint()
    with pytest.raises(ValueError, match="Failed partial"):
        training.restore_checkpoint(
            payload,
            expected_settings=item.settings,
            expected_source_sha256=item.source_sha256,
            expected_runtime=item.runtime,
        )


def test_masks_must_be_complete_without_silent_anchor_semantic_change():
    tensors = data()
    clean, masked = trainer(), trainer()
    left = clean.train_batch(*tensors, epoch=0, batch=0)
    right = masked.train_batch(
        *tensors, epoch=0, batch=0, transition_valid=torch.ones(3, 8, dtype=torch.bool)
    )
    assert left["total_loss"] == right["total_loss"]
    mask = torch.ones(3, 8, dtype=torch.bool)
    mask[0, -1] = False
    with pytest.raises(ValueError, match="complete episodes"):
        trainer().train_batch(*tensors, epoch=0, batch=0, transition_valid=mask)
    with pytest.raises(ValueError, match="Finite"):
        trainer().train_batch(tensors[0].double(), *tensors[1:], epoch=0, batch=0)


@pytest.mark.parametrize("arm", training.ARMS)
def test_safe_checkpoint_restore_includes_adam_rng_teacher_and_same_next_update(arm):
    item = trainer(arm, 0 if arm == "anchor" else 0.5)
    first = item.next_permutation()
    item.train_batch(*data(), epoch=0, batch=0, indices=first[:3])
    exported = item.export_checkpoint()
    output = io.BytesIO()
    torch.save(exported, output)
    output.seek(0)
    loaded = torch.load(output, weights_only=True)
    rng = torch.get_rng_state().clone()
    restored = training.restore_checkpoint(
        loaded,
        expected_settings=item.settings,
        expected_source_sha256=item.source_sha256,
        expected_runtime=item.runtime,
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert_states(restored.student.state_dict(), item.student.state_dict())
    assert training.canonical_state_hash(restored.optimizer.state_dict()) == training.canonical_state_hash(
        item.optimizer.state_dict()
    )
    assert torch.equal(item.next_permutation(), restored.next_permutation())
    if arm == "latent":
        assert_states(restored.auxiliary.teacher.state_dict(), item.auxiliary.teacher.state_dict())
        assert any(
            not torch.equal(value, restored.student.state_dict()[name])
            for name, value in restored.auxiliary.teacher.state_dict().items()
        )
    for current in (item, restored):
        current.train_batch(*data(), epoch=0, batch=1, indices=torch.arange(3))
    assert_states(restored.student.state_dict(), item.student.state_dict())
    if arm != "anchor":
        assert_states(restored.auxiliary.state_dict(), item.auxiliary.state_dict())


@pytest.mark.parametrize(
    "change",
    [
        "noise",
        "residual",
        "teacher",
        "optimizer",
        "groups",
        "state",
        "counter",
        "extra",
        "nan",
        "adam_missing",
        "adam_step",
        "adam_moment",
        "adam_negative",
        "log_encoding",
    ],
)
def test_configuration_state_and_ownership_drift_rejected_even_if_rehashed(change):
    item = trainer("latent", 0.5)
    item.train_batch(*data(), epoch=0, batch=0)
    payload = item.export_checkpoint()
    if change == "noise":
        payload["settings"]["noise_std"] = 0.1
    elif change == "residual":
        payload["auxiliary_configuration"]["model"]["residual_reward"] = False
    elif change == "teacher":
        payload["auxiliary_state"] = {
            k: v for k, v in payload["auxiliary_state"].items() if not k.startswith("teacher.")
        }
    elif change == "optimizer":
        payload["optimizer_state"]["param_groups"][0]["lr"] = 0.1
    elif change == "groups":
        payload["optimizer_group_names"][0].pop()
    elif change == "state":
        payload["student_state"]["extra"] = torch.zeros(1)
    elif change == "counter":
        payload["ema_updates"] += 1
    elif change == "extra":
        payload["extra"] = 1
    elif change == "adam_missing":
        payload["optimizer_state"]["state"].pop(0)
    elif change == "adam_step":
        payload["optimizer_state"]["state"][0]["step"].add_(1)
    elif change == "adam_moment":
        payload["optimizer_state"]["state"][0]["exp_avg"] = torch.zeros(1)
    elif change == "adam_negative":
        payload["optimizer_state"]["state"][0]["exp_avg_sq"].fill_(-1)
    elif change == "log_encoding":
        payload["log_chain_sha256"] = "x" * 64
    else:
        key = next(iter(payload["student_state"]))
        payload["student_state"][key].view(-1)[0] = float("nan")
        with pytest.raises(ValueError):
            training.canonical_state_hash(payload)
        return
    payload["integrity_sha256"] = training.canonical_state_hash(
        {k: v for k, v in payload.items() if k != "integrity_sha256"}
    )
    with pytest.raises(ValueError):
        training.restore_checkpoint(
            payload,
            expected_settings=item.settings,
            expected_source_sha256=item.source_sha256,
            expected_runtime=item.runtime,
        )


def test_live_student_and_teacher_python_scalars_checked():
    item = trainer("latent", 0.5)
    item.auxiliary.teacher.noise_std = 0.3
    with pytest.raises(ValueError, match="scalar configuration"):
        item.train_batch(*data(), epoch=0, batch=0)


def test_named_buffers_survive_actual_restore_and_ema_copies_exactly(monkeypatch):
    original = GRUResidualRewardWorldModel.__init__

    def init(self, *args, **kw):
        original(self, *args, **kw)
        self.register_buffer("counter", torch.tensor(2, dtype=torch.int64))
        self.register_buffer("transient", torch.tensor(3.0), persistent=False)

    monkeypatch.setattr(GRUResidualRewardWorldModel, "__init__", init)
    item = trainer("latent", 0.5)
    item.student.counter.fill_(13)
    item.student.transient.fill_(17.0)
    item.train_batch(*data(), epoch=0, batch=0, capture_ema_witness=True)
    assert item.auxiliary.teacher.counter.item() == 13 and item.auxiliary.teacher.transient.item() == 17.0
    restored = training.restore_checkpoint(
        item.export_checkpoint(),
        expected_settings=item.settings,
        expected_source_sha256=item.source_sha256,
        expected_runtime=item.runtime,
    )
    assert restored.student.counter.item() == restored.auxiliary.teacher.counter.item() == 13
    assert restored.student.transient.item() == restored.auxiliary.teacher.transient.item() == 17.0
    payload = item.export_checkpoint()
    payload["student_buffers"]["counter"].add_(1)
    payload["integrity_sha256"] = training.canonical_state_hash(
        {k: v for k, v in payload.items() if k != "integrity_sha256"}
    )
    with pytest.raises(ValueError, match="Persistent buffer"):
        training.restore_checkpoint(
            payload,
            expected_settings=item.settings,
            expected_source_sha256=item.source_sha256,
            expected_runtime=item.runtime,
        )


def test_initialization_rng_states_cannot_be_reassigned_to_other_role():
    cfg = settings()
    initial = training.make_initialization(1121, 1123, cfg)
    initial["rng_states"]["student_before"] = initial["rng_states"]["predictor_before"].clone()
    initial["hashes"]["rng_states"] = training.canonical_tensor_hash(initial["rng_states"])
    with pytest.raises(ValueError, match="RNG role"):
        training.make_trainer("anchor", initial, 1129, 0.0, cfg)


def test_calibration_pool_vectors_formula_and_initial_rng_immutability():
    cfg = settings()
    initial = [training.make_initialization(1201 + i, 1301 + i, cfg) for i in range(3)]
    before = [training.canonical_state_hash(item) for item in initial]
    rng = torch.get_rng_state().clone()
    tensors = data(129)
    public = dict(zip(("packets", "commands", "rewards"), tensors, strict=True))
    result = training.calibrate(initial, public, cfg)
    receipt, gradients = result["receipt"], result["gradients"]
    assert len(receipt["rows"]) == 12 and all(len(row) == 12 for row in receipt["norms"].values())
    assert receipt["training_indices"] == list(range(128))
    assert len(gradients) == 12 * 3 * 8
    assert all(p.startswith(training.BACKBONE_PREFIXES) for p in receipt["backbone_parameter_names"])
    for row in receipt["rows"]:
        for arm in training.ARMS:
            prefix = f"pair{row['pair']}/batch{row['batch']}/{arm}/"
            selected = [v for key, v in gradients.items() if key.startswith(prefix)]
            norm = sum(float(v.double().square().sum()) for v in selected) ** 0.5
            assert norm == pytest.approx(row["losses"][arm]["backbone_norm"], rel=1e-14)
    for arm in ("raw", "latent"):
        wanted = np.clip(
            0.5 * np.median(receipt["norms"]["anchor"]) / np.median(receipt["norms"][arm]), 0.01, 100.0
        )
        assert receipt["multipliers"][arm]["value"] == wanted
    assert before == [training.canonical_state_hash(item) for item in initial]
    assert torch.equal(rng, torch.get_rng_state())
    assert receipt["optimizer_updates"] == receipt["ema_updates"] == 0
    assert (
        sum(v for k, v in receipt["costs"].items() if k != "wall_seconds") <= receipt["costs"]["wall_seconds"]
    )


def test_calibration_rejects_zero_norm_and_preserves_rng_on_deadline(monkeypatch):
    cfg = settings()
    initial = [training.make_initialization(1201 + i, 1301 + i, cfg) for i in range(3)]
    public = dict(zip(("packets", "commands", "rewards"), data(128), strict=True))
    rng = torch.get_rng_state().clone()

    def fail():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        training.calibrate(initial, public, cfg, deadline_check=fail)
    assert torch.equal(rng, torch.get_rng_state())
    monkeypatch.setattr(training, "_norm", lambda _: 0.0)
    with pytest.raises(ValueError, match="Zero/nonfinite"):
        training.calibrate(initial, public, cfg)


def test_settings_interoperate_with_new_protocol_and_bind_scalars():
    from openjev.research import reacher_objective_protocol as protocol

    cfg = training.TrainingSettings.from_plan(protocol.settings(engineering=True))
    assert cfg == training.TrainingSettings()
    assert training.TrainingSettings.from_configuration(cfg.configuration()) == cfg
    with pytest.raises(ValueError):
        dataclasses.replace(cfg, residual_reward=False)
    with pytest.raises(ValueError):
        dataclasses.replace(cfg, variance_weight=0.1)
    with pytest.raises(ValueError):
        dataclasses.replace(cfg, learning_rate=float("nan"))
