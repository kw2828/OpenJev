"""Engineering CPU data, only synthetic seed410; no study or model evaluation."""

import copy
import io

import pytest
import torch

from openjev.research import reacher_memory_training as training
from openjev.research.reacher_world_models import sequence_loss


@pytest.fixture(autouse=True)
def isolate():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


def settings(**kwargs):
    return training.MemoryTrainingSettings(
        hidden_size=4, mlp_width=7, train_episodes=5, steps=6, epochs=2, batch_size=2, **kwargs
    )


def data(cfg):
    generator = torch.Generator().manual_seed(410)
    angles = torch.randn(cfg.train_episodes, cfg.steps + 1, 2, generator=generator) * 0.2
    packets = torch.zeros(cfg.train_episodes, cfg.steps + 1, 8)
    packets[..., :2], packets[..., 2:4] = angles.cos(), angles.sin()
    packets[..., 4:6] = torch.rand(cfg.train_episodes, 1, 2, generator=generator) * 0.1
    packets[..., 6] = 1
    packets[:, 2:4, :4] = 0
    packets[:, 2:4, 6] = 0
    packets[:, 2:4, 7] = torch.tensor([0.02, 0.04])
    commands = torch.randn(cfg.train_episodes, cfg.steps, 2, generator=generator).clamp(-1, 1) * 0.1
    rewards = -commands.square().sum(-1) - 0.1
    return {"packets": packets, "commands": commands, "rewards": rewards}


def trainer(kind="residual_gru", cfg=None):
    cfg = cfg or settings()
    return training.make_trainer(
        kind,
        training.make_initialization(410, 410, cfg),
        training.make_orders(410, cfg),
        data(cfg),
        source_sha256={"synthetic": "a" * 64},
        runtime={"engineering": True},
    )


def restore(item, payload=None, **overrides):
    expected = {
        "expected_kind": item.kind,
        "expected_settings": item.settings,
        "expected_initialization_sha256": item.initialization["integrity_sha256"],
        "expected_orders_sha256": item.orders["integrity_sha256"],
        "expected_data_sha256": item.data_sha256,
        "expected_source_sha256": item.source_sha256,
        "expected_runtime": item.runtime,
    }
    expected.update(overrides)
    return training.restore_checkpoint(payload or item.export_checkpoint(), data(item.settings), **expected)


def reseal(payload):
    payload.pop("integrity_sha256")
    return training._seal(payload)


def equal_tensors(left, right):
    assert set(left) == set(right)
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_shared_gru_payload_and_independent_mlp_named_initialization_preserve_global_rng():
    cfg = settings()
    before = torch.get_rng_state().clone()
    initial = training.make_initialization(410, 410, cfg)
    orders = training.make_orders(410, cfg)
    cohort = data(cfg)
    items = [training.make_trainer(k, initial, orders, cohort) for k in training.REGISTRY]
    assert torch.equal(before, torch.get_rng_state())
    assert len(set(initial["roles"].values())) == 2
    for item in items[:3]:
        equal_tensors(item.student.state_dict(), initial["states"]["gru"])
    equal_tensors(items[3].student.state_dict(), initial["states"]["mlp"])
    with torch.no_grad():
        next(items[0].student.parameters()).add_(1)
    equal_tensors(items[1].student.state_dict(), initial["states"]["gru"])
    cohort["commands"].zero_()
    assert not torch.equal(items[0]._data["commands"], cohort["commands"])
    orders["orders"].zero_()
    assert not torch.equal(items[0].orders["orders"], orders["orders"])
    assert [type(item.student) for item in items] == list(training.REGISTRY.values())


@pytest.mark.parametrize("kind", training.REGISTRY)
def test_two_updates_match_unchanged_sequence_loss_adam_and_clipping_exactly(kind):
    item = trainer(kind)
    model = copy.deepcopy(item.student)
    kwargs = {k: v for k, v in training.ADAM.items() if k != "name"}
    kwargs["betas"] = tuple(kwargs["betas"])
    optimizer = torch.optim.Adam(model.parameters(), lr=item.settings.learning_rate, **kwargs)
    for update in range(2):
        indices = item.next_indices()
        batch = [item._data[k][indices] for k in ("packets", "commands", "rewards")]
        optimizer.zero_grad(set_to_none=True)
        loss, values = sequence_loss(model, *batch, **item.settings.anchor_kwargs())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), item.settings.gradient_clip, error_if_nonfinite=True
        )
        optimizer.step()
        metrics = item.train_next()
        assert metrics["anchor"] == values
        assert metrics["loss"] == float(loss.detach())
        assert metrics["gradient_norm"] == float(norm)
        assert metrics["indices"] == indices.tolist()
        assert metrics["update"] == update + 1
        equal_tensors(item.student.state_dict(), model.state_dict())
        assert training.canonical_state_hash(item.optimizer.state_dict()) == training.canonical_state_hash(
            optimizer.state_dict()
        )
        assert all(p.grad is None for p in item.student.parameters())


@pytest.mark.parametrize("kind", training.REGISTRY)
def test_full_safe_restore_and_uninterrupted_continuation_including_epoch_boundary(kind):
    item = trainer(kind)
    item.train_next()
    stream = io.BytesIO()
    torch.save(item.export_checkpoint(), stream)
    stream.seek(0)
    payload = torch.load(stream, weights_only=True)
    before = torch.get_rng_state().clone()
    resumed = restore(item, payload)
    assert torch.equal(before, torch.get_rng_state())
    assert type(resumed.student) is training.REGISTRY[kind]
    for _ in range(4):
        assert torch.equal(item.next_indices(), resumed.next_indices())
        left, right = item.train_next(), resumed.train_next()
        for key in ("indices", "anchor", "loss", "gradient_norm", "work", "update"):
            assert left[key] == right[key]
        equal_tensors(item.student.state_dict(), resumed.student.state_dict())
        assert training.canonical_state_hash(item.optimizer.state_dict()) == training.canonical_state_hash(
            resumed.optimizer.state_dict()
        )
        assert item.cursor == resumed.cursor
    assert item.cursor == {"epoch": 1, "batch": 2}


def test_exact_order_and_final_partial_batch_all_arms():
    items = [trainer(k) for k in training.REGISTRY]
    for epoch in range(2):
        combined = []
        for batch in range(3):
            rows = [item.train_next() for item in items]
            assert all(row["indices"] == rows[0]["indices"] for row in rows)
            assert all((row["epoch"], row["batch"]) == (epoch, batch) for row in rows)
            combined += rows[0]["indices"]
        assert sorted(combined) == list(range(5))
    for item in items:
        assert item.cursor == {"epoch": 2, "batch": 0}
        with pytest.raises(ValueError, match="completed"):
            item.next_indices()
        restored = restore(item)
        assert restored.cursor == item.cursor


@pytest.mark.parametrize(
    "original,wrong",
    [
        ("residual_gru", "current_gru"),
        ("current_gru", "bounded_gru"),
        ("bounded_gru", "residual_gru"),
        ("packet_mlp", "current_gru"),
    ],
)
def test_same_tensor_names_cannot_silently_restore_wrong_model_class(original, wrong):
    item = trainer(original)
    item.train_next()
    with pytest.raises(ValueError, match="kind/settings"):
        restore(item, expected_kind=wrong)
    payload = item.export_checkpoint()
    payload["model_configuration"]["model_class"] = training.REGISTRY[wrong].__name__
    reseal(payload)
    with pytest.raises(ValueError, match="semantics"):
        restore(item, payload)


@pytest.mark.parametrize(
    "mutation",
    [
        "cursor",
        "step",
        "moment",
        "optimizer_order",
        "recipe",
        "order",
        "initialization",
        "extra",
        "failed",
        "dtype",
        "configuration",
        "data",
        "source",
        "runtime",
    ],
)
def test_checkpoint_corruption_rejected_even_with_recomputed_outer_hash(mutation):
    item = trainer()
    item.train_next()
    payload = item.export_checkpoint()
    if mutation == "cursor":
        payload["cursor"]["batch"] += 1
    elif mutation == "step":
        payload["optimizer_state"]["state"][0]["step"] += 1
    elif mutation == "moment":
        payload["optimizer_state"]["state"][0]["exp_avg_sq"].fill_(-1)
    elif mutation == "optimizer_order":
        payload["optimizer_state"]["param_groups"][0]["params"].reverse()
    elif mutation == "recipe":
        payload["optimizer_state"]["param_groups"][0]["lr"] *= 2
    elif mutation == "order":
        payload["orders"]["orders"][0] = payload["orders"]["orders"][0].flip(0)
        reseal(payload["orders"])
    elif mutation == "initialization":
        payload["initialization"]["states"]["gru"]["observation_update.weight_ih"].add_(1)
        reseal(payload["initialization"])
    elif mutation == "extra":
        payload["extra"] = 0
    elif mutation == "failed":
        payload["failed"] = True
    elif mutation == "dtype":
        payload["student_state"]["observation_update.weight_ih"] = payload["student_state"][
            "observation_update.weight_ih"
        ].double()
    elif mutation == "configuration":
        payload["model_configuration"]["real_assimilation"] = "wrong"
    elif mutation == "data":
        payload["data_sha256"] = "b" * 64
    elif mutation == "source":
        payload["source_sha256"]["synthetic"] = "b" * 64
    else:
        payload["runtime"]["engineering"] = False
    reseal(payload)
    with pytest.raises(ValueError):
        restore(item, payload)


@pytest.mark.parametrize("change", ["extra", "action", "time", "missing", "initial_missing", "nonfinite"])
def test_public_data_contract_rejects_privileged_or_misaligned_inputs(change):
    cfg = settings()
    public = data(cfg)
    if change == "extra":
        public["raw_qvel"] = torch.zeros(5, 7, 2)
    elif change == "action":
        public["commands"][0, 0, 0] = 1.1
    elif change == "time":
        public["commands"] = public["commands"][:, :-1]
    elif change == "missing":
        public["packets"][0, 2, 0] = 123.0
    elif change == "initial_missing":
        public["packets"][:, 0, :4] = 0
        public["packets"][:, 0, 6] = 0
    else:
        public["rewards"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        training.make_trainer(
            "residual_gru",
            training.make_initialization(410, 410, cfg),
            training.make_orders(410, cfg),
            public,
        )


def test_bounded_real_packet_action_alignment_during_unchanged_loss(monkeypatch):
    item = trainer("bounded_gru")
    seen = []
    original = item.student.assimilate

    def capture(state, packet):
        result = original(state, packet)
        seen.append(training.clone(result))
        return result

    monkeypatch.setattr(item.student, "assimilate", capture)
    indices = item.next_indices()
    item.train_next()
    assert len(seen) == item.settings.steps
    for step, state in enumerate(seen):
        width = min(3, step + 1)
        torch.testing.assert_close(
            state["real_packets"][:, -width:],
            item._data["packets"][indices, step + 1 - width : step + 1],
            rtol=0,
            atol=0,
        )
        if width > 1:
            torch.testing.assert_close(
                state["real_actions"][:, -(width - 1) :],
                item._data["commands"][indices, step + 1 - width : step],
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("kind", ("current_gru", "packet_mlp"))
def test_current_packet_models_erase_poisoned_prior_prediction_in_training_prefix(kind, monkeypatch):
    reference, poisoned = trainer(kind), trainer(kind)
    original = poisoned.student.assimilate

    def capture(state, packet):
        bad = {key: torch.full_like(value, float("nan")) for key, value in state.items()}
        return original(bad, packet)

    monkeypatch.setattr(poisoned.student, "assimilate", capture)
    a, b = reference.train_next(), poisoned.train_next()
    assert a["anchor"] == b["anchor"]
    equal_tensors(reference.student.state_dict(), poisoned.student.state_dict())


@pytest.mark.parametrize("exception", (RuntimeError, KeyboardInterrupt))
def test_failed_optimizer_preserves_baseexception_and_blocks_restore(monkeypatch, exception):
    item = trainer()

    def fail():
        raise exception("synthetic failure")

    monkeypatch.setattr(item.optimizer, "step", fail)
    with pytest.raises(exception, match="synthetic"):
        item.train_next()
    assert item.failed and item.successful_updates == item.optimizer_steps == 0
    assert all(p.grad is None for p in item.student.parameters())
    with pytest.raises(ValueError, match="terminal"):
        restore(item)
    with pytest.raises(ValueError, match="terminal"):
        item.train_next()


def test_order_and_owned_data_mutation_fail_before_optimizer():
    item = trainer()
    item.orders["orders"][0] = item.orders["orders"][0].flip(0)
    with pytest.raises(ValueError, match="order changed"):
        item.train_next()
    assert item.optimizer_steps == 0
    item = trainer()
    item._data["commands"][0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="data changed"):
        item.train_next()
    assert item.optimizer_steps == 0


def test_operation_accounting_distinguishes_bounded_replay_and_mlp():
    rows = {kind: trainer(kind).train_next() for kind in training.REGISTRY}
    interfaces = rows["residual_gru"]["work"]["interfaces"]
    assert all(row["work"]["interfaces"] == interfaces for row in rows.values())
    a = interfaces["sample_forward_evaluations"]["student_assimilate"]
    d = sum(
        interfaces["sample_forward_evaluations"][k]
        for k in ("student_prefix_advance", "student_rollout_advance")
    )
    assert rows["residual_gru"]["work"]["operations"]["gru_cell_sample_calls"] == a + d
    assert rows["bounded_gru"]["work"]["operations"]["gru_cell_sample_calls"] == 5 * a + d
    assert rows["packet_mlp"]["work"]["operations"]["gru_cell_sample_calls"] == 0
    assert all(row["work"]["compute_matched"] is False for row in rows.values())


def test_checkpoint_file_roundtrip_is_exclusive_and_preserves_initial_cursor(tmp_path):
    item = trainer("packet_mlp")
    path = tmp_path / "checkpoint.pt"
    item.save_checkpoint(path)
    restored = restore(item, torch.load(path, weights_only=True))
    assert restored.cursor == {"epoch": 0, "batch": 0}
    assert restored.optimizer.state_dict()["state"] == {}
    assert torch.equal(restored.next_indices(), item.next_indices())
    with pytest.raises(FileExistsError):
        item.save_checkpoint(path)


@pytest.mark.parametrize("fail_at", (1, 2, 3, 4))
def test_deadline_before_or_after_update_is_terminal_without_swallowing_interrupt(fail_at):
    item = trainer("bounded_gru")
    calls = 0

    def deadline():
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise KeyboardInterrupt("synthetic deadline")

    with pytest.raises(KeyboardInterrupt, match="deadline"):
        item.train_next(deadline_check=deadline)
    assert item.failed
    assert item.optimizer_steps == item.successful_updates == (1 if fail_at == 4 else 0)
    with pytest.raises(ValueError, match="terminal"):
        restore(item)


@pytest.mark.parametrize(
    "mutation", ("noise_std", "hidden_size", "residual_reward", "optimizer", "requires_grad")
)
def test_live_configuration_and_optimizer_drift_fail_before_updates(mutation):
    item = trainer()
    if mutation == "noise_std":
        item.student.noise_std = 0.2
    elif mutation == "hidden_size":
        item.student.hidden_size += 1
    elif mutation == "residual_reward":
        item.student.residual_reward = False
    elif mutation == "optimizer":
        item.optimizer.param_groups[0]["lr"] *= 2
    else:
        next(item.student.parameters()).requires_grad_(False)
    with pytest.raises(ValueError):
        item.train_next()
    assert item.failed and item.optimizer_steps == 0


def test_configuration_delegates_run_status_to_protocol_and_receipts():
    cfg = settings()
    for value in [cfg.configuration(), *(training.model_configuration(k, cfg) for k in training.REGISTRY)]:
        assert value["run_status_authority"] == "enclosing protocol and execution receipts"
        assert not {"engineering_only", "integrated_study", "integrated_into_existing_study"} & value.keys()
