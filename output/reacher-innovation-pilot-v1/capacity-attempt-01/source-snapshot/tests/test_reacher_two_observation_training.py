"""Seed410 engineering only. Optimizer steps occur solely in split/resume fixture."""

import copy
import io
from dataclasses import replace
from itertools import pairwise

import pytest
import torch

from openjev.research import reacher_two_observation_training as training
from openjev.research.reacher_memory_training import make_orders
from openjev.research.reacher_reward_residual import GRUResidualRewardWorldModel
from openjev.research.reacher_two_observation_history import TwoObservationHistoryGRUWorldModel


@pytest.fixture(autouse=True)
def isolated_engineering_rng():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


def inputs(**overrides):
    cfg = training.TrainingSettings(hidden_size=3, mlp_width=7, train_episodes=3,
                                   steps=4, epochs=2, batch_size=2, rollout_horizon=3, **overrides)
    return configured_inputs(cfg)


def configured_inputs(cfg):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        base = GRUResidualRewardWorldModel(hidden_size=cfg.hidden_size, dt=cfg.dt,
                                          noise_std=cfg.noise_std, residual_reward=cfg.residual_reward)
    initial = training.clone(base.state_dict())
    angles = torch.linspace(-0.3, 0.4, cfg.train_episodes * (cfg.steps + 1) * 2).reshape(cfg.train_episodes, cfg.steps + 1, 2)
    packets = torch.zeros(cfg.train_episodes, cfg.steps + 1, 8)
    packets[..., :2], packets[..., 2:4] = angles.cos(), angles.sin()
    packets[..., 4:6] = torch.tensor([0.08, -0.04])
    packets[..., 6] = 1
    packets[:, 2, :4] = 0
    packets[:, 2, 6], packets[:, 2, 7] = 0, cfg.dt
    commands = torch.linspace(-0.4, 0.6, cfg.train_episodes * cfg.steps * 2).reshape(cfg.train_episodes, cfg.steps, 2)
    data = {"packets": packets, "commands": commands, "rewards": -commands.square().sum(-1) - 0.1}
    orders = make_orders(410, cfg, role_name="engineering/two-observation/paired-order")
    kwargs = {
        "settings": cfg,
        "expected_initial_sha256": training.canonical_tensor_hash(initial),
        "expected_orders_sha256": training.canonical_state_hash(orders),
        "expected_data_sha256": training.canonical_tensor_hash(data),
        "provenance": {"scope": "synthetic_seed410", "pair_id": "synthetic_pair",
                       "initial_state_kind": "original_initialization", "initial_source_sha256": "a" * 64},
        "source_sha256": {"synthetic_fixture": "b" * 64},
        "runtime": {"scope": "isolated_cpu_seed410_engineering"},
    }
    return initial, orders, data, kwargs


def make(spec=None):
    initial, orders, data, kwargs = spec or inputs()
    return training.TwoObservationTrainer(initial, orders, data, **kwargs)


def expected(spec, payload):
    _, _, _, kw = spec
    return {
        "expected_checkpoint_sha256": payload["integrity_sha256"],
        "expected_kind": training.KIND, "expected_settings": kw["settings"],
        **{key: kw[key] for key in ("expected_initial_sha256", "expected_orders_sha256", "expected_data_sha256")},
        "expected_provenance": kw["provenance"], "expected_source_sha256": kw["source_sha256"],
        "expected_runtime": kw["runtime"],
    }


def restore(spec, payload, **overrides):
    bindings = expected(spec, payload)
    bindings.update(overrides)
    return training.restore_checkpoint(payload, spec[2], **bindings)


def reseal(payload):
    payload.pop("integrity_sha256", None)
    return training._seal(payload)


@pytest.fixture(scope="module")
def split_resume():
    """The only optimizer-step fixture: four tiny updates, three repeated on resume."""
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        torch.set_num_threads(1)
        before = torch.get_rng_state().clone()
        spec = inputs()
        continuous = make(spec)
        first_row = continuous.train_next()
        first = continuous.export_checkpoint()
        buffer = io.BytesIO()
        torch.save(first, buffer)
        buffer.seek(0)
        safe = torch.load(buffer, weights_only=True)
        resumed = restore(spec, safe)
        pairs = []
        for _ in range(3):
            left, right = continuous.train_next(), resumed.train_next()
            for key in ("indices", "indices_sha256", "loss", "anchor", "gradient_norm", "gradient_clipped",
                        "work", "observed_neural_sample_calls", "rng_identity_sha256", "update", "epoch", "batch"):
                assert left[key] == right[key], key
            assert training.canonical_tensor_hash(continuous.student.state_dict()) == training.canonical_tensor_hash(resumed.student.state_dict())
            assert training.canonical_state_hash(continuous.optimizer.state_dict()) == training.canonical_state_hash(resumed.optimizer.state_dict())
            assert continuous.cursor == resumed.cursor
            pairs.append((left, right))
        assert torch.equal(before, torch.get_rng_state())
        result = {"spec": spec, "first": first, "first_row": first_row, "continuous": continuous.export_checkpoint(),
                  "resumed": resumed.export_checkpoint(), "rows": pairs}
    torch.set_num_threads(threads)
    return result


def test_exact_split_resume_final_model_optimizer_and_cursor_including_epoch_boundary(split_resume):
    left, right = split_resume["continuous"], split_resume["resumed"]
    assert left["cursor"] == right["cursor"] == {"epoch": 2, "batch": 0}
    assert left["successful_updates"] == right["successful_updates"] == 4
    assert training.canonical_tensor_hash(left["student_state"]) == training.canonical_tensor_hash(right["student_state"])
    assert training.canonical_state_hash(left["optimizer_state"]) == training.canonical_state_hash(right["optimizer_state"])
    assert training.canonical_state_hash(left["rng"]) == training.canonical_state_hash(right["rng"])
    assert left["rng"]["consumed_epoch_permutations"] == 2
    assert torch.equal(left["rng"]["current_rng"], left["rng"]["final_rng"])
    resumed = restore(split_resume["spec"], right)
    assert type(resumed.student) is TwoObservationHistoryGRUWorldModel
    assert all(p.grad is None for p in resumed.student.parameters())
    with pytest.raises(ValueError, match="completed"):
        resumed.next_indices()


def test_initial_tensors_are_loaded_exactly_without_regeneration_or_aliasing():
    spec = inputs()
    before = torch.get_rng_state().clone()
    item = make(spec)
    assert torch.equal(before, torch.get_rng_state())
    assert training.canonical_tensor_hash(item.student.state_dict()) == spec[3]["expected_initial_sha256"]
    assert not item.optimizer.state
    assert item.rng_identity()["consumed_epoch_permutations"] == 0
    assert torch.equal(item.rng_identity()["current_rng"], spec[1]["initial_rng"])
    spec[0][next(iter(spec[0]))].add_(1)
    spec[1]["orders"].zero_()
    spec[2]["commands"].zero_()
    assert training.canonical_tensor_hash(item.student.state_dict()) == item.initialization["tensor_sha256"]
    assert item.orders["orders"].any() and item._data["commands"].any()


def test_complete_class_settings_and_safe_checkpoint_contract(split_resume, tmp_path):
    spec, payload = split_resume["spec"], split_resume["first"]
    assert payload["model_configuration"]["model_class"] == "TwoObservationHistoryGRUWorldModel"
    assert payload["model_configuration"]["max_real_packets"] == 12
    assert payload["model_configuration"]["max_issued_commands"] == 11
    assert payload["settings"]["objective"] == "unchanged_sequence_loss"
    assert payload["settings"]["optimizer"] == training.ADAM
    assert training.TrainingSettings.from_configuration(payload["settings"]) == spec[3]["settings"]
    restored = restore(spec, payload)
    target = tmp_path / "resume.pt"
    restored.save_checkpoint(target)
    loaded = torch.load(target, weights_only=True)
    assert type(restore(spec, loaded).student) is TwoObservationHistoryGRUWorldModel
    with pytest.raises(FileExistsError):
        restored.save_checkpoint(target)


def test_full_twelve_eleven_reconstruction_work_includes_partial_final_batch(split_resume):
    rows = [split_resume["first_row"]] + [pair[0] for pair in split_resume["rows"]]
    assert [len(row["indices"]) for row in rows] == [2, 1, 2, 1]
    for row in rows:
        count = len(row["indices"])
        work = row["work"]["operations"]
        assert work["replayed_observation_update_samples"] == 12 * count * 4
        assert work["replayed_transition_samples"] == 11 * count * 4
        assert work["advance_samples"] == count * (4 + 3 * 2)
        assert row["observed_neural_sample_calls"] == {key: work[key] for key in ("gru_cell_sample_calls", "linear_layer_sample_calls")}
        assert not row["work"]["compute_matched"]
        assert work["startup_masked_work_is_counted"]
        assert row["costs"]["batch_wall_seconds"] >= sum(row["costs"][key] for key in ("forward_seconds", "backward_seconds", "gradient_clip_seconds", "optimizer_seconds"))
        assert row["anchor"]["kl_nats"] == 0
    assert rows[0]["previous_log_sha256"] == training.canonical_state_hash([])
    for earlier, later in pairwise(rows):
        assert later["previous_log_sha256"] == earlier["log_sha256"]
        assert later["log_sha256"] == training.canonical_state_hash({k: v for k, v in later.items() if k != "log_sha256"})


@pytest.mark.parametrize("field", ["expected_checkpoint_sha256", "expected_kind", "expected_initial_sha256", "expected_orders_sha256",
                                  "expected_data_sha256", "expected_provenance", "expected_source_sha256", "expected_runtime", "expected_settings"])
def test_wrong_external_resume_bindings_rejected_with_rng_unchanged(split_resume, field):
    value = "c" * 64
    if field == "expected_kind":
        value = "residual_gru"
    elif field in {"expected_provenance", "expected_source_sha256", "expected_runtime"}:
        value = {"changed": "c" * 64}
    elif field == "expected_settings":
        value = replace(split_resume["spec"][3]["settings"], noise_std=0.1)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        restore(split_resume["spec"], split_resume["first"], **{field: value})
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("change", ["class", "configuration", "cursor", "updates", "rng", "order", "initial", "names", "weights",
                                   "adam_ids", "adam_step", "adam_moment", "adam_missing", "adam_recipe", "failed", "extra",
                                   "attempt_work", "attempt_cursor", "provenance", "time", "live_rng"])
def test_semantic_corruption_rejected_even_after_outer_checksum_is_rebound(split_resume, change):
    spec, payload = split_resume["spec"], training.clone(split_resume["first"])
    if change == "class":
        payload["model_configuration"]["model_class"] = "GRUResidualRewardWorldModel"
    elif change == "configuration":
        payload["model_configuration"]["valid_observations"] = 3
    elif change == "cursor":
        payload["cursor"]["batch"] += 1
    elif change == "updates":
        payload["optimizer_steps"] += 1
    elif change == "rng":
        payload["rng"]["current_rng"][0] ^= 1
    elif change == "order":
        payload["orders"]["orders"][0] = payload["orders"]["orders"][0].flip(0)
        reseal(payload["orders"])
    elif change == "initial":
        payload["initialization"]["weights"][next(iter(payload["initialization"]["weights"]))].add_(1)
    elif change == "names":
        payload["optimizer_group_names"][0].reverse()
    elif change == "weights":
        key = next(iter(payload["student_state"]))
        payload["student_state"][key] = payload["student_state"][key].double()
    elif change == "adam_ids":
        payload["optimizer_state"]["param_groups"][0]["params"].reverse()
    elif change == "adam_step":
        payload["optimizer_state"]["state"][0]["step"] += 1
    elif change == "adam_moment":
        payload["optimizer_state"]["state"][0]["exp_avg_sq"].fill_(-1)
    elif change == "adam_missing":
        payload["optimizer_state"]["state"].pop(0)
    elif change == "adam_recipe":
        payload["optimizer_state"]["param_groups"][0]["lr"] *= 2
    elif change == "failed":
        payload["failed"] = True
    elif change == "extra":
        payload["extra"] = 1
    elif change == "attempt_work":
        payload["last_attempt"]["completed_neural_sample_calls"]["gru_cell_sample_calls"] -= 1
    elif change == "attempt_cursor":
        payload["last_attempt"]["cursor"]["batch"] += 1
    elif change == "provenance":
        payload["provenance"]["pair_id"] = "other"
    elif change == "time":
        payload["training_wall_seconds"] = -1
    else:
        payload["rng"]["neural_rng"] = "allow hidden stochastic draws"
    reseal(payload)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        restore(spec, payload)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("change", ["initial", "order_rng", "order_shape", "order_extra", "order_seed", "data", "provenance", "runtime", "source"])
def test_constructor_authentication_and_complete_order_schema_rejection(change):
    spec = inputs()
    initial, orders, data, kw = spec
    if change == "initial":
        initial[next(iter(initial))].add_(1)
    elif change.startswith("order"):
        if change == "order_rng":
            orders["initial_rng"][0] ^= 1
        elif change == "order_shape":
            orders["orders"] = orders["orders"].float()
        elif change == "order_extra":
            orders["extra"] = 0
        else:
            orders["seed"] = True
        reseal(orders)
        kw["expected_orders_sha256"] = training.canonical_state_hash(orders)
    elif change == "data":
        data["commands"].mul_(0.5)
    elif change in {"runtime", "provenance"}:
        kw[change] = {}
    else:
        kw["source_sha256"] = {"unbound": "not-a-hash"}
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        make(spec)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("change", ["privileged", "target", "age", "missing", "initial_missing", "action", "nonfinite"])
def test_complete_public_training_contract_fails_closed(change):
    spec = inputs()
    data = spec[2]
    if change == "privileged":
        data["qvel"] = torch.zeros(3, 5, 2)
    elif change == "target":
        data["packets"][:, -1, 4] += 0.01
    elif change == "age":
        data["packets"][:, 2, 7] += 0.01
    elif change == "missing":
        data["packets"][:, 2, 0] = 999
    elif change == "initial_missing":
        data["packets"][:, 0, :4] = 0
        data["packets"][:, 0, 6] = 0
    elif change == "action":
        data["commands"][:, 0, 0] = 1.01
    else:
        data["rewards"][0, 0] = float("nan")
    if change != "nonfinite":
        spec[3]["expected_data_sha256"] = training.canonical_tensor_hash(data)
    with pytest.raises(ValueError):
        make(spec)


def test_complete_data_preflight_rejects_unsupported_history_without_model_forward(monkeypatch):
    cfg = replace(inputs()[3]["settings"], steps=13)
    spec = configured_inputs(cfg)
    spec[2]["packets"][:, 1:, :4] = 0
    spec[2]["packets"][:, 1:, 6] = 0
    spec[2]["packets"][:, 1:, 7] = torch.arange(1, 14).float() * cfg.dt
    spec[3]["expected_data_sha256"] = training.canonical_tensor_hash(spec[2])
    def forbidden(*_args, **_kwargs):
        raise AssertionError("No model construction required for invalid data")
    monkeypatch.setattr(training, "_construct", forbidden)
    with pytest.raises(ValueError, match="overflow"):
        make(spec)


@pytest.mark.parametrize("kind", ["class", "noise", "target_metadata", "rng_cache", "optimizer_owner"])
def test_live_configuration_parameter_ownership_and_rng_drift_rejected_before_forward(kind):
    item = make()
    if kind == "class":
        item.student = GRUResidualRewardWorldModel(hidden_size=3)
    elif kind == "noise":
        item.student.noise_std = 0.1
    elif kind == "target_metadata":
        item.model_configuration["max_real_packets"] = 3
    elif kind == "rng_cache":
        item._order_rng_states[0][0] ^= 1
    else:
        item.optimizer.param_groups[0]["params"].reverse()
    with pytest.raises(ValueError):
        item._check_live()
    assert item.successful_updates == 0


def test_constructor_random_scope_restored_even_if_helper_draws_then_raises(monkeypatch):
    spec = inputs()
    def broken(_settings):
        torch.rand(2)
        raise RuntimeError("construction failure")
    monkeypatch.setattr(training, "_construct", broken)
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="construction failure"):
        make(spec)
    assert torch.equal(before, torch.get_rng_state())


def test_zero_update_resume_cannot_substitute_trained_or_changed_tensors():
    spec = inputs()
    payload = make(spec).export_checkpoint()
    payload["student_state"][next(iter(payload["student_state"]))].add_(0.01)
    reseal(payload)
    with pytest.raises(ValueError, match="Zero-update"):
        restore(spec, payload)


def test_failure_after_forward_preserves_paid_neural_work_without_an_optimizer_step():
    spec = inputs()
    item = make(spec)
    initial = training.canonical_tensor_hash(item.student.state_dict())
    calls = 0
    def stop_after_forward():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError("after paid engineering forward")
    with pytest.raises(TimeoutError, match="paid engineering"):
        item.train_next(deadline_check=stop_after_forward)
    assert item.failed and item.optimizer_steps == item.successful_updates == 0
    expected = item._work(2)["operations"]
    assert item.last_attempt["completed_neural_sample_calls"] == {key: expected[key] for key in ("gru_cell_sample_calls", "linear_layer_sample_calls")}
    assert training.canonical_tensor_hash(item.student.state_dict()) == initial
    assert not item.optimizer.state and all(p.grad is None for p in item.student.parameters())
    assert all(not module._forward_hooks for module in item.student.modules())
    with pytest.raises(ValueError, match="terminal"):
        restore(spec, item.export_checkpoint())


@pytest.mark.parametrize("failure", ["deadline", "keyboard_interrupt", "unexpected_rng"])
def test_pre_forward_failure_is_terminal_preserved_and_rng_restored(failure):
    spec = inputs()
    item = make(spec)
    before = torch.get_rng_state().clone()
    initial = training.canonical_tensor_hash(item.student.state_dict())
    def stop():
        if failure == "deadline":
            raise TimeoutError("engineering deadline")
        if failure == "keyboard_interrupt":
            raise KeyboardInterrupt("engineering interrupt")
        torch.rand(1)
    error_type = {"deadline": TimeoutError, "keyboard_interrupt": KeyboardInterrupt, "unexpected_rng": ValueError}[failure]
    with pytest.raises(error_type):
        item.train_next(deadline_check=stop)
    assert item.failed and item.successful_updates == item.optimizer_steps == 0
    assert item.last_attempt["completed_neural_sample_calls"] == {"gru_cell_sample_calls": 0, "linear_layer_sample_calls": 0}
    assert item.failure["type"] == error_type.__name__
    assert training.canonical_tensor_hash(item.student.state_dict()) == initial
    assert torch.equal(before, torch.get_rng_state())
    assert not item.optimizer.state and all(p.grad is None for p in item.student.parameters())
    assert all(not module._forward_hooks for module in item.student.modules())
    failed = item.export_checkpoint()
    with pytest.raises(ValueError, match="terminal"):
        restore(spec, failed)
    with pytest.raises(ValueError, match="terminal"):
        item.train_next()


def test_additive_settings_do_not_admit_unbounded_sequences_or_implicit_metadata():
    cfg = inputs()[3]["settings"]
    with pytest.raises(ValueError, match="fifty"):
        replace(cfg, steps=51)
    with pytest.raises(TypeError):
        training.TwoObservationTrainer({}, {}, {})
    config = copy.deepcopy(cfg.configuration())
    config["objective"] = "different loss"
    with pytest.raises(ValueError):
        training.TrainingSettings.from_configuration(config)
