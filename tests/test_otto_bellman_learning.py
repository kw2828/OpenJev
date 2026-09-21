"""Tiny synthetic continuation fits only; no scientific checkpoints or data."""
import copy

import numpy as np
import pytest
import torch

from openjev.research import otto_bellman_learning as learning
from openjev.research.otto_return_value import INPUT_DIM, export_head


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def checkpoint():
    return {"version": "otto-return-value-v1", "kind": "mlp8", "input_dim": INPUT_DIM,
            "c0": np.asarray(.25, dtype=np.float32),
            "first_weight": np.full((8, INPUT_DIM), .001, dtype=np.float32),
            "final_weight": np.linspace(.01, .08, 8, dtype=np.float32),
            "hidden_bias": np.linspace(.001, .008, 8, dtype=np.float32),
            "output_bias": np.asarray(-.01, dtype=np.float32)}


def data(rows=3):
    x = np.zeros((rows, INPUT_DIM), dtype=np.float32)
    x[:, 0] = np.linspace(.1, .9, rows, dtype=np.float32)
    x[:, -3:] = .05
    return x, np.linspace(.2, .8, rows, dtype=np.float64)


def recipe(**overrides):
    return learning.Recipe(**{"epochs": 2, "batch_size": 2, **overrides})


def test_production_recipe_exact_and_restore_preserves_every_float32_byte():
    assert learning.Recipe() == learning.Recipe(40, 128, .001, 5., 5)
    assert learning.continue_learning.__kwdefaults__["recipe"] == learning.Recipe()
    initial = checkpoint()
    before_rng = torch.random.get_rng_state().clone()
    for seed in (10101, 10102):
        model = learning.restore_mlp(initial, seed)
        restored = export_head(model)
        assert learning.checkpoint_identity(restored) == learning.checkpoint_identity(initial)
        for name, array in initial.items():
            if isinstance(array, np.ndarray):
                assert restored[name].dtype == np.float32 and restored[name].tobytes() == array.tobytes()
        assert not model.c0.requires_grad
    assert torch.equal(torch.random.get_rng_state(), before_rng)


def test_one_update_matches_independent_fresh_adam_route_exactly():
    initial, (x, y) = checkpoint(), data()
    model = learning.restore_mlp(initial, 10101)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    assert not optimizer.state
    order = np.random.default_rng(10101 + 30000).permutation(len(x))
    prediction = model(torch.from_numpy(x[order]))
    loss = ((prediction - torch.from_numpy(y.astype(np.float32)[order]))**2).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    expected = export_head(model)
    actual = learning.continue_learning(initial, x, y, seed=10101, mode="mc", recipe=recipe(epochs=1, batch_size=3))
    for name, value in expected.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual.final_export[name], value)
    assert actual.optimizer_steps == (1, 1, 1, 1)
    assert actual.epochs[0]["training_mse_normalized"] == float(loss.detach())


def test_mc_and_fixed_backup_share_orders_updates_and_final_parameters():
    x, y = data(5)
    orders, events = {"mc": [], "backup": []}, []
    outputs = {}
    for mode in ("mc", "backup"):
        def hook(event, payload, mode=mode):
            if event == "order":
                orders[mode].append(payload["order"].copy())
            if event == "operation_return" and payload["operation"] == "optimizer_update":
                events.append(payload)
        outputs[mode] = learning.continue_learning(checkpoint(), x, y, seed=10102, mode=mode,
            recipe=recipe(epochs=3), hook=hook,
            target_builder=(lambda _frozen, _metadata: y.copy()) if mode == "backup" else None)
    rng = np.random.default_rng(10102 + 30000)
    for mc, backup in zip(orders["mc"], orders["backup"], strict=True):
        np.testing.assert_array_equal(mc, backup)
        np.testing.assert_array_equal(mc, rng.permutation(5))
    assert outputs["mc"].final_identity == outputs["backup"].final_identity
    assert outputs["mc"].progress["calls"]["target_refresh"] == {"attempted": 0, "returned": 0}
    assert outputs["backup"].progress["calls"]["target_refresh"] == {"attempted": 1, "returned": 1}
    assert all(result.optimizer_steps == (9,) * 4 for result in outputs.values())
    assert len(events) == 18 and all(p["result"]["gradient_norm_before_clip"] >= 0 for p in events)


def test_refresh_before_epochs_one_and_six_and_frozen_array_hooks():
    x, y = data()
    snapshots, target_events, epochs = [], [], []
    def targets(frozen, metadata):
        assert not torch.is_grad_enabled()
        assert metadata["rows"] == 3
        snapshots.append((frozen, copy.deepcopy(metadata)))
        return np.full(3, frozen.normalized(x.astype(np.float64)).mean() + .5, dtype=np.float64)
    def hook(event, payload):
        if event == "refresh":
            target_events.append(payload)
        if event == "epoch":
            epochs.append(payload)
    result = learning.continue_learning(checkpoint(), x, y, seed=10101, mode="backup",
        recipe=recipe(epochs=6), target_builder=targets, hook=hook)
    assert [item[1]["epoch"] for item in snapshots] == [1, 6]
    assert [item[1]["source_epochs_completed"] for item in snapshots] == [0, 5]
    assert snapshots[0][1]["checkpoint_identity"] == result.initial_identity
    assert snapshots[1][1]["checkpoint_identity"] != result.initial_identity
    assert [e["target_refresh_epoch"] for e in epochs] == [1, 1, 1, 1, 1, 6]
    assert result.progress["updates_completed"] == 12 and result.progress["refreshes_completed"] == 2
    for event in target_events:
        np.testing.assert_array_equal(event["targets_float32"], event["targets_float64"].astype(np.float32))
        with pytest.raises(ValueError):
            event["targets_float64"].setflags(write=True)
        with pytest.raises(TypeError):
            event["checkpoint"]["kind"] = "other"
    assert snapshots[0][0].first_weight[0, 0] == float(checkpoint()["first_weight"][0, 0])
    assert result.final_export["c0"].tobytes() == checkpoint()["c0"].tobytes()


def test_epoch_mse_weights_tail_by_rows_and_input_ownership():
    initial, (x, y) = checkpoint(), data(5)
    initial_copy, x_copy, y_copy = copy.deepcopy(initial), x.copy(), y.copy()
    updates, final_events = [], []
    def hook(event, payload):
        if event == "operation_return" and payload["operation"] == "optimizer_update":
            updates.append(payload["result"])
        if event == "final":
            final_events.append(payload)
    result = learning.continue_learning(initial, x, y, seed=10101, mode="mc", recipe=recipe(epochs=1), hook=hook)
    assert [u["rows"] for u in updates] == [2, 2, 1]
    expected = sum(u["loss"] * u["rows"] for u in updates) / 5
    assert result.epochs[0]["training_mse_normalized"] == expected
    assert result.epochs[0]["training_mse_normalized"] != sum(u["loss"] for u in updates) / 3
    np.testing.assert_array_equal(x, x_copy)
    np.testing.assert_array_equal(y, y_copy)
    for name in initial:
        if isinstance(initial[name], np.ndarray):
            np.testing.assert_array_equal(initial[name], initial_copy[name])
    assert len(final_events) == 1 and final_events[0]["checkpoint_identity"] == result.final_identity
    assert result.final_identity != result.initial_identity


def test_later_refresh_failure_retains_prior_completed_updates_and_original_cause():
    x, y = data()
    calls, failures = [], []
    error = RuntimeError("synthetic refresh failure")
    def targets(_frozen, metadata):
        calls.append(metadata["epoch"])
        if len(calls) == 2:
            raise error
        return y.copy()
    def hook(event, payload):
        if event == "failure":
            failures.append(payload)
    with pytest.raises(learning.LearningFailure) as caught:
        learning.continue_learning(checkpoint(), x, y, seed=10101, mode="backup",
            recipe=recipe(epochs=5, refresh_every=2), target_builder=targets, hook=hook)
    progress = caught.value.progress
    assert caught.value.original is error and caught.value.__cause__ is error
    assert calls == [1, 3] and len(failures) == 1
    assert progress["epochs_completed"] == 2 and progress["updates_completed"] == 4
    assert progress["refreshes_completed"] == 1
    assert progress["calls"]["target_refresh"] == {"attempted": 2, "returned": 1}


@pytest.mark.parametrize("event", ["operation_attempt", "operation_return"])
def test_budget_hook_failure_stops_without_retry_and_counts_real_return(event):
    seen = []
    def hook(name, payload):
        if name == event and payload["operation"] == "optimizer_update":
            seen.append(name)
            raise TimeoutError("caller budget exhausted")
    with pytest.raises(learning.LearningFailure) as caught:
        learning.continue_learning(checkpoint(), *data(), seed=10101, mode="mc", recipe=recipe(), hook=hook)
    progress = caught.value.progress
    returned = int(event == "operation_return")
    assert seen == [event]
    assert progress["calls"]["optimizer_update"] == {"attempted": 1, "returned": returned}
    assert progress["updates_completed"] == returned
    assert isinstance(caught.value.__cause__, TimeoutError)


def test_failure_hook_does_not_replace_primary_failure():
    original = RuntimeError("original target failure")
    def targets(_frozen, _metadata):
        raise original
    def hook(event, _payload):
        if event == "failure":
            raise OSError("log failed")
    with pytest.raises(learning.LearningFailure) as caught:
        learning.continue_learning(checkpoint(), *data(), seed=10101, mode="backup",
            recipe=recipe(), target_builder=targets, hook=hook)
    assert caught.value.original is original and caught.value.__cause__ is original
    assert "log failed" in caught.value.__notes__[0]


def test_signed_backup_targets_and_growth_are_retained_without_clipping():
    x, y = data(2)
    supplied = np.array([-.5, 20.], dtype=np.float64)
    records = []
    def hook(event, payload):
        if event == "refresh":
            records.append(payload)
    result = learning.continue_learning(checkpoint(), x, y, seed=10101, mode="backup",
        recipe=recipe(epochs=1), target_builder=lambda _frozen, _metadata: supplied, hook=hook)
    np.testing.assert_array_equal(records[0]["targets_float64"], supplied)
    np.testing.assert_array_equal(records[0]["targets_float32"], supplied.astype(np.float32))
    assert records[0]["statistics"]["negative_count"] == 1
    assert records[0]["statistics"]["maximum"] == 20.
    assert result.progress["updates_completed"] == 1


def test_target_checkpoint_hook_precedes_builder_and_failure_prevents_readout():
    events = []
    def hook(event, payload):
        if event == "target_checkpoint":
            assert payload["checkpoint_identity"] == learning.checkpoint_identity(payload["checkpoint"])
            events.append((event, payload["epoch"]))
    def targets(_frozen, metadata):
        assert events[-1] == ("target_checkpoint", metadata["epoch"])
        events.append(("builder", metadata["epoch"]))
        return data()[1]
    learning.continue_learning(checkpoint(), *data(), seed=10101, mode="backup",
        recipe=recipe(epochs=2, refresh_every=1), target_builder=targets, hook=hook)
    assert events == [("target_checkpoint", 1), ("builder", 1), ("target_checkpoint", 2), ("builder", 2)]
    def failing_hook(event, _payload):
        if event == "target_checkpoint":
            raise OSError("checkpoint publication failed")
    with pytest.raises(learning.LearningFailure) as caught:
        learning.continue_learning(checkpoint(), *data(), seed=10101, mode="backup",
            recipe=recipe(), target_builder=lambda *_: pytest.fail("builder before durable checkpoint"), hook=failing_hook)
    assert caught.value.progress["calls"]["target_refresh"] == {"attempted": 0, "returned": 0}
    assert caught.value.progress["updates_completed"] == 0


@pytest.mark.parametrize("bad", ["dtype", "shape", "nan", "overflow"])
def test_bad_backup_targets_stop_before_optimizer_update(bad):
    def targets(_frozen, _metadata):
        if bad == "dtype":
            return np.zeros(3, dtype=np.float32)
        if bad == "shape":
            return np.zeros(2, dtype=np.float64)
        return np.full(3, np.nan if bad == "nan" else 1e100, dtype=np.float64)
    with pytest.raises(learning.LearningFailure) as caught:
        learning.continue_learning(checkpoint(), *data(), seed=10101, mode="backup",
            recipe=recipe(), target_builder=targets)
    assert caught.value.progress["updates_completed"] == 0
    assert caught.value.progress["calls"]["target_refresh"] == {"attempted": 1, "returned": 0}


@pytest.mark.parametrize("mutation", ["family", "weights", "features", "target", "mode", "callback", "seed"])
def test_invalid_inputs_rejected_before_any_operation(mutation):
    initial, (x, y), events = checkpoint(), data(), []
    options = {"seed": 10101, "mode": "mc", "recipe": recipe(), "hook": lambda event, _payload: events.append(event)}
    if mutation == "family":
        initial["kind"] = "min8"
    elif mutation == "weights":
        initial["first_weight"] = initial["first_weight"].astype(np.float64)
    elif mutation == "features":
        x = x.astype(np.float64)
    elif mutation == "target":
        y[0] = np.inf
    elif mutation == "mode":
        options["mode"] = "other"
    elif mutation == "callback":
        options["target_builder"] = lambda *_: y
    else:
        options["seed"] = True
    with pytest.raises(ValueError):
        learning.continue_learning(initial, x, y, **options)
    assert not events


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"batch_size": 1025}, {"batch_size": True},
    {"learning_rate": np.nan}, {"learning_rate": True}, {"gradient_clip": 0}, {"refresh_every": -1}])
def test_invalid_recipe(kwargs):
    with pytest.raises(ValueError):
        learning.Recipe(**kwargs)
