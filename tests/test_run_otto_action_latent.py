"""Fabricated runner loss, archive validation and paired forecast boundaries."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from openjev.research import otto_action_latent_model as models
from openjev.research import otto_action_latent_ridge as ridge

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("_test_action_latent_runner", SCRIPTS / "run_otto_action_latent.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def data(horizon=4):
    prefix = np.linspace(-.3, .7, 3 * 9 * 31, dtype=np.float32).reshape(3, 9, 31)
    outcomes = np.tile(np.arange(horizon, dtype=np.int64) % 4, (3, 1))
    outcomes[1, 1:] = 4
    outcomes[2] = 4
    alive = outcomes != 4
    costs = np.broadcast_to(np.array([64., -64., 128., -128.], np.float32), (3, horizon, 4)).copy()
    costs[~alive] = 0
    continuation = np.zeros((3, horizon, 31), np.float32)
    continuation[..., 19:21] = np.array([.3, .6], np.float32)
    continuation[~alive] = 0
    return {"prefix": prefix, "prefix_lengths": np.full(3, 9, np.int64),
            "actions": np.tile(np.arange(horizon, dtype=np.int64) % 4, (3, 1)),
            "continuation": continuation, "outcomes": outcomes, "raw_costs": costs,
            "legal": np.broadcast_to(alive[..., None], (3, horizon, 4)).copy(),
            "case_ids": np.array(["fabricated:a", "fabricated:b", "fabricated:c"]),
            "regimes": np.array(["lambda3"] * 3)}


def tensors(values):
    return {k: torch.from_numpy(v.copy()) for k, v in values.items() if k not in ("case_ids", "regimes")}


def prediction(horizon=4):
    return {"outcome_logits": torch.zeros(3, horizon, 5, requires_grad=True),
            "cost_contrasts": torch.zeros(3, horizon, 4, requires_grad=True),
            "aux_features": torch.zeros(3, horizon, 2, requires_grad=True)}


@pytest.mark.parametrize("horizon", (4, 8))
def test_fabricated_archive_roundtrip_owns_arrays_and_matches_fixed_horizon(tmp_path, horizon):
    source = data(horizon)
    snapshots = {k: v.tobytes() for k, v in source.items()}
    path = tmp_path / "fabricated.npz"
    runner.save_npz(path, np, **source)
    loaded = runner.load_data(path, np, horizon)
    assert set(loaded) == set(source)
    for name in source:
        assert loaded[name].dtype == source[name].dtype and loaded[name].shape == source[name].shape
        assert loaded[name].tobytes() == snapshots[name]
        assert not np.shares_memory(loaded[name], source[name])
    with pytest.raises(FileExistsError):
        runner.save_npz(path, np, **source)
    with pytest.raises(ValueError, match="schema"):
        runner.load_data(path, np, 8 if horizon == 4 else 4)


@pytest.mark.parametrize("defect", ("extra", "prefix_shape", "prefix_dtype", "prefix_nan", "duplicate_case",
                                  "short_prefix", "length_dtype", "action_dtype", "action_range", "outcome_range",
                                  "unabsorbed", "illegal_alive", "legal_found", "legal_dtype", "future_nan", "cost_nan"))
def test_malformed_fabricated_data_rejected_before_training(tmp_path, defect):
    source = data()
    if defect == "extra":
        source["extra"] = np.zeros(1)
    elif defect == "prefix_shape":
        source["prefix"] = source["prefix"][:, :8]
    elif defect == "prefix_dtype":
        source["prefix"] = source["prefix"].astype(np.float64)
    elif defect == "prefix_nan":
        source["prefix"][0, 0, 0] = np.nan
    elif defect == "duplicate_case":
        source["case_ids"][1] = source["case_ids"][0]
    elif defect == "short_prefix":
        source["prefix_lengths"][0] = 8
    elif defect == "length_dtype":
        source["prefix_lengths"] = source["prefix_lengths"].astype(np.float32)
    elif defect == "action_dtype":
        source["actions"] = source["actions"].astype(np.float32)
    elif defect == "action_range":
        source["actions"][0, 0] = 4
    elif defect == "outcome_range":
        source["outcomes"][0, 0] = 5
    elif defect == "unabsorbed":
        source["outcomes"][1, 2] = 0
    elif defect == "illegal_alive":
        source["legal"][0, 0] = False
    elif defect == "legal_found":
        source["legal"][2, 0, 0] = True
    elif defect == "legal_dtype":
        source["legal"] = source["legal"].astype(np.int64)
    elif defect == "future_nan":
        source["continuation"][0, 0, 19] = np.nan
    else:
        source["raw_costs"][0, 0, 0] = np.nan
    path = tmp_path / "invalid.npz"
    np.savez(path, **source)
    with pytest.raises(ValueError):
        runner.load_data(path, np, 4)


def test_loss_matches_equal_case_support_with_all_terminal_case_retained(monkeypatch):
    monkeypatch.setitem(runner.c.CONFIG, "cost_weight", 1.)
    monkeypatch.setitem(runner.c.CONFIG, "aux_weight", .1)
    target = tensors(data())
    estimate = prediction()
    actual = runner.loss_for(estimate, target, torch)
    # Every row contributes CE. q/aux average surviving rows within each case,
    # then divide by all3 cases, retaining the all-found case with zero weight.
    expected = math.log(5) + (2.5 + 2.5 + 0.) / 3 + .1 * (.225 + .225 + 0.) / 3
    assert actual.item() == pytest.approx(expected, rel=2e-7)
    actual.backward()
    alive = target["outcomes"] != 4
    assert not bool(estimate["cost_contrasts"].grad[~alive].any())
    assert not bool(estimate["aux_features"].grad[~alive].any())
    assert bool(estimate["outcome_logits"].grad[~alive].abs().sum() > 0)


def test_loss_cost_gauge_target_columns_and_masked_rows_do_not_change_objective():
    original = tensors(data())
    estimate = prediction()
    baseline = runner.loss_for(estimate, original, torch)
    changed = {name: value.clone() for name, value in original.items()}
    alive = changed["outcomes"] != 4
    changed["raw_costs"][alive] += 64 * 9
    changed["raw_costs"][~alive] = 100
    changed["continuation"][..., :19] = -17
    changed["continuation"][..., 21:] = 200
    changed["continuation"][~alive] = 99
    changed_prediction = {name: value.clone() for name, value in estimate.items()}
    changed_prediction["cost_contrasts"][~alive] = 44
    changed_prediction["aux_features"][~alive] = -31
    actual = runner.loss_for(changed_prediction, changed, torch)
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)


def test_explicit_cost_scale_changes_only_cost_weighting(monkeypatch):
    monkeypatch.setitem(runner.c.CONFIG, "cost_weight", 1.)
    monkeypatch.setitem(runner.c.CONFIG, "aux_weight", 0.)
    target = tensors(data())
    value = runner.loss_for(prediction(), target, torch, .5)
    assert value.item() == pytest.approx(math.log(5) + (5 / 3) / .25, rel=2e-7)
    # Standardized coordinates preserve the dimensionless loss and gradient.
    standardized = prediction()
    first = runner.loss_for(standardized, target, torch, 1.)
    scaled_target = {name: value.clone() for name, value in target.items()}
    scaled_target["raw_costs"] *= .01
    second_prediction = dict(standardized)
    second_prediction["cost_contrasts"] = standardized["cost_contrasts"] * .01
    second = runner.loss_for(second_prediction, scaled_target, torch, .01)
    torch.testing.assert_close(first, second, rtol=2e-7, atol=1e-7)
    grad_first = torch.autograd.grad(first, standardized["cost_contrasts"], retain_graph=True)[0]
    grad_second = torch.autograd.grad(second, standardized["cost_contrasts"])[0]
    torch.testing.assert_close(grad_first, grad_second, rtol=2e-7, atol=1e-7)


def test_train_cost_scale_uses_same_equal_case_centered_support_and_fixed_floor():
    source = data()
    snapshot = {name: value.tobytes() for name, value in source.items()}
    scale, variance = runner.train_cost_scale(source, np, return_variance=True)
    assert variance == pytest.approx(5 / 3, abs=1e-15)
    assert scale == float(np.float32(math.sqrt(5 / 3)))
    assert all(source[name].tobytes() == value for name, value in snapshot.items())
    changed = {name: value.copy() for name, value in source.items()}
    changed["raw_costs"] += 512
    changed["raw_costs"][changed["outcomes"] == 4] = [1024, -512, 64, -128]
    assert runner.train_cost_scale(changed, np, return_variance=True) == (scale, variance)
    changed["raw_costs"][:] = 32
    floor_scale, zero_variance = runner.train_cost_scale(changed, np, return_variance=True)
    assert zero_variance == 0.
    assert floor_scale == float(np.float32(.001))


@pytest.mark.parametrize("kind", models.KINDS)
def test_paired_panel_same_initial_forecast_current_labels_only_after_prediction(kind):
    target = tensors(data())
    model = models.make_model(kind, 73, cost_scale=.02)
    blind = model.blind_rollout(target["prefix"], target["prefix_lengths"], target["actions"])
    normal = model.normal_rollout(target["prefix"], target["prefix_lengths"], target["actions"],
                                  target["continuation"], found=target["outcomes"] == 4)
    torch.testing.assert_close(blind["outcome_logits"][:, 0], normal["outcome_logits"][:, 0], rtol=0, atol=0)
    changed = target["continuation"].clone()
    changed[:, 2] += .2
    other = model.normal_rollout(target["prefix"], target["prefix_lengths"], target["actions"],
                                 changed, found=target["outcomes"] == 4)
    torch.testing.assert_close(normal["outcome_logits"][:, :3], other["outcome_logits"][:, :3], rtol=0, atol=0)
    assert not torch.equal(normal["prior_states"][0, 3], other["prior_states"][0, 3])
    combined = .5 * (runner.loss_for(blind, target, torch, .02) + runner.loss_for(normal, target, torch, .02))
    assert torch.isfinite(combined)
    combined.backward()
    assert model.cost_head.weight.grad is not None and bool(model.cost_head.weight.grad.abs().sum() > 0)


def test_ridge_reference_retains_prefix_history_and_freezes_whole_terminal_row():
    values = data(8)
    p, lengths, actions = (values[name] for name in ("prefix", "prefix_lengths", "actions"))
    blind = ridge.design(p, lengths, actions)
    assert blind.shape == (3, 8, 336)
    changed = p.copy()
    changed[:, 0, 0] += 1
    assert not np.array_equal(blind, ridge.design(changed, lengths, actions))
    normal = ridge.design(p, lengths, actions, values["continuation"], values["outcomes"] == 4)
    np.testing.assert_array_equal(normal[1, 1], normal[1, 2])
    np.testing.assert_array_equal(normal[1, 1], normal[1, 7])
    np.testing.assert_array_equal(normal[2], np.broadcast_to(normal[2, 0], normal[2].shape))
    altered_actions = actions.copy()
    altered_actions[:, 4:] = (altered_actions[:, 4:] + 2) % 4
    np.testing.assert_array_equal(blind[:, :4], ridge.design(p, lengths, altered_actions)[:, :4])
