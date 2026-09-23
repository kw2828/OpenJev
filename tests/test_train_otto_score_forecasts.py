"""Fabricated training contracts; no saved experiments, teacher or environments."""
from __future__ import annotations

import copy
import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from openjev.research import otto_score_forecast_data as data
from openjev.research.otto_recurrent_scores import make_head

spec = importlib.util.spec_from_file_location("_test_forecast_train", Path(__file__).parents[1] / "scripts/train_otto_score_forecasts.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def metric(agreement=.75, gap=2.):
    group = {"episode_weighted_agreement": agreement, "episode_weighted_raw_gap": gap,
             "by_age": {str(age): {"episode_weighted_raw_gap": gap} for age in (1, 2, 3)}}
    return {"by_regime": {regime: copy.deepcopy(group) for regime in ("lambda3", "lambda4")}}


def inputs():
    models = [{"family": kind, "seed": seed, "metrics": metric(gap=1. if kind == "residual_gru" else 2.)}
              for kind in runner.KINDS for seed in runner.SEEDS]
    return models, metric(), {regime: {str(age): 4 for age in (1, 2, 3)} for regime in ("lambda3", "lambda4")}


def episode(name, length, q):
    x = np.zeros((length, 31), np.float32)
    x[:, 15] = np.arange(length) / 2188
    x[:, 16] = (np.arange(length) % 4) / 2188
    x[:, 17] = 1
    return {"id": name, "regime": "lambda3", "split": "valid", "features": x,
            "teacher_scores": np.tile(np.asarray(q, np.float32), (length, 1)),
            "legal": np.ones((length, 4), np.bool_)}


def test_exact_forty_five_rules_and_one_bad_age_is_not_averaged_away():
    models, hold, support = inputs()
    rules = runner.criteria(models, hold, support)
    assert len(rules) == len({r["name"] for r in rules}) == 45
    assert all(r["passes"] for r in rules)
    models[0]["metrics"]["by_regime"]["lambda4"]["by_age"]["3"]["episode_weighted_raw_gap"] = 2.1
    failures = [r["name"] for r in runner.criteria(models, hold, support) if not r["passes"]]
    assert failures == ["lambda4.225001.age3.gap_vs_hold"]


def test_zero_gap_reference_requires_exact_zero_and_current_mlp_cannot_replace_candidate():
    models, hold, support = inputs()
    for group in hold["by_regime"].values():
        group["episode_weighted_raw_gap"] = 0.
    for row in models:
        if row["family"] == "current_mlp":
            row["metrics"] = metric(1., 0.)
    rules = runner.criteria(models, hold, support)
    failed = [r for r in rules if not r["passes"]]
    assert len(failed) == 6 and all(r["name"].endswith(".gap_vs_hold") for r in failed)
    models[0]["metrics"]["by_regime"]["lambda3"]["episode_weighted_raw_gap"] = np.nextafter(0., 1.)
    rule = next(r for r in runner.criteria(models, hold, support) if r["name"] == "lambda3.225001.gap_vs_hold")
    assert not rule["passes"] and rule["threshold"] == 0.


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "substitute", "nonfinite"])
def test_invalid_fit_inventory_or_metric_rejected(mutation):
    models, hold, support = inputs()
    if mutation == "missing":
        models.pop()
    elif mutation == "duplicate":
        models.append(copy.deepcopy(models[0]))
    elif mutation == "substitute":
        models[-1]["seed"] = 999
    else:
        models[0]["metrics"]["by_regime"]["lambda3"]["episode_weighted_raw_gap"] = float("nan")
    with pytest.raises(ValueError):
        runner.criteria(models, hold, support)


def test_eligible_centered_loss_ignores_illegal_scores_and_common_offset():
    target = torch.tensor([[[1., 3., -999., 999.]]])
    prediction = torch.tensor([[[102., 106., 9999., -9999.]]], requires_grad=True)
    legal = torch.tensor([[[True, True, False, False]]])
    loss = runner.centered_loss(torch, prediction, target, legal)
    assert loss.dtype == torch.float32 and float(loss.detach()) == 1. / 4096
    loss.sum().backward()
    assert torch.equal(prediction.grad[..., 2:], torch.zeros(1, 1, 2))
    assert float(prediction.grad.sum()) == 0.
    assert torch.equal(loss, runner.centered_loss(torch, prediction + 128, target, legal))


@pytest.mark.parametrize("kind", runner.KINDS)
def test_query_only_batch_has_zero_parameter_gradients_and_defined_adam_step(kind):
    model = make_head(kind, 7)
    x, anchor, lengths = torch.zeros(2, 4, 31), torch.ones(2, 4), torch.ones(2, dtype=torch.int64)
    prediction = model(x, anchor, lengths)
    assert not prediction.requires_grad
    legal = torch.zeros(2, 4, 4, dtype=torch.bool); legal[:, 0] = True
    loss = runner.batch_loss(torch, model, prediction, prediction, legal, torch.zeros(2, 4), 3.)
    loss.backward()
    assert loss.dtype == torch.float32 and float(loss.detach()) == 0.
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.step()
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    assert all(int(v["step"]) == 1 for v in optimizer.state.values())


def test_tensor_conversion_keeps_full_windows_and_uses_float32_training_weights(tmp_path):
    run = runner.Run(types.SimpleNamespace(output=tmp_path))
    run.np, run.torch = np, torch
    w = data.build_windows([episode("short", 1, [0, 1, 2, 3]), episode("long", 7, [0, 1, 2, 3])])
    t = run.tensors(w)
    assert len(t["lengths"]) == 3 and t["lengths"].tolist() == [1, 4, 3]
    assert t["nonquery_weights"].dtype == torch.float32
    assert torch.equal(t["nonquery_weights"], torch.tensor(w["nonquery_weights"].copy(), dtype=torch.float32))
    assert t["lengths"].dtype == torch.int64 and t["legal"].dtype == torch.bool


def test_validation_arrays_are_not_opened_before_all_checkpoints(tmp_path):
    run = runner.Run(types.SimpleNamespace(output=tmp_path))
    # No NumPy runtime or directory is supplied, so lifecycle rejection must
    # precede any attempt to decode an array.
    run.valid_allowed = False
    for count in (0, 11, 12):
        run.receipt["fits_completed"] = count
        with pytest.raises(ValueError, match="before VALID"):
            run.episodes("valid")
    with pytest.raises(ValueError, match="fixed episode stage"):
        run.episodes("test")


def test_collector_metrics_rebuild_episode_weights_and_preserve_prediction_order():
    arms = ["neural", "analytic", "period4_hold", "analytic"]
    rows = [episode(str(i), n, [0, 2, 3, 4]) for i, n in enumerate((4, 2, 7, 1))]
    identities = [{"episode_id": row["id"], "arm": arm} for row, arm in zip(rows, arms, strict=True)]
    w = data.build_windows(rows)
    prediction = w["targets"].copy()
    prediction[w["episode_index"] == 1, 1] = [9, 0, 2, 3]
    before = prediction.tobytes()
    result = runner.metric_report(data, np, w, rows, identities, prediction)
    analytic = result["by_collector"]["analytic"]["overall"]
    assert analytic["episodes"] == 2 and analytic["weight_mass"] == .5
    assert analytic["episode_weighted_raw_gap"] == 1.
    assert analytic["zero_support_episode_ids"] == ["3"]
    assert result["overall"]["episode_weighted_raw_gap"] == .5
    assert result["by_collector"]["neural"]["overall"]["episode_weighted_raw_gap"] == 0.
    assert prediction.tobytes() == before


def test_secondary_publication_failure_preserves_original_error(monkeypatch, tmp_path):
    run = runner.Run(types.SimpleNamespace(output=tmp_path / "run"))
    original = RuntimeError("original optimizer failure")
    def fail():
        raise original
    def failed_write(*args):
        raise OSError("disk failure")
    monkeypatch.setattr(run, "admit", fail)
    monkeypatch.setattr(runner, "write", failed_write)
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    with pytest.raises(RuntimeError, match="original optimizer") as caught:
        run.execute()
    assert caught.value is original and "disk failure" in caught.value.__notes__[0]
