"""Fabricated shared-head objective, chronology, accounting and closure contracts."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openjev.research import otto_action_focused_loss as losses
from openjev.research import otto_action_focused_metrics as metrics
from openjev.research import otto_prequery_data as data
from openjev.research import otto_prequery_loss as baseline_loss
from openjev.research import otto_prequery_scores as models
from openjev.research import otto_score_forecast_data as windows

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runner = module("_action_focused_training_fixture", "scripts/train_otto_action_focused.py")
old = module("_prequery_training_reference", "scripts/train_otto_prequery_calibration.py")


def history(lengths=(37, 5), *, positive=True):
    offsets = np.array([0, *np.cumsum(lengths)], np.int64)
    total = int(offsets[-1])
    features, scores, targets, prior_targets = (np.zeros((total, n), np.float32) for n in (31, 4, 4, 4))
    query_mask, prior_mask = (np.zeros(total, np.bool_) for _ in range(2))
    weights, prior_weights = (np.zeros(total, np.float64) for _ in range(2))
    legal = np.ones((total, 4), np.bool_)
    for index, length in enumerate(lengths):
        low = int(offsets[index]); steps = np.arange(length)
        features[low:low + length, 2:6] = 1
        features[low:low + length, 15] = steps / 2188
        features[low:low + length, 16] = steps % 4 / 2188
        features[low:low + length, 17] = 1
        features[low:low + length, 0] = (index + 1) / 10
        q = low + steps[steps % 4 == 0]
        query_mask[q] = True
        scores[q] = np.array([31., 35., 33., 32.], np.float32) + np.arange(len(q))[:, None] * np.array([.5, -.25, .125, .75], np.float32)
        p = q[1:]; prior_mask[p] = True; prior_targets[p] = scores[p]
        if len(p):
            prior_weights[p] = 1 / (54 * len(p))
        if positive:
            selected = low + steps[(steps >= 33) & (steps % 4 != 0)]
            targets[selected] = np.array([34., 30., 32., 37.], np.float32)
            weights[selected] = .025
    return {"version": data.VERSION, "features": features, "query_scores": scores,
        "targets": targets, "legal": legal, "actions": np.zeros(total, np.int64),
        "query_mask": query_mask, "weights": weights, "episode_offsets": offsets,
        "prior_targets": prior_targets, "prior_weights": prior_weights, "prior_mask": prior_mask,
        "episode_ids": tuple(f"e{i}" for i in range(len(lengths))),
        "episode_regimes": tuple("lambda3" for _ in lengths),
        "episode_splits": tuple("train" for _ in lengths), "counts": {"rows": total, "episodes": len(lengths)}}


def nonzero(model):
    with torch.no_grad():
        model.output.weight.copy_(torch.linspace(-.04, .06, model.output.weight.numel()).reshape_as(model.output.weight))
        if model.kind == "innovation":
            model.correction.weight.fill_(.03)
    return model


@pytest.mark.parametrize("kind", models.KINDS)
def test_aux_production_batch_exactly_preserves_original_parameters_gradients_and_adam(kind, monkeypatch):
    fixture = history()
    before = {k: v.tobytes() for k, v in fixture.items() if isinstance(v, np.ndarray)}
    new, reference = (nonzero(models.make_head(kind, 31)) for _ in range(2))
    opt, old_opt = (torch.optim.Adam(m.parameters(), lr=.003) for m in (new, reference))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("AUX optimization must not evaluate SPO")

    monkeypatch.setattr(losses.spo_base, "weighted_spo_plus_loss", forbidden)
    result = runner.batch_update(torch, models, data, losses, new, opt, fixture, [0, 1],
        objective="aux", check=lambda: None, stage=lambda *_: None)
    expected = old.batch_update(torch, models, data, baseline_loss, reference, old_opt, fixture, [0, 1],
        objective="query_aux", check=lambda: None, stage=lambda *_: None)
    for key in expected:
        assert result[key] == ("aux" if key == "objective" else expected[key])
    assert result["spo_loss"] == result["spo_loss_calls"] == result["spo_weighted_rows"] == 0
    assert result["backward_chunks"] == 2
    for a, b in zip(new.parameters(), reference.parameters(), strict=True):
        assert torch.equal(a, b) and torch.equal(a.grad, b.grad)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(opt.state[a][key], old_opt.state[b][key])
    assert all(fixture[k].tobytes() == value for k, value in before.items())


@pytest.mark.parametrize("kind", models.KINDS)
def test_spo_production_batch_matches_direct_formula_reference_and_fixed_parameters(kind):
    fixture = history()
    new, reference = (nonzero(models.make_head(kind, 71)) for _ in range(2))
    opt, old_opt = (torch.optim.Adam(m.parameters(), lr=.003) for m in (new, reference))
    initial = {k: p.detach().clone() for k, p in new.named_parameters()}
    calls = []
    original_forward = new.forward

    def record(*args, **kwargs):
        state = kwargs["carry"]
        assert state.hidden.grad_fn is None and not state.hidden.requires_grad
        assert all(torch.equal(p, initial[k]) for k, p in new.named_parameters())
        assert set(kwargs) == {"features", "query_scores", "lengths", "query_mask", "episode_ends", "carry"}
        assert torch.isnan(kwargs["query_scores"][~kwargs["query_mask"]]).all()
        calls.append((torch.is_grad_enabled(), kwargs["lengths"].tolist(), state.absolute_step.tolist()))
        return original_forward(*args, **kwargs)

    def direct(prediction, prior, targets, legal, weights, prior_targets, prior_weights, prior_mask, *, objective):
        terms = baseline_loss.weighted_loss(prediction, prior, targets, legal, weights,
            prior_targets, prior_weights, prior_mask, objective=objective)
        # Independently evaluate the literal paper formula in float64. Finite
        # masks are applied first; only selected nonqueries enter this oracle.
        mask = weights > 0
        if bool(mask.any()):
            c, z = targets[mask].detach().double() / 64, prediction[mask].double() / 64
            eligible = legal[mask]
            minimum = c.masked_fill(~eligible, float("inf")).min(-1, keepdim=True).values
            best = eligible & (c == minimum)
            oracle = best.double() / best.sum(-1, keepdim=True)
            rows = (c - 2 * z).masked_fill(~eligible, float("-inf")).max(-1).values
            rows = rows + 2 * (oracle * z).sum(-1) - minimum[:, 0]
            extra = (rows * weights[mask].detach().double()).sum() * (54 / prediction.shape[0])
        else:
            extra = prediction.sum() * 0
        return {**terms, "total": terms["total"] + extra}

    new.forward = record
    result = runner.batch_update(torch, models, data, losses, new, opt, fixture, [0, 1],
        objective="spo", check=lambda: None, stage=lambda *_: None)
    expected = old.batch_update(torch, models, data, SimpleNamespace(weighted_loss=direct), reference, old_opt,
        fixture, [0, 1], objective="query_aux", check=lambda: None, stage=lambda *_: None)
    assert calls == [(True, [32, 5], [0, 0]), (True, [5, 0], [32, 5])]
    assert result["backward_chunks"] == result["spo_loss_calls"] == 2
    assert result["spo_weighted_rows"] == 3 and result["prior_rows"] == 10
    assert result["loss"] == pytest.approx(expected["loss"], rel=2e-6, abs=1e-7)
    assert result["loss"] == pytest.approx(result["nonquery_loss"] + result["prior_loss"] + result["spo_loss"])
    for a, b in zip(new.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-7)
        assert opt.state[a]["step"] == old_opt.state[b]["step"] == 1


def test_query_only_batch_advances_all_adam_moments_after_real_gradient():
    model = nonzero(models.make_head("innovation", 3))
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    runner.batch_update(torch, models, data, losses, model, optimizer, history(), [0, 1],
        objective="spo", check=lambda: None, stage=lambda *_: None)
    before = {p: (optimizer.state[p]["exp_avg"].clone(), optimizer.state[p]["exp_avg_sq"].clone())
              for p in model.parameters()}
    result = runner.batch_update(torch, models, data, losses, model, optimizer, history((1, 1), positive=False),
        [0, 1], objective="spo", check=lambda: None, stage=lambda *_: None)
    assert result["backward_chunks"] == result["spo_loss_calls"] == result["loss"] == 0
    assert result["optimizer_step"] == 2
    for parameter in model.parameters():
        assert parameter.grad is not None and not torch.count_nonzero(parameter.grad)
        assert optimizer.state[parameter]["step"] == 2
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], before[parameter][0] * .9)
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg_sq"], before[parameter][1] * .999)


def test_final_rescore_labels_spo_diagnostic_separately_from_aux_objective():
    fixture = history((5,), positive=False)
    fixture["weights"][1] = .1
    fixture["prior_targets"][:] = 0; fixture["prior_weights"][4] = .2
    saved = {"predictions": np.zeros((5, 4), np.float32), "prior": np.zeros((5, 4), np.float32),
             "prior_mask": fixture["prior_mask"]}
    saved["predictions"][1] = [64., -64., 0., 0.]
    saved["prior"][4] = [128., -128., 0., 0.]
    aux = runner.rescore_losses(torch, losses, np, saved, fixture, "aux")
    spo = runner.rescore_losses(torch, losses, np, saved, fixture, "spo")
    assert aux["nonquery"] == spo["nonquery"] == pytest.approx(.05, abs=1e-8)
    assert aux["prior"] == spo["prior"] == pytest.approx(.4, abs=1e-8)
    assert aux["spo"] == spo["spo"] == pytest.approx(.2, abs=1e-8)
    assert aux["total"] == pytest.approx(.45, abs=2e-8)
    assert spo["total"] == pytest.approx(.65, abs=2e-8)


def test_valid_barrier_complete_prior_support_and_real_window_projection(tmp_path):
    identities = [r for r in runner.cohort() if r["stage"] == "valid"]
    fixture = history([9, 5, 1] * 12, positive=False)
    raw = np.arange(len(fixture["features"]) * 4, dtype=np.float32).reshape(-1, 4) / 4 + 20
    np.savez(tmp_path / "valid.npz", features=fixture["features"], raw_q=raw, legal=fixture["legal"],
        actions=fixture["actions"], correction=fixture["query_mask"], episode_offsets=fixture["episode_offsets"])
    run = runner.Run(SimpleNamespace(output=tmp_path))
    with pytest.raises(ValueError, match="before VALID"):
        run.validation()
    run.np, run.torch, run.data, run.models, run.metrics, run.windows = np, torch, data, models, metrics, windows
    run.collection_dir = tmp_path; run.collection_plan = {"cohort": identities}
    run.receipt["fits_completed"] = 11; run.valid_allowed = True; run.check = lambda: None
    with pytest.raises(ValueError, match="before VALID"):
        run.validation()
    run.receipt["fits_completed"] = 12
    chronological, grouped, episodes, returned = run.validation()
    assert returned == identities and all("actions" not in e for e in episodes)
    assert runner.window_predictions(np, grouped, chronological, raw).tobytes() == grouped["targets"].tobytes()
    assert not chronological["prior_weights"].any()
    expected = np.concatenate([(np.arange(t) % 4 == 0) & (np.arange(t) >= 4) for t in [9, 5, 1] * 12])
    assert np.array_equal(chronological["prior_mask"], expected)
    saved, work = run.predict(models.make_head("innovation", 3), chronological, "validation_forward")
    assert np.array_equal(saved["prior_mask"], expected) and work["prior_rows"] == 36
    assert not saved["prior"][~expected].any()
    assert run.receipt["validation_forward_rows"] == len(raw)


def test_return_publication_failure_retains_uncertain_optimizer_update(tmp_path, monkeypatch):
    run = runner.Run(SimpleNamespace(output=tmp_path)); run.check = lambda: None
    run.clock = SimpleNamespace(now_ns=lambda: 100)
    run.torch = run.models = run.data = run.losses = object()

    def event(record, filename):
        if record["event"] == "return":
            raise OSError("return fsync failed")

    def update(*_args, objective, stage, **_kwargs):
        assert objective == "aux"
        stage("optimizer_update", None)
        return {"forward_chunks": 2, "backward_chunks": 2, "forward_rows": 42, "optimizer_step": 1}

    run.event = event; monkeypatch.setattr(runner, "batch_update", update)
    with pytest.raises(OSError, match="return fsync"):
        run.batch(None, None, None, [0, 1], "innovation_aux", 11, 0, 0)
    assert run.receipt["optimizer_steps"] == 0 and run.receipt["pending"]["stage"] == "optimizer_update"


def test_four_cell_initialization_pairing_and_complete_payload_inventory():
    rows = []
    for cell in runner.KINDS:
        architecture, readout, objective = runner.CELLS[cell]
        model = models.make_head(architecture, 79)
        tensors = model.state_dict()
        full, core = hashlib.sha256(), hashlib.sha256()
        for name, value in tensors.items():
            raw = name.encode() + value.numpy().tobytes(); full.update(raw)
            if not name.startswith("correction."):
                core.update(raw)
        rows.append({"family": cell, "seed": 79, "architecture": architecture, "readout": readout,
            "objective": objective, "initial_sha256": full.hexdigest(), "initial_core_sha256": core.hexdigest(),
            "initial_tensors": {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in tensors.items()}})
        assert sum(p.numel() for p in model.parameters()) == {"innovation": 5978, "innovation_gru": 5996}[architecture]
    runner.check_paired_initialization(rows)
    corrupted = copy.deepcopy(rows); corrupted[1]["initial_tensors"]["output.bias"] = "different"
    with pytest.raises(ValueError, match="paired initialization"):
        runner.check_paired_initialization(corrupted)
    with pytest.raises(ValueError, match="complete single-seed"):
        runner.check_paired_initialization(rows[:-1])
    assert len(runner.PAYLOADS) == 49
    assert sum(name.startswith("training-prediction-") for name in runner.PAYLOADS) == 12
    assert sum(name.startswith("prediction-") for name in runner.PAYLOADS) == 13
    assert runner.CONFIG["required_conditions"] == 41
    assert runner.CONFIG["prior_coefficient"] == runner.CONFIG["spo_coefficient"] == 1
    assert runner.CONFIG["capacity_evidence_scope"] == runner.CAPACITY_SCOPE
    assert runner.CAPACITY == "scripts/qualify_otto_prequery_capacity.py"


def test_exact_reserved_cohort_matches_metadata_only_collector():
    collector = module("_action_focused_collector_metadata", "scripts/collect_otto_action_focused.py")
    actual = runner.cohort()
    assert actual == collector.cohort()
    assert len(actual) == 90
    for stage, starts, count in (("train", (291000001, 292000001), 9),
                                 ("valid", (293000001, 294000001), 6)):
        for regime, first in zip(("lambda3", "lambda4"), starts, strict=True):
            rows = [r for r in actual if r["stage"] == stage and r["regime"] == regime]
            assert len(rows) == 3 * count and {r["seed"] for r in rows} == set(range(first, first + count))
    assert runner.SEEDS == (295000001, 295000002, 295000003)
    assert runner.SELECTION_START == 296000001
