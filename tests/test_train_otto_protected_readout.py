"""Fabricated staged forks, frozen gradients, canonical evaluation and VALID barrier."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openjev.research import otto_action_focused_loss as losses
from openjev.research import otto_prequery_data as data
from openjev.research import otto_prequery_scores as pretrain
from openjev.research import otto_protected_readout_metrics as metrics
from openjev.research import otto_protected_training_model as models

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runner = module("_protected_training_fixture", "scripts/train_otto_protected_readout.py")
old = module("_protected_original_batch_fixture", "scripts/train_otto_action_focused.py")


def history(lengths=(37, 5), *, positive=True):
    """Only constructed values, with unequal true tails and later-query targets."""
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


def backbone(seed=31):
    model = pretrain.make_head("innovation_gru", seed)
    with torch.no_grad():
        model.output.weight.copy_(torch.linspace(-.04, .06, model.output.weight.numel()).reshape_as(model.output.weight))
    return model


def optimizer(model):
    return torch.optim.Adam([p for _, p in runner.trainable_parameters(model)], lr=.003, weight_decay=0.)


def update(model, opt, fixture, objective="aux"):
    return runner.batch_update(torch, models, data, losses, model, opt, fixture, [0, 1],
        objective=objective, check=lambda: None, stage=lambda *_: None)


def predictor(tmp_path):
    run = runner.Run(SimpleNamespace(output=tmp_path))
    run.np, run.torch, run.models, run.pretrain_models = np, torch, models, pretrain
    run.data, run.metrics, run.losses = data, metrics, losses
    run.check = lambda: None
    return run


def test_pretraining_aux_batch_exactly_retains_original_loss_parameters_gradients_and_adam():
    fixture = history()
    immutable = {k: v.tobytes() for k, v in fixture.items() if isinstance(v, np.ndarray)}
    model, reference = backbone(), backbone()
    opt, refopt = optimizer(model), optimizer(reference)
    result = update(model, opt, fixture)
    expected = old.batch_update(torch, pretrain, data, losses, reference, refopt, fixture, [0, 1],
        objective="aux", check=lambda: None, stage=lambda *_: None)
    assert all(result[k] == value for k, value in expected.items())
    assert result["loss_chunks"] == result["differentiable_chunks"] == result["backward_chunks"] == 2
    assert result["skipped_backward_chunks"] == 0 and result["trainable_parameter_count"] == 5996
    for value, ref in zip(model.parameters(), reference.parameters(), strict=True):
        assert torch.equal(value, ref) and torch.equal(value.grad, ref.grad)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(opt.state[value][key], refopt.state[ref][key])
    assert all(fixture[k].tobytes() == value for k, value in immutable.items())


@pytest.mark.parametrize("objective", ("aux", "spo"))
def test_frozen_prior_only_chunks_record_constant_loss_without_backward_or_backbone_adam(objective):
    model = models.from_pretrained("frozen", 31, backbone().state_dict())
    initial = runner.state_witness(model)
    opt = optimizer(model)
    result = update(model, opt, history(positive=False), objective)
    assert result["forward_chunks"] == result["loss_chunks"] == result["no_grad_chunks"] == 2
    assert result["skipped_backward_chunks"] == 2 and result["backward_chunks"] == result["differentiable_chunks"] == 0
    assert result["prior_loss"] > 0 and result["loss"] == result["prior_loss"]
    assert result["spo_loss"] == result["nonquery_loss"] == 0 and result["prior_rows"] == 10
    assert result["optimizer_step"] == 1 and result["trainable_parameter_count"] == 116
    assert runner.state_witness(model) == initial and result["frozen_backbone_unchanged"] is True
    for name, parameter in model.named_parameters():
        if name.startswith("action_residual."):
            assert parameter.grad is not None and not torch.count_nonzero(parameter.grad)
            assert opt.state[parameter]["step"] == 1
        else:
            assert parameter.grad is None and parameter not in opt.state


@pytest.mark.parametrize("objective", ("aux", "spo"))
def test_frozen_action_learning_preserves_base_priors_and_carry_while_changing_actions(tmp_path, objective):
    parent = backbone()
    model = models.from_pretrained("frozen", 31, parent.state_dict())
    fixture = history()
    result = update(model, optimizer(model), fixture, objective)
    assert result["backward_chunks"] == result["differentiable_chunks"] == 1
    assert result["skipped_backward_chunks"] == 1 and result["no_grad_chunks"] == 1
    assert result["loss_chunks"] == 2 and result["spo_weighted_rows"] == (3 if objective == "spo" else 0)
    assert runner.state_witness(model, backbone_only=True) == runner.state_witness(parent)
    assert any(torch.count_nonzero(p) for name, p in model.named_parameters() if name.startswith("action_residual."))
    ref, _ = runner.canonical_clone(pretrain, models, parent, "pretrained", 31)
    canonical, _ = runner.canonical_clone(pretrain, models, model, "frozen_" + objective, 31)
    run = predictor(tmp_path)
    expected, _ = run.predict(ref, fixture, "training_rescore")
    observed, _ = run.predict(canonical, fixture, "training_rescore")
    assert runner.same_frozen_predictions(np, observed, expected)
    assert not np.array_equal(observed["predictions"], expected["predictions"])
    packet = data.batch_chunk(fixture, [0, 1], 0)
    inputs = {k: torch.from_numpy(v) for k, v in packet["model_inputs"].items()}
    with torch.no_grad():
        a = canonical(**inputs, carry=canonical.initial_carry(2))
        b = ref(**inputs, carry=ref.initial_carry(2))
    for name in ("hidden", "raw_anchor", "has_query", "absolute_step", "ended"):
        assert torch.equal(getattr(a.carry, name), getattr(b.carry, name))


def test_joint_adaptation_trains_backbone_and_residual_and_counts_both_chunks():
    model = models.from_pretrained("joint", 31, backbone().state_dict())
    before = runner.state_witness(model, backbone_only=True)
    opt = optimizer(model)
    result = update(model, opt, history(), "spo")
    assert result["trainable_parameter_count"] == 6112
    assert result["backward_chunks"] == result["differentiable_chunks"] == result["loss_chunks"] == 2
    assert result["no_grad_chunks"] == result["skipped_backward_chunks"] == 0
    assert runner.state_witness(model, backbone_only=True) != before
    assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert len(opt.state) == 8 and all(opt.state[p]["step"] == 1 for p in model.parameters())


def test_query_only_batch_after_learning_advances_only_trainable_adam_moments():
    model = models.from_pretrained("frozen", 31, backbone().state_dict())
    opt = optimizer(model)
    update(model, opt, history(), "spo")
    before = {p: (opt.state[p]["exp_avg"].clone(), opt.state[p]["exp_avg_sq"].clone())
              for _, p in runner.trainable_parameters(model)}
    result = update(model, opt, history((1, 1), positive=False), "spo")
    assert result["loss"] == result["backward_chunks"] == result["loss_chunks"] == result["skipped_backward_chunks"] == 0
    assert result["no_grad_chunks"] == 1 and result["optimizer_step"] == 2
    for parameter, (mean, square) in before.items():
        assert not torch.count_nonzero(parameter.grad)
        torch.testing.assert_close(opt.state[parameter]["exp_avg"], mean * .9)
        torch.testing.assert_close(opt.state[parameter]["exp_avg_sq"], square * .999)
    assert all(p not in opt.state for p in model.parameters() if not p.requires_grad)


def test_optimizer_rejects_frozen_backbone_members_even_before_forward():
    model = models.from_pretrained("frozen", 31, backbone().state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    with pytest.raises(ValueError, match="exactly the trainable parameters"):
        update(model, opt, history())
    assert len(opt.state) == 0


def test_four_forks_have_zero_residual_independent_storage_and_no_shared_optimizer():
    parent = backbone()
    before = runner.state_witness(parent)
    forks = [models.from_pretrained(kind.split("_")[0], 31, parent.state_dict()) for kind in runner.BRANCHES]
    optimizers = [optimizer(model) for model in forks]
    assert len({id(opt) for opt in optimizers}) == 4
    assert all(len(opt.state) == 0 for opt in optimizers)
    assert all(runner.state_witness(model) == runner.state_witness(forks[0]) for model in forks)
    for name in parent.state_dict():
        assert len({m.state_dict()[name].data_ptr() for m in [parent, *forks]}) == 5
    with torch.no_grad():
        forks[0].action_residual.bias.fill_(1.)
    assert all(not torch.count_nonzero(m.action_residual.bias) for m in forks[1:])
    assert runner.state_witness(parent) == before


@pytest.mark.parametrize("kind", runner.KINDS)
def test_canonical_clone_uses_matched_flags_exact_weights_and_no_grad_without_changing_training_model(tmp_path, kind):
    source = backbone() if kind == "pretrained" else models.from_pretrained(kind.split("_")[0], 31, backbone().state_dict())
    original_flags = [p.requires_grad for p in source.parameters()]
    clone, metadata = runner.canonical_clone(pretrain, models, source, kind, 31)
    assert metadata == metrics.evaluation_metadata(kind)
    assert [p.requires_grad for p in source.parameters()] == original_flags
    assert all(p.requires_grad == name.startswith("action_residual.") for name, p in clone.named_parameters())
    observed = []
    method = clone.forward

    def record(*args, **kwargs):
        observed.append(torch.is_grad_enabled())
        assert kwargs["carry"].hidden.grad_fn is None
        assert torch.isnan(kwargs["query_scores"][~kwargs["query_mask"]]).all()
        return method(*args, **kwargs)

    clone.forward = record
    run = predictor(tmp_path)
    saved, work = run.predict(clone, history(), "validation_forward")
    assert observed == [False, False] and work["forward_rows"] == 42
    assert set(saved) == {"predictions", "base_predictions", "prior", "prior_mask"}
    assert run.receipt["validation_forward_rows"] == 42
    assert runner.state_witness(clone) == runner.state_witness(source)
    assert all(p.grad is None for p in clone.parameters())


@pytest.mark.parametrize("key", ("base_predictions", "prior", "prior_mask"))
def test_frozen_equality_rejects_changed_canonical_reference_bytes(key):
    expected = {"predictions": np.zeros((2, 4), np.float32), "prior": np.zeros((2, 4), np.float32),
                "prior_mask": np.zeros(2, np.bool_)}
    observed = {**copy.deepcopy(expected), "base_predictions": expected["predictions"].copy()}
    observed["predictions"][:] = 9  # Protected actions are permitted to differ.
    assert runner.same_frozen_predictions(np, observed, expected)
    observed[key].flat[0] = 1
    with pytest.raises(ValueError, match="frozen canonical pretrained equality"):
        runner.same_frozen_predictions(np, observed, expected)


def test_valid_decode_is_rejected_before_fifteenth_completed_checkpoint_even_when_flag_is_set(tmp_path):
    run = runner.Run(SimpleNamespace(output=tmp_path))
    run.np = SimpleNamespace(load=lambda *_a, **_k: pytest.fail("VALID was decoded before barrier"))
    for allowed, completed in ((False, 15), (True, 14), (False, 0)):
        run.valid_allowed = allowed; run.receipt["fits_completed"] = completed
        with pytest.raises(ValueError, match="all fifteen final checkpoints"):
            run.validation()


def test_all_stage_exposures_fresh_optimizer_orders_checkpoints_and_durable_validation_barrier(tmp_path, monkeypatch):
    """Exercise the entire fixed schedule with mocked updates, not a training run.

    Real numerical gradient behavior is covered by the batch tests above. This
    test instead checks ownership and every scheduling/publication boundary.
    """
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    run = predictor(tmp_path)
    ticks = iter(range(0, 10**8, 1000))
    run.clock = SimpleNamespace(now_ns=lambda: next(ticks))
    events, owners, calls = [], [], {}

    def event(record, filename="progress.jsonl"):
        if record["event"] == "all_checkpoints_closed_before_VALID":
            assert not run.valid_allowed
            assert len(record["checkpoints"]) == 15
            assert all((tmp_path / name).is_file() for name in record["checkpoints"])
        events.append((filename, record))

    run.event = event

    def fabricated_update(_torch, _models, _data, _losses, model, opt, fixture, indices, *, objective, check, stage):
        if id(opt) not in calls:
            assert not opt.state
            owners.append((model, opt))
            calls[id(opt)] = 0
        calls[id(opt)] += 1
        stage("optimizer_update", None)
        return {"forward_chunks": 1, "backward_chunks": 0, "no_grad_chunks": 1, "forward_rows": len(indices),
            "loss_chunks": 0, "differentiable_chunks": 0, "skipped_backward_chunks": 0,
            "spo_loss_calls": 0, "spo_weighted_rows": 0, "optimizer_step": calls[id(opt)]}

    monkeypatch.setattr(runner, "batch_update", fabricated_update)
    fixture = history((1,) * 54, positive=False)
    fits = []
    for seed in runner.SEEDS:
        parent, fit, saved = run.fit("pretrained", seed, fixture)
        fits.append(fit)
        for kind in runner.BRANCHES:
            _, branch, _ = run.fit(kind, seed, fixture, backbone_state=parent.state_dict(),
                pretrained_fit=fit, pretrained_saved=saved)
            fits.append(branch)
        panel = fits[-5:]
        runner.check_paired_initialization(panel)
        assert panel[0]["episode_exposures"] == 80 * 54 and panel[0]["steps"] == 720
        assert all(row["episode_exposures"] == 40 * 54 and row["steps"] == 360 for row in panel[1:])
        assert all(row["episode_orders"] == panel[0]["episode_orders"][:40] for row in panel[1:])
        assert all(sorted(order) == list(range(54)) for row in panel for order in row["episode_orders"])
        assert all(row["evaluation"] == metrics.evaluation_metadata(row["family"]) for row in panel)
        assert [row["trainable_parameter_count"] for row in panel] == [5996, 116, 116, 6112, 6112]
        assert all(row["train_rescore"]["forward_rows"] == 54 for row in panel)
    assert len(owners) == len({id(model) for model, _ in owners}) == len({id(opt) for _, opt in owners}) == 15
    assert run.receipt["optimizer_steps"] == 6480 and run.receipt["fits_completed"] == 15
    assert run.receipt["training_forward_rows"] == 6480 * 6
    assert len([record for filename, record in events if filename == "work.jsonl"]) == 12960
    assert len([r for _, r in events if r["event"] == "epoch"]) == 36
    assert len([r for _, r in events if r["event"] == "fit_complete"]) == 15
    with pytest.raises(ValueError, match="all fifteen completed stages"):
        run.close_training(fits[:-1])
    assert not run.valid_allowed
    broken = copy.deepcopy(fits); broken[1]["initial_residual_zero"] = False
    with pytest.raises(ValueError, match="exact fresh checkpoint fork"):
        run.close_training(broken)
    assert not run.valid_allowed
    last = tmp_path / fits[-1]["checkpoint_path"]
    original_bytes = last.read_bytes(); last.write_bytes(b"fabricated checkpoint corruption")
    with pytest.raises(ValueError, match="durable final checkpoint barrier"):
        run.close_training(fits)
    assert not run.valid_allowed
    last.write_bytes(original_bytes)
    run.close_training(fits)
    assert run.valid_allowed
    assert events[-1][1]["event"] == "all_checkpoints_closed_before_VALID"
    assert len(runner.PAYLOADS) == 58
    assert sum(name.startswith("training-prediction-") for name in runner.PAYLOADS) == 15
    assert sum(name.startswith("prediction-") for name in runner.PAYLOADS) == 16


def test_failed_return_publication_retains_uncertain_optimizer_attempt(tmp_path, monkeypatch):
    run = runner.Run(SimpleNamespace(output=tmp_path)); run.check = lambda: None
    run.clock = SimpleNamespace(now_ns=lambda: 100)
    run.torch = run.models = run.data = run.losses = object()

    def event(record, filename):
        if record["event"] == "return":
            raise OSError("return fsync failed")

    def attempted(*_args, stage, **_kwargs):
        stage("optimizer_update", None)
        return {"optimizer_step": 1}

    run.event = event; monkeypatch.setattr(runner, "batch_update", attempted)
    with pytest.raises(OSError, match="return fsync failed"):
        run.batch(None, None, None, [0, 1], "frozen_aux", 301000001, 0, 0)
    assert run.receipt["optimizer_steps"] == 0 and run.receipt["pending"]["stage"] == "optimizer_update"


def test_body_preserves_fifteen_fit_order_barrier_and_complete_saved_prediction_schema(tmp_path, monkeypatch):
    """Fake fitting and prediction isolate terminal orchestration from learning."""
    run = predictor(tmp_path)
    ticks = iter(range(10000))
    run.clock = SimpleNamespace(now_ns=lambda: next(ticks))
    for name in ("set_num_threads", "set_num_interop_threads", "use_deterministic_algorithms"):
        monkeypatch.setattr(torch, name, lambda *_: None)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    records, files, published, fit_order = [], {}, {}, []
    output = {"predictions": np.zeros((1, 4), np.float32), "base_predictions": np.zeros((1, 4), np.float32),
              "prior": np.zeros((1, 4), np.float32), "prior_mask": np.zeros(1, np.bool_)}
    support = {"query_scores": np.zeros((1, 4), np.float32), "valid_mask": np.ones((1, 4), np.bool_),
               "counts": {"fixture": True}}

    def criteria(values, hold, *, technical_complete):
        assert not technical_complete and len(values) == 15
        records.extend(values)
        return [{"name": str(i), "passes": False} for i in range(29)]

    metric_provider = SimpleNamespace(FAMILIES=runner.KINDS, SEEDS=runner.SEEDS, criteria=criteria,
        prior_metrics=lambda *_: {"fixture_prior": True}, gate_decisions=lambda _: {"fixture_gate": False})
    providers = {runner.MODELS: models, runner.PRETRAIN_MODELS: pretrain, runner.DATA: data,
                 runner.LOSS: losses, runner.WINDOWS: object(), runner.METRICS: metric_provider}
    monkeypatch.setattr(runner, "load", lambda path, _name: providers[str(path.relative_to(runner.ROOT))])
    monkeypatch.setattr(runner, "runtime_record", lambda: {"fixture": True})
    monkeypatch.setattr(runner, "metric_report", lambda *_: {"fixture_metric": True})
    monkeypatch.setattr(runner, "window_predictions", lambda *_: np.zeros((1, 4, 4), np.float32))
    run.training = lambda: {"counts": {"fixture_train": True}}
    run.publish = lambda name, value: published.__setitem__(name, value)
    run.npz = lambda name, values: files.__setitem__(name, values)
    run.event = lambda value: None

    def fit(kind, seed, fixture, *, backbone_state=None, pretrained_fit=None, pretrained_saved=None):
        assert not run.valid_allowed
        fit_order.append((kind, seed))
        if kind == "pretrained":
            assert backbone_state is pretrained_fit is pretrained_saved is None
            model = backbone(seed)
        else:
            assert pretrained_fit["family"] == "pretrained" and pretrained_fit["seed"] == seed
            assert pretrained_saved is output
            model = models.from_pretrained(kind.split("_")[0], seed, backbone_state)
        canonical, evaluation = runner.canonical_clone(pretrain, models, model, kind, seed)
        architecture, readout, objective = runner.CELLS[kind]
        row = {"family": kind, "seed": seed, "architecture": architecture, "readout": readout,
               "objective": objective, "evaluation": evaluation,
               "final_backbone_sha256": runner.state_witness(model, backbone_only=True)[0]}
        return canonical, row, output

    def close(fits):
        assert fit_order == [(kind, seed) for seed in runner.SEEDS for kind in runner.KINDS]
        assert published["fits.json"]["fits"] == fits and len(fits) == 15
        run.valid_allowed = True

    def validation():
        assert run.valid_allowed
        return {"counts": {"fixture_valid": True}}, support, [], []

    run.fit, run.close_training, run.validation = fit, close, validation
    run.predict = lambda *_: (output, {"forward_rows": 1})
    run.body()
    assert len(files) == 16 and set(files["prediction-hold.npz"]) == {"predictions"}
    for name, values in files.items():
        if name != "prediction-hold.npz":
            assert set(values) == {"predictions", "base_predictions", "prior", "prior_mask"}
    assert [(r["family"], r["seed"]) for r in records] == fit_order
    assert all(r["evaluation"] == metrics.evaluation_metadata(r["family"]) for r in records)
    assert all(r["base_equals_pretrained"] is (True if r["family"].startswith("frozen_") else None) for r in records)
    summary = published["summary.json"]
    assert summary["required_conditions"] == 29 and summary["scientific_conditions"] == 28
    assert summary["technical_complete_pending_saved_audit"] is True
    assert summary["requires_successful_original_supervisor_and_saved_audit"] is True
    assert summary["capacity_evidence_scope"] == runner.CAPACITY_SCOPE
