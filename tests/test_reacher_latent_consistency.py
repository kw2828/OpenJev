"""Synthetic engineering fixtures only; no collected data or optimizer runs."""

import copy

import pytest
import torch
from torch import nn

from openjev.research.reacher_latent_consistency import (
    LatentConsistencyAuxiliary,
    collapse_diagnostics,
    vicreg_terms,
)
from openjev.research.reacher_reward_residual import GRUResidualRewardWorldModel
from openjev.research.reacher_world_models import GaussianRSSM, GRUWorldModel


@pytest.fixture(autouse=True)
def single_thread_seed():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(135)
    yield
    torch.set_num_threads(threads)


def sequence(batch=2, steps=6):
    packets = torch.zeros(batch, steps + 1, 8)
    packets[..., :4] = torch.randn(batch, steps + 1, 4)
    packets[..., 4:6] = torch.tensor([0.1, -0.2])
    packets[..., 6] = 1
    commands = torch.randn(batch, steps, 2).clamp(-1, 1)
    return packets, commands


class AdditiveGRU(GRUWorldModel):
    """A hand-computable causal model that deliberately trusts supplied packets."""

    def __init__(self):
        super().__init__(hidden_size=2)
        self.gain = nn.Parameter(torch.tensor(1.0))

    def assimilate(self, state, packet):
        return {"packet": packet, "hidden": state["hidden"] + packet[:, :2]}

    def advance(self, state, action):
        hidden = state["hidden"] + self.gain * action
        return {"packet": state["packet"], "hidden": hidden}, action, action.sum(-1)


def identity_head(auxiliary):
    with torch.no_grad():
        auxiliary.predictor.weight.copy_(torch.eye(auxiliary.teacher.hidden_size))
        auxiliary.predictor.bias.zero_()


@pytest.mark.parametrize("model_class", [GRUWorldModel, GRUResidualRewardWorldModel])
def test_gradients_reach_student_and_predictor_but_never_teacher(model_class):
    model = model_class(hidden_size=8).train()
    auxiliary = LatentConsistencyAuxiliary(model).train()
    packets, commands = sequence()
    loss, metrics = auxiliary(model, packets, commands)
    assert torch.isfinite(loss)
    assert metrics["valid_pairs_by_horizon"] == {"1": 12, "3": 8, "7": 0}
    loss.backward()
    assert model.transition.weight_ih.grad.abs().sum() > 0
    assert model.observation_update.weight_ih.grad.abs().sum() > 0
    assert auxiliary.predictor.weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in auxiliary.teacher.parameters())
    assert not auxiliary.teacher.training
    assert all(p is not q for p in model.parameters() for q in auxiliary.parameters())


def test_teacher_stays_eval_and_forward_does_not_update_weights():
    model = GRUWorldModel(hidden_size=8)
    auxiliary = LatentConsistencyAuxiliary(model)
    before = copy.deepcopy(auxiliary.teacher.state_dict())
    with torch.no_grad():
        next(model.parameters()).add_(3)
    auxiliary.train()
    assert auxiliary.training and auxiliary.predictor.training and not auxiliary.teacher.training
    auxiliary.teacher.train()  # Even an accidental external toggle is corrected.
    auxiliary(model, *sequence())
    assert not auxiliary.teacher.training
    for key, value in auxiliary.teacher.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_ema_updates_parameters_and_copies_all_buffers_exactly():
    model = GRUWorldModel(hidden_size=4)
    model.register_buffer("float_stat", torch.tensor([1.5, 2.5]))
    model.register_buffer("counter", torch.tensor(2, dtype=torch.long))
    model.register_buffer("temporary", torch.tensor([9.0]), persistent=False)
    auxiliary = LatentConsistencyAuxiliary(model, momentum=0.75)
    old = {key: value.detach().clone() for key, value in auxiliary.teacher.named_parameters()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(4)
        model.float_stat.copy_(torch.tensor([3.0, -8.0]))
        model.counter.fill_(12)
        model.temporary.fill_(77)
    auxiliary.update_teacher(model)
    for key, parameter in auxiliary.teacher.named_parameters():
        torch.testing.assert_close(parameter, old[key] + 1, rtol=1e-6, atol=1e-7)
    for key, buffer in auxiliary.teacher.named_buffers():
        torch.testing.assert_close(buffer, dict(model.named_buffers())[key], rtol=0, atol=0)
        assert buffer.data_ptr() != dict(model.named_buffers())[key].data_ptr()
    assert not auxiliary.teacher.training


def test_ema_schema_rejection_happens_before_any_parameter_change():
    model = GRUWorldModel(hidden_size=4)
    auxiliary = LatentConsistencyAuxiliary(model)
    old = copy.deepcopy(auxiliary.teacher.state_dict())
    with torch.no_grad():
        next(model.parameters()).add_(5)
    model.register_buffer("unexpected", torch.tensor(1))
    with pytest.raises(ValueError, match="membership"):
        auxiliary.update_teacher(model)
    for key, value in auxiliary.teacher.state_dict().items():
        torch.testing.assert_close(value, old[key], rtol=0, atol=0)


def test_action_alignment_and_teacher_endpoint_assimilation_are_exact():
    model = AdditiveGRU()
    auxiliary = LatentConsistencyAuxiliary(model, horizons=(1, 3))
    identity_head(auxiliary)
    packets, actions = sequence(batch=1, steps=3)
    packets[0, :, :2] = torch.tensor([[1, 2], [10, 20], [100, 200], [1000, 2000]])
    actions[0] = torch.tensor([[3, 5], [7, 11], [13, 17]])
    pairs = auxiliary.pairs(model, packets, actions)
    torch.testing.assert_close(pairs[1].prediction[0, 0], torch.tensor([4.0, 7.0]))
    torch.testing.assert_close(pairs[1].target[0, 0], torch.tensor([14.0, 27.0]))
    torch.testing.assert_close(pairs[1].prediction[0, 1], torch.tensor([21.0, 38.0]))
    torch.testing.assert_close(pairs[3].prediction[0, 0], torch.tensor([24.0, 35.0]))
    torch.testing.assert_close(pairs[3].target[0, 0], torch.tensor([1134.0, 2255.0]))
    assert not pairs[3].target.requires_grad


def test_future_measurements_change_teacher_targets_but_not_root_rollouts():
    model = AdditiveGRU()
    auxiliary = LatentConsistencyAuxiliary(model, horizons=(1, 3, 6))
    identity_head(auxiliary)
    packets, commands = sequence(batch=1)
    before = auxiliary.pairs(model, packets, commands)
    changed = packets.clone()
    changed[:, 1:, :2] += 100
    after = auxiliary.pairs(model, changed, commands)
    for horizon in auxiliary.horizons:
        torch.testing.assert_close(before[horizon].prediction[:, 0], after[horizon].prediction[:, 0],
                                   rtol=0, atol=0)
        assert not torch.equal(before[horizon].target[:, 0], after[horizon].target[:, 0])


def test_rollout_gradient_uses_only_commands_in_its_future_window():
    model = AdditiveGRU()
    auxiliary = LatentConsistencyAuxiliary(model, horizons=(3,))
    identity_head(auxiliary)
    packets, commands = sequence(batch=1, steps=6)
    commands.requires_grad_()
    prediction = auxiliary.pairs(model, packets, commands)[3].prediction[:, 0].sum()
    gradient, = torch.autograd.grad(prediction, commands)
    torch.testing.assert_close(gradient[:, :3], torch.ones(1, 3, 2))
    torch.testing.assert_close(gradient[:, 3:], torch.zeros(1, 3, 2))


def test_missing_angular_placeholders_are_discarded_before_both_branches():
    model = AdditiveGRU()
    auxiliary = LatentConsistencyAuxiliary(model)
    packets, commands = sequence()
    packets[:, 2:4, 6] = 0
    clean = packets.clone()
    clean[:, 2:4, :4] = 0
    poisoned = packets.clone()
    poisoned[:, 2:4, :4] = float("nan")
    before = auxiliary.pairs(model, clean, commands)
    after = auxiliary.pairs(model, poisoned, commands)
    for horizon in auxiliary.horizons:
        for field in ("prediction", "target", "valid"):
            torch.testing.assert_close(getattr(before[horizon], field), getattr(after[horizon], field),
                                       rtol=0, atol=0)


def test_validity_requires_both_measured_root_and_endpoint():
    model = GRUWorldModel(hidden_size=4)
    auxiliary = LatentConsistencyAuxiliary(model)
    packets, commands = sequence(batch=1)
    packets[0, :, 6] = torch.tensor([1, 0, 1, 1, 0, 1, 1])
    _, metrics = auxiliary(model, packets, commands)
    assert metrics["valid_pairs_by_horizon"] == {"1": 2, "3": 3, "7": 0}


@pytest.mark.parametrize("missing_packets", [6, 10])
def test_gap_spanning_horizon_includes_transition_to_next_valid_measurement(missing_packets):
    model = GRUWorldModel(hidden_size=4)
    horizons = (1, 3, missing_packets, missing_packets + 1)
    auxiliary = LatentConsistencyAuxiliary(model, horizons=horizons)
    packets, commands = sequence(batch=1, steps=missing_packets + 1)
    packets[:, 1:-1, 6] = 0
    packets[:, 1:-1, :4] = float("nan")
    pairs = auxiliary.pairs(model, packets, commands)
    assert not pairs[missing_packets].valid.any()
    assert pairs[missing_packets + 1].valid.tolist() == [[True]]
    _, metrics = auxiliary(model, packets, commands)
    assert metrics["valid_pairs_by_horizon"][str(missing_packets)] == 0
    assert metrics["valid_pairs_by_horizon"][str(missing_packets + 1)] == 1


def test_default_horizon_bridges_six_missing_packets():
    model = GRUWorldModel(hidden_size=4)
    auxiliary = LatentConsistencyAuxiliary(model)
    assert auxiliary.horizons == (1, 3, 7)
    packets, commands = sequence(batch=1, steps=7)
    packets[:, 1:-1, 6] = 0
    loss, metrics = auxiliary(model, packets, commands)
    assert metrics["valid_pairs_by_horizon"] == {"1": 0, "3": 0, "7": 1}
    loss.backward()
    assert model.transition.weight_ih.grad.abs().sum() > 0


def test_terminal_endpoint_allowed_but_no_targets_or_data_after_it():
    model = GRUWorldModel(hidden_size=4)
    auxiliary = LatentConsistencyAuxiliary(model)
    packets, commands = sequence(batch=1, steps=6)
    mask = torch.tensor([[True, True, False, False, False, False]])
    expected, expected_metrics = auxiliary(model, packets[:, :3], commands[:, :2])
    packets[:, 3:] = float("nan")
    commands[:, 2:] = float("nan")
    loss, metrics = auxiliary(model, packets, commands, transition_valid=mask)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    assert metrics["valid_pairs_by_horizon"] == expected_metrics["valid_pairs_by_horizon"]
    assert metrics["valid_pairs_by_horizon"] == {"1": 2, "3": 0, "7": 0}


def test_episode_rows_have_independent_terminal_lengths():
    model = GRUWorldModel(hidden_size=4)
    auxiliary = LatentConsistencyAuxiliary(model)
    mask = torch.tensor([[True, True, False, False, False, False], [True] * 6])
    _, metrics = auxiliary(model, *sequence(), transition_valid=mask)
    assert metrics["valid_pairs_by_horizon"] == {"1": 8, "3": 4, "7": 0}


def test_restarted_transition_mask_cannot_join_episodes():
    model = GRUWorldModel(hidden_size=4)
    auxiliary = LatentConsistencyAuxiliary(model)
    mask = torch.tensor([[True, False, True, False, False, False]])
    with pytest.raises(ValueError, match="restart/wrap"):
        auxiliary(model, *sequence(batch=1), transition_valid=mask)


def test_no_targets_produces_zero_loss_and_zero_gradient():
    model = GRUWorldModel(hidden_size=4)
    auxiliary = LatentConsistencyAuxiliary(model, variance_weight=1, covariance_weight=1)
    loss, metrics = auxiliary(model, *sequence(), transition_valid=torch.zeros(2, 6, dtype=torch.bool))
    assert loss.item() == 0 and metrics["nonempty_horizons"] == 0
    assert not metrics["regularization_defined"]
    assert metrics["student"]["samples"] == 0
    loss.backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.parameters())
    assert auxiliary.predictor.weight.grad.abs().sum() == 0


def test_forward_call_accounting_includes_teacher_and_shared_rollout_prefixes():
    model = GRUWorldModel(hidden_size=4)
    auxiliary = LatentConsistencyAuxiliary(model)
    _, metrics = auxiliary(model, *sequence())
    assert metrics["batch_forward_calls"] == {
        "student_assimilate": 7, "student_prefix_advance": 6,
        "student_open_loop_advance": 21, "student_predictor": 10,
        "teacher_assimilate": 7, "teacher_advance": 6,
    }


def test_horizons_without_complete_windows_are_empty_not_wrapped():
    model = GRUWorldModel(hidden_size=4)
    auxiliary = LatentConsistencyAuxiliary(model, horizons=(3, 6))
    pairs = auxiliary.pairs(model, *sequence(steps=2))
    for pair in pairs.values():
        assert pair.prediction.shape == pair.target.shape == (2, 0, 4)
        assert pair.valid.shape == (2, 0)
    loss, metrics = auxiliary(model, *sequence(steps=2))
    assert loss == 0 and metrics["nonempty_horizons"] == 0


def test_equal_horizon_weighting_and_optional_penalties_are_explicit():
    model = GRUWorldModel(hidden_size=4)
    auxiliary = LatentConsistencyAuxiliary(model, variance_weight=0.2, covariance_weight=0.3)
    packets, commands = sequence()
    pairs = auxiliary.pairs(model, packets, commands)
    expected = torch.stack([(p.prediction[p.valid] - p.target[p.valid]).square().mean()
                            for p in pairs.values() if p.valid.any()]).mean()
    features = torch.cat([p.prediction[p.valid] for p in pairs.values()])
    variance, covariance = vicreg_terms(features)
    loss, metrics = auxiliary(model, packets, commands)
    torch.testing.assert_close(loss, expected + 0.2 * variance + 0.3 * covariance)
    assert metrics["regularization_defined"]


def test_collapse_diagnostics_detect_constant_and_known_covariance_rank():
    constant = collapse_diagnostics(torch.ones(5, 3))
    assert constant["mean_std"] == constant["min_std"] == constant["effective_rank"] == 0
    full_rank = collapse_diagnostics(torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]))
    assert full_rank["effective_rank"] == pytest.approx(2)
    rank_one = collapse_diagnostics(torch.tensor([[-1., -2.], [0., 0.], [1., 2.]]))
    assert rank_one["effective_rank"] == pytest.approx(1)
    assert collapse_diagnostics(torch.empty(0, 3))["effective_rank"] == 0
    assert collapse_diagnostics(torch.ones(1, 3))["effective_rank"] == 0


def test_vicreg_terms_match_unbiased_covariance_and_have_gradients():
    features = torch.tensor([[-0.2, -0.4], [0., 0.], [0.2, 0.4]], requires_grad=True)
    variance, covariance = vicreg_terms(features)
    expected_variance = (1 - torch.tensor([0.04 + 1e-4, 0.16 + 1e-4]).sqrt()).mean()
    torch.testing.assert_close(variance, expected_variance)
    assert covariance.item() == pytest.approx(0.0064)
    (variance + covariance).backward()
    assert features.grad.abs().sum() > 0
    one = torch.ones(1, 3, requires_grad=True)
    assert all(x == 0 for x in vicreg_terms(one))


@pytest.mark.parametrize("horizons", [(), (0,), (-1,), (1, 1), (True,), (1.5,)])
def test_invalid_horizons_rejected(horizons):
    with pytest.raises(ValueError):
        LatentConsistencyAuxiliary(GRUWorldModel(hidden_size=4), horizons=horizons)


@pytest.mark.parametrize("kwargs", [{"momentum": -0.1}, {"momentum": 1.1}, {"momentum": float("nan")},
                                   {"variance_weight": -1}, {"covariance_weight": float("inf")}])
def test_invalid_ema_and_penalty_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        LatentConsistencyAuxiliary(GRUWorldModel(hidden_size=4), **kwargs)


def test_stochastic_state_semantics_are_explicitly_unsupported():
    with pytest.raises(TypeError, match="deterministic"):
        LatentConsistencyAuxiliary(GaussianRSSM(hidden_size=4))


@pytest.mark.parametrize("bad", ["visible_nan", "command_nan", "flag", "age", "dtype", "shape", "mask_dtype"])
def test_invalid_available_inputs_fail_closed(bad):
    model = GRUWorldModel(hidden_size=4)
    auxiliary = LatentConsistencyAuxiliary(model)
    packets, commands = sequence()
    kwargs = {}
    if bad == "visible_nan":
        packets[0, 0, 0] = float("nan")
    elif bad == "command_nan":
        commands[0, 0, 0] = float("nan")
    elif bad == "flag":
        packets[0, 0, 6] = 0.5
    elif bad == "age":
        packets[0, 0, 7] = -1
    elif bad == "dtype":
        commands = commands.double()
    elif bad == "shape":
        packets = packets[:, :-1]
    else:
        kwargs["transition_valid"] = torch.ones(2, 6)
    with pytest.raises(ValueError):
        auxiliary(model, packets, commands, **kwargs)
