"""Fabricated integration oracles; no scientific generator or saved model."""

import itertools

import numpy as np
import pytest
import torch

from openjev.research import finite_rounded_models as models
from openjev.research.finite_expected_count_bridge import torch_objective as old_objective
from openjev.research.finite_factorized_dynamics_models import make_model as old_model
from openjev.research.finite_factorized_dynamics_models import prefix_predictions as old_prefix

SEED = 941101
ARMS = ("original_free", "matched_free", "rounded")


def public_prefix(found_step=None):
    value = torch.zeros((1, 9, 31), dtype=torch.float32)
    value[0, 0, 6] = value[0, 0, 9] = 1
    length = 9 if found_step is None else found_step + 1
    for step in range(1, length):
        value[0, step, (step + 1) % 4] = 1
        value[0, step, 8 if step == found_step else 4 + step % 4] = 1
    return value, torch.tensor([length], dtype=torch.int64)


def mixed_prefix():
    items = [public_prefix(found) for found in (None, 1, 3, 8)]
    return torch.cat([row[0] for row in items]), torch.cat([row[1] for row in items])


def blocks():
    return (torch.tensor([[0, 1, 3, 2, 2, 1, 0, 3]], dtype=torch.int64),
            torch.tensor([[0, 2, 1, 3, 1, 0, 4, 4]], dtype=torch.int64))


def compare_outputs(first, second):
    assert set(first) == set(second)
    for name in first:
        if name == "work":
            continue
        if first[name] is None:
            assert second[name] is None
        else:
            torch.testing.assert_close(first[name], second[name], atol=1e-12, rtol=0, msg=name)


def joint_loss(model):
    prefix, lengths = public_prefix()
    actions, observations = blocks()
    blind = model.blind_rollout(prefix, lengths, actions[:, :2])
    observed = model.observed_rollout(prefix, lengths, actions[:, :2], observations[:, :2])
    target = torch.tensor([.12, -.08, .03, -.07], dtype=torch.float64)
    return ((blind["cost_contrasts"] - target).square().mean()
            + (observed["cost_contrasts"] - target.flip(0)).square().mean()
            + blind["survival_mass"].square().mean()
            - observed["probabilities"][0, 1, 2].log())


def custom_parameters():
    action, destination, origin = np.indices((4, 8, 8))
    transition = ((3 * action + 5 * destination + 7 * origin + 3 * destination * origin
                   + action * origin) % 19 - 9) / 4
    emission = np.full((4, 8), 1 / 8, dtype=np.float64)
    head = np.empty((4, 8), dtype=np.float64)
    for state in range(8):
        emission[(state + 1) % 4, state] = 5 / 8
        head[:, state] = np.roll(np.array([.55, .25, .15, .05]), state % 4)
    hazard = np.array([[1 / 16 + ((a + state) % 3) / 32 for state in range(8)]
                       for a in range(4)], dtype=np.float64)
    return {"transition_logits": transition.astype(np.float64), "emission_logits": np.log(emission),
            "hazard_logits": np.log(hazard / (1 - hazard)), "cost_logits": np.log(head)}


def inject(model, parameters):
    with torch.no_grad():
        for name, value in parameters.items():
            getattr(model, name).copy_(torch.from_numpy(value))


def rounded_reference(logits):
    """Independent finite4 normalization and the declared positive-slack map."""
    value = logits.copy()
    for _ in range(4):
        for axis in (2, 1):
            maximum = value.max(axis=axis, keepdims=True)
            value -= maximum + np.log(np.exp(value - maximum).sum(axis=axis, keepdims=True))
    raw = np.exp(value)
    tau = 1 - 1e-8
    rows = raw.sum(2)
    row_contracted = raw * (tau / np.maximum(rows, tau))[:, :, None]
    columns = row_contracted.sum(1)
    contracted = row_contracted * (tau / np.maximum(columns, tau))[:, None, :]
    row_deficit, column_deficit = 1 - contracted.sum(2), 1 - contracted.sum(1)
    correction = (row_deficit / row_deficit.sum(1, keepdims=True))[:, :, None] * column_deficit[:, None, :]
    return contracted + correction, (rows > tau, columns > tau)


def numpy_fields(parameters, *, rounded):
    def softmax(value, axis):
        value = np.exp(value - value.max(axis=axis, keepdims=True))
        return value / value.sum(axis=axis, keepdims=True)

    if rounded:
        transition, _branches = rounded_reference(parameters["transition_logits"])
    else:
        transition = softmax(parameters["transition_logits"], 1)
    emission = softmax(parameters["emission_logits"], 0)
    hazard = 1 / (1 + np.exp(-parameters["hazard_logits"]))
    return transition, emission, hazard, .25 - softmax(parameters["cost_logits"], 0)


def enumerate_history(transition, emission, hazard, actions, observations):
    """Exhaust all hidden paths, including destination hazard before odor."""
    assert len(observations) == len(actions) + 1 and observations[0] < 4
    assert 4 not in observations[:-1]
    masses = np.zeros(8, dtype=np.float64)
    for path in itertools.product(range(8), repeat=len(observations)):
        probability = emission[observations[0], path[0]] / 8
        for step, action in enumerate(actions, 1):
            before, after = path[step - 1], path[step]
            probability *= transition[action, after, before]
            if observations[step] == 4:
                probability *= hazard[action, after]
            else:
                probability *= (1 - hazard[action, after]) * emission[observations[step], after]
        masses[path[-1]] += probability
    return masses


def test_original_free_matches_frozen_initialization_outputs_losses_and_gradients():
    current = models.make_model("original_free", SEED)
    frozen = old_model("factorized", SEED)
    assert set(current.state_dict()) == set(frozen.state_dict())
    for name, value in current.state_dict().items():
        torch.testing.assert_close(value, frozen.state_dict()[name], atol=0, rtol=0)
    prefix, lengths = mixed_prefix()
    compare_outputs(models.prefix_predictions(current, prefix, lengths), old_prefix(frozen, prefix, lengths))
    ordinary, ordinary_lengths = public_prefix()
    actions, observations = blocks()
    compare_outputs(current.blind_rollout(ordinary, ordinary_lengths, actions),
                    frozen.blind_rollout(ordinary, ordinary_lengths, actions))
    compare_outputs(current.observed_rollout(ordinary, ordinary_lengths, actions, observations),
                    frozen.observed_rollout(ordinary, ordinary_lengths, actions, observations))
    for model, objective in ((current, models.torch_objective), (frozen, old_objective)):
        model.zero_grad(set_to_none=True)
        result = objective(model, prefix, lengths, .001)
        result["loss"].backward()
        model._test_result = result
    for key in ("loss", "log_likelihood", "log_prior"):
        torch.testing.assert_close(current._test_result[key], frozen._test_result[key], atol=1e-12, rtol=0)
    for name, parameter in current.named_parameters():
        other = dict(frozen.named_parameters())[name]
        if name == "cost_logits":
            assert parameter.grad is None and other.grad is None
        else:
            torch.testing.assert_close(parameter.grad, other.grad, atol=1e-12, rtol=0)
    for model in (current, frozen):
        model.zero_grad(set_to_none=True)
    first, second = joint_loss(current), joint_loss(frozen)
    torch.testing.assert_close(first, second, atol=1e-12, rtol=0)
    first.backward()
    second.backward()
    for name, parameter in current.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(frozen.named_parameters())[name].grad,
                                   atol=1e-12, rtol=0)


def test_matched_free_and_rounded_have_equal_initial_functions_and_preserve_rng():
    rng = torch.random.get_rng_state().clone()
    free = models.make_model("matched_free", SEED)
    rounded = models.make_model("rounded", SEED)
    original = models.make_model("original_free", SEED)
    assert torch.equal(torch.random.get_rng_state(), rng)
    for name in ("emission_logits", "hazard_logits", "cost_logits"):
        assert torch.equal(getattr(free, name), getattr(rounded, name))
        assert torch.equal(getattr(free, name), getattr(original, name))
    a, b = models.probability_fields(free), models.probability_fields(rounded)
    for name in ("transition", "emission", "hazard"):
        torch.testing.assert_close(a[name], b[name], atol=1e-12, rtol=0)
    torch.testing.assert_close(free.readout_matrix(), rounded.readout_matrix(), atol=1e-12, rtol=0)
    first_snapshot, second_snapshot = free.dynamics_snapshot(), rounded.dynamics_snapshot()
    for name in ("reset_emission", "observed", "found", "blind"):
        torch.testing.assert_close(first_snapshot[name], second_snapshot[name],
                                   atol=1e-12, rtol=0)
    prefix, lengths = mixed_prefix()
    compare_outputs(models.prefix_predictions(free, prefix, lengths), models.prefix_predictions(rounded, prefix, lengths))
    ordinary, ordinary_lengths = public_prefix()
    actions, observations = blocks()
    for horizon in (1, 2, 8):
        compare_outputs(free.blind_rollout(ordinary, ordinary_lengths, actions[:, :horizon]),
                        rounded.blind_rollout(ordinary, ordinary_lengths, actions[:, :horizon]))
        compare_outputs(free.observed_rollout(ordinary, ordinary_lengths, actions[:, :horizon], observations[:, :horizon]),
                        rounded.observed_rollout(ordinary, ordinary_lengths, actions[:, :horizon], observations[:, :horizon]))


def test_rounded_prefix_and_forecast_match_independent_hidden_path_enumeration():
    model = models.make_model("rounded", SEED)
    parameters = custom_parameters()
    inject(model, parameters)
    transition, emission, hazard, head = numpy_fields(parameters, rounded=True)
    prefix, lengths = public_prefix()
    prediction = models.prefix_predictions(model, prefix, lengths)
    actions, labels = [], [2]
    previous_mass = float(emission[2].sum() / 8)
    reset_law = np.r_[emission.sum(1) / 8, 0.]
    np.testing.assert_allclose(prediction["probabilities"][0, 0].detach().numpy(), reset_law, atol=1e-12, rtol=0)
    for step in (1, 2):
        actions.append((step + 1) % 4)
        expected = np.array([enumerate_history(transition, emission, hazard, actions, labels + [label]).sum()
                             / previous_mass for label in range(5)])
        np.testing.assert_allclose(prediction["probabilities"][0, step].detach().numpy(), expected,
                                   atol=1e-12, rtol=0)
        observed = step % 4
        assert float(prediction["nll"][0, step].detach()) == pytest.approx(-np.log(expected[observed]), abs=1e-12)
        labels.append(observed)
        previous_mass = float(enumerate_history(transition, emission, hazard, actions, labels).sum())
    masses = enumerate_history(transition, emission, hazard, actions, labels)
    posterior = masses / masses.sum()
    small_prefix = prefix[:, :3].clone()
    small_lengths = torch.tensor([3], dtype=torch.int64)
    np.testing.assert_allclose(model.encode_prefix(small_prefix, small_lengths)[0].detach().numpy(), posterior,
                               atol=1e-12, rtol=0)
    next_action = 1
    next_masses = [enumerate_history(transition, emission, hazard, actions + [next_action], labels + [label])
                   for label in range(5)]
    prior = sum(next_masses[:4]) / masses.sum()
    outcome_law = np.array([row.sum() / masses.sum() for row in next_masses])
    forecast = model.observed_rollout(small_prefix, small_lengths,
                                     torch.tensor([[next_action]]), torch.tensor([[4]]))
    np.testing.assert_allclose(forecast["prior_states"][0, 0].detach().numpy(), prior, atol=1e-12, rtol=0)
    np.testing.assert_allclose(forecast["probabilities"][0, 0].detach().numpy(), outcome_law, atol=1e-12, rtol=0)
    np.testing.assert_allclose(forecast["cost_contrasts"][0, 0].detach().numpy(), head @ prior, atol=1e-12, rtol=0)
    assert float(forecast["survival_mass"][0, 0].detach()) == pytest.approx(float(prior.sum()), abs=1e-12)
    assert torch.count_nonzero(forecast["posterior_states"]) == 0
    terminal, terminal_lengths = public_prefix(2)
    terminated = models.prefix_predictions(model, terminal, terminal_lengths)
    expected_mass = enumerate_history(transition, emission, hazard, [2, 3], [2, 1, 4]).sum()
    assert float(terminated["nll"].sum().detach()) == pytest.approx(-np.log(expected_mass), abs=1e-12)
    assert torch.count_nonzero(terminated["probabilities"][:, 3:]) == 0
    assert torch.count_nonzero(terminated["nll"][:, 3:]) == 0


def test_prefix_objective_gradient_matches_hidden_path_finite_difference_with_same_rounded_prior():
    model = models.make_model("rounded", SEED)
    parameters = custom_parameters()
    inject(model, parameters)
    prefix, lengths = public_prefix(2)
    objective = models.torch_objective(model, prefix, lengths, .001)
    gradient, = torch.autograd.grad(objective["loss"], model.transition_logits)
    direction = np.sin(np.arange(256, dtype=np.float64)).reshape(4, 8, 8)

    def scalar_loss(logits):
        values = {**parameters, "transition_logits": logits}
        transition, emission, hazard, _head = numpy_fields(values, rounded=True)
        likelihood = enumerate_history(transition, emission, hazard, [2, 3], [2, 1, 4]).sum()
        prior = .001 * (np.log(transition).sum() + np.log(emission).sum()
                        + np.log(hazard).sum() + np.log1p(-hazard).sum())
        return (-np.log(likelihood) - prior) / 3

    assert float(objective["loss"].detach()) == pytest.approx(scalar_loss(parameters["transition_logits"]), abs=1e-12)
    step = 1e-5
    # The finite-difference claim concerns a smooth neighborhood of the actual
    # finite algorithm, not an infinite balancing limit or a maximum tie.
    _, center_branches = rounded_reference(parameters["transition_logits"])
    for sign in (-1, 1):
        _, perturbed_branches = rounded_reference(parameters["transition_logits"] + sign * step * direction)
        for center, perturbed in zip(center_branches, perturbed_branches, strict=True):
            np.testing.assert_array_equal(center, perturbed)
    expected = (scalar_loss(parameters["transition_logits"] + step * direction)
                - scalar_loss(parameters["transition_logits"] - step * direction)) / (2 * step)
    actual = float((gradient.detach().numpy() * direction).sum())
    assert np.linalg.norm(gradient.detach().numpy()) > 1e-6
    assert actual == pytest.approx(expected, abs=2e-8, rel=2e-6)


@pytest.mark.parametrize("arm", ARMS)
def test_prefix_gradient_excludes_head_and_joint_gradient_includes_every_parameter(arm):
    model = models.make_model(arm, SEED)
    prefix, lengths = mixed_prefix()
    head = model.cost_logits.detach().clone()
    models.torch_objective(model, prefix, lengths, .001)["loss"].backward()
    assert model.cost_logits.grad is None
    for name, value in model.named_parameters():
        if name != "cost_logits":
            assert value.grad is not None and torch.isfinite(value.grad).all()
            assert torch.count_nonzero(value.grad) > 0
    assert torch.equal(model.cost_logits, head)
    model.zero_grad(set_to_none=True)
    joint_loss(model).backward()
    for value in model.parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0


@pytest.mark.parametrize("arm", ARMS)
def test_mass_hazard_order_and_linear_head_remain_unchanged(arm):
    model = models.make_model(arm, SEED)
    inject(model, custom_parameters())
    fields = models.probability_fields(model)
    snapshot = model.dynamics_snapshot()
    transition, emission, hazard = (fields[key] for key in ("transition", "emission", "hazard"))
    surviving = (1 - hazard[:, :, None]) * transition
    found = (hazard[:, :, None] * transition).sum(1)
    torch.testing.assert_close(snapshot["blind"], surviving, atol=1e-12, rtol=0)
    torch.testing.assert_close(snapshot["observed"], emission[None, :, :, None] * surviving[:, None], atol=1e-12, rtol=0)
    torch.testing.assert_close(snapshot["found"], found, atol=1e-12, rtol=0)
    torch.testing.assert_close(snapshot["observed"].sum((1, 2)) + snapshot["found"],
                               torch.ones((4, 8), dtype=torch.float64), atol=1e-12, rtol=0)
    assert float((surviving.sum(2) - 1).abs().max()) > .01


@pytest.mark.parametrize("arm", ARMS)
def test_state_dictionary_roundtrip_preserves_every_output_without_aliases(arm):
    model = models.make_model(arm, SEED)
    inject(model, custom_parameters())
    checkpoint = {name: value.detach().clone() for name, value in model.state_dict().items()}
    restored = models.make_model(arm, SEED)
    restored.load_state_dict(checkpoint, strict=True)
    for name, value in restored.state_dict().items():
        assert torch.equal(value, checkpoint[name]) and value.data_ptr() != checkpoint[name].data_ptr()
    prefix, lengths = mixed_prefix()
    compare_outputs(models.prefix_predictions(model, prefix, lengths), models.prefix_predictions(restored, prefix, lengths))
    prefix, lengths = public_prefix()
    actions, observations = blocks()
    compare_outputs(model.blind_rollout(prefix, lengths, actions), restored.blind_rollout(prefix, lengths, actions))
    compare_outputs(model.observed_rollout(prefix, lengths, actions, observations),
                    restored.observed_rollout(prefix, lengths, actions, observations))


@pytest.mark.parametrize("arm", ARMS)
def test_public_input_guards_reject_oracle_hidden_features_and_poisoned_padding(arm):
    model = models.make_model(arm, SEED)
    prefix, lengths = public_prefix()
    actions, _observations = blocks()
    with pytest.raises((TypeError, ValueError)):
        model.blind_rollout(prefix, lengths, actions, oracle_prefix=torch.full((1, 8), 1 / 8, dtype=torch.float64))
    prefix[0, 0, 12] = 1
    with pytest.raises(ValueError):
        models.prefix_predictions(model, prefix, lengths)
    prefix, lengths = public_prefix(2)
    prefix[0, 5, 3] = 1
    with pytest.raises(ValueError):
        models.prefix_predictions(model, prefix, lengths)


def assert_kernel_work(work, calls):
    # Every construction executes the fixed finite4-plus-rounding function.
    expected = {"rounded_transition_calls": 1, "row_logsumexp_calls": 4,
                "column_logsumexp_calls": 4, "normalization_vectors": 256,
                "normalization_entries": 2048, "completed_sweeps": 4,
                "exponential_calls": 1, "exponential_entries": 256,
                "matrix_sum_calls": 7, "vector_sum_calls": 2,
                "contraction_maximum_calls": 2, "contraction_division_entries": 64,
                "contraction_product_entries": 512, "deficit_subtraction_entries": 64,
                "deficit_normalization_entries": 32, "rank_one_product_entries": 256,
                "rank_one_addition_entries": 256, "correction_mass_sum_calls": 1,
                "logarithm_calls": 1, "logarithm_entries": 256, "check_calls": 11}
    for key, value in expected.items():
        assert work[key] == calls * value, key


@pytest.mark.parametrize("arm", ARMS)
def test_metadata_constructor_and_distinct_route_work_are_complete_and_stable(arm):
    model = models.make_model(arm, SEED)
    metadata = model.parameter_metadata()
    assert metadata["count"] == metadata["trainable_count"] == 352
    assert metadata["parameter_bytes"] == 2816 and metadata["buffer_bytes"] == 0
    assert metadata["transition_stored_parameters"] == 256
    assert metadata["transition_ideal_constraint_degrees"] == (196 if arm == "rounded" else 224)
    assert metadata["rounded_sweeps"] == (4 if arm == "rounded" else None)
    assert metadata["rounded_slack"] == (1e-8 if arm == "rounded" else None)
    assert metadata["rounded_tolerance"] == (1e-12 if arm == "rounded" else None)
    assert metadata["privileged_prefix"] is False and metadata["known_operators"] is False
    assert metadata["fixed_cost_readout"] is False
    constructor = metadata["construction_work"]
    assert set(constructor) == set(models.WORK_KEYS) and len(constructor) == 71
    assert constructor["adapter_check_calls"] == 1
    assert_kernel_work(constructor, int(arm == "matched_free"))
    for key, scale in {"matching_log_calls": 1, "matching_log_entries": 256,
                       "matching_copy_calls": 1, "matching_copy_entries": 256,
                       "matching_verification_softmax_calls": 1,
                       "matching_verification_probability_columns": 32}.items():
        assert constructor[key] == scale * (arm == "matched_free")
    assert constructor["probability_field_calls"] == 0

    prefix, lengths = mixed_prefix()
    objective = models.torch_objective(model, prefix, lengths, .001)
    work = objective["work"]
    assert set(work) == set(models.WORK_KEYS)
    assert all(type(value) is int and value >= 0 for value in work.values())
    assert objective["valid_events"] == 24
    assert work["prefix_probability_rows"] == work["prefix_nll_rows"] == 24
    assert work["prefix_filter_calls"] == 8 and work["prefix_filter_rows"] == 17
    assert work["reset_emission_rows"] == 4
    assert work["factorization_calls"] == work["probability_field_calls"] == 1
    assert work["factor_product_entries"] == 1536 and work["factor_found_reduction_columns"] == 32
    assert work["emission_softmax_calls"] == work["hazard_sigmoid_calls"] == 1
    assert work["transition_softmax_calls"] == int(arm != "rounded")
    assert_kernel_work(work, int(arm == "rounded"))
    assert work["prior_transition_log_softmax_calls"] == int(arm != "rounded")
    assert work["prior_emission_log_softmax_calls"] == 1 and work["prior_hazard_logsigmoid_calls"] == 2
    assert work["prior_normalization_vectors"] == (8 if arm == "rounded" else 40)
    assert work["prior_normalization_entries"] == (32 if arm == "rounded" else 288)
    assert work["prior_logsigmoid_entries"] == 64
    assert work["cost_head_softmax_calls"] == work["cost_readout_calls"] == 0

    ordinary, ordinary_lengths = public_prefix()
    actions, observations = blocks()
    for observed in (False, True):
        result = (model.observed_rollout(ordinary, ordinary_lengths, actions[:, :2], observations[:, :2])
                  if observed else model.blind_rollout(ordinary, ordinary_lengths, actions[:, :2]))
        route = result["work"]
        assert set(route) == set(models.WORK_KEYS)
        assert route["probability_field_calls"] == route["factorization_calls"] == 2
        assert route["prefix_filter_calls"] == route["prefix_filter_rows"] == 8
        assert route["cost_readout_calls"] == route["cost_readout_rows"] == 2
        assert route["cost_head_softmax_calls"] == 2 and route["cost_head_probability_rows"] == 16
        assert route["operator_marginal_sum_calls"] == 1
        assert route["event_probability_rows"] == 2 * observed
        assert route["prior_transition_log_softmax_calls"] == route["prior_emission_log_softmax_calls"] == 0
        assert_kernel_work(route, 2 * (arm == "rounded"))
    assert model.parameter_metadata() == metadata


def test_probability_export_owns_storage_preserves_gradient_and_accumulates_work():
    model = models.make_model("rounded", SEED)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    work = {"external_count": 9}
    result = models.probability_fields(model, work=work)
    assert result["work"] is work and model.last_work is work and work["external_count"] == 9
    assert result["transition"].data_ptr() != model.transition_logits.data_ptr()
    assert result["transition"].data_ptr() != result["log_transition"].data_ptr()
    assert result["transition"].grad_fn is not None and result["log_transition"].grad_fn is not None
    detached = result["transition"].detach().clone()
    models.probability_fields(model, work=work)
    assert_kernel_work(work, 2)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])
    with torch.no_grad():
        model.transition_logits[0, 0, 0] += .01
    assert torch.equal(result["transition"].detach(), detached)


def test_selected_underflow_transition_remains_positive_through_prefix_prior_and_forecast():
    model = models.make_model("rounded", SEED)
    with torch.no_grad():
        model.transition_logits.fill_(-1000.)
        for state in range(8):
            model.transition_logits[:, state, state] = 1000.
    before = model.transition_logits.detach().clone()
    fields = models.probability_fields(model)
    expected = (1 - 1e-8) * torch.eye(8, dtype=torch.float64) + 1e-8 / 8
    torch.testing.assert_close(fields["transition"], expected.expand(4, 8, 8), atol=1e-12, rtol=0)
    torch.testing.assert_close(fields["log_transition"], fields["transition"].log(), atol=0, rtol=0)
    assert_kernel_work(fields["work"], 1)
    prefix, lengths = mixed_prefix()
    objective = models.torch_objective(model, prefix, lengths, .001)
    objective["loss"].backward()
    assert torch.isfinite(objective["loss"])
    assert model.transition_logits.grad is not None and torch.isfinite(model.transition_logits.grad).all()
    # A finite saturated gradient may be zero. This is guard coverage, not a
    # claim that vanished pre-round exponential directions are recoverable.
    assert model.cost_logits.grad is None
    ordinary, ordinary_lengths = public_prefix()
    actions, observations = blocks()
    for result in (model.blind_rollout(ordinary, ordinary_lengths, actions),
                   model.observed_rollout(ordinary, ordinary_lengths, actions, observations)):
        for value in result.values():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
    assert torch.equal(model.transition_logits, before)


def test_overflow_and_external_stop_preserve_work_without_retry_or_parameter_mutation():
    model = models.make_model("rounded", SEED)
    with torch.no_grad():
        model.transition_logits.fill_(-1e308)
        model.transition_logits[:, :, 0] = 1e308
    before = model.transition_logits.detach().clone()
    with pytest.raises(ValueError, match="row-normalized logs: finite") as caught:
        models.probability_fields(model)
    work = caught.value.rounded_model_work
    assert work == model.last_work and work is not model.last_work
    assert work["rounded_transition_calls"] == work["row_logsumexp_calls"] == 1
    assert work["column_logsumexp_calls"] == work["completed_sweeps"] == 0
    assert work["check_calls"] == 1 and work["normalization_entries"] == 256
    assert work["exponential_calls"] == work["matrix_sum_calls"] == work["emission_softmax_calls"] == 0
    assert torch.equal(model.transition_logits, before)
    model.last_work["completed_sweeps"] += 1
    assert work["completed_sweeps"] == 0

    model = models.make_model("rounded", SEED)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    error = RuntimeError("external stop")
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        if calls == 3:  # field-entry, kernel-entry, first completed sweep
            raise error

    model.check = stop
    with pytest.raises(RuntimeError) as caught:
        models.probability_fields(model)
    assert caught.value is error and calls == 3
    work = error.rounded_model_work
    assert work["adapter_check_calls"] == work["rounded_transition_calls"] == 1
    assert work["check_calls"] == 2 and work["completed_sweeps"] == 1
    assert work["row_logsumexp_calls"] == work["column_logsumexp_calls"] == 1
    assert work["exponential_calls"] == work["emission_softmax_calls"] == 0
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])


@pytest.mark.parametrize("name", ("transition_logits", "emission_logits", "hazard_logits", "cost_logits"))
@pytest.mark.parametrize("value", (float("nan"), float("inf")))
def test_nonfinite_parameter_is_rejected_before_normalization(name, value):
    model = models.make_model("rounded", SEED)
    with torch.no_grad():
        getattr(model, name).reshape(-1)[0] = value
    with pytest.raises(ValueError) as caught:
        models.probability_fields(model)
    work = caught.value.rounded_model_work
    assert work["rounded_transition_calls"] == work["probability_field_calls"] == 0


def test_matched_constructor_stop_exposes_only_performed_work():
    error = RuntimeError("constructor stop")
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise error

    with pytest.raises(RuntimeError) as caught:
        models.make_model("matched_free", SEED, check=stop)
    assert caught.value is error and calls == 3
    work = error.rounded_model_work
    assert work["adapter_check_calls"] == 1 and work["rounded_transition_calls"] == 1
    assert work["completed_sweeps"] == 1 and work["check_calls"] == 2
    assert work["matching_log_calls"] == work["matching_copy_calls"] == 0


@pytest.mark.parametrize("name", ("emission_logits", "hazard_logits", "cost_logits"))
def test_saturated_probability_fields_or_head_stop_without_clipping(name):
    model = models.make_model("rounded", SEED)
    with torch.no_grad():
        parameter = getattr(model, name)
        parameter.fill_(-1000.)
        parameter[0] = 1000.
    prefix, lengths = public_prefix()
    actions, _observations = blocks()
    with pytest.raises(ValueError, match="strict.*probability bounds"):
        model.blind_rollout(prefix, lengths, actions[:, :1])


def test_observed_forecasts_precede_current_labels_and_ignore_future_actions():
    model = models.make_model("rounded", SEED)
    inject(model, custom_parameters())
    prefix, lengths = public_prefix()
    actions, observations = blocks()
    original = model.observed_rollout(prefix, lengths, actions, observations)
    other_labels = observations.clone()
    other_labels[:, 2:] = 4
    terminal = model.observed_rollout(prefix, lengths, actions, other_labels)
    for key in ("cost_contrasts", "survival_mass", "probabilities", "prior_states"):
        torch.testing.assert_close(original[key][:, :3], terminal[key][:, :3], atol=1e-12, rtol=0)
    assert torch.count_nonzero(terminal["prior_states"][:, 3:]) == 0
    assert torch.equal(terminal["probabilities"][:, 3:, 4], torch.ones((1, 5), dtype=torch.float64))
    future = actions.clone()
    future[:, 3:] = (future[:, 3:] + 1) % 4
    baseline = model.blind_rollout(prefix, lengths, actions)
    altered = model.blind_rollout(prefix, lengths, future)
    for key in ("cost_contrasts", "survival_mass", "prior_states"):
        torch.testing.assert_close(baseline[key][:, :3], altered[key][:, :3], atol=1e-12, rtol=0)
