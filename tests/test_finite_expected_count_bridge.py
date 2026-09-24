"""Fabricated public histories check the actual recurrent-training boundary."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from openjev.research.finite_expected_count import expected_counts, tokens_from_prefix
from openjev.research.finite_expected_count_bridge import (
    DYNAMICS,
    fit_prefix,
    import_probabilities,
    numpy_objective,
    probabilities,
    torch_objective,
)
from openjev.research.finite_factorized_dynamics_models import make_model


def public_histories():
    # Four surviving histories and four first-found histories. No world sampler
    # or stored benchmark data is consulted; all public symbols are prescribed.
    prefix = np.zeros((8, 9, 31), np.float32)
    lengths = np.array([9, 9, 9, 9, 2, 4, 6, 8], np.int64)
    for row, length in enumerate(lengths):
        prefix[row, 0, 4 + row % 4] = 1
        prefix[row, 0, 9] = 1
        for step in range(1, length):
            prefix[row, step, (row + 2 * step) % 4] = 1
            obs = 4 if row >= 4 and step == length - 1 else (row + step) % 4
            prefix[row, step, 4 + obs] = 1
    return torch.from_numpy(prefix), torch.from_numpy(lengths)


def numpy_loss(model, prefix, lengths):
    args = tokens_from_prefix(prefix.numpy(), lengths.numpy())
    p = probabilities(model)
    result = expected_counts(*p, args['actions'], args['observations'], args['lengths'])
    return -numpy_objective(*p, result, .001) / int(lengths.sum())


def test_torch_likelihood_and_prior_match_separate_numpy_inference():
    model = make_model('factorized', 937101)
    prefix, lengths = public_histories()
    result = torch_objective(model, prefix, lengths, .001)
    assert float(result['loss'].detach()) == pytest.approx(numpy_loss(model, prefix, lengths), abs=2e-12)
    assert result['valid_events'] == int(lengths.sum())


@pytest.mark.parametrize(('name', 'index'), [
    ('transition_logits', (1, 2, 3)), ('emission_logits', (2, 5)), ('hazard_logits', (3, 6))])
def test_gradient_matches_numerical_derivative_of_independent_likelihood(name, index):
    model = make_model('factorized', 937101)
    prefix, lengths = public_histories()
    torch_objective(model, prefix, lengths, .001)['loss'].backward()
    parameter = getattr(model, name)
    analytic = float(parameter.grad[index])
    original, step = float(parameter[index].detach()), 1e-5
    with torch.no_grad():
        parameter[index] = original + step
    high = numpy_loss(model, prefix, lengths)
    with torch.no_grad():
        parameter[index] = original - step
    low = numpy_loss(model, prefix, lengths)
    with torch.no_grad():
        parameter[index] = original
    assert analytic == pytest.approx((high - low) / (2 * step), abs=2e-9, rel=2e-5)
    assert model.cost_logits.grad is None


@pytest.mark.parametrize('method', ['em', 'gradient'])
def test_both_prefix_learners_preserve_head_and_forward_probability_validity(method):
    model = make_model('factorized', 937101)
    prefix, lengths = public_histories()
    head = model.cost_logits.detach().clone()
    initial = {name: getattr(model, name).detach().clone() for name in DYNAMICS}
    checks = []
    result = fit_prefix(model, prefix, lengths, method, 3, check=lambda: checks.append(True))
    assert checks and result['updates'] == 3 and len(result['history']) == 3
    assert result['head_unchanged'] and torch.equal(head, model.cost_logits)
    assert model.cost_logits.grad is None
    assert any(not torch.equal(initial[name], getattr(model, name)) for name in DYNAMICS)
    assert result['compute_matched'] is False and result['hidden_state_input'] is False
    if method == 'em':
        assert all(r['penalized_log_likelihood_after'] >= r['penalized_log_likelihood_before'] - 1e-9
                   for r in result['history'])
        assert result['work']['numpy_expectation_passes'] == 6
        assert result['work']['full_gradient_passes'] == 0
    else:
        assert result['work']['full_gradient_passes'] == 3
        assert result['work']['numpy_expectation_passes'] == 0
    assert result['final_penalized_log_likelihood'] == pytest.approx(
        -numpy_loss(model, prefix, lengths) * int(lengths.sum()), abs=2e-10)


def test_import_is_owned_exact_and_rejects_boundary_or_bad_columns():
    model = make_model('factorized', 937101)
    initial = probabilities(model)
    head = model.cost_logits.detach().clone()
    assert import_probabilities(model, *initial) <= 1e-12
    for value, after in zip(initial, probabilities(model), strict=True):
        np.testing.assert_allclose(value, after, atol=1e-12, rtol=0)
    assert torch.equal(head, model.cost_logits)
    initial[0][0, 0, 0] = 0
    with pytest.raises(ValueError, match='strict'):
        import_probabilities(model, *initial)
    arrays = probabilities(model)
    arrays[1][0, 0] *= .5
    with pytest.raises(ValueError, match='normalized'):
        import_probabilities(model, *arrays)


def test_methods_start_from_identical_operators_without_shared_mutable_parameters():
    left, right = [make_model('factorized', 937101) for _ in range(2)]
    for a, b in zip(left.parameters(), right.parameters(), strict=True):
        assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()
    prefix, lengths = public_histories()
    original_right = [x.detach().clone() for x in right.parameters()]
    fit_prefix(left, prefix, lengths, 'em', 1)
    assert all(torch.equal(a, b) for a, b in zip(original_right, right.parameters(), strict=True))


def test_cannot_use_other_model_or_zero_prior_or_run_with_invalid_public_history():
    prefix, lengths = public_histories()
    with pytest.raises(ValueError, match='factorized'):
        fit_prefix(make_model('matched_free', 937101), prefix, lengths, 'em', 1)
    model = make_model('factorized', 937101)
    with pytest.raises(ValueError, match='prior'):
        fit_prefix(model, prefix, lengths, 'em', 1, pseudocount=0.)
    bad = prefix.clone()
    bad[4, -1, 11] = 1
    with pytest.raises(ValueError, match='padding'):
        fit_prefix(model, bad, lengths, 'em', 1)
