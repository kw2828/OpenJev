"""Expected-count and gradient prefix fitting for an unchanged recurrent model.

Both methods optimize total public-prefix log likelihood plus pseudocount
times the sum of logs of T/O and both hazard outcomes. The gradient method
divides the negative objective by the fixed valid-event denominator. No cost
head or hidden-world target enters these updates. Conversions preserve the
actual probabilities; no clipping or latent-state alignment is performed.

This module supplies fixed-update engineering primitives, not an admitted
scientific schedule or a claim that an EM sweep and an Adam step cost equally.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from openjev.research.finite_expected_count import expected_counts, map_update, tokens_from_prefix
from openjev.research.finite_factorized_dynamics_models import FactorizedDynamicsModel, prefix_predictions

DYNAMICS = ('transition_logits', 'emission_logits', 'hazard_logits')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_model(model):
    require(type(model) is FactorizedDynamicsModel and model.study_arm == 'factorized',
            'unchanged factorized recurrent model required')
    model._parameters_valid()


def probabilities(model):
    validate_model(model)
    with torch.no_grad():
        arrays = (model.transition_logits.softmax(1).numpy().copy(),
                  model.emission_logits.softmax(0).numpy().copy(),
                  model.hazard_logits.sigmoid().numpy().copy())
    require(all(np.isfinite(x).all() and ((x > 0) & (x < 1)).all() for x in arrays),
            'strict probabilities without repair')
    return arrays


def import_probabilities(model, transition, emission, hazard):
    validate_model(model)
    values = (transition, emission, hazard)
    shapes = ((4, 8, 8), (4, 8), (4, 8))
    for value, shape in zip(values, shapes, strict=True):
        require(isinstance(value, np.ndarray) and value.dtype == np.float64
                and value.shape == shape and np.isfinite(value).all()
                and bool(((value > 0) & (value < 1)).all()), 'strict finite probability import')
    require(np.max(np.abs(transition.sum(1) - 1)) <= 1e-12
            and np.max(np.abs(emission.sum(0) - 1)) <= 1e-12,
            'normalized probabilities; no clipping or renormalization')
    logits = (np.log(transition), np.log(emission), np.log(hazard) - np.log1p(-hazard))
    require(all(np.isfinite(x).all() for x in logits), 'finite exact log transforms')
    head = model.cost_logits.detach().clone()
    with torch.no_grad():
        for name, value in zip(DYNAMICS, logits, strict=True):
            getattr(model, name).copy_(torch.from_numpy(value))
    actual = probabilities(model)
    error = max(float(np.max(np.abs(a - b))) for a, b in zip(actual, values, strict=True))
    require(error <= 1e-12, 'probability logit roundtrip')
    require(torch.equal(head, model.cost_logits), 'pretraining preserves cost-head bytes')
    return error


def torch_objective(model, prefix, lengths, pseudocount):
    validate_model(model)
    require(type(pseudocount) is float and math.isfinite(pseudocount) and pseudocount > 0,
            'positive finite prior count')
    result = prefix_predictions(model, prefix, lengths)
    log_prior = pseudocount * (
        model.transition_logits.log_softmax(1).sum()
        + model.emission_logits.log_softmax(0).sum()
        + torch.nn.functional.logsigmoid(model.hazard_logits).sum()
        + torch.nn.functional.logsigmoid(-model.hazard_logits).sum())
    nll = result['nll'].sum()
    events = int(lengths.sum())
    loss = (nll - log_prior) / events
    require(bool(torch.isfinite(loss)), 'finite likelihood-plus-prior objective')
    return {'loss': loss, 'log_likelihood': -nll, 'log_prior': log_prior,
            'valid_events': events, 'work': result['work']}


def numpy_objective(transition, emission, hazard, counts, pseudocount):
    log_prior = pseudocount * (np.log(transition).sum() + np.log(emission).sum()
                              + np.log(hazard).sum() + np.log1p(-hazard).sum())
    value = float(counts['log_likelihood'] + log_prior)
    require(math.isfinite(value), 'finite expected-count penalized objective')
    return value


def fit_prefix(model, prefix, lengths, method, updates, *, pseudocount=.001,
               learning_rate=.003, gradient_clip=5., check=lambda: None):
    """Fixed passes, every update retained, no best-checkpoint selection.

Inputs are public CPU tensors, never an oracle state or future target. EM
retains its real post-update logit representation for the next E-step. Gradient
uses full-batch Adam; its objective need not improve at each step. EM's
monotonicity check is for this penalized prefix objective, not the later joint
control objective. All diagnostic/count passes are charged separately.
"""
    validate_model(model)
    require(method in ('em', 'gradient') and type(updates) is int and 1 <= updates <= 10000,
            'declared method and positive bounded updates')
    require(type(learning_rate) is float and math.isfinite(learning_rate) and learning_rate > 0
            and type(gradient_clip) is float and math.isfinite(gradient_clip) and gradient_clip > 0,
            'positive finite gradient settings')
    require(type(pseudocount) is float and math.isfinite(pseudocount) and pseudocount > 0,
            'positive finite prior count')
    start = time.perf_counter()
    check()
    # The Torch route validates the complete public token contract first.
    with torch.no_grad():
        original = torch_objective(model, prefix, lengths, pseudocount)
    tokens = tokens_from_prefix(prefix.detach().numpy(), lengths.detach().numpy())
    head = model.cost_logits.detach().clone()
    history = []
    work = {'full_gradient_passes': 0, 'torch_diagnostic_passes': 1,
            'numpy_expectation_passes': 0, 'map_updates': 0, 'adam_updates': 0,
            'probability_imports': 0, 'valid_events': int(lengths.sum()),
            'attempts': len(prefix), 'gradient_event_exposures': 0,
            'expectation_event_exposures': 0}
    parameters = [getattr(model, name) for name in DYNAMICS]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate) if method == 'gradient' else None

    def expectation():
        check()
        t, o, h = probabilities(model)
        result = expected_counts(t, o, h, tokens['actions'], tokens['observations'],
                                 tokens['lengths'], check=check)
        work['numpy_expectation_passes'] += 1
        work['expectation_event_exposures'] += work['valid_events']
        return (t, o, h), result

    for index in range(updates):
        check()
        began = time.perf_counter()
        before = None
        roundtrip_error = 0.
        if method == 'em':
            values, result = expectation()
            before = numpy_objective(*values, result, pseudocount)
            updated = map_update(*values, result['counts'], pseudocount=pseudocount)
            work['map_updates'] += 1
            roundtrip_error = import_probabilities(model, updated['transition'],
                                                  updated['emission'], updated['hazard'])
            work['probability_imports'] += 1
            actual, after_counts = expectation()
            after = numpy_objective(*actual, after_counts, pseudocount)
            require(after >= before - 1e-9, 'penalized EM likelihood decreased beyond roundoff')
        else:
            optimizer.zero_grad(set_to_none=True)
            result = torch_objective(model, prefix, lengths, pseudocount)
            before = float((result['log_likelihood'] + result['log_prior']).detach())
            result['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
            require(bool(torch.isfinite(norm)), 'finite gradient norm')
            optimizer.step()
            work['full_gradient_passes'] += 1
            work['adam_updates'] += 1
            work['gradient_event_exposures'] += work['valid_events']
            with torch.no_grad():
                diagnostic = torch_objective(model, prefix, lengths, pseudocount)
            work['torch_diagnostic_passes'] += 1
            after = float(diagnostic['log_likelihood'] + diagnostic['log_prior'])
        require(torch.equal(head, model.cost_logits), 'unchanged cost head at every prefix update')
        probabilities(model)
        history.append({'update': index + 1, 'penalized_log_likelihood_before': before,
                        'penalized_log_likelihood_after': after,
                        'roundtrip_max_abs': roundtrip_error, 'seconds': time.perf_counter() - began})
        check()
    with torch.no_grad():
        final = torch_objective(model, prefix, lengths, pseudocount)
    work['torch_diagnostic_passes'] += 1
    return {'method': method, 'updates': updates, 'pseudocount': pseudocount,
            'initial_penalized_log_likelihood': float(original['log_likelihood'] + original['log_prior']),
            'final_penalized_log_likelihood': float(final['log_likelihood'] + final['log_prior']),
            'history': history, 'work': work, 'seconds': time.perf_counter() - start,
            'head_unchanged': torch.equal(head, model.cost_logits),
            'compute_matched': False, 'hidden_state_input': False}
