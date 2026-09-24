"""Prospective rounded transport with original and matched free controls.

All three arms retain the frozen factorized emission/hazard model, public
prefix filter, absorbing recurrence and learned linear cost head. Only the
transition parameterization or its initialization changes. T[a,next,current]
is column softmax for the free arms and the fixed four-sweep, positive-slack,
rank-one rounding algorithm for rounded. Rounding T does not balance the surviving
operator diag(1-h)T. It neither prevents uniform collapse nor proves memory
retention. The constraint supplies task-derived structure, not a discovery.

The original constructor creates exactly the same352 CPU float64 parameters
using its local seeded streams. matched_free replaces only its T logits by
log of the rounded initial T, then verifies its column softmax within1e-12.
This is function matching, not gradient or compute matching. The ideal
doubly-stochastic manifold has196 transition degrees versus224 for the free
columns; computed rounded output only satisfies its measured residual bound.
The1e-8 slack is a declared part of the function, including a small smoothing
change on already balanced matrices. This is not an infinite Sinkhorn limit.

Prefix likelihood and its optional prior share one constructed set of fields.
The rounded prior uses the returned logT from that same graph, never a column
softmax substitute. Other prior logs retain the original arithmetic and have
explicit extra counters. Joint prefix NLL has no prior. Head computation,
constructor matching, diagnostic exports and each prefix/forecast construction
are separate counted work. Counters describe forward operations, not FLOPs,
autograd or validation. The kernel's named diagnostic sums are counted;
unlisted scalar extrema/diagnostics are not. No tensors are cached between calls.

The fixed rank-one correction is always executed, never selected after a
failure. Failure triggers no fallback, extra correction or added sweeps. The
last route's mutable work is available as model.last_work; raised exceptions receive an owned copy
as rounded_model_work (also on constructor failure). Constructor and metadata
work are fixed after initialization; last_work is not part of model metadata.
"""
from __future__ import annotations

import math

import torch

from openjev.research.finite_cost_readout_models import HEAD_WORK_KEYS, _prefix_inputs
from openjev.research.finite_factorized_dynamics_models import (
    FACTOR_WORK_KEYS,
    FactorizedDynamicsModel,
)
from openjev.research.finite_rounded_transition import SLACK, SWEEPS, TOLERANCE, rounded_transition
from openjev.research.finite_rounded_transition import WORK_KEYS as KERNEL_WORK_KEYS
from openjev.research.finite_shared_filter_models import PREFIX_WORK_KEYS
from openjev.research.otto_observation_operator_model import WORK_KEYS as BASE_WORK_KEYS
from openjev.research.otto_observation_operator_model import _finite, require

VERSION = 'finite-rounded-model-v1'
ARMS = ('original_free', 'matched_free', 'rounded')
DYNAMICS = ('transition_logits', 'emission_logits', 'hazard_logits')
ADAPTER_WORK_KEYS = (
    'probability_field_calls', 'adapter_check_calls',
    'prior_transition_log_softmax_calls', 'prior_emission_log_softmax_calls',
    'prior_hazard_logsigmoid_calls', 'prior_normalization_vectors',
    'prior_normalization_entries', 'prior_logsigmoid_entries',
    'matching_log_calls', 'matching_log_entries', 'matching_copy_calls', 'matching_copy_entries',
    'matching_verification_softmax_calls', 'matching_verification_probability_columns',
    'privileged_prefix_rows', 'event_probability_rows', 'prefix_probability_rows', 'prefix_nll_rows',
)
WORK_KEYS = (*BASE_WORK_KEYS, *PREFIX_WORK_KEYS, *HEAD_WORK_KEYS, *FACTOR_WORK_KEYS,
             *KERNEL_WORK_KEYS, *ADAPTER_WORK_KEYS)


def work_counts(work=None):
    """Initialize every route counter, preserving a supplied accumulator."""
    if work is None:
        work = {}
    require(type(work) is dict and all(type(value) is int and value >= 0 for value in work.values()),
            'work is a dictionary of nonnegative Python integers')
    for key in WORK_KEYS:
        work.setdefault(key, 0)
    return work


def _failure(error, work):
    error.rounded_model_work = dict(work)


def _validate_model(model):
    require(type(model) is RoundedDynamicsModel, 'declared rounded-dynamics adapter only')
    model._parameters_valid()


class RoundedDynamicsModel(FactorizedDynamicsModel):
    """Same checkpoint tensor roster, with an explicit transport arm."""

    def __init__(self, arm, seed, *, check=lambda: None):
        require(type(arm) is str and arm in ARMS, 'declared rounded-transport arm')
        require(callable(check), 'callable external check')
        self.transport_arm, self.check = arm, check
        constructor_work = work_counts()
        try:
            constructor_work['adapter_check_calls'] += 1
            check()
            super().__init__('factorized', seed)
            self.construction_work = constructor_work
            self.last_work = constructor_work
            if arm == 'matched_free':
                with torch.no_grad():
                    fields = rounded_transition(self.transition_logits, check=check, work=constructor_work)
                    target = fields['transition']
                    constructor_work['matching_log_calls'] += 1
                    constructor_work['matching_log_entries'] += 256
                    logits = target.log()
                    _finite(logits, 'matched initial log transition')
                    constructor_work['matching_copy_calls'] += 1
                    constructor_work['matching_copy_entries'] += 256
                    self.transition_logits.copy_(logits)
                    constructor_work['matching_verification_softmax_calls'] += 1
                    constructor_work['matching_verification_probability_columns'] += 32
                    actual = self.transition_logits.softmax(1)
                    _finite(actual, 'matched initial transition')
                    require(bool(((actual - target).abs() <= TOLERANCE).all()),
                            'initial matched-free transition differs beyond1e-12')
            self._parameters_valid()
        except BaseException as error:
            _failure(error, constructor_work)
            raise

    def _parameters_valid(self):
        require(self.transport_arm in ARMS and callable(self.check) and self.study_arm == 'factorized',
                'unchanged rounded transport and inherited factorized configuration')
        super()._parameters_valid()

    def _fields(self, work):
        self.last_work = work_counts(work)
        try:
            self._parameters_valid()
            work['probability_field_calls'] += 1
            work['adapter_check_calls'] += 1
            self.check()
            log_transition, diagnostics = None, None
            if self.transport_arm == 'rounded':
                result = rounded_transition(self.transition_logits, check=self.check, work=work)
                transition, log_transition = result['transition'], result['log_transition']
                diagnostics = result['diagnostics']
            else:
                work['transition_softmax_calls'] += 1
                work['transition_probability_columns'] += 32
                transition = self.transition_logits.softmax(1)
            work['emission_softmax_calls'] += 1
            work['emission_probability_columns'] += 8
            emission = self.emission_logits.softmax(0)
            work['hazard_sigmoid_calls'] += 1
            work['hazard_probability_entries'] += 32
            hazard = self.hazard_logits.sigmoid()
            for value, name in ((transition, 'transition'), (emission, 'emission'), (hazard, 'hazard')):
                _finite(value, name)
                require(bool(((value > 0) & (value < 1)).all()),
                        'strict factor probability bounds; no clipping: ' + name)
            return {'transition': transition, 'log_transition': log_transition,
                    'emission': emission, 'hazard': hazard, 'diagnostics': diagnostics, 'work': work}
        except BaseException as error:
            _failure(error, work)
            raise

    def _factorize_fields(self, fields, work):
        try:
            work['factorization_calls'] += 1
            transition, emission, hazard = fields['transition'], fields['emission'], fields['hazard']
            work['factor_product_entries'] += 256
            surviving = (1 - hazard[:, :, None]) * transition
            work['factor_product_entries'] += 1024
            observed = emission[None, :, :, None] * surviving[:, None]
            work['factor_product_entries'] += 256
            found_products = hazard[:, :, None] * transition
            work['factor_found_reduction_columns'] += 32
            found = found_products.sum(1)
            for value, name in ((surviving, 'survival products'), (observed, 'observed products'),
                                (found_products, 'found products'), (found, 'found mass')):
                _finite(value, name)
                require(bool((value > 0).all()), 'factor product underflow rejected: ' + name)
            require(bool(((observed.sum((1, 2)) + found - 1).abs() <= self._tolerance()).all()),
                    'stochastic factored outgoing mass')
            return emission, observed, found
        except BaseException as error:
            _failure(error, work)
            raise

    def _prefix_components(self, work):
        return self._factorize_fields(self._fields(work), work)

    def _operators(self, work):
        _emission, observed, found = self._prefix_components(work)
        marginal = observed.sum(1)
        work['operator_marginal_sum_calls'] += 1
        return {'observed': observed, 'found': found, 'blind': marginal, 'blind_found': found}, marginal

    def _initial(self, prefix, lengths, oracle_prefix, work):
        self.last_work = work_counts(work)
        try:
            return super()._initial(prefix, lengths, oracle_prefix, work)
        except BaseException as error:
            _failure(error, work)
            raise

    def _rollout(self, prefix, lengths, actions, observations=None, *, oracle_prefix=None):
        self.last_work = work_counts()
        try:
            return super()._rollout(prefix, lengths, actions, observations, oracle_prefix=oracle_prefix)
        except BaseException as error:
            _failure(error, self.last_work)
            raise

    def probability_fields(self, *, work=None):
        return probability_fields(self, work=work)

    def parameter_metadata(self):
        metadata = super().parameter_metadata()
        metadata.update({
            'version': VERSION, 'arm': self.transport_arm, 'transport_arm': self.transport_arm,
            'inherited_dynamics_kind': 'factorized', 'transition_stored_parameters': 256,
            'transition_ideal_constraint_degrees': 196 if self.transport_arm == 'rounded' else 224,
            'transition_constraint': 'fixed four sweeps, slack contractions and rank-one rounding'
                                     if self.transport_arm == 'rounded' else 'column softmax',
            'transition_degrees_scope': 'Ideal normalized manifolds; finite rounded output has measured residuals.',
            'rounded_sweeps': SWEEPS if self.transport_arm == 'rounded' else None,
            'rounded_slack': SLACK if self.transport_arm == 'rounded' else None,
            'rounded_tolerance': TOLERANCE if self.transport_arm == 'rounded' else None,
            'matched_initialization': self.transport_arm == 'matched_free',
            'scope': 'Same352 stored parameters. Rounded T is a task-derived structural prior, not a retention guarantee. '
                     'Initial function matching does not match gradients, optimization geometry or compute. '
                     'Forward, prior, constructor and snapshot normalization work is disclosed separately; '
                     'backward and validation operations are not FLOP counts.',
        })
        return metadata


def make_model(arm, seed, *, check=lambda: None):
    return RoundedDynamicsModel(arm, seed, check=check)


def probability_fields(model, *, work=None):
    """Owned differentiable T/O/h. Free arms do not compute unused logT.

    log_transition and diagnostics are None for free arms. A supplied work
    accumulator and model.last_work retain operations on failure. This export
    is separate from dynamics_snapshot and incurs one actual field construction.
    """
    work = work_counts(work)
    try:
        _validate_model(model)
        return model._fields(work)
    except BaseException as error:
        _failure(error, work)
        raise


def _prefix_result(model, prefix, lengths, work):
    """Frozen prefix arithmetic, with fields retained locally for the prior."""
    _validate_model(model)
    active = _prefix_inputs(prefix, lengths)
    batch = len(prefix)
    fields = model._fields(work)
    emission, operators, found = model._factorize_fields(fields, work)
    reset_law = emission.sum(-1) / 8
    require(bool((reset_law.sum() - 1).abs() <= model._tolerance()), 'normalized reset event law')
    probabilities = [torch.cat((reset_law, reset_law.new_zeros(1))).expand(batch, 5).clone()]
    labels = prefix[:, 0, 4:9].argmax(-1)
    evidence = reset_law[labels]
    require(bool((evidence > 0).all()), 'positive reset observed probability')
    nll = [-evidence.log()]
    initial_mass = emission[labels] / 8
    require(bool((initial_mass > 0).all()), 'reset mass product underflow rejected')
    state = initial_mass / evidence[:, None]
    model._state(state, normalized=True)
    work['reset_emission_rows'] += batch
    work['prefix_probability_rows'] += batch
    work['prefix_nll_rows'] += batch
    event_law = torch.cat((operators.sum(2), found[:, None]), 1)
    require(bool(((event_law.sum(1) - 1).abs() <= model._tolerance()).all()), 'stochastic event columns')
    for step in range(1, 9):
        lanes = torch.nonzero(active[:, step], as_tuple=False).flatten()
        row_probabilities = operators.new_zeros((batch, 5))
        row_nll = operators.new_zeros(batch)
        if lanes.numel():
            actions = prefix[lanes, step, :4].argmax(-1)
            labels = prefix[lanes, step, 4:9].argmax(-1)
            predicted = model._multiply(event_law[actions], state[lanes])
            require(bool(((predicted.sum(-1) - 1).abs() <= model._tolerance()).all()), 'unit predictive event mass')
            observed = predicted.gather(1, labels[:, None]).squeeze(1)
            require(bool((observed > 0).all()), 'positive observed probability; no clipping')
            row_probabilities = row_probabilities.index_copy(0, lanes, predicted)
            row_nll = row_nll.index_copy(0, lanes, -observed.log())
            ordinary = torch.nonzero(labels < 4, as_tuple=False).flatten()
            updated = state.new_zeros((len(lanes), 8))
            if ordinary.numel():
                branch = model._multiply(operators[actions[ordinary], labels[ordinary]], state[lanes[ordinary]])
                branch_mass = branch.sum(-1)
                require(bool((branch_mass > 0).all()), 'positive conditioning evidence')
                posterior = branch / branch_mass[:, None]
                model._state(posterior, normalized=True)
                updated = updated.index_copy(0, ordinary, posterior)
                work['prefix_filter_calls'] += 1
                work['prefix_filter_rows'] += len(ordinary)
            state = state.index_copy(0, lanes, updated)
            work['prefix_probability_rows'] += len(lanes)
            work['prefix_nll_rows'] += len(lanes)
        probabilities.append(row_probabilities)
        nll.append(row_nll)
    probabilities, nll = torch.stack(probabilities, 1), torch.stack(nll, 1)
    _finite(probabilities, 'prefix predictions')
    _finite(nll, 'prefix negative log-likelihood')
    return {'probabilities': probabilities, 'nll': nll, 'work': work}, fields


def prefix_predictions(model, prefix, lengths):
    """Pre-label reset/action probabilities and NLL; found suffix padding zero."""
    work = work_counts()
    try:
        _validate_model(model)
        model.last_work = work
        result, _fields = _prefix_result(model, prefix, lengths, work)
        return result
    except BaseException as error:
        _failure(error, work)
        raise


def torch_objective(model, prefix, lengths, pseudocount):
    """Original total prefix likelihood plus prior, divided by valid events.

    Only this prefix-pretraining helper includes the positive pseudocount.
    The rounded transition prior shares forward's normalized logT. Free-arm
    log-softmax and common O/h logs retain the frozen objective operation order.
    No cost head is called; every extra prior normalization is counted here.
    """
    work = work_counts()
    try:
        _validate_model(model)
        model.last_work = work
        require(type(pseudocount) is float and math.isfinite(pseudocount) and pseudocount > 0,
                'positive finite prior count')
        result, fields = _prefix_result(model, prefix, lengths, work)
        if model.transport_arm == 'rounded':
            log_transition = fields['log_transition']
        else:
            work['prior_transition_log_softmax_calls'] += 1
            work['prior_normalization_vectors'] += 32
            work['prior_normalization_entries'] += 256
            log_transition = model.transition_logits.log_softmax(1)
        work['prior_emission_log_softmax_calls'] += 1
        work['prior_normalization_vectors'] += 8
        work['prior_normalization_entries'] += 32
        log_emission = model.emission_logits.log_softmax(0)
        work['prior_hazard_logsigmoid_calls'] += 1
        work['prior_logsigmoid_entries'] += 32
        log_hazard = torch.nn.functional.logsigmoid(model.hazard_logits)
        work['prior_hazard_logsigmoid_calls'] += 1
        work['prior_logsigmoid_entries'] += 32
        log_surviving = torch.nn.functional.logsigmoid(-model.hazard_logits)
        log_prior = pseudocount * (log_transition.sum() + log_emission.sum()
                                  + log_hazard.sum() + log_surviving.sum())
        nll = result['nll'].sum()
        events = int(lengths.sum())
        loss = (nll - log_prior) / events
        _finite(loss, 'likelihood-plus-prior objective')
        return {'loss': loss, 'log_likelihood': -nll, 'log_prior': log_prior,
                'valid_events': events, 'work': work}
    except BaseException as error:
        _failure(error, work)
        raise
