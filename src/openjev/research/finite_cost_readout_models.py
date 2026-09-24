"""Prospective fixed versus trainable bounded linear cost readouts.

All arms share the same reset and branch-operator initialization and recurrence.
Only the learned arm adds 32 logits. Its columns are .25-softmax(logits), with
the softmax over decisions, initialized from .9*onehot+.025. Fixed softened
costs are .9 times the known exact basis. Known world-aligned initialization
remains privileged; this is not unstructured task learning or state recovery.

The real-valued learned costs are strictly interior to (-.75,.25). Float64
subtraction may round a cost to a boundary while its probability remains
strictly positive; bounded roundoff is accepted without clipping. Probability
underflow or saturation to exactly zero/one is still rejected.
The bias-free cost matrix acts linearly on normalized or surviving mass. It
does not renormalize blind mass and gives exactly zero for absorbed mass.
All learned arithmetic is CPU float64; public tokens remain float32. Every
actual learned-head softmax in a rollout is charged, including repeated calls
at different horizons. readout_matrix() is a separate diagnostic export;
callers must record those calls outside forward work. No numerical clipping,
files, data collection or optimization occurs here.
"""
from __future__ import annotations

import torch
from torch import nn

from openjev.research.finite_factor_models import OPERATOR_SEED_XOR
from openjev.research.finite_shared_filter_models import (
    PREFIX_WORK_KEYS,
    RESET_SEED_XOR,
    SharedFilterModel,
)
from openjev.research.otto_observation_operator_model import _finite, _tensor, _work, require

VERSION = 'finite-cost-readout-model-v1'
ARMS = ('fixed_exact', 'fixed_softened', 'learned_readout')
DELTA = .1
HEAD_WORK_KEYS = ('cost_head_softmax_calls', 'cost_head_probability_rows')


def _preferred():
    states = torch.arange(8, device='cpu')
    return ((states ^ (states >> 1)) & 3)


def _exact_costs():
    return torch.where(torch.arange(4, device='cpu')[:, None] == _preferred(),
                       -.75, .25).to(torch.float64)


class CostReadoutModel(SharedFilterModel):
    """Frozen shared-filter recurrence with an explicitly chosen readout arm."""

    def __init__(self, arm, seed):
        require(type(arm) is str and arm in ARMS, 'declared cost-readout arm')
        require(type(seed) is int and 0 <= seed < 2**32, 'uint32 seed')
        nn.Module.__init__(self)
        self.readout_arm, self.arm, self.seed = arm, 'shared_filter', seed
        self.kind, self.width = 'tied', 8
        self.learned_prefix = self.learned_operators = True
        self.operator_seed, self.reset_seed = seed ^ OPERATOR_SEED_XOR, seed ^ RESET_SEED_XOR
        reset_generator = torch.Generator(device='cpu').manual_seed(self.reset_seed)
        self.reset_logits = nn.Parameter(.05 * torch.randn(
            (4, 8), generator=reset_generator, device='cpu', dtype=torch.float64))
        operator_generator = torch.Generator(device='cpu').manual_seed(self.operator_seed)
        self.observed_logits = nn.Parameter(.05 * torch.randn(
            (4, 33, 8), generator=operator_generator, device='cpu', dtype=torch.float64))
        if arm == 'learned_readout':
            onehot = (torch.arange(4, device='cpu')[:, None] == _preferred()).to(torch.float64)
            self.cost_logits = nn.Parameter(((1 - DELTA) * onehot + DELTA / 4).log())
        else:
            self.register_buffer('costs', _exact_costs() * (1 if arm == 'fixed_exact' else 1 - DELTA))
        self._parameters_valid()

    def _parameters_valid(self):
        require(self.readout_arm in ARMS and self.arm == 'shared_filter' and self.kind == 'tied'
                and self.width == 8 and self.learned_prefix is True and self.learned_operators is True,
                'unchanged cost-readout configuration')
        shapes = {'reset_logits': (4, 8), 'observed_logits': (4, 33, 8)}
        if self.readout_arm == 'learned_readout':
            shapes['cost_logits'] = (4, 8)
        named = dict(self.named_parameters())
        require(set(named) == set(shapes), 'no unused parameter baggage')
        for name, value in named.items():
            _tensor(value, torch.float64, shapes[name], name)
            _finite(value, name)
            require(value.requires_grad, 'declared trainable parameters')
        if self.readout_arm == 'learned_readout':
            require(not dict(self.named_buffers()), 'learned arm has no unused cost buffer')
        else:
            require(set(dict(self.named_buffers())) == {'costs'}, 'fixed cost buffer only')
            _tensor(self.costs, torch.float64, (4, 8), 'fixed costs')
            require(not self.costs.requires_grad, 'fixed costs cannot train')
            expected = _exact_costs() * (1 if self.readout_arm == 'fixed_exact' else 1 - DELTA)
            require(torch.equal(self.costs, expected), 'unchanged declared fixed cost basis')

    def _matrix(self, work):
        for key in HEAD_WORK_KEYS:
            work.setdefault(key, 0)
        if self.readout_arm != 'learned_readout':
            return self.costs.clone()
        probabilities = self.cost_logits.softmax(0)
        work['cost_head_softmax_calls'] += 1
        work['cost_head_probability_rows'] += 8
        _finite(probabilities, 'cost-head probabilities')
        require(bool(((probabilities > 0) & (probabilities < 1)).all()), 'strict cost-head probability bounds; no clipping')
        costs = .25 - probabilities
        tolerance = self._tolerance()
        require(bool(((costs >= -.75 - tolerance) & (costs <= .25 + tolerance)).all()),
                'finite-precision learned cost bounds; no clipping')
        require(bool((costs.sum(0).abs() <= self._tolerance()).all()), 'centered learned cost columns')
        return costs

    def readout_matrix(self):
        """Owned differentiable matrix; one learned-arm diagnostic softmax call."""
        self._parameters_valid()
        return self._matrix({})

    def _read(self, state, work):
        value = state @ self._matrix(work).T
        _finite(value, 'linear cost moment')
        work['cost_readout_calls'] += 1
        work['cost_readout_rows'] += len(state)
        return value

    def parameter_metadata(self):
        self._parameters_valid()
        def records(values):
            return {name: {'shape': list(value.shape), 'count': value.numel(), 'dtype': str(value.dtype),
                           'bytes': value.numel() * value.element_size(), 'requires_grad': value.requires_grad}
                    for name, value in values}
        parameters, buffers = records(self.named_parameters()), records(self.named_buffers())
        return {'version': VERSION, 'arm': self.readout_arm, 'seed': self.seed, 'mass_width': 8,
                'parameters': parameters, 'buffers': buffers,
                'count': sum(row['count'] for row in parameters.values()),
                'trainable_count': sum(row['count'] for row in parameters.values()),
                'parameter_bytes': sum(row['bytes'] for row in parameters.values()),
                'buffer_bytes': sum(row['bytes'] for row in buffers.values()),
                'privileged_prefix': False, 'known_operators': False,
                'fixed_cost_readout': self.readout_arm != 'learned_readout',
                'privileged_readout_initialization': True, 'uniform_reset_prior': True,
                'prefix_operator_shared': True, 'readout_delta': DELTA,
                'head_logits': 32 if self.readout_arm == 'learned_readout' else 0,
                'head_identifiable_degrees': 24 if self.readout_arm == 'learned_readout' else 0,
                'operator_initialization_seed': self.operator_seed, 'reset_initialization_seed': self.reset_seed,
                'precision': 'float32 public tokens; float64 filter, operator, state and readout arithmetic',
                'scope': 'Actual parameter and buffer bytes. Forward head softmaxes counted per readout; diagnostic exports separate. No matched-compute, identification or unprivileged-initialization claim.'}


def make_model(arm, seed):
    return CostReadoutModel(arm, seed)


def _prefix_inputs(prefix, lengths):
    require(isinstance(prefix, torch.Tensor) and prefix.ndim == 3, 'prefix [N,9,31]')
    batch = len(prefix)
    require(batch > 0, 'nonempty prediction batch')
    _tensor(prefix, torch.float32, (batch, 9, 31), 'all-attempt prefix')
    _tensor(lengths, torch.int64, (batch,), 'all-attempt lengths')
    require(bool(((lengths >= 2) & (lengths <= 9)).all()), 'initial odor plus at least one action event')
    _finite(prefix, 'all prefix rows including padding')
    active = torch.arange(9, device='cpu')[None] < lengths[:, None]
    require(bool((prefix[~active] == 0).all()), 'zero padding')
    rows = prefix[active]
    require(bool(((rows == 0) | (rows == 1)).all()) and bool((rows[:, 10:] == 0).all()),
            'binary public tokens only')
    require(bool((rows[:, 4:9].sum(-1) == 1).all()), 'one event per active row')
    initial = prefix[:, 0]
    require(bool((initial[:, :4] == 0).all()) and bool((initial[:, 8] == 0).all())
            and bool((initial[:, 9] == 1).all()), 'ordinary reset odor without action or hazard')
    later = prefix[:, 1:][active[:, 1:]]
    require(bool((later[:, :4].sum(-1) == 1).all()) and bool((later[:, 9] == 0).all()),
            'one action per later row without reset')
    found = prefix[:, :, 8] == 1
    last = torch.arange(9, device='cpu')[None] == lengths[:, None] - 1
    require(not bool((found & ~last).any()) and bool((found.any(-1) | (lengths == 9)).all()),
            'only first-found termination can shorten prefix')
    return active


def prefix_predictions(model, prefix, lengths):
    """Return pre-label probabilities[N,9,5], event NLL[N,9] and work.

    Reset found probability and every padded output are exact zero. Valid NLL
    uses the actual recorded event; it never clips probabilities or averages
    by a case's realized length. Full five-event predictions precede each
    action's observation. Returned tensors retain gradients and own storage.
    """
    require(type(model) is CostReadoutModel, 'declared cost-readout model only')
    model._parameters_valid()
    active = _prefix_inputs(prefix, lengths)
    batch = len(prefix)
    work = {**_work(), **dict.fromkeys(PREFIX_WORK_KEYS, 0), **dict.fromkeys(HEAD_WORK_KEYS, 0),
            'prefix_probability_rows': 0, 'prefix_nll_rows': 0}
    emission = model.reset_logits.softmax(0)
    work['reset_emission_softmax_calls'] += 1
    _finite(emission, 'reset emission')
    require(bool((emission > 0).all()), 'reset emission underflow rejected')
    reset_law = emission.sum(-1) / 8
    require(bool((reset_law.sum() - 1).abs() <= model._tolerance()), 'normalized reset event law')
    probabilities = [torch.cat((reset_law, reset_law.new_zeros(1))).expand(batch, 5).clone()]
    initial_labels = prefix[:, 0, 4:9].argmax(-1)
    evidence = reset_law[initial_labels]
    require(bool((evidence > 0).all()), 'positive reset observed probability')
    nll = [-evidence.log()]
    initial_mass = emission[initial_labels] / 8
    require(bool((initial_mass > 0).all()), 'reset mass product underflow rejected')
    state = initial_mass / evidence[:, None]
    model._state(state, normalized=True)
    work['reset_emission_rows'] += batch
    work['prefix_probability_rows'] += batch
    work['prefix_nll_rows'] += batch
    columns = model.observed_logits.softmax(1)
    work['prefix_operator_softmax_calls'] += 1
    _finite(columns, 'prefix operators')
    require(bool((columns > 0).all()), 'prefix operator underflow rejected')
    operators, found = columns[:, :-1].reshape(4, 4, 8, 8), columns[:, -1]
    event_law = torch.cat((operators.sum(2), found[:, None]), 1)
    require(bool(((event_law.sum(1) - 1).abs() <= model._tolerance()).all()), 'stochastic event columns')
    for step in range(1, 9):
        lanes = torch.nonzero(active[:, step], as_tuple=False).flatten()
        row_probabilities = columns.new_zeros((batch, 5))
        row_nll = columns.new_zeros(batch)
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
    return {'probabilities': probabilities, 'nll': nll, 'work': work}
