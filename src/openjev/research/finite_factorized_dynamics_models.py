"""Prospective factorized dynamics and initially function-matched free controls.

B[a,o,n,s] = O[o,n] * (1-h[a,n]) * T[a,n,s]. O also supplies the
reset emission, without transition or hazard. A=sum_o B, and the found row
is sum_n h*T. These are task-derived conditional independences, not learned
structure. No true dynamics, hazard, emission values or oracle state enter
construction. The unchanged learned cost head has privileged world-aligned
initialization. Its linear readout acts on surviving mass without normalization.

All trainable arithmetic is CPU float64; public tokens are float32. Each
factorization uses 256 survival products, 1024 observation products and 256
found products, followed by 32 found-column reductions. Prefix and forecast
construct their probabilities separately. Counts describe executed structural
work, not guards, backward FLOPs or matched compute. Diagnostic snapshots have
separate work records. Matched-free construction performs one factorization
and two log transforms outside forward work; no model is secretly constructed.

Local streams use seed XOR 0x9E3779B9 (T/dense B), XOR 0x85EBCA6B (O/reset),
and XOR 0xC2B2AE35 (hazard). Domain separation is not statistical independence.
The generic hazard center -log(32) matches the flat 33-way initializer, not
the world's hazards. No fitting, files, generation or persistent state lives
here. Existing recurrence, cost head and input guards remain unchanged.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from openjev.research.finite_cost_readout_models import (
    DELTA,
    HEAD_WORK_KEYS,
    CostReadoutModel,
    _preferred,
    _prefix_inputs,
)
from openjev.research.finite_factor_models import OPERATOR_SEED_XOR, FactorModel
from openjev.research.finite_shared_filter_models import PREFIX_WORK_KEYS, RESET_SEED_XOR
from openjev.research.otto_observation_operator_model import _finite, _tensor, _work, require

VERSION = 'finite-factorized-dynamics-model-v1'
ARMS = ('factorized', 'matched_free', 'dense_free')
HAZARD_SEED_XOR = 0xC2B2AE35
FACTOR_WORK = {
    'factorization_calls': 1,
    'transition_softmax_calls': 1,
    'transition_probability_columns': 32,
    'emission_softmax_calls': 1,
    'emission_probability_columns': 8,
    'hazard_sigmoid_calls': 1,
    'hazard_probability_entries': 32,
    'factor_product_entries': 1536,
    'factor_found_reduction_columns': 32,
}
FACTOR_WORK_KEYS = tuple(FACTOR_WORK)


def _extra_work(work):
    for key in (*PREFIX_WORK_KEYS, *HEAD_WORK_KEYS, *FACTOR_WORK_KEYS):
        work.setdefault(key, 0)
    return work


def _normal(shape, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return .05 * torch.randn(shape, generator=generator, device='cpu', dtype=torch.float64)


def _factorize(transition_logits, emission_logits, hazard_logits, work, tolerance):
    transition = transition_logits.softmax(1)
    emission = emission_logits.softmax(0)
    hazard = hazard_logits.sigmoid()
    for value, name in ((transition, 'transition'), (emission, 'emission'), (hazard, 'hazard')):
        _finite(value, name)
        require(bool(((value > 0) & (value < 1)).all()), 'strict factor probability bounds; no clipping: ' + name)
    surviving = (1 - hazard[:, :, None]) * transition
    observed = emission[None, :, :, None] * surviving[:, None]
    found_products = hazard[:, :, None] * transition
    found = found_products.sum(1)
    for value, name in ((surviving, 'survival products'), (observed, 'observed products'),
                        (found_products, 'found products'), (found, 'found mass')):
        _finite(value, name)
        require(bool((value > 0).all()), 'factor product underflow rejected: ' + name)
    require(bool(((observed.sum((1, 2)) + found - 1).abs() <= tolerance).all()),
            'stochastic factored outgoing mass')
    for key, amount in FACTOR_WORK.items():
        work[key] = work.get(key, 0) + amount
    return emission, observed, found


class FactorizedDynamicsModel(CostReadoutModel):
    """Same mass recurrence and learned head, three explicit dynamics arms."""

    def __init__(self, arm, seed):
        require(type(arm) is str and arm in ARMS, 'declared factorized-dynamics arm')
        require(type(seed) is int and 0 <= seed < 2**32, 'uint32 seed')
        nn.Module.__init__(self)
        self.study_arm, self.seed = arm, seed
        self.arm, self.readout_arm, self.kind, self.width = 'shared_filter', 'learned_readout', 'tied', 8
        self.learned_prefix = self.learned_operators = True
        self.operator_seed, self.reset_seed = seed ^ OPERATOR_SEED_XOR, seed ^ RESET_SEED_XOR
        self.hazard_seed = seed ^ HAZARD_SEED_XOR
        self.construction_work = dict.fromkeys(FACTOR_WORK_KEYS, 0)
        self.construction_work['matching_log_calls'] = 0
        self.construction_work['matching_log_entries'] = 0
        if arm == 'dense_free':
            self.reset_logits = nn.Parameter(_normal((4, 8), self.reset_seed))
            self.observed_logits = nn.Parameter(_normal((4, 33, 8), self.operator_seed))
        else:
            transition = _normal((4, 8, 8), self.operator_seed)
            emission = _normal((4, 8), self.reset_seed)
            hazard = -math.log(32) + _normal((4, 8), self.hazard_seed)
            if arm == 'factorized':
                self.transition_logits = nn.Parameter(transition)
                self.emission_logits = nn.Parameter(emission)
                self.hazard_logits = nn.Parameter(hazard)
            else:
                reset, branches, found = _factorize(
                    transition, emission, hazard, self.construction_work, self._tolerance())
                self.reset_logits = nn.Parameter(reset.log())
                self.observed_logits = nn.Parameter(torch.cat((branches.reshape(4, 32, 8), found[:, None]), 1).log())
                self.construction_work['matching_log_calls'] = 2
                self.construction_work['matching_log_entries'] = 1088
        onehot = (torch.arange(4, device='cpu')[:, None] == _preferred()).to(torch.float64)
        self.cost_logits = nn.Parameter(((1 - DELTA) * onehot + DELTA / 4).log())
        self._parameters_valid()

    def _parameters_valid(self):
        require(self.study_arm in ARMS and self.arm == 'shared_filter' and self.readout_arm == 'learned_readout'
                and self.kind == 'tied' and self.width == 8
                and self.learned_prefix is True and self.learned_operators is True,
                'unchanged factorized-dynamics configuration')
        shapes = {'cost_logits': (4, 8)}
        shapes.update({'transition_logits': (4, 8, 8), 'emission_logits': (4, 8), 'hazard_logits': (4, 8)}
                      if self.study_arm == 'factorized' else {'reset_logits': (4, 8), 'observed_logits': (4, 33, 8)})
        require(set(dict(self.named_parameters())) == set(shapes) and not dict(self.named_buffers()),
                'exact trainable roster and no unused buffers')
        for name, value in self.named_parameters():
            _tensor(value, torch.float64, shapes[name], name)
            _finite(value, name)
            require(value.requires_grad, 'all declared parameters trainable')

    def _prefix_components(self, work):
        self._parameters_valid()
        _extra_work(work)
        if self.study_arm == 'factorized':
            return _factorize(self.transition_logits, self.emission_logits, self.hazard_logits, work, self._tolerance())
        emission = self.reset_logits.softmax(0)
        work['reset_emission_softmax_calls'] += 1
        _finite(emission, 'reset emission')
        require(bool((emission > 0).all()), 'reset emission underflow rejected')
        columns = self.observed_logits.softmax(1)
        work['prefix_operator_softmax_calls'] += 1
        _finite(columns, 'prefix operators')
        require(bool((columns > 0).all()), 'prefix operator underflow rejected')
        return emission, columns[:, :-1].reshape(4, 4, 8, 8), columns[:, -1]

    def _operators(self, work):
        self._parameters_valid()
        _extra_work(work)
        if self.study_arm != 'factorized':
            return FactorModel._operators(self, work)
        _emission, observed, found = _factorize(
            self.transition_logits, self.emission_logits, self.hazard_logits, work, self._tolerance())
        marginal = observed.sum(1)
        work['operator_marginal_sum_calls'] += 1
        return {'observed': observed, 'found': found, 'blind': marginal, 'blind_found': found}, marginal

    def _initial(self, prefix, lengths, oracle_prefix, work):
        require(oracle_prefix is None, 'public filter must not receive oracle state')
        batch = self._prefix_inputs(prefix, lengths)
        emission, operators, _found = self._prefix_components(work)
        labels = prefix[:, 0, 4:8].argmax(-1)
        unnormalized = emission[labels] / 8
        evidence = unnormalized.sum(-1)
        require(bool((unnormalized > 0).all()) and bool((evidence > 0).all()), 'positive reset mass and evidence')
        state = unnormalized / evidence[:, None]
        self._state(state, normalized=True)
        work['reset_emission_rows'] += batch
        for step in range(1, prefix.shape[1]):
            lanes = torch.nonzero(lengths > step, as_tuple=False).flatten()
            if lanes.numel():
                actions = prefix[lanes, step, :4].argmax(-1)
                odors = prefix[lanes, step, 4:8].argmax(-1)
                branch = self._multiply(operators[actions, odors], state[lanes])
                evidence = branch.sum(-1)
                require(bool((evidence > 0).all()), 'positive prefix evidence')
                conditioned = branch / evidence[:, None]
                self._state(conditioned, normalized=True)
                state = state.index_copy(0, lanes, conditioned)
                work['prefix_filter_calls'] += 1
                work['prefix_filter_rows'] += len(lanes)
        return state

    def dynamics_snapshot(self):
        """Owned differentiable dynamics; count this work outside rollout routes."""
        work = _extra_work(_work())
        emission, observed, found = self._prefix_components(work)
        blind = observed.sum(1)
        work['operator_marginal_sum_calls'] += 1
        return {'reset_emission': emission.clone(), 'observed': observed.clone(),
                'found': found.clone(), 'blind': blind.clone(), 'work': work}

    def parameter_metadata(self):
        self._parameters_valid()
        records = {name: {'shape': list(value.shape), 'count': value.numel(),
            'dtype': str(value.dtype), 'bytes': value.numel() * value.element_size(), 'requires_grad': value.requires_grad}
            for name, value in self.named_parameters()}
        return {'version': VERSION, 'arm': self.study_arm, 'seed': self.seed, 'mass_width': 8,
            'parameters': records, 'buffers': {}, 'count': sum(r['count'] for r in records.values()),
            'trainable_count': sum(r['count'] for r in records.values()),
            'parameter_bytes': sum(r['bytes'] for r in records.values()), 'buffer_bytes': 0,
            'privileged_prefix': False, 'known_operators': False, 'fixed_cost_readout': False,
            'privileged_readout_initialization': True, 'uniform_reset_prior': True,
            'prefix_operator_shared': True, 'reset_emission_shared': self.study_arm == 'factorized',
            'head_logits': 32, 'head_identifiable_degrees': 24, 'readout_delta': DELTA,
            'operator_initialization_seed': self.operator_seed, 'reset_initialization_seed': self.reset_seed,
            'hazard_initialization_seed': self.hazard_seed if self.study_arm != 'dense_free' else None,
            'construction_work': self.construction_work.copy(),
            'precision': 'float32 public tokens; float64 filter, operator, state and readout arithmetic',
            'scope': 'Conditional structure, capacity and parameterization are bundled. Matching functions does not match gradients. Structural counters exclude guards/autograd and constructor draws; no matched-compute claim.'}


def make_model(arm, seed):
    return FactorizedDynamicsModel(arm, seed)


def prefix_predictions(model, prefix, lengths):
    """Predict reset/action events before conditioning; zero terminal padding."""
    require(type(model) is FactorizedDynamicsModel, 'declared factorized-dynamics model only')
    model._parameters_valid()
    active = _prefix_inputs(prefix, lengths)
    batch = len(prefix)
    work = _extra_work(_work())
    work.update({'prefix_probability_rows': 0, 'prefix_nll_rows': 0})
    emission, operators, found = model._prefix_components(work)
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
    return {'probabilities': probabilities, 'nll': nll, 'work': work}
