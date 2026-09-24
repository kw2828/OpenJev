"""Prospective public-history filters sharing (or separating) forecast operators.

The reset odor precedes actions and hazards. Its learned emission has four
outcomes per state and conditions a known uniform prior. Each subsequent
public action/ordinary-odor pair conditions a learned branch B. Prefix state
is normalized; inherited blind forecasts retain unnormalized surviving mass.
No oracle prefix argument is accepted by the filter public API. The fixed
cost basis and uniform prior remain privileged synthetic-world assumptions.

Filter parameters, states and arithmetic are CPU float64; public tokens are
float32. The GRU arm is exactly the existing learned_learned FactorModel.
Prefix and forecast operator softmaxes are computed separately, even when
tied, and recorded separately in work. Work records structural calls/rows,
not validation FLOPs, autograd work or a matched-compute claim. This module
contains no fitting, generator, file access or persistent recurrent state.
"""
from __future__ import annotations

import torch
from torch import nn

from openjev.research.finite_factor_models import OPERATOR_SEED_XOR, FactorModel
from openjev.research.otto_observation_operator_model import _finite, _work, require

VERSION = 'finite-shared-filter-model-v1'
ARMS = ('shared_filter', 'untied_filter', 'gru_prefix')
RESET_SEED_XOR = 0x85EBCA6B
PREFIX_WORK_KEYS = ('prefix_filter_calls', 'prefix_filter_rows',
                    'reset_emission_softmax_calls', 'reset_emission_rows',
                    'prefix_operator_softmax_calls')


class SharedFilterModel(FactorModel):
    """Eight-state learned prefix filtering with a fixed linear cost basis."""

    def __init__(self, arm, seed):
        require(type(arm) is str and arm in ARMS[:2], 'declared filter arm')
        require(type(seed) is int and 0 <= seed < 2**32, 'uint32 seed')
        nn.Module.__init__(self)
        self.arm, self.seed, self.kind, self.width = arm, seed, 'tied', 8
        self.learned_prefix = self.learned_operators = True
        self.operator_seed = seed ^ OPERATOR_SEED_XOR
        self.reset_seed = seed ^ RESET_SEED_XOR
        states = torch.arange(8, device='cpu')
        self.register_buffer('costs', torch.where(
            torch.arange(4, device='cpu')[:, None] == ((states ^ (states >> 1)) & 3),
            -.75, .25).to(torch.float64))
        reset_generator = torch.Generator(device='cpu').manual_seed(self.reset_seed)
        self.reset_logits = nn.Parameter(.05 * torch.randn(
            (4, 8), generator=reset_generator, device='cpu', dtype=torch.float64))
        operator_generator = torch.Generator(device='cpu').manual_seed(self.operator_seed)
        self.observed_logits = nn.Parameter(.05 * torch.randn(
            (4, 33, 8), generator=operator_generator, device='cpu', dtype=torch.float64))
        if arm == 'untied_filter':
            self.prefix_observed_logits = nn.Parameter(self.observed_logits.detach().clone())
        self._parameters_valid()

    def _parameters_valid(self):
        require(self.arm in ARMS[:2] and self.kind == 'tied' and self.width == 8
                and self.learned_prefix is True and self.learned_operators is True,
                'unchanged filter configuration')
        shapes = {'reset_logits': (4, 8), 'observed_logits': (4, 33, 8)}
        if self.arm == 'untied_filter':
            shapes['prefix_observed_logits'] = (4, 33, 8)
        named = dict(self.named_parameters())
        require(set(named) == set(shapes), 'no unused parameter baggage')
        for name, value in named.items():
            require(value.device.type == 'cpu' and value.layout == torch.strided
                    and value.dtype == torch.float64 and tuple(value.shape) == shapes[name]
                    and value.requires_grad, 'CPU float64 trainable parameter: ' + name)
            _finite(value, name)
        require(set(dict(self.named_buffers())) == {'costs'}, 'fixed cost buffer only')
        require(self.costs.device.type == 'cpu' and self.costs.dtype == torch.float64
                and tuple(self.costs.shape) == (4, 8) and not self.costs.requires_grad,
                'fixed CPU float64 cost basis')
        states = torch.arange(8, device='cpu')
        expected = torch.where(
            torch.arange(4, device='cpu')[:, None] == ((states ^ (states >> 1)) & 3),
            -.75, .25).to(torch.float64)
        require(torch.equal(self.costs, expected), 'unchanged fixed true cost readout')

    def _prefix_inputs(self, prefix, lengths):
        batch = super()._prefix_inputs(prefix, lengths)
        require(prefix.shape[1] <= 9, 'at most initial odor and eight public updates')
        _finite(prefix, 'entire prefix including padding')
        active = torch.arange(prefix.shape[1], device='cpu')[None] < lengths[:, None]
        require(bool((prefix[~active] == 0).all()), 'all-zero padded rows')
        rows = prefix[active]
        require(bool(((rows == 0) | (rows == 1)).all()), 'binary public tokens')
        require(bool((rows[:, 8] == 0).all()) and bool((rows[:, 10:] == 0).all()),
                'ordinary prefix odors and no hidden features')
        require(bool((rows[:, 4:8].sum(-1) == 1).all()), 'one ordinary odor per active row')
        initial = prefix[:, 0]
        require(bool((initial[:, :4] == 0).all()) and bool((initial[:, 9] == 1).all()),
                'reset row has initial odor, reset marker and no action')
        later = prefix[:, 1:][active[:, 1:]]
        require(bool((later[:, :4].sum(-1) == 1).all()) and bool((later[:, 9] == 0).all()),
                'later rows have one action and no reset marker')
        return batch

    def _initial(self, prefix, lengths, oracle_prefix, work):
        require(oracle_prefix is None, 'public filter must not receive oracle state')
        batch = self._prefix_inputs(prefix, lengths)
        for key in PREFIX_WORK_KEYS:
            work.setdefault(key, 0)
        reset_emission = self.reset_logits.softmax(0)
        work['reset_emission_softmax_calls'] += 1
        _finite(reset_emission, 'reset emission')
        require(bool((reset_emission > 0).all()), 'reset emission underflow rejected')
        initial_odors = prefix[:, 0, 4:8].argmax(-1)
        unnormalized = reset_emission[initial_odors] / 8
        evidence = unnormalized.sum(-1)
        require(bool((unnormalized > 0).all()) and bool((evidence > 0).all()),
                'positive initial emission mass and evidence')
        state = unnormalized / evidence[:, None]
        work['reset_emission_rows'] += batch
        self._state(state, normalized=True)
        # Separate normalization is intentional: tying parameters does not hide
        # the extra prefix softmax from the call accounting.
        logits = self.observed_logits if self.arm == 'shared_filter' else self.prefix_observed_logits
        probabilities = logits.softmax(1)
        work['prefix_operator_softmax_calls'] += 1
        _finite(probabilities, 'prefix branch probabilities')
        require(bool((probabilities > 0).all()), 'prefix operator underflow rejected')
        operators = probabilities[:, :-1].reshape(4, 4, 8, 8)
        require(bool(((operators.sum((1, 2)) + probabilities[:, -1] - 1).abs()
                      <= self._tolerance()).all()), 'stochastic prefix outgoing mass')
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

    def encode_prefix(self, prefix, lengths):
        return self._initial(prefix, lengths, None, _work())

    def blind_rollout(self, prefix, lengths, actions):
        return self._rollout(prefix, lengths, actions)

    def observed_rollout(self, prefix, lengths, actions, observations):
        require(observations is not None, 'explicit observations')
        return self._rollout(prefix, lengths, actions, observations)

    def forward(self, prefix, lengths, actions):
        return self.blind_rollout(prefix, lengths, actions)

    def parameter_metadata(self):
        self._parameters_valid()
        def records(values):
            return {name: {'shape': list(value.shape), 'count': value.numel(),
                           'dtype': str(value.dtype), 'bytes': value.numel() * value.element_size(),
                           'requires_grad': value.requires_grad} for name, value in values}
        parameters, buffers = records(self.named_parameters()), records(self.named_buffers())
        return {'version': VERSION, 'arm': self.arm, 'seed': self.seed, 'mass_width': 8,
                'parameters': parameters, 'buffers': buffers,
                'count': sum(row['count'] for row in parameters.values()),
                'trainable_count': sum(row['count'] for row in parameters.values()),
                'parameter_bytes': sum(row['bytes'] for row in parameters.values()),
                'buffer_bytes': sum(row['bytes'] for row in buffers.values()),
                'privileged_prefix': False, 'known_operators': False, 'fixed_cost_readout': True,
                'uniform_reset_prior': True, 'prefix_operator_shared': self.arm == 'shared_filter',
                'operator_initialization_seed': self.operator_seed,
                'reset_initialization_seed': self.reset_seed,
                'initialization': '.05 normal; local CPU generators, forecast seed XOR 0x9E3779B9, reset seed XOR 0x85EBCA6B; untied prefix clones forecast logits',
                'precision': 'float32 public tokens; float64 learned filter/operator/state/cost arithmetic',
                'scope': 'Actual storage and parameter counts; domain separation is not an independence claim. Call/row work excludes guards and autograd; no matched-compute claim.'}


def make_model(arm, seed):
    require(type(arm) is str and arm in ARMS, 'declared shared-filter study arm')
    if arm == 'gru_prefix':
        return FactorModel('learned_learned', seed)
    return SharedFilterModel(arm, seed)
