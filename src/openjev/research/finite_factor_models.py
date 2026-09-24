"""Prospective prefix/operator factorial with a fixed true-world cost readout.

Each arm name is prefix_operator. Exact prefixes are explicit privileged
probabilities at the prefix boundary only. No future oracle state is injected.
Learned prefixes reject privileged input. Exact operators are buffers derived
once at construction from finite_observation_world.world(.12); learned operators
use a local .05-normal stream with seed XOR 0x9E3779B9. Prefix initialization
uses the unmodified seed. This domain separation preserves cross-arm pairing;
it is not a mathematical independence claim. Blind A always sums B.

This subclass reuses the qualified pure mass validation, propagation and
conditioning methods. It constructs only the modules that an arm actually uses:
no learned readout, unused prefix network or independent blind operator. A
zero-parameter exact_exact reference must not be passed to an optimizer.
Prefix encoding is CPU float32; projection, operators and fixed cost moments
are float64. Parameter counts do not imply matched compute. Known operators
and cost semantics are privileged world structure, not learned discoveries.
"""
from __future__ import annotations

import torch
from torch import nn

from openjev.research.finite_observation_world import world
from openjev.research.otto_observation_operator_model import (
    ObservationOperatorModel,
    _finite,
    _tensor,
    _work,
    require,
)

ARMS = ('exact_exact', 'learned_exact', 'exact_learned', 'learned_learned')
VERSION = 'finite-factor-model-v1'
OPERATOR_SEED_XOR = 0x9E3779B9


class FactorModel(ObservationOperatorModel):
    def __init__(self, arm, seed):
        require(type(arm) is str and arm in ARMS, 'declared factorial arm')
        require(type(seed) is int and 0 <= seed < 2**32, 'uint32 seed')
        nn.Module.__init__(self)
        self.arm, self.seed, self.kind, self.width = arm, seed, 'tied', 8
        self.operator_seed = seed ^ OPERATOR_SEED_XOR
        self.learned_prefix = arm.startswith('learned_')
        self.learned_operators = arm.endswith('_learned')
        states = torch.arange(8, device='cpu')
        costs = torch.where(torch.arange(4, device='cpu')[:, None] == ((states ^ (states >> 1)) & 3),
                            -.75, .25).to(torch.float64)
        self.register_buffer('costs', costs)
        if self.learned_prefix:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                self.assimilation = nn.GRUCell(31, 28, device='cpu', dtype=torch.float32)
                self.projection = nn.Linear(28, 8, device='cpu', dtype=torch.float64)
        if self.learned_operators:
            generator = torch.Generator(device='cpu').manual_seed(self.operator_seed)
            self.observed_logits = nn.Parameter(.05 * torch.randn(
                (4, 33, 8), generator=generator, device='cpu', dtype=torch.float64))
        else:
            known = world(.12)
            self.register_buffer('known_observed', torch.from_numpy(known['B'].copy()))
            self.register_buffer('known_found', torch.from_numpy(known['found'].copy()))
        self._parameters_valid()

    def _parameters_valid(self):
        require(self.arm in ARMS and self.kind == 'tied' and self.width == 8
                and self.learned_prefix == self.arm.startswith('learned_')
                and self.learned_operators == self.arm.endswith('_learned'), 'unchanged factorial configuration')
        shapes = {}
        if self.learned_prefix:
            shapes.update({'assimilation.weight_ih': (84, 31), 'assimilation.weight_hh': (84, 28),
                           'assimilation.bias_ih': (84,), 'assimilation.bias_hh': (84,),
                           'projection.weight': (8, 28), 'projection.bias': (8,)})
        if self.learned_operators:
            shapes['observed_logits'] = (4, 33, 8)
        named = dict(self.named_parameters())
        require(set(named) == set(shapes), 'no unused parameter baggage')
        for name, parameter in named.items():
            dtype = torch.float32 if name.startswith('assimilation.') else torch.float64
            _tensor(parameter, dtype, shapes[name], name)
            _finite(parameter, name)
            require(parameter.requires_grad, 'all declared parameters are trainable')
        buffers = {'costs': (4, 8)}
        if not self.learned_operators:
            buffers.update({'known_observed': (4, 4, 8, 8), 'known_found': (4, 8)})
        require(set(dict(self.named_buffers())) == set(buffers), 'exact fixed buffer roster')
        for name, buffer in self.named_buffers():
            _tensor(buffer, torch.float64, buffers[name], name)
            _finite(buffer, name)
            require(not buffer.requires_grad, 'fixed buffers have no gradient')
        states = torch.arange(8, device='cpu')
        expected = torch.where(torch.arange(4, device='cpu')[:, None] == ((states ^ (states >> 1)) & 3),
                               -.75, .25).to(torch.float64)
        require(torch.equal(self.costs, expected), 'unchanged fixed true cost readout')

    def _operators(self, work):
        self._parameters_valid()
        if self.learned_operators:
            probabilities = self.observed_logits.softmax(dim=1)
            work['operator_observed_softmax_calls'] += 1
            _finite(probabilities, 'learned operator probabilities')
            require(bool((probabilities > 0).all()), 'operator underflow rejected')
            observed, found = probabilities[:, :-1].reshape(4, 4, 8, 8), probabilities[:, -1]
        else:
            observed, found = self.known_observed.clone(), self.known_found.clone()
            require(bool((observed > 0).all()) and bool((found > 0).all()), 'positive known world operators')
        marginal = observed.sum(1)
        work['operator_marginal_sum_calls'] += 1
        require(bool(((marginal.sum(-2) + found - 1).abs() <= self._tolerance()).all()), 'stochastic outgoing mass')
        return {'observed': observed, 'found': found, 'blind': marginal, 'blind_found': found}, marginal

    def _read(self, state, work):
        value = state @ self.costs.T
        _finite(value, 'fixed linear cost moment')
        work['cost_readout_calls'] += 1
        work['cost_readout_rows'] += len(state)
        return value

    def _initial(self, prefix, lengths, oracle_prefix, work):
        batch = self._prefix_inputs(prefix, lengths)
        if self.learned_prefix:
            require(oracle_prefix is None, 'learned prefix must not receive oracle state')
            return self._encode(prefix, lengths, work)
        require(oracle_prefix is not None, 'exact prefix requires explicit privileged boundary state')
        _tensor(oracle_prefix, torch.float64, (batch, 8), 'oracle prefix')
        mass = self._state(oracle_prefix, normalized=True)
        require(bool(((mass - 1).abs() <= self._tolerance()).all()), 'nonterminal normalized oracle prefix')
        work['privileged_prefix_rows'] += batch
        return oracle_prefix.detach().clone()

    def encode_prefix(self, prefix, lengths, *, oracle_prefix=None):
        work = {**_work(), 'privileged_prefix_rows': 0, 'event_probability_rows': 0}
        return self._initial(prefix, lengths, oracle_prefix, work)

    def _rollout(self, prefix, lengths, actions, observations=None, *, oracle_prefix=None):
        horizon = self._rollout_inputs(prefix, lengths, actions, observations)
        work = {**_work(), 'privileged_prefix_rows': 0, 'event_probability_rows': 0}
        initial = self._initial(prefix, lengths, oracle_prefix, work)
        operators, marginal = self._operators(work)
        event_law = torch.cat((operators['observed'].sum(2), operators['found'][:, None]), 1)
        state, records, events = initial, [], []
        for step in range(horizon):
            if observations is None:
                result = self._prior(state, actions[:, step], marginal, operators['found'], work)
                state = result['prior_state']
            else:
                probabilities = self._multiply(event_law[actions[:, step]], state)
                terminal = torch.zeros_like(probabilities)
                terminal[:, 4] = 1
                probabilities = torch.where((state.sum(-1) == 0)[:, None], terminal, probabilities)
                require(bool(((probabilities.sum(-1) - 1).abs() <= self._tolerance()).all()), 'full event probability mass')
                events.append(probabilities)
                work['event_probability_rows'] += len(state)
                result = self._observed(state, actions[:, step], observations[:, step], operators, marginal, work)
                state = result['posterior_state']
            records.append(result)
        result = {'prefix_state': initial, 'prior_states': torch.stack([r['prior_state'] for r in records], 1),
                  'posterior_states': None if observations is None else torch.stack([r['posterior_state'] for r in records], 1),
                  'cost_contrasts': torch.stack([r['cost_contrasts'] for r in records], 1),
                  'survival_mass': torch.stack([r['survival_mass'] for r in records], 1),
                  'found_increments': torch.stack([r['found_increment'] for r in records], 1),
                  'evidence': None if observations is None else torch.stack([r['evidence'] for r in records], 1),
                  'work': work}
        if observations is not None:
            result['probabilities'] = torch.stack(events, 1)
        return result

    def blind_rollout(self, prefix, lengths, actions, *, oracle_prefix=None):
        return self._rollout(prefix, lengths, actions, oracle_prefix=oracle_prefix)

    def observed_rollout(self, prefix, lengths, actions, observations, *, oracle_prefix=None):
        require(observations is not None, 'explicit observations')
        return self._rollout(prefix, lengths, actions, observations, oracle_prefix=oracle_prefix)

    def forward(self, prefix, lengths, actions, *, oracle_prefix=None):
        return self.blind_rollout(prefix, lengths, actions, oracle_prefix=oracle_prefix)

    def parameter_metadata(self):
        self._parameters_valid()
        def records(values):
            return {name: {'shape': list(value.shape), 'count': value.numel(), 'dtype': str(value.dtype),
                           'bytes': value.numel() * value.element_size(), 'requires_grad': value.requires_grad}
                    for name, value in values}
        parameters, buffers = records(self.named_parameters()), records(self.named_buffers())
        return {'version': VERSION, 'arm': self.arm, 'seed': self.seed, 'mass_width': 8,
                'parameters': parameters, 'buffers': buffers,
                'count': sum(row['count'] for row in parameters.values()),
                'trainable_count': sum(row['count'] for row in parameters.values() if row['requires_grad']),
                'parameter_bytes': sum(row['bytes'] for row in parameters.values()),
                'buffer_bytes': sum(row['bytes'] for row in buffers.values()),
                'privileged_prefix': not self.learned_prefix, 'known_operators': not self.learned_operators,
                'fixed_cost_readout': True, 'known_operator_source': 'finite_observation_world.world(epsilon=0.12)',
                'prefix_initialization_seed': self.seed if self.learned_prefix else None,
                'operator_initialization_seed': self.operator_seed if self.learned_operators else None,
                'operator_initialization': '.05 normal, local CPU generator seed = fit seed XOR 0x9E3779B9; domain-separated pairing, not an independence claim',
                'scope': 'Actual storage; no unused modules. Mixed precision, privileged inputs and unequal counts preclude a matched-compute claim.'}


def make_model(arm, seed):
    return FactorModel(arm, seed)
