"""Explicit observation-regime reference for the fixed eight-state world.

This zero-parameter control is privileged, not a learned model. The constructor
owns the known branch, found and centered-cost buffers for epsilon .12 or .30.
A scalar epsilon buffer binds the declared regime into the ordinary state hash.
Only the prefix-boundary posterior is injected; future conditioning and blind
propagation use the frozen FactorModel methods without mutation or replacement.

The two worlds have the same transition, hazard and cost laws. Their blind
marginals agree up to floating-point summation, so blind-only parity does not
validate the regime. The independent audit must also check observed event laws
and posterior updates against the correct emission probabilities.
"""
from __future__ import annotations

import torch
from torch import nn

from openjev.research.finite_factor_models import OPERATOR_SEED_XOR, FactorModel
from openjev.research.finite_observation_world import world
from openjev.research.otto_observation_operator_model import _finite, _tensor, require

VERSION = 'finite-regime-reference-v1'
EPSILONS = (.12, .30)


class RegimeReference(FactorModel):
    def __init__(self, epsilon):
        require(type(epsilon) is float and epsilon in EPSILONS, 'explicit epsilon .12 or .30')
        nn.Module.__init__(self)
        self.arm, self.seed, self.kind, self.width = 'exact_exact', 0, 'tied', 8
        self.operator_seed = OPERATOR_SEED_XOR
        self.learned_prefix = self.learned_operators = False
        self.epsilon = epsilon
        known = world(epsilon)
        self.register_buffer('costs', torch.from_numpy(known['costs'].copy()))
        self.register_buffer('known_observed', torch.from_numpy(known['B'].copy()))
        self.register_buffer('known_found', torch.from_numpy(known['found'].copy()))
        self.register_buffer('regime_epsilon', torch.tensor(epsilon, dtype=torch.float64, device='cpu'))
        self._parameters_valid()

    def _parameters_valid(self):
        require(self.arm == 'exact_exact' and self.seed == 0 and self.kind == 'tied' and self.width == 8
                and self.operator_seed == OPERATOR_SEED_XOR
                and self.learned_prefix is False and self.learned_operators is False,
                'immutable exact reference configuration')
        require(type(self.epsilon) is float and self.epsilon in EPSILONS,
                'declared reference observation regime')
        require(not list(self.named_parameters()) and not list(self.named_children()),
                'reference has no trainable or unused modules')
        shapes = {'costs': (4, 8), 'known_observed': (4, 4, 8, 8),
                  'known_found': (4, 8), 'regime_epsilon': ()}
        require(set(dict(self.named_buffers())) == set(shapes), 'exact reference buffer roster')
        for name, value in self.named_buffers():
            _tensor(value, torch.float64, shapes[name], name)
            _finite(value, name)
            require(not value.requires_grad, 'fixed reference buffers')
        require(float(self.regime_epsilon) == self.epsilon, 'epsilon metadata must match saved state')
        states = torch.arange(8, device='cpu')
        expected = torch.where(torch.arange(4, device='cpu')[:, None] == ((states ^ (states >> 1)) & 3),
                               -.75, .25).to(torch.float64)
        require(torch.equal(self.costs, expected), 'fixed true centered cost buffer')

    def parameter_metadata(self):
        record = super().parameter_metadata()
        return {**record, 'version': VERSION, 'epsilon': self.epsilon,
                'known_operator_source': f'finite_observation_world.world(epsilon={self.epsilon})',
                'operator_initialization': 'deterministic exact world buffers; no random draw',
                'regime_state_buffer': 'regime_epsilon',
                'scope': 'Privileged zero-parameter reference only; actual CPU float64 storage. '
                         'No learned-prefix claim and no oracle update after the prefix boundary.'}


def make_reference(epsilon):
    return RegimeReference(epsilon)
