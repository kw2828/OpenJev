"""Causal observation-reliability filters over a frozen finite-state backbone.

The joint mass retains noise-mode/state correlations. A bank reset mixes only
the noise mode and preserves the state marginal. Recurrent gates use previous
memory; the current observation enters memory only after its predictive law
has been returned and its joint posterior formed. No noise labels or target
states are accepted. Found is absorbing. Reset has no transition or hazard.

Only numerical memory features use a log floor. Predictive probabilities,
posterior normalizers and blind mass are never clipped or repaired. Learned
reliability is an effective correction to a learned observation model, not a
claim that the true noise parameter has been identified.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

VERSION = 'finite-reliability-filter-v1'
ARMS = ('unchanged', 'global', 'static_bank', 'markov_bank', 'recurrent_bank', 'reset_bank')
Q_VALUES = (0., 2. / 7., 4. / 7.)
FEATURE_LOG_FLOOR = 1e-12
TOLERANCE = 1e-12
FIELD_SHAPES = {'transition': (4, 8, 8), 'emission': (4, 8),
                'hazard': (4, 8), 'costs': (4, 8)}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tensor(value, dtype, shape, name):
    _require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
             and value.dtype == dtype and tuple(value.shape) == tuple(shape),
             'CPU tensor with declared shape and dtype: ' + name)


def _finite(value, name):
    _require(bool(torch.isfinite(value).all()), 'finite ' + name)


class ReliabilityFilter(nn.Module):
    """Six fixed interfaces sharing a copied, non-trainable T/O/h/C backbone.

    ``state_count`` counts joint probability entries plus recurrent memory,
    excluding the Boolean terminal flag and temporary predictive quantities.
    In particular, a three-mode bank stores24 joint entries, not just8+3.
    """

    def __init__(self, fields: Mapping[str, torch.Tensor], arm: str, seed: int):
        super().__init__()
        _require(type(arm) is str and arm in ARMS, 'declared reliability arm')
        _require(type(seed) is int and 0 <= seed < 2**32, 'uint32 local seed')
        _require(isinstance(fields, Mapping) and set(fields) == set(FIELD_SHAPES),
                 'exact four frozen backbone fields')
        for name, shape in FIELD_SHAPES.items():
            value = fields[name]
            _tensor(value, torch.float64, shape, name)
            _finite(value, name)
            if name != 'costs':
                _require(bool(((value >= 0) & (value <= 1)).all()),
                         'probabilities in closed unit interval: ' + name)
            self.register_buffer(name, value.detach().clone())
        _require(bool(((self.transition.sum(1) - 1).abs() <= TOLERANCE).all()),
                 'column-stochastic transition')
        _require(bool(((self.emission.sum(0) - 1).abs() <= TOLERANCE).all()),
                 'column-stochastic emission')
        self.arm, self.seed = arm, seed
        self.mode_count = 1 if arm in ('unchanged', 'global') else 3
        self.memory_size = 4 if arm in ('recurrent_bank', 'reset_bank') else 0
        if arm == 'global':
            self.global_logit = nn.Parameter(torch.zeros((), dtype=torch.float64))
        if arm in ('markov_bank', 'recurrent_bank', 'reset_bank'):
            self.prior_logits = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        if arm == 'markov_bank':
            self.reset_logit = nn.Parameter(torch.tensor(-4., dtype=torch.float64))
        if self.memory_size:
            # All initialization draws are local and restored on context exit.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.gru = nn.GRUCell(7, 4, device='cpu', dtype=torch.float64)
                self.reset_head = nn.Linear(4, 1, device='cpu', dtype=torch.float64)
                nn.init.zeros_(self.reset_head.weight)
                nn.init.constant_(self.reset_head.bias, -4.)

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def state_count(self):
        return 8 * self.mode_count + self.memory_size

    def parameter_metadata(self):
        return {'version': VERSION, 'arm': self.arm, 'seed': self.seed,
                'parameter_count': self.parameter_count, 'state_count': self.state_count,
                'mode_count': self.mode_count, 'memory_size': self.memory_size,
                'dtype': 'float64', 'device': 'cpu',
                'feature_log_floor': FEATURE_LOG_FLOOR,
                'q_values': None if self.arm == 'global' else list(Q_VALUES[:self.mode_count]),
                'global_q': '(4/7)*sigmoid(global_logit)' if self.arm == 'global' else None,
                'frozen_buffer_entries': sum(buffer.numel() for buffer in self.buffers()),
                'frozen_buffer_bytes': sum(buffer.numel() * buffer.element_size() for buffer in self.buffers()),
                'state_scope': 'Joint mode/state mass and recurrent memory; excludes terminal flag and temporaries.'}

    def _components(self):
        for name, parameter in self.named_parameters():
            _require(parameter.device.type == 'cpu' and parameter.dtype == torch.float64,
                     'CPU float64 parameters: ' + name)
            _finite(parameter, name)
        if self.arm == 'global':
            q = ((4. / 7.) * self.global_logit.sigmoid()).reshape(1)
        else:
            q = self.emission.new_tensor(Q_VALUES[:self.mode_count])
        emissions = (1 - q[:, None, None]) * self.emission[None] + q[:, None, None] / 4
        prior = (self.prior_logits.softmax(0) if hasattr(self, 'prior_logits')
                 else self.emission.new_full((self.mode_count,), 1. / self.mode_count))
        return emissions, prior

    def _update_memory(self, hidden, conditional_likelihood, mode_posterior, evidence):
        if not self.memory_size:
            return hidden
        features = torch.cat((conditional_likelihood.clamp_min(FEATURE_LOG_FLOOR).log(),
                              mode_posterior,
                              -evidence.clamp_min(FEATURE_LOG_FLOOR).log()[:, None]), dim=1)
        _finite(features, 'memory features')
        previous = torch.zeros_like(hidden) if self.arm == 'reset_bank' else hidden
        result = self.gru(features, previous)
        _finite(result, 'updated memory')
        return result

    def _reset_rates(self, hidden, batch):
        if self.arm == 'markov_bank':
            return self.reset_logit.sigmoid().expand(batch)
        if self.memory_size:
            return self.reset_head(hidden).squeeze(-1).sigmoid()
        return self.emission.new_zeros(batch)

    def _initial(self, labels, emissions, prior):
        batch = len(labels)
        joint = prior[None, :, None].expand(batch, -1, 8) / 8
        by_mode = emissions.sum(-1) / 8
        probabilities = (prior[:, None] * by_mode).sum(0).expand(batch, 4)
        probabilities = torch.cat((probabilities, probabilities.new_zeros(batch, 1)), dim=1)
        likelihood = emissions[:, labels, :].permute(1, 0, 2)
        branches = joint * likelihood
        evidence = branches.sum((1, 2))
        _require(bool((evidence > 0).all()), 'possible reset observations')
        posterior = branches / evidence[:, None, None]
        hidden = self.emission.new_zeros(batch, self.memory_size)
        conditional = by_mode[:, labels].transpose(0, 1)
        hidden = self._update_memory(hidden, conditional, posterior.sum(2), evidence)
        return posterior, hidden, probabilities

    def _step(self, joint, hidden, actions, labels, emissions, prior):
        """One nonabsorbed step. Current labels cannot affect returned rates/law."""
        batch = len(actions)
        rates = self._reset_rates(hidden, batch)
        mixed = ((1 - rates[:, None, None]) * joint
                 + rates[:, None, None] * prior[None, :, None] * joint.sum(1)[:, None, :])
        mode_prior = mixed.sum(2)
        moved = torch.einsum('bij,bkj->bki', self.transition[actions], mixed)
        surviving = moved * (1 - self.hazard[actions])[:, None, :]
        odor_mass = torch.einsum('bks,kos->bko', surviving, emissions)
        found_mass = (moved * self.hazard[actions][:, None, :]).sum(2)
        probabilities = torch.cat((odor_mass.sum(1), found_mass.sum(1)[:, None]), dim=1)
        _finite(probabilities, 'predictive event probabilities')
        _require(bool((probabilities >= 0).all())
                 and bool(((probabilities.sum(1) - 1).abs() <= TOLERANCE).all()),
                 'normalized predictive event law without repair')
        found = labels == 4
        # Index0 is a harmless placeholder on found lanes, never scored as odor.
        odor_labels = torch.where(found, torch.zeros_like(labels), labels)
        branches = surviving * emissions[:, odor_labels, :].permute(1, 0, 2)
        observed_mode_mass = torch.where(found[:, None], found_mass, branches.sum(2))
        evidence = observed_mode_mass.sum(1)
        _require(bool((evidence > 0).all()), 'possible action observations')
        posterior = branches / evidence[:, None, None]
        posterior = torch.where(found[:, None, None], torch.zeros_like(posterior), posterior)
        denominator = torch.where(mode_prior > 0, mode_prior, torch.ones_like(mode_prior))
        conditional = torch.where(mode_prior > 0, observed_mode_mass / denominator,
                                  torch.zeros_like(mode_prior))
        hidden = self._update_memory(hidden, conditional, observed_mode_mass / evidence[:, None], evidence)
        hidden = torch.where(found[:, None], torch.zeros_like(hidden), hidden)
        return posterior, hidden, probabilities, rates

    def forward(self, actions, observations):
        """Process reset plus T public events, predicting each before conditioning.

        Inputs are CPU int64 actions[B,T] and observations[B,T+1]. Reset is an
        odor0..3; found4 terminates and all later event labels must be4. Actions
        after found remain declared0..3 but cannot affect any output. Returned
        post_states are normalized state marginals or exact zeros after found.
        """
        _require(isinstance(actions, torch.Tensor) and actions.ndim == 2,
                 'actions[B,T]')
        batch, horizon = actions.shape
        _require(batch > 0, 'nonempty batch')
        _tensor(actions, torch.int64, (batch, horizon), 'actions')
        _tensor(observations, torch.int64, (batch, horizon + 1), 'observations')
        _require(bool(((actions >= 0) & (actions < 4)).all()), 'actions0..3')
        _require(bool(((observations >= 0) & (observations <= 4)).all())
                 and bool((observations[:, 0] < 4).all()), 'reset odor0..3; later events0..4')
        found = observations == 4
        _require(bool((~(found.cumsum(1) > 0) | found).all()), 'found followed only by absorbing labels')
        emissions, prior = self._components()
        joint, hidden, probability = self._initial(observations[:, 0], emissions, prior)
        states, probabilities, rates = [joint.sum(1)], [probability], []
        alive = torch.ones(batch, dtype=torch.bool, device='cpu')
        for step in range(horizon):
            lanes = torch.nonzero(alive, as_tuple=False).flatten()
            next_joint, next_hidden = torch.zeros_like(joint), torch.zeros_like(hidden)
            probability = self.emission.new_zeros(batch, 5)
            probability[:, 4] = 1
            rate = self.emission.new_zeros(batch)
            if lanes.numel():
                result = self._step(joint[lanes], hidden[lanes], actions[lanes, step],
                                    observations[lanes, step + 1], emissions, prior)
                next_joint = next_joint.index_copy(0, lanes, result[0])
                next_hidden = next_hidden.index_copy(0, lanes, result[1])
                probability = probability.index_copy(0, lanes, result[2])
                rate = rate.index_copy(0, lanes, result[3])
            joint, hidden = next_joint, next_hidden
            alive = alive & (observations[:, step + 1] != 4)
            states.append(joint.sum(1)); probabilities.append(probability); rates.append(rate)
        post_states = torch.stack(states, dim=1)
        result = {'probabilities': torch.stack(probabilities, dim=1),
                  'post_states': post_states,
                  'post_costs': torch.einsum('ds,bts->btd', self.costs, post_states),
                  'reset_rates': torch.stack(rates, dim=1) if rates else self.emission.new_zeros(batch, 0)}
        for name, value in result.items():
            _finite(value, name)
        return result

    def blind_forks(self, post_states, fork_actions):
        """Unconditional surviving mass and linear costs for eight blind steps.

        Each fork starts from its supplied posterior marginal, with no future
        labels or reliability updates. The noise mode sums out of these
        shared transition/hazard dynamics. Found input states remain zero.
        """
        _require(isinstance(post_states, torch.Tensor) and post_states.ndim == 3,
                 'post_states[B,L,8]')
        batch, length, width = post_states.shape
        _require(batch > 0 and length > 0 and width == 8, 'nonempty posterior shape')
        _tensor(post_states, torch.float64, (batch, length, 8), 'post_states')
        _tensor(fork_actions, torch.int64, (batch, length, 8), 'fork_actions')
        _finite(post_states, 'post_states')
        totals = post_states.sum(-1)
        _require(bool((post_states >= 0).all())
                 and bool(((totals == 0) | ((totals - 1).abs() <= TOLERANCE)).all()),
                 'normalized posterior or absorbing zero')
        _require(bool(((fork_actions >= 0) & (fork_actions < 4)).all()), 'fork actions0..3')
        transition = (1 - self.hazard[:, :, None]) * self.transition
        state, costs, survival = post_states.reshape(-1, 8), [], []
        actions = fork_actions.reshape(-1, 8)
        for step in range(8):
            state = torch.einsum('bij,bj->bi', transition[actions[:, step]], state)
            costs.append(torch.einsum('ds,bs->bd', self.costs, state).reshape(batch, length, 4))
            survival.append(state.sum(1).reshape(batch, length))
        result = {'costs': torch.stack(costs, dim=2), 'survival': torch.stack(survival, dim=2)}
        for name, value in result.items():
            _finite(value, name)
        return result
