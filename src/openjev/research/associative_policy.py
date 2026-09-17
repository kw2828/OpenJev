"""GRU policy controls and an episode-local delta-rule associative memory.

This is a small adaptation of established fast-weight memory mechanisms, not a
new algorithm or a trained world model. No predictive auxiliary objective is
included here. A separately frozen experiment must establish any benefit.

The fast modes add a 16x16 runtime state, not 256 optimizer parameters. With the
default dimensions both register 4,273 extra parameters; ``fast_global`` freezes
the gate's 64 zero weights and uses its learned bias, leaving 4,209 trainable.
``fast_selective`` learns all 4,273. Their initial write strengths are identical.
The feedforward control adds 4,321 trainable parameters, a close capacity control
rather than an assertion of exactly equal active parameter counts.

References: Ba et al. (2016), arXiv:1610.06258; Schlag et al. (2021),
arXiv:2102.11174; Irie et al. (2021), arXiv:2106.06295.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from openjev.research.predictive_memory import PredictiveMemory

MODES = ('gru', 'feedforward', 'fast_global', 'fast_selective')


def delta_write(store, key, value, strength):
    """Correct one key-value association; unit keys make strength a write fraction.

Shapes are ``[batch, dimension, dimension]`` for the store, ``[batch,
dimension]`` for keys/values, and ``[batch, 1]`` for write strengths. The caller
normalizes keys and bounds strengths. No state is modified in place.
"""
    retrieved = torch.bmm(store, key.unsqueeze(-1)).squeeze(-1)
    correction = strength * (value - retrieved)
    return store + correction.unsqueeze(-1) * key.unsqueeze(-2)


class AssociativePolicy(nn.Module):
    def __init__(self, mode='gru', observation_dim=984, hidden=64, store_dim=16,
                 adapter_hidden=33):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f'Unknown associative policy mode: {mode}')
        if min(observation_dim, hidden, store_dim, adapter_hidden) < 1:
            raise ValueError('Model dimensions must be positive')
        self.mode = mode
        self.hidden = hidden
        self.observation_dim = observation_dim
        self.store_dim = store_dim
        self.uses_store = mode.startswith('fast_')
        self.state_size = hidden + (store_dim * store_dim if self.uses_store else 0)

        # Reuse the frozen initialization sequence exactly, including its random
        # draws before actor/value orthogonal initialization. Auxiliary prediction
        # heads belong only to this temporary object and are not registered here.
        backbone = PredictiveMemory(observation_dim, hidden)
        self.encoder = backbone.encoder
        self.memory = backbone.memory
        self.actor = backbone.actor
        self.value = backbone.value

        if mode == 'feedforward':
            self.adapter = nn.Sequential(nn.Linear(hidden, adapter_hidden), nn.Tanh(),
                                         nn.Linear(adapter_hidden, hidden))
        elif self.uses_store:
            self.key = nn.Linear(hidden, store_dim)
            self.query = nn.Linear(hidden, store_dim)
            self.content = nn.Linear(hidden, store_dim)
            self.readout = nn.Linear(store_dim, hidden)
            self.write_gate = nn.Linear(hidden, 1)
            nn.init.zeros_(self.write_gate.weight)
            nn.init.constant_(self.write_gate.bias, math.log(.1 / .9))
            if mode == 'fast_global':
                self.write_gate.weight.requires_grad_(False)

    def initial_state(self, batch):
        if not isinstance(batch, int) or batch < 1:
            raise ValueError('Batch size must be a positive integer')
        return self.actor.weight.new_zeros(batch, self.state_size)

    def observe(self, observations, previous_actions, state, reset, current_only=False,
                reset_store=False):
        """Consume one partial observation and return logits, value and flat state.

        Episode reset clears both memories and the previous action. ``current_only``
        clears both memories while retaining the previous-action input within an
        episode. ``reset_store`` clears only the matrix before this step's write;
        it can be a bool or one boolean per batch slot. Reads affect the actor and
        value only, never the GRU's recurrent update or the stored GRU state.
        """
        if observations.ndim != 2 or observations.shape[-1] != self.observation_dim:
            raise ValueError('Wrong observation shape')
        batch = observations.shape[0]
        if state.shape != (batch, self.state_size):
            raise ValueError('Wrong recurrent state shape')
        if reset.shape != (batch,) or reset.dtype != torch.bool or previous_actions.shape != (batch,):
            raise ValueError('Expected one action and boolean reset flag per batch slot')
        if not torch.isfinite(observations).all() or not torch.isfinite(state).all():
            raise ValueError('Observations and recurrent state must be finite')
        if isinstance(reset_store, torch.Tensor):
            if reset_store.shape != (batch,) or reset_store.dtype != torch.bool:
                raise ValueError('Store reset flags must match the batch')
        elif not isinstance(reset_store, bool):
            raise TypeError('Store reset must be boolean')

        retained = (~reset).to(state.dtype).unsqueeze(-1)
        state = torch.zeros_like(state) if current_only else state * retained
        actions = F.one_hot(previous_actions.long(), 7).to(observations.dtype) * retained
        encoded = self.encoder(observations)
        reactive = self.memory(torch.cat([encoded, actions], -1), state[:, :self.hidden])
        decision = reactive
        next_state = reactive
        if self.mode == 'feedforward':
            decision = reactive + self.adapter(encoded)
        elif self.uses_store:
            store = state[:, self.hidden:].reshape(batch, self.store_dim, self.store_dim)
            if isinstance(reset_store, torch.Tensor):
                store = store * (~reset_store).to(store.dtype)[:, None, None]
            elif reset_store:
                store = torch.zeros_like(store)
            key = F.normalize(self.key(encoded), dim=-1, eps=1e-6)
            query = F.normalize(self.query(encoded), dim=-1, eps=1e-6)
            content = torch.tanh(self.content(encoded))
            strength = (self.write_gate(encoded).sigmoid() if self.mode == 'fast_selective'
                        else self.write_gate.bias.sigmoid().expand(batch, 1))
            store = delta_write(store, key, content, strength)
            read = torch.bmm(store, query.unsqueeze(-1)).squeeze(-1)
            decision = reactive + self.readout(read)
            next_state = torch.cat([reactive, store.flatten(1)], -1)
        return self.actor(decision), self.value(decision).squeeze(-1), next_state

    def sequence(self, observations, previous_actions, state, resets, current_only=False,
                 reset_store=False):
        """Replay causal sequences with separate episode and optional store masks."""
        if (observations.ndim != 3 or len(observations) < 1
                or previous_actions.shape != observations.shape[:2]
                or resets.shape != observations.shape[:2]):
            raise ValueError('Expected nonempty time-by-batch observation, action and reset sequences')
        if isinstance(reset_store, torch.Tensor):
            if reset_store.shape != resets.shape or reset_store.dtype != torch.bool:
                raise ValueError('Sequence store reset mask must match episode reset mask')
        elif not isinstance(reset_store, bool):
            raise TypeError('Store reset must be boolean')
        logits, values, states = [], [], []
        for t in range(len(observations)):
            store_mask = reset_store[t] if isinstance(reset_store, torch.Tensor) else reset_store
            logit, value, state = self.observe(observations[t], previous_actions[t], state,
                                              resets[t], current_only, store_mask)
            logits.append(logit)
            values.append(value)
            states.append(state)
        return torch.stack(logits), torch.stack(values), torch.stack(states)
