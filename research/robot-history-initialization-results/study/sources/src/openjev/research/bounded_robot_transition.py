"""Small bounded LPV baseline with recurrent, instantaneous and constant scheduling.

This uses established recurrent parameter-varying dynamics ideas. Every latent
operator has a common norm bound. The complete nonlinear/scheduler recurrence
is not claimed to be incrementally contracting or to have stable gradients.
No unconstrained AR2 bypass, future output labels or persistent caches exist.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import nn

if TYPE_CHECKING:
    import numpy as np

VERSION = 'bounded-robot-transition-v1'
KINDS = ('recurrent', 'instant', 'constant')
LATENT = 12
HIDDEN = 8
EXPERTS = 2
RADIUS = .9999


class LPVState(NamedTuple):
    """Caller-owned latent state and recurrent scheduler state, if any."""

    x: torch.Tensor
    h: torch.Tensor | None


class Operators(NamedTuple):
    """Graph-attached per-call operators; never cached across calls or updates."""

    matrix: torch.Tensor
    scale: torch.Tensor
    raw_spectral_norm: torch.Tensor


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


class BoundedLPV(nn.Module):
    """Two-expert 12-state LPV with an optional 8-state scheduling GRU.

    condition(q,u) accepts equal [B,C,6] arrays, C>=2. Scheduler conditioning
    uses observed [q_t,u_t] for t=1,...,C-2. The returned latent state is
    [q_(C-1),q_(C-1)-q_(C-2)]. The next torque u_(C-1) predicts q_C.
    step(state,u) returns prediction and new state. forward(q,u,future_u)
    returns [B,H,6] predictions; rollout(future_u,state) also returns final state.
    """

    def __init__(self, kind: str = 'recurrent', seed: int = 0, *, dtype: torch.dtype = torch.float32):
        super().__init__()
        require(type(kind) is str and kind in KINDS, 'declared LPV kind required')
        require(type(seed) is int and 0 <= seed < 2**63, 'nonnegative local integer seed required')
        require(dtype in (torch.float32, torch.float64), 'CPU float32 or float64 required')
        self.kind, self.initialization_seed = kind, seed
        generator = torch.Generator(device='cpu').manual_seed(seed)
        random_matrix = torch.randn((EXPERTS, LATENT, LATENT), dtype=dtype, generator=generator)
        self.raw_matrix = nn.Parameter(.999 * torch.eye(LATENT, dtype=dtype)[None]
                                       + .003 * (random_matrix - random_matrix.transpose(-1, -2)) / math.sqrt(LATENT))
        self.input_matrix = nn.Parameter(.01 * torch.randn((EXPERTS, LATENT, 6), dtype=dtype, generator=generator))
        self.expert_bias = nn.Parameter(torch.zeros(EXPERTS, LATENT, dtype=dtype))
        initial_d = torch.cat((torch.zeros(6, dtype=dtype),
                               torch.full((6,), math.atanh(math.log(10) / 3), dtype=dtype)))
        self.scale_raw = nn.Parameter(initial_d)
        if kind != 'constant':
            # Constructor initialization is local and paired across both kinds.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.scheduler = nn.GRUCell(12, HIDDEN, dtype=dtype)
                self.gate = nn.Linear(HIDDEN, EXPERTS, dtype=dtype)
            with torch.no_grad():
                self.gate.weight.zero_()
                self.gate.bias.zero_()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def state_scalars(self) -> int:
        return LATENT + (HIDDEN if self.kind == 'recurrent' else 0)

    def _validate_parameters(self) -> torch.dtype:
        require(self.kind in KINDS, 'declared LPV kind required')
        expected = {'raw_matrix': (2, 12, 12), 'input_matrix': (2, 12, 6),
                    'expert_bias': (2, 12), 'scale_raw': (12,)}
        if self.kind != 'constant':
            expected.update({'scheduler.weight_ih': (24, 12), 'scheduler.weight_hh': (24, 8),
                             'scheduler.bias_ih': (24,), 'scheduler.bias_hh': (24,),
                             'gate.weight': (2, 8), 'gate.bias': (2,)})
        actual = dict(self.named_parameters())
        require(set(actual) == set(expected) and not dict(self.named_buffers()), 'exact LPV parameter/buffer roster')
        dtype = self.raw_matrix.dtype
        require(dtype in (torch.float32, torch.float64), 'CPU float32 or float64 parameters required')
        for name, shape in expected.items():
            self._tensor(actual[name], shape, dtype, name)
        return dtype

    @staticmethod
    def _tensor(value: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype, name: str) -> None:
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == dtype
                and tuple(value.shape) == shape and bool(torch.isfinite(value).all()),
                'finite CPU tensor with exact shape/dtype: ' + name)

    def _state(self, state: LPVState, batch: int, dtype: torch.dtype) -> None:
        require(isinstance(state, LPVState), 'explicit LPVState required')
        self._tensor(state.x, (batch, LATENT), dtype, 'latent state')
        if self.kind == 'recurrent':
            self._tensor(state.h, (batch, HIDDEN), dtype, 'scheduler state')
        else:
            require(state.h is None, 'nonrecurrent scheduler has no retained hidden state')

    def _operators(self) -> Operators:
        norm = torch.linalg.matrix_norm(self.raw_matrix, ord=2, dim=(-2, -1))
        matrix = RADIUS * self.raw_matrix / norm.clamp_min(1)[:, None, None]
        return Operators(matrix, torch.exp(3 * torch.tanh(self.scale_raw)), norm)

    def operators(self) -> Operators:
        """Return owned differentiable operators for mathematical diagnostics."""
        self._validate_parameters()
        return self._operators()

    def _condition(self, q: torch.Tensor, u: torch.Tensor) -> LPVState:
        hidden = None
        if self.kind == 'recurrent':
            hidden = q.new_zeros(len(q), HIDDEN)
            for t in range(1, q.shape[1] - 1):
                hidden = self.scheduler(torch.cat((q[:, t], u[:, t]), dim=-1), hidden)
        return LPVState(torch.cat((q[:, -1], q[:, -1] - q[:, -2]), dim=-1), hidden)

    def _contexts(self, q: torch.Tensor, u: torch.Tensor, dtype: torch.dtype) -> None:
        require(isinstance(q, torch.Tensor) and q.ndim == 3 and q.shape[0] > 0
                and q.shape[1] >= 2 and q.shape[2] == 6, 'context shape [positive B,C>=2,6]')
        self._tensor(q, tuple(q.shape), dtype, 'observed context positions')
        self._tensor(u, tuple(q.shape), dtype, 'observed context torques')

    def condition(self, q_context: torch.Tensor, u_context: torch.Tensor) -> LPVState:
        dtype = self._validate_parameters()
        self._contexts(q_context, u_context, dtype)
        state = self._condition(q_context, u_context)
        self._finite(q_context.new_empty(len(q_context), 0, 6), state)
        return state

    def _step(self, state: LPVState, inputs: torch.Tensor, operators: Operators) -> tuple[torch.Tensor, LPVState]:
        hidden = None
        if self.kind == 'constant':
            mixing = inputs.new_full((len(inputs), EXPERTS), .5)
        else:
            old = state.h if self.kind == 'recurrent' else inputs.new_zeros(len(inputs), HIDDEN)
            hidden_now = self.scheduler(torch.cat((state.x[:, :6], inputs), dim=-1), old)
            mixing = torch.softmax(self.gate(hidden_now), dim=-1)
            if self.kind == 'recurrent':
                hidden = hidden_now
        transformed = state.x * operators.scale
        expert = torch.einsum('eij,bj->bei', operators.matrix, transformed)
        expert = expert + torch.einsum('eij,bj->bei', self.input_matrix, inputs) + self.expert_bias
        updated = (mixing[:, :, None] * expert).sum(dim=1) / operators.scale
        return updated[:, :6].clone(), LPVState(updated, hidden)

    @staticmethod
    def _finite(prediction: torch.Tensor, state: LPVState) -> None:
        require(bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(state.x).all())
                and (state.h is None or bool(torch.isfinite(state.h).all())),
                'nonfinite LPV output; no rollout clipping or repair')

    def step(self, state: LPVState, inputs: torch.Tensor) -> tuple[torch.Tensor, LPVState]:
        dtype = self._validate_parameters()
        require(isinstance(inputs, torch.Tensor) and inputs.ndim == 2 and inputs.shape[0] > 0,
                'step input shape [positive B,6]')
        self._tensor(inputs, (len(inputs), 6), dtype, 'next torque')
        self._state(state, len(inputs), dtype)
        output, final = self._step(state, inputs, self._operators())
        self._finite(output, final)
        return output, final

    def _rollout(self, sequence: torch.Tensor, state: LPVState, operators: Operators) -> tuple[torch.Tensor, LPVState]:
        outputs, current = [], state
        for t in range(sequence.shape[1]):
            output, current = self._step(current, sequence[:, t], operators)
            outputs.append(output)
        prediction = torch.stack(outputs, dim=1) if outputs else sequence.new_empty(len(sequence), 0, 6)
        if not outputs:
            current = LPVState(state.x.clone(), None if state.h is None else state.h.clone())
        self._finite(prediction, current)
        return prediction, current

    def rollout(self, future_u: torch.Tensor, state: LPVState) -> tuple[torch.Tensor, LPVState]:
        dtype = self._validate_parameters()
        require(isinstance(future_u, torch.Tensor) and future_u.ndim == 3 and future_u.shape[0] > 0,
                'future torque shape [positive B,H,6]')
        self._tensor(future_u, (len(future_u), future_u.shape[1], 6), dtype, 'future torques')
        self._state(state, len(future_u), dtype)
        return self._rollout(future_u, state, self._operators())

    def forward(self, q_context: torch.Tensor, u_context: torch.Tensor, future_u: torch.Tensor) -> torch.Tensor:
        dtype = self._validate_parameters()
        self._contexts(q_context, u_context, dtype)
        require(isinstance(future_u, torch.Tensor) and future_u.ndim == 3, 'future torque rank3')
        self._tensor(future_u, (len(q_context), future_u.shape[1], 6), dtype, 'future torques')
        # The two small spectral norms are computed once, outside both loops.
        return self._rollout(future_u, self._condition(q_context, u_context), self._operators())[0]

    def model_spec(self) -> dict[str, object]:
        dtype = self._validate_parameters()
        inactive = 192 if self.kind == 'instant' else 0
        return {'version': VERSION, 'kind': self.kind, 'seed': self.initialization_seed,
                'dtype': str(dtype).removeprefix('torch.'), 'parameter_count': self.parameter_count,
                'trainable_parameter_count': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'parameter_bytes': sum(p.numel() * p.element_size() for p in self.parameters()),
                'structurally_inactive_parameter_count': inactive,
                'inactive_scope': 'instant scheduler.weight_hh multiplies zero hidden state' if inactive else None,
                'state_scalars': self.state_scalars, 'state_bytes_per_stream': self.state_scalars * self.raw_matrix.element_size(),
                'buffer_bytes': 0, 'radius_cap': RADIUS, 'scale_log_bound': 3.,
                'persistent_cache': False, 'retained_trajectory': False,
                'stability_scope': 'bounded forcing gives bounded latent/output trajectories in exact arithmetic; not full-system incremental contraction or gradient stability',
                'normalization': 'explicit spectral cap on latent operators; no output clipping or adaptive jitter',
                'initial_gate': 'uniform mixture; scheduler gradients initially blocked by zero gate weight',
                'timing': 'condition observed indices1..C-2; nextu[C-1] predicts q[C]',
                'storage_scope': 'parameters and explicit caller state; external normalization, request arrays and temporary workspace excluded'}

    def export_numpy(self) -> dict[str, np.ndarray]:
        self._validate_parameters()
        return {name: value.detach().cpu().numpy().copy() for name, value in self.state_dict().items()}

    def export_json(self) -> dict[str, object]:
        return {'spec': self.model_spec(),
                'parameters': {name: value.detach().cpu().tolist() for name, value in self.named_parameters()}}
