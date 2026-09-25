"""Current-state scheduled robot transitions with four anchored reflections.

Established Householder/LPV structure, not an architecture novelty claim.
Bounded families have a common-metric forced-state bound in exact arithmetic,
not incremental contraction of their state-dependent gates or gradient bounds.
Only explicit caller state persists. No measurement or target input is accepted.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn

from openjev.research.compact_robot_gate import SHAPES, CompactResetGate

VERSION = 'structured-robot-transition-v1'
KINDS = ('householder', 'dense_bounded', 'dense_unbounded', 'dense_mlp')
RADIUS = .9999
INITIAL_RADIUS = .999
LATENT = 12
ANCHORS = (0, 0, 6, 6)


class Prepared(NamedTuple):
    scale: torch.Tensor
    matrix: torch.Tensor | None
    raw_spectral_norm: torch.Tensor | None
    vectors: torch.Tensor | None
    decay: torch.Tensor | None


def require(ok, message):
    if not ok:
        raise ValueError(message)


class StructuredRobotTransition(nn.Module):
    """Old training API: condition(q,u), then forward(future_u,state).

    Contexts are [B,C,6], C>=2; latent state is [q_last, q_last-q_previous].
    No historical gate conditioning occurs. The first future torque is u[C-1]
    and predicts q[C]. All four kinds retain twelve latent values per stream.
    """

    def __init__(self, kind='householder', seed=0, *, dtype=torch.float32):
        super().__init__()
        require(kind in KINDS, 'declared structured transition kind required')
        require(type(seed) is int and 0 <= seed < 2**63, 'nonnegative integer seed required')
        require(dtype in (torch.float32, torch.float64), 'CPU float32 or float64 required')
        self.kind, self.initialization_seed = kind, seed
        forcing_rng = torch.Generator(device='cpu').manual_seed(seed)
        self.input_matrix = nn.Parameter(.01 * torch.randn((2, 12, 6), generator=forcing_rng, dtype=dtype))
        self.expert_bias = nn.Parameter(torch.zeros((2, 12), dtype=dtype))
        self.scale_raw = nn.Parameter(torch.cat((torch.zeros(6, dtype=dtype),
            torch.full((6,), math.atanh(math.log(10) / 3), dtype=dtype))))
        if kind == 'householder':
            reflector_rng = torch.Generator(device='cpu').manual_seed(seed ^ 0x484F5553)
            pairs = .01 * torch.randn((2, 2, 11), generator=reflector_rng, dtype=dtype)
            # Independent trainable entries, identical within each pair initially.
            self.reflection_raw = nn.Parameter(pairs.repeat_interleave(2, dim=1))
            ratio = INITIAL_RADIUS / RADIUS
            self.decay_raw = nn.Parameter(torch.full((2, 12), math.log(ratio / (1-ratio)), dtype=dtype))
        else:
            self.raw_matrix = nn.Parameter((INITIAL_RADIUS / RADIUS) * torch.eye(12, dtype=dtype).repeat(2, 1, 1))
        if kind == 'dense_mlp':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.gate = nn.Sequential(nn.Linear(12, 8, dtype=dtype), nn.Tanh(), nn.Linear(8, 2, dtype=dtype))
            with torch.no_grad():
                self.gate[2].weight.zero_()
                self.gate[2].bias.zero_()
        else:
            self.gate = CompactResetGate(seed, dtype=dtype)

    @property
    def parameter_count(self):
        return sum(value.numel() for value in self.parameters())

    @property
    def state_scalars(self):
        return 12

    @staticmethod
    def _tensor(value, shape, dtype, name):
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                and value.layout == torch.strided and value.dtype == dtype
                and tuple(value.shape) == shape and bool(torch.isfinite(value).all()),
                'finite CPU tensor with exact shape/dtype: ' + name)

    def _validate_parameters(self):
        require(self.kind in KINDS, 'declared structured transition kind required')
        shapes = {'input_matrix': (2, 12, 6), 'expert_bias': (2, 12), 'scale_raw': (12,)}
        if self.kind == 'householder':
            shapes.update(reflection_raw=(2, 4, 11), decay_raw=(2, 12))
        else:
            shapes['raw_matrix'] = (2, 12, 12)
        if self.kind == 'dense_mlp':
            shapes.update({'gate.0.weight': (8, 12), 'gate.0.bias': (8,), 'gate.2.weight': (2, 8), 'gate.2.bias': (2,)})
        else:
            shapes.update({'gate.' + key: shape for key, shape in SHAPES.items()})
        values = dict(self.named_parameters())
        require(set(values) == set(shapes) and not dict(self.named_buffers()), 'exact structured parameter/buffer roster')
        dtype = self.input_matrix.dtype
        require(dtype in (torch.float32, torch.float64), 'float32 or float64 parameters required')
        for key, shape in shapes.items():
            self._tensor(values[key], shape, dtype, key)
        return dtype

    def _prepare(self):
        scale = torch.exp(3 * torch.tanh(self.scale_raw))
        if self.kind == 'householder':
            free = torch.tanh(self.reflection_raw)
            vectors = torch.stack([torch.cat((free[:, k, :anchor], free.new_ones(2, 1), free[:, k, anchor:]), dim=-1)
                                   for k, anchor in enumerate(ANCHORS)], dim=1)
            return Prepared(scale, None, None, vectors, RADIUS * torch.sigmoid(self.decay_raw))
        if self.kind == 'dense_unbounded':
            return Prepared(scale, RADIUS * self.raw_matrix, None, None, None)
        norm = torch.linalg.matrix_norm(self.raw_matrix, ord=2, dim=(-2, -1))
        return Prepared(scale, RADIUS * self.raw_matrix / norm.clamp_min(1)[:, None, None], norm, None, None)

    def _mix(self, state, inputs):
        value = torch.cat((state[:, :6], inputs), dim=-1)
        return torch.softmax(self.gate(value), dim=-1) if self.kind == 'dense_mlp' else self.gate._forward_unchecked(value)

    @staticmethod
    def _vectors(mixing, prepared):
        return torch.einsum('be,ekd->bkd', mixing, prepared.vectors)

    @staticmethod
    def _reflect(value, vector):
        return value - 2 * vector * (vector * value).sum(-1, keepdim=True) / vector.square().sum(-1, keepdim=True)

    def _transition(self, state, mixing, prepared):
        value = state * prepared.scale
        if self.kind == 'householder':
            vectors = self._vectors(mixing, prepared)
            for k in range(4):
                value = self._reflect(value, vectors[:, k])
            decay = mixing @ prepared.decay
            return decay * value
        return (mixing[:, :, None] * torch.einsum('eij,bj->bei', prepared.matrix, value)).sum(1)

    def _step(self, state, inputs, prepared):
        mixing = self._mix(state, inputs)
        forced = torch.einsum('eij,bj->bei', self.input_matrix, inputs) + self.expert_bias
        updated = (self._transition(state, mixing, prepared) + (mixing[:, :, None] * forced).sum(1)) / prepared.scale
        return updated[:, :6].clone(), updated

    @staticmethod
    def _finite(prediction, state):
        require(bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(state).all()),
                'nonfinite structured transition output; no repair')

    def condition(self, q_context, u_context):
        dtype = self._validate_parameters()
        require(isinstance(q_context, torch.Tensor) and q_context.ndim == 3
                and q_context.shape[0] > 0 and q_context.shape[1] >= 2 and q_context.shape[2] == 6,
                'context shape [positive B,C>=2,6] required')
        self._tensor(q_context, tuple(q_context.shape), dtype, 'observed positions')
        self._tensor(u_context, tuple(q_context.shape), dtype, 'observed torques')
        state = torch.cat((q_context[:, -1], q_context[:, -1] - q_context[:, -2]), dim=-1)
        self._finite(state[:, :6], state)
        return state

    def step(self, state, inputs):
        dtype = self._validate_parameters()
        require(isinstance(inputs, torch.Tensor) and inputs.ndim == 2 and inputs.shape[0] > 0,
                'step torque shape [positive B,6] required')
        self._tensor(inputs, (len(inputs), 6), dtype, 'next torque')
        self._tensor(state, (len(inputs), 12), dtype, 'explicit latent state')
        prediction, final = self._step(state, inputs, self._prepare())
        self._finite(prediction, final)
        return prediction, final

    def forward(self, future_u, state):
        dtype = self._validate_parameters()
        require(isinstance(future_u, torch.Tensor) and future_u.ndim == 3 and future_u.shape[0] > 0,
                'future torque shape [positive B,H,6] required')
        self._tensor(future_u, (len(future_u), future_u.shape[1], 6), dtype, 'future torques')
        self._tensor(state, (len(future_u), 12), dtype, 'explicit latent state')
        prepared = self._prepare()
        outputs, current = [], state
        for t in range(future_u.shape[1]):
            prediction, current = self._step(current, future_u[:, t], prepared)
            outputs.append(prediction)
        result = torch.stack(outputs, dim=1) if outputs else future_u.new_empty(len(future_u), 0, 6)
        if not outputs:
            current = state.clone()
        self._finite(result, current)
        return result, current

    def operators(self, mixing):
        """Graph-attached metric-space matrices for diagnostics, not a cache.

        Householder forward uses four vector updates, never dense matrices.
        This diagnostic deliberately materializes the corresponding matrices.
        """
        dtype = self._validate_parameters()
        require(isinstance(mixing, torch.Tensor) and mixing.ndim == 2 and mixing.shape[0] > 0, 'mixing shape [B,2]')
        self._tensor(mixing, (len(mixing), 2), dtype, 'mixing')
        require(bool((mixing >= 0).all()) and bool(torch.allclose(mixing.sum(1), mixing.new_ones(len(mixing)), atol=1e-6, rtol=0)), 'simplex mixing')
        prepared = self._prepare()
        vectors, decay = None, None
        if self.kind == 'householder':
            vectors = self._vectors(mixing, prepared)
            matrix = torch.eye(12, dtype=dtype).expand(len(mixing), 12, 12).clone()
            for k in range(4):
                vector = vectors[:, k]
                reflected = torch.eye(12, dtype=dtype)[None] - 2 * vector[:, :, None] * vector[:, None, :] / vector.square().sum(-1)[:, None, None]
                matrix = reflected @ matrix
            decay = mixing @ prepared.decay
            matrix = decay[:, :, None] * matrix
        else:
            matrix = torch.einsum('be,eij->bij', mixing, prepared.matrix)
        return {'matrix': matrix, 'scale': prepared.scale, 'mixing': mixing.clone(),
                'reflection_vectors': vectors, 'decay': decay, 'raw_spectral_norm': prepared.raw_spectral_norm}

    def model_spec(self):
        dtype = self._validate_parameters()
        bounded = self.kind != 'dense_unbounded'
        return {'version': VERSION, 'kind': self.kind, 'seed': self.initialization_seed,
                'dtype': str(dtype).removeprefix('torch.'), 'parameter_count': self.parameter_count,
                'parameter_bytes': sum(p.numel() * p.element_size() for p in self.parameters()),
                'state_scalars': 12, 'state_bytes_per_stream': 12 * self.input_matrix.element_size(),
                'buffer_bytes': 0, 'structurally_inactive_parameter_count': 0,
                'radius_cap': RADIUS if bounded else None, 'scale_log_bound': 3.,
                'stability_scope': 'bounded forced trajectories in exact arithmetic; not incremental contraction or bounded gradients' if bounded else 'no bounded-trajectory guarantee',
                'conditioning': 'last observed q and difference only; no historical gate state',
                'timing': 'first futureu[C-1] predicts q[C]; subsequent rollout uses predictions only',
                'initial_operator': '.999I in real arithmetic, not a bitwise cross-family claim',
                'persistent_cache': False, 'retained_trajectory': False,
                'storage_scope': 'parameters and caller state; normalization, request arrays, gradients, optimizer and workspace excluded'}
