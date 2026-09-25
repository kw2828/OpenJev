"""Fixed local/history features initialize the unchanged dense robot transition.

This is an observed-prefix information comparison, not a new transition or a
Bayesian posterior. A learned residual may alter all twelve latent coordinates,
including the first six position coordinates. Only caller-owned state persists.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from openjev.research.structured_robot_transition import StructuredRobotTransition, require

VERSION = 'robot-history-initializer-v1'
INITIALIZERS = ('last_two', 'local_affine', 'temporal_affine')
CONTEXT = 32
FEATURES = 30
SLOPE_DENOMINATOR = 2247.5


class _ZeroAffine(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((12, FEATURES), dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(12, dtype=dtype))

    def forward(self, features):
        return F.linear(features, self.weight, self.bias)


class HistoryInitializedDense(nn.Module):
    """condition(q[B,32,6], u[B,32,6]), then forward(future_u, state).

    All inputs use the caller's common normalization. Conditioning finishes at
    time 31 and never uses u[31] numerically; the first forecast torque is u[31]
    and predicts q[32]. No observed position enters the subsequent forecast.
    """

    def __init__(self, seed=0, initializer='last_two', *, dtype=torch.float32):
        super().__init__()
        require(initializer in INITIALIZERS, 'declared history initializer required')
        self.initializer = initializer
        self.cell = StructuredRobotTransition('dense_mlp', seed, dtype=dtype)
        if initializer != 'last_two':
            # No random head initialization and no change to the cell's RNG use.
            # A child registered after cell also preserves common parameter order.
            self.head = _ZeroAffine(dtype)

    @property
    def parameter_count(self):
        return sum(value.numel() for value in self.parameters())

    @property
    def state_scalars(self):
        return 12

    @property
    def added_parameter_count(self):
        return 0 if self.initializer == 'last_two' else 372

    def _validate_parameters(self):
        require(self.initializer in INITIALIZERS, 'declared history initializer required')
        modules = {'cell'} if self.initializer == 'last_two' else {'cell', 'head'}
        require(set(self._modules) == modules and type(self.cell) is StructuredRobotTransition
                and self.cell.kind == 'dense_mlp', 'exact dense_mlp child cell required')
        dtype = self.cell._validate_parameters()
        require(not self._parameters and not dict(self.named_buffers()),
                'exact initializer parameter/buffer roster required')
        if self.initializer != 'last_two':
            require(type(self.head) is _ZeroAffine and not self.head._modules
                    and set(self.head._parameters) == {'weight', 'bias'}, 'exact affine head required')
            self.cell._tensor(self.head.weight, (12, FEATURES), dtype, 'head.weight')
            self.cell._tensor(self.head.bias, (12,), dtype, 'head.bias')
        return dtype

    def _base(self, q_context, u_context):
        self._validate_parameters()
        require(isinstance(q_context, torch.Tensor) and q_context.ndim == 3
                and q_context.shape[0] > 0 and tuple(q_context.shape[1:]) == (CONTEXT, 6),
                'context shape [positive B,32,6] required')
        # Frozen validation includes all u entries. The actual last-two rule
        # reads no torque; u[31] remains excluded from both feature definitions.
        return self.cell.condition(q_context, u_context)

    def _features(self, q_context, u_context, base):
        require(self.initializer != 'last_two', 'last_two has no affine features')
        difference = base[:, 6:]
        common = torch.cat((base[:, :6], difference, u_context[:, 30]), dim=-1)
        if self.initializer == 'local_affine':
            extra = (difference.square(), torch.tanh(u_context[:, 30]))
        else:
            times = torch.arange(30, dtype=q_context.dtype, device='cpu') - 14.5
            slope = (q_context[:, :30] * times[None, :, None]).sum(1) / SLOPE_DENOMINATOR
            extra = (slope, u_context[:, :30].mean(1))
        features = torch.cat((common, *extra), dim=-1)
        self.cell._tensor(features, (len(base), FEATURES), q_context.dtype, 'initializer features')
        return features

    def initialization_features(self, q_context, u_context):
        """Return owned, graph-attached 30 features for either affine mode."""
        return self._features(q_context, u_context, self._base(q_context, u_context))

    def condition(self, q_context, u_context):
        base = self._base(q_context, u_context)
        if self.initializer == 'last_two':
            return base
        features = self._features(q_context, u_context, base)
        state = base + self.head(features)
        self.cell._finite(state[:, :6], state)
        return state

    def step(self, state, inputs):
        self._validate_parameters()
        return self.cell.step(state, inputs)

    def forward(self, future_u, state):
        self._validate_parameters()
        return self.cell(future_u, state)

    def model_spec(self):
        dtype = self._validate_parameters()
        return {
            'version': VERSION, 'initializer': self.initializer, 'transition_kind': 'dense_mlp',
            'seed': self.cell.initialization_seed, 'dtype': str(dtype).removeprefix('torch.'),
            'parameter_count': self.parameter_count, 'transition_parameter_count': 590,
            'added_parameter_count': self.added_parameter_count,
            'parameter_bytes': sum(p.numel() * p.element_size() for p in self.parameters()),
            'state_scalars': 12, 'state_bytes_per_stream': 12 * self.cell.input_matrix.element_size(),
            'buffer_bytes': 0, 'structurally_inactive_parameter_count': 0,
            'context_length': CONTEXT, 'feature_count': 0 if self.initializer == 'last_two' else FEATURES,
            'head_initialization': 'absent' if self.initializer == 'last_two' else 'all zero',
            'conditioning': {
                'last_two': 'concat(q31, q31-q30)',
                'local_affine': 'base + affine(q31, dq31, u30, dq31 squared, tanh(u30))',
                'temporal_affine': 'base + affine(q31, dq31, u30, slope(q0:30), mean(u0:30))',
            }[self.initializer],
            'history_slope_denominator': SLOPE_DENOMINATOR if self.initializer == 'temporal_affine' else None,
            'conditioning_last_torque_index': None if self.initializer == 'last_two' else 30,
            'timing': 'condition at time31; first future u31 predicts q32; forecast uses predictions only',
            'state_semantics': 'learned residual can alter all12 coordinates; not a Bayesian posterior',
            'persistent_cache': False, 'retained_trajectory': False,
            'storage_scope': 'parameters and caller state; normalization, request arrays, gradients, optimizer '
                             'and workspace excluded',
            'transition_spec': self.cell.model_spec(),
        }
