"""Constant-gain prefix observation correction with frozen robot dynamics.

The gain is an established observer baseline, not a Kalman posterior or a
physical-state estimate. Only caller-owned state persists between forecast steps.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from openjev.research.robot_history_initializer import INITIALIZERS, HistoryInitializedDense
from openjev.research.structured_robot_transition import StructuredRobotTransition, require

VERSION = 'robot-observer-initializer-v1'
MODES = (*INITIALIZERS, 'observer_fixed', 'observer_learned', 'observer_zero')
CONTEXT = 32


def _fixed_gain(dtype, zero=False):
    return torch.zeros((12, 6), dtype=dtype) if zero else torch.eye(6, dtype=dtype).repeat(2, 1)


class FrozenObserverInitializer(HistoryInitializedDense):
    """condition(q[B,32,6], u[B,32,6]), then forward(future_u, state).

    The child ``cell`` and affine ``head`` retain the predecessor's exact state
    dictionary keys. Load an inherited transition with ``cell.load_state_dict``;
    the caller authenticates those weights. All 590 cell parameters stay frozen.
    Learned initializers retain gradients through every frozen transition.
    """

    def __init__(self, seed=0, mode='last_two', *, dtype=torch.float32):
        require(mode in MODES, 'declared frozen observer initializer required')
        super().__init__(seed, mode if mode in INITIALIZERS else 'last_two', dtype=dtype)
        self.mode = mode
        for value in self.cell.parameters():
            value.requires_grad_(False)
        if mode == 'observer_learned':
            self.gain = nn.Parameter(_fixed_gain(dtype))
        elif mode in ('observer_fixed', 'observer_zero'):
            self.register_buffer('gain', _fixed_gain(dtype, mode == 'observer_zero'))

    @property
    def added_parameter_count(self):
        return 372 if self.mode in INITIALIZERS[1:] else 72 if self.mode == 'observer_learned' else 0

    @property
    def trainable_parameter_count(self):
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    @property
    def frozen_parameter_count(self):
        return sum(value.numel() for value in self.parameters() if not value.requires_grad)

    def _validate_parameters(self):
        require(self.mode in MODES and self.initializer == (self.mode if self.mode in INITIALIZERS else 'last_two'),
                'fixed initializer mode/feature identity required')
        if self.mode in INITIALIZERS:
            dtype = super()._validate_parameters()
        else:
            require(set(self._modules) == {'cell'} and type(self.cell) is StructuredRobotTransition
                    and self.cell.kind == 'dense_mlp', 'exact frozen dense_mlp cell required')
            dtype = self.cell._validate_parameters()
            learned = self.mode == 'observer_learned'
            require(set(self._parameters) == ({'gain'} if learned else set())
                    and set(self._buffers) == (set() if learned else {'gain'})
                    and set(dict(self.named_buffers())) == (set() if learned else {'gain'}), 'exact observer gain storage roster')
            self.cell._tensor(self.gain, (12, 6), dtype, 'observation correction gain')
            if not learned:
                require(not self.gain.requires_grad and torch.equal(self.gain, _fixed_gain(dtype, self.mode == 'observer_zero')),
                        'fixed observer gain cannot change')
        require(self.cell.parameter_count == 590
                and all(not p.requires_grad and p.grad is None for p in self.cell.parameters()), 'frozen cell parameters without stored gradients required')
        return dtype

    def condition(self, q_context, u_context):
        if self.mode in INITIALIZERS:
            return super().condition(q_context, u_context)
        dtype = self._validate_parameters()
        require(isinstance(q_context, torch.Tensor) and q_context.ndim == 3 and q_context.shape[0] > 0
                and tuple(q_context.shape[1:]) == (CONTEXT, 6), 'observer context shape [positive B,32,6] required')
        shape = tuple(q_context.shape)
        self.cell._tensor(q_context, shape, dtype, 'observed positions')
        self.cell._tensor(u_context, shape, dtype, 'observed torques')
        state = torch.cat((q_context[:, 1], q_context[:, 1]-q_context[:, 0]), dim=-1)
        self.cell._finite(state[:, :6], state)
        # One preparation, including the two spectral norms, for all 30 prefix
        # steps. Nothing is cached on the module or detached from the graph.
        prepared = self.cell._prepare()
        for t in range(2, CONTEXT):
            prediction, prior = self.cell._step(state, u_context[:, t-1], prepared)
            self.cell._finite(prediction, prior)
            innovation = q_context[:, t]-prediction
            self.cell._tensor(innovation, (len(state), 6), dtype, 'observer innovation')
            state = prior + F.linear(innovation, self.gain)
            self.cell._finite(state[:, :6], state)
        return state

    def initialization_features(self, q_context, u_context):
        require(self.mode in INITIALIZERS[1:], 'only affine initializers have 30 features')
        return super().initialization_features(q_context, u_context)

    def model_spec(self):
        dtype = self._validate_parameters()
        observer = self.mode not in INITIALIZERS
        parameter_bytes = sum(p.numel()*p.element_size() for p in self.parameters())
        buffer_bytes = sum(p.numel()*p.element_size() for p in self.buffers())
        return {
            'version': VERSION, 'mode': self.mode, 'transition_kind': 'dense_mlp',
            'seed': self.cell.initialization_seed, 'dtype': str(dtype).removeprefix('torch.'),
            'parameter_count': self.parameter_count, 'transition_parameter_count': 590,
            'initializer_parameter_count': self.added_parameter_count,
            'trainable_parameter_count': self.trainable_parameter_count,
            'frozen_parameter_count': self.frozen_parameter_count,
            'parameter_bytes': parameter_bytes,
            'trainable_parameter_bytes': sum(p.numel()*p.element_size() for p in self.parameters() if p.requires_grad),
            'frozen_parameter_bytes': sum(p.numel()*p.element_size() for p in self.parameters() if not p.requires_grad),
            'buffer_bytes': buffer_bytes, 'gain_buffer_scalars': 72 if self.mode in ('observer_fixed', 'observer_zero') else 0,
            'state_scalars': 12, 'state_bytes_per_stream': 12*self.cell.input_matrix.element_size(),
            'context_length': CONTEXT, 'prefix_transition_count': 30 if observer else 0,
            'prefix_operator_preparations': 1 if observer else 0,
            'conditioning': 'z1=[q1,q1-q0]; for t=2..31: prior=f(z,u[t-1]); z=prior+K(q[t]-prior[:6])'
                            if observer else self.mode + ' unchanged history initializer',
            'gain_initialization': 'zero' if self.mode == 'observer_zero' else '[I;I]' if observer else 'absent',
            'conditioning_last_torque_index': 30 if self.mode != 'last_two' else None,
            'timing': 'condition ends at time31; first forecast u31 predicts q32; no future observations',
            'state_semantics': 'bottom six coordinates are learned state, not verified physical velocity',
            'stability_scope': 'no observer, incremental-stability, covariance or calibration guarantee',
            'persistent_cache': False, 'retained_trajectory': False,
            'storage_scope': 'parameters, fixed gain buffers and caller state; normalization, requests, gradients, '
                             'optimizer and temporary workspace excluded',
            'transition_spec': self.cell.model_spec(),
        }
