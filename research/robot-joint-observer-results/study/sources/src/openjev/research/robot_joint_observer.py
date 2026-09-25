"""Jointly train a conventional prefix observer and the qualified transition.

The two observer modes differ in prefix access and correction depth. There is
no covariance, learned readout, forecast observation, or stability guarantee.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from openjev.research.robot_history_initializer import CONTEXT, HistoryInitializedDense
from openjev.research.robot_position_observer import DIAGNOSTIC_KEYS, _position_gain, _prefix_maxima
from openjev.research.structured_robot_transition import StructuredRobotTransition, require

VERSION = 'robot-joint-observer-v1'
MODES = ('long', 'short', 'last_two', 'temporal')
COUNTS = {'long': 662, 'short': 662, 'last_two': 590, 'temporal': 962}


class JointObserver(HistoryInitializedDense):
    """condition(q[B,32,6],u), then forward(future_u,state).

    Constructor parameters are all trainable. Explicitly freezing every
    parameter for inference is supported; mixed trainability is rejected.
    The caller loads the same inherited cell.* values into every fresh arm.
    """

    def __init__(self, seed=0, mode='long', *, dtype=torch.float32):
        require(mode in MODES, 'declared joint observer mode required')
        super().__init__(seed, 'temporal_affine' if mode == 'temporal' else 'last_two', dtype=dtype)
        self.mode = mode
        if mode in ('long', 'short'):
            self.gain = nn.Parameter(_position_gain(dtype))

    @property
    def added_parameter_count(self):
        return 72 if self.mode in ('long', 'short') else 372 if self.mode == 'temporal' else 0

    @property
    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _validate_parameters(self):
        require(self.mode in MODES and self.initializer ==
                ('temporal_affine' if self.mode == 'temporal' else 'last_two'), 'joint mode/initializer identity')
        if self.mode in ('long', 'short'):
            require(set(self._modules) == {'cell'} and type(self.cell) is StructuredRobotTransition
                    and self.cell.kind == 'dense_mlp' and set(self._parameters) == {'gain'}
                    and not dict(self.named_buffers()), 'exact joint observer parameter/buffer roster')
            dtype = self.cell._validate_parameters()
            self.cell._tensor(self.gain, (12, 6), dtype, 'observation correction gain')
        else:
            dtype = super()._validate_parameters()
        parameters = tuple(self.parameters())
        require(self.cell.parameter_count == 590 and self.parameter_count == COUNTS[self.mode], 'exact joint parameter count')
        require(len({p.requires_grad for p in parameters}) == 1, 'all parameters trainable or all frozen, never a mixed mask')
        require(parameters[0].requires_grad or all(p.grad is None for p in parameters), 'frozen inference retains no parameter gradients')
        return dtype

    def condition(self, q_context, u_context, diagnostics=None):
        """Long: t2..31; short: z30 from q29/q30, then u30 and q31 once.

        Both finish at time31, and u31 first predicts q32 in the forecast.
        Detached optional maxima do not change the learning path or persist.
        Continued controls use the unchanged qualified history initializer.
        """
        require(diagnostics is None or (type(diagnostics) is dict and not diagnostics),
                'diagnostics must be None or an empty caller-owned dict')
        if self.mode in ('last_two', 'temporal'):
            state = super().condition(q_context, u_context)
            if diagnostics is not None:
                diagnostics.update(dict.fromkeys(DIAGNOSTIC_KEYS, 0.))
                diagnostics['prefix_steps'] = 0
                _prefix_maxima(diagnostics, state, 'state')
            return state
        dtype = self._validate_parameters()
        require(isinstance(q_context, torch.Tensor) and q_context.ndim == 3 and q_context.shape[0] > 0
                and tuple(q_context.shape[1:]) == (CONTEXT, 6), 'observer context shape [positive B,32,6] required')
        shape = tuple(q_context.shape)
        self.cell._tensor(q_context, shape, dtype, 'observed positions')
        self.cell._tensor(u_context, shape, dtype, 'observed torques')
        if diagnostics is not None:
            diagnostics.update(dict.fromkeys(DIAGNOSTIC_KEYS, 0.))
            diagnostics['prefix_steps'] = 0
        start = 1 if self.mode == 'long' else 30
        state = torch.cat((q_context[:, start], q_context[:, start]-q_context[:, start-1]), dim=-1)
        self.cell._finite(state[:, :6], state)
        if diagnostics is not None:
            _prefix_maxima(diagnostics, state, 'state')
        prepared = self.cell._prepare()
        for t in range(start+1, CONTEXT):
            prediction, prior = self.cell._step(state, u_context[:, t-1], prepared)
            self.cell._finite(prediction, prior)
            if diagnostics is not None:
                _prefix_maxima(diagnostics, prior, 'state')
            innovation = q_context[:, t]-prediction
            self.cell._tensor(innovation, (len(state), 6), dtype, 'observer innovation')
            if diagnostics is not None:
                _prefix_maxima(diagnostics, innovation, 'innovation')
            state = prior+F.linear(innovation, self.gain)
            self.cell._finite(state[:, :6], state)
            if diagnostics is not None:
                _prefix_maxima(diagnostics, state, 'state')
                diagnostics['prefix_steps'] += 1
        return state

    def model_spec(self):
        dtype = self._validate_parameters()
        observer = self.mode in ('long', 'short')
        return {'version': VERSION, 'mode': self.mode, 'transition_kind': 'dense_mlp',
                'seed': self.cell.initialization_seed, 'dtype': str(dtype).removeprefix('torch.'),
                'parameter_count': self.parameter_count, 'transition_parameter_count': 590,
                'initializer_parameter_count': self.added_parameter_count,
                'trainable_parameter_count': self.trainable_parameter_count,
                'frozen_parameter_count': self.parameter_count-self.trainable_parameter_count,
                'parameter_bytes': sum(p.numel()*p.element_size() for p in self.parameters()),
                'buffer_bytes': 0, 'state_scalars': 12, 'state_bytes_per_stream': 12*self.cell.input_matrix.element_size(),
                'structurally_inactive_parameter_count': 0, 'context_length': CONTEXT,
                'gain_initialization': '[I;0]' if observer else 'absent',
                'head_initialization': 'all zero' if self.mode == 'temporal' else 'absent',
                'prefix_transition_count': 30 if self.mode == 'long' else 1 if self.mode == 'short' else 0,
                'prefix_operator_preparations': 1 if observer else 0,
                'conditioning': {'long': 'z1=[q1,q1-q0]; correct q2..q31 using u1..u30',
                                 'short': 'z30=[q30,q30-q29]; predict with u30 and correct q31 once',
                                 'last_two': 'unchanged HistoryInitializedDense last_two',
                                 'temporal': 'unchanged HistoryInitializedDense temporal_affine'}[self.mode],
                'conditioning_last_torque_index': None if self.mode == 'last_two' else 30,
                'timing': 'condition ends at time31; first forecast u31 predicts q32; no future observations',
                'state_semantics': 'bottom six coordinates are learned state, not verified physical velocity',
                'stability_scope': 'No observer-loop, incremental-stability, gradient, covariance or calibration guarantee.',
                'diagnostic_keys': list(DIAGNOSTIC_KEYS),
                'diagnostic_scope': 'Caller-owned finite prefix maxima; controls report final initialized state and zero corrections.',
                'persistent_cache': False, 'retained_trajectory': False,
                'storage_scope': 'parameters and caller state; normalization, requests, gradients, optimizer and workspace excluded',
                'transition_spec': self.cell.model_spec()}
