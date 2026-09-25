"""Position-only initial correction for a frozen, qualified robot transition.

This changes the initial gain to [I;0]. It is a conventional constant-gain
observer experiment, not a Kalman estimator or a stability guarantee. Optional
caller-owned diagnostics inspect finite prefix states without changing them.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from openjev.research.robot_observer_initializer import CONTEXT, FrozenObserverInitializer
from openjev.research.structured_robot_transition import StructuredRobotTransition, require

VERSION = 'robot-position-observer-v1'
MODES = ('learned', 'fixed')
DIAGNOSTIC_KEYS = ('prefix_steps', 'max_state_abs', 'max_state_norm64', 'max_innovation_abs', 'max_innovation_norm64')


def _position_gain(dtype):
    return torch.cat((torch.eye(6, dtype=dtype), torch.zeros((6, 6), dtype=dtype)))


def _prefix_maxima(diagnostics, value, kind):
    """Detached float64 reductions only; caller has already checked finiteness."""
    with torch.no_grad():
        numeric = value.detach().to(torch.float64)
        diagnostics['max_'+kind+'_abs'] = max(diagnostics['max_'+kind+'_abs'], float(numeric.abs().max()))
        diagnostics['max_'+kind+'_norm64'] = max(diagnostics['max_'+kind+'_norm64'],
                                                float(torch.linalg.vector_norm(numeric, dim=-1).max()))


def gradient_diagnostics(model):
    """Inspect existing PRE-CLIPPING gradients, without mutation or repair.

    ``numel`` counts entries with a present gradient, including any frozen
    parameter that incorrectly received one. Missing gradients are not zeros
    and are not counted. An empty collection reports zero magnitudes and
    all_finite=True; the trainer must separately enforce its expected roster.
    Any NaN/Inf entry makes both magnitudes None. The norm is accumulated in
    float64 from maximum-scaled pieces; unrepresentable norms are None while
    all_finite continues to describe the entries, not their aggregate norm.
    """
    require(isinstance(model, nn.Module), 'gradient diagnostics require a torch module')
    gradients = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    require(all(g.device.type == 'cpu' and g.layout == torch.strided and g.dtype in (torch.float32, torch.float64)
                for g in gradients), 'dense CPU float32/float64 gradients required')
    numel = sum(g.numel() for g in gradients)
    nonfinite = sum(int((~torch.isfinite(g)).sum()) for g in gradients)
    result = {'numel': numel, 'nonfinite_count': nonfinite, 'max_abs': None, 'norm64': None, 'all_finite': nonfinite == 0}
    if nonfinite:
        return result
    maximum = max((float(g.abs().max()) for g in gradients if g.numel()), default=0.)
    result['max_abs'] = maximum
    if maximum == 0:
        result['norm64'] = 0.
    else:
        squares = math.fsum(float((g.to(torch.float64)/maximum).square().sum()) for g in gradients)
        norm = maximum*math.sqrt(squares)
        result['norm64'] = norm if math.isfinite(norm) else None
    return result


class FrozenPositionObserver(FrozenObserverInitializer):
    """Same public forecast API and cell.* keys as the qualified observer.

    The learned gain accepts arbitrary trained finite 12x6 values. The fixed
    gain is a checked buffer equal to [I;0]. All 590 cell parameters remain
    frozen, while autograd traverses the complete prefix and forecast.
    """

    def __init__(self, seed=0, mode='learned', *, dtype=torch.float32):
        require(mode in MODES, 'declared position observer mode required')
        super().__init__(seed, 'observer_learned' if mode == 'learned' else 'observer_fixed', dtype=dtype)
        self.mode = mode
        with torch.no_grad():
            self.gain.copy_(_position_gain(dtype))

    @property
    def added_parameter_count(self):
        return 72 if self.mode == 'learned' else 0

    def _validate_parameters(self):
        require(self.mode in MODES and self.initializer == 'last_two', 'fixed position observer mode/feature identity required')
        require(set(self._modules) == {'cell'} and type(self.cell) is StructuredRobotTransition
                and self.cell.kind == 'dense_mlp', 'exact frozen dense_mlp cell required')
        dtype = self.cell._validate_parameters()
        learned = self.mode == 'learned'
        require(set(self._parameters) == ({'gain'} if learned else set())
                and set(self._buffers) == (set() if learned else {'gain'})
                and set(dict(self.named_buffers())) == (set() if learned else {'gain'}), 'exact position observer gain storage roster')
        self.cell._tensor(self.gain, (12, 6), dtype, 'observation correction gain')
        if not learned:
            require(not self.gain.requires_grad and torch.equal(self.gain, _position_gain(dtype)), 'fixed position observer gain cannot change')
        require(self.cell.parameter_count == 590 and all(not p.requires_grad and p.grad is None for p in self.cell.parameters()),
                'frozen cell parameters without stored gradients required')
        return dtype

    def condition(self, q_context, u_context, diagnostics=None):
        """Assimilate t=2..31 using u[t-1]; optional empty dict gets five scalars.

        Maxima cover the batch's initial, predicted and corrected finite states,
        and each finite innovation; norm64 is the maximum per-row L2 norm.
        prefix_steps counts completed corrections, not batch examples. On a
        numerical failure the caller keeps the maxima observed before failure.
        No trajectory/tensor is stored in the dict or on the model. With None,
        no diagnostic conversion, norm, reduction or scalar extraction occurs.
        """
        require(diagnostics is None or (type(diagnostics) is dict and not diagnostics), 'diagnostics must be None or an empty caller-owned dict')
        dtype = self._validate_parameters()
        require(isinstance(q_context, torch.Tensor) and q_context.ndim == 3 and q_context.shape[0] > 0
                and tuple(q_context.shape[1:]) == (CONTEXT, 6), 'observer context shape [positive B,32,6] required')
        shape = tuple(q_context.shape)
        self.cell._tensor(q_context, shape, dtype, 'observed positions')
        self.cell._tensor(u_context, shape, dtype, 'observed torques')
        if diagnostics is not None:
            diagnostics.update(dict.fromkeys(DIAGNOSTIC_KEYS, 0.))
            diagnostics['prefix_steps'] = 0
        state = torch.cat((q_context[:, 1], q_context[:, 1]-q_context[:, 0]), dim=-1)
        self.cell._finite(state[:, :6], state)
        if diagnostics is not None:
            _prefix_maxima(diagnostics, state, 'state')
        prepared = self.cell._prepare()
        for t in range(2, CONTEXT):
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
        spec = super().model_spec()
        spec.update(version=VERSION, gain_initialization='[I;0]', gain_buffer_scalars=72 if self.mode == 'fixed' else 0,
                    stability_scope='No learned-observer, incremental-stability, covariance or calibration guarantee.',
                    diagnostic_scope='Optional caller-owned maxima over finite initial/prior/corrected prefix states and innovations; no trajectory.',
                    diagnostic_keys=list(DIAGNOSTIC_KEYS))
        return spec
