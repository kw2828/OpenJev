"""Four versus twelve reflections with the frozen structured-cell equations.

Only reflector capacity changes. Four reflections delegate to the original
numerical and gradient paths. Twelve retain its first four initialized vectors
and append four independent, initially identical pairs. Pair cancellation gives
the same .999I initial metric-space operator in real arithmetic, not bitwise
equality between different reflection counts. Forward never builds dense
operators or retains a prepared cache. This is a capacity comparison, not a
novel Householder construction or a nonlinear contraction guarantee.
"""
from __future__ import annotations

import torch
from torch import nn

from openjev.research.compact_robot_gate import SHAPES
from openjev.research.structured_robot_transition import (
    ANCHORS,
    RADIUS,
    Prepared,
    StructuredRobotTransition,
    require,
)

VERSION = 'reflection-capacity-v1'
REFLECTION_COUNTS = (4, 12)
EXTRA_SEED_XOR = 0x43415031


class ReflectionCapacityRobotTransition(StructuredRobotTransition):
    """Use condition(q,u), then forward(future_u,state), with explicit state12.

    ``kind`` stays householder for inherited equation guards; ``reflections``
    and model_spec identify the capacity. No extra parameter names or buffers
    are introduced. Counts4/12 have630/806 trainable scalar parameters.
    """

    def __init__(self, reflections=4, seed=0, *, dtype=torch.float32):
        require(type(reflections) is int and reflections in REFLECTION_COUNTS,
                'reflection count must be exactly4 or12')
        super().__init__('householder', seed, dtype=dtype)
        self.reflections = reflections
        if reflections == 12:
            rng = torch.Generator(device='cpu').manual_seed(seed ^ EXTRA_SEED_XOR)
            pairs = .01 * torch.randn((2, 4, 11), generator=rng, dtype=dtype)
            # Replacing an existing Parameter preserves its registration order.
            self.reflection_raw = nn.Parameter(torch.cat(
                (self.reflection_raw.detach(), pairs.repeat_interleave(2, dim=1)), dim=1))

    @property
    def anchors(self):
        return ANCHORS * (self.reflections // 4)

    def _validate_parameters(self):
        require(self.kind == 'householder' and type(self.reflections) is int
                and self.reflections in REFLECTION_COUNTS, 'declared reflection capacity required')
        if self.reflections == 4:
            return super()._validate_parameters()
        shapes = {'input_matrix': (2, 12, 6), 'expert_bias': (2, 12), 'scale_raw': (12,),
                  'reflection_raw': (2, 12, 11), 'decay_raw': (2, 12)}
        shapes.update({'gate.' + key: shape for key, shape in SHAPES.items()})
        values = dict(self.named_parameters())
        require(set(values) == set(shapes) and not dict(self.named_buffers()),
                'exact structured parameter/buffer roster')
        dtype = self.input_matrix.dtype
        require(dtype in (torch.float32, torch.float64), 'float32 or float64 parameters required')
        for key, shape in shapes.items():
            self._tensor(values[key], shape, dtype, key)
        return dtype

    def _prepare(self):
        if self.reflections == 4:
            return super()._prepare()
        scale = torch.exp(3 * torch.tanh(self.scale_raw))
        free = torch.tanh(self.reflection_raw)
        vectors = torch.stack([
            torch.cat((free[:, k, :anchor], free.new_ones(2, 1), free[:, k, anchor:]), dim=-1)
            for k, anchor in enumerate(self.anchors)], dim=1)
        return Prepared(scale, None, None, vectors, RADIUS * torch.sigmoid(self.decay_raw))

    def _transition(self, state, mixing, prepared):
        if self.reflections == 4:
            return super()._transition(state, mixing, prepared)
        value = state * prepared.scale
        vectors = self._vectors(mixing, prepared)
        for k in range(self.reflections):
            value = self._reflect(value, vectors[:, k])
        decay = mixing @ prepared.decay
        return decay * value

    def operators(self, mixing):
        """Explicit diagnostic only; dense matrices are never used by forward."""
        if self.reflections == 4:
            return super().operators(mixing)
        dtype = self._validate_parameters()
        require(isinstance(mixing, torch.Tensor) and mixing.ndim == 2 and mixing.shape[0] > 0,
                'mixing shape [B,2]')
        self._tensor(mixing, (len(mixing), 2), dtype, 'mixing')
        require(bool((mixing >= 0).all()) and bool(torch.allclose(
            mixing.sum(1), mixing.new_ones(len(mixing)), atol=1e-6, rtol=0)), 'simplex mixing')
        prepared = self._prepare()
        vectors = self._vectors(mixing, prepared)
        matrix = torch.eye(12, dtype=dtype).expand(len(mixing), 12, 12).clone()
        for k in range(self.reflections):
            vector = vectors[:, k]
            reflected = torch.eye(12, dtype=dtype)[None] - 2 * vector[:, :, None] * vector[:, None, :] / vector.square().sum(-1)[:, None, None]
            matrix = reflected @ matrix
        decay = mixing @ prepared.decay
        matrix = decay[:, :, None] * matrix
        return {'matrix': matrix, 'scale': prepared.scale, 'mixing': mixing.clone(),
                'reflection_vectors': vectors, 'decay': decay, 'raw_spectral_norm': None}

    def model_spec(self):
        result = super().model_spec()
        result.update(version=VERSION, reflections=self.reflections, anchors=list(self.anchors),
                      extra_pair_seed_xor=EXTRA_SEED_XOR if self.reflections == 12 else None)
        return result
