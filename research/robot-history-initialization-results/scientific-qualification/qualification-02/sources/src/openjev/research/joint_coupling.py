"""Input-driven AR2 residual coupling with explicit measured-prefix conditioning.

The dense linear base is supplied by the caller, normally a FIT-only ridge
estimate. The three arms share that base and paired residual initialization.
The graph describes only a residual approximation, not a mechanical law. This
module neither fits a base nor guarantees bounded or incrementally stable runs.
"""
from __future__ import annotations

import math
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import nn

if TYPE_CHECKING:
    import numpy as np

VERSION = 'joint-coupling-v1'
ARMS = ('chain_memory', 'rewired_memory', 'chain_instant')
NODES = 6
EDGE_COUNT = 10
CHAIN = (0, 1, 2, 3, 4, 5)
REWIRED = (0, 1, 3, 2, 4, 5)
PARAMETER_SHAPES = MappingProxyType({
    'base_weight': (6, 25), 'local_gain': (6,), 'local_weights': (6, 3),
    'local_bias': (6,), 'edge_encoder': (10, 6), 'edge_bias': (10,),
    'decay_logits': (10,), 'edge_gain': (6,),
})


class JointState(NamedTuple):
    """At time t: q_t, q_(t-1), u_(t-1), and m_t for memory arms only."""

    q: torch.Tensor
    prevq: torch.Tensor
    prevu: torch.Tensor
    edges: torch.Tensor | None


class _Fields(NamedTuple):
    decay: torch.Tensor
    degrees: torch.Tensor


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _edges(arm: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    # Row 0 is receiver, row 1 sender. Reciprocal directed slots stay paired
    # across graphs; physical input/output joint identities are not relabeled.
    order = REWIRED if arm == 'rewired_memory' else CHAIN
    pairs = [pair for left, right in pairwise(order)
             for pair in ((left, right), (right, left))]
    return tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs)


class JointCoupling(nn.Module):
    """Six-joint AR2 plus local and edge residuals, 266 trainable scalars.

    ``step(u_t, state_t)`` predicts q_(t+1), then shifts the explicit state.
    ``condition(q_context, u_context)`` accepts equal [batch,C,6] arrays, C>=2.
    It uses only measured q and torque through index C-2 to build edge memory.
    The returned state holds q[C-1], q[C-2], u[C-2]. The next forward input must
    start at u[C-1], whose target is q[C]. No future output labels are accepted.
    The instantaneous arm retains 18 state values; memory arms retain 28.
    """

    def __init__(self, arm: str, seed: int, base_coefficients: torch.Tensor,
                 *, dtype: torch.dtype = torch.float32):
        super().__init__()
        _require(type(arm) is str and arm in ARMS, 'unknown joint-coupling arm')
        _require(type(seed) is int and 0 <= seed < 2**63, 'nonnegative local integer seed required')
        _require(dtype in (torch.float32, torch.float64), 'float32 or float64 required')
        self._validate_tensor(base_coefficients, (6, 25), dtype, 'base coefficients')
        self.arm = arm
        self.initialization_seed = seed
        generator = torch.Generator(device='cpu').manual_seed(seed)
        self.base_weight = nn.Parameter(base_coefficients.detach().clone())
        self.local_gain = nn.Parameter(torch.zeros(6, dtype=dtype))
        self.local_weights = nn.Parameter(.05 * torch.randn((6, 3), generator=generator, dtype=dtype))
        self.local_bias = nn.Parameter(torch.zeros(6, dtype=dtype))
        self.edge_encoder = nn.Parameter(.05 * torch.randn((10, 6), generator=generator, dtype=dtype))
        self.edge_bias = nn.Parameter(torch.zeros(10, dtype=dtype))
        self.decay_logits = nn.Parameter(torch.full((10,), math.log(.9 / .1), dtype=dtype))
        self.edge_gain = nn.Parameter(torch.zeros(6, dtype=dtype))
        self.register_buffer('edge_index', torch.tensor(_edges(arm), dtype=torch.int64))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def state_scalars(self) -> int:
        return 18 if self.arm == 'chain_instant' else 28

    def _validate_parameters(self) -> torch.dtype:
        _require(self.arm in ARMS, 'unknown joint-coupling arm')
        parameters = dict(self.named_parameters())
        _require(set(parameters) == set(PARAMETER_SHAPES), 'complete parameter roster required')
        dtype = self.base_weight.dtype
        _require(dtype in (torch.float32, torch.float64), 'float32 or float64 parameters required')
        for name, shape in PARAMETER_SHAPES.items():
            self._validate_tensor(parameters[name], shape, dtype, 'parameter ' + name)
        _require(set(dict(self.named_buffers())) == {'edge_index'} and self.edge_index.device.type == 'cpu'
                 and self.edge_index.dtype == torch.int64 and tuple(self.edge_index.shape) == (2, 10)
                 and torch.equal(self.edge_index, torch.tensor(_edges(self.arm), dtype=torch.int64)),
                 'fixed declared graph buffer required')
        return dtype

    @staticmethod
    def _validate_tensor(value: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype, name: str) -> None:
        _require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == dtype
                 and tuple(value.shape) == shape and bool(torch.isfinite(value).all()),
                 'finite CPU tensor with exact shape/dtype: ' + name)

    def _validate_state(self, state: JointState, batch: int, dtype: torch.dtype) -> None:
        _require(isinstance(state, JointState), 'explicit JointState required')
        for name in ('q', 'prevq', 'prevu'):
            self._validate_tensor(getattr(state, name), (batch, 6), dtype, name)
        if self.arm == 'chain_instant':
            _require(state.edges is None, 'instantaneous control has no recurrent edge state')
        else:
            self._validate_tensor(state.edges, (batch, 10), dtype, 'edge state')

    def initial_state(self, batch: int) -> JointState:
        """All-zero fixture state; real forecasts should use measured condition()."""
        dtype = self._validate_parameters()
        _require(type(batch) is int and batch > 0, 'positive batch size required')
        return JointState(*(torch.zeros(batch, 6, dtype=dtype) for _ in range(3)),
                          None if self.arm == 'chain_instant' else torch.zeros(batch, 10, dtype=dtype))

    def _fields(self) -> _Fields:
        return _Fields(torch.sigmoid(self.decay_logits),
                       torch.bincount(self.edge_index[0], minlength=6).to(self.base_weight.dtype))

    def _messages(self, inputs: torch.Tensor, q: torch.Tensor, prevq: torch.Tensor,
                  edges: torch.Tensor | None, fields: _Fields) -> torch.Tensor:
        receivers, senders = self.edge_index[0], self.edge_index[1]
        delta = q - prevq
        features = torch.stack((q[:, receivers], delta[:, receivers], q[:, senders],
                                delta[:, senders], inputs[:, receivers], inputs[:, senders]), dim=-1)
        encoded = torch.tanh((features * self.edge_encoder).sum(dim=-1) + self.edge_bias)
        messages = (1 - fields.decay) * encoded
        if self.arm != 'chain_instant':
            messages = messages + fields.decay * edges
        return messages

    def _step(self, inputs: torch.Tensor, state: JointState, fields: _Fields) -> tuple[torch.Tensor, JointState]:
        features = torch.cat((state.q, state.prevq, inputs, state.prevu,
                              inputs.new_ones(inputs.shape[0], 1)), dim=-1)
        base = features @ self.base_weight.T
        local_features = torch.stack((state.q, state.q - state.prevq, inputs), dim=-1)
        local = torch.tanh((local_features * self.local_weights).sum(dim=-1) + self.local_bias)
        messages = self._messages(inputs, state.q, state.prevq, state.edges, fields)
        aggregate = torch.zeros_like(inputs).index_add(1, self.edge_index[0], messages) / fields.degrees
        prediction = base + self.local_gain * local + self.edge_gain * aggregate
        # Copies keep outputs/states owned without detaching their gradients.
        return prediction, JointState(prediction.clone(), state.q.clone(), inputs.clone(),
                                      None if self.arm == 'chain_instant' else messages)

    @staticmethod
    def _validate_output(output: torch.Tensor, state: JointState) -> None:
        _require(bool(torch.isfinite(output).all())
                 and all(bool(torch.isfinite(value).all()) for value in state if value is not None),
                 'nonfinite joint-coupling output; no clipping or repair')

    def condition(self, q_context: torch.Tensor, u_context: torch.Tensor) -> JointState:
        dtype = self._validate_parameters()
        _require(isinstance(q_context, torch.Tensor) and q_context.ndim == 3 and q_context.shape[0] > 0
                 and q_context.shape[1] >= 2 and q_context.shape[2] == 6,
                 'conditioning requires [positive batch,C>=2,6]')
        batch, length, _ = q_context.shape
        self._validate_tensor(q_context, (batch, length, 6), dtype, 'conditioning positions')
        self._validate_tensor(u_context, (batch, length, 6), dtype, 'conditioning torques')
        edges, fields = None, self._fields()
        if self.arm != 'chain_instant':
            edges = q_context.new_zeros(batch, 10)
            for index in range(1, length - 1):
                edges = self._messages(u_context[:, index], q_context[:, index], q_context[:, index - 1],
                                       edges, fields)
        state = JointState(q_context[:, -1].clone(), q_context[:, -2].clone(), u_context[:, -2].clone(), edges)
        self._validate_output(q_context.new_empty(batch, 0, 6), state)
        return state

    def step(self, inputs: torch.Tensor, state: JointState) -> tuple[torch.Tensor, JointState]:
        dtype = self._validate_parameters()
        _require(isinstance(inputs, torch.Tensor) and inputs.ndim == 2 and inputs.shape[0] > 0,
                 'step input requires shape [positive batch,6]')
        self._validate_tensor(inputs, (inputs.shape[0], 6), dtype, 'step input')
        self._validate_state(state, inputs.shape[0], dtype)
        output, final = self._step(inputs, state, self._fields())
        self._validate_output(output, final)
        return output, final

    def forward(self, sequence: torch.Tensor, state: JointState) -> tuple[torch.Tensor, JointState]:
        dtype = self._validate_parameters()
        _require(isinstance(sequence, torch.Tensor) and sequence.ndim == 3 and sequence.shape[0] > 0
                 and sequence.shape[2] == 6, 'sequence requires shape [positive batch,time,6]')
        batch, length, _ = sequence.shape
        self._validate_tensor(sequence, (batch, length, 6), dtype, 'sequence')
        self._validate_state(state, batch, dtype)
        current, outputs, fields = state, [], self._fields()
        for index in range(length):
            output, current = self._step(sequence[:, index], current, fields)
            outputs.append(output)
        predictions = torch.stack(outputs, dim=1) if outputs else sequence.new_empty(batch, 0, 6)
        final = current if outputs else JointState(*(value.clone() if value is not None else None for value in state))
        self._validate_output(predictions, final)
        return predictions, final

    def parameter_metadata(self) -> dict[str, object]:
        dtype = self._validate_parameters()
        return {'version': VERSION, 'arm': self.arm, 'seed': self.initialization_seed,
                'dtype': str(dtype).removeprefix('torch.'), 'parameter_count': self.parameter_count,
                'trainable_parameter_count': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'parameter_bytes': sum(p.numel() * p.element_size() for p in self.parameters()),
                'state_scalars': self.state_scalars,
                'state_bytes_per_stream': self.state_scalars * self.base_weight.element_size(),
                'buffer_bytes': self.edge_index.numel() * self.edge_index.element_size(),
                'buffer_shapes': {'edge_index': [2, 10]},
                'edge_order': [list(pair) for pair in zip(*_edges(self.arm), strict=True)],
                'base_feature_order': ['q', 'prevq', 'u_now', 'prevu', 'constant_one'],
                'retained_trajectory': False, 'persistent_cache': False,
                'timing': 'u_t with q_t and q_(t-1) predicts q_(t+1)',
                'initial_gradient_scope': 'zero gains block initial encoder/decay gradients; gains and base remain trainable',
                'stability_scope': 'no boundedness or incremental-stability guarantee',
                'storage_scope': 'parameters plus explicit caller state and graph buffer; external normalization and workspace excluded'}

    def export_numpy(self) -> dict[str, np.ndarray]:
        """Owned canonical state_dict arrays, including the charged graph buffer."""
        self._validate_parameters()
        return {name: value.detach().cpu().numpy().copy() for name, value in self.state_dict().items()}

    def export_json(self) -> dict[str, object]:
        """JSON-compatible named parameters, graph buffer and storage metadata."""
        return {'metadata': self.parameter_metadata(),
                'parameters': {name: value.detach().cpu().tolist() for name, value in self.named_parameters()},
                'buffers': {'edge_index': self.edge_index.tolist()}}
