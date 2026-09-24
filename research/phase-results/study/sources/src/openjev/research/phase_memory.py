"""Small amplitude-dependent oscillatory memory and matched controls.

Each input advances two damped two-dimensional states before producing its
output. Rotation preserves state norm, so bounded inputs imply bounded states
when all learned parameters are finite. This is not a claim of incremental
stability, bounded gradients, or an exact physical realization.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

VERSION = 'phase-memory-v1'
ARMS = ('energy_phase', 'fixed_phase', 'nonlinear_readout')
RHO_CAP = .9999
BASE_KEYS = ('rho_logits', 'angle_logits', 'input_weights', 'readout_weights',
             'feedthrough', 'bias')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class PhaseMemory(nn.Module):
    """CPU float32 by default; float64 is available for numerical qualification.

    ``forward`` accepts [batch,time,1] and returns predictions [batch,time]
    plus final state [batch,2,2]. ``step`` accepts input [batch]. Neither API
    accepts measured outputs or mutates its supplied state. State is explicit;
    the module never retains a trajectory or a hidden state between calls.
    """

    def __init__(self, arm, seed, *, dtype=torch.float32):
        super().__init__()
        _require(arm in ARMS, 'unknown phase-memory arm')
        _require(type(seed) is int and 0 <= seed < 2**63, 'nonnegative local integer seed required')
        _require(dtype in (torch.float32, torch.float64), 'float32 or float64 required')
        self.arm = arm
        self.initialization_seed = seed
        generator = torch.Generator(device='cpu').manual_seed(seed)
        radius = torch.tensor([.99, .95], dtype=dtype) / RHO_CAP
        self.rho_logits = nn.Parameter(torch.log(radius / (1 - radius)))
        frequency = torch.tensor([.1, .3], dtype=dtype)
        self.angle_logits = nn.Parameter(torch.log(frequency / (1 - frequency)))
        self.input_weights = nn.Parameter(.1 * torch.randn(2, 2, generator=generator, dtype=dtype))
        self.readout_weights = nn.Parameter(.1 * torch.randn(2, 2, generator=generator, dtype=dtype))
        self.feedthrough = nn.Parameter(torch.zeros((), dtype=dtype))
        self.bias = nn.Parameter(torch.zeros((), dtype=dtype))
        if arm == 'energy_phase':
            self.phase_logits = nn.Parameter(torch.zeros(2, dtype=dtype))
            self.energy_scales_raw = nn.Parameter(torch.full((2,), math.log(math.expm1(1.)), dtype=dtype))
        elif arm == 'nonlinear_readout':
            self.cubic_readout_weights = nn.Parameter(torch.zeros(2, 2, dtype=dtype))

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def state_scalars(self):
        return 4

    def _validate_parameters(self):
        _require(self.arm in ARMS, 'unknown phase-memory arm')
        dtype = self.rho_logits.dtype
        _require(dtype in (torch.float32, torch.float64), 'float32 or float64 parameters required')
        _require(all(p.device.type == 'cpu' and p.dtype == dtype and bool(torch.isfinite(p).all())
                     for p in self.parameters()), 'finite, common-dtype CPU parameters required')
        return dtype

    def initial_state(self, batch):
        dtype = self._validate_parameters()
        _require(type(batch) is int and batch > 0, 'positive batch size required')
        return torch.zeros(batch, 2, 2, dtype=dtype, device='cpu')

    @staticmethod
    def _validate_tensor(tensor, shape, dtype, name):
        _require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu'
                 and tensor.dtype == dtype and tuple(tensor.shape) == tuple(shape)
                 and bool(torch.isfinite(tensor).all()), f'finite CPU {name} with exact shape/dtype required')

    def _fields(self):
        # One forward call owns these graph-attached tensors. Nothing is cached
        # across calls, so optimizer updates cannot leave stale parameters.
        radius = RHO_CAP * torch.sigmoid(self.rho_logits)
        angle = math.pi * torch.sigmoid(self.angle_logits)
        if self.arm == 'energy_phase':
            return radius, angle, .5 * torch.tanh(self.phase_logits), F.softplus(self.energy_scales_raw)
        return radius, None, torch.cos(angle), torch.sin(angle)

    def _step(self, inputs, state, fields):
        radius, angle, first_field, second_field = fields
        if self.arm == 'energy_phase':
            energy = state.square().sum(dim=-1)
            angle = angle + first_field * torch.tanh(second_field * energy)
            cosine, sine = torch.cos(angle), torch.sin(angle)
        else:
            cosine, sine = first_field, second_field
        first, second = state[..., 0], state[..., 1]
        rotated = torch.stack((cosine * first - sine * second,
                               sine * first + cosine * second), dim=-1)
        updated = radius[None, :, None] * rotated + inputs[:, None, None] * self.input_weights[None]
        prediction = (updated * self.readout_weights).sum(dim=(-2, -1))
        if self.arm == 'nonlinear_readout':
            cubic = updated * updated.square().sum(dim=-1, keepdim=True)
            prediction = prediction + (cubic * self.cubic_readout_weights).sum(dim=(-2, -1))
        prediction = prediction + self.feedthrough * inputs + self.bias
        return prediction, updated

    def step(self, inputs, state):
        dtype = self._validate_parameters()
        _require(isinstance(inputs, torch.Tensor) and inputs.ndim == 1 and len(inputs) > 0,
                 'step input must be a nonempty batch vector')
        self._validate_tensor(inputs, (len(inputs),), dtype, 'step input')
        self._validate_tensor(state, (len(inputs), 2, 2), dtype, 'state')
        prediction, updated = self._step(inputs, state, self._fields())
        _require(bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(updated).all()),
                 'nonfinite phase-memory output')
        return prediction, updated

    def forward(self, sequence, state=None):
        dtype = self._validate_parameters()
        _require(isinstance(sequence, torch.Tensor) and sequence.ndim == 3
                 and sequence.shape[0] > 0 and sequence.shape[2] == 1,
                 'sequence must have shape [positive batch,time,1]')
        batch, length, _ = sequence.shape
        self._validate_tensor(sequence, (batch, length, 1), dtype, 'sequence')
        if state is None:
            state = torch.zeros(batch, 2, 2, dtype=dtype)
        else:
            self._validate_tensor(state, (batch, 2, 2), dtype, 'state')
        current, outputs = state, []
        fields = self._fields()
        for index in range(length):
            prediction, current = self._step(sequence[:, index, 0], current, fields)
            outputs.append(prediction)
        predictions = torch.stack(outputs, dim=1) if outputs else sequence.new_empty(batch, 0)
        final = current if outputs else current.clone()
        _require(bool(torch.isfinite(predictions).all()) and bool(torch.isfinite(final).all()),
                 'nonfinite phase-memory output')
        return predictions, final

    def parameter_metadata(self):
        dtype = self._validate_parameters()
        return {'version': VERSION, 'arm': self.arm, 'seed': self.initialization_seed,
                'dtype': str(dtype).removeprefix('torch.'), 'parameter_count': self.parameter_count,
                'trainable_parameter_count': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'parameter_bytes': sum(p.numel() * p.element_size() for p in self.parameters()),
                'state_scalars': 4, 'state_bytes_per_stream': 4 * self.rho_logits.element_size(),
                'buffers': 0, 'retained_trajectory': False,
                'timing': 'input advances state before same-index output',
                'stability_scope': 'bounded-input bounded-state; not incremental or gradient stability'}

    def export_numpy(self):
        """Owned parameter arrays in canonical state_dict order; no recurrent state."""
        self._validate_parameters()
        return {name: tensor.detach().cpu().numpy().copy() for name, tensor in self.state_dict().items()}

    def export_json(self):
        """JSON-compatible metadata and exact float32/float64 parameter values."""
        return {'metadata': self.parameter_metadata(),
                'parameters': {name: tensor.detach().cpu().tolist()
                               for name, tensor in self.state_dict().items()}}
