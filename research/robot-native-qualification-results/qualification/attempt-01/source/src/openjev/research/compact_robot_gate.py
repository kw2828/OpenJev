"""Compact current-input equivalent of a zero-hidden-state GRU gate.

For h_previous=0, weight_hh is inactive. Reset/update input and hidden biases
enter only through their sum and are folded. The candidate hidden bias must
remain separate because the reset gate multiplies it. The resulting 338 stored
parameters represent the same inference function as GRUCell(12,8)+Linear(8,2),
within ordinary floating-point reassociation error, not bitwise equality.

The seeded constructor pairs the prior BoundedLPV instant scheduler/head
initialization, including its zero head. Every stored coordinate can affect the
function, although that zero head initially blocks encoder gradients. Folding
changes the training parameterization: equal learning rates do not imply equal
optimizer trajectories to the redundant two-bias representation. This is a
conventional algebraic simplification, not a new recurrent architecture.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

VERSION = 'compact-robot-gate-v1'
INPUTS, HIDDEN, OUTPUTS = 12, 8, 2
SHAPES = {'input_weight': (24, 12), 'reset_update_bias': (16,),
          'candidate_input_bias': (8,), 'candidate_hidden_bias': (8,),
          'head_weight': (2, 8), 'head_bias': (2,)}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _tensor(value, shape, dtype, name):
    _require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape
             and value.device.type == 'cpu' and value.layout == torch.strided
             and value.dtype == dtype and bool(torch.isfinite(value).all()),
             'finite CPU tensor of declared shape/dtype required: ' + name)


class CompactResetGate(nn.Module):
    """Trainable, stateless 12-input/two-output reset-GRU gate."""

    def __init__(self, seed=0, *, dtype=torch.float32):
        super().__init__()
        _require(type(seed) is int and 0 <= seed < 2**63, 'nonnegative integer seed required')
        _require(dtype in (torch.float32, torch.float64), 'float32 or float64 required')
        # Match BoundedLPV's independent scheduler RNG scope exactly. Temporary
        # construction parameters are discarded, not retained by this module.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            gru = nn.GRUCell(INPUTS, HIDDEN, dtype=dtype)
            head = nn.Linear(HIDDEN, OUTPUTS, dtype=dtype)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
        self.initialization_seed = seed
        self._copy_parameters(gru, head)

    @classmethod
    def from_prior(cls, gru, head):
        """Copy standard CPU GRUCell/Linear weights; retain no source storage."""
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.initialization_seed = None
        result._copy_parameters(gru, head)
        return result

    def _copy_parameters(self, gru, head):
        _require(type(gru) is nn.GRUCell and gru.input_size == INPUTS and gru.hidden_size == HIDDEN
                 and gru.bias, 'standard GRUCell(12,8,bias=True) required')
        _require(type(head) is nn.Linear and head.in_features == HIDDEN and head.out_features == OUTPUTS
                 and head.bias is not None, 'standard Linear(8,2,bias=True) required')
        expected = {'weight_ih': (24, 12), 'weight_hh': (24, 8), 'bias_ih': (24,), 'bias_hh': (24,)}
        _require(set(dict(gru.named_parameters())) == set(expected) and not dict(gru.named_buffers()),
                 'exact prior GRU parameter/buffer roster')
        _require(set(dict(head.named_parameters())) == {'weight', 'bias'} and not dict(head.named_buffers()),
                 'exact prior head parameter/buffer roster')
        dtype = gru.weight_ih.dtype
        _require(dtype in (torch.float32, torch.float64), 'prior float32 or float64 required')
        for name, shape in expected.items():
            _tensor(getattr(gru, name), shape, dtype, 'gru.' + name)
        _tensor(head.weight, (2, 8), dtype, 'head.weight')
        _tensor(head.bias, (2,), dtype, 'head.bias')
        with torch.no_grad():
            values = {'input_weight': gru.weight_ih,
                      'reset_update_bias': gru.bias_ih[:16] + gru.bias_hh[:16],
                      'candidate_input_bias': gru.bias_ih[16:],
                      'candidate_hidden_bias': gru.bias_hh[16:],
                      'head_weight': head.weight, 'head_bias': head.bias}
            for name, value in values.items():
                setattr(self, name, nn.Parameter(value.detach().clone()))
        self._validate()

    def _validate(self):
        actual = dict(self.named_parameters())
        _require(set(actual) == set(SHAPES) and not dict(self.named_buffers()), 'exact compact parameter/buffer roster')
        dtype = self.input_weight.dtype
        _require(dtype in (torch.float32, torch.float64), 'float32 or float64 parameters required')
        for name, shape in SHAPES.items():
            _tensor(actual[name], shape, dtype, name)
        return dtype

    def forward(self, x):
        """Map finite CPU [12] or [B,12] inputs to corresponding probabilities."""
        dtype = self._validate()
        _require(isinstance(x, torch.Tensor) and x.ndim in (1, 2) and x.shape[-1] == INPUTS
                 and (x.ndim == 1 or x.shape[0] > 0), 'input shape [12] or nonempty[B,12] required')
        _tensor(x, tuple(x.shape), dtype, 'input')
        probabilities = self._forward_unchecked(x)
        _require(bool(torch.isfinite(probabilities).all()), 'nonfinite compact gate output; no repair')
        return probabilities

    def _forward_unchecked(self, x):
        """Pure tensor kernel for an enclosing validated rollout.

        Caller owns parameter/input validation and finite-output checks. This
        avoids duplicating boundary checks at every step without caching any
        tensors or changing the differentiable function.
        """
        affine = F.linear(x, self.input_weight)
        reset_update = torch.sigmoid(affine[..., :16] + self.reset_update_bias)
        reset, update = reset_update[..., :8], reset_update[..., 8:]
        candidate = torch.tanh(affine[..., 16:] + self.candidate_input_bias + reset * self.candidate_hidden_bias)
        hidden = (1 - update) * candidate
        return torch.softmax(F.linear(hidden, self.head_weight, self.head_bias), dim=-1)

    def spec(self):
        dtype = self._validate()
        count = sum(value.numel() for value in self.parameters())
        return {'version': VERSION, 'seed': self.initialization_seed, 'dtype': str(dtype).removeprefix('torch.'),
                'parameter_count': count, 'parameter_bytes': sum(value.numel()*value.element_size() for value in self.parameters()),
                'prior_parameter_count': 546, 'removed_inactive_parameters': 192, 'folded_redundant_parameters': 16,
                'structurally_inactive_parameters': 0, 'state_scalars': 0, 'buffer_bytes': 0,
                'initialization': 'prior paired seed and zero head' if self.initialization_seed is not None else 'owned copy of supplied prior gate',
                'equivalence': 'reset-GRU inference within floating-point reassociation error, not bitwise identity',
                'training_scope': 'bias folding changes optimization coordinates; no same-optimizer-trajectory claim',
                'storage_scope': 'parameters only; request arrays, gradients, optimizer and temporary workspace excluded'}
