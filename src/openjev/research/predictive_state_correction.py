"""Block-coordinate predictive-inference reference with explicit caller state.

This is established recurrent prediction plus a masked observation-loss gradient,
not a novelty, calibrated-posterior or global stability claim. Only arrived
observations may correct state; forecast inputs alone advance it afterwards.
Scalar-output or rank-one decoders can make the selected block independent of
residual direction. A SISO task cannot identify adaptive observation routing.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

MODES = ('dense', 'selective', 'rewired')
HORIZONS = (1, 8, 32)
EPSILON = 1e-6
ETA = 1.0
VERSION = 'predictive-state-correction-v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class CorrectionParameters(NamedTuple):
    matrix: torch.Tensor
    bias: torch.Tensor
    denominator: torch.Tensor


class PredictiveStateCorrection(nn.Module):
    """condition(y[B,C,O],u[B,C-1,I]), then rollout(u_future[B,H,I],state).

    The first supplied transition input predicts the next supplied observation:
    observation index t pairs with input index t-1. After conditioning, the first
    future input predicts the next observation. These are positional indices;
    dataset adapters define physical timing. The state is [B,latent_dim];
    no observations, inputs, trajectories or prepared tensors persist internally.
    """

    def __init__(self, obs_dim, input_dim, *, mode='dense', latent_dim=60,
                 aux_width=64, seed=0, dtype=torch.float32):
        super().__init__()
        _require(mode in MODES, 'declared correction mode required')
        _require(all(type(v) is int and v > 0 for v in (obs_dim, input_dim, latent_dim, aux_width)),
                 'positive integer observation, input, latent and auxiliary dimensions required')
        _require(latent_dim % 3 == 0, 'latent dimension must divide into three equal blocks')
        _require(type(seed) is int and 0 <= seed < 2**63, 'nonnegative integer seed required')
        _require(dtype in (torch.float32, torch.float64), 'CPU float32 or float64 required')
        self.mode, self.obs_dim, self.input_dim = mode, obs_dim, input_dim
        self.latent_dim, self.aux_width, self.initialization_seed = latent_dim, aux_width, seed
        self.block_dim = latent_dim // 3
        # Identical construction order in every mode and no parent RNG mutation.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
            self.transition = nn.GRUCell(input_dim, latent_dim, dtype=dtype, device='cpu')
            self.observation = nn.Linear(latent_dim, obs_dim, dtype=dtype, device='cpu')
            self.initializer = nn.Linear(obs_dim, latent_dim, dtype=dtype, device='cpu')
            self.auxiliary = nn.ModuleDict({str(h): nn.Sequential(
                nn.Linear(self.block_dim + h*input_dim, aux_width, dtype=dtype, device='cpu'),
                nn.Tanh(), nn.Linear(aux_width, obs_dim, dtype=dtype, device='cpu')) for h in HORIZONS})

    def _shapes(self):
        n, o, i, w = self.latent_dim, self.obs_dim, self.input_dim, self.aux_width
        shapes = {'transition.weight_ih': (3*n, i), 'transition.weight_hh': (3*n, n),
                  'transition.bias_ih': (3*n,), 'transition.bias_hh': (3*n,),
                  'observation.weight': (o, n), 'observation.bias': (o,),
                  'initializer.weight': (n, o), 'initializer.bias': (n,)}
        for h in HORIZONS:
            shapes.update({f'auxiliary.{h}.0.weight': (w, self.block_dim+h*i),
                           f'auxiliary.{h}.0.bias': (w,), f'auxiliary.{h}.2.weight': (o, w),
                           f'auxiliary.{h}.2.bias': (o,)})
        return shapes

    @staticmethod
    def _tensor(value, shape, dtype, name):
        _require(isinstance(value, torch.Tensor) and value.layout == torch.strided
                 and value.device.type == 'cpu' and value.dtype == dtype and tuple(value.shape) == shape
                 and bool(torch.isfinite(value).all()), 'finite CPU tensor with exact shape/dtype: '+name)

    def _validate_parameters(self):
        _require(self.mode in MODES and self.latent_dim > 0 and self.latent_dim % 3 == 0
                 and self.block_dim == self.latent_dim//3, 'declared mode and three-block geometry')
        _require(type(self.transition) is nn.GRUCell and type(self.observation) is nn.Linear
                 and type(self.initializer) is nn.Linear and type(self.auxiliary) is nn.ModuleDict
                 and tuple(self.auxiliary) == tuple(map(str, HORIZONS)), 'declared module roster')
        _require(all(type(head) is nn.Sequential and len(head) == 3
                     and type(head[0]) is nn.Linear and type(head[1]) is nn.Tanh and type(head[2]) is nn.Linear
                     for head in self.auxiliary.values()), 'declared auxiliary MLP heads')
        values, shapes = dict(self.named_parameters()), self._shapes()
        _require(set(values) == set(shapes) and not dict(self.named_buffers()), 'exact parameter roster and no buffers')
        dtype = self.observation.weight.dtype
        _require(dtype in (torch.float32, torch.float64), 'CPU float32 or float64 parameters')
        for key, shape in shapes.items():
            self._tensor(values[key], shape, dtype, key)
        return dtype

    def _state(self, value, dtype):
        _require(isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[0] > 0,
                 'state shape [positive B,latent_dim] required')
        self._tensor(value, (len(value), self.latent_dim), dtype, 'state')

    def _prepare_correction(self):
        matrix = self.observation.weight
        denominator = matrix.square().sum() + EPSILON
        _require(bool(torch.isfinite(denominator)) and bool(denominator > 0), 'finite positive correction denominator')
        return CorrectionParameters(matrix, self.observation.bias, denominator)

    def _correct(self, prior, observed, prepared):
        error = observed - F.linear(prior, prepared.matrix, prepared.bias)
        gradient = error @ prepared.matrix
        _require(bool(torch.isfinite(error).all()) and bool(torch.isfinite(gradient).all()),
                 'nonfinite correction residual or gradient; no repair')
        if self.mode != 'dense':
            energies = gradient.reshape(len(prior), 3, self.block_dim).square().sum(-1)
            _require(bool(torch.isfinite(energies).all()), 'nonfinite correction block energy; no repair')
            block = energies.argmax(-1)  # First maximal index gives deterministic lowest-index ties.
            if self.mode == 'rewired':
                block = (block + 1) % 3
            mask = F.one_hot(block, 3).to(dtype=gradient.dtype).repeat_interleave(self.block_dim, dim=-1)
            gradient = gradient * mask  # Use the chosen destination block's OWN gradient.
        corrected = prior + ETA * gradient / prepared.denominator
        _require(bool(torch.isfinite(corrected).all()), 'nonfinite corrected state; no repair')
        return corrected

    def correct(self, prior, observed):
        """One differentiable masked gradient step for the fixed linear decoder.

        All three masks use eta=1 and denominator ||C||_F^2+1e-6. In exact
        arithmetic this cannot increase the current squared observation residual.
        This local property is not a guarantee about subsequent recurrent steps.
        """
        dtype = self._validate_parameters()
        self._state(prior, dtype)
        self._tensor(observed, (len(prior), self.obs_dim), dtype, 'arrived observation')
        return self._correct(prior, observed, self._prepare_correction())

    def condition(self, observations, inputs):
        dtype = self._validate_parameters()
        _require(isinstance(observations, torch.Tensor) and observations.ndim == 3
                 and observations.shape[0] > 0 and observations.shape[1] >= 1,
                 'observation context shape [positive B,C>=1,obs_dim] required')
        batch, context = observations.shape[:2]
        self._tensor(observations, (batch, context, self.obs_dim), dtype, 'observation context')
        self._tensor(inputs, (batch, context-1, self.input_dim), dtype, 'context transition inputs')
        state = torch.tanh(self.initializer(observations[:, 0]))
        prepared = self._prepare_correction()
        for t in range(1, context):
            prior = self.transition(inputs[:, t-1], state)
            _require(bool(torch.isfinite(prior).all()), 'nonfinite transition state; no repair')
            state = self._correct(prior, observations[:, t], prepared)
        _require(bool(torch.isfinite(state).all()), 'nonfinite conditioned state; no repair')
        return state

    def _step(self, state, inputs):
        current = self.transition(inputs, state)
        predicted = self.observation(current)
        _require(bool(torch.isfinite(current).all()) and bool(torch.isfinite(predicted).all()),
                 'nonfinite forecast state or prediction; no repair')
        return predicted, current

    def step(self, state, inputs):
        dtype = self._validate_parameters()
        self._state(state, dtype)
        self._tensor(inputs, (len(state), self.input_dim), dtype, 'next input')
        return self._step(state, inputs)

    def rollout(self, future_inputs, state):
        dtype = self._validate_parameters()
        self._state(state, dtype)
        _require(isinstance(future_inputs, torch.Tensor) and future_inputs.ndim == 3,
                 'future inputs shape [B,H,input_dim] required')
        self._tensor(future_inputs, (len(state), future_inputs.shape[1], self.input_dim), dtype, 'future inputs')
        predictions, current = [], state
        for t in range(future_inputs.shape[1]):
            predicted, current = self._step(current, future_inputs[:, t])
            predictions.append(predicted)
        if not predictions:
            return future_inputs.new_empty(len(state), 0, self.obs_dim), state.clone()
        return torch.stack(predictions, dim=1), current

    def forward(self, future_inputs, state):
        return self.rollout(future_inputs, state)

    def aux_decode(self, state, future_inputs, horizon):
        """Endpoint head: its own block and only the executed input prefix [:h].

        All supplied inputs must be finite, but later finite inputs do not enter
        the head. Observations or targets are never accepted by this interface.
        """
        dtype = self._validate_parameters()
        self._state(state, dtype)
        _require(type(horizon) is int and horizon in HORIZONS, 'auxiliary horizon must be1,8,32')
        _require(isinstance(future_inputs, torch.Tensor) and future_inputs.ndim == 3
                 and future_inputs.shape[1] >= horizon, 'enough future inputs for auxiliary horizon')
        self._tensor(future_inputs, (len(state), future_inputs.shape[1], self.input_dim), dtype, 'auxiliary inputs')
        block = HORIZONS.index(horizon)
        features = torch.cat((state[:, block*self.block_dim:(block+1)*self.block_dim],
                              future_inputs[:, :horizon].reshape(len(state), horizon*self.input_dim)), dim=-1)
        output = self.auxiliary[str(horizon)](features)
        _require(bool(torch.isfinite(output).all()), 'nonfinite auxiliary prediction; no repair')
        return output

    def parameter_count(self, *, include_aux=True):
        _require(type(include_aux) is bool, 'include_aux must be bool')
        return sum(p.numel() for name, p in self.named_parameters()
                   if include_aux or not name.startswith('auxiliary.'))

    @property
    def state_scalars(self):
        return self.latent_dim

    def model_spec(self):
        dtype = self._validate_parameters()
        total, deployed = self.parameter_count(), self.parameter_count(include_aux=False)
        item_bytes = self.observation.weight.element_size()
        return {'version': VERSION, 'mode': self.mode, 'obs_dim': self.obs_dim, 'input_dim': self.input_dim,
                'latent_dim': self.latent_dim, 'blocks': 3, 'block_dim': self.block_dim,
                'auxiliary_horizons': list(HORIZONS), 'auxiliary_width': self.aux_width,
                'parameter_count': total, 'non_auxiliary_parameter_count': deployed,
                'auxiliary_parameter_count': total-deployed, 'parameter_bytes': total*item_bytes,
                'non_auxiliary_parameter_bytes': deployed*item_bytes, 'buffer_bytes': 0,
                'state_scalars': self.latent_dim, 'state_bytes_per_stream': self.latent_dim*item_bytes,
                'dtype': str(dtype).removeprefix('torch.'), 'initialization_seed': self.initialization_seed,
                'eta': ETA, 'epsilon': EPSILON, 'persistent_cache': False, 'retained_trajectory': False,
                'context_inputs': 'exactly C-1 inputs; input index t-1 pairs with observation index t; physical timing is dataset-specific',
                'selection': 'all blocks' if self.mode == 'dense' else 'largest own gradient block; lowest-index tie'
                             if self.mode == 'selective' else 'cyclic successor of largest block; destination own gradient',
                'routing_limitation': 'Scalar-output and rank-one decoders give a fixed block winner whenever the correction gradient is nonzero; multivariate output alone does not guarantee adaptive routing.',
                'guarantee': 'nonincrease of current fixed-linear-decoder squared residual in exact arithmetic only',
                'limitations': 'No global recurrence stability, predictive calibration, physical-state or novelty guarantee.',
                'storage_scope': 'parameters and caller state; requests, normalization, optimizer and workspace excluded'}
