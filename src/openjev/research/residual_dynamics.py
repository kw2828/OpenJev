"""Functional residual dynamics and inference-only recursive least squares.

These are conventional GRU prediction and ridge/RLS adaptation components, not
an exact Bayesian filter or a novelty claim. No environment, data, or training
loop is owned here. The caller alone admits actual transition labels to RLS.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

NORMALIZATION_EPS = 1e-8


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _dimension(value, name):
    _require(type(value) is int and value > 0, name + ' must be a positive integer')


def _scale(value, size):
    _require(isinstance(value, Tensor) and value.shape == (size,)
             and value.dtype in (torch.float32, torch.float64)
             and bool(torch.isfinite(value).all()) and bool((value > 0).all()),
             'delta_scale must be a positive finite float32/64 vector')
    return value.detach().clone()


def _tensor(value, shape, reference, name):
    _require(isinstance(value, Tensor) and tuple(value.shape) == tuple(shape)
             and value.dtype == reference.dtype and value.device == reference.device
             and bool(torch.isfinite(value).all()), 'Invalid ' + name)


def _feature(value):
    return torch.cat((F.normalize(value, dim=-1, eps=NORMALIZATION_EPS),
                      value.new_ones(len(value), 1)), dim=-1)


class ResidualDynamics(nn.Module):
    """Action-conditioned GRU with an exact, initially persistent output skip.

    Start with initial -> assimilate(o0). Each actual action uses advance, then
    assimilate(o1) on that returned prior. The prior's previous_obs is the last
    root observation, so actual deltas never subtract the predicted endpoint.
    Private forecasts call advance repeatedly without assimilating predictions.
    The caller owns real/imagined admission and any temporal boundary masks.

    Leading translation coordinates are excluded from both GRU inputs and
    public RLS features; their deltas and output accumulator remain available.
    This gives translation invariance, at the cost of excluding explicit
    position-dependent terrain effects. Features use per-row L2 normalization
    followed by an unnormalized intercept. This is not coordinate whitening.
    """

    def __init__(self, obs_dim=9, action_dim=40, hidden_size=64, *,
                 delta_scale: Tensor, translation_dims=3):
        super().__init__()
        for name, value in (('obs_dim', obs_dim), ('action_dim', action_dim),
                            ('hidden_size', hidden_size)):
            _dimension(value, name)
        _require(type(translation_dims) is int and 0 <= translation_dims <= obs_dim,
                 'translation_dims must be an integer in [0, obs_dim]')
        self.obs_dim, self.action_dim, self.hidden_size = obs_dim, action_dim, hidden_size
        self.translation_dims = translation_dims
        self.register_buffer('delta_scale', _scale(delta_scale, obs_dim))
        options = {'dtype': self.delta_scale.dtype, 'device': self.delta_scale.device}
        self.observation_update = nn.GRUCell(2 * obs_dim - translation_dims, hidden_size, **options)
        self.transition = nn.GRUCell(action_dim + 2 * obs_dim - translation_dims, hidden_size, **options)
        self.readout = nn.Linear(hidden_size, obs_dim, **options)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def configuration(self):
        return {'version': 'residual-dynamics-v1', 'model_class': type(self).__name__,
                'obs_dim': self.obs_dim, 'action_dim': self.action_dim,
                'hidden_size': self.hidden_size, 'translation_dims': self.translation_dims,
                'delta_scale': self.delta_scale.detach().cpu().tolist(),
                'output': 'current observation plus delta_scale times linear prior hidden',
                'readout_initialization': 'zero weight and bias; exact persistence',
                'feature_normalization': 'L2 eps=1e-8 then intercept1',
                'real_delta_reference': 'previous_obs saved before the preceding advance',
                'state_ownership': 'functional caller-owned tensors',
                'forecast_information': 'state and supplied action only'}

    def initial(self, batch: int, device=None):
        _dimension(batch, 'batch')
        _require(self.delta_scale.dtype in (torch.float32, torch.float64), 'Float32/64 model required')
        _require(device is None or torch.device(device) == self.delta_scale.device, 'Device mismatch')
        return {**{name: self.delta_scale.new_zeros(batch, self.obs_dim)
                   for name in ('obs', 'previous_obs', 'delta')},
                'hidden': self.delta_scale.new_zeros(batch, self.hidden_size),
                'has_obs': torch.zeros(batch, 1, dtype=torch.bool, device=self.delta_scale.device)}

    def _state(self, state):
        _require(type(state) is dict and set(state) == {'obs', 'previous_obs', 'delta', 'hidden', 'has_obs'},
                 'Invalid residual state keys')
        _require(isinstance(state['obs'], Tensor) and state['obs'].ndim == 2
                 and len(state['obs']) > 0, 'Invalid observation state')
        batch = len(state['obs'])
        for name in ('obs', 'previous_obs', 'delta', 'hidden'):
            _tensor(state[name], (batch, self.hidden_size if name == 'hidden' else self.obs_dim),
                    self.delta_scale, name)
        flag = state['has_obs']
        _require(isinstance(flag, Tensor) and flag.shape == (batch, 1)
                 and flag.dtype == torch.bool and flag.device == self.delta_scale.device,
                 'Invalid has_obs mask')
        return batch

    def assimilate(self, state, obs: Tensor):
        batch = self._state(state)
        _tensor(obs, (batch, self.obs_dim), self.delta_scale, 'observation')
        previous = torch.where(state['has_obs'], state['previous_obs'], obs)
        delta = (obs - previous) / self.delta_scale
        hidden = self.observation_update(torch.cat((obs[:, self.translation_dims:], delta), -1),
                                         state['hidden'])
        result = {'obs': obs.clone(), 'previous_obs': previous.clone(), 'delta': delta,
                  'hidden': hidden, 'has_obs': torch.ones_like(state['has_obs'])}
        self._state(result)
        return result

    def advance(self, state, action: Tensor):
        batch = self._state(state)
        _require(bool(state['has_obs'].all()), 'Assimilate an initial observation before advance')
        _tensor(action, (batch, self.action_dim), self.delta_scale, 'action')
        public = torch.cat((state['obs'][:, self.translation_dims:], state['delta'], action), -1)
        transition_input = torch.cat((action, state['obs'][:, self.translation_dims:], state['delta']), -1)
        hidden = self.transition(transition_input, state['hidden'])
        delta = self.readout(hidden)
        prediction = state['obs'] + self.delta_scale * delta
        result = {'obs': prediction.clone(), 'previous_obs': state['obs'].clone(),
                  'delta': delta, 'hidden': hidden, 'has_obs': state['has_obs'].clone()}
        features = {'public': _feature(public), 'latent': _feature(hidden),
                    'bias': prediction.new_ones(batch, 1)}
        self._state(result)
        _require(all(bool(torch.isfinite(value).all()) for value in features.values()), 'Nonfinite features')
        return result, prediction, features

    def parameter_and_state_counts(self):
        size = self.delta_scale.element_size()
        count = sum(parameter.numel() for parameter in self.parameters())
        state_floats = 3 * self.obs_dim + self.hidden_size
        return {'active_parameters': count, 'registered_parameters': count,
                'parameter_bytes': count * size, 'buffer_bytes': self.delta_scale.numel() * size,
                'state_floats_per_case': state_floats, 'state_bools_per_case': 1,
                'state_bytes_per_case': state_floats * size + 1,
                'feature_dims': {'public': 2 * self.obs_dim - self.translation_dims + self.action_dim + 1,
                                 'latent': self.hidden_size + 1, 'bias': 1},
                'scope': 'tensor payload; excludes activations, temporary copies, optimizer and Python objects'}


class RLSAdapter(nn.Module):
    """Episode-local multioutput ridge regression, lambda=1 and forgetting=1.

    observe takes residual targets in delta_scale units, not absolute endpoints:
    (actual_next_obs - unadapted_predicted_obs) / delta_scale. It must be called
    only on real context transitions. correct returns an observation-unit
    correction to add to the unadapted prediction. Neither method updates slow
    model weights; both detach their inputs. No update happens in correct.
    Covariance is an inverse normal matrix, not calibrated predictive variance.
    """

    def __init__(self, feature_dim: int, obs_dim: int, delta_scale: Tensor):
        super().__init__()
        _dimension(feature_dim, 'feature_dim')
        _dimension(obs_dim, 'obs_dim')
        self.feature_dim, self.obs_dim = feature_dim, obs_dim
        self.register_buffer('delta_scale', _scale(delta_scale, obs_dim))

    def configuration(self):
        return {'version': 'residual-rls-v1', 'feature_dim': self.feature_dim, 'obs_dim': self.obs_dim,
                'regularization': 1.0, 'forgetting': 1.0,
                'delta_scale': self.delta_scale.detach().cpu().tolist(),
                'target': 'actual minus unadapted prediction, divided by delta_scale',
                'correction_units': 'observation units', 'updates': 'caller-supplied real transitions only',
                'gradients': 'none; slow weights remain fixed'}

    @torch.no_grad()
    def initial(self, batch: int, device=None):
        _dimension(batch, 'batch')
        _require(self.delta_scale.dtype in (torch.float32, torch.float64), 'Float32/64 adapter required')
        _require(device is None or torch.device(device) == self.delta_scale.device, 'Device mismatch')
        return {'weights': self.delta_scale.new_zeros(batch, self.feature_dim, self.obs_dim),
                'covariance': torch.eye(self.feature_dim, dtype=self.delta_scale.dtype,
                                        device=self.delta_scale.device).expand(batch, -1, -1).clone(),
                'updates': torch.zeros(batch, 1, dtype=torch.int64, device=self.delta_scale.device)}

    def _state(self, state):
        _require(type(state) is dict and set(state) == {'weights', 'covariance', 'updates'}, 'Invalid RLS state keys')
        weights = state['weights']
        _require(isinstance(weights, Tensor) and weights.ndim == 3 and len(weights) > 0, 'Invalid RLS weights')
        batch = len(weights)
        _tensor(weights, (batch, self.feature_dim, self.obs_dim), self.delta_scale, 'RLS weights')
        covariance = state['covariance']
        _tensor(covariance, (batch, self.feature_dim, self.feature_dim), self.delta_scale, 'RLS covariance')
        _require(bool((covariance.diagonal(dim1=-2, dim2=-1) > 0).all())
                 and torch.allclose(covariance, covariance.transpose(-1, -2), rtol=1e-5, atol=1e-7),
                 'RLS covariance must be symmetric with positive diagonal')
        updates = state['updates']
        _require(isinstance(updates, Tensor) and updates.shape == (batch, 1)
                 and updates.dtype == torch.int64 and updates.device == self.delta_scale.device
                 and bool((updates >= 0).all()), 'Invalid RLS update counts')
        return batch

    @torch.no_grad()
    def observe(self, state, phi: Tensor, target_delta_residual: Tensor):
        batch = self._state(state)
        _tensor(phi, (batch, self.feature_dim), self.delta_scale, 'RLS feature')
        _tensor(target_delta_residual, (batch, self.obs_dim), self.delta_scale, 'RLS residual target')
        phi, target_delta_residual = phi.detach(), target_delta_residual.detach()
        covariance = state['covariance'].detach()
        projected = torch.bmm(covariance, phi.unsqueeze(-1)).squeeze(-1)
        denominator = 1 + (phi * projected).sum(-1, keepdim=True)
        _require(bool(torch.isfinite(denominator).all()) and bool((denominator > 0).all()),
                 'Invalid RLS gain denominator')
        gain = projected / denominator
        error = target_delta_residual - torch.bmm(phi.unsqueeze(1), state['weights'].detach()).squeeze(1)
        updated_covariance = covariance - gain.unsqueeze(-1) * projected.unsqueeze(1)
        updated_covariance = (updated_covariance + updated_covariance.transpose(-1, -2)) * .5
        result = {'weights': state['weights'].detach() + gain.unsqueeze(-1) * error.unsqueeze(1),
                  'covariance': updated_covariance, 'updates': state['updates'] + 1}
        self._state(result)
        return result

    @torch.no_grad()
    def correct(self, state, phi: Tensor):
        batch = self._state(state)
        _tensor(phi, (batch, self.feature_dim), self.delta_scale, 'RLS feature')
        result = torch.bmm(phi.detach().unsqueeze(1), state['weights'].detach()).squeeze(1) * self.delta_scale
        _require(bool(torch.isfinite(result).all()), 'Nonfinite RLS correction')
        return result

    def parameter_and_state_counts(self):
        count = self.feature_dim * self.obs_dim + self.feature_dim ** 2
        return {'active_parameters': 0, 'state_floats_per_case': count, 'state_int64_per_case': 1,
                'state_bytes_per_case': count * self.delta_scale.element_size() + 8,
                'scope': 'tensor payload only; full analytic covariance, no hidden optimizer'}
