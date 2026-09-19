"""Small functional action-conditioned filters, without environment or training IO.

The matrix arms test full row transport against scalar retention. They are
experimental delta-memory variants, not novel filtering algorithms or biological
models. The diagonal control is Kalman-like in a learned coordinate system, not
an RKN reproduction or a claim of calibrated uncertainty. All slow parameters
are fixed during inference; explicit episode state changes through pure methods.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MODES = ('transported_delta', 'decay_delta', 'gru', 'diagonal_filter')
KEY_SIZE, VALUE_SIZE, HIDDEN_SIZE = 16, 4, 64
NORMALIZATION_EPS = 1e-8
VARIANCE_FLOOR = 1e-4
TRANSITION_BOUND = .99


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class ActionFilterModel(nn.Module):
    """One common interface for four genuinely different update rules.

    ``assimilate`` consumes a real observation without advancing time.
    ``advance`` consumes only an action and returns the next prior plus its
    observation prediction. Startup is ``initial`` then ``assimilate(o0)``.
    The harness owns masks, issued-action alignment and terminal boundaries.
    Forecast branches must not call ``assimilate`` on future measurements.

    A delta matrix stores 64 floats. GRU stores 64; the diagonal filter stores
    64 means plus 64 variance scales. Parameter and compute budgets differ.
    Both key and value encodings use epsilon-stabilized L2 normalization.
    """

    def __init__(self, mode: str, obs_dim: int, action_dim: int, hidden_size: int = 64,
                 *, readout_hidden: bool = False):
        super().__init__()
        _require(mode in MODES, 'Unknown action-filter mode')
        _require(type(obs_dim) is int and obs_dim > 0 and type(action_dim) is int and action_dim > 0,
                 'Positive integer observation/action dimensions required')
        _require(type(hidden_size) is int and hidden_size == HIDDEN_SIZE,
                 'This compact comparison fixes hidden_size=64 and matrix shape16x4')
        _require(type(readout_hidden) is bool, 'readout_hidden must be bool')
        self.mode, self.obs_dim, self.action_dim = mode, obs_dim, action_dim
        self.hidden_size, self.readout_hidden = hidden_size, readout_hidden
        # Same decoder names/shapes in every arm. Pairing belongs to the caller,
        # which may copy matching named tensors before any optimization.
        self.readout = (nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ELU(),
                                      nn.Linear(hidden_size, obs_dim))
                        if readout_hidden else nn.Linear(hidden_size, obs_dim))
        if mode in ('transported_delta', 'decay_delta'):
            self.key = nn.Linear(obs_dim, KEY_SIZE)
            self.value = nn.Linear(obs_dim, VALUE_SIZE)
            self.write_gate = nn.Linear(obs_dim, 1)
            self.retention = nn.Linear(action_dim, 1)
            self.action_drive = nn.Linear(action_dim, hidden_size, bias=False)
            if mode == 'transported_delta':
                # Identity plus two full row-stochastic operators. Convex row
                # mixing cannot amplify the entrywise maximum absolute state.
                self.row_logits = nn.Parameter(torch.zeros(2, KEY_SIZE, KEY_SIZE))
                self.mixture = nn.Linear(action_dim, 3)
        elif mode == 'gru':
            self.observation_update = nn.GRUCell(obs_dim, hidden_size)
            self.transition = nn.GRUCell(action_dim, hidden_size)
        else:
            self.observation_mean = nn.Linear(obs_dim, hidden_size)
            self.observation_noise = nn.Linear(obs_dim, hidden_size)
            self.transition_scale = nn.Linear(action_dim, hidden_size)
            self.transition_drive = nn.Linear(action_dim, hidden_size)
            self.process_noise = nn.Linear(action_dim, hidden_size)

    def configuration(self):
        return {'version': 'action-filter-models-v1', 'model_class': type(self).__name__,
            'mode': self.mode, 'obs_dim': self.obs_dim, 'action_dim': self.action_dim,
            'hidden_size': self.hidden_size, 'readout_hidden': self.readout_hidden,
            'matrix_shape': [KEY_SIZE, VALUE_SIZE] if 'delta' in self.mode else None,
            'key_and_value_normalization': 'L2 with eps=1e-8' if 'delta' in self.mode else None,
            'transport': 'convex action-dependent mixture of identity and two full row-stochastic16x16 maps'
                         if self.mode == 'transported_delta' else None,
            'diagonal_variance_floor': VARIANCE_FLOOR if self.mode == 'diagonal_filter' else None,
            'diagonal_transition_bound': TRANSITION_BOUND if self.mode == 'diagonal_filter' else None,
            'state_ownership': 'functional caller-owned tensors; no mutable module memory',
            'forecast_information': 'current state and supplied action only',
            'uncertainty_scope': 'learned positive latent scales; no calibration or exact RKN claim'}

    def initial(self, batch: int, device=None):
        _require(type(batch) is int and batch > 0, 'Positive integer batch required')
        parameter = next(self.parameters())
        _require(parameter.dtype in (torch.float32, torch.float64), 'Float32/float64 model required')
        _require(device is None or torch.device(device) == parameter.device, 'State/model device mismatch')
        if 'delta' in self.mode:
            return {'memory': parameter.new_zeros(batch, KEY_SIZE, VALUE_SIZE)}
        if self.mode == 'gru':
            return {'hidden': parameter.new_zeros(batch, self.hidden_size)}
        return {'mean': parameter.new_zeros(batch, self.hidden_size),
                'variance': parameter.new_ones(batch, self.hidden_size)}

    def _state(self, state):
        _require(type(state) is dict, 'Explicit dictionary state required')
        shapes = ({'memory': (KEY_SIZE, VALUE_SIZE)} if 'delta' in self.mode else
                  {'hidden': (self.hidden_size,)} if self.mode == 'gru' else
                  {'mean': (self.hidden_size,), 'variance': (self.hidden_size,)})
        _require(set(state) == set(shapes), 'State keys differ from actual model class/mode')
        parameter = next(self.parameters())
        batch = None
        for name, shape in shapes.items():
            value = state[name]
            _require(isinstance(value, Tensor) and value.ndim == len(shape) + 1
                     and tuple(value.shape[1:]) == shape and len(value) > 0
                     and value.dtype == parameter.dtype and value.device == parameter.device
                     and value.dtype in (torch.float32, torch.float64) and bool(torch.isfinite(value).all()),
                     'Invalid state tensor: ' + name)
            _require(batch is None or batch == len(value), 'State batch mismatch')
            batch = len(value)
        if self.mode == 'diagonal_filter':
            _require(bool((state['variance'] > 0).all()), 'Strictly positive variance required')
        return batch

    def _input(self, value, batch, size, name):
        parameter = next(self.parameters())
        _require(isinstance(value, Tensor) and value.shape == (batch, size)
                 and value.dtype == parameter.dtype and value.device == parameter.device
                 and bool(torch.isfinite(value).all()), 'Invalid ' + name)

    def assimilate(self, state, obs: Tensor):
        """Write one actual public observation; no action/time transition."""
        batch = self._state(state)
        self._input(obs, batch, self.obs_dim, 'observation')
        if 'delta' in self.mode:
            key = F.normalize(self.key(obs), dim=-1, eps=NORMALIZATION_EPS)
            value = F.normalize(self.value(obs), dim=-1, eps=NORMALIZATION_EPS)
            prior = state['memory']
            residual = value - torch.einsum('bkv,bk->bv', prior, key)
            strength = self.write_gate(obs).sigmoid()
            result = {'memory': prior + strength[:, :, None] * key[:, :, None] * residual[:, None, :]}
        elif self.mode == 'gru':
            result = {'hidden': self.observation_update(obs, state['hidden'])}
        else:
            encoded = self.observation_mean(obs)
            noise = F.softplus(self.observation_noise(obs)) + VARIANCE_FLOOR
            prior = state['variance']
            gain = prior / (prior + noise)
            result = {'mean': state['mean'] + gain * (encoded - state['mean']),
                      # Equivalent posterior p*r/(p+r), avoiding 1-g cancellation.
                      'variance': noise * gain}
        self._state(result)
        return result

    def advance(self, state, action: Tensor):
        """Action-only prediction. No learned observation is fed back as evidence."""
        batch = self._state(state)
        self._input(action, batch, self.action_dim, 'action')
        if 'delta' in self.mode:
            prior = state['memory']
            if self.mode == 'transported_delta':
                rows = self.row_logits.softmax(dim=-1)
                moved = torch.einsum('ijk,bkv->bijv', rows, prior)
                weights = self.mixture(action).softmax(dim=-1)
                prior = weights[:, :1, None] * prior + torch.einsum('bi,bijv->bjv', weights[:, 1:], moved)
            retention = self.retention(action).sigmoid()[:, :, None]
            drive = self.action_drive(action).tanh().reshape(batch, KEY_SIZE, VALUE_SIZE)
            result = {'memory': retention * prior + (1 - retention) * drive}
            features = result['memory'].flatten(1)
        elif self.mode == 'gru':
            result = {'hidden': self.transition(action, state['hidden'])}
            features = result['hidden']
        else:
            scale = TRANSITION_BOUND * self.transition_scale(action).tanh()
            process = F.softplus(self.process_noise(action)) + VARIANCE_FLOOR
            result = {'mean': scale * state['mean'] + self.transition_drive(action),
                      'variance': scale.square() * state['variance'] + process}
            features = result['mean']
        prediction = self.readout(features)
        self._state(result)
        _require(bool(torch.isfinite(prediction).all()), 'Nonfinite predicted observation')
        return result, prediction

    def parameter_and_state_counts(self):
        """Actual parameter/state storage, not matched total computation or peak RSS."""
        item_bytes = next(self.parameters()).element_size()
        parameters = sum(p.numel() for p in self.parameters())
        state = self.hidden_size * (2 if self.mode == 'diagonal_filter' else 1)
        return {'registered_parameters': parameters, 'active_parameters': parameters,
            'parameter_bytes': parameters * item_bytes, 'state_scalars_per_case': state,
            'state_bytes_per_case': state * item_bytes,
            'state_scope': 'persistent tensor payload only; excludes copies, activations, optimizer and Python objects',
            'transport_matrix_parameters': self.row_logits.numel() if self.mode == 'transported_delta' else 0,
            'transport_mixture_parameters': sum(p.numel() for p in self.mixture.parameters())
                if self.mode == 'transported_delta' else 0}
