"""Explicit-library native inference for the qualified 32-hidden residual GRU.

This interprets the supplied 5,916-parameter Torch roster using the frozen GRU32
equations. It does not call a supplied module's forward or condition methods.
Every request validates and exports all parameters afresh. There is no prepared
weight cache, model construction, build, download, training or library discovery.

Rust performs context conditioning and the entire sequential rollout in float32.
Python physical-input normalization is float64 before the float32 cast; physical
predictions are float64. Explicit final states stay standardized float32, ordered
q[6], previous_q[6], previous_u[6], hidden[32]. No future positions or targets are
accepted. Numerical agreement with Torch is a qualified tolerance claim, never
bitwise equivalence or an unmeasured speed claim.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import torch
from torch import nn

VERSION = 'native-robot-gru-v1'
ABI_VERSION = 1
PARAMETER_COUNT = 5916
STATE_SCALARS = 50
PARAMETERS = (
    ('base_weight', (6, 25)),
    ('gru.weight_ih', (96, 24)),
    ('gru.weight_hh', (96, 32)),
    ('gru.bias_ih', (96,)),
    ('gru.bias_hh', (96,)),
    ('head.weight', (6, 32)),
    ('head.bias', (6,)),
)
NORMALIZERS = ('q_mean', 'q_std', 'u_mean', 'u_std')
_POINTER = ctypes.POINTER(ctypes.c_float)
_PAIR = [_POINTER, ctypes.c_size_t]
_ERRORS = {1: 'dimension or length', 2: 'pointer, alignment or overlapping buffers',
           3: 'nonfinite numerical value'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, shape, dtype, name):
    _require(type(value) is np.ndarray and value.dtype == np.dtype(dtype)
             and value.shape == shape and np.isfinite(value).all(),
             f'finite NumPy {dtype} array with exact shape required: {name}')
    return value


def _future(value, dtype, batch=None):
    _require(type(value) is np.ndarray and value.ndim == 3 and value.shape[0] > 0
             and value.shape[2] == 6 and (batch is None or value.shape[0] == batch),
             'future torques must have shape [positive B,H,6]')
    return _array(value, value.shape, dtype, 'future torques')


def _arguments(*arrays):
    return [part for value in arrays for part in (value.ctypes.data_as(_POINTER), value.size)]


class NativeRobotGRU:
    """Retain existing GRU32 weights and an explicitly selected ABI1 library.

    Required modules are exact GRUCell(24,32) and Linear(32,6), with the seven
    named arrays above and no buffers or additional submodules/parameters. This
    is a parameter-schema adapter, not a guarantee about arbitrary Python forward
    overrides. The native equations are the frozen residual GRU32 definition.
    Subsequent parameter changes are visible on the next call; concurrent model
    mutation during a request is unsupported. Loading belongs outside request
    timing in both backends and must be reported separately.
    """

    def __init__(self, model, library_path=None):
        _require(library_path is not None, 'explicit library_path required; no search or build fallback')
        self.model = model
        self._validate_parameters()
        path = Path(library_path).expanduser().resolve(strict=True)
        _require(path.is_file(), 'library_path must be a regular file')
        library = ctypes.CDLL(str(path))
        version = library.gr_abi_version
        version.argtypes, version.restype = [], ctypes.c_uint32
        _require(version() == ABI_VERSION, 'native GRU ABI version mismatch')
        count = library.gr_parameter_count
        count.argtypes, count.restype = [], ctypes.c_size_t
        _require(count() == PARAMETER_COUNT, 'native GRU parameter schema mismatch')
        request = library.gr_request_v1
        request.argtypes = [ctypes.c_size_t] * 3 + _PAIR * 6
        request.restype = ctypes.c_int32
        rollout = library.gr_rollout_v1
        rollout.argtypes = [ctypes.c_size_t] * 2 + _PAIR * 5
        rollout.restype = ctypes.c_int32
        self.library_path = str(path)
        self._library = library
        self._request = request
        self._rollout = rollout

    def _validate_parameters(self):
        model = self.model
        _require(isinstance(model, nn.Module), 'Torch module with qualified GRU32 parameter roster required')
        children = dict(model.named_children())
        _require(set(children) == {'gru', 'head'} and type(children['gru']) is nn.GRUCell
                 and type(children['head']) is nn.Linear, 'exact GRUCell and Linear submodules required')
        _require(model.gru.input_size == 24 and model.gru.hidden_size == 32 and model.gru.bias
                 and model.head.in_features == 32 and model.head.out_features == 6,
                 'qualified GRU32 architecture dimensions required')
        parameters = dict(model.named_parameters())
        _require(set(parameters) == {name for name, _ in PARAMETERS} and not list(model.buffers()),
                 'exact seven parameter arrays and no buffers required')
        for name, shape in PARAMETERS:
            value = parameters[name]
            _require(value.dtype == torch.float32 and value.device.type == 'cpu' and tuple(value.shape) == shape
                     and torch.isfinite(value).all().item() and value.grad is None,
                     f'finite CPU float32 parameter with cleared gradient required: {name}')
        return parameters

    def _pack(self):
        parameters = self._validate_parameters()
        packed = np.empty(PARAMETER_COUNT, dtype=np.float32)
        offset = 0
        for name, _ in PARAMETERS:
            # Row-major logical export observes values even for Fortran-layout
            # coefficients. The copy/packing work belongs to every request.
            owned = np.array(parameters[name].detach().numpy(), dtype=np.float32, order='C', copy=True)
            packed[offset:offset + owned.size] = owned.reshape(-1)
            offset += owned.size
        _require(offset == PARAMETER_COUNT and np.isfinite(packed).all(), 'finite complete GRU parameter pack')
        return packed

    def storage(self):
        """Retained numeric payload only; allocation/workspace is not a peak estimate."""
        parameters = list(self.model.parameters())
        buffers = list(self.model.buffers())
        return {
            'parameter_count': sum(value.numel() for value in parameters),
            'parameter_bytes': sum(value.numel() * value.element_size() for value in parameters),
            'buffer_bytes': sum(value.numel() * value.element_size() for value in buffers),
            'retained_gradient_bytes': sum(value.grad.numel() * value.grad.element_size()
                                           for value in parameters if value.grad is not None),
            'state_bytes_per_stream': 200, 'normalizer_bytes': 192, 'retained_prepared_bytes': 0,
            'scope': 'raw Torch parameters/buffers/gradients; caller state and normalizers listed separately; '
                     'library code, Python objects and temporary request arrays/workspace excluded',
        }

    @staticmethod
    def _finish(code, prediction, final):
        if code != 0:
            raise ValueError('native GRU failed: ' + _ERRORS.get(code, f'unknown status {code}'))
        _require(np.isfinite(prediction).all() and np.isfinite(final).all(), 'finite native GRU outputs required')

    @staticmethod
    def _work(batch, horizon, context, inputs, prediction, final):
        return {
            'parameter_validations': 1, 'parameter_preparations': 0, 'ffi_calls': 1,
            'batch': batch, 'context': context, 'horizon': horizon,
            'context_gru_steps': batch * max(context - 2, 0), 'rollout_gru_steps': batch * horizon,
            'packed_parameter_bytes': PARAMETER_COUNT * 4, 'parameter_export_piece_bytes': PARAMETER_COUNT * 4,
            'input_copy_bytes': sum(value.nbytes for value in inputs),
            'output_buffer_bytes': prediction.nbytes + final.nbytes,
            'normalized_input_bytes': 0, 'cast_input_bytes': 0, 'physical_output_bytes': 0,
            'retained_prepared_bytes': 0,
            'scope': 'actual explicit array payloads; export pieces are cumulative, not simultaneous peak; '
                     'native stack, allocator/Python workspace and arithmetic temporaries not measured',
        }

    @torch.no_grad()
    def rollout(self, future_u, state):
        """Continue standardized float32 inputs/state, repacking every parameter.

        Returns owned prediction[B,H,6], final_state[B,50] and explicit work.
        H=0 is an empty forecast and owned unchanged state. No mutation or cache.
        """
        _future(future_u, 'float32')
        batch, horizon = future_u.shape[:2]
        _array(state, (batch, STATE_SCALARS), 'float32', 'standardized GRU state')
        parameters = self._pack()
        future = np.array(future_u, dtype=np.float32, order='C', copy=True)
        initial = np.array(state, dtype=np.float32, order='C', copy=True)
        prediction = np.empty((batch, horizon, 6), dtype=np.float32)
        final = np.empty((batch, STATE_SCALARS), dtype=np.float32)
        code = self._rollout(batch, horizon, *_arguments(parameters, future, initial, prediction, final))
        self._finish(code, prediction, final)
        return {'prediction': prediction, 'final_state': final,
                'work': self._work(batch, horizon, 0, (future, initial), prediction, final)}

    @torch.no_grad()
    def request(self, q_context, u_context, future_u, normalizers):
        """Full physical request, including observed-prefix GRU conditioning.

        Context indices1..C-2 update the hidden state. The state starts from q[C-1],
        q[C-2], u[C-2]. Future input0 is measured torque u[C-1]. Supplied
        u_context[C-1] is shape/finite validated but never used in conditioning.
        """
        _require(type(q_context) is np.ndarray and q_context.ndim == 3 and q_context.shape[0] > 0
                 and q_context.shape[1] >= 2 and q_context.shape[2] == 6, 'context shape [positive B,C>=2,6]')
        batch, context = q_context.shape[:2]
        _array(q_context, q_context.shape, 'float64', 'physical observed positions')
        _array(u_context, q_context.shape, 'float64', 'physical observed torques')
        _future(future_u, 'float64', batch)
        horizon = future_u.shape[1]
        _require(type(normalizers) is dict and set(normalizers) == set(NORMALIZERS), 'exact normalizer roster')
        for key in NORMALIZERS:
            _array(normalizers[key], (6,), 'float64', key)
        _require(np.all(normalizers['q_std'] > 0) and np.all(normalizers['u_std'] > 0), 'positive scales required')
        parameters = self._pack()
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            normalized = ((q_context - normalizers['q_mean']) / normalizers['q_std'],
                          (u_context - normalizers['u_mean']) / normalizers['u_std'],
                          (future_u - normalizers['u_mean']) / normalizers['u_std'])
            q, u, future = (np.array(value, dtype=np.float32, order='C', copy=True) for value in normalized)
        _require(all(np.isfinite(value).all() for value in (q, u, future)), 'finite normalized float32 inputs')
        prediction = np.empty((batch, horizon, 6), dtype=np.float32)
        final = np.empty((batch, STATE_SCALARS), dtype=np.float32)
        code = self._request(batch, context, horizon,
                             *_arguments(parameters, q, u, future, prediction, final))
        self._finish(code, prediction, final)
        with np.errstate(over='ignore', invalid='ignore'):
            physical = prediction.astype(np.float64) * normalizers['q_std'] + normalizers['q_mean']
        _require(np.isfinite(physical).all(), 'finite physical GRU predictions required; no repair')
        work = self._work(batch, horizon, context, (q, u, future), prediction, final)
        work.update(normalized_input_bytes=sum(value.nbytes for value in normalized),
                    cast_input_bytes=q.nbytes + u.nbytes + future.nbytes, physical_output_bytes=physical.nbytes)
        return {'prediction': physical, 'final_state': final, 'work': work}
