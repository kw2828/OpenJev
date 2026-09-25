"""Explicitly loaded hybrid Rust inference for the qualified structured cells.

Each call keeps Torch parameter validation and preparation, including spectral
norms, inside the request. Rust fuses only the sequential gate/transition/forcing
rollout. The original raw model weights are retained, never a prepared cache.
No build, download, library discovery, model creation or training occurs here.
This is inference only; it makes no gradient, pure-native or speed claim.

ABI1 arithmetic is float32 without fast-math. Reduction order and elementary
functions differ from Torch, so parity needs the prospective fixture tolerance.
``request`` takes physical float64 inputs and returns physical float64 forecasts
plus an explicit standardized float32 latent state. ``rollout`` works entirely
in standardized float32 coordinates. Neither accepts future positions/targets.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import torch

from openjev.research.structured_robot_transition import StructuredRobotTransition

VERSION = 'native-robot-transition-v1'
ABI_VERSION = 1
KINDS = ('householder', 'dense_bounded', 'dense_unbounded', 'dense_mlp')
PACKED_SCALARS = dict(zip(KINDS, (638, 806, 806, 590), strict=True))
NORMALIZERS = ('q_mean', 'q_std', 'u_mean', 'u_std')
_F32_POINTER = ctypes.POINTER(ctypes.c_float)
_ERRORS = {1: 'kind, dimension or length', 2: 'pointer, alignment or overlapping buffers',
           3: 'nonfinite numerical value', 4: 'nonpositive scale or reflection denominator'}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _array(value, shape, dtype, name):
    _require(type(value) is np.ndarray and value.dtype == dtype and value.shape == shape
             and np.isfinite(value).all(), f'finite NumPy {dtype} array of exact shape required: {name}')
    return value


class NativeRobotTransition:
    """Hold one existing cell and an explicitly selected ABI1 shared library.

    ``library_path=None`` is deliberately an error, not a search/build fallback.
    Library loading and model construction belong outside request timing in both
    backends. The caller owns the cell; subsequent parameter changes are observed
    on the next request because all fields are prepared and exported again.
    Do not mutate cell parameters concurrently with a request.
    """

    def __init__(self, cell, library_path=None):
        _require(library_path is not None, 'an explicit library_path is required; no automatic build or discovery')
        _require(type(cell) is StructuredRobotTransition, 'exact qualified StructuredRobotTransition required')
        _require(cell.kind in KINDS, 'one of the four structured families required')
        path = Path(library_path).expanduser().resolve(strict=True)
        _require(path.is_file(), 'library_path must be a regular file')
        library = ctypes.CDLL(str(path))
        version = library.rt_abi_version
        version.argtypes, version.restype = [], ctypes.c_uint32
        _require(version() == ABI_VERSION, 'native robot transition ABI version mismatch')
        size = library.rt_parameter_count
        size.argtypes, size.restype = [ctypes.c_uint32], ctypes.c_size_t
        for code, kind in enumerate(KINDS):
            _require(size(code) == PACKED_SCALARS[kind], 'native packed parameter schema mismatch')
        function = library.rt_rollout_v1
        function.argtypes = [ctypes.c_uint32, ctypes.c_size_t, ctypes.c_size_t,
                             _F32_POINTER, ctypes.c_size_t, _F32_POINTER, ctypes.c_size_t,
                             _F32_POINTER, ctypes.c_size_t, _F32_POINTER, ctypes.c_size_t,
                             _F32_POINTER, ctypes.c_size_t]
        function.restype = ctypes.c_int32
        self.cell = cell
        self.library_path = str(path)
        self._library = library
        self._function = function

    def storage(self):
        """Actual retained numeric payloads, excluding temporary request arrays."""
        values = list(self.cell.parameters())
        buffers = list(self.cell.buffers())
        gradients = [value.grad for value in values if value.grad is not None]
        return {'parameter_count': sum(p.numel() for p in values),
                'parameter_bytes': sum(p.numel() * p.element_size() for p in values),
                'buffer_bytes': sum(p.numel() * p.element_size() for p in buffers),
                'retained_gradient_bytes': sum(p.numel() * p.element_size() for p in gradients),
                'state_bytes_per_stream': 48, 'normalizer_bytes': 192,
                'retained_prepared_bytes': 0,
                'scope': 'raw Torch model weights, buffers and gradients; explicit caller state and normalizers listed separately; shared library code, Python objects, request arrays and temporary workspace excluded'}

    def _pack(self):
        dtype = self.cell._validate_parameters()
        _require(dtype == torch.float32, 'native rollout requires CPU float32 model parameters')
        _require(all(p.grad is None for p in self.cell.parameters()), 'inference requires cleared parameter gradients')
        prepared = self.cell._prepare()
        pieces = [prepared.scale, self.cell.input_matrix, self.cell.expert_bias]
        if self.cell.kind == 'householder':
            pieces += [prepared.vectors, prepared.decay]
        else:
            pieces += [prepared.matrix]
        gate = self.cell.gate
        if self.cell.kind == 'dense_mlp':
            pieces += [gate[0].weight, gate[0].bias, gate[2].weight, gate[2].bias]
        else:
            pieces += [gate.input_weight, gate.reset_update_bias, gate.candidate_input_bias,
                       gate.candidate_hidden_bias, gate.head_weight, gate.head_bias]
        packed = np.empty(PACKED_SCALARS[self.cell.kind], dtype=np.float32)
        offset = 0
        for value in pieces:
            # Explicit owned export, including possible noncontiguous weights.
            owned = np.array(value.detach().numpy(), dtype=np.float32, order='C', copy=True).reshape(-1)
            packed[offset:offset + owned.size] = owned
            offset += owned.size
        _require(offset == len(packed) and np.isfinite(packed).all(), 'finite prepared parameter export required')
        _require(np.all(packed[:12] > 0), 'strictly positive prepared scale required')
        prepared_bytes = sum(value.numel() * value.element_size() for value in prepared if value is not None)
        return packed, prepared_bytes

    @torch.no_grad()
    def rollout(self, future_u, state):
        """Return owned prediction[B,H,6], final_state[B,12] and work metadata.

        H=0 returns an empty forecast and an owned copy of the unchanged state.
        Noncontiguous inputs are accepted and copied. All validations, qualified
        preparation, exports, FFI and output finite checks occur within this call.
        The final state can be passed to a later rollout without observations.
        """
        _require(type(future_u) is np.ndarray and future_u.ndim == 3 and future_u.shape[0] > 0
                 and future_u.shape[2] == 6, 'future torque shape [positive B,H,6] required')
        batch, horizon = future_u.shape[:2]
        _array(future_u, (batch, horizon, 6), np.dtype('float32'), 'future torques')
        _array(state, (batch, 12), np.dtype('float32'), 'explicit latent state')
        packed, prepared_bytes = self._pack()
        future = np.array(future_u, dtype=np.float32, order='C', copy=True)
        initial = np.array(state, dtype=np.float32, order='C', copy=True)
        prediction = np.empty((batch, horizon, 6), dtype=np.float32)
        final = np.empty((batch, 12), dtype=np.float32)
        code = self._function(KINDS.index(self.cell.kind), batch, horizon,
                              packed.ctypes.data_as(_F32_POINTER), packed.size,
                              future.ctypes.data_as(_F32_POINTER), future.size,
                              initial.ctypes.data_as(_F32_POINTER), initial.size,
                              prediction.ctypes.data_as(_F32_POINTER), prediction.size,
                              final.ctypes.data_as(_F32_POINTER), final.size)
        if code != 0:
            raise ValueError('native robot transition failed: ' + _ERRORS.get(code, f'unknown status {code}'))
        _require(np.isfinite(prediction).all() and np.isfinite(final).all(), 'nonfinite native output; no repair')
        work = {'parameter_validations': 1, 'parameter_preparations': 1, 'ffi_calls': 1,
                'batch': batch, 'horizon': horizon, 'steps': batch * horizon,
                'prepared_tensor_bytes': prepared_bytes, 'packed_parameter_bytes': packed.nbytes,
                'parameter_export_piece_bytes': packed.nbytes,
                'input_copy_bytes': future.nbytes + initial.nbytes,
                'output_buffer_bytes': prediction.nbytes + final.nbytes,
                'normalized_input_bytes': 0, 'cast_input_bytes': 0,
                'condition_state_bytes': 0, 'physical_output_bytes': 0,
                'retained_prepared_bytes': 0,
                'scope': 'array payloads and explicit operations within this call; export pieces are cumulative, not simultaneous peak; Torch/native/Python allocator workspace and intermediate arithmetic arrays are not measured'}
        return {'prediction': prediction, 'final_state': final, 'work': work}

    @torch.no_grad()
    def request(self, q_context, u_context, future_u, normalizers):
        """Complete conditional request from physical float64 input arrays.

        Future measured torques are conditioned on; no future positions are
        accepted. The forecast is physical float64, while the returned state is
        standardized float32. Historical torque/context validation remains the
        qualified cell.condition path even though initialization uses only the
        last two observed position samples. Parameter validation therefore occurs
        twice, matching condition followed by forward in the Torch comparator.
        """
        _require(type(q_context) is np.ndarray and q_context.ndim == 3 and q_context.shape[0] > 0
                 and q_context.shape[1] >= 2 and q_context.shape[2] == 6, 'context shape [positive B,C>=2,6] required')
        batch = q_context.shape[0]
        _array(q_context, q_context.shape, np.dtype('float64'), 'physical observed positions')
        _array(u_context, q_context.shape, np.dtype('float64'), 'physical observed torques')
        _require(type(future_u) is np.ndarray and future_u.ndim == 3
                 and future_u.shape[0] == batch and future_u.shape[2] == 6, 'physical future torque shape [B,H,6] required')
        _array(future_u, future_u.shape, np.dtype('float64'), 'physical future torques')
        _require(type(normalizers) is dict and set(normalizers) == set(NORMALIZERS), 'exact four normalizer arrays required')
        for key in NORMALIZERS:
            _array(normalizers[key], (6,), np.dtype('float64'), key)
        _require(np.all(normalizers['q_std'] > 0) and np.all(normalizers['u_std'] > 0), 'positive normalization scales required')
        _require(self.cell.input_matrix.dtype == torch.float32, 'native request requires CPU float32 model parameters')
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            q = (q_context - normalizers['q_mean']) / normalizers['q_std']
            u = (u_context - normalizers['u_mean']) / normalizers['u_std']
            future = (future_u - normalizers['u_mean']) / normalizers['u_std']
            q32, u32, future32 = (np.asarray(value, dtype=np.float32) for value in (q, u, future))
        _require(all(np.isfinite(value).all() for value in (q32, u32, future32)), 'finite normalized float32 request required')
        state = self.cell.condition(torch.from_numpy(q32), torch.from_numpy(u32)).detach().numpy()
        result = self.rollout(future32, state)
        with np.errstate(over='ignore', invalid='ignore'):
            physical = result['prediction'].astype(np.float64) * normalizers['q_std'] + normalizers['q_mean']
        _require(np.isfinite(physical).all(), 'nonfinite physical output; no repair')
        result['prediction'] = physical
        result['work'].update(parameter_validations=2,
                              normalized_input_bytes=q.nbytes + u.nbytes + future.nbytes,
                              cast_input_bytes=q32.nbytes + u32.nbytes + future32.nbytes,
                              condition_state_bytes=state.nbytes, physical_output_bytes=physical.nbytes)
        return result
