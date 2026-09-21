"""NumPy inference contract for OTTO's released ``zoo_model_2_3_2``.

The source contract is ValueModel/RLPolicy/SourceTracking at official revision
1467029f399dc5eeac8652499a9c8326ecab4575. Only NumPy is imported: callers supply
separately authenticated tensors and public beliefs, positions and likelihoods.
There is no loader, simulator reference, mutable inference cache or training.

Hidden layers are ReLU, the final output is linear. Upstream nonnegative weight
constraints apply during training, not inference. The policy's 1e-10 floor is
applied to branch masses before normalization, never to model output. NumPy
float32 GEMM/reduction is not a claim of bitwise TensorFlow parity; that requires
separate runtime qualification on the actual tensors.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

VERSION = "otto-pretrained-value-v1"
N = 53
CENTERED = 105
MAX_BATCH = 16
EPSILON = 1e-10
EXPECTED_SHAPES = MappingProxyType({
    "kernel_0": (11025, 1024), "bias_0": (1024,),
    "kernel_1": (1024, 1024), "bias_1": (1024,),
    "kernel_2": (1024, 1024), "bias_2": (1024,),
    "kernel_3": (1024, 1), "bias_3": (1,),
})
KEYS = tuple(EXPECTED_SHAPES)
D4_ORDER = ("identity", "transpose", "reverse_x", "reverse_x_transpose",
            "reverse_xy", "reverse_xy_transpose", "reverse_y", "reverse_y_transpose")


def _batch_limit(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("max_batch must be an integer")
    if not 1 <= value <= MAX_BATCH:
        raise ValueError("max_batch must be in 1..16")
    return int(value)


def _probabilities(value, shape, name):
    if not isinstance(value, np.ndarray) or value.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError(f"{name} must be a float32/float64 ndarray")
    if value.shape != shape or not np.isfinite(value).all() or (value < 0).any():
        raise ValueError(f"{name} has invalid shape or probability values")
    return value


def _positions(value, batch):
    raw = np.asarray(value)
    if raw.shape != (batch, 2) or raw.dtype.kind not in "iu" or (raw < 0).any() or (raw >= N).any():
        raise ValueError("positions must contain two integer coordinates in 0..52 per row")
    # Mixed bool/integer sequences can otherwise coerce booleans to integers.
    if any(isinstance(x, (bool, np.bool_)) for row in value for x in row):
        raise ValueError("positions must not contain booleans")
    return raw.astype(np.int64, copy=False)


def _batch(value, side, max_batch, name):
    bound = _batch_limit(max_batch)
    if not isinstance(value, np.ndarray) or value.ndim != 3 or not 1 <= value.shape[0] <= bound:
        raise ValueError(f"{name} batch must be in 1..{bound}")
    result = _probabilities(value, (value.shape[0], side, side), name)
    # Zero and subnormalized branches are required by upstream RLPolicy.
    if (result.sum(axis=(1, 2), dtype=np.float64) > 1 + 1e-6).any():
        raise ValueError(f"{name} mass exceeds one")
    return result


def center_beliefs(beliefs, positions, *, max_batch=MAX_BATCH):
    """Zero-pad each 53-square grid with its agent at (52,52), without repair.

    An input cell (i,j) maps to (52+i-agent_x,52+j-agent_y). Probability mass is
    not normalized; an impossible successor remains all-zero. The sole numeric
    conversion is upstream's final float32 cast of the centered input.
    """
    source = _batch(beliefs, N, max_batch, "beliefs")
    agents = _positions(positions, len(source))
    result = np.zeros((len(source), CENTERED, CENTERED), dtype=np.float32)
    for index, (x, y) in enumerate(agents):
        result[index, N - 1 - x:2 * N - 1 - x, N - 1 - y:2 * N - 1 - y] = source[index]
    return result


def symmetry_inputs(centered, *, max_batch=MAX_BATCH):
    """Return (8,B,105,105) in the exact upstream transform-major order."""
    source = _batch(centered, CENTERED, max_batch, "centered").astype(np.float32, copy=False)
    pair = np.stack((source, source.transpose(0, 2, 1)), axis=0)
    four = np.concatenate((pair, pair[:, :, ::-1, :]), axis=0)
    return np.concatenate((four, four[:, :, ::-1, ::-1]), axis=0)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class PretrainedValue:
    """Immutable, strict eight-tensor model, with no file loading or chunking."""

    _weights: tuple
    max_batch: int

    def __init__(self, tensors, *, max_batch=MAX_BATCH):
        bound = _batch_limit(max_batch)
        if not isinstance(tensors, Mapping) or set(tensors) != set(KEYS):
            raise ValueError("tensors must contain exactly the eight expected keys")
        weights = []
        for key, shape in EXPECTED_SHAPES.items():
            value = tensors[key]
            if (not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32")
                    or value.shape != shape or not np.isfinite(value).all()):
                raise ValueError(f"invalid tensor {key}: expected finite native float32 {shape}")
            # Immutable bytes backing prevents callers re-enabling WRITEABLE.
            weights.append(np.frombuffer(value.tobytes(order="C"), dtype=np.float32).reshape(shape))
        object.__setattr__(self, "_weights", tuple(weights))
        object.__setattr__(self, "max_batch", bound)

    @property
    def tensors(self):
        return MappingProxyType(dict(zip(KEYS, self._weights, strict=True)))

    def storage_bytes(self):
        """Persistent arrays only; input/output, workspace and Python overhead excluded."""
        arrays = {key: int(value.nbytes) for key, value in zip(KEYS, self._weights, strict=True)}
        return {"immutable_array_bytes": sum(arrays.values()), "arrays": arrays,
                "parameters": sum(value.size for value in self._weights),
                "mutable_array_bytes": 0, "max_batch": self.max_batch,
                "max_symmetry_batch": 8 * self.max_batch,
                "scope": "Weights only; excludes caller arrays, temporary inference workspace and Python overhead."}

    def _forward(self, flat):
        value = flat
        with np.errstate(over="raise", invalid="raise"):
            for layer in range(4):
                value = value @ self._weights[2 * layer] + self._weights[2 * layer + 1]
                if not np.isfinite(value).all():
                    raise FloatingPointError("nonfinite dense output")
                if layer < 3:
                    np.maximum(value, np.float32(0), out=value)
        return value

    def predict_centered(self, centered, *, sym_avg=False):
        """Upstream layer/symmetry order, float32 arithmetic and raw linear output."""
        if not isinstance(sym_avg, bool):
            raise TypeError("sym_avg must be bool")
        source = _batch(centered, CENTERED, self.max_batch, "centered")
        batch = len(source)
        if sym_avg:
            inputs = symmetry_inputs(source, max_batch=self.max_batch).reshape(8 * batch, -1)
        else:
            inputs = np.asarray(source, dtype=np.float32, order="C").reshape(batch, -1)
        result = self._forward(inputs)
        if sym_avg:
            with np.errstate(over="raise", invalid="raise"):
                result = result.reshape(8, batch, 1).mean(axis=0, dtype=np.float32)
        if result.dtype != np.float32 or result.shape != (batch, 1) or not np.isfinite(result).all():
            raise FloatingPointError("invalid value result")
        return result

    def predict_beliefs(self, beliefs, positions, *, sym_avg=False):
        return self.predict_centered(center_beliefs(beliefs, positions, max_batch=self.max_batch),
                                     sym_avg=sym_avg)


def policy_inputs(belief, position, kernel):
    """The 16 upstream successor inputs and floored branch masses, in action/hit order.

    All four actions are evaluated, including blocked moves that stay in place.
    The public kernel is used unchanged, including zero-origin/zero-tail entries.
    No legal-action masking, likelihood floor, branch-weight renormalization or
    extra exclusion is applied. Arithmetic before the final casts is float64.
    """
    if not isinstance(belief, np.ndarray) or belief.dtype != np.dtype("float64"):
        raise ValueError("policy belief must be float64")
    _batch(belief[np.newaxis], N, 1, "belief")
    agent = _positions([position], 1)[0]
    if not isinstance(kernel, np.ndarray) or kernel.dtype != np.dtype("float64"):
        raise ValueError("policy kernel must be float64")
    _probabilities(kernel, (4, 107, 107), "kernel")
    if (kernel > 1).any():
        raise ValueError("invalid categorical kernel")
    branches = np.empty((16, N, N), dtype=np.float64)
    positions = np.empty((16, 2), dtype=np.int64)
    masses = np.empty((4, 4), dtype=np.float32)
    for action in range(4):
        moved = agent.copy()
        axis, direction = action // 2, 2 * (action % 2) - 1
        if 0 <= moved[axis] + direction < N:
            moved[axis] += direction
        x, y = N - moved
        evidence = kernel[:, x:x + N, y:y + N]
        joint = belief[np.newaxis] * evidence
        for hit in range(4):
            probability = np.maximum(EPSILON, np.sum(joint[hit]))
            branches[4 * action + hit] = joint[hit] / probability
            positions[4 * action + hit] = moved
            masses[action, hit] = probability
    return center_beliefs(branches, positions), masses


def value_policy(model, belief, position, kernel, *, sym_avg=True):
    """Return (first upstream near-tie action, float32[4] expected scores).

    This preserves upstream policy semantics, including boundary actions and
    strictly-less-than-1e-10 ties. It does not adapt them to a legal-masked actor.
    """
    inputs, masses = policy_inputs(belief, position, kernel)
    values = model.predict_centered(inputs, sym_avg=sym_avg)
    if (not isinstance(values, np.ndarray) or values.dtype != np.float32 or values.shape != (16, 1)
            or not np.isfinite(values).all()):
        raise ValueError("model must return finite float32[16,1] values")
    with np.errstate(over="raise", invalid="raise"):
        scores = np.float32(1) + np.sum(masses * values.reshape(4, 4), axis=1, dtype=np.float32)
    if not np.isfinite(scores).all():
        raise FloatingPointError("nonfinite expected policy scores")
    choices = np.flatnonzero(np.abs(scores - scores.min()) < EPSILON)
    return int(choices[0]), scores
