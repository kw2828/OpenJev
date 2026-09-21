"""Fixed-coordinate conditioning of the ordinary width-eight scalar model.

Both arms keep raw public features, the raw-mass baseline and model capacity.
Gain53 changes optimization coordinates, not information or architecture. Its
inverse initialization compensation is approximate after float32 rounding.
"""
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from openjev.research.otto_return_value import (
    INPUT_DIM,
    MAX_BATCH,
    SCALE,
    SPATIAL_DIM,
    value_features,
)

VERSION = "otto-conditioned-value-v1"
KINDS = ("gain1", "gain53")
GAINS = {"gain1": 1, "gain53": 53}
SHAPES = {"weight_0": (8, INPUT_DIM), "bias_0": (8,), "weight_1": (1, 8), "bias_1": (1,)}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _kind(kind):
    _require(isinstance(kind, str) and kind in KINDS, "unknown conditioning arm")
    return kind


def _scalar(value):
    if isinstance(value, np.ndarray):
        _require(value.shape == () and value.dtype.kind in "USiu", "scalar metadata required")
        return value.item()
    return value


def _integer(value, expected, message):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
             and int(value) == expected, message)


def _sensing(value):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.integer, np.floating))
             and np.isfinite(value) and value > 0, "positive finite sensing length required")
    return float(value)


def parameter_count(kind):
    _kind(kind)
    return 88241


def make_head(kind, seed, c0):
    """Clone exactly the fresh capacity mlp8 initialization, then compensate.

Only the gain53 spatial first-layer columns are divided by53, in float32.
Context columns, both biases, final weights and c0 remain byte-identical.
Raw input is never modified. The c0 skip always sums its unscaled mass.
"""
    import torch

    from openjev.research.otto_capacity_value import make_head as capacity_head

    kind = _kind(kind)
    template = capacity_head("mlp8", seed, c0)
    arrays = {name: parameter.detach().clone() for name, parameter in template.named_parameters()}
    baseline = template.c0.detach().clone()
    if GAINS[kind] != 1:
        arrays["weight_0"][:, :SPATIAL_DIM].div_(GAINS[kind])

    class ConditionedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kind, self.spatial_gain = kind, GAINS[kind]
            for name, array in arrays.items():
                self.register_parameter(name, torch.nn.Parameter(array))
            self.register_buffer("c0", baseline)

        def forward(self, x):
            _require(isinstance(x, torch.Tensor) and x.ndim == 2 and x.shape[1] == INPUT_DIM
                     and 0 < len(x) <= MAX_BATCH and x.device.type == "cpu"
                     and x.dtype == self.weight_0.dtype and bool(torch.isfinite(x).all()),
                     "bounded finite CPU features matching model dtype required")
            hidden = x
            if self.spatial_gain != 1:
                hidden = x.clone()
                hidden[:, :SPATIAL_DIM] *= self.spatial_gain
            hidden = torch.relu(hidden @ self.weight_0.T + self.bias_0)
            residual = (hidden @ self.weight_1.T + self.bias_1)[:, 0]
            result = self.c0 * x[:, :SPATIAL_DIM].sum(dim=1) + residual
            _require(bool(torch.isfinite(result).all()), "nonfinite normalized value")
            return result

    return ConditionedHead()


def validate_head(head):
    """Require exact fixed metadata and native finite float32 tensor shapes."""
    _require(isinstance(head, Mapping), "checkpoint mapping required")
    fields = {"version", "kind", "input_dim", "spatial_gain", "c0", *SHAPES}
    _require(set(head) == fields, "exact conditioning checkpoint membership required")
    kind = _kind(_scalar(head["kind"]))
    _require(_scalar(head["version"]) == VERSION, "checkpoint version")
    _integer(_scalar(head["input_dim"]), INPUT_DIM, "checkpoint input dimension")
    _integer(_scalar(head["spatial_gain"]), GAINS[kind], "fixed gain must agree with arm")
    for name, shape in {"c0": (), **SHAPES}.items():
        array = head[name]
        _require(isinstance(array, np.ndarray) and array.dtype == np.float32 and array.shape == shape
                 and np.isfinite(array).all(), f"invalid finite float32 array {name}")
    return kind


def export_head(model):
    """Export owned float32 arrays, excluding optimizer state and datasets."""
    import torch

    kind = _kind(getattr(model, "kind", None))
    _integer(getattr(model, "spatial_gain", None), GAINS[kind], "model gain disagrees with arm")
    parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    _require(set(parameters) == set(SHAPES) and set(buffers) == {"c0"}, "exact learned tensors and baseline required")
    result = {"version": VERSION, "kind": kind, "input_dim": INPUT_DIM, "spatial_gain": GAINS[kind]}
    for name, tensor in {**parameters, **buffers}.items():
        _require(tensor.device.type == "cpu" and tensor.dtype == torch.float32, "CPU float32 export required")
        result[name] = tensor.detach().numpy().copy()
    validate_head(result)
    return result


def _immutable(array):
    converted = array.astype(np.float64)
    return np.frombuffer(converted.tobytes(order="C"), dtype=np.float64).reshape(converted.shape)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class FrozenValue:
    """Validate once; retain only immutable float64 upcasts of saved parameters.

The residual scales spatial inputs explicitly before the first matmul, matching
the double-Torch reference's ordering. No folded-weight shortcut or standardizer
is used. Baseline/context remain raw, zero/low-mass inputs and signed outputs are
unchanged. The external branch builder authenticates the otherwise unused kernel.
"""

    kind: str
    spatial_gain: int
    sensing_length: float | None
    c0: np.ndarray
    weights: tuple
    biases: tuple

    def __init__(self, head, sensing_length=None):
        kind = validate_head(head)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "spatial_gain", GAINS[kind])
        object.__setattr__(self, "sensing_length", None if sensing_length is None else _sensing(sensing_length))
        object.__setattr__(self, "c0", _immutable(head["c0"]))
        object.__setattr__(self, "weights", tuple(_immutable(head[f"weight_{i}"]) for i in range(2)))
        object.__setattr__(self, "biases", tuple(_immutable(head[f"bias_{i}"]) for i in range(2)))

    def normalized(self, features):
        _require(isinstance(features, np.ndarray) and features.dtype == np.float64 and features.ndim == 2
                 and features.shape[1] == INPUT_DIM and 0 < len(features) <= MAX_BATCH
                 and np.isfinite(features).all(), "bounded finite float64 features required")
        with np.errstate(over="raise", invalid="raise"):
            hidden = features
            if self.spatial_gain != 1:
                hidden = features.copy()
                hidden[:, :SPATIAL_DIM] *= self.spatial_gain
            hidden = np.maximum(hidden @ self.weights[0].T + self.biases[0], 0)
            residual = (hidden @ self.weights[1].T + self.biases[1])[:, 0]
            result = self.c0 * features[:, :SPATIAL_DIM].sum(axis=1, dtype=np.float64) + residual
        _require(result.shape == (len(features),) and np.isfinite(result).all(), "nonfinite normalized value")
        return result

    def __call__(self, centered, positions, kernel):
        _require(self.sensing_length is not None, "branch callback needs bound sensing length")
        with np.errstate(over="raise", invalid="raise"):
            result = SCALE * self.normalized(value_features(centered, positions, self.sensing_length))
        _require(np.isfinite(result).all(), "nonfinite physical value")
        return result

    def storage_bytes(self):
        return {"parameter_array_bytes": sum(a.nbytes for a in (*self.weights, *self.biases)),
                "baseline_array_bytes": self.c0.nbytes, "mutable_array_bytes": 0,
                "scope": "Owned immutable float64 weights, biases and c0; no retained float32 copy, inputs or history. Explicit scaling workspace and scalar gain metadata excluded."}
