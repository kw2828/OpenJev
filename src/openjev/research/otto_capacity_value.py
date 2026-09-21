"""Ordinary scalar capacity controls, with identical public features and baseline.

Fresh CPU training uses float32; deployed NumPy uses immutable float64 upcasts.
No model, checkpoint, dataset, optimizer or simulator is loaded by this module.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt

import numpy as np

from openjev.research.otto_return_value import (
    INPUT_DIM,
    MAX_BATCH,
    SCALE,
    SPATIAL_DIM,
    value_features,
)

VERSION = "otto-capacity-value-v1"
KINDS = ("mlp8", "mlp128", "deep128")
HIDDEN = {"mlp8": (8,), "mlp128": (128,), "deep128": (128, 128, 128)}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _scalar(value):
    if isinstance(value, np.ndarray):
        _require(value.shape == () and value.dtype.kind in "USiu", "scalar metadata required")
        return value.item()
    return value


def _dimensions(kind):
    _require(isinstance(kind, str) and kind in KINDS, "unknown capacity family")
    return (INPUT_DIM, *HIDDEN[kind], 1)


def _shapes(kind):
    dims = _dimensions(kind)
    return {name: shape for i, (left, right) in enumerate(zip(dims[:-1], dims[1:], strict=True))
            for name, shape in ((f"weight_{i}", (right, left)), (f"bias_{i}", (right,)))}


def parameter_count(kind):
    return sum(int(np.prod(shape)) for shape in _shapes(kind).values())


def _sensing(value):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, float, np.integer, np.floating)), "numeric sensing length required")
    value = float(value)
    _require(np.isfinite(value) and value > 0, "positive finite sensing length required")
    return value


def make_head(kind, seed, c0):
    """Create a fresh CPU float32 scalar head using only one local RNG.

Hidden weights are normal with std sqrt(2/fan_in); output weights use
sqrt(1/fan_in). Biases are zero. The first layer is drawn in eight-row blocks
to guarantee identical first eight rows for every family and identical first
128 rows for the two large families, without relying on shape-dependent RNG
prefix behavior. Later draws differ because layer shapes and fan-ins differ.
The fixed float32 c0 buffer multiplies raw spatial mass exactly once.
"""
    import torch

    dims = _dimensions(kind)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and 0 <= int(seed) < 2**63, "bounded nonboolean integer seed required")
    _require(not isinstance(c0, (bool, np.bool_)) and isinstance(c0, (int, float, np.integer, np.floating)),
             "numeric baseline required")
    with np.errstate(over="raise", invalid="raise"):
        baseline = np.float32(c0)
    _require(np.isfinite(baseline), "finite float32 baseline required")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    arrays = {}
    for i, (left, right) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        if i == 0:
            weight = torch.cat([torch.randn((8, left), generator=generator, dtype=torch.float32, device="cpu")
                                for _ in range(right // 8)], dim=0)
        else:
            weight = torch.randn((right, left), generator=generator, dtype=torch.float32, device="cpu")
        arrays[f"weight_{i}"] = weight * sqrt((1 if i == len(dims)-2 else 2) / left)
        arrays[f"bias_{i}"] = torch.zeros(right, dtype=torch.float32, device="cpu")

    class CapacityHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            for name, value in arrays.items():
                self.register_parameter(name, torch.nn.Parameter(value))
            self.register_buffer("c0", torch.tensor(float(baseline), dtype=torch.float32, device="cpu"))

        def forward(self, x):
            _require(isinstance(x, torch.Tensor) and x.ndim == 2 and x.shape[1] == INPUT_DIM
                     and 0 < len(x) <= MAX_BATCH and x.device.type == "cpu"
                     and x.dtype == self.weight_0.dtype and bool(torch.isfinite(x).all()),
                     "bounded finite CPU features matching model dtype required")
            hidden = x
            for i in range(len(dims)-1):
                hidden = hidden @ getattr(self, f"weight_{i}").T + getattr(self, f"bias_{i}")
                if i < len(dims)-2:
                    hidden = torch.relu(hidden)
            result = self.c0 * x[:, :SPATIAL_DIM].sum(dim=1) + hidden[:, 0]
            _require(bool(torch.isfinite(result).all()), "nonfinite normalized value")
            return result

    return CapacityHead()


def validate_head(head):
    """Require exact family metadata and finite native float32 checkpoint arrays."""
    _require(isinstance(head, Mapping), "checkpoint mapping required")
    _require({"version", "kind", "input_dim", "c0"} <= set(head), "missing checkpoint fields")
    kind, dim = _scalar(head["kind"]), _scalar(head["input_dim"])
    shapes = {"c0": (), **_shapes(kind)}
    _require(_scalar(head["version"]) == VERSION and not isinstance(dim, (bool, np.bool_))
             and isinstance(dim, (int, np.integer)) and int(dim) == INPUT_DIM, "checkpoint version or input dimension")
    _require(set(head) == {"version", "kind", "input_dim", *shapes}, "exact checkpoint membership required")
    for name, shape in shapes.items():
        value = head[name]
        _require(isinstance(value, np.ndarray) and value.dtype == np.float32 and value.shape == shape
                 and np.isfinite(value).all(), f"invalid float32 array {name}")
    return kind


def export_head(model):
    """Return independent float32 arrays with no optimizer or dataset fields."""
    import torch

    kind = getattr(model, "kind", None)
    shapes = _shapes(kind)
    parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    _require(set(parameters) == set(shapes) and set(buffers) == {"c0"}, "exact learned tensors and baseline required")
    result = {"version": VERSION, "kind": kind, "input_dim": INPUT_DIM}
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
    """Validate once and own only immutable float64 arrays; no retained f32 copy.

normalized accepts saved float32 features upcast to float64, or independently
constructed float64 features. The caller binds which view is used. The branch
callback uses unchanged value_features and returns physical values, 64 times
normalized values. Kernel authenticity belongs to the external branch builder.
Neither zero inputs nor signed predictions receive a repair or clamp.
"""

    kind: str
    sensing_length: float | None
    c0: np.ndarray
    weights: tuple
    biases: tuple

    def __init__(self, head, sensing_length=None):
        kind = validate_head(head)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "sensing_length", None if sensing_length is None else _sensing(sensing_length))
        object.__setattr__(self, "c0", _immutable(head["c0"]))
        layers = len(_dimensions(kind))-1
        object.__setattr__(self, "weights", tuple(_immutable(head[f"weight_{i}"]) for i in range(layers)))
        object.__setattr__(self, "biases", tuple(_immutable(head[f"bias_{i}"]) for i in range(layers)))

    def normalized(self, features):
        _require(isinstance(features, np.ndarray) and features.dtype == np.float64 and features.ndim == 2
                 and features.shape[1] == INPUT_DIM and 0 < len(features) <= MAX_BATCH
                 and np.isfinite(features).all(), "bounded finite float64 features required")
        with np.errstate(over="raise", invalid="raise"):
            hidden = features
            for i, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
                hidden = hidden @ weight.T + bias
                if i < len(self.weights)-1:
                    hidden = np.maximum(hidden, 0)
            result = self.c0 * features[:, :SPATIAL_DIM].sum(axis=1, dtype=np.float64) + hidden[:, 0]
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
                "scope": "Owned immutable float64 weights, biases and c0; no retained float32 copy, inputs or history. Transient workspace and scalar metadata excluded."}
