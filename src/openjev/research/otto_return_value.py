"""Small scalar return readouts over public, agent-centered belief mass.

Torch is imported only by the training factory/exporter. Deployment uses
float64 NumPy arithmetic on an immutable upcast of stored float32 parameters.
Values are signed; neither zero-input nor terminal-value repairs are applied.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

VERSION = "otto-return-value-v1"
KINDS = ("min8", "mlp8", "homogeneous8")
GRID = 53
CENTERED = 105
SPATIAL_DIM = CENTERED**2
INPUT_DIM = SPATIAL_DIM + 3
WIDTH = 8
SCALE = 64
MAX_BATCH = 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             f"{name} must be a nonboolean integer")
    return int(value)


def _sensing(value):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, float, np.integer, np.floating)), "sensing length must be numeric")
    value = float(value)
    _require(np.isfinite(value) and value > 0, "sensing length must be finite and positive")
    return value


def _scalar(value):
    if isinstance(value, np.ndarray):
        _require(value.shape == () and value.dtype.kind in "USiu", "invalid scalar metadata")
        return value.item()
    return value


def _immutable(array):
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def value_features(centered, positions, sensing_length, *, dtype="float64"):
    """Return bounded Bx11028 features without changing any supplied mass.

The input is float64[B,105,105], with integer positions[B,2] in 0..52.
Flattened raw belief is followed by mass*x/52, mass*y/52 and mass*lambda/5.
The float32 training view is a final cast of this same float64 construction.
Zero and subnormalized branches are accepted, including all-zero branches.
"""
    _require(dtype in ("float64", "float32"), "dtype must be float64 or float32")
    _require(isinstance(centered, np.ndarray) and centered.dtype == np.float64 and centered.ndim == 3
             and centered.shape[1:] == (CENTERED, CENTERED) and 0 < centered.shape[0] <= MAX_BATCH,
             "bounded float64[B,105,105] centered beliefs required")
    _require(np.isfinite(centered).all() and (centered >= 0).all(), "invalid centered belief entries")
    positions = np.asarray(positions)
    _require(positions.shape == (centered.shape[0], 2) and positions.dtype.kind in "iu"
             and (positions >= 0).all() and (positions < GRID).all(), "integer positions[B,2] in 0..52 required")
    sensing_length = _sensing(sensing_length)
    flat = centered.reshape(centered.shape[0], SPATIAL_DIM)
    with np.errstate(over="raise", invalid="raise"):
        mass = flat.sum(axis=1, dtype=np.float64)
        _require((mass <= 1 + 1e-6).all(), "belief mass exceeds public contract")
        result = np.empty((centered.shape[0], INPUT_DIM), dtype=np.float64)
        result[:, :SPATIAL_DIM] = flat
        result[:, SPATIAL_DIM:SPATIAL_DIM + 2] = mass[:, None] * positions / (GRID - 1)
        result[:, -1] = mass * sensing_length / 5
        result = result.astype(dtype, copy=False)
    _require(np.isfinite(result).all(), "nonfinite features")
    return result


def parameter_count(kind):
    _require(kind in KINDS, "unknown value family")
    return {"min8": 88224, "mlp8": 88241, "homogeneous8": 88232}[kind]


def make_head(kind, seed, c0):
    """CPU head with paired local initialization; model(x) is normalized[B].

One fresh CPU Generator draws the common 8xD first weight, then the common
8-vector output weight for either MLP. All draws are float32 randn * .01;
biases start at zero. The shared c0 mass baseline is a buffer, not a parameter.
Calling .double() on a copy supplies the double-precision parity reference.
"""
    import torch

    _require(kind in KINDS, "unknown value family")
    seed = _integer(seed, "seed")
    _require(0 <= seed < 2**63, "seed outside supported range")
    _require(not isinstance(c0, (bool, np.bool_))
             and isinstance(c0, (int, float, np.integer, np.floating)), "c0 must be a numeric scalar")
    with np.errstate(over="raise", invalid="raise"):
        baseline = np.float32(c0)
    _require(np.isfinite(baseline), "c0 must be finite float32")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = torch.randn((WIDTH, INPUT_DIM), generator=generator, dtype=torch.float32, device="cpu") * 0.01
    final = (torch.randn((WIDTH,), generator=generator, dtype=torch.float32, device="cpu") * 0.01
             if kind != "min8" else None)

    class ReturnHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.first_weight = torch.nn.Parameter(first)
            self.register_buffer("c0", torch.tensor(float(baseline), dtype=torch.float32, device="cpu"))
            if kind != "min8":
                self.final_weight = torch.nn.Parameter(final)
            if kind == "mlp8":
                self.hidden_bias = torch.nn.Parameter(torch.zeros(WIDTH, dtype=torch.float32, device="cpu"))
                self.output_bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device="cpu"))

        def forward(self, x):
            _require(x.ndim == 2 and x.shape[1] == INPUT_DIM and 0 < x.shape[0] <= MAX_BATCH
                     and x.device.type == "cpu" and x.dtype == self.first_weight.dtype
                     and torch.isfinite(x).all().item(), "bounded finite CPU features matching model dtype required")
            first_values = x @ self.first_weight.T
            if self.kind == "min8":
                residual = first_values.min(dim=1).values
            else:
                if self.kind == "mlp8":
                    first_values = first_values + self.hidden_bias
                residual = torch.relu(first_values) @ self.final_weight
                if self.kind == "mlp8":
                    residual = residual + self.output_bias
            result = self.c0 * x[:, :SPATIAL_DIM].sum(dim=1) + residual
            _require(torch.isfinite(result).all().item(), "nonfinite normalized values")
            return result

    return ReturnHead()


def _parameter_shapes(kind):
    shapes = {"first_weight": (WIDTH, INPUT_DIM)}
    if kind != "min8":
        shapes["final_weight"] = (WIDTH,)
    if kind == "mlp8":
        shapes.update(hidden_bias=(WIDTH,), output_bias=())
    return shapes


def validate_head(head):
    """Check a mapping, including an allow_pickle=False NPZ materialization.

Metadata strings/integers may be Python scalars or zero-dimensional NumPy
arrays. The c0 buffer and every weight must be finite float32 ndarrays with
exact shapes. Extra keys, biases on homogeneous families and dtype repairs
are rejected. No file is opened here.
"""
    _require(isinstance(head, Mapping), "head export must be a mapping")
    _require({"version", "kind", "input_dim", "c0"} <= set(head), "missing value metadata")
    kind = _scalar(head["kind"])
    _require(isinstance(kind, str) and kind in KINDS, "unknown value family")
    _require(_scalar(head["version"]) == VERSION
             and _integer(_scalar(head["input_dim"]), "input dimension") == INPUT_DIM,
             "unsupported value export metadata")
    shapes = {"c0": (), **_parameter_shapes(kind)}
    _require(set(head) == {"version", "kind", "input_dim", *shapes}, "exact family export keys required")
    for name, shape in shapes.items():
        array = head[name]
        _require(isinstance(array, np.ndarray) and array.dtype == np.float32 and array.shape == shape
                 and np.isfinite(array).all(), f"invalid float32 value array {name}")
    return kind


def export_head(model):
    """Export the learned float32 parameters and exact float32 c0 buffer."""
    import torch

    kind = getattr(model, "kind", None)
    _require(kind in KINDS, "known value head required")
    shapes = _parameter_shapes(kind)
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    _require(set(parameters) == set(shapes) and set(buffers) == {"c0"}, "exact value tensors required")
    result = {"version": VERSION, "kind": kind, "input_dim": INPUT_DIM}
    for name, tensor in {**parameters, **buffers}.items():
        _require(tensor.device.type == "cpu" and tensor.dtype == torch.float32,
                 "only CPU float32 checkpoint tensors may be exported")
        result[name] = tensor.detach().numpy().copy()
    validate_head(result)
    return result


@dataclass(frozen=True, slots=True, init=False, eq=False)
class FrozenValue:
    """Immutable f64 deployed readout of f32 weights, no persistent input cache.

    normalized(features) returns signed normalized values and needs no bound
    sensing length. The callable branch interface requires one and returns
    physical search costs (64 times normalized values). It does not inspect kernel:
the branch builder and its caller authenticate that public sensor separately.
There is no input, probability, output or zero-branch repair.
"""

    kind: str
    sensing_length: float | None
    c0: np.ndarray
    first_weight: np.ndarray
    final_weight: np.ndarray | None
    hidden_bias: np.ndarray | None
    output_bias: np.ndarray | None

    def __init__(self, head, sensing_length=None):
        kind = validate_head(head)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "sensing_length", None if sensing_length is None else _sensing(sensing_length))
        for name in ("c0", "first_weight", "final_weight", "hidden_bias", "output_bias"):
            array = _immutable(head[name].astype(np.float64)) if name in head else None
            object.__setattr__(self, name, array)

    @staticmethod
    def _features(x):
        _require(isinstance(x, np.ndarray) and x.dtype == np.float64 and x.ndim == 2
                 and x.shape[1] == INPUT_DIM and 0 < x.shape[0] <= MAX_BATCH and np.isfinite(x).all(),
                 "bounded finite float64[B,11028] features required")
        return x

    def normalized(self, features):
        x = self._features(features)
        with np.errstate(over="raise", invalid="raise"):
            first = x @ self.first_weight.T
            if self.kind == "min8":
                residual = first.min(axis=1)
            else:
                if self.hidden_bias is not None:
                    first = first + self.hidden_bias
                residual = np.maximum(first, 0) @ self.final_weight
                if self.output_bias is not None:
                    residual = residual + self.output_bias
            result = self.c0 * x[:, :SPATIAL_DIM].sum(axis=1, dtype=np.float64) + residual
        _require(result.shape == (x.shape[0],) and np.isfinite(result).all(), "nonfinite normalized values")
        return result

    def __call__(self, centered_z, successors, kernel):
        _require(self.sensing_length is not None, "branch callback requires a bound sensing length")
        features = value_features(centered_z, successors, self.sensing_length)
        with np.errstate(over="raise", invalid="raise"):
            values = SCALE * self.normalized(features)
        _require(np.isfinite(values).all(), "nonfinite physical values")
        return values

    def plane_indices(self, features):
        """Diagnostic first argmin per row, available only for min8."""
        _require(self.kind == "min8", "plane usage is defined only for min8")
        x = self._features(features)
        with np.errstate(over="raise", invalid="raise"):
            values = x @ self.first_weight.T
        _require(np.isfinite(values).all(), "nonfinite plane values")
        return values.argmin(axis=1)

    def storage_bytes(self):
        arrays = (self.first_weight, self.final_weight, self.hidden_bias, self.output_bias)
        return {"parameter_array_bytes": sum(a.nbytes for a in arrays if a is not None),
                "baseline_array_bytes": self.c0.nbytes, "mutable_array_bytes": 0,
                "scope": "Owned immutable float64 parameter upcasts and c0; excludes scalar metadata, caller inputs and transient batch workspace. No stored float32 copy or history."}
