"""Five public-belief scalar readouts from otto-spatial-design-review.md.

Public geometry helpers accept native float64 NumPy arrays, int64 positions,
and batches 1..128. Torch heads accept CPU float32 (or .double() float64)
centered[B,105,105], integer positions[B,2], and same-dtype sensing[B]. The
canonical mass sums the original centered array; physical unpacking is lossless.
No extra normalization, clipping, terminal override or homogeneity is imposed.

Checkpoints have exactly version/kind/input_dim plus c0 (float32 scalar) and
family-specific finite float32 arrays returned by parameter_shapes(). Weight
names are {conv,cell,readout}_weight_i and corresponding _bias_i, with standard
(out,in) or (out,in,3,3) orientation. Torch is imported only by make/export.
FrozenValue independently computes NumPy float64 inference from owned immutable
upcasts. Reduction-order rounding requires separate deployed-action parity
qualification; this module does not claim bitwise Torch/NumPy equality.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType

import numpy as np

VERSION = "otto-spatial-value-v1"
KINDS = ("spatial", "neighbor_free", "cnn", "dense128", "statistics")
GRID, CENTERED, INPUT_DIM, MAX_BATCH, SCALE = 53, 105, 11028, 128, 64
EPSILON = 1e-10


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _kind(kind):
    _require(isinstance(kind, str) and kind in KINDS, "unknown spatial-value kind")
    return kind


def _integer(value):
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))


def _scalar(value):
    if isinstance(value, np.ndarray):
        _require(value.shape == () and value.dtype.kind in "USiu", "scalar checkpoint metadata required")
        return value.item()
    return value


def _sensing_scalar(value):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, float, np.integer, np.floating))
             and np.isfinite(value) and value > 0, "positive finite sensing length required")
    return float(value)


def _array(value, shape, name, dtype=np.float64):
    _require(isinstance(value, np.ndarray) and value.dtype == dtype and value.shape == shape
             and np.isfinite(value).all(), f"finite native {np.dtype(dtype).name} {name}{shape} required")


def _batch(value, tail, name):
    _require(isinstance(value, np.ndarray) and value.ndim == len(tail) + 1
             and 0 < len(value) <= MAX_BATCH, f"bounded batch for {name} required")
    _array(value, (len(value), *tail), name)
    _require((value >= 0).all(), f"nonnegative {name} required")
    return len(value)


def _positions(positions, batch):
    _array(positions, (batch, 2), "positions", np.int64)
    _require(((positions >= 0) & (positions < GRID)).all(), "positions must lie in 0..52")


def unpack_centered(centered, positions):
    """Return an owned physical field and original centered row-major mass.

Every outside-rectangle entry must be exactly zero, not merely low mass.
No positive mass, tiny branch or negative zero is repaired or renormalized.
"""
    batch = _batch(centered, (CENTERED, CENTERED), "centered")
    _positions(positions, batch)
    physical = np.empty((batch, GRID, GRID), dtype=np.float64)
    for i, (x, y) in enumerate(positions):
        a, b = GRID - 1 - int(x), GRID - 1 - int(y)
        source = centered[i]
        _require(not (np.any(source[:a]) or np.any(source[a + GRID:])
                      or np.any(source[a:a + GRID, :b]) or np.any(source[a:a + GRID, b + GRID:])),
                 "nonzero mass outside physical board")
        physical[i] = source[a:a + GRID, b:b + GRID]
    with np.errstate(over="raise", invalid="raise"):
        mass = centered.reshape(batch, -1).sum(axis=1, dtype=np.float64)
    _require((mass <= 1 + 1e-6).all(), "belief mass exceeds public contract")
    return physical, mass


def _slices(offset):
    return slice(max(0, -offset), min(GRID, GRID - offset)), slice(max(0, offset), min(GRID, GRID + offset))


def _box_sum(physical, k):
    # Separable additions avoid cancellation from prefix-sum subtraction.
    horizontal, result = np.zeros_like(physical), np.zeros_like(physical)
    radius = k // 2
    for offset in range(-radius, radius + 1):
        target, source = _slices(offset)
        horizontal[:, :, target] += physical[:, :, source]
    for offset in range(-radius, radius + 1):
        target, source = _slices(offset)
        result[:, target, :] += horizontal[:, source, :]
    return result


def box_sum(physical, k):
    """Unnormalized zero-extended square sums on the actual 53-square board."""
    _batch(physical, (GRID, GRID), "physical")
    _require(_integer(k) and k in (3, 9, 27), "box width must be 3, 9 or 27")
    with np.errstate(over="raise", invalid="raise"):
        result = _box_sum(physical, int(k))
    _require(np.isfinite(result).all(), "nonfinite box sum")
    return result


def _feature_inputs(physical, mass, positions, sensing):
    batch = _batch(physical, (GRID, GRID), "physical")
    _array(mass, (batch,), "mass")
    _positions(positions, batch)
    _array(sensing, (batch,), "sensing")
    _require((sensing > 0).all(), "positive sensing lengths required")
    _require(((mass >= 0) & (mass <= 1 + 1e-6)).all(), "invalid raw branch mass")
    total = physical.reshape(batch, -1).sum(axis=1, dtype=np.float64)
    _require(np.all(np.abs(total - mass) <= 32 * np.finfo(np.float64).eps),
             "physical mass disagrees with canonical centered mass")
    return batch


def _geometry(mass, positions, sensing):
    batch = len(mass)
    x, y = np.indices((GRID, GRID), dtype=np.float64)
    result = np.empty((batch, GRID, GRID, 8), dtype=np.float64)
    result[..., 0] = (x - positions[:, 0, None, None]) / 52
    result[..., 1] = (y - positions[:, 1, None, None]) / 52
    result[..., 2], result[..., 3] = x / 52, (52 - x) / 52
    result[..., 4], result[..., 5] = y / 52, (52 - y) / 52
    result[..., 6], result[..., 7] = sensing[:, None, None] / 5, mass[:, None, None]
    return result


def spatial_features(physical, mass, positions, sensing, neighbor_free=False):
    """Return Bx53x53x12 in the exact channel order stated in the design."""
    batch = _feature_inputs(physical, mass, positions, sensing)
    _require(isinstance(neighbor_free, bool), "neighbor_free must be bool")
    result = np.empty((batch, GRID, GRID, 12), dtype=np.float64)
    result[..., 0] = 53 * physical
    for index, k in enumerate((3, 9, 27), 1):
        result[..., index] = physical if neighbor_free else _box_sum(physical, k)
    result[..., 4:] = _geometry(mass, positions, sensing)
    return result


def _context(mass, positions, sensing):
    return np.column_stack((mass, mass * positions[:, 0] / 52,
                            mass * positions[:, 1] / 52, mass * sensing / 5))


def statistics_features(physical, mass, positions, sensing):
    """Twelve raw-mass statistics; entropy ignores entries at or below 1e-10."""
    _feature_inputs(physical, mass, positions, sensing)
    x, y = np.indices((GRID, GRID), dtype=np.float64)
    dx, dy = x - positions[:, 0, None, None], y - positions[:, 1, None, None]
    logs = np.zeros_like(physical)
    keep = physical > EPSILON
    logs[keep] = -np.log2(physical[keep])
    def reduce(value):
        return value.sum(axis=(1, 2), dtype=np.float64)
    return np.column_stack((mass, mass * sensing / 5, mass * positions[:, 0] / 52,
                            mass * positions[:, 1] / 52, reduce(physical * logs) / math.log2(GRID**2),
                            reduce(physical * np.abs(dx)) / 52, reduce(physical * np.abs(dy)) / 52,
                            reduce(physical * dx**2) / 52**2, reduce(physical * dy**2) / 52**2,
                            reduce(physical * dx * dy) / 52**2, reduce(physical**2),
                            physical.max(axis=(1, 2))))


def parameter_shapes(kind):
    """New dict of exact checkpoint parameter names and shapes, in RNG order."""
    kind = _kind(kind)
    result = {}
    if kind == "cnn":
        for i, channels in enumerate((1, 4, 4)):
            result[f"conv_weight_{i}"] = (4, channels, 3, 3)
            result[f"conv_bias_{i}"] = (4,)
    if kind in ("spatial", "neighbor_free", "cnn"):
        sizes = (20, 8, 8) if kind == "cnn" else (12, 24, 8)
        for i, (before, after) in enumerate(pairwise(sizes)):
            result[f"cell_weight_{i}"], result[f"cell_bias_{i}"] = (after, before), (after,)
    sizes = (INPUT_DIM, 128, 1) if kind == "dense128" else (12, 16, 1)
    for i, (before, after) in enumerate(pairwise(sizes)):
        result[f"readout_weight_{i}"], result[f"readout_bias_{i}"] = (after, before), (after,)
    return result


def parameter_count(kind):
    return sum(math.prod(shape) for shape in parameter_shapes(kind).values())


def make_head(kind, seed, c0):
    """Local-generator CPU model, paired spatial/neighbor-free hidden draws.

Hidden weights use He normal sqrt(2/fan_in); every bias and the final readout
weight start at zero. Thus every arm initially returns c0*canonical raw mass.
The independent NumPy implementation below is not used by Torch forward.
"""
    import torch
    from torch.nn import functional

    kind = _kind(kind)
    _require(_integer(seed) and 0 <= seed < 2**63, "valid nonboolean Torch seed required")
    _require(not isinstance(c0, (bool, np.bool_))
             and isinstance(c0, (int, float, np.integer, np.floating)), "numeric c0 required")
    with np.errstate(over="raise", invalid="raise"):
        baseline = np.float32(c0)
    _require(np.isfinite(baseline), "finite float32 c0 required")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    shapes = parameter_shapes(kind)

    def inputs(centered, positions, sensing, dtype):
        _require(isinstance(centered, torch.Tensor) and centered.device.type == "cpu"
                 and centered.dtype == dtype and dtype in (torch.float32, torch.float64)
                 and centered.ndim == 3 and centered.shape[1:] == (CENTERED, CENTERED)
                 and 0 < len(centered) <= MAX_BATCH, "bounded matching CPU centered tensor required")
        batch = len(centered)
        _require(isinstance(positions, torch.Tensor) and positions.device.type == "cpu"
                 and positions.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
                 and positions.shape == (batch, 2) and bool(((positions >= 0) & (positions < GRID)).all()),
                 "in-board integer CPU positions required")
        _require(isinstance(sensing, torch.Tensor) and sensing.device.type == "cpu" and sensing.dtype == dtype
                 and sensing.shape == (batch,) and bool(torch.isfinite(sensing).all())
                 and bool((sensing > 0).all()), "positive finite matching CPU sensing required")
        _require(bool(torch.isfinite(centered).all()) and bool((centered >= 0).all()), "invalid centered mass")
        physical = []
        for i, (x, y) in enumerate(positions.tolist()):
            a, b = 52 - x, 52 - y
            source = centered[i]
            _require(not any(bool(torch.any(region != 0)) for region in
                             (source[:a], source[a + GRID:], source[a:a + GRID, :b], source[a:a + GRID, b + GRID:])),
                     "nonzero mass outside physical board")
            physical.append(source[a:a + GRID, b:b + GRID])
        mass = centered.reshape(batch, -1).sum(dim=1)
        _require(bool((mass <= 1 + 1e-6).all()), "belief mass exceeds public contract")
        return torch.stack(physical), mass, positions.to(dtype=dtype)

    def geometry(mass, positions, sensing):
        axis = torch.arange(GRID, dtype=mass.dtype, device="cpu")
        x, y = torch.meshgrid(axis, axis, indexing="ij")
        shape = (len(mass), GRID, GRID)
        return torch.stack(((x - positions[:, 0, None, None]) / 52,
                            (y - positions[:, 1, None, None]) / 52,
                            (x / 52).expand(shape), ((52 - x) / 52).expand(shape),
                            (y / 52).expand(shape), ((52 - y) / 52).expand(shape),
                            (sensing[:, None, None] / 5).expand(shape), mass[:, None, None].expand(shape)), dim=-1)

    def boxes(physical, k):
        # Independent Torch separable convolution on the physical board.
        ones = torch.ones((1, 1, 1, k), dtype=physical.dtype, device="cpu")
        horizontal = functional.conv2d(physical[:, None], ones, padding=(0, k // 2))
        return functional.conv2d(horizontal, ones.transpose(2, 3), padding=(k // 2, 0))[:, 0]

    def stats(physical, mass, positions, sensing):
        axis = torch.arange(GRID, dtype=physical.dtype, device="cpu")
        x, y = torch.meshgrid(axis, axis, indexing="ij")
        dx, dy = x - positions[:, 0, None, None], y - positions[:, 1, None, None]
        keep = physical > EPSILON
        safe = torch.where(keep, physical, torch.ones_like(physical))
        entropy = (physical * -torch.log2(safe)).sum(dim=(1, 2))
        def reduce(value):
            return value.sum(dim=(1, 2))
        return torch.stack((mass, mass * sensing / 5, mass * positions[:, 0] / 52, mass * positions[:, 1] / 52,
                            entropy / math.log2(GRID**2), reduce(physical * dx.abs()) / 52,
                            reduce(physical * dy.abs()) / 52, reduce(physical * dx**2) / 52**2,
                            reduce(physical * dy**2) / 52**2, reduce(physical * dx * dy) / 52**2,
                            reduce(physical**2), physical.amax(dim=(1, 2))), dim=1)

    class SpatialHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            for name, shape in shapes.items():
                if "bias" in name or name == "readout_weight_1":
                    array = torch.zeros(shape, dtype=torch.float32, device="cpu")
                else:
                    array = torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu")
                    array *= math.sqrt(2 / math.prod(shape[1:]))
                self.register_parameter(name, torch.nn.Parameter(array))
            self.register_buffer("c0", torch.tensor(float(baseline), dtype=torch.float32, device="cpu"))

        def layers(self, value, prefix, *, last_linear=False):
            for i in range(2):
                value = value @ getattr(self, f"{prefix}_weight_{i}").T + getattr(self, f"{prefix}_bias_{i}")
                if not (last_linear and i == 1):
                    value = torch.relu(value)
            return value

        def forward(self, centered, positions, sensing):
            physical, mass, positions = inputs(centered, positions, sensing, self.c0.dtype)
            context = torch.stack((mass, mass * positions[:, 0] / 52,
                                   mass * positions[:, 1] / 52, mass * sensing / 5), dim=1)
            if self.kind == "dense128":
                features = torch.cat((53 * centered.reshape(len(centered), -1), context[:, 1:]), dim=1)
            elif self.kind == "statistics":
                features = stats(physical, mass, positions, sensing)
            else:
                geom = geometry(mass, positions, sensing)
                if self.kind == "cnn":
                    current, levels = 53 * physical[:, None], []
                    for i, dilation in enumerate((1, 3, 9)):
                        current = torch.relu(functional.conv2d(current, getattr(self, f"conv_weight_{i}"),
                                             getattr(self, f"conv_bias_{i}"), padding=dilation, dilation=dilation))
                        levels.append(current.permute(0, 2, 3, 1))
                    local = torch.cat((*levels, geom), dim=-1)
                else:
                    local = torch.stack((53 * physical, *(physical if self.kind == "neighbor_free"
                                          else boxes(physical, k) for k in (3, 9, 27))), dim=-1)
                    local = torch.cat((local, geom), dim=-1)
                hidden = self.layers(local, "cell")
                pooled = (physical[..., None] * hidden).sum(dim=(1, 2))
                features = torch.cat((pooled, context), dim=1)
            result = self.c0 * mass + self.layers(features, "readout", last_linear=True)[:, 0]
            _require(bool(torch.isfinite(result).all()), "nonfinite normalized value")
            return result

    return SpatialHead()


def validate_head(head):
    """Validate exact metadata and native finite float32 tensors; return kind."""
    _require(isinstance(head, Mapping), "checkpoint mapping required")
    _require({"version", "kind", "input_dim", "c0"} <= set(head), "missing checkpoint metadata")
    kind = _kind(_scalar(head["kind"]))
    dimension = _scalar(head["input_dim"])
    _require(_scalar(head["version"]) == VERSION and _integer(dimension) and dimension == INPUT_DIM,
             "unsupported checkpoint version/input dimension")
    shapes = {"c0": (), **parameter_shapes(kind)}
    _require(set(head) == {"version", "kind", "input_dim", *shapes}, "exact checkpoint membership required")
    for name, shape in shapes.items():
        _array(head[name], shape, name, np.float32)
    return kind


def export_head(model):
    """Return owned float32 arrays, excluding optimizer state and fixed geometry."""
    import torch

    kind = _kind(getattr(model, "kind", None))
    parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    _require(set(parameters) == set(parameter_shapes(kind)) and set(buffers) == {"c0"},
             "exact learned tensors and baseline required")
    result = {"version": VERSION, "kind": kind, "input_dim": INPUT_DIM}
    for name, tensor in {**parameters, **buffers}.items():
        _require(tensor.device.type == "cpu" and tensor.dtype == torch.float32, "CPU float32 export required")
        result[name] = tensor.detach().numpy().copy()
    validate_head(result)
    return result


def _immutable(array):
    converted = array.astype(np.float64)
    return np.frombuffer(converted.tobytes(order="C"), dtype=np.float64).reshape(converted.shape)


def _dense_layers(value, arrays, prefix, *, last_linear=False):
    for i in range(2):
        value = value @ arrays[f"{prefix}_weight_{i}"].T + arrays[f"{prefix}_bias_{i}"]
        if not (last_linear and i == 1):
            value = np.maximum(value, 0)
    return value


def _convolve(value, weight, bias, dilation):
    # Channels-last explicit cross-correlation; no padded activation survives.
    result = np.broadcast_to(bias, (*value.shape[:3], len(bias))).copy()
    for i in range(3):
        tx, sx = _slices((i - 1) * dilation)
        for j in range(3):
            ty, sy = _slices((j - 1) * dilation)
            result[:, tx, ty] += value[:, sx, sy] @ weight[:, :, i, j].T
    return np.maximum(result, 0)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class FrozenValue:
    """Immutable float64 deployment without retained Torch or float32 weights.

normalized() requires a sensing vector. The branch callback requires bound
sensing_length and validates kernel shape/dtype only: actual kernel values,
origin, finiteness, range and sensing identity must be authenticated by the
qualified upstream branch builder. The readout never consults that kernel.
"""

    kind: str
    sensing_length: float | None
    arrays: Mapping

    def __init__(self, head, sensing_length=None):
        kind = validate_head(head)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "sensing_length", None if sensing_length is None else _sensing_scalar(sensing_length))
        object.__setattr__(self, "arrays", MappingProxyType({name: _immutable(head[name])
                           for name in ("c0", *parameter_shapes(kind))}))

    def normalized(self, centered, positions, sensing):
        physical, mass = unpack_centered(centered, positions)
        _array(sensing, (len(centered),), "sensing")
        _require((sensing > 0).all(), "positive sensing lengths required")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            context = _context(mass, positions, sensing)
            if self.kind == "dense128":
                features = np.concatenate((53 * centered.reshape(len(centered), -1), context[:, 1:]), axis=1)
            elif self.kind == "statistics":
                features = statistics_features(physical, mass, positions, sensing)
            else:
                if self.kind == "cnn":
                    current, levels = 53 * physical[..., None], []
                    for i, dilation in enumerate((1, 3, 9)):
                        current = _convolve(current, self.arrays[f"conv_weight_{i}"],
                                            self.arrays[f"conv_bias_{i}"], dilation)
                        levels.append(current)
                    local = np.concatenate((*levels, _geometry(mass, positions, sensing)), axis=-1)
                else:
                    local = spatial_features(physical, mass, positions, sensing, self.kind == "neighbor_free")
                hidden = _dense_layers(local, self.arrays, "cell")
                pooled = np.sum(physical[..., None] * hidden, axis=(1, 2), dtype=np.float64)
                features = np.concatenate((pooled, context), axis=1)
            result = self.arrays["c0"] * mass + _dense_layers(features, self.arrays, "readout", last_linear=True)[:, 0]
        _require(result.shape == (len(centered),) and np.isfinite(result).all(), "nonfinite normalized value")
        return result

    def __call__(self, centered, positions, kernel):
        _require(self.sensing_length is not None, "branch callback requires bound sensing length")
        _require(isinstance(kernel, np.ndarray) and kernel.dtype == np.float64 and kernel.shape == (4, 107, 107),
                 "qualified float64 branch kernel required")
        _require(isinstance(centered, np.ndarray) and centered.ndim == 3, "centered batch required")
        sensing = np.full(len(centered), self.sensing_length, dtype=np.float64)
        with np.errstate(over="raise", invalid="raise"):
            result = SCALE * self.normalized(centered, positions, sensing)
        _require(np.isfinite(result).all(), "nonfinite physical value")
        return result

    def storage_bytes(self):
        return {"parameter_array_bytes": sum(array.nbytes for name, array in self.arrays.items() if name != "c0"),
                "baseline_array_bytes": self.arrays["c0"].nbytes, "mutable_array_bytes": 0,
                "scope": "Owned immutable float64 parameters and baseline; excludes scalar metadata, inputs, and temporary geometry/feature/convolution workspace."}
