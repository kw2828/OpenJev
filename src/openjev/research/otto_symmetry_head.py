"""Full public-belief action readouts with a fixed D4 coordinate convention.

Features contain no simulator, source, history, learned standardizer or policy
target. Lower scores are better. The caller owns training data, loss weighting,
action eligibility, lifecycle and timing. Torch is imported only on its paths.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

VERSION = "otto-symmetry-head-v1"
SIZE, INPUT_DIM, MAX_BATCH = 53, 2836, 512
KINDS = ("d4_shared", "dense_augmented")
WEIGHTS = ("weight0", "bias0", "weight1", "bias1", "weight2", "bias2")
PUBLIC_FIELDS = frozenset({"position", "hit", "done", "step", "valid_actions"})
POSITION, LEGAL, SENSING, FORECAST = 2809, 2811, 2815, 2816


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, lower=0, upper=None):
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    _require(value >= lower and (upper is None or value <= upper), f"invalid {name}")
    return int(value)


def _immutable(array):
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _transform_vector(vector, group):
    x, y = vector
    if group >= 4:
        y = -y
    for _ in range(group % 4):
        x, y = -y, x
    return x, y


def _tables():
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    indices, signs, permutations = [], [], []
    for group in range(8):
        permutation = [directions.index(_transform_vector(d, group)) for d in directions]
        index = np.arange(INPUT_DIM, dtype=np.int64)
        sign = np.ones(INPUT_DIM, dtype=np.float32)
        grid = np.arange(SIZE * SIZE).reshape(SIZE, SIZE)
        if group >= 4:
            grid = grid[:, ::-1]
        index[:POSITION] = np.rot90(grid, group % 4).reshape(-1)
        matrix = np.array([_transform_vector((1, 0), group), _transform_vector((0, 1), group)]).T
        for row in range(2):
            col = int(np.flatnonzero(matrix[row])[0])
            index[POSITION + row] = POSITION + col
            sign[POSITION + row] = matrix[row, col]
        for action, transformed in enumerate(permutation):
            index[LEGAL + transformed] = LEGAL + action
            index[FORECAST + transformed * 5:FORECAST + transformed * 5 + 5] = np.arange(
                FORECAST + action * 5, FORECAST + action * 5 + 5)
        indices.append(index)
        signs.append(sign)
        permutations.append(permutation)
    return tuple(_immutable(np.asarray(a)) for a in (indices, signs, permutations))


_INDICES, _SIGNS, _PERMUTATIONS = _tables()
_NORTH_FRAMES = _immutable(np.array([np.flatnonzero(_PERMUTATIONS[:, a] == 0)
                                    for a in range(4)], dtype=np.int64))


def action_permutation(group):
    """Original action -> transformed action, for 0:-x, 1:+x, 2:-y, 3:+y.

    g=0..3 rotates the grid with np.rot90(g); g=4..7 first reflects its
    second coordinate, then rotates np.rot90(g % 4).
    """
    return tuple(int(a) for a in _PERMUTATIONS[_integer(group, "group", 0, 7)])


def transform_features(x, group):
    """Apply the exact signed/permuted feature representation; returns a copy."""
    group = _integer(group, "group", 0, 7)
    x = np.asarray(x)
    _require(x.ndim in (1, 2) and x.shape[-1] == INPUT_DIM and x.dtype == np.float32
             and np.isfinite(x).all(), "finite float32 feature vector or matrix required")
    return x[..., _INDICES[group]] * _SIGNS[group]


def transform_actions(values, group):
    """Relabel last-axis four-action values, masks or target probabilities."""
    group = _integer(group, "group", 0, 7)
    values = np.asarray(values)
    _require(values.ndim >= 1 and values.shape[-1] == 4, "last axis must have four actions")
    return values[..., np.argsort(_PERMUTATIONS[group])].copy()


def _public(packet):
    if isinstance(packet, tuple) and hasattr(packet, "_asdict"):
        packet = packet._asdict()
    _require(isinstance(packet, Mapping) and set(packet) == PUBLIC_FIELDS, "exact public packet fields required")
    _require(isinstance(packet["done"], (bool, np.bool_)) and not packet["done"], "nonterminal packet required")
    position = packet["position"]
    _require(isinstance(position, (list, tuple, np.ndarray)) and len(position) == 2, "two coordinates required")
    position = tuple(_integer(v, "coordinate", 0, 52) for v in position)
    step = _integer(packet["step"], "step")
    hit = _integer(packet["hit"], "hit", 0, 3)
    if step == 0:
        _require(position == (26, 26) and hit > 0, "initial public packet must have a positive center hit")
    values = packet["valid_actions"]
    _require(isinstance(values, (list, tuple)), "canonical public action sequence required")
    actions = tuple(_integer(a, "action", 0, 3) for a in values)
    expected = tuple(a for a in range(4) if 0 <= position[a // 2] + 2 * (a % 2) - 1 < SIZE)
    _require(actions == expected, "public inbounds directions disagree with position")
    return position, actions


@dataclass(frozen=True, slots=True, init=False)
class PublicFeatureMap:
    """Immutable known public sensor and fixed-scale features, with no history.

    Order: 53*sqrt(p) flattened C-order (2809), centered position (2),
    inbounds indicators (4), sensing_length/5 (1), action-major forecasts
    (4*5): [p(clamped successor), joint nonfound hit masses 0..3].
    Forecasts include blocked stays. Kernel origin is zero; no conditional
    posterior, mass floor, normalization, entropy or lookahead is computed.
    Kernel inputs may be synthetic, but their exact D4 symmetry is required.
    """
    kernel: np.ndarray
    sensing_length: float

    def __init__(self, kernel, sensing_length):
        _require(isinstance(kernel, np.ndarray) and kernel.dtype == np.float64
                 and kernel.shape == (4, 107, 107) and np.isfinite(kernel).all()
                 and np.all((kernel >= 0) & (kernel <= 1)) and not kernel[:, 53, 53].any(),
                 "finite float64 public kernel [4,107,107] in [0,1] with zero origin required")
        _require(np.array_equal(kernel, kernel.transpose(0, 2, 1))
                 and np.array_equal(kernel, kernel[:, ::-1, :]), "D4-symmetric public kernel required")
        _require(isinstance(sensing_length, Real) and not isinstance(sensing_length, (bool, np.bool_))
                 and np.isfinite(sensing_length) and sensing_length > 0, "positive finite sensing length required")
        object.__setattr__(self, "kernel", _immutable(kernel))
        object.__setattr__(self, "sensing_length", float(sensing_length))

    def features(self, probability, public_packet):
        position, actions = _public(public_packet)
        _require(isinstance(probability, np.ndarray) and probability.dtype == np.float64
                 and probability.shape == (SIZE, SIZE) and np.isfinite(probability).all()
                 and np.all(probability >= 0), "finite nonnegative float64[53,53] public belief required")
        x = np.empty(INPUT_DIM, dtype=np.float32)
        x[:POSITION] = (SIZE * np.sqrt(probability)).reshape(-1)
        x[POSITION:LEGAL] = [p / 26 - 1 for p in position]
        x[LEGAL:SENSING] = 0
        x[[LEGAL + a for a in actions]] = 1
        x[SENSING] = self.sensing_length / 5
        for action in range(4):
            successor = list(position)
            axis = action // 2
            successor[axis] = min(52, max(0, successor[axis] + 2 * (action % 2) - 1))
            row, col = 53 - successor[0], 53 - successor[1]
            likelihood = self.kernel[:, row:row + SIZE, col:col + SIZE]
            start = FORECAST + 5 * action
            x[start] = probability[tuple(successor)]
            x[start + 1:start + 5] = np.sum(likelihood * probability, axis=(1, 2), dtype=np.float64)
        _require(np.isfinite(x).all(), "feature overflow")
        return x

    def storage_bytes(self):
        return {"immutable_array_bytes": self.kernel.nbytes, "mutable_array_bytes": 0,
                "scope": "Owned immutable sensor kernel; excludes scalar metadata and temporary feature workspace."}


def parameter_count(kind):
    _require(kind in KINDS, "unknown head kind")
    return 91329 if kind == "d4_shared" else 91380


def make_head(kind, seed):
    """Seed-local CPU float32 2836->32->16->(1 or 4) Tanh readout.

    model(x) always returns four raw costs. Shared forward averages the two
    frame scalar costs mapping each original action to -x. Dense forward is
    unaugmented; training_losses supplies its complete D4 augmentation.
    """
    import torch

    _require(kind in KINDS, "unknown head kind")
    seed = _integer(seed, "seed", 0, 2**63 - 1)

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.core = torch.nn.Sequential(
                torch.nn.Linear(INPUT_DIM, 32, dtype=torch.float32, device="cpu"), torch.nn.Tanh(),
                torch.nn.Linear(32, 16, dtype=torch.float32, device="cpu"), torch.nn.Tanh(),
                torch.nn.Linear(16, 1 if kind == "d4_shared" else 4, dtype=torch.float32, device="cpu"))
            self.register_buffer("indices", torch.tensor(_INDICES.copy()), persistent=False)
            self.register_buffer("signs", torch.tensor(_SIGNS.copy()), persistent=False)
            self.register_buffer("permutations", torch.tensor(_PERMUTATIONS.copy()), persistent=False)
            self.register_buffer("north_frames", torch.tensor(_NORTH_FRAMES.copy()), persistent=False)

        def views(self, x):
            return x[..., self.indices] * self.signs

        def forward(self, x):
            if self.kind == "dense_augmented":
                return self.core(x)
            scalar = self.core(self.views(x)).squeeze(-1)
            return scalar[..., self.north_frames].mean(dim=-1)

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return Head()


def training_losses(model, x, targets, allowed):
    """Per-state CE; caller applies episode weights and its batch denominator.

    Targets are distributions over allowed actions. Dense loss averages eight
    separately masked CEs, not CE of an ensemble. Shared loss uses its single
    four-cost group aggregate. No detach, weights or optimizer occur here.
    """
    import torch

    _require(getattr(model, "kind", None) in KINDS, "known head required")
    _require(x.ndim == 2 and x.shape[1] == INPUT_DIM and x.dtype == torch.float32
             and x.device.type == "cpu" and torch.isfinite(x).all().item(), "finite CPU float32 features required")
    _require(targets.shape == allowed.shape == (x.shape[0], 4) and targets.dtype == torch.float32
             and allowed.dtype == torch.bool and targets.device == allowed.device == x.device
             and torch.isfinite(targets).all().item() and (targets >= 0).all().item()
             and allowed.any(dim=-1).all().item() and (targets[~allowed] == 0).all().item()
             and torch.allclose(targets.sum(dim=-1), torch.ones(x.shape[0]), atol=1e-6, rtol=0),
             "finite normalized targets and nonempty allowed masks required")

    def ce(costs, probabilities, mask):
        _require(torch.isfinite(costs).all().item(), "nonfinite training costs")
        logp = torch.log_softmax((-costs).masked_fill(~mask, -torch.inf), dim=-1)
        return -(probabilities * logp.masked_fill(~mask, 0)).sum(dim=-1)

    if model.kind == "d4_shared":
        return ce(model(x), targets, allowed)
    inverse = torch.argsort(model.permutations, dim=-1)
    return ce(model.core(model.views(x)), targets[:, inverse], allowed[:, inverse]).mean(dim=-1)


def export_head(model):
    """Copy the six CPU float32 parameter arrays; no fitted normalization."""
    import torch

    kind = getattr(model, "kind", None)
    _require(kind in KINDS and isinstance(model.core, torch.nn.Sequential) and len(model.core) == 5
             and type(model.core[1]) is torch.nn.Tanh and type(model.core[3]) is torch.nn.Tanh,
             "exact Tanh readout required")
    result = {"version": VERSION, "kind": kind, "input_dim": INPUT_DIM}
    for index, layer in enumerate(model.core[i] for i in (0, 2, 4)):
        _require(type(layer) is torch.nn.Linear, "Linear layers required")
        for name, tensor in (("weight", layer.weight), ("bias", layer.bias)):
            _require(tensor is not None and tensor.device.type == "cpu" and tensor.dtype == torch.float32,
                     "CPU float32 parameters required")
            result[f"{name}{index}"] = tensor.detach().numpy().copy()
    validate_head(result)
    return result


def validate_head(head):
    _require(isinstance(head, dict) and set(head) == {"version", "kind", "input_dim", *WEIGHTS}
             and head["version"] == VERSION and head["kind"] in KINDS
             and _integer(head["input_dim"], "input dimension") == INPUT_DIM, "exact head export schema required")
    outputs = 1 if head["kind"] == "d4_shared" else 4
    shapes = ((32, INPUT_DIM), (32,), (16, 32), (16,), (outputs, 16), (outputs,))
    for key, shape in zip(WEIGHTS, shapes, strict=True):
        value = head[key]
        _require(isinstance(value, np.ndarray) and value.dtype == np.float32 and value.shape == shape
                 and np.isfinite(value).all(), f"invalid parameter array {key}")
    return head["kind"]


@dataclass(frozen=True, slots=True, init=False)
class FrozenHead:
    """One-time validated, immutable NumPy weights, no persistent input cache.

    scores accepts one vector or at most MAX_BATCH vectors. dense_ensemble is
    inference-only and reuses dense weights, averaging inverse-permuted costs.
    All four scores remain finite and unmasked; selection belongs to caller.
    """
    kind: str
    arrays: tuple

    def __init__(self, head):
        kind = validate_head(head)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "arrays", tuple(_immutable(head[k]) for k in WEIGHTS))

    def scores(self, x, *, kind=None):
        route = self.kind if kind is None else kind
        _require(route == self.kind or (self.kind == "dense_augmented" and route == "dense_ensemble"),
                 "incompatible inference route")
        x = np.asarray(x)
        _require(x.dtype == np.float32 and x.ndim in (1, 2) and x.shape[-1] == INPUT_DIM
                 and (x.ndim == 1 or 0 < x.shape[0] <= MAX_BATCH) and np.isfinite(x).all(),
                 "bounded finite float32 features required")
        value = x if route == "dense_augmented" else x[..., _INDICES] * _SIGNS
        for index in range(3):
            value = value @ self.arrays[2 * index].T + self.arrays[2 * index + 1]
            if index < 2:
                value = np.tanh(value)
        if route == "d4_shared":
            value = value[..., 0][..., _NORTH_FRAMES].mean(axis=-1, dtype=np.float32)
        elif route == "dense_ensemble":
            value = np.take_along_axis(value, np.broadcast_to(_PERMUTATIONS, value.shape), axis=-1).mean(
                axis=-2, dtype=np.float32)
        _require(value.shape == (*x.shape[:-1], 4) and np.isfinite(value).all(), "nonfinite or malformed four costs")
        return value

    def storage_bytes(self):
        return {"parameter_array_bytes": sum(a.nbytes for a in self.arrays), "mutable_array_bytes": 0,
                "shared_transform_array_bytes": sum(a.nbytes for a in (_INDICES, _SIGNS, _PERMUTATIONS, _NORTH_FRAMES)),
                "scope": "Owned immutable weights plus separately reported module-shared transform tables; excludes feature kernel, caller belief, scalar metadata and temporary batch workspace."}
