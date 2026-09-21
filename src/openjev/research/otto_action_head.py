"""Public-state features and a small CPU action-score head, without planning.

Actors are supplied by the caller. Feature extraction never decodes a compact
belief, accesses an environment, or invokes a planner. Training-set membership,
supervision and execution budgets belong to the caller. NumPy inference uses
exported float32 weights; Torch is imported only for construction/export.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

VERSION = "otto-action-head-v1"
ARMS = ("dct16_neutral", "dct16_nearest", "recent32_hard", "full_bayes")
PUBLIC_FIELDS = {"position", "hit", "done", "step", "valid_actions"}
SIZE, HORIZON = 53, 2188
WEIGHTS = ("weight0", "bias0", "weight1", "bias1", "weight2", "bias2")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
            and value >= minimum, f"{name} must be an integer >= {minimum}")
    return int(value)


def input_dimension(arm):
    require(arm in ARMS, "unknown action-head arm")
    return (2809 if arm == "full_bayes" else 256) + 2809 + 15


def legal_actions(values):
    require(isinstance(values, (list, tuple)) and bool(values), "nonempty legal-action sequence required")
    actions = tuple(integer(a, "action") for a in values)
    require(actions == tuple(sorted(set(actions))) and actions[-1] < 4, "canonical legal actions required")
    return actions


def packet_values(packet, initial_hit):
    if isinstance(packet, Mapping):
        values = dict(packet)
    elif isinstance(packet, tuple) and set(getattr(packet, "_fields", ())) == PUBLIC_FIELDS:
        values = dict(zip(packet._fields, packet, strict=True))
    else:
        raise ValueError("exact public packet mapping or named tuple required")
    require(set(values) == PUBLIC_FIELDS, "packet must contain only the five public fields")
    require(isinstance(values["done"], (bool, np.bool_)) and not values["done"], "terminal packet has no action features")
    position = values["position"]
    require(isinstance(position, (tuple, list)) and len(position) == 2, "two public coordinates required")
    position = tuple(integer(p, "coordinate") for p in position)
    require(max(position) < SIZE, "position outside grid")
    step, hit = integer(values["step"], "step"), integer(values["hit"], "hit")
    initial_hit = integer(initial_hit, "initial_hit", 1)
    require(step <= HORIZON and hit < 4 and initial_hit < 4, "public time or hit outside frozen range")
    actions = legal_actions(values["valid_actions"])
    expected = tuple(a for a in range(4) if 0 <= position[a // 2] + (-1 if a % 2 == 0 else 1) < SIZE)
    require(actions == expected, "legal actions disagree with public boundary")
    if step == 0:
        require(position == (26, 26) and hit == initial_hit, "initial packet must be the supplied center hit")
    return position, step, hit, initial_hit, actions


def features(arm, actor, public_packet, initial_hit, *, sensing_length):
    """Return float32 [evidence, exact excluded bits, 15 public context values].

    Recent slots are right-aligned, oldest to newest. A slot is [x/26-1,
    y/26-1, onehot(hit,4), valid, age/32]; unused leading slots are zero.
    Initial evidence is represented by context, not inserted into the ring.
    Full probabilities use the invertible 53*sqrt(p) scale (grid RMS one).
    The last context scalar is the publicly supplied sensing length divided by 4.
    """
    dimension = input_dimension(arm)
    require(isinstance(sensing_length, Real) and not isinstance(sensing_length, (bool, np.bool_))
            and sensing_length in (3, 4), "sensing_length must be the supplied known model value 3 or 4")
    position, step, hit, initial_hit, actions = packet_values(public_packet, initial_hit)
    require(integer(actor.step, "actor.step") == step and tuple(actor.position) == position,
            "actor and public packet must refer to the same completed prefix")
    excluded = np.asarray(actor.excluded)
    require(excluded.shape == (SIZE, SIZE) and excluded.dtype == np.bool_ and not excluded.all()
            and excluded[position], "exact nonempty-support Boolean exclusion mask required")
    if arm.startswith("dct16_"):
        require(actor.q == 16 and actor.extension == arm.removeprefix("dct16_")
                and actor.initial_hit == initial_hit and actor.done is False, "DCT actor identity mismatch")
        coefficients = np.asarray(actor.coefficients)
        require(coefficients.shape == (16, 16) and np.isfinite(coefficients).all(), "finite 16x16 coefficients required")
        evidence = (np.sign(coefficients) * np.log1p(np.abs(coefficients))).reshape(-1)
    elif arm == "full_bayes":
        require(actor.mode == arm, "full actor identity mismatch")
        probability = np.asarray(actor.probabilities)
        require(probability.shape == (SIZE, SIZE) and np.isfinite(probability).all()
                and np.all(probability >= 0) and abs(float(probability.sum()) - 1) <= 1e-10
                and not probability[excluded].any(), "normalized full belief with exact exclusions required")
        evidence = (SIZE * np.sqrt(probability)).reshape(-1)
    else:
        require(actor.mode == arm and integer(actor.count, "history count") == step, "recent actor identity/count mismatch")
        history = np.asarray(actor.history)
        require(history.shape == (32, 3) and np.issubdtype(history.dtype, np.integer), "integer 32x3 public history required")
        evidence = np.zeros((32, 8), dtype=np.float32)
        first = max(0, step - 32)
        for slot, index in enumerate(range(first, step), start=32 - (step - first)):
            x, y, category = (int(v) for v in history[index % 32])
            require(0 <= x < SIZE and 0 <= y < SIZE and 0 <= category < 4, "invalid retained reading")
            evidence[slot, :2] = (x / 26 - 1, y / 26 - 1)
            evidence[slot, 2 + category] = 1
            evidence[slot, 6:] = (1, (step - index - 1) / 32)
        if step:
            require(tuple(int(v) for v in history[(step - 1) % 32]) == (*position, hit), "latest retained reading disagrees with packet")
        evidence = evidence.reshape(-1)
    context = np.zeros(15, dtype=np.float32)
    context[:2] = [p / 26 - 1 for p in position]
    context[2 + initial_hit - 1] = 1
    context[5 + hit] = 1
    context[9] = np.log1p(step) / np.log1p(HORIZON)
    context[[10 + a for a in actions]] = 1
    context[14] = sensing_length / 4
    result = np.concatenate((evidence, excluded.reshape(-1), context)).astype(np.float32)
    require(result.shape == (dimension,) and np.isfinite(result).all(), "finite exact feature shape required")
    return result


def fit_standardizer(train_features):
    """Fit population moments on the caller's TRAIN matrix, with std floor 1."""
    x = np.asarray(train_features)
    require(x.ndim == 2 and min(x.shape) > 0 and np.isfinite(x).all(), "nonempty finite training matrix required")
    mean = np.mean(x, axis=0, dtype=np.float64)
    scale = np.maximum(np.std(x, axis=0, dtype=np.float64), 1)
    mean, scale = mean.astype(np.float32), scale.astype(np.float32)
    require(np.isfinite(mean).all() and np.isfinite(scale).all(), "training moments overflowed float32")
    return mean, scale


def standardize(x, mean, scale):
    x, mean, scale = (np.asarray(v, dtype=np.float32) for v in (x, mean, scale))
    require(x.ndim in (1, 2) and mean.ndim == scale.ndim == 1 and x.shape[-1] == len(mean) == len(scale),
            "standardizer dimension mismatch")
    require(np.isfinite(x).all() and np.isfinite(mean).all() and np.isfinite(scale).all()
            and np.all(scale >= 1), "finite standardizer with scale >=1 required")
    result = (x - mean) / scale
    require(np.isfinite(result).all(), "standardized input overflow")
    return result


def make_head(input_dim, seed):
    """Seed-local CPU float32 MLP; leaves global Torch RNG unchanged."""
    import torch

    dimension, seed = integer(input_dim, "input_dim", 1), integer(seed, "seed")
    require(seed < 2**63, "seed outside supported range")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(dimension, 64, device="cpu", dtype=torch.float32), torch.nn.Tanh(),
            torch.nn.Linear(64, 32, device="cpu", dtype=torch.float32), torch.nn.Tanh(),
            torch.nn.Linear(32, 4, device="cpu", dtype=torch.float32))
    return model


def parameter_count(input_dim_or_model):
    if isinstance(input_dim_or_model, Integral):
        return 64 * integer(input_dim_or_model, "input_dim", 1) + 2276
    return sum(p.numel() for p in input_dim_or_model.parameters())


def export_head(model, mean, scale):
    """Copy strict CPU float32 MLP parameters and TRAIN standardizer to NumPy."""
    import torch

    require(isinstance(model, torch.nn.Sequential) and len(model) == 5, "five-layer Sequential required")
    require(type(model[1]) is torch.nn.Tanh and type(model[3]) is torch.nn.Tanh, "Tanh activations required")
    layers = [model[i] for i in (0, 2, 4)]
    require(all(type(layer) is torch.nn.Linear for layer in layers)
            and [(layer.in_features, layer.out_features) for layer in layers]
            == [(layers[0].in_features, 64), (64, 32), (32, 4)], "frozen MLP dimensions required")
    dimension = layers[0].in_features
    standardize(np.zeros(dimension, dtype=np.float32), mean, scale)
    result = {"version": VERSION, "input_dim": dimension,
              "mean": np.asarray(mean, dtype=np.float32).copy(), "scale": np.asarray(scale, dtype=np.float32).copy()}
    for index, layer in enumerate(layers):
        for name, tensor in (("weight", layer.weight), ("bias", layer.bias)):
            require(tensor is not None and tensor.dtype == torch.float32 and tensor.device.type == "cpu",
                    "CPU float32 weights and biases required")
            value = tensor.detach().numpy().copy()
            require(np.isfinite(value).all(), "finite head parameters required")
            result[f"{name}{index}"] = value
    validate_head(result)
    return result


def validate_head(head):
    require(isinstance(head, dict) and set(head) == {"version", "input_dim", "mean", "scale", *WEIGHTS}
            and head["version"] == VERSION, "exact exported head schema required")
    dimension = integer(head["input_dim"], "input_dim", 1)
    shapes = ((64, dimension), (64,), (32, 64), (32,), (4, 32), (4,))
    for key, shape in zip(WEIGHTS, shapes, strict=True):
        a = head[key]
        require(isinstance(a, np.ndarray) and a.dtype == np.float32 and a.shape == shape
                and np.isfinite(a).all(), f"invalid exported array: {key}")
    for key in ("mean", "scale"):
        a = head[key]
        require(isinstance(a, np.ndarray) and a.dtype == np.float32 and a.shape == (dimension,)
                and np.isfinite(a).all(), f"invalid exported array: {key}")
    require(np.all(head["scale"] >= 1), "standardizer scale must be at least one")
    return dimension


def predict(head_dict, x, valid_actions):
    """Four lower-is-better scores, with illegal actions represented by None."""
    dimension = validate_head(head_dict)
    arrays = tuple(head_dict[k] for k in (*WEIGHTS, "mean", "scale"))
    return _forward(arrays, dimension, x, valid_actions)


def _forward(arrays, dimension, x, valid_actions):
    actions = legal_actions(valid_actions)
    value = np.asarray(x, dtype=np.float32)
    require(value.shape == (dimension,), "one complete feature vector required")
    value = standardize(value, arrays[6], arrays[7])
    for index in range(3):
        value = arrays[2 * index] @ value + arrays[2 * index + 1]
        if index < 2:
            value = np.tanh(value)
    require(value.shape == (4,) and np.isfinite(value).all(), "finite four-action output required")
    return [float(value[a]) if a in actions else None for a in range(4)]


def storage_bytes(head_dict):
    """Exported array payload only; excludes actor state, workspaces and Python."""
    validate_head(head_dict)
    return _storage(tuple(head_dict[k] for k in (*WEIGHTS, "mean", "scale")))


def _storage(arrays):
    weights = sum(a.nbytes for a in arrays[:6])
    moments = arrays[6].nbytes + arrays[7].nbytes
    return {"parameter_count": sum(a.size for a in arrays[:6]), "parameter_bytes": weights,
            "standardizer_bytes": moments, "total_array_bytes": weights + moments,
            "scope": "Head and standardizer array payloads; excludes actor, temporary arrays and Python overhead."}


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class FrozenHead:
    """Validate once and retain only immutable bytes-backed inference arrays.

    Construction is a paid per-checkpoint operation. Per-call input, action and
    output checks remain; weight validation is not repeated. No exported mapping
    or mutable weight copy is retained. Array payload reporting excludes Python
    metadata, construction temporaries and inference workspaces.
    """

    _arrays: tuple[np.ndarray, ...]
    _dimension: int

    def __init__(self, exportdict):
        dimension = validate_head(exportdict)
        arrays = tuple(np.frombuffer(exportdict[k].tobytes(order="C"), dtype=np.float32)
                       .reshape(exportdict[k].shape) for k in (*WEIGHTS, "mean", "scale"))
        object.__setattr__(self, "_dimension", dimension)
        object.__setattr__(self, "_arrays", arrays)

    def predict(self, x, valid_actions):
        return _forward(self._arrays, self._dimension, x, valid_actions)

    def storage_bytes(self):
        return _storage(self._arrays)
