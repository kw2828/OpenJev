"""Small query gates: NumPy deployment, lazy CPU Torch construction only.

The caller authenticates files/data and owns training, evidence and admission.
There is no planner, simulator, file loading or global RNG mutation here.
"""
from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np

VERSION = "otto-query-gate-model-v1"
KINDS = ("gru32", "mlp190")
SEEDS = (40101, 40102, 40103)
INPUT_DIM = 31
THRESHOLD = .05
PARITY_ATOL = 2e-5
SHAPES = {
    "gru32": {"recurrent.weight_ih_l0": (96, 31), "recurrent.weight_hh_l0": (96, 32),
              "recurrent.bias_ih_l0": (96,), "recurrent.bias_hh_l0": (96,),
              "output.weight": (1, 32), "output.bias": (1,)},
    "mlp190": {"hidden.weight": (190, 31), "hidden.bias": (190,),
               "output.weight": (1, 190), "output.bias": (1,)},
}
META = {"version", "kind", "seed", "threshold", "input_dim"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def owned(value):
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)


def parameter_count(kind):
    require(kind in KINDS, "declared model kind")
    return sum(math.prod(shape) for shape in SHAPES[kind].values())


def make_gate(kind, seed):
    """Standard pinned-Torch initialization, followed by the shared constant head.

    fork_rng restores the caller's CPU RNG. No GPU initialization is requested.
    Both models begin with the same constant logit float32(log(19)); GRU state
    can differ internally, but the zero output weights hide it before fitting.
    """
    require(kind in KINDS and type(seed) is int and seed in SEEDS, "fixed kind/seed")
    import torch

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kind, self.seed = kind, seed
            self.state_size = 32 if kind == "gru32" else 0
            if kind == "gru32":
                self.recurrent = torch.nn.GRU(31, 32, batch_first=True, device="cpu", dtype=torch.float32)
            else:
                self.hidden = torch.nn.Linear(31, 190, device="cpu", dtype=torch.float32)
            self.output = torch.nn.Linear(32 if kind == "gru32" else 190, 1, device="cpu", dtype=torch.float32)
            with torch.no_grad():
                self.output.weight.zero_()
                self.output.bias.fill_(math.log(19))

        def forward(self, features, state):
            if self.kind == "gru32":
                hidden, final = self.recurrent(features, state.unsqueeze(0))
                return self.output(hidden).squeeze(-1), final[0]
            hidden = torch.tanh(self.hidden(features))
            return self.output(hidden).squeeze(-1), state

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = Gate()
    require(sum(p.numel() for p in model.parameters()) == parameter_count(kind), "exact parameter count")
    return model


def export_gate(model):
    result = {"version": np.asarray(VERSION), "kind": np.asarray(model.kind),
              "seed": np.asarray(model.seed, dtype=np.int64), "threshold": np.asarray(THRESHOLD, dtype=np.float64),
              "input_dim": np.asarray(INPUT_DIM, dtype=np.int64)}
    for name, value in model.state_dict().items():
        result[name] = owned(value.detach().cpu().numpy())
    validate_export(result)
    return result


def validate_export(export):
    require(isinstance(export, dict) and META <= set(export), "complete plain checkpoint mapping")
    for name in META:
        require(isinstance(export[name], np.ndarray) and export[name].shape == (), "scalar checkpoint metadata")
    kind = export["kind"].item()
    require(export["kind"].dtype.kind == "U" and kind in KINDS
            and export["version"].dtype.kind == "U" and export["version"].item() == VERSION,
            "declared checkpoint kind/version")
    require(export["seed"].dtype == np.int64 and export["seed"].item() in SEEDS
            and export["input_dim"].dtype == np.int64 and export["input_dim"].item() == INPUT_DIM
            and export["threshold"].dtype == np.float64 and export["threshold"].item() == THRESHOLD,
            "fixed checkpoint metadata")
    require(set(export) == META | SHAPES[kind].keys(), "exact checkpoint tensors")
    for name, shape in SHAPES[kind].items():
        a = export[name]
        require(isinstance(a, np.ndarray) and a.dtype == np.float32 and a.shape == shape
                and np.isfinite(a).all(), f"finite float32 checkpoint tensor: {name}")
    return kind


def restore_gate(export):
    """Restore one CPU Torch copy for counted parity; caller counts construction."""
    kind = validate_export(export)
    import torch
    model = make_gate(kind, int(export["seed"]))
    model.load_state_dict({name: torch.from_numpy(export[name].copy()) for name in SHAPES[kind]}, strict=True)
    return model.eval()


def sigmoid(value):
    """Finite float32 logistic without positive-exponent overflow."""
    a = np.asarray(value, dtype=np.float32)
    result = np.empty_like(a)
    positive = a >= 0
    result[positive] = np.float32(1) / (np.float32(1) + np.exp(-a[positive]))
    exp = np.exp(a[~positive])
    result[~positive] = exp / (np.float32(1) + exp)
    return result


class FrozenGate:
    """Owned immutable float32 saved weights, with caller-owned recurrent state.

    step(features, state) returns (query, next_state, logit, probability).
    __call__ returns only (query,next_state) for QueryGateActor. The threshold is
    applied to the Python float of the computed float32 probability, >= 0.05.
    No threshold snapping or hidden-state clipping is performed.
    """
    def __init__(self, export):
        self.kind = validate_export(export)
        self.seed = int(export["seed"])
        self.state_size = 32 if self.kind == "gru32" else 0
        self.weights = MappingProxyType({name: owned(export[name]) for name in SHAPES[self.kind]})

    def initial_state(self):
        return owned(np.zeros(self.state_size, dtype=np.float32))

    def storage_bytes(self):
        return sum(a.nbytes for a in self.weights.values()) + self.state_size * 4

    def step(self, features, state):
        require(isinstance(features, np.ndarray) and features.dtype == np.float32
                and features.shape == (INPUT_DIM,) and np.isfinite(features).all(), "finite float32[31] features")
        require(isinstance(state, np.ndarray) and state.dtype == np.float32
                and state.shape == (self.state_size,) and np.isfinite(state).all(), "finite declared float32 state")
        w = self.weights
        if self.kind == "gru32":
            x = w["recurrent.weight_ih_l0"] @ features + w["recurrent.bias_ih_l0"]
            h = w["recurrent.weight_hh_l0"] @ state + w["recurrent.bias_hh_l0"]
            reset = sigmoid(x[:32] + h[:32])
            update = sigmoid(x[32:64] + h[32:64])
            candidate = np.tanh(x[64:] + reset * h[64:])
            hidden = (np.float32(1) - update) * candidate + update * state
            next_state = owned(hidden)
        else:
            hidden = np.tanh(w["hidden.weight"] @ features + w["hidden.bias"])
            next_state = owned(state)
        logit = np.asarray((w["output.weight"] @ hidden + w["output.bias"])[0], dtype=np.float32)
        probability = float(sigmoid(logit))
        require(np.isfinite(hidden).all() and np.isfinite(logit) and math.isfinite(probability), "finite deployed output")
        return probability >= THRESHOLD, next_state, float(logit), probability

    def __call__(self, features, state):
        query, next_state, _, _ = self.step(features, state)
        return query, next_state
