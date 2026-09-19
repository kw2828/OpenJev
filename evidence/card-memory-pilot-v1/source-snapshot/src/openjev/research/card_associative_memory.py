"""Learned-key associative memories for public card observations.

This is a functional sequence component, not an environment or fitted model.
The innovation rules inflate diagonal residual scales. They are not exact
Bayesian posteriors and do not establish uncertainty calibration. Local and
matched inflation agree along the current key only for the SAME input state,
event and shared parameters; their posterior state and future reads can differ.

Positions and ranks are public event indices. Padding never changes state.
Read-before-write ordering is controlled explicitly by the caller. No RNG seed,
training loop, hidden deck or environment is accepted by this module.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

MODES = ("delta", "gated_delta", "kalman", "innovation_local", "innovation_matched", "gru")
KALMAN_MODES = ("kalman", "innovation_local", "innovation_matched")
INNOVATION_MODES = ("innovation_local", "innovation_matched")
POSITIONS, RANKS, KEY_DIM, VALUE_DIM = 52, 13, 32, 16
STATE_DIM = KEY_DIM * VALUE_DIM
State = dict[str, torch.Tensor | None]


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def clone_state(state: State) -> State:
    """Copy storage while preserving its autograd history."""
    if set(state) != {"S", "P"} or not isinstance(state["S"], torch.Tensor):
        raise ValueError("State must contain S tensor and optional P tensor")
    if state["P"] is not None and not isinstance(state["P"], torch.Tensor):
        raise ValueError("P must be a tensor or None")
    return {name: None if value is None else value.clone() for name, value in state.items()}


def detach_state(state: State) -> State:
    """Detach graph and copy storage; subsequent mutation cannot affect input."""
    return {name: None if value is None else value.detach() for name, value in clone_state(state).items()}


class CardAssociativeMemory(nn.Module):
    """One of six updates sharing learned key/value/query/readout definitions.

    Gate outputs are alpha,beta; noise outputs are r,omega. Initial event-gate
    weights are zero with alpha=.99 and beta=.5. Initial noise weights are zero
    with r=.1 and omega=.001, including their softplus floors. Values and keys
    are learned embeddings. Only keys/queries are L2 normalized.
    """

    def __init__(self, mode: str):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown memory mode: {mode}")
        self.mode = mode
        # Construct shared modules first. Pairing across modes is explicit via
        # copy_shared_initialization, not a claim about implicit RNG ordering.
        self.key_embedding = nn.Embedding(POSITIONS, KEY_DIM)
        self.value_embedding = nn.Embedding(RANKS, VALUE_DIM)
        self.decoder = nn.Linear(VALUE_DIM, RANKS)
        if mode == "gru":
            self.gru = nn.GRUCell(KEY_DIM + VALUE_DIM, STATE_DIM)
        else:
            self.gate = nn.Linear(KEY_DIM + VALUE_DIM, 2)
            nn.init.zeros_(self.gate.weight)
            with torch.no_grad():
                self.gate.bias.copy_(torch.tensor([math.log(.99 / .01), 0.0]))
        if mode in KALMAN_MODES:
            self.noise = nn.Linear(KEY_DIM + VALUE_DIM, 2)
            nn.init.zeros_(self.noise.weight)
            with torch.no_grad():
                self.noise.bias.copy_(torch.tensor([_inverse_softplus(.1 - 1e-4),
                                                    _inverse_softplus(.001 - 1e-5)]))
        if mode in INNOVATION_MODES:
            self.innovation_scale = nn.Parameter(torch.tensor(_inverse_softplus(.1)))

    def init_state(self, batch: int, device: torch.device | str | None = None) -> State:
        if type(batch) is not int or batch < 1:
            raise ValueError("batch must be a positive integer")
        parameter = self.key_embedding.weight
        target = parameter.device if device is None else torch.device(device)
        if target != parameter.device:
            raise ValueError("State device must match model device; move the model first")
        return {"S": parameter.new_zeros((batch, KEY_DIM, VALUE_DIM)),
                "P": parameter.new_ones((batch, KEY_DIM)) if self.mode in KALMAN_MODES else None}

    def _state(self, state: State) -> tuple[torch.Tensor, torch.Tensor | None]:
        if type(state) is not dict or set(state) != {"S", "P"}:
            raise ValueError("State must have exactly S and P")
        matrix, variance = state["S"], state["P"]
        reference = self.key_embedding.weight
        if (not isinstance(matrix, torch.Tensor) or matrix.ndim != 3
                or matrix.shape[0] < 1 or matrix.shape[1:] != (KEY_DIM, VALUE_DIM)
                or matrix.dtype != reference.dtype or matrix.device != reference.device
                or not torch.isfinite(matrix).all()):
            raise ValueError("Invalid S shape, device, dtype or finite values")
        if self.mode in KALMAN_MODES:
            if (not isinstance(variance, torch.Tensor) or variance.shape != matrix.shape[:2]
                    or variance.dtype != reference.dtype or variance.device != reference.device
                    or not torch.isfinite(variance).all() or not (variance > 0).all()):
                raise ValueError("Kalman-family P must be finite and strictly positive")
        elif variance is not None:
            raise ValueError("Non-Kalman state has P=None")
        return matrix, variance

    def _keys(self, positions: torch.Tensor) -> torch.Tensor:
        keys = self.key_embedding(positions)
        norms = torch.linalg.vector_norm(keys, dim=-1, keepdim=True)
        if not torch.isfinite(keys).all() or not (norms > 0).all():
            raise ValueError("Learned key vectors must be finite and nonzero")
        return keys / norms

    @staticmethod
    def _indices(indices: torch.Tensor, shape: tuple, device: torch.device, upper: int, name: str) -> None:
        if (not isinstance(indices, torch.Tensor) or indices.dtype != torch.long
                or indices.device != device or indices.shape != shape
                or not ((indices >= 0) & (indices < upper)).all()):
            raise ValueError(f"Invalid {name} shape, dtype, device or range")

    def predict(self, state: State, queries: torch.Tensor) -> torch.Tensor:
        """Return rank logits [B,Q,13] without modifying state or seeing rank."""
        matrix, _ = self._state(state)
        if not isinstance(queries, torch.Tensor) or queries.ndim != 2:
            raise ValueError("queries must have shape [B,Q]")
        self._indices(queries, (matrix.shape[0], queries.shape[1]), matrix.device, POSITIONS, "queries")
        keys = self._keys(queries)
        return self.decoder(torch.einsum("bkd,bqk->bqd", matrix, keys))

    def write(self, state: State, pos: torch.Tensor, rank: torch.Tensor,
              valid: torch.Tensor) -> tuple[State, dict[str, torch.Tensor]]:
        """Functionally ingest one observed public event per batch member.

        Arbitrary integer sentinels are allowed only in masked pos/rank rows.
        Diagnostics on padding are zero. GRU gain_along_key is unavailable and
        represented by zero with gain_available=False, not a claimed GRU gain.
        added_trace is innovation inflation only, excluding base process noise.
        key_kurtosis is sum(k^4)/sum(k^2)^2, the inverse participation ratio.
        """
        matrix, variance = self._state(state)
        batch = matrix.shape[0]
        if (not isinstance(valid, torch.Tensor) or valid.shape != (batch,)
                or valid.dtype != torch.bool or valid.device != matrix.device):
            raise ValueError("valid must be a bool tensor [B] on the state device")
        for indices, upper, name in ((pos, POSITIONS, "pos"), (rank, RANKS, "rank")):
            if (not isinstance(indices, torch.Tensor) or indices.shape != (batch,)
                    or indices.dtype != torch.long or indices.device != matrix.device):
                raise ValueError(f"Invalid {name} shape, dtype or device")
            self._indices(indices[valid], (int(valid.sum()),), matrix.device, upper, name)
        positions = torch.where(valid, pos, 0)
        ranks = torch.where(valid, rank, 0)
        key, value = self._keys(positions), self.value_embedding(ranks)
        event = torch.cat((key, value), dim=-1)
        key_squared = key.square()
        key_norm_squared = key_squared.sum(dim=-1)
        key_fourth_sum = key_squared.square().sum(dim=-1)
        alpha = beta = None
        if self.mode != "gru":
            alpha, beta = self.gate(event).sigmoid().unbind(dim=-1)
        prior_matrix = matrix if self.mode in ("delta", "gru") else alpha[:, None, None] * matrix
        residual = value - torch.einsum("bkd,bk->bd", prior_matrix, key)
        residual_power = residual.square().mean(dim=-1)
        added_trace = torch.zeros_like(residual_power)
        gain_along_key = torch.zeros_like(residual_power)
        next_variance = None
        if self.mode == "gru":
            next_matrix = self.gru(event, matrix.reshape(batch, STATE_DIM)).reshape(batch, KEY_DIM, VALUE_DIM)
        elif self.mode in ("delta", "gated_delta"):
            gain = beta[:, None] * key
            gain_along_key = (gain * key).sum(dim=-1)
            next_matrix = prior_matrix + gain[:, :, None] * residual[:, None, :]
        else:
            r, omega = (F.softplus(self.noise(event)) + event.new_tensor([1e-4, 1e-5])).unbind(dim=-1)
            prior_variance = alpha[:, None].square() * variance + omega[:, None]
            if self.mode in INNOVATION_MODES:
                c = F.softplus(self.innovation_scale) * residual_power
                if self.mode == "innovation_local":
                    addition = c[:, None] * key_squared
                else:
                    diffuse = c * key_fourth_sum / key_norm_squared
                    addition = diffuse[:, None].expand_as(prior_variance)
                prior_variance = prior_variance + addition
                added_trace = addition.sum(dim=-1)
            denominator = r + (prior_variance * key_squared).sum(dim=-1)
            gain = prior_variance * key / denominator[:, None]
            gain_along_key = (gain * key).sum(dim=-1)
            next_matrix = prior_matrix + gain[:, :, None] * residual[:, None, :]
            next_variance = (prior_variance.reciprocal() + key_squared / r[:, None]).reciprocal()
            next_variance = torch.where(valid[:, None], next_variance, variance)
        next_matrix = torch.where(valid[:, None, None], next_matrix, matrix)
        diagnostics = {
            "gain_along_key": torch.where(valid, gain_along_key, 0),
            "gain_available": valid & (self.mode != "gru"),
            "key_kurtosis": torch.where(valid, key_fourth_sum / key_norm_squared.square(), 0),
            "added_trace": torch.where(valid, added_trace, 0),
            "residual_power": torch.where(valid, residual_power, 0),
        }
        return {"S": next_matrix, "P": next_variance}, diagnostics

    def parameter_counts(self) -> dict[str, int]:
        """Report actual allocation and used scalar coordinates, not matching."""
        registered = sum(parameter.numel() for parameter in self.parameters())
        unused_gate_row = KEY_DIM + VALUE_DIM + 1 if self.mode in ("delta", *KALMAN_MODES) else 0
        return {"registered_parameters": registered, "used_scalar_parameters": registered - unused_gate_row,
                "state_scalars_per_case": STATE_DIM + (KEY_DIM if self.mode in KALMAN_MODES else 0),
                "state_bytes_per_case": (STATE_DIM + (KEY_DIM if self.mode in KALMAN_MODES else 0))
                * self.key_embedding.weight.element_size()}

    def configuration(self) -> dict:
        """Fresh JSON-safe semantics binding, including identical-schema arms.

        Local and matched state_dict keys alone cannot identify their update.
        The caller must preserve this configuration with saved weights.
        """
        rules = {
            "delta": "S_new=S+beta*k*(v-S^T*k)^T",
            "gated_delta": "S_prior=alpha*S; S_new=S_prior+beta*k*(v-S_prior^T*k)^T",
            "kalman": "S_prior=alpha*S; p=alpha^2*P+omega; addition=0",
            "innovation_local": "S_prior=alpha*S; p=alpha^2*P+omega+c*k^2",
            "innovation_matched": "S_prior=alpha*S; p=alpha^2*P+omega+c*sum(k^4)/sum(k^2)",
            "gru": "S_new=reshape(GRUCell(concat(k,v),flatten(S)),32,16)",
        }
        return {
            "version": "card-associative-memory-v1", "model_class": type(self).__name__, "mode": self.mode,
            "positions": POSITIONS, "ranks": RANKS, "key_dim": KEY_DIM, "value_dim": VALUE_DIM,
            "key_query": "shared learned Embedding(52,32), exact L2 normalization of nonzero vectors",
            "value": "learned Embedding(13,16), unnormalized",
            "readout": "shared Linear(16,13) applied to S^T*query",
            "initial_S": "zeros[batch,32,16]",
            "initial_P": "ones[batch,32]" if self.mode in KALMAN_MODES else None,
            "update_rule": rules[self.mode],
            "kalman_update": ("gain=p*k/(r+sum(p*k^2)); S_new=S_prior+gain*(v-S_prior^T*k)^T; "
                              "P_new=1/(1/p+k^2/r)") if self.mode in KALMAN_MODES else None,
            "event_gate": "Linear(concat(k,v))->sigmoid(alpha,beta)" if self.mode != "gru" else None,
            "noise": "r=softplus(noise[0])+1e-4; omega=softplus(noise[1])+1e-5"
            if self.mode in KALMAN_MODES else None,
            "innovation_c": "softplus(innovation_scale)*mean((v-S_prior^T*k)^2)"
            if self.mode in INNOVATION_MODES else None,
            "innovation_residual_gradient": "fully_differentiable" if self.mode in INNOVATION_MODES else None,
            "innovation_matching_scope": "same input state/event/parameters, current-key gain only"
            if self.mode in INNOVATION_MODES else None,
            "masking": "invalid write rows preserve S and P exactly; diagnostics zero",
            "posterior_claim": "diagonal approximation; no exact Bayesian posterior or calibration claim",
            "parameter_counts": self.parameter_counts(),
        }


def copy_shared_initialization(source: CardAssociativeMemory, target: CardAssociativeMemory) -> tuple[str, ...]:
    """Copy common named tensors only; no seed changes or new module creation.

    Arm-specific tensors absent from source keep their own initialization.
    Caller records the resulting full state_dict for each arm. This does not
    claim that GRU and associative update parameter counts are equal.
    """
    if type(source) is not CardAssociativeMemory or type(target) is not CardAssociativeMemory:
        raise TypeError("Expected exact CardAssociativeMemory instances")
    origin, destination = source.state_dict(), target.state_dict()
    shared = tuple(sorted(origin.keys() & destination.keys()))
    for name in shared:
        if origin[name].shape != destination[name].shape or origin[name].dtype != destination[name].dtype:
            raise ValueError(f"Incompatible shared tensor: {name}")
    with torch.no_grad():
        for name in shared:
            destination[name].copy_(origin[name])
    return shared
