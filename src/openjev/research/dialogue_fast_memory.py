"""Query-independent streamed dialogue memory with separate candidate reads.

Dense Kalman/Joseph equations follow Kalman associative memory, section 3 of
https://arxiv.org/html/2609.07816v1, not its diagonal scan approximation. Each
value coordinate shares one key covariance and scalar observation variance.
Learned latent residuals include representation error and bias; their normalized
magnitude is not calibrated semantic confidence. Innovation-dependent covariance
inflation is a heuristic, not exact Bayesian conditioning. No feedback detaches.

States are caller-owned tensors; no external cache, RNG seed, data, labels, or
schema metadata is retained. An update accepts only a turn and its validity.
All reads are after the indexed turn. Padding contributes no memory update.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

METHODS = ("current", "gru", "attention", "gated_delta", "kalman", "innovation_kalman")
KALMAN = ("kalman", "innovation_kalman")
State = dict[str, torch.Tensor]


def state_bytes(state: State) -> int:
    """Actual retained tensor payload, including attention padding/mask.

    Excludes parameters, autograd graphs, temporary arrays and Python overhead.
    """
    return sum(value.numel() * value.element_size() for value in state.values())


class DialogueMemory(nn.Module):
    """Common feature/readout modules and six explicit memory mechanisms.

    Gated delta and both Kalman variants have identical parameter names/shapes;
    paired initialization requires explicitly copying their state dictionaries.
    Matrix-memory keys and shared readout keys are unit normalized (zero stays
    zero). Attention instead uses raw projected queries/keys with the standard
    1/sqrt(width) score scale and a learned nonnegative recency penalty per real
    turn (initially .1); its shared readout still uses normalized keys.
    Values are linear projections. Gates start at sigmoid(3)/sigmoid(0).
    """

    def __init__(self, method: str, input_dim: int = 384, width: int = 16,
                 gru_width: int = 128, innovation_eta: float = .1):
        super().__init__()
        if method not in METHODS:
            raise ValueError("Unknown memory method")
        if any(type(x) is not int or x < 1 for x in (input_dim, width, gru_width)):
            raise ValueError("Dimensions must be positive integers")
        if isinstance(innovation_eta, bool) or not math.isfinite(innovation_eta) or innovation_eta < 0:
            raise ValueError("innovation_eta must be finite and nonnegative")
        self.method, self.input_dim, self.width, self.gru_width = method, input_dim, width, gru_width
        self.innovation_eta = float(innovation_eta)
        self.turn_features = nn.Sequential(nn.Linear(input_dim, 96), nn.Tanh(),
                                           nn.Linear(96, 96), nn.Tanh())
        self.key = nn.Linear(96, width)
        self.value = nn.Linear(96, width)
        self.gates = nn.Linear(96, 2)
        nn.init.zeros_(self.gates.weight)
        with torch.no_grad():
            self.gates.bias.copy_(torch.tensor([3., 0.]))
        self.query = nn.Linear(2 * input_dim, width)
        self.readout = nn.Sequential(nn.Linear(3 * width, 96), nn.Tanh(), nn.Linear(96, 1))
        if method == "attention":
            self.log_decay = nn.Parameter(torch.tensor(math.log(math.expm1(.1))))
        if method == "gru":
            self.gru = nn.GRUCell(96, gru_width)
            self.gru_value = nn.Linear(gru_width, width)

    def _tensor(self, value, shape, name):
        ref = self.query.weight
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.dtype not in (torch.float32, torch.float64)
                or value.dtype != ref.dtype or value.device != ref.device
                or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid {name} shape, dtype, device or finite values")

    @staticmethod
    def _mask(value, shape, device, name):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.dtype != torch.bool or value.device != device):
            raise ValueError(f"Invalid {name}")

    def initial_state(self, batch_size: int, *, device=None, dtype=None) -> State:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        ref = self.query.weight
        if ((device is not None and torch.device(device) != ref.device)
                or (dtype is not None and dtype != ref.dtype)):
            raise ValueError("State must match model device and dtype")
        if self.method == "current":
            return {"value": ref.new_zeros(batch_size, self.width)}
        if self.method == "gru":
            return {"hidden": ref.new_zeros(batch_size, self.gru_width)}
        if self.method == "attention":
            return {"keys": ref.new_zeros(batch_size, 0, self.width),
                    "values": ref.new_zeros(batch_size, 0, self.width),
                    "valid": torch.zeros(batch_size, 0, dtype=torch.bool, device=ref.device)}
        state = {"S": ref.new_zeros(batch_size, self.width, self.width)}
        if self.method in KALMAN:
            state["P"] = torch.eye(self.width, device=ref.device, dtype=ref.dtype).expand(
                batch_size, -1, -1).clone()
        return state

    def _state(self, state: State, batch: int):
        if type(state) is not dict:
            raise ValueError("State must be a tensor dictionary")
        if self.method == "current":
            expected = {"value": (batch, self.width)}
        elif self.method == "gru":
            expected = {"hidden": (batch, self.gru_width)}
        elif self.method == "attention":
            if ("keys" not in state or not isinstance(state["keys"], torch.Tensor)
                    or state["keys"].ndim != 3):
                raise ValueError("Invalid attention keys")
            steps = state["keys"].shape[1]
            expected = {"keys": (batch, steps, self.width),
                        "values": (batch, steps, self.width), "valid": (batch, steps)}
        else:
            expected = {"S": (batch, self.width, self.width)}
            if self.method in KALMAN:
                expected["P"] = (batch, self.width, self.width)
        if set(state) != set(expected):
            raise ValueError("State keys do not match memory method")
        for name, shape in expected.items():
            if name == "valid":
                self._mask(state[name], shape, self.query.weight.device, "attention valid")
            else:
                self._tensor(state[name], shape, name)

    def update(self, turn: torch.Tensor, valid: torch.Tensor, state: State) -> tuple[State, dict]:
        """Functional real-turn update; questions/candidates cannot enter this API."""
        if not isinstance(turn, torch.Tensor) or turn.ndim != 2 or turn.shape[0] < 1:
            raise ValueError("turn must be [B,D]")
        batch = turn.shape[0]
        self._tensor(turn, (batch, self.input_dim), "turn")
        self._mask(valid, (batch,), turn.device, "turn validity")
        self._state(state, batch)
        feature = self.turn_features(torch.where(valid[:, None], turn, torch.zeros_like(turn)))
        projected_key = self.key(feature)
        key = projected_key if self.method == "attention" else F.normalize(projected_key, dim=-1)
        value = self.value(feature)
        alpha, beta = self.gates(feature).sigmoid().unbind(-1)
        diagnostics = {"valid": valid.clone(), "alpha": alpha, "beta": beta}
        if self.method == "current":
            proposed = {"value": value}
        elif self.method == "gru":
            proposed = {"hidden": self.gru(feature, state["hidden"])}
        elif self.method == "attention":
            proposed = {"keys": torch.cat((state["keys"], key[:, None]), dim=1),
                        "values": torch.cat((state["values"], value[:, None]), dim=1),
                        "valid": torch.cat((state["valid"], valid[:, None]), dim=1)}
            return proposed, diagnostics
        else:
            predicted = alpha[:, None, None] * state["S"]
            residual = value - torch.einsum("bkv,bk->bv", predicted, key)
            if self.method == "gated_delta":
                gain = beta[:, None] * key
                proposed = {"S": predicted + gain[:, :, None] * residual[:, None, :]}
            else:
                identity = torch.eye(self.width, dtype=turn.dtype, device=turn.device)
                prior = alpha[:, None, None].square() * state["P"] + .01 * identity
                noise = (1 - beta) / beta + .01
                if not torch.isfinite(noise).all():
                    raise ValueError("Nonfinite observation noise from saturated beta")
                pk = torch.einsum("bij,bj->bi", prior, key)
                base_variance = noise + (key * pk).sum(-1)
                residual_mse = residual.square().mean(-1)
                ratio = residual_mse / base_variance
                inflation = (self.innovation_eta * (ratio - 1).clamp(0, 10)
                             if self.method == "innovation_kalman" else torch.zeros_like(ratio))
                prior = prior + inflation[:, None, None] * key[:, :, None] * key[:, None, :]
                pk = torch.einsum("bij,bj->bi", prior, key)
                variance = noise + (key * pk).sum(-1)
                gain = pk / variance[:, None]
                A = identity - gain[:, :, None] * key[:, None, :]
                covariance = (A @ prior @ A.transpose(-1, -2)
                              + noise[:, None, None] * gain[:, :, None] * gain[:, None, :])
                proposed = {"S": predicted + gain[:, :, None] * residual[:, None, :],
                            "P": (covariance + covariance.transpose(-1, -2)) * .5}
                diagnostics.update(observation_noise=noise, predicted_variance=base_variance,
                                   residual_mse=residual_mse, normalized_innovation=ratio,
                                   inflation=inflation, gain=gain)
        result = {name: torch.where(valid.reshape(batch, *([1] * (value.ndim - 1))),
                                    value, state[name]) for name, value in proposed.items()}
        if any(not torch.isfinite(value).all() for value in result.values()):
            raise ValueError("Nonfinite memory update")
        return result, diagnostics

    def read(self, state: State, query_emb: torch.Tensor, candidate_emb: torch.Tensor,
             candidate_mask: torch.Tensor) -> torch.Tensor:
        """Read-only candidate scores. Masked candidates receive negative infinity."""
        if not isinstance(query_emb, torch.Tensor) or query_emb.ndim != 3:
            raise ValueError("query_emb must be [B,Q,D]")
        batch, queries, _ = query_emb.shape
        if not isinstance(candidate_emb, torch.Tensor) or candidate_emb.ndim != 4:
            raise ValueError("candidate_emb must be [B,Q,C,D]")
        candidates = candidate_emb.shape[2]
        if min(batch, queries, candidates) < 1:
            raise ValueError("Empty read batch, query or candidate dimension")
        self._tensor(query_emb, (batch, queries, self.input_dim), "query embeddings")
        self._tensor(candidate_emb, (batch, queries, candidates, self.input_dim), "candidate embeddings")
        self._mask(candidate_mask, (batch, queries, candidates), query_emb.device, "candidate mask")
        if not candidate_mask.any(-1).all():
            raise ValueError("Each query needs a valid candidate")
        self._state(state, batch)
        expanded = query_emb[:, :, None].expand(-1, -1, candidates, -1)
        projected_query = self.query(torch.cat((expanded, candidate_emb), dim=-1))
        key = F.normalize(projected_query, dim=-1)
        if self.method == "current":
            recalled = state["value"][:, None, None].expand_as(key)
        elif self.method == "gru":
            recalled = self.gru_value(state["hidden"])[:, None, None].expand_as(key)
        elif self.method == "attention":
            scores = torch.einsum("bqck,btk->bqct", projected_query, state["keys"]) / math.sqrt(self.width)
            if scores.shape[-1] == 0:
                recalled = torch.zeros_like(key)
            else:
                # Age counts only actual valid turns; padding never ages memory.
                age = state["valid"].sum(-1, keepdim=True) - state["valid"].cumsum(-1)
                scores = scores - F.softplus(self.log_decay) * age[:, None, None].to(scores.dtype)
                mask = state["valid"][:, None, None]
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
                weights = (scores - scores.max(-1, keepdim=True).values).exp() * mask
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(scores.dtype).tiny)
                recalled = torch.einsum("bqct,btv->bqcv", weights, state["values"])
        else:
            recalled = torch.einsum("bkv,bqck->bqcv", state["S"], key)
        logits = self.readout(torch.cat((key, recalled, key * recalled), dim=-1)).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite candidate scores")
        return logits.masked_fill(~candidate_mask, -torch.inf)

    def forward(self, turns, valid, query_emb, candidate_emb, query_times, candidate_mask):
        """Return [B,Q,C] scores using only each selected real-turn prefix."""
        if not isinstance(turns, torch.Tensor) or turns.ndim != 3:
            raise ValueError("turns must be [B,T,D]")
        batch, steps, _ = turns.shape
        if min(batch, steps) < 1:
            raise ValueError("Empty turn sequence")
        self._tensor(turns, (batch, steps, self.input_dim), "turn sequence")
        self._mask(valid, (batch, steps), turns.device, "sequence validity")
        if not isinstance(query_emb, torch.Tensor) or query_emb.ndim != 3:
            raise ValueError("query_emb must be [B,Q,D]")
        queries = query_emb.shape[1]
        self._tensor(query_emb, (batch, queries, self.input_dim), "query embeddings")
        if (not isinstance(query_times, torch.Tensor) or query_times.shape != (batch, queries)
                or query_times.dtype != torch.int64 or query_times.device != turns.device
                or (query_times < 0).any() or (query_times >= steps).any()
                or not valid.gather(1, query_times).all()):
            raise ValueError("query_times must index valid real turns")
        if not isinstance(candidate_emb, torch.Tensor) or candidate_emb.ndim != 4:
            raise ValueError("candidate_emb must be [B,Q,C,D]")
        candidates = candidate_emb.shape[2]
        self._tensor(candidate_emb, (batch, queries, candidates, self.input_dim), "candidate embeddings")
        self._mask(candidate_mask, (batch, queries, candidates), turns.device, "candidate mask")
        if min(queries, candidates) < 1 or not candidate_mask.any(-1).all():
            raise ValueError("Each query needs a valid candidate")
        result = turns.new_zeros(batch, queries, candidates)
        state = self.initial_state(batch)
        for t in range(steps):
            state, _ = self.update(turns[:, t], valid[:, t], state)
            rows, slots = (query_times == t).nonzero(as_tuple=True)
            if rows.numel():
                selected = {name: value.index_select(0, rows) for name, value in state.items()}
                result[rows, slots] = self.read(selected, query_emb[rows, slots][:, None],
                                               candidate_emb[rows, slots][:, None],
                                               candidate_mask[rows, slots][:, None])[:, 0]
        return result
