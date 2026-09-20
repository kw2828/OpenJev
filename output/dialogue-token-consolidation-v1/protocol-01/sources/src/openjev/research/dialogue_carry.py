"""A learned categorical carry baseline using only public text embeddings.

Each question replays its own causal prefix. The state is a distribution over
its supplied candidates, not a teacher-forced gold value. A learned operation
distribution contains one write operation per candidate and one CARRY operation:
    p_next = op_carry * p_previous + op_values.
This is an explicit semantic state tracker, not a new memory architecture.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch
from torch import nn


def _boolean(value: torch.Tensor, shape: tuple[int, ...], device: torch.device, name: str) -> None:
    if (not isinstance(value, torch.Tensor) or value.shape != shape
            or value.dtype != torch.bool or value.device != device):
        raise ValueError(f"{name} must be a bool tensor with shape {shape} on {device}")


def _indices(value: torch.Tensor, shape: tuple[int, ...], device: torch.device, name: str) -> None:
    if (not isinstance(value, torch.Tensor) or value.shape != shape
            or value.dtype != torch.long or value.device != device):
        raise ValueError(f"{name} must be an int64 tensor with shape {shape} on {device}")


class LearnedCarry(nn.Module):
    """Candidate-equivariant scorer plus a functional categorical carry state.

    query_times are zero-based inclusive indices in turns; -1 asks for the
    initial state. Invalid turns are skipped, including between valid turns.
    none_index identifies NOT_MENTIONED, defaults to candidate zero, and must
    be permuted with candidates. Candidate padding always has log probability
    -inf. Callers train ordinary state CE on their eligible query records.
    """

    def __init__(self, input_dim: int = 384, projection_dim: int = 96, hidden_dim: int = 64):
        super().__init__()
        if any(type(x) is not int or x < 1 for x in (input_dim, projection_dim, hidden_dim)):
            raise ValueError("Dimensions must be positive builtin integers")
        self.input_dim, self.projection_dim, self.hidden_dim = input_dim, projection_dim, hidden_dim
        self.turn_projection = nn.Linear(input_dim, projection_dim)
        self.query_projection = nn.Linear(input_dim, projection_dim)
        self.candidate_projection = nn.Linear(input_dim, projection_dim)
        self.value_score = nn.Sequential(nn.Linear(6 * projection_dim, hidden_dim), nn.Tanh(),
                                         nn.Linear(hidden_dim, 1))
        self.carry_score = nn.Sequential(nn.Linear(3 * projection_dim, hidden_dim), nn.Tanh(),
                                         nn.Linear(hidden_dim, 1))

    def configuration(self) -> dict:
        return {"class": type(self).__name__, "input_dim": self.input_dim,
                "projection_dim": self.projection_dim, "hidden_dim": self.hidden_dim,
                "query_time": "zero-based inclusive; -1 initial",
                "update": "p_next=operation_carry*p_previous+operation_values",
                "state": "candidate log probabilities; no gold feedback",
                "parameter_count": sum(p.numel() for p in self.parameters())}

    def _float(self, value: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
        parameter = self.turn_projection.weight
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.dtype != parameter.dtype or value.device != parameter.device
                or value.dtype not in (torch.float32, torch.float64)):
            raise ValueError(f"{name} must match model dtype/device and shape {shape}")

    def _schema(self, query: torch.Tensor, candidates: torch.Tensor,
                candidate_mask: torch.Tensor) -> tuple[int, int, int]:
        if not isinstance(query, torch.Tensor) or query.ndim != 3:
            raise ValueError("query must be [B,Q,D]")
        b, q, _ = query.shape
        if (min(b, q) < 1 or not isinstance(candidates, torch.Tensor)
                or candidates.ndim != 4 or candidates.shape[2] < 1):
            raise ValueError("Nonempty candidates must be [B,Q,C,D]")
        c = candidates.shape[2]
        self._float(query, (b, q, self.input_dim), "query")
        self._float(candidates, (b, q, c, self.input_dim), "candidates")
        _boolean(candidate_mask, (b, q, c), query.device, "candidate_mask")
        if (not candidate_mask.any(-1).all() or not torch.isfinite(query).all()
                or not torch.isfinite(candidates[candidate_mask]).all()):
            raise ValueError("Queries/valid candidates must be finite with at least one candidate per query")
        return b, q, c

    def initial(self, candidate_mask: torch.Tensor, none_index: torch.Tensor | None = None) -> torch.Tensor:
        """Return independently owned [B,Q,C] one-hot NOT_MENTIONED log state."""
        reference = self.turn_projection.weight
        if (not isinstance(candidate_mask, torch.Tensor) or candidate_mask.ndim != 3
                or min(candidate_mask.shape) < 1):
            raise ValueError("candidate_mask must be nonempty [B,Q,C]")
        _boolean(candidate_mask, tuple(candidate_mask.shape), reference.device, "candidate_mask")
        b, q, c = candidate_mask.shape
        if none_index is None:
            none_index = torch.zeros((b, q), dtype=torch.long, device=reference.device)
        _indices(none_index, (b, q), reference.device, "none_index")
        if (not ((none_index >= 0) & (none_index < c)).all()
                or not candidate_mask.gather(-1, none_index[..., None]).all()):
            raise ValueError("NOT_MENTIONED must identify an unmasked candidate")
        result = reference.new_full((b, q, c), -torch.inf)
        return result.scatter(-1, none_index[..., None], 0.0)

    def _project_schema(self, query: torch.Tensor, candidates: torch.Tensor,
                        mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Sanitize arbitrary padded payloads before projection, including NaNs.
        clean = torch.where(mask[..., None], candidates, 0.0)
        return self.query_projection(query).tanh(), self.candidate_projection(clean).tanh()

    def _update(self, state: torch.Tensor, turn: torch.Tensor, query: torch.Tensor,
                candidates: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """All rows here are active, flattened queries; schema is projected."""
        u = self.turn_projection(turn).tanh()
        carry = self.carry_score(torch.cat((u, query, u * query), -1))
        u = u[:, None, :].expand_as(candidates)
        q = query[:, None, :].expand_as(candidates)
        scores = self.value_score(torch.cat((u, q, candidates, u * q, u * candidates,
                                            q * candidates), -1)).squeeze(-1)
        scores = scores.masked_fill(~mask, -torch.inf)
        operation = torch.log_softmax(torch.cat((scores, carry), -1), -1)
        # Avoid undefined derivatives of logaddexp(-inf,-inf) on padding.
        carried = (state + operation[:, -1:]).masked_fill(~mask, 0.0)
        written = operation[:, :-1].masked_fill(~mask, 0.0)
        return torch.logaddexp(carried, written).masked_fill(~mask, -torch.inf)

    def step(self, state: torch.Tensor, turn: torch.Tensor, query: torch.Tensor,
             candidates: torch.Tensor, candidate_mask: torch.Tensor,
             active: torch.Tensor) -> torch.Tensor:
        """Streaming one-turn update; inactive rows and all inputs are untouched.

        The caller keeps the same schema and state across turns, even when a
        slot has no scored label at that turn. This API reprojects its supplied
        schema on each call; forward caches these projections for prefix replay.
        """
        b, q, c = self._schema(query, candidates, candidate_mask)
        self._float(state, (b, q, c), "state")
        self._float(turn, (b, q, self.input_dim), "turn")
        _boolean(active, (b, q), query.device, "active")
        if (torch.isnan(state).any() or torch.isposinf(state).any()
                or not torch.isneginf(state[~candidate_mask]).all()
                or not torch.allclose(torch.logsumexp(state, -1), torch.zeros_like(state[..., 0]),
                                      atol=1e-5, rtol=0)
                or not torch.isfinite(turn[active]).all()):
            raise ValueError("State must be normalized/masked and active turns finite")
        qp, cp = self._project_schema(query, candidates, candidate_mask)
        ids = active.flatten().nonzero().flatten()
        out = state.reshape(b * q, c).clone()
        if ids.numel():
            updated = self._update(out[ids], turn.reshape(b * q, -1)[ids],
                                   qp.reshape(b * q, -1)[ids], cp.reshape(b * q, c, -1)[ids],
                                   candidate_mask.reshape(b * q, c)[ids])
            out = out.index_copy(0, ids, updated)
        return out.reshape(b, q, c)

    def forward(self, turns: torch.Tensor, valid: torch.Tensor, query: torch.Tensor,
                candidates: torch.Tensor, query_times: torch.Tensor,
                candidate_mask: torch.Tensor, none_index: torch.Tensor | None = None) -> torch.Tensor:
        """Predict each question after its full valid prefix, without gold state."""
        b, q, c = self._schema(query, candidates, candidate_mask)
        if not isinstance(turns, torch.Tensor) or turns.ndim != 3:
            raise ValueError("turns must be [B,T,D]")
        t = turns.shape[1]
        self._float(turns, (b, t, self.input_dim), "turns")
        _boolean(valid, (b, t), query.device, "valid")
        _indices(query_times, (b, q), query.device, "query_times")
        if not ((query_times >= -1) & (query_times < t)).all():
            raise ValueError("query_times must be -1 or a zero-based index below T")
        qp, cp = self._project_schema(query, candidates, candidate_mask)
        qp, cp = qp.reshape(b * q, -1), cp.reshape(b * q, c, -1)
        mask = candidate_mask.reshape(b * q, c)
        state = self.initial(candidate_mask, none_index).reshape(b * q, c)
        for index in range(int(query_times.max().item()) + 1):
            active = valid[:, index, None] & (query_times >= index)
            ids = active.flatten().nonzero().flatten()
            if not ids.numel():
                continue
            turn = turns[:, index, None, :].expand(b, q, self.input_dim).reshape(b * q, -1)[ids]
            if not torch.isfinite(turn).all():
                raise ValueError("Consumed turn embeddings must be finite")
            state = state.index_copy(0, ids, self._update(state[ids], turn, qp[ids], cp[ids], mask[ids]))
        return state.reshape(b, q, c)


def work_counts(valid: torch.Tensor, query_times: torch.Tensor, candidate_mask: torch.Tensor) -> dict:
    """Count actual successful forward work, including replay and padded scores.

    Counts are scalar input rows to linear stacks, not FLOPs or measured time.
    Schema projections are cached within forward only. This excludes the frozen
    text encoder, validation/copies, autograd and optimizer, which callers time.
    """
    if (not isinstance(valid, torch.Tensor) or valid.ndim != 2
            or not isinstance(query_times, torch.Tensor) or query_times.ndim != 2
            or not isinstance(candidate_mask, torch.Tensor) or candidate_mask.ndim != 3):
        raise ValueError("Expected valid[B,T], query_times[B,Q], candidate_mask[B,Q,C]")
    b, t = valid.shape
    q, c = query_times.shape[1], candidate_mask.shape[2]
    if min(b, q, c) < 1:
        raise ValueError("B,Q,C must be positive")
    _boolean(valid, (b, t), valid.device, "valid")
    _indices(query_times, (b, q), valid.device, "query_times")
    _boolean(candidate_mask, (b, q, c), valid.device, "candidate_mask")
    if (not ((query_times >= -1) & (query_times < t)).all()
            or not candidate_mask.any(-1).all()):
        raise ValueError("Invalid query times or empty candidate sets")
    active = valid[:, None, :] & (torch.arange(t, device=valid.device)[None, None, :] <= query_times[..., None])
    updates = int(active.sum().item())
    valid_scores = int((active.sum(-1) * candidate_mask.sum(-1)).sum().item())
    return {"query_projection_rows": b * q, "candidate_projection_rows": b * q * c,
            "turn_projection_rows": updates, "carry_scorer_rows": updates,
            "value_scorer_rows": updates * c, "valid_candidate_scores": valid_scores,
            "padded_candidate_scores": updates * c - valid_scores,
            "recurrent_query_updates": updates, "output_queries": b * q,
            "state_scalars": b * q * c,
            "scope": "forward stack rows; includes separate prefix replay per query"}


def literal_mention_carry(user_turns: Sequence[str], candidate_values: Sequence[str], *,
                          none_index: int = 0, excluded_indices: Sequence[int] = ()) -> list[int]:
    """After each USER turn, carry or select its longest uniquely mentioned value.

    Match casefolded literal strings at Unicode word boundaries. A tie between
    candidate identities carries the old value. NOT_MENTIONED is never matched.
    This deliberately has no negation, pronoun, yes/no or schema parser: e.g.
    'not Italian' still mentions Italian. It cannot consume annotation metadata.
    """
    if (isinstance(user_turns, (str, bytes)) or isinstance(candidate_values, (str, bytes))
            or not isinstance(user_turns, Sequence) or not isinstance(candidate_values, Sequence)
            or not candidate_values or any(type(x) is not str for x in user_turns)
            or any(type(x) is not str or not x.strip() for x in candidate_values)):
        raise ValueError("Supply sequences of user strings and nonempty candidate strings")
    size = len(candidate_values)
    if type(none_index) is not int or not 0 <= none_index < size:
        raise ValueError("Invalid none_index")
    if any(type(x) is not int or not 0 <= x < size for x in excluded_indices):
        raise ValueError("Invalid excluded_indices")
    excluded = set(excluded_indices) | {none_index}
    values = [value.strip().casefold() for value in candidate_values]
    patterns = {i: re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)")
                for i, value in enumerate(values) if i not in excluded}
    current, result = none_index, []
    for utterance in user_turns:
        text = utterance.casefold()
        matches = [i for i, pattern in patterns.items() if pattern.search(text)]
        if matches:
            longest = max(len(values[i]) for i in matches)
            winners = [i for i in matches if len(values[i]) == longest]
            if len(winners) == 1:
                current = winners[0]
        result.append(current)
    return result
