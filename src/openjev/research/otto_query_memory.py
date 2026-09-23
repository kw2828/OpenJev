"""Causal query-written, history-conditioned score correction.

This standalone engineering kernel contains no learned weights, model calls or
hidden episode state. Its normalized delta write optimizes the same residual
that its current cue reads. A history feature is not an eligibility gradient or
evidence of temporal credit assignment. ``trace_scrambled`` rotates the PAST
trace before mixing it with the current key: rotation preserves the past trace
norm, not the mixed cue or its angle. It is a representation perturbation, not
a temporal-order shuffle.

The caller supplies causal keys and a complete pre-assimilation shadow forecast
using the same frozen readout as base_action. In particular, a shadow forecast
that omits an already-added static action residual is not interchangeable.
Neither semantic property can be established from tensor shapes alone.

All active query outputs and zero padding are copied bitwise from base_action;
the caller must supply identical query_scores on those active queries. Only
later queries can write. Targets/base predictions are detached. Gradients may
flow through keys, traces and writes within a call; input carry must be explicitly
detached between calls. This makes truncation a visible caller decision.

Work counters count executed vector operations and coordinate terms, not
hardware FLOPs, elapsed time, or nonzero changes. A zero step size still executes
a write; no_write suppresses writes and their error calculation. The practical
none baseline performs no memory arithmetic. No compute-match claim is made.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

VERSION = "otto-query-memory-v1"
MODES = ("none", "last_error", "instant_delta", "trace_delta", "trace_additive", "trace_scrambled")
MATRIX_MODES = MODES[2:]
SCORE_DIM, SCALE = 4, 64.0
COUNTERS = ("active_steps", "query_steps", "key_steps", "eligible_write_steps",
            "key_normalizations", "cue_normalizations", "trace_updates", "past_trace_rotations",
            "matrix_decays", "matrix_reads", "matrix_writes", "innovation_calculations",
            "last_error_decays", "last_error_reads", "last_error_writes", "action_corrections")


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Config:
    mode: str = "none"
    key_dim: int = 2
    step_size: float = .25
    trace_decay: float = .75
    memory_decay: float = 1.0
    last_error_decay: float = .75
    epsilon: float = 1e-6

    def __post_init__(self):
        require(type(self.mode) is str and self.mode in MODES, "declared memory mode")
        require(type(self.key_dim) is int and self.key_dim >= 2, "integer key_dim >= 2")
        for name in ("step_size", "trace_decay", "memory_decay", "last_error_decay", "epsilon"):
            value = getattr(self, name)
            require(type(value) in (int, float) and math.isfinite(value), "finite scalar " + name)
            require(value > 0 if name == "epsilon" else 0 <= value <= 1, "bounded scalar " + name)
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class Carry:
    config: Config
    matrix: torch.Tensor
    trace: torch.Tensor
    last_error: torch.Tensor
    absolute_step: torch.Tensor
    has_query: torch.Tensor
    ended: torch.Tensor


@dataclass(frozen=True)
class Forecast:
    """Actions in raw score units; prewrite_correction in score/64 units.

    The correction is exposed at every key step, including queries before their
    writes, and is positive zero at first queries and padding. It never includes
    the current query target. The none mode exposes an all-zero correction.
    """

    action_prediction: torch.Tensor
    prewrite_correction: torch.Tensor
    carry: Carry
    work: dict[str, int]
    work_units: dict[str, int]


def detach_carry(carry: Carry) -> Carry:
    require(isinstance(carry, Carry), "memory Carry")
    return Carry(carry.config, *(getattr(carry, name).detach().clone() for name in
                 ("matrix", "trace", "last_error", "absolute_step", "has_query", "ended")))


def centered(value):
    return value - value.mean(dim=-1, keepdim=True)


def normalized(value, epsilon):
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    require(bool(torch.isfinite(norm).all()), "finite key/cue norm")
    result = value / norm.clamp_min(epsilon)
    require(bool(torch.isfinite(result).all()), "finite normalized key/cue")
    return result


def tensor(value, dtype, shape, name):
    require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.dtype == dtype
            and value.layout == torch.strided and tuple(value.shape) == tuple(shape), name)


def units(work, dimension):
    """Dimensions of executed vector operations; normalization includes its norm."""
    return {
        "normalized_coordinates": dimension * (work["key_normalizations"] + work["cue_normalizations"]),
        "trace_mixed_coordinates": dimension * work["trace_updates"],
        "past_trace_permuted_coordinates": dimension * work["past_trace_rotations"],
        "matrix_decayed_coordinates": SCORE_DIM * dimension * work["matrix_decays"],
        "matrix_read_terms": SCORE_DIM * dimension * work["matrix_reads"],
        "matrix_write_terms": SCORE_DIM * dimension * work["matrix_writes"],
        "innovation_coordinates": SCORE_DIM * work["innovation_calculations"],
        "last_error_decayed_coordinates": SCORE_DIM * work["last_error_decays"],
        "last_error_read_coordinates": SCORE_DIM * work["last_error_reads"],
        "last_error_written_coordinates": SCORE_DIM * work["last_error_writes"],
        "action_corrected_coordinates": SCORE_DIM * work["action_corrections"],
    }


class QueryMemory(torch.nn.Module):
    """Pure CPU float32 state transition with explicit, independently owned carry."""

    def __init__(self, config: Config):
        super().__init__()
        require(isinstance(config, Config), "memory Config")
        self.config = config

    def initial_carry(self, batch: int) -> Carry:
        require(type(batch) is int and batch > 0, "positive integer batch")
        return Carry(self.config, torch.zeros(batch, SCORE_DIM, self.config.key_dim, dtype=torch.float32),
                     torch.zeros(batch, self.config.key_dim, dtype=torch.float32),
                     torch.zeros(batch, SCORE_DIM, dtype=torch.float32),
                     torch.zeros(batch, dtype=torch.int64), torch.zeros(batch, dtype=torch.bool),
                     torch.zeros(batch, dtype=torch.bool))

    def _carry(self, carry, batch):
        require(isinstance(carry, Carry) and carry.config == self.config, "matching memory carry configuration")
        for name, dtype, shape in (
            ("matrix", torch.float32, (batch, SCORE_DIM, self.config.key_dim)),
            ("trace", torch.float32, (batch, self.config.key_dim)),
            ("last_error", torch.float32, (batch, SCORE_DIM)),
            ("absolute_step", torch.int64, (batch,)), ("has_query", torch.bool, (batch,)),
            ("ended", torch.bool, (batch,)),
        ):
            value = getattr(carry, name)
            tensor(value, dtype, shape, "carry " + name)
            require(not value.requires_grad and value.grad_fn is None, "input carry must be explicitly detached")
            if dtype == torch.float32:
                require(bool(torch.isfinite(value).all()), "finite carry " + name)
        require(bool((carry.absolute_step >= 0).all())
                and torch.equal(carry.has_query, carry.absolute_step > 0), "carry chronology/query identity")
        fresh = ~carry.has_query
        require(not bool(carry.ended[fresh].any()) and all(bool((value[fresh] == 0).all())
                for value in (carry.matrix, carry.trace, carry.last_error)), "pristine unstarted carry")

    def forward(self, base_action, shadow_prior, keys, query_scores, query_mask, prior_mask,
                key_mask, lengths, episode_ends, *, carry=None, no_write=False):
        """Process arbitrary public query schedules in strict chronological order.

        All score tensors are float32[B,T,4], keys float32[B,T,D], masks bool[B,T],
        lengths int64[B], ends bool[B]. At absolute step zero, query must be true
        and key/prior false. At every later active step key is true, and prior is
        exactly query. No mask is true in padding. Masked-out numerical operands
        may be poisoned; zero base padding is still part of the output contract.
        """
        require(type(no_write) is bool, "Boolean no_write intervention")
        require(isinstance(base_action, torch.Tensor) and base_action.ndim == 3, "base_action rank")
        batch, span, scores = base_action.shape
        require(batch > 0 and span > 0 and scores == SCORE_DIM, "positive batch/span and four scores")
        for name, value in (("base_action", base_action), ("shadow_prior", shadow_prior), ("query_scores", query_scores)):
            tensor(value, torch.float32, (batch, span, SCORE_DIM), "CPU float32 " + name)
        tensor(keys, torch.float32, (batch, span, self.config.key_dim), "CPU float32 keys")
        for name, value in (("query_mask", query_mask), ("prior_mask", prior_mask), ("key_mask", key_mask)):
            tensor(value, torch.bool, (batch, span), "CPU bool " + name)
        tensor(lengths, torch.int64, (batch,), "CPU int64 lengths")
        tensor(episode_ends, torch.bool, (batch,), "CPU bool episode_ends")
        require(bool(((lengths >= 0) & (lengths <= span)).all()), "lengths within chunk")
        carry = self.initial_carry(batch) if carry is None else carry
        self._carry(carry, batch)
        require(not bool((carry.ended & (lengths > 0)).any()), "ended lanes cannot advance")
        require(not bool((episode_ends & (lengths == 0) & ~carry.has_query).any()), "no empty new episode end")
        active = torch.arange(span)[None, :] < lengths[:, None]
        # Guard int64 overflow before constructing absolute chronology.
        require(bool((carry.absolute_step <= torch.iinfo(torch.int64).max - span).all()), "bounded int64 chronology")
        absolute = carry.absolute_step[:, None] + torch.arange(span)[None, :]
        first = active & (absolute == 0)
        require(not bool((query_mask & ~active).any()) and bool(query_mask[first].all()), "first active step query and no padding query")
        require(torch.equal(key_mask, active & ~first), "keys exactly on later active steps")
        require(torch.equal(prior_mask, query_mask & key_mask), "prior exactly on later actual queries")
        require(bool(torch.isfinite(base_action[active]).all()) and bool((base_action[~active] == 0).all()),
                "finite active base and zero base padding")
        require(bool(torch.isfinite(query_scores[query_mask]).all()), "finite actual query scores")
        require(torch.equal(base_action[query_mask].detach().view(torch.int32),
                            query_scores[query_mask].detach().view(torch.int32)), "bitwise exact base/query contract")
        mode, cfg = self.config.mode, self.config
        if mode in MATRIX_MODES:
            require(bool(torch.isfinite(keys[key_mask]).all()), "finite consumed keys")
        if mode != "none" and not no_write:
            require(bool(torch.isfinite(shadow_prior[prior_mask]).all()), "finite consumed shadow priors")
        matrix, trace, last = (getattr(carry, name).clone() for name in ("matrix", "trace", "last_error"))
        output = [base_action[:, step].detach().clone() for step in range(span)]
        corrections = [torch.zeros(batch, SCORE_DIM, dtype=torch.float32) for _ in range(span)]
        work = {name: 0 for name in COUNTERS}
        work.update(active_steps=int(active.sum()), query_steps=int(query_mask.sum()), key_steps=int(key_mask.sum()),
                    eligible_write_steps=int(prior_mask.sum()))

        def innovation(index, step):
            work["innovation_calculations"] += int(index.numel())
            result = centered((query_scores[index, step].detach() - shadow_prior[index, step].detach()) / SCALE)
            require(bool(torch.isfinite(result).all()), "finite normalized query innovation")
            return result

        for step in range(span):
            index = torch.nonzero(key_mask[:, step], as_tuple=False).flatten()
            if not index.numel() or mode == "none":
                continue
            count = int(index.numel())
            writes = prior_mask[index, step]
            reads = ~query_mask[index, step]
            if mode == "last_error":
                current = last[index] * cfg.last_error_decay
                work["last_error_decays"] += count
                corrections[step] = corrections[step].index_copy(0, index, current)
                work["last_error_reads"] += count
                if bool(reads.any()):
                    selected = index[reads]
                    correction = current[reads]
                    result = output[step][selected] + SCALE * correction
                    # With no writes, a zero state has exactly the baseline's bits.
                    if no_write:
                        result = torch.where((correction == 0).all(-1, keepdim=True), output[step][selected], result)
                    output[step] = output[step].index_copy(0, selected, result)
                    work["action_corrections"] += int(selected.numel())
                if not no_write and bool(writes.any()):
                    positions = torch.nonzero(writes, as_tuple=False).flatten()
                    current = current.index_copy(0, positions, innovation(index[writes], step))
                    work["last_error_writes"] += int(positions.numel())
                last = last.index_copy(0, index, current)
                continue

            key = normalized(keys[index, step], cfg.epsilon)
            work["key_normalizations"] += count
            past = trace[index]
            current_trace = (1 - cfg.trace_decay) * key + cfg.trace_decay * past
            work["trace_updates"] += count
            if mode == "instant_delta":
                cue = key
            else:
                cue = current_trace
                if mode == "trace_scrambled":
                    shift = 1 + absolute[index, step] % (cfg.key_dim - 1)
                    coordinate = (torch.arange(cfg.key_dim)[None, :] - shift[:, None]) % cfg.key_dim
                    rotated = past.gather(1, coordinate)
                    cue = (1 - cfg.trace_decay) * key + cfg.trace_decay * rotated
                    work["past_trace_rotations"] += count
                cue = normalized(cue, cfg.epsilon)
                work["cue_normalizations"] += count
            trace = trace.index_copy(0, index, current_trace)
            before = matrix[index] * cfg.memory_decay
            work["matrix_decays"] += count
            prediction = centered(torch.einsum("bkd,bd->bk", before, cue))
            require(bool(torch.isfinite(prediction).all()), "finite memory read")
            corrections[step] = corrections[step].index_copy(0, index, prediction)
            work["matrix_reads"] += count
            if bool(reads.any()):
                selected = index[reads]
                result = output[step][selected] + SCALE * prediction[reads]
                if no_write:
                    zero = (before[reads] == 0).all(dim=(1, 2))[:, None]
                    result = torch.where(zero, output[step][selected], result)
                output[step] = output[step].index_copy(0, selected, result)
                work["action_corrections"] += int(selected.numel())
            if not no_write and bool(writes.any()):
                positions = torch.nonzero(writes, as_tuple=False).flatten()
                target = innovation(index[writes], step)
                error = target if mode == "trace_additive" else target - prediction[writes]
                selected_cue = cue[writes]
                denominator = cfg.epsilon + selected_cue.square().sum(-1)
                require(bool(torch.isfinite(denominator).all()) and bool((denominator > 0).all()), "finite positive write denominator")
                updated = before[writes] + cfg.step_size * error[:, :, None] * selected_cue[:, None, :] / denominator[:, None, None]
                before = before.index_copy(0, positions, updated)
                work["matrix_writes"] += int(positions.numel())
            matrix = matrix.index_copy(0, index, before)
        action = torch.stack(output, dim=1)
        require(bool(torch.isfinite(action[active]).all()) and all(bool(torch.isfinite(value).all())
                for value in (matrix, trace, last)), "finite memory transition and active corrections")
        next_carry = Carry(cfg, matrix, trace, last, carry.absolute_step.clone() + lengths,
                           carry.has_query.clone() | (lengths > 0), carry.ended.clone() | episode_ends)
        return Forecast(action, torch.stack(corrections, dim=1), next_carry, work, units(work, cfg.key_dim))
