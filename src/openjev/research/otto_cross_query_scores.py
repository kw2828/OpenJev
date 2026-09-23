"""Pure chronological predictors with genuine-query score correction.

The caller owns public trajectories, query-only teacher inputs, loss targets,
optimizer scheduling and costs. No skipped teacher score is consumed here.
Chunks start at period-four boundaries; only explicit episode ends may finish
mid-period. Numerical state persists across chunks and queries as specified by
each family. The module itself stores no episode state.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

VERSION = "otto-cross-query-scores-v1"
KINDS = ("innovation", "innovation_gru", "persistent_direct", "reset_direct")
FEATURE_DIM, PERIOD, SCORE_DIM, SCALE, HORIZON = 31, 4, 4, 64.0, 2188
PARAMETERS = {"innovation": 5978, "innovation_gru": 5996,
              "persistent_direct": 5862, "reset_direct": 5862}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parameter_count(kind):
    require(kind in KINDS, "declared cross-query family")
    return PARAMETERS[kind]


@dataclass(frozen=True)
class Carry:
    """Owned state tensors; input state is never mutated by a forward call.

    Returned floating state can retain its autograd graph. Call detach_carry
    before another chunk to enforce the caller's explicit TBPTT boundary.
    The raw anchor is the last actual query vector, not a predicted score.
    Ended lanes retain their final hidden/anchor values and cannot advance.
    """

    kind: str
    hidden: torch.Tensor
    raw_anchor: torch.Tensor
    has_query: torch.Tensor
    absolute_step: torch.Tensor
    ended: torch.Tensor


def detach_carry(carry):
    """Return detached, separately owned copies, without modifying the source."""
    require(isinstance(carry, Carry) and carry.kind in KINDS, "declared carry")
    return Carry(carry.kind, *(value.detach().clone() for value in (
        carry.hidden, carry.raw_anchor, carry.has_query, carry.absolute_step, carry.ended)))


def initial_carry(kind, batch):
    require(kind in KINDS and type(batch) is int and batch > 0, "positive declared batch")
    width = 28 if kind == "innovation_gru" else 29
    return Carry(kind, torch.zeros(batch, width, dtype=torch.float32, device="cpu"),
                 torch.zeros(batch, SCORE_DIM, dtype=torch.float32, device="cpu"),
                 torch.zeros(batch, dtype=torch.bool, device="cpu"),
                 torch.zeros(batch, dtype=torch.int64, device="cpu"),
                 torch.zeros(batch, dtype=torch.bool, device="cpu"))


def tensor(value, dtype, shape, name):
    require(isinstance(value, torch.Tensor) and value.dtype == dtype
            and value.device.type == "cpu" and tuple(value.shape) == tuple(shape), name)
    return value


def centered(value):
    return value - value.mean(dim=-1, keepdim=True)


class CrossQueryHead(torch.nn.Module):
    def __init__(self, kind, seed):
        super().__init__()
        self.kind, self.seed = kind, seed
        self.width = 28 if kind == "innovation_gru" else 29
        inputs = 40 if kind == "innovation_gru" else 35
        self.recurrent = torch.nn.GRU(inputs, self.width, batch_first=True,
                                      device="cpu", dtype=torch.float32)
        self.output = torch.nn.Linear(self.width, SCORE_DIM, device="cpu", dtype=torch.float32)
        if kind == "innovation":
            self.correction = torch.nn.Linear(SCORE_DIM, self.width, bias=False,
                                              device="cpu", dtype=torch.float32)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.zero_()
            if kind == "innovation":
                self.correction.weight.zero_()

    def initial_carry(self, batch):
        return initial_carry(self.kind, batch)

    def _input(self, features, raw_anchor, error=None, query=False):
        # Features may be [B,31] or [B,L,31]; anchors have matching leading axes.
        parts = [features, centered(raw_anchor / SCALE)]
        if self.kind == "innovation_gru":
            parts.append(torch.zeros_like(raw_anchor) if error is None else error)
            parts.append(torch.full((*features.shape[:-1], 1), float(query), dtype=torch.float32, device="cpu"))
        return torch.cat(parts, dim=-1)

    def _step(self, inputs, hidden):
        _, state = self.recurrent(inputs[:, None, :], hidden[None, :, :])
        return state[0]

    def _prediction(self, hidden, raw_anchor):
        return (raw_anchor / SCALE + centered(self.output(hidden))) * SCALE

    def _validate_carry(self, carry, batch):
        require(isinstance(carry, Carry) and carry.kind == self.kind, "matching carry family")
        for value, dtype, shape, name in (
            (carry.hidden, torch.float32, (batch, self.width), "carry hidden"),
            (carry.raw_anchor, torch.float32, (batch, SCORE_DIM), "carry raw anchor"),
            (carry.has_query, torch.bool, (batch,), "carry query flag"),
            (carry.absolute_step, torch.int64, (batch,), "carry absolute step"),
            (carry.ended, torch.bool, (batch,), "carry end flag"),
        ):
            tensor(value, dtype, shape, name)
            require(not value.requires_grad and value.grad_fn is None, "input carry must be explicitly detached")
        require(bool(torch.isfinite(carry.hidden).all()) and bool(torch.isfinite(carry.raw_anchor).all()),
                "finite carry state")
        require(bool(((carry.absolute_step >= 0) & (carry.absolute_step <= HORIZON)).all())
                and torch.equal(carry.has_query, carry.absolute_step > 0), "carry chronology/query identity")
        fresh = ~carry.has_query
        require(not bool(carry.ended[fresh].any()) and bool((carry.hidden[fresh] == 0).all())
                and bool((carry.raw_anchor[fresh] == 0).all()), "pristine unstarted lanes")
        require(torch.equal(carry.raw_anchor / SCALE * SCALE, carry.raw_anchor), "anchor scale roundtrip")

    def forward(self, features, query_scores, lengths, query_mask, *, carry=None, episode_ends):
        """Return (raw predictions [B,T,4], Carry) for a chronological chunk.

        CPU float32 features[B,T,31] and query_scores[B,T,4]; CPU int64 lengths[B]
        are in0..T. Bool query_mask[B,T] must equal active absolute-step%4==0;
        bool episode_ends[B] declares actual episode ends (found or censored),
        including endings exactly on a chunk boundary. T is1..2188. Ended lanes
        allow length0 only.

        Only active features and actual query score rows are read numerically.
        Skipped score slots and all padded features/scores may contain arbitrary
        nonfinite poison. Padding predictions are zero. Query predictions copy
        supplied raw float32 vectors exactly. Teacher costs must roundtrip through
        division/multiplication by64, preserving the qualified initial-hold domain.

        The three nonquery rows of each period share one nn.GRU sequence call per
        active-length group. No padded transition advances state. Nonquery outputs
        are offsets from the last true query anchor, never cumulative predictions.
        Input tensors and carry are not modified. Numerical output/carry is causal;
        validation can reject a malformed later active row before returning.
        """
        require(isinstance(features, torch.Tensor) and features.ndim == 3, "feature rank")
        batch, span, dimension = features.shape
        require(batch > 0 and 1 <= span <= HORIZON and dimension == FEATURE_DIM, "bounded feature geometry")
        tensor(features, torch.float32, (batch, span, FEATURE_DIM), "CPU float32 features")
        tensor(query_scores, torch.float32, (batch, span, SCORE_DIM), "CPU float32 query-only scores")
        tensor(lengths, torch.int64, (batch,), "CPU int64 lengths")
        tensor(query_mask, torch.bool, (batch, span), "CPU bool query mask")
        tensor(episode_ends, torch.bool, (batch,), "CPU bool episode ends")
        require(bool(((lengths >= 0) & (lengths <= span)).all()), "lengths within chunk")
        require(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in self.parameters()),
                "CPU float32 parameters")
        carry = self.initial_carry(batch) if carry is None else carry
        self._validate_carry(carry, batch)
        require(not bool((carry.ended & (lengths > 0)).any()), "ended lanes cannot advance")
        require(not bool((episode_ends & (lengths == 0) & ~carry.has_query).any()), "no empty new episode")
        advancing = lengths > 0
        require(bool((carry.absolute_step[advancing] % PERIOD == 0).all()), "chunks begin at query boundaries")
        require(bool(((lengths % PERIOD == 0) | episode_ends | carry.ended).all()),
                "nonterminal chunks end at query boundaries")
        next_steps = carry.absolute_step + lengths
        require(bool((next_steps <= HORIZON).all())
                and not bool(((next_steps == HORIZON) & ~(episode_ends | carry.ended)).any()),
                "horizon requires genuine episode end")
        active = torch.arange(span, device="cpu")[None, :] < lengths[:, None]
        absolute = carry.absolute_step[:, None] + torch.arange(span, device="cpu")[None, :]
        require(torch.equal(query_mask, active & (absolute % PERIOD == 0)), "exact active query mask")
        require(bool(torch.isfinite(features[active]).all()), "finite active features")
        require(torch.equal(features[:, :, 15][active], (absolute[active] / HORIZON).to(torch.float32))
                and torch.equal(features[:, :, 16][active], ((absolute[active] % PERIOD) / HORIZON).to(torch.float32))
                and bool((features[:, :, 17][active] == 1).all()), "declared public chronology features")
        visible = query_scores[query_mask]
        require(bool(torch.isfinite(visible).all()) and torch.equal(visible / SCALE * SCALE, visible),
                "finite query scores with exact scale roundtrip")

        hidden, anchor = carry.hidden.clone(), carry.raw_anchor.clone()
        has_query = carry.has_query.clone()
        outputs = [torch.zeros(batch, SCORE_DIM, dtype=torch.float32, device="cpu") for _ in range(span)]
        for offset in range(0, span, PERIOD):
            index = torch.nonzero(lengths > offset, as_tuple=False).flatten()
            if not index.numel():
                break
            current, observed = features[index, offset], query_scores[index, offset]
            old_hidden, old_anchor = hidden[index], anchor[index]
            if self.kind in ("persistent_direct", "reset_direct"):
                incoming = torch.zeros_like(old_hidden) if self.kind == "reset_direct" else old_hidden
                corrected = self._step(self._input(current, observed), incoming)
            else:
                first = ~has_query[index]
                corrected = torch.zeros_like(old_hidden)
                if bool(first.any()):
                    selected = torch.nonzero(first, as_tuple=False).flatten()
                    state = self._step(self._input(current[selected], observed[selected], query=True),
                                       torch.zeros_like(old_hidden[selected]))
                    corrected = corrected.index_copy(0, selected, state)
                if bool((~first).any()):
                    selected = torch.nonzero(~first, as_tuple=False).flatten()
                    prior = self._step(self._input(current[selected], old_anchor[selected]), old_hidden[selected])
                    prediction = self._prediction(prior, old_anchor[selected])
                    error = centered((observed[selected] - prediction) / SCALE)
                    require(bool(torch.isfinite(prior).all()) and bool(torch.isfinite(prediction).all())
                            and bool(torch.isfinite(error).all()), "finite prequery prediction and innovation")
                    if self.kind == "innovation":
                        state = prior + torch.tanh(self.correction(error))
                    else:
                        state = self._step(self._input(current[selected], observed[selected], error, query=True), prior)
                    corrected = corrected.index_copy(0, selected, state)
            hidden = hidden.index_copy(0, index, corrected)
            anchor = anchor.index_copy(0, index, observed)
            has_query = has_query.index_fill(0, index, True)
            outputs[offset] = outputs[offset].index_copy(0, index, observed)
            for count in range(1, min(PERIOD, span - offset)):
                group = torch.nonzero((lengths - offset - 1).clamp(0, PERIOD - 1) == count,
                                      as_tuple=False).flatten()
                if not group.numel():
                    continue
                inputs = self._input(features[group, offset + 1:offset + count + 1],
                                     anchor[group, None, :].expand(-1, count, -1))
                sequence, state = self.recurrent(inputs, hidden[group][None, :, :])
                raw = self._prediction(sequence, anchor[group, None, :])
                hidden = hidden.index_copy(0, group, state[0])
                for age in range(count):
                    outputs[offset + age + 1] = outputs[offset + age + 1].index_copy(0, group, raw[:, age])
        prediction = torch.stack(outputs, dim=1)
        require(bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(hidden).all()), "finite model results")
        result = Carry(self.kind, hidden, anchor, has_query, next_steps.clone(), carry.ended | episode_ends)
        return prediction, result


def make_head(kind, seed):
    """Fresh locally seeded model, without changing the caller's Torch RNG."""
    require(kind in KINDS and type(seed) is int and 0 <= seed < 2**32, "declared family and uint32 seed")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = CrossQueryHead(kind, seed)
    require(sum(p.numel() for p in model.parameters()) == parameter_count(kind), "exact parameter count")
    return model
