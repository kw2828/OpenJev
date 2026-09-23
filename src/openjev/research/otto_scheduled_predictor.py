"""Period-aware GRU predictor for fabricated engineering checks, not a study.

This separate forward admits actual query periods four and eight. It preserves
the qualified period-four predictor's parameter names, arithmetic and grouped
nonquery GRU calls. The existing sources are never modified or monkeypatched.
There are eight state tensors and6112 parameters, including the existing116
parameter residual whose output is scaled by64 before centering.

Later-query keys are the hidden state AFTER advancing the public observation
but BEFORE assimilating that query's scores. Nonquery keys are the existing
forecast hidden states. First-query and padded keys are positive zero. The
shadow prior adds the existing residual to that causal prior; it does not feed
the innovation, recurrent state, raw anchor or deployed query prediction.
There is no fast memory or additional recurrent pass in this module.

Frozen mode retains the original parameter flags (frozen backbone, trainable
residual flags) but runs this ENTIRE predictor under no_grad. Joint mode obeys
the caller's gradient context. Period-four bitwise comparisons use the original
with equivalent parameter flags and effective gradient context, including an
outer no_grad for the frozen original's residual. Constructor/factories preserve
the caller's CPU RNG; from_state validates and copies tensors without file IO.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch

from openjev.research import otto_cross_query_scores as base
from openjev.research import otto_protected_training_model as original

VERSION = "otto-scheduled-predictor-v1"
ORIGINAL_SOURCE_SHA256 = "2168dfa34096e154445599cba93964e6c60cfd6ccd3b77ac84624039e6db8bd0"
MODES, QUERY_PERIODS = ("frozen", "joint"), (4, 8)
KIND, WIDTH, FEATURE_DIM, SCORE_DIM, SCALE, HORIZON = "innovation_gru", 28, 31, 4, 64.0, 2188
PARAMETERS, BACKBONE_PARAMETERS, RESIDUAL_PARAMETERS = 6112, 5996, 116
STATE_SHAPES = {**original.BACKBONE_SHAPES,
                "action_residual.weight": (4, 28), "action_residual.bias": (4,)}
require = base.require


@dataclass(frozen=True)
class Carry:
    """Owned chronological state with an explicit, immutable schedule identity."""

    query_period: int
    base: base.Carry


@dataclass(frozen=True)
class Forecast:
    """CPU outputs; joint-mode floating tensors may retain gradient graphs.

    prediction/action_prediction/prior/shadow_prior are[B,T,4]; keys_hidden is
    [B,T,28]. Bool masks are[B,T]. Prior fields are positive zero off prior_mask;
    keys are positive zero off key_mask. Containers own their state, but frozen
    dataclasses do not make tensors read-only. work_counts counts calls and token
    transitions actually executed, not FLOPs, wall time or a compute estimate.
    """

    prediction: torch.Tensor
    action_prediction: torch.Tensor
    prior: torch.Tensor
    shadow_prior: torch.Tensor
    prior_mask: torch.Tensor
    keys_hidden: torch.Tensor
    key_mask: torch.Tensor
    carry: Carry
    work_counts: dict[str, int]


def _configuration(mode, seed, query_period):
    require(mode in MODES and type(seed) is int and 0 <= seed < 2**32,
            "declared mode and uint32 seed")
    require(type(query_period) is int and query_period in QUERY_PERIODS,
            "query period must be four or eight")


def verify_sources():
    path = Path(original.__file__)
    require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == ORIGINAL_SOURCE_SHA256,
            "immutable normalized predictor source pin")


def detach_carry(carry):
    """Detach and clone every base tensor; never alias or modify input state."""
    require(isinstance(carry, Carry) and type(carry.query_period) is int
            and carry.query_period in QUERY_PERIODS and isinstance(carry.base, base.Carry)
            and carry.base.kind == KIND, "declared scheduled carry")
    return Carry(carry.query_period, base.detach_carry(carry.base))


def parameter_count(mode, *, trainable_only=False):
    require(mode in MODES and type(trainable_only) is bool, "declared parameter count mode")
    return RESIDUAL_PARAMETERS if trainable_only and mode == "frozen" else PARAMETERS


class ScheduledPredictor(original.ProtectedTrainingModel):
    """Same seeded weights and period-four kernel grouping, separate schedule.

    A fresh instance has no persistent episode state. Use explicit detached carry
    between TBPTT chunks. Chunks start/end at this period's query boundaries,
    except genuine episode endings; fixed32-step training chunks satisfy both
    schedules. Reentrant/concurrent forwards are rejected by the inherited lock.
    """

    def __init__(self, mode, seed, query_period):
        _configuration(mode, seed, query_period)
        verify_sources()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            super().__init__(mode, seed)
        self._query_period = query_period
        require(set(self.state_dict()) == set(STATE_SHAPES)
                and sum(p.numel() for p in self.parameters()) == PARAMETERS,
                "exact original eight-tensor predictor")

    @property
    def query_period(self):
        return self._query_period

    def initial_carry(self, batch):
        return Carry(self.query_period, base.initial_carry(KIND, batch))

    def forward(self, features, query_scores, lengths, query_mask, *, carry=None, episode_ends):
        require(self._action_lock.acquire(blocking=False), "scheduled predictor cannot be reentered or used concurrently")
        try:
            _configuration(self.mode, self.seed, self.query_period)
            require(self.kind == KIND and self._captured is self._action_hidden is None,
                    "unchanged family and no stale inherited capture")
            named = dict(self.named_parameters())
            require(set(named) == set(STATE_SHAPES) and all(
                p.requires_grad == (self.mode == "joint" or name.startswith("action_residual."))
                and p.device.type == "cpu" and p.dtype == torch.float32
                and tuple(p.shape) == STATE_SHAPES[name] for name, p in named.items()),
                "unchanged predictor parameter shapes and gradient locks")
            with torch.no_grad() if self.mode == "frozen" else nullcontext():
                return self._scheduled_forward(features, query_scores, lengths, query_mask,
                                               carry=carry, episode_ends=episode_ends)
        finally:
            self._action_lock.release()

    def _scheduled_forward(self, features, query_scores, lengths, query_mask, *, carry, episode_ends):
        period = self.query_period
        require(isinstance(features, torch.Tensor) and features.ndim == 3, "feature rank")
        batch, span, dimension = features.shape
        require(batch > 0 and 1 <= span <= HORIZON and dimension == FEATURE_DIM, "bounded feature geometry")
        base.tensor(features, torch.float32, (batch, span, FEATURE_DIM), "CPU float32 features")
        base.tensor(query_scores, torch.float32, (batch, span, SCORE_DIM), "CPU float32 query-only scores")
        base.tensor(lengths, torch.int64, (batch,), "CPU int64 lengths")
        base.tensor(query_mask, torch.bool, (batch, span), "CPU bool query mask")
        base.tensor(episode_ends, torch.bool, (batch,), "CPU bool episode ends")
        require(bool(((lengths >= 0) & (lengths <= span)).all()), "lengths within chunk")
        carry = self.initial_carry(batch) if carry is None else carry
        require(isinstance(carry, Carry) and type(carry.query_period) is int
                and carry.query_period == period, "carry cannot switch query period")
        state = carry.base
        base.CrossQueryHead._validate_carry(self, state, batch)
        require(not bool((state.ended & (lengths > 0)).any()), "ended lanes cannot advance")
        require(not bool((episode_ends & (lengths == 0) & ~state.has_query).any()), "no empty new episode")
        advancing = lengths > 0
        require(bool((state.absolute_step[advancing] % period == 0).all()), "chunks begin at query boundaries")
        require(bool(((lengths % period == 0) | episode_ends | state.ended).all()),
                "nonterminal chunks end at query boundaries")
        next_steps = state.absolute_step + lengths
        require(bool((next_steps <= HORIZON).all())
                and not bool(((next_steps == HORIZON) & ~(episode_ends | state.ended)).any()),
                "horizon requires genuine episode end")
        active = torch.arange(span, device="cpu")[None, :] < lengths[:, None]
        absolute = state.absolute_step[:, None] + torch.arange(span, device="cpu")[None, :]
        require(torch.equal(query_mask, active & (absolute % period == 0)), "exact active query mask")
        require(bool(torch.isfinite(features[active]).all()), "finite active features")
        require(torch.equal(features[:, :, 15][active], (absolute[active] / HORIZON).to(torch.float32))
                and torch.equal(features[:, :, 16][active], ((absolute[active] % period) / HORIZON).to(torch.float32))
                and bool((features[:, :, 17][active] == 1).all()), "declared public chronology features")
        visible = query_scores[query_mask]
        require(bool(torch.isfinite(visible).all()) and torch.equal(visible / SCALE * SCALE, visible),
                "finite query scores with exact scale roundtrip")
        prior_mask = query_mask & (absolute >= period)
        key_mask = active & (absolute >= 1)
        counts = {"recurrent_calls": 0, "recurrent_token_transitions": 0,
                  "base_readout_calls": 0, "base_readout_rows": 0,
                  "action_readout_calls": 0, "action_readout_rows": 0,
                  "shadow_readout_calls": 0, "shadow_readout_rows": 0,
                  "active_rows": int(active.sum()), "query_rows": int(query_mask.sum()),
                  "later_query_rows": int(prior_mask.sum()),
                  "nonquery_rows": int((active & ~query_mask).sum()), "key_rows": int(key_mask.sum())}

        def step(inputs, hidden):
            counts["recurrent_calls"] += 1
            counts["recurrent_token_transitions"] += len(inputs)
            return self._step(inputs, hidden)

        def predict(hidden, anchor):
            counts["base_readout_calls"] += 1
            counts["base_readout_rows"] += hidden.numel() // WIDTH
            return self._prediction(hidden, anchor)

        hidden, anchor, has_query = state.hidden.clone(), state.raw_anchor.clone(), state.has_query.clone()
        outputs = [torch.zeros(batch, SCORE_DIM, dtype=torch.float32, device="cpu") for _ in range(span)]
        priors = [torch.zeros_like(row) for row in outputs]
        keys = [torch.zeros(batch, WIDTH, dtype=torch.float32, device="cpu") for _ in range(span)]
        later, nonqueries = [], []
        # Keep the original ordering and tensor indexing, including separate
        # first-query/later-query groups and grouped nonquery sequences.
        for offset in range(0, span, period):
            index = torch.nonzero(lengths > offset, as_tuple=False).flatten()
            if not index.numel():
                break
            current, observed = features[index, offset], query_scores[index, offset]
            old_hidden, old_anchor = hidden[index], anchor[index]
            first = ~has_query[index]
            corrected = torch.zeros_like(old_hidden)
            if bool(first.any()):
                selected = torch.nonzero(first, as_tuple=False).flatten()
                value = step(self._input(current[selected], observed[selected], query=True),
                             torch.zeros_like(old_hidden[selected]))
                corrected = corrected.index_copy(0, selected, value)
            if bool((~first).any()):
                selected = torch.nonzero(~first, as_tuple=False).flatten()
                prior_hidden = step(self._input(current[selected], old_anchor[selected]), old_hidden[selected])
                prediction = predict(prior_hidden, old_anchor[selected])
                error = base.centered((observed[selected] - prediction) / SCALE)
                require(bool(torch.isfinite(prior_hidden).all()) and bool(torch.isfinite(prediction).all())
                        and bool(torch.isfinite(error).all()), "finite prequery prediction and innovation")
                value = step(self._input(current[selected], observed[selected], error, query=True), prior_hidden)
                corrected = corrected.index_copy(0, selected, value)
                lanes = index[selected]
                priors[offset] = priors[offset].index_copy(0, lanes, prediction)
                keys[offset] = keys[offset].index_copy(0, lanes, prior_hidden)
                later.append((offset, lanes, prior_hidden, prediction))
            hidden = hidden.index_copy(0, index, corrected)
            anchor = anchor.index_copy(0, index, observed)
            has_query = has_query.index_fill(0, index, True)
            outputs[offset] = outputs[offset].index_copy(0, index, observed)
            for count in range(1, min(period, span - offset)):
                group = torch.nonzero((lengths - offset - 1).clamp(0, period - 1) == count,
                                      as_tuple=False).flatten()
                if not group.numel():
                    continue
                inputs = self._input(features[group, offset + 1:offset + count + 1],
                                     anchor[group, None, :].expand(-1, count, -1))
                counts["recurrent_calls"] += 1
                counts["recurrent_token_transitions"] += len(group) * count
                sequence, value = self.recurrent(inputs, hidden[group][None, :, :])
                raw = predict(sequence, anchor[group, None, :])
                hidden = hidden.index_copy(0, group, value[0])
                for age in range(count):
                    outputs[offset + age + 1] = outputs[offset + age + 1].index_copy(0, group, raw[:, age])
                    keys[offset + age + 1] = keys[offset + age + 1].index_copy(0, group, sequence[:, age])
                nonqueries.append((offset, group, count, sequence))
        prediction = torch.stack(outputs, dim=1)
        require(bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(hidden).all()), "finite model results")
        action = [prediction[:, offset].clone() for offset in range(span)]
        # Exactly the original residual call geometry and scale-before-center
        # arithmetic. The new query-shadow readouts follow this unchanged path.
        for offset, group, count, sequence in nonqueries:
            counts["action_readout_calls"] += 1
            counts["action_readout_rows"] += len(group) * count
            raw = self.action_residual(sequence)
            residual = raw - raw.mean(dim=-1, keepdim=True)
            require(bool(torch.isfinite(residual).all()), "finite centered action residual")
            for age in range(count):
                position = offset + age + 1
                value = prediction[group, position] + residual[:, age]
                action[position] = action[position].index_copy(0, group, value)
        shadows = [torch.zeros_like(row) for row in outputs]
        for offset, lanes, prior_hidden, prior in later:
            counts["shadow_readout_calls"] += 1
            counts["shadow_readout_rows"] += len(lanes)
            raw = self.action_residual(prior_hidden)
            residual = raw - raw.mean(dim=-1, keepdim=True)
            value = prior + residual
            require(bool(torch.isfinite(value).all()), "finite causal shadow prior")
            shadows[offset] = shadows[offset].index_copy(0, lanes, value)
        action_prediction, keys_hidden = torch.stack(action, dim=1), torch.stack(keys, dim=1)
        require(bool(torch.isfinite(action_prediction).all()) and bool(torch.isfinite(keys_hidden).all()),
                "finite action predictions and keys")
        require(counts["recurrent_token_transitions"] == counts["active_rows"] + counts["later_query_rows"]
                and counts["base_readout_rows"] == counts["key_rows"]
                and counts["action_readout_rows"] == counts["nonquery_rows"]
                and counts["shadow_readout_rows"] == counts["later_query_rows"], "complete scheduled work accounting")
        result = base.Carry(KIND, hidden, anchor, has_query, next_steps.clone(), state.ended | episode_ends)
        return Forecast(prediction, action_prediction, torch.stack(priors, dim=1), torch.stack(shadows, dim=1),
                        prior_mask, keys_hidden, key_mask, Carry(period, result), counts)


def make_head(mode, seed, query_period):
    return ScheduledPredictor(mode, seed, query_period)


def from_state(mode, seed, query_period, state):
    """Validate all eight finite CPU float32 tensors before constructing a model.

    Copies preserve values without aliases or source mutation. The selected mode
    controls destination gradient flags; source flags are not inherited. The
    caller owns checkpoint provenance; this function performs no file loading.
    """
    _configuration(mode, seed, query_period)
    require(isinstance(state, Mapping) and set(state) == set(STATE_SHAPES), "exact eight predictor state keys")
    for name, shape in STATE_SHAPES.items():
        value = state[name]
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                and value.dtype == torch.float32 and value.layout == torch.strided
                and tuple(value.shape) == shape and bool(torch.isfinite(value).all()),
                "finite CPU float32 predictor tensor: " + name)
    model = make_head(mode, seed, query_period)
    model.load_state_dict(state, strict=True)
    return model
