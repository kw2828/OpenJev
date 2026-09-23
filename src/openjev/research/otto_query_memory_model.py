"""Explicit composition of the qualified slow predictor and query memory.

This is engineering infrastructure, not an admitted experiment or efficacy
claim. Matrix modes learn only a bias-free hidden-to-key projection. Their slow
predictor is frozen under no_grad; its mandatory residual requires_grad flag
does not make it an effective optimization parameter. Use
effective_named_parameters() to construct an optimizer.

The none mode also permits an ordinary joint predictor. Its original action and
prior graphs are retained, rather than using the memory kernel's detached copy.
None and last_error construct no projection. A matrix projection processes a
fixed [batch,28] tensor once per physical step with any eligible key. Keeping
that geometry across chunks avoids changing GEMM/GEMV rounding at tails.

No module stores episode state. The composed carry joins both clocks before
either forward, and detach_carry clones both states. Query/padding outputs stay
exact. Corrected shadow priors exist only at genuine later-query rows and add
the normalized PREWRITE correction to the complete slow shadow forecast.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch

from openjev.research import otto_query_memory as memory
from openjev.research import otto_scheduled_predictor as predictor

VERSION = "otto-query-memory-model-v1"
PREDICTOR_SOURCE_SHA256 = "e30c3e311bdde24e1dd5692e29e74578412744728afb3f2d9fb93de09eeaaf27"
MEMORY_SOURCE_SHA256 = "5b705bf3b9d376e13309d3f4932bccfc263c59fce51e90eebc87a290e8de2409"
DEFAULT_KEY_DIM = 8
require = memory.require


@dataclass(frozen=True)
class Carry:
    slow: predictor.Carry
    fast: memory.Carry


@dataclass(frozen=True)
class Forecast:
    prediction: torch.Tensor
    slow_action_prediction: torch.Tensor
    prior: torch.Tensor
    shadow_prior: torch.Tensor
    action_prediction: torch.Tensor
    corrected_shadow_prior: torch.Tensor
    prewrite_correction: torch.Tensor
    prior_mask: torch.Tensor
    key_mask: torch.Tensor
    carry: Carry
    work_counts: dict[str, int]
    memory_work_units: dict[str, int]
    parameter_metadata: dict


def verify_sources():
    for module, expected in ((predictor, PREDICTOR_SOURCE_SHA256), (memory, MEMORY_SOURCE_SHA256)):
        source = Path(module.__file__)
        require(source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == expected,
                "immutable qualified composition source: " + module.__name__)


def configuration(config, seed, query_period, slow_mode):
    config = memory.Config(key_dim=DEFAULT_KEY_DIM) if config is None else config
    require(isinstance(config, memory.Config), "memory Config")
    require(type(seed) is int and 0 <= seed < 2**32, "uint32 projection/model seed")
    require(type(query_period) is int and query_period in predictor.QUERY_PERIODS, "supported query period")
    require(slow_mode in predictor.MODES and (slow_mode == "frozen" or config.mode == "none"),
            "memory adaptation requires frozen slow predictor; joint is ordinary none only")
    return config


def validate_slow_state(state):
    require(isinstance(state, Mapping) and set(state) == set(predictor.STATE_SHAPES), "exact eight slow state keys")
    for name, shape in predictor.STATE_SHAPES.items():
        value = state[name]
        memory.tensor(value, torch.float32, shape, "CPU float32 slow state: " + name)
        require(bool(torch.isfinite(value).all()), "finite slow state: " + name)


def validate_projection_state(config, value):
    if config.mode not in memory.MATRIX_MODES:
        require(value is None, "no projection state for unpadded none/last_error")
    else:
        memory.tensor(value, torch.float32, (config.key_dim, predictor.WIDTH), "CPU float32 projection state")
        require(bool(torch.isfinite(value).all()), "finite projection state")


def detach_carry(carry):
    require(isinstance(carry, Carry), "composed Carry")
    return Carry(predictor.detach_carry(carry.slow), memory.detach_carry(carry.fast))


class QueryMemoryModel(torch.nn.Module):
    """Own a supplied slow module without modifying its weights or flags."""

    def __init__(self, slow, config=None, seed=0):
        super().__init__()
        require(type(slow) is predictor.ScheduledPredictor, "qualified ScheduledPredictor instance")
        config = configuration(config, seed, slow.query_period, slow.mode)
        verify_sources()
        validate_slow_state(slow.state_dict())
        self.slow, self.config, self.seed = slow, config, seed
        self.memory = memory.QueryMemory(config)
        self.projection = None
        if config.mode in memory.MATRIX_MODES:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                self.projection = torch.nn.Linear(predictor.WIDTH, config.key_dim, bias=False,
                                                  device="cpu", dtype=torch.float32)

    def effective_named_parameters(self):
        """Actual gradient paths, distinct from inherited requires_grad flags."""
        if self.config.mode == "none" and self.slow.mode == "joint":
            return [("slow." + name, value) for name, value in self.slow.named_parameters()]
        if self.projection is not None:
            return [("projection.weight", self.projection.weight)]
        return []

    def parameter_metadata(self):
        named = list(self.named_parameters())
        flags = [(name, value) for name, value in named if value.requires_grad]
        effective = self.effective_named_parameters()
        return {"names": [name for name, _ in named], "count": sum(p.numel() for _, p in named),
                "requires_grad_names": [name for name, _ in flags],
                "requires_grad_count": sum(p.numel() for _, p in flags),
                "effective_names": [name for name, _ in effective],
                "effective_count": sum(p.numel() for _, p in effective)}

    def initial_carry(self, batch):
        return Carry(self.slow.initial_carry(batch), self.memory.initial_carry(batch))

    def _joined_carry(self, carry, batch):
        require(isinstance(carry, Carry) and isinstance(carry.slow, predictor.Carry)
                and type(carry.slow.query_period) is int and carry.slow.query_period == self.slow.query_period,
                "composed carry schedule identity")
        # These are validation-only methods from the pinned components, before
        # either numerical forward or a projection call.
        self.slow._validate_carry(carry.slow.base, batch)
        self.memory._carry(carry.fast, batch)
        for name in ("absolute_step", "has_query", "ended"):
            require(torch.equal(getattr(carry.slow.base, name), getattr(carry.fast, name)),
                    "joined slow/fast carry " + name)
        return carry

    def forward(self, features, query_scores, lengths, query_mask, *, episode_ends,
                carry=None, no_write=False):
        require(type(no_write) is bool, "Boolean no_write intervention")
        configuration(self.config, self.seed, self.slow.query_period, self.slow.mode)
        require(isinstance(features, torch.Tensor) and features.ndim == 3 and features.shape[0] > 0,
                "composed feature rank/batch")
        batch, span, _ = features.shape
        memory.tensor(lengths, torch.int64, (batch,), "CPU int64 composed lengths")
        memory.tensor(episode_ends, torch.bool, (batch,), "CPU bool composed episode ends")
        state = self.initial_carry(batch) if carry is None else carry
        state = self._joined_carry(state, batch)
        require(not bool((state.fast.ended & (lengths > 0)).any()), "ended composed lanes cannot advance")
        if self.config.mode in memory.MATRIX_MODES:
            require(type(self.projection) is torch.nn.Linear and self.projection.bias is None
                    and self.projection.weight.requires_grad, "unchanged bias-free trainable projection")
            validate_projection_state(self.config, self.projection.weight)
        else:
            require(self.projection is None, "unpadded baseline has no projection")
        with torch.no_grad() if self.slow.mode == "frozen" else nullcontext():
            slow = self.slow(features, query_scores, lengths, query_mask,
                             episode_ends=episode_ends, carry=state.slow)
        projection_calls = projection_rows = projection_key_rows = 0
        projected = []
        for step in range(span):
            if self.projection is not None and bool(slow.key_mask[:, step].any()):
                # Fixed physical-batch shape; never flatten variable chunk time.
                key = self.projection(slow.keys_hidden[:, step])
                key = torch.where(slow.key_mask[:, step, None], key, torch.zeros_like(key))
                projected.append(key)
                projection_calls += 1
                projection_rows += batch
                projection_key_rows += int(slow.key_mask[:, step].sum())
            else:
                projected.append(torch.zeros(batch, self.config.key_dim, dtype=torch.float32))
        keys = torch.stack(projected, dim=1)
        fast = self.memory(slow.action_prediction, slow.shadow_prior, keys, query_scores, query_mask,
                           slow.prior_mask, slow.key_mask, lengths, episode_ends,
                           carry=state.fast, no_write=no_write)
        # The kernel intentionally detaches the slow path. Ordinary joint
        # training must use the original graphs instead of that detached copy.
        action = slow.action_prediction if self.config.mode == "none" else fast.action_prediction
        corrected = slow.shadow_prior.clone()
        zero_intervention = no_write and bool((state.fast.matrix == 0).all()) and bool((state.fast.last_error == 0).all())
        if self.config.mode != "none" and not zero_intervention:
            lane, step = torch.nonzero(slow.prior_mask, as_tuple=True)
            if lane.numel():
                values = slow.shadow_prior[lane, step] + predictor.SCALE * fast.prewrite_correction[lane, step]
                require(bool(torch.isfinite(values).all()), "finite corrected shadow prior")
                corrected = corrected.index_put((lane, step), values)
        result_carry = Carry(slow.carry, fast.carry)
        # Result states may carry projection graphs, so validate only clock joins
        # here; the next call must explicitly detach both components.
        for name in ("absolute_step", "has_query", "ended"):
            require(torch.equal(getattr(result_carry.slow.base, name), getattr(result_carry.fast, name)),
                    "result slow/fast carry " + name)
        counts = {"slow_" + name: value for name, value in slow.work_counts.items()}
        counts.update({"memory_" + name: value for name, value in fast.work.items()})
        counts.update(projection_calls=projection_calls, projection_rows=projection_rows,
                      projection_key_rows=projection_key_rows,
                      projection_linear_terms=projection_rows * predictor.WIDTH * self.config.key_dim)
        return Forecast(slow.prediction, slow.action_prediction, slow.prior, slow.shadow_prior,
                        action, corrected, fast.prewrite_correction, slow.prior_mask, slow.key_mask,
                        result_carry, counts, dict(fast.work_units), self.parameter_metadata())


def make_model(config=None, seed=0, query_period=4, *, slow_mode="frozen"):
    config = configuration(config, seed, query_period, slow_mode)
    return QueryMemoryModel(predictor.make_head(slow_mode, seed, query_period), config, seed)


def from_states(config, seed, query_period, slow_state, projection_weight, *, slow_mode="frozen"):
    """Strict in-memory copies only; no checkpoint or empirical file loading."""
    config = configuration(config, seed, query_period, slow_mode)
    validate_slow_state(slow_state)
    validate_projection_state(config, projection_weight)
    model = QueryMemoryModel(predictor.from_state(slow_mode, seed, query_period, slow_state), config, seed)
    if model.projection is not None:
        with torch.no_grad():
            model.projection.weight.copy_(projection_weight)
    return model
