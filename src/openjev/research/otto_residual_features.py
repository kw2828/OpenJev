"""Pure frozen-predictor feature extraction for a prospective residual screen.

The caller supplies two existing qualified ScheduledPredictor instances in
frozen P4 mode, an explicit CPU float32 [8,28] projection tensor and complete
flat NumPy arrays. This module loads no file/checkpoint, constructs no model,
fits no parameter and calls no teacher or environment. Source/checkpoint/cohort
provenance and phase admission remain the caller's responsibility.

Execution geometry is exactly one complete episode at a time, physical batch
one, chronological 32-step chunks including a poisoned final padding suffix.
The pretrained and Joint AUX predictors each execute their own full forward.
Projection always receives [1,28] and produces [1,8] at each eligible key step;
variable-length time is never flattened into its batch dimension.

Keys are float32-normalized with norm.clamp_min(1e-6). The carried trace is the
UNNORMALIZED .25 * key + .75 * previous_trace. The exported cue normalizes that
trace using the same float32 arithmetic order as the qualified trace-delta
kernel. Every episode starts with zero trace. Step zero has no key/cue; later
queries expose a pre-assimilation key and complete slow shadow prior. The cache
contains no estimator read, residual write, correction or hidden target.

All forwards run under no_grad. Models keep their weights, parameter flags and
training flags. The CPU RNG is preserved, including when a forward raises.
Caller arrays and the projection tensor are never written; returned arrays own
their storage. Nonquery score poison is copied bitwise and remains unconsumed.
Work counters describe actual calls/rows and coordinate terms, not FLOPs,
latency, equality of compute or a deployment speedup.
"""
from __future__ import annotations

from itertools import pairwise

import numpy as np
import torch

from openjev.research import otto_residual_contract as contract
from openjev.research import otto_scheduled_predictor as predictor

CHUNK = 32
EPSILON = 1e-6
TRACE_DECAY = .75
require = contract.require


def _model_snapshot(model, label):
    require(type(model) is predictor.ScheduledPredictor and model.mode == "frozen" and model.query_period == 4,
            label + " must be the qualified frozen P4 ScheduledPredictor")
    require(model.kind == predictor.KIND and model._captured is model._action_hidden is None,
            label + " must have no inherited stale capture")
    parameters = dict(model.named_parameters())
    require(set(parameters) == set(predictor.STATE_SHAPES) and set(model.state_dict()) == set(predictor.STATE_SHAPES),
            label + " requires all eight qualified slow tensors")
    for name, value in parameters.items():
        require(value.device.type == "cpu" and value.dtype == torch.float32 and value.layout == torch.strided
                and tuple(value.shape) == predictor.STATE_SHAPES[name] and bool(torch.isfinite(value).all())
                and value.requires_grad == name.startswith("action_residual."), label + " parameter shape, values and flags")
    return {"weights": {name: value.detach().numpy().tobytes() for name, value in model.state_dict().items()},
            "flags": {name: value.requires_grad for name, value in parameters.items()},
            "training": {name: value.training for name, value in model.named_modules()},
            "mode": model.mode, "query_period": model.query_period, "seed": model.seed}


def _normalize(value):
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    require(bool(torch.isfinite(norm).all()), "finite projected key or cue norm")
    normalized = value / norm.clamp_min(EPSILON)
    require(bool(torch.isfinite(normalized).all()), "finite normalized projected key or cue")
    return normalized


def _add_slow_counts(counts, prefix, values):
    require(isinstance(values, dict) and bool(values), "qualified slow work counters")
    for name, value in values.items():
        require(type(name) is str and type(value) is int and value >= 0, "nonnegative integer slow work count")
        key = prefix + name
        counts[key] = counts.get(key, 0) + value


def _packet(features, scores, masks, low, high, start):
    length = min(CHUNK, high - low - start)
    take = slice(low + start, low + start + length)
    # Copies avoid caller aliases, including when the caller's NumPy buffers are
    # read-only or strided. Masked scores keep their original poison bits.
    feature_chunk = torch.full((1, CHUNK, contract.FEATURE_DIM), float("nan"), dtype=torch.float32)
    score_chunk = torch.full((1, CHUNK, contract.SCORE_DIM), float("nan"), dtype=torch.float32)
    feature_chunk[0, :length] = torch.from_numpy(features[take].copy())
    score_chunk[0, :length] = torch.from_numpy(scores[take].copy())
    query = torch.zeros((1, CHUNK), dtype=torch.bool)
    prior = torch.zeros_like(query)
    key = torch.zeros_like(query)
    for target, name in ((query, "query_mask"), (prior, "prior_mask"), (key, "key_mask")):
        target[0, :length] = torch.from_numpy(masks[name][take].copy())
    return {"features": feature_chunk, "query_scores": score_chunk,
            "query_mask": query, "lengths": torch.tensor([length], dtype=torch.int64),
            "episode_ends": torch.tensor([start + length == high - low], dtype=torch.bool)}, prior, key, take


def build_cache(pretrained_slow, joint_slow, projection_weight, features, query_scores, episode_offsets):
    """Build one complete, independently owned cache with fixed P4 geometry.

    ``features`` is float32[T,31], ``query_scores`` is float32[T,4] with only
    actual query rows consumed, and ``episode_offsets`` is int64[N+1]. Both
    model arguments are existing ScheduledPredictor objects, not state files.
    ``projection_weight`` is CPU torch.float32[8,28]; its requires_grad flag
    may be either value and is left unchanged. All episodes must be nonempty
    and no longer than the qualified horizon. Invalid chronology anywhere in
    the split is rejected before either model executes.
    """
    masks = contract.validate_feature_inputs(features, query_scores, episode_offsets)
    pretrained_snapshot = _model_snapshot(pretrained_slow, "pretrained")
    joint_snapshot = _model_snapshot(joint_slow, "joint")
    require(isinstance(projection_weight, torch.Tensor) and projection_weight.device.type == "cpu"
            and projection_weight.dtype == torch.float32 and projection_weight.layout == torch.strided
            and tuple(projection_weight.shape) == (contract.KEY_DIM, predictor.WIDTH)
            and bool(torch.isfinite(projection_weight).all()), "finite CPU float32 [8,28] fixed projection")
    projection_snapshot = projection_weight.detach().numpy().tobytes(), projection_weight.requires_grad
    projection = projection_weight.detach().clone().contiguous()
    # Freeze private input snapshots after complete validation. No view exposed
    # in the result or passed to either predictor aliases a caller array.
    features = features.copy(order="C")
    scores = query_scores.copy(order="C")
    offsets = episode_offsets.copy()
    rows = len(features)
    cache = {"version": contract.CACHE_VERSION, "query_period": contract.QUERY_PERIOD,
        "base_action": np.zeros((rows, 4), dtype=np.float32), "joint_action": np.zeros((rows, 4), dtype=np.float32),
        "shadow_prior": np.zeros((rows, 4), dtype=np.float32), "cues": np.zeros((rows, 8), dtype=np.float32),
        "query_scores": scores.copy(), "episode_offsets": offsets.copy(), **{name: value.copy() for name, value in masks.items()}}
    counts = {name: 0 for name in ("pretrained_forward_chunks", "joint_forward_chunks", "projection_calls",
        "projection_rows", "projection_key_rows", "projection_linear_terms", "cue_key_normalizations",
        "cue_normalizations", "cue_trace_updates", "cue_normalized_coordinates", "cue_trace_mixed_coordinates")}
    rng = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        for low, high in pairwise(offsets):
            low, high = int(low), int(high)
            pretrained_carry = pretrained_slow.initial_carry(1)
            joint_carry = joint_slow.initial_carry(1)
            trace = torch.zeros((1, contract.KEY_DIM), dtype=torch.float32)
            for start in range(0, high - low, CHUNK):
                packet, prior_mask, key_mask, take = _packet(features, scores, masks, low, high, start)
                length = int(packet["lengths"][0])
                slow = pretrained_slow(**packet, carry=pretrained_carry)
                counts["pretrained_forward_chunks"] += 1
                _add_slow_counts(counts, "pretrained_slow_", slow.work_counts)
                joint = joint_slow(**packet, carry=joint_carry)
                counts["joint_forward_chunks"] += 1
                _add_slow_counts(counts, "joint_slow_", joint.work_counts)
                for forecast in (slow, joint):
                    require(torch.equal(forecast.prior_mask, prior_mask) and torch.equal(forecast.key_mask, key_mask),
                            "both predictors retain the same complete observation schedule")
                    for value in (forecast.action_prediction, forecast.shadow_prior, forecast.keys_hidden):
                        require(bool(torch.isfinite(value).all()), "finite frozen predictor output")
                        padding = value[:, length:].detach().numpy()
                        require(padding.tobytes() == np.zeros_like(padding).tobytes(), "positive-zero predictor padding")
                    require(forecast.carry.base.absolute_step.tolist() == [start + length]
                            and forecast.carry.base.has_query.tolist() == [True]
                            and forecast.carry.base.ended.tolist() == packet["episode_ends"].tolist(), "complete chunk carry chronology")
                cache["base_action"][take] = slow.action_prediction[0, :length].detach().numpy()
                cache["joint_action"][take] = joint.action_prediction[0, :length].detach().numpy()
                cache["shadow_prior"][take] = slow.shadow_prior[0, :length].detach().numpy()
                for step in range(length):
                    if not bool(key_mask[0, step]):
                        continue
                    projected = torch.nn.functional.linear(slow.keys_hidden[:, step], projection)
                    key = _normalize(projected)
                    # Match the original float32 ordering and preserve this
                    # UNNORMALIZED trace across chunks; normalize only the read.
                    trace = (1 - TRACE_DECAY) * key + TRACE_DECAY * trace
                    cue = _normalize(trace)
                    cache["cues"][low + start + step] = cue[0].detach().numpy()
                    counts["projection_calls"] += 1
                    counts["projection_rows"] += 1
                    counts["projection_key_rows"] += 1
                    counts["projection_linear_terms"] += predictor.WIDTH * contract.KEY_DIM
                    counts["cue_key_normalizations"] += 1
                    counts["cue_normalizations"] += 1
                    counts["cue_trace_updates"] += 1
                    counts["cue_normalized_coordinates"] += 2 * contract.KEY_DIM
                    counts["cue_trace_mixed_coordinates"] += contract.KEY_DIM
                pretrained_carry = predictor.detach_carry(slow.carry)
                joint_carry = predictor.detach_carry(joint.carry)
            require(pretrained_carry.base.absolute_step.tolist() == joint_carry.base.absolute_step.tolist() == [high - low]
                    and pretrained_carry.base.ended.tolist() == joint_carry.base.ended.tolist() == [True],
                    "both complete episodes consumed before reset")
        require(torch.equal(torch.random.get_rng_state(), rng), "feature extraction never consumes CPU randomness")
    require(_model_snapshot(pretrained_slow, "pretrained") == pretrained_snapshot
            and _model_snapshot(joint_slow, "joint") == joint_snapshot, "model weights and flags unchanged by caching")
    require((projection_weight.detach().numpy().tobytes(), projection_weight.requires_grad) == projection_snapshot,
            "caller projection bytes and flags unchanged")
    require(torch.equal(torch.random.get_rng_state(), rng), "caller CPU RNG preserved")
    key_rows, query_rows, prior_rows = (int(masks[name].sum()) for name in ("key_mask", "query_mask", "prior_mask"))
    expected_chunks = sum((int(high - low) + CHUNK - 1) // CHUNK for low, high in pairwise(offsets))
    require(counts["pretrained_forward_chunks"] == counts["joint_forward_chunks"] == expected_chunks
            and counts["projection_calls"] == counts["projection_rows"] == counts["projection_key_rows"] == key_rows,
            "complete separate forwards and fixed-geometry projections")
    for prefix in ("pretrained_slow_", "joint_slow_"):
        require(counts[prefix + "active_rows"] == rows and counts[prefix + "query_rows"] == query_rows
                and counts[prefix + "later_query_rows"] == prior_rows and counts[prefix + "key_rows"] == key_rows
                and counts[prefix + "recurrent_token_transitions"] == rows + prior_rows,
                "complete charged slow row and transition counts")
    cache["work_counts"] = counts
    contract.validate_cache(cache)
    return cache
