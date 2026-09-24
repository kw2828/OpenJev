"""Fabricated cache integration checks; no checkpoints, datasets or admission."""
from __future__ import annotations

import copy
import random
from itertools import pairwise

import numpy as np
import pytest
import torch

from openjev.research import otto_query_memory as memory
from openjev.research import otto_query_memory_model as composed
from openjev.research import otto_residual_contract as contract
from openjev.research import otto_residual_features as features_module
from openjev.research import otto_scheduled_predictor as predictor


def fabricated_inputs(lengths=(1, 4, 5, 31, 32, 33, 65)):
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    features = np.linspace(-.35, .55, int(offsets[-1]) * 31, dtype=np.float32).reshape(-1, 31)
    scores = np.full((int(offsets[-1]), 4), np.nan, dtype=np.float32)
    for low, high in pairwise(offsets):
        steps = np.arange(high - low)
        features[low:high, 15] = (steps / 2188).astype(np.float32)
        features[low:high, 16] = ((steps % 4) / 2188).astype(np.float32)
        features[low:high, 17] = 1
        for step in steps[steps % 4 == 0]:
            scores[low + step] = np.array([4., 7., 2., 6.], dtype=np.float32) + step * np.array(
                [.125, -.25, .5, .25], dtype=np.float32)
        scores[low, 0] = np.float32(-0.)
    return features, scores, offsets


def supplied_models():
    pretrained = predictor.make_head("frozen", 517, 4)
    joint = predictor.make_head("frozen", 719, 4)
    with torch.no_grad():
        for index, model in enumerate((pretrained, joint), 1):
            model.output.weight.copy_(torch.arange(112).sin().reshape(4, 28) * (.015 * index))
            model.action_residual.weight.copy_(torch.arange(112).cos().reshape(4, 28) * (.003 * index))
            model.action_residual.bias.copy_(torch.tensor([.3, -.2, .1, .4]) * index)
    pretrained.train(True)
    joint.train(False)
    projection = (torch.arange(224, dtype=torch.float32).sin().reshape(8, 28) * .07).requires_grad_()
    return pretrained, joint, projection


def bits(value):
    if isinstance(value, torch.Tensor):
        return value.detach().contiguous().numpy().tobytes()
    return value.tobytes()


def model_witness(model):
    return {
        "weights": {name: bits(value) for name, value in model.state_dict().items()},
        "flags": {name: value.requires_grad for name, value in model.named_parameters()},
        "grads": {name: None if value.grad is None else bits(value.grad) for name, value in model.named_parameters()},
        "training": {name: value.training for name, value in model.named_modules()},
    }


def chunk_packet(features, scores, low, high, start):
    count = min(32, high - low - start)
    values = torch.full((1, 32, 31), float("nan"), dtype=torch.float32)
    answers = torch.full((1, 32, 4), float("inf"), dtype=torch.float32)
    values[0, :count] = torch.from_numpy(features[low + start:low + start + count].copy())
    answers[0, :count] = torch.from_numpy(scores[low + start:low + start + count].copy())
    query = torch.zeros((1, 32), dtype=torch.bool)
    query[0, :count] = torch.arange(start, start + count) % 4 == 0
    return count, {"features": values, "query_scores": answers, "query_mask": query,
                   "lengths": torch.tensor([count]), "episode_ends": torch.tensor([start + count == high - low])}


def uncached_oracle(pretrained, joint, projection, features, scores, offsets):
    """Original composition plus an independent normalized EMA, never cache code."""
    slow = {name: value.detach().clone() for name, value in pretrained.state_dict().items()}
    model = composed.from_states(memory.Config("trace_delta", 8), 73, 4, slow, projection.detach().clone())
    joint_copy = predictor.from_state("frozen", 79, 4, joint.state_dict())
    output = {name: [] for name in ("base_action", "joint_action", "shadow_prior", "cues",
                                    "query_mask", "prior_mask", "key_mask")}
    projections = []
    hook = model.projection.register_forward_hook(lambda _module, _args, result: projections.append(result.detach().clone()))
    try:
        with torch.no_grad():
            for low, high in pairwise(offsets):
                carry = joint_carry = None
                trace = torch.zeros((1, 8), dtype=torch.float32)
                for start in range(0, int(high - low), 32):
                    count, packet = chunk_packet(features, scores, int(low), int(high), start)
                    projections.clear()
                    result = model(**packet, carry=carry, no_write=True)
                    ordinary = joint_copy(**packet, carry=joint_carry)
                    keys = iter(projections)
                    cues = torch.zeros((count, 8), dtype=torch.float32)
                    for step in range(count):
                        if result.key_mask[0, step]:
                            raw = next(keys)
                            normalized = raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-6)
                            trace = .25 * normalized + .75 * trace
                            cues[step] = (trace / torch.linalg.vector_norm(trace, dim=-1, keepdim=True).clamp_min(1e-6))[0]
                    assert next(keys, None) is None
                    assert bits(trace) == bits(result.carry.fast.trace)
                    for name, value in (("base_action", result.slow_action_prediction[0, :count]),
                                        ("joint_action", ordinary.action_prediction[0, :count]),
                                        ("shadow_prior", result.shadow_prior[0, :count]),
                                        ("cues", cues), ("query_mask", packet["query_mask"][0, :count]),
                                        ("prior_mask", result.prior_mask[0, :count]),
                                        ("key_mask", result.key_mask[0, :count])):
                        output[name].append(value.detach().numpy().copy())
                    carry = composed.detach_carry(result.carry)
                    joint_carry = predictor.detach_carry(ordinary.carry)
    finally:
        hook.remove()
    return {name: np.concatenate(values) for name, values in output.items()}


def build(values=None, models=None):
    values = fabricated_inputs() if values is None else values
    models = supplied_models() if models is None else models
    return features_module.build_cache(*models, *values)


def test_cache_matches_uncached_composition_and_independent_trace_across_32_step_boundaries():
    values, models = fabricated_inputs(), supplied_models()
    expected = uncached_oracle(*models, *values)
    cache = build(values, models)
    assert set(cache) == contract.CACHE_FIELDS
    assert cache["version"] == "otto-residual-cache-v1" and cache["query_period"] == 4
    for name, expected_value in expected.items():
        assert cache[name].dtype == expected_value.dtype
        assert cache[name].shape == expected_value.shape
        assert bits(cache[name]) == bits(expected_value), name
    assert bits(cache["query_scores"]) == bits(values[1])
    assert bits(cache["episode_offsets"]) == bits(values[2])
    starts = values[2][:-1]
    assert not cache["prior_mask"][starts].any() and not cache["key_mask"][starts].any()
    assert bits(cache["cues"][starts]) == bits(np.zeros((len(starts), 8), np.float32))
    assert bits(cache["shadow_prior"][~cache["prior_mask"]]) == bits(
        np.zeros((int((~cache["prior_mask"]).sum()), 4), np.float32))
    assert bits(cache["base_action"][cache["query_mask"]]) == bits(values[1][cache["query_mask"]])
    assert bits(cache["joint_action"][cache["query_mask"]]) == bits(values[1][cache["query_mask"]])
    assert np.any(cache["joint_action"][~cache["query_mask"]] != cache["base_action"][~cache["query_mask"]])


def test_nonquery_label_poison_does_not_change_any_model_output_or_work():
    values, models = fabricated_inputs((33, 5)), supplied_models()
    original = build(values, models)
    changed = tuple(value.copy() for value in values)
    changed[1][~original["query_mask"]] = np.array([np.inf, -np.inf, 1e30, np.nan], np.float32)
    poisoned = build(changed, models)
    for name in original:
        if name == "query_scores":
            continue
        if isinstance(original[name], np.ndarray):
            assert bits(original[name]) == bits(poisoned[name]), name
        else:
            assert original[name] == poisoned[name]
    assert bits(poisoned["query_scores"]) == bits(changed[1])


def test_query_labels_do_not_change_their_prewrite_shadow_or_cue_and_suffix_is_causal():
    values, models = fabricated_inputs((33,)), supplied_models()
    original = build(values, models)
    changed = tuple(value.copy() for value in values)
    changed[1][12] += np.array([8., -4., 3., -7.], np.float32)
    observed = build(changed, models)
    for name in ("cues", "shadow_prior"):
        assert bits(original[name][:13]) == bits(observed[name][:13]), name
    for name in ("base_action", "joint_action"):
        assert bits(original[name][:12]) == bits(observed[name][:12]), name
        assert bits(observed[name][12]) == bits(changed[1][12])
    future = tuple(value.copy() for value in values)
    future[0][20:, 0:4] += .125
    future[1][20::4] += np.array([1., -2., 3., -2.], np.float32)
    later = build(future, models)
    for name in ("base_action", "joint_action", "shadow_prior", "cues"):
        assert bits(original[name][:20]) == bits(later[name][:20]), name


def test_episode_resets_match_independent_single_episode_calls():
    values, models = fabricated_inputs((1, 31, 32, 33)), supplied_models()
    combined = build(values, models)
    for low, high in zip(values[2][:-1], values[2][1:], strict=True):
        single = build((values[0][low:high].copy(), values[1][low:high].copy(),
                        np.array([0, high - low], np.int64)), models)
        for name in ("base_action", "joint_action", "shadow_prior", "cues", "query_scores",
                     "query_mask", "prior_mask", "key_mask"):
            assert bits(combined[name][low:high]) == bits(single[name]), name


def test_cache_work_counts_match_episode_schedule_and_fixed_projection_geometry():
    lengths = (1, 4, 5, 31, 32, 33, 65)
    cache = build(fabricated_inputs(lengths))
    counts = cache["work_counts"]
    rows = sum(lengths)
    queries = sum((length - 1) // 4 + 1 for length in lengths)
    later = queries - len(lengths)
    keys = rows - len(lengths)
    groups = sum((length - 2) // 4 + 1 if length > 1 else 0 for length in lengths)
    chunks = sum((length + 31) // 32 for length in lengths)
    assert counts["pretrained_forward_chunks"] == counts["joint_forward_chunks"] == chunks
    for prefix in ("pretrained_slow_", "joint_slow_"):
        expected = {"active_rows": rows, "query_rows": queries, "later_query_rows": later,
                    "key_rows": keys, "nonquery_rows": rows - queries,
                    "recurrent_calls": len(lengths) + 2 * later + groups,
                    "recurrent_token_transitions": rows + later,
                    "base_readout_calls": later + groups, "base_readout_rows": keys,
                    "action_readout_calls": groups, "action_readout_rows": rows - queries,
                    "shadow_readout_calls": later, "shadow_readout_rows": later}
        for name, value in expected.items():
            assert counts[prefix + name] == value, prefix + name
    for name in ("projection_calls", "projection_rows", "projection_key_rows", "cue_key_normalizations",
                 "cue_normalizations", "cue_trace_updates"):
        assert counts[name] == keys
    assert counts["projection_linear_terms"] == 224 * keys
    assert counts["cue_normalized_coordinates"] == 16 * keys
    assert counts["cue_trace_mixed_coordinates"] == 8 * keys
    assert all(type(value) is int and value >= 0 for value in counts.values())


def test_complete_2188_step_horizon_and_2189_rejection_before_any_forward(monkeypatch):
    models = supplied_models()
    cache = build(fabricated_inputs((2188,)), models)
    assert cache["episode_offsets"].tolist() == [0, 2188]
    steps = np.arange(2188)
    np.testing.assert_array_equal(cache["query_mask"], steps % 4 == 0)
    np.testing.assert_array_equal(cache["prior_mask"], (steps > 0) & (steps % 4 == 0))
    np.testing.assert_array_equal(cache["key_mask"], steps > 0)
    assert cache["base_action"].shape == cache["joint_action"].shape == (2188, 4)
    assert cache["cues"].shape == (2188, 8)
    counts = cache["work_counts"]
    assert counts["pretrained_forward_chunks"] == counts["joint_forward_chunks"] == 69
    assert counts["projection_key_rows"] == 2187
    for prefix in ("pretrained_slow_", "joint_slow_"):
        for name, expected in {"active_rows": 2188, "query_rows": 547,
                               "later_query_rows": 546, "key_rows": 2187,
                               "nonquery_rows": 1641, "recurrent_token_transitions": 2734}.items():
            assert counts[prefix + name] == expected

    def forbidden(*_args, **_kwargs):
        pytest.fail("overlong complete episode reached a frozen model forward")

    for model in models[:2]:
        monkeypatch.setattr(model, "forward", forbidden)
    with pytest.raises(ValueError, match="bounded complete episodes"):
        build(fabricated_inputs((2189,)), models)


def test_cache_preserves_weights_gradients_flags_training_modes_rng_and_input_ownership():
    values, models = fabricated_inputs((5, 33)), supplied_models()
    for model in models[:2]:
        model.action_residual.bias.grad = torch.full_like(model.action_residual.bias, .125)
    models[2].grad = torch.full_like(models[2], .25)
    before = [model_witness(model) for model in models[:2]]
    projection = (bits(models[2]), models[2].requires_grad, bits(models[2].grad))
    inputs = [bits(value) for value in values]
    torch_rng = torch.random.get_rng_state().clone()
    numpy_rng, python_rng = copy.deepcopy(np.random.get_state()), random.getstate()
    cache = build(values, models)
    assert [model_witness(model) for model in models[:2]] == before
    assert (bits(models[2]), models[2].requires_grad, bits(models[2].grad)) == projection
    assert [bits(value) for value in values] == inputs
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    current_numpy = np.random.get_state()
    assert numpy_rng[0] == current_numpy[0] and bits(numpy_rng[1]) == bits(current_numpy[1])
    assert numpy_rng[2:] == current_numpy[2:] and python_rng == random.getstate()
    for source in values:
        for value in cache.values():
            if isinstance(value, np.ndarray):
                assert not np.shares_memory(source, value)
    for value in cache.values():
        if isinstance(value, np.ndarray):
            value[...] = 0
    assert [bits(value) for value in values] == inputs
    assert [model_witness(model) for model in models[:2]] == before


@pytest.mark.parametrize("defect", ("late_feature_nan", "late_chronology", "late_query_nan", "features_dtype",
                                   "scores_dtype", "offsets_dtype", "empty_episode", "incomplete_offsets"))
def test_entire_input_roster_is_rejected_before_either_model_forward(defect, monkeypatch):
    features, scores, offsets = fabricated_inputs((33, 9))
    models = supplied_models()
    if defect == "late_feature_nan":
        features[-1, 0] = np.nan
    elif defect == "late_chronology":
        features[-1, 16] += .125
    elif defect == "late_query_nan":
        scores[-1, 0] = np.nan
    elif defect == "features_dtype":
        features = features.astype(np.float64)
    elif defect == "scores_dtype":
        scores = scores.astype(np.float64)
    elif defect == "offsets_dtype":
        offsets = offsets.astype(np.int32)
    elif defect == "empty_episode":
        offsets = np.array([0, 33, 33, 42], np.int64)
    else:
        offsets[-1] -= 1

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid later input reached a frozen model forward")

    for model in models[:2]:
        monkeypatch.setattr(model, "forward", forbidden)
    with pytest.raises((ValueError, TypeError)):
        build((features, scores, offsets), models)


@pytest.mark.parametrize("defect", ("projection_nan", "projection_dtype", "projection_shape", "pretrained_period",
                                   "joint_mode", "slow_weight_nan", "slow_flag"))
def test_invalid_model_or_projection_fails_before_any_forward(defect, monkeypatch):
    pretrained, joint, projection = supplied_models()
    if defect == "projection_nan":
        projection = projection.detach().clone()
        projection[-1, -1] = float("nan")
    elif defect == "projection_dtype":
        projection = projection.double()
    elif defect == "projection_shape":
        projection = projection.T
    elif defect == "pretrained_period":
        pretrained = predictor.make_head("frozen", 517, 8)
    elif defect == "joint_mode":
        joint = predictor.make_head("joint", 719, 4)
    elif defect == "slow_weight_nan":
        with torch.no_grad():
            joint.output.bias[0] = float("nan")
    else:
        joint.output.weight.requires_grad_(True)

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid model/projection reached a frozen model forward")

    monkeypatch.setattr(pretrained, "forward", forbidden)
    monkeypatch.setattr(joint, "forward", forbidden)
    with pytest.raises((ValueError, TypeError)):
        build(models=(pretrained, joint, projection))
