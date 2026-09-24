"""Fabricated feature-cache checks against the unchanged composed predictor."""
from __future__ import annotations

import copy
import math
from itertools import pairwise

import numpy as np
import pytest
import torch
from test_otto_query_memory_data import census

from openjev.research import otto_direct_readout as solver
from openjev.research import otto_direct_readout_cache as cache_module
from openjev.research import otto_query_memory as memory
from openjev.research import otto_query_memory_data as data
from openjev.research import otto_query_memory_model as composed
from openjev.research import otto_scheduled_predictor as predictor


def history(lengths=(1, 4, 5, 31, 32, 33, 65)):
    flat, identities = census(lengths)
    flat["raw_q"][flat["episode_offsets"][:-1], 0] = np.float32(-0.)
    return data.project_census(flat, identities, query_period=4, expected_stage="train")


def state():
    model = predictor.make_head("frozen", 53, 4)
    with torch.no_grad():
        model.output.weight.copy_(torch.arange(112, dtype=torch.float32).sin().reshape(4, 28) * .02)
        model.output.bias.copy_(torch.tensor([.1, -.2, .3, -.1]))
        model.action_residual.weight.copy_(torch.arange(112, dtype=torch.float32).cos().reshape(4, 28) * .003)
        model.action_residual.bias.copy_(torch.tensor([.04, -.03, .02, -.01]))
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def bits(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().contiguous().numpy()
    return value.tobytes()


def ordinary(state_values, values):
    """Actual ordinary composition, including the original packet/carry path."""
    model = composed.from_states(memory.Config("none", 8), 53, 4, state_values, None, slow_mode="frozen")
    model.eval()
    count = int(values["episode_offsets"][-1])
    result = np.zeros((count, 4), np.float32)
    with torch.no_grad():
        for index, (low, high) in enumerate(pairwise(values["episode_offsets"])):
            low, high = int(low), int(high)
            carry = model.initial_carry(1)
            for start in range(0, high - low, 32):
                packet = data.batch_chunk(values, [index], start)
                inputs = {name: torch.from_numpy(value) for name, value in packet["model_inputs"].items()}
                output = model(**inputs, carry=carry)
                length = int(inputs["lengths"][0])
                take = slice(low + start, low + start + length)
                query = values["query_mask"][take]
                prior = values["prior_mask"][take]
                block = result[take]
                block[~query] = output.action_prediction[0, :length].numpy()[~query]
                block[prior] = output.corrected_shadow_prior[0, :length].numpy()[prior]
                assert bits(output.action_prediction[0, :length].numpy()[query]) == bits(values["query_scores"][take][query])
                carry = composed.detach_carry(output.carry)
    return result


@pytest.mark.parametrize("changed", (False, True))
def test_affine_parent_and_changed_head_match_ordinary_normalized_inference(changed):
    values, supplied = history(), state()
    cache = cache_module.extract(supplied, 53, values)
    head = {name: value.clone() for name, value in supplied.items()}
    if changed:
        head["action_residual.weight"] += torch.linspace(-.03, .04, 112).reshape(4, 28)
        head["action_residual.bias"] += torch.tensor([.1, -.2, .05, .03])
    reference = ordinary(head, values).astype(np.float64) / 64
    predicted = cache_module.affine_prediction(cache, cache_module.theta_from_state(head))
    np.testing.assert_allclose(predicted, reference, atol=1e-5, rtol=1e-5)
    assert np.any(cache["prior_mask"]) and np.any(cache["nonquery_mask"])
    if not changed:
        assert bits(cache["parent_prediction"]) == bits(ordinary(supplied, values))
    else:
        altered = cache_module.extract(head, 53, values)
        for name in ("z", "base", "query_mask", "prior_mask", "support_mask", "episode_offsets"):
            assert bits(cache[name]) == bits(altered[name])
        assert not np.array_equal(cache["parent_prediction"][cache["nonquery_mask"]],
                                  altered["parent_prediction"][cache["nonquery_mask"]])
        assert not np.array_equal(cache["parent_prediction"][cache["prior_mask"]],
                                  altered["parent_prediction"][cache["prior_mask"]])


def test_owned_storage_masks_exact_work_and_no_caller_or_rng_changes():
    lengths = (1, 4, 5, 31, 32, 33, 65)
    values, supplied = history(lengths), state()
    before_arrays = {name: bits(value) for name, value in values.items() if isinstance(value, np.ndarray)}
    before_state = {name: bits(value) for name, value in supplied.items()}
    rng = torch.random.get_rng_state().clone()
    checks = []
    cache = cache_module.extract(supplied, 53, values, check=lambda: checks.append(None))
    assert set(cache) == cache_module.CACHE_FIELDS
    assert bits(rng) == bits(torch.random.get_rng_state())
    assert all(bits(supplied[name]) == value for name, value in before_state.items())
    assert all(bits(values[name]) == value for name, value in before_arrays.items())
    for name, value in cache.items():
        if isinstance(value, np.ndarray):
            assert value.flags.owndata
            assert all(not np.shares_memory(value, source) for source in values.values() if isinstance(source, np.ndarray))
    assert cache["z"].dtype == np.float64 and cache["z"].shape == (sum(lengths), 29)
    assert np.array_equal(cache["z"][:, 28], np.ones(sum(lengths)))
    assert np.array_equal(cache["z"][:, :28].astype(np.float32).astype(np.float64), cache["z"][:, :28])
    assert np.array_equal(cache["support_mask"], cache["nonquery_mask"] | cache["prior_mask"])
    starts = values["episode_offsets"][:-1]
    for name in ("base", "parent_prediction"):
        assert bits(cache[name][starts]) == bits(np.zeros((len(starts), 4), np.float32))
    query = sum((length + 3) // 4 for length in lengths)
    prior = query - len(lengths)
    groups = sum((length + 2) // 4 for length in lengths)
    expected = {"model_constructions": 1, "forward_chunks": sum((length + 31) // 32 for length in lengths),
                "complete_episodes": len(lengths), "slow_recurrent_calls": query + prior + groups,
                "slow_recurrent_token_transitions": sum(lengths) + prior,
                "slow_base_readout_calls": prior + groups, "slow_base_readout_rows": sum(lengths) - len(lengths),
                "slow_action_readout_calls": groups, "slow_action_readout_rows": sum(lengths) - query,
                "slow_shadow_readout_calls": prior, "slow_shadow_readout_rows": prior,
                "slow_active_rows": sum(lengths), "slow_query_rows": query, "slow_later_query_rows": prior,
                "slow_nonquery_rows": sum(lengths) - query, "slow_key_rows": sum(lengths) - len(lengths)}
    assert cache["work"] == expected
    assert len(checks) == 1 + expected["forward_chunks"]
    assert math.isfinite(cache["seconds"]) and cache["seconds"] >= 0


def test_loss_labels_are_not_read_and_nonquery_input_poison_is_unconsumed():
    values, supplied = history((33, 5)), state()
    baseline = cache_module.extract(supplied, 53, values)

    class NoRead:
        def __array__(self, *args, **kwargs):
            raise AssertionError("loss labels were read")

        def __getitem__(self, key):
            raise AssertionError("loss labels were read")

    changed = dict(values)
    for name in ("targets", "legal", "nonquery_weights", "prior_weights", "actions"):
        changed[name] = NoRead()
    changed["query_scores"] = values["query_scores"].copy()
    changed["query_scores"][~values["query_mask"]] = np.array([np.nan, np.inf, -np.inf, 1e30], np.float32)
    actual = cache_module.extract(supplied, 53, changed)
    assert not {"targets", "legal", "query_scores", "weights"} & actual.keys()
    for name, value in baseline.items():
        if isinstance(value, np.ndarray):
            assert bits(value) == bits(actual[name])
    assert baseline["work"] == actual["work"]


def test_later_query_cache_is_prewrite_and_episodes_reset():
    values, supplied = history((9, 9)), state()
    baseline = cache_module.extract(supplied, 53, values)
    altered = dict(values)
    altered["query_scores"] = values["query_scores"].copy()
    altered["query_scores"][4] += np.array([3., -1., 2., 0.], np.float32)
    changed = cache_module.extract(supplied, 53, altered)
    for name in ("z", "base", "parent_prediction"):
        assert bits(baseline[name][:5]) == bits(changed[name][:5])
        assert bits(baseline[name][9:]) == bits(changed[name][9:])
    assert not np.array_equal(baseline["z"][5:9], changed["z"][5:9])


def test_all_single_step_episodes_have_zero_supported_affine_outputs():
    supplied = state()
    cache = cache_module.extract(supplied, 53, history((1, 1)))
    assert not cache["support_mask"].any()
    result = cache_module.affine_prediction(cache, cache_module.theta_from_state(supplied))
    assert bits(result) == bits(np.zeros((2, 4), np.float64))
    assert cache["work"]["slow_action_readout_rows"] == cache["work"]["slow_shadow_readout_rows"] == 0


@pytest.mark.parametrize("defect", ("period", "count", "age", "score", "query", "prior", "offset"))
def test_invalid_history_fails_before_model_construction(monkeypatch, defect):
    values = {name: value.copy() if isinstance(value, np.ndarray) else value for name, value in history((9,)).items()}
    if defect == "period":
        values["query_period"] = 8
    elif defect == "count":
        values["episode_count"] = 2
    elif defect == "age":
        values["features"][4, 16] = np.float32(4 / 2188)
    elif defect == "score":
        values["query_scores"][4, 0] = np.nan
    elif defect == "query":
        values["query_mask"][1] = True
    elif defect == "prior":
        values["prior_mask"][0] = True
    else:
        values["episode_offsets"][-1] -= 1
    monkeypatch.setattr(predictor, "from_state", lambda *args, **kwargs: pytest.fail("model construction before validation"))
    with pytest.raises(ValueError):
        cache_module.extract({}, 53, values)


def test_interruption_preserves_caller_state_and_rng():
    supplied, values = state(), history((33,))
    snapshot = {name: bits(value) for name, value in supplied.items()}
    rng = bits(torch.random.get_rng_state())
    calls = []

    def stop():
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError("fabricated stop")

    with pytest.raises(RuntimeError, match="fabricated stop"):
        cache_module.extract(supplied, 53, values, check=stop)
    assert bits(torch.random.get_rng_state()) == rng
    assert all(bits(supplied[name]) == value for name, value in snapshot.items())


@pytest.mark.parametrize("defect", ("theta_shape", "theta_dtype", "theta_nan", "mask", "bias", "base"))
def test_affine_guard_failures_preserve_cache_and_theta(defect):
    supplied = state()
    cache = cache_module.extract(supplied, 53, history((5,)))
    theta = cache_module.theta_from_state(supplied)
    if defect == "theta_shape":
        theta = theta[:, :28]
    elif defect == "theta_dtype":
        theta = theta.astype(np.float32)
    elif defect == "theta_nan":
        theta[0, 0] = np.nan
    elif defect == "mask":
        cache["prior_mask"][0] = True
    elif defect == "bias":
        cache["z"][1, -1] = 0
    else:
        cache["base"][0] = 1
    before = {name: bits(value) for name, value in cache.items() if isinstance(value, np.ndarray)}
    theta_before = bits(theta)
    with pytest.raises(ValueError):
        cache_module.affine_prediction(cache, theta)
    assert bits(theta) == theta_before
    assert all(bits(cache[name]) == value for name, value in before.items())


def test_affine_formula_centering_owned_results_and_head_export_copy():
    supplied = state()
    cache = cache_module.extract(supplied, 53, history((5,)))
    theta = cache_module.theta_from_state(supplied)
    original = theta.copy()
    result = cache_module.affine_prediction(cache, theta)
    common = np.arange(29, dtype=np.float64) / 100
    shifted = cache_module.affine_prediction(cache, theta + common[None])
    np.testing.assert_allclose(result, shifted, atol=1e-12, rtol=1e-12)
    support = cache["support_mask"]
    correction = result[support] - cache["base"][support].astype(np.float64) / 64
    np.testing.assert_allclose(correction.sum(axis=1), 0, atol=1e-14, rtol=0)
    assert result.flags.owndata and not np.shares_memory(result, cache["base"])
    theta[:] = 3
    np.testing.assert_array_equal(cache_module.theta_from_state(supplied), original)
    cached = copy.deepcopy(cache)
    result[:] = 11
    assert all(bits(cache[name]) == bits(value) for name, value in cached.items() if isinstance(value, np.ndarray))


def test_weighted_solve_float32_export_and_ordinary_inference_integration():
    """One canonical cache, both fixed solvers, then the real inference path.

    Synthetic regression labels use a small known contrast head plus tiny
    off-subspace noise. They are not copied into the model's query inputs.
    Natural correlated GRU features exercise the fixed numerical rank cutoff.
    """
    values, parent = history((33, 37, 65)), state()
    cache = cache_module.extract(parent, 53, values)
    theta = cache_module.theta_from_state(parent)
    known_B = np.sin(np.arange(87, dtype=np.float64).reshape(3, 29)) * .007
    known_increment = solver.contrast_basis() @ known_B
    parent_affine = cache_module.affine_prediction(cache, theta)
    noise = np.cos(np.arange(len(cache["z"]) * 4, dtype=np.float64).reshape(-1, 4)) * 1e-12
    targets = parent_affine + cache["z"] @ known_increment.T + noise
    legal = values["legal"].copy()
    nonquery_indices = np.flatnonzero(cache["nonquery_mask"])
    legal[nonquery_indices[::2], 0] = False
    legal[cache["prior_mask"]] = True
    weights = values["nonquery_weights"] / legal.sum(axis=1) + values["prior_weights"] / 4
    A, b = solver.build_design(cache["z"], targets - parent_affine, legal, weights)
    snapshot = {name: bits(value) for name, value in parent.items()}
    for mode in ("ols", "ridge"):
        result = solver.solve_design(A, b, mode=mode)
        assert result["diagnostics"]["total_objective"] <= result["diagnostics"]["objective_before"] + 1e-12
        assert result["diagnostics"]["coordinates"] == 87
        solved_theta = theta + result["increment"]
        solved_prediction = cache_module.affine_prediction(cache, solved_theta)
        # Direct legal-subset arithmetic checks the assembled loss independently.
        scalar_loss = 0.
        for row in np.flatnonzero(weights > 0):
            error = (solved_prediction[row] - targets[row])[legal[row]]
            centered = error - sum(float(x) for x in error) / len(error)
            scalar_loss += float(weights[row]) * sum(float(x) ** 2 for x in centered)
        assert scalar_loss == pytest.approx(result["diagnostics"]["data_loss"], rel=1e-8, abs=1e-12)
        exported = {name: value.clone() for name, value in parent.items()}
        head = solved_theta.astype(np.float32)
        exported["action_residual.weight"] = torch.from_numpy(head[:, :28].copy())
        exported["action_residual.bias"] = torch.from_numpy(head[:, 28].copy())
        expected = cache_module.affine_prediction(cache, cache_module.theta_from_state(exported))
        actual = ordinary(exported, values).astype(np.float64) / 64
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        for name in parent:
            if not name.startswith("action_residual."):
                assert bits(exported[name]) == snapshot[name]
        assert all(bits(parent[name]) == before for name, before in snapshot.items())
