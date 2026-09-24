"""Fabricated estimator integration with independent algebraic oracles only."""
from __future__ import annotations

import copy
from itertools import pairwise

import numpy as np
import pytest

from openjev.research import otto_residual_contract as contract
from openjev.research import otto_residual_replay as replay_module


def fabricated_cache(lengths=(1, 4, 5, 9, 33)):
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    rows = int(offsets[-1])
    rng = np.random.default_rng(157)
    masks = {name: np.zeros(rows, np.bool_) for name in ("query_mask", "prior_mask", "key_mask")}
    scores = np.full((rows, 4), np.nan, np.float32)
    shadow = np.full((rows, 4), np.inf, np.float32)
    cues = rng.normal(size=(rows, 8)).astype(np.float32)
    cues /= np.linalg.norm(cues, axis=1, keepdims=True)
    base = rng.normal(size=(rows, 4)).astype(np.float32)
    joint = (rng.normal(size=(rows, 4)) + .75).astype(np.float32)
    for episode, (low, high) in enumerate(pairwise(offsets)):
        steps = np.arange(high - low)
        masks["query_mask"][low:high] = steps % 4 == 0
        masks["prior_mask"][low:high] = (steps > 0) & (steps % 4 == 0)
        masks["key_mask"][low + 1:high] = True
        cues[low] = np.nan
        for step in steps[steps % 4 == 0]:
            scores[low + step] = np.array([4., 7., 2., 6.], np.float32) + step * np.array(
                [.125, -.25, .5, .25], np.float32) + episode * .25
            if step:
                shadow[low + step] = np.array([2., 4., 1., 8.], np.float32) + step * .125
        scores[low, 0] = np.float32(-0.)
    query = masks["query_mask"]
    base[query] = scores[query]
    joint[query] = scores[query]
    return {"version": "otto-residual-cache-v1", "query_period": 4, "base_action": base,
            "joint_action": joint, "shadow_prior": shadow, "cues": cues, "query_scores": scores,
            "episode_offsets": offsets, "work_counts": {"fabricated_rows": rows}, **masks}


def center(value):
    return value - value.mean(axis=0)


def target(cache, row):
    return center((cache["query_scores"][row].astype(np.float64) -
                   cache["shadow_prior"][row].astype(np.float64)) / 64.)


def state_bytes(state):
    return {key: (value.dtype.str, value.shape, value.tobytes()) if isinstance(value, np.ndarray) else value
            for key, value in state.items()}


def cache_bytes(cache):
    return {key: (value.dtype.str, value.shape, value.tobytes()) if isinstance(value, np.ndarray)
            else copy.deepcopy(value) for key, value in cache.items()}


def replay(cache, method, tau=1.):
    return replay_module.replay(cache, method, tau=tau if method in contract.RLS_METHODS else None)


def spy_posteriors(monkeypatch):
    instances = []
    original = replay_module.BayesianScoreMemory

    class Recorded(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr(replay_module, "BayesianScoreMemory", Recorded)
    return instances


def batch_posterior(cues, targets, tau):
    if not cues:
        return np.zeros((4, 8)), np.eye(8) * tau
    design = np.asarray(cues, np.float64)
    responses = np.asarray(targets, np.float64)
    precision = np.eye(8) / tau + design.T @ design
    covariance = np.linalg.solve(precision, np.eye(8))
    weights = np.linalg.solve(precision, design.T @ responses).T
    return weights, covariance


@pytest.mark.parametrize("tau", (.01, .1, 1., 10.))
def test_full_replay_matches_independent_batch_information_form_before_each_write(tau, monkeypatch):
    cache = fabricated_cache((1, 4, 5, 9, 33))
    instances = spy_posteriors(monkeypatch)
    result = replay(cache, "rls_full", tau)
    assert len(instances) == 5
    expected_correction = np.zeros_like(result["prewrite_correction"])
    expected_variance = np.full(len(cache["base_action"]), np.nan)
    for episode, (low, high) in enumerate(zip(cache["episode_offsets"][:-1], cache["episode_offsets"][1:], strict=True)):
        observed_cues, observed_targets = [], []
        for row in range(int(low) + 1, int(high)):
            weights, covariance = batch_posterior(observed_cues, observed_targets, tau)
            cue = cache["cues"][row].astype(np.float64)
            expected_correction[row] = center(weights @ cue)
            expected_variance[row] = cue @ covariance @ cue
            if cache["prior_mask"][row]:
                observed_cues.append(cue)
                observed_targets.append(target(cache, row))
        expected_weights, expected_covariance = batch_posterior(observed_cues, observed_targets, tau)
        state = instances[episode].state_dict()
        np.testing.assert_allclose(state["weights"], expected_weights, rtol=2e-11, atol=2e-13)
        np.testing.assert_allclose(state["covariance"], expected_covariance, rtol=2e-11, atol=2e-13)
        assert state["labeled_updates"] == len(observed_targets)
    np.testing.assert_allclose(result["prewrite_correction"], expected_correction, rtol=2e-11, atol=2e-13)
    np.testing.assert_allclose(result["epistemic_variance"], expected_variance, rtol=2e-11, atol=2e-13, equal_nan=True)
    assert result["action_scores"].dtype == np.float32
    assert result["prewrite_correction"].dtype == result["epistemic_variance"].dtype == np.float64


def test_diagonal_moment_projection_differs_from_full_on_exact_mixed_cue_example(monkeypatch):
    cache = fabricated_cache((10,))
    cache["cues"][1:] = 0
    cache["cues"][4, :2] = [1., 1.]
    cache["cues"][8, :2] = [1., -1.]
    first = np.array([1., -1., 0., 0.])
    second = np.array([0., 1., -1., 0.])
    cache["shadow_prior"][[4, 8]] = 0
    cache["query_scores"][4] = (64 * first).astype(np.float32)
    cache["query_scores"][8] = (64 * second).astype(np.float32)
    for name in ("base_action", "joint_action"):
        cache[name][cache["query_mask"]] = cache["query_scores"][cache["query_mask"]]
    instances = spy_posteriors(monkeypatch)
    full = replay(cache, "rls_full")
    diagonal = replay(cache, "rls_diagonal")
    full_state, diagonal_state = [instance.state_dict() for instance in instances]
    expected = np.column_stack(((first + second) / 3, (first - second) / 3))
    np.testing.assert_allclose(full_state["weights"][:, :2], expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(full_state["covariance"][:2, :2], np.eye(2) / 3, rtol=0, atol=1e-15)
    expected = np.column_stack((first / 3 + 2 * second / 7, first / 3 - 2 * second / 7))
    np.testing.assert_allclose(diagonal_state["weights"][:, :2], expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(diagonal_state["covariance"][:2], [10 / 21, 10 / 21], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(full_state["covariance"][2:, 2:], np.eye(6))
    np.testing.assert_array_equal(diagonal_state["covariance"][2:], np.ones(6))
    np.testing.assert_allclose(full["prewrite_correction"][8], 0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(diagonal["prewrite_correction"][8], 0, rtol=0, atol=1e-15)
    assert full["epistemic_variance"][8] == pytest.approx(2.)
    assert diagonal["epistemic_variance"][8] == pytest.approx(4 / 3)


@pytest.mark.parametrize("method", ("last_error", "trace_delta"))
def test_non_bayesian_controls_follow_independent_float64_equations(method):
    cache = fabricated_cache((1, 4, 5, 9, 33))
    result = replay(cache, method)
    expected = np.zeros_like(result["prewrite_correction"])
    for low, high in zip(cache["episode_offsets"][:-1], cache["episode_offsets"][1:], strict=True):
        last, matrix = np.zeros(4), np.zeros((4, 8))
        for row in range(int(low) + 1, int(high)):
            cue = cache["cues"][row].astype(np.float64)
            if method == "last_error":
                last *= .75
                expected[row] = last
                if cache["prior_mask"][row]:
                    last = target(cache, row)
            else:
                expected[row] = center(matrix @ cue)
                if cache["prior_mask"][row]:
                    error = target(cache, row) - expected[row]
                    for coordinate in range(4):
                        for key in range(8):
                            matrix[coordinate, key] += .25 * error[coordinate] * cue[key] / (1e-6 + sum(cue * cue))
    np.testing.assert_allclose(result["prewrite_correction"], expected, rtol=2e-12, atol=2e-14)
    assert np.isnan(result["epistemic_variance"]).all()
    nonquery = ~cache["query_mask"]
    expected_scores = (cache["base_action"][nonquery].astype(np.float64) + 64 * expected[nonquery]).astype(np.float32)
    np.testing.assert_array_equal(result["action_scores"][nonquery], expected_scores)


@pytest.mark.parametrize("method,multiplier", (("rls_shrink_025", .25), ("rls_shrink_050", .5), ("rls_shrink_075", .75)))
def test_strict_attenuation_changes_only_applied_read_not_any_episode_posterior(method, multiplier, monkeypatch):
    cache = fabricated_cache((1, 9, 33))
    instances = spy_posteriors(monkeypatch)
    full = replay(cache, "rls_full", .1)
    expected_states = [state_bytes(model.state_dict()) for model in instances]
    instances.clear()
    weaker = replay(cache, method, .1)
    assert [state_bytes(model.state_dict()) for model in instances] == expected_states
    np.testing.assert_array_equal(weaker["prewrite_correction"], multiplier * full["prewrite_correction"])
    assert weaker["epistemic_variance"].tobytes() == full["epistemic_variance"].tobytes()
    assert weaker["work_counts"] == full["work_counts"]
    assert weaker["state_bytes"] == full["state_bytes"] == 768


@pytest.mark.parametrize("method", contract.METHODS)
def test_first_queries_hidden_targets_and_inactive_cues_never_enter_an_estimator(method):
    cache = fabricated_cache((1, 4, 5, 9))
    original = replay(cache, method)
    changed = copy.deepcopy(cache)
    changed["query_scores"][~cache["query_mask"]] = [np.inf, -np.inf, 1e30, np.nan]
    changed["shadow_prior"][~cache["prior_mask"]] = [-np.inf, 1e30, np.nan, np.inf]
    changed["cues"][~cache["key_mask"]] = np.inf
    # First answers remain visible to the already-computed backbone; the
    # estimator itself must neither read nor write their score residuals.
    starts = cache["episode_offsets"][:-1]
    changed["query_scores"][starts] = [256., -64., 32., -224.]
    changed["base_action"][starts] = changed["query_scores"][starts]
    changed["joint_action"][starts] = changed["query_scores"][starts]
    other = replay(changed, method)
    assert other["prewrite_correction"].tobytes() == original["prewrite_correction"].tobytes()
    assert other["epistemic_variance"].tobytes() == original["epistemic_variance"].tobytes()
    assert other["action_scores"][cache["key_mask"]].tobytes() == original["action_scores"][cache["key_mask"]].tobytes()
    assert other["work_counts"] == original["work_counts"]


@pytest.mark.parametrize("method", ("last_error", "trace_delta", "rls_full", "rls_diagonal"))
def test_query_correction_is_prewrite_and_later_labels_cannot_change_earlier_reads(method):
    cache = fabricated_cache((14,))
    original = replay(cache, method)
    changed = copy.deepcopy(cache)
    changed["query_scores"][8] += [16., -8., 4., -12.]
    for name in ("base_action", "joint_action"):
        changed[name][8] = changed["query_scores"][8]
    other = replay(changed, method)
    assert original["prewrite_correction"][:9].tobytes() == other["prewrite_correction"][:9].tobytes()
    assert original["action_scores"][:8].tobytes() == other["action_scores"][:8].tobytes()
    assert not np.array_equal(original["prewrite_correction"][9:], other["prewrite_correction"][9:])
    # Since base_action at an actual query equals the answer, mistakenly using
    # it instead of the full shadow forecast makes all residual targets zero.
    wrong_shadow = copy.deepcopy(cache)
    wrong_shadow["shadow_prior"][cache["prior_mask"]] = cache["query_scores"][cache["prior_mask"]]
    wrong = replay(wrong_shadow, method)
    assert not np.any(wrong["prewrite_correction"])
    assert np.any(original["prewrite_correction"])


@pytest.mark.parametrize("method", contract.METHODS)
def test_query_bits_zero_support_episodes_fresh_state_and_owned_outputs(method):
    cache = fabricated_cache((1, 4, 5, 9))
    before = cache_bytes(cache)
    result = replay(cache, method)
    repeated = replay(cache, method)
    assert cache_bytes(cache) == before
    assert result["action_scores"][cache["query_mask"]].tobytes() == cache["query_scores"][cache["query_mask"]].tobytes()
    for name in ("action_scores", "prewrite_correction", "epistemic_variance"):
        assert result[name].tobytes() == repeated[name].tobytes()
        assert not np.shares_memory(result[name], repeated[name])
        for source in cache.values():
            if isinstance(source, np.ndarray):
                assert not np.shares_memory(result[name], source)
    assert not result["prewrite_correction"][:10].any()
    for low, high in zip(cache["episode_offsets"][:-1], cache["episode_offsets"][1:], strict=True):
        single = {name: value[low:high].copy() if isinstance(value, np.ndarray) and name != "episode_offsets"
                  else copy.deepcopy(value) for name, value in cache.items()}
        single["episode_offsets"] = np.array([0, high - low], np.int64)
        isolated = replay(single, method)
        for name in ("action_scores", "prewrite_correction", "epistemic_variance"):
            assert result[name][low:high].tobytes() == isolated[name].tobytes(), name
    if method in ("pretrained", "joint_aux"):
        name = "joint_action" if method == "joint_aux" else "base_action"
        assert result["action_scores"].tobytes() == cache[name].tobytes()
    for name in ("action_scores", "prewrite_correction", "epistemic_variance"):
        result[name][:] = 123
    assert cache_bytes(cache) == before


def test_float32_inputs_are_promoted_before_regression_and_action_is_rounded_only_once():
    cache = fabricated_cache((6,))
    cache["cues"][1:] = 0
    cache["cues"][1:, 0] = 1
    cache["shadow_prior"][4] = 0
    cache["query_scores"][4] = np.array([2.**-23, -2.**-23, -2.**-47, 0.], np.float32)
    for name in ("base_action", "joint_action"):
        cache[name][4] = cache["query_scores"][4]
    cache["base_action"][5] = [1., 0., 0., 0.]
    result = replay(cache, "rls_full", 1.)
    assert result["prewrite_correction"][5, 0] == 2.**-30 + 2.**-56
    assert result["action_scores"][5, 0] == np.nextafter(np.float32(1), np.float32(2))
    early_round = np.float32(1) + np.float32(64) * np.float32(result["prewrite_correction"][5, 0])
    assert early_round == np.float32(1)


@pytest.mark.parametrize("method", contract.METHODS)
def test_selected_work_counts_and_state_allocation_follow_schedule_not_cache_counters(method):
    lengths = (1, 4, 5, 9, 33)
    cache = fabricated_cache(lengths)
    cache["work_counts"] = {"unrelated_cached_work": 10**9}
    result = replay(cache, method)
    counts = result["work_counts"]
    rows, episodes = sum(lengths), len(lengths)
    queries = sum((length - 1) // 4 + 1 for length in lengths)
    writes, keys, nonqueries = queries - episodes, rows - episodes, rows - queries
    for name, expected in {"rows": rows, "episodes": episodes, "query_rows": queries,
                           "first_query_rows": episodes, "key_rows": keys, "later_query_rows": writes,
                           "nonquery_rows": nonqueries, "cache_validation_calls": 1,
                           "cache_rows_validated": rows, "base_rows_copied": rows,
                           "query_rows_copied": queries}.items():
        assert counts[name] == expected, name
    for name in ("neural_model_calls", "projection_calls", "optimizer_calls", "teacher_calls", "native_calls"):
        assert counts[name] == 0
    assert all(type(value) is int and value >= 0 for value in counts.values())
    sizes = {"pretrained": 0, "joint_aux": 0, "last_error": 32, "trace_delta": 256,
             "rls_full": 768, "rls_diagonal": 320, "rls_shrink_025": 768,
             "rls_shrink_050": 768, "rls_shrink_075": 768}
    assert result["state_bytes"] == sizes[method]
    residual = method not in ("pretrained", "joint_aux")
    assert counts["state_initializations"] == (episodes if residual else 0)
    assert counts["residual_target_rows"] == (writes if residual else 0)
    assert counts["residual_target_coordinates"] == (4 * writes if residual else 0)
    assert counts["applied_correction_rows"] == (keys if residual else 0)
    assert counts["applied_correction_coordinates"] == (4 * keys if residual else 0)
    assert counts["action_score_rows"] == (nonqueries if residual else 0)
    assert counts["float32_action_roundings"] == (4 * nonqueries if residual else 0)
    if method in contract.RLS_METHODS:
        covariance_entries = 8 if method == "rls_diagonal" else 64
        assert counts["bayesian_initializations"] == episodes
        assert counts["bayesian_predict_calls"] == nonqueries
        assert counts["bayesian_observe_calls"] == writes
        assert counts["bayesian_direction_terms"] == covariance_entries * keys
        assert counts["bayesian_variance_terms"] == 8 * keys
        assert counts["bayesian_mean_read_terms"] == 32 * keys
        assert counts["bayesian_mean_update_coordinates"] == 32 * writes
        assert counts["bayesian_covariance_update_entries"] == covariance_entries * writes
        assert counts["bayesian_covariance_validations"] == episodes + writes
        assert counts["bayesian_cholesky_calls"] == (0 if method == "rls_diagonal" else episodes + writes)
    elif method == "last_error":
        assert counts["last_error_decays"] == counts["last_error_reads"] == keys
        assert counts["last_error_writes"] == writes
    elif method == "trace_delta":
        assert counts["delta_matrix_decays"] == counts["delta_matrix_reads"] == keys
        assert counts["delta_matrix_writes"] == writes
        assert counts["delta_matrix_read_terms"] == 32 * keys
        assert counts["delta_matrix_write_coordinates"] == 32 * writes


@pytest.mark.parametrize("method,tau", (("rls_full", None), ("rls_full", 0), ("rls_full", .2),
    ("rls_full", True), ("rls_full", float("nan")), ("rls_full", float("inf")),
    ("trace_delta", 1.), ("joint_aux", .1), ("rls_shrink_100", 1.)))
def test_invalid_method_or_ratio_cannot_initialize_or_mutate_a_cache(method, tau, monkeypatch):
    cache = fabricated_cache((9,))
    before = cache_bytes(cache)

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid declaration initialized an estimator")

    monkeypatch.setattr(replay_module, "BayesianScoreMemory", forbidden)
    with pytest.raises((ValueError, TypeError)):
        replay_module.replay(cache, method, tau=tau)
    assert cache_bytes(cache) == before


@pytest.mark.parametrize("defect", ("mask", "cue_nan", "shadow_nan", "query_nan", "query_bits", "offsets",
                                   "cue_dtype", "extra", "negative_work"))
def test_invalid_cache_is_rejected_before_initializing_an_estimator(defect, monkeypatch):
    cache = fabricated_cache((9,))
    if defect == "mask":
        cache["prior_mask"][0] = True
    elif defect == "cue_nan":
        cache["cues"][-1, 0] = np.nan
    elif defect == "shadow_nan":
        cache["shadow_prior"][8, 0] = np.nan
    elif defect == "query_nan":
        cache["query_scores"][8, 0] = np.nan
    elif defect == "query_bits":
        cache["base_action"][0, 0] = np.float32(0.)
    elif defect == "offsets":
        cache["episode_offsets"][-1] -= 1
    elif defect == "cue_dtype":
        cache["cues"] = cache["cues"].astype(np.float64)
    elif defect == "extra":
        cache["targets"] = np.zeros((9, 4), np.float32)
    else:
        cache["work_counts"] = {"bad": -1}
    before = cache_bytes(cache)

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid cache initialized an estimator")

    monkeypatch.setattr(replay_module, "BayesianScoreMemory", forbidden)
    with pytest.raises((ValueError, TypeError)):
        replay(cache, "rls_full")
    assert cache_bytes(cache) == before


@pytest.mark.parametrize("method", ("last_error", "trace_delta", "rls_full", "rls_diagonal"))
def test_final_float32_overflow_rejects_without_altering_cache_or_silently_resetting(method):
    cache = fabricated_cache((6,))
    cache["cues"][1:] = 0
    cache["cues"][1:, 0] = 1
    maximum = np.finfo(np.float32).max
    cache["query_scores"][4] = np.array([maximum, -maximum, maximum, -maximum], np.float32)
    cache["shadow_prior"][4] = -cache["query_scores"][4]
    for name in ("base_action", "joint_action"):
        cache[name][4] = cache["query_scores"][4]
    cache["base_action"][5] = cache["query_scores"][4]
    before = cache_bytes(cache)
    for _ in range(2):
        with pytest.raises((ValueError, FloatingPointError)):
            replay(cache, method)
        assert cache_bytes(cache) == before
    clean = fabricated_cache((6,))
    assert np.isfinite(replay(clean, method)["action_scores"]).all()
