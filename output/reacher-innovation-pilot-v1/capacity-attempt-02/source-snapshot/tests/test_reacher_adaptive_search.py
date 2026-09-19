"""Synthetic analytic scores only; no prospective robot or model evaluation."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from openjev.research.reacher_adaptive_search import ANCHORS, MIN_STD, SearchInputs, search


def arrays(n=2, chunks=4):
    rng = np.random.default_rng(817)
    return (rng.normal(size=(n, 64, chunks, 2)),
            rng.normal(size=(n, 192, chunks, 2)),
            tuple(rng.normal(size=(n, k, chunks, 2)) for k in (64, 64, 63)))


def inputs(n=2, chunks=4):
    return SearchInputs(*arrays(n, chunks))


def quadratic(bank):
    return -np.square(bank.astype(np.float64) - .13).sum(axis=(2, 3))


@pytest.mark.parametrize("method,sizes", [("rs64", [64]), ("rs256", [256]),
                                           ("cem256", [64, 64, 64, 64])])
def test_exact_evaluation_budget_and_complete_evidence(method, sizes):
    calls = []

    def scorer(bank):
        calls.append(bank.copy())
        assert bank.dtype == np.float32
        assert not bank.flags.writeable
        return quadratic(bank)

    result = search(method, inputs(), scorer, step=0)
    assert [b.shape[1] for b in calls] == sizes
    count = sum(sizes)
    np.testing.assert_array_equal(np.concatenate(calls, axis=1), result.sequences)
    np.testing.assert_array_equal(quadratic(result.sequences), result.scores)
    assert result.candidate_evaluations_per_case == count
    assert result.candidate_evaluations == 2 * count
    assert result.imagined_transitions_per_case == count * 12
    assert result.imagined_transitions == 2 * count * 12
    assert [(s.start, s.stop) for s in result.stages] == list(zip(
        np.cumsum([0, *sizes[:-1]]), np.cumsum(sizes), strict=True))
    assert len(set(result.candidate_ids)) == count
    for i, selected in enumerate(result.selected_ids):
        np.testing.assert_array_equal(result.selected_sequences[i], result.sequences[i, selected])
        np.testing.assert_array_equal(result.selected_actions[i], result.sequences[i, selected, 0])


def test_all_three_arms_share_exact_initial_bank_with_original_anchor_and_scale_allocation():
    draws = inputs()
    results = [search(method, draws, quadratic, step=0) for method in ("rs64", "rs256", "cem256")]
    for result in results[1:]:
        np.testing.assert_array_equal(result.sequences[:, :64], results[0].sequences)
        np.testing.assert_array_equal(result.scores[:, :64], results[0].scores)
        assert result.input_identities == results[0].input_identities
    expected = draws.initial * np.r_[np.full(32, .25), np.full(32, .75)][None, :, None, None]
    expected = np.clip(expected, -1, 1).astype(np.float32)
    for i, anchor in enumerate(ANCHORS):
        expected[:, i] = anchor
    np.testing.assert_array_equal(results[0].sequences, np.repeat(expected, 3, axis=2))
    extra = draws.random_extra * np.r_[np.full(96, .25), np.full(96, .75)][None, :, None, None]
    expected_extra = np.repeat(np.clip(extra, -1, 1).astype(np.float32), 3, axis=2)
    np.testing.assert_array_equal(results[1].sequences[:, 64:], expected_extra)
    stage = results[1].stages[0]
    assert stage.innovation_source == "initial+random_extra"
    assert stage.innovation_count == 256
    assert stage.fixed_scales.shape == (256,)
    assert results[1].candidate_ids[:64] == results[0].candidate_ids
    assert results[1].candidate_ids[64:] == tuple(f"random_extra/{i}" for i in range(192))


def test_final_mean_is_paid_candidate_255_and_is_quantized():
    result = search("cem256", inputs(), quadratic, step=0)
    assert [s.innovation_count for s in result.stages] == [64, 64, 64, 63]
    assert result.stages[-1].mean_candidate_id == 255
    assert all(s.mean_candidate_id is None for s in result.stages[:-1])
    expected = np.repeat(result.stages[-1].proposal_mean.astype(np.float32), 3, axis=1)
    np.testing.assert_array_equal(result.sequences[:, 255], expected)
    assert result.candidate_ids[-1] == "cem/3/mean"
    np.testing.assert_array_equal(result.scores[:, 255], quadratic(expected[:, None])[:, 0])


def test_paid_mean_can_win_without_an_extra_callback():
    calls = 0

    def scorer(bank):
        nonlocal calls
        calls += 1
        values = np.zeros(bank.shape[:2])
        if calls == 4:
            values[:, -1] = 10.
        return values

    result = search("cem256", inputs(), scorer, step=0)
    assert calls == 4
    np.testing.assert_array_equal(result.selected_ids, [255, 255])
    np.testing.assert_array_equal(result.selected_sequences, result.sequences[:, 255])


@pytest.mark.parametrize("method", ["rs64", "rs256", "cem256"])
def test_case_batching_does_not_change_proposals_or_selected_commands(method):
    draws = inputs()
    batched = search(method, draws, quadratic, step=0)
    for case in range(2):
        isolated = SearchInputs(draws.initial[case:case + 1], draws.random_extra[case:case + 1],
                                tuple(v[case:case + 1] for v in draws.cem))
        result = search(method, isolated, quadratic, step=0)
        np.testing.assert_array_equal(result.sequences[0], batched.sequences[case])
        np.testing.assert_array_equal(result.scores[0], batched.scores[case])
        np.testing.assert_array_equal(result.selected_actions[0], batched.selected_actions[case])


def test_unused_future_blocks_do_not_affect_shortened_horizon_search():
    original = inputs()
    initial, extra, cem = arrays()
    for value in (initial, extra, *cem):
        value[:, :, 1:] = 17.
    changed = SearchInputs(initial, extra, cem)
    left = search("cem256", original, quadratic, step=47)
    right = search("cem256", changed, quadratic, step=47)
    np.testing.assert_array_equal(left.sequences, right.sequences)
    np.testing.assert_array_equal(left.scores, right.scores)
    assert left.input_identities != right.input_identities


def test_refit_uses_current_generation_eight_elites_of_quantized_commands_independently_per_case():
    def scorer(bank):
        target = np.array([.28, -.29])[:, None, None, None]
        return -np.square(bank.astype(np.float64) - target).sum(axis=(2, 3))

    result = search("cem256", inputs(), scorer, step=0)
    for stage in result.stages[1:]:
        previous = slice(stage.start - 64, stage.start)
        local = np.argsort(-result.scores[:, previous], axis=1, kind="stable")[:, :8]
        global_ids = local + stage.start - 64
        np.testing.assert_array_equal(stage.source_elite_ids, global_ids)
        elite = result.sequences[np.arange(2)[:, None], global_ids, ::3].astype(np.float64)
        np.testing.assert_array_equal(stage.proposal_mean, elite.mean(axis=1))
        np.testing.assert_array_equal(stage.proposal_std, np.maximum(elite.std(axis=1), MIN_STD))
    assert not np.array_equal(result.stages[1].proposal_mean[0], result.stages[1].proposal_mean[1])


def test_global_best_survives_worse_later_generations_and_ties_use_earliest_id():
    call_count = 0

    def scorer(bank):
        nonlocal call_count
        values = np.full(bank.shape[:2], -float(call_count))
        if call_count == 0:
            values[0, [5, 9]] = 2.
            values[1, [12, 20]] = 3.
        call_count += 1
        return values

    result = search("cem256", inputs(), scorer, step=0)
    np.testing.assert_array_equal(result.selected_ids, [5, 12])
    np.testing.assert_array_equal(result.stages[1].source_elite_ids[0], [5, 9, 0, 1, 2, 3, 4, 6])


def test_collapsed_elites_get_exact_numerical_floor():
    initial, extra, cem = arrays(n=1)
    initial[:] = 0.
    result = search("cem256", SearchInputs(initial, extra, cem),
                    lambda bank: -np.square(bank).sum(axis=(2, 3)), step=0)
    np.testing.assert_array_equal(result.stages[1].proposal_mean, np.zeros((1, 4, 2)))
    np.testing.assert_array_equal(result.stages[1].proposal_std, np.full((1, 4, 2), MIN_STD))


@pytest.mark.parametrize("step,horizon", [(0, 12), (38, 12), (39, 11), (47, 3), (49, 1)])
def test_terminal_shortening_and_partial_action_blocks(step, horizon):
    result = search("cem256", inputs(), quadratic, step=step)
    assert result.horizon == horizon
    assert result.sequences.shape == (2, 256, horizon, 2)
    assert result.imagined_transitions_per_case == 256 * horizon
    for t in range(horizon):
        np.testing.assert_array_equal(result.sequences[:, :, t], result.sequences[:, :, (t // 3) * 3])
    assert result.stages[-1].proposal_mean.shape == (2, (horizon + 2) // 3, 2)


def test_no_extra_model_transition_is_charged_for_final_mean_or_best_selection():
    total = sum(search("cem256", inputs(n=1), quadratic, step=t).imagined_transitions for t in range(50))
    assert total == 136704


def test_clipping_precedes_scoring_and_all_commands_use_float32():
    initial, extra, cem = arrays(n=1)
    initial[:, 8] = 1e300
    initial[:, 9] = -1e300
    initial[:, 10] = .123456789123456789
    result = search("rs64", SearchInputs(initial, extra, cem), quadratic, step=0)
    assert result.sequences.dtype == np.float32
    np.testing.assert_array_equal(result.sequences[:, 8], np.ones((1, 12, 2), np.float32))
    np.testing.assert_array_equal(result.sequences[:, 9], -np.ones((1, 12, 2), np.float32))
    assert (result.sequences[:, 10] == np.float32(.25 * initial[0, 10, 0, 0])).all()
    assert np.all((-1 <= result.sequences) & (result.sequences <= 1))


def test_inputs_callback_banks_and_outputs_cannot_be_mutated_and_source_arrays_are_untouched():
    initial, extra, cem = arrays()
    initial_before, extra_before, cem_before = initial.copy(), extra.copy(), tuple(v.copy() for v in cem)
    draws = SearchInputs(initial, extra, cem)

    def scorer(bank):
        with pytest.raises(ValueError):
            bank.setflags(write=True)
        return quadratic(bank)

    result = search("cem256", draws, scorer, step=0)
    np.testing.assert_array_equal(initial, initial_before)
    np.testing.assert_array_equal(extra, extra_before)
    for actual, expected in zip(cem, cem_before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    initial[:] = 991
    np.testing.assert_array_equal(draws.initial, initial_before)
    for value in (draws.initial, draws.random_extra, *draws.cem, result.scores,
                  result.sequences, result.selected_actions, result.selected_sequences,
                  result.selected_ids, result.stages[-1].proposal_mean,
                  result.stages[-1].proposal_std, result.stages[-1].source_elite_ids):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        draws.initial = initial


def test_callback_scores_are_copied_even_when_callback_reuses_a_buffer():
    buffer = np.zeros((2, 64))
    call = 0

    def scorer(bank):
        nonlocal call
        buffer[:] = call
        call += 1
        return buffer

    result = search("cem256", inputs(), scorer, step=0)
    buffer[:] = -999
    np.testing.assert_array_equal(result.scores[0], np.repeat(np.arange(4), 64))
    np.testing.assert_array_equal(result.selected_ids, [192, 192])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_draws_fail_before_any_callback(bad):
    initial, extra, cem = arrays()
    cem[-1][0, 0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite real"):
        SearchInputs(initial, extra, cem)


@pytest.mark.parametrize("bad", [np.zeros((2, 63)), np.zeros((2, 64, 1)), np.zeros((64,)),
                                 np.ones((2, 64), bool), np.zeros((2, 64), complex),
                                 np.full((2, 64), np.nan), np.full((2, 64), np.inf),
                                 np.full((2, 64), "0")])
def test_malformed_callback_outputs_fail_without_retry(bad):
    calls = 0

    def scorer(bank):
        nonlocal calls
        calls += 1
        return bad

    with pytest.raises(ValueError):
        search("cem256", inputs(), scorer, step=0)
    assert calls == 1


@pytest.mark.parametrize("which,wrong", [(0, np.zeros((2, 63, 4, 2))),
                                       (0, np.zeros((0, 64, 4, 2))),
                                       (0, np.zeros((2, 64, 0, 2))),
                                       (0, np.zeros((2, 64, 4, 3))),
                                       (1, np.zeros((1, 192, 4, 2))),
                                       (2, (np.zeros((2, 64, 4, 2)),) * 3),
                                       (2, [np.zeros((2, k, 4, 2)) for k in (64, 64, 63)]),
                                       (2, ())])
def test_malformed_innovation_membership_is_rejected(which, wrong):
    values = list(arrays())
    values[which] = wrong
    with pytest.raises(ValueError):
        SearchInputs(*values)


@pytest.mark.parametrize("kwargs", [{"step": 50}, {"step": -1}, {"step": True},
                                    {"step": 1.0}, {"steps": 0}, {"steps": False},
                                    {"planning_horizon": 0}, {"planning_horizon": 13},
                                    {"action_block": 0}, {"action_block": 1.5}])
def test_invalid_boundaries_fail_before_scoring(kwargs):
    def scorer(bank):
        pytest.fail("invalid plan reached scoring")

    with pytest.raises(ValueError):
        search("rs64", inputs(), scorer, **({"step": 0} | kwargs))


def test_unknown_method_noncallable_and_wrong_input_type_fail():
    with pytest.raises(ValueError, match="method"):
        search("ic em", inputs(), quadratic, step=0)
    with pytest.raises(TypeError, match="callable"):
        search("rs64", inputs(), 9, step=0)
    with pytest.raises(TypeError, match="SearchInputs"):
        search("rs64", None, quadratic, step=0)


def test_callback_exception_is_terminal_and_not_converted_to_scores():
    def scorer(bank):
        raise RuntimeError("deliberate scoring failure")

    with pytest.raises(RuntimeError, match="deliberate scoring failure"):
        search("cem256", inputs(), scorer, step=0)


def test_input_identities_bind_unused_innovations_and_shapes():
    original = inputs()
    initial, extra, cem = arrays()
    extra[0, 0, 0, 0] += 1
    changed = SearchInputs(initial, extra, cem)
    left = search("rs64", original, quadratic, step=0)
    right = search("rs64", changed, quadratic, step=0)
    np.testing.assert_array_equal(left.sequences, right.sequences)
    assert dict(left.input_identities)["random_extra"] != dict(right.input_identities)["random_extra"]
    assert len(left.input_identities) == 5
