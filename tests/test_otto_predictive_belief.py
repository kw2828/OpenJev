"""Small fabricated categorical grids only; no native code or empirical data."""
from __future__ import annotations

import inspect
from fractions import Fraction

import numpy as np
import pytest

from openjev.research import otto_predictive_belief as m


def fixture():
    belief = np.array([.125, .25, .375, .25], np.float64)
    positions = np.array([1, 2, 1, 0], np.int64)
    sensor = np.array([[.5, .25, .125, .125], [.25, .5, .125, .125],
                       [.125, .125, .5, .25], [.125, .25, .125, .5]], np.float64)
    likelihoods = np.stack([np.roll(sensor, h, axis=1) for h in range(len(positions))])
    for likelihood, position in zip(likelihoods, positions, strict=True):
        likelihood[position] = 0.
    return belief, likelihoods, positions


def enumerated_tree(belief, likelihoods, positions):
    """Independent rational enumeration of every source and observation branch."""
    totals = [[Fraction(0) for _ in range(5)] for _ in positions]

    def branch(source, horizon, weight, terminated):
        if horizon == len(positions):
            return
        if terminated or source == positions[horizon]:
            totals[horizon][4] += weight
            branch(source, horizon + 1, weight, True)
        else:
            for odor in range(4):
                evidence = weight * Fraction(float(likelihoods[horizon, source, odor]))
                totals[horizon][odor] += evidence
                branch(source, horizon + 1, evidence, False)

    for source, probability in enumerate(belief):
        branch(source, 0, Fraction(float(probability)), False)
    return np.array([[float(value) for value in row] for row in totals], np.float64)


def test_cdf_law_matches_sampler_interval_boundaries_and_records_rounding():
    source = np.array([.5, 1e-20, .25, 0., .25], np.float64)
    snapshot = source.tobytes()
    result = m.cdf_law(source)
    np.testing.assert_array_equal(result["probabilities"], [.5, 0., .25, 0., .25])
    assert result["raw_mass"] == 1. and result["cdf_mass"] == 1.
    assert result["max_abs_adjustment"] == 1e-20
    assert result["uniform_bits"] == 53 and result["max_cdf_quantization_adjustment"] == 0.
    assert result["cdf_positive_entries_rounded_to_zero"] == 1
    assert result["positive_entries_rounded_to_zero"] == 1
    assert source.tobytes() == snapshot and not np.shares_memory(source, result["probabilities"])
    # searchsorted(side=right) assigns boundaries to the next positive interval.
    cumulative = np.cumsum(source)
    cumulative /= cumulative[-1]
    assert [np.searchsorted(cumulative, u, side="right") for u in (0., .5, .75)] == [0, 2, 4]
    slightly_short = np.array([.2, .3, .5], np.float64) * (1 - 5e-11)
    repaired = m.cdf_law(slightly_short)
    boundaries = np.cumsum(slightly_short) / np.cumsum(slightly_short)[-1]
    # Independent rational ceil counts of default 53-bit grid points.
    endpoints = []
    for boundary in boundaries:
        value = Fraction(float(boundary)) * 2**53
        endpoints.append(-(-value.numerator // value.denominator))
    expected = np.array([end - start for start, end in zip([0] + endpoints[:-1], endpoints, strict=True)],
                        np.float64) / 2**53
    np.testing.assert_array_equal(repaired["probabilities"], expected)
    assert repaired["raw_mass"] == float(slightly_short.sum())
    assert repaired["cdf_mass"] == float(np.cumsum(slightly_short)[-1])
    np.testing.assert_allclose(m.normalize_prior(repaired["probabilities"])["belief"], expected, rtol=0, atol=1e-16)


@pytest.mark.parametrize("source", [np.array([.1, .2, .3, .4]), np.array([.125, .25, 0., .625]),
                                     np.array([.126, .001, .873]), np.array([0., 1., 0.])])
def test_three_bit_cdf_law_equals_complete_independent_uniform_grid_enumeration(source):
    result = m.cdf_law(source, uniform_bits=3)
    cumulative = np.cumsum(source)
    cumulative /= cumulative[-1]
    observed = np.zeros(len(source), np.int64)
    for integer in range(8):
        uniform = Fraction(integer, 8)
        # Strict comparison, including exact equality with dyadic boundaries.
        selected = next(i for i, boundary in enumerate(cumulative) if uniform < Fraction(float(boundary)))
        observed[selected] += 1
    np.testing.assert_array_equal(result["probabilities"], observed / 8)
    continuous = np.diff(np.r_[0., cumulative])
    assert result["max_cdf_quantization_adjustment"] == float(np.abs(observed / 8 - continuous).max())
    assert result["positive_entries_rounded_to_zero"] == int(np.count_nonzero((source > 0) & (observed == 0)))
    assert result["uniform_bits"] == 3 and result["probabilities"].sum() == 1.


def test_finite_grid_can_remove_positive_cdf_interval_without_clipping():
    source = np.array([.126, .001, .873], np.float64)
    result = m.cdf_law(source, uniform_bits=3)
    np.testing.assert_array_equal(result["probabilities"], [.25, 0., .75])
    assert result["cdf_positive_entries_rounded_to_zero"] == 0
    assert result["positive_entries_rounded_to_zero"] == 1


@pytest.mark.parametrize("bits", [True, 0, -1, 54, 3., "53", np.nan])
def test_uniform_grid_bit_width_is_explicit_and_bounded(bits):
    with pytest.raises(ValueError, match="uniform_bits"):
        m.cdf_law(np.array([.5, .5]), uniform_bits=bits)


@pytest.mark.parametrize("source", [np.array([0., 0.]), np.array([.1, .2]), np.array([1., -1e-20]),
                                     np.array([np.nan, 1.]), np.array([np.inf]), np.array([], np.float64),
                                     np.array([.5, .5], np.float32)])
def test_cdf_law_rejects_zero_malformed_or_materially_subnormalized_inputs(source):
    with pytest.raises(ValueError):
        m.cdf_law(source)


def test_blind_marginals_equal_complete_rational_source_observation_tree():
    belief, likelihoods, positions = fixture()
    result = m.blind_marginals(belief, likelihoods, positions)
    np.testing.assert_allclose(result, enumerated_tree(belief, likelihoods, positions), rtol=0, atol=1e-15)
    np.testing.assert_array_equal(result[:, 4], [.25, .625, .625, .75])
    np.testing.assert_allclose(result.sum(axis=1), 1., rtol=0, atol=1e-15)
    np.testing.assert_array_equal(result[0], m.one_step(belief, likelihoods[0], positions[0]))


def test_blind_repeated_single_position_does_not_recount_found_or_condition_on_odor():
    belief, likelihoods, _ = fixture()
    repeated = np.repeat(likelihoods[:1], 5, axis=0)
    result = m.blind_marginals(belief, repeated, np.ones(5, np.int64))
    np.testing.assert_array_equal(result, np.repeat(result[:1], 5, axis=0))
    assert result[0, 4] == .25
    assert tuple(inspect.signature(m.blind_marginals).parameters) == ("belief", "likelihoods", "positions")
    with pytest.raises(TypeError):
        m.blind_marginals(belief, repeated, np.ones(5, np.int64), known_found=True)


def test_every_source_visited_makes_blind_suffix_exactly_absorbing_without_labels():
    belief, _, _ = fixture()
    positions = np.array([0, 1, 2, 3, 0], np.int64)
    likelihoods = np.full((5, 4, 4), .25, np.float64)
    result = m.blind_marginals(belief, likelihoods, positions)
    np.testing.assert_array_equal(result[-2:], [[0, 0, 0, 0, 1], [0, 0, 0, 0, 1]])


def test_one_step_found_exclusion_and_posterior_exact_rational_oracle():
    belief, likelihoods, positions = fixture()
    # At position1, odor0 has masses [1/16,0,3/64,1/32].
    result = m.observe(belief, likelihoods[0], positions[0], 0)
    assert result["evidence"] == 9 / 64
    np.testing.assert_array_equal(result["belief"], [4 / 9, 0, 1 / 3, 2 / 9])
    assert result["known_terminal"] is False
    found = m.observe(belief, likelihoods[0], positions[0], known_found=True)
    assert found["evidence"] == .25 and found["known_terminal"] is True
    np.testing.assert_array_equal(found["belief"], [0, 1, 0, 0])


def test_normal_posterior_mixture_recovers_blind_second_step_distribution():
    belief, likelihoods, positions = fixture()
    first = m.one_step(belief, likelihoods[0], positions[0])
    average = np.zeros(5, np.float64)
    average[4] = first[4]
    for odor in range(4):
        posterior = m.observe(belief, likelihoods[0], positions[0], odor)
        assert posterior["evidence"] == first[odor]
        average += first[odor] * m.one_step(posterior["belief"], likelihoods[1], positions[1])
    np.testing.assert_allclose(average, m.blind_marginals(belief, likelihoods, positions)[1], rtol=0, atol=1e-15)


def test_known_terminal_is_explicit_not_inferred_from_point_mass():
    belief = np.array([0., 1.], np.float64)
    likelihood = np.array([[0., 0., 0., 0.], [.25, .25, .25, .25]], np.float64)
    np.testing.assert_array_equal(m.one_step(belief, likelihood, 0), [.25, .25, .25, .25, 0])
    np.testing.assert_array_equal(m.one_step(belief, likelihood, 0, known_terminal=True), [0, 0, 0, 0, 1])
    with pytest.raises(ValueError, match="explicit boolean"):
        m.one_step(belief, likelihood, 0, known_terminal=1)


def test_initial_roundoff_correction_is_bounded_recorded_and_does_not_remove_small_support():
    source = np.array([1e-20, .4, .6], np.float64) * (1 - 5e-11)
    before = source.tobytes()
    result = m.normalize_prior(source)
    assert result["input_mass"] == float(source.sum())
    assert result["normalization_factor"] == 1 / result["input_mass"]
    assert result["mass_tolerance"] == 1e-10
    assert result["correction_l1"] == float(np.abs(result["belief"] - source).sum())
    assert result["belief"][0] > 0 and result["belief"].sum() == 1.
    assert source.tobytes() == before and not np.shares_memory(source, result["belief"])
    with pytest.raises(ValueError, match="correction tolerance"):
        m.normalize_prior(source, mass_tolerance=0.)


@pytest.mark.parametrize("prior", [np.array([.2, .3]), np.array([0., 0.]), np.array([1e-12, 2e-12]),
                                    np.array([1., -1e-15]), np.array([np.nan, 1.]), np.array([np.inf]),
                                    np.array([], np.float64), np.array([.5, .5], np.float32)])
def test_invalid_or_legacy_subnormalized_initial_priors_fail(prior):
    with pytest.raises(ValueError):
        m.normalize_prior(prior)


@pytest.mark.parametrize("tolerance", [True, -1., 1e-9, np.inf, np.nan, "1e-10"])
def test_correction_tolerance_cannot_silently_widen(tolerance):
    with pytest.raises(ValueError):
        m.normalize_prior(np.array([.5, .5]), mass_tolerance=tolerance)


def test_positive_subfloor_evidence_normalizes_and_zero_evidence_fails():
    belief = np.array([.25, .25, .5], np.float64)
    likelihood = np.array([[0., 0., 0., 0.], [1e-20, .5, 0., .5 - 1e-20],
                           [2e-20, .5, 0., .5 - 2e-20]], np.float64)
    result = m.observe(belief, likelihood, 0, 0)
    assert 0 < result["evidence"] < 1e-10
    np.testing.assert_allclose(result["belief"], [0., .2, .8], rtol=1e-15, atol=0)
    assert result["belief"].sum() == pytest.approx(1., abs=1e-15)
    with pytest.raises(ValueError, match="zero-evidence odor"):
        m.observe(belief, likelihood, 0, 2)
    with pytest.raises(ValueError, match="zero-evidence found"):
        m.observe(np.array([0., .5, .5]), likelihood, 0, known_found=True)


def test_product_underflow_fails_without_support_repair():
    belief = np.array([.25, .75], np.float64)
    likelihood = np.array([[np.nextafter(0., 1.), 1., 0., 0.], [0., 0., 0., 0.]], np.float64)
    with pytest.raises(FloatingPointError, match="underflowed"):
        m.observe(belief, likelihood, 1, 0)


@pytest.mark.parametrize("defect", ["float32", "negative", "nan", "above_one", "bad_shape", "sum",
                                    "zero_nonorigin", "partial_origin", "negative_position", "outside_position",
                                    "bool_position", "float_position", "subnormalized_belief"])
def test_one_step_rejects_malformed_probabilities_and_indices(defect):
    belief, likelihoods, _ = fixture()
    likelihood, position = likelihoods[0].copy(), 1
    if defect == "float32":
        likelihood = likelihood.astype(np.float32)
    elif defect == "negative":
        likelihood[0, 0] = -1e-15
    elif defect == "nan":
        likelihood[0, 0] = np.nan
    elif defect == "above_one":
        likelihood[0, 0] = 1.1
    elif defect == "bad_shape":
        likelihood = likelihood[:, :3]
    elif defect == "sum":
        likelihood[0] *= .99
    elif defect == "zero_nonorigin":
        likelihood[0] = 0.
    elif defect == "partial_origin":
        likelihood[1, 0] = .5
    elif defect == "negative_position":
        position = -1
    elif defect == "outside_position":
        position = 4
    elif defect == "bool_position":
        position = True
    elif defect == "float_position":
        position = 1.
    else:
        belief *= .99
    with pytest.raises(ValueError):
        m.one_step(belief, likelihood, position)


@pytest.mark.parametrize("odor,found", [(None, False), (-2, False), (4, False), (True, False),
                                       (1., False), (0, True), (None, 1)])
def test_observation_contract_rejects_sentinel_or_ambiguous_flags(odor, found):
    belief, likelihoods, positions = fixture()
    with pytest.raises(ValueError):
        m.observe(belief, likelihoods[0], positions[0], odor, known_found=found)


def test_all_outputs_owned_inputs_unchanged_and_blind_future_kernel_causal():
    belief, likelihoods, positions = fixture()
    snapshots = [v.tobytes() for v in (belief, likelihoods, positions)]
    result = m.blind_marginals(belief, likelihoods, positions)
    step = m.one_step(belief, likelihoods[0], positions[0])
    posterior = m.observe(belief, likelihoods[0], positions[0], 0)["belief"]
    assert all(not np.shares_memory(out, source)
               for out in (result, step, posterior) for source in (belief, likelihoods, positions))
    changed = likelihoods.copy()
    changed[2:] = np.roll(changed[2:], 1, axis=2)
    np.testing.assert_array_equal(result[:2], m.blind_marginals(belief, changed, positions)[:2])
    result[:] = 0
    step[:] = 0
    posterior[:] = 0
    assert [v.tobytes() for v in (belief, likelihoods, positions)] == snapshots
    with pytest.raises(ValueError):
        m.blind_marginals(belief, likelihoods, positions.astype(np.float64))
    with pytest.raises(ValueError):
        m.blind_marginals(belief, likelihoods[:0], positions[:0])
    with pytest.raises(ValueError):
        m.blind_marginals(belief, likelihoods, np.array([1, 2, 1, 99]))
