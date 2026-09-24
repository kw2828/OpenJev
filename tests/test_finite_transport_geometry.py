"""Independent algebraic fixtures; no saved model or scientific data is used."""

import itertools

import numpy as np
import pytest

from openjev.research import finite_transport_geometry as geometry

HORIZONS = (0, 1, 2, 4, 8)
SPECTRAL_THRESHOLDS = ("1e-8", "1e-6", "1e-4", "1e-2")
GRAM_THRESHOLDS = ("1e-12", "1e-10", "1e-8", "1e-6")


def gamma(length):
    """Standard dot-product error factor for float64 round-to-nearest."""
    unit_roundoff = np.finfo(np.float64).eps / 2
    return length * unit_roundoff / (1 - length * unit_roundoff)


def parameters(*, uniform=False, near_identity=False):
    identity = np.eye(8, dtype=np.float64)
    transition = np.empty((4, 8, 8), dtype=np.float64)
    for action in range(4):
        if uniform:
            transition[action] = np.full((8, 8), 1 / 8)
        else:
            epsilon = 1 / 8 if near_identity else (action + 1) / 8
            permutation = identity if near_identity else np.roll(identity, action + 1, axis=0)
            transition[action] = (1 - epsilon) * permutation + epsilon / 8
    emission = np.full((4, 8), 1 / 8, dtype=np.float64)
    probabilities = np.full((4, 8), 1 / 8, dtype=np.float64)
    for state in range(8):
        emission[state % 4, state] = 5 / 8
        probabilities[(state ^ (state >> 1)) % 4, state] = 5 / 8
    hazard = np.array(
        [[1 / 16 + ((state + 2 * action) % 5) / 32 for state in range(8)]
         for action in range(4)], dtype=np.float64
    )
    if uniform or near_identity:
        hazard.fill(1 / 4)
    return {
        "transition_logits": np.log(transition),
        "emission_logits": np.log(emission),
        "hazard_logits": np.log(hazard / (1 - hazard)),
        "cost_logits": np.log(probabilities),
    }


def costs():
    result = np.full((4, 8), 1 / 8, dtype=np.float64)
    for state in range(8):
        result[(state ^ (state >> 1)) % 4, state] = -3 / 8
    return result


def enumerated_gram(surviving, initial_gram, horizon):
    """Average forward action words, with destination-state matrix convention."""
    total = np.zeros((8, 8), dtype=np.float64)
    for actions in itertools.product(range(4), repeat=horizon):
        forward = np.eye(8, dtype=np.float64)
        for action in actions:
            forward = surviving[action] @ forward
        total += forward.T @ initial_gram @ forward
    return total / (4 ** horizon)


def record(result, channel, horizon):
    return next(row for row in result[channel] if row["horizon"] == horizon)


def test_helmert_basis_spans_only_zero_mass_contrasts_and_is_owned():
    basis = geometry.zero_sum_basis()
    assert basis.shape == (8, 7) and basis.dtype == np.float64
    for column in range(7):
        denominator = np.sqrt((column + 1) * (column + 2))
        expected = np.zeros(8, dtype=np.float64)
        expected[:column + 1] = 1 / denominator
        expected[column + 1] = -(column + 1) / denominator
        np.testing.assert_array_equal(basis[:, column], expected)
    np.testing.assert_allclose(basis.T @ basis, np.eye(7), atol=3e-16, rtol=0)
    np.testing.assert_allclose(basis.sum(0), 0, atol=3e-16, rtol=0)
    np.testing.assert_allclose(basis @ basis.T, np.eye(8) - 1 / 8, atol=3e-16, rtol=0)
    basis[:] = 0
    assert np.linalg.matrix_rank(geometry.zero_sum_basis()) == 7


def test_probability_fields_use_destination_hazard_and_owned_centered_head():
    source = parameters()
    before = {key: value.copy() for key, value in source.items()}
    fields = geometry.probability_fields(source)
    assert set(fields) == {"transition", "emission", "hazard", "costs", "surviving"}
    # Denominator reduction, division, then independent resummation. The
    # positive exponentials cancel algebraically regardless of exp accuracy.
    np.testing.assert_allclose(fields["transition"].sum(1), 1, atol=4 * gamma(8), rtol=0)
    np.testing.assert_allclose(fields["emission"].sum(0), 1, atol=4 * gamma(4), rtol=0)
    # The centered subtraction adds bounded rounding to the same normalization.
    np.testing.assert_allclose(fields["costs"].sum(0), 0, atol=4 * gamma(4), rtol=0)
    np.testing.assert_allclose(fields["costs"], costs(), atol=3e-16, rtol=0)
    for action, destination, origin in itertools.product(range(4), range(8), range(8)):
        expected = ((1 - fields["hazard"][action, destination])
                    * fields["transition"][action, destination, origin])
        assert fields["surviving"][action, destination, origin] == expected
    for key, value in source.items():
        np.testing.assert_array_equal(value, before[key])
        assert all(not np.shares_memory(value, output) for output in fields.values())
    fields["transition"][:] = 0
    assert np.all(geometry.probability_fields(source)["transition"] > 0)


def test_uniform_transition_erases_zero_sum_contrasts_with_constant_hazard():
    report = geometry.analyze_parameters(parameters(uniform=True))
    for item in report["transition"]:
        assert item["dobrushin_delta"] == 0
        np.testing.assert_allclose(item["column_entropies"], np.log(8), atol=1e-15, rtol=0)
        np.testing.assert_allclose(item["contrast_singular_values"], 0, atol=1e-15, rtol=0)
        assert item["contrast_ranks"] == dict.fromkeys(SPECTRAL_THRESHOLDS, 0)
        assert item["row_stochastic_deviation_max"] == 0
        assert item["uniform_image_l1"] == 0
    for channel in ("state", "cost"):
        for horizon in HORIZONS[1:]:
            item = record(report["grams"], channel, horizon)
            np.testing.assert_allclose(item["projected_gram"], 0, atol=1e-15, rtol=0)
            assert item["ranks"] == dict.fromkeys(GRAM_THRESHOLDS, 0)


def test_near_identity_has_closed_form_contraction_and_gram_spectrum():
    report = geometry.analyze_parameters(parameters(near_identity=True))
    contraction = 7 / 8
    survival = 3 / 4
    basis = geometry.zero_sum_basis()
    initial_cost = basis.T @ costs().T @ costs() @ basis
    for item in report["transition"]:
        assert item["dobrushin_delta"] == pytest.approx(contraction, abs=2e-15)
        np.testing.assert_allclose(item["contrast_singular_values"], contraction,
                                   atol=2e-15, rtol=0)
        assert item["contrast_ranks"] == dict.fromkeys(SPECTRAL_THRESHOLDS, 7)
    for horizon in HORIZONS:
        multiplier = (contraction * survival) ** (2 * horizon)
        state = record(report["grams"], "state", horizon)
        head = record(report["grams"], "cost", horizon)
        np.testing.assert_allclose(state["projected_gram"], multiplier * np.eye(7),
                                   atol=3e-15, rtol=2e-14)
        np.testing.assert_allclose(head["projected_gram"], multiplier * initial_cost,
                                   atol=3e-15, rtol=2e-14)


@pytest.mark.parametrize("shift", [0, 3])
def test_exact_identity_and_permutation_helpers_preserve_state_energy(shift):
    permutation = np.roll(np.eye(8), shift, axis=0)
    surviving = np.repeat(permutation[None], 4, axis=0)
    result = geometry.transport_grams(surviving, costs())
    basis = geometry.zero_sum_basis()
    # Two length-8 products versus one have a componentwise bound
    # (3*gamma8 + gamma8**2) * abs(Q).T@abs(Q). Use 4*gamma8 and
    # outward-bound the computed product scale; this is not a bitwise identity.
    product_scale = max(1.0, float(np.max(np.abs(basis).T @ np.abs(basis)))) / (1 - gamma(8))
    projection_tolerance = 4 * gamma(8) * product_scale
    assert result["horizons"] == list(HORIZONS)
    for horizon in HORIZONS:
        forward = np.linalg.matrix_power(permutation, horizon)
        expected = forward.T @ costs().T @ costs() @ forward
        np.testing.assert_allclose(record(result, "cost", horizon)["gram"], expected,
                                   atol=0, rtol=0)
        np.testing.assert_allclose(record(result, "state", horizon)["gram"], np.eye(8),
                                   atol=0, rtol=0)
        np.testing.assert_allclose(record(result, "state", horizon)["projected_gram"],
                                   basis.T @ basis, atol=projection_tolerance, rtol=0)


@pytest.mark.parametrize("horizon", [2, 4])
def test_full_space_recurrence_equals_independent_action_word_enumeration(horizon):
    fields = geometry.probability_fields(parameters())
    result = geometry.transport_grams(fields["surviving"], fields["costs"])
    basis = geometry.zero_sum_basis()
    for channel, initial in (("cost", fields["costs"].T @ fields["costs"]),
                             ("state", np.eye(8))):
        expected = enumerated_gram(fields["surviving"], initial, horizon)
        item = record(result, channel, horizon)
        np.testing.assert_allclose(item["gram"], expected, atol=2e-15, rtol=3e-14)
        np.testing.assert_allclose(item["projected_gram"], basis.T @ expected @ basis,
                                   atol=2e-15, rtol=3e-14)
        eigenvalues = np.linalg.eigvalsh(np.asarray(item["projected_gram"]))
        np.testing.assert_array_equal(item["eigenvalues"], eigenvalues)
        assert item["ranks"] == {threshold: int((eigenvalues > float(threshold)).sum())
                                 for threshold in GRAM_THRESHOLDS}
        assert item["trace"] == pytest.approx(float(np.trace(basis.T @ expected @ basis)),
                                               abs=2e-15, rel=3e-14)


def test_survival_coupling_is_not_erased_by_intermediate_contrast_projection():
    column = np.array([3 / 8, -1 / 8, -1 / 8, -1 / 8], dtype=np.float64)
    head = np.repeat(column[:, None], 8, axis=1)
    survival = np.arange(1, 9, dtype=np.float64) / 8
    surviving = np.repeat(np.diag(survival)[None], 4, axis=0)
    result = geometry.transport_grams(surviving, head)
    basis = geometry.zero_sum_basis()
    np.testing.assert_allclose(record(result, "cost", 0)["projected_gram"], 0,
                               atol=2e-16, rtol=0)
    for horizon in HORIZONS[1:]:
        projected = basis.T @ (survival ** horizon)
        expected = float(column @ column) * np.outer(projected, projected)
        actual = record(result, "cost", horizon)
        np.testing.assert_allclose(actual["projected_gram"], expected, atol=3e-16, rtol=2e-14)
        assert actual["trace"] > 0


def test_simultaneous_latent_permutation_preserves_all_geometry_spectra():
    source = parameters()
    permutation = np.array([3, 0, 6, 2, 7, 1, 5, 4])
    transformed = {key: value[..., permutation].copy() for key, value in source.items()}
    transformed["transition_logits"] = source["transition_logits"][:, permutation][:, :, permutation]
    first = geometry.analyze_parameters(source)
    second = geometry.analyze_parameters(transformed)
    for old, new in zip(first["transition"], second["transition"], strict=True):
        for key in ("dobrushin_delta", "row_stochastic_deviation_max", "uniform_image_l1"):
            assert new[key] == pytest.approx(old[key], abs=2e-15, rel=2e-14)
        np.testing.assert_allclose(sorted(new["column_entropies"]),
                                   sorted(old["column_entropies"]), atol=2e-15, rtol=0)
        np.testing.assert_allclose(new["contrast_singular_values"], old["contrast_singular_values"],
                                   atol=2e-15, rtol=2e-14)
        assert new["contrast_ranks"] == old["contrast_ranks"]
    for channel in ("emission", "cost_head"):
        np.testing.assert_allclose(sorted(first[channel]["column_distances"]),
                                   sorted(second[channel]["column_distances"]), atol=2e-15, rtol=0)
        np.testing.assert_allclose(first[channel]["contrast_singular_values"],
                                   second[channel]["contrast_singular_values"], atol=2e-15, rtol=2e-14)
    for channel in ("state", "cost"):
        for horizon in HORIZONS:
            old = record(first["grams"], channel, horizon)
            new = record(second["grams"], channel, horizon)
            np.testing.assert_allclose(new["eigenvalues"], old["eigenvalues"],
                                       atol=2e-15, rtol=3e-14)
            assert new["ranks"] == old["ranks"]


def test_head_only_permutation_preserves_head_geometry_but_changes_transport_alignment():
    direction = np.array([3 / 8, -1 / 8, -1 / 8, -1 / 8])
    head = np.zeros((4, 8), dtype=np.float64)
    head[:, 0] = direction
    permuted = head[:, [1, 0, 2, 3, 4, 5, 6, 7]]
    surviving = np.repeat(np.diag(np.arange(1, 9, dtype=np.float64) / 8)[None], 4, axis=0)
    first = geometry.transport_grams(surviving, head)
    second = geometry.transport_grams(surviving, permuted)
    np.testing.assert_allclose(record(first, "cost", 0)["eigenvalues"],
                               record(second, "cost", 0)["eigenvalues"], atol=2e-16, rtol=0)
    assert record(second, "cost", 1)["trace"] == pytest.approx(
        4 * record(first, "cost", 1)["trace"], abs=2e-16
    )
    for horizon in HORIZONS:
        np.testing.assert_array_equal(record(first, "state", horizon)["gram"],
                                      record(second, "state", horizon)["gram"])


def test_orthonormal_contrast_coordinate_change_preserves_eigenvalues():
    fields = geometry.probability_fields(parameters())
    basis = geometry.zero_sum_basis()
    rotation = np.eye(7)
    rotation[:2, :2] = [[3 / 5, -4 / 5], [4 / 5, 3 / 5]]
    rotated = basis @ rotation
    first = geometry.transport_grams(fields["surviving"], fields["costs"], basis=basis)
    second = geometry.transport_grams(fields["surviving"], fields["costs"], basis=rotated)
    for channel in ("state", "cost"):
        for horizon in HORIZONS:
            old, new = record(first, channel, horizon), record(second, channel, horizon)
            np.testing.assert_array_equal(old["gram"], new["gram"])
            np.testing.assert_allclose(new["projected_gram"],
                                       rotation.T @ np.asarray(old["projected_gram"]) @ rotation,
                                       atol=2e-15, rtol=3e-14)
            np.testing.assert_allclose(new["eigenvalues"], old["eigenvalues"],
                                       atol=2e-15, rtol=3e-14)
            assert new["ranks"] == old["ranks"]


def test_head_and_emission_have_three_dimensional_structural_contrast_limit():
    report = geometry.analyze_parameters(parameters())
    fields = geometry.probability_fields(parameters())
    basis = geometry.zero_sum_basis()
    for channel, matrix in (("emission", fields["emission"]), ("cost_head", fields["costs"])):
        item = report[channel]
        assert item["structural_rank_upper_bound"] == 3
        assert item["pairs"] == [[left, right] for left in range(8) for right in range(left + 1, 8)]
        expected = [float(np.linalg.norm(matrix[:, left] - matrix[:, right]))
                    for left, right in item["pairs"]]
        np.testing.assert_allclose(item["column_distances"], expected, atol=0, rtol=0)
        np.testing.assert_allclose(item["contrast_singular_values"],
                                   np.linalg.svd(matrix @ basis, compute_uv=False), atol=0, rtol=0)
        assert item["contrast_ranks"] == dict.fromkeys(SPECTRAL_THRESHOLDS, 3)


def test_dobrushin_and_non_doubly_stochastic_drift_use_column_orientation():
    source = parameters()
    transition = np.full((8, 8), 1 / 16, dtype=np.float64)
    transition[0, :] += 1 / 2
    transition[:, 1] = np.roll(transition[:, 1], 1)
    source["transition_logits"] = np.log(np.repeat(transition[None], 4, axis=0))
    report = geometry.analyze_parameters(source)
    for item in report["transition"]:
        assert item["dobrushin_delta"] == pytest.approx(1 / 2, abs=2e-15)
        assert item["row_stochastic_deviation_max"] == pytest.approx(3, abs=2e-15)
        assert item["uniform_image_l1"] == pytest.approx(3 / 4, abs=2e-15)


@pytest.mark.parametrize("field", ["transition_logits", "emission_logits", "hazard_logits", "cost_logits"])
@pytest.mark.parametrize("failure", ["dtype", "shape", "nan", "inf", "not_array", "underflow"])
def test_parameter_domain_failures_are_deterministic_and_do_not_mutate(field, failure):
    source = parameters()
    if failure == "dtype":
        source[field] = source[field].astype(np.float32)
    elif failure == "shape":
        source[field] = source[field][:-1]
    elif failure == "not_array":
        source[field] = source[field].tolist()
    else:
        source[field].flat[0] = {"nan": np.nan, "inf": np.inf, "underflow": -1000}[failure]
    before = {key: np.asarray(value).copy() for key, value in source.items()}
    with pytest.raises((TypeError, ValueError)):
        geometry.analyze_parameters(source)
    for key, value in source.items():
        np.testing.assert_array_equal(value, before[key])


@pytest.mark.parametrize("change", ["missing", "extra", "not_mapping"])
def test_parameter_key_roster_is_exact(change):
    source = parameters()
    if change == "missing":
        source.pop("cost_logits")
    elif change == "extra":
        source["unregistered"] = np.ones(1, dtype=np.float64)
    else:
        source = list(source.values())
    with pytest.raises((TypeError, ValueError)):
        geometry.probability_fields(source)


@pytest.mark.parametrize("change", ["shape", "negative", "superstochastic", "nan", "head_nonfinite",
                                    "head_uncentered", "basis_shape", "basis_nonorthogonal", "basis_not_zero_sum"])
def test_transport_helper_rejects_invalid_probability_or_contrast_domains(change):
    surviving = np.repeat(np.eye(8)[None], 4, axis=0)
    head = costs()
    basis = geometry.zero_sum_basis()
    if change == "shape":
        surviving = surviving[:3]
    elif change == "negative":
        surviving[0, 0, 1] = -1e-3
    elif change == "superstochastic":
        surviving[0, 0, 0] = 1.01
    elif change == "nan":
        surviving[0, 0, 0] = np.nan
    elif change == "head_nonfinite":
        head[0, 0] = np.inf
    elif change == "head_uncentered":
        head[0, 0] += 1
    elif change == "basis_shape":
        basis = basis[:, :6]
    elif change == "basis_nonorthogonal":
        basis[:, 0] *= 2
    elif change == "basis_not_zero_sum":
        basis[:, 0] = 1 / np.sqrt(8)
    with pytest.raises((TypeError, ValueError)):
        geometry.transport_grams(surviving, head, basis=basis)


def test_zero_survival_is_absorbing_and_input_arrays_are_not_mutated():
    surviving = np.zeros((4, 8, 8), dtype=np.float64)
    head = costs()
    basis = geometry.zero_sum_basis()
    originals = [item.copy() for item in (surviving, head, basis)]
    result = geometry.transport_grams(surviving, head, basis=basis)
    for channel in ("state", "cost"):
        for horizon in HORIZONS[1:]:
            item = record(result, channel, horizon)
            np.testing.assert_array_equal(item["gram"], np.zeros((8, 8)))
            assert item["trace"] == 0 and item["ranks"] == dict.fromkeys(GRAM_THRESHOLDS, 0)
    result["state"][0]["gram"][0][0] = -5
    for actual, original in zip((surviving, head, basis), originals, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_checks_are_called_and_abort_propagates_without_mutation():
    source = parameters()
    before = {key: value.copy() for key, value in source.items()}
    calls = []

    def reject():
        calls.append(None)
        raise RuntimeError("fabricated stop")

    with pytest.raises(RuntimeError, match="fabricated stop"):
        geometry.analyze_parameters(source, check=reject)
    assert calls
    for key in source:
        np.testing.assert_array_equal(source[key], before[key])


def test_named_work_accounts_for_all_horizons_and_global_rng_is_unchanged():
    before = np.random.get_state()
    callbacks = []
    report = geometry.analyze_parameters(parameters(), check=lambda: callbacks.append(None))
    after = np.random.get_state()
    assert after[0] == before[0] and after[2:] == before[2:]
    np.testing.assert_array_equal(after[1], before[1])
    gram_work = {
        "check_calls": 19,
        "cost_initial_gram_matrix_products": 1,
        "gram_recurrence_steps": 8,
        "gram_channel_updates": 16,
        "gram_action_conjugations": 64,
        "gram_recurrence_matrix_products": 128,
        "gram_measurements": 10,
        "gram_projection_matrix_products": 20,
        "eigenvalue_decompositions": 10,
    }
    assert report["grams"]["work"] == gram_work
    assert report["work"] == {
        "parameter_arrays": 4,
        "parameter_entries": 352,
        "softmax_calls": 3,
        "softmax_columns": 48,
        "sigmoid_calls": 1,
        "sigmoid_entries": 32,
        "surviving_transport_matrices": 4,
        "surviving_product_entries": 256,
        "transition_matrices_analyzed": 4,
        "transition_column_pair_l1_distances": 112,
        "column_entropies": 32,
        "column_pair_euclidean_distances": 56,
        "spectral_matrix_products": 10,
        "singular_value_decompositions": 6,
        "uniform_matrix_vector_products": 4,
        **{key: value for key, value in gram_work.items() if key != "check_calls"},
        "check_calls": 26,
    }
    assert len(callbacks) == 26
