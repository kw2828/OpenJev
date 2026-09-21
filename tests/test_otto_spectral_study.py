"""Synthetic saved-path analysis checks, with no OTTO or real-study inputs."""
from __future__ import annotations

import builtins
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/study_otto_spectral_memory.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("synthetic_otto_spectral_study", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner():
    return load_runner()


def test_top_level_import_has_no_numerical_or_simulator_execution(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"numpy", "scipy", "otto", "isotropic", "torch", "jax"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert len(load_runner().ARMS) == 12


def test_all_twelve_identities_and_rotations_are_balanced(runner):
    expected = {"full_bayes", "exact_log", "recent32", "recent32_hard"}
    expected.update(f"dct{q}_{extension}" for q in (4, 8, 16, 53) for extension in ("neutral", "nearest"))
    assert set(runner.ARMS) == expected
    for case in (0, 7, 95):
        orders = [runner.arm_order(case, step) for step in range(12)]
        assert all(len(order) == len(set(order)) == 12 for order in orders)
        assert all({order[position] for order in orders} == expected for position in range(12))
    assert runner.arm_order(3, 4) == runner.arm_order(7, 0)


@pytest.mark.parametrize("case,step", [(-1, 0), (0, -1), (True, 0), (0, 0.5)])
def test_rotation_rejects_noninteger_or_negative_indices(runner, case, step):
    with pytest.raises(ValueError, match="rotation"):
        runner.arm_order(case, step)


@pytest.mark.parametrize("steps,expected", [(0, False), (31, False), (32, False), (33, True), (178, True)])
def test_eligibility_is_strictly_more_than_32_completed_observations(runner, steps, expected):
    assert runner.eligible_prefix(steps) is expected


def synthetic_cases():
    return [{"initial_hit": hit, "decisions": {1: 1, 2: 2, 3: 4}[hit],
             "all_means": {"metric": {1: 2, 2: 4, 3: 8}[hit]},
             "eligible_prefixes": 0, "eligible_means": None}
            for hit in (1, 2, 3) for _ in range(32)]


def test_case_and_prefix_weighting_have_distinct_hand_computed_denominators(runner):
    cases = synthetic_cases()
    before = copy.deepcopy(cases)
    result = runner.population(cases, {1: 0.5, 2: 0.25, 3: 0.25})
    assert cases == before
    assert result["selected_cases"] == result["all_cases"] == 96
    assert result["prefixes"] == 224
    assert result["selected_case_mixture_mass"] == 1
    assert result["unweighted_case_means"]["metric"] == pytest.approx(14 / 3)
    assert result["mixture_weighted_case_means"]["metric"] == 4
    assert result["mixture_weighted_prefix_means"]["metric"] == 5.5
    assert result["by_initial_hit"][2]["prefixes"] == 64


def test_eligible_population_renormalizes_case_and_prefix_weights_separately(runner):
    cases = synthetic_cases()
    cases[0].update(eligible_prefixes=2, eligible_means={"metric": 10})
    cases[64].update(eligible_prefixes=6, eligible_means={"metric": 30})
    result = runner.population(cases, {1: 0.5, 2: 0.25, 3: 0.25}, eligible=True)
    assert result["all_cases"] == 96 and result["selected_cases"] == 2
    assert result["prefixes"] == 8
    assert result["selected_case_mixture_mass"] == 0.75 / 32
    assert result["unweighted_case_means"]["metric"] == 20
    assert result["mixture_weighted_case_means"]["metric"] == pytest.approx(50 / 3)
    assert result["mixture_weighted_prefix_means"]["metric"] == 22
    assert result["by_initial_hit"][2]["case_means"] is None
    assert result["by_initial_hit"][2]["prefix_means"] is None


def test_empty_eligible_population_serializes_null_not_nan(runner):
    result = runner.population(synthetic_cases(), {1: 0.5, 2: 0.25, 3: 0.25}, eligible=True)
    assert result["selected_cases"] == result["prefixes"] == 0
    assert result["mixture_weighted_case_means"] is None
    assert result["mixture_weighted_prefix_means"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("records,weights", [
    ([{"m": 1}, {"different": 2}], None),
    ([{"m": 1}], [0]), ([{"m": 1}], [-1]), ([{"m": 1}], [math.inf]),
    ([{"m": 1}], [1, 1]), ([{"m": math.nan}], None),
])
def test_aggregation_rejects_missing_metrics_invalid_weights_and_nonfinite_values(runner, records, weights):
    with pytest.raises(ValueError):
        runner.mean_records(records, weights)


def test_stable_log_kl_retains_underflowed_probability_without_floor(runner):
    full = np.array([0.5, 0.5], dtype=np.float64)
    logs, probabilities = runner.normalized_logs(np, np.array([-1000.0, 0.0]))
    runner.validate_decoded(np, logs, probabilities, (2,))
    scores, full_scores = [3.0, 1.0, None, 2.0], [1.0, 4.0, None, 2.0]
    result = runner.belief_metrics(np, np.log(full), full, logs, probabilities, scores, full_scores)
    assert probabilities[0] == 0 and logs[0] == -1000
    assert result["kl_full_to_approx"] == pytest.approx(500 - math.log(2))
    assert result["tv_to_full"] == 0.5
    assert result["probability_underflow_cells"] == 1
    assert result["finite_log_support_cells"] == 2
    assert result["positive_probability_cells"] == 1
    assert result["full_objective_excess"] == 3
    assert result["action"] == 1 and result["matches_full_action"] is False


def test_reference_zero_mass_does_not_create_zero_times_infinity_nan(runner):
    logs, probabilities = np.array([0.0, -np.inf]), np.array([1.0, 0.0])
    result = runner.belief_metrics(np, logs, probabilities, logs, probabilities,
                                  [None, 2.0, 2.0, 3.0], [None, 2.0, 2.0, 3.0])
    assert result["kl_full_to_approx"] == result["tv_to_full"] == 0
    assert result["probability_underflow_cells"] == 0
    assert result["action"] == 1 and result["tie_count"] == 2


def test_true_missing_approximate_support_is_rejected_instead_of_floored(runner):
    with pytest.raises(ValueError, match="log support"):
        runner.belief_metrics(np, np.log([0.5, 0.5]), np.array([0.5, 0.5]),
                              np.array([-np.inf, 0.0]), np.array([0.0, 1.0]), [1., 2., 3., 4.], [1., 2., 3., 4.])


@pytest.mark.parametrize("values", [[-np.inf, -np.inf], [0., np.nan], [0., np.inf]])
def test_invalid_or_empty_log_support_fails(runner, values):
    with pytest.raises(ValueError):
        runner.normalized_logs(np, np.array(values))


def test_saved_qualification_checks_support_tolerance_and_selected_ties(runner):
    recorded = {"scores": [1.0, 1.0, None, 2.0], "action": 0}
    assert runner.parity([1.0, 1.0, None, 2.0 + 5e-9], recorded) < 1e-8
    with pytest.raises(ValueError, match="score parity"):
        runner.parity([1.0, 1.0, None, 2.0 + 2e-8], recorded)
    with pytest.raises(ValueError, match="score support"):
        runner.parity([1.0, 1.0, 3.0, 2.0], recorded)
    with pytest.raises(ValueError, match="selected action"):
        runner.parity([1.0, 1.0 - 2e-10, None, 2.0], recorded)


def test_flatten_keeps_every_arm_rank_and_extension(runner):
    modes = {arm: dict.fromkeys(runner.MEASURES, 1.0) for arm in runner.ARMS}
    sensitivity = {q: {"actions_disagree": False, "tv_between_extensions": 0.1} for q in runner.RANKS}
    flat = runner.flatten(modes, sensitivity)
    assert len(flat) == 12 * 4 + 3 * 2
    assert "dct53_nearest.kl_full_to_approx" in flat
    assert flat["fill.q16.actions_disagree"] == 0
    del modes["recent32_hard"]
    with pytest.raises(KeyError):
        runner.flatten(modes, sensitivity)


class TinyAnalytic:
    """Synthetic likelihood table, not a simulator or physical sensor model."""

    N, NHITS = 5, 2

    @staticmethod
    def moved(position, action):
        result = list(position)
        axis = action // 2
        result[axis] = max(0, min(4, result[axis] + (-1 if action % 2 == 0 else 1)))
        return tuple(result)

    @staticmethod
    def prior(kernel, hit):
        prior = np.ones((5, 5), dtype=np.float64)
        prior[2, 2] = 0
        return prior / prior.sum()

    @staticmethod
    def likelihood_at(kernel, position):
        likelihoods = np.full((2, 5, 5), 0.5, dtype=np.float64)
        likelihoods[1, 0, 0] = 0
        return likelihoods

    @staticmethod
    def normalized(probabilities):
        assert probabilities.sum() > 0
        return probabilities / probabilities.sum()


def packet(step, hit=0):
    position = [2, 2 + step % 2]
    return {"position": position, "hit": hit, "step": step, "done": False,
            "valid_actions": [a for a in range(4) if TinyAnalytic.moved(tuple(position), a) != tuple(position)]}


def test_recent_boundary_forgets_old_soft_zero_but_hard_control_keeps_it(runner):
    actors = {mode: runner.PublicBaseline(packet(0), None, TinyAnalytic, np, mode)
              for mode in ("full_bayes", "recent32", "recent32_hard")}
    storage = {mode: actor.storage_bytes() for mode, actor in actors.items()}
    for step in range(1, 33):
        for actor in actors.values():
            actor.update(packet(step, hit=int(step == 1)))
    assert all(actor.decode()[1][0, 0] == 0 for actor in actors.values())
    for actor in actors.values():
        actor.update(packet(33))
    assert actors["recent32"].decode()[1][0, 0] > 0
    assert actors["recent32_hard"].decode()[1][0, 0] == 0
    assert actors["full_bayes"].decode()[1][0, 0] == 0
    assert actors["recent32"].excluded[2, 2] and actors["recent32"].excluded[2, 3]
    assert actors["recent32_hard"].excluded[0, 0]
    assert not actors["recent32"].excluded[0, 0]
    for mode, actor in actors.items():
        assert actor.storage_bytes() == storage[mode]
        runner.validate_decoded(np, *actor.decode(), (5, 5))


def test_public_updates_reject_future_or_private_packets_before_mutation(runner):
    actor = runner.PublicBaseline(packet(0), None, TinyAnalytic, np, "recent32_hard")
    before = actor.decode()[1].copy()
    for bad in (packet(2), {**packet(1), "source": [0, 0]}, {**packet(1), "done": True, "hit": -2}):
        with pytest.raises(ValueError):
            actor.update(bad)
        np.testing.assert_array_equal(actor.decode()[1], before)
        assert actor.step == actor.count == 0


def test_external_plan_rejection_precedes_numerical_or_saved_input_loading(runner, tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n")
    args = SimpleNamespace(plan=plan, plan_sha256="0" * 64, output=tmp_path / "attempt")
    real_sha = runner.sha
    monkeypatch.setattr(runner, "sha", lambda path: runner.CLOCK_PIN if str(path).endswith("suspend_clock.py") else real_sha(path))
    monkeypatch.setattr(runner, "SuspendClock", lambda: SimpleNamespace(backend="synthetic", now_ns=lambda: 1))
    monkeypatch.setattr(runner, "load_analytic", lambda: pytest.fail("saved input loader reached"))
    monkeypatch.setattr(runner, "load_spectral", lambda _: pytest.fail("numerical model loader reached"))
    with pytest.raises(ValueError, match="external spectral study plan pin"):
        runner.execute(args)
    assert not (args.output / "receipt.json").exists()
    failure = json.loads((args.output / "failed.json").read_text())
    assert failure["status"] == "failed" and failure["progress"]["prefixes"] == 0


def test_exclusive_output_collision_preserves_existing_bytes(runner, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "receipt.json"
    marker.write_bytes(b"preserved\n")
    with pytest.raises(FileExistsError):
        runner.execute(SimpleNamespace(output=output))
    assert marker.read_bytes() == b"preserved\n"
    assert {p.name for p in output.iterdir()} == {"receipt.json"}
