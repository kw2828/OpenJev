"""Fabricated scoring and rule witnesses for the residual diagnostic."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("residual_study_fixture",
                Path(__file__).resolve().parents[1]/"scripts/residual_memory_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scoring_matches_explicit_decision_cost(study):
    prediction = {"prob_positive": np.array([[.1, .5, .9]]), "component_mean": np.zeros((1, 3, 1)),
                  "log_weights": np.zeros((1, 3, 1)), "log_prob": np.array([[-1., -2., -3.]])}
    result = study.score(prediction, np.array([[-1., 1., 1.]]), np.array([[.8, .5, .1]]))
    assert result["regret"] == pytest.approx(1.4/3)
    assert result["nll"] == 2.
    assert result["negative"] == result["positive"] == result["defer"] == pytest.approx(1/3)
    assert result["always_defer_regret"] == pytest.approx(.1/3)


def fabricated_rows(study):
    return [{"phase": phase, "cohort": cohort, "mode": mode, "fit_seed": seed,
             **dict.fromkeys(study.METRICS, .1), "regret": .01 if mode == "fic" else .02,
             "nll": .4 if mode == "fic" else .6}
            for phase in study.CONFIG["populations"] for cohort in range(3)
            for mode in (*study.MODES, "true_gp") for seed in (11, 23, 37)]


def test_all_eleven_conditions_required(study):
    result = study.summarize(fabricated_rows(study))
    assert result["gate"] == "RESIDUAL_CONTROL_QUALIFIED"
    assert len(result["checks"]) == 11
    rows = fabricated_rows(study)
    for row in rows:
        if row["phase"] == "shift" and row["cohort"] == 2 and row["mode"] == "fic":
            row["regret"] = .021
    result = study.summarize(rows)
    assert result["gate"] == "RESIDUAL_CONTROL_NOT_QUALIFIED"
    assert any(not r["passed"] for r in result["checks"] if "cohort_2" in r["name"])


def test_density_improvement_cannot_hide_bad_decisions(study):
    rows = fabricated_rows(study)
    for row in rows:
        if row["phase"] == "shift" and row["mode"] == "fic":
            row["regret"] = .2
            row["nll"] = -1.
    assert study.summarize(rows)["gate"] == "RESIDUAL_CONTROL_NOT_QUALIFIED"


def test_fresh_seed_namespaces_and_full_request_counts(study):
    seeds = [study.CONFIG["namespace"]+r["offset"]+c
             for r in study.CONFIG["populations"].values() for c in range(3)]
    assert len(seeds) == len(set(seeds)) == 9
    assert study.CONFIG["contexts"]*len(seeds)*study.CONFIG["queries"] == 4608
    assert study.CONFIG["training_updates"] == 0


@pytest.mark.parametrize("mode", ["sor", "query_only", "fic", "full"])
def test_evaluation_is_serializable_and_targets_do_not_route(study, mode):
    data = {"bx": np.array([[[[0., 0.], [1., 1.]]]]), "by": np.array([[[.2, -.1]]]),
            "fx": np.zeros((1, 1, 2, 2)), "fy": np.zeros((1, 1, 2)),
            "qx": np.ones((1, 1, 2)), "target": np.zeros((1, 1)),
            "private_selected_block": object()}
    first, diagnostics = study.evaluate(data, mode, 1., 1.)
    json.dumps(diagnostics, allow_nan=False)
    data["target"][:] = 20.
    second, _ = study.evaluate(data, mode, 1., 1.)
    for key in ("component_mean", "component_variance", "log_weights", "prob_positive"):
        np.testing.assert_array_equal(first[key], second[key])
    assert not np.array_equal(first["log_prob"], second["log_prob"])
