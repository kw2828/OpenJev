"""Causality, generation law and arithmetic checks on small fabricated fixtures."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("function_study", ROOT / "scripts/function_reuse_reference_study.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def toy(monkeypatch):
    monkeypatch.setattr(study, "CONFIG", {"namespace": 949600, "cohorts": 1, "blocks": [2], "cases": 2,
        "dimensions": 2, "basis_examples": 4, "fewshot_examples": 1, "queries": 3, "choices": 3})
    return study.generate(0, 2, 0)


def public(data):
    return {key: data[key] for key in ("basis_x", "basis_y", "fewshot_x", "fewshot_y",
                                     "query_x", "cost_vectors")}


def test_generation_law_and_independent_cases(monkeypatch):
    first = toy(monkeypatch)
    repeat = study.generate(0, 2, 0)
    other = study.generate(0, 2, 1)
    for key in first:
        np.testing.assert_array_equal(first[key], repeat[key])
    assert not np.array_equal(first["basis_x"], other["basis_x"])
    for k in range(2):
        np.testing.assert_allclose(first["basis_y"][k], first["basis_x"][k] @ first["matrices"][k])
    for j, k in enumerate(first["true_index"]):
        np.testing.assert_allclose(first["fewshot_y"][j], first["fewshot_x"][j] @ first["matrices"][k])
        np.testing.assert_allclose(first["target"][j], first["query_x"][j] @ first["matrices"][k])
    np.testing.assert_allclose(np.linalg.norm(first["cost_vectors"], axis=-1), 1.0)


@pytest.mark.parametrize("private", ["matrices", "true_index", "target", "seed", "future"])
def test_private_fields_rejected(monkeypatch, private):
    data = toy(monkeypatch)
    with pytest.raises(ValueError, match="public predictor boundary"):
        study.evaluate({**public(data), private: np.zeros(1)})


def test_private_targets_and_future_requests_do_not_affect_current_answer(monkeypatch):
    data = toy(monkeypatch)
    before = study.evaluate(public(data))
    data["matrices"][:] = -999
    data["target"][:] = 999
    data["true_index"][:] = 0
    data["fewshot_y"][1:] = 0
    data["query_x"][1:] = 0
    after = study.evaluate(public(data))
    for key in ("lookup_prediction", "fewshot_prediction", "selected_index", "block_residuals"):
        np.testing.assert_array_equal(before[key][0], after[key][0])


def test_supplied_costs_change_decisions_only(monkeypatch):
    data = toy(monkeypatch)
    a = study.evaluate(public(data))
    data["cost_vectors"] *= -1
    b = study.evaluate(public(data))
    np.testing.assert_array_equal(a["lookup_prediction"], b["lookup_prediction"])
    np.testing.assert_array_equal(a["block_residuals"], b["block_residuals"])
    assert np.any(a["lookup_choice"] != b["lookup_choice"])


def test_fixed_denominator_and_zero_regret(monkeypatch):
    data = toy(monkeypatch)
    result = study.evaluate(public(data))
    arrays = {k: np.stack([v, v]) for k, v in {**data, **result}.items()}
    row = study.summarize(arrays, 0, 2)
    assert row["cases"] == 2 and row["queries"] == 6
    assert row["retrieval_accuracy"] == 1.0
    assert row["lookup_regret"] == 0.0
    assert all(row["conditions"].values())
    true_cost = arrays["cost_vectors"][0, 0] @ arrays["target"][0, 0]
    arrays["lookup_choice"][0, 0] = int(np.argmax(true_cost))
    changed = study.summarize(arrays, 0, 2)
    assert changed["lookup_regret"] == pytest.approx((max(true_cost) - min(true_cost)) / 6)
    assert changed["conditions"]["numerical_regret"] is False
    assert changed["lookup_action_accuracy"] == pytest.approx(5 / 6)


def test_missing_public_input_rejected(monkeypatch):
    data = public(toy(monkeypatch))
    del data["fewshot_y"]
    with pytest.raises(ValueError, match="public predictor boundary"):
        study.evaluate(data)
