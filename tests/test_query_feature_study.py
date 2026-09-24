"""Fabricated contract tests, not empirical benchmark examples."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def study():
    path = Path(__file__).resolve().parents[1] / "scripts/query_feature_study.py"
    spec = importlib.util.spec_from_file_location("query_feature_study_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_batch_preserves_context_request_alignment(study):
    data = {"bx": np.arange(24, dtype=float).reshape(2, 2, 3, 2),
            "by": np.arange(12, dtype=float).reshape(2, 2, 3),
            "fx": np.arange(32, dtype=float).reshape(2, 2, 4, 2),
            "fy": np.arange(16, dtype=float).reshape(2, 2, 4),
            "qx": np.arange(8, dtype=float).reshape(2, 2, 2),
            "target": object(), "private_selected_block": object()}
    batch = study.public_batch(data, np.array([1, 0]))
    assert set(batch) == set(study.PUBLIC)
    np.testing.assert_array_equal(batch["bx"].numpy(), data["bx"][[1, 1, 0, 0]])
    np.testing.assert_array_equal(batch["fx"].numpy(), data["fx"][[1, 0]].reshape(4, 4, 2))


def test_decision_cost_and_abstention_witness(study):
    p = np.array([[.1, .5, .9]])
    pred = {"prob_positive": p, "component_mean": np.zeros((1, 3, 1)),
            "log_weights": np.zeros((1, 3, 1)), "log_prob": np.array([[-1., -2., -3.]])}
    scored = study.metrics(pred, np.array([[-1., 1., 1.]]), np.array([[.8, .5, .1]]))
    np.testing.assert_allclose(scored["regret"], [[.6, 0., .8]])
    np.testing.assert_allclose(scored["always_defer_regret"], [[0., 0., .1]], atol=1e-15)
    np.testing.assert_array_equal(scored["defer"], [[0., 1., 0.]])
    np.testing.assert_array_equal(scored["negative"], [[1., 0., 0.]])
    np.testing.assert_array_equal(scored["positive"], [[0., 0., 1.]])
    np.testing.assert_array_equal(scored["nll"], [[1., 2., 3.]])


def rows(study, candidate_regret=.008, candidate_nll=.41):
    return [{"phase": phase, "cohort": cohort, "arm": arm, "fit_seed": seed,
             "regret": candidate_regret if arm == "centered16" else .01,
             "nll": candidate_nll if arm == "centered16" else .4,
             **dict.fromkeys(("brier", "mse", "defer", "negative", "positive", "always_defer_regret"), .1)}
            for phase in ("base", "shift") for cohort in range(3)
            for arm in (*study.ARMS, "full_gp") for seed in (11, 23, 37)]


def test_gate_requires_regret_and_nll(study):
    assert study.summarize(rows(study))["gate"] == "CONTINUE_TO_SELECTIVE_REFINEMENT"
    assert study.summarize(rows(study, candidate_regret=.0095))["gate"] == "DO_NOT_ADVANCE_THIS_CANDIDATE"
    assert study.summarize(rows(study, candidate_nll=.43))["gate"] == "DO_NOT_ADVANCE_THIS_CANDIDATE"


def test_gate_rejects_one_cohort_reversal_despite_overall_gain(study):
    values = rows(study, candidate_regret=.001)
    for row in values:
        if row["phase"] == "shift" and row["cohort"] == 2 and row["arm"] == "centered16":
            row["regret"] = .011
    summary = study.summarize(values)
    assert summary["means"]["shift"]["centered16"]["regret"] < .009
    assert summary["gate"] == "DO_NOT_ADVANCE_THIS_CANDIDATE"


def test_frozen_update_and_exposure_counts(study):
    c = study.CONFIG
    assert c["epochs"]*c["train_contexts"]//c["batch_contexts"] == 512
    assert c["epochs"]*c["train_contexts"]*c["train_queries"] == 32768


def test_paired_initialization_and_parameter_counts(study):
    for seed in study.SEEDS:
        a, b = study.make_model("static16", seed), study.make_model("centered16", seed)
        assert a.parameter_count == b.parameter_count == 624
        for x, y in zip(a.parameters(), b.parameters(), strict=True):
            assert bool((x == y).all())
    assert study.make_model("static32", 11).parameter_count == 1152
    assert study.make_model("nystrom16", 11).parameter_count == 2


def test_failure_preserves_partial_fit_without_retry(study, monkeypatch, tmp_path):
    class Broken(torch.nn.Linear):
        def forward(self, **batch):
            raise RuntimeError("fabricated failure before first update")

    monkeypatch.setattr(study, "make_model", lambda *_: Broken(2, 1, dtype=torch.float64))
    monkeypatch.setitem(study.CONFIG, "train_contexts", 2)
    monkeypatch.setitem(study.CONFIG, "batch_contexts", 2)
    data = {"bx": np.ones((2, 2, 3, 2)), "by": np.ones((2, 2, 3)),
            "fx": np.ones((2, 2, 4, 2)), "fy": np.ones((2, 2, 4)),
            "qx": np.ones((2, 2, 2)), "target": np.ones((2, 2))}
    with pytest.raises(RuntimeError, match="fabricated failure"):
        study.train(tmp_path, data, "static16", 11)
    partial = json.loads((tmp_path / "static16-11-partial.json").read_text())
    assert partial["completed_updates"] == 0
    assert partial["completed_epochs"] == []
    assert (tmp_path / "static16-11-partial.pt").is_file()
    assert not (tmp_path / "static16-11.pt").exists()


def test_verifier_rejects_missing_source_roster(study, tmp_path):
    (tmp_path / "registration.json").write_text(json.dumps({"config": study.CONFIG, "sources": {}}))
    with pytest.raises(ValueError, match="source roster"):
        study.verify_sources(tmp_path)
