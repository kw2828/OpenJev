"""Synthetic inverse-loss-weight identities and saved-only failure boundaries."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("weight_prior", ROOT/"scripts/diagnose_dialogue_weight_prior.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


@pytest.fixture
def parent():
    return d.load_parent()


def packet(probabilities, kinds, previous):
    raw = np.full((len(probabilities), 12), -np.inf, np.float32)
    types = np.full(raw.shape, -1, np.int64)
    sizes = np.array([len(p) for p in probabilities], np.int64)
    for i, (p, k) in enumerate(zip(probabilities, kinds, strict=True)):
        raw[i, :len(p)] = np.log(np.array(p, np.float64))
        types[i, :len(p)] = k
    return raw, sizes, np.asarray(previous, np.int64), types


def test_fixed_analytic_weights_and_recorded_training_cast():
    values, record = d.weights()
    assert d.COUNTS == (17666, 9246, 2299) and record["fit_rows"] == 29211
    expected = [float(Fraction(29211, 3*n)) for n in d.COUNTS]
    np.testing.assert_array_equal(values, expected)
    assert list(record["analytic_float64"].values()) == expected
    assert list(record["training_float32_representation"].values()) == np.array(expected, np.float32).astype(np.float64).tolist()
    assert np.any(values != values.astype(np.float32).astype(np.float64))


def test_two_candidate_none_and_assigned_prior_closed_form(parent):
    args = packet([[.5, .5], [.5, .5]], [[0, 1], [0, 4]], [0, 1])
    result, validation, _, _ = d.correct(*args, parent)
    expected = np.array([17666/(17666+2299), 9246/(9246+2299)])
    np.testing.assert_allclose(np.exp(result["corrected"])[np.arange(2), args[2]], expected, atol=1e-15)
    assert validation["alternative_conditional_max_abs_error"] == 0
    assert validation["keep_odds_shift"]["none_previous"]["rows"] == 1
    assert validation["keep_odds_shift"]["assigned_previous"]["rows"] == 1


@pytest.mark.parametrize("assigned_type", [1, 2, 3, 4])
def test_all_assigned_previous_types_use_same_fit_count(parent, assigned_type):
    args = packet([[.2, .5, .3]], [[0, assigned_type, 4]], [1])
    result, validation, _, _ = d.correct(*args, parent)
    previous = args[2]
    old = result["original"][0, 1]-parent.logsumexp(result["original"][:, [0, 2]])[0]
    new = result["corrected"][0, 1]-parent.logsumexp(result["corrected"][:, [0, 2]])[0]
    assert previous[0] != 0
    assert new-old == pytest.approx(np.log(9246/2299), abs=1e-12)
    assert validation["keep_odds_shift"]["none_previous"]["observed_min"] is None


def test_population_weighting_inverse_recovers_known_distribution(parent):
    truth = np.array([.1, .3, .6], np.float64)
    w, _ = d.weights()
    weighted = truth*np.array([w[2], w[2], w[1]])
    weighted /= weighted.sum()
    args = packet([weighted], [[0, 1, 4]], [2])
    result, _, _, _ = d.correct(*args, parent)
    np.testing.assert_allclose(np.exp(result["corrected"][0, :3]), truth, atol=2e-8)


def test_unequal_candidate_support_and_permutation_equivariance(parent):
    args = packet([[.1, .2, .7], [.2, .3, .1, .4]], [[0, 1, 4], [0, 1, 2, 3]], [2, 0])
    result, validation, _, _ = d.correct(*args, parent)
    permutation = np.array([2, 0, 1]+list(range(3, 12))); inverse = np.argsort(permutation)
    moved, _, _, _ = d.correct(args[0][:, permutation], args[1], inverse[args[2]], args[3][:, permutation], parent)
    for readout in d.READOUTS:
        np.testing.assert_allclose(moved[readout][:, inverse], result[readout], atol=1e-12)
    assert validation["corrected_maximum_mass_error"] <= 1e-12
    assert validation["alternative_conditional_max_abs_error"] <= 1e-10
    assert np.isneginf(result["corrected"][0, 3:]).all()


def test_extremely_small_finite_alternatives_and_raw_nll_drift(parent):
    raw, sizes, previous, types = packet([[.5, .3, .2]], [[0, 1, 2]], [0])
    raw[0, :3] = [0, -1000, -1001]
    result, validation, _, _ = d.correct(raw, sizes, previous, types, parent)
    assert np.isfinite(result["corrected"][0, :3]).all()
    assert result["corrected"][0, 1] < -1000 and np.exp(result["corrected"][0, 1]) == 0
    assert validation["keep_odds_identity_max_abs_error"] <= 1e-10
    raw = packet([[.2, .3, .5]], [[0, 1, 2]], [0])[0]
    raw[0, :3] += np.float32(5e-7)
    result, _, norm, z = d.correct(raw, sizes, previous, types, parent)
    assert z[0] > 0 and norm["max_abs_log_z"] <= 2e-6
    assert -result["original"][0, 2]+float(raw[0, 2]) == pytest.approx(z[0], abs=1e-15)


def test_exact_alternative_tie_uses_canonical_first_maximum(parent):
    args = packet([[.0001, .49995, .49995]], [[0, 1, 4]], [0])
    result, _, _, _ = d.correct(*args, parent)
    assert result["corrected"][0, 1] == result["corrected"][0, 2]
    assert np.argmax(result["corrected"]) == 1


def test_positive_odds_shift_can_only_switch_to_previous(parent):
    args = packet([[.1, .6, .2, .1], [.01, .59, .2, .2], [.8, .1, .05, .05], [.3, .3, .2, .2]],
                  [[0, 1, 2, 3]]*4, [0, 0, 0, 1])
    result, validation, _, _ = d.correct(*args, parent)
    before, after = (np.argmax(result[r], axis=1) for r in d.READOUTS)
    assert np.all((after == before) | (after == args[2]))
    assert validation["hard_choice_witness"]["other_changes"] == 0
    assert validation["hard_choice_witness"]["previous_choice_reversals"] == 0
    assert validation["hard_choice_witness"]["switches_to_previous"] > 0
    # Enumerate every possible changed target, not a selectively favorable label.
    for i in range(len(before)):
        for target in range(4):
            if target != args[2][i]:
                assert int(after[i] == target) <= int(before[i] == target)


@pytest.mark.parametrize("kind", ["mass", "nan", "padding", "dtype", "prior", "type", "type_padding"])
def test_invalid_saved_inputs_rejected(parent, kind):
    raw, sizes, previous, types = packet([[.2, .3, .5]], [[0, 1, 4]], [2])
    if kind == "mass": raw[0, 0] = 0
    elif kind == "nan": raw[0, 1] = np.nan
    elif kind == "padding": raw[0, 5] = -5
    elif kind == "dtype": raw = raw.astype(np.float64)
    elif kind == "prior": previous[0] = 3
    elif kind == "type": types[0, 2] = 8
    else: types[0, 7] = 0
    with pytest.raises(ValueError): d.correct(raw, sizes, previous, types, parent)


def fixture_rows():
    result = []
    ids = ["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:True", "value:False"]
    for i, (target, previous) in enumerate(((2, 0), (1, 0), (0, 0), (2, 2), (2, 0), (3, 3), (0, 0), (1, 2))):
        kind = ("unmentioned_retention" if target == previous == 0 else "assigned_retention" if target == previous
                else "first_assignment" if previous == 0 else "revision")
        result.append({"row_index": i, "split": "train", "admission": "admitted", "heldout_service": i < 4,
            "candidate_count": 4, "candidate_types": [0, 1, 2, 3], "current_label_index": target,
            "previous_current_index": previous, "current_candidate_id": ids[target], "previous_candidate_id": ids[previous],
            "current_value_group": ("none", "dontcare", "true", "false")[target], "derived_bin": kind,
            "service": "held" if i < 4 else "seen", "dialogue_id": str(i//2)})
    return result


def test_full_hierarchy_labels_change_only_metrics(parent, monkeypatch):
    arithmetic = parent.load_portable().load_arithmetic()
    rows = fixture_rows()
    raw = packet([[.2, .1, .6, .1]]*len(rows), [[0, 1, 2, 3]]*len(rows), [0]*len(rows))[0]
    packets = {name: {"row_indices": np.arange(len(rows), dtype=np.int64), "log_probs": raw.copy()} for name in arithmetic.ORDER}
    captured = []; original = d.correct
    def observed(*args):
        result = original(*args)
        captured.append({key: value.copy() for key, value in result[0].items()})
        return result
    monkeypatch.setattr(d, "correct", observed)
    before = d.analyze(rows, arithmetic.row_metadata(rows), packets, parent, arithmetic)
    changed = copy.deepcopy(rows)
    changed[0].update(current_label_index=1, current_candidate_id="reserved:DONTCARE", current_value_group="dontcare")
    after = d.analyze(changed, arithmetic.row_metadata(changed), packets, parent, arithmetic)
    assert len(captured) == 18 and len(before["fits"]) == 18 and len(before["comparisons"]) == 6
    for first, second in zip(captured[:9], captured[9:], strict=True):
        for key in first: np.testing.assert_array_equal(first[key], second[key])
    cell = "heldout_service/changed"
    assert before["fits"]["token_mean-original-6201"]["cells"][cell]["counts"]["correct"] == 1
    assert after["fits"]["token_mean-original-6201"]["cells"][cell]["counts"]["correct"] == 0
    for category in d.CATEGORIES:
        for control in (m+"-original" for m in d.METHODS):
            paired = before["comparisons"][category][control]["seeds"]["6201"]["cells"]["heldout_service/all"]
            q = paired["paired"]
            assert sum(v for k, v in q.items() if k != "rows") == 4
            assert paired["count_differences"]["correct"] == q["wrong_to_correct"]-q["correct_to_wrong"]
    empty = before["fits"]["flat_stratum-corrected-6201"]["cells"]["heldout_service/retained"]["rare"]["dontcare_recall"]
    assert empty == {"numerator": 0, "denominator": 0, "rate": None}
    assert "continuation" not in before and len(before["means"]) == 6


def test_authentication_failure_before_prediction_loading(tmp_path, monkeypatch):
    args = argparse.Namespace(run=tmp_path/"run", report=tmp_path/"report", audit=tmp_path/"audit", out=tmp_path/"out")
    args.run.mkdir(); (args.run/"completed.json").write_text("{}")
    def forbidden(*_a, **_kw): pytest.fail("Predictions decoded before input authentication")
    monkeypatch.setattr(d.np, "load", forbidden)
    with pytest.raises(ValueError, match="Fixed input pin"):
        d.execute(args)
    failed = json.loads((args.out/"failed.json").read_text())
    assert failed["model_calls"] == 0 and failed["request"]["run"] == str(args.run)
    assert not (args.out/"receipt.json").exists()


def test_bad_helper_preserves_failure_and_exclusive_output(tmp_path, monkeypatch):
    args = argparse.Namespace(run=tmp_path/"run", report=tmp_path/"report", audit=tmp_path/"audit", out=tmp_path/"out")
    def broken(): raise ValueError("synthetic helper corruption")
    monkeypatch.setattr(d, "load_parent", broken)
    with pytest.raises(ValueError, match="synthetic helper corruption"):
        d.execute(args)
    before = (args.out/"failed.json").read_bytes()
    with pytest.raises(FileExistsError): d.execute(args)
    assert (args.out/"failed.json").read_bytes() == before


def test_output_cap_failure_is_preserved(tmp_path, monkeypatch):
    args = argparse.Namespace(run=tmp_path/"run", report=tmp_path/"report", audit=tmp_path/"audit", out=tmp_path/"out")
    monkeypatch.setattr(d, "LIMITS", {**d.LIMITS, "output_bytes": 1})
    with pytest.raises(ValueError, match="output cap"):
        d.execute(args)
    assert (args.out/"failed.json").exists() and not (args.out/"receipt.json").exists()
