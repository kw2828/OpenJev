"""Independent correction identities and synthetic producer compatibility."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("weight_prior_audit_test", ROOT/"scripts/audit_dialogue_weight_prior.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
h = audit.load_helpers()


def fixture():
    rows = []
    for i, (y, previous) in enumerate(((2, 0), (1, 0), (0, 0), (2, 2),
                                      (2, 0), (3, 3), (0, 0), (1, 2))):
        transition = ("unmentioned_retention" if y == previous == 0 else "assigned_retention" if y == previous
                      else "first_assignment" if previous == 0 else "revision")
        rows.append({"row_index": i, "split": "train", "admission": "admitted", "heldout_service": i < 4,
                     "candidate_count": 4, "candidate_types": [0, 1, 2, 3], "current_label_index": y,
                     "previous_current_index": previous, "current_candidate_id": str(y), "previous_candidate_id": str(previous),
                     "current_value_group": ("none", "dontcare", "true", "false")[y], "derived_bin": transition,
                     "service": "held" if i < 4 else "seen", "dialogue_id": str(i//2)})
    return rows


def raw(probabilities):
    result = np.full((len(probabilities), 12), -np.inf, np.float32)
    for i, values in enumerate(probabilities):
        result[i, :len(values)] = np.log(values)
    return result


def transform(logs, previous, types):
    return audit.correct(logs, np.isfinite(logs), np.asarray(previous, np.int64), np.asarray(types, np.int64), h)


def test_closed_form_none_and_assigned_prior_odds_with_nonzero_none_index():
    logs = raw([[.2, .3, .5], [.2, .3, .5]])
    types = [[1, 0, 4]+[-1]*9]*2
    original, corrected, _, _, _ = transform(logs, [1, 2], types)
    for i, (prior, count) in enumerate(((1, 17666), (2, 9246))):
        p = np.exp(original[i, :3]); ratio = count/2299
        denominator = (1-p[prior])+p[prior]*ratio
        expected = p/denominator; expected[prior] *= ratio
        np.testing.assert_allclose(np.exp(corrected[i, :3]), expected, atol=1e-15, rtol=1e-12)
        other = [j for j in range(3) if j != prior]
        observed = corrected[i, prior]-np.logaddexp.reduce(corrected[i, other])
        baseline = original[i, prior]-np.logaddexp.reduce(original[i, other])
        assert math.isclose(observed-baseline, math.log(ratio), abs_tol=1e-14)


def test_two_candidate_uniform_distribution_matches_count_ratio():
    _, corrected, _, _, _ = transform(raw([[.5, .5]]), [0], [[0, 1]+[-1]*10])
    np.testing.assert_allclose(np.exp(corrected[0, :2]), [17666/(17666+2299), 2299/(17666+2299)], atol=1e-15)


def test_permutation_equivariance_including_previous_type_and_padding():
    logs = raw([[.15, .25, .35, .25]])
    types = np.asarray([[0, 1, 2, 4]+[-1]*8])
    prior = np.array([2], np.int64)
    original = audit.correct(logs, np.isfinite(logs), prior, types, h)[1]
    order = np.array([3, 0, 2, 1]+list(range(4, 12)))
    permuted = audit.correct(logs[:, order], np.isfinite(logs[:, order]),
                            np.array([int(np.flatnonzero(order == prior[0])[0])]), types[:, order], h)[1]
    np.testing.assert_allclose(permuted, original[:, order], atol=1e-14, rtol=1e-12)
    assert np.isneginf(permuted[:, 4:]).all()


def test_extreme_finite_logs_preserve_nll_without_probability_floor():
    logs = np.full((1, 12), -np.inf, np.float32); logs[0, :3] = [0, -1000, -1001]
    original, corrected, _, _, witness = transform(logs, [0], [[0, 1, 2]+[-1]*9])
    assert np.isfinite(corrected[0, :3]).all() and corrected[0, 2] < -1001
    assert np.exp(corrected[0, 2]) == 0
    assert abs((corrected[0, 1]-corrected[0, 2])-(original[0, 1]-original[0, 2])) < 1e-12
    assert witness["alternative_conditional_max_abs_error"] < 1e-10


def test_analytic_weights_are_not_replaced_by_float32_representations():
    values = audit.analytic_weights()
    assert sum(audit.COUNTS.values()) == audit.TOTAL
    for name, count in audit.COUNTS.items():
        assert values[name]["analytic_float64"] == 29211/(3*count)
        assert values[name]["training_float32"] == float(np.float32(29211/(3*count)))
        assert values[name]["analytic_float64"] != values[name]["training_float32"]


def test_raw_normalization_shift_is_signed_and_retained_correction_separate():
    logs = raw([[.5, .3, .2]]); logs[0, :3] += np.float32(5e-7)
    original, corrected, _, z, _ = transform(logs, [0], [[0, 1, 4]+[-1]*9])
    assert z[0] > 0
    assert math.isclose(-original[0, 2]+float(logs[0, 2]), z[0], abs_tol=1e-15)
    assert corrected[0, 0] > original[0, 0] and corrected[0, 2] < original[0, 2]


@pytest.mark.parametrize("bad", ["positive", "padding", "dtype", "previous", "type"])
def test_invalid_public_inputs_and_logs_rejected(bad):
    logs = raw([[.5, .3, .2]]); prior = np.array([0], np.int64); types = np.array([[0, 1, 4]+[-1]*9])
    mask = np.isfinite(logs)
    if bad == "positive": logs[0, 0] = 1
    elif bad == "padding": logs[0, 5] = -10
    elif bad == "dtype": logs = logs.astype(np.float64)
    elif bad == "previous": prior[0] = 3
    else: types[0, 5] = 2
    with pytest.raises(ValueError):
        audit.correct(logs, mask, prior, types, h)


def test_synthetic_producer_to_independent_audit_all_eighteen_and_label_independence():
    spec = importlib.util.spec_from_file_location("weight_prior_producer_fixture", ROOT/"scripts/diagnose_dialogue_weight_prior.py")
    producer = importlib.util.module_from_spec(spec); spec.loader.exec_module(producer)
    parent = producer.load_parent(); arithmetic = parent.load_portable().load_arithmetic()
    rows = fixture(); packets = {}
    for name in h.PREDICTION_PINS:
        values = [.5, .1, .3, .1] if name.startswith("token_mean") else [.2, .1, .6, .1]
        packets[name] = {"row_indices": np.arange(len(rows), dtype=np.int64), "log_probs": raw([values]*len(rows))}
    reported = producer.analyze(rows, arithmetic.row_metadata(rows), packets, parent, arithmetic)
    result = audit.aggregate(rows, packets, reported, h)
    assert result["counts"]["constructions"] == 18 and result["counts"]["primary_cells"] == 90
    assert result["counts"]["paired_cells"] == 270 and result["scalar_comparisons"] > 10000
    meta = h.metadata(rows)
    before = audit.correct(packets["token_mean-6201"]["log_probs"], meta["mask"], meta["previous"], meta["types"], h)[1]
    changed = copy.deepcopy(rows)
    changed[0].update(current_label_index=1, current_candidate_id="1", current_value_group="dontcare")
    other = h.metadata(changed)
    after = audit.correct(packets["token_mean-6201"]["log_probs"], other["mask"], other["previous"], other["types"], h)[1]
    np.testing.assert_array_equal(before, after)
    reported["fits"]["token_mean-corrected-6201"]["cells"]["heldout_service/changed"]["counts"]["correct"] += 1
    with pytest.raises(ValueError, match="Exact mismatch"):
        audit.aggregate(rows, packets, reported, h)


def test_helper_failure_preserved_and_exclusive(tmp_path, monkeypatch):
    args = argparse.Namespace(run=tmp_path/"run", diagnostic=tmp_path/"diagnostic", out=tmp_path/"out",
                              run_sha256="0"*64, summary_sha256="0"*64, receipt_sha256="0"*64)
    def failed():
        raise ValueError("synthetic frozen helper failure")
    monkeypatch.setattr(audit, "load_helpers", failed)
    with pytest.raises(ValueError, match="frozen helper"):
        audit.execute(args)
    assert h.read(args.out/"failed.json")["model_calls"] == 0
    before = (args.out/"failed.json").read_bytes()
    with pytest.raises(FileExistsError):
        audit.execute(args)
    assert before == (args.out/"failed.json").read_bytes()
