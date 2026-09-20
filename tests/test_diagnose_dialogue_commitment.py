"""Artificial distribution algebra and authentication tests, no study outputs."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("commitment_diagnostic", ROOT/"scripts/diagnose_dialogue_commitment.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def logs(probabilities):
    array = np.full((len(probabilities), 12), -np.inf, np.float32)
    for i, probabilities_i in enumerate(probabilities):
        array[i, :len(probabilities_i)] = np.log(np.asarray(probabilities_i, np.float64))
    return array


def test_cross_formula_unequal_support_and_nonzero_prior():
    m = logs([[.1, .2, .7], [.2, .2, .3, .3]])
    a = logs([[.5, .4, .1], [.6, .1, .1, .2]])
    sizes, previous = np.array([3, 4]), np.array([1, 3])
    out, validation, _, _ = d.construct(m, a, sizes, previous)
    qm = np.exp(d.normalize(m, sizes, previous)[0]); qa = np.exp(d.normalize(a, sizes, previous)[0])
    for i, n in enumerate(sizes):
        other = np.arange(n) != previous[i]
        expected = (1-qm[i, previous[i]])*qa[i, :n][other]/qa[i, :n][other].sum()
        np.testing.assert_allclose(np.exp(out["MA"][i, :n][other]), expected, atol=1e-15)
        assert out["MA"][i, previous[i]] == out["MM"][i, previous[i]]
    assert validation["MM"]["self_argmax_exact"] and validation["AA"]["self_argmax_exact"]
    assert all(v["maximum_mass_error"] <= 1e-12 for v in validation.values())


def test_two_candidates_cross_reduces_to_mass_donor():
    m, a = logs([[.7, .3], [.2, .8]]), logs([[.1, .9], [.9, .1]])
    out, _, _, _ = d.construct(m, a, np.array([2, 2]), np.array([0, 1]))
    np.testing.assert_array_equal(out["MA"], out["MM"])
    np.testing.assert_array_equal(out["AM"], out["AA"])


def test_conditional_concentration_can_change_donors_argmax():
    m, a = logs([[.4, .3, .3]]), logs([[.1, .8, .1]])
    out, _, _, _ = d.construct(m, a, np.array([3]), np.array([0]))
    assert np.argmax(out["MM"]) == 0
    assert np.argmax(out["MA"]) == 1
    assert out["MA"][0, 0] == out["MM"][0, 0]


def test_extreme_logs_use_direct_alternative_log_mass_without_floor():
    m = np.full((1, 12), -np.inf, np.float32)
    m[0, :3] = [0, -1000, -1001]
    a = logs([[.2, .4, .4]])
    out, _, _, _ = d.construct(m, a, np.array([3]), np.array([0]))
    assert np.isfinite(out["MA"][0, :3]).all()
    assert out["MA"][0, 1] < -999
    assert np.exp(out["MA"][0, 1]) == 0  # Finite log loss survives exp underflow.
    assert np.isneginf(out["MA"][0, 3:]).all()


def test_self_cells_score_canonical_normalized_logs_and_preserve_ties():
    m, a = logs([[.4, .4, .2]]), logs([[.2, .4, .4]])
    sizes, prior = np.array([3]), np.array([2])
    out, witness, _, _ = d.construct(m, a, sizes, prior)
    np.testing.assert_array_equal(out["MM"], d.normalize(m, sizes, prior)[0])
    np.testing.assert_array_equal(out["AA"], d.normalize(a, sizes, prior)[0])
    assert np.argmax(out["MM"]) == np.argmax(m) == 0
    assert np.argmax(out["AA"]) == np.argmax(a) == 1
    assert witness["MM"]["self_reconstruction_max_abs"] <= 1e-10


def test_cross_exact_tie_uses_first_maximum_without_tolerance():
    m, a = logs([[.2, .4, .4]]), logs([[.5, .25, .25]])
    out, _, _, _ = d.construct(m, a, np.array([3]), np.array([0]))
    assert out["MA"][0, 1] == out["MA"][0, 2]
    assert np.argmax(out["MA"]) == 1


@pytest.mark.parametrize("kind", ["mass", "padding", "nan", "positive", "dtype", "prior", "one", "bool"])
def test_bad_saved_distribution_or_support_rejected(kind):
    raw = logs([[.2, .3, .5]])
    sizes, prior = np.array([3]), np.array([0])
    if kind == "mass": raw[0, 0] = -.1
    elif kind == "padding": raw[0, 8] = -10
    elif kind == "nan": raw[0, 1] = np.nan
    elif kind == "positive": raw[0, 0] = .01
    elif kind == "dtype": raw = raw.astype(np.float64)
    elif kind == "prior": prior[0] = 3
    elif kind == "one": sizes[0] = 1
    else: prior = np.array([False])
    with pytest.raises(ValueError):
        d.normalize(raw, sizes, prior)


def fixture_rows():
    result = []
    identities = ["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:True", "value:False"]
    for i, (target, previous) in enumerate(((2, 0), (1, 0), (0, 0), (2, 2), (2, 0), (3, 3), (0, 0), (1, 2))):
        kind = ("unmentioned_retention" if target == previous == 0 else "assigned_retention" if target == previous
                else "first_assignment" if previous == 0 else "revision")
        result.append({"row_index": i, "split": "train", "admission": "admitted", "heldout_service": i < 4,
            "candidate_count": 4, "candidate_types": [0, 1, 2, 3], "current_label_index": target,
            "previous_current_index": previous, "current_candidate_id": identities[target],
            "previous_candidate_id": identities[previous], "current_value_group": ("none", "dontcare", "true", "false")[target],
            "derived_bin": kind, "service": "held" if i < 4 else "seen", "dialogue_id": str(i//2)})
    return result


def test_full_artificial_hierarchy_and_pair_orientation():
    arithmetic = d.load_portable().load_arithmetic()
    rows = fixture_rows(); meta = arithmetic.row_metadata(rows)
    packets = {}
    for name in arithmetic.ORDER:
        p = [.5, .1, .3, .1] if name.startswith("token_mean") else [.2, .1, .6, .1]
        packets[name] = {"row_indices": np.arange(len(rows), dtype=np.int64), "log_probs": logs([p]*len(rows))}
    result = d.analyze(rows, meta, packets, arithmetic)
    assert len(result["fits"]) == 15
    mm = result["fits"]["MM-6201"]
    assert len(mm["cells"]) == 15 and set(mm["services"]) == {"held", "seen"}
    retained = mm["cells"]["heldout_service/retained"]
    assert retained["counts"]["correct"] == 1 and retained["rows"] == 2
    assert retained["rare"]["dontcare_recall"] == {"numerator": 0, "denominator": 0, "rate": None}
    for constructed in d.CONSTRUCTIONS:
        for control in d.CONTROLS:
            paired = result["comparisons"][constructed][control]["seeds"]["6201"]["cells"]["heldout_service/all"]
            quadrants = paired["paired"]
            assert sum(v for k, v in quadrants.items() if k != "rows") == quadrants["rows"] == 4
            assert paired["count_differences"]["correct"] == quadrants["wrong_to_correct"]-quadrants["correct_to_wrong"]
    assert result["tradeoffs"]["seeds"]["6201"]["AA"]["heldout_service"]["retained_errors_recovered_vs_AA"] == 0
    assert result["tradeoffs"]["seeds"]["6201"]["MM"]["heldout_service"]["changed_correct_gain_vs_MM"] == 0
    assert "continuation" not in result


def test_label_changes_do_not_change_construction(monkeypatch):
    arithmetic = d.load_portable().load_arithmetic()
    rows = fixture_rows()
    packets = {name: {"row_indices": np.arange(len(rows), dtype=np.int64),
                     "log_probs": logs([[.2, .1, .6, .1]]*len(rows))} for name in arithmetic.ORDER}
    captured = []
    original_construct = d.construct
    def observe(*args):
        result = original_construct(*args)
        captured.append({name: value.copy() for name, value in result[0].items()})
        return result
    monkeypatch.setattr(d, "construct", observe)
    before = d.analyze(rows, arithmetic.row_metadata(rows), packets, arithmetic)
    changed = copy.deepcopy(rows)
    changed[0].update(current_label_index=1, current_candidate_id="reserved:DONTCARE", current_value_group="dontcare")
    after = d.analyze(changed, arithmetic.row_metadata(changed), packets, arithmetic)
    assert len(captured) == 6
    for first, second in zip(captured[:3], captured[3:], strict=True):
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
            np.testing.assert_array_equal(np.argmax(first[name], axis=1), np.argmax(second[name], axis=1))
    cell = "heldout_service/changed"
    assert before["fits"]["AA-6201"]["cells"][cell]["counts"]["correct"] == 1
    assert after["fits"]["AA-6201"]["cells"][cell]["counts"]["correct"] == 0
    assert before["fits"]["AA-6201"]["cells"][cell]["metrics"]["nll"] != after["fits"]["AA-6201"]["cells"][cell]["metrics"]["nll"]


def test_log_normalization_nll_drift_has_correct_sign():
    raw = logs([[.2, .3, .5]])
    raw[0, :3] += np.float32(5e-7)
    q, witnesses, z = d.normalize(raw, np.array([3]), np.array([1]))
    assert z[0] > 0 and witnesses["max_abs_log_z"] <= 2e-6
    np.testing.assert_allclose(-q[0, 2]+float(raw[0, 2]), z[0], atol=1e-15)


def test_equal_groups_and_empty_support_are_explicit():
    group = {"indices": np.arange(3), "equal_service": (np.array([0, 0, 1]), np.array([2, 1])),
             "equal_dialogue": (np.array([0, 1, 2]), np.array([1, 1, 1]))}
    values = np.array([0., 0., 1.])
    assert d.means(values, group) == {"row": 1/3, "equal_service": .5, "equal_dialogue": 1/3}
    group["indices"] = np.array([], np.int64)
    assert d.means(values, group) == dict.fromkeys(("row", "equal_service", "equal_dialogue"))


def test_gold_branch_is_selected_candidate_not_largest_mass():
    arithmetic = d.load_portable().load_arithmetic()
    rows = fixture_rows()[:1]; meta = arithmetic.row_metadata(rows)
    normalized = np.full((1, 12), -np.inf, np.float64)
    normalized[0, :4] = np.log([.35, .05, .3, .3])
    values, choice, _, _ = d.vectors(normalized, meta)
    assert choice[0] == 0 and values["wrong_selected_branch"][0] == 1


def test_auth_hashes_all_predictions_before_any_metadata_decode(tmp_path, monkeypatch):
    portable = d.load_portable()
    run, report, audit = (tmp_path/name for name in ("run", "report", "audit"))
    for path in (run, report, audit): path.mkdir()
    paths = {"completed": run/"completed.json", "plan": run/"plan.json", "rows": run/"evaluation-rows.jsonl",
             "report_receipt": report/"receipt.json", "report_summary": report/"summary.json", "audit_receipt": audit/"receipt.json"}
    for path in paths.values(): path.write_bytes(b"not decoded")
    monkeypatch.setattr(d, "PINS", {name: portable.digest(path) for name, path in paths.items()})
    predictions = {}
    for name in d.PREDICTIONS:
        path = run/"fits"/name/"predictions.npz"; path.parent.mkdir(parents=True)
        path.write_bytes(b"not an array"); predictions[name] = portable.digest(path)
    monkeypatch.setattr(d, "PREDICTIONS", predictions)
    last = run/"fits"/list(predictions)[-1]/"predictions.npz"
    last.write_bytes(b"changed")
    def forbidden(*_a, **_kw): pytest.fail("Decoding before all prediction pins")
    arithmetic = argparse.Namespace(read=forbidden)
    monkeypatch.setattr(d.np, "load", forbidden)
    with pytest.raises(ValueError, match="Fixed prediction pin"):
        d.authenticate(argparse.Namespace(run=run, report=report, audit=audit), portable, arithmetic, lambda: None)


def test_source_failure_is_preserved_exclusively(tmp_path, monkeypatch):
    args = argparse.Namespace(run=tmp_path/"run", report=tmp_path/"report", audit=tmp_path/"audit",
                              out=tmp_path/"result", spec_sha256="0"*64)
    def broken(): raise ValueError("synthetic source mismatch")
    monkeypatch.setattr(d, "load_portable", broken)
    with pytest.raises(ValueError, match="synthetic source mismatch"):
        d.execute(args)
    failed = json.loads((args.out/"failed.json").read_text())
    assert failed["request"]["spec_sha256"] == "0"*64 and failed["model_calls"] == 0
    before = (args.out/"failed.json").read_bytes()
    with pytest.raises(FileExistsError): d.execute(args)
    assert before == (args.out/"failed.json").read_bytes()


def test_arbitrary_prior_does_not_assume_none_is_zero():
    probabilities = [[.2, .3, .5], [.1, .6, .3]]
    m = logs(probabilities); a = logs(probabilities[::-1])
    sizes, prior = np.array([3, 3]), np.array([2, 1])
    original = d.construct(m, a, sizes, prior)[0]
    permutation = np.array([2, 0, 1]+list(range(3, 12)))
    inv = np.argsort(permutation)
    changed = d.construct(m[:, permutation], a[:, permutation], sizes, inv[prior])[0]
    for key in original:
        np.testing.assert_allclose(changed[key][:, inv], original[key], atol=1e-12)


def test_one_seed_tradeoff_not_hidden_by_mean():
    records = [{"gain": 10, "harm": 3}, {"gain": 2, "harm": -1}, {"gain": -3, "harm": -2}]
    untouched = copy.deepcopy(records)
    assert d.average(records) == {"gain": 3., "harm": 0.}
    assert records == untouched
