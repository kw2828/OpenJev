"""Small hand-computable checks; no real study artifacts or neural imports."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("independent_copy_audit", Path(__file__).with_name("audit.py"))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def arrays():
    p = np.zeros((6, 12), np.float64)
    p[:, :3] = [[.8, .1, .1], [.1, .2, .7], [.2, .2, .6], [.6, .3, .1], [.1, .2, .7], [.2, .4, .4]]
    return {"probabilities": p, "labels": np.array([0, 2, 2, 0, 2, 1]),
        "choice": p.argmax(1), "bin": np.array(["unmentioned_retention", "assigned_retention", "revision"] * 2),
        "unseen": np.array([False] * 3 + [True] * 3), "dialogue": np.array(["a"] * 3 + ["b"] * 3),
        "time": np.array([0, 1, 2] * 2), "query": np.array([0] * 3 + [1] * 3)}


def test_hand_metrics_and_reference_do_not_invent_probability():
    a = arrays()
    result = audit.metrics(a)
    assert result["seen"]["micro"]["accuracy"] == 1
    assert result["seen"]["micro"]["nll"] == pytest.approx(-np.log(.8 * .7 * .6) / 3)
    # [.8,.1,.1] against class0 => .06; [.1,.2,.7] => .14; [.2,.2,.6] => .24.
    assert result["seen"]["micro"]["brier"] == pytest.approx((.06 + .14 + .24) / 3)
    assert result["all"]["micro"]["count"] == 6
    assert result["all"]["bins"]["clear"]["nll"] is None
    ref = audit.metrics(a, np.zeros(6, int))
    assert ref["seen"]["macro_three"]["accuracy"] == pytest.approx(1 / 3)
    assert "nll" not in ref["seen"]["micro"]


def test_true_zero_target_is_infinite_not_floored():
    a = arrays()
    a["probabilities"][0] = 0
    a["probabilities"][0, 2] = 1
    result = audit.metrics(a)
    assert result["seen"]["micro"]["nll"] is None
    assert result["seen"]["micro"]["zero_target_probabilities"] == 1
    assert result["unseen"]["micro"]["nll"] is not None


def fixture_gate():
    rows, refs = {}, {"literal": {}}
    for method in audit.METHODS:
        hits = {"readout": 125, "scalar": 125, "selective": 126, "selective_no_lexical": 180, "candidate_gru": 126}[method]
        for seed in audit.SEEDS:
            rows[f"{method}-{seed}"] = {p: {"strata": {s: {"count": 200, "correct": hits} for s in audit.STRATA},
                "micro": {"nll": .4}, "revision": {"count": 100, "correct": 60}} for p in ("seen", "unseen")}
    refs["literal"] = {p: {"strata": {s: {"count": 200, "correct": 120} for s in audit.STRATA},
        "revision": {"count": 100, "correct": 61}} for p in ("seen", "unseen")}
    return rows, refs


def test_exact_inclusive_all_thirteen_boundaries():
    rows, refs = fixture_gate()
    result = audit.gate(rows, refs)
    assert result["passed"] and result["checks_passed"] == result["checks_total"] == 13
    # The nongated no-lexical ablation cannot become the selected primary.
    assert rows["selective_no_lexical-4101"]["seen"]["strata"]["changed"]["correct"] > 126
    rows["selective-4101"]["seen"]["strata"]["changed"]["correct"] -= 1
    result = audit.gate(rows, refs)
    assert not result["passed"]
    assert not next(x for x in result["checks"] if x["name"] == "seen/macro_gain_over_scalar_half_pp")["passed"]


def test_missing_stratum_and_nll_cannot_pass():
    rows, refs = fixture_gate()
    rows["selective-4101"]["seen"]["strata"]["changed"] = {"count": 0, "correct": 0}
    rows["selective-4101"]["seen"]["micro"]["nll"] = None
    result = audit.gate(rows, refs)
    assert not result["passed"]
    assert not next(x for x in result["checks"] if x["name"] == "seen/micro_nll_nonworse_scalar")["passed"]


@pytest.mark.parametrize("mutation", ["padding", "choice", "nan", "labels"])
def test_saved_prediction_corruption(tmp_path, mutation):
    a = arrays()
    expected = {k: a[k].copy() for k in audit.FIELDS}
    counts = np.full(6, 3)
    if mutation == "padding":
        a["probabilities"][0, 11] = .01
    elif mutation == "choice":
        a["choice"][0] = 2
    elif mutation == "nan":
        a["probabilities"][0, 0] = np.nan
    else:
        a["labels"][0] = 2
    path = tmp_path / "predictions.npz"
    np.savez(path, **a)
    with pytest.raises(ValueError):
        audit.load_predictions(path, expected, counts)


def ledger_fixture():
    dialogs = [{"turns": [0] * 5, "queries": [{"query": 0}, {"query": 1}]},
               {"turns": [0] * 2, "queries": [{"query": 0}]}]
    witness = audit.empty_witness()
    witness.update(forward_calls=1, forward_returned=1, advance_calls=5, advance_returned=5,
        valid_turns=7, executed_valid_question_slots=14, real_question_updates=12,
        incoming_checks=14, feature_checks=14, result_checks=14, mass_checks=14, mass_min=.1, mass_max=.2)
    row = {"indices": [0, 1], "invariants": witness, "actor_shapes": {"real_turns": 7,
        "padded_turn_positions": 10, "padded_query_positions": 20, "padded_candidate_positions": 80,
        "real_question_steps": 12}}
    return row, dialogs, [{"candidate_ids": [0, 1, 2]}, {"candidate_ids": [0, 1, 2, 3]}]


def test_public_unscored_and_dummy_coverage_independently_counted():
    row, dialogs, queries = ledger_fixture()
    result, shapes, _, orders = audit.audit_ledger([row], dialogs, queries, "scalar", False)
    assert result["real_question_updates"] == 12 and result["feature_checks"] == 14
    assert shapes["padded_query_positions"] == 20 and orders == [[0, 1]]


@pytest.mark.parametrize("mutation", ["coverage", "mass", "duplicate", "tolerance"])
def test_witness_tamper_rejected(mutation):
    row, dialogs, queries = ledger_fixture()
    if mutation == "coverage":
        row["invariants"]["feature_checks"] -= 1
    elif mutation == "mass":
        row["invariants"]["mass_max"] = 1.1
    elif mutation == "duplicate":
        row["indices"] = [0, 0]
    else:
        row["invariants"]["incoming_max_sum_error"] = 3e-6
    with pytest.raises(ValueError):
        audit.audit_ledger([row], dialogs, queries, "scalar", False)


def test_failure_and_exclusive_output_without_reading_study(tmp_path):
    args = SimpleNamespace(run=tmp_path / "absent", report=tmp_path / "also_absent", out=tmp_path / "audit",
        plan_sha256="wrong", completed_sha256="0" * 64, report_receipt_sha256="1" * 64)
    with pytest.raises(ValueError, match="Different frozen study"):
        audit.audit(args)
    failed = audit.read(Path(args.out) / "failed.json")
    assert failed["progress"]["fits_checked"] == [] and failed["authenticated_files"] == {}
    assert not (Path(args.out) / "completed.json").exists()
    with pytest.raises(FileExistsError):
        audit.audit(args)


def test_metrics_family_is_mean_of_three_fits():
    rows = {f"{m}-{s}": audit.metrics(arrays()) for m in audit.METHODS for s in audit.SEEDS}
    base = copy.deepcopy(rows["selective-4101"]["seen"])
    rows["selective-4101"]["seen"]["micro"]["nll"] += .3
    family = audit.families(rows)["selective"]["seen"]
    assert family["micro"]["nll"] == pytest.approx(base["micro"]["nll"] + .1)
    assert family["micro"]["count_per_fit"] == 3
