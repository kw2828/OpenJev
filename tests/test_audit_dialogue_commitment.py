"""Closed-form saved-arithmetic and artifact-corruption checks, no real data."""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1]/"scripts/audit_dialogue_commitment.py"
SPEC = importlib.util.spec_from_file_location("commitment_audit_test", SOURCE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def logs(values, width=12):
    result = np.full((len(values), width), -np.inf, np.float32)
    for i, row in enumerate(values):
        result[i, :len(row)] = np.log(row)
    return result


def row(i, target, previous, *, service="A", types=(0, 1, 2, 4)):
    transition = ("unmentioned_retention" if target == previous and types[target] == 0 else
                  "assigned_retention" if target == previous else "first_assignment" if types[previous] == 0 else
                  "clear" if types[target] == 0 else "revision")
    return {"row_index": i, "candidate_count": len(types), "candidate_types": list(types),
            "current_label_index": target, "previous_current_index": previous,
            "current_candidate_id": "id"+str(target), "previous_candidate_id": "id"+str(previous),
            "heldout_service": True, "split": "train", "admission": "admitted",
            "derived_bin": transition, "current_value_group": ("none", "dontcare", "true", "false", "other")[types[target]],
            "service": service, "dialogue_id": "d"+str(i)}


def test_closed_form_cross_uses_mass_and_conditional_concentration():
    mask = np.asarray([[True]*4+[False]*8])
    m, _, _ = audit.normalize(logs([[.4, .2, .2, .2]]), mask)
    a, _, _ = audit.normalize(logs([[.1, .72, .09, .09]]), mask)
    result, _ = audit.hybrid(m, a, np.array([0]), mask)
    np.testing.assert_allclose(np.exp(result[0, :4]), [.4, .48, .06, .06], atol=3e-8, rtol=0)
    assert np.argmax(m) == 0 and np.argmax(result) == 1


def test_two_candidates_have_no_alternative_ranking_freedom():
    mask = np.asarray([[True, True]+[False]*10])
    m, _, _ = audit.normalize(logs([[.7, .3]]), mask)
    a, _, _ = audit.normalize(logs([[.2, .8]]), mask)
    ma, _ = audit.hybrid(m, a, np.array([0]), mask)
    am, _ = audit.hybrid(a, m, np.array([0]), mask)
    np.testing.assert_allclose(ma[mask], m[mask], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(am[mask], a[mask], rtol=1e-12, atol=1e-10)


def test_tiny_change_mass_and_underflowed_gold_keep_finite_nll():
    rows = [row(0, 2, 0)]
    meta = audit.metadata(rows)
    raw = np.full((1, 12), -np.inf, np.float32)
    raw[0, :4] = [0, -1000, -1001, -1002]
    q, _, _ = audit.normalize(raw, meta["mask"])
    h, _ = audit.hybrid(q, q, meta["previous"], meta["mask"], self_cell=True)
    _, values, _ = audit.score(h, meta)
    assert np.isfinite(h[meta["mask"]]).all()
    assert values["nll"].tolist() == [1001.]
    assert values["brier"].tolist() == [2.]


def test_self_scores_canonical_logs_and_exact_first_ties():
    rows = [row(0, 1, 3)]
    meta = audit.metadata(rows)
    q, _, _ = audit.normalize(logs([[.4, .4, .1, .1]]), meta["mask"])
    h, _ = audit.hybrid(q, q, meta["previous"], meta["mask"], self_cell=True)
    assert np.array_equal(q, h)
    choice, _, ties = audit.score(h, meta)
    assert choice.tolist() == [0] and ties.tolist() == [True]


def test_selected_branch_is_not_maximum_branch_mass():
    rows = [row(0, 2, 0)]
    meta = audit.metadata(rows)
    q, _, _ = audit.normalize(logs([[.35, .05, .3, .3]]), meta["mask"])
    _, values, _ = audit.score(q, meta)
    assert values["wrong_selected_branch"].tolist() == [1.]
    assert values["wrong_value"].tolist() == [0.]


def test_primary_partition_weighting_and_absent_rare_support():
    rows = [row(0, 0, 0), row(1, 2, 2), row(2, 3, 2, service="B")]
    meta = audit.metadata(rows)
    masks = audit.primary_masks(meta)
    assert {k: int(m.sum()) for k, m in masks.items()} == {
        "all": 3, "changed": 1, "retained": 2, "unmentioned_retention": 1, "assigned_retention": 1}
    value = audit.means(np.array([0., 0., 1.]), rows, masks["all"])
    assert value == {"row": 1/3, "equal_service": .5, "equal_dialogue": 1/3}
    q, _, _ = audit.normalize(logs([[.7, .1, .1, .1]]*3), meta["mask"])
    choice, vectors, ties = audit.score(q, meta)
    result = audit.cell(rows, meta, choice, vectors, ties, masks["changed"])
    assert result["rare"]["true_recall"] == {"numerator": 0, "denominator": 0, "rate": None}
    assert result["rare"]["true_false_positive"]["denominator"] == 1


def test_paired_repairs_harms_reconcile_to_accuracy_change():
    rows = [row(i, 2, 0) for i in range(4)]
    target = np.array([2]*4)
    a, b = np.array([2, 0, 2, 0]), np.array([2, 2, 0, 0])
    result = audit.paired(rows, target, a, b,
                         {"accuracy": (a == target).astype(float)},
                         {"accuracy": (b == target).astype(float)}, np.ones(4, bool))
    assert result["paired"] == {"rows": 4, "wrong_to_correct": 1, "correct_to_wrong": 1, "both_correct": 1, "both_wrong": 1}
    assert result["metric_differences"]["accuracy"]["row"] == 0


@pytest.mark.parametrize("corruption", ("support", "mass", "dtype"))
def test_invalid_source_probabilities_are_not_repaired(corruption):
    raw = logs([[.7, .1, .1, .1]])
    mask = np.isfinite(raw)
    if corruption == "support":
        raw[0, 5] = -5
    elif corruption == "mass":
        raw[0, 0] = 0
    else:
        raw = raw.astype(np.float64)
    with pytest.raises(ValueError):
        audit.normalize(raw, mask)


def test_input_row_mapping_and_metric_tampering_rejected():
    rows = [row(0, 2, 0)]
    rows[0]["previous_candidate_id"] = rows[0]["current_candidate_id"]
    with pytest.raises(ValueError, match="mapping"):
        audit.metadata(rows)
    with pytest.raises(ValueError, match="Exact mismatch"):
        audit.compare({"correct_to_wrong": 2}, {"correct_to_wrong": 1})
    with pytest.raises(ValueError, match="Numeric mismatch"):
        audit.compare({"nll": float("nan")}, {"nll": 1.})
    with pytest.raises(ValueError, match="All nine"):
        audit.aggregate([], {}, {})


def test_manifest_rejects_mutation_and_foreign_member(tmp_path):
    audit.write(tmp_path/"receipt.json", {})
    audit.write(tmp_path/"summary.json", {"count": 1})
    members = {"summary.json": audit.item(tmp_path/"summary.json")}
    audit.manifest(tmp_path, members, terminal="receipt.json")
    audit.write(tmp_path/"foreign.json", {})
    with pytest.raises(ValueError, match="membership"):
        audit.manifest(tmp_path, members, terminal="receipt.json")
    (tmp_path/"foreign.json").unlink()
    (tmp_path/"summary.json").write_text('{"count": 2}\n')
    with pytest.raises(ValueError, match="Payload identity"):
        audit.manifest(tmp_path, members, terminal="receipt.json")


def test_log_normalization_shift_matches_direct_target_nll():
    mask = np.asarray([[True]*3+[False]*9])
    raw = logs([[.5, .3, .2]])
    raw[0, :3] += np.float32(1e-7)
    q, z, witness = audit.normalize(raw, mask)
    assert math.isclose(-q[0, 1]+float(raw[0, 1]), z[0], abs_tol=1e-15)
    assert witness["max_abs_log_z"] > 0


def test_failed_authentication_is_preserved_and_directory_cannot_be_reused(tmp_path):
    run, diagnostic, out = (tmp_path/name for name in ("run", "diagnostic", "out"))
    run.mkdir(); diagnostic.mkdir()
    audit.write(run/"completed.json", {})
    args = argparse.Namespace(run=str(run), diagnostic=str(diagnostic), out=str(out),
                              run_sha256="0"*64, summary_sha256="0"*64, receipt_sha256="0"*64)
    with pytest.raises(ValueError, match="fixed run"):
        audit.execute(args)
    assert audit.read(out/"failed.json")["status"] == "failed"
    assert not (out/"receipt.json").exists()
    with pytest.raises(FileExistsError):
        audit.execute(args)


def test_all_nine_artificial_producer_outputs_match_independent_arithmetic():
    # Producer imports are confined to this integration test. The auditor has
    # no dependency on its implementation or its metric helpers.
    spec = importlib.util.spec_from_file_location("commitment_producer_fixture",
        SOURCE.parent/"diagnose_dialogue_commitment.py")
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    original = producer.load_portable().load_arithmetic()
    rows = [row(i, y, previous, service="A" if i < 4 else "B", types=(0, 1, 2, 3))
            for i, (y, previous) in enumerate(((2, 0), (1, 0), (0, 0), (2, 2),
                                             (2, 0), (3, 3), (0, 0), (1, 2)))]
    for i in range(4, 8):
        rows[i]["heldout_service"] = False
    packets = {}
    for name in audit.PREDICTION_PINS:
        probabilities = ([.5, .1, .3, .1] if name.startswith("token_mean") else
                         [.2, .1, .6, .1] if name.startswith("token_aligned") else [.1, .4, .2, .3])
        packets[name] = {"row_indices": np.arange(len(rows), dtype=np.int64),
                         "log_probs": logs([probabilities]*len(rows))}
    result = producer.analyze(rows, original.row_metadata(rows), packets, original)
    checked = audit.aggregate(rows, packets, result)
    assert checked["counts"] == {"original_fits": 9, "constructions": 15, "primary_cells": 75,
                                 "paired_cells": 225, "rows_per_fit": 8,
                                 "primary_support": {"all": 4, "changed": 2, "retained": 2,
                                                     "unmentioned_retention": 1, "assigned_retention": 1}}
    assert checked["scalar_comparisons"] > 10000
    result["fits"]["MA-6203"]["cells"]["heldout_service/retained"]["counts"]["error"] += 1
    with pytest.raises(ValueError, match="Exact mismatch"):
        audit.aggregate(rows, packets, result)
    packets["token_aligned-6203"]["row_indices"] = np.arange(len(rows), dtype=np.int64)[::-1]
    with pytest.raises(ValueError, match="row identity"):
        audit.aggregate(rows, packets, result)


def test_selected_bytes_authenticated_without_opening_checkpoint_or_rows(tmp_path, monkeypatch):
    run = tmp_path/"run"; run.mkdir()
    diagnostic = tmp_path/"diagnostic"; diagnostic.mkdir()
    audit.write(diagnostic/"summary.json", {})
    audit.write(diagnostic/"receipt.json", {})
    audit.write(run/"plan.json", {"source_sha256": {}})
    (run/"evaluation-rows.jsonl").write_bytes(b"must not decode")
    members = {"plan.json": audit.item(run/"plan.json"),
               "evaluation-rows.jsonl": audit.item(run/"evaluation-rows.jsonl")}
    phantom = {"sha256": "0"*64, "bytes": 100}
    for name in ("started.json", "references.npz", *(f"orders-{s}.npy" for s in audit.SEEDS)):
        members[name] = phantom
    predictions = {}
    for name in audit.PREDICTION_PINS:
        directory = run/"fits"/name; directory.mkdir(parents=True)
        path = directory/"predictions.npz"; path.write_bytes(b"opaque synthetic prediction")
        members[f"fits/{name}/predictions.npz"] = audit.item(path)
        predictions[name] = audit.sha(path)
        for file in ("completed.json", "updates.jsonl", "weights.pt"):
            members[f"fits/{name}/{file}"] = phantom  # Deliberately nonexistent, outside metric audit.
    plan_pin, rows_pin = audit.sha(run/"plan.json"), audit.sha(run/"evaluation-rows.jsonl")
    audit.write(run/"completed.json", {"status": "completed", "completed_fits": list(predictions),
                "plan_sha256": plan_pin, "source_sha256": {}, "files": members})
    run_pin = audit.sha(run/"completed.json")
    monkeypatch.setattr(audit, "RUN_PIN", run_pin)
    monkeypatch.setattr(audit, "PLAN_PIN", plan_pin)
    monkeypatch.setattr(audit, "ROWS_PIN", rows_pin)
    monkeypatch.setattr(audit, "PREDICTION_PINS", predictions)
    (run/"fits"/"token_aligned-6203"/"predictions.npz").write_bytes(b"corrupted")
    def forbidden(*_args, **_kwargs):
        pytest.fail("Array decoding before selected payload authentication")
    monkeypatch.setattr(audit.np, "load", forbidden)
    args = argparse.Namespace(run=run, run_sha256=run_pin, diagnostic=diagnostic,
        summary_sha256=audit.sha(diagnostic/"summary.json"), receipt_sha256=audit.sha(diagnostic/"receipt.json"))
    with pytest.raises(ValueError, match="Selected metric payload"):
        audit.authenticate(args)
