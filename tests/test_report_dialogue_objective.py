"""Synthetic saved-array, exact-rule and failure-boundary tests only."""
from __future__ import annotations

import copy
import json
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import report_dialogue_objective as report
from test_report_dialogue_alignment import packet, references, rows_fixture


def historical_fixture(rows):
    meta, cells, services = report.layouts(rows)
    logs = packet(rows)["log_probs"].astype(float)
    logs -= np.logaddexp.reduce(logs, axis=1)[:, None]
    values, choice, selected, ties = report.vectors(logs, meta)
    fit = {"cells": {k: report.describe(values, choice, selected, ties, meta, g) for k, g in cells.items()},
           "services": {s: {k: report.describe(values, choice, selected, ties, meta, g) for k, g in gs.items()} for s, gs in services.items()}}
    categories = [f"{m}-{r}" for m in ("flat_stratum", "token_mean", "token_aligned") for r in ("original", "corrected")]
    return {"fits": {f"{c}-{s}": copy.deepcopy(fit) for c in categories for s in report.SEEDS},
            "means": {c: copy.deepcopy(fit) for c in categories}}


def packets_fixture(rows):
    return {name: packet(rows) for name in report.FIT_ORDER}


def cell(correct=50, total=100, nll=1., fp=2):
    return {"rows": total, "counts": {"correct": correct, "error": total-correct},
            "metrics": {"nll": {"row": nll, "equal_service": nll}},
            "rare": {"true_false_positive": {"numerator": fp, "denominator": 200}}}


def gate_fixture():
    fits = {}
    for category in report.CATEGORIES:
        for seed in report.SEEDS:
            u = category == "uniform-original"
            cells = {"all": cell(61 if u else 60, 100, .9 if u else 1.),
                     "changed": cell(52 if u else 50), "retained": cell(91 if u else 90)}
            fits[f"{category}-{seed}"] = {"cells": {"heldout_service/"+k: copy.deepcopy(v) for k, v in cells.items()},
                                            "services": {"s1": copy.deepcopy(cells)}}
    historical = {"fits": {f"flat_stratum-corrected-{s}": copy.deepcopy(fits[f"stratum-corrected-{s}"]) for s in report.SEEDS}}
    return fits, historical


def test_analytic_readouts_use_public_prior_not_gold_and_roundtrip():
    rows = rows_fixture(); p = packet(rows); meta, _, _ = report.layouts(rows)
    sizes = np.array([r["candidate_count"] for r in rows])
    weights = sum(report.COUNTS)/(3*np.array(report.COUNTS))
    for method, sign, variant in (("stratum", -1, "corrected"), ("uniform", 1, "reweighted")):
        got, witness, _ = report.readouts(p["log_probs"], sizes, meta["previous"], meta["types"], method)
        for i, row in enumerate(rows):
            w = np.full(4, weights[2]); w[row["previous_current_index"]] = weights[0 if row["candidate_types"][row["previous_current_index"]] == 0 else 1]
            expected = np.exp(p["log_probs"][i, :4].astype(float))*w**sign
            expected /= expected.sum()
            np.testing.assert_allclose(np.exp(got[variant][i, :4]), expected, atol=1e-15)
        assert witness["raw_maximum_mass_error"] < 2e-6
        restored = got[variant].copy()
        logw = np.full(restored.shape, np.log(weights[2]))
        logw[np.arange(len(rows)), meta["previous"]] = np.log(weights[np.where(meta["types"][np.arange(len(rows)), meta["previous"]] == 0, 0, 1)])
        restored -= sign*logw; restored -= np.logaddexp.reduce(restored, axis=1)[:, None]
        np.testing.assert_allclose(restored, got["original"], atol=1e-14)


def test_finite_extreme_log_nll_is_not_floored_and_selected_branch_is_candidate():
    rows = rows_fixture()[:1]; p = packet(rows)
    p["log_probs"][0, :4] = [0., -1000., -2000., -1000.]
    meta, _, _ = report.layouts(rows)
    outputs, _, _ = report.readouts(p["log_probs"], np.array([4]), meta["previous"], meta["types"], "uniform")
    values, *_ = report.vectors(outputs["original"], meta)
    assert values["nll"][0] == 2000 and values["brier"][0] == 2
    p["log_probs"][0, :4] = np.log([.35, .05, .31, .29])
    values, *_ = report.vectors(p["log_probs"].astype(float), meta)
    assert values["wrong_selected_branch"][0] == 1 and values["wrong_value"][0] == 0


@pytest.mark.parametrize("corrupt", ["mass", "padding", "positive", "nan"])
def test_invalid_distributions_rejected(corrupt):
    rows = rows_fixture(); p = packet(rows); meta, _, _ = report.layouts(rows)
    if corrupt == "mass": p["log_probs"][0, :4] += .01
    if corrupt == "padding": p["log_probs"][0, 4] = -100
    if corrupt == "positive": p["log_probs"][0, 0] = .1
    if corrupt == "nan": p["log_probs"][0, 0] = np.nan
    with pytest.raises(ValueError): report.readouts(p["log_probs"], np.full(len(rows), 4), meta["previous"], meta["types"], "stratum")


def test_all_readouts_primary_secondary_counts_and_history_visible():
    rows = rows_fixture(); packets = packets_fixture(rows)
    summary = report.aggregate(rows, packets, references(rows), historical_fixture(rows))
    assert len(summary["fits"]) == 12 and all(len(f["cells"]) == 15 for f in summary["fits"].values())
    assert len(summary["historical"]["fits"]) == 18
    for label in ("primary", "secondary"):
        for seed in report.SEEDS:
            c = summary["pairs"][label]["seeds"][str(seed)]["cells"]["heldout_service/all"]
            counts = c["paired"]
            assert sum(counts[k] for k in ("wrong_to_correct", "correct_to_wrong", "both_correct", "both_wrong")) == counts["rows"]
            assert counts["wrong_to_correct"]-counts["correct_to_wrong"] == c["count_differences"]["correct"]
    assert json.loads(json.dumps(summary, allow_nan=False))["continuation"]["total_checks"] == 13


@pytest.mark.parametrize("failure", ["missing_fit", "reordered_ids", "prior_mapping"])
def test_atomic_membership_and_row_joins_fail(failure):
    rows = rows_fixture(); packets = packets_fixture(rows)
    if failure == "missing_fit": packets.pop(report.FIT_ORDER[0])
    if failure == "reordered_ids": packets[report.FIT_ORDER[0]]["row_indices"] = packets[report.FIT_ORDER[0]]["row_indices"][::-1]
    if failure == "prior_mapping": rows[0]["previous_current_index"] = 1
    with pytest.raises(ValueError): report.aggregate(rows, packets, references(rows), historical_fixture(rows))


def test_exact_thirteen_rules_inclusive_boundaries():
    fits, history = gate_fixture()
    result = report.continuation(fits, history, ["s1"])
    assert result["passed"] and result["checks_passed"] == result["total_checks"] == 13
    assert result["checks"]["objective_changed_accuracy_row"]["mean_difference"] == .02
    for seed in report.SEEDS:
        f = fits[f"uniform-original-{seed}"]
        for c in (f["cells"]["heldout_service/all"], f["services"]["s1"]["all"]):
            c["rows"] = 400; c["counts"] = {"correct": 241, "error": 159}
        h = history["fits"][f"flat_stratum-corrected-{seed}"]
        for c in (h["cells"]["heldout_service/all"], h["services"]["s1"]["all"]):
            c["rows"] = 400; c["counts"] = {"correct": 240, "error": 160}
        f["cells"]["heldout_service/all"]["rare"]["true_false_positive"]["numerator"] = 3
    result = report.continuation(fits, history, ["s1"])
    assert result["checks"]["practical_accuracy_row"]["passed"]
    assert result["checks"]["practical_true_false_positive"]["passed"]


def test_joint_seed_intersection_not_separate_favorable_sets():
    fits, history = gate_fixture()
    # 6201 fails only service change; 6202 fails only row change. Each view has
    # two favorable seeds but only 6203 satisfies their intersection.
    fits["uniform-original-6201"]["services"]["s1"]["changed"]["counts"] = {"correct": 50, "error": 50}
    fits["uniform-original-6202"]["cells"]["heldout_service/changed"]["counts"] = {"correct": 50, "error": 50}
    got = report.continuation(fits, history, ["s1"])
    assert got["checks"]["objective_joint_seeds"]["common_seeds"] == 1
    assert not got["checks"]["objective_joint_seeds"]["passed"]
    fits, history = gate_fixture()
    fits["uniform-original-6201"]["services"]["s1"]["all"]["metrics"]["nll"]["equal_service"] = 2.
    fits["uniform-original-6201"]["cells"]["heldout_service/all"]["metrics"]["nll"]["equal_service"] = 2.
    fits["uniform-original-6202"]["cells"]["heldout_service/all"]["metrics"]["nll"]["row"] = 2.
    assert report.continuation(fits, history, ["s1"])["checks"]["practical_joint_seeds"]["common_seeds"] == 1


def test_equal_service_rates_are_rational_not_window_pooled():
    fit = {"cells": {"heldout_service/all": cell(9, 11)}, "services": {"a": {"all": cell(9, 10)}, "b": {"all": cell(0, 1)}}}
    assert report.exact_rate(fit, "all", "accuracy", "row", ["a", "b"]) == Fraction(9, 11)
    assert report.exact_rate(fit, "all", "accuracy", "equal_service", ["a", "b"]) == Fraction(9, 20)


@pytest.mark.parametrize("kind", ["zero_support", "nonfinite", "true_false_positive"])
def test_undefined_or_harmful_required_metrics_cannot_pass(kind):
    fits, history = gate_fixture()
    if kind == "zero_support":
        for fit in list(fits.values())+list(history["fits"].values()):
            fit["cells"]["heldout_service/changed"] = cell(0, 0)
            fit["services"]["s1"]["changed"] = cell(0, 0)
    if kind == "nonfinite": fits["uniform-original-6201"]["cells"]["heldout_service/all"]["metrics"]["nll"]["row"] = float("nan")
    if kind == "true_false_positive":
        for seed in report.SEEDS: fits[f"uniform-original-{seed}"]["cells"]["heldout_service/all"]["rare"]["true_false_positive"]["numerator"] = 4
    assert not report.continuation(fits, history, ["s1"])["passed"]


def test_external_digest_fails_before_decoding(tmp_path, monkeypatch):
    (tmp_path/"completed.json").write_text("{}")
    monkeypatch.setattr(report, "read", lambda _: pytest.fail("No JSON before external pin"))
    with pytest.raises(ValueError, match="External completed"): report.authenticate_run(tmp_path, "0"*64)


def test_manifest_tamper_and_extra_member_are_rejected(tmp_path):
    (tmp_path/"x").write_bytes(b"one")
    manifest = {"x": {"sha256": report.digest(tmp_path/"x"), "bytes": 3}}
    (tmp_path/"completed.json").write_text("{}")
    report.check_manifest(tmp_path, manifest, {"x"})
    (tmp_path/"x").write_bytes(b"two")
    with pytest.raises(ValueError): report.check_manifest(tmp_path, manifest, {"x"})
    (tmp_path/"x").write_bytes(b"one"); (tmp_path/"foreign").write_text("extra")
    with pytest.raises(ValueError): report.check_manifest(tmp_path, manifest, {"x"})


def test_exclusive_failure_receipt_preserves_original(tmp_path, monkeypatch):
    out = tmp_path/"report"; args = SimpleNamespace(run=tmp_path/"run", run_sha256="bad", out=out)
    def failure(*_): raise ValueError("injected auth failure")
    monkeypatch.setattr(report, "authenticate_run", failure)
    with pytest.raises(ValueError, match="injected auth"): report.execute(args)
    failed = json.loads((out/"failed.json").read_text())
    assert failed["model_calls"] == 0 and failed["status"] == "failed"
    with pytest.raises(FileExistsError): report.execute(args)
    real_write = report.write
    def fail_receipt(path, value):
        if path.name == "failed.json": raise OSError("secondary disk issue")
        real_write(path, value)
    monkeypatch.setattr(report, "write", fail_receipt); args.out = tmp_path/"other"
    with pytest.raises(ValueError, match="injected auth") as caught: report.execute(args)
    assert "secondary disk issue" in caught.value.__notes__[0]


def test_authentication_code_has_no_model_or_float_cache_loading():
    import ast
    tree = ast.parse(report.Path(report.__file__).read_text())
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(v and (v.startswith("torch") or "study_dialogue" in v or "diagnose_dialogue" in v) for v in imports)


def test_tiny_actual_runner_saved_packets_to_report(tmp_path, monkeypatch):
    # Six one-update synthetic fits; production authentication is not bypassed
    # in the reporter. This test exercises only its pure saved-array API.
    import study_dialogue_objective as runner
    import torch
    from test_study_dialogue_objective import synthetic_run

    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        args = synthetic_run(tmp_path, monkeypatch)
        runner.execute(args)
        rows = [report.base.decode(line) for line in (args.out/"evaluation-rows.jsonl").read_text().splitlines()]
        packets = {}
        for name in report.FIT_ORDER:
            with np.load(args.out/"fits"/name/"predictions.npz", allow_pickle=False) as z:
                packets[name] = {k: z[k] for k in z.files}
        with np.load(args.out/"references.npz", allow_pickle=False) as z:
            refs = {k: z[k] for k in z.files}
        summary = report.aggregate(rows, packets, refs, historical_fixture(rows))
        assert len(summary["fits"]) == 12 and summary["continuation"]["total_checks"] == 13
        for seed in report.SEEDS:
            raw = packets[f"uniform-{seed}"]["log_probs"].astype(float)
            target = np.array([r["current_label_index"] for r in rows])
            expected = -raw[np.arange(len(rows)), target] + np.logaddexp.reduce(raw, axis=1)
            actual = summary["fits"][f"uniform-original-{seed}"]["cells"]["all/all"]["metrics"]["nll"]["row"]
            assert actual == pytest.approx(expected.mean(), abs=1e-14)
    finally:
        torch.set_num_threads(original_threads)


def test_saved_only_report_completion_receipt_and_text(tmp_path, monkeypatch):
    rows = rows_fixture(); packets = packets_fixture(rows); refs = references(rows)
    run = tmp_path/"run"; run.mkdir()
    for name, values in packets.items():
        dest = run/"fits"/name; dest.mkdir(parents=True)
        np.savez(dest/"predictions.npz", **values)
    np.savez(run/"references.npz", **refs)
    (run/"completed.json").write_text("{}")
    done = {"files": {"evaluation-rows.jsonl": {"sha256": "row-pin"}}, "plan_sha256": "plan-pin",
            "wall_seconds": 8., "process_lifetime_peak_rss_bytes": 100}
    records = {name: {"wall_seconds": 1., "training_wall_seconds": .8, "evaluation": {"wall_seconds": .1, "work": {}},
                     "configuration": {"parameters": 124482}, "counts": {}, "training_work": {}} for name in report.FIT_ORDER}
    monkeypatch.setattr(report, "authenticate_run", lambda *a: (rows, done, {"source_sha256": {}, "split": {}}, records, 100, {"wall_seconds": 1.}))
    monkeypatch.setattr(report, "historical_inputs", lambda *a: {**historical_fixture(rows), "provenance": {}})
    monkeypatch.setattr(report, "check_manifest", lambda *a: None)
    monkeypatch.setattr(report.old, "peak_rss", lambda: 100)
    args = SimpleNamespace(run=run, run_sha256=report.digest(run/"completed.json"), out=tmp_path/"report")
    report.execute(args)
    receipt = report.read(args.out/"receipt.json")
    assert set(receipt["files"]) == {"started.json", "summary.json", "report.md"}
    assert receipt["technical_validity_passed"] and receipt["model_calls"] == receipt["checkpoint_deserializations"] == 0
    assert "13 checks" in (args.out/"report.md").read_text()
    for name, item in receipt["files"].items(): assert item["sha256"] == report.digest(args.out/name)
