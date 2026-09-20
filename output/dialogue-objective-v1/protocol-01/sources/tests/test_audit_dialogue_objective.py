"""Artificial distributions, rational gates and a relocated selected-byte tree."""
from __future__ import annotations

import copy
import json
from fractions import Fraction
from types import SimpleNamespace

import audit_dialogue_objective as a
import numpy as np
import pytest


def rows_fixture():
    rows = []
    ids = ("reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:true", "value:false")
    for i, (previous, target, service, held) in enumerate([
        (0, 2, "A", True), (0, 1, "A", True), (0, 0, "A", True), (2, 2, "A", True),
        (0, 2, "B", True), (0, 0, "B", True), (2, 0, "C", False)]):
        transition = ("unmentioned_retention" if previous == target == 0 else
                      "assigned_retention" if previous == target else "first_assignment" if previous == 0 else
                      "clear" if target == 0 else "revision")
        rows.append({"row_index": i, "split": "train", "admission": "admitted", "candidate_count": 4,
            "candidate_types": [0, 1, 2, 3], "current_label_index": target, "previous_current_index": previous,
            "current_candidate_id": ids[target], "previous_candidate_id": ids[previous],
            "current_value_group": ("none", "dontcare", "true", "false")[target], "derived_bin": transition,
            "heldout_service": held, "service": service, "dialogue_id": f"d{i//2}", "query_id": service+"/q", "query_index": 0})
    return rows


def raw(probabilities):
    result = np.full((len(probabilities), 12), -np.inf, np.float32)
    for i, p in enumerate(probabilities): result[i, :len(p)] = np.log(p)
    return result


def fixtures():
    h = a.load_helper(); rows = rows_fixture(); meta = h.metadata(rows)
    packets = {}
    for i, name in enumerate(a.FIT_ORDER):
        p = [.55, .05, .3, .1] if name.startswith("stratum") else [.32, .08, .5, .1]
        values = [p.copy() for _ in rows]; values[i % len(rows)] = [.1, .2, .6, .1]
        packets[name] = {"row_indices": np.arange(len(rows), dtype=np.int64), "log_probs": raw(values)}
    q, _, _ = h.normalize(raw([[.6, .1, .2, .1]]*len(rows)), meta["mask"])
    choice, values, ties = h.score(q, meta)
    strata = {"all": np.ones(len(rows), bool), "changed": meta["labels"] != meta["previous"]}
    strata.update(retained=~strata["changed"], unmentioned_retention=~strata["changed"] & (meta["target_types"] == 0),
                  assigned_retention=~strata["changed"] & (meta["target_types"] != 0))
    populations = {"all": np.ones(len(rows), bool), "heldout_service": meta["held"], "seen_service": ~meta["held"]}
    fit = {"cells": {f"{panel}/{s}": h.cell(rows, meta, choice, values, ties, pop & mask)
                     for panel, pop in populations.items() for s, mask in strata.items()},
           "services": {service: {s: h.cell(rows, meta, choice, values, ties,
                np.asarray([r["service"] == service for r in rows]) & mask) for s, mask in strata.items()}
                        for service in ("A", "B", "C")}}
    categories = [f"{m}-{r}" for m in ("flat_stratum", "token_mean", "token_aligned") for r in ("original", "corrected")]
    historical = {"status": "completed", "fits": {f"{c}-{s}": copy.deepcopy(fit) for c in categories for s in a.SEEDS},
                  "means": {c: copy.deepcopy(fit) for c in categories}}
    return rows, packets, historical


@pytest.mark.parametrize("direction", [-1, 1])
def test_odds_shift_closed_form_and_extreme_logs(direction):
    h = a.load_helper(); rows = rows_fixture()[:2]; rows[1]["previous_current_index"] = 2
    mask = np.zeros((2, 12), bool); mask[:, :4] = True
    logs = raw([[.4, .1, .3, .2], [1, 1e-30, 1e-20, 1e-10]])
    # Construct an exactly normalized extreme source without exponentiating -1000.
    logs[1, :4] = [0, -1000, -1001, -1002]
    q, _, _ = h.normalize(logs, mask)
    previous = np.array([0, 2], np.int64); types = np.tile(np.array([0, 1, 2, 3]+[-1]*8), (2, 1))
    shifted = a.shifted(q, previous, types, mask, direction)
    for i, j in enumerate(previous):
        keep = direction*np.log(a.COUNTS[0 if j == 0 else 1]/a.COUNTS[2])
        alt = [k for k in range(4) if k != j]
        assert shifted[i, j]-np.logaddexp.reduce(shifted[i, alt]) == pytest.approx(
            q[i, j]-np.logaddexp.reduce(q[i, alt])+keep, abs=1e-10)
        np.testing.assert_allclose(shifted[i, alt]-np.logaddexp.reduce(shifted[i, alt]),
                                   q[i, alt]-np.logaddexp.reduce(q[i, alt]), atol=1e-10)
    assert np.isfinite(shifted[mask]).all() and np.isneginf(shifted[~mask]).all()
    np.testing.assert_allclose(np.exp(shifted).sum(1), 1, atol=1e-12)


def test_sign_roundtrip_padding_and_first_alternative_ties():
    mask = np.array([[True]*3+[False]*9, [True]*4+[False]*8])
    types = np.array([[0, 1, 4]+[-1]*9, [2, 0, 1, 4]+[-1]*8])
    previous = np.array([2, 1], np.int64)
    q, _, _ = a.load_helper().normalize(raw([[.45, .45, .1], [.4, .1, .1, .4]]), mask)
    result = a.shifted(a.shifted(q, previous, types, mask, 1), previous, types, mask, -1)
    np.testing.assert_allclose(q[mask], result[mask], atol=1e-12)
    assert result.argmax(1).tolist() == [0, 0]
    with pytest.raises(ValueError): a.shifted(q, previous+10, types, mask, 1)


def gate_fixture():
    fits, exact = {}, {}
    for category in a.READOUTS:
        for seed in a.SEEDS:
            key = f"{category}-{seed}"
            fit = {"cells": {"heldout_service/all": {"metrics": {"nll": {"row": 1., "equal_service": 1.}},
                "rare": {"true_false_positive": {"numerator": 3, "denominator": 200}}}}}
            fits[key] = fit
            acc = {"all": Fraction(241, 400), "changed": Fraction(52, 100), "retained": Fraction(9, 10)}
            if category == "stratum-corrected": acc["changed"] = Fraction(1, 2)
            exact[key] = {s: {"row": v, "equal_service": v} for s, v in acc.items()}
    hfit = {"cells": {"heldout_service/all": {"rows": 400, "counts": {"correct": 240},
        "metrics": {"nll": {"row": 1., "equal_service": 1.}},
        "rare": {"true_false_positive": {"numerator": 2, "denominator": 200}}}},
        "services": {"A": {"all": {"rows": 400, "counts": {"correct": 240}}}}}
    historical = {"fits": {f"flat_stratum-corrected-{s}": copy.deepcopy(hfit) for s in a.SEEDS}}
    return fits, exact, historical


def test_exact_thirteen_inclusive_boundaries_and_same_seed_conjunction():
    fits, exact, hist = gate_fixture()
    got = a.decisions(fits, exact, hist, ["A"])
    assert got["checks_passed"] == got["total_checks"] == 13 and got["passed"]
    for seed in a.SEEDS:
        exact[f"uniform-original-{seed}"]["changed"]["row"] -= Fraction(1, 1000000)
    assert not a.decisions(fits, exact, hist, ["A"])["checks"]["objective_changed_accuracy_row"]["passed"]
    fits, exact, hist = gate_fixture()
    # Both weightings each have two favorable seeds, but only one is common.
    exact["uniform-original-6201"]["changed"]["row"] = Fraction(1, 2)
    exact["uniform-original-6203"]["changed"]["equal_service"] = Fraction(1, 2)
    assert not a.decisions(fits, exact, hist, ["A"])["checks"]["objective_joint_seeds"]["passed"]


def test_nll_and_rare_false_positive_safeguards_fail_without_tolerance():
    fits, exact, hist = gate_fixture()
    fits["uniform-original-6201"]["cells"]["heldout_service/all"]["metrics"]["nll"]["row"] += 1e-9
    got = a.decisions(fits, exact, hist, ["A"])
    assert not got["checks"]["objective_nll_row"]["passed"]
    assert not got["checks"]["practical_nll_row"]["passed"]
    fits, exact, hist = gate_fixture()
    fits["uniform-original-6201"]["cells"]["heldout_service/all"]["rare"]["true_false_positive"]["numerator"] = 4
    assert not a.decisions(fits, exact, hist, ["A"])["checks"]["practical_true_false_positive"]["passed"]


def producer_fixture():
    import report_dialogue_objective as producer  # Test-only independent compatibility.
    rows, packets, hist = fixtures()
    refs = {"row_indices": np.arange(len(rows), dtype=np.int64),
            "previous_indices": np.array([r["previous_current_index"] for r in rows], np.int64),
            "literal_indices": np.zeros(len(rows), np.int64)}
    result = producer.aggregate(rows, packets, refs, hist)
    return rows, packets, hist, result


def test_real_producer_synthetic_arithmetic_all_twelve_cells_and_pairs():
    rows, packets, historical, producer = producer_fixture()
    h = a.load_helper(); result = a.reconstruct(rows, packets, historical, h)
    assert result["counts"]["primary_cells"] == 60 and result["counts"]["paired_cells"] == 30
    assert a.verify(result, producer, h) > 3000


@pytest.mark.parametrize("corrupt", ["cell", "gate", "pair", "missing"])
def test_bad_report_metrics_or_decisions_rejected(corrupt):
    rows, packets, historical, producer = producer_fixture()
    result = a.reconstruct(rows, packets, historical, a.load_helper())
    if corrupt == "cell": producer["fits"]["uniform-original-6201"]["cells"]["heldout_service/changed"]["counts"]["correct"] += 1
    elif corrupt == "gate": producer["continuation"]["checks"]["objective_nll_row"]["passed"] ^= True
    elif corrupt == "pair": producer["pairs"]["primary"]["seeds"]["6201"]["cells"]["heldout_service/all"]["paired"]["wrong_to_correct"] += 1
    else: producer["fits"].pop("uniform-original-6203")
    with pytest.raises(ValueError): a.verify(result, producer, a.load_helper())


def artifact_fixture(tmp_path, monkeypatch):
    """Moved artificial32-file run, independent of cached absolute input paths."""
    h = a.load_helper(); rows, packets, historical, producer = producer_fixture()
    root, run, report = (tmp_path/n for n in ("repo", "moved-run", "report"))
    for p in (root, run, report): p.mkdir()
    sources = {}
    for name in a.REQUIRED_SOURCES:
        dest = root/name; dest.parent.mkdir(parents=True, exist_ok=True)
        source = a.ROOT/name
        dest.write_bytes(source.read_bytes() if source.exists() else b"synthetic source\n")
        sources[name] = h.sha(dest)
    plan = {"version": "dialogue-objective-v1", "expected_fits": list(a.FIT_ORDER), "limits": a.TRAIN_LIMITS,
            "model_method": "token_aligned", "objective": {"stratum_counts": list(a.COUNTS)},
            "source_sha256": sources, "runtime": {"test": True}, "evaluation_row_indices": list(range(len(rows)))}
    h.write(run/"plan.json", plan)
    (run/"evaluation-rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    h.write(run/"started.json", {})
    (run/"references.npz").write_bytes(b"opaque unused")
    for seed in a.SEEDS: (run/f"orders-{seed}.npy").write_bytes(b"opaque never decoded")
    for name in a.FIT_ORDER:
        dest = run/"fits"/name; dest.mkdir(parents=True)
        np.savez(dest/"predictions.npz", **packets[name])
        (dest/"weights.pt").write_bytes(b"not a valid checkpoint")
        (dest/"updates.jsonl").write_bytes(b"not valid JSON and never opened")
        method, seed = name.split("-")
        h.write(dest/"completed.json", {"status": "completed", "version": "dialogue-objective-v1",
            "method": method, "objective_weighting": method, "model_method": "token_aligned", "seed": int(seed),
            "plan_sha256": h.sha(run/"plan.json"), "files": {n: h.item(dest/n) for n in ("weights.pt", "updates.jsonl", "predictions.npz")}})
    done = {"status": "completed", "version": "dialogue-objective-v1", "phase": "train",
            "completed_fits": list(a.FIT_ORDER), "expected_fits": list(a.FIT_ORDER), "limits": a.TRAIN_LIMITS,
            "no_retry": True, "quality_metrics_computed": False, "encoder_calls": 0, "official_dev_inference": False,
            "test_contents_accessed": False, "wall_seconds": 2., "process_lifetime_peak_rss_bytes": 100,
            "plan_sha256": h.sha(run/"plan.json"), "source_sha256": sources, "runtime": plan["runtime"],
            "files": {p.relative_to(run).as_posix(): h.item(p) for p in run.rglob("*") if p.is_file()}}
    h.write(run/"completed.json", done)
    producer.update(status="completed", version=a.REPORT_VERSION, technical_validity_passed=True,
                    execution_completed_sha256=h.sha(run/"completed.json"), plan_sha256=h.sha(run/"plan.json"),
                    source_sha256=sources)
    h.write(report/"summary.json", producer); h.write(report/"started.json", {})
    (report/"report.md").write_text("Artificial report")
    h.write(report/"receipt.json", {"status": "completed", "version": a.REPORT_VERSION,
        "execution_completed_sha256": h.sha(run/"completed.json"), "plan_sha256": h.sha(run/"plan.json"),
        "files": {n: h.item(report/n) for n in ("started.json", "summary.json", "report.md")},
        "technical_validity_passed": True, "execution_members": done["files"],
        "source_sha256": {"scripts/report_dialogue_objective.py": sources["scripts/report_dialogue_objective.py"], **a.REPORT_HELPERS},
        "model_calls": 0, "encoder_calls": 0, "checkpoint_deserializations": 0, "no_retry": True,
        "limits": a.LIMITS, "wall_seconds": 1., "process_lifetime_peak_rss_bytes": 100,
        "continuation_passed": producer["continuation"]["passed"]})
    hd = root/a.HISTORICAL/"diagnostic-01"; hd.mkdir(parents=True)
    ha = root/a.HISTORICAL/"audit-01"; ha.mkdir()
    h.write(hd/"summary.json", historical)
    h.write(hd/"receipt.json", {"status": "completed", "files": {"summary.json": h.item(hd/"summary.json")},
                                     "input_sha256": {"rows": h.sha(run/"evaluation-rows.jsonl")}})
    h.write(ha/"receipt.json", {"status": "completed", "agreement": True,
        "producer_summary_sha256": h.sha(hd/"summary.json"), "producer_receipt_sha256": h.sha(hd/"receipt.json")})
    monkeypatch.setattr(a, "ROOT", root)
    monkeypatch.setattr(a, "ROWS_PIN", h.sha(run/"evaluation-rows.jsonl"))
    monkeypatch.setattr(a, "EXPECTED_ROWS", len(rows))
    monkeypatch.setattr(a, "EXPECTED_SUPPORT", {k: int(v.sum()) for k, v in h.primary_masks(h.metadata(rows)).items()})
    monkeypatch.setattr(a, "HISTORICAL_PINS", {n: h.sha(root/a.HISTORICAL/n) for n in a.HISTORICAL_PINS})
    return SimpleNamespace(run=run, completed_sha256=h.sha(run/"completed.json"), report=report,
                           report_receipt_sha256=h.sha(report/"receipt.json"), out=tmp_path/"audit")


def test_relocated_complete_artifacts_execute_without_training_payload_decoding(tmp_path, monkeypatch):
    args = artifact_fixture(tmp_path, monkeypatch)
    result = a.execute(args)
    assert result["agreement"] is True and result["counts"]["readouts"] == 12
    assert a.load_helper().read(args.out/"receipt.json")["model_calls"] == 0
    with pytest.raises(FileExistsError): a.execute(args)


@pytest.mark.parametrize("corruption", ["prediction", "receipt", "incomplete"])
def test_authentication_fails_before_npz_loading_and_preserves_failure(tmp_path, monkeypatch, corruption):
    args = artifact_fixture(tmp_path, monkeypatch)
    if corruption == "prediction":
        (args.run/"fits/uniform-6203/predictions.npz").write_bytes(b"tampered")
    elif corruption == "receipt":
        (args.report/"receipt.json").write_text("{}")
    else:
        h = a.load_helper(); done = h.read(args.run/"completed.json"); done["completed_fits"].pop()
        (args.run/"completed.json").write_text(json.dumps(done)); args.completed_sha256 = h.sha(args.run/"completed.json")
    monkeypatch.setattr(a.np, "load", lambda *args, **kwargs: pytest.fail("No arrays before complete authentication"))
    with pytest.raises(ValueError): a.execute(args)
    assert a.load_helper().read(args.out/"failed.json")["status"] == "failed"
    assert not (args.out/"receipt.json").exists()


@pytest.mark.parametrize("field", ["technical_validity_passed", "execution_members", "source_sha256", "model_calls", "wall_seconds"])
def test_resealed_report_cannot_inherit_false_or_missing_technical_validity(tmp_path, monkeypatch, field):
    args = artifact_fixture(tmp_path, monkeypatch); h = a.load_helper()
    receipt = h.read(args.report/"receipt.json")
    receipt[field] = ({"technical_validity_passed": False, "execution_members": {}, "source_sha256": {},
                       "model_calls": 1, "wall_seconds": 61})[field]
    (args.report/"receipt.json").write_text(json.dumps(receipt))
    args.report_receipt_sha256 = h.sha(args.report/"receipt.json")
    monkeypatch.setattr(a.np, "load", lambda *args, **kwargs: pytest.fail("No arrays before technical admission"))
    with pytest.raises(ValueError, match="technical/execution"): a.execute(args)
    assert h.read(args.out/"failed.json")["status"] == "failed"


def test_resealed_summary_must_also_declare_technical_validity(tmp_path, monkeypatch):
    args = artifact_fixture(tmp_path, monkeypatch); h = a.load_helper()
    summary = h.read(args.report/"summary.json"); summary["technical_validity_passed"] = False
    (args.report/"summary.json").write_text(json.dumps(summary))
    receipt = h.read(args.report/"receipt.json"); receipt["files"]["summary.json"] = h.item(args.report/"summary.json")
    (args.report/"receipt.json").write_text(json.dumps(receipt)); args.report_receipt_sha256 = h.sha(args.report/"receipt.json")
    monkeypatch.setattr(a.np, "load", lambda *args, **kwargs: pytest.fail("No arrays before summary technical validity"))
    with pytest.raises(ValueError, match="technical summary"): a.execute(args)
