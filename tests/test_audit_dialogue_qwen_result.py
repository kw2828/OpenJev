"""Artificial saved artifacts only; no live scores, models or tokenizer loads.

The frozen reporter's existing synthetic fixture supplies the envelope and the
comparison output. Auditor arithmetic is exercised unchanged; only the tiny
fixture's root, support and historical-reference pin replace production values.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


a = load("qwen_independent_audit_test", ROOT/"scripts/audit_dialogue_qwen_result.py")
fixture_source = load("qwen_report_fixture_for_audit", ROOT/"tests/test_report_dialogue_qwen_observation.py")
r = fixture_source.r


def overwrite(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, allow_nan=False)+"\n")


@pytest.fixture
def packet(tmp_path, monkeypatch):
    tree = fixture_source.tree.__wrapped__(tmp_path, monkeypatch)
    r.execute(tree.args)  # Synthetic report only, never a production run.
    monkeypatch.setattr(a, "ROOT", r.ROOT)
    monkeypatch.setattr(a, "REFERENCE_SHA", r.REFERENCE_SHA)
    monkeypatch.setattr(a, "SUPPORT", {key: r.SUPPORT[key] for key in a.STRATA})
    args = SimpleNamespace(prepared=tree.args.prepared, plan_sha256=tree.args.plan_sha256,
                           run=tree.args.run, run_sha256=tree.args.run_sha256,
                           report=tree.args.out, report_receipt_sha256=a.sha(tree.args.out/"receipt.json"),
                           report_summary_sha256=a.sha(tree.args.out/"summary.json"), out=tmp_path/"audit")
    return SimpleNamespace(tree=tree, args=args)


def reseal_report(packet):
    """Update fake authentication pins so semantic corruption reaches arithmetic."""
    args = packet.args
    receipt = a.read(args.report/"receipt.json")
    receipt["files"]["summary.json"] = a.item(args.report/"summary.json")
    overwrite(args.report/"receipt.json", receipt)
    args.report_receipt_sha256 = a.sha(args.report/"receipt.json")
    args.report_summary_sha256 = a.sha(args.report/"summary.json")


def replace_scores(packet, scores):
    args = packet.args
    (args.run/"scores.jsonl").write_bytes(b"".join(r.encoded(v) for v in scores))
    done = a.read(args.run/"completed.json")
    done["files"]["scores.jsonl"] = a.item(args.run/"scores.jsonl")
    overwrite(args.run/"completed.json", done)
    args.run_sha256 = a.sha(args.run/"completed.json")
    for name in ("summary.json", "receipt.json"):
        value = a.read(args.report/name)
        value["execution_completed_sha256"] = args.run_sha256
        overwrite(args.report/name, value)
    reseal_report(packet)


def assert_failed(packet):
    assert a.read(packet.args.out/"failed.json")["status"] == "failed"
    assert not (packet.args.out/"receipt.json").exists()


def test_complete_synthetic_artifact_audit_and_exclusivity(packet):
    a.execute(packet.args)
    summary = a.read(packet.args.out/"summary.json")
    receipt = a.read(packet.args.out/"receipt.json")
    assert summary["agreement"] and summary["primary_rows_per_arm"] == 12
    assert summary["metric_cells"] == 42 and summary["decision_checks"] == 16
    assert summary["scalar_checks"] > 400
    assert summary["fits"]["current"]["cells"]["all"]["counts"] == {"correct": 6, "error": 6}
    assert summary["fits"]["history4"]["cells"]["changed"]["counts"] == {"correct": 2, "error": 4}
    assert summary["fits"]["current"]["cells"]["all"]["metrics"]["nll"]["row"] == pytest.approx(
        2+math.log1p(3*math.exp(-4)), abs=1e-12)
    assert not summary["decisions"]["semantic_strength"]["behavioral"]["passed"]
    assert summary["decisions"]["added_history"]["behavioral"]["passed"]
    assert receipt["source_sha256"] == a.sha(a.__file__)
    assert receipt["model_calls"] == receipt["tokenizer_calls"] == receipt["checkpoint_deserializations"] == 0
    assert all(a.item(packet.args.out/name) == value for name, value in receipt["files"].items())
    pin = a.sha(packet.args.out/"receipt.json")
    with pytest.raises(FileExistsError):
        a.execute(packet.args)
    assert a.sha(packet.args.out/"receipt.json") == pin


@pytest.mark.parametrize("damage", ["metric", "gate"])
def test_resealed_report_arithmetic_or_gate_corruption_rejected(packet, damage):
    summary = a.read(packet.args.report/"summary.json")
    if damage == "metric":
        summary["arms"]["history4"]["cells"]["all"]["metrics"]["nll"]["equal_service"] += .01
    else:
        check = summary["decisions"]["semantic_strength"]["behavioral"]["checks"]["changed_accuracy_row"]
        check["passed"] = not check["passed"]
    overwrite(packet.args.report/"summary.json", summary)
    reseal_report(packet)
    with pytest.raises(ValueError, match="disagreement"):
        a.execute(packet.args)
    assert_failed(packet)
    failure_pin = a.sha(packet.args.out/"failed.json")
    with pytest.raises(FileExistsError):
        a.execute(packet.args)
    assert a.sha(packet.args.out/"failed.json") == failure_pin


def test_resealed_probability_corruption_rejected(packet):
    scores = packet.tree.scores
    scores[0]["questions"][0]["probabilities"][0] += .01
    replace_scores(packet, scores)
    with pytest.raises(ValueError, match="conditional probabilities/logs"):
        a.execute(packet.args)
    assert_failed(packet)


@pytest.mark.parametrize("damage", ["missing", "reordered"])
def test_resealed_request_coverage_corruption_rejected(packet, damage):
    scores = list(packet.tree.scores)
    if damage == "missing":
        scores.pop()
    else:
        scores[0], scores[1] = scores[1], scores[0]
    replace_scores(packet, scores)
    with pytest.raises(ValueError, match="Missing score request|Score request identity/order"):
        a.execute(packet.args)
    assert_failed(packet)


@pytest.mark.parametrize("damage", ["manifest_bytes", "lineage"])
def test_authentication_rejects_before_score_decode(packet, monkeypatch, damage):
    if damage == "manifest_bytes":
        with (packet.args.run/"scores.jsonl").open("a") as stream:
            stream.write("\n")
    else:
        receipt = a.read(packet.args.report/"receipt.json")
        receipt["execution_completed_sha256"] = "0"*64
        overwrite(packet.args.report/"receipt.json", receipt)
        packet.args.report_receipt_sha256 = a.sha(packet.args.report/"receipt.json")
    original = a.decode

    def guarded(value):
        text = value.decode() if isinstance(value, bytes) else value
        if '"candidate_logits"' in text:
            raise AssertionError("Quality decoded before authentication")
        return original(value)

    monkeypatch.setattr(a, "decode", guarded)
    with pytest.raises(ValueError, match="Payload digest/size|Completed run/report lineage"):
        a.execute(packet.args)
    assert_failed(packet)


def test_underflow_nll_and_description_order_tie_use_raw_logs(packet):
    request = packet.tree.requests[0]
    ids = packet.tree.scores[0]["questions"][0]["candidate_ids"]
    saved = fixture_source.distribution([-2000., 0., 0., -3000.], ids,
                                         request["ordered_candidate_ids"][0], index=0)
    assert saved["selected_id"] == "value:True" and saved["probabilities"][0] == 0
    packet.tree.scores[0]["questions"][0] = saved
    replace_scores(packet, packet.tree.scores)
    records = a.reconstruct(packet.args.prepared, packet.args.run, packet.tree.plan, lambda: None)
    assert records["current"][0]["nll"] == pytest.approx(2000+math.log(2), abs=1e-12)
    saved["selected_id"] = r.DC  # Same maximum, but the wrong prompt-label tie winner.
    replace_scores(packet, packet.tree.scores)
    with pytest.raises(ValueError, match="Frozen prompt-label maximum/tie"):
        a.reconstruct(packet.args.prepared, packet.args.run, packet.tree.plan, lambda: None)


def test_row_weighting_differs_from_equal_service_and_empty_cells(monkeypatch):
    records = {}
    for service in range(6):
        for _ in range(3 if service == 0 else 1):
            correct = int(service == 0)
            records[len(records)] = {"service": str(service), "changed": service == 0,
                "accuracy": correct, "error": 1-correct, "nll": float(1-correct), "brier": float(1-correct)}
    monkeypatch.setattr(a, "SUPPORT", {"all": 8, "changed": 3, "retained": 5})
    fit = a.calculate({"current": records})["current"]
    assert fit["cells"]["all"]["metrics"]["accuracy"] == {"row": 3/8, "equal_service": 1/6}
    assert fit["services"]["1"]["changed"]["rows"] == 0
    assert fit["services"]["1"]["changed"]["metrics"]["nll"] == {"row": None, "equal_service": None}


def test_exact_behavioral_boundary_and_mean_first_proper_score():
    upper = float(np.nextafter(1., np.inf))
    controls = [fixture_source.decision_fit(50, 10, score=v) for v in (1., upper, upper)]
    candidate = fixture_source.decision_fit(52, 10, score=upper)
    checks = a.rules(candidate, controls)
    assert checks["behavioral"]["passed"] and checks["proper_score_nonregression"]["passed"]
    assert checks["behavioral"]["checks"]["changed_accuracy_row"]["difference"] == .02
    assert all(v["difference"] == 0 for v in checks["proper_score_nonregression"]["checks"].values())
    candidate = fixture_source.decision_fit(52, 11, score=upper)
    assert not a.rules(candidate, controls)["behavioral"]["passed"]


def test_late_output_limit_demotes_success_and_preserves_failure(packet, monkeypatch):
    original = a.write

    def write(path, value):
        original(path, value)
        if Path(path).name == "receipt.json":
            monkeypatch.setitem(a.LIMITS, "output_bytes", 0)

    monkeypatch.setattr(a, "write", write)
    with pytest.raises(ValueError, match="Audit output cap"):
        a.execute(packet.args)
    assert_failed(packet)
    assert a.read(packet.args.out/"late-receipt.json")["agreement"]
