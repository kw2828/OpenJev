"""Tiny artificial artifacts only; actual auditor never imports the report code."""
from __future__ import annotations

import copy
import importlib.util
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


a = load("independent_lexical_result_test", ROOT/"scripts/audit_dialogue_qwen_lexical_result.py")
fixtures = load("lexical_result_artificial_fixture", ROOT/"tests/test_report_dialogue_qwen_lexical_ablation.py")
r = fixtures.r


@pytest.fixture
def packet(tmp_path, monkeypatch):
    tree = fixtures.tree.__wrapped__(tmp_path, monkeypatch)
    r.execute(tree.args)  # Only artificial report fixtures, never real scores.
    h = a.load_helper()
    monkeypatch.setattr(h, "ROOT", tree.h.ROOT)
    monkeypatch.setattr(h, "REFERENCE_SHA", tree.h.REFERENCE_SHA)
    monkeypatch.setattr(h, "SUPPORT", {name: tree.h.SUPPORT[name] for name in h.STRATA})
    for name in a.SOURCES:
        dest = tree.h.ROOT/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/name).read_bytes())
    for key in ("BASE_PLAN", "BASE_PREP", "BASE_RUN", "BASE_SUMMARY", "BASE_REPORT_RECEIPT"):
        monkeypatch.setattr(a, key, getattr(r, key))
    monkeypatch.setattr(a, "NEW_PLAN", tree.args.plan_sha256)
    monkeypatch.setattr(a, "ROOT", tree.h.ROOT)
    monkeypatch.setattr(a, "load_helper", lambda check=lambda: None: h)
    args = SimpleNamespace(**{name: getattr(tree.args, name) for name in
        ("baseline_prepared", "baseline_run", "baseline_report", "prepared", "plan_sha256", "run", "run_sha256")},
        report=tree.args.out, report_receipt_sha256=h.sha(tree.args.out/"receipt.json"),
        report_summary_sha256=h.sha(tree.args.out/"summary.json"), out=tmp_path/"independent-audit")
    return SimpleNamespace(args=args, tree=tree, h=h)


def reseal_report(packet):
    p, h = packet.args, packet.h
    receipt = h.read(p.report/"receipt.json")
    receipt["files"]["summary.json"] = h.item(p.report/"summary.json")
    fixtures.save(packet.tree.h, p.report/"receipt.json", receipt)
    p.report_summary_sha256, p.report_receipt_sha256 = h.sha(p.report/"summary.json"), h.sha(p.report/"receipt.json")


def reseal_run(packet, scores):
    p, h = packet.args, packet.h
    fixtures.save_lines(packet.tree.h, p.run/"scores.jsonl", scores)
    done = h.read(p.run/"completed.json")
    done["files"]["scores.jsonl"] = h.item(p.run/"scores.jsonl")
    fixtures.save(packet.tree.h, p.run/"completed.json", done)
    p.run_sha256 = h.sha(p.run/"completed.json")
    for name in ("receipt.json", "summary.json"):
        obj = h.read(p.report/name)
        obj["execution_completed_sha256"] = p.run_sha256
        fixtures.save(packet.tree.h, p.report/name, obj)
    reseal_report(packet)


def test_complete_artificial_audit_84_cells_16_rules_and_exclusivity(packet):
    summary = a.execute(packet.args)
    assert summary["agreement"] and summary["metric_cells"] == 84 and summary["decision_checks"] == 16
    assert summary["rows_per_arm"] == 12 and summary["paired_requests"] == 12
    assert summary["original"] == summary["no_flags"]
    assert summary["continuation"]["checks_passed"] == 12 and not summary["continuation"]["passed"]
    assert summary["scalar_checks"] > 800
    assert summary["no_flags"]["current"]["cells"]["all"]["metrics"]["nll"]["row"] == pytest.approx(
        2+math.log1p(3*math.exp(-4)), abs=1e-12)
    receipt = packet.h.read(packet.args.out/"receipt.json")
    assert receipt["producer_summary_sha256"] == packet.args.report_summary_sha256
    assert receipt["model_calls"] == receipt["tokenizer_calls"] == receipt["checkpoint_deserializations"] == 0
    assert all(packet.h.item(packet.args.out/name) == descriptor for name, descriptor in receipt["files"].items())
    pin = packet.h.sha(packet.args.out/"receipt.json")
    with pytest.raises(FileExistsError):
        a.execute(packet.args)
    assert packet.h.sha(packet.args.out/"receipt.json") == pin


@pytest.mark.parametrize("damage", ["metric", "decision"])
def test_resealed_new_report_arithmetic_corruption(packet, damage):
    report = packet.h.read(packet.args.report/"summary.json")
    if damage == "metric":
        report["no_flags"]["history4"]["services"]["s2"]["retained"]["metrics"]["brier"]["row"] += .01
    else:
        check = report["continuation"]["checks"]["current/retained/error/row"]
        check["passed"] = not check["passed"]
    fixtures.save(packet.tree.h, packet.args.report/"summary.json", report)
    reseal_report(packet)
    with pytest.raises(ValueError, match="disagreement"):
        a.execute(packet.args)
    assert packet.h.read(packet.args.out/"failed.json")["status"] == "failed"
    assert not (packet.args.out/"receipt.json").exists()


@pytest.mark.parametrize("damage", ["probability", "tie", "missing", "reordered"])
def test_resealed_new_score_corruption_rejected(packet, damage):
    scores = copy.deepcopy(packet.tree.f.scores)
    if damage == "probability":
        scores[0]["questions"][0]["probabilities"][0] += .02
    elif damage == "tie":
        question = scores[0]["questions"][0]
        ordered = packet.tree.requests[0]["ordered_candidate_ids"][0]
        question = fixtures.old_tests.distribution([2, 2, 2, 2], question["candidate_ids"], ordered, index=0)
        question["selected_id"] = ordered[-1]
        scores[0]["questions"][0] = question
    elif damage == "missing":
        scores.pop()
    else:
        scores[0], scores[1] = scores[1], scores[0]
    reseal_run(packet, scores)
    with pytest.raises(ValueError, match="probabilities/logs|maximum/tie|Missing score|identity/order"):
        a.execute(packet.args)
    assert (packet.args.out/"failed.json").exists()


@pytest.mark.parametrize("damage", ["unsealed_scores", "missing_completion", "failed_completion", "wrong_experiment"])
def test_authentication_stops_before_either_score_decode(packet, monkeypatch, damage):
    if damage == "unsealed_scores":
        (packet.args.run/"scores.jsonl").write_text("not JSON")
    elif damage == "missing_completion":
        (packet.args.run/"completed.json").unlink()
    elif damage == "failed_completion":
        done = packet.h.read(packet.args.run/"completed.json")
        done["status"] = "failed"
        fixtures.save(packet.tree.h, packet.args.run/"completed.json", done)
        packet.args.run_sha256 = packet.h.sha(packet.args.run/"completed.json")
    else:
        receipt = packet.h.read(packet.args.report/"receipt.json")
        receipt["experiment_id"] = "wrong"
        fixtures.save(packet.tree.h, packet.args.report/"receipt.json", receipt)
        packet.args.report_receipt_sha256 = packet.h.sha(packet.args.report/"receipt.json")
    monkeypatch.setattr(packet.h, "reconstruct", lambda *_: pytest.fail("Score reconstruction before authentication"))
    with pytest.raises((ValueError, FileNotFoundError)):
        a.execute(packet.args)
    failure = packet.h.read(packet.args.out/"failed.json")
    assert failure["status"] == "failed" and failure["model_calls"] == 0


def test_exact_inclusive_boundaries_and_adjacent_float_score_harm():
    control = {arm: fixtures.old_tests.decision_fit(50, 10) for arm in a.ARMS}
    candidate = {arm: fixtures.old_tests.decision_fit(49, 8) for arm in a.ARMS}
    result = a.continuation(control, candidate)
    assert result["passed"] and result["checks_passed"] == 16
    candidate["current"] = fixtures.old_tests.decision_fit(48, 8)
    assert a.continuation(control, candidate)["checks_passed"] == 14
    candidate["current"] = fixtures.old_tests.decision_fit(49, 9)
    assert a.continuation(control, candidate)["checks_passed"] == 14
    candidate["current"] = fixtures.old_tests.decision_fit(49, 8, float(np.nextafter(1., np.inf)))
    result = a.continuation(control, candidate)
    assert result["checks_passed"] == 12
    assert result["checks"]["current/all/nll/row"]["difference"] > 0


def test_equal_service_counts_not_pooled_and_no_support_is_undefined():
    old, new = fixtures.old_tests.decision_fit(50, 10), fixtures.old_tests.decision_fit(49, 8)
    old["services"] = {"large": fixtures.old_tests.decision_fit(50, 10)["cells"],
                       "small": fixtures.old_tests.decision_fit(50, 10)["cells"]}
    new["services"] = {"large": fixtures.old_tests.decision_fit(49, 8)["cells"],
                       "small": fixtures.old_tests.decision_fit(50, 10)["cells"]}
    result = a.continuation(dict.fromkeys(a.ARMS, old), dict.fromkeys(a.ARMS, new))
    assert result["checks"]["current/retained/error/row"]["passed"]
    assert not result["checks"]["current/retained/error/equal_service"]["passed"]
    assert result["checks"]["current/retained/error/equal_service"]["difference"] == -.01
    empty = {"rows": 0, "counts": {"correct": 0, "error": 0}}
    for fit in (old, new):
        fit["cells"]["changed"] = empty
        for service in fit["services"].values():
            service["changed"] = empty
    result = a.continuation(dict.fromkeys(a.ARMS, old), dict.fromkeys(a.ARMS, new))
    assert result["checks"]["current/changed/accuracy/row"]["difference"] is None
    assert not result["checks"]["current/changed/accuracy/equal_service"]["passed"]


def test_late_cap_failure_demotes_success_and_does_not_replace_attempt(packet, monkeypatch):
    original = a.write
    def write(path, value):
        original(path, value)
        if Path(path).name == "receipt.json":
            monkeypatch.setitem(a.LIMITS, "output_bytes", 0)
    monkeypatch.setattr(a, "write", write)
    with pytest.raises(ValueError, match="output cap"):
        a.execute(packet.args)
    assert (packet.args.out/"late-receipt.json").exists() and (packet.args.out/"failed.json").exists()
    assert not (packet.args.out/"receipt.json").exists()
    with pytest.raises(FileExistsError):
        a.execute(packet.args)
