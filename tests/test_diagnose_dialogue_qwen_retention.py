"""Artificial saved artifacts only, with no real dialogue/model/tokenizer reads."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d = load("retention_diagnostic_test", ROOT/"scripts/diagnose_dialogue_qwen_retention.py")
fixture = load("retention_artificial_report_fixture", ROOT/"tests/test_report_dialogue_qwen_observation.py")
r = fixture.r


def test_literal_occurrences_are_unicode_complete_utterances_not_semantic_types():
    exchanges = [{"USER": "New", "SYSTEM": "ＡＰＰＬＥ cafe"},
                 {"USER": "York; Café\t Noir is mentioned.", "SYSTEM": "pineapple and apple"}]
    assert d.literal_flags("value:café noir", exchanges) == {
        "latest_user": True, "latest_system": False, "older_history4_user": False, "older_history4_system": False}
    assert d.literal_flags("value:apple", exchanges) == {
        "latest_user": False, "latest_system": True, "older_history4_user": False, "older_history4_system": True}
    assert not any(d.literal_flags("value:New York", exchanges).values())
    assert not any(d.literal_flags("value:apple", [{"USER": "pineapple", "SYSTEM": "applejack"}]).values())
    assert d.literal_flags("value:C++", [{"USER": "Use C++!", "SYSTEM": ""}])["latest_user"]
    for cid in (d.NONE, d.DC, "value:True", "value:FALSE"):
        assert set(d.literal_flags(cid, exchanges).values()) == {None}
    assert not d.VISIBILITY["current"]["older_history4_user"]
    assert d.VISIBILITY["history4"]["older_history4_user"]


def test_fixed_probability_edges_and_empty_statistics():
    assert [d.prior_bin(p) for p in (0, .009, .01, .1, .5, .9, .99, 1)] == [
        d.BINS[i] for i in (0, 0, 1, 2, 3, 4, 5, 5)]
    for p in (-.01, 1.01, float("nan")):
        with pytest.raises(ValueError, match="range"):
            d.prior_bin(p)
    value = d.describe([])
    assert value["rows"] == value["correct"] == value["error"] == 0
    assert value["previous_probability_mean"] is None and value["previous_competition_rank_mean"] is None
    assert sum(value["previous_probability_bins"].values()) == 0


def artificial_selection():
    rows, public = {a: {} for a in d.ARMS}, {a: {} for a in d.ARMS}
    situations = [("TRUE", False), ("OTHER", False), ("DONTCARE", False), ("NOT_MENTIONED", False),
                  ("TRUE", True), ("OTHER", True)]
    for group, (kind, changed) in enumerate(situations):
        for number in range(4 if group != 2 else 1):
            index = group*10+number
            for arm in d.ARMS:
                rows[arm][index] = {"row_index": index, "dialogue_id": f"dialogue-{group}-{number//2}",
                    "correct": group == 3 and arm == "current", "changed": changed, "target_type": kind,
                    "subtype": "first_assignment" if changed else "unmentioned_retention" if group == 3 else "assigned_retention"}
                public[arm][index] = {"context": "CURRENT ONLY" if arm == "current" else "HISTORY ONLY",
                    "question": "Exact public question with previous value.",
                    "candidates": [{"id": "c00", "description": "Exact candidate description"}],
                    "canonical_id_map": {"c00": "reserved:NOT_MENTIONED"}}
    return rows, public


def test_blinded_selection_exact_hash_order_distinct_dialogues_sparse_support():
    rows, public = artificial_selection()
    packets, ledger, counts = d.selection(rows, public)
    assert len(ledger) == 11 and counts[d.REVIEW_STRATA[2]]["selected_rows"] == 1
    for stratum in d.REVIEW_STRATA:
        eligible = [i for i in rows["current"] if d.review_stratum(rows["current"][i], rows["history4"][i]) == stratum]
        ordered = sorted(eligible, key=lambda i: hashlib.sha256(f"openjev-retention-diagnostic-v1:{i}".encode()).digest())
        expected, dialogues = [], set()
        for i in ordered:
            dialogue = rows["current"][i]["dialogue_id"]
            if dialogue not in dialogues and len(expected) < 2:
                expected.append(i)
                dialogues.add(dialogue)
        assert {x["rows"]["current"]["row_index"] for x in ledger if x["selection_stratum"] == stratum} == set(expected)
    assert [x["case_id"] for x in packets["current"]["cases"]] == sorted(x["case_id"] for x in ledger)
    assert [x["case_id"] for x in packets["current"]["cases"]] == [x["case_id"] for x in packets["history4"]["cases"]]
    for arm in d.ARMS:
        for case in packets[arm]["cases"]:
            assert set(case) == {"case_id", "context", "question", "candidates"}
            assert case["context"] == ("CURRENT ONLY" if arm == "current" else "HISTORY ONLY")
            assert case["candidates"] == [{"id": "c00", "description": "Exact candidate description"}]
        serialized = json.dumps(packets[arm])
        assert "dialogue-" not in serialized and "row_index" not in serialized and "canonical_id_map" not in serialized
        assert "selection_stratum" not in serialized and "correct" not in serialized


@pytest.fixture
def packet(tmp_path, monkeypatch):
    tree = fixture.tree.__wrapped__(tmp_path, monkeypatch)
    # Upgrade only artificial public contexts, then reseal the entire fake chain.
    for request in tree.requests:
        latest = {"USER": "Synthetic current request", "SYSTEM": "Synthetic current proposal"}
        exchanges = [latest] if request["arm"] == "current" else [
            {"USER": "Synthetic earlier request", "SYSTEM": "Synthetic earlier proposal"}, latest]
        request["request"]["context"] = json.dumps({"exchanges_oldest_first": exchanges})
    (tree.args.prepared/"requests.jsonl").write_bytes(b"".join(r.encoded(v) for v in tree.requests))
    tree.plan["files"]["requests.jsonl"] = r.item(tree.args.prepared/"requests.jsonl")
    (tree.args.prepared/"plan.json").write_bytes(r.encoded(tree.plan))
    tree.args.plan_sha256 = r.sha(tree.args.prepared/"plan.json")
    done = r.read(tree.args.prepared/"completed.json")
    done["plan_sha256"] = tree.args.plan_sha256
    tree.seal(tree.args.prepared, done)
    for path in (tmp_path/"pilot", tree.args.run):
        (path/"plan.json").write_bytes((tree.args.prepared/"plan.json").read_bytes())
        started, done = r.read(path/"started.json"), r.read(path/"completed.json")
        started["request"]["plan_sha256"] = done["plan_sha256"] = tree.args.plan_sha256
        if path == tree.args.run:
            started["request"]["pilot_sha256"] = done["pilot_completed_sha256"] = r.sha(tmp_path/"pilot/completed.json")
        (path/"started.json").write_bytes(r.encoded(started))
        tree.seal(path, done)
    tree.args.run_sha256 = r.sha(tree.args.run/"completed.json")
    r.execute(tree.args)  # Synthetic reporter; no model/real row data.
    helper = d.load_helper()
    monkeypatch.setattr(helper, "ROOT", r.ROOT)
    monkeypatch.setattr(helper, "REFERENCE_SHA", r.REFERENCE_SHA)
    monkeypatch.setattr(helper, "SUPPORT", {key: r.SUPPORT[key] for key in helper.STRATA})
    monkeypatch.setattr(d, "load_helper", lambda check=lambda: None: helper)
    args = SimpleNamespace(prepared=tree.args.prepared, plan_sha256=tree.args.plan_sha256,
        run=tree.args.run, run_sha256=tree.args.run_sha256, report=tree.args.out,
        report_receipt_sha256=r.sha(tree.args.out/"receipt.json"),
        report_summary_sha256=r.sha(tree.args.out/"summary.json"),
        protocol_sha256=d.sha(ROOT/d.PROTOCOL), out=tmp_path/"diagnostic")
    monkeypatch.setattr(d, "PINS", {key: getattr(args, key) for key in d.PINS})
    return SimpleNamespace(args=args, tree=tree, helper=helper)


def test_synthetic_authenticated_complete_diagnostic_and_exclusivity(packet):
    summary = d.execute(packet.args)
    out = packet.args.out
    receipt = packet.helper.read(out/"receipt.json")
    assert summary["row_records"] == 24 and len(summary["services"]) == 6
    for arm in d.ARMS:
        assert len(summary["arms"][arm]) == 7
        overall = summary["arms"][arm]["__all__"]
        assert sum(g["statistics"]["rows"] for g in overall["typed_groups"]) == 12
        assert overall["cells"]["all"]["correct"] + overall["cells"]["all"]["error"] == 12
        assert overall["cells"]["clear"]["previous_probability_mean"] is None
    assert summary["arms"]["current"]["__all__"]["cells"]["all"]["correct"] == 6
    assert summary["arms"]["history4"]["__all__"]["cells"]["all"]["correct"] == 8
    assert receipt["model_calls"] == receipt["reviewer_calls"] == receipt["tokenizer_calls"] == 0
    assert all(d.item(out/name) == desc for name, desc in receipt["files"].items())
    assert set(receipt["files"]) == {"started.json", "rows.jsonl", "evaluator-selection.jsonl", "current-review.json",
                                     "history4-review.json", "summary.json"}
    before = d.sha(out/"receipt.json")
    with pytest.raises(FileExistsError):
        d.execute(packet.args)
    assert d.sha(out/"receipt.json") == before


def test_manifest_corruption_is_rejected_before_individual_score_decode(packet, monkeypatch):
    with (packet.args.run/"scores.jsonl").open("a") as stream:
        stream.write(" ")
    def forbidden(*args):
        raise AssertionError("Individual score decoding reached")
    monkeypatch.setattr(packet.helper, "reconstruct", forbidden)
    with pytest.raises(ValueError, match="Payload digest/size"):
        d.execute(packet.args)
    assert packet.helper.read(packet.args.out/"failed.json")["status"] == "failed"
    assert not (packet.args.out/"receipt.json").exists()


def test_prior_competition_rank_counts_strictly_greater_not_tie_order(packet):
    tree = packet.tree
    request = tree.requests[0]
    ids = tree.scores[0]["questions"][0]["candidate_ids"]
    # Prior NONE tied for first, but prompt-label first TRUE wins.
    tree.scores[0]["questions"][0] = fixture.distribution([2., -1., 2., -1.], ids,
                                                                       request["ordered_candidate_ids"][0], 0)
    (packet.args.run/"scores.jsonl").write_bytes(b"".join(r.encoded(s) for s in tree.scores))
    base = packet.helper.reconstruct(packet.args.prepared, packet.args.run, tree.plan, lambda: None)
    rows, _ = d.derive(packet.args.prepared, packet.args.run, packet.helper, base)
    assert rows["current"][0]["previous_competition_rank"] == 1
    assert rows["current"][0]["selected_id"] == "value:True"
    assert rows["current"][0]["literal_flags"] == dict.fromkeys(d.FLAGS, None)


@pytest.mark.parametrize("damage", ["question", "latest_exchange", "label_bin"])
def test_cross_arm_or_evaluator_semantic_mismatch_rejected(packet, damage):
    tree = packet.tree
    base = packet.helper.reconstruct(packet.args.prepared, packet.args.run, tree.plan, lambda: None)
    requests, labels = copy.deepcopy(tree.requests), copy.deepcopy(tree.labels)
    request = next(q for q in requests if q["arm"] == "history4")
    if damage == "question":
        request["request"]["questions"] = copy.deepcopy(request["request"]["questions"])
        request["request"]["questions"][0]["question"] += " changed public question"
    elif damage == "latest_exchange":
        value = json.loads(request["request"]["context"])
        value["exchanges_oldest_first"][-1]["USER"] = "different latest user"
        request["request"]["context"] = json.dumps(value)
    else:
        labels[0]["derived_bin"] = "revision"
    (packet.args.prepared/"requests.jsonl").write_bytes(b"".join(r.encoded(q) for q in requests))
    (packet.args.prepared/"labels.jsonl").write_bytes(b"".join(r.encoded(q) for q in labels))
    with pytest.raises(ValueError, match="Matched public|Latest public|Evaluator transition"):
        d.derive(packet.args.prepared, packet.args.run, packet.helper, base)


def test_sparse_group_reconciliation_rejects_changed_saved_summary(packet):
    base = packet.helper.reconstruct(packet.args.prepared, packet.args.run, packet.tree.plan, lambda: None)
    rows, _ = d.derive(packet.args.prepared, packet.args.run, packet.helper, base)
    summary = packet.helper.read(packet.args.report/"summary.json")
    summary["arms"]["current"]["services"]["s0"]["unmentioned_retention"]["counts"]["error"] += 1
    with pytest.raises(ValueError, match="count reconciliation"):
        d.aggregate(rows, summary)


def test_late_output_cap_preserves_failure_and_demotes_receipt(packet, monkeypatch):
    original = d.write
    def capped(path, value):
        original(path, value)
        if Path(path).name == "receipt.json":
            monkeypatch.setitem(d.LIMITS, "output_bytes", 0)
    monkeypatch.setattr(d, "write", capped)
    with pytest.raises(ValueError, match="output cap"):
        d.execute(packet.args)
    assert (packet.args.out/"failed.json").exists() and (packet.args.out/"late-receipt.json").exists()
    assert not (packet.args.out/"receipt.json").exists()
