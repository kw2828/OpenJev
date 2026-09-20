"""Artificial saved chains only; no corpus, tokenizer, model or real score reads."""
from __future__ import annotations

import copy
import importlib.util
from array import array
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


r = module(REPO/"scripts/report_dialogue_qwen_lexical_ablation.py", "ablation_report_test")
old_tests = module(REPO/"tests/test_report_dialogue_qwen_observation.py", "old_saved_fixture")
transform = module(REPO/r.TRANSFORM, "synthetic_transform")


def save(h, path, value):
    path.write_bytes(h.encoded(value))


def save_lines(h, path, rows):
    path.write_bytes(b"".join(h.encoded(v) for v in rows))


def phases(h, fixture, prepared, pilot, run, plan, requests):
    """Reseal two artificial phases without invoking the inference runner."""
    pin = h.sha(prepared/"plan.json")
    timings = copy.deepcopy(fixture.timings)
    for timing, request in zip(timings, requests, strict=True):
        timing["work"] = h.work_for(request)
    for path, phase in ((pilot, "pilot"), (run, "run")):
        path.mkdir(exist_ok=True)
        request = {"prepared": str(prepared), "plan_sha256": pin}
        if phase == "run":
            request.update(pilot=str(pilot), pilot_sha256=h.sha(pilot/"completed.json"))
        save(h, path/"started.json", {"version": h.RUN_VERSION, "phase": phase, "request": request,
             "limits": h.RUN_LIMITS[phase], "labels_accessed": False, "no_retry": True})
        (path/"plan.json").write_bytes((prepared/"plan.json").read_bytes())
        save_lines(h, path/"timings.jsonl", timings)
        if phase == "run":
            save_lines(h, path/"scores.jsonl", fixture.scores)
        done = {"status": "completed", "version": h.RUN_VERSION, "phase": phase, "plan_sha256": pin,
            "source_sha256": plan["source_sha256"], "runtime": plan["runtime"], "model": plan["model"],
            "limits": h.RUN_LIMITS[phase], "labels_accessed": False, "quality_outputs_saved": phase == "run",
            "generated_tokens": 0, "optimizer_updates": 0, "no_retry": True, "wall_seconds": 1.,
            "model_load_seconds": .1, "process_lifetime_peak_rss_bytes": 100, "request_seconds": .12, "model_calls": 12,
            "progress": {"requests_completed": 12, "questions_completed": 24, "forward_calls_attempted": 12,
                         "forward_calls_returned": 12, "active_request_id": None},
            "work_totals": {k: sum(t["work"][k] for t in timings) for k in timings[0]["work"]}}
        if phase == "pilot":
            arms = {}
            for arm in h.ARMS:
                rate = max(t["seconds"]/t["work"]["input_token_slots"] for t in timings if t["arm"] == arm)
                scaled = rate*plan["input_token_slots"][arm]
                arms[arm] = {"max_seconds_per_charged_slot": rate, "max_seconds_per_request": .01,
                    "token_scaled_seconds": scaled, "request_scaled_seconds": .06, "variable_seconds": scaled}
            done["projection"] = {"arms": arms, "factor": 2, "fixed_seconds": 60, "pilot_whole_wall_seconds": 1.,
                "admission_limit_seconds": 7200, "projected_full_seconds": 62, "admitted": True,
                "request_scaled_projection_seconds_descriptive_only": 62}
        else:
            done["pilot_completed_sha256"] = h.sha(pilot/"completed.json")
        fixture.seal(path, done)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    f = old_tests.tree.__wrapped__(tmp_path, monkeypatch)
    h, root = old_tests.r, old_tests.r.ROOT
    # Copy only source bytes to an isolated fake checkout. No real artifacts are read.
    for name in r.ADDED_SOURCES | {r.HELPER, "scripts/report_dialogue_qwen_lexical_ablation.py",
                                    "tests/test_report_dialogue_qwen_lexical_ablation.py"}:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((REPO/name).read_bytes())
    f.requests = [copy.deepcopy(q) for q in f.requests]
    for request in f.requests:
        for question in request["request"]["questions"]:
            question["question"] = transform.TASK+"\nSynthetic schema\n"+question["question"]+transform.LEGEND
    prepared, run, pilot = f.args.prepared, f.args.run, tmp_path/"pilot"
    save_lines(h, prepared/"requests.jsonl", f.requests)
    f.plan["files"]["requests.jsonl"] = h.item(prepared/"requests.jsonl")
    f.plan["strata"] = h.SUPPORT
    f.plan["prompt_tokens"] = {a: {"min": 2, "max": 3, "sum": 30} for a in h.ARMS}
    save(h, prepared/"plan.json", f.plan)
    prep = h.read(prepared/"completed.json")
    prep["plan_sha256"] = h.sha(prepared/"plan.json")
    f.seal(prepared, prep)
    phases(h, f, prepared, pilot, run, f.plan, f.requests)
    f.args.plan_sha256, f.args.run_sha256 = h.sha(prepared/"plan.json"), h.sha(run/"completed.json")
    h.execute(f.args)
    for key, value in {"ROOT": root, "BASE_PLAN": f.args.plan_sha256, "BASE_PREP": h.sha(prepared/"completed.json"),
            "BASE_RUN": f.args.run_sha256, "BASE_SUMMARY": h.sha(f.args.out/"summary.json"),
            "BASE_REPORT_RECEIPT": h.sha(f.args.out/"receipt.json")}.items():
        monkeypatch.setattr(r, key, value)
    real_load = r.load_old
    def fake_load(extended=False):
        child = real_load(extended)
        child.SUPPORT = h.SUPPORT
        child.runtime = h.runtime
        child.REFERENCE_SHA = h.REFERENCE_SHA
        return child
    monkeypatch.setattr(r, "load_old", fake_load)
    newprep, newpilot, newrun = (tmp_path/n for n in ("no-flags-prepared", "no-flags-pilot", "no-flags-run"))
    newprep.mkdir()
    requests = [transform.transform_request(q) for q in f.requests]
    for q in requests:
        q["tokens"] = [[1], [1, 2]]
    save_lines(h, newprep/"requests.jsonl", requests)
    (newprep/"labels.jsonl").write_bytes((prepared/"labels.jsonl").read_bytes())
    plan = copy.deepcopy(f.plan)
    plan.update(experiment_id=r.EXPERIMENT, protocol_sha256=h.sha(root/r.PROTOCOL),
        parent={"prepared_path": str(prepared.resolve()), "plan_sha256": r.BASE_PLAN, "completed_sha256": r.BASE_PREP,
                "files": f.plan["files"], "source_sha256": f.plan["source_sha256"]}, transformation="fixed subtraction",
        source_sha256={**f.plan["source_sha256"], **{n: h.sha(root/n) for n in r.ADDED_SOURCES}},
        files={n: h.item(newprep/n) for n in ("requests.jsonl", "labels.jsonl")},
        input_token_slots=dict.fromkeys(h.ARMS, 24), prompt_tokens={a: {"min": 1, "max": 2, "sum": 18} for a in h.ARMS})
    save(h, newprep/"plan.json", plan)
    save(h, newprep/"started.json", {"synthetic": True})
    prep.update(experiment_id=r.EXPERIMENT, plan_sha256=h.sha(newprep/"plan.json"), parent_plan_sha256=r.BASE_PLAN,
                parent_completed_sha256=r.BASE_PREP, labels_decoded=False, baseline_scores_accessed=False,
                checkpoint_deserializations=0)
    f.seal(newprep, prep)
    phases(h, f, newprep, newpilot, newrun, plan, requests)
    args = SimpleNamespace(baseline_prepared=prepared, baseline_run=run, baseline_report=f.args.out,
        prepared=newprep, plan_sha256=h.sha(newprep/"plan.json"), run=newrun, run_sha256=h.sha(newrun/"completed.json"),
        protocol_sha256=plan["protocol_sha256"], out=tmp_path/"ablation-report")
    def reseal():
        save(h, newprep/"plan.json", plan)
        p = h.read(newprep/"completed.json")
        p["plan_sha256"] = h.sha(newprep/"plan.json")
        f.seal(newprep, p)
        phases(h, f, newprep, newpilot, newrun, plan, requests)
        args.plan_sha256, args.run_sha256 = h.sha(newprep/"plan.json"), h.sha(newrun/"completed.json")
    return SimpleNamespace(args=args, h=h, f=f, plan=plan, requests=requests, reseal=reseal)


def test_private_helper_union_never_mutates_original():
    original, extended = r.load_old(), r.load_old(True)
    assert len(original.SOURCES) == 9 and len(extended.SOURCES) == 14
    assert extended.SOURCES == original.SOURCES | r.ADDED_SOURCES
    assert len(r.load_old().SOURCES) == 9


def test_complete_paired_chain_and_fixed_negative_continuation(tree):
    result = r.execute(tree.args)
    assert result["technical_validity_passed"] and result["support"] == tree.h.SUPPORT
    assert result["continuation"]["checks_passed"] == 12
    assert not result["continuation"]["passed"]
    assert result["original"] == result["no_flags"]
    assert result["costs"]["token_slot_reduction"] == dict.fromkeys(tree.h.ARMS, 12)
    for arm in tree.h.ARMS:
        for value in result["paired"][arm]["cells"].values():
            assert value["wrong_to_correct"] == value["correct_to_wrong"] == 0
    assert "nll" not in result["original_input_descriptive_controls"]["literal"]["cells"]["all"]
    receipt = tree.h.read(tree.args.out/"receipt.json")
    assert receipt["model_calls"] == receipt["tokenizer_calls"] == 0
    assert set(receipt["files"]) == {"started.json", "summary.json", "report.md"}
    with pytest.raises(FileExistsError):
        r.execute(tree.args)


@pytest.mark.parametrize("damage", ["study", "inputs", "labels", "cohort", "sources", "scores"])
def test_authentication_failure_precedes_all_payload_decoding(tree, monkeypatch, damage):
    if damage == "study":
        tree.plan["experiment_id"] = "wrong-study"
    elif damage == "inputs":
        tree.plan["inputs"]["extra"] = {"sha256": "0"*64, "bytes": 0}
    elif damage == "labels":
        path = tree.args.prepared/"labels.jsonl"
        path.write_bytes(path.read_bytes()+b"\n")
        tree.plan["files"]["labels.jsonl"] = tree.h.item(path)
    elif damage == "cohort":
        tree.plan["row_count"] -= 1
    elif damage == "sources":
        tree.plan["source_sha256"]["extra.py"] = "0"*64
    tree.reseal()
    if damage == "scores":
        (tree.args.run/"scores.jsonl").write_text("corrupt")
    monkeypatch.setattr(r, "lines", lambda *_: pytest.fail("Decoded task payload before authentication"))
    with pytest.raises(ValueError):
        r.execute(tree.args)
    assert tree.h.read(tree.args.out/"failed.json")["status"] == "failed"
    assert not (tree.args.out/"receipt.json").exists()


@pytest.mark.parametrize("damage", ["candidate", "context", "cohort"])
def test_exact_public_pair_rejects_unapproved_change(tree, damage):
    changed = copy.deepcopy(tree.requests)
    if damage == "candidate":
        changed[0]["canonical_id_maps"][0]["c00"] = "value:unexpected"
    elif damage == "context":
        changed[0]["request"]["context"] += " unauthorized"
    else:
        changed.pop()
    save_lines(tree.h, tree.args.prepared/"requests.jsonl", changed)
    with pytest.raises(ValueError):
        r.paired_requests(tree.args.baseline_prepared/"requests.jsonl", tree.args.prepared/"requests.jsonl",
                          transform, tree.h, lambda: None)


def test_compaction_is_lossless_and_compare_precedes_context_drop(tree):
    original, new = r.paired_requests(tree.args.baseline_prepared/"requests.jsonl", tree.args.prepared/"requests.jsonl",
                                     transform, tree.h, lambda: None)
    assert isinstance(new[0]["tokens"][0], array) and list(new[0]["tokens"][1]) == [1, 2]
    assert original[0]["request"]["context"] == new[0]["request"]["context"] == ""
    tree.h.request_layout(original, tree.f.labels, tree.f.plan)
    r.validate_new_requests(new, tree.plan, tree.h)


def test_inclusive_exact_count_boundaries_and_score_equality():
    h = r.load_old()
    original = dict.fromkeys(h.ARMS, old_tests.decision_fit(50, 10))
    new = dict.fromkeys(h.ARMS, old_tests.decision_fit(49, 8))
    result = r.continuation(original, new, h)
    assert result["passed"] and result["checks_passed"] == 16
    new["current"] = old_tests.decision_fit(48, 8)
    result = r.continuation(original, new, h)
    assert result["checks_passed"] == 14 and not result["arms"]["current"]
    new["current"] = old_tests.decision_fit(49, 9)
    assert r.continuation(original, new, h)["checks_passed"] == 14
    new["current"] = old_tests.decision_fit(49, 8, np.nextafter(1., np.inf))
    assert r.continuation(original, new, h)["checks_passed"] == 12


def test_pairs_probability_arithmetic_and_type_axes(tree):
    h, f = tree.h, tree.f
    old = (f.logs, f.choices)
    new = ({a: f.logs["history4"] for a in h.ARMS}, {a: f.choices["history4"] for a in h.ARMS})
    pairs = r.paired(f.rows, old, new, h)
    assert (pairs["current"]["cells"]["all"]["wrong_to_correct"],
            pairs["current"]["cells"]["all"]["correct_to_wrong"]) == (2, 0)
    assert pairs["current"]["cells"]["all"]["metric_differences"]["nll"]["row"] == pytest.approx(-2/3)
    assert set(pairs["current"]["candidate_types"]) == {"previous", "target", "original_selected", "no_flags_selected"}
    assert h.candidate_type("value:True") == 2


def test_missing_helper_preserves_failed_attempt(tmp_path, monkeypatch):
    args = SimpleNamespace(**{n: tmp_path/n for n in ("prepared", "run", "baseline_prepared", "baseline_run", "baseline_report")},
                           plan_sha256="a"*64, run_sha256="b"*64, protocol_sha256="c"*64, out=tmp_path/"out")
    monkeypatch.setattr(r, "ROOT", tmp_path/"absent")
    with pytest.raises(FileNotFoundError):
        r.execute(args)
    assert (args.out/"failed.json").exists()
