"""Saved synthetic distributions only; no corpus, model, tokenizer or device calls."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1]/"scripts/report_dialogue_qwen_observation.py"
SPEC = importlib.util.spec_from_file_location("qwen_saved_report", PATH)
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def distribution(z, ids, ordered, index=0):
    z = np.asarray(z, np.float64)
    shift = float(np.logaddexp.reduce(z-z.max()))
    full = math.log(float(np.exp(z-z.max()).sum())+(151936-len(z))*math.exp(-10))
    logs = z-z.max()-shift
    return {"row_index": index, "question_id": f"r{index}", "candidate_ids": list(ids),
            "label_ids": [32+ordered.index(cid) for cid in ids],
            "selected_id": next(cid for cid in ordered if z[ids.index(cid)] == z.max()),
            "tie_break": "first frozen prompt label", "vocabulary_size": 151936,
            "candidate_logits": z.tolist(), "log_probs": logs.tolist(), "probabilities": np.exp(logs).tolist(),
            "candidate_max_logit": float(z.max()), "candidate_max_tie_count": int((z == z.max()).sum()),
            "candidate_shifted_log_partition": shift, "full_vocabulary_max_logit": float(z.max()),
            "full_vocabulary_shifted_log_partition": full, "log_candidate_token_mass": shift-full,
            "candidate_token_mass": math.exp(shift-full)}


def test_boolean_suffix_casefold_preserves_reserved_and_literal_none():
    assert [r.candidate_type(v) for v in (r.NONE, r.DC, "value:True", "value:FALSE", "value:None",
                                         "reserved:true", "value:true")] == [0, 1, 2, 3, 4, 4, 2]


def test_extreme_logs_and_prompt_order_tie():
    ids = [r.NONE, r.DC, "value:True", "value:False"]
    ordered = [ids[2], ids[3], ids[0], ids[1]]
    saved = distribution([-2000, 1000, 1000, -3000], ids, ordered)
    logs, choice = r.reconstruct_question(saved, {"candidate_ids": ids, "ordered": ordered}, 0, "r0")
    assert choice == 2 and -logs[0] == pytest.approx(3000+math.log(2))
    assert saved["probabilities"][0] == 0
    saved["selected_id"] = r.DC
    with pytest.raises(ValueError, match="tie selection"):
        r.reconstruct_question(saved, {"candidate_ids": ids, "ordered": ordered}, 0, "r0")


@pytest.mark.parametrize("field", ["log_probs", "probabilities", "label_ids", "candidate_ids",
                                  "candidate_max_tie_count", "candidate_shifted_log_partition",
                                  "full_vocabulary_shifted_log_partition", "candidate_token_mass"])
def test_inconsistent_saved_witness_rejected(field):
    ids = [r.NONE, r.DC, "value:True", "value:False"]
    saved = distribution([0, 1, 2, 3], ids, ids)
    if isinstance(saved[field], list):
        saved[field][0] = "wrong" if field == "candidate_ids" else saved[field][0]+.1
    else:
        saved[field] += .1
    with pytest.raises(ValueError):
        r.reconstruct_question(saved, {"candidate_ids": ids, "ordered": ids}, 0, "r0")


def test_actual_producer_readout_compatibility_on_artificial_logits():
    spec = importlib.util.spec_from_file_location("qwen_readout_only", PATH.with_name("run_dialogue_qwen_observation.py"))
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    ids = [r.NONE, r.DC, "value:True", "value:False"]
    ordered = [ids[2], ids[3], ids[0], ids[1]]
    row = {"row_indices": [7], "request": {"questions": [{"id": "r7", "candidates":
           [{"id": f"c{i:02}"} for i in range(4)]}]}, "ordered_candidate_ids": [ordered],
           "canonical_id_maps": [dict(zip((f"c{i:02}" for i in range(4)), ids, strict=True))],
           "label_ids": list(range(32, 44))}
    raw = np.full((1, 151936), -100, np.float32)
    raw[0, 32:36] = [3, 3, -2000, 0]
    saved = producer.output_questions(row, raw)[0]
    logs, choice = r.reconstruct_question(saved, {"candidate_ids": ids, "ordered": ordered}, 7, "r7")
    assert choice == 2 and -logs[0] > 2000


def test_equal_group_means_and_empty_support():
    rows = [{"service": x, "dialogue_id": x} for x in "AAAB"]
    assert r.means(np.array([1, 1, 1, 0]), rows, np.ones(4, bool)) == {
        "row": .75, "equal_service": .5, "equal_dialogue": .5}
    assert set(r.means(np.zeros(4), rows, np.zeros(4, bool)).values()) == {None}


def decision_fit(changed_correct, retained_errors, score=1):
    def c(n, correct):
        return {"rows": n, "counts": {"correct": correct, "error": n-correct},
                "metrics": {k: dict.fromkeys(r.WEIGHTINGS, score) for k in ("nll", "brier")}}
    cells = {"all": c(200, changed_correct+100-retained_errors), "changed": c(100, changed_correct),
             "retained": c(100, 100-retained_errors)}
    return {"cells": cells, "services": {"one": cells}}


def test_exact_two_point_boundary_and_independent_score_flags():
    control = decision_fit(50, 10)
    result = r.decisions(decision_fit(52, 10, 2), [control])
    assert result["behavioral"]["passed"] and not result["proper_score_nonregression"]["passed"]
    assert not r.decisions(decision_fit(51, 10), [control])["behavioral"]["passed"]
    assert not r.decisions(decision_fit(52, 11), [control])["behavioral"]["passed"]
    assert r.decisions(decision_fit(52, 10), [decision_fit(v, 10) for v in (49, 50, 51)])["behavioral"]["passed"]


def test_proper_score_equality_uses_fixed_historical_mean():
    a = float(np.nextafter(1., np.inf))
    controls = [decision_fit(50, 10, score=v) for v in (1., a, a)]
    result = r.decisions(decision_fit(52, 10, score=a), controls)
    assert all(v["difference"] == 0 and v["passed"] for v in result["proper_score_nonregression"]["checks"].values())


@pytest.fixture
def tree(tmp_path, monkeypatch):
    root, prepared, pilot, run = (tmp_path/n for n in ("repo", "prepared", "pilot", "run"))
    monkeypatch.setattr(r, "ROOT", root)
    monkeypatch.setattr(r, "SUPPORT", dict(zip(r.STRATA, (12, 6, 6, 3, 3), strict=True)))
    monkeypatch.setattr(r, "runtime", lambda: {"synthetic": True})
    for name in r.SOURCES | {"scripts/report_dialogue_qwen_observation.py", "tests/test_report_dialogue_qwen_observation.py"}:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source "+name)
    for path in (prepared, pilot, run):
        path.mkdir()
    ids = [r.NONE, r.DC, "value:True", "value:False"]
    labels, requests, scores, timings = [], [], [], []
    for service in range(6):
        questions, orders, maps = [], [], []
        for j in range(2):
            index = 2*service+j
            previous = r.NONE if j or service < 3 else ids[2]
            target = (ids[2] if service % 2 else r.DC) if j else previous
            labels.append({"row_index": index, "dialogue_id": f"d{service}", "time": 1, "query_id": f"q{index}",
                           "service": f"s{service}", "slot": f"slot{j}", "current_candidate_id": target,
                           "previous_candidate_id": previous, "candidate_count": 4, "boolean_slot": True,
                           "derived_bin": "first_assignment" if j else "unmentioned_retention" if service < 3 else "assigned_retention",
                           "current_value_group": r.TYPE_NAMES[r.candidate_type(target)]})
            candidates = []
            for k, text in enumerate(("Zebra", "Dog", "Apple", "Bird")):
                flags = [0]*10
                flags[4] = int(ids[k] == previous)
                desc = text+"\nCandidate type: "+("NOT_MENTIONED", "DONTCARE", "TRUE", "FALSE")[k]
                desc += "\nPublic lexical flags: "+",".join(map(str, flags))
                candidates.append({"id": f"c{k:02}", "description": desc})
            mapping = dict(zip((c["id"] for c in candidates), ids, strict=True))
            questions.append({"id": f"r{index}", "question": "Previous: "+previous, "candidates": candidates})
            orders.append([mapping[c["id"]] for c in sorted(candidates, key=lambda c: c["description"])])
            maps.append(mapping)
        for arm in r.ARMS if service % 2 == 0 else r.ARMS[::-1]:
            request = {"request_id": f"{arm}:d{service}:1:0", "arm": arm, "dialogue_id": f"d{service}", "time": 1,
                       "row_indices": [2*service, 2*service+1], "request": {"context": "Synthetic public context", "questions": questions},
                       "tokens": [[1, 2], [1, 2, 3]], "ordered_candidate_ids": orders,
                       "canonical_id_maps": maps, "label_ids": list(range(32, 44))}
            requests.append(request)
            output = []
            for index in request["row_indices"]:
                chosen = labels[index]["current_candidate_id"] if arm == "history4" and index % 3 == 0 else labels[index]["previous_candidate_id"]
                logits = [-2.]*4
                logits[ids.index(chosen)] = 2.
                output.append(distribution(logits, ids, orders[index % 2], index))
            score = {k: request[k] for k in ("request_id", "arm", "dialogue_id", "time")}
            score["questions"] = output
            scores.append(score)
            timings.append({"request_id": request["request_id"], "arm": arm, "seconds": .01, "questions": 2,
                            "work": r.work_for(request), "finite_logits": True,
                            "maximum_probability_mass_error": max(abs(math.fsum(q["probabilities"])-1) for q in output),
                            "serialized_score_bytes": len(r.encoded(score)), "mlx_peak_active_bytes": 20,
                            "mlx_cached_allocator_bytes": 10})
    plan = {"version": r.RUN_VERSION, "method": "batch", "runtime": r.runtime(), "limits": r.RUN_LIMITS,
            "model": {"id": r.MODEL[0], "revision": r.MODEL[1], "snapshot_path": "NEVER_OPEN_MODEL_FILES",
                      "files_sha256": dict.fromkeys(r.MODEL_FILES, "1"*64)},
            "source_sha256": {n: r.sha(root/n) for n in r.SOURCES}, "row_count": 12, "decisions": 24,
            "input_token_slots": dict.fromkeys(r.ARMS, 36), "request_counts": dict.fromkeys(r.ARMS, 6),
            "pilot_request_ids": [q["request_id"] for q in requests]}
    rows, layouts = r.request_layout(requests, labels, plan)
    logs, choices, _ = r.reconstruct(requests, scores, timings, layouts, rows)
    values = r.vectors(rows, logs["current"], choices["current"])
    groups = r.masks(rows)
    cells = {"heldout_service/"+k: r.cell(rows, choices["current"], values, m) for k, m in groups.items()}
    services = {s: {k: r.cell(rows, choices["current"], values, m & np.asarray([v["service"] == s for v in rows]))
                    for k, m in groups.items()} for s in sorted({v["service"] for v in rows})}
    history = {"status": "completed", "technical_validity_passed": True, "version": "dialogue-objective-v1", "fits": {},
               "historical": {"fits": {f"flat_stratum-corrected-{seed}": {"cells": cells, "services": services} for seed in r.SEEDS}}}
    reference = root/r.REFERENCE
    reference.parent.mkdir(parents=True)
    r.write(reference, history)
    monkeypatch.setattr(r, "REFERENCE_SHA", r.sha(reference))
    plan["inputs"] = {r.REFERENCE: r.item(reference)}
    for name, data in (("requests.jsonl", requests), ("labels.jsonl", labels)):
        (prepared/name).write_bytes(b"".join(r.encoded(v) for v in data))
    plan["files"] = {n: r.item(prepared/n) for n in ("requests.jsonl", "labels.jsonl")}
    r.write(prepared/"plan.json", plan)
    r.write(prepared/"started.json", {"synthetic": True})
    plan_pin = r.sha(prepared/"plan.json")
    def seal(path, done):
        done["files"] = {p.name: r.item(p) for p in path.iterdir() if p.name != "completed.json"}
        (path/"completed.json").write_bytes(r.encoded(done))
    seal(prepared, {"status": "completed", "plan_sha256": plan_pin, "model_calls": 0, "encoder_calls": 0,
                    "tokenizer_only": True, "official_dev_dialogues_accessed": False, "test_contents_accessed": False,
                    "wall_seconds": .1, "peak_rss_bytes": 100})
    for path, phase in ((pilot, "pilot"), (run, "run")):
        request = {"prepared": str(prepared), "plan_sha256": plan_pin}
        if phase == "run":
            request.update(pilot=str(pilot), pilot_sha256=r.sha(pilot/"completed.json"))
        r.write(path/"started.json", {"version": r.RUN_VERSION, "phase": phase, "request": request,
                                    "limits": r.RUN_LIMITS[phase], "labels_accessed": False, "no_retry": True})
        (path/"plan.json").write_bytes((prepared/"plan.json").read_bytes())
        (path/"timings.jsonl").write_bytes(b"".join(r.encoded(t) for t in timings))
        if phase == "run":
            (path/"scores.jsonl").write_bytes(b"".join(r.encoded(s) for s in scores))
        done = {"status": "completed", "version": r.RUN_VERSION, "phase": phase, "plan_sha256": plan_pin,
                "source_sha256": plan["source_sha256"], "runtime": plan["runtime"], "model": plan["model"],
                "limits": r.RUN_LIMITS[phase], "labels_accessed": False, "quality_outputs_saved": phase == "run",
                "generated_tokens": 0, "optimizer_updates": 0, "no_retry": True, "wall_seconds": 1.,
                "model_load_seconds": .1, "process_lifetime_peak_rss_bytes": 100, "request_seconds": .12, "model_calls": 12,
                "progress": {"requests_completed": 12, "questions_completed": 24, "forward_calls_attempted": 12,
                             "forward_calls_returned": 12, "active_request_id": None},
                "work_totals": {k: sum(t["work"][k] for t in timings) for k in r.work_for(requests[0])}}
        if phase == "pilot":
            done["projection"] = {"arms": {a: {"max_seconds_per_charged_slot": .01/6, "max_seconds_per_request": .01,
                  "token_scaled_seconds": .01/6*36, "request_scaled_seconds": .06, "variable_seconds": .01/6*36} for a in r.ARMS},
                  "factor": 2, "fixed_seconds": 60, "pilot_whole_wall_seconds": 1., "admission_limit_seconds": 7200,
                  "projected_full_seconds": 62, "admitted": True, "request_scaled_projection_seconds_descriptive_only": 62}
        else:
            done["pilot_completed_sha256"] = r.sha(pilot/"completed.json")
        seal(path, done)
    args = SimpleNamespace(prepared=prepared, plan_sha256=plan_pin, run=run,
                           run_sha256=r.sha(run/"completed.json"), out=tmp_path/"report")
    return SimpleNamespace(args=args, rows=rows, requests=requests, labels=labels, scores=scores, timings=timings,
                           layouts=layouts, logs=logs, choices=choices, plan=plan, seal=seal)


def test_full_saved_envelope_and_references(tree):
    summary = r.execute(tree.args)
    assert summary["technical_validity_passed"] and summary["counts"]["decisions"] == 24
    assert len(summary["services"]) == 6
    assert summary["controls"]["carry"]["cells"]["changed"]["accuracy"]["row"] == 0
    assert "nll" not in summary["controls"]["literal"]["cells"]["all"]
    pair = summary["paired"]["history4_minus_current"]["cells"]["all"]
    assert (pair["wrong_to_correct"], pair["correct_to_wrong"]) == (2, 0)
    assert len(summary["historical"]["corrected_flat"]["seeds"]) == 3
    receipt = r.read(tree.args.out/"receipt.json")
    assert receipt["model_calls"] == 0 and len(receipt["files"]) == 3
    with pytest.raises(FileExistsError):
        r.execute(tree.args)


def test_branch_partition_and_supported_false_positive_denominator(tree):
    values = r.vectors(tree.rows, tree.logs["current"], tree.choices["current"])
    cell = r.cell(tree.rows, tree.choices["current"], values, r.masks(tree.rows)["changed"])
    assert cell["counts"]["error"] == cell["counts"]["wrong_selected_branch"]+cell["counts"]["wrong_value"]
    assert cell["rare"]["false_recall"]["rate"] is None
    assert cell["rare"]["true_false_positive"]["denominator"] == 3


@pytest.mark.parametrize("damage", ["bytes", "incomplete", "extra", "source", "runtime", "plan", "overbudget"])
def test_authentication_rejects_before_prediction_decode(tree, monkeypatch, damage):
    path = tree.args.run
    done = r.read(path/"completed.json")
    if damage == "bytes":
        (path/"scores.jsonl").write_text("corrupt")
    elif damage == "extra":
        (path/"unrecorded").write_text("extra")
    elif damage == "source":
        (r.ROOT/next(iter(r.SOURCES))).write_text("changed")
    elif damage == "plan":
        tree.args.plan_sha256 = "0"*64
    else:
        done[{"incomplete": "status", "runtime": "runtime", "overbudget": "wall_seconds"}[damage]] = {
            "incomplete": "failed", "runtime": {"different": True}, "overbudget": 7201}[damage]
        (path/"completed.json").write_bytes(r.encoded(done))
        tree.args.run_sha256 = r.sha(path/"completed.json")
    original = r.decode
    def guard(value):
        if b'"candidate_logits"' in (value.encode() if isinstance(value, str) else value):
            raise AssertionError("Prediction decoded before technical admission")
        return original(value)
    monkeypatch.setattr(r, "decode", guard)
    with pytest.raises(ValueError):
        r.execute(tree.args)
    assert (tree.args.out/"failed.json").exists()


@pytest.mark.parametrize("damage", ["reorder", "selected", "labels", "lexical", "target"])
def test_resealed_semantic_corruption(tree, damage):
    requests, scores, timings, labels = map(copy.deepcopy, (tree.requests, tree.scores, tree.timings, tree.labels))
    if damage == "reorder":
        scores.reverse()
    elif damage == "selected":
        scores[0]["questions"][0]["selected_id"] = r.DC
    elif damage == "labels":
        scores[0]["questions"][0]["label_ids"].reverse()
    elif damage == "lexical":
        requests[0]["request"]["questions"][0]["candidates"][0]["description"] += ",0"
    else:
        labels[0]["current_candidate_id"] = "absent"
    for score, timing in zip(scores, timings, strict=True):
        timing["serialized_score_bytes"] = len(r.encoded(score))
    with pytest.raises(ValueError):
        rows, layouts = r.request_layout(requests, labels, tree.plan)
        r.reconstruct(requests, scores, timings, layouts, rows)


def test_label_perturbation_changes_metrics_not_reconstructed_choices(tree):
    rows = copy.deepcopy(tree.rows)
    rows[0]["target"] = 2
    logs, choices, _ = r.reconstruct(tree.requests, tree.scores, tree.timings, tree.layouts, rows)
    for arm in r.ARMS:
        np.testing.assert_array_equal(logs[arm], tree.logs[arm])
        np.testing.assert_array_equal(choices[arm], tree.choices[arm])
    old = r.vectors(tree.rows, tree.logs["current"], tree.choices["current"])
    new = r.vectors(rows, logs["current"], choices["current"])
    assert old["accuracy"].sum() == 6 and new["accuracy"].sum() == 5


def test_late_cap_preserves_demoted_receipt(tree, monkeypatch):
    original = r.write
    def write(path, value):
        original(path, value)
        if Path(path).name == "receipt.json":
            monkeypatch.setitem(r.LIMITS, "output_bytes", 0)
    monkeypatch.setattr(r, "write", write)
    with pytest.raises(ValueError, match="output cap"):
        r.execute(tree.args)
    assert (tree.args.out/"receipt-before-error.json").exists()
    assert (tree.args.out/"failed.json").exists() and not (tree.args.out/"receipt.json").exists()
