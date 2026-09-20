"""Synthetic saved-artifact checks; no corpus, encoder or model execution."""
import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("report_copy_v2_test", ROOT / "scripts/report_dialogue_copy_v2.py")
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)
base = reporter.base


def test_original_thirteen_checks_and_inclusive_boundaries_remain_unchanged():
    rows = {}
    for method in reporter.METHODS:
        hits = {"readout": 122, "scalar": 125, "selective": 126,
                "selective_no_lexical": 200, "candidate_gru": 126}[method]
        for seed in reporter.SEEDS:
            metric = {"strata": {s: {"count": 200, "correct": hits} for s in reporter.old.STRATA},
                      "micro": {"nll": .4}, "revision": {"count": 100, "correct": 60}}
            rows[f"{method}-{seed}"] = {"metrics": {p: copy.deepcopy(metric) for p in reporter.PANELS}}
    literal = {"strata": {s: {"count": 200, "correct": 120} for s in reporter.old.STRATA},
               "revision": {"count": 100, "correct": 61}}
    refs = {"literal": {p: copy.deepcopy(literal) for p in reporter.PANELS}}
    result = reporter.old.criteria(rows, refs)
    assert result["passed"] and result["checks_total"] == result["checks_passed"] == 13
    rows["selective-4101"]["metrics"]["seen"]["strata"]["changed"]["correct"] -= 1
    assert not reporter.old.criteria(rows, refs)["passed"]


def test_actor_coverage_includes_unscored_and_dummy_question_updates():
    dialogs = [{"turns": [0] * 5, "queries": [{"query": 1}, {"query": 2}]},
               {"turns": [0] * 2, "queries": [{"query": 3}]}]
    assert reporter.actor_counts(dialogs) == {"real_question_steps": 12,
        "executed_question_steps": 14, "advance_calls": 5, "forward_calls": 1}


def test_historical_deltas_are_descriptive_and_preserve_missing_nll():
    metric = {"macro_three": {"accuracy": .6}, "micro": {"nll": None, "brier": .5},
              "revision": {"accuracy": .4}}
    families = {m: {p: copy.deepcopy(metric) for p in reporter.PANELS} for m in reporter.METHODS}
    historical = {"study": "dialogue-copy-v1", "status": "completed", "families": copy.deepcopy(families)}
    historical["families"]["scalar"]["seen"]["macro_three"]["accuracy"] = .7
    result = reporter.historical_deltas(families, historical)
    assert result["methods"]["scalar"]["seen"]["macro_accuracy"]["v2_minus_v1"] == pytest.approx(-.1)
    assert result["methods"]["scalar"]["seen"]["micro_nll"]["v2_minus_v1"] is None
    assert "outside the continuation rule" in result["scope"]
    historical["study"] = "dialogue-copy-v2"
    with pytest.raises(ValueError, match="Historical"):
        reporter.historical_deltas(families, historical)


def packet_fixture():
    dialogs, queries = [], []
    for i, unseen in enumerate((False, True)):
        queries.append({"split": "dev", "candidate_ids": ["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:None", "value:other"],
                        "candidates": [0, 1, 2, 3]})
        dialogs.append({"id": f"synthetic-{i}", "turns": [0]*5, "queries": [
            {"query": i, "time": t, "label": label, "bin": label_bin, "unseen": unseen, "dontcare": label == 1}
            for t, (label, label_bin) in enumerate(zip((0, 2, 2, 3, 1),
                ("unmentioned_retention", "first_assignment", "assigned_retention", "revision", "revision"), strict=True))]})
    return {"queries": queries, "cohorts": {"train": [copy.deepcopy(dialogs[0])], "dev": dialogs}}


def witness(method, b=1):
    # This handwritten fixture has B dialogues, five turns and one question.
    n = b*5
    transport = method in ("scalar", "selective", "selective_no_lexical")
    return {"forward_calls": 1, "forward_returned": 1, "advance_calls": 5, "advance_returned": 5,
        "valid_turns": n, "executed_valid_question_slots": n, "real_question_updates": n,
        "incoming_checks": n, "feature_checks": n, "result_checks": n, "mass_checks": n if transport else 0,
        "mass_above_one_count": 0, "incoming_max_sum_error": 1e-7, "feature_max_sum_error": 1e-7,
        "result_max_sum_error": 1e-7, "mass_max_overshoot": 0.,
        "mass_min": .2 if transport else None, "mass_max": .4 if transport else None, "tolerance": 2e-6}


def work(b=1):
    return {"real_turns": 5*b, "padded_turn_positions": 5*b, "padded_query_positions": 5*b,
            "padded_candidate_positions": 20*b, "real_question_steps": 5*b}


def replace_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def seal(run, completion):
    completion["files"] = {p.relative_to(run).as_posix(): {"sha256": base.sha(p), "bytes": p.stat().st_size}
                           for p in run.rglob("*") if p.is_file() and p != run / "completed.json"}
    replace_json(run / "completed.json", completion)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    run, packet, lexical, prior = [root / name for name in ("run", "packet", "lexical", "prior")]
    for folder in (run, packet, lexical, prior):
        folder.mkdir()
    for name in reporter.SOURCES:
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
    data = packet_fixture()
    base.write(packet / "packet.json", data)
    for name in ("encoder-plan.json", "encoder-source.py", "features.npy"):
        (packet / name).write_bytes(b"opaque synthetic encoder bytes")
    base.write(packet / "completed.json", {"status": "completed", "cohorts": {"dev": {"queries": 10, "dialogues": 2, "turns": 10}},
        "files": {p.name: base.sha(p) for p in packet.iterdir()}, "wall_seconds": 1., "unique_texts": 2,
        "encoder_sequences": 2, "encoder_tokens_with_special": 4})
    for name in ("started.json", "index.json", "lexical.npy"):
        (lexical / name).write_bytes(b"opaque synthetic lexical bytes")
    base.write(lexical / "completed.json", {"status": "completed", "packet_completed_sha256": base.sha(packet / "completed.json"),
        "files": {p.name: base.sha(p) for p in lexical.iterdir()},
        "source_sha256": {reporter.OLD_PATH: reporter.OLD_SHA}})
    original_plan = {"study": "dialogue-copy-v1", "config": copy.deepcopy(reporter.CONFIG),
        "practical_checks": copy.deepcopy(reporter.PRACTICAL_CHECKS),
        "runtime": {k: "synthetic" for k in ("python", "torch", "numpy", "platform")},
        "packet_completed_sha256": base.sha(packet / "completed.json"),
        "lexical_completed_sha256": base.sha(lexical / "completed.json"),
        "source_sha256": {n: base.sha(root / n) for n in reporter.old.REQUIRED_SOURCES},
        "loss_counts": {"0": 1, "1": 1, "2": 3}, "loss_weights": [5/3, 5/3, 5/9], "optimizer_updates_per_fit": 20}
    base.write(prior / "plan.json", original_plan)
    originals = []
    for seed in reporter.SEEDS:
        for method in reporter.METHODS:
            params = 103411 if method == "candidate_gru" else 99458
            row = {"method": method, "seed": seed, "status": "completed", "parameters": params,
                   "initial_tensors_sha256": ("b" if method == "candidate_gru" else "a")*64,
                   "configuration": {"class": "DialogueCopyMemory", "method": method, "parameters": params,
                                     "projection_dim": 64, "hidden_dim": 64, "gru_width": 16}}
            folder = prior / "fits" / f"{method}-{seed}"
            folder.mkdir(parents=True)
            base.write(folder / "completed.json", row)
            originals.append(row)
    base.write(prior / "completed.json", {"status": "completed", "fit_count": 15,
        "plan_sha256": base.sha(prior / "plan.json"), "fits": originals})
    # Test-only external lineage pins for fully synthetic metadata; production constants remain immutable.
    monkeypatch.setattr(reporter, "OLD_PLAN", base.sha(prior / "plan.json"))
    monkeypatch.setattr(reporter, "OLD_COMPLETED", base.sha(prior / "completed.json"))
    plan = {**original_plan, "study": "dialogue-copy-v2", "implementation_version": reporter.VERSION,
        "wall_cap_seconds": 3600., "normalization_tolerance": 2e-6,
        "source_sha256": {n: base.sha(root / n) for n in reporter.SOURCES},
        "paths": {"packet": "packet", "lexical": "lexical", "old_study": "prior"},
        "old_plan_sha256": reporter.OLD_PLAN, "old_completed_sha256": reporter.OLD_COMPLETED}
    base.write(run / "plan.json", plan)
    base.write(run / "started.json", {"status": "started", "plan_sha256": base.sha(run / "plan.json"),
                                     "runtime": plan["runtime"], "wall_cap_seconds": 3600.})
    expected, _ = base.expected_records(data)
    probabilities = np.zeros((10, 12), np.float32)
    probabilities[:, :4] = .1
    probabilities[np.arange(10), expected["labels"]] = .7
    records = []
    for original in originals:
        method, seed = original["method"], original["seed"]
        folder = run / "fits" / f"{method}-{seed}"
        folder.mkdir(parents=True)
        (folder / "weights.pt").write_bytes(b"opaque tensor file, never deserialized")
        np.savez_compressed(folder / "dev-predictions.npz", **expected, probabilities=probabilities, choice=probabilities.argmax(-1))
        ledger = [{"epoch": e, "update": e+1, "indices": [0], "loss": .5, "supervised_queries": 5,
                   "actor_shapes": work(), "invariants": witness(method)} for e in range(20)]
        (folder / "batches.jsonl").write_text("".join(json.dumps(row)+"\n" for row in ledger))
        inv = witness(method)
        for key in reporter.COUNT_KEYS:
            inv[key] *= 20
        configuration = {**original["configuration"], "class": "MonitoredCopyMemoryV2", "implementation_version": reporter.VERSION,
            "normalization": "log_b -= logsumexp(log_b) before features/transition and after each real update",
            "padding": "exact original state values; no normalization committed on padded turns",
            "parameter_schema": "same as DialogueCopyMemory for the same method and dimensions",
            "old_fit_scope": "parameter loading possible, but not corrected training or empirical efficacy evidence"}
        evaluation = {"queries": 10, "wall_seconds": .2, "actor_shapes": work(2), "invariants": witness(method, 2),
            "batches": [{"indices": [0, 1], "actor_shapes": work(2), "invariants": witness(method, 2)}]}
        record = {**original, "configuration": configuration, "parameters_with_final_gradient": 99000,
            "original_initial_tensors_sha256": original["initial_tensors_sha256"], "common_initial_tensors_sha256": "a"*64,
            "epochs": 20, "updates": 20, "training_queries": 100, "training_losses": [.5]*20,
            "training_actor_shapes": {k: v*20 for k, v in work().items()}, "training_invariants": inv,
            "train_wall_seconds": 1., "fit_wall_seconds_before_receipt": 1.3, "evaluation": evaluation,
            "weights_sha256": base.sha(folder / "weights.pt"), "predictions_sha256": base.sha(folder / "dev-predictions.npz"),
            "batches_sha256": base.sha(folder / "batches.jsonl")}
        base.write(folder / "completed.json", record)
        records.append(record)
    np.savez_compressed(run / "references.npz", **expected, pred_none=np.zeros(10, np.int64), pred_literal=np.full(10, 2, np.int64))
    seal(run, {"status": "completed", "study": "dialogue-copy-v2", "fit_count": 15, "fits": records,
        "plan_sha256": base.sha(run / "plan.json"), "external_model_api_calls": 0, "normalization_tolerance": 2e-6,
        "resume_authorized": False, "wall_seconds": 21., "references": {"file": "references.npz", "queries": 10,
            "wall_seconds": .1, "sha256": base.sha(run / "references.npz")}})
    return root, run, packet, lexical


def invoke(tree, out):
    root, run, packet, lexical = tree
    return reporter.report(run, packet, lexical, base.sha(run / "plan.json"), base.sha(run / "completed.json"), out, root=root)


def test_synthetic_complete_report_all_fits_counts_metrics_plot_and_failed_gate(tree, tmp_path):
    out = tmp_path / "report"
    receipt = invoke(tree, out)
    assert receipt["technical_validity_passed"] and not receipt["continuation_passed"]
    assert len(receipt["execution_members"]) == 64
    assert receipt["neural_calls"] == receipt["checkpoint_deserializations"] == 0
    summary = base.read(out / "summary.json")
    assert summary["optimizer_updates"] == 300 and summary["supervised_presentations"] == 1500
    assert summary["technical_validity"]["training"]["result_checks"] == 1500
    assert summary["technical_validity"]["evaluation"]["result_checks"] == 150
    assert summary["technical_validity"]["training"]["mass_checks"] == 900
    assert summary["families"]["scalar"]["unseen"]["micro"]["nll"] == pytest.approx(-math.log(np.float32(.7)))
    assert summary["families"]["scalar"]["unseen"]["micro"]["brier"] == pytest.approx(.12)
    assert len(summary["continuation_gate"]["checks"]) == 13
    assert (out / "comparison.png").read_bytes().startswith(b"\x89PNG")
    for name, item in receipt["files"].items():
        assert base.sha(out / name) == item["sha256"]
    with pytest.raises(FileExistsError):
        invoke(tree, out)


@pytest.mark.parametrize("kind", ["weights", "extra_member", "source", "started", "cap", "tolerance", "config",
    "initial", "common_initial", "configuration", "missing_feature_check", "mass_violation", "batch_missing",
    "cursor", "label", "evaluation_order", "aggregate", "whole_cost"])
def test_resealed_corruption_cannot_pass_technical_validity(tree, tmp_path, kind):
    root, run, _, _ = tree
    folder = run / "fits/selective-4101"
    done = base.read(run / "completed.json")
    own = base.read(folder / "completed.json")
    if kind == "weights":
        (folder / "weights.pt").write_bytes(b"changed")
    elif kind == "extra_member":
        (run / "hidden-partial").write_bytes(b"unpermitted")
    elif kind == "source":
        (root / "scripts/study_dialogue_copy_v2.py").write_bytes(b"changed")
    elif kind == "started":
        start = base.read(run / "started.json")
        start["runtime"]["torch"] = "other"
        replace_json(run / "started.json", start)
    elif kind in ("cap", "tolerance", "config"):
        plan = base.read(run / "plan.json")
        if kind == "config":
            plan["config"]["gradient_clip"] = 2.
        else:
            plan["wall_cap_seconds" if kind == "cap" else "normalization_tolerance"] *= 2
        replace_json(run / "plan.json", plan)
        done["plan_sha256"] = base.sha(run / "plan.json")
    elif kind == "initial":
        own["initial_tensors_sha256"] = own["original_initial_tensors_sha256"] = "c"*64
    elif kind == "common_initial":
        own["common_initial_tensors_sha256"] = "c"*64
    elif kind == "configuration":
        own["configuration"]["normalization"] = "inference only"
    elif kind in ("missing_feature_check", "mass_violation", "batch_missing", "cursor"):
        ledger = [json.loads(line) for line in (folder / "batches.jsonl").read_text().splitlines()]
        if kind == "missing_feature_check":
            ledger[0]["invariants"]["feature_checks"] -= 1
        elif kind == "mass_violation":
            ledger[0]["invariants"]["incoming_max_sum_error"] = .02
        elif kind == "cursor":
            ledger[0]["update"] = 2
        else:
            ledger.pop()
        (folder / "batches.jsonl").write_text("".join(json.dumps(row)+"\n" for row in ledger))
        own["batches_sha256"] = base.sha(folder / "batches.jsonl")
    elif kind == "label":
        path = folder / "dev-predictions.npz"
        with np.load(path, allow_pickle=False) as saved:
            arrays = dict(saved)
        arrays["labels"][0] = 2
        np.savez_compressed(path, **arrays)
        own["predictions_sha256"] = base.sha(path)
    elif kind == "evaluation_order":
        own["evaluation"]["batches"][0]["indices"] = [1, 0]
    elif kind == "aggregate":
        own["training_invariants"]["result_checks"] -= 1
    else:
        done["wall_seconds"] = 1.
    replace_json(folder / "completed.json", own)
    done["fits"][2] = own
    seal(run, done)
    out = tmp_path / "failed-report"
    with pytest.raises(ValueError):
        invoke(tree, out)
    assert (out / "failed.json").exists() and not (out / "receipt.json").exists()


def test_monitor_roundoff_allowed_but_count_and_magnitude_bound():
    ds = packet_fixture()["cohorts"]["train"]
    record = witness("scalar")
    record.update(mass_max=1+1e-6, mass_max_overshoot=(1+1e-6)-1, mass_above_one_count=1)
    reporter.validate_invariants(record, ds, "scalar")
    record["mass_above_one_count"] = 0
    with pytest.raises(ValueError, match="overshoot"):
        reporter.validate_invariants(record, ds, "scalar")


def test_failure_receipt_error_preserves_original_exception(tree, tmp_path, monkeypatch):
    (tree[1] / "foreign").write_bytes(b"no")
    original = base.write
    def failed_write(path, obj):
        if Path(path).name == "failed.json":
            raise OSError("disk failure")
        return original(path, obj)
    monkeypatch.setattr(base, "write", failed_write)
    with pytest.raises(ValueError, match="64 execution") as error:
        invoke(tree, tmp_path / "failure")
    assert "disk failure" in error.value.__notes__[0]
