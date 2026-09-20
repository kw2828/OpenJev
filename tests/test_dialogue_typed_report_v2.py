"""Small synthetic saved-output checks; no corpus, model or encoder calls."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import report_dialogue_typed_v2 as report


def rows_fixture():
    identities = ("reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:true")
    rows = []
    for i, (old, new, service, dialogue, held) in enumerate([
            (0, 2, "s1", "d1", True), (0, 1, "s1", "d1", True),
            (0, 0, "s1", "d2", True), (2, 2, "s1", "d2", True),
            (0, 2, "s2", "d3", True), (0, 0, "s2", "d3", True),
            (2, 0, "s3", "d4", False)]):
        transition = (report.BINS[0] if old == new == 0 else report.BINS[1] if old == new else
                      report.BINS[2] if old == 0 else report.BINS[4] if new == 0 else report.BINS[3])
        rows.append({"row_index": i, "split": "train", "admission": "admitted", "candidate_count": 3,
                     "candidate_types": [0, 1, 2], "current_label_index": new, "previous_current_index": old,
                     "current_candidate_id": identities[new], "previous_candidate_id": identities[old],
                     "current_value_group": ("none", "dontcare", "true")[new], "derived_bin": transition,
                     "heldout_service": held, "service": service, "dialogue_id": dialogue, "query_id": service+"/q",
                     "query_index": 0})
    return rows


def packet(rows, probability=.6):
    values = np.full((len(rows), 3), (1-probability)/2)
    values[np.arange(len(rows)), [r["current_label_index"] for r in rows]] = probability
    logs = np.full((len(rows), 12), -np.inf, np.float32)
    logs[:, :3] = np.log(values)
    return {"row_indices": np.asarray([r["row_index"] for r in rows], np.int64), "log_probs": logs}


def references(rows):
    return {"row_indices": np.asarray([r["row_index"] for r in rows], np.int64),
            "previous_indices": np.asarray([r["previous_current_index"] for r in rows], np.int64),
            "literal_indices": np.asarray([r["current_label_index"] for r in rows], np.int64),
            "type_frequency_indices": np.zeros(len(rows), np.int64)}


def test_distinct_row_service_and_dialogue_weighting():
    rows = rows_fixture()
    group = report.layout(rows, np.asarray([True, True, True, True, True, True, False]))
    values = np.asarray([1., 1., 3., 3., 10., 10., 0.])
    got = report.means(values, group)
    assert got == {"row": 28/6, "equal_service": 6., "equal_dialogue": 14/3}


def test_all_twelve_all_groups_and_accuracy_only_references():
    rows = rows_fixture()
    got = report.aggregate(rows, {name: packet(rows) for name in report.FITS}, references(rows))
    assert len(got["fits"]) == 12 and got["groups_per_fit"] == 144
    cell = got["fits"]["typed_balanced-6101"]["cells"]["heldout_service/changed"]
    assert cell["rows"] == 3 and cell["services"] == 2
    assert cell["accuracy"]["row"] == 1
    assert cell["brier"]["row"] == pytest.approx(.24, abs=1e-7)
    assert "nll" not in got["references"]["literal"]["all/all"]
    assert got["continuation_allowed"] is False
    assert got["descriptive_factorial_contrasts"]["all/all"]["nll"]["row"]["interaction"]["mean_difference"] == 0


def test_raw_log_nll_remains_finite_when_probability_underflows():
    rows = rows_fixture()[:1]; saved = packet(rows)
    saved["log_probs"][0, :3] = [0., -1000., -1000.]
    values, _, _ = report.predictions(rows, saved)
    assert values["nll"].tolist() == [1000.]
    assert values["brier"].tolist() == [2.]


def test_fit_only_balancing_retains_equal_stratum_mass():
    rows = rows_fixture()
    recipe = report.objective_from_rows(rows)
    for weights in recipe["weights"].values():
        for counts, values in zip(recipe["counts"], weights, strict=True):
            assert sum(n*w for n, w in zip(counts, values, strict=True) if n) == pytest.approx(len(rows)/3)
    assert recipe["nonempty_categories"] == [1, 1, 3]


def passing_cells():
    fits = {}
    for seed in report.SEEDS:
        for method in (report.BASELINE, report.CANDIDATE):
            typed = method == report.CANDIDATE
            decisions = {key: {"numerator": 15 if typed else 10, "denominator": 100} for key in
                         ("true_changed_recall", "dontcare_changed_recall")}
            decisions.update({key: {"numerator": 11 if typed else 10, "denominator": 200} for key in
                              ("true_false_positive_rate", "dontcare_false_positive_rate", "retained_error")})
            fits[f"{method}-{seed}"] = {"cells": {"heldout_service/changed": {"nll": {"equal_service": .95 if typed else 1.}}}, "decisions": decisions}
    return fits


def test_exact_inclusive_fraction_thresholds():
    result = report.continuation(passing_cells())
    assert result["passed"] and result["checks_passed"] == result["total_checks"] == 9


@pytest.mark.parametrize("key", ["true_changed_recall", "dontcare_changed_recall", "true_false_positive_rate", "retained_error"])
def test_missing_denominator_is_unqualified(key):
    fits = passing_cells()
    for name in fits:
        fits[name]["decisions"][key] = {"numerator": 0, "denominator": 0}
    assert not report.continuation(fits)["passed"]


def test_rare_recall_gain_with_false_positive_harm_fails():
    fits = passing_cells()
    for seed in report.SEEDS:
        fits[f"{report.CANDIDATE}-{seed}"]["decisions"]["true_false_positive_rate"]["numerator"] = 12
    got = report.continuation(fits)
    assert not got["passed"]
    assert next(c for c in got["checks"] if c["name"] == "true_changed_recall")["passed"]


def test_good_mean_cannot_hide_bad_seed_and_zero_baseline_is_undefined():
    fits = passing_cells()
    for seed, loss in zip(report.SEEDS, (1.01, .2, .2), strict=True):
        fits[f"{report.CANDIDATE}-{seed}"]["cells"]["heldout_service/changed"]["nll"]["equal_service"] = loss
    got = report.continuation(fits)
    assert got["checks"][0]["passed"] and not got["passed"]
    for fit in fits.values(): fit["cells"]["heldout_service/changed"]["nll"]["equal_service"] = 0.
    assert not report.continuation(fits)["passed"]


def test_true_false_positive_denominator_only_supported_nontarget_rows():
    rows = rows_fixture()
    rows[-1]["candidate_types"] = [0, 1, 4]
    choice = np.asarray([r["current_label_index"] for r in rows])
    result = report.decision_counts(rows, choice)
    assert result["true_false_positive_rate"] == {"numerator": 0, "denominator": 3}
    assert result["dontcare_false_positive_rate"]["denominator"] == 5


@pytest.mark.parametrize("mutation", ["duplicate", "prior_id", "target_type", "flag", "support"])
def test_atomic_row_mapping_rejects_corruption(mutation):
    rows = rows_fixture()
    if mutation == "duplicate": rows[1]["row_index"] = 0
    if mutation == "prior_id": rows[0]["previous_candidate_id"] = rows[0]["current_candidate_id"]
    if mutation == "target_type": rows[0]["candidate_types"] = [0, 1, 4]
    if mutation == "flag": rows[0]["heldout_service"] = 1
    if mutation == "support": rows[0]["candidate_types"] = [0, 0, 2]
    with pytest.raises(ValueError): report.validate_rows(rows)


def test_prediction_order_mass_and_fit_omission_rejected():
    rows = rows_fixture(); p = packet(rows)
    p["row_indices"] = p["row_indices"][::-1]
    with pytest.raises(ValueError, match="identity"): report.predictions(rows, p)
    p = packet(rows); p["log_probs"][0, :3] = -.1
    with pytest.raises(ValueError, match="mass"): report.predictions(rows, p)
    with pytest.raises(ValueError, match="twelve"): report.aggregate(rows, {}, references(rows))


def test_mutated_or_foreign_manifest_member_rejected(tmp_path):
    (tmp_path/"payload").write_text("original")
    (tmp_path/"completed.json").write_text("{}")
    manifest = {"payload": {"sha256": report.digest(tmp_path/"payload"), "bytes": 8}}
    report.check_manifest(tmp_path, manifest, {"payload"})
    (tmp_path/"payload").write_text("mutation")
    with pytest.raises(ValueError, match="hash/size"): report.check_manifest(tmp_path, manifest, {"payload"})
    (tmp_path/"foreign").write_text("extra")
    with pytest.raises(ValueError, match="membership"): report.check_manifest(tmp_path, manifest, {"payload"})


def tree_manifest(path):
    return {p.relative_to(path).as_posix(): {"sha256": report.digest(p), "bytes": p.stat().st_size}
            for p in path.rglob("*") if p.is_file() and p.name != "completed.json"}


def synthetic_run(tmp_path, monkeypatch):
    root = tmp_path/"repo"; root.mkdir(); monkeypatch.setattr(report, "ROOT", root)
    monkeypatch.setattr(report, "correction_metadata", lambda pin: {"synthetic_fixture": True, "run_sha256": pin})
    prepared, frozen, run = (root/name for name in ("prepared", "frozen", "run"))
    for path in (prepared, frozen, run): path.mkdir()
    rows = rows_fixture()
    fit = copy.deepcopy(rows[:4])
    for i, row in enumerate(fit): row.update(row_index=i+100, service="fit", query_id="fit/q", dialogue_id="fit/d")
    canonical = [{k: v for k, v in r.items() if k not in ("candidate_types", "heldout_service")} for r in rows+fit]
    catalog = {"queries": [{"candidate_ids": ["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:true"],
                             "candidate_values": ["NOT_MENTIONED", "DONTCARE", "True"], "boolean_slot": True}]}
    report.write(prepared/"catalog.json", catalog)
    (prepared/"rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in canonical))
    for name in ("started.json", "summary.json"): report.write(prepared/name, {})
    report.write(prepared/"completed.json", {"files": tree_manifest(prepared)})
    sources = {}
    for name in report.SOURCE_NAMES:
        for base in (root, frozen/"sources"):
            path = base/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
        sources[name] = report.digest(root/name)
    order_info = {}
    for seed in report.SEEDS:
        order = np.tile(np.arange(len(fit), dtype=np.int64), (20, 1))
        for base in (run, frozen): np.save(base/f"orders-{seed}.npy", order)
        order_info[str(seed)] = {"file": f"orders-{seed}.npy", "sha256": report.digest(run/f"orders-{seed}.npy")}
    fit_order = [f"{m}-{s}" for i, s in enumerate(report.SEEDS) for m in report.METHODS[i:]+report.METHODS[:i]]
    split_root = root/"output/dialogue-typed-v1/split-design-01"; split_root.mkdir(parents=True)
    report.write(split_root/"membership.json", {"admitted_row_indices": {
        "fit": [r["row_index"] for r in fit], "evaluation": [r["row_index"] for r in rows],
        "primary_heldout_service": [r["row_index"] for r in rows if r["heldout_service"]],
        "secondary_nonheldout_service": [r["row_index"] for r in rows if not r["heldout_service"]],
        "secondary_seen_in_fit": [],
        "secondary_absent_from_fit": [r["row_index"] for r in rows if not r["heldout_service"]]}})
    report.write(split_root/"receipt.json", {"status": "completed", "alternate_splits_evaluated": 0,
                                            "files": tree_manifest(split_root)})
    plan = {"version": report.VERSION, "config": report.CONFIG, "limits": report.LIMITS, "runtime": {"fake": True},
            "source_sha256": sources, "prepared_path": str(prepared), "prepared_completed_sha256": report.digest(prepared/"completed.json"),
            "fit_row_indices": [r["row_index"] for r in fit], "evaluation_row_indices": [r["row_index"] for r in rows],
            "expected_fits": fit_order, "split": {"heldout_services": ["s1", "s2"]}, "orders": order_info,
            "updates_per_fit": 20, "evaluation_batches_per_fit": 1, "objective": report.objective_from_rows(fit),
            "split_receipt_sha256": report.digest(split_root/"receipt.json")}
    for base in (run, frozen): report.write(base/"plan.json", plan)
    pin = report.digest(run/"plan.json")
    report.write(frozen/"started.json", {})
    report.write(frozen/"completed.json", {"status": "completed", "phase": "freeze", "plan_sha256": pin, "files": tree_manifest(frozen)})
    report.write(run/"started.json", {"version": report.VERSION, "phase": "train", "runtime": plan["runtime"],
                                     "request": {"plan": str(frozen/"plan.json"), "plan_sha256": pin}})
    (run/"evaluation-rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    refs = references(rows); refs["type_frequency_indices"][:] = 2
    np.savez(run/"references.npz", **refs)
    def norm(n, batches=1):
        return {"batches": batches, "rows": n, "supported_candidates": 3*n, "masked_candidates": 0,
                "max_abs_mass_error": 0., "min_supported_log_prob": -2., "max_supported_log_prob": -.2}
    counts = {"forward_attempted": 21, "forward_returned": 21, "backward_attempted": 20, "backward_returned": 20,
              "optimizer_attempted": 20, "optimizer_returned": 20, "training_rows": 20*len(fit), "evaluation_rows": len(rows)}
    for name in fit_order:
        method, seed = name.rsplit("-", 1); dest = run/"fits"/name; dest.mkdir(parents=True)
        (dest/"weights.pt").write_bytes(b"opaque fake checkpoint")
        np.savez(dest/"predictions.npz", **packet(rows))
        journal = [{"epoch": e, "start": 0, "row_indices": [r["row_index"] for r in fit], "weighted_loss": 1.,
                    "normalization": norm(len(fit)), "work": {"rows": len(fit)}, "wall_seconds": .01} for e in range(20)]
        (dest/"updates.jsonl").write_text("".join(json.dumps(e)+"\n" for e in journal))
        config = {"class": "DialogueTypedObservation", "version": "dialogue-typed-observation-v1", "mode": method.split("_")[0],
                  "attention_mode": "candidate", "input_dim": 384, "projection_dim": 64, "hidden_dim": 64,
                  "feature_dim": 401, "parameters": 173506, "attention_parameters": 73728,
                  "shared_scorer_parameters": 99778, "schema_pair": "[query;candidate]"}
        report.write(dest/"completed.json", {"status": "completed", "method": method, "seed": int(seed), "plan_sha256": pin,
                     "orders_sha256": order_info[seed]["sha256"], "initial_state_sha256": "a"*64, "configuration": config,
                     "counts": counts, "files": tree_manifest(dest), "training_normalization": norm(20*len(fit), 20),
                     "training_work": {"rows": 20*len(fit)}, "training_wall_seconds": .8,
                     "evaluation": {"normalization": norm(len(rows)), "work": {"rows": len(rows)}, "wall_seconds": .1}, "wall_seconds": 1.})
    manifest = {p.relative_to(run).as_posix(): {"sha256": report.digest(p), "bytes": p.stat().st_size} for p in run.rglob("*") if p.is_file()}
    report.write(run/"completed.json", {"status": "completed", "phase": "train", "version": report.VERSION,
                 "completed_fits": fit_order, "expected_fits": fit_order, "files": manifest, "plan_sha256": pin,
                 "source_sha256": sources, "limits": report.LIMITS, "runtime": plan["runtime"], "wall_seconds": 20.,
                 "process_lifetime_peak_rss_bytes": 1000,
                 "progress": {"completed_fits": fit_order, "active_fit": None, "totals": {k: 12*v for k, v in counts.items()}}})
    return run, report.digest(run/"completed.json")


def test_complete_authenticated_synthetic_cli_and_exclusive_output(tmp_path, monkeypatch):
    run, pin = synthetic_run(tmp_path, monkeypatch)
    args = SimpleNamespace(run=run, run_sha256=pin, out=tmp_path/"report")
    assert report.execute(args) is False
    receipt = report.read(args.out/"receipt.json")
    assert receipt["execution_files"] == 56 and receipt["technical_validity_passed"]
    assert set(receipt["files"]) == {"summary.json", "report.md"}
    with pytest.raises(FileExistsError): report.execute(args)


def test_wrong_pinned_completion_preserves_failure_without_prediction_load(tmp_path, monkeypatch):
    run, _ = synthetic_run(tmp_path, monkeypatch)
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("No numeric payload before authentication"))
    args = SimpleNamespace(run=run, run_sha256="0"*64, out=tmp_path/"failed-report")
    with pytest.raises(ValueError, match="External"): report.execute(args)
    assert report.read(args.out/"failed.json")["status"] == "failed"


def projected_membership():
    fit = [{"row_index": 1, "service": "seen"}]
    rows = [{"row_index": 3, "service": "held"}, {"row_index": 4, "service": "seen"}, {"row_index": 7, "service": "absent"}]
    membership = {"fit": [1], "evaluation": [3, 4, 7], "primary_heldout_service": [3],
                  "secondary_nonheldout_service": [4, 7], "secondary_seen_in_fit": [4], "secondary_absent_from_fit": [7]}
    return membership, fit, rows


def test_six_projected_subpanels_and_empty_absent_subset_are_valid():
    membership, fit, rows = projected_membership()
    report.validate_split_membership(membership, fit, rows, {"held"})
    rows = rows[:2]
    membership.update(evaluation=[3, 4], secondary_nonheldout_service=[4], secondary_absent_from_fit=[])
    report.validate_split_membership(membership, fit, rows, {"held"})


@pytest.mark.parametrize("key", ["fit", "evaluation", "primary_heldout_service", "secondary_nonheldout_service",
                                  "secondary_seen_in_fit", "secondary_absent_from_fit"])
def test_each_projected_subpanel_is_checked_not_discarded(key):
    membership, fit, rows = projected_membership()
    membership[key] = []
    with pytest.raises(ValueError, match="subpanel"): report.validate_split_membership(membership, fit, rows, {"held"})


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered", "duplicate", "bool"])
def test_projected_schema_and_integer_order_are_strict(mutation):
    membership, fit, rows = projected_membership()
    if mutation == "missing": membership.pop("secondary_seen_in_fit")
    if mutation == "extra": membership["foreign"] = []
    if mutation == "reordered": membership["evaluation"] = [7, 4, 3]
    if mutation == "duplicate": membership["secondary_seen_in_fit"] = [4, 4]
    if mutation == "bool": membership["fit"] = [True]
    with pytest.raises(ValueError): report.validate_split_membership(membership, fit, rows, {"held"})


def test_all_original_functions_except_authentication_and_output_are_ast_identical():
    root = Path(__file__).resolve().parents[1]
    original = root/"scripts/report_dialogue_typed.py"
    require_pin = report.digest(original)
    assert require_pin == report.ORIGINAL_REPORTER_SHA256
    old = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(original.read_text()).body if isinstance(n, ast.FunctionDef)}
    new = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(Path(report.__file__).read_text()).body if isinstance(n, ast.FunctionDef)}
    assert {name for name in old if old[name] != new[name]} == {"authenticate_run", "execute"}
    assert set(new)-set(old) == {"validate_split_membership", "correction_metadata"}


def test_correction_provenance_guards_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    original = tmp_path/"scripts/report_dialogue_typed.py"; original.parent.mkdir(); original.write_text("original")
    failed = tmp_path/"output/dialogue-typed-v1/analysis-01/failed.json"; failed.parent.mkdir(parents=True)
    report.write(failed, {"status": "failed", "execution_completed_sha256": "a"*64, "model_calls": 0, "wall_seconds": 1.})
    tests = tmp_path/"tests/test_dialogue_typed_report_v2.py"; tests.parent.mkdir(); tests.write_text("tests")
    monkeypatch.setattr(report, "ORIGINAL_REPORTER_SHA256", report.digest(original))
    monkeypatch.setattr(report, "FAILED_REPORT_SHA256", report.digest(failed))
    metadata = report.correction_metadata("a"*64)
    assert metadata["training_rerun"] is False and len(metadata["membership_fields_verified"]) == 6
    assert metadata["corrected_tests_sha256"] == report.digest(tests)
    with pytest.raises(ValueError, match="unchanged training"): report.correction_metadata("b"*64)
    failed.write_text("changed")
    with pytest.raises(ValueError, match="failed report identity"): report.correction_metadata("a"*64)
