"""Synthetic reporting checks; no study predictions or model calls."""
from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import report_dialogue_conditional as report


def rows_fixture():
    ids = ("reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:true")
    entries = [(0, 2, True, "d0"), (2, 2, True, "d0"), (0, 1, True, "d0"),
               (2, 0, False, "d2"), (0, 0, False, "d2"), (1, 2, True, "d1")]
    rows = []
    for i, (previous, target, unseen, dialogue) in enumerate(entries):
        old, current = ids[previous], ids[target]
        group = ("unmentioned_retention" if previous == target == 0 else "assigned_retention" if previous == target
                 else "first_assignment" if previous == 0 else "clear" if target == 0 else "revision")
        rows.append({"row_index": i * 2, "split": "dev", "admission": "admitted", "candidate_count": 3,
                     "current_label_index": target, "previous_current_index": previous, "unseen": unseen,
                     "derived_bin": group, "current_value_group": ("none", "dontcare", "true")[target],
                     "dialogue_id": dialogue, "query_id": "q" + str(i % 2),
                     "previous_candidate_id": old, "current_candidate_id": current})
    return rows


def packet(rows, target_probability=.5):
    probabilities = np.full((len(rows), 3), (1. - target_probability) / 2.)
    probabilities[np.arange(len(rows)), [r["current_label_index"] for r in rows]] = target_probability
    values = np.full((len(rows), 12), -np.inf, dtype=np.float32)
    values[:, :3] = np.log(probabilities).astype(np.float32)
    return {"log_probs": values, "row_indices": np.asarray([r["row_index"] for r in rows], dtype=np.int64)}


def references(rows):
    return {"row_indices": np.asarray([r["row_index"] for r in rows], dtype=np.int64),
            "previous_indices": np.asarray([r["previous_current_index"] for r in rows], dtype=np.int64),
            "literal_indices": np.asarray([r["current_label_index"] for r in rows], dtype=np.int64)}


def all_predictions(rows, candidate_probability=.6):
    return {(mode, seed): packet(rows, candidate_probability if mode == "candidate" else .5)
            for mode in report.MODES for seed in report.SEEDS}


def test_direct_nll_brier_denominators_and_all_fixed_groups():
    rows = rows_fixture()
    p = packet(rows)
    original = p["log_probs"].copy()
    result = report.fit_metrics(rows, **p)
    cell = result["cells"]["unseen/changed"]
    assert cell["rows"] == 3 and cell["dialogues"] == 2
    assert cell["distinct_schema_queries"] == 2 and cell["dialogue_query_streams"] == 2
    assert cell["accuracy"] == 1. and cell["nll"] == pytest.approx(np.log(2.), abs=3e-8)
    assert cell["brier"] == pytest.approx(.375, abs=3e-8)
    assert set(result["cells"]) == set(report.groups(rows))
    assert result["cells"]["unseen/value/false"]["accuracy"] is None
    assert result["cells"]["unseen/value/false"]["nll"] is None
    assert result["cells"]["unseen/value/false"]["equal_dialogue_brier"] is None
    assert np.array_equal(p["log_probs"], original)


def test_equal_dialogue_means_are_not_row_weighted_means():
    rows = rows_fixture()
    selected = report.groups(rows)["unseen/changed"]
    values = np.asarray([1., 0., 3., 0., 0., 10.])
    result = report.summarize_values(rows, selected, values)
    assert result["row_mean"] == pytest.approx(14/3)
    assert result["equal_dialogue_mean"] == 6.


def test_nll_uses_finite_log_probabilities_without_underflow_floor():
    rows = [rows_fixture()[0]]
    p = packet(rows)
    p["log_probs"][0, :3] = np.asarray([0., -1000., -1000.], np.float32)
    result = report.fit_metrics(rows, **p)["cells"]["unseen/changed"]
    assert result["nll"] == 1000. and result["accuracy"] == 0.
    assert result["brier"] == 2.


@pytest.mark.parametrize("corruption", ["dtype", "shape", "rows", "nan", "support_inf", "positive",
                                        "padding", "mass", "mask_nan"])
def test_invalid_saved_predictions_rejected(corruption):
    rows = rows_fixture()
    p = packet(rows)
    if corruption == "dtype":
        p["log_probs"] = p["log_probs"].astype(np.float64)
    elif corruption == "shape":
        p["log_probs"] = p["log_probs"][:, :3]
    elif corruption == "rows":
        p["row_indices"] = p["row_indices"][::-1]
    elif corruption == "nan":
        p["log_probs"][0, 0] = np.nan
    elif corruption == "support_inf":
        p["log_probs"][0, 0] = -np.inf
    elif corruption == "positive":
        p["log_probs"][0, 0] = 1e-8
    elif corruption == "padding":
        p["log_probs"][0, 7] = -1000.
    elif corruption == "mass":
        p["log_probs"][0, :3] = np.log(.2)
    else:
        p["log_probs"][0, 7] = np.nan
    with pytest.raises(ValueError):
        report.fit_metrics(rows, **p)


def test_all_nine_fits_and_every_pair_are_required():
    rows = rows_fixture()
    p = all_predictions(rows)
    result = report.analyze(rows, p, references(rows))
    assert len(result["fits"]) == 9 and len(result["primary"]["paired"]) == 3
    assert result["primary"]["primary_nll_rule_passed"] is True
    assert result["primary"]["mean_candidate_minus_slot_nll"] == pytest.approx(np.log(.5/.6), abs=1e-7)
    assert result["architecture_advantage_established"] is False
    p[("candidate", 5303)] = packet(rows, .49)
    result = report.analyze(rows, p, references(rows))
    assert result["primary"]["mean_candidate_minus_slot_nll"] < 0
    assert result["primary"]["primary_nll_rule_passed"] is False
    del p[("mean", 5301)]
    with pytest.raises(ValueError, match="nine"):
        report.analyze(rows, p, references(rows))


def test_ties_and_empty_primary_support_fail_without_missing_zero_confusion():
    rows = rows_fixture()
    result = report.analyze(rows, all_predictions(rows, .5), references(rows))
    assert result["primary"]["mean_candidate_minus_slot_nll"] == 0.
    assert result["primary"]["primary_nll_rule_passed"] is False
    rows = [rows[1]]
    result = report.analyze(rows, all_predictions(rows), references(rows))
    assert result["primary"]["rows"] == 0
    assert result["primary"]["mean_candidate_minus_slot_nll"] is None
    assert result["primary"]["relative_mean_nll_change"] is None
    assert result["primary"]["primary_nll_rule_passed"] is False


def test_reference_accuracy_uses_same_denominator_and_no_probability_scores():
    rows = rows_fixture()
    result = report.reference_metrics(rows, **references(rows))
    carry = result["previous_gold_carry"]["cells"]
    assert carry["unseen/changed"]["accuracy"] == 0.
    assert carry["unseen/retained"]["accuracy"] == 1.
    assert carry["unseen/changed"]["rows"] == 3
    assert "nll" not in carry["unseen/changed"]
    assert result["literal_carry"]["cells"]["unseen/changed"]["accuracy"] == 1.
    ref = references(rows)
    ref["previous_indices"][0] = 1
    with pytest.raises(ValueError, match="previous"):
        report.reference_metrics(rows, **ref)


@pytest.mark.parametrize("field,value", [("row_index", True), ("candidate_count", True),
                                         ("current_label_index", -1), ("previous_current_index", True),
                                         ("unseen", 1), ("derived_bin", "revision"),
                                         ("split", "train"), ("admission", "first_public_turn")])
def test_bad_evaluator_metadata_rejected(field, value):
    rows = rows_fixture()
    rows[0][field] = value
    with pytest.raises(ValueError):
        report.validate_rows(rows)


def test_duplicate_rows_and_wrong_canonical_prior_mapping_rejected():
    rows = rows_fixture()
    rows[1]["row_index"] = rows[0]["row_index"]
    with pytest.raises(ValueError):
        report.validate_rows(rows)
    rows = rows_fixture()
    rows[0]["previous_current_index"] = rows[0]["current_label_index"]
    with pytest.raises(ValueError, match="mapping"):
        report.validate_rows(rows)


def test_analysis_preserves_all_input_artifacts_and_has_no_neural_imports():
    rows = rows_fixture()
    p, refs = all_predictions(rows), references(rows)
    old = copy.deepcopy((rows, p, refs))
    report.analyze(rows, p, refs)
    assert rows == old[0]
    for key in p:
        for field in p[key]:
            assert np.array_equal(p[key][field], old[1][key][field])
    for key in refs:
        assert np.array_equal(refs[key], old[2][key])
    assert "import torch" not in inspect.getsource(report)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def file_record(path):
    return {"sha256": report.digest(path), "bytes": path.stat().st_size}


def manifest(path):
    return {p.relative_to(path).as_posix(): file_record(p) for p in path.rglob("*")
            if p.is_file() and p != path / "completed.json"}


def seal(path):
    record = report.read_json(path / "completed.json")
    record["files"] = manifest(path)
    write_json(path / "completed.json", record)
    return report.digest(path / "completed.json")


def synthetic_geometry(rows, mode):
    """Independent producer-shaped metadata; no actor/model construction."""
    b, c = len(rows), max(r["candidate_count"] for r in rows)
    lengths = [r["cache"]["token_stop"] - r["cache"]["token_start"] for r in rows]
    length = 0 if mode == "mean" else max(lengths)
    floats = b * (384 if mode == "mean" else length * 385) + b * 384 + b * c * 395
    work = {"rows": b, "supported_candidate_positions": sum(r["candidate_count"] for r in rows),
            "padded_candidate_positions": b*c, "supported_token_positions": 0 if mode == "mean" else sum(lengths),
            "padded_token_positions": b*length, "attention_score_positions": b*c*length,
            "token_key_positions": b*length, "attention_query_positions": 0 if mode == "mean" else b*c,
            "scorer_positions": b*c, "float_input_scalars": floats, "float_input_bytes": floats*4,
            "boolean_input_bytes": b*c+b*length, "max_token_length": length, "max_candidate_count": c}
    shapes = {"observation": [b, 384] if mode == "mean" else [b, length, 384],
              "query": [b, 384], "candidates": [b, c, 384], "candidate_mask": [b, c],
              "lexical": [b, c, 10], "previous_onehot": [b, c]}
    if mode != "mean":
        shapes.update(token_mask=[b, length], token_prior=[b, length])
    return work, shapes


def normalization(work, batches=1):
    return {"batches": batches, "rows": work["rows"],
            "supported_candidates": work["supported_candidate_positions"],
            "masked_candidates": work["padded_candidate_positions"]-work["supported_candidate_positions"],
            "max_abs_mass_error": 1e-8, "min_supported_log_prob": -3., "max_supported_log_prob": -.1}


def synthetic_configuration(mode):
    return {"class": "DialogueConditionalObservation", "version": "dialogue-conditional-observation-v1",
            "mode": mode, "input_dim": 384, "projection_dim": 64, "hidden_dim": 64, "feature_dim": 396,
            "parameters": 99393 if mode == "mean" else 173121, "shared_scorer_parameters": 99393,
            "attention_parameters": 0 if mode == "mean" else 73728, "zero_input_parameters": 64,
            "softmax_shift_parameters": 1, "attention_width": None if mode == "mean" else 64,
            "schema_pair": None if mode == "mean" else "[query;query]" if mode == "slot" else "[query;candidate]",
            "lexical_fields": ["user_match", "system_match", "unique_longest_user", "unique_longest_system",
                               "literal_current", "literal_previous", "is_none", "is_dontcare",
                               "affirmative_cue_for_true", "negative_cue_for_false"]}


@pytest.fixture
def complete_run(tmp_path, monkeypatch):
    """Nine fake fits with real manifests/joins, no Torch or scientific inputs."""
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "FEATURE_PARENTS", {})
    prepared, frozen, run = (tmp_path / name for name in ("prepared", "frozen", "run"))
    for path in (prepared, frozen, run):
        path.mkdir()
    ids = ["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:true"]
    ledger, catalog, train, dev = [], [], [], []
    for split, examples in (("train", [(0, 0, False), (2, 2, False), (0, 2, False)]),
                            ("dev", [(0, 2, True), (2, 2, True), (0, 1, True),
                                     (2, 0, False), (0, 0, False), (1, 2, True)])):
        for previous, current, unseen in examples:
            qi = len(catalog)
            query = {"query_index": qi, "query_id": f"q{qi}", "split": split, "service": f"s{qi}",
                     "slot": "flag", "candidate_ids": ids, "candidate_values": [None, None, "TrUe"],
                     "boolean_slot": True}
            catalog.append(query)
            for step, label in enumerate((previous, current)):
                row = {"row_index": len(ledger), "query_index": qi, "query_id": query["query_id"],
                       "split": split, "service": query["service"], "slot": "flag", "dialogue_id": f"d{qi}",
                       "time": step, "candidate_count": 3, "current_label_index": label,
                       "current_candidate_id": ids[label], "current_value_group": ("none", "dontcare", "true")[label],
                       "boolean_slot": True, "unseen": unseen,
                       "admission": "admitted" if step else "first_public_turn",
                       "cache": {"token_start": qi * 5, "token_stop": qi * 5 + 2 + qi % 3,
                                 "lexical_start": 30*len(ledger), "lexical_candidates": 3, "lexical_stride": 10}}
                if step:
                    row.update(previous_current_index=previous, previous_candidate_id=ids[previous],
                               previous_row_index=len(ledger)-1,
                               derived_bin="unmentioned_retention" if previous == label == 0 else
                               "assigned_retention" if previous == label else "first_assignment" if previous == 0 else
                               "clear" if label == 0 else "revision")
                    (train if split == "train" else dev).append(row)
                ledger.append(row)
    write_json(prepared / "catalog.json", {"queries": catalog})
    (prepared / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ledger))
    write_json(prepared / "started.json", {"synthetic": True})
    write_json(prepared / "summary.json", {"synthetic": True, "admitted_rows": len(train)+len(dev)})
    lexical = np.zeros(30*len(ledger), dtype=np.float32)
    for row in ledger:
        lexical[row["cache"]["lexical_start"]+row["current_label_index"]*10+4] = 1.
    lexical_path = tmp_path / "lexical.npy"
    np.save(lexical_path, lexical, allow_pickle=False)
    write_json(prepared / "completed.json", {"status": "completed", "phase": "prepare",
               "plan_sha256": "f"*64, "files": manifest(prepared),
               "wall_seconds": .1, "wall_scope": "Synthetic metadata preparation",
               "authenticated_inputs": {str(lexical_path): file_record(lexical_path)}})
    sources = {}
    for name in report.SOURCE_NAMES:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("Synthetic source identity fixture: " + name + "\n")
        dest = frozen / "sources" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
        sources[name] = report.digest(p)
    orders, order_records, schedules = {}, {}, {}
    for seed in report.SEEDS:
        rng = np.random.default_rng(seed)
        orders[seed] = np.stack([rng.permutation(len(train)) for _ in range(20)]).astype(np.int64)
        for folder in (run, frozen):
            np.save(folder / f"orders-{seed}.npy", orders[seed], allow_pickle=False)
        order_records[str(seed)] = {"file": f"orders-{seed}.npy", "shape": [20, len(train)],
                                   "sha256": report.digest(run / f"orders-{seed}.npy")}
        for mode in report.MODES:
            work, _ = synthetic_geometry(train, mode)
            eval_work, _ = synthetic_geometry(dev, mode)
            totals = {k: 20*v for k, v in work.items() if not k.startswith("max_")}
            schedules[f"{mode}-{seed}"] = {
                "training": {"batches": 20, "totals": totals, "maximum_batch_float_input_bytes": work["float_input_bytes"]},
                "evaluation": {"batches": 1, "totals": {k: v for k, v in eval_work.items() if not k.startswith("max_")},
                               "maximum_batch_float_input_bytes": eval_work["float_input_bytes"]}}
    plan = {"version": report.VERSION, "expected_fits": report.FIT_ORDER, "source_sha256": sources,
            "config": {"methods": list(report.MODES), "seeds": list(report.SEEDS), "epochs": 20, "batch_size": 256,
                       "learning_rate": .001, "weight_decay": .0001, "gradient_clip": 1., "input_dim": 384,
                       "projection_dim": 64, "hidden_dim": 64, "threads": 4, "interop_threads": 1,
                       "dtype": "float32", "deterministic": True},
            "limits": {"wall_seconds": 3600., "rss_bytes": 6*1024**3, "output_bytes": 512*1024**2},
            "runtime": {"fixture": "synthetic"}, "prepared_path": str(prepared),
            "prepared_completed_sha256": report.digest(prepared / "completed.json"),
            "prepared_files": manifest(prepared), "admitted_train_rows": len(train), "admitted_dev_rows": len(dev),
            "objective": {"counts": [1, 1, 1], "weights": [1., 1., 1.]}, "orders": order_records,
            "updates_per_fit": 20, "evaluation_batches_per_fit": 1, "work_schedules": schedules,
            "feature_headers": {"lexical": {"path": str(lexical_path), "shape": list(lexical.shape)}}}
    for folder in (run, frozen):
        write_json(folder / "plan.json", plan)
    plan_pin = report.digest(run / "plan.json")
    (frozen / "prepared-completed.json").write_bytes((prepared / "completed.json").read_bytes())
    write_json(frozen / "started.json", {"synthetic": True})
    write_json(frozen / "completed.json", {"status": "completed", "phase": "freeze",
               "plan_sha256": plan_pin, "files": manifest(frozen)})
    write_json(run / "started.json", {"request": {"plan": str(frozen / "plan.json"), "plan_sha256": plan_pin}})
    np.savez(run / "references.npz", **references(dev))
    counts = {"forward_attempted": 21, "forward_returned": 21, "backward_attempted": 20,
              "backward_returned": 20, "optimizer_attempted": 20, "optimizer_returned": 20,
              "training_rows": 60, "evaluation_rows": 6}
    fit_summaries = []
    for name in report.FIT_ORDER:
        mode, seed_text = name.rsplit("-", 1)
        seed = int(seed_text)
        dest = run / "fits" / name
        dest.mkdir(parents=True)
        (dest / "weights.pt").write_bytes(b"Synthetic opaque checkpoint. Must never be loaded.\n")
        np.savez(dest / "dev-predictions.npz", **packet(dev, .6 if mode == "candidate" else .5))
        events = []
        for epoch, order in enumerate(orders[seed]):
            rows = [train[int(i)] for i in order]
            work, shapes = synthetic_geometry(rows, mode)
            events.append({"epoch": epoch, "start": 0, "update": epoch+1,
                           "row_indices": [r["row_index"] for r in rows], "work": work, "actor_shapes": shapes,
                           "normalization": normalization(work), "weighted_loss": .8,
                           "phase_seconds": {"assembly": .01, "forward_validation_loss": .01,
                                             "backward_clip": .01, "optimizer": .01}})
        (dest / "updates.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        train_work = schedules[name]["training"]["totals"]
        eval_work = schedules[name]["evaluation"]["totals"]
        record = {"status": "completed", "method": mode, "seed": seed, "epochs": 20, "counts": counts,
                  "initial_common_sha256": f"{seed:064x}", "initial_attention_sha256": None if mode == "mean" else f"{seed+1:064x}",
                  "orders_sha256": order_records[str(seed)]["sha256"], "configuration": synthetic_configuration(mode),
                  "training_normalization": normalization(train_work, 20), "training_work": train_work,
                  "evaluation": {"rows": len(dev), "work": eval_work, "normalization": normalization(eval_work), "wall_seconds": .1},
                  "wall_seconds": 1., "process_lifetime_peak_rss_bytes": 1000, "plan_sha256": plan_pin,
                  "files": manifest(dest)}
        write_json(dest / "completed.json", record)
        fit_summaries.append({"method": mode, "seed": seed, "counts": counts})
    done = {"status": "completed", "version": report.VERSION, "phase": "train", "no_retry": True,
            "completed_fits": report.FIT_ORDER, "expected_fits": report.FIT_ORDER, "quality_metrics_computed": False,
            "encoder_calls": 0, "test_contents_accessed": False, "files": manifest(run), "plan_sha256": plan_pin,
            "runtime": plan["runtime"], "source_sha256": sources, "prepared_completed_sha256": plan["prepared_completed_sha256"],
            "wall_seconds": 12., "wall_scope": "Synthetic complete run", "process_lifetime_peak_rss_bytes": 2000,
            "progress": {"completed_fits": report.FIT_ORDER, "active_fit": None, "totals": {k: 9*v for k, v in counts.items()}},
            "fits": fit_summaries}
    write_json(run / "completed.json", done)
    return SimpleNamespace(root=tmp_path, run=run, frozen=frozen, prepared=prepared, plan=plan, train=train, dev=dev,
                           ledger=ledger, catalog=catalog, lexical=lexical_path, pin=report.digest(run / "completed.json"))


def test_full_saved_envelope_and_cli_without_models(complete_run, monkeypatch, capsys):
    f = complete_run
    out = f.root / "report"
    monkeypatch.setattr("sys.argv", ["report", "--run", str(f.run), "--completed-sha256", f.pin, "--out", str(out)])
    report.main()
    receipt = json.loads(capsys.readouterr().out)
    summary = report.read_json(out / "summary.json")
    assert receipt["status"] == "completed" and receipt["model_calls"] == 0
    assert receipt["summary_sha256"] == report.digest(out / "summary.json")
    assert summary["technical_admission_passed"] and summary["continuation_allowed"]
    assert len(summary["fits"]) == 9 and len(summary["cost"]["fits"]) == 9
    assert summary["primary"]["rows"] == 3
    assert summary["architecture_advantage_established"] is False
    assert len(manifest(f.run)) == 42
    with pytest.raises(FileExistsError):
        report.execute_report(f.run, f.pin, out)


@pytest.mark.parametrize("corruption", ["incomplete", "wall", "rss", "runtime", "extra", "source", "corrupt"])
def test_technical_failures_precede_prediction_decoding(complete_run, monkeypatch, corruption):
    f = complete_run
    done = report.read_json(f.run / "completed.json")
    if corruption == "incomplete":
        done["completed_fits"] = done["completed_fits"][:-1]
    elif corruption == "wall":
        done["wall_seconds"] = 3600.001
    elif corruption == "rss":
        done["process_lifetime_peak_rss_bytes"] = 6*1024**3+1
    elif corruption == "runtime":
        done["runtime"] = {"forged": True}
    elif corruption == "extra":
        (f.run / "partial.json").write_text("{}")
    elif corruption == "source":
        (f.root / "scripts/report_dialogue_conditional.py").write_text("changed")
    else:
        (f.run / "fits/mean-5301/weights.pt").write_bytes(b"changed")
    write_json(f.run / "completed.json", done)
    pin = report.digest(f.run / "completed.json")
    def forbidden(*args, **kwargs):
        pytest.fail("Predictions or references decoded before technical admission")
    monkeypatch.setattr(report, "load_npz", forbidden)
    out = f.root / "failure"
    with pytest.raises(ValueError):
        report.execute_report(f.run, pin, out)
    failed = report.read_json(out / "failed.json")
    assert failed["status"] == "failed" and failed["source_run_completed_sha256"] == pin
    assert not (out / "summary.json").exists() and not (out / "receipt.json").exists()


@pytest.mark.parametrize("corruption", ["journal_order", "journal_geometry", "journal_mass", "journal_missing",
                                        "configuration", "initialization", "counts", "fit_cost"])
def test_resealed_fit_corruption_rejected(complete_run, corruption):
    f = complete_run
    dest = f.run / "fits/candidate-5303"
    record = report.read_json(dest / "completed.json")
    if corruption.startswith("journal_"):
        path = dest / "updates.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        if corruption == "journal_order":
            events[0]["row_indices"].reverse()
        elif corruption == "journal_geometry":
            events[0]["actor_shapes"]["observation"][1] += 1
        elif corruption == "journal_mass":
            events[0]["normalization"]["max_abs_mass_error"] = .1
        else:
            events.pop()
        path.write_text("".join(json.dumps(e) + "\n" for e in events))
    elif corruption == "configuration":
        record["configuration"]["parameters"] += 1
    elif corruption == "initialization":
        record["initial_common_sha256"] = "a"*64
    elif corruption == "counts":
        record["counts"]["optimizer_returned"] -= 1
    else:
        record["evaluation"]["wall_seconds"] = record["wall_seconds"]+1
    write_json(dest / "completed.json", record)
    seal(dest)
    pin = seal(f.run)
    with pytest.raises(ValueError):
        report.execute_report(f.run, pin, f.root / "failure")
    assert (f.root / "failure/failed.json").exists()


def test_resealed_noncanonical_permutation_fails_before_journal(complete_run):
    f = complete_run
    path = f.run / "orders-5301.npy"
    orders = np.load(path, allow_pickle=False)
    orders[0] = orders[0][::-1]
    np.save(path, orders, allow_pickle=False)
    (f.frozen / path.name).write_bytes(path.read_bytes())
    f.plan["orders"]["5301"]["sha256"] = report.digest(path)
    for folder in (f.run, f.frozen):
        write_json(folder / "plan.json", f.plan)
    plan_pin = report.digest(f.run / "plan.json")
    frozen_done = report.read_json(f.frozen / "completed.json")
    frozen_done["plan_sha256"] = plan_pin
    write_json(f.frozen / "completed.json", frozen_done)
    seal(f.frozen)
    request = report.read_json(f.run / "started.json")
    request["request"]["plan_sha256"] = plan_pin
    write_json(f.run / "started.json", request)
    done = report.read_json(f.run / "completed.json")
    done["plan_sha256"] = plan_pin
    write_json(f.run / "completed.json", done)
    pin = seal(f.run)
    with pytest.raises(ValueError, match="permutations"):
        report.execute_report(f.run, pin, f.root / "failure")


@pytest.mark.parametrize("corruption", ["literal", "previous", "prediction"])
def test_resealed_analysis_payload_corruption_preserved(complete_run, corruption):
    f = complete_run
    if corruption == "prediction":
        path = f.run / "fits/slot-5302/dev-predictions.npz"
        arrays = packet(f.dev)
        arrays["log_probs"][0, 0] = np.nan
        np.savez(path, **arrays)
        seal(path.parent)
    else:
        arrays = references(f.dev)
        arrays[corruption + "_indices"][0] = 1
        np.savez(f.run / "references.npz", **arrays)
    pin = seal(f.run)
    with pytest.raises(ValueError):
        report.execute_report(f.run, pin, f.root / "failure")
    assert (f.root / "failure/failed.json").exists()


@pytest.mark.parametrize("corruption", ["join", "current_id", "previous_id", "adjacency", "value", "transition"])
def test_direct_ledger_join_and_previous_gold_validation(complete_run, corruption):
    f = complete_run
    rows = copy.deepcopy(f.ledger)
    row = rows[7]
    if corruption == "join":
        row["service"] = "foreign"
    elif corruption == "current_id":
        row["current_candidate_id"] = "value:foreign"
    elif corruption == "previous_id":
        row["previous_candidate_id"] = "value:foreign"
    elif corruption == "adjacency":
        row["time"] = 2
    elif corruption == "value":
        row["current_value_group"] = "false"
    else:
        row["derived_bin"] = "revision"
    (f.prepared / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        report.read_ledger(f.prepared, f.plan)


def test_failure_receipt_error_preserves_original_failure(complete_run, monkeypatch):
    f = complete_run
    original = Path.write_text
    def fail_only_receipt(path, *args, **kwargs):
        if path.name == "failed.json":
            raise OSError("synthetic full disk")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", fail_only_receipt)
    with pytest.raises(ValueError, match="External run completion identity") as caught:
        report.execute_report(f.run, "0"*64, f.root / "failure")
    if callable(getattr(caught.value, "add_note", None)):
        assert any("synthetic full disk" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("corruption", [None, "receipt", "binding", "wall"])
def test_inherited_cache_cost_is_authenticated_and_separate(tmp_path, monkeypatch, corruption):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "FEATURE_PARENTS", {"tokens": "parent"})
    parent = tmp_path / "parent"
    parent.mkdir()
    payload = parent / "tokens.npy"
    payload.write_bytes(b"Opaque inherited cache, never numerically decoded")
    done = {"status": "completed", "wall_seconds": 4., "files": {"tokens.npy": file_record(payload)}}
    if corruption == "wall":
        done["wall_seconds"] = -1
    write_json(parent / "completed.json", done)
    inputs = {str(p): file_record(p) for p in parent.iterdir()}
    if corruption == "receipt":
        (parent / "completed.json").write_text("changed")
    elif corruption == "binding":
        inputs[str(payload)]["bytes"] += 1
    if corruption:
        with pytest.raises(ValueError):
            report.inherited_costs({"authenticated_inputs": inputs})
    else:
        result = report.inherited_costs({"authenticated_inputs": inputs})["tokens"]
        assert result["recorded_wall_seconds"] == 4.
        assert result["recorded_manifest_payload_bytes"] == payload.stat().st_size
        assert result["completed_sha256"] == report.digest(parent / "completed.json")
