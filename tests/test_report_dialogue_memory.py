"""Saved-only synthetic metrics, corruption and complete21-fit artifact tests."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/report_dialogue_memory.py"
SPEC = importlib.util.spec_from_file_location("dialogue_report_test", PATH)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def packet():
    queries, dialogs = [], []
    labels = [0, 2, 2, 1, 0]
    bins = ["unmentioned_retention", "first_assignment", "assigned_retention", "revision", "clear"]
    for i, unseen in enumerate((False, True)):
        queries.append({"split": "dev", "candidate_ids": ["reserved:NOT_MENTIONED", "reserved:DONTCARE",
                                                           "value:None", "value:other"],
                        "candidates": [0, 1, 2, 3]})
        dialogs.append({"id": f"dialog-{i}", "turns": [0] * 5, "queries": [
            {"query": i, "time": t, "label": y, "bin": label_bin, "unseen": unseen, "dontcare": y == 1}
            for t, (y, label_bin) in enumerate(zip(labels, bins, strict=True))]})
    return {"queries": queries, "cohorts": {"dev": dialogs, "train": [copy.deepcopy(dialogs[0])]}}


def arrays():
    expected, counts = reporter.expected_records(packet())
    n = len(counts)
    p = np.zeros((n, 12), np.float32)
    p[:, :4] = .1
    p[np.arange(n), expected["labels"]] = .7
    return {**expected, "probabilities": p, "choice": p.argmax(-1)}, expected, counts


def test_handcomputed_metrics_brier_and_three_strata():
    a, expected, counts = arrays()
    reporter.validate_predictions(a, expected, counts)
    result = reporter.metrics(a)
    for panel, size in (("all", 10), ("seen", 5), ("unseen", 5)):
        row = result[panel]
        assert row["micro"]["count"] == size
        assert row["micro"]["accuracy"] == 1
        assert row["micro"]["nll"] == pytest.approx(-np.log(np.float32(.7)), abs=1e-7)
        assert row["micro"]["brier"] == pytest.approx(.12, abs=1e-7)
        assert row["revision"]["count"] == size // 5
        assert row["not_mentioned_count"] == 2 * size // 5
        assert row["dontcare_count"] == size // 5
    # More changed records must not dominate the three-stratum macro metric.
    a["probabilities"][[1, 3, 4, 6, 8, 9]] = 0
    a["probabilities"][[1, 3, 4, 6, 8, 9], 3] = 1
    a["choice"] = a["probabilities"].argmax(-1)
    result = reporter.metrics(a)["all"]
    assert result["micro"]["accuracy"] == .4
    assert result["macro_three"]["accuracy"] == pytest.approx(2 / 3)
    assert result["micro"]["nll"] is None
    assert result["micro"]["zero_target_probabilities"] == 6


@pytest.mark.parametrize("kind", ["nan", "negative", "sum", "padding", "choice", "label", "bin",
                                  "unseen", "dialogue", "time", "query", "duplicate", "missing", "foreign"])
def test_prediction_corruption_rejected(kind):
    a, expected, counts = arrays()
    if kind == "nan":
        a["probabilities"][0, 0] = np.nan
    elif kind == "negative":
        a["probabilities"][0, 0] = -.1
    elif kind == "sum":
        a["probabilities"][0, 0] = .2
    elif kind == "padding":
        a["probabilities"][0, 11] = .1
    elif kind == "duplicate":
        a = {k: np.concatenate((v[:-1], v[:1])) for k, v in a.items()}
    elif kind == "missing":
        a = {k: v[:-1] for k, v in a.items()}
    elif kind == "foreign":
        a["extra"] = np.zeros(10)
    else:
        field = "labels" if kind == "label" else kind
        a[field] = a[field].copy()
        if field in ("bin", "dialogue"):
            a[field][0] = "foreign"
        elif field == "unseen":
            a[field][0] = True
        else:
            a[field][0] += 1
    with pytest.raises(ValueError):
        reporter.validate_predictions(a, expected, counts)


@pytest.mark.parametrize("kind", ["none_id", "dontcare", "transition", "duplicate", "time"])
def test_packet_metadata_corruption(kind):
    p = packet()
    r = p["cohorts"]["dev"][0]["queries"]
    if kind == "none_id":
        p["queries"][0]["candidate_ids"][0] = "value:None"
    elif kind == "dontcare":
        r[3]["dontcare"] = False
    elif kind == "transition":
        r[3]["bin"] = "first_assignment"
    elif kind == "duplicate":
        r.append(copy.deepcopy(r[-1]))
    else:
        r[2]["time"] = 1
    with pytest.raises(ValueError):
        reporter.expected_records(p)


def gate_rows():
    rows = {}
    for method in reporter.METHODS:
        for seed in reporter.SEEDS:
            correct = 61 if method == "innovation_kalman" else 62 if method in ("gru", "attention", "carry") else 60
            row = {"strata": {s: {"count": 100, "correct": correct} for s in reporter.STRATA},
                   "micro": {"nll": .4}}
            rows[f"{method}-{seed}"] = {"metrics": {panel: copy.deepcopy(row) for panel in reporter.PANELS}}
    return rows


def test_exact_percentage_point_thresholds_and_all_eleven_checks():
    rows = gate_rows()
    gate = reporter.criteria(rows)
    assert gate["passed"] and gate["checks_passed"] == gate["checks_total"] == 11
    # Exactly one point over Kalman and one point below strongest conventional both pass.
    target = rows["innovation_kalman-1729"]["metrics"]["seen"]["strata"]
    target["changed"]["correct"] = 60
    gate = reporter.criteria(rows)
    assert not gate["passed"]
    assert not next(c for c in gate["checks"] if c["name"] == "seen/macro_gain_over_kalman_1pp")["passed"]


def test_good_mean_cannot_replace_two_strict_paired_wins():
    rows = gate_rows()
    for seed, count in zip(reporter.SEEDS, (59, 60, 64), strict=True):
        for s in reporter.STRATA:
            rows[f"innovation_kalman-{seed}"]["metrics"]["seen"]["strata"][s]["correct"] = count
    gate = reporter.criteria(rows)
    check = {c["name"]: c for c in gate["checks"]}
    assert check["seen/macro_gain_over_kalman_1pp"]["passed"]
    assert not check["seen/strict_macro_win_at_least_two_pairs"]["passed"]


def test_nll_missing_bins_and_zero_target_cannot_pass():
    rows = gate_rows()
    rows["innovation_kalman-1729"]["metrics"]["seen"]["micro"]["nll"] = None
    rows["kalman-2718"]["metrics"]["unseen"]["strata"]["changed"]["count"] = 0
    gate = reporter.criteria(rows)
    assert not gate["passed"]
    check = {c["name"]: c for c in gate["checks"]}
    assert not check["seen/micro_nll_nonworse_kalman"]["passed"]
    assert not check["unseen/macro_gain_over_kalman_1pp"]["passed"]


def test_deterministic_references_accuracy_only_and_invalid_choice():
    _, expected, counts = arrays()
    reference = {**expected, "pred_none": np.zeros(10, dtype=np.int64),
                 "pred_literal": np.full(10, 2, dtype=np.int64)}
    result = reporter.reference_metrics(reference, expected, counts)
    assert result["none"]["all"]["micro"]["accuracy"] == .4
    assert result["literal"]["all"]["micro"]["accuracy"] == .4
    assert set(result["literal"]["seen"]["micro"]) == {"count", "correct", "accuracy"}
    reference["pred_literal"][0] = 1
    with pytest.raises(ValueError, match="Reserved"):
        reporter.reference_metrics(reference, expected, counts)


@pytest.fixture
def tree(tmp_path):
    root, run, features = tmp_path / "root", tmp_path / "run", tmp_path / "features"
    root.mkdir()
    run.mkdir()
    features.mkdir()
    for name in reporter.SOURCES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("synthetic source bytes: " + name)
    data = packet()
    reporter.write(features / "packet.json", data)
    for name in ("encoder-plan.json", "encoder-source.py", "features.npy"):
        (features / name).write_bytes(b"synthetic bytes, not an encoder/model")
    feature_receipt = {"status": "completed", "cohorts": {"dev": {"queries": 10, "dialogues": 2}},
                       "wall_seconds": 1., "unique_texts": 4, "encoder_sequences": 4,
                       "encoder_tokens_with_special": 12,
                       "files": {p.name: reporter.sha(p) for p in features.iterdir()}}
    reporter.write(features / "completed.json", feature_receipt)
    plan = {"study": "dialogue-memory-v1", "runtime": {"python": "synthetic"},
            "config": {"methods": list(reporter.METHODS), "seeds": list(reporter.SEEDS),
                       "epochs": 12, "batch_size": 32}, "practical_checks": reporter.PRACTICAL_CHECKS,
            "packet_completed_sha256": reporter.sha(features / "completed.json"),
            "source_sha256": {n: reporter.sha(root / n) for n in reporter.SOURCES}}
    reporter.write(run / "plan.json", plan)
    predictions, _, _ = arrays()
    refs = {k: v for k, v in predictions.items() if k not in ("choice", "probabilities")}
    refs.update(pred_none=np.zeros(10, dtype=np.int64), pred_literal=np.full(10, 2, dtype=np.int64))
    np.savez_compressed(run / "references.npz", **refs)
    records = []
    for seed in reporter.SEEDS:
        for method in reporter.METHODS:
            folder = run / "fits" / f"{method}-{seed}"
            folder.mkdir(parents=True)
            (folder / "weights.pt").write_bytes(b"opaque hash-only checkpoint; never loaded")
            np.savez_compressed(folder / "dev-predictions.npz", **predictions)
            record = {"method": method, "seed": seed, "status": "completed", "epochs": 12,
                      "updates": 12, "training_queries": 60, "training_losses": [.5] * 12,
                      "parameters": 100, "parameters_with_final_gradient": 90,
                      "initial_tensors_sha256": "a" * 64, "train_wall_seconds": 1.,
                      "training_actor_shapes": {"real_turns": 60, "padded_turn_positions": 60,
                                                "padded_query_positions": 60, "padded_candidate_positions": 240},
                      "evaluation": {"queries": 10, "wall_seconds": .2},
                      "weights_sha256": reporter.sha(folder / "weights.pt"),
                      "predictions_sha256": reporter.sha(folder / "dev-predictions.npz")}
            reporter.write(folder / "completed.json", record)
            records.append(record)
    completion = {"status": "completed", "fit_count": 21, "fits": records, "wall_seconds": 30.,
                  "plan_sha256": reporter.sha(run / "plan.json"),
                  "references": {"file": "references.npz", "sha256": reporter.sha(run / "references.npz"),
                                 "queries": 10, "wall_seconds": .01}}
    reporter.write(run / "completed.json", completion)
    return root, run, features


def test_full_saved_only_report_retains_failed_gate_and_hashes(tree, tmp_path):
    root, run, features = tree
    out = tmp_path / "report"
    receipt = reporter.report(run, features, reporter.sha(run / "plan.json"),
                              reporter.sha(run / "completed.json"), out, root=root)
    assert receipt["status"] == "completed" and not receipt["continuation_passed"]
    assert receipt["fit_count"] == 21 and len(receipt["execution_members"]) == 64
    assert receipt["neural_calls"] == receipt["encoder_calls"] == 0
    assert (out / "accuracy.png").read_bytes().startswith(b"\x89PNG")
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["rows"]) == 21
    assert summary["families"]["carry"]["unseen"]["revision"]["count_per_fit"] == 1
    for name, record in receipt["files"].items():
        assert reporter.sha(out / name) == record["sha256"]
    with pytest.raises(FileExistsError):
        reporter.report(run, features, reporter.sha(run / "plan.json"),
                        reporter.sha(run / "completed.json"), out, root=root)


@pytest.mark.parametrize("kind", ["weight", "prediction", "source", "missing_fit", "fit_receipt",
                                  "execution_failure", "packet_failure", "criteria", "reference"])
def test_authentication_failure_preserved(tree, tmp_path, kind):
    root, run, features = tree
    first = run / "fits" / "current-1729"
    if kind in ("weight", "prediction"):
        (first / ("weights.pt" if kind == "weight" else "dev-predictions.npz")).write_bytes(b"changed")
    elif kind == "source":
        (root / "scripts/study_dialogue_memory.py").write_text("changed")
    elif kind == "missing_fit":
        (run / "fits" / "carry-3141").rename(run / "fits" / "wrong")
    elif kind == "fit_receipt":
        (first / "completed.json").write_text("{}")
    elif kind == "reference":
        (run / "references.npz").write_bytes(b"changed")
    elif kind.endswith("failure"):
        (run if kind == "execution_failure" else features).joinpath("failed.json").write_text("{}")
    else:
        plan = reporter.read(run / "plan.json")
        plan["practical_checks"]["macro_gain_over_kalman"] = 0
        (run / "plan.json").write_text(json.dumps(plan))
    out = tmp_path / "failed-report"
    with pytest.raises((ValueError, KeyError)):
        reporter.report(run, features, reporter.sha(run / "plan.json"),
                        reporter.sha(run / "completed.json"), out, root=root)
    assert (out / "failed.json").is_file() and not (out / "receipt.json").exists()
