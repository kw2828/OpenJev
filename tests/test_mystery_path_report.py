"""Synthetic dictionaries and byte fixtures only; no environment/model/RNG calls."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/report_mystery_path_qualification.py"
SPEC = importlib.util.spec_from_file_location("mystery_report", SOURCE)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def records(counts=None):
    counts = counts or {
        "full": [52, 51, 51, 51], "reference": [61, 61, 61, 61],
        "last16": [42, 42, 41, 41], "last32": [42, 42, 41, 41],
        "erase_on_failure": [42, 42, 41, 41],
    }
    result = []
    for layout in range(64):
        for order in range(4):
            for mode in report.MODES:
                result.append({
                    "layout_index": layout, "order_index": order, "mode": mode,
                    "seed": 1000 + layout, "success": layout < counts[mode][order],
                    "steps": 100, "falls": 2, "elapsed_seconds": 2.0,
                    "environment_seconds": 0.5, "policy_seconds": 0.2,
                    "parsing_seconds": 0.1, "update_seconds": 0.1,
                    "serialization_seconds": 0.1, "reset_seconds": 0.1,
                    "evaluator_seconds": 0.1, "peak_serialized_state_bytes": 100,
                    "peak_retained_python_bytes": 200,
                    "npz_path": f"episodes/{layout:03d}-{mode}-order{order}.npz",
                    "npz_sha256": "a" * 64,
                    "layout_sha256": hashlib.sha256(f"synthetic-layout-{layout}".encode()).hexdigest(),
                })
    return result


def set_success(rows, mode, order, layout, value):
    next(r for r in rows if (r["mode"], r["order_index"], r["layout_index"]) ==
         (mode, order, layout))["success"] = value


def check_named(summary, name):
    return next(c for c in summary["gate"]["checks"] if c["name"] == name)


def test_hand_computable_exact_boundaries_preserve_all_rows_and_reference():
    rows = records()
    original = copy.deepcopy(rows)
    result = report.evaluate_records(rows)
    assert result["gate"]["passed"]
    assert result["gate"]["checks_passed"] == result["gate"]["checks_total"] == 17
    assert result["modes"]["full"]["successes"] == 205
    assert result["modes"]["reference"]["successes"] == 244
    assert result["paired_success_tables"]["last16"]["pooled"] == {
        "pairs": 256, "both_succeed": 166, "full_only": 39,
        "comparator_only": 0, "neither_succeeds": 51, "full_minus_comparator_successes": 39,
    }
    assert result["modes"]["full"]["mean_steps_all_episodes"] == 100
    assert result["modes"]["full"]["mean_falls_all_episodes"] == 2
    assert result["modes"]["full"]["amortized_milliseconds_per_environment_step"] == 20
    assert len(result["records"]) == 1280
    assert rows == original


@pytest.mark.parametrize(("mode", "layout", "name"), [
    ("reference", 60, "reference_95_percent"), ("full", 51, "full_80_percent"),
])
def test_one_success_below_competence_threshold_fails(mode, layout, name):
    rows = records()
    set_success(rows, mode, 0, layout, False)
    summary = report.evaluate_records(rows)
    assert not check_named(summary, name)["passed"]
    assert not summary["gate"]["passed"]


def test_pooled_39_passes_and_38_fails_without_rounded_percent_comparison():
    rows = records()
    name = "full_minus_last32_15_percentage_points_pooled"
    assert check_named(report.evaluate_records(rows), name)["passed"]
    set_success(rows, "last32", 0, 42, True)
    result = report.evaluate_records(rows)
    assert check_named(result, name)["actual_success_count"] == 38
    assert not check_named(result, name)["passed"]
    assert not result["gate"]["passed"]


def test_one_priority_order_cannot_be_rescued_by_strong_pooled_margin():
    counts = {"full": [60] * 4, "reference": [64] * 4,
              "last16": [56, 30, 30, 30], "last32": [30] * 4, "erase_on_failure": [30] * 4}
    rows = records(counts)
    name = "full_minus_last16_5_percentage_points_order0"
    assert check_named(report.evaluate_records(rows), name)["passed"]
    set_success(rows, "last16", 0, 56, True)
    result = report.evaluate_records(rows)
    assert check_named(result, "full_minus_last16_15_percentage_points_pooled")["passed"]
    assert not check_named(result, name)["passed"]
    assert result["gate"]["decision"] == "CLOSE_THIS_RECIPE"


def test_discordant_pairs_and_all_episode_weighted_timing():
    rows = records()
    set_success(rows, "last16", 0, 0, False)
    set_success(rows, "last16", 0, 63, True)
    rows[0]["elapsed_seconds"] = 3
    rows[0]["steps"] = 50
    result = report.evaluate_records(rows)
    pairs = result["paired_success_tables"]["last16"]["pooled"]
    assert pairs["full_only"] == 40 and pairs["comparator_only"] == 1
    assert pairs["full_minus_comparator_successes"] == 39
    aggregate = result["modes"]["full"]
    assert aggregate["total_steps"] == 25550
    assert aggregate["timing_seconds"]["elapsed_seconds"] == 513
    assert aggregate["amortized_milliseconds_per_environment_step"] == 513000 / 25550


@pytest.mark.parametrize("mutation", [
    lambda rows: rows.pop(), lambda rows: rows.append(copy.deepcopy(rows[0])),
    lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
])
def test_missing_extra_and_duplicate_episodes_rejected(mutation):
    rows = records()
    mutation(rows)
    with pytest.raises(ValueError):
        report.evaluate_records(rows)


@pytest.mark.parametrize(("field", "value"), [
    ("mode", "foreign"), ("layout_index", 64), ("layout_index", True),
    ("order_index", 4), ("order_index", -1), ("seed", -1), ("seed", True),
    ("success", 1), ("success", "false"), ("steps", 0), ("steps", 129),
    ("steps", 1.0), ("steps", True), ("falls", -1), ("falls", 101),
    ("elapsed_seconds", 0), ("elapsed_seconds", float("nan")),
    ("policy_seconds", float("inf")), ("parsing_seconds", -0.1),
    ("reset_seconds", True), ("evaluator_seconds", "1"),
    ("peak_serialized_state_bytes", -1), ("peak_retained_python_bytes", 0.5),
    ("npz_path", "../escape.npz"), ("npz_path", "episodes/001-full-order0.npz"),
    ("npz_sha256", "not-a-sha"), ("layout_sha256", "A" * 64),
])
def test_malformed_scalar_and_foreign_identity_rejected(field, value):
    rows = records()
    rows[0][field] = value
    with pytest.raises(ValueError):
        report.evaluate_records(rows)


@pytest.mark.parametrize("field", ["layout_sha256", "seed"])
def test_pairing_requires_same_layout_and_seed_across_every_controller_and_order(field):
    rows = records()
    rows[5][field] = "f" * 64 if field == "layout_sha256" else 777
    with pytest.raises(ValueError, match="Mismatched paired"):
        report.evaluate_records(rows)


def test_different_layouts_cannot_reuse_seed_but_same_generated_path_is_not_excluded():
    rows = records()
    for row in rows:
        if row["layout_index"] == 1:
            row["seed"] = 1000
    with pytest.raises(ValueError, match="reuse the same seed"):
        report.evaluate_records(rows)
    rows = records()
    for row in rows:
        row["layout_sha256"] = "b" * 64
    result = report.evaluate_records(rows)
    assert result["episodes"] == 1280
    assert result["distinct_layout_sha256_count"] == 1


def test_record_schema_is_strict():
    rows = records()
    rows[0]["unscoped_metric"] = 1
    with pytest.raises(ValueError, match="fields"):
        report.evaluate_records(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False))


@pytest.fixture
def attempt(tmp_path):
    root = tmp_path / "repo"
    run = root / "attempt"
    run.mkdir(parents=True)
    rows = records()
    bindings = {}
    for field, relative in report.BINDING_FILES.items():
        path = root / relative
        write_json(path, {"scope": "synthetic test fixture only", "binding": field})
        bindings[field] = report.sha256(path)
    for row in rows:
        path = run / row["npz_path"]
        path.parent.mkdir(exist_ok=True)
        # Only byte authenticity is tested here. The fake audit receipt does not
        # represent a real replay or certify these deliberately non-NPZ bytes.
        path.write_bytes(b"synthetic payload, not an environment recording\n")
        row["npz_sha256"] = report.sha256(path)
    (run / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    digest = report.sha256(run / "episodes.jsonl")
    write_json(run / "completed.json", {"status": "complete", "episodes": 1280,
                                        "total_steps": 128000, "elapsed_seconds": 2600.0,
                                        "episodes_sha256": digest, **bindings})
    write_json(run / "audit/completed.json", {"status": "passed", "episodes": 1280,
                                               "episodes_sha256": digest, "steps": 128000})
    return root, run


@pytest.mark.parametrize("kind", ["missing", "corrupt", "symlink"])
def test_every_payload_must_exist_and_hash_match(attempt, kind):
    root, run = attempt
    path = run / "episodes/063-reference-order3.npz"
    if kind == "missing":
        path.unlink()
    elif kind == "corrupt":
        path.write_bytes(b"corrupt final member")
    else:
        path.unlink()
        path.symlink_to(run / "episodes/000-full-order0.npz")
    with pytest.raises((ValueError, FileNotFoundError)):
        report.authenticate_attempt(run, root=root)


@pytest.mark.parametrize(("file", "field", "value"), [
    ("completed.json", "status", "running"), ("audit/completed.json", "status", "failed"),
    ("completed.json", "episodes", True), ("audit/completed.json", "episodes", 1279),
    ("completed.json", "total_steps", 127999), ("completed.json", "episodes_sha256", "f" * 64),
    ("audit/completed.json", "episodes_sha256", "e" * 64),
    ("audit/completed.json", "steps", 127999),
    ("completed.json", "protocol_sha256", "d" * 64),
])
def test_completion_audit_and_bound_bytes_required(attempt, file, field, value):
    root, run = attempt
    path = run / file
    payload = json.loads(path.read_text())
    payload[field] = value
    write_json(path, payload)
    with pytest.raises(ValueError):
        report.authenticate_attempt(run, root=root)


def test_jsonl_bytes_are_bound_before_metrics_read(attempt):
    root, run = attempt
    with (run / "episodes.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="JSONL hash"):
        report.authenticate_attempt(run, root=root)


@pytest.mark.parametrize("directory", [".", "audit"])
@pytest.mark.parametrize("name", ["failed.json", "late-completion.json", "cleanup-error.json",
                                 "completion-before-cleanup-error.json"])
def test_failure_markers_override_success_receipts(attempt, directory, name):
    root, run = attempt
    write_json(run / directory / name, {"status": "failed"})
    with pytest.raises(ValueError, match="Retained failure marker"):
        report.authenticate_attempt(run, root=root)


@pytest.mark.parametrize("text", ['{"episodes":1280,"episodes":1280}', '{"x":NaN}'])
def test_ambiguous_or_nonfinite_json_is_rejected(text):
    with pytest.raises(ValueError):
        report._decode(text)


def test_failure_receipt_and_exclusive_output_preserve_original(attempt, tmp_path):
    root, run = attempt
    (run / "audit/completed.json").unlink()
    out = tmp_path / "report"
    with pytest.raises(FileNotFoundError):
        report.report_saved(run, out, root=root)
    failed = (out / "failed.json").read_bytes()
    assert json.loads(failed)["status"] == "failed"
    with pytest.raises(FileExistsError):
        report.report_saved(run, out, root=root)
    assert (out / "failed.json").read_bytes() == failed


def test_secondary_receipt_failure_does_not_mask_original_without_add_note(tmp_path, monkeypatch):
    class OriginalFailure(ValueError):
        add_note = None

    original = OriginalFailure("original admission error")

    def fail_authentication(*args, **kwargs):
        raise original

    def fail_preservation(*args, **kwargs):
        raise OSError("receipt storage unavailable")

    monkeypatch.setattr(report, "authenticate_attempt", fail_authentication)
    monkeypatch.setattr(report, "_write_json", fail_preservation)
    with pytest.raises(OriginalFailure) as caught:
        report.report_saved(tmp_path / "unused", tmp_path / "failed-report")
    assert caught.value is original


def test_synthetic_saved_output_report_and_plot_have_exact_members(attempt, tmp_path):
    root, run = attempt
    out = tmp_path / "synthetic-report-only"
    result = report.report_saved(run, out, root=root)
    assert {p.name for p in out.iterdir()} == {"summary.json", "report.md", "figure.png"}
    assert result["status"] == "complete" and result["gate"]["passed"]
    assert result["inputs"]["verified_episode_payloads"] == 1280
    assert (out / "figure.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert report.sha256(out / "figure.png") == result["report_files"]["figure.png"]["sha256"]
    assert json.loads((out / "summary.json").read_text()) == result
    prose = (out / "report.md").read_text()
    assert "17/17" in prose and "not an alternative simulator" in prose
    assert "Last 32 transitions" in prose and "Peak retained Python bytes" in prose


def test_failed_qualification_is_reported_as_failure_not_authentication_error():
    rows = records()
    set_success(rows, "reference", 0, 60, False)
    result = report.evaluate_records(rows)
    prose = report.markdown(result)
    assert "FAIL: 16/17" in prose and "CLOSE_THIS_RECIPE" in prose
    assert result["episodes"] == 1280 and len(result["records"]) == 1280
