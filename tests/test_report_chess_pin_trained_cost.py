"""Synthetic saved-artifact reports only; no model or chess-engine execution."""
import ast
import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/report_chess_pin_trained_cost.py"
spec = importlib.util.spec_from_file_location("saved_chess_cost_report", SCRIPT)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def complete(root, folder, plan_hash):
    path = root / folder
    write(path / "completed.json", {"status": "completed", "plan_sha256": plan_hash,
          "files": {f.relative_to(path).as_posix(): r.sha(f) for f in path.rglob("*")
                    if f.is_file() and f.name != "completed.json"}})


def rebind_cost(root):
    audit = r.read(root / r.AUDIT / "receipt.json")
    complete(root, r.EXECUTION, audit["plan_sha256"])
    audit["execution_receipt_sha256"] = r.sha(root / r.EXECUTION / "completed.json")
    audit["summary_sha256"] = r.sha(root / r.EXECUTION / "summary.json")
    audit["native_replay_sha256"] = r.sha(root / r.AUDIT / "native-replay.jsonl")
    write(root / r.AUDIT / "receipt.json", audit)
    digest = r.sha(root / r.AUDIT / "receipt.json")
    terminal = r.read(root / r.LAUNCHER / "terminal.json")
    terminal["audit_receipt_sha256"] = digest
    write(root / r.LAUNCHER / "terminal.json", terminal)
    return digest


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-cost")
    source = root / "scripts/chess_pin_trained_cost.py"
    source.parent.mkdir()
    source.write_text("# Synthetic frozen source, never executed.\n")
    qp = "evidence/chess-pin-quality-v3/protocol/plan.json"
    qe = "runs/chess-pin-quality-v3/execution"
    qa = "evidence/chess-pin-quality-v3/audit/receipt.json"
    quality_plan = {"prepared_unix": 1, "sources": {source.relative_to(root).as_posix(): r.sha(source)}}
    write(root / qp, quality_plan)
    quality_hash = r.sha(root / qp)
    quality = {"status": "completed", "plan_sha256": quality_hash,
               "gate_checks": [{"name": str(i), "passed": i < 2} for i in range(16)],
               "quality_gate_passed": False, "numerical_gate_passed": True, "continuation_passed": False}
    write(root / qe / "summary.json", quality)
    complete(root, qe, quality_hash)
    write(root / qa, {"status": "completed", "plan_sha256": quality_hash,
                     "summary_sha256": r.sha(root / qe / "summary.json"),
                     "execution_receipt_sha256": r.sha(root / qe / "completed.json"),
                     "gate_recomputed": quality["gate_checks"]})
    coverage = {"roots": [{"index": i, "root_pin_present": i % 2 == 0,
                           "pin_change_present": i % 3 == 0} for i in range(128)]}
    plan = {"protocol": {"version": "trained-pin-complete-native-cost-v2", "methods": list(r.METHODS),
            "seeds": list(r.SEEDS), "roots": 128, "repeats": 9, "timed_records": 31104,
            "native_audit_decisions": 3456, "warmups_per_method_seed": 2, "score_tolerance": 1e-5,
            "time_cap_seconds": 1800, "audit_time_cap_seconds": 1800,
            "limits": "Synthetic shared-host illustration; no performance evidence.",
            "timing": "Complete synthetic decision timings excluding file I/O."},
            "quality_plan_sha256": quality_hash,
            "quality_signature": {k: v for k, v in quality_plan.items() if k != "prepared_unix"},
            "quality_bindings": {"plan": qp, "execution": qe, "audit": qa},
            "panel": [{"id": f"synthetic-{i}"} for i in range(128)], "pin_input_coverage": coverage}
    write(root / r.PLAN, plan)
    plan_hash = r.sha(root / r.PLAN)

    def record(seed, index, repeat, method, timed):
        row = {"seed": seed, "root_index": index, "repeat": repeat, "method": method,
               "id": f"synthetic-{index}", "prediction": {"menus": ["a1a2", "a1b1"],
               "scores": [-1.0, 2.0], "choice": "a1b1"},
               "comparison": {"max_score_error": 0.0, "score_tolerance_passed": True, "choice_matches": True}}
        if timed:
            row["milliseconds"] = (r.METHODS.index(method) + 1) * (index + 1) * (1 + seed / 1000) + repeat / 10
        return row

    timed, warm, replay = [], [], []
    for si, seed in enumerate((97, 109, 127)):
        for method in r.METHODS:
            for repeat in (0, 1):
                warm.append(record(seed, 0, repeat, method, False))
        for index in range(128):
            for repeat in range(9):
                order = list(r.METHODS)
                offset = (si + index + repeat) % 9
                for method in order[offset:] + order[:offset]:
                    timed.append(record(seed, index, repeat, method, True))
            for method in r.METHODS:
                replay.append(record(seed, index, 0, method, False))
    lines(root / r.EXECUTION / "timings.jsonl", timed)
    lines(root / r.EXECUTION / "warmups.jsonl", warm)
    lines(root / r.AUDIT / "native-replay.jsonl", replay)
    evidence = {"quality_gate_passed": False, "quality_numerical_gate_passed": True,
                "quality_continuation_passed": False,
                "hashes": {p: r.sha(root / p) for p in (qa, qe + "/summary.json", qe + "/completed.json")}}
    strata = {}
    for field in ("root_pin_present", "pin_change_present"):
        for present in (False, True):
            indices = {v["index"] for v in coverage["roots"] if v[field] == present}
            strata[f"{field}={str(present).lower()}"] = r.aggregate([row for row in timed if row["root_index"] in indices])
    common = {"status": "completed", "plan_sha256": plan_hash, "quality_evidence": evidence,
              "pin_input_coverage": coverage, "new_training_updates": 0, "new_engine_calls": 0,
              "external_model_calls": 0, "limits": plan["protocol"]["limits"], "wall_seconds": 2.0}
    write(root / r.EXECUTION / "started.json", {"synthetic": True})
    write(root / r.EXECUTION / "summary.json", {**common, "timings": r.aggregate(timed), "strata": strata,
          "timed_checks": r.comparisons(timed), "warmup_checks": r.comparisons(warm),
          "cost_path_numerical_gate_passed": True})
    complete(root, r.EXECUTION, plan_hash)
    write(root / r.AUDIT / "receipt.json", {**common,
          "summary_sha256": r.sha(root / r.EXECUTION / "summary.json"),
          "execution_receipt_sha256": r.sha(root / r.EXECUTION / "completed.json"),
          "auditor_sha256": r.sha(source), "timing_records_checked": 31104, "warmup_records_checked": 54,
          "native_audit": r.comparisons(replay), "native_replay_sha256": r.sha(root / r.AUDIT / "native-replay.jsonl"),
          "cost_path_numerical_gate_passed": True})
    digest = r.sha(root / r.AUDIT / "receipt.json")
    write(root / r.LAUNCHER / "terminal.json", {"status": "completed", "retry": False,
          "plan_sha256": plan_hash, "audit_receipt_sha256": digest})
    for phase in ("run", "audit"):
        write(root / r.LAUNCHER / f"{phase}-terminal.json", {"status": "exited", "returncode": 0,
              "error": None, "wall_seconds": 3.0})
    return root, plan_hash, quality_hash, digest


@pytest.fixture
def saved(template, tmp_path, monkeypatch):
    original, plan_hash, quality_hash, digest = template
    root = tmp_path / "repo"
    shutil.copytree(original, root)
    monkeypatch.setattr(r, "PLAN_SHA", plan_hash)
    monkeypatch.setattr(r, "QUALITY_PLAN_SHA", quality_hash)
    return root, digest


def test_complete_failed_quality_is_reportable(saved):
    root, digest = saved
    report = r.authenticate(root, digest)
    assert report["coverage"] == {"methods": 9, "seeds": 3, "roots": 128, "repeats": 9,
                                  "timings": 31104, "warmups": 54, "native_audit": 3456}
    assert report["quality_checks_passed"] == 2
    assert report["quality_gate_passed"] is report["quality_continuation_passed"] is False
    assert report["cost_path_numerical_gate_passed"] is True
    assert report["model_calls"] == report["engine_calls"] == 0
    assert set(report["timings"]["median_complete_ms"]) == set(r.METHODS)


@pytest.mark.parametrize("member", [r.EXECUTION + "/timings.jsonl", r.AUDIT + "/native-replay.jsonl",
                                   r.PLAN, "scripts/chess_pin_trained_cost.py",
                                   "runs/chess-pin-quality-v3/execution/summary.json"])
def test_corrupted_bound_input_rejected(saved, member):
    root, digest = saved
    with (root / member).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        r.authenticate(root, digest)


def test_external_identity_precedes_payload_read(saved):
    root, _ = saved
    (root / r.EXECUTION / "summary.json").write_text("not JSON")
    with pytest.raises(ValueError, match="hash mismatch.*receipt"):
        r.authenticate(root, "0" * 64)


def test_missing_bound_artifact(saved):
    root, digest = saved
    (root / r.AUDIT / "native-replay.jsonl").unlink()
    with pytest.raises(ValueError, match="Missing or nonregular"):
        r.authenticate(root, digest)


def test_unlisted_primary_member(saved):
    root, digest = saved
    (root / r.EXECUTION / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="Exact execution member"):
        r.authenticate(root, digest)


def test_incomplete_lifecycle(saved):
    root, digest = saved
    value = r.read(root / r.LAUNCHER / "terminal.json")
    value["status"] = "running"
    write(root / r.LAUNCHER / "terminal.json", value)
    with pytest.raises(ValueError, match="Completed cost lifecycle"):
        r.authenticate(root, digest)


def test_rebound_missing_timing_fails_full_coverage(saved):
    root, _ = saved
    path = root / r.EXECUTION / "timings.jsonl"
    path.write_bytes(b"\n".join(path.read_bytes().splitlines()[1:]) + b"\n")
    digest = rebind_cost(root)
    with pytest.raises(ValueError, match="Complete ordered timings"):
        r.authenticate(root, digest)


def test_rebound_false_aggregate_rejected(saved):
    root, _ = saved
    summary = r.read(root / r.EXECUTION / "summary.json")
    summary["timings"]["median_complete_ms"]["joint"] += 1
    write(root / r.EXECUTION / "summary.json", summary)
    with pytest.raises(ValueError, match="Saved timing aggregate"):
        r.authenticate(root, rebind_cost(root))


def test_failed_cost_comparison_remains_reportable(saved):
    root, _ = saved
    path = root / r.AUDIT / "native-replay.jsonl"
    replay = [r.decode(line) for line in path.read_bytes().splitlines()]
    replay[0]["comparison"].update(max_score_error=.2, score_tolerance_passed=False)
    lines(path, replay)
    audit = r.read(root / r.AUDIT / "receipt.json")
    audit["native_audit"] = r.comparisons(replay)
    audit["cost_path_numerical_gate_passed"] = False
    write(root / r.AUDIT / "receipt.json", audit)
    report = r.authenticate(root, rebind_cost(root))
    assert report["cost_path_numerical_gate_passed"] is False
    assert report["native_audit"]["failed_records"] == 1


def test_pairwise_ratio_is_not_ratio_of_medians():
    rows = []
    for index, (wldn, joint) in enumerate(((1, 5), (2, 4), (100, 1))):
        for method in r.METHODS:
            rows.append({"seed": 97, "root_index": index, "repeat": 0, "method": method,
                         "milliseconds": joint if method == "joint" else wldn})
    result = r.aggregate(rows)
    assert result["median_paired_ratio_to_wldn"]["joint"] == 2
    shifted = copy.deepcopy(rows)
    for row in shifted:
        if row["method"] == "joint":
            row["milliseconds"] = (4, 1, 5)[row["root_index"]]
    changed = r.aggregate(shifted)
    assert changed["median_complete_ms"] == result["median_complete_ms"]
    assert changed["median_complete_ms"]["joint"] / changed["median_complete_ms"]["wldn"] == 2
    assert changed["median_paired_ratio_to_wldn"]["joint"] == .5
    assert set(changed["per_seed_median_ms"]) == {"97"}


def test_render_formats_failed_gate_and_exclusive_output(saved, tmp_path):
    root, digest = saved
    out = tmp_path / "figure"
    receipt = r.render(root, digest, out)
    assert set(receipt["files"]) == {"native-cost.png", "native-cost.svg", "native-cost.pdf", "report.json", "report.md"}
    assert all(r.sha(out / name) == value for name, value in receipt["files"].items())
    assert "failed (2/16 checks)" in (out / "report.md").read_text()
    assert "No Elo" in (out / "report.md").read_text()
    with pytest.raises(FileExistsError):
        r.render(root, digest, out)


def test_no_neural_or_engine_imports():
    imports = []
    for node in ast.walk(ast.parse(SCRIPT.read_text())):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    assert all(name.split(".")[0] in {"__future__", "argparse", "hashlib", "json", "math", "re",
                                       "statistics", "pathlib", "matplotlib"} for name in imports)


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_strict_json_rejects_ambiguity(raw):
    with pytest.raises(ValueError):
        r.decode(raw)
