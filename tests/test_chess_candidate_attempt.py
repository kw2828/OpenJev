"""Synthetic recovery checks; no saved failed journals or neural runs are read."""

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openjev.research import chess_candidate_attempt as attempt

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def studies():
    return load_script("chess_candidate_study"), load_script("chess_candidate_recovery")


@pytest.fixture(scope="module")
def frozen_schedule(studies):
    old, _ = studies
    schedule = old.schedule(97)
    return SimpleNamespace(computation=old.computation, schedule=lambda seed: schedule if seed == 97 else [])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def read(path):
    return json.loads(path.read_text())


def write_rows(path, values):
    path.write_text("".join(json.dumps(row) + "\n" for row in values))


@pytest.fixture
def evidence(tmp_path, frozen_schedule):
    """Minimal synthetic members accepted by the recovery-only admission rule."""
    root = tmp_path / "synthetic"
    old = root / attempt.OLD_EXECUTION
    source = root / "frozen-source.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Synthetic frozen source, not a model\n")
    write(root / attempt.OLD_PLAN, {"sources": {"frozen-source.py": attempt.sha(source)}})
    plan_hash = attempt.sha(root / attempt.OLD_PLAN)
    for member in attempt.EXPECTED_MEMBERS:
        write(old / member, {"synthetic_member": member})
    write(old / "started.json", {"status": "started", "plan_sha256": plan_hash})
    write(old / "failed.json", {
        "status": "failed", "plan_sha256": plan_hash,
        "error": "[Errno 32] Broken pipe", "wall_seconds": 123.25,
    })
    journal = []
    for step in frozen_schedule.schedule(97)[:128]:
        journal.append({
            "step": step["step"], "epoch": 1, "examples": 128,
            "root_depth": 4, "branch_depth": 2,
            "indices_sha256": hashlib.sha256(json.dumps(step["indices"]).encode()).hexdigest(),
            "loss": 1.0, "policy_ce": .9, "value_mse": .2, "gradient_norm": .8,
            **frozen_schedule.computation("action_only", 128, 256),
        })
    write_rows(old / "action_only-97/learning.jsonl", journal)
    write(old / "data/completed.json", {
        "status": "completed", "counts": {"dev": 2048, "shift": 2048},
        "teacher_calls": 4756, "requested_nodes": 9512000,
        "files": {name: attempt.sha(old / "data" / name)
                  for name in attempt.DATA_MEMBERS - {"completed.json"}},
    })
    return SimpleNamespace(root=root, old=old, journal=journal, study=frozen_schedule)


def test_admission_preserves_every_byte_and_discarded_compute(evidence):
    got = attempt.audit(evidence.root, evidence.study)
    assert got["files"] == attempt.inventory(evidence.old)
    assert set(got["files"]) == attempt.EXPECTED_MEMBERS
    assert got["original_plan_sha256"] == attempt.sha(evidence.root / attempt.OLD_PLAN)
    assert got["completed_optimizer_updates"] == 128
    assert got["discarded_training_examples"] == 16384
    assert got["discarded_computation"] == evidence.study.computation("action_only", 16384, 32768)
    assert got["completed_checkpoints"] == got["neural_evaluation_rows"] == 0
    assert got["reused_teacher_calls"] == 4756
    assert got["reused_requested_nodes"] == 9512000
    assert "No partial checkpoint, optimizer or RNG state reused" in got["policy"]
    assert "not a second fresh sample" in got["data_scope"]
    assert "Do not add reused data generation time again" in got["timing_scope"]


@pytest.mark.parametrize("extra", ["action_only-97/weights.pt", "action_only-97/dev.jsonl", "completed.json"])
def test_any_checkpoint_neural_output_or_completed_execution_rejects_recovery(evidence, extra):
    write(evidence.old / extra, {"synthetic": True})
    with pytest.raises(ValueError, match="membership changed or neural output"):
        attempt.audit(evidence.root, evidence.study)


def test_missing_expected_member_rejects_recovery(evidence):
    (evidence.old / "baselines.json").unlink()
    with pytest.raises(ValueError, match="membership"):
        attempt.audit(evidence.root, evidence.study)


def test_frozen_v1_source_change_rejects_recovery(evidence):
    (evidence.root / "frozen-source.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="Original frozen source"):
        attempt.audit(evidence.root, evidence.study)


@pytest.mark.parametrize(("filename", "key", "value"), [
    ("failed.json", "status", "completed"),
    ("failed.json", "error", "out of memory"),
    ("failed.json", "plan_sha256", "0" * 64),
    ("failed.json", "wall_seconds", -1),
    ("failed.json", "wall_seconds", float("nan")),
    ("started.json", "status", "failed"),
    ("started.json", "plan_sha256", "0" * 64),
])
def test_only_exact_terminal_failure_for_bound_plan_is_admitted(evidence, filename, key, value):
    path = evidence.old / filename
    row = read(path)
    row[key] = value
    write(path, row)
    with pytest.raises(ValueError, match="terminal pipe failure"):
        attempt.audit(evidence.root, evidence.study)


@pytest.mark.parametrize("size", [0, 127, 129])
def test_no_additional_or_missing_partial_updates_are_admitted(evidence, size):
    rows = evidence.journal[:size]
    if size == 129:
        rows = rows + [rows[-1]]
    write_rows(evidence.old / "action_only-97/learning.jsonl", rows)
    with pytest.raises(ValueError, match="Partial training budget"):
        attempt.audit(evidence.root, evidence.study)


@pytest.mark.parametrize(("key", "value"), [
    ("step", 2), ("epoch", 2), ("examples", 127),
    ("root_depth", 3), ("branch_depth", 3), ("indices_sha256", "0" * 64),
])
def test_partial_journal_must_follow_original_minibatches(evidence, key, value):
    evidence.journal[0][key] = value
    write_rows(evidence.old / "action_only-97/learning.jsonl", evidence.journal)
    with pytest.raises(ValueError, match="frozen schedule"):
        attempt.audit(evidence.root, evidence.study)


@pytest.mark.parametrize(("key", "value"), [
    ("loss", -1), ("policy_ce", float("nan")),
    ("value_mse", float("inf")), ("gradient_norm", -1),
])
def test_partial_statistics_must_be_finite_and_nonnegative(evidence, key, value):
    evidence.journal[0][key] = value
    write_rows(evidence.old / "action_only-97/learning.jsonl", evidence.journal)
    with pytest.raises(ValueError, match="training statistic"):
        attempt.audit(evidence.root, evidence.study)


def test_partial_loss_components_must_sum(evidence):
    evidence.journal[0]["loss"] = 1.5
    write_rows(evidence.old / "action_only-97/learning.jsonl", evidence.journal)
    with pytest.raises(ValueError, match="loss arithmetic"):
        attempt.audit(evidence.root, evidence.study)


@pytest.mark.parametrize("value", [127, True, 256.0])
def test_partial_candidate_count_is_an_integer_at_least_one_per_position(evidence, value):
    evidence.journal[0]["candidate_evaluations"] = value
    write_rows(evidence.old / "action_only-97/learning.jsonl", evidence.journal)
    with pytest.raises(ValueError, match="candidate count"):
        attempt.audit(evidence.root, evidence.study)


@pytest.mark.parametrize("key", [
    "root_core_iterations", "candidate_refinement_iterations", "successor_root_iterations",
    "total_core_iterations", "delta_convolutions", "successor_encoders",
])
def test_partial_computation_totals_cannot_change(evidence, key):
    evidence.journal[0][key] += 1
    write_rows(evidence.old / "action_only-97/learning.jsonl", evidence.journal)
    with pytest.raises(ValueError, match="computation arithmetic"):
        attempt.audit(evidence.root, evidence.study)


@pytest.mark.parametrize("change", ["status", "count", "missing", "hash", "bytes"])
def test_reused_data_completion_and_member_hashes_are_required(evidence, change):
    path = evidence.old / "data/completed.json"
    receipt = read(path)
    if change == "status":
        receipt["status"] = "failed"
    elif change == "count":
        receipt["counts"]["shift"] -= 1
    elif change == "missing":
        del receipt["files"]["dev.jsonl"]
    elif change == "hash":
        receipt["files"]["dev.jsonl"] = "0" * 64
    else:
        (evidence.old / "data/dev.jsonl").write_text("changed\n")
    write(path, receipt)
    with pytest.raises(ValueError, match="Reusable data receipt"):
        attempt.audit(evidence.root, evidence.study)


def test_inventory_rejects_symlink_instead_of_following_it(evidence):
    target = evidence.old / "baselines.json"
    target.unlink()
    target.symlink_to(evidence.old / "panels.json")
    with pytest.raises(ValueError, match="Nonregular"):
        attempt.audit(evidence.root, evidence.study)


def test_copy_is_byte_identical_and_destination_is_exclusive(evidence, tmp_path):
    recovery = attempt.audit(evidence.root, evidence.study)
    before = attempt.inventory(evidence.old)
    destination = tmp_path / "recovery-data"
    attempt.copy_data(evidence.root, destination, recovery)
    assert attempt.inventory(destination) == attempt.inventory(evidence.old / "data")
    assert attempt.inventory(evidence.old) == before
    with pytest.raises(FileExistsError):
        attempt.copy_data(evidence.root, destination, recovery)
    assert attempt.inventory(evidence.old) == before


def test_copy_rechecks_source_bytes_against_the_frozen_recovery(evidence, tmp_path):
    recovery = attempt.audit(evidence.root, evidence.study)
    (evidence.old / "data/dev.jsonl").write_text("changed after recovery freeze\n")
    destination = tmp_path / "recovery-data"
    with pytest.raises(ValueError, match="changed before copy"):
        attempt.copy_data(evidence.root, destination, recovery)
    assert not destination.exists()


def test_copy_checks_destination_bytes_after_transfer(evidence, tmp_path, monkeypatch):
    recovery = attempt.audit(evidence.root, evidence.study)
    actual_copytree = attempt.shutil.copytree

    def corrupted_copy(source, destination):
        result = actual_copytree(source, destination)
        (Path(destination) / "dev.jsonl").write_text("corrupted transfer\n")
        return result

    monkeypatch.setattr(attempt.shutil, "copytree", corrupted_copy)
    with pytest.raises(ValueError, match="Copied panels differ"):
        attempt.copy_data(evidence.root, tmp_path / "recovery-data", recovery)


def test_v1_frozen_sources_still_match_the_prospective_plan():
    # The original plan's source manifest is read, never its failed run or model records.
    plan = read(ROOT / attempt.OLD_PLAN)
    assert "scripts/chess_candidate_study.py" in plan["sources"]
    assert all(attempt.sha(ROOT / name) == digest for name, digest in plan["sources"].items())


def test_recovery_preserves_scientific_protocol_schedule_and_prior_sources(studies):
    old, recovery = studies
    assert set(old.PROTOCOL) == set(recovery.PROTOCOL)
    changed = {key for key in old.PROTOCOL if old.PROTOCOL[key] != recovery.PROTOCOL[key]}
    assert changed == {"version", "selection"}
    assert recovery.PROTOCOL["version"] == "chess-candidate-v2"
    assert "fresh initialization" in recovery.PROTOCOL["selection"]
    assert "no best epoch/seed, further retries" in recovery.PROTOCOL["selection"]
    assert recovery.ENGINE == old.ENGINE
    assert recovery.PREFLIGHTS == old.PREFLIGHTS
    assert set(recovery.SOURCES) - set(old.SOURCES) == {
        "scripts/chess_candidate_recovery.py", "src/openjev/research/chess_candidate_attempt.py",
        "tests/test_chess_candidate_attempt.py",
    }
    assert set(old.SOURCES) <= set(recovery.SOURCES)
    assert old.configs() == recovery.configs()
    for seed in old.PROTOCOL["seeds"]:
        assert old.schedule(seed) == recovery.schedule(seed)


def functions(module):
    tree = ast.parse(Path(module.__file__).read_text())
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def result_dict(function, name):
    return next(node.value for node in ast.walk(function)
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))


def strip_dict_keys(node, removed):
    items = [(key, value) for key, value in zip(node.keys, node.values, strict=True)
             if not (isinstance(key, ast.Constant) and key.value in removed)]
    node.keys, node.values = map(list, zip(*items, strict=True))


def test_recovery_has_no_unannounced_model_training_evaluation_or_gate_code_changes(studies):
    old, recovery = map(functions, studies)
    assert old.keys() == recovery.keys()
    permitted = {"signature", "run", "report"}
    for name in old.keys() - permitted:
        assert ast.dump(old[name]) == ast.dump(recovery[name]), name

    # Normalize the three declared recovery changes and demand exact remaining AST parity.
    signature = copy.deepcopy(recovery["signature"])
    returned = next(node.value for node in ast.walk(signature) if isinstance(node, ast.Return))
    strip_dict_keys(returned, {"recovery"})
    assert ast.dump(old["signature"]) == ast.dump(signature)

    run = copy.deepcopy(recovery["run"])
    copies = [node for node in ast.walk(run) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and node.func.attr == "copy_data"]
    assert len(copies) == 1
    expected_copy = ast.parse('attempt.copy_data(ROOT, out/"data", frozen["recovery"])').body[0].value
    assert ast.dump(copies[0]) == ast.dump(expected_copy)
    expected_generation = ast.parse('data.generate(out/"data", ROOT/ENGINE, exclusions["states"])').body[0].value
    copies[0].func, copies[0].args, copies[0].keywords = (
        expected_generation.func, expected_generation.args, expected_generation.keywords,
    )
    assert ast.dump(old["run"]) == ast.dump(run)

    report = copy.deepcopy(recovery["report"])
    strip_dict_keys(result_dict(report, "result"), {"recovery", "attempt_accounting"})
    assert ast.dump(old["report"]) == ast.dump(report)


def test_report_adds_discarded_cost_without_double_counting_data(studies):
    _, recovery = studies
    values = result_dict(functions(recovery)["report"], "result")
    accounting = next(value for key, value in zip(values.keys, values.values, strict=True)
                      if isinstance(key, ast.Constant) and key.value == "attempt_accounting")
    expression = ast.fix_missing_locations(ast.Expression(accounting))
    got = eval(compile(expression, "synthetic-attempt-accounting", "eval"), {}, {
        "plan": {"recovery": {"failure": {"wall_seconds": 123.25}, "completed_optimizer_updates": 128}},
        "complete": {"wall_seconds": 1000.0},
        "costs": {str(i): {"updates": 1536} for i in range(12)},
    })
    assert got == {
        "failed_wall_seconds": 123.25, "successful_execution_wall_seconds": 1000.0,
        "total_attempt_wall_seconds": 1123.25, "final_fit_updates": 18432,
        "discarded_updates": 128, "total_optimizer_updates": 18560,
        "data_generated_once": True,
    }
