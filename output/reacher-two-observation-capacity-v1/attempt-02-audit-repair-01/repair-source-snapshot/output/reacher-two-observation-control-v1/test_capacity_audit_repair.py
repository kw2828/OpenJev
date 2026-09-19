"""Pure schema/AST and byte-only orchestration tests; no numerical audit."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("capacity_audit_repair_test_target", Path(__file__).with_name("capacity_audit_repair.py"))
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


@pytest.mark.parametrize("arm,key", [("residual_gru", "model_work"), ("cached_gru", "model_work"), ("two_observation_gru", "aggregate_model_work")])
def test_exact_class_aggregate_with_no_alias(arm, key):
    state = {key: {"assimilate": 3200, "nested": {"transition": 8749056}}, "other": "retained"}
    before = copy.deepcopy(state)
    result = repair.aggregate_model_work(state, arm)
    assert result == state[key]
    result["nested"]["transition"] = -1
    assert state == before


@pytest.mark.parametrize("state,arm", [({}, "residual_gru"), ({"aggregate_model_work": {"x": 1}}, "residual_gru"),
    ({"model_work": {"x": 1}}, "two_observation_gru"), ({"model_work": {"x": 1}, "aggregate_model_work": {"x": 1}}, "cached_gru"),
    ({"model_work": []}, "cached_gru"), ({"model_work": {}}, "cached_gru"), ({"model_work": {"x": 1}}, "unknown"), ([], "residual_gru")])
def test_wrong_or_ambiguous_schema_rejected(state, arm):
    with pytest.raises(ValueError): repair.aggregate_model_work(state, arm)


def test_original_sources_pinned_and_import_does_not_invoke_backend():
    original = repair.load_original(repair.ROOT)
    assert original.sha(repair.ROOT / repair.ORIGINAL_HELPER) == repair.ORIGINAL_HELPER_SHA
    assert original.sha(repair.ROOT / repair.ORIGINAL_TESTS) == repair.ORIGINAL_TESTS_SHA
    assert callable(original._audit_measurements) and callable(original.projection)
    assert not hasattr(original, "aggregate_model_work")


def test_saved_audit_body_exactly_preserved_except_declared_adapter():
    old_tree = ast.parse((repair.ROOT / repair.ORIGINAL_HELPER).read_text())
    new_tree = ast.parse(Path(repair.__file__).read_text())
    old = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == "_audit_measurements")
    new = copy.deepcopy(next(node for node in new_tree.body if isinstance(node, ast.FunctionDef) and node.name == "audit_measurements"))
    assert len(new.body[1:4]) == 3 and all(isinstance(node, ast.Assign) for node in new.body[1:4])
    new.body = [copy.deepcopy(old.body[0]), *new.body[4:]]
    new.args.args = new.args.args[1:]
    new.name = old.name
    replacements = 0
    for node in ast.walk(new):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "aggregate_model_work":
            assert ast.unparse(node.value) == "aggregate_model_work(state, row['arm'])"
            node.value = ast.parse('state["aggregate_model_work"]', mode="eval").body
            replacements += 1
    assert replacements == 1
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    original = repair.load_original(repair.ROOT)
    adapted = SimpleNamespace(**vars(original))
    for name in (repair.HELPER, repair.TEST_HELPER, "fixture-source.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("byte-only fixture\n")
    source = {"fixture-source.py": repair.sha(tmp_path / "fixture-source.py")}
    binding = {"scientific_sources": source, "sources": source, "runtime": {"fake": True}}
    adapted.bind_rehearsal = lambda *args: binding
    monkeypatch.setattr(repair, "load_original", lambda root: adapted)
    attempt = tmp_path / "measurement"
    attempt.mkdir()
    rows = [{**row, "whole_call_seconds": 1e-8} for row in original.row_manifest()]
    fit = {"updates": 24, "wall_seconds": 5e-8, "constructor_seconds": 1e-8,
           "update_call_seconds": 2e-8, "log_validation_flush_seconds": 1e-8}
    original.write(attempt / "execution/measurement.json", {"rows": rows, "actual_training_updates": 24,
        "phases": {"fit_seconds": 5e-8, "full48_trainer_setup_seconds": 1e-8}})
    done = {"status": "completed", "engineering": True, "namespace": original.NAMESPACE, "scope": original.SCOPE,
        "audit_cap_seconds": 100, "cap_seconds": 100, "wall_seconds": 1., "qualification_ready": False,
        "row_count": 14, "new_fits": 1, "optimizer_updates": 24, "full48_setup_only_count": 1,
        "native_control_transitions": 44800, "rehearsal": {}, "source_sha256": source, "profile_source_sha256": source,
        "runtime": binding["runtime"], "hashing_seconds": 1e-8, "files": original.inventory(attempt, float("inf"))}
    original.write(attempt / "completed.json", done)
    digest = original.sha(attempt / "completed.json")
    monkeypatch.setattr(repair, "COMPLETED_SHA", digest)
    prior = tmp_path / "failed-audit"
    original.write(prior / "started.json", {"scope": original.SCOPE, "completed_sha256": digest})
    original.write(prior / "failed.json", {"status": "failed", "scope": original.SCOPE, "exception_type": "KeyError",
        "error": "KeyError('aggregate_model_work')", "automatic_retry": False})
    failed = {"path": "failed-audit/failed.json", "sha256": original.sha(prior / "failed.json")}
    calls = []
    def audit(*args):
        _, _, _, out, _, _, _ = args
        from openjev.research import reacher_two_observation_protocol as protocol
        calls.append("fake_audit")
        original.write(out / "training-audit.json", {"fixture": True})
        return {"rows": [{**row, "audit_seconds": 1e-8} for row in original.row_manifest()],
            "training_audit_seconds": 1e-8, "fit": fit, "settings": protocol.settings(engineering=True),
            "new_model_calls": 0, "new_optimizer_steps": 0}
    monkeypatch.setattr(repair, "audit_measurements", audit)
    return SimpleNamespace(root=tmp_path, attempt=attempt, digest=digest, failed=failed,
        original=original, adapted=adapted, calls=calls, out=tmp_path / "repaired-audit")


def run(case):
    return repair.audit_saved(case.attempt, case.digest, case.out, root=case.root, failed_audit=case.failed, authorize_saved_audit=True)


def test_fake_success_preserves_originals_and_discloses_repair(fake):
    before = fake.original.inventory(fake.attempt, float("inf"))
    failed_before = fake.original.inventory(fake.root / "failed-audit", float("inf"))
    assert run(fake) == fake.out
    done = fake.original.read(fake.out / "completed.json")
    assert done["profile_source_sha256"] == fake.original.read(fake.attempt / "completed.json")["profile_source_sha256"]
    assert set(done["audit_repair"]["repair_source_sha256"]) == {repair.HELPER, repair.TEST_HELPER}
    assert done["audit_repair"]["measurement_rerun"] is False and done["qualification_ready"] is False
    fake.original.verify_members(fake.out, done["files"], float("inf"), extra=("completed.json",))
    assert fake.original.inventory(fake.attempt, float("inf")) == before
    assert fake.original.inventory(fake.root / "failed-audit", float("inf")) == failed_before
    assert fake.calls == ["fake_audit"]
    with pytest.raises(FileExistsError): run(fake)
    assert fake.calls == ["fake_audit"]


def test_no_authorization_no_output_or_source_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(repair, "load_original", lambda *args: pytest.fail("No import"))
    with pytest.raises(ValueError, match="authorization"):
        repair.audit_saved(tmp_path / "missing", "a" * 64, tmp_path / "out", root=tmp_path, failed_audit={})
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("bad", ["completion", "measurement", "failed_receipt", "extra_member", "wrong_failure"])
def test_corruption_stops_before_replay(fake, bad):
    if bad == "completion": fake.digest = "f" * 64
    elif bad == "measurement": (fake.attempt / "execution/measurement.json").write_text("changed")
    elif bad == "failed_receipt": fake.failed["sha256"] = "f" * 64
    elif bad == "extra_member": (fake.attempt / "unexpected.bin").write_bytes(b"extra")
    else:
        path = fake.root / fake.failed["path"]
        value = json.loads(path.read_text()); value["exception_type"] = "OtherError"
        path.unlink(); repair.write(path, value); fake.failed["sha256"] = repair.sha(path)
    with pytest.raises(ValueError): run(fake)
    assert not fake.calls and (fake.out / "failed.json").exists()
    assert not (fake.out / "completed.json").exists()


def test_post_completion_cap_failure_invalidates_only_new_audit(fake):
    check = fake.adapted.check
    def capped(deadline):
        if (fake.out / "completed.json").exists(): raise TimeoutError("post-completion audit cap")
        check(deadline)
    fake.adapted.check = capped
    with pytest.raises(TimeoutError, match="post-completion"): run(fake)
    assert (fake.out / "invalid-completion.json").exists() and (fake.out / "failed.json").exists()
    assert fake.original.sha(fake.attempt / "completed.json") == fake.digest


def test_new_output_cannot_be_inside_measurement(fake):
    fake.out = fake.attempt / "nested-audit"
    with pytest.raises(ValueError, match="separate audit"):
        run(fake)
    assert not fake.out.exists() and not fake.calls


def test_new_output_cannot_mutate_failed_audit_tree(fake):
    parent = fake.root / "failed-audit"
    before = fake.original.inventory(parent, float("inf"))
    fake.out = parent / "repair"
    with pytest.raises(ValueError, match="within the failed audit"):
        run(fake)
    assert fake.original.inventory(parent, float("inf")) == before
    assert not fake.calls
