"""Synthetic operator, capture, and failure tests; no corpus or scientific fit."""
import copy
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    spec = importlib.util.spec_from_file_location("qualify_dialogue_evidence_tested", ROOT / "scripts/qualify_dialogue_evidence.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
finally:
    sys.path.pop(0)


@pytest.fixture(autouse=True)
def isolated():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        yield
    torch.set_num_threads(threads)


def fixture():
    features = np.arange(12 * 384, dtype=np.float32).reshape(12, 384) / 10000
    queries = [{"text": 4, "candidates": [6, 7, 8], "candidate_values": [None, None, "red"]}, {"text": 5, "candidates": [6, 7, 8, 9], "candidate_values": [None, None, "true", "false"]}]
    ds = [
        {"id": "a", "turns": [0, 1, 2], "queries": [
            {"query": 1, "time": 0, "label": 2, "bin": "first_assignment", "unseen": False},
            {"query": 0, "time": 1, "label": 1, "bin": "first_assignment", "unseen": True},
            {"query": 1, "time": 2, "label": 2, "bin": "assigned_retention", "unseen": False}],
         "layout": {"id": "a", "query_ids": [0, 1], "shape": [3, 2, 4, 10], "offset": 0}},
        {"id": "longer-id", "turns": [3], "queries": [
            {"query": 0, "time": 0, "label": 0, "bin": "unmentioned_retention", "unseen": True}],
         "layout": {"id": "longer-id", "query_ids": [0], "shape": [1, 1, 3, 10], "offset": 240}},
    ]
    return ds, queries, features, np.zeros(270, np.float32)


def progress():
    return {"pieces": [], "model_forward_calls": 0, "model_forward_returned": 0,
            "advance_attempted": 0, "advance_returned": 0, "returned_rows": 0,
            "matched_rows": 0, "actor_work": Counter(), "active_dialogues": []}


def small_model(method):
    return tool.TraceMemory(method, projection_dim=4, hidden_dim=4)


@pytest.mark.parametrize("method", ["scalar", "selective"])
def test_capture_parent_parity_and_original_prediction_convention(method):
    plain = tool.DialogueCopyMemory(method, projection_dim=4, hidden_dim=4)
    traced = small_model(method)
    traced.load_state_dict(plain.state_dict())
    actor, _, _ = tool.old.make_batch(*fixture())
    traced.reset_trace()
    with torch.inference_mode():
        wanted, observed = plain(*actor), traced(*actor)
    assert torch.equal(wanted, observed)
    assert traced.advance_attempted == traced.advance_returned == 3
    for t, row in enumerate(traced.trace):
        np.testing.assert_array_equal(row["result"], wanted[:, t].softmax(-1).numpy())
        if t == 0:
            np.testing.assert_array_equal(row["old_b"][..., 0], np.ones((2, 2)))
    retained = traced.trace[0]["old_b"].copy()
    with torch.no_grad():
        for parameter in traced.parameters():
            parameter.add_(1)
    np.testing.assert_array_equal(retained, traced.trace[0]["old_b"])


@pytest.mark.parametrize("method", ["scalar", "selective"])
def test_saved_original_order_and_batch_work_are_replayed(method, tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "ROWS", 4)
    # Separate batches deliberately have different Unicode widths.
    monkeypatch.setattr(tool.old, "CONFIG", {**tool.old.CONFIG, "batch_size": 1})
    plain = tool.DialogueCopyMemory(method, projection_dim=4, hidden_dim=4)
    model = small_model(method)
    model.load_state_dict(plain.state_dict())
    saved = tmp_path / "saved.npz"
    tool.old.evaluate(plain, *fixture(), saved)
    with np.load(saved, allow_pickle=False) as data:
        expected = {k: data[k] for k in data.files}
    work = progress()
    arrays, agreement = tool.replay_batches(model, *fixture(), expected, work, lambda: None)
    assert agreement == {"bitwise_equal": True, "max_probability_abs_difference": 0., "exact_metadata_and_choices": True}
    assert work["model_forward_calls"] == work["model_forward_returned"] == 2
    assert work["advance_returned"] == 4 and work["matched_rows"] == 4
    assert work["actor_work"]["real_question_steps"] == 7
    assert arrays["dialogue"].tolist() == ["a", "a", "a", "longer-id"]
    assert arrays["query"].tolist() == [1, 0, 1, 0]
    assert arrays["candidate_count"].tolist() == [4, 3, 4, 3]
    assert np.count_nonzero(arrays["old_b"][:, 4:]) == 0
    assert np.count_nonzero(arrays["writer"][:, 4:]) == 0
    assert np.isfinite(arrays["departure_mass"]).all()


def manual():
    y = np.array([1, 2, 2, 1, 2])
    choice = np.array([1, 1, 2, 0, 1])
    old_b = np.zeros((5, 12), np.float32)
    old_b[:, :3] = [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]
    writer = np.zeros_like(old_b)
    writer[:, :3] = [[.1, .8, .1], [.1, .2, .7], [.1, .8, .1], [.1, .2, .7], [.8, .1, .1]]
    p = np.zeros_like(old_b)
    p[np.arange(5), choice] = 1
    return {"old_b": old_b, "writer": writer, "departure_mass": np.full(5, .2),
            "candidate_count": np.full(5, 3), "labels": y, "choice": choice,
            "probabilities": p, "result": p.copy(), "dialogue": np.full(5, "x"),
            "query": np.zeros(5, np.int64), "time": np.arange(5), "unseen": np.ones(5, bool),
            "bin": np.array(["first_assignment", "revision", "assigned_retention", "revision", "revision"]),
            "boolean_slot": np.zeros(5, bool), "user_unique": np.array([1, 2, -1, -1, 2]), "system_unique": np.full(5, -1)}


def test_counterfactual_product_uses_saved_factors_without_trajectory_feedback():
    arrays = manual()
    writer, product = tool.factors(arrays)
    # At row1, pi=(1/15,13/15,1/15); multiply by (.1,.2,.7).
    expected = np.array([1, 26, 7], float) / 34
    np.testing.assert_allclose(product[1, :3], expected, atol=1e-8)
    assert writer[1].argmax() == 2 and product[1].argmax() == 1
    altered = copy.deepcopy(arrays)
    altered["old_b"][0] = altered["writer"][0]
    _, second = tool.factors(altered)
    np.testing.assert_array_equal(product[1:], second[1:])


def test_exact_writer_partition_prior_correct_adjacency_and_repair_break():
    result = tool.diagnostic(manual())
    row = result["strata"]["unseen/revision"]
    assert row["count"] == row["wrong"] == 3
    assert row["wrong_writer_partition"] == {"writer_gold": 1, "writer_previous_gold": 1, "writer_other": 1}
    assert row["one_step_counterfactual"]["force_write"]["repair_wrong"] == 1
    assert row["one_step_counterfactual"]["force_write"]["break_correct"] == 0
    ret = result["strata"]["unseen/retention"]
    assert ret["one_step_counterfactual"]["force_write"]["break_correct"] == 1
    prior = result["strata"]["unseen/revision/prior_correct_adjacent"]
    assert prior["count"] == 2  # t1 and t3, not t4 whose previous prediction was wrong.
    stats = row["factor_statistics"]["wrong"]["old_b_max"]
    assert stats == {"count": 3, "sum": 3., "quantiles": [1., 1., 1., 1., 1.]}
    assert result["strata"]["seen/all"]["factor_statistics"]["all"]["writer_max"]["quantiles"] is None


@pytest.mark.parametrize("field", ["labels", "choice", "query", "time", "dialogue", "bin", "unseen"])
def test_replay_refuses_metadata_or_choice_mismatch(field):
    arrays = manual()
    expected = {k: v.copy() for k, v in arrays.items() if k in tool.FIELDS}
    if expected[field].dtype.kind == "U":
        expected[field][0] = "z"
    elif field == "unseen":
        expected[field][0] = False
    else:
        expected[field][0] += 1
    with pytest.raises(ValueError, match="metadata/choice"):
        tool.compare_replay(arrays, expected)


def test_probability_tolerance_and_bitwise_report():
    arrays = manual()
    expected = {k: arrays[k].copy() for k in tool.FIELDS}
    expected["probabilities"][0, 0] += np.float32(5e-7)
    result = tool.compare_replay(arrays, expected)
    assert not result["bitwise_equal"]
    expected["probabilities"][0, 0] = np.float32(2e-6)
    with pytest.raises(ValueError, match="probability mismatch"):
        tool.compare_replay(arrays, expected)


@pytest.mark.parametrize("mutation", ["nan", "negative", "padding", "mass", "zero"])
def test_invalid_factors_cannot_silently_normalize(mutation):
    arrays = manual()
    if mutation == "nan":
        arrays["writer"][0, 0] = np.nan
    elif mutation == "negative":
        arrays["old_b"][0, 0] = -1
    elif mutation == "padding":
        arrays["old_b"][0, 11] = .1
    elif mutation == "mass":
        arrays["departure_mass"][0] = 1.1
    else:
        arrays["writer"][0] = 0
    with pytest.raises(ValueError, match="factors"):
        tool.factors(arrays)


def test_replay_failure_keeps_returned_mismatching_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "ROWS", 4)
    model = small_model("scalar")
    saved = tmp_path / "original.npz"
    plain = tool.DialogueCopyMemory("scalar", projection_dim=4, hidden_dim=4)
    plain.load_state_dict(model.state_dict())
    tool.old.evaluate(plain, *fixture(), saved)
    with np.load(saved, allow_pickle=False) as data:
        expected = {k: data[k] for k in data.files}
    expected["probabilities"][0, 0] += .01
    work = progress()
    with pytest.raises(ValueError, match="probability mismatch") as caught:
        tool.replay_batches(model, *fixture(), expected, work, lambda: None)
    assert work["returned_rows"] == 4 and work["matched_rows"] == 0
    tool.preserve_failure(tmp_path, caught.value, work, model, [], tool.time.perf_counter())
    with np.load(tmp_path / "partial.npz", allow_pickle=False) as saved:
        assert len(saved["labels"]) == 4
    assert (tmp_path / "partial-batch.npz").exists()
    assert tool.read(tmp_path / "failed.json")["active_progress"]["model_forward_returned"] == 1


def test_restoration_strict_configuration_and_rng_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(tool.old, "CONFIG", {**tool.old.CONFIG, "projection_dim": 4, "hidden_dim": 4})
    plain = tool.DialogueCopyMemory("scalar", projection_dim=4, hidden_dim=4)
    torch.save(plain.state_dict(), tmp_path / "weights.pt")
    row = {"method": "scalar", "weights_sha256": tool.sha(tmp_path / "weights.pt"), "configuration": plain.configuration()}
    before = torch.get_rng_state().clone()
    restored = tool.restore_model(tmp_path, row)
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in plain.state_dict().items())
    row["configuration"]["method"] = "selective"
    with pytest.raises(ValueError, match="configuration"):
        tool.restore_model(tmp_path, row)
    assert torch.equal(before, torch.get_rng_state())


def fake_run(monkeypatch, tmp_path):
    plan_path = tmp_path / "plan.json"
    tool.write(plan_path, {"wall_cap_seconds": 100, "source_sha256": {}})
    args = SimpleNamespace(plan=plan_path, plan_sha256=tool.sha(plan_path), out=tmp_path / "replay")
    monkeypatch.setattr(tool, "FITS", ["scalar-4101"])
    monkeypatch.setattr(tool, "authenticate", lambda *a: ({"old_study": tmp_path, "packet": tmp_path, "lexical": tmp_path},
                                                       {"scalar-4101": {"weights_sha256": "a", "predictions_sha256": "b"}}))
    monkeypatch.setattr(tool.old, "load_inputs", lambda *a: ({"dev": []}, [], None, None))
    monkeypatch.setattr(tool.torch, "set_num_threads", lambda *a: None)
    monkeypatch.setattr(tool.torch, "set_num_interop_threads", lambda *a: None)
    monkeypatch.setattr(tool.torch, "use_deterministic_algorithms", lambda *a: None)
    monkeypatch.setattr(tool, "restore_model", lambda *a: SimpleNamespace(trace=[]))
    folder = tmp_path / "fits/scalar-4101"
    folder.mkdir(parents=True)
    tool.save_arrays(folder / "dev-predictions.npz", {"dummy": np.zeros(1)})

    def batches(model, ds, q, f, lex, expected, work, check):
        work.update(model_forward_calls=1, model_forward_returned=1, matched_rows=5, returned_rows=5)
        return manual(), {"bitwise_equal": True}
    monkeypatch.setattr(tool, "replay_batches", batches)
    return args


def test_fake_outer_success_has_exclusive_outputs_and_bound_manifest(tmp_path, monkeypatch):
    args = fake_run(monkeypatch, tmp_path)
    receipt = tool.replay(args)
    completed = tool.read(args.out / "completed.json")
    assert receipt["completed_sha256"] == tool.sha(args.out / "completed.json")
    assert completed["model_forward_calls"] == 1 and completed["matched_rows"] == 5
    assert len(completed["files"]) == 4
    for name, entry in completed["files"].items():
        assert tool.sha(args.out / name) == entry["sha256"]
    with pytest.raises(FileExistsError):
        tool.replay(args)


def test_wrong_plan_digest_refuses_before_output(tmp_path, monkeypatch):
    args = fake_run(monkeypatch, tmp_path)
    args.plan_sha256 = "wrong"
    with pytest.raises(ValueError, match="Changed file"):
        tool.replay(args)
    assert not args.out.exists()


def test_started_write_failure_preserves_original_even_if_failure_write_fails(tmp_path, monkeypatch):
    args = fake_run(monkeypatch, tmp_path)
    error = OSError("original started failure")
    def broken(path, obj):
        if Path(path).name == "started.json":
            raise error
        raise OSError("secondary receipt failure")
    monkeypatch.setattr(tool, "write", broken)
    monkeypatch.setattr(tool.old, "write", broken)
    with pytest.raises(OSError) as caught:
        tool.replay(args)
    assert caught.value is error
    assert any("secondary receipt failure" in n for n in error.__notes__)


def test_late_completion_cap_demotes_terminal_without_doublecount(tmp_path, monkeypatch):
    args = fake_run(monkeypatch, tmp_path)
    clock = [0.]
    monkeypatch.setattr(tool.time, "perf_counter", lambda: clock[0])
    original = tool.write
    def slow(path, obj):
        original(path, obj)
        if Path(path) == args.out / "completed.json":
            clock[0] = 101.
    monkeypatch.setattr(tool, "write", slow)
    with pytest.raises(TimeoutError):
        tool.replay(args)
    assert not (args.out / "completed.json").exists()
    assert (args.out / "invalid-completion.json").exists()
    failure = tool.read(args.out / "failed.json")
    assert failure["active_progress"] == {} and len(failure["completed_fits"]) == 1


def test_checkpoint_failure_has_no_model_call(tmp_path, monkeypatch):
    args = fake_run(monkeypatch, tmp_path)
    def failure(*args):
        raise ValueError("bad checkpoint")
    monkeypatch.setattr(tool, "restore_model", failure)
    with pytest.raises(ValueError, match="bad checkpoint"):
        tool.replay(args)
    receipt = tool.read(args.out / "failed.json")
    assert receipt["active_progress"]["model_forward_calls"] == 0


def test_primary_intersection_not_overlapping_marginals_and_empty_five_bin_support():
    arrays = manual()
    result = tool.diagnostic(arrays)["strata"]
    primary = result["unseen/revision/prior_correct_adjacent_user_unique_gold"]
    assert primary["count"] == primary["wrong"] == 1
    assert primary["wrong_writer_partition"] == {"writer_gold": 1, "writer_previous_gold": 0, "writer_other": 0}
    assert primary["one_step_counterfactual"]["force_write"]["repair_wrong"] == 1
    assert result["unseen/revision/user_unique_gold"]["count"] == 2
    assert result["unseen/revision/prior_correct_adjacent"]["count"] == 2
    assert result["unseen/revision/prior_wrong/stale"]["count"] == 1
    assert result["unseen/revision/prior_correct/other_wrong"]["count"] == 1
    assert result["unseen/bin/clear"]["count"] == 0
    assert result["unseen/bin/first_assignment"]["count"] == 1
    assert sum(result["unseen/revision/evidence/" + name]["count"] for name in (
        "both_gold", "user_gold_only", "system_gold_only", "neither_gold")) == 3
    assert sum(result["unseen/primary/evidence/" + name]["count"] for name in (
        "both_gold", "user_gold_only", "system_gold_only", "neither_gold")) == 1


def test_retention_new_wrong_already_wrong_and_no_prior_partition():
    arrays = manual()
    arrays["labels"][:] = 1
    arrays["choice"][:] = [1, 2, 2, 1, 0]
    arrays["bin"][:] = "assigned_retention"
    row = tool.diagnostic(arrays)["strata"]
    assert row["unseen/retention/new_wrong"]["count"] == 2
    assert row["unseen/retention/already_wrong"]["count"] == 1
    assert row["unseen/retention/wrong_no_prior"]["count"] == 0
    assert sum(row["unseen/retention/" + k]["count"] for k in ("new_wrong", "already_wrong", "wrong_no_prior")) == 3


def test_partial_materialization_failure_never_replaces_original(tmp_path, monkeypatch):
    model = SimpleNamespace(trace=[{"old_b": np.zeros((1, 1, 2))}])
    original = ValueError("original replay mismatch")
    def allocation(*args, **kwargs):
        raise MemoryError("synthetic partial allocation")
    monkeypatch.setattr(tool.np, "stack", allocation)
    tool.preserve_failure(tmp_path, original, {}, model, [], tool.time.perf_counter())
    assert "synthetic partial allocation" in original.__notes__[0]
    assert tool.read(tmp_path / "failed.json")["error"] == "original replay mismatch"


def test_float32_factor_overshoot_preserved_without_clipping_or_renormalization():
    arrays = manual()
    arrays["departure_mass"][0] = float(np.float32(1.) + np.float32(1e-6))
    before = arrays["departure_mass"].copy()
    _, product = tool.factors(arrays)
    b, w, m = arrays["old_b"][0].astype(float), arrays["writer"][0].astype(float), before[0]
    uniform = np.r_[np.full(3, 1/3), np.zeros(9)]
    literal = ((1-m)*b + m*uniform)*w
    literal /= literal.sum()
    np.testing.assert_array_equal(product[0], literal)
    np.testing.assert_array_equal(arrays["departure_mass"], before)
    stats = tool.diagnostic(arrays)["factor_roundoff"]
    assert stats["departure_mass_above_one_count"] == 1
    assert stats["departure_mass_max_overshoot"] == before[0]-1
    arrays["departure_mass"][0] = 1.00001
    with pytest.raises(ValueError, match="factors"):
        tool.factors(arrays)


def byte_tree(monkeypatch, tmp_path):
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    for name in tool.old.SOURCES + tool.NEW_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source bytes\n")
    for folder in ("packet", "lexical"):
        target = tmp_path / folder
        target.mkdir()
        (target / "payload").write_bytes(b"opaque synthetic payload")
        tool.write(target / "completed.json", {"status": "completed", "files": {"payload": tool.sha(target / "payload")}})
    study = tmp_path / "old"
    study.mkdir()
    plan = {"runtime": tool.old.runtime(), "config": tool.old.CONFIG, "practical_checks": tool.old.PRACTICAL_CHECKS,
            "source_sha256": {name: tool.sha(tmp_path / name) for name in tool.old.SOURCES},
            "packet_completed_sha256": tool.sha(tmp_path / "packet/completed.json"),
            "lexical_completed_sha256": tool.sha(tmp_path / "lexical/completed.json")}
    tool.write(study / "plan.json", plan)
    monkeypatch.setattr(tool, "OLD_PLAN", tool.sha(study / "plan.json"))
    records = []
    for seed in tool.old.SEEDS:
        for method in tool.old.METHODS:
            path = study / "fits" / f"{method}-{seed}"
            path.mkdir(parents=True)
            for file in ("weights.pt", "dev-predictions.npz"):
                (path / file).write_bytes(b"unloaded synthetic bytes")
            record = {"method": method, "seed": seed, "status": "completed",
                      "weights_sha256": tool.sha(path / "weights.pt"),
                      "predictions_sha256": tool.sha(path / "dev-predictions.npz")}
            tool.write(path / "completed.json", record)
            records.append(record)
    (study / "references.npz").write_bytes(b"synthetic")
    tool.write(study / "completed.json", {"status": "completed", "fit_count": 15, "fits": records,
                "plan_sha256": tool.OLD_PLAN, "references": {"sha256": tool.sha(study / "references.npz")}})
    monkeypatch.setattr(tool, "OLD_COMPLETED", tool.sha(study / "completed.json"))
    return SimpleNamespace(old_study=study, packet=tmp_path / "packet", lexical=tmp_path / "lexical",
                           out=tmp_path / "frozen", wall_cap_seconds=300.)


def test_freeze_authenticates_exact_old_and_new_closure_without_deserialization(tmp_path, monkeypatch):
    args = byte_tree(monkeypatch, tmp_path)
    monkeypatch.setattr(tool.torch, "load", lambda *a, **k: pytest.fail("freeze must not load weights"))
    frozen = tool.freeze(args)
    assert frozen["plan_sha256"] == tool.sha(args.out / "plan.json")
    plan = tool.read(args.out / "plan.json")
    assert len(plan["source_sha256"]) == 16 and len(plan["fits"]) == 6
    assert plan["factor_roundoff"] == 2e-6
    (args.old_study / "unexpected").write_text("extra")
    with pytest.raises(ValueError, match="file closure"):
        tool.authenticate(plan)


def test_freeze_refuses_changed_source_and_different_cap_before_writing(tmp_path, monkeypatch):
    args = byte_tree(monkeypatch, tmp_path)
    args.wall_cap_seconds = 301.
    with pytest.raises(ValueError, match="cap"):
        tool.freeze(args)
    assert not args.out.exists()
    args.wall_cap_seconds = 300.
    (tmp_path / tool.old.SOURCES[0]).write_text("changed old source")
    with pytest.raises(ValueError, match="source rebinding"):
        tool.freeze(args)
    assert not args.out.exists()


def test_evidence_crosses_prior_correctness_and_actual_stale_other_outcomes():
    rows = tool.diagnostic(manual())["strata"]
    for base in ("revision/prior_correct", "revision/prior_wrong", "revision/any_prior/stale",
                 "revision/any_prior/other_wrong", "revision/prior_correct/stale",
                 "revision/prior_correct/other_wrong", "revision/prior_wrong/stale",
                 "revision/prior_wrong/other_wrong"):
        crossed = [rows["unseen/" + base + "/evidence/" + name] for name in (
            "both_gold", "user_gold_only", "system_gold_only", "neither_gold")]
        assert sum(r["count"] for r in crossed) == rows["unseen/" + base]["count"]
        for key in ("writer_gold", "writer_previous_gold", "writer_other"):
            assert sum(r["wrong_writer_partition"][key] for r in crossed) == rows["unseen/" + base]["wrong_writer_partition"][key]
    stale = rows["unseen/revision/prior_correct/stale/evidence/user_gold_only"]
    assert stale["count"] == 1 and stale["wrong_writer_partition"]["writer_gold"] == 1
    other = rows["unseen/revision/prior_correct/other_wrong/evidence/neither_gold"]
    assert other["count"] == 1 and other["wrong_writer_partition"]["writer_previous_gold"] == 1
