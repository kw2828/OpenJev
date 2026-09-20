"""Artificial-only objective, initialization, closure and failure checks."""
from __future__ import annotations

import copy
import time
from types import SimpleNamespace

import numpy as np
import pytest
import study_dialogue_objective as s
import torch
from test_dialogue_alignment_common import setup


@pytest.fixture(autouse=True)
def small_threads():
    original = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def active(tmp_path):
    budget = s.Budget(tmp_path, time.monotonic())
    budget.progress["active_fit"] = {"name": "synthetic", "counts": dict.fromkeys(s.base.COUNTERS, 0)}
    return budget


def test_paired_initializers_match_original_draw_sequence_without_aliasing():
    original, expected = s.common.init_models(torch, 6201)
    models, witness = s.init_models(torch, 6201)
    assert set(witness["full_state_sha256"].values()) == {expected["full_state_sha256"]["token_aligned"]}
    assert witness["original_sequence"] == expected
    assert witness["original_sequence"]["full_state_sha256"]["token_mean"] == witness["full_state_sha256"]["stratum"]
    for method in s.METHODS:
        assert models[method].mode == "aligned"
        for name, p in models[method].named_parameters():
            assert torch.equal(p, dict(original["token_aligned"].named_parameters())[name])
    a, b = models.values()
    with torch.no_grad(): next(a.parameters()).add_(1)
    assert not torch.equal(next(a.parameters()), next(b.parameters()))
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.001) for m in models.values()]
    assert optimizers[0].state is not optimizers[1].state


@pytest.mark.parametrize("weighting", s.METHODS)
def test_microtail_loss_and_gradient_match_full_batch_ce(tmp_path, monkeypatch, weighting):
    fit, _, types, arrays, schema = setup(); rows = fit[:5]
    models, _ = s.init_models(torch, 6201)
    model = models[weighting]; whole = copy.deepcopy(model)
    initial = [p.detach().clone() for p in model.parameters()]
    recipe = s.typed.objective(fit)
    monkeypatch.setattr(s, "CONFIG", {**s.CONFIG, "microbatch_size": 2})
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    record = s.update(model, optimizer, weighting, rows, arrays, types, schema, recipe, torch, active(tmp_path))
    actor, _ = s.common.actor(rows, arrays, types, schema, s.MODEL_METHOD)
    scores = whole(**{k: torch.from_numpy(v) for k, v in actor.items()})
    labels = torch.from_numpy(s.base.supervision(rows)[0])
    if weighting == "uniform":
        loss = torch.nn.functional.nll_loss(scores, labels)
    else:
        weights = torch.from_numpy(s.typed.sample_weights(rows, recipe, "stratum"))
        loss = (-scores[torch.arange(len(rows)), labels]*weights).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(whole.parameters(), s.CONFIG["gradient_clip"], error_if_nonfinite=True)
    assert record["weighted_loss"] == pytest.approx(float(loss.detach()), abs=1e-6)
    assert record["microbatches"] == record["normalization"]["batches"] == 3
    assert record["normalization"]["rows"] == 5
    for p, expected, start in zip(model.parameters(), whole.parameters(), initial, strict=True):
        assert p.grad is not None and bool(torch.isfinite(p.grad).all())
        torch.testing.assert_close(p.grad, expected.grad, atol=2e-6, rtol=2e-5)
        expected_update = start*(1-.001*.0001)-.001*p.grad/(p.grad.abs()+1e-8)
        torch.testing.assert_close(p, expected_update, atol=1e-7, rtol=1e-6)
    assert not any("max_" in k for k in record["work"])


def test_uniform_weights_ignore_strata_and_actor_ignores_current_labels():
    fit, _, types, arrays, schema = setup()
    poison = [{**r, "current_label_index": -100, "current_value_group": "invalid",
               "derived_bin": "invalid", "heldout_service": not r["heldout_service"]} for r in fit]
    np.testing.assert_array_equal(s.loss_weights(poison, None, "uniform"), np.ones(len(fit), np.float32))
    first, work = s.common.actor(fit, arrays, types, schema, s.MODEL_METHOD)
    second, again = s.common.actor(poison, arrays, types, schema, s.MODEL_METHOD)
    assert work == again
    for k in first: np.testing.assert_array_equal(first[k], second[k])
    with pytest.raises(ValueError, match="Unknown objective"):
        s.loss_weights(fit, None, "balanced")
    weights = s.loss_weights(fit, s.typed.objective(fit), "stratum")
    assert weights.mean() == pytest.approx(1.)


def test_failure_counts_record_completed_micros_without_optimizer(tmp_path, monkeypatch):
    fit, _, types, arrays, schema = setup()
    model = s.init_models(torch, 6201)[0]["uniform"]
    optimizer = torch.optim.AdamW(model.parameters())
    monkeypatch.setattr(s, "CONFIG", {**s.CONFIG, "microbatch_size": 2})
    forward, calls = model.forward, []
    def fail(**inputs):
        calls.append(1)
        if len(calls) == 2: raise RuntimeError("second micro")
        return forward(**inputs)
    monkeypatch.setattr(model, "forward", fail)
    budget = active(tmp_path)
    with pytest.raises(RuntimeError, match="second micro"):
        s.update(model, optimizer, "uniform", fit[:5], arrays, types, schema, None, torch, budget)
    assert budget.progress["totals"] == {"forward_attempted": 2, "forward_returned": 1,
        "backward_attempted": 1, "backward_returned": 1, "optimizer_attempted": 0,
        "optimizer_returned": 0, "training_rows": 2, "evaluation_rows": 0}


def test_production_work_and_orders(tmp_path):
    assert s.expected_work(29211, 13599) == {"updates_per_fit": 2300, "training_microbatches_per_fit": 18260,
        "evaluation_batches_per_fit": 54, "evaluation_microbatches_per_fit": 425}
    items = {}
    for seed in s.SEEDS:
        path = tmp_path/f"orders-{seed}.npy"
        np.save(path, np.tile(np.array([2, 0, 1], np.int64), (20, 1)), allow_pickle=False)
        items[str(seed)] = {"file": path.name, "sha256": s.sha(path)}
    assert s.load_orders(tmp_path, 3, items)[6201].shape == (20, 3)
    np.save(tmp_path/"orders-6201.npy", np.zeros((20, 3), np.int64))
    with pytest.raises(ValueError, match="Orders hash"): s.load_orders(tmp_path, 3, items)
    with pytest.raises(ValueError, match="Complete paired"): s.load_orders(tmp_path, 3)


def freeze_fixture(tmp_path, monkeypatch):
    """Real new freeze closure, with artificial parent metadata and sources."""
    root = tmp_path/"repo"; root.mkdir()
    parent_dir = tmp_path/"parent"; parent_dir.mkdir()
    monkeypatch.setattr(s, "ROOT", root)
    monkeypatch.setattr(s, "NEW_SOURCES", ("runner.py", "protocol.md"))
    for name in ("old.py", *s.NEW_SOURCES): (root/name).write_text(name)
    fit, evaluation, _, _, schema = setup()
    objective = s.typed.objective(fit)
    orders = {}
    for seed in s.SEEDS:
        path = parent_dir/f"orders-{seed}.npy"; np.save(path, np.tile(np.arange(len(fit), dtype=np.int64), (20, 1)))
        orders[str(seed)] = {"file": path.name, "sha256": s.sha(path), "shape": [20, len(fit)]}
    parent = {"version": "parent", "config": {}, "limits": {}, "runtime": s.base.runtime(),
              "source_sha256": {"old.py": s.sha(root/"old.py")}, "allocation": "old", "initialization": "old",
              "fit_row_indices": [r["row_index"] for r in fit], "evaluation_row_indices": [r["row_index"] for r in evaluation],
              "schema_cache_path": "synthetic", "schema_completed_sha256": "test", "orders": orders,
              "objective": objective, "split": {"heldout_services": ["synthetic"]}, "feature_headers": {},
              "quality_metrics_in_runner": False, "no_retry": True, "expected_fits": [],
              **s.expected_work(len(fit), len(evaluation))}
    s.write(parent_dir/"plan.json", parent); s.write(parent_dir/"completed.json", {"status": "completed"})
    monkeypatch.setattr(s, "PARENT_PLAN", parent_dir/"plan.json")
    monkeypatch.setattr(s, "PARENT_PIN", s.sha(parent_dir/"plan.json"))
    monkeypatch.setattr(s, "PARENT_FREEZE_COMPLETED_PIN", s.sha(parent_dir/"completed.json"))
    monkeypatch.setattr(s, "authenticate_parent", lambda budget: parent)
    monkeypatch.setattr(s.schema_prep, "authenticate_cache_metadata", lambda *a: (schema["index"], schema["offsets"]))
    monkeypatch.setattr(s.common, "load_metadata", lambda *a: (None, fit, evaluation, parent["split"]))
    monkeypatch.setattr(s.base, "headers", dict)
    monkeypatch.setattr(s, "process_load", dict)
    args = SimpleNamespace(phase="freeze", out=tmp_path/"freeze")
    s.execute(args)
    return args.out/"plan.json"


def test_freeze_is_metadata_only_and_exact_closure(tmp_path, monkeypatch):
    def forbidden(*a): pytest.fail("Metadata freeze must not load backend/float arrays")
    monkeypatch.setattr(s, "backend", forbidden); monkeypatch.setattr(s, "load_training_inputs", forbidden)
    path = freeze_fixture(tmp_path, monkeypatch)
    plan = s.validate_plan(path, s.sha(path), active(tmp_path))
    assert plan["objective_weightings"] == s.WEIGHTINGS
    assert s.read(path.parent/"completed.json")["model_calls"] == 0
    (path.parent/"extra.txt").write_text("unbound")
    with pytest.raises(ValueError, match="Freeze exact files"):
        s.validate_plan(path, s.sha(path), active(tmp_path))


@pytest.mark.parametrize("field", ["runtime", "objective", "orders", "source_sha256", "objective_weightings"])
def test_resealed_plan_metadata_tamper_rejected(tmp_path, monkeypatch, field):
    path = freeze_fixture(tmp_path, monkeypatch)
    plan = s.read(path); plan[field] = {}; path.write_bytes(s.base.prep.encoded(plan))
    with pytest.raises(ValueError): s.validate_plan(path, s.sha(path), active(tmp_path))


def test_snapshot_bytes_and_external_pin_fail_before_backend(tmp_path, monkeypatch):
    path = freeze_fixture(tmp_path, monkeypatch)
    (path.parent/"sources/old.py").write_text("changed")
    with pytest.raises(ValueError, match="payload drift"): s.validate_plan(path, s.sha(path), active(tmp_path))
    monkeypatch.setattr(s, "authenticate_parent", lambda *a: pytest.fail("Must fail external pin first"))
    with pytest.raises(ValueError, match="External plan"): s.validate_plan(path, "bad", active(tmp_path))


def synthetic_run(tmp_path, monkeypatch):
    fit, evaluation, types, arrays, schema = setup()
    monkeypatch.setattr(s, "CONFIG", {**s.CONFIG, "epochs": 1})
    folder = tmp_path/"plan"; folder.mkdir(); orders = {}
    for i, seed in enumerate(s.SEEDS):
        path = folder/f"orders-{seed}.npy"
        np.save(path, np.roll(np.arange(len(fit), dtype=np.int64), i)[None])
        orders[str(seed)] = {"file": path.name, "sha256": s.sha(path), "shape": [1, len(fit)]}
    plan = {"orders": orders, "objective": s.typed.objective(fit), "source_sha256": {}, "schema_completed_sha256": "cache",
            **s.expected_work(len(fit), len(evaluation))}
    path = folder/"plan.json"; s.write(path, plan)
    monkeypatch.setattr(s, "validate_plan", lambda *a: plan)
    monkeypatch.setattr(s, "load_training_inputs", lambda *a: (fit, evaluation, types, arrays, schema))
    monkeypatch.setattr(s, "backend", lambda: torch)
    monkeypatch.setattr(s, "end_authentication", lambda *a: None)
    monkeypatch.setattr(s, "process_load", dict)
    return SimpleNamespace(phase="train", out=tmp_path/"run", plan=path, plan_sha256=s.sha(path))


def test_all_six_artificial_fits_save_raw_predictions_and_ordered_receipts(tmp_path, monkeypatch):
    args = synthetic_run(tmp_path, monkeypatch)
    s.execute(args); done = s.read(args.out/"completed.json")
    assert done["completed_fits"] == ["stratum-6201", "uniform-6201", "uniform-6202", "stratum-6202", "stratum-6203", "uniform-6203"]
    assert done["quality_metrics_computed"] is False and done["encoder_calls"] == 0
    assert len(done["files"]) == 31 and len([p for p in args.out.rglob("*") if p.is_file()]) == 32
    assert done["progress"]["totals"] == {"forward_attempted": 12, "forward_returned": 12,
        "backward_attempted": 6, "backward_returned": 6, "optimizer_attempted": 6,
        "optimizer_returned": 6, "training_rows": 36, "evaluation_rows": 24}
    for seed in s.SEEDS:
        receipts = []
        for method in s.METHODS:
            dest = args.out/"fits"/f"{method}-{seed}"; rec = s.read(dest/"completed.json"); receipts.append(rec)
            assert rec["model_method"] == "token_aligned" and rec["objective_weighting"] == method
            assert rec["configuration"]["mode"] == "aligned"
            assert rec["orders_sha256"] == s.sha(args.plan.parent/f"orders-{seed}.npy")
            assert set(rec["files"]) == {"updates.jsonl", "weights.pt", "predictions.npz"}
            with np.load(dest/"predictions.npz", allow_pickle=False) as packet:
                assert set(packet.files) == {"row_indices", "log_probs"}
                assert packet["row_indices"].tolist() == [6, 7, 8, 9]
                assert packet["log_probs"].shape == (4, 12) and packet["log_probs"].dtype == np.float32
                assert np.isneginf(packet["log_probs"][:, 4:]).all()
        assert receipts[0]["initial_state_sha256"] == receipts[1]["initial_state_sha256"]
    with pytest.raises(FileExistsError): s.execute(args)


def test_failed_training_preserves_partial_weights_request_and_original_error(tmp_path, monkeypatch):
    args = synthetic_run(tmp_path, monkeypatch)
    error = RuntimeError("synthetic update stop")
    def fail(*a): raise error
    monkeypatch.setattr(s, "update", fail)
    with pytest.raises(RuntimeError) as got: s.execute(args)
    assert got.value is error
    failed = s.read(args.out/"failed.json")
    assert failed["request"]["plan_sha256"] == args.plan_sha256
    assert failed["progress"]["completed_fits"] == []
    assert (args.out/"fits/stratum-6201/partial-weights.pt").exists()
    assert not (args.out/"completed.json").exists()


def test_budget_wall_rss_storage_and_completion_demotion(tmp_path, monkeypatch):
    monkeypatch.setattr(s.base, "peak_rss", lambda: 10)
    budget = s.Budget(tmp_path, time.monotonic()-6001)
    with pytest.raises(TimeoutError): budget.check()
    budget.start = time.monotonic()
    with pytest.raises(ValueError, match="storage"): budget.storage(s.LIMITS["output_bytes"]+1)
    monkeypatch.setattr(s.base, "peak_rss", lambda: s.LIMITS["rss_bytes"]+1)
    with pytest.raises(ValueError, match="RSS"): budget.check()
    monkeypatch.setattr(s.base, "peak_rss", lambda: 10)
    monkeypatch.setattr(s, "freeze", lambda *a: {"model_calls": 0})
    monkeypatch.setattr(s, "process_load", dict)
    original = s.Budget.storage
    def late(b, extra=0):
        if (b.out/"completed.json").exists(): raise TimeoutError("late cap")
        return original(b, extra)
    monkeypatch.setattr(s.Budget, "storage", late)
    args = SimpleNamespace(phase="freeze", out=tmp_path/"failure")
    with pytest.raises(TimeoutError, match="late cap"): s.execute(args)
    assert not (args.out/"completed.json").exists() and (args.out/"late-completion.json").exists()
    assert s.read(args.out/"failed.json")["limits"] == s.FREEZE_LIMITS


def test_metadata_phase_has_separate_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(s.base, "peak_rss", lambda: 10)
    b = s.Budget(tmp_path, time.monotonic()-61, "freeze")
    with pytest.raises(TimeoutError): b.check()
    b.start = time.monotonic()
    with pytest.raises(ValueError, match="storage"): b.storage(s.FREEZE_LIMITS["output_bytes"]+1)
    monkeypatch.setattr(s.base, "peak_rss", lambda: s.FREEZE_LIMITS["rss_bytes"]+1)
    with pytest.raises(ValueError, match="RSS"): b.check()
    s.Budget(tmp_path, time.monotonic(), "train").check()
