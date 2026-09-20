"""Tiny synthetic update, provenance, complete execution and failure tests."""
from __future__ import annotations

import copy
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import report_dialogue_alignment as reporter
import study_dialogue_alignment as s
import torch
from test_dialogue_alignment_common import setup


@pytest.fixture(autouse=True)
def small_threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def active(tmp_path):
    budget = s.Budget(tmp_path, time.monotonic())
    budget.progress["active_fit"] = {"name": "synthetic", "counts": dict.fromkeys(s.base.COUNTERS, 0)}
    return budget


def test_work_counts_follow_effective_boundaries_and_short_tail(monkeypatch):
    assert s.expected_work(29211, 13599) == {
        "updates_per_fit": 2300, "training_microbatches_per_fit": 18260,
        "evaluation_batches_per_fit": 54, "evaluation_microbatches_per_fit": 425}
    monkeypatch.setattr(s, "CONFIG", {**s.CONFIG, "epochs": 2, "batch_size": 3, "microbatch_size": 2})
    assert s.expected_work(7, 4) == {"updates_per_fit": 6, "training_microbatches_per_fit": 10,
                                  "evaluation_batches_per_fit": 2, "evaluation_microbatches_per_fit": 3}


@pytest.mark.parametrize("method", s.METHODS)
def test_microbatch_weighted_sum_matches_whole_batch_update_and_gradients(tmp_path, monkeypatch, method):
    fit, _, types, arrays, schema = setup(); rows = fit[:5]
    models, _ = s.common.init_models(torch, 6201)
    model = models[method]; whole = copy.deepcopy(model)
    initial = [p.detach().clone() for p in model.parameters()]
    recipe = s.typed.objective(fit)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    full_optimizer = torch.optim.AdamW(whole.parameters(), lr=.001, weight_decay=.0001)
    monkeypatch.setattr(s, "CONFIG", {**s.CONFIG, "microbatch_size": 2})
    record = s.update(model, optimizer, method, rows, arrays, types, schema, recipe, torch, active(tmp_path))
    actor, _ = s.common.actor(rows, arrays, types, schema, method)
    scores = whole(**{k: torch.from_numpy(v) for k, v in actor.items()})
    labels, _ = s.base.supervision(rows)
    weights = torch.from_numpy(s.typed.sample_weights(rows, recipe, "stratum"))
    loss = (-scores[torch.arange(len(rows)), torch.from_numpy(labels)]*weights).mean()
    loss.backward(); torch.nn.utils.clip_grad_norm_(whole.parameters(), s.CONFIG["gradient_clip"], error_if_nonfinite=True)
    full_optimizer.step()
    assert record["weighted_loss"] == pytest.approx(float(loss.detach()), abs=1e-6)
    assert record["microbatches"] == 3 and record["normalization"]["batches"] == 3
    assert record["normalization"]["rows"] == 5
    for p, expected, start in zip(model.parameters(), whole.parameters(), initial, strict=True):
        assert p.grad is not None and expected.grad is not None
        torch.testing.assert_close(p.grad, expected.grad, atol=2e-6, rtol=2e-5)
        # First-step AdamW is sensitive to nearly zero gradients. Verify its
        # actual accumulated gradient, not bit-parity across GEMM batch shapes.
        exact_update = start*(1-.001*.0001)-.001*p.grad/(p.grad.abs()+1e-8)
        torch.testing.assert_close(p, exact_update, atol=1e-7, rtol=1e-6)
    assert not any("max_" in key for key in record["work"])


def test_actor_never_consumes_current_labels_bins_or_panel_membership():
    fit, _, types, arrays, schema = setup()
    other = [{**row, "current_label_index": -100, "current_candidate_id": "poison",
              "current_value_group": "false", "derived_bin": "clear", "heldout_service": False} for row in fit]
    for method in s.METHODS:
        first, work = s.common.actor(fit, arrays, types, schema, method)
        second, again = s.common.actor(other, arrays, types, schema, method)
        assert work == again
        for key in first: np.testing.assert_array_equal(first[key], second[key])


def test_failed_second_microbatch_keeps_first_work_without_optimizer(tmp_path, monkeypatch):
    fit, _, types, arrays, schema = setup()
    model = s.common.init_models(torch, 6201)[0]["token_mean"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    monkeypatch.setattr(s, "CONFIG", {**s.CONFIG, "microbatch_size": 2})
    original = model.forward; calls = []
    def fail(**inputs):
        calls.append(1)
        if len(calls) == 2: raise RuntimeError("second forward")
        return original(**inputs)
    monkeypatch.setattr(model, "forward", fail)
    budget = active(tmp_path)
    with pytest.raises(RuntimeError, match="second forward"):
        s.update(model, optimizer, "token_mean", fit[:5], arrays, types, schema, s.typed.objective(fit), torch, budget)
    assert budget.progress["totals"] == {"forward_attempted": 2, "forward_returned": 1,
        "backward_attempted": 1, "backward_returned": 1, "optimizer_attempted": 0,
        "optimizer_returned": 0, "training_rows": 2, "evaluation_rows": 0}


def test_evaluation_preserves_canonical_returned_prefix_on_failure(tmp_path, monkeypatch):
    _, rows, types, arrays, schema = setup()
    model = s.common.init_models(torch, 6201)[0]["token_aligned"]
    original = model.forward; calls = []
    def fail(**inputs):
        calls.append(1)
        if len(calls) == 2: raise RuntimeError("evaluation stop")
        return original(**inputs)
    monkeypatch.setattr(model, "forward", fail)
    monkeypatch.setattr(s, "CONFIG", {**s.CONFIG, "batch_size": 3, "microbatch_size": 2})
    budget = active(tmp_path); budget.partial_path = tmp_path; budget.partial_model = model
    with pytest.raises(RuntimeError, match="evaluation stop") as caught:
        s.evaluate(model, "token_aligned", rows, arrays, types, schema, torch, budget)
    s.preserve_failure(tmp_path, budget, caught.value)
    assert budget.partial_predictions["completed_rows"] == 2
    with np.load(tmp_path/"partial-predictions.npz", allow_pickle=False) as saved:
        assert saved["row_indices"].tolist() == [6, 7]
        assert saved["log_probs"].shape == (2, 12)
        assert np.isneginf(saved["log_probs"][:, 4:]).all()
    assert (tmp_path/"partial-weights.pt").exists()
    assert s.read(tmp_path/"failed.json")["progress"]["totals"]["evaluation_rows"] == 2


def synthetic_plan(tmp_path, monkeypatch):
    fit, evaluation, types, arrays, schema = setup()
    monkeypatch.setattr(s, "CONFIG", {**s.CONFIG, "epochs": 1, "batch_size": 3, "microbatch_size": 2})
    folder = tmp_path/"freeze"; folder.mkdir(); orders = {}
    for index, seed in enumerate(s.SEEDS):
        order = np.roll(np.arange(len(fit), dtype=np.int64), index)[None]
        p = folder/f"orders-{seed}.npy"; np.save(p, order, allow_pickle=False)
        orders[str(seed)] = {"file": p.name, "sha256": s.sha(p), "shape": list(order.shape)}
    plan = {"objective": s.typed.objective(fit), "orders": orders, "source_sha256": {},
            "runtime": s.base.runtime(), "schema_completed_sha256": "cache", **s.expected_work(len(fit), len(evaluation))}
    path = folder/"plan.json"; s.write(path, plan)
    monkeypatch.setattr(s, "validate_plan", lambda *a: plan)
    monkeypatch.setattr(s, "load_training_inputs", lambda *a: (fit, evaluation, types, arrays, schema))
    monkeypatch.setattr(s, "backend", lambda: torch)
    monkeypatch.setattr(s, "end_authentication", lambda *a: None)
    monkeypatch.setattr(s, "process_load", lambda: {"status": "synthetic"})
    return SimpleNamespace(phase="train", out=tmp_path/"run", plan=path, plan_sha256=s.sha(path)), fit


def test_all_nine_synthetic_fits_pair_initializers_orders_manifest_and_report(tmp_path, monkeypatch):
    args, fit = synthetic_plan(tmp_path, monkeypatch)
    s.execute(args)
    done = s.read(args.out/"completed.json")
    assert done["completed_fits"] == s.FIT_ORDER
    assert done["original_capacity_admitted"] is False and done["quality_metrics_computed"] is False
    assert len(done["files"]) == 43 and len(list(args.out.rglob("*"))) >= 44
    assert done["progress"]["active_fit"] is None
    assert done["progress"]["totals"] == {"forward_attempted": 63, "forward_returned": 63,
        "backward_attempted": 36, "backward_returned": 36, "optimizer_attempted": 18,
        "optimizer_returned": 18, "training_rows": 54, "evaluation_rows": 36}
    for seed in s.SEEDS:
        records = {}; expected_order = np.load(args.plan.parent/f"orders-{seed}.npy", allow_pickle=False)[0]
        for method in s.METHODS:
            dest = args.out/"fits"/f"{method}-{seed}"; meta = s.read(dest/"completed.json"); records[method] = meta
            assert meta["training_microbatches"] == 4 and meta["training_effective_batches"] == 2
            assert meta["evaluation"]["effective_batches"] == 2 and meta["evaluation"]["microbatches"] == 3
            assert meta["training_normalization"]["batches"] == 4 and meta["training_normalization"]["rows"] == 6
            assert set(meta["files"]) == {"weights.pt", "updates.jsonl", "predictions.npz"}
            journal = s.base.read_rows(dest/"updates.jsonl")
            assert [i for r in journal for i in r["row_indices"]] == [fit[i]["row_index"] for i in expected_order]
            with np.load(dest/"predictions.npz", allow_pickle=False) as packet:
                assert packet["row_indices"].tolist() == [6, 7, 8, 9]
                assert packet["log_probs"].dtype == np.float32 and packet["log_probs"].shape == (4, 12)
                assert np.isneginf(packet["log_probs"][:, 4:]).all()
                np.testing.assert_allclose(np.exp(packet["log_probs"].astype(np.float64)).sum(-1), 1., atol=2e-6)
        assert records["token_mean"]["initial_state_sha256"] == records["token_aligned"]["initial_state_sha256"]
        assert len({m["initial_common_sha256"] for m in records.values()}) == 1
    with np.load(args.out/"references.npz", allow_pickle=False) as refs:
        assert set(refs.files) == {"row_indices", "previous_indices", "literal_indices"}
    assert all(r["candidate_types"] == [0, 1, 2, 3] for r in s.base.read_rows(args.out/"evaluation-rows.jsonl"))
    # Exercise the real saved-array metric/report path after all tiny fits.
    # Production authenticate_run is deliberately NOT bypassed and called:
    # its real parent pins/cohort sizes are outside this synthetic test scope.
    packets, receipts = {}, {}
    for name in s.FIT_ORDER:
        dest = args.out/"fits"/name
        with np.load(dest/"predictions.npz", allow_pickle=False) as archive:
            packets[name] = {key: archive[key] for key in archive.files}
        receipts[name] = s.read(dest/"completed.json")
    with np.load(args.out/"references.npz", allow_pickle=False) as archive:
        refs = {key: archive[key] for key in archive.files}
    rows = s.base.read_rows(args.out/"evaluation-rows.jsonl")
    summary = reporter.aggregate(rows, packets, refs)
    assert summary["groups_per_fit"] == 144 and summary["continuation"]["total_checks"] == 22
    assert set(summary["fits"]) == set(s.FIT_ORDER) and set(summary["references"]) == {"previous", "literal"}
    for name, packet in packets.items():
        target = np.asarray([row["current_label_index"] for row in rows])
        choices = packet["log_probs"].argmax(-1)
        expected_nll = -packet["log_probs"][np.arange(4), target].astype(np.float64).mean()
        all_cell = summary["fits"][name]["cells"]["all/all"]
        assert all_cell["nll"]["row"] == pytest.approx(expected_nll, abs=1e-12)
        assert all_cell["accuracy"]["row"] == float((choices == target).mean())
        assert summary["fits"][name]["decisions"]["accuracy"]["denominator"] == 2
    summary["costs"] = {"whole_wall_seconds": done["wall_seconds"],
        "scope": "Synthetic serialization integration only; no production authentication or efficacy claim.",
        "per_fit": {name: {"training_wall_seconds": r["training_wall_seconds"],
            "evaluation_wall_seconds": r["evaluation"]["wall_seconds"], "wall_seconds": r["wall_seconds"]}
            for name, r in receipts.items()}}
    report_path = tmp_path/"synthetic-summary.json"; reporter.write(report_path, summary)
    reloaded = reporter.read(report_path)
    assert reloaded == summary
    text = reporter.report_text(reloaded)
    assert all(name in text for name in s.FIT_ORDER) and "/22 expanded checks" in text
    missing = dict(packets); missing.pop(s.FIT_ORDER[-1])
    with pytest.raises(ValueError, match="All nine"): reporter.aggregate(rows, missing, refs)
    corrupt = dict(packets); corrupt[s.FIT_ORDER[0]] = {**packets[s.FIT_ORDER[0]], "row_indices": np.array([7, 6, 8, 9], np.int64)}
    with pytest.raises(ValueError):
        reporter.aggregate(rows, corrupt, refs)
    with pytest.raises(FileExistsError): s.execute(args)


def test_own_budget_does_not_inherit_old_3600_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(s.base, "peak_rss", lambda: 10)
    b = s.Budget(tmp_path, time.monotonic()-4000); b.check()
    b.start -= 4000
    with pytest.raises(TimeoutError): b.check()
    b.start = time.monotonic(); monkeypatch.setattr(s.base, "peak_rss", lambda: s.LIMITS["rss_bytes"]+1)
    with pytest.raises(ValueError, match="RSS"): b.check()


def test_original_error_survives_failure_receipt_error(tmp_path, monkeypatch):
    args = SimpleNamespace(phase="freeze", out=tmp_path/"run")
    original = RuntimeError("primary")
    def fail(*a): raise original
    monkeypatch.setattr(s, "freeze", fail)
    write = s.write
    def break_receipt(path, value):
        if Path(path).name == "failed.json": raise OSError("secondary")
        return write(path, value)
    monkeypatch.setattr(s, "write", break_receipt)
    monkeypatch.setattr(s, "process_load", dict)
    with pytest.raises(RuntimeError) as got: s.execute(args)
    assert got.value is original


def test_completed_payload_is_demoted_if_final_cap_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "freeze", lambda *a: {"model_calls": 0})
    monkeypatch.setattr(s, "process_load", dict)
    original = s.Budget.storage
    def late(budget, extra=0):
        if (budget.out/"completed.json").exists(): raise TimeoutError("late cap")
        return original(budget, extra)
    monkeypatch.setattr(s.Budget, "storage", late)
    args = SimpleNamespace(phase="freeze", out=tmp_path/"run")
    with pytest.raises(TimeoutError, match="late cap"): s.execute(args)
    assert (args.out/"late-completion.json").exists() and not (args.out/"completed.json").exists()
    assert s.read(args.out/"failed.json")["status"] == "failed"


def test_order_hash_and_permutation_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "CONFIG", {**s.CONFIG, "epochs": 1})
    for seed in s.SEEDS: np.save(tmp_path/f"orders-{seed}.npy", np.array([[2, 0, 1]], np.int64))
    assert s.load_orders(tmp_path, 3)[s.SEEDS[0]].tolist() == [[2, 0, 1]]
    np.save(tmp_path/f"orders-{s.SEEDS[0]}.npy", np.array([[2, 0, 0]], np.int64))
    with pytest.raises(ValueError, match="Complete paired"): s.load_orders(tmp_path, 3)


def test_external_pin_rejected_before_parent_authentication(tmp_path, monkeypatch):
    path = tmp_path/"plan.json"; path.write_text("{}")
    def forbidden(*a): pytest.fail("Parent must not be read before external pin")
    monkeypatch.setattr(s, "authenticate_capacity", forbidden)
    with pytest.raises(ValueError, match="External plan"): s.validate_plan(path, "wrong", active(tmp_path))
