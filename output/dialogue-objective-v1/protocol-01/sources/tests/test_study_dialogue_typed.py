"""Synthetic split, actor, objective, execution and failure checks."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import study_dialogue_typed as s


def row(i, target=0, previous=0):
    ids = ["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:True", "value:False"]
    kind = ("unmentioned_retention" if target == previous == 0 else "assigned_retention" if target == previous
            else "first_assignment" if previous == 0 else "clear" if target == 0 else "revision")
    return {"row_index": i, "split": "train", "admission": "admitted", "dialogue_id": f"d{i}",
            "service": "service", "slot": "slot", "query_id": "service/slot", "query_index": 0,
            "candidate_count": 4, "current_label_index": target, "previous_current_index": previous,
            "current_candidate_id": ids[target], "previous_candidate_id": ids[previous],
            "current_value_group": s.VALUES[target], "derived_bin": kind, "heldout_service": True,
            "cache": {"pooled_index": 0, "query_feature_index": 1, "candidate_feature_indices": [2, 3, 4, 5],
                "token_start": 0, "token_stop": 3, "lexical_start": i*40, "lexical_candidates": 4, "lexical_stride": 10}}


def fixture():
    rows = [row(0), row(1), row(2, 2, 2), row(3, 2), row(4, 1), row(5, 3)]
    evaluation = [row(6, 2), row(7, 1), row(8), row(9, 2, 2)]
    rng = np.random.default_rng(11)
    arrays = {"features": rng.normal(size=(6, 384)).astype(np.float32),
              "tokens": rng.normal(size=(3, 384)).astype(np.float32),
              "priors": np.full(3, 1/3, np.float32), "lexical": np.zeros(400, np.float32)}
    for r in rows+evaluation: arrays["lexical"][r["cache"]["lexical_start"]+4] = 1.
    queries = [{"query_index": 0, "candidate_ids": ["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:True", "value:False"],
                "candidate_values": [None, None, "True", "False"], "boolean_slot": True}]
    return rows, evaluation, queries, arrays


def test_split_is_service_hash_based_and_excludes_whole_dialogues():
    services = {f"d{i}": [f"s{i}"] for i in range(10)}
    rows = [{**row(i), "service": f"s{i}"} for i in range(10)]
    fit, ev, split = s.split_rows(rows, services)
    held = split["heldout_services"]
    nonheld = next(r["service"] for r in fit)
    services["mixed"] = [nonheld, held[0]]
    extra = {**row(10), "dialogue_id": "mixed", "service": nonheld}
    newfit, newev, newsplit = s.split_rows(rows+[extra], services)
    assert newsplit["heldout_services"] == held
    assert extra["row_index"] not in [r["row_index"] for r in newfit]
    assert next(r for r in newev if r["dialogue_id"] == "mixed")["heldout_service"] is False
    changed = [{**r, "current_value_group": "dontcare", "current_label_index": 1} for r in rows+[extra]]
    a, b, _ = s.split_rows(changed, dict(reversed(list(services.items()))))
    assert [r["row_index"] for r in a] == [r["row_index"] for r in newfit]
    assert [r["row_index"] for r in b] == [r["row_index"] for r in newev]
    assert len(fit)+len(ev) == len(rows)


def test_public_types_do_not_confuse_literal_none_and_reserved_none():
    _, _, queries, _ = fixture()
    assert s.public_candidate_types(queries) == {0: [0, 1, 2, 3]}
    q = copy.deepcopy(queries[0]); q["boolean_slot"] = False
    q["candidate_ids"][2] = "value:None"; q["candidate_values"][2] = "None"
    assert s.public_candidate_types([q])[0] == [0, 1, 4, 4]
    q["candidate_ids"][0] = "value:none"
    with pytest.raises(ValueError, match="support"): s.public_candidate_types([q])


def test_weighting_preserves_stratum_mass_and_balances_only_nonempty_types():
    rows = [row(i) for i in range(7)] + [row(7, 2, 2), row(8, 2), row(9, 2), row(10, 1)]
    recipe = s.objective(rows)
    for weighting in ("stratum", "balanced"):
        weights = s.sample_weights(rows, recipe, weighting)
        assert weights.sum() == pytest.approx(len(rows))
        for k in range(3):
            assert weights[[s.base.stratum(r) == k for r in rows]].sum() == pytest.approx(len(rows)/3)
    balanced = s.sample_weights(rows, recipe, "balanced")
    assert balanced[-1] == pytest.approx(2*balanced[-2])
    assert recipe["weights"]["balanced"][2][3] is None
    with pytest.raises(ValueError, match="Unsupported"): s.sample_weights([row(99, 3)], recipe, "balanced")
    with pytest.raises(ValueError, match="stratum"): s.objective([row(0)])


def test_actor_ignores_current_target_and_uses_public_catalog_types():
    rows, _, queries, arrays = fixture()
    types = s.public_candidate_types(queries)
    a, work = s.actor(rows[:2], arrays, types)
    poisoned = [{**r, "current_label_index": 3, "current_candidate_id": "fake", "current_value_group": "false",
                 "derived_bin": "clear", "heldout_service": False, "candidate_types": [4, 4, 4, 4]} for r in rows[:2]]
    b, _ = s.actor(poisoned, arrays, types)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert a["candidate_types"].tolist() == [[0, 1, 2, 3]]*2
    assert work["candidate_type_input_bytes"] == 2*4*8
    shifted = [{**rows[0], "previous_current_index": 2}]
    c, _ = s.actor(shifted, arrays, types)
    assert not np.array_equal(a["previous_onehot"][:1], c["previous_onehot"])


def test_reference_frequency_uses_fit_labels_only():
    fit, evaluation, queries, arrays = fixture()
    types = s.public_candidate_types(queries)
    a = s.references(evaluation, fit, arrays, types)
    changed = [{**r, "current_label_index": 3, "current_value_group": "false"} for r in evaluation]
    b = s.references(changed, fit, arrays, types)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_twelve_fit_synthetic_end_to_end(tmp_path, monkeypatch):
    import torch
    fit, evaluation, queries, arrays = fixture()
    monkeypatch.setitem(s.CONFIG, "epochs", 1); monkeypatch.setitem(s.CONFIG, "batch_size", 3)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    freeze = tmp_path/"freeze"; freeze.mkdir()
    orders = {}
    for seed in s.SEEDS:
        path = freeze/f"orders-{seed}.npy"
        np.save(path, np.arange(len(fit), dtype=np.int64)[None])
        orders[str(seed)] = {"file": path.name, "sha256": s.sha(path)}
    plan = {"fit_row_indices": [r["row_index"] for r in fit], "evaluation_row_indices": [r["row_index"] for r in evaluation],
            "split": {"heldout_services": ["service"]}, "objective": s.objective(fit), "feature_headers": {},
            "updates_per_fit": 2, "evaluation_batches_per_fit": 2, "orders": orders, "source_sha256": {}, "runtime": s.base.runtime()}
    p = freeze/"plan.json"; s.write(p, plan)
    monkeypatch.setattr(s, "validate_plan", lambda *a: plan)
    monkeypatch.setattr(s, "load_metadata", lambda *a: (queries, fit, evaluation, {"heldout_services": ["service"]}))
    monkeypatch.setattr(s.base, "headers", dict)
    monkeypatch.setattr(s.base, "load_arrays", lambda p: arrays)
    monkeypatch.setattr(s, "source_map", lambda *a: {})
    monkeypatch.setattr(s.base, "authenticate_prepared", lambda *a, **kw: None)
    original_sha = s.sha
    monkeypatch.setattr(s, "sha", lambda path, *a: s.SPLIT_PIN if Path(path) == s.SPLIT_METADATA/"receipt.json" else original_sha(path, *a))
    out = tmp_path/"run"
    args = SimpleNamespace(phase="train", out=out, plan=p, plan_sha256=original_sha(p))
    s.execute(args)
    done = s.read(out/"completed.json")
    assert done["completed_fits"] == s.FIT_ORDER
    assert done["quality_metrics_computed"] is False
    for seed in s.SEEDS:
        hashes = set()
        for method in s.METHODS:
            d = out/"fits"/f"{method}-{seed}"
            meta = s.read(d/"completed.json"); hashes.add(meta["initial_state_sha256"])
            assert meta["counts"]["optimizer_returned"] == 2
            assert meta["counts"]["evaluation_rows"] == 4
            with np.load(d/"predictions.npz") as packet:
                assert packet["row_indices"].tolist() == [6, 7, 8, 9]
                assert np.allclose(np.exp(packet["log_probs"].astype(np.float64)).sum(-1), 1., atol=2e-6)
        assert len(hashes) == 1
    saved = s.base.read_rows(out/"evaluation-rows.jsonl")
    assert all(r["candidate_types"] == [0, 1, 2, 3] for r in saved)


def test_failure_preserves_prediction_prefix_and_original_error(tmp_path, monkeypatch):
    def fail(out, budget, args):
        budget.partial_path = out/"partial"; budget.partial_path.mkdir()
        budget.partial_predictions = {"log_probs": np.zeros((4, 12), np.float32), "row_indices": np.arange(4, dtype=np.int64), "completed_rows": 2}
        raise RuntimeError("original failure")
    monkeypatch.setattr(s, "train", fail)
    out = tmp_path/"failed"
    with pytest.raises(RuntimeError, match="original failure"):
        s.execute(SimpleNamespace(phase="train", out=out))
    assert s.read(out/"failed.json")["status"] == "failed"
    with np.load(out/"partial/partial-predictions.npz") as p:
        assert p["row_indices"].tolist() == [0, 1] and p["log_probs"].shape == (2, 12)
    write = s.write
    def broken_receipt(path, value):
        if path.name == "failed.json": raise OSError("disk")
        write(path, value)
    monkeypatch.setattr(s, "write", broken_receipt)
    with pytest.raises(RuntimeError, match="original failure") as e:
        s.execute(SimpleNamespace(phase="train", out=tmp_path/"receipt-failure"))
    assert any("Failure receipt" in n for n in e.value.__notes__)


def test_snapshot_source_cannot_be_rebound_by_changing_only_manifest(tmp_path, monkeypatch):
    name = "fixture.py"; payload = tmp_path/"sources"/name; payload.parent.mkdir()
    payload.write_text("original")
    sources = {name: s.sha(payload)}
    monkeypatch.setattr(s, "source_map", lambda *a: sources)
    plan = {"version": s.VERSION, "config": s.CONFIG, "limits": s.LIMITS, "runtime": s.base.runtime(),
            "expected_fits": s.FIT_ORDER, "source_sha256": sources, "prepared_completed_sha256": s.PREPARED_PIN,
            "split_receipt_sha256": s.SPLIT_PIN}
    s.write(tmp_path/"plan.json", plan); s.write(tmp_path/"started.json", {})
    for seed in s.SEEDS: np.save(tmp_path/f"orders-{seed}.npy", np.zeros((1, 1), np.int64))
    pin = s.sha(tmp_path/"plan.json")
    done = {"status": "completed", "phase": "freeze", "plan_sha256": pin, "files": s.base.prep.manifest(tmp_path)}
    s.write(tmp_path/"completed.json", done)
    budget = s.base.Budget(tmp_path, __import__("time").monotonic())
    s.validate_plan(tmp_path/"plan.json", pin, budget)
    payload.write_text("changed")
    done["files"]["sources/"+name] = {"sha256": s.sha(payload), "bytes": payload.stat().st_size}
    (tmp_path/"completed.json").unlink(); s.write(tmp_path/"completed.json", done)
    with pytest.raises(ValueError, match="Snapshot source"):
        s.validate_plan(tmp_path/"plan.json", pin, budget)
