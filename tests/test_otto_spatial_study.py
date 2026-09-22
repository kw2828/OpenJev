"""Synthetic data and fixed-training mechanics; no empirical caches or episodes."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("spatial_study_test", ROOT / "scripts/study_otto_spatial.py")
S = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = S
SPEC.loader.exec_module(S)


def independent_center(beliefs, positions):
    result = np.zeros((len(beliefs), 105, 105), np.float64)
    for b, (qx, qy) in enumerate(positions):
        for x in range(53):
            for y in range(53):
                result[b, 52-int(qx)+x, 52-int(qy)+y] = beliefs[b, x, y]
    return result


def dataset(split="train"):
    """Two fully represented fictional teacher paths, lengths3 and2."""
    beliefs = np.zeros((5, 53, 53), np.float64)
    beliefs[0, 4, 7], beliefs[0, 29, 31] = .25, .75
    beliefs[1, 19, 3] = 5e-11
    dense = np.arange(1, 2810, dtype=np.float64).reshape(53, 53)
    dense[52, 0] = 0
    beliefs[3] = dense/dense.sum()
    beliefs[4, 52, 0] = .5
    positions = np.array([[0, 0], [0, 52], [26, 26], [52, 0], [52, 52]], np.int64)
    sensing = np.array([3., 3., 3., 4., 4.])
    centered = independent_center(beliefs, positions)
    mass = centered.reshape(5, -1).sum(axis=1)
    features = np.column_stack((centered.reshape(5, -1), mass*positions[:, 0]/52,
                                mass*positions[:, 1]/52, mass*sensing/5)).astype(np.float32)
    targets = np.array([3, 2, 1, 2, 1], np.float32)/64
    rows = []
    for i in range(5):
        regime, seed, hit, prefix, total = (("lambda3", 910001, 1, i, 3) if i < 3
                                           else ("lambda4", 920002, 2, i-3, 2))
        if split == "valid":
            seed += 20000
        x, y = positions[i].tolist()
        allowed = [a for a, ok in enumerate((x > 0, x < 52, y > 0, y < 52)) if ok]
        rows.append({"row_index": i, "episode_id": f"{split}:{regime}:{seed}:teacher", "stage": split,
                     "regime": regime, "seed": seed, "initial_hit": hit, "prefix_index": prefix,
                     "total_steps": total, "target": float(targets[i]),
                     "public": {"position": [x, y], "step": prefix, "hit": hit if prefix == 0 else 0,
                                "done": False, "valid_actions": allowed},
                     "posterior": {"sha256": hashlib.sha256(beliefs[i].tobytes()).hexdigest(),
                                   "mass": float(beliefs[i].sum())}})
    return {"features": features, "target": targets, "beliefs": beliefs,
            "positions": positions, "sensing_length": sensing}, rows


def test_center_batch_matches_independent_cell_scatter_and_final_float32_cast():
    arrays, _ = dataset()
    expected = independent_center(arrays["beliefs"], arrays["positions"])
    before = arrays["beliefs"].copy()
    exact = S.center_batch(arrays["beliefs"], arrays["positions"], dtype="float64")
    single = S.center_batch(arrays["beliefs"], arrays["positions"], dtype="float32")
    np.testing.assert_array_equal(exact, expected)
    np.testing.assert_array_equal(single, expected.astype(np.float32))
    np.testing.assert_array_equal(arrays["beliefs"], before)
    assert exact.dtype == np.float64 and single.dtype == np.float32
    assert not np.shares_memory(exact, before)
    assert exact[1].sum() == 5e-11 and exact[2].sum() == 0


@pytest.mark.parametrize("fault", ["negative", "nan", "dtype", "shape", "position", "outside", "empty", "batch"])
def test_center_batch_rejects_invalid_public_geometry(fault):
    arrays, _ = dataset()
    beliefs, positions = arrays["beliefs"], arrays["positions"]
    if fault == "negative":
        beliefs[0, 0, 0] = -1e-30
    elif fault == "nan":
        beliefs[0, 0, 0] = np.nan
    elif fault == "dtype":
        beliefs = beliefs.astype(np.float32)
    elif fault == "shape":
        beliefs = beliefs[:, :52]
    elif fault == "position":
        positions = positions.astype(np.float64)
    elif fault == "outside":
        positions[0, 0] = 53
    elif fault == "empty":
        beliefs, positions = beliefs[:0], positions[:0]
    else:
        beliefs, positions = np.repeat(beliefs[:1], 129, axis=0), np.repeat(positions[:1], 129, axis=0)
    with pytest.raises((ValueError, TypeError)):
        S.center_batch(beliefs, positions)


@pytest.mark.parametrize("split", ["train", "valid"])
def test_dataset_preserves_complete_zero_subfloor_and_original_target_rows(split):
    arrays, rows = dataset(split)
    before = {k: v.copy() for k, v in arrays.items()}
    metadata = copy.deepcopy(rows)
    S.validate_dataset(arrays, rows, split)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])
    assert rows == metadata and len(rows) == 5


@pytest.mark.parametrize("fault", ["hash", "mass", "position", "lambda", "target", "legacy_feature", "row_index",
                                   "prefix", "total_steps", "stage", "terminal", "allowed", "nonfinite", "extra",
                                   "reset_hit"])
def test_dataset_rejects_mismatched_saved_public_evidence(fault):
    arrays, rows = dataset()
    if fault == "hash":
        rows[0]["posterior"]["sha256"] = "0"*64
    elif fault == "mass":
        rows[0]["posterior"]["mass"] = .5
    elif fault == "position":
        rows[0]["public"]["position"] = [1, 1]
    elif fault == "lambda":
        arrays["sensing_length"][0] = 5.
    elif fault == "target":
        arrays["target"][0] = 5/64
        rows[0]["target"] = 5/64
    elif fault == "legacy_feature":
        arrays["features"][0, -1] += .01
    elif fault == "row_index":
        rows[1]["row_index"] = 0
    elif fault == "prefix":
        rows[1]["prefix_index"] = 0
    elif fault == "total_steps":
        rows[0]["total_steps"] = 4
    elif fault == "stage":
        rows[0]["stage"] = "eval"
    elif fault == "terminal":
        rows[0]["public"].update(done=True, hit=-2, valid_actions=[])
    elif fault == "allowed":
        rows[0]["public"]["valid_actions"] = [0, 1, 2, 3]
    elif fault == "nonfinite":
        arrays["beliefs"][0, 0, 0] = np.inf
    elif fault == "reset_hit":
        rows[0]["public"]["hit"] = 0
    else:
        arrays["hidden_source"] = np.ones((5, 2))
    with pytest.raises((ValueError, TypeError)):
        S.validate_dataset(arrays, rows, "train")


def test_incomplete_episode_cannot_be_relabelled_complete_rows():
    arrays, rows = dataset()
    arrays = {key: value[:-1].copy() for key, value in arrays.items()}
    with pytest.raises(ValueError):
        S.validate_dataset(arrays, rows[:-1], "train")


def test_production_allocation_and_exact_initial_final_prediction_closure():
    cfg = S.configuration()
    assert cfg["kinds"] == ["spatial", "neighbor_free", "cnn", "dense128", "statistics"]
    assert cfg["seeds"] == [10101, 10102, 10103]
    assert (cfg["train_rows"], cfg["valid_rows"], cfg["epochs"], cfg["batch_size"]) == (5589, 1109, 80, 128)
    assert (cfg["prediction_batch"], cfg["learning_rate"], cfg["gradient_clip"]) == (16, .001, 5)
    assert cfg["validation_selection"] is cfg["scalar_admission_gate"] is cfg["policy_evaluation"] is False
    assert cfg["alias_floor"] is None
    calls = S.expected_calls()
    assert calls == {"model_initialization": 15, "optimizer_initialization": 15, "checkpoint_export": 30,
                     "training_forward": 52800, "backward": 52800, "optimizer_update": 52800,
                     "deployment_construction": 15, "parity_restore": 15,
                     "numpy_prediction": 6300, "torch_prediction": 6300}
    assert sum(calls.values()) == 171090
    names = S.payload_names()
    assert len(names) == 56
    assert {n for n in names if n.endswith(".npz")} == {
        f"{phase}-{kind}-{seed}.npz" for seed in (10101, 10102, 10103)
        for kind in cfg["kinds"] for phase in ("initial", "final", "predictions")}
    assert not any("best" in n or "eval" in n or "disposable" in n for n in names)
    assert S.limits() == {"native_seconds": 7200, "rss_bytes": 8*1024**3, "output_bytes": 2*1024**3,
                          "optimizer_updates": 52800, "native_steps": 0, "native_resets": 0,
                          "external_model_calls": 0}
    assert S.AUDIT_LIMITS == {"seconds": 1800, "rss_bytes": 4*1024**3, "output_bytes": 128*1024**2}


def jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def tiny_fit(tmp_path_factory):
    """Two epochs of fabricated rows exercise the actual five-family loop."""
    import torch

    from openjev.research import otto_spatial_value

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    out = tmp_path_factory.mktemp("spatial-synthetic-training")
    run = S.Run(SimpleNamespace(output=out))
    cfg = S.configuration()
    cfg.update(seeds=[10101], epochs=2, batch_size=3, prediction_batch=2, train_rows=5, valid_rows=5)
    run.plan = {"configuration": cfg, "expected_calls": S.expected_calls(cfg), "limits": S.limits()}
    run.np, run.torch, run.model = np, torch, otto_spatial_value
    run.data, run.evidence = {}, {}
    for split in ("train", "valid"):
        arrays, rows = dataset(split)
        run.evidence[split] = S.validate_dataset(arrays, rows, split)
        run.data[split] = {k: v for k, v in arrays.items() if k != "features"}
    run.c0 = float(np.float32(np.mean(run.data["train"]["target"], dtype=np.float64)))
    run.check = lambda: None
    try:
        for kind in cfg["kinds"]:
            run.fit(kind, 10101)
        yield run
    finally:
        torch.set_num_threads(previous_threads)


def test_real_tiny_loop_pairs_orders_and_records_each_row_weighted_loss(tiny_fit):
    run = tiny_fit
    order_rows = jsonlines(run.out/"epoch-orders.jsonl")
    updates = jsonlines(run.out/"updates.jsonl")
    curves = jsonlines(run.out/"fit-curves.jsonl")
    rng = np.random.default_rng(30101)
    orders = {epoch: rng.permutation(5).astype(np.int64) for epoch in (1, 2)}
    assert len(order_rows) == len(curves) == 10 and len(updates) == 20
    for row in order_rows:
        expected = orders[row["epoch"]]
        assert row["order"] == expected.tolist()
        assert row["sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()
    for curve in curves:
        group = [u for u in updates if (u["fit_id"], u["epoch"]) == (curve["fit_id"], curve["epoch"])]
        assert [u["rows"] for u in group] == [3, 2]
        assert [u["update_index"] for u in group] == [2*curve["epoch"]-1, 2*curve["epoch"]]
        expected_order = orders[curve["epoch"]]
        for i, row in enumerate(group):
            ids = expected_order[i*3:i*3+row["rows"]]
            assert row["batch_indices_sha256"] == hashlib.sha256(ids.tobytes()).hexdigest()
            assert np.isfinite(row["loss"]) and np.isfinite(row["gradient_norm"])
        expected_loss = sum(u["rows"]*u["loss"] for u in group)/5
        assert curve["training_mse_normalized"] == pytest.approx(expected_loss, abs=1e-15)


def test_real_tiny_loop_preserves_initial_function_and_fixed_final_checkpoints(tiny_fit):
    run = tiny_fit
    assert run.receipt["completed_fits"] == 5 and run.pending == []
    assert [f["kind"] for f in run.fits] == list(S.KINDS)
    assert {k: v["attempted"] for k, v in run.calls.items()} == run.plan["expected_calls"]
    assert all(v["attempted"] == v["returned"] for v in run.calls.values())
    initial = {}
    for fit in run.fits:
        kind = fit["kind"]
        assert fit["optimizer_steps_before"] == {} and set(fit["optimizer_steps"].values()) == {4}
        assert fit["updates"] == 4 and fit["epochs"] == 2
        assert fit["c0_float32"] == run.c0 and fit["parity_rows"] == 10 and fit["parity_passed"]
        assert fit["training_seconds"] <= fit["fit_seconds"]
        assert fit["deployment_setup_seconds"]+fit["prediction_seconds"] == pytest.approx(fit["diagnostic_seconds"], abs=1e-12)
        with np.load(run.out/f"initial-{kind}-10101.npz", allow_pickle=False) as archive:
            initial[kind] = {k: archive[k] for k in archive.files}
        assert np.count_nonzero(initial[kind]["readout_weight_1"]) == 0
        assert np.count_nonzero(initial[kind]["readout_bias_1"]) == 0
        assert float(initial[kind]["c0"]) == run.c0
        with np.load(run.out/f"final-{kind}-10101.npz", allow_pickle=False) as final:
            assert any(not np.array_equal(final[k], initial[kind][k]) for k in final.files if "weight" in k or "bias" in k)
    for key in initial["spatial"]:
        if key != "kind":
            np.testing.assert_array_equal(initial["spatial"][key], initial["neighbor_free"][key])


def test_real_tiny_final_views_use_original_float64_geometry_and_all_rows(tiny_fit):
    run = tiny_fit
    rows = jsonlines(run.out/"parity.jsonl")
    assert len(rows) == 30
    for row in rows:
        data, offset, count = run.data[row["split"]], row["offset"], row["rows"]
        selected = slice(offset, offset+count)
        exact = independent_center(data["beliefs"][selected], data["positions"][selected])
        assert row["input_sha256"]["centered"] == hashlib.sha256(exact.tobytes()).hexdigest()
        assert row["passed"] and row["rows"] == (1 if offset == 4 else 2)
    for fit in run.fits:
        with np.load(run.out/f"predictions-{fit['kind']}-10101.npz", allow_pickle=False) as archive:
            assert set(archive.files) == {"train_numpy", "train_torch64", "valid_numpy", "valid_torch64"}
            for split in ("train", "valid"):
                prediction = archive[split+"_numpy"]
                assert prediction.shape == (5,) and prediction.dtype == np.float64
                np.testing.assert_allclose(prediction, archive[split+"_torch64"], atol=1e-8, rtol=1e-10)
                error = prediction-run.data[split]["target"].astype(np.float64)
                assert fit["metrics"][split]["mse_normalized"] == pytest.approx(float(np.mean(error**2)), abs=1e-15)
                assert fit["metrics"][split]["mae_physical"] == pytest.approx(64*float(np.mean(abs(error))), abs=1e-13)


def test_actual_failed_operation_keeps_attempt_and_pending_identity(tmp_path):
    run = S.Run(SimpleNamespace(output=tmp_path))
    run.plan = {"expected_calls": {"training_forward": 1}, "limits": S.limits()}
    run.context = {"fit_id": "fabricated", "epoch": 1, "batch": 0}
    run.check = lambda: None

    def fail():
        raise RuntimeError("synthetic forward failure")

    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        run.call("training_forward", fail)
    journal = jsonlines(tmp_path/"work.jsonl")
    assert len(journal) == 1 and journal[0]["event"] == "attempt"
    assert run.calls["training_forward"]["attempted"] == 1
    assert run.calls["training_forward"]["returned"] == 0
    assert run.pending == [{k: v for k, v in journal[0].items() if k != "event"}]
    with pytest.raises(ValueError, match="cap before"):
        run.call("training_forward", lambda: None)


def test_bad_external_plan_pin_rejected_before_decode(tmp_path, monkeypatch):
    path = tmp_path/"plan.json"
    path.write_text("not json")
    monkeypatch.setattr(S, "read", lambda _: pytest.fail("unauthenticated plan decoded"))
    with pytest.raises(ValueError, match="external plan pin"):
        S.authenticate(SimpleNamespace(plan=path, plan_sha256="0"*64))


def test_qualification_weights_cannot_be_added_to_empirical_input_roles(tmp_path):
    plan = {"version": S.VERSION, "status": "frozen_before_execution", "mode": "study",
            "configuration": S.configuration(), "limits": S.limits(), "expected_calls": S.expected_calls(),
            "independent_audit_limits": S.AUDIT_LIMITS,
            "sources": {**dict.fromkeys(S.NEW, "0"*64), S.BASE: S.BASE_PIN},
            "inputs": {key: {} for key in ("capacity_plan", "capacity_receipt", "capacity_terminal", "capacity_audit",
                                           "train_data", "train_rows", "valid_data", "valid_rows", "qualification_weights")}}
    path = tmp_path/"plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="exact prior input roles"):
        S.authenticate(SimpleNamespace(plan=path, plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


@pytest.mark.parametrize("existing", ["file", "symlink"])
def test_freezer_refuses_nonexclusive_path_before_lineage_reads(tmp_path, monkeypatch, existing):
    spec = importlib.util.spec_from_file_location("spatial_freezer_test", ROOT/"scripts/freeze_otto_spatial.py")
    freezer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = freezer
    spec.loader.exec_module(freezer)
    target = tmp_path/"plan.json"
    target.write_text("preserve")
    output = target
    if existing == "symlink":
        link = tmp_path/"link"
        link.symlink_to(tmp_path, target_is_directory=True)
        output = link/"new.json"
    monkeypatch.setattr(freezer.S, "regular", lambda _: pytest.fail("lineage read before exclusive output guard"))
    with pytest.raises(ValueError, match="exclusive"):
        freezer.freeze(SimpleNamespace(output=output))
    assert target.read_text() == "preserve"


def test_aggregate_keeps_all_fifteen_models_without_scalar_selection():
    fits = []
    for seed in S.SEEDS:
        for kind in S.KINDS:
            value = float(len(fits)+1)
            metric = {"mse_normalized": value, "mae_physical": 2*value,
                      "negative_predictions": int(value), "minimum_normalized": -value,
                      "maximum_normalized": value}
            fits.append({"fit_id": f"{kind}@{seed}", "kind": kind, "seed": seed,
                         "metrics": {"train": dict(metric), "valid": dict(metric)},
                         "fit_seconds": value, "diagnostic_seconds": value/2})
    result = S.aggregate(fits)
    assert len(result["metrics"]) == 15 and result["fits"] == [r["fit_id"] for r in fits]
    assert result["scalar_admission_gate"] is result["alias_floor"] is None
    assert result["learned_architecture_advantage_established"] is False
    assert result["family_means"]["spatial"]["train"]["mse_normalized"] == 6
    with pytest.raises(ValueError, match="all fifteen"):
        S.aggregate(fits[:-1])
    with pytest.raises(ValueError, match="all fifteen"):
        S.aggregate(fits[::-1])


def test_execute_preserves_binding_failure_without_training_or_completed_receipt(tmp_path):
    out = tmp_path/"failed"
    run = S.Run(SimpleNamespace(output=out))

    def fail():
        raise RuntimeError("synthetic bind failure")

    run.bind = fail
    with pytest.raises(RuntimeError, match="synthetic bind failure"):
        run.execute()
    failure = json.loads((out/"failed.json").read_text())
    assert failure["status"] == "failed" and failure["completed_fits"] == 0
    assert failure["calls"] == {} and failure["pending"] == []
    assert not (out/"receipt.json").exists()
