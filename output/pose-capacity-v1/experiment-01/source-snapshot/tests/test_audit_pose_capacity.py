"""Hand-built arrays and interval traces only; no fits, model calls or data."""
import copy
import gzip
import json
import math

import audit_pose_capacity as audit
import numpy as np
import pytest
from test_audit_pose_crossfit import constant_fixture


def one_window(*, capped=False):
    record, data, settings = constant_fixture(capped=capped)
    data = tuple(x.reshape(1, 2, *x.shape[2:]) for x in data)
    record["input_shape"] = [1, 2, 3]
    return record, data, settings


@pytest.mark.parametrize("capped", [False, True])
def test_independent_single_window_certificate_and_retained_cap(capped):
    record, data, settings = one_window(capped=capped)
    arrays, results = audit.verify_records([record], *data, **settings)
    assert arrays["certified"].tolist() == [not capped]
    assert results[0]["certified"] is (not capped)
    assert arrays["continuous_position_mse"].tolist() == [0.]
    assert arrays["continuous_rotation_lower_mse"][0] <= (math.pi / 4) ** 2
    assert arrays["continuous_rotation_upper_mse"][0] >= (math.pi / 4) ** 2
    assert arrays["hard_step"][0, 1] == 0
    assert arrays["hard_window"][0, 1] > .5
    assert arrays["calls"].tolist() == [3 if capped else 5]


def test_negative_position_normal_equation_values_are_valid():
    record, data, settings = one_window()
    fp, fr, sp, sr, tp, tr = data
    tp = -tp
    record["alpha"][0] = 0.
    record["position"].update(numerator=-12., unclipped_alpha=-.5, objective_m2=3.)
    arrays, _ = audit.verify_records([record], fp, fr, sp, sr, tp, tr, **settings)
    assert arrays["continuous_position_mse"][0] == 3.


@pytest.mark.parametrize("damage", ["objective", "bounds", "leaves", "selected", "certified", "coverage"])
def test_corrupt_saved_constant_trace_rejected(damage):
    record, data, settings = one_window(capped=damage == "certified")
    if damage == "objective":
        record["rotation"]["evaluations"][0]["objective"] += .01
    elif damage == "bounds":
        record["rotation"]["lower_bound"] += .01
    elif damage == "leaves":
        record["rotation"]["leaves"] = [1]
    elif damage == "selected":
        record["alpha"][1] = .25
    elif damage == "certified":
        record["rotation"]["certified"] = True
    with pytest.raises(ValueError):
        audit.verify_records([] if damage == "coverage" else [record], *data, **settings)


@pytest.mark.parametrize("damage", ["dtype", "shape", "nan", "value", "missing", "extra", "calls", "certified"])
def test_saved_array_corruption_rejected(damage):
    record, data, settings = one_window()
    expected, _ = audit.verify_records([record], *data, **settings)
    saved = {k: v.copy() for k, v in expected.items()}
    audit.verify_arrays(saved, expected)
    if damage == "dtype":
        saved["alpha"] = saved["alpha"].astype(np.float32)
    elif damage == "shape":
        saved["alpha"] = saved["alpha"].reshape(2)
    elif damage == "nan":
        saved["alpha"][0, 0] = np.nan
    elif damage == "value":
        saved["hard_window"][0, 0] += .01
    elif damage == "missing":
        saved.pop("alpha")
    elif damage == "extra":
        saved["unaccounted"] = np.zeros(1)
    elif damage == "calls":
        saved["calls"][0] += 1
    else:
        saved["certified"][0] = False
    with pytest.raises(ValueError):
        audit.verify_arrays(saved, expected)


def test_inclusive_threshold_and_unresolved_interval():
    exact = audit.capacity_comparison(.81, .81, 1.)
    assert exact["observed_oracle_reaches_threshold"] and not exact["excluded_even_optimistically"]
    assert audit.capacity_comparison(.810001, .9, 1.)["excluded_even_optimistically"]
    assert audit.capacity_comparison(.8, .82, 1.)["interval_status"] == "unresolved"
    zero = audit.capacity_comparison(0., 0., 0.)
    assert not zero["observed_oracle_reaches_threshold"]
    assert zero["optimistic_improvement_percent"] is None


def test_projected_bounds_do_not_claim_float32_impossibility():
    value = audit.capacity_comparison(.9, 1., 1., projected=True)
    assert value["projected_lower_below_reported_threshold"] is False
    assert value["projected_upper_below_reported_threshold"] is False
    assert "excluded_even_optimistically" not in value
    assert "interval_status" not in value


def test_mean_squared_errors_are_pooled_before_sqrt_all_controls_retained():
    ids = np.array([[0, 0], [1, 0]], np.int64)
    arrays = {panel: {seed: {"hard_window": np.full((2, 2), mse), "hard_step": np.full((2, 2), mse / 2),
              "continuous_position_mse": np.full(2, mse), "continuous_rotation_lower_mse": np.full(2, mse),
              "continuous_rotation_upper_mse": np.full(2, mse + 1e-8)}
              for seed, mse in zip(audit.SEEDS, (.01, .04, .09), strict=True)} for panel in audit.PANELS}
    summary = {group: {panel: {v: {e: {"rmse": 1., "mse": 1.} for e in audit.ENDPOINTS} for v in variants}
                      for panel in audit.PANELS}
               for group, variants in (("families", audit._audit.VARIANTS), ("references", audit._audit.REFERENCES))}
    families, comparisons = audit.aggregate(arrays, {p: ids for p in audit.PANELS}, summary)
    assert math.isclose(families["test_sin"]["continuous"]["position"]["lower_rmse"], math.sqrt(.14 / 3))
    assert not math.isclose(families["test_sin"]["continuous"]["position"]["lower_rmse"], .2)
    for panel in audit.PANELS:
        for method in comparisons[panel].values():
            for endpoint in method.values():
                assert endpoint["control_count"] == 33
                assert set(endpoint["controls"]) == set(audit.CONTROLS)


@pytest.mark.parametrize("damage", ["panel", "seed", "order", "source", "start", "bool", "seconds"])
def test_trace_window_provenance_precedes_numerical_checks(monkeypatch, damage):
    ids = np.array([[0, 7]], np.int64)
    payload = {"panel": "test_sin", "seed": 1101,
               "records": [{"window": 0, "source_id": 0, "window_start": 7, "optimization": {}, "seconds": .1}]}
    record = payload["records"][0]
    if damage in ("panel", "seed"):
        payload[damage] = "wrong"
    elif damage == "order":
        record["window"] = 1
    elif damage == "source":
        record["source_id"] = 1
    elif damage == "start":
        record["window_start"] = 8
    elif damage == "bool":
        record["window"] = False
    else:
        record["seconds"] = -1.
    monkeypatch.setattr(audit, "verify_records", lambda *a, **k: pytest.fail("must reject before certificate arithmetic"))
    with pytest.raises(ValueError):
        audit.verify_trace(payload, "test_sin", 1101, ids, ())


def test_exact_twenty_members_and_frozen_repair_source():
    assert len(audit.expected_members()) == 20
    assert audit.sha(audit.ROOT / "scripts/audit_pose_crossfit_repair.py") == audit.REPAIR_SHA256
    assert audit.sha(audit.ROOT / "tests/test_audit_pose_crossfit_repair.py") == audit.REPAIR_TEST_SHA256
    assert audit._audit.close_number is audit._repair.close_number


def test_synthetic_trace_preserves_caps_and_times(monkeypatch):
    ids = np.array([[0, 7]], np.int64)
    payload = {"panel": "test_sin", "seed": 1101,
               "records": [{"window": 0, "source_id": 0, "window_start": 7, "optimization": {}, "seconds": .1}]}
    monkeypatch.setattr(audit, "verify_records", lambda *a, **k: ({"certified": np.array([False])}, [{"certified": False}]))
    arrays, verified, seconds = audit.verify_trace(copy.deepcopy(payload), "test_sin", 1101, ids, ())
    assert np.array_equal(arrays["ids"], ids) and not verified[0]["certified"] and seconds == .1


def test_squared_canonical_control_threshold_not_rounded_rmse():
    result = audit.capacity_comparison(.81, .81, 1.00000001, control_mse=1.)
    assert result["threshold_mse"] == .81 and result["observed_oracle_reaches_threshold"]
    assert audit.capacity_comparison(.810000001, .82, 1.00000001, control_mse=1.)["excluded_even_optimistically"]


def synthetic_outer(tmp_path, monkeypatch, *, certified=True):
    """Real20-file outer shape; numerical certificates and parent auth are stubbed."""
    experiment, out = tmp_path / "experiment", tmp_path / "report"
    run = experiment / "run-01"
    run.mkdir(parents=True)
    parent_run, parent_report = tmp_path / "parent-run", tmp_path / "parent-report"
    parent_run.mkdir(); parent_report.mkdir()
    summary = {group: {panel: {v: {e: {"rmse": 1., "mse": 1.} for e in audit.ENDPOINTS} for v in variants}
                      for panel in audit.PANELS}
               for group, variants in (("families", audit._audit.VARIANTS), ("references", audit._audit.REFERENCES))}
    parent = {"run": parent_run, "report": parent_report, "summary": summary,
              "members": {}, "report_members": {}}
    protocol = {"sources": {"synthetic": "synthetic"}}
    monkeypatch.setattr(audit, "validate_inputs", lambda *a: (protocol, parent))
    ids = np.column_stack((np.repeat(np.arange(10), 16), np.tile(np.arange(16), 10))).astype(np.int64)
    arrays = {"ids": ids, "hard_window": np.full((160, 2), .25), "hard_step": np.full((160, 2), .20),
              "continuous_position_mse": np.full(160, .1), "continuous_rotation_lower_mse": np.full(160, .1),
              "continuous_rotation_upper_mse": np.full(160, .1 + (1e-8 if certified else .01)),
              "alpha": np.full((160, 2), .5), "calls": np.full(160, 3, np.int64),
              "certified": np.full(160, certified, bool)}
    verified = [{"gap": 1e-8 if certified else .01, "certified": certified,
                 "status": "tolerance_reached" if certified else "evaluation_cap"} for _ in range(160)]
    monkeypatch.setattr(audit, "load_inputs", lambda *a: (ids, ()))
    monkeypatch.setattr(audit, "verify_trace", lambda *a: (arrays, verified, .1))
    audit.write_json(run / "started.json", {"protocol_sha256": "protocol"})
    rows = []
    for panel in audit.PANELS:
        for seed in audit.SEEDS:
            prefix = f"{panel}-{seed}"
            trace = run / f"{prefix}-traces.json.gz"
            with gzip.open(trace, "wt") as stream:
                json.dump({}, stream)
            bounds = run / f"{prefix}-bounds.npz"
            np.savez_compressed(bounds, **arrays)
            row = {"panel": panel, "seed": seed, "trace_sha256": audit.sha(trace), "bounds_sha256": audit.sha(bounds),
                   "constant_searches": 160, "objective_evaluations": 480, "certified_searches": 160 if certified else 0,
                   "search_seconds": .1, "wall_seconds": .2}
            audit.write_json(run / f"{prefix}-row.json", row)
            rows.append(row)
    done = {"status": "completed", "protocol_sha256": "protocol", "rows": rows, "constant_searches": 960,
            "objective_evaluations": 2880, "certified_searches": 960 if certified else 0, "wall_seconds": 1.3,
            "files": audit.members(run)}
    audit.write_json(run / "completed.json", done)
    return experiment, out, audit.sha(run / "completed.json")


@pytest.mark.parametrize("certified", [False, True])
def test_real_shaped_outer_seals_twenty_files_and_keeps_capped_result(tmp_path, monkeypatch, certified):
    experiment, out, digest = synthetic_outer(tmp_path, monkeypatch, certified=certified)
    result = audit.audit(experiment, out, protocol_sha256="protocol", completed_sha256=digest)
    assert result["status"] == "completed" and result["all_certified"] is certified
    assert result["verified_searches"] == 960 and result["no_qualification_gate"]
    assert len(result["execution_members"]) == 20 and set(result["files"]) == {"summary.json", "window-errors.npz"}
    summary = audit.read_json(out / "summary.json")
    assert summary["certificate_summary"]["all_certified"] is certified
    assert summary["certificate_summary"]["precise_projected_continuous_claims_available"] is certified
    assert summary["counts"]["new_constant_optimizations_by_audit"] == 0
    assert "qualification_passed" not in result and "continuation_gate" not in summary
    with pytest.raises(FileExistsError):
        audit.audit(experiment, out, protocol_sha256="protocol", completed_sha256=digest)


@pytest.mark.parametrize("damage", ["extra", "missing", "unsealed", "order", "counter", "time"])
def test_outer_corruption_preserves_failure_without_completion(tmp_path, monkeypatch, damage):
    experiment, out, digest = synthetic_outer(tmp_path, monkeypatch)
    run = experiment / "run-01"
    if damage == "extra":
        (run / "extra.txt").write_text("unaccounted")
    elif damage == "missing":
        (run / "test_sin-1101-row.json").unlink()
    elif damage == "unsealed":
        (run / "started.json").write_text("{}")
    else:
        done = audit.read_json(run / "completed.json")
        if damage == "order":
            done["rows"] = done["rows"][::-1]
        elif damage == "counter":
            done["objective_evaluations"] += 1
        else:
            done["wall_seconds"] = .1
        (run / "completed.json").write_text(json.dumps(done))
        digest = audit.sha(run / "completed.json")
    with pytest.raises(ValueError):
        audit.audit(experiment, out, protocol_sha256="protocol", completed_sha256=digest)
    assert audit.read_json(out / "failed.json")["status"] == "failed"
    assert not (out / "receipt.json").exists()


def test_runner_wrong_external_protocol_refuses_before_output_or_optimizer(tmp_path, monkeypatch):
    import run_pose_capacity as runner
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "protocol.json").write_text("{}")
    monkeypatch.setattr(runner, "fit_constant", lambda *a, **k: pytest.fail("must not optimize"))
    monkeypatch.setattr(runner, "sources", lambda: pytest.fail("digest checked before source traversal"))
    with pytest.raises(ValueError, match="external frozen protocol digest"):
        runner.run(experiment, "wrong digest")
    assert not (experiment / "run-01").exists()


@pytest.mark.parametrize("failed_receipt_write", [False, True])
def test_runner_retains_returned_searches_and_original_exception(tmp_path, monkeypatch, failed_receipt_write):
    import run_pose_capacity as runner
    experiment, parent = tmp_path / "experiment", tmp_path / "parent"
    experiment.mkdir(); (parent / "run-01").mkdir(parents=True)
    ids = np.column_stack((np.repeat(np.arange(10), 16), np.tile(np.arange(16), 10))).astype(np.int64)
    p = np.zeros((160, 25, 3), np.float32)
    r = np.broadcast_to(np.eye(3, dtype=np.float32), (160, 25, 3, 3)).copy()
    np.savez_compressed(parent / "run-01/test_sin-targets.npz", p=p, R=r, ids=ids)
    monkeypatch.setattr(runner, "validate_protocol", lambda *a: {"parent_experiment": str(parent)})
    monkeypatch.setattr(runner.torch, "set_num_threads", lambda *a: None)
    monkeypatch.setattr(runner.parent_audit._previous, "load_prediction", lambda *a, **k: (p, r, None))
    attempted = []
    original_error = RuntimeError("injected third numerical search failure")
    def fake_fit(*args, **kwargs):
        assert kwargs == {"tolerance": 1e-7, "max_evaluations": 4095}
        assert args[0].shape == (1, 25, 3)
        attempted.append(len(attempted))
        if len(attempted) == 3:
            raise original_error
        return {"synthetic_returned_record": len(attempted)}
    monkeypatch.setattr(runner, "fit_constant", fake_fit)
    if failed_receipt_write:
        writer = runner.write_json
        def fail_only_failure(path, value):
            if path.name == "failed.json":
                raise OSError("injected receipt failure")
            return writer(path, value)
        monkeypatch.setattr(runner, "write_json", fail_only_failure)
    with pytest.raises(RuntimeError) as caught:
        runner.run(experiment, "synthetic external digest")
    assert caught.value is original_error and len(attempted) == 3
    run = experiment / "run-01"
    trace_path = run / "partial-test_sin-1101-traces.json.gz"
    with gzip.open(trace_path, "rt") as stream:
        partial = json.load(stream)
    assert partial["panel"] == "test_sin" and partial["seed"] == 1101
    assert [x["window"] for x in partial["records"]] == [0, 1]
    assert [x["optimization"]["synthetic_returned_record"] for x in partial["records"]] == [1, 2]
    assert all(x["seconds"] >= 0 for x in partial["records"])
    assert not (run / "completed.json").exists()
    if failed_receipt_write:
        assert "Could not retain failure receipt" in " ".join(original_error.__notes__)
    else:
        failure = audit.read_json(run / "failed.json")
        assert failure["partial_trace"] == {"path": trace_path.name, "sha256": audit.sha(trace_path), "returned_searches": 2}
        assert failure["protocol_sha256"] == "synthetic external digest"
