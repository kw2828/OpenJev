"""Fabricated conditional TEST audit checks; no empirical payload access."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
import time
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openjev.research import otto_query_memory as memory
from openjev.research import otto_query_memory_gate as gate
from openjev.research import otto_query_memory_metrics as metrics
from openjev.research import otto_query_memory_model as composed

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_test_independent_query_memory_test_audit", ROOT / "scripts/audit_otto_query_memory_test.py")
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def identities():
    rows = []
    for regime, first in (("lambda3", 307000001), ("lambda4", 308000001)):
        for case in range(6):
            for arm in ("analytic", "neural", "period4_hold"):
                rows.append({"stage": "test", "episode_id": f"{regime}-{case}-{arm}", "episode_index": len(rows),
                             "regime": regime, "case": case, "seed": first + case, "arm": arm})
    return rows


def census(lengths=None):
    lengths = lengths or [1, 4, 5, 8, 9, 33, 65, 17, 32] * 4
    offsets = np.array([0, *np.cumsum(lengths)], np.int64)
    total = int(offsets[-1])
    rng = np.random.default_rng(918)
    features = rng.normal(size=(total, 31)).astype(np.float32)
    raw_q = rng.normal(size=(total, 4)).astype(np.float32)
    legal = rng.random((total, 4)) > .4
    legal[:, 0] = True
    correction = np.zeros(total, np.bool_)
    for low, high in pairwise(offsets):
        steps = np.arange(high - low)
        features[low:high, 15], features[low:high, 16] = steps / 2188, steps % 4 / 2188
        features[low:high, 17] = 1
        correction[low:high] = steps % 4 == 0
    return {"features": features, "raw_q": raw_q, "legal": legal,
            "actions": np.zeros(total, np.int64), "correction": correction, "episode_offsets": offsets}


@pytest.mark.parametrize("period", (4, 8))
def test_independent_full_test_scopes_match_qualified_metrics_with_unsupported_paths(period):
    flat, ids = census(), identities()
    target, legal, offsets = flat["raw_q"], flat["legal"], flat["episode_offsets"]
    scores = target[:, ::-1].copy()
    prior = np.full_like(target, np.nan)
    for low, high in pairwise(offsets):
        steps = np.arange(high - low)
        prior[low:high][(steps > 0) & (steps % period == 0)] = scores[low:high][(steps > 0) & (steps % period == 0)]
    actual = audit.scalar_report(np, ids, target, legal, scores, prior, offsets, "trace_delta", audit.SEEDS[0], period)
    episodes = [metrics.episode_metrics(identity, period, target[low:high], legal[low:high], scores[low:high],
                                       prequery_forecast=prior[low:high])
                for identity, low, high in zip(ids, offsets[:-1], offsets[1:], strict=True)]
    expected = metrics.aggregate_episodes(episodes, family="trace_delta", fit_seed=audit.SEEDS[0], expected_identities=ids)
    audit.close_equal(actual, expected, "independent TEST scalar report")
    assert actual["scopes"]["later"]["overall"]["zero_support_episode_ids"]
    assert set(actual["scopes"]["full"]["overall"]["by_age"]) == {str(age) for age in range(1, period)}
    assert "by_age" not in actual["scopes"]["common_full"]["overall"]


def test_common_support_keeps_steps_five_to_seven_and_excludes_every_p4_query():
    flat, ids = census([17] * 36), identities()
    reports = [audit.scalar_report(np, ids, flat["raw_q"], flat["legal"], flat["raw_q"], flat["raw_q"],
                                  flat["episode_offsets"], "pretrained", audit.SEEDS[0], period) for period in (4, 8)]
    for report in reports:
        scopes = report["scopes"]
        assert scopes["common_initial"]["overall"]["rows"] == 6 * 36
        assert scopes["common_later"]["overall"]["rows"] == 6 * 36
        assert scopes["common_full"]["overall"]["rows"] == 12 * 36
    assert reports[0]["scopes"]["initial"]["overall"]["rows"] == 3 * 36
    assert reports[1]["scopes"]["initial"]["overall"]["rows"] == 7 * 36
    for scope in ("common_full", "common_initial", "common_later"):
        audit.close_equal(reports[0]["scopes"][scope], reports[1]["scopes"][scope], "identical common support")


@pytest.mark.parametrize("period", (4, 8))
def test_history_only_changes_age_and_hides_every_unobserved_target(period):
    flat = census()
    before = {name: value.tobytes() for name, value in flat.items()}
    result = audit.expected_history(np, flat, identities(), period)
    assert all(value.tobytes() == before[name] for name, value in flat.items())
    assert np.array_equal(result["features"][:, [i for i in range(31) if i != 16]],
                          flat["features"][:, [i for i in range(31) if i != 16]])
    assert np.isnan(result["query_scores"][~result["query_mask"]]).all()
    assert result["query_scores"][result["query_mask"]].tobytes() == flat["raw_q"][result["query_mask"]].tobytes()
    for low, high in pairwise(result["episode_offsets"]):
        steps = np.arange(high - low)
        assert np.array_equal(result["features"][low:high, 16], (steps % period / 2188).astype(np.float32))
        assert np.array_equal(result["query_mask"][low:high], steps % period == 0)
    changed = copy.deepcopy(flat)
    changed["features"][2, 16] = .5
    with pytest.raises(ValueError, match="public observation clocks"):
        audit.expected_history(np, changed, identities(), period)


def gate_reports(candidate=9., support=4):
    reports = []
    for period in audit.PERIODS:
        for seed in audit.SEEDS:
            for family in audit.VIEWS:
                value = candidate if family == "trace_delta" else 10.
                leaf = {"episodes": 18, "declared_case_count": 6, "case_weighted_raw_gap": value,
                        "episode_weighted_raw_gap": value, "supported_case_count": support}
                reports.append({"family": family, "seed": seed, "query_period": period, "stage": "test", "episodes": 36,
                    "scopes": {scope: {"by_regime": {regime: copy.deepcopy(leaf) for regime in ("lambda3", "lambda4")}}
                               for scope in ("full", "later")}})
    return reports


def condition(result, suffix):
    return next(row for row in result["conditions"] if row["name"] == suffix)


def test_all25_test_conditions_require_technical_closure_and_exact_ten_percent_boundary():
    reports = gate_reports()
    result = audit.independent_gate(reports)
    assert result["passed_conditions"] == 24 and result["total_conditions"] == 25 and not result["passed"]
    assert audit.independent_gate(reports, technical_complete=True)["passed"]
    assert not audit.independent_gate(gate_reports(math.nextafter(9., math.inf)), technical_complete=True)["passed"]
    for support in (3, 4):
        result = audit.independent_gate(gate_reports(support=support), technical_complete=True)
        assert result["passed"] is (support == 4)


def test_one_shifted_panel_or_paired_seed_regression_cannot_be_hidden_by_other_panels():
    reports = gate_reports(8.)
    row = next(r for r in reports if r["family"] == "trace_delta" and r["seed"] == audit.SEEDS[0] and r["query_period"] == 8)
    row["scopes"]["later"]["by_regime"]["lambda4"]["case_weighted_raw_gap"] = 10.1
    result = audit.independent_gate(reports, technical_complete=True)
    assert condition(result, "lambda4:P8:later_gap_10pct")["passed"]
    assert not condition(result, f"lambda4:P8:seed_{audit.SEEDS[0]}_nonregression")["passed"]
    assert not result["passed"]
    reports = gate_reports(0.)
    for row in reports:
        if row["family"] == "pretrained":
            row["scopes"]["later"]["by_regime"]["lambda3"]["case_weighted_raw_gap"] = 0.
    assert not condition(audit.independent_gate(reports, technical_complete=True), "lambda3:P4:later_gap_10pct")["passed"]


def test_gate_means_controls_before_selecting_strongest_and_keeps_seed_guards():
    reports = gate_reports(5.)
    for row in reports:
        if row["family"] in audit.CONTROLS:
            index = audit.SEEDS.index(row["seed"])
            values = (1., 10., 10.) if row["family"] == "pretrained" else (10., 1., 10.)
            for regime in ("lambda3", "lambda4"):
                row["scopes"]["later"]["by_regime"][regime]["case_weighted_raw_gap"] = values[index]
    result = audit.independent_gate(reports, technical_complete=True)
    assert condition(result, "lambda3:P4:later_gap_10pct")["best_control"] == 7.
    assert condition(result, "lambda3:P4:later_gap_10pct")["passed"]
    assert not condition(result, f"lambda3:P4:seed_{audit.SEEDS[0]}_nonregression")["passed"]


@pytest.mark.parametrize("defect", ("missing", "period", "duplicate", "nonfinite", "support"))
def test_test_gate_fails_closed_on_missing_mismatched_or_invalid_reports(defect):
    reports = gate_reports()
    if defect == "missing":
        reports.pop()
    elif defect == "period":
        reports[0]["query_period"] = 16
    elif defect == "duplicate":
        reports[0] = copy.deepcopy(reports[1])
    elif defect == "nonfinite":
        reports[0]["scopes"]["later"]["by_regime"]["lambda3"]["case_weighted_raw_gap"] = float("nan")
    else:
        reports[0]["scopes"]["later"]["by_regime"]["lambda3"]["supported_case_count"] = 3
    with pytest.raises(ValueError):
        audit.independent_gate(reports)


def synthetic_chunk(lengths, start, period):
    features = torch.full((len(lengths), 32, 31), float("nan"))
    scores = torch.full((len(lengths), 32, 4), float("nan"))
    mask = torch.zeros(len(lengths), 32, dtype=torch.bool)
    local = [max(0, min(32, length - start)) for length in lengths]
    for lane, size in enumerate(local):
        steps = torch.arange(start, start + size)
        features[lane, :size] = torch.sin(steps[:, None] * .1 + torch.arange(31)[None, :] * .03 + lane)
        features[lane, :size, 15] = steps / 2188
        features[lane, :size, 16] = (steps % period) / 2188
        features[lane, :size, 17] = 1
        mask[lane, :size] = steps % period == 0
        query = steps[steps % period == 0]
        scores[lane, :size][mask[lane, :size]] = torch.tensor([4., 7., 2., 6.]) + query[:, None] * torch.tensor([.125, -.25, .5, .25])
    return {"features": features, "query_scores": scores, "query_mask": mask,
            "lengths": torch.tensor(local, dtype=torch.int64),
            "episode_ends": torch.tensor([size > 0 and start + size == length
                for size, length in zip(local, lengths, strict=True)])}


@pytest.mark.parametrize("period", (4, 8))
@pytest.mark.parametrize("view", audit.VIEWS)
def test_independent_p4_p8_schedule_matches_mixed_length_model_geometry(period, view):
    lengths = (1, 4, 5, 8, 9, 33, 65)
    mode = "none" if view in ("pretrained", "joint_aux") else "trace_delta" if view == "trace_no_write" else view
    model = composed.make_model(memory.Config(mode, key_dim=8), seed=1001, query_period=period, slow_mode="frozen")
    for first in range(0, len(lengths), 6):
        batch, carry = lengths[first:first + 6], None
        for start in range(0, max(batch), 32):
            packet = synthetic_chunk(batch, start, period)
            with torch.no_grad():
                result = model(**packet, carry=carry, no_write=view == "trace_no_write")
            work, units = audit.scheduled_work(packet["lengths"].tolist(), start, view, period)
            audit.close_equal(result.work_counts, work, "independent TEST operations")
            audit.close_equal(result.memory_work_units, units, "independent TEST units")
            carry = composed.detach_carry(result.carry)
        assert carry.fast.absolute_step.tolist() == list(batch)
        assert carry.fast.ended.tolist() == [True] * len(batch)


def test_independent_test_gate_matches_every_qualified_gate_field():
    flat, ids = census([17] * 36), identities()
    reports = []
    for period in audit.PERIODS:
        for seed in audit.SEEDS:
            for view in audit.VIEWS:
                scores = flat["raw_q"] if view == "trace_delta" else flat["raw_q"][:, ::-1].copy()
                reports.append(audit.scalar_report(np, ids, flat["raw_q"], flat["legal"], scores,
                    flat["raw_q"], flat["episode_offsets"], view, seed, period))
    for technical in (False, True):
        audit.close_equal(audit.independent_gate(reports, technical_complete=technical),
            gate.continuation_gate(reports, ids, technical_complete=technical), "entire independent TEST gate")


def test_independent_test_audit_capacity(tmp_path):
    """One declared six-path scalar probe; all48-view projection is a heuristic."""
    flat, ids = census([2188] * 6), identities()[:6]
    periods = []
    for period in audit.PERIODS:
        start = time.perf_counter()
        result = audit.scalar_report(np, ids, flat["raw_q"], flat["legal"], flat["raw_q"][:, ::-1].copy(),
            flat["raw_q"], flat["episode_offsets"], "trace_delta", audit.SEEDS[0], period)
        seconds = time.perf_counter() - start
        assert result["episodes"] == 6 and result["query_period"] == period
        periods.append({"query_period": period, "episodes": 6, "rows": 6 * 2188, "seconds": seconds})
    projected = max(row["seconds"] / row["rows"] for row in periods) * 3780864 * 2
    record = {"version": "otto-query-memory-test-audit-capacity-v1", "periods": periods,
        "worst_case_rows": 3780864, "multiplier": 2, "projected_seconds": projected,
        "threshold_seconds": 480, "reserve_seconds": 120, "audit_cap_seconds": 600,
        "projected_total_seconds": projected + 120, "empirical_reads": 0, "model_calls": 0,
        "admitted": projected <= 480, "scope": "fabricated scalar capacity heuristic; original600-second cap remains binding"}
    (tmp_path / "test-audit-capacity.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    assert record["admitted"]


def metadata_fixture(item):
    files = {}

    def pin(name):
        value = {"sha256": hashlib.sha256(name.encode()).hexdigest(), "bytes": 1}
        files[name] = value
        return value

    fits = []
    for seed in audit.SEEDS:
        for family in audit.FITS:
            name = f"checkpoint-{family}-{seed}.npz"
            fits.append({"family": family, "seed": seed, "checkpoint_path": name, "checkpoint": pin(name),
                         "final": {"sha256": name}, "final_slow": {"sha256": "slow" + str(seed)}})
    views = []
    for period in audit.PERIODS:
        for seed in audit.SEEDS:
            for view in audit.VIEWS:
                parent = "pretrained" if view == "last_error" else "trace_delta" if view == "trace_no_write" else view
                fit = next(f for f in fits if (f["seed"], f["family"]) == (seed, parent))
                name = f"test-P{period}-prediction-{view}-{seed}.npz"
                views.append({"query_period": period, "seed": seed, "view": view, "stage": "test", "parent_fit": parent,
                    "prediction_path": name, "prediction": pin(name), "checkpoint_path": fit["checkpoint_path"],
                    "checkpoint": fit["checkpoint"], "model_before": copy.deepcopy(fit["final"]),
                    "model_after": copy.deepcopy(fit["final"]), "clone_seconds": .1, "inference_and_loss_seconds": .2,
                    "metric_seconds": .1, "serialization_seconds": .1})
    for period in audit.PERIODS:
        pin(f"test-P{period}-history.npz")
    zero = ("optimizer_calls", "teacher_calls", "native_calls", "train_array_decodes", "dev_array_decodes")
    item.evaluator = SimpleNamespace(ZERO_COUNTS=zero)
    item.worker = {"files": files, "model_calls": 288, "views_completed": 48, "checkpoint_decodes": 18,
                   "test_array_decodes": 1, **dict.fromkeys(zero, 0)}
    item.plan = {"configuration": {"fabricated": True}, "runtime": {"fabricated": True},
                 "inputs": {"training_receipt": {"sha256": "fabricated", "bytes": 1, "path": "training.json"}}}
    item.context = {"training": {"fits": fits, "run": item.run / "training"},
                    "collection": {"run": item.run / "collection", "receipt": {"files": {"test.npz": pin("test.npz")}}}}
    values = {"forks.json": {"forks": []}, "test-views.json": {"stage": "test", "technical_complete": False, "views": views},
        "checkpoint-manifest.json": {"training_receipt": item.plan["inputs"]["training_receipt"], "checkpoints": copy.deepcopy(fits)},
        "runtime.json": {**item.plan["runtime"], "torch_threads": 1, "interop_threads": 1,
                         "deterministic": True, "cuda_used": False, "mps_used": False},
        "summary.json": {"version": audit.PRODUCER_VERSION, "configuration": item.plan["configuration"],
            "views": copy.deepcopy(views), "technical_complete": False, "setup_seconds": .3,
            **{k: item.worker[k] for k in (*zero, "views_completed", "checkpoint_decodes", "test_array_decodes", "model_calls")}}}
    return values


@pytest.mark.parametrize("defect", (None, "missing", "period", "mutated_checkpoint", "alternate_parent", "source_pin",
                                   "manifest", "promotion", "summary", "timing", "runtime"))
def test_metadata_authenticates_all48_views_and_same18_final_checkpoints_before_decode(tmp_path, monkeypatch, defect):
    item = audit.Audit(SimpleNamespace(output=tmp_path / "audit"))
    item.run = tmp_path
    values = metadata_fixture(item)
    row = values["test-views.json"]["views"][-1]
    if defect == "missing":
        values["test-views.json"]["views"].pop()
    elif defect == "period":
        row["query_period"] = 4
    elif defect == "mutated_checkpoint":
        row["model_after"] = {"sha256": "different"}
    elif defect == "alternate_parent":
        row["parent_fit"] = "trace_additive"
    elif defect == "source_pin":
        row["checkpoint"] = {"sha256": "different", "bytes": 1}
    elif defect == "manifest":
        values["checkpoint-manifest.json"]["checkpoints"].pop()
    elif defect == "promotion":
        values["test-views.json"]["technical_complete"] = True
    elif defect == "summary":
        values["summary.json"]["views"].pop()
    elif defect == "timing":
        row["metric_seconds"] = float("nan")
    elif defect == "runtime":
        values["runtime.json"]["mps_used"] = True
    monkeypatch.setattr(audit, "read", lambda path: values[path.name])
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("metadata cannot decode arrays"))
    if defect is None:
        item.authenticate_metadata()
        assert len(item.allowed_arrays) == 69
        assert len(item.views) == 48 and len(item.fits) == 18
    else:
        with pytest.raises(ValueError):
            item.authenticate_metadata()
    assert item.authenticated is False and item.receipt["array_decodes"] == 0


@pytest.mark.parametrize("filename", ("train.npz", "dev.npz", "checkpoint-other.npz", "test.npz"))
def test_no_numerical_decode_before_authenticated_original_closure_or_outside_exact_allowlist(tmp_path, monkeypatch, filename):
    item = audit.Audit(SimpleNamespace(output=tmp_path / "audit"))
    item.np = np
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("unadmitted numerical decode"))
    with pytest.raises(ValueError, match="original closure checks"):
        item.arrays(tmp_path / filename, {})
    item.authenticated = True
    with pytest.raises(ValueError, match="only the69"):
        item.arrays(tmp_path / filename, {})
    assert item.receipt["array_decodes"] == item.receipt["test_array_decodes"] == 0


def first_view_journal(item):
    lengths = [1, 4, 5, 8, 9, 33] + [1] * 30
    item.histories_by_period = {8: {"episode_offsets": np.array([0, *np.cumsum(lengths)], np.int64)}}
    item.views = [{"view": "trace_delta", "seed": audit.SEEDS[0], "query_period": 8}]
    local = [min(32, n) for n in lengths[:6]]
    work, units = audit.scheduled_work(local, 0, "trace_delta", 8)
    identity = {"call_id": 1, "phase": "test", "view": "trace_delta", "seed": audit.SEEDS[0],
                "query_period": 8, "episode_indices": list(range(6)), "start": 0}
    return [{"event": "attempt", **identity}, {"event": "return", **identity,
             "rows": sum(local), "work_counts": work, "memory_work_units": units}]


@pytest.mark.parametrize("defect", (None, "period", "rows", "query_count", "matrix_reads", "call_id"))
def test_independent_journal_rejects_schedule_and_cost_drift_in_first_pair(tmp_path, monkeypatch, defect):
    item = audit.Audit(SimpleNamespace(output=tmp_path / "audit"))
    item.np, item.run = np, tmp_path
    rows = first_view_journal(item)
    if defect == "period":
        rows[1]["query_period"] = 4
    elif defect == "rows":
        rows[1]["rows"] += 1
    elif defect == "query_count":
        rows[1]["work_counts"]["slow_query_rows"] += 1
    elif defect == "matrix_reads":
        rows[1]["memory_work_units"]["matrix_read_terms"] += 32
    elif defect == "call_id":
        rows[1]["call_id"] = 2
    monkeypatch.setattr(item, "rows", lambda _path: iter(rows))
    message = "complete original TEST work pair" if defect is None else "independent scheduled TEST work return"
    with pytest.raises(ValueError, match=message):
        item.journal()


def test_failed_original_closure_preserves_failure_and_never_decodes_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit.dev, "ROOT", tmp_path)
    monkeypatch.setattr(audit.signal, "signal", lambda *_args: None)
    item = audit.Audit(SimpleNamespace(output=tmp_path / "audit"))

    def fail():
        raise ValueError("fabricated unclosed TEST producer")

    monkeypatch.setattr(item, "admit", fail)
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("array decoded after failed admission"))
    with pytest.raises(ValueError, match="unclosed TEST"):
        item.execute()
    receipt = json.loads((tmp_path / "audit/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["complete"] is False
    assert receipt["array_decodes"] == receipt["model_calls"] == receipt["optimizer_calls"] == 0
    assert receipt["continuation_admitted"] is False and not (tmp_path / "audit/audit.json").exists()
