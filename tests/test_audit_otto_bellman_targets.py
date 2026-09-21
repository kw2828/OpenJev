"""Synthetic independent-target audit tests; no producer/environment imports."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PIN = "a" * 64


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("test_bellman_saved_audit", ROOT / "scripts/audit_otto_bellman_targets.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def archive(c0=.5, bias=.125, slope=.25):
    first = np.zeros((8, 11028), np.float32)
    first[:, -3] = slope
    return {"version": np.asarray("otto-return-value-v1"), "kind": np.asarray("mlp8"),
            "input_dim": np.asarray(11028), "c0": np.asarray(c0, np.float32),
            "first_weight": first, "final_weight": np.full(8, .125, np.float32),
            "hidden_bias": np.zeros(8, np.float32), "output_bias": np.asarray(bias, np.float32)}


def fixture_data(count=3):
    beliefs = np.zeros((count, 53, 53), np.float64)
    dense = np.arange(1, 2810, dtype=np.float64).reshape(53, 53)
    beliefs[:] = dense / dense.sum()
    positions = np.asarray([[0, 0], [26, 26], [52, 52]][:count], np.int64)
    sensing = np.asarray([3., 4., 3.][:count], np.float64)
    allowed = [[1, 3], [0, 1, 2, 3], [0, 2]][:count]
    kernels = {}
    for lam in (3., 4.):
        k = np.empty((4, 107, 107), np.float64)
        for hit in range(4):
            k[hit].fill((hit + 1) / 16 if lam == 3 else (4 - hit) / 16)
        k[:, 53, 53] = 0
        kernels[lam] = k
    return {"beliefs": beliefs, "positions": positions, "sensing_length": sensing}, allowed, kernels


def manual_payload(head, train, allowed, kernels, indices):
    """Scalar oracle for context-only synthetic heads, not the imported audit."""
    assert not head["first_weight"][:, :11025].any()
    rows = []
    for index in indices:
        belief, position = train["beliefs"][index], train["positions"][index]
        lam = train["sensing_length"][index]
        masses, floors, values, successors = [], [], [], []
        for action, delta in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            x, y = (max(0, min(52, int(position[j]) + delta[j])) for j in range(2))
            for hit in range(4):
                joint = belief * kernels[lam][hit, 53 - x:106 - x, 53 - y:106 - y]
                raw = float(joint.sum())
                floor = max(raw, 1e-10)
                mass = float((joint / floor).sum())
                context = [mass * x / 52, mass * y / 52, mass * lam / 5]
                hidden = [max(0., math.fsum(float(w) * v for w, v in zip(row[-3:], context, strict=True)) + float(b))
                          for row, b in zip(head["first_weight"], head["hidden_bias"], strict=True)]
                value = 64 * (float(head["c0"]) * mass + float(head["output_bias"])
                              + math.fsum(h * float(w) for h, w in zip(hidden, head["final_weight"], strict=True)))
                masses.append(raw)
                floors.append(floor)
                values.append(value)
                successors.append([x, y])
        masses, floors, values = np.asarray(masses).reshape(4, 4), np.asarray(floors).reshape(4, 4), np.asarray(values)
        contributions = floors * values.reshape(4, 4)
        costs = 1 + contributions.sum(axis=1)
        minimum = min(costs[a] for a in allowed[index])
        minimum_action = next(a for a in allowed[index] if costs[a] == minimum)
        deployed = next(a for a in allowed[index] if abs(costs[a] - minimum) < 1e-10)
        rows.append({"row_indices": np.asarray(index, np.int64), "raw_masses": masses, "weights": floors,
                     "branch_values": values, "branch_contributions": contributions,
                     "successors": np.asarray(successors, np.int64), "costs": costs,
                     "eligible_masks": np.asarray([a in allowed[index] for a in range(4)], bool),
                     "minimum_costs": np.asarray(minimum), "targets_float64": np.asarray(minimum / 64),
                     "targets_float32": np.asarray(minimum / 64, np.float32),
                     "minimum_actions": np.asarray(minimum_action, np.int64),
                     "deployed_actions": np.asarray(deployed, np.int64),
                     "deployed_cost_gaps": np.asarray(costs[deployed] - minimum)})
    return {name: np.stack([r[name] for r in rows]) for name in rows[0]}


def bundle(indices=(2, 0, 1), head=None):
    head = archive() if head is None else head
    train, allowed, kernels = fixture_data()
    payload = manual_payload(head, train, allowed, kernels, indices)
    kwargs = {"train": train, "eligible_actions": allowed, "kernels": kernels,
              "expected_row_indices": list(indices), "checkpoint_sha256": PIN,
              "expected_checkpoint_sha256": PIN, "c0": float(head["c0"])}
    return payload, head, kwargs


def test_every_selected_row_independent_arithmetic_and_honest_counts(audit):
    payload, head, kwargs = bundle()
    before = {k: v.tobytes() for k, v in payload.items()}
    ticks = []
    result = audit.audit_refresh(payload, head, **kwargs, check=lambda: ticks.append(True))
    assert result["agreement"] and result["rows"] == 3
    assert result["numeric_items_compared"] == sum(v.size for v in payload.values()) == 333
    assert result["audit_checkpoint_readout_calls"] == 3 and result["audit_branch_rows"] == 48
    assert result["training_calls"] == result["simulator_calls"] == 0
    assert result["maximum_absolute_difference"] < 1e-10
    assert ticks == [True] * 4
    assert before == {k: v.tobytes() for k, v in payload.items()}


def test_zero_subfloor_and_signed_biased_values_are_preserved(audit):
    payload, head, kwargs = bundle(head=archive(c0=0., bias=-.5, slope=0.))
    kwargs["train"]["beliefs"][0].fill(0)
    kwargs["train"]["beliefs"][1] *= 1e-20
    payload = manual_payload(head, kwargs["train"], kwargs["eligible_actions"], kwargs["kernels"], [2, 0, 1])
    assert (payload["branch_values"] == -32).all()
    assert (payload["weights"][1:] == 1e-10).all()
    assert payload["targets_float64"][0] < 0
    assert audit.audit_refresh(payload, head, **kwargs)["agreement"]


def test_true_minimum_is_distinct_from_deployed_near_tie(audit):
    head = archive(c0=0., bias=0., slope=1e-12)
    head["final_weight"][:] = -.125
    payload, head, kwargs = bundle(indices=(1,), head=head)
    assert payload["minimum_actions"].tolist() == [1]
    assert payload["deployed_actions"].tolist() == [0]
    assert 0 < payload["deployed_cost_gaps"][0] < 1e-10
    assert audit.audit_refresh(payload, head, **kwargs)["agreement"]
    payload["minimum_actions"][0] = 0
    with pytest.raises(ValueError, match="minimum_actions"):
        audit.audit_refresh(payload, head, **kwargs)


@pytest.mark.parametrize("field,invalid", [("raw_masses", -1e-12), ("weights", 0.), ("deployed_cost_gaps", -1e-12)])
def test_tolerance_never_admits_impossible_mass_floor_or_gap(audit, field, invalid):
    payload, head, kwargs = bundle()
    kwargs["train"]["beliefs"].fill(0)
    payload = manual_payload(head, kwargs["train"], kwargs["eligible_actions"], kwargs["kernels"], [2, 0, 1])
    payload[field].reshape(-1)[0] = invalid
    with pytest.raises(ValueError, match="preserved positive floor"):
        audit.audit_refresh(payload, head, **kwargs)


def test_saved_floor_identity_rejects_doubled_zero_mass_weight(audit):
    payload, head, kwargs = bundle()
    kwargs["train"]["beliefs"].fill(0)
    payload = manual_payload(head, kwargs["train"], kwargs["eligible_actions"], kwargs["kernels"], [2, 0, 1])
    assert (payload["raw_masses"] == 0).all()
    assert (payload["weights"] == 1e-10).all()
    payload["weights"][0, 0, 0] = 2e-10
    with pytest.raises(ValueError, match="exact saved mass-to-floor identity"):
        audit.audit_refresh(payload, head, **kwargs)


@pytest.mark.parametrize("field", ["raw_masses", "weights", "branch_values", "branch_contributions", "successors",
                                  "costs", "eligible_masks", "minimum_costs", "targets_float64", "targets_float32",
                                  "minimum_actions", "deployed_actions", "deployed_cost_gaps", "row_indices"])
def test_each_saved_field_is_actually_checked(audit, field):
    payload, head, kwargs = bundle()
    value = payload[field].reshape(-1)
    value[0] = not value[0] if value.dtype == bool else value[0] + 1
    with pytest.raises(ValueError):
        audit.audit_refresh(payload, head, **kwargs)


def test_float32_cast_exactly_matches_saved_float64_not_tolerance(audit):
    payload, head, kwargs = bundle()
    payload["targets_float32"][0] = np.nextafter(payload["targets_float32"][0], np.float32(np.inf))
    with pytest.raises(ValueError, match="targets_float32"):
        audit.audit_refresh(payload, head, **kwargs)


def test_float32_cast_preserves_zero_sign_bytes(audit):
    payload, head, kwargs = bundle(indices=(1,), head=archive(c0=0., bias=-39062500., slope=0.))
    kwargs["train"]["beliefs"].fill(0)
    payload = manual_payload(head, kwargs["train"], kwargs["eligible_actions"], kwargs["kernels"], [1])
    assert payload["targets_float64"].tolist() == [0.]
    assert audit.audit_refresh(payload, head, **kwargs)["agreement"]
    payload["targets_float32"][0] = np.float32(-0.)
    with pytest.raises(ValueError, match="targets_float32"):
        audit.audit_refresh(payload, head, **kwargs)


def test_float64_tolerance_does_not_relax_cast_or_discrete_decisions(audit):
    payload, head, kwargs = bundle()
    payload["targets_float64"][0] += 1e-12
    payload["targets_float32"][0] = np.float32(payload["targets_float64"][0])
    assert audit.audit_refresh(payload, head, **kwargs)["agreement"]
    payload["targets_float64"][0] += 1e-6
    with pytest.raises(ValueError, match="targets_float64"):
        audit.audit_refresh(payload, head, **kwargs)


@pytest.mark.parametrize("fault", ["missing", "extra", "dtype", "shape", "nan", "reordered", "duplicate", "out_of_range"])
def test_schema_and_canonical_subset_rejections(audit, fault):
    payload, head, kwargs = bundle()
    if fault == "missing":
        del payload["weights"]
    elif fault == "extra":
        payload["score"] = np.ones(3)
    elif fault == "dtype":
        payload["weights"] = payload["weights"].astype(np.float32)
    elif fault == "shape":
        payload["branch_values"] = payload["branch_values"][:, :15]
    elif fault == "nan":
        payload["costs"][0, 0] = np.nan
    elif fault == "reordered":
        kwargs["expected_row_indices"] = [0, 1, 2]
    elif fault == "duplicate":
        kwargs["expected_row_indices"] = [2, 2, 1]
    else:
        kwargs["expected_row_indices"] = [2, 0, 3]
    with pytest.raises(ValueError):
        audit.audit_refresh(payload, head, **kwargs)


def test_bad_checkpoint_join_rejects_before_independent_inference(audit, monkeypatch):
    payload, head, kwargs = bundle()
    monkeypatch.setattr(audit.A, "predict", lambda *_args: pytest.fail("must reject before inference"))
    kwargs["checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="checkpoint identity"):
        audit.audit_refresh(payload, head, **kwargs)


@pytest.mark.parametrize("fault", ["baseline", "architecture", "origin", "kernel_nan", "lambda5", "belief_mass", "mask"])
def test_model_and_public_input_contract(audit, fault):
    payload, head, kwargs = bundle()
    if fault == "baseline":
        kwargs["c0"] = .75
    elif fault == "architecture":
        head["kind"] = np.asarray("min8")
    elif fault == "origin":
        kwargs["kernels"][3.][:, 53, 53] = .1
    elif fault == "kernel_nan":
        kwargs["kernels"][4.][0, 0, 0] = np.nan
    elif fault == "lambda5":
        kwargs["train"]["sensing_length"][0] = 5.
    elif fault == "belief_mass":
        kwargs["train"]["beliefs"][0] *= 2
    else:
        kwargs["eligible_actions"][0] = [0, 1, 3]
    with pytest.raises(ValueError):
        audit.audit_refresh(payload, head, **kwargs)


def test_callback_failure_propagates_before_next_readout(audit, monkeypatch):
    payload, head, kwargs = bundle()
    original, calls = audit.A.predict, []
    def prediction(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(audit.A, "predict", prediction)
    ticks = []
    def check():
        ticks.append(True)
        if len(ticks) == 3:
            raise TimeoutError("audit budget")
    with pytest.raises(TimeoutError, match="audit budget"):
        audit.audit_refresh(payload, head, **kwargs, check=check)
    assert calls == [True]
