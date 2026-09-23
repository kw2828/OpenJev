"""Fabricated regression for the separately versioned saved-audit cost repair."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def module():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_otto_sparse_query_v2.py"
    spec = importlib.util.spec_from_file_location("_sparse_saved_audit_v2_fixture", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def costs():
    return {"original_setup_wall_seconds": 2., "validation_wall_seconds": 3.,
        "threshold_wall_seconds": .5, "validation_setup_seconds": .125,
        "calibration_paid_seconds": 3.625, "evaluation_wall_seconds": 5., "journal_io_seconds": .25,
        "operation_seconds": {"tensorflow_value": {"attempted": 7, "returned": 7, "seconds": 1.5}},
        "scope": "Fabricated nested timer ledger; no actual operations."}


def test_nested_operation_seconds_is_a_ledger_not_an_eighth_scalar(module):
    value = costs()
    before = copy.deepcopy(value)
    assert module.valid_physical_scalars(value)
    assert len(module.PHYSICAL_SCALARS) == 7
    assert "operation_seconds" not in module.PHYSICAL_SCALARS
    assert value == before


def test_every_declared_scalar_still_rejects_invalid_values(module):
    for key in module.PHYSICAL_SCALARS:
        for bad in (-.001, float("nan"), float("inf"), -float("inf"), True, "1.0", {}, None):
            value = costs()
            value[key] = bad
            assert not module.valid_physical_scalars(value), (key, bad)
    value = costs()
    for key in module.PHYSICAL_SCALARS:
        value[key] = 0
    assert module.valid_physical_scalars(value)


def test_exact_cost_schema_does_not_silently_ignore_unexpected_fields(module):
    for missing in (*module.PHYSICAL_SCALARS, "scope", "operation_seconds"):
        value = costs()
        del value[missing]
        assert not module.valid_physical_scalars(value)
    value = costs()
    value["unpriced_extra_seconds"] = 1.
    assert not module.valid_physical_scalars(value)
    for key, bad in (("operation_seconds", 1.), ("scope", {})):
        value = costs()
        value[key] = bad
        assert not module.valid_physical_scalars(value)
