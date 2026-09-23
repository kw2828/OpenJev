"""Synthetic report fixtures only; no empirical evidence files are opened."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_query_memory_report_fixture", ROOT / "scripts/render_otto_query_memory.py")
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def fixture(stage):
    periods, cases = ((4,), 3) if stage == "dev" else ((4, 8), 6)
    reports = []
    for period in periods:
        for seed_index, seed in enumerate(renderer.SEEDS):
            for family_index, family in enumerate(renderer.FAMILIES):
                value = 4. + family_index + seed_index * .1 + period * .01
                if family == renderer.CANDIDATE:
                    value += 10.  # Intentionally poor candidate must remain conspicuous.
                leaf = {"episodes": 3 * cases, "declared_case_count": cases, "supported_case_count": cases,
                        "case_weighted_raw_gap": value}
                reports.append({"stage": stage, "episodes": 6 * cases, "family": family, "seed": seed, "query_period": period,
                    "scopes": {scope: {"by_regime": {regime: copy.deepcopy(leaf) for regime in renderer.REGIMES}}
                               for scope in renderer.SCOPES}})
    conditions = [{"name": "technical_completion", "passed": True}]
    for period in periods:
        for regime in renderer.REGIMES:
            prefix = f"{regime}:P{period}:"
            conditions.append({"name": prefix + "supported_cases", "passed": True, "actual": cases, "required": 2 if stage == "dev" else 4})
            for name in ["later_gap_10pct", "full_gap_nonregression", *[f"seed_{seed}_nonregression" for seed in renderer.SEEDS]]:
                conditions.append({"name": prefix + name, "passed": False, "candidate": 18.14,
                                   "controls": {name: 4.14 for name in renderer.CONTROLS}})
    gate = {"stage": stage, "candidate": renderer.CANDIDATE, "technical_complete": True, "conditions": conditions,
            "passed_conditions": sum(row["passed"] for row in conditions), "total_conditions": len(conditions), "passed": False}
    return {"stage": stage, "agreement": True, "metrics": reports, "gate": gate}


@pytest.mark.parametrize("stage,records,panels,conditions", [("dev", 24, 4, 13), ("test", 48, 8, 25)])
def test_all_panels_and_failed_candidate_remain_visible(stage, records, panels, conditions):
    audit = fixture(stage)
    original = copy.deepcopy(audit)
    data = renderer.validate_report(audit, stage=stage, synthetic=True)
    assert data["record_count"] == records and len(data["panels"]) == panels
    assert data["gate"]["total_conditions"] == conditions and data["gate"]["passed"] is False
    assert all([row["family"] for row in panel["rows"]] == list(renderer.FAMILIES) for panel in data["panels"])
    assert all(len(row["seed_gaps"]) == 3 for panel in data["panels"] for row in panel["rows"])
    markdown = renderer.markdown_report(data)
    assert renderer.SYNTHETIC_LABEL in markdown and f"{stage.upper()} FAIL" in markdown
    assert markdown.count("`trace_delta` (candidate)") == panels
    assert all(condition["name"] in markdown for condition in audit["gate"]["conditions"])
    assert markdown.count("**FAIL**") == sum(not row["passed"] for row in audit["gate"]["conditions"])
    assert renderer.LIMITATION in markdown and audit == original
    assert data == renderer.validate_report(audit, stage=stage, synthetic=True)
    assert markdown == renderer.markdown_report(data)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected", "period", "stage", "case_count", "support", "nan", "nested_inf", "gate_missing", "gate_lie"])
def test_incomplete_or_nonfinite_saved_result_fails_closed(mutation):
    audit = fixture("test")
    if mutation == "missing":
        audit["metrics"].pop()
    elif mutation == "duplicate":
        audit["metrics"][-1] = copy.deepcopy(audit["metrics"][0])
    elif mutation == "unexpected":
        audit["metrics"][0]["family"] = "replacement_winner"
    elif mutation == "period":
        audit["metrics"][0]["query_period"] = 16
    elif mutation == "stage":
        audit["metrics"][0]["stage"] = "dev"
    elif mutation == "case_count":
        audit["metrics"][0]["scopes"]["later"]["by_regime"]["lambda3"]["declared_case_count"] = 5
    elif mutation == "support":
        audit["metrics"][0]["scopes"]["later"]["by_regime"]["lambda3"]["supported_case_count"] = 5
    elif mutation == "nan":
        audit["metrics"][0]["scopes"]["later"]["by_regime"]["lambda3"]["case_weighted_raw_gap"] = float("nan")
    elif mutation == "nested_inf":
        audit["other_diagnostic"] = {"nested": [float("inf")]}
    elif mutation == "gate_missing":
        audit["gate"]["conditions"].pop()
    else:
        audit["gate"]["passed"] = True
    with pytest.raises(ValueError):
        renderer.validate_report(audit, stage="test", synthetic=True)


def test_negative_gap_and_boolean_score_rejected():
    for value in (-.001, True):
        audit = fixture("dev")
        audit["metrics"][0]["scopes"]["full"]["by_regime"]["lambda4"]["case_weighted_raw_gap"] = value
        with pytest.raises(ValueError, match="finite nonnegative"):
            renderer.validate_report(audit, stage="dev", synthetic=True)


def test_synthetic_render_writes_only_labelled_artifact_pair(tmp_path):
    from PIL import Image

    target = tmp_path / "synthetic-render"
    result = renderer.render(fixture("test"), stage="test", output=target, synthetic=True)
    assert result["synthetic"] is True and result["record_count"] == 48 and len(result["panels"]) == 8
    assert set(result["files"]) == {"query-memory-test.md", "query-memory-test.png"}
    assert {path.name for path in target.iterdir()} == set(result["files"])
    assert renderer.SYNTHETIC_LABEL in (target / "query-memory-test.md").read_text()
    with Image.open(target / "query-memory-test.png") as image:
        assert image.format == "PNG" and image.width >= 2000 and image.height >= 2500
        assert image.info["Description"] == renderer.SYNTHETIC_LABEL
        image.verify()
    assert all(pin["bytes"] > 0 and len(pin["sha256"]) == 64 for pin in result["files"].values())
    with pytest.raises(ValueError, match="exclusive"):
        renderer.render(fixture("test"), stage="test", output=target, synthetic=True)


def test_validation_happens_before_any_output_creation(tmp_path):
    invalid = fixture("dev")
    invalid["metrics"].pop()
    target = tmp_path / "not-created"
    with pytest.raises(ValueError):
        renderer.render(invalid, stage="dev", output=target, synthetic=True)
    assert not target.exists()
