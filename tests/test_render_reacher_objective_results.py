"""Synthetic saved-audit rendering checks; no world-model or simulator calls."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import render_reacher_objective_results as figures


def synthetic_summary():
    summary = {"status": "completed", "version": figures.VERSION, "engineering": True,
               "saved_output_only": True, "plan_sha256": "a" * 64, "execution_completed_sha256": "b" * 64,
               "new_model_calls": 0, "new_policy_calls": 0, "new_fits": 0,
               "coverage": {"fits": 9, "control_rows": 57, "control_episodes_per_row": 2, "prediction_episodes": 2},
               "fits": {}, "control": {}, "costs": {"fit_wall_seconds": 18., "calibration_seconds": 1.5},
               "continuation_gate": {"passed": True, "checks": [{"passed": True} for _ in range(31)]},
               "paired_descriptive_comparisons": {}, "limits": ["Synthetic test fixture only."]}
    means = {"anchor": [10., 20., 40.], "raw": [8., 22., 36.], "latent": [9., 16., 28.]}
    for arm in figures.ARMS:
        for pair in figures.PAIRS:
            name = f"{arm}-{pair}"
            summary["fits"][name] = {"name": name, "arm": arm, "pair": pair, "wall_seconds": 2.,
                "training_seconds": 1.5, "setup_seconds": .1, "updates": 4, "ema_updates": 4 if arm == "latent" else 0,
                "parameters": 10, "trainable_parameters": 10 if arm == "anchor" else 12}
    for panel in figures.PANELS:
        rows = {}
        for arm in figures.ARMS:
            for index, pair in enumerate(figures.PAIRS):
                value = means[arm][index]
                rows[f"{arm}-{pair}"] = {"episode_costs": [value - 1, value + 1], "mean_cost": value,
                    "decision_wall_seconds": .5, "decision_seconds": [.01] * 50,
                    "per_case_amortized_seconds": .005, "candidate_evaluations": 25600,
                    "imagined_transitions": 273408}
                if panel != "full":
                    rows[f"{arm}-{pair}-reset"] = {"episode_costs": [value * 1.2 - 1, value * 1.2 + 1],
                                                    "mean_cost": value * 1.2}
        for name in ("known_state", "zero", "particle", "uniform"):
            rows[name] = {"episode_costs": [8., 10.], "mean_cost": 9.}
        summary["control"][panel] = rows
    return summary


def saved_audit(tmp_path, summary=None):
    folder = tmp_path / "audit"
    folder.mkdir()
    summary = synthetic_summary() if summary is None else summary
    figures.write(folder / "summary.json", summary)
    (folder / "README.md").write_text("Engineering test\n")
    receipt = {"status": "completed", "version": figures.VERSION, "engineering": True,
               "saved_output_only": True, "plan_sha256": summary["plan_sha256"],
               "execution_completed_sha256": summary["execution_completed_sha256"],
               "source_sha256": {"synthetic-source.py": "c" * 64}, "runtime": {"fixture": True},
               "costs": summary["costs"],
               "files": {name: figures.sha(folder / name) for name in ("summary.json", "README.md")}}
    figures.write(folder / "receipt.json", receipt)
    return folder, figures.sha(folder / "receipt.json")


def test_means_and_pair_percentages_are_recomputed_without_conflating_ratios(tmp_path):
    folder, digest = saved_audit(tmp_path)
    summary, receipt = figures.load_audit(folder, digest)
    data = figures.figure_data(summary, receipt, digest)
    assert data["versus_anchor"]["ordinary"]["raw"]["per_pair_percent"]["pair1"] == pytest.approx(10.)
    value = data["versus_anchor"]["ordinary"]["raw"]["family_mean_percent"]
    assert value == pytest.approx(100 * (66 / 70 - 1))
    assert value != pytest.approx(np.mean([-20, 10, -10]))
    assert data["latent_versus_raw"]["ordinary"]["family_mean_percent"] == pytest.approx(100 * (53 / 66 - 1))
    assert data["latent_versus_raw"]["shift"]["per_pair_percent"]["pair0"] == pytest.approx(12.5)
    assert data["reset_penalty"]["shift"]["latent"]["family_mean_percent"] == pytest.approx(20.)
    assert data["control"]["full"]["raw-pair0"]["per_case_amortized_ms"] == pytest.approx(5.)


@pytest.mark.parametrize("kind", ["receipt_hash", "member_tamper", "failed_receipt", "path_escape", "symlink"])
def test_authentication_rejects_untrusted_inputs_before_creating_figures(tmp_path, kind):
    folder, digest = saved_audit(tmp_path)
    if kind == "receipt_hash":
        digest = "0" * 64
    elif kind == "member_tamper":
        (folder / "README.md").write_text("Changed after audit")
    elif kind == "symlink":
        source = folder / "README.md"
        moved = tmp_path / "outside.md"
        source.rename(moved)
        source.symlink_to(moved)
    else:
        receipt = figures.read(folder / "receipt.json")
        if kind == "failed_receipt":
            receipt["status"] = "failed"
        else:
            receipt["files"]["../outside.md"] = "c" * 64
        figures.write(folder / "receipt.json", receipt)
        digest = figures.sha(folder / "receipt.json")
    out = tmp_path / "figures"
    with pytest.raises(ValueError):
        figures.render(folder, digest, out)
    assert not out.exists()


@pytest.mark.parametrize("kind", ["missing_fit", "missing_reset", "mean", "nonfinite", "timing", "gate"])
def test_data_validation_rejects_partial_or_inconsistent_results(tmp_path, kind):
    folder, digest = saved_audit(tmp_path)
    summary, receipt = figures.load_audit(folder, digest)
    summary = copy.deepcopy(summary)
    if kind == "missing_fit":
        del summary["fits"]["anchor-pair2"]
    elif kind == "missing_reset":
        del summary["control"]["ordinary"]["raw-pair2-reset"]
    elif kind == "mean":
        summary["control"]["full"]["anchor-pair0"]["mean_cost"] += 1
    elif kind == "nonfinite":
        summary["control"]["full"]["raw-pair0"]["episode_costs"][0] = float("nan")
    elif kind == "timing":
        summary["control"]["full"]["raw-pair0"]["per_case_amortized_seconds"] *= 2
    else:
        summary["continuation_gate"]["passed"] = False
    with pytest.raises(ValueError):
        figures.figure_data(summary, receipt, digest)


def test_renders_all_formats_with_engineering_watermark_and_bound_numeric_data(tmp_path):
    folder, digest = saved_audit(tmp_path)
    out = tmp_path / "figures"
    receipt = figures.render(folder, digest, out)
    expected = {f"{name}.{extension}" for name in figures.FIGURES for extension in ("png", "svg", "pdf")}
    assert set(receipt["files"]) == expected | {"figure-data.json", "README.md"}
    assert receipt["new_model_calls"] == receipt["new_engine_calls"] == 0
    for name, value in receipt["files"].items():
        assert figures.sha(out / name) == value
    for name in figures.FIGURES:
        assert (out / f"{name}.png").read_bytes().startswith(b"\x89PNG")
        assert (out / f"{name}.pdf").read_bytes().startswith(b"%PDF")
        assert "ENGINEERING FIXTURE: NOT RESEARCH RESULTS" in (out / f"{name}.svg").read_text()
    assert figures.read(out / "figure-data.json")["provenance"]["audit_receipt_sha256"] == digest
    assert "Latent vs raw" in (out / "README.md").read_text()
    assert "latent_versus_raw" in (out / "figure-data.json").read_text()
    with pytest.raises(ValueError, match="Exclusive"):
        figures.render(folder, digest, out)
