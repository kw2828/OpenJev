"""Fabricated covariance and evidence witnesses, without production imports."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import audit_residual_memory_study as audit


def fixture():
    return {"bx": np.zeros((1, 2, 1, 2)), "by": np.array([[[0.], [2.]]]),
            "fx": np.zeros((1, 1, 2, 2)), "fy": np.ones((1, 1, 2)),
            "qx": np.zeros((1, 1, 2)), "target": np.zeros((1, 1)),
            "private_selected_block": np.zeros((1, 1), dtype=np.int64)}


def projected_scalar_variance():
    # A direct linear solve, independent of the audit's Cholesky decomposition.
    z = np.array([(a, b) for a in (-2., -2/3, 2/3, 2.)
                  for b in (-2., -2/3, 2/3, 2.)])
    kz = np.exp(-.5*(z**2).sum(1))
    kzz = np.exp(-.5*((z[:, None]-z[None])**2).sum(2))+1e-6*np.eye(16)
    return float(kz@np.linalg.solve(kzz, kz))


@pytest.mark.parametrize("mode", audit.MODES)
def test_repeated_location_matches_scalar_latent_gaussian(mode):
    data = fixture()
    q = 1. if mode == "full" else projected_scalar_variance()
    v = .0225+(1-q if mode == "fic" else 0.)
    result = audit.reconstruct(data, 1., 1., mode)
    expected_mean = q*(np.array([0., 2.])+2)/(v+3*q)
    expected_variance = v+q*v/(v+3*q)+(1-q if mode == "query_only" else 0.)
    np.testing.assert_allclose(result["component_mean"][0, 0], expected_mean, atol=1e-12)
    np.testing.assert_allclose(result["component_variance"], expected_variance, atol=1e-12)
    latent = q*v/(q+v)
    residual = 1-q*np.array([0., 2.])/(q+v)
    logdet = math.log(v)+math.log(v+2*latent)
    loglik = -.5*(2*math.log(2*math.pi)+logdet+2*residual**2/(v+2*latent))
    np.testing.assert_allclose(result["log_weights"][0, 0], loglik-logsumexp(loglik), atol=1e-12)


def test_query_only_preserves_routing_and_means_but_adds_residual_once():
    data = fixture()
    data["qx"][0, 0] = [.3, -.4]
    sor = audit.reconstruct(data, 1., 1., "sor")
    corrected = audit.reconstruct(data, 1., 1., "query_only")
    np.testing.assert_array_equal(sor["component_mean"], corrected["component_mean"])
    np.testing.assert_array_equal(sor["log_weights"], corrected["log_weights"])
    assert np.all(corrected["component_variance"] > sor["component_variance"])


def test_full_joint_request_evidence_is_not_product_of_scalar_marginals():
    result = audit.reconstruct(fixture(), 1., 1., "full")
    noise = .0225
    residual = 1-np.array([0., 2.])/(1+noise)
    variance = noise+noise/(1+noise)
    independent = -(math.log(2*math.pi*variance)+residual**2/variance)
    independent -= logsumexp(independent)
    assert not np.allclose(result["log_weights"][0, 0], independent, atol=1e-3)


def test_distant_inputs_expose_missing_prior_variance_without_clipping():
    data = fixture()
    for name in ("bx", "fx", "qx"):
        data[name][:] = 100.
    sor = audit.reconstruct(data, 1., 1., "sor")
    fic = audit.reconstruct(data, 1., 1., "fic")
    np.testing.assert_allclose(sor["component_variance"], .0225, atol=1e-12)
    np.testing.assert_allclose(fic["component_variance"], 1.0225, atol=1e-12)
    np.testing.assert_array_equal(fic["component_mean"], np.zeros((1, 1, 2)))
    np.testing.assert_allclose(fic["prob_positive"], .5, atol=1e-12)


@pytest.mark.parametrize("mode", audit.MODES)
def test_no_private_target_future_request_reads_or_input_mutation(mode):
    data = fixture()
    originals = {k: v.copy() for k, v in data.items()}
    before = audit.reconstruct(data, .8, 1.3, mode)
    for name in data:
        np.testing.assert_array_equal(data[name], originals[name])
    data["target"][:] = np.nan
    data["private_selected_block"][:] = 900
    for name in ("fx", "fy", "qx"):
        data[name] = np.concatenate((data[name], data[name]+.7), axis=1)
    after = audit.reconstruct(data, .8, 1.3, mode)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name][:, :1])


def test_component_permutation_leaves_predictive_mixture_unchanged():
    data = fixture()
    before = audit.reconstruct(data, 1., 1., "fic")
    data["bx"], data["by"] = data["bx"][:, ::-1], data["by"][:, ::-1]
    after = audit.reconstruct(data, 1., 1., "fic")
    for name in ("component_mean", "component_variance", "log_weights"):
        np.testing.assert_array_equal(before[name][..., ::-1], after[name])
    np.testing.assert_allclose(before["prob_positive"], after["prob_positive"], atol=1e-12)


def test_materially_negative_residual_fails_declared_roundoff_floor():
    np.testing.assert_array_equal(audit.residual_diagonal(1., np.array([1.+5e-13])), [0.])
    with pytest.raises(ValueError, match="residual"):
        audit.residual_diagonal(1., np.array([1.+2e-12]))


@pytest.mark.parametrize("name", ("component_mean", "component_variance", "log_weights", "prob_positive"))
def test_saved_prediction_mutation_is_rejected(name):
    data = fixture()
    exact = audit.reconstruct(data, 1., 1., "fic")
    saved = {k: v.copy() for k, v in exact.items()}
    saved["log_prob"] = audit.density(exact, data["target"])
    audit.compare_prediction(saved, exact, data["target"])
    saved[name].flat[0] += .01
    with pytest.raises(ValueError):
        audit.compare_prediction(saved, exact, data["target"])


def test_external_stop_propagates_before_reconstruction_loop():
    failure = RuntimeError("fixture stop")

    def stop():
        raise failure

    with pytest.raises(RuntimeError) as caught:
        audit.reconstruct(fixture(), 1., 1., "fic", check=stop)
    assert caught.value is failure


def all_rows():
    rows = []
    for phase, cohort, mode, seed in audit.identities():
        values = {name: 0. for name in audit.METRICS}
        values.update(regret=0. if mode == "true_gp" else .05 if mode == "fic" else .1,
                      nll=.8 if mode == "fic" else 1., always_defer_regret=.08)
        rows.append({"phase": phase, "cohort": cohort, "mode": mode, "fit_seed": seed, **values})
    return rows


def test_exact_eleven_conditions_and_single_base_failure():
    rows = all_rows()
    result = audit.classification(rows)
    assert result["gate"] == "RESIDUAL_CONTROL_QUALIFIED"
    assert len(result["checks"]) == 11
    assert all(r["passed"] for r in result["checks"])
    for row in rows:
        if row["phase"] == "base" and row["mode"] == "fic":
            row["regret"] = .11
    result = audit.classification(rows)
    assert result["gate"] == "RESIDUAL_CONTROL_NOT_QUALIFIED"
    assert [r["name"] for r in result["checks"] if not r["passed"]] == ["base_regret_preserved"]


def test_favorable_average_cannot_rescue_one_shift_cohort():
    rows = all_rows()
    for row in rows:
        if row["phase"] == "shift" and row["mode"] == "fic":
            row["regret"] = .11 if row["cohort"] == 0 else .01
    result = audit.classification(rows)
    assert result["gate"] == "RESIDUAL_CONTROL_NOT_QUALIFIED"
    assert [r["name"] for r in result["checks"] if not r["passed"]] == ["shift_cohort_0_regret_win"]
    with pytest.raises(ValueError, match="117"):
        audit.classification(rows[:-1])


def resource_fixture():
    rows = []
    for phase, cohort, mode, seed in audit.identities():
        if cohort:
            continue
        population = audit.CONFIG["populations"][phase]
        k, n = population["blocks"], population["basis_points"]
        rows.append({"phase": phase, "mode": mode, "fit_seed": seed,
            "cache_array_bytes": 8*k*(n*n+3*n) if mode in ("full", "true_gp") else 2304+2176*k,
            "input_archive_bytes": 24*k*n, "queries_per_archive": 4,
            "timings": [{"archive_seconds": .01, "query_seconds": .03, "total_seconds": .04} for _ in range(20)],
            "median_context_seconds": .04})
    return rows


def test_cache_storage_and_full_latency_reconcile_without_timing_replay():
    rows = resource_fixture()
    audit.validate_resources(rows)
    assert len(rows) == 39
    assert audit.cache_bytes("fic", 4, 16) == audit.cache_bytes("fic", 4, 128) == 11008
    rows[0]["cache_array_bytes"] -= 8
    with pytest.raises(ValueError, match="cache"):
        audit.validate_resources(rows)
    rows = resource_fixture()
    rows[0]["timings"][0]["total_seconds"] = .08
    with pytest.raises(ValueError, match="timing scope"):
        audit.validate_resources(rows)


def diagnostic_fixture():
    rows = []
    for phase, cohort, mode, seed in audit.identities():
        population = audit.CONFIG["populations"][phase]
        k, n = population["blocks"], population["basis_points"]

        def count_record(count):
            return {"residual_points": count, "minimum_raw_residual": .1 if count else None,
                    "residual_floor_count": 0}

        rows.append({"phase": phase, "cohort": cohort, "mode": mode, "fit_seed": seed,
            "archive_seconds": .01, "query_seconds": .03, "context_diagnostics": [
                {"archive": count_record(k*n if mode == "fic" else 0),
                 "queries": [count_record(5 if mode == "fic" else 1 if mode == "query_only" else 0)
                             for _ in range(4)], "cache_array_bytes": audit.cache_bytes(mode, k, n)}
                for _ in range(128)]})
    return rows


def test_residual_diagnostics_separate_unperformed_and_failed_corrections():
    rows = diagnostic_fixture()
    audit.validate_diagnostics(rows)
    rows[0]["context_diagnostics"][0]["archive"]["minimum_raw_residual"] = 0.
    with pytest.raises(ValueError, match="unperformed"):
        audit.validate_diagnostics(rows)
    rows = diagnostic_fixture()
    fic = next(r for r in rows if r["mode"] == "fic")
    fic["context_diagnostics"][0]["queries"][0]["residual_floor_count"] = 1
    with pytest.raises(ValueError, match="floor bound"):
        audit.validate_diagnostics(rows)


def test_same_fitted_kernel_pair_errors_use_all_components():
    pred = {"log_weights": np.log(np.array([[[.75, .25]]])), "prob_positive": np.array([[.2]])}
    ref = {"log_weights": np.log(np.array([[[.5, .5]]])), "prob_positive": np.array([[.6]])}
    result = audit.paired_diagnostics(pred, ref)
    assert result["mean_weight_l1"] == pytest.approx(.5)
    assert result["mean_sign_probability_absolute_error"] == pytest.approx(.4)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def refresh(folder):
    dump(folder/"manifest.json", {str(p.relative_to(folder)): audit.descriptor(p)
        for p in folder.rglob("*") if p.is_file() and p.name != "manifest.json"})


def opaque_fixture(tmp_path, monkeypatch):
    root, parent, folder = tmp_path/"repo", tmp_path/"parent", tmp_path/"run"
    monkeypatch.setattr(audit, "ROOT", root)
    monkeypatch.setattr(audit, "PARENT", parent)
    sources = {}
    for name in audit.SOURCES:
        for path in (root/name, folder/"source"/name):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"Opaque source fixture.")
        sources[name] = audit.descriptor(root/name)
    dump(parent/"registration.json", {"frozen_parent": True})
    dump(parent/"run-receipt.json", {"state": "EXITED", "exit_code": 0})
    checkpoints, kernels = {}, []
    for seed in audit.SEEDS:
        name = f"nystrom16-{seed}.pt"
        for path in (parent/name, folder/name):
            path.write_bytes(b"Opaque checkpoint, never deserialized.")
        checkpoints[name] = audit.descriptor(parent/name)
        kernels.append({"fit_seed": seed, "length": 1., "amplitude": 1., "checkpoint": name,
                        "checkpoint_descriptor": checkpoints[name]})
    refresh(parent)
    dump(folder/"registration.json", {"config": audit.CONFIG, "sources": sources,
        "parent": {"manifest": audit.descriptor(parent/"manifest.json"),
            "registration": audit.descriptor(parent/"registration.json"), "checkpoints": checkpoints},
        "environment": {"python": sys.version, "numpy": np.__version__, "scipy": audit.scipy.__version__,
            "platform": audit.platform.platform(), "threads": 1, "torch": "opaque runtime"}})
    dump(folder/"kernels.json", kernels)
    dump(folder/"run-receipt.json", {"state": "EXITED", "exit_code": 0, "elapsed_seconds": 1.,
                                    "training_updates": 0})
    for name in ("started.json", "metrics.json", "resources.json", "diagnostics.json", "pairwise.json", "summary.json"):
        dump(folder/name, {})
    for phase, cohort, mode, seed in audit.identities():
        (folder/f"{phase}-{cohort}-{mode}-{seed}.npz").write_bytes(b"Never decoded in source fixture.")
        (folder/f"{phase}-{cohort}-data.npz").write_bytes(b"Never decoded in source fixture.")
    refresh(folder)
    return folder


def test_closed_inventory_and_parent_checkpoint_lineage_are_opaque(tmp_path, monkeypatch):
    folder = opaque_fixture(tmp_path, monkeypatch)
    result = audit.authenticate(folder)
    assert result["raw_files"] == 138
    assert len(result["sources"]) == 9
    assert set(audit.kernel_records(folder)) == {11, 23, 37}
    (folder/"nystrom16-11.pt").write_bytes(b"tampered inherited bytes")
    refresh(folder)
    with pytest.raises(ValueError, match="checkpoint identity"):
        audit.authenticate(folder)


def test_rehashed_extra_file_and_unclosed_receipt_fail(tmp_path, monkeypatch):
    folder = opaque_fixture(tmp_path, monkeypatch)
    (folder/"unexpected.npz").write_bytes(b"extra")
    refresh(folder)
    with pytest.raises(ValueError, match="roster"):
        audit.authenticate(folder)
    (folder/"unexpected.npz").unlink()
    dump(folder/"run-receipt.json", {"state": "FAILED", "exit_code": 1, "elapsed_seconds": 1., "training_updates": 0})
    refresh(folder)
    with pytest.raises(ValueError, match="receipt"):
        audit.authenticate(folder)
