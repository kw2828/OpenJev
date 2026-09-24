"""Independent fabricated audit witnesses, with no study generation or models."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp, ndtr, ndtri

PATH = Path(__file__).resolve().parents[1]/"scripts/audit_query_feature_study.py"
SPEC = importlib.util.spec_from_file_location("independent_query_audit", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def repeated_location(fewshots=1):
    return {"bx": np.zeros((1, 2, 1, 2)), "by": np.array([[[0.], [2.]]]),
            "fx": np.zeros((1, 1, fewshots, 2)), "fy": np.ones((1, 1, fewshots)),
            "qx": np.zeros((1, 1, 2)), "target": np.zeros((1, 1)),
            "private_selected_block": np.zeros((1, 1), np.int64)}


@pytest.mark.parametrize("fewshots", [1, 2, 4])
def test_augmented_gp_matches_scalar_shared_latent_posterior(fewshots):
    data = repeated_location(fewshots)
    result = audit.exact_reference(data)
    noise = .0225
    expected_mean = (np.array([0., 2.])+fewshots)/(1+fewshots+noise)
    expected_variance = noise+noise/(1+fewshots+noise)
    np.testing.assert_allclose(result["component_mean"][0, 0], expected_mean, atol=1e-12)
    np.testing.assert_allclose(result["component_variance"], expected_variance, atol=1e-12)
    # Given one archive observation, F requests share the remaining latent
    # uncertainty. Their covariance eigenvalues are v+F*s and F-1 copies v.
    latent = noise/(1+noise)
    residual = 1-np.array([0., 2.])/(1+noise)
    logdet = (fewshots-1)*math.log(noise)+math.log(noise+fewshots*latent)
    expected_log = -.5*(fewshots*math.log(2*math.pi)+logdet
                       +fewshots*residual**2/(noise+fewshots*latent))
    expected_log -= logsumexp(expected_log)
    np.testing.assert_allclose(result["log_weights"][0, 0], expected_log, atol=1e-12)
    expected_positive = np.sum(np.exp(expected_log)*ndtr(expected_mean/math.sqrt(expected_variance)))
    assert result["prob_positive"][0, 0] == pytest.approx(expected_positive, abs=1e-12)


def test_joint_density_does_not_multiply_correlated_marginals():
    data = repeated_location(2)
    result = audit.exact_reference(data)
    noise = .0225
    residual = 1-np.array([0., 2.])/(1+noise)
    marginal_variance = noise+noise/(1+noise)
    wrong = -(math.log(2*math.pi*marginal_variance)+residual**2/marginal_variance)
    wrong -= logsumexp(wrong)
    assert not np.allclose(result["log_weights"][0, 0], wrong, atol=1e-3)


def test_reference_never_reads_private_identity_target_or_future_request():
    data = repeated_location(2)
    before = audit.exact_reference(data)
    data["target"][:] = np.nan
    data["private_selected_block"][:] = 999
    after = audit.exact_reference(data)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    for name in ("fx", "fy", "qx"):
        data[name] = np.concatenate((data[name], data[name]+3), axis=1)
    later = audit.exact_reference(data)
    for name in before:
        np.testing.assert_allclose(before[name][:, 0], later[name][:, 0], atol=1e-12)


def test_block_permutation_only_permutes_components():
    data = repeated_location(2)
    before = audit.exact_reference(data)
    data["bx"], data["by"] = data["bx"][:, ::-1], data["by"][:, ::-1]
    after = audit.exact_reference(data)
    for name in ("component_mean", "component_variance", "log_weights"):
        np.testing.assert_allclose(before[name][..., ::-1], after[name], atol=1e-12)
    np.testing.assert_allclose(before["prob_positive"], after["prob_positive"], atol=1e-12)


def prediction_fixture():
    p = np.array([[.05, .5, .95, .2]])
    target = np.array([[-1., 0., 1., 1.]])
    variance = np.full(p.shape+(1,), .1)
    mean = ndtri(p)[..., None]*np.sqrt(variance)
    weights = np.zeros_like(mean)
    density = -.5*(math.log(2*math.pi)+np.log(variance[..., 0])
                    +(target-mean[..., 0])**2/variance[..., 0])
    return {"component_mean": mean, "component_variance": variance,
            "log_weights": weights, "prob_positive": p, "log_prob": density}, target


def test_scoring_is_noisy_target_density_and_public_conditional_decision():
    prediction, target = prediction_fixture()
    oracle = np.array([[.1, .9, .8, .7]])
    values, evidence = audit.score(prediction, target, oracle)
    assert values["regret"] == pytest.approx(.15, abs=1e-14)
    assert values["defer"] == .25
    assert values["nll"] == pytest.approx(-prediction["log_prob"].mean())
    assert values["brier"] == pytest.approx(np.mean((prediction["prob_positive"]-(target > 0))**2))
    assert evidence["action_counts"] == {"0": 2, "1": 1, "2": 1}
    assert evidence["always_defer_regret"] == pytest.approx(.05)
    assert evidence["context_metrics"]["regret"] == pytest.approx([.15])


@pytest.mark.parametrize("field", ["component_variance", "log_weights", "prob_positive", "log_prob"])
def test_malformed_saved_distributions_are_rejected(field):
    prediction, target = prediction_fixture()
    prediction[field] = prediction[field].copy()
    prediction[field].flat[0] += .3
    with pytest.raises(ValueError):
        audit.validate_prediction(prediction, target, 1)


def all_rows():
    rows = []
    for phase in ("base", "shift"):
        for cohort in range(3):
            for arm in (*audit.ARMS, "full_gp"):
                for seed in ((None,) if arm == "full_gp" else audit.SEEDS):
                    rows.append({"phase": phase, "cohort": cohort, "arm": arm, "fit_seed": seed,
                        "regret": 0. if arm == "full_gp" else .8 if arm == "centered16" else 1.,
                        "nll": .2, "brier": .1, "mse": 1., "defer": .3,
                        "negative": .2, "positive": .5, "always_defer_regret": .1})
    return rows


def test_exact_twentyfour_rule_conditions_and_all_four_controls():
    result = audit.classification(all_rows())
    assert result["gate"] == "CONTINUE_TO_SELECTIVE_REFINEMENT"
    assert len(result["checks"]) == 8
    assert {(r["phase"], r["control"]) for r in result["checks"]} == {
        (phase, arm) for phase in ("base", "shift") for arm in audit.CONTROLS}
    assert sum(r[k] for r in result["checks"] for k in
               ("ten_percent_regret_gain", "nll_noninferiority", "all_cohort_mean_wins")) == 24


def test_one_losing_cohort_blocks_favorable_overall_mean():
    rows = all_rows()
    for row in rows:
        if row["phase"] == "base" and row["arm"] == "centered16":
            row["regret"] = 1.1 if row["cohort"] == 0 else .1
    result = audit.classification(rows)
    assert result["gate"] == "DO_NOT_ADVANCE_THIS_CANDIDATE"
    base = [r for r in result["checks"] if r["phase"] == "base"]
    assert all(r["ten_percent_regret_gain"] and not r["all_cohort_mean_wins"] for r in base)


def test_nll_is_additive_margin_and_zero_baseline_does_not_pass_tie():
    rows = all_rows()
    for row in rows:
        if row["arm"] == "centered16":
            row["nll"] = .220001
    assert audit.classification(rows)["gate"] == "DO_NOT_ADVANCE_THIS_CANDIDATE"
    for row in rows:
        row["regret"], row["nll"] = 0., -1.
    result = audit.classification(rows)
    assert all(not r["ten_percent_regret_gain"] for r in result["checks"])
    assert result["gate"] == "DO_NOT_ADVANCE_THIS_CANDIDATE"


def resources():
    rows = []
    for phase, blocks in (("base", 4), ("shift", 8)):
        for arm in audit.ARMS:
            for seed in audit.SEEDS:
                centered = arm.startswith("centered")
                wide = arm == "static32"
                rows.append({"phase": phase, "arm": arm, "fit_seed": seed,
                    "requests_per_archive": 4, "cached_static": not centered,
                    "archive_array_bytes": 384*blocks,
                    "parameter_bytes": 16 if "nystrom" in arm else 9216 if wide else 4992,
                    "buffer_bytes": 256 if "nystrom" in arm else 0,
                    "cache_tensor_bytes": 0 if centered else blocks*(20992 if wide else 6400),
                    "timings": [{"archive_seconds": .01, "queries_seconds": .03, "total_seconds": .04} for _ in range(20)],
                    "median_context_seconds": .04})
        rows.append({"phase": phase, "arm": "full_gp", "fit_seed": None,
            "requests_per_archive": 4, "cached_static": False,
            "archive_array_bytes": 384*blocks, "timings": [.05]*20, "median_context_seconds": .05,
            "note": "Exact-law NumPy reference recomputes archive factorization for each request."})
    return rows


def test_resources_include_all_five_cache_tensors_and_all_repetitions():
    rows = resources()
    audit.validate_resources(rows)
    rows[0]["cache_tensor_bytes"] -= 8
    with pytest.raises(ValueError, match="cache tensor"):
        audit.validate_resources(rows)


def test_latency_medians_and_nonadditive_scope_corruption_fail():
    rows = resources()
    rows[0]["median_context_seconds"] = .03
    with pytest.raises(ValueError, match="median"):
        audit.validate_resources(rows)
    rows = resources()
    rows[0]["timings"][0]["total_seconds"] = .05
    with pytest.raises(ValueError, match="latency scope"):
        audit.validate_resources(rows)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_checkpoint_barrier_and_all_fifteen_fit_records(tmp_path):
    fits, checkpoints = [], {}
    for seed in audit.SEEDS:
        for arm in audit.ARMS:
            row = {"arm": arm, "fit_seed": seed, "parameters": audit.parameter_count(arm),
                   "updates": 512, "request_exposures": 32768, "train_seconds": .1,
                   "curve": [{"epoch": e, "train_nll": -.2, "mean_gradient_norm_before_clip": .1}
                             for e in range(1, 17)]}
            fits.append(row)
            dump(tmp_path/f"{arm}-{seed}-fit.json", row)
            path = tmp_path/f"{arm}-{seed}.pt"
            path.write_bytes(b"Opaque fixture; never decoded.")
            checkpoints[path.name] = audit.descriptor(path)["sha256"]
    dump(tmp_path/"fits.json", fits)
    dump(tmp_path/"started.json", {"started_unix": 1.})
    dump(tmp_path/"checkpoint-barrier.json", {"checkpoints": checkpoints, "recorded_unix": 2.,
                                            "evaluation_generation_started": False})
    result = audit.fit_metadata(tmp_path)
    assert len(result["checkpoints"]) == 15
    assert result["request_exposures_per_fit"] == 32768
    (tmp_path/"static16-11.pt").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="barrier"):
        audit.fit_metadata(tmp_path)


def test_data_support_and_private_index_are_validated_without_rng_replay():
    data = repeated_location(2)
    audit.validate_data(data, 1, 1, 2, 2., basis=1, fewshots=2)
    data["private_selected_block"][0, 0] = 2
    with pytest.raises(ValueError, match="private index"):
        audit.validate_data(data, 1, 1, 2, 2., basis=1, fewshots=2)
    data["private_selected_block"][0, 0] = 0
    data["qx"][0, 0, 0] = 3.
    with pytest.raises(ValueError, match="coordinate support"):
        audit.validate_data(data, 1, 1, 2, 2., basis=1, fewshots=2)


def opaque_run_fixture(tmp_path, monkeypatch):
    root, folder = tmp_path/"repo", tmp_path/"run"
    monkeypatch.setattr(audit, "ROOT", root)
    sources = {}
    for name in audit.SOURCES:
        for path in (root/name, folder/"source"/name):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"Fabricated source identity only.")
        sources[name] = audit.descriptor(root/name)["sha256"]
    environment = {"python": audit.sys.version, "numpy": np.__version__,
        "scipy": audit.scipy.__version__, "platform": audit.platform.platform(),
        "machine": audit.platform.machine(), "threads": 1, "torch": "opaque producer runtime"}
    dump(folder/"registration.json", {"config": audit.CONFIG, "sources": sources, "environment": environment})
    stems = [f"{arm}-{seed}" for seed in audit.SEEDS for arm in audit.ARMS]
    names = {"started.json", "train.npz", "fits.json", "checkpoint-barrier.json",
             "metrics.json", "resources.json", "summary.json"}
    names.update(stem+suffix for stem in stems for suffix in (".pt", "-fit.json"))
    for phase in ("base", "shift"):
        for cohort in range(3):
            stem = f"{phase}-{cohort}"
            names.update((stem+"-data.npz", stem+"-full_gp.npz"))
            names.update(stem+"-"+fit+".npz" for fit in stems)
    for name in names:
        (folder/name).write_bytes(b"Never numerically decoded in authentication fixture.")
    dump(folder/"run-receipt.json", {"state": "EXITED", "exit_code": 0, "elapsed_seconds": 1.})
    refresh_manifest(folder)
    return root, folder


def refresh_manifest(folder):
    dump(folder/"manifest.json", {str(path.relative_to(folder)): audit.descriptor(path)
        for path in folder.rglob("*") if path.is_file() and path.name != "manifest.json"})


def test_full_source_and_raw_inventory_authentication_is_opaque(tmp_path, monkeypatch):
    root, folder = opaque_run_fixture(tmp_path, monkeypatch)
    record = audit.authenticate(folder)
    assert record["payload_files"] == 141
    assert len(record["sources"]) == 11
    (root/"scripts/query_feature_study.py").write_bytes(b"altered source")
    with pytest.raises(ValueError, match="source/snapshot"):
        audit.authenticate(folder)


def test_rehashed_extra_payload_and_unclosed_producer_rejected(tmp_path, monkeypatch):
    _, folder = opaque_run_fixture(tmp_path, monkeypatch)
    (folder/"unexpected.txt").write_text("extra")
    refresh_manifest(folder)
    with pytest.raises(ValueError, match="payload roster"):
        audit.authenticate(folder)
    (folder/"unexpected.txt").unlink()
    dump(folder/"run-receipt.json", {"state": "FAILED", "exit_code": 1, "elapsed_seconds": 1.})
    refresh_manifest(folder)
    with pytest.raises(ValueError, match="producer receipt"):
        audit.authenticate(folder)
