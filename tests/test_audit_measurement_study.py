"""Fabricated independent-auditor checks; no scientific data or producer calls."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_measurement_study as a


def fixture(steps=4, batch=1):
    points = a.grid()
    x = np.broadcast_to(points[:steps], (batch, steps, 2)).copy()
    y = np.broadcast_to(np.sin(np.arange(steps) / 7), (batch, steps)).copy()
    paths = np.array([[points[[i, i+2, i+4, i+6]] for i in (0, 1, 17, 18)]] * 4)
    paths = np.broadcast_to(paths.reshape(1, 4, 4, 4, 2), (batch, 4, 4, 4, 2)).copy()
    return {"x": x, "y": y, "paths": paths, "exposure": np.zeros((batch, 4, 4))}


def test_one_observation_rational_original_law_and_no_future_noise():
    point = np.array([[0., 0.]])
    mean, cov, rank, cutoff = a.conditional(point, np.array([2.]), np.repeat(point, 4, axis=0))
    prior, noise = 1.00001, .09
    expected_mean = prior * 2 / (prior + noise)
    expected_variance = prior * noise / (prior + noise)
    np.testing.assert_allclose(mean, expected_mean, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(cov, expected_variance, atol=1e-14, rtol=1e-14)
    prediction = a.moments(np.tile(mean, 4), np.tile(cov, (4, 4)), 1)
    np.testing.assert_allclose(prediction["variance"], expected_variance, atol=1e-14, rtol=1e-14)
    assert rank == 1 and cutoff == 0.


def test_direct_correlated_linear_measurement_not_independent_pseudonoise():
    x = np.array([[-1., 0.], [.25, 0.], [1., 0.]])
    y = np.array([1., -.5, 2.])
    points = np.array([[-.25, 0.], [.5, 0.]])
    projection = np.array([[1., 2., 0.], [0., 1., 1.]])
    k = a.kernel(x, x) + .09 * np.eye(3)
    z = projection @ y
    cross = a.kernel(points, x) @ projection.T
    expected_mean = cross @ np.linalg.solve(projection @ k @ projection.T, z)
    expected_cov = a.kernel(points, points) - cross @ np.linalg.solve(projection @ k @ projection.T, cross.T)
    mean, covariance, rank, cutoff = a.conditional(x, y, points, projection)
    np.testing.assert_allclose(mean, expected_mean, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(covariance, expected_cov, atol=1e-12, rtol=1e-12)
    assert rank == 2 and cutoff > 0
    # Event noise transforms with A A^T and is correlated for overlapping sums.
    assert (.09 * projection @ projection.T)[0, 1] == .18


def test_dependent_measurement_rows_retain_same_posterior():
    data = fixture(3)
    x, y = data["x"][0], data["y"][0]
    points = data["paths"][0, 0, 0]
    one = np.array([[1., 2., 3.]])
    repeated = np.concatenate((one, 2 * one, np.zeros_like(one)))
    reference = a.conditional(x, y, points, one)
    actual = a.conditional(x, y, points, repeated)
    for lhs, rhs in zip(actual[:2], reference[:2], strict=True):
        np.testing.assert_allclose(lhs, rhs, atol=1e-12, rtol=1e-12)
    assert actual[2] == reference[2] == 1
    with pytest.raises(ValueError, match="nonempty"):
        a.conditional(x, y, points, np.zeros((2, 3)))


def test_full_column_rank_projection_recovers_all_observation_information():
    data = fixture(3)
    x, y = data["x"][0], data["y"][0]
    points = data["paths"][0].reshape(-1, 2)
    direct = a.conditional(x, y, points)
    projected = a.conditional(x, y, points, np.array([[1., 2., 0.], [0., 1., 2.], [2., 0., 1.], [1., 1., 1.]]))
    for lhs, rhs in zip(direct[:2], projected[:2], strict=True):
        np.testing.assert_allclose(lhs, rhs, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("kind", a.KINDS)
def test_basis_owned_finite_fixed_shape(kind):
    first, second = a.basis(kind), a.basis(kind)
    assert first.shape == (118, 289) and first.dtype == np.float64
    assert np.isfinite(first).all() and not np.shares_memory(first, second)
    np.testing.assert_array_equal(first, second)
    if kind != "bins":
        np.testing.assert_allclose(first @ first.T, np.eye(118), atol=1e-12, rtol=1e-12)
    else:
        np.testing.assert_array_equal(first.sum(0), np.ones(289))
        np.testing.assert_array_equal(first.argmax(0), np.arange(289) * 118 // 289)


def test_dct_low_frequency_order_is_squared_frequency_then_lexicographic():
    result = a.basis("dct")
    n = np.arange(17)
    constant = np.full(17, 1 / math.sqrt(17))
    first = math.sqrt(2/17) * np.cos(np.pi * (n + .5) / 17)
    np.testing.assert_allclose(result[0], np.outer(constant, constant).ravel(), atol=1e-15)
    np.testing.assert_allclose(result[1], np.outer(constant, first).ravel(), atol=1e-15)
    np.testing.assert_allclose(result[2], np.outer(first, constant).ravel(), atol=1e-15)


def test_packed_mask_exact_little_bits_and_tail_padding():
    result = a.packed_mask([np.array([0, 7, 8, 288])])
    assert result.dtype == np.uint8 and result.shape == (1, 37)
    assert result[0, 0] == 129 and result[0, 1] == 1 and result[0, 36] == 1
    assert np.count_nonzero(np.unpackbits(result, bitorder="little")) == 4


def test_raw_hybrid_sorted_labels_and_exact_prefix_prediction():
    data = fixture(4)
    order = [3, 1, 2, 0]
    data["x"] = data["x"][:, order]
    data["y"] = data["y"][:, order]
    before = copy.deepcopy(data)
    reference = a.predict(data, "full")
    for kind in a.KINDS:
        state = a.reconstructed_state(data["x"], data["y"], kind + "118")
        np.testing.assert_array_equal(state["values"][0, :4], np.sin(np.arange(4) / 7))
        assert not state["values"][:, 4:].any()
        prediction = a.predict(data, kind + "118")
        for key in reference:
            np.testing.assert_array_equal(prediction[key], reference[key])
        np.testing.assert_array_equal(prediction["rank"], [4])
        np.testing.assert_array_equal(prediction["rank_cutoff"], [0.])
    for key in data:
        np.testing.assert_array_equal(data[key], before[key])


def test_119th_event_converts_every_label_once_into_fixed_bin_sums():
    data = fixture(119)
    state = a.reconstructed_state(data["x"], data["y"], "bins118")
    expected = np.zeros(118)
    for identity, value in enumerate(data["y"][0]):
        expected[identity * 118 // 289] += value
    np.testing.assert_allclose(state["values"][0], expected, atol=1e-14)
    assert np.unpackbits(state["mask"], bitorder="little").sum() == 119


def test_new_coverage_tie_lowest_grid_id_old_control_tie_oldest():
    points = a.grid()
    newest = a.retained_indices(points[118::-1], "coverage118")
    np.testing.assert_array_equal(a.grid_ids(points[118::-1][newest]), np.arange(1, 119))
    oldest = a.retained_indices(points[98::-1], "coverage98")
    np.testing.assert_array_equal(a.grid_ids(points[98::-1][oldest]), np.arange(97, -1, -1))


def test_recent98_retains_chronological_labels_and_zero_unused_storage():
    data = fixture(102)
    state = a.reconstructed_state(data["x"], data["y"], "recent98")
    np.testing.assert_array_equal(state["ids"][0], np.arange(4, 102))
    np.testing.assert_array_equal(state["values"], data["y"][:, 4:])
    small = fixture(3)
    padded = a.reconstructed_state(small["x"], small["y"], "recent98")
    assert not padded["ids"][:, 3:].any() and not padded["values"][:, 3:].any()


def test_metric_gaussian_tail_defer_and_private_exposure_score_only():
    prediction = {"mean": np.full((1, 1, 4), .5), "variance": np.ones((1, 1, 4)),
                  "risk": np.full((1, 1, 4), .5)}
    ref = {"risk": np.array([[[0., .3, .5, 1.]]])}
    result = a.metrics(prediction, ref, np.full((1, 1, 4), .5))
    assert result["defer"] == 1. and result["regret"] == pytest.approx(.18)
    assert result["nll"] == pytest.approx(.5 * math.log(2 * math.pi))
    assert result["brier"] == .25 and result["coverage90"] == 1.
    other = a.metrics(prediction, ref, np.full((1, 1, 4), 10.))
    assert other["regret"] == result["regret"] and other["defer"] == result["defer"]
    bad = copy.deepcopy(prediction)
    bad["risk"] = ndtr(bad["mean"])
    with pytest.raises(ValueError, match="tail"):
        a.metrics(bad, ref, np.zeros((1, 1, 4)))


@pytest.mark.parametrize("change", [lambda d: d["x"].__setitem__((0, 0, 0), .1),
                                   lambda d: d["x"].__setitem__((0, 0), d["x"][0, 1]),
                                   lambda d: d["paths"].__setitem__((0, 0, 0, 1, 1), 1.25),
                                   lambda d: d["y"].__setitem__((0, 0), np.nan)])
def test_malformed_public_evidence_rejected(change):
    data = fixture(4)
    a.validate_data(data, 1, 4, "axial")
    change(data)
    with pytest.raises(ValueError):
        a.validate_data(data, 1, 4, "axial")


def saved_state(data, method):
    result = a.reconstructed_state(data["x"], data["y"], method)
    result["step"] = np.array([data["y"].shape[1]], dtype=np.int64)
    if method[:-3] in a.KINDS:
        result["kind"] = np.array([a.KINDS.index(method[:-3])], dtype=np.uint8)
    return result


@pytest.mark.parametrize("method", a.METHODS[:-1])
def test_saved_boundary_independently_validated_with_complete_storage(method):
    data = fixture(120)
    state = saved_state(data, method)
    a.validate_state(state, data, method, {k: a.basis(k) for k in a.KINDS})
    assert a.resident_bytes(method, 120) <= 1024
    bad = copy.deepcopy(state)
    bad["values"][0, 4] += .01
    with pytest.raises(ValueError, match="state|sums"):
        a.validate_state(bad, data, method, {k: a.basis(k) for k in a.KINDS})


@pytest.mark.parametrize("change", [
    lambda s: s["mask"].__setitem__((0, -1), 128),
    lambda s: s["step"].__setitem__(0, 3),
    lambda s: s["kind"].__setitem__(0, 2),
    lambda s: s.update(extra=np.zeros(1)),
    lambda s: s.update(kind=s["kind"].astype(np.int64)),
    lambda s: s["values"].__setitem__((0, 0), -0.),
])
def test_boundary_rejects_padding_counter_kind_extra_storage_and_signed_zero(change):
    data = fixture(4)
    state = saved_state(data, "spectral118")
    change(state)
    with pytest.raises(ValueError):
        a.validate_state(state, data, "spectral118", {})


def all_rows():
    result = []
    for population, cohort, method in a.identities():
        metric = {name: .1 for name in a.METRICS}
        metric["always_defer_regret"] = .2
        metric["regret"] = (0. if population in ("base", "shift") else .01) if method == "spectral118" else .1
        result.append({"population": population, "cohort": cohort, "method": method, "metrics": metric})
    return result


def intervals():
    return {name: {"ci_high": -.001} for name in ("long", "long_shift")}


def test_all_26_fixed_conditions_and_no_alternative_selection():
    rows = all_rows()
    tests = a.conditions(rows, intervals())
    assert len(tests) == 26 and len({row["name"] for row in tests}) == 26
    assert all(row["pass"] for row in tests)
    assert sum("mean-versus" in row["name"] for row in tests) == 6
    assert sum("exact-prefix" in row["name"] for row in tests) == 6
    assert sum("best-raw" in row["name"] for row in tests) == 6
    for row in rows:
        if row["population"] == "base" and row["cohort"] == 2:
            row["metrics"]["regret"] = 1e-4 if row["method"] == "spectral118" else 0.
    failed = a.conditions(rows, intervals())
    assert [row["name"] for row in failed if not row["pass"]] == ["base/2/exact-prefix"]


def test_one_bad_cohort_cannot_be_rescued_by_population_mean():
    rows = all_rows()
    for row in rows:
        if row["population"] == "long" and row["method"] == "spectral118":
            row["metrics"]["regret"] = .091 if row["cohort"] == 2 else 0.
    tests = {row["name"]: row for row in a.conditions(rows, intervals())}
    assert not tests["long/2/best-raw"]["pass"]
    assert all(tests["long/mean-versus-" + method]["pass"] for method in a.RAW)


def test_exact_paired_ci_tie_is_failure_and_fixed_nll_limit_inclusive():
    bounds = intervals()
    bounds["long"]["ci_high"] = 0.
    rows = all_rows()
    for row in rows:
        if row["method"] == "spectral118":
            row["metrics"]["nll"] = .1 + .02
    result = {row["name"]: row for row in a.conditions(rows, bounds)}
    assert not result["long/paired-upper"]["pass"]
    assert all(result[p + "/nll"]["pass"] for p in a.POPULATIONS)


@pytest.mark.parametrize("change", [lambda rows: rows.pop(),
                                    lambda rows: rows.__setitem__(0, rows[1]),
                                    lambda rows: rows[0]["metrics"].update(regret=float("nan"))])
def test_rule_rejects_missing_duplicate_or_nonfinite_group(change):
    rows = all_rows()
    change(rows)
    with pytest.raises(ValueError):
        a.conditions(rows, intervals())


def test_whole_field_bootstrap_retains_every_context_and_fixed_seed():
    values = np.full(384, -.125)
    result = a.bootstrap(values, 553261824)
    assert result == {"difference_mean": -.125, "ci_low": -.125, "ci_high": -.125,
                      "context_differences": values.tolist(), "seed": 553261824, "repetitions": 1000}
    with pytest.raises(ValueError):
        a.bootstrap(np.repeat(values, 4), 553261824)
    with pytest.raises(ValueError):
        a.bootstrap(values, 999)


def all_resources():
    return [{"population": p, "method": m,
             "logical_bytes": a.resident_bytes(m, a.POPULATIONS[p]["observations"]),
             "median_ms": 5.5, "all_ms": list(range(1, 11)), "peak_python_bytes": 12345}
            for p in a.POPULATIONS for m in a.METHODS]


@pytest.mark.parametrize("change", [lambda rows: rows.pop(),
    lambda rows: rows[0].update(logical_bytes=1021),
    lambda rows: rows[0].update(median_ms=1.),
    lambda rows: rows[0]["all_ms"].__setitem__(0, -1.),
    lambda rows: rows[0].update(peak_python_bytes=True),
])
def test_resource_roster_bytes_original_samples_and_median(change):
    rows = all_resources()
    a.validate_resources(rows)
    change(rows)
    with pytest.raises(ValueError):
        a.validate_resources(rows)


def test_whole_context_regret_average_precedes_resampling():
    truth = {"risk": np.array([[[0., 1., 1., 1.], [1., 0., 1., 1.]]])}
    prediction = {"risk": np.full((1, 2, 4), .5)}
    np.testing.assert_allclose(a.field_regrets(prediction, truth), [.18], rtol=0, atol=1e-16)


def write_json(path, value):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def opaque_evidence(tmp_path, monkeypatch):
    """An opaque roster proves admission without loading any numerical file."""
    root, folder = tmp_path / "repo", tmp_path / "run"
    root.mkdir()
    folder.mkdir()
    monkeypatch.setattr(a, "ROOT", root)
    for name in a.expected_files():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"opaque fixture bytes")
    sources = {}
    for name in a.SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("fixture source " + name).encode())
        (folder / "source" / name).write_bytes(path.read_bytes())
        sources[name] = a.descriptor(path)
    parent = root / "research/retention-results/summary.json"
    write_json(parent, {"gate": "FAIL"})
    plan = {"config": a.CONFIG, "sources": sources, "parent_result": a.descriptor(parent),
            "environment": {"python": sys.version, "numpy": np.__version__, "scipy": a.scipy.__version__,
                            "platform": a.platform.platform(), "threads": 1}}
    write_json(folder / "registration.json", plan)
    pin = a.descriptor(folder / "registration.json")
    write_json(folder / "started.json", {"version": "measurement-v1", "registration": pin})
    write_json(folder / "run-status.json", {"state": "COMPLETE", "elapsed_seconds": 10.,
                                           "metric_groups": 84, "resources": 28})
    process = tmp_path / "run-process.json"
    write_json(process, {"state": "EXITED", "returncode": 0, "elapsed_seconds": 12.,
                         "command": [str(root / ".venv/bin/python"), str(root / "scripts/measurement_study.py"),
                                     "run", "--out", str(folder), "--registration-sha256", pin["sha256"]]})
    repin_manifest(folder)
    return folder, process


def repin_manifest(folder):
    write_json(folder / "manifest.json", {"files": {name: a.descriptor(folder / name) for name in a.expected_files()}})


def test_exact_opaque_inventory_admitted_without_array_decode(tmp_path, monkeypatch):
    folder, process = opaque_evidence(tmp_path, monkeypatch)
    monkeypatch.setattr(a.np, "load", lambda *_args, **_kwargs: pytest.fail("no array decode permitted"))
    result = a.authenticate(folder, process)
    assert result["manifest_members"] == 183 and len(result["sources"]) == 8
    assert len(a.identities()) == 84 and len(a.expected_files()) == 183


@pytest.mark.parametrize("corruption", ["current_source", "source_snapshot", "parent", "missing_pin", "extra_file",
    "changed_payload", "failed_status", "failed_process", "boolean_returncode", "over_cap", "bad_nested_time",
    "wrong_command", "wrong_registration", "wrong_runtime", "wrong_roster"])
def test_rejects_invalid_provenance_before_numerical_decode(tmp_path, monkeypatch, corruption):
    folder, process = opaque_evidence(tmp_path, monkeypatch)
    monkeypatch.setattr(a.np, "load", lambda *_args, **_kwargs: pytest.fail("admission must precede numerical reads"))
    if corruption == "current_source":
        (a.ROOT / min(a.SOURCES)).write_text("changed")
    elif corruption == "source_snapshot":
        (folder / "source" / min(a.SOURCES)).write_text("changed")
        repin_manifest(folder)
    elif corruption == "parent":
        (a.ROOT / "research/retention-results/summary.json").write_text("changed")
    elif corruption == "missing_pin":
        manifest = a.read(folder / "manifest.json")
        manifest["files"].pop("data/base-0.npz")
        write_json(folder / "manifest.json", manifest)
    elif corruption == "extra_file":
        (folder / "unregistered.txt").write_text("extra")
    elif corruption == "changed_payload":
        (folder / "pred/base-0-full.npz").write_text("changed")
    elif corruption == "failed_status":
        status = a.read(folder / "run-status.json")
        status["state"] = "FAILED"
        write_json(folder / "run-status.json", status)
        repin_manifest(folder)
    elif corruption in ("wrong_runtime", "wrong_roster"):
        plan = a.read(folder / "registration.json")
        if corruption == "wrong_runtime":
            plan["environment"]["threads"] = 2
        else:
            plan["sources"].pop(min(a.SOURCES))
        write_json(folder / "registration.json", plan)
        repin_manifest(folder)
    else:
        terminal = a.read(process)
        if corruption == "failed_process":
            terminal["returncode"] = 1
        elif corruption == "boolean_returncode":
            terminal["returncode"] = False
        elif corruption == "over_cap":
            terminal["elapsed_seconds"] = 1201.
        elif corruption == "bad_nested_time":
            terminal["elapsed_seconds"] = 1.
        elif corruption == "wrong_command":
            terminal["command"][2] = "register"
        elif corruption == "wrong_registration":
            terminal["command"][-1] = "0" * 64
        write_json(process, terminal)
    with pytest.raises(ValueError):
        a.audit(folder, process=process)


def test_array_schema_and_nonfinite_predictions_fail_on_fabricated_file(tmp_path):
    path = tmp_path / "tiny.npz"
    np.savez(path, x=np.ones(2), hidden=np.zeros(1))
    with pytest.raises(ValueError, match="schema"):
        a.load_arrays(path, {"x"}, {"npz_decodes": 0, "array_loads": 0})
    with pytest.raises(ValueError, match="numerical agreement"):
        a.close(np.array([np.inf]), np.array([1.]), "nonfinite")
