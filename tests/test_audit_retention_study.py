"""Fabricated independent-auditor checks; no study generation or model imports."""
from __future__ import annotations

import copy
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_retention_study as audit


def coordinates(count):
    return np.array([(a, b) for a in np.arange(-2., 2.01, .25)
                     for b in np.arange(-2., 2.01, .25)])[:count]


def law(x, z):
    delta = x[:, None] - z[None]
    return np.exp(-np.sum(delta * delta, axis=-1) / 2) + 1e-5 * np.all(delta == 0, axis=-1)


def posterior(x, y):
    prior = law(x, x)
    system = prior + .09 * np.eye(len(y))
    return prior @ np.linalg.solve(system, y), prior - prior @ np.linalg.solve(system, prior)


def paths():
    options = np.array([[[a + .5 * t, b] for t in range(4)]
                        for a, b in ((-2., -2.), (-1.5, -1.), (-1., 0.), (-.5, 1.))])
    return np.broadcast_to(options, (1, 4, 4, 4, 2)).copy()


def data(steps=10):
    x = coordinates(steps)[None]
    return {"x": x, "y": np.sin(x[..., 0]) + np.cos(x[..., 1]),
            "paths": paths(), "exposure": np.zeros((1, 4, 4))}


def zero_policy():
    return {key: np.zeros(shape) for key, shape in audit.POLICY_SHAPES.items()}


def test_kernel_nugget_is_spatial_and_not_independent_observation_noise():
    x = np.array([[0., 0.], [0., 0.], [1e-6, 0.]])
    got = audit.kernel(x, x)
    assert got[0, 0] == got[0, 1] == 1.00001
    assert got[0, 2] == pytest.approx(math.exp(-.5e-12), abs=0, rel=1e-15)


def test_full_reference_one_observation_and_repeated_path_has_full_covariance():
    x = np.zeros((1, 1, 2))
    y = np.array([[.7]])
    requests = np.zeros((1, 1, 4, 4, 2))
    got = audit.exact_reference(x, y, requests)
    k = 1.00001
    expected_mean, expected_var = .7 * k / (k + .09), k * .09 / (k + .09)
    np.testing.assert_allclose(got["mean"], expected_mean, rtol=1e-13)
    np.testing.assert_allclose(got["variance"], expected_var, rtol=1e-13)
    np.testing.assert_allclose(got["diag_variance"], expected_var / 4, rtol=1e-13)
    np.testing.assert_allclose(got["risk"], ndtr((expected_mean - .5) / math.sqrt(expected_var)))


def test_projected_replay_without_deletion_matches_dense_gaussian():
    d = data(8)
    traces = np.full((1, 8), -1, dtype=np.int16)
    got, work = audit.replay_projected(d["x"], d["y"], traces, "kl8")
    mean, cov = posterior(d["x"][0], d["y"][0])
    np.testing.assert_allclose(got["mean"][0], mean, atol=2e-12)
    np.testing.assert_allclose(got["cov"][0], cov, atol=2e-12)
    assert work == {"priority_checks": 0, "near_tie_priority_checks": 0}
    np.testing.assert_array_equal(got["Z"], d["x"])


def test_first_fifo_compression_is_dense_posterior_marginal():
    d = data(10)
    traces = np.full((1, 10), -1, dtype=np.int16)
    traces[:, -1] = 0
    got, work = audit.replay_projected(d["x"], d["y"], traces, "fifo9")
    mean, cov = posterior(d["x"][0], d["y"][0])
    np.testing.assert_array_equal(got["Z"], d["x"][:, 1:])
    np.testing.assert_allclose(got["mean"][0], mean[1:], atol=2e-12)
    np.testing.assert_allclose(got["cov"][0], cov[1:, 1:], atol=2e-12)
    assert work["priority_checks"] == 0
    traces[0, -1] = 1
    with pytest.raises(ValueError, match="FIFO"):
        audit.replay_projected(d["x"], d["y"], traces, "fifo9")


def conditional_kl(x, mean, cov):
    prior = law(x, x)
    result = []
    for j in range(len(x)):
        keep = np.arange(len(x)) != j
        beta = np.linalg.solve(prior[np.ix_(keep, keep)], prior[keep, j])
        gamma = np.linalg.solve(cov[np.ix_(keep, keep)], cov[keep, j])
        r = prior[j, j] - prior[j, keep] @ beta
        v = cov[j, j] - cov[j, keep] @ gamma
        expected_square = cov[j, j] - 2 * beta @ cov[keep, j]
        expected_square += beta @ cov[np.ix_(keep, keep)] @ beta
        expected_square += (mean[j] - beta @ mean[keep]) ** 2
        result.append(.5 * (math.log(r / v) + expected_square / r - 1))
    return np.array(result)


def test_deletion_kl_matches_independent_conditional_gaussian_formula():
    x = np.array([[-1., -.5], [.25, .5], [1., -.25]])
    mean, cov = posterior(x, np.array([.2, -.6, 1.1]))
    expected = conditional_kl(x, mean, cov)
    got = audit.deletion_priorities(x[None], mean[None], cov[None])[0]
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=1e-11)
    scores = audit.deletion_priorities(x[None], mean[None], cov[None], zero_policy())[0]
    np.testing.assert_allclose(scores, np.log(expected + 1e-12), atol=1e-11)


def test_analytic_drop_replay_rejects_nonminimum_and_counts_tolerance_ties(monkeypatch):
    d = data(9)
    traces = np.full((1, 9), -1, dtype=np.int16)
    mean, cov = posterior(d["x"][0], d["y"][0])
    expected = conditional_kl(d["x"][0], mean, cov)
    traces[0, -1] = expected.argmin()
    got, work = audit.replay_projected(d["x"], d["y"], traces, "kl8")
    keep = np.arange(9) != traces[0, -1]
    np.testing.assert_allclose(got["mean"][0], mean[keep], atol=3e-12)
    assert work["priority_checks"] == 1
    traces[0, -1] = expected.argmax()
    with pytest.raises(ValueError, match="minimizes"):
        audit.replay_projected(d["x"], d["y"], traces, "kl8")
    monkeypatch.setattr(audit, "deletion_priorities", lambda *args: np.zeros((1, 9)))
    _, work = audit.replay_projected(d["x"], d["y"], traces, "kl8")
    assert work == {"priority_checks": 1, "near_tie_priority_checks": 1}


def test_fic_batch_state_matches_data_space_gaussian_conditioning():
    d = data(7)
    got = audit.batch_fic(d["x"], d["y"])
    anchors = np.array([(a, b) for a in (-1.8, .1, 1.8) for b in (-1.8, .1, 1.8)])
    prior = law(anchors, anchors)
    cross = law(anchors, d["x"][0])
    projected = cross.T @ np.linalg.solve(prior, cross)
    system = projected + np.diag(1.00001 - projected.diagonal() + .09)
    expected_mean = cross @ np.linalg.solve(system, d["y"][0])
    expected_cov = prior - cross @ np.linalg.solve(system, cross.T)
    np.testing.assert_allclose(got["mean"][0], expected_mean, atol=1e-12)
    np.testing.assert_allclose(got["cov"][0], expected_cov, atol=1e-12)


def test_raw_retention_oldest_ties_and_label_pairing():
    x = np.array([[[float(i), 0.] for i in range(42)]])
    y = np.arange(42, dtype=float)[None]
    for method in ("recent41", "coverage41"):
        got = audit.replay_raw(x, y, method)
        np.testing.assert_array_equal(got["Z"], x[:, 1:])
        np.testing.assert_array_equal(got["y"], y[:, 1:])


def test_saved_state_validation_rejects_extra_storage_and_replay_corruption():
    d = data(10)
    traces = np.full((1, 10), -1, dtype=np.int16)
    traces[:, -1] = 0
    state, _ = audit.replay_projected(d["x"], d["y"], traces, "fifo9")
    state.update(step=np.array(10, dtype=np.int64), traces=traces,
                 resident_bytes=np.array(904, dtype=np.int64))
    counts = dict.fromkeys(("projected_state_replays", "priority_checks", "near_tie_priority_checks"), 0)
    audit.validate_state(state, d, "fifo9", None, counts, lambda: None)
    assert counts["projected_state_replays"] == 1
    with pytest.raises(ValueError, match="hidden arrays"):
        audit.validate_state(state | {"archive": d["x"]}, d, "fifo9", None, counts, lambda: None)
    state["mean"][0, 0] += .01
    with pytest.raises(ValueError, match="replay"):
        audit.validate_state(state, d, "fifo9", None, counts, lambda: None)


@pytest.mark.parametrize("damage", ["repeated", "dtype", "path", "private_shape"])
def test_data_schema_rejects_malformed_saved_inputs(damage):
    d = data()
    audit.validate_data(d, 1, 10, "axial")
    if damage == "repeated":
        d["x"][0, 1] = d["x"][0, 0]
    elif damage == "dtype":
        d["y"] = d["y"].astype(np.float32)
    elif damage == "path":
        d["paths"][0, 0, 0, 2, 0] += .25
    else:
        d["exposure"] = d["exposure"][:, :3]
    with pytest.raises(ValueError):
        audit.validate_data(d, 1, 10, "axial")


def test_metrics_conditional_regret_first_ties_and_all_path_nll():
    mean = np.zeros((1, 1, 4))
    variance = np.full_like(mean, .01)
    prediction = {"mean": mean, "variance": variance, "risk": ndtr((mean - .5) / np.sqrt(variance))}
    reference = {"risk": np.array([[[.3, .05, .4, .6]]])}
    exposure = np.array([[[0., .1, -.1, .2]]])
    got = audit.metrics(prediction, reference, exposure)
    assert got["regret"] == pytest.approx(.25)  # All predictions tie, so action zero wins.
    assert got["defer"] == 0
    assert got["always_defer_regret"] == pytest.approx(.13)
    assert got["mse"] == pytest.approx(.015)
    assert got["nll"] == pytest.approx(.5 * (math.log(2 * math.pi * .01) + 1.5))
    assert got["coverage90"] == .75


def rows():
    return [{"phase": phase, "cohort": cohort, "method": method,
             **{metric: (.02 if metric == "regret" and method.startswith("learned-") else
                          .1 if metric == "regret" else .2 if metric == "always_defer_regret" else 0.)
                for metric in audit.METRICS}}
            for phase, cohort, method in audit.identities()]


def test_exact_99_group_roster_and_42_frozen_conditions():
    result = audit.classification(rows())
    assert len(result["checks"]) == 42
    assert all(item["passed"] for item in result["checks"])
    assert result["gate"] == "ADVANCE_RETENTION"
    broken = rows()
    broken[-1] = broken[0]
    with pytest.raises(ValueError, match="99"):
        audit.classification(broken)


def test_best_control_and_per_seed_guard_cannot_be_rescued_by_average():
    values = rows()
    for row in values:
        if row["phase"] == "base" and row["method"] == "learned-11":
            row["regret"] = .11
    checks = {item["name"]: item["passed"] for item in audit.classification(values)["checks"]}
    assert checks["base_regret_vs_kl8"]
    assert not checks["base_learned-11"]
    values = rows()
    for row in values:
        if row["phase"] == "long" and row["method"] == "kl9":
            row["nll"] = -.1
    checks = {item["name"]: item["passed"] for item in audit.classification(values)["checks"]}
    assert not checks["long_nll"]


def training():
    actions = np.zeros((512, 32, 64), dtype=np.int16)
    actions[:, :, :8] = -1
    arrays = {"actions": actions, "rewards": np.full((512, 8, 4), -.01),
              "fields": np.tile(np.arange(256).reshape(32, 8), (16, 1))}
    record = {"fit_seed": 11, "updates": 512, "seconds": 1., "parameter_bytes": 264,
              "group_trajectories": 16384,
              "trace": [{"epoch": i // 32, "update": i + 1, "loss": 0., "mean_reward": -.01,
                         "group_reward_std": 0., "entropy": 1., "gradient_norm": .2,
                         "kl_roundoff_floors": 0} for i in range(512)]}
    return record, arrays


@pytest.mark.parametrize("damage", [None, "warmup", "fields", "reward", "floor", "updates"])
def test_training_512_update_array_and_trace_attestations(damage):
    record, arrays = training()
    if damage == "warmup":
        arrays["actions"][0, 0, 0] = 0
    elif damage == "fields":
        arrays["fields"][0, 0] = 1
    elif damage == "reward":
        record["trace"][3]["mean_reward"] = -.5
    elif damage == "floor":
        record["trace"][0]["kl_roundoff_floors"] = -1
    elif damage == "updates":
        record["updates"] = 511
    if damage:
        with pytest.raises(ValueError):
            audit.validate_training(record, arrays, 11)
    else:
        audit.validate_training(record, arrays, 11)


def resources():
    return [{"phase": phase, "method": method, "timings_seconds": [.01] * 10,
             "median_seconds": .01, "resident_bytes": audit.resident_bytes(method, config["observations"]),
             "python_numpy_traced_peak_bytes": 2000, "traced_current_bytes": 100,
             "native_workspace_measured": False, "retained_gradient_tensors": 0,
             "scope": "One context, complete stream then four requests; native allocations and peak RSS unmeasured."}
            for phase, config in audit.CONFIG["populations"].items() for method in audit.METHODS]


def test_storage_and_gradient_accounting():
    assert [audit.resident_bytes(method, 64) for method in
            ("learned-11", "kl8", "kl9", "fic9", "recent41")] == [1008, 744, 904, 904, 1024]
    assert audit.resident_bytes("learned-11", 64) == 8 * (16 + 8 + 64) + 40 + 264 == 1008
    assert audit.resident_bytes("kl9", 64) == 8 * (18 + 9 + 81) + 40 == 904
    audit.validate_resources(resources())
    for value in (1, False):
        values = resources()
        values[0]["retained_gradient_tensors"] = value
        with pytest.raises(ValueError, match="gradients"):
            audit.validate_resources(values)
    values = resources()
    values[0]["resident_bytes"] = 744
    with pytest.raises(ValueError, match="policy weights"):
        audit.validate_resources(values)
    values = resources()
    values[0]["median_seconds"] = .02
    with pytest.raises(ValueError, match="median"):
        audit.validate_resources(values)


def test_policy_initial_boundary_and_shape():
    values = zero_policy()
    audit.validate_policy(values, initial=True)
    values["output.bias"][0] = .1
    audit.validate_policy(values)
    with pytest.raises(ValueError, match="zero initial"):
        audit.validate_policy(values, initial=True)
    values["hidden.bias"] = np.zeros(3)
    with pytest.raises(ValueError, match="declared array"):
        audit.validate_policy(values)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def evidence(tmp_path, monkeypatch):
    root, folder = tmp_path / "repo", tmp_path / "study"
    root.mkdir()
    folder.mkdir()
    monkeypatch.setattr(audit, "ROOT", root)
    for name in audit.SOURCES:
        source = root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("fabricated source\n" + name)
        snapshot = folder / "source" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
    parent = root / "research/residual-memory-results/summary.json"
    write_json(parent, {"fabricated": True})
    for name in audit.expected_files():
        path = folder / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"opaque evidence, deliberately not NPZ")
    write_json(folder / "registration.json", {"config": copy.deepcopy(audit.CONFIG),
        "sources": {name: audit.descriptor(root / name) for name in audit.SOURCES},
        "parent_result": audit.descriptor(parent), "environment": {"python": sys.version,
        "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(), "threads": 1}})
    write_json(folder / "run-receipt.json", {"state": "EXITED", "exit_code": 0,
                                           "training_updates": 1536, "elapsed_seconds": 1.})
    refresh_manifest(folder)
    return root, folder


def refresh_manifest(folder):
    write_json(folder / "manifest.json", {name: audit.descriptor(folder / name) for name in audit.expected_files()})


def test_complete_opaque_roster_admission_without_decoding(tmp_path, monkeypatch):
    _, folder = evidence(tmp_path, monkeypatch)
    admitted = audit.authenticate(folder)
    assert admitted["manifest_members"] == 226
    assert len(audit.SOURCES) == 13
    assert len([name for name in audit.expected_files() if name.endswith(".npz")]) == 200
    assert len([name for name in audit.expected_files() if name.endswith("-state.npz")]) == 81


@pytest.mark.parametrize("damage", ["source", "snapshot", "payload", "extra", "failed", "config"])
def test_admission_tampering_stops_before_any_array_decode(tmp_path, monkeypatch, damage):
    root, folder = evidence(tmp_path, monkeypatch)
    if damage in ("source", "snapshot"):
        path = (root if damage == "source" else folder / "source") / min(audit.SOURCES)
        path.write_text("changed")
    elif damage == "payload":
        (folder / "train-data.npz").write_bytes(b"changed")
    elif damage == "extra":
        (folder / "unexpected.json").write_text("{}")
    elif damage == "failed":
        receipt = audit.read(folder / "run-receipt.json")
        receipt["exit_code"] = 1
        write_json(folder / "run-receipt.json", receipt)
        refresh_manifest(folder)
    else:
        plan = audit.read(folder / "registration.json")
        plan["config"]["epochs"] = 17
        write_json(folder / "registration.json", plan)
        refresh_manifest(folder)
    def forbidden(*args, **kwargs):
        raise AssertionError("array decoding reached before admission")
    monkeypatch.setattr(audit, "load_arrays", forbidden)
    with pytest.raises(ValueError):
        audit.audit(folder)


def test_compare_rejects_boolean_counter_and_nonfinite_scalar():
    with pytest.raises(ValueError, match="identity"):
        audit.compare(True, 1)
    with pytest.raises(ValueError, match="agreement"):
        audit.compare(float("nan"), 1.)
