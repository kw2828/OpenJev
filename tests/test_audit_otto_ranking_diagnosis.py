"""Fabricated scalar and closure fixtures, with no saved scientific inputs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/audit_otto_ranking_diagnosis.py"
SPEC = importlib.util.spec_from_file_location("ranking_saved_audit_tests", SOURCE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def row(q, p, legal=(True, True, True, True)):
    return audit.row_statistics(np.asarray(q, np.float32), np.asarray(p, np.float32), np.asarray(legal, bool))


def test_pairwise_partition_has_hand_calculated_legal_mse():
    result = row([0, 2, 4, 6], [0, 4, 8, 10])
    metric = result["metrics"]
    assert metric["legal_centered_mse"] == 2.75
    assert metric["pair_mse_best_best"] == 0
    assert metric["pair_mse_best_rest"] == 2.25
    assert metric["pair_mse_rest_rest"] == .5
    assert result["teacher_margin"] == 2
    assert result["max_residuals"] == {"pair_mse": 0, "strict_margin": 0}
    assert row([100, 102, 104, 106], [-8, -4, 0, 2])["metrics"] == metric


def test_strict_binary32_near_boundary_differs_from_exact_argmin():
    epsilon = np.float32(1e-10)
    below = np.nextafter(epsilon, np.float32(0))
    result = row([below, 0, 2, 3], [below, 0, 2, 3])
    assert result["action"] == 0
    assert result["metrics"]["raw_gap"] == float(below)
    assert result["metrics"]["first_argmin_match"] == 1
    assert result["metrics"]["selection_differs_from_strict_mass"] == 1
    assert result["metrics"]["strict_argmin_raw_gap"] == 0
    assert row([epsilon, 0, 2, 3], [epsilon, 0, 2, 3])["action"] == 1
    assert row([0, 0, -100, 0], [0, 0, -100, 0], [False, True, False, True])["action"] == 1


def test_extreme_finite_scores_preserve_f64_gap_when_f32_subtraction_overflows():
    largest = np.finfo(np.float32).max
    result = row([-largest, largest, 0, 0], [largest, -largest, 0, 0], [True, True, False, False])
    assert result["action"] == 1
    assert result["metrics"]["raw_gap"] == 2 * float(largest)
    assert np.isfinite(list(result["metrics"].values())).all()
    assert result["max_residuals"]["strict_margin"] == 0


@pytest.mark.parametrize(("q", "p", "key"), [
    ([0, 1, 2, 3], [0, 2, 3, 4], "near_all_best_mass"),
    ([0, 1, 2, 3], [1, 0, 3, 4], "near_no_best_mass"),
    ([0, 1, 2, 3], [0, 0, 3, 4], "near_mixed_correct_mass"),
    ([1, 0, 2, 3], [0, 0, 3, 4], "near_mixed_wrong_mass"),
])
def test_near_classes_form_exhaustive_partition(q, p, key):
    values = row(q, p)["metrics"]
    assert values[key] == 1
    assert sum(values[k] for k in ("near_all_best_mass", "near_no_best_mass",
                                  "near_mixed_correct_mass", "near_mixed_wrong_mass")) == 1


def test_best_best_pairs_and_unsupported_margin_are_not_dropped():
    result = row([0, 0, 0, 0], [0, 2, 4, 6])
    assert result["teacher_margin"] is None
    assert result["metrics"]["pair_mse_best_best"] == 5
    assert result["metrics"]["pair_mse_best_rest"] == result["metrics"]["pair_mse_rest_rest"] == 0
    assert result["correct"] is True


def test_equal_episode_weight_retains_empty_episode_and_margin_support():
    q = np.array([[0, 2, 3, 4]] * 3, np.float32)
    good = audit.matrix(q[:1], q[:1], np.ones((1, 4), bool))
    wrong = audit.matrix(q, np.array([[2, 0, 3, 4]] * 3, np.float32), np.ones((3, 4), bool))
    empty = good[:0]
    result = audit.combine([audit.individual(good), audit.individual(wrong), audit.individual(empty)])
    assert result["rows"] == 4 and result["episodes"] == 3 and result["supported_episodes"] == 2
    assert result["weight_mass"] == pytest.approx(2 / 3)
    assert result["metrics"]["agreement"] == pytest.approx(1 / 3)
    assert result["metrics"]["raw_gap"] == pytest.approx(2 / 3)
    assert result["teacher_margin"] == {"rows": 4, "mass": 2 / 3, "contribution": 4 / 3, "conditional_mean": 2}
    seed_mean = audit.combine([result, result, result], seeds=True)
    assert seed_mean["episodes"] == 3 and seed_mean["rows"] == 4
    assert seed_mean["supported_episodes"] == 2


def test_phase_contributions_use_full_denominator_and_sum_to_full_delta():
    q = np.tile(np.array([0, 2, 3, 4], np.float32), (4, 1))
    p = np.tile(np.array([2, 0, 3, 4], np.float32), (4, 1))
    baseline = audit.matrix(q, p, np.ones((4, 4), bool))
    p[2:] = q[2:]
    candidate = audit.matrix(q, p, np.ones((4, 4), bool))
    full = audit.paired(baseline, candidate, 4)
    initial = audit.paired(baseline[:2], candidate[:2], 4)
    post = audit.paired(baseline[2:], candidate[2:], 4)
    primary = audit.paired(baseline[2:], candidate[2:], 2)
    assert list(full["cells"]) == ["CC", "CW", "WC", "WW"]
    assert full["cells"]["WC"]["mass"] == .5
    assert full["cells"]["WW"]["mass"] == .5
    for key, expected in (("agreement", .5), ("raw_gap", -1.)):
        total = lambda value, k=key: sum(c["delta"][k] for c in value["cells"].values())
        assert total(full) == expected == total(initial) + total(post)
        reweight = total(post) - total(primary)
        assert expected == total(primary) + total(initial) + reweight
    assert post["rows"] == 2 and post["normalization_rows"] == 4 and post["weight_mass"] == .5


def test_correct_correct_transitions_retain_positive_raw_gap():
    q = np.array([[np.float32(9e-11), 0, 2, 3]], np.float32)
    b = audit.matrix(q, np.array([[1, 0, 2, 3]], np.float32), np.ones((1, 4), bool))
    c = audit.matrix(q, np.array([[0, 1, 2, 3]], np.float32), np.ones((1, 4), bool))
    result = audit.paired(b, c, 1)
    assert result["cells"]["CC"]["rows"] == 1
    assert result["cells"]["CC"]["delta"]["raw_gap"] == float(q[0, 0]) > 0
    assert all(result["cells"][name]["mass"] == 0 for name in ("CW", "WC", "WW"))


@pytest.mark.parametrize("change", ["dtype", "shape", "nonfinite", "mask", "empty"])
def test_invalid_row_contract_rejected(change):
    q, p, mask = np.zeros(4, np.float32), np.ones(4, np.float32), np.ones(4, bool)
    if change == "dtype":
        q = q.astype(np.float64)
    elif change == "shape":
        p = p[:3]
    elif change == "nonfinite":
        q[3] = np.nan
    elif change == "mask":
        mask = mask.astype(np.int64)
    else:
        mask[:] = False
    with pytest.raises(ValueError):
        audit.row_statistics(q, p, mask)


def geometry_fixture():
    identities = [{"episode_id": f"{r}-{c}-{a}", "regime": r, "case": c, "arm": a}
                  for r in ("lambda3", "lambda4") for c in range(6)
                  for a in ("analytic", "neural", "period4_hold")]
    episode_lengths = np.array([1, 2, 3, 4, 5, 6] * 6, np.int64)
    rows = [(i, s, min(4, int(n) - s)) for i, n in enumerate(episode_lengths) for s in range(0, int(n), 4)]
    lengths = np.array([r[2] for r in rows], np.int64)
    valid = np.arange(4) < lengths[:, None]
    q = np.zeros((len(rows), 4, 4), np.float32)
    return {"episode_ids": tuple(i["episode_id"] for i in identities),
            "episode_regimes": tuple(i["regime"] for i in identities), "episode_lengths": episode_lengths,
            "episode_index": np.array([r[0] for r in rows], np.int64),
            "step_offsets": np.array([r[1] for r in rows], np.int64), "lengths": lengths,
            "valid_mask": valid, "nonquery_mask": valid & (np.arange(4) > 0), "targets": q,
            "legal": np.repeat(valid[:, :, None], 4, axis=2), "query_scores": q[:, 0].copy()}, identities


def test_geometry_retains_every_tail_and_rejects_replaced_window():
    windows, identities = geometry_fixture()
    _, _, episodes, steps = audit.geometry(windows, identities)
    assert list(zip(episodes, steps, strict=True)) == [(i, s) for i, n in enumerate(windows["episode_lengths"])
                                                      for s in range(n) if s % 4]
    assert 0 not in episodes and 5 in steps
    corrupt = copy.deepcopy(windows)
    corrupt["step_offsets"][-1] = 0
    with pytest.raises(ValueError, match="chronological"):
        audit.geometry(corrupt, identities)
    corrupt = copy.deepcopy(windows)
    corrupt["query_scores"][0, 0] = 1
    with pytest.raises(ValueError, match="query anchors"):
        audit.geometry(corrupt, identities)


def test_comparison_rejects_trailing_keys_nonfinite_and_wrong_exact_types():
    audit.Comparison().same({"x": 1.}, {"x": 1. + 1e-13})
    for actual in ({"x": 1., "extra": 0}, {"x": float("nan")}, {"x": 1.1}):
        with pytest.raises(ValueError):
            audit.Comparison().same({"x": 1.}, actual)
    with pytest.raises(ValueError):
        audit.Comparison().same({"x": 1}, {"x": True})
    with pytest.raises(ValueError):
        audit.Comparison().same([None], [None, None])


def test_closed_diagnostic_requires_original_payload_pins_and_exact_argument_set(tmp_path):
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path

    def pin(path):
        raw = Path(path).read_bytes()
        return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    launch = save("launch.json", {"command": "fabricated"})
    directory = tmp_path / "diagnosis-01"
    directory.mkdir()
    for name, value in (("started.json", {"phase": "diagnosis", "launch": {"command": "fabricated"},
                                          "plan_sha256": "plan"}), ("diagnosis.json", {})):
        (directory / name).write_text(json.dumps(value))
    limits = {"seconds": 240, "rss_bytes": 2 * 1024**3, "output_bytes": 128 * 1024**2}
    worker = {"version": "v", "phase": "diagnosis", "limits": limits,
              "requires_successful_original_supervisor": True, "new_scientific_gate": False,
              "npz_decodes": 26, "model_calls": 0, "native_calls": 0, "optimizer_calls": 0,
              "sources": {}, "inputs": {}, "peak_rss_bytes": 100, "started_ns": 10, "finished_ns": 20,
              "wall_seconds": 1e-8, "files": {name: pin(directory / name) for name in ("started.json", "diagnosis.json")}}
    worker_path = directory / "receipt.json"
    worker_path.write_text(json.dumps(worker))
    terminal = save("terminal.json", {})
    options = {"--plan": "plan", "--plan-sha256": "plan", "--supervision": str(launch), "--output": str(directory)}
    joins = []

    def original_join(_worker, _terminal, **kwargs):
        joins.append(kwargs)
        return options

    helper = SimpleNamespace(pin=pin, read=lambda p: json.loads(Path(p).read_text()), VERSION="v", LIMITS=limits,
                             original_join=original_join)
    job = SimpleNamespace(plan={"outputs": {"diagnosis": str(directory)}, "sources": {}, "inputs": {}}, check=lambda: None)
    args = SimpleNamespace(worker=worker_path, worker_sha256=pin(worker_path)["sha256"], terminal=terminal,
                           terminal_sha256=pin(terminal)["sha256"], plan=tmp_path / "plan.json", plan_sha256="plan")
    acknowledged, descriptors = audit.diagnostic_closure(helper, job, args)
    assert acknowledged == worker and set(descriptors) == {"worker", "terminal", "launch", "result"}
    assert joins[0]["script"] == "diagnose_otto_ranking.py" and joins[0]["cap"] == 240
    options["--unplanned"] = "x"
    with pytest.raises(ValueError, match="arguments"):
        audit.diagnostic_closure(helper, job, args)
    del options["--unplanned"]
    (directory / "diagnosis.json").write_text("{\"extra\":1}")
    with pytest.raises(ValueError, match="payload"):
        audit.diagnostic_closure(helper, job, args)
