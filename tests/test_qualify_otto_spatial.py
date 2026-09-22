"""Independent count and cost arithmetic for synthetic resource qualification."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/qualify_otto_spatial.py"
SPEC = importlib.util.spec_from_file_location("spatial_qualifier_tested", PATH)
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


def records():
    rows = []
    for kind in Q.KINDS:
        for stage in ("model_init", "optimizer_init", "checkpoint_write"):
            rows.append({"kind": kind, "stage": stage, "seconds": .1})
        for stage, sizes, count in (("update", (128, 85), 5), ("diagnostic", (128, 85), 4),
                                    ("deployment", (16,), 7)):
            for batch in sizes:
                for rep in range(count):
                    first = rep == 0 if stage == "diagnostic" else rep < 2
                    row = {"kind": kind, "stage": stage, "batch": batch,
                           "seconds": 100 if first else .01, "repeat": rep}
                    row["first" if stage == "diagnostic" else "warmup"] = first
                    rows.append(row)
    return rows


def test_projection_includes_tails_all_families_and_final_diagnostics():
    result = Q.project(records())
    # 43 full + 85-row tail = 5589; 8 full + same tail = 1109.
    expected = .3 + 80*44*.01 + 53*.01
    assert result["fifteen_fit_seconds"] == pytest.approx(15*expected)
    assert result["two_times_allowance_seconds"] == pytest.approx(30*expected)
    assert result["not_admission"] is True
    for family in result["families"].values():
        assert family["one_fit_seconds"] == pytest.approx(expected)
        assert family["head_only_seconds_for_72_cases_three_seeds_at_cap"] == pytest.approx(.01*2188*72*3)


def test_projection_never_silently_omits_family():
    with pytest.raises(ValueError, match="missing projection"):
        Q.project([r for r in records() if r["kind"] != "cnn"])


def test_timing_report_retains_cold_warm_and_measured_samples():
    distributions = Q.timing_distributions(records())
    assert len(distributions) == 25
    for d in distributions:
        assert d["all_seconds"][0] == 100
        assert d["warm_max"] == .01
        assert len(d["warm_seconds"]) == (5 if d["stage"] == "deployment" else 3)


def test_fixed_fixture_is_local_deterministic_and_has_exact_physical_support():
    before = np.random.get_state()
    first, second = Q.synthetic_fixture(), Q.synthetic_fixture()
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    z, q, sensing, target = first
    assert z.shape == (128, 105, 105) and q.dtype == np.int64
    assert z.dtype == sensing.dtype == target.dtype == np.float32
    assert set(sensing) == {3, 4, 5}
    assert np.all(z >= 0) and np.all(z.sum(axis=(1, 2)) <= 1+1e-6)
    assert np.any(z.sum(axis=(1, 2)) == 0) and np.any((z.sum(axis=(1, 2)) > 0) & (z.sum(axis=(1, 2)) < 1e-10))
    for field, position in zip(z, q, strict=True):
        mask = np.ones((105, 105), dtype=bool)
        x, y = 52-position
        mask[x:x+53, y:y+53] = False
        assert not field[mask].any()


def test_exact_call_allocation():
    assert Q.EXPECTED["torch_forward"] == 5*(2*5+2*4+1)
    assert Q.EXPECTED["numpy_forward"] == 5*(1+2+5)
    assert Q.EXPECTED["optimizer_update"] == Q.EXPECTED["backward"] == 5*2*5


def test_no_overwrite_frozen_plan_or_artifact(tmp_path):
    path = tmp_path/"record.json"
    Q.write(path, {"status": "first"})
    with pytest.raises(FileExistsError):
        Q.write(path, {"status": "replacement"})


def test_clock_failure_preserves_original_failure_and_null_timing(tmp_path):
    run = Q.Run(SimpleNamespace(output=tmp_path/"run", plan_sha256="synthetic-test"))

    class BrokenClock:
        def now_ns(self):
            raise RuntimeError("clock failed")

    def fail():
        run.clock, run.start = BrokenClock(), 10
        raise ValueError("original failure")

    run.bind = fail
    assert run.execute() == 1
    receipt = json.loads((run.out/"receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["finished_ns"] is None
    assert receipt["wall_seconds"] is None
    assert "original failure" in receipt["errors"][0] and "terminal clock" in receipt["errors"][1]
