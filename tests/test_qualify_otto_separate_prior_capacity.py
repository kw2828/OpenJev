"""Synthetic input geometry and durable boundary tests; no timed qualification."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openjev.research import otto_prequery_data as data
from openjev.research import otto_sampled_forecast_data as sampling

SPEC = importlib.util.spec_from_file_location("_separate_prior_capacity_fixture",
    Path(__file__).resolve().parents[1] / "scripts/qualify_otto_separate_prior_capacity.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_fixed_worstcase_inputs_have_complete_prior_and_sampled_nonquery_weights(tmp_path):
    run = runner.Run(SimpleNamespace(output=tmp_path)); run.check = lambda: None
    run.np, run.data = np, data
    history, descriptors = run.synthetic(torch, sampling)
    assert history["features"].shape == (6 * 2188, 31)
    assert history["episode_offsets"].tolist() == [i * 2188 for i in range(7)]
    assert int(history["prior_mask"].sum()) == 6 * 546
    assert int((history["weights"] > 0).sum()) == 6 * 8 * 3
    assert not np.any((history["weights"] > 0) & history["prior_mask"])
    assert np.all(history["prior_weights"][history["prior_mask"]] == 1 / (54 * 546))
    assert set(descriptors) <= set(history)
    saved = json.loads((tmp_path / "synthetic.json").read_text())
    for i, selection in enumerate(saved["selections"]):
        assert selection == sampling.select_windows(2188, runner.CONFIG["selection_seeds"][i])
        assert len(selection["start_offsets"]) == 8
    for start in range(0, 2188, 32):
        packet = data.batch_chunk(history, list(range(6)), start)
        assert packet["model_inputs"]["lengths"].tolist() == [min(32, 2188 - start)] * 6
        assert np.any(packet["prior_mask"])
    assert runner.CONFIG["admission_seconds"] == 16200 and runner.CONFIG["training_cap_seconds"] == 21600
    assert runner.LIMITS["seconds"] == 180 and runner.CONFIG["fixed_overhead_seconds"] == 180
    assert len(runner.KINDS) == 8
    assert set(runner.CELLS.values()) == {(architecture, readout, objective)
        for architecture in ("innovation", "innovation_gru")
        for readout in ("shared", "separate") for objective in ("mse", "query_aux")}


def test_failed_return_never_acknowledges_capacity_operation(tmp_path):
    run = runner.Run(SimpleNamespace(output=tmp_path)); run.check = lambda: None
    run.clock = SimpleNamespace(now_ns=lambda: 10)
    events = []

    def emit(record):
        events.append(record)
        if record["event"] == "return":
            raise OSError("injected durable return failure")

    run.event = emit
    with pytest.raises(OSError, match="durable return"):
        run.call("innovation_shared_aux", "optimizer_update", lambda: None)
    assert run.receipt["optimizer_updates"] == 0 and run.receipt["pending"]["operation"] == "optimizer_update"
    assert [r["event"] for r in events] == ["attempt", "return"]
