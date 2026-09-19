"""Runner boundaries use synthetic arrays or already completed tiny fixtures."""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import reacher_geometry_study as study
import torch

from openjev.research import reacher_geometry_protocol as protocol


def test_union_preserves_every_slot_and_exact_signed_zero_identity():
    values = np.zeros((5, 12, 2), dtype=np.float32)
    values[1, 0, 0] = -0.
    values[3, 1, 0] = .5
    unique, mapping, first = study.deduplicate_sequences(values)
    assert first.tolist() == [0, 1, 3]
    assert mapping.tolist() == [0, 1, 0, 2, 0]
    assert values.tobytes() == unique[mapping].tobytes()
    unique[:] = .75
    assert values[0, 0, 0] == 0.


def test_inherited_members_include_all_six_fits_and_exact_public_histories():
    names = study.inherited_members({"arms": list(protocol.ARMS)})
    assert len(names) == 50
    assert sum(name.endswith("checkpoint.pt") for name in names) == 6
    assert {name for name in names if name.startswith("control/")} == {
        f"control/{panel}/residual_gru-pair0/episodes.{suffix}"
        for panel in protocol.PANELS for suffix in ("npz", "json")}


def test_prepare_does_not_draw_or_construct_models(tmp_path):
    cfg = protocol.settings()
    parent = {key: cfg[key] for key in ("hidden_size", "mlp_width", "dt", "noise_std")}
    with patch.object(study, "authenticated_sources", return_value=(parent, {"fixture": True})), \
         patch.object(study, "stream_contract", return_value={"fixture": True}), \
         patch.object(study, "sha", return_value="0" * 64), \
         patch.object(study.training, "_construct", side_effect=AssertionError("No model")), \
         patch.object(protocol, "draw_control_inputs", side_effect=AssertionError("No draw")), \
         patch.object(protocol, "diagnostic_inputs", side_effect=AssertionError("No draw")):
        study.prepare(tmp_path / "freeze", cap_seconds=123, audit_cap_seconds=321)
    saved = study.read(tmp_path / "freeze" / "plan.json")
    assert saved["new_fits"] == 0 and saved["cap_seconds"] == 123
    assert saved["fit_order"] == cfg["fit_order"]


@pytest.fixture
def inherited_engineering(tmp_path):
    import reacher_geometry_fixture as fixture

    if not (study.ROOT / fixture.PARENT / "execution" / "completed.json").exists():
        pytest.skip("Optional completed engineering checkpoint fixture is not present")
    parent, source = fixture.parent()
    plan = protocol.settings(engineering=True)
    plan.update(engineering=True, hidden_size=parent["hidden_size"], mlp_width=parent["mlp_width"],
                runtime=parent["runtime"], parent_source=source)
    study.copy_inherited(plan, tmp_path, float("inf"))
    return plan, tmp_path


def test_restore_is_inference_only_and_preserves_all_tensors(inherited_engineering):
    plan, folder = inherited_engineering
    rng_before = torch.get_rng_state().clone()
    with patch.object(torch.optim, "Adam", side_effect=AssertionError("No optimizer")), \
         patch.object(study.training, "CacheTrainer", side_effect=AssertionError("No trainer")):
        models = study.restore_students(plan, folder, float("inf"))
    assert torch.equal(rng_before, torch.get_rng_state())
    assert list(models) == plan["fit_order"]
    for name, model in models.items():
        weights = torch.load(folder / "inherited" / "fits" / name / "weights.pt", weights_only=True)
        snapshot = torch.load(folder / "model-states" / f"{name}-before.pt", weights_only=True)
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        assert all(torch.equal(value, weights[key]) and torch.equal(value, snapshot[key])
                   for key, value in model.state_dict().items())


@pytest.mark.parametrize("field,value", [("kind", "cached_gru"), ("failed", True),
    ("cursor", {"epoch": 0, "batch": 0}), ("settings", {"bad": True})])
def test_restore_rejects_resealed_wrong_configuration(inherited_engineering, field, value):
    plan, folder = inherited_engineering
    original_load = torch.load

    def corrupted(path, *args, **kwargs):
        payload = original_load(path, *args, **kwargs)
        if Path(path).name == "checkpoint.pt":
            payload = copy.deepcopy(payload)
            payload[field] = value
            payload["integrity_sha256"] = study.training.canonical_state_hash(
                {k: v for k, v in payload.items() if k != "integrity_sha256"})
        return payload

    with patch.object(torch, "load", side_effect=corrupted), \
         patch.object(study.training, "_construct", side_effect=AssertionError("No invalid model")), \
         pytest.raises(ValueError, match="checkpoint|schedule"):
        study.restore_students(plan, folder, float("inf"))
