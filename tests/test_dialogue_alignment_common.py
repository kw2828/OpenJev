"""Synthetic actor provenance, geometry, and paired initializer checks."""
from __future__ import annotations

import copy

import dialogue_alignment_common as c
import numpy as np
import pytest
from test_study_dialogue_typed import fixture


def setup():
    fit, evaluation, queries, arrays = fixture()
    offsets = np.asarray([0, 2, 5, 9, 11, 16], np.int64)
    index = {"unique_texts": 5, "queries": [{"query_index": 0,
             "candidate_feature_indices": [2, 3, 4, 5], "candidate_token_ids": [1, 2, 3, 4]}]}
    schema = c.schema_metadata(index, offsets)
    schema["tokens"] = np.random.default_rng(10).normal(size=(16, 384)).astype(np.float32)
    schema["priors"] = np.concatenate([np.full(n, 1/n, np.float32) for n in np.diff(offsets)])
    return fit, evaluation, c.typed.public_candidate_types(queries), arrays, schema


def test_metadata_adapter_preserves_inherited_authentication_bookkeeping(monkeypatch):
    called = []
    def inherited(budget, payloads):
        budget.check()
        budget.progress["hashed_files"] += 1
        assert payloads is True
        return "authenticated"
    monkeypatch.setattr(c.typed, "load_metadata", inherited)
    assert c.load_metadata(lambda: called.append(True)) == "authenticated"
    assert called == [True]


def test_exact_schema_token_assembly_and_current_label_invariance():
    fit, _, types, arrays, schema = setup()
    data, _ = c.actor(fit, arrays, types, schema, "token_aligned")
    assert data["schema_tokens"].shape == (6, 4, 5, 384)
    for j, (a, b) in enumerate(c.candidate_spans(fit[0], schema)):
        np.testing.assert_array_equal(data["schema_tokens"][0, j, :b-a], schema["tokens"][a:b])
        assert data["schema_mask"][0, j].sum() == b-a
        assert data["schema_prior"][0, j].sum() == pytest.approx(1.)
        assert not data["schema_tokens"][0, j, b-a:].any()
    changed = [{**r, "current_label_index": 3, "derived_bin": "clear",
                "current_value_group": "false", "heldout_service": False} for r in fit]
    again, _ = c.actor(changed, arrays, types, schema, "token_mean")
    for key in data:
        np.testing.assert_array_equal(data[key], again[key])
    flat, _ = c.actor(fit, arrays, types, schema, "flat_stratum")
    original, _ = c.typed.actor(fit, arrays, types)
    assert flat.keys() == original.keys()
    for key in flat:
        np.testing.assert_array_equal(flat[key], original[key])


def test_geometry_counts_real_and_padded_positions_and_microbatch_tail():
    fit, _, _, _, schema = setup()
    g = c.geometry(fit, schema)
    assert g["supported_schema_token_positions"] == 6*14
    assert g["padded_schema_token_positions"] == 6*4*5
    assert g["supported_pairwise_positions"] == 6*3*14
    assert g["padded_pairwise_positions"] == 6*4*3*5
    total = c.effective_geometry(fit*6, schema)
    assert total["microbatches"] == 2
    assert total["rows"] == 36
    assert total["padded_schema_token_positions"] == 36*4*5
    broken = copy.deepcopy(fit[0]); broken["cache"]["candidate_feature_indices"] = [5, 4, 3, 2]
    with pytest.raises(ValueError, match="binding"):
        c.geometry([broken], schema)


def test_common_initialization_is_semantic_and_complete_new_arm_copy():
    import torch

    models, hashes = c.init_models(torch, 6201)
    assert len(set(hashes["common_state_sha256"].values())) == 1
    assert hashes["full_state_sha256"]["token_mean"] == hashes["full_state_sha256"]["token_aligned"]
    assert models["flat_stratum"].decision_mode == "flat"
    assert models["flat_stratum"].mode == "candidate"
    assert not hasattr(models["token_mean"], "turn_projection")
    assert "token_projection.weight" not in hashes["common_tensors"]
