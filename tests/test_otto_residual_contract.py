"""Fabricated cache-boundary and exact view-roster checks, with no model calls."""
from copy import deepcopy

import numpy as np
import pytest

from openjev.research import otto_residual_contract as contract


def example():
    lengths = (1, 9)
    offsets = np.array([0, 1, 10], dtype=np.int64)
    step = np.concatenate([np.arange(n) for n in lengths])
    query = step % 4 == 0
    scores = np.full((10, 4), np.nan, np.float32)
    scores[query] = np.array([-0., 1., 2., 3.], np.float32)
    base = np.ones((10, 4), np.float32)
    base[query] = scores[query]
    shadow = np.full((10, 4), np.nan, np.float32)
    prior = query & (step > 0)
    shadow[prior] = 0.
    cue = np.zeros((10, 8), np.float32)
    cue[:, 0] = 1.
    cue[step == 0] = np.nan
    cache = {"version": contract.CACHE_VERSION, "query_period": 4, "base_action": base,
             "joint_action": base.copy(), "shadow_prior": shadow, "cues": cue, "query_scores": scores,
             "query_mask": query, "prior_mask": prior, "key_mask": step > 0, "episode_offsets": offsets,
             "work_counts": {"some_actual_work": 2}}
    features = np.zeros((10, 31), np.float32)
    features[:, 15] = (step / 2188).astype(np.float32)
    features[:, 16] = ((step % 4) / 2188).astype(np.float32)
    features[:, 17] = 1.
    return cache, features


def test_masks_reset_at_episode_boundaries_and_validators_do_not_mutate():
    cache, features = example()
    saved = deepcopy(cache)
    masks = contract.validate_feature_inputs(features, cache["query_scores"], cache["episode_offsets"])
    assert np.flatnonzero(masks["query_mask"]).tolist() == [0, 1, 5, 9]
    assert np.flatnonzero(masks["prior_mask"]).tolist() == [5, 9]
    contract.validate_cache(cache)
    for name, value in cache.items():
        if isinstance(value, np.ndarray):
            assert value.tobytes() == saved[name].tobytes()
        else:
            assert value == saved[name]


@pytest.mark.parametrize("field", ["cues", "shadow_prior", "query_scores"])
def test_unconsumed_poison_is_allowed_but_consumed_poison_is_rejected(field):
    cache, _ = example()
    mask = cache[{"cues": "key_mask", "shadow_prior": "prior_mask", "query_scores": "query_mask"}[field]]
    cache[field][~mask] = np.inf
    contract.validate_cache(cache)
    cache[field][np.flatnonzero(mask)[0], 0] = np.nan
    with pytest.raises(ValueError):
        contract.validate_cache(cache)


@pytest.mark.parametrize("field", ["base_action", "joint_action"])
def test_query_identity_includes_signed_zero(field):
    cache, _ = example()
    cache[field][0, 0] = 0.
    with pytest.raises(ValueError, match="bitwise"):
        contract.validate_cache(cache)


@pytest.mark.parametrize("offsets", [[0, 0, 10], [0, 9, 8, 10], [1, 10], [0, 11], [0, -1, 10], [0]])
def test_malformed_complete_episode_geometry_rejected(offsets):
    cache, _ = example()
    cache["episode_offsets"] = np.array(offsets, dtype=np.int64)
    with pytest.raises(ValueError):
        contract.validate_cache(cache)


@pytest.mark.parametrize("name", ["query_mask", "prior_mask", "key_mask"])
def test_wrong_query_or_first_step_schedule_rejected(name):
    cache, _ = example()
    cache[name][0] = not cache[name][0]
    with pytest.raises(ValueError):
        contract.validate_cache(cache)


@pytest.mark.parametrize("column", [15, 16, 17])
def test_public_chronology_must_reset_exactly(column):
    cache, features = example()
    features[1, column] += 1.
    with pytest.raises(ValueError):
        contract.validate_feature_inputs(features, cache["query_scores"], cache["episode_offsets"])


@pytest.mark.parametrize("defect", ["extra_field", "missing_field", "dtype", "nonfinite_action", "negative_count",
                                    "boolean_count", "boolean_period", "version"])
def test_incompatible_cache_is_rejected(defect):
    cache, _ = example()
    if defect == "extra_field":
        cache["targets"] = np.zeros((10, 4), np.float32)
    elif defect == "missing_field":
        del cache["joint_action"]
    elif defect == "dtype":
        cache["cues"] = cache["cues"].astype(np.float64)
    elif defect == "nonfinite_action":
        cache["base_action"][2, 0] = np.inf
    elif defect == "negative_count":
        cache["work_counts"]["bad"] = -1
    elif defect == "boolean_count":
        cache["work_counts"]["bad"] = True
    elif defect == "boolean_period":
        cache["query_period"] = True
    else:
        cache["version"] = "wrong"
    with pytest.raises(ValueError):
        contract.validate_cache(cache)


def test_complete_rosters_have_every_baseline_once_and_every_ratio():
    dev = contract.view_specs("dev")
    assert len(dev) == len(set(dev)) == 72
    for method in contract.METHODS:
        rows = [row for row in dev if row[0] == method]
        assert len(rows) == (3 if method in contract.BASELINES else 12)
    for tau in contract.TAUS:
        confirm = contract.view_specs("confirm", tau)
        assert len(confirm) == len(set(confirm)) == 27
        assert {t for m, t, _ in confirm if m in contract.RLS_METHODS} == {tau}
        assert {s for _, _, s in confirm} == set(contract.FIT_SEEDS)
    assert {contract.view_name(m, t) for m, t, _ in dev} == {
        *contract.BASELINES, *(f"{m}@tau={t:g}" for m in contract.RLS_METHODS for t in contract.TAUS)}


@pytest.mark.parametrize("stage,tau", [("dev", 1.), ("confirm", None), ("test", 1.), (True, None),
                                      ("confirm", .5), ("confirm", True), ("confirm", float("nan"))])
def test_incomplete_or_posthoc_rosters_rejected(stage, tau):
    with pytest.raises(ValueError):
        contract.view_specs(stage, tau)


@pytest.mark.parametrize("method,tau", [("pretrained", 1.), ("joint_aux", .1), ("rls_full", None),
                                       ("rls_full", -1), ("rls_full", float("inf")), ("rls_full", "1"),
                                       ("rls_shrink_100", 1.), ([], None), ("rls_full", 10**400)])
def test_method_configuration_cannot_add_an_identical_or_undeclared_control(method, tau):
    with pytest.raises(ValueError):
        contract.method_config(method, tau)
