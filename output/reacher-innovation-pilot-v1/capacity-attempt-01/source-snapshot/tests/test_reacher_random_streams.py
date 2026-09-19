"""Catch the actual protocol defect, without model or simulator evaluation."""

import hashlib
import json

import numpy as np
import pytest

from openjev.research import reacher_random_streams
from openjev.research.reacher_random_streams import (
    collisions,
    concrete_streams,
    domain_bases,
    generator_manifest,
    validate_separation,
)


def old_plan():
    return {"train_episodes": 768, "prediction_episodes": 96, "control_episodes": 64,
            "steps": 50, "train_seed": 64100001, "prediction_seed": 66200001,
            "control_seed": 66300001, "schedule_seed": 66400001, "noise_seed": 66500001,
            "exploration_seed": 66600001, "candidate_seed": 66700001,
            "filter_seed": 66800001, "bootstrap_seed": 66900001}


def test_equal_future_noise_draws_survive_candidate_anchors():
    plan = old_plan()
    noise = np.random.default_rng(plan["noise_seed"] + 200000).normal(0., .05, (50, 2))
    bank = np.random.default_rng(plan["candidate_seed"]).normal(0., .25, (64, 64, 4, 2))
    # Candidate seven is untouched by the runner's zero/six constant anchors.
    np.testing.assert_array_equal(np.clip(bank[0, 7], -1, 1).astype(np.float32),
                                  np.clip(5 * noise[28:32], -1, 1).astype(np.float32))


def test_legacy_cross_role_collisions_are_rejected():
    found = collisions(concrete_streams(old_plan()))
    row = next(row for row in found if row["seed"] == 66700001)
    assert set(row["roles"]) == {"prediction/exploration/0", "control/actuator_noise/0",
                                  "planner/candidate_bank/0"}
    assert any(set(row["roles"]) == {"floor/uniform/0", "analysis/bootstrap/0"} for row in found)
    with pytest.raises(ValueError, match="Cross-role"):
        validate_separation(old_plan())


def test_domain_blocks_are_deterministic_and_all_concrete_seeds_are_distinct():
    plan = old_plan() | domain_bases("reacher-reward-residual-control-v2")
    assert domain_bases("reacher-reward-residual-control-v2") == domain_bases("reacher-reward-residual-control-v2")
    previous = concrete_streams(old_plan(), include_train=True)
    current = validate_separation(plan, prior_registries=[previous])
    assert len(current) == len(set(current.values()))
    assert all(0 <= seed < 2**48 + 300000 for seed in current.values())
    assert len(current) == 4 * 96 + 3 * 64 + 50 + 64 + 2


def test_cross_study_overlap_is_rejected_even_under_a_different_role():
    plan = old_plan() | domain_bases("reacher-reward-residual-control-v2")
    with pytest.raises(ValueError, match="prior"):
        validate_separation(plan, prior_registries=[{"unrelated/role": plan["candidate_seed"]}])


def test_new_namespace_has_no_concrete_overlap():
    first = old_plan() | domain_bases("engineering-a")
    second = old_plan() | domain_bases("engineering-b")
    validate_separation(second, prior_registries=[concrete_streams(first)])


def test_filter_child_generators_match_native_constructor_and_are_distinct():
    registry = {"planner/particle_filter/0": 931, "control/actuator_noise/0": 932}
    manifest = generator_manifest(registry)
    native_children = np.random.SeedSequence(931).spawn(3)
    assert len(manifest) == 4
    assert len({row["initial_state_sha256"] for row in manifest.values()}) == 4
    for index, role in enumerate(("initial", "process_noise", "resample")):
        child = manifest[f"planner/particle_filter/0/{role}"]
        assert child["entropy"] == 931 and child["spawn_key"] == [index]
        assert child["bit_generator"] == "PCG64" and child["draws_for_manifest"] == 0
        native_generator = np.random.Generator(np.random.PCG64(native_children[index]))
        native_state = json.dumps(native_generator.bit_generator.state, sort_keys=True,
                                  separators=(",", ":"))
        assert child["initial_state_sha256"] == hashlib.sha256(native_state.encode()).hexdigest()
    assert manifest["control/actuator_noise/0"]["spawn_key"] == []


def test_renaming_a_role_does_not_hide_identical_generator_state():
    manifest = generator_manifest({"public/candidate": 731, "hidden/disturbance": 731})
    assert manifest["public/candidate"]["initial_state_sha256"] == manifest["hidden/disturbance"]["initial_state_sha256"]


def test_distinct_current_seeds_do_not_hide_identical_generator_states(monkeypatch):
    plan = old_plan() | domain_bases("reacher-reward-residual-control-v2")
    registry = concrete_streams(plan)
    assert len(registry) == len(set(registry.values()))

    def colliding_manifest(current):
        manifest = generator_manifest(current)
        roles = list(manifest)
        manifest[roles[1]]["initial_state_sha256"] = manifest[roles[0]]["initial_state_sha256"]
        return manifest

    monkeypatch.setattr(reacher_random_streams, "generator_manifest", colliding_manifest)
    with pytest.raises(ValueError, match="Current concrete generators have identical initial states"):
        validate_separation(plan)


def test_distinct_prior_seeds_do_not_hide_identical_generator_states(monkeypatch):
    plan = old_plan() | domain_bases("reacher-reward-residual-control-v2")
    current = concrete_streams(plan)
    prior = {"prior/unrelated_role": 731}
    assert set(current.values()).isdisjoint(prior.values())
    current_state = next(iter(generator_manifest(current).values()))["initial_state_sha256"]

    def colliding_prior_manifest(registry):
        manifest = generator_manifest(registry)
        if registry == prior:
            manifest["prior/unrelated_role"]["initial_state_sha256"] = current_state
        return manifest

    monkeypatch.setattr(reacher_random_streams, "generator_manifest", colliding_prior_manifest)
    with pytest.raises(ValueError, match="Current generator states overlap a prior generator manifest"):
        validate_separation(plan, prior_registries=[prior])
