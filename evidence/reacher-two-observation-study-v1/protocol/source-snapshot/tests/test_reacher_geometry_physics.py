"""Small engineering physics fixtures only; seed410, no fitted students."""

import copy
from dataclasses import replace

import mujoco
import numpy as np
import pytest
import torch

from openjev.research import reacher_geometry_physics as physics
from openjev.research.reacher_adaptive_search import SearchInputs, search
from openjev.research.reacher_geometry_reward import geometry_reward_components
from openjev.research.robotics_reacher import make_env


@pytest.fixture
def native():
    env = make_env()
    env.reset(seed=410)
    try:
        yield env.unwrapped
    finally:
        env.close()


def roots(native, count=1):
    q = np.broadcast_to(native.data.qpos, (count, 4)).copy()
    v = np.broadcast_to(native.data.qvel, (count, 4)).copy()
    v[:, 2:] = 0.
    return q, v, q[:, 2:].astype(np.float32)


def draws(count=1, horizon=12, block=3):
    rng = np.random.default_rng(410)
    chunks = (horizon + block - 1) // block
    values = [rng.normal(size=(count, k, chunks, 2)) for k in (64, 192, 64, 64, 63)]
    return SearchInputs(values[0], values[1], tuple(values[2:]))


def commands(n=1, k=3, h=3):
    return np.linspace(-.4, .4, n * k * h * 2, dtype=np.float32).reshape(n, k, h, 2)


def hand_rollout(model, q, v, target, bank, step):
    n, k, h, _ = bank.shape
    positions = np.zeros((n, k, h + 1, 4), np.float64)
    velocities = np.zeros_like(positions)
    angles = np.zeros((n, k, h, 4), np.float32)
    data = mujoco.MjData(model)
    for i in range(n):
        for j in range(k):
            mujoco.mj_resetData(model, data)
            data.qpos[:], data.qvel[:], data.time = q[i], v[i], step * .02
            mujoco.mj_forward(model, data)
            positions[i, j, 0], velocities[i, j, 0] = data.qpos, data.qvel
            for offset in range(h):
                data.ctrl[:] = bank[i, j, offset]
                mujoco.mj_step(model, data, nstep=2)
                mujoco.mj_rnePostConstraint(model, data)
                positions[i, j, offset + 1], velocities[i, j, offset + 1] = data.qpos, data.qvel
                angles[i, j, offset] = np.r_[np.cos(data.qpos[:2]), np.sin(data.qpos[:2])]
    expanded = np.broadcast_to(target[:, None, None], (n, k, h, 2)).copy()
    component = geometry_reward_components(torch.from_numpy(angles), torch.from_numpy(expanded), torch.from_numpy(bank.copy()), .05)
    scores = np.zeros((n, k), np.float32)
    for offset in range(h):
        scores += np.clip(component.reward.numpy()[:, :, offset], -2.5, 0.)
    return scores, positions, velocities, component


def test_hand_native_rollout_exact_scores_components_and_counts(native):
    q, v, target = roots(native, 2)
    q[1, :2] += [.07, -.11]
    v[1, :2] += [.2, -.1]
    bank = commands(2)
    planner = physics.PhysicsGeometryCEM(native.model)
    expected, positions, velocities, component = hand_rollout(native.model, q, v, target, bank, 12)
    scores, journal = planner.score_bank(q, v, target, bank, step=12)
    np.testing.assert_array_equal(scores, expected)
    saved, meta = journal["banks"][0]["arrays"], journal["banks"][0]["metadata"]
    np.testing.assert_array_equal(saved["qpos"], positions)
    np.testing.assert_array_equal(saved["qvel"], velocities)
    for name in physics._COMPONENTS:
        np.testing.assert_array_equal(saved["geometry_" + name], getattr(component, name).numpy())
    assert meta["candidate_sequences_requested"] == meta["candidate_sequences_scored"] == 6
    assert meta["native_transitions_attempted"] == meta["native_transitions_completed"] == 18
    assert meta["native_substeps_attempted"] == meta["native_substeps_completed"] == 36
    assert meta["reset_calls_completed"] == meta["forward_calls_completed"] == 6
    assert meta["postconstraint_calls_completed"] == 18 and meta["geometry_calls_completed"] == 1
    assert meta["geometry_samples_completed"] == 18 and meta["sum_offsets_completed"] == 3
    assert saved["initialized"].all() and saved["native_completed"].all() and saved["scored"].all()
    assert meta["native_seconds"] + meta["geometry_seconds"] <= meta["wall_seconds"]
    assert not scores.flags.writeable
    assert journal["selected"] is None and journal["status"] == "completed"


def test_cem_kernel_exact_budget_mean_order_and_selected_advance_from_root(native):
    q, v, target = roots(native)
    inputs = draws()
    planner = physics.PhysicsGeometryCEM(native.model)
    result, journal = planner.plan(q, v, target, inputs, step=47)
    assert result.horizon == 3 and result.candidate_evaluations == 256
    assert result.imagined_transitions == 768 and result.stages[-1].mean_candidate_id == 255
    assert [item["arrays"]["commands"].shape for item in journal["banks"]] == [(1, 64, 3, 2)] * 4
    recorded = iter(journal["banks"])
    def saved_callback(bank):
        row = next(recorded)["arrays"]
        np.testing.assert_array_equal(bank, row["commands"])
        return row["scores"]
    reconstructed = search("cem256", inputs, saved_callback, step=47)
    np.testing.assert_array_equal(result.sequences, reconstructed.sequences)
    np.testing.assert_array_equal(result.scores, reconstructed.scores)
    np.testing.assert_array_equal(result.selected_ids, reconstructed.selected_ids)
    assert result.selected_ids[0] == int(result.scores[0].argmax())
    selected = journal["selected"]["arrays"]
    one = result.selected_actions[:, None, None].copy()
    expected, positions, velocities, _ = hand_rollout(native.model, q, v, target, one, 47)
    np.testing.assert_array_equal(selected["qpos"], positions)
    np.testing.assert_array_equal(selected["qvel"], velocities)
    np.testing.assert_array_equal(selected["scores"], expected)
    np.testing.assert_array_equal(selected["qpos"][0, 0, 0], q[0])
    global_id = int(result.selected_ids[0])
    candidate = journal["banks"][global_id // 64]["arrays"]
    np.testing.assert_array_equal(selected["qpos"][0, 0, 1], candidate["qpos"][0, global_id % 64, 1])
    assert planner.snapshot()["lifetime"]["native_transitions_completed"] == 769
    assert planner.snapshot()["lifetime"]["native_substeps_completed"] == 1538


@pytest.mark.parametrize("step,expected", [(0, 12), (47, 3), (49, 1)])
def test_terminal_shortening_keeps_all_paid_candidates(native, step, expected):
    q, v, target = roots(native)
    planner = physics.PhysicsGeometryCEM(native.model)
    result, journal = planner.plan(q, v, target, draws(), step=step)
    assert result.horizon == expected and result.imagined_transitions == 256 * expected
    assert sum(row["metadata"]["native_transitions_completed"] for row in journal["banks"]) == 256 * expected
    assert journal["selected"]["metadata"]["native_transitions_completed"] == 1


def test_candidate_order_and_case_order_do_not_leak_scratch_state(native):
    q, v, target = roots(native, 2)
    q[1, 0] += .3
    bank = commands(2, 4, 2)
    planner = physics.PhysicsGeometryCEM(native.model)
    first, trace = planner.score_bank(q, v, target, bank, step=8)
    second, reverse = planner.score_bank(q[::-1].copy(), v[::-1].copy(), target[::-1].copy(), bank[::-1, ::-1].copy(), step=8)
    np.testing.assert_array_equal(first, second[::-1, ::-1])
    np.testing.assert_array_equal(trace["banks"][0]["arrays"]["qpos"], reverse["banks"][0]["arrays"]["qpos"][::-1, ::-1])


def test_model_roots_inputs_native_state_and_snapshots_are_isolated(native):
    q, v, target = roots(native)
    bank = commands()
    original = tuple(value.copy() for value in (q, v, target, bank, native.data.qpos, native.data.qvel))
    planner = physics.PhysicsGeometryCEM(native.model)
    baseline, journal = planner.score_bank(q, v, target, bank, step=0)
    native.model.opt.gravity[:] = [0., 0., -3.]
    changed, _ = planner.score_bank(q, v, target, bank, step=0)
    np.testing.assert_array_equal(baseline, changed)
    for value, expected in zip((q, v, target, bank, native.data.qpos, native.data.qvel), original, strict=True):
        np.testing.assert_array_equal(value, expected)
    journal["roots"]["qpos"][:] = 900
    snapshot = planner.snapshot()
    snapshot["last_operation"]["banks"][0]["arrays"]["qpos"][:] = -900
    assert (planner.snapshot()["last_operation"]["banks"][0]["arrays"]["qpos"] != -900).all()
    assert planner.configuration()["model_copied"] is True


def test_no_generated_randomness(native, monkeypatch):
    q, v, target = roots(native)
    inputs = draws()
    before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    monkeypatch.setattr(np.random, "default_rng", lambda *a, **k: pytest.fail("Adapter created RNG"))
    planner = physics.PhysicsGeometryCEM(native.model)
    planner.plan(q, v, target, inputs, step=49)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    after = np.random.get_state()
    assert before is not None and np.array_equal(numpy_before[1], after[1]) and numpy_before[2:] == after[2:]


def test_clip_each_float32_step_not_aggregate_cost(native, monkeypatch):
    q, v, target = roots(native)
    bank = np.zeros((1, 1, 3, 2), np.float32)
    def forced(*args):
        result = geometry_reward_components(*args)
        return replace(result, reward=torch.tensor([[[-3., .5, -3.]]], dtype=torch.float32))
    monkeypatch.setattr(physics, "geometry_reward_components", forced)
    score, journal = physics.PhysicsGeometryCEM(native.model).score_bank(q, v, target, bank, step=0)
    assert score.dtype == np.float32 and score[0, 0] == -5.
    assert journal["banks"][0]["metadata"]["sum_offsets_completed"] == 3


def test_native_failure_prefix_exact_returned_substeps_and_terminal_state(native, monkeypatch):
    q, v, target = roots(native)
    planner = physics.PhysicsGeometryCEM(native.model)
    actual = mujoco.mj_step
    calls = 0
    failure = RuntimeError("synthetic native interruption")
    def fail(model, data):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise failure
        actual(model, data)
    monkeypatch.setattr(physics.mujoco, "mj_step", fail)
    with pytest.raises(RuntimeError) as seen:
        planner.score_bank(q, v, target, commands(h=3), step=0)
    assert seen.value is failure
    snapshot = planner.snapshot()
    record = snapshot["last_operation"]["banks"][0]
    meta, arrays = record["metadata"], record["arrays"]
    assert meta["native_substeps_attempted"] == 4 and meta["native_substeps_completed"] == 3
    assert meta["native_transitions_attempted"] == 2 and meta["native_transitions_completed"] == 1
    assert arrays["native_completed"].sum() == 1 and not arrays["scored"].any()
    np.testing.assert_array_equal(arrays["native_substeps_completed"][0, 0], [2, 1, 0])
    assert meta["cursor"] == {"case": 0, "candidate": 0, "offset": 1}
    assert record["failure_native_state"]["time"] == pytest.approx(.03)
    assert snapshot["failed"] and snapshot["lifetime"]["operations_failed"] == 1
    with pytest.raises(RuntimeError, match="terminal"):
        planner.plan(q, v, target, draws(), step=0)


def test_geometry_failure_retains_completed_native_prefix(native, monkeypatch):
    q, v, target = roots(native)
    planner = physics.PhysicsGeometryCEM(native.model)
    def reject(*args):
        raise ValueError("synthetic geometry failure")
    monkeypatch.setattr(physics, "geometry_reward_components", reject)
    with pytest.raises(ValueError, match="synthetic geometry"):
        planner.score_bank(q, v, target, commands(k=2, h=2), step=0)
    record = planner.snapshot()["last_operation"]["banks"][0]
    assert record["metadata"]["native_transitions_completed"] == 4
    assert record["metadata"]["geometry_samples_attempted"] == 4
    assert record["metadata"]["geometry_samples_completed"] == 0
    assert record["arrays"]["native_completed"].all() and not record["arrays"]["scored"].any()


@pytest.mark.parametrize("failed_call", [2, 5])
def test_failed_cem_or_selected_stage_preserves_all_prior_paid_work(native, monkeypatch, failed_call):
    q, v, target = roots(native)
    inputs = draws()
    planner = physics.PhysicsGeometryCEM(native.model)
    calls = 0
    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == failed_call:
            raise ValueError("synthetic stage interruption")
        return geometry_reward_components(*args)
    monkeypatch.setattr(physics, "geometry_reward_components", interrupted)
    with pytest.raises(ValueError, match="stage interruption"):
        planner.plan(q, v, target, inputs, step=49)
    journal = planner.snapshot()["last_operation"]
    assert journal["search"]["input_identities"] == list(inputs.identities())
    assert [row["metadata"]["candidate_start"] for row in journal["banks"]] == list(range(0, min(failed_call, 4) * 64, 64))
    assert all(row["metadata"]["status"] == "completed" for row in journal["banks"][:failed_call - 1])
    if failed_call == 2:
        assert len(journal["banks"]) == 2 and journal["search"]["status"] == "failed"
        assert journal["selected"] is None
        assert planner.snapshot()["lifetime"]["native_transitions_completed"] == 128
        assert planner.snapshot()["lifetime"]["geometry_samples_completed"] == 64
    else:
        assert len(journal["banks"]) == 4 and journal["search"]["status"] == "completed"
        assert journal["search"]["candidate_evaluations"] == 256
        assert journal["selected"]["metadata"]["status"] == "failed"
        assert planner.snapshot()["lifetime"]["native_transitions_completed"] == 257
        assert planner.snapshot()["lifetime"]["geometry_samples_completed"] == 256
        np.testing.assert_array_equal(journal["selected"]["arrays"]["qpos"][0, 0, 0], q[0])


def test_initial_deadline_failure_is_retained_without_native_work(native):
    q, v, target = roots(native)
    planner = physics.PhysicsGeometryCEM(native.model)
    with pytest.raises(TimeoutError):
        planner.score_bank(q, v, target, commands(), step=0, deadline=-1.)
    snapshot = planner.snapshot()
    assert snapshot["last_operation"]["banks"] == [] and snapshot["failed"]
    assert snapshot["lifetime"]["native_transitions_attempted"] == 0
    assert snapshot["last_operation"]["wall_seconds"] > 0


@pytest.mark.parametrize("mutation", ["qpos_dtype", "qvel_shape", "target_dtype", "target_value", "moving_target",
                                       "nan", "bank_dtype", "command_range", "horizon", "terminal", "step_bool"])
def test_malformed_inputs_rejected_before_native_work(native, mutation):
    q, v, target = roots(native)
    bank, step = commands(), 0
    if mutation == "qpos_dtype": q = q.astype(np.float32)
    elif mutation == "qvel_shape": v = v[0]
    elif mutation == "target_dtype": target = target.astype(np.float64)
    elif mutation == "target_value": target[0, 0] += .1
    elif mutation == "moving_target": v[0, 2] = .01
    elif mutation == "nan": q[0, 0] = np.nan
    elif mutation == "bank_dtype": bank = bank.astype(np.float64)
    elif mutation == "command_range": bank[0, 0, 0, 0] = 1.1
    elif mutation == "horizon": step = 49
    elif mutation == "terminal": step = 50
    else: step = True
    planner = physics.PhysicsGeometryCEM(native.model)
    with pytest.raises(ValueError):
        planner.score_bank(q, v, target, bank, step=step)
    assert planner.snapshot()["last_operation"] is None and not planner.snapshot()["failed"]


def test_no_live_environment_or_data_and_no_changed_geometry(native):
    for value in (native, native.data, object()):
        with pytest.raises(TypeError):
            physics.PhysicsGeometryCEM(value)
    changed = copy.copy(native.model)
    index = mujoco.mj_name2id(changed, mujoco.mjtObj.mjOBJ_BODY, "fingertip")
    changed.body_pos[index, 0] = .12
    with pytest.raises(ValueError, match="geometry"):
        physics.PhysicsGeometryCEM(changed)


def test_private_model_drift_fails_without_native_scoring(native):
    q, v, target = roots(native)
    planner = physics.PhysicsGeometryCEM(native.model)
    planner._model.opt.gravity[2] = -4.
    with pytest.raises(ValueError, match="Private model changed"):
        planner.score_bank(q, v, target, commands(), step=0)
    assert planner.snapshot()["failed"] and planner.snapshot()["lifetime"]["native_transitions_completed"] == 0
