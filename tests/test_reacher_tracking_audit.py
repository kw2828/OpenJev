"""Tiny engineering410 evidence and corruption tests, no scientific data."""
import copy
import json
import time

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("gymnasium.envs.mujoco.reacher_v5")

from openjev.research import reacher_tracking_audit as audit
from openjev.research.reacher_adaptive_search import SearchInputs
from openjev.research.reacher_geometry_physics import PhysicsGeometryCEM
from openjev.research.reacher_tracking_dynamics import TrackingDynamicsEpisode, nominal_model
from openjev.research.reacher_tracking_identification import TrackingAngleObserver, WindowedGainIdentifier


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


@pytest.fixture(scope="module")
def model():
    return nominal_model()


@pytest.fixture(scope="module")
def evidence(model):
    rng = np.random.default_rng(410)
    inputs = SearchInputs(rng.normal(size=(1, 64, 2, 2)), rng.normal(size=(1, 192, 2, 2)),
                          tuple(rng.normal(size=(1, k, 2, 2)) for k in (64, 64, 63)))
    native = copy.copy(model); native.actuator_gear[:, 0] *= .7
    qpos = np.array([[.1, -.2, .12, -.04]]); qvel = np.array([[.02, -.01, 0., 0.]])
    target = qpos[:, 2:].astype(np.float32)
    planner = PhysicsGeometryCEM(native, steps=3, planning_horizon=3, action_block=2, noise_std=.05)
    result, journal = planner.plan(qpos, qvel, target, inputs, step=1)
    selected = journal["selected"]["arrays"]
    values = {"root_qpos": qpos[0], "root_qvel": qvel[0],
              "public_packet": np.r_[np.cos(qpos[0, :2]), np.sin(qpos[0, :2]), target[0], 1., 0.].astype(np.float32),
              "planning_gain": np.asarray(.7), "action": result.selected_actions[0],
              "sequences": result.sequences[0], "scores": result.scores[0], "selected_id": result.selected_ids[0],
              "predicted_angles": np.concatenate([b["arrays"]["predicted_angles"][0] for b in journal["banks"]]),
              "raw_rewards": np.concatenate([b["arrays"]["geometry_reward"][0] for b in journal["banks"]]),
              "selected_qpos": selected["qpos"][0, 0], "selected_qvel": selected["qvel"][0, 0],
              "selected_angles": selected["predicted_angles"][0, 0, 0],
              "selected_reward": selected["geometry_reward"][0, 0, 0]}
    values = {k: np.asarray(v).copy() for k, v in values.items()}
    mapping = dict(zip(audit.INPUT_NAMES, (inputs.initial, inputs.random_extra, *inputs.cem), strict=True))
    return values, mapping, journal


def test_actual_saved_search_near_terminal_and_native_replay(model, evidence):
    values, inputs, _ = evidence
    info = audit.verify_search(values, inputs, step=1, steps=3, horizon=3, block=2)
    assert info["horizon"] == 2
    result = audit.replay_candidates(model, values, step=1, noise_std=.05)
    assert result["candidate_transitions"] == 512
    assert result["selected_transitions"] == 1
    assert result["max_native_abs_error"] == 0
    assert result["max_geometry_abs_error"] < 5e-7


@pytest.mark.parametrize("key", ["sequences", "scores", "raw_rewards", "action", "selected_id"])
def test_corrupted_search_is_rejected(evidence, key):
    values, inputs, _ = evidence
    values = {k: v.copy() for k, v in values.items()}
    if key == "selected_id":
        values[key][...] = (int(values[key])+1) % 256
    else:
        values[key].flat[-1] += .1
    with pytest.raises(ValueError):
        audit.verify_search(values, inputs, step=1, steps=3, horizon=3, block=2)


def test_paid_mean_and_tie_use_scored_float32_chunks():
    inputs = {key: np.zeros((1, k, 2, 2), np.float64)
              for key, k in zip(audit.INPUT_NAMES, (64, 192, 64, 64, 63), strict=True)}
    anchors = ((0., 0.), (.1, 0.), (-.1, 0.), (0., .1), (0., -.1), (.2, .2), (-.2, -.2))
    chunks = np.zeros((64, 2, 2), np.float32)
    for i, a in enumerate(anchors):
        chunks[i] = a
    banks = [np.repeat(chunks, 2, 1)[:, :3]]
    for _ in range(3):
        chunks = np.broadcast_to(chunks[:8].astype(np.float64).mean(0), (64, 2, 2)).astype(np.float32)
        banks.append(np.repeat(chunks, 2, 1)[:, :3])
    values = {"sequences": np.concatenate(banks), "scores": np.full(256, -7.5),
              "raw_rewards": np.full((256, 3), -10., np.float32), "selected_id": np.asarray(0, np.int64),
              "action": np.zeros(2, np.float32)}
    assert audit.verify_search(values, inputs, step=0, steps=3, horizon=3, block=2)["selected_id"] == 0


@pytest.mark.parametrize("key", ["predicted_angles", "planning_gain", "selected_qpos", "selected_angles", "selected_reward"])
def test_native_candidate_selected_or_gain_tamper_rejected(model, evidence, key):
    values, _, _ = evidence
    values = {k: v.copy() for k, v in values.items()}; values[key].flat[-1] += .01
    with pytest.raises(ValueError):
        audit.replay_candidates(model, values, step=1, noise_std=.05)


def test_replay_deadline_precedes_candidate_native_work(model, evidence):
    with pytest.raises(TimeoutError):
        audit.replay_candidates(model, evidence[0], step=1, noise_std=.05, deadline=time.perf_counter()-1)


def test_analytic_cost_charged_once_and_target_changes_geometry():
    angles = np.array([1., 1., 0., 0.], np.float32)
    command = np.array([.5, -.25], np.float32)
    target = np.array([.21, 0.], np.float32)
    assert abs(float(audit.geometry_reward(angles, target, command, 0.)) + .3125) < 1e-7
    assert audit.geometry_reward(angles, target, command, .05) < -.3125


@pytest.fixture(scope="module")
def episode():
    env = TrackingDynamicsEpisode(horizon=4, target_path=np.array([[.12, -.04], [.12, -.04],
                    [-.06, .10], [-.06, .10], [.08, .09]]), gear_multiplier=np.array([.7, .7, 1.3, 1.3]),
                    sensor_schedule=np.ones(5, bool), noise=np.full((4, 2), .0001))
    try:
        env.reset(410)
        for command in ((.015, 0), (0, -.015), (-.015, 0), (0, .015)):
            env.step(command)
        return json.loads(json.dumps(json_safe(env.episode_record())))
    finally:
        env.close()


def test_native_rewards_before_target_event_and_hidden_gain(model, episode):
    report = audit.audit_episode(episode, model)
    assert report["transitions"] == 4
    assert report["max_abs_error"] == 0
    assert report["mean_cost"] > 0


@pytest.mark.parametrize("kind", ["target", "gain", "reward", "next_state", "command", "terminal"])
def test_native_timeline_corruption_rejected(model, episode, kind):
    value = copy.deepcopy(episode); transition = value["audit"]["transitions"][1]
    if kind == "target":
        transition["reward_target"] = transition["next_target"]
    elif kind == "gain":
        value["metadata"]["gear_multiplier"][1] = 1.3
        transition["gear_multiplier"], transition["actuator_gear"] = 1.3, 260.
    elif kind == "reward":
        transition["reward_ctrl"] *= .7**2
    elif kind == "next_state":
        value["audit"]["decision_states"][2]["qvel"][0] += .01
    elif kind == "command":
        transition["command"][0] += .01
    else:
        transition["truncated"] = True
    with pytest.raises(ValueError):
        audit.audit_episode(value, model)


def test_public_derivative_wrap_and_new_target():
    packets = np.array([np.r_[np.cos([q, -.1]), np.sin([q, -.1]), goal, 1., 0.]
                        for q, goal in ((3.12, [.1, .1]), (3.15, [.1, .1]), (3.19, [-.1, .1]))], np.float32)
    observer = TrackingAngleObserver(packets[0], dt=.02)
    expected = [observer.estimate()]
    expected += [observer.update(p) for p in packets[1:]]
    q, v = audit.public_roots(packets)
    equal_q, equal_v = np.array([x[0] for x in expected]), np.array([x[1] for x in expected])
    np.testing.assert_array_equal(q, equal_q); np.testing.assert_array_equal(v, equal_v)


def test_identifier_old_root_rolling_window_and_no_future_velocity(model, episode):
    packets = np.asarray(episode["policy"]["packets"], np.float32)
    commands = np.asarray(episode["policy"]["commands"], np.float32)
    gains = np.array([.5, 1., 1.5]); window = 2
    identifier = WindowedGainIdentifier(model, packets[0], gains=gains, window=window, prior_gain=1., frame_skip=2)
    roots, velocity = audit.public_roots(packets); retained = []; prior = 1.
    for t in range(4):
        identifier.update(commands[t], packets[t+1])
        trace = json_safe(identifier.snapshot()["last_trace"])
        prior, retained, result = audit.replay_identification(model, trace, root_qpos=roots[t],
            root_qvel=velocity[t], command=commands[t], next_packet=packets[t+1], gains=gains,
            retained=retained, prior_gain=prior, window=window, step=t)
        assert result["max_abs_error"] == 0 and result["transitions"] == 3
        assert prior == identifier.estimate()[2]
        assert len(retained) == min(t+1, window)
    trace["root_qvel"] = velocity[-1].tolist()
    with pytest.raises(ValueError, match="root_qvel"):
        audit.replay_identification(model, trace, root_qpos=roots[3], root_qvel=velocity[3], command=commands[3],
            next_packet=packets[4], gains=gains, retained=retained, prior_gain=prior, window=window, step=3)


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    from openjev.research.reacher_search_protocol import save_inputs
    from openjev.research.reacher_tracking_rollout import run_row
    folder = tmp_path_factory.mktemp("tracking-audit-engineering410")
    rng = np.random.default_rng(410); stems = []
    for t in range(3):
        inputs = SearchInputs(rng.normal(size=(1, 64, 2, 2)), rng.normal(size=(1, 192, 2, 2)),
                              tuple(rng.normal(size=(1, k, 2, 2)) for k in (64, 64, 63)))
        stem = folder/"inputs"/f"{t:03d}"
        save_inputs(stem, inputs, f"engineering410/{t}"); stems.append(stem)
    for arm in audit.ARMS:
        run_row(arm=arm, target_path=np.array([[.12, -.04], [.12, -.04], [-.06, .10], [-.06, .10]]),
                gain_schedule=np.array([.5, 1.5, 1.5]), noise=np.full((3, 2), .0001), reset_seed=410,
                inputs_by_step=stems, gain_grid=np.array([.5, 1., 1.5]), window=2, freeze_after=1,
                noise_std=.05, planning_horizon=3, action_block=2, out=folder/arm,
                deadline=time.monotonic()+120, engineering=True)
    return folder, stems


@pytest.mark.parametrize("arm", audit.ARMS)
def test_complete_real_writer_row_against_independent_audit(rows, model, arm):
    folder, inputs = rows
    report = audit.audit_row(folder/arm, nominal_model=model, inputs_by_step=inputs)
    assert report["status"] == "completed"
    assert report["native_control"]["transitions"] == 3
    assert report["new_model_calls"] == report["new_planner_calls"] == report["new_rng_draws"] == 0
    assert report["candidate_transitions_checked"] == (0 if arm == "zero" else 256*6)
    assert report["selected_transitions_checked"] == (0 if arm == "zero" else 3)
    assert report["identifier_transitions_checked"] == {"adaptive": 9, "frozen": 3}.get(arm, 0)


def copy_row(rows, tmp_path, arm):
    import shutil
    source, stems = rows
    out = tmp_path/arm; shutil.copytree(source/arm, out)
    return out, stems


def reseal(folder):
    path = folder/"completed.json"; completed = json.loads(path.read_text())
    completed["files"] = {name: audit.file_hash(folder/name) for name in completed["files"]}
    path.write_text(json.dumps(completed))


def test_resealed_inappropriate_oracle_gain_is_rejected(rows, tmp_path, model):
    folder, inputs = copy_row(rows, tmp_path, "nominal")
    path = folder/"decisions/000.npz"
    with np.load(path, allow_pickle=False) as saved:
        values = dict(saved)
    values["planning_gain"] = np.array(.5); np.savez_compressed(path, **values); reseal(folder)
    with pytest.raises(ValueError, match="causal allowed"):
        audit.audit_row(folder, nominal_model=model, inputs_by_step=inputs)


def test_resealed_frozen_postcutoff_identifier_is_rejected(rows, tmp_path, model):
    folder, inputs = copy_row(rows, tmp_path, "frozen")
    path = folder/"observations/002.json"; value = json.loads(path.read_text())
    value["identifier_updated"] = True; path.write_text(json.dumps(value)); reseal(folder)
    with pytest.raises(ValueError, match="frozen update boundary"):
        audit.audit_row(folder, nominal_model=model, inputs_by_step=inputs)


def test_exact_member_boundary_rejects_failure_and_extras(rows, tmp_path, model):
    folder, inputs = copy_row(rows, tmp_path, "zero")
    (folder/"failed.json").write_text('{}')
    with pytest.raises(ValueError, match="membership"):
        audit.audit_row(folder, nominal_model=model, inputs_by_step=inputs)


def test_all_zero_input_binding_still_verified(rows, tmp_path, model):
    folder, inputs = copy_row(rows, tmp_path, "zero")
    path = folder/"started.json"; value = json.loads(path.read_text())
    value["inputs_by_step"][0]["npz_sha256"] = '0'*64
    path.write_text(json.dumps(value)); reseal(folder)
    with pytest.raises(ValueError, match="external innovation bytes"):
        audit.audit_row(folder, nominal_model=model, inputs_by_step=inputs)


def test_no_planner_search_identifier_or_torch_import_in_auditor():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(audit.__file__).read_text())
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    modules += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(module and module.startswith(("openjev", "torch")) for module in modules)


@pytest.mark.parametrize("kind", ["copies", "ring", "startup", "cutoff"])
def test_resealed_identifier_setup_or_state_tamper_rejected(rows, tmp_path, model, kind):
    folder, inputs = copy_row(rows, tmp_path, "frozen")
    path = folder/("initial-controller.json" if kind == "startup" else "controller.json")
    value = json.loads(path.read_text()); identifier = value["identifier"]
    if kind == "copies":
        identifier["costs"]["native_model_copies"] = 0
    elif kind == "ring":
        identifier["costs"]["retained_residual_scalars"] += 1
    elif kind == "startup":
        identifier["costs"]["native_transitions_completed"] = 1
    else:
        identifier["observer"]["qpos_estimate"][0] += .01
    path.write_text(json.dumps(value)); reseal(folder)
    with pytest.raises(ValueError):
        audit.audit_row(folder, nominal_model=model, inputs_by_step=inputs)
