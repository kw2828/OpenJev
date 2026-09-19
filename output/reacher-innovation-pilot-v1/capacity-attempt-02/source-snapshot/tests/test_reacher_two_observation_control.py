"""Synthetic public packets and isolated seed410 only; no native/model assets."""

import copy
import json
import time

import numpy as np
import pytest
import torch

from openjev.research import reacher_geometry_control as kernel
from openjev.research import reacher_two_observation_control as control
from openjev.research.reacher_adaptive_search import SearchInputs, search
from openjev.research.reacher_cache_training import REGISTRY
from openjev.research.reacher_geometry_reward import geometry_reward
from openjev.research.reacher_objective_training import canonical_tensor_hash
from openjev.research.reacher_reward_residual import GRUResidualRewardWorldModel
from openjev.research.reacher_two_observation_history import REAL_KEYS, TwoObservationHistoryGRUWorldModel


@pytest.fixture(autouse=True)
def isolated_seed410():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


def plan(horizon=2, block=1):
    return {"engineering": True, "steps": 50, "hidden_size": 4,
            "dt": .02, "noise_std": .05, "planning_horizon": horizon,
            "action_block": block, "score_modes": ["geometry"]}


def model():
    return TwoObservationHistoryGRUWorldModel(hidden_size=4).eval()


def packet(n=1, *, valid=True, age=0., angle=.1):
    return np.tile(np.array([np.cos(angle), np.cos(-angle), np.sin(angle), np.sin(-angle),
                            .05, -.03, float(valid), age], np.float32), (n, 1))


def draws(n=1, horizon=2, block=1):
    rng = np.random.default_rng(410)
    chunks = (horizon + block - 1) // block
    arrays = [rng.normal(size=(n, k, chunks, 2)) for k in (64, 192, 64, 64, 63)]
    return SearchInputs(arrays[0], arrays[1], tuple(arrays[2:]))


def root_at(student, index, n=1):
    with torch.no_grad():
        state = student.assimilate(student.initial(n), torch.from_numpy(packet(n)))
        for _ in range(index):
            state, _, _ = student.advance(state, torch.zeros(n, 2))
            state = student.assimilate(state, torch.from_numpy(packet(n)))
    return state


def tensor_state(arrays):
    return {key: torch.from_numpy(value.copy()) for key, value in arrays.items()}


def test_bank_matches_direct_rollout_geometry_sequential_sum_and_root_isolation():
    student, cfg = model(), plan()
    root = root_at(student, 0, 2)
    original = canonical_tensor_hash(root)
    bank = np.linspace(-.4, .4, 24, dtype=np.float32).reshape(2, 3, 2, 2)
    unchanged = bank.copy()
    scores, diagnostic = control.score_bank(cfg, student, root, bank)
    arrays = diagnostic["arrays"]
    with torch.no_grad():
        state = {key: value.repeat_interleave(3, 0) for key, value in root.items()}
        total = np.zeros((2, 3), np.float32)
        for t in range(2):
            action = torch.from_numpy(bank[:, :, t].reshape(6, 2).copy())
            state, angles, reward = student.advance(state, action)
            np.testing.assert_array_equal(arrays["predicted_angles"][:, :, t], angles.reshape(2, 3, 4))
            np.testing.assert_array_equal(arrays["learned_rewards"][:, :, t], reward.reshape(2, 3))
            geometry = geometry_reward(angles, root["packet"][:, 4:6].repeat_interleave(3, 0), action, .05)
            total += geometry.clamp(-2.5, 0).reshape(2, 3).numpy()
            for key in REAL_KEYS:
                torch.testing.assert_close(state[key], root[key].repeat_interleave(3, 0), rtol=0, atol=0)
    np.testing.assert_array_equal(scores, total)
    changed = bank.copy()
    changed[:, 0] *= -1
    other, _ = control.score_bank(cfg, student, root, changed)
    np.testing.assert_array_equal(other[:, 1:], scores[:, 1:])
    np.testing.assert_array_equal(bank, unchanged)
    assert canonical_tensor_hash(root) == original
    assert diagnostic["metadata"]["history_work"]["completed_advance_history_clone_tensor_bytes"] == 12 * 596


@pytest.mark.parametrize("step,horizon", [(0, 12), (47, 3), (49, 1)])
def test_full_cem_exact_frozen_kernel_parity_paid_mean_and_terminal(step, horizon):
    cfg, student = plan(12, 3), model()
    cfg["engineering"] = False
    root = root_at(student, step)
    innovation = draws(horizon=12, block=3)
    before_rng = torch.get_rng_state().clone()
    actual = control.score_search(cfg, student, root, innovation, step)

    def original_score(bank):
        record = kernel._journal(root, bank, "geometry")
        return kernel._score_bank(cfg, student, root, bank, "geometry", float("inf"), record)

    with torch.no_grad():
        expected = search("cem256", innovation, original_score, step=step, steps=50,
                          planning_horizon=12, action_block=3)
    for key in ("sequences", "scores", "selected_actions", "selected_ids", "selected_sequences"):
        np.testing.assert_array_equal(getattr(actual[0], key), getattr(expected, key))
    assert actual[0].horizon == horizon and actual[2] == [64] * 4
    assert actual[0].stages[-1].mean_candidate_id == 255
    assert actual[4]["imagined_transitions"] == actual[4]["geometry_samples"] == 256 * horizon
    assert actual[0].selected_ids.item() == np.argmax(actual[0].scores[0])
    assert torch.equal(before_rng, torch.get_rng_state())
    assert all(parameter.grad is None for parameter in student.parameters())
    assert actual[4]["neural_kernels"]["transition"]["completed_sample_calls"] == 256 * horizon


def test_decision_matches_search_from_reconstructed_root_counts_all_hooks_and_selected_advance():
    student, cfg, innovation = model(), plan(), draws()
    registry = dict(REGISTRY)
    adapter = control.TwoObservationController(cfg, student, 1)
    decision = adapter.decide(packet(), innovation)
    root, carried = tensor_state(decision.root_state), tensor_state(decision.carried_state)
    with torch.no_grad():
        expected_root = student.assimilate(student.initial(1), torch.from_numpy(packet()))
        expected_carried, angles, rewards = student.advance(expected_root, torch.from_numpy(decision.action.copy()))
    assert canonical_tensor_hash(root) == canonical_tensor_hash(expected_root)
    assert canonical_tensor_hash(carried) == canonical_tensor_hash(expected_carried)
    np.testing.assert_array_equal(decision.diagnostics["arrays"]["selected_angles"], angles)
    np.testing.assert_array_equal(decision.diagnostics["arrays"]["selected_learned_reward"], rewards)
    meta = decision.metadata
    reconstruction = meta["reconstruction_neural_kernels"]
    assert reconstruction["observation_update"]["completed_sample_calls"] == 12
    assert reconstruction["transition"]["completed_sample_calls"] == 11
    assert sum(value["completed_sample_calls"] for name, value in reconstruction.items() if "head." in name) == 44
    assert meta["reconstruction_analytic_reward"] == {"completed_samples_lower_bound": 11,
        "completed_samples_upper_bound": 11, "exact": True}
    op = meta["model_work"]["operations"]
    assert op["gru_cell_sample_calls"] == 12 + 11 + 512 + 1
    assert op["linear_layer_sample_calls"] == 4 * (11 + 512 + 1)
    assert op["analytic_reward_sample_calls"] == 11 + 512 + 1
    assert meta["controller_seconds"] >= sum(meta[k] for k in ("reconstruction_seconds", "search_seconds", "selected_advance_seconds"))
    assert meta["root_snapshot_tensor_bytes"] == 4 * 4 + 644
    assert REGISTRY == registry
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in student.modules())


def test_missing_poison_sanitization_and_full_window_reacquisition_use_actual_commands():
    cfg, student = plan(1), model()
    adapter = control.TwoObservationController(cfg, student, 1)
    innovation = draws(horizon=1)
    issued, actual = None, []
    for t in range(13):
        visible = t in (0, 1, 12)
        public = packet(valid=visible, age=0 if visible else (t - 1) * .02, angle=.1 + t / 100)
        if not visible:
            public[:, :4] = np.nan
        decision = adapter.decide(public, innovation, issued_action=issued)
        if t == 11:
            np.testing.assert_array_equal(decision.root_state["real_indices"], np.arange(12)[None])
            np.testing.assert_array_equal(decision.root_state["real_actions"][0], np.array(actual))
            assert (decision.root_state["real_packets"][0, 2:, :4] == 0).all()
        if t == 12:
            np.testing.assert_array_equal(decision.root_state["real_indices"], np.arange(1, 13)[None])
            np.testing.assert_array_equal(decision.root_state["real_actions"][0], np.array(actual[1:]))
            assert decision.root_state["real_present"].all()
        actual.append(decision.action[0].copy())
        issued = decision.action.copy()
    assert adapter.snapshot()["next_step"] == 13


def test_outputs_snapshots_and_plan_cannot_alias_controller_history():
    cfg, student = plan(1), model()
    adapter = control.TwoObservationController(cfg, student, 1)
    first = adapter.decide(packet(), draws(horizon=1))
    issued = first.action.copy()
    original = adapter.snapshot()
    for mapping in (first.root_state, first.carried_state, adapter.snapshot()["state"]):
        for value in mapping.values():
            value[...] = 0
    first.action[:] = .99
    first.diagnostics["arrays"]["selected_actions"][:] = .88
    cfg["planning_horizon"] = 99
    current = adapter.snapshot()
    for key in current["state"]:
        np.testing.assert_array_equal(current["state"][key], original["state"][key])
    second = adapter.decide(packet(angle=.2), draws(horizon=1), issued_action=issued)
    np.testing.assert_array_equal(second.root_state["real_actions"][:, -1], issued)


@pytest.mark.parametrize("bad", ["different", "missing", "dtype"])
def test_wrong_actual_issued_command_fails_before_history_change_and_no_retry(bad, tmp_path):
    adapter = control.TwoObservationController(plan(1), model(), 1)
    first = adapter.decide(packet(), draws(horizon=1))
    before = adapter.snapshot()
    action = first.action.copy()
    if bad == "different":
        action[0, 0] += .001
    elif bad == "missing":
        action = None
    else:
        action = action.astype(np.float64)
    with pytest.raises(ValueError, match="Actual issued"):
        adapter.decide(packet(), draws(horizon=1), issued_action=action, failure_out=tmp_path / "failed")
    snapshot = adapter.snapshot()
    assert snapshot["next_step"] == 1
    for key in before["state"]:
        np.testing.assert_array_equal(snapshot["state"][key], before["state"][key])
    assert snapshot["failure"]["reconstruction_attempted_samples"] == 0
    assert json.loads((tmp_path / "failed" / "failed.json").read_text())["phase"] == "validation"
    with pytest.raises(ValueError, match="no automatic retry"):
        adapter.decide(packet(), draws(horizon=1), issued_action=first.action)


def test_reconstruction_failure_retains_partial_kernel_work_removes_hooks(tmp_path):
    student = model()
    adapter = control.TwoObservationController(plan(), student, 1)
    calls = 0

    def fail(_module, _args):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("synthetic transition failure")

    handle = student.transition.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="synthetic transition"):
            adapter.decide(packet(), draws(), failure_out=tmp_path / "failed")
    finally:
        handle.remove()
    failure = adapter.snapshot()["failure"]
    assert failure["reconstruction_attempted_samples"] == 1
    assert failure["reconstruction_completed_samples"] == 0
    assert failure["reconstruction_neural_kernels"]["observation_update"]["completed_sample_calls"] == 4
    assert failure["reconstruction_neural_kernels"]["transition"]["completed_sample_calls"] == 3
    assert failure["reconstruction_analytic_reward"]["completed_samples_lower_bound"] == 3
    assert failure["reconstruction_seconds"] > 0
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in student.modules())


def test_search_failure_preserves_prefix_and_private_prior_state(tmp_path):
    student = model()
    adapter = control.TwoObservationController(plan(), student, 1)
    calls = 0

    def fail(_module, args):
        nonlocal calls
        if len(args[0]) == 64:
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic candidate failure")

    handle = student.transition.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="candidate failure"):
            adapter.decide(packet(), draws(), failure_out=tmp_path / "failed")
    finally:
        handle.remove()
    failure = adapter.snapshot()["failure"]
    assert adapter.snapshot()["next_step"] == 0
    assert failure["reconstruction_completed_samples"] == 1
    assert failure["search_failure"]["banks"][0]["model_advance_samples"] == 64
    assert failure["search_failure"]["banks"][0]["model_advance_attempted_samples"] == 128
    assert failure["search_seconds"] > 0
    with np.load(tmp_path / "failed" / "search" / "bank-000.npz") as saved:
        assert saved["predicted_angles"].shape == (1, 64, 1, 4)
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in student.modules())


@pytest.mark.parametrize("bad", ["class", "training", "double", "residual", "horizon", "mode"])
def test_exact_class_configuration_boundary(bad):
    student, cfg = model(), plan()
    if bad == "class":
        student = GRUResidualRewardWorldModel(hidden_size=4).eval()
    elif bad == "training":
        student.train()
    elif bad == "double":
        student.double()
    elif bad == "residual":
        student.residual_reward = False
    elif bad == "horizon":
        cfg["engineering"] = False
    elif bad == "mode":
        cfg["score_modes"] = ["learned"]
    with pytest.raises(ValueError):
        control.TwoObservationController(cfg, student, 1)


def test_clock_mismatch_terminal_and_expired_cap_make_no_forward_call():
    student = model()
    state = root_at(student, 49)
    before = canonical_tensor_hash(state)
    with pytest.raises(ValueError, match="step differs"):
        control.score_search(plan(), student, state, draws(), 48)
    with pytest.raises(ValueError, match="terminal"):
        control.score_bank(plan(), student, state, np.zeros((1, 2, 2, 2), np.float32))
    with pytest.raises(TimeoutError):
        control.score_search(plan(), student, state, draws(), 49, time.monotonic() - 1)
    terminal = root_at(student, 50)
    with pytest.raises(ValueError, match="terminal50"):
        control.score_search(plan(), student, terminal, draws(), 50)
    assert canonical_tensor_hash(state) == before


def test_failure_destination_is_not_overwritten(tmp_path):
    prior = tmp_path / "existing"
    prior.mkdir()
    (prior / "sentinel").write_text("keep")
    adapter = control.TwoObservationController(plan(), model(), 1)
    with pytest.raises(ValueError, match="exclusive"):
        adapter.decide(packet(), draws(), failure_out=prior)
    assert list(prior.iterdir()) == [prior / "sentinel"]
    assert (prior / "sentinel").read_text() == "keep"


def test_live_weight_drift_rejected_before_reconstruction():
    student = model()
    adapter = control.TwoObservationController(plan(), student, 1)
    with torch.no_grad():
        next(student.parameters()).add_(.01)
    with pytest.raises(ValueError, match="Live model"):
        adapter.decide(packet(), draws())
    assert adapter.snapshot()["failure"]["reconstruction_attempted_samples"] == 0


def test_identical_retained_suffix_discards_older_neural_state():
    student = model()
    state = root_at(student, 3)
    with torch.no_grad():
        carried, _, _ = student.advance(state, torch.tensor([[.1, -.1]]))
        poisoned = copy.deepcopy(carried)
        poisoned["hidden"][:] = float("nan")
        poisoned["packet"][:, :4] = float("nan")
        a = student.assimilate(carried, torch.from_numpy(packet(angle=.3)))
        b = student.assimilate(poisoned, torch.from_numpy(packet(angle=.3)))
    assert canonical_tensor_hash(a) == canonical_tensor_hash(b)
    left = control.score_search(plan(), student, a, draws(), 4)
    right = control.score_search(plan(), student, b, draws(), 4)
    np.testing.assert_array_equal(left[0].scores, right[0].scores)
    np.testing.assert_array_equal(left[0].selected_actions, right[0].selected_actions)


@pytest.mark.parametrize("target_x", [.21, 9.])
def test_fixed_toy_policy_analytic_score_and_global_earliest_tie(target_x):
    cfg, student = plan(12, 3), model()
    cfg["noise_std"] = student.noise_std = 0.
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.zero_()
        student.observation_head[2].bias.copy_(torch.tensor([1., 1., 0., 0.]))
    public = packet()
    public[:, 4:6] = np.array([target_x, 0], np.float32)
    adapter = control.TwoObservationController(cfg, student, 1)
    decision = adapter.decide(public, draws(horizon=12, block=3))
    expected = np.zeros((1, 256), np.float32)
    distance = np.float32(abs(np.float32(.1) + np.float32(.11) - np.float32(target_x)))
    for t in range(12):
        expected += np.clip(-distance - (decision.search_result.sequences[:, :, t] ** 2).sum(-1), -2.5, 0)
    np.testing.assert_allclose(decision.search_result.scores, expected, rtol=0, atol=2e-6)
    assert decision.search_result.selected_ids.tolist() == [0]
    np.testing.assert_array_equal(decision.action, np.zeros((1, 2), np.float32))
    if target_x == 9.:
        np.testing.assert_array_equal(decision.search_result.scores, np.full((1, 256), -30, np.float32))
