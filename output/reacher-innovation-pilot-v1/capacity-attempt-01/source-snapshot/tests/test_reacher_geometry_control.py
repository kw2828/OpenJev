"""Tiny fresh engineering fixtures only; no inherited models or scored seeds."""

import json

import numpy as np
import pytest
import torch

from openjev.research import reacher_cache_control as original
from openjev.research import reacher_geometry_control as control
from openjev.research.reacher_adaptive_search import SearchInputs
from openjev.research.reacher_cache_training import REGISTRY
from openjev.research.reacher_geometry_reward import geometry_reward
from openjev.research.reacher_objective_training import canonical_tensor_hash
from openjev.research.robotics_reacher import native_replay


@pytest.fixture(autouse=True)
def isolate_rng_and_threads():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


def plan():
    return {"rng_namespace": "reacher-geometry-control-engineering-unit-v1", "engineering": True,
        "steps": 50, "control_episodes": 1, "noise_std": .05, "dt": .02,
        "planning_horizon": 2, "action_block": 1, "hidden_size": 4, "mlp_width": 7}


def model(kind):
    torch.manual_seed(410)
    size = {"width": 7} if kind == "cached_mlp" else {"hidden_size": 4}
    return REGISTRY[kind](**size, noise_std=.05).eval()


def root(student, n=1):
    packet = torch.tensor([[1., 1., 0., 0., .05, -.03, 1., 0.]]).repeat(n, 1)
    with torch.no_grad():
        return student.assimilate(student.initial(n), packet)


def innovations(horizon=2, block=1, n=1):
    chunks = (horizon + block - 1) // block
    rng = np.random.default_rng(410)
    arrays = [rng.normal(size=(n, k, chunks, 2)) for k in (64, 192, 64, 64, 63)]
    return SearchInputs(arrays[0], arrays[1], tuple(arrays[2:]))


def cases(panel="ordinary"):
    schedule = np.ones(51, dtype=bool)
    if panel != "full":
        gap = 6 if panel == "ordinary" else 10
        schedule[8:8 + gap] = False
        schedule[28:28 + gap] = False
    return [control.ControlCase(410, 410, schedule)]


def records(stem):
    metadata = json.loads(stem.with_suffix(".json").read_text())
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as values:
        result = []
        for index, meta in enumerate(metadata):
            record = {"metadata": meta, "policy": {}, "audit": {}}
            for key in values.files:
                category, name = key.split("__", 1)
                record[category][name] = values[key][index].copy()
            result.append(record)
    return result


@pytest.mark.parametrize("kind", control.KINDS)
def test_same_bank_has_identical_transitions_and_original_rewards_in_both_modes(kind):
    cfg, student = plan(), model(kind)
    state = root(student, 2)
    bank = np.linspace(-.3, .3, 2 * 5 * 2 * 2, dtype=np.float32).reshape(2, 5, 2, 2)
    original_bank = bank.copy()
    initial = canonical_tensor_hash(state)
    learned_scores, learned = control.score_bank(cfg, student, state, bank, "learned")
    geometry_scores, geometry = control.score_bank(cfg, student, state, bank, "geometry")
    for key in ("predicted_angles", "learned_rewards", "commands", "root_target"):
        np.testing.assert_array_equal(learned["arrays"][key], geometry["arrays"][key])
    assert learned["metadata"]["terminal_state_sha256"] == geometry["metadata"]["terminal_state_sha256"]
    assert learned["metadata"]["prefix_state_sha256"] == geometry["metadata"]["prefix_state_sha256"]
    assert len(learned["metadata"]["prefix_state_sha256"]) == bank.shape[2]
    assert canonical_tensor_hash(state) == initial
    np.testing.assert_array_equal(bank, original_bank)
    assert learned["metadata"]["model"] == original.model_identity(student, cfg)
    assert learned["metadata"]["geometry_samples"] == learned["metadata"]["geometry_seconds"] == 0
    assert not any(key.startswith("geometry_") for key in learned["arrays"])
    assert geometry["metadata"]["geometry_samples"] == 20
    for scores, diagnostic in ((learned_scores, learned), (geometry_scores, geometry)):
        expected = np.zeros((2, 5), np.float32)
        for offset in range(2):
            expected += np.clip(diagnostic["arrays"]["selected_rewards"][:, :, offset], -2.5, 0)
        np.testing.assert_array_equal(scores, expected)
    angles = torch.from_numpy(geometry["arrays"]["predicted_angles"])
    target = torch.from_numpy(np.broadcast_to(state["packet"][:, None, None, 4:6].numpy(), (2, 5, 2, 2)).copy())
    expected = geometry_reward(angles, target, torch.from_numpy(bank.copy()), .05).numpy()
    np.testing.assert_array_equal(expected, geometry["arrays"]["selected_rewards"])
    assert all(parameter.grad is None for parameter in student.parameters())


@pytest.mark.parametrize("kind", control.KINDS)
def test_learned_mode_exact_original_cem_parity_and_no_geometry_calls(kind, monkeypatch):
    cfg, student, draws = plan(), model(kind), innovations()
    state = root(student)
    expected, raw, sizes, _, old_work = original.score_search(cfg, student, state, draws, 0)
    monkeypatch.setattr(control, "geometry_reward_components",
                        lambda *_: (_ for _ in ()).throw(AssertionError("Geometry leaked into baseline")))
    actual, changed_raw, changed_sizes, _, work, diagnostic = control.score_search(
        cfg, student, state, draws, 0, score_mode="learned")
    np.testing.assert_array_equal(actual.sequences, expected.sequences)
    np.testing.assert_array_equal(actual.scores, expected.scores)
    np.testing.assert_array_equal(actual.selected_actions, expected.selected_actions)
    np.testing.assert_array_equal(changed_raw, raw)
    assert changed_sizes == sizes == [64] * 4
    assert work["model_work"] == old_work["model_work"]
    assert work["geometry_samples"] == 0
    assert diagnostic["metadata"]["model"] == original.model_identity(student, cfg)


@pytest.mark.parametrize("kind", control.KINDS)
def test_paired_cem_shared_initial_bank_and_terminal_truncation(kind):
    cfg, student = plan(), model(kind)
    cfg.update(engineering=False, planning_horizon=12, action_block=3)
    state, draws = root(student), innovations(12, 3)
    before = canonical_tensor_hash(state)
    a = control.score_search(cfg, student, state, draws, 47, score_mode="learned")
    b = control.score_search(cfg, student, state, draws, 47, score_mode="geometry")
    np.testing.assert_array_equal(a[0].sequences[:, :64], b[0].sequences[:, :64])
    np.testing.assert_array_equal(a[5]["arrays"]["predicted_angles"][:, :64],
                                  b[5]["arrays"]["predicted_angles"][:, :64])
    assert a[0].horizon == b[0].horizon == 3
    assert a[4]["imagined_transitions"] == b[4]["geometry_samples"] == 256 * 3
    assert canonical_tensor_hash(state) == before
    assert [row["candidate_start"] for row in b[5]["metadata"]["callbacks"]] == [0, 64, 128, 192]


@pytest.mark.parametrize("kind", control.KINDS)
@pytest.mark.parametrize("mode", control.SCORE_MODES)
def test_native_row_retains_original_state_schema_selected_action_alignment_and_replay(kind, mode, tmp_path):
    cfg, student, draws = plan(), model(kind), innovations()
    initial = canonical_tensor_hash(student.state_dict())
    out = tmp_path / "control"
    timing = control.learned_control(cfg, student, "ordinary", [draws] * 50, out,
                                     cases=cases(), score_mode=mode)
    record = records(out / "episodes")[0]
    assert native_replay(record)["transitions"] == 50
    assert canonical_tensor_hash(student.state_dict()) == initial
    assert timing["observation_assimilations"] == timing["executed_action_advances"] == 50
    assert timing["row_wall_seconds"] >= timing["setup_seconds"] + sum(timing["decision_seconds"]) + sum(timing["native_step_seconds"])
    meta = json.loads((out / "state-work.json").read_text())
    assert set(meta) == {"model", "steps", "state_arrays_bytes", "max_single_candidate_state_tensor_bytes",
                         "tensor_bytes_are_not_peak_process_memory", "aggregate_model_work", "auxiliary_teacher_calls", "reset_calls"}
    with np.load(out / "states.npz", allow_pickle=False) as states, np.load(out / "executed_predictions.npz", allow_pickle=False) as executed:
        np.testing.assert_array_equal(states["root__packet"][0], record["policy"]["packets"][:-1])
        np.testing.assert_array_equal(states["carried__packet"][..., :4], executed["angles"])
        for step in (0, 10, 49):
            with np.load(out / "scoring" / f"{step:03d}.npz", allow_pickle=False) as scores, np.load(out / "decisions" / f"{step:03d}.npz", allow_pickle=False) as trace:
                np.testing.assert_array_equal(scores["selected_rewards"], trace["raw_rewards"])
                np.testing.assert_array_equal(scores["selected_learned_reward"], executed["rewards"][:, step])
                np.testing.assert_array_equal(scores["selected_angles"], executed["angles"][:, step])
                selected = trace["selected_ids"]
                np.testing.assert_array_equal(scores["commands"][np.arange(1), selected, 0], record["policy"]["commands"][None, step])
                saved_root = {k.removeprefix("root__"): torch.from_numpy(states[k][:, step].copy())
                              for k in states.files if k.startswith("root__")}
                with torch.no_grad():
                    carried, angles, reward = student.advance(saved_root, torch.from_numpy(record["policy"]["commands"][None, step].copy()))
                for key, value in carried.items():
                    np.testing.assert_array_equal(value.numpy(), states["carried__" + key][:, step])
                np.testing.assert_array_equal(angles.numpy(), executed["angles"][:, step])
                np.testing.assert_array_equal(reward.numpy(), executed["rewards"][:, step])
            scoring = json.loads((out / "scoring" / f"{step:03d}.json").read_text())
            assert scoring["selected_advance"]["geometry_samples"] == (mode == "geometry")
            assert scoring["selected_advance"]["root_sha256"] == meta["steps"][step]["root_sha256"]
        if kind == "cached_mlp":
            assert (states["root__imagined_depth"] == 0).all()
            assert (states["carried__imagined_depth"] == 1).all()
            np.testing.assert_array_equal(states["root__cached_angles"], states["carried__cached_angles"])


@pytest.mark.parametrize("mode", control.SCORE_MODES)
def test_partial_bank_preserves_original_interrupt_and_completed_prefix(mode, tmp_path, monkeypatch):
    cfg, student, state = plan(), model("residual_gru"), None
    state = root(student)
    bank = np.zeros((1, 3, 2, 2), np.float32)
    calls = 0
    original_cap = control.check_cap
    sentinel = KeyboardInterrupt("synthetic stopped scoring")

    def interrupt(deadline):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise sentinel
        original_cap(deadline)

    monkeypatch.setattr(control, "check_cap", interrupt)
    destination = tmp_path / "partial"
    with pytest.raises(KeyboardInterrupt) as actual:
        control.score_bank(cfg, student, state, bank, mode, failure_out=destination)
    assert actual.value is sentinel
    receipt = json.loads((destination / "failed.json").read_text())
    assert receipt["banks"][0]["completed_scored_offsets"] == 1
    with np.load(destination / "bank-000.npz", allow_pickle=False) as arrays:
        assert arrays["predicted_angles"].shape == (1, 3, 1, 4)
        np.testing.assert_array_equal(arrays["commands"], bank)
        np.testing.assert_array_equal(arrays["root__packet"], state["packet"].numpy())


def test_invalid_geometry_preserves_predictions_and_never_falls_back(tmp_path):
    cfg, student = plan(), model("residual_gru")
    with torch.no_grad():
        student.observation_head[-1].weight.zero_()
        student.observation_head[-1].bias.zero_()
    state, bank = root(student), np.zeros((1, 2, 2, 2), np.float32)
    control.score_bank(cfg, student, state, bank, "learned")
    with pytest.raises(ValueError, match="norms"):
        control.score_bank(cfg, student, state, bank, "geometry", failure_out=tmp_path / "failed")
    receipt = json.loads((tmp_path / "failed" / "failed.json").read_text())
    row = receipt["banks"][0]
    assert row["completed_model_offsets"] == 1 and row["completed_scored_offsets"] == 0
    assert row["geometry_attempted_samples"] == 2 and row["geometry_samples"] == 0
    with np.load(tmp_path / "failed" / "bank-000.npz", allow_pickle=False) as arrays:
        assert arrays["predicted_angles"].shape == (1, 2, 1, 4)
        assert "selected_rewards" not in arrays.files


def test_selected_scoring_failure_preserves_search_root_and_carried_state(tmp_path, monkeypatch):
    cfg, student, draws = plan(), model("cached_mlp"), innovations()
    original_geometry = control.geometry_reward_components
    calls = 0
    sentinel = RuntimeError("synthetic selected failure")

    def fail_selected(*args):
        nonlocal calls
        calls += 1
        if calls == 9:  # Four search callbacks of two transitions, then selected advance.
            raise sentinel
        return original_geometry(*args)

    monkeypatch.setattr(control, "geometry_reward_components", fail_selected)
    out = tmp_path / "failed-row"
    with pytest.raises(RuntimeError) as actual:
        control.learned_control(cfg, student, "ordinary", [draws] * 50, out, cases=cases(), score_mode="geometry")
    assert actual.value is sentinel
    assert (out / "partial-root.npz").exists() and (out / "partial-carried.npz").exists()
    receipt = json.loads((out / "failed.json").read_text())
    assert receipt["completed_steps_by_case"] == [0]
    meta = json.loads((out / "partial-active-scoring.json").read_text())
    assert meta["selected_advance"]["geometry_attempted_samples"] == 1
    assert meta["selected_advance"]["geometry_samples"] == 0
    with np.load(out / "partial-active-scoring.npz", allow_pickle=False) as arrays:
        assert arrays["predicted_angles"].shape == (1, 256, 2, 4)
        assert arrays["selected_angles"].shape == (1, 4)


@pytest.mark.parametrize("invalid", ["mode", "model", "horizon", "bank_dtype", "action", "shape"])
def test_configuration_and_candidate_validation(invalid):
    cfg, student, mode = plan(), model("residual_gru"), "learned"
    bank = np.zeros((1, 2, 2, 2), np.float32)
    if invalid == "mode":
        mode = "hybrid"
    elif invalid == "model":
        student = REGISTRY["encoded_current_gru"](hidden_size=4).eval()
    elif invalid == "horizon":
        cfg["engineering"] = False
    elif invalid == "bank_dtype":
        bank = bank.astype(np.float64)
    elif invalid == "action":
        bank[0, 0, 0, 0] = 1.01
    else:
        bank = np.zeros((2, 2, 2, 2), np.float32)
    with pytest.raises(ValueError):
        control.score_bank(cfg, student, root(student), bank, mode)
