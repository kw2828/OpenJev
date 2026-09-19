"""Engineering-only native fixtures and synthetic seed410, never study cases."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reacher_world_model_study as legacy

from openjev.research import reacher_cache_control as control
from openjev.research import reacher_cache_training as training
from openjev.research import reacher_search_protocol as artifacts
from openjev.research.reacher_adaptive_search import SearchInputs
from openjev.research.robotics_reacher import ReacherEpisode, native_replay


@pytest.fixture(autouse=True)
def isolate():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


def plan():
    return {
        "rng_namespace": "reacher-cache-control-engineering-unit-v1",
        "engineering": True,
        "steps": 50,
        "control_episodes": 1,
        "noise_std": 0.05,
        "dt": 0.02,
        "planning_horizon": 2,
        "action_block": 1,
        "prediction_horizons": [1, 3, 7],
        "hidden_size": 4,
        "mlp_width": 7,
    }


def model(kind):
    width = {"width": 7} if kind in ("packet_mlp", "cached_mlp") else {"hidden_size": 4}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        return training.REGISTRY[kind](**width, noise_std=0.05).eval()


def cases(panel="ordinary", count=1):
    schedule = np.ones(51, dtype=bool)
    if panel != "full":
        gap = 6 if panel == "ordinary" else 10
        schedule[8 : 8 + gap] = False
        schedule[28 : 28 + gap] = False
    # Repeated literal410 is an intentionally tiny engineering fixture, not a
    # claim of independent production streams. The helper selects no seeds.
    return [control.ControlCase(410, 410, schedule) for _ in range(count)]


def innovations(count=1):
    rng = np.random.default_rng(410)
    arrays = [rng.normal(size=(count, n, 2, 2)) for n in (64, 192, 64, 64, 63)]
    return SearchInputs(arrays[0], arrays[1], tuple(arrays[2:]))


def public_record():
    env = ReacherEpisode(noise_std=0.05)
    case = cases()[0]
    try:
        env.reset(case.reset_seed, case.sensor_schedule, noise_seed=case.noise_seed)
        commands = np.linspace(-0.12, 0.12, 100, dtype=np.float32).reshape(50, 2)
        for command in commands:
            env.step(command)
        return env.episode_record()
    finally:
        env.close()


@pytest.mark.parametrize("kind", training.REGISTRY)
@pytest.mark.parametrize("panel", control.PANELS)
def test_complete_closed_loop_each_class_and_panel_native_replay_and_alignment(kind, panel, tmp_path):
    cfg, student, bank = plan(), model(kind), innovations()
    before = training.canonical_tensor_hash(student.state_dict())
    destination = tmp_path / f"{kind}-{panel}"
    timing = control.learned_control(cfg, student, panel, [bank] * 50, destination, cases=cases(panel))
    record = legacy.load_records(destination / "episodes")[0]
    assert native_replay(record)["transitions"] == 50
    assert timing["observation_assimilations"] == timing["executed_action_advances"] == 50
    assert len(timing["decision_seconds"]) == len(timing["native_step_seconds"]) == 50
    assert timing["row_wall_seconds"] >= timing["setup_seconds"] + sum(timing["decision_seconds"]) + sum(
        timing["native_step_seconds"]
    )
    assert before == training.canonical_tensor_hash(student.state_dict())
    assert all(p.grad is None for p in student.parameters())
    state_work = json.loads((destination / "state-work.json").read_text())
    assert state_work["model"]["kind"] == kind
    assert state_work["reset_calls"] == state_work["auxiliary_teacher_calls"] == 0
    assert state_work["aggregate_model_work"]["advance_samples"] == 50 + 256 * (49 * 2 + 1)
    with (
        np.load(destination / "states.npz", allow_pickle=False) as states,
        np.load(destination / "executed_predictions.npz", allow_pickle=False) as predictions,
    ):
        assert state_work["state_arrays_bytes"] == sum(states[k].nbytes for k in states.files)
        np.testing.assert_array_equal(states["root__packet"][0], record["policy"]["packets"][:-1])
        np.testing.assert_array_equal(states["carried__packet"][..., :4], predictions["angles"])
        assert (states["carried__packet"][..., 6] == 0).all()
        for step in range(50):
            with np.load(destination / "decisions" / f"{step:03d}.npz", allow_pickle=False) as trace:
                selected = trace["selected_ids"]
                chosen = trace["chunks"][np.arange(1), selected, 0]
                np.testing.assert_array_equal(chosen[0], record["policy"]["commands"][step])
                np.testing.assert_allclose(
                    predictions["rewards"][:, step],
                    trace["raw_rewards"][np.arange(1), selected, 0],
                    rtol=1e-5,
                    atol=1e-6,
                )
            root = {
                key.split("__", 1)[1]: torch.from_numpy(states[key][:, step].copy())
                for key in states.files
                if key.startswith("root__")
            }
            carried = {
                key.split("__", 1)[1]: torch.from_numpy(states[key][:, step].copy())
                for key in states.files
                if key.startswith("carried__")
            }
            assert training.canonical_tensor_hash(root) == state_work["steps"][step]["root_sha256"]
            assert training.canonical_tensor_hash(carried) == state_work["steps"][step]["carried_sha256"]
            if kind in ("encoded_current_gru", "cached_gru", "cached_mlp"):
                assert (states["root__imagined_depth"][:, step] == 0).all()
                assert (states["carried__imagined_depth"][:, step] == 1).all()
            if kind in ("cached_gru", "cached_mlp"):
                previous = record["policy"]["packets"][:step + 1]
                last = np.flatnonzero(previous[:, 6])[-1]
                np.testing.assert_array_equal(states["root__cached_angles"][0, step], previous[last, :4])
                np.testing.assert_array_equal(states["carried__cached_angles"][:, step], states["root__cached_angles"][:, step])
    assert not (destination / "failed.json").exists()


@pytest.mark.parametrize("kind", training.REGISTRY)
def test_private_candidates_do_not_mutate_real_root_or_retained_history(kind):
    cfg, student, record, bank = plan(), model(kind), public_record(), innovations()
    packet = torch.from_numpy(record["policy"]["packets"][:1].copy())
    root = student.assimilate(student.initial(1), packet)
    before = training.canonical_tensor_hash(root)
    result, raw, sizes, _, work = control.score_search(cfg, student, root, bank, 0)
    assert sizes == [64, 64, 64, 64]
    assert work["imagined_transitions"] == 512
    assert work["max_single_candidate_state_tensor_bytes"] == 64 * work["root_tensor_bytes"]
    assert training.canonical_tensor_hash(root) == before
    for candidate in (0, 63, 64, 255):
        future = {key: value.clone() for key, value in root.items()}
        for offset in range(2):
            with torch.no_grad():
                future, _, reward = student.advance(
                    future, torch.from_numpy(result.sequences[:, candidate, offset].copy())
                )
            np.testing.assert_allclose(reward.numpy(), raw[:, candidate, offset], atol=1e-6, rtol=1e-5)
    if kind in ("encoded_current_gru", "cached_gru", "cached_mlp"):
        with pytest.raises(ValueError):
            student.assimilate(future, packet)
        issued, _, _ = student.advance(root, torch.from_numpy(result.selected_actions.copy()))
        next_real = student.assimilate(issued, torch.from_numpy(record["policy"]["packets"][1:2].copy()))
        assert (next_real["imagined_depth"] == 0).all()
    assert training.canonical_tensor_hash(root) == before


@pytest.mark.parametrize("kind", training.REGISTRY)
def test_prediction_all_complete_windows_and_no_privileged_inputs(kind):
    cfg, student, record = plan(), model(kind), public_record()

    class PublicOnly(dict):
        def __getitem__(self, key):
            if key != "policy":
                raise AssertionError("Privileged input read")
            return super().__getitem__(key)

    records = [PublicOnly(policy=record["policy"])]
    values, costs = control.prediction_record(cfg, student, records)
    assert np.array_equal(values["one_step_angles"], values["h1_angles"])
    observed = record["policy"]["packets"][None, :, 6] > 0.5
    for horizon in cfg["prediction_horizons"]:
        assert values[f"h{horizon}_angles"].shape == (1, 51 - horizon, 4)
        np.testing.assert_array_equal(
            values[f"h{horizon}_valid"], observed[:, :-horizon] & observed[:, horizon:]
        )
    assert costs["student_assimilations"] == costs["prefix_advances"] == 50
    assert costs["imagined_advances"] == sum(min(7, 50 - root) for root in range(50))
    assert costs["teacher_or_auxiliary_calls"] == 0
    assert costs["model_work"]["advance_samples"] == costs["prefix_advances"] + costs["imagined_advances"]
    changed = copy.deepcopy(record)
    # Future measurements must not affect a past root or its open-loop window.
    # Keep issued actions unchanged: multi-step windows legitimately use them.
    changed["policy"]["packets"][20:, :4] *= -1
    changed["audit"] = {"unreadable": object()}
    later, _ = control.prediction_record(cfg, student, [changed])
    for key in ("one_step_angles", "one_step_rewards", "h1_angles", "h3_angles", "h7_angles"):
        np.testing.assert_array_equal(later[key][:, :20], values[key][:, :20])


def test_saved_innovations_path_is_loaded_without_new_rng_draws(tmp_path, monkeypatch):
    cfg, student, bank = plan(), model("packet_mlp"), innovations()
    stem = tmp_path / "inputs"
    artifacts.save_inputs(stem, bank, "engineering-only/410")
    # Wrapper reset/noise still legitimately use their explicitly supplied RNG.
    # The named input loader itself must not call an innovation generator.
    monkeypatch.setattr(
        artifacts, "draw_inputs", lambda *_: (_ for _ in ()).throw(AssertionError("Unexpected draw"))
    )
    out = tmp_path / "loaded"
    control.learned_control(cfg, student, "ordinary", [stem] * 50, out, cases=cases())
    trace = json.loads((out / "decisions" / "000.json").read_text())
    assert trace["input_identities"] == dict(bank.identities())


@pytest.mark.parametrize("bad", ("class", "train", "width", "noise", "steps", "panel", "cases", "inputs"))
def test_incompatible_actual_class_or_configuration_fails_before_native_calls(bad, tmp_path, monkeypatch):
    cfg, student, panel, case_list, banks = (
        plan(),
        model("encoded_current_gru"),
        "ordinary",
        cases(),
        [innovations()] * 50,
    )
    if bad == "class":

        class Wrong(type(student)):
            pass

        student = Wrong(hidden_size=4, noise_std=0.05).eval()
    elif bad == "train":
        student.train()
    elif bad == "width":
        cfg["hidden_size"] = 8
    elif bad == "noise":
        cfg["noise_std"] = 0.1
    elif bad == "steps":
        cfg["steps"] = 49
    elif bad == "panel":
        panel = "reset"
    elif bad == "cases":
        case_list = []
    else:
        banks = banks[:-1]
    monkeypatch.setattr(
        control,
        "ReacherEpisode",
        lambda **_: (_ for _ in ()).throw(AssertionError("Native call before validation")),
    )
    with pytest.raises(ValueError):
        control.learned_control(cfg, student, panel, banks, tmp_path / "bad", cases=case_list)


def test_ragged_native_failure_preserves_all_completed_cases_and_original_interrupt(tmp_path, monkeypatch):
    cfg = plan()
    cfg["control_episodes"] = 2
    created = []

    class FailingEpisode(ReacherEpisode):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.index = len(created)
            self.was_closed = False
            created.append(self)

        def step(self, action):
            if self.index == 1 and self.step_index == 0:
                raise KeyboardInterrupt("synthetic native failure")
            return super().step(action)

        def close(self):
            self.was_closed = True
            super().close()

    monkeypatch.setattr(control, "ReacherEpisode", FailingEpisode)
    out = tmp_path / "failed"
    with pytest.raises(KeyboardInterrupt, match="synthetic native failure"):
        control.learned_control(
            cfg, model("cached_gru"), "ordinary", [innovations(2)] * 50, out, cases=cases(count=2)
        )
    receipt = json.loads((out / "failed.json").read_text())
    assert receipt["completed_steps_by_case"] == [1, 0]
    partial = json.loads((out / "partial-episodes" / "manifest.json").read_text())
    assert partial["completed_steps_by_case"] == [1, 0]
    assert all(env.was_closed for env in created)
    for index, length in enumerate((1, 0)):
        assert (
            len(legacy.load_records(out / "partial-episodes" / f"{index:03d}")[0]["policy"]["commands"])
            == length
        )


def test_preservation_failure_does_not_mask_original_baseexception(tmp_path, monkeypatch):
    monkeypatch.setattr(
        control, "score_search", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("original"))
    )
    monkeypatch.setattr(control, "_save_partial", lambda *_: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(KeyboardInterrupt, match="original") as caught:
        control.learned_control(
            plan(),
            model("residual_gru"),
            "ordinary",
            [innovations()] * 50,
            tmp_path / "failure",
            cases=cases(),
        )
    assert "preservation" in caught.value.__notes__[0]


def test_case_schedule_copy_is_immutable_and_rejects_missing_initial_packet():
    schedule = np.ones(51, dtype=bool)
    case = control.ControlCase(410, 410, schedule)
    schedule[:] = False
    assert case.sensor_schedule.all()
    with pytest.raises(ValueError):
        case.sensor_schedule.flags.writeable = True
    with pytest.raises(ValueError):
        control.ControlCase(410, 410, schedule)


def test_wrong_public_prediction_shapes_and_privileged_channels_rejected():
    record = public_record()
    record["policy"]["hidden_qvel"] = np.zeros((51, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="public"):
        control.prediction_record(plan(), model("packet_mlp"), [record])
    record = public_record()
    record["policy"]["commands"] = record["policy"]["commands"][:-1]
    with pytest.raises(ValueError, match="histories"):
        control.prediction_record(plan(), model("packet_mlp"), [record])
