"""Fake-only orchestration tests: no learned or native environment execution."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research import reacher_two_observation_episode as episode


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def harness(tmp_path, monkeypatch):
    state = SimpleNamespace(envs=[], controllers=[], calls=[], inputs=[], fail_reset=None, fail_native=None,
        fail_controller=None, fail_close=None, corrupt_command=False, corrupt_packet=False, early_end=False,
        corrupt_decision=False, reset_dt=.02, wrong_input_count=False)
    state.identity = {"kind": "fake-controller-only", "weight_tensor_sha256": "a" * 64}
    state.plan = {"steps": 50, "dt": .02, "noise_std": .05, "planning_horizon": 12,
                  "action_block": 3, "control_episodes": 2, "engineering": True}
    schedule = np.ones(51, dtype=bool)
    schedule[2:12] = False
    state.cases = [episode.ControlCase(410, 410, schedule) for _ in range(2)]
    state.out = tmp_path / "row"
    stems = []
    for step in range(50):
        stem = tmp_path / "input" / f"{step:03d}"
        stem.parent.mkdir(exist_ok=True)
        stem.with_suffix(".npz").write_bytes(b"synthetic-unparsed-innovation-fixture")
        stem.with_suffix(".json").write_text(json.dumps({"synthetic_step": step}))
        stems.append(stem)
    state.stems = stems

    def load_inputs(stem):
        result = SimpleNamespace(initial=np.zeros((1 if state.wrong_input_count else 2, 64, 4, 2)),
                                 identities=lambda: {"synthetic": str(stem)})
        state.inputs.append(result)
        return result

    class FakeEnv:
        def __init__(self, noise_std):
            assert noise_std == .05
            self.case = len(state.envs)
            self.dt, self.step_index, self.finished = state.reset_dt, 0, False
            self.closes = 0
            state.envs.append(self)

        def reset(self, seed, schedule, *, noise_seed):
            assert seed == noise_seed == 410
            if self.case == state.fail_reset:
                raise RuntimeError("synthetic reset failure")
            self.schedule = schedule.copy()
            self.packets, self.commands, self.rewards = [], [], []
            value = self.packet()
            self.packets.append(value)
            return value.copy()

        def packet(self):
            visible = self.schedule[self.step_index]
            last = np.flatnonzero(self.schedule[:self.step_index + 1])[-1]
            result = np.array([1, .8, self.step_index / 100, .1, .1, -.1,
                               visible, (self.step_index - last) * .02], dtype=np.float32)
            if not visible:
                result[:4] = 0
            return result

        def step(self, command):
            if (self.case, self.step_index) == state.fail_native:
                raise KeyboardInterrupt("synthetic native failure")
            assert type(command) is np.ndarray and command.dtype == np.float32 and command.shape == (2,)
            self.commands.append(command.copy())
            if state.corrupt_command:
                self.commands[-1][0] += .1
            self.step_index += 1
            self.rewards.append(-(.1 * self.step_index + self.case))
            self.finished = self.step_index == 50 or state.early_end
            value = self.packet()
            self.packets.append(value)
            result = value.copy()
            if state.corrupt_packet:
                result[4] += .01
            return result

        def episode_record(self):
            t = self.step_index
            commands = np.array(self.commands, dtype=np.float32).reshape(t, 2)
            return {"policy": {"packets": np.stack(self.packets), "commands": commands},
                "audit": {"rewards": np.array(self.rewards, dtype=np.float64),
                    "qvel": np.full((t + 1, 4), 123456., dtype=np.float64),
                    "qpos": np.full((t + 1, 4), -987654., dtype=np.float64),
                    "raw_obs": np.full((t + 1, 10), 678910., dtype=np.float64),
                    "integration_state": np.zeros((t + 1, 8)), "time": np.arange(t + 1) * .02,
                    "reward_dist": np.array(self.rewards), "reward_ctrl": np.zeros(t),
                    "actuator_noise": np.full((t, 2), .01), "applied_actions": commands.astype(float) + .01},
                "metadata": {"synthetic_only": True, "case": self.case}}

        def audit_record(self):
            raise AssertionError("Privileged scalar audit_record must not be read")

        def close(self):
            self.closes += 1
            if self.case == state.fail_close:
                raise RuntimeError("synthetic close failure")

    class FakeController:
        def __init__(self, plan, model, batch):
            assert plan == state.plan and model is state.model and batch == 2
            self.seen, self.actions, self._failed, self._state, self.last = [], [], None, {}, None
            state.controllers.append(self)

        def decide(self, packet, inputs, *, issued_action, deadline, failure_out):
            t = len(self.seen)
            assert any(inputs is saved for saved in state.inputs)
            assert packet.shape == (2, 8) and packet.dtype == np.float32
            assert np.max(np.abs(packet)) < 100  # Privileged audit canaries cannot enter here.
            if t == 0:
                assert issued_action is None
            else:
                actual = np.stack([env.episode_record()["policy"]["commands"][-1] for env in state.envs])
                assert np.array_equal(issued_action, actual)
                assert np.array_equal(issued_action, self.last)
                self.actions.append(issued_action.copy())
            if t == state.fail_controller:
                self._failed = {"step": t, "phase": "synthetic_controller_failure"}
                raise KeyboardInterrupt("synthetic controller failure")
            self.seen.append(packet.copy())
            action = np.array([[t / 100, -.1], [t / 100, -.2]], dtype=np.float32)
            packets = np.stack(self.seen, axis=1)
            b = len(packet)
            root = {"hidden": np.zeros((b, 3), dtype=np.float32), "packet": packet.copy(),
                "real_packets": np.zeros((b, 12, 8), dtype=np.float32),
                "real_actions": np.zeros((b, 11, 2), dtype=np.float32),
                "real_present": np.zeros((b, 12), dtype=bool),
                "real_indices": np.full((b, 12), -1, dtype=np.int64),
                "real_index": np.full((b, 1), t, dtype=np.int64), "real_target": packet[:, 4:6].copy(),
                "pending_action": np.zeros((b, 2), dtype=np.float32), "imagined_depth": np.zeros((b, 1), dtype=np.int64)}
            history_actions = np.stack(self.actions, 1) if self.actions else np.zeros((b, 0, 2), dtype=np.float32)
            for case in range(b):
                visible = np.flatnonzero(packets[case, :, 6])
                anchor = visible[-2] if len(visible) > 1 else 0
                start = 12 - (t + 1 - anchor)
                root["real_packets"][case, start:] = packets[case, anchor:]
                root["real_present"][case, start:] = True
                root["real_indices"][case, start:] = np.arange(anchor, t + 1)
                if t > anchor:
                    root["real_actions"][case, start:] = history_actions[case, anchor:]
            carried = copy.deepcopy(root)
            carried["pending_action"] = action.copy()
            carried["imagined_depth"][:] = 1
            carried["packet"][:, :4] = .25
            carried["packet"][:, 6] = 0
            carried["packet"][:, 7] += np.float32(.02)
            self._state, self.last = copy.deepcopy(carried), action.copy()
            meta = {"step": t, "model": state.identity, "search_seconds": 0., "controller_seconds": 0.,
                    "reconstruction_neural_kernels": {"synthetic_only": True}}
            if state.corrupt_decision:
                meta["step"] = t + 1
            state.calls.append((packet.copy(), None if issued_action is None else issued_action.copy()))
            # These are fabricated outputs, not a surrogate model evaluation.
            return SimpleNamespace(action=action, search_result=SimpleNamespace(selected_actions=action.copy(), imagined_transitions=24),
                raw_rewards=np.full((b, 1, 1), -.5, dtype=np.float32), callback_sizes=[64] * 4,
                root_state=root, carried_state=carried, metadata=meta,
                diagnostics={"arrays": {"selected_angles": np.full((b, 4), .25, dtype=np.float32),
                    "selected_learned_reward": np.full(b, -2, dtype=np.float32),
                    "selected_reward": np.full(b, -.5, dtype=np.float32)}, "metadata": {"synthetic_step": t}})

        def snapshot(self):
            return {"state": copy.deepcopy(self._state), "previous_selected_action": None if self.last is None else self.last.copy(),
                    "next_step": len(self.seen), "failure": self._failed, "setup_seconds": 0.}

    def save_trace(stem, result, raw, callbacks, seconds):
        assert callbacks == [64] * 4 and seconds == 0.
        stem.parent.mkdir(parents=True, exist_ok=True)
        episode.artifacts.save_npz(stem.with_suffix(".npz"), selected_actions=result.selected_actions, raw_rewards=raw)
        episode.artifacts.write(stem.with_suffix(".json"), {"synthetic_only": True})

    state.model = object()
    monkeypatch.setattr(episode, "ReacherEpisode", FakeEnv)
    monkeypatch.setattr(episode, "TwoObservationController", FakeController)
    monkeypatch.setattr(episode, "model_identity", lambda model, plan: copy.deepcopy(state.identity))
    monkeypatch.setattr(episode, "work_accounting", lambda model, **kw: {**kw, "synthetic_only": True})
    monkeypatch.setattr(episode.artifacts, "load_inputs", load_inputs)
    monkeypatch.setattr(episode.artifacts, "save_trace", save_trace)
    state.progress = {}
    state.run = lambda: episode.learned_control(state.plan, state.model, "ordinary", state.stems,
                                               state.out, progress=state.progress, cases=state.cases)
    return state


def test_complete_fifty_decisions_final_packet_and_actual_action_acknowledgment(harness):
    h = harness
    timing = h.run()
    assert len(h.calls) == 50 and len(h.inputs) == 50
    assert [env.step_index for env in h.envs] == [50, 50]
    assert [env.closes for env in h.envs] == [1, 1]
    assert h.progress == {"phase": "completed", "step": 50, "completed_model_decisions": 50}
    assert timing["observation_assimilations"] == timing["executed_action_advances"] == 100
    assert len(timing["decision_seconds"]) == len(timing["native_step_seconds"]) == 50
    assert timing["row_wall_seconds"] >= timing["row_payload_wall_seconds"] > 0
    assert sum(timing["decision_seconds"]) + sum(timing["native_step_seconds"]) <= timing["row_wall_seconds"]
    with np.load(h.out / "episodes.npz") as saved:
        packets, commands = saved["policy__packets"], saved["policy__commands"]
        assert packets.shape == (2, 51, 8) and commands.shape == (2, 50, 2)
        assert np.allclose(-saved["audit__rewards"].sum(1), [127.5, 177.5])
        for t, (seen, acknowledged) in enumerate(h.calls):
            assert np.array_equal(seen, packets[:, t])
            if t:
                assert np.array_equal(acknowledged, commands[:, t - 1])
        assert np.all(saved["audit__qvel"] == 123456)
    with np.load(h.out / "states.npz") as saved:
        assert saved["root__real_packets"].shape == (2, 50, 12, 8)
        assert saved["root__real_indices"][0, 11].tolist() == list(range(12))
        assert saved["root__real_indices"][0, 12].tolist() == list(range(1, 13))
        assert np.array_equal(saved["root__real_actions"][0, 12], commands[0, 1:12])
        assert np.array_equal(saved["carried__pending_action"], commands)
        assert np.array_equal(saved["root__real_packets"], saved["carried__real_packets"])
        assert np.all(saved["root__imagined_depth"] == 0) and np.all(saved["carried__imagined_depth"] == 1)
    with np.load(h.out / "executed_predictions.npz") as saved:
        assert saved["angles"].shape == (2, 50, 4)
        assert np.all(saved["rewards"] == -2)  # Original learned reward, not chosen geometry reward.
    assert len(list((h.out / "decisions").glob("*.npz"))) == 50
    assert len(list((h.out / "scoring").glob("*.npz"))) == 50
    assert len(list((h.out / "controller-decisions").glob("*.json"))) == 50
    assert read(h.out / "state-work.json")["aggregate_model_work"]["advance_samples"] == 100 + 50 * 24
    completed = read(h.out / "completed.json")
    assert completed["native_steps_per_case"] == [50, 50]
    assert completed["public_packets_per_case"] == 51
    members = {str(p.relative_to(h.out)) for p in h.out.rglob("*") if p.is_file()} - {"completed.json"}
    assert set(completed["files"]) == members
    assert all(episode.artifacts.sha(h.out / key) == value for key, value in completed["files"].items())
    for row in read(h.out / "inputs.json")["steps"]:
        stem = Path(row["stem"])
        assert row["files_sha256"] == {suffix: episode.artifacts.sha(stem.with_suffix(suffix)) for suffix in (".npz", ".json")}


def test_native_failure_retains_ragged_prefix_and_unissued_decision(harness):
    h = harness
    h.fail_native = (1, 2)
    with pytest.raises(KeyboardInterrupt, match="native failure"):
        h.run()
    failure = read(h.out / "failed.json")
    assert failure["completed_steps_by_case"] == [3, 2]
    assert failure["completed_model_decisions"] == 3
    assert failure["active_native_cases"] == 1 and failure["active_native_seconds"] > 0
    assert read(h.out / "partial-episodes/manifest.json")["completed_steps_by_case"] == [3, 2]
    with np.load(h.out / "partial-states.npz") as saved:
        assert saved["root__packet"].shape == (2, 3, 8)
    assert not (h.out / "completed.json").exists()
    assert [env.closes for env in h.envs] == [1, 1]


def test_controller_failure_keeps_prior_native_and_controller_boundary(harness):
    h = harness
    h.fail_controller = 2
    with pytest.raises(KeyboardInterrupt, match="controller failure"):
        h.run()
    failure = read(h.out / "failed.json")
    assert failure["completed_steps_by_case"] == [2, 2]
    assert failure["completed_model_decisions"] == 2
    assert failure["controller_failure"]["step"] == 2
    assert failure["active_decision_seconds"] > 0
    assert read(h.out / "partial-controller-state.json")["failure"]["phase"] == "synthetic_controller_failure"
    assert [env.closes for env in h.envs] == [1, 1]


@pytest.mark.parametrize("flag", ["corrupt_command", "corrupt_packet", "early_end"])
def test_reject_native_command_packet_or_termination_mismatch(harness, flag):
    h = harness
    setattr(h, flag, True)
    with pytest.raises(ValueError):
        h.run()
    failure = read(h.out / "failed.json")
    assert failure["phase"] == "native_step" and failure["completed_steps_by_case"] == [1, 0]
    assert len(h.calls) == 1 and not (h.out / "completed.json").exists()


def test_exclusive_output_never_retries_or_overwrites(harness):
    h = harness
    h.out.mkdir()
    (h.out / "sentinel").write_text("retain")
    with pytest.raises(FileExistsError):
        h.run()
    assert (h.out / "sentinel").read_text() == "retain"
    assert h.envs == h.calls == []


def test_reset_failure_closes_constructed_envs_and_retains_successful_reset(harness):
    h = harness
    h.fail_reset = 1
    with pytest.raises(RuntimeError, match="reset failure"):
        h.run()
    assert [env.closes for env in h.envs] == [1, 1]
    failure = read(h.out / "failed.json")
    assert failure["initialized_cases"] == 1 and failure["completed_steps_by_case"] == [0]
    assert h.calls == []


def test_close_failure_is_terminal_not_completed_and_not_retried(harness):
    h = harness
    h.fail_close = 0
    with pytest.raises(RuntimeError, match="close failure"):
        h.run()
    assert [env.closes for env in h.envs] == [1, 1]
    assert read(h.out / "failed.json")["phase"] == "cleanup"
    assert not (h.out / "completed.json").exists()


def test_secondary_cleanup_and_preservation_errors_do_not_mask_original(harness, monkeypatch):
    h = harness
    h.fail_native, h.fail_close = (1, 0), 0
    monkeypatch.setattr(episode, "_save_partial", lambda *a: (_ for _ in ()).throw(OSError("synthetic partial write failure")))
    with pytest.raises(KeyboardInterrupt, match="native failure") as caught:
        h.run()
    assert any("close failure" in note for note in caught.value.__notes__)
    assert any("partial write failure" in note for note in caught.value.__notes__)
    assert (h.out / "partial-states.npz").is_file() and (h.out / "failed.json").is_file()


def test_decision_io_failure_preserves_model_decision_before_native_step(harness, monkeypatch):
    h = harness
    monkeypatch.setattr(episode, "_save_decision", lambda *a: (_ for _ in ()).throw(OSError("synthetic trace write")))
    with pytest.raises(OSError, match="trace write"):
        h.run()
    failure = read(h.out / "failed.json")
    assert failure["completed_model_decisions"] == 1 and failure["completed_steps_by_case"] == [0, 0]
    assert (h.out / "partial-states.npz").exists()


def test_expired_cap_preserves_failure_before_any_construction(harness):
    h = harness
    with pytest.raises(TimeoutError):
        episode.learned_control(h.plan, h.model, "ordinary", h.stems, h.out, -1, cases=h.cases)
    assert h.envs == []
    assert read(h.out / "failed.json")["phase"] == "validation"


def test_cap_during_partial_native_batch_is_retained(harness, monkeypatch):
    h = harness
    original = episode.check_cap
    def cap(deadline):
        if h.envs and h.envs[0].step_index == 1 and h.envs[1].step_index == 0:
            raise TimeoutError("synthetic between-case cap")
        return original(deadline)
    monkeypatch.setattr(episode, "check_cap", cap)
    with pytest.raises(TimeoutError, match="between-case"):
        h.run()
    assert read(h.out / "failed.json")["completed_steps_by_case"] == [1, 0]
    assert [env.closes for env in h.envs] == [1, 1]


def test_saved_input_changes_during_load_rejected_before_decision(harness, monkeypatch):
    h = harness
    original = episode.artifacts.load_inputs
    def changed(stem):
        result = original(stem)
        stem.with_suffix(".npz").write_bytes(b"corrupted-after-load")
        return result
    monkeypatch.setattr(episode.artifacts, "load_inputs", changed)
    with pytest.raises(ValueError, match="changed while loading"):
        h.run()
    assert h.calls == []
    assert [env.closes for env in h.envs] == [1, 1]


@pytest.mark.parametrize("mutation", ["wrong_steps", "wrong_dt", "missing_input", "in_memory_input", "wrong_case_count",
                                      "wrong_input_count", "wrong_native_dt", "wrong_decision_clock"])
def test_preflight_and_boundary_rejections(harness, mutation):
    h = harness
    if mutation == "wrong_steps":
        h.plan["steps"] = 49
    elif mutation == "wrong_dt":
        h.plan["dt"] = .03
    elif mutation == "missing_input":
        h.stems = h.stems[:-1]
    elif mutation == "in_memory_input":
        h.stems[0] = object()
    elif mutation == "wrong_case_count":
        h.cases = h.cases[:-1]
    elif mutation == "wrong_input_count":
        h.wrong_input_count = True
    elif mutation == "wrong_native_dt":
        h.reset_dt = .01
    elif mutation == "wrong_decision_clock":
        h.corrupt_decision = True
    with pytest.raises(ValueError):
        h.run()
    assert (h.out / "failed.json").exists() and not (h.out / "completed.json").exists()
    assert all(env.closes == 1 for env in h.envs)


def test_missing_late_input_rejected_before_any_native_setup(harness):
    h = harness
    h.stems[-1].with_suffix(".npz").unlink()
    with pytest.raises(ValueError, match="must exist before native setup"):
        h.run()
    assert h.envs == []
    assert read(h.out / "failed.json")["phase"] == "validation"
