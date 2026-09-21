"""Scripted public observations and pure beliefs only; no real environment imports."""
from __future__ import annotations

import builtins
import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from openjev.research.rocksample_memory_controls import PublicTransition
from openjev.research.rocksample_particle_belief import ParticleRockBelief
from openjev.research.rocksample_quality_belief import QualityRockBelief

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/study_rocksample_policy_value.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("synthetic_policy_value", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner():
    return load_runner()


def observation(position=(0, 0), rock=None, reading=1):
    result = np.zeros(33)
    result[position[0]], result[11 + position[1]] = 1, 1
    if rock is not None:
        result[22 + rock] = reading
    return result


class OpaqueState:
    def __getattr__(self, name):
        raise AssertionError(f"public arm accessed private state: {name}")


class PublicInfo(dict):
    def __getitem__(self, key):
        assert key in {"truncated", "time_limit_reached"}, f"private info accessed: {key}"
        return super().__getitem__(key)


class ScriptedEnvironment:
    """Return predetermined fixtures; do not simulate transitions or draw outcomes."""

    def __init__(self, outputs, *, initial=(0, 0), privileged=False, fail_step=False):
        self.outputs = list(outputs)
        self.initial = initial
        self.privileged = privileged
        self.fail_step = fail_step
        self.actions, self.keys, self.reset_keys = [], [], []
        self.hidden_reads = []
        self._maps = np.array([(0, c) for c in range(10)] + [(1, 0)])
        self._state = (SimpleNamespace(env_state=SimpleNamespace(rock_morality=np.ones(11)))
                       if privileged else OpaqueState())

    @property
    def rock_positions(self):
        assert self.privileged, "public arm accessed true rock map"
        self.hidden_reads.append("map")
        return self._maps

    def reset(self, key, params):
        self.reset_keys.append(key)
        return observation(self.initial), self._state

    def step(self, key, state, action, params):
        assert state is self._state
        self.actions.append(action)
        self.keys.append(key)
        if self.fail_step:
            raise RuntimeError("scripted unreturned transition")
        expected, obs, reward, done, truncated = self.outputs.pop(0)
        assert action == expected
        return obs, self._state, reward, done, PublicInfo(truncated=truncated, time_limit_reached=truncated)


class FakeRandom:
    @staticmethod
    def PRNGKey(seed):
        return int(seed)

    @staticmethod
    def split(key):
        return key + 1, key + 1_000_000


class GuardBelief:
    """Audit every argument entering a public belief, while retaining real pure math."""

    def __init__(self, inner, calls):
        self.inner, self.calls = inner, calls
        self.size, self.rocks = inner.size, inner.rocks

    def copy(self):
        self.calls.append(("copy",))
        return GuardBelief(self.inner.copy(), self.calls)

    def reset(self):
        self.calls.append(("reset",))
        self.inner.reset()

    def diagnostics(self):
        return self.inner.diagnostics()

    def condition_check(self, rock, position, positive):
        assert type(rock) is int and type(positive) is bool
        assert type(position) is tuple and all(type(x) is int for x in position)
        self.calls.append(("check", rock, position, positive))
        return GuardBelief(self.inner.condition_check(rock, position, positive), self.calls)

    def sample(self, position):
        assert type(position) is tuple and all(type(x) is int for x in position)
        self.calls.append(("sample", position))
        self.inner.sample(position)

    def quality_probabilities(self):
        return self.inner.quality_probabilities()


def harness(runner, monkeypatch, arm, outputs, plans, *, prior=None, initial=(0, 0), fail_step=False):
    calls, decisions, emitted = [], [], []
    counters = {"steps_attempted": 0, "steps_returned": 0}
    checks = []
    env = ScriptedEnvironment(outputs, initial=initial, privileged=arm == "privileged", fail_step=fail_step)
    wrapped = GuardBelief(prior or ParticleRockBelief(particles=16), calls)
    monkeypatch.setattr(runner, "QualityRockBelief", lambda: GuardBelief(QualityRockBelief(), calls))

    def explicit_reference(maps, qualities):
        assert arm == "privileged"
        calls.append(("privileged_reference",))
        return GuardBelief(ParticleRockBelief.from_hypotheses(maps, qualities), calls)

    monkeypatch.setattr(runner, "ParticleRockBelief", SimpleNamespace(from_hypotheses=explicit_reference))

    def planner(belief, coordinate, ledger, remaining_steps, checks_remaining):
        assert isinstance(belief, GuardBelief)
        assert ledger.shape == (11, 10) and ledger.dtype == bool
        index = len(decisions)
        decisions.append({"position": coordinate, "ledger": ledger.copy(), "remaining": remaining_steps,
                          "checks_remaining": checks_remaining,
                          "qualities": belief.quality_probabilities().copy()})
        return copy.deepcopy(plans[index])

    monkeypatch.setattr(runner, "plan_action", planner)

    def invoke():
        return runner.play_episode(env, object(), arm, 22001, 430000, wrapped,
                                   SimpleNamespace(random=FakeRandom),
                                   lambda kind, row: emitted.append((kind, copy.deepcopy(row))),
                                   lambda: checks.append(True), counters)

    return SimpleNamespace(run=invoke, env=env, calls=calls, decisions=decisions, emitted=emitted,
                           counters=counters, checks=checks, prior=wrapped)


def macro(kind, actions, work=None):
    return {"kind": kind, "actions": actions, "work": {} if work is None else work}


def exit_outputs():
    return [(1, observation((0, column)) if column < 10 else observation((4, 4)),
             0 if column < 10 else 10, column == 10, False) for column in range(1, 11)]


def test_import_has_no_benchmark_or_neural_dependencies(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"jax", "jaxlib", "pobax", "gymnax", "torch", "brax", "navix"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert load_runner().ARMS == ("exit", "full", "recent128", "latest", "quality", "privileged")


@pytest.mark.parametrize("arm", ["exit", "full", "recent128", "latest", "quality", "privileged"])
def test_actual_episode_loop_keeps_public_inputs_separate_and_accounts_for_every_step(runner, monkeypatch, arm):
    outputs = exit_outputs()
    plans = []
    if arm != "exit":
        outputs = [(5, observation(rock=0), 0, False, False), (4, observation(), 10, False, False), *outputs]
        plans = [macro("sense", [5], {"calls": 1}), macro("exploit", [4], {"calls": 2}),
                 macro("exit", [1] * 10, {"calls": 3})]
    test = harness(runner, monkeypatch, arm, outputs, plans)
    before = test.prior.inner._log_quality.copy()
    row = test.run()
    expected_steps = 10 if arm == "exit" else 12
    assert row["arm"] == arm and row["steps"] == expected_steps
    assert row["raw_return"] == (10 if arm == "exit" else 20)
    expected_discounted = 10 * 0.99**9 if arm == "exit" else 10 * 0.99 + 10 * 0.99**11
    assert row["discounted_return_099"] == pytest.approx(expected_discounted)
    assert row["exited"] and not row["truncated"]
    assert row["checks"] == row["samples"] == row["good_samples"] == (0 if arm == "exit" else 1)
    assert row["bad_samples"] == row["empty_samples"] == 0
    assert row["samples"] == row["good_samples"] + row["bad_samples"] + row["empty_samples"]
    assert test.counters["steps_attempted"] == test.counters["steps_returned"] == expected_steps
    assert test.env.reset_keys == [22001]
    assert test.env.keys == list(range(1_430_000, 1_430_000 + expected_steps))
    assert not test.env.outputs
    assert len([value for kind, value in test.emitted if kind == "transitions"]) == expected_steps + 1
    assert all(row[key] >= 0 for key in ("filter_seconds", "planning_seconds", "environment_seconds"))
    assert row["controller_seconds"] == row["filter_seconds"] + row["planning_seconds"]
    np.testing.assert_array_equal(test.prior.inner._log_quality, before)
    if arm == "privileged":
        assert test.env.hidden_reads == ["map"]
        assert ("privileged_reference",) in test.calls
        assert row["information"] == "true map and qualities"
    else:
        assert not test.env.hidden_reads and ("privileged_reference",) not in test.calls
        assert row["information"] == "public history only"
    if arm != "exit":
        assert [d["checks_remaining"] for d in test.decisions] == [64, 63, 63]
        assert [int(d["ledger"].sum()) for d in test.decisions] == [0, 0, 1]
        assert test.decisions[-1]["ledger"][0, 0]
        assert row["work"]["calls"] == 6
        assert row["plan_kinds"] == {"sense": 1, "exploit": 1, "exit": 1}
    else:
        assert not test.decisions and row["plans"] == 1


@pytest.mark.parametrize("arm", ["recent128", "latest"])
def test_partial_reconstruction_sees_all_completed_primitive_actions_and_old_depletion(runner, monkeypatch, arm):
    prior = ParticleRockBelief.from_hypotheses([[(0, c) for c in range(10)] + [(1, 0)]], qualities=1)
    prefix = [(4, observation(), 10, False, False)] + [(0, observation(), 0, False, False)] * 128
    outputs = [*prefix, (5, observation(rock=0, reading=-1), 0, False, False), *exit_outputs()]
    plans = [macro("exploit", [4] + [0] * 128), macro("sense", [5]), macro("exit", [1] * 10)]
    captured = []
    original_reconstruct = runner.reconstruct

    def checked_reconstruct(initial, history, mode):
        assert all(type(event) is PublicTransition for event in history)
        captured.append((mode, list(history)))
        return original_reconstruct(initial, history, mode)

    monkeypatch.setattr(runner, "reconstruct", checked_reconstruct)
    test = harness(runner, monkeypatch, arm, outputs, plans, prior=prior)
    row = test.run()
    assert [len(history) for _, history in captured] == [0, 129, 130]
    assert all(mode == arm for mode, _ in captured)
    assert captured[1][1][0] == PublicTransition((0, 0), 4)
    assert sum(event.action == 0 for event in captured[1][1]) == 128
    assert test.decisions[1]["qualities"][0] == 0
    assert row["raw_return"] == 20 and row["steps"] == 140


def test_plan_counts_sum_but_maximum_diagnostics_take_the_maximum(runner, monkeypatch):
    outputs = [(5, observation(rock=0), 0, False, False), *exit_outputs()]
    plans = [macro("sense", [5], {"calls": 3, "maximum_leaves_per_bundle": 8,
                                  "maximum_branch_mass_error": 2e-16}),
             macro("exit", [1] * 10, {"calls": 4, "maximum_leaves_per_bundle": 16,
                                      "maximum_branch_mass_error": 1e-16})]
    row = harness(runner, monkeypatch, "full", outputs, plans).run()
    assert row["work"] == {"calls": 7, "maximum_leaves_per_bundle": 16,
                            "maximum_branch_mass_error": 2e-16}


def test_true_truncation_is_accounted_without_assimilating_reset_observation(runner, monkeypatch):
    monkeypatch.setattr(runner, "HORIZON", 2)
    outputs = [(0, observation(), 0, False, False), (5, observation((4, 4)), 0, True, True)]
    test = harness(runner, monkeypatch, "full", outputs, [macro("sense", [0, 5])])
    row = test.run()
    assert row["truncated"] and not row["exited"]
    assert row["steps"] == 2 and row["checks"] == 1 and row["raw_return"] == 0
    assert not any(call[0] == "check" for call in test.calls)
    assert test.counters["steps_attempted"] == test.counters["steps_returned"] == 2


def test_native_boundary_inside_macro_fails_with_returned_step_preserved(runner, monkeypatch):
    test = harness(runner, monkeypatch, "full", [(1, observation((4, 4)), 10, True, False)],
                   [macro("exit", [1, 1])], initial=(0, 9))
    with pytest.raises(ValueError, match="inside a committed macro"):
        test.run()
    assert test.counters["steps_attempted"] == test.counters["steps_returned"] == 1
    assert test.emitted[-1][1]["done"] is True


def test_resampling_rejected_before_second_environment_call(runner, monkeypatch):
    test = harness(runner, monkeypatch, "full", [(4, observation(), 10, False, False)],
                   [macro("exploit", [4, 4])])
    with pytest.raises(ValueError, match="resample"):
        test.run()
    assert test.env.actions == [4]
    assert test.counters["steps_attempted"] == test.counters["steps_returned"] == 1


def test_unreturned_environment_call_is_distinct_from_a_returned_transition(runner, monkeypatch):
    test = harness(runner, monkeypatch, "full", [], [macro("sense", [5])], fail_step=True)
    with pytest.raises(RuntimeError, match="unreturned transition"):
        test.run()
    assert test.counters["steps_attempted"] == 1 and test.counters["steps_returned"] == 0
    assert [value["step"] for kind, value in test.emitted if kind == "transitions"] == [-1]


@pytest.mark.parametrize("actions,match", [([], "invalid committed"), ([True], "invalid committed"),
                                          ([16], "invalid committed"), ([0] * 4, "invalid committed"),
                                          ([5, 5], "sensing budget")])
def test_invalid_macro_fails_before_any_environment_step(runner, monkeypatch, actions, match):
    monkeypatch.setattr(runner, "HORIZON", 3)
    monkeypatch.setattr(runner, "MAX_CHECKS", 1)
    test = harness(runner, monkeypatch, "full", [], [macro("sense", actions)])
    with pytest.raises(ValueError, match=match):
        test.run()
    assert test.counters["steps_attempted"] == test.counters["steps_returned"] == 0


def test_missing_native_horizon_boundary_fails_instead_of_returning_a_partial_episode(runner, monkeypatch):
    monkeypatch.setattr(runner, "HORIZON", 2)
    test = harness(runner, monkeypatch, "full", [(0, observation(), 0, False, False)] * 2,
                   [macro("exploit", [0, 0])])
    with pytest.raises(ValueError, match="time limit failed"):
        test.run()
    assert test.counters["steps_attempted"] == test.counters["steps_returned"] == 2


def cohort(runner):
    records = []
    values = {"exit": 10, "full": 30, "recent128": 20, "latest": 20, "quality": 20, "privileged": 40}
    for mi, map_seed in enumerate(runner.MAP_SEEDS):
        for ri, reset_seed in enumerate(runner.RESET_SEEDS):
            for arm in runner.ARMS:
                raw = values[arm]
                samples = (raw - 10) // 10
                plans = 1 + ri + mi
                records.append({"map_seed": map_seed, "reset_seed": reset_seed, "arm": arm,
                                "raw_return": raw, "discounted_return_099": raw * 0.9,
                                "steps": 20 + 7 * mi + 3 * ri, "checks": ri, "samples": samples,
                                "good_samples": samples, "bad_samples": 0, "empty_samples": 0,
                                "controller_seconds": 0.3, "filter_seconds": 0.1, "planning_seconds": 0.2,
                                "environment_seconds": 0.01 * (1 + mi), "plans": plans,
                                "exited": True, "truncated": False,
                                "ess_adequate_plans": plans, "ess_evaluated_plans": plans})
    return records


def test_exact_192_matched_cases_all_ten_criteria_and_unweighted_episode_means(runner):
    rows = cohort(runner)
    summary = runner.summarize(rows)
    assert len(rows) == summary["episodes"] == 8 * 4 * 6
    assert summary["learned_pilot_admitted"] is True
    assert len(summary["criteria"]) == 10 and all(item["passes"] for item in summary["criteria"])
    assert [item["name"] for item in summary["criteria"]] == [
        "raw_gain_vs_recent128", "positive_maps_vs_recent128", "raw_gain_vs_latest", "positive_maps_vs_latest",
        "raw_gain_vs_quality", "positive_maps_vs_quality", "full_gain_vs_exit", "privileged_gain_vs_exit",
        "finite_particle_adequacy", "complete_cohort"]
    for row in rows:
        if row["arm"] == "full":
            index = runner.RESET_SEEDS.index(row["reset_seed"])
            row["raw_return"] = [10, 20, 30, 40][index]
            row["steps"] = [10, 100, 200, 1000][index]
    summary = runner.summarize(rows)
    assert summary["scores_equal_map_means"]["full"]["raw_return"] == 25
    assert summary["scores_equal_map_means"]["full"]["steps"] == 327.5
    assert all(item["scores"]["full"]["raw_return"] == 25 for item in summary["by_map"])


@pytest.mark.parametrize("name", ["raw_gain_vs_recent128", "positive_maps_vs_recent128",
                                  "raw_gain_vs_latest", "positive_maps_vs_latest",
                                  "raw_gain_vs_quality", "positive_maps_vs_quality",
                                  "full_gain_vs_exit", "privileged_gain_vs_exit", "finite_particle_adequacy"])
def test_each_numerical_gate_can_fail_alone_and_prevents_admission(runner, name):
    rows = cohort(runner)
    for row in rows:
        if name.startswith("raw_gain") and row["arm"] == name.split("_vs_")[1]:
            row["raw_return"] = 30  # no mean advantage, but fix positive-map test separately below
        elif name.startswith("positive_maps") and row["arm"] == name.split("_vs_")[1]:
            if row["map_seed"] in runner.MAP_SEEDS[:3]:
                row["raw_return"] = 30
        elif name == "full_gain_vs_exit":
            if row["arm"] == "exit":
                row["raw_return"] = 30
            elif row["arm"] == "privileged":
                row["raw_return"] = 50
        elif name == "privileged_gain_vs_exit" and row["arm"] == "privileged":
            row["raw_return"] = 20
        elif name == "finite_particle_adequacy" and row["arm"] == "full":
            row["ess_adequate_plans"] = 0
    if name.startswith("raw_gain"):
        for row in rows:
            if row["arm"] == name.split("_vs_")[1]:
                # Four real-reward-multiple episodes average27.5, a positive but<5 gain on all maps.
                row["raw_return"] = 20 if row["reset_seed"] == runner.RESET_SEEDS[0] else 30
    summary = runner.summarize(rows)
    assert summary["learned_pilot_admitted"] is False
    assert [item["name"] for item in summary["criteria"] if not item["passes"]] == [name]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "extra", "wrong_map", "wrong_reset", "wrong_arm"])
def test_incomplete_or_mismatched_cohort_is_rejected_before_any_passing_summary(runner, defect):
    rows = cohort(runner)
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows[-1] = rows[0].copy()
    elif defect == "extra":
        rows.append(rows[0].copy())
    else:
        key, value = {"wrong_map": ("map_seed", 999), "wrong_reset": ("reset_seed", 999),
                      "wrong_arm": ("arm", "unplanned")}[defect]
        rows[-1][key] = value
    with pytest.raises(ValueError, match="fixed cohort"):
        runner.summarize(rows)


def test_raw_gain_thresholds_are_inclusive_and_positive_maps_exclude_ties(runner):
    rows = cohort(runner)
    for row in rows:
        if row["arm"] == "full":
            row["raw_return"] = 10 if row["reset_seed"] in runner.RESET_SEEDS[:2] else 20
        elif row["arm"] in ("recent128", "latest", "quality"):
            row["raw_return"] = 10
        elif row["arm"] == "privileged":
            row["raw_return"] = 30
    assert runner.summarize(rows)["learned_pilot_admitted"] is True  #5,5,5,5,20 inclusive margins
    rows = cohort(runner)
    for row in rows:
        if row["map_seed"] in runner.MAP_SEEDS[:2] and row["arm"] in ("recent128", "latest", "quality"):
            row["raw_return"] = 30
    summary = runner.summarize(rows)
    assert summary["learned_pilot_admitted"] is True
    for criterion in summary["criteria"]:
        if criterion["name"].startswith("positive_maps"):
            assert criterion["value"] == 6
            assert criterion["map_gains"] == [0, 0, 10, 10, 10, 10, 10, 10]


def test_particle_adequacy_is_plan_weighted_with_exact_ninety_percent_boundary(runner):
    rows = cohort(runner)
    full = [row for row in rows if row["arm"] == "full"]
    for row in full:
        row["ess_evaluated_plans"], row["ess_adequate_plans"] = 10, 9
    summary = runner.summarize(rows)
    criterion = next(item for item in summary["criteria"] if item["name"] == "finite_particle_adequacy")
    assert criterion["passes"] and criterion["value"] == 0.9
    assert criterion["numerator"] == 288 and criterion["denominator"] == 320
    full[0]["ess_adequate_plans"] = 8
    assert not runner.summarize(rows)["learned_pilot_admitted"]
    for row in full:
        row["ess_evaluated_plans"] = row["ess_adequate_plans"] = 1
    full[0]["ess_evaluated_plans"], full[0]["ess_adequate_plans"] = 10, 0
    criterion = next(item for item in runner.summarize(rows)["criteria"] if item["name"] == "finite_particle_adequacy")
    assert criterion["numerator"] == 31 and criterion["denominator"] == 41
    assert not criterion["passes"]  #31/32 adequate episodes would incorrectly pass
    for row in full:
        row["ess_evaluated_plans"] = row["ess_adequate_plans"] = 0
    assert not runner.summarize(rows)["learned_pilot_admitted"]


def execution_fixture(runner, monkeypatch, tmp_path, *, bad_pin=False, expire_after_publication=False):
    """Qualify only lifecycle flow with temp bytes and in-memory module stubs.

    Scientific scoring is tested separately on the complete192-case cohort.
    Here an empty map iterable and a mocked summary bypass all environment work.
    """
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "synthetic-output"
    qualified = tmp_path / "output/rocksample-runtime-v1/qualification-01"
    qualified.mkdir(parents=True)

    def write_json(path, value):
        path.write_text(json.dumps(value))

    source = tmp_path / "synthetic-source.txt"
    source.write_text("synthetic pinned bytes, never executed")
    gymnax_source = tmp_path / "synthetic-gymnax.txt"
    gymnax_source.write_text("synthetic dependency bytes, never imported")
    imports_path = qualified / "imports.json"
    write_json(imports_path, {"gymnax_environment": {"path": str(gymnax_source),
                                                     "sha256": runner.sha(gymnax_source)},
                              "upstream_imported_source_sha256": {}, "versions": {"synthetic-runtime": "1"}})
    completed = qualified / "completed.json"
    write_json(completed, {"status": "completed", "files": {
        "imports.json": {"sha256": runner.sha(imports_path), "bytes": imports_path.stat().st_size}}})
    terminal = qualified.parent / "qualification-process-01.terminal.json"
    write_json(terminal, {"status": "completed"})
    plan_path = tmp_path / "plan.json"
    if bad_pin:
        plan_path.write_text("this is deliberately not JSON; pin validation must precede decoding")
        plan_pin = "0" * 64
    else:
        write_json(plan_path, {"sources": {source.name: runner.sha(source)},
                              "runtime_qualification_sha256": runner.sha(completed),
                              "runtime_terminal_sha256": runner.sha(terminal)})
        plan_pin = runner.sha(plan_path)
    args = SimpleNamespace(output=output, plan=plan_path, plan_sha256=plan_pin,
                           supervision=tmp_path / "supervision.json")
    argv = [str(SCRIPT), "--output", str(output), "--plan", str(plan_path), "--plan-sha256", plan_pin,
            "--supervision", str(args.supervision)]
    monkeypatch.setattr(runner.sys, "argv", argv)
    deadline = 1800 * 10**9

    class Clock:
        backend = "synthetic-only"

        def now_ns(self):
            return deadline if expire_after_publication and (output / "receipt.json").exists() else 1

    monkeypatch.setattr(runner, "SuspendClock", Clock)
    monkeypatch.setattr(runner.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=1024))
    write_json(args.supervision, {"command": [runner.sys.executable, *argv], "pid": runner.os.getpid(),
                                 "pgid": runner.os.getpgrp(), "parent_pid": runner.os.getppid(),
                                 "cap_seconds": 1800, "clock_backend": Clock.backend, "cwd": str(tmp_path),
                                 "started_ns": 0, "deadline_ns": deadline})
    runtime_calls = []

    def forbidden_environment(*args, **kwargs):
        raise AssertionError("lifecycle test must not instantiate or step any environment")

    stubs = {}
    for name in ("qualify_rocksample_runtime", "jax", "pobax", "pobax.envs", "pobax.envs.jax",
                 "pobax.envs.jax.rocksample", "pobax.envs.wrappers", "pobax.envs.wrappers.gymnax"):
        module = ModuleType(name)
        module.__path__ = []
        stubs[name] = module
        monkeypatch.setitem(runner.sys.modules, name, module)
    stubs["qualify_rocksample_runtime"].authenticate_source = lambda: runtime_calls.append("authenticate")
    stubs["jax"].devices = lambda: [SimpleNamespace(platform="cpu")]
    stubs["jax"].random = SimpleNamespace(PRNGKey=forbidden_environment)
    stubs["pobax.envs.jax.rocksample"].RockSample = forbidden_environment
    stubs["pobax.envs.wrappers.gymnax"].TimeLimitWrapper = forbidden_environment
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: "1" if name == "synthetic-runtime" else None)
    monkeypatch.setattr(runner, "MAP_SEEDS", ())
    monkeypatch.setattr(runner, "ParticleRockBelief", lambda: SimpleNamespace(maps=np.zeros((1, 1, 2))))
    monkeypatch.setattr(runner, "play_episode", forbidden_environment)

    def lifecycle_summary(rows):
        assert rows == []
        return {"learned_pilot_admitted": False, "synthetic_lifecycle_fixture": True}

    monkeypatch.setattr(runner, "summarize", lifecycle_summary)
    return args, runtime_calls


def test_execute_rejects_external_plan_pin_before_decode_or_runtime_authentication(runner, monkeypatch, tmp_path):
    args, runtime_calls = execution_fixture(runner, monkeypatch, tmp_path, bad_pin=True)
    original_read = Path.read_text

    def guard_plan_decode(path, *read_args, **kwargs):
        assert path != args.plan, "mismatched plan must not be decoded"
        return original_read(path, *read_args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guard_plan_decode)
    with pytest.raises(ValueError, match="externally pinned plan changed"):
        runner.execute(args)
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["steps_attempted"] == receipt["steps_returned"] == receipt["completed_episodes"] == 0
    assert not runtime_calls and not (args.output / "summary.json").exists()


@pytest.mark.parametrize("rename_fails", [False, True])
def test_late_completion_failure_preserves_original_timeout_even_if_rename_fails(runner, monkeypatch,
                                                                               tmp_path, rename_fails):
    args, runtime_calls = execution_fixture(runner, monkeypatch, tmp_path, expire_after_publication=True)
    original_rename = Path.rename

    def injected_rename(path, target):
        if rename_fails and path == args.output / "receipt.json":
            raise OSError("synthetic late-publication rename failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", injected_rename)
    with pytest.raises(TimeoutError, match="native deadline expired") as failure:
        runner.execute(args)
    assert len(runtime_calls) >= 2
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["steps_attempted"] == receipt["steps_returned"] == receipt["completed_episodes"] == 0
    assert json.loads((args.output / "summary.json").read_text())["synthetic_lifecycle_fixture"]
    if rename_fails:
        # Filesystem failure leaves the earlier receipt intact. A consumer must
        # also require a successful supervisor terminal before accepting it.
        assert receipt["status"] == "completed"
        assert not (args.output / "late-completed.json").exists()
        assert any("Late failure publication also failed" in note and "rename failure" in note
                   for note in failure.value.__notes__)
    else:
        assert receipt["status"] == "failed" and receipt["late_failure"] is True
        assert "TimeoutError" in receipt["error"]
        earlier = json.loads((args.output / "late-completed.json").read_text())
        assert earlier["status"] == "completed"
        assert earlier["files"] == receipt["files"]
    assert not receipt["learned_pilot_admitted"]
