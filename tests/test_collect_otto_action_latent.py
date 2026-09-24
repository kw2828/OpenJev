"""Fabricated collector lifecycle tests; no native environment or trained model."""
from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research import otto_public as public
from openjev.research.otto_query_gate import ReadOnlyBeliefView, _analytic, _features
from openjev.research.otto_released_policy import PublicBeliefView, ReleasedPolicyActor, _move

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


C = sys.modules.get("otto_action_latent_common") or module(
    "scripts/otto_action_latent_common.py", "otto_action_latent_common")
M = module("scripts/collect_otto_action_latent.py", "action_latent_collection_tests")
REFERENCE = module("scripts/study_otto_released_reference.py", "action_latent_original_recorder_tests")


def test_roster_is_exact_disjoint_288_cases_and_balanced_initial_hits():
    rows = C.roster()
    assert len(rows) == len({r["id"] for r in rows}) == len({r["seed"] for r in rows}) == 288
    assert {r["split"] for r in rows} == {"train", "dev"}
    expected = {("train", "lambda3"): set(range(320000001, 320000193)),
                ("dev", "lambda3"): set(range(321000001, 321000049)),
                ("dev", "lambda4"): set(range(322000001, 322000049))}
    for pair, seeds in expected.items():
        selected = [r for r in rows if (r["split"], r["regime"]) == pair]
        assert {r["seed"] for r in selected} == seeds
        assert all(sum(r["initial_hit"] == hit for r in selected) == len(seeds) // 3 for hit in (1, 2, 3))
    assert not {r["seed"] for r in rows} & set(C.CONFIG["fit_seeds"])


def test_action_block_is_reproducible_shared_prefix_and_global_rng_independent():
    before = np.random.get_state()
    a = M.committed_actions(np, 320000001, 4)
    b = M.committed_actions(np, 320000001, 8)
    after = np.random.get_state()
    np.testing.assert_array_equal(a, b[:4])
    np.testing.assert_array_equal(b, M.committed_actions(np, 320000001, 8))
    assert a.dtype == b.dtype == np.int64 and bool(((b >= 0) & (b < 4)).all())
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


@pytest.mark.parametrize("seed,horizon", ((True, 4), (1., 4), (1, 3), (1, 9), (1, True)))
def test_action_block_rejects_unregistered_geometry(seed, horizon):
    with pytest.raises(ValueError, match="declared action block"):
        M.committed_actions(np, seed, horizon)


def test_caps_equal_all_declared_prefix_and_block_work():
    prefix_steps = 8 * 288
    block_steps = 192 * 4 + 96 * 8
    assert M.CALL_CAPS["native_step"] == M.CALL_CAPS["actor_update"] == prefix_steps + block_steps == 3840
    assert M.CALL_CAPS["analytic_score"] == M.CALL_CAPS["feature_build"] == 9 * 288 + block_steps == 4128
    assert M.CALL_CAPS["teacher_score"] == M.CALL_CAPS["tensorflow_value"] == block_steps == 1536
    assert M.CALL_CAPS["native_reset"] == M.CALL_CAPS["actor_construction"] == 288


@pytest.mark.parametrize("outcomes", ([0, 1, 2, 3], [4, 4, 4, 4], [3, 0, 4, 4]))
def test_absorbing_targets_accept_live_hits_then_found(outcomes):
    M.absorbing(np.asarray(outcomes, dtype=np.int64))


@pytest.mark.parametrize("outcomes", ([4, 0], [1, 4, 3], [-2, 4], [5]))
def test_absorbing_targets_reject_native_sentinel_or_post_found_hits(outcomes):
    with pytest.raises(ValueError, match="absorbing terminal outcome"):
        M.absorbing(np.asarray(outcomes, dtype=np.int64))


class AnalyticPolicy:
    def __init__(self, *, env, model, sym_avg):
        self.env = env
        assert model is None and sym_avg is False

    def _value_policy(self):
        return 0, np.arange(4, dtype=np.float64)


class TeacherPolicy:
    def __init__(self, *, env, model, sym_avg):
        self.env, self.model = env, model
        assert sym_avg is True

    def _value_policy(self):
        # The inherited recorder deliberately inspects this exact frame/local.
        probs = np.full((4, 4), .25, dtype=np.float32)
        assert probs.shape == (4, 4)
        self.model(np.zeros((16, 105, 105), dtype=np.float32), sym_avg=True)
        return 0, np.arange(4, dtype=np.float32)


@pytest.fixture
def fabricated_collector(tmp_path):
    created = []

    def build(*, found_at=None, future_hit=1):
        out = tmp_path / str(len(created))
        out.mkdir()
        run = M.Collector.__new__(M.Collector)
        run.out, run.check = out, lambda: None
        run.ledger = M.Ledger(run)
        run.reference = REFERENCE
        run.view_class, run.analytic, run.features = ReadOnlyBeliefView, _analytic, _features
        kernel = np.full((4, 107, 107), .25, dtype=np.float64)
        kernel[:, 53, 53] = 0
        environments, updates, calls = [], [], []

        class Environment:
            N, Ndim, Nactions, Nhits = 53, 2, 4, 4

            def __init__(self, initial_hit):
                self.agent, self.obs, self.steps = [26, 26], {"hit": initial_hit, "done": False}, 0
                self.source, self.p_Poisson = np.array([40, 40]), kernel
                self.belief = PublicBeliefView(kernel)
                self.belief._reset(initial_hit)

            @property
            def p_source(self):
                return self.belief.p_source

            def _move(self, action, position):
                return _move(action, position)

            def step(self, action, *, quiet):
                assert quiet is True and not self.obs["done"]
                if self.steps >= 8:
                    committed = json.loads((out / "commitments.jsonl").read_text())
                    assert committed["actions"][self.steps - 8] == action
                    assert committed["commit_before_native_step"] == 8
                self.agent = _move(action, self.agent)[0]
                self.steps += 1
                done = self.steps == found_at
                hit = -2 if done else future_hit if self.steps > 8 else 1
                self.obs = {"hit": hit, "done": done}
                if done:
                    self.source = np.asarray(self.agent)
                self.belief._observe(REFERENCE.packet(public.observation(self, self.steps)))
                return hit, 0., done

        def seeded(_source, _seed, _config, *, initial_hit):
            env = Environment(initial_hit)
            environments.append(env)
            return env

        class Actor(ReleasedPolicyActor):
            def update(self, action, packet):
                assert self._pending_action is None
                super().update(action, packet)
                updates.append(packet.copy())

        def analytic(packet, likelihood, *, allow_stay):
            assert allow_stay is False
            return Actor(packet, likelihood, None, AnalyticPolicy, sym_avg=False)

        def fake_value(inputs, *, training, sym_avg):
            assert inputs.shape == (16, 105, 105) and training is False and sym_avg is True
            calls.append("fabricated value return")
            return SimpleNamespace(numpy=lambda: np.zeros((16, 1), np.float32))

        run.runtime = SimpleNamespace(np=np, source=object(), public=SimpleNamespace(
            seeded_environment=seeded, observation=public.observation), kernels={"base": kernel, "shift": kernel},
            analytic=analytic, policy=TeacherPolicy, model=fake_value)
        run.environments, run.updates, run.fake_value_calls = environments, updates, calls
        created.append(run)
        return run

    yield build
    for run in created:
        run.ledger.close()


@pytest.mark.parametrize("split,horizon,steps,features", (("train", 4, 12, 13), ("dev", 8, 16, 17)))
def test_complete_fake_case_has_nine_prefix_observations_and_exact_work(fabricated_collector, split, horizon, steps, features):
    run = fabricated_collector()
    identity = next(row for row in C.roster() if row["split"] == split)
    row, arrays = run.case(identity)
    assert row["excluded"] is None and row["native_steps"] == steps
    assert arrays["prefix"].shape == (9, 31) and arrays["prefix_lengths"] == 9
    assert arrays["actions"].shape == arrays["outcomes"].shape == (horizon,)
    assert arrays["continuation"].shape == (horizon, 31)
    assert len(run.updates) == steps and len(run.fake_value_calls) == horizon
    for channel in ("native_step", "actor_update"):
        assert run.ledger.calls[channel]["returned"] == steps
    for channel in ("analytic_score", "feature_build"):
        assert run.ledger.calls[channel]["returned"] == features
    assert run.ledger.calls["teacher_score"]["returned"] == run.ledger.calls["tensorflow_value"]["returned"] == horizon
    assert run.ledger.last_seconds["tensorflow_value"] >= 0
    assert not run.ledger.pending


def test_changed_future_odors_cannot_change_prefix_or_committed_actions(fabricated_collector):
    identity = next(row for row in C.roster() if row["split"] == "dev")
    first, second = fabricated_collector(future_hit=0), fabricated_collector(future_hit=3)
    _, a = first.case(identity)
    _, b = second.case(identity)
    assert a["prefix"].tobytes() == b["prefix"].tobytes()
    assert a["actions"].tobytes() == b["actions"].tobytes()
    assert not np.array_equal(a["continuation"], b["continuation"])
    np.testing.assert_array_equal(a["outcomes"], np.zeros(8, dtype=np.int64))
    np.testing.assert_array_equal(b["outcomes"], np.full(8, 3, dtype=np.int64))


@pytest.mark.parametrize("found_at", (1, 8))
def test_prefix_found_case_is_excluded_without_replacement_or_block(fabricated_collector, found_at):
    run = fabricated_collector(found_at=found_at)
    row, arrays = run.case(C.roster()[0])
    assert arrays is None and row["excluded"] == "found_during_observed_prefix"
    assert row["native_steps"] == len(run.updates) == found_at
    assert run.ledger.calls["native_reset"]["returned"] == 1
    assert run.ledger.calls["teacher_score"]["returned"] == run.ledger.calls["backend_binding"]["returned"] == 0
    assert not (run.out / "commitments.jsonl").exists()


@pytest.mark.parametrize("found_at", (9, 10, 12))
def test_terminal_step_assimilated_once_then_absorbing_without_teacher_or_native_calls(fabricated_collector, found_at):
    run = fabricated_collector(found_at=found_at)
    row, arrays = run.case(C.roster()[0])
    live = found_at - 9
    assert row["found_in_block"] is True and row["native_steps"] == found_at
    assert run.ledger.calls["native_step"]["returned"] == len(run.updates) == found_at
    assert run.updates[-1]["done"] and run.updates[-1]["hit"] == -2
    assert run.ledger.calls["teacher_score"]["returned"] == live
    np.testing.assert_array_equal(arrays["outcomes"][live:], np.full(4 - live, 4))
    assert not arrays["legal"][live:].any()
    assert arrays["raw_costs"][live:].tobytes() == np.zeros((4 - live, 4), np.float32).tobytes()
    assert arrays["continuation"][live:].tobytes() == np.zeros((4 - live, 31), np.float32).tobytes()


def test_ledger_caps_prevent_call_and_failure_keeps_pending_attempt(tmp_path):
    run = SimpleNamespace(out=tmp_path, check=lambda: None)
    ledger = M.Ledger(run)
    try:
        ledger.calls["native_reset"]["attempted"] = M.CALL_CAPS["native_reset"]
        invoked = []
        with pytest.raises(ValueError, match="call cap"):
            ledger.call("native_reset", lambda: invoked.append(True))
        assert not invoked

        def failed():
            raise RuntimeError("fabricated failure")

        with pytest.raises(RuntimeError, match="fabricated failure"):
            ledger.call("native_step", failed)
        assert ledger.calls["native_step"]["attempted"] == 1 and ledger.calls["native_step"]["returned"] == 0
        assert len(ledger.pending) == 1 and ledger.pending[0]["channel"] == "native_step"
        ledger.flush()
        with gzip.open(tmp_path / "work.jsonl.gz", "rt") as stream:
            first = json.loads(stream.readline())
        assert first["event"] == "attempt" and first["channel"] == "native_step"
    finally:
        ledger.close()


def test_setup_rejects_changed_native_sources_before_inherited_setup(tmp_path, monkeypatch):
    called = []
    prior = {"sources": {"inherited.py": "sha"}, "runtime": {}}
    reference = SimpleNamespace(Run=SimpleNamespace(setup=lambda _run: called.append("setup")))
    monkeypatch.setattr(C, "original_native", lambda: (prior, {}, reference, None))
    monkeypatch.setattr(C, "desc", lambda _path: {"sha256": "changed", "bytes": 1})
    run = M.Collector.__new__(M.Collector)
    run.out, run.plan = tmp_path, {"native_sources": {"inherited.py": {"sha256": "old", "bytes": 1}}}
    with pytest.raises(ValueError, match="native source pins"):
        run.setup()
    assert called == []


def test_fixed_plan_roster_and_every_source_and_input_are_authenticated(monkeypatch):
    pins = {name: {"sha256": name, "bytes": 1} for name in C.SOURCES}
    pins["seed-review.json"] = {"sha256": "review", "bytes": 2}
    monkeypatch.setattr(C, "desc", lambda path: pins[str(path)].copy())
    plan = {"version": C.VERSION, "config": copy.deepcopy(C.CONFIG), "roster": C.roster(),
            "sources": {name: pins[name].copy() for name in C.SOURCES}, "inputs": {"seed-review.json": pins["seed-review.json"].copy()}}
    C.check_plan(plan)
    changed = copy.deepcopy(plan)
    changed["roster"][0]["seed"] += 1
    with pytest.raises(ValueError, match="fixed protocol"):
        C.check_plan(changed)
    for section in ("sources", "inputs"):
        changed = copy.deepcopy(plan)
        changed[section][next(iter(changed[section]))]["sha256"] = "changed"
        with pytest.raises(ValueError, match="unchanged"):
            C.check_plan(changed)


@pytest.fixture
def closed_fixture(tmp_path):
    directory = tmp_path / "collect"
    directory.mkdir()
    launch_path, terminal_path = tmp_path / "launch.json", tmp_path / "terminal.json"
    start = 100
    launch = {"started_ns": start, "deadline_ns": start + C.CAPS["collect"] * 10**9,
              "cap_seconds": C.CAPS["collect"], "command": ["fake-native", "collector.py", "--supervision", str(launch_path)]}
    C.write(launch_path, launch)
    C.write(directory / "started.json", {"launch": launch, "plan_sha256": "registered"})
    receipt = {"version": C.VERSION, "phase": "collect", "status": "completed", "started_ns": 101,
               "finished_ns": 200, "plan_sha256": "registered", "supervision": C.desc(launch_path),
               "files": {"started.json": C.desc(directory / "started.json")}}
    C.write(directory / "receipt.json", receipt)
    terminal = {**launch, "finished_ns": 201, "status": "completed", "returncode": 0,
                "group_absent": True, "timed_out": False, "error": None, "clock_error": None,
                "timing_available": True, "cleanup": {"reaped": True, "group_absent": True, "errors": []}}
    C.write(terminal_path, terminal)
    return directory, terminal_path, terminal


def test_original_successful_fake_process_and_payload_closure(closed_fixture):
    directory, terminal, _ = closed_fixture
    assert C.closed(directory, terminal)["phase"] == "collect"
    (directory / "started.json").write_text('{}\n')
    with pytest.raises((ValueError, KeyError)):
        C.closed(directory, terminal)


@pytest.mark.parametrize("defect", ("clock", "timing", "cleanup", "unreaped", "deadline", "late", "chronology"))
def test_failed_original_process_cannot_close_even_with_exit_zero(closed_fixture, defect):
    directory, path, terminal = closed_fixture
    if defect == "clock":
        terminal["clock_error"] = "fabricated clock failure"
    elif defect == "timing":
        terminal["timing_available"] = False
    elif defect == "cleanup":
        terminal["cleanup"]["errors"] = ["fake process remains"]
    elif defect == "unreaped":
        terminal["cleanup"]["reaped"] = False
    elif defect == "deadline":
        terminal["deadline_ns"] += 1
    elif defect == "late":
        terminal["finished_ns"] = terminal["deadline_ns"] + 1
    else:
        terminal["finished_ns"] = 199
    path.write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match="original process closure"):
        C.closed(directory, path)
