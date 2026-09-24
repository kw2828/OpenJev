"""Fabricated fresh DEV-only collector checks; no native backend or data."""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research import otto_predictive_belief as bayes
from openjev.research import otto_public as public
from openjev.research import otto_sampler_law as sampler
from openjev.research.otto_query_gate import ReadOnlyBeliefView, _analytic, _features
from openjev.research.otto_released_policy import PublicBeliefView, ReleasedPolicyActor, _move

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


C = sys.modules.get("otto_action_effect_common") or module(
    "scripts/otto_action_effect_common.py", "otto_action_effect_common")
M = module("scripts/collect_otto_action_effect.py", "action_effect_collector_tests")
REFERENCE = module("scripts/study_otto_released_reference.py", "action_effect_reference_tests")


class FabricatedSensor:
    def __init__(self, *args, **kwargs):
        raise AssertionError("no environment may be constructed")

    def _set_mu0_Poisson(self):
        assert self.Ndim == 2 and self.Nhits == 4 and self.R_dt == 2.
        self.mu0_Poisson = 7.

    def _mean_number_of_hits(self, distance):
        assert distance > 0 and self.mu0_Poisson == 7.
        assert not hasattr(self, "source") and not hasattr(self, "agent")
        return distance

    def _Poisson(self, mu, hit):
        return self._Poisson_unbounded(mu, hit)

    def _Poisson_unbounded(self, mu, hit):
        assert mu > 0 and hit in (0, 1, 2)
        return (.125, .25, .375)[hit]


def test_scalar_sensor_table_uses_no_constructor_and_counts_every_primitive():
    checks = []
    built = sampler.build_sensor_law(FabricatedSensor, 3., check=lambda: checks.append(True))
    assert len(checks) == 106
    assert built["probabilities"].shape == built["raw_probabilities"].shape == (105, 105, 4)
    expected = np.broadcast_to([.125, .25, .375, .25], (105, 105, 4)).copy()
    expected[52, 52] = 0
    np.testing.assert_array_equal(built["probabilities"], expected)
    np.testing.assert_array_equal(built["raw_probabilities"], expected)
    assert not np.shares_memory(built["probabilities"], built["raw_probabilities"])
    assert built["metadata"]["work"] == {"mu0_calls": 1, "distance_calls": 11024,
        "mean_calls": 11024, "poisson_calls": 66144, "poisson_unbounded_calls": 66144, "cdf_calls": 11024}
    assert built["metadata"]["uniform_bits"] == 53


def test_lookup_row_major_geometry_owned_output_and_committed_opposites():
    table = np.zeros((105, 105, 4), np.float64)
    table[..., 0] = np.arange(105)[:, None] / 104
    table[..., 1] = np.arange(105)[None, :] / 104
    q = np.array([9, 40], np.int64)
    result = sampler.likelihood_at(table, q)
    for source in ((0, 0), (9, 40), (52, 52)):
        np.testing.assert_array_equal(result[source[0] * 53 + source[1]], table[source[0] - q[0] + 52, source[1] - q[1] + 52])
    assert not np.shares_memory(result, table)
    actions = np.array([0, 3, 0, 3], np.int64)
    original = sampler.planned_positions(q, actions)
    opposite = sampler.planned_positions(q, actions ^ 1)
    np.testing.assert_array_equal(original + opposite, np.broadcast_to(2 * q, original.shape))
    np.testing.assert_array_equal(q, [9, 40])
    with pytest.raises(ValueError, match="in bounds"):
        sampler.planned_positions(np.array([0, 0]), np.array([0, 0, 0, 0], np.int64))
    with pytest.raises(ValueError):
        sampler.likelihood_at(table, np.array([53, 0]))


def test_roster_caps_and_random_action_stream_are_fixed_before_outcomes():
    roster = C.roster()
    assert len(roster) == len({r["seed"] for r in roster}) == len({r["id"] for r in roster}) == 256
    assert {r["split"] for r in roster} == {"dev"}
    for regime in ("lambda3", "lambda4"):
        assert sum(r["split"] == "dev" and r["regime"] == regime for r in roster) == 128
    assert {r['seed'] for r in roster if r['regime'] == 'lambda3'} == set(range(328000001, 328000129))
    assert {r['seed'] for r in roster if r['regime'] == 'lambda4'} == set(range(329000001, 329000129))
    block = 256 * 8
    assert M.CALL_CAPS["native_step"] == 256 * 8 + block == 4096
    assert M.CALL_CAPS["feature_build"] == 256 * 9 + block == 4352
    assert M.CALL_CAPS["teacher_score"] == M.CALL_CAPS["normal_oracle"] == block == 2048
    assert M.CALL_CAPS["sampler_lookup"] == 256 * 8 + block * 2 == 6144
    assert M.CALL_CAPS["committed_positions"] == 256 * 2
    seed = roster[0]["seed"]
    np.testing.assert_array_equal(M.committed_actions(np, seed, 8), M.committed_actions(np, seed, 8))
    with pytest.raises(ValueError):
        M.committed_actions(np, seed, 4)
    with pytest.raises(ValueError):
        M.committed_actions(np, True, 4)


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
        probs = np.full((4, 4), .25, np.float32)
        assert probs.shape == (4, 4)
        self.model(np.zeros((16, 105, 105), np.float32), sym_avg=True)
        return 0, np.arange(4, dtype=np.float32)


@pytest.fixture
def fabricated_collector(tmp_path, monkeypatch):
    created = []
    monkeypatch.setattr(M, "committed_actions", lambda _np, _seed, horizon: np.full(horizon, 2, np.int64))

    def build(*, found_at=None, future_hit=1):
        out = tmp_path / str(len(created))
        out.mkdir()
        run = M.Collector.__new__(M.Collector)
        run.out, run.check, run.reference = out, lambda: None, REFERENCE
        run.ledger = M.Ledger(run)
        run.view_class, run.analytic, run.features = ReadOnlyBeliefView, _analytic, _features
        run.sampler, run.bayes = sampler, bayes
        kernel = np.full((4, 107, 107), .25, np.float64)
        kernel[:, 53, 53] = 0
        table = np.full((105, 105, 4), .25, np.float64)
        table[52, 52] = 0
        run.sensor_laws = {regime: table.copy() for regime in ("lambda3", "lambda4")}
        run.sensor_raw = {regime: table.copy() for regime in ("lambda3", "lambda4")}
        environments, updates, calls = [], [], []

        class Environment:
            N, Ndim, Nactions, Nhits = 53, 2, 4, 4

            def __init__(self, initial_hit):
                self.agent, self.obs, self.steps = [26, 26], {"hit": initial_hit, "done": False}, 0
                self.source, self.p_Poisson = np.array([40, 40]), kernel
                self.belief = PublicBeliefView(kernel)
                self.belief._reset(initial_hit)
                initial = self.p_source.reshape(-1)
                self.draw_log = [{"channel": "source", "probabilities": initial.tolist(),
                                  "cdf_mass": float(np.cumsum(initial)[-1]), "selected_index": 2160}]

            @property
            def p_source(self):
                return self.belief.p_source

            def _move(self, action, position):
                return _move(action, position)

            def step(self, action, *, quiet):
                assert quiet and not self.obs["done"]
                if self.steps >= 8:
                    committed = json.loads((out / "commitments.jsonl").read_text())
                    assert committed["actions"][self.steps - 8] == action
                self.agent = _move(action, self.agent)[0]
                self.steps += 1
                done = self.steps == found_at
                hit = -2 if done else future_hit if self.steps > 8 else 1
                self.obs = {"hit": hit, "done": done}
                if done:
                    self.source = np.asarray(self.agent)
                else:
                    self.draw_log.append({"channel": "hit", "probabilities": [.25] * 4,
                                          "cdf_mass": 1., "selected_index": hit})
                self.belief._observe(REFERENCE.packet(public.observation(self, self.steps)))
                return hit, 0., done

        def seeded(_source, _seed, _config, *, initial_hit):
            env = Environment(initial_hit)
            environments.append(env)
            return env

        class Actor(ReleasedPolicyActor):
            def update(self, action, packet):
                super().update(action, packet)
                updates.append(packet.copy())

        def analytic(packet, likelihood, *, allow_stay):
            assert not allow_stay
            return Actor(packet, likelihood, None, AnalyticPolicy, sym_avg=False)

        def fake_value(inputs, *, training, sym_avg):
            assert inputs.shape == (16, 105, 105) and training is False and sym_avg is True
            calls.append(True)
            return SimpleNamespace(numpy=lambda: np.zeros((16, 1), np.float32))

        run.runtime = SimpleNamespace(np=np, source=object(), public=SimpleNamespace(
            seeded_environment=seeded, observation=public.observation), kernels={"base": kernel, "shift": kernel},
            analytic=analytic, policy=TeacherPolicy, model=fake_value)
        run.environments, run.updates, run.fake_calls = environments, updates, calls
        created.append(run)
        return run

    yield build
    for run in created:
        run.ledger.close()


@pytest.mark.parametrize("regime", ["lambda3", "lambda4"])
def test_fake_case_keeps_reconstructable_prefix_and_exact_oracle_work(fabricated_collector, regime):
    run = fabricated_collector()
    horizon = 8
    identity = next(r for r in C.roster() if r["regime"] == regime)
    row, arrays = run.case(identity)
    assert row["excluded"] is None and row["native_steps"] == 8 + horizon
    assert len(arrays) == 14  # plus case_ids and regimes in the saved aggregate
    assert arrays["initial_belief"].shape == (2809,) and arrays["initial_belief"].dtype == np.float64
    assert arrays["prefix_actions"].shape == arrays["prefix_outcomes"].shape == (8,)
    np.testing.assert_array_equal(arrays["prefix_position"], [18, 26])
    for name in ("gap_oracle", "normal_oracle", "opposite_oracle"):
        assert arrays[name].shape == (horizon, 5)
        np.testing.assert_allclose(arrays[name].sum(-1), 1., rtol=0, atol=1e-12)
    np.testing.assert_array_equal(arrays["gap_oracle"][0], arrays["normal_oracle"][0])
    assert run.ledger.calls["shadow_update"]["returned"] == 8 + horizon
    assert run.ledger.calls["sampler_lookup"]["returned"] == 8 + 2 * horizon
    assert run.ledger.calls["normal_oracle"]["returned"] == horizon
    assert run.ledger.calls["gap_oracle"]["returned"] == run.ledger.calls["opposite_oracle"]["returned"] == 1
    assert len(run.fake_calls) == horizon and not run.ledger.pending
    run.ledger.close()
    with gzip.open(run.out / "shadow.jsonl.gz", "rt") as stream:
        shadows = [json.loads(line) for line in stream]
    assert len(shadows) == 9 + horizon and all("shadow_sha256" in r for r in shadows)
    assert not any("belief" in r for r in shadows)


def test_future_realizations_do_not_change_gap_opposite_or_initial_inputs(fabricated_collector):
    identity = next(r for r in C.roster() if r["split"] == "dev")
    a, b = fabricated_collector(future_hit=0), fabricated_collector(future_hit=3)
    _, first = a.case(identity)
    _, second = b.case(identity)
    for name in ("prefix", "actions", "initial_belief", "prefix_actions", "prefix_outcomes", "prefix_position",
                 "gap_oracle", "opposite_oracle"):
        np.testing.assert_array_equal(first[name], second[name])
    assert not np.array_equal(first["continuation"], second["continuation"])


@pytest.mark.parametrize("found_at", [1, 8])
def test_prefix_terminal_exclusion_has_no_replacement_or_future_annotation(fabricated_collector, found_at):
    run = fabricated_collector(found_at=found_at)
    row, arrays = run.case(C.roster()[0])
    assert arrays is None and row["excluded"] == "found_during_observed_prefix"
    assert row["native_steps"] == run.ledger.calls["shadow_update"]["returned"] == found_at
    assert not (run.out / "commitments.jsonl").exists()
    assert run.ledger.calls["gap_oracle"]["returned"] == run.ledger.calls["teacher_score"]["returned"] == 0


@pytest.mark.parametrize("found_at", [9, 10, 12])
def test_known_terminal_suffix_is_onehot_but_first_found_is_forecast(fabricated_collector, found_at):
    run = fabricated_collector(found_at=found_at)
    row, arrays = run.case(C.roster()[0])
    first_found = found_at - 9
    assert row["found_in_block"] and row["native_steps"] == found_at
    assert 0 < arrays["normal_oracle"][first_found, 4] < 1
    suffix = arrays["normal_oracle"][first_found + 1:]
    np.testing.assert_array_equal(suffix, np.broadcast_to([0., 0., 0., 0., 1.], suffix.shape))
    assert (arrays["gap_oracle"][:, 4] < 1).all()
    assert run.ledger.calls["normal_oracle"]["returned"] == 8
    assert run.ledger.calls["shadow_update"]["returned"] == found_at
    assert run.ledger.calls["teacher_score"]["returned"] == first_found


def test_draw_mismatch_is_technical_failure_and_preserves_pending_work(fabricated_collector):
    run = fabricated_collector()
    run.sensor_raw["lambda3"][...] = [.5, .25, .125, .125]
    with pytest.raises(ValueError, match="sampler probabilities"):
        run.case(C.roster()[0])
    assert run.ledger.pending[-1]["channel"] == "sampler_draw_validation"
    assert run.ledger.calls["sampler_draw_validation"]["attempted"] == 2
    assert run.ledger.calls["sampler_draw_validation"]["returned"] == 1


def test_resource_caps_fail_before_invocation(tmp_path):
    run = SimpleNamespace(out=tmp_path, check=lambda: None)
    ledger = M.Ledger(run)
    called = []
    try:
        ledger.calls["normal_oracle"]["attempted"] = M.CALL_CAPS["normal_oracle"]
        with pytest.raises(ValueError, match="call cap"):
            ledger.call("normal_oracle", lambda: called.append(True))
        assert called == []
    finally:
        ledger.close()


@pytest.mark.parametrize('defect', ['train', 'unknown_regime', 'old_seed', 'changed_hit'])
def test_nonregistered_identity_fails_before_runtime_or_native_reset(defect):
    identity = dict(C.roster()[0])
    if defect == 'train':
        identity['split'] = 'train'
    elif defect == 'unknown_regime':
        identity['regime'] = 'lambda5'
    elif defect == 'old_seed':
        identity['seed'] = 324000001
    else:
        identity['initial_hit'] = identity['initial_hit'] % 3 + 1
    run = M.Collector.__new__(M.Collector)
    # No runtime or ledger exists: rejection must precede their first access.
    with pytest.raises(ValueError, match='registered fresh DEV identity'):
        run.case(identity)


def test_training_lineage_failure_precedes_native_setup(tmp_path, monkeypatch):
    called = []

    def rejected():
        raise ValueError('fabricated changed TRAIN descriptor')

    monkeypatch.setattr(C, 'training_reference', rejected)
    monkeypatch.setattr(C, 'original_native', lambda: called.append(True))
    run = M.Collector.__new__(M.Collector)
    run.out = tmp_path
    with pytest.raises(ValueError, match='TRAIN descriptor'):
        run.setup()
    assert not called and not list(tmp_path.iterdir())


def test_body_writes_only_dev_archive_and_training_linkage_without_array_read(fabricated_collector, monkeypatch):
    run = fabricated_collector()
    reference = {'train': {'path': 'opaque-old-train.npz', 'sha256': 'a' * 64, 'bytes': 123},
                 'parent_closure': {'sha256': 'b' * 64, 'bytes': 45},
                 'teacher_cost_linkage': 'same original released four-action teacher', 'array_decodes': 0}
    run.training_reference, run.receipt = reference, {}
    run.setup = lambda: None
    oracle = np.full((8, 5), .25, np.float64)
    oracle[:, 4] = 0
    arrays = {'prefix': np.zeros((9, 31), np.float32), 'prefix_lengths': np.int64(9),
              'actions': np.zeros(8, np.int64), 'continuation': np.zeros((8, 31), np.float32),
              'outcomes': np.zeros(8, np.int64), 'raw_costs': np.zeros((8, 4), np.float32),
              'legal': np.ones((8, 4), np.bool_), 'initial_belief': np.full(2809, 1 / 2809, np.float64),
              'prefix_actions': np.zeros(8, np.int64), 'prefix_outcomes': np.zeros(8, np.int64),
              'prefix_position': np.array([18, 26], np.int64), 'gap_oracle': oracle.copy(),
              'normal_oracle': oracle.copy(), 'opposite_oracle': oracle.copy()}

    def fabricated_case(identity):
        return {'identity': identity, 'excluded': None, 'native_steps': 16}, arrays

    run.case = fabricated_case
    with monkeypatch.context() as context:
        context.setattr(np, 'load', lambda *a, **k: pytest.fail('collector must not decode any prior array'))
        run.body()
    assert not (run.out / 'train.npz').exists()
    with np.load(run.out / 'dev.npz', allow_pickle=False) as archive:
        assert set(archive.files) == set(arrays) | {'case_ids', 'regimes'}
        assert archive['prefix'].shape == (256, 9, 31)
        assert set(archive['regimes']) == {'lambda3', 'lambda4'}
    summary = json.loads((run.out / 'summary.json').read_text())
    assert summary['counts'] == {'dev': {'lambda3': 128, 'lambda4': 128}}
    assert summary['training_reference'] == reference
    assert summary['train_array_decodes'] == summary['new_train_cases'] == 0
