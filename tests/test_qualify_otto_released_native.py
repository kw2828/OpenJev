"""Synthetic mechanics and lifecycle only; no OTTO or TensorFlow import/call."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research.otto_released_policy import ReleasedPolicyActor

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(f'fixture_{name}', ROOT / f'scripts/{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


M = script('qualify_otto_released_native')
HELPER = script('qualify_otto_pretrained')


def test_frozen_paths_cover_all_blocks_sources_prefixes_and_exact_counts():
    cases = M.fixtures()
    assert [r['seed_evaluation_only'] for r in cases] == list(range(840001, 840009))
    assert len(cases) == 8 and sum(len(r['actions']) for r in cases) == 446
    assert sum(r['prefix_step'] for r in cases) == 64
    for row in cases:
        pos, blocked, found = [26, 26], [], []
        for step, action in enumerate(row['actions'], 1):
            before = list(pos)
            axis, delta = action // 2, 2 * (action % 2) - 1
            pos[axis] = min(52, max(0, pos[axis] + delta))
            if pos == before:
                blocked.append(step)
            if pos == row['source_evaluation_only']:
                found.append(step)
        if row['censored']:
            assert found == blocked == [] and pos == [26, 26]
            assert row['hits'] == [0, 1, 2, 3]
        else:
            assert found == [211] and blocked == [27, 54, 107, 160]
            assert [row['actions'][i - 1] for i in blocked] == [0, 2, 1, 3]


def test_array_comparison_is_exact_including_dtype_signed_zero_and_nonfinite():
    a = np.zeros((2, 2), dtype=np.float64)
    assert M.array_identity(a, a.copy(), np)
    b = a.copy()
    b[0, 0] = -0.
    assert not M.array_identity(a, b, np)
    assert not M.array_identity(a, a.astype(np.float32), np)
    assert not M.array_identity(a, a.reshape(4), np)
    assert not M.array_identity(np.array([np.nan]), np.array([np.nan]), np)


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


class FakeModel:
    def __call__(self, inputs, *, training, sym_avg):
        assert training is False and sym_avg is True
        return FakeTensor(np.full((16, 1), 3., dtype=np.float32))


class FakePolicy:
    def __init__(self, env, model, sym_avg):
        self.env, self.model, self.sym_avg = env, model, sym_avg

    def _value_policy(self):
        probs = np.full((4, 4), .25, dtype=np.float32)
        inputs = np.zeros((16, 105, 105), dtype=np.float32)
        x, y = 52 - self.env.agent[0], 52 - self.env.agent[1]
        inputs[:, x:x + 53, y:y + 53] = self.env.p_source
        values = self.model(FakeTensor(inputs), sym_avg=self.sym_avg).numpy().reshape(4, 4)
        costs = np.asarray(1 + (probs * values).sum(axis=1), dtype=np.float32)
        return int(np.argmin(costs)), costs


class FakeEnvironment:
    N, Nhits, Nactions = 53, 4, 4

    def __init__(self, initial_hit, kernel):
        self.agent, self.source = [26, 26], np.array([52, 52])
        self.p_Poisson, self.obs = kernel, {'hit': initial_hit, 'done': False}
        self.p_source = np.ones((53, 53), dtype=np.float64) / 2808
        self.p_source[26, 26] = 0
        self.assimilate(initial_hit, False)

    def assimilate(self, hit, done):
        position = tuple(self.agent)
        if done:
            self.p_source = np.zeros((53, 53), dtype=np.float64)
            self.p_source[position] = 1
        else:
            self.p_source[position] = 0
            x, y = 53 - position[0], 53 - position[1]
            self.p_source *= self.p_Poisson[hit, x:x + 53, y:y + 53]
            if self.p_source.sum() > 1e-10:
                self.p_source /= self.p_source.sum()

    def step(self, action, *, hit, quiet):
        assert not self.obs['done'] and quiet is True
        axis, delta = action // 2, 2 * (action % 2) - 1
        self.agent[axis] = min(52, max(0, self.agent[axis] + delta))
        done = self.agent == self.source.tolist()
        hit = -2 if done else hit
        self.obs = {'hit': hit, 'done': done}
        self.assimilate(hit, done)
        return hit, int(done), done


def fake_observation(env, step):
    valid = tuple(a for a in range(4) if 0 <= env.agent[a // 2] + 2 * (a % 2) - 1 < 53)
    return SimpleNamespace(position=tuple(env.agent), hit=env.obs['hit'], done=env.obs['done'],
                           step=step, valid_actions=() if env.obs['done'] else valid)


def test_complete_runner_accounting_with_fake_environment_and_fake_model(tmp_path):
    kernel = np.full((4, 107, 107), .25, dtype=np.float64)
    kernel[:, 53, 53] = 0
    ledger = M.Ledger(tmp_path, lambda: None)
    seen = []

    def seeded(source_class, seed, config, *, initial_hit):
        assert source_class is FakeEnvironment
        assert config['lambda_over_dx'] in (3., 4.)
        seen.append(seed)
        return FakeEnvironment(initial_hit, kernel)

    result = M.mechanical_run(FakeEnvironment, seeded, fake_observation, ReleasedPolicyActor,
                              FakePolicy, FakeModel(), HELPER, {'base': kernel, 'shift': kernel}, ledger, tmp_path, np)
    assert result['qualified'] is True and result['numpy_port_admitted'] is False
    assert result['prior_numpy_action_failure_preserved'] is True and result['autonomous_episodes'] == 0
    assert seen == list(range(840001, 840009))
    assert ledger.state['calls']['native_step'] == {'attempted': 446, 'returned': 446}
    assert ledger.state['calls']['tensorflow_value'] == {'attempted': 16, 'returned': 16}
    assert ledger.state['calls']['actor_update'] == {'attempted': 510, 'returned': 510}
    assert len(list(tmp_path.glob('prefix-*.npz'))) == 8
    transitions = [json.loads(line) for line in (tmp_path / 'transitions.jsonl').read_text().splitlines()]
    assert len(transitions) == 454
    assert sum(r['kind'] == 'step' and r['public']['done'] for r in transitions) == 2
    assert all(r['final_update_assimilated'] for r in result['cases'])
    assert sum(r['censored'] for r in result['cases']) == 6


def test_ledger_journals_attempt_before_call_and_preserves_pending_on_failure(tmp_path):
    ledger = M.Ledger(tmp_path, lambda: None)
    ledger.context('mechanical', 'fake-case')

    def bad_call():
        row = json.loads((tmp_path / 'work.jsonl').read_text().splitlines()[-1])
        assert row['event'] == 'attempt' and row['calls']['native_step'] == {'attempted': 1, 'returned': 0}
        raise RuntimeError('synthetic native failure')

    with pytest.raises(RuntimeError, match='synthetic native failure'):
        ledger.call('native_step', bad_call)
    assert ledger.state['pending_call']['channel'] == 'native_step'
    with pytest.raises(ValueError, match='uncertain'):
        ledger.context('another')


def test_regressed_post_found_guard_cannot_hide_a_model_forward(tmp_path):
    class LateGuardActor(ReleasedPolicyActor):
        def choose(self):
            if self.public['done']:
                self._policy._value_policy()
                raise RuntimeError('incorrect guard after the forward')
            return super().choose()

    kernel = np.full((4, 107, 107), .25, dtype=np.float64)
    kernel[:, 53, 53] = 0
    ledger = M.Ledger(tmp_path, lambda: None)

    def seeded(_source_class, _seed, _config, *, initial_hit):
        return FakeEnvironment(initial_hit, kernel)

    with pytest.raises(ValueError, match='post-found guards invoke no work'):
        M.mechanical_run(FakeEnvironment, seeded, fake_observation, LateGuardActor,
                         FakePolicy, FakeModel(), HELPER, {'base': kernel, 'shift': kernel}, ledger, tmp_path, np)
    assert ledger.state['calls']['native_step'] == {'attempted': 223, 'returned': 223}
    # Four intended paired prefixes, plus the accidental primary-actor call.
    assert ledger.state['calls']['tensorflow_value'] == {'attempted': 9, 'returned': 9}
    journal = [json.loads(line) for line in (tmp_path / 'work.jsonl').read_text().splitlines()]
    assert any(row['event'] == 'attempt' and row['pending_call']['channel'] == 'tensorflow_value'
               and row['pending_call']['ordinal'] == 9 for row in journal)


def test_bad_external_plan_pin_fails_before_inputs_or_runtime_inspection(tmp_path, monkeypatch):
    plan = tmp_path / 'plan.json'
    plan.write_text('{}')
    monkeypatch.setattr(M.importlib.metadata, 'distributions', lambda: pytest.fail('runtime inspected before plan pin'))
    with pytest.raises(ValueError, match='external native plan pin'):
        M.authenticate(SimpleNamespace(plan=plan, plan_sha256='0' * 64))


def test_existing_output_is_never_modified(tmp_path, monkeypatch):
    out = tmp_path / 'existing'
    out.mkdir()
    (out / 'marker').write_text('unchanged')
    run = M.Run(argparse.Namespace(output=out))
    monkeypatch.setattr(run, 'bind', lambda: pytest.fail('existing output bound'))
    with pytest.raises(FileExistsError):
        run.execute()
    assert (out / 'marker').read_text() == 'unchanged'


def test_failure_receipt_preserves_pending_work_and_primary_error(tmp_path, monkeypatch):
    run = M.Run(argparse.Namespace(output=tmp_path / 'failed'))
    monkeypatch.setattr(run.ledger, 'check', lambda: None)

    def fail_bind():
        run.ledger.context('model_load')

        def failure():
            raise RuntimeError('fake weight load failure')
        run.ledger.call('tensorflow_load', failure)

    monkeypatch.setattr(run, 'bind', fail_bind)
    with pytest.raises(RuntimeError, match='fake weight load failure'):
        run.execute()
    receipt = json.loads((run.out / 'receipt.json').read_text())
    assert receipt['status'] == 'failed' and receipt['qualified'] is False
    assert receipt['work']['calls']['tensorflow_load'] == {'attempted': 1, 'returned': 0}
    assert receipt['work']['pending_call']['channel'] == 'tensorflow_load'
