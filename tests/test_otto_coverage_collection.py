"""Fake collection boundaries only; no scientific arrays, OTTO or model calls."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research import otto_coverage_selection as selection
from openjev.research.otto_public import observation

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'coverage_collection_test', ROOT / 'scripts/collect_otto_coverage.py')
S = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = S
SPEC.loader.exec_module(S)
PUBLIC_KEYS = {'position', 'hit', 'done', 'step', 'valid_actions'}


def forbidden(*_args, **_kwargs):
    raise AssertionError('unexpected data, analytic chooser or model operation')


def posterior(step):
    result = np.zeros((53, 53), dtype=np.float64)
    result[step, 0] = 1.
    return result


class FakeEnvironment:
    N, Ndim, Nhits, Nactions = 53, 2, 4, 4

    def __init__(self, found_after):
        self.found_after = found_after
        self.agent, self.steps = [26, 26], 0
        self.obs = {'hit': 1, 'done': False}
        self.source = np.asarray([7, 9])  # Evaluator-only marker, never an actor input.
        self.p_source = posterior(0)
        self.draw_log = []

    def _move(self, action, position):
        result = list(position)
        result[action // 2] += -1 if action % 2 == 0 else 1
        valid = all(0 <= v < self.N for v in result)
        return (result if valid else list(position)), valid

    def step(self, action, *, quiet):
        assert quiet and not self.obs['done']
        self.agent, valid = self._move(action, self.agent)
        assert valid
        self.steps += 1
        done = self.steps == self.found_after
        self.obs = {'hit': -2 if done else self.steps % 4, 'done': done}
        self.p_source = posterior(self.steps)
        return self.obs['hit'], .125, done


class FakeActor:
    def __init__(self, public, kernel, *, allow_stay):
        assert set(public) == PUBLIC_KEYS and not allow_stay
        assert isinstance(public['position'], tuple) and isinstance(public['valid_actions'], tuple)
        self._public, self._belief = dict(public), posterior(0)
        self.updates = []

    @property
    def public(self):
        return dict(self._public)

    @property
    def belief(self):
        return self._belief.copy()

    @property
    def _policy(self):
        forbidden()

    def update(self, action, public):
        assert set(public) == PUBLIC_KEYS
        assert public['step'] == self._public['step'] + 1
        self.updates.append((action, dict(public)))
        self._public, self._belief = dict(public), posterior(public['step'])


class RecordingReservoir:
    """Snapshot only chosen arrivals, before a fake actor advances."""

    def __init__(self, wanted=None):
        self.wanted, self.identities, self.saved = wanted, [], []

    def offer(self, identity, payload_factory):
        index = len(self.identities)
        self.identities.append(identity)
        if self.wanted is not None and index not in self.wanted:
            return False
        self.saved.append(payload_factory())
        return True


@pytest.fixture
def fake_episode(monkeypatch, tmp_path):
    created = []

    def make(*, found_after=None, horizon=3, reservoir=None, forward_error=None):
        monkeypatch.setattr(S, 'HORIZON', horizon)
        out = tmp_path / f'episode-{len(created)}'
        out.mkdir()
        run = S.Run(SimpleNamespace(output=out))
        run.plan = {'limits': dict(S.LIMITS)}
        run.np, run.observation, run.selector = np, observation, selection
        run.check = lambda: None
        env = FakeEnvironment(found_after)
        actors, forwards = [], []

        def actor(public, kernel, *, allow_stay):
            result = FakeActor(public, kernel, allow_stay=allow_stay)
            actors.append(result)
            return result

        def environment(regime, seed, hit):
            assert (regime, seed, hit) == ('lambda3', 13100001, 1)
            return run.call('native_reset', lambda: env)

        def branch(belief, position, kernel, valid):
            np.testing.assert_array_equal(belief, posterior(env.steps))
            assert tuple(position) == tuple(env.agent)
            return SimpleNamespace(eligible_actions=tuple(valid), raw_masses=np.ones((4, 4)),
                                   weights=np.ones((4, 4)), position=tuple(position))

        def scores(branches, callback, *, arithmetic):
            assert arithmetic == 'float64'
            callback(np.zeros((16, 1, 1)), np.tile(branches.position, (16, 1)), None)
            return np.asarray([0., 1., 2., 3.])

        def physical(head, z, positions, sensing_length):
            assert head is run.heads[10101] and sensing_length == 3.
            assert z.shape == (16, 1, 1) and positions.shape == (16, 2)
            forwards.append(env.steps)
            if forward_error is not None:
                raise forward_error
            return np.arange(16, dtype=np.float64)

        run.actor_class, run.environment, run.physical = actor, environment, physical
        run.branches = SimpleNamespace(rl_branches=branch, explicit_scores=scores,
                                       select_action=lambda costs, valid: min(valid, key=lambda a: costs[a]))
        run.heads, run.kernels = {10101: object()}, {'lambda3': object()}
        run.available = Counter()
        reservoir = reservoir if reservoir is not None else RecordingReservoir()
        run.reservoirs = {('lambda3', 1, 10101): reservoir}
        created.append(run)
        return run, env, actors, forwards, reservoir

    yield make
    for run in created:
        for handle in run.handles.values():
            handle.close()


@pytest.mark.parametrize(('found_after', 'steps'), [(2, 2), (None, 3)])
def test_only_preaction_prefixes_and_final_found_or_censor_update(fake_episode, found_after, steps):
    run, env, actors, forwards, reservoir = fake_episode(found_after=found_after)
    result = run.episode('lambda3', 13100001, 1, 10101, 0)
    assert result['steps'] == steps and result['found'] is (found_after is not None)
    assert forwards == list(range(steps))
    assert len(actors) == 1 and len(actors[0].updates) == steps
    assert [v['metadata']['prefix_index'] for v in reservoir.saved] == list(range(steps))
    assert all(not v['metadata']['public']['done'] for v in reservoir.saved)
    assert len(reservoir.identities) == steps and len(set(reservoir.identities)) == steps
    for prefix, saved in enumerate(reservoir.saved):
        np.testing.assert_array_equal(saved['belief'], posterior(prefix))
        assert saved['metadata']['public']['step'] == prefix
        assert saved['metadata']['public']['position'] == (26 - prefix, 26)
        assert saved['metadata']['posterior']['sha256'] == hashlib.sha256(posterior(prefix).tobytes()).hexdigest()
        assert not {'target', 'return', 'total_steps', 'source_evaluation_only'} & saved['metadata'].keys()
    assert result['updates'] == steps and result['final_update_assimilated']
    assert result['training_labels_generated'] is False
    assert result['final_public']['step'] == steps
    assert tuple(result['final_public']['valid_actions']) == (() if found_after else (0, 1, 2, 3))
    np.testing.assert_array_equal(actors[0].belief, env.p_source)
    assert run.calls['native_step']['attempted'] == run.calls['native_step']['returned'] == steps
    assert run.calls['value_forward']['attempted'] == run.calls['value_forward']['returned'] == steps
    assert run.pending == []
    records = [json.loads(line) for line in (run.out / 'collection-transitions.jsonl').read_text().splitlines()]
    assert [r['kind'] for r in records] == ['reset'] + ['step'] * steps
    assert records[-1]['posterior_after']['sha256'] == hashlib.sha256(posterior(steps).tobytes()).hexdigest()


def test_lazy_selected_snapshot_survives_later_actor_updates(fake_episode):
    reservoir = RecordingReservoir(wanted={0})
    run, _, actors, _, _ = fake_episode(reservoir=reservoir)
    run.episode('lambda3', 13100001, 1, 10101, 0)
    assert len(reservoir.identities) == 3 and len(reservoir.saved) == 1
    saved = reservoir.saved[0]
    actors[0]._belief[:] = 42.
    actors[0]._public['position'] = (52, 52)
    np.testing.assert_array_equal(saved['belief'], posterior(0))
    assert saved['metadata']['public']['position'] == (26, 26)
    assert saved['metadata']['public']['valid_actions'] == (0, 1, 2, 3)


def test_failed_forward_keeps_preaction_and_pending_without_native_step(fake_episode):
    error = RuntimeError('synthetic readout failure')
    run, env, actors, forwards, reservoir = fake_episode(forward_error=error)
    with pytest.raises(RuntimeError) as caught:
        run.episode('lambda3', 13100001, 1, 10101, 0)
    assert caught.value is error and forwards == [0]
    assert env.steps == 0 and actors[0].updates == []
    assert len(reservoir.saved) == 1
    assert 'native_step' not in run.calls
    assert run.calls['value_forward']['attempted'] == 1 and run.calls['value_forward']['returned'] == 0
    assert [p['channel'] for p in run.pending] == ['value_forward']
    work = [json.loads(line) for line in (run.out / 'work.jsonl').read_text().splitlines()]
    assert [r[1] for r in work if r[2] == 'value_forward'] == [0]


def test_fixed_72_rotated_paired_cases_and_balanced_hit_cells():
    rows = list(S.collection_order())
    assert len(rows) == len(set(rows)) == 72
    assert len({(r, seed) for r, seed, _, _, _ in rows}) == 24
    for regime, first in (('lambda3', 13100001), ('lambda4', 13200001)):
        for case in range(12):
            block = [r for r in rows if r[0] == regime and r[4] == case]
            assert {r[1] for r in block} == {first + case}
            assert {r[2] for r in block} == {1 + case % 3}
            expected = (10101, 10102, 10103)
            shift = case % 3
            assert tuple(r[3] for r in block) == expected[shift:] + expected[:shift]
    cells = Counter((r, hit, collector) for r, _, hit, collector, _ in rows)
    assert len(cells) == 18 and set(cells.values()) == {4}


def test_external_plan_pin_rejects_before_decoding(monkeypatch, tmp_path):
    plan = tmp_path / 'bad-plan.json'
    plan.write_text('{}')
    monkeypatch.setattr(S, 'read', forbidden)
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(ValueError, match='external plan pin'):
        S.authenticate(SimpleNamespace(plan=plan, plan_sha256='0' * 64))


@pytest.mark.parametrize('roles', [set(), {'capacity_plan'},
                                 {'capacity_plan', 'capacity_receipt', 'capacity_terminal', 'capacity_audit', 'eval'}])
def test_lineage_roles_reject_before_input_or_array_read(monkeypatch, tmp_path, roles):
    path = tmp_path / 'plan.json'
    plan = {'version': S.VERSION, 'status': 'frozen_before_execution',
            'configuration': S.configuration(), 'limits': S.LIMITS,
            'independent_audit_limits': S.AUDIT_LIMITS,
            'sources': {**dict.fromkeys(S.NEW, 'source'), S.CAPACITY: S.CAPACITY_PIN},
            'inputs': {role: {} for role in roles}}
    monkeypatch.setattr(S, 'read', lambda p: plan if p == path else forbidden())
    monkeypatch.setattr(S, 'regular', Path)
    monkeypatch.setattr(S, 'sha', lambda p: 'plan' if p == path else
                        S.CAPACITY_PIN if str(p) == S.CAPACITY else 'source')
    monkeypatch.setattr(S, 'descriptor', forbidden)
    monkeypatch.setattr(S.C, 'authenticate', forbidden)
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(ValueError, match='exact lineage roles'):
        S.authenticate(SimpleNamespace(plan=path, plan_sha256='plan'))


def test_exclusive_output_prevents_any_bind_or_setup(monkeypatch, tmp_path):
    run = S.Run(SimpleNamespace(output=tmp_path))
    monkeypatch.setattr(run, 'bind', forbidden)
    monkeypatch.setattr(run, 'setup', forbidden)
    with pytest.raises(ValueError, match='exclusive output'):
        run.execute()
    assert list(tmp_path.iterdir()) == []


def test_failed_work_receipt_survives_flush_and_close_errors(monkeypatch, tmp_path):
    out = tmp_path / 'failed'
    run = S.Run(SimpleNamespace(output=out))
    original = RuntimeError('original operation failure')

    class BrokenClose:
        def close(self):
            raise OSError('secondary close failure')

    def operation():
        raise original

    def bind():
        run.plan = {'limits': S.LIMITS}

    def setup():
        run.handles['broken'] = BrokenClose()
        run.call('value_forward', operation)

    def sync():
        raise OSError('secondary flush failure')

    monkeypatch.setattr(run, 'bind', bind)
    monkeypatch.setattr(run, 'setup', setup)
    monkeypatch.setattr(run, 'sync', sync)
    with pytest.raises(RuntimeError) as caught:
        run.execute()
    assert caught.value is original
    assert any('flush failure' in note for note in original.__notes__)
    assert any('close failure' in note for note in original.__notes__)
    receipt = json.loads((out / 'failed.json').read_text())
    assert receipt['status'] == 'failed' and receipt['completed_episodes'] == 0
    assert receipt['calls']['value_forward']['attempted'] == 1
    assert receipt['calls']['value_forward']['returned'] == 0
    assert receipt['pending'][0]['channel'] == 'value_forward'
    assert not (out / 'receipt.json').exists()
    assert (out / 'work.jsonl').read_text().count('\n') == 1


def test_late_cap_failure_demotes_completed_receipt(monkeypatch, tmp_path):
    out = tmp_path / 'late'
    run = S.Run(SimpleNamespace(output=out))
    run.plan = {'synthetic': True}
    run.start = 0
    run.clock = SimpleNamespace(now_ns=lambda: 100, backend='synthetic')
    run.setup_seconds = 0.
    run.calls = {k: {'attempted': n, 'returned': n, 'seconds': 0.}
                 for k, n in (('native_reset', 74), ('native_step', 72), ('value_forward', 72))}

    def episode(_regime, _seed, _hit, _collector, _case):
        return {'source_evaluation_only': [1, 2], 'draws_evaluation_only': [],
                'steps': 1, 'found': True, 'controller_seconds': 0.,
                'environment_seconds': 0., 'sampling_seconds': 0.}

    def prepare():
        for name in S.payload_names() - {'summary.json'}:
            (out / name).write_bytes(b'fake closed payload')
        return {'synthetic': True}

    def check():
        if (out / 'receipt.json').exists():
            raise TimeoutError('post-publication deadline')

    monkeypatch.setattr(run, 'bind', lambda: None)
    monkeypatch.setattr(run, 'setup', lambda: None)
    monkeypatch.setattr(run, 'episode', episode)
    monkeypatch.setattr(run, 'prepare_mixture', prepare)
    monkeypatch.setattr(run, 'check', check)
    monkeypatch.setattr(S, 'authenticate', lambda args: run.plan)
    with pytest.raises(TimeoutError, match='post-publication deadline'):
        run.execute()
    assert not (out / 'receipt.json').exists()
    assert json.loads((out / 'invalid-completed-receipt.json').read_text())['status'] == 'completed'
    assert json.loads((out / 'failed.json').read_text())['status'] == 'failed'
