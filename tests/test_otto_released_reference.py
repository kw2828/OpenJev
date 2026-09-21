"""Synthetic cohort, instrumentation and public episode checks; no OTTO/TF calls."""
from __future__ import annotations

import builtins
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research.otto_released_policy import ReleasedPolicyActor

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location('synthetic_released_reference', ROOT / 'scripts/study_otto_released_reference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def m():
    return load_script()


def weights():
    return {'base': {1: .5, 2: .25, 3: .25}, 'shift': {1: .25, 2: .5, 3: .25}}


def rows(m):
    result = []
    for cohort, seed, block, hit, arm in m.case_order():
        steps = 90 if arm == 'released_tf' else 100
        row = {k: 0. for k in m.METRICS}
        row.update(cohort=cohort, seed=seed, block=block, initial_hit=hit, arm=arm,
                   steps=steps, capped_time=steps, found=True, stuck_steps=0, blocked_steps=0,
                   choose_seconds=2. if arm == 'released_tf' else 1., state_array_bytes=22472,
                   choose_calls=steps, update_calls=steps, model_forward_calls=steps if arm == 'released_tf' else 0)
        row['controller_seconds'] = row['choose_seconds']
        row['controller_instrumented_seconds'] = row['controller_seconds'] + .125
        row['controller_excluded_io_seconds'] = .125
        result.append(row)
    return result


def test_import_does_not_import_numeric_frameworks_or_environments(monkeypatch):
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        assert name.split('.')[0] not in {'numpy', 'tensorflow', 'tf_keras', 'scipy', 'isotropic', 'torch'}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guard)
    load_script()


def test_exact_cohort_rotation_and_worst_case_work(m):
    order = list(m.case_order())
    assert len(order) == len(set(order)) == 576
    assert [r[-1] for r in order[:3]] == list(m.ARMS)
    assert [r[-1] for r in order[3:6]] == list(m.ARMS[1:] + m.ARMS[:1])
    assert m.LIMITS['native_steps'] == 1260288 and m.LIMITS['tensorflow_value_calls'] == 420096
    for name, first in (('base', 850001), ('shift', 860001)):
        current = [r for r in order if r[0] == name]
        assert {r[1] for r in current} == set(range(first, first + 96))
        assert all(sum(r[2] == b and r[3] == h and r[4] == a for r in current) == 4
                   for b in range(8) for h in (1, 2, 3) for a in m.ARMS)


def test_all_six_twelve_and_sixteen_rules_and_distinct_compute_result(m):
    result = m.summarize(rows(m), weights())
    assert result['competent_reference'] and result['stronger_value_teacher']
    assert not result['utility_compute_advantage']
    assert not result['learned_pilot_admission'] and not result['inherited_gate_revised']
    assert sum(len(c['competence_checks']) for c in result['cohorts'].values()) == 6
    assert sum(len(c['stronger_teacher_checks']) for c in result['cohorts'].values()) == 12
    assert sum(len(c['utility_compute_checks']) for c in result['cohorts'].values()) == 16
    for c in result['cohorts'].values():
        assert c['unweighted_counts']['released_tf'] == {'episodes': 96, 'found': 96, 'censored': 0, 'total_steps': 8640}


def test_unequal_regime_strata_are_weighted_not_pooled(m):
    cohort = rows(m)
    for row in cohort:
        if row['arm'] == 'released_tf':
            value = {1: 80, 2: 100, 3: 120}[row['initial_hit']]
            row.update(steps=value, capped_time=value, choose_calls=value, update_calls=value, model_forward_calls=value)
    result = m.summarize(cohort, weights())
    assert result['cohorts']['base']['means']['released_tf']['capped_time'] == 95
    assert result['cohorts']['shift']['means']['released_tf']['capped_time'] == 100
    assert result['cohorts']['base']['strata'][3]['released_tf']['capped_time'] == 120
    assert result['cohorts']['base']['blocks'][0]['means']['released_tf']['capped_time'] == 95
    assert not result['stronger_value_teacher']


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'order', 'seed', 'hit', 'block', 'early_censor',
                                       'dropped_terminal_update', 'fake_model_calls', 'nan', 'cost', 'io'])
def test_bad_cohort_or_cost_cannot_be_scored(m, corruption):
    cohort = rows(m)
    if corruption == 'missing':
        cohort.pop()
    elif corruption == 'duplicate':
        cohort[-1] = copy.deepcopy(cohort[0])
    elif corruption == 'order':
        cohort[0], cohort[1] = cohort[1], cohort[0]
    elif corruption in ('seed', 'hit', 'block'):
        cohort[0]['initial_hit' if corruption == 'hit' else corruption] += 1
    elif corruption == 'early_censor':
        cohort[0]['found'] = False
    elif corruption == 'dropped_terminal_update':
        cohort[0]['update_calls'] -= 1
    elif corruption == 'fake_model_calls':
        cohort[1]['model_forward_calls'] = 1
    elif corruption == 'nan':
        cohort[0]['choose_seconds'] = math.nan
    elif corruption == 'cost':
        cohort[0]['controller_seconds'] += .5
    else:
        cohort[0]['controller_excluded_io_seconds'] = -.5
    with pytest.raises(ValueError):
        m.summarize(cohort, weights())


def test_horizon_failure_is_counted_and_final_found_at_horizon_is_distinct(m):
    cohort = rows(m)
    for row in cohort:
        if row['arm'] == 'released_tf' and row['initial_hit'] == 1:
            row.update(steps=2188, capped_time=2188, choose_calls=2188, update_calls=2188,
                       model_forward_calls=2188, found=False)
    result = m.summarize(cohort, weights())
    assert result['cohorts']['base']['means']['released_tf']['found'] == .5
    assert result['cohorts']['shift']['means']['released_tf']['found'] == .75
    assert not result['competent_reference'] and not result['stronger_value_teacher']
    for row in cohort:
        row['found'] = True
    assert m.summarize(cohort, weights())['cohorts']['base']['means']['released_tf']['found'] == 1


@pytest.mark.parametrize('which', ['success', 'moves_all4', 'moves_inbounds', 'blocks_all4', 'blocks_inbounds'])
def test_neither_control_or_regime_can_be_skipped(m, which):
    means = {a: {'found': 1., 'capped_time': 100., 'controller_seconds': 1.} for a in m.ARMS}
    means['released_tf']['capped_time'] = 90.
    blocks = [{'means': copy.deepcopy(means)} for _ in range(8)]
    if which == 'success':
        means['released_tf']['found'] = .94
    elif which.startswith('moves'):
        control = 'analytic_' + which.split('_', 1)[1]
        means[control]['capped_time'] = 80.
    else:
        control = 'analytic_' + which.split('_', 1)[1]
        for block in blocks[:3]:
            block['means'][control]['capped_time'] = 89.
    competence, teacher, _ = m.criteria(means, blocks)
    assert not all(c['passes'] for c in teacher)
    if not which.startswith('blocks'):
        assert not all(c['passes'] for c in competence)


def test_nested_ledger_excludes_measured_journal_io_once(m, tmp_path, monkeypatch):
    clock, journal = [0.], []
    monkeypatch.setattr(m.time, 'perf_counter', lambda: clock[0])

    def emit(_path, record):
        journal.append(record)
        clock[0] += 2.

    monkeypatch.setattr(m, 'append', emit)
    ledger = m.Ledger(tmp_path, lambda: None)

    def numeric():
        clock[0] += 5.
        return 'actual result'

    def outer():
        value = ledger.call('tensorflow_value', numeric)
        ledger.emit('forwards.jsonl', {'value': value})
        return value

    assert ledger.call('actor_choose', outer) == 'actual result'
    assert ledger.last_seconds['actor_choose'] == ledger.last_seconds['tensorflow_value'] == 5.
    assert ledger.last_instrumented['actor_choose'] == 11.
    assert ledger.last_io['actor_choose'] == 6.
    assert ledger.pending == []
    attempts = [r for r in journal if r.get('event') == 'attempt']
    assert attempts[0]['parent_call_id'] is None and attempts[1]['parent_call_id'] == attempts[0]['call_id']


def test_nested_failure_preserves_both_uncertain_attempts(m, tmp_path):
    ledger = m.Ledger(tmp_path, lambda: None)

    def fail():
        raise RuntimeError('synthetic forward failure')

    with pytest.raises(RuntimeError, match='synthetic forward failure'):
        ledger.call('actor_choose', lambda: ledger.call('tensorflow_value', fail))
    assert [r['channel'] for r in ledger.pending] == ['actor_choose', 'tensorflow_value']
    assert ledger.calls['tensorflow_value']['attempted'] == 1 and ledger.calls['tensorflow_value']['returned'] == 0
    assert len((tmp_path / 'work.jsonl').read_text().splitlines()) == 2


class Tensor:
    def __init__(self, value):
        self.value, self.shape = value, value.shape

    def numpy(self):
        return self.value


class FakeModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, inputs, *, training, sym_avg):
        assert not training and sym_avg and inputs.shape == (16, 105, 105)
        self.calls += 1
        return Tensor(np.full((16, 1), 3., dtype=np.float32))


class FakePolicy:
    def __init__(self, env, model, sym_avg):
        self.env, self.model, self.sym_avg = env, model, sym_avg

    def _value_policy(self):
        probs = np.full((4, 4), .25, dtype=np.float32)
        values = self.model(Tensor(np.zeros((16, 105, 105), dtype=np.float32)), sym_avg=self.sym_avg).numpy().reshape(4, 4)
        costs = np.asarray(1 + (probs * values).sum(axis=1), dtype=np.float32)
        return 0, costs


class FakeAnalytic(ReleasedPolicyActor):
    def __init__(self, initial, kernel, *, allow_stay):
        assert set(initial) == {'position', 'hit', 'done', 'step', 'valid_actions'}
        assert type(allow_stay) is bool
        super().__init__(initial, kernel, None, FakePolicy, sym_avg=False)

    def choose(self):
        assert not self.public['done'] and self._pending_action is None
        self._pending_action = 0
        return 0, np.ones(4, dtype=np.float64)


def fake_runtime(kernel, *, found):
    class Environment:
        N, Nhits, Nactions, draw_source = 53, 4, 4, True

        def __init__(self, initial_hit):
            self.agent = [26, 26]
            self.source = np.array([24, 26] if found else [52, 52])
            self.obs, self.p_Poisson, self.agent_stuck = {'hit': initial_hit, 'done': False}, kernel, False
            self.p_source = np.ones((53, 53), dtype=np.float64) / 2808
            self.p_source[26, 26] = 0
            self.assimilate(initial_hit, False)
            self.draw_log = [{'channel': 'source', 'index': 0, 'uniform': .25,
                              'selected_index': int(self.source[0] * 53 + self.source[1]), 'cdf_mass': 1.}]

        def assimilate(self, hit, done):
            pos = tuple(self.agent)
            if done:
                self.p_source = np.zeros((53, 53), dtype=np.float64)
                self.p_source[pos] = 1
                return
            self.p_source[pos] = 0
            x, y = 53 - pos[0], 53 - pos[1]
            self.p_source *= kernel[hit, x:x + 53, y:y + 53]
            mass = self.p_source.sum()
            if mass > 1e-10:
                self.p_source /= mass

        def step(self, action, *, quiet):
            assert quiet and not self.obs['done']
            self.agent[action // 2] += 2 * (action % 2) - 1
            done = self.agent == self.source.tolist()
            hit = -2 if done else 0
            if not done:
                self.draw_log.append({'channel': 'hit', 'index': len(self.draw_log) - 1,
                                      'uniform': .5, 'selected_index': 0, 'cdf_mass': 1.})
            self.assimilate(hit, done)
            self.obs = {'hit': hit, 'done': done}
            return hit, float(done), done

    def seeded(cls, seed, config, *, initial_hit):
        assert cls is Environment and seed == 850001 and config['Ngrid'] == 53
        return Environment(initial_hit)

    def observation(env, step):
        return SimpleNamespace(position=tuple(env.agent), hit=env.obs['hit'], done=env.obs['done'],
                               step=step, valid_actions=() if env.obs['done'] else (0, 1, 2, 3))

    return SimpleNamespace(np=np, source=Environment, released=ReleasedPolicyActor, analytic=FakeAnalytic,
        public=SimpleNamespace(seeded_environment=seeded, observation=observation), policy=FakePolicy,
        model=FakeModel(), kernels={'base': kernel},
        setup={'model_setup_seconds': .192, 'model_setup_instrumented_seconds': .384})


@pytest.mark.parametrize('arm', ['released_tf', 'analytic_all4', 'analytic_inbounds'])
@pytest.mark.parametrize('found', [True, False])
def test_actual_episode_updates_found_and_censored_end_and_counts_real_forwards(m, tmp_path, monkeypatch, arm, found):
    monkeypatch.setattr(m, 'HORIZON', 3)
    kernel = np.full((4, 107, 107), .25, dtype=np.float64)
    kernel[:, 53, 53] = 0
    runtime = fake_runtime(kernel, found=found)
    run = m.Run(SimpleNamespace(output=tmp_path))
    run.check = lambda: None
    run.ledger.check = lambda: None
    row = run.episode(runtime, ('base', 850001, 0, 1, arm))
    expected = 2 if found else 3
    assert row['steps'] == row['update_calls'] == row['choose_calls'] == expected
    assert row['found'] is found and row['final_update_assimilated']
    assert runtime.model.calls == row['model_forward_calls'] == (expected if arm == 'released_tf' else 0)
    assert run.ledger.calls['native_step']['returned'] == expected
    trace = [json.loads(line) for line in (tmp_path / 'transitions.jsonl').read_text().splitlines()]
    assert len(trace) == expected + 1 and trace[-1]['public']['done'] is found
    assert trace[-1]['posterior_after']['exact']
    if arm == 'released_tf':
        forwards = [json.loads(line) for line in (tmp_path / 'forwards.jsonl').read_text().splitlines()]
        assert len(forwards) == expected and all(r['values'] == [3.] * 16 for r in forwards)
        assert row['model_setup_allocation_seconds'] == .001
    else:
        assert not (tmp_path / 'forwards.jsonl').exists() and row['model_setup_allocation_seconds'] == 0


def test_bad_plan_pin_rejected_before_auth_import(m, tmp_path, monkeypatch):
    path = tmp_path / 'plan.json'
    path.write_text('{}')
    monkeypatch.setattr(m, 'load', lambda *_: pytest.fail('import before plan authentication'))
    with pytest.raises(ValueError, match='external study plan pin'):
        m.authenticate(SimpleNamespace(plan=path, plan_sha256='0' * 64))


def test_existing_output_is_not_modified(m, tmp_path, monkeypatch):
    marker = tmp_path / 'marker'
    marker.write_text('keep')
    run = m.Run(SimpleNamespace(output=tmp_path))
    monkeypatch.setattr(run, 'bind', lambda: pytest.fail('existing directory admitted'))
    with pytest.raises(FileExistsError):
        run.execute()
    assert marker.read_text() == 'keep'


def test_pending_forward_failure_receipt_retains_primary_error(m, tmp_path, monkeypatch):
    run = m.Run(SimpleNamespace(output=tmp_path / 'failed'))
    monkeypatch.setattr(run, 'bind', lambda: None)
    run.ledger.check = lambda: None

    def failed_body():
        def failure():
            raise RuntimeError('uncertain fake forward')
        run.ledger.call('tensorflow_value', failure)

    monkeypatch.setattr(run, 'body', failed_body)
    with pytest.raises(RuntimeError, match='uncertain fake forward'):
        run.execute()
    receipt = json.loads((run.out / 'receipt.json').read_text())
    assert receipt['status'] == 'failed' and receipt['work']['pending'][0]['channel'] == 'tensorflow_value'
    assert receipt['work']['calls']['tensorflow_value']['returned'] == 0
