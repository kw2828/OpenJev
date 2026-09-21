"""Synthetic contract checks. No upstream environment or scientific input loads."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('return_value_study_tests', ROOT / 'scripts/study_otto_return_value.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def rows():
    result = []
    for regime, seed, block, hit, arm in study.evaluation_order():
        family = arm.split('@')[0]
        moves = 10 if family == 'min8' else 11 if family == 'analytic_inbounds' else 12
        cost = .5 if family == 'min8' else 1. if family == 'analytic_inbounds' else .6
        result.append({'regime': regime, 'seed': seed, 'block': block, 'initial_hit': hit, 'arm': arm,
                       'steps': moves, 'found': True, 'updates': moves, 'blocked_steps': 0,
                       'init_seconds': 0., 'choose_seconds': cost, 'update_seconds': 0., 'setup_allocation_seconds': 0.,
                       'controller_seconds': cost, 'environment_seconds': 0., 'state_bytes': 22472})
    return result


MIXTURE = {r: {1: .75, 2: .1875, 3: .0625} for r in study.REGIMES}


def test_all54_conditions_and_three_regimes():
    result = study.summary(rows(), MIXTURE)
    assert [len(result[k]) for k in ('competence_checks', 'compression_checks', 'architecture_checks')] == [18, 12, 24]
    assert result['pilot_continuation'] is True
    assert result['learned_architecture_advantage_established'] is False
    assert result['inherited_gate_revised'] is False
    assert set(result['regimes']) == {'lambda3', 'lambda4', 'lambda5'}


def test_mixture_means_not_pooled_and_all_seeds_retained():
    data = rows()
    for row in data:
        if row['arm'] == 'min8@10101':
            row['steps'] = row['updates'] = {1: 10, 2: 20, 3: 40}[row['initial_hit']]
    result = study.summary(data, MIXTURE)
    assert result['regimes']['lambda3']['means']['min8@10101']['steps'] == 13.75
    assert result['regimes']['lambda3']['family_means']['min8']['steps'] == 11.25
    assert not result['pilot_continuation']


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'order', 'seed', 'hit', 'block', 'censor', 'last_update', 'blocked', 'cost', 'nan'])
def test_incomplete_or_malformed_evaluation_rejected(mutation):
    data = rows()
    if mutation == 'missing':
        data.pop()
    elif mutation == 'duplicate':
        data[-1] = copy.deepcopy(data[0])
    elif mutation == 'order':
        data[0], data[1] = data[1], data[0]
    elif mutation in ('seed', 'hit', 'block'):
        data[0][{'hit': 'initial_hit'}.get(mutation, mutation)] += 1
    elif mutation == 'censor':
        data[0]['found'] = False
    elif mutation == 'last_update':
        data[0]['updates'] -= 1
    elif mutation == 'blocked':
        data[0]['blocked_steps'] = 1
    elif mutation == 'cost':
        data[0]['controller_seconds'] += .01
    else:
        data[0]['state_bytes'] = float('nan')
    with pytest.raises(ValueError):
        study.summary(data, MIXTURE)


@pytest.mark.parametrize('family,metric,value,group', [
    ('min8', 'steps', 20, 'competence_checks'),
    ('min8', 'controller_seconds', .81, 'compression_checks'),
    ('homogeneous8', 'steps', 10, 'architecture_checks'),
    ('mlp8', 'controller_seconds', .49, 'architecture_checks'),
])
def test_no_rule_family_or_ensemble_can_be_ignored(family, metric, value, group):
    data = rows()
    for row in data:
        if row['arm'].startswith(family + '@'):
            row[metric] = value
            if metric == 'steps':
                row['updates'] = value
            else:
                row['choose_seconds'] = value
    result = study.summary(data, MIXTURE)
    assert not result['pilot_continuation']
    assert any(not c['passes'] for c in result[group])




def test_failed_operation_preserves_pending_and_compact_context(tmp_path):
    run = study.Run(SimpleNamespace(output=tmp_path))
    run.context = {'phase': 'eval', 'episode': 'caseA', 'step': 3}
    with pytest.raises(RuntimeError, match='original'):
        run.call('head_predict', lambda: (_ for _ in ()).throw(RuntimeError('original')))
    assert run.calls['head_predict']['attempted'] == 1 and run.calls['head_predict']['returned'] == 0
    assert run.pending[0]['context']['step'] == 3
    assert json.loads((tmp_path / 'work-contexts.jsonl').read_text()) == {'id': 0, 'context': {'phase': 'eval', 'episode': 'caseA'}}
    assert json.loads((tmp_path / 'work.jsonl').read_text()) == [1, 0, 'head_predict', 0, 3]
    for handle in run.handles.values():
        handle.close()


def test_auth_rejects_bad_external_plan_before_import(tmp_path, monkeypatch):
    plan = tmp_path / 'plan.json'
    plan.write_text('{}')
    monkeypatch.setattr(study, 'load', lambda *args: pytest.fail('must fail before source imports'))
    with pytest.raises(ValueError, match='external plan hash'):
        study.authenticate(SimpleNamespace(plan=plan, plan_sha256='bad'))


def test_execute_preserves_primary_when_final_cap_also_fails(tmp_path, monkeypatch):
    out = tmp_path / 'run'
    run = study.Run(SimpleNamespace(output=out))
    def bind():
        run.start = 1
        run.clock = SimpleNamespace(now_ns=lambda: 2, backend='test')
    run.bind = bind
    run.body = lambda: (_ for _ in ()).throw(RuntimeError('primary failure'))
    run.check = lambda: (_ for _ in ()).throw(ValueError('deadline cap'))
    with pytest.raises(RuntimeError, match='primary failure') as error:
        run.execute()
    receipt = json.loads((out / 'receipt.json').read_text())
    assert receipt['status'] == 'failed' and 'primary failure' in receipt['error']
    assert 'deadline cap' in error.value.__notes__[0]


def test_execute_output_collision_does_not_change_existing_file(tmp_path):
    out = tmp_path / 'run'
    out.mkdir()
    marker = out / 'existing'
    marker.write_bytes(b'unchanged')
    run = study.Run(SimpleNamespace(output=out))
    with pytest.raises(ValueError, match='exclusive'):
        run.execute()
    assert marker.read_bytes() == b'unchanged' and list(out.iterdir()) == [marker]


def test_work_cap_rejects_before_actual_operation(tmp_path):
    run = study.Run(SimpleNamespace(output=tmp_path))
    run.calls['native_step'] = {'attempted': study.LIMITS['native_steps'], 'returned': study.LIMITS['native_steps'], 'seconds': 0.}
    with pytest.raises(ValueError, match='before invocation'):
        run.call('native_step', lambda: pytest.fail('over-budget operation ran'))
    assert not run.pending and not list(tmp_path.iterdir())


def test_clock_failure_preserves_original_error_and_null_final_time(tmp_path):
    run = study.Run(SimpleNamespace(output=tmp_path / 'run'))
    def bind():
        run.start = 1
        run.clock = SimpleNamespace(now_ns=lambda: (_ for _ in ()).throw(RuntimeError('clock broke')), backend='fake')
    run.bind = bind
    run.body = lambda: (_ for _ in ()).throw(ValueError('original body'))
    with pytest.raises(ValueError, match='original body'):
        run.execute()
    receipt = json.loads((run.out / 'receipt.json').read_text())
    assert receipt['status'] == 'failed' and receipt['finished_ns'] is None
    assert 'clock broke' in receipt['finalization_errors'][0]


def test_late_limit_failure_demotes_completion_preserving_original(tmp_path, monkeypatch):
    launch = tmp_path / 'launch.json'
    launch.write_text('{}')
    run = study.Run(SimpleNamespace(output=tmp_path / 'run', supervision=launch))
    def bind():
        run.start, run.plan = 1, {}
        run.clock = SimpleNamespace(now_ns=lambda: 2, backend='fake')
        run.receipt['supervision_sha256'] = study.sha(launch)
    def body():
        run.calls['native_reset'] = {'attempted': 731, 'returned': 731}
        run.calls['optimizer_update'] = {'attempted': 31680, 'returned': 31680}
        run.receipt.update(prepared_episodes=240, completed_fits=9, completed_episodes=720)
        return {'pilot_continuation': False}
    checks = []
    def check():
        checks.append(None)
        if len(checks) == 3:
            raise ValueError('late cap')
    run.bind, run.body, run.check = bind, body, check
    monkeypatch.setattr(study, 'authenticate', lambda args: {})
    with pytest.raises(ValueError, match='late cap'):
        run.execute()
    assert json.loads((run.out / 'completion-before-late-failure.json').read_text())['status'] == 'completed'
    assert json.loads((run.out / 'receipt.json').read_text())['status'] == 'failed'
    assert json.loads((run.out / 'late-failure.json').read_text())['status'] == 'failed'


def test_exact_counts_rotation_disjoint_old_data_and_uniform_targets():
    order = list(study.evaluation_order())
    assert len(order) == 720 == len(set(order))
    assert order[0] == ('lambda3', 11100001, 0, 1, 'min8@10101')
    assert order[10][-1] == study.ARMS[1]
    teachers = list(study.training_order())
    assert len(teachers) == 240 and sum(r[0] == 'train' for r in teachers) == 192
    assert all(t[2] < 1000000 for t in teachers)
    assert all(t[1] > 11000000 for t in order)
    assert study.LIMITS['native_steps'] == 1575616
    assert study.LIMITS['native_resets'] == 731
    assert study.LIMITS['optimizer_updates'] == 31680
    assert len(study.prior_payload_names()) == 41
    assert len(study.payload_names()) == 44
    assert study.return_target(170, 0) == 170 / 64
    assert study.return_target(170, 169) == 1 / 64
    assert study.return_target(2, 1) == study.return_target(170, 169)


@pytest.mark.parametrize('length,index', [(0, 0), (2189, 0), (2, 2), (2, -1), (True, 0), (2, True)])
def test_returns_reject_terminal_or_invalid_prefix(length, index):
    with pytest.raises(ValueError):
        study.return_target(length, index)


def test_center_exact_raw_mass_and_position_not_normalized():
    b = np.zeros((3, 53, 53), dtype=np.float64)
    b[0, 0, 52], b[1, 7, 9] = .3, 1e-14
    pos = np.array([[0, 52], [52, 0], [26, 26]], dtype=np.int64)
    centered = study.center(b, pos, np)
    assert centered.shape == (3, 105, 105)
    assert centered[0, 52, 52] == .3 and centered[1, 7, 61] == 1e-14
    assert np.array_equal(centered.sum((1, 2)), b.sum((1, 2)))


def public(t, done=False):
    return {'position': (26 - t, 26), 'step': t, 'hit': -2 if done else 1,
            'done': done, 'valid_actions': () if done else (0, 1, 2, 3)}


def belief(t, done=False):
    value = np.zeros((53, 53), dtype=np.float64)
    if done:
        value[26 - t, 26] = 1
    else:
        value[1, 1], value[2, 2] = .25 + .125 * t, .75 - .125 * t
    return value


class PublicActor:
    def __init__(self, initial, kernel, allow_stay):
        assert not allow_stay
        self.public, self.belief = study.packet(initial), belief(0)
        self._policy = SimpleNamespace(_value_policy=self.analytic)

    @staticmethod
    def analytic():
        return 0, np.array([1., 2., 3., 4.])

    def update(self, action, after):
        assert action == 0 and after['step'] == self.public['step'] + 1
        self.public = study.packet(after)
        self.belief = belief(after['step'], after['done'])

    def storage_bytes(self):
        return {'mutable_array_bytes': self.belief.nbytes}


def teacher_fixture():
    identity = 'train:lambda3:910001:teacher'
    row = {'episode_id': identity, 'stage': 'train', 'regime': 'lambda3', 'seed': 910001, 'initial_hit': 1,
           'steps': 2, 'updates': 2, 'found': True, 'final_update_assimilated': True, 'final_public': public(2, True)}
    reset = {'kind': 'reset', 'episode_id': identity, 'public': public(0), 'posterior_after': study.posterior_witness(belief(0), np)}
    events = [{'kind': 'step', 'episode_id': identity, 'step': t, 'action': 0, 'public': public(t, t == 2),
               'posterior_before': study.posterior_witness(belief(t - 1), np),
               'posterior_after': study.posterior_witness(belief(t, t == 2), np)} for t in (1, 2)]
    return row, reset, events


def test_teacher_replay_uses_all_current_states_and_stops_before_other_stream():
    row, reset, events = teacher_fixture()
    stream = iter([*(json.dumps(e) for e in events), 'MUST NOT READ DAGGER'])
    result = study.teacher_records(row, reset, stream, PublicActor, None, np)
    assert [r[1]['prefix_index'] for r in result] == [0, 1]
    assert [r[1]['target'] for r in result] == [2 / 64, 1 / 64]
    assert np.array_equal(result[0][0], belief(0))
    assert np.array_equal(result[1][0], belief(1))
    assert next(stream) == 'MUST NOT READ DAGGER'


@pytest.mark.parametrize('corruption', ['censored', 'missing_final_update', 'bad_hash', 'wrong_step', 'missing_step'])
def test_teacher_replay_fails_without_dropping_censored_or_corrupt_rows(corruption):
    row, reset, events = teacher_fixture()
    if corruption == 'censored':
        row['found'] = False
    elif corruption == 'missing_final_update':
        row['final_update_assimilated'] = False
    elif corruption == 'bad_hash':
        events[0]['posterior_after']['sha256'] = 'bad'
    elif corruption == 'wrong_step':
        events[1]['step'] = 9
    else:
        events.pop()
    with pytest.raises((ValueError, StopIteration)):
        study.teacher_records(row, reset, iter(json.dumps(e) for e in events), PublicActor, None, np)


@pytest.mark.parametrize('found', [True, False])
@pytest.mark.parametrize('learned', [True, False])
def test_episode_explicit_branches_has_no_teacher_call_and_keeps_final_update(tmp_path, monkeypatch, found, learned):
    monkeypatch.setattr(study, 'HORIZON', 2)
    run = study.Run(SimpleNamespace(output=tmp_path))
    run.np, run.actor_class = np, PublicActor
    run.kernels = {'lambda3': np.zeros((4, 107, 107))}
    calls = []

    class Env:
        def __init__(self):
            self.steps, self.p_source = 0, belief(0)
            self.source, self.draw_log = np.array([24, 26]), []

        def step(self, action, quiet):
            assert action == 0 and quiet is True
            self.steps += 1
            done = found and self.steps == 2
            self.p_source = belief(self.steps, done)
            return -2 if done else 1, float(done), done

    def branches(b, pos, kernel, eligible):
        calls.append(('branch', tuple(pos)))
        assert isinstance(b, np.ndarray) and b.shape == (53, 53)
        return SimpleNamespace(raw_masses=np.zeros((4, 4)), weights=np.full((4, 4), 1e-10),
                               eligible_actions=eligible, centered_z=np.zeros((16, 105, 105)), successors=np.zeros((16, 2), dtype=np.int64))

    def explicit(branch, callback, arithmetic):
        assert arithmetic == 'float64'
        values = callback(branch.centered_z, branch.successors, run.kernels['lambda3'])
        assert values.shape == (16,) and np.all(values == 64)
        return np.array([1., 2., 3., 4.])

    class Head:
        def normalized(self, x):
            calls.append(('head', len(x)))
            return np.ones(len(x), dtype=np.float64)

        def storage_bytes(self):
            return {'parameter_array_bytes': 8}

    def feature(z, pos, lam):
        assert lam == 3 and len(z) == len(pos) == 16
        return np.ones((16, 2))

    run.model = SimpleNamespace(value_features=feature)
    run.branches = SimpleNamespace(rl_branches=branches, explicit_scores=explicit,
                                   select_action=lambda scores, eligible: study.choose(scores, eligible, np))
    run.environment = lambda *args: Env()
    run.public = lambda env, step: public(step, found and step == 2)
    result = run.episode('lambda3', 11100001, 1, 'min8@10101' if learned else 'analytic_inbounds', 0, Head() if learned else None)
    assert result['steps'] == result['updates'] == 2 and result['found'] is found
    assert result['final_update_assimilated']
    assert run.calls['native_step']['returned'] == 2
    assert ('analytic_choose' not in run.calls) if learned else run.calls['analytic_choose']['returned'] == 2
    assert run.calls.get('value_forward', {}).get('returned', 0) == (2 if learned else 0)
    events = [json.loads(line) for line in (tmp_path / 'eval-transitions.jsonl').read_text().splitlines()]
    assert events[-1]['public']['done'] is found
    assert events[-1]['weights'] == ([[1e-10] * 4] * 4 if learned else None)
    assert events[-1]['values'] == ([64.] * 16 if learned else None)
    assert result['choose_seconds'] == pytest.approx(result['choose_instrumented_seconds'] - result['choose_excluded_io_seconds'])
    for handle in run.handles.values():
        handle.close()
