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
spec = importlib.util.spec_from_file_location('symmetry_study_tests', ROOT / 'scripts/study_otto_symmetry_head.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def rows():
    result = []
    for regime, seed, block, hit, arm in study.evaluation_order():
        family = arm.split('@')[0]
        moves = 10 if family == 'shared' else 11 if family == 'analytic_inbounds' else 12
        cost = .5 if family == 'shared' else 1. if family == 'analytic_inbounds' else .6
        result.append({'regime': regime, 'seed': seed, 'block': block, 'initial_hit': hit, 'arm': arm,
                       'steps': moves, 'found': True, 'updates': moves, 'blocked_steps': 0,
                       'init_seconds': 0., 'choose_seconds': cost, 'update_seconds': 0., 'setup_allocation_seconds': 0.,
                       'controller_seconds': cost, 'environment_seconds': 0., 'state_bytes': 22472})
    return result


MIXTURE = {r: {1: .75, 2: .1875, 3: .0625} for r in study.REGIMES}


@pytest.mark.parametrize('length', [1, 2, 63, 64, 65, 101, 2188])
def test_prefix_sampling_is_rounded_unique_and_preserves_ends(length):
    actual = study.sample_indices(length)
    count = min(length, 64)
    assert actual == ([0] if count == 1 else [round(i * (length - 1) / (count - 1)) for i in range(count)])
    assert actual == sorted(set(actual)) and len(actual) == count
    assert actual[0] == 0 and actual[-1] == length - 1
    if length == 65:
        assert actual != [i * 64 // 63 for i in range(64)]


@pytest.mark.parametrize('length', [0, -1, 2189, True, 2.5])
def test_invalid_prefix_length_rejected(length):
    with pytest.raises(ValueError):
        study.sample_indices(length)


def test_episode_weights_equalize_unequal_lengths_and_pool_collectors():
    metadata = [{'episode_id': 'teacher:A'}] + [{'episode_id': 'student1:A'}] * 2 + [{'episode_id': 'student2:A'}] * 3
    weights = study.row_weights(metadata)
    assert weights == [2., 1., 1., 2 / 3, 2 / 3, 2 / 3]
    assert sum(weights) == 6
    for episode in ('teacher:A', 'student1:A', 'student2:A'):
        assert sum(w for w, row in zip(weights, metadata, strict=True) if row['episode_id'] == episode) == 2


def test_fixed_cohort_and_caps():
    order = list(study.evaluation_order())
    assert len(order) == 720 == len(set(order))
    assert order[0] == ('lambda3', 970001, 0, 1, 'shared@9101')
    assert order[10][-1] == study.ARMS[1]
    for r in study.REGIMES:
        assert len({t[1] for t in order if t[0] == r}) == 24
        for arm in study.ARMS:
            for h in (1, 2, 3):
                assert sum(t[0] == r and t[3] == h and t[4] == arm for t in order) == 8
    assert study.LIMITS['native_steps'] == 2415808
    assert study.LIMITS['native_resets'] == 1115
    assert study.LIMITS['optimizer_updates'] == 63360


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
        if row['arm'] == 'shared@9101':
            row['steps'] = row['updates'] = {1: 10, 2: 20, 3: 40}[row['initial_hit']]
    result = study.summary(data, MIXTURE)
    assert result['regimes']['lambda3']['means']['shared@9101']['steps'] == 13.75
    assert result['regimes']['lambda3']['family_means']['shared']['steps'] == 11.25
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
    ('shared', 'steps', 20, 'competence_checks'),
    ('shared', 'controller_seconds', .81, 'compression_checks'),
    ('dense_ensemble', 'steps', 10, 'architecture_checks'),
    ('dense', 'controller_seconds', .49, 'architecture_checks'),
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


def test_soft_targets_preserve_ties_and_exclude_blocked_costs():
    target, mask = study.teacher_target(np.array([np.inf, 2., 2., 3.]), (1, 2, 3), np)
    assert target[0] == 0 and not mask[0]
    assert target[1] == target[2] > target[3]
    assert target.sum() == pytest.approx(1)
    assert study.choose(np.array([-100., 2., 2., 3.], dtype=np.float32), (1, 2, 3), np) == 1


def fake_run(tmp_path, monkeypatch, *, found):
    monkeypatch.setattr(study, 'HORIZON', 2)
    run = study.Run(SimpleNamespace(output=tmp_path, plan=tmp_path / 'plan.json', plan_sha256='bad', supervision=tmp_path / 'supervision.json'))
    run.np = np
    run.kernels = {'lambda3': np.zeros((1,), dtype=np.float64)}
    policy_calls, feature_calls, head_calls = [], [], []

    def initial():
        p = np.ones((53, 53), dtype=np.float64)
        p[26, 26] = 0
        return p / p.sum()

    def updated(p, pos, done):
        p = p.copy()
        if done:
            p.fill(0)
            p[pos] = 1
        else:
            p[pos] = 0
            p /= p.sum()
        return p

    class Env:
        def __init__(self):
            self.agent, self.source = [26, 26], np.array([25, 25])
            self.p_source, self.obs = initial(), {'hit': 1, 'done': False}
            self.draw_log = []
            self.steps = 0

        def step(self, action, quiet):
            self.steps += 1
            self.agent[action // 2] += 2 * (action % 2) - 1
            done = found and self.steps == 2
            self.obs = {'hit': -2 if done else 0, 'done': done}
            self.p_source = updated(self.p_source, tuple(self.agent), done)
            return self.obs['hit'], float(done), done

    class Actor:
        def __init__(self, public, kernel, allow_stay):
            assert not allow_stay and set(public) == {'position', 'step', 'hit', 'done', 'valid_actions'}
            self.public, self.belief = public, initial()
            self._policy = SimpleNamespace(_value_policy=self.label)

        def label(self):
            policy_calls.append(self.public['step'])
            return 0, np.array([0., 1., 2., 3.])

        def choose(self):
            raise AssertionError('Teacher pending action must not be committed for student collection')

        def update(self, action, public):
            assert public['step'] == self.public['step'] + 1
            self.public, self.belief = public, updated(self.belief, public['position'], public['done'])

        def storage_bytes(self):
            return {'mutable_array_bytes': self.belief.nbytes}

    class Feature:
        def __init__(self, kernel, lam):
            assert lam == 3
            self.kernel = kernel

        def features(self, belief, packet):
            assert type(belief) is np.ndarray and isinstance(packet, dict)
            feature_calls.append(packet['step'])
            return np.array([packet['step']], dtype=np.float32)

    class Head:
        def scores(self, feature, *, kind=None):
            head_calls.append((feature.tolist(), kind))
            return np.array([2., 1., 0., 3.], dtype=np.float32)

        def storage_bytes(self):
            return {'parameter_array_bytes': 16}

    run.environment = lambda *args: Env()
    run.actor_class = Actor
    run.model = SimpleNamespace(PublicFeatureMap=Feature)
    run.public = lambda env, step: {'position': tuple(env.agent), 'step': step, 'hit': env.obs['hit'],
                                  'done': env.obs['done'], 'valid_actions': () if env.obs['done'] else (0, 1, 2, 3)}
    return run, Head(), policy_calls, feature_calls, head_calls


@pytest.mark.parametrize('found', [True, False])
@pytest.mark.parametrize('stage', ['dagger', 'eval'])
def test_actual_episode_uses_student_action_and_assimilates_final(tmp_path, monkeypatch, found, stage):
    run, head, labels, features, heads = fake_run(tmp_path, monkeypatch, found=found)
    row, selected = run.episode(stage, 'lambda3', 950001, 1, 'shared@9101', head=head)
    assert row['updates'] == row['steps'] == 2 and row['found'] is found
    assert row['final_update_assimilated'] is True
    assert features == [0, 1] and len(heads) == 2
    assert labels == ([0, 1] if stage == 'dagger' else [])
    assert len(selected) == (2 if stage == 'dagger' else 0)
    assert run.calls['head_predict']['returned'] == 2
    assert run.calls['native_step']['returned'] == 2
    paths = list(tmp_path.glob('*transitions.jsonl'))
    transitions = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert [r['action'] for r in transitions[1:]] == [2, 2]
    assert transitions[-1]['public']['done'] is found
    for handle in run.handles.values():
        handle.close()


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
        run.calls['native_reset'] = {'attempted': 1115, 'returned': 1115}
        run.receipt.update(collection_episodes=384, completed_stage_fits=12, completed_episodes=720)
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
