"""Synthetic four-arm accounting and selection control; no native or TF calls."""
from __future__ import annotations

import builtins
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research.otto_restricted_policy import RestrictedPolicyActor

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


FIXTURES = module('tests/test_otto_released_reference.py', 'boundary_old_synthetic_fixtures')


@pytest.fixture
def m():
    return module('scripts/study_otto_boundary_control.py', 'boundary_synthetic_runner')


def rows(m):
    result = FIXTURES.rows(m)
    for row in result:
        steps = {'released_tf': 90, 'released_inbounds': 80, 'analytic_all4': 100, 'analytic_inbounds': 100}[row['arm']]
        row.update(steps=steps, capped_time=steps, update_calls=steps, choose_calls=steps,
                   model_forward_calls=steps if row['arm'] in m.NEURAL else 0)
    return result


def test_hash_bound_namespaces_keep_original_authentication_constants(m):
    assert m.PRIOR.VERSION == 'otto-released-reference-v1'
    assert m.PRIOR.ARMS == ('released_tf', 'analytic_all4', 'analytic_inbounds')
    assert m.PRIOR.LIMITS['native_resets'] == 576
    assert m.BASE.VERSION == 'otto-boundary-control-v1' and m.BASE.EPISODES == 768
    assert m.PRIOR.authenticate is not m.BASE.authenticate
    assert m.PRIOR.Run.setup.__code__ is not m.Run.setup.__code__


def test_import_is_stdlib_only_before_external_plan(monkeypatch):
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        assert name.split('.')[0] not in {'numpy', 'tensorflow', 'tf_keras', 'scipy', 'isotropic'}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guard)
    module('scripts/study_otto_boundary_control.py', 'boundary_clean_import')


def test_four_arm_rotation_and_complete_caps(m):
    order = list(m.case_order())
    assert len(order) == len(set(order)) == 768
    assert [r[-1] for r in order[:4]] == list(m.ARMS)
    assert [r[-1] for r in order[4:8]] == list(m.ARMS[1:] + m.ARMS[:1])
    assert m.LIMITS['native_steps'] == 1680384 and m.LIMITS['tensorflow_value_calls'] == 840192
    assert m.NEURAL_EPISODES == 384
    for name, seed in (('base', 870001), ('shift', 880001)):
        chosen = [r for r in order if r[0] == name]
        assert {r[1] for r in chosen} == set(range(seed, seed + 96))
        assert all(sum(r[2] == b and r[3] == h and r[4] == a for r in chosen) == 4
                   for a in m.ARMS for b in range(8) for h in (1, 2, 3))


def test_all_5_6_12_16_conditions_and_no_automatic_admission(m):
    result = m.summarize(rows(m), FIXTURES.weights())
    assert result['candidate_arm'] == 'released_inbounds'
    assert result['restriction_benefit'] and result['competent_reference'] and result['stronger_value_teacher']
    assert result['utility_compute_advantage']  # Synthetic restricted actor has the analytic cost and fewer moves.
    assert not result['learned_pilot_admission'] and not result['inherited_gate_revised']
    assert len(result['restriction_benefit_checks']) == 5
    for name, expected in (('competence_checks', 6), ('stronger_teacher_checks', 12), ('utility_compute_checks', 16)):
        assert sum(len(c[name]) for c in result['cohorts'].values()) == expected


def test_no_strict_restriction_gain_means_failed_fifth_rule(m):
    cohort = rows(m)
    for row in cohort:
        if row['arm'] == 'released_inbounds':
            row.update(steps=90, capped_time=90, update_calls=90, choose_calls=90, model_forward_calls=90)
    result = m.summarize(cohort, FIXTURES.weights())
    assert [r['passes'] for r in result['restriction_benefit_checks']] == [True, True, True, True, False]
    assert not result['restriction_benefit']


def test_shift_cannot_be_rescued_by_baseline_and_strata_are_weighted(m):
    cohort = rows(m)
    for row in cohort:
        if row['arm'] == 'released_inbounds':
            step = {1: 80, 2: 100, 3: 120}[row['initial_hit']]
            row.update(steps=step, capped_time=step, choose_calls=step, update_calls=step, model_forward_calls=step)
    result = m.summarize(cohort, FIXTURES.weights())
    assert result['cohorts']['base']['means']['released_inbounds']['capped_time'] == 95
    assert result['cohorts']['shift']['means']['released_inbounds']['capped_time'] == 100
    assert not result['restriction_benefit'] and not result['stronger_value_teacher']


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'seed', 'order', 'block', 'hit', 'censor', 'calls', 'blocked', 'io'])
def test_complete_cohort_rejects_invalid_cases(m, defect):
    cohort = rows(m)
    if defect == 'missing':
        cohort.pop()
    elif defect == 'duplicate':
        cohort[-1] = copy.deepcopy(cohort[0])
    elif defect == 'order':
        cohort[0], cohort[1] = cohort[1], cohort[0]
    elif defect in ('seed', 'block', 'hit'):
        cohort[0]['initial_hit' if defect == 'hit' else defect] += 1
    elif defect == 'censor':
        cohort[0]['found'] = False
    elif defect == 'calls':
        cohort[1]['model_forward_calls'] = 0
    elif defect == 'blocked':
        cohort[1]['blocked_steps'] = 1
    else:
        cohort[0]['controller_excluded_io_seconds'] = -1
    with pytest.raises(ValueError):
        m.summarize(cohort, FIXTURES.weights())


def public_step(t, allowed=(1, 2, 3)):
    return {'position': [0, 26], 'hit': 0, 'done': False, 'step': t, 'valid_actions': list(allowed)}


def decision(t, action):
    return {'public_before': public_step(t), 'public_after': public_step(t + 1), 'action': action,
            'posterior_sha256': 'posterior', 'raw_costs_sha256': 'raw-costs'}


def test_shared_prefix_permits_only_the_predeclared_boundary_divergence(m):
    left = [decision(0, 1), decision(1, 0)]
    right = [decision(0, 1), decision(1, 1)]
    result = m.paired_prefix(left, right)
    assert result['common_prefix_decisions_checked'] == 2 and result['first_divergent_action_step'] == 2
    assert m.paired_prefix(left, copy.deepcopy(left))['first_divergent_action_step'] is None


@pytest.mark.parametrize('defect', ['cost', 'posterior', 'public', 'allowed_divergence', 'after', 'missing'])
def test_shared_prefix_detects_more_than_selector_changes(m, defect):
    left, right = [decision(0, 1), decision(1, 1)], [decision(0, 1), decision(1, 1)]
    if defect == 'cost':
        right[0]['raw_costs_sha256'] = 'changed'
    elif defect == 'posterior':
        right[0]['posterior_sha256'] = 'changed'
    elif defect == 'public':
        right[0]['public_before']['hit'] = 1
    elif defect == 'allowed_divergence':
        right[0]['action'] = 2
    elif defect == 'after':
        right[0]['public_after']['hit'] = 1
    else:
        right.pop()
    with pytest.raises(ValueError):
        m.paired_prefix(left, right)


def test_single_deferred_setup_metadata_write_does_not_touch_inference(m, tmp_path):
    record = {'allocated_over_released_episodes': 192, 'model_setup_seconds': 3.84, 'weights': 'unchanged'}
    target = tmp_path / 'setup.json'
    m.BASE.write(target, record)
    assert not target.exists() and record['allocated_over_released_episodes'] == 384
    assert record['model_setup_seconds'] == 3.84 and record['weights'] == 'unchanged'
    m.write(target, record)
    assert json.loads(target.read_text()) == record
    with pytest.raises(FileExistsError):
        m.write(target, record)


def runtime(kernel, *, found):
    rt = FIXTURES.fake_runtime(kernel, found=found)
    old_seeded = rt.public.seeded_environment

    def seeded(cls, seed, config, *, initial_hit):
        assert seed == 870001
        return old_seeded(cls, 850001, config, initial_hit=initial_hit)

    rt.public.seeded_environment = seeded
    rt.restricted = RestrictedPolicyActor
    return rt


@pytest.mark.parametrize('arm', ['released_tf', 'released_inbounds', 'analytic_all4', 'analytic_inbounds'])
@pytest.mark.parametrize('found', [True, False])
def test_four_arm_real_fake_call_accounting_and_final_update(m, tmp_path, monkeypatch, arm, found):
    monkeypatch.setattr(m, 'HORIZON', 3)
    kernel = np.full((4, 107, 107), .25, dtype=np.float64)
    kernel[:, 53, 53] = 0
    rt = runtime(kernel, found=found)
    run = m.Run(SimpleNamespace(output=tmp_path))
    run.check = run.ledger.check = lambda: None
    row = run.episode(rt, ('base', 870001, 0, 1, arm))
    expected = 2 if found else 3
    assert row['steps'] == row['update_calls'] == expected
    assert row['found'] is found and row['final_update_assimilated']
    assert rt.model.calls == row['model_forward_calls'] == (expected if arm in m.NEURAL else 0)
    assert row['model_setup_allocation_seconds'] == (.0005 if arm in m.NEURAL else 0)
    events = [json.loads(line) for line in (tmp_path / 'transitions.jsonl').read_text().splitlines()]
    assert events[-1]['public']['done'] is found
    assert all(len(r['selection_mask']) == 4 for r in events[1:])
    if arm in m.NEURAL:
        assert all(r['costs'] == [4.] * 4 for r in events[1:])


def test_actual_boundary_selection_changes_only_choice_with_one_fake_forward(m, tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'HORIZON', 28)
    kernel = np.full((4, 107, 107), .25, dtype=np.float64)
    kernel[:, 53, 53] = 0
    paths, results = {}, {}
    for arm in m.NEURAL:
        rt = runtime(kernel, found=False)
        original = rt.source

        class BoundaryEnvironment(original):
            def step(self, action, *, quiet):
                assert quiet and not self.obs['done']
                axis = action // 2
                self.agent[axis] = max(0, min(52, self.agent[axis] + 2 * (action % 2) - 1))
                self.draw_log.append({'channel': 'hit', 'index': len(self.draw_log) - 1,
                                      'uniform': .5, 'selected_index': 0, 'cdf_mass': 1.})
                self.assimilate(0, False)
                self.obs = {'hit': 0, 'done': False}
                return 0, 0., False

        rt.source = BoundaryEnvironment
        rt.public.seeded_environment = lambda _cls, _seed, _config, *, initial_hit, cls=BoundaryEnvironment: cls(initial_hit)

        def observation(env, step):
            valid = tuple(a for a in range(4) if 0 <= env.agent[a // 2] + 2 * (a % 2) - 1 <= 52)
            return SimpleNamespace(position=tuple(env.agent), hit=0 if step else 1, done=False, step=step, valid_actions=valid)

        rt.public.observation = observation
        out = tmp_path / arm
        out.mkdir()
        run = m.Run(SimpleNamespace(output=out))
        run.check = run.ledger.check = lambda: None
        results[arm] = run.episode(rt, ('base', 870001, 0, 1, arm))
        paths[arm] = run.neural_path
        assert rt.model.calls == 28
    witness = m.paired_prefix(paths['released_tf'], paths['released_inbounds'])
    assert witness['first_divergent_action_step'] == witness['common_prefix_decisions_checked'] == 27
    assert results['released_tf']['blocked_steps'] == 2 and results['released_inbounds']['blocked_steps'] == 0
    assert paths['released_tf'][26]['raw_costs_sha256'] == paths['released_inbounds'][26]['raw_costs_sha256']


def test_bad_plan_pin_stops_before_prior_authentication(m, tmp_path, monkeypatch):
    path = tmp_path / 'plan.json'
    path.write_text('{}')
    monkeypatch.setattr(m.PRIOR, 'authenticate', lambda *_: pytest.fail('prior evidence opened'))
    with pytest.raises(ValueError, match='external boundary plan pin'):
        m.authenticate(SimpleNamespace(plan=path, plan_sha256='0' * 64))
