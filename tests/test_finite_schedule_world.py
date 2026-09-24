"""Small engineering schedules and independent scalar physical-law checks."""
from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pytest

from openjev.research import finite_schedule_world as w

NAMESPACE = 949501


def physical():
    transition = np.empty((4,8,8),np.float64)
    hazard = np.empty((4,8),np.float64)
    costs = np.empty((4,8),np.float64)
    for a in range(4):
        for j in range(8):
            hazard[a,j] = float(Fraction(1+((j//4) ^ (a%2)),200))
            costs[a,j] = -.75 if a == ((j ^ (j//2)) % 4) else .25
            for k in range(8):
                destination = (k ^ 1,(k+1)%8,(2*k)%8+k//4,k ^ 4)[a]
                transition[a,j,k] = float(Fraction(1,400)+(Fraction(49,50) if j == destination else 0))
    return transition,hazard,costs


def path_oracle(actions,labels,epsilons):
    masses = [Fraction(0)]*8
    for path in itertools.product(range(8),repeat=len(labels)):
        error = Fraction(str(epsilons[0]))
        weight = Fraction(1,8)*(1-error if path[0]%4 == labels[0] else error/3)
        for t,a in enumerate(actions,1):
            j,k = path[t],path[t-1]
            destination = (k ^ 1,(k+1)%8,(2*k)%8+k//4,k ^ 4)[a]
            weight *= Fraction(1,400)+(Fraction(49,50) if j == destination else 0)
            hazard = Fraction(1+((j//4) ^ (a%2)),200)
            error = Fraction(str(epsilons[t]))
            weight *= hazard if labels[t] == 4 else (1-hazard)*(1-error if j%4 == labels[t] else error/3)
        masses[path[-1]] += weight
    evidence = sum(masses)
    posterior = [Fraction(0)]*8 if labels[-1] == 4 else [v/evidence for v in masses]
    return evidence,np.asarray(posterior,np.float64)


@pytest.mark.parametrize('family',range(4))
@pytest.mark.parametrize('boundary',(8,24))
def test_schedule_prior_is_one_iid_family_draw_then_switch_only_if_needed(family,boundary):
    class FakeRng:
        def __init__(self):
            self.calls = []

        def integers(self,low,high):
            self.calls.append((low,high))
            return family if (low,high) == (0,4) else boundary

    rng = FakeRng()
    eps,switch,actual = w._schedule(32,rng)
    assert actual == family and rng.calls == ([(0,4)] if family < 2 else [(0,4),(8,25)])
    first,last = ((.12,.12),(.30,.30),(.12,.30),(.30,.12))[family]
    assert eps.dtype == np.float64 and eps.shape == (33,)
    if family < 2:
        assert switch == -1 and np.all(eps == first)
    else:
        assert switch == boundary
        assert np.all(eps[:switch] == first) and np.all(eps[switch:] == last)


def test_independent_stream_contract_family_draws_and_accounting():
    result = w.generate(NAMESPACE,9,24)
    p,t,a,c = (result[name] for name in ('public','targets','audit','counts'))
    for case in range(9):
        seeds = np.random.SeedSequence([NAMESPACE,case]).spawn(4)
        actions,forks,schedule,_ = [np.random.Generator(np.random.PCG64(seed)) for seed in seeds]
        np.testing.assert_array_equal(p['actions'][case],actions.integers(0,4,size=24,dtype=np.int64))
        np.testing.assert_array_equal(t['fork_actions'][case],forks.integers(0,4,size=(25,8),dtype=np.int64))
        expected_family = int(schedule.integers(0,4))
        assert a['family'][case] == expected_family
        assert a['switch_indices'][case] == (int(schedule.integers(8,25)) if expected_family >= 2 else -1)
        assert p['case_ids'][case] == f'ns{NAMESPACE}-case{case:010d}'
    assert a['family'].dtype == np.int64
    assert c['version'] == 'finite-schedule-world-v1'
    assert c['family_draws'] == c['rng_case_sequences'] == c['reset_event_draws'] == c['retained'] == 9
    assert c['rng_streams'] == 36 and c['episode_action_draws'] == 9*24
    assert c['fork_action_draws'] == c['fork_transition_products'] == 9*25*8
    assert c['fork_boundaries'] == 9*25 and c['excluded_found'] == 0
    assert c['family_counts'] == {str(i):int((a['family'] == i).sum()) for i in range(4)}
    assert c['switch_draws'] == int((a['family'] >= 2).sum())
    assert c['valid_events'] == int(p['lengths'].sum()) == c['emission_constructions']
    assert c['transition_event_draws'] == c['valid_events']-9
    assert c['valid_events']+c['absorbed_suffix_events'] == 9*25
    assert c['found_episodes'] == int((p['observations'] == 4).any(1).sum())
    assert c['targets_privileged'] is True


def test_small_history_exact_likelihood_posterior_and_all_forks(monkeypatch):
    selected = iter((2,1,4))
    monkeypatch.setattr(w,'_sample',lambda _rng,_p:next(selected))
    result = w.generate(NAMESPACE,1,24)
    p,t,a = (result[name] for name in ('public','targets','audit'))
    transition,hazard,costs = physical()
    previous = Fraction(1)
    for boundary in range(3):
        labels = p['observations'][0,:boundary+1].tolist()
        actions = p['actions'][0,:boundary].tolist()
        eps = a['epsilons'][0,:boundary+1].tolist()
        expected = [.25,.25,.25,.25,0.] if not boundary else [
            float(path_oracle(actions,[*labels[:-1],event],eps)[0]/previous) for event in range(5)]
        np.testing.assert_allclose(t['probabilities'][0,boundary],expected,rtol=1e-12,atol=1e-14)
        previous,state = path_oracle(actions,labels,eps)
        np.testing.assert_allclose(t['post_states'][0,boundary],state,rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(t['post_costs'][0,boundary],costs@state,rtol=1e-12,atol=1e-14)
        for h,action in enumerate(t['fork_actions'][0,boundary]):
            state = np.array([sum(transition[action,j,k]*(1-hazard[action,j])*state[k] for k in range(8)) for j in range(8)])
            expected_cost = np.array([sum(costs[d,j]*state[j] for j in range(8)) for d in range(4)])
            np.testing.assert_allclose(t['fork_costs'][0,boundary,h],expected_cost,rtol=1e-11,atol=1e-14)
    assert p['lengths'].tolist() == [3]


def test_found_is_scored_once_then_zero_posterior_and_forks(monkeypatch):
    selected = iter((3,4))
    monkeypatch.setattr(w,'_sample',lambda _rng,_p:next(selected))
    result = w.generate(NAMESPACE,1)
    p,t,c = (result[name] for name in ('public','targets','counts'))
    assert p['lengths'].tolist() == [2] and np.all(p['observations'][0,1:] == 4)
    assert t['probabilities'][0,0,4] == 0 and 0 < t['probabilities'][0,1,4] < 1
    np.testing.assert_array_equal(t['probabilities'][0,2:],np.tile([0,0,0,0,1],(31,1)))
    for name in ('post_states','post_costs','fork_costs'):
        assert not t[name][0,1:].any()
    assert c['found_episodes'] == c['transition_event_draws'] == 1
    assert c['valid_events'] == 2 and c['absorbed_suffix_events'] == 31


def test_prefix_and_case_extensions_preserve_all_existing_boundaries():
    short,long = w.generate(NAMESPACE,2,24),w.generate(NAMESPACE,4,32)
    for group in ('public','targets','audit'):
        for key,value in short[group].items():
            expected = long[group][key][:2]
            if key == 'lengths':
                expected = np.minimum(expected,25)
            elif value.ndim > 1:
                expected = expected[:,:value.shape[1]]
            np.testing.assert_array_equal(value,expected)


def test_switch_event_predicts_with_new_noise_before_assimilation(monkeypatch):
    monkeypatch.setattr(w,'_schedule',lambda steps,_rng:(np.r_[np.full(8,.12),np.full(steps-7,.30)],8,2))
    monkeypatch.setattr(w,'_sample',lambda _rng,_p:0)
    result = w.generate(NAMESPACE,1,24)
    p,t = result['public'],result['targets']
    transition,hazard,_ = physical()
    action = p['actions'][0,7]
    state = t['post_states'][0,7]
    prior = np.array([sum(transition[action,j,k]*(1-hazard[action,j])*state[k] for k in range(8)) for j in range(8)])
    def law(error):
        return np.array([sum((1-error if j%4 == o else error/3)*prior[j] for j in range(8)) for o in range(4)])
    np.testing.assert_allclose(t['probabilities'][0,8,:4],law(.30),rtol=1e-12,atol=1e-14)
    assert np.max(np.abs(law(.30)-law(.12))) > 1e-3
    assert np.all(t['probabilities'][0,:,4] >= 0)
    np.testing.assert_allclose(t['probabilities'].sum(-1),1,rtol=0,atol=1e-12)
    np.testing.assert_allclose(t['post_states'].sum(-1),1,rtol=0,atol=1e-12)


def test_future_observations_do_not_change_precommitted_actions_or_prior_targets(monkeypatch):
    def forced(last):
        labels = iter((0,1,last,4))
        monkeypatch.setattr(w,'_sample',lambda _rng,_p:next(labels))
        return w.generate(NAMESPACE,1,24)
    left,right = forced(0),forced(2)
    np.testing.assert_array_equal(left['public']['actions'],right['public']['actions'])
    np.testing.assert_array_equal(left['targets']['fork_actions'],right['targets']['fork_actions'])
    np.testing.assert_array_equal(left['targets']['probabilities'][:,:3],right['targets']['probabilities'][:,:3])
    np.testing.assert_array_equal(left['targets']['fork_costs'][:,:2],right['targets']['fork_costs'][:,:2])
    assert not np.array_equal(left['targets']['post_states'][:,2],right['targets']['post_states'][:,2])


def test_ownership_public_privilege_boundary_and_global_rng():
    before = np.random.get_state()
    first,second = w.generate(NAMESPACE,2,24),w.generate(NAMESPACE,2,24)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1],after[1])
    assert set(first['public']) == {'actions','observations','lengths','case_ids'}
    assert set(first['targets']) == {'probabilities','post_states','post_costs','fork_actions','fork_costs'}
    assert set(first['audit']) == {'epsilons','switch_indices','family'}
    assert first['counts'] == second['counts']
    for group in ('public','targets','audit'):
        for name in first[group]:
            np.testing.assert_array_equal(first[group][name],second[group][name])
            assert not np.shares_memory(first[group][name],second[group][name])
    first['targets']['post_states'].fill(9.)
    assert second['targets']['post_states'].max() <= 1
    assert set(first['public']['case_ids']).isdisjoint(w.generate(NAMESPACE+1,2,24)['public']['case_ids'])


def test_zero_episodes_has_complete_empty_owned_schema():
    result = w.generate(NAMESPACE,0)
    assert result['public']['actions'].shape == (0,32)
    assert result['targets']['fork_costs'].shape == (0,33,8,4)
    assert result['audit']['family'].dtype == np.int64
    assert result['counts']['valid_events'] == result['counts']['retained'] == 0
    assert result['counts']['family_counts'] == dict.fromkeys(('0','1','2','3'),0)


@pytest.mark.parametrize('args',[(True,1,32),(-1,1,32),(2**32,1,32),
    (NAMESPACE,True,32),(NAMESPACE,-1,32),(NAMESPACE,1,23),(NAMESPACE,1,32.),(NAMESPACE,1,False)])
def test_rejects_invalid_configuration(args):
    with pytest.raises(ValueError):
        w.generate(*args)
