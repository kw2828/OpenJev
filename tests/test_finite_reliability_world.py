"""Small fabricated episodes and independent rational hidden-path oracles."""
from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pytest

from openjev.research import finite_reliability_world as w

NAMESPACE = 949001


def rational_fields(epsilon):
    error = Fraction(str(epsilon))
    odor = [[1-error if s % 4 == o else error/3 for s in range(8)] for o in range(4)]
    transition = []
    hazard = []
    for a in range(4):
        matrix = [[Fraction(0) for _ in range(8)] for _ in range(8)]
        for current in range(8):
            target = (current ^ 1,(current+1)%8,(current*2)%8+current//4,current ^ 4)[a]
            for destination in range(8):
                matrix[destination][current] = Fraction(1,400)+(Fraction(49,50) if destination == target else 0)
        transition.append(matrix)
        hazard.append([Fraction(1+((s//4) ^ (a%2)),200) for s in range(8)])
    costs = [[Fraction(-3,4) if action == ((s ^ (s//2)) % 4) else Fraction(1,4)
              for s in range(8)] for action in range(4)]
    return transition,odor,hazard,costs


def enumerate_history(actions,observations,epsilons):
    """Sum every physical-state path directly, including a terminal found event."""
    assert len(observations) == len(actions)+1 == len(epsilons)
    matrices,_,hazards,_ = rational_fields(.12)
    laws = [rational_fields(epsilon)[1] for epsilon in epsilons]
    posterior_mass = [Fraction(0) for _ in range(8)]
    for path in itertools.product(range(8),repeat=len(observations)):
        weight = Fraction(1,8)*laws[0][observations[0]][path[0]]
        for t,action in enumerate(actions,1):
            now,before = path[t],path[t-1]
            weight *= matrices[action][now][before]
            if observations[t] == 4:
                weight *= hazards[action][now]
            else:
                weight *= (1-hazards[action][now])*laws[t][observations[t]][now]
        posterior_mass[path[-1]] += weight
    evidence = sum(posterior_mass)
    posterior = [Fraction(0)]*8 if observations[-1] == 4 else [x/evidence for x in posterior_mass]
    return evidence,posterior


def rational_forks(state,actions):
    transitions,_,hazards,costs = rational_fields(.12)
    output = []
    for action in actions:
        state = [sum(transitions[action][j][k]*(1-hazards[action][j])*state[k] for k in range(8)) for j in range(8)]
        output.append([sum(costs[d][s]*state[s] for s in range(8)) for d in range(4)])
    return np.asarray(output,dtype=np.float64)


@pytest.mark.parametrize('epsilon',[.12,.30,.48,.60])
def test_emissions_and_uniform_reset_have_independent_exact_law(epsilon):
    _,expected,_,_ = rational_fields(epsilon)
    actual = w.emission(epsilon)
    np.testing.assert_allclose(actual,np.asarray(expected,np.float64),rtol=1e-14,atol=1e-15)
    np.testing.assert_allclose(actual.sum(0),1.,rtol=0,atol=1e-14)
    np.testing.assert_allclose(actual @ np.full(8,1/8),np.full(4,.25),rtol=0,atol=1e-14)
    actual.fill(99.)
    assert w.emission(epsilon).max() < 1


def test_generated_short_prefix_matches_exhaustive_hidden_paths_and_post_event_forks(monkeypatch):
    # Two ordinary transition observations then found. Remaining slots absorb.
    forced = iter((2,1,3,4))
    monkeypatch.setattr(w,'_sample',lambda _rng,_prob:int(next(forced)))
    output = w.generate(NAMESPACE,'base',1,24)
    public,targets = output['public'],output['targets']
    labels = public['observations'][0]
    actions = public['actions'][0]
    epsilon = output['audit']['epsilons'][0]
    previous_evidence = Fraction(1)
    for boundary in range(4):
        if boundary == 0:
            expected_p = [Fraction(1,4)]*4+[Fraction(0)]
        else:
            expected_p = []
            for next_event in range(5):
                evidence,_ = enumerate_history(actions[:boundary].tolist(),[*labels[:boundary].tolist(),next_event],epsilon[:boundary+1])
                expected_p.append(evidence/previous_evidence)
        np.testing.assert_allclose(targets['probabilities'][0,boundary],np.array(expected_p,np.float64),rtol=1e-12,atol=1e-14)
        evidence,state = enumerate_history(actions[:boundary].tolist(),labels[:boundary+1].tolist(),epsilon[:boundary+1])
        np.testing.assert_allclose(targets['post_states'][0,boundary],np.array(state,np.float64),rtol=1e-12,atol=1e-14)
        costs = rational_fields(.12)[3]
        expected_cost = [sum(costs[d][s]*state[s] for s in range(8)) for d in range(4)]
        np.testing.assert_allclose(targets['post_costs'][0,boundary],np.array(expected_cost,np.float64),rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(targets['fork_costs'][0,boundary],rational_forks(state,targets['fork_actions'][0,boundary]),rtol=1e-11,atol=1e-14)
        previous_evidence = evidence
    assert public['lengths'].tolist() == [4]
    assert output['counts']['retained'] == 1 and output['counts']['excluded_found'] == 0


def test_first_found_and_all_absorbing_slots_remain_in_public_schema(monkeypatch):
    assigned = iter((0,4))
    calls = []
    def sample(_rng,probability):
        calls.append(probability.copy())
        return next(assigned)
    monkeypatch.setattr(w,'_sample',sample)
    result = w.generate(NAMESPACE,'switch',1)
    p,t,c = result['public'],result['targets'],result['counts']
    assert set(p) == {'actions','observations','lengths','case_ids'}
    assert set(t) == {'probabilities','post_states','post_costs','fork_actions','fork_costs'}
    assert len(calls) == 2 and p['lengths'].tolist() == [2]
    assert np.all(p['observations'][0,1:] == 4)
    assert calls[0][4] == 0 and 0 < calls[1][4] < 1
    np.testing.assert_array_equal(t['probabilities'][0,2:],np.tile([0,0,0,0,1],(31,1)))
    assert not t['post_states'][0,1:].any()
    assert not t['post_costs'][0,1:].any()
    assert not t['fork_costs'][0,1:].any()
    assert c['found_episodes'] == c['transition_event_draws'] == 1
    assert c['valid_events'] == c['emission_constructions'] == 2
    assert c['absorbed_suffix_events'] == 31
    assert c['episode_action_draws'] == 32 and c['fork_action_draws'] == 33*8
    assert c['fork_transition_products'] == 33*8


def test_private_train_cycle_switch_boundary_and_evaluation_families(monkeypatch):
    monkeypatch.setattr(w,'_sample',lambda _rng,_p:0)
    train = w.generate(NAMESPACE,'train',8,24)
    assert train['audit']['family'].tolist() == list(w.TRAIN_FAMILIES)*2
    assert train['counts']['family_counts'] == dict.fromkeys(w.TRAIN_FAMILIES,2)
    for i,row in enumerate(train['audit']['epsilons']):
        low,high = ((.12,.12),(.48,.48),(.12,.48),(.48,.12))[i%4]
        changed = int(train['audit']['switch_indices'][i])
        if low == high:
            assert changed == -1 and np.all(row == low)
        else:
            assert 8 <= changed <= 24
            assert np.all(row[:changed] == low) and np.all(row[changed:] == high)
    assert train['counts']['switch_draws'] == 4
    for split,epsilon in [('base',.12),('shift',.30),('stress',.60)]:
        result = w.generate(NAMESPACE,split,1,24)
        assert np.all(result['audit']['epsilons'] == epsilon)
        assert result['audit']['switch_indices'].tolist() == [-1]
    switched = w.generate(NAMESPACE,'switch',2,24)['audit']
    assert switched['family'].tolist() == ['switch_012_to_030','switch_030_to_012']
    for i,(first,last) in enumerate(((.12,.30),(.30,.12))):
        boundary = switched['switch_indices'][i]
        assert np.all(switched['epsilons'][i,:boundary] == first)
        assert np.all(switched['epsilons'][i,boundary:] == last)


def test_varying_emission_leaves_physical_transport_found_and_costs_unchanged():
    physical = w.world(.12)
    state = np.arange(1,9,dtype=np.float64)/36
    transition,_,hazard,costs = rational_fields(.12)
    for action in range(4):
        expected_prior = [sum(transition[action][j][k]*(1-hazard[action][j])*Fraction(k+1,36) for k in range(8)) for j in range(8)]
        expected_found = sum(transition[action][j][k]*hazard[action][j]*Fraction(k+1,36) for j in range(8) for k in range(8))
        found_probabilities = []
        for epsilon in (.12,.30,.48,.60):
            branches,probabilities = w._predict(physical,state,action,w.emission(epsilon))
            np.testing.assert_allclose(branches.sum(0),np.array(expected_prior,np.float64),rtol=1e-12,atol=1e-14)
            assert probabilities[4] == pytest.approx(float(expected_found),rel=1e-12,abs=1e-14)
            found_probabilities.append(probabilities[4])
        assert len(set(found_probabilities)) == 1
    np.testing.assert_array_equal(physical['costs'],np.array(costs,np.float64))


def test_current_switch_event_uses_new_private_law_before_assimilation(monkeypatch):
    monkeypatch.setattr(w,'_sample',lambda _rng,_p:0)
    result = w.generate(NAMESPACE,'switch',1,24)
    t = int(result['audit']['switch_indices'][0])
    state = result['targets']['post_states'][0,t-1]
    action = int(result['public']['actions'][0,t-1])
    _,current = w._predict(w.world(.12),state,action,w.emission(.30))
    _,old = w._predict(w.world(.12),state,action,w.emission(.12))
    np.testing.assert_allclose(result['targets']['probabilities'][0,t],current,rtol=1e-12,atol=1e-14)
    assert np.max(np.abs(current-old)) > 1e-3


def test_reproducibility_global_rng_ownership_and_disjoint_namespace():
    before = np.random.get_state()
    first = w.generate(NAMESPACE,'train',3,24)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1],after[1])
    second = w.generate(NAMESPACE,'train',3,24)
    assert first['counts'] == second['counts']
    for group in ('public','targets','audit'):
        for key in first[group]:
            np.testing.assert_array_equal(first[group][key],second[group][key])
            assert not np.shares_memory(first[group][key],second[group][key])
    first['targets']['post_states'].fill(999)
    assert second['targets']['post_states'].max() <= 1
    distinct = w.generate(NAMESPACE+1,'train',3,24)
    assert set(second['public']['case_ids']).isdisjoint(distinct['public']['case_ids'])
    assert not np.array_equal(second['public']['actions'],distinct['public']['actions'])


def test_independent_upfront_streams_make_extensions_and_future_events_causal(monkeypatch):
    short = w.generate(NAMESPACE,'base',1,24)
    long = w.generate(NAMESPACE,'base',2,32)
    np.testing.assert_array_equal(short['public']['actions'],long['public']['actions'][:1,:24])
    np.testing.assert_array_equal(short['public']['observations'],long['public']['observations'][:1,:25])
    for name in ('probabilities','post_states','post_costs','fork_actions','fork_costs'):
        np.testing.assert_array_equal(short['targets'][name],long['targets'][name][:1,:25])
    def forced(last):
        labels = iter([0,1,2,last]+[0]*21)
        monkeypatch.setattr(w,'_sample',lambda _rng,_p:next(labels))
        return w.generate(NAMESPACE,'base',1,24)
    left,right = forced(0),forced(3)
    np.testing.assert_array_equal(left['public']['actions'],right['public']['actions'])
    np.testing.assert_array_equal(left['targets']['fork_actions'],right['targets']['fork_actions'])
    np.testing.assert_array_equal(left['targets']['probabilities'][:,:4],right['targets']['probabilities'][:,:4])
    np.testing.assert_array_equal(left['targets']['fork_costs'][:,:3],right['targets']['fork_costs'][:,:3])
    assert not np.array_equal(left['targets']['post_states'][:,3],right['targets']['post_states'][:,3])


def test_zero_episodes_and_full_shapes_dtypes():
    result = w.generate(NAMESPACE,'stress',0)
    assert result['public']['actions'].shape == (0,32)
    assert result['public']['observations'].shape == (0,33)
    assert result['public']['case_ids'].dtype == np.dtype('U64')
    assert result['audit']['family'].dtype == np.dtype('U32')
    assert result['targets']['fork_costs'].shape == (0,33,8,4)
    assert result['targets']['post_states'].dtype == np.float64
    assert result['counts']['valid_events'] == result['counts']['retained'] == 0


@pytest.mark.parametrize('args',[(True,'train',1,24),(-1,'base',1,24),(2**32,'base',1,24),
    (949001,'unknown',1,24),(949001,'base',True,24),(949001,'base',-1,24),
    (949001,'base',1,23),(949001,'base',1,True)])
def test_invalid_generation_contracts_fail(args):
    with pytest.raises(ValueError):
        w.generate(*args)


@pytest.mark.parametrize('epsilon',[0,1,float('nan'),float('inf'),True])
def test_invalid_emission_probability_rejected(epsilon):
    with pytest.raises(ValueError):
        w.emission(epsilon)
