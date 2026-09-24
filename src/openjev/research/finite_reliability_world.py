"""Public episode streams with a private, changing observation-reliability law.

The physical surviving transition A, found law and centered decision costs are
owned copies from frozen finite_observation_world.world(.12). Only the odor
emission matrix changes. Reset has a uniform state prior and no found hazard.
An event at index t>0 follows actions[t-1]. A switch index is the first event
using the new epsilon, including that boundary, never a public input.

Every attempted episode is retained, including the first found event. Lengths
count reset plus events through first found; observations after that length
are the absorbing label4, probabilities are onehot found, and all posterior
states/costs and blind fork costs are zero. Actions and all eight-action fork
blocks are allocated before any events, on streams separate from event draws.

Targets condition on the complete actual private noise path, including its
current epsilon. They are privileged evaluation references, not training
inputs, an attainable public-history Bayes floor, or a claim of calibrated
public predictions. Paired expected-cost comparisons can be computed under
that reference law, but methods observing only public history have less
information. Training is authorized to use only actual public-event NLL.
No hidden physical state is sampled or exposed: exact conditional mixtures
produce the correct event law for each prescribed noise path.
"""
from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np

from openjev.research.finite_observation_world import world

VERSION = 'finite-reliability-world-v1'
SPLIT_CODES = MappingProxyType({'train':0,'base':1,'shift':2,'switch':3,'stress':4})
TRAIN_FAMILIES = ('static_012','static_048','switch_012_to_048','switch_048_to_012')
FORK_HORIZON = 8
MASS_TOLERANCE = 1e-12


def require(ok,message):
    if not ok:
        raise ValueError(message)


def emission(epsilon):
    """Owned [odor,state] float64 law; low two state bits specify the clean odor."""
    require(type(epsilon) in (int,float) and math.isfinite(epsilon) and 0 < epsilon < 1,
        'finite emission error strictly between zero and one')
    result = np.full((4,8),epsilon/3,dtype=np.float64)
    states = np.arange(8,dtype=np.int64)
    result[states & 3,states] = 1-epsilon
    require(np.isfinite(result).all() and (result > 0).all()
        and np.max(np.abs(result.sum(0)-1)) <= MASS_TOLERANCE,'stochastic positive odor law')
    return result


def _mass(value,normalized,name):
    require(value.dtype == np.float64 and np.isfinite(value).all() and (value >= 0).all(),
        'finite nonnegative float64 '+name)
    total = float(value.sum(dtype=np.float64))
    require(math.isfinite(total) and total <= 1+MASS_TOLERANCE
        and (not normalized or abs(total-1) <= MASS_TOLERANCE),'bounded probability mass: '+name)
    return total


def _sample(rng,probabilities):
    total = _mass(probabilities,True,'event distribution')
    return int(rng.choice(5,p=probabilities/total))


def _schedule(split,index,steps,rng):
    if split == 'train':
        family = TRAIN_FAMILIES[index % 4]
        first,last = ((.12,.12),(.48,.48),(.12,.48),(.48,.12))[index % 4]
    elif split == 'switch':
        first,last = (.12,.30) if index % 2 == 0 else (.30,.12)
        family = 'switch_012_to_030' if index % 2 == 0 else 'switch_030_to_012'
    else:
        first = last = {'base':.12,'shift':.30,'stress':.60}[split]
        family = {'base':'static_012','shift':'static_030','stress':'static_060'}[split]
    changed = first != last
    switch = int(rng.integers(8,25)) if changed else -1
    epsilons = np.full(steps+1,first,dtype=np.float64)
    if changed:
        epsilons[switch:] = last
    return epsilons,switch,family


def _predict(physical,state,action,odor_law):
    """All five next-event probabilities precede that event's assimilation."""
    _mass(state,True,'active conditional state')
    prior = physical['A'][action] @ state
    branches = odor_law * prior[None,:]
    probability = np.concatenate((branches.sum(1,dtype=np.float64),
        np.array([physical['found'][action] @ state],dtype=np.float64)))
    _mass(probability,True,'current path-conditional prediction')
    return branches,probability


def _fork_costs(physical,state,actions):
    """Unconditional surviving-mass propagation from a post-observation state."""
    require(actions.dtype == np.int64 and actions.shape == (FORK_HORIZON,)
        and ((actions >= 0) & (actions < 4)).all(),'eight committed fork actions')
    _mass(state,False,'fork boundary state')
    result = np.zeros((FORK_HORIZON,4),dtype=np.float64)
    current = state.copy()
    for h,action in enumerate(actions):
        current = physical['A'][int(action)] @ current
        _mass(current,False,'unconditional surviving fork state')
        result[h] = physical['costs'] @ current
    return result


def generate(namespace:int,split:str,episodes:int,steps:int=32):
    """Generate owned arrays with public, targets, audit and counts containers.

    Public: actions i64[N,T], observations i64[N,T+1], lengths i64[N],
    case_ids U64[N]. Targets: probabilities f64[N,T+1,5] before the event;
    post_states f64[N,T+1,8], post_costs f64[N,T+1,4] after that event;
    fork_actions i64[N,T+1,8], fork_costs f64[N,T+1,8,4]. Audit:
    epsilons f64[N,T+1], switch_indices i64[N] (-1 static), family U32[N].

    Each SeedSequence([namespace, split_code, case_index]) spawns four PCG64
    generators in fixed order: episode actions, fork actions, private schedule,
    events. All episode and fork action blocks are drawn upfront. Episode
    count and step extensions preserve earlier cases and their existing event
    boundaries. There is no global RNG mutation, file I/O, model or optimizer.
    """
    require(type(namespace) is int and 0 <= namespace < 2**32,'uint32 namespace')
    require(type(split) is str and split in SPLIT_CODES,'declared split name')
    require(type(episodes) is int and 0 <= episodes < 2**32,'uint32 episode count')
    require(type(steps) is int and 24 <= steps < 2**32,'episode steps at least24')
    physical = world(.12)
    public = {'actions':np.zeros((episodes,steps),dtype=np.int64),
        'observations':np.full((episodes,steps+1),4,dtype=np.int64),
        'lengths':np.full(episodes,steps+1,dtype=np.int64),
        'case_ids':np.array([f'ns{namespace}-split{SPLIT_CODES[split]}-case{i:010d}' for i in range(episodes)],dtype='U64')}
    targets = {'probabilities':np.zeros((episodes,steps+1,5),dtype=np.float64),
        'post_states':np.zeros((episodes,steps+1,8),dtype=np.float64),
        'post_costs':np.zeros((episodes,steps+1,4),dtype=np.float64),
        'fork_actions':np.zeros((episodes,steps+1,FORK_HORIZON),dtype=np.int64),
        'fork_costs':np.zeros((episodes,steps+1,FORK_HORIZON,4),dtype=np.float64)}
    audit = {'epsilons':np.zeros((episodes,steps+1),dtype=np.float64),
        'switch_indices':np.full(episodes,-1,dtype=np.int64),'family':np.empty(episodes,dtype='U32')}
    counts = {'version':VERSION,'namespace':namespace,'split':split,'split_code':SPLIT_CODES[split],
        'episodes':episodes,'steps':steps,'retained':episodes,'excluded_found':0,
        'rng_case_sequences':episodes,'rng_streams':4*episodes,'episode_action_draws':episodes*steps,
        'fork_action_draws':episodes*(steps+1)*FORK_HORIZON,'switch_draws':0,
        'reset_event_draws':episodes,'transition_event_draws':0,'found_episodes':0,
        'valid_events':0,'absorbed_suffix_events':0,'emission_constructions':0,
        'fork_boundaries':episodes*(steps+1),'fork_transition_products':episodes*(steps+1)*FORK_HORIZON,
        'family_counts':{},'targets_privileged':True,'training_inputs':'public only',
        'oracle_scope':'Risk reference conditional on the private realized noise path; not an attainable public-history Bayes floor or public calibration target.'}
    for case in range(episodes):
        seeds = np.random.SeedSequence([namespace,SPLIT_CODES[split],case]).spawn(4)
        action_rng,fork_rng,schedule_rng,event_rng = (np.random.Generator(np.random.PCG64(seed)) for seed in seeds)
        public['actions'][case] = action_rng.integers(0,4,size=steps,dtype=np.int64)
        targets['fork_actions'][case] = fork_rng.integers(0,4,size=(steps+1,FORK_HORIZON),dtype=np.int64)
        epsilons,switch,family = _schedule(split,case,steps,schedule_rng)
        audit['epsilons'][case],audit['switch_indices'][case],audit['family'][case] = epsilons,switch,family
        counts['switch_draws'] += int(switch >= 0)
        counts['family_counts'][family] = counts['family_counts'].get(family,0)+1
        state = np.full(8,1/8,dtype=np.float64)
        absorbed = False
        for t in range(steps+1):
            if absorbed:
                targets['probabilities'][case,t,4] = 1
            else:
                odor_law = emission(float(epsilons[t]))
                counts['emission_constructions'] += 1
                if t == 0:
                    branches = odor_law * state[None,:]
                    probability = np.concatenate((branches.sum(1,dtype=np.float64),np.zeros(1,dtype=np.float64)))
                else:
                    branches,probability = _predict(physical,state,int(public['actions'][case,t-1]),odor_law)
                    counts['transition_event_draws'] += 1
                targets['probabilities'][case,t] = probability
                observed = _sample(event_rng,probability)
                require(0 <= observed <= 4 and (t > 0 or observed < 4),'valid reset or transition event')
                public['observations'][case,t] = observed
                if observed == 4:
                    absorbed = True
                    state = np.zeros(8,dtype=np.float64)
                    public['lengths'][case] = t+1
                    counts['found_episodes'] += 1
                else:
                    evidence = float(probability[observed])
                    require(evidence > 0,'strictly positive observed evidence')
                    state = branches[observed]/evidence
                    _mass(state,True,'post-observation conditional state')
                targets['post_states'][case,t] = state
                targets['post_costs'][case,t] = physical['costs'] @ state
            targets['fork_costs'][case,t] = _fork_costs(physical,state,targets['fork_actions'][case,t])
        counts['valid_events'] += int(public['lengths'][case])
        counts['absorbed_suffix_events'] += steps+1-int(public['lengths'][case])
    require(counts['valid_events'] == counts['reset_event_draws']+counts['transition_event_draws']
        and counts['valid_events']+counts['absorbed_suffix_events'] == episodes*(steps+1)
        and counts['emission_constructions'] == counts['valid_events']
        and sum(counts['family_counts'].values()) == episodes,'complete all-attempt episode accounting')
    require(all(np.isfinite(value).all() for key,value in public.items() if key != 'case_ids')
        and all(np.isfinite(value).all() for value in targets.values())
        and np.isfinite(audit['epsilons']).all(),'finite complete reliability-world output')
    return {'public':public,'targets':targets,'audit':audit,'counts':counts}
