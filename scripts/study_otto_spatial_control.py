"""Fixed sixteen-arm autonomous comparison of all completed spatial heads; no fitting."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = 'scripts/study_otto_bellman_control.py'
BASE_PIN = '055aadac7349ccc90da6fd21c28b1bb61f2a0b7205a64f8f2444b960d87beab6'
SCALAR = 'scripts/study_otto_spatial.py'
SCALAR_PIN = '548c21ab961400313e8ff77697ea8a4ee78bffd0be29227506d0236c0cbbd212'


def pinned(name, pin, module_name):
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != pin:
        raise ValueError('unchanged inherited source '+name)
    spec = importlib.util.spec_from_file_location(module_name, ROOT/name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


B = pinned(BASE, BASE_PIN, '_spatial_control_base')
S = pinned(SCALAR, SCALAR_PIN, '_spatial_control_scalar')
P = B.P
require, sha, read, write, regular, descriptor, closed = B.require, B.sha, B.read, B.write, B.regular, B.descriptor, B.closed
VERSION = 'otto-spatial-control-v1'
SEEDS, FAMILIES = (10101, 10102, 10103), ('spatial', 'neighbor_free', 'cnn', 'dense128', 'statistics')
ARMS = tuple(f'{kind}@{seed}' for seed in SEEDS for kind in FAMILIES)+('analytic_inbounds',)
REGIMES = {'lambda3': 3., 'lambda4': 4., 'lambda5': 5.}
FIRST = {'lambda3': 16100001, 'lambda4': 16200001, 'lambda5': 16300001}
TEMPLATE_FIRST, CASES, HORIZON = 16500001, 24, 2188
TIMES, METRICS = P.TIMES, P.METRICS
NEW = ('scripts/study_otto_spatial_control.py', 'scripts/freeze_otto_spatial_control.py',
       'scripts/audit_otto_spatial_control.py', 'tests/test_otto_spatial_control.py',
       'tests/test_audit_otto_spatial_control.py', 'research/otto-spatial-control-protocol.md',
       'scripts/review_otto_spatial_seeds.py', 'tests/test_review_otto_spatial_seeds.py')
AUDIT_DEPENDENCIES = {
    'scripts/audit_otto_conditioning_control.py':'094fd41e7b9c34d0df580c1d99aef0e2f1bc2c83997afad36cd8c15f14794e1b',
    'scripts/audit_otto_conditioning.py':'28259542e97940c3dd9482dae4089a878dd946ca22c8adf352ed70aff6a1e51f',
}


INPUTS = {
    'spatial_plan': ('output/otto-spatial-study-v1/plan-01.json', '2f204b5f3aa98f7aa54e8537ed6df16110596944f54f4b7f63fd39053d9a9a62'),
    'spatial_receipt': ('output/otto-spatial-study-v1/run-01/receipt.json', '2f2ed04be8b2ba4be8eeb9a3c756035900a91dda31f580c3a06f5c1bc1961c49'),
    'spatial_terminal': ('output/otto-spatial-study-v1/process-01.terminal.json', '6b7d5254a6ec8d2e0c3f89fe7d4b4f155229d3e5db0bb68b08e4b38b731b1178'),
    'spatial_audit': ('output/otto-spatial-study-v1/audit-01/receipt.json', '17c2094ca5a95fec2299a51bd50f0b8956e97da76787b365dd538e4cf3b9736a'),
    'scalar_receipt': ('output/otto-return-value-v1/run-01/receipt.json', '965cc3eee7c77ea64700137bda2961af1a259f1a98d15b54041f9530a7f66ac3'),
}
POSITIONS = ((26,26), (0,0), (52,52))
BELIEFS = ('asymmetric','point_successor','zero','subfloor')
ATOL, RTOL = 1e-8, 1e-10


def limits(mode):
    require(mode in ('qualify','study'), 'known mode')
    return {'native_seconds':600 if mode == 'qualify' else 21600,
            'rss_bytes':(4 if mode == 'qualify' else 8)*1024**3,
            'output_bytes':256*1024**2 if mode == 'qualify' else 12*1024**3,
            'native_steps':0 if mode == 'qualify' else 1152*HORIZON,
            'native_resets':0 if mode == 'qualify' else 1155,
            'optimizer_updates':0,'target_readouts':0,
            'model_load':15,'parity_restore':15 if mode == 'qualify' else 0,
            'parity_numpy_forward':780 if mode == 'qualify' else 0,
            'parity_torch_forward':780 if mode == 'qualify' else 0,
            'value_forward':0 if mode == 'qualify' else 1080*HORIZON,
            'analytic_choose':0 if mode == 'qualify' else 72*HORIZON}


def audit_limits(mode):
    return {'native_seconds':600 if mode == 'qualify' else 21600,
            'rss_bytes':4*1024**3,'output_bytes':256*1024**2 if mode == 'qualify' else 1024**3}


def expected_calls(mode):
    cap = limits(mode)
    keys = ('model_load','parity_restore','parity_numpy_forward','parity_torch_forward') if mode == 'qualify' else ('model_load','native_reset','native_step','value_forward','analytic_choose')
    mapped = {'native_reset':'native_resets','native_step':'native_steps'}
    return {k:cap[mapped.get(k,k)] for k in keys}


def qualification_ids():
    rows = [{'tuple_id':f'{split}:{i}','source':'cache','split':split,'row_index':i}
            for split in ('train','valid') for i in range(8)]
    rows += [{'tuple_id':f'{regime}:{label}:{kind}','source':'synthetic','regime':regime,
              'position':list(position),'belief_kind':kind}
             for regime in REGIMES for label,position in zip(('center','lower','upper'),POSITIONS,strict=True) for kind in BELIEFS]
    return rows


def synthetic_tuple(regime, position, kind, np):
    require(regime in REGIMES and tuple(position) in POSITIONS and kind in BELIEFS, 'fixed synthetic tuple')
    x,y = np.indices((53,53),dtype=np.int64)
    raw = 1+((17*x+29*y+7*x*y)%97)
    raw[tuple(position)] = 0
    field = raw.astype(np.float64)
    field /= field.sum(dtype=np.float64)
    allowed = [a for a in range(4) if 0 <= position[a//2]+2*(a%2)-1 < 53]
    if kind == 'subfloor':
        field *= np.float64(1e-12)
    elif kind in ('zero','point_successor'):
        field.fill(0.)
        if kind == 'point_successor':
            point = list(position); action = allowed[0]
            point[action//2] += 2*(action%2)-1
            field[tuple(point)] = 1.
    return field, np.asarray(position,dtype=np.int64), float(REGIMES[regime]), allowed


def configuration(mode):
    require(mode in ('qualify','study'), 'known mode')
    return {'arms':list(ARMS),'candidate':'spatial','cases_per_regime':0 if mode == 'qualify' else CASES,
            'horizon':HORIZON,'evaluation_first_seeds':FIRST,'template_first_seed':TEMPLATE_FIRST,
            'initial_hit':'1+case%3','block':'case//3','arm_rotation':'(regime_index*24+case)%16',
            'qualification_tuples':qualification_ids(),
            'synthetic_formula':'int64 1+((17*x+29*y+7*x*y)%97); current cell zero; float64 normalization; subfloor1e-12; point at first eligible successor',
            'parity_atol':ATOL,'parity_rtol':RTOL,'parity_units':'physical 64*normalized value and costs; exact eligible first action',
            'counts':'qualification exact; study maxima except15 loads/1155 resets',
            'amortization_searches':[1,100,10000],'external_model_calls':0,'scalar_gate_required':False}


def payload_names(mode):
    require(mode in ('qualify','study'), 'known mode')
    names = {'started.json','runtime.json','inference-setup.json','work-contexts.jsonl','work.jsonl','summary.json','serialization-projection.json'}
    return names | ({'preparation.json','qualification-arrays.npz','qualification-tuples.jsonl','parity.jsonl'} if mode == 'qualify' else
                    {'native-setup.json','eval-transitions.jsonl','eval-episodes.jsonl','evaluation.jsonl'})


def serialization_fixture():
    """Finite maximal-width fabricated records, using the actual JSON writer schema.

    Float values are serialization sentinels, not physically possible outcomes.
    Every scalar uses maximal signed float64 spelling; lengths dominate actual
    bounded IDs/coordinates/counters. No source, model, cache or simulator call.
    """
    huge = -1.7976931348623157e308
    arm = max(ARMS,key=len); episode = f'eval:lambda5:16300024:{arm}'
    public = {'position':[52,52],'hit':-2,'done':False,'step':2188,'valid_actions':[0,1,2,3]}
    posterior = {'sha256':'f'*64,'mass':huge}
    identity = {'episode_id':episode,'stage':'eval','regime':'lambda5','seed':16300024,
                'initial_hit':3,'arm':arm,'block':7}
    step = {'kind':'step','episode_id':episode,'step':2188,'action':3,'costs':[huge]*4,
        'allowed_actions':[0,1,2,3],'raw_masses':[[huge]*4 for _ in range(4)],
        'weights':[[huge]*4 for _ in range(4)],'values':[huge]*16,'public':public,
        'posterior_before':posterior,'posterior_after':posterior,
        'choose_seconds':huge,'choose_instrumented_seconds':huge,'choose_excluded_io_seconds':huge,
        'update_seconds':huge,'environment_seconds':huge,'native_p_end':huge}
    reset = {'kind':'reset',**identity,'public':public,'posterior_after':posterior,'source_evaluation_only':[52,52]}
    draw = {'channel':'initial','index':2188,'uniform':huge,'selected_index':2808,'cdf_mass':huge}
    episode_row = {**identity,'steps':2188,'found':False,'updates':2188,'blocked_steps':0,
        'init_seconds':huge,'choose_seconds':huge,'update_seconds':huge,'setup_allocation_seconds':huge,
        'controller_seconds':huge,'choose_instrumented_seconds':huge,'choose_excluded_io_seconds':huge,
        'environment_seconds':huge,'state_bytes':99999999,
        'storage':{'public_actor':{'immutable_array_bytes':99999999,'mutable_array_bytes':99999999,'scope':'X'*1024},
                   'head':{'parameter_array_bytes':99999999,'baseline_array_bytes':8,'mutable_array_bytes':0,'scope':'X'*1024},
                   'branch_workspace':'16*105*105 float64 centered u and z plus finite branch/features workspace, transient'},
        'zero_mass_decisions':2188,'final_posterior_mass':huge,'last256_lag2_matches':254,'last256_lag2_pairs':254,
        'distinct_preaction_positions':2188,'source_evaluation_only':[52,52],'draws_evaluation_only':[],
        'final_public':public,'final_update_assimilated':True}
    return {'step':step,'reset':reset,'draw':draw,'episode':episode_row,
            'attempt':[99999999,0,'parity_numpy_forward',9999,2188],
            'return':[99999999,1,'parity_numpy_forward',huge,huge,huge],
            'context':{'id':9999,'context':{'phase':'inference_setup','episode':episode,'arm':arm,'regime':'lambda5'}}}


def serialization_projection():
    fixtures = serialization_fixture()
    widths = {k:len((json.dumps(v,separators=(',',':'),allow_nan=False)+'\n').encode()) for k,v in fixtures.items()}
    steps,episodes = 1152*HORIZON,1152
    # Both learned and analytic decisions have exactly two journaled operations.
    call_count = 2*steps+1155+15
    projected = {
        'steps':steps*widths['step'], 'resets':episodes*widths['reset'],
        'operation_journal':call_count*(widths['attempt']+widths['return']),
        'contexts':(episodes+15+3)*widths['context'],
        'episode_records':2*episodes*widths['episode'],
        'duplicated_draws':2*(steps+2*episodes)*(widths['draw']+1),
        'fixed_metadata_allowance':64*1024**2,
    }
    total = sum(projected.values())
    return {'version':'otto-spatial-control-serialization-v1','serialized_fixture_bytes':widths,
            'components':projected,'projected_bytes':total,'worker_output_cap':12*1024**3,
            'within_worker_cap':total <= 12*1024**3,'native_step_cap':steps,
            'learned_forward_cap':1080*HORIZON,'episode_count':episodes,
            'qualification_shared_array_bytes':52*(53*53*8+2*8+8+4+2*16*105*105*8+2*4*4*8+16*2*8),
            'scope':'Prospective fabricated JSON byte envelope, not observed sizes; all transition fields, four work events per decision, reset/context/setup, and draw evidence duplicated in both episode files. Hard runtime output cap remains authoritative.'}


def audit_projection(mode, sources):
    """Only import the pure projection after all its inherited modules are bound."""
    for name,pin in AUDIT_DEPENDENCIES.items():
        require(sources[name] == pin and sha(regular(name)) == pin, 'fixed audit projection dependency')
    name = 'scripts/audit_otto_spatial_control.py'
    helper = pinned(name,sources[name],'_spatial_control_audit_projection')
    return helper.journal_projection(mode)


def authenticate(args):
    require(args.plan.is_absolute() and not any(p.is_symlink() for p in (args.plan,*args.plan.parents))
            and sha(args.plan) == args.plan_sha256, 'external plan pin before decoding')
    plan = read(args.plan); mode = plan['mode']
    require(plan['version'] == VERSION and plan['status'] == 'frozen_before_execution'
            and plan['configuration'] == configuration(mode) and plan['limits'] == limits(mode)
            and plan['expected_calls'] == expected_calls(mode) and plan['independent_audit_limits'] == audit_limits(mode)
            and plan['serialization_projection'] == serialization_projection(), 'fixed complete contract')
    require(plan['serialization_projection']['within_worker_cap'], 'prospective serialized storage allocation')
    roles = {*INPUTS,'seed_review',*('kernel_'+r for r in REGIMES),*('head_'+a.replace('@','_') for a in ARMS[:-1])}
    if mode == 'qualify': roles |= {'train_data','valid_data','train_rows','valid_rows'}
    require(set(plan['inputs']) == roles, 'exact input roles')
    paths = {}
    for role,item in plan['inputs'].items():
        paths[role] = regular(item['path'])
        require(descriptor(paths[role]) == {k:item[k] for k in ('bytes','sha256')}, 'bound input '+role)
    for role,(name,pin) in INPUTS.items():
        require(paths[role] == regular(name) and sha(paths[role]) == pin, 'fixed original evidence')
    review = read(paths['seed_review'])
    require(review['status'] == 'completed' and review['passed'] is True and review['overlap'] == review['overlaps'] == []
            and review['historical_declaration_matches'] == [] and review['proposed_unique_seeds'] == 75
            and review['proposed_ranges_inclusive'] == {**{'eval_'+r:[n,n+23] for r,n in FIRST.items()},'templates':[TEMPLATE_FIRST,TEMPLATE_FIRST+2]}, 'fresh fixed seed ranges')
    prior = S.authenticate(SimpleNamespace(plan=paths['spatial_plan'],plan_sha256=sha(paths['spatial_plan'])))
    worker,audit = closed(paths['spatial_receipt']),closed(paths['spatial_audit'])
    require(worker['mode'] == 'study' and worker['completed_fits'] == 15 and worker['pending'] == []
            and worker['parity_passed'] is True and worker['initial_pairing_passed'] is True
            and worker['plan_sha256'] == sha(paths['spatial_plan']) and worker['sources'] == prior['sources']
            and worker['inputs'] == prior['inputs'] and set(worker['files']) == S.payload_names(), 'completed all15 fixed spatial fits')
    require({k:v['returned'] for k,v in worker['calls'].items()} == S.expected_calls()
            and all(v['attempted'] == v['returned'] for v in worker['calls'].values()), 'complete prior fitting work')
    S.C.terminal_identity(paths['spatial_receipt'],paths['spatial_plan'],paths['spatial_terminal'],SCALAR)
    require(audit['agreement'] is True and audit['worker_sha256'] == sha(paths['spatial_receipt'])
            and audit['plan_sha256'] == sha(paths['spatial_plan']) and audit['terminal_sha256'] == sha(paths['spatial_terminal'])
            and audit['source']['sha256'] == prior['sources']['scripts/audit_otto_spatial_study.py'], 'completed prior independent audit')
    union = dict(prior['sources'])
    for name,pin in AUDIT_DEPENDENCIES.items():
        require(sha(regular(name)) == pin and (name not in union or union[name] == pin), 'unchanged audit dependency')
        union[name] = pin
    for name in NEW:
        pin = sha(regular(name)); require(name not in union or union[name] == pin, 'additive sources only'); union[name] = pin
    require(plan['sources'] == union and plan['sources'][BASE] == BASE_PIN and plan['sources'][SCALAR] == SCALAR_PIN, 'exact unchanged source union')
    for name,pin in plan['sources'].items(): require(sha(regular(name)) == pin, 'unchanged source '+name)
    require(plan['audit_serialization_projection'] == audit_projection(mode,plan['sources'])
            and plan['audit_study_serialization_projection'] == audit_projection('study',plan['sources']),
            'prospective independent audit serialization allocation')
    for key in ('python_executable','python_version','all_distributions'): require(plan[key] == prior[key], 'same qualified runtime')
    original = read(paths['scalar_receipt'])
    require(original['status'] == 'completed', 'completed original native kernel lineage')
    for regime in REGIMES:
        name = f'kernel-{regime}.npz'
        require(paths['kernel_'+regime] == paths['scalar_receipt'].parent/name and descriptor(paths['kernel_'+regime]) == original['files'][name], 'exact native kernel')
    for arm in ARMS[:-1]:
        name = 'final-'+arm.replace('@','-')+'.npz'
        require(paths['head_'+arm.replace('@','_')] == paths['spatial_receipt'].parent/name
                and descriptor(paths['head_'+arm.replace('@','_')]) == worker['files'][name], 'every fixed final checkpoint')
    if mode == 'qualify':
        require(plan['qualification'] == {}, 'first deployment qualification')
        for role in ('train_data','valid_data','train_rows','valid_rows'): require(plan['inputs'][role] == prior['inputs'][role], 'same historical qualification prefixes')
    else:
        require(set(plan['qualification']) == {'plan','receipt','terminal','audit'}, 'complete deployed qualification')
        qp = {}
        for role,item in plan['qualification'].items():
            qp[role] = regular(item['path']); require(descriptor(qp[role]) == {k:item[k] for k in ('bytes','sha256')}, 'qualification pin')
        qplan = authenticate(SimpleNamespace(plan=qp['plan'],plan_sha256=sha(qp['plan'])))
        qworker,qaudit = closed(qp['receipt']),closed(qp['audit'])
        require(qplan['mode'] == qworker['mode'] == 'qualify' and qworker['parity_passed'] is True and qworker['completed_episodes'] == 0
                and qworker['pending'] == [] and qworker['plan_sha256'] == sha(qp['plan']) and qworker['sources'] == qplan['sources']
                and qworker['inputs'] == qplan['inputs'] and set(qworker['files']) == payload_names('qualify')
                and {k:v['returned'] for k,v in qworker['calls'].items()} == expected_calls('qualify')
                and all(v['attempted'] == v['returned'] for v in qworker['calls'].values()), 'complete exact all-head qualification')
        S.C.terminal_identity(qp['receipt'],qp['plan'],qp['terminal'],'scripts/study_otto_spatial_control.py')
        require(qaudit['agreement'] is True and qaudit['worker_sha256'] == sha(qp['receipt'])
                and qaudit['plan_sha256'] == sha(qp['plan']) and qaudit['terminal_sha256'] == sha(qp['terminal'])
                and qaudit['source']['sha256'] == qplan['sources']['scripts/audit_otto_spatial_control.py'], 'independent deployed qualification audit')
        require(plan['sources'] == qplan['sources'] and all(plan['inputs'][r] == qplan['inputs'][r] for r in roles), 'same qualified all-head deployment')
    return plan


def cohort():
    for ri,regime in enumerate(REGIMES):
        for case in range(CASES):
            offset = (ri*CASES+case)%len(ARMS)
            for arm in ARMS[offset:]+ARMS[:offset]:
                yield regime,FIRST[regime]+case,case//3,1+case%3,arm


def evaluation_order(first):
    require(first == FIRST, 'fixed fresh cohort')
    return cohort()


def aggregate(rows, mixtures):
    require([(r['regime'],r['seed'],r['block'],r['initial_hit'],r['arm']) for r in rows] == list(cohort()), 'all1152 ordered episodes')
    for r in rows:
        require(type(r['found']) is bool and type(r['steps']) is int and 1 <= r['steps'] <= HORIZON
                and (r['found'] or r['steps'] == HORIZON) and r['updates'] == r['steps']
                and r['final_update_assimilated'] is True and r['blocked_steps'] == 0, 'complete found/censored paths')
        require(all(math.isfinite(r[m]) and r[m] >= 0 for m in METRICS)
                and abs(r['controller_seconds']-math.fsum(r[k] for k in TIMES)) <= 1e-9, 'finite complete controller costs')
    regimes,competence,controls,improvement = {},[],[],[]
    def add(group,name,value,threshold,passes): group.append({'name':name,'value':value,'threshold':threshold,'passes':bool(passes)})
    for regime in REGIMES:
        weights = {int(k):v for k,v in mixtures[regime].items()}
        require(set(weights) == {1,2,3} and all(math.isfinite(v) and 0 < v < 1 for v in weights.values()) and abs(math.fsum(weights.values())-1) <= 1e-12, 'positive normalized hit mixture')
        selected = [r for r in rows if r['regime'] == regime]
        def weighted(subset,metric,weights=weights):
            return math.fsum(weights[h]*math.fsum(float(r[metric]) for r in subset if r['initial_hit'] == h)/sum(r['initial_hit'] == h for r in subset) for h in (1,2,3))
        means = {a:{m:weighted([r for r in selected if r['arm'] == a],m) for m in METRICS} for a in ARMS}
        blocks = [{a:weighted([r for r in selected if r['arm'] == a and r['block'] == b],'steps') for a in ARMS} for b in range(8)]
        family = {f:{m:math.fsum(means[f'{f}@{seed}'][m] for seed in SEEDS)/3 for m in METRICS} for f in FAMILIES}
        teacher,candidate = means['analytic_inbounds'],family['spatial']
        for kind in FAMILIES:
            group = competence if kind == 'spatial' else controls
            for seed in SEEDS:
                value = means[f'{kind}@{seed}']
                add(group,f'{regime}.{kind}.{seed}.success',value['found'],.95,value['found'] >= .95)
                add(group,f'{regime}.{kind}.{seed}.moves',value['steps'],1.05*teacher['steps'],value['steps'] <= 1.05*teacher['steps'])
        for kind in FAMILIES[1:]:
            ref = family[kind]
            positive = sum(math.fsum(b[f'{kind}@{seed}']-b[f'spatial@{seed}'] for seed in SEEDS)/3 > 0 for b in blocks)
            add(improvement,f'{regime}.{kind}.success',candidate['found'],ref['found'],candidate['found'] >= ref['found'])
            add(improvement,f'{regime}.{kind}.moves',candidate['steps'],.95*ref['steps'],candidate['steps'] <= .95*ref['steps'])
            add(improvement,f'{regime}.{kind}.positive_blocks',positive,6,positive >= 6)
            add(improvement,f'{regime}.{kind}.cost',candidate['controller_seconds'],ref['controller_seconds'],candidate['controller_seconds'] <= ref['controller_seconds'])
        regimes[regime] = {'weights':weights,'means':means,'family_means':family,'blocks':blocks,
            'strata':{str(h):{a:{m:math.fsum(float(r[m]) for r in selected if r['arm'] == a and r['initial_hit'] == h)/8 for m in METRICS} for a in ARMS} for h in (1,2,3)},
            'raw_counts':{a:{'found':sum(r['found'] for r in selected if r['arm'] == a),'episodes':24} for a in ARMS}}
    require((len(competence),len(improvement),len(controls)) == (18,48,72), 'all66 candidate and72 descriptive conditions')
    return {'version':VERSION,'episodes':len(rows),'paired_cases':72,'regimes':regimes,'competence_checks':competence,
            'improvement_checks':improvement,'control_competence_checks':controls,
            'pilot_continuation':all(r['passes'] for r in competence+improvement),'learned_architecture_advantage_established':False}


def summarize(rows,mixtures,first):
    require(first == FIRST, 'fixed fresh cohort')
    return aggregate(rows,mixtures)


class Run(B.Run):
    def __init__(self, args):
        super().__init__(args)
        self.receipt = {'version': VERSION, 'status': 'started', 'external_model_calls': 0,
                        'completed_episodes': 0, 'completed_fits': 0, 'parity_passed': False}

    def begin(self, channel, *, check=True):
        require(channel in expected_calls(self.plan['mode']), 'declared operation only')
        require(self.calls.get(channel,{}).get('attempted',0) < expected_calls(self.plan['mode'])[channel], 'operation allocation before invocation')
        return super().begin(channel, check=check)

    def bind(self):
        require(sha(ROOT/P.CLOCK) == P.CLOCK_PIN, 'qualified clock')
        self.clock = B.load(ROOT/P.CLOCK, '_spatial_control_clock').SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns()-self.start < 5*10**9, 'supervision missing')
            time.sleep(.01)
        self.launch, self.plan = read(self.args.supervision), authenticate(self.args)
        launch = self.launch
        command = list(launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and launch['pid'] == os.getpid()
                and launch['pgid'] == os.getpgrp() and launch['parent_pid'] == os.getppid()
                and launch['cwd'] == str(ROOT) == str(Path.cwd()), 'actual supervised process')
        require(launch['clock_backend'] == self.clock.backend and launch['started_ns'] <= self.start < launch['deadline_ns']
                and launch['cap_seconds'] == self.plan['limits']['native_seconds']
                and launch['deadline_ns'] == launch['started_ns']+launch['cap_seconds']*10**9
                and launch['watchdog_sha256'] == self.plan['sources']['scripts/supervise_dialogue_observation_v2.py']
                and launch['clock_source_sha256'] == P.CLOCK_PIN, 'shared bounded supervisor')
        self.receipt.update(mode=self.plan['mode'], limits=self.plan['limits'], sources=self.plan['sources'],
                            inputs=self.plan['inputs'], plan_sha256=self.args.plan_sha256,
                            supervision_sha256=sha(self.args.supervision))
        write(self.out/'started.json', {'request': {k:str(v) for k,v in vars(self.args).items()}, 'launch':launch, 'started_ns':self.start})

    def setup(self):
        os.environ.update(B.THREADS)
        tick = time.perf_counter()
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.np, self.torch = np, torch
        sys.path.insert(0, str(ROOT/'src'))
        from openjev.research.otto_public import observation, seeded_environment
        from openjev.research.otto_reference_control import SpaceAwareActor
        self.observation, self.seeded, self.actor_class = observation, seeded_environment, SpaceAwareActor
        self.SourceTracking = B.load(ROOT/P.UPSTREAM, '_spatial_control_native').SourceTracking
        model_tick = time.perf_counter()
        from openjev.research import otto_spatial_value, otto_value_branches
        self.model, self.branches = otto_spatial_value, otto_value_branches
        self.model_module_setup_seconds = time.perf_counter()-model_tick
        self.kernels, self.mixtures = {}, {}
        for regime in REGIMES:
            with np.load(regular(self.plan['inputs']['kernel_'+regime]['path']), allow_pickle=False) as saved:
                kernel, weights = saved['likelihood'], saved['initial_hit_weights']
            require(kernel.shape == (4,107,107) and kernel.dtype == np.float64 and weights.shape == (4,) and weights[0] == 0, 'fixed public kernel')
            kernel.setflags(write=False)
            self.kernels[regime], self.mixtures[regime] = kernel, {h:float(weights[h]) for h in (1,2,3)}
        self.shared_setup_seconds = time.perf_counter()-tick-self.model_module_setup_seconds
        if self.plan['mode'] == 'study':
            checks = []
            for i, regime in enumerate(REGIMES):
                self.context = {'phase':'native_setup', 'regime':regime}
                env = self.environment(regime, TEMPLATE_FIRST+i, None)
                weights = np.asarray(next(r for r in env.draw_log if r['channel'] == 'initial')['probabilities'])
                require(np.array_equal(env.p_Poisson, self.kernels[regime]) and all(float(weights[h]) == self.mixtures[regime][h] for h in (1,2,3)), 'native cached kernel parity')
                checks.append({'regime':regime, 'template_seed':TEMPLATE_FIRST+i, 'cached_kernel_exact':True, 'initial_public':self.public(env,0)})
            write(self.out/'native-setup.json', {'checks':checks, 'native_resets':3})
        write(self.out/'runtime.json', {'python':sys.version, 'executable':sys.executable, 'environment':B.THREADS,
              'torch_threads':torch.get_num_threads(), 'torch_interop_threads':torch.get_num_interop_threads(),
              'shared_setup_seconds':self.shared_setup_seconds, 'model_module_setup_seconds':self.model_module_setup_seconds,
              'module_allocation_episodes':1080 if self.plan['mode'] == 'study' else 0,
              'module_scope':'Spatial value and explicit branch imports; common setup separately reported'})

    def load_heads(self):
        self.heads, setup = {}, {}
        for arm in ARMS[:-1]:
            path = regular(self.plan['inputs']['head_'+arm.replace('@','_')]['path'])
            self.context = {'phase':'inference_setup', 'arm':arm}
            tick, io = time.perf_counter(), self.io_seconds
            def construct(path=path,arm=arm):
                arrays = self.restore(path)
                require(self.model.validate_head(arrays) == arm.split('@')[0], 'exact named checkpoint family')
                return self.model.FrozenValue(arrays)
            self.heads[arm] = self.call('model_load', construct)
            raw, excluded = time.perf_counter()-tick, self.io_seconds-io
            require(raw >= excluded, 'nonnegative complete head load')
            setup[arm] = {'seconds':raw-excluded, 'instrumented_seconds':raw, 'excluded_io_seconds':excluded,
                          'checkpoint':str(path.relative_to(ROOT)), 'sha256':sha(path),
                          'allocated_episodes':72 if self.plan['mode'] == 'study' else 0, 'storage':self.heads[arm].storage_bytes()}
        write(self.out/'inference-setup.json', setup)
        return setup

    def restore_reference(self, arm):
        kind, seed = arm.split('@')
        arrays = self.restore(regular(self.plan['inputs']['head_'+arm.replace('@','_')]['path']))
        require(self.model.validate_head(arrays) == kind, 'exact Torch reference family')
        reference = self.model.make_head(kind, int(seed), float(arrays['c0']))
        with self.torch.no_grad():
            for name, tensor in reference.named_parameters():
                tensor.copy_(self.torch.from_numpy(arrays[name]))
            reference.c0.copy_(self.torch.from_numpy(arrays['c0']))
        return reference.double().eval()

    def physical(self, head, centered, positions, sensing):
        return 64*head.normalized(centered, positions, self.np.full(len(centered),sensing,dtype=self.np.float64))

    def qualification_inputs(self):
        np = self.np
        started = time.perf_counter()
        cache = {}
        for split,count in (('train',5589),('valid',1109)):
            with np.load(regular(self.plan['inputs'][split+'_data']['path']),allow_pickle=False) as archive:
                cache[split] = {k:archive[k][:8].copy() for k in ('beliefs','positions','sensing_length')}
            metadata = [json.loads(line) for line in regular(self.plan['inputs'][split+'_rows']['path']).read_text().splitlines()]
            require(len(metadata) == count and [r['row_index'] for r in metadata] == list(range(count)), 'fixed complete cache metadata')
            cache[split]['rows'] = metadata[:8]
        shapes = {'beliefs':((52,53,53),np.float64),'positions':((52,2),np.int64),
                  'sensing_length':((52,),np.float64),'eligible_mask':((52,4),np.bool_),
                  'centered_u':((52,16,105,105),np.float64),'centered_z':((52,16,105,105),np.float64),
                  'raw_masses':((52,4,4),np.float64),'weights':((52,4,4),np.float64),'successors':((52,16,2),np.int64)}
        arrays = {k:np.empty(shape,dtype=dtype) for k,(shape,dtype) in shapes.items()}
        records = []
        for index,identity in enumerate(qualification_ids()):
            self.check()
            record = dict(identity)
            if identity['source'] == 'cache':
                split,i = identity['split'],identity['row_index']; d = cache[split]; row = d['rows'][i]
                belief,position,lam = d['beliefs'][i],d['positions'][i],float(d['sensing_length'][i])
                allowed,regime = row['public']['valid_actions'],row['regime']
                require(belief.dtype == np.float64 and belief.shape == (53,53) and position.dtype == np.int64
                        and position.tolist() == list(row['public']['position']) and lam == REGIMES[regime]
                        and row['stage'] == split and row['public']['done'] is False
                        and hashlib.sha256(belief.tobytes()).hexdigest() == row['posterior']['sha256']
                        and float(belief.sum(dtype=np.float64)) == row['posterior']['mass'], 'exact historical public tuple')
                record.update(regime=regime,episode_id=row['episode_id'],prefix_index=row['prefix_index'])
            else:
                regime = identity['regime']
                belief,position,lam,allowed = synthetic_tuple(regime,identity['position'],identity['belief_kind'],np)
            expected_allowed = [a for a in range(4) if 0 <= int(position[a//2])+2*(a%2)-1 < 53]
            require(list(allowed) == expected_allowed, 'exact inbounds qualification mask')
            branch = self.branches.rl_branches(belief,position,self.kernels[regime],allowed)
            arrays['beliefs'][index],arrays['positions'][index],arrays['sensing_length'][index] = belief,position,lam
            arrays['eligible_mask'][index] = [a in allowed for a in range(4)]
            for k in ('centered_u','centered_z','raw_masses','weights','successors'): arrays[k][index] = getattr(branch,k)
            hashes = {k:hashlib.sha256(v[index].tobytes()).hexdigest() for k,v in arrays.items()}
            record.update(index=index,position=position.tolist(),sensing_length=lam,allowed_actions=list(allowed),array_sha256=hashes)
            records.append(record); self.emit('qualification-tuples.jsonl',record)
        path = self.out/'qualification-arrays.npz'
        np.savez_compressed(path,**arrays)
        evidence = {k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.tobytes()).hexdigest(),'bytes':v.nbytes} for k,v in arrays.items()}
        write(self.out/'preparation.json',{'seconds':time.perf_counter()-started,'tuples':52,'array_file':descriptor(path),
              'arrays':evidence,'scope':'Shared tuples once;16 fixed historical public prefixes plus36 prescribed synthetic tuples; targets not decoded.'})
        return arrays,records

    def qualify(self, prepared):
        np,torch = self.np,self.torch
        arrays,records = prepared
        maxima = {'values':0.,'costs':0.}
        for arm in ARMS[:-1]:
            self.context = {'phase':'parity_restore','fit_id':arm}
            reference = self.call('parity_restore',lambda arm=arm:self.restore_reference(arm))
            for index,identity in enumerate(records):
                self.context = {'phase':'parity','fit_id':arm,'step':index}
                z,positions = arrays['centered_z'][index],arrays['successors'][index]
                lam = float(arrays['sensing_length'][index])
                sensing = np.full(16,lam,dtype=np.float64)
                values_np = self.call('parity_numpy_forward',lambda z=z,positions=positions,lam=lam,arm=arm:self.physical(self.heads[arm],z,positions,lam))
                def expected(reference=reference,z=z,positions=positions,sensing=sensing):
                    with torch.no_grad():
                        return 64*reference(torch.from_numpy(z),torch.from_numpy(positions),torch.from_numpy(sensing)).numpy()
                values_ref = self.call('parity_torch_forward',expected)
                weight = arrays['weights'][index]
                costs_np = 1+np.sum(weight*values_np.reshape(4,4),axis=1,dtype=np.float64)
                costs_ref = 1+np.sum(weight*values_ref.reshape(4,4),axis=1,dtype=np.float64)
                allowed = identity['allowed_actions']
                action_np,action_ref = (self.branches.select_action(v,allowed) for v in (costs_np,costs_ref))
                require(values_np.shape == values_ref.shape == (16,) and values_np.dtype == values_ref.dtype == np.float64
                        and np.isfinite(values_np).all() and np.isfinite(values_ref).all(), 'finite physical branch values')
                differences = {'values':float(np.max(np.abs(values_np-values_ref))),'costs':float(np.max(np.abs(costs_np-costs_ref)))}
                maxima = {k:max(maxima[k],differences[k]) for k in maxima}
                passed = bool(np.all(np.abs(values_np-values_ref) <= ATOL+RTOL*np.abs(values_ref))
                              and np.all(np.abs(costs_np-costs_ref) <= ATOL+RTOL*np.abs(costs_ref)) and action_np == action_ref)
                self.emit('parity.jsonl',{'fit_id':arm,'tuple_id':identity['tuple_id'],'tuple_index':index,
                    'checkpoint_sha256':self.plan['inputs']['head_'+arm.replace('@','_')]['sha256'],
                    'array_sha256':identity['array_sha256'],'allowed_actions':allowed,
                    'values_numpy':values_np.tolist(),'values_torch':values_ref.tolist(),
                    'costs_numpy':costs_np.tolist(),'costs_torch':costs_ref.tolist(),
                    'action_numpy':action_np,'action_torch':action_ref,'maximum_error':differences,'passed':passed})
                require(passed,'every prescribed physical value/cost/action must qualify')
            self.sync()
        self.receipt['parity_passed'] = True
        write(self.out/'summary.json',{'version':VERSION,'mode':'qualify','parity_records':780,'shared_tuples':52,
              'parity_passed':True,'completed_episodes':0,'maximum_error':maxima,'atol':ATOL,'rtol':RTOL,
              'learned_architecture_advantage_established':False})

    def evaluate(self, setup):
        rows, sources, uniforms = [], {}, {}
        for regime, seed, block, hit, arm in evaluation_order(FIRST):
            row = self.episode(regime,seed,hit,arm,block,self.heads.get(arm))
            allocation = setup[arm]['seconds']/72+self.model_module_setup_seconds/1080 if arm in setup else 0.
            row['setup_allocation_seconds'] = allocation
            row['controller_seconds'] += allocation
            self.emit('evaluation.jsonl',row)
            key = (regime,seed)
            require(key not in sources or sources[key] == row['source_evaluation_only'], 'matched source')
            sources[key] = row['source_evaluation_only']
            for draw in row['draws_evaluation_only']:
                identity = (*key,draw['channel'],draw['index'])
                require(identity not in uniforms or uniforms[identity] == draw['uniform'], 'matched categorical uniform')
                uniforms[identity] = draw['uniform']
            rows.append(row)
            self.receipt['completed_episodes'] = len(rows)
            if len(rows) % 16 == 0:
                print(json.dumps({'episodes':len(rows),'native_steps':self.calls['native_step']['returned']}),flush=True)
        self.completed_rows = rows
        result = summarize(rows,self.mixtures,FIRST)
        result.update(amortization=self.amortization(result,setup),inference_setup=setup,
                      shared_setup_seconds=self.shared_setup_seconds,model_module_setup_seconds=self.model_module_setup_seconds,
                      calls=self.calls,paired_source_cases=len(sources),paired_uniforms=len(uniforms))
        write(self.out/'summary.json',result)
        return result

    def amortization(self,result,setup):
        prior_path = regular(self.plan['inputs']['spatial_receipt']['path'])
        prior, worker = read(prior_path.parent/'summary.json'), read(prior_path)
        scenarios = {}
        for regime,panel in result['regimes'].items():
            scenarios[regime] = {}
            for arm,metrics in panel['means'].items():
                if arm == 'analytic_inbounds':
                    fit,prep,deployment,utility = 0.,0.,0.,metrics['controller_seconds']
                else:
                    fit,prep = prior['training_costs'][arm],prior['preparation_seconds']/15
                    deployment = setup[arm]['seconds']+self.model_module_setup_seconds/15
                    utility = metrics['controller_seconds']-metrics['setup_allocation_seconds']
                scenarios[regime][arm] = {'fit_seconds':fit,'preparation_share_seconds':prep,'deployment_setup_seconds':deployment,
                    'controller_without_deployment_seconds':utility,
                    'seconds_per_search':{str(h):utility+(fit+prep+deployment)/h for h in (1,100,10000)}}
        return {'scope':'Prior fit plus preparation and deployment amortization; qualification and final diagnostics separately reported. Analytic actor initialization paid per search.',
                'training_reference':{k:prior[k] for k in ('training_costs','diagnostic_costs','preparation_seconds')}|{'worker_seconds':worker['wall_seconds']},
                'scenarios':scenarios}

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists() and not any(p.is_symlink() for p in self.out.parents), 'exclusive output')
        self.out.mkdir(parents=True)
        try:
            self.bind()
            write(self.out/'serialization-projection.json',self.plan['serialization_projection'])
            self.setup()
            prepared = self.qualification_inputs() if self.plan['mode'] == 'qualify' else None
            setup = self.load_heads()
            if self.plan['mode'] == 'qualify':
                self.qualify(prepared)
            else:
                result = self.evaluate(setup)
                self.receipt.update(pilot_continuation=result['pilot_continuation'],parity_passed=True)
            self.sync()
            for stream in self.handles.values():
                stream.close()
            self.handles.clear()
            require({p.name for p in self.out.iterdir()} == payload_names(self.plan['mode']), 'exact output payload')
            require(authenticate(self.args) == self.plan and sha(self.args.supervision) == self.receipt['supervision_sha256'], 'unchanged end identity')
            require(not self.pending and all(v['attempted'] == v['returned'] for v in self.calls.values()), 'all attempted operations returned')
            expected = {'model_load':15}
            if self.plan['mode'] == 'qualify':
                expected.update(parity_restore=15,parity_numpy_forward=780,parity_torch_forward=780)
                require(set(self.calls) == set(expected), 'qualification performs no extra calls')
            else:
                expected['native_reset'] = 1155
                expected['native_step'] = sum(r['steps'] for r in self.completed_rows)
                expected['value_forward'] = sum(r['steps'] for r in self.completed_rows if r['arm'] != 'analytic_inbounds')
                expected['analytic_choose'] = sum(r['steps'] for r in self.completed_rows if r['arm'] == 'analytic_inbounds')
                require(self.receipt['completed_episodes'] == 1152 and set(self.calls) == {'model_load','native_reset','native_step','value_forward','analytic_choose'}, 'complete autonomous cohort/work types')
            require(all(self.calls[k]['returned'] == v for k,v in expected.items()), 'complete fixed work')
            self.check()
            end = self.clock.now_ns()
            self.receipt.update(status='completed',calls=self.calls,pending=self.pending,started_ns=self.start,finished_ns=end,
                clock_backend=self.clock.backend,wall_seconds=(end-self.start)/1e9,artifact_io_seconds=self.io_seconds,
                files={p.name:descriptor(p) for p in self.out.iterdir()})
            write(self.out/'receipt.json',self.receipt)
            self.check()
        except BaseException as error:
            self.receipt.update(status='failed',calls=self.calls,pending=self.pending,error=repr(error),traceback=traceback.format_exc())
            cleanup_errors = []
            for name,stream in self.handles.items():
                if not stream.closed:
                    for operation in ('flush','close'):
                        try:
                            getattr(stream,operation)()
                        except BaseException as secondary:  # noqa: BLE001 - Try every cleanup and preserve the primary error.
                            cleanup_errors.append(f'{name}.{operation}: {secondary!r}')
            if cleanup_errors:
                self.receipt['cleanup_errors'] = cleanup_errors
                error.add_note('; '.join(cleanup_errors))
            try:
                if (self.out/'receipt.json').exists():
                    (self.out/'receipt.json').rename(self.out/'invalid-completed-receipt.json')
                write(self.out/'failed.json',self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - Preserve the original failure.
                error.add_note(repr(secondary))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan','output','supervision'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    Run(parser.parse_args()).execute()
