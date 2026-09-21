"""Fresh autonomous evaluation of both fixed conditioning arms, without fitting."""
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
SCALAR = 'scripts/study_otto_conditioning.py'
SCALAR_PIN = '9c999001f2e71b535df42e9781cc15def102a7fae2bb5a0af62ab0a6544dcc1d'


def pinned(name, pin, module_name):
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != pin:
        raise ValueError('unchanged inherited source '+name)
    spec = importlib.util.spec_from_file_location(module_name, ROOT/name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


B = pinned(BASE, BASE_PIN, '_conditioning_control_base')
S = pinned(SCALAR, SCALAR_PIN, '_conditioning_control_scalar')
P = B.P
require, sha, read, write, regular, descriptor, closed = B.require, B.sha, B.read, B.write, B.regular, B.descriptor, B.closed
VERSION = 'otto-conditioning-control-v1'
SEEDS, FAMILIES = (10101, 10102, 10103), ('gain1', 'gain53')
ARMS = tuple(f'{kind}@{seed}' for seed in SEEDS for kind in FAMILIES)+('analytic_inbounds',)
REGIMES = {'lambda3': 3., 'lambda4': 4., 'lambda5': 5.}
FIRST = {'lambda3': 15100001, 'lambda4': 15200001, 'lambda5': 15300001}
TEMPLATE_FIRST, CASES, HORIZON = 15500001, 24, 2188
TIMES, METRICS = P.TIMES, P.METRICS
NEW = ('scripts/study_otto_conditioning_control.py', 'scripts/freeze_otto_conditioning_control.py',
       'scripts/audit_otto_conditioning_control.py', 'tests/test_otto_conditioning_control.py',
       'tests/test_audit_otto_conditioning_control.py', 'research/otto-conditioning-control-protocol.md')


def limits(mode):
    require(mode in ('qualify', 'study'), 'known mode')
    return {'native_seconds': 180 if mode == 'qualify' else 5400,
            'rss_bytes': (4 if mode == 'qualify' else 8)*1024**3,
            'output_bytes': 128*1024**2 if mode == 'qualify' else 6*1024**3,
            'native_steps': 0 if mode == 'qualify' else 504*HORIZON,
            'native_resets': 0 if mode == 'qualify' else 507,
            'optimizer_updates': 0, 'target_readouts': 0,
            'value_forward': 0 if mode == 'qualify' else 432*HORIZON,
            'model_load': 6, 'parity_restore': 6 if mode == 'qualify' else 0,
            'parity_numpy_forward': 96 if mode == 'qualify' else 0,
            'parity_torch_forward': 96 if mode == 'qualify' else 0}


def audit_limits(mode):
    return {'native_seconds': 180 if mode == 'qualify' else 1800,
            'rss_bytes': 4*1024**3, 'output_bytes': 128*1024**2}


def configuration(mode):
    return {'arms': list(ARMS), 'cases_per_regime': 0 if mode == 'qualify' else CASES,
            'horizon': HORIZON, 'evaluation_first_seeds': FIRST, 'template_first_seed': TEMPLATE_FIRST,
            'initial_hit': '1+case%3', 'block': 'case//3', 'arm_rotation': '(regime_index*24+case)%7',
            'parity_states': [{'split': split, 'row_index': i} for split in ('train', 'valid') for i in range(8)],
            'parity': 'six fixed heads;16 branch values,four costs,exact eligible action;atol=rtol=1e-10',
            'amortization_searches': [1, 100, 10000], 'external_model_calls': 0,
            'scalar_gate_required': False}


def payload_names(mode):
    names = {'started.json', 'runtime.json', 'inference-setup.json', 'work-contexts.jsonl', 'work.jsonl', 'summary.json'}
    return names | ({'preparation.json', 'parity.jsonl'} if mode == 'qualify' else
                    {'native-setup.json', 'eval-transitions.jsonl', 'eval-episodes.jsonl', 'evaluation.jsonl'})


def authenticate(args):
    require(args.plan.is_absolute() and not args.plan.is_symlink() and sha(args.plan) == args.plan_sha256, 'external plan pin')
    plan = read(args.plan)
    mode = plan['mode']
    require(plan['version'] == VERSION and plan['status'] == 'frozen_before_execution'
            and plan['configuration'] == configuration(mode) and plan['limits'] == limits(mode)
            and plan['independent_audit_limits'] == audit_limits(mode), 'fixed autonomous contract')
    require(set(NEW) <= plan['sources'].keys(), 'complete new source closure')
    for name, pin in plan['sources'].items():
        require(sha(regular(name)) == pin, 'unchanged source '+name)
    roles = {'conditioning_plan', 'conditioning_receipt', 'conditioning_terminal', 'conditioning_audit',
             'scalar_receipt', 'seed_review', *('kernel_'+r for r in REGIMES),
             *('head_'+a.replace('@', '_') for a in ARMS[:-1])}
    if mode == 'qualify':
        roles |= {'train_data', 'valid_data', 'train_rows', 'valid_rows'}
    require(set(plan['inputs']) == roles, 'exact admitted input roles')
    paths = {}
    for role, item in plan['inputs'].items():
        paths[role] = regular(item['path'])
        require(descriptor(paths[role]) == {k: item[k] for k in ('bytes', 'sha256')}, 'input byte identity '+role)
    seed_review = read(paths['seed_review'])
    require(seed_review['status'] == 'completed' and seed_review['passed'] is True
            and seed_review['overlap'] == seed_review['overlaps'] == []
            and seed_review['historical_declaration_matches'] == []
            and seed_review['proposed_unique_seeds'] == 75
            and seed_review['proposed_ranges_inclusive'] ==
                {**{'eval_'+r:[n,n+23] for r,n in FIRST.items()},'templates':[TEMPLATE_FIRST,TEMPLATE_FIRST+2]},
            'successful scoped review of exact fresh ranges')
    prior = S.authenticate(SimpleNamespace(plan=paths['conditioning_plan'], plan_sha256=sha(paths['conditioning_plan'])))
    worker, audit = closed(paths['conditioning_receipt']), closed(paths['conditioning_audit'])
    require(prior['mode'] == worker['mode'] == 'study' and worker['completed_fits'] == 6
            and worker['parity_passed'] and worker['initial_pairing_passed'] and not worker['pending']
            and worker['plan_sha256'] == sha(paths['conditioning_plan']) and worker['sources'] == prior['sources']
            and worker['inputs'] == prior['inputs'] and set(worker['files']) == S.payload_names('study'), 'closed six-fit conditioning study')
    S.C.terminal_identity(paths['conditioning_receipt'], paths['conditioning_plan'], paths['conditioning_terminal'], SCALAR)
    require(audit['agreement'] is True and audit['worker_sha256'] == sha(paths['conditioning_receipt'])
            and audit['plan_sha256'] == sha(paths['conditioning_plan'])
            and audit['terminal_sha256'] == sha(paths['conditioning_terminal'])
            and audit['source']['sha256'] == prior['sources']['scripts/audit_otto_conditioning.py'], 'completed independent scalar audit')
    require(all(plan['sources'].get(k) == v for k, v in prior['sources'].items()), 'unchanged inherited source closure')
    for key in ('python_executable', 'python_version', 'all_distributions'):
        require(plan[key] == prior[key], 'unchanged qualified runtime')
    require(sha(paths['scalar_receipt']) == '965cc3eee7c77ea64700137bda2961af1a259f1a98d15b54041f9530a7f66ac3', 'original scalar receipt identity')
    original = read(paths['scalar_receipt'])
    require(original['status'] == 'completed', 'completed original scalar lineage')
    for regime in REGIMES:
        require(paths['kernel_'+regime] == paths['scalar_receipt'].parent/f'kernel-{regime}.npz'
                and descriptor(paths['kernel_'+regime]) == original['files'][f'kernel-{regime}.npz'], 'same qualified kernel')
    for arm in ARMS[:-1]:
        name = 'final-'+arm.replace('@', '-')+'.npz'
        require(paths['head_'+arm.replace('@', '_')] == paths['conditioning_receipt'].parent/name
                and descriptor(paths['head_'+arm.replace('@', '_')]) == worker['files'][name], 'fixed final head')
    if mode == 'qualify':
        require(plan['qualification'] == {}, 'disposable mechanical qualification')
        for role in ('train_data', 'valid_data', 'train_rows', 'valid_rows'):
            require(plan['inputs'][role] == prior['inputs'][role], 'same fixed parity cache')
    else:
        require(set(plan['qualification']) == {'plan', 'receipt', 'terminal', 'audit'}, 'complete qualified deployment inputs')
        qp = {}
        for role, item in plan['qualification'].items():
            qp[role] = regular(item['path'])
            require(descriptor(qp[role]) == {k: item[k] for k in ('bytes', 'sha256')}, 'qualification external byte binding')
        qplan = authenticate(SimpleNamespace(plan=qp['plan'], plan_sha256=sha(qp['plan'])))
        qworker, qaudit = closed(qp['receipt']), closed(qp['audit'])
        require(qplan['mode'] == qworker['mode'] == 'qualify' and qworker['parity_passed'] is True
                and qworker['completed_episodes'] == 0 and not qworker['pending']
                and qworker['plan_sha256'] == sha(qp['plan']) and qworker['sources'] == qplan['sources']
                and qworker['inputs'] == qplan['inputs'] and set(qworker['files']) == payload_names('qualify'), 'completed deployment qualification')
        expected = {'model_load': 6, 'parity_restore': 6, 'parity_numpy_forward': 96, 'parity_torch_forward': 96}
        require({k: v['returned'] for k, v in qworker['calls'].items()} == expected
                and all(v['attempted'] == v['returned'] for v in qworker['calls'].values()), 'closed qualification work')
        S.C.terminal_identity(qp['receipt'], qp['plan'], qp['terminal'], 'scripts/study_otto_conditioning_control.py')
        require(qaudit['agreement'] is True and qaudit['worker_sha256'] == sha(qp['receipt'])
                and qaudit['plan_sha256'] == sha(qp['plan']) and qaudit['terminal_sha256'] == sha(qp['terminal'])
                and qaudit['source']['sha256'] == qplan['sources']['scripts/audit_otto_conditioning_control.py'], 'independent deployed parity audit')
        require(plan['sources'] == qplan['sources'] and all(plan['inputs'][r] == qplan['inputs'][r] for r in roles), 'same qualified deployment')
    return plan


def evaluation_order(first):
    require(first == FIRST, 'fixed fresh evaluation ranges')
    for ri, regime in enumerate(REGIMES):
        for case in range(CASES):
            offset = (ri*CASES+case) % len(ARMS)
            for arm in ARMS[offset:]+ARMS[:offset]:
                yield regime, first[regime]+case, case//3, 1+case%3, arm


def summarize(rows, mixtures, first):
    require([(r['regime'], r['seed'], r['block'], r['initial_hit'], r['arm']) for r in rows] == list(evaluation_order(first)), 'exact504 complete episodes')
    for r in rows:
        require(type(r['found']) is bool and type(r['steps']) is int and 1 <= r['steps'] <= HORIZON
                and (r['found'] or r['steps'] == HORIZON) and r['updates'] == r['steps']
                and r['final_update_assimilated'] is True and r['blocked_steps'] == 0, 'complete found/censored paths')
        require(all(math.isfinite(r[m]) and r[m] >= 0 for m in METRICS)
                and abs(r['controller_seconds']-math.fsum(r[k] for k in TIMES)) <= 1e-9, 'complete finite controller costs')
    regimes, competence, controls, improvement = {}, [], [], []
    def add(group, name, value, threshold, passes):
        group.append({'name': name, 'value': value, 'threshold': threshold, 'passes': bool(passes)})
    for regime in REGIMES:
        weights = {int(k): v for k, v in mixtures[regime].items()}
        require(set(weights) == {1, 2, 3} and all(math.isfinite(v) and 0 < v < 1 for v in weights.values())
                and abs(math.fsum(weights.values())-1) <= 1e-12, 'positive normalized hit mixture')
        selected = [r for r in rows if r['regime'] == regime]
        def weighted(subset, metric, weights=weights):
            return math.fsum(weights[h]*math.fsum(float(r[metric]) for r in subset if r['initial_hit'] == h)
                             /sum(r['initial_hit'] == h for r in subset) for h in (1, 2, 3))
        means = {a: {m: weighted([r for r in selected if r['arm'] == a], m) for m in METRICS} for a in ARMS}
        blocks = [{a: weighted([r for r in selected if r['arm'] == a and r['block'] == b], 'steps') for a in ARMS} for b in range(8)]
        family = {f: {m: math.fsum(means[f'{f}@{s}'][m] for s in SEEDS)/3 for m in METRICS} for f in FAMILIES}
        teacher, candidate, reference = means['analytic_inbounds'], family['gain53'], family['gain1']
        for kind, group in (('gain53', competence), ('gain1', controls)):
            for seed in SEEDS:
                value = means[f'{kind}@{seed}']
                add(group, f'{regime}.{kind}.{seed}.success', value['found'], .95, value['found'] >= .95)
                add(group, f'{regime}.{kind}.{seed}.moves', value['steps'], 1.05*teacher['steps'], value['steps'] <= 1.05*teacher['steps'])
        positive = sum(math.fsum(b[f'gain1@{s}']-b[f'gain53@{s}'] for s in SEEDS)/3 > 0 for b in blocks)
        add(improvement, regime+'.success', candidate['found'], reference['found'], candidate['found'] >= reference['found'])
        add(improvement, regime+'.moves', candidate['steps'], .95*reference['steps'], candidate['steps'] <= .95*reference['steps'])
        add(improvement, regime+'.positive_blocks', positive, 6, positive >= 6)
        add(improvement, regime+'.cost', candidate['controller_seconds'], reference['controller_seconds'], candidate['controller_seconds'] <= reference['controller_seconds'])
        regimes[regime] = {'weights': weights, 'means': means, 'family_means': family, 'blocks': blocks,
            'strata': {str(h): {a: {m: math.fsum(float(r[m]) for r in selected if r['arm'] == a and r['initial_hit'] == h)/8
                                   for m in METRICS} for a in ARMS} for h in (1, 2, 3)},
            'raw_counts': {a: {'found': sum(r['found'] for r in selected if r['arm'] == a), 'episodes':24} for a in ARMS}}
    return {'version': VERSION, 'episodes': len(rows), 'paired_cases': 72, 'regimes': regimes,
            'competence_checks': competence, 'improvement_checks': improvement, 'control_competence_checks': controls,
            'pilot_continuation': all(r['passes'] for r in competence+improvement), 'learned_architecture_advantage_established': False}


class Run(B.Run):
    def __init__(self, args):
        super().__init__(args)
        self.receipt = {'version': VERSION, 'status': 'started', 'external_model_calls': 0,
                        'completed_episodes': 0, 'completed_fits': 0, 'parity_passed': False}

    def begin(self, channel, *, check=True):
        if channel in ('model_load', 'value_forward', 'parity_restore', 'parity_numpy_forward', 'parity_torch_forward'):
            require(self.calls.get(channel, {}).get('attempted', 0) < self.plan['limits'][channel], 'readout/load allocation before invocation')
        return super().begin(channel, check=check)

    def bind(self):
        require(sha(ROOT/P.CLOCK) == P.CLOCK_PIN, 'qualified clock')
        self.clock = B.load(ROOT/P.CLOCK, '_conditioning_control_clock').SuspendClock()
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
        self.SourceTracking = B.load(ROOT/P.UPSTREAM, '_conditioning_control_native').SourceTracking
        model_tick = time.perf_counter()
        from openjev.research import otto_conditioned_value, otto_value_branches
        self.model, self.branches = otto_conditioned_value, otto_value_branches
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
              'module_allocation_episodes':432 if self.plan['mode'] == 'study' else 0,
              'module_scope':'Conditioned value and explicit branch imports; common setup separately reported'})

    def load_heads(self):
        self.heads, setup = {}, {}
        for arm in ARMS[:-1]:
            path = regular(self.plan['inputs']['head_'+arm.replace('@','_')]['path'])
            self.context = {'phase':'inference_setup', 'arm':arm}
            tick, io = time.perf_counter(), self.io_seconds
            self.heads[arm] = self.call('model_load', lambda path=path:self.model.FrozenValue(self.restore(path)))
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
        self.model.validate_head(arrays)
        reference = self.model.make_head(kind, int(seed), float(arrays['c0']))
        with self.torch.no_grad():
            for name, tensor in reference.named_parameters():
                tensor.copy_(self.torch.from_numpy(arrays[name]))
            reference.c0.copy_(self.torch.from_numpy(arrays['c0']))
        return reference.double().eval()

    def qualify(self):
        np, torch = self.np, self.torch
        tick = time.perf_counter()
        data = {}
        for split, count in (('train',5589), ('valid',1109)):
            with np.load(regular(self.plan['inputs'][split+'_data']['path']), allow_pickle=False) as archive:
                data[split] = {k:archive[k][:8].copy() for k in ('beliefs','positions','sensing_length')}
            rows = [json.loads(line) for line in regular(self.plan['inputs'][split+'_rows']['path']).read_text().splitlines()]
            require(len(rows) == count and [r['row_index'] for r in rows] == list(range(count)), 'authenticated cache row order')
            data[split]['rows'] = rows[:8]
        prep_seconds = time.perf_counter()-tick
        write(self.out/'preparation.json', {'seconds':prep_seconds, 'states':self.plan['configuration']['parity_states'],
                                          'scope':'First eight TRAIN and VALID states; no targets, fitting or policy evaluation'})
        for arm in ARMS[:-1]:
            self.context = {'phase':'parity_restore', 'fit_id':arm}
            reference = self.call('parity_restore', lambda arm=arm:self.restore_reference(arm))
            for split in ('train','valid'):
                for index in range(8):
                    self.context = {'phase':'parity', 'fit_id':arm, 'split':split, 'step':index}
                    d = data[split]
                    lam = float(d['sensing_length'][index]); row = d['rows'][index]
                    branch = self.branches.rl_branches(d['beliefs'][index], d['positions'][index], self.kernels[row['regime']], row['public']['valid_actions'])
                    saved = {}
                    def values(z, positions, kernel, route, lam=lam, arm=arm, reference=reference, saved=saved):
                        features = self.model.value_features(z, positions, lam)
                        def operation():
                            if route == 'numpy':
                                return 64*self.heads[arm].normalized(features)
                            with torch.no_grad():
                                return 64*reference(torch.from_numpy(features)).numpy()
                        result = self.call('parity_'+route+'_forward', operation)
                        saved[route] = result
                        return result
                    score_np = self.branches.explicit_scores(branch, lambda z,p,k,values=values:values(z,p,k,'numpy'), arithmetic='float64')
                    score_ref = self.branches.explicit_scores(branch, lambda z,p,k,values=values:values(z,p,k,'torch'), arithmetic='float64')
                    action_np = self.branches.select_action(score_np, branch.eligible_actions)
                    action_ref = self.branches.select_action(score_ref, branch.eligible_actions)
                    close = lambda a,b:bool(np.all(np.abs(a-b) <= 1e-10+1e-10*np.abs(b)))
                    passed = close(saved['numpy'],saved['torch']) and close(score_np,score_ref) and action_np == action_ref
                    self.emit('parity.jsonl', {'fit_id':arm, 'split':split, 'row_index':index,
                        'position':d['positions'][index].tolist(), 'sensing_length':lam,
                        'eligible_mask':[a in branch.eligible_actions for a in range(4)],
                        'checkpoint_sha256':self.plan['inputs']['head_'+arm.replace('@','_')]['sha256'],
                        'posterior_sha256':hashlib.sha256(d['beliefs'][index].tobytes()).hexdigest(),
                        'raw_masses':branch.raw_masses.tolist(), 'weights':branch.weights.tolist(),
                        'values_numpy':saved['numpy'].tolist(), 'values_torch':saved['torch'].tolist(),
                        'costs_numpy':score_np.tolist(), 'costs_torch':score_ref.tolist(),
                        'allowed_actions':list(branch.eligible_actions), 'action_numpy':action_np, 'action_torch':action_ref, 'passed':passed})
                    require(passed, 'all fixed deployed branch/cost/action parity')
        self.receipt['parity_passed'] = True
        write(self.out/'summary.json', {'version':VERSION,'mode':'qualify','parity_records':96,'parity_passed':True,
              'completed_episodes':0,'preparation_seconds':prep_seconds,'learned_architecture_advantage_established':False})

    def evaluate(self, setup):
        rows, sources, uniforms = [], {}, {}
        for regime, seed, block, hit, arm in evaluation_order(FIRST):
            row = self.episode(regime,seed,hit,arm,block,self.heads.get(arm))
            allocation = setup[arm]['seconds']/72+self.model_module_setup_seconds/432 if arm in setup else 0.
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
            if len(rows) % 7 == 0:
                print(json.dumps({'episodes':len(rows),'native_steps':self.calls['native_step']['returned']}),flush=True)
        result = summarize(rows,self.mixtures,FIRST)
        result.update(amortization=self.amortization(result,setup),inference_setup=setup,
                      shared_setup_seconds=self.shared_setup_seconds,model_module_setup_seconds=self.model_module_setup_seconds,
                      calls=self.calls,paired_source_cases=len(sources),paired_uniforms=len(uniforms))
        write(self.out/'summary.json',result)
        return result

    def amortization(self,result,setup):
        prior_path = regular(self.plan['inputs']['conditioning_receipt']['path'])
        prior, worker = read(prior_path.parent/'summary.json'), read(prior_path)
        scenarios = {}
        for regime,panel in result['regimes'].items():
            scenarios[regime] = {}
            for arm,metrics in panel['means'].items():
                if arm == 'analytic_inbounds':
                    fit,prep,deployment,utility = 0.,0.,0.,metrics['controller_seconds']
                else:
                    fit,prep = prior['training_costs'][arm],prior['preparation_seconds']/6
                    deployment = setup[arm]['seconds']+self.model_module_setup_seconds/6
                    utility = metrics['controller_seconds']-metrics['setup_allocation_seconds']
                scenarios[regime][arm] = {'fit_seconds':fit,'preparation_share_seconds':prep,'deployment_setup_seconds':deployment,
                    'controller_without_deployment_seconds':utility,
                    'seconds_per_search':{str(h):utility+(fit+prep+deployment)/h for h in (1,100,10000)}}
        return {'scope':'Prior fit plus preparation and deployment amortization; qualification, initial pairing and diagnostics separately reported. Analytic actor initialization paid per search.',
                'training_reference':{k:prior[k] for k in ('training_costs','preparation_seconds','initial_pairing_seconds')}|{'worker_seconds':worker['wall_seconds']},
                'scenarios':scenarios}

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists() and not any(p.is_symlink() for p in self.out.parents), 'exclusive output')
        self.out.mkdir(parents=True)
        try:
            self.bind(); self.setup()
            setup = self.load_heads()
            if self.plan['mode'] == 'qualify':
                self.qualify()
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
            expected = {'model_load':6}
            if self.plan['mode'] == 'qualify':
                expected.update(parity_restore=6,parity_numpy_forward=96,parity_torch_forward=96)
                require(set(self.calls) == set(expected), 'qualification performs no extra calls')
            else:
                expected['native_reset'] = 507
                require(self.receipt['completed_episodes'] == 504 and set(self.calls) == {'model_load','native_reset','native_step','value_forward','analytic_choose'}, 'complete autonomous cohort/work types')
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
