"""Fixed-data ordinary-capacity screen; no collection, branches or policy evaluation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OLD = 'scripts/study_otto_return_value.py'
OLD_PIN = '3afdf21506dc60db9a91a18200dfe005133ab1b9398f3baa12614498253abc11'
VERSION = 'otto-capacity-v1'
KINDS, SEEDS = ('mlp8', 'mlp128', 'deep128'), (10101, 10102, 10103)
THREADS = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                          'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
NEW = ('scripts/study_otto_capacity.py', 'src/openjev/research/otto_capacity_value.py',
       'tests/test_otto_capacity_value.py', 'tests/test_otto_capacity.py', 'research/otto-capacity-protocol.md')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


require(sha(ROOT / OLD) == OLD_PIN, 'qualified scalar lineage helper')
P = load(ROOT / OLD, '_capacity_scalar_lineage')
read, write, descriptor, regular, closed = P.read, P.write, P.descriptor, P.regular, P.closed


def configuration(mode):
    require(mode in ('qualify', 'study'), 'known mode')
    return {'kinds': list(KINDS), 'seeds': [10101] if mode == 'qualify' else list(SEEDS),
            'rows': 256 if mode == 'qualify' else 5589, 'epochs': 6 if mode == 'qualify' else 80,
            'batch_size': 128, 'learning_rate': .001, 'gradient_clip': 5.,
            'shuffle_seed_offset': 20000, 'prediction_batch': 256, 'checkpoint': 'fixed final',
            'inputs': 'cached float32; float64 upcast for scalar diagnostics',
            'initialization': 'local seed; hidden normal sqrt(2/fan_in), final sqrt(1/fan_in); zero biases',
            'target': '(T-t)/64, uniform row MSE', 'qualification_selection': 'linspace(0,5588,256,int64)'}


def limits(mode):
    require(mode in ('qualify', 'study'), 'known limits')
    return {'native_seconds': 120 if mode == 'qualify' else 1800,
            'rss_bytes': (4 if mode == 'qualify' else 8)*1024**3,
            'output_bytes': (128*1024**2 if mode == 'qualify' else 1024**3),
            'optimizer_updates': 36 if mode == 'qualify' else 31680,
            'native_steps': 0, 'native_resets': 0, 'external_model_calls': 0}


def payload_names(mode):
    names = {'started.json', 'runtime.json', 'preparation.json', 'work.jsonl', 'initializations.jsonl',
             'epoch-orders.jsonl', 'updates.jsonl', 'fit-curves.jsonl', 'fits.jsonl', 'parity.jsonl', 'summary.json'}
    for seed in configuration(mode)['seeds']:
        for kind in KINDS:
            names.update(f'{prefix}-{kind}-{seed}.npz' for prefix in ('initial', 'final', 'predictions'))
    return names


def terminal_identity(worker_path, plan_path, terminal_path, runner):
    worker, plan, terminal = read(worker_path), read(plan_path), read(terminal_path)
    started = read(worker_path.parent / 'started.json')
    launch, request = started['launch'], started['request']
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and not terminal['timed_out']
            and terminal['group_absent'] and terminal['cleanup']['group_absent'] and terminal['cleanup']['reaped']
            and terminal['cleanup']['errors'] == [] and terminal['error'] is None and terminal['clock_error'] is None,
            'successful original parent terminal')
    for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend',
                'cap_seconds', 'watchdog_sha256', 'clock_source_sha256'):
        require(terminal[key] == launch[key], 'parent/worker launch join')
    require(sha(Path(request['supervision'])) == worker['supervision_sha256']
            and read(Path(request['supervision'])) == launch, 'launch file identity')
    require(request == {'plan': str(plan_path), 'plan_sha256': sha(plan_path),
                        'output': str(worker_path.parent), 'supervision': request['supervision']}, 'original request identity')
    command = list(launch['command'])
    if command[1:2] == ['-u']:
        command.pop(1)
    require(command == [plan['python_executable'], str(ROOT / runner), '--plan', str(plan_path),
                        '--plan-sha256', sha(plan_path), '--output', str(worker_path.parent),
                        '--supervision', request['supervision']], 'exact process command')
    require(launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns']
            and worker['started_ns'] == started['started_ns']
            and worker['wall_seconds'] == (worker['finished_ns']-worker['started_ns'])/1e9
            and launch['cap_seconds'] == plan['limits']['native_seconds']
            and launch['deadline_ns'] == launch['started_ns']+launch['cap_seconds']*10**9
            and terminal['elapsed_ns'] == terminal['finished_ns']-terminal['started_ns']
            and launch['cwd'] == str(ROOT) and launch['pid'] == launch['pgid'] != launch['parent_pid']
            and launch['clock_source_sha256'] == P.CLOCK_PIN
            and launch['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py'], 'bounded original process')


def authenticate(args):
    require(args.plan.is_absolute() and not args.plan.is_symlink() and sha(args.plan) == args.plan_sha256, 'external plan pin')
    plan = read(args.plan)
    mode = plan['mode']
    require(plan['version'] == VERSION and plan['status'] == 'frozen_before_execution'
            and plan['configuration'] == configuration(mode) and plan['limits'] == limits(mode), 'frozen study contract')
    require(set(NEW) <= plan['sources'].keys() and plan['sources'][OLD] == OLD_PIN, 'complete source closure')
    for name, pin in plan['sources'].items():
        require(sha(regular(name)) == pin, 'unchanged source: '+name)
    roles = {'prior_plan', 'prior_receipt', 'prior_terminal', 'prior_audit_receipt', 'train_data', 'train_rows'}
    if mode == 'study':
        roles |= {'valid_data', 'valid_rows'}
    require(set(plan['inputs']) == roles, 'exact input roles; no EVAL input')
    require(plan['independent_audit_limits'] == {'seconds': 600, 'rss_bytes': 4*1024**3,
                                               'output_bytes': 128*1024**2}, 'fixed audit limits')
    if mode == 'qualify':
        require(plan['qualification'] == {}, 'qualification cannot inherit fitted weights')
    paths = {}
    for role, desc in plan['inputs'].items():
        paths[role] = regular(desc['path'])
        require(descriptor(paths[role]) == {k: desc[k] for k in ('sha256', 'bytes')}, 'input descriptor')
    prior = P.authenticate(SimpleNamespace(plan=paths['prior_plan'], plan_sha256=sha(paths['prior_plan'])))
    require(all(plan['sources'].get(k) == v for k, v in prior['sources'].items()), 'inherited closure')
    worker, audit = closed(paths['prior_receipt']), closed(paths['prior_audit_receipt'])
    require(set(worker['files']) == P.payload_names() and worker['version'] == prior['version']
            and worker['sources'] == prior['sources'] and worker['inputs'] == prior['inputs']
            and worker['plan_sha256'] == sha(paths['prior_plan']) and worker['completed_fits'] == 9
            and worker['completed_episodes'] == 720 and worker['pending'] == [], 'complete prior scalar study')
    terminal_identity(paths['prior_receipt'], paths['prior_plan'], paths['prior_terminal'], OLD)
    require(set(audit['files']) == {'started.json', 'summary.json'} and audit['agreement'] is True
            and audit['worker_sha256'] == sha(paths['prior_receipt']) and audit['plan_sha256'] == sha(paths['prior_plan'])
            and audit['terminal_sha256'] == sha(paths['prior_terminal'])
            and audit['source']['sha256'] == plan['sources']['scripts/audit_otto_return_value.py'], 'independent input audit')
    for split in ('train', 'valid') if mode == 'study' else ('train',):
        for suffix, role in (('data.npz', 'data'), ('rows.jsonl', 'rows')):
            name = split+'-'+suffix
            require(paths[split+'_'+role] == paths['prior_receipt'].parent/name
                    and descriptor(paths[split+'_'+role]) == worker['files'][name], 'original cached dataset identity')
    require(sys.executable == plan['python_executable'] and sys.version.split()[0] == plan['python_version']
            and {d.metadata['Name']: d.version for d in importlib.metadata.distributions()} == plan['all_distributions'], 'unchanged runtime')
    if mode == 'study':
        require(set(plan['qualification']) == {'plan', 'receipt', 'terminal'}, 'qualification roles')
        qpaths = {k: regular(v['path']) for k, v in plan['qualification'].items()}
        for k, p in qpaths.items():
            require(descriptor(p) == {n: plan['qualification'][k][n] for n in ('sha256', 'bytes')}, 'qualification pin')
        qp = authenticate(SimpleNamespace(plan=qpaths['plan'], plan_sha256=sha(qpaths['plan'])))
        qr = closed(qpaths['receipt'])
        require(qp['mode'] == qr['mode'] == 'qualify' and qr['completed_fits'] == 3 and qr['parity_passed']
                and qr['plan_sha256'] == sha(qpaths['plan']) and qr['sources'] == qp['sources']
                and qr['inputs'] == qp['inputs'] and qr['pending'] == [] and set(qr['files']) == payload_names('qualify')
                and qr['calls']['optimizer_update']['returned'] == 36
                and all(v['attempted'] == v['returned'] for v in qr['calls'].values()), 'completed disposable qualification')
        terminal_identity(qpaths['receipt'], qpaths['plan'], qpaths['terminal'], 'scripts/study_otto_capacity.py')
        require(all(plan['sources'].get(k) == v for k, v in qp['sources'].items()), 'qualified code unchanged')
        require(read(qpaths['receipt'].parent/'summary.json')['qualification_projection_seconds'] <= 1200, 'prospective cost admission')
    return plan


def alias_floor(x, y):
    groups = {}
    for row, value in zip(x, y, strict=True):
        groups.setdefault(row.tobytes(), []).append(float(value))
    noise = []
    for values in groups.values():
        mean = math.fsum(values)/len(values)
        noise.append(math.fsum((v-mean)**2 for v in values))
    return {'rows': len(y), 'unique_groups': len(groups), 'duplicate_groups': sum(len(v)>1 for v in groups.values()),
            'duplicate_rows': sum(len(v) for v in groups.values() if len(v)>1),
            'conflicting_groups': sum(min(v)!=max(v) for v in groups.values()),
            'irreducible_mse_normalized': math.fsum(noise)/len(y)}


def metrics(prediction, target, np):
    error = prediction-target.astype(np.float64)
    return {'rows': len(target), 'mse_normalized': float(np.mean(error**2)),
            'mae_physical': float(64*np.mean(np.abs(error))), 'negative_predictions': int((prediction<0).sum()),
            'minimum_normalized': float(prediction.min()), 'maximum_normalized': float(prediction.max())}


def gates(values, floor):
    checks = []
    for kind in KINDS[1:]:
        for seed in SEEDS:
            a, b = values[f'{kind}@{seed}'], values[f'mlp8@{seed}']
            for key, value, bound in (
                ('train_excess', a['train']['mse_normalized']-floor, .8*(b['train']['mse_normalized']-floor)),
                ('valid_mse', a['valid']['mse_normalized'], .9*b['valid']['mse_normalized'])):
                checks.append({'name': f'{kind}.{seed}.{key}', 'value': value, 'threshold': bound, 'passes': value<=bound})
    return checks


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.calls, self.pending, self.sequence, self.context = {}, [], 0, {}
        self.receipt = {'version': VERSION, 'status': 'started', 'completed_fits': 0, 'external_model_calls': 0,
                        'native_steps': 0, 'native_resets': 0, 'parity_passed': False}

    def check(self):
        require(self.clock.now_ns() < self.launch['deadline_ns'], 'native deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= self.plan['limits']['rss_bytes'], 'RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= self.plan['limits']['output_bytes'], 'output cap')

    def emit(self, name, row):
        with (self.out/name).open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
            stream.flush()

    def call(self, channel, operation):
        self.check()
        count = self.calls.setdefault(channel, {'attempted': 0, 'returned': 0, 'seconds': 0.})
        if channel == 'optimizer_update':
            require(count['attempted'] < self.plan['limits']['optimizer_updates'], 'update allocation before call')
        self.sequence += 1
        item = {'id': self.sequence, 'channel': channel, 'context': dict(self.context)}
        self.pending.append(item)
        count['attempted'] += 1
        self.emit('work.jsonl', {**item, 'event': 'attempt'})
        tick = time.perf_counter()
        result = operation()
        seconds = time.perf_counter()-tick
        count['seconds'] += seconds
        count['returned'] += 1
        require(self.pending.pop() == item, 'closed operation')
        self.emit('work.jsonl', {**item, 'event': 'return', 'seconds': seconds})
        return result

    def save(self, name, arrays):
        with (self.out/name).open('xb') as stream:
            self.np.savez_compressed(stream, **arrays)
        return sha(self.out/name)

    def bind(self):
        require(sha(ROOT/P.CLOCK) == P.CLOCK_PIN, 'qualified clock')
        self.clock = load(ROOT/P.CLOCK, '_capacity_clock').SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns()-self.start < 5*10**9, 'supervision missing')
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        self.plan = authenticate(self.args)
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
                and launch['clock_source_sha256'] == P.CLOCK_PIN, 'shared qualified supervision')
        self.receipt.update(mode=self.plan['mode'], limits=self.plan['limits'], sources=self.plan['sources'],
                            inputs=self.plan['inputs'], plan_sha256=self.args.plan_sha256,
                            supervision_sha256=sha(self.args.supervision))
        write(self.out/'started.json', {'request': {k: str(v) for k,v in vars(self.args).items()},
                                      'launch': launch, 'started_ns': self.start})
        self.check()

    def prepare(self):
        os.environ.update(THREADS)
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.np, self.torch = np, torch
        sys.path.insert(0, str(ROOT/'src'))
        from openjev.research import otto_capacity_value
        self.model = otto_capacity_value
        write(self.out/'runtime.json', {'python': sys.version, 'executable': sys.executable, 'threads': THREADS,
            'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
            'torch_deterministic': torch.are_deterministic_algorithms_enabled()})
        self.data, self.indices = {}, {}
        for split in ('train', 'valid') if self.plan['mode']=='study' else ('train',):
            n = 5589 if split=='train' else 1109
            with np.load(regular(self.plan['inputs'][split+'_data']['path']), allow_pickle=False) as a:
                x, y = a['features'], a['target']
            rows = [json.loads(line) for line in regular(self.plan['inputs'][split+'_rows']['path']).read_text().splitlines()]
            require(x.shape == (n,11028) and x.dtype == np.float32 and y.shape == (n,) and y.dtype == np.float32
                    and np.isfinite(x).all() and np.isfinite(y).all() and len(rows)==n, 'cached scalar TRAIN/VALID contract')
            require(len({r['episode_id'] for r in rows}) == (192 if split=='train' else 48), 'complete episode cohort')
            for i, r in enumerate(rows):
                require(r['row_index']==i and r['stage']==split and not r['public']['done']
                        and r['target']==float(y[i])==(r['total_steps']-r['prefix_index'])/64, 'cached target identity')
            if split=='train':
                self.c0 = float(np.float32(np.mean(y, dtype=np.float64)))
            selected = np.linspace(0,5588,256,dtype=np.int64) if self.plan['mode']=='qualify' else np.arange(n,dtype=np.int64)
            self.indices[split] = selected.tolist()
            self.data[split] = x[selected].copy(), y[selected].copy()
        self.floor = alias_floor(*self.data['train'])
        write(self.out/'preparation.json', {'row_indices': self.indices, 'c0_float32': self.c0, 'alias_floor': self.floor,
              'features': {k: {'sha256': hashlib.sha256(x.tobytes()).hexdigest(), 'rows': len(x)} for k,(x,y) in self.data.items()}})

    def fit(self, kind, seed):
        np, torch, cfg = self.np, self.torch, self.plan['configuration']
        fit_id = f'{kind}@{seed}'
        self.context = {'fit_id': fit_id}
        tick = time.perf_counter()
        model = self.call('model_initialization', lambda: self.model.make_head(kind,seed,self.c0))
        initial = self.call('checkpoint_export', lambda: self.model.export_head(model))
        initial_sha = self.save(f'initial-{kind}-{seed}.npz', initial)
        self.emit('initializations.jsonl', {'fit_id': fit_id, 'kind': kind, 'seed': seed, 'initial_sha256': initial_sha,
                                         'parameter_count': self.model.parameter_count(kind), 'c0_float32': self.c0})
        optimizer = self.call('optimizer_initialization', lambda: torch.optim.Adam(model.parameters(),lr=cfg['learning_rate']))
        require(not optimizer.state, 'fresh optimizer')
        x, y = self.data['train']
        tx, ty = torch.from_numpy(x.copy()), torch.from_numpy(y.copy())
        rng, total_updates = np.random.default_rng(seed+20000), 0
        for epoch in range(1,cfg['epochs']+1):
            order = rng.permutation(len(x))
            order_sha = hashlib.sha256(order.tobytes()).hexdigest()
            self.emit('epoch-orders.jsonl', {'fit_id':fit_id,'epoch':epoch,'order':order.tolist(),'sha256':order_sha})
            weighted = []
            for batch, offset in enumerate(range(0,len(x),cfg['batch_size'])):
                ids = order[offset:offset+cfg['batch_size']]
                self.context = {'fit_id':fit_id,'epoch':epoch,'batch':batch}
                def update(ids=ids):
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.mean((model(tx[ids])-ty[ids])**2)
                    require(bool(torch.isfinite(loss)), 'finite loss')
                    loss.backward()
                    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'all gradients finite')
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                    optimizer.step()
                    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'finite parameters')
                    return float(loss.detach()), float(norm)
                loss, norm = self.call('optimizer_update', update)
                total_updates += 1
                weighted.append(loss*len(ids))
                self.emit('updates.jsonl', {**self.context,'rows':len(ids),'loss':loss,'gradient_norm':norm,'update_index':total_updates})
            self.emit('fit-curves.jsonl', {'fit_id':fit_id,'epoch':epoch,'rows':len(x),'updates':len(weighted),
                      'training_mse_normalized':math.fsum(weighted)/len(x),'order_sha256':order_sha})
            if epoch%10==0 or epoch==cfg['epochs']:
                print(json.dumps({'fit':fit_id,'epoch':epoch,'mse':math.fsum(weighted)/len(x)}),flush=True)
        self.context = {'fit_id':fit_id}
        final = self.call('checkpoint_export',lambda:self.model.export_head(model))
        final_sha = self.save(f'final-{kind}-{seed}.npz',final)
        steps = {name:int(optimizer.state[p]['step'].item()) for name,p in model.named_parameters()}
        require(set(steps.values())=={total_updates}, 'all parameters updated')
        fit_seconds = time.perf_counter()-tick
        frozen = self.model.FrozenValue(final)
        double = self.call('parity_restore',lambda:copy.deepcopy(model).double().eval())
        parts = [v[0][:8].astype(np.float64) for v in self.data.values()]
        probe = np.concatenate([*parts,np.zeros((1,11028)),self.data['train'][0][:1].astype(np.float64)*.5])
        self.context = {'fit_id':fit_id,'backend':'numpy'}
        a = self.call('parity_forward',lambda:frozen.normalized(probe))
        self.context = {'fit_id':fit_id,'backend':'torch'}
        with torch.no_grad():
            z = self.call('parity_forward',lambda:double(torch.from_numpy(probe)).numpy())
        passed = bool(np.all(np.abs(a-z)<=1e-10+1e-10*np.abs(z)))
        self.emit('parity.jsonl', {'fit_id':fit_id,'features_sha256':hashlib.sha256(probe.tobytes()).hexdigest(),
                  'rows':len(probe),'numpy':a.tolist(),'torch':z.tolist(),'passed':passed})
        require(passed, 'fixed Torch/NumPy parity')
        predictions, stats = {}, {}
        for split,(vx,vy) in self.data.items():
            parts = []
            for offset in range(0,len(vx),256):
                self.context = {'fit_id':fit_id,'split':split,'offset':offset}
                features = vx[offset:offset+256].astype(np.float64)
                parts.append(self.call('saved_prediction',lambda features=features:frozen.normalized(features)))
            predictions[split] = np.concatenate(parts)
            stats[split] = metrics(predictions[split],vy,np)
        self.save(f'predictions-{kind}-{seed}.npz',predictions)
        record = {'fit_id':fit_id,'kind':kind,'seed':seed,'rows':len(x),'epochs':cfg['epochs'],'fit_seconds':fit_seconds,
                  'parameter_count':self.model.parameter_count(kind),'optimizer_steps':steps,'initial_sha256':initial_sha,
                  'checkpoint_sha256':final_sha,'metrics':stats}
        self.emit('fits.jsonl',record)
        self.fits.append(record)
        self.receipt['completed_fits'] += 1

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists() and not any(p.is_symlink() for p in self.out.parents), 'exclusive output')
        self.out.mkdir(parents=True,exist_ok=False)
        try:
            self.bind()
            self.prepare()
            self.fits = []
            for seed in self.plan['configuration']['seeds']:
                for kind in KINDS:
                    self.fit(kind,seed)
            self.receipt['parity_passed'] = True
            values = {r['fit_id']:r['metrics'] for r in self.fits}
            checks = gates(values,self.floor['irreducible_mse_normalized']) if self.plan['mode']=='study' else []
            projection = 3*math.fsum(r['fit_seconds'] for r in self.fits)*(3520/12) if self.plan['mode']=='qualify' else None
            result = {'version':VERSION,'mode':self.plan['mode'],'rows':{k:len(v[0]) for k,v in self.data.items()},
                      'alias_floor':self.floor,'metrics':values,'checks':checks,
                      'capacity_screen_passed':all(c['passes'] for c in checks) if checks else None,
                      'family_admission':{k:all(c['passes'] for c in checks if c['name'].startswith(k+'.')) for k in KINDS[1:]} if checks else {},
                      'qualification_projection_seconds':projection,'training_costs':{r['fit_id']:r['fit_seconds'] for r in self.fits},
                      'learned_architecture_advantage_established':False,
                      'scope':'Fixed-data scalar fitting only; no policy, simulator, held-out competence or novelty claim.'}
            write(self.out/'summary.json',result)
            require(authenticate(self.args)==self.plan, 'end source/runtime/input identity')
            require(not self.pending and all(c['attempted']==c['returned'] for c in self.calls.values())
                    and self.calls['optimizer_update']['returned']==self.plan['limits']['optimizer_updates'], 'complete fixed work')
            require({p.name for p in self.out.iterdir()}==payload_names(self.plan['mode']), 'closed output membership')
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status='completed',calls=self.calls,pending=self.pending,started_ns=self.start,finished_ns=finished,
                                clock_backend=self.clock.backend,wall_seconds=(finished-self.start)/1e9,
                                files={p.name:descriptor(p) for p in self.out.iterdir()})
            write(self.out/'receipt.json',self.receipt)
            self.check()
        except BaseException as error:
            self.receipt.update(status='failed',calls=self.calls,pending=self.pending,error=repr(error),traceback=traceback.format_exc())
            try:
                if (self.out/'receipt.json').exists():
                    (self.out/'receipt.json').rename(self.out/'invalid-completed-receipt.json')
                write(self.out/'failed.json',self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - Preserve original failure.
                error.add_note(f'Failure publication: {secondary!r}')
            raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan','output','supervision'):
        parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    Run(parser.parse_args()).execute()
