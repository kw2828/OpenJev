"""Prospective, bounded readout/state ablation on previously exposed DEV paths.

Metadata planning never imports numerical libraries. Original studies stay closed.
The supervised worker fits every branch before opening DEV; no TEST interface exists.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import time
import traceback
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = 'scripts/run_otto_readout_ablation.py'
VERSION = 'otto-readout-state-ablation-v1'
PRIOR_PLAN = 'output/otto-residual-reanalysis-v1/dev-evaluation-plan-01.json'
PRIOR_PIN = '24d7052f81aac7e4caaf232d60654de08df856606a02bc1153d5baaf4551c1e5'
CLOCK = 'src/openjev/research/suspend_clock.py'
SUPERVISOR = 'scripts/supervise_dialogue_observation_v2.py'
SEEDS = (309000001, 309000002, 309000003)
ARMS = ('action_residual_only', 'both_readouts', 'full_joint')
VIEWS = ('pretrained', *ARMS)
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')
LIMITS = {'train': {'seconds': 5400, 'rss_bytes': 4 * 1024**3, 'output_bytes': 512 * 1024**2},
          'audit': {'seconds': 600, 'rss_bytes': 2 * 1024**3, 'output_bytes': 128 * 1024**2}}
CONFIG = {'arms': list(ARMS), 'fit_seeds': list(SEEDS), 'epochs': 40, 'batch': 6, 'chunk': 32,
          'train_episodes': 54, 'dev_episodes': 18, 'updates_per_fit': 360, 'fits': 9, 'views': 12,
          'query_period': 4, 'evaluation_batch': 1, 'learning_rate': .003, 'clip': 5.,
          'objective': 'legal-centered nonquery MSE plus all-four prequery MSE, each in /64 units',
          'margin': .05, 'dev_reused': True, 'held_out_evidence': False, 'test_admitted': False,
          'continuation': 'full_joint reduces later case-weighted raw gap by at least 5% versus each readout and parent in every seed/setting, with no full-gap regression',
          'orders': 'PCG64(fit_seed) restarted for each branch; final checkpoints only'}
NEW = {SELF, 'scripts/audit_otto_readout_ablation.py', 'scripts/qualify_otto_readout_ablation.py',
       'scripts/qualify_otto_readout_capacity.py',
       'src/openjev/research/otto_readout_ablation_model.py',
       'src/openjev/research/otto_readout_ablation_training.py',
       'tests/test_otto_readout_ablation_model.py', 'tests/test_otto_readout_ablation_training.py',
       'tests/test_audit_otto_readout_ablation.py', 'tests/test_run_otto_readout_ablation.py',
       'research/otto-readout-state-ablation-protocol.md'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def regular(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_file() and path.is_relative_to(ROOT) and '..' not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents)), 'regular contained input')
    return path


def descriptor(value):
    path = regular(value)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def read(value):
    return json.loads(regular(value).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify(record):
    require(descriptor(record['path']) == record, 'unchanged exact input descriptor')


def authenticate():
    """Old provenance as bytes/JSON only, with a completed native bridge proof."""
    require(descriptor(PRIOR_PLAN)['sha256'] == PRIOR_PIN, 'closed prior analysis source anchor')
    prior = read(PRIOR_PLAN)
    sources = dict(prior['sources'])
    for name, pin in sources.items():
        require(descriptor(name)['sha256'] == pin, 'unchanged historical source: ' + name)
    lineage = load('scripts/otto_residual_lineage.py', '_readout_lineage').authenticate()
    old = load('scripts/train_otto_query_memory.py', '_readout_original')
    require(old.runtime_record() == lineage['runtime'], 'qualified original training runtime')
    inputs = {k: v for k, v in prior['inputs'].items() if k != 'engineering'}
    bridge = load('scripts/otto_residual_native_bridge.py', '_readout_bridge')
    dev_plan, dev_receipt, dev_dir = bridge.verify_bridge(inputs)
    train_plan = read(lineage['evidence']['prior_input:collection_plan']['path'])
    train_receipt = read(lineage['evidence']['prior_input:collection_receipt']['path'])
    train_dir = Path(lineage['evidence']['prior_input:collection_receipt']['path']).parent
    train_desc = descriptor(train_dir / 'train.npz')
    require({k: train_desc[k] for k in ('sha256', 'bytes')} == train_receipt['files']['train.npz'], 'TRAIN census identity')
    dev_desc = descriptor(dev_dir / 'dev.npz')
    require({k: dev_desc[k] for k in ('sha256', 'bytes')} == dev_receipt['files']['dev.npz'], 'DEV census identity')
    train_ids = [x for x in train_plan['cohort'] if x['stage'] == 'train']
    dev_ids = dev_plan['execution_cohort']
    require(len(train_ids) == 54 and len(dev_ids) == 18 and all(x['stage'] == 'dev' for x in dev_ids), 'complete admitted rosters')
    require(not ({x['seed'] for x in train_ids} & {x['seed'] for x in dev_ids}), 'distinct TRAIN/DEV case seeds')
    for name in NEW:
        sources[name] = descriptor(name)['sha256']
    return {'sources': sources, 'lineage': lineage, 'bridge_inputs': inputs,
            'train': {'descriptor': train_desc, 'identities': train_ids},
            'dev': {'descriptor': dev_desc, 'identities': dev_ids}, 'runtime': old.runtime_record()}


def engineering(path):
    record = read(path)
    require(record['status'] == 'passed' and record['sources_before'] == record['sources_after']
            and len(record['commands']) == 3 and record['metadata_handoff_passed']
            and all(x['returncode'] == 0 and not x['timed_out'] and x['reaped'] and x['group_absent']
                    for x in record['commands']), 'completed engineering qualification')
    for name, pin in record['sources_after'].items():
        require(descriptor(name)['sha256'] == pin, 'qualified source unchanged')
    require(NEW <= set(record['sources_after']), 'all new sources qualified')
    for item in record['files'].values():
        verify(item)
    capacity = read(record['files']['capacity.json']['path'])
    require(capacity['status'] == 'passed' and capacity['empirical_array_decodes'] == 0
            and capacity['checkpoint_decodes'] == 0 and capacity['projected_seconds'] <= 4050,
            'fabricated capacity below 75 percent of fixed cap')
    return descriptor(path)


def validate_plan(plan):
    require(plan['version'] == VERSION and plan['status'] == 'registered_before_gradients'
            and plan['configuration'] == CONFIG and plan['limits'] == LIMITS, 'fixed prospective recipe')
    actual = authenticate()
    require(all(plan[k] == v for k, v in actual.items()), 'unchanged sources, data, lineage and runtime')
    require(engineering(plan['engineering']['path']) == plan['engineering'], 'same qualification receipt')
    return actual


def freeze(args):
    auth = authenticate()
    qualified = engineering(args.engineering)
    require(not any(x in sys.modules for x in ('numpy', 'torch', 'tensorflow', 'jax', 'mlx')), 'metadata-only registration')
    write(args.output, {'version': VERSION, 'status': 'registered_before_gradients',
                       'created_unix_ns': time.time_ns(), 'configuration': CONFIG, 'limits': LIMITS,
                       'engineering': qualified, **auth})
    print(json.dumps({'status': 'registered_before_gradients', 'plan': descriptor(args.output)}), flush=True)


def expected_payloads(phase):
    if phase == 'audit':
        return {'started.json', 'runtime.json', 'audit.json'}
    require(phase == 'train', 'declared phase payloads')
    return {'started.json', 'runtime.json', 'work.jsonl', 'progress.jsonl', 'fits.json', 'views.json',
            'training-barrier.json', 'summary.json', 'train-history.json', 'train-history.npz',
            'dev-history.json', 'dev-history.npz'} | {
            f'checkpoint-{arm}-{seed}.npz' for seed in SEEDS for arm in ARMS} | {
            f'prediction-{family}-{seed}.npz' for seed in SEEDS for family in VIEWS}


def process_closure(planpath, receiptpath, terminalpath):
    plan, receipt, terminal = read(planpath), read(receiptpath), read(terminalpath)
    validate_plan(plan)
    phase = receipt['phase']
    require(phase in LIMITS and receipt['version'] == VERSION and receipt['status'] == 'completed'
            and receipt['pending'] is None and receipt['plan'] == descriptor(planpath), 'complete bound producer receipt')
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
            and terminal['error'] is None and terminal['clock_error'] is None
            and terminal['cleanup']['reaped'] and terminal['cleanup']['group_absent']
            and terminal['cleanup']['errors'] == [] and terminal['group_absent']
            and terminal['timing_available'] and terminal['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
            and terminal['pid'] == terminal['pgid'] != terminal['parent_pid']
            and Path(terminal['cwd']) == ROOT, 'original successful process closure')
    launch = read(receipt['supervision']['path'])
    verify(receipt['supervision'])
    require(terminal['command'] == launch['command'] and terminal['pid'] == launch['pid']
            and terminal['started_ns'] == launch['started_ns'] and terminal['deadline_ns'] == launch['deadline_ns']
            and terminal['cap_seconds'] == LIMITS[phase]['seconds']
            and terminal['finished_ns'] < terminal['deadline_ns']
            and launch['watchdog_sha256'] == plan['sources'][SUPERVISOR]
            and launch['clock_source_sha256'] == plan['sources'][CLOCK], 'same bounded original launch')
    require(all(terminal[k] == v for k, v in launch.items())
            and launch['started_ns'] <= receipt['started_ns'] < receipt['finished_ns'] <= terminal['finished_ns']
            and receipt['sources'] == plan['sources'] and receipt['limits'] == LIMITS[phase]
            and receipt['teacher_calls'] == receipt['native_calls'] == receipt['test_array_decodes'] == 0
            and set(receipt['files']) == expected_payloads(phase), 'complete original worker containment and scope')
    if phase == 'train':
        require(receipt['fits_completed'] == 9 and receipt['optimizer_steps'] == 3240
                and receipt['episode_exposures'] == 19440 and receipt['views_completed'] == 12
                and receipt['checkpoint_decodes'] == 3 and receipt['array_decodes'] == 5, 'complete training and view counters')
    else:
        require(receipt['agreement'] is True, 'independent audit agreement')
        verify(receipt['producer_receipt'])
        verify(receipt['producer_terminal'])
    command = terminal['command']
    require(command == [str(ROOT / '.venv/bin/python'), str(ROOT / SELF), phase,
                        '--plan', str(regular(planpath)), '--plan-sha256', descriptor(planpath)['sha256'],
                        '--supervision', receipt['supervision']['path'], '--output', str(regular(receiptpath).parent),
                        *receipt.get('extra_arguments', [])], 'exact admitted worker command')
    directory = regular(receiptpath).parent
    require({p.name for p in directory.iterdir()} == set(receipt['files']) | {'receipt.json'}, 'closed payload roster')
    started = read(directory / 'started.json')
    require(started['launch'] == launch and started['started_ns'] == receipt['started_ns'], 'worker start joins launch')
    for name, pin in receipt['files'].items():
        require(Path(pin['path']) == directory / name, 'contained declared payload')
        verify(pin)
    return plan, receipt, directory


class BoundRun:
    def __init__(self, args):
        self.args, self.phase, self.out = args, args.mode, args.output
        self.clock = self.start = self.launch = self.plan = None
        self.owns_output = False
        self.receipt = {'version': VERSION, 'phase': self.phase, 'status': 'started', 'pending': None,
                        'fits_completed': 0, 'optimizer_steps': 0, 'episode_exposures': 0,
                        'teacher_calls': 0, 'native_calls': 0, 'test_array_decodes': 0,
                        'checkpoint_decodes': 0, 'array_decodes': 0, 'views_completed': 0,
                        'counter_scope': 'completed returned operations; pending stage preserves uncertain partial work'}

    def initialize(self):
        self.out.mkdir(exist_ok=False)
        self.owns_output = True
        self.clock = load(CLOCK, '_readout_clock').SuspendClock()
        self.start = self.clock.now_ns()
        self.receipt['started_ns'] = self.start

    def check(self):
        if self.launch is not None:
            require(self.clock.now_ns() < self.launch['deadline_ns'], 'fixed original worker deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= LIMITS[self.phase]['rss_bytes'], 'RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir()) < LIMITS[self.phase]['output_bytes'] - 1024**2, 'output cap with failure reserve')

    def bind(self):
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, 'original launch available')
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        require(self.launch['command'] == [sys.executable, *sys.argv]
                and self.launch['pid'] == os.getpid() and self.launch['pgid'] == os.getpgrp()
                and self.launch['parent_pid'] == os.getppid() and Path.cwd() == ROOT
                and self.launch['cap_seconds'] == LIMITS[self.phase]['seconds']
                and self.launch['clock_backend'] == self.clock.backend
                and self.launch['started_ns'] <= self.start < self.launch['deadline_ns']
                and self.launch['deadline_ns'] == self.launch['started_ns'] + LIMITS[self.phase]['seconds'] * 10**9,
                'original detached worker identity')
        require(descriptor(self.args.plan)['sha256'] == self.args.plan_sha256, 'external plan pin')
        self.plan = read(self.args.plan)
        validate_plan(self.plan)
        require(self.launch['clock_source_sha256'] == self.plan['sources'][CLOCK]
                and self.launch['watchdog_sha256'] == self.plan['sources'][SUPERVISOR], 'bound supervisor code')
        require(all(os.environ.get(k) == '1' for k in THREADS), 'one numerical CPU thread')
        self.receipt.update(plan=descriptor(self.args.plan), supervision=descriptor(self.args.supervision),
                            sources=self.plan['sources'], limits=LIMITS[self.phase])
        self.publish('started.json', {'request': {k: str(v) for k, v in vars(self.args).items()},
                                     'launch': self.launch, 'started_ns': self.start})
        self.publish('runtime.json', self.plan['runtime'])
        self.check()

    def publish(self, name, value):
        self.check()
        write(self.out / name, value)
        return descriptor(self.out / name)

    def event(self, value, filename='work.jsonl'):
        require(filename in ('work.jsonl', 'progress.jsonl'), 'declared journal')
        self.check()
        with (self.out / filename).open('a') as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def save_arrays(self, name, arrays):
        self.check()
        with (self.out / name).open('xb') as stream:
            self.np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        return descriptor(self.out / name)

    def decode(self, pin):
        self.check()
        verify(pin)
        self.receipt['pending'] = {'decode': pin}
        with self.np.load(pin['path'], allow_pickle=False) as archive:
            result = {k: archive[k] for k in archive.files}
        self.receipt['array_decodes'] += 1
        self.receipt['pending'] = None
        return result

    def history(self, stage):
        require(stage in ('train', 'dev'), 'TRAIN and DEV only')
        if stage == 'dev':
            require(self.receipt['fits_completed'] == 9 and self.receipt['optimizer_steps'] == 3240
                    and (self.out / 'training-barrier.json').is_file(), 'all final fits closed before DEV')
        from openjev.research import otto_query_memory_data as data
        record = self.plan[stage]
        history = data.project_census(self.decode(record['descriptor']), record['identities'], query_period=4, expected_stage=stage)
        self.save_arrays(stage + '-history.npz', {k: v for k, v in history.items() if isinstance(v, self.np.ndarray)})
        self.publish(stage + '-history.json', {k: v for k, v in history.items() if not isinstance(v, self.np.ndarray)})
        return history

    def fit(self, family, seed, parent, history):
        from openjev.research import otto_readout_ablation_model as models
        from openjev.research import otto_readout_ablation_training as training
        old = load('scripts/train_otto_query_memory.py', '_readout_fit_helpers')
        tick = self.clock.now_ns()
        model = models.from_state(family, seed, parent)
        initial = old.state_witness(model)
        optimizer = training.construct_optimizer(model)
        frozen = [name for name, _ in model.named_parameters() if name not in dict(model.effective_named_parameters())]
        rng = self.np.random.Generator(self.np.random.PCG64(seed))
        permutations, orders = hashlib.sha256(), []
        steps = exposures = 0
        for epoch in range(40):
            order = rng.permutation(54).astype(self.np.int64)
            orders.append(order.tolist())
            permutations.update(order.tobytes())
            for offset in range(0, 54, 6):
                indices = order[offset:offset + 6].tolist()
                call = {'call_id': self.receipt['optimizer_steps'] + 1, 'family': family, 'seed': seed,
                        'epoch': epoch + 1, 'batch': offset // 6, 'episode_indices': indices}
                self.receipt['pending'] = call
                self.event({'event': 'attempt', **call})
                def stage(name, start, call=call):
                    self.receipt['pending'] = {**call, 'stage': name, 'start': start}
                result = training.batch_update(model, optimizer, history, indices, check=self.check, stage=stage)
                steps += 1
                exposures += 6
                self.receipt['optimizer_steps'] += 1
                self.receipt['episode_exposures'] += 6
                require(result['optimizer_step'] == steps and result['optimizer_updates'] == 1, 'one matched Adam update')
                self.event({'event': 'return', **call, 'result': result})
                self.receipt['pending'] = None
            print(json.dumps({'family': family, 'seed': seed, 'epoch': epoch + 1, 'updates': steps}), flush=True)
        final = old.state_witness(model)
        require(all(initial['tensors'][name] == final['tensors'][name] for name in frozen), 'frozen parameter bytes unchanged')
        pin = self.save_arrays(f'checkpoint-{family}-{seed}.npz', {k: v.detach().numpy().copy() for k, v in model.state_dict().items()})
        row = {'family': family, 'seed': seed, 'steps': steps, 'epochs': 40, 'exposures': exposures,
               'parameters': model.parameter_metadata(), 'initial': initial, 'final': final,
               'frozen_names': frozen, 'frozen_unchanged': True, 'permutation_sha256': permutations.hexdigest(),
               'orders': orders, 'seconds': (self.clock.now_ns() - tick) / 1e9, 'checkpoint': pin,
               'optimizer_steps': {k: int(optimizer.state[v]['step']) for k, v in model.effective_named_parameters()}}
        self.event({'event': 'fit_complete', **row}, 'progress.jsonl')
        self.receipt['fits_completed'] += 1
        return {k: v.detach().clone() for k, v in model.slow.state_dict().items()}, row

    def predict(self, family, seed, state, history):
        from openjev.research import otto_query_memory_data as data
        from openjev.research import otto_query_memory_metrics as metrics
        from openjev.research import otto_query_memory_model as models
        np, torch = self.np, self.torch
        tick = self.clock.now_ns()
        model = models.from_states(models.memory.Config('none', key_dim=8), seed, 4, state, None, slow_mode='frozen')
        model.eval()
        offsets = history['episode_offsets']
        total = int(offsets[-1])
        saved = {k: np.zeros((total, 4), np.float32) for k in ('action_prediction', 'corrected_shadow_prior')}
        saved.update(prior_mask=np.zeros(total, np.bool_), episode_offsets=offsets.copy())
        work = {}
        with torch.no_grad():
            for index, (low, high) in enumerate(pairwise(offsets)):
                low, high = int(low), int(high)
                carry = model.initial_carry(1)
                for start in range(0, high - low, 32):
                    self.check()
                    self.receipt['pending'] = {'view': family, 'seed': seed, 'episode': index, 'start': start}
                    packet = data.batch_chunk(history, [index], start)
                    inputs = {k: torch.from_numpy(v) for k, v in packet['model_inputs'].items()}
                    forecast = model(**inputs, carry=carry)
                    length = int(inputs['lengths'][0])
                    for name in ('action_prediction', 'corrected_shadow_prior', 'prior_mask'):
                        value = getattr(forecast, name)[0]
                        require(value[length:].detach().numpy().tobytes() == np.zeros_like(value[length:].detach().numpy()).tobytes(), 'positive zero padding')
                        saved[name][low + start:low + start + length] = value[:length].detach().numpy()
                    carry = models.detach_carry(forecast.carry)
                    for k, v in forecast.work_counts.items():
                        work[k] = work.get(k, 0) + v
                    self.receipt['pending'] = None
                require(bool(carry.slow.base.ended.all()) and int(carry.slow.base.absolute_step[0]) == high - low, 'complete DEV episode')
        require(saved['action_prediction'][history['query_mask']].tobytes() == history['query_scores'][history['query_mask']].tobytes(), 'exact observed query scores')
        require(np.array_equal(saved['prior_mask'], history['prior_mask']), 'exact later-query support')
        episodes = []
        for i, identity in enumerate(self.plan['dev']['identities']):
            low, high = map(int, offsets[i:i + 2])
            episodes.append(metrics.episode_metrics(identity, 4, history['targets'][low:high], history['legal'][low:high],
                saved['action_prediction'][low:high], prequery_forecast=saved['corrected_shadow_prior'][low:high]))
        report = metrics.aggregate_episodes(episodes, family=family, fit_seed=seed, expected_identities=self.plan['dev']['identities'])
        pin = self.save_arrays(f'prediction-{family}-{seed}.npz', saved)
        row = {'family': family, 'seed': seed, 'metrics': report, 'prediction': pin,
               'seconds': (self.clock.now_ns() - tick) / 1e9, 'work': work}
        self.event({'event': 'view_complete', **row}, 'progress.jsonl')
        self.receipt['views_completed'] += 1
        return row

    def train(self):
        import numpy as np
        import torch
        self.np, self.torch = np, torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        history = self.history('train')
        fitted, rows = {}, []
        for seed in SEEDS:
            raw = self.decode(self.plan['lineage']['checkpoints']['pretrained'][str(seed)])
            self.receipt['checkpoint_decodes'] += 1
            require(all(k.startswith('slow.') for k in raw), 'parent has only slow tensors')
            parent = {k.removeprefix('slow.'): torch.from_numpy(v.copy()) for k, v in raw.items()}
            fitted['pretrained', seed] = parent
            for family in ARMS:
                state, row = self.fit(family, seed, parent, history)
                fitted[family, seed] = state
                rows.append(row)
        fit_pin = self.publish('fits.json', rows)
        self.publish('training-barrier.json', {'fits_completed': 9, 'optimizer_steps': 3240,
            'episode_exposures': 19440, 'fits': fit_pin, 'checkpoints': [r['checkpoint'] for r in rows],
            'created_ns': self.clock.now_ns(), 'dev_decodes': 0})
        dev = self.history('dev')
        views = [self.predict(family, seed, fitted[family, seed], dev) for seed in SEEDS for family in VIEWS]
        self.publish('views.json', views)
        self.publish('summary.json', {'status': 'completed_pending_independent_audit', 'fits': 9, 'views': 12,
                                     'scope': 'reused development paths; no autonomous or held-out claim'})

    def audit(self):
        plan, receipt, directory = process_closure(self.args.plan, self.args.producer_receipt, self.args.producer_terminal)
        require(receipt['phase'] == 'train' and read(self.args.producer_terminal)['finished_ns'] <= self.launch['started_ns'], 'audit follows completed training')
        self.receipt['extra_arguments'] = ['--producer-receipt', str(self.args.producer_receipt), '--producer-terminal', str(self.args.producer_terminal)]
        self.receipt['producer_receipt'] = descriptor(self.args.producer_receipt)
        self.receipt['producer_terminal'] = descriptor(self.args.producer_terminal)
        import numpy as np
        audit = load('scripts/audit_otto_readout_ablation.py', '_readout_independent_audit')
        result = audit.audit(np, plan, receipt, directory, self.check)
        self.publish('audit.json', result)
        self.receipt['audit_counts'] = result['counts']
        self.receipt['array_decodes'] = result['counts']['array_decodes']
        self.receipt['checkpoint_decodes'] = result['counts']['checkpoint_decodes']
        self.receipt['views_completed'] = 12
        self.receipt['agreement'] = True

    def finish(self, error=None):
        if not self.owns_output:
            return
        if error is None:
            require(descriptor(self.args.plan)['sha256'] == self.args.plan_sha256, 'plan unchanged at finish')
            validate_plan(self.plan)
            verify(self.receipt['supervision'])
            self.check()
            require({p.name for p in self.out.iterdir()} == expected_payloads(self.phase), 'all expected outputs before success')
        try:
            finished = self.clock.now_ns() if self.clock is not None else None
        except BaseException as clock_error:
            if error is None:
                raise
            self.receipt['clock_error'] = repr(clock_error)
            finished = None
        self.receipt.update(status='completed' if error is None else 'failed', error=error,
                            finished_ns=finished,
                            wall_seconds=None if finished is None or self.start is None else (finished - self.start) / 1e9,
                            requires_original_supervisor_closure=True,
                            files={p.name: descriptor(p) for p in sorted(self.out.iterdir()) if p.is_file()})
        write(self.out / 'receipt.json', self.receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--engineering', type=Path, required=True)
    plan.add_argument('--output', type=Path, required=True)
    for phase in ('train', 'audit'):
        run = sub.add_parser(phase)
        for name in ('plan', 'supervision', 'output'):
            run.add_argument('--' + name, type=Path, required=True)
        run.add_argument('--plan-sha256', required=True)
        if phase == 'audit':
            run.add_argument('--producer-receipt', type=Path, required=True)
            run.add_argument('--producer-terminal', type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and args.output.is_absolute() and args.output.is_relative_to(ROOT)
            and '..' not in args.output.parts and not args.output.exists()
            and not any(p.is_symlink() for p in (args.output, *args.output.parents)), 'exclusive contained output')
    if args.mode == 'plan':
        freeze(args)
        return
    run = BoundRun(args)
    def interrupted(_signum, _frame):
        raise TimeoutError('original supervisor termination')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        run.initialize()
        run.bind()
        run.train() if args.mode == 'train' else run.audit()
        run.check()
        run.finish()
    except BaseException:
        error = traceback.format_exc()
        run.finish(error)
        raise


if __name__ == '__main__':
    main()
