"""Fresh five-family spatial scalar fitting on closed teacher caches only.

Authentication may hash prior payloads, but only the original TRAIN/VALID cache
is decoded. No qualification weights, branch policy, simulator or model service
is invoked. Final diagnostics use original float64 inputs in both backends.
"""
from __future__ import annotations

import argparse
import copy
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
BASE = 'scripts/study_otto_capacity.py'
BASE_PIN = '5043e550d17346b759bc8e1b3507a79721137b34c71ebfee79cccc807f7da7de'
VERSION = 'otto-spatial-study-v1'
KINDS = ('spatial', 'neighbor_free', 'cnn', 'dense128', 'statistics')
SEEDS = (10101, 10102, 10103)
ROW_COUNTS = {'train': 5589, 'valid': 1109}
EPISODE_COUNTS = {'train': 192, 'valid': 48}
FIRST_SEEDS = {'train': (910001, 920001), 'valid': (930001, 940001)}
AUDIT_LIMITS = {'seconds': 1800, 'rss_bytes': 4*1024**3, 'output_bytes': 128*1024**2}
NEW = ('scripts/study_otto_spatial.py', 'scripts/freeze_otto_spatial.py',
       'scripts/audit_otto_spatial_study.py', 'tests/test_otto_spatial_study.py',
       'tests/test_audit_otto_spatial_study.py', 'research/otto-spatial-training-protocol.md',
       'research/otto-spatial-training-design.md', 'scripts/audit_otto_spatial_qualification.py')
QUALIFICATION = {
    'plan': ('output/otto-spatial-qualification-v1/plan-01.json', '417c8d8504bc5c41e4231d2f188f21a3858adcedc6497bd3108fabb69dccac8e'),
    'receipt': ('output/otto-spatial-qualification-v1/run-01/receipt.json', '4af3c4d2d912333a2ad4cc7d7ab185bdfa3e0eada7ea4bde916e3d91741241f6'),
    'terminal': ('output/otto-spatial-qualification-v1/process-01.terminal.json', 'f41ef39d1abccf9f254ea2f18e3184dc14f5cbef01e305b0d2ac4f2b373696ad'),
    'audit': ('output/otto-spatial-qualification-v1/audit-01/receipt.json', 'c7eacb73114531b85c83ba273a97906138eb8f29dcc0a7382b7198a28f8cc9fb'),
}


def load_base():
    if hashlib.sha256((ROOT/BASE).read_bytes()).hexdigest() != BASE_PIN:
        raise ValueError('unchanged capacity lifecycle')
    spec = importlib.util.spec_from_file_location('_spatial_capacity', ROOT/BASE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


C = load_base()
P = C.P
require, sha, load = C.require, C.sha, C.load
read, write, regular, descriptor, closed = C.read, C.write, C.regular, C.descriptor, C.closed
THREADS = C.THREADS


def configuration():
    return {'kinds': list(KINDS), 'seeds': list(SEEDS), 'train_rows': 5589, 'valid_rows': 1109,
            'epochs': 80, 'batch_size': 128, 'prediction_batch': 16, 'learning_rate': .001,
            'gradient_clip': 5., 'shuffle_seed_offset': 20000, 'checkpoint': 'fixed final epoch80',
            'target': '(T-t)/64; uniform row MSE', 'baseline': 'float32(mean(TRAIN float32 targets, dtype=float64))',
            'initialization': 'fresh local generator; hidden He; zero biases and final readout; no restored qualification weights',
            'training_inputs': 'original float64 centered posterior cast to float32; explicit position and sensing',
            'diagnostic_inputs': 'original float64 centered posterior and sensing; explicit int64 position',
            'parity_atol': 1e-8, 'parity_rtol': 1e-10, 'scalar_admission_gate': False,
            'validation_selection': False, 'alias_floor': None, 'policy_evaluation': False}


def expected_calls(cfg=None):
    cfg = configuration() if cfg is None else cfg
    fits = len(cfg['kinds'])*len(cfg['seeds'])
    updates = fits*cfg['epochs']*math.ceil(cfg['train_rows']/cfg['batch_size'])
    predictions = fits*sum(math.ceil(cfg[s+'_rows']/cfg['prediction_batch']) for s in ('train', 'valid'))
    return {'model_initialization': fits, 'checkpoint_export': 2*fits, 'optimizer_initialization': fits,
            'training_forward': updates, 'backward': updates, 'optimizer_update': updates,
            'deployment_construction': fits, 'parity_restore': fits,
            'numpy_prediction': predictions, 'torch_prediction': predictions}


def limits():
    return {'native_seconds': 7200, 'rss_bytes': 8*1024**3, 'output_bytes': 2*1024**3,
            'optimizer_updates': 52800, 'native_steps': 0, 'native_resets': 0, 'external_model_calls': 0}


def payload_names():
    names = {'started.json', 'runtime.json', 'preparation.json', 'summary.json', 'work.jsonl',
             'initializations.jsonl', 'epoch-orders.jsonl', 'updates.jsonl', 'fit-curves.jsonl',
             'fits.jsonl', 'parity.jsonl'}
    names.update(f'{phase}-{kind}-{seed}.npz' for seed in SEEDS for kind in KINDS
                 for phase in ('initial', 'final', 'predictions'))
    return names


def authenticate_qualification(binding):
    require(set(binding) == set(QUALIFICATION), 'exact synthetic qualification roles')
    paths = {}
    for role, (name, pin) in QUALIFICATION.items():
        paths[role] = regular(name)
        require(binding[role] == {'path': name, **descriptor(paths[role])}
                and sha(paths[role]) == pin, 'original qualification pin')
    qp, terminal = read(paths['plan']), read(paths['terminal'])
    worker, audit = closed(paths['receipt']), closed(paths['audit'])
    require(qp['version'] == worker['version'] == 'otto-spatial-qualification-v1'
            and qp['status'] == 'frozen_before_execution' and qp['root'] == str(ROOT)
            and len(qp['sources']) == 9 and worker['sources'] == qp['sources']
            and worker['plan_sha256'] == sha(paths['plan']) and worker['errors'] == [], 'qualified synthetic source identity')
    files = {'started.json', 'runtime.json', 'fixtures.npz', 'calls.jsonl', 'operations.jsonl',
             'synthetic-losses.jsonl', 'summary.json'}
    files.update(f'{prefix}-{kind}.npz' for prefix in ('disposable', 'parity') for kind in KINDS)
    require(set(worker['files']) == files and worker['calls'] == {
        k: {'attempted': n, 'returned': n} for k, n in qp['expected_calls'].items()}, 'complete synthetic work')
    require(all(worker[k] == 0 for k in ('scientific_data_rows', 'native_steps', 'native_resets', 'external_model_calls')),
            'synthetic qualification only')
    for name, pin in qp['sources'].items():
        require(sha(regular(name)) == pin, 'unchanged qualified source')
    started = read(paths['receipt'].parent/'started.json')
    launch = started['launch']
    supervision = Path(started['request']['supervision'])
    require(supervision.is_absolute() and not any(p.is_symlink() for p in (supervision, *supervision.parents))
            and read(supervision) == launch, 'qualification launch identity')
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
            and terminal['error'] is None and terminal['clock_error'] is None and terminal['timing_available'] is True
            and terminal['group_absent'] is True and terminal['cleanup']['reaped'] is True
            and terminal['cleanup']['group_absent'] is True and terminal['cleanup']['errors'] == [], 'original qualification parent completed')
    require(all(terminal[k] == v for k, v in launch.items()), 'qualification parent/launch joins')
    require(launch['command'] == [qp['runtime']['python_executable'], str(ROOT/'scripts/qualify_otto_spatial.py'), 'run',
            '--plan', str(paths['plan']), '--plan-sha256', sha(paths['plan']), '--output', str(paths['receipt'].parent),
            '--supervision', str(supervision)] and launch['cwd'] == str(ROOT)
            and launch['pid'] == launch['pgid'] != launch['parent_pid'], 'exact original qualification command')
    require(launch['cap_seconds'] == qp['limits']['seconds'] == 600
            and launch['deadline_ns'] == launch['started_ns']+600*10**9
            and launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns']
            and worker['wall_seconds'] == (worker['finished_ns']-worker['started_ns'])/1e9
            and terminal['elapsed_ns'] == terminal['finished_ns']-terminal['started_ns']
            and terminal['wall_seconds'] == terminal['elapsed_ns']/1e9
            and 0 < worker['peak_rss_bytes'] <= qp['limits']['rss_bytes'], 'qualified original resource bounds')
    require(set(audit['files']) == {'started.json', 'summary.json'} and audit['agreement'] is True
            and audit['source']['sha256'] == sha(regular('scripts/audit_otto_spatial_qualification.py'))
            and all(audit['inputs'][str(path)] == descriptor(path) for path in (*paths.values(), supervision) if path != paths['audit'])
            and audit['model_forward_calls'] == audit['optimizer_calls'] == audit['native_calls'] == 0,
            'independent saved qualification audit')
    require(read(paths['receipt'].parent/'runtime.json') == qp['runtime'], 'qualification runtime witness')
    return qp


def authenticate(args):
    require(args.plan.is_absolute() and not any(p.is_symlink() for p in (args.plan, *args.plan.parents))
            and sha(args.plan) == args.plan_sha256, 'external plan pin before decode')
    plan = read(args.plan)
    require(plan['version'] == VERSION and plan['status'] == 'frozen_before_execution'
            and plan['mode'] == 'study' and plan['configuration'] == configuration()
            and plan['limits'] == limits() and plan['expected_calls'] == expected_calls()
            and plan['independent_audit_limits'] == AUDIT_LIMITS, 'fixed spatial study allocation')
    roles = {'capacity_plan', 'capacity_receipt', 'capacity_terminal', 'capacity_audit',
             'train_data', 'train_rows', 'valid_data', 'valid_rows'}
    require(set(plan['inputs']) == roles and set(NEW) <= set(plan['sources'])
            and plan['sources'][BASE] == BASE_PIN, 'exact prior input roles and required sources')
    for name, pin in plan['sources'].items():
        require(sha(regular(name)) == pin, 'unchanged source: '+name)
    paths = {}
    for role, item in plan['inputs'].items():
        paths[role] = regular(item['path'])
        require(descriptor(paths[role]) == {k: item[k] for k in ('bytes', 'sha256')}, 'bound prior input')
    prior = C.authenticate(SimpleNamespace(plan=paths['capacity_plan'], plan_sha256=sha(paths['capacity_plan'])))
    worker, audit = closed(paths['capacity_receipt']), closed(paths['capacity_audit'])
    require(prior['mode'] == worker['mode'] == 'study' and worker['completed_fits'] == 9
            and worker['pending'] == [] and worker['parity_passed'] is True
            and worker['plan_sha256'] == sha(paths['capacity_plan']) and worker['sources'] == prior['sources']
            and worker['inputs'] == prior['inputs'] and set(worker['files']) == C.payload_names('study'), 'closed capacity lineage')
    C.terminal_identity(paths['capacity_receipt'], paths['capacity_plan'], paths['capacity_terminal'], BASE)
    require(audit['agreement'] is True and audit['worker_sha256'] == sha(paths['capacity_receipt'])
            and audit['plan_sha256'] == sha(paths['capacity_plan']) and audit['terminal_sha256'] == sha(paths['capacity_terminal'])
            and audit['source']['sha256'] == prior['sources']['scripts/audit_otto_capacity.py'], 'independent capacity audit')
    qp = authenticate_qualification(plan['qualification'])
    union = dict(prior['sources'])
    for name, pin in qp['sources'].items():
        require(name not in union or union[name] == pin, 'consistent source union')
        union[name] = pin
    union.update({name: sha(regular(name)) for name in NEW})
    require(plan['sources'] == union, 'exact inherited and additive source closure')
    for role in ('train_data', 'train_rows', 'valid_data', 'valid_rows'):
        require(plan['inputs'][role] == prior['inputs'][role], 'original cached dataset; no new collection')
    require(qp['runtime']['python_executable'] == prior['python_executable']
            and qp['runtime']['python_version'].split()[0] == prior['python_version']
            and qp['runtime']['distributions'] == prior['all_distributions'], 'same qualified numerical runtime')
    for key in ('python_executable', 'python_version', 'all_distributions'):
        require(plan[key] == prior[key], 'same frozen runtime')
    return plan


def center_batch(beliefs, positions, dtype='float64'):
    import numpy as np
    require(dtype in ('float32', 'float64') and isinstance(beliefs, np.ndarray) and beliefs.dtype == np.float64
            and beliefs.ndim == 3 and beliefs.shape[1:] == (53, 53) and 0 < len(beliefs) <= 128
            and np.isfinite(beliefs).all() and (beliefs >= 0).all(), 'bounded original float64 posterior batch')
    require(isinstance(positions, np.ndarray) and positions.dtype == np.int64 and positions.shape == (len(beliefs), 2)
            and ((positions >= 0) & (positions <= 52)).all(), 'explicit in-board int64 positions')
    out = np.zeros((len(beliefs), 105, 105), dtype=np.float64)
    for i, (x, y) in enumerate(positions):
        out[i, 52-x:105-x, 52-y:105-y] = beliefs[i]
    require((out.reshape(len(out), -1).sum(axis=1) <= 1+1e-6).all(), 'bounded raw mass')
    return out.astype(dtype, copy=False)


def validate_dataset(arrays, rows, split):
    """Validate all tuples; production cohort sizes/order are checked by prepare.

The generic positive-row contract permits short synthetic complete episodes.
It does not authorize selecting any subset of the fixed empirical cache.
"""
    import numpy as np
    require(split in ROW_COUNTS and isinstance(rows, list) and len(rows) > 0
            and set(arrays) == {'features', 'target', 'beliefs', 'positions', 'sensing_length'}, 'exact original dataset schema')
    n = len(rows)
    for name, shape, dtype in (('features', (n, 11028), np.float32), ('target', (n,), np.float32),
                               ('beliefs', (n, 53, 53), np.float64), ('positions', (n, 2), np.int64),
                               ('sensing_length', (n,), np.float64)):
        a = arrays[name]
        require(isinstance(a, np.ndarray) and a.shape == shape and a.dtype == dtype and np.isfinite(a).all(), 'cache dtype/shape/finiteness')
    episodes, previous, last = [], None, None
    keys = {'row_index', 'episode_id', 'stage', 'regime', 'seed', 'initial_hit', 'prefix_index',
            'total_steps', 'public', 'posterior', 'target'}
    for i, row in enumerate(rows):
        require(set(row) == keys and type(row['row_index']) is int and row['row_index'] == i
                and row['stage'] == split and row['regime'] in ('lambda3', 'lambda4')
                and type(row['seed']) is int and type(row['initial_hit']) is int and row['initial_hit'] in (1, 2, 3)
                and type(row['total_steps']) is int and 1 <= row['total_steps'] <= 2188
                and type(row['prefix_index']) is int and 0 <= row['prefix_index'] < row['total_steps'], 'canonical row identity')
        identity = (row['episode_id'], row['regime'], row['seed'], row['initial_hit'], row['total_steps'])
        require(row['episode_id'] == f"{split}:{row['regime']}:{row['seed']}:teacher", 'teacher-only episode identity')
        if previous is None or identity != previous:
            require(row['prefix_index'] == 0 and (previous is None or last == previous[-1]-1)
                    and row['episode_id'] not in {e[0] for e in episodes}, 'complete contiguous unique teacher episodes')
            episodes.append(identity)
        else:
            require(row['prefix_index'] == last+1, 'every pre-action teacher prefix exactly once')
        previous, last = identity, row['prefix_index']
        p, q, public = arrays['beliefs'][i], arrays['positions'][i], row['public']
        x, y = map(int, q)
        allowed = [a for a, okay in enumerate((x > 0, x < 52, y > 0, y < 52)) if okay]
        require(set(public) == {'position', 'hit', 'done', 'step', 'valid_actions'}
                and public['position'] == [x, y] and public['done'] is False
                and type(public['step']) is int and public['step'] == last
                and type(public['hit']) is int and 0 <= public['hit'] <= 3 and public['valid_actions'] == allowed
                and (last != 0 or public['hit'] == row['initial_hit']),
                'explicit nonterminal public position and step')
        z = center_batch(p[None], q[None])[0]
        mass, sensing = float(z.sum()), float(int(row['regime'][-1]))
        require(row['posterior'] == {'sha256': hashlib.sha256(p.tobytes()).hexdigest(), 'mass': float(p.sum())}, 'cached posterior hash')
        legacy = np.concatenate((z.ravel(), [mass*x/52, mass*y/52, mass*sensing/5])).astype(np.float32)
        target = np.float32((row['total_steps']-last)/64)
        require(np.array_equal(legacy, arrays['features'][i]) and arrays['sensing_length'][i] == sensing
                and arrays['target'][i] == target and row['target'] == float(target), 'legacy features/target/sensing join')
        require(np.array_equal(z[52-x:105-x, 52-y:105-y], p), 'lossless physical round trip')
    require(last == previous[-1]-1, 'last teacher episode complete')
    return {'rows': n, 'episodes': [list(e) for e in episodes],
            'array_sha256': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()},
            'metadata_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}


def metrics(prediction, target):
    import numpy as np
    require(prediction.dtype == np.float64 and prediction.shape == target.shape and np.isfinite(prediction).all(), 'finite final scalar predictions')
    error = prediction-target.astype(np.float64)
    return {'rows': len(target), 'mse_normalized': float(np.mean(error**2)), 'mae_physical': float(64*np.mean(np.abs(error))),
            'negative_predictions': int((prediction < 0).sum()), 'minimum_normalized': float(prediction.min()),
            'maximum_normalized': float(prediction.max())}


def aggregate(fits):
    expected = [f'{kind}@{seed}' for seed in SEEDS for kind in KINDS]
    require([r['fit_id'] for r in fits] == expected, 'all fifteen final fits in fixed order')
    means = {}
    for kind in KINDS:
        group = [r for r in fits if r['kind'] == kind]
        means[kind] = {s: {key: math.fsum(r['metrics'][s][key] for r in group)/len(group)
                          for key in ('mse_normalized', 'mae_physical', 'negative_predictions',
                                      'minimum_normalized', 'maximum_normalized')}
                       for s in ('train', 'valid')}
    return {'version': VERSION, 'metrics': {r['fit_id']: r['metrics'] for r in fits}, 'family_means': means,
            'training_costs': {r['fit_id']: r['fit_seconds'] for r in fits},
            'diagnostic_costs': {r['fit_id']: r['diagnostic_seconds'] for r in fits},
            'rows': dict(ROW_COUNTS), 'fits': expected, 'scalar_admission_gate': None, 'alias_floor': None,
            'learned_architecture_advantage_established': False,
            'scope': 'All final original-float64 scalar predictions; exposed VALID descriptive only. No autonomous control, architecture novelty or superiority admission.'}


class Run(C.Run):
    def __init__(self, args):
        super().__init__(args)
        self.receipt.update(version=VERSION, initial_pairing_passed=False)
        self.fits, self.initial_pins = [], {}

    def call(self, channel, operation):
        require(channel in self.plan['expected_calls'], 'declared operation only')
        require(self.calls.get(channel, {}).get('attempted', 0) < self.plan['expected_calls'][channel], 'operation cap before call')
        result = super().call(channel, operation)
        self.check()
        return result

    def bind(self):
        require(sha(ROOT/P.CLOCK) == P.CLOCK_PIN, 'qualified native clock')
        self.clock = load(ROOT/P.CLOCK, '_spatial_clock').SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns()-self.start < 5*10**9, 'supervision missing')
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        require(self.clock.now_ns() < self.launch['deadline_ns'], 'parent deadline before authentication')
        self.plan = authenticate(self.args)
        launch, command = self.launch, list(self.launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and launch['pid'] == os.getpid()
                and launch['pgid'] == os.getpgrp() and launch['parent_pid'] == os.getppid()
                and launch['cwd'] == str(ROOT) == str(Path.cwd()), 'actual supervised worker identity')
        require(launch['clock_backend'] == self.clock.backend and launch['started_ns'] <= self.start < launch['deadline_ns']
                and launch['cap_seconds'] == self.plan['limits']['native_seconds']
                and launch['deadline_ns'] == launch['started_ns']+launch['cap_seconds']*10**9
                and launch['watchdog_sha256'] == self.plan['sources']['scripts/supervise_dialogue_observation_v2.py']
                and launch['clock_source_sha256'] == P.CLOCK_PIN, 'original bounded supervision')
        self.receipt.update(mode='study', limits=self.plan['limits'], sources=self.plan['sources'], inputs=self.plan['inputs'],
                            qualification=self.plan['qualification'], plan_sha256=self.args.plan_sha256,
                            supervision_sha256=sha(self.args.supervision))
        write(self.out/'started.json', {'request': {k: str(v) for k, v in vars(self.args).items()},
                                      'launch': launch, 'started_ns': self.start})
        self.check()

    def prepare(self):
        tick = time.perf_counter()
        require(all(os.environ.get(k) == v for k, v in THREADS.items()), 'one CPU thread before framework import')
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.np, self.torch = np, torch
        sys.path.insert(0, str(ROOT/'src'))
        from openjev.research import otto_spatial_value
        self.model = otto_spatial_value
        require(tuple(self.model.KINDS) == KINDS, 'qualified five model families')
        write(self.out/'runtime.json', {'python': sys.version, 'executable': sys.executable, 'threads': THREADS,
              'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
              'torch_deterministic': torch.are_deterministic_algorithms_enabled(), 'model_source_sha256': sha(ROOT/'src/openjev/research/otto_spatial_value.py')})
        self.data, self.evidence = {}, {}
        for split in ('train', 'valid'):
            self.check()
            with np.load(regular(self.plan['inputs'][split+'_data']['path']), allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
            rows = [json.loads(line) for line in regular(self.plan['inputs'][split+'_rows']['path']).read_text().splitlines()]
            evidence = validate_dataset(arrays, rows, split)
            count = EPISODE_COUNTS[split]//2
            expected = [(f'{split}:lambda{lam}:{first+case}:teacher', f'lambda{lam}', first+case, 1+case % 3)
                        for lam, first in zip((3, 4), FIRST_SEEDS[split], strict=True) for case in range(count)]
            require(evidence['rows'] == ROW_COUNTS[split] and [tuple(e[:4]) for e in evidence['episodes']] == expected,
                    'complete original teacher cohort and all prefixes')
            self.evidence[split] = evidence
            self.data[split] = {k: v for k, v in arrays.items() if k != 'features'}
            self.check()
        self.c0 = float(np.float32(np.mean(self.data['train']['target'], dtype=np.float64)))
        source_plan = read(regular(self.plan['inputs']['capacity_plan']['path']))
        original_receipt = regular(source_plan['inputs']['prior_receipt']['path'])
        require(read(original_receipt.parent/'preparation.json')['c0_float32'] == self.c0, 'same original TRAIN-only c0')
        self.preparation_seconds = time.perf_counter()-tick
        write(self.out/'preparation.json', {'rows': dict(ROW_COUNTS), 'episodes': dict(EPISODE_COUNTS),
              'c0_float32': self.c0, 'datasets': self.evidence, 'preparation_seconds': self.preparation_seconds,
              'training_view': 'lossless centered float64 then float32 cast',
              'diagnostic_view': 'original centered float64; no cached-float32 upcast', 'alias_floor': None})

    def fit(self, kind, seed):
        np, torch, cfg = self.np, self.torch, self.plan['configuration']
        fit_id, fit_start = f'{kind}@{seed}', time.perf_counter()
        self.context = {'fit_id': fit_id}
        model = self.call('model_initialization', lambda: self.model.make_head(kind, seed, self.c0))
        initial = self.call('checkpoint_export', lambda: self.model.export_head(model))
        require(self.model.validate_head(initial) == kind and float(initial['c0']) == self.c0
                and all(not np.any(v) for k, v in initial.items() if 'bias' in k or k == 'readout_weight_1'),
                'algebraic common initial function with zero final readout')
        initial_sha = self.save(f'initial-{kind}-{seed}.npz', initial)
        tensors = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in initial.items() if isinstance(v, np.ndarray)}
        if kind == 'spatial':
            self.initial_pins[seed] = tensors
        if kind == 'neighbor_free':
            require(self.initial_pins.get(seed) == tensors, 'paired spatial and neighbor-free initialization')
        self.emit('initializations.jsonl', {'fit_id': fit_id, 'kind': kind, 'seed': seed, 'initial_sha256': initial_sha,
                  'tensor_sha256': tensors, 'c0_float32': self.c0, 'zero_final_and_biases': True,
                  'paired_spatial_tensors': True if kind == 'neighbor_free' else None,
                  'parameter_count': self.model.parameter_count(kind)})
        optimizer = self.call('optimizer_initialization', lambda: torch.optim.Adam(model.parameters(), lr=cfg['learning_rate']))
        require(not optimizer.state, 'fresh empty Adam state')
        train, n = self.data['train'], len(self.data['train']['target'])
        rng, updates = np.random.default_rng(seed+cfg['shuffle_seed_offset']), 0
        training_start = time.perf_counter()
        for epoch in range(1, cfg['epochs']+1):
            order = rng.permutation(n).astype(np.int64, copy=False)
            order_sha = hashlib.sha256(order.tobytes()).hexdigest()
            self.emit('epoch-orders.jsonl', {'fit_id': fit_id, 'epoch': epoch, 'rows': n, 'order': order.tolist(), 'sha256': order_sha})
            losses = []
            for batch, offset in enumerate(range(0, n, cfg['batch_size'])):
                ids = order[offset:offset+cfg['batch_size']]
                self.context = {'fit_id': fit_id, 'epoch': epoch, 'batch': batch}
                self.check()
                z = center_batch(train['beliefs'][ids], train['positions'][ids], dtype='float32')
                inputs = (torch.from_numpy(z), torch.from_numpy(train['positions'][ids]),
                          torch.from_numpy(train['sensing_length'][ids].astype(np.float32)))
                y = torch.from_numpy(train['target'][ids])
                optimizer.zero_grad(set_to_none=True)
                prediction = self.call('training_forward', lambda inputs=inputs: model(*inputs))
                loss = torch.mean((prediction-y)**2)
                require(bool(torch.isfinite(loss)), 'finite pre-update uniform row MSE')
                self.call('backward', loss.backward)
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'finite present gradients')
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                self.call('optimizer_update', optimizer.step)
                require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'finite updated weights')
                updates += 1
                value = float(loss.detach())
                losses.append(value*len(ids))
                self.emit('updates.jsonl', {**self.context, 'rows': len(ids), 'batch_indices_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
                          'loss': value, 'gradient_norm': float(norm), 'update_index': updates})
            self.emit('fit-curves.jsonl', {'fit_id': fit_id, 'epoch': epoch, 'rows': n, 'updates': len(losses),
                      'training_mse_normalized': math.fsum(losses)/n, 'order_sha256': order_sha,
                      'scope': 'row-weighted mean of pre-update minibatch losses; not final-checkpoint MSE'})
            if epoch % 10 == 0 or epoch == cfg['epochs']:
                print(json.dumps({'fit_id': fit_id, 'epoch': epoch, 'updates': updates}), flush=True)
        training_seconds = time.perf_counter()-training_start
        self.context = {'fit_id': fit_id}
        final = self.call('checkpoint_export', lambda: self.model.export_head(model))
        final_sha = self.save(f'final-{kind}-{seed}.npz', final)
        steps = {name: int(optimizer.state[p]['step'].item()) for name, p in model.named_parameters()}
        require(set(steps.values()) == {updates} and updates == cfg['epochs']*math.ceil(n/cfg['batch_size']), 'every parameter received fixed updates')
        fit_seconds = time.perf_counter()-fit_start
        diagnostic_start = time.perf_counter()

        def restore():
            with np.load(self.out/f'final-{kind}-{seed}.npz', allow_pickle=False) as archive:
                saved = {k: archive[k] for k in archive.files}
            require(set(saved) == set(final) and all(np.array_equal(saved[k], final[k]) for k in saved), 'published final checkpoint consumed')
            return self.model.FrozenValue(saved)

        frozen = self.call('deployment_construction', restore)
        double = self.call('parity_restore', lambda: copy.deepcopy(model).double().eval())
        prediction_start = time.perf_counter()
        deployment_setup_seconds = prediction_start-diagnostic_start
        predictions, measured, max_error, parity_rows = {}, {}, 0., 0
        for split in ('train', 'valid'):
            data = self.data[split]
            actual, reference = [], []
            for offset in range(0, len(data['target']), cfg['prediction_batch']):
                stop = min(offset+cfg['prediction_batch'], len(data['target']))
                self.context = {'fit_id': fit_id, 'split': split, 'offset': offset}
                z = center_batch(data['beliefs'][offset:stop], data['positions'][offset:stop])
                q, sensing = data['positions'][offset:stop], data['sensing_length'][offset:stop]
                a = self.call('numpy_prediction', lambda z=z, q=q, sensing=sensing: frozen.normalized(z, q, sensing))
                with torch.no_grad():
                    b = self.call('torch_prediction', lambda z=z, q=q, sensing=sensing:
                                  double(torch.from_numpy(z), torch.from_numpy(q), torch.from_numpy(sensing)).numpy())
                delta = float(np.max(np.abs(a-b)))
                passed = bool(np.all(np.abs(a-b) <= cfg['parity_atol']+cfg['parity_rtol']*np.abs(b)))
                self.emit('parity.jsonl', {**self.context, 'rows': stop-offset,
                          'input_sha256': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in
                                           (('centered', z), ('positions', q), ('sensing_length', sensing))},
                          'maximum_absolute_difference': delta, 'passed': passed})
                require(passed, 'all final original-float64 Torch/NumPy scalar parity')
                actual.append(a)
                reference.append(b)
                max_error, parity_rows = max(max_error, delta), parity_rows+len(a)
            predictions[split+'_numpy'], predictions[split+'_torch64'] = np.concatenate(actual), np.concatenate(reference)
            measured[split] = metrics(predictions[split+'_numpy'], data['target'])
        prediction_sha = self.save(f'predictions-{kind}-{seed}.npz', predictions)
        diagnostic_end = time.perf_counter()
        diagnostic_seconds = diagnostic_end-diagnostic_start
        prediction_seconds = diagnostic_end-prediction_start
        record = {'fit_id': fit_id, 'kind': kind, 'seed': seed, 'epochs': cfg['epochs'], 'training_rows': n,
                  'validation_rows': len(self.data['valid']['target']), 'updates': updates, 'optimizer_steps_before': {},
                  'optimizer_steps': steps, 'parameter_count': self.model.parameter_count(kind), 'c0_float32': self.c0,
                  'initial_sha256': initial_sha, 'checkpoint_sha256': final_sha, 'prediction_sha256': prediction_sha,
                  'training_array_sha256': self.evidence['train']['array_sha256'],
                  'row_order_sha256': self.evidence['train']['metadata_sha256'], 'metrics': measured,
                  'training_seconds': training_seconds, 'fit_seconds': fit_seconds, 'diagnostic_seconds': diagnostic_seconds,
                  'deployment_setup_seconds': deployment_setup_seconds, 'prediction_seconds': prediction_seconds,
                  'parity_rows': parity_rows, 'maximum_parity_error': max_error, 'parity_passed': True,
                  'storage': frozen.storage_bytes()}
        self.emit('fits.jsonl', record)
        self.fits.append(record)
        self.receipt['completed_fits'] += 1
        self.check()

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists() and not any(p.is_symlink() for p in self.out.parents), 'exclusive output')
        self.out.mkdir(parents=True, exist_ok=False)
        try:
            self.bind()
            self.prepare()
            for seed in self.plan['configuration']['seeds']:
                for kind in self.plan['configuration']['kinds']:
                    self.fit(kind, seed)
            self.receipt.update(parity_passed=True, initial_pairing_passed=True)
            result = aggregate(self.fits)
            result.update(preparation_seconds=self.preparation_seconds, calls=self.calls,
                          training_updates=self.calls['optimizer_update']['returned'])
            write(self.out/'summary.json', result)
            require(authenticate(self.args) == self.plan, 'end source/runtime/input authentication')
            require(not self.pending and set(self.calls) == set(self.plan['expected_calls'])
                    and all(self.calls[k]['attempted'] == self.calls[k]['returned'] == n
                            for k, n in self.plan['expected_calls'].items()), 'exact completed operation allocation')
            require({p.name for p in self.out.iterdir()} == payload_names(), 'exact closed output membership')
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status='completed', calls=self.calls, pending=self.pending, started_ns=self.start,
                                finished_ns=finished, clock_backend=self.clock.backend, wall_seconds=(finished-self.start)/1e9,
                                files={p.name: descriptor(p) for p in self.out.iterdir()})
            write(self.out/'receipt.json', self.receipt)
            self.check()
        except BaseException as error:
            self.receipt.update(status='failed', calls=self.calls, pending=self.pending, error=repr(error), traceback=traceback.format_exc())
            try:
                if (self.out/'receipt.json').exists():
                    (self.out/'receipt.json').rename(self.out/'invalid-completed-receipt.json')
                write(self.out/'failed.json', self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - Preserve original failure evidence.
                error.add_note(f'Failure publication: {secondary!r}')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'output', 'supervision'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    Run(parser.parse_args()).execute()
