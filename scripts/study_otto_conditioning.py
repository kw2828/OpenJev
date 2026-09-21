"""Matched input-conditioning screen with unchanged cached targets and model class."""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = 'scripts/study_otto_capacity.py'
BASE_PIN = '5043e550d17346b759bc8e1b3507a79721137b34c71ebfee79cccc807f7da7de'
VERSION = 'otto-conditioning-v1'
KINDS, SEEDS = ('gain1', 'gain53'), (10101, 10102, 10103)
NEW = ('scripts/study_otto_conditioning.py', 'scripts/freeze_otto_conditioning.py',
       'scripts/audit_otto_conditioning.py', 'src/openjev/research/otto_conditioned_value.py',
       'tests/test_otto_conditioned_value.py', 'tests/test_otto_conditioning.py',
       'tests/test_audit_otto_conditioning.py', 'research/otto-conditioning-protocol.md')


def load_base():
    import hashlib
    if hashlib.sha256((ROOT/BASE).read_bytes()).hexdigest() != BASE_PIN:
        raise ValueError('unchanged capacity mechanics')
    spec = importlib.util.spec_from_file_location('_conditioning_capacity', ROOT/BASE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


C = load_base()
P = C.P
require, sha, load = C.require, C.sha, C.load
read, write, regular, descriptor, closed = C.read, C.write, C.regular, C.descriptor, C.closed


def configuration(mode):
    cfg = C.configuration(mode)
    return {**cfg, 'kinds': list(KINDS), 'spatial_gains': {'gain1': 1, 'gain53': 53},
            'initialization': 'capacity mlp8 seed draws; first spatial columns divided by gain in float32',
            'baseline': 'c0 times raw spatial mass; context unchanged',
            'initial_pair': 'all selected TRAIN rows, zero input, first TRAIN input times 0.5; batches256',
            'initial_pair_tolerance': {'absolute': 1e-6, 'relative': 1e-6},
            'qualification_projection_limit_seconds': 400,
            'policy_stage': 'separately frozen autonomous comparison after technical completion regardless of scalar gate'}


def limits(mode):
    require(mode in ('qualify', 'study'), 'known mode')
    return {'native_seconds': 120 if mode == 'qualify' else 600,
            'rss_bytes': (4 if mode == 'qualify' else 8)*1024**3,
            'output_bytes': (128 if mode == 'qualify' else 512)*1024**2,
            'optimizer_updates': 24 if mode == 'qualify' else 21120,
            'native_steps': 0, 'native_resets': 0, 'external_model_calls': 0}


def payload_names(mode):
    names = {'started.json', 'runtime.json', 'preparation.json', 'work.jsonl', 'initializations.jsonl',
             'epoch-orders.jsonl', 'updates.jsonl', 'fit-curves.jsonl', 'fits.jsonl', 'parity.jsonl',
             'initial-pairing.jsonl', 'summary.json'}
    for seed in configuration(mode)['seeds']:
        for kind in KINDS:
            names.update(f'{prefix}-{kind}-{seed}.npz' for prefix in ('initial', 'final', 'predictions'))
    return names


def authenticate(args):
    require(args.plan.is_absolute() and not args.plan.is_symlink() and sha(args.plan) == args.plan_sha256, 'external plan pin')
    plan = read(args.plan)
    mode = plan['mode']
    require(plan['version'] == VERSION and plan['status'] == 'frozen_before_execution'
            and plan['configuration'] == configuration(mode) and plan['limits'] == limits(mode), 'fixed conditioning contract')
    require(plan['independent_audit_limits'] == {'seconds': 600, 'rss_bytes': 4*1024**3,
                                               'output_bytes': 128*1024**2}, 'independent audit allocation')
    require(set(NEW) <= plan['sources'].keys() and plan['sources'][BASE] == BASE_PIN, 'new source closure')
    for name, pin in plan['sources'].items():
        require(sha(regular(name)) == pin, 'unchanged source: '+name)
    roles = {'capacity_plan', 'capacity_receipt', 'capacity_terminal', 'capacity_audit', 'train_data', 'train_rows'}
    if mode == 'study':
        roles |= {'valid_data', 'valid_rows'}
    require(set(plan['inputs']) == roles, 'exact input roles; no collection or old EVAL')
    paths = {}
    for role, item in plan['inputs'].items():
        paths[role] = regular(item['path'])
        require(descriptor(paths[role]) == {k: item[k] for k in ('bytes', 'sha256')}, 'input hash and size')
    prior = C.authenticate(SimpleNamespace(plan=paths['capacity_plan'], plan_sha256=sha(paths['capacity_plan'])))
    worker, audit = closed(paths['capacity_receipt']), closed(paths['capacity_audit'])
    require(prior['mode'] == worker['mode'] == 'study' and worker['completed_fits'] == 9
            and worker['pending'] == [] and worker['parity_passed'] is True
            and worker['plan_sha256'] == sha(paths['capacity_plan']) and worker['sources'] == prior['sources']
            and worker['inputs'] == prior['inputs'] and set(worker['files']) == C.payload_names('study'), 'completed capacity evidence')
    C.terminal_identity(paths['capacity_receipt'], paths['capacity_plan'], paths['capacity_terminal'], BASE)
    require(audit['agreement'] is True and audit['worker_sha256'] == sha(paths['capacity_receipt'])
            and audit['plan_sha256'] == sha(paths['capacity_plan']) and audit['terminal_sha256'] == sha(paths['capacity_terminal'])
            and audit['source']['sha256'] == prior['sources']['scripts/audit_otto_capacity.py'], 'independent capacity audit')
    require(all(plan['sources'].get(k) == v for k, v in prior['sources'].items()), 'inherited immutable sources')
    for role in ('train_data', 'train_rows', 'valid_data', 'valid_rows') if mode == 'study' else ('train_data', 'train_rows'):
        require(plan['inputs'][role] == prior['inputs'][role], 'same authenticated cached dataset')
    for key in ('python_executable', 'python_version', 'all_distributions'):
        require(plan[key] == prior[key], 'same authenticated runtime')
    if mode == 'qualify':
        require(plan['qualification'] == {}, 'fresh disposable qualification')
    else:
        require(set(plan['qualification']) == {'plan', 'receipt', 'terminal'}, 'complete qualification roles')
        qpaths = {}
        for role, item in plan['qualification'].items():
            qpaths[role] = regular(item['path'])
            require(descriptor(qpaths[role]) == {k: item[k] for k in ('bytes', 'sha256')}, 'qualification pin')
        qp = authenticate(SimpleNamespace(plan=qpaths['plan'], plan_sha256=sha(qpaths['plan'])))
        qr = closed(qpaths['receipt'])
        require(qp['mode'] == qr['mode'] == 'qualify' and qr['completed_fits'] == 2
                and qr['parity_passed'] is True and qr['initial_pairing_passed'] is True
                and qr['pending'] == [] and qr['plan_sha256'] == sha(qpaths['plan'])
                and qr['sources'] == qp['sources'] and qr['inputs'] == qp['inputs']
                and set(qr['files']) == payload_names('qualify')
                and qr['calls']['optimizer_update']['returned'] == 24
                and all(v['attempted'] == v['returned'] for v in qr['calls'].values()), 'closed disposable qualification')
        C.terminal_identity(qpaths['receipt'], qpaths['plan'], qpaths['terminal'], 'scripts/study_otto_conditioning.py')
        require(all(plan['sources'].get(k) == v for k, v in qp['sources'].items()), 'qualified sources unchanged')
        require(read(qpaths['receipt'].parent/'summary.json')['qualification_projection_seconds'] <= 400,
                'fixed prospective cost admission')
    return plan


def gates(values, floor):
    checks = []
    for seed in SEEDS:
        candidate, control = values[f'gain53@{seed}'], values[f'gain1@{seed}']
        for key, value, bound in (
            ('train_excess', candidate['train']['mse_normalized']-floor, .8*(control['train']['mse_normalized']-floor)),
            ('valid_mse', candidate['valid']['mse_normalized'], .9*control['valid']['mse_normalized'])):
            checks.append({'name': f'gain53.{seed}.{key}', 'value': value, 'threshold': bound, 'passes': value <= bound})
    return checks


class Run(C.Run):
    def __init__(self, args):
        super().__init__(args)
        self.receipt.update(version=VERSION, initial_pairing_passed=False)

    def bind(self):
        require(sha(ROOT/P.CLOCK) == P.CLOCK_PIN, 'qualified clock')
        self.clock = load(ROOT/P.CLOCK, '_conditioning_clock').SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns()-self.start < 5*10**9, 'supervision missing')
            time.sleep(.01)
        self.launch, self.plan = read(self.args.supervision), authenticate(self.args)
        launch = self.launch
        command = list(launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        import os
        require(command == [sys.executable, *sys.argv] and launch['pid'] == os.getpid()
                and launch['pgid'] == os.getpgrp() and launch['parent_pid'] == os.getppid()
                and launch['cwd'] == str(ROOT) == str(Path.cwd()), 'actual supervised process')
        require(launch['clock_backend'] == self.clock.backend and launch['started_ns'] <= self.start < launch['deadline_ns']
                and launch['cap_seconds'] == self.plan['limits']['native_seconds']
                and launch['deadline_ns'] == launch['started_ns']+launch['cap_seconds']*10**9
                and launch['watchdog_sha256'] == self.plan['sources']['scripts/supervise_dialogue_observation_v2.py']
                and launch['clock_source_sha256'] == P.CLOCK_PIN, 'qualified bounded supervision')
        self.receipt.update(mode=self.plan['mode'], limits=self.plan['limits'], sources=self.plan['sources'],
                            inputs=self.plan['inputs'], plan_sha256=self.args.plan_sha256,
                            supervision_sha256=sha(self.args.supervision))
        write(self.out/'started.json', {'request': {k: str(v) for k, v in vars(self.args).items()},
                                      'launch': launch, 'started_ns': self.start})
        self.check()

    def prepare(self):
        tick = time.perf_counter()
        super().prepare()
        from openjev.research import otto_conditioned_value
        self.model = otto_conditioned_value
        prior = regular(self.plan['inputs']['capacity_receipt']['path']).parent/'preparation.json'
        require(self.c0 == read(prior)['c0_float32'], 'unchanged TRAIN-only baseline')
        self.preparation_seconds = time.perf_counter()-tick

    def save(self, name, arrays):
        path = self.out/name
        if path.exists():
            expected = {f'initial-{kind}-{seed}.npz' for seed in self.plan['configuration']['seeds'] for kind in KINDS}
            require(name in expected and path.is_file() and not path.is_symlink(), 'only prepublished initial checkpoints may be reused')
            with self.np.load(path, allow_pickle=False) as saved:
                require(set(saved.files) == set(arrays), 'paired initialization tensor membership')
                for key, value in arrays.items():
                    value = self.np.asarray(value)
                    require(saved[key].dtype == value.dtype and saved[key].shape == value.shape
                            and saved[key].tobytes() == value.tobytes(), 'identical initial checkpoint consumed by fit')
            return sha(path)
        return super().save(name, arrays)

    def initial_pairing(self):
        import hashlib
        np = self.np
        tick = time.perf_counter()
        raw = self.data['train'][0].astype(np.float64)
        probe = np.concatenate([raw, np.zeros((1, 11028)), raw[:1]*.5])
        for seed in self.plan['configuration']['seeds']:
            predictions, pins = {}, {}
            for kind in KINDS:
                self.context = {'seed': seed, 'kind': kind, 'phase': 'initial_pair'}
                model = self.call('initial_pair_initialization', lambda kind=kind, seed=seed: self.model.make_head(kind, seed, self.c0))
                arrays = self.call('initial_pair_export', lambda model=model: self.model.export_head(model))
                pins[kind] = self.save(f'initial-{kind}-{seed}.npz', arrays)
                head = self.model.FrozenValue(arrays)
                parts = []
                for offset in range(0, len(probe), 256):
                    self.context = {'seed': seed, 'kind': kind, 'phase': 'initial_pair', 'offset': offset}
                    x = probe[offset:offset+256]
                    parts.append(self.call('initial_pair_forward', lambda head=head, x=x: head.normalized(x)))
                predictions[kind] = np.concatenate(parts)
            error = np.abs(predictions['gain53']-predictions['gain1'])
            passed = bool(np.all(error <= 1e-6+1e-6*np.abs(predictions['gain1'])))
            self.emit('initial-pairing.jsonl', {'seed': seed, 'features_sha256': hashlib.sha256(probe.tobytes()).hexdigest(),
                      'rows': len(probe), 'initial_sha256': pins, 'predictions': {k: v.tolist() for k, v in predictions.items()},
                      'maximum_difference': float(error.max()), 'tolerance': {'absolute': 1e-6, 'relative': 1e-6}, 'passed': passed})
            require(passed, 'initial represented-function pairing')
        self.receipt['initial_pairing_passed'] = True
        self.pairing_seconds = time.perf_counter()-tick

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists() and not any(p.is_symlink() for p in self.out.parents), 'exclusive output')
        self.out.mkdir(parents=True, exist_ok=False)
        try:
            self.bind()
            self.prepare()
            self.initial_pairing()
            self.fits = []
            for seed in self.plan['configuration']['seeds']:
                for kind in KINDS:
                    self.fit(kind, seed)
            self.receipt['parity_passed'] = True
            values = {r['fit_id']: r['metrics'] for r in self.fits}
            checks = gates(values, self.floor['irreducible_mse_normalized']) if self.plan['mode'] == 'study' else []
            projection = 3*C.math.fsum(r['fit_seconds'] for r in self.fits)*(3520/12) if self.plan['mode'] == 'qualify' else None
            summary = {'version': VERSION, 'mode': self.plan['mode'], 'rows': {k: len(v[0]) for k, v in self.data.items()},
                       'alias_floor': self.floor, 'metrics': values, 'checks': checks,
                       'conditioning_screen_passed': all(c['passes'] for c in checks) if checks else None,
                       'initial_pairing_passed': True, 'qualification_projection_seconds': projection,
                       'preparation_seconds': self.preparation_seconds, 'initial_pairing_seconds': self.pairing_seconds,
                       'training_costs': {r['fit_id']: r['fit_seconds'] for r in self.fits},
                       'learned_architecture_advantage_established': False,
                       'scope': 'Fixed-data optimization control only; fresh autonomous evidence is separately required regardless of scalar gate.'}
            write(self.out/'summary.json', summary)
            require(authenticate(self.args) == self.plan, 'unchanged end identity')
            require(not self.pending and all(v['attempted'] == v['returned'] for v in self.calls.values())
                    and self.calls['optimizer_update']['returned'] == self.plan['limits']['optimizer_updates'], 'complete fixed work')
            require({p.name for p in self.out.iterdir()} == payload_names(self.plan['mode']), 'exact payload closure')
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
            except BaseException as secondary:  # noqa: BLE001 - Preserve primary failure.
                error.add_note(repr(secondary))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'output', 'supervision'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    Run(parser.parse_args()).execute()
