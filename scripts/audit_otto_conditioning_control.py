"""Independent saved qualification and autonomous conditioning-control audit.

Only authenticated saved checkpoints and public packets are replayed. There is
no training, simulator, producer policy or producer readout call. Original
posterior cache truth, Torch execution, native randomness and timing remain
source-bound execution evidence. Local NumPy checkpoint readouts are counted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import signal
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPERS = {
    'scripts/audit_otto_bellman_control.py': '37490345ad60d0c4a18228d65040d704179d77ab6e3586c77bf144cc96dee49d',
    'scripts/audit_otto_conditioning.py': '28259542e97940c3dd9482dae4089a878dd946ca22c8adf352ed70aff6a1e51f',
}


def independent_module(path, pin, name):
    if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != pin:
        raise ValueError('independent helper source pin: '+path)
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E, G = (independent_module(path, pin, f'_conditioning_control_independent_{i}')
        for i, (path, pin) in enumerate(HELPERS.items()))
B, require = E.B, E.require
VERSION = 'otto-conditioning-control-saved-audit-v1'
RUNNER = 'scripts/study_otto_conditioning_control.py'
SEEDS, FAMILIES = (10101, 10102, 10103), ('gain1', 'gain53')
ARMS = tuple(f'{family}@{seed}' for seed in SEEDS for family in FAMILIES)+('analytic_inbounds',)
REGIMES = {'lambda3': 3., 'lambda4': 4., 'lambda5': 5.}
FIRST = {'lambda3': 15100001, 'lambda4': 15200001, 'lambda5': 15300001}
HORIZON, NTRAIN, NVALID = 2188, 5589, 1109
TIMES, METRICS = E.TIMES, E.METRICS
LIMITS = {
    'qualify': {'native_seconds': 180, 'rss_bytes': 4*1024**3, 'output_bytes': 128*1024**2},
    'study': {'native_seconds': 1800, 'rss_bytes': 4*1024**3, 'output_bytes': 128*1024**2},
}
SCOPE = __doc__


def evaluation_order():
    for ri, (regime, first) in enumerate(FIRST.items()):
        for case in range(24):
            offset = (ri*24+case) % len(ARMS)
            for arm in ARMS[offset:]+ARMS[:offset]:
                yield regime, first+case, 1+case % 3, arm, case//3


def payload_names(mode):
    require(mode in LIMITS, 'declared audit mode')
    names = {'started.json', 'runtime.json', 'inference-setup.json', 'work-contexts.jsonl', 'work.jsonl', 'summary.json'}
    return names | ({'preparation.json', 'parity.jsonl'} if mode == 'qualify' else
                    {'native-setup.json', 'eval-transitions.jsonl', 'eval-episodes.jsonl', 'evaluation.jsonl'})


def aggregate(rows, mixtures):
    require([(r['regime'], r['seed'], r['initial_hit'], r['arm'], r['block']) for r in rows]
            == list(evaluation_order()), 'complete ordered 504-case trial')
    for row in rows:
        require(type(row['found']) is bool and type(row['steps']) is int and 1 <= row['steps'] <= HORIZON
                and (row['found'] or row['steps'] == HORIZON) and row['updates'] == row['steps']
                and row['blocked_steps'] == 0 and row['final_update_assimilated'] is True, 'complete episode outcome')
        for metric in METRICS:
            E.finite(float(row[metric]) if metric == 'found' else row[metric], 'finite recorded metric')
        require(abs(row['controller_seconds']-math.fsum(row[k] for k in TIMES)) <= 1e-9, 'complete controller cost')
    panels, competence, improvement, baseline_competence = {}, [], [], []

    def add(group, name, value, threshold, passed):
        group.append({'name': name, 'value': value, 'threshold': threshold, 'passes': bool(passed)})

    for regime in REGIMES:
        weights = {int(k): v for k, v in mixtures[regime].items()}
        require(set(weights) == {1, 2, 3} and all(type(w) in (int, float) and math.isfinite(w) and 0 < w < 1 for w in weights.values())
                and abs(math.fsum(weights.values())-1) <= 1e-12, 'positive normalized mixture')
        selected = [r for r in rows if r['regime'] == regime]

        def mean(subset, metric, weights=weights):
            return math.fsum(weights[h]*math.fsum(float(r[metric]) for r in subset if r['initial_hit'] == h)
                             / sum(r['initial_hit'] == h for r in subset) for h in (1, 2, 3))

        means = {a: {m: mean([r for r in selected if r['arm'] == a], m) for m in METRICS} for a in ARMS}
        blocks = [{a: mean([r for r in selected if r['arm'] == a and r['block'] == b], 'steps') for a in ARMS} for b in range(8)]
        families = {f: {m: math.fsum(means[f'{f}@{s}'][m] for s in SEEDS)/3 for m in METRICS} for f in FAMILIES}
        for family, group in (('gain53', competence), ('gain1', baseline_competence)):
            for seed in SEEDS:
                fit = means[f'{family}@{seed}']
                add(group, f'{regime}.{family}.{seed}.success', fit['found'], .95, fit['found'] >= .95)
                bound = 1.05*means['analytic_inbounds']['steps']
                add(group, f'{regime}.{family}.{seed}.moves', fit['steps'], bound, fit['steps'] <= bound)
        candidate, control = families['gain53'], families['gain1']
        wins = sum(math.fsum(b[f'gain1@{s}']-b[f'gain53@{s}'] for s in SEEDS)/3 > 0 for b in blocks)
        prefix = regime
        add(improvement, prefix+'.success', candidate['found'], control['found'], candidate['found'] >= control['found'])
        add(improvement, prefix+'.moves', candidate['steps'], .95*control['steps'], candidate['steps'] <= .95*control['steps'])
        add(improvement, prefix+'.positive_blocks', wins, 6, wins >= 6)
        add(improvement, prefix+'.cost', candidate['controller_seconds'], control['controller_seconds'],
            candidate['controller_seconds'] <= control['controller_seconds'])
        panels[regime] = {'weights': {str(k): v for k, v in weights.items()}, 'means': means,
                         'family_means': families, 'blocks': blocks,
                         'strata': {str(h): {a: {m: math.fsum(float(r[m]) for r in selected if r['arm'] == a and r['initial_hit'] == h)/8
                                               for m in METRICS} for a in ARMS} for h in (1, 2, 3)},
                         'raw_counts': {a: {'found': sum(r['found'] for r in selected if r['arm'] == a), 'episodes': 24} for a in ARMS}}
    return {'version': 'otto-conditioning-control-v1', 'episodes': 504, 'paired_cases': 72, 'regimes': panels, 'competence_checks': competence,
            'improvement_checks': improvement, 'control_competence_checks': baseline_competence,
            'pilot_continuation': all(r['passes'] for r in competence+improvement),
            'learned_architecture_advantage_established': False}


def amortization(result, setup, fit_seconds, preparation_seconds, module_seconds):
    E.finite(preparation_seconds, 'preparation cost')
    E.finite(module_seconds, 'module cost')
    scenarios = {}
    for regime, panel in result['regimes'].items():
        scenarios[regime] = {}
        for arm, metrics in panel['means'].items():
            if arm == 'analytic_inbounds':
                learn = preparation = deployment = 0.
                utility = metrics['controller_seconds']
            else:
                learn, preparation = fit_seconds[arm], preparation_seconds/6
                deployment = setup[arm]['seconds']+module_seconds/6
                utility = metrics['controller_seconds']-metrics['setup_allocation_seconds']
            for value in (learn, preparation, deployment, utility):
                E.finite(value, 'nonnegative amortization component')
            scenarios[regime][arm] = {'fit_seconds': learn, 'preparation_share_seconds': preparation,
                'deployment_setup_seconds': deployment, 'controller_without_deployment_seconds': utility,
                'seconds_per_search': {str(h): utility+(learn+preparation+deployment)/h for h in (1, 100, 10000)}}
    return scenarios


def parity_record(row, arm, split, index, values, costs, allowed, np, *, raw, weights,
                  checkpoint_sha256, posterior_sha256, position, sensing_length):
    require(set(row) == {'fit_id', 'split', 'row_index', 'values_numpy', 'values_torch', 'costs_numpy', 'costs_torch',
                        'allowed_actions', 'action_numpy', 'action_torch', 'passed', 'raw_masses', 'weights',
                        'checkpoint_sha256', 'posterior_sha256', 'position', 'sensing_length', 'eligible_mask'}
            and (row['fit_id'], row['split'], row['row_index']) == (arm, split, index)
            and row['allowed_actions'] == allowed and row['checkpoint_sha256'] == checkpoint_sha256
            and row['posterior_sha256'] == posterior_sha256, 'complete ordered parity identity')
    require(row['position'] == position and row['sensing_length'] == sensing_length
            and isinstance(row['eligible_mask'], list) and all(type(value) is bool for value in row['eligible_mask'])
            and row['eligible_mask'] == [a in allowed for a in range(4)], 'qualified public position sensing and mask')
    require(raw.shape == weights.shape == (4, 4) and raw.dtype == weights.dtype == np.float64
            and np.isfinite(raw).all() and (raw >= 0).all()
            and np.array_equal(weights, np.maximum(raw, 1e-10))
            and row['raw_masses'] == raw.tolist() and row['weights'] == weights.tolist(),
            'exact independent raw branch masses and floors')
    for suffix in ('numpy', 'torch'):
        saved_values, saved_costs = np.asarray(row['values_'+suffix], np.float64), np.asarray(row['costs_'+suffix], np.float64)
        require(saved_values.shape == values.shape == (16,) and saved_costs.shape == costs.shape == (4,)
                and np.isfinite(saved_values).all() and np.isfinite(saved_costs).all()
                and bool(np.all(np.abs(saved_values-values) <= 1e-10+1e-10*np.abs(values)))
                and bool(np.all(np.abs(saved_costs-costs) <= 1e-10+1e-10*np.abs(costs))), 'independent parity branch values and costs')
        reconstructed = E.costs(saved_values, weights, np)
        require(bool(np.all(np.abs(saved_costs-reconstructed) <= 1e-10+1e-10*np.abs(reconstructed))), 'saved physical branch reduction')
        require(type(row['action_'+suffix]) is int and row['action_'+suffix] == E.choice(row['costs_'+suffix], allowed, True, np)
                == E.choice(costs.tolist(), allowed, True, np), 'independent exact eligible parity action')
    for label in ('values', 'costs'):
        a, b = np.asarray(row[label+'_numpy'], np.float64), np.asarray(row[label+'_torch'], np.float64)
        require(bool(np.all(np.abs(a-b) <= 1e-10+1e-10*np.abs(b))), 'original NumPy/Torch tolerance predicate')
    require(row['passed'] is True, 'all original parity conditions pass')


class Audit(E.Audit):
    """Reuse independent public episode mechanics, never producer numerical code."""

    def __init__(self, args):
        super().__init__(args)
        require(args.mode in LIMITS, 'declared qualification or study audit')
        self.limits = LIMITS[args.mode]
        self.receipt.update(version=VERSION, mode=args.mode, scope=SCOPE, limits=self.limits)

    def check(self):
        require(self.clock.now_ns()-self.start < self.limits['native_seconds']*10**9, 'native audit deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= self.limits['rss_bytes'], 'audit RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= self.limits['output_bytes'], 'audit output cap')

    def authenticate(self):
        a, c = self.args, self.c
        for path in (a.plan, a.run, a.terminal, a.output):
            require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'absolute nonsymlink paths')
        for path, pin in ((a.plan, a.plan_sha256), (a.run/'receipt.json', a.receipt_sha256), (a.terminal, a.terminal_sha256)):
            c.equal(B.digest(path, self.check)['sha256'], pin, 'external identity before decoding')
        plan, worker, terminal = B.read(a.plan), B.read(a.run/'receipt.json'), B.read(a.terminal)
        require(plan['version'] == 'otto-conditioning-control-v1' and plan['mode'] == a.mode
                and plan['status'] == 'frozen_before_execution', 'prospective matching mode')
        c.equal(plan['independent_audit_limits'], self.limits, 'prospective audit allocation')
        for name, pin in plan['sources'].items():
            path = ROOT/name
            require(path.resolve().is_relative_to(ROOT), 'repository source closure')
            c.equal(B.digest(path, self.check)['sha256'], pin, 'unchanged source: '+name)
        for name in (RUNNER, 'scripts/audit_otto_conditioning_control.py', 'tests/test_audit_otto_conditioning_control.py'):
            c.equal(B.digest(ROOT/name, self.check)['sha256'], plan['sources'][name], 'prospectively bound new source')
        for name, pin in HELPERS.items():
            c.equal(plan['sources'][name], pin, 'same frozen independent arithmetic')
        require(worker['status'] == 'completed' and worker['version'] == 'otto-conditioning-control-v1'
                and worker['mode'] == a.mode and worker['pending'] == []
                and worker['external_model_calls'] == 0 and worker['completed_fits'] == 0,
                'completed no-training worker')
        for key in ('sources', 'inputs', 'limits'):
            c.equal(worker[key], plan[key], 'worker frozen '+key)
        c.equal(worker['plan_sha256'], a.plan_sha256, 'worker plan pin')
        names = payload_names(a.mode)
        c.equal(set(worker['files']), names, 'exact payload membership')
        c.equal({p.name for p in a.run.iterdir()}, names | {'receipt.json'}, 'closed successful directory')
        for name in names:
            c.equal(B.digest(a.run/name, self.check), worker['files'][name], 'closed payload before arrays')
        started = B.read(a.run/'started.json')
        request, launch = started['request'], started['launch']
        c.equal(request, {'plan': str(a.plan), 'plan_sha256': a.plan_sha256, 'output': str(a.run),
                         'supervision': request['supervision']}, 'original worker request')
        c.equal(B.digest(Path(request['supervision']), self.check)['sha256'], worker['supervision_sha256'], 'original launch file pin')
        c.equal(B.read(Path(request['supervision'])), launch, 'original launch contents')
        require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
                and terminal['group_absent'] is True and terminal['cleanup']['group_absent'] is True
                and terminal['cleanup']['reaped'] is True and terminal['cleanup']['errors'] == []
                and terminal['error'] is None and terminal['clock_error'] is None and terminal['timing_available'] is True,
                'successful parent before scientific decode')
        for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend',
                    'cap_seconds', 'clock_source_sha256', 'watchdog_sha256'):
            c.equal(terminal[key], launch[key], 'same closed parent')
        command = list(launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        require(len(command) == 10, 'exact original CLI fields')
        c.equal(command[:2], [plan['python_executable'], str(ROOT/RUNNER)], 'actual source and interpreter')
        c.equal(dict(zip(command[2::2], command[3::2], strict=True)),
                {f'--{k.replace("_", "-")}': v for k, v in request.items()}, 'original CLI bindings')
        cap = plan['limits']['native_seconds']
        require(launch['cap_seconds'] == cap and launch['pid'] == launch['pgid'] and launch['parent_pid'] != launch['pid']
                and launch['cwd'] == str(ROOT) and launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
                and worker['clock_backend'] == launch['clock_backend'] and launch['deadline_ns'] == launch['started_ns']+cap*10**9
                and launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns'],
                'native timing enclosure')
        c.equal(started['started_ns'], worker['started_ns'], 'worker clock origin')
        c.equal(worker['wall_seconds'], (worker['finished_ns']-worker['started_ns'])/1e9, 'worker elapsed')
        c.equal(terminal['elapsed_ns'], terminal['finished_ns']-terminal['started_ns'], 'parent elapsed')
        c.equal(terminal['wall_seconds'], terminal['elapsed_ns']/1e9, 'parent wall interval')
        c.equal(launch['clock_source_sha256'], B.CLOCK_PIN, 'qualified native clock')
        c.equal(launch['watchdog_sha256'], plan['sources']['scripts/supervise_dialogue_observation_v2.py'], 'qualified supervisor')
        require(type(worker['peak_rss_bytes']) is int and 0 < worker['peak_rss_bytes'] <= plan['limits']['rss_bytes']
                and sum(p.stat().st_size for p in a.run.iterdir()) <= plan['limits']['output_bytes'], 'recorded resource limits')
        # Sole producer use authenticates complete old inputs and runtime. No
        # producer preparation, prediction, branch or aggregate is imported here.
        producer = B.load(ROOT/RUNNER, '_conditioning_control_lineage_only')
        c.equal(producer.authenticate(argparse.Namespace(plan=a.plan, plan_sha256=a.plan_sha256)), plan,
                'closed prior scalar/conditioning inputs, qualified mode and runtime')
        self.receipt.update(plan_sha256=a.plan_sha256, worker_sha256=a.receipt_sha256,
                            terminal_sha256=a.terminal_sha256, producer_source_sha256=plan['sources'][RUNNER])
        return plan, worker

    def path(self, plan, role):
        item = plan['inputs'][role]
        path = Path(item['path'])
        path = path if path.is_absolute() else ROOT/path
        self.c.equal(B.digest(path, self.check), {k: item[k] for k in ('bytes', 'sha256')}, 'explicit bound input '+role)
        return path

    def predicted(self, x, arm):
        self.check()
        self.receipt['saved_checkpoint_readout_calls'] += 1
        result = G.conditioned_predict(x, self.heads[arm], self.baseline, arm.split('@')[0], self.np)
        self.receipt['saved_network_rows'] += len(x)
        self.check()
        return result

    def numerical_inputs(self, plan):
        import numpy as np
        self.np, self.maximum_prediction_error = np, 0.
        self.kernels, self.mixtures, self.heads, self.datasets = {}, {}, {}, {}
        self.conditioning_run = self.path(plan, 'conditioning_receipt').parent
        self.training_reference = B.read(self.conditioning_run/'summary.json')
        self.baseline = B.read(self.conditioning_run/'preparation.json')['c0_float32']
        require(type(self.baseline) is float and math.isfinite(self.baseline)
                and float(np.float32(self.baseline)) == self.baseline, 'same authenticated float32 TRAIN baseline')
        for regime in REGIMES:
            archive = self.load_arrays(self.path(plan, 'kernel_'+regime))
            self.c.equal(set(archive), {'likelihood', 'initial_hit_weights'}, 'exact kernel payload')
            kernel, weight = archive['likelihood'], archive['initial_hit_weights']
            require(kernel.shape == (4, 107, 107) and kernel.dtype == np.float64 and np.isfinite(kernel).all()
                    and (kernel >= 0).all() and (kernel <= 1).all() and not kernel[:, 53, 53].any(), 'unchanged kernel geometry')
            require(weight.shape == (4,) and weight.dtype == np.float64 and weight[0] == 0
                    and np.isfinite(weight).all() and (weight[1:] > 0).all()
                    and abs(float(weight.sum())-1) <= 1e-12, 'unchanged positive initial mixture')
            self.kernels[regime], self.mixtures[regime] = kernel, {h: float(weight[h]) for h in (1, 2, 3)}
        for arm in ARMS[:-1]:
            kind, seed = arm.split('@')
            archive = self.load_arrays(self.path(plan, f'head_{kind}_{seed}'))
            self.heads[arm] = G.checkpoint(archive, kind, self.baseline, np)
        if self.args.mode == 'qualify':
            for split, n in (('train', NTRAIN), ('valid', NVALID)):
                # The old complete data audit already establishes public cache
                # truth. Here only the fixed first eight rows are replayed.
                archive = self.load_arrays(self.path(plan, split+'_data'))
                self.c.equal(set(archive), {'features', 'target', 'beliefs', 'positions', 'sensing_length'}, 'canonical scalar cache closure')
                for key, shape, dtype in (('features', (n, 11028), np.float32), ('target', (n,), np.float32),
                                         ('beliefs', (n, 53, 53), np.float64), ('positions', (n, 2), np.int64),
                                         ('sensing_length', (n,), np.float64)):
                    require(archive[key].shape == shape and archive[key].dtype == dtype, 'authenticated cache shape/dtype')
                metadata = iter(B.lines(self.path(plan, split+'_rows'), self.check))
                selected = {key: value[:8].copy() for key, value in archive.items()}
                selected['rows'] = [next(metadata) for _ in range(8)]
                for index, row in enumerate(selected['rows']):
                    require(row['row_index'] == index and row['stage'] == split, 'same fixed first eight rows')
                    probability, position = selected['beliefs'][index], selected['positions'][index].tolist()
                    require(np.isfinite(probability).all() and (probability >= 0).all(), 'finite nonnegative qualified input')
                    self.c.equal(position, row['public']['position'], 'same public cached position')
                    rebuilt = E.A.state_features(probability, position, float(selected['sensing_length'][index]), np)
                    require(rebuilt.tobytes() == selected['features'][index].tobytes(), 'independent raw cached feature bytes')
                    self.c.equal(E.A.A.posterior_record(probability), row['posterior'], 'cached public posterior fingerprint')
                self.datasets[split] = selected
                del archive

    def begin_work(self):
        contexts = list(B.lines(self.args.run/'work-contexts.jsonl', self.check))
        self.c.equal([r['id'] for r in contexts], list(range(len(contexts))), 'contiguous operation contexts')
        require(all(set(r) == {'id', 'context'} and isinstance(r['context'], dict) and 'step' not in r['context'] for r in contexts)
                and len({json.dumps(r['context'], sort_keys=True) for r in contexts}) == len(contexts), 'unique causal contexts')
        self.context_count = len(contexts)
        self.work = E.Work(iter(B.lines(self.args.run/'work.jsonl', self.check)), [r['context'] for r in contexts], self.c)

    def finish_work(self, worker, expected):
        B.exhausted(self.work.iterator, 'complete journal')
        self.c.equal(self.work.stack, [], 'no pending operation')
        self.c.equal(self.work.used_contexts, set(range(self.context_count)), 'no unused context')
        self.c.tree(worker['calls'], self.work.counts, 'all attempts returns and operation durations')
        self.c.equal({k: v['attempted'] for k, v in self.work.counts.items()}, expected, 'exact attempted call inventory')
        self.c.equal({k: v['returned'] for k, v in self.work.counts.items()}, expected, 'exact returned call inventory')
        return self.work.counts

    def inference_setup(self, plan):
        setup = B.read(self.args.run/'inference-setup.json')
        self.c.equal(set(setup), set(ARMS[:-1]), 'all six separately loaded heads')
        for arm in ARMS[:-1]:
            record = setup[arm]
            path = self.path(plan, 'head_'+arm.replace('@', '_'))
            call = self.work.call('model_load', {'phase': 'inference_setup', 'arm': arm})
            self.c.equal((record['checkpoint'], record['sha256'], record['allocated_episodes']),
                         (str(path.relative_to(ROOT)), B.digest(path)['sha256'], 72 if self.args.mode == 'study' else 0),
                         'unchanged final checkpoint and allocation denominator')
            for key in ('seconds', 'instrumented_seconds', 'excluded_io_seconds'):
                E.finite(record[key], 'finite measured head-load duration')
            self.c.close(record['seconds'], record['instrumented_seconds']-record['excluded_io_seconds'], 'net complete head setup')
            require(record['seconds']+1e-9 >= call['seconds'], 'outer setup encloses counted restore/validation')
            self.c.equal({k: v for k, v in record['storage'].items() if k != 'scope'}, E.A.head_storage('mlp8'), 'complete immutable float64 head bytes')
        return setup

    def qualification(self, plan, worker, runtime):
        np, c = self.np, self.c
        require(worker['completed_episodes'] == 0 and worker['parity_passed'] is True, 'completed branch-only qualification')
        setup = self.inference_setup(plan)
        prep = B.read(self.args.run/'preparation.json')
        seconds = E.finite(prep['seconds'], 'paid qualification preparation')
        c.equal(prep, {'seconds': seconds,
                      'states': [{'split': split, 'row_index': i} for split in ('train', 'valid') for i in range(8)],
                      'scope': 'First eight TRAIN and VALID states; no targets, fitting or policy evaluation'}, 'fixed mechanical states')
        iterator = iter(B.lines(self.args.run/'parity.jsonl', self.check))
        for arm in ARMS[:-1]:
            self.work.call('parity_restore', {'phase': 'parity_restore', 'fit_id': arm})
            checkpoint_pin = plan['inputs']['head_'+arm.replace('@', '_')]['sha256']
            for split in ('train', 'valid'):
                data = self.datasets[split]
                for index in range(8):
                    context = {'phase': 'parity', 'fit_id': arm, 'split': split, 'step': index}
                    self.work.call('parity_numpy_forward', context)
                    self.work.call('parity_torch_forward', context)
                    metadata = data['rows'][index]
                    probability = data['beliefs'][index]
                    features, raw, weights = E.branches(probability, data['positions'][index].tolist(),
                        self.kernels[metadata['regime']], float(data['sensing_length'][index]), np)
                    values = 64*self.predicted(features, arm)
                    costs = E.costs(values, weights, np)
                    row = next(iterator)
                    parity_record(row, arm, split, index, values, costs, metadata['public']['valid_actions'], np,
                        raw=raw, weights=weights, checkpoint_sha256=checkpoint_pin,
                        posterior_sha256=hashlib.sha256(probability.tobytes()).hexdigest(),
                        position=data['positions'][index].tolist(), sensing_length=float(data['sensing_length'][index]))
                    self.arrays_close(np.asarray(row['values_numpy'], np.float64), values, 'qualification saved NumPy values')
                    self.arrays_close(np.asarray(row['values_torch'], np.float64), values, 'qualification saved Torch values')
                    self.arrays_close(np.asarray(row['costs_numpy'], np.float64), costs, 'qualification saved NumPy costs')
                    self.arrays_close(np.asarray(row['costs_torch'], np.float64), costs, 'qualification saved Torch costs')
                    c.count += 8
        B.exhausted(iterator, 'all96 ordered parity records')
        expected = {'model_load': 6, 'parity_restore': 6, 'parity_numpy_forward': 96, 'parity_torch_forward': 96}
        calls = self.finish_work(worker, expected)
        c.equal((self.receipt['saved_checkpoint_readout_calls'], self.receipt['saved_network_rows']), (96, 1536), 'all independent qualification rows')
        summary = {'version': 'otto-conditioning-control-v1', 'mode': 'qualify', 'parity_records': 96, 'parity_passed': True,
                   'completed_episodes': 0, 'preparation_seconds': seconds, 'learned_architecture_advantage_established': False}
        c.tree(B.read(self.args.run/'summary.json'), summary, 'complete original mechanical result')
        disjoint = (seconds+math.fsum(r['seconds'] for r in setup.values())+runtime['shared_setup_seconds']
                    +runtime['model_module_setup_seconds']+math.fsum(calls[k]['seconds'] for k in expected if k != 'model_load'))
        require(disjoint <= worker['wall_seconds']+1e-8, 'disjoint qualification work within worker interval')
        require(math.fsum(r['excluded_io_seconds'] for r in setup.values()) <= worker['artifact_io_seconds']+1e-8,
                'only measured setup I/O excluded')
        return {**summary, 'audit_version': VERSION, 'agreement': True, 'comparisons': c.count,
                'maximum_prediction_difference': self.maximum_prediction_error, 'actual_work': expected,
                'saved_checkpoint_readout_calls': 96, 'saved_network_rows': 1536,
                'costs': {'runtime': runtime, 'inference_setup': setup, 'disjoint_accounted_seconds': disjoint,
                          'worker_seconds': worker['wall_seconds']}, 'scope': SCOPE}

    def study(self, plan, worker, runtime):
        c = self.c
        require(worker['completed_episodes'] == 504 and worker['parity_passed'] is True, 'complete autonomous cohort')
        native = B.read(self.args.run/'native-setup.json')
        c.equal(native['native_resets'], 3, 'exact template resets')
        require(len(native['checks']) == 3, 'all three native kernel checks')
        for i, (regime, row) in enumerate(zip(REGIMES, native['checks'], strict=True)):
            self.work.call('native_reset', {'phase': 'native_setup', 'regime': regime})
            c.equal((row['regime'], row['template_seed'], row['cached_kernel_exact']), (regime, 15500001+i, True), 'same kernel/template witness')
            hit = row['initial_public']['hit']
            require(type(hit) is int and 1 <= hit <= 3, 'positive template initial hit')
            c.equal(row['initial_public'], B.packet([26, 26], hit, False, 0), 'public template packet')
        setup = self.inference_setup(plan)
        episodes = iter(B.lines(self.args.run/'eval-episodes.jsonl', self.check))
        allocated = iter(B.lines(self.args.run/'evaluation.jsonl', self.check))
        events = iter(B.lines(self.args.run/'eval-transitions.jsonl', self.check))
        self.sources, self.uniforms, rows = {}, {}, []
        for identity in evaluation_order():
            row = self.episode(next(episodes), identity, events)
            arm = row['arm']
            allocation = setup[arm]['seconds']/72+runtime['model_module_setup_seconds']/432 if arm in setup else 0.
            result = {**row, 'setup_allocation_seconds': allocation, 'controller_seconds': row['controller_seconds']+allocation}
            c.equal(next(allocated), result, 'complete allocated controller cost')
            rows.append({k: v for k, v in result.items() if k != 'draws_evaluation_only'})
        for stream, label in ((episodes, 'episodes'), (allocated, 'allocated episodes'), (events, 'all public transitions')):
            B.exhausted(stream, 'complete '+label)
        expected = {'model_load': 6, 'native_reset': 507, 'native_step': sum(r['steps'] for r in rows),
                    'analytic_choose': sum(r['steps'] for r in rows if r['arm'] == 'analytic_inbounds'),
                    'value_forward': sum(r['steps'] for r in rows if r['arm'] != 'analytic_inbounds')}
        calls = self.finish_work(worker, expected)
        require(expected['native_step'] <= plan['limits']['native_steps'] and expected['native_reset'] <= plan['limits']['native_resets']
                and expected['value_forward'] <= plan['limits']['value_forward'], 'prospective native and learned operation caps')
        c.equal((self.receipt['saved_checkpoint_readout_calls'], self.receipt['saved_network_rows']),
                (expected['value_forward'], 16*expected['value_forward']), 'one independent16-row readout per learned decision')
        result = aggregate(rows, self.mixtures)
        saved = B.read(self.args.run/'summary.json')
        for key, value in result.items():
            c.tree(saved[key], value, 'independent summary/'+key)
        c.equal(worker['pilot_continuation'], result['pilot_continuation'], 'all30 unchanged scientific decisions')
        c.tree(saved['calls'], calls, 'summary actual work')
        c.equal(saved['inference_setup'], setup, 'summary paid head setup')
        for key in ('shared_setup_seconds', 'model_module_setup_seconds'):
            c.equal(saved[key], runtime[key], 'summary setup scope')
        c.equal((saved['paired_source_cases'], saved['paired_uniforms']), (72, len(self.uniforms)), 'all paired source and random channels')
        scenarios = amortization(result, setup, self.training_reference['training_costs'],
                                self.training_reference['preparation_seconds'], runtime['model_module_setup_seconds'])
        c.tree(saved['amortization']['scenarios'], scenarios, 'H1/H100/H10000 independent accounting')
        training_reference = {k: self.training_reference[k] for k in ('training_costs', 'preparation_seconds', 'initial_pairing_seconds')}
        training_reference['worker_seconds'] = B.read(self.conditioning_run/'receipt.json')['wall_seconds']
        c.equal(saved['amortization']['training_reference'], training_reference, 'already-paid fitting intervals unchanged')
        disjoint = (math.fsum(r['controller_seconds']+r['environment_seconds'] for r in rows)
                    +runtime['shared_setup_seconds']+calls['native_reset']['seconds'])
        require(disjoint <= worker['wall_seconds']+1e-8, 'complete nonoverlapping runtime accounting')
        excluded = math.fsum(r['choose_excluded_io_seconds'] for r in rows)+math.fsum(r['excluded_io_seconds'] for r in setup.values())
        require(excluded <= worker['artifact_io_seconds']+1e-8, 'only measured artifact I/O excluded')
        return {**result, 'audit_version': VERSION, 'agreement': True, 'amortization': scenarios,
                'comparisons': c.count, 'maximum_scalar_difference': c.maximum_difference,
                'maximum_prediction_difference': self.maximum_prediction_error, 'actual_work': expected,
                'saved_checkpoint_readout_calls': self.receipt['saved_checkpoint_readout_calls'],
                'saved_network_rows': self.receipt['saved_network_rows'],
                'condition_counts': {'competence': 18, 'improvement': 12, 'control_competence_descriptive': 18},
                'costs': {'training_reference': training_reference, 'runtime': runtime, 'inference_setup': setup,
                          'disjoint_accounted_seconds': disjoint, 'worker_seconds': worker['wall_seconds']}, 'scope': SCOPE}

    def compute(self, plan, worker):
        self.numerical_inputs(plan)
        runtime = B.read(self.args.run/'runtime.json')
        self.c.equal((runtime['python'], runtime['executable']), (sys.version, plan['python_executable']), 'recorded numerical interpreter')
        self.c.equal((runtime['torch_threads'], runtime['torch_interop_threads'], runtime['module_allocation_episodes']),
                     (1, 1, 0 if self.args.mode == 'qualify' else 432), 'CPU1 and module allocation')
        self.c.equal(runtime['environment'], {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}, 'single-thread numerical settings')
        for key in ('shared_setup_seconds', 'model_module_setup_seconds'):
            E.finite(runtime[key], 'finite runtime setup')
        self.begin_work()
        return self.qualification(plan, worker, runtime) if self.args.mode == 'qualify' else self.study(plan, worker, runtime)

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists() and not any(p.is_symlink() for p in self.out.parents),
                'exclusive audit output')
        self.out.mkdir(parents=True, exist_ok=False)
        old_alarm = signal.getsignal(signal.SIGALRM)
        try:
            require(B.digest(ROOT/B.CLOCK)['sha256'] == B.CLOCK_PIN, 'qualified suspend-inclusive clock')
            self.clock = B.load(ROOT/B.CLOCK, '_conditioning_control_audit_clock').SuspendClock()
            self.start = self.clock.now_ns()
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('audit wall alarm')))
            signal.setitimer(signal.ITIMER_REAL, self.limits['native_seconds'])
            self.receipt.update(source=B.digest(Path(__file__), self.check), independent_helpers=HELPERS)
            B.write(self.out/'started.json', {'request': {k: str(v) for k, v in vars(self.args).items()}, 'limits': self.limits,
                                            'clock_backend': self.clock.backend, 'started_ns': self.start})
            plan, worker = self.authenticate()
            summary = self.compute(plan, worker)
            self.authenticate()
            B.write(self.out/'summary.json', summary)
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status='completed', agreement=True, clock_backend=self.clock.backend, started_ns=self.start,
                                finished_ns=finished, wall_seconds=(finished-self.start)/1e9, comparisons=self.c.count,
                                files={p.name: B.digest(p, self.check) for p in self.out.iterdir()})
            B.write(self.out/'receipt.json', self.receipt)
            self.check()
            return self.receipt
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            self.receipt.update(status='failed', agreement=False, error=repr(error), traceback=traceback.format_exc())
            try:
                if (self.out/'receipt.json').exists():
                    (self.out/'receipt.json').rename(self.out/'invalid-completed-receipt.json')
                B.write(self.out/'failed.json', self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - Preserve original audit failure.
                error.add_note(f'Failure publication: {secondary!r}')
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=tuple(LIMITS), required=True)
    for flag in ('plan', 'run', 'terminal', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    for flag in ('plan-sha256', 'receipt-sha256', 'terminal-sha256'):
        parser.add_argument('--'+flag, required=True)
    Audit(parser.parse_args()).execute()
