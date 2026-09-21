"""Independent saved-only audit of paired MC/Bellman continuation.

No optimizer, environment or producer numerical functions are executed. Pinned
independent filtering/readout helpers are reused; lineage authentication alone
is delegated to the producer. Every target refresh and deployed learned choice
is replayed. Optimizer trajectories, Torch execution, RNG and timing truth remain
source-bound evidence, not independently repeated training or simulation.
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
HELPER = ROOT / 'scripts/audit_otto_return_value.py'
HELPER_PIN = 'ad3d8e92c0bae3b9dd573af40e1cbb49d1dc771ff23f30500667a46f94980de8'
TARGET = 'scripts/audit_otto_bellman_targets.py'
TARGET_PIN = '1bf5409b5c4ecddd70bb58db54fc3aaedd7cea494b0794555169e96a859cfe2b'
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_PIN:
    raise ValueError('pinned independent scalar auditor')
_spec = importlib.util.spec_from_file_location('_bellman_control_independent', HELPER)
A = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = A
_spec.loader.exec_module(A)
B, require = A.B, A.require
features, branches, costs, choice = A.features, A.branches, A.costs, A.choice
VERSION = 'otto-bellman-control-saved-audit-v1'
RUNNER = 'scripts/study_otto_bellman_control.py'
LIMITS = {'native_seconds': 5400, 'rss_bytes': 4*1024**3, 'output_bytes': 256*1024**2}
SEEDS = (10101, 10102, 10103)
FAMILIES = ('mc', 'backup', 'reference')
ARMS = tuple(f'{f}@{s}' for s in SEEDS for f in FAMILIES) + ('analytic_inbounds',)
REGIMES = {'lambda3': 3., 'lambda4': 4., 'lambda5': 5.}
FIRST = {'lambda3': 12100001, 'lambda4': 12200001, 'lambda5': 12300001}
HORIZON, DIMENSION, NTRAIN, NVALID = 2188, 11028, 5589, 1109
TIMES, METRICS = A.TIMES, A.METRICS
SCOPE = __doc__


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, label)
    return value


def evaluation_order():
    for ri, (regime, first) in enumerate(FIRST.items()):
        for case in range(48):
            offset = (ri*48+case) % 10
            for arm in ARMS[offset:]+ARMS[:offset]:
                yield regime, first+case, 1+(case % 6)//2, arm, case//6


def payload_names():
    names = {'started.json', 'runtime.json', 'preparation.json', 'work-contexts.jsonl', 'work.jsonl',
             'initializations.jsonl', 'updates.jsonl', 'epoch-orders.jsonl', 'fit-curves.jsonl', 'fits.jsonl',
             'target-checkpoints.jsonl', 'target-refreshes.jsonl', 'parity.jsonl', 'target-audit.json',
             'training-costs.json', 'prediction-diagnostics.jsonl', 'native-setup.json', 'inference-setup.json',
             'eval-transitions.jsonl', 'eval-episodes.jsonl', 'evaluation.jsonl', 'summary.json'}
    for seed in SEEDS:
        names.add(f'mc-targets-{seed}.npz')
        names.update(f'{p}-{m}-{seed}.npz' for p in ('initial', 'final', 'predictions') for m in ('mc', 'backup'))
        names.update(f'{p}-backup-{seed}-{i:02d}.npz' for p in ('target-checkpoint', 'targets') for i in range(8))
    return names


def content_identity(archive):
    record = {'version': archive['version'].item(), 'kind': archive['kind'].item(),
              'input_dim': archive['input_dim'].item(),
              'arrays': {k: {'dtype': str(archive[k].dtype), 'shape': list(archive[k].shape),
                             'sha256': hashlib.sha256(archive[k].tobytes(order='C')).hexdigest()}
                         for k in ('c0', 'first_weight', 'final_weight', 'hidden_bias', 'output_bias')}}
    encoded = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**record, 'content_sha256': hashlib.sha256(encoded).hexdigest()}


def target_statistics(values, cast, np):
    return {'rows': len(values), 'minimum': float(values.min()), 'maximum': float(values.max()),
            'negative_count': int(np.count_nonzero(values < 0)), 'mean': float(np.mean(values, dtype=np.float64)),
            'maximum_float32_cast_error': float(np.max(np.abs(cast.astype(np.float64)-values)))}


class Work(A.Work):
    """Stream the journal, including one refresh enclosing 5,589 readouts."""
    def begin(self, channel, context):
        row = next(self.iterator)
        self.sequence += 1
        self.c.equal(row[:3], [self.sequence, 0, channel], 'ordered durable attempt')
        require(len(row) == 5 and type(row[3]) is int and 0 <= row[3] < len(self.contexts), 'work context index')
        self.used_contexts.add(row[3])
        actual = dict(self.contexts[row[3]])
        if row[4] is not None:
            actual['step'] = row[4]
        self.c.equal(actual, context, 'causal work context')
        self.counts.setdefault(channel, {'attempted': 0, 'returned': 0, 'seconds': 0.})['attempted'] += 1
        if not hasattr(self, 'stack'):
            self.stack = []
        self.stack.append((self.sequence, channel))
        return self.sequence

    def end(self, channel, call_id):
        self.c.equal(self.stack.pop(), (call_id, channel), 'LIFO operation closure')
        row = next(self.iterator)
        self.c.equal(row[:3], [call_id, 1, channel], 'matching durable return')
        require(len(row) == 6, 'work return schema')
        for value in row[3:]:
            finite(value, 'nonnegative operation duration')
        seconds, raw, excluded = row[3:]
        self.c.close(seconds, raw-excluded, 'net operation duration')
        self.counts[channel]['returned'] += 1
        self.counts[channel]['seconds'] += seconds
        return {'seconds': seconds, 'instrumented_seconds': raw, 'excluded_io_seconds': excluded}

    def call(self, channel, context):
        return self.end(channel, self.begin(channel, context))


def aggregate(rows, mixtures):
    require([(r['regime'], r['seed'], r['initial_hit'], r['arm'], r['block']) for r in rows]
            == list(evaluation_order()), 'complete ordered 1440-case trial')
    for row in rows:
        require(type(row['found']) is bool and type(row['steps']) is int and 1 <= row['steps'] <= HORIZON
                and (row['found'] or row['steps'] == HORIZON) and row['updates'] == row['steps']
                and row['blocked_steps'] == 0 and row['final_update_assimilated'] is True, 'complete episode outcome')
        for metric in METRICS:
            finite(float(row[metric]) if metric == 'found' else row[metric], 'finite recorded metric')
        require(abs(row['controller_seconds']-math.fsum(row[k] for k in TIMES)) <= 1e-9, 'complete controller cost')
    panels, competence, improvement = {}, [], []
    def add(group, name, value, threshold, passed):
        group.append({'name': name, 'value': value, 'threshold': threshold, 'passes': bool(passed)})
    for regime in REGIMES:
        weights = {int(k): v for k, v in mixtures[regime].items()}
        require(set(weights) == {1, 2, 3} and all(math.isfinite(w) and 0 < w < 1 for w in weights.values())
                and abs(math.fsum(weights.values())-1) <= 1e-12, 'positive normalized mixture')
        selected = [r for r in rows if r['regime'] == regime]
        def mean(subset, metric, weights=weights):
            return math.fsum(weights[h]*math.fsum(float(r[metric]) for r in subset if r['initial_hit'] == h)
                             / sum(r['initial_hit'] == h for r in subset) for h in (1, 2, 3))
        means = {a: {m: mean([r for r in selected if r['arm'] == a], m) for m in METRICS} for a in ARMS}
        blocks = [{a: mean([r for r in selected if r['arm'] == a and r['block'] == b], 'steps') for a in ARMS} for b in range(8)]
        families = {f: {m: math.fsum(means[f'{f}@{s}'][m] for s in SEEDS)/3 for m in METRICS} for f in FAMILIES}
        for seed in SEEDS:
            fit = means[f'backup@{seed}']
            add(competence, f'{regime}.{seed}.success', fit['found'], .95, fit['found'] >= .95)
            bound = 1.05*means['analytic_inbounds']['steps']
            add(competence, f'{regime}.{seed}.moves', fit['steps'], bound, fit['steps'] <= bound)
        candidate = families['backup']
        for control in ('mc', 'reference'):
            ref = families[control]
            wins = sum(math.fsum(b[f'{control}@{s}']-b[f'backup@{s}'] for s in SEEDS)/3 > 0 for b in blocks)
            prefix = f'{regime}.{control}'
            add(improvement, prefix+'.success', candidate['found'], ref['found'], candidate['found'] >= ref['found'])
            add(improvement, prefix+'.moves', candidate['steps'], .95*ref['steps'], candidate['steps'] <= .95*ref['steps'])
            add(improvement, prefix+'.positive_blocks', wins, 6, wins >= 6)
            add(improvement, prefix+'.controller_cost', candidate['controller_seconds'], ref['controller_seconds'],
                candidate['controller_seconds'] <= ref['controller_seconds'])
        panels[regime] = {'weights': {str(k): v for k, v in weights.items()}, 'means': means,
                         'family_means': families, 'blocks': blocks,
                         'strata': {str(h): {a: {m: math.fsum(float(r[m]) for r in selected if r['arm'] == a and r['initial_hit'] == h)/16
                                                for m in METRICS} for a in ARMS} for h in (1, 2, 3)},
                         'raw_counts': {a: {'found': sum(r['found'] for r in selected if r['arm'] == a), 'episodes': 48} for a in ARMS}}
    return {'episodes': 1440, 'paired_cases': 144, 'regimes': panels, 'competence_checks': competence,
            'improvement_checks': improvement, 'pilot_continuation': all(r['passes'] for r in competence+improvement),
            'learned_architecture_advantage_established': False}


def amortization(result, setup, fit_seconds, preparation_seconds, module_seconds):
    finite(preparation_seconds, 'preparation cost')
    finite(module_seconds, 'module cost')
    scenarios = {}
    for regime, panel in result['regimes'].items():
        scenarios[regime] = {}
        for arm, metrics in panel['means'].items():
            if arm == 'analytic_inbounds':
                learn = preparation = deployment = 0.
                utility = metrics['controller_seconds']
            else:
                learn = fit_seconds.get(arm, 0.)
                preparation = preparation_seconds/6 if arm in fit_seconds else 0.
                deployment = setup[arm]['seconds']+module_seconds/9
                utility = metrics['controller_seconds']-metrics['setup_allocation_seconds']
            for value in (learn, preparation, deployment, utility):
                finite(value, 'nonnegative amortization component')
            scenarios[regime][arm] = {'continuation_seconds': learn, 'preparation_share_seconds': preparation,
                'deployment_setup_seconds': deployment, 'controller_without_deployment_seconds': utility,
                'seconds_per_search': {str(h): utility+(learn+preparation+deployment)/h for h in (1, 100, 10000)}}
    return scenarios


def analytic_scores(probability, position, kernel, np):
    """Public space-aware formula; blocked actions remain unavailable."""
    grid = np.indices((53, 53))
    result = [None]*4
    for action in range(4):
        target = B.move(position, action)
        if target == position:
            continue
        end_mass = probability[tuple(target)]
        if end_mass > 1-1e-10:
            result[action] = -1e-10
            continue
        surviving = probability.copy()
        surviving[tuple(target)] = 0
        if surviving.sum() > 1e-10:
            surviving /= surviving.sum()
        x, y = target
        joint = surviving*kernel[:, 53-x:106-x, 53-y:106-y]
        masses = joint.sum(axis=(1, 2))
        distance = abs(grid[0]-x)+abs(grid[1]-y)
        total = 0.
        for hit in range(4):
            posterior = joint[hit]/masses[hit] if masses[hit] > 1e-10 else joint[hit]
            log = np.zeros((53, 53))
            keep = posterior > 1e-10
            log[keep] = -np.log2(posterior[keep])
            entropy = np.sum(posterior*log)
            estimate = np.sum(posterior*distance)+2**(entropy-1)-.5
            value = np.log2(estimate) if estimate > 0 else estimate
            total += (1-end_mass)*masses[hit]*value
        result[action] = float(total)
    require(all(v is None or math.isfinite(v) for v in result), 'finite independent analytic scores')
    return result


class Audit(A.Audit):
    def __init__(self, args):
        super().__init__(args)
        self.receipt.update(version=VERSION, scope=SCOPE, limits=LIMITS)

    def check(self):
        require(self.clock.now_ns()-self.start < LIMITS['native_seconds']*10**9, 'native audit deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= LIMITS['rss_bytes'], 'audit RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'audit output cap')

    def authenticate(self):
        a, c = self.args, self.c
        for path in (a.plan, a.run, a.terminal, a.output):
            require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'absolute nonsymlink paths')
        for path, pin in ((a.plan, a.plan_sha256), (a.run/'receipt.json', a.receipt_sha256), (a.terminal, a.terminal_sha256)):
            c.equal(B.digest(path, self.check)['sha256'], pin, 'external evidence identity before decode')
        plan, worker, terminal = B.read(a.plan), B.read(a.run/'receipt.json'), B.read(a.terminal)
        c.equal(plan['independent_audit_limits'], {'seconds': LIMITS['native_seconds'],
                'rss_bytes': LIMITS['rss_bytes'], 'output_bytes': LIMITS['output_bytes']}, 'prospective independent audit allocation')
        for name in (RUNNER, 'scripts/audit_otto_bellman_control.py', 'tests/test_audit_otto_bellman_control.py'):
            c.equal(B.digest(ROOT/name, self.check)['sha256'], plan['sources'][name], 'source identity before lineage import')
        reader = B.load(ROOT/RUNNER, '_return_value_lineage_only')
        c.equal(reader.authenticate(a), plan, 'qualified producer lineage only')
        require(worker['status'] == 'completed' and worker['version'] == 'otto-bellman-control-v1'
                and worker['completed_fits'] == 6 and worker['completed_episodes'] == 1440
                and worker['external_model_calls'] == 0 and worker['pending'] == [], 'complete run before numerical decode')
        for key in ('sources', 'inputs', 'limits'):
            c.equal(worker[key], plan[key], 'worker frozen plan join')
        c.equal(worker['plan_sha256'], a.plan_sha256, 'worker plan pin')
        c.equal(set(worker['files']), payload_names(), 'exact payload manifest')
        c.equal({p.name for p in a.run.iterdir()}, payload_names() | {'receipt.json'}, 'no missing late or extra payload')
        for name, expected in worker['files'].items():
            path = a.run/name
            require(path.is_file() and not path.is_symlink(), 'regular payload')
            c.equal(B.digest(path, self.check), expected, 'complete payload hashes before arrays')
        started = B.read(a.run/'started.json')
        request, launch = started['request'], started['launch']
        c.equal(request, {'plan': str(a.plan), 'plan_sha256': a.plan_sha256, 'supervision': request['supervision'], 'output': str(a.run)}, 'actual worker request')
        c.equal(B.digest(Path(request['supervision']), self.check)['sha256'], worker['supervision_sha256'], 'launch pin')
        c.equal(B.read(Path(request['supervision'])), launch, 'actual launch contents')
        require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
                and terminal['group_absent'] is True and terminal['cleanup']['group_absent'] is True
                and terminal['cleanup']['reaped'] is True and terminal['cleanup']['errors'] == []
                and terminal['error'] is None and terminal['clock_error'] is None and terminal['timing_available'] is True,
                'successful complete parent before scientific decode')
        for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend', 'cap_seconds', 'clock_source_sha256', 'watchdog_sha256'):
            c.equal(terminal[key], launch[key], 'parent launch identity')
        command = list(launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        c.equal(command[:2], [plan['python_executable'], str(ROOT/RUNNER)], 'actual interpreter/source')
        require(len(command) == 10, 'four CLI bindings')
        c.equal(dict(zip(command[2::2], command[3::2], strict=True)),
                {f'--{k.replace("_", "-")}': v for k, v in request.items()}, 'actual argument equality')
        cap = plan['limits']['native_seconds']
        require(launch['cap_seconds'] == cap and launch['pid'] == launch['pgid'] and launch['parent_pid'] != launch['pid']
                and launch['cwd'] == str(ROOT) and launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
                and worker['clock_backend'] == launch['clock_backend'] and launch['deadline_ns'] == launch['started_ns']+cap*10**9
                and launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns'],
                'strict native time enclosure')
        c.equal(started['started_ns'], worker['started_ns'], 'worker clock origin')
        c.equal(worker['wall_seconds'], (worker['finished_ns']-worker['started_ns'])/1e9, 'worker elapsed')
        c.equal(terminal['elapsed_ns'], terminal['finished_ns']-terminal['started_ns'], 'parent elapsed')
        c.equal(terminal['wall_seconds'], terminal['elapsed_ns']/1e9, 'parent seconds')
        c.equal(launch['clock_source_sha256'], B.CLOCK_PIN, 'qualified native clock')
        c.equal(launch['watchdog_sha256'], plan['sources']['scripts/supervise_dialogue_observation_v2.py'], 'qualified parent source')
        require(type(worker['peak_rss_bytes']) is int and 0 < worker['peak_rss_bytes'] <= plan['limits']['rss_bytes']
                and sum(p.stat().st_size for p in a.run.iterdir()) <= plan['limits']['output_bytes'], 'recorded resource bounds')
        self.receipt.update(plan_sha256=a.plan_sha256, worker_sha256=a.receipt_sha256, terminal_sha256=a.terminal_sha256,
                            producer_source_sha256=plan['sources'][RUNNER], authentication_reuse='Producer.authenticate lineage only.')
        require(plan['mode'] == 'study' and worker['mode'] == 'study' and worker['target_refreshes'] == 24
                and worker['target_audit_passed'] is True and worker['parity_passed'] is True
                and worker['target_audit_readouts'] == 134136, 'complete continuation study and prerequisites')
        self.c.equal(B.digest(ROOT/TARGET, self.check)['sha256'], TARGET_PIN, 'independent target replay source')
        self.c.equal(plan['sources'][TARGET], TARGET_PIN, 'same qualified independent target arithmetic')
        return plan, worker

    def episode(self, row, identity, events):
        np, c = self.np, self.c
        regime, seed, hit, arm, block = identity
        episode_id = f'eval:{regime}:{seed}:{arm}'
        canonical = {'episode_id': episode_id, 'stage': 'eval', 'regime': regime, 'seed': seed,
                     'initial_hit': hit, 'arm': arm, 'block': block}
        c.equal({k: row[k] for k in canonical}, canonical, 'all1440 evaluation identities')
        steps = row['steps']
        require(type(steps) is int and 1 <= steps <= HORIZON and type(row['found']) is bool
                and (row['found'] or steps == HORIZON), 'complete found/censored duration')
        context = {'phase': 'eval', 'episode': episode_id, 'step': 0}
        self.work.call('native_reset', context)
        public = B.packet([26, 26], hit, False, 0)
        source = row['source_evaluation_only']
        require(len(source) == 2 and all(type(v) is int and 0 <= v < 53 for v in source)
                and source != [26, 26], 'evaluator source geometry')
        probability = B.posterior(np.ones((53, 53), np.float64)/2808, public, self.kernels[regime], np)
        state = A.A.posterior_record(probability)
        c.equal(next(events), {'kind': 'reset', **canonical, 'public': public, 'posterior_after': state,
                               'source_evaluation_only': source}, 'evaluation initial public state')
        pair = (regime, seed)
        if pair in self.sources:
            c.equal((source, public), self.sources[pair], 'paired sampled source and reset')
        else:
            self.sources[pair] = (source, public)
        draws = iter(row['draws_evaluation_only'])
        first = next(draws)
        c.equal((first['channel'], first['index'], first['selected_index']), ('source', 0, 53*source[0]+source[1]), 'source draw identity')
        learned = arm != 'analytic_inbounds'
        totals = {key: [] for key in ('choose_seconds', 'update_seconds', 'environment_seconds', 'choose_instrumented_seconds', 'choose_excluded_io_seconds')}
        positions, zero = [], 0
        for step in range(1, steps+1):
            self.check()
            ctx = {**context, 'step': step}
            operation = self.work.call('value_forward' if learned else 'analytic_choose', ctx)
            native = self.work.call('native_step', ctx)
            event = next(events)
            c.equal((event['kind'], event['episode_id'], event['step']), ('step', episode_id, step), 'every ordered evaluation decision')
            allowed = public['valid_actions']
            c.equal(event['allowed_actions'], allowed, 'public-only action mask')
            action = choice(event['costs'], allowed, learned, np)
            c.equal(event['action'], action, 'exact first eligible recorded-cost action')
            if learned:
                x, raw, weight = branches(probability, public['position'], self.kernels[regime], REGIMES[regime], np)
                c.equal(event['raw_masses'], raw.tolist(), 'all sixteen independent raw branch masses')
                c.equal(event['weights'], weight.tolist(), 'all sixteen independent exact floors')
                value = 64*self.predicted(x, arm)
                rebuilt = costs(value, weight, np)
                self.arrays_close(np.asarray(event['values'], np.float64), value, 'independent physical branch values')
                self.arrays_close(np.asarray(event['costs'], np.float64), rebuilt, 'independent explicit four costs')
                c.equal(action, choice(rebuilt.tolist(), allowed, True, np), 'independent exact selected action; no tie exemption')
            else:
                c.equal((event['raw_masses'], event['weights'], event['values']), (None, None, None), 'analytic has no neural telemetry')
                rebuilt = analytic_scores(probability, public['position'], self.kernels[regime], np)
                for i in allowed:
                    c.close(event['costs'][i], rebuilt[i], 'independent analytic objective')
                c.equal(action, choice(rebuilt, allowed, False, np), 'independent analytic action')
            positions.append(tuple(public['position']))
            zero += state['mass'] == 0
            target = B.move(public['position'], action)
            found = target == source
            require(target != public['position'] and (not found or step == steps), 'inbounds motion and immediate found stopping')
            observed = event['public']['hit']
            require(type(observed) is int and (observed == -2 if found else 0 <= observed < 4), 'hit category or terminal sentinel')
            after = B.packet(target, observed, found, step)
            c.equal(event['public'], after, 'independent public transition')
            c.equal(event['native_p_end'], float(found), 'sampled native termination')
            c.equal(event['posterior_before'], state, 'before-action public posterior')
            probability = B.posterior(probability, after, self.kernels[regime], np)
            state = A.A.posterior_record(probability)
            c.equal(event['posterior_after'], state, 'all updates including final found/censor')
            if not found:
                draw = next(draws)
                c.equal((draw['channel'], draw['index'], draw['selected_index']), ('hit', step-1, observed), 'chronological public hit draw')
            for key, values in totals.items():
                value = event[key]
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'finite nonnegative operation timing')
                values.append(value)
            c.close(event['environment_seconds'], native['seconds'], 'environment operation cost')
            c.close(event['choose_seconds'], event['choose_instrumented_seconds']-event['choose_excluded_io_seconds'], 'only recorded I/O excluded')
            require(event['choose_seconds']+1e-9 >= operation['seconds'], 'complete branch/feature/readout/mask duration encloses scalar operation')
            public = after
        B.exhausted(draws, 'episode draw sequence')
        for draw in row['draws_evaluation_only']:
            require(0 <= draw['uniform'] < 1 and math.isfinite(draw['cdf_mass']) and abs(draw['cdf_mass']-1) < 1e-10, 'finite recorded random draw')
            key = (*pair, draw['channel'], draw['index'])
            if key in self.uniforms:
                c.equal(draw['uniform'], self.uniforms[key], 'paired channel-index uniform witness')
            else:
                self.uniforms[key] = draw['uniform']
        c.equal(row['final_public'], public, 'complete final packet')
        c.equal((row['found'], row['updates'], row['blocked_steps'], row['final_update_assimilated']),
                (public['done'], steps, 0, True), 'all complete outcomes/updates retained')
        tail = positions[-256:]
        diagnostic = (zero, state['mass'], sum(tail[i] == tail[i-2] for i in range(2, len(tail))),
                      max(0, len(tail)-2), len(set(positions)))
        c.equal(tuple(row[k] for k in ('zero_mass_decisions', 'final_posterior_mass', 'last256_lag2_matches', 'last256_lag2_pairs', 'distinct_preaction_positions')),
                diagnostic, 'descriptive mass and spatial repetition only')
        for key, values in totals.items():
            c.close(row[key], math.fsum(values), 'summed full episode operation costs')
        c.close(row['setup_allocation_seconds'], 0., 'physical episode before setup allocation')
        c.close(row['controller_seconds'], math.fsum(row[k] for k in TIMES), 'complete physical controller cost')
        require(math.isfinite(row['init_seconds']) and row['init_seconds'] >= 0, 'finite controller initialization')
        c.equal(row['state_bytes'], 22472, 'full public belief state storage')
        storage = row['storage']
        c.equal(set(storage), {'public_actor', 'head', 'branch_workspace'}, 'all retained storage components')
        c.equal({k: v for k, v in storage['public_actor'].items() if k != 'scope'},
                {'immutable_array_bytes': 366368+91592, 'mutable_array_bytes': 22472,
                 'immutable_arrays': {'observation_kernel': 366368, 'manhattan_distance_table': 91592}}, 'including inherited unused distance table')
        c.equal({k: v for k, v in storage['head'].items() if k != 'scope'} if learned else storage['head'],
                A.head_storage('mlp8') if learned else None, 'actual float64 runtime head storage')
        c.equal(storage['branch_workspace'], '16*105*105 float64 centered u and z plus finite branch/features workspace, transient', 'transient workspace disclosure')
        return row

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists() and not any(p.is_symlink() for p in self.out.parents),
                'exclusive absolute audit output')
        self.out.mkdir(parents=True, exist_ok=False)
        old_alarm = signal.getsignal(signal.SIGALRM)
        try:
            require(B.digest(ROOT/B.CLOCK)['sha256'] == B.CLOCK_PIN, 'qualified suspend-inclusive clock')
            self.clock = B.load(ROOT/B.CLOCK, '_bellman_control_audit_clock').SuspendClock()
            self.start = self.clock.now_ns()
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('audit emergency cap')))
            signal.setitimer(signal.ITIMER_REAL, LIMITS['native_seconds'])
            self.receipt.update(source=B.digest(Path(__file__), self.check), independent_helper={'path': str(HELPER), 'sha256': HELPER_PIN})
            B.write(self.out/'started.json', {'request': {k: str(v) for k, v in vars(self.args).items()}, 'limits': LIMITS,
                                            'clock_backend': self.clock.backend, 'started_ns': self.start})
            plan, worker = self.authenticate()
            result = self.compute(plan, worker)
            self.authenticate()
            B.write(self.out/'summary.json', result)
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status='completed', agreement=True, clock_backend=self.clock.backend, started_ns=self.start,
                                finished_ns=finished, wall_seconds=(finished-self.start)/1e9, comparisons=result['comparisons'],
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
            except BaseException as secondary:  # noqa: BLE001 - Preserve primary failure.
                error.add_note(f'Failure preservation: {secondary!r}')
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)

    def load_arrays(self, path):
        self.check()
        with self.np.load(path, allow_pickle=False) as archive:
            return {k: archive[k] for k in archive.files}

    def predicted(self, x, arm):
        self.receipt['saved_checkpoint_readout_calls'] += 1
        self.receipt['saved_network_rows'] += len(x)
        return A.predict(x, self.heads[arm], 'mlp8', self.np)

    def numerical_inputs(self, plan):
        import numpy as np
        self.np, self.maximum_prediction_error = np, 0.
        self.scalar_run = (ROOT/plan['inputs']['prior_receipt']['path']).parent
        original_plan = B.read(ROOT/plan['inputs']['prior_plan']['path'])
        self.prior_run = (ROOT/original_plan['inputs']['prior_receipt']['path']).parent
        self.kernels, self.mixtures, self.datasets, self.heads = {}, {}, {}, {}
        for regime in REGIMES:
            archive = self.load_arrays(self.scalar_run/f'kernel-{regime}.npz')
            self.c.equal(set(archive), {'likelihood', 'initial_hit_weights'}, 'kernel array closure')
            kernel, weight = archive['likelihood'], archive['initial_hit_weights']
            require(kernel.shape == (4, 107, 107) and kernel.dtype == np.float64 and np.isfinite(kernel).all()
                    and (kernel >= 0).all() and (kernel <= 1).all() and not kernel[:, 53, 53].any(), 'frozen kernel geometry')
            require(weight.shape == (4,) and weight.dtype == np.float64 and weight[0] == 0
                    and np.isfinite(weight).all() and (weight[1:] > 0).all()
                    and abs(float(weight.sum())-1) <= 1e-12, 'positive initial mixture')
            self.kernels[regime], self.mixtures[regime] = kernel, {h: float(weight[h]) for h in (1, 2, 3)}
        for split, n in (('train', NTRAIN), ('valid', NVALID)):
            data = self.load_arrays(self.scalar_run/f'{split}-data.npz')
            self.c.equal(set(data), {'features', 'target', 'beliefs', 'positions', 'sensing_length'}, 'dataset closure')
            for key, shape, dtype in (('features', (n, DIMENSION), np.float32), ('target', (n,), np.float32),
                                     ('beliefs', (n, 53, 53), np.float64), ('positions', (n, 2), np.int64),
                                     ('sensing_length', (n,), np.float64)):
                require(data[key].shape == shape and data[key].dtype == dtype and np.isfinite(data[key]).all(), 'canonical cache geometry')
            data['rows'] = list(B.lines(self.scalar_run/f'{split}-rows.jsonl', self.check))
            self.c.equal([r['row_index'] for r in data['rows']], list(range(n)), 'complete cache row order')
            self.datasets[split] = data
        self.baseline = float(np.float32(self.datasets['train']['target'].mean(dtype=np.float64)))
        self.original_exports = {}
        for seed in SEEDS:
            original = self.load_arrays(self.scalar_run/f'final-mlp8-{seed}.npz')
            self.original_exports[seed] = original
            self.heads[f'reference@{seed}'] = A.checkpoint(original, 'mlp8', self.baseline, np)
            for mode in ('mc', 'backup'):
                self.heads[f'{mode}@{seed}'] = A.checkpoint(self.load_arrays(self.args.run/f'final-{mode}-{seed}.npz'),
                                                          'mlp8', self.baseline, np)
        # Inherited independent scalar audit reconstructs all 240 original teacher
        # episodes only; neither DAgger nor prior EVAL is advanced into.
        self.reconstruct_data()
        self.c.equal(self.data_rows, {'train': NTRAIN, 'valid': NVALID}, 'complete inherited current-state cache')
        prep = B.read(self.args.run/'preparation.json')
        self.c.equal({k: prep[k] for k in ('input_rows', 'input_episodes', 'selected_rows', 'row_indices', 'mode')},
                     {'input_rows': NTRAIN, 'input_episodes': 192, 'selected_rows': NTRAIN,
                      'row_indices': list(range(NTRAIN)), 'mode': 'study'}, 'all TRAIN rows, no qualification subset')
        self.c.equal(prep['input_files'], {name: B.digest(self.scalar_run/name, self.check)
                                          for name in ('train-data.npz', 'train-rows.jsonl')}, 'exact dataset file joins')
        self.preparation_seconds = finite(prep['seconds'], 'paid preparation')

    def parity_model(self, arm, iterator):
        np, c = self.np, self.c
        self.work.call('parity_restore', {'phase': 'parity_restore', 'fit_id': arm})
        data = self.datasets['train']
        for index in range(16):
            self.check()
            row = next(iterator)
            c.equal((row['fit_id'], row['row_index']), (arm, index), 'all fixed first16 TRAIN parity rows')
            context = {'phase': 'parity', 'fit_id': arm, 'step': index}
            for channel in ('parity_numpy_forward', 'parity_torch_forward', 'parity_numpy_forward', 'parity_torch_forward'):
                self.work.call(channel, context)
            scalar = self.predicted(self.raw_features('train', index, index+1), arm)
            x, _, weight = branches(data['beliefs'][index], data['positions'][index].tolist(),
                                    self.kernels[data['rows'][index]['regime']], float(data['sensing_length'][index]), np)
            values = 64*self.predicted(x, arm)
            score = costs(values, weight, np)
            for label, expected in (('scalar', scalar), ('values', values), ('costs', score)):
                actual = np.asarray(row[label+'_numpy'], np.float64)
                self.arrays_close(actual, expected, 'independent parity '+label)
                self.arrays_close(actual, np.asarray(row[label+'_torch'], np.float64), 'recorded Torch-double parity '+label)
            allowed = data['rows'][index]['public']['valid_actions']
            c.equal(row['allowed_actions'], allowed, 'fixed parity eligibility')
            expected_action = choice(score.tolist(), allowed, True, np)
            c.equal((row['action_numpy'], row['action_torch'], row['passed']), (expected_action, expected_action, True), 'exact parity actions')
            for suffix in ('numpy', 'torch'):
                c.equal(row['action_'+suffix], choice(row['costs_'+suffix], allowed, True, np), 'strict saved-cost tie')

    def predictions(self, arm, diagnostics):
        np, c = self.np, self.c
        archive = self.load_arrays(self.args.run/f'predictions-{arm.replace("@", "-")}.npz')
        c.equal(set(archive), {'train', 'valid'}, 'both fixed final prediction splits')
        stats = {}
        for split in ('train', 'valid'):
            data = self.datasets[split]
            require(archive[split].dtype == np.float64 and archive[split].shape == data['target'].shape, 'saved prediction geometry')
            for index in range(len(data['target'])):
                if index % 256 == 0:
                    self.check()
                self.work.call('saved_value_forward', {'phase': 'saved_predictions', 'fit_id': arm,
                    'split': split, 'step': (index//256)*256})
                prediction = self.predicted(self.raw_features(split, index, index+1), arm)
                self.arrays_close(archive[split][index:index+1], prediction, 'every final current-state value')
            error = archive[split]-data['target'].astype(np.float64)
            stats[split] = {'rows': len(error), 'mse_normalized': float(np.mean(error**2)),
                            'mae_physical': float(64*np.mean(np.abs(error)))}
        recorded = next(diagnostics)
        c.equal(recorded['fit_id'], arm, 'ordered descriptive prediction record')
        c.tree(recorded['statistics'], stats, 'independent final MC diagnostics')
        self.final_diagnostics[arm] = stats

    def checkpoint_record(self, record, filename):
        self.c.equal(record['path'], filename, 'canonical checkpoint/payload path')
        self.c.equal({k: record[k] for k in ('sha256', 'bytes')}, B.digest(self.args.run/filename, self.check), 'saved bytes identity')

    def refresh(self, arm, seed, index, checkpoints, refreshes, progress):
        np, c = self.np, self.c
        epoch = 1+5*index
        export_op = self.work.call('checkpoint_export', {'fit_id': arm, 'phase': 'target_checkpoint', 'epoch': epoch})
        progress['calls']['checkpoint_export']['attempted'] += 1
        progress['calls']['checkpoint_export']['returned'] += 1
        record = next(checkpoints)
        c.equal((record['fit_id'], record['refresh_index'], record['epoch']), (arm, index, epoch), 'every delayed refresh checkpoint')
        filename = f'target-checkpoint-backup-{seed}-{index:02d}.npz'
        self.checkpoint_record(record, filename)
        checkpoint = self.load_arrays(self.args.run/filename)
        A.checkpoint(checkpoint, 'mlp8', self.baseline, np)
        identity = content_identity(checkpoint)
        c.equal(record['content_identity'], identity, 'refresh exact f32 checkpoint content')
        if index == 0:
            c.equal(identity, content_identity(self.original_exports[seed]), 'first refresh from original checkpoint')
        handle = self.work.begin('target_refresh', {'fit_id': arm, 'phase': 'target_refresh', 'epoch': epoch, 'refresh_index': index})
        for row in range(NTRAIN):
            if row % 256 == 0:
                self.check()
            self.work.call('target_readout', {'phase': 'target_generation', 'fit_id': arm, 'refresh_index': index, 'step': row})
        operation = self.work.end('target_refresh', handle)
        progress['calls']['target_refresh']['attempted'] += 1
        progress['calls']['target_refresh']['returned'] += 1
        progress['refreshes_completed'] += 1
        progress['context'] = {'phase': 'target_refresh', 'epoch': epoch, 'refresh_index': index}
        refresh = next(refreshes)
        metadata = {'epoch': epoch, 'refresh_index': index, 'source_epochs_completed': epoch-1,
                    'rows': NTRAIN, 'checkpoint_identity': identity}
        c.equal({k: refresh[k] for k in metadata}, metadata, 'frozen target provenance')
        c.equal(refresh['fit_id'], arm, 'target fit join')
        c.equal(refresh['checkpoint'], {k: record[k] for k in ('path', 'sha256', 'bytes', 'content_identity')}, 'durable checkpoint before target readout')
        self.checkpoint_record(refresh['payload'], f'targets-backup-{seed}-{index:02d}.npz')
        payload = self.load_arrays(self.args.run/refresh['payload']['path'])
        c.equal(payload['row_indices'].tolist(), list(range(NTRAIN)), 'all canonical target rows in order')
        c.equal(refresh['statistics'], target_statistics(payload['targets_float64'], payload['targets_float32'], np), 'saved target statistics')
        c.equal(refresh['progress'], progress, 'refresh progress')
        self.refresh_records.append(refresh)
        return {**metadata, 'statistics': refresh['statistics']}, export_op['seconds']+operation['seconds']

    def fit_records(self):
        np, c = self.np, self.c
        streams = {name: iter(B.lines(self.args.run/(name+'.jsonl'), self.check)) for name in
                   ('fits', 'initializations', 'updates', 'epoch-orders', 'fit-curves', 'target-checkpoints',
                    'target-refreshes', 'parity', 'prediction-diagnostics')}
        self.fit_seconds, self.final_diagnostics, self.refresh_records = {}, {}, []
        for seed in SEEDS:
            self.parity_model(f'reference@{seed}', streams['parity'])
            mc = self.load_arrays(self.args.run/f'mc-targets-{seed}.npz')
            c.equal(set(mc), {'targets_float64', 'targets_float32'}, 'MC target cache fields')
            for name, dtype in (('targets_float64', np.float64), ('targets_float32', np.float32)):
                require(mc[name].dtype == dtype and np.array_equal(mc[name], self.datasets['train']['target'].astype(dtype)), 'unchanged MC labels/cast')
            for mode in ('mc', 'backup'):
                arm = f'{mode}@{seed}'
                paid = 0.
                progress = {'mode': mode, 'seed': seed, 'epochs_completed': 0, 'updates_completed': 0,
                            'refreshes_completed': 0, 'context': {}, 'calls': {k: {'attempted': 0, 'returned': 0} for k in
                            ('restore', 'optimizer_initialization', 'checkpoint_export', 'target_refresh', 'optimizer_update')}}
                for channel in ('restore', 'optimizer_initialization'):
                    paid += self.work.call(channel, {'fit_id': arm, 'phase': channel})['seconds']
                    progress['calls'][channel] = {'attempted': 1, 'returned': 1}
                progress['context'] = {'phase': 'optimizer_initialization'}
                initial = next(streams['initializations'])
                initial_path = self.args.run/f'initial-{mode}-{seed}.npz'
                archive = self.load_arrays(initial_path)
                A.checkpoint(archive, 'mlp8', self.baseline, np)
                identity = content_identity(archive)
                c.equal(identity, content_identity(self.original_exports[seed]), 'identical paired checkpoint initialization')
                expected = {'fit_id': arm, 'initial_source': str((self.scalar_run/f'final-mlp8-{seed}.npz').relative_to(ROOT)),
                    'initial_source_sha256': B.digest(self.scalar_run/f'final-mlp8-{seed}.npz')['sha256'],
                    'saved': B.digest(initial_path), 'checkpoint_identity': identity,
                    'recipe': {'epochs': 40, 'batch_size': 128, 'learning_rate': .001, 'gradient_clip': 5., 'refresh_every': 5},
                    'mode': mode, 'seed': seed, 'rows': NTRAIN, 'optimizer_states_before': 0,
                    'feature_copy_bytes': NTRAIN*DIMENSION*4, 'progress': progress}
                c.equal(initial, expected, 'complete fresh optimizer/restoration witness')
                rng, recorded_refreshes = np.random.default_rng(seed+30000), []
                for epoch in range(1, 41):
                    if mode == 'backup' and (epoch-1) % 5 == 0:
                        refresh, duration = self.refresh(arm, seed, (epoch-1)//5, streams['target-checkpoints'],
                                                         streams['target-refreshes'], progress)
                        recorded_refreshes.append(refresh)
                        paid += duration
                    order = rng.permutation(NTRAIN)
                    order_sha = hashlib.sha256(order.tobytes()).hexdigest()
                    c.equal(next(streams['epoch-orders']), {'fit_id': arm, 'epoch': epoch, 'rows': NTRAIN,
                            'sha256': order_sha, 'order': order.tolist()}, 'paired exact epoch permutation')
                    losses, norms = [], []
                    for offset in range(0, NTRAIN, 128):
                        size = min(128, NTRAIN-offset)
                        context = {'phase': 'optimizer_update', 'epoch': epoch, 'batch_index': offset//128, 'offset': offset, 'rows': size}
                        paid += self.work.call('optimizer_update', {'fit_id': arm, **context})['seconds']
                        update = next(streams['updates'])
                        c.equal({k: update[k] for k in ('fit_id', *context)}, {'fit_id': arm, **context}, 'every atomic update row/order exposure')
                        losses.append(finite(update['loss'], 'finite MSE')*size)
                        norms.append(finite(update['gradient_norm_before_clip'], 'finite unclipped norm'))
                        progress['calls']['optimizer_update']['attempted'] += 1
                        progress['calls']['optimizer_update']['returned'] += 1
                        progress['updates_completed'] += 1
                        progress['context'] = context
                    progress['epochs_completed'] = epoch
                    curve = {'fit_id': arm, 'epoch': epoch, 'rows': NTRAIN, 'updates': 44, 'order_sha256': order_sha,
                             'training_mse_normalized': math.fsum(losses)/NTRAIN,
                             'mean_gradient_norm_before_clip': math.fsum(norms)/44,
                             'maximum_gradient_norm_before_clip': max(norms),
                             'target_refresh_epoch': 1+5*((epoch-1)//5) if mode == 'backup' else None, 'progress': progress}
                    c.tree(next(streams['fit-curves']), curve, 'independent weighted epoch statistics and counts')
                paid += self.work.call('checkpoint_export', {'fit_id': arm, 'phase': 'final_export'})['seconds']
                progress['calls']['checkpoint_export']['attempted'] += 1
                progress['calls']['checkpoint_export']['returned'] += 1
                progress['context'] = {'phase': 'final_export'}
                final_path = self.args.run/f'final-{mode}-{seed}.npz'
                fit = next(streams['fits'])
                expected = {'fit_id': arm, 'seed': seed, 'mode': mode, 'rows': NTRAIN, 'epochs': 40,
                    'optimizer_steps': [1760]*4, 'initial_identity': identity,
                    'final_identity': content_identity(self.load_arrays(final_path)), 'checkpoint': final_path.name,
                    'checkpoint_sha256': B.digest(final_path)['sha256'], 'progress': progress, 'refreshes': recorded_refreshes}
                c.equal({k: fit[k] for k in expected}, expected, 'fixed final fit and delayed snapshots')
                duration = finite(fit['fit_seconds'], 'complete fit wall')
                require(duration+1e-8 >= paid, 'fit encloses nonoverlapping recorded operations')
                self.fit_seconds[arm] = duration
                self.parity_model(arm, streams['parity'])
                self.predictions(arm, streams['prediction-diagnostics'])
        for name, iterator in streams.items():
            B.exhausted(iterator, 'closed '+name)
        require(len(self.refresh_records) == 24, 'all24 refresh records')

    def audit_targets(self):
        np, c = self.np, self.c
        require(B.digest(ROOT/TARGET, self.check)['sha256'] == TARGET_PIN, 'independent target helper unchanged')
        helper = B.load(ROOT/TARGET, '_bellman_control_target_replay')
        published = B.read(self.args.run/'target-audit.json')
        c.equal(published['status'], 'completed', 'in-worker independent target audit completed')
        require(len(published['refreshes']) == 24, 'complete in-worker target reports')
        results = []
        for record, previous in zip(self.refresh_records, published['refreshes'], strict=True):
            arm, index = record['fit_id'], record['refresh_index']
            self.work.call('target_audit', {'phase': 'target_audit', 'fit_id': arm, 'refresh_index': index})
            self.receipt['active_target_refresh'] = {'fit_id': arm, 'refresh_index': index,
                'scope': 'Completed replay counters below exclude this refresh until its full return.'}
            payload = self.load_arrays(self.args.run/record['payload']['path'])
            path = self.args.run/record['checkpoint']['path']
            checkpoint = self.load_arrays(path)
            result = helper.audit_refresh(payload, checkpoint, train=self.datasets['train'],
                eligible_actions=[r['public']['valid_actions'] for r in self.datasets['train']['rows']],
                kernels={REGIMES[k]: v for k, v in self.kernels.items()},
                expected_row_indices=np.arange(NTRAIN, dtype=np.int64),
                checkpoint_sha256=B.digest(path)['sha256'], expected_checkpoint_sha256=record['checkpoint']['sha256'],
                c0=self.baseline, check=self.check)
            completed = {'fit_id': arm, 'refresh_index': index, **result}
            c.tree(previous, completed, 'every target audit row/checkpoint/count/result')
            c.count += result['numeric_items_compared']
            self.receipt['saved_checkpoint_readout_calls'] += result['audit_checkpoint_readout_calls']
            self.receipt['saved_network_rows'] += result['audit_branch_rows']
            self.receipt['active_target_refresh'] = None
            results.append(completed)
        return results

    def compute(self, plan, worker):
        self.numerical_inputs(plan)
        c = self.c
        runtime = B.read(self.args.run/'runtime.json')
        c.equal((runtime['python'], runtime['executable']), (sys.version, plan['python_executable']), 'recorded interpreter')
        c.equal((runtime['torch_threads'], runtime['torch_interop_threads'], runtime['module_allocation_episodes']),
                (1, 1, 1296), 'CPU1 and complete model-module setup allocation')
        c.equal(runtime['environment'], {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}, 'single-thread numerical controls')
        for key in ('shared_setup_seconds', 'model_module_setup_seconds'):
            finite(runtime[key], 'finite setup duration')
        contexts = list(B.lines(self.args.run/'work-contexts.jsonl', self.check))
        c.equal([r['id'] for r in contexts], list(range(len(contexts))), 'contiguous journal contexts')
        require(all(set(r) == {'id', 'context'} and 'step' not in r['context'] for r in contexts)
                and len({json.dumps(r['context'], sort_keys=True) for r in contexts}) == len(contexts), 'unique context identity')
        self.work = Work(iter(B.lines(self.args.run/'work.jsonl', self.check)), [r['context'] for r in contexts], c)
        native = B.read(self.args.run/'native-setup.json')
        c.equal(native['native_resets'], 3, 'only three native setup resets')
        require(len(native['checks']) == 3, 'all three setup regimes')
        for i, (regime, row) in enumerate(zip(REGIMES, native['checks'], strict=True)):
            self.work.call('native_reset', {'phase': 'native_setup', 'regime': regime})
            c.equal((row['regime'], row['template_seed'], row['cached_kernel_exact']), (regime, 12500001+i, True), 'native kernel/mixture witness')
            hit = row['initial_public']['hit']
            require(type(hit) is int and 1 <= hit <= 3, 'positive template hit')
            c.equal(row['initial_public'], B.packet([26, 26], hit, False, 0), 'public template geometry')
        self.fit_records()
        targets = self.audit_targets()
        setup = B.read(self.args.run/'inference-setup.json')
        c.equal(set(setup), set(ARMS[:-1]), 'all nine independent head loads')
        for arm, record in setup.items():
            family, seed = arm.split('@')
            path = self.scalar_run/f'final-mlp8-{seed}.npz' if family == 'reference' else self.args.run/f'final-{family}-{seed}.npz'
            c.equal((record['checkpoint'], record['sha256'], record['allocated_episodes']),
                    (str(path.relative_to(ROOT)), B.digest(path)['sha256'], 144), 'actual checkpoint source and all144 recipients')
            finite(record['seconds'], 'head load wall')
            c.equal({k: v for k, v in record['storage'].items() if k != 'scope'}, A.head_storage('mlp8'), 'identical full runtime parameter count')
        episodes = iter(B.lines(self.args.run/'eval-episodes.jsonl', self.check))
        allocated = iter(B.lines(self.args.run/'evaluation.jsonl', self.check))
        events = iter(B.lines(self.args.run/'eval-transitions.jsonl', self.check))
        self.sources, self.uniforms = {}, {}
        rows = []
        for identity in evaluation_order():
            row = self.episode(next(episodes), identity, events)
            arm = row['arm']
            allocation = setup[arm]['seconds']/144+runtime['model_module_setup_seconds']/1296 if arm in setup else 0.
            completed = {**row, 'setup_allocation_seconds': allocation, 'controller_seconds': row['controller_seconds']+allocation}
            c.equal(next(allocated), completed, 'exact allocated complete controller costs')
            rows.append({k: v for k, v in completed.items() if k != 'draws_evaluation_only'})
        for iterator, name in ((episodes, 'episodes'), (allocated, 'allocated episodes'), (events, 'transitions'),
                               (self.work.iterator, 'operation journal')):
            B.exhausted(iterator, 'complete '+name)
        c.equal(self.work.stack, [], 'no pending nested calls')
        c.equal(self.work.used_contexts, set(range(len(contexts))), 'no unused context')
        c.tree(worker['calls'], self.work.counts, 'every recorded attempt/return/time')
        expected = {'restore': 6, 'optimizer_initialization': 6, 'checkpoint_export': 30, 'target_refresh': 24,
                    'optimizer_update': 10560, 'target_readout': 134136, 'target_audit': 24,
                    'parity_restore': 9, 'parity_numpy_forward': 288, 'parity_torch_forward': 288,
                    'saved_value_forward': 40188, 'native_reset': 1443, 'native_step': sum(r['steps'] for r in rows),
                    'analytic_choose': sum(r['steps'] for r in rows if r['arm'] == 'analytic_inbounds'),
                    'value_forward': sum(r['steps'] for r in rows if r['arm'] != 'analytic_inbounds')}
        c.equal({k: v['returned'] for k, v in self.work.counts.items()}, expected, 'all fixed and outcome-dependent work')
        for channel, cap in (('native_step', 'native_steps'), ('native_reset', 'native_resets'),
                             ('optimizer_update', 'optimizer_updates'), ('target_readout', 'target_readouts')):
            require(expected[channel] <= plan['limits'][cap], 'original operation limit')
        derived = aggregate(rows, self.mixtures)
        published = B.read(self.args.run/'summary.json')
        for key, value in derived.items():
            c.tree(published[key], value, 'independent summary/'+key)
        c.equal(published['version'], 'otto-bellman-control-v1', 'study identity')
        c.equal(worker['pilot_continuation'], derived['pilot_continuation'], 'unchanged42-condition decision')
        c.tree(published['calls'], self.work.counts, 'summary call counts')
        c.equal(published['inference_setup'], setup, 'summary head setup')
        for key in ('shared_setup_seconds', 'model_module_setup_seconds'):
            c.equal(published[key], runtime[key], 'summary setup')
        c.equal((published['paired_source_cases'], published['paired_uniforms']), (144, len(self.uniforms)), 'paired public cases/draw channels')
        training = B.read(self.args.run/'training-costs.json')
        c.equal(training['preparation_seconds'], self.preparation_seconds, 'paid TRAIN preparation')
        c.equal(training['fits'], self.fit_seconds, 'all six complete continuation fit walls')
        scenarios = amortization(derived, setup, self.fit_seconds, self.preparation_seconds, runtime['model_module_setup_seconds'])
        c.tree(published['amortization']['scenarios'], scenarios, 'H1/H100/H10000 independently reconstructed')
        for label, filename in (('original_training_costs', 'training-costs.json'), ('original_runtime', 'runtime.json')):
            c.equal(published['amortization'][label], B.read(self.scalar_run/filename), 'unchanged prior investment disclosure')
        # Nested refresh/readout intervals are NOT added to complete fit wall.
        # Parity, saved diagnostics and target audits happen outside fit walls.
        separate = math.fsum(self.work.counts[k]['seconds'] for k in
                             ('parity_restore', 'parity_numpy_forward', 'parity_torch_forward', 'saved_value_forward', 'target_audit'))
        disjoint = (self.preparation_seconds+math.fsum(self.fit_seconds.values())+separate
                    +math.fsum(r['controller_seconds']+r['environment_seconds'] for r in rows)+runtime['shared_setup_seconds'])
        require(disjoint <= worker['wall_seconds']+1e-8, 'nonoverlapping paid work within actual native interval')
        require(math.fsum(r['choose_excluded_io_seconds'] for r in rows) <= worker['artifact_io_seconds']+1e-8, 'excluded I/O is recorded')
        return {'version': VERSION, 'agreement': True, **derived, 'amortization': scenarios,
                'comparisons': c.count, 'maximum_scalar_difference': c.maximum_difference,
                'maximum_prediction_difference': self.maximum_prediction_error,
                'dataset_rows': self.data_rows, 'c0_float32': self.baseline, 'fits': 6, 'unchanged_references': 3,
                'target_refreshes': targets, 'epoch_orders': 240, 'parity_records': 144,
                'final_prediction_diagnostics': self.final_diagnostics, 'actual_work': expected,
                'saved_checkpoint_readout_calls': self.receipt['saved_checkpoint_readout_calls'],
                'saved_network_rows': self.receipt['saved_network_rows'],
                'condition_counts': {'competence': 18, 'improvement': 24},
                'costs': {'training': training, 'runtime': runtime, 'inference_setup': setup,
                          'disjoint_accounted_seconds': disjoint, 'worker_seconds': worker['wall_seconds']},
                'scope': SCOPE}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'run', 'terminal', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    for flag in ('plan-sha256', 'receipt-sha256', 'terminal-sha256'):
        parser.add_argument('--'+flag, required=True)
    Audit(parser.parse_args()).execute()
