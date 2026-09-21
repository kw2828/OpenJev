"""Fresh fixed-policy TRAIN collection for the matched coverage intervention.

No fitting, target labels or held-out evaluation. Original scalar mechanics stay
unchanged; the new loop saves public pre-action states and all terminal updates.
"""
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
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CAPACITY = 'scripts/study_otto_capacity.py'
CAPACITY_PIN = '5043e550d17346b759bc8e1b3507a79721137b34c71ebfee79cccc807f7da7de'
BELLMAN = 'scripts/study_otto_bellman_control.py'
BELLMAN_PIN = '055aadac7349ccc90da6fd21c28b1bb61f2a0b7205a64f8f2444b960d87beab6'
VERSION = 'otto-coverage-collection-v1'
SEEDS = (10101, 10102, 10103)
REGIMES = {'lambda3': 3., 'lambda4': 4.}
FIRST = {'lambda3': 13100001, 'lambda4': 13200001}
TEMPLATE_FIRST, CASES, HORIZON = 13500001, 12, 2188
LIMITS = {'native_seconds': 900, 'rss_bytes': 8*1024**3, 'output_bytes': 2*1024**3,
          'native_steps': 72*HORIZON, 'native_resets': 74, 'optimizer_updates': 0, 'target_readouts': 0}
AUDIT_LIMITS = {'seconds': 900, 'rss_bytes': 4*1024**3, 'output_bytes': 128*1024**2}
NEW = ('scripts/collect_otto_coverage.py', 'scripts/freeze_otto_coverage.py',
       'scripts/audit_otto_coverage_collection.py', 'src/openjev/research/otto_coverage_selection.py',
       'tests/test_otto_coverage_selection.py', 'tests/test_otto_coverage_collection.py',
       'tests/test_audit_otto_coverage_collection.py', 'research/otto-coverage-protocol.md')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


require(sha(ROOT/CAPACITY) == CAPACITY_PIN, 'unchanged capacity lineage')
C = load(ROOT/CAPACITY, '_coverage_capacity_lineage')
P = C.P
read, write, regular, descriptor, closed = P.read, P.write, P.regular, P.descriptor, P.closed
require(sha(ROOT/BELLMAN) == BELLMAN_PIN, 'unchanged operation accounting source')
B = load(ROOT/BELLMAN, '_coverage_operation_accounting')


def configuration():
    return {'collector_seeds': list(SEEDS), 'regimes': REGIMES, 'first_seeds': FIRST,
            'template_first_seed': TEMPLATE_FIRST, 'cases_per_regime': CASES,
            'initial_hit': '1+case%3', 'rotation': '(regime_index*12+case)%3', 'horizon': HORIZON,
            'teacher_rows': 2799, 'student_rows': 2790, 'mixture_rows': 5589,
            'sampling': 'smallest SHA256(otto-coverage-v1|identity), tie identity; without replacement',
            'strata': 'preserve original lambda/initial-hit counts; integer largest remainders',
            'collector_quota': 'equal within stratum; ascending seed receives remainder',
            'insufficient_quota': 'preserve attempt and stop; no extra cases or replacement',
            'targets': 'none; no return label or fitting', 'external_model_calls': 0}


def payload_names():
    return {'started.json', 'runtime.json', 'native-setup.json', 'inference-setup.json',
            'work-contexts.jsonl', 'work.jsonl', 'collection-transitions.jsonl', 'collection-episodes.jsonl',
            'selection.json', 'mixture-data.npz', 'mixture-rows.jsonl', 'summary.json'}


def collection_order():
    for ri, regime in enumerate(REGIMES):
        for case in range(CASES):
            offset = (ri*CASES+case) % len(SEEDS)
            for collector in SEEDS[offset:]+SEEDS[:offset]:
                yield regime, FIRST[regime]+case, 1+case % 3, collector, case


def authenticate(args):
    require(args.plan.is_absolute() and not args.plan.is_symlink() and sha(args.plan) == args.plan_sha256, 'external plan pin')
    plan = read(args.plan)
    require(plan['version'] == VERSION and plan['status'] == 'frozen_before_execution'
            and plan['configuration'] == configuration() and plan['limits'] == LIMITS
            and plan['independent_audit_limits'] == AUDIT_LIMITS, 'fixed collection contract')
    require(set(NEW) <= plan['sources'].keys() and plan['sources'][CAPACITY] == CAPACITY_PIN, 'complete new source closure')
    for name, pin in plan['sources'].items():
        require(sha(regular(name)) == pin, 'unchanged source: '+name)
    require(set(plan['inputs']) == {'capacity_plan', 'capacity_receipt', 'capacity_terminal', 'capacity_audit'}, 'exact lineage roles')
    paths = {}
    for role, item in plan['inputs'].items():
        paths[role] = regular(item['path'])
        require(descriptor(paths[role]) == {k: item[k] for k in ('bytes', 'sha256')}, 'input pin')
    prior = C.authenticate(SimpleNamespace(plan=paths['capacity_plan'], plan_sha256=sha(paths['capacity_plan'])))
    worker, audit = closed(paths['capacity_receipt']), closed(paths['capacity_audit'])
    require(prior['mode'] == 'study' and worker['mode'] == 'study' and worker['completed_fits'] == 9
            and worker['pending'] == [] and worker['parity_passed'] is True
            and worker['plan_sha256'] == sha(paths['capacity_plan']) and worker['sources'] == prior['sources']
            and worker['inputs'] == prior['inputs'] and set(worker['files']) == C.payload_names('study'), 'complete prior capacity evidence')
    C.terminal_identity(paths['capacity_receipt'], paths['capacity_plan'], paths['capacity_terminal'], CAPACITY)
    require(audit['agreement'] is True and audit['worker_sha256'] == sha(paths['capacity_receipt'])
            and audit['plan_sha256'] == sha(paths['capacity_plan'])
            and audit['terminal_sha256'] == sha(paths['capacity_terminal'])
            and audit['source']['sha256'] == prior['sources']['scripts/audit_otto_capacity.py'], 'independent capacity audit')
    require(all(plan['sources'].get(k) == v for k, v in prior['sources'].items()), 'inherited immutable sources')
    require(plan['python_executable'] == prior['python_executable'] and plan['python_version'] == prior['python_version']
            and plan['all_distributions'] == prior['all_distributions'], 'same authenticated runtime')
    require(plan['seed_review']['passed'] is True and plan['seed_review']['overlap'] == [], 'prospective seed nonreuse review')
    review = regular(plan['seed_ledger_review']['path'])
    require(descriptor(review) == {k: plan['seed_ledger_review'][k] for k in ('bytes', 'sha256')}
            and read(review)['passed'] is True and read(review)['overlaps'] == [], 'actual ledger seed nonreuse review')
    return plan


class Run:
    emit, sync = P.Run.emit, P.Run.sync
    public, witness, physical, restore = P.Run.public, P.Run.witness, P.Run.physical, P.Run.restore
    check, begin, end, call = B.Run.check, B.Run.begin, B.Run.end, B.Run.call

    def __init__(self, args):
        self.args, self.out = args, args.output
        self.plan = self.launch = self.clock = self.start = None
        self.handles, self.calls, self.pending, self.last, self.context_ids = {}, {}, [], {}, {}
        self.sequence, self.io_seconds = 0, 0.
        self.context = {'phase': 'bind'}
        self.receipt = {'version': VERSION, 'status': 'started', 'completed_episodes': 0,
                        'external_model_calls': 0, 'optimizer_updates': 0, 'target_readouts': 0}

    def bind(self):
        require(sha(ROOT/P.CLOCK) == P.CLOCK_PIN, 'qualified suspend-inclusive clock')
        self.clock = load(ROOT/P.CLOCK, '_coverage_clock').SuspendClock()
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
                and launch['cap_seconds'] == LIMITS['native_seconds']
                and launch['deadline_ns'] == launch['started_ns']+LIMITS['native_seconds']*10**9
                and launch['clock_source_sha256'] == P.CLOCK_PIN
                and launch['watchdog_sha256'] == self.plan['sources']['scripts/supervise_dialogue_observation_v2.py'], 'shared native supervision')
        self.receipt.update(plan_sha256=self.args.plan_sha256, supervision_sha256=sha(self.args.supervision),
                            sources=self.plan['sources'], inputs=self.plan['inputs'], limits=LIMITS)
        write(self.out/'started.json', {'request': {k: str(v) for k, v in vars(self.args).items()},
                                      'launch': launch, 'started_ns': self.start})

    def environment(self, regime, seed, hit):
        config = {'Ndim': 2, 'lambda_over_dx': REGIMES[regime], 'R_dt': 2., 'Ngrid': 53,
                  'Nhits': 4, 'draw_source': True, 'norm_Poisson': 'Euclidean'}
        return self.call('native_reset', lambda: self.seeded(self.SourceTracking, seed, config, initial_hit=hit))

    def setup(self):
        os.environ.update(C.THREADS)
        tick = time.perf_counter()
        import numpy as np
        self.np = np
        sys.path.insert(0, str(ROOT/'src'))
        from openjev.research import otto_coverage_selection, otto_return_value, otto_value_branches
        from openjev.research.otto_public import observation, seeded_environment
        from openjev.research.otto_reference_control import SpaceAwareActor
        self.selector = otto_coverage_selection
        self.model, self.branches = otto_return_value, otto_value_branches
        self.observation, self.seeded, self.actor_class = observation, seeded_environment, SpaceAwareActor
        self.SourceTracking = load(ROOT/P.UPSTREAM, '_coverage_native').SourceTracking
        capacity_plan = read(regular(self.plan['inputs']['capacity_plan']['path']))
        self.prior_run = regular(capacity_plan['inputs']['prior_receipt']['path']).parent
        self.kernels, self.mixtures = {}, {}
        checks = []
        for i, regime in enumerate(REGIMES):
            with np.load(self.prior_run/f'kernel-{regime}.npz', allow_pickle=False) as a:
                self.kernels[regime], self.mixtures[regime] = a['likelihood'], a['initial_hit_weights']
            self.context = {'phase': 'native_setup', 'regime': regime}
            env = self.environment(regime, TEMPLATE_FIRST+i, None)
            weights = np.asarray(next(v for v in env.draw_log if v['channel'] == 'initial')['probabilities'])
            require(np.array_equal(env.p_Poisson, self.kernels[regime])
                    and np.array_equal(weights, self.mixtures[regime]), 'native cached kernel/mixture parity')
            self.kernels[regime].setflags(write=False)
            checks.append({'regime': regime, 'seed': TEMPLATE_FIRST+i, 'exact': True, 'public': self.public(env, 0)})
        write(self.out/'native-setup.json', {'checks': checks, 'resets': 2})
        self.heads, setup = {}, {}
        for seed in SEEDS:
            path = self.prior_run/f'final-mlp8-{seed}.npz'
            self.context = {'phase': 'collector_setup', 'collector_seed': seed}
            start = time.perf_counter()
            self.heads[seed] = self.call('model_load', lambda path=path: self.model.FrozenValue(self.restore(path)))
            setup[str(seed)] = {'path': str(path.relative_to(ROOT)), **descriptor(path),
                               'seconds': time.perf_counter()-start, 'storage': self.heads[seed].storage_bytes()}
        write(self.out/'inference-setup.json', setup)
        self.original_rows = [json.loads(v) for v in (self.prior_run/'train-rows.jsonl').read_text().splitlines()]
        self.quotas = self.selector.coverage_quotas(self.original_rows)
        require(json.loads(json.dumps(self.quotas)) == self.plan['quota_table'], 'prospective exact quota table')
        self.reservoirs, self.available = {}, Counter()
        for q in self.quotas:
            for collector, quota in q['collector_rows'].items():
                self.reservoirs[(q['regime'], q['initial_hit'], collector)] = self.selector.SaltedSelector(quota)
        self.setup_seconds = time.perf_counter()-tick
        write(self.out/'runtime.json', {'python': sys.version, 'executable': sys.executable, 'threads': C.THREADS,
                                       'setup_seconds': self.setup_seconds, 'optimizer_calls': 0})

    def episode(self, regime, seed, hit, collector, case):
        episode_id = f'train:{regime}:{seed}:mlp8@{collector}'
        self.context = {'phase': 'collection', 'episode': episode_id, 'step': 0}
        env = self.environment(regime, seed, hit)
        current = self.public(env, 0)
        tick = time.perf_counter()
        actor = self.actor_class(current, self.kernels[regime], allow_stay=False)
        init_seconds = time.perf_counter()-tick
        state = self.witness(actor, env, current)
        identity = {'episode_id': episode_id, 'stage': 'train_collection', 'regime': regime,
                    'seed': seed, 'initial_hit': hit, 'collector_seed': collector, 'case': case}
        self.emit('collection-transitions.jsonl', {'kind': 'reset', **identity, 'public': current,
                  'posterior_after': state, 'source_evaluation_only': env.source.tolist()})
        choose_seconds = update_seconds = native_seconds = sampling_seconds = 0.
        raw_choices = excluded_choices = 0.
        for step in range(1, HORIZON+1):
            self.context['step'] = step
            self.check()
            row_id = self.selector.student_identity(episode_id, step-1)
            def snapshot(step=step, current=current, state=state):
                return {'belief': actor.belief.copy(), 'metadata': {**identity, 'prefix_index': step-1,
                        'public': dict(current), 'posterior': dict(state), 'source': 'student', 'source_row_index': None}}
            tick = time.perf_counter()
            self.reservoirs[(regime, hit, collector)].offer(row_id, snapshot)
            self.available[(regime, hit, collector)] += 1
            sampling_seconds += time.perf_counter()-tick
            tick, io = time.perf_counter(), self.io_seconds
            branch = self.branches.rl_branches(actor.belief, current['position'], self.kernels[regime], current['valid_actions'])
            values = None
            def callback(z, positions, _kernel):
                nonlocal values
                values = self.call('value_forward', lambda: self.physical(self.heads[collector], z, positions, REGIMES[regime]), check=False)
                return values
            costs = self.branches.explicit_scores(branch, callback, arithmetic='float64')
            action = self.branches.select_action(costs, branch.eligible_actions)
            raw, excluded = time.perf_counter()-tick, self.io_seconds-io
            elapsed = raw-excluded
            require(elapsed >= 0, 'complete nonnegative choice cost')
            choose_seconds += elapsed
            raw_choices += raw
            excluded_choices += excluded
            result = self.call('native_step', lambda action=action: env.step(action, quiet=True))
            native = self.last['native_step']['seconds']
            native_seconds += native
            after = self.public(env, step)
            require((int(result[0]), bool(result[2])) == (after['hit'], after['done']), 'native event identity')
            tick = time.perf_counter()
            actor.update(action, after)
            update = time.perf_counter()-tick
            update_seconds += update
            after_state = self.witness(actor, env, after)
            self.emit('collection-transitions.jsonl', {'kind': 'step', 'episode_id': episode_id, 'step': step,
                'action': action, 'costs': costs.tolist(), 'allowed_actions': list(current['valid_actions']),
                'raw_masses': branch.raw_masses.tolist(), 'weights': branch.weights.tolist(), 'values': values.tolist(),
                'public': after, 'posterior_before': state, 'posterior_after': after_state,
                'choose_seconds': elapsed, 'choose_instrumented_seconds': raw, 'choose_excluded_io_seconds': excluded,
                'update_seconds': update, 'environment_seconds': native, 'native_p_end': float(result[1])})
            require(current['position'] != after['position'], 'only inbounds movement')
            current, state = after, after_state
            if current['done']:
                break
        draws = [{k: v[k] for k in ('channel', 'index', 'uniform', 'selected_index', 'cdf_mass')} for v in env.draw_log]
        row = {**identity, 'steps': step, 'found': current['done'], 'updates': step, 'blocked_steps': 0,
               'init_seconds': init_seconds, 'choose_seconds': choose_seconds, 'update_seconds': update_seconds,
               'controller_seconds': init_seconds+choose_seconds+update_seconds, 'environment_seconds': native_seconds,
               'choose_instrumented_seconds': raw_choices, 'choose_excluded_io_seconds': excluded_choices,
               'sampling_seconds': sampling_seconds, 'source_evaluation_only': env.source.tolist(),
               'draws_evaluation_only': draws, 'final_public': current, 'final_posterior_mass': state['mass'],
               'final_update_assimilated': True, 'training_labels_generated': False}
        self.emit('collection-episodes.jsonl', row)
        self.sync()
        return row

    def prepare_mixture(self):
        np = self.np
        tick = time.perf_counter()
        with np.load(self.prior_run/'train-data.npz', allow_pickle=False) as a:
            teacher = {k: a[k] for k in ('features', 'beliefs', 'positions', 'sensing_length')}
        selected, groups = [], []
        for q in self.quotas:
            reservoir = self.selector.SaltedSelector(q['teacher_rows'])
            for i, row in enumerate(self.original_rows):
                if (row['regime'], row['initial_hit']) == (q['regime'], q['initial_hit']):
                    reservoir.offer(self.selector.teacher_identity(i), lambda i=i: i)
            items = reservoir.finish()
            groups.append({'source': 'teacher', 'regime': q['regime'], 'initial_hit': q['initial_hit'],
                           'collector_seed': None, 'available': q['original_rows'], 'selected': len(items),
                           'rows': [{k: item[k] for k in ('key', 'identity')} for item in items]})
            for item in items:
                i = item['payload']
                r = self.original_rows[i]
                selected.append({'belief': teacher['beliefs'][i], 'feature': teacher['features'][i],
                    'metadata': {'source': 'teacher', 'source_row_index': i, 'collector_seed': None,
                        'episode_id': r['episode_id'], 'stage': 'train', 'seed': r['seed'], 'case': None,
                        'regime': r['regime'], 'initial_hit': r['initial_hit'], 'prefix_index': r['prefix_index'],
                        'public': r['public'], 'posterior': r['posterior']}, 'key': item['key'], 'identity': item['identity']})
        for q in self.quotas:
            for collector in SEEDS:
                key = (q['regime'], q['initial_hit'], collector)
                items = self.reservoirs[key].finish()
                groups.append({'source': 'student', 'regime': key[0], 'initial_hit': key[1],
                    'collector_seed': collector, 'available': self.available[key], 'selected': len(items),
                    'rows': [{k: item[k] for k in ('key', 'identity')} for item in items]})
                for item in items:
                    selected.append({**item['payload'], 'key': item['key'], 'identity': item['identity']})
        require(len(selected) == 5589, 'exact complete mixture quota')
        beliefs = np.stack([v['belief'] for v in selected])
        positions = np.asarray([v['metadata']['public']['position'] for v in selected], dtype=np.int64)
        sensing = np.asarray([REGIMES[v['metadata']['regime']] for v in selected], dtype=np.float64)
        features = np.empty((5589, 11028), dtype=np.float32)
        for i, row in enumerate(selected):
            if row['metadata']['source'] == 'teacher':
                features[i] = row['feature']
            else:
                centered = P.center(beliefs[i:i+1], positions[i:i+1], np)
                features[i] = self.model.value_features(centered, positions[i:i+1], sensing[i], dtype='float32')[0]
            metadata = {'mixture_index': i, **row['metadata'], 'identity': row['identity'], 'key': row['key']}
            self.emit('mixture-rows.jsonl', metadata)
        counts = Counter((v['metadata']['regime'], v['metadata']['initial_hit']) for v in selected)
        require(counts == Counter((r['regime'], r['initial_hit']) for r in self.original_rows), 'preserved sensing/hit row exposure')
        self.check()
        with (self.out/'mixture-data.npz').open('xb') as stream:
            np.savez_compressed(stream, features=features, beliefs=beliefs, positions=positions, sensing_length=sensing)
        original = {v.tobytes(order='C') for v in teacher['features']}
        student = [v.tobytes(order='C') for v in features[2799:]]
        support = {'student_rows': 2790, 'student_unique_feature_rows': len(set(student)),
                   'student_rows_also_in_full_teacher': sum(v in original for v in student),
                   'student_unique_features_not_in_full_teacher': len(set(student)-original),
                   'identity': 'exact cached float32 feature bytes; duplicate inputs retained'}
        write(self.out/'selection.json', {'quotas': self.quotas, 'groups': groups, 'support': support,
              'row_order': 'teacher strata/hash order, then student strata/collector/hash order',
              'preparation_seconds': time.perf_counter()-tick, 'targets_generated': False})
        return support

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists() and not any(p.is_symlink() for p in self.out.parents), 'exclusive output')
        self.out.mkdir(parents=True, exist_ok=False)
        primary_error = None
        try:
            self.bind()
            self.setup()
            rows, sources, uniforms = [], {}, {}
            tick = time.perf_counter()
            for regime, seed, hit, collector, case in collection_order():
                row = self.episode(regime, seed, hit, collector, case)
                key = (regime, seed)
                require(key not in sources or sources[key] == row['source_evaluation_only'], 'paired native source')
                sources[key] = row['source_evaluation_only']
                for draw in row['draws_evaluation_only']:
                    index = (*key, draw['channel'], draw['index'])
                    require(index not in uniforms or uniforms[index] == draw['uniform'], 'paired random-channel uniforms')
                    uniforms[index] = draw['uniform']
                rows.append(row)
                self.receipt['completed_episodes'] = len(rows)
                if len(rows) % 6 == 0:
                    print(json.dumps({'completed_episodes': len(rows), 'native_steps': self.calls['native_step']['returned']}), flush=True)
            collection_seconds = time.perf_counter()-tick
            support = self.prepare_mixture()
            self.sync()
            summary = {'version': VERSION, 'episodes': len(rows), 'paired_cases': len(sources),
                'native_steps': sum(r['steps'] for r in rows), 'found': sum(r['found'] for r in rows),
                'censored': sum(not r['found'] for r in rows), 'coverage_admitted': True,
                'support': support, 'collection_seconds': collection_seconds,
                'controller_seconds': math.fsum(r['controller_seconds'] for r in rows),
                'environment_seconds': math.fsum(r['environment_seconds'] for r in rows),
                'sampling_seconds': math.fsum(r['sampling_seconds'] for r in rows),
                'setup_seconds': self.setup_seconds, 'native_resets': self.calls['native_reset']['returned'],
                'training_calls': 0, 'target_readouts': 0, 'learned_architecture_advantage_established': False,
                'scope': 'Fresh TRAIN exposure only; collection outcomes are not an independent efficacy comparison.'}
            write(self.out/'summary.json', summary)
            self.sync()
            for handle in self.handles.values():
                handle.close()
            self.handles.clear()
            require(authenticate(self.args) == self.plan, 'unchanged end identity')
            require(len(rows) == 72 and len(sources) == 24 and self.calls['native_reset']['returned'] == 74
                    and self.calls['native_step']['returned'] == self.calls['value_forward']['returned'] == summary['native_steps']
                    and not self.pending and all(v['attempted'] == v['returned'] for v in self.calls.values()), 'complete bounded work')
            require({p.name for p in self.out.iterdir()} == payload_names(), 'exact payload closure')
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status='completed', coverage_admitted=True, calls=self.calls, pending=self.pending,
                started_ns=self.start, finished_ns=finished, clock_backend=self.clock.backend,
                wall_seconds=(finished-self.start)/1e9, files={p.name: descriptor(p) for p in self.out.iterdir()})
            write(self.out/'receipt.json', self.receipt)
            self.check()
        except BaseException as error:
            primary_error = error
            self.receipt.update(status='failed', calls=self.calls, pending=self.pending,
                                error=repr(error), traceback=traceback.format_exc())
            try:
                self.sync()
            except BaseException as secondary:  # noqa: BLE001 - Preserve the original failure.
                error.add_note('Failure flush: '+repr(secondary))
            try:
                if (self.out/'receipt.json').exists():
                    (self.out/'receipt.json').rename(self.out/'invalid-completed-receipt.json')
                write(self.out/'failed.json', self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - Preserve the original failure.
                error.add_note(str(secondary))
            raise
        finally:
            for handle in self.handles.values():
                try:
                    handle.close()
                except BaseException as secondary:
                    if primary_error is not None:
                        primary_error.add_note('Failure cleanup: '+repr(secondary))
                    else:
                        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'output', 'supervision'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    Run(parser.parse_args()).execute()
