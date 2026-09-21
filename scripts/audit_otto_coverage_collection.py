"""Saved-only independent TRAIN collection and deterministic-mixture audit.

No simulator, collector policy, optimizer or producer selection executes. Local
readouts of authenticated saved checkpoints are explicitly counted. Native RNG
and physical timing remain execution evidence, not independently rerun science.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'otto-coverage-collection-saved-audit-v1'
RUNNER = 'scripts/collect_otto_coverage.py'
HELPER = 'scripts/audit_otto_return_value.py'
HELPER_PIN = 'ad3d8e92c0bae3b9dd573af40e1cbb49d1dc771ff23f30500667a46f94980de8'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
LIMITS = {'seconds': 900, 'rss_bytes': 4*1024**3, 'output_bytes': 128*1024**2}
SEEDS = (10101, 10102, 10103)
REGIMES = {'lambda3': 3., 'lambda4': 4.}
FIRST = {'lambda3': 13100001, 'lambda4': 13200001}
CELLS = tuple((regime, hit) for regime in REGIMES for hit in (1, 2, 3))
HORIZON = 2188
SALT = 'otto-coverage-v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def descriptor(path, check=lambda: None):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'regular absolute input')
    value, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while block := stream.read(1024**2):
            check()
            value.update(block)
            size += len(block)
    return {'sha256': value.hexdigest(), 'bytes': size}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def bound_path(value):
    """Historical descriptors can be repository-relative; never depend on cwd."""
    path = Path(value)
    require('..' not in path.parts, 'canonical authenticated descriptor path')
    return path if path.is_absolute() else ROOT/path


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def lines(path, check=lambda: None):
    with Path(path).open() as stream:
        for line in stream:
            check()
            yield json.loads(line)


def equal(actual, expected, name):
    require(actual == expected, name)


def close(actual, expected, name):
    require(type(actual) in (float, int) and math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10), name)


def tree(actual, expected, name):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), name+' fields')
        for key in expected:
            tree(actual[key], expected[key], name+'.'+str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), name+' rows')
        for a, b in zip(actual, expected, strict=True):
            tree(a, b, name)
    elif type(expected) is float:
        close(actual, expected, name)
    else:
        require(type(actual) is type(expected) and actual == expected, name)


def collection_order():
    for regime_index, (regime, first) in enumerate(FIRST.items()):
        for case in range(12):
            shift = (12*regime_index+case) % 3
            for collector in SEEDS[shift:]+SEEDS[:shift]:
                yield regime, first+case, 1+case % 3, collector


def largest_remainders(counts, total):
    """Exact integer Hamilton allocation; lexicographic cell order breaks ties."""
    require(isinstance(counts, dict) and counts and all(type(v) is int and v >= 0 for v in counts.values())
            and type(total) is int and 0 <= total <= sum(counts.values()) and sum(counts.values()) > 0,
            'nonnegative integer proportional quota')
    denominator = sum(counts.values())
    values = {key: total*n//denominator for key, n in counts.items()}
    priority = sorted(counts, key=lambda key: (-(total*counts[key] % denominator), key))
    for key in priority[:total-sum(values.values())]:
        values[key] += 1
    require(sum(values.values()) == total and all(values[k] <= counts[k] for k in counts), 'quota sum and bounds')
    return values


def collector_quotas(total):
    require(type(total) is int and total >= 0, 'integer collector quota')
    return {seed: total//3+int(i < total % 3) for i, seed in enumerate(SEEDS)}


def ranked_selection(identities, quota):
    """Independent full sort, deliberately separate from the producer reservoir."""
    require(type(quota) is int and quota >= 0 and all(isinstance(s, str) and s for s in identities),
            'nonnegative quota and canonical row identities')
    require(len(identities) == len(set(identities)), 'every offered row ID unique')
    require(len(identities) >= quota, 'no outcome-driven refill for insufficient fixed coverage')
    return sorted(identities, key=lambda s: (hashlib.sha256((SALT+'|'+s).encode()).hexdigest(), s))[:quota]


def quota_records(teacher_rows):
    counts = dict.fromkeys(CELLS, 0)
    for index, row in enumerate(teacher_rows):
        require(row['row_index'] == index and (row['regime'], row['initial_hit']) in counts,
                'canonical original teacher row/stratum identity')
        counts[(row['regime'], row['initial_hit'])] += 1
    require(len(teacher_rows) == 5589 and all(counts.values()), 'all5589 original rows and six strata')
    student = largest_remainders(counts, 2790)
    return [{'regime': regime, 'initial_hit': hit, 'original_rows': count,
             'student_rows': student[(regime, hit)], 'teacher_rows': count-student[(regime, hit)],
             'collector_rows': {str(k): v for k, v in collector_quotas(student[(regime, hit)]).items()}}
            for (regime, hit), count in counts.items()]


def recorded_episode(row, identity):
    """Validate fixed membership and found/censored stopping independently."""
    regime, seed, hit, collector = identity
    require((row['regime'], row['seed'], row['initial_hit'], row['collector_seed'])
            == (regime, seed, hit, collector), 'complete rotating collection identity')
    require(type(row['steps']) is int and 1 <= row['steps'] <= HORIZON and type(row['found']) is bool
            and (row['found'] or row['steps'] == HORIZON), 'all found and censored trajectories retained')
    require(row['updates'] == row['steps'] and row['final_update_assimilated'] is True, 'final found/censored public update')
    return row['steps']


def closed(directory, pin, names, check):
    require(directory.is_absolute() and directory.is_dir()
            and not any(p.is_symlink() for p in (directory, *directory.parents)), 'regular evidence directory')
    require(descriptor(directory/'receipt.json', check)['sha256'] == pin, 'external completed receipt pin')
    receipt = read(directory/'receipt.json')
    require(receipt['status'] == 'completed' and set(receipt['files']) == set(names)
            and {p.name for p in directory.iterdir()} == set(names) | {'receipt.json'}, 'exact complete payload closure')
    for name, value in receipt['files'].items():
        require(Path(name).name == name and descriptor(directory/name, check) == value, 'bound payload bytes before numeric decoding')
    return receipt


def terminal_identity(plan, plan_path, worker, started, terminal, run, check):
    launch, request = started['launch'], started['request']
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
            and terminal['group_absent'] is True and terminal['error'] is None and terminal['clock_error'] is None
            and terminal['cleanup']['group_absent'] is True and terminal['cleanup']['reaped'] is True
            and terminal['cleanup']['errors'] == [], 'successful original supervisor and absent process group')
    for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend',
                'cap_seconds', 'watchdog_sha256', 'clock_source_sha256'):
        equal(launch[key], terminal[key], 'same parent launch/terminal')
    require(request == {'plan': str(plan_path), 'plan_sha256': descriptor(plan_path, check)['sha256'],
                        'output': str(run), 'supervision': request['supervision']}, 'exact worker request')
    require(descriptor(request['supervision'], check)['sha256'] == worker['supervision_sha256']
            and read(request['supervision']) == launch, 'exact original launch file')
    command = list(launch['command'])
    if command[1:2] == ['-u']:
        command.pop(1)
    require(command[:2] == [plan['python_executable'], str(ROOT/RUNNER)] and len(command) == 10
            and set(command[2::2]) == {'--plan', '--plan-sha256', '--output', '--supervision'}
            and dict(zip(command[2::2], command[3::2], strict=True)) == {'--'+k.replace('_', '-'): v for k, v in request.items()},
            'literal process interpreter/script/options bind request')
    require(launch['cwd'] == str(ROOT) and launch['pid'] == launch['pgid'] != launch['parent_pid']
            and all(type(launch[k]) is int and launch[k] > 0 for k in ('pid', 'pgid', 'parent_pid')), 'isolated process identity')
    require(all(type(value) is int and value >= 0 for value in (launch['started_ns'], worker['started_ns'],
            worker['finished_ns'], terminal['finished_ns'], launch['deadline_ns'])), 'integer native times')
    require(worker['clock_backend'] == launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
            and launch['clock_source_sha256'] == CLOCK_PIN
            and launch['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py']
            and launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns']
            and worker['started_ns'] == started['started_ns'] and launch['cap_seconds'] == plan['limits']['native_seconds'] == 900
            and launch['deadline_ns'] == launch['started_ns']+900*10**9
            and worker['wall_seconds'] == (worker['finished_ns']-worker['started_ns'])/1e9
            and terminal['elapsed_ns'] == terminal['finished_ns']-launch['started_ns']
            and terminal['wall_seconds'] == terminal['elapsed_ns']/1e9, 'strict native enclosure and worker elapsed')


def public_update(probability, public, event, source, kernel, helper, np):
    require(type(event['step']) is int and event['step'] == public['step']+1
            and type(event['action']) is int and event['action'] in public['valid_actions']
            and event['allowed_actions'] == public['valid_actions'], 'ordered public legal movement')
    target = helper.B.move(public['position'], event['action'])
    found = target == source
    hit = event['public']['hit']
    require(type(hit) is int and (hit == -2 if found else 0 <= hit <= 3), 'found sentinel versus nonterminal category')
    after = helper.B.packet(target, hit, found, event['step'])
    require(event['public'] == after and target != public['position']
            and event['posterior_before'] == helper.A.posterior_record(probability), 'exact pre/post public trace join')
    updated = helper.B.posterior(probability, after, kernel, np)
    require(event['posterior_after'] == helper.A.posterior_record(updated)
            and event['native_p_end'] == float(found), 'every independent public update including final packet')
    return updated, after


def branch_witness(event, values, raw, weights, rebuilt, helper, np):
    """Strict structural floors plus declared f64 arithmetic and exact actions."""
    observed_raw = np.asarray(event['raw_masses'], np.float64)
    observed_weights = np.asarray(event['weights'], np.float64)
    require(observed_raw.shape == observed_weights.shape == (4, 4)
            and np.isfinite(observed_raw).all() and np.isfinite(observed_weights).all()
            and (observed_raw >= 0).all() and np.array_equal(observed_weights, np.maximum(observed_raw, 1e-10)),
            'saved nonnegative branch masses and exact floor identity')
    maximum = 0.
    for recorded, expected, label in ((observed_raw, raw, 'raw branch masses'), (observed_weights, weights, 'floored masses'),
                                      (np.asarray(event['values'], np.float64), values, 'all sixteen physical values'),
                                      (np.asarray(event['costs'], np.float64), rebuilt, 'all four physical costs')):
        require(recorded.shape == expected.shape and np.isfinite(recorded).all(), label+' finite shape')
        difference = np.abs(recorded-expected)
        require(np.all(difference <= 1e-10+1e-10*np.abs(expected)), label+' independent numerical agreement')
        maximum = max(maximum, float(difference.max(initial=0)))
    action = helper.choice(event['costs'], event['allowed_actions'], True, np)
    require(type(event['action']) is int and event['action'] == action
            and action == helper.choice(rebuilt.tolist(), event['allowed_actions'], True, np), 'exact saved and independently replayed near-tie action')
    return maximum


PAYLOADS = {'started.json', 'runtime.json', 'native-setup.json', 'inference-setup.json', 'work-contexts.jsonl',
            'work.jsonl', 'collection-transitions.jsonl', 'collection-episodes.jsonl', 'selection.json',
            'mixture-data.npz', 'mixture-rows.jsonl', 'summary.json'}
SCOPE = ('Independent complete public filtering, all16 saved collector branch values/costs and exact choices, '
         '72 fixed trajectories, terminal/censor chronology, cost joins, salted selection and5589 mixture states. '
         'No simulation, training, target generation or producer numerical/selection code executes. '
         'Native sensor/RNG and physical timing truth remain authenticated execution evidence. '
         'Original teacher posterior truth is inherited from its completed independent audit.')


class Audit:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = None
        self.readout_attempts = self.readout_returns = self.readout_rows = 0
        self.maximum_difference = 0.

    def check(self):
        require(self.clock.now_ns()-self.start < LIMITS['seconds']*10**9, 'saved audit native deadline')
        self.rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
        require(self.rss <= LIMITS['rss_bytes'], 'saved audit RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'saved audit output cap')

    def authenticate(self):
        args = self.args
        require(descriptor(args.plan, self.check)['sha256'] == args.plan_sha256, 'external plan pin')
        plan = read(args.plan)
        require(plan['version'] == 'otto-coverage-collection-v1' and plan['status'] == 'frozen_before_execution'
                and plan['independent_audit_limits'] == LIMITS, 'prospective collection/audit allocation')
        own = str(Path(__file__).resolve().relative_to(ROOT))
        require(plan['sources'][own] == descriptor(Path(__file__), self.check)['sha256']
                and 'tests/test_audit_otto_coverage_collection.py' in plan['sources']
                and plan['sources'][HELPER] == HELPER_PIN and plan['sources'][CLOCK] == CLOCK_PIN, 'pinned independent audit closure')
        for name, pin in plan['sources'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts
                    and descriptor(ROOT/name, self.check)['sha256'] == pin, 'all unchanged inherited/new sources')
        require(descriptor(args.terminal, self.check)['sha256'] == args.terminal_sha256, 'external original terminal pin')
        worker = closed(args.run, args.receipt_sha256, PAYLOADS, self.check)
        require(worker['version'] == plan['version'] and worker['sources'] == plan['sources']
                and worker['inputs'] == plan['inputs'] and worker['limits'] == plan['limits']
                and worker['plan_sha256'] == args.plan_sha256 and worker['completed_episodes'] == 72
                and worker['coverage_admitted'] is True and worker['pending'] == []
                and all(worker[k] == 0 for k in ('external_model_calls', 'optimizer_updates', 'target_readouts')),
                'successful complete untrained collection before numerical decoding')
        require(type(worker['peak_rss_bytes']) is int and 0 < worker['peak_rss_bytes'] <= plan['limits']['rss_bytes']
                and sum(p.stat().st_size for p in args.run.iterdir()) <= plan['limits']['output_bytes'], 'worker resource witnesses')
        terminal_identity(plan, args.plan, worker, read(args.run/'started.json'), read(args.terminal), args.run, self.check)
        producer = load(ROOT/RUNNER, '_coverage_audit_lineage')
        require(producer.authenticate(argparse.Namespace(plan=args.plan, plan_sha256=args.plan_sha256)) == plan,
                'complete capacity/scalar input, audit, runtime and qualification lineage')
        self.plan, self.worker = plan, worker
        capacity = read(bound_path(plan['inputs']['capacity_plan']['path']))
        self.scalar_run = bound_path(capacity['inputs']['prior_receipt']['path']).parent
        self.scalar_receipt = read(self.scalar_run/'receipt.json')

    def arrays(self, path, names=None):
        self.check()
        with self.np.load(path, allow_pickle=False) as archive:
            if names is None:
                result = {k: archive[k] for k in archive.files}
            else:
                require(set(names) <= set(archive.files), 'required cached array members')
                result = {k: archive[k] for k in names}
        self.check()
        return result

    def selected_state(self, identity, probability, metadata):
        if identity not in self.selected:
            return
        require(identity not in self.verified_selected, 'selected public state verified exactly once')
        np, E = self.np, self.E
        index, recorded = self.selected[identity]
        key = hashlib.sha256((SALT+'|'+identity).encode()).hexdigest()
        equal(recorded, {'mixture_index': index, **metadata, 'identity': identity, 'key': key}, 'complete selected state metadata')
        expected = E.state_features(probability, metadata['public']['position'], REGIMES[metadata['regime']], np)
        require(self.mixture['beliefs'][index].tobytes() == probability.tobytes()
                and self.mixture['features'][index].tobytes() == expected.tobytes()
                and self.mixture['positions'][index].tolist() == metadata['public']['position']
                and self.mixture['sensing_length'][index] == REGIMES[metadata['regime']], 'exact selected public belief/features/context')
        self.verified_selected.add(identity)

    def episode(self, row, identity, events):
        E, np = self.E, self.np
        regime, seed, hit, collector = identity
        steps = recorded_episode(row, identity)
        case = seed-FIRST[regime]
        episode_id = f'train:{regime}:{seed}:mlp8@{collector}'
        canonical = {'episode_id': episode_id, 'stage': 'train_collection', 'regime': regime, 'seed': seed,
                     'initial_hit': hit, 'collector_seed': collector, 'case': case}
        equal({k: row[k] for k in canonical}, canonical, 'all72 canonical TRAIN trajectories')
        context = {'phase': 'collection', 'episode': episode_id, 'step': 0}
        reset_cost = self.work.call('native_reset', context)['seconds']
        public = E.B.packet([26, 26], hit, False, 0)
        probability = E.B.posterior(np.ones((53, 53), np.float64)/2808, public, self.kernels[regime], np)
        source = row['source_evaluation_only']
        require(isinstance(source, list) and len(source) == 2 and all(type(v) is int and 0 <= v <= 52 for v in source)
                and source != [26, 26], 'evaluator-only sampled source')
        equal(next(events), {'kind': 'reset', **canonical, 'public': public,
              'posterior_after': E.A.posterior_record(probability), 'source_evaluation_only': source}, 'initial public posterior/reset')
        pair = (regime, seed)
        require(pair not in self.sources or self.sources[pair] == (source, public), 'paired source and initial public packet')
        self.sources[pair] = (source, public)
        draws = iter(row['draws_evaluation_only'])
        draw = next(draws)
        equal((draw['channel'], draw['index'], draw['selected_index']), ('source', 0, 53*source[0]+source[1]), 'conditional source draw witness')
        times = {k: [] for k in ('choose_seconds', 'choose_instrumented_seconds', 'choose_excluded_io_seconds',
                                  'update_seconds', 'environment_seconds')}
        for step in range(1, steps+1):
            self.check()
            state = E.A.posterior_record(probability)
            row_id = episode_id+':'+str(step-1)
            self.candidates[(regime, hit, collector)].append(row_id)
            self.selected_state(row_id, probability, {**canonical, 'prefix_index': step-1, 'public': public,
                                'posterior': state, 'source': 'student', 'source_row_index': None})
            operation = self.work.call('value_forward', {**context, 'step': step})
            native = self.work.call('native_step', {**context, 'step': step})
            event = next(events)
            require(event['kind'] == 'step' and event['episode_id'] == episode_id, 'ordered trajectory event identity')
            x, raw, weights = E.branches(probability, public['position'], self.kernels[regime], REGIMES[regime], np)
            self.readout_attempts += 1
            values = 64*E.predict(x, self.heads[collector], 'mlp8', np)
            self.readout_returns += 1
            self.readout_rows += 16
            costs = E.costs(values, weights, np)
            self.maximum_difference = max(self.maximum_difference, branch_witness(event, values, raw, weights, costs, E, np))
            probability, public = public_update(probability, public, event, source, self.kernels[regime], E, np)
            require(not public['done'] or step == steps, 'immediate stop on found; no post-terminal action')
            if not public['done']:
                draw = next(draws)
                equal((draw['channel'], draw['index'], draw['selected_index']), ('hit', step-1, public['hit']), 'chronological public observation draw')
            for key, values_at_steps in times.items():
                value = event[key]
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'finite nonnegative step timing')
                values_at_steps.append(value)
            close(event['environment_seconds'], native['seconds'], 'native-step cost join')
            close(event['choose_seconds'], event['choose_instrumented_seconds']-event['choose_excluded_io_seconds'], 'only measured I/O removed')
            require(event['choose_seconds']+1e-9 >= operation['seconds'], 'whole choice encloses recorded readout')
        require(next(draws, None) is None, 'no extra native draws')
        for draw in row['draws_evaluation_only']:
            require(set(draw) == {'channel', 'index', 'uniform', 'selected_index', 'cdf_mass'}
                    and type(draw['uniform']) in (int, float) and 0 <= draw['uniform'] < 1
                    and math.isfinite(draw['cdf_mass']) and abs(draw['cdf_mass']-1) < 1e-10, 'finite categorical draw witnesses')
            key = (*pair, draw['channel'], draw['index'])
            require(key not in self.uniforms or self.uniforms[key] == draw['uniform'], 'paired random channel/index uniforms')
            self.uniforms[key] = draw['uniform']
        equal(row['final_public'], public, 'final found/censored public packet')
        require(row['found'] is public['done'] and row['blocked_steps'] == 0
                and row['training_labels_generated'] is False, 'no intervention or fabricated return labels')
        equal(row['final_posterior_mass'], float(probability.sum()), 'final public posterior mass')
        for key, values_at_steps in times.items():
            close(row[key], math.fsum(values_at_steps), 'all episode step costs')
        for key in ('init_seconds', 'sampling_seconds'):
            require(type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] >= 0, 'finite paid actor/sampling cost')
        close(row['controller_seconds'], math.fsum(row[k] for k in ('init_seconds', 'choose_seconds', 'update_seconds')), 'complete controller cost')
        self.collection_reset_seconds += reset_cost
        return row

    def compute(self):
        import numpy as np
        self.np = np
        E = self.E = load(ROOT/HELPER, '_coverage_independent_math')
        self.c = E.B.Comparisons()
        run = self.args.run
        contexts = list(lines(run/'work-contexts.jsonl', self.check))
        require([r['id'] for r in contexts] == list(range(len(contexts)))
                and all(set(r) == {'id', 'context'} and 'step' not in r['context'] for r in contexts)
                and len({json.dumps(r['context'], sort_keys=True) for r in contexts}) == len(contexts), 'unique sequential operation contexts')
        self.work = E.Work(iter(lines(run/'work.jsonl', self.check)), [r['context'] for r in contexts], self.c)
        self.kernels, self.heads = {}, {}
        setup = read(run/'native-setup.json')
        require(setup['resets'] == 2 and len(setup['checks']) == 2, 'exact two kernel template resets')
        setup_call_seconds = 0.
        for i, regime in enumerate(REGIMES):
            arrays = self.arrays(self.scalar_run/f'kernel-{regime}.npz')
            require(set(arrays) == {'likelihood', 'initial_hit_weights'}, 'original kernel payload')
            kernel = arrays['likelihood']
            require(kernel.shape == (4, 107, 107) and kernel.dtype == np.float64 and np.isfinite(kernel).all()
                    and (kernel >= 0).all() and (kernel <= 1).all() and (kernel[:, 53, 53] == 0).all(), 'exact finite public likelihood kernel')
            self.kernels[regime] = kernel
            record = setup['checks'][i]
            require(record == {'regime': regime, 'seed': 13500001+i, 'exact': True, 'public': record['public']}
                    and record['public'] == E.B.packet([26, 26], record['public']['hit'], False, 0)
                    and type(record['public']['hit']) is int and record['public']['hit'] in (1, 2, 3), 'template source-bound native parity witness')
            setup_call_seconds += self.work.call('native_reset', {'phase': 'native_setup', 'regime': regime})['seconds']
        inference = read(run/'inference-setup.json')
        equal(set(inference), {str(seed) for seed in SEEDS}, 'only three unchanged original collectors')
        baseline = None
        for seed in SEEDS:
            path = self.scalar_run/f'final-mlp8-{seed}.npz'
            record, archive = inference[str(seed)], self.arrays(path)
            c0 = float(archive['c0'])
            require(baseline is None or baseline == c0, 'same original c0 across collectors')
            baseline = c0
            self.heads[seed] = E.checkpoint(archive, 'mlp8', c0, np)
            desc = descriptor(path, self.check)
            require(record['path'] == str(path.relative_to(ROOT)) and {k: record[k] for k in desc} == desc
                    and desc == self.scalar_receipt['files'][path.name], 'original scalar checkpoint joins')
            equal({k: v for k, v in record['storage'].items() if k != 'scope'}, E.head_storage('mlp8'), 'stored float64 head arrays')
            operation = self.work.call('model_load', {'phase': 'collector_setup', 'collector_seed': seed})
            require(type(record['seconds']) in (int, float) and math.isfinite(record['seconds'])
                    and record['seconds'] >= operation['seconds'], 'paid load envelope')
            setup_call_seconds += record['seconds']
        runtime = read(run/'runtime.json')
        require(runtime['python'] == sys.version and runtime['executable'] == sys.executable
                and runtime['threads'] == {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                                           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
                and type(runtime['optimizer_calls']) is int and runtime['optimizer_calls'] == 0 and math.isfinite(runtime['setup_seconds'])
                and runtime['setup_seconds'] >= setup_call_seconds, 'unchanged CPU1 and paid setup')
        teacher_rows = list(lines(self.scalar_run/'train-rows.jsonl', self.check))
        quotas = quota_records(teacher_rows)
        equal(quotas, self.plan['quota_table'], 'prospectively frozen six-stratum quota table')
        teacher = self.arrays(self.scalar_run/'train-data.npz', ('features', 'beliefs', 'positions', 'sensing_length'))
        self.mixture = self.arrays(run/'mixture-data.npz')
        require(set(self.mixture) == {'features', 'beliefs', 'positions', 'sensing_length'}, 'unlabeled mixture array closure')
        for data in (teacher, self.mixture):
            for name, shape, dtype in (('features', (5589, 11028), np.float32), ('beliefs', (5589, 53, 53), np.float64),
                                       ('positions', (5589, 2), np.int64), ('sensing_length', (5589,), np.float64)):
                require(data[name].shape == shape and data[name].dtype == dtype and np.isfinite(data[name]).all(), 'exact finite source/mixture geometry')
            require((data['beliefs'] >= 0).all() and (data['positions'] >= 0).all() and (data['positions'] <= 52).all(), 'valid public state values')
        metadata = list(lines(run/'mixture-rows.jsonl', self.check))
        require(len(metadata) == 5589 and [r['mixture_index'] for r in metadata] == list(range(5589))
                and len({r['identity'] for r in metadata}) == 5589, 'complete unique ordered mixture rows')
        self.selected = {r['identity']: (i, r) for i, r in enumerate(metadata)}
        self.verified_selected = set()
        self.candidates = {(r, h, s): [] for r, h in CELLS for s in SEEDS}
        expected_ids, groups = [], []
        for q in quotas:
            cell = q['regime'], q['initial_hit']
            available = [f'teacher:{i}' for i, r in enumerate(teacher_rows) if (r['regime'], r['initial_hit']) == cell]
            selected = ranked_selection(available, q['teacher_rows'])
            expected_ids.extend(selected)
            groups.append(self.group('teacher', cell, None, available, selected))
            for identity in selected:
                index = int(identity.split(':')[1])
                row = teacher_rows[index]
                expected = {'source': 'teacher', 'source_row_index': index, 'collector_seed': None,
                            'episode_id': row['episode_id'], 'stage': 'train', 'seed': row['seed'], 'case': None,
                            'regime': row['regime'], 'initial_hit': row['initial_hit'], 'prefix_index': row['prefix_index'],
                            'public': row['public'], 'posterior': row['posterior']}
                require(row['stage'] == 'train' and not row['public']['done'], 'original teacher pre-action state only')
                self.selected_state(identity, teacher['beliefs'][index], expected)
                require(identity in self.selected and self.mixture['features'][self.selected[identity][0]].tobytes()
                        == teacher['features'][index].tobytes(), 'teacher feature bytes preserved unchanged')
        events = iter(lines(run/'collection-transitions.jsonl', self.check))
        episodes = iter(lines(run/'collection-episodes.jsonl', self.check))
        self.sources, self.uniforms, self.collection_reset_seconds = {}, {}, 0.
        rows = [self.episode(next(episodes), identity, events) for identity in collection_order()]
        for iterator in (events, episodes, self.work.iterator):
            require(next(iterator, None) is None, 'no extra trajectory or operation records')
        equal(self.work.used_contexts, set(range(len(contexts))), 'no unconsumed operation contexts')
        for q in quotas:
            cell = q['regime'], q['initial_hit']
            for seed in SEEDS:
                available = self.candidates[(*cell, seed)]
                selected = ranked_selection(available, q['collector_rows'][str(seed)])
                expected_ids.extend(selected)
                groups.append(self.group('student', cell, seed, available, selected))
        equal([r['identity'] for r in metadata], expected_ids, 'independent salted selection and exact final ordering')
        equal(self.verified_selected, set(expected_ids), 'every selected row independently reconstructed exactly once')
        counts = {(r, h): sum(m['regime'] == r and m['initial_hit'] == h for m in metadata) for r, h in CELLS}
        equal(counts, {(q['regime'], q['initial_hit']): q['original_rows'] for q in quotas}, 'preserved six original stratum counts')
        original = {row.tobytes() for row in teacher['features']}
        student = [row.tobytes() for row in self.mixture['features'][2799:]]
        support = {'student_rows': 2790, 'student_unique_feature_rows': len(set(student)),
                   'student_rows_also_in_full_teacher': sum(value in original for value in student),
                   'student_unique_features_not_in_full_teacher': len(set(student)-original),
                   'identity': 'exact cached float32 feature bytes; duplicate inputs retained'}
        selection = read(run/'selection.json')
        require(type(selection['preparation_seconds']) in (int, float) and math.isfinite(selection['preparation_seconds'])
                and selection['preparation_seconds'] >= 0, 'paid selection preparation')
        tree(selection, {'quotas': quotas, 'groups': groups, 'support': support,
             'row_order': 'teacher strata/hash order, then student strata/collector/hash order',
             'preparation_seconds': selection['preparation_seconds'], 'targets_generated': False}, 'complete deterministic selection')
        total_steps = sum(r['steps'] for r in rows)
        require(total_steps <= 157536 and self.readout_attempts == self.readout_returns == total_steps
                and self.readout_rows == total_steps*16 and len(self.sources) == 24, 'all bounded local collection replay calls')
        equal({k: v['returned'] for k, v in self.work.counts.items()},
              {'native_reset': 74, 'model_load': 3, 'value_forward': total_steps, 'native_step': total_steps}, 'exact collection channel counts')
        tree(self.worker['calls'], self.work.counts, 'all original attempted/returned operation times')
        summary = read(run/'summary.json')
        collection_seconds = summary['collection_seconds']
        paid = math.fsum(r['controller_seconds']+r['environment_seconds']+r['sampling_seconds'] for r in rows)+self.collection_reset_seconds
        require(type(collection_seconds) in (int, float) and math.isfinite(collection_seconds) and collection_seconds >= paid
                and runtime['setup_seconds']+collection_seconds+selection['preparation_seconds'] <= self.worker['wall_seconds'], 'disjoint physical phase enclosure')
        expected = {'version': 'otto-coverage-collection-v1', 'episodes': 72, 'paired_cases': 24, 'native_steps': total_steps,
                    'found': sum(r['found'] for r in rows), 'censored': sum(not r['found'] for r in rows), 'coverage_admitted': True,
                    'support': support, 'collection_seconds': collection_seconds,
                    'controller_seconds': math.fsum(r['controller_seconds'] for r in rows),
                    'environment_seconds': math.fsum(r['environment_seconds'] for r in rows),
                    'sampling_seconds': math.fsum(r['sampling_seconds'] for r in rows),
                    'setup_seconds': runtime['setup_seconds'], 'native_resets': 74, 'training_calls': 0, 'target_readouts': 0,
                    'learned_architecture_advantage_established': False,
                    'scope': 'Fresh TRAIN exposure only; collection outcomes are not an independent efficacy comparison.'}
        tree(summary, expected, 'complete collection metadata, costs and descriptive outcomes')
        return {**expected, 'audit_version': VERSION, 'agreement': True, 'scope': SCOPE,
                'quotas': quotas, 'groups': groups, 'selected_teacher_states': 2799, 'selected_student_states': 2790,
                'saved_checkpoint_readout_calls': self.readout_returns, 'saved_network_rows': self.readout_rows,
                'maximum_readout_difference': self.maximum_difference, 'work_checks': self.c.count,
                'maximum_work_difference': self.c.maximum_difference}

    @staticmethod
    def group(source, cell, collector, available, selected):
        return {'source': source, 'regime': cell[0], 'initial_hit': cell[1], 'collector_seed': collector,
                'available': len(available), 'selected': len(selected),
                'rows': [{'key': hashlib.sha256((SALT+'|'+identity).encode()).hexdigest(), 'identity': identity} for identity in selected]}

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists()
                and not any(p.is_symlink() for p in self.out.parents), 'exclusive audit output')
        self.out.mkdir(parents=True, exist_ok=False)
        try:
            require(descriptor(ROOT/CLOCK)['sha256'] == CLOCK_PIN, 'qualified native clock')
            self.clock = load(ROOT/CLOCK, '_coverage_audit_clock').SuspendClock()
            self.start = self.clock.now_ns()
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('saved audit wall cap')))
            signal.setitimer(signal.ITIMER_REAL, LIMITS['seconds'])
            write(self.out/'started.json', {'version': VERSION, 'source': descriptor(Path(__file__)), 'limits': LIMITS,
                  'request': {k: str(v) for k, v in vars(self.args).items()}, 'clock_backend': self.clock.backend,
                  'started_ns': self.start, 'scope': SCOPE})
            self.authenticate()
            summary = self.compute()
            self.authenticate()
            write(self.out/'summary.json', summary)
            self.check()
            finished = self.clock.now_ns()
            require(finished-self.start < LIMITS['seconds']*10**9, 'strict final native limit')
            receipt = {'version': VERSION, 'status': 'completed', 'agreement': True, 'source': descriptor(Path(__file__), self.check),
                'plan_sha256': self.args.plan_sha256, 'worker_sha256': self.args.receipt_sha256,
                'terminal_sha256': self.args.terminal_sha256, 'producer_source_sha256': self.plan['sources'][RUNNER],
                'limits': LIMITS, 'clock_backend': self.clock.backend, 'started_ns': self.start, 'finished_ns': finished,
                'elapsed_ns': finished-self.start, 'wall_seconds': (finished-self.start)/1e9, 'peak_rss_bytes': self.rss,
                'saved_checkpoint_readout_calls': self.readout_returns, 'saved_network_rows': self.readout_rows,
                'scope': SCOPE, 'files': {name: descriptor(self.out/name, self.check) for name in ('started.json', 'summary.json')}}
            write(self.out/'receipt.json', receipt)
            self.check()
            signal.setitimer(signal.ITIMER_REAL, 0)
            return receipt
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                if (self.out/'receipt.json').exists():
                    (self.out/'receipt.json').rename(self.out/'invalid-completed-receipt.json')
                write(self.out/'failed.json', {'version': VERSION, 'status': 'failed', 'error': repr(error), 'scope': SCOPE,
                    'saved_readout_attempts': self.readout_attempts, 'saved_readout_returns': self.readout_returns,
                    'saved_network_rows': self.readout_rows, 'started_ns': self.start,
                    'finished_ns': None, 'elapsed_ns': None, 'wall_seconds': None})
            except BaseException as secondary:  # noqa: BLE001 - Preserve original failure.
                error.add_note(f'Failure publication: {secondary!r}')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'run', 'terminal', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    for flag in ('plan-sha256', 'receipt-sha256', 'terminal-sha256'):
        parser.add_argument('--'+flag, required=True)
    Audit(parser.parse_args()).execute()
