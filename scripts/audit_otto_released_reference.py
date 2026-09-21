"""Independent saved-output accounting for the complete released OTTO comparison.

Only the source-bound producer.authenticate is reused, for read-only lineage.
No model, policy, actor or simulator is imported or executed. Public posterior
arithmetic, recorded value-to-cost consistency, grouping and rules are separate.
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
VERSION = 'otto-released-reference-saved-audit-v1'
RUNNER = 'scripts/study_otto_released_reference.py'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
ARMS = ('released_tf', 'analytic_all4', 'analytic_inbounds')
REGIMES = {'base': 850001, 'shift': 860001}
HORIZON = 2188
LIMITS = {'native_seconds': 300, 'rss_bytes': 4 * 1024**3, 'output_bytes': 128 * 1024**2}
PAYLOADS = {'started.json', 'runtime.json', 'setup.json', 'weights.jsonl', 'work.jsonl',
            'forwards.jsonl', 'transitions.jsonl', 'episodes.jsonl', 'summary.json'}
CHANNELS = ('native_reset', 'native_step', 'actor_construction', 'actor_choose', 'actor_update',
            'tensorflow_construction', 'tensorflow_build', 'tensorflow_load', 'tensorflow_value')
TIMES = ('actor_initialization_seconds', 'choose_seconds', 'update_seconds', 'model_setup_allocation_seconds')
METRICS = ('capped_time', 'found', 'stuck_steps', 'blocked_steps', *TIMES, 'controller_seconds',
           'model_forward_seconds', 'environment_initialization_seconds', 'environment_seconds',
           'episode_seconds', 'state_array_bytes')
SCOPE = ('Independent complete membership, public posterior reconstruction, stored-score choices, '
         'recorded neural branch masses/value-to-cost arithmetic, paired initialization, work/cost sums '
         'and all declared aggregate conditions. No independent neural values or analytic policy scores, '
         'RNG generation, runtime/model execution or timing truth; these remain authenticated producer evidence. '
         'Recorded source/hit draw identities and paired uniforms are checked, not regenerated. '
         'No model/simulator calls, training, new efficacy metric or revision of prior failed gates.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def digest(path, check=lambda: None):
    value, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            check()
            size += len(block)
            value.update(block)
    return {'sha256': value.hexdigest(), 'bytes': size}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def lines(path, check=lambda: None):
    with path.open() as stream:
        for line in stream:
            check()
            yield json.loads(line)


def exhausted(iterator, label):
    require(next(iterator, None) is None, f'no extra {label}')


def identities():
    for regime_index, (regime, seed) in enumerate(REGIMES.items()):
        for case in range(96):
            offset = (96 * regime_index + case) % 3
            for arm in ARMS[offset:] + ARMS[:offset]:
                yield {'cohort': regime, 'seed': seed + case, 'block': case // 12,
                       'initial_hit': 1 + (case % 12) // 4, 'arm': arm}


def move(position, action):
    require(type(action) is int and 0 <= action < 4, 'action ID')
    result = list(position)
    axis = action // 2
    result[axis] = max(0, min(52, result[axis] + (-1 if action % 2 == 0 else 1)))
    return result


def packet(position, hit, done, step):
    return {'position': list(position), 'hit': hit, 'done': done, 'step': step,
            'valid_actions': [] if done else [a for a in range(4) if move(position, a) != list(position)]}


def stuck_counter(repeated, two_ago, target):
    repeated = repeated + 1 if target == two_ago else 0
    return repeated, repeated > 8


def posterior(probability, public, kernel, np):
    result = probability.copy()
    x, y = public['position']
    if public['done']:
        result.fill(0)
        result[x, y] = 1.
    else:
        result[x, y] = 0.
        result *= kernel[public['hit'], 53-x:106-x, 53-y:106-y]
        result[(result < 0) & (result > -1e-15)] = 0.
        mass = result.sum()
        require(np.isfinite(result).all() and (result >= 0).all() and math.isfinite(mass), 'valid public posterior')
        if mass > 1e-10:
            result /= mass
    return result


def witness(probability):
    return {'exact': True, 'mass': float(probability.sum()),
            'sha256': hashlib.sha256(probability.tobytes()).hexdigest()}


def branch_masses(probability, position, kernel, np):
    result = []
    for action in range(4):
        x, y = move(position, action)
        joint = probability[None] * kernel[:, 53-x:106-x, 53-y:106-y]
        result.append([max(1e-10, float(joint[h].sum())) for h in range(4)])
    return np.asarray(result, dtype=np.float32)


def selected_action(costs, allowed, np, neural=False):
    require(isinstance(costs, list) and len(costs) == 4, 'four recorded costs')
    require(all((type(costs[a]) in (int, float) and math.isfinite(costs[a])) if a in allowed
                else costs[a] is None for a in range(4)), 'exact null blocked-cost convention')
    # Preserve the original float32 subtraction in the released policy's tie test.
    values = np.asarray([math.inf if x is None else x for x in costs], dtype=np.float32 if neural else np.float64)
    if neural:
        require(values.tolist() == costs, 'recorded neural costs are exact float32 values')
    return int(np.flatnonzero(np.abs(values - values.min()) < 1e-10)[0])


def neural_costs(record, costs, masses, np):
    actual = np.asarray(record['branch_masses'], dtype=np.float32)
    require(actual.shape == (4, 4) and actual.tolist() == record['branch_masses']
            and np.array_equal(actual, masses), 'independent public branch mass identity')
    values = np.asarray(record['values'], dtype=np.float32)
    require(values.shape == (16,) and values.tolist() == record['values'] and np.isfinite(values).all(), 'sixteen saved float32 neural values')
    errors = []
    for a in range(4):
        terms = [float(actual[a, h]) * float(values[4*a+h]) for h in range(4)]
        expected = 1. + math.fsum(terms)
        # Four f32 products, a reduction and +1. This is an arithmetic bound,
        # never a relaxed action/tie gate and never model-value regeneration.
        bound = 8 * 2**-23 * (1. + math.fsum(abs(t) for t in terms))
        errors.append(abs(float(costs[a]) - expected))
        require(errors[-1] <= bound, 'saved TF cost outside float32 arithmetic bound')
    return max(errors)


class Comparisons:
    def __init__(self):
        self.count, self.maximum_difference = 0, 0.

    def equal(self, actual, expected, name):
        self.count += 1
        require(actual == expected, name)

    def close(self, actual, expected, name):
        self.count += 1
        require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected), name)
        difference = abs(actual - expected)
        self.maximum_difference = max(self.maximum_difference, difference)
        require(difference <= 1e-9 + 2e-12 * abs(expected), name)

    def tree(self, actual, expected, name='summary'):
        if isinstance(expected, dict):
            self.equal(set(actual), set(expected), name + ' keys')
            for key in expected:
                self.tree(actual[key], expected[key], name + '/' + str(key))
        elif isinstance(expected, list):
            self.equal(len(actual), len(expected), name + ' length')
            for a, e in zip(actual, expected, strict=True):
                self.tree(a, e, name)
        elif type(expected) is float:
            self.close(actual, expected, name)
        else:
            self.equal(actual, expected, name)


class Journal:
    def __init__(self, iterator, compare):
        self.iterator, self.c = iterator, compare
        self.sequence = 0
        self.counts = {ch: {'attempted': 0, 'returned': 0, 'seconds': 0.,
                           'instrumented_seconds': 0., 'excluded_io_seconds': 0.} for ch in CHANNELS}
        self.last_context = None

    def call(self, channel, context, nested=False, parent=None):
        self.sequence += 1
        counts = self.counts[channel]
        counts['attempted'] += 1
        identity = {'call_id': self.sequence, 'channel': channel, 'ordinal': counts['attempted'],
                    'parent_call_id': parent, 'context': context}
        self.c.equal(next(self.iterator), {'event': 'attempt', **identity}, 'durable ordered call attempt')
        child = self.call('tensorflow_value', context, parent=self.sequence)[0] if nested else None
        returned = next(self.iterator)
        self.c.equal(set(returned), {'event', *identity, 'seconds', 'instrumented_seconds', 'excluded_io_seconds'}, 'return fields')
        self.c.equal({k: returned[k] for k in ('event', *identity)}, {'event': 'return', **identity}, 'matching call return')
        for key in ('seconds', 'instrumented_seconds', 'excluded_io_seconds'):
            require(type(returned[key]) in (int, float) and math.isfinite(returned[key]) and returned[key] >= 0, 'nonnegative recorded call duration')
            counts[key] += returned[key]
        self.c.close(returned['seconds'], returned['instrumented_seconds'] - returned['excluded_io_seconds'], 'net call cost')
        if child:
            require(child['instrumented_seconds'] <= returned['instrumented_seconds'] + 1e-9, 'nested forward time enclosure')
        counts['returned'] += 1
        self.last_context = context
        return returned, child


def aggregate(rows, weights):
    """Independent conditional means and frozen float64 threshold decisions."""
    require([{k: r[k] for k in ('cohort', 'seed', 'block', 'initial_hit', 'arm')} for r in rows] == list(identities()), 'all576 ordered episodes')
    cohorts, paired = {}, []
    for regime, first_seed in REGIMES.items():
        w = weights[regime]
        require(set(w) == {1, 2, 3} and all(0 < x < 1 and math.isfinite(x) for x in w.values())
                and abs(math.fsum(w.values()) - 1) <= 1e-12, 'qualified initial-hit mixture')
        selected = [r for r in rows if r['cohort'] == regime]

        def group(records, w=w):
            result = {}
            for arm in ARMS:
                strata = {h: [r for r in records if r['arm'] == arm and r['initial_hit'] == h] for h in (1, 2, 3)}
                require(all(strata.values()), 'complete conditional support')
                result[arm] = {m: math.fsum(w[h] * math.fsum(float(r[m]) for r in strata[h]) / len(strata[h])
                                          for h in (1, 2, 3)) for m in METRICS}
            return result

        means = group(selected)
        strata = {str(h): {a: {m: math.fsum(float(r[m]) for r in selected if r['initial_hit'] == h and r['arm'] == a) / 32
                               for m in METRICS} for a in ARMS} for h in (1, 2, 3)}
        blocks = [{'block': b, 'means': group([r for r in selected if r['block'] == b])} for b in range(8)]
        released = means[ARMS[0]]
        competence = [{'name': 'released_success_at_least_95pct', 'passes': released['found'] >= .95}]
        teacher, pareto = [], []
        for control in ARMS[1:]:
            other = means[control]
            competence.append({'name': f'moves_at_most_105pct_{control}', 'passes': released['capped_time'] <= 1.05 * other['capped_time']})
            positives = sum(b['means'][control]['capped_time'] > b['means'][ARMS[0]]['capped_time'] for b in blocks)
            teacher.extend([{'name': f'no_success_loss_vs_{control}', 'passes': released['found'] >= other['found']},
                            {'name': f'moves_at_most_95pct_{control}', 'passes': released['capped_time'] <= .95 * other['capped_time']},
                            {'name': f'positive_blocks_vs_{control}', 'value': positives, 'threshold': 6, 'passes': positives >= 6}])
            pareto.extend([{'name': f'no_success_loss_vs_{control}', 'passes': released['found'] >= other['found']},
                           {'name': f'no_more_moves_vs_{control}', 'passes': released['capped_time'] <= other['capped_time']},
                           {'name': f'no_more_controller_cost_vs_{control}', 'passes': released['controller_seconds'] <= other['controller_seconds']},
                           {'name': f'strict_move_or_cost_gain_vs_{control}', 'passes': released['capped_time'] < other['capped_time'] or released['controller_seconds'] < other['controller_seconds']}])
        cohorts[regime] = {'initial_hit_weights': {str(h): w[h] for h in w}, 'means': means, 'strata': strata, 'blocks': blocks,
                           'competence_checks': competence, 'stronger_teacher_checks': teacher, 'utility_compute_checks': pareto,
                           'unweighted_counts': {a: {'episodes': 96, 'found': sum(r['found'] for r in selected if r['arm'] == a),
                               'censored': sum(not r['found'] for r in selected if r['arm'] == a),
                               'total_steps': sum(r['steps'] for r in selected if r['arm'] == a)} for a in ARMS}}
        for case in range(96):
            records = {r['arm']: r for r in selected if r['seed'] == first_seed + case}
            paired.append({'cohort': regime, 'case': case, 'seed': first_seed + case,
                           'initial_hit': 1 + (case % 12) // 4, 'block': case // 12,
                           'arms': {a: {'found': records[a]['found'], 'moves': records[a]['steps'], 'controller_seconds': records[a]['controller_seconds']} for a in ARMS},
                           'released_minus_control': {a: {m: float(records[ARMS[0]][m]) - float(records[a][m]) for m in ('found', 'capped_time', 'controller_seconds')} for a in ARMS[1:]}})
    return {'cohorts': cohorts, 'competent_reference': all(c['passes'] for r in cohorts.values() for c in r['competence_checks']),
            'stronger_value_teacher': all(c['passes'] for r in cohorts.values() for c in r['stronger_teacher_checks']),
            'utility_compute_advantage': all(c['passes'] for r in cohorts.values() for c in r['utility_compute_checks'])}, paired


def verify_episode(row, context, transitions, forwards, journal, kernel, setup, np, c, paired_uniforms):
    for key in context:
        c.equal(row[key], context[key], 'episode identity')
    require(type(row['steps']) is int and 1 <= row['steps'] <= HORIZON and type(row['found']) is bool
            and (row['found'] or row['steps'] == HORIZON), 'complete found/censored episode')
    require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0 for k in METRICS if k != 'found'), 'finite episode metrics')
    source = row['source_evaluation_only']
    require(len(source) == 2 and all(type(x) is int and 0 <= x < 53 for x in source) and source != [26, 26], 'sampled evaluator source')
    reset_context = {'phase': 'episode_reset', **context}
    reset, _ = journal.call('native_reset', reset_context)
    construction, _ = journal.call('actor_construction', reset_context)
    current = packet((26, 26), context['initial_hit'], False, 0)
    c.equal(row['initial_public'], current, 'paired public reset')
    probability = np.ones((53, 53), dtype=np.float64) / 2808
    probability = posterior(probability, current, kernel, np)
    state = witness(probability)
    c.equal(next(transitions), {'kind': 'reset', **context, 'public': current, 'posterior_after': state,
                               'source_evaluation_only': source}, 'recorded reset/public belief')
    draws = iter(row['draws_evaluation_only'])
    first = next(draws)
    c.equal((first['channel'], first['index'], first['selected_index']), ('source', 0, source[0] * 53 + source[1]), 'conditioned source draw identity')
    costs = {'choose_seconds': [], 'update_seconds': [], 'environment_seconds': [], 'model_forward_seconds': []}
    raw = {k: [] for k in ('choose_seconds', 'update_seconds')}
    blocked, stuck, max_cost_error, hits = 0, 0, 0., 0
    two_ago, repeated = [0, 0], 0  # Pinned upstream reset's _agento before first action.
    for step in range(1, row['steps'] + 1):
        decision = {'phase': 'decision', **context, 'step': step}
        neural = context['arm'] == 'released_tf'
        choose, model = journal.call('actor_choose', decision, nested=neural)
        native, _ = journal.call('native_step', decision)
        update, _ = journal.call('actor_update', decision)
        record = next(transitions)
        for key, value in {'kind': 'step', **context, 'step': step}.items():
            c.equal(record[key], value, 'ordered public transition')
        allowed = list(range(4)) if context['arm'] != 'analytic_inbounds' else current['valid_actions']
        c.equal(record['allowed_actions'], allowed, 'declared action eligibility')
        action = selected_action(record['costs'], allowed, np, neural)
        c.equal(record['action'], action, 'exact first near-tie choice')
        if neural:
            forward = next(forwards)
            c.equal(forward['context'], decision, 'forward decision identity')
            c.equal(forward['ordinal'], journal.counts['tensorflow_value']['returned'], 'forward ordinal')
            c.equal((forward['symmetry_average'], forward['input_shape']), (True, [16, 105, 105]), 'forward shape/symmetry witness')
            c.close(forward['seconds'], model['seconds'], 'forward journal cost')
            max_cost_error = max(max_cost_error, neural_costs(forward, record['costs'], branch_masses(probability, current['position'], kernel, np), np))
        target = move(current['position'], action)
        done = target == source
        require(not done or step == row['steps'], 'first found event ends episode')
        hit = record['public']['hit']
        require(type(hit) is int and (hit == -2 if done else 0 <= hit < 4), 'hit/sentinel semantics')
        public = packet(target, hit, done, step)
        c.equal(record['public'], public, 'native public geometry')
        c.equal(record['native_p_end'], float(done), 'sampled-source native end value')
        c.equal(record['posterior_before'], state, 'causal prior posterior')
        probability = posterior(probability, public, kernel, np)
        state = witness(probability)
        c.equal(record['posterior_after'], state, 'independent completed public posterior')
        is_blocked = target == current['position']
        c.equal(record['blocked'], is_blocked, 'blocked stay')
        repeated, is_stuck = stuck_counter(repeated, two_ago, target)
        c.equal(record['stuck'], is_stuck, 'upstream consecutive two-step-return threshold')
        blocked += is_blocked
        stuck += record['stuck']
        if not done:
            draw = next(draws)
            c.equal((draw['channel'], draw['index'], draw['selected_index']), ('hit', hits, hit), 'observed categorical draw')
            hits += 1
        for key, expected in {'choose_seconds': choose['seconds'], 'choose_instrumented_seconds': choose['instrumented_seconds'],
                              'choose_excluded_io_seconds': choose['excluded_io_seconds'], 'update_seconds': update['seconds'],
                              'update_instrumented_seconds': update['instrumented_seconds'], 'environment_seconds': native['seconds'],
                              'model_forward_seconds': model['seconds'] if neural else 0., 'model_forward_calls': int(neural)}.items():
            c.close(record[key], expected, 'transition recorded work cost')
        for key, values in costs.items():
            values.append(record[key])
        raw['choose_seconds'].append(choose['instrumented_seconds'])
        raw['update_seconds'].append(update['instrumented_seconds'])
        two_ago, current = current['position'], public
    exhausted(draws, 'episode draws')
    for draw in row['draws_evaluation_only']:
        require(type(draw['uniform']) in (int, float) and 0 <= draw['uniform'] < 1
                and math.isfinite(draw['cdf_mass']) and abs(draw['cdf_mass'] - 1) < 1e-10, 'recorded categorical draw range')
        key = (context['cohort'], context['seed'], draw['channel'], draw['index'])
        if key in paired_uniforms:
            c.equal(draw['uniform'], paired_uniforms[key], 'paired stream uniform')
        else:
            paired_uniforms[key] = draw['uniform']
    for key, expected in {'found': current['done'], 'capped_time': row['steps'], 'choose_calls': row['steps'],
                          'update_calls': row['steps'], 'model_forward_calls': row['steps'] if context['arm'] == 'released_tf' else 0,
                          'blocked_steps': blocked, 'stuck_steps': stuck, 'final_update_assimilated': True}.items():
        c.equal(row[key], expected, 'complete episode totals')
    times = {k: math.fsum(v) for k, v in costs.items()}
    times.update(actor_initialization_seconds=construction['seconds'], environment_initialization_seconds=reset['seconds'],
                 model_setup_allocation_seconds=setup['model_setup_seconds'] / 192 if context['arm'] == 'released_tf' else 0.)
    for key, value in times.items():
        c.close(row[key], value, 'episode complete net cost')
    instrumented = {'actor_initialization_seconds': construction['instrumented_seconds'],
                    **{k: math.fsum(v) for k, v in raw.items()},
                    'model_setup_allocation_seconds': setup['model_setup_instrumented_seconds'] / 192 if context['arm'] == 'released_tf' else 0.}
    c.tree(row['instrumented_component_seconds'], instrumented, 'instrumented episode components')
    c.close(row['controller_seconds'], math.fsum(times[k] for k in TIMES), 'full controller cost')
    c.close(row['controller_instrumented_seconds'], math.fsum(instrumented.values()), 'full instrumented controller cost')
    c.close(row['controller_excluded_io_seconds'], row['controller_instrumented_seconds'] - row['controller_seconds'], 'excluded nested I/O')
    require(row['episode_seconds'] >= construction['instrumented_seconds'] + reset['instrumented_seconds']
            + math.fsum(raw['choose_seconds']) + math.fsum(raw['update_seconds']) + times['environment_seconds'] - 1e-8, 'episode interval enclosure')
    c.equal(row['state_array_bytes'], 53 * 53 * 8, 'owned mutable posterior size')
    c.equal(row['storage']['mutable_array_bytes'], row['state_array_bytes'], 'state storage join')
    c.equal(row['storage']['immutable_array_bytes'], (4 + int(context['arm'] != 'released_tf')) * 107 * 107 * 8, 'kernel and cached geometry storage')
    return max_cost_error


class Audit:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = None
        self.c = Comparisons()
        self.receipt = {'version': VERSION, 'status': 'started', 'scope': SCOPE, 'limits': LIMITS,
                        'model_calls': 0, 'simulator_calls': 0, 'policy_calls': 0}

    def check(self):
        require(self.clock.now_ns() - self.start < LIMITS['native_seconds'] * 10**9, 'audit native deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= LIMITS['rss_bytes'], 'audit RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'audit output cap')

    def authenticate(self):
        a, c = self.args, self.c
        for path in (a.plan, a.run, a.terminal, a.output):
            require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'absolute nonsymlink paths')
        for path, pin in ((a.plan, a.plan_sha256), (a.run / 'receipt.json', a.receipt_sha256), (a.terminal, a.terminal_sha256)):
            c.equal(digest(path, self.check)['sha256'], pin, 'external input identity')
        plan, worker, terminal = read(a.plan), read(a.run / 'receipt.json'), read(a.terminal)
        for name in (RUNNER, 'scripts/audit_otto_released_reference.py', 'tests/test_audit_otto_released_reference.py'):
            c.equal(digest(ROOT / name, self.check)['sha256'], plan['sources'][name], 'pre-import source identity')
        auth = load(ROOT / RUNNER, '_released_reference_audit_auth_only')
        validated, paths = auth.authenticate(a)
        c.equal(validated, plan, 'read-only producer authentication')
        require(worker['status'] == 'completed' and worker['version'] == 'otto-released-reference-v1', 'complete reference run before evidence decoding')
        for key in ('sources', 'inputs', 'limits'):
            c.equal(worker[key], plan[key], 'worker plan join')
        c.equal(worker['plan_sha256'], a.plan_sha256, 'worker plan pin')
        c.equal(set(worker['files']), PAYLOADS, 'exact complete payload names')
        c.equal({p.name for p in a.run.iterdir()}, PAYLOADS | {'receipt.json'}, 'no missing/late/extra evidence')
        for name, expected in worker['files'].items():
            p = a.run / name
            require(p.is_file() and not p.is_symlink(), 'regular payload')
            c.equal(digest(p, self.check), expected, 'all payload hashes before decode')
        require(worker['completed_episodes'] == 576 and worker['training_updates'] == worker['external_model_calls'] == 0
                and worker['numpy_port_admitted'] is False and worker['prior_numpy_action_failure_preserved'] is True, 'fixed complete nontraining scope')
        require(0 < worker['peak_rss_bytes'] <= plan['limits']['rss_bytes']
                and sum(p.stat().st_size for p in a.run.iterdir()) <= plan['limits']['output_bytes'], 'producer resource evidence')
        started = read(a.run / 'started.json')
        request, launch = started['request'], started['launch']
        c.equal(request, {'plan': str(a.plan), 'plan_sha256': a.plan_sha256, 'supervision': request['supervision'], 'output': str(a.run)}, 'original invocation')
        launch_path = Path(request['supervision'])
        c.equal(digest(launch_path, self.check)['sha256'], worker['supervision_sha256'], 'external launch binding')
        c.equal(read(launch_path), launch, 'stored launch identity')
        require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
                and terminal['group_absent'] is True and terminal['cleanup']['group_absent'] is True
                and terminal['cleanup']['reaped'] is True and terminal['cleanup']['errors'] == []
                and terminal['clock_error'] is None and terminal['error'] is None and terminal['timing_available'] is True, 'successful absent parent process')
        for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend',
                    'watchdog_sha256', 'clock_source_sha256', 'cap_seconds'):
            c.equal(terminal[key], launch[key], 'parent common field identity')
        command = list(launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        c.equal(command[:2], [plan['runtime']['python_executable'], str(ROOT / RUNNER)], 'actual interpreter/runner')
        require(len(command) == 10, 'four exact CLI bindings')
        c.equal(dict(zip(command[2::2], command[3::2], strict=True)), {f'--{k.replace("_", "-")}': v for k, v in request.items()}, 'actual argument identity')
        require(launch['cap_seconds'] == 1800 and launch['pid'] == launch['pgid'] and launch['parent_pid'] != launch['pid']
                and launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
                and worker['clock_backend'] == launch['clock_backend']
                and launch['deadline_ns'] == launch['started_ns'] + 1800 * 10**9
                and launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns'], 'strict shared native intervals')
        c.equal(started['started_ns'], worker['started_ns'], 'worker start identity')
        c.equal(started['clock_backend'], worker['clock_backend'], 'worker start clock identity')
        c.equal(terminal['elapsed_ns'], terminal['finished_ns'] - terminal['started_ns'], 'parent elapsed')
        c.equal(terminal['wall_seconds'], terminal['elapsed_ns'] / 1e9, 'parent seconds')
        c.equal(worker['wall_seconds'], (worker['finished_ns'] - worker['started_ns']) / 1e9, 'worker seconds')
        c.equal(launch['cwd'], str(ROOT), 'actual cwd')
        c.equal(launch['watchdog_sha256'], plan['sources']['scripts/supervise_dialogue_observation_v2.py'], 'watchdog bytes')
        c.equal(launch['clock_source_sha256'], CLOCK_PIN, 'clock bytes')
        runtime = read(a.run / 'runtime.json')
        c.equal(runtime['executable'], plan['runtime']['python_executable'], 'recorded runtime interpreter')
        c.equal(runtime['all_distributions'], plan['runtime']['all_distributions'], 'recorded runtime closure')
        c.equal(runtime['keras_configuration'], {'floatx': 'float32', 'image_data_format': 'channels_last', 'mixed_precision_policy': 'float32'}, 'recorded Keras mode')
        require(runtime['keras_module'].startswith('tf_keras.') and runtime['visible_devices']
                and all('GPU' not in v for v in runtime['visible_devices']), 'recorded CPU legacy Keras')
        self.receipt.update(plan_sha256=a.plan_sha256, worker_sha256=a.receipt_sha256, terminal_sha256=a.terminal_sha256,
                            producer_source_sha256=plan['sources'][RUNNER], authentication_reuse='Source-bound producer.authenticate only.')
        return plan, paths, worker

    def compute(self, paths, worker):
        import numpy as np

        c, run = self.c, self.args.run
        kernels, weights = {}, {}
        for regime in REGIMES:
            with np.load(paths[f'{regime}_kernel'], allow_pickle=False) as archive:
                c.equal(set(archive.files), {'likelihood', 'initial_hit_weights'}, 'qualified kernel keys')
                k, w = archive['likelihood'], archive['initial_hit_weights']
                require(k.shape == (4, 107, 107) and k.dtype == np.float64 and np.isfinite(k).all()
                        and (k >= 0).all() and (k <= 1).all() and not k[:, 53, 53].any(), 'qualified likelihood geometry')
                require(w.shape == (4,) and w[0] == 0, 'mixture shape')
                kernels[regime], weights[regime] = k, {h: float(w[h]) for h in (1, 2, 3)}
        setup, published = read(run / 'setup.json'), read(run / 'summary.json')
        c.close(setup['model_setup_seconds'], setup['model_setup_instrumented_seconds'] - setup['model_setup_excluded_io_seconds'], 'setup net of I/O')
        require(setup['model_setup_seconds'] >= 0 and setup['model_setup_excluded_io_seconds'] >= 0
                and setup['allocated_over_released_episodes'] == 192 and setup['model_parameters'] == 13390849
                and setup['model_tensor_bytes'] == 53563396, 'setup/storage scope')
        metadata = read(paths['tensor_metadata'])
        weight_rows = list(lines(run / 'weights.jsonl', self.check))
        expected_weights = []
        for i in range(4):
            for kind in ('kernel', 'bias'):
                key = f'{kind}_{i}'
                record = metadata['datasets'][key]
                expected_weights.append({'id': key, 'shape': record['shape'], 'sha256': record['sha256_c_order'], 'exact': True})
        c.equal(weight_rows, expected_weights, 'ordered loaded-weight identity witnesses')
        journal = Journal(iter(lines(run / 'work.jsonl', self.check)), c)
        for name in ('tensorflow_construction', 'tensorflow_build', 'tensorflow_load'):
            journal.call(name, {'phase': 'model_setup'})
        require(math.fsum(journal.counts[n]['instrumented_seconds'] for n in ('tensorflow_construction', 'tensorflow_build', 'tensorflow_load')) <= setup['model_setup_instrumented_seconds'] + 1e-9, 'nested setup enclosure')
        transitions, forwards = iter(lines(run / 'transitions.jsonl', self.check)), iter(lines(run / 'forwards.jsonl', self.check))
        episodes = iter(lines(run / 'episodes.jsonl', self.check))
        rows, pairs, uniforms, max_cost_error = [], {}, {}, 0.
        for identity in identities():
            row = next(episodes)
            max_cost_error = max(max_cost_error, verify_episode(row, identity, transitions, forwards, journal,
                kernels[identity['cohort']], setup, np, c, uniforms))
            pair = (identity['cohort'], identity['seed'])
            initial = (row['source_evaluation_only'], row['initial_public'])
            if pair in pairs:
                c.equal(initial, pairs[pair], 'paired source/public state')
            else:
                pairs[pair] = initial
            # Drop potentially long draw records after checks, keep all metric rows.
            rows.append({k: v for k, v in row.items() if k != 'draws_evaluation_only'})
        for iterator, name in ((episodes, 'episodes'), (transitions, 'transitions'), (forwards, 'forwards'), (journal.iterator, 'work')):
            exhausted(iterator, name)
        c.tree(worker['work']['calls'], journal.counts, 'complete work ledger')
        c.equal(worker['work']['pending'], [], 'no unresolved calls')
        c.equal(worker['work']['sequence'], journal.sequence, 'total call sequence')
        c.equal(worker['work']['context'], journal.last_context, 'last work context')
        expected_work = {ch: journal.counts[ch]['returned'] for ch in CHANNELS}
        c.equal(expected_work, {'native_reset': 576, 'native_step': sum(r['steps'] for r in rows),
                'actor_construction': 576, 'actor_choose': sum(r['steps'] for r in rows), 'actor_update': sum(r['steps'] for r in rows),
                'tensorflow_construction': 1, 'tensorflow_build': 1, 'tensorflow_load': 1,
                'tensorflow_value': sum(r['steps'] for r in rows if r['arm'] == 'released_tf')}, 'complete expected work')
        derived, paired = aggregate(rows, weights)
        for key, value in derived.items():
            c.tree(published[key], value, 'independent aggregates/' + key)
        c.equal(published['episodes'], 576, 'summary count')
        c.equal(published['paired_source_cases'], len(pairs), 'paired case count')
        c.equal(published['paired_uniform_checks'], len(uniforms), 'paired draw count')
        c.equal(published['actual_work'], expected_work, 'summary work')
        c.equal(published['all_public_beliefs_exact'], True, 'complete public belief reconstruction')
        c.tree(published['setup'], setup, 'published setup')
        for name in ('learned_pilot_admission', 'inherited_gate_revised', 'numpy_port_admitted'):
            c.equal(published[name], False, 'unchanged study scope')
        for name in ('competent_reference', 'stronger_value_teacher', 'utility_compute_advantage'):
            c.equal(worker[name], derived[name], 'receipt scientific result')
        for field, key in (('sum_controller_seconds', 'controller_seconds'), ('sum_controller_instrumented_seconds', 'controller_instrumented_seconds'), ('sum_excluded_controller_io_seconds', 'controller_excluded_io_seconds')):
            c.close(published[field], math.fsum(r[key] for r in rows), 'complete cost arithmetic')
        require(published['sum_excluded_controller_io_seconds'] <= worker['work']['artifact_io_seconds'] + 1e-8
                and published['artifact_io_seconds'] <= worker['work']['artifact_io_seconds'] + 1e-8, 'I/O exclusion within recorded physical writing')
        disjoint = published['sum_controller_seconds'] + math.fsum(r['environment_seconds'] + r['environment_initialization_seconds'] for r in rows)
        require(disjoint <= worker['wall_seconds'] + 1e-8, 'disjoint computation inside worker duration')
        return {'version': VERSION, 'agreement': True, **derived, 'paired_cases': paired, 'episodes': 576,
                'public_updates': expected_work['actor_update'], 'public_resets': 576, 'actual_work': expected_work,
                'maximum_independent_neural_cost_difference': max_cost_error, 'comparisons': c.count,
                'maximum_aggregate_or_timing_difference': c.maximum_difference,
                'condition_counts': {'competence': 6, 'promising_teacher_candidate': 12, 'utility_compute': 16},
                'scope': SCOPE}

    def execute(self):
        self.out.mkdir(parents=True, exist_ok=False)
        old_alarm = signal.getsignal(signal.SIGALRM)
        try:
            require(digest(ROOT / CLOCK)['sha256'] == CLOCK_PIN, 'qualified native clock')
            self.clock = load(ROOT / CLOCK, '_released_reference_audit_clock').SuspendClock()
            self.start = self.clock.now_ns()
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('audit emergency wall cap')))
            signal.setitimer(signal.ITIMER_REAL, LIMITS['native_seconds'])
            self.receipt['source'] = digest(Path(__file__), self.check)
            write(self.out / 'started.json', {'request': {k: str(v) for k, v in vars(self.args).items()},
                                              'limits': LIMITS, 'clock_backend': self.clock.backend, 'started_ns': self.start})
            _, paths, worker = self.authenticate()
            result = self.compute(paths, worker)
            self.authenticate()  # Source, input, manifest and actual process pins remain fixed.
            write(self.out / 'summary.json', result)
            text = ['# Released OTTO reference: independent saved-output readback', '',
                    f"Agreement: true. Episodes: 576. Competent reference: {result['competent_reference']}. Promising teacher candidate: {result['stronger_value_teacher']}. Utility/compute comparison: {result['utility_compute_advantage']}.", '',
                    '| Regime | Arm | Weighted success | Weighted capped moves | Controller seconds/episode |',
                    '|---|---|---:|---:|---:|']
            for regime, data in result['cohorts'].items():
                for arm, mean in data['means'].items():
                    text.append(f"| {regime} | {arm} | {mean['found']:.6f} | {mean['capped_time']:.6f} | {mean['controller_seconds']:.6f} |")
            text.extend(['', 'All paired cases, strata, blocks and every condition are retained in summary.json.', '', SCOPE, ''])
            with (self.out / 'report.md').open('x') as stream:
                stream.write('\n'.join(text))
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status='completed', agreement=True, clock_backend=self.clock.backend, started_ns=self.start,
                                finished_ns=finished, wall_seconds=(finished - self.start) / 1e9,
                                comparisons=result['comparisons'], files={p.name: digest(p, self.check) for p in self.out.iterdir()})
            write(self.out / 'receipt.json', self.receipt)
            self.check()
            return self.receipt
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            self.receipt.update(status='failed', agreement=False, error=repr(error), traceback=traceback.format_exc())
            try:
                if (self.out / 'receipt.json').exists():
                    (self.out / 'receipt.json').rename(self.out / 'invalid-completed-receipt.json')
                write(self.out / 'failed.json', self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - Preserve the primary failure.
                error.add_note(f'Failure preservation: {secondary!r}')
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'run', 'terminal', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    for flag in ('plan-sha256', 'receipt-sha256', 'terminal-sha256'):
        parser.add_argument('--' + flag, required=True)
    Audit(parser.parse_args()).execute()
