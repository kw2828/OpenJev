"""Saved-only independent audit of the prospective four-arm boundary study.

Reuses hash-authenticated independent audit primitives, never producer metrics.
The two neural arms retain all four raw float32 costs; only selection eligibility
differs. No model, policy, actor, simulator or RNG is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import signal
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / 'scripts/audit_otto_released_reference.py'
BASE_PIN = '6f63cbc79107fc3046d28645fa4336fe04bfb91ee53368c78ba3971ced1942e1'
if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_PIN:
    raise ValueError('frozen independent arithmetic helper identity')
_spec = importlib.util.spec_from_file_location('_boundary_independent_reference_audit', BASE_PATH)
B = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = B
_spec.loader.exec_module(B)

VERSION = 'otto-boundary-control-saved-audit-v1'
RUNNER = 'scripts/study_otto_boundary_control.py'
ARMS = ('released_tf', 'released_inbounds', 'analytic_all4', 'analytic_inbounds')
NEURAL = frozenset(ARMS[:2])
REGIMES = {'base': 870001, 'shift': 880001}
EPISODES, NEURAL_EPISODES = 768, 384
LIMITS = B.LIMITS
PAYLOADS = B.PAYLOADS | {'paired-prefixes.jsonl'}
SCOPE = ('Independent complete cohort, public posterior, stored raw-score/legal-selection, branch masses, '
         'recorded neural value-to-cost arithmetic, paired initialization, work/cost sums and frozen-rule readback. '
         'Reuses frozen independent auditor posterior/Journal/numeric primitives, not producer scoring. '
         'Original model values, analytic scores, actual model/simulator/RNG execution and measured timing truth '
         'remain inherited authenticated evidence. No inference, policy calls, intervention or old-gate revision.')
require, read, write, digest, load = B.require, B.read, B.write, B.digest, B.load


def identities():
    for regime_index, (regime, first) in enumerate(REGIMES.items()):
        for case in range(96):
            shift = (regime_index * 96 + case) % 4
            for arm in ARMS[shift:] + ARMS[:shift]:
                yield {'cohort': regime, 'seed': first + case, 'block': case // 12,
                       'initial_hit': 1 + (case % 12) // 4, 'arm': arm}


def choose_from_raw(costs, allowed, arm, np):
    require(len(costs) == 4 and allowed and allowed == sorted(set(allowed)), 'ordered nonempty action subset')
    require(all(type(a) is int and 0 <= a < 4 for a in allowed), 'eligible action IDs')
    if arm in NEURAL:
        require(all(type(x) in (int, float) and math.isfinite(x) for x in costs), 'all four finite raw neural costs')
        raw = np.asarray(costs, np.float32)
        require(raw.tolist() == costs, 'exact saved float32 neural costs')
        selected = raw[np.asarray(allowed)]
        ties = np.flatnonzero(np.abs(selected - selected.min()) < 1e-10)
        return allowed[int(ties[0])]
    return B.selected_action(costs, allowed, np, neural=False)


def verify_episode(row, identity, transitions, forwards, journal, kernel, setup, np, c, paired_uniforms, prefix=None):
    """New four-arm orchestration around unchanged independent scalar primitives."""
    c.equal({k: row[k] for k in identity}, identity, 'episode identity')
    steps, arm = row['steps'], identity['arm']
    neural = arm in NEURAL
    require(type(steps) is int and 1 <= steps <= 2188 and type(row['found']) is bool
            and (row['found'] or steps == 2188), 'complete found/censored episode')
    require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0 for k in B.METRICS if k != 'found'), 'finite episode metrics')
    source = row['source_evaluation_only']
    require(len(source) == 2 and all(type(x) is int and 0 <= x < 53 for x in source) and source != [26,26], 'sampled source')
    reset_context = {'phase': 'episode_reset', **identity}
    reset, _ = journal.call('native_reset', reset_context)
    construction, _ = journal.call('actor_construction', reset_context)
    public = B.packet([26,26], identity['initial_hit'], False, 0)
    c.equal(row['initial_public'], public, 'public reset')
    p = B.posterior(np.ones((53,53), np.float64) / 2808, public, kernel, np)
    state = B.witness(p)
    c.equal(next(transitions), {'kind':'reset', **identity, 'public':public, 'posterior_after':state,
                               'source_evaluation_only':source}, 'complete public reset')
    draws = iter(row['draws_evaluation_only'])
    first = next(draws)
    c.equal((first['channel'],first['index'],first['selected_index']), ('source',0,source[0]*53+source[1]), 'source draw identity')
    durations = {k:[] for k in ('choose_seconds','update_seconds','environment_seconds','model_forward_seconds')}
    raw_durations = {k:[] for k in ('choose_seconds','update_seconds')}
    previous, repeated, hits, blocked, stuck = [0,0], 0, 0, 0, 0
    maximum_error = 0.
    for step in range(1,steps+1):
        context = {'phase':'decision', **identity, 'step':step}
        choice, model = journal.call('actor_choose', context, nested=neural)
        native, _ = journal.call('native_step', context)
        update, _ = journal.call('actor_update', context)
        event = next(transitions)
        c.equal({k:event[k] for k in ('kind',*identity,'step')}, {'kind':'step',**identity,'step':step}, 'ordered transition')
        allowed = list(range(4)) if arm in ('released_tf','analytic_all4') else public['valid_actions']
        c.equal(event['allowed_actions'], allowed, 'eligible selection set')
        c.equal(event['selection_mask'], [a in allowed for a in range(4)], 'selection mask')
        raw = np.asarray([math.inf if x is None else x for x in event['costs']], np.float32 if neural else np.float64)
        c.equal(event['raw_costs_sha256'], hashlib.sha256(raw.tobytes()).hexdigest(), 'all-four raw cost bytes')
        action = choose_from_raw(event['costs'], allowed, arm, np)
        c.equal(event['action'], action, 'first eligible raw-score near-tie action')
        if neural:
            forward = next(forwards)
            c.equal(forward['context'], context, 'one causal forward')
            c.equal(forward['ordinal'], journal.counts['tensorflow_value']['returned'], 'forward ordinal')
            c.equal((forward['symmetry_average'],forward['input_shape']), (True,[16,105,105]), 'all16 original branches')
            c.close(forward['seconds'], model['seconds'], 'nested forward duration')
            maximum_error = max(maximum_error, B.neural_costs(forward,event['costs'],B.branch_masses(p,public['position'],kernel,np),np))
        target = B.move(public['position'],action)
        found = target == source
        require(not found or step == steps, 'first finding ends episode')
        hit = event['public']['hit']
        require(type(hit) is int and (hit == -2 if found else 0 <= hit < 4), 'public hit/found semantics')
        after = B.packet(target,hit,found,step)
        c.equal(event['public'],after,'movement/found public packet')
        c.equal(event['native_p_end'],float(found),'native sampled end witness')
        c.equal(event['posterior_before'],state,'causal prior belief')
        if neural and prefix is not None:
            prefix.append({'before': public, 'after': after, 'state': state['sha256'],
                           'costs': raw.tobytes(), 'action': action})
        p = B.posterior(p,after,kernel,np)
        state = B.witness(p)
        c.equal(event['posterior_after'],state,'independently reconstructed posterior')
        is_blocked = target == public['position']
        repeated, is_stuck = B.stuck_counter(repeated,previous,target)
        c.equal(event['blocked'],is_blocked,'blocked motion')
        c.equal(event['stuck'],is_stuck,'upstream nine-return stuck predicate')
        require(not is_blocked or arm in ('released_tf','analytic_all4'), 'restriction forbids blocked choice')
        blocked += is_blocked; stuck += is_stuck
        if not found:
            draw = next(draws)
            c.equal((draw['channel'],draw['index'],draw['selected_index']),('hit',hits,hit),'completed hit draw')
            hits += 1
        for k,value in {'choose_seconds':choice['seconds'], 'choose_instrumented_seconds':choice['instrumented_seconds'],
                        'choose_excluded_io_seconds':choice['excluded_io_seconds'], 'update_seconds':update['seconds'],
                        'update_instrumented_seconds':update['instrumented_seconds'], 'environment_seconds':native['seconds'],
                        'model_forward_seconds':model['seconds'] if neural else 0., 'model_forward_calls':int(neural)}.items():
            c.close(event[k],value,'recorded operation interval')
        for k,values in durations.items():
            values.append(event[k])
        raw_durations['choose_seconds'].append(choice['instrumented_seconds'])
        raw_durations['update_seconds'].append(update['instrumented_seconds'])
        previous,public = public['position'],after
    B.exhausted(draws,'episode draws')
    for draw in row['draws_evaluation_only']:
        require(0 <= draw['uniform'] < 1 and math.isfinite(draw['cdf_mass']) and abs(draw['cdf_mass']-1) < 1e-10, 'valid categorical witness')
        key = (identity['cohort'],identity['seed'],draw['channel'],draw['index'])
        if key in paired_uniforms:
            c.equal(draw['uniform'],paired_uniforms[key],'paired random stream witness')
        else:
            paired_uniforms[key] = draw['uniform']
    for k,value in {'found':public['done'],'capped_time':steps,'choose_calls':steps,'update_calls':steps,
                    'model_forward_calls':steps if neural else 0,'blocked_steps':blocked,'stuck_steps':stuck,
                    'final_update_assimilated':True}.items():
        c.equal(row[k],value,'full episode work/outcome')
    measured = {k:math.fsum(v) for k,v in durations.items()}
    measured.update(actor_initialization_seconds=construction['seconds'],environment_initialization_seconds=reset['seconds'],
                    model_setup_allocation_seconds=setup['model_setup_seconds']/384 if neural else 0.)
    for k,value in measured.items():
        c.close(row[k],value,'episode net complete duration')
    instrumented = {'actor_initialization_seconds':construction['instrumented_seconds'],
                    **{k:math.fsum(v) for k,v in raw_durations.items()},
                    'model_setup_allocation_seconds':setup['model_setup_instrumented_seconds']/384 if neural else 0.}
    c.tree(row['instrumented_component_seconds'],instrumented,'instrumented episode costs')
    c.close(row['controller_seconds'],math.fsum(measured[k] for k in B.TIMES),'complete controller cost')
    c.close(row['controller_instrumented_seconds'],math.fsum(instrumented.values()),'raw complete controller cost')
    c.close(row['controller_excluded_io_seconds'],row['controller_instrumented_seconds']-row['controller_seconds'],'excluded journal cost')
    require(row['episode_seconds'] >= construction['instrumented_seconds']+reset['instrumented_seconds']
            +math.fsum(raw_durations['choose_seconds'])+math.fsum(raw_durations['update_seconds'])+measured['environment_seconds']-1e-8,'episode duration enclosure')
    c.equal(row['state_array_bytes'],22472,'posterior bytes')
    c.equal(row['storage']['mutable_array_bytes'],22472,'mutable owned state')
    c.equal(row['storage']['immutable_array_bytes'],(4+int(not neural))*107*107*8,'kernel/cached geometry bytes')
    return maximum_error


def paired_prefix(original, restricted):
    """Only common histories are comparable; stop after first changed action."""
    require(original and restricted, 'nonempty paired neural histories')
    count, divergence = 0, None
    for left, right in zip(original, restricted, strict=False):
        require(all(left[k] == right[k] for k in ('before', 'state', 'costs')), 'common-prefix raw cost/state/public identity')
        count += 1
        if left['action'] != right['action']:
            require(left['action'] not in left['before']['valid_actions']
                    and right['action'] in right['before']['valid_actions'], 'first difference must restrict original blocked action')
            divergence = left['before']['step'] + 1
            break
        require(left['after'] == right['after'], 'equal choices preserve paired public observations')
    if divergence is None:
        require(len(original) == len(restricted), 'identical choices require complete matching trajectories')
    return {'common_prefix_decisions_checked': count, 'first_divergent_action_step': divergence,
            'raw_cost_and_posterior_parity': True, 'first_divergence_requires_original_blocked': True}


def aggregate(rows, weights):
    """Independent conditional means and frozen float64 threshold decisions."""
    require([{k: r[k] for k in ('cohort', 'seed', 'block', 'initial_hit', 'arm')} for r in rows] == list(identities()), 'all768 ordered episodes')
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
                                          for h in (1, 2, 3)) for m in B.METRICS}
            return result

        means = group(selected)
        strata = {str(h): {a: {m: math.fsum(float(r[m]) for r in selected if r['initial_hit'] == h and r['arm'] == a) / 32
                               for m in B.METRICS} for a in ARMS} for h in (1, 2, 3)}
        blocks = [{'block': b, 'means': group([r for r in selected if r['block'] == b])} for b in range(8)]
        released = means['released_inbounds']
        competence = [{'name': 'released_success_at_least_95pct', 'passes': released['found'] >= .95}]
        teacher, pareto = [], []
        for control in ARMS[2:]:
            other = means[control]
            competence.append({'name': f'moves_at_most_105pct_{control}', 'passes': released['capped_time'] <= 1.05 * other['capped_time']})
            positives = sum(b['means'][control]['capped_time'] > b['means']['released_inbounds']['capped_time'] for b in blocks)
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
                           'restricted_minus_comparator': {a: {m: float(records['released_inbounds'][m]) - float(records[a][m]) for m in ('found', 'capped_time', 'controller_seconds')} for a in ('released_tf', *ARMS[2:])}})
    restriction, strict = [], False
    for regime, data in cohorts.items():
        original, candidate = data['means']['released_tf'], data['means']['released_inbounds']
        restriction.extend([{'name': f'{regime}.no_success_loss', 'passes': candidate['found'] >= original['found']},
                            {'name': f'{regime}.no_more_moves', 'passes': candidate['capped_time'] <= original['capped_time']}])
        strict |= candidate['found'] > original['found'] or candidate['capped_time'] < original['capped_time']
    restriction.append({'name': 'strict_success_or_move_improvement_in_at_least_one_regime', 'passes': strict})
    return {'restriction_benefit_checks': restriction, 'restriction_benefit': all(c['passes'] for c in restriction), 'cohorts': cohorts, 'competent_reference': all(c['passes'] for r in cohorts.values() for c in r['competence_checks']),
            'stronger_value_teacher': all(c['passes'] for r in cohorts.values() for c in r['stronger_teacher_checks']),
            'utility_compute_advantage': all(c['passes'] for r in cohorts.values() for c in r['utility_compute_checks'])}, paired


class Audit(B.Audit):
    def __init__(self, args):
        super().__init__(args)
        self.receipt.update(version=VERSION, scope=SCOPE)

    def authenticate(self):
        a, c = self.args, self.c
        for path in (a.plan, a.run, a.terminal, a.output):
            require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'absolute nonsymlink paths')
        for path, pin in ((a.plan, a.plan_sha256), (a.run / 'receipt.json', a.receipt_sha256), (a.terminal, a.terminal_sha256)):
            c.equal(digest(path, self.check)['sha256'], pin, 'external input identity')
        plan, worker, terminal = read(a.plan), read(a.run / 'receipt.json'), read(a.terminal)
        for name in (RUNNER, 'scripts/audit_otto_boundary_control.py', 'tests/test_audit_otto_boundary_control.py', 'scripts/audit_otto_released_reference.py'):
            c.equal(digest(ROOT / name, self.check)['sha256'], plan['sources'][name], 'pre-import source identity')
        auth = load(ROOT / RUNNER, '_boundary_producer_auth_only')
        validated, paths = auth.authenticate(a)
        c.equal(validated, plan, 'read-only producer authentication')
        require(worker['status'] == 'completed' and worker['version'] == 'otto-boundary-control-v1', 'complete reference run before evidence decoding')
        for key in ('sources', 'inputs', 'limits'):
            c.equal(worker[key], plan[key], 'worker plan join')
        c.equal(worker['plan_sha256'], a.plan_sha256, 'worker plan pin')
        c.equal(set(worker['files']), PAYLOADS, 'exact complete payload names')
        c.equal({p.name for p in a.run.iterdir()}, PAYLOADS | {'receipt.json'}, 'no missing/late/extra evidence')
        for name, expected in worker['files'].items():
            p = a.run / name
            require(p.is_file() and not p.is_symlink(), 'regular payload')
            c.equal(digest(p, self.check), expected, 'all payload hashes before decode')
        require(worker['completed_episodes'] == 768 and worker['training_updates'] == worker['external_model_calls'] == 0
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
        c.equal(launch['clock_source_sha256'], B.CLOCK_PIN, 'clock bytes')
        require(worker['paired_neural_prefixes'] == 192 and worker['candidate_arm'] == 'released_inbounds'
                and worker['inherited_gate_revised'] is False, 'all common-prefix comparisons completed')
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
                and setup['allocated_over_released_episodes'] == 384 and setup['model_parameters'] == 13390849
                and setup['model_tensor_bytes'] == 53563396, 'setup/storage scope')
        metadata = read(paths['tensor_metadata'])
        weight_rows = list(B.lines(run / 'weights.jsonl', self.check))
        expected_weights = []
        for i in range(4):
            for kind in ('kernel', 'bias'):
                key = f'{kind}_{i}'
                record = metadata['datasets'][key]
                expected_weights.append({'id': key, 'shape': record['shape'], 'sha256': record['sha256_c_order'], 'exact': True})
        c.equal(weight_rows, expected_weights, 'ordered loaded-weight identity witnesses')
        journal = B.Journal(iter(B.lines(run / 'work.jsonl', self.check)), c)
        for name in ('tensorflow_construction', 'tensorflow_build', 'tensorflow_load'):
            journal.call(name, {'phase': 'model_setup'})
        require(math.fsum(journal.counts[n]['instrumented_seconds'] for n in ('tensorflow_construction', 'tensorflow_build', 'tensorflow_load')) <= setup['model_setup_instrumented_seconds'] + 1e-9, 'nested setup enclosure')
        transitions, forwards = iter(B.lines(run / 'transitions.jsonl', self.check)), iter(B.lines(run / 'forwards.jsonl', self.check))
        episodes = iter(B.lines(run / 'episodes.jsonl', self.check))
        rows, pairs, uniforms, max_cost_error = [], {}, {}, 0.
        prefixes, prefix_rows = {}, []
        saved_prefixes = iter(B.lines(run / 'paired-prefixes.jsonl', self.check))
        for identity in identities():
            row = next(episodes)
            prefix = []
            max_cost_error = max(max_cost_error, verify_episode(row, identity, transitions, forwards, journal,
                kernels[identity['cohort']], setup, np, c, uniforms, prefix))
            if identity['arm'] in NEURAL:
                prefixes[identity['arm']] = prefix
                if set(prefixes) == NEURAL:
                    record = {'cohort': identity['cohort'], 'seed': identity['seed'],
                              **paired_prefix(prefixes['released_tf'], prefixes['released_inbounds'])}
                    c.equal(next(saved_prefixes), record, 'independent common-prefix comparison')
                    prefix_rows.append(record)
                    prefixes.clear()
            pair = (identity['cohort'], identity['seed'])
            initial = (row['source_evaluation_only'], row['initial_public'])
            if pair in pairs:
                c.equal(initial, pairs[pair], 'paired source/public state')
            else:
                pairs[pair] = initial
            # Drop potentially long draw records after checks, keep all metric rows.
            rows.append({k: v for k, v in row.items() if k != 'draws_evaluation_only'})
        require(not prefixes and len(prefix_rows) == 192, 'all192 complete neural common-prefix comparisons')
        c.equal(published['paired_neural_prefixes'], prefix_rows, 'summary prefix witnesses')
        B.exhausted(saved_prefixes, 'paired prefixes')
        for iterator, name in ((episodes, 'episodes'), (transitions, 'transitions'), (forwards, 'forwards'), (journal.iterator, 'work')):
            B.exhausted(iterator, name)
        c.tree(worker['work']['calls'], journal.counts, 'complete work ledger')
        c.equal(worker['work']['pending'], [], 'no unresolved calls')
        c.equal(worker['work']['sequence'], journal.sequence, 'total call sequence')
        c.equal(worker['work']['context'], journal.last_context, 'last work context')
        expected_work = {ch: journal.counts[ch]['returned'] for ch in B.CHANNELS}
        c.equal(expected_work, {'native_reset': 768, 'native_step': sum(r['steps'] for r in rows),
                'actor_construction': 768, 'actor_choose': sum(r['steps'] for r in rows), 'actor_update': sum(r['steps'] for r in rows),
                'tensorflow_construction': 1, 'tensorflow_build': 1, 'tensorflow_load': 1,
                'tensorflow_value': sum(r['steps'] for r in rows if r['arm'] in NEURAL)}, 'complete expected work')
        derived, paired = aggregate(rows, weights)
        for key, value in derived.items():
            c.tree(published[key], value, 'independent aggregates/' + key)
        c.equal(published['episodes'], 768, 'summary count')
        c.equal(published['paired_source_cases'], len(pairs), 'paired case count')
        c.equal(published['paired_uniform_checks'], len(uniforms), 'paired draw count')
        c.equal(published['actual_work'], expected_work, 'summary work')
        c.equal(published['all_public_beliefs_exact'], True, 'complete public belief reconstruction')
        c.tree(published['setup'], setup, 'published setup')
        for name in ('learned_pilot_admission', 'inherited_gate_revised', 'numpy_port_admitted'):
            c.equal(published[name], False, 'unchanged study scope')
        for name in ('restriction_benefit', 'competent_reference', 'stronger_value_teacher', 'utility_compute_advantage'):
            c.equal(worker[name], derived[name], 'receipt scientific result')
        for field, key in (('sum_controller_seconds', 'controller_seconds'), ('sum_controller_instrumented_seconds', 'controller_instrumented_seconds'), ('sum_excluded_controller_io_seconds', 'controller_excluded_io_seconds')):
            c.close(published[field], math.fsum(r[key] for r in rows), 'complete cost arithmetic')
        require(published['sum_excluded_controller_io_seconds'] <= worker['work']['artifact_io_seconds'] + 1e-8
                and published['artifact_io_seconds'] <= worker['work']['artifact_io_seconds'] + 1e-8, 'I/O exclusion within recorded physical writing')
        disjoint = published['sum_controller_seconds'] + math.fsum(r['environment_seconds'] + r['environment_initialization_seconds'] for r in rows)
        require(disjoint <= worker['wall_seconds'] + 1e-8, 'disjoint computation inside worker duration')
        return {'version': VERSION, 'agreement': True, **derived, 'paired_cases': paired, 'episodes': 768,
                'public_updates': expected_work['actor_update'], 'public_resets': 768, 'actual_work': expected_work,
                'maximum_independent_neural_cost_difference': max_cost_error, 'comparisons': c.count,
                'maximum_aggregate_or_timing_difference': c.maximum_difference,
                'condition_counts': {'restriction_benefit': 5, 'competence': 6, 'promising_teacher_candidate': 12, 'utility_compute': 16},
                'paired_neural_prefixes': prefix_rows, 'scope': SCOPE}

    def execute(self):
        self.out.mkdir(parents=True, exist_ok=False)
        old_alarm = signal.getsignal(signal.SIGALRM)
        try:
            require(digest(ROOT / B.CLOCK)['sha256'] == B.CLOCK_PIN, 'qualified native clock')
            self.clock = load(ROOT / B.CLOCK, '_released_reference_audit_clock').SuspendClock()
            self.start = self.clock.now_ns()
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('audit emergency wall cap')))
            signal.setitimer(signal.ITIMER_REAL, LIMITS['native_seconds'])
            self.receipt['source'] = digest(Path(__file__), self.check)
            self.receipt['independent_helper'] = {'path': str(BASE_PATH), 'sha256': BASE_PIN}
            write(self.out / 'started.json', {'request': {k: str(v) for k, v in vars(self.args).items()},
                                              'limits': LIMITS, 'clock_backend': self.clock.backend, 'started_ns': self.start})
            _, paths, worker = self.authenticate()
            result = self.compute(paths, worker)
            self.authenticate()  # Source, input, manifest and actual process pins remain fixed.
            write(self.out / 'summary.json', result)
            text = ['# OTTO movement restriction: independent saved-output readback', '',
                    f"Agreement: true. Episodes: 768. Competent reference: {result['competent_reference']}. Promising teacher candidate: {result['stronger_value_teacher']}. Utility/compute comparison: {result['utility_compute_advantage']}.", '',
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
