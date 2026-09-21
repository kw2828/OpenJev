"""Fresh selection-only boundary control with unchanged original TF inference.

Two hash-authenticated namespaces reuse the frozen reference runner: one retains
its original authentication globals, the other supplies unchanged setup,
instrumentation and lifecycle under this prospectively declared configuration.
Only setup metadata publication is deferred to allocate its single cost over 384
neural episodes. Original files, model math, weights and runtime stay unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIOR_SOURCE = 'scripts/study_otto_released_reference.py'
PRIOR_PIN = 'da2a7e418dd29819bddf190f46d1707f1163a22b01a048b33d3dedab875ad87e'
PRIOR_AUDITOR = 'scripts/audit_otto_released_reference.py'
PRIOR_AUDITOR_PIN = '6f63cbc79107fc3046d28645fa4336fe04bfb91ee53368c78ba3971ced1942e1'
RESTRICTED = 'src/openjev/research/otto_restricted_policy.py'
RESTRICTED_PIN = '3e63240b3502b4d4633f9c1671b4ad7ce34466f21c74a4d470bed91d77514630'
PREFLIGHT = 'scripts/qualify_otto_restricted_policy.py'
PROTOCOL = 'research/otto-boundary-control-protocol.md'
VERSION = 'otto-boundary-control-v1'
ARMS = ('released_tf', 'released_inbounds', 'analytic_all4', 'analytic_inbounds')
NEURAL = ARMS[:2]
CONTROLS = ARMS[2:]
CASES, HORIZON, EPISODES, NEURAL_EPISODES = 96, 2188, 768, 384
COHORTS = {name: {'first_seed': seed, 'lambda_over_dx': lam}
           for name, seed, lam in (('base', 870001, 3.), ('shift', 880001, 4.))}
LIMITS = {'native_seconds': 1800, 'rss_bytes': 4 * 1024**3, 'output_bytes': 512 * 1024**2,
          'native_steps': EPISODES * HORIZON, 'native_resets': EPISODES,
          'tensorflow_value_calls': NEURAL_EPISODES * HORIZON}
CONFIGURATION = {'arms': list(ARMS), 'cohorts': COHORTS, 'cases_per_cohort': CASES,
                 'blocks': 8, 'cases_per_hit_per_block': 4, 'horizon': HORIZON,
                 'episodes': EPISODES, 'neural_episodes': NEURAL_EPISODES,
                 'Ngrid': 53, 'Nhits': 4, 'Ndim': 2, 'R_dt': 2., 'norm_Poisson': 'Euclidean',
                 'rotation': 'global_case_index modulo 4', 'symmetry_average': True,
                 'public_native_belief_parity': 'exact', 'common_prefix_cost_parity': 'exact',
                 'new_native_qualification_steps': 0, 'numpy_port_admitted': False, 'learned_pilot_admission': False}
ROLES = {'prior_plan', 'prior_receipt', 'prior_terminal', 'prior_audit_receipt', 'boundary_preflight_receipt'}
REQUIRED = {PRIOR_SOURCE, PRIOR_AUDITOR, RESTRICTED, PREFLIGHT, PROTOCOL,
            'tests/test_otto_restricted_policy.py', 'scripts/study_otto_boundary_control.py', 'tests/test_otto_boundary_control.py',
            'scripts/audit_otto_boundary_control.py', 'tests/test_audit_otto_boundary_control.py'}
SCOPE = ('Fresh four-arm boundary-selection comparison, with unchanged original TF scores and exact public-only state. '
         'Released_inbounds changes final eligibility only. This is a changed policy, not a correction to original OTTO. '
         'No training, recurrent-memory/architecture novelty, previous-gate revision or NumPy-port admission.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


def frozen_base(name):
    path = ROOT / PRIOR_SOURCE
    require(not path.is_symlink() and sha(path) == PRIOR_PIN, 'frozen reference source before stdlib import')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PRIOR = frozen_base('_boundary_untouched_reference')
BASE = frozen_base('_boundary_configured_reference')
read, write, regular, payloads = PRIOR.read, PRIOR.write, PRIOR.regular, PRIOR.payloads
TIMES, METRICS = PRIOR.TIMES, PRIOR.METRICS
packet, belief_witness, ForwardRecorder, Ledger = PRIOR.packet, PRIOR.belief_witness, PRIOR.ForwardRecorder, PRIOR.Ledger


def authenticate(args):
    require(args.plan.is_absolute() and not args.plan.is_symlink() and sha(args.plan) == args.plan_sha256, 'external boundary plan pin')
    plan = read(args.plan)
    require(plan['version'] == VERSION and plan['status'] == 'frozen_before_native_run'
            and plan['configuration'] == CONFIGURATION and plan['limits'] == LIMITS, 'fixed boundary comparison')
    require(REQUIRED <= plan['sources'].keys() and plan['sources'][PRIOR_SOURCE] == PRIOR_PIN
            and plan['sources'][PRIOR_AUDITOR] == PRIOR_AUDITOR_PIN and plan['sources'][RESTRICTED] == RESTRICTED_PIN,
            'required immutable source closure')
    for name, pin in plan['sources'].items():
        require(sha(regular(name)) == pin, f'current source identity: {name}')
    require(set(plan['inputs']) == ROLES, 'exact boundary qualification input roles')
    paths = {}
    for role, record in plan['inputs'].items():
        paths[role] = regular(record['path'])
        require(PRIOR.descriptor(paths[role]) == {'sha256': record['sha256'], 'bytes': record['bytes']}, f'input pin: {role}')
    prior, inherited = PRIOR.authenticate(types.SimpleNamespace(plan=paths['prior_plan'], plan_sha256=sha(paths['prior_plan'])))
    require(plan['runtime'] == prior['runtime'] and all(plan['sources'].get(n) == h for n, h in prior['sources'].items()),
            'unchanged complete prior runtime/source closure')
    receipt = payloads(paths['prior_receipt'])
    require(receipt['version'] == prior['version'] and receipt['plan_sha256'] == sha(paths['prior_plan'])
            and receipt['sources'] == prior['sources'] and receipt['inputs'] == prior['inputs']
            and receipt['completed_episodes'] == 576 and receipt['work']['pending'] == []
            and all(receipt[k] is False for k in ('competent_reference', 'stronger_value_teacher', 'utility_compute_advantage')),
            'complete prior study with failed aggregate rules preserved')
    terminal = read(paths['prior_terminal'])
    started = read(paths['prior_receipt'].parent / 'started.json')
    launch_path = Path(started['request']['supervision'])
    require(sha(launch_path) == receipt['supervision_sha256'] and read(launch_path) == started['launch'], 'prior actual launch join')
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
            and terminal['group_absent'] is True and terminal['cleanup']['errors'] == []
            and terminal['started_ns'] <= receipt['started_ns'] <= receipt['finished_ns'] <= terminal['finished_ns'] < terminal['deadline_ns'],
            'successful completed prior supervisor')
    for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend'):
        require(terminal[key] == started['launch'][key], 'prior supervisor identity')
    audit = payloads(paths['prior_audit_receipt'])
    require(audit['version'] == 'otto-released-reference-saved-audit-v1' and audit['agreement'] is True
            and audit['plan_sha256'] == sha(paths['prior_plan']) and audit['worker_sha256'] == sha(paths['prior_receipt'])
            and audit['terminal_sha256'] == sha(paths['prior_terminal']) and audit['source']['sha256'] == PRIOR_AUDITOR_PIN
            and audit['model_calls'] == audit['simulator_calls'] == audit['policy_calls'] == 0, 'complete independent prior audit join')
    preflight = payloads(paths['boundary_preflight_receipt'])
    require(preflight['version'] == 'otto-restricted-policy-qualification-v1'
            and preflight['qualified'] is True and preflight['agreement'] is True
            and preflight['model_calls'] == preflight['simulator_calls'] == preflight['native_calls'] == 0
            and preflight['comparisons'] == preflight['fake_policy_calls'] == preflight['public_actor_constructions'] == 16
            and preflight['public_updates'] == 128
            and preflight['self_source']['sha256'] == plan['sources'][PREFLIGHT]
            and preflight['sources'][RESTRICTED] == RESTRICTED_PIN
            and all(plan['sources'].get(n) == h for n, h in preflight['sources'].items()), 'qualified restricted selection helper')
    native_plan = regular(prior['inputs']['qualification_plan']['path'])
    native_receipt = regular(prior['inputs']['qualification_receipt']['path'])
    native_terminal = regular(prior['inputs']['qualification_terminal']['path'])
    for key, value in {'plan': str(native_plan), 'plan_sha256': sha(native_plan), 'run': str(native_receipt.parent),
                       'receipt_sha256': sha(native_receipt), 'terminal': str(native_terminal), 'terminal_sha256': sha(native_terminal)}.items():
        require(preflight['request'][key] == value, 'selector preflight uses the same qualified original neural costs')
    return plan, inherited


def case_order():
    for regime_index, (name, config) in enumerate(COHORTS.items()):
        for i in range(CASES):
            offset = (regime_index * CASES + i) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                yield name, config['first_seed'] + i, i // 12, 1 + (i % 12) // 4, arm


def criteria(means, blocks):
    # Exact prior thresholds, applied to the new prospectively fixed candidate.
    renamed = {**means, 'released_tf': means['released_inbounds']}
    renamed_blocks = [{**b, 'means': {**b['means'], 'released_tf': b['means']['released_inbounds']}} for b in blocks]
    return PRIOR.criteria(renamed, renamed_blocks)


def restriction_checks(cohorts):
    checks, strict = [], False
    for regime, data in cohorts.items():
        original, candidate = data['means']['released_tf'], data['means']['released_inbounds']
        checks.extend([{'name': f'{regime}.no_success_loss', 'passes': candidate['found'] >= original['found']},
                       {'name': f'{regime}.no_more_moves', 'passes': candidate['capped_time'] <= original['capped_time']}])
        strict |= candidate['found'] > original['found'] or candidate['capped_time'] < original['capped_time']
    checks.append({'name': 'strict_success_or_move_improvement_in_at_least_one_regime', 'passes': strict})
    return checks


def paired_prefix(original, restricted):
    require(original and restricted, 'two nonempty neural public paths')
    checked, first_divergence = 0, None
    for left, right in zip(original, restricted, strict=False):
        require(left['public_before'] == right['public_before'] and left['posterior_sha256'] == right['posterior_sha256']
                and left['raw_costs_sha256'] == right['raw_costs_sha256'], 'byte-exact raw neural costs and state on every shared prefix')
        checked += 1
        if left['action'] != right['action']:
            require(left['action'] not in left['public_before']['valid_actions']
                    and right['action'] in right['public_before']['valid_actions'], 'first action divergence requires an original blocked choice')
            first_divergence = left['public_before']['step'] + 1
            break
        require(left['public_after'] == right['public_after'], 'paired actions and common random streams preserve public history')
    if first_divergence is None:
        require(len(original) == len(restricted), 'identical choices cannot silently lose final transitions')
    return {'common_prefix_decisions_checked': checked, 'first_divergent_action_step': first_divergence,
            'raw_cost_and_posterior_parity': True, 'first_divergence_requires_original_blocked': True}


def deferred_setup_write(path, value):
    if path.name == 'setup.json':
        require(value['allocated_over_released_episodes'] == 192, 'unchanged inherited setup metadata before adaptation')
        value['allocated_over_released_episodes'] = NEURAL_EPISODES
        return
    write(path, value)


for _key in ('VERSION', 'ARMS', 'CONTROLS', 'CASES', 'HORIZON', 'EPISODES', 'COHORTS', 'LIMITS', 'CONFIGURATION', 'PROTOCOL', 'SCOPE'):
    setattr(BASE, _key, globals()[_key])
BASE.authenticate = authenticate
BASE.write = deferred_setup_write


def summarize(rows, weights):
    expected = list(case_order())
    require(len(rows) == EPISODES and [(r['cohort'], r['seed'], r['block'], r['initial_hit'], r['arm']) for r in rows] == expected,
            'complete exact 768-episode rotated cohort')
    require(set(weights) == set(COHORTS), 'both regime mixtures')
    for row in rows:
        require(type(row['steps']) is int and 1 <= row['steps'] <= HORIZON and row['capped_time'] == row['steps']
                and type(row['found']) is bool and (row['found'] or row['steps'] == HORIZON), 'found/censored horizon semantics')
        require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0 for k in METRICS if k != 'found'), 'finite nonnegative metrics')
        require(all(type(row[k]) is int and 0 <= row[k] <= row['steps'] for k in ('stuck_steps', 'blocked_steps')), 'integer movement counts')
        require(row['arm'] not in ('released_inbounds', 'analytic_inbounds') or row['blocked_steps'] == 0,
                'restricted blocked count is a validity invariant')
        require(row['choose_calls'] == row['update_calls'] == row['steps']
                and row['model_forward_calls'] == (row['steps'] if row['arm'] in NEURAL else 0), 'actual operation count joins')
        require(abs(row['controller_seconds'] - math.fsum(row[k] for k in TIMES)) <= 1e-9, 'complete controller cost sum')
        require(type(row['controller_instrumented_seconds']) in (int, float)
                and type(row['controller_excluded_io_seconds']) in (int, float)
                and math.isfinite(row['controller_instrumented_seconds']) and math.isfinite(row['controller_excluded_io_seconds'])
                and row['controller_excluded_io_seconds'] >= 0
                and abs(row['controller_instrumented_seconds'] - row['controller_seconds'] - row['controller_excluded_io_seconds']) <= 1e-9,
                'raw controller time and excluded I/O reconstruction')
    result = {}
    for name in COHORTS:
        mixture = weights[name]
        require(set(mixture) == {1, 2, 3} and all(math.isfinite(w) and 0 < w < 1 for w in mixture.values())
                and abs(math.fsum(mixture.values()) - 1) <= 1e-12, 'fixed positive initial-hit mixture')
        selected = [r for r in rows if r['cohort'] == name]

        def group(subset, mixture=mixture):
            return {arm: {metric: math.fsum(mixture[h] * math.fsum(float(r[metric]) for r in subset if r['arm'] == arm and r['initial_hit'] == h)
                                          / sum(r['arm'] == arm and r['initial_hit'] == h for r in subset) for h in (1, 2, 3))
                          for metric in METRICS} for arm in ARMS}

        means = group(selected)
        strata = {h: {a: {m: math.fsum(float(r[m]) for r in selected if r['initial_hit'] == h and r['arm'] == a) / 32
                         for m in METRICS} for a in ARMS} for h in (1, 2, 3)}
        blocks = [{'block': b, 'means': group([r for r in selected if r['block'] == b])} for b in range(8)]
        gates = criteria(means, blocks)
        counts = {a: {'episodes': 96, 'found': sum(r['found'] for r in selected if r['arm'] == a),
                      'censored': sum(not r['found'] for r in selected if r['arm'] == a),
                      'total_steps': sum(r['steps'] for r in selected if r['arm'] == a)} for a in ARMS}
        result[name] = {'initial_hit_weights': mixture, 'means': means, 'strata': strata, 'blocks': blocks,
                        'unweighted_counts': counts,
                        'competence_checks': gates[0], 'stronger_teacher_checks': gates[1], 'utility_compute_checks': gates[2]}
    restriction = restriction_checks(result)
    return {'version': VERSION, 'scope': SCOPE, 'episodes': EPISODES, 'cohorts': result,
            'competent_reference': all(c['passes'] for r in result.values() for c in r['competence_checks']),
            'stronger_value_teacher': all(c['passes'] for r in result.values() for c in r['stronger_teacher_checks']),
            'utility_compute_advantage': all(c['passes'] for r in result.values() for c in r['utility_compute_checks']),
            'restriction_benefit_checks': restriction, 'restriction_benefit': all(c['passes'] for c in restriction),
            'candidate_arm': 'released_inbounds', 'learned_pilot_admission': False, 'inherited_gate_revised': False, 'numpy_port_admitted': False}


class Run(BASE.Run):
    def setup(self):
        runtime = super().setup()
        tick = time.perf_counter()
        require(sha(ROOT / RESTRICTED) == RESTRICTED_PIN, 'unchanged restricted actor before import')
        runtime.restricted = PRIOR.load(ROOT / RESTRICTED, '_boundary_restricted_actor').RestrictedPolicyActor
        runtime.setup['shared_evaluator_setup_seconds'] += time.perf_counter() - tick
        runtime.setup['reuse'] = {'source': PRIOR_SOURCE, 'sha256': PRIOR_PIN,
                                 'inference_changed': False, 'setup_allocation_episodes': NEURAL_EPISODES,
                                 'setup_metadata_publication': 'Deferred once until restricted actor import completes.'}
        write(self.out / 'setup.json', runtime.setup)
        return runtime

    # Episode and cohort methods below are explicit local adaptations of the
    # pinned runner. Model loading, call recording and lifecycle stay inherited.
    def episode(self, runtime, identity):
        np = runtime.np
        cohort, seed, block, hit, arm = identity
        context = {'cohort': cohort, 'seed': seed, 'block': block, 'initial_hit': hit, 'arm': arm}
        episode_tick = time.perf_counter()
        self.ledger.context = {'phase': 'episode_reset', **context}
        config = {'Ndim': 2, 'Ngrid': 53, 'Nhits': 4, 'lambda_over_dx': COHORTS[cohort]['lambda_over_dx'],
                  'R_dt': 2., 'norm_Poisson': 'Euclidean'}
        env = self.ledger.call('native_reset', lambda: runtime.public.seeded_environment(runtime.source, seed, config, initial_hit=hit))
        environment_init = self.ledger.last_seconds['native_reset']
        require(env.N == 53 and env.Nhits == env.Nactions == 4 and env.draw_source is True
                and env.p_Poisson.dtype == runtime.kernels[cohort].dtype
                and env.p_Poisson.tobytes() == runtime.kernels[cohort].tobytes(), 'native environment matches known public kernel')
        current = packet(runtime.public.observation(env, 0))
        initial_public = current
        recorder = ForwardRecorder(runtime.model, runtime.policy._value_policy.__code__, self.ledger, np)

        def construct():
            if arm in NEURAL:
                constructor = runtime.restricted if arm == 'released_inbounds' else runtime.released
                return constructor(current, runtime.kernels[cohort], recorder, runtime.policy, sym_avg=True)
            return runtime.analytic(current, runtime.kernels[cohort], allow_stay=arm == 'analytic_all4')

        actor = self.ledger.call('actor_construction', construct)
        times = {'actor_initialization_seconds': self.ledger.last_seconds['actor_construction'],
                 'choose_seconds': 0., 'update_seconds': 0.,
                 'model_setup_allocation_seconds': runtime.setup['model_setup_seconds'] / NEURAL_EPISODES if arm in NEURAL else 0.}
        raw_times = {'actor_initialization_seconds': self.ledger.last_instrumented['actor_construction'],
                     'choose_seconds': 0., 'update_seconds': 0.,
                     'model_setup_allocation_seconds': runtime.setup['model_setup_instrumented_seconds'] / NEURAL_EPISODES if arm in NEURAL else 0.}
        storage = actor.storage_bytes()
        state = belief_witness(actor, env, current, np)
        self.ledger.emit('transitions.jsonl', {'kind': 'reset', **context, 'public': current, 'posterior_after': state,
                                             'source_evaluation_only': env.source.tolist()})
        native_seconds = model_seconds = 0.
        stuck = blocked = 0
        neural_path = []
        for step in range(1, HORIZON + 1):
            self.ledger.context = {'phase': 'decision', **context, 'step': step}
            require(current['done'] is False, 'no action after found')
            before_calls = self.ledger.calls['tensorflow_value']['returned']
            action, costs = self.ledger.call('actor_choose', actor.choose)
            choose = self.ledger.last_seconds['actor_choose']
            choose_raw, choose_io = self.ledger.last_instrumented['actor_choose'], self.ledger.last_io['actor_choose']
            allowed = tuple(range(4)) if arm in ('released_tf', 'analytic_all4') else current['valid_actions']
            require(action in allowed and costs.shape == (4,)
                    and costs.dtype == (np.float32 if arm in NEURAL else np.float64), 'declared action and score contract')
            require(all(np.isfinite(costs[a]) if arm in NEURAL or a in allowed else np.isposinf(costs[a]) for a in range(4)),
                    'all neural costs remain finite; only analytic excluded moves are infinite')
            permitted_costs = costs[list(allowed)]
            require(allowed[int(np.flatnonzero(np.abs(permitted_costs - permitted_costs.min()) < 1e-10)[0])] == action,
                    'exact first near-tie selection among permitted actions')
            forwards = self.ledger.calls['tensorflow_value']['returned'] - before_calls
            require(forwards == int(arm in NEURAL), 'one actual model call per released decision and zero per analytic decision')
            model_cost = self.ledger.last_seconds['tensorflow_value'] if forwards else 0.
            result = self.ledger.call('native_step', lambda action=action: env.step(action, quiet=True))
            native_cost = self.ledger.last_seconds['native_step']
            after = packet(runtime.public.observation(env, step))
            require((int(result[0]), bool(result[2])) == (after['hit'], after['done']), 'native returned public event')
            self.ledger.call('actor_update', lambda action=action, after=after: actor.update(action, after))
            update = self.ledger.last_seconds['actor_update']
            update_raw = self.ledger.last_instrumented['actor_update']
            after_state = belief_witness(actor, env, after, np)
            is_blocked = current['position'] == after['position']
            stuck += int(env.agent_stuck)
            blocked += int(is_blocked)
            times['choose_seconds'] += choose
            times['update_seconds'] += update
            raw_times['choose_seconds'] += choose_raw
            raw_times['update_seconds'] += update_raw
            native_seconds += native_cost
            model_seconds += model_cost
            raw_costs_sha256 = hashlib.sha256(costs.tobytes()).hexdigest()
            if arm in NEURAL:
                neural_path.append({'public_before': current, 'posterior_sha256': state['sha256'],
                                    'raw_costs_sha256': raw_costs_sha256, 'action': int(action), 'public_after': after})
            self.ledger.emit('transitions.jsonl', {'kind': 'step', **context, 'step': step, 'action': int(action),
                'costs': [float(v) if np.isfinite(v) else None for v in costs], 'allowed_actions': list(allowed), 'selection_mask': [a in allowed for a in range(4)],
                'raw_costs_sha256': raw_costs_sha256,
                'public': after, 'posterior_before': state, 'posterior_after': after_state,
                'native_p_end': float(result[1]), 'blocked': is_blocked, 'stuck': bool(env.agent_stuck),
                'choose_seconds': choose, 'choose_instrumented_seconds': choose_raw, 'choose_excluded_io_seconds': choose_io,
                'update_seconds': update, 'update_instrumented_seconds': update_raw,
                'environment_seconds': native_cost, 'model_forward_seconds': model_cost, 'model_forward_calls': forwards})
            current, state = after, after_state
            if current['done']:
                break
        require(arm not in ('released_inbounds', 'analytic_inbounds') or blocked == 0, 'restricted policies never take blocked moves')
        self.neural_path = neural_path
        require(actor.storage_bytes() == storage and current['step'] == step, 'complete final update and fixed actor state storage')
        draws = [{k: r[k] for k in ('channel', 'index', 'uniform', 'selected_index', 'cdf_mass')} for r in env.draw_log]
        require([r['index'] for r in draws if r['channel'] == 'source'] == [0]
                and [r['index'] for r in draws if r['channel'] == 'hit'] == list(range(step - int(current['done'])))
                and all(r['channel'] in ('source', 'hit') and 0 <= r['uniform'] < 1 and math.isfinite(r['cdf_mass']) and r['cdf_mass'] > 0 for r in draws),
                'complete paired categorical source and nonterminal hit draws')
        require(next(r['selected_index'] for r in draws if r['channel'] == 'source') == int(env.source[0]) * 53 + int(env.source[1]),
                'saved source draw agrees with evaluator source')
        row = {**context, 'steps': step, 'capped_time': step, 'found': current['done'], 'stuck_steps': stuck, 'blocked_steps': blocked,
               **times, 'controller_seconds': math.fsum(times.values()), 'controller_instrumented_seconds': math.fsum(raw_times.values()),
               'controller_excluded_io_seconds': math.fsum(raw_times.values()) - math.fsum(times.values()),
               'instrumented_component_seconds': raw_times, 'model_forward_seconds': model_seconds,
               'environment_initialization_seconds': environment_init, 'environment_seconds': native_seconds,
               'episode_seconds': time.perf_counter() - episode_tick, 'state_array_bytes': storage['mutable_array_bytes'],
               'storage': storage, 'choose_calls': step, 'update_calls': step, 'model_forward_calls': step if arm in NEURAL else 0,
               'initial_public': initial_public,
               'source_evaluation_only': env.source.tolist(), 'draws_evaluation_only': draws, 'final_update_assimilated': True}
        self.ledger.emit('episodes.jsonl', row)
        return row

    def body(self):
        runtime = self.setup()
        rows, source_pairs, initial_pairs, hit_pairs = [], {}, {}, {}
        neural_pairs, prefix_checks = {}, []
        for identity in case_order():
            row = self.episode(runtime, identity)
            case = (row['cohort'], row['seed'])
            if row['arm'] in NEURAL:
                neural_pairs[row['arm']] = self.neural_path
                if set(neural_pairs) == set(NEURAL):
                    witness = {'cohort': case[0], 'seed': case[1], **paired_prefix(neural_pairs['released_tf'], neural_pairs['released_inbounds'])}
                    self.ledger.emit('paired-prefixes.jsonl', witness)
                    prefix_checks.append(witness)
                    neural_pairs = {}
            if case in source_pairs:
                require(row['source_evaluation_only'] == source_pairs[case] and row['initial_public'] == initial_pairs[case],
                        'paired sampled source and initial public state across all arms')
            else:
                source_pairs[case] = row['source_evaluation_only']
                initial_pairs[case] = row['initial_public']
            for draw in row['draws_evaluation_only']:
                key = (*case, draw['channel'], draw['index'])
                if key in hit_pairs:
                    require(hit_pairs[key] == draw['uniform'], 'paired source/hit uniform stream at each shared draw index')
                else:
                    hit_pairs[key] = draw['uniform']
            rows.append(row)
            self.receipt['completed_episodes'] = len(rows)
            print(json.dumps({'completed_episodes': len(rows), 'native_steps': self.ledger.calls['native_step']['returned'],
                              'model_forwards': self.ledger.calls['tensorflow_value']['returned']}), flush=True)
        require(not self.ledger.pending and all(r['attempted'] == r['returned'] for r in self.ledger.calls.values()), 'all actual work returned')
        total_steps = sum(r['steps'] for r in rows)
        expected = {'native_reset': EPISODES, 'native_step': total_steps, 'actor_construction': EPISODES,
                    'actor_choose': total_steps, 'actor_update': total_steps,
                    'tensorflow_value': sum(r['steps'] for r in rows if r['arm'] in NEURAL),
                    'tensorflow_construction': 1, 'tensorflow_build': 1, 'tensorflow_load': 1}
        require({k: r['returned'] for k, r in self.ledger.calls.items()} == expected, 'complete actual operation count closure')
        require(not neural_pairs and len(prefix_checks) == 192, 'all 192 paired neural-prefix checks complete')
        summary = summarize(rows, runtime.weights)
        summary.update(paired_neural_prefixes=prefix_checks, setup=runtime.setup, actual_work=expected, paired_source_cases=len(source_pairs),
                       paired_uniform_checks=len(hit_pairs), all_public_beliefs_exact=True,
                       sum_controller_seconds=math.fsum(r['controller_seconds'] for r in rows),
                       sum_controller_instrumented_seconds=math.fsum(r['controller_instrumented_seconds'] for r in rows),
                       sum_excluded_controller_io_seconds=math.fsum(r['controller_excluded_io_seconds'] for r in rows),
                       artifact_io_seconds=self.ledger.io_seconds,
                       timing_scope='Primary controller cost includes actor initialization, all decisions and all updates, '
                       'plus one complete model/runtime setup allocated over 384 neural episodes. Nested measured journal '
                       'and telemetry serialization/fsync are subtracted; raw instrumented costs are retained. '
                       'Forward cost is a subset of choose cost, not added twice. Environment and evaluator parity checks '
                       'are separate. Whole worker/supervisor cost includes all I/O, validation and process overhead. '
                       'Episode time excludes its final row serialization; first cold forward is included.')
        write(self.out / 'summary.json', summary)
        self.receipt.update(restriction_benefit=summary['restriction_benefit'], candidate_arm='released_inbounds',
                            paired_neural_prefixes=len(prefix_checks), inherited_gate_revised=False)
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'supervision', 'output'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    Run(parser.parse_args()).execute()
