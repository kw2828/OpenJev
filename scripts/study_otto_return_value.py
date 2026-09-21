"""Fixed scalar-return regression with explicit public observation branches.

All training states come from the completed prior TRAIN teacher trajectories.
The matched readouts share targets, row exposure and shuffle orders. Evaluation
uses new seeds and the supplied kernel; no old EVAL or DAgger state is decoded.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'otto-return-value-v1'
MODEL = 'src/openjev/research/otto_return_value.py'
BRANCHES = 'src/openjev/research/otto_value_branches.py'
PROTOCOL = 'research/otto-return-value-protocol.md'
OLD = 'scripts/study_otto_symmetry_head.py'
OLD_PIN = '7f8db5e1b004783deefaffd3dfc98ab571b5c1ee0278d72f7fe94fc6e9b700dd'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
UPSTREAM = 'tmp/otto-source-review-01/isotropic/classes/sourcetracking.py'
FAMILIES, SEEDS = ('min8', 'mlp8', 'homogeneous8'), (10101, 10102, 10103)
ARMS = tuple(f'{f}@{s}' for s in SEEDS for f in FAMILIES) + ('analytic_inbounds',)
REGIMES = {'lambda3': 3., 'lambda4': 4., 'lambda5': 5.}
FIRST = {'eval': {'lambda3': 11100001, 'lambda4': 11200001, 'lambda5': 11300001}}
HORIZON, CASES, EPISODES = 2188, 24, 720
LIMITS = {'native_seconds': 5400, 'rss_bytes': 8 * 1024**3, 'output_bytes': 6 * 1024**3,
          'native_steps': EPISODES * HORIZON + 256, 'native_resets': 731, 'optimizer_updates': 31680}
CONFIGURATION = {'families': list(FAMILIES), 'fit_seeds': list(SEEDS), 'arms': list(ARMS), 'regimes': REGIMES,
                 'first_seeds': FIRST, 'training_episodes': 192, 'training_rows': 5589, 'validation_episodes': 48,
                 'evaluation_cases_per_regime': CASES, 'horizon': HORIZON, 'epochs': 80,
                 'batch_size': 128, 'learning_rate': .001, 'gradient_norm_clip': 5., 'target_scale': 64,
                 'training_weighting': 'uniform row mean squared error; all teacher prefixes',
                 'baseline': 'float32(mean(float32 TRAIN targets, dtype=float64))',
                 'shuffle_seed_offset': 20000, 'checkpoint_rule': 'fixed final epoch80; no validation selection',
                 'validation_every_epochs': 10, 'deployment_arithmetic': 'float64 upcast of float32 parameters',
                 'branch_route': 'all16 explicit max-floor RL branches; no clipping or fused route',
                 'parity_states': 'first8 TRAIN then first8 VALID; scalar and four costs, exact eligible action',
                 'parity_atol': 1e-10, 'parity_rtol': 1e-10,
                 'qualification_first_seed': 11400001, 'qualification_cases': 8, 'qualification_horizon': 32,
                 'template_first_seed': 11500001, 'setup_template_resets': 3, 'Ngrid': 53, 'Nhits': 4,
                 'R_dt': 2., 'Ndim': 2, 'norm_Poisson': 'Euclidean', 'episodes': EPISODES,
                 'runtime_head_allocation_episodes': 72, 'model_module_allocation_episodes': 648}
ROLES = {'prior_plan', 'prior_receipt', 'prior_terminal', 'prior_audit_receipt'}
REQUIRED = {MODEL, BRANCHES, PROTOCOL, CLOCK, UPSTREAM, OLD,
            'src/openjev/research/otto_public.py', 'src/openjev/research/otto_released_policy.py',
            'src/openjev/research/otto_reference_control.py', 'scripts/study_otto_return_value.py',
            'tests/test_otto_return_value_study.py', 'tests/test_otto_return_value.py',
            'tests/test_otto_value_branches.py', 'scripts/audit_otto_return_value.py',
            'tests/test_audit_otto_return_value.py', 'scripts/supervise_dialogue_observation_v2.py'}
THREADS = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                          'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
TIMES = ('init_seconds', 'choose_seconds', 'update_seconds', 'setup_allocation_seconds')
METRICS = ('steps', 'found', *TIMES, 'controller_seconds', 'environment_seconds', 'state_bytes')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def descriptor(path):
    return {'sha256': sha(path), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def regular(name):
    p = Path(name)
    require(not p.is_absolute() and '..' not in p.parts and p.parts, 'contained relative input')
    p = ROOT / p
    require(p.is_file() and not any(v.is_symlink() for v in (p, *p.parents)), 'regular nonsymlink input')
    return p


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def closed(path):
    receipt = read(path)
    require(receipt['status'] == 'completed', 'completed input receipt')
    files = receipt['files']
    actual = {str(p.relative_to(path.parent)) for p in path.parent.rglob('*') if p.is_file()}
    require(actual == set(files) | {path.name}, 'exact closed payload membership')
    for name, desc in files.items():
        rel = Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts, 'payload containment')
        p = path.parent / rel
        require(not any(v.is_symlink() for v in (p, *p.parents)) and descriptor(p) == desc, 'closed payload hash')
    return receipt


def prior_payload_names():
    names = {'started.json', 'runtime.json', 'native-setup.json', 'qualification.json', 'qualification.jsonl',
             'collection-transitions.jsonl', 'collection-episodes.jsonl', 'work-contexts.jsonl', 'work.jsonl',
             'fits.jsonl', 'fit-curves.jsonl', 'epoch-orders.jsonl', 'inference-setup.json', 'eval-transitions.jsonl',
             'eval-episodes.jsonl', 'evaluation.jsonl', 'summary.json', 'pooled-weights.npz', 'pooling.json', 'training-costs.json'}
    names.update(f'kernel-{r}.npz' for r in REGIMES)
    names.update(f'{split}-{suffix}' for split in ('train', 'valid', 'dagger') for suffix in ('data.npz', 'rows.jsonl'))
    names.update(f'{phase}-{family}-{seed}.npz' for phase in ('initial', 'final')
                 for family in ('shared', 'dense') for seed in (9101, 9102, 9103))
    return names


def payload_names():
    names = {'started.json', 'runtime.json', 'native-setup.json', 'qualification.json', 'qualification.jsonl',
             'work-contexts.jsonl', 'work.jsonl', 'preparation.json', 'fits.jsonl', 'fit-curves.jsonl',
             'epoch-orders.jsonl', 'parity.jsonl', 'parity.json', 'inference-setup.json',
             'eval-transitions.jsonl', 'eval-episodes.jsonl', 'evaluation.jsonl', 'summary.json', 'training-costs.json'}
    names.update(f'kernel-{r}.npz' for r in REGIMES)
    names.update(f'{split}-{suffix}' for split in ('train', 'valid') for suffix in ('data.npz', 'rows.jsonl'))
    names.update(f'{prefix}-{family}-{seed}.npz' for prefix in ('final', 'predictions') for family in FAMILIES for seed in SEEDS)
    return names


def authenticate(args):
    require(args.plan.is_absolute() and not args.plan.is_symlink() and sha(args.plan) == args.plan_sha256, 'external plan hash')
    plan = read(args.plan)
    require(plan['version'] == VERSION and plan['status'] == 'frozen_before_native_run'
            and plan['configuration'] == CONFIGURATION and plan['limits'] == LIMITS, 'frozen scientific configuration')
    require(REQUIRED <= plan['sources'].keys() and plan['sources'][CLOCK] == CLOCK_PIN
            and plan['sources'][OLD] == OLD_PIN, 'complete source closure')
    for name, pin in plan['sources'].items():
        require(sha(regular(name)) == pin, f'source hash: {name}')
    require(set(plan['inputs']) == ROLES, 'exact prior input roles')
    paths = {}
    for role, value in plan['inputs'].items():
        paths[role] = regular(value['path'])
        require(descriptor(paths[role]) == {k: value[k] for k in ('sha256', 'bytes')}, 'external prior input descriptor')
    old = load(ROOT / OLD, '_return_prior_auth')
    prior = old.authenticate(SimpleNamespace(plan=paths['prior_plan'], plan_sha256=sha(paths['prior_plan'])))
    require(all(plan['sources'].get(k) == v for k, v in prior['sources'].items()), 'unchanged inherited source closure')
    worker, audit = closed(paths['prior_receipt']), closed(paths['prior_audit_receipt'])
    require(set(worker['files']) == prior_payload_names() and set(audit['files']) == {'started.json', 'summary.json'}, 'prior exact41 and audit2 payloads')
    terminal, started = read(paths['prior_terminal']), read(paths['prior_receipt'].parent / 'started.json')
    require(worker['version'] == prior['version'] and worker['sources'] == prior['sources'] and worker['inputs'] == prior['inputs']
            and worker['plan_sha256'] == sha(paths['prior_plan']) and worker['completed_episodes'] == 720
            and worker['completed_stage_fits'] == 12 and worker['collection_episodes'] == 384
            and worker['pending'] == [] and worker['external_model_calls'] == 0
            and all(v['attempted'] == v['returned'] for v in worker['calls'].values()), 'complete prior study identity')
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
            and terminal['group_absent'] is True and terminal['cleanup']['errors'] == []
            and terminal['cleanup']['group_absent'] is True and terminal['cleanup']['reaped'] is True
            and terminal['error'] is None and terminal['clock_error'] is None and terminal['timing_available'] is True
            and terminal['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < terminal['deadline_ns'],
            'successful prior parent terminal')
    request, launch = started['request'], started['launch']
    require(sha(Path(request['supervision'])) == worker['supervision_sha256'] and read(Path(request['supervision'])) == launch, 'prior launch pin')
    require(request == {'plan': str(paths['prior_plan']), 'plan_sha256': sha(paths['prior_plan']),
                        'output': str(paths['prior_receipt'].parent), 'supervision': request['supervision']}, 'prior exact request')
    for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend',
                'cap_seconds', 'watchdog_sha256', 'clock_source_sha256'):
        require(terminal[key] == launch[key], 'prior parent/worker launch identity')
    command = list(launch['command'])
    if command[1:2] == ['-u']:
        command.pop(1)
    require(command[:2] == [prior['python_executable'], str(ROOT / OLD)] and len(command) == 10
            and dict(zip(command[2::2], command[3::2], strict=True)) == {f'--{k.replace("_", "-")}': v for k, v in request.items()}, 'prior command identity')
    require(worker['limits'] == prior['limits'] and 0 < worker['peak_rss_bytes'] <= prior['limits']['rss_bytes']
            and sum(p.stat().st_size for p in paths['prior_receipt'].parent.iterdir()) <= prior['limits']['output_bytes']
            and launch['cap_seconds'] == 5400 and launch['deadline_ns'] == launch['started_ns'] + 5400 * 10**9
            and worker['clock_backend'] == launch['clock_backend'] and launch['clock_source_sha256'] == CLOCK_PIN
            and launch['watchdog_sha256'] == prior['sources']['scripts/supervise_dialogue_observation_v2.py']
            and launch['cwd'] == str(ROOT) and launch['pid'] == launch['pgid'] != launch['parent_pid']
            and worker['started_ns'] == started['started_ns']
            and worker['wall_seconds'] == (worker['finished_ns'] - worker['started_ns']) / 1e9
            and terminal['elapsed_ns'] == terminal['finished_ns'] - terminal['started_ns']
            and terminal['wall_seconds'] == terminal['elapsed_ns'] / 1e9, 'prior resource and timing identities')
    audit_request = read(paths['prior_audit_receipt'].parent / 'started.json')['request']
    require(audit['version'] == 'otto-symmetry-head-saved-audit-v1' and audit['agreement'] is True
            and audit['plan_sha256'] == sha(paths['prior_plan']) and audit['worker_sha256'] == sha(paths['prior_receipt'])
            and audit['terminal_sha256'] == sha(paths['prior_terminal'])
            and audit['producer_source_sha256'] == OLD_PIN
            and audit['source']['sha256'] == plan['sources']['scripts/audit_otto_symmetry_head.py']
            and audit['training_calls'] == audit['simulator_calls'] == audit['remote_model_calls'] == 0
            and audit_request['run'] == str(paths['prior_receipt'].parent)
            and audit_request['plan'] == str(paths['prior_plan']) and audit_request['terminal'] == str(paths['prior_terminal'])
            and audit_request['receipt_sha256'] == sha(paths['prior_receipt'])
            and audit_request['plan_sha256'] == sha(paths['prior_plan'])
            and audit_request['terminal_sha256'] == sha(paths['prior_terminal']), 'completed independent prior audit')
    require(sys.version.split()[0] == plan['python_version'] and sys.executable == plan['python_executable'], 'runtime interpreter identity')
    require(plan['runtime_versions'] == {'numpy': '2.5.3', 'scipy': '1.18.1', 'torch': '2.14.0'}, 'fixed numerical versions')
    for name, version in plan['runtime_versions'].items():
        require(importlib.metadata.version(name) == version, 'installed runtime version')
    require({d.metadata['Name']: d.version for d in importlib.metadata.distributions()} == plan['all_distributions'], 'installed distribution closure')
    return plan


def packet(value):
    result = dict(value)
    result['position'] = tuple(result['position'])
    result['valid_actions'] = tuple(result['valid_actions'])
    return result


def posterior_witness(belief, np):
    require(belief.shape == (53, 53) and belief.dtype == np.float64 and np.isfinite(belief).all(), 'finite public float64 posterior')
    return {'sha256': hashlib.sha256(belief.tobytes()).hexdigest(), 'mass': float(belief.sum())}


def training_order():
    for stage, count, firsts in (('train', 96, (910001, 920001)), ('valid', 24, (930001, 940001))):
        for regime, first in zip(('lambda3', 'lambda4'), firsts, strict=True):
            for case in range(count):
                yield stage, regime, first + case, 1 + case % 3


def return_target(total_steps, prefix_index):
    require(type(total_steps) is int and 1 <= total_steps <= HORIZON
            and type(prefix_index) is int and 0 <= prefix_index < total_steps, 'current nonterminal teacher prefix')
    return (total_steps - prefix_index) / 64


def center(beliefs, positions, np):
    require(beliefs.dtype == np.float64 and beliefs.ndim == 3 and beliefs.shape[1:] == (53, 53)
            and positions.shape == (len(beliefs), 2), 'current exact belief layout')
    result = np.zeros((len(beliefs), 105, 105), dtype=np.float64)
    for i, (belief, position) in enumerate(zip(beliefs, positions, strict=True)):
        r, c = (52 - int(v) for v in position)
        result[i, r:r + 53, c:c + 53] = belief
    return result


def teacher_records(row, reset, transitions, actor_class, kernel, np, check=lambda: None):
    """Replay one complete saved public teacher path, preserving every prefix."""
    episode_id = row['episode_id']
    require(row['found'] is True and type(row['steps']) is int and 1 <= row['steps'] <= HORIZON
            and row['updates'] == row['steps'] and row['final_update_assimilated'] is True, 'uncensored complete teacher episode')
    require(reset['kind'] == 'reset' and reset['episode_id'] == episode_id
            and reset['public']['step'] == 0 and reset['public']['hit'] == row['initial_hit'], 'teacher reset join')
    actor = actor_class(reset['public'], kernel, allow_stay=False)
    state = posterior_witness(actor.belief, np)
    require(state == reset['posterior_after'], 'reconstructed teacher reset posterior')
    records = []
    for t in range(row['steps']):
        if t % 64 == 0:
            check()
        belief, current = actor.belief, actor.public
        require(current['step'] == t and current['done'] is False, 'all pre-action teacher prefixes')
        metadata = {k: row[k] for k in ('episode_id', 'stage', 'regime', 'seed', 'initial_hit')}
        metadata.update(prefix_index=t, total_steps=row['steps'], public=current, posterior=state,
                        target=return_target(row['steps'], t))
        records.append((belief, metadata))
        event = json.loads(next(transitions))
        require(event['kind'] == 'step' and event['episode_id'] == episode_id and event['step'] == t + 1
                and event['posterior_before'] == state and event['action'] in current['valid_actions'], 'teacher step and pre-action join')
        actor.update(event['action'], event['public'])
        state = posterior_witness(actor.belief, np)
        require(state == event['posterior_after'], 'exact replayed teacher posterior')
    require(actor.public['done'] is True and actor.public == packet(row['final_public']), 'teacher found final state')
    return records


def evaluation_order():
    for ri, (regime, first) in enumerate(FIRST['eval'].items()):
        for case in range(CASES):
            offset = (ri * CASES + case) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                yield regime, first + case, case // 3, 1 + case % 3, arm


def choose(scores, allowed, np):
    mask = np.asarray([a in allowed for a in range(4)], dtype=bool)
    require(scores.shape == (4,) and np.isfinite(scores[mask]).all() and mask.any(), 'finite permitted costs')
    best = scores[mask].min()
    return int(np.flatnonzero(mask & (np.abs(scores - best) < 1e-10))[0])


def summary(rows, mixtures):
    expected = list(evaluation_order())
    require(len(rows) == EPISODES and [(r['regime'], r['seed'], r['block'], r['initial_hit'], r['arm']) for r in rows] == expected,
            'exact complete720 episode cohort')
    for row in rows:
        require(type(row['steps']) is int and 1 <= row['steps'] <= HORIZON and type(row['found']) is bool
                and (row['found'] or row['steps'] == HORIZON) and row['updates'] == row['steps']
                and row['blocked_steps'] == 0, 'complete final updates and censored outcomes')
        require(all(math.isfinite(row[k]) and row[k] >= 0 for k in METRICS)
                and abs(row['controller_seconds'] - math.fsum(row[k] for k in TIMES)) <= 1e-9, 'complete finite cost')
    regimes, competence, compression, architecture = {}, [], [], []

    def check(group, name, value, threshold, passes):
        group.append({'name': name, 'value': value, 'threshold': threshold, 'passes': bool(passes)})

    for regime in REGIMES:
        weights = {int(k): v for k, v in mixtures[regime].items()}
        require(set(weights) == {1, 2, 3} and all(0 < v < 1 for v in weights.values())
                and abs(math.fsum(weights.values()) - 1) <= 1e-12, 'positive normalized mixture')
        selected = [r for r in rows if r['regime'] == regime]

        def weighted(subset, metric, weights=weights):
            return math.fsum(weights[h] * math.fsum(float(r[metric]) for r in subset if r['initial_hit'] == h)
                             / sum(r['initial_hit'] == h for r in subset) for h in (1, 2, 3))

        means = {a: {m: weighted([r for r in selected if r['arm'] == a], m) for m in METRICS} for a in ARMS}
        blocks = [{a: weighted([r for r in selected if r['arm'] == a and r['block'] == b], 'steps') for a in ARMS} for b in range(8)]
        family = {f: {m: math.fsum(means[f'{f}@{s}'][m] for s in SEEDS) / 3 for m in METRICS}
                  for f in FAMILIES}
        teacher, candidate = means['analytic_inbounds'], family['min8']
        for seed in SEEDS:
            value = means[f'min8@{seed}']
            check(competence, f'{regime}.{seed}.success', value['found'], .95, value['found'] >= .95)
            check(competence, f'{regime}.{seed}.moves', value['steps'], 1.05 * teacher['steps'], value['steps'] <= 1.05 * teacher['steps'])
        check(compression, f'{regime}.success', candidate['found'], teacher['found'], candidate['found'] >= teacher['found'])
        check(compression, f'{regime}.moves', candidate['steps'], 1.05 * teacher['steps'], candidate['steps'] <= 1.05 * teacher['steps'])
        check(compression, f'{regime}.cost80', candidate['controller_seconds'], .8 * teacher['controller_seconds'], candidate['controller_seconds'] <= .8 * teacher['controller_seconds'])
        worst = max(means[f'min8@{s}']['controller_seconds'] for s in SEEDS)
        check(compression, f'{regime}.every_cost', worst, teacher['controller_seconds'], worst < teacher['controller_seconds'])
        for control in ('mlp8', 'homogeneous8'):
            ref = family[control]
            positive = sum(math.fsum(b[f'{control}@{s}'] - b[f'min8@{s}'] for s in SEEDS) / 3 > 0 for b in blocks)
            check(architecture, f'{regime}.{control}.success', candidate['found'], ref['found'], candidate['found'] >= ref['found'])
            check(architecture, f'{regime}.{control}.moves', candidate['steps'], .95 * ref['steps'], candidate['steps'] <= .95 * ref['steps'])
            check(architecture, f'{regime}.{control}.positive_blocks', positive, 6, positive >= 6)
            check(architecture, f'{regime}.{control}.cost', candidate['controller_seconds'], ref['controller_seconds'], candidate['controller_seconds'] <= ref['controller_seconds'])
        regimes[regime] = {'weights': weights, 'means': means, 'family_means': family, 'blocks': blocks,
                          'strata': {str(h): {a: {m: math.fsum(float(r[m]) for r in selected if r['arm'] == a and r['initial_hit'] == h) / 8
                                                 for m in METRICS} for a in ARMS} for h in (1, 2, 3)},
                          'raw_counts': {a: {'found': sum(r['found'] for r in selected if r['arm'] == a), 'episodes': 24} for a in ARMS}}
    require((len(competence), len(compression), len(architecture)) == (18, 12, 24), 'all54 conditions')
    return {'version': VERSION, 'episodes': len(rows), 'regimes': regimes,
            'competence_checks': competence, 'compression_checks': compression, 'architecture_checks': architecture,
            'pilot_continuation': all(c['passes'] for c in competence + compression + architecture),
            'learned_architecture_advantage_established': False, 'inherited_gate_revised': False}


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = self.launch = self.plan = None
        self.handles, self.calls, self.pending = {}, {}, []
        self.sequence, self.io_seconds = 0, 0.
        self.context_ids = {}
        self.context = {'phase': 'setup'}
        self.last = {}
        self.receipt = {'version': VERSION, 'status': 'started', 'limits': LIMITS, 'external_model_calls': 0,
                        'completed_fits': 0, 'completed_episodes': 0, 'prepared_episodes': 0, 'training_updates': 0}

    def check(self):
        if self.launch:
            require(self.clock.now_ns() < self.launch['deadline_ns'], 'shared native deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= LIMITS['rss_bytes'], 'RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'output cap')
        for channel, cap in (('native_step', LIMITS['native_steps']), ('native_reset', LIMITS['native_resets'])):
            require(self.calls.get(channel, {}).get('attempted', 0) <= cap, 'native work cap')
        require(self.calls.get('optimizer_update', {}).get('attempted', 0) <= LIMITS['optimizer_updates'], 'optimizer update cap')

    def emit(self, name, value):
        tick = time.perf_counter()
        if name not in self.handles:
            self.handles[name] = (self.out / name).open('x')
        stream = self.handles[name]
        stream.write(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')
        stream.flush()  # Survives worker termination; fsync at each closed episode.
        self.io_seconds += time.perf_counter() - tick

    def call(self, channel, operation, *, check=True):
        if check:
            self.check()
        record = self.calls.setdefault(channel, {'attempted': 0, 'returned': 0, 'seconds': 0.})
        caps = {'native_step': LIMITS['native_steps'], 'native_reset': LIMITS['native_resets'],
                'optimizer_update': LIMITS['optimizer_updates']}
        require(channel not in caps or record['attempted'] < caps[channel], 'work allocation before invocation')
        record['attempted'] += 1
        self.sequence += 1
        call_id = self.sequence
        self.pending.append({'id': call_id, 'channel': channel, 'context': dict(self.context)})
        # Save episode/phase identity once, retaining the current step per call.
        context = {k: v for k, v in self.context.items() if k != 'step'}
        context_key = json.dumps(context, sort_keys=True)
        if context_key not in self.context_ids:
            context_id = len(self.context_ids)
            self.context_ids[context_key] = context_id
            self.emit('work-contexts.jsonl', {'id': context_id, 'context': context})
        self.emit('work.jsonl', [call_id, 0, channel, self.context_ids[context_key], self.context.get('step')])
        tick, io = time.perf_counter(), self.io_seconds
        value = operation()
        raw, excluded = time.perf_counter() - tick, self.io_seconds - io
        duration = raw - excluded
        require(duration >= 0, 'nonnegative measured operation cost')
        record['returned'] += 1
        record['seconds'] += duration
        require(self.pending[-1]['id'] == call_id, 'nested operation accounting')
        self.pending.pop()
        self.last[channel] = {'seconds': duration, 'instrumented_seconds': raw, 'excluded_io_seconds': excluded}
        self.emit('work.jsonl', [call_id, 1, channel, duration, raw, excluded])
        return value

    def sync(self):
        for stream in self.handles.values():
            stream.flush()
            os.fsync(stream.fileno())

    def bind(self):
        require(sha(ROOT / CLOCK) == CLOCK_PIN, 'clock source before import')
        self.clock = load(ROOT / CLOCK, '_return_clock').SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, 'supervision missing')
            time.sleep(.01)
        self.launch, self.plan = read(self.args.supervision), authenticate(self.args)
        launch = self.launch
        command = list(launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and launch['pid'] == os.getpid()
                and launch['pgid'] == os.getpgrp() and launch['parent_pid'] == os.getppid()
                and launch['cwd'] == str(ROOT) == str(Path.cwd()), 'actual supervised process identity')
        require(launch['clock_backend'] == self.clock.backend and launch['started_ns'] <= self.start < launch['deadline_ns']
                and launch['cap_seconds'] == LIMITS['native_seconds']
                and launch['deadline_ns'] == launch['started_ns'] + LIMITS['native_seconds'] * 10**9
                and launch['watchdog_sha256'] == self.plan['sources']['scripts/supervise_dialogue_observation_v2.py']
                and launch['clock_source_sha256'] == CLOCK_PIN, 'actual shared supervisor cap')
        self.receipt.update(plan_sha256=self.args.plan_sha256, supervision_sha256=sha(self.args.supervision),
                            sources=self.plan['sources'], inputs=self.plan['inputs'])
        write(self.out / 'started.json', {'request': {k: str(v) for k, v in vars(self.args).items()}, 'launch': launch, 'started_ns': self.start})

    def setup(self):
        os.environ.update(THREADS)
        tick = time.perf_counter()
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.np, self.torch = np, torch
        sys.path.insert(0, str(ROOT / 'src'))
        from openjev.research.otto_public import observation, seeded_environment
        from openjev.research.otto_reference_control import SpaceAwareActor
        self.observation, self.seeded, self.actor_class = observation, seeded_environment, SpaceAwareActor
        self.SourceTracking = load(ROOT / UPSTREAM, '_return_source').SourceTracking
        model_tick = time.perf_counter()
        self.model = load(ROOT / MODEL, '_return_model')
        self.branches = load(ROOT / BRANCHES, '_return_branches')
        self.model_module_setup_seconds = time.perf_counter() - model_tick
        self.shared_setup_seconds = time.perf_counter() - tick - self.model_module_setup_seconds
        self.prior_run = regular(self.plan['inputs']['prior_receipt']['path']).parent
        self.kernels, self.mixtures = {}, {}
        checks = []
        for i, (regime, _) in enumerate(REGIMES.items()):
            self.context = {'phase': 'native_setup', 'regime': regime}
            env = self.environment(regime, 11500001 + i, None)
            kernel = env.p_Poisson.copy()
            weights = np.asarray(next(r for r in env.draw_log if r['channel'] == 'initial')['probabilities'])
            with np.load(self.prior_run / f'kernel-{regime}.npz', allow_pickle=False) as archive:
                require(np.array_equal(kernel, archive['likelihood']) and np.array_equal(weights, archive['initial_hit_weights']),
                        'exact inherited kernel and initial-hit mixture')
            require(env.N == 53 and env.Nhits == 4 and kernel.shape == (4, 107, 107)
                    and weights.shape == (4,) and weights[0] == 0 and np.all(weights[1:] > 0), 'native geometry and mixture')
            kernel.setflags(write=False)
            self.kernels[regime] = kernel
            self.mixtures[regime] = {h: float(weights[h]) for h in (1, 2, 3)}
            np.savez_compressed(self.out / f'kernel-{regime}.npz', likelihood=kernel, initial_hit_weights=weights)
            checks.append({'regime': regime, 'template_seed': 11500001 + i, 'old_kernel_exact': True,
                           'source_evaluation_only': env.source.tolist(), 'initial_public': self.public(env, 0)})
        write(self.out / 'native-setup.json', {'checks': checks, 'native_resets': 3})
        write(self.out / 'runtime.json', {'python': sys.version, 'executable': sys.executable,
              'versions': {n: importlib.metadata.version(n) for n in self.plan['runtime_versions']},
              'environment': THREADS, 'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
              'shared_setup_seconds': self.shared_setup_seconds, 'model_module_setup_seconds': self.model_module_setup_seconds,
              'module_allocation_episodes': 648, 'module_scope': 'value model and explicit branch imports; excludes common Torch/native setup'})

    def prepare(self):
        np = self.np
        records = {'train': [], 'valid': []}
        with (self.prior_run / 'collection-episodes.jsonl').open() as episodes, (self.prior_run / 'collection-transitions.jsonl').open() as transitions:
            for stage, regime, seed, hit in training_order():
                self.check()
                row = json.loads(next(episodes))
                episode_id = f'{stage}:{regime}:{seed}:teacher'
                require((row['episode_id'], row['stage'], row['regime'], row['seed'], row['initial_hit'], row['arm'])
                        == (episode_id, stage, regime, seed, hit, 'teacher'), 'exact prior teacher cohort/order')
                reset = json.loads(next(transitions))
                records[stage].extend(teacher_records(row, reset, transitions, self.actor_class, self.kernels[regime], np, self.check))
                self.receipt['prepared_episodes'] += 1
        data = {}
        for stage, items in records.items():
            beliefs = np.stack([v[0] for v in items])
            metadata = [v[1] for v in items]
            positions = np.asarray([r['public']['position'] for r in metadata], dtype=np.int64)
            sensing = np.asarray([REGIMES[r['regime']] for r in metadata], dtype=np.float64)
            targets = np.asarray([r['target'] for r in metadata], dtype=np.float32)
            features = np.empty((len(items), self.model.INPUT_DIM), dtype=np.float32)
            for offset in range(0, len(items), 128):
                self.check()
                end = min(offset + 128, len(items))
                # Batches may straddle a regime boundary; construct each regime
                # subset with its own supplied sensor context.
                centered = center(beliefs[offset:end], positions[offset:end], np)
                for lam in (3., 4.):
                    mask = sensing[offset:end] == lam
                    if mask.any():
                        features[offset:end][mask] = self.model.value_features(centered[mask], positions[offset:end][mask], lam, dtype='float32')
            data[stage] = {'features': features, 'target': targets, 'beliefs': beliefs, 'positions': positions,
                           'sensing_length': sensing, 'metadata': metadata}
            np.savez_compressed(self.out / f'{stage}-data.npz', **{k: v for k, v in data[stage].items() if k != 'metadata'})
            for i, row in enumerate(metadata):
                self.emit(f'{stage}-rows.jsonl', {'row_index': i, **row})
        require(len(data['train']['target']) == 5589 and self.receipt['prepared_episodes'] == 240
                and len(data['valid']['target']) >= 8, 'complete fixed teacher exposure')
        self.c0 = np.float32(np.mean(data['train']['target'], dtype=np.float64))
        write(self.out / 'preparation.json', {'status': 'completed', 'source_receipt_sha256': self.plan['inputs']['prior_receipt']['sha256'],
              'c0_float32': float(self.c0), 'episodes': {'train': 192, 'valid': 48},
              'rows': {s: len(d['target']) for s, d in data.items()}, 'all_found': True, 'all_prefixes': True,
              'dagger_or_evaluation_decoded': False, 'weighting': 'uniform row',
              'data': {s: descriptor(self.out / f'{s}-data.npz') for s in data}})
        self.sync()
        return data['train'], data['valid']

    def dataset_features(self, data, offset, end):
        np = self.np
        centered = center(data['beliefs'][offset:end], data['positions'][offset:end], np)
        result = np.empty((end - offset, self.model.INPUT_DIM), dtype=np.float64)
        for lam in (3., 4.):
            mask = data['sensing_length'][offset:end] == lam
            if mask.any():
                result[mask] = self.model.value_features(centered[mask], data['positions'][offset:end][mask], lam)
        return result

    def validation(self, model, data):
        torch = self.torch
        total = 0.
        model.eval()
        with torch.no_grad():
            for offset in range(0, len(data['target']), 128):
                x = torch.from_numpy(data['features'][offset:offset + 128])
                y = torch.from_numpy(data['target'][offset:offset + 128])
                predictions = self.call('validation_forward', lambda x=x: model(x))
                total += float(((predictions - y)**2).sum())
        return total / len(data['target'])

    def predict_dataset(self, head, data):
        np = self.np
        values = np.empty(len(data['target']), dtype=np.float64)
        planes = np.zeros(8, dtype=np.int64) if head.kind == 'min8' else None
        for offset in range(0, len(values), 128):
            self.check()
            end = min(offset + 128, len(values))
            x = self.dataset_features(data, offset, end)
            values[offset:end] = self.call('saved_value_forward', lambda x=x: head.normalized(x))
            if planes is not None:
                active = self.call('plane_diagnostic', lambda x=x: head.plane_indices(x))
                planes += np.bincount(active, minlength=8)
        return values, {'mse_normalized': float(np.mean((values - data['target'])**2)),
                        'negative_count': int((values < 0).sum()), 'rows': len(values),
                        'negative_fraction': float((values < 0).mean()),
                        'plane_counts': planes.tolist() if planes is not None else None}

    def fit(self, train, valid):
        np, torch = self.np, self.torch
        tx, ty = torch.from_numpy(train['features']), torch.from_numpy(train['target'])
        array_hashes = {k: hashlib.sha256(train[k].tobytes()).hexdigest() for k in ('features', 'target')}
        metadata_hash = hashlib.sha256(json.dumps(train['metadata'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        all_parity = []
        for seed in SEEDS:
            for family in FAMILIES:
                fit_id = f'{family}@{seed}'
                self.context = {'phase': 'fit', 'fit_id': fit_id}
                start = time.perf_counter()
                model = self.model.make_head(family, seed, self.c0)
                initial = self.model.export_head(model)
                initial_hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in initial.items() if isinstance(v, np.ndarray)}
                initial_biases_zero = all(not np.any(initial[k]) for k in ('hidden_bias', 'output_bias') if k in initial)
                require(sum(p.numel() for p in model.parameters()) == self.model.parameter_count(family), 'exact family parameter count')
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                rng, curve = np.random.default_rng(seed + 20000), []
                for epoch in range(1, 81):
                    model.train()
                    order = rng.permutation(len(tx))
                    self.emit('epoch-orders.jsonl', {'fit_id': fit_id, 'epoch': epoch, 'rows': len(tx),
                              'sha256': hashlib.sha256(order.tobytes()).hexdigest()})
                    total = 0.
                    for offset in range(0, len(order), 128):
                        idx = order[offset:offset + 128]
                        self.context.update(epoch=epoch, batch=offset // 128)
                        def update(idx=idx, model=model, optimizer=optimizer):
                            loss = ((model(tx[idx]) - ty[idx])**2).mean()
                            require(bool(torch.isfinite(loss)), 'finite mean squared loss')
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'finite present gradients')
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                            optimizer.step()
                            return float(loss.detach())
                        total += self.call('optimizer_update', update) * len(idx)
                        self.receipt['training_updates'] += 1
                    if epoch % 10 == 0:
                        item = {'fit_id': fit_id, 'epoch': epoch, 'training_mse_normalized': total / len(tx),
                                'validation_mse_normalized': self.validation(model, valid)}
                        curve.append(item)
                        self.emit('fit-curves.jsonl', item)
                        print(json.dumps(item), flush=True)
                self.context = {'phase': 'fit_export', 'fit_id': fit_id}
                exported = self.model.export_head(model)
                path = self.out / f'final-{family}-{seed}.npz'
                np.savez_compressed(path, **exported)
                head = self.model.FrozenValue(self.restore(path))
                predictions, diagnostics = {}, {}
                for split, data in (('train', train), ('valid', valid)):
                    predictions[split], diagnostics[split] = self.predict_dataset(head, data)
                np.savez_compressed(self.out / f'predictions-{family}-{seed}.npz', **predictions)
                parity = self.parity(fit_id, model, head, train, valid)
                all_parity.extend(parity)
                steps = [int(s['step'].item()) for s in optimizer.state.values() if 'step' in s]
                require(steps and set(steps) == {3520}, 'complete80 epoch Adam updates')
                self.emit('fits.jsonl', {'fit_id': fit_id, 'family': family, 'seed': seed, 'epochs': 80,
                    'training_rows': len(tx), 'validation_rows': len(valid['target']), 'training_episodes': 192,
                    'training_array_sha256': array_hashes, 'row_order_sha256': metadata_hash, 'c0_float32': float(self.c0),
                    'optimizer_steps': steps, 'optimizer_steps_before': [], 'initial_array_sha256': initial_hashes,
                    'initial_biases_zero': initial_biases_zero, 'parameter_count': self.model.parameter_count(family),
                    'checkpoint': path.name, 'checkpoint_sha256': sha(path), 'fit_seconds': time.perf_counter() - start,
                    'curve': curve, 'diagnostics': diagnostics, 'parity_records': len(parity),
                    'parity_passed': all(r['passed'] for r in parity)})
                self.receipt['completed_fits'] += 1
                self.sync()
        result = {'status': 'completed', 'records': len(all_parity), 'fits': 9, 'atol': 1e-10, 'rtol': 1e-10,
                  'all_passed': all(r['passed'] for r in all_parity), 'exact_action_required': True,
                  'states_per_fit': [{'split': s, 'row_index': i} for s in ('train', 'valid') for i in range(8)]}
        write(self.out / 'parity.json', result)
        require(len(all_parity) == 144 and result['all_passed'], 'fixed learned explicit parity before evaluation')

    def restore(self, path):
        with self.np.load(path, allow_pickle=False) as archive:
            return {k: archive[k] for k in archive.files}

    def physical(self, head, z, positions, lam):
        return 64 * head.normalized(self.model.value_features(z, positions, lam))

    def parity(self, fit_id, model, head, train, valid):
        np, torch = self.np, self.torch
        reference = copy.deepcopy(model).double().eval()
        rows = []
        for split, data in (('train', train), ('valid', valid)):
            for index in range(8):
                self.context = {'phase': 'parity', 'fit_id': fit_id, 'split': split, 'row_index': index}
                belief, position, lam = data['beliefs'][index], data['positions'][index], float(data['sensing_length'][index])
                x = self.dataset_features(data, index, index + 1)
                def torch_values(z, positions, kernel, lam=lam):
                    features = self.model.value_features(z, positions, lam)
                    with torch.no_grad():
                        return 64 * self.call('parity_torch_forward', lambda: reference(torch.from_numpy(features))).numpy()
                with torch.no_grad():
                    scalar_ref = float(self.call('parity_torch_forward', lambda x=x: reference(torch.from_numpy(x))).item())
                scalar_np = float(self.call('parity_numpy_forward', lambda x=x: head.normalized(x))[0])
                branch = self.branches.rl_branches(belief, position, self.kernels[f'lambda{int(lam)}'], data['metadata'][index]['public']['valid_actions'])
                saved = {}
                def actual(z, positions, kernel, lam=lam, saved=saved):
                    values = self.call('parity_numpy_forward', lambda: self.physical(head, z, positions, lam))
                    saved['numpy'] = values
                    return values
                def expected(z, positions, kernel, saved=saved, torch_values=torch_values):
                    values = torch_values(z, positions, kernel)
                    saved['torch'] = values
                    return values
                costs_ref = self.branches.explicit_scores(branch, expected, arithmetic='float64')
                costs_np = self.branches.explicit_scores(branch, actual, arithmetic='float64')
                action_ref = self.branches.select_action(costs_ref, branch.eligible_actions)
                action_np = self.branches.select_action(costs_np, branch.eligible_actions)
                close = lambda a, b: bool(np.all(np.abs(np.asarray(a) - b) <= 1e-10 + 1e-10 * np.abs(b)))
                passed = close(scalar_np, scalar_ref) and close(saved['numpy'], saved['torch']) and close(costs_np, costs_ref) and action_ref == action_np
                row = {'fit_id': fit_id, 'split': split, 'row_index': index,
                       'episode_id': data['metadata'][index]['episode_id'], 'prefix_index': data['metadata'][index]['prefix_index'],
                       'posterior_sha256': data['metadata'][index]['posterior']['sha256'], 'scalar_numpy': scalar_np,
                       'scalar_torch': scalar_ref, 'raw_masses': branch.raw_masses.tolist(), 'weights': branch.weights.tolist(),
                       'values_numpy': saved['numpy'].tolist(), 'values_torch': saved['torch'].tolist(),
                       'costs_numpy': costs_np.tolist(), 'costs_torch': costs_ref.tolist(),
                       'action_numpy': action_np, 'action_torch': action_ref, 'allowed_actions': list(branch.eligible_actions), 'passed': passed}
                self.emit('parity.jsonl', row)
                rows.append(row)
        return rows

    def environment(self, regime, seed, initial_hit):
        config = {'Ndim': 2, 'lambda_over_dx': REGIMES[regime], 'R_dt': 2., 'Ngrid': 53,
                  'Nhits': 4, 'draw_source': True, 'norm_Poisson': 'Euclidean'}
        return self.call('native_reset', lambda: self.seeded(self.SourceTracking, seed, config, initial_hit=initial_hit))

    def public(self, env, step):
        return dict(self.observation(env, step)._asdict())

    def witness(self, actor, env, public):
        belief = actor.belief
        require(actor.public == public and belief.shape == (53, 53) and belief.dtype == self.np.float64
                and belief.tobytes() == env.p_source.tobytes() and self.np.isfinite(belief).all(), 'exact native/public belief')
        return {'sha256': hashlib.sha256(belief.tobytes()).hexdigest(), 'mass': float(belief.sum())}

    def qualify(self):
        checks = []
        before = self.calls.get('native_step', {}).get('returned', 0)
        for case in range(8):
            self.context = {'phase': 'qualification', 'case': case}
            env = self.environment('lambda5', 11400001 + case, 1 + case % 3)
            current = self.public(env, 0)
            actor = self.actor_class(current, self.kernels['lambda5'], allow_stay=False)
            self.witness(actor, env, current)
            self.emit('qualification.jsonl', {'kind': 'reset', 'case': case, 'public': current, 'source_evaluation_only': env.source.tolist()})
            for step in range(1, 33):
                action = (0, 2, 1, 3)[(step - 1) % 4]
                self.context['step'] = step
                self.call('native_step', lambda action=action, env=env: env.step(action, quiet=True))
                after = self.public(env, step)
                actor.update(action, after)
                state = self.witness(actor, env, after)
                self.emit('qualification.jsonl', {'kind': 'step', 'case': case, 'step': step, 'action': action, 'public': after, 'posterior': state})
                if after['done']:
                    break
            checks.append({'case': case, 'steps': step, 'found': after['done'], 'passed': True})
        count = self.calls['native_step']['returned'] - before
        require(count <= 256 and len(checks) == 8, 'fixed bounded kernel5 qualification')
        write(self.out / 'qualification.json', {'status': 'completed', 'checks': checks, 'native_steps': count, 'resets': 8,
              'scope': 'Public filtering at prescribed fresh mechanical prefixes; no neural policy or efficacy evidence.'})
        self.sync()

    def episode(self, regime, seed, hit, arm, block, head):
        np = self.np
        episode_id = f'eval:{regime}:{seed}:{arm}'
        self.context = {'phase': 'eval', 'episode': episode_id, 'step': 0}
        env = self.environment(regime, seed, hit)
        current = self.public(env, 0)
        tick = time.perf_counter()
        actor = self.actor_class(current, self.kernels[regime], allow_stay=False)
        init_seconds = time.perf_counter() - tick
        state = self.witness(actor, env, current)
        identity = {'episode_id': episode_id, 'stage': 'eval', 'regime': regime, 'seed': seed,
                    'initial_hit': hit, 'arm': arm, 'block': block}
        self.emit('eval-transitions.jsonl', {'kind': 'reset', **identity, 'public': current,
                  'posterior_after': state, 'source_evaluation_only': env.source.tolist()})
        choices = updates = native = raw_choices = excluded_io = 0.
        positions, zero_decisions = [], 0
        for step in range(1, HORIZON + 1):
            self.context['step'] = step
            self.check()
            telemetry = None
            tick, io = time.perf_counter(), self.io_seconds
            if head is None:
                _, costs = self.call('analytic_choose', actor._policy._value_policy, check=False)
                action = choose(costs, current['valid_actions'], np)
            else:
                branches = self.branches.rl_branches(actor.belief, current['position'], self.kernels[regime], current['valid_actions'])
                values = None
                def callback(z, successor, kernel):
                    nonlocal values
                    values = self.call('value_forward', lambda: self.physical(head, z, successor, REGIMES[regime]), check=False)
                    return values
                costs = self.branches.explicit_scores(branches, callback, arithmetic='float64')
                action = self.branches.select_action(costs, branches.eligible_actions)
                # Serialize after the complete controller interval; these are
                # references to the just-computed arrays, not extra inference.
                telemetry = (branches.raw_masses, branches.weights, values)
            raw = time.perf_counter() - tick
            excluded = self.io_seconds - io
            elapsed = raw - excluded
            require(elapsed >= 0, 'complete net choice time')
            choices += elapsed
            raw_choices += raw
            excluded_io += excluded
            positions.append(current['position'])
            zero_decisions += state['mass'] == 0
            result = self.call('native_step', lambda action=action: env.step(action, quiet=True))
            native_elapsed = self.last['native_step']['seconds']
            native += native_elapsed
            after = self.public(env, step)
            require((int(result[0]), bool(result[2])) == (after['hit'], after['done']), 'returned native event')
            tick = time.perf_counter()
            actor.update(action, after)
            update_elapsed = time.perf_counter() - tick
            updates += update_elapsed
            after_state = self.witness(actor, env, after)
            self.emit('eval-transitions.jsonl', {'kind': 'step', 'episode_id': episode_id, 'step': step, 'action': action,
                'costs': [float(v) if np.isfinite(v) else None for v in costs], 'allowed_actions': list(current['valid_actions']),
                'raw_masses': telemetry[0].tolist() if telemetry is not None else None,
                'weights': telemetry[1].tolist() if telemetry is not None else None,
                'values': telemetry[2].tolist() if telemetry is not None else None,
                'public': after, 'posterior_before': state, 'posterior_after': after_state,
                'choose_seconds': elapsed, 'choose_instrumented_seconds': raw, 'choose_excluded_io_seconds': excluded,
                'update_seconds': update_elapsed, 'environment_seconds': native_elapsed, 'native_p_end': float(result[1])})
            require(current['position'] != after['position'], 'inbounds actions always move')
            current, state = after, after_state
            if current['done']:
                break
        tail = positions[-256:]
        lag2_count = sum(tail[i] == tail[i - 2] for i in range(2, len(tail)))
        draws = [{k: r[k] for k in ('channel', 'index', 'uniform', 'selected_index', 'cdf_mass')} for r in env.draw_log]
        row = {**identity, 'steps': step, 'found': current['done'], 'updates': step, 'blocked_steps': 0,
               'init_seconds': init_seconds, 'choose_seconds': choices, 'update_seconds': updates,
               'setup_allocation_seconds': 0., 'controller_seconds': init_seconds + choices + updates,
               'choose_instrumented_seconds': raw_choices, 'choose_excluded_io_seconds': excluded_io,
               'environment_seconds': native, 'state_bytes': actor.storage_bytes()['mutable_array_bytes'],
               'storage': {'public_actor': actor.storage_bytes(), 'head': head.storage_bytes() if head is not None else None,
                           'branch_workspace': '16*105*105 float64 centered u and z plus finite branch/features workspace, transient'},
               'zero_mass_decisions': zero_decisions, 'final_posterior_mass': state['mass'],
               'last256_lag2_matches': lag2_count, 'last256_lag2_pairs': max(0, len(tail) - 2),
               'distinct_preaction_positions': len(set(positions)),
               'source_evaluation_only': env.source.tolist(), 'draws_evaluation_only': draws, 'final_public': current,
               'final_update_assimilated': True}
        self.emit('eval-episodes.jsonl', row)
        self.sync()
        return row

    def evaluate(self):
        heads, setup = {}, {}
        for arm in ARMS[:-1]:
            family, seed = arm.split('@')
            path = self.out / f'final-{family}-{seed}.npz'
            self.context = {'phase': 'inference_setup', 'arm': arm}
            tick = time.perf_counter()
            heads[arm] = self.model.FrozenValue(self.restore(path))
            setup[arm] = {'seconds': time.perf_counter() - tick, 'checkpoint': path.name, 'sha256': sha(path),
                          'allocated_episodes': 72, 'storage': heads[arm].storage_bytes()}
        write(self.out / 'inference-setup.json', setup)
        rows, sources, uniforms = [], {}, {}
        for regime, seed, block, hit, arm in evaluation_order():
            row = self.episode(regime, seed, hit, arm, block, heads.get(arm))
            allocation = (setup[arm]['seconds'] / 72 + self.model_module_setup_seconds / 648) if arm in setup else 0.
            row['setup_allocation_seconds'] = allocation
            row['controller_seconds'] += allocation
            self.emit('evaluation.jsonl', row)
            key = (regime, seed)
            source = row['source_evaluation_only']
            require(key not in sources or sources[key] == source, 'matched sampled source')
            sources[key] = source
            for draw in row['draws_evaluation_only']:
                index = (*key, draw['channel'], draw['index'])
                require(index not in uniforms or uniforms[index] == draw['uniform'], 'matched categorical uniform channel')
                uniforms[index] = draw['uniform']
            rows.append(row)
            self.receipt['completed_episodes'] = len(rows)
            if len(rows) % 10 == 0:
                print(json.dumps({'phase': 'eval', 'episodes': len(rows), 'native_steps': self.calls['native_step']['returned']}), flush=True)
        result = summary(rows, self.mixtures)
        result.update(inference_setup=setup, shared_setup_seconds=self.shared_setup_seconds,
                      model_module_setup_seconds=self.model_module_setup_seconds, model_module_allocation_episodes=648,
                      calls=self.calls, paired_source_cases=len(sources), paired_uniforms=len(uniforms),
                      all_public_beliefs_exact=True, scope='Fixed analytic-policy return regression with explicit observation branches. '
                      'Lambda3/4 trained; lambda5 unseen supplied kernel. All old failures retained. '
                      'No new learned memory, no calibrated probabilities, no novelty claim. Final censored/found updates retained.')
        write(self.out / 'summary.json', result)
        return result

    def body(self):
        self.setup()
        self.qualify()
        costs = {}
        tick = time.perf_counter()
        train, valid = self.prepare()
        costs['saved_teacher_preparation_seconds'] = time.perf_counter() - tick
        tick = time.perf_counter()
        self.fit(train, valid)
        costs['fitting_and_parity_seconds'] = time.perf_counter() - tick
        del train, valid
        write(self.out / 'training-costs.json', costs)
        result = self.evaluate()
        self.sync()
        require({p.name for p in self.out.iterdir() if p.is_file()} == payload_names(), 'complete declared output closure')
        return result

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists(), 'exclusive absolute output')
        self.out.mkdir(parents=True, exist_ok=False)
        primary = None
        try:
            self.bind()
            result = self.body()
            require(authenticate(self.args) == self.plan and sha(self.args.supervision) == self.receipt['supervision_sha256'], 'unchanged source/input/launch closure')
            require(not self.pending and all(v['attempted'] == v['returned'] for v in self.calls.values())
                    and self.calls['native_reset']['returned'] == 731 and self.receipt['prepared_episodes'] == 240
                    and self.receipt['completed_fits'] == 9
                    and self.calls['optimizer_update']['returned'] == 31680 and self.receipt['completed_episodes'] == EPISODES,
                    'all fits, episodes and actual work completed')
            self.receipt.update(status='completed', pilot_continuation=result['pilot_continuation'])
            self.check()
        except BaseException as error:
            primary = error
            self.receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            finalization = []
            try:
                self.sync()
                for stream in self.handles.values():
                    stream.close()
            except BaseException as error:  # noqa: BLE001 - preserve primary error and remaining work.
                finalization.append(f'Journal finalization: {error!r}')
            try:
                end = self.clock.now_ns() if self.clock else None
                self.receipt.update(calls=self.calls, pending=self.pending, context=self.context,
                    started_ns=self.start, finished_ns=end, clock_backend=self.clock.backend if self.clock else None,
                    wall_seconds=(end - self.start) / 1e9 if end is not None and self.start is not None else None,
                    artifact_io_seconds=self.io_seconds,
                    files={p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()})
                self.check()
            except BaseException as error:  # noqa: BLE001 - a cap failure must still get a failure receipt.
                finalization.append(f'Final clock/hash/limit check: {error!r}')
                self.receipt.setdefault('started_ns', self.start)
                self.receipt.setdefault('finished_ns', None)
                self.receipt.setdefault('wall_seconds', None)
            if finalization:
                self.receipt.update(status='failed', finalization_errors=finalization,
                                    calls=self.calls, pending=self.pending, context=self.context)
            try:
                write(self.out / 'receipt.json', self.receipt)
            except BaseException as error:  # noqa: BLE001 - retain the original execution exception.
                finalization.append(f'Receipt publication: {error!r}')
            if finalization:
                if primary is not None:
                    primary.add_note('; '.join(finalization))
                else:
                    raise RuntimeError('; '.join(finalization))
        try:
            self.check()
        except BaseException as error:
            try:
                write(self.out / 'late-failure.json', {'status': 'failed', 'error': repr(error)})
                (self.out / 'receipt.json').rename(self.out / 'completion-before-late-failure.json')
                self.receipt.update(status='failed', error=f'Late publication failure: {error!r}',
                    files={p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()})
                write(self.out / 'receipt.json', self.receipt)
            except OSError as secondary:
                error.add_note(f'Late failure preservation: {secondary!r}')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'supervision', 'output'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    Run(parser.parse_args()).execute()
