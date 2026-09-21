"""Frozen public-input policy-distillation pilot with autonomous evaluation.

The shared full-Bayes teacher supplies heuristic action preferences, not Q values.
Each learned actor receives only its declared memory and public context.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from openjev.research.suspend_clock import SuspendClock

VERSION = 'otto-action-head-v1'
FAMILIES = ('dct16_neutral', 'dct16_nearest', 'recent32_hard', 'full_bayes')
FIT_SEEDS = (7901, 7902, 7903)
CASES, HORIZON = 48, 2188
TRAIN_CASES, VALID_CASES, COLLECTION_HORIZON = 384, 96, 256
PLANNERS = ('full_bayes', 'dct16_neutral', 'dct16_nearest', 'recent32_hard')
ARMS = tuple(f'{f}@{seed}' for seed in FIT_SEEDS for f in FAMILIES) + tuple(f'{f}@planner' for f in PLANNERS)
CONFIGURATION = {'families': list(FAMILIES), 'fit_seeds': list(FIT_SEEDS), 'arms': list(ARMS),
 'training_cases': TRAIN_CASES, 'validation_cases': VALID_CASES, 'collection_horizon': COLLECTION_HORIZON,
 'training_first_seed': 670001, 'validation_first_seed': 680001, 'exploration_probability': .15,
 'evaluation_first_seeds': {'base': 690001, 'shift': 700001}, 'cases_per_regime': CASES,
 'evaluation_horizon': HORIZON, 'epochs': 40, 'batch_size': 256, 'learning_rate': .0003,
 'teacher_temperature': .25, 'checkpoint_epochs': list(range(5, 41, 5)),
 'checkpoint_rule': 'minimum finite validation cross entropy; first checkpoint wins exact ties',
 'training_weighting': 'equal decision rows across balanced initial-hit episode strata',
 'learned_architecture_claim': False}
LIMITS = {'native_seconds': 5400, 'rss_bytes': 8 * 1024**3, 'output_bytes': 2 * 1024**3,
 'native_steps': (TRAIN_CASES + VALID_CASES) * COLLECTION_HORIZON + 2 * CASES * len(ARMS) * HORIZON + 2048}
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
REQUIRED = {'scripts/study_otto_action_head.py', 'src/openjev/research/otto_action_head.py',
 'tests/test_otto_action_head.py', 'tests/test_otto_action_head_study.py', 'research/otto-action-head-protocol.md',
 'scripts/study_otto_spectral_control.py', 'scripts/study_otto_spectral_memory.py', 'scripts/audit_otto_large_memory.py',
 'src/openjev/research/otto_spectral_memory.py', 'src/openjev/research/otto_public.py',
 'src/openjev/research/suspend_clock.py', 'scripts/supervise_dialogue_observation_v2.py'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024**2), b''):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def emit(f, value):
    f.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def restore_head(path, np):
    with np.load(path, allow_pickle=False) as archive:
        head = {k: archive[k] for k in archive.files}
    for key in ('version', 'input_dim'):
        require(head[key].ndim == 0, 'scalar checkpoint metadata')
        head[key] = head[key].item()
    return head


def evaluation_order():
    for ri, (regime, first) in enumerate(CONFIGURATION['evaluation_first_seeds'].items()):
        for case in range(CASES):
            rotation = (ri * CASES + case) % len(ARMS)
            for arm in ARMS[rotation:] + ARMS[:rotation]:
                yield regime, first + case, case // 6, 1 + case % 6 // 2, arm


def teacher_target(scores, np):
    valid = np.array([v is not None for v in scores], dtype=bool)
    require(len(scores) == 4 and valid.any(), 'teacher legal action support')
    values = np.asarray([0 if v is None else v for v in scores], dtype=np.float64)
    require(np.isfinite(values).all(), 'finite teacher scores')
    costs = values[valid]
    scaled = -(costs - costs.min()) / max(float(costs.max() - costs.min()), 1e-8) / CONFIGURATION['teacher_temperature']
    probability = np.exp(scaled)
    probability /= probability.sum()
    target = np.zeros(4, dtype=np.float32)
    target[valid] = probability
    return target, valid


def aggregate(rows, weights):
    require(len(rows) == 2 * CASES * len(ARMS), 'complete autonomous episode coverage')
    expected = list(evaluation_order())
    for row, key in zip(rows, expected, strict=True):
        require(tuple(row[k] for k in ('regime', 'seed', 'block', 'initial_hit', 'arm')) == key, 'exact evaluation order/identity')
        require(1 <= row['steps'] <= HORIZON and (row['found'] or row['steps'] == HORIZON), 'capped failures retained')
        require(row['updates'] == row['steps'] - int(row['found']), 'all nonterminal readings updated')
        require(math.isclose(row['controller_seconds'], sum(row[k] for k in ('init_seconds', 'update_seconds', 'decision_seconds', 'setup_allocation_seconds')), abs_tol=1e-9), 'complete controller cost')
        require(all(math.isfinite(row[k]) and row[k] >= 0 for k in ('controller_seconds', 'environment_seconds', 'state_bytes')), 'finite measured metrics')
    def mean(selected, name, ws):
        return math.fsum(ws[h] * math.fsum(float(r[name]) for r in selected if r['initial_hit'] == h) / sum(r['initial_hit'] == h for r in selected) for h in (1, 2, 3))
    result = {}
    for regime in ('base', 'shift'):
        ws = {int(k): v for k, v in weights[regime].items()}
        metrics = ('steps', 'found', 'controller_seconds', 'environment_seconds', 'init_seconds', 'update_seconds', 'decision_seconds', 'setup_allocation_seconds', 'state_bytes')
        means = {a: {m: mean([r for r in rows if r['regime'] == regime and r['arm'] == a], m, ws) for m in metrics} for a in ARMS}
        blocks = [{a: mean([r for r in rows if r['regime'] == regime and r['arm'] == a and r['block'] == block], 'steps', ws) for a in ARMS} for block in range(8)]
        family = {f: {m: math.fsum(means[f'{f}@{seed}'][m] for seed in FIT_SEEDS) / 3 for m in metrics} for f in FAMILIES}
        full = means['full_bayes@planner']
        criteria = {}
        for f in FAMILIES[:2]:
            c = family[f]
            gains = [math.fsum(b[f'recent32_hard@{seed}'] - b[f'{f}@{seed}'] for seed in FIT_SEEDS) / 3 for b in blocks]
            checks = {
              'every_fit_success_at_least_95pct': all(means[f'{f}@{seed}']['found'] >= .95 for seed in FIT_SEEDS),
              'mean_success_no_worse_than_full_planner': c['found'] >= full['found'],
              'mean_moves_at_most_105pct_full_planner': c['steps'] <= 1.05 * full['steps'],
              'mean_controller_at_most_80pct_full_planner': c['controller_seconds'] <= .8 * full['controller_seconds'],
              'every_fit_controller_cheaper_than_full_planner': all(means[f'{f}@{seed}']['controller_seconds'] < full['controller_seconds'] for seed in FIT_SEEDS),
              'mean_moves_at_most_95pct_recent_head': c['steps'] <= .95 * family['recent32_hard']['steps'],
              'positive_blocks_vs_recent_head_at_least_six': sum(g > 0 for g in gains) >= 6,
              'mean_moves_at_most_105pct_full_head': c['steps'] <= 1.05 * family['full_bayes']['steps'],
              'mean_controller_no_more_than_full_head': c['controller_seconds'] <= family['full_bayes']['controller_seconds'],
              'state_at_most_20pct_full': c['state_bytes'] <= .2 * full['state_bytes']}
            criteria[f] = {'checks': checks, 'passes': all(checks.values()), 'block_gains_vs_recent_head': gains}
        result[regime] = {'weights': ws, 'means': means, 'family_means': family, 'blocks': blocks, 'criteria': criteria,
          'strata': {str(h): {a: {m: math.fsum(float(r[m]) for r in rows if r['regime'] == regime and r['arm'] == a and r['initial_hit'] == h) / (CASES // 3) for m in metrics} for a in ARMS} for h in (1, 2, 3)},
          'full_head_competent': all(means[f'full_bayes@{seed}']['found'] >= .95 and means[f'full_bayes@{seed}']['steps'] <= 1.05 * full['steps'] for seed in FIT_SEEDS)}
    return {'episodes': len(rows), 'regimes': result,
      'readout_pilot_passes': all(r['full_head_competent'] and all(c['passes'] for c in r['criteria'].values()) for r in result.values()),
      'inherited_gate_revised': False, 'learned_architecture_advantage_established': False}


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.clock, self.start, self.launch = None, None, None
        self.receipt = {'status': 'started', 'native_steps_attempted': 0, 'native_steps_returned': 0, 'training_updates': 0,
                        'completed_fits': 0, 'completed_episodes': 0, 'external_model_calls': 0, 'limits': LIMITS}

    def check(self):
        if self.launch:
            require(self.clock.now_ns() < self.launch['deadline_ns'], 'shared native deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= LIMITS['rss_bytes'], 'RSS cap')
        require(sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file()) <= LIMITS['output_bytes'], 'output cap')
        require(self.receipt['native_steps_attempted'] <= LIMITS['native_steps'], 'native call cap')

    def step(self, env, action, hit=None):
        self.check()
        require(self.receipt['native_steps_attempted'] < LIMITS['native_steps'], 'native allocation')
        if self.receipt['phase'] == 'qualification':
            require(self.receipt['native_steps_attempted'] < 2048, 'qualification allocation')
        self.receipt['native_steps_attempted'] += 1
        result = env.step(action, hit=hit, quiet=True)
        self.receipt['native_steps_returned'] += 1
        return result

    def authenticate(self):
        p = self.plan
        require(p['version'] == VERSION and p['configuration'] == CONFIGURATION and p['limits'] == LIMITS and p['status'] == 'frozen_before_run', 'plan identity')
        require(REQUIRED <= p['sources'].keys(), 'complete local source closure')
        for name, pin in p['sources'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts and not (ROOT / name).is_symlink() and sha(ROOT / name) == pin, f'source: {name}')
        upstream = ROOT / 'tmp/otto-source-review-01'
        require(subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() == p['upstream_commit'], 'upstream revision')
        require(not subprocess.check_output(['git', '-C', str(upstream), 'diff', 'HEAD', '--name-only'], text=True).strip(), 'unmodified upstream')
        require(not any(x.endswith('.py') for x in subprocess.check_output(['git', '-C', str(upstream), 'ls-files', '--others', '--exclude-standard'], text=True).splitlines()), 'no untracked upstream Python')
        for name, pin in p['upstream_sources'].items():
            require(sha(upstream / name) == pin, f'upstream source {name}')
        require(sys.version.split()[0] == p['python_version'] and str(Path(sys.executable).absolute()) == p['python_executable'], 'Python identity')
        for name, version in p['runtime_versions'].items():
            require(importlib.metadata.version(name) == version, f'runtime {name}')

    def bind(self):
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, 'supervision missing')
            time.sleep(.01)
        self.launch = json.loads(self.args.supervision.read_text())
        require(sha(self.args.plan) == self.args.plan_sha256, 'external plan digest')
        self.plan = json.loads(self.args.plan.read_text())
        l = self.launch
        command = list(l['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and l['pid'] == os.getpid() and l['pgid'] == os.getpgrp() and l['parent_pid'] == os.getppid(), 'exact supervised process')
        require(l['clock_backend'] == self.clock.backend and l['cap_seconds'] == LIMITS['native_seconds'] and l['started_ns'] <= self.start < l['deadline_ns'] and l['deadline_ns'] == l['started_ns'] + LIMITS['native_seconds'] * 10**9, 'native supervisor clock')
        require(Path(l['cwd']).resolve() == ROOT == Path.cwd().resolve(), 'study cwd')
        require(l['watchdog_sha256'] == self.plan['sources']['scripts/supervise_dialogue_observation_v2.py'] and l['clock_source_sha256'] == self.plan['sources']['src/openjev/research/suspend_clock.py'], 'supervisor source')
        self.authenticate()
        self.receipt.update(plan_sha256=self.args.plan_sha256, supervision_sha256=sha(self.args.supervision), sources=self.plan['sources'])
        write(self.out / 'started.json', {'request': {k: str(v) for k, v in vars(self.args).items()}, 'started_ns': self.start, 'launch': self.launch})

    def collect(self, split, first, count):
        np = self.np
        xs = {a: [] for a in FAMILIES}
        ys, masks, metadata, episodes = [], [], [], []
        with (self.out / f'{split}-transitions.jsonl').open('x') as journal:
            for case in range(count):
                self.check()
                seed, hit = first + case, 1 + case % 3
                self.receipt['active'] = {'phase': split, 'seed': seed}
                env = self.seeded(self.SourceTracking, seed, self.control.COHORTS['base']['config'], initial_hit=hit)
                packet = self.control.public(self.observation(env, 0))
                actors = {a: self.saved.make_actor(a, packet, self.models['base'], self.kernels['base'], self.analytic, np) for a in FAMILIES}
                rng = np.random.default_rng(np.random.SeedSequence([seed, 99]))
                emit(journal, {'kind': 'reset', 'seed': seed, 'initial_hit': hit, 'source_evaluation_only': env.source.tolist(), 'public': packet})
                for step in range(1, COLLECTION_HORIZON + 1):
                    # All family targets come from the same public full-history teacher.
                    belief = actors['full_bayes'].probabilities
                    require(float(np.abs(belief - env.p_source).sum() / 2) <= 1e-10, 'collection full/native parity')
                    scores = self.analytic.action_scores(belief, tuple(packet['position']), self.kernels['base'], True)
                    target, valid = teacher_target(scores, np)
                    for a in FAMILIES:
                        xs[a].append(self.head.features(a, actors[a], packet, hit, sensing_length=3.0))
                    ys.append(target)
                    masks.append(valid)
                    metadata.append((seed, hit, step - 1))
                    teacher_action = self.saved.choose(scores)[0]
                    explore = float(rng.random()) < CONFIGURATION['exploration_probability']
                    action = int(rng.choice(packet['valid_actions'])) if explore else teacher_action
                    self.step(env, action)
                    after = self.control.public(self.observation(env, step))
                    emit(journal, {'kind': 'step', 'seed': seed, 'step': step, 'public': after, 'action': action,
                                   'teacher_action': teacher_action, 'teacher_scores': scores, 'exploration_branch': explore})
                    if not after['done']:
                        for actor in actors.values():
                            actor.update(after)
                    packet = after
                    if packet['done']:
                        break
                episodes.append({'seed': seed, 'initial_hit': hit, 'steps': step, 'found': packet['done']})
                journal.flush()
                if case % 24 == 0:
                    print(json.dumps({'phase': split, 'episodes': case + 1, 'rows': len(ys)}), flush=True)
        arrays = {a: np.stack(values) for a, values in xs.items()}
        for a, x in arrays.items():
            np.savez_compressed(self.out / f'{split}-{a}.npz', features=x)
        target, mask, meta = np.stack(ys), np.stack(masks), np.asarray(metadata, dtype=np.int64)
        np.savez_compressed(self.out / f'{split}-targets.npz', target=target, valid=mask, metadata=meta)
        write(self.out / f'{split}-episodes.json', episodes)
        return arrays, target, mask, meta

    def fit(self, training, validation):
        np, torch = self.np, self.torch
        trainx, target, valid, _ = training
        valx, valtarget, valvalid, _ = validation
        heads, records = {}, []
        self.head_setup_seconds = {}
        target_t, valid_t = torch.from_numpy(target), torch.from_numpy(valid)
        vt, vm = torch.from_numpy(valtarget), torch.from_numpy(valvalid)
        for seed in FIT_SEEDS:
            for family in FAMILIES:
                start = time.perf_counter()
                self.receipt['active'] = {'phase': 'fit', 'seed': seed, 'family': family}
                mean, scale = self.head.fit_standardizer(trainx[family])
                tx = torch.from_numpy(self.head.standardize(trainx[family], mean, scale))
                vx = torch.from_numpy(self.head.standardize(valx[family], mean, scale))
                model = self.head.make_head(tx.shape[1], seed)
                optimizer = torch.optim.Adam(model.parameters(), lr=CONFIGURATION['learning_rate'])
                rng = np.random.default_rng(seed)
                best, best_epoch, exported, curve = math.inf, None, None, []
                for epoch in range(1, CONFIGURATION['epochs'] + 1):
                    model.train()
                    order = rng.permutation(len(tx))
                    total = 0.0
                    for offset in range(0, len(order), CONFIGURATION['batch_size']):
                        self.check()
                        idx = order[offset:offset + CONFIGURATION['batch_size']]
                        logits = (-model(tx[idx])).masked_fill(~valid_t[idx], -1e9)
                        loss = -(target_t[idx] * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
                        require(bool(torch.isfinite(loss)), 'finite training loss')
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'finite gradients')
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                        self.receipt['training_updates'] += 1
                        total += float(loss.detach()) * len(idx)
                    if epoch in CONFIGURATION['checkpoint_epochs']:
                        model.eval()
                        losses, matches = [], 0
                        with torch.no_grad():
                            for offset in range(0, len(vx), 512):
                                scores = model(vx[offset:offset + 512])
                                logits = (-scores).masked_fill(~vm[offset:offset + 512], -1e9)
                                losses.extend((-(vt[offset:offset + 512] * torch.log_softmax(logits, dim=1)).sum(dim=1)).tolist())
                                matches += int((logits.argmax(dim=1) == vt[offset:offset + 512].argmax(dim=1)).sum())
                        ce = math.fsum(losses) / len(losses)
                        require(math.isfinite(ce), 'finite validation CE')
                        if ce < best:
                            best, best_epoch = ce, epoch
                            exported = self.head.export_head(model, mean, scale)
                            # Actual float32 CPU export parity, without another fit.
                            with torch.no_grad():
                                expected = model(vx[:8]).numpy()
                            errors = []
                            for i in range(min(8, len(vx))):
                                actual = self.head.predict(exported, valx[family][i], [0, 1, 2, 3])
                                errors.append(float(np.max(np.abs(np.asarray(actual) - expected[i]))))
                            require(max(errors) <= 2e-5, 'torch/numpy exported head parity')
                        curve.append({'epoch': epoch, 'training_ce': total / len(tx), 'validation_ce': ce, 'validation_argmax_agreement': matches / len(vx)})
                        print(json.dumps({'phase': 'fit', 'family': family, 'seed': seed, **curve[-1]}), flush=True)
                require(exported is not None, 'selected a complete checkpoint')
                name = f'{family}@{seed}'
                path = self.out / f'head-{family}-{seed}.npz'
                np.savez_compressed(path, **exported)
                setup_start = time.perf_counter()
                restored = restore_head(path, np)
                runtime_head = self.head.FrozenHead(restored)
                self.head_setup_seconds[name] = time.perf_counter() - setup_start
                # Evaluation uses the serialized checkpoint, not the final training model.
                require(runtime_head.predict(valx[family][0], [0, 1, 2, 3]) == self.head.predict(exported, valx[family][0], [0, 1, 2, 3]), 'saved checkpoint exact replay')
                heads[name] = runtime_head
                record = {'arm': name, 'family': family, 'seed': seed, 'input_dim': tx.shape[1],
                  'parameters': self.head.parameter_count(tx.shape[1]), 'selected_epoch': best_epoch, 'validation_ce': best,
                  'curve': curve, 'training_rows': len(tx), 'validation_rows': len(vx), 'fit_seconds': time.perf_counter() - start,
                  'checkpoint': path.name, 'checkpoint_sha256': sha(path), 'checkpoint_load_validation_seconds': self.head_setup_seconds[name], 'storage': runtime_head.storage_bytes()}
                records.append(record)
                write(self.out / f'fit-{family}-{seed}.json', record)
                self.receipt['completed_fits'] += 1
        write(self.out / 'fits.json', records)
        return heads

    def evaluate(self, heads):
        np = self.np
        rows = []
        with (self.out / 'evaluation-transitions.jsonl').open('x') as traces, (self.out / 'evaluation-episodes.jsonl').open('x') as journal:
            for regime, seed, block, hit, arm in evaluation_order():
                self.check()
                family, mode = arm.split('@')
                self.receipt['active'] = {'phase': 'evaluation', 'regime': regime, 'seed': seed, 'arm': arm}
                tick = time.perf_counter()
                env = self.seeded(self.SourceTracking, seed, self.control.COHORTS[regime]['config'], initial_hit=hit)
                env_init = time.perf_counter() - tick
                env_seconds = env_init
                packet = self.control.public(self.observation(env, 0))
                tick = time.perf_counter()
                actor = self.saved.make_actor(family, packet, self.models[regime], self.kernels[regime], self.analytic, np)
                init = time.perf_counter() - tick
                storage = actor.storage_bytes()
                evolving = storage.get('state_array_bytes', storage.get('mutable_array_bytes'))
                allocation = self.qualified['shared_model_initialization_seconds'][regime] / CASES if family.startswith('dct') else 0.0
                if mode != 'planner':
                    allocation += self.head_setup_seconds[arm] / CASES
                updating = decision = 0.0
                updates = 0
                emit(traces, {'kind': 'reset', 'regime': regime, 'seed': seed, 'arm': arm, 'initial_hit': hit, 'block': block,
                              'public': packet, 'source_evaluation_only': env.source.tolist()})
                for step in range(1, HORIZON + 1):
                    tick = time.perf_counter()
                    if mode == 'planner':
                        probabilities = actor.decode()[1]
                        scores = self.analytic.action_scores(probabilities, tuple(packet['position']), self.kernels[regime], True)
                    else:
                        feature = self.head.features(family, actor, packet, hit, sensing_length=self.control.COHORTS[regime]['config']['lambda_over_dx'])
                        scores = heads[arm].predict(feature, packet['valid_actions'])
                    action = self.saved.choose(scores)[0]
                    spent = time.perf_counter() - tick
                    decision += spent
                    require(action in packet['valid_actions'], 'legal autonomous action')
                    tick = time.perf_counter()
                    self.step(env, action)
                    env_spent = time.perf_counter() - tick
                    env_seconds += env_spent
                    after = self.control.public(self.observation(env, step))
                    update_spent = 0.0
                    if not after['done']:
                        tick = time.perf_counter()
                        actor.update(after)
                        update_spent = time.perf_counter() - tick
                        updating += update_spent
                        updates += 1
                    emit(traces, {'kind': 'step', 'regime': regime, 'seed': seed, 'arm': arm, 'step': step,
                       'public': after, 'action': action, 'scores': scores, 'decision_seconds': spent,
                       'update_seconds': update_spent, 'environment_seconds': env_spent})
                    packet = after
                    if packet['done']:
                        break
                row = {'regime': regime, 'seed': seed, 'block': block, 'initial_hit': hit, 'arm': arm,
                  'steps': step, 'found': packet['done'], 'updates': updates, 'state_bytes': evolving,
                  'init_seconds': init, 'update_seconds': updating, 'decision_seconds': decision,
                  'setup_allocation_seconds': allocation, 'controller_seconds': init + updating + decision + allocation,
                  'environment_seconds': env_seconds, 'environment_initialization_seconds': env_init, 'storage': storage, 'source_evaluation_only': env.source.tolist(),
                  'draws': [{k: d[k] for k in ('channel', 'index', 'uniform', 'selected_index', 'cdf_mass')} for d in env.draw_log]}
                rows.append(row)
                emit(journal, row)
                journal.flush()
                traces.flush()
                self.receipt['completed_episodes'] = len(rows)
                print(json.dumps({'phase': 'evaluation', 'episodes': len(rows), 'arm': arm, 'steps': step, 'found': packet['done']}), flush=True)
        result = aggregate(rows, self.qualified['initial_hit_weights'])
        result['head_setup_seconds'] = self.head_setup_seconds
        result['head_storage'] = {name: head.storage_bytes() for name, head in heads.items()}
        result['shared_model_storage'] = {k: v.storage_bytes() for k, v in self.models.items()}
        result['shared_model_initialization_seconds'] = self.qualified['shared_model_initialization_seconds']
        result['timing_scope'] = 'Single rotated CPU pass; feature extraction, normalization, head call, legality/tie choice, all updates, actor init and model setup included. One checkpoint load/validation per48case workload is allocated, as is applicable sharedmodel construction; actual setup only once is logically allocated perarm/regime. Per-family immutable weights/standardizer additional to evolving state.'
        write(self.out / 'summary.json', result)
        self.receipt['readout_pilot_passes'] = result['readout_pilot_passes']

    def execute(self):
        primary = None
        try:
            self.clock = SuspendClock()
            self.bind()
            for key, value in THREAD_ENV.items():
                os.environ[key] = value
            import numpy as np
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            self.np, self.torch = np, torch
            sys.path.insert(0, str(ROOT / 'tmp/otto-source-review-01'))
            from isotropic.classes.heuristicpolicy import HeuristicPolicy
            from isotropic.classes.sourcetracking import SourceTracking

            from openjev.research.otto_public import observation, seeded_environment
            self.SourceTracking, self.observation, self.seeded = SourceTracking, observation, seeded_environment
            for name in ('sourcetracking', 'heuristicpolicy', 'policy'):
                require(Path(sys.modules[f'isotropic.classes.{name}'].__file__).resolve() == ROOT / f'tmp/otto-source-review-01/isotropic/classes/{name}.py', 'pinned normal upstream import')
            self.control = load('scripts/study_otto_spectral_control.py', 'action_head_control')
            self.saved = load('scripts/study_otto_spectral_memory.py', 'action_head_saved')
            self.analytic = load('scripts/audit_otto_large_memory.py', 'action_head_analytic')
            spectral = load('src/openjev/research/otto_spectral_memory.py', 'action_head_spectral')
            self.head = load('src/openjev/research/otto_action_head.py', 'action_head_model')
            write(self.out / 'imports.json', {'python': sys.version, 'executable': sys.executable, 'versions': {n: importlib.metadata.version(n) for n in self.plan['runtime_versions']}, 'threads': THREAD_ENV, 'torch_threads': torch.get_num_threads()})
            self.receipt['phase'] = 'qualification'
            self.qualified, self.models, self.kernels = self.control.Run.qualify(self, SourceTracking, HeuristicPolicy, seeded_environment, observation, self.saved, spectral, self.analytic, np)
            self.receipt['qualification_calls'] = self.receipt['native_steps_returned']
            self.receipt['phase'] = 'collection'
            start = time.perf_counter()
            training = self.collect('train', 670001, TRAIN_CASES)
            validation = self.collect('validation', 680001, VALID_CASES)
            self.receipt['collection_seconds'] = time.perf_counter() - start
            self.receipt['collection_rows'] = {'train': len(training[1]), 'validation': len(validation[1])}
            require(set(training[3][:, 0]).isdisjoint(validation[3][:, 0]), 'disjoint train/validation episodes')
            self.receipt['phase'] = 'training'
            heads = self.fit(training, validation)
            del training, validation
            self.receipt['phase'] = 'evaluation'
            self.evaluate(heads)
            self.authenticate()
            require(sha(self.args.plan) == self.args.plan_sha256 and sha(self.args.supervision) == self.receipt['supervision_sha256'], 'unchanged plan/launch')
            require(self.receipt['completed_fits'] == 12 and self.receipt['completed_episodes'] == 2 * CASES * len(ARMS) and self.receipt['native_steps_attempted'] == self.receipt['native_steps_returned'], 'complete work')
            self.check()
            self.receipt['status'] = 'completed'
        except BaseException as error:
            primary = error
            self.receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            failures = []
            try:
                self.receipt.update(started_ns=self.start, finished_ns=self.clock.now_ns(), clock_backend=self.clock.backend)
                self.receipt['wall_seconds'] = None if self.start is None else (self.receipt['finished_ns'] - self.start) / 1e9
            except BaseException as error:  # noqa: BLE001 - preserve primary failure
                self.receipt.update(finished_ns=None, wall_seconds=None)
                failures.append(f'clock finalization: {error!r}')
            try:
                self.receipt['files'] = {str(p.relative_to(self.out)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in self.out.rglob('*') if p.is_file()}
                self.check()
            except BaseException as error:  # noqa: BLE001 - preserve primary failure
                failures.append(f'payload closure or final limits: {error!r}')
            try:
                if failures:
                    self.receipt.update(status='failed', finalization_errors=failures)
                write(self.out / 'receipt.json', self.receipt)
            except BaseException as error:  # noqa: BLE001 - preserve primary failure
                failures.append(f'receipt publication: {error!r}')
                print(json.dumps({'status': 'failed', 'original_error': repr(primary), 'finalization_errors': failures}), file=sys.stderr)
            if failures and primary is None:
                raise RuntimeError('; '.join(failures))
        try:
            self.check()
        except BaseException as error:
            self.receipt.update(status='failed', error=f'late publication: {error!r}')
            try:
                write(self.out / 'late-failure.json', self.receipt)
            except OSError as secondary:
                error.add_note(f'Late failure publication: {secondary!r}')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'supervision', 'output'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    Run(parser.parse_args()).execute()
