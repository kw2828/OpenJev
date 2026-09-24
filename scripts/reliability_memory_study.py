"""Registered causal reliability-memory pilot; preserves every arm and cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from openjev.research import suspend_clock
from openjev.research.finite_head_initialization import make_model
from openjev.research.finite_reliability_filter import ARMS, ReliabilityFilter
from openjev.research.finite_reliability_world import generate

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'output/finite-action-range-study-v1'
PARENT_RECEIPT = ROOT / 'research/finite-action-range-results/receipt.json'
PROTOCOL = 'research/reliability-memory-protocol.md'
TRAINED = ('global', 'markov_bank', 'recurrent_bank', 'reset_bank')
STRATA = ('base', 'shift', 'switch', 'stress')
CONFIG = {'cohorts': 5, 'namespace': 438260924, 'seed': 438261001,
          'train_episodes': 512, 'dev_episodes': 512, 'steps': 32,
          'updates': 256, 'batch_size': 64, 'learning_rate': .01,
          'gradient_clip': 5., 'fit_cap_seconds': 180.}
CAPS = {'qualify': 180, 'run': 1800, 'audit': 300}
TESTS = ['tests/test_finite_reliability_filter.py', 'tests/test_finite_reliability_world.py',
         'tests/test_reliability_memory_study.py']
EXTRA_SOURCES = [PROTOCOL, *TESTS, 'scripts/audit_reliability_memory.py']


def require(value, message):
    if not value:
        raise ValueError(message)


def desc(path):
    data = Path(path).read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def save(path, values):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **values)


def inventory(folder):
    return {str(p.relative_to(folder)): desc(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def runtime():
    return {'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__,
            'torch': torch.__version__, 'platform': platform.platform(),
            'threads': torch.get_num_threads()}


def parent_check():
    receipt = read(PARENT_RECEIPT)
    require(receipt['status'] == 'PASS', 'closed parent publication')
    parent = read(ROOT / 'research/finite-action-range-results/summary.json')
    require(parent['status'] == 'ACTION_RANGE_ADVANCE_FAIL'
            and sum(parent['audit']['advance']['conditions'].values()) == 17,
            'historical parent remains FAIL17/54')
    for name, pin in receipt['sources'].items():
        require(desc(ROOT / name) == pin, 'unchanged inherited source: ' + name)
    result = []
    for i in range(5):
        relative = f'run-01/cohort-{i:02d}/final-rounded_mse-{437261001+i}.npz'
        require(desc(PARENT / relative) == receipt['study_files'][relative], 'fixed original parent checkpoint')
        result.append({'path': str(PARENT / relative), **desc(PARENT / relative)})
    return {'receipt': desc(PARENT_RECEIPT), 'checkpoints': result,
            'status': 'ACTION_RANGE_ADVANCE_FAIL', 'conditions_passed': 17}


def register(folder):
    require(Path.cwd() == ROOT and not folder.exists(), 'exclusive registration in actual checkout')
    require(torch.get_num_threads() == 1, 'one numerical thread')
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.is_relative_to(ROOT / 'src') and path.suffix == '.py':
                paths.add(path)
    paths.update(ROOT / name for name in EXTRA_SOURCES)
    paths.add(ROOT / 'scripts/supervise_dialogue_observation_v2.py')
    sources = {str(p.relative_to(ROOT)): desc(p) for p in sorted(paths)}
    plan = {'version': 'reliability-memory-v1', 'config': CONFIG, 'caps': CAPS,
            'parent': parent_check(), 'runtime': runtime(), 'sources': sources,
            'output': str(folder), 'registered_unix': time.time(),
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    folder.mkdir(parents=True)
    for name in sources:
        target = folder / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    write(folder / 'registration.json', plan)
    print(json.dumps(desc(folder / 'registration.json')))


def validate(folder, sha):
    require(desc(folder / 'registration.json')['sha256'] == sha, 'exact registered plan')
    plan = read(folder / 'registration.json')
    require(plan['config'] == CONFIG and plan['caps'] == CAPS and plan['runtime'] == runtime(),
            'same configuration and runtime')
    require(plan['output'] == str(folder) and Path.cwd() == ROOT, 'registered checkout/output')
    for name, pin in plan['sources'].items():
        require(desc(ROOT / name) == pin and desc(folder / 'sources' / name) == pin,
                'unchanged frozen source: ' + name)
    require(parent_check() == plan['parent'], 'unchanged parent provenance')
    return plan


def closed(folder, phase, sha):
    receipt = read(folder / f'{phase}.receipt.json')
    launch = bind_launch(folder, phase, sha)
    terminal = read(folder / f'{phase}-native-01.terminal.json')
    require(receipt['status'] == 'PASS' and receipt['plan_sha256'] == sha
            and receipt['files'] == inventory(folder / phase), 'complete original phase inventory')
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0
            and not terminal['timed_out'] and terminal['error'] is None
            and terminal['clock_error'] is None and terminal['group_absent']
            and terminal['cleanup']['reaped'] and not terminal['cleanup']['errors']
            and not terminal['cleanup']['signals'] and terminal['cap_seconds'] == CAPS[phase],
            'original native phase closes cleanly')
    for key, value in launch.items():
        require(terminal[key] == value, 'original launch/terminal join: ' + key)
    require(terminal['command'] == receipt['command'] == phase_command(folder, phase, sha),
            'original command joins receipt')
    require(terminal['started_ns'] <= terminal['finished_ns'] < terminal['deadline_ns']
            and terminal['elapsed_ns'] == terminal['finished_ns'] - terminal['started_ns']
            and terminal['wall_seconds'] == terminal['elapsed_ns'] / 1e9,
            'original closure within native deadline')
    return receipt, terminal


def phase_command(folder, phase, sha):
    return [sys.executable, str(Path(__file__).resolve()), '--phase', phase,
            '--output', str(folder), '--plan-sha256', sha,
            '--supervision', str(folder / f'{phase}-native-01.launch.json')]


def bind_launch(folder, phase, sha, *, live=False):
    launch = read(folder / f'{phase}-native-01.launch.json')
    require(launch['version'] == 'dialogue-observation-supervision-v2'
            and launch['command'] == phase_command(folder, phase, sha)
            and launch['cwd'] == str(ROOT) and launch['cap_seconds'] == CAPS[phase],
            'exact registered native launch')
    require(launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
            and launch['clock_source_sha256'] == desc(suspend_clock.__file__)['sha256']
            and launch['watchdog_sha256'] == desc(ROOT / 'scripts/supervise_dialogue_observation_v2.py')['sha256']
            and launch['deadline_ns'] == launch['started_ns'] + CAPS[phase] * 10**9
            and launch['pid'] == launch['pgid'] and launch['pid'] > 0 and launch['parent_pid'] > 0,
            'native source, process and deadline binding')
    if live:
        clock = suspend_clock.SuspendClock()
        require(clock.backend == launch['clock_backend']
                and launch['started_ns'] <= clock.now_ns() < launch['deadline_ns']
                and launch['pid'] == os.getpid() and launch['pgid'] == os.getpgrp()
                and launch['parent_pid'] == os.getppid(), 'actual live native worker')
    return launch


def export_fields(plan, index, folder):
    parent = plan['parent']['checkpoints'][index]
    with np.load(parent['path'], allow_pickle=False) as archive:
        state = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
    model = make_model('rounded_random', 437261001 + index)
    model.load_state_dict(state, strict=True)
    with torch.no_grad():
        fields = model.probability_fields()
        fields = {name: fields[name].detach().clone() for name in ('transition', 'emission', 'hazard')}
        fields['costs'] = model.readout_matrix().detach().clone()
    save(folder / 'backbone.npz', {k: v.numpy() for k, v in fields.items()})
    write(folder / 'backbone.json', {'parent': parent, 'stored_parent_parameters': 352,
                                   'exported_fields': desc(folder / 'backbone.npz')})
    return fields


def save_data(folder, name, data):
    arrays = {group + '/' + key: value for group in ('public', 'targets', 'audit')
              for key, value in data[group].items()}
    save(folder / f'{name}.npz', arrays)
    write(folder / f'{name}.json', data['counts'])


def model_arrays(model):
    return {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}


def batch_order(episodes, updates, size, seed):
    require(episodes % size == 0, 'whole deterministic shuffled epochs')
    rng = np.random.Generator(np.random.PCG64(seed))
    values = []
    while len(values) < updates:
        values.extend(rng.permutation(episodes).reshape(-1, size))
    return np.array(values[:updates], dtype=np.int64)


def train(model, public, batches, folder, config):
    clock = suspend_clock.SuspendClock()
    started = clock.now_ns()
    before = {k: v.detach().clone() for k, v in model.named_buffers()}
    save(folder / f'{model.arm}-initial.npz', model_arrays(model))
    actions = torch.from_numpy(public['actions'])
    observations = torch.from_numpy(public['observations'])
    lengths = torch.from_numpy(public['lengths'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    log = folder / f'{model.arm}-updates.jsonl'
    with log.open('x') as stream:
        for update, indices in enumerate(batches):
            require((clock.now_ns() - started) / 1e9 < config['fit_cap_seconds'], 'fixed per-fit cap')
            ids = torch.from_numpy(indices)
            output = model(actions[ids], observations[ids])
            picked = output['probabilities'].gather(2, observations[ids, :, None]).squeeze(2)
            mask = torch.arange(observations.shape[1])[None] < lengths[ids, None]
            require(bool((picked[mask] > 0).all()), 'positive actual-event probabilities')
            loss = -picked[mask].log().mean()
            require(bool(torch.isfinite(loss)), 'finite training log loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                        for p in model.parameters()), 'finite gradients for every parameter')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'finite accepted parameters')
            stream.write(json.dumps({'update': update + 1, 'loss': float(loss.detach()),
                                     'gradient_norm_before_clip': float(norm),
                                     'event_exposures': int(mask.sum()),
                                     'episode_exposures': len(indices)}) + '\n')
            stream.flush()
    require(all(torch.equal(value, before[name]) for name, value in model.named_buffers()),
            'frozen backbone unchanged')
    save(folder / f'{model.arm}-final.npz', model_arrays(model))
    opt = optimizer.state_dict()
    save(folder / f'{model.arm}-adam.npz', {f'{key}/{name}': value.detach().numpy().copy()
          for key, values in opt['state'].items() for name, value in values.items()})
    write(folder / f'{model.arm}-adam.json', {'param_groups': opt['param_groups']})
    seconds = (clock.now_ns() - started) / 1e9
    require(seconds < config['fit_cap_seconds'], 'complete fit below fixed cap')
    return {'arm': model.arm, 'seed': model.seed, 'updates': len(batches), 'seconds': seconds,
            'parameters': model.parameter_count, 'state_scalars': model.state_count,
            'batches': desc(folder / 'batches.npz'), 'initial': desc(folder / f'{model.arm}-initial.npz'),
            'final': desc(folder / f'{model.arm}-final.npz')}


def metrics(data, predictions):
    public, targets = data['public'], data['targets']
    labels = public['observations']
    n, steps = labels.shape
    event_mask = np.arange(steps)[None] < public['lengths'][:, None]
    alive = event_mask & (labels != 4)
    late = alive & (np.arange(steps)[None] >= 8)
    supported = late.sum(1) > 0
    require(int(supported.sum()) >= 256, 'at least256 late-supported episodes')
    prob = predictions['probabilities']
    require(prob.shape == (n, steps, 5) and np.isfinite(prob).all() and (prob >= 0).all()
            and np.max(np.abs(prob.sum(2) - 1)) < 1e-12, 'valid normalized event predictions')
    chosen_prob = np.take_along_axis(prob, labels[:, :, None], axis=2)[:, :, 0]
    require((chosen_prob[event_mask] > 0).all(), 'positive scored probabilities')
    log_loss = float(-np.log(chosen_prob[event_mask]).mean())
    truth = targets['fork_costs'][:, :, (3, 7), :]
    predicted = predictions['fork_costs']
    require(predicted.shape == truth.shape and np.isfinite(predicted).all(), 'all finite H4/H8 costs')
    regret = np.take_along_axis(truth, predicted.argmin(-1)[..., None], axis=-1)[..., 0] - truth.min(-1)
    denominator = np.maximum(late.sum(1), 1)
    per_episode = (regret * late[:, :, None]).sum(1) / denominator[:, None]
    post_truth, post_pred = targets['post_costs'], predictions['post_costs']
    require(post_pred.shape == post_truth.shape and np.isfinite(post_pred).all(), 'finite immediate costs')
    post_regret = np.take_along_axis(post_truth, post_pred.argmin(-1)[..., None], axis=-1)[..., 0] - post_truth.min(-1)
    result = {'primary_regret': float(per_episode[supported].mean()),
              'h4_regret': float(per_episode[supported, 0].mean()),
              'h8_regret': float(per_episode[supported, 1].mean()),
              'post_regret': float(((post_regret * late).sum(1)[supported] / denominator[supported]).mean()),
              'event_log_loss': log_loss, 'event_count': int(event_mask.sum()),
              'late_episode_support': int(supported.sum()), 'late_unsupported': int((~supported).sum()),
              'late_boundaries': int(late.sum()), 'switch_delays': []}
    switches = data['audit']['switch_indices']
    for delay in (1, 2, 4, 8):
        indices = switches + delay - 1
        selected = np.flatnonzero((switches >= 0) & (indices < steps))
        selected = selected[alive[selected, indices[selected]]]
        result['switch_delays'].append({'delay': delay, 'episodes': len(selected),
            'regret': float(regret[selected, indices[selected]].mean()) if len(selected) else None})
    return result


def qualify(plan, folder):
    for i, command in enumerate((['.venv/bin/ruff', 'check', 'src/openjev/research/finite_reliability_filter.py',
             'src/openjev/research/finite_reliability_world.py', 'scripts/reliability_memory_study.py',
             'scripts/audit_reliability_memory.py', *TESTS],
            [sys.executable, '-m', 'pytest', '-q', *TESTS])):
        with (folder / f'command-{i}.log').open('x') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    probe = folder / 'exposure'
    probe.mkdir()
    fields = export_fields(plan, 0, probe)
    data = generate(949401, 'train', 64, steps=32)
    save_data(probe, 'train', data)
    batches = batch_order(64, 8, 64, 949402)
    save(probe / 'batches.npz', {'indices': batches})
    rows = []
    for arm in TRAINED:
        model = ReliabilityFilter(fields, arm, 949403)
        row = train(model, data['public'], batches, probe, {**CONFIG, 'fit_cap_seconds': 30.})
        row['projected_seconds'] = 2 * row['seconds'] * CONFIG['updates'] / 8
        rows.append(row)
    write(folder / 'feasibility.json', rows)
    require(all(row['projected_seconds'] < 120 for row in rows)
            and 5 * sum(row['projected_seconds'] for row in rows) < 1200,
            'fixed per-fit and complete-training feasibility bounds')
    return {'fits': len(rows), 'train_only': True}


def run(plan, folder):
    models, fit_rows = [], []
    for cohort in range(CONFIG['cohorts']):
        location = folder / f'cohort-{cohort:02d}'
        location.mkdir()
        fields = export_fields(plan, cohort, location)
        data = generate(CONFIG['namespace'] + cohort, 'train', CONFIG['train_episodes'], steps=CONFIG['steps'])
        save_data(location, 'train', data)
        batches = batch_order(CONFIG['train_episodes'], CONFIG['updates'], CONFIG['batch_size'], CONFIG['seed'] + cohort)
        save(location / 'batches.npz', {'indices': batches})
        cohort_models = {arm: ReliabilityFilter(fields, arm, CONFIG['seed'] + cohort) for arm in ARMS}
        order = TRAINED[cohort % len(TRAINED):] + TRAINED[:cohort % len(TRAINED)]
        for arm in order:
            row = {'cohort': cohort, **train(cohort_models[arm], data['public'], batches, location, CONFIG)}
            fit_rows.append(row)
            with (folder / 'fits.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
        for arm in ('unchanged', 'static_bank'):
            save(location / f'{arm}-final.npz', model_arrays(cohort_models[arm]))
        models.append(cohort_models)
    write(folder / 'checkpoint-barrier.json', {'fits': fit_rows, 'checkpoints': {
        str(p.relative_to(folder)): desc(p) for p in sorted(folder.glob('cohort-*/*-final.npz'))}})
    rows = []
    for cohort, cohort_models in enumerate(models):
        location = folder / f'cohort-{cohort:02d}'
        for split in STRATA:
            data = generate(CONFIG['namespace'] + cohort, split, CONFIG['dev_episodes'], steps=CONFIG['steps'])
            save_data(location, split, data)
            actions = torch.from_numpy(data['public']['actions'])
            observations = torch.from_numpy(data['public']['observations'])
            fork_actions = torch.from_numpy(data['targets']['fork_actions'])
            for arm_index in range(len(ARMS)):
                arm = ARMS[(arm_index + cohort + STRATA.index(split)) % len(ARMS)]
                model = cohort_models[arm]
                started = time.perf_counter()
                with torch.no_grad():
                    result = model(actions, observations)
                    forks = model.blind_forks(result['post_states'], fork_actions)
                seconds = time.perf_counter() - started
                predictions = {key: value.numpy().copy() for key, value in result.items()}
                predictions['fork_costs'] = forks['costs'][:, :, (3, 7), :].numpy().copy()
                predictions['fork_survival'] = forks['survival'][:, :, (3, 7)].numpy().copy()
                target = location / f'prediction-{split}-{arm}.npz'
                save(target, predictions)
                row = {'cohort': cohort, 'arm': arm, 'split': split, 'inference_seconds': seconds,
                       'parameters': model.parameter_count, 'state_scalars': model.state_count,
                       'prediction': desc(target), 'data': desc(location / f'{split}.npz'),
                       'checkpoint': desc(location / f'{arm}-final.npz'), **metrics(data, predictions)}
                rows.append(row)
                with (folder / 'results.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
    write(folder / 'summary.json', {'version': 'reliability-memory-v1', 'fits': fit_rows, 'rows': rows,
          'config': CONFIG, 'scope': 'Frozen learned backbones; sample-event NLL training; privileged path-conditioned risk reference.'})
    return {'fits': len(fit_rows), 'rows': len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('register', 'qualify', 'run', 'audit'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--supervision', type=Path)
    args = parser.parse_args()
    folder = args.output.resolve()
    if args.phase == 'register':
        register(folder)
        return
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    output = folder / args.phase
    output.mkdir()
    status, error, result = 'FAILED', None, None
    started = time.perf_counter()
    try:
        plan = validate(folder, args.plan_sha256)
        require(args.supervision == folder / f'{args.phase}-native-01.launch.json', 'fixed original supervision path')
        launch = bind_launch(folder, args.phase, args.plan_sha256, live=True)
        require(launch['command'] == command, 'current argv matches registered launch')
        predecessor = 'run' if args.phase == 'audit' else 'qualify' if args.phase == 'run' else None
        if predecessor:
            _, terminal = closed(folder, predecessor, args.plan_sha256)
            require(terminal['clock_backend'] == launch['clock_backend']
                    and terminal['finished_ns'] <= launch['started_ns'], 'predecessor closed before current phase')
        if args.phase == 'qualify':
            result = qualify(plan, output)
        elif args.phase == 'run':
            result = run(plan, output)
        else:
            from audit_reliability_memory import audit
            result = audit(folder / 'run', output)
        validate(folder, args.plan_sha256)
        status = 'PASS'
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        write(folder / f'{args.phase}.receipt.json', {'status': status, 'error': error,
              'plan_sha256': args.plan_sha256, 'phase': args.phase, 'result': result,
              'seconds': time.perf_counter() - started, 'command': command,
              'files': inventory(output), 'runtime': runtime()})


if __name__ == '__main__':
    main()
