"""Frozen public-history headroom diagnostic; no training or model selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import reliability_memory_study as parent_study
import torch

from openjev.research import suspend_clock
from openjev.research.finite_observation_world import world
from openjev.research.finite_reliability_filter import ARMS as OLD_ARMS
from openjev.research.finite_reliability_filter import ReliabilityFilter
from openjev.research.finite_schedule_filter import ScheduleFilter, schedule_bank
from openjev.research.finite_schedule_world import generate

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'output/reliability-memory-v1'
PARENT_SHA = '6ccfbc7cb0c7204e9a6c922c5c6a1d7cf4ceca799aba98813f3b1f5a0fce446c'
PUBLICATION = ROOT / 'research/reliability-memory-results/receipt.json'
PUBLICATION_SHA = 'e0292ea6866156addd6799d25f58b69d5355236ce1d215486bd8ab5f9082470b'
CONFIG = {'cohorts': 5, 'namespace': 439260924, 'episodes': 512, 'steps': 32}
CAPS = {'qualify': 180, 'run': 900, 'audit': 300}
ARMS = (*OLD_ARMS, 'learned_exact', 'true_exact', 'learned_static2', 'true_static2')
CONTROLS = (*OLD_ARMS, 'learned_static2')
TESTS = ['tests/test_finite_schedule_filter.py', 'tests/test_finite_schedule_world.py',
         'tests/test_schedule_headroom_study.py', 'tests/test_schedule_headroom_audit.py']
EXTRA = ['research/schedule-headroom-protocol.md', 'scripts/audit_schedule_headroom.py', *TESTS]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def desc(path):
    value = Path(path).read_bytes()
    return {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def save(path, arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def load(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def inventory(folder):
    return {str(p.relative_to(folder)): desc(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def runtime():
    return {'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__,
            'torch': torch.__version__, 'platform': platform.platform(), 'threads': torch.get_num_threads()}


def parent_check():
    require(desc(PUBLICATION)['sha256'] == PUBLICATION_SHA, 'pinned completed parent publication')
    receipt = read(PUBLICATION)
    require(receipt['status'] == 'PASS' and receipt['scientific_status'] == 'FAIL', 'parent failure unchanged')
    parent_study.validate(PARENT, PARENT_SHA)
    for phase in ('qualify', 'run', 'audit'):
        parent_study.closed(PARENT, phase, PARENT_SHA)
    old = read(PARENT / 'audit/audit.json')
    require(old['result']['status'] == 'FAIL' and sum(old['conditions'].values()) == 8, 'parent remains FAIL8/13')
    inputs = {}
    for cohort in range(5):
        for name in ('backbone.npz', *(arm + '-final.npz' for arm in OLD_ARMS)):
            relative = f'run/cohort-{cohort:02d}/{name}'
            pin = desc(PARENT / relative)
            require(pin == receipt['study_files'][relative], 'original final model/field provenance')
            inputs[relative] = pin
    return {'publication': desc(PUBLICATION), 'plan_sha256': PARENT_SHA,
            'conditions_passed': 8, 'scientific_status': 'FAIL', 'inputs': inputs}


def register(folder):
    require(Path.cwd() == ROOT and not folder.exists(), 'exclusive registration in actual checkout')
    require(torch.get_num_threads() == 1, 'one numerical thread')
    paths = {Path(__file__).resolve(), Path(parent_study.__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(ROOT / 'src') and path.suffix == '.py':
                paths.add(path)
    paths.update(ROOT / name for name in EXTRA)
    paths.add(ROOT / 'scripts/supervise_dialogue_observation_v2.py')
    sources = {str(p.relative_to(ROOT)): desc(p) for p in sorted(paths)}
    plan = {'version': 'schedule-headroom-v1', 'config': CONFIG, 'caps': CAPS,
            'arms': list(ARMS), 'sources': sources, 'runtime': runtime(), 'parent': parent_check(),
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
    require(plan['config'] == CONFIG and plan['caps'] == CAPS and plan['arms'] == list(ARMS)
            and plan['runtime'] == runtime(), 'same configuration and runtime')
    require(plan['output'] == str(folder) and Path.cwd() == ROOT, 'registered checkout/output')
    for name, pin in plan['sources'].items():
        require(desc(ROOT / name) == pin and desc(folder / 'sources' / name) == pin,
                'unchanged frozen source: ' + name)
    require(parent_check() == plan['parent'], 'unchanged original parent')
    return plan


def phase_command(folder, phase, sha):
    return [sys.executable, str(Path(__file__).resolve()), '--phase', phase,
            '--output', str(folder), '--plan-sha256', sha,
            '--supervision', str(folder / f'{phase}-native-01.launch.json')]


def bind_launch(folder, phase, sha, *, live=False):
    launch = read(folder / f'{phase}-native-01.launch.json')
    require(launch['version'] == 'dialogue-observation-supervision-v2'
            and launch['command'] == phase_command(folder, phase, sha)
            and launch['cwd'] == str(ROOT) and launch['cap_seconds'] == CAPS[phase], 'exact native launch')
    require(launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
            and launch['clock_source_sha256'] == desc(suspend_clock.__file__)['sha256']
            and launch['watchdog_sha256'] == desc(ROOT / 'scripts/supervise_dialogue_observation_v2.py')['sha256']
            and launch['deadline_ns'] == launch['started_ns'] + CAPS[phase] * 10**9
            and launch['pid'] == launch['pgid'] and launch['pid'] > 0 and launch['parent_pid'] > 0,
            'native source, clock and process binding')
    if live:
        clock = suspend_clock.SuspendClock()
        require(clock.backend == launch['clock_backend']
                and launch['started_ns'] <= clock.now_ns() < launch['deadline_ns']
                and launch['pid'] == os.getpid() and launch['pgid'] == os.getpgrp()
                and launch['parent_pid'] == os.getppid(), 'actual live native worker')
    return launch


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
    require(terminal['command'] == receipt['command'] == phase_command(folder, phase, sha), 'receipt command joins')
    require(terminal['started_ns'] <= terminal['finished_ns'] < terminal['deadline_ns']
            and terminal['elapsed_ns'] == terminal['finished_ns'] - terminal['started_ns']
            and terminal['wall_seconds'] == terminal['elapsed_ns'] / 1e9, 'original deadline arithmetic')
    return receipt, terminal


def field_view(raw):
    t, h = raw['transition'], raw['hazard']
    return {'A': t * (1 - h[:, :, None]), 'found': (t * h[:, :, None]).sum(1),
            'emission': raw['emission'].copy(), 'costs': raw['costs'].copy()}


def construct(location, cohort):
    raw = load(location / 'backbone.npz')
    fields = {key: torch.from_numpy(value) for key, value in raw.items()}
    models = {}
    for arm in OLD_ARMS:
        model = ReliabilityFilter(fields, arm, 438261001 + cohort)
        state = load(location / f'state-{arm}.npz')
        model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()}, strict=True)
        model.eval()
        require(all(np.array_equal(v.detach().numpy(), state[k]) for k, v in model.state_dict().items()),
                'original final state loaded exactly')
        models[arm] = model
    bank, prior = schedule_bank(CONFIG['steps'])
    actual = world(.12)
    actual = {k: actual[k] for k in ('A', 'found', 'emission', 'costs')}
    for label, law in (('learned', field_view(raw)), ('true', actual)):
        models[label + '_exact'] = ScheduleFilter(law, bank, prior)
        models[label + '_static2'] = ScheduleFilter(law, bank[:2], np.full(2, .5, dtype=np.float64))
    return models


def predict(model, arm, data):
    public, target = data['public'], data['targets']
    if arm in OLD_ARMS:
        actions = torch.from_numpy(public['actions'])
        observations = torch.from_numpy(public['observations'])
        forks = torch.from_numpy(target['fork_actions'])
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        started = time.perf_counter()
        with torch.no_grad():
            output = model(actions, observations)
            blind = model.blind_forks(output['post_states'], forks)
        seconds = time.perf_counter() - started
        require(all(torch.equal(value, before[k]) for k, value in model.state_dict().items()), 'unchanged frozen model')
        result = {k: v.numpy().copy() for k, v in output.items()}
        result['fork_costs'] = blind['costs'][:, :, (3, 7), :].numpy().copy()
        result['fork_survival'] = blind['survival'][:, :, (3, 7)].numpy().copy()
    else:
        started = time.perf_counter()
        result = model.filter(public['actions'], public['observations'])
        blind = model.fork(result['post_states'], target['fork_actions'])
        seconds = time.perf_counter() - started
        result['fork_costs'] = blind['costs'][:, :, (3, 7), :].copy()
        result['fork_survival'] = blind['survival'][:, :, (3, 7)].copy()
    require(0 < seconds < 120, 'bounded complete inference call')
    return result, seconds


def metrics(data, predictions):
    target, public = data['targets'], data['public']
    truth = target['fork_costs'][:, :, (3, 7), :]
    chosen = predictions['fork_costs'].argmin(-1)
    regret = np.take_along_axis(truth, chosen[..., None], axis=-1)[..., 0] - truth.min(-1)
    require(np.isfinite(regret).all() and (regret >= 0).all(), 'finite nonnegative conditional regret')
    late = regret[:, 8:]
    per_episode = late.mean((1, 2))
    post_chosen = predictions['post_costs'].argmin(-1)
    post = np.take_along_axis(target['post_costs'], post_chosen[..., None], axis=-1)[..., 0]
    post -= target['post_costs'].min(-1)
    picked = np.take_along_axis(predictions['probabilities'], public['observations'][..., None], axis=-1)[..., 0]
    mask = np.arange(picked.shape[1])[None, :] < public['lengths'][:, None]
    require((picked[mask] > 0).all() and np.isfinite(picked).all(), 'positive actual-event probabilities')
    families = data['audit']['family']
    return {'primary_regret': float(per_episode.mean()),
            'h4_regret': float(late[:, :, 0].mean()), 'h8_regret': float(late[:, :, 1].mean()),
            'post_regret': float(post[:, 8:].mean()), 'event_nll': float(-np.log(picked[mask]).mean()),
            'event_count': int(mask.sum()), 'found_episodes': int((public['observations'] == 4).any(1).sum()),
            'per_episode_regret': per_episode.tolist(),
            'family_regret': [{'family': f, 'episodes': int((families == f).sum()),
                               'primary_regret': float(per_episode[families == f].mean())} for f in range(4)]}


def classify(rows):
    require(len(rows) == 5 * len(ARMS), 'complete diagnostic row count')
    by = {(r['cohort'], r['arm']): r for r in rows}
    require(set(by) == {(c, a) for c in range(5) for a in ARMS}, 'complete cohort-arm grid')
    means = {a: float(np.mean([by[c, a]['primary_regret'] for c in range(5)])) for a in ARMS}
    candidates = {}
    for candidate in ('learned_exact', 'true_exact'):
        conditions, comparisons = {}, []
        for control in CONTROLS:
            wins = sum(by[c, candidate]['primary_regret'] < by[c, control]['primary_regret'] for c in range(5))
            conditions[control + '/mean_gain10pct'] = means[control] > 0 and means[candidate] <= .9 * means[control]
            conditions[control + '/paired_wins4of5'] = wins >= 4
            comparisons.append({'control': control, 'candidate_mean': means[candidate],
                                'control_mean': means[control], 'paired_wins': wins})
        candidate_base = float(np.mean([by[c, candidate]['family_regret'][0]['primary_regret'] for c in range(5)]))
        original_base = float(np.mean([by[c, 'unchanged']['family_regret'][0]['primary_regret'] for c in range(5)]))
        conditions['base/unchanged/noninferiority'] = candidate_base <= 1.05 * original_base + 1e-6
        candidates[candidate] = {'passed': all(conditions.values()), 'conditions': conditions,
                                 'comparisons': comparisons, 'candidate_base': candidate_base, 'unchanged_base': original_base}
    wins = sum(by[c, 'true_exact']['primary_regret'] < by[c, 'true_static2']['primary_regret'] for c in range(5))
    history = {'mean_gain10pct': means['true_static2'] > 0 and means['true_exact'] <= .9 * means['true_static2'],
               'paired_wins4of5': wins >= 4}
    if candidates['learned_exact']['passed']:
        status = 'LEARNED_MODEL_HEADROOM'
    elif candidates['true_exact']['passed'] and all(history.values()):
        status = 'MODEL_MISMATCH_HEADROOM'
    elif candidates['true_exact']['passed']:
        status = 'TRUE_MODEL_ADVANTAGE_WITHOUT_SCHEDULE_MEMORY'
    else:
        status = 'NO_REGISTERED_HEADROOM'
    return {'classification': status, 'candidates': candidates, 'true_history_conditions': history,
            'true_history_paired_wins': wins, 'means': means}


def copy_models(folder, cohorts):
    pins = {}
    for cohort in range(cohorts):
        location = folder / f'cohort-{cohort:02d}'
        location.mkdir()
        for old, new in [('backbone.npz', 'backbone.npz'), *[(a + '-final.npz', 'state-' + a + '.npz') for a in OLD_ARMS]]:
            source = PARENT / 'run' / f'cohort-{cohort:02d}' / old
            target = location / new
            shutil.copyfile(source, target)
            require(desc(source) == desc(target), 'byte-exact original model copy')
            pins[str(target.relative_to(folder))] = {'source': str(source), **desc(target)}
    write(folder / 'model-copy-barrier.json', pins)


def save_data(location, data):
    save(location / 'data.npz', {group + '/' + key: value for group in ('public', 'targets', 'audit')
                                for key, value in data[group].items()})
    write(location / 'data.json', data['counts'])


def qualify(folder):
    for i, command in enumerate((['.venv/bin/ruff', 'check', __file__, 'scripts/audit_schedule_headroom.py',
              'src/openjev/research/finite_schedule_filter.py', 'src/openjev/research/finite_schedule_world.py', *TESTS],
             [sys.executable, '-m', 'pytest', '-q', *TESTS])):
        with (folder / f'command-{i}.log').open('x') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
    probe = folder / 'exposure'
    probe.mkdir()
    copy_models(probe, 1)
    location = probe / 'cohort-00'
    models = construct(location, 0)
    started = time.perf_counter()
    data = generate(949599, 64, steps=32)
    generation = time.perf_counter() - started
    require(all(row >= 1 for row in np.bincount(data['audit']['family'], minlength=4)), 'probe family support')
    save_data(location, data)
    times = {}
    for arm in ARMS:
        result, seconds = predict(models[arm], arm, data)
        save(location / f'prediction-{arm}.npz', result)
        times[arm] = seconds
    projection = 2 * 5 * (512 / 64) * (generation + sum(times.values()))
    result = {'generation_seconds': generation, 'inference_seconds': times,
              'conservative_projected_seconds': projection, 'episodes': 64, 'effectiveness_scored': False}
    write(folder / 'feasibility.json', result)
    require(projection < 450, 'complete projected computation below half scientific cap')
    return {'probe_episodes': 64, 'models': 10, 'effectiveness_scored': False}


def run(folder):
    copy_models(folder, 5)
    rows = []
    for cohort in range(5):
        location = folder / f'cohort-{cohort:02d}'
        models = construct(location, cohort)
        data = generate(CONFIG['namespace'] + cohort, CONFIG['episodes'], steps=CONFIG['steps'])
        require(all(row >= 64 for row in np.bincount(data['audit']['family'], minlength=4)), 'fixed family support')
        save_data(location, data)
        for arm in ARMS[cohort:] + ARMS[:cohort]:
            predictions, seconds = predict(models[arm], arm, data)
            save(location / f'prediction-{arm}.npz', predictions)
            old = arm in OLD_ARMS
            info = {'kind': 'frozen_reliability' if old else 'exact_schedule' if arm.endswith('exact') else 'static_two_mode',
                    'physics': 'true' if arm.startswith('true_') else 'learned',
                    'backbone': None if arm.startswith('true_') else desc(location / 'backbone.npz'),
                    'checkpoint': desc(location / f'state-{arm}.npz') if old else None,
                    'schedule_hypotheses': None if old else 36 if arm.endswith('exact') else 2,
                    'added_stored_parameters': models[arm].parameter_count if old else 0}
            row = {'cohort': cohort, 'arm': arm, 'inference_seconds': seconds,
                   'state_scalars': models[arm].state_count if old else 288 if arm.endswith('exact') else 16,
                   'source_info': info, **metrics(data, predictions)}
            rows.append(row)
            with (folder / 'rows.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
    result = classify(rows)
    write(folder / 'summary.json', {'version': 'schedule-headroom-v1', 'config': CONFIG, 'rows': rows,
          'result': result, 'training_calls': 0, 'scope': 'Diagnostic supplied prior; additive private-path conditional risk.'})
    return {'rows': 50, 'models': 50, 'training_calls': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=('register', 'qualify', 'run', 'audit'))
    parser.add_argument('--output', required=True, type=Path)
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
    started = time.perf_counter()
    status, error, result = 'FAILED', None, None
    try:
        validate(folder, args.plan_sha256)
        require(args.supervision == folder / f'{args.phase}-native-01.launch.json', 'fixed native path')
        launch = bind_launch(folder, args.phase, args.plan_sha256, live=True)
        require(launch['command'] == command, 'current argv matches registered launch')
        predecessor = {'run': 'qualify', 'audit': 'run'}.get(args.phase)
        if predecessor:
            _, terminal = closed(folder, predecessor, args.plan_sha256)
            require(terminal['clock_backend'] == launch['clock_backend']
                    and terminal['finished_ns'] <= launch['started_ns'], 'predecessor closed before this phase')
        if args.phase == 'qualify':
            result = qualify(output)
        elif args.phase == 'run':
            result = run(output)
        else:
            from audit_schedule_headroom import audit
            result = audit(folder / 'run', output)
        validate(folder, args.plan_sha256)
        status = 'PASS'
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        write(folder / f'{args.phase}.receipt.json', {'status': status, 'error': error, 'phase': args.phase,
              'plan_sha256': args.plan_sha256, 'command': command, 'result': result,
              'files': inventory(output), 'seconds': time.perf_counter() - started, 'runtime': runtime()})


if __name__ == '__main__':
    main()
