# SPDX-License-Identifier: GPL-3.0-only
"""Fresh v3 attempt after the v2 process disappeared before evaluation.

Scientific code and criteria are unchanged. The isolated v2 engine receives
only a new checkpoint version and plan validator. No prior fit is resumed.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_PLAN = 'evidence/chess-pin-quality-v2/protocol/plan.json'
OLD_PLAN_SHA = '9f9eba11e6abaa9b15c085a0098edbe938e7db848a323a3646f7f9468aadfc48'
OLD_EXECUTION = 'runs/chess-pin-quality-v2/execution'
OLD_LAUNCH = 'evidence/chess-pin-quality-v2/launch.json'
REVIEW = 'evidence/chess-pin-quality-v2/interruption-review.json'
VERSION = 'canonical-matched-pin-quality-v3'
CODE = ['scripts/chess_pin_quality_recovery.py', 'tests/test_chess_pin_quality_recovery.py',
        'scripts/launch_chess_pin_quality.py', 'tests/test_launch_chess_pin_quality.py']


def load_engine():
    spec = importlib.util.spec_from_file_location('_isolated_pin_quality_v2', ROOT/'scripts/chess_pin_quality_study_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_engine()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def tree(directory):
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('Nonregular partial artifact')
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = sha(path)
    return result


def partial_inventory(directory, old, plan_hash):
    """Validate only the retained training prefix, without running predictions."""
    directory = Path(directory)
    if {p.name for p in directory.iterdir()} != {'started.json', 'fits'}:
        raise ValueError('Expected an interrupted pre-evaluation training attempt')
    started = read(directory/'started.json')
    if started['plan_sha256'] != plan_hash:
        raise ValueError('Partial execution plan differs')
    protocol = old['protocol']
    schedule = [(arm, seed, base.name(arm, seed)) for seed in protocol['seeds'] for arm in protocol['arms']]
    present = {p.name for p in (directory/'fits').iterdir()}
    if not present or present != {row[2] for row in schedule[:len(present)]}:
        raise ValueError('Partial fits are not the declared schedule prefix')
    fitted, updates, update_seconds, fit_seconds = [], 0, 0., 0.
    for index, (arm, seed, identity) in enumerate(schedule[:len(present)]):
        folder = directory/'fits'/identity
        files = {p.name for p in folder.iterdir()}
        finished = 'training.json' in files
        expected = {'initial.pt', 'learning.jsonl'} | ({'training.json', 'weights.pt'} if finished else set())
        if files != expected or not finished and index != len(present)-1:
            raise ValueError('Unexpected partial-fit membership or ordering')
        records = base.source.prior.rows(folder/'learning.jsonl')
        if not 0 < len(records) <= protocol['updates_per_fit']:
            raise ValueError('Invalid retained update count')
        if finished and len(records) != protocol['updates_per_fit']:
            raise ValueError('Completed fit has missing updates')
        prefix = {**protocol, 'updates_per_fit': len(records)}
        base.check_learning(records, seed, prefix)
        seconds = math.fsum(r['update_seconds'] for r in records)
        item = {'name': identity, 'arm': arm, 'seed': seed, 'completed': finished,
                'logged_updates': len(records), 'logged_update_seconds': seconds}
        if finished:
            meta = read(folder/'training.json')
            if (meta['status'] != 'completed' or meta['plan_sha256'] != plan_hash
                    or meta['arm'] != arm or meta['seed'] != seed or meta['updates'] != len(records)
                    or meta['examples_seen'] != len(records)*protocol['batch_size']
                    or meta['parameters'] != protocol['parameters'][arm]
                    or not math.isfinite(meta['seconds']) or meta['seconds'] <= 0):
                raise ValueError('Completed partial-fit receipt differs')
            for key, name in [('initial_sha256', 'initial.pt'), ('weights_sha256', 'weights.pt'),
                              ('learning_sha256', 'learning.jsonl')]:
                if meta[key] != sha(folder/name):
                    raise ValueError('Retained fit bytes changed')
            item['completed_fit_seconds'] = meta['seconds']
            fit_seconds += meta['seconds']
        fitted.append(item)
        updates += len(records)
        update_seconds += seconds
    return {'fits': fitted, 'completed_fits': sum(v['completed'] for v in fitted),
            'logged_completed_updates': updates, 'logged_update_seconds': update_seconds,
            'completed_fit_seconds': fit_seconds, 'quality_predictions': 0,
            'evaluation_outputs_absent': True,
            'last_artifact_mtime_unix': max(p.stat().st_mtime for p in directory.rglob('*') if p.is_file()),
            'execution_started_unix': started['unix'], 'pid_at_start': started['pid'],
            'cost_scope': 'Logged update time is a lower bound; completed-fit time overlaps it and must not be added. Exact process end, unlogged work and terminal wall time are unknown.'}


def prior_plan():
    path = ROOT/OLD_PLAN
    if sha(path) != OLD_PLAN_SHA:
        raise ValueError('Prior frozen plan changed')
    old, digest = base.validate_plan(path)
    if digest != OLD_PLAN_SHA:
        raise ValueError('Prior validator identity differs')
    return old


def record_interruption(out):
    old = prior_plan()
    launch = read(ROOT/OLD_LAUNCH)
    pid = launch['pid_at_launch']
    process = subprocess.run(['ps', '-p', str(pid), '-o', 'pid=,lstart=,comm='],
                             capture_output=True, text=True, check=False)
    if process.returncode != 1 or process.stdout.strip():
        raise ValueError('Original PID is present or absence cannot be established')
    before = tree(ROOT/OLD_EXECUTION)
    inventory = partial_inventory(ROOT/OLD_EXECUTION, old, OLD_PLAN_SHA)
    after = tree(ROOT/OLD_EXECUTION)
    if before != after or inventory['pid_at_start'] != pid:
        raise ValueError('Partial evidence changed during preservation')
    value = {'status': 'interrupted_process_missing_before_evaluation', 'observed_unix': time.time(),
        'plan_sha256': OLD_PLAN_SHA, 'launch_sha256': sha(ROOT/OLD_LAUNCH),
        'process_observation': {'pid': pid, 'ps_returncode': process.returncode,
                                'ps_stdout': process.stdout, 'ps_stderr': process.stderr},
        'tool_session_at_launch': launch['tool_session_at_launch'],
        'cause': 'Unknown. The original tool session was separately queried and returned Unknown process id; the PID is absent. No exit receipt survived.',
        'raw_execution_untouched': True, 'frozen_sources_unchanged': True,
        'partial_files_sha256': before, **inventory}
    write(out, value)
    return value


def retained_attempt(old):
    review = read(ROOT/REVIEW)
    expected = partial_inventory(ROOT/OLD_EXECUTION, old, OLD_PLAN_SHA)
    observation = review['process_observation']
    if (review['status'] != 'interrupted_process_missing_before_evaluation'
            or review['plan_sha256'] != OLD_PLAN_SHA or review['launch_sha256'] != sha(ROOT/OLD_LAUNCH)
            or review['partial_files_sha256'] != tree(ROOT/OLD_EXECUTION)
            or review['raw_execution_untouched'] is not True or review['frozen_sources_unchanged'] is not True
            or any(review[k] != v for k, v in expected.items())
            or observation['pid'] != expected['pid_at_start'] or observation['ps_returncode'] != 1
            or observation['ps_stdout'].strip()):
        raise ValueError('Preserved interruption differs')
    return review


def signature():
    old = prior_plan()
    review = retained_attempt(old)
    protocol = copy.deepcopy(old['protocol'])
    protocol['version'] = VERSION
    protocol['prior_knowledge'] += (
        f" V2 disappeared after {review['logged_completed_updates']} logged updates and "
        f"{review['completed_fits']} completed fits, before evaluation. All partial artifacts and costs are retained. "
        'V3 repeats the complete fixed study from fresh initializations with the same recipes, order, data and criteria; '
        'no prior checkpoint is reused or selected. The detached launcher records terminal process status in regular files.')
    return {'protocol': protocol, 'prior_plan_sha256': OLD_PLAN_SHA,
        'prior_signature': {k: v for k, v in old.items() if k != 'prepared_unix'},
        'interruption_review_sha256': sha(ROOT/REVIEW),
        'sources': {path: sha(ROOT/path) for path in CODE},
        'recovery': {'fresh_initializations': True, 'partial_checkpoint_resume': False,
                     'prior_updates_retained': review['logged_completed_updates'],
                     'prior_update_seconds_retained': review['logged_update_seconds'],
                     'prior_fit_seconds_overlapping': review['completed_fit_seconds'],
                     'prior_termination_cause_known': False,
                     'execution_ceiling_seconds': protocol['time_cap_seconds'],
                     'audit_ceiling_seconds': protocol['audit_time_cap_seconds']}}


def prepare(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    plan = signature()
    plan['prepared_unix'] = time.time()
    out.mkdir(parents=True, exist_ok=False)
    write(out/'plan.json', plan)
    print(json.dumps({'prepared': str(out/'plan.json'), 'sha256': sha(out/'plan.json'),
                      'fits': plan['protocol']['fits'], 'updates': plan['protocol']['training_updates']}), flush=True)


def validate_plan(path):
    plan = read(path)
    expected = signature()
    if set(plan) != set(expected) | {'prepared_unix'} or any(plan[k] != v for k, v in expected.items()):
        raise ValueError('Frozen recovery protocol or prior evidence changed')
    if not 0 < plan['prepared_unix'] <= time.time():
        raise ValueError('Invalid recovery timestamp')
    return plan, sha(path)


def execution_engine():
    engine = load_engine()
    engine.VERSION = VERSION
    engine.validate_plan = validate_plan
    return engine


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['record-interruption', 'prepare', 'verify', 'run', 'audit'])
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--execution', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.command == 'record-interruption':
        result = record_interruption(args.out)
        print(json.dumps({k: result[k] for k in ('status', 'completed_fits', 'logged_completed_updates')}))
    elif args.command == 'prepare':
        prepare(args.out)
    elif args.command == 'verify':
        plan, digest = validate_plan(args.plan)
        print(json.dumps({'verified': True, 'sha256': digest, 'fits': plan['protocol']['fits']}))
    elif args.command == 'run':
        execution_engine().run(args.plan, args.out)
    else:
        execution_engine().audit(args.plan, args.execution, args.out)
