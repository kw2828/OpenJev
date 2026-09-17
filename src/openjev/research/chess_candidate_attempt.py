"""Explicit recovery accounting for the terminal first candidate-study attempt.

This admits exactly the recorded output-pipe failure before any complete fit or
neural evaluation. It does not resume partial weights or regenerate holdouts.
"""
import hashlib
import json
import math
import shutil
from pathlib import Path

OLD_PLAN = 'evidence/chess-candidate-v1/protocol/plan.json'
OLD_EXECUTION = 'runs/chess-candidate-v1/execution'
DATA_MEMBERS = {'analyses.jsonl', 'dev.jsonl', 'shift.jsonl', 'excluded-states.json',
                'games.jsonl', 'started.json', 'completed.json'}
EXPECTED_MEMBERS = {f'data/{name}' for name in DATA_MEMBERS} | {
    'started.json', 'failed.json', 'panels.json', 'baselines.json',
    'training-cache.json', 'action_only-97/learning.jsonl',
}

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def inventory(directory):
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('Nonregular failed-attempt evidence')
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = {
                'sha256': sha(path), 'bytes': path.stat().st_size}
    return result

def audit(root, study):
    root = Path(root)
    old = root / OLD_EXECUTION
    members = inventory(old)
    if set(members) != EXPECTED_MEMBERS:
        raise ValueError('Failed-attempt membership changed or neural output exists')
    read = lambda p: json.loads(Path(p).read_text())
    old_hash = sha(root / OLD_PLAN)
    old_plan = read(root / OLD_PLAN)
    if any(sha(root / p) != h for p, h in old_plan['sources'].items()):
        raise ValueError('Original frozen source changed')
    failed, started = read(old / 'failed.json'), read(old / 'started.json')
    if (failed['status'] != 'failed' or failed['error'] != '[Errno 32] Broken pipe'
            or started['status'] != 'started'
            or failed['plan_sha256'] != old_hash or started['plan_sha256'] != old_hash
            or not math.isfinite(failed['wall_seconds']) or failed['wall_seconds'] < 0):
        raise ValueError('Expected terminal pipe failure is absent')
    records = [json.loads(line) for line in (old / 'action_only-97/learning.jsonl').read_text().splitlines()]
    if len(records) != 128:
        raise ValueError('Partial training budget differs')
    totals = study.computation('action_only', 0, 0)
    for row, step in zip(records, study.schedule(97)[:128], strict=True):
        if (row['step'] != step['step'] or row['epoch'] != 1 or row['examples'] != 128
                or row['root_depth'] != 4 or row['branch_depth'] != 2
                or row['indices_sha256'] != hashlib.sha256(json.dumps(step['indices']).encode()).hexdigest()):
            raise ValueError('Partial optimizer journal differs from frozen schedule')
        for key in ('loss', 'policy_ce', 'value_mse', 'gradient_norm'):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError('Invalid partial training statistic')
        if not math.isclose(row['loss'], row['policy_ce'] + .5 * row['value_mse'], rel_tol=2e-6, abs_tol=1e-6):
            raise ValueError('Partial loss arithmetic differs')
        candidates = row['candidate_evaluations']
        if type(candidates) is not int or candidates < 128:
            raise ValueError('Invalid partial candidate count')
        for key, value in study.computation('action_only', 128, candidates).items():
            if row[key] != value:
                raise ValueError('Partial computation arithmetic differs')
            totals[key] += value
    receipt = read(old / 'data/completed.json')
    if (receipt['status'] != 'completed' or receipt['counts'] != {'dev': 2048, 'shift': 2048}
            or set(receipt['files']) != DATA_MEMBERS - {'completed.json'}
            or any(sha(old / 'data' / p) != h for p, h in receipt['files'].items())):
        raise ValueError('Reusable data receipt changed')
    return {
        'version': 'candidate-output-pipe-recovery-v1',
        'original_plan': OLD_PLAN, 'original_plan_sha256': old_hash,
        'failed_execution': OLD_EXECUTION, 'files': members,
        'failure': failed, 'completed_optimizer_updates': 128,
        'discarded_training_examples': 16384, 'discarded_computation': totals,
        'completed_checkpoints': 0, 'neural_evaluation_rows': 0,
        'reused_data_receipt_sha256': sha(old / 'data/completed.json'),
        'reused_teacher_calls': receipt['teacher_calls'],
        'reused_requested_nodes': receipt['requested_nodes'],
        'policy': 'One separately frozen recovery attempt from fresh initialization. Same models, seeds, data, labels, updates, evaluation panels and gates. Discarded 128 updates count as extra study cost. No partial checkpoint, optimizer or RNG state reused. No further retries or replacement games.',
        'data_scope': 'Byte-identical previously generated development panels; no neural predictions existed at amendment freeze. Data generated once in failed attempt, copied and revalidated in recovery; not a second fresh sample.',
        'timing_scope': 'Failed execution wall time includes data generation, validation, cache construction and partial fit; separate partial-fit time unavailable. Do not add reused data generation time again.',
    }

def copy_data(root, destination, recovery):
    source = Path(root) / recovery['failed_execution'] / 'data'
    expected = {k.removeprefix('data/'): v for k, v in recovery['files'].items() if k.startswith('data/')}
    if inventory(source) != expected:
        raise ValueError('Previously generated panel changed before copy')
    shutil.copytree(source, destination)
    if inventory(destination) != expected:
        raise ValueError('Copied panels differ from the failed attempt')
