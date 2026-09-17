"""Read completed PPO training logs without loading models or running episodes.

The two supported schemas share total objective and policy entropy at positions
0 and 1. Remaining cue-study entries are predictive auxiliary losses; remaining
associative-study entries are actor and critic losses. Objective totals combine
different terms and are not an isolated comparison of policy optimization.

Success windows overlap and contain at most the last 100 completed episodes.
They are training observations, not held-out evaluations or gradient diagnostics.
"""

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

SCHEMAS = {
    'cue': ('total_objective', 'policy_entropy_nats', 'aux_reward_mse',
            'aux_termination_bce', 'aux_latent_mse', 'aux_observation_ce'),
    'associative': ('total_objective', 'policy_entropy_nats', 'ppo_actor_loss', 'critic_mse'),
}
WINDOW = 100
MILESTONES = (65_536, 262_144)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require_integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def require_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')


def snapshot(row, schema):
    return {key: row[key] for key in ('update', 'interactions', 'completed_episodes',
                                    'recent_success', 'recent_return')} | {
        'losses': dict(zip(SCHEMAS[schema], row['mean_losses'], strict=True))}


def audit_fit(directory, schema):
    """Audit immutable completed files; fail if receipts and logs disagree."""
    directory = Path(directory)
    receipt_path, log_path = directory/'completed.json', directory/'learning.jsonl'
    receipt_bytes, log_bytes = receipt_path.read_bytes(), log_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    if receipt.get('status') != 'completed' or digest(log_bytes) != receipt.get('learning_sha256'):
        raise ValueError(f'Completion/log hash mismatch: {directory}')
    if directory.name != f"{receipt['arm']}-{receipt['seed']}":
        raise ValueError(f'Fit directory and receipt identity differ: {directory}')
    rows = [json.loads(line) for line in log_bytes.decode().splitlines()]
    if not rows:
        raise ValueError(f'Empty completed training log: {directory}')
    previous_episodes, previous_elapsed = 0, -1.
    increment = rows[0]['interactions']
    require_integer(increment, 'interactions per update', 1)
    intervals, max_episode_increment = [], 0
    for update, row in enumerate(rows, 1):
        for key in ('update', 'interactions', 'completed_episodes'):
            require_integer(row[key], key)
        if row['update'] != update or row['interactions'] != update*increment:
            raise ValueError(f'Nonsequential update/interaction counters: {directory}')
        episodes = row['completed_episodes']
        if episodes < previous_episodes:
            raise ValueError(f'Completed episode count decreased: {directory}')
        max_episode_increment = max(max_episode_increment, episodes-previous_episodes)
        previous_episodes = episodes
        require_finite(row['elapsed_seconds'], 'elapsed_seconds')
        if row['elapsed_seconds'] < 0 or row['elapsed_seconds'] < previous_elapsed:
            raise ValueError(f'Training clock decreased: {directory}')
        previous_elapsed = row['elapsed_seconds']
        losses = row['mean_losses']
        if len(losses) != len(SCHEMAS[schema]):
            raise ValueError(f'Unexpected {schema} loss schema: {directory}')
        for value in losses:
            require_finite(value, 'logged loss')
        if losses[1] < -1e-7 or losses[1] > math.log(7)+1e-6:
            raise ValueError(f'Invalid seven-action entropy: {directory}')
        if episodes == 0:
            if row['recent_success'] is not None or row['recent_return'] is not None:
                raise ValueError(f'Outcome window before any completed episode: {directory}')
        else:
            for key in ('recent_success', 'recent_return'):
                require_finite(row[key], key)
                if not 0 <= row[key] <= 1:
                    raise ValueError(f'Invalid {key}: {directory}')
            count = row['recent_success']*min(WINDOW, episodes)
            if abs(count-round(count)) > 1e-7:
                raise ValueError(f'Window success fraction is not an episode count: {directory}')
            if (row['recent_success'] == 0) != (row['recent_return'] == 0):
                raise ValueError(f'Success/positive-return window mismatch: {directory}')
            # Union ordinal episode intervals, avoiding double-counting repeated windows.
            lo, hi = max(1, episodes-WINDOW+1), episodes
            if intervals and lo <= intervals[-1][1]+1:
                intervals[-1][1] = max(intervals[-1][1], hi)
            else:
                intervals.append([lo, hi])
    final = rows[-1]
    if (final['interactions'] != receipt['interactions']
            or final['completed_episodes'] != receipt['completed_episodes']):
        raise ValueError(f'Final counters differ from completed receipt: {directory}')
    require_finite(receipt['training_seconds'], 'receipt training_seconds')
    if receipt['training_seconds'] < final['elapsed_seconds']:
        raise ValueError(f'Completion predates final training log: {directory}')
    if receipt_path.read_bytes() != receipt_bytes or log_path.read_bytes() != log_bytes:
        raise ValueError(f'Completed files changed during audit: {directory}')
    covered = sum(hi-lo+1 for lo, hi in intervals)
    observed = [row for row in rows if row['recent_success'] is not None]
    positive = [row for row in observed if row['recent_success'] > 0]
    last_positive_update = positive[-1]['update'] if positive else 0
    return {
        'arm': receipt['arm'], 'seed': receipt['seed'], 'schema': schema,
        'directory': str(directory), 'receipt_sha256': digest(receipt_bytes),
        'learning_sha256': digest(log_bytes), 'checkpoint_sha256_from_receipt': receipt.get('checkpoint_sha256'),
        'checkpoint_loaded_or_verified': False, 'updates': len(rows),
        'interactions': final['interactions'], 'completed_episodes': final['completed_episodes'],
        'training_seconds': receipt['training_seconds'], 'loss_fields': list(SCHEMAS[schema]),
        'validation': {'completion_and_log_hash': True, 'sequential_fixed_interaction_updates': True,
                       'monotonic_completed_episodes_and_clock': True, 'final_receipt_counters': True},
        'snapshots': {'first': snapshot(rows[0], schema), **{
            str(step): next((snapshot(row, schema) for row in rows if row['interactions'] == step), None)
            for step in MILESTONES}, 'final': snapshot(final, schema)},
        'reward_windows': {
            'window_size_completed_episodes': WINDOW, 'logged_nonempty_windows': len(observed),
            'positive_windows': len(positive), 'zero_windows': len(observed)-len(positive),
            'first_logged_positive': snapshot(positive[0], schema) if positive else None,
            'last_logged_positive': snapshot(positive[-1], schema) if positive else None,
            'maximum_logged_success_fraction': max((row['recent_success'] for row in observed), default=None),
            'updates_since_last_positive_window': len(rows)-last_positive_update,
            'covered_completed_episodes': covered,
            'uncovered_completed_episodes': final['completed_episodes']-covered,
            'covered_fraction': covered/final['completed_episodes'] if final['completed_episodes'] else None,
            'maximum_episode_count_increment': max_episode_increment,
            'positive_episodes_exact_if_provably_zero': 0 if not positive and covered == final['completed_episodes'] else None,
            'interpretation': 'Window counts overlap. Positive windows are not unique rewarded episodes. '
                              'Full coverage with no positive window establishes zero rewarded completed episodes.',
        },
    }


def audit_studies(studies):
    """Snapshot completion receipts; never open a log lacking one at scan time."""
    summaries = []
    for schema, directory in studies:
        if schema not in SCHEMAS:
            raise ValueError(f'Unknown schema: {schema}')
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f'Fit root is not a directory: {directory}')
        fit_dirs = sorted(path for path in directory.iterdir() if path.is_dir())
        completed = [path for path in fit_dirs if (path/'completed.json').is_file()]
        incomplete = [path.name for path in fit_dirs if path not in completed]
        fits = [audit_fit(path, schema) for path in completed]
        summaries.append({'schema': schema, 'fits_root': str(directory),
                          'completed_fits_at_snapshot': len(fits),
                          'incomplete_fit_directories_skipped': incomplete, 'fits': fits})
    return {
        'version': 'ppo-training-discovery-audit-v1', 'status': 'provisional_training_log_audit',
        'created_utc': datetime.now(UTC).isoformat(),
        'audit_source_sha256': digest(Path(__file__).read_bytes()), 'studies': summaries,
        'scope': 'Saved training logs from completed fits only; no models or evaluation episodes run',
        'limitations': [
            'This snapshot does not establish whole-study completion or held-out effectiveness',
            'Rolling windows overlap and can omit episodes when more than 100 finish between logged updates',
            'First positive window is an observation boundary, not an exact reward discovery timestamp',
            'Only loss positions 0 and 1 share field meanings across schemas; objective compositions differ',
            'Loss magnitudes and low critic MSE do not establish gradient magnitude or dominance',
            'Raw advantages, actor/entropy gradient norms and individual reward events are absent from these logs',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', action='append', required=True, metavar='SCHEMA=FIT_DIRECTORY',
                        help='Repeat with cue=... and/or associative=...')
    parser.add_argument('--out', type=Path, required=True, help='New provisional JSON path; no overwrite')
    args = parser.parse_args()
    studies = []
    for item in args.study:
        schema, separator, directory = item.partition('=')
        if not separator or not directory or schema not in SCHEMAS:
            parser.error('--study must be cue=PATH or associative=PATH')
        studies.append((schema, Path(directory)))
    if args.out.exists():
        raise FileExistsError(args.out)
    result = audit_studies(studies)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'out': str(args.out.resolve()), 'completed_fits': {
        row['fits_root']: row['completed_fits_at_snapshot'] for row in result['studies']}}))


if __name__ == '__main__':
    main()
