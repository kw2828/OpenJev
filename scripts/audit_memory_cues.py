"""Posthoc cue-visibility audit of existing intact MiniGrid evaluation episodes.

This replays the same development worlds with unchanged final policies. It does
not train, choose checkpoints or provide independent confirmation. Privileged
cue coordinates/type are used only to audit visibility in the actual partial
observation; they never enter policy inputs or action selection.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from minigrid.core.constants import OBJECT_TO_IDX

from openjev.research.memory_env import OBS_SIZE, MemoryBatch
from openjev.research.predictive_memory import PredictiveMemory

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('current_ppo', 'recurrent_ppo', 'reward_prediction', 'world_prediction')
TRAINING_SEEDS = (17, 29, 43)
SIZES = (11, 17, 23)
BATCH_SIZE = 16
EPISODES = 128
MAX_STEPS = 128
EVALUATION_SEED_START = 8_000_000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def cue_visible(env, encoded_observation):
    """Equivalent to agent_sees, using the exact already-collected observation.

    Native image indexing is [view_x, view_y, channel]. The encoded object group
    is the first 11 values of each cell's 20 features. Comparing only that
    observed object category reproduces MiniGrid's agent_sees type comparison.
    Avoiding a second gen_obs call makes this instrumentation inexpensive.
    """
    cue_x, cue_y = 1, env.height // 2 - 1
    relative = env.relative_coords(cue_x, cue_y)
    if relative is None:
        return False
    view_x, view_y = relative
    cue = env.grid.get(cue_x, cue_y)
    if cue is None or cue.type not in ('key', 'ball'):
        raise ValueError('Native Memory cue changed unexpectedly')
    begin = (view_x * env.agent_view_size + view_y) * 20
    observed_type = int(np.argmax(encoded_observation[begin:begin+11]))
    return observed_type == OBJECT_TO_IDX[cue.type]


def load_records(run_dir):
    completed = json.loads((run_dir / 'completed.json').read_text())
    training = json.loads((run_dir / 'training-completed.json').read_text())
    expected_fits = {(arm, seed) for arm in ARMS for seed in TRAINING_SEEDS}
    if (completed.get('status') != 'completed' or completed.get('fits') != 12
            or training.get('status') != 'completed' or len(training['fits']) != 12
            or {(row['arm'], row['seed']) for row in training['fits']} != expected_fits):
        raise ValueError('Audit requires all 12 completed frozen training fits')
    records = []
    for arm, seed in sorted(expected_fits):
        checkpoint = run_dir / 'fits' / f'{arm}-{seed}' / 'model.pt'
        fit = json.loads((checkpoint.parent / 'completed.json').read_text())
        path = run_dir / 'evaluation' / f'{arm}-{seed}-intact.json'
        record = json.loads(path.read_text())
        checkpoint_hash = sha(checkpoint)
        if (record['arm'], record['seed'], record['mode']) != (arm, seed, 'intact'):
            raise ValueError('Intact evaluation identity mismatch')
        if (fit.get('status') != 'completed' or fit['arm'] != arm or fit['seed'] != seed
                or fit['checkpoint_sha256'] != checkpoint_hash
                or record['checkpoint_sha256'] != checkpoint_hash):
            raise ValueError('Checkpoint does not match fit and evaluation receipts')
        if len(record['results']) != 3 or {row['size'] for row in record['results']} != set(SIZES):
            raise ValueError('Expected exactly the three original evaluation sizes')
        for result in record['results']:
            first = EVALUATION_SEED_START + result['size'] * 10000
            if [row['seed'] for row in result['episodes']] != list(range(first, first+EPISODES)):
                raise ValueError('Expected the original 128 ordered evaluation seeds per size')
        records.append({'arm': arm, 'seed': seed, 'checkpoint': checkpoint,
                        'checkpoint_sha256': checkpoint_hash, 'evaluation_path': path,
                        'evaluation_sha256': sha(path), 'results': record['results']})
    return completed, records


def verify_episode(actual, expected):
    for field in ('seed', 'length', 'success', 'wrong_goal', 'timeout'):
        if actual[field] != expected[field]:
            raise ValueError(f'Episode {actual["seed"]} replay mismatch for {field}: '
                             f'{actual[field]!r} versus {expected[field]!r}')
    if not math.isclose(actual['return'], expected['return'], rel_tol=0, abs_tol=1e-9):
        raise ValueError(f'Episode {actual["seed"]} native return replay mismatch')


@torch.inference_mode()
def audit_size(model, arm, size, expected_rows):
    rows = []
    for offset in range(0, len(expected_rows), BATCH_SIZE):
        expected = expected_rows[offset:offset+BATCH_SIZE]
        n = len(expected)
        with MemoryBatch(n, size, expected[0]['seed'], MAX_STEPS) as envs:
            observations = envs.reset()
            previous = torch.zeros(n, dtype=torch.long)
            state = torch.zeros(n, model.hidden)
            reset = torch.ones(n, dtype=torch.bool)
            active = np.ones(n, dtype=bool)
            visible_before = np.zeros(n, dtype=bool)
            diagnostics = [{'first_seen_step': None, 'last_seen_step': None,
                            'initial_cue_seen': False, 'visible_observations': 0,
                            'cue_reacquisitions': 0, 'action_counts': [0]*7} for _ in range(n)]
            for step in range(MAX_STEPS):
                for i in np.flatnonzero(active):
                    visible = cue_visible(envs.envs[i], observations[i])
                    row = diagnostics[i]
                    if visible:
                        if row['first_seen_step'] is None:
                            row['first_seen_step'] = step
                        elif not visible_before[i]:
                            row['cue_reacquisitions'] += 1
                        row['last_seen_step'] = step
                        row['visible_observations'] += 1
                        if step == 0:
                            row['initial_cue_seen'] = True
                    visible_before[i] = visible
                # Exact original inputs and batching. Audit fields do not enter here.
                logits, _, state = model.observe(torch.from_numpy(observations), previous,
                                                  state, reset, arm == 'current_ppo')
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite policy output during replay')
                actions = logits.argmax(-1)
                for i in np.flatnonzero(active):
                    diagnostics[i]['action_counts'][int(actions[i])] += 1
                following, _, terminal, timeout, infos = envs.step(actions.numpy())
                for i, info in enumerate(infos):
                    if not active[i] or 'episode' not in info:
                        continue
                    episode = info['episode']
                    actual = {**episode, 'wrong_goal': bool(terminal[i] and not episode['success']),
                              'timeout': bool(timeout[i] and not terminal[i])}
                    verify_episode(actual, expected[i])
                    diagnostic = diagnostics[i]
                    seen = diagnostic['first_seen_step'] is not None
                    rows.append({**actual, **diagnostic, 'cue_seen': seen,
                                 'sought_cue_after_start': bool(seen and not diagnostic['initial_cue_seen']),
                                 'steps_since_last_seen_at_final_action': (
                                     step-diagnostic['last_seen_step'] if seen else None)})
                    if sum(diagnostic['action_counts']) != episode['length']:
                        raise ValueError('Action counts do not cover the exact replayed episode')
                    active[i] = False
                observations, previous = following, actions
                reset = torch.from_numpy(terminal | timeout)
                if not active.any():
                    break
            if active.any():
                raise ValueError('A replayed first episode failed to finish at the native cap')
    rows.sort(key=lambda row: row['seed'])
    if [row['seed'] for row in rows] != [row['seed'] for row in expected_rows]:
        raise ValueError('Replayed episode identities changed')
    return rows


def summarize(rows):
    if not rows:
        raise ValueError('Cannot summarize an empty audit')
    seen = [row for row in rows if row['cue_seen']]
    unseen = [row for row in rows if not row['cue_seen']]
    histogram = {'initial_step_0': 0, 'steps_1_7': 0, 'steps_8_15': 0,
                 'steps_16_31': 0, 'steps_32_63': 0, 'steps_64_127': 0, 'never': 0}
    for row in rows:
        step = row['first_seen_step']
        if step is None:
            key = 'never'
        elif step == 0:
            key = 'initial_step_0'
        elif step <= 7:
            key = 'steps_1_7'
        elif step <= 15:
            key = 'steps_8_15'
        elif step <= 31:
            key = 'steps_16_31'
        elif step <= 63:
            key = 'steps_32_63'
        else:
            key = 'steps_64_127'
        histogram[key] += 1
    return {'episodes': len(rows), 'cue_seen_count': len(seen), 'cue_not_seen_count': len(unseen),
            'cue_seen_fraction': len(seen)/len(rows),
            'initial_cue_seen_fraction': sum(row['initial_cue_seen'] for row in rows)/len(rows),
            'sought_cue_after_start_fraction': sum(row['sought_cue_after_start'] for row in rows)/len(rows),
            'success_count': sum(row['success'] for row in rows),
            'success': sum(row['success'] for row in rows)/len(rows),
            'success_seen_count': sum(row['success'] for row in seen),
            'success_not_seen_count': sum(row['success'] for row in unseen),
            'success_given_seen': sum(row['success'] for row in seen)/len(seen) if seen else None,
            'success_given_not_seen': sum(row['success'] for row in unseen)/len(unseen) if unseen else None,
            'first_seen_step_histogram': histogram,
            'total_action_counts': [sum(row['action_counts'][action] for row in rows) for action in range(7)],
            'mean_steps_since_last_seen_at_final_action': (
                float(np.mean([row['steps_since_last_seen_at_final_action'] for row in seen])) if seen else None),
            'episodes_with_cue_reacquisition': sum(row['cue_reacquisitions'] > 0 for row in rows)}


def audit(run_dir, out):
    started_path = out.with_suffix('.started.json')
    progress_path = out.with_suffix('.progress.jsonl')
    if out.suffix.lower() != '.json':
        raise ValueError('Output must have a .json extension')
    if any(path.exists() for path in (out, started_path, progress_path)):
        raise FileExistsError('Audit output, start receipt and progress log must all be fresh')
    completed, records = load_records(run_dir)
    provenance = {
        'study': 'recurrent-world-v1', 'kind': 'posthoc_same_episode_cue_visibility_diagnostic',
        'plan_sha256': completed['plan_sha256'],
        'run_completed_sha256': sha(run_dir / 'completed.json'),
        'dependencies': {name: importlib.metadata.version(name)
                         for name in ('minigrid', 'gymnasium', 'torch', 'numpy')},
        'source_sha256': {name: sha(ROOT / name) for name in
                          ('scripts/audit_memory_cues.py', 'src/openjev/research/memory_env.py',
                           'src/openjev/research/predictive_memory.py')},
        'visibility_definition': 'Cue category visible in the exact partial observation before an action',
        'step_indexing': 'Initial policy observation is step0; terminal observation is not acted on or counted',
        'privileged_information': 'Cue coordinates/type used for audit only, never policy input',
        'batch_size': BATCH_SIZE, 'max_steps': MAX_STEPS, 'new_training': False,
        'independent_confirmation': False,
        'limitations': [
            'Posthoc analysis of the same primary development episodes, not new evaluation worlds',
            'Seeing the cue does not prove that the policy remembers or uses its identity',
            'Conditional success is descriptive, not the causal effect of revealing the cue',
            'Sought cue means it became visible after the first observation; intentional seeking is not inferred',
            'Pooled episode counts repeat the same worlds across training seeds and arms',
        ],
    }
    write_new(started_path, {**provenance, 'status': 'started', 'started_unix': time.time()})
    results = []
    begin = time.perf_counter()
    with progress_path.open('x') as progress:
        for record in records:
            model = PredictiveMemory(OBS_SIZE, 64)
            model.load_state_dict(torch.load(record['checkpoint'], map_location='cpu', weights_only=True))
            model.eval()
            if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
                raise ValueError('Checkpoint contains nonfinite weights')
            for result in record['results']:
                rows = audit_size(model, record['arm'], result['size'], result['episodes'])
                report = {'arm': record['arm'], 'training_seed': record['seed'], 'size': result['size'],
                          'checkpoint_sha256': record['checkpoint_sha256'],
                          'evaluation_sha256': record['evaluation_sha256'],
                          'summary': summarize(rows), 'episodes': rows, 'all_original_outcomes_matched': True}
                results.append(report)
                progress.write(json.dumps(report, allow_nan=False)+'\n')
                progress.flush()
                print(json.dumps({key: report[key] for key in ('arm', 'training_seed', 'size', 'summary')}),
                      flush=True)
            if (sha(record['checkpoint']) != record['checkpoint_sha256']
                    or sha(record['evaluation_path']) != record['evaluation_sha256']):
                raise ValueError('Checkpoint or evaluation receipt changed during audit')
    if len(results) != 36 or sum(len(row['episodes']) for row in results) != 4608:
        raise ValueError('Audit did not cover exactly all original intact policy episodes')
    groups = [{'arm': arm, 'size': size,
               'summary': summarize([row for result in results if result['arm'] == arm and result['size'] == size
                                     for row in result['episodes']])} for arm in ARMS for size in SIZES]
    output = {**provenance, 'status': 'completed', 'fits': 12, 'episodes_replayed': 4608,
              'all_original_outcomes_matched': True, 'per_fit_size': results, 'pooled_by_arm_size': groups,
              'overall': summarize([row for result in results for row in result['episodes']]),
              'elapsed_seconds': time.perf_counter()-begin, 'progress_sha256': sha(progress_path)}
    write_new(out, output)
    print(json.dumps({'out': str(out), 'episodes_replayed': 4608,
                      'all_original_outcomes_matched': True, 'overall': output['overall']}, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    audit(args.run, args.out)


if __name__ == '__main__':
    main()
