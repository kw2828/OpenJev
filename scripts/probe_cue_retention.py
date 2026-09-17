"""Frozen forced-path cue-retention probe for completed cue-memory policies.

Paired worlds differ only in the initial key/ball cue, with stored goals unchanged.
The same forward actions are imposed in both worlds. This diagnostic measures
cue-dependent hidden state and action probabilities, not learned-policy success.
It neither trains a decoder nor fits or changes any model parameter.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from minigrid.core.world_object import Ball, Key

from openjev.research.cue_memory_env import make_cue_env
from openjev.research.memory_env import OBS_SIZE, encode_obs
from openjev.research.predictive_memory import PredictiveMemory

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    'version': 'cue-retention-probe-v1', 'study': 'cue-memory-v1',
    'arms': ['current_ppo', 'recurrent_ppo', 'reward_prediction', 'world_prediction'],
    'training_seeds': [61, 73, 89], 'sizes': [11, 17, 23], 'pairs_per_size': 16,
    'seed_base': 60_000_000, 'size_seed_multiplier': 10000,
    'view_size': 7, 'max_steps': 128, 'torch_threads': 1,
    'path': 'Native forward action2 repeated size-3 times, from x1 east to fork center',
    'intervention': 'Swap key/ball cue only; retain branch objects and original stored reward goals',
    'policy_inputs': 'Partial observation, previous forced action and causal recurrent state',
    'normalized_distance': 'RMS(h_native-h_swapped)/max(sqrt(mean((h_native^2+h_swapped^2)/2)),1e-8)',
    'metrics': ['hidden_l2_distance', 'hidden_rms_distance', 'latent_rms_normalized_distance',
                'policy_probability_l1_distance', 'argmax_disagreement', 'cue_visible_fraction'],
    'current_control': 'Exact hidden-state and policy equality wherever current observations are identical',
    'interpretation': 'Cue-dependent state retention and output sensitivity, not task performance or decoded memory',
    'new_training': False, 'independent_confirmation': False,
}
SOURCES = ['scripts/probe_cue_retention.py', 'src/openjev/research/cue_memory_env.py',
           'src/openjev/research/memory_env.py', 'src/openjev/research/predictive_memory.py',
           'tests/test_probe_cue_retention.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def signature():
    return {'protocol': PROTOCOL, 'source_sha256': {name: sha(ROOT/name) for name in SOURCES},
            'dependencies': {name: importlib.metadata.version(name)
                             for name in ('minigrid', 'gymnasium', 'torch', 'numpy')},
            'python': platform.python_version()}


def paired_environments(size, seeds):
    environments = []
    try:
        for seed in seeds:
            original, swapped = make_cue_env(size, seed), make_cue_env(size, seed)
            environments.extend([original, swapped])
            x, y = 1, size//2-1
            cue = swapped.grid.get(x, y)
            if not isinstance(cue, (Key, Ball)):
                raise TypeError('Expected native key/ball cue')
            swapped.grid.set(x, y, Ball(cue.color) if isinstance(cue, Key) else Key(cue.color))
            changed = np.argwhere(original.grid.encode() != swapped.grid.encode())
            if changed.tolist() != [[x, y, 0]]:
                raise ValueError('Paired maps must differ only in the cue object category')
            if (original.success_pos != swapped.success_pos or original.failure_pos != swapped.failure_pos
                    or original.np_random.bit_generator.state != swapped.np_random.bit_generator.state):
                raise ValueError('Cue swap changed goals or RNG state')
        return environments
    except BaseException:
        for env in environments:
            env.close()
        raise


def metric_means(rows):
    fields = [*PROTOCOL['metrics'], 'observations_identical', 'native_forward_probability',
              'swapped_forward_probability']
    return {name: float(np.mean([row[name] for row in rows])) for name in fields}


@torch.inference_mode()
def probe_size(model, arm, size, seeds=None):
    if arm not in PROTOCOL['arms'] or size not in PROTOCOL['sizes']:
        raise ValueError('Unknown arm or probe size')
    if seeds is None:
        start = PROTOCOL['seed_base']+size*PROTOCOL['size_seed_multiplier']
        seeds = list(range(start, start+PROTOCOL['pairs_per_size']))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Distinct, nonempty probe seeds required')
    environments = paired_environments(size, seeds)
    n = len(environments)
    state = torch.zeros(n, model.hidden)
    previous = torch.zeros(n, dtype=torch.long)
    reset = torch.ones(n, dtype=torch.bool)
    pair_rows = [[] for _ in seeds]
    observations_digest = hashlib.sha256()
    current_only = arm == 'current_ppo'
    try:
        for step in range(size-2):
            observations = np.stack([encode_obs(env.gen_obs()) for env in environments])
            observations_digest.update(observations.tobytes())
            visibility = [env.agent_sees(1, size//2-1) for env in environments]
            logits, _, state = model.observe(torch.from_numpy(observations), previous, state, reset, current_only)
            if not torch.isfinite(state).all() or not torch.isfinite(logits).all():
                raise ValueError('Nonfinite state or policy output')
            probabilities = logits.softmax(-1)
            for pair, seed in enumerate(seeds):
                a, b = pair*2, pair*2+1
                same = np.array_equal(observations[a], observations[b])
                if visibility[a] != visibility[b]:
                    raise ValueError('Changing cue identity changed its visibility')
                if step == 0 and (same or not visibility[a]):
                    raise ValueError('Initial partial observations must expose the different cues')
                if step > 0 and (not same or visibility[a]):
                    raise ValueError('Forced path must hide the cue and equalize all subsequent observations')
                if (current_only and same
                        and (not torch.equal(state[a], state[b]) or not torch.equal(logits[a], logits[b]))):
                    raise ValueError('Current-only policy retains a difference despite identical inputs')
                difference = state[a]-state[b]
                rms = difference.square().mean().sqrt()
                scale = ((state[a].square()+state[b].square())/2).mean().sqrt().clamp_min(1e-8)
                pair_rows[pair].append({
                    'seed': seed, 'step': step, 'x': 1+step,
                    'hidden_l2_distance': float(difference.norm()), 'hidden_rms_distance': float(rms),
                    'latent_rms_normalized_distance': float(rms/scale),
                    'policy_probability_l1_distance': float((probabilities[a]-probabilities[b]).abs().sum()),
                    'argmax_disagreement': bool(logits[a].argmax() != logits[b].argmax()),
                    'cue_visible_fraction': (float(visibility[a])+float(visibility[b]))/2,
                    'observations_identical': bool(same),
                    'native_forward_probability': float(probabilities[a, 2]),
                    'swapped_forward_probability': float(probabilities[b, 2]),
                    'native_argmax': int(logits[a].argmax()), 'swapped_argmax': int(logits[b].argmax()),
                })
            if step == size-3:
                break
            for env in environments:
                _, reward, terminated, truncated, _ = env.step(2)
                if terminated or truncated or reward != 0:
                    raise ValueError('Forced path reached a terminal state before the fork center')
                if tuple(env.agent_pos) != (step+2, size//2) or env.agent_dir != 0:
                    raise ValueError('Forward action did not follow the declared native corridor path')
            previous = torch.full((n,), 2, dtype=torch.long)
            reset = torch.zeros(n, dtype=torch.bool)
        if any(tuple(env.agent_pos) != (size-2, size//2) for env in environments):
            raise ValueError('Forced path did not end at the fork center')
        per_step = [{'step': step, **metric_means([rows[step] for rows in pair_rows])}
                    for step in range(size-2)]
        last_visible = [next(row for row in reversed(rows) if row['cue_visible_fraction'] > 0)
                        for rows in pair_rows]
        retention = [rows[-1]['hidden_rms_distance']/rows[0]['hidden_rms_distance']
                     for rows in pair_rows if rows[0]['hidden_rms_distance'] > 0]
        return {'arm': arm, 'size': size, 'pairs': len(seeds), 'seeds': seeds,
                'forced_actions_per_pair_member': [2]*(size-3),
                'paired_observations_sha256': observations_digest.hexdigest(),
                'initial': metric_means([rows[0] for rows in pair_rows]),
                'last_cue_visible': metric_means(last_visible),
                'last_cue_visible_steps': [row['step'] for row in last_visible],
                'fork': metric_means([rows[-1] for rows in pair_rows]),
                'mean_fork_to_initial_hidden_rms_ratio': float(np.mean(retention)) if retention else None,
                'pairs_with_nonzero_initial_state_difference': len(retention),
                'current_only_equality_check_passed': True if current_only else None,
                'all_post_initial_observations_identical': True,
                'per_step': per_step, 'per_pair': pair_rows}
    finally:
        for env in environments:
            env.close()


def completed_checkpoints(execution, study_plan):
    study = json.loads(study_plan.read_text())
    study_hash = sha(study_plan)
    if study['protocol']['version'] != PROTOCOL['study']:
        raise ValueError('Expected the frozen cue-memory-v1 study plan')
    for name, expected in study['source_hashes'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Frozen study source changed: {name}')
    for filename in ('completed.json', 'core/completed.json'):
        receipt = json.loads((execution/filename).read_text())
        if receipt.get('status') != 'completed' or receipt.get('plan_sha256') != study_hash:
            raise ValueError('Both outer and core study executions must be complete with this plan')
        if (filename == 'completed.json' and receipt.get('additional_evaluations') != 24
                or filename == 'core/completed.json' and receipt.get('fits') != 12):
            raise ValueError('Study completion receipt has the wrong condition count')
    training = json.loads((execution/'core/training-completed.json').read_text())
    expected_fits = {(arm, seed) for arm in PROTOCOL['arms'] for seed in PROTOCOL['training_seeds']}
    if (training.get('status') != 'completed' or len(training['fits']) != 12
            or {(row['arm'], row['seed']) for row in training['fits']} != expected_fits):
        raise ValueError('Wrong completed training identities or count')
    records = []
    for fit in training['fits']:
        directory = execution/'core/fits'/f'{fit["arm"]}-{fit["seed"]}'
        if (json.loads((directory/'completed.json').read_text()) != fit
                or fit['status'] != 'completed'
                or sha(directory/'model.pt') != fit['checkpoint_sha256']
                or sha(directory/'learning.jsonl') != fit['learning_sha256']):
            raise ValueError('Training receipt or checkpoint identity mismatch')
        evaluation = json.loads((execution/'core/evaluation'/f'{fit["arm"]}-{fit["seed"]}-intact.json').read_text())
        if (evaluation['arm'], evaluation['seed'], evaluation['mode']) != (fit['arm'], fit['seed'], 'intact'):
            raise ValueError('Primary evaluation identity mismatch')
        if evaluation['checkpoint_sha256'] != fit['checkpoint_sha256']:
            raise ValueError('Primary evaluation checkpoint mismatch')
        records.append({'arm': fit['arm'], 'seed': fit['seed'], 'checkpoint': directory/'model.pt',
                        'checkpoint_sha256': fit['checkpoint_sha256']})
    return records


def run(plan, study_plan, execution, out):
    if json.loads(plan.read_text()) != signature():
        raise ValueError('Frozen retention probe source, protocol or runtime changed')
    started, progress = out.with_suffix('.started.json'), out.with_suffix('.progress.jsonl')
    if out.suffix.lower() != '.json' or any(path.exists() for path in (out, started, progress)):
        raise ValueError('Probe JSON and its start/progress paths must be fresh')
    fits = completed_checkpoints(execution, study_plan)
    provenance = {'probe_plan_sha256': sha(plan), 'study_plan_sha256': sha(study_plan),
                  'outer_completed_sha256': sha(execution/'completed.json'),
                  'core_completed_sha256': sha(execution/'core/completed.json'), **signature()}
    write_new(started, {**provenance, 'status': 'started', 'started_unix': time.time()})
    begin, results = time.perf_counter(), []
    with progress.open('x') as stream:
        for fit in fits:
            model = PredictiveMemory(OBS_SIZE, 64)
            model.load_state_dict(torch.load(fit['checkpoint'], map_location='cpu', weights_only=True))
            model.eval()
            for size in PROTOCOL['sizes']:
                result = {'training_seed': fit['seed'], 'checkpoint_sha256': fit['checkpoint_sha256'],
                          **probe_size(model, fit['arm'], size)}
                results.append(result)
                stream.write(json.dumps(result, allow_nan=False)+'\n')
                stream.flush()
                print(json.dumps({'arm': fit['arm'], 'seed': fit['seed'], 'size': size,
                                  'initial': result['initial'], 'fork': result['fork']}), flush=True)
            if sha(fit['checkpoint']) != fit['checkpoint_sha256']:
                raise ValueError('Checkpoint changed during retention probe')
    expected = {(arm, seed, size) for arm in PROTOCOL['arms'] for seed in PROTOCOL['training_seeds']
                for size in PROTOCOL['sizes']}
    if len(results) != 36 or {(r['arm'], r['training_seed'], r['size']) for r in results} != expected:
        raise ValueError('Probe did not cover every final checkpoint and size')
    for size in PROTOCOL['sizes']:
        if len({r['paired_observations_sha256'] for r in results if r['size'] == size}) != 1:
            raise ValueError('Forced paired observation sequences differ across models')
    output = {**provenance, 'status': 'completed', 'results': results, 'fits': 12,
              'paired_trajectories': sum(r['pairs'] for r in results),
              'elapsed_seconds': time.perf_counter()-begin, 'progress_sha256': sha(progress),
              'limitations': ['Forced actions bypass the policy, so this is not learned-policy performance',
                              'A nonzero state difference does not prove decodable or usable memory',
                              'Unchanged argmax can coexist with changed action probabilities',
                              'This is a posthoc mechanism diagnostic on separately declared probe worlds']}
    write_new(out, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'run'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--study-plan', type=Path)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(PROTOCOL['torch_threads'])
    if args.command == 'freeze':
        write_new(args.out, signature())
    else:
        run(args.plan, args.study_plan, args.run, args.out)


if __name__ == '__main__':
    main()
