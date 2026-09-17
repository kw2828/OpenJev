"""Bounded supervised four-context learnability diagnostic, separate from RL.

Native cue-visible partial observations provide an initial cue and two visible
fork objects. A teacher derives the final left/right turn from those observations
alone. Training uses complete forced-forward sequences and final-turn cross
entropy. Final probes reuse the same four selected seeds at several corridor
lengths. These are not unseen unique maps, gameplay results or RL-trained weights.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.envs import MemoryEnv
from torch.nn import functional as F

from openjev.research.associative_policy import MODES, AssociativePolicy
from openjev.research.cue_memory_env import make_cue_env
from openjev.research.memory_env import encode_obs

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    'version': 'associative-learnability-v1', 'modes': list(MODES), 'seeds': [101, 113, 127],
    'train_size': 11, 'evaluation_sizes': [11, 17, 23], 'updates': 500, 'batch_size': 4,
    'optimizer': 'Adam', 'learning_rate': .001, 'betas': [.9, .999], 'epsilon': 1e-8,
    'weight_decay': 0., 'gradient_clipping': None, 'torch_threads': 1,
    'hidden': 64, 'store_dim': 16, 'adapter_hidden': 33,
    'training_context_seed_start': 70_000_000, 'training_seed_search_limit': 10000,
    'context_selection': 'First native training-size seed for each cue/upper-branch key/ball combination',
    'forced_path': 'Action2 from x1 east to fork; final target0(top) or1(bottom) from visible objects',
    'objective': 'Final-observation seven-action cross entropy; full sequence gradients',
    'batch': 'All four unique balanced native contexts in fixed canonical order at every update',
    'evaluation': 'Final checkpoint only; reuse the exact four selected training seeds at each size',
    'conditions': ['intact', 'reset_all', 'reset_store'], 'fit_order_seed': 731017,
    'state_ablation': 'reset_all clears GRU/store each step; reset_store clears only matrix before each write',
    'selection': 'No early stopping, checkpoint selection, budget extension or accuracy-based retuning',
    'scope': 'Supervised choice learnability and forced-path length transfer, not gameplay or RL',
    'rl_warmstart': False, 'unseen_unique_map_test': False, 'novelty_established': False,
}
SOURCES = ['scripts/associative_learnability.py', 'tests/test_associative_learnability.py',
           'src/openjev/research/associative_policy.py', 'src/openjev/research/predictive_memory.py',
           'src/openjev/research/cue_memory_env.py', 'src/openjev/research/memory_env.py']
KEY, BALL = OBJECT_TO_IDX['key'], OBJECT_TO_IDX['ball']
CONTEXTS = tuple((cue, upper) for cue in (KEY, BALL) for upper in (KEY, BALL))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def teacher_choice(initial_observation, fork_observation):
    """Read only observable cue/branch categories in east-facing native images.

    The initial image must contain exactly one key/ball. At the fork the two
    branch objects occupy native view positions (1,6) and (5,6), above and below
    the agent respectively. These are partial-image positions, not hidden map
    coordinates or stored reward-goal fields.
    """
    if initial_observation['direction'] != 0 or fork_observation['direction'] != 0:
        raise ValueError('The declared forced path faces east at both endpoints')
    initial, fork = np.asarray(initial_observation['image']), np.asarray(fork_observation['image'])
    if initial.shape != (7, 7, 3) or fork.shape != (7, 7, 3):
        raise ValueError('Expected native partial 7x7 images')
    first_locations = np.argwhere(np.isin(initial[..., 0], [KEY, BALL]))
    final_locations = np.argwhere(np.isin(fork[..., 0], [KEY, BALL]))
    if len(first_locations) != 1 or final_locations.tolist() != [[1, 6], [5, 6]]:
        raise ValueError('The cue and both branch objects must be observable at the declared endpoints')
    cue = int(initial[tuple(first_locations[0])][0])
    upper, lower = int(fork[1, 6, 0]), int(fork[5, 6, 0])
    if {upper, lower} != {KEY, BALL}:
        raise ValueError('Native fork requires one key and one ball')
    return (0 if cue == upper else 1), (cue, upper)


def collect_sequence(size, seed):
    """Collect actual partial observations; the final turn is never a model input."""
    env = make_cue_env(size, seed)
    try:
        first = env.gen_obs()
        observations = [encode_obs(first)]
        for step in range(size-3):
            obs, reward, terminated, truncated, _ = env.step(2)
            if terminated or truncated or reward != 0:
                raise ValueError('Forced forward path unexpectedly reached an outcome')
            if tuple(env.agent_pos) != (step+2, size//2) or env.agent_dir != 0:
                raise ValueError('Native action2 did not follow the declared corridor path')
            observations.append(encode_obs(obs))
        label, context = teacher_choice(first, obs)
        observations = np.stack(observations)
        previous = np.full(len(observations), 2, dtype=np.int64)
        previous[0] = 0  # Masked by the initial reset flag, never the teacher target.
        resets = np.zeros(len(observations), dtype=bool)
        resets[0] = True
        return {'observations': observations, 'previous': previous, 'resets': resets,
                'label': label, 'context': context, 'seed': int(seed), 'size': int(size)}
    finally:
        env.close()


def packet(examples):
    data = {'observations': np.stack([row['observations'] for row in examples], axis=1),
            'previous': np.stack([row['previous'] for row in examples], axis=1),
            'resets': np.stack([row['resets'] for row in examples], axis=1),
            'labels': np.asarray([row['label'] for row in examples], dtype=np.int64)}
    metadata = {'size': examples[0]['size'], 'contexts': [
        {'seed': row['seed'], 'cue': row['context'][0], 'upper_branch': row['context'][1],
         'target_turn': row['label'], 'observations_sha256': array_sha(row['observations'])} for row in examples],
        'array_sha256': {name: array_sha(value) for name, value in data.items()},
        'unique_observation_sequences': len({array_sha(row['observations']) for row in examples}),
        'target_counts': {str(label): int(np.count_nonzero(data['labels'] == label)) for label in (0, 1)}}
    return data, metadata


def training_packet():
    selected = {}
    start = PROTOCOL['training_context_seed_start']
    for seed in range(start, start+PROTOCOL['training_seed_search_limit']):
        example = collect_sequence(PROTOCOL['train_size'], seed)
        selected.setdefault(example['context'], example)
        if len(selected) == 4:
            data, metadata = packet([selected[context] for context in CONTEXTS])
            if metadata['unique_observation_sequences'] != 4 or metadata['target_counts'] != {'0': 2, '1': 2}:
                raise ValueError('Training contexts must be four distinct, balanced native sequences')
            return data, {**metadata, 'seeds_searched': seed-start+1}
    raise ValueError('Training-only seed search did not find all four contexts within its fixed limit')


def signature():
    _, metadata = training_packet()
    return {'protocol': PROTOCOL, 'source_sha256': {name: sha(ROOT/name) for name in SOURCES},
            'dependencies': {name: importlib.metadata.version(name)
                             for name in ('minigrid', 'gymnasium', 'torch', 'numpy')},
            'native_environment_source_sha256': sha(inspect.getfile(MemoryEnv)),
            'training_data': metadata, 'python': platform.python_version(), 'lock_sha256': sha(ROOT/'uv.lock')}


def new_model(mode, seed):
    torch.manual_seed(seed)
    return AssociativePolicy(mode, hidden=PROTOCOL['hidden'], store_dim=PROTOCOL['store_dim'],
                             adapter_hidden=PROTOCOL['adapter_hidden'])


def final_logits(model, data, condition='intact'):
    if condition not in PROTOCOL['conditions']:
        raise ValueError('Unknown state ablation')
    observations = torch.from_numpy(data['observations'])
    previous = torch.from_numpy(data['previous'])
    resets = torch.from_numpy(data['resets'])
    logits, _, _ = model.sequence(observations, previous, model.initial_state(observations.shape[1]), resets,
                                  current_only=condition == 'reset_all', reset_store=condition == 'reset_store')
    return logits[-1]


def fit(mode, seed, data, out, plan_hash, *, updates=None):
    count = PROTOCOL['updates'] if updates is None else updates
    if count not in (2, PROTOCOL['updates']):
        raise ValueError('Only the fixed full budget or two-update implementation smoke is allowed')
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'started.json', {'mode': mode, 'seed': seed, 'updates': count,
                                 'plan_sha256': plan_hash, 'smoke': count == 2, 'started_unix': time.time()})
    model = new_model(mode, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=PROTOCOL['learning_rate'],
                                 betas=tuple(PROTOCOL['betas']), eps=PROTOCOL['epsilon'],
                                 weight_decay=PROTOCOL['weight_decay'])
    labels = torch.from_numpy(data['labels'])
    active_parameters = set()
    begin = time.perf_counter()
    with (out/'learning.jsonl').open('x') as logfile:
        for update in range(count):
            model.train()
            logits = final_logits(model, data)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite supervised loss; retain receipts and stop')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise ValueError('Nonfinite gradient; retain receipts and stop')
                    active_parameters.add(name)
            optimizer.step()
            record = {'update': update+1, 'final_turn_cross_entropy': float(loss.detach()),
                      'sequences_seen': (update+1)*4, 'elapsed_seconds': time.perf_counter()-begin}
            logfile.write(json.dumps(record, allow_nan=False)+'\n')
            logfile.flush()
            if count == 2 or (update+1) % 100 == 0:
                print(json.dumps({'mode': mode, 'seed': seed, **record}), flush=True)
    checkpoint = out/'model.pt'
    torch.save(model.state_dict(), checkpoint)
    result = {'status': 'completed', 'mode': mode, 'seed': seed, 'plan_sha256': plan_hash,
              'updates': count, 'sequences_seen': count*4, 'unique_training_sequences': 4,
              'training_seconds': time.perf_counter()-begin,
              'registered_parameters': sum(value.numel() for value in model.parameters()),
              'trainable_parameters': sum(value.numel() for value in model.parameters() if value.requires_grad),
              'gradient_parameters': sum(value.numel() for name, value in model.named_parameters()
                                         if name in active_parameters),
              'gradient_parameter_names': sorted(active_parameters), 'checkpoint_sha256': sha(checkpoint),
              'learning_sha256': sha(out/'learning.jsonl'), 'smoke': count == 2,
              'supervised_only': True, 'rl_warmstart': False}
    write_new(out/'completed.json', result)
    return result


@torch.inference_mode()
def evaluate(model, data, metadata, condition):
    model.eval()
    logits = final_logits(model, data, condition)
    probabilities = logits.softmax(-1).numpy()
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(-1), 1., atol=1e-6):
        raise ValueError('Invalid action probabilities')
    predictions = probabilities.argmax(-1)
    labels = data['labels']
    rows = [{**context, 'predicted_action': int(predictions[i]),
             'probabilities': probabilities[i].tolist(), 'correct': bool(predictions[i] == labels[i])}
            for i, context in enumerate(metadata['contexts'])]
    return {'condition': condition, 'size': metadata['size'], 'examples': rows,
            'accuracy': float(np.mean(predictions == labels)),
            'cross_entropy': float(F.cross_entropy(logits, torch.from_numpy(labels))),
            'target_probability': float(np.mean(probabilities[np.arange(len(labels)), labels])),
            'data_array_sha256': metadata['array_sha256'], 'unique_contexts': metadata['unique_observation_sequences']}


def run(plan, out):
    frozen = json.loads(plan.read_text())
    if frozen != signature():
        raise ValueError('Frozen source, runtime, protocol or training data changed')
    out.mkdir(parents=True, exist_ok=False)
    plan_hash = sha(plan)
    write_new(out/'started.json', {'plan_sha256': plan_hash, 'started_unix': time.time(),
                                 'scope': PROTOCOL['scope']})
    data, training_metadata = training_packet()
    data_dir = out/'data'
    data_dir.mkdir()
    np.savez_compressed(data_dir/'training.npz', **data)
    write_new(data_dir/'training.json', {**training_metadata, 'npz_sha256': sha(data_dir/'training.npz')})
    jobs = [(mode, seed) for mode in MODES for seed in PROTOCOL['seeds']]
    random.Random(PROTOCOL['fit_order_seed']).shuffle(jobs)
    write_new(out/'fit-order.json', jobs)
    fits = [fit(mode, seed, data, out/'fits'/f'{mode}-{seed}', plan_hash) for mode, seed in jobs]
    write_new(out/'training-completed.json', {'status': 'completed', 'fits': fits, 'plan_sha256': plan_hash})
    # Evaluation packets are created only after all training; there is no model or seed selection from them.
    selected_seeds = [row['seed'] for row in training_metadata['contexts']]
    panels = {}
    for size in PROTOCOL['evaluation_sizes']:
        panel, metadata = packet([collect_sequence(size, seed) for seed in selected_seeds])
        panels[size] = panel, metadata
        np.savez_compressed(data_dir/f'evaluation-{size}.npz', **panel)
        write_new(data_dir/f'evaluation-{size}.json',
                  {**metadata, 'npz_sha256': sha(data_dir/f'evaluation-{size}.npz'),
                   'same_selected_seed_ids_as_training': True, 'unseen_unique_maps': False})
    evaluations = []
    for mode, seed in jobs:
        directory = out/'fits'/f'{mode}-{seed}'
        receipt = json.loads((directory/'completed.json').read_text())
        if (sha(directory/'model.pt') != receipt['checkpoint_sha256']
                or sha(directory/'learning.jsonl') != receipt['learning_sha256']):
            raise ValueError('Fit checkpoint or learning log changed before evaluation')
        model = new_model(mode, seed)
        model.load_state_dict(torch.load(directory/'model.pt', map_location='cpu', weights_only=True))
        results = [evaluate(model, *panels[size], condition) for size in PROTOCOL['evaluation_sizes']
                   for condition in PROTOCOL['conditions']]
        record = {'mode': mode, 'seed': seed, 'checkpoint_sha256': receipt['checkpoint_sha256'], 'results': results}
        evaluations.append(record)
        write_new(out/'evaluation'/f'{mode}-{seed}.json', record)
        print(json.dumps({'evaluation': f'{mode}-{seed}', 'accuracy': {
            f"{row['size']}-{row['condition']}": row['accuracy'] for row in results}}), flush=True)
    expected_fits = {(mode, seed) for mode in MODES for seed in PROTOCOL['seeds']}
    if len(fits) != 12 or {(row['mode'], row['seed']) for row in fits} != expected_fits:
        raise ValueError('Supervised fit identities/count mismatch')
    averages = {mode: {str(size): {condition: {
        key: float(np.mean([result[key] for evaluation in evaluations if evaluation['mode'] == mode
                            for result in evaluation['results']
                            if result['size'] == size and result['condition'] == condition]))
        for key in ('accuracy', 'cross_entropy', 'target_probability')}
        for condition in PROTOCOL['conditions']} for size in PROTOCOL['evaluation_sizes']} for mode in MODES}
    summary = {'protocol': PROTOCOL, 'plan_sha256': plan_hash, 'fits': fits, 'evaluations': evaluations,
               'averages': averages, 'optimizer_updates': sum(row['updates'] for row in fits),
               'training_seconds': sum(row['training_seconds'] for row in fits),
               'training_data': training_metadata,
               'evaluation_data': {str(size): metadata for size, (_, metadata) in panels.items()},
               'claim': 'Four-context supervised final-turn learnability and forced-path length transfer only',
               'gameplay_evaluated': False, 'rl_training': False, 'rl_warmstart': False,
               'unseen_unique_map_test': False, 'novelty_established': False, 'independent_confirmation': False}
    write_new(out/'summary.json', summary)
    write_new(out/'completed.json', {'status': 'completed', 'fits': 12, 'optimizer_updates': 6000,
                                   'plan_sha256': plan_hash, 'summary_sha256': sha(out/'summary.json'),
                                   'finished_unix': time.time()})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'smoke', 'run'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(PROTOCOL['torch_threads'])
    if args.command == 'freeze':
        write_new(args.out, signature())
    elif args.command == 'smoke':
        data, _ = training_packet()
        for mode in MODES:
            fit(mode, 7, data, args.out/mode, 'implementation-smoke-not-frozen-study', updates=2)
    else:
        run(args.plan, args.out)


if __name__ == '__main__':
    main()
