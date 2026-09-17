"""Separate 2x2 supervised diagnostic of memory credit and action suppression.

The GRU model, initialization and four observable-teacher contexts are unchanged.
Factors are recursive state versus an unchanged differentiable first-state cache,
and seven-action cross entropy versus cross entropy over the two turn logits.
This is four-context supervised choice learning, not RL or gameplay performance.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    '_optimization_frozen_teacher', ROOT/'scripts/associative_learnability.py')
teacher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(teacher)

PATHS = ('recursive', 'cache_h0')
LOSSES = ('ce7', 'ce2')
PROTOCOL = {
    'version': 'memory-optimization-v1', 'teacher_protocol': 'associative-learnability-v1',
    'paths': list(PATHS), 'losses': list(LOSSES), 'model': 'unchanged AssociativePolicy gru',
    'seeds': [101, 113, 127], 'train_size': 11, 'evaluation_sizes': [11, 17, 23],
    'updates': 500, 'batch_size': 4, 'optimizer': 'Adam', 'learning_rate': .001,
    'betas': [.9, .999], 'epsilon': 1e-8, 'weight_decay': 0., 'gradient_clipping': None,
    'torch_threads': 1, 'fit_order_seed': 831039,
    'cache_h0': 'Compute first-observation state once; every later observation uses that same undetached state',
    'ce7': 'Final seven-action logits versus native turn0/turn1 target',
    'ce2': 'Final logits0:2 only; model retains seven output logits',
    'fork_ablation': 'At final observation only: zero cached h0 or recursive current_only=True',
    'training_data': 'Exact four balanced observable-teacher contexts selected by the unchanged earlier script',
    'evaluation_data': 'Same four selected seed IDs at all sizes; not unseen unique maps',
    'selection': 'Final checkpoint only; all three seeds; fixed budget; no early stopping or selection',
    'saturation_threshold': .95,
    'state_distance': 'Mean Euclidean distance between opposite-cue pairs with the same upper branch',
    'scope': 'Supervised choice optimization and forced-path length transfer, not gameplay or RL',
    'rl_training': False, 'rl_warmstart': False, 'novelty_established': False,
    'independent_confirmation': False,
}
SOURCES = ['scripts/memory_optimization_diagnostic.py', 'tests/test_memory_optimization_diagnostic.py',
           *teacher.SOURCES]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def signature():
    _, metadata = teacher.training_packet()
    return {'protocol': PROTOCOL, 'source_sha256': {name: sha(ROOT/name) for name in SOURCES},
            'training_data': metadata,
            'dependencies': {name: importlib.metadata.version(name)
                             for name in ('minigrid', 'gymnasium', 'torch', 'numpy')},
            'python': platform.python_version(), 'lock_sha256': sha(ROOT/'uv.lock')}


def forward_sequence(model, observations, previous, resets, path, *, ablate_fork=False):
    """Consume causal inputs, preserving the first-state computation graph.

    Cached mode is intentionally a fixed first-observation memory intervention:
    intermediate observations do not update the cache. At every time step the
    current observation still affects that step's output through the same GRU.
    """
    if path not in PATHS:
        raise ValueError('Unknown state path')
    if (observations.ndim != 3 or len(observations) < 2
            or previous.shape != observations.shape[:2] or resets.shape != observations.shape[:2]
            or not resets[0].all() or resets[1:].any()):
        raise ValueError('Expected complete single-episode sequences with only their initial reset')
    state = model.initial_state(observations.shape[1])
    first_state = None
    logits, values, states = [], [], []
    for t, (obs, action, reset) in enumerate(zip(observations, previous, resets, strict=True)):
        incoming = state if path == 'recursive' or t == 0 else first_state
        at_fork = t == len(observations)-1
        clear_recursive = ablate_fork and at_fork and path == 'recursive'
        if ablate_fork and at_fork and path == 'cache_h0':
            incoming = torch.zeros_like(first_state)
        logit, value, state = model.observe(obs, action, incoming, reset, current_only=clear_recursive)
        if t == 0:
            first_state = state  # Deliberately not detached and not overwritten later.
        logits.append(logit)
        values.append(value)
        states.append(state)
    return torch.stack(logits), torch.stack(values), torch.stack(states)


def outputs(model, data, path, *, ablate_fork=False):
    return forward_sequence(model, torch.from_numpy(data['observations']), torch.from_numpy(data['previous']),
                            torch.from_numpy(data['resets']), path, ablate_fork=ablate_fork)


def objective(logits, labels, loss_kind):
    if loss_kind == 'ce7':
        return F.cross_entropy(logits, labels)
    if loss_kind == 'ce2':
        return F.cross_entropy(logits[:, :2], labels)
    raise ValueError('Unknown loss')


def cue_pairs(metadata):
    pairs = []
    for upper in (teacher.KEY, teacher.BALL):
        group = [i for i, row in enumerate(metadata['contexts']) if row['upper_branch'] == upper]
        if len(group) != 2 or {metadata['contexts'][i]['cue'] for i in group} != {teacher.KEY, teacher.BALL}:
            raise ValueError('Expected both cue identities under each fork mapping')
        pairs.append(group)
    return pairs


@torch.inference_mode()
def activation_diagnostic(model, data, metadata, path):
    logits, _, states = outputs(model, data, path)
    encoded = model.encoder(torch.from_numpy(data['observations']))
    pairs = cue_pairs(metadata)
    cue_distance = [float(torch.stack([(states[t, a]-states[t, b]).norm() for a, b in pairs]).mean())
                    for t in range(len(states))]
    return {'encoder_saturation_fraction': float((encoded.abs() > PROTOCOL['saturation_threshold']).float().mean()),
            'fork_state_saturation_fraction': float((states[-1].abs() > PROTOCOL['saturation_threshold']).float().mean()),
            'initial_state_saturation_fraction': float((states[0].abs() > PROTOCOL['saturation_threshold']).float().mean()),
            'initial_cue_pair_l2': cue_distance[0], 'fork_cue_pair_l2': cue_distance[-1],
            'per_step_cue_pair_l2': cue_distance,
            'conditional_binary_cross_entropy': float(F.cross_entropy(logits[-1, :, :2],
                                                                      torch.from_numpy(data['labels']))),
            'unused_action_probability': float(logits[-1].softmax(-1)[:, 2:].sum(-1).mean()),
            'interpretation': 'Activation statistics and distances; not percentage of retained information'}


def fit(path, loss_kind, seed, data, metadata, out, plan_hash, *, smoke=False):
    updates = 2 if smoke else PROTOCOL['updates']
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'started.json', {'path': path, 'loss': loss_kind, 'seed': seed, 'updates': updates,
                                 'plan_sha256': plan_hash, 'smoke': smoke, 'started_unix': time.time()})
    model = teacher.new_model('gru', seed)
    before = activation_diagnostic(model, data, metadata, path)
    write_new(out/'before.json', before)
    optimizer = torch.optim.Adam(model.parameters(), lr=PROTOCOL['learning_rate'],
                                 betas=tuple(PROTOCOL['betas']), eps=PROTOCOL['epsilon'],
                                 weight_decay=PROTOCOL['weight_decay'])
    labels = torch.from_numpy(data['labels'])
    begin = time.perf_counter()
    with (out/'learning.jsonl').open('x') as logfile:
        for update in range(updates):
            model.train()
            logits = outputs(model, data, path)[0][-1]
            loss = objective(logits, labels, loss_kind)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss; preserve receipts and stop')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(not torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                   if parameter.grad is not None):
                raise ValueError('Nonfinite gradient; preserve receipts and stop')
            optimizer.step()
            record = {'update': update+1, 'optimized_loss': float(loss.detach()),
                      'sequences_seen': (update+1)*4, 'elapsed_seconds': time.perf_counter()-begin}
            logfile.write(json.dumps(record, allow_nan=False)+'\n')
            logfile.flush()
            if smoke or (update+1) % 100 == 0:
                print(json.dumps({'path': path, 'loss': loss_kind, 'seed': seed, **record}), flush=True)
    model.eval()
    after = activation_diagnostic(model, data, metadata, path)
    write_new(out/'after.json', after)
    torch.save(model.state_dict(), out/'model.pt')
    result = {'status': 'completed', 'path': path, 'loss': loss_kind, 'seed': seed,
              'updates': updates, 'sequences_seen': updates*4, 'unique_training_contexts': 4,
              'training_seconds': time.perf_counter()-begin, 'plan_sha256': plan_hash,
              'checkpoint_sha256': sha(out/'model.pt'), 'learning_sha256': sha(out/'learning.jsonl'),
              'before_sha256': sha(out/'before.json'), 'after_sha256': sha(out/'after.json'),
              'smoke': smoke, 'supervised_only': True, 'rl_warmstart': False}
    write_new(out/'completed.json', result)
    return result


@torch.inference_mode()
def evaluate(model, data, metadata, path, *, ablate_fork):
    model.eval()
    logits = outputs(model, data, path, ablate_fork=ablate_fork)[0][-1]
    probabilities7 = logits.softmax(-1).numpy()
    probabilities2 = logits[:, :2].softmax(-1).numpy()
    if not np.isfinite(probabilities7).all() or not np.isfinite(probabilities2).all():
        raise ValueError('Nonfinite evaluation probabilities')
    labels = data['labels']
    predicted7, predicted2 = probabilities7.argmax(-1), probabilities2.argmax(-1)
    rows = [{**context, 'predicted_action7': int(predicted7[i]), 'predicted_candidate2': int(predicted2[i]),
             'probabilities7': probabilities7[i].tolist(), 'probabilities2': probabilities2[i].tolist()}
            for i, context in enumerate(metadata['contexts'])]
    return {'size': metadata['size'], 'condition': 'fork_state_erased' if ablate_fork else 'intact',
            'accuracy7': float(np.mean(predicted7 == labels)), 'accuracy2': float(np.mean(predicted2 == labels)),
            'conditional_binary_cross_entropy': float(F.cross_entropy(logits[:, :2], torch.from_numpy(labels))),
            'cross_entropy7': float(F.cross_entropy(logits, torch.from_numpy(labels))),
            'target_probability7': float(np.mean(probabilities7[np.arange(len(labels)), labels])),
            'target_probability2': float(np.mean(probabilities2[np.arange(len(labels)), labels])),
            'examples': rows, 'data_array_sha256': metadata['array_sha256']}


def run(plan, out):
    if json.loads(plan.read_text()) != signature():
        raise ValueError('Frozen diagnostic protocol, source, runtime or data changed')
    plan_hash = sha(plan)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'started.json', {'plan_sha256': plan_hash, 'started_unix': time.time()})
    data, metadata = teacher.training_packet()
    (out/'data').mkdir()
    np.savez_compressed(out/'data/training.npz', **data)
    write_new(out/'data/training.json', {**metadata, 'npz_sha256': sha(out/'data/training.npz')})
    jobs = [(path, loss, seed) for path in PATHS for loss in LOSSES for seed in PROTOCOL['seeds']]
    random.Random(PROTOCOL['fit_order_seed']).shuffle(jobs)
    write_new(out/'fit-order.json', jobs)
    fits = [fit(path, loss, seed, data, metadata, out/'fits'/f'{path}-{loss}-{seed}', plan_hash)
            for path, loss, seed in jobs]
    write_new(out/'training-completed.json', {'status': 'completed', 'fits': fits, 'plan_sha256': plan_hash})
    selected_seeds = [row['seed'] for row in metadata['contexts']]
    panels = {}
    for size in PROTOCOL['evaluation_sizes']:
        panel, description = teacher.packet([teacher.collect_sequence(size, seed) for seed in selected_seeds])
        panels[size] = panel, description
        np.savez_compressed(out/'data'/f'evaluation-{size}.npz', **panel)
        write_new(out/'data'/f'evaluation-{size}.json', description)
    evaluations, activation_changes = [], []
    for receipt in fits:
        path, loss, seed = receipt['path'], receipt['loss'], receipt['seed']
        directory = out/'fits'/f'{path}-{loss}-{seed}'
        for filename, key in [('model.pt', 'checkpoint_sha256'), ('learning.jsonl', 'learning_sha256'),
                              ('before.json', 'before_sha256'), ('after.json', 'after_sha256')]:
            if sha(directory/filename) != receipt[key]:
                raise ValueError('Fit artifact identity changed')
        model = teacher.new_model('gru', seed)
        model.load_state_dict(torch.load(directory/'model.pt', map_location='cpu', weights_only=True))
        results = [evaluate(model, *panels[size], path, ablate_fork=ablated)
                   for size in PROTOCOL['evaluation_sizes'] for ablated in (False, True)]
        record = {'path': path, 'loss': loss, 'seed': seed,
                  'checkpoint_sha256': receipt['checkpoint_sha256'], 'results': results}
        evaluations.append(record)
        write_new(out/'evaluation'/f'{path}-{loss}-{seed}.json', record)
        activation_changes.append({'path': path, 'loss': loss, 'seed': seed,
                                   'before': json.loads((directory/'before.json').read_text()),
                                   'after': json.loads((directory/'after.json').read_text())})
        print(json.dumps({'evaluation': f'{path}-{loss}-{seed}', 'results': [
            {key: result[key] for key in ('size', 'condition', 'accuracy7', 'accuracy2',
                                          'conditional_binary_cross_entropy')} for result in results]}), flush=True)
    expected = {(path, loss, seed) for path in PATHS for loss in LOSSES for seed in PROTOCOL['seeds']}
    if len(fits) != 12 or {(r['path'], r['loss'], r['seed']) for r in fits} != expected:
        raise ValueError('Wrong factorial fit count or identities')
    averages = []
    for path in PATHS:
        for loss in LOSSES:
            for size in PROTOCOL['evaluation_sizes']:
                for condition in ('intact', 'fork_state_erased'):
                    rows = [r for e in evaluations if e['path'] == path and e['loss'] == loss
                            for r in e['results'] if r['size'] == size and r['condition'] == condition]
                    if len(rows) != 3:
                        raise ValueError('Each factorial condition needs all three final seeds')
                    averages.append({'path': path, 'loss': loss, 'size': size, 'condition': condition,
                                     **{key: float(np.mean([r[key] for r in rows])) for key in
                                        ('accuracy7', 'accuracy2', 'conditional_binary_cross_entropy',
                                         'target_probability7', 'target_probability2')}})
    summary = {'protocol': PROTOCOL, 'plan_sha256': plan_hash, 'fits': fits, 'evaluations': evaluations,
               'averages': averages, 'activation_changes': activation_changes, 'training_data': metadata,
               'evaluation_data': {str(size): description for size, (_, description) in panels.items()},
               'optimizer_updates': sum(row['updates'] for row in fits),
               'training_seconds': sum(row['training_seconds'] for row in fits),
               'limitations': ['Same four training contexts; length probes reuse their seed IDs',
                               'Cached first state is an explicit memory intervention, not a learned write policy',
                               'Two-candidate accuracy does not establish seven-action gameplay ability',
                               'Activation saturation and distances do not measure percentage information retained'],
               'rl_training': False, 'rl_warmstart': False, 'novelty_established': False,
               'independent_confirmation': False}
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
        data, metadata = teacher.training_packet()
        for path in PATHS:
            for loss in LOSSES:
                fit(path, loss, 7, data, metadata, args.out/f'{path}-{loss}',
                    'implementation-smoke-not-frozen-study', smoke=True)
    else:
        run(args.plan, args.out)


if __name__ == '__main__':
    main()
