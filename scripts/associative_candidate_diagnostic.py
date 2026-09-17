"""Fixed four-context candidate-conditioned associative-memory diagnostic.

The prior model, teacher, observations, optimizer and budget remain unchanged.
Only training cross entropy changes from seven actions to the two turn logits.
The model retains all seven outputs. No RL or gameplay is performed here.
"""

import argparse
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
    '_candidate_frozen_teacher', ROOT / 'scripts/associative_learnability.py')
teacher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(teacher)

PROTOCOL = {
    **teacher.PROTOCOL,
    'version': 'associative-candidate-v1',
    'teacher_protocol': 'associative-learnability-v1',
    'objective': 'Final turn logits0:2 cross entropy; unchanged seven-output model and complete sequence gradients',
    'sole_training_intervention': 'Cross entropy normalizes over candidates0,1 instead of all seven actions',
    'evaluation': 'Report both seven-action and two-candidate probabilities; same four selected seeds at every length',
    'telemetry': 'Post-fit intact sequences only; write strengths and opposite-cue state distances are descriptive',
    'continuation': {
        'candidate': 'fast_selective', 'accuracy2_each_fit_each_size': 1.,
        'conditional_ce_each_fit_each_size_strictly_below': .05,
        'longest_size_store_reset_accuracy2_drop_minimum': .25,
        'longest_size_accuracy2_gain_each_control_minimum': .25,
        'controls': ['gru', 'feedforward', 'fast_global'],
        'scope': 'Development continuation only, not robust generalization, gameplay, RL or novelty',
    },
    'comparison': 'Post-run exact tensor equality of freshly trained GRU fits versus memory-optimization-v1 recursive ce2',
    'scope': 'Four-context supervised candidate choice and forced-path length transfer only',
    'independent_confirmation': False,
}
SOURCES = ['scripts/associative_candidate_diagnostic.py', 'tests/test_associative_candidate_diagnostic.py',
           *teacher.SOURCES]
METRICS = ('accuracy7', 'accuracy2', 'cross_entropy7', 'conditional_binary_cross_entropy',
           'target_probability7', 'target_probability2')
sha, write_new = teacher.sha, teacher.write_new


def signature():
    _, metadata = teacher.training_packet()
    return {'protocol': PROTOCOL, 'source_sha256': {name: sha(ROOT / name) for name in SOURCES},
            'training_data': metadata,
            'dependencies': {name: importlib.metadata.version(name)
                             for name in ('minigrid', 'gymnasium', 'torch', 'numpy')},
            'python': platform.python_version(), 'lock_sha256': sha(ROOT / 'uv.lock')}


def objective(logits, labels):
    if logits.ndim != 2 or logits.shape[1] != 7 or labels.shape != logits.shape[:1]:
        raise ValueError('Expected seven model logits and one turn label per context')
    if labels.dtype != torch.long or not torch.isin(labels, labels.new_tensor([0, 1])).all():
        raise ValueError('Teacher labels must be native left/right turn IDs 0 and 1')
    return F.cross_entropy(logits[:, :2], labels)


def fit(mode, seed, data, out, plan_hash, *, smoke=False):
    if mode not in PROTOCOL['modes'] or (smoke and seed != 7) or (not smoke and seed not in PROTOCOL['seeds']):
        raise ValueError('Only the declared fits or seed-7 implementation smoke are allowed')
    updates = 2 if smoke else PROTOCOL['updates']
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'started.json', {'mode': mode, 'seed': seed, 'updates': updates,
                                     'plan_sha256': plan_hash, 'smoke': smoke, 'started_unix': time.time()})
    model = teacher.new_model(mode, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=PROTOCOL['learning_rate'],
                                 betas=tuple(PROTOCOL['betas']), eps=PROTOCOL['epsilon'],
                                 weight_decay=PROTOCOL['weight_decay'])
    labels = torch.from_numpy(data['labels'])
    active = set()
    begin = time.perf_counter()
    with (out / 'learning.jsonl').open('x') as logfile:
        for update in range(updates):
            model.train()
            logits = teacher.final_logits(model, data)
            loss = objective(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite candidate loss; preserve receipts and stop')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise ValueError('Nonfinite gradient; preserve receipts and stop')
                    active.add(name)
            optimizer.step()
            row = {'update': update + 1, 'conditional_binary_cross_entropy': float(loss.detach()),
                   'sequences_seen': (update + 1) * 4, 'elapsed_seconds': time.perf_counter() - begin}
            logfile.write(json.dumps(row, allow_nan=False) + '\n')
            logfile.flush()
            if smoke or (update + 1) % 100 == 0:
                print(json.dumps({'mode': mode, 'seed': seed, **row}), flush=True)
    torch.save(model.state_dict(), out / 'model.pt')
    receipt = {
        'status': 'completed', 'mode': mode, 'seed': seed, 'updates': updates,
        'sequences_seen': updates * 4, 'unique_training_contexts': 4,
        'training_seconds': time.perf_counter() - begin, 'plan_sha256': plan_hash,
        'checkpoint_sha256': sha(out / 'model.pt'), 'learning_sha256': sha(out / 'learning.jsonl'),
        'registered_parameters': sum(parameter.numel() for parameter in model.parameters()),
        'trainable_parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        'gradient_parameters': sum(parameter.numel() for name, parameter in model.named_parameters() if name in active),
        'gradient_parameter_names': sorted(active), 'smoke': smoke, 'supervised_only': True, 'rl_warmstart': False,
    }
    write_new(out / 'completed.json', receipt)
    return receipt


@torch.inference_mode()
def evaluate(model, data, metadata, condition):
    model.eval()
    logits = teacher.final_logits(model, data, condition)
    probabilities7 = logits.softmax(-1).numpy()
    probabilities2 = logits[:, :2].softmax(-1).numpy()
    for probabilities in (probabilities7, probabilities2):
        if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(-1), 1., rtol=0, atol=1e-6)):
            raise ValueError('Invalid final action or candidate probabilities')
    labels = data['labels']
    predictions7, predictions2 = probabilities7.argmax(-1), probabilities2.argmax(-1)
    rows = [{**context, 'probabilities7': probabilities7[index].tolist(),
             'probabilities2': probabilities2[index].tolist(),
             'predicted_action7': int(predictions7[index]), 'predicted_candidate2': int(predictions2[index]),
             'correct7': bool(predictions7[index] == labels[index]),
             'correct2': bool(predictions2[index] == labels[index])}
            for index, context in enumerate(metadata['contexts'])]
    return {
        'size': metadata['size'], 'condition': condition,
        'accuracy7': float(np.mean(predictions7 == labels)), 'accuracy2': float(np.mean(predictions2 == labels)),
        'cross_entropy7': float(F.cross_entropy(logits, torch.from_numpy(labels))),
        'conditional_binary_cross_entropy': float(objective(logits, torch.from_numpy(labels))),
        'target_probability7': float(probabilities7[np.arange(len(labels)), labels].mean()),
        'target_probability2': float(probabilities2[np.arange(len(labels)), labels].mean()),
        'examples': rows, 'unique_contexts': 4, 'data_array_sha256': metadata['array_sha256'],
    }


def cue_pairs(metadata):
    pairs = []
    for upper in (teacher.KEY, teacher.BALL):
        group = [i for i, row in enumerate(metadata['contexts']) if row['upper_branch'] == upper]
        if len(group) != 2 or {metadata['contexts'][i]['cue'] for i in group} != {teacher.KEY, teacher.BALL}:
            raise ValueError('Telemetry requires opposite cues with the same visible branch mapping')
        pairs.append(group)
    return pairs


@torch.inference_mode()
def telemetry(model, data, metadata):
    """Observe trained state without supplying labels or context metadata to the model."""
    model.eval()
    observations = torch.from_numpy(data['observations'])
    previous, resets = torch.from_numpy(data['previous']), torch.from_numpy(data['resets'])
    pairs = cue_pairs(metadata)
    state = model.initial_state(observations.shape[1])
    rows = []
    for step, (obs, action, reset) in enumerate(zip(observations, previous, resets, strict=True)):
        old_state = state
        _, _, state = model.observe(obs, action, state, reset)
        if not torch.isfinite(state).all():
            raise ValueError('Nonfinite post-fit state telemetry')
        row = {'step': step, 'gru_opposite_cue_l2': float(torch.stack([
            (state[a, :model.hidden] - state[b, :model.hidden]).norm() for a, b in pairs]).mean()),
            'store_opposite_cue_frobenius': None, 'write_strengths': None,
            'mean_write_strength': None, 'mean_store_update_frobenius': None}
        if model.uses_store:
            encoded = model.encoder(obs)
            strength = (model.write_gate(encoded).sigmoid() if model.mode == 'fast_selective'
                        else model.write_gate.bias.sigmoid().expand(len(obs), 1))
            stores = state[:, model.hidden:]
            prior_store = old_state[:, model.hidden:] * (~reset).to(state.dtype)[:, None]
            row.update({
                'store_opposite_cue_frobenius': float(torch.stack([
                    (stores[a] - stores[b]).norm() for a, b in pairs]).mean()),
                'write_strengths': strength.flatten().tolist(), 'mean_write_strength': float(strength.mean()),
                'mean_store_update_frobenius': float((stores - prior_store).norm(dim=-1).mean()),
            })
        rows.append(row)
    return {'size': metadata['size'], 'condition': 'intact', 'context_seed_ids': [r['seed'] for r in metadata['contexts']],
            'per_step': rows, 'data_array_sha256': metadata['array_sha256'],
            'interpretation': 'Post-fit state/write statistics, not information content or proof of usable memory'}


def continuation(evaluations):
    expected = {(mode, seed) for mode in PROTOCOL['modes'] for seed in PROTOCOL['seeds']}
    if len(evaluations) != len(expected) or {(r['mode'], r['seed']) for r in evaluations} != expected:
        raise ValueError('Continuation needs all declared final fits')
    rows = {}
    for record in evaluations:
        expected_conditions = {(size, condition) for size in PROTOCOL['evaluation_sizes']
                               for condition in PROTOCOL['conditions']}
        if (len(record['results']) != len(expected_conditions)
                or {(r['size'], r['condition']) for r in record['results']} != expected_conditions):
            raise ValueError('Continuation needs every map size and state condition')
        for result in record['results']:
            for key in METRICS:
                if not np.isfinite(result[key]) or result[key] < 0:
                    raise ValueError('Continuation metrics must be finite and nonnegative')
            if any(result[key] > 1 for key in ('accuracy7', 'accuracy2', 'target_probability7', 'target_probability2')):
                raise ValueError('Continuation accuracy and probability must lie between zero and one')
            rows[(record['mode'], record['seed'], result['size'], result['condition'])] = result
    rule, candidate = PROTOCOL['continuation'], PROTOCOL['continuation']['candidate']
    means = {mode: {str(size): {condition: {
        key: float(np.mean([rows[(mode, seed, size, condition)][key] for seed in PROTOCOL['seeds']]))
        for key in METRICS} for condition in PROTOCOL['conditions']}
        for size in PROTOCOL['evaluation_sizes']} for mode in PROTOCOL['modes']}
    checks = {}
    for size in PROTOCOL['evaluation_sizes']:
        checks[f'perfect_all_fits_size_{size}'] = all(
            rows[(candidate, seed, size, 'intact')]['accuracy2'] >= rule['accuracy2_each_fit_each_size']
            for seed in PROTOCOL['seeds'])
        checks[f'conditional_ce_size_{size}'] = all(
            rows[(candidate, seed, size, 'intact')]['conditional_binary_cross_entropy']
            < rule['conditional_ce_each_fit_each_size_strictly_below']
            for seed in PROTOCOL['seeds'])
    longest = str(max(PROTOCOL['evaluation_sizes']))
    candidate_accuracy = means[candidate][longest]['intact']['accuracy2']
    checks['longest_store_reset_drop'] = (candidate_accuracy - means[candidate][longest]['reset_store']['accuracy2']
                                        >= rule['longest_size_store_reset_accuracy2_drop_minimum'])
    for control in rule['controls']:
        checks[f'longest_gain_over_{control}'] = (
            candidate_accuracy - means[control][longest]['intact']['accuracy2']
            >= rule['longest_size_accuracy2_gain_each_control_minimum'])
    return means, checks


def run(plan, out):
    if json.loads(plan.read_text()) != signature():
        raise ValueError('Frozen candidate diagnostic protocol, source, runtime or teacher data changed')
    plan_hash = sha(plan)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'started.json', {'plan_sha256': plan_hash, 'started_unix': time.time()})
    data, metadata = teacher.training_packet()
    (out / 'data').mkdir()
    np.savez_compressed(out / 'data/training.npz', **data)
    write_new(out / 'data/training.json', {**metadata, 'npz_sha256': sha(out / 'data/training.npz')})
    jobs = [(mode, seed) for mode in PROTOCOL['modes'] for seed in PROTOCOL['seeds']]
    random.Random(PROTOCOL['fit_order_seed']).shuffle(jobs)
    write_new(out / 'fit-order.json', jobs)
    fits = [fit(mode, seed, data, out / 'fits' / f'{mode}-{seed}', plan_hash) for mode, seed in jobs]
    write_new(out / 'training-completed.json', {'status': 'completed', 'fits': fits, 'plan_sha256': plan_hash})
    selected_seeds = [row['seed'] for row in metadata['contexts']]
    panels = {}
    for size in PROTOCOL['evaluation_sizes']:
        panel, description = teacher.packet([teacher.collect_sequence(size, seed) for seed in selected_seeds])
        panels[size] = panel, description
        np.savez_compressed(out / 'data' / f'evaluation-{size}.npz', **panel)
        write_new(out / 'data' / f'evaluation-{size}.json', {
            **description, 'npz_sha256': sha(out / 'data' / f'evaluation-{size}.npz'),
            'same_selected_seed_ids_as_training': True, 'unseen_unique_maps': False})
    evaluations, diagnostics = [], []
    for receipt in fits:
        mode, seed = receipt['mode'], receipt['seed']
        directory = out / 'fits' / f'{mode}-{seed}'
        if (sha(directory / 'model.pt') != receipt['checkpoint_sha256']
                or sha(directory / 'learning.jsonl') != receipt['learning_sha256']
                or json.loads((directory / 'completed.json').read_text()) != receipt):
            raise ValueError('Training artifact or receipt changed before evaluation')
        model = teacher.new_model(mode, seed)
        model.load_state_dict(torch.load(directory / 'model.pt', map_location='cpu', weights_only=True))
        results = [evaluate(model, *panels[size], condition)
                   for size in PROTOCOL['evaluation_sizes'] for condition in PROTOCOL['conditions']]
        record = {'mode': mode, 'seed': seed, 'checkpoint_sha256': receipt['checkpoint_sha256'], 'results': results}
        evaluations.append(record)
        write_new(out / 'evaluation' / f'{mode}-{seed}.json', record)
        diagnostic = {'mode': mode, 'seed': seed, 'checkpoint_sha256': receipt['checkpoint_sha256'],
                      'results': [telemetry(model, *panels[size]) for size in PROTOCOL['evaluation_sizes']]}
        diagnostics.append(diagnostic)
        write_new(out / 'telemetry' / f'{mode}-{seed}.json', diagnostic)
        print(json.dumps({'evaluation': f'{mode}-{seed}', 'results': [
            {key: r[key] for key in ('size', 'condition', 'accuracy7', 'accuracy2', 'conditional_binary_cross_entropy')}
            for r in results]}), flush=True)
    averages, checks = continuation(evaluations)
    summary = {
        'protocol': PROTOCOL, 'plan_sha256': plan_hash, 'fits': fits, 'evaluations': evaluations,
        'averages': averages, 'continuation_checks': checks, 'continuation_passed': all(checks.values()),
        'telemetry': diagnostics, 'training_data': metadata,
        'evaluation_data': {str(size): description for size, (_, description) in panels.items()},
        'optimizer_updates': sum(row['updates'] for row in fits),
        'training_seconds': sum(row['training_seconds'] for row in fits),
        'limitations': ['Same four training contexts; length probes reuse the selected seed IDs',
                        'Two-candidate accuracy does not establish seven-action gameplay ability',
                        'Write strengths and cue distances do not establish information content or novelty'],
        'gameplay_evaluated': False, 'rl_training': False, 'rl_warmstart': False,
        'unseen_unique_map_test': False, 'novelty_established': False, 'independent_confirmation': False,
    }
    write_new(out / 'summary.json', summary)
    write_new(out / 'completed.json', {'status': 'completed', 'fits': 12, 'optimizer_updates': 6000,
                                      'plan_sha256': plan_hash, 'summary_sha256': sha(out / 'summary.json'),
                                      'finished_unix': time.time()})
    return summary


def compare_gru(run_dir, reference, out):
    """Audit final tensor equality; neither checkpoint initializes any training run."""
    summaries = []
    for directory, version in ((run_dir, PROTOCOL['version']), (reference, 'memory-optimization-v1')):
        summary = json.loads((directory / 'summary.json').read_text())
        completed = json.loads((directory / 'completed.json').read_text())
        if (completed['status'] != 'completed' or completed['summary_sha256'] != sha(directory / 'summary.json')
                or completed['plan_sha256'] != summary['plan_sha256']
                or summary['protocol']['version'] != version or completed['fits'] != 12
                or completed['optimizer_updates'] != 6000):
            raise ValueError('Tensor comparison requires both complete frozen diagnostic runs')
        summaries.append(summary)
    left, right = summaries
    common = ('seeds', 'updates', 'batch_size', 'learning_rate', 'betas', 'epsilon', 'weight_decay')
    if left['training_data'] != right['training_data'] or any(
            left['protocol'][key] != right['protocol'][key] for key in common):
        raise ValueError('GRU comparison requires identical teacher data and optimizer settings')
    records = []
    for seed in PROTOCOL['seeds']:
        a_receipts = [r for r in left['fits'] if r['mode'] == 'gru' and r['seed'] == seed]
        b_receipts = [r for r in right['fits'] if r['path'] == 'recursive' and r['loss'] == 'ce2' and r['seed'] == seed]
        if len(a_receipts) != 1 or len(b_receipts) != 1:
            raise ValueError('Missing or duplicated corresponding GRU fit')
        paths = [run_dir / 'fits' / f'gru-{seed}' / 'model.pt',
                 reference / 'fits' / f'recursive-ce2-{seed}' / 'model.pt']
        for path, receipt in zip(paths, [a_receipts[0], b_receipts[0]], strict=True):
            if sha(path) != receipt['checkpoint_sha256']:
                raise ValueError('Comparison checkpoint differs from completed training receipt')
        a, b = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
        if set(a) != set(b):
            raise ValueError('GRU checkpoint parameter names differ')
        equal = {name: torch.equal(a[name], b[name]) for name in a}
        records.append({'seed': seed, 'candidate_checkpoint_sha256': sha(paths[0]),
                        'reference_checkpoint_sha256': sha(paths[1]), 'tensor_equality': equal,
                        'all_tensors_equal': all(equal.values())})
    result = {'status': 'completed', 'candidate_summary_sha256': sha(run_dir / 'summary.json'),
              'reference_summary_sha256': sha(reference / 'summary.json'),
              'comparison_source_sha256': sha(Path(__file__)), 'fits': records,
              'all_tensors_equal': all(row['all_tensors_equal'] for row in records),
              'scope': 'Reproduction audit of independently trained GRU CE2 controls, not an efficacy result',
              'training_checkpoints_reused': False}
    write_new(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'smoke', 'run', 'compare'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(PROTOCOL['torch_threads'])
    if args.command == 'freeze':
        write_new(args.out, signature())
    elif args.command == 'smoke':
        data, _ = teacher.training_packet()
        for mode in PROTOCOL['modes']:
            fit(mode, 7, data, args.out / mode, 'implementation-smoke-not-frozen-study', smoke=True)
    elif args.command == 'compare':
        compare_gru(args.run, args.reference, args.out)
    else:
        run(args.plan, args.out)


if __name__ == '__main__':
    main()
