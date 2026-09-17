"""Learn matched text heads without touching the reserved confirmation/calibration data."""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from openjev.research.learned_associative import MODES, LearnedAssociativeHead
from openjev.research.text_distillation import digest, file_hash, write_new

PROTOCOL = {
    'version': 'learned-associative-v1', 'seeds': [17, 29, 43], 'modes': list(MODES),
    'epochs': 30, 'batch_size': 256, 'learning_rate': .001, 'weight_decay': .0001,
    'clip_gradient': 1., 'latent_dim': 128, 'steps': 2, 'anchor': .5, 'beta': 10., 'top_k': 8,
    'train': '15,000 known-intent examples; frozen MiniLM sentence embeddings',
    'selection': '1,547 development examples from associative-text-v1; no epoch or seed selection',
    'rejection': 'All heads use original embedding centroid support with fixed threshold 0.48 from prior development',
    'support_threshold': .48, 'loss': 'Final-step cross entropy; identical minibatch orders within each seed',
    'prototype_initialization': 'Mean initial projected training states per class, then normalize',
    'cpu_threads': 4, 'latency_batch': 1, 'latency_repeats': 200, 'latency_warmup': 30,
    'continuation': {'minimum_accuracy_gain_over_best_control': .005,
                     'minimum_utility_gain_over_best_control': 0.,
                     'minimum_paired_seed_wins': 2, 'max_matrix_cost_multiple': 3.,
                     'interpretation': 'Development screen only; independent confirmation required'},
}


def signature():
    import openjev.research.learned_associative as architecture
    return {'study': file_hash(__file__), 'architecture': file_hash(architecture.__file__)}


def load_development(packet):
    """Allowlist parts: do not read test or calibration rows or embeddings."""
    manifest = json.loads((packet/'manifest.json').read_text())
    if manifest['protocol']['version'] != 'associative-text-v1':
        raise ValueError('Unexpected data protocol')
    parts = {}
    for name in ('train', 'tune'):
        info = manifest['parts'][name]
        if (file_hash(packet/f'{name}.json') != info['rows_sha256'] or
                file_hash(packet/f'{name}.npy') != info['vectors_sha256']):
            raise ValueError('Frozen data hash mismatch')
        rows = json.loads((packet/f'{name}.json').read_text())
        parts[name] = (torch.from_numpy(np.load(packet/f'{name}.npy', allow_pickle=False)),
                       torch.tensor([r['label'] for r in rows]), rows)
    if {r['text_sha256'] for r in parts['train'][2]} & {r['text_sha256'] for r in parts['tune'][2]}:
        raise ValueError('Training/development overlap')
    return manifest, parts


def freeze(args):
    manifest, _ = load_development(args.packet)
    write_new(args.out, {'protocol': PROTOCOL, 'code': signature(), 'packet_sha256': digest(manifest),
                         'environment': {k: importlib.metadata.version(k) for k in ('torch', 'numpy')},
                         'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})


def checked_plan(args):
    plan = json.loads(args.plan.read_text())
    manifest, parts = load_development(args.packet)
    if plan['protocol'] != PROTOCOL or plan['code'] != signature() or plan['packet_sha256'] != digest(manifest):
        raise ValueError('Frozen plan, data or source changed')
    torch.set_num_threads(PROTOCOL['cpu_threads'])
    return plan, parts


def evaluate(logits, gold, support):
    known = gold >= 0
    accepted = support >= PROTOCOL['support_threshold']
    predicted = logits.argmax(1)
    correct = predicted == gold
    id_accuracy = float(correct[known].float().mean())
    accepted_correct = float((correct & accepted)[known].float().mean())
    oos_recall = float((~accepted[~known]).float().mean())
    probs = torch.softmax(logits, dim=1)
    target = F.one_hot(gold[known], logits.shape[1]).float()
    return {'id_accuracy': id_accuracy, 'id_correct_and_accepted': accepted_correct,
            'oos_recall': oos_recall, 'balanced_utility': .5*(accepted_correct+oos_recall),
            'id_nll': float(F.cross_entropy(logits[known], gold[known])),
            'id_brier': float(((probs[known]-target)**2).sum(1).mean())}


@torch.no_grad()
def latency(model, query):
    for _ in range(PROTOCOL['latency_warmup']):
        model(query)
    elapsed = []
    for _ in range(PROTOCOL['latency_repeats']):
        start = time.perf_counter()
        model(query)
        elapsed.append((time.perf_counter()-start)*1000)
    return {'median_ms': float(np.median(elapsed)), 'p95_ms': float(np.percentile(elapsed, 95)),
            'scope': 'CPU head only, batch one, four threads, excludes encoder and support gate'}


def train(args):
    plan, parts = checked_plan(args)
    if args.out.exists():
        raise ValueError('Fit directory exists; do not overwrite or silently retry')
    args.out.mkdir(parents=True)
    write_new(args.out/'started.json', {'plan_sha256': digest(plan), 'device': args.device})
    x, y, _ = parts['train']
    query, gold, rows = parts['tune']
    classes = int(y.max())+1
    prototypes = F.normalize(torch.stack([x[y == k].mean(0) for k in range(classes)]), dim=1)
    support = (query @ prototypes.T).max(1).values
    for seed in PROTOCOL['seeds']:
        for mode in PROTOCOL['modes']:
            model = LearnedAssociativeHead(mode, classes=classes, seed=seed, **{k: PROTOCOL[k] for k in
                                            ('latent_dim', 'steps', 'anchor', 'beta', 'top_k')})
            start = time.perf_counter()
            model.initialize_prototypes(x, y)
            model.to(args.device).train()
            tx, ty = x.to(args.device), y.to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL['learning_rate'],
                                          weight_decay=PROTOCOL['weight_decay'])
            rng = torch.Generator().manual_seed(seed)
            history = []
            for epoch in range(PROTOCOL['epochs']):
                order = torch.randperm(len(x), generator=rng).to(args.device)
                losses = []
                for indices in order.split(PROTOCOL['batch_size']):
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(tx[indices]), ty[indices])
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL['clip_gradient'], error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                history.append({'epoch': epoch+1, 'updates': len(losses), 'loss': float(np.mean(losses))})
                if (epoch+1) % 10 == 0:
                    print(json.dumps({'mode': mode, 'seed': seed, **history[-1]}), flush=True)
            if args.device == 'mps':
                torch.mps.synchronize()
            elif args.device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            model.to('cpu').eval()
            checkpoint = args.out/f'{mode}-{seed}.pt'
            torch.save(model.state_dict(), checkpoint)
            with torch.no_grad():
                logits = model(query)
            result = {'mode': mode, 'seed': seed, 'plan_sha256': digest(plan), 'training_seconds': elapsed,
                      'cost': model.cost(), 'history': history, 'metrics': evaluate(logits, gold, support),
                      'latency': latency(model, x[:1]), 'checkpoint_sha256': file_hash(checkpoint)}
            write_new(args.out/f'{mode}-{seed}.json', result)
            # Labels/predictions stay local. No test/calibration data are loaded in this program.
            write_new(args.out/f'{mode}-{seed}-predictions.json', [
                {'id': row['id'], 'label': int(label), 'prediction': int(pred), 'support': float(s)}
                for row, label, pred, s in zip(rows, gold, logits.argmax(1), support, strict=True)])
            print(json.dumps({k: result[k] for k in ('mode', 'seed', 'metrics', 'training_seconds')}), flush=True)
    write_new(args.out/'completed.json', {'plan_sha256': digest(plan), 'fits': 12, 'status': 'completed'})


def report(args):
    plan, _ = checked_plan(args)
    completed = json.loads((args.runs/'completed.json').read_text())
    if completed != {'plan_sha256': digest(plan), 'fits': 12, 'status': 'completed'}:
        raise ValueError('All frozen fits must complete before selection')
    methods = {}
    for mode in PROTOCOL['modes']:
        fits = []
        for seed in PROTOCOL['seeds']:
            r = json.loads((args.runs/f'{mode}-{seed}.json').read_text())
            if (r['plan_sha256'] != digest(plan) or r['seed'] != seed or r['mode'] != mode or
                    r['checkpoint_sha256'] != file_hash(args.runs/f'{mode}-{seed}.pt')):
                raise ValueError('Fit/checkpoint identity mismatch')
            fits.append(r)
        methods[mode] = {'fits': fits, 'mean_metrics': {k: float(np.mean([r['metrics'][k] for r in fits]))
                                                      for k in fits[0]['metrics']}}
    control = max(('metric', 'feedforward'), key=lambda m: methods[m]['mean_metrics']['id_accuracy'])
    candidate = max(('recurrent', 'sparse_recurrent'), key=lambda m: methods[m]['mean_metrics']['id_accuracy'])
    delta = {k: methods[candidate]['mean_metrics'][k]-methods[control]['mean_metrics'][k]
             for k in ('id_accuracy', 'balanced_utility')}
    wins = sum(a['metrics']['id_accuracy'] > b['metrics']['id_accuracy'] for a, b in
               zip(methods[candidate]['fits'], methods[control]['fits'], strict=True))
    ratio = (methods[candidate]['fits'][0]['cost']['matrix_multiply_accumulates_per_query'] /
             methods[control]['fits'][0]['cost']['matrix_multiply_accumulates_per_query'])
    gate = PROTOCOL['continuation']
    passed = (delta['id_accuracy'] >= gate['minimum_accuracy_gain_over_best_control'] and
              delta['balanced_utility'] >= gate['minimum_utility_gain_over_best_control'] and
              wins >= gate['minimum_paired_seed_wins'] and ratio <= gate['max_matrix_cost_multiple'])
    write_new(args.out, {'status': 'development_only', 'plan': plan, 'methods': methods,
                        'selected_candidate': candidate, 'strongest_control': control, 'deltas': delta,
                        'paired_seed_wins': wins, 'matrix_cost_multiple': ratio,
                        'continuation': 'eligible_for_confirmation' if passed else 'stop_before_confirmation',
                        'confirmation_executed': False,
                        'limits': ['Development data have been reused across architecture screens.',
                                   'One dataset and only 50 OOS development examples.',
                                   'Frozen transformer backbone; this is not a new language-model architecture.',
                                   'Parameter matching does not imply equal training or inference compute.']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'train', 'report'])
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    args = parser.parse_args()
    {'freeze': freeze, 'train': train, 'report': report}[args.stage](args)


if __name__ == '__main__':
    main()
