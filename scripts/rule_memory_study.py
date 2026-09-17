"""Frozen RuleTaker development comparison; no official test or teacher calls."""
import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from openjev.research.rule_memory import MODES, RuleMemoryHead, english_inputs
from openjev.research.text_distillation import digest, file_hash, write_new
from openjev.research.text_student import ENCODER_ID, ENCODER_REVISION

PARTS = ('train', 'dev_in', 'dev_shift')
EXPECTED = {'train': 'f94fb239544ef019f4d457ed12c8ea9f1b8a84a89ebaf3783c926aafc96d7937',
            'dev_in': '89fe954596e225a4131e154da5304feb3493d48e8589fd254cca82056cb21956',
            'dev_shift': '32e4f0eea75313b5e485c6c42295cd724ab991b4b6f5c141ddeb6623b2c50311'}
PROTOCOL = {
    'version': 'rule-memory-v1', 'modes': list(MODES), 'seeds': [17, 29, 43],
    'encoder': ENCODER_ID, 'encoder_revision': ENCODER_REVISION,
    'encoder_max_tokens': 256, 'encoder_batch': 128, 'latent_dim': 128,
    'epochs': 20, 'batch_size': 256, 'learning_rate': .001, 'weight_decay': .0001,
    'clip_gradient': 1., 'cpu_threads': 4, 'label_order': ['false', 'true'],
    'loss': 'Final binary cross entropy; same minibatch order for all heads within a seed',
    'features': 'Frozen normalized mean-pooled MiniLM per English clause and assertion; no truncation',
    'selection': 'Final epoch only; all three seeds; no checkpoint or seed selection',
    'controls': ['question_only', 'mean_context', 'single_read'],
    'candidates': ['recurrent_3', 'recurrent_6', 'coverage_3'],
    'coverage': 'Subtract cumulative prior attention weights from logits with fixed coefficient 1',
    'latency_repeats': 200, 'latency_warmup': 30, 'bootstrap_draws': 2000, 'bootstrap_seed': 5279,
    'continuation': {'min_shift_accuracy_gain': .02, 'max_in_accuracy_loss': .01,
                     'max_shift_nll_increase': 0., 'min_paired_seed_wins': 2,
                     'min_world_bootstrap_lower_bound': 0., 'max_head_mac_ratio': 3.,
                     'scope': 'Development selection, not independent evidence or a novelty claim'},
}


def signature():
    import openjev.research.rule_memory as architecture
    return {'driver': file_hash(__file__), 'architecture': file_hash(architecture.__file__)}


def load_packet(packet):
    manifest = json.loads((packet/'manifest.json').read_text())
    data, seen = {}, set()
    for part in PARTS:
        if file_hash(packet/f'{part}.json') != EXPECTED[part]:
            raise ValueError('Need the exact audited depth-filtered packet')
        worlds = json.loads((packet/f'{part}.json').read_text())
        keys = {w['world_sha256'] for w in worlds}
        if seen & keys or len(keys) != len(worlds):
            raise ValueError('Overlapping worlds')
        seen.update(keys)
        data[part] = worlds
    return manifest, data


def freeze(args):
    manifest, _ = load_packet(args.packet)
    write_new(args.out, {'protocol': PROTOCOL, 'code': signature(), 'packet_sha256': digest(manifest),
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'environment': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'transformers')},
        'platform': platform.platform(), 'python': platform.python_version()})


def checked(args):
    plan = json.loads(args.plan.read_text())
    manifest, data = load_packet(args.packet)
    if plan['protocol'] != PROTOCOL or plan['code'] != signature() or plan['packet_sha256'] != digest(manifest):
        raise ValueError('Frozen source, protocol or packet changed')
    torch.set_num_threads(PROTOCOL['cpu_threads'])
    return plan, data


@torch.no_grad()
def prepare(args):
    plan, data = checked(args)
    if args.out.exists():
        raise ValueError('Feature directory exists; no overwrite')
    args.out.mkdir(parents=True)
    write_new(args.out/'started.json', {'plan_sha256': digest(plan), 'device': args.device})
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_ID, revision=ENCODER_REVISION, local_files_only=True)
    model = AutoModel.from_pretrained(ENCODER_ID, revision=ENCODER_REVISION, local_files_only=True,
                                      attn_implementation='eager').to(args.device).eval()
    texts = sorted({t for worlds in data.values() for w in worlds for group in english_inputs(w) for t in group})
    lookup = {t: i for i, t in enumerate(texts)}
    vectors, token_counts = [], []
    started = time.perf_counter()
    for start in range(0, len(texts), PROTOCOL['encoder_batch']):
        inputs = tokenizer(texts[start:start+PROTOCOL['encoder_batch']], padding=True,
                           truncation=False, return_tensors='pt')
        if inputs['input_ids'].shape[1] > PROTOCOL['encoder_max_tokens']:
            raise ValueError('Sentence exceeds token budget; no silent truncation')
        token_counts.extend(inputs['attention_mask'].sum(1).tolist())
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        hidden = model(**inputs).last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        vectors.append(F.normalize((hidden*mask).sum(1)/mask.sum(1), dim=1).cpu())
        if start % 12800 == 0:
            print(json.dumps({'encoded': min(start+128, len(texts)), 'total': len(texts)}), flush=True)
    table = torch.cat(vectors).numpy()
    elapsed = time.perf_counter()-started
    manifest = {'plan_sha256': digest(plan), 'device': args.device, 'encoding_seconds': elapsed,
        'unique_english_strings': len(texts), 'unique_unpadded_tokens': sum(token_counts),
        'max_sentence_tokens': max(token_counts), 'parts': {}, 'test_opened': False,
        'cache_scope': 'Deduplicated deterministic frozen features only; no label fitting in encoding'}
    for part, worlds in data.items():
        max_slots = max(len(english_inputs(w)[0]) for w in worlds)
        memory = np.zeros((len(worlds), max_slots, table.shape[1]), dtype=np.float32)
        mask = np.zeros((len(worlds), max_slots), dtype=bool)
        queries, rows, indices = [], [], []
        for i, w in enumerate(worlds):
            clauses, questions = english_inputs(w)
            memory[i, :len(clauses)] = table[[lookup[t] for t in clauses]]
            mask[i, :len(clauses)] = True
            for question, q in zip(questions, w['questions'], strict=True):
                queries.append(table[lookup[question]])
                indices.append(i)
                rows.append({'world': i, 'question_id': q['id'], 'gold': PROTOCOL['label_order'].index(q['gold']),
                             'depth': q['depth']})
        path = args.out/f'{part}.npz'
        np.savez(path, memory=memory, mask=mask, query=np.stack(queries), world=np.array(indices, dtype=np.int64))
        write_new(args.out/f'{part}.json', rows)
        manifest['parts'][part] = {'arrays_sha256': file_hash(path), 'rows_sha256': file_hash(args.out/f'{part}.json'),
            'worlds': len(worlds), 'questions': len(rows), 'max_slots': max_slots,
            'mean_slots_per_query': float(mask.sum(1)[indices].mean()),
            'mean_slots_per_world': float(mask.sum(1).mean())}
    write_new(args.out/'manifest.json', manifest)
    print(json.dumps(manifest), flush=True)


def load_features(args, plan):
    manifest = json.loads((args.features/'manifest.json').read_text())
    if manifest['plan_sha256'] != digest(plan):
        raise ValueError('Feature plan mismatch')
    parts = {}
    for part in PARTS:
        info = manifest['parts'][part]
        if (info['arrays_sha256'] != file_hash(args.features/f'{part}.npz') or
                info['rows_sha256'] != file_hash(args.features/f'{part}.json')):
            raise ValueError('Feature hash mismatch')
        with np.load(args.features/f'{part}.npz', allow_pickle=False) as data:
            arrays = {k: torch.from_numpy(data[k]) for k in data.files}
        rows = json.loads((args.features/f'{part}.json').read_text())
        arrays['gold'] = torch.tensor([r['gold'] for r in rows])
        arrays['depth'] = torch.tensor([r['depth'] for r in rows])
        parts[part] = arrays
    return manifest, parts


def batch_inputs(data, indices):
    w = data['world'][indices]
    return data['query'][indices], data['memory'][w], data['mask'][w]


def metrics(logits, gold, depths):
    probabilities = logits.softmax(-1)
    correct = (logits.argmax(1) == gold).float()
    return {'questions': len(gold), 'accuracy': float(correct.mean()),
        'nll': float(F.cross_entropy(logits, gold)),
        'multiclass_brier': float(((probabilities-F.one_hot(gold, 2))**2).sum(1).mean()),
        'by_depth': {str(d): {'n': int((depths == d).sum()), 'accuracy': float(correct[depths == d].mean())}
                     for d in sorted(set(depths.tolist()))},
        'by_label': {PROTOCOL['label_order'][c]: float(correct[gold == c].mean()) for c in (0, 1)}}


@torch.no_grad()
def predict(model, data):
    return torch.cat([model(*batch_inputs(data, index)) for index in
                      torch.arange(len(data['query'])).split(PROTOCOL['batch_size'])])


@torch.no_grad()
def latency(model, data):
    # First 200 queries in frozen order, preserving the part's padded batch shape.
    inputs = [batch_inputs(data, torch.tensor([i])) for i in range(PROTOCOL['latency_repeats'])]
    for i in range(PROTOCOL['latency_warmup']):
        model(*inputs[i])
    elapsed = []
    for x in inputs:
        t = time.perf_counter()
        model(*x)
        elapsed.append((time.perf_counter()-t)*1e6)
    return {'median_us': float(np.median(elapsed)), 'p95_us': float(np.percentile(elapsed, 95)),
            'scope': 'CPU, batch one, 4 threads, precomputed embeddings; includes memory projections, excludes encoder'}


def train(args):
    plan, _ = checked(args)
    manifest, parts = load_features(args, plan)
    if args.out.exists():
        raise ValueError('Fit directory exists; no overwrite or silent retry')
    if args.device != 'cpu':
        raise ValueError('Frozen head comparison uses CPU')
    args.out.mkdir(parents=True)
    write_new(args.out/'started.json', {'plan_sha256': digest(plan), 'features_sha256': digest(manifest)})
    data = parts['train']
    for seed in PROTOCOL['seeds']:
        for mode in MODES:
            model = RuleMemoryHead(mode, seed=seed, latent_dim=PROTOCOL['latent_dim']).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL['learning_rate'],
                                          weight_decay=PROTOCOL['weight_decay'])
            rng = torch.Generator().manual_seed(seed)
            started = time.perf_counter()
            history = []
            for epoch in range(PROTOCOL['epochs']):
                total_loss, updates = 0., 0
                order = torch.randperm(len(data['query']), generator=rng)
                for index in order.split(PROTOCOL['batch_size']):
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(*batch_inputs(data, index)), data['gold'][index])
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL['clip_gradient'], error_if_nonfinite=True)
                    optimizer.step()
                    total_loss += float(loss.detach())*len(index)
                    updates += 1
                history.append({'epoch': epoch+1, 'updates': updates, 'loss': total_loss/len(order)})
                if (epoch+1) % 5 == 0:
                    print(json.dumps({'mode': mode, 'seed': seed, **history[-1]}), flush=True)
            elapsed = time.perf_counter()-started
            model.eval()
            checkpoint = args.out/f'{mode}-{seed}.pt'
            torch.save(model.state_dict(), checkpoint)
            result = {'mode': mode, 'seed': seed, 'plan_sha256': digest(plan), 'training_seconds': elapsed,
                'checkpoint_sha256': file_hash(checkpoint), 'history': history, 'parts': {}}
            predictions = {}
            for part in ('dev_in', 'dev_shift'):
                dev = parts[part]
                logits = predict(model, dev)
                predictions[part] = logits.numpy()
                result['parts'][part] = {'metrics': metrics(logits, dev['gold'], dev['depth']),
                    'latency': latency(model, dev), 'cost': model.cost(dev['memory'].shape[1])}
            pred_path = args.out/f'{mode}-{seed}-logits.npz'
            np.savez(pred_path, **predictions)
            result['logits_sha256'] = file_hash(pred_path)
            write_new(args.out/f'{mode}-{seed}.json', result)
            print(json.dumps({'mode': mode, 'seed': seed, 'training_seconds': elapsed,
                              'accuracy': {p: r['metrics']['accuracy'] for p, r in result['parts'].items()}}), flush=True)
    write_new(args.out/'completed.json', {'plan_sha256': digest(plan), 'status': 'completed',
                                         'fits': len(MODES)*len(PROTOCOL['seeds'])})


def world_interval(difference, world, draws=2000, seed=5279):
    """Pair predictions, average training seeds, then resample whole worlds."""
    difference, world = np.asarray(difference), np.asarray(world)
    if difference.ndim != 1 or difference.shape != world.shape or not len(world):
        raise ValueError('Need aligned nonempty query differences and world IDs')
    unique, inverse = np.unique(world, return_inverse=True)
    totals = np.bincount(inverse, weights=difference)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(unique), size=(draws, len(unique)))
    bootstrap = totals[sample].sum(1)/counts[sample].sum(1)
    return {'mean': float(difference.mean()), 'lower': float(np.quantile(bootstrap, .025)),
            'upper': float(np.quantile(bootstrap, .975)), 'worlds': len(unique), 'draws': draws,
            'scope': 'Descriptive paired world bootstrap after averaging fixed seeds; selection and seed uncertainty excluded'}


def report(args):
    plan, _ = checked(args)
    feature_manifest, parts = load_features(args, plan)
    expected = {'plan_sha256': digest(plan), 'status': 'completed', 'fits': len(MODES)*len(PROTOCOL['seeds'])}
    if json.loads((args.runs/'completed.json').read_text()) != expected:
        raise ValueError('All planned fits must complete before selection')
    methods, correctness = {}, {}
    for mode in MODES:
        fits, correctness[mode] = [], {}
        for seed in PROTOCOL['seeds']:
            fit = json.loads((args.runs/f'{mode}-{seed}.json').read_text())
            if (fit['plan_sha256'] != digest(plan) or fit['mode'] != mode or fit['seed'] != seed or
                    fit['checkpoint_sha256'] != file_hash(args.runs/f'{mode}-{seed}.pt') or
                    fit['logits_sha256'] != file_hash(args.runs/f'{mode}-{seed}-logits.npz')):
                raise ValueError('Fit identity or prediction hash mismatch')
            with np.load(args.runs/f'{mode}-{seed}-logits.npz', allow_pickle=False) as logits:
                for part in ('dev_in', 'dev_shift'):
                    recomputed = metrics(torch.from_numpy(logits[part]), parts[part]['gold'], parts[part]['depth'])
                    if recomputed != fit['parts'][part]['metrics']:
                        raise ValueError('Metric replay mismatch')
                    correctness[mode].setdefault(part, []).append(
                        (logits[part].argmax(1) == parts[part]['gold'].numpy()).astype(float))
            fits.append(fit)
        means = {}
        for part in ('dev_in', 'dev_shift'):
            means[part] = {k: float(np.mean([f['parts'][part]['metrics'][k] for f in fits]))
                           for k in ('accuracy', 'nll', 'multiclass_brier')}
        methods[mode] = {'fits': fits, 'means': means}
    control = max(PROTOCOL['controls'], key=lambda m: methods[m]['means']['dev_shift']['accuracy'])
    candidate = max(PROTOCOL['candidates'], key=lambda m: methods[m]['means']['dev_shift']['accuracy'])
    deltas = {p: {k: methods[candidate]['means'][p][k]-methods[control]['means'][p][k]
                  for k in ('accuracy', 'nll', 'multiclass_brier')} for p in ('dev_in', 'dev_shift')}
    intervals = {p: world_interval(np.mean(correctness[candidate][p], axis=0)-
                                   np.mean(correctness[control][p], axis=0), parts[p]['world'].numpy(),
                                   PROTOCOL['bootstrap_draws'], PROTOCOL['bootstrap_seed'])
                 for p in ('dev_in', 'dev_shift')}
    wins = sum(a.mean() > b.mean() for a, b in zip(correctness[candidate]['dev_shift'],
                                                 correctness[control]['dev_shift'], strict=True))
    ratio = (methods[candidate]['fits'][0]['parts']['dev_shift']['cost']['matrix_macs_per_query']/
             methods[control]['fits'][0]['parts']['dev_shift']['cost']['matrix_macs_per_query'])
    gate = PROTOCOL['continuation']
    checks = {'shift_gain': deltas['dev_shift']['accuracy'] >= gate['min_shift_accuracy_gain'],
              'in_retention': deltas['dev_in']['accuracy'] >= -gate['max_in_accuracy_loss'],
              'shift_nll': deltas['dev_shift']['nll'] <= gate['max_shift_nll_increase'],
              'seed_wins': wins >= gate['min_paired_seed_wins'],
              'world_interval': intervals['dev_shift']['lower'] > gate['min_world_bootstrap_lower_bound'],
              'head_cost': ratio <= gate['max_head_mac_ratio']}
    write_new(args.out, {'status': 'development_only', 'plan': plan, 'features': feature_manifest,
        'methods': methods, 'strongest_control': control, 'selected_candidate': candidate,
        'deltas': deltas, 'paired_world_intervals': intervals, 'paired_seed_wins': wins,
        'head_mac_ratio': ratio, 'continuation_checks': checks,
        'continuation': 'eligible_for_confirmation' if all(checks.values()) else 'stop_before_confirmation',
        'confirmation_executed': False,
        'limits': ['Frozen transformer encoder plus established memory-network mechanisms, not novelty evidence.',
                   'Development selection only; world bootstrap excludes model selection and training-seed uncertainty.',
                   'Changing question depth also changes source-world distribution.',
                   'Question-only and mean controls use fewer active parameters; attention variants match exactly.',
                   'CPU latency excludes sentence encoder; offline feature cache is not end-to-end serving latency.',
                   'Astra teacher calls and teacher-student training have not run.']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'prepare', 'train', 'report'])
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--plan', type=Path)
    p.add_argument('--features', type=Path)
    p.add_argument('--runs', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    args = p.parse_args()
    {'freeze': freeze, 'prepare': prepare, 'train': train, 'report': report}[args.stage](args)
