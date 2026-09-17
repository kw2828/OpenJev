"""Development selection followed by one excluded-item CLINC confirmation."""
import argparse
import hashlib
import importlib.metadata
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from openjev.research.associative_text import AssociativeRouter, conformal_threshold
from openjev.research.text_distillation import digest, file_hash, write_new
from openjev.research.text_student import ENCODER_ID, ENCODER_REVISION

PROTOCOL = {
    'version': 'associative-text-v1', 'encoder': ENCODER_ID, 'revision': ENCODER_REVISION,
    'clinc_revision': '828f8093932c8fe6ca7936c3d2e52903b1c523de',
    'selection_seed': 19814, 'top_k': 32, 'beta': 30., 'anchor': .5, 'steps': 2,
    'encoder_max_tokens': 256, 'encoder_batch': 128, 'retrieval_batch': 128, 'cpu_threads': 4,
    'gate_quantiles': [.25, .5, .75], 'temperatures': [.02, .05, .1, .2],
    'rejection_thresholds': np.linspace(0, 1, 101).tolist(), 'alpha': .1,
    'selection_utility': '0.5 * correct accepted ID fraction + 0.5 * OOS recall - 0.02 * memory passes/query',
    'train': 'All unique known-intent training utterances; OOS train unused by all arms',
    'tune_calibration': 'Official val+oos_val split 50/50 within class after deduplication',
    'confirmation': 'Official test+oos_test, excluding all previous v1 evaluation contexts and earlier split duplicates',
    'continuation': 'Selected gated recurrent beats gated nearest utility at same gate AND is within 1pp of '
                    'always-nearest utility at <=0.5 memory passes/query; require paired lower bound >= -0.01 vs nearest',
    'scope': 'Frozen transformer encoder plus sparse recurrent decision head; not a post-transformer language model',
}


def signature():
    import openjev.research.associative_text as architecture
    return {'study': file_hash(__file__), 'architecture': file_hash(architecture.__file__)}


def text_key(text):
    return hashlib.sha256(' '.join(text.casefold().split()).encode()).hexdigest()


@torch.no_grad()
def encode(rows, model, tokenizer, device):
    result = []
    for start in range(0, len(rows), PROTOCOL['encoder_batch']):
        texts = [r['text'] for r in rows[start:start+PROTOCOL['encoder_batch']]]
        inputs = tokenizer(texts, padding=True, truncation=False, return_tensors='pt')
        if inputs['input_ids'].shape[1] > PROTOCOL['encoder_max_tokens']:
            raise ValueError('Unexpected long query; no silent truncation')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        hidden = model(**inputs).last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        pooled = (hidden*mask).sum(1)/mask.sum(1)
        result.append(torch.nn.functional.normalize(pooled, dim=1).cpu())
    return torch.cat(result)


def prepare(args):
    if args.out.exists():
        raise ValueError('Fresh output directory required')
    data = json.loads(args.source.read_text())
    prior = json.loads(args.prior.read_text())
    excluded = {text_key(r['context']) for r in prior['evaluation'] if r['task'] == 'clinc_domains'}
    classes = sorted({label for _, label in data['train']})
    seen, parts, audit = set(), {}, {}
    for split, source in [('train', data['train']), ('val', data['val']+data['oos_val']),
                          ('test', data['test']+data['oos_test'])]:
        rows, counts = [], Counter()
        for i, (text, label) in enumerate(source):
            key = text_key(text)
            if key in seen or (split == 'test' and key in excluded):
                counts['excluded_duplicate_or_prior_evaluation'] += 1
                continue
            seen.add(key)
            rows.append({'id': f'{split}:{i}', 'text': text, 'label': classes.index(label) if label != 'oos' else -1,
                         'text_sha256': key})
        parts[split] = rows
        audit[split] = {'source': len(source), 'retained': len(rows), 'exclusions': dict(counts)}
    rng = random.Random(PROTOCOL['selection_seed'])
    tune, calibration = [], []
    for label in range(-1, len(classes)):
        rows = [r for r in parts['val'] if r['label'] == label]
        rng.shuffle(rows)
        tune.extend(rows[:len(rows)//2])
        calibration.extend(rows[len(rows)//2:])
    del parts['val']
    parts['tune'], parts['calibration'] = tune, calibration
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_ID, revision=ENCODER_REVISION, local_files_only=True)
    model = AutoModel.from_pretrained(ENCODER_ID, revision=ENCODER_REVISION, local_files_only=True,
                                      attn_implementation='eager').to(args.device).eval()
    args.out.mkdir(parents=True)
    manifest = {'protocol': PROTOCOL, 'code': signature(), 'source_sha256': file_hash(args.source),
                'prior_sha256': file_hash(args.prior), 'audit': audit, 'classes': classes,
                'environment': {m: importlib.metadata.version(m) for m in ('torch', 'transformers', 'numpy')},
                'device': args.device, 'parts': {}}
    for part, rows in parts.items():
        start = time.perf_counter()
        vectors = encode(rows, model, tokenizer, args.device)
        np.save(args.out/f'{part}.npy', vectors.numpy(), allow_pickle=False)
        write_new(args.out/f'{part}.json', rows)
        manifest['parts'][part] = {'rows': len(rows), 'rows_sha256': file_hash(args.out/f'{part}.json'),
                                   'vectors_sha256': file_hash(args.out/f'{part}.npy'),
                                   'embedding_seconds': time.perf_counter()-start,
                                   'ids': [r['id'] for r in rows]}
        print(json.dumps({'part': part, 'rows': len(rows), 'seconds': manifest['parts'][part]['embedding_seconds']}), flush=True)
    write_new(args.out/'manifest.json', manifest)


def load(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['protocol'] != PROTOCOL or manifest['code'] != signature():
        raise ValueError('Frozen execution code or protocol changed')
    parts = {}
    for part, info in manifest['parts'].items():
        if (file_hash(directory/f'{part}.json') != info['rows_sha256'] or
                file_hash(directory/f'{part}.npy') != info['vectors_sha256']):
            raise ValueError('Data changed')
        rows = json.loads((directory/f'{part}.json').read_text())
        parts[part] = (torch.from_numpy(np.load(directory/f'{part}.npy', allow_pickle=False)),
                       torch.tensor([r['label'] for r in rows]), rows)
    torch.set_num_threads(PROTOCOL['cpu_threads'])
    router = AssociativeRouter(parts['train'][0], parts['train'][1], len(manifest['classes']),
                                **{k: PROTOCOL[k] for k in ('top_k', 'beta', 'anchor', 'steps')})
    return manifest, parts, router


def predict(router, query, mode, threshold=None):
    scores, support, masks, times, total_passes = [], [], [], [], 0
    for start in range(0, len(query), PROTOCOL['retrieval_batch']):
        batch = query[start:start+PROTOCOL['retrieval_batch']]
        t = time.perf_counter()
        s, confidence, active, passes = router.predict(batch, mode, threshold)
        times.append(time.perf_counter()-t)
        scores.append(s)
        support.append(confidence)
        masks.append(active)
        total_passes += passes
    return torch.cat(scores), torch.cat(support), torch.cat(masks), total_passes/len(query), sum(times)


def metrics(scores, support, gold, rejection, temperature):
    accepted = support >= rejection
    predicted = scores.argmax(1)
    known = gold >= 0
    correct = (predicted == gold) & accepted
    id_success = float(correct[known].float().mean())
    oos_recall = float((~accepted[~known]).float().mean())
    probs = torch.softmax(scores/temperature, dim=1)
    return {'id_accuracy_without_rejection': float((predicted[known] == gold[known]).float().mean()),
            'id_correct_and_accepted': id_success, 'id_rejection_rate': float((~accepted[known]).float().mean()),
            'oos_recall': oos_recall, 'balanced_utility': .5*(id_success+oos_recall),
            'id_nll': float(-probs[known, gold[known]].clamp_min(1e-12).log().mean())}


def develop(args):
    manifest, parts, router = load(args.packet)
    if args.out.exists():
        raise ValueError('Selection already exists')
    query, gold, _ = parts['tune']
    margin = router.prototype(query)[2]
    configs = [{'name': m, 'mode': m, 'gate_threshold': None} for m in ('prototype', 'nearest', 'recurrent')]
    for fraction in PROTOCOL['gate_quantiles']:
        threshold = float(torch.quantile(margin, fraction))
        configs.extend({'name': f'{mode}-{fraction}', 'mode': mode, 'gate_threshold': threshold}
                       for mode in ('gated_nearest', 'gated_recurrent'))
    for config in configs:
        scores, support, active, passes, seconds = predict(router, query, config['mode'], config['gate_threshold'])
        known = gold >= 0
        temperature = min(PROTOCOL['temperatures'], key=lambda t:
                          float(torch.nn.functional.cross_entropy(scores[known]/t, gold[known])))
        rejection = max(PROTOCOL['rejection_thresholds'], key=lambda t:
                        metrics(scores, support, gold, t, temperature)['balanced_utility'])
        config.update(temperature=temperature, rejection=rejection, memory_passes_per_query=passes,
                      head_seconds=seconds, active_fraction=float(active.float().mean()),
                      metrics=metrics(scores, support, gold, rejection, temperature))
        print(json.dumps(config), flush=True)
    selected = max([c for c in configs if c['mode'] == 'gated_recurrent'],
                   key=lambda c: c['metrics']['balanced_utility']-.02*c['memory_passes_per_query'])['name']
    write_new(args.out, {'manifest_sha256': digest(manifest), 'code': signature(), 'configs': configs,
                         'selected': selected, 'selection_uses': 'tune only; test predictions not computed'})


def interval(a, b, gold):
    rng = np.random.default_rng(9851)
    difference = np.asarray(a)-np.asarray(b)
    known, unknown = np.flatnonzero(gold >= 0), np.flatnonzero(gold < 0)
    draws = [.5*(difference[rng.choice(known, len(known))].mean()+
                 difference[rng.choice(unknown, len(unknown))].mean()) for _ in range(4000)]
    return np.percentile(draws, [2.5, 97.5]).tolist()


def confirm(args):
    manifest, parts, router = load(args.packet)
    selection = json.loads(args.selection.read_text())
    if selection['manifest_sha256'] != digest(manifest) or selection['code'] != signature() or args.out.exists():
        raise ValueError('Changed selection or existing confirmation output')
    # Record the sealed selection before making any confirmation prediction.
    write_new(args.out/'started.json', {'selection_sha256': file_hash(args.selection), 'code': signature()})
    cq, cy, _ = parts['calibration']
    query, gold, rows = parts['test']
    results, correctness = {}, {}
    for config in selection['configs']:
        mode, gate = config['mode'], config['gate_threshold']
        cs, _, _, _, _ = predict(router, cq, mode, gate)
        known = cy >= 0
        threshold = conformal_threshold(torch.softmax(cs[known]/config['temperature'], dim=1), cy[known])
        scores, support, active, passes, seconds = predict(router, query, mode, gate)
        measured = metrics(scores, support, gold, config['rejection'], config['temperature'])
        probability = torch.softmax(scores/config['temperature'], dim=1)
        sets = 1-probability <= threshold
        known = gold >= 0
        measured.update(id_set_coverage=float(sets[known, gold[known]].float().mean()),
                        id_mean_set_size=float(sets[known].sum(1).float().mean()))
        correctness[config['name']] = torch.where(known, (scores.argmax(1) == gold) &
                        (support >= config['rejection']), support < config['rejection']).numpy().astype(float)
        results[config['name']] = {'config': config, 'metrics': measured, 'head_seconds': seconds,
                                  'memory_passes_per_query': passes, 'active_fraction': float(active.float().mean()),
                                  'conformal_threshold': threshold}
        # No source passages in prediction evidence; labels and scores stay local.
        write_new(args.out/f"{config['name']}-predictions.json", [
            {'id': row['id'], 'gold': int(y), 'prediction': int(pred), 'support': float(s), 'active': bool(a)}
            for row, y, pred, s, a in zip(rows, gold, scores.argmax(1), support, active, strict=True)])
        print(json.dumps({'method': config['name'], **measured, 'passes': passes}), flush=True)
    selected = selection['selected']
    matched = selected.replace('gated_recurrent', 'gated_nearest')
    contrasts = {b: {'utility_delta': results[selected]['metrics']['balanced_utility']-
                     results[b]['metrics']['balanced_utility'],
                     'paired_95_interval': interval(correctness[selected], correctness[b], gold.numpy())}
                 for b in ('nearest', matched)}
    passed = (contrasts[matched]['utility_delta'] > 0 and contrasts['nearest']['utility_delta'] >= -.01
              and contrasts['nearest']['paired_95_interval'][0] >= -.01
              and results[selected]['memory_passes_per_query'] <= .5)
    write_new(args.out/'summary.json', {'protocol': PROTOCOL, 'selection': selection, 'manifest_sha256': digest(manifest),
               'results': results, 'selected_contrasts': contrasts, 'continuation': 'pass' if passed else 'fail',
               'test_counts': {'known': int((gold >= 0).sum()), 'oos': int((gold < 0).sum())},
               'claim_limits': ['Development-selected model, exploratory paired intervals.',
                                'CLINC excludes old evaluation items; not the official full-test score.',
                                'ID-only conformal calibration is not an OOS guarantee.',
                                'Head timing is batched CPU time and excludes the shared transformer encoder.',
                                'No seed variability for deterministic heads; encoder remains pretrained.']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'develop', 'confirm'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--prior', type=Path)
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='mps', choices=['mps', 'cpu', 'cuda'])
    args = parser.parse_args()
    {'prepare': prepare, 'develop': develop, 'confirm': confirm}[args.stage](args)


if __name__ == '__main__':
    main()
