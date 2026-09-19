"""Development comparison of streamed text memories on supplied-schema decisions.

Only official train/dev are read. This is a categorical SGD subtask, not full DST.
Raw text, embeddings, weights and individual predictions remain in ignored runs/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
ENCODER = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
METHODS = ['current', 'gru', 'attention', 'gated_delta', 'kalman', 'innovation_kalman', 'carry']
SEEDS = [1729, 2718, 3141]
TRAIN_DIALOGUES = 2048
CONFIG = {'epochs': 12, 'batch_size': 32, 'learning_rate': .001,
          'weight_decay': .0001, 'gradient_clip': 1., 'threads': 4,
          'width': 16, 'gru_width': 128, 'innovation_eta': .1,
          'train_dialogues': TRAIN_DIALOGUES, 'seeds': SEEDS, 'methods': METHODS,
          'loss': 'equal weight across changed, assigned-retention, unmentioned-retention bins',
          'selection': 'final epoch, all seeds; no development selection or test access'}
PRACTICAL_CHECKS = {'panels': ['seen', 'unseen'], 'primary': 'innovation_kalman',
                    'macro_definition': 'equal three-stratum mean',
                    'macro_gain_over_kalman': .01, 'micro_nll_no_worse_than': 'kalman',
                    'strict_paired_macro_wins_over_kalman': 2,
                    'macro_no_worse_than': 'gated_delta',
                    'conventional_controls': ['gru', 'attention', 'carry'],
                    'maximum_macro_deficit_to_best_conventional': .01,
                    'required_complete_fits': 21, 'total_panel_checks': 10}
SOURCES = ['scripts/study_dialogue_memory.py', 'scripts/prepare_sgd_state.py',
           'src/openjev/research/dialogue_state_data.py',
           'src/openjev/research/dialogue_fast_memory.py',
           'src/openjev/research/dialogue_carry.py',
           'scripts/report_dialogue_memory.py',
           'tests/test_dialogue_state_data.py', 'tests/test_dialogue_fast_memory.py',
           'tests/test_dialogue_carry.py', 'tests/test_study_dialogue_memory.py',
           'tests/test_report_dialogue_memory.py',
           'research/dialogue-memory-protocol.md']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def read_lines(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def loss_bin(value):
    if value == 'unmentioned_retention':
        return 0
    if value == 'assigned_retention':
        return 1
    if value in ('first_assignment', 'revision', 'clear'):
        return 2
    raise ValueError(f'Unknown transition bin: {value}')


def assemble(data):
    """Label fields remain separate from actor inputs; select by dialogue ID only."""
    catalogs = json.loads((data / 'catalog.json').read_text())
    texts, text_lookup, queries, query_lookup = [], {}, [], {}

    def text_index(text):
        if text not in text_lookup:
            text_lookup[text] = len(texts)
            texts.append(text)
        return text_lookup[text]

    for split in ('train', 'dev'):
        for q in catalogs[split]:
            key = (split, q['query_id'])
            assert key not in query_lookup
            cs = q['candidates']
            assert cs[0]['id'] == 'reserved:NOT_MENTIONED'
            assert 3 <= len(cs) <= 12 and len({c['id'] for c in cs}) == len(cs)
            query_lookup[key] = len(queries)
            queries.append({'id': q['query_id'], 'split': split, 'service': q['service'],
                            'slot': q['slot'], 'text': text_index(q['query_text']),
                            'candidates': [text_index(c['text']) for c in cs],
                            'candidate_ids': [c['id'] for c in cs],
                            'candidate_values': [c.get('value') for c in cs]})
    cohorts = {}
    for split in ('train', 'dev'):
        ds = read_lines(data / f'{split}-dialogues.jsonl')
        if split == 'train':
            ds = sorted(ds, key=lambda d: hashlib.sha256(
                ('openjev-sgd-v1:' + d['dialogue_id']).encode()).hexdigest())[:TRAIN_DIALOGUES]
            if len(ds) != TRAIN_DIALOGUES:
                raise ValueError('Insufficient train dialogues')
        ds = sorted(ds, key=lambda d: d['dialogue_id'])
        by_id = {}
        for d in ds:
            turn_ids, features, user_text = {}, [], []
            for step, t in enumerate(d['user_turns']):
                ti = t['turn_index']
                si = t['previous_system_turn_index']
                u = d['turns'][ti]
                assert u['speaker'] == 'USER'
                system = '' if si is None else d['turns'][si]['utterance']
                assert si is None or (si < ti and d['turns'][si]['speaker'] == 'SYSTEM')
                features.append(text_index('System: ' + system + '\nUser: ' + u['utterance']))
                user_text.append(u['utterance'])
                turn_ids[ti] = step
            by_id[d['dialogue_id']] = {'id': d['dialogue_id'], 'turns': features,
                                      'user_text': user_text, 'queries': [], '_time': turn_ids}
        for r in read_lines(data / f'{split}-labels.jsonl'):
            if r['dialogue_id'] not in by_id:
                continue
            d = by_id[r['dialogue_id']]
            qi = query_lookup[(split, r['query_id'])]
            assert queries[qi]['candidate_ids'][r['label_index']] == r['label_id']
            d['queries'].append({'query': qi, 'time': d['_time'][r['turn_index']],
                                 'label': r['label_index'], 'bin': r['bin'],
                                 'unseen': r['unseen_service'], 'dontcare': r['is_dontcare']})
        # Dialogues without categorical questions contribute no model loss.
        for d in by_id.values():
            del d['_time']
        cohorts[split] = [d for d in by_id.values() if d['queries']]
    return texts, queries, cohorts


def encode(args):
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        if not args.data_sha256 or sha(data / 'completed.json') != args.data_sha256:
            raise ValueError('Externally pinned data receipt required')
        prepared = json.loads((data / 'completed.json').read_text())
        assert prepared['status'] == 'completed' and not (data / 'failed.json').exists()
        assert prepared['test_contents_accessed'] is False
        for name, item in prepared['files'].items():
            if sha(data / name) != item['sha256']:
                raise ValueError('Changed prepared data: ' + name)
        for name, digest in prepared['implementation_sha256'].items():
            if sha(ROOT / name) != digest:
                raise ValueError('Changed data implementation: ' + name)
        (out / 'encoder-source.py').write_bytes(Path(__file__).read_bytes())
        inputs = {p.name: sha(p) for p in sorted(data.glob('*')) if p.is_file()}
        texts, queries, cohorts = assemble(data)
        write(out / 'encoder-plan.json', {'encoder': ENCODER, 'revision': REVISION,
              'source_sha256': sha(__file__), 'input_sha256': inputs, 'text_count': len(texts),
              'chunk_tokens': 254, 'chunking': 'nonoverlapping; token-weighted mean, no truncation',
              'device': args.device, 'dtype': 'float32', 'batch_size': 128,
              'train_selection': '2048 smallest SHA256(openjev-sgd-v1:dialogue_id), before label filtering',
              'label_access': 'train and dev only; no official test', 'torch': torch.__version__})
        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(ENCODER, revision=REVISION, local_files_only=True)
        model = AutoModel.from_pretrained(ENCODER, revision=REVISION, local_files_only=True,
                                          attn_implementation='eager').to(args.device).eval()
        model.requires_grad_(False)
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        weights = np.zeros(len(texts), dtype=np.int64)
        chunks, overlength = [], 0
        tokens = tokenizer(texts, add_special_tokens=False, truncation=False)['input_ids']
        for i, ids in enumerate(tokens):
            overlength += int(len(ids) > 254)
            if not ids:
                raise ValueError('Empty encoding input')
            for start in range(0, len(ids), 254):
                part = ids[start:start + 254]
                chunks.append((i, [tokenizer.cls_token_id, *part, tokenizer.sep_token_id], len(part)))
        chunk_tokens = 0
        with torch.inference_mode():
            for start in range(0, len(chunks), 128):
                batch = chunks[start:start + 128]
                encoded = tokenizer.pad({'input_ids': [r[1] for r in batch]},
                                         padding=True, return_tensors='pt')
                encoded = {k: v.to(args.device) for k, v in encoded.items()}
                hidden = model(**encoded).last_hidden_state
                mask = encoded['attention_mask'].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                values = pooled.cpu().numpy()
                for (i, ids, w), v in zip(batch, values, strict=True):
                    vectors[i] += w * v
                    weights[i] += w
                    chunk_tokens += len(ids)
                if start % (128 * 100) == 0:
                    print(json.dumps({'encoded_chunks': min(start + 128, len(chunks)),
                                      'total_chunks': len(chunks)}), flush=True)
        vectors /= weights[:, None]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or np.any(norms <= 1e-12):
            raise ValueError('Invalid frozen embeddings')
        vectors /= norms
        np.save(out / 'features.npy', vectors, allow_pickle=False)
        write(out / 'packet.json', {'queries': queries, 'cohorts': cohorts})
        # Bind actual cached model/tokenizer bytes rather than only a repository label.
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(ENCODER, revision=REVISION, local_files_only=True))
        model_files = {p.name: sha(p) for p in sorted(snapshot.iterdir()) if p.is_file()}
        for name, digest in inputs.items():
            assert sha(data / name) == digest, name
        write(out / 'completed.json', {'status': 'completed', 'unique_texts': len(texts),
              'encoder_sequences': len(chunks), 'encoder_tokens_with_special': chunk_tokens,
              'overlength_texts_chunked': overlength, 'truncated_tokens': 0,
              'cohorts': {s: {'dialogues': len(ds), 'queries': sum(len(d['queries']) for d in ds),
                            'bins': dict(Counter(q['bin'] for d in ds for q in d['queries']))}
                          for s, ds in cohorts.items()},
              'wall_seconds': time.perf_counter() - started,
              'model_files_sha256': model_files,
              'files': {n: sha(out / n) for n in
                        ['encoder-plan.json', 'encoder-source.py', 'features.npy', 'packet.json']}})
    except BaseException as e:
        write(out / 'failed.json', {'status': 'failed', 'error_type': type(e).__name__,
                                   'error': str(e), 'wall_seconds': time.perf_counter() - started})
        raise


def load_packet(path):
    path = Path(path)
    receipt = json.loads((path / 'completed.json').read_text())
    assert receipt['status'] == 'completed' and not (path / 'failed.json').exists()
    for name, digest in receipt['files'].items():
        if sha(path / name) != digest:
            raise ValueError('Changed packet member: ' + name)
    return json.loads((path / 'packet.json').read_text()), np.load(path / 'features.npy', allow_pickle=False)


def make_batch(ds, queries, features):
    b, t, q = len(ds), max(len(d['turns']) for d in ds), max(len(d['queries']) for d in ds)
    c = max(len(queries[r['query']]['candidates']) for d in ds for r in d['queries'])
    turns = np.zeros((b, t, 384), np.float32)
    valid = np.zeros((b, t), bool)
    qe = np.zeros((b, q, 384), np.float32)
    ce = np.zeros((b, q, c, 384), np.float32)
    times = np.zeros((b, q), np.int64)
    cmask = np.zeros((b, q, c), bool)
    labels = np.full((b, q), -100, np.int64)
    bins = np.zeros((b, q), np.int64)
    for i, d in enumerate(ds):
        turns[i, :len(d['turns'])] = features[d['turns']]
        valid[i, :len(d['turns'])] = True
        for j, r in enumerate(d['queries']):
            entry = queries[r['query']]
            qe[i, j] = features[entry['text']]
            n = len(entry['candidates'])
            ce[i, j, :n] = features[entry['candidates']]
            cmask[i, j, :n] = True
            times[i, j], labels[i, j], bins[i, j] = r['time'], r['label'], loss_bin(r['bin'])
        # Padded queries are ignored by the loss; keep a defined NONE-only distribution.
        cmask[i, len(d['queries']):, 0] = True
    actor = [torch.from_numpy(x) for x in (turns, valid, qe, ce, times, cmask)]
    return actor, torch.from_numpy(labels), torch.from_numpy(bins)


def make_model(method):
    if method == 'carry':
        from openjev.research.dialogue_carry import LearnedCarry
        return LearnedCarry()
    from openjev.research.dialogue_fast_memory import DialogueMemory
    return DialogueMemory(method, width=CONFIG['width'], gru_width=CONFIG['gru_width'],
                          innovation_eta=CONFIG['innovation_eta'])


def tensor_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def freeze(args):
    packet, _ = load_packet(args.packet)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    counts = Counter(loss_bin(q['bin']) for d in packet['cohorts']['train'] for q in d['queries'])
    if set(counts) != {0, 1, 2}:
        raise ValueError('All three training strata required')
    total = sum(counts.values())
    plan = {'study': 'dialogue-memory-v1', 'config': CONFIG,
            'packet_completed_sha256': sha(Path(args.packet) / 'completed.json'),
            'source_sha256': {name: sha(ROOT / name) for name in SOURCES},
            'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                        'numpy': np.__version__, 'platform': platform.platform()},
            'loss_counts': dict(counts), 'loss_weights': [total / (3 * counts[i]) for i in range(3)],
            'practical_checks': PRACTICAL_CHECKS,
            'scope': 'Development only, supplied categorical schema queries, fixed pretrained encoder',
            'prospective_checks': 'See frozen research/dialogue-memory-protocol.md'}
    write(out / 'plan.json', plan)
    print(json.dumps({'plan_sha256': sha(out / 'plan.json')}), flush=True)


def validate_plan(args):
    run = Path(args.out)
    if sha(run / 'plan.json') != args.plan_sha256:
        raise ValueError('External plan digest mismatch')
    plan = json.loads((run / 'plan.json').read_text())
    if plan['config'] != CONFIG or plan['practical_checks'] != PRACTICAL_CHECKS:
        raise ValueError('Configuration changed')
    runtime = {'python': platform.python_version(), 'torch': torch.__version__,
               'numpy': np.__version__, 'platform': platform.platform()}
    if plan['runtime'] != runtime:
        raise ValueError('Runtime changed since freeze')
    for name, digest in plan['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    if sha(Path(args.packet) / 'completed.json') != plan['packet_completed_sha256']:
        raise ValueError('Feature receipt changed')
    return plan


def evaluate(model, ds, queries, features, destination):
    model.eval()
    arrays = {k: [] for k in ['probabilities', 'labels', 'choice', 'bin', 'unseen', 'dialogue', 'time', 'query']}
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(ds), CONFIG['batch_size']):
            subset = ds[start:start + CONFIG['batch_size']]
            actor, _, _ = make_batch(subset, queries, features)
            logits = model(*actor)
            probs = logits.softmax(-1).numpy()
            for i, d in enumerate(subset):
                for j, r in enumerate(d['queries']):
                    n = len(queries[r['query']]['candidates'])
                    p = probs[i, j, :n]
                    if not np.isfinite(p).all() or not np.isclose(p.sum(), 1., atol=1e-5):
                        raise ValueError('Invalid evaluation probabilities')
                    padded = np.zeros(12, np.float32)
                    padded[:n] = p
                    for k, value in [('probabilities', padded), ('labels', r['label']),
                                     ('choice', int(p.argmax())), ('bin', r['bin']),
                                     ('unseen', r['unseen']), ('dialogue', d['id']),
                                     ('time', r['time']), ('query', r['query'])]:
                        arrays[k].append(value)
    np.savez_compressed(destination, **{k: np.asarray(v) for k, v in arrays.items()})
    return {'queries': len(arrays['labels']), 'wall_seconds': time.perf_counter() - started}


def train(args):
    plan = validate_plan(args)
    packet, features = load_packet(args.packet)
    run = Path(args.out) / 'fits'
    run.mkdir(exist_ok=False)
    torch.set_num_threads(CONFIG['threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data, queries = packet['cohorts']['train'], packet['queries']
    weight = torch.tensor(plan['loss_weights'], dtype=torch.float32)
    records, active = [], None
    started = time.perf_counter()
    try:
        for seed in SEEDS:
            # Independent generator keeps order paired despite architecture-specific initialization.
            order = np.random.default_rng(seed)
            orders = [order.permutation(len(data)) for _ in range(CONFIG['epochs'])]
            matrix_initial = None
            for method in METHODS:
                active = run / f'{method}-{seed}'
                active.mkdir(exist_ok=False)
                torch.manual_seed(seed)
                model = make_model(method)
                initial_digest = tensor_digest(model)
                if method in ('gated_delta', 'kalman', 'innovation_kalman'):
                    if matrix_initial is None:
                        matrix_initial = initial_digest
                    if initial_digest != matrix_initial:
                        raise ValueError('Matrix-memory initial tensors were not paired')
                optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                                              weight_decay=CONFIG['weight_decay'])
                losses, updates, native_queries = [], 0, 0
                actor_work = Counter()
                fit_start = time.perf_counter()
                model.train()
                for epoch, indices in enumerate(orders):
                    loss_total, count = 0., 0
                    for start in range(0, len(data), CONFIG['batch_size']):
                        subset = [data[int(i)] for i in indices[start:start + CONFIG['batch_size']]]
                        actor, labels, bins = make_batch(subset, queries, features)
                        b, t = actor[1].shape
                        q, c = actor[5].shape[1:]
                        actor_work.update({'real_turns': int(actor[1].sum()),
                                           'padded_turn_positions': b * t,
                                           'padded_query_positions': b * q,
                                           'padded_candidate_positions': b * q * c})
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(*actor)
                        valid = labels != -100
                        values = nn.functional.cross_entropy(logits[valid], labels[valid], reduction='none')
                        loss = (values * weight[bins[valid]]).mean()
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite training loss')
                        loss.backward()
                        norm = nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'],
                                                       error_if_nonfinite=True)
                        if not torch.isfinite(norm):
                            raise ValueError('Nonfinite gradient')
                        optimizer.step()
                        n = int(valid.sum())
                        native_queries += n
                        updates += 1
                        loss_total += float(loss.detach()) * n
                        count += n
                    losses.append(loss_total / count)
                    print(json.dumps({'method': method, 'seed': seed, 'epoch': epoch + 1,
                                      'training_loss': losses[-1],
                                      'seconds': time.perf_counter() - fit_start}), flush=True)
                train_seconds = time.perf_counter() - fit_start
                torch.save(model.state_dict(), active / 'weights.pt')
                evaluation = evaluate(model, packet['cohorts']['dev'], queries, features,
                                      active / 'dev-predictions.npz')
                row = {'method': method, 'seed': seed, 'status': 'completed', 'epochs': len(losses),
                       'updates': updates, 'training_queries': native_queries, 'training_losses': losses,
                       'training_actor_shapes': dict(actor_work),
                       'initial_tensors_sha256': initial_digest,
                       'parameters': sum(p.numel() for p in model.parameters()),
                       'parameters_with_final_gradient': sum(p.numel() for p in model.parameters()
                                                              if p.grad is not None),
                       'train_wall_seconds': train_seconds, 'evaluation': evaluation,
                       'weights_sha256': sha(active / 'weights.pt'),
                       'predictions_sha256': sha(active / 'dev-predictions.npz')}
                write(active / 'completed.json', row)
                records.append(row)
        validate_plan(args)
        write(Path(args.out) / 'completed.json', {'status': 'completed', 'fits': records,
              'fit_count': len(records), 'plan_sha256': args.plan_sha256,
              'wall_seconds': time.perf_counter() - started, 'external_model_api_calls': 0,
              'scope': 'All frozen final fits; official development only. No test evaluation.'})
    except BaseException as e:
        write(Path(args.out) / 'failed.json', {'status': 'failed', 'completed_fits': records,
              'active_fit': None if active is None else active.name,
              'error_type': type(e).__name__, 'error': str(e),
              'wall_seconds': time.perf_counter() - started})
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['encode', 'freeze', 'train'])
    p.add_argument('--data')
    p.add_argument('--data-sha256')
    p.add_argument('--packet')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='mps')
    p.add_argument('--plan-sha256')
    a = p.parse_args()
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    {'encode': encode, 'freeze': freeze, 'train': train}[a.mode](a)


if __name__ == '__main__':
    main()
