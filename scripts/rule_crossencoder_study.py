"""Full-context representation diagnostic after the sentence-memory screen.

Compare a frozen full-context encoder with the same encoder fine-tuned on gold
labels. This is an adaptive development follow-up, not an independent test.
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from openjev.research.text_distillation import digest, file_hash, write_new
from openjev.research.text_student import ENCODER_ID, ENCODER_REVISION

SOURCE = Path(__file__).with_name('rule_memory_study.py')
SPEC = importlib.util.spec_from_file_location('rule_memory_runner', SOURCE)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
PROTOCOL = {'version': 'rule-crossencoder-v1', 'seeds': [17, 29, 43],
    'encoder': ENCODER_ID, 'revision': ENCODER_REVISION, 'max_tokens': 512,
    'warmup_head_epochs': 30, 'head_batch': 256, 'head_lr': .001,
    'finetune_epochs': 3, 'batch_size': 32, 'finetune_lr': .00002,
    'weight_decay': .01, 'clip_gradient': 1., 'cpu_threads': 4,
    'parts': list(BASE.PARTS), 'candidate_order': ['false', 'true'],
    'selection': 'Final epoch, all three seeds, both development splits; no official test',
    'comparison': 'Gold-supervised representation diagnostic; no novel-method or cost-matched claim',
    'warm_start': 'Train scalar binary head on frozen joint context/assertion features, then fine-tune all weights',
    'pooling': 'Attention-mask mean of last hidden states; shared binary linear classifier',
    'teacher': 'No Astra or other external teacher requests',
}


def signature():
    return {'driver': file_hash(__file__), 'audited_data_and_metrics': file_hash(SOURCE)}


def checked(args):
    plan = json.loads(args.plan.read_text())
    manifest, worlds = BASE.load_packet(args.packet)
    if plan['code'] != signature() or plan['protocol'] != PROTOCOL or plan['packet_sha256'] != digest(manifest):
        raise ValueError('Frozen protocol, data or code changed')
    torch.set_num_threads(PROTOCOL['cpu_threads'])
    return plan, worlds


def freeze(args):
    manifest, _ = BASE.load_packet(args.packet)
    write_new(args.out, {'protocol': PROTOCOL, 'code': signature(), 'packet_sha256': digest(manifest),
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'context': 'Planned after observing weak early development fits in rule-memory-v1'})


def tokenize_worlds(worlds, tokenizer):
    # Text allowlist is separate from supervision and reporting metadata.
    contexts = [w['context'] for w in worlds for _ in w['questions']]
    assertions = [q['text'] for w in worlds for q in w['questions']]
    inputs = tokenizer(contexts, assertions, padding=True, truncation=False, return_tensors='pt')
    if inputs['input_ids'].shape[1] > PROTOCOL['max_tokens']:
        raise ValueError('Full English input exceeds context limit; no truncation')
    return {'inputs': inputs, 'gold': torch.tensor([int(q['gold'] == 'true') for w in worlds for q in w['questions']]),
            'depth': torch.tensor([q['depth'] for w in worlds for q in w['questions']]),
            'world': np.array([i for i, w in enumerate(worlds) for _ in w['questions']])}


def batch(data, indices, device):
    length = int(data['inputs']['attention_mask'][indices].sum(1).max())
    return {k: v[indices, :length].to(device) for k, v in data['inputs'].items()}


def pooled(encoder, inputs):
    hidden = encoder(**inputs).last_hidden_state
    mask = inputs['attention_mask'].unsqueeze(-1)
    return (hidden*mask).sum(1)/mask.sum(1)


@torch.no_grad()
def encode(encoder, data, device):
    encoder.eval()
    return torch.cat([pooled(encoder, batch(data, index, device)).cpu() for index in
                      torch.arange(len(data['gold'])).split(PROTOCOL['batch_size'])])


def score(logits, data):
    return BASE.metrics(logits, data['gold'], data['depth'])


def run(args):
    plan, worlds = checked(args)
    if args.out.exists():
        raise ValueError('New run directory required; no overwrite or silent retry')
    args.out.mkdir(parents=True)
    write_new(args.out/'started.json', {'plan_sha256': digest(plan), 'device': args.device})
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_ID, revision=ENCODER_REVISION, local_files_only=True)
    data = {p: tokenize_worlds(w, tokenizer) for p, w in worlds.items()}
    encoder = AutoModel.from_pretrained(ENCODER_ID, revision=ENCODER_REVISION, local_files_only=True,
                                      attn_implementation='eager').to(args.device)
    start = time.perf_counter()
    features = {p: encode(encoder, d, args.device) for p, d in data.items()}
    feature_seconds = time.perf_counter()-start
    for p, x in features.items():
        np.save(args.out/f'{p}-frozen.npy', x.numpy(), allow_pickle=False)
    write_new(args.out/'features.json', {'plan_sha256': digest(plan), 'forward_seconds': feature_seconds,
        'arrays': {p: file_hash(args.out/f'{p}-frozen.npy') for p in features},
        'max_pair_tokens': {p: d['inputs']['input_ids'].shape[1] for p, d in data.items()}})
    del encoder
    for seed in PROTOCOL['seeds']:
        torch.manual_seed(seed)
        head = nn.Linear(384, 2)
        opt = torch.optim.AdamW(head.parameters(), lr=PROTOCOL['head_lr'], weight_decay=PROTOCOL['weight_decay'])
        rng = torch.Generator().manual_seed(seed)
        start = time.perf_counter()
        history = []
        for epoch in range(PROTOCOL['warmup_head_epochs']):
            order = torch.randperm(len(data['train']['gold']), generator=rng)
            total, steps = 0., 0
            for index in order.split(PROTOCOL['head_batch']):
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(head(features['train'][index]), data['train']['gold'][index])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), PROTOCOL['clip_gradient'], error_if_nonfinite=True)
                opt.step()
                total += float(loss.detach())*len(index)
                steps += 1
            history.append({'epoch': epoch+1, 'updates': steps, 'loss': total/len(order)})
        warmup_seconds = time.perf_counter()-start
        head.eval()
        with torch.no_grad():
            logits = {p: head(features[p]) for p in ('dev_in', 'dev_shift')}
        torch.save(head.state_dict(), args.out/f'frozen-{seed}.pt')
        result = {'seed': seed, 'mode': 'frozen_joint', 'plan_sha256': digest(plan), 'history': history,
            'head_training_seconds': warmup_seconds, 'active_parameters': 770,
            'checkpoint_sha256': file_hash(args.out/f'frozen-{seed}.pt'),
            'metrics': {p: score(v, data[p]) for p, v in logits.items()}}
        np.savez(args.out/f'frozen-{seed}-logits.npz', **{p: v.numpy() for p, v in logits.items()})
        result['logits_sha256'] = file_hash(args.out/f'frozen-{seed}-logits.npz')
        write_new(args.out/f'frozen-{seed}.json', result)
        print(json.dumps({k: result[k] for k in ('seed', 'mode', 'metrics')}), flush=True)

        torch.manual_seed(seed)
        encoder = AutoModel.from_pretrained(ENCODER_ID, revision=ENCODER_REVISION, local_files_only=True,
                                          attn_implementation='eager').to(args.device).train()
        head.to(args.device).train()
        parameters = list(encoder.parameters())+list(head.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=PROTOCOL['finetune_lr'], weight_decay=PROTOCOL['weight_decay'])
        rng = torch.Generator().manual_seed(seed)
        start, history = time.perf_counter(), []
        for epoch in range(PROTOCOL['finetune_epochs']):
            order = torch.randperm(len(data['train']['gold']), generator=rng)
            total, steps = 0., 0
            for index in order.split(PROTOCOL['batch_size']):
                optimizer.zero_grad(set_to_none=True)
                logits = head(pooled(encoder, batch(data['train'], index, args.device)))
                loss = F.cross_entropy(logits, data['train']['gold'][index].to(args.device))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite fine-tuning loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, PROTOCOL['clip_gradient'], error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach().cpu())*len(index)
                steps += 1
                if steps % 200 == 0:
                    print(json.dumps({'seed': seed, 'epoch': epoch+1, 'step': steps,
                                      'loss': total/min(steps*PROTOCOL['batch_size'], len(order))}), flush=True)
            history.append({'epoch': epoch+1, 'updates': steps, 'loss': total/len(order)})
        seconds = time.perf_counter()-start
        encoder.eval()
        head.to('cpu').eval()
        with torch.no_grad():
            logits = {p: head(encode(encoder, data[p], args.device)) for p in ('dev_in', 'dev_shift')}
        path = args.out/f'finetuned-{seed}.pt'
        torch.save({'encoder': {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
                    'head': head.state_dict()}, path)
        np.savez(args.out/f'finetuned-{seed}-logits.npz', **{p: v.numpy() for p, v in logits.items()})
        result = {'seed': seed, 'mode': 'finetuned_joint', 'plan_sha256': digest(plan), 'history': history,
            'additional_training_seconds': seconds, 'shared_warmup_seconds': warmup_seconds,
            'active_parameters': sum(p.numel() for p in parameters), 'checkpoint_sha256': file_hash(path),
            'logits_sha256': file_hash(args.out/f'finetuned-{seed}-logits.npz'),
            'metrics': {p: score(v, data[p]) for p, v in logits.items()}}
        write_new(args.out/f'finetuned-{seed}.json', result)
        print(json.dumps({k: result[k] for k in ('seed', 'mode', 'metrics', 'additional_training_seconds')}), flush=True)
        del encoder, optimizer, parameters
    write_new(args.out/'completed.json', {'plan_sha256': digest(plan), 'status': 'completed',
                                         'frozen_head_fits': 3, 'finetuned_fits': 3})


def report(args):
    plan, worlds = checked(args)
    completed = json.loads((args.runs/'completed.json').read_text())
    if completed != {'plan_sha256': digest(plan), 'status': 'completed', 'frozen_head_fits': 3, 'finetuned_fits': 3}:
        raise ValueError('Incomplete fit family')
    methods = {}
    for mode in ('frozen', 'finetuned'):
        fits = []
        for seed in PROTOCOL['seeds']:
            r = json.loads((args.runs/f'{mode}-{seed}.json').read_text())
            if (r['plan_sha256'] != digest(plan) or r['seed'] != seed or r['mode'] != mode+'_joint' or
                    r['checkpoint_sha256'] != file_hash(args.runs/f'{mode}-{seed}.pt') or
                    r['logits_sha256'] != file_hash(args.runs/f'{mode}-{seed}-logits.npz')):
                raise ValueError('Fit identity mismatch')
            with np.load(args.runs/f'{mode}-{seed}-logits.npz', allow_pickle=False) as logits:
                for part in ('dev_in', 'dev_shift'):
                    meta = {'gold': torch.tensor([int(q['gold'] == 'true') for w in worlds[part] for q in w['questions']]),
                            'depth': torch.tensor([q['depth'] for w in worlds[part] for q in w['questions']])}
                    if score(torch.from_numpy(logits[part]), meta) != r['metrics'][part]:
                        raise ValueError('Metrics do not replay')
            fits.append(r)
        methods[mode] = {'fits': fits, 'means': {part: {metric: float(np.mean([r['metrics'][part][metric] for r in fits]))
            for metric in ('accuracy', 'nll', 'multiclass_brier')} for part in ('dev_in', 'dev_shift')}}
    write_new(args.out, {'status': 'development_only', 'plan': plan, 'methods': methods,
        'features': json.loads((args.runs/'features.json').read_text()),
        'confirmation_executed': False, 'teacher_requests': 0,
        'limits': ['Adaptive follow-up using the same development worlds; no independent confirmation.',
                   'Fine-tuning changes representation, trainable parameter count and training compute.',
                   'No novel architecture, Astra training, calibrated-probability or arbitrary-candidate claim.']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'run', 'report'])
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--plan', type=Path)
    p.add_argument('--runs', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    args = p.parse_args()
    {'freeze': freeze, 'run': run, 'report': report}[args.stage](args)
