"""Frozen, small-data Astra teacher/student experiment. See docs/text-distillation.md."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .text_student import (
    ENCODER_ID,
    ENCODER_REVISION,
    MAX_LENGTH,
    CandidateStudent,
    categorical_metrics,
    ordered_candidates,
    pair_text,
)

PROTOCOL = {
    'version': 'astra-text-distillation-v1',
    'student': ENCODER_ID, 'student_revision': ENCODER_REVISION,
    'teacher': 'gpt-6-astra', 'reasoning': 'low', 'max_output_tokens': 1024,
    'selection_seed': 76219, 'fit_seeds': [17, 29, 43],
    'arms': ['untrained', 'gold', 'astra'], 'epochs': 3, 'learning_rate': 2e-5,
    'weight_decay': 0.01, 'gradient_clip': 1.0, 'max_pair_tokens': MAX_LENGTH,
    'boolq_train_per_class': 32, 'boolq_eval_per_class': 42,
    'clinc_train_per_domain': 6, 'clinc_train_oos': 4, 'clinc_eval_per_class': 8,
    'teacher_cost_ceiling_usd': 20.0,
    'sources': {
        'boolq': {'repo': 'google/boolq', 'revision': '35b264d03638db9f4ce671b711558bf7ff0f80d5',
                  'train': 'train', 'evaluation': 'validation', 'license': 'CC-BY-SA-3.0'},
        'clinc': {'repo': 'clinc/oos-eval', 'revision': '828f8093932c8fe6ca7936c3d2e52903b1c523de',
                  'train': 'train+oos_train', 'evaluation': 'test+oos_test', 'license': 'CC-BY-3.0'},
    },
    'scope': 'Balanced small-data pilot; CLINC is 10-domain-plus-OOS adaptation, not 150-intent benchmark',
    'selection': 'Filter >512 tokens and duplicates before seeded class-stratified sampling; train priority',
    'fit': 'All encoder weights plus shared scalar head; CE over candidates; one question per update',
    'calibration': 'none; candidate-score softmax, not teacher/token probabilities',
    'continuation': {'each_task_astra_minus_untrained_accuracy_min': 0.10,
                     'each_task_astra_minus_gold_accuracy_min': -0.05,
                     'interpretation': 'Descriptive screening rule, not significance or novelty evidence'},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def source_paths():
    here = Path(__file__).parent
    return [here / name for name in ('text_student.py', 'text_distillation.py', 'text_teacher.py', 'text_report.py')]


def context_key(record):
    return digest([record['question'].strip().casefold(), record['context'].strip().casefold()])


def make_boolq(rows, split):
    return [{'id': f'boolq:{split}:{i}', 'task': 'boolq', 'source_split': split,
             'question': r['question'], 'context': r['passage'],
             'candidates': [{'id': 'yes', 'description': 'Yes'}, {'id': 'no', 'description': 'No'}],
             'gold': 'yes' if r['answer'] else 'no'} for i, r in enumerate(rows)]


def make_clinc(rows, domains, split):
    intent_domain = {intent: domain for domain, intents in domains.items() for intent in intents}
    candidates = [{'id': domain, 'description': domain.replace('_', ' ') + ': ' +
                   ', '.join(intent.replace('_', ' ') for intent in sorted(intents))}
                  for domain, intents in sorted(domains.items())]
    candidates.append({'id': 'oos', 'description': 'Out of scope: no listed supported intent matches this request'})
    return [{'id': f'clinc:{split}:{i}', 'task': 'clinc_domains', 'source_split': split,
             'context': text, 'question': 'Which domain contains the supported intent requested by the user? '
             'Select out of scope if none of the listed intents matches.', 'candidates': candidates,
             'gold': 'oos' if intent == 'oos' else intent_domain[intent]} for i, (text, intent) in enumerate(rows)]


def select_rows(rows, per_class, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row['gold']].append(row)
    selected = []
    rng = random.Random(seed)
    if set(groups) != set(per_class):
        raise ValueError('Unexpected benchmark classes')
    for key in sorted(groups):
        pool = sorted(groups[key], key=lambda r: r['id'])
        if len(pool) < per_class[key]:
            raise ValueError(f'Insufficient eligible rows for {key}')
        selected.extend(rng.sample(pool, per_class[key]))
    rng.shuffle(selected)
    return selected


def prepare(source_dir, out):
    if out.exists():
        raise ValueError('Output exists; refuse to change a frozen packet')
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_ID, revision=ENCODER_REVISION, local_files_only=True)
    clinc = json.loads((source_dir / 'clinc.json').read_text())
    domains = json.loads((source_dir / 'domains.json').read_text())
    boolq = {s: [json.loads(line) for line in (source_dir / f'boolq-{s}.jsonl').read_text().splitlines()]
             for s in ('train', 'validation')}
    pools = {
        'boolq_train': make_boolq(boolq['train'], 'train'),
        'boolq_eval': make_boolq(boolq['validation'], 'validation'),
        'clinc_train': make_clinc(clinc['train'] + clinc['oos_train'], domains, 'train'),
        'clinc_eval': make_clinc(clinc['test'] + clinc['oos_test'], domains, 'test'),
    }
    counts = {'boolq_train': dict.fromkeys(('yes', 'no'), PROTOCOL['boolq_train_per_class']),
              'boolq_eval': dict.fromkeys(('yes', 'no'), PROTOCOL['boolq_eval_per_class']),
              'clinc_train': {**dict.fromkeys(domains, PROTOCOL['clinc_train_per_domain']),
                              'oos': PROTOCOL['clinc_train_oos']},
              'clinc_eval': dict.fromkeys([*domains, 'oos'], PROTOCOL['clinc_eval_per_class'])}
    selections, audit, seen = {}, {}, set()
    # Train pools come first within each task; exact train/eval duplication is excluded.
    for name, rows in pools.items():
        eligible, reasons = [], Counter()
        for row in rows:
            key = context_key(row)
            if key in seen:
                reasons['duplicate'] += 1
                continue
            seen.add(key)
            cs = ordered_candidates(row)
            tokenized = tokenizer([pair_text(row)]*len(cs), [c['description'] for c in cs],
                                  truncation=False, verbose=False)
            if max(map(len, tokenized['input_ids'])) > MAX_LENGTH:
                reasons['overlong'] += 1
                continue
            eligible.append(row)
        selections[name] = select_rows(eligible, counts[name], PROTOCOL['selection_seed'])
        audit[name] = {'source_rows': len(rows), 'eligible_rows': len(eligible),
                       'excluded': dict(reasons), 'selected': len(selections[name]),
                       'selected_classes': dict(Counter(r['gold'] for r in selections[name]))}
    packet = {'protocol': PROTOCOL, 'selection_audit': audit,
              'train': selections['boolq_train'] + selections['clinc_train'],
              'evaluation': selections['boolq_eval'] + selections['clinc_eval']}
    hashes = {p.name: file_hash(p) for p in source_dir.iterdir()
              if p.name in ('clinc.json', 'domains.json', 'boolq-train.jsonl', 'boolq-validation.jsonl')}
    manifest = {'packet_sha256': digest(packet), 'source_sha256': hashes,
                'code_sha256': {p.name: file_hash(p) for p in source_paths()},
                'prepared_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'environment': {m: importlib.metadata.version(m) for m in
                                ('torch', 'transformers', 'numpy', 'huggingface-hub')},
                'platform': platform.platform(),
                'published_selection': {s: [{'id': r['id'], 'sha256': digest(r)} for r in packet[s]]
                                        for s in ('train', 'evaluation')}}
    write_new(out / 'packet.json', packet)
    write_new(out / 'manifest.json', manifest)
    print(json.dumps({'packet_sha256': manifest['packet_sha256'], 'selection': audit}, indent=2))


def load_packet(directory):
    packet = json.loads((directory / 'packet.json').read_text())
    manifest = json.loads((directory / 'manifest.json').read_text())
    if digest(packet) != manifest['packet_sha256'] or packet['protocol'] != PROTOCOL:
        raise ValueError('Frozen packet/protocol mismatch')
    if manifest['code_sha256'] != {p.name: file_hash(p) for p in source_paths()}:
        raise ValueError('Execution code changed after freeze; do not run this packet')
    for split in ('train', 'evaluation'):
        if len({r['id'] for r in packet[split]}) != len(packet[split]):
            raise ValueError('Duplicate example IDs')
    if {context_key(r) for r in packet['train']} & {context_key(r) for r in packet['evaluation']}:
        raise ValueError('Train/evaluation context overlap')
    return packet, manifest


def teacher_labels(teacher_dir, packet, manifest):
    from .text_teacher import build_payload, checked_result, reserve
    summary = json.loads((teacher_dir / 'summary.json').read_text())
    if (summary['status'] != 'completed' or summary['producer'] != 'official_responses'
            or summary['packet_sha256'] != manifest['packet_sha256']):
        raise ValueError('Astra supervision requires a completed matching official run')
    labels = {}
    for row in packet['train']:
        receipt = json.loads((teacher_dir / 'receipts' / f"{digest(row['id'])}.json").read_text())
        response_file = teacher_dir / 'responses' / f"{digest(row['id'])}.json"
        if (receipt['status'] != 'completed' or receipt['model'] != PROTOCOL['teacher']
                or receipt['payload_sha256'] != digest(build_payload(row))
                or receipt['response_sha256'] != file_hash(response_file)
                or receipt['choice'] not in {c['id'] for c in row['candidates']}):
            raise ValueError('Missing or inconsistent teacher receipt')
        actual_choice, _, _ = checked_result(json.loads(response_file.read_text()),
                                            {c['id'] for c in row['candidates']}, reserve(build_payload(row)))
        if actual_choice != receipt['choice']:
            raise ValueError('Teacher label differs from preserved response')
        labels[row['id']] = receipt['choice']
    return labels


def fit_student(model, rows, labels, seed):
    optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL['learning_rate'],
                                 weight_decay=PROTOCOL['weight_decay'])
    rng, history = random.Random(seed), []
    start = time.perf_counter()
    for epoch in range(PROTOCOL['epochs']):
        order = list(rows)
        rng.shuffle(order)
        model.train()
        losses = []
        for row in order:
            inputs, candidates = model.tokenize(row)
            target = [c['id'] for c in candidates].index(labels[row['id']])
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0),
                                                     torch.tensor([target], device=model.device_name))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({'epoch': epoch + 1, 'updates': len(losses), 'mean_loss': float(np.mean(losses))})
        print(json.dumps({'seed': seed, **history[-1]}), flush=True)
    if model.device_name == 'mps':
        torch.mps.synchronize()
    return history, time.perf_counter() - start


def train(packet_dir, out, arm, teacher_dir, device):
    packet, manifest = load_packet(packet_dir)
    if out.exists():
        raise ValueError('Output exists; training cannot overwrite an earlier fit')
    labels = (teacher_labels(teacher_dir, packet, manifest) if arm == 'astra'
              else {r['id']: r['gold'] for r in packet['train']})
    out.mkdir(parents=True)
    write_new(out / 'started.json', {'arm': arm, 'packet_sha256': manifest['packet_sha256'], 'device': device})
    all_results = []
    for seed in PROTOCOL['fit_seeds']:
        model = CandidateStudent(seed=seed, device=device)
        fit_start = time.perf_counter()
        if arm == 'untrained':
            history, train_seconds = [], 0.
        else:
            history, train_seconds = fit_student(model, packet['train'], labels, seed)
            model.save(out / f'seed-{seed}')
        # Warm up with a training item. No test labels enter optimization or selection.
        model.decide_record(packet['train'][0])
        predictions = [{'id': row['id'], 'task': row['task'], 'gold': row['gold'],
                        **model.decide_record(row)} for row in packet['evaluation']]
        metrics = {task: categorical_metrics([r for r in predictions if r['task'] == task])
                   for task in sorted({r['task'] for r in predictions})}
        result = {'arm': arm, 'seed': seed, 'packet_sha256': manifest['packet_sha256'],
                  'history': history, 'training_seconds': train_seconds,
                  'fit_and_evaluation_seconds': time.perf_counter() - fit_start,
                  'metrics': metrics, 'predictions': predictions}
        write_new(out / f'result-{seed}.json', result)
        all_results.append(result)
        del model
        if device == 'mps':
            torch.mps.empty_cache()
        print(json.dumps({'arm': arm, 'seed': seed, 'metrics': metrics}), flush=True)
    write_new(out / 'completed.json', {'status': 'completed', 'arm': arm,
                                      'packet_sha256': manifest['packet_sha256'],
                                      'fit_seeds': [r['seed'] for r in all_results]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--sources', type=Path, required=True)
    prep.add_argument('--out', type=Path, required=True)
    fit = sub.add_parser('train')
    fit.add_argument('--packet', type=Path, required=True)
    fit.add_argument('--out', type=Path, required=True)
    fit.add_argument('--arm', choices=PROTOCOL['arms'], required=True)
    fit.add_argument('--teacher', type=Path)
    fit.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.sources, args.out)
    else:
        if args.arm == 'astra' and args.teacher is None:
            parser.error('--teacher is required for Astra supervision')
        train(args.packet, args.out, args.arm, args.teacher, args.device)


if __name__ == '__main__':
    main()
