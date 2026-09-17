"""Post-hoc surface-form audit; never substitutes for frozen confirmation.

Question negation was inspected after weak memory-head development results.
This exports a training-only two-bin baseline and four-group diagnostics.
"""
import argparse
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from openjev.research.text_distillation import file_hash, write_new


def has_not(text):
    return bool(re.search(r'\bnot\b', text.casefold()))


def cells(worlds):
    return [(int(q['gold'] == 'true'), int(has_not(q['text'])), int(q['depth']))
            for w in worlds for q in w['questions']]


def evaluate(predictions, rows):
    rows = np.array(rows)
    gold, negation, depths = rows.T
    predictions = np.array(predictions)
    if predictions.shape != gold.shape:
        raise ValueError('Predictions must align with the frozen question order')
    correct = predictions == gold
    groups = {}
    for label in (0, 1):
        for neg in (0, 1):
            mask = (gold == label) & (negation == neg)
            if not mask.any():
                raise ValueError('Need all four label/negation groups')
            groups[f'label={label},not={neg}'] = {'n': int(mask.sum()), 'accuracy': float(correct[mask].mean())}
    return {'accuracy': float(correct.mean()), 'four_group_macro_accuracy': float(np.mean(
        [v['accuracy'] for v in groups.values()])), 'groups': groups,
        'depth_counts': {str(d): {f'label={g},not={n}': int(((depths == d) & (gold == g) & (negation == n)).sum())
                                for g in (0, 1) for n in (0, 1)} for d in sorted(set(depths))}}


def main(args):
    source = Path(__file__).with_name('rule_memory_study.py')
    spec = importlib.util.spec_from_file_location('rule_memory_audit_source', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, worlds = module.load_packet(args.packet)
    rows = {p: cells(w) for p, w in worlds.items()}
    counts = Counter((label, neg) for label, neg, _ in rows['train'])
    # Fit two probabilities on training labels only, with fixed Laplace smoothing.
    p_true = {neg: (counts[1, neg]+1)/(counts[0, neg]+counts[1, neg]+2) for neg in (0, 1)}
    baseline, models = {}, {}
    for part in ('dev_in', 'dev_shift'):
        p = np.array([p_true[neg] for _, neg, _ in rows[part]])
        y = np.array([gold for gold, _, _ in rows[part]])
        baseline[part] = evaluate((p >= .5).astype(int), rows[part])
        baseline[part]['nll'] = float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())
    for family, directory, modes in [('sentence_memory', args.memory, module.PROTOCOL['modes']),
                                     ('joint_encoder', args.crossencoder, ['frozen', 'finetuned'])]:
        for mode in modes:
            fits = []
            for seed in (17, 29, 43):
                path = directory/f'{mode}-{seed}-logits.npz'
                metadata = json.loads((directory/f'{mode}-{seed}.json').read_text())
                if file_hash(path) != metadata['logits_sha256']:
                    raise ValueError('Prediction hash mismatch')
                with np.load(path, allow_pickle=False) as logits:
                    fits.append({'seed': seed, 'logits_sha256': file_hash(path),
                        'parts': {part: evaluate(logits[part].argmax(1), rows[part])
                                  for part in ('dev_in', 'dev_shift')}})
            models[f'{family}:{mode}'] = {'fits': fits, 'means': {part: {
                metric: float(np.mean([f['parts'][part][metric] for f in fits]))
                for metric in ('accuracy', 'four_group_macro_accuracy')}
                for part in ('dev_in', 'dev_shift')}}
    write_new(args.out, {'status': 'post_hoc_development_diagnostic', 'code_sha256': file_hash(__file__),
        'baseline_rule': 'Predict training-majority label in each question contains/not-contains the word not bin',
        'training_p_true': {str(k): v for k, v in p_true.items()}, 'baseline': baseline, 'models': models,
        'limits': ['Negation inspected after memory development results; this baseline was not preregistered.',
                   'Four-group macro accuracy equally weights gold label crossed with question negation.',
                   'This reweighting diagnoses one shortcut, not all shortcuts or reasoning validity.',
                   'No model re-selection or training change; original continuation outcome is retained.',
                   'All figures remain development evidence, not independent confirmation.']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'memory', 'crossencoder', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    main(p.parse_args())
