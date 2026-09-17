"""Describe saved shifted-length errors; no model calls or threshold validation."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory):
    summary_path = directory / 'summary.json'
    summary = json.loads(summary_path.read_text())
    completed = json.loads((directory / 'completed.json').read_text())
    if (summary['protocol']['version'] != 'memory-optimization-v1'
            or completed['status'] != 'completed'
            or completed['summary_sha256'] != sha(summary_path)):
        raise ValueError('Expected the completed frozen optimization diagnostic')
    rows = []
    for fit in summary['evaluations']:
        if fit['path'] != 'recursive' or fit['loss'] != 'ce2':
            continue
        for result in fit['results']:
            if result['size'] not in (17, 23) or result['condition'] != 'intact':
                continue
            for example in result['examples']:
                p = example['probabilities2']
                if (len(p) != 2 or not all(math.isfinite(x) and 0 <= x <= 1 for x in p)
                        or abs(sum(p) - 1) > 1e-6):
                    raise ValueError('Invalid conditional probabilities')
                predicted = max(range(2), key=lambda i: p[i])
                if predicted != example['predicted_candidate2']:
                    raise ValueError('Prediction does not match probabilities')
                rows.append({'fit_seed': fit['seed'], 'size': result['size'],
                             'context_seed': example['seed'], 'target': example['target_turn'],
                             'predicted': predicted, 'correct': predicted == example['target_turn'],
                             'confidence': max(p),
                             'entropy_bits': -sum(x * math.log2(x) for x in p if x > 0)})
    identities = {(r['fit_seed'], r['size'], r['context_seed']) for r in rows}
    expected = {(seed, size, context['seed']) for seed in (101, 113, 127)
                for size in (17, 23) for context in summary['training_data']['contexts']}
    if len(rows) != 24 or identities != expected:
        raise ValueError('Expected all 24 shifted-length decisions')
    wrong = [r for r in rows if not r['correct']]
    correct = [r for r in rows if r['correct']]
    if not wrong or not correct:
        raise ValueError('This descriptive audit requires both correct decisions and errors')
    low_confidence = [r for r in rows if r['confidence'] < .99]
    minimum_error_entropy = min(r['entropy_bits'] for r in wrong)
    all_error_gate = [r for r in rows if r['entropy_bits'] >= minimum_error_entropy]
    return {
        'source_summary_sha256': sha(summary_path), 'script_sha256': sha(Path(__file__)),
        'scope': 'Post-hoc descriptive audit on four reused contexts, three fits, two lengths; no model calls',
        'decisions': rows, 'decision_count': len(rows), 'errors': len(wrong),
        'wrong_confidence_min': min(r['confidence'] for r in wrong),
        'wrong_confidence_mean': sum(r['confidence'] for r in wrong) / len(wrong),
        'wrong_confidence_max': max(r['confidence'] for r in wrong),
        'wrong_entropy_mean_bits': sum(r['entropy_bits'] for r in wrong) / len(wrong),
        'correct_entropy_mean_bits': sum(r['entropy_bits'] for r in correct) / len(correct),
        'illustrative_99_percent_confidence_gate': {
            'flagged': len(low_confidence), 'errors_flagged': sum(not r['correct'] for r in low_confidence)},
        'posthoc_threshold_fitted_to_catch_all_observed_errors': {
            'minimum_entropy_bits': minimum_error_entropy, 'flagged': len(all_error_gate),
            'correct_flagged': sum(r['correct'] for r in all_error_gate),
            'independent_validation': False},
        'calibration_claim': False, 'general_entropy_gate_claim': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'decisions'}, indent=2))


if __name__ == '__main__':
    main()
