"""Explicit adapter around the unchanged frozen trainer; only head dynamics change."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np

from openjev.research.corrective_associative import VARIANTS, CorrectiveAssociativeHead
from openjev.research.text_distillation import digest, file_hash, write_new

BASE_PATH = Path(__file__).with_name('learned_associative_study.py')
spec = importlib.util.spec_from_file_location('frozen_learned_associative_runner', BASE_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

PROTOCOL = copy.deepcopy(runner.PROTOCOL)
PROTOCOL.update(version='corrective-associative-v1', modes=list(VARIANTS), correction_rate=.25,
                candidates=['inhibitory_2', 'residual_2', 'residual_3'],
                controls=['metric', 'feedforward', 'attractive_2'],
                recurrence_steps={'attractive_2': 2, 'inhibitory_2': 2, 'residual_2': 2, 'residual_3': 3})
PROTOCOL['continuation']['nll_no_worse_than_control'] = True
PROTOCOL['continuation']['minimum_accuracy_gain_over_best_control'] = .005


def signature():
    import openjev.research.corrective_associative as architecture
    import openjev.research.learned_associative as base_architecture
    return {'study': file_hash(__file__), 'trainer': file_hash(BASE_PATH),
            'architecture': file_hash(architecture.__file__), 'base_architecture': file_hash(base_architecture.__file__)}


def write_execution_record(path, value):
    # v1's optimizer is reusable, but its receipt has a fixed fit count. Derive
    # this study's count and verify every output before writing the receipt.
    if path.name == 'completed.json' and value.get('status') == 'completed':
        for mode in PROTOCOL['modes']:
            for seed in PROTOCOL['seeds']:
                if not (path.parent/f'{mode}-{seed}.json').is_file():
                    raise ValueError('Cannot complete a study with missing fits')
        value = {**value, 'fits': len(PROTOCOL['modes'])*len(PROTOCOL['seeds'])}
    write_new(path, value)


def report(args):
    plan, _ = runner.checked_plan(args)
    completed = json.loads((args.runs/'completed.json').read_text())
    if (completed['plan_sha256'] != digest(plan) or completed['status'] != 'completed' or
            completed['fits'] != len(PROTOCOL['modes'])*len(PROTOCOL['seeds'])):
        raise ValueError('All fits must finish before selection')
    methods = {}
    for mode in PROTOCOL['modes']:
        fits = []
        for seed in PROTOCOL['seeds']:
            r = json.loads((args.runs/f'{mode}-{seed}.json').read_text())
            if (r['mode'] != mode or r['seed'] != seed or r['plan_sha256'] != digest(plan) or
                    r['checkpoint_sha256'] != file_hash(args.runs/f'{mode}-{seed}.pt')):
                raise ValueError('Fit identity or checkpoint mismatch')
            fits.append(r)
        methods[mode] = {'fits': fits, 'mean_metrics': {k: float(np.mean([r['metrics'][k] for r in fits]))
                                                       for k in fits[0]['metrics']}}
    control = max(PROTOCOL['controls'], key=lambda m: methods[m]['mean_metrics']['id_accuracy'])
    candidate = max(PROTOCOL['candidates'], key=lambda m: methods[m]['mean_metrics']['id_accuracy'])
    delta = {k: methods[candidate]['mean_metrics'][k]-methods[control]['mean_metrics'][k]
             for k in ('id_accuracy', 'balanced_utility', 'id_nll')}
    wins = sum(a['metrics']['id_accuracy'] > b['metrics']['id_accuracy'] for a, b in
               zip(methods[candidate]['fits'], methods[control]['fits'], strict=True))
    ratio = (methods[candidate]['fits'][0]['cost']['matrix_multiply_accumulates_per_query'] /
             methods[control]['fits'][0]['cost']['matrix_multiply_accumulates_per_query'])
    gate = PROTOCOL['continuation']
    passed = (delta['id_accuracy'] >= gate['minimum_accuracy_gain_over_best_control'] and
              delta['balanced_utility'] >= gate['minimum_utility_gain_over_best_control'] and
              delta['id_nll'] <= 0 and wins >= gate['minimum_paired_seed_wins'] and
              ratio <= gate['max_matrix_cost_multiple'])
    write_new(args.out, {'status': 'development_only', 'plan': plan, 'methods': methods,
                        'verified_fits': sum(len(v['fits']) for v in methods.values()),
                        'selected_candidate': candidate, 'strongest_control': control, 'deltas': delta,
                        'paired_seed_wins': wins, 'matrix_cost_multiple': ratio,
                        'continuation': 'eligible_for_confirmation' if passed else 'stop_before_confirmation',
                        'confirmation_executed': False,
                        'limits': ['Repeated architecture selection on the same development set.',
                                   'Only 50 OOS development examples; rejection settings reused.',
                                   'Frozen transformer encoder; finite updates trained by backpropagation.',
                                   'Not an implementation of predictive-coding learning or implicit equilibrium differentiation.']})


def main():
    # Reuse the exact optimization and data access functions without editing v1.
    runner.PROTOCOL = PROTOCOL
    runner.LearnedAssociativeHead = CorrectiveAssociativeHead
    runner.signature = signature
    runner.write_new = write_execution_record
    runner.report = report
    runner.main()


if __name__ == '__main__':
    main()
