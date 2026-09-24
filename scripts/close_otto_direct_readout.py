"""Join the original completed direct-solve worker and saved-output audit."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'producer-receipt', 'producer-terminal', 'audit-receipt', 'audit-terminal', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('_direct_close', ROOT / 'scripts/run_otto_direct_readout.py')
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    run.require(args.output.is_absolute() and args.output.is_relative_to(ROOT) and not args.output.exists(),
                'exclusive contained closure')
    plan, producer, _ = run.process_closure(args.plan, args.producer_receipt, args.producer_terminal)
    other, audit, directory = run.process_closure(args.plan, args.audit_receipt, args.audit_terminal)
    run.require(plan == other and producer['phase'] == 'train' and audit['phase'] == 'audit'
                and audit['producer_receipt'] == run.descriptor(args.producer_receipt)
                and audit['producer_terminal'] == run.descriptor(args.producer_terminal)
                and run.read(args.producer_terminal)['finished_ns'] <= run.read(args.audit_terminal)['started_ns'],
                'audit follows and binds the original producer')
    result = run.read(directory / 'audit.json')
    run.require(result['agreement'] is True and result['counts'] == audit['audit_counts'], 'saved audit agrees')
    counts = result['counts']
    run.require(all(counts[k] == 0 for k in ('model_calls', 'solver_calls', 'teacher_calls', 'native_calls', 'test_array_decodes')),
                'audit has no new numerical models or fits')
    decision = result['comparisons']
    run.require(len(decision['cells']) == decision['total_cells'] == 6
                and decision['passed_cells'] == sum(c['passed'] for c in decision['cells']), 'all paired cells retained')
    passed = all(c['passed'] for c in decision['cells'])
    run.require(not any(n in sys.modules for n in ('numpy', 'torch', 'tensorflow', 'jax', 'mlx')), 'metadata-only closure')
    run.write(args.output, {'version': 'otto-direct-readout-closure-v1',
        'status': 'REUSED_DEV_PASS' if passed else 'REUSED_DEV_FAIL', 'technical_complete': True,
        **{k: run.descriptor(getattr(args, k)) for k in ('plan', 'producer_receipt', 'producer_terminal', 'audit_receipt', 'audit_terminal')},
        'audit': run.descriptor(directory / 'audit.json'), 'counts': counts, 'diagnostic': decision,
        'dev_reused': True, 'fresh_dev': False, 'held_out_from_training': True,
        'held_out_evidence': False, 'test_admitted': False, 'confirmation_admitted': False,
        'prior_studies_remain_closed': True, 'closure_numerical_array_decodes': 0,
        'closure_source': run.descriptor(Path(__file__).resolve())})
    print(run.descriptor(args.output), flush=True)


if __name__ == '__main__':
    main()
