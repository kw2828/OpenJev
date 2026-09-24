"""Join already-completed original processes; metadata only, no scientific calls."""
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
    spec = importlib.util.spec_from_file_location('_readout_closure_producer', ROOT / 'scripts/run_otto_readout_compute.py')
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    run.require(args.output.is_absolute() and args.output.is_relative_to(ROOT) and not args.output.exists(), 'exclusive contained closure')
    plan, producer, _ = run.process_closure(args.plan, args.producer_receipt, args.producer_terminal)
    audit_plan, audit_receipt, audit_directory = run.process_closure(args.plan, args.audit_receipt, args.audit_terminal)
    run.require(plan == audit_plan and producer['phase'] == 'train' and audit_receipt['phase'] == 'audit', 'same registered study')
    run.require(audit_receipt['producer_receipt'] == run.descriptor(args.producer_receipt)
                and audit_receipt['producer_terminal'] == run.descriptor(args.producer_terminal), 'audit binds exact producer')
    run.require(run.read(args.producer_terminal)['finished_ns'] <= run.read(args.audit_terminal)['started_ns'], 'audit after original producer closure')
    audit = run.read(audit_directory / 'audit.json')
    run.require(audit['agreement'] is True and audit['counts'] == audit_receipt['audit_counts']
                and audit['counts']['array_decodes'] == audit_receipt['array_decodes'] == 22
                and audit['counts']['checkpoint_decodes'] == audit_receipt['checkpoint_decodes'] == 9
                and audit['counts']['views_completed'] == audit_receipt['views_completed'] == 9
                and audit['counts']['journal_updates_checked'] == 3078
                and audit['counts']['fits_checked'] == 6
                and audit['counts']['episode_exposures_checked'] == 18468
                and all(audit['counts'][k] == 0 for k in ('model_calls', 'optimizer_calls', 'teacher_calls', 'native_calls', 'test_array_decodes')),
                'complete independent saved-output audit')
    result = audit['comparisons']
    run.require(result['total_cells'] == len(result['cells']) == 6
                and result['passed_cells'] == sum(x['passed'] for x in result['cells']) and result['efficacy_passed']
                == (result['passed_cells'] == 6) == all(x['passed'] for x in result['cells']), 'saved fixed efficacy identity')
    comparability = result['compute_comparability']
    run.require(len(comparability['pairs']) == 3 and comparability['passed'] == all(x['passed'] for x in comparability['pairs'])
                and result['overall_passed'] == (result['efficacy_passed'] and comparability['passed']),
                'separate efficacy and actual timing comparability')
    run.require(not any(k in sys.modules for k in ('numpy', 'torch', 'tensorflow', 'jax', 'mlx')), 'metadata-only closure')
    status = 'COMPARABLE_DEV_PASS' if result['overall_passed'] else 'DEV_FAIL'
    run.write(args.output, {'version': 'otto-readout-compute-closure-v1', 'status': status,
        **{k: run.descriptor(getattr(args, k)) for k in ('plan', 'producer_receipt', 'producer_terminal', 'audit_receipt', 'audit_terminal')},
        'audit': run.descriptor(audit_directory / 'audit.json'), 'technical_complete': True,
        'diagnostic': result, 'counts': audit['counts'], 'dev_reused': False, 'fresh_dev': True, 'held_out_from_training': True, 'development_only': True,
        'test_admitted': False, 'confirmation_admitted': False, 'held_out_evidence': False,
        'prior_studies_remain_closed': True, 'closure_numerical_array_decodes': 0,
        'closure_source': run.descriptor(Path(__file__).resolve())})
    print(status, run.descriptor(args.output), flush=True)


if __name__ == '__main__':
    main()
