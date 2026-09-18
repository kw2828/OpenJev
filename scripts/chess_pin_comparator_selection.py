# SPDX-License-Identifier: GPL-3.0-only
"""Freeze comparator selection before union outcomes; apply only after full audit."""
import argparse
import json
from pathlib import Path
import time

import chess_union_difference_study as quality
from openjev.research import chess_pin_study as study

ROOT = quality.ROOT
PLAN = 'evidence/chess-union-difference-quality-v1/protocol/plan.json'
RUN = 'runs/chess-union-difference-quality-v1/execution'
AUDIT = 'evidence/chess-union-difference-quality-v1/audit/receipt.json'
CODE = ['src/openjev/research/chess_pin_study.py', 'tests/test_chess_pin_study.py',
        'scripts/chess_pin_comparator_selection.py', 'tests/test_chess_pin_comparator_selection.py']
RULE = {
    'version': 'pin-comparators-before-union-outcomes-v1',
    'core_arms': list(study.CORE_ARMS),
    'selection': 'For each original panel, select the highest mean agreement across seeds97/109/127 among original WLDN and child,union,edits,rotated. Add each distinct winning union arm to the seven core arms. All selected arms will be freshly trained under the common canonical feature contract.',
    'ties': 'Exact ties prefer WLDN, then child,union,edits,rotated. Added union arms retain child,union,edits,rotated order.',
    'acceptance_independent': 'Do not condition selection on the original union hypothesis passing. A gate-failing study may still contain the strongest comparator.',
    'gate': 'Require completed12-fit execution, all86016 prediction replays,18432 checked training updates and the full frozen audit receipt before reading outcome metrics for selection.',
    'scope': 'Select comparator identities only. No pin fit, new acceptance threshold, seed replacement, quality outcome, native equivalence or inference-cost claim.',
    'limits': 'Adaptive comparator selection from previously exposed development panels. Not independent confirmation; old fitted predictions retain their original input contract and cannot be relabelled as new canonical fits.',
}


def read(path):
    return json.loads(Path(path).read_text())


def signature():
    frozen = read(ROOT/PLAN); current = quality.signature()
    if not all(frozen[k] == v for k, v in current.items()): raise ValueError('Frozen union signature changed')
    return {'rule': RULE, 'sources': {p: quality.source.file_hash(ROOT/p) for p in CODE},
            'union_signature': current, 'union_plan_sha256': quality.source.file_hash(ROOT/PLAN)}


def prepare(out):
    plan = signature(); execution = ROOT/RUN
    outcome_files = [p.name for p in execution.glob('*.jsonl') if p.name.endswith(('-dev.jsonl', '-shift.jsonl'))]
    if outcome_files or (execution/'summary.json').exists() or (execution/'all-training-complete.json').exists():
        raise RuntimeError('Pre-outcome comparator freeze requires training still incomplete and no evaluation outputs')
    completed_fits = sum(read(p).get('status') == 'completed' for p in execution.glob('*/training.json'))
    if not 0 < completed_fits < 12: raise RuntimeError('Expected the already-running unfinished quality study')
    plan['prepared_unix'] = time.time(); plan['completed_fits_at_freeze'] = completed_fits
    plan['evaluation_outputs_absent_at_freeze'] = True
    out.mkdir(parents=True, exist_ok=False); quality.source.prior.write(out/'plan.json', plan)
    print(json.dumps({'plan_sha256': quality.source.file_hash(out/'plan.json'), 'completed_fits': completed_fits,
                      'selection_applied': False}), flush=True)


def validate_audit(audit, summary, hashes):
    if (audit['status'] != 'completed' or audit['plan_sha256'] != hashes['plan']
            or audit['summary_sha256'] != hashes['summary'] or audit['execution_receipt_sha256'] != hashes['completed']
            or audit['auditor_sha256'] != hashes['auditor'] or audit['evaluation_predictions_replayed'] != 86016
            or audit['new_head_predictions_replayed'] != 61440 or audit['copied_reference_predictions_replayed'] != 24576
            or audit['training_updates_checked'] != 18432 or not audit['fresh_initial_states_exact']
            or audit['bootstrap_intervals_recomputed'] != 10 or not 0 < audit['wall_seconds'] <= 1800
            or audit['gate_recomputed'] != summary['gate_checks'] or summary['status'] != 'completed'
            or summary['plan_sha256'] != hashes['plan'] or summary['fits'] != 12
            or summary['training_updates'] != 18432 or summary['new_prediction_records'] != 61440
            or summary['copied_reference_records'] != 24576 or not 0 < summary['wall_seconds'] <= 14400):
        raise ValueError('Full union audit identity or coverage differs')
    # A failed acceptance gate is deliberately not a failed evidence audit.


def audited_outcomes():
    path = ROOT/AUDIT
    if not path.is_file(): raise RuntimeError('Comparator selection awaits the complete union evidence audit')
    audit = read(path)
    if audit.get('status') != 'completed': raise RuntimeError('Union evidence audit is not complete')
    execution = ROOT/RUN; plan_hash = quality.source.file_hash(ROOT/PLAN)
    quality.diagnostic.manifest(execution, plan_hash)
    summary = read(execution/'summary.json')
    hashes = {'plan': plan_hash, 'summary': quality.source.file_hash(execution/'summary.json'),
              'completed': quality.source.file_hash(execution/'completed.json'),
              'auditor': quality.source.file_hash(ROOT/'scripts/chess_union_difference_study.py')}
    validate_audit(audit, summary, hashes)
    return summary, audit


def select(plan_path, out):
    frozen = read(plan_path); current = signature()
    if not all(frozen[k] == v for k, v in current.items()): raise ValueError('Frozen selection rule changed')
    summary, audit = audited_outcomes()
    stamp = read(ROOT/RUN/'all-training-complete.json')
    if (not 0 < frozen['prepared_unix'] < stamp['unix'] or not 0 < frozen['completed_fits_at_freeze'] < 12
            or frozen['evaluation_outputs_absent_at_freeze'] is not True):
        raise ValueError('Comparator rule was not frozen before quality evaluation')
    selected = study.select_comparators(summary['metrics'])
    result = {'status': 'completed', 'rule_plan_sha256': quality.source.file_hash(plan_path),
              'union_audit_sha256': quality.source.file_hash(ROOT/AUDIT),
              'union_summary_sha256': audit['summary_sha256'], 'union_gate_passed': summary['continuation_passed'],
              **selected, 'future_fit_count': 3*len(selected['fit_arms']), 'quality_training_launched': False,
              'limits': RULE['limits']}
    out.mkdir(parents=True, exist_ok=False); quality.source.prior.write(out/'selection.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'select'])
    parser.add_argument('--plan', type=Path); parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    if args.command == 'prepare': prepare(args.out)
    else: select(args.plan, args.out)
