# SPDX-License-Identifier: GPL-3.0-only
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import chess_pin_comparator_selection as runner


def fixture():
    hashes = {'plan': 'plan', 'summary': 'summary', 'completed': 'completed', 'auditor': 'auditor'}
    gates = [{'passed': False}]*10
    summary = {'status': 'completed', 'plan_sha256': 'plan', 'fits': 12, 'training_updates': 18432,
               'new_prediction_records': 61440, 'copied_reference_records': 24576, 'wall_seconds': 14000,
               'gate_checks': gates, 'continuation_passed': False}
    audit = {'status': 'completed', 'plan_sha256': 'plan', 'summary_sha256': 'summary',
             'execution_receipt_sha256': 'completed', 'auditor_sha256': 'auditor',
             'evaluation_predictions_replayed': 86016, 'new_head_predictions_replayed': 61440,
             'copied_reference_predictions_replayed': 24576, 'training_updates_checked': 18432,
             'fresh_initial_states_exact': True, 'bootstrap_intervals_recomputed': 10,
             'wall_seconds': 1000, 'gate_recomputed': gates}
    return audit, summary, hashes


def test_failed_hypothesis_remains_eligible_but_incomplete_or_wrong_audit_does_not():
    audit, summary, hashes = fixture(); runner.validate_audit(audit, summary, hashes)
    for key, value in (('evaluation_predictions_replayed', 61440), ('status', 'failed'),
                       ('training_updates_checked', 1536), ('summary_sha256', 'other'),
                       ('fresh_initial_states_exact', False), ('wall_seconds', 1801)):
        bad = copy.deepcopy(audit); bad[key] = value
        with pytest.raises(ValueError): runner.validate_audit(bad, summary, hashes)


def test_missing_audit_rejected_before_reading_any_outcome(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    def forbid_read(path): raise AssertionError('Outcome read before audit existed')
    monkeypatch.setattr(runner, 'read', forbid_read)
    with pytest.raises(RuntimeError, match='awaits'): runner.audited_outcomes()


def test_post_training_rule_timestamp_cannot_be_applied(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    frozen = {'rule': {}, 'prepared_unix': 20, 'completed_fits_at_freeze': 11,
              'evaluation_outputs_absent_at_freeze': True}
    plan = tmp_path/'plan.json'; plan.write_text(json.dumps(frozen))
    stamp = tmp_path/runner.RUN/'all-training-complete.json'; stamp.parent.mkdir(parents=True)
    stamp.write_text(json.dumps({'unix': 10}))
    monkeypatch.setattr(runner, 'signature', lambda: {'rule': {}})
    monkeypatch.setattr(runner, 'audited_outcomes', lambda: ({}, {}))
    with pytest.raises(ValueError, match='before quality evaluation'):
        runner.select(plan, tmp_path/'selection')
    assert not (tmp_path/'selection').exists()
