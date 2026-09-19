"""Signed-check and receipt tests only; no completed experiment is audited."""
import copy
import json
from types import SimpleNamespace

import audit_pose_crossfit_repair as repair
import numpy as np
import pytest
from test_audit_pose_crossfit import constant_fixture


@pytest.mark.parametrize("actual, expected", [(-.1820086539, -.1820086539), (-.0298659847, -.0298659847),
                                             (-3, -3), (0., 0.), (-1., -1. + 5e-11)])
def test_signed_finite_values_use_original_tolerance(actual, expected):
    repair.close_number(actual, expected, 'signed scalar')


@pytest.mark.parametrize("actual, expected", [(True, 1), (1, True), (float('nan'), 0.),
    (float('inf'), float('inf')), (-float('inf'), 0.), (0., float('nan')),
    (-1., -1. + 1e-7), ('-1', -1.), (np.float32(-1.), -1.)])
def test_invalid_values_and_real_mismatches_still_rejected(actual, expected):
    with pytest.raises(ValueError):
        repair.close_number(actual, expected, 'signed scalar')


def test_negative_normal_equation_passes_original_full_certificate_after_only_scalar_repair():
    original = repair.load_original()
    record, arrays, settings = constant_fixture()
    record = copy.deepcopy(record)
    fp, fr, sp, sr, tp, tr = arrays
    tp = -np.ones_like(tp)
    record['alpha'][0] = 0.
    record['position'].update(numerator=-12., denominator=24., unclipped_alpha=-.5, objective_m2=3.)
    with pytest.raises(ValueError, match='invalid constant position numerator'):
        original.verify_constant(record, fp, fr, sp, sr, tp, tr, **settings)
    original_body, original_finite = original.audit.__code__, original.finite
    original.close_number = repair.close_number
    result = original.verify_constant(record, fp, fr, sp, sr, tp, tr, **settings)
    assert result['alpha'][0] == 0. and result['certified']
    assert result['float32_blend_diagnostic']['position_mse_m2'] == 3.
    assert original.audit.__code__ is original_body and original.finite is original_finite
    record['position']['numerator'] = -11.
    with pytest.raises(ValueError, match='constant position numerator'):
        original.verify_constant(record, fp, fr, sp, sr, tp, tr, **settings)


def test_frozen_original_source_and_tests_are_unchanged():
    assert repair.sha(repair.ORIGINAL) == repair.ORIGINAL_SHA256
    assert repair.sha(repair.ORIGINAL_TEST) == repair.ORIGINAL_TEST_SHA256


def test_changed_original_bytes_are_never_executed(tmp_path, monkeypatch):
    file = tmp_path / 'original.py'
    file.write_text("raise AssertionError('must not execute')\n")
    monkeypatch.setattr(repair, 'ORIGINAL', file)
    with pytest.raises(ValueError, match='source/test changed'):
        repair.load_original()


def test_receipt_metadata_precedes_exclusive_write_and_keeps_failed_gate(tmp_path):
    original = repair.load_original()
    metadata = {'version': 'synthetic', 'original_auditor_sha256': repair.ORIGINAL_SHA256}
    writer = repair.repaired_writer(original.write_json, metadata)
    summary = {'status': 'completed', 'continuation_gate': {'passed': False, 'total_checks': 1921}}
    writer(tmp_path / 'summary.json', summary)
    payload = (tmp_path / 'summary.json').read_bytes()
    assert 'audit_repair' not in summary
    receipt = {'status': 'completed', 'qualification_passed': False, 'files': original.members(tmp_path)}
    writer(tmp_path / 'receipt.json', receipt)
    saved = json.loads((tmp_path / 'receipt.json').read_text())
    assert saved['audit_repair'] == metadata and saved['qualification_passed'] is False
    assert saved['files']['summary.json']['sha256'] == repair.sha(tmp_path / 'summary.json')
    with pytest.raises(FileExistsError):
        writer(tmp_path / 'summary.json', summary)
    assert (tmp_path / 'summary.json').read_bytes() == payload
    writer(tmp_path / 'failed.json', {'status': 'failed', 'error': 'synthetic'})
    assert json.loads((tmp_path / 'failed.json').read_text())['audit_repair'] == metadata


def test_synthetic_failed_receipt_and_new_source_bindings(tmp_path, monkeypatch):
    failed = tmp_path / 'failed.json'
    failed.write_text(json.dumps({'status': 'failed', 'auditor_sha256': repair.ORIGINAL_SHA256,
                                 'error': "ValueError('invalid constant position numerator')"}))
    monkeypatch.setattr(repair, 'FAILED_SHA256', repair.sha(failed))
    metadata = repair.metadata(failed)
    assert metadata['original_failure_sha256'] == repair.sha(failed)
    assert metadata['repair_source_sha256'] == repair.sha(repair.__file__)
    assert metadata['repair_test_sha256'] == repair.sha(repair.REPAIR_TEST)
    assert metadata['measurement_repeated'] is False and metadata['automatic_retry'] is False
    failed.write_text('{}')
    with pytest.raises(ValueError, match='original failed audit digest'):
        repair.metadata(failed)


def test_delegates_once_with_only_private_checker_and_writer_changed(tmp_path, monkeypatch):
    calls = []
    original = SimpleNamespace(close_number=None, write_json=lambda p, value: None)
    raw = {'status': 'completed', 'qualification_passed': False}
    def fake_audit(experiment, out, **kwargs):
        assert original.close_number is repair.close_number
        calls.append((experiment, out, kwargs))
        return raw
    original.audit = fake_audit
    marker = {'repair_source_sha256': 'synthetic', 'repair_test_sha256': 'synthetic'}
    monkeypatch.setattr(repair, 'load_original', lambda: original)
    monkeypatch.setattr(repair, 'metadata', lambda failed: marker)
    experiment = tmp_path / 'experiment'; out = tmp_path / 'new-report'; failed = tmp_path / 'old-report/failed.json'
    result = repair.audit(experiment, out, protocol_sha256='synthetic-plan',
                          completed_sha256=repair.COMPLETED_SHA256, failed_audit=failed)
    assert len(calls) == 1 and calls[0][2] == {'protocol_sha256': 'synthetic-plan', 'completed_sha256': repair.COMPLETED_SHA256}
    assert result == {**raw, 'audit_repair': marker} and 'audit_repair' not in raw


@pytest.mark.parametrize('failure', ['completion', 'old_report', 'execution'])
def test_no_delegation_on_wrong_execution_or_evidence_destination(tmp_path, monkeypatch, failure):
    def forbidden():
        raise AssertionError('audit must not load')
    monkeypatch.setattr(repair, 'load_original', forbidden)
    experiment = tmp_path / 'experiment'; failed = tmp_path / 'old-report/failed.json'; out = tmp_path / 'new-report'
    completed = repair.COMPLETED_SHA256
    if failure == 'completion':
        completed = 'wrong'
    elif failure == 'old_report':
        out = failed.parent / 'nested'
    else:
        out = experiment / 'nested'
    with pytest.raises(ValueError):
        repair.audit(experiment, out, protocol_sha256='synthetic', completed_sha256=completed, failed_audit=failed)
