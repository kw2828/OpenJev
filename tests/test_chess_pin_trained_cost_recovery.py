# SPDX-License-Identifier: GPL-3.0-only
import copy
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import chess_pin_trained_cost as original_cost
import chess_pin_trained_cost_recovery as recovery


@pytest.fixture
def old_tests(monkeypatch):
    path = recovery.ROOT/'tests/test_chess_pin_trained_cost.py'
    spec = importlib.util.spec_from_file_location('_isolated_cost_fixtures', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'cost', recovery.cost)
    return module


def test_isolation_protocol_and_frozen_predecessor_identity():
    assert recovery.cost is not original_cost and recovery.cost is not recovery.legacy
    assert recovery.cost.study is not original_cost.study
    assert original_cost.study.VERSION == 'canonical-matched-pin-quality-v2'
    assert recovery.cost.study.VERSION == 'canonical-matched-pin-quality-v3'
    assert recovery.cost.study.__file__ == original_cost.study.__file__
    assert recovery.cost.__file__ == original_cost.__file__
    assert recovery.cost.PLAN == recovery.QUALITY_PLAN
    assert recovery.cost.EXECUTION == recovery.EXECUTION
    assert recovery.cost.AUDIT == recovery.AUDIT
    old = recovery.recovery.read(recovery.ROOT/recovery.OLD_PLAN)
    assert recovery.recovery.sha(recovery.ROOT/recovery.OLD_PLAN) == recovery.OLD_PLAN_SHA
    assert recovery.recovery.sha(recovery.ROOT/recovery.QUALITY_PLAN) == recovery.QUALITY_PLAN_SHA
    assert all(recovery.recovery.sha(recovery.ROOT/path) == digest for path, digest in old['sources'].items())
    expected = copy.deepcopy(old['protocol']); expected['version'] = recovery.VERSION
    assert recovery.cost.PROTOCOL == expected
    assert recovery.legacy.PROTOCOL == original_cost.PROTOCOL == old['protocol']
    quality = recovery.recovery.read(recovery.ROOT/recovery.QUALITY_PLAN)
    assert old['quality_signature'] == quality['prior_signature']


def test_fresh_checkpoint_version_and_plan_are_required(tmp_path):
    study = recovery.cost.study
    head = study.mechanics.make_head('joint', 97)
    directory = tmp_path/'fits'/study.name('joint', 97)
    directory.mkdir(parents=True)
    value = {'version': study.VERSION, 'seed': 97, 'arm': 'joint',
             'plan_sha256': recovery.QUALITY_PLAN_SHA, 'state_dict': head.state_dict()}
    path = directory/'weights.pt'
    torch.save(value, path)
    loaded = study.load_head(tmp_path, 97, 'joint', recovery.QUALITY_PLAN_SHA)
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in head.state_dict().items())
    for field, wrong in [('version', original_cost.study.VERSION), ('plan_sha256', 'old-plan'),
                         ('seed', 109), ('arm', 'wldn')]:
        bad = {**value, field: wrong}; torch.save(bad, path)
        with pytest.raises(AssertionError, match='checkpoint identity'):
            study.load_head(tmp_path, 97, 'joint', recovery.QUALITY_PLAN_SHA)


@pytest.mark.parametrize('index', range(5))
def test_reused_full_native_equivalence_fixtures(old_tests, index):
    old_tests.test_complete_optimized_paths_match_full_native_inputs_with_nonzero_heads(old_tests.FENS[index])


@pytest.mark.parametrize('name', [
    'test_unused_input_work_is_skipped_and_bad_identity_rejected',
    'test_protocol_membership_rotates_every_method_into_every_slot',
    'test_pin_coverage_matches_independent_detector_and_transition_records',
    'test_complete_summary_preserves_failures_and_checks_raw_vectors_membership',
    'test_ratio_uses_matched_pairs_and_empty_strata_remain_null',
    'test_preoutcome_freeze_rejects_evaluation_or_terminal_markers',
    'test_no_quality_outputs_are_read_before_complete_audit',
    'test_quality_failure_is_allowed_but_incomplete_audit_or_late_freeze_is_not',
    'test_primary_and_audit_execute_complete_native_fixture_and_detect_saved_corruption',
    'test_deadlines_remain_bounded',
])
def test_reused_cost_fixture_contracts(old_tests, name, tmp_path, monkeypatch):
    function = getattr(old_tests, name)
    available = {'tmp_path': tmp_path, 'monkeypatch': monkeypatch}
    function(**{key: available[key] for key in inspect.signature(function).parameters})


def test_quality_authentication_keeps_actual_kernel_auditor_identity(old_tests, tmp_path, monkeypatch):
    plan, path, audit = old_tests.quality_fixture(tmp_path, monkeypatch)
    expected, evidence = recovery.cost.authenticate(plan)
    assert len(expected) == 18 and evidence['quality_continuation_passed'] is False
    assert recovery.cost.study.__file__.endswith('chess_pin_quality_study_v2.py')
    audit['auditor_sha256'] = recovery.recovery.sha(Path(recovery.recovery.__file__))
    path.write_text(json.dumps(audit))
    with pytest.raises(AssertionError, match='audit coverage or identity'):
        recovery.cost.authenticate(plan)


@pytest.fixture
def signature_fixture(tmp_path, monkeypatch):
    tracked = tmp_path/'wrapper.py'; tracked.write_text('frozen source')
    old = {'protocol': copy.deepcopy(recovery.legacy.PROTOCOL),
           'quality_signature': {'protocol': {'version': 'quality-v2'}},
           'quality_plan_sha256': 'prior-quality', 'panel': [{'id': 'root0'}],
           'pin_input_coverage': {'roots': []}, 'input_sha256': {'inputs': 'hash'},
           'prepared_unix': 1.}
    native = {'protocol': copy.deepcopy(recovery.cost.PROTOCOL),
              'quality_signature': {'prior_signature': copy.deepcopy(old['quality_signature']),
                                    'prior_plan_sha256': old['quality_plan_sha256']},
              'quality_plan_sha256': recovery.QUALITY_PLAN_SHA,
              **{key: copy.deepcopy(old[key]) for key in ('panel', 'pin_input_coverage', 'input_sha256')}}
    monkeypatch.setattr(recovery, 'prior_plan', lambda: copy.deepcopy(old))
    monkeypatch.setattr(recovery, 'native_signature', lambda: {
        **copy.deepcopy(native), 'sources': {'wrapper.py': recovery.recovery.sha(tracked)}})
    return old, native, tracked


def test_signature_binds_lineage_paths_sources_and_unchanged_panel(signature_fixture):
    old, _, _ = signature_fixture
    value = recovery.signature()
    assert value['prior_cost_plan_sha256'] == recovery.OLD_PLAN_SHA
    assert value['prior_cost_signature'] == {k: v for k, v in old.items() if k != 'prepared_unix'}
    assert value['quality_bindings'] == {'plan': recovery.QUALITY_PLAN,
                                       'execution': recovery.EXECUTION, 'audit': recovery.AUDIT}


@pytest.mark.parametrize('field', ['quality_plan_sha256', 'prior_signature', 'prior_plan_sha256',
                                 'protocol', 'panel', 'pin_input_coverage', 'input_sha256'])
def test_lineage_protocol_and_input_corruption_rejected(signature_fixture, field):
    _, native, _ = signature_fixture
    if field in ('prior_signature', 'prior_plan_sha256'):
        native['quality_signature'][field] = 'corrupt'
    else:
        native[field] = 'corrupt'
    with pytest.raises(ValueError):
        recovery.signature()


def plan_fixture(path):
    value = {**recovery.signature(), 'freeze_state': {'completed_fits': 0,
             'quality_started_unix': 1., 'evaluation_outputs_absent': True}, 'prepared_unix': 2.}
    path.write_text(json.dumps(value))
    return value


def test_strict_plan_and_freeze_membership_reject_corruption(signature_fixture, tmp_path):
    path = tmp_path/'plan.json'; good = plan_fixture(path)
    assert recovery.validate_plan(path)[0] == good
    bads = [{**good, 'extra': 'field'}, {k: v for k, v in good.items() if k != 'sources'},
            {**good, 'sources': {}}, {**good, 'prior_cost_plan_sha256': 'wrong'},
            {**good, 'quality_bindings': {}}]
    for field, value in [('completed_fits', True), ('completed_fits', 24), ('completed_fits', -1),
                         ('evaluation_outputs_absent', False), ('quality_started_unix', float('nan')),
                         ('quality_started_unix', 3.), ('extra', True)]:
        bads.append({**good, 'freeze_state': {**good['freeze_state'], field: value}})
    for stamp in (True, float('nan'), -1., .5, float('inf')):
        bads.append({**good, 'prepared_unix': stamp})
    for bad in bads:
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            recovery.validate_plan(path)


def test_source_change_invalidates_new_plan(signature_fixture, tmp_path):
    _, _, tracked = signature_fixture
    path = tmp_path/'plan.json'; plan_fixture(path)
    recovery.validate_plan(path)
    tracked.write_text('changed source')
    with pytest.raises(ValueError, match='protocol or sources'):
        recovery.validate_plan(path)


def test_predecessor_plan_hash_and_validator_identity_required(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    path = tmp_path/recovery.OLD_PLAN; path.parent.mkdir(parents=True); path.write_text('{}')
    monkeypatch.setattr(recovery, 'OLD_PLAN_SHA', recovery.recovery.sha(path))
    monkeypatch.setattr(recovery.legacy, 'validate_plan', lambda p: ({'frozen': True}, recovery.OLD_PLAN_SHA))
    assert recovery.prior_plan() == {'frozen': True}
    monkeypatch.setattr(recovery.legacy, 'validate_plan', lambda p: ({}, 'wrong'))
    with pytest.raises(ValueError, match='validation identity'):
        recovery.prior_plan()
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='predecessor cost plan changed'):
        recovery.prior_plan()
