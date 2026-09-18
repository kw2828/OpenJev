"""Constructed fixtures for interruption preservation and unchanged study scope."""
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import chess_pin_quality_recovery as recovery
import chess_pin_quality_study_v2 as original
from test_chess_pin_quality_study_v2 import ARMS, inputs, tiny_protocol


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def fixture_attempt(tmp_path):
    _, _, data, graphs, targets = inputs(tmp_path)
    p = tiny_protocol()
    p.update(arms=['joint', 'separable'], methods=['base', 'joint', 'separable'], fits=2, training_updates=6)
    execution = tmp_path/'execution'
    (execution/'fits').mkdir(parents=True)
    recovery.write(execution/'started.json', {'unix': time.time(), 'pid': 101,
                                             'plan_sha256': 'a'*64})
    for arm in p['arms']:
        recovery.base.train_fit(97, arm, data, graphs, targets, execution/'fits'/f'{arm}-97',
                                'a'*64, p, time.monotonic())
    partial = execution/'fits/separable-97'
    records = recovery.base.source.prior.rows(partial/'learning.jsonl')[:2]
    (partial/'learning.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (partial/'training.json').unlink()
    (partial/'weights.pt').unlink()
    return execution, {'protocol': p}


def test_interrupted_prefix_costs_and_completed_artifact_hashes(tmp_path):
    execution, old = fixture_attempt(tmp_path)
    actual = recovery.partial_inventory(execution, old, 'a'*64)
    assert actual['completed_fits'] == 1
    assert actual['logged_completed_updates'] == 5
    assert actual['quality_predictions'] == 0 and actual['evaluation_outputs_absent']
    assert [v['completed'] for v in actual['fits']] == [True, False]
    assert actual['logged_update_seconds'] > 0 and actual['completed_fit_seconds'] > 0
    (execution/'fits/joint-97/initial.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='bytes changed'):
        recovery.partial_inventory(execution, old, 'a'*64)


@pytest.mark.parametrize('fault', ['evaluation', 'extra_fit', 'missing_fit', 'nan_loss', 'extra_checkpoint'])
def test_partial_audit_rejects_outcomes_selection_and_bad_updates(tmp_path, fault):
    execution, old = fixture_attempt(tmp_path)
    if fault == 'evaluation':
        (execution/'predictions').mkdir()
    elif fault == 'extra_fit':
        (execution/'fits/wldn-999').mkdir()
    elif fault == 'missing_fit':
        (execution/'fits/joint-97').rename(execution/'fits/unused')
    elif fault == 'extra_checkpoint':
        (execution/'fits/separable-97/best.pt').write_bytes(b'fixture')
    else:
        p = execution/'fits/separable-97/learning.jsonl'
        records = recovery.base.source.prior.rows(p)
        records[0]['loss'] = float('nan')
        p.write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises((ValueError, AssertionError)):
        recovery.partial_inventory(execution, old, 'a'*64)


def stub_signature(monkeypatch):
    old = {'protocol': original.protocol(ARMS, 20573.874974548817), 'sources': {'previous': 'd'*64},
           'prepared_unix': 1.}
    evidence = {'logged_completed_updates': 8239, 'completed_fits': 5,
                'logged_update_seconds': 3147., 'completed_fit_seconds': 3016.}
    monkeypatch.setattr(recovery, 'prior_plan', lambda: copy.deepcopy(old))
    monkeypatch.setattr(recovery, 'retained_attempt', lambda _: dict(evidence))
    monkeypatch.setattr(recovery, 'sha', lambda path: 'c'*64)
    return old


def test_protocol_preserves_every_scientific_field_and_does_not_mutate_v2(monkeypatch):
    old = stub_signature(monkeypatch)
    before = copy.deepcopy(old)
    value = recovery.signature()
    assert old == before
    for key, item in old['protocol'].items():
        if key not in ('version', 'prior_knowledge'):
            assert value['protocol'][key] == item
    assert value['protocol']['fits'] == 24 and value['protocol']['training_updates'] == 36864
    assert value['protocol']['quality_checks'] == 16
    assert value['protocol']['time_cap_seconds'] == 43200
    assert value['protocol']['audit_time_cap_seconds'] == 7200
    assert value['recovery']['partial_checkpoint_resume'] is False
    assert value['recovery']['prior_termination_cause_known'] is False


def test_execution_engine_is_isolated_and_uses_new_checkpoint_identity():
    before_version, before_validate = original.VERSION, original.validate_plan
    separate = recovery.execution_engine()
    assert separate.VERSION == recovery.VERSION
    assert separate.validate_plan is recovery.validate_plan
    assert original.VERSION == before_version and original.validate_plan is before_validate
    assert recovery.base.VERSION == before_version
    assert recovery.base.validate_plan is not recovery.validate_plan


@pytest.mark.parametrize('mutation', ['gate', 'budget', 'source', 'extra'])
def test_frozen_plan_rejects_changes(tmp_path, monkeypatch, mutation):
    stub_signature(monkeypatch)
    recovery.prepare(tmp_path/'protocol')
    path = tmp_path/'protocol/plan.json'
    recovery.validate_plan(path)
    plan = recovery.read(path)
    if mutation == 'gate': plan['protocol']['quality_checks'] -= 1
    elif mutation == 'budget': plan['protocol']['time_cap_seconds'] += 1
    elif mutation == 'source': plan['sources']['extra'] = 'e'*64
    else: plan['extra'] = 'ignored'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='Frozen recovery'):
        recovery.validate_plan(path)


def test_live_or_uncertain_process_blocks_failure_record(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    monkeypatch.setattr(recovery, 'OLD_LAUNCH', 'launch.json')
    recovery.write(tmp_path/'launch.json', {'pid_at_launch': 101})
    monkeypatch.setattr(recovery, 'prior_plan', dict)
    for code, stdout in [(0, '101 python'), (2, ''), (0, '')]:
        monkeypatch.setattr(recovery.subprocess, 'run',
                            lambda *a, code=code, stdout=stdout, **kw: subprocess.CompletedProcess(a, code, stdout, ''))
        with pytest.raises(ValueError, match='PID is present or absence'):
            recovery.record_interruption(tmp_path/'review.json')
    assert not (tmp_path/'review.json').exists()


def test_isolated_v3_real_fixture_updates_match_v2_but_checkpoint_versions_differ(tmp_path):
    _, _, data, graphs, targets = inputs(tmp_path)
    p = tiny_protocol()
    engine = recovery.execution_engine()
    for label, module in [('old', original), ('new', engine)]:
        directory = tmp_path/label/'fits/joint-97'
        directory.parent.mkdir(parents=True)
        module.train_fit(97, 'joint', data, graphs, targets, directory, label+'-fixture', p, time.monotonic())
    left = original.load_head(tmp_path/'old', 97, 'joint', 'old-fixture')
    right = engine.load_head(tmp_path/'new', 97, 'joint', 'new-fixture')
    assert left.state_dict().keys() == right.state_dict().keys()
    assert all(torch.equal(value, right.state_dict()[key]) for key, value in left.state_dict().items())
    with pytest.raises(AssertionError, match='checkpoint identity'):
        engine.load_head(tmp_path/'old', 97, 'joint', 'old-fixture')
