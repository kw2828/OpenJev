"""Small fabricated presentation checks, with no outcomes or rendering."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('spatial_control_report_test', ROOT/'scripts/report_otto_spatial_control.py')
S = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = S
SPEC.loader.exec_module(S)


def test_wrong_external_plan_is_refused_before_any_evidence_read(tmp_path, monkeypatch):
    args = SimpleNamespace(plan=tmp_path/'plan', run=tmp_path/'run', audit=tmp_path/'audit',
                           terminal=tmp_path/'terminal', plan_sha256='0'*64)
    monkeypatch.setattr(S, 'digest', lambda *_: pytest.fail('untrusted plan caused an evidence read'))
    with pytest.raises(ValueError, match='exact external plan'):
        S.authenticate(args, lambda: None)


@pytest.mark.parametrize('defect', ['missing', 'started', 'failed', 'late_failure'])
def test_missing_incomplete_or_late_failed_audit_cannot_be_closed(tmp_path, defect):
    audit = tmp_path/'audit'
    audit.mkdir()
    pin = '0'*64
    if defect != 'missing':
        S.write(audit/'receipt.json', {'status': 'completed' if defect == 'late_failure' else defect, 'files': {}})
        pin = S.digest(audit/'receipt.json')['sha256']
    if defect == 'late_failure':
        S.write(audit/'failed.json', {'status': 'failed'})
    with pytest.raises(ValueError):
        S.closed(audit, pin, set(), lambda: None)


def criteria():
    out = {'pilot_continuation': True}
    for key, families in [('competence_checks', ['spatial']),
                          ('control_competence_checks', ['neighbor_free', 'cnn', 'dense128', 'statistics'])]:
        out[key] = [{'name': f'lambda{lam}.{family}.{seed}.{metric}', 'passes': True}
                    for lam in (3, 4, 5) for family in families for seed in (10101, 10102, 10103)
                    for metric in ('success', 'moves')]
    out['improvement_checks'] = [{'name': f'lambda{lam}.{family}.{metric}', 'passes': True}
        for lam in (3, 4, 5) for family in ('neighbor_free', 'cnn', 'dense128', 'statistics')
        for metric in ('success', 'moves', 'positive_blocks', 'cost')]
    return out


def test_all138_conditions_retained_and_descriptive_failures_do_not_change_gate():
    data = criteria()
    for row in data['control_competence_checks']:
        row['passes'] = False
    counts = S.condition_counts(data)
    assert counts == {'competence_checks': {'passed': 18, 'total': 18},
                      'improvement_checks': {'passed': 48, 'total': 48},
                      'control_competence_checks': {'passed': 0, 'total': 72}}
    data['competence_checks'][0]['passes'] = False
    with pytest.raises(ValueError, match='exact frozen screen'):
        S.condition_counts(data)


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'wrong_control', 'not_boolean'])
def test_partial_or_relabelled_criterion_set_is_refused(defect):
    data = criteria()
    if defect == 'missing':
        data['control_competence_checks'].pop()
    elif defect == 'duplicate':
        data['improvement_checks'][-1] = copy.deepcopy(data['improvement_checks'][0])
    elif defect == 'wrong_control':
        data['improvement_checks'][0]['name'] = 'lambda3.unregistered.success'
    else:
        data['competence_checks'][0]['passes'] = 1
    with pytest.raises(ValueError, match='criterion identities'):
        S.condition_counts(data)


def episodes():
    arms = [f'{kind}@{seed}' for seed in (10101, 10102, 10103)
            for kind in ('spatial', 'neighbor_free', 'cnn', 'dense128', 'statistics')]+['analytic_inbounds']
    rows = []
    for ri, first in enumerate((16100001, 16200001, 16300001)):
        for case in range(24):
            offset = (ri*24+case) % 16
            for arm in arms[offset:]+arms[:offset]:
                rows.append({'regime': f'lambda{ri+3}', 'seed': first+case, 'block': case//3,
                    'initial_hit': 1+case % 3, 'arm': arm, 'steps': 1, 'found': True, 'updates': 1,
                    'blocked_steps': 0, 'final_update_assimilated': True})
    return rows


@pytest.mark.parametrize('defect', [None, 'missing', 'duplicate', 'wrong_rotation', 'short_censor', 'missing_update', 'wrong_calls'])
def test_exact1152_cohort_and_full_episode_accounting(tmp_path, defect):
    rows, calls = episodes(), {'value_forward': 1080, 'analytic_choose': 72}
    if defect == 'missing':
        rows.pop()
    elif defect == 'duplicate':
        rows[-1] = rows[0]
    elif defect == 'wrong_rotation':
        rows[16], rows[17] = rows[17], rows[16]
    elif defect == 'short_censor':
        rows[0]['found'] = False
    elif defect == 'missing_update':
        rows[0]['final_update_assimilated'] = False
    elif defect == 'wrong_calls':
        calls['value_forward'] -= 1
    path = tmp_path/'episodes.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    if defect:
        with pytest.raises(ValueError):
            S.membership(path, calls, lambda: None)
    else:
        raw, strata = S.membership(path, calls, lambda: None)
        assert len(rows) == 1152 and len(raw) == 48 and set(raw.values()) == {24}
        assert len(strata) == 144 and set(strata.values()) == {8}
