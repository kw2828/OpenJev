"""Fabricated-only independent BLA scoring, causal replay and opaque admission."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_fsm_author_bla as audit


def evaluations():
    return [{'family': f, 'seed': s, 'rows': [{'record_id': r, 'status': 'complete', 'rmse': .5 if f == audit.CANDIDATE else 1.} for r in audit.DEV]} for f, s in audit.ORDERED]


def bla(value=1.):
    return [{'record_id': r, 'status': 'complete', 'rmse': value} for r in audit.DEV]


def test_fixed_four_rules_and_score_only_control_selection():
    old = evaluations()
    for e in old:
        e['timing_error'] = 'deliberate old latency failure'
    result = audit.decisions(old, bla(.75), True)
    assert result['passed'] == result['total'] == 4
    assert result['strongest_control'] == audit.BLA
    assert result['candidate'] == 'tanh_feedback-lr0.0003'
    assert result['per_seed_means'][audit.CANDIDATE] == dict.fromkeys(map(str, audit.SEEDS), .5)
    assert result['per_amplitude_means'][audit.CANDIDATE] == {'100mV': .5, '200mV': .5}


@pytest.mark.parametrize('case', ['mean', 'seed', 'record', 'amplitude', 'incomplete', 'missing_candidate', 'nan_candidate'])
def test_each_frozen_rule_can_reject(case):
    old, reference = evaluations(), bla(.75)
    if case == 'mean':
        for e in old:
            if e['family'] == audit.CANDIDATE:
                for r in e['rows']:
                    r['rmse'] = .74
    elif case == 'seed':
        for e in old:
            if e['family'] == audit.CANDIDATE and e['seed'] == 9201:
                for r in e['rows']:
                    r['rmse'] = .75
    elif case == 'record':
        for e in old:
            if e['family'] == audit.CANDIDATE:
                e['rows'][0]['rmse'] = .77
    elif case == 'amplitude':
        for e in old:
            if e['family'] == audit.CANDIDATE:
                for r in e['rows'][:6]:
                    r['rmse'] = .75
    elif case == 'missing_candidate':
        next(e for e in old if e['family'] == audit.CANDIDATE)['rows'].pop()
    elif case == 'nan_candidate':
        next(e for e in old if e['family'] == audit.CANDIDATE)['rows'][0]['rmse'] = float('nan')
    result = audit.decisions(old, reference, case != 'incomplete')
    assert result['status'] == 'DO_NOT_CONTINUE_REFERENCE_CHECK'
    name = {'mean': 'five_percent_below_strongest_control', 'seed': 'every_seed_below_strongest_control', 'record': 'no_record_over_two_percent_strongest_control', 'amplitude': 'both_amplitudes_below_strongest_control'}.get(case)
    if name:
        assert result['conditions'][name] is False
    else:
        assert result['passed'] == 0


def test_stochastic_strongest_uses_paired_seed_not_pooled_mean():
    old = evaluations()
    for e in old:
        if e['family'] == audit.NEURAL[2]:
            for r in e['rows']:
                r['rmse'] = {9201: .25, 9202: 1., 9203: 1.}[e['seed']]
    result = audit.decisions(old, bla(2.), True)
    assert result['strongest_control'] == audit.NEURAL[2]
    assert result['conditions']['five_percent_below_strongest_control'] is True
    assert result['conditions']['every_seed_below_strongest_control'] is False


def test_duplicate_or_reordered_comparison_slot_rejected():
    old = evaluations()
    old[0] = old[1]
    with pytest.raises(ValueError, match='25 fixed'):
        audit.decisions(old, bla(), True)


def test_metrics_independent_known_native_channel_units():
    target = np.zeros((32, 128, 3), dtype=np.float64)
    prediction = np.broadcast_to(np.array([1., 2., 3.]), target.shape).copy()
    scored = audit.metrics(prediction, target, np.array([2., 3., 4.]))
    assert scored['mse'] == 14/3
    assert scored['rmse'] == pytest.approx(np.sqrt(14/3))
    assert scored['per_channel_rmse'] == [1., 2., 3.]
    assert scored['native_output_per_channel_rmse'] == [2., 6., 12.]
    prediction.fill(1e308)
    with pytest.raises(ValueError, match='overflow'):
        audit.metrics(prediction, target, np.ones(3))


def fabricated_model():
    a = np.zeros((28, 28))
    a[:3, :3] = np.diag([.5, .25, .75])
    b = np.zeros((28, 3)); b[:3] = np.eye(3)*.2
    c = np.zeros((3, 28)); c[:, :3] = np.eye(3)
    return {'A': a, 'B_u': b, 'C_y': c, 'D_yu': np.diag([.3, .4, .5]), 'u_mean': np.array([3., -2., 1.]), 'u_std': np.array([2., 3., 4.]), 'y_mean': np.array([-1., 2., 6.]), 'y_std': np.array([.5, 2., 3.]), 'ts': np.array(1/6400)}


def causal_fixture():
    m = fabricated_model()
    rng = np.random.default_rng(41)
    inputs = rng.normal(size=(2, 227, 3))
    state = np.zeros((2, 28)); state[:, :3] = [[1., 2., -1.], [-2., 3., .5]]
    output = []
    for t in range(227):
        output.append(state@m['C_y'].T+inputs[:, t]@m['D_yu'].T)
        state = state@m['A'].T+inputs[:, t]@m['B_u'].T
    physical = np.stack(output, axis=1)*m['y_std']+m['y_mean']
    u = inputs*m['u_std']+m['u_mean']
    y = np.concatenate([np.full((2, 1, 3), 1234.), physical[:, :99]], axis=1)
    return m, y, u[:, :99].copy(), u[:, 99:].copy(), physical[:, 99:], state


def test_independent_replay_rank_deficiency_direct_feedthrough_and_causality():
    m, y, u, future, target, state = causal_fixture()
    before = [x.copy() for x in (y, u, future)]
    pred, final, singular, residual, rank = audit.replay(m, y, u, future)
    np.testing.assert_allclose(pred, target, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(final, state, rtol=1e-12, atol=1e-12)
    assert rank == 3 and singular.shape == (28,)
    assert np.max(residual) < 1e-12
    for x, old in zip((y, u, future), before, strict=True):
        np.testing.assert_array_equal(x, old)
    y[:, 0] *= -77
    future[:, 10:] += 2
    changed = audit.replay(m, y, u, future)[0]
    np.testing.assert_array_equal(pred[:, :10], changed[:, :10])
    assert not np.array_equal(pred[:, 10], changed[:, 10])


def test_frequency_response_scalar_diagonal_oracle():
    m = fabricated_model()
    z = np.exp(2j*np.pi*np.arange(1, 3840)/8192)
    target = np.zeros((3839, 3, 3), dtype=np.complex128)
    for i, pole in enumerate((.5, .25, .75)):
        target[:, i, i] = .2/(z-pole)+m['D_yu'][i, i]
    assert audit.frequency_error(m, target) < 1e-30
    target += 1
    assert audit.frequency_error(m, target) == pytest.approx(1.)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def opaque_fixture(tmp_path, monkeypatch):
    root = tmp_path/'repo'; root.mkdir()
    study = root/'output/fsm-author-bla-study-v1'; study.mkdir(parents=True)
    reference = root/'output/fsm-linear-controls-study-v1'; reference.mkdir(parents=True)
    monkeypatch.setattr(audit, 'ROOT', root)
    source = root/'fake.py'; source.write_text('opaque source')
    paths = {}
    def item(key, relative_path, value):
        p = root/relative_path
        write(p, value)
        paths[key] = p
        return {'path': relative_path, 'sha256': audit.pin(p)['sha256']}
    prereqs = {}
    prereqs['parent_summary'] = item('parent_summary', 'output/fsm-linear-controls-study-v1/summary.json', {'status': 'DEVELOPMENT_PASS'})
    prereqs['parent_evaluations'] = item('parent_evaluations', 'output/fsm-linear-controls-study-v1/evaluations.json', [])
    prereqs['parent_registration'] = item('parent_registration', 'parent-registration.json', {})
    prereqs['parent_process'] = item('parent_process', 'parent-process.json', {'observed_exit_code': 0, 'summary_sha256': audit.pin(paths['parent_summary'])['sha256']})
    prereqs['parent_closure'] = item('parent_closure', 'output/fsm-linear-controls-study-v1/closure.json', {'status': 'completed', 'summary_sha256': audit.pin(paths['parent_summary'])['sha256']})
    (reference/'invalid.npz').write_bytes(b'not an array; admission must not decode')
    prior = {'status': 'PASS', 'agreement': True, 'inputs': {'registration': audit.pin(paths['parent_registration']), 'process': audit.pin(paths['parent_process']), 'files': audit.inventory(reference)}}
    prereqs['parent_audit'] = item('parent_audit', 'parent-audit.json', prior)
    plan = {'version': audit.VERSION, 'experiment': audit.EXPERIMENT, 'source_sha256': {'fake.py': audit.pin(source)['sha256']}, 'prerequisites': prereqs}
    write(root/audit.REG, plan)
    reg_sha = audit.pin(root/audit.REG)['sha256']
    monkeypatch.setattr(audit, 'REGISTRATION_SHA256', reg_sha)
    process = root/'process/process.json'; process.parent.mkdir()
    log = process.parent/'process.log'; log.write_text('closed')
    terminal = {'command': [str(root/'research/fsm_author/.venv/bin/python'), str(root/'research/fsm_author/scripts/fit_bla.py'), '--registration', audit.REG, '--output', 'output/fsm-author-bla-study-v1'], 'registration_sha256': reg_sha, 'timeout_seconds': 1800, 'end_identity_matches': True, 'status': 'completed', 'observed_exit_code': 0, 'elapsed_seconds': 1., 'log_sha256': audit.pin(log)['sha256'], 'pid': 1}
    write(process, terminal)
    write(study/'started.json', {'registration_sha256': reg_sha, 'pid': 1})
    monkeypatch.setattr(audit.np, 'load', lambda *_a, **_k: pytest.fail('numeric decode during admission'))
    return root, study, process, paths


def test_opaque_authenticate_never_decodes(tmp_path, monkeypatch):
    _, study, process, _ = opaque_fixture(tmp_path, monkeypatch)
    _, inputs, _, _ = audit.authenticate(study, process)
    assert 'invalid.npz' in inputs['reference_files']
    assert inputs['files'] == audit.inventory(study)


@pytest.mark.parametrize('case', ['source', 'registration', 'parent_payload', 'process_live', 'exit', 'identity', 'log', 'command', 'pid', 'parent_audit', 'symlink'])
def test_admission_tamper_before_any_decoder(tmp_path, monkeypatch, case):
    root, study, process, paths = opaque_fixture(tmp_path, monkeypatch)
    if case == 'source':
        (root/'fake.py').write_text('changed')
    elif case == 'registration':
        (root/audit.REG).write_text('{}')
    elif case == 'parent_payload':
        (paths['parent_summary'].parent/'invalid.npz').write_bytes(b'changed')
    elif case == 'log':
        (process.parent/'process.log').write_text('changed')
    elif case == 'parent_audit':
        paths['parent_audit'].write_text('{}')
    elif case == 'symlink':
        (study/'alias').symlink_to(root/'fake.py')
    else:
        terminal = audit.read(process)
        key, value = {'process_live': ('status', 'running'), 'exit': ('observed_exit_code', 1), 'identity': ('end_identity_matches', False), 'command': ('command', ['wrong']), 'pid': ('pid', 2)}[case]
        terminal[key] = value
        write(process, terminal)
    with pytest.raises(ValueError):
        audit.authenticate(study, process)


def test_no_admitted_raw_archive_or_parent_fit_invocation_in_auditor_source():
    source = Path(audit.__file__).read_text()
    assert 'import freq_statespace' not in source and 'import openjev' not in source
    assert 'allow_pickle=False' in source
    assert hashlib.sha256(source.encode()).hexdigest()
    # Scalars/metadata helpers do not change caller objects.
    old = evaluations(); before = copy.deepcopy(old)
    audit.decisions(old, bla(), True)
    assert old == before
