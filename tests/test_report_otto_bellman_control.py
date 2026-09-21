"""Fabricated presentation and authentication fixtures; no scientific evidence."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_test_bellman_report', ROOT / 'scripts/report_otto_bellman_control.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def summary(passes=False):
    regimes, amortization = {}, {}
    for ri, regime in enumerate(M.REGIMES):
        means = {arm: {'found': .8 + .01 * i, 'steps': 20. + ri + i, 'controller_seconds': .01 * (i + 1)}
                 for i, arm in enumerate(M.ARMS)}
        regimes[regime] = {'means': means, 'weights': {'1': .5, '2': .3, '3': .2},
            'family_means': {f: {k: sum(means[f'{f}@{s}'][k] for s in M.SEEDS) / 3
                                for k in ('found', 'steps', 'controller_seconds')} for f in M.FAMILIES}}
        amortization[regime] = {a: {'seconds_per_search': {'1': 10., '100': 1., '10000': .1}} for a in M.ARMS}
    names = [f'{r}.{s}.{k}' for r in M.REGIMES for s in M.SEEDS for k in ('success', 'moves')]
    names += [f'{r}.{c}.{k}' for r in M.REGIMES for c in ('mc', 'reference')
              for k in ('success', 'moves', 'positive_blocks', 'controller_cost')]
    checks = [{'name': n, 'value': .4, 'threshold': .5, 'passes': passes} for n in names]
    return {'version': 'otto-bellman-control-saved-audit-v1', 'agreement': True, 'episodes': 1440,
        'paired_cases': 144, 'regimes': regimes, 'competence_checks': checks[:18], 'improvement_checks': checks[18:],
        'pilot_continuation': passes, 'learned_architecture_advantage_established': False, 'comparisons': 123456,
        'amortization': amortization, 'costs': {'worker_seconds': 999., 'training': {'preparation_seconds': 2.,
            'fits': {f'{f}@{s}': 3. for f in ('mc', 'backup') for s in M.SEEDS}}}}


def args(output):
    return SimpleNamespace(output=output, run=ROOT / 'output/fabricated/run', audit=ROOT / 'output/fabricated/audit')


def test_all_30_rows_90_bars_and_failed_gate_kept(tmp_path):
    result = summary()
    markdown, svg, values = M.present(result, args(tmp_path))
    assert '**FAIL: 0/42' in markdown and 'Competence: 0/18; paired improvement: 0/24' in markdown
    assert len(values['rows']) == 30 and values['gate_total'] == 42
    tree = ElementTree.fromstring(svg)
    assert len(tree.findall('.//{http://www.w3.org/2000/svg}rect')) == 91
    for regime in M.REGIMES:
        for arm in M.ARMS:
            row = next(r for r in values['rows'] if r['regime'] == regime and r['arm'] == arm)
            assert row['controller_seconds'] == result['regimes'][regime]['means'][arm]['controller_seconds']
    assert 'not equal total compute' in markdown and 'not additional searches' in markdown
    assert 'original scalar study remains a 0/54 failure' in markdown
    assert 'connectome or recurrent-model advantage' in markdown
    assert markdown.count('| backup@10101 |') == 4  # Three settings plus paid fit cost.


def test_passing_gate_is_not_an_architecture_claim(tmp_path):
    markdown, svg, _ = M.present(summary(True), args(tmp_path))
    assert '**PASS: 42/42' in markdown and 'PASS: 42/42' in svg
    assert 'does not establish an architecture' in markdown


def test_equal_values_have_equal_widths_across_different_regime_maxima(tmp_path):
    result = summary()
    for i, regime in enumerate(M.REGIMES):
        result['regimes'][regime]['means'][M.ARMS[0]].update(found=.9, steps=17., controller_seconds=.25)
        result['regimes'][regime]['means']['analytic_inbounds'].update(steps=100. * (i + 1), controller_seconds=2. * (i + 1))
    _, svg, _ = M.present(result, args(tmp_path))
    tree = ElementTree.fromstring(svg)
    bars = tree.findall('.//{http://www.w3.org/2000/svg}rect')[1:]
    for metric in range(3):
        assert len({bars[regime * 30 + metric * 10].attrib['width'] for regime in range(3)}) == 1
    assert len(tree.findall('.//{http://www.w3.org/2000/svg}line')) == 9
    assert 'same zero-based scale' in svg


@pytest.mark.parametrize('corrupt', ['missing', 'duplicate', 'boolean', 'overall'])
def test_rules_reject_missing_or_inconsistent_conditions(corrupt):
    result = summary()
    if corrupt == 'missing':
        result['competence_checks'].pop()
    elif corrupt == 'duplicate':
        result['improvement_checks'][0] = result['improvement_checks'][1]
    elif corrupt == 'boolean':
        result['competence_checks'][0]['passes'] = 0
    else:
        result['pilot_continuation'] = True
    with pytest.raises(ValueError):
        M.rules(result)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False))


def repin(directory):
    receipt = M.read(directory / 'receipt.json')
    receipt['files'] = {p.name: M.digest(p) for p in directory.iterdir() if p.name != 'receipt.json'}
    dump(directory / 'receipt.json', receipt)
    return M.digest(directory / 'receipt.json')['sha256']


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(M, 'ROOT', tmp_path)
    for name in (M.RUNNER, M.AUDITOR):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text('fabricated source')
    sources = {n: M.digest(tmp_path / n)['sha256'] for n in (M.RUNNER, M.AUDITOR)}
    monkeypatch.setattr(M, 'AUDITOR_PIN', sources[M.AUDITOR])
    a = SimpleNamespace(plan=tmp_path / 'plan.json', terminal=tmp_path / 'terminal.json',
                        run=tmp_path / 'run', audit=tmp_path / 'audit', output=tmp_path / 'report')
    plan = {'version': 'otto-bellman-control-v1', 'status': 'frozen_before_execution', 'mode': 'study',
            'sources': sources, 'inputs': {}, 'limits': {}}
    dump(a.plan, plan)
    a.plan_sha256 = M.digest(a.plan)['sha256']
    launch = {'pid': 2, 'pgid': 2, 'parent_pid': 1, 'command': ['fabricated'], 'cwd': str(tmp_path),
              'started_ns': 1, 'deadline_ns': 100, 'clock_backend': 'fabricated', 'cap_seconds': 99}
    launch_path = tmp_path / 'launch.json'
    dump(launch_path, launch)
    terminal = {**launch, 'status': 'completed', 'returncode': 0, 'timed_out': False, 'group_absent': True,
                'cleanup': {'reaped': True, 'errors': []}, 'error': None, 'clock_error': None, 'finished_ns': 9}
    dump(a.terminal, terminal)
    a.terminal_sha256 = M.digest(a.terminal)['sha256']
    for name in M.payload_names():
        dump(a.run / name, {'fabricated': True})
    result = summary()
    producer = copy.deepcopy(result)
    producer['amortization'] = {'scenarios': result['amortization']}
    dump(a.run / 'summary.json', producer)
    dump(a.run / 'started.json', {'launch': launch, 'request': {'plan': str(a.plan), 'plan_sha256': a.plan_sha256,
         'output': str(a.run), 'supervision': str(launch_path)}})
    worker = {**plan, 'status': 'completed', 'plan_sha256': a.plan_sha256, 'completed_episodes': 1440,
              'completed_fits': 6, 'pending': [], 'external_model_calls': 0, 'target_refreshes': 24,
              'target_audit_passed': True, 'parity_passed': True, 'calls': {'example': {'attempted': 1, 'returned': 1}},
              'supervision_sha256': M.digest(launch_path)['sha256'], 'started_ns': 2, 'finished_ns': 8, 'pilot_continuation': False}
    dump(a.run / 'receipt.json', worker)
    a.receipt_sha256 = repin(a.run)
    request = {k: str(getattr(a, k)) for k in ('plan', 'run', 'terminal', 'plan_sha256', 'receipt_sha256', 'terminal_sha256')}
    dump(a.audit / 'started.json', {'request': {**request, 'output': str(a.audit)}})
    dump(a.audit / 'summary.json', result)
    audit = {'version': result['version'], 'status': 'completed', 'agreement': True, 'worker_sha256': a.receipt_sha256,
             'plan_sha256': a.plan_sha256, 'terminal_sha256': a.terminal_sha256, 'source': {'sha256': sources[M.AUDITOR]},
             'producer_source_sha256': sources[M.RUNNER]}
    dump(a.audit / 'receipt.json', audit)
    a.audit_receipt_sha256 = repin(a.audit)
    reader = M.Report(a)
    monkeypatch.setattr(reader, 'check', lambda: None)
    return a, reader


def test_closed_synthetic_evidence_and_all_payloads_authenticated(evidence):
    _, reader = evidence
    assert len(M.payload_names()) == 91
    assert reader.authenticate()['episodes'] == 1440
    assert set(reader.receipt['inputs']) == {'plan', 'terminal', 'worker', 'audit', 'audit_summary', 'worker_summary'}


@pytest.mark.parametrize('corrupt', ['receipt', 'payload', 'extra', 'source', 'terminal', 'audit_failed', 'audit_gate'])
def test_authentication_rejects_altered_or_inconsistent_evidence(evidence, corrupt):
    a, reader = evidence
    if corrupt == 'receipt':
        a.receipt_sha256 = '0' * 64
    elif corrupt == 'payload':
        (a.run / 'mc-targets-10101.npz').write_text('altered bytes')
    elif corrupt == 'extra':
        (a.run / 'late.json').write_text('{}')
    elif corrupt == 'source':
        (M.ROOT / M.RUNNER).write_text('altered source')
    elif corrupt == 'terminal':
        terminal = M.read(a.terminal)
        terminal['returncode'] = 1
        dump(a.terminal, terminal)
        a.terminal_sha256 = M.digest(a.terminal)['sha256']  # Even a new external pin cannot make failure success.
    else:
        if corrupt == 'audit_failed':
            record = M.read(a.audit / 'receipt.json')
            record['agreement'] = False
            dump(a.audit / 'receipt.json', record)
        else:
            record = M.read(a.audit / 'summary.json')
            record['pilot_continuation'] = True
            dump(a.audit / 'summary.json', record)
        a.audit_receipt_sha256 = repin(a.audit)
    with pytest.raises(ValueError):
        reader.authenticate()


def test_exclusive_staging_and_failure_preservation(tmp_path, monkeypatch):
    a = args(tmp_path / 'stage')
    monkeypatch.setattr(M.Report, 'authenticate', lambda self: summary())
    receipt = M.execute(a)
    assert receipt['status'] == 'completed' and receipt['model_calls'] == 0
    assert set(receipt['files']) == {M.REPORT, M.FIGURE, 'plotted-values.json'}
    preserved = (a.output / 'receipt.json').read_bytes()
    with pytest.raises(ValueError, match='exclusive'):
        M.execute(a)
    assert (a.output / 'receipt.json').read_bytes() == preserved
    a.output = tmp_path / 'failed'
    def fail(_):
        raise ValueError('fabricated authentication failure')
    monkeypatch.setattr(M.Report, 'authenticate', fail)
    with pytest.raises(ValueError, match='fabricated'):
        M.execute(a)
    assert M.read(a.output / 'failed.json')['status'] == 'failed'
    assert not (a.output / M.REPORT).exists()
