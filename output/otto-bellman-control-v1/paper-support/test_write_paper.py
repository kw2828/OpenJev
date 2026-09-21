"""Fabricated presentation checks; no study inputs, plotting, or compilation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('_paper_test', ROOT / 'scripts/write_otto_bellman_paper.py')
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def fabricated():
    regimes, amortization = {}, {}
    for regime in P.R.REGIMES:
        metric = {'found': .875, 'steps': 17.5, 'controller_seconds': .02}
        regimes[regime] = {'means': {a: dict(metric) for a in P.R.ARMS},
                          'family_means': {f: dict(metric) for f in P.R.FAMILIES},
                          'raw_counts': {a: {'episodes': 48, 'found': 40} for a in P.R.ARMS}}
        amortization[regime] = {a: {'seconds_per_search': {'1': 12.3, '100': .123, '10000': .021}} for a in P.R.ARMS}
    competence = [f'{r}.{s}.{k}' for r in P.R.REGIMES for s in P.R.SEEDS for k in ('success', 'moves')]
    improvement = [f'{r}.{c}.{k}' for r in P.R.REGIMES for c in ('mc', 'reference')
                   for k in ('success', 'moves', 'positive_blocks', 'controller_cost')]
    def checks(names):
        return [{'name': n, 'value': 1., 'threshold': 1., 'passes': True} for n in names]
    result = {'regimes': regimes, 'amortization': amortization, 'episodes': 1440, 'paired_cases': 144,
              'dataset_rows': {'train': 5589, 'valid': 1109}, 'competence_checks': checks(competence),
              'improvement_checks': checks(improvement), 'pilot_continuation': False,
              'costs': {'training': {'preparation_seconds': 1.2, 'fits': dict.fromkeys(P.FIT_IDS, 2.3)}, 'worker_seconds': 34.5},
              'final_prediction_diagnostics': {a: {s: {'mse_normalized': .01, 'mae_physical': 2.} for s in ('train', 'valid')} for a in P.FIT_IDS},
              'comparisons': 12345, 'saved_checkpoint_readout_calls': 246}
    result['competence_checks'][0]['passes'] = False
    return result


def test_complete_paper_preserves_failure_raw_counts_and_horizons():
    value = fabricated()
    text = P.document(value, {'audit': {'sha256': 'a'*64}})
    assert len(P.records(value)) == 30
    assert 'FAIL: 41/42' in text and '17/18 competence; 24/24' in text
    assert text.count('40/48') == 30 and text.count('87.50') >= 42
    assert text.count('12.3000') == text.count('0.1230') == text.count('0.0210') == 30
    for check in value['competence_checks'] + value['improvement_checks']:
        assert P.tex(check['name']) in text
    for arm in P.R.ARMS:
        assert text.count(P.label(arm)) >= 3
    assert r'Success (\%)' in text and r'\textbackslash{}\%' not in text
    assert 'not total training compute' in text and 'not optimal costs' in text
    assert 'improve autonomous control' in text and 'improve a competent controller' not in text


def test_outcome_prose_uses_each_setting_and_guards_perfect_baseline():
    value = fabricated()
    for i, regime in enumerate(P.R.REGIMES):
        value['regimes'][regime]['family_means']['backup']['steps'] = 10. + i
        value['regimes'][regime]['family_means']['mc']['steps'] = 20. + i
        value['regimes'][regime]['means']['analytic_inbounds']['found'] = 1.
    value['regimes']['lambda5']['family_means']['reference']['steps'] = 11.
    text = P.outcome(value)
    assert all(f'{r}: {10+i:.3f} versus {20+i:.3f}' in text for i, r in enumerate(P.R.REGIMES))
    assert 'moves were higher' in text and r'100\% weighted success in all three' in text
    value['regimes']['lambda5']['means']['analytic_inbounds']['found'] = .75
    assert r'lambda5: 75.00\%' in P.outcome(value)
    assert r'100\% weighted success in all three' not in P.outcome(value)


@pytest.mark.parametrize('defect', ['arm', 'fit', 'gate', 'raw', 'nonfinite'])
def test_incomplete_or_contradictory_summary_rejected(defect):
    value = fabricated()
    if defect == 'arm':
        del value['regimes']['lambda5']['means']['reference@10103']
    elif defect == 'fit':
        del value['final_prediction_diagnostics']['mc@10102']
    elif defect == 'gate':
        value['pilot_continuation'] = True
    elif defect == 'raw':
        value['regimes']['lambda3']['raw_counts']['backup@10101']['episodes'] = 47
    else:
        value['regimes']['lambda4']['means']['analytic_inbounds']['steps'] = float('inf')
    with pytest.raises(ValueError):
        P.records(value)


def test_escaping_cannot_inject_latex():
    assert P.tex(r'a_b% & \input{x}') == r'a\_b\% \& \textbackslash{}input\{x\}'


def test_authentication_failure_precedes_render_and_is_preserved(tmp_path, monkeypatch):
    def denied(_):
        raise ValueError('fabricated unauthenticated evidence')
    def forbidden(*_):
        raise AssertionError('render must not start')
    monkeypatch.setattr(P.R.Report, 'authenticate', denied)
    monkeypatch.setattr(P, 'plot', forbidden)
    output = tmp_path / 'denied'
    with pytest.raises(ValueError, match='unauthenticated'):
        P.execute(SimpleNamespace(output=output))
    assert (output / 'failed.json').exists()
    assert not (output / 'receipt.json').exists()
    assert not (output / 'paper.tex').exists()
    assert not (output / 'comparison.pdf').exists()
