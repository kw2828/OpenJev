"""Fabricated scalar-only publication checks; no measured outputs or models."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import plot_robot_structured as plotter


def fixture():
    cfg = {'version': plotter.STUDY_VERSION, 'arms': list(plotter.FRESH_ARMS),
           'comparison_arms': list(plotter.ARMS), 'context': 32, 'dev_horizon': 128,
           'horizons': [64, 128], 'seeds': [8101, 8102, 8103],
           'learning_rates': [.001, .003], 'partitions': {'dev': ['fake-A', 'fake-B']}}
    rows, fits, resources = [], [], []
    parameters = dict(zip(plotter.ARMS, (630, 806, 806, 590, 5916, 1014), strict=True))
    def resource(arm, seed, rate=None):
        count = parameters.get(arm, 0)
        return {'arm': arm, 'seed': seed, 'learning_rate': rate, 'parameters': count,
                'parameter_bytes': count*4, 'state_bytes': 200 if arm == 'gru32' else 48,
                'buffer_bytes': 0, 'normalizer_bytes': 192,
                'inactive_parameters': 192 if arm == 'legacy_instant' else 0}
    for arm in plotter.ARMS:
        for seed in cfg['seeds']:
            for rate in cfg['learning_rates']:
                fits.append({'arm': arm, 'seed': seed, 'learning_rate': rate,
                             'origin': 'cached_parent' if arm == 'legacy_instant' else 'fresh',
                             'resources': resource(arm, seed, rate),
                             'fit': {'status': 'PASS', 'completed_updates': 4096,
                                     'optimizer_seconds': 2.5}})
            value = {8101: .001, 8102: .003, 8103: .020}[seed]
            resources.append({**resource(arm, seed, .001), 'timing': {'median_seconds': value}})
    resources.extend({**resource(a, None), 'timing': {'median_seconds': .0001}} for a in plotter.REFERENCES)
    for recording in cfg['partitions']['dev']:
        for arm in (*plotter.ARMS, *plotter.REFERENCES):
            for seed in cfg['seeds'] if arm in plotter.ARMS else [None]:
                for rate in cfg['learning_rates'] if arm in plotter.ARMS else [None]:
                    for horizon in cfg['horizons']:
                        value = {8101: 1., 8102: 2., 8103: 9., None: .5}[seed]
                        rows.append({'recording': recording, 'arm': arm, 'seed': seed,
                                     'learning_rate': rate, 'horizon': horizon, 'status': 'PASS', 'error': None,
                                     'metrics': {'standardized_rmse': value, 'physical_rmse_deg': value*2,
                                                 'standardized_sse': value*value*120, 'scalars': 120,
                                                 'windows': 1, 'per_joint_rmse_deg': [value*2]*6}})
    selection = {'selected_rates': dict.fromkeys(plotter.ARMS, .001),
                 'selected_ridge': 'causal_ridge_1',
                 'options': {a: [{'rate': r, 'eligible': True, 'pooled_rmse': 5.}
                                for r in cfg['learning_rates']] for a in plotter.ARMS},
                 'ridge_options': [{'arm': a, 'eligible': True, 'pooled_rmse': .5}
                                   for a in plotter.REFERENCES[:2]]}
    conditions = [{'name': 'fabricated-condition-' + str(i), 'passed': i < 2} for i in range(61)]
    outcome = {'total': 61, 'passed': 2, 'conditions': conditions,
               'status': 'DO_NOT_ADVANCE_STRUCTURED_TRANSITION'}
    return {'version': plotter.STUDY_VERSION, 'config': cfg, 'rows': rows,
            'selection': selection, 'result': outcome}, resources, fits


def test_complete_negative_fixture_preserves_160_36_22_and_centers():
    results, resources, fits = fixture()
    values = plotter.extract(results, resources, fits)
    assert len(values['all_rows']) == 160 and len(values['fits']) == 36
    assert len(values['resources']) == 22 and len(values['result']['conditions']) == 61
    assert values['result']['status'] == 'DO_NOT_ADVANCE_STRUCTURED_TRANSITION'
    assert values['panels'][0]['entries'][0]['center'] == 4.
    assert values['latency_ms'][0]['center'] == 3.
    assert [p['seed'] for p in values['panels'][0]['entries'][0]['points']] == [8101, 8102, 8103]
    assert values['all_rows'] is results['rows'] and values['fits'] is fits
    table = plotter.tables(values)
    assert table.count('fabricated-condition-') == 61
    assert 'Legacy instant (cached)' in table and 'historical' in table
    assert 'DO_NOT_ADVANCE_STRUCTURED_TRANSITION' in table
    assert 'CONFIRM' in table and 'not an ensemble' in table


def test_ineligible_family_and_failed_rows_remain_visible():
    results, resources, fits = fixture()
    results['selection']['selected_rates']['householder'] = None
    resources = [r for r in resources if r['arm'] != 'householder']
    for row in results['rows']:
        if row['arm'] == 'householder':
            row.update(status='FAILED', metrics=None, error={'type': 'FailedTrainingAttempt'})
    for fit in fits:
        if fit['arm'] == 'householder':
            fit['fit'].update(status='FAILED', completed_updates=7)
    values = plotter.extract(results, resources, fits)
    assert len(values['all_rows']) == 160 and len(values['fits']) == 36
    assert all(p['entries'][0]['status'] == 'INELIGIBLE' and p['entries'][0]['center'] is None for p in values['panels'])
    assert values['latency_ms'][0]['status'] == 'NOT TIMED'
    table = plotter.tables(values)
    assert 'FailedTrainingAttempt' in table and '| FAILED | 7 |' in table


@pytest.mark.parametrize('damage', ['row_missing', 'row_duplicate', 'fit_missing', 'wrong_origin',
                                    'resource_missing', 'resource_duplicate', 'wrong_rate',
                                    'nan_metric', 'wrong_gate_total', 'success_mismatch'])
def test_bad_saved_scalar_rosters_fail_closed(damage):
    results, resources, fits = fixture()
    if damage == 'row_missing':
        results['rows'].pop()
    elif damage == 'row_duplicate':
        results['rows'][-1] = copy.deepcopy(results['rows'][0])
    elif damage == 'fit_missing':
        fits.pop()
    elif damage == 'wrong_origin':
        fits[-1]['origin'] = 'fresh'
    elif damage == 'resource_missing':
        resources.pop()
    elif damage == 'resource_duplicate':
        resources[-1] = copy.deepcopy(resources[0])
    elif damage == 'wrong_rate':
        resources[0]['learning_rate'] = .003
    elif damage == 'nan_metric':
        results['rows'][0]['metrics']['standardized_rmse'] = float('nan')
    elif damage == 'wrong_gate_total':
        results['result']['total'] = 60
    else:
        results['result']['status'] = 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL'
    with pytest.raises(ValueError):
        plotter.extract(results, resources, fits)


@pytest.mark.parametrize('roundoff', [0., 5e-13])
def test_scalar_render_preserves_every_row_without_plot_or_model_execution(tmp_path, monkeypatch, roundoff):
    results, resources, fits = fixture()
    study, output = tmp_path/'study', tmp_path/'plot'
    study.mkdir()
    for name, value in [('results.json', results), ('resources.json', resources), ('fits.json', fits)]:
        (study/name).write_text(json.dumps(value))
    auth = {'study': str(study), 'plan': {'config': results['config']},
            'receipt': {'scientific_result': results['result']['status'], 'registration_sha256': 'f'*64},
            'inputs': {'fabricated': {'sha256': '0'*64, 'bytes': 0}},
            'audit': {'results': {k: results[k] for k in ('rows', 'selection', 'result')}, 'resources': resources}}
    auth = copy.deepcopy(auth)
    auth['audit']['results']['rows'][0]['metrics']['standardized_rmse'] += roundoff
    monkeypatch.setattr(plotter, 'authenticate', lambda *args: copy.deepcopy(auth))
    def fake_figure(values, folder):
        assert len(values['all_rows']) == 160
        (folder/'benchmark.png').write_bytes(b'fabricated plot')
        (folder/'benchmark.pdf').write_bytes(b'fabricated plot')
        return ['linear']*3
    monkeypatch.setattr(plotter, 'figure', fake_figure)
    receipt = plotter.render(study, output)
    assert set(receipt['outputs']) == {'benchmark.png', 'benchmark.pdf', 'all-candidates.csv', 'table.md', 'plotted-values.json'}
    assert len((output/'all-candidates.csv').read_text().splitlines()) == 161
    saved = json.loads((output/'plotted-values.json').read_text())
    assert saved['all_rows'] == results['rows'] and saved['fits'] == fits
    assert saved['result']['passed'] == 2


@pytest.mark.parametrize('actual,expected', [
    ({'rmse': 1.+1e-8}, {'rmse': 1.}),
    ({'passed': True}, {'passed': 1}),
    ({'seed': 8101}, {'seed': 8102}),
    ({'status': 'PASS'}, {'status': 'FAILED'}),
    ({'rmse': float('nan')}, {'rmse': 1.}),
    ({'rmse': 1.}, {'rmse': 1., 'extra': 0}),
])
def test_roundoff_comparison_keeps_scientific_identities_and_failures_exact(actual, expected):
    with pytest.raises(ValueError):
        plotter.scalar_agreement(actual, expected)
