"""Fabricated scalar-only publication checks; no measured outputs or models."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import package_robot_reflection_capacity as packager
import plot_robot_reflection_capacity as plotter


def fixture():
    cfg = {'version': plotter.STUDY_VERSION, 'arms': list(plotter.FRESH_ARMS),
           'comparison_arms': list(plotter.ARMS), 'context': 32, 'dev_horizon': 128,
           'horizons': [64, 128], 'seeds': [8101, 8102, 8103],
           'learning_rates': [.001, .003], 'partitions': {'dev': ['fake-A', 'fake-B']}}
    rows, fits, resources = [], [], []
    parameters = dict(zip(plotter.ARMS, (806, 630, 806, 806, 590, 5916, 1014), strict=True))
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
                             'origin': 'fresh' if arm == 'householder12' else 'cached_parent',
                             'resources': resource(arm, seed, rate),
                             'fit': {'status': 'PASS', 'completed_updates': 4096,
                                     'optimizer_seconds': 2.5}})
                item = fits[-1]
                if item['origin'] == 'fresh':
                    item['effective_status'] = 'PASS'
                else:
                    item['parent_origin'] = 'cached_parent' if arm == 'legacy_instant' else 'fresh'
                    item['parent_fit'] = {'fit': copy.deepcopy(item['fit']), 'origin': item['parent_origin']}
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
    conditions = [{'name': 'fabricated-condition-' + str(i), 'passed': i < 2} for i in range(69)]
    outcome = {'total': 69, 'passed': 2, 'conditions': conditions,
               'status': 'DO_NOT_ADVANCE_REFLECTION_CAPACITY', 'accuracy': {'passed': 2, 'total': 67}, 'compute': {'passed': 0, 'total': 2}}
    return {'version': plotter.STUDY_VERSION, 'config': cfg, 'rows': rows,
            'selection': selection, 'result': outcome}, resources, fits


def test_complete_negative_fixture_preserves_184_42_25_and_centers():
    results, resources, fits = fixture()
    values = plotter.extract(results, resources, fits)
    assert len(values['all_rows']) == 184 and len(values['fits']) == 42
    assert len(values['resources']) == 25 and len(values['result']['conditions']) == 69
    assert values['result']['status'] == 'DO_NOT_ADVANCE_REFLECTION_CAPACITY'
    assert values['panels'][0]['entries'][0]['center'] == 4.
    assert values['latency_ms'][0]['center'] == 3.
    assert [p['seed'] for p in values['panels'][0]['entries'][0]['points']] == [8101, 8102, 8103]
    assert values['all_rows'] is results['rows'] and values['fits'] is fits
    table = plotter.tables(values)
    assert table.count('fabricated-condition-') == 69
    assert 'Legacy instant (cached)' in table and 'historical' in table
    assert 'DO_NOT_ADVANCE_REFLECTION_CAPACITY' in table
    assert 'CONFIRM' in table and 'not an ensemble' in table


def test_ineligible_family_and_failed_rows_remain_visible():
    results, resources, fits = fixture()
    results['selection']['selected_rates']['householder12'] = None
    resources = [r for r in resources if r['arm'] != 'householder12']
    for row in results['rows']:
        if row['arm'] == 'householder12':
            row.update(status='FAILED', metrics=None, error={'type': 'FailedTrainingAttempt'})
    for fit in fits:
        if fit['arm'] == 'householder12':
            fit['fit'].update(status='FAILED', completed_updates=7)
            fit['effective_status'] = 'FAILED'
    values = plotter.extract(results, resources, fits)
    assert len(values['all_rows']) == 184 and len(values['fits']) == 42
    assert all(p['entries'][0]['status'] == 'INELIGIBLE' and p['entries'][0]['center'] is None for p in values['panels'])
    assert values['latency_ms'][0]['status'] == 'NOT TIMED'
    table = plotter.tables(values)
    assert 'FailedTrainingAttempt' in table and '| FAILED / FAILED | 7 |' in table


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
        assert len(values['all_rows']) == 184
        (folder/'benchmark.png').write_bytes(b'fabricated plot')
        (folder/'benchmark.pdf').write_bytes(b'fabricated plot')
        return ['linear']*4
    monkeypatch.setattr(plotter, 'figure', fake_figure)
    receipt = plotter.render(study, output)
    assert set(receipt['outputs']) == {'benchmark.png', 'benchmark.pdf', 'all-candidates.csv', 'table.md', 'plotted-values.json'}
    assert len((output/'all-candidates.csv').read_text().splitlines()) == 185
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


def test_split_counts_storage_and_preservation_failure_are_explicit():
    results, resources, fits = fixture()
    fits[0]['effective_status'] = 'FAILED'
    values = plotter.extract(results, resources, fits)
    assert values['result']['accuracy'] == {'passed': 2, 'total': 67}
    assert values['result']['compute'] == {'passed': 0, 'total': 2}
    assert values['storage_kib'][0]['bytes'] == 3464
    assert values['storage_kib'][1]['bytes'] == 2760
    assert values['storage_kib'][2]['bytes'] == 3464
    assert values['storage_kib'][4]['bytes'] == 2600
    assert 'FAILED / PASS' in plotter.tables(values)
    results['result']['accuracy']['passed'] = 3
    with pytest.raises(ValueError, match='accuracy67'):
        plotter.extract(results, resources, fits)


def original_roster():
    results, _, _ = fixture()
    cfg = results['config']
    names = {'registration.json', 'runtime.json', 'normalizers.npz', 'parent-normalizers.npz',
             'linear.npz', 'causal_ridge_1.npz', 'causal_ridge_1.json', 'causal_ridge_100.npz',
             'causal_ridge_100.json', 'parent-fits.json', 'checkpoint-barrier.json',
             'results.json', 'resources.json', 'fits.json'}
    names |= {f'batches-{s}.npz' for s in cfg['seeds']}
    names |= {f'completed-fit-{i:02d}.json' for i in range(1, 43)}
    keys = [f'{a}-{s}-lr{i}' for a in plotter.ARMS for s in cfg['seeds'] for i in range(2)]
    names |= {f'{key}/{f}' for key in keys for f in packager.FIT_FILES}
    names |= {'sources/' + name for name in plotter.SOURCES}
    names |= {f'dev-windows-{name}.npz' for name in cfg['partitions']['dev']}
    names |= {f'prediction-{name}-{arm}.npz' for name in cfg['partitions']['dev'] for arm in plotter.REFERENCES}
    names |= {f'prediction-{name}-{key}.npz' for name in cfg['partitions']['dev'] for key in keys}
    return {'config': cfg}, {name: {'sha256': '0'*64, 'bytes': 1} for name in names}


def test_exact_derived_roster_excludes_only_two_targets_and_allows_failed_forecast():
    plan, roster = original_roster()
    excluded = packager.study_roster(roster, plan)
    assert set(excluded) == {'dev-windows-fake-A.npz', 'dev-windows-fake-B.npz'}
    roster.pop('prediction-fake-A-householder12-8101-lr0.npz')
    assert packager.study_roster(roster, plan) == excluded
    assert sum('/final.npz' in key for key in roster) == 42
    assert sum(key.startswith('prediction-') for key in roster) == 91


@pytest.mark.parametrize('damage', ['raw', 'target_alias', 'checkpoint_missing', 'source_missing', 'target_missing', 'traversal'])
def test_package_rejects_unregistered_file_types_or_incomplete_originals(damage):
    plan, roster = original_roster()
    if damage == 'raw':
        roster['dev-data-secret.npz'] = {}
    elif damage == 'target_alias':
        roster['targets.npz'] = {}
    elif damage == 'traversal':
        roster['../outside.json'] = {}
    elif damage == 'checkpoint_missing':
        roster.pop('householder-8102-lr1/final.npz')
    elif damage == 'source_missing':
        roster.pop('sources/' + plotter.SOURCES[0])
    else:
        roster.pop('dev-windows-fake-A.npz')
    with pytest.raises(ValueError, match='authored child'):
        packager.study_roster(roster, plan)


@pytest.mark.parametrize('which', ['render', 'package'])
def test_failed_admission_precedes_all_scalar_reads_and_output_creation(tmp_path, monkeypatch, which):
    seen = []
    def denied(*args):
        seen.append('authentication')
        raise ValueError('original audit not closed')
    monkeypatch.setattr(plotter, 'authenticate', denied)
    monkeypatch.setattr(plotter, 'read_json', lambda *args: pytest.fail('scalar read before admission'))
    output = tmp_path/'output'
    with pytest.raises(ValueError, match='not closed'):
        if which == 'render':
            plotter.render(tmp_path/'study', output)
        else:
            packager.package(tmp_path/'study', tmp_path/'audit.json', tmp_path/'plot', tmp_path/'engineering', output)
    assert seen == ['authentication'] and not output.exists()


def test_wrong_registration_rejects_before_original_process_or_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(plotter, 'ROOT', tmp_path)
    registration = tmp_path/'research/robot-reflection-capacity-registration.json'
    registration.parent.mkdir()
    registration.write_text('{"untrusted": "not admitted"}')
    monkeypatch.setattr(plotter, 'read_json', lambda *args: pytest.fail('untrusted JSON read'))
    with pytest.raises(ValueError, match='registration hash'):
        plotter.authenticate(tmp_path/'study')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_package_opaque_roundtrip_keeps_all_cached_forecasts_and_exclusion_hashes(tmp_path, monkeypatch):
    study, audit_dir, plots, engineering, output = [tmp_path/name for name in ('study', 'audit', 'plot', 'engineering', 'public')]
    plan, roster = original_roster()
    for name in roster:
        path = study/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('opaque:' + name).encode())
        roster[name] = plotter.pin(path)
    write_json(study/'manifest.json', {'files': roster})
    write_json(study/'receipt.json', {'status': 'PASS'})
    audit_path = audit_dir/'audit.json'
    write_json(audit_path, {'status': 'PASS'})
    write_json(audit_dir/'manifest.json', {'files': {'audit.json': plotter.pin(audit_path)}})
    for name in (*packager.PLOT_FILES, 'receipt.json'):
        path = plots/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('plot:' + name).encode())
    process_names = ('run-launch-01.json', 'run-process-01.json', 'run-process-01.log',
                     'audit-process-01.json', 'audit-process-01.log', 'plot-process-01.json', 'plot-process-01.log')
    for name in process_names:
        path = engineering/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('process:' + name).encode())
    fake_root = tmp_path/'repo'
    monkeypatch.setattr(plotter, 'ROOT', fake_root)
    for name in packager.DELIVERY_FILES:
        path = fake_root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('source:' + name).encode())
    write_json(fake_root/'research/robot-structured-registration.json', {'parent': True})
    closure = engineering/'parent-receipt.json'
    write_json(closure, {'status': 'PASS'})
    plan.update(parent_closure={'receipt': {'path': str(closure), **plotter.pin(closure)}}, data={'external-only': {'sha256': '1'*64, 'bytes': 99}})
    result = fixture()[0]['result']
    auth = {'study': str(study), 'audit_path': str(audit_path), 'plan': plan,
            'audit': {'results': {'result': result}}}
    monkeypatch.setattr(plotter, 'authenticate', lambda *args: copy.deepcopy(auth))
    monkeypatch.setattr(packager, 'plot_admission', lambda *args: {'fabricated': True})
    monkeypatch.setattr(packager, 'qualification_files', lambda *args: set())
    receipt = packager.package(study, audit_path, plots, engineering, output)
    manifest = json.loads((output/'manifest.json').read_text())
    assert receipt['excluded_targets'] == 2
    assert len(manifest['excluded_target_payloads']) == 2
    for name, expected in manifest['files'].items():
        assert plotter.pin(output/name) == expected
    for name, descriptor in manifest['copy_sources'].items():
        assert plotter.pin(output/name) == plotter.pin(descriptor['path'])
    assert not (output/'study/dev-windows-fake-A.npz').exists()
    assert not (output/'study/dev-windows-fake-B.npz').exists()
    assert sum(name.startswith('study/prediction-') for name in manifest['files']) == 92
    assert sum(name.endswith('/final.npz') for name in manifest['files']) == 42
    assert manifest['external_measurement_pins'] == plan['data']
    assert 'DO_NOT_ADVANCE_REFLECTION_CAPACITY' in (output/'README.md').read_text()
    with pytest.raises(FileExistsError):
        packager.package(study, audit_path, plots, engineering, output)
