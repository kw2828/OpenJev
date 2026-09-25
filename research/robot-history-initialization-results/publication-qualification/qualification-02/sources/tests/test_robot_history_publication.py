"""Fabricated saved-scalar presentation fixtures; no scientific inputs."""
import copy
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import plot_robot_history_initialization as plotter


def fixture():
    cfg = {'version': plotter.STUDY_VERSION, 'arms': list(plotter.PRIMARY),
           'comparison_arms': list(plotter.ARMS), 'context': 32, 'dev_horizon': 128,
           'horizons': [64, 128], 'seeds': [8101, 8102, 8103], 'learning_rates': [.001, .003],
           'partitions': {'dev': ['fabricated-dev-A', 'fabricated-dev-B']},
           'older_permutation': [*range(29, -1, -1), 30, 31]}
    rows, resources, fits = [], [], []
    def score(recording, arm, seed, rate, horizon, value):
        return {'recording': recording, 'arm': arm, 'seed': seed, 'learning_rate': rate,
                'horizon': horizon, 'status': 'PASS', 'error': None,
                'metrics': {'horizon': horizon, 'standardized_rmse': value, 'physical_rmse_deg': value * 2,
                            'standardized_sse': value**2 * 120, 'scalars': 120, 'windows': 1,
                            'per_joint_rmse_deg': [value * 2] * 6}}
    for name in cfg['partitions']['dev']:
        for arm in plotter.ARMS:
            for seed in cfg['seeds']:
                for rate in cfg['learning_rates']:
                    for h in cfg['horizons']:
                        rows.append(score(name, arm, seed, rate, h, 1 + (seed - 8101) / 10))
        for arm in plotter.REFERENCES:
            for h in cfg['horizons']:
                rows.append(score(name, arm, None, None, h, 2.))
    def resource(arm, seed):
        count = 962 if arm in ('temporal_affine', 'local_affine') else 590
        return {'arm': arm, 'seed': seed, 'learning_rate': .001 if seed is not None else None,
                'parameters': count, 'parameter_bytes': count * 4, 'state_bytes': 48,
                'buffer_bytes': 0, 'normalizer_bytes': 192, 'inactive_parameters': 0,
                'timing': {'median_seconds': .001 + ((seed or 8101) - 8101) * .001,
                           'seconds': [.001] * 20}}
    for arm in plotter.ARMS:
        for seed in cfg['seeds']:
            for ri, rate in enumerate(cfg['learning_rates']):
                r = resource(arm, seed)
                fits.append({'key': f'{arm}-{seed}-lr{ri}', 'arm': arm, 'seed': seed, 'learning_rate': rate,
                             'origin': 'fresh' if arm in plotter.PRIMARY else 'cached_parent',
                             'effective_status': 'PASS', 'fit': {'status': 'PASS', 'completed_updates': 4096, 'optimizer_seconds': 1.},
                             'resources': {k: v for k, v in r.items() if k not in ('arm', 'seed', 'learning_rate', 'timing')}})
            resources.append(resource(arm, seed))
    resources += [resource(arm, None) for arm in plotter.REFERENCES]
    selection = {'selected_rates': dict.fromkeys(plotter.ARMS, .001),
                 'options': {a: [{'rate': r, 'eligible': True, 'pooled_rmse': 1.1} for r in cfg['learning_rates']] for a in plotter.ARMS}}
    result = {'status': 'DO_NOT_ADVANCE_HISTORY_INITIALIZATION', 'passed': 0, 'total': 5,
              'conditions': [{'name': name, 'passed': False} for name in plotter.CONDITIONS]}
    prows = [copy.deepcopy(r) for r in rows if r['arm'] in plotter.PRIMARY and r['learning_rate'] == .001]
    checks = [{'recording': name, 'arm': arm, 'seed': seed, 'learning_rate': .001,
               'status': 'PASS', 'error': None, 'prediction_saved': True,
               'check': {'exact_invariance': True, 'maximum_absolute_change': 0.}}
              for name in cfg['partitions']['dev'] for arm in plotter.PRIMARY for seed in cfg['seeds']]
    permutation = {'scientific_gate': False, 'permutation': cfg['older_permutation'], 'rows': prows, 'checks': checks}
    return {'version': plotter.STUDY_VERSION, 'config': cfg, 'rows': rows, 'selection': selection, 'result': result}, resources, fits, permutation


def test_all_candidates_seeds_costs_and_diagnostics_survive():
    values = plotter.extract(*fixture())
    assert len(values['all_rows']) == 208 and len(values['fits']) == 48
    assert len(values['resources']) == 28 and len(values['permutation']['rows']) == 36
    assert len(values['permutation']['checks']) == 18 and len(values['order']) == 12
    assert values['order'][:3] == ['temporal_affine', 'local_affine', 'last_two']
    for panel in values['panels']:
        for row in panel['entries']:
            assert len(row['points']) == (3 if row['arm'] in plotter.ARMS else 1)
            assert row['center'] == pytest.approx(1.1 if row['arm'] in plotter.ARMS else 2.)
    temporal = values['latency_ms'][0]
    assert [p['value'] for p in temporal['points']] == [1., 2., 3.]
    assert temporal['center'] == 2.
    assert values['storage_kib'][0]['bytes'] == 4088
    assert values['storage_kib'][2]['bytes'] == 2600


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'nonfinite', 'missing_fit', 'cached_as_fresh',
                                   'extra_cost', 'missing_cost', 'changed_gate', 'wrong_outcome',
                                   'missing_diagnostic', 'duplicate_diagnostic', 'different_permutation', 'gating_diagnostic'])
def test_incomplete_or_changed_saved_comparison_rejected(damage):
    results, resources, fits, permutation = fixture()
    if damage == 'missing': results['rows'].pop()
    elif damage == 'duplicate': results['rows'][-1] = copy.deepcopy(results['rows'][0])
    elif damage == 'nonfinite': results['rows'][0]['metrics']['standardized_rmse'] = float('nan')
    elif damage == 'missing_fit': fits.pop()
    elif damage == 'cached_as_fresh': fits[-1]['origin'] = 'fresh'
    elif damage == 'extra_cost': resources.append(copy.deepcopy(resources[0]))
    elif damage == 'missing_cost': resources.pop()
    elif damage == 'changed_gate': results['result']['conditions'][0]['name'] = 'easier gate'
    elif damage == 'wrong_outcome': results['result']['status'] = 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL'
    elif damage == 'missing_diagnostic': permutation['rows'].pop()
    elif damage == 'duplicate_diagnostic': permutation['rows'][-1] = copy.deepcopy(permutation['rows'][0])
    elif damage == 'different_permutation': permutation['permutation'] = list(range(32))
    else: permutation['scientific_gate'] = True
    with pytest.raises(ValueError):
        plotter.extract(results, resources, fits, permutation)


def test_failed_family_and_failed_reference_remain_visible_not_zero():
    results, resources, fits, permutation = fixture()
    results['selection']['selected_rates']['gru10'] = None
    for option in results['selection']['options']['gru10']: option.update(eligible=False, pooled_rmse=None)
    for row in results['rows']:
        if row['arm'] in ('gru10', 'persistence'):
            row.update(status='FAILED', metrics=None, error={'type': 'FabricatedFailure'})
    resources = [r for r in resources if r['arm'] not in ('gru10', 'persistence')]
    values = plotter.extract(results, resources, fits, permutation)
    for arm in ('gru10', 'persistence'):
        panel = next(r for r in values['panels'][0]['entries'] if r['arm'] == arm)
        cost = next(r for r in values['latency_ms'] if r['arm'] == arm)
        assert panel['center'] is None and cost['center'] is None
        assert panel['points'] == [] and cost['points'] == []
    ref = next(r for r in values['storage_kib'] if r['arm'] == 'persistence')
    assert ref['bytes'] is None and ref['status'] == 'UNAVAILABLE'
    assert len(values['fits']) == 48 and len(values['all_rows']) == 208


def test_diagnostic_failure_is_printed_without_changing_outcome():
    results, resources, fits, permutation = fixture()
    for row in permutation['rows']:
        if row['arm'] == 'temporal_affine' and row['seed'] == 8101:
            row.update(status='FAILED', metrics=None, error={'type': 'NonfiniteDiagnostic'})
    for row in permutation['checks']:
        if row['arm'] == 'temporal_affine' and row['seed'] == 8101:
            row.update(status='FAILED', check=None, error={'type': 'NonfiniteDiagnostic'})
    values = plotter.extract(results, resources, fits, permutation)
    assert values['result'] == results['result']
    table = plotter.tables(values)
    assert 'NonfiniteDiagnostic' in table and '0/5 criteria' in table
    assert 'All 208 ordinary scores' in table and 'all 36 descriptive scores' in table


def test_shared_accuracy_limits_and_explicit_dynamic_scales():
    values = plotter.extract(*fixture())
    specs = plotter.axis_specs(values)
    assert specs[0] == specs[1] and specs[0]['limits'][0] == 0
    values['panels'][0]['entries'][0]['points'][0]['value'] = 1000.
    specs = plotter.axis_specs(values)
    assert specs[0] == specs[1] and specs[0]['scale'] == 'log'
    values['panels'][1]['entries'][0]['points'][0]['value'] = 0.
    specs = plotter.axis_specs(values)
    assert specs[0] == specs[1] and specs[0]['scale'] == 'symlog'
    assert specs[0]['limits'][0] == 0


def test_csv_retains_all_rows_and_uses_lf(tmp_path):
    results, *_ = fixture()
    path = tmp_path / 'all.csv'
    plotter.write_csv(path, results['rows'])
    raw = path.read_bytes()
    assert b'\r\n' not in raw
    with path.open() as handle: rows = list(csv.DictReader(handle))
    assert len(rows) == 208 and json.loads(rows[0]['per_joint_rmse_deg']) == [2.] * 6
    with pytest.raises(FileExistsError): plotter.write_csv(path, results['rows'])


def test_fabricated_plot_creates_standalone_exports(tmp_path):
    values = plotter.extract(*fixture())
    scales = plotter.figure(values, tmp_path, fabricated=True)
    assert len(scales) == 4 and scales[0] == scales[1]
    assert (tmp_path / 'benchmark.png').read_bytes().startswith(b'\x89PNG')
    assert (tmp_path / 'benchmark.pdf').read_bytes().startswith(b'%PDF')


def test_unsafe_paths_and_nonfinite_json_rejected(tmp_path):
    for name in ('../escape', '/outside'):
        with pytest.raises(ValueError): plotter.relative_path(tmp_path, name)
    path = tmp_path / 'bad.json'; path.write_text('{"score": NaN}')
    with pytest.raises(ValueError): plotter.read_json(path)


@pytest.mark.parametrize('terminal', ['missing', 'run_failure', 'audit_failure', 'external_timeout', 'audit_source_drift'])
def test_missing_or_unsuccessful_original_closure_blocks_score_reads(tmp_path, monkeypatch, terminal):
    monkeypatch.setattr(plotter, 'ROOT', tmp_path)
    study = tmp_path / 'output/robot-history-initialization-study-v1'
    engineering = tmp_path / 'output/robot-history-initialization-engineering-v1'
    audit_path = tmp_path / 'audit/audit.json'
    registration = tmp_path / 'research/robot-history-initialization-registration.json'
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    write(registration, {'version': plotter.STUDY_VERSION})
    monkeypatch.setattr(plotter, 'PLAN_SHA', plotter.pin(registration)['sha256'])
    write(engineering / 'run-launch-01.json', {})
    write(engineering / 'audit-process-01.json', {'returncode': 1 if terminal == 'audit_failure' else 0,
                                                'source_unchanged': terminal != 'audit_source_drift'})
    write(audit_path, {})
    write(audit_path.parent / 'manifest.json', {})
    if terminal != 'missing':
        write(engineering / 'run-process-01.json', {'returncode': 1 if terminal == 'run_failure' else 0,
                                                   'external_timeout': terminal == 'external_timeout'})
    original = plotter.read_json
    def guarded(path):
        assert Path(path).name not in ('results.json', 'resources.json', 'fits.json', 'permutation.json'), 'unclosed scores read'
        return original(path)
    monkeypatch.setattr(plotter, 'read_json', guarded)
    with pytest.raises(ValueError):
        plotter.authenticate(study, audit_path, engineering)
