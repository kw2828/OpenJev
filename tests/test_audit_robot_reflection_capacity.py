"""Independent fabricated audit fixtures; no campaign arrays, models or fits."""
import copy
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_robot_reflection_capacity as audit

ARMS = ('householder12', 'householder', 'dense_bounded', 'dense_unbounded',
        'dense_mlp', 'gru32', 'legacy_instant')
REFS = ('causal_ridge_1', 'causal_ridge_100', 'linear_frozen', 'persistence')
SEEDS, RATES, DEV = (8101, 8102, 8103), (.001, .003), ('fabricated-a', 'fabricated-b')
PARAMETERS = dict(zip(ARMS, (806, 630, 806, 806, 590, 5916, 1014), strict=True))


def metric(value, horizon):
    return {'standardized_rmse': value, 'standardized_sse': value**2 * 120,
            'scalars': 120, 'physical_rmse_deg': value,
            'per_joint_rmse_deg': [value] * 6, 'windows': 1, 'horizon': horizon}


def scalar_fixture():
    cfg = {'partitions': {'dev': list(DEV)}, 'seeds': list(SEEDS),
           'learning_rates': list(RATES), 'horizons': [64, 128],
           'r4_mean_ratio': .95, 'r4_paired_ratio': 1.,
           'mean_ratio': 1.02, 'paired_ratio': 1.05, 'latency_ratio': 1.05}
    rows = []
    for recording in DEV:
        for arm in ARMS:
            for seed in SEEDS:
                for rate in RATES:
                    for horizon in (64, 128):
                        rows.append({'recording': recording, 'arm': arm, 'seed': seed,
                                     'learning_rate': rate, 'horizon': horizon, 'status': 'PASS',
                                     'metrics': metric(.8 if arm == 'householder12' else 1., horizon),
                                     'error': None})
        for arm in REFS:
            for horizon in (64, 128):
                rows.append({'recording': recording, 'arm': arm, 'seed': None,
                             'learning_rate': None, 'horizon': horizon, 'status': 'PASS',
                             'metrics': metric(1., horizon), 'error': None})
    return cfg, rows


def resource_fields(arm):
    neural = arm in ARMS
    count = PARAMETERS[arm] if neural else 445440 if arm in REFS[:2] else 150 if arm == 'linear_frozen' else 0
    state = (50 if arm == 'gru32' else 12) if neural else 192 if arm in REFS[:2] else 18 if arm == 'linear_frozen' else 6
    size = 4 if neural else 8
    item = {'parameters': count, 'parameter_bytes': count * size,
            'state_scalars': state, 'state_bytes': state * size,
            'normalizer_bytes': 192, 'buffer_bytes': 16 if arm in REFS[:2] else 0,
            'dtype': 'float32' if neural else 'float64', 'input_bytes': 9216,
            'output_bytes': 6144, 'temporary_workspace': 'not measured'}
    if neural:
        item['inactive_parameters'] = 192 if arm == 'legacy_instant' else 0
    return item


def resource_fixture():
    fits, resources = [], []
    # Producer order: six fresh fits, then its unchanged original parent order.
    order = [('householder12', seed, rate) for seed in SEEDS for rate in RATES]
    order += [(arm, seed, rate) for seed in SEEDS for rate in RATES for arm in ARMS[1:]]
    for arm, seed, rate in order:
        fields = resource_fields(arm)
        fits.append({'arm': arm, 'seed': seed, 'learning_rate': rate, 'resources': fields})
        if rate == .001:
            scale = 1.05 if arm == 'householder12' else 1.
            values = [scale * (.81 + .02 * i) for i in range(20)]
            resources.append({'arm': arm, 'seed': seed, 'learning_rate': rate, **fields,
                              'timing': {'seconds': values, 'median_seconds': scale,
                                         'p95_seconds': scale * 1.171}})
    for arm in REFS:
        resources.append({'arm': arm, 'seed': None, **resource_fields(arm),
                          'timing': {'seconds': [.1] * 20, 'median_seconds': .1, 'p95_seconds': .1}})
    return fits, resources


def condition_map(result):
    return {row['name']: row['passed'] for row in result['conditions']}


def test_closed_form_scoring_has_independent_horizon_and_physical_scale():
    target = np.zeros((2, 128, 6), dtype=np.float64)
    prediction = np.ones_like(target)
    prediction[:, 64:] = 3.
    scales = np.arange(1., 7.)
    short = audit.scored(prediction, target, scales, 64, np)
    long = audit.scored(prediction, target, scales, 128, np)
    assert short['standardized_sse'] == 768. and short['standardized_rmse'] == 1.
    assert short['physical_rmse_deg'] == math.sqrt(91 / 6)
    assert short['per_joint_rmse_deg'] == scales.tolist()
    assert long['standardized_sse'] == 7680. and long['scalars'] == 1536
    assert long['standardized_rmse'] == math.sqrt(5)
    np.testing.assert_allclose(long['per_joint_rmse_deg'], scales * math.sqrt(5), rtol=1e-15)


@pytest.mark.parametrize('value,reason', [(float('nan'), 'nonfinite prediction/target'),
                                        (1e300, 'nonfinite metric arithmetic')])
def test_nonfinite_or_overflowed_forecast_stays_failed(value, reason):
    target = np.zeros((1, 128, 6))
    rows = audit.metric_rows({}, np.full_like(target, value), target, np.ones(6), None, np)
    assert [r['horizon'] for r in rows] == [64, 128]
    assert all(r['status'] == 'FAILED' and r['metrics'] is None
               and r['error'] == {'type': 'NonfiniteEvaluation', 'message': reason} for r in rows)


def test_independent_rule_complete_roster_and_all69_names():
    cfg, rows = scalar_fixture()
    _, resources = resource_fixture()
    selection, result = audit.decisions(rows, resources, cfg)
    assert len(rows) == 184 and len(resources) == 25
    assert set(selection['selected_rates']) == set(ARMS)
    assert set(selection['selected_rates'].values()) == {.001}
    assert selection['selected_ridge'] == 'causal_ridge_1'
    suffixes = {'mean_5pct/householder', *(f'seed{s}_no_harm/householder' for s in SEEDS),
                *(f'mean_within_2pct/{a}' for a in ARMS[2:]),
                *(f'seed{s}_within_5pct/{a}' for a in ARMS[2:] for s in SEEDS),
                'mean_5pct/linear_frozen', 'mean_5pct/persistence', 'within_5pct_causal_ridge',
                *(f'joint{j}_no_10pct_harm' for j in range(6))}
    expected = {'all_selected_families_and_causal_ridge_eligible',
                *(f'{dev}/{suffix}' for dev in DEV for suffix in suffixes),
                'at_most_105pct_dense_bounded_latency', 'at_most_105pct_dense_mlp_latency'}
    assert len(suffixes) == 33 and len(expected) == 69
    assert set(condition_map(result)) == expected and all(condition_map(result).values())
    assert result['accuracy'] == {'passed': 67, 'total': 67}
    assert result['compute'] == {'passed': 2, 'total': 2}
    assert result['passed'] == result['total'] == 69


@pytest.mark.parametrize('arm,expected', [('householder12', 0), ('householder', 60),
                                        ('legacy_instant', 60), ('gru32', 48)])
def test_failed_family_preserved_and_cannot_rescue_primary(arm, expected):
    cfg, rows = scalar_fixture()
    _, resources = resource_fixture()
    for row in rows:
        if row['arm'] == arm:
            row.update(status='FAILED', metrics=None, error={'type': 'FailedTrainingAttempt'})
    resources = [row for row in resources if row['arm'] != arm]
    selection, result = audit.decisions(rows, resources, cfg)
    assert selection['selected_rates'][arm] is None
    assert result['passed'] == expected and result['total'] == 69
    assert result['status'] == 'DO_NOT_ADVANCE_REFLECTION_CAPACITY'
    assert not condition_map(result)['all_selected_families_and_causal_ridge_eligible']


def test_one_failed_rate_selects_all_seeds_at_other_rate():
    cfg, rows = scalar_fixture()
    _, resources = resource_fixture()
    row = next(r for r in rows if r['arm'] == 'householder12' and r['learning_rate'] == .001 and r['horizon'] == 128)
    row.update(status='FAILED', metrics=None, error={'type': 'FailedTrainingAttempt'})
    for r in resources:
        if r['arm'] == 'householder12':
            r['learning_rate'] = .003
    selection, result = audit.decisions(rows, resources, cfg)
    assert selection['selected_rates']['householder12'] == .003
    assert not selection['options']['householder12'][0]['eligible']
    assert result['passed'] == 69


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'wrong_seed'])
def test_incomplete_metric_grid_cannot_hide_a_condition(damage):
    cfg, rows = scalar_fixture()
    _, resources = resource_fixture()
    if damage == 'missing':
        rows.pop()
    elif damage == 'duplicate':
        rows[-1] = copy.deepcopy(rows[0])
    else:
        rows[0]['seed'] = 999
    with pytest.raises((ValueError, KeyError)):
        audit.decisions(rows, resources, cfg)


def test_saved_omitted_gate_or_changed_boolean_is_not_scalar_roundoff():
    cfg, rows = scalar_fixture()
    _, resources = resource_fixture()
    _, actual = audit.decisions(rows, resources, cfg)
    changed = copy.deepcopy(actual)
    changed['conditions'].pop()
    with pytest.raises(ValueError):
        audit.close(actual, changed, 'saved gate roster')
    changed = copy.deepcopy(actual)
    changed['conditions'][0]['passed'] = False
    with pytest.raises(ValueError):
        audit.close(actual, changed, 'saved gate identity')


def test_resource_accounting_and_raw_samples_are_reconciled():
    cfg, rows = scalar_fixture()
    fits, resources = resource_fixture()
    selection, _ = audit.decisions(rows, resources, cfg)
    audit.validate_resources(resources, fits, selection, rows, {}, np)


@pytest.mark.parametrize('damage', ['sample', 'median', 'p95', 'missing', 'nan', 'zero',
                                    'count', 'bytes', 'rate', 'seed', 'inactive'])
def test_timing_and_storage_tampering_rejected(damage):
    cfg, rows = scalar_fixture()
    fits, resources = resource_fixture()
    selection, _ = audit.decisions(rows, resources, cfg)
    row = resources[0]
    if damage == 'sample':
        row['timing']['seconds'][10] += .01
    elif damage in ('median', 'p95'):
        row['timing'][damage + '_seconds'] += .01
    elif damage == 'missing':
        row['timing']['seconds'].pop()
    elif damage in ('nan', 'zero'):
        row['timing']['seconds'][0] = float('nan') if damage == 'nan' else 0.
    elif damage == 'count':
        resources.pop()
    elif damage == 'bytes':
        row['parameter_bytes'] -= 4
    elif damage == 'rate':
        row['learning_rate'] = .003
    elif damage == 'seed':
        row['seed'] = 999
    else:
        row['inactive_parameters'] = 192
    with pytest.raises(ValueError):
        audit.validate_resources(resources, fits, selection, rows, {}, np)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_saved_scalar_is_never_equal(value):
    with pytest.raises(ValueError):
        audit.close(value, value, 'nonfinite')


def fit_fixture(status='PASS', updates=4096):
    trace = [{'update': i + 1, 'loss': .5, 'gradient_norm_before_clip': 2.} for i in range(updates)]
    receipt = {'status': status, 'error': None if status == 'PASS' else {
        'type': 'FitFailure', 'message': 'nonfinite training gradient norm'},
        'completed_updates': updates, 'requested_updates': 4096, 'learning_rate': .001,
        'optimizer_seconds': .5, 'fit_seconds': 1.,
        'fit_cap_scope': 'construction,optimizer setup,loop and checkpoint preservation through trace;receipt serialization follows',
        'timing_scope': 'optimizer loop including batch construction and finite checks, excluding model construction and saved files',
        'files': {name: {'bytes': 1, 'sha256': '0' * 64} for name in
                  ('initial.npz', 'final.npz', 'optimizer.npz', 'trace.json')}}
    return receipt, trace


def test_complete_fit_and_honest_failed_partial_trace_are_both_auditable():
    for status, updates in [('PASS', 4096), ('FAILED', 0), ('FAILED', 3), ('FAILED', 4096)]:
        receipt, trace = fit_fixture(status, updates)
        audit.validate_fit_receipt(receipt, trace, .001, 1800.)
        assert receipt['status'] == status and receipt['completed_updates'] == updates


@pytest.mark.parametrize('damage', ['short_pass', 'trace_gap', 'nan_loss', 'negative_gradient',
                                    'boolean_update', 'bad_rate', 'missing_evidence', 'bad_timing',
                                    'cap', 'erased_failure', 'programming_failure'])
def test_fit_ledger_never_promotes_partial_or_corrupt_attempt(damage):
    receipt, trace = fit_fixture()
    if damage == 'short_pass':
        trace.pop(); receipt['completed_updates'] -= 1
    elif damage == 'trace_gap':
        trace[1]['update'] = 3
    elif damage == 'nan_loss':
        trace[0]['loss'] = float('nan')
    elif damage == 'negative_gradient':
        trace[0]['gradient_norm_before_clip'] = -1.
    elif damage == 'boolean_update':
        trace[0]['update'] = True
    elif damage == 'bad_rate':
        receipt['learning_rate'] = .003
    elif damage == 'missing_evidence':
        del receipt['files']['optimizer.npz']
    elif damage == 'bad_timing':
        receipt['optimizer_seconds'] = 2.
    elif damage == 'cap':
        receipt['fit_seconds'] = 1800.01
    else:
        receipt['status'] = 'FAILED'
        receipt['error'] = None if damage == 'erased_failure' else {'type': 'RuntimeError', 'message': 'schema bug'}
    with pytest.raises(ValueError):
        audit.validate_fit_receipt(receipt, trace, .001, 1800.)


def metadata_fixture(tmp_path, monkeypatch):
    """New admission runs for real; only the already-qualified parent is a stub."""
    import audit_robot_structured
    root = tmp_path / 'repository'
    study = root / 'output/robot-reflection-capacity-study-v1'
    engineering = root / 'output/robot-reflection-capacity-engineering-v1'
    parent = root / 'output/robot-structured-study-v1'
    parent_engineering = root / 'output/robot-structured-engineering-v1'
    registration = root / 'research/robot-reflection-capacity-registration.json'
    monkeypatch.setattr(audit, 'ROOT', root)
    monkeypatch.setattr(audit, 'COMMIT', 'fabricated-prefit-commit')
    for key in audit.THREADS:
        monkeypatch.setenv(key, '1')

    def raw(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return audit.pin(path)

    def write(path, value):
        return raw(path, (json.dumps(value, sort_keys=True) + '\n').encode())

    sources = {}
    for name in audit.SOURCES:
        content = ('# fabricated source: ' + name + '\n').encode()
        raw(root / name, content)
        raw(study / 'sources' / name, content)
        sources[name] = audit.descriptor(root / name)
    cfg, _ = scalar_fixture()
    cfg['partitions']['fit'] = [f'fit-{i}' for i in range(7)]
    prior_cfg = dict(cfg, storage_ratio=.8)
    data = {f'{split}-data-{name}.npz': raw(root / 'fabricated-data' / f'{split}-{name}.npz', b'opaque, not NPZ')
            for split in ('fit', 'dev') for name in cfg['partitions'][split]}
    data.update({name: raw(root / 'fabricated-data' / name, b'opaque reference')
                 for name in ('normalizers.npz', 'references.npz')})
    prior = {'config': prior_cfg, 'sources': {name: sources[name] for name in audit.SOURCES[:21]}, 'data': data}
    parent_reg = root / 'research/robot-structured-registration.json'
    write(parent_reg, prior)
    parent_sha = audit.descriptor(parent_reg)['sha256']
    monkeypatch.setattr(audit, 'PARENT_SHA', parent_sha)
    prior_inputs = {'parent_data': data}
    parent_calls = []

    def authenticate_parent(folder, process):
        assert folder == parent and process == parent_engineering / 'run-process-01.json'
        parent_calls.append((folder, process))
        return prior, prior_inputs

    monkeypatch.setattr(audit_robot_structured, 'authenticate', authenticate_parent)
    payloads = {}
    for name in audit.PAYLOADS:
        content = ('opaque parent payload: ' + name).encode()
        payloads[name] = raw(parent / name, content)
        copied = 'parent-normalizers.npz' if name == 'normalizers.npz' else 'parent-fits.json' if name == 'fits.json' else name
        raw(study / copied, content)
    closure = {
        'manifest': write(parent / 'manifest.json', {'files': {name: audit.descriptor(item['path']) for name, item in payloads.items()}}),
        'receipt': write(parent / 'receipt.json', {'status': 'PASS', 'registration_sha256': parent_sha}),
        'process': write(parent_engineering / 'run-process-01.json', {'returncode': 0, 'registration_sha256': parent_sha}),
    }
    closure['audit'] = write(root / 'output/robot-structured-audit-v1/audit.json', {
        'status': 'PASS', 'agreement': True, 'registration_sha256': parent_sha,
        'inputs': prior_inputs, 'source_pins': prior['sources'],
        'auditor': audit.pin(root / 'scripts/audit_robot_structured.py')})
    closure['audit_process'] = write(parent_engineering / 'audit-process-01.json', {
        'returncode': 0, 'audit_output': audit.descriptor(closure['audit']['path'])})
    launcher = root / 'scripts/launch_robot_reflection_capacity.py'
    raw(launcher, b'# fabricated launcher\n')
    launcher_pin = audit.descriptor(launcher)
    commands = [
        ['.venv/bin/ruff', 'check', 'src/openjev/research/reflection_capacity.py', 'tests/test_reflection_capacity.py',
         'scripts/robot_reflection_capacity_study.py', 'tests/test_robot_reflection_capacity_study.py',
         'scripts/launch_robot_reflection_capacity.py'],
        ['.venv/bin/python', '-m', 'pytest', '-q', 'tests/test_reflection_capacity.py',
         'tests/test_robot_reflection_capacity_study.py'],
    ]
    qualification_rows = []
    for i, command in enumerate(commands, 1):
        log = engineering / f'qualification-01-command-{i}.log'
        raw(log, b'fabricated fixture, not an execution log\n')
        qualification_rows.append({'command': command, 'returncode': 0, 'seconds': .1,
                                   'log': str(log), 'sha256': audit.descriptor(log)['sha256']})
    qualification = write(engineering / 'qualification-01.json', {
        'status': 'PASS', 'sources': sources, 'launcher': launcher_pin, 'sources_unchanged': True,
        'thread_env': dict.fromkeys(audit.THREADS, '1'), 'commands': qualification_rows,
        'created_utc': '2026-01-01T00:00:00+00:00'})
    cfg.update(version='robot-reflection-capacity-study-v1', arms=['householder12'], comparison_arms=list(ARMS),
               fit_cap_seconds=1800., wall_cap_seconds=10800., cached_parent_refit=False,
               reflections=12, parameters=806, state_scalars=12)
    plan = {'version': 'robot-reflection-capacity-study-v1', 'config': cfg, 'sources': sources,
            'parent_registration_sha256': parent_sha, 'data': data, 'parent_closure': closure,
            'parent_payloads': payloads, 'qualification': qualification, 'launcher': launcher_pin,
            'created_utc': '2026-01-01T00:01:00+00:00'}
    clock = 'mach_continuous_time' if sys.platform == 'darwin' else 'CLOCK_BOOTTIME'
    write(study / 'runtime.json', {'python': sys.version, 'platform': audit.platform.platform(),
          'machine': audit.platform.machine(), 'thread_env': dict.fromkeys(audit.THREADS, '1'),
          'torch_threads': 1, 'numpy': importlib.metadata.version('numpy'),
          'torch': importlib.metadata.version('torch'), 'clock': clock})
    process_path = engineering / 'run-process-01.json'
    raw(process_path.with_suffix('.log'), b'fabricated run fixture\n')

    def seal():
        # Re-pin intentional fixture edits so each corruption reaches its own join.
        write(registration, plan)
        raw(study / 'registration.json', registration.read_bytes())
        sha = audit.descriptor(registration)['sha256']
        monkeypatch.setattr(audit, 'PLAN_SHA', sha)
        launch = {'command': list(audit.COMMAND), 'prefit_commit': audit.COMMIT,
                  'started_utc': '2026-01-01T00:02:00+00:00', 'registration_sha256': sha,
                  'launcher': launcher_pin, 'thread_env': dict.fromkeys(audit.THREADS, '1'),
                  'scope': 'fabricated admission fixture only'}
        write(engineering / 'run-launch-01.json', launch)
        write(process_path, {**launch, 'returncode': 0, 'elapsed_seconds': 2., 'external_timeout': False,
                             'log': audit.descriptor(process_path.with_suffix('.log'))})
        write(study / 'receipt.json', {'status': 'PASS', 'registration_sha256': sha,
              'fits': 42, 'fresh_fit_attempts': 6, 'cached_fit_records': 36, 'rows': 184,
              'raw_decodes': 0, 'reference_refits': 0, 'legacy_refits': 0, 'parent_refits': 0,
              'saved_fit_loads': 7, 'saved_dev_loads': 2, 'confirmation_access': False,
              'official_test_access': False, 'seconds': 1., 'monotonic_seconds': 1., 'clock': clock})
        write(study / 'manifest.json', {'files': {str(p.relative_to(study)): audit.descriptor(p)
              for p in study.rglob('*') if p.is_file() and p.name not in ('manifest.json', 'receipt.json')}})

    seal()
    def forbid(*args, **kwargs):
        pytest.fail('metadata admission attempted an array decode')
    monkeypatch.setattr(np, 'load', forbid)
    return {'root': root, 'study': study, 'process': process_path, 'plan': plan,
            'prior_inputs': prior_inputs, 'parent_calls': parent_calls, 'write': write, 'seal': seal}


def test_complete190_payload_admission_is_opaque_and_does_not_decode(tmp_path, monkeypatch):
    case = metadata_fixture(tmp_path, monkeypatch)
    plan, inputs = audit.authenticate(case['study'], case['process'])
    assert plan == case['plan'] and len(plan['sources']) == 29
    assert len(plan['parent_payloads']) == 190 and len(plan['data']) == 11
    assert inputs['parent_payloads'] == plan['parent_payloads']
    assert inputs['run_receipt'] == audit.pin(case['process'])
    assert len(case['parent_calls']) == 1


@pytest.mark.parametrize('damage', ['source', 'snapshot', 'missing_payload', 'changed_parent_payload',
                                    'changed_copied_payload', 'parent_process', 'parent_audit_join',
                                    'parent_audit_output', 'run_command', 'run_commit', 'run_returncode',
                                    'run_timeout', 'run_log', 'extra_file', 'manifest_hash',
                                    'qualification_log', 'qualification_command'])
def test_provenance_damage_stops_before_any_array_decode(tmp_path, monkeypatch, damage):
    case = metadata_fixture(tmp_path, monkeypatch)
    root, study, plan, process = (case[k] for k in ('root', 'study', 'plan', 'process'))
    write, seal = case['write'], case['seal']
    key = 'householder-8101-lr0/final.npz'
    if damage in ('source', 'snapshot'):
        base = root if damage == 'source' else study / 'sources'
        (base / audit.SOURCES[-1]).write_bytes(b'changed frozen source')
    elif damage == 'missing_payload':
        del plan['parent_payloads'][key]
        seal()
    elif damage == 'changed_parent_payload':
        Path(plan['parent_payloads'][key]['path']).write_bytes(b'changed parent checkpoint bytes')
    elif damage == 'changed_copied_payload':
        (study / key).write_bytes(b'changed copied checkpoint bytes')
        seal()
    elif damage.startswith('parent_'):
        closure_key = 'process' if damage == 'parent_process' else 'audit' if damage == 'parent_audit_join' else 'audit_process'
        path = Path(plan['parent_closure'][closure_key]['path'])
        value = audit.read(path)
        if damage == 'parent_process':
            value['returncode'] = 7
        elif damage == 'parent_audit_join':
            value['inputs'] = {'parent_data': {}}
        else:
            value['audit_output']['sha256'] = '0' * 64
        plan['parent_closure'][closure_key] = write(path, value)
        seal()
    elif damage.startswith('run_'):
        if damage == 'run_log':
            process.with_suffix('.log').write_bytes(b'changed original log')
        else:
            value = audit.read(process)
            if damage == 'run_command':
                value['command'][-1] = 'different-output'
            elif damage == 'run_commit':
                value['prefit_commit'] = 'another-commit'
            elif damage == 'run_returncode':
                value['returncode'] = 9
            else:
                value['external_timeout'] = True
            write(process, value)
    elif damage == 'extra_file':
        (study / 'unlisted.txt').write_text('unlisted')
    elif damage == 'manifest_hash':
        path = study / 'manifest.json'
        value = audit.read(path)
        value['files']['runtime.json']['sha256'] = '0' * 64
        write(path, value)
    elif damage == 'qualification_log':
        qual = audit.read(plan['qualification']['path'])
        Path(qual['commands'][0]['log']).write_bytes(b'changed original qualification log')
    else:
        path = Path(plan['qualification']['path'])
        value = audit.read(path)
        value['commands'][1]['command'].pop()
        plan['qualification'] = write(path, value)
        seal()
    with pytest.raises(ValueError):
        audit.authenticate(study, process)
