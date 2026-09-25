"""Fabricated scalar/metadata qualification only; no measured arrays or fits."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import robot_reflection_capacity_study as study


def metric(value, horizon):
    return {'standardized_rmse': value, 'standardized_sse': value**2*120,
            'scalars': 120, 'physical_rmse_deg': value, 'per_joint_rmse_deg': [value]*6,
            'windows': 1, 'horizon': horizon}


def fixture_rows():
    cfg, rows = study.config(), []
    for recording in cfg['partitions']['dev']:
        for arm in study.ALL_ARMS:
            for seed in cfg['seeds']:
                for rate in cfg['learning_rates']:
                    for horizon in cfg['horizons']:
                        value = .8 if arm == 'householder12' else 1.
                        rows.append({'recording': recording, 'arm': arm, 'seed': seed,
                                     'learning_rate': rate, 'horizon': horizon, 'status': 'PASS',
                                     'metrics': metric(value, horizon), 'error': None})
        for arm in study.REFERENCES:
            for horizon in cfg['horizons']:
                rows.append({'recording': recording, 'arm': arm, 'seed': None,
                             'learning_rate': None, 'horizon': horizon, 'status': 'PASS',
                             'metrics': metric(1., horizon), 'error': None})
    return cfg, rows


def fixture_resources():
    counts = {'householder12': 806, 'householder': 630, 'dense_bounded': 806,
              'dense_unbounded': 806, 'dense_mlp': 590, 'gru32': 5916,
              'legacy_instant': 1014}
    rows = []
    for arm in study.ALL_ARMS:
        for seed in study.config()['seeds']:
            rows.append({'arm': arm, 'seed': seed, 'learning_rate': .001, 'parameters': counts[arm],
                         'parameter_bytes': counts[arm]*4, 'state_bytes': 200 if arm == 'gru32' else 48,
                         'normalizer_bytes': 192, 'buffer_bytes': 0,
                         'timing': {'median_seconds': 1.05 if arm == 'householder12' else 1.}})
    rows.extend({'arm': arm, 'seed': None, 'timing': {'median_seconds': .1}}
                for arm in study.REFERENCES)
    return rows


def rule(rows, cfg=None, resources=None):
    cfg = study.config() if cfg is None else cfg
    return study.evaluate_rule(rows, study.select(rows, cfg),
                               fixture_resources() if resources is None else resources, cfg)


def conditions(result):
    return {item['name']: item['passed'] for item in result['conditions']}


def test_complete_rosters_and_paired_selection():
    cfg, rows = fixture_rows()
    assert tuple(study.ARMS) == ('householder12',)
    assert len(study.ALL_ARMS) == 7 and len(study.PAYLOADS) == 190
    assert len(cfg['seeds']) == 3 and cfg['seeds'] == [8101, 8102, 8103]
    assert cfg['learning_rates'] == [.001, .003]
    assert len(study.ALL_ARMS)*len(cfg['seeds'])*len(cfg['learning_rates']) == 42
    assert cfg['updates'] == 4096 and cfg['context'] == 32
    assert cfg['train_horizon'] == cfg['dev_horizon'] == 128
    assert len(rows) == 184 and len(fixture_resources()) == 25
    selected = study.select(rows, cfg)
    assert set(selected['selected_rates']) == set(study.ALL_ARMS)
    assert set(selected['selected_rates'].values()) == {.001}
    assert selected['selected_ridge'] == 'causal_ridge_1'
    actual = rule(rows, cfg)
    assert actual['passed'] == actual['total'] == 69
    assert len(conditions(actual)) == 69
    assert actual['accuracy'] == {'passed': 67, 'total': 67}
    assert actual['compute'] == {'passed': 2, 'total': 2}
    assert actual['status'] == 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL'


@pytest.mark.parametrize('arm', ['householder12', 'householder', 'dense_bounded',
                                'dense_unbounded', 'dense_mlp', 'gru32', 'legacy_instant'])
@pytest.mark.parametrize('damage', ['missing', 'failed', 'nonfinite'])
def test_every_family_must_have_one_complete_finite_recipe(arm, damage):
    cfg, rows = fixture_rows()
    chosen = [row for row in rows if row['arm'] == arm and row['horizon'] == 128]
    if damage == 'missing':
        rows = [row for row in rows if row not in chosen]
    elif damage == 'failed':
        for row in chosen:
            row.update(status='FAILED', metrics=None)
    else:
        for row in chosen:
            row['metrics']['standardized_rmse'] = float('nan')
    assert study.select(rows, cfg)['selected_rates'][arm] is None
    actual = rule(rows, cfg)
    assert not conditions(actual)['all_selected_families_and_causal_ridge_eligible']
    assert actual['passed'] < 69 and actual['status'] != 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL'


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'failed', 'nan', 'wrong_seed'])
def test_invalid_lower_rate_uses_only_other_complete_recipe(damage):
    cfg, rows = fixture_rows()
    index = next(i for i, row in enumerate(rows) if row['arm'] == 'householder12'
                 and row['horizon'] == 128 and row['learning_rate'] == .001)
    if damage == 'missing':
        rows.pop(index)
    elif damage == 'duplicate':
        rows.append(copy.deepcopy(rows[index]))
    elif damage == 'failed':
        rows[index].update(status='FAILED', metrics=None)
    elif damage == 'nan':
        rows[index]['metrics']['standardized_rmse'] = float('nan')
    else:
        rows[index]['seed'] = 100
    selected = study.select(rows, cfg)
    assert selected['selected_rates']['householder12'] == .003
    assert selected['options']['householder12'][0]['eligible'] is False


def test_selection_uses_pooled_h128_sse_not_mean_rmse_or_h64():
    cfg, rows = fixture_rows()
    lower = [row for row in rows if row['arm'] == 'householder12'
             and row['horizon'] == 128 and row['learning_rate'] == .001]
    for row in rows:
        if row['arm'] == 'householder12':
            row['metrics'] = metric(1., row['horizon'])
    for row in lower:
        row['metrics'] = metric(2., 128)
        row['metrics'].update(standardized_sse=4., scalars=1)
    lower[0]['metrics'].update(standardized_rmse=0., standardized_sse=0., scalars=10000)
    for row in rows:
        if row['arm'] == 'householder12' and row['horizon'] == 64 and row['learning_rate'] == .001:
            row.update(status='FAILED', metrics=None)
    assert study.select(rows, cfg)['selected_rates']['householder12'] == .001


def test_ridge_requires_distinct_complete_dev_roster():
    cfg, rows = fixture_rows()
    subset = [row for row in rows if row['arm'] == 'causal_ridge_1' and row['horizon'] == 128]
    subset[1]['recording'] = subset[0]['recording']
    assert study.select(rows, cfg)['selected_ridge'] == 'causal_ridge_100'
    for row in rows:
        if row['arm'] == 'causal_ridge_100':
            row.update(status='FAILED', metrics=None)
    assert not conditions(rule(rows, cfg))['all_selected_families_and_causal_ridge_eligible']


def test_inherited_window_torque_alignment_and_target_isolation():
    q = np.arange(210., dtype=np.float64).reshape(35, 6)
    u = 1000+q
    record = {'q': q.copy(), 'u': u.copy()}
    choices = {'record': np.array([0], np.int64), 'start': np.array([3], np.int64)}
    batch = study.old.window_batch([record], choices, 5, 7)
    assert np.array_equal(batch['q_context'][0], q[3:8])
    assert np.array_equal(batch['u_context'][0], u[3:8])
    assert np.array_equal(batch['future_u'][0], u[7:14])
    assert np.array_equal(batch['target'][0], q[8:15])
    record['q'][8:] = np.nan
    mutated = study.old.window_batch([record], choices, 5, 7)
    for name in ('q_context', 'u_context', 'future_u'):
        assert np.array_equal(batch[name], mutated[name])
    calls = []
    class Spy:
        def condition(self, context, torque):
            calls.append((context.numpy().copy(), torque.numpy().copy()))
            return context[:, -1]
        def __call__(self, future, state):
            calls.append(future.numpy().copy())
            return future.cumsum(1)+state[:, None], state
    first = study.old.infer(Spy(), batch)
    batch['target'] = object()
    assert np.array_equal(first.numpy(), study.old.infer(Spy(), batch).numpy())
    assert len(calls) == 4 and all(x.dtype == np.float32 for x in calls[0])


def test_inherited_batch_contract_exact_and_family_independent():
    cfg = study.config()
    local = dict(cfg, updates=3, batch_size=2, context=4, train_horizon=5, skip=2)
    lengths = [20, 23, 29]
    for seed in cfg['seeds']:
        expected_rng = np.random.Generator(np.random.PCG64(seed+520000))
        record = expected_rng.integers(0, 3, size=(3, 2), dtype=np.int64)
        start = np.empty_like(record)
        for index in np.ndindex(start.shape):
            start[index] = expected_rng.integers(2, lengths[record[index]]-4-5+1)
        for _arm in study.ALL_ARMS:
            actual = study.old.make_batches(lengths, seed, local)
            assert np.array_equal(actual['record'], record)
            assert np.array_equal(actual['start'], start)


def test_parent_npz_pin_precedes_decode_and_preserves_layout(tmp_path, monkeypatch):
    filename = 'linear.npz'
    path = tmp_path/filename
    coefficient = np.asfortranarray(np.arange(150.).reshape(6, 25))
    np.savez(path, coefficient=coefficient)
    item = {'path': str(path), **study.old.descriptor(path)}
    plan = {'parent_payloads': {filename: item}}
    actual = study.load_parent_arrays(plan, filename)['coefficient']
    assert actual.flags.f_contiguous and actual.flags.owndata
    assert np.array_equal(actual, coefficient)
    item['sha256'] = '0'*64
    def forbidden(*args, **kwargs):
        pytest.fail('array decoding preceded opaque pin validation')
    monkeypatch.setattr(study.np, 'load', forbidden)
    with pytest.raises(ValueError):
        study.load_parent_arrays(plan, filename)


def test_parent_copy_is_exact_exclusive_and_pin_checked(tmp_path):
    path = tmp_path/'original.json'; path.write_bytes(b'{"not":"a model"}\n')
    plan = {'parent_payloads': {'fits.json': {'path': str(path), **study.old.descriptor(path)}}}
    copied = tmp_path/'child'/'fits.json'
    study.copy_parent(plan, 'fits.json', copied)
    assert copied.read_bytes() == path.read_bytes()
    with pytest.raises(FileExistsError):
        study.copy_parent(plan, 'fits.json', copied)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):
        study.copy_parent(plan, 'fits.json', tmp_path/'different.json')


OTHER_CONTROLS = ('dense_bounded', 'dense_unbounded', 'dense_mlp', 'gru32', 'legacy_instant')
ACCURACY_SUFFIXES = (
    'mean_5pct/householder',
    *(f'seed{seed}_no_harm/householder' for seed in (8101, 8102, 8103)),
    *(suffix for arm in OTHER_CONTROLS for suffix in
      (f'mean_within_2pct/{arm}', *(f'seed{seed}_within_5pct/{arm}' for seed in (8101, 8102, 8103)))),
    'mean_5pct/linear_frozen', 'mean_5pct/persistence', 'within_5pct_causal_ridge',
    *(f'joint{joint}_no_10pct_harm' for joint in range(6)),
)


@pytest.mark.parametrize('recording_index', [0, 1])
@pytest.mark.parametrize('suffix', ACCURACY_SUFFIXES)
def test_every_registered_accuracy_condition_is_sensitive(recording_index, suffix):
    cfg, rows = fixture_rows()
    recording = cfg['partitions']['dev'][recording_index]
    if suffix.startswith('joint'):
        joint = int(suffix[5])
        for row in rows:
            if row['arm'] == 'householder12' and row['recording'] == recording and row['horizon'] == 128:
                row['metrics']['per_joint_rmse_deg'][joint] = 1.101
    else:
        target_arm = suffix.rsplit('/', 1)[-1]
        targets = set(study.REFERENCES[:2]) if suffix == 'within_5pct_causal_ridge' else {target_arm}
        seed = int(suffix.split('_', 1)[0][4:]) if suffix.startswith('seed') else None
        for row in rows:
            if row['recording'] == recording and row['horizon'] == 128 and row['arm'] in targets and (seed is None or row['seed'] == seed):
                row['metrics'] = metric(.7, 128)
    actual = rule(rows, cfg)
    assert not conditions(actual)[recording+'/'+suffix]
    assert actual['accuracy']['passed'] < 67 and actual['compute']['passed'] == 2
    assert actual['status'] == 'DO_NOT_ADVANCE_REFLECTION_CAPACITY'
    other = cfg['partitions']['dev'][1-recording_index]
    assert all(conditions(actual)[other+'/'+key] for key in ACCURACY_SUFFIXES)


def test_exact_independent_condition_name_roster():
    cfg, rows = fixture_rows()
    expected = {'all_selected_families_and_causal_ridge_eligible'}
    expected.update(recording+'/'+suffix for recording in cfg['partitions']['dev'] for suffix in ACCURACY_SUFFIXES)
    expected.update('at_most_105pct_'+arm+'_latency' for arm in ('dense_bounded', 'dense_mlp'))
    assert len(ACCURACY_SUFFIXES) == 33 and len(expected) == 69
    assert set(conditions(rule(rows, cfg))) == expected


@pytest.mark.parametrize('arm', ['dense_bounded', 'dense_mlp'])
@pytest.mark.parametrize('damage', ['slower', 'missing', 'duplicate', 'wrong_rate', 'zero', 'negative', 'nan', 'infinite'])
def test_each_compute_gate_is_mandatory_complete_positive_and_finite(arm, damage):
    cfg, rows = fixture_rows()
    resources = fixture_resources()
    subset = [row for row in resources if row['arm'] == arm]
    if damage == 'slower':
        for row in subset:
            row['timing']['median_seconds'] = .999
    elif damage == 'missing':
        resources.remove(subset[0])
    elif damage == 'duplicate':
        subset[0]['seed'] = subset[1]['seed']
    elif damage == 'wrong_rate':
        subset[0]['learning_rate'] = .003
    else:
        subset[0]['timing']['median_seconds'] = {'zero': 0., 'negative': -.1, 'nan': float('nan'), 'infinite': float('inf')}[damage]
    actual = rule(rows, cfg, resources)
    assert not conditions(actual)['at_most_105pct_'+arm+'_latency']
    assert actual['accuracy'] == {'passed': 67, 'total': 67}
    assert actual['compute'] == {'passed': 1, 'total': 2}
    assert actual['status'] == 'DO_NOT_ADVANCE_REFLECTION_CAPACITY'


def test_utility_uses_median_of_fit_medians_and_has_no_storage_promotion():
    cfg, rows = fixture_rows()
    resources = fixture_resources()
    candidate = [row for row in resources if row['arm'] == 'householder12']
    for row, value in zip(candidate, [.2, 1.05, 9.], strict=True):
        row['timing']['median_seconds'] = value
        row['parameter_bytes'] = 10**8
    actual = rule(rows, cfg, resources)
    assert actual['passed'] == 69 and not any('storage' in key for key in conditions(actual))


@pytest.mark.parametrize('arm', ['causal_ridge_1', 'causal_ridge_100', 'linear_frozen', 'persistence'])
@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'nonfinite'])
def test_complete_frozen_reference_roster_cannot_be_bypassed(arm, damage):
    cfg, rows = fixture_rows()
    row = next(r for r in rows if r['arm'] == arm and r['horizon'] == 128)
    if damage == 'missing':
        rows.remove(row)
    elif damage == 'duplicate':
        rows.append(copy.deepcopy(row))
    else:
        row['metrics']['per_joint_rmse_deg'][0] = float('inf')
    actual = rule(rows, cfg)
    if damage == 'nonfinite' and arm in study.REFERENCES[:2]:
        assert study.select(rows, cfg)['selected_ridge'] != arm
        assert actual['passed'] == 69
    elif damage == 'nonfinite':
        recording = row['recording']
        assert not conditions(actual)[recording+'/mean_5pct/'+arm]
        assert actual['status'] == 'DO_NOT_ADVANCE_REFLECTION_CAPACITY'
    else:
        assert not conditions(actual)['all_selected_families_and_causal_ridge_eligible']


def test_caller_cannot_forge_selection_to_hide_an_ineligible_recipe():
    cfg, rows = fixture_rows()
    selected = study.select(rows, cfg)
    selected['selected_rates']['householder12'] = .003
    with pytest.raises(ValueError, match='selection'):
        study.evaluate_rule(rows, selected, fixture_resources(), cfg)


def test_native_deadline_equality_and_preserved_original_receipt():
    class Clock:
        def __init__(self, value): self.value = value
        def now_ns(self): return self.value
    study.check_deadline(Clock(1_999_999_999), 0, 2., whole=False)
    with pytest.raises(study.old.FitFailure):
        study.check_deadline(Clock(2_000_000_000), 0, 2., whole=False)
    with pytest.raises(study.old.WholeStudyTimeout):
        study.check_deadline(Clock(2_000_000_001), 0, 2., whole=True)
    original = {'status': 'PASS', 'completed_updates': 4096}
    row = {'fit': original, 'effective_status': 'FAILED'}
    assert study.effective_status(row) == 'FAILED' and original['status'] == 'PASS'
    assert study.effective_status({'fit': original}) == 'PASS'


def fake_admission(tmp_path, monkeypatch):
    import plot_robot_structured
    for name in study.old.THREADS:
        monkeypatch.setenv(name, '1')
    root, parent = tmp_path/'repo', tmp_path/'parent'
    engineering = tmp_path/'engineering'
    root.mkdir(); parent.mkdir(); engineering.mkdir()
    monkeypatch.setattr(study, 'ROOT', root)
    monkeypatch.setattr(study, 'PARENT_FOLDER', parent)
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return {'path': str(path), **study.old.descriptor(path)}
    sources = {}
    for name in study.SOURCES:
        path = root/name
        write(path, {'fabricated_source': name})
        sources[name] = study.old.descriptor(path)
    write(root/study.PARENT_REGISTRATION, {'frozen_parent': True})
    payloads = {name: write(parent/name, {'fabricated_payload': name}) for name in study.PAYLOADS}
    descriptors = {name: {key: value[key] for key in ('bytes', 'sha256')} for name, value in payloads.items()}
    closure = {
        'manifest': write(parent/'manifest.json', {'files': descriptors}),
        'receipt': write(parent/'receipt.json', {'status': 'PASS'}),
        'process': write(engineering/'run-process-01.json', {'returncode': 0}),
        'audit': write(tmp_path/'audit.json', {'status': 'PASS', 'agreement': True}),
        'audit_process': write(engineering/'audit-process-01.json', {'returncode': 0}),
    }
    admission = {'plan': {'data': {'fabricated': 'metadata only'}}, 'engineering': str(engineering),
                 'audit_path': closure['audit']['path'],
                 'inputs': {item['path']: {key: item[key] for key in ('bytes', 'sha256')} for item in closure.values()}}
    monkeypatch.setattr(plot_robot_structured, 'authenticate', lambda folder: admission)
    log = tmp_path/'qualification.log'; log.write_text('fabricated qualification passed')
    qualification = write(tmp_path/'qualification.json', {'status': 'PASS', 'sources': sources,
        'commands': [{'returncode': 0, 'log': str(log), 'sha256': study.old.descriptor(log)['sha256']}]})
    launcher = root/'scripts/launch_robot_reflection_capacity.py'; launcher.write_text('# fabricated launcher')
    plan = {'version': study.VERSION, 'config': study.config(), 'sources': sources,
            'parent_registration_sha256': study.old.descriptor(root/study.PARENT_REGISTRATION)['sha256'],
            'data': admission['plan']['data'], 'parent_closure': closure, 'parent_payloads': payloads,
            'qualification': qualification, 'launcher': study.old.descriptor(launcher)}
    registration = tmp_path/'registration.json'; write(registration, plan)
    return registration, plan, admission


def test_complete_cached_roster_and_opaque_authentication_before_any_array(tmp_path, monkeypatch):
    registration, plan, _ = fake_admission(tmp_path, monkeypatch)
    expected = {f'{arm}-{seed}-lr{ri}/{name}' for arm in study.parent.ALL_ARMS
                for seed in (8101, 8102, 8103) for ri in range(2) for name in study.LEGACY_FILES}
    assert len(expected) == 180 and expected <= set(plan['parent_payloads'])
    assert len(plan['parent_payloads']) == 190
    monkeypatch.setattr(study.np, 'load', lambda *a, **k: pytest.fail('metadata admission decoded arrays'))
    actual, sha = study.authenticate(registration)
    assert actual == plan and sha == hashlib.sha256(registration.read_bytes()).hexdigest()


@pytest.mark.parametrize('damage', ['source', 'parent_sha', 'closure_join', 'payload_missing',
                                    'payload_bytes', 'qualification', 'qualification_log', 'launcher'])
def test_corrupt_provenance_stops_before_array_loading(tmp_path, monkeypatch, damage):
    registration, plan, admission = fake_admission(tmp_path, monkeypatch)
    if damage == 'source':
        (study.ROOT/study.SOURCES[-1]).write_text('changed')
    elif damage == 'parent_sha':
        plan['parent_registration_sha256'] = '0'*64
    elif damage == 'closure_join':
        admission['inputs'][plan['parent_closure']['process']['path']]['sha256'] = '0'*64
    elif damage == 'payload_missing':
        del plan['parent_payloads']['householder-8101-lr0/final.npz']
    elif damage == 'payload_bytes':
        Path(plan['parent_payloads']['householder-8101-lr0/final.npz']['path']).write_text('changed')
    elif damage == 'qualification':
        path = Path(plan['qualification']['path'])
        value = json.loads(path.read_text()); value['status'] = 'FAIL'; path.write_text(json.dumps(value))
        plan['qualification'].update(study.old.descriptor(path))
    elif damage == 'qualification_log':
        (tmp_path/'qualification.log').write_text('changed')
    else:
        (study.ROOT/'scripts/launch_robot_reflection_capacity.py').write_text('changed')
    registration.write_text(json.dumps(plan))
    monkeypatch.setattr(study.np, 'load', lambda *a, **k: pytest.fail('array decoded before failed admission'))
    with pytest.raises(ValueError):
        study.authenticate(registration)


def test_original_parent_closure_failure_prevents_output_and_array_access(tmp_path, monkeypatch):
    import plot_robot_structured
    registration, _plan, _admission = fake_admission(tmp_path, monkeypatch)
    error = ValueError('original parent closure incomplete')
    def reject(*args): raise error
    monkeypatch.setattr(plot_robot_structured, 'authenticate', reject)
    monkeypatch.setattr(study.np, 'load', lambda *a, **k: pytest.fail('array decoded before parent closure'))
    output = tmp_path/'new-run'
    with pytest.raises(ValueError) as caught:
        study.run(registration, output)
    assert caught.value is error and not output.exists()


def test_raw_initial_pairing_only_uses_exact_saved_common_parameters():
    class Value:
        def __init__(self, value): self.value = value
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value
    current = {'cell.input_matrix': np.zeros((2, 12, 6)), 'cell.reflection_raw': np.zeros((2, 12, 11))}
    original = {'cell.input_matrix': current['cell.input_matrix'].copy(),
                'cell.reflection_raw': current['cell.reflection_raw'][:, :4].copy()}
    class FakeModel:
        def state_dict(self): return {name: Value(value) for name, value in current.items()}
    pairing = study.initial_pairing(FakeModel(), original)
    assert pairing['first_four_reflections_equal'] and all(pairing['exact_common_parameters'].values())
    current['cell.reflection_raw'][:, 4:] = 99
    assert study.initial_pairing(FakeModel(), original)['first_four_reflections_equal']
    original['cell.input_matrix'][0, 0, 0] = 1
    with pytest.raises(ValueError, match='initialization differs'):
        study.initial_pairing(FakeModel(), original)


def test_inherited_resource_accounting_with_fake_payloads_only():
    class Payload:
        def __init__(self, size): self.size = size
        def numel(self): return self.size
        def element_size(self): return 4
    class FakeModel:
        def parameters(self): return iter([Payload(806)])
        def buffers(self): return iter([])
    item = study.resource_model(FakeModel(), 'householder12')
    assert item['parameters'] == 806 and item['parameter_bytes'] == 3224
    assert item['state_scalars'] == 12 and item['state_bytes'] == 48
    assert item['buffer_bytes'] == 0 and item['normalizer_bytes'] == 192
    assert sum(item[key] for key in ('parameter_bytes', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')) == 3464
    assert item['input_bytes'] == 9216 and item['output_bytes'] == 6144
    assert item['inactive_parameters'] == 0 and item['temporary_workspace'] == 'not measured'
