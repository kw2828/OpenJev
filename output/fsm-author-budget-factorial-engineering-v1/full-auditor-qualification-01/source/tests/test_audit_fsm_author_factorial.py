# SPDX-License-Identifier: GPL-3.0-or-later
"""Fabricated scalar, path and closure witnesses; no production imports."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/audit_fsm_author_factorial.py'
SPEC = importlib.util.spec_from_file_location('factorial_audit_fixture', SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def rows(value):
    values = [value]*12 if np.isscalar(value) else value
    return [{'record_id': rid, 'status': 'complete', 'rmse': float(v)}
            for rid, v in zip(audit.DEV, values, strict=True)]


def decision_fixture():
    refs = [{'family': f, 'seed': seed, 'rows': rows(.5 if f == audit.old.CANDIDATE else 5.)}
            for f, seed in audit.old.ORDERED]
    cells = {name: {'rows': rows(1.)} for name in audit.CELLS}
    return refs, rows(4.), cells


def test_equal_record_rmse_then_equal_seed_not_flat_squared_error():
    refs, bla, cells = decision_fixture()
    for entry in refs:
        if entry['family'] == audit.old.CANDIDATE:
            seed = entry['seed']-9200
            entry['rows'] = rows([seed/10]*6+[3*seed/10]*6)
    result = audit.decisions(refs, bla, cells, fit_complete=True, matrix_complete=True)
    assert result['family_means'][audit.old.CANDIDATE] == pytest.approx(.4)
    assert result['per_seed_means'][audit.old.CANDIDATE] == pytest.approx({'9201': .2, '9202': .4, '9203': .6})
    assert result['passed'] == 4 and result['strongest_control'] == 'new16'
    assert set(result['family_means']) == {*audit.old.FAMILIES, audit.old.BLA, 'new16', 'new64'}
    assert 'old16' not in result['family_means'] and 'old64' not in result['family_means']


@pytest.mark.parametrize('kind', ['capped_fit', 'timing_failure', 'missing_new_record', 'missing_reference'])
def test_incomplete_cannot_promote_but_accurate_control_not_dropped(kind):
    refs, bla, cells = decision_fixture()
    fit_complete, matrix_complete = True, True
    cells['new64']['rows'] = rows(.3)
    if kind == 'capped_fit':
        fit_complete = False
    elif kind == 'timing_failure':
        matrix_complete = False
    elif kind == 'missing_new_record':
        cells['new16']['rows'][3] = {'record_id': audit.DEV[3], 'status': 'incomplete'}
    else:
        refs[0]['rows'][1] = {'record_id': audit.DEV[1], 'status': 'failed'}
    result = audit.decisions(refs, bla, cells, fit_complete=fit_complete, matrix_complete=matrix_complete)
    assert result['passed'] == 0 and result['reference_complete'] is False
    if kind != 'capped_fit':
        assert result['strongest_control'] == 'new64'
        assert result['family_means']['new64'] == pytest.approx(.3)
    else:
        assert 'new16' not in result['family_means'] and 'new64' not in result['family_means']


def test_five_percent_boundary_and_paired_seed_rule_are_distinct():
    refs, bla, cells = decision_fixture()
    for entry in refs:
        if entry['family'] == audit.old.CANDIDATE:
            entry['rows'] = rows(.95)
    result = audit.decisions(refs, bla, cells, fit_complete=True, matrix_complete=True)
    assert result['passed'] == 4
    for entry in refs:
        if entry['family'] == audit.old.CANDIDATE:
            entry['rows'] = rows(np.nextafter(.95, 1.))
    result = audit.decisions(refs, bla, cells, fit_complete=True, matrix_complete=True)
    assert result['conditions']['five_percent_below_strongest_control'] is False
    assert result['passed'] == 3
    # Make a stochastic reference strongest and one paired control stronger than candidate.
    refs, bla, cells = decision_fixture()
    family = audit.old.NEURAL[0]
    for entry in refs:
        if entry['family'] == family:
            entry['rows'] = rows(.2 if entry['seed'] == 9201 else 1.1)
    result = audit.decisions(refs, bla, cells, fit_complete=True, matrix_complete=True)
    assert result['strongest_control'] == family
    assert result['conditions']['five_percent_below_strongest_control']
    assert not result['conditions']['every_seed_below_strongest_control']


def test_single_record_harm_and_amplitude_mean_are_independent():
    refs, bla, cells = decision_fixture()
    for entry in refs:
        if entry['family'] == audit.old.CANDIDATE:
            entry['rows'] = rows([1.03]+[.5]*11)
    result = audit.decisions(refs, bla, cells, fit_complete=True, matrix_complete=True)
    assert result['passed'] == 3
    assert not result['conditions']['no_record_over_two_percent_strongest_control']
    assert result['conditions']['both_amplitudes_below_strongest_control']
    for entry in refs:
        if entry['family'] == audit.old.CANDIDATE:
            entry['rows'] = rows([1.]*6+[.1]*6)
    result = audit.decisions(refs, bla, cells, fit_complete=True, matrix_complete=True)
    assert result['passed'] == 3
    assert not result['conditions']['both_amplitudes_below_strongest_control']


def test_missing_duplicate_or_nonfinite_record_cannot_make_survivor_mean():
    with pytest.raises(ValueError, match='twelve'):
        audit.aggregate(rows(1.)[:-1])
    duplicate = rows(1.)
    duplicate[-1]['record_id'] = duplicate[0]['record_id']
    with pytest.raises(ValueError, match='twelve'):
        audit.aggregate(duplicate)
    invalid = rows(1.)
    invalid[-1]['rmse'] = np.inf
    assert audit.aggregate(invalid) is None


def test_four_contrasts_and_interaction_have_explicit_signs():
    result = audit.contrasts({'old16': 10., 'old64': 8., 'new16': 7., 'new64': 4.})
    effects = result['effects']
    assert [effects[k]['absolute_reduction'] for k in ('training_at16', 'training_at64', 'inference_old', 'inference_new')] == [3., 4., 2., 3.]
    assert effects['training_at16']['relative_reduction_percent'] == 30.
    assert effects['training_at64']['relative_reduction_percent'] == 50.
    assert result['interaction_absolute'] == 1.
    missing = audit.contrasts({'old16': 0., 'old64': 0., 'new16': 1., 'new64': None})
    assert missing['effects']['training_at16']['absolute_reduction'] == -1.
    assert missing['effects']['training_at16']['relative_reduction_percent'] is None
    assert missing['effects']['training_at64']['absolute_reduction'] is None
    assert missing['interaction_absolute'] is None


def timed(value):
    return {'status': 'complete', 'request_ms': value,
            'copy_ms': value/4, 'initializer_ms': value/2, 'rollout_ms': value/4}


def timing_fixture():
    warmups = [{'cell': cell, 'record_id': audit.DEV[0], 'start': 0, **timed(1.)} for cell in audit.CELLS]
    samples = []
    for slot in range(24):
        rid, start = audit.DEV[slot//2], (0, 7936)[slot % 2]
        for position in range(4):
            cell = audit.CELLS[(slot+position) % 4]
            samples.append({'slot': slot, 'position': position, 'cell': cell,
                'record_id': rid, 'start': start, **timed(float(slot+1))})
    return {'warmups': warmups, 'samples': samples,
            'scope': 'Fresh legal copies, normalization, seed, full GN/SVD and physical H128 forecast; disk/scoring excluded'}


def test_exact_balanced96_samples_and_median_of24_saved_durations():
    timing = timing_fixture()
    medians, complete = audit.timing_summary(timing)
    assert complete and medians == dict.fromkeys(audit.CELLS, 12.5)
    for cell in audit.CELLS:
        assert [sum(r['cell'] == cell and r['position'] == p for r in timing['samples']) for p in range(4)] == [6]*4


@pytest.mark.parametrize('mutation', ['missing', 'rotation', 'zero', 'nan', 'components', 'warmup_identity', 'warmup'])
def test_timing_tamper_rejected_or_explicitly_incomplete(mutation):
    timing = timing_fixture()
    if mutation == 'missing':
        timing['samples'].pop()
    elif mutation == 'rotation':
        timing['samples'][1]['position'] = 0
    elif mutation == 'zero':
        timing['samples'][0]['request_ms'] = 0.
    elif mutation == 'nan':
        timing['samples'][0]['request_ms'] = float('nan')
    elif mutation == 'components':
        timing['samples'][0]['initializer_ms'] += 1.
    elif mutation == 'warmup_identity':
        timing['warmups'][0]['start'] = 7936
    else:
        timing['warmups'][0] = {'cell': 'old16', 'record_id': audit.DEV[0], 'start': 0,
            'status': 'failed', 'error': {'type': 'FloatingPointError', 'message': 'known failure', 'traceback': 'fixture'}}
        medians, complete = audit.timing_summary(timing)
        assert not complete and medians['old16'] is None and medians['new64'] == 12.5
        return
    with pytest.raises(ValueError):
        audit.timing_summary(timing)


def pair_fixture(early=False):
    trace = [{'iteration': i, 'objective': float(32-i), 'trials': [{'alpha': 1., 'accepted': True, 'nonfinite': False}]} for i in range(2 if early else 16)]
    row = {'status': 'GRADIENT_TOL' if early else 'ITERATION_CAP', 'trace': trace,
           'directions_considered': len(trace), 'final_objective': 10.}
    short = {'policy': audit.replay.policy(iterations=16), 'linear_seed_states': [[1.]],
             'linear_seed_diagnostics': [{'rank': 1}], 'requests': [row]}
    long = copy.deepcopy(short)
    long['policy']['iterations'] = 64
    if not early:
        long['requests'][0]['trace'].append({'iteration': 16, 'objective': 10., 'trials': []})
        long['requests'][0]['directions_considered'] = 17
        long['requests'][0]['status'] = 'GRADIENT_TOL'
        long['requests'][0]['final_objective'] = 9.
    bank = {k: np.array([[1.]]) for k in ('prediction', 'final_state', 'forecast_state', 'solved_context_start_states')}
    return short, long, bank, copy.deepcopy(bank)


@pytest.mark.parametrize('early', [True, False])
def test_exact_same_seed_prefix_and_early_stop(early):
    assert audit.compare_budgets(*pair_fixture(early))


@pytest.mark.parametrize('mutation', ['seed', 'prefix', 'worsen', 'early_state', 'early_work'])
def test_budget_attribution_cannot_change_seed_path_or_early_stop(mutation):
    short, long, a, b = pair_fixture(mutation.startswith('early'))
    if mutation == 'seed':
        long['linear_seed_states'][0][0] += 1.
    elif mutation == 'prefix':
        long['requests'][0]['trace'][0]['trials'][0]['alpha'] = .5
    elif mutation == 'worsen':
        long['requests'][0]['final_objective'] = 11.
    elif mutation == 'early_state':
        b['forecast_state'][0, 0] += 1.
    else:
        long['requests'][0]['directions_considered'] += 1
    with pytest.raises(ValueError):
        audit.compare_budgets(short, long, a, b)


def test_standardized_hand_metric_and_score_overflow():
    prediction = np.broadcast_to(np.array([1., 2., 3.]), (2, 5, 3)).copy()
    result = audit.scored(prediction, np.zeros_like(prediction), np.array([2., 4., 8.]))
    assert result['mse'] == pytest.approx(14/3)
    assert result['per_channel_rmse'] == [1., 2., 3.]
    assert result['native_output_per_channel_rmse'] == [2., 8., 24.]
    assert result['requests'] == 2 and result['horizon'] == 5
    large = np.full((1, 2, 3), 1e308)
    with pytest.raises(ValueError, match='metric overflow'):
        audit.scored(large, -large, np.ones(3))
    with pytest.raises(ValueError, match='metric overflow'):
        audit.scored(np.full((1, 2, 3), 2.), np.zeros((1, 2, 3)), np.full(3, 1e308))


def test_failed_pair_and_exact_original_parity_are_visible_outcomes():
    short, _long, _a, b = pair_fixture()
    assert audit.pair_parity({'status': 'failed'}, {'status': 'complete'}, None, b) == {
        'status': 'unavailable', 'reason': 'numerical forecast failure'}
    assert audit.old_parity({'status': 'failed'}, None, None, None, None)['status'] == 'FAILED'
    bank = {k: np.ones((1, 1)) for k in ('prediction', 'final_state', 'forecast_state',
        'linear_seed_states', 'solved_context_start_states')}
    cached = {'starts': np.array([0], dtype=np.int64), 'y_context': np.zeros((1, 100, 3)),
              'u_context': np.zeros((1, 99, 3)), 'future_u': np.zeros((1, 128, 3))}
    saved = {**bank, **cached}
    row = {'status': 'complete', 'context': short}
    assert audit.old_parity(row, bank, row, saved, cached)['status'] == 'PASS'
    saved = copy.deepcopy(saved)
    saved['prediction'][0, 0] = np.nextafter(1., 2.)
    result = audit.old_parity(row, bank, row, saved, cached)
    assert result['status'] == 'FAILED' and result['checks']['prediction'] is False


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def process_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    study, folder = tmp_path/'study', tmp_path/'process'
    study.mkdir(); folder.mkdir()
    producer = tmp_path/'research/fsm_author/scripts/evaluate_nllfr_factorial.py'
    supervisor = tmp_path/'research/fsm_author/scripts/run_nllfr_factorial.py'
    for path in (producer, supervisor):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fabricated source\n')
    paths = {k+'_process': put(tmp_path/(k+'-fit.json'), {'closed': True}) for k in ('old', 'new')}
    plan = {'source_sha256': {str(producer.relative_to(tmp_path)): audit.pin(producer)['sha256']}}
    registration = put(tmp_path/'registration.json', plan)
    put(study/'started.json', {'registration_sha256': audit.pin(registration)['sha256'], 'pid': 123})
    copied = study/'source'/producer.relative_to(tmp_path)
    copied.parent.mkdir(parents=True, exist_ok=True)
    copied.write_bytes(producer.read_bytes())
    (study/'opaque.npz').write_bytes(b'not a numeric archive')
    (folder/'process.log').write_bytes(b'original fake log\n')
    command = [str(tmp_path/'research/fsm_author/.venv/bin/python'), str(producer),
               '--registration', str(registration), '--output', str(study)]
    launch = {'status': 'running', 'phase': 'evaluate', 'command': command,
        'registration_sha256': audit.pin(registration)['sha256'], 'started_time_ns': 1,
        'timeout_seconds': 3600, 'rss_cap_bytes': 32*1024**3, 'environment': audit.old.ENV,
        'producer': audit.descriptor(producer), 'supervisor': audit.descriptor(supervisor),
        'parent_fit_processes': {k: audit.descriptor(paths[k+'_process']) for k in ('old', 'new')}}
    put(folder/'launch.json', launch)
    terminal = {**launch, 'status': 'completed', 'observed_exit_code': 0, 'pid': 123,
        'end_identity_matches': True, 'outcome': 'ORIGINAL_PROCESS_COMPLETE',
        'elapsed_seconds': 1., 'wall_elapsed_seconds': 1.1, 'peak_polled_child_rss_bytes': 1234,
        'log_sha256': audit.pin(folder/'process.log')['sha256'], 'artifacts': audit.old.inventory(study)}
    process = put(folder/'process.json', terminal)
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('metadata must not decode arrays'))
    return plan, registration, study, process, paths, terminal


def test_closed_inventory_is_opaque_and_original_terminal_bound(tmp_path, monkeypatch):
    args = process_fixture(tmp_path, monkeypatch)
    terminal, files = audit.closed_process(*args[:5])
    assert terminal['status'] == 'completed' and 'opaque.npz' in files


@pytest.mark.parametrize('kind', ['timeout', 'memory_limit'])
def test_stop_race_exit_zero_stays_incomplete(tmp_path, monkeypatch, kind):
    *args, terminal = process_fixture(tmp_path, monkeypatch)
    terminal.update(status=kind, outcome='INCOMPLETE')
    put(args[3], terminal)
    actual, _ = audit.closed_process(*args)
    assert actual['observed_exit_code'] == 0 and actual['status'] == kind


@pytest.mark.parametrize('mutation', ['live', 'exit', 'command', 'parent', 'source', 'env',
                                      'cap', 'log', 'extra', 'missing', 'launch', 'identity'])
def test_original_closure_tamper_fails_before_decode(tmp_path, monkeypatch, mutation):
    *args, terminal = process_fixture(tmp_path, monkeypatch)
    plan, reg, study, process, paths = args
    if mutation == 'live':
        terminal['status'] = 'running'
    elif mutation == 'exit':
        terminal['observed_exit_code'] = 7
    elif mutation == 'command':
        terminal['command'] = ['echo', 'replacement']
    elif mutation == 'parent':
        terminal['parent_fit_processes']['new']['sha256'] = '0'*64
    elif mutation == 'source':
        Path(terminal['producer']['path']).write_text('# source drift')
    elif mutation == 'env':
        terminal['environment'] = {**terminal['environment'], 'OMP_NUM_THREADS': '2'}
    elif mutation == 'cap':
        terminal['timeout_seconds'] = 7200
    elif mutation == 'log':
        (process.parent/'process.log').write_text('changed log')
    elif mutation == 'extra':
        (study/'unlisted.json').write_text('{}')
    elif mutation == 'missing':
        (study/'opaque.npz').unlink()
    elif mutation == 'launch':
        launch = audit.read(process.parent/'launch.json')
        launch['started_time_ns'] = 2
        put(process.parent/'launch.json', launch)
    else:
        terminal['end_identity_matches'] = False
    put(process, terminal)
    with pytest.raises(ValueError):
        audit.closed_process(plan, reg, study, process, paths)


def test_admission_rejects_changed_source_before_parent_or_array_work(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    source = tmp_path/'fake.py'
    source.write_text('original')
    plan = {'version': audit.VERSION, 'experiment': audit.EXPERIMENT,
        'source_sha256': {'fake.py': audit.pin(source)['sha256']}, 'prerequisites': {},
        'output': 'study', 'process_directory': 'process'}
    reg = put(tmp_path/'registration.json', plan)
    monkeypatch.setattr(audit, 'MIN_SOURCES', {'fake.py'})
    monkeypatch.setattr(audit, 'REQUIRED', set())
    monkeypatch.setattr(audit, 'source_freeze', lambda _: ({}, reg))
    monkeypatch.setattr(audit, 'parent_admission', lambda _: pytest.fail('parents must not run'))
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('must not decode'))
    source.write_text('changed')
    with pytest.raises(ValueError, match='source drift'):
        audit.authenticate(tmp_path/'study', tmp_path/'process/process.json', tmp_path/'freeze.json')


def test_prefix_mismatch_rejects_before_any_cached_request_access(tmp_path, monkeypatch):
    old_inputs = {'attested': True}
    old_audit = {'study': str(tmp_path/'old'), 'inputs': old_inputs,
                 'results': {'fit_status': 'iteration_cap_reached'}}
    new_study = tmp_path/'new'
    new_study.mkdir()
    paths = {key: put(tmp_path/(key+'.json'), {}) for key in ('old_audit', 'old_audit_process',
        'old_process', 'old_registration', 'old_evaluation_process', 'new_audit', 'new_audit_process',
        'new_process', 'new_registration', 'new_audit_freeze')}
    terminal = {'status': 'completed', 'observed_exit_code': 0, 'end_identity_matches': True, 'artifacts': {}}
    put(paths['new_process'], terminal)
    new_inputs = {'registration': audit.descriptor(paths['new_registration']),
        'process': audit.descriptor(paths['new_process']), 'freeze': audit.descriptor(paths['new_audit_freeze']), 'files': {}}
    newer = {'study': str(new_study), 'inputs': new_inputs,
        'results': {'budget_only_attributable': False, 'prefix': {'exact': False}}}
    monkeypatch.setattr(audit, 'closed_audit', lambda path, *args: old_audit if path == paths['old_audit'] else newer)
    monkeypatch.setattr(audit.old, 'authenticate', lambda *a: ({}, old_inputs, {}, terminal, terminal))
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('no cache decoding'))
    with pytest.raises(ValueError, match='exact attributed budget fit before DEV'):
        audit.parent_admission(paths)


def test_failure_result_short_circuits_all_numerical_work(tmp_path, monkeypatch):
    inputs = {'opaque': 'same'}
    terminal = {'status': 'timeout'}
    monkeypatch.setattr(audit, 'authenticate', lambda *a: ({}, inputs, {}, terminal, {}, {}))
    monkeypatch.setattr(audit, 'load_arrays', lambda *a: pytest.fail('partial process must not decode arrays'))
    monkeypatch.setattr(audit.replay, 'replay_request', lambda *a, **k: pytest.fail('partial process must not replay'))
    result = audit.audit(tmp_path/'study', tmp_path/'process', tmp_path/'freeze')
    assert result['status'] == 'PASS' and result['agreement']
    assert result['scientific_status'] == 'REFERENCE_INCOMPLETE'
    assert result['results']['matrix_status'] == 'FACTORIAL_INCOMPLETE'
    assert result['counts']['request_replays'] == 0


def test_unknown_numeric_bank_cannot_be_silently_hidden_by_failure():
    wanted = np.array([1., np.inf, -np.inf, np.nan])
    audit.compare_numeric(wanted.copy(), wanted)
    changed = wanted.copy()
    changed[1] = -np.inf
    with pytest.raises(ValueError, match='infinity mask'):
        audit.compare_numeric(changed, wanted)
    changed = wanted.copy()
    changed[0] = 2.
    with pytest.raises(ValueError, match='array disagreement'):
        audit.compare_numeric(changed, wanted)


def identity_fixture():
    plan = {'source_sha256': {'declared.py': 'source'}, 'prerequisites': {'opaque': {'path': 'opaque.json', 'sha256': 'input'}}}
    preflight = {'python': '3.synthetic', 'executable': '/isolated/python', 'versions': {'numpy': 'synthetic'},
        'upstream_direct_url': {'url': 'file:///source'}, 'installed_source_matches': {'module.py': {'sha256': 'installed'}}}
    identity = {**plan, 'python': preflight['python'], 'executable': preflight['executable'],
        'versions': preflight['versions'], 'direct_url': preflight['upstream_direct_url'],
        'installed_author_sha256': {'module.py': 'installed'}, 'thread_environment': {**audit.old.ENV, 'XLA_FLAGS': None}}
    return identity, plan, preflight


def test_evaluator_runtime_identity_uses_actual_schema_not_fit_receipt():
    identity, plan, preflight = identity_fixture()
    assert not {'status', 'sources', 'environment', 'installed_source_matches'} & identity.keys()
    audit.evaluator_identity(identity, copy.deepcopy(identity), plan, preflight)


@pytest.mark.parametrize('key', ['source_sha256', 'prerequisites', 'python', 'executable', 'versions',
                               'direct_url', 'installed_author_sha256', 'thread_environment'])
def test_evaluator_runtime_identity_requires_all_source_runtime_and_environment_joins(key):
    identity, plan, preflight = identity_fixture()
    changed = copy.deepcopy(identity)
    changed[key] = 'changed'
    with pytest.raises(ValueError, match='qualified evaluator runtime'):
        audit.evaluator_identity(changed, copy.deepcopy(changed), plan, preflight)
    with pytest.raises(ValueError, match='identity closure'):
        audit.evaluator_identity(identity, changed, plan, preflight)


def test_standardized_saved_reference_rescore_reports_original_unchanged():
    refs, bla, cells = decision_fixture()
    old = audit.decisions(refs, bla, cells, fit_complete=False, matrix_complete=False)
    parent = {'prior_continuation': copy.deepcopy(old), 'reference_banks': [{'opaque': 'bound'}]}
    rebuilt = copy.deepcopy(refs)
    rebuilt[0]['rows'] = rows(3.)
    derived = audit.decisions(rebuilt, bla, cells, fit_complete=False, matrix_complete=False)
    maps = ('family_means', 'per_record_means', 'per_seed_means', 'per_amplitude_means')
    report = {'banks': parent['reference_banks'], 'historical_continuation': copy.deepcopy(old),
        'rescored_continuation': {**old, **{k: derived[k] for k in maps}},
        'rows': [{**row, 'family': e['family'], 'seed': e['seed']} for e in rebuilt for row in e['rows']]
                +[{**row, 'family': audit.old.BLA, 'seed': None} for row in bla],
        'scope': '312 unchanged normalized prediction/target banks rescored; no model calls or historical report edits'}
    audit.validate_reference_report(report, parent, rebuilt, bla, cells)
    assert parent['prior_continuation']['family_means'][refs[0]['family']] == 5.
    report['historical_continuation']['family_means'][refs[0]['family']] = 3.
    with pytest.raises(ValueError):
        audit.validate_reference_report(report, parent, rebuilt, bla, cells)
