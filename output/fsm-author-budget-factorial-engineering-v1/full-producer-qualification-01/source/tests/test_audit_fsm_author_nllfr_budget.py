# SPDX-License-Identifier: GPL-3.0-or-later
"""Fabricated FIT-only witnesses, never production or measured-data calls."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1]/'scripts/audit_fsm_author_nllfr_budget.py'
SPEC = importlib.util.spec_from_file_location('independent_nllfr_budget_audit', PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def model():
    m = {'A': .5*np.eye(28), 'B_u': np.zeros((28, 3)), 'C_y': np.zeros((3, 28)),
         'D_yu': np.zeros((3, 3)), 'B_w': np.zeros((28, 8)), 'C_z': np.zeros((16, 28)),
         'D_yw': np.zeros((3, 8)), 'D_zu': np.zeros((16, 3)),
         'W0': np.zeros((64, 16)), 'W1': np.zeros((64, 64)), 'W2': np.zeros((8, 64)),
         'b0': np.zeros(64), 'b1': np.zeros(64), 'b2': np.zeros(8),
         'u_mean': np.zeros(3), 'u_std': np.ones(3),
         'y_mean': np.zeros(3), 'y_std': np.ones(3), 'ts': np.array(1/6400.)}
    return m


def trace(n=4, stopped=False):
    return {'loss_history': np.array([4., 2., 3., 1.])[:n].copy(),
            'iter_times': np.arange(n, dtype=np.float64), 'iter_count': np.array(n),
            'author_stop_flag': np.array(stopped), 'wall_time': np.array(2.)}


def test_prefix_is_exact_not_close_and_preserves_first_mismatch():
    parent = np.array([4., 2., 3., 1.])
    loss = np.r_[parent, .5]
    assert audit.prefix_result(loss, parent, required=4) == {
        'required': 4, 'compared': 4, 'exact': True, 'first_difference': None,
        'max_absolute_difference': 0.}
    loss[1] = np.nextafter(loss[1], np.inf)
    result = audit.prefix_result(loss, parent, required=4)
    assert result['exact'] is False and result['first_difference'] == 1
    assert result['max_absolute_difference'] == loss[1]-parent[1]


def test_short_matching_prefix_is_not_attributable():
    parent = np.array([4., 2., 3., 1.])
    assert audit.prefix_result(parent[:2], parent, required=4) == {
        'required': 4, 'compared': 2, 'exact': False, 'first_difference': None,
        'max_absolute_difference': 0.}


@pytest.mark.parametrize('change', ['dtype', 'nan', 'negative', 'rank', 'parent_length'])
def test_invalid_prefix_evidence(change):
    parent = np.array([4., 2., 3., 1.]); loss = parent.copy()
    if change == 'dtype':
        loss = loss.astype(np.float32)
    elif change == 'nan':
        loss[0] = np.nan
    elif change == 'negative':
        loss[0] = -1.
    elif change == 'rank':
        loss = loss[None]
    else:
        parent = parent[:-1]
    with pytest.raises(ValueError):
        audit.prefix_result(loss, parent, required=4)


def test_trace_allows_nonmonotone_loss_and_rejects_early_nonstop():
    assert audit.validate_trace(trace(), cap=4) == (4, False)
    assert audit.validate_trace(trace(2, True), cap=4) == (2, True)
    with pytest.raises(ValueError, match='cap'):
        audit.validate_trace(trace(2, False), cap=4)


@pytest.mark.parametrize('key,value', [('iter_count', np.array(4.)),
    ('author_stop_flag', np.array(1)), ('iter_times', np.array([0., 1., np.inf, 3.])),
    ('loss_history', np.zeros(3)), ('wall_time', np.array(-1.))])
def test_trace_schema_failures(key, value):
    item = trace(); item[key] = value
    with pytest.raises(ValueError):
        audit.validate_trace(item, cap=4)


def test_all_initial_parameters_and_nested_bla_owned_values():
    m = model(); base = {k: m[k].copy() for k in audit.old.BASE}
    assert audit.validate_initial(m, copy.deepcopy(m), base, copy.deepcopy(base)) == {}
    final = copy.deepcopy(m); final['W0'][0, 0] = 2.
    changed = audit.validate_initial(m, copy.deepcopy(m), base, copy.deepcopy(base), final)
    assert len(changed) == 14 and sum(v['total_scalars'] for v in changed.values()) == 7473
    assert changed['W0']['changed_scalars'] == 1
    assert sum(v['changed_scalars'] for v in changed.values()) == 1


@pytest.mark.parametrize('change', ['initial_weight', 'initial_dtype', 'nested_bla', 'final_norm', 'final_geometry'])
def test_changed_start_or_frozen_fields_fail(change):
    m = model(); initial = copy.deepcopy(m); final = copy.deepcopy(m)
    base = {k: m[k].copy() for k in audit.old.BASE}; expected = copy.deepcopy(base)
    if change == 'initial_weight':
        initial['b2'][0] = 1.
    elif change == 'initial_dtype':
        initial['W0'] = initial['W0'].astype(np.float32)
    elif change == 'nested_bla':
        base['A'][0, 0] = .7
    elif change == 'final_norm':
        final['y_std'][0] = 2.
    else:
        final['W2'] = final['W2'][:, :-1]
    with pytest.raises(ValueError):
        audit.validate_initial(initial, m, base, expected, final)


def cache_fixture():
    # Four samples x two realizations x two periods: x=t+10*r+100*p+channel.
    raw = np.arange(4.)[:, None, None, None] + 10*np.arange(2.)[None, None, :, None]
    raw = raw + 100*np.arange(2.)[None, None, None, :] + np.arange(3.)[None, :, None, None]
    variance = 1.25+25+2500
    normalized = (raw-(56.5+np.arange(3.))[None, :, None, None])/np.sqrt(variance)
    m = model()
    for name in ('u', 'y'):
        m[name+'_mean'] = 56.5+np.arange(3.)
        m[name+'_std'] = np.full(3, np.sqrt(variance))
    cache = {'f_idx': np.arange(1, 3840), 'training_x0': np.zeros((28, 2)),
             'independent_x0': np.zeros((28, 2))}
    for name in ('u', 'y'):
        cache['raw_fit_'+name] = raw.copy()
        cache[name] = normalized.mean(axis=3)
        cache[name.upper()] = np.fft.rfft(normalized, axis=0).mean(axis=3)
    return cache, m


def test_audited_fit_cache_normalization_periods_and_zero_forcing_x0():
    cache, m = cache_fixture()
    rebuilt, x0 = audit.reconstruct_cache(cache, m, samples=4, realizations=2, offset=1)
    assert np.all(x0 == 0)
    # First realization period-mean standardized samples: -6.5,-5.5,-4.5,-3.5.
    expected = np.array([-6.5, -5.5, -4.5, -3.5])/np.sqrt(2526.25)
    np.testing.assert_allclose(rebuilt['u'][:, 0, 0], expected, rtol=0, atol=2e-16)
    np.testing.assert_allclose(rebuilt['U'][:, 0, 0], np.fft.rfft(expected), rtol=0, atol=2e-16)


@pytest.mark.parametrize('key', ['u', 'Y', 'training_x0', 'f_idx'])
def test_cached_numeric_corruption_detected(key):
    cache, m = cache_fixture(); cache[key].flat[0] += 1
    with pytest.raises(ValueError):
        audit.reconstruct_cache(cache, m, samples=4, realizations=2, offset=1)


def test_full_spectral_objective_dc_nyquist_and_interior_not_halved():
    m = model()
    y = np.zeros((8, 3, 1))
    y[:, 0, 0] = 2+3*(-1.)**np.arange(8)+4*np.cos(2*np.pi*np.arange(8)/8)
    loss = audit.old.native_objective(m, np.zeros_like(y), np.fft.rfft(y, axis=0), np.zeros((28, 1)), offset=1)
    # DC16, Nyquist24 and the unweighted interior magnitude16: (256+576+256)/5.
    assert loss == pytest.approx(1088/5, rel=0, abs=1e-12)


@pytest.mark.parametrize('stop,prefix,terminal,expected', [
    (True, True, True, 'FIT_ONLY_COMPLETE'), (False, True, True, 'FIT_ONLY_INCOMPLETE'),
    (True, False, True, 'FIT_ONLY_INCOMPLETE'), (True, True, False, 'FIT_ONLY_INCOMPLETE')])
def test_fit_only_status_never_promotes_cap_or_prefix_failure(stop, prefix, terminal, expected):
    assert audit.scientific_status(stop, prefix, terminal) == expected


def closed_fixture(tmp_path, monkeypatch, status='completed', code=0):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if not isinstance(value, str) else value)
        return path
    registration = write(tmp_path/'registration.json', {})
    parent = write(tmp_path/'parent/process.json', {'opaque': 'already closed original'})
    study = tmp_path/'study'; study.mkdir()
    write(study/'final.npz', 'intentionally invalid NPZ; no decode in admission')
    producer = write(tmp_path/'research/fsm_author/scripts/fit_nllfr_budget.py', '# fake source')
    supervisor = write(producer.with_name('run_nllfr_budget_study.py'), '# fake supervisor')
    process = tmp_path/'process/process.json'
    log = write(process.parent/'process.log', 'original log')
    command = [str(tmp_path/'research/fsm_author/.venv/bin/python'), str(producer),
               '--registration', str(registration), '--output', str(study)]
    launch = {'phase': 'fit', 'command': command, 'status': 'running',
              'registration_sha256': audit.pin(registration)['sha256'], 'started_time_ns': 123,
              'timeout_seconds': 18000, 'rss_cap_bytes': 32*1024**3, 'environment': audit.old.ENV,
              'supervisor': audit.descriptor(supervisor), 'producer': audit.descriptor(producer),
              'parent_fit_process': audit.descriptor(parent)}
    write(process.parent/'launch.json', launch)
    terminal = {**launch, 'status': status, 'observed_exit_code': code, 'pid': 1234,
                'elapsed_seconds': 2., 'wall_elapsed_seconds': 2.1, 'peak_polled_child_rss_bytes': 100,
                'end_identity_matches': True, 'identity_error': None,
                'log_sha256': audit.pin(log)['sha256'], 'artifacts': audit.old.inventory(study),
                'outcome': 'ORIGINAL_PROCESS_COMPLETE' if status == 'completed' else 'INCOMPLETE'}
    write(process, terminal)
    monkeypatch.setattr(audit.np, 'load', lambda *a, **k: pytest.fail('numerical decode forbidden'))
    return ({'source_sha256': {}}, registration, study, process, {'parent_process': parent}), terminal, write


def test_original_closed_inventory_admitted_without_npz_decode(tmp_path, monkeypatch):
    args, expected, _ = closed_fixture(tmp_path, monkeypatch)
    terminal, files = audit.closed_process(*args)
    assert terminal == expected and set(files) == {'final.npz'}


@pytest.mark.parametrize('status', ['timeout', 'memory_limit'])
def test_selected_stop_with_racing_zero_exit_stays_incomplete(tmp_path, monkeypatch, status):
    args, _, _ = closed_fixture(tmp_path, monkeypatch, status, 0)
    terminal, _ = audit.closed_process(*args)
    assert terminal['status'] == status and terminal['outcome'] == 'INCOMPLETE'


@pytest.mark.parametrize('change', ['live', 'argv', 'exit', 'identity', 'parent', 'cap',
                                   'env', 'source', 'log', 'extra', 'missing', 'launch'])
def test_original_process_tampering_rejected(tmp_path, monkeypatch, change):
    args, terminal, write = closed_fixture(tmp_path, monkeypatch)
    _, _, study, process, _ = args
    if change == 'live':
        terminal['status'] = 'running'
    elif change == 'argv':
        terminal['command'][-1] = str(tmp_path/'other')
    elif change == 'exit':
        terminal['observed_exit_code'] = 1
    elif change == 'identity':
        terminal['end_identity_matches'] = False
    elif change == 'parent':
        terminal['parent_fit_process']['sha256'] = '0'*64
    elif change == 'cap':
        terminal['timeout_seconds'] += 1
    elif change == 'env':
        terminal['environment'] = {**terminal['environment'], 'OMP_NUM_THREADS': '2'}
    elif change == 'source':
        write(Path(terminal['producer']['path']), '# changed source')
    elif change == 'log':
        write(process.parent/'process.log', 'changed log')
    elif change == 'extra':
        write(study/'extra.txt', 'unexpected')
    elif change == 'missing':
        (study/'final.npz').unlink()
    else:
        launch = audit.read(process.parent/'launch.json'); launch['started_time_ns'] += 1
        write(process.parent/'launch.json', launch)
    write(process, terminal)
    with pytest.raises(ValueError):
        audit.closed_process(*args)


def test_honest_prefix_failure_finite_fit_is_auditable_incomplete():
    t = trace(stopped=True)
    prefix = audit.prefix_result(np.array([4., 2.01, 3., 1.]), np.array([4., 2., 3., 1.]), required=4)
    fit = {'iterations': 4, 'author_stop_flag': True, 'optimizer_status': 'complete',
           'status': 'prefix_mismatch', 'budget_only_attributable': False,
           'initial': {'opaque': 'initial'}, 'final': {'opaque': 'final'}, 'prefix': {'opaque': 'prefix'},
           'trainable_scalars': 7473, 'discarded_vendor_warmup_steps': 1,
           'zip_roundtrip_all_numeric_arrays_bitwise_equal': True, 'author_reported_seconds': 2.,
           'optimization_and_preservation_compile_inclusive_seconds': 3., 'peak_ru_maxrss_bytes': 100}
    assert audit.validate_fit(fit, t, prefix, fit['initial'], fit['final'], fit['prefix']) == 'FIT_ONLY_INCOMPLETE'
    fit['status'] = 'complete'
    with pytest.raises(ValueError, match='semantics'):
        audit.validate_fit(fit, t, prefix, fit['initial'], fit['final'], fit['prefix'])


def freeze_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    def write(name, value):
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if not isinstance(value, str) else value)
        return audit.descriptor(path)
    source = {name: write(name, '# fabricated frozen source') for name in (*audit.SOURCES, audit.HELPER)}
    monkeypatch.setattr(audit, 'HELPER_SHA', source[audit.HELPER]['sha256'])
    qualified = {k: source[k] for k in audit.SOURCES}
    snapshots = {k: write('snapshot/'+k, '# fabricated frozen source') for k in audit.SOURCES}
    commands = [['.venv/bin/ruff', 'check', *audit.SOURCES],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', audit.SOURCES[1]]]
    preflight = write('preflight.json', {'sources': qualified, 'snapshots': snapshots, 'commands': commands})
    logs = [write(f'command-{i}.log', 'PASS fabricated') for i in (1, 2)]
    qualification = write('qualification.json', {'status': 'PASS', 'sources_unchanged': True,
        'sources_before': qualified, 'sources_after': qualified, 'preflight': preflight,
        'commands': [{'command': c, 'returncode': 0, 'log': log} for c, log in zip(commands, logs, strict=True)]})
    registration = write('future-registration.json', {'version': audit.VERSION})
    held = {'status': 'FROZEN_BEFORE_EMPIRICAL_AUDIT', 'registration': registration,
            'sources': source, 'qualification': qualification}
    write('freeze.json', held)
    return tmp_path/'freeze.json', held, write


def test_auditor_freeze_binds_later_registration_without_circular_source(tmp_path, monkeypatch):
    path, expected, _ = freeze_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(audit.np, 'load', lambda *a, **k: pytest.fail('arrays forbidden'))
    held, registration = audit.source_freeze(path)
    assert held == expected and registration == tmp_path/'future-registration.json'


@pytest.mark.parametrize('change', ['source', 'helper', 'registration', 'log', 'snapshot', 'qualification', 'roster'])
def test_auditor_freeze_rejects_predecode_drift(tmp_path, monkeypatch, change):
    path, held, write = freeze_fixture(tmp_path, monkeypatch)
    if change == 'source':
        write(audit.SOURCES[0], '# changed auditor')
    elif change == 'helper':
        write(audit.HELPER, '# changed independent helper')
    elif change == 'registration':
        write('future-registration.json', {'version': 'changed'})
    elif change == 'log':
        write('command-1.log', 'changed')
    elif change == 'snapshot':
        write('snapshot/'+audit.SOURCES[1], '# changed snapshot')
    elif change == 'qualification':
        write('qualification.json', {'status': 'FAIL'})
    else:
        held['sources'].pop(audit.SOURCES[1]); write('freeze.json', held)
    with pytest.raises(ValueError):
        audit.source_freeze(path)
