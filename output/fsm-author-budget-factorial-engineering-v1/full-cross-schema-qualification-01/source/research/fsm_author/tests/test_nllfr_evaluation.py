# SPDX-License-Identifier: GPL-3.0-or-later
"""Fabricated request/evidence checks, with no author or nonlinear model calls."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

COMPONENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPONENT/'src'))
SPEC = importlib.util.spec_from_file_location('nllfr_evaluation_under_test',
                                            COMPONENT/'scripts/evaluate_nllfr.py')
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def fabricated_records():
    mean, scale = np.array([10., 20., 30.]), np.array([2., 4., 8.])
    records = []
    for i, rid in enumerate(evaluation.DEV_IDS):
        u = (np.arange(8192, dtype=np.float64)[:, None]/8192
             + i + np.array([0., .25, .5]))
        records.append(SimpleNamespace(record_id=rid, u=u, y=mean+u*scale))
    return records, {'y_mean': mean, 'y_scale': scale}


class ObservedOnly:
    """Reject future-label slicing while the request is still in progress."""

    def __init__(self, values, allowed_start=0):
        self.values = values
        self.allowed_start = allowed_start
        self.slices = []

    def __getitem__(self, key):
        assert isinstance(key, slice)
        assert key.start == self.allowed_start and key.stop == self.allowed_start+100
        self.slices.append((key.start, key.stop))
        return self.values[key]


def test_one_request_uses_99_arrived_inputs_and_excludes_targets_from_timing(monkeypatch):
    record = fabricated_records()[0][0]
    record.y = ObservedOnly(record.y, allowed_start=256)
    phases = []

    def condition(_arrays, yc, uc):
        phases.append('condition')
        np.testing.assert_array_equal(yc[0], record.y.values[256:356])
        np.testing.assert_array_equal(uc[0], record.u[257:356])
        return np.full((1, 28), 7.), {'marker': 'observations only'}

    def rollout(_arrays, fu, state):
        phases.append('rollout')
        np.testing.assert_array_equal(fu[0], record.u[356:484])
        np.testing.assert_array_equal(state, np.full((1, 28), 7.))
        return fu.copy(), state+1

    ticks = iter([10., 10.001, 10.003, 10.007])
    monkeypatch.setattr(evaluation.time, 'perf_counter', lambda: next(ticks))
    monkeypatch.setattr(evaluation, 'condition', condition)
    monkeypatch.setattr(evaluation, 'physical_rollout', rollout)
    prediction, final, forecast, diagnostic, inputs, timing = evaluation.request({}, record, 256)
    assert phases == ['condition', 'rollout'] and record.y.slices == [(256, 356)]
    assert [v.shape for v in inputs] == [(1, 100, 3), (1, 99, 3), (1, 128, 3)]
    assert diagnostic == {'marker': 'observations only'}
    np.testing.assert_array_equal(prediction[0], record.u[356:484])
    np.testing.assert_array_equal(final, forecast+1)
    assert timing == pytest.approx({'request_ms': 7., 'slice_ms': 1.,
                                   'initializer_ms': 2., 'rollout_ms': 4.})


@pytest.mark.parametrize('error', [FloatingPointError('overflow'), np.linalg.LinAlgError('singular'),
    *(ValueError(s) for s in ('nonfinite linear seed system', 'nonfinite linear seed',
        'nonfinite scaled gradient', 'nonfinite context direction',
        'nonfinite final Jacobian diagnostics', 'nonfinite physical outputs',
        'nonfinite score inputs', 'score overflow'))])
def test_only_declared_numeric_errors_are_failed_requests(error):
    assert evaluation.numerical_failure(error) is True


@pytest.mark.parametrize('error', [ValueError('wrong dimensions'), ValueError('model mutation'),
                                 OSError('disk unavailable'), RuntimeError('bug'),
                                 TypeError('wrong schema'), AssertionError('mismatch')])
def test_structural_and_preservation_errors_are_not_numeric_outcomes(error):
    assert evaluation.numerical_failure(error) is False


def test_unknown_value_error_subclass_cannot_impersonate_numeric_guard():
    class SchemaError(ValueError):
        pass
    assert evaluation.numerical_failure(SchemaError('score overflow')) is False


def fake_pipeline(monkeypatch, tmp_path, failure_call=None, exception=None, mutation=False):
    """Exercise the real roster/scoring loop with a causal algebraic stub only."""
    records, norm = fabricated_records()
    arrays = {'token': np.array([1., 2., 3.])}
    saved, documents, calls = {}, {}, []
    monkeypatch.setattr(evaluation, 'validate', lambda _a: (28, 16, 8, 64))

    def save(path, **values):
        key = str(Path(path).relative_to(tmp_path))
        assert key not in saved
        saved[key] = {k: v.copy() for k, v in values.items()}

    def write(path, value):
        key = str(Path(path).relative_to(tmp_path))
        assert key not in documents
        documents[key] = json.loads(json.dumps(value, default=evaluation.json_value, allow_nan=False))

    def request(model, record, start):
        calls.append((record.record_id, start))
        if len(calls) == failure_call:
            raise exception
        inputs = evaluation.requests(record, np.array([start], dtype=np.int64))
        # A known direct-feedthrough fixture; no future y values are read here.
        pred = norm['y_mean'] + (inputs[2]+np.array([1., 2., 3.]))*norm['y_scale']
        if mutation:
            model['token'][0] += 1.
        diagnostic = {'requests': [{'status': 'STALLED', 'accepted_steps': 0}],
                      'linear_seed_states': np.zeros((1, 28)),
                      'solved_context_start_states': np.ones((1, 28))}
        return (pred, np.full((1, 28), 4.), np.full((1, 28), 2.), diagnostic, inputs,
                {'request_ms': 7., 'slice_ms': 1., 'initializer_ms': 2., 'rollout_ms': 4.})

    monkeypatch.setattr(evaluation.np, 'savez_compressed', save)
    monkeypatch.setattr(evaluation, 'write', write)
    monkeypatch.setattr(evaluation, 'event', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluation, 'request', request)
    return arrays, records, norm, saved, documents, calls


@pytest.mark.parametrize('failed_first', [False, True])
def test_complete_roster_or_preserved_numeric_failure_without_partial_record_average(
        monkeypatch, tmp_path, failed_first):
    args = fake_pipeline(monkeypatch, tmp_path, failure_call=1 if failed_first else None,
                         exception=FloatingPointError('fabricated overflow'))
    arrays, records, norm, saved, documents, calls = args
    result = evaluation.evaluate(arrays, records, norm, tmp_path)
    assert len(result['requests']) == 384 and len(result['rows']) == 12
    expected = [(r.record_id, s) for r in records for s in range(0, 7937, 256)]
    assert calls[:384] == expected
    assert calls[384:] == [(records[0].record_id, 0)] + [
        (r.record_id, s) for r in records for s in (0, 7936)]
    assert len(result['timings']) == 24 and result['median_request_ms'] == 7.
    assert result['timing_error'] is None
    assert len([k for k in saved if k.endswith('.inputs.npz')]) == 384
    banks = [k for k in saved if k.count('/') == 1]
    assert len(banks) == 12-int(failed_first)
    assert sum(r['status'] == 'complete' for r in result['requests']) == 384-int(failed_first)
    assert result['context_status_counts'] == {'STALLED': 384-int(failed_first)}
    assert result['persistent_numeric_bytes'] == 3*8+28*8+9*8
    assert 'not maximum over every internal trajectory' in result['retained_state_max_scope']
    for row in result['rows'][int(failed_first):]:
        assert row['mse'] == pytest.approx(14/3, rel=1e-13)
        assert row['rmse'] == pytest.approx(np.sqrt(14/3), rel=1e-13)
        assert row['per_channel_rmse'] == pytest.approx([1., 2., 3.])
        assert row['native_output_per_channel_rmse'] == pytest.approx([2., 8., 24.])
        assert row['requests'] == 32 and row['horizon'] == 128
    first = f'evaluation/{records[0].record_id}/0000'
    if failed_first:
        assert documents[first+'.json']['status'] == 'failed'
        assert first+'.npz' not in saved and first+'.inputs.npz' in saved
        assert result['rows'][0] == {'record_id': records[0].record_id, 'status': 'incomplete',
                                    'completed_requests': 31, 'expected_requests': 32}
    else:
        assert documents[first+'.json']['retained_state_max_abs'] == 4.
        np.testing.assert_allclose(saved[first+'.npz']['prediction']-saved[first+'.npz']['target'],
                                   np.broadcast_to([1., 2., 3.], (1, 128, 3)), rtol=0, atol=2e-14)
    assert documents['evaluation.json'] == result


@pytest.mark.parametrize('error', [ValueError('shape mismatch'), OSError('disk failure')])
def test_unknown_failure_aborts_after_preserving_attempt_inputs(monkeypatch, tmp_path, error):
    arrays, records, norm, saved, documents, calls = fake_pipeline(
        monkeypatch, tmp_path, failure_call=1, exception=error)
    with pytest.raises(type(error), match=str(error)):
        evaluation.evaluate(arrays, records, norm, tmp_path)
    assert len(calls) == 1 and len(saved) == 1 and not documents
    assert next(iter(saved)).endswith('/0000.inputs.npz')


@pytest.mark.parametrize('call,expected_samples', [(385, 0), (386, 0), (389, 3)])
def test_numeric_timing_failure_retains_scores_but_has_no_median_or_retry(
        monkeypatch, tmp_path, call, expected_samples):
    arrays, records, norm, _saved, _documents, calls = fake_pipeline(
        monkeypatch, tmp_path, failure_call=call, exception=np.linalg.LinAlgError('fixture singular'))
    result = evaluation.evaluate(arrays, records, norm, tmp_path)
    assert all(r['status'] == 'complete' for r in result['rows'])
    assert len(calls) == call and len(result['timings']) == expected_samples
    assert result['median_request_ms'] is None and 'fixture singular' in result['timing_error']


def test_model_mutation_is_fatal_not_an_unavailable_forecast(monkeypatch, tmp_path):
    arrays, records, norm, _saved, documents, calls = fake_pipeline(monkeypatch, tmp_path, mutation=True)
    with pytest.raises(ValueError, match='model mutation'):
        evaluation.evaluate(arrays, records, norm, tmp_path)
    assert len(calls) == 32 and 'evaluation.json' not in documents


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'reordered', 'mean_dtype',
    'mean_shape', 'scale_zero', 'scale_negative', 'scale_inf', 'mean_nan', 'extra_norm'])
def test_roster_and_scoring_coordinates_reject_before_requests(monkeypatch, tmp_path, mutation):
    arrays, records, norm, saved, documents, calls = fake_pipeline(monkeypatch, tmp_path)
    if mutation == 'missing':
        records.pop()
    elif mutation == 'duplicate':
        records[1] = records[0]
    elif mutation == 'reordered':
        records.reverse()
    elif mutation == 'mean_dtype':
        norm['y_mean'] = norm['y_mean'].astype(np.float32)
    elif mutation == 'mean_shape':
        norm['y_mean'] = norm['y_mean'][None]
    elif mutation == 'extra_norm':
        norm['extra'] = np.zeros(3)
    else:
        key = 'y_mean' if mutation == 'mean_nan' else 'y_scale'
        norm[key][0] = {'scale_zero': 0., 'scale_negative': -1., 'scale_inf': np.inf,
                        'mean_nan': np.nan}[mutation]
    with pytest.raises(ValueError, match='exact DEV roster|invalid common normalizer'):
        evaluation.evaluate(arrays, records, norm, tmp_path)
    assert not calls and not saved and not documents


def metadata_fixture(monkeypatch, tmp_path):
    """Only opaque sentinels and fake installed metadata, never a model archive."""
    monkeypatch.setattr(evaluation, 'ROOT', tmp_path)
    environment = {**{k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')},
        'JAX_PLATFORMS': 'cpu', 'JAX_ENABLE_X64': 'True', 'PYTHONHASHSEED': '0',
        'PYTHONDONTWRITEBYTECODE': '1'}
    for k, v in environment.items():
        monkeypatch.setenv(k, v)

    def put(name, value):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value))
        return path

    producer = put('research/fsm_author/scripts/fit_nllfr.py', '# fake producer')
    supervisor = put('research/fsm_author/scripts/run_nllfr_study.py', '# fake supervisor')
    vendor = put('installed/freq_statespace/core.py', '# fake qualified vendor')
    direct_url = {'url': 'https://example.invalid/qualified-fixture'}
    preflight = put('meta/runtime.json', {'status': 'PASS', 'python': evaluation.platform.python_version(),
        'executable': sys.executable, 'versions': {'fabricated-package': '1.0'},
        'upstream_direct_url': direct_url,
        'installed_source_matches': {'core.py': {'sha256': evaluation.sha(vendor), 'bytes': vendor.stat().st_size}}})
    monkeypatch.setattr(evaluation.importlib.metadata, 'version', lambda _name: '1.0')
    monkeypatch.setattr(evaluation.importlib.metadata, 'distribution', lambda _name: SimpleNamespace(
        read_text=lambda _name: json.dumps(direct_url), locate_file=lambda _name: vendor.parent))
    cfg = {'output': 'output/fake-study', 'process_directory': 'output/fake-process',
           'source_sha256': {str(p.relative_to(tmp_path)): evaluation.sha(p) for p in (producer, supervisor)},
           'prerequisites': {'runtime_preflight': {'path': str(preflight.relative_to(tmp_path)),
                                                 'sha256': evaluation.sha(preflight)}},
           'experiment': {'outer_timeout_seconds': 18000, 'rss_cap_bytes': 32*1024**3}}
    registration = put('research/registration.json', cfg)
    source = tmp_path/cfg['output']
    put(cfg['output']+'/final.npz', 'Not an NPZ. Admission must not decode this.')
    put(cfg['output']+'/final.zip', 'Not a ZIP. Admission must not decode this.')
    put(cfg['output']+'/fit.json', {'status': 'complete'})
    put(cfg['output']+'/summary.json', {'status': 'FIT_COMPLETE'})
    put(cfg['output']+'/solver-trace.npz', 'Opaque trace sentinel.')
    log = put(cfg['process_directory']+'/process.log', 'Original fake terminal log.\n')
    process = {'phase': 'fit', 'status': 'completed', 'observed_exit_code': 0,
        'registration_sha256': evaluation.sha(registration), 'end_identity_matches': True,
        'log_sha256': evaluation.sha(log), 'command': [str(tmp_path/'research/fsm_author/.venv/bin/python'),
            str(producer), '--registration', str(registration), '--output', str(source)],
        'timeout_seconds': 18000, 'rss_cap_bytes': 32*1024**3, 'environment': environment,
        'producer': evaluation.pin(producer), 'supervisor': evaluation.pin(supervisor),
        'artifacts': {str(p.relative_to(source)): {k: evaluation.pin(p)[k] for k in ('sha256', 'bytes')}
                      for p in source.rglob('*') if p.is_file()}}
    process_path = put(cfg['process_directory']+'/process.json', process)

    def forbidden(*_args, **_kwargs):
        raise AssertionError('numerical decode/model call during metadata admission')

    monkeypatch.setattr(evaluation.np, 'load', forbidden)
    monkeypatch.setattr(evaluation, 'condition', forbidden)
    monkeypatch.setattr(evaluation, 'physical_rollout', forbidden)
    monkeypatch.setattr(evaluation.fsm_data, 'read_npz_estimation', forbidden)
    return cfg, registration, process, process_path, source, vendor


@pytest.mark.parametrize('fit_status', ['complete', 'iteration_cap_reached'])
def test_closed_metadata_admission_does_not_decode_numeric_sentinels(monkeypatch, tmp_path, fit_status):
    cfg, registration, process, path, source, _vendor = metadata_fixture(monkeypatch, tmp_path)
    (source/'fit.json').write_text(json.dumps({'status': fit_status}))
    process['artifacts']['fit.json'] = {k: evaluation.pin(source/'fit.json')[k] for k in ('sha256', 'bytes')}
    path.write_text(json.dumps(process))
    admitted, fit, provenance = evaluation.authenticate(cfg, registration)
    assert admitted == source and fit['status'] == fit_status
    assert provenance['process_sha256'] == evaluation.sha(path)
    assert provenance['registration_sha256'] == evaluation.sha(registration)


@pytest.mark.parametrize('mutation', ['running', 'exit', 'phase', 'identity', 'registration',
    'argv', 'environment', 'cap', 'rss', 'producer_pin', 'supervisor_pin', 'source', 'vendor',
    'log', 'trace', 'unlisted', 'missing', 'symlink', 'inventory_bytes'])
def test_original_closure_and_complete_inventory_tampering_rejected_before_decode(
        monkeypatch, tmp_path, mutation):
    cfg, registration, process, path, source, vendor = metadata_fixture(monkeypatch, tmp_path)
    if mutation == 'running':
        process['status'] = 'running'
    elif mutation == 'exit':
        process['observed_exit_code'] = 1
    elif mutation == 'phase':
        process['phase'] = 'evaluate'
    elif mutation == 'identity':
        process['end_identity_matches'] = False
    elif mutation == 'registration':
        registration.write_text('{}')
    elif mutation == 'argv':
        process['command'][1] = 'unrelated.py'
    elif mutation == 'environment':
        process['environment']['OMP_NUM_THREADS'] = '2'
    elif mutation == 'cap':
        process['timeout_seconds'] += 1
    elif mutation == 'rss':
        process['rss_cap_bytes'] += 1
    elif mutation in ('producer_pin', 'supervisor_pin'):
        process[mutation.removesuffix('_pin')]['sha256'] = '0'*64
    elif mutation == 'source':
        (tmp_path/next(iter(cfg['source_sha256']))).write_text('changed')
    elif mutation == 'vendor':
        vendor.write_text('changed')
    elif mutation == 'log':
        path.with_name('process.log').write_text('changed')
    elif mutation == 'trace':
        (source/'solver-trace.npz').write_text('changed')
    elif mutation == 'unlisted':
        (source/'extra.txt').write_text('unlisted')
    elif mutation == 'missing':
        (source/'solver-trace.npz').unlink()
    elif mutation == 'symlink':
        (source/'linked.txt').symlink_to(vendor)
    else:
        process['artifacts']['final.npz']['bytes'] += 1
    path.write_text(json.dumps(process))
    with pytest.raises(ValueError):
        evaluation.authenticate(cfg, registration)
