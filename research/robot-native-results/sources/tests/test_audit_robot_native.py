"""Fabricated saved evidence only; no models, library or measured data."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_robot_native as audit


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def work(arm, batch):
    gru = arm == 'gru32'
    pieces = {'householder': (638, 528), 'dense_bounded': (806, 1208),
              'dense_unbounded': (806, 1200), 'dense_mlp': (590, 1208), 'gru32': (5916, 0)}
    packed, prepared = pieces[arm]
    state = 50 if gru else 12
    row = {'parameter_validations': 1 if gru else 2, 'parameter_preparations': 0 if gru else 1,
           'ffi_calls': 1, 'batch': batch, 'horizon': 128, 'packed_parameter_bytes': packed*4,
           'parameter_export_piece_bytes': packed*4, 'input_copy_bytes': batch*4*(1152 if gru else 780),
           'output_buffer_bytes': batch*4*(128*6+state), 'normalized_input_bytes': batch*8*1152,
           'cast_input_bytes': batch*4*1152, 'physical_output_bytes': batch*8*128*6,
           'retained_prepared_bytes': 0, 'scope': 'fabricated counters'}
    if gru:
        row.update(context=32, context_gru_steps=batch*30, rollout_gru_steps=batch*128)
    else:
        row.update(steps=batch*128, condition_state_bytes=batch*48, prepared_tensor_bytes=prepared)
    return row


def parity(arm='householder'):
    arrays, checks, works = {}, {}, {}
    for label, n in [('batch22', 22), ('batch1', 1)]:
        for name in ('torch_standardized', 'torch_physical', 'native_physical', 'native_standardized'):
            arrays[label+'/'+name] = np.zeros((n, 128, 6), dtype=np.float64)
        for name in ('torch_final', 'native_final', 'native_standard_final'):
            arrays[label+'/'+name] = np.zeros((n, 50 if arm == 'gru32' else 12), dtype=np.float32)
        for name, left, _ in audit.CHECKS:
            checks[label+'/'+name] = {'passed': True, 'values': arrays[label+'/'+left].size,
                                    'nonfinite': 0, 'violations': 0, 'max_absolute_error': 0.0}
        works[label+'/physical'] = work(arm, n)
        works[label+'/standardized'] = work(arm, n)
    return {'checks': checks, 'errors': [], 'work': works, 'passed': True}, arrays


@pytest.mark.parametrize('arm', audit.ARMS)
def test_full_saved_geometry_and_work_include_dense_norms(arm):
    row, arrays = parity(arm)
    assert audit.validate_parity(row, arrays, arm) == {'passed': True, 'checks': 8, 'arrays': 14, 'backend_errors': 0}
    if arm in ('dense_bounded', 'dense_mlp'):
        row['work']['batch22/physical']['prepared_tensor_bytes'] -= 8
        with pytest.raises(ValueError, match='identity'):
            audit.validate_parity(row, arrays, arm)


def test_fixed_threshold_and_saved_failed_comparison():
    result = audit.array_comparison(np.array([0.000009, 1.000019]), np.array([0., 1.]))
    assert result['passed'] and result['violations'] == 0
    result = audit.array_comparison(np.array([0.000011, 1.000021]), np.array([0., 1.]))
    assert not result['passed'] and result['violations'] == 2
    row, arrays = parity()
    arrays['batch1/native_standard_final'][0, 0] = .01
    row['checks']['batch1/standardized_final_state'] = audit.array_comparison(
        arrays['batch1/native_standard_final'], arrays['batch1/torch_final'])
    row['passed'] = False
    assert audit.validate_parity(row, arrays, 'householder')['passed'] is False


@pytest.mark.parametrize('damage', ['missing', 'extra', 'dtype', 'shape', 'work', 'claimed_pass'])
def test_corrupt_saved_parity_is_rejected(damage):
    row, arrays = parity()
    if damage == 'missing':
        arrays.pop('batch1/native_standardized')
    elif damage == 'extra':
        arrays['hidden_target'] = np.zeros(1)
    elif damage == 'dtype':
        arrays['batch22/native_final'] = arrays['batch22/native_final'].astype(np.float64)
    elif damage == 'shape':
        arrays['batch1/native_final'] = np.zeros((1, 11), dtype=np.float32)
    elif damage == 'work':
        row['work']['batch1/physical']['retained_prepared_bytes'] = 4
    else:
        arrays['batch1/native_physical'][0, 0, 0] = 1.
    with pytest.raises(ValueError):
        audit.validate_parity(row, arrays, 'householder')


def test_failed_backend_requires_explicit_error_and_no_invented_outputs():
    row, arrays = parity('gru32')
    row['errors'] = [{'batch': 'batch1', 'backend': 'native_physical',
                      'type': 'ValueError', 'message': 'fabricated nonfinite native state'}]
    for key in ('native_physical', 'native_final'):
        arrays.pop('batch1/'+key)
    for key in ('physical', 'physical_final_state'):
        row['checks'].pop('batch1/'+key)
    row['work'].pop('batch1/physical')
    row['passed'] = False
    assert audit.validate_parity(row, arrays, 'gru32') == {'passed': False, 'checks': 6, 'arrays': 12, 'backend_errors': 1}
    row['errors'].clear()
    with pytest.raises(ValueError):
        audit.validate_parity(row, arrays, 'gru32')


def pairs():
    return [{'phase': 'warmup' if index < 3 else 'timed', 'repetition': index if index < 3 else index-3,
             'order': ['torch', 'native'] if (index if index < 3 else index-3) % 2 == 0 else ['native', 'torch'],
             'seconds': {'torch': float(index+1), 'native': 2.0}} for index in range(33)]


def test_exact_timing_alternation_and_ratio_of_medians():
    result = audit.timing_summary(pairs())
    assert result['median_seconds'] == {'torch': 18.5, 'native': 2.0}
    assert result['torch_over_native'] == 9.25
    assert result['paired_ratios'] == [x/2 for x in range(4, 34)]


@pytest.mark.parametrize('damage', ['missing', 'zero', 'nonfinite', 'order', 'duplicate_repetition'])
def test_invalid_timing_sample_or_roster_rejected(damage):
    rows = pairs()
    if damage == 'missing': rows.pop()
    elif damage == 'zero': rows[0]['seconds']['torch'] = 0.0
    elif damage == 'nonfinite': rows[0]['seconds']['native'] = float('inf')
    elif damage == 'order': rows[3]['order'].reverse()
    else: rows[-1]['repetition'] -= 1
    with pytest.raises(ValueError): audit.timing_summary(rows)


def test_family_aggregation_preserves_each_recording_and_seed():
    rows = [{'recording': recording, 'arm': arm, 'seed': seed,
             'median_seconds': {'torch': value*4, 'native': value}}
            for recording in ('a', 'b') for arm in audit.ARMS
            for seed, value in zip(audit.SEEDS, (1.0, 3.0, 20.0), strict=True)]
    families, ratios = audit.aggregate(rows, ['a', 'b'])
    assert len(families) == 10 and len(ratios) == 8
    assert all(row['median_of_fit_medians_seconds'] == {'torch': 12.0, 'native': 3.0} for row in families)
    assert all(row['native_latency_ratio'] == 1.0 for row in ratios)
    with pytest.raises(ValueError): audit.aggregate(rows[:-1], ['a', 'b'])


def test_bad_process_stops_before_array_decode(tmp_path, monkeypatch):
    config = {'arms': list(audit.ARMS), 'seeds': list(audit.SEEDS), 'rtol': 1e-5, 'atol': 1e-5,
              'cap_seconds': 900, 'warmups': 3, 'repeats': 30, 'parity_records': 30, 'batch_sizes': [22, 1]}
    registration, process = tmp_path/'registration.json', tmp_path/'process.json'
    write(registration, {'version': 'robot-native-benchmark-v1', 'config': config, 'command': ['fake']})
    write(process, {'command': ['fake'], 'returncode': 7, 'inputs_unchanged': True, 'elapsed_seconds': 1.0})
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('no array before closed receipt'))
    with pytest.raises(ValueError, match='closed original'):
        audit.audit(tmp_path, process, registration)


def terminal_fixture(folder, failed):
    """Complete 30-slot saved output with hand-zero forecasts and known timings."""
    fits, originals, resources, parity_rows, timing_rows = [], [], [], [], []
    recordings = ['fake_a', 'fake_b']
    for arm in audit.ARMS:
        for seed in audit.SEEDS:
            fit = {'key': f'{arm}-{seed}-lr0', 'arm': arm, 'seed': seed, 'learning_rate': .001}
            fits.append(fit)
            count, state = audit.PARAMETERS[arm], 50 if arm == 'gru32' else 12
            storage = {'parameter_count': count, 'parameter_bytes': 4*count, 'buffer_bytes': 0,
                       'retained_gradient_bytes': 0, 'state_bytes_per_stream': state*4,
                       'normalizer_bytes': 192, 'retained_prepared_bytes': 0, 'scope': 'fixture'}
            identity = {'fit_key': fit['key'], 'arm': arm, 'seed': seed, 'learning_rate': .001}
            resources.append({**identity, 'storage': storage, 'retained_gradient_tensors': 0})
            originals.append({'arm': arm, 'seed': seed, 'learning_rate': .001, 'parameters': count,
                              'parameter_bytes': count*4, 'state_bytes': state*4, 'buffer_bytes': 0,
                              'normalizer_bytes': 192})
            for recording in recordings:
                row, arrays = parity(arm)
                if failed and not parity_rows:
                    arrays['batch1/native_standard_final'][0, 0] = 1.0
                    row['checks']['batch1/standardized_final_state'].update(passed=False, violations=1, max_absolute_error=1.0)
                    row['passed'] = False
                filename = f'parity-{recording}-{fit["key"]}.npz'
                np.savez_compressed(folder/filename, **arrays)
                row.update(**identity, recording=recording, file=filename)
                parity_rows.append(row)
                write(folder/f'parity-{len(parity_rows):02d}.json', row)
                if not failed:
                    timing = {**fit, 'recording': recording, 'status': 'PASS', 'pairs': pairs(),
                              'median_seconds': {'torch': 18.5, 'native': 2.0}, 'torch_over_native': 9.25,
                              'paired_ratios': [x/2 for x in range(4, 34)]}
                    timing_rows.append(timing)
                    write(folder/f'timing-{len(timing_rows):02d}.json', timing)
    write(folder/'parity.json', parity_rows)
    write(folder/'definition.json', {})
    write(folder/'host.json', {'threads': dict.fromkeys(audit.THREADS, '1'), 'torch_threads': 1, 'torch_interop_threads': 1})
    outcome, error, result = 'FAILED', {'type': 'ValueError', 'message': 'all30 parity comparisons must pass before timing'}, None
    if not failed:
        outcome, error = 'PASS', None
        write(folder/'parity-barrier.json', {'passed': True, 'comparisons': 30, 'timed_requests_started': 0})
        families, ratios = audit.aggregate(timing_rows, recordings)
        result = {'family_timings': families, 'native_family_ratios': ratios, 'scope': 'fixture'}
    summary = {'version': 'robot-native-benchmark-v1', 'status': outcome, 'error': error,
               'parity': parity_rows, 'timings': timing_rows, 'resources': resources, 'result': result,
               'elapsed_seconds': 1.0, 'clock_error': None, 'inputs_unchanged': True,
               'confirmation_access': False, 'raw_measurement_access': False, 'target_scoring': False}
    write(folder/'summary.json', summary)
    files = {p.name: audit.pin(p) for p in folder.iterdir()}
    return {'inputs': {str(folder/name): value for name, value in files.items()}, 'manifest': files,
            'receipt': {'elapsed_seconds': 1.0, 'status': outcome, 'error': error, 'parity_comparisons': 30,
                        'timed_slots': len(timing_rows)}, 'process': {'returncode': 1 if failed else 0},
            'selected': fits, 'recordings': recordings, 'original_audit': {'resources': originals}}


@pytest.mark.parametrize('failed', [False, True])
def test_saved_terminal_audit_separates_evidence_agreement_from_benchmark_outcome(tmp_path, monkeypatch, failed):
    proof = terminal_fixture(tmp_path, failed)
    monkeypatch.setattr(audit, 'authenticate', lambda *a: copy.deepcopy(proof))
    result = audit.audit(tmp_path, None, None)
    assert result['status'] == 'PASS' and result['agreement'] is True
    assert result['benchmark_status'] == ('FAILED' if failed else 'PASS')
    assert result['parity_passed'] == (29 if failed else 30)
    assert result['counts']['npz_decodes'] == 30 and result['counts']['array_loads'] == 420
    assert result['counts']['parity_checks'] == 240 and result['counts']['model_calls'] == 0
    assert result['counts']['timed_pairs'] == (0 if failed else 900)


def test_parity_failure_cannot_keep_favorable_timing(tmp_path, monkeypatch):
    proof = terminal_fixture(tmp_path, True)
    summary = json.loads((tmp_path/'summary.json').read_text())
    summary['timings'] = [{'fabricated_favorable_result': True}]
    write(tmp_path/'summary.json', summary)
    proof['inputs'][str(tmp_path/'summary.json')] = audit.pin(tmp_path/'summary.json')
    monkeypatch.setattr(audit, 'authenticate', lambda *a: proof)
    with pytest.raises(ValueError, match='zero timing'):
        audit.audit(tmp_path, None, None)
