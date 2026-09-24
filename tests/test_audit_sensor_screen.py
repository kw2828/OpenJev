"""Hand-sized chronology and regression witnesses, not sensor-data evaluation."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_sensor_screen as a


def example():
    timestamps = np.array([-30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -1, 0, 2, 3], dtype=np.int64)
    i = np.arange(len(timestamps), dtype=np.float64)
    x = np.column_stack([np.sin((i + 1) * (j + 1) / 10) + i / (j + 2) for j in range(7)])
    y = .7 + x[:, 1] - .1 * x[:, 4]
    eligible = timestamps + 24 < 0
    mean, scale = a.normalization(x, eligible)
    return timestamps, x, y, eligible, mean, scale


def run_example(timestamps, x, y, eligible, mean, scale):
    return a.replay(timestamps, x, y, start_hour=0, end_hour=4,
                    mean=mean, scale=scale, fit_mask=eligible)


def test_population_normalization_uses_only_explicit_eligible_rows():
    x = np.tile(np.array([1., 3., 100.])[:, None], (1, 7))
    mask = np.array([True, True, False])
    mean, scale = a.normalization(x, mask)
    np.testing.assert_array_equal(mean, np.full(7, 2.))
    np.testing.assert_array_equal(scale, np.ones(7))
    x[2] = np.nan
    later_mean, later_scale = a.normalization(x, mask)
    np.testing.assert_array_equal(mean, later_mean)
    np.testing.assert_array_equal(scale, later_scale)


def test_design_intercept_s2_index_and_all_quadratic_cross_terms():
    x = np.arange(1., 8.)[None]
    mean, scale = np.zeros(7), np.ones(7)
    np.testing.assert_array_equal(a.design(x, mean, scale, 's2linear'), [[1., 2.]])
    np.testing.assert_array_equal(a.design(x, mean, scale, 's2quadratic'), [[1., 2., 4.]])
    np.testing.assert_array_equal(a.design(x, mean, scale, 'linear7'), [[1., 1., 2., 3., 4., 5., 6., 7.]])
    quadratic = a.design(x, mean, scale, 'quadratic7')
    assert quadratic.shape == (1, 36)
    np.testing.assert_array_equal(quadratic[0, 8:15], [1., 2., 3., 4., 5., 6., 7.])
    np.testing.assert_array_equal(quadratic[0, 15:21], [4., 6., 8., 10., 12., 14.])
    np.testing.assert_array_equal(quadratic[0, -3:], [36., 42., 49.])


def test_ridge_penalizes_intercept_and_information_matches_rational_case():
    result = a.precision_fit(np.array([[1., -1.], [1., 1.]]), np.array([1., 3.]))
    np.testing.assert_array_equal(result['precision'], np.diag([2 + 1e-6, 2 + 1e-6]))
    np.testing.assert_array_equal(result['information'], [4., 2.])
    np.testing.assert_allclose(result['coefficient'], [4 / (2 + 1e-6), 2 / (2 + 1e-6)], atol=1e-15)
    assert result['coefficient'][0] < 2.


def test_release_at_boundary_is_not_in_fit_and_is_used_before_prediction():
    args = example()
    result = run_example(*args)
    timestamps, x, y, eligible, mean, scale = args
    first_due = int(np.flatnonzero(timestamps == -24)[0])
    current = int(np.flatnonzero(timestamps == 0)[0])
    design = a.design(x, mean, scale, 'linear7')
    initial = a.precision_fit(design[eligible], y[eligible])
    expected_a = initial['precision'] + np.outer(design[first_due], design[first_due])
    expected_b = initial['information'] + design[first_due] * y[first_due]
    expected_prediction = design[current] @ np.linalg.solve(expected_a, expected_b)
    assert result['predictions']['rls-linear7-lambda1'][current] == pytest.approx(expected_prediction, rel=1e-7, abs=1e-7)
    assert result['assimilated_indices'][0] == first_due
    assert result['predictions']['persistence'][current] == y[first_due]
    bad = eligible.copy()
    bad[first_due] = True
    with pytest.raises(ValueError, match='strictly before'):
        run_example(timestamps, x, y, bad, mean, scale)


def test_discount_every_clock_hour_including_gap_and_initial_hour():
    args = example()
    result = run_example(*args)
    timestamps, x, y, eligible, mean, scale = args
    design = a.design(x, mean, scale, 'linear7')
    initial = a.precision_fit(design[eligible], y[eligible])
    decay = .995
    expected_a = decay**4 * initial['precision']
    expected_b = decay**4 * initial['information']
    for hour in range(4):
        position = int(np.flatnonzero(timestamps == hour - 24)[0])
        expected_a += decay**(3 - hour) * np.outer(design[position], design[position])
        expected_b += decay**(3 - hour) * design[position] * y[position]
    final = result['final']['rls-linear7-lambda.995']
    np.testing.assert_allclose(final['precision'], expected_a, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(final['information'], expected_b, rtol=1e-14, atol=1e-14)
    assert result['clock_hours'] == 4
    assert 1 not in timestamps
    assert result['prediction_indices'].size == 3


def test_lambda_one_is_batch_fit_of_each_newly_released_unique_label_once():
    args = example()
    result = run_example(*args)
    timestamps, x, y, _eligible, mean, scale = args
    all_eligible = (timestamps + 24 < 4)
    for kind in ('linear7', 'quadratic7'):
        reference = a.precision_fit(a.design(x[all_eligible], mean, scale, kind), y[all_eligible])
        final = result['final']['rls-' + kind + '-lambda1']
        np.testing.assert_array_equal(final['precision'], reference['precision'])
        np.testing.assert_array_equal(final['information'], reference['information'])
    assert len(set(result['assimilated_indices'])) == 4


def test_unreleased_labels_never_affect_predictions_or_final_statistics():
    args = example()
    before = run_example(*args)
    timestamps, x, y, eligible, mean, scale = args
    changed = y.copy()
    changed[timestamps + 24 >= 4] = 1e9
    after = run_example(timestamps, x, changed, eligible, mean, scale)
    for name in a.METHODS:
        np.testing.assert_array_equal(before['predictions'][name], after['predictions'][name])
    for name in before['final']:
        for key in before['final'][name]:
            np.testing.assert_array_equal(before['final'][name][key], after['final'][name][key])


def test_missing_input_skips_rls_but_valid_due_label_updates_persistence():
    timestamps, x, y, eligible, mean, scale = example()
    x[timestamps == -24, 2] = np.nan
    result = run_example(timestamps, x, y, eligible, mean, scale)
    first_due = int(np.flatnonzero(timestamps == -24)[0])
    current = int(np.flatnonzero(timestamps == 0)[0])
    assert first_due not in result['assimilated_indices']
    assert first_due in result['label_release_indices']
    assert result['predictions']['persistence'][current] == y[first_due]


def test_target_missingness_does_not_suppress_a_current_public_prediction():
    timestamps, x, y, eligible, mean, scale = example()
    y[timestamps >= 0] = np.nan
    result = run_example(timestamps, x, y, eligible, mean, scale)
    assert all(np.isfinite(values[timestamps >= 0]).all() for values in result['predictions'].values())


def test_future_inputs_cannot_change_earlier_prediction_and_inputs_owned():
    args = example()
    copied = copy.deepcopy(args)
    before = run_example(*args)
    timestamps, x, y, eligible, mean, scale = args
    changed = x.copy()
    changed[timestamps >= 2] += 100.
    after = run_example(timestamps, changed, y, eligible, mean, scale)
    for name in a.METHODS:
        np.testing.assert_array_equal(before['predictions'][name][timestamps < 2], after['predictions'][name][timestamps < 2])
    for original, copy_ in zip(args, copied, strict=True):
        np.testing.assert_array_equal(original, copy_)


def test_check_exception_propagates_without_mutating_inputs():
    class Stop(Exception):
        pass
    args = example()
    copied = copy.deepcopy(args)
    timestamps, x, y, eligible, mean, scale = args
    def check():
        raise Stop('deadline')
    with pytest.raises(Stop, match='deadline'):
        a.replay(timestamps, x, y, start_hour=0, end_hour=4, mean=mean, scale=scale, fit_mask=eligible, check=check)
    for original, copy_ in zip(args, copied, strict=True):
        np.testing.assert_array_equal(original, copy_)


@pytest.mark.parametrize('change', [lambda t, x, y: t.__setitem__(1, t[0]),
    lambda t, x, y: x.__setitem__((0, 0), np.inf), lambda t, x, y: y.__setitem__(0, np.inf)])
def test_invalid_clock_or_infinite_numeric_input_rejected(change):
    timestamps, x, y, *_ = example()
    change(timestamps, x, y)
    with pytest.raises(ValueError):
        a.clock_inputs(timestamps, x, y)


def csv_fixture(rows):
    header = ';'.join(a.COLUMNS) + ';;\n'
    output = []
    for date, clock, public, label in rows:
        values = ['UNUSED' for _ in a.COLUMNS]
        values[:2] = [date, clock]
        for name, value in zip(a.FEATURES, public, strict=True):
            values[a.COLUMNS.index(name)] = value
        values[a.COLUMNS.index('C6H6(GT)')] = label
        output.append(';'.join(values) + ';;\n')
    return (header + ''.join(output) + ';' * 16 + '\n').encode()


def test_parser_reads_train_only_and_retains_missingness_ids_and_gap():
    raw = csv_fixture([
        ('01/03/2004', '00.00.00', ['1,25'] * 7, '2,5'),
        ('01/03/2004', '02.00.00', ['-200', '1', '2', '3', '4', '', '6'], '-200,0'),
        ('01/07/2004', '00.00.00', ['NEVER PARSE AS NUMBER'] * 7, 'PRIVATE NOT NUMERIC'),
    ])
    result = a.parse_train(raw)
    np.testing.assert_array_equal(result['row_ids'], [1, 2])
    assert result['timestamp_hours'][1] - result['timestamp_hours'][0] == 2
    assert result['x'][0, 0] == 1.25 and result['y'][0] == 2.5
    np.testing.assert_array_equal(result['valid_inputs'][1], [False, True, True, True, True, False, True])
    assert not result['valid_target'][1] and np.isnan(result['y'][1])


def test_parser_rejects_numeric_corruption_only_when_train_is_admitted():
    raw = csv_fixture([('01/03/2004', '00.00.00', ['1'] * 7, 'UNPARSEABLE')])
    with pytest.raises(ValueError, match='TRAIN value'):
        a.parse_train(raw)
    duplicate = csv_fixture([('01/03/2004', '00.00.00', ['1'] * 7, '1')] * 2)
    with pytest.raises(ValueError, match='chronology'):
        a.parse_train(duplicate)


def score_fixture():
    timestamps = np.arange(1024, dtype=np.int64)
    x = np.ones((1024, 7))
    y = np.zeros(1024)
    predictions = {method: np.ones(1024) for method in a.METHODS}
    predictions['rls-linear7-lambda1'][:] = .5
    states = {'fit_mask': timestamps < 512, 'fit_target_std': np.asarray(10.)}
    return timestamps, x, y, predictions, states


def score_fixture_result(args):
    return a.score(*args, start_hour=512, june_hour=768, end_hour=1024)


def test_six_conditions_include_same_adaptive_configuration_both_months():
    args = score_fixture()
    result = score_fixture_result(args)
    assert result['passed'] == result['total'] == 6
    assert result['outcome'] == 'QUALIFIES_LEARNED_MEMORY_SCREEN'
    assert result['adaptive_qualifiers'] == ['rls-linear7-lambda1']
    predictions = args[3]
    predictions['rls-linear7-lambda1'][768:] = 1.
    predictions['rls-linear7-lambda.995'][768:] = .5
    changed = score_fixture_result(args)
    assert changed['passed'] == 5 and changed['adaptive_qualifiers'] == []
    assert changed['outcome'] == 'REJECT_BENZENE_MEMORY_BENCHMARK'


def test_persistence_cannot_rescue_missing_adaptive_gain_and_static_floor_strict():
    args = score_fixture()
    for method in a.METHODS:
        args[3][method][:] = 1.
    args[3]['persistence'][:] = 0.
    result = score_fixture_result(args)
    assert result['passed'] == 5 and not result['adaptive_qualifiers']
    # Dyadic error and exact binary threshold avoid an incidental .1 RMS roundoff.
    args[4]['fit_target_std'] = np.asarray(12.5)
    for method in a.KINDS:
        args[3][method][:] = .125
    result = score_fixture_result(args)
    assert result['passed'] == 3
    assert not result['conditions'][3]['passed'] and not result['conditions'][4]['passed']


def test_all_complete_case_rows_scored_and_support_not_rescued():
    args = score_fixture()
    args[1][513, 0] = np.nan
    result = score_fixture_result(args)
    assert result['month_rows']['may'] == 255 and not result['conditions'][1]['passed']
    assert result['metrics']['may']['s2linear']['rmse'] == 1.


def test_state_export_pending_queue_and_actual_dense_resource_bytes():
    args = example()
    timestamps, x, y, *_ = args
    predictions, states, resources = a.reconstruct(timestamps, x, y, start_hour=0, end_hour=4)
    assert len(predictions) == 9 and len(states) == 37
    assert resources['s2linear'] == 129 and resources['s2quadratic'] == 137
    assert resources['linear7'] == 177 and resources['quadratic7'] == 401
    assert resources['rls-linear7-lambda1'] == 2049
    assert resources['rls-quadratic7-lambda.995'] == 12129
    assert resources['persistence'] == 17
    np.testing.assert_array_equal(states['initial_queue'][0], x[timestamps == -24][0])
    np.testing.assert_array_equal(states['final-rls-linear7-lambda1-queue'][0], x[timestamps == 0][0])
    assert np.isnan(states['final-rls-linear7-lambda1-queue'][1]).all()
    assert states['final-rls-linear7-lambda1-clock'] == 4


def test_saved_values_reject_omitted_predictions_nonfinite_and_wrong_queue():
    actual = np.array([1., np.nan])
    a.array_agreement(actual, actual.copy(), 'correct')
    with pytest.raises(ValueError):
        a.array_agreement(np.array([np.nan, np.nan]), actual, 'omitted')
    with pytest.raises(ValueError):
        a.array_agreement(np.array([np.inf, np.nan]), actual, 'infinite')
    with pytest.raises(ValueError):
        a.array_agreement(np.array([-0.]), np.array([0.]), 'raw queue', exact=True)


def write_json(path, value):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def receipt_fixture(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    wrapper = root / 'output/sensor-screen-engineering-v1/invoke.py'
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text('opaque wrapper')
    monkeypatch.setattr(a, 'ROOT', root)
    path = tmp_path / 'qualification.json'
    log = tmp_path / 'qualification.log'
    log.write_text('fabricated pass log')
    sources = {name: {'sha256': 'a' * 64, 'bytes': 1} for name in a.SOURCES}
    argv = ['.venv/bin/python', '-m', 'pytest', '-q', 'tests/test_sensor_data.py',
            'tests/test_sensor_screen.py', 'tests/test_audit_sensor_screen.py']
    receipt = {'state': 'EXITED', 'returncode': 0, 'argv': argv.copy(), 'elapsed_seconds': 1.,
               'thread_env': {name: '1' for name in a.THREADS}, 'sources_before': sources,
               'sources_after': copy.deepcopy(sources), 'log_path': 'qualification.log',
               'log': a.descriptor(log), 'timeout_seconds': 300, 'wrapper': a.descriptor(wrapper)}
    write_json(path, receipt)
    return path, receipt, sources, argv


@pytest.mark.parametrize('change', [lambda r: r.update(returncode=False), lambda r: r.update(state='TIMEOUT'),
    lambda r: r.update(elapsed_seconds=301.), lambda r: r.update(timeout_seconds=301),
    lambda r: r['argv'].append('unregistered_test.py'), lambda r: r['sources_after'].pop(min(a.SOURCES)),
    lambda r: r['thread_env'].update(OMP_NUM_THREADS='2'), lambda r: r.update(log_path='../qualification.log'),
    lambda r: r['log'].update(sha256='0' * 64), lambda r: r['wrapper'].update(bytes=0)])
def test_original_receipt_rejects_wrong_command_sources_log_timeout_before_arrays(tmp_path, monkeypatch, change):
    path, receipt, sources, argv = receipt_fixture(tmp_path, monkeypatch)
    assert a.closed_receipt(path, sources, argv) == receipt
    monkeypatch.setattr(a.np, 'load', lambda *_args, **_kwargs: pytest.fail('metadata-only boundary'))
    change(receipt)
    write_json(path, receipt)
    with pytest.raises(ValueError):
        a.closed_receipt(path, sources, argv)
