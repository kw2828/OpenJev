"""Synthetic saved-conditioning witnesses, without training or scientific inputs."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_conditioning_audit_tests', ROOT/'scripts/audit_otto_conditioning.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def archive(kind='gain1'):
    rng = np.random.default_rng(11)
    w = rng.normal(0, .01, (8, A.DIMENSION)).astype(np.float32)
    if kind == 'gain53':
        w[:, :A.SPATIAL_DIMENSION] /= np.float32(53)
    return {'version': np.asarray('otto-conditioned-value-v1'), 'kind': np.asarray(kind),
            'input_dim': np.asarray(A.DIMENSION), 'spatial_gain': np.asarray(A.GAINS[kind]),
            'c0': np.asarray(.25, np.float32), 'weight_0': w, 'bias_0': np.zeros(8, np.float32),
            'weight_1': rng.normal(0, .1, (1, 8)).astype(np.float32), 'bias_1': np.zeros(1, np.float32)}


def test_initial_pair_exact_spatial_division_and_unchanged_other_tensors():
    raw, scaled = archive(), archive('gain53')
    assert A.initial_pair(raw, scaled, np)
    assert np.array_equal(raw['weight_0'][:, -3:], scaled['weight_0'][:, -3:])
    assert A.count_parameters(A.checkpoint(raw, 'gain1', .25, np)) == 88241
    assert A.count_parameters(A.checkpoint(scaled, 'gain53', .25, np)) == 88241


@pytest.mark.parametrize('defect', ['context', 'uncompensated', 'overcompensated', 'output', 'bias', 'baseline'])
def test_initial_pair_rejects_changes_beyond_fixed_conditioning(defect):
    raw, scaled = archive(), archive('gain53')
    if defect == 'context':
        scaled['weight_0'][:, -3:] /= 53
    elif defect == 'uncompensated':
        scaled['weight_0'] = raw['weight_0'].copy()
    elif defect == 'overcompensated':
        scaled['weight_0'][:, :11025] /= 53
    elif defect == 'output':
        scaled['weight_1'][0, 0] += 1
    elif defect == 'bias':
        raw['bias_0'][0] = scaled['bias_0'][0] = 1
    else:
        scaled['c0'] = np.asarray(.5, np.float32)
    with pytest.raises(ValueError):
        A.initial_pair(raw, scaled, np)


@pytest.mark.parametrize('defect', ['version', 'gain', 'boolean_gain', 'extra', 'missing', 'shape', 'dtype', 'nan', 'c0'])
def test_checkpoint_rejects_unqualified_schema(defect):
    value = archive('gain53')
    if defect == 'version':
        value['version'] = np.asarray('capacity')
    elif defect == 'gain':
        value['spatial_gain'] = np.asarray(1)
    elif defect == 'boolean_gain':
        value['spatial_gain'] = np.asarray(True)
    elif defect == 'extra':
        value['optimizer'] = np.asarray(0)
    elif defect == 'missing':
        del value['weight_1']
    elif defect == 'shape':
        value['weight_0'] = value['weight_0'].T
    elif defect == 'dtype':
        value['weight_0'] = value['weight_0'].astype(np.float64)
    elif defect == 'nan':
        value['weight_0'][0, 0] = np.nan
    else:
        value['c0'] = np.asarray(.5, np.float32)
    with pytest.raises(ValueError):
        A.checkpoint(value, 'gain53', .25, np)


@pytest.mark.parametrize('kind', ['gain1', 'gain53'])
def test_explicit_scalar_oracle_separates_scaled_spatial_raw_context_and_baseline(kind):
    x = np.zeros((4, A.DIMENSION), np.float64)
    x[:, :2] = [[.25, .75], [1., 0.], [0., 0.], [.125, .375]]
    x[:, -3:] = [[.5, .25, .6], [1., .5, .8], [0., 0., 0.], [.25, .125, .3]]
    w0 = np.zeros((8, A.DIMENSION), np.float64)
    w0[0, [0, 1, -3, -2, -1]] = [2., -1., 3., -4., 5.]
    w0[1, [0, 1, -1]] = [-1., 2., -2.]
    b0 = np.zeros(8); b0[:2] = [.1, -.2]
    w1 = np.zeros((1, 8)); w1[0, :2] = [3., -2.]
    layers = [(w0, b0), (w1, np.asarray([-.75]))]
    original = x.tobytes()
    expected = []
    gain = A.GAINS[kind]
    for row in x:
        first = max(2*gain*row[0]-gain*row[1]+3*row[-3]-4*row[-2]+5*row[-1]+.1, 0)
        second = max(-gain*row[0]+2*gain*row[1]-2*row[-1]-.2, 0)
        expected.append(.25*(row[0]+row[1])+3*first-2*second-.75)
    actual = A.conditioned_predict(x, layers, .25, kind, np)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert x.tobytes() == original and actual[2] == pytest.approx(-.45)


def test_dense_asymmetric_pair_close_but_not_assumed_bit_identical():
    rng = np.random.default_rng(27)
    x = rng.uniform(0, 1, (7, A.DIMENSION))
    x[:, :11025] /= x[:, :11025].sum(axis=1, keepdims=True)
    original = x.copy()
    a, b = archive(), archive('gain53')
    left = A.conditioned_predict(x, A.checkpoint(a, 'gain1', .25, np), .25, 'gain1', np)
    right = A.conditioned_predict(x, A.checkpoint(b, 'gain53', .25, np), .25, 'gain53', np)
    np.testing.assert_allclose(left, right, atol=1e-6, rtol=1e-6)
    assert not np.array_equal(left, right)
    assert np.array_equal(x, original)


def test_batch_partition_matches_independent_single_row_readouts():
    rng = np.random.default_rng(4)
    x = rng.uniform(size=(19, A.DIMENSION)).astype(np.float64)
    layers = A.checkpoint(archive('gain53'), 'gain53', .25, np)
    together = A.conditioned_predict(x, layers, .25, 'gain53', np)
    separate = np.concatenate([A.conditioned_predict(row[None], layers, .25, 'gain53', np) for row in x])
    np.testing.assert_allclose(together, separate, atol=1e-10, rtol=1e-10)


def synthetic_fits(narrow_train=.1, scaled_train=.075, narrow_valid=.1, scaled_valid=.089):
    fits = []
    for seed in A.SEEDS:
        for kind in A.WIDTHS:
            fits.append({'fit_id': f'{kind}@{seed}', 'metrics': {
                'train': {'rows': 5589, 'mse_normalized': narrow_train if kind == 'gain1' else scaled_train},
                'valid': {'rows': 1109, 'mse_normalized': narrow_valid if kind == 'gain1' else scaled_valid}}})
    return fits


def test_six_criteria_are_paired_per_seed_and_apply_to_signed_excess():
    result = A.aggregate(synthetic_fits(), .05)
    assert len(result['checks']) == 6 and all(row['passes'] for row in result['checks'])
    assert result['conditioning_screen_passed'] is True
    assert [r['name'] for r in result['checks']] == [f'gain53.{seed}.{name}' for seed in A.SEEDS for name in ('train_excess', 'valid_mse')]
    below_floor = A.aggregate(synthetic_fits(narrow_train=.04, scaled_train=.045), .05)
    assert all(row['value'] < 0 and not row['passes'] for row in below_floor['checks'][::2])


def test_training_gain_alone_cannot_pass_validation_or_overall_rule():
    result = A.aggregate(synthetic_fits(scaled_train=.05, scaled_valid=.091), .05)
    assert [r['passes'] for r in result['checks']] == [True, False]*3
    assert result['conditioning_screen_passed'] is False


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'order', 'missing_split', 'missing_rows', 'nan'])
def test_aggregation_rejects_incomplete_or_malformed_saved_fits(defect):
    fits = synthetic_fits()
    if defect == 'missing':
        fits.pop()
    elif defect == 'duplicate':
        fits[-1] = fits[0]
    elif defect == 'order':
        fits.reverse()
    elif defect == 'missing_split':
        del fits[0]['metrics']['valid']
    elif defect == 'missing_rows':
        fits[0]['metrics']['valid']['rows'] = 1108
    else:
        fits[0]['metrics']['train']['mse_normalized'] = np.nan
    with pytest.raises(ValueError):
        A.aggregate(fits, .05)


def test_exact30_payloads_include_all18_checkpoints_predictions_and_pairing():
    names = A.payload_names()
    assert len(names) == 30 and 'initial-pairing.jsonl' in names
    assert {f'{phase}-{kind}-{seed}.npz' for seed in A.SEEDS for kind in A.WIDTHS
            for phase in ('initial', 'final', 'predictions')} <= names


def test_work_requires_complete_counts_for_six_fits_not_old_nine():
    work = A.Work([])
    expected = {'model_initialization': 6, 'optimizer_initialization': 6, 'checkpoint_export': 12,
                'optimizer_update': 21120, 'parity_restore': 6, 'parity_forward': 12, 'saved_prediction': 162,
                'initial_pair_initialization': 6, 'initial_pair_export': 6, 'initial_pair_forward': 132}
    work.counts = {name: {'attempted': n, 'returned': n, 'seconds': .1} for name, n in expected.items()}
    assert work.finish() == work.counts
    work.counts['model_initialization']['returned'] = 9
    with pytest.raises(ValueError, match='counts'):
        work.finish()


def pairing_fixture():
    import hashlib
    probe = np.zeros((5, A.DIMENSION), np.float64)
    predictions = {'gain1': np.asarray([0., -2., .1, 10., .05]),
                   'gain53': np.asarray([0., -2.+1e-8, .1, 10.-1e-8, .05])}
    record = {'seed': 10101, 'features_sha256': hashlib.sha256(probe.tobytes()).hexdigest(), 'rows': len(probe),
              'predictions': {k: v.tolist() for k, v in predictions.items()},
              'maximum_difference': float(np.abs(predictions['gain1']-predictions['gain53']).max()),
              'tolerance': {'absolute': 1e-6, 'relative': 1e-6}, 'passed': True,
              'initial_sha256': {'gain1': 'a'*64, 'gain53': 'b'*64}}
    return record, probe, predictions


def test_pairing_record_checks_tight_independent_replay_before_looser_equivalence():
    record, probe, predictions = pairing_fixture()
    assert A.pairing_record(record, 10101, probe, predictions, {'gain1': 'a'*64, 'gain53': 'b'*64}, np) == record['maximum_difference']
    record['predictions']['gain53'][1] += 1e-7  # Under pairing tolerance, above replay tolerance.
    with pytest.raises(ValueError, match='replay'):
        A.pairing_record(record, 10101, probe, predictions, {'gain1': 'a'*64, 'gain53': 'b'*64}, np)


@pytest.mark.parametrize('defect', ['subset', 'hash', 'tolerance', 'passed', 'maximum', 'missing_arm', 'nonfinite', 'checkpoint'])
def test_pairing_rejects_incomplete_or_repaired_witness(defect):
    record, probe, predictions = pairing_fixture()
    if defect == 'subset':
        record['rows'] -= 1
    elif defect == 'hash':
        record['features_sha256'] = 'bad'
    elif defect == 'tolerance':
        record['tolerance']['absolute'] = 1e-5
    elif defect == 'passed':
        record['passed'] = False
    elif defect == 'maximum':
        record['maximum_difference'] = -1e-15
    elif defect == 'missing_arm':
        del record['predictions']['gain1']
    elif defect == 'checkpoint':
        record['initial_sha256']['gain53'] = 'c'*64
    else:
        record['predictions']['gain53'][0] = float('nan')
    with pytest.raises(ValueError):
        A.pairing_record(record, 10101, probe, predictions, {'gain1': 'a'*64, 'gain53': 'b'*64}, np)


def test_pairing_reference_is_gain1_and_original_tolerance_is_not_relaxed():
    record, probe, predictions = pairing_fixture()
    predictions['gain53'][1] = -2.+4e-6
    record['predictions']['gain53'] = predictions['gain53'].tolist()
    record['maximum_difference'] = float(np.abs(predictions['gain1']-predictions['gain53']).max())
    with pytest.raises(ValueError, match='predicate'):
        A.pairing_record(record, 10101, probe, predictions, {'gain1': 'a'*64, 'gain53': 'b'*64}, np)


def test_pending_initial_pair_call_is_not_counted_as_successful_work():
    context = {'phase': 'initial_pair', 'seed': 10101, 'kind': 'gain1', 'offset': 0}
    work = A.Work([{'id': 1, 'event': 'attempt', 'channel': 'initial_pair_forward', 'context': context}])
    with pytest.raises(StopIteration):
        work.take('initial_pair_forward', context)
    assert work.counts == {}


def test_completed_output_requires_initial_pair_payload(tmp_path):
    names = A.payload_names()
    for name in names:
        (tmp_path/name).write_bytes(b'fabricated payload')
    A.write(tmp_path/'receipt.json', {'status': 'completed', 'files': {n: A.digest(tmp_path/n) for n in names}})
    pin = A.digest(tmp_path/'receipt.json')['sha256']
    assert A.manifest(tmp_path, pin, names, lambda: None)['status'] == 'completed'
    (tmp_path/'initial-pairing.jsonl').unlink()
    with pytest.raises(ValueError, match='closed'):
        A.manifest(tmp_path, pin, names, lambda: None)


def test_clock_failure_leaves_explicit_failure_evidence(tmp_path, monkeypatch):
    import argparse
    from types import SimpleNamespace
    def unavailable():
        raise RuntimeError('synthetic unavailable clock')
    monkeypatch.setattr(A, 'load', lambda *_: SimpleNamespace(SuspendClock=unavailable))
    audit = A.Audit(argparse.Namespace(output=tmp_path/'audit'))
    with pytest.raises(RuntimeError, match='unavailable clock'):
        audit.execute()
    failed = A.read(audit.out/'failed.json')
    assert failed['status'] == 'failed' and failed['started_ns'] is None and failed['wall_seconds'] is None
    assert failed['independent_readout_attempts'] == failed['independent_returned_rows'] == 0
