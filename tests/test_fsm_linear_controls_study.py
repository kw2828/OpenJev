"""Fabricated qualification for stronger conventional controls, never measurements."""
import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from openjev.research import fsm_data, fsm_linear
from openjev.research import fsm_linear_controls_study as study
from openjev.research.fsm_residual import FSMResidual


def fixture():
    n, c, r, p = np.indices((240, 3, 6, 2))
    data = fsm_data.records_from_fixture({
        f'{key}_{amp}_train': np.asarray(np.sin(n*.071+c*.13+r*.79+p*.33)
                                         +(.2 if key == 'y' else 0), dtype=np.float64)
        for amp in fsm_data.AMPLITUDES for key in ('u', 'y')})
    norm = fsm_data.fit_normalizer(data.partition('fit'))
    return data, norm, study.normalized_records(data.partition('fit'), norm)


@pytest.mark.parametrize('order', study.ORDERS)
def test_shared_statistics_exactly_match_existing_ridge_and_preserve_inputs(order):
    _, _, records = fixture()
    before = [(r.u.copy(), r.y.copy()) for r in records]
    stats = study.sufficient_statistics(records, order)
    reverse = study.sufficient_statistics(tuple(reversed(records)), order)
    for a, b in zip(stats, reverse, strict=True):
        np.testing.assert_array_equal(a, b)
    for alpha in study.ALPHAS:
        model, error = study.solve_statistics(*stats, order=order, alpha=alpha)
        legacy = fsm_linear.fit_varx(records, order=order, alpha=alpha)
        np.testing.assert_array_equal(model.coefficients, legacy.coefficients)
        assert model.fit_rows == 12*(240-order) and error <= 1e-10
    for r, (u, y) in zip(records, before, strict=True):
        np.testing.assert_array_equal(r.u, u); np.testing.assert_array_equal(r.y, y)


def test_statistics_reject_dev_and_duplicate_records():
    data, _, records = fixture()
    with pytest.raises(ValueError): study.sufficient_statistics(data.partition('dev'), 32)
    with pytest.raises(ValueError): study.sufficient_statistics(records[:-1]+(records[0],), 32)


def test_overflowed_certificate_norm_cannot_turn_into_zero_error():
    with pytest.raises(ValueError, match='nonfinite ridge certificate norm'):
        study.solve_statistics(np.eye(196)*1e200, np.ones((196, 3))*1e100,
                               12*208, order=32, alpha=1e-6)


@pytest.mark.parametrize('damage', ['nan', 'asymmetric', 'bad_dtype', 'negative_alpha', 'singular'])
def test_invalid_statistics_cannot_be_repaired(damage):
    gram, cross, rows = np.eye(196), np.ones((196, 3)), 12*208
    alpha = 1e-6
    if damage == 'nan': gram[0, 0] = np.nan
    elif damage == 'asymmetric': gram[0, 1] = 1.
    elif damage == 'bad_dtype': gram = gram.astype(np.float32)
    elif damage == 'negative_alpha': alpha = -1.
    elif damage == 'singular': gram[-1, -1] = 0.
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        study.solve_statistics(gram, cross, rows, order=32, alpha=alpha)


@pytest.mark.parametrize('kind', ['no_stats', 'timeout', 'singular'])
def test_failed_linear_fit_receipt_retains_reason_and_any_checkpoint(tmp_path, kind):
    stats = (np.eye(196), np.ones((196, 3)), 12*208)
    cap = 120.
    if kind == 'no_stats': stats = None
    elif kind == 'timeout': cap = -1.
    else: stats[0][-1, -1] = 0.
    model, receipt = study.fit_linear('varx32-ridge1e-06', stats, {'solve_timeout_seconds': cap}, tmp_path/'fit')
    assert model is None and receipt['status'] == 'failed' and receipt['error']
    assert (tmp_path/'fit/receipt.json').is_file()
    assert (tmp_path/'fit/model.npz').exists() == (kind == 'timeout')


def results():
    fits = [{'family': f, 'status': 'complete'} for f in study.LINEAR]
    folds = [{'family': study.FOLDED, 'seed': s, 'status': 'complete', 'parity': {'passed': True}} for s in study.SEEDS]
    evaluations = []
    for family in study.FAMILIES:
        seeds = (None,) if family in (*study.LINEAR, study.NATIVE) else study.SEEDS
        for seed in seeds:
            error = .8 if family == study.SELECTED['tanh_feedback'] else 1.
            evaluations.append({'family': family, 'seed': seed,
                'rows': [{'record_id': rid, 'status': 'complete', 'rmse': error} for rid in study.DEV_IDS],
                'request_ms': [1.]*24, 'median_request_ms': 1., 'timing_error': None,
                'persistent_numeric_bytes': 100})
    return fits, folds, evaluations


def test_global_stronger_linear_control_can_defeat_previously_selected_candidate():
    fits, folds, evaluations = results()
    for e in evaluations:
        if e['family'] == 'varx96-ridge1e-06':
            for row in e['rows']: row['rmse'] = .5
    summary = study.summarize(evaluations, fits, folds)
    assert summary['strongest_control'] == 'varx96-ridge1e-06'
    assert summary['candidate'] == study.SELECTED['tanh_feedback']
    assert not summary['conditions']['five_percent_below_strongest_control']
    assert not summary['conditions']['every_seed_below_strongest_control']
    assert summary['status'] == 'DEVELOPMENT_FAIL'


def test_timing_failure_never_replaces_strongest_accuracy_reference():
    fits, folds, evaluations = results()
    for e in evaluations:
        if e['family'] == 'varx96-ridge1e-06':
            for row in e['rows']: row['rmse'] = .5
            e['request_ms'] = []; e['median_request_ms'] = None; e['timing_error'] = 'clock failure'
    summary = study.summarize(evaluations, fits, folds)
    assert summary['strongest_control'] == 'varx96-ridge1e-06'
    assert not summary['conditions']['all_25_evaluations_complete']
    assert not summary['conditions']['five_percent_below_strongest_control']


@pytest.mark.parametrize('kind', ['missing_fit', 'duplicate_fit', 'fit_failed', 'missing_eval', 'duplicate_eval',
    'duplicate_record', 'nonfinite_error', 'missing_timing', 'missing_fold', 'duplicate_fold', 'failed_fold', 'parity_failure'])
def test_failed_rosters_or_partial_evidence_never_pass(kind):
    fits, folds, evaluations = results()
    if kind == 'missing_fit': fits.pop()
    elif kind == 'duplicate_fit': fits[0] = fits[1]
    elif kind == 'fit_failed': fits[0]['status'] = 'failed'
    elif kind == 'missing_eval': evaluations.pop()
    elif kind == 'duplicate_eval': evaluations[0] = evaluations[1]
    elif kind == 'duplicate_record': evaluations[0]['rows'][0] = evaluations[0]['rows'][1]
    elif kind == 'nonfinite_error': evaluations[0]['rows'][0]['rmse'] = np.nan
    elif kind == 'missing_timing': evaluations[0]['request_ms'].pop()
    elif kind == 'missing_fold': folds.pop()
    elif kind == 'duplicate_fold': folds[0] = folds[1]
    elif kind == 'failed_fold': folds[0]['status'] = 'failed'
    elif kind == 'parity_failure': folds[0]['parity']['passed'] = False
    assert study.summarize(evaluations, fits, folds)['status'] == 'DEVELOPMENT_FAIL'


def test_all_gates_pass_and_reference_identity_is_global():
    fits, folds, evaluations = results()
    summary = study.summarize(evaluations, fits, folds)
    assert summary['status'] == 'DEVELOPMENT_PASS' and summary['passed'] == 9
    assert summary['strongest_control'] == min(f for f in study.FAMILIES if f != study.SELECTED['tanh_feedback'])
    other = study.summarize(list(reversed(evaluations)), list(reversed(fits)), list(reversed(folds)))
    assert other == summary


def test_pooled_seed_success_cannot_hide_one_bad_candidate_seed():
    fits, folds, evaluations = results()
    for e in evaluations:
        if e['family'] == study.SELECTED['tanh_feedback']:
            for row in e['rows']: row['rmse'] = 1.05 if e['seed'] == 9201 else .1
    summary = study.summarize(evaluations, fits, folds)
    assert summary['conditions']['five_percent_below_strongest_control']
    assert not summary['conditions']['every_seed_below_strongest_control']


@pytest.mark.parametrize('perturb', [False, True])
def test_parity_preserves_mismatching_forecast_bank_without_timing(tmp_path, perturb):
    data, norm, _ = fixture()
    coefficients = np.zeros((3, 196)); coefficients[:, 96:99] = np.eye(3)*.3
    model = fsm_linear.VARXModel(coefficients, 32, 1e-6, 12*208, fsm_linear.FIT_IDS)
    unfused = tmp_path/'unfused'; unfused.mkdir()
    cfg = {'context': 100, 'horizon': 128, 'parity_atol': 1e-9, 'parity_rtol': 1e-9}
    for record in data.partition('dev'):
        inputs, target = study.normalized_request(record, np.array([0]), 100, 128, norm)
        expected = fsm_linear.predict(model, *inputs)
        if perturb: expected[:, -1, 1] += 1e-4
        np.savez(unfused/(record.record_id+'.npz'), prediction=expected, target=target, starts=np.array([0]))
    receipt = study.folded_parity(model, unfused, data.partition('dev'), cfg, norm, tmp_path/'parity')
    assert receipt['passed'] is (not perturb)
    assert len(list((tmp_path/'parity').glob('*.npz'))) == 12
    assert all(r['passed'] is (not perturb) for r in receipt['records'])


def test_failed_fold_is_ineligible_even_if_predicted_rows_look_complete():
    fits, folds, evaluations = results(); folds[0]['parity']['passed'] = False
    other = copy.deepcopy(evaluations)
    for e in other:
        if e['family'] == study.FOLDED:
            for r in e['rows']: r['rmse'] = .01
    summary = study.summarize(other, fits, folds)
    assert not summary['families'][study.FOLDED]['score_eligible']
    assert summary['strongest_control'] != study.FOLDED


@pytest.mark.parametrize('mode', ['complete', 'fold_failure', 'folded_io_failure'])
def test_fabricated_pipeline_closes_fits_before_dev_and_preserves_failures(tmp_path, monkeypatch, mode):
    data, norm, _ = fixture()
    coefficients = np.zeros((3, 196)); coefficients[:, 96:99] = np.eye(3)*.3
    backbone = fsm_linear.VARXModel(coefficients, 32, 1e-6, 12*208, fsm_linear.FIT_IDS)
    models = {}
    for arch, family in study.SELECTED.items():
        kind, placement = arch.split('_', 1)
        for seed in study.SEEDS:
            models[(family, seed)] = FSMResidual(coefficients, order=32, mode=placement,
                                               residual_kind=kind, seed=seed)
    cfg = {'orders': list(study.ORDERS), 'alphas': list(study.ALPHAS), 'seeds': list(study.SEEDS),
        'selected_parent_recipes': study.SELECTED, 'context': 100, 'horizon': 128, 'evaluation_stride': 256,
        'parity_atol': 1e-9, 'parity_rtol': 1e-9, 'statistics_timeout_seconds': 120., 'solve_timeout_seconds': 120.}
    data_path = tmp_path/'fabricated.txt'; data_path.write_text('synthetic fixture only')
    plan = tmp_path/'plan.json'
    study.write(plan, {'experiment': cfg, 'parent_artifacts': {}, 'source_sha256': {},
                      'data_sha256': study.sha(data_path)})
    monkeypatch.setattr(study, 'validate_pins', lambda protocol: None)
    monkeypatch.setattr(study, 'load_parent', lambda protocol: (backbone, norm))
    monkeypatch.setattr(study, 'parent_models', lambda protocol, backbone: (models, []))
    monkeypatch.setattr(study.fsm_data, 'read_npz_estimation', lambda path: replace(data, source_sha256=study.sha(path)))
    out = tmp_path/'run'; original_evaluate = study.evaluate; original_fold = study.fold_affine_feedback
    def guarded_evaluate(model, receipt, records, config, normalizer, directory, **kwargs):
        assert (out/'fits-closed.json').is_file()
        assert len(json.loads((out/'fits.json').read_text())) == 9
        if receipt['family'] == study.FOLDED:
            assert json.loads((directory.parent/'parity.json').read_text())['passed']
            if mode == 'folded_io_failure':
                directory.mkdir()
                raise OSError('original artificial write failure')
        return original_evaluate(model, receipt, records, config, normalizer, directory, **kwargs)
    def guarded_fold(model, parent):
        if mode == 'fold_failure' and model.initialization_seed == 9202:
            raise ValueError('artificial failed fold')
        return original_fold(model, parent)
    monkeypatch.setattr(study, 'evaluate', guarded_evaluate)
    monkeypatch.setattr(study, 'fold_affine_feedback', guarded_fold)
    if mode == 'folded_io_failure':
        with pytest.raises(OSError, match='original artificial write failure'):
            study.run(plan, data_path, out)
        return
    study.run(plan, data_path, out)
    evaluations = json.loads((out/'evaluations.json').read_text())
    assert len(evaluations) == 25 and {(e['family'], e['seed']) for e in evaluations} == study.ROSTER
    folds = json.loads((out/'folds.json').read_text())
    assert len(folds) == 3
    if mode == 'fold_failure':
        failed = next(e for e in evaluations if e['family'] == study.FOLDED and e['seed'] == 9202)
        assert all(r['status'] == 'not_run' for r in failed['rows'])
        assert failed['request_ms'] == []
        assert next(f for f in folds if f['seed'] == 9203)['status'] == 'complete'
    else:
        assert all(f['status'] == 'complete' and f['parity']['passed'] for f in folds)
