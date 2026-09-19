"""Causal-label and orchestration checks; no real model/environment calls."""
import json
from pathlib import Path

import card_calibration_common as common
import fit_card_calibration as fit
import numpy as np
import pytest


def public_arrays():
    obs = np.full((5, 52), 13, dtype=np.int64)
    obs[1, 0] = 1
    obs[2, :2] = [1, 2]
    obs[3, 2] = 3
    obs[4, 2:4] = [3, 4]
    return {'observations': obs, 'raw_probabilities': np.full((4, 52, 13), 1 / 13, dtype=np.float64)}


def test_labels_precede_reveal_and_use_last_public_visibility():
    ep = common.public_episode(public_arrays())
    assert not ep['target_mask'][:3].any()
    assert np.flatnonzero(ep['target_mask'][3]).tolist() == [0, 1]
    assert ep['targets'][3, :2].tolist() == [1, 2]
    assert ep['ages'][3, :2].tolist() == [1, 1]
    assert ep['targets'][3, 3] == -1  # Rank4 appears only AFTER this decision.


def test_future_public_values_cannot_change_earlier_targets():
    a = public_arrays(); b = public_arrays()
    b['observations'][4, 3] = 9
    first, second = common.public_episode(a), common.public_episode(b)
    for key in ('targets', 'target_mask', 'ages'):
        np.testing.assert_array_equal(first[key], second[key])


def test_changed_public_rank_is_rejected():
    arrays = public_arrays(); arrays['observations'][2, 0] = 7
    with pytest.raises(ValueError, match='changed rank'):
        common.public_episode(arrays)


def test_input_hash_binds_values_schema_and_episode_order():
    a = common.public_episode(public_arrays())
    b = {k: v.copy() for k, v in a.items()}
    b['raw_probabilities'][0, 0, :2] += [.01, -.01]
    assert fit.input_digest([a, b]) != fit.input_digest([b, a])
    assert fit.input_digest([a]) == fit.input_digest([{k: v.copy() for k, v in a.items()}])


def test_expected_calibration_closure():
    expected = common.calibration_members()
    assert len(expected) == 55
    assert 'fits/kalman-pair1/completed.json' in expected
    assert all('exact' not in name and 'last32' not in name for name in expected)


@pytest.fixture
def fake_run(monkeypatch, tmp_path):
    source = tmp_path / 'old'; source.mkdir()
    (source / 'completed.json').write_text('{}')
    (source / 'inputs.json').write_text('{}')
    recipe = {'lineage': {'controller_completed': {'path': 'old/completed.json', 'sha256': 'prior'},
                          'controller_inputs': {'path': 'old/inputs.json', 'sha256': 'inputs'}}}
    checkpoints = {name: {'checkpoint_sha256': 'weights-' + name} for name in common.LEARNED}
    monkeypatch.setattr(fit, 'authenticate', lambda *args: (recipe, {}, checkpoints, {}))
    episode = common.public_episode(public_arrays())
    monkeypatch.setattr(fit, 'collect', lambda *args, **kwargs: ([episode] * 64, [{'index': i} for i in range(64)]))
    monkeypatch.setattr(fit, 'fit_temperature', lambda episodes: {
        'status': 'fitted', 'beta': 1.0, 'temperature': 1.0, 'oracle_evaluations': 3,
        'baseline_nll': 2.0, 'calibrated_nll': 2.0,
        'counts': {'episodes': 64, 'eligible_episodes': 64, 'query_cards': 128}})
    monkeypatch.setattr(fit, 'score_episodes', lambda *args, **kwargs: {'all': {'nll': 2.0}})
    args = (tmp_path, tmp_path / 'protocol', 'p', tmp_path / 'inputs', 'i',
            tmp_path / 'bindings', 'b', tmp_path / 'map', 'm')
    return args


def test_complete_fit_manifest_and_authenticated_scalar_mapping(fake_run, tmp_path):
    out = tmp_path / 'calibration'
    done = fit.run(*fake_run, out)
    assert done['source_episodes'] == 1152 and done['fitted_parameters'] == 18
    assert done['oracle_evaluations'] == 54 and done['scoring_calls'] == 54
    assert set(done['files']) == common.calibration_members()
    checked = common.authenticate_calibration(out, fit.sha(out / 'completed.json'), 'p', 'i', 'b', 'm')
    assert checked == done


def test_changed_fitted_scalar_cannot_pass_closure(fake_run, tmp_path):
    out = tmp_path / 'calibration'
    fit.run(*fake_run, out)
    digest = fit.sha(out / 'completed.json')
    path = out / 'fits/kalman-pair1/fit.json'
    data = json.loads(path.read_text()); data['beta'] = 2.0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='member changed'):
        common.authenticate_calibration(out, digest, 'p', 'i', 'b', 'm')


def test_partial_fit_failure_preserves_completion_counts_and_stops(fake_run, tmp_path, monkeypatch):
    calls = 0
    original = fit.fit_temperature

    def fail_on_second(episodes):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ArithmeticError('fixture fitter failure')
        return original(episodes)

    monkeypatch.setattr(fit, 'fit_temperature', fail_on_second)
    out = tmp_path / 'failed-fit'
    with pytest.raises(ArithmeticError, match='fixture fitter failure'):
        fit.run(*fake_run, out)
    failed = json.loads((out / 'failed.json').read_text())
    assert len(failed['completed_fits']) == 1 and failed['oracle_evaluations_returned'] == 3
    assert failed['active_returned_fit'] is None and calls == 2
    assert failed['unfinished_oracle_work'] == 'unknown'
    assert not (out / 'completed.json').exists()


def test_exclusive_outputs_are_not_resumed(fake_run, tmp_path):
    out = tmp_path / 'prior'; out.mkdir()
    with pytest.raises(FileExistsError):
        fit.run(*fake_run, out)


def test_late_storage_failure_demotes_completion(fake_run, tmp_path, monkeypatch):
    original = fit.dump

    def late_failure(path, value):
        original(path, value)
        if Path(path).name == 'completed.json' and Path(path).parent.name == 'late':
            raise OSError('fixture late completion failure')

    monkeypatch.setattr(fit, 'dump', late_failure)
    out = tmp_path / 'late'
    with pytest.raises(OSError, match='late completion'):
        fit.run(*fake_run, out)
    assert (out / 'invalid-completion.json').exists() and (out / 'failed.json').exists()
    assert not (out / 'completed.json').exists()


def test_reciprocal_roundoff_is_not_a_calibration_failure(fake_run, tmp_path, monkeypatch):
    original = fit.fit_temperature
    # A legitimate inverse can differ by one ULP when multiplied back.
    beta = 3.7
    assert beta * (1.0 / beta) != 1.0

    def fitted(episodes):
        result = original(episodes)
        result.update(beta=beta, temperature=1.0 / beta)
        return result

    monkeypatch.setattr(fit, 'fit_temperature', fitted)
    out = tmp_path / 'reciprocal'
    fit.run(*fake_run, out)
    checked = common.authenticate_calibration(out, fit.sha(out / 'completed.json'), 'p', 'i', 'b', 'm')
    assert checked['fits']['kalman-pair0']['beta'] == beta
