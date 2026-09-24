"""Tiny engineering namespace checks; never use registered study cases/seeds."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_finite_shared_filter as r

CONFIG = {'seed_namespace': 930001, 'fit_seeds': [930101], 'train_attempts': 8,
          'dev_attempts': 8, 'epochs': 1, 'batch_size': 8}


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_three_fits_follow_exact_train_control_and_finish_before_dev(tmp_path, monkeypatch):
    folder = tmp_path / 'engineering'
    generate, factor, learned, adam = r.generate_split, r.factor_model, r.learned_model, torch.optim.Adam
    generation_calls, constructions, optimizer_calls, routing = [], [], [], []

    def train_control_exists():
        assert (folder / 'oracle-train.npz').is_file()
        errors = json.loads((folder / 'oracle-train-check.json').read_text())
        assert set(errors) == set(r.ORACLE_FIELDS)
        assert all(value <= 1e-12 for value in errors.values())

    def generate_checked(split_id, attempts, horizon, **kwargs):
        assert kwargs == {'seed_namespace': 930001, 'include_oracle': True}
        assert attempts == 8 and horizon == (2 if split_id == 0 else 8)
        generation_calls.append(split_id)
        if split_id:
            train_control_exists()
            barrier = json.loads((folder / 'checkpoint-barrier.json').read_text())
            fits = read_lines(folder / 'fits.jsonl')
            assert barrier['fit_count'] == len(fits) == 3
            assert barrier['dev_generation_count'] == 0 and barrier['oracle_train_verified'] is True
            assert {(fit['arm'], fit['seed']) for fit in fits} == {(arm, 930101) for arm in r.ARMS}
            for item in barrier['checkpoints']:
                raw = (folder / item['path']).read_bytes()
                assert item['bytes'] == len(raw) and item['sha256'] == hashlib.sha256(raw).hexdigest()
        return generate(split_id, attempts, horizon, **kwargs)

    def observe_routing(model, arm):
        for name in ('blind_rollout', 'observed_rollout'):
            original = getattr(model, name)
            def hook(*args, _original=original, _name=name, **kwargs):
                expected = {'oracle_prefix'} if arm == 'exact_exact' else set()
                assert set(kwargs) == expected
                if expected:
                    assert kwargs['oracle_prefix'].dtype == torch.float64
                routing.append((arm, _name, bool(expected)))
                return _original(*args, **kwargs)
            monkeypatch.setattr(model, name, hook)
        return model

    def factor_checked(arm, seed):
        assert arm == 'exact_exact'
        constructions.append(arm)
        model = factor(arm, seed)
        assert not list(model.parameters())
        return observe_routing(model, arm)

    def learned_checked(arm, seed):
        train_control_exists()
        assert arm in r.ARMS and generation_calls == [0]
        constructions.append(arm)
        return observe_routing(learned(arm, seed), arm)

    def adam_checked(parameters, *args, **kwargs):
        train_control_exists()
        parameters = list(parameters)
        assert parameters  # Never optimize the zero-parameter exact control.
        optimizer_calls.append(len(parameters))
        return adam(parameters, *args, **kwargs)

    monkeypatch.setattr(r, 'generate_split', generate_checked)
    monkeypatch.setattr(r, 'factor_model', factor_checked)
    monkeypatch.setattr(r, 'learned_model', learned_checked)
    original_objective = r.objective
    def objective_checked(blind, observed, batch):
        assert 'oracle_prefix' not in batch
        return original_objective(blind, observed, batch)
    monkeypatch.setattr(r, 'objective', objective_checked)
    monkeypatch.setattr(torch.optim, 'Adam', adam_checked)
    result = r.run(folder, CONFIG, lambda: None)
    assert generation_calls == [0, 1]
    assert constructions == ['exact_exact', *r.ARMS]
    assert len(optimizer_calls) == 3 and routing
    assert len(result['rows']) == 12 and len(result['baseline_rows']) == 4
    assert result['oracle_metadata']['parameter_count'] == 0
    assert set(result['oracle_checks']) == {'train', 'base'}
    fits = {fit['arm']: fit for fit in result['fits']}
    assert fits['shared_filter']['reset_initial_sha256'] == fits['untied_filter']['reset_initial_sha256']
    assert len({fit['operator_initial_sha256'] for fit in fits.values()}) == 1
    for arm in ('shared_filter', 'untied_filter'):
        assert fits[arm]['prefix_operator_initial_sha256'] == fits[arm]['operator_initial_sha256']
        assert fits[arm]['prefix_initial_sha256'] is None
    assert fits['gru_prefix']['prefix_initial_sha256'] is not None
    assert fits['gru_prefix']['reset_initial_sha256'] is None
    assert all(fit['oracle_prefix_input'] is False for fit in fits.values())
    work = result['structural_work']
    assert set(work) == set(r.ARMS)
    for arm in r.ARMS:
        assert set(work[arm]) == set(r.ROUTES)
        assert all(row['privileged_prefix_rows'] == 0 for row in work[arm].values())
    assert len({fit['case_order_sha256'] for fit in fits.values()}) == 1
    n = result['dataset_counts']['train']['retained']
    for fit in fits.values():
        assert fit['updates'] == fit['epochs'] == 1
        assert fit['training_cases'] == fit['training_case_exposures'] == n
    counts = result['counts']
    for name in ('model_constructions', 'fit_count', 'checkpoint_writes', 'optimizer_attempts',
                 'optimizer_steps', 'training_blind_rollouts', 'training_observed_rollouts',
                 'evaluation_blind_rollouts', 'evaluation_observed_rollouts', 'evaluation_shuffled_rollouts'):
        assert counts[name] == 3, name
    assert counts['oracle_model_constructions'] == 1
    assert counts['oracle_blind_rollouts'] == counts['oracle_observed_rollouts'] == 2
    assert counts['training_case_exposures'] == 3 * n
    for name in ('array_decodes', 'checkpoint_decodes', 'external_model_calls', 'native_calls', 'teacher_calls'):
        assert counts[name] == 0
    for row in result['prediction_times']:
        assert row['model_state_before'] == row['model_state_after'] == fits[row['arm']]['final_state_sha256']
    for split in ('train', 'base'):
        with np.load(folder / f'{split}.npz', allow_pickle=False) as data:
            cases = len(data['case_ids'])
            assert all(str(case).startswith('ns930001-') for case in data['case_ids'])
        with np.load(folder / f'oracle-{split}.npz', allow_pickle=False) as predictions:
            assert set(predictions.files) == set(r.ORACLE_FIELDS)
            assert all(predictions[key].dtype == np.float64 and np.isfinite(predictions[key]).all()
                       for key in predictions.files)
        if split == 'base':
            with np.load(folder / 'predictions-base.npz', allow_pickle=False) as predictions:
                assert set(predictions.files) == {f'{arm}__930101__{field}' for arm in r.ARMS for field in r.FIELDS}
                for key in predictions.files:
                    field = key.rsplit('__', 1)[1]
                    shape = (cases, 8, 5) if field == 'observed_probabilities' else (cases, 8) if field.endswith('survival') else (cases, 8, 4)
                    assert predictions[key].shape == shape and predictions[key].dtype == np.float64
                    assert np.isfinite(predictions[key]).all()
    with pytest.raises(FileExistsError):
        r.run(folder, CONFIG, lambda: None)
    assert generation_calls == [0, 1]


def test_wrong_train_oracle_stops_before_any_learned_model_or_optimizer(tmp_path, monkeypatch):
    generate, factor = r.generate_split, r.factor_model
    constructed = []

    def wrong_target(split_id, attempts, horizon, **kwargs):
        assert split_id == 0 and kwargs['seed_namespace'] == 930001
        result = generate(split_id, attempts, horizon, **kwargs)
        result['data']['blind_costs'][0, 0, 0] += .01
        return result

    def reference_only(arm, seed):
        assert arm == 'exact_exact'
        constructed.append(arm)
        return factor(arm, seed)

    def forbidden(*args, **kwargs):
        pytest.fail('training must not start after a failed exact TRAIN control')

    monkeypatch.setattr(r, 'generate_split', wrong_target)
    monkeypatch.setattr(r, 'factor_model', reference_only)
    monkeypatch.setattr(r, 'learned_model', forbidden)
    monkeypatch.setattr(torch.optim, 'Adam', forbidden)
    folder = tmp_path / 'failed-control'
    with pytest.raises(ValueError, match='exact reference reproduces every target'):
        r.run(folder, CONFIG, lambda: None)
    failure = json.loads((folder / 'failure.json').read_text())
    assert failure['counts']['model_constructions'] == failure['counts']['fit_count'] == 0
    assert failure['counts']['optimizer_attempts'] == failure['counts']['dev_generation_count'] == 0
    assert constructed == ['exact_exact']
    assert not (folder / 'checkpoint-barrier.json').exists()


class Recorder:
    def __init__(self):
        self.training = True
        self.calls = []

    def eval(self):
        self.training = False

    def train(self, mode):
        self.training = mode

    def output(self, prefix, lengths, actions):
        ids = prefix[:, 0, 4:7].argmax(-1)
        self.calls.append((ids.clone(), lengths.clone(), actions.clone()))
        batch, horizon = actions.shape
        return {'cost_contrasts': torch.zeros((batch, horizon, 4), dtype=torch.float64),
                'survival_mass': torch.ones((batch, horizon), dtype=torch.float64),
                'probabilities': torch.full((batch, horizon, 5), .2, dtype=torch.float64),
                'work': {'privileged_prefix_rows': 0, 'prefix_filter_rows': batch * 2}}

    def blind_rollout(self, prefix, lengths, actions):
        return self.output(prefix, lengths, actions)

    def observed_rollout(self, prefix, lengths, actions, observations):
        return self.output(prefix, lengths, actions)


def test_shuffle_moves_only_public_prefix_and_keeps_forecast_actions():
    data = {'prefix': np.zeros((3, 3, 31), dtype=np.float32),
            'lengths': np.array([1, 2, 3], dtype=np.int64),
            'actions': np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
            'observations': np.zeros((3, 2), dtype=np.int64)}
    for case in range(3):
        data['prefix'][case, 0, 4 + case] = 1
    # Deliberately invalid oracle bytes must be irrelevant for a learned arm.
    oracle = np.full((3, 8), np.nan, dtype=np.float64)
    counts = dict.fromkeys(('evaluation_blind_rollouts', 'evaluation_observed_rollouts',
                           'evaluation_shuffled_rollouts', 'evaluation_case_views'), 0)
    work = {'shared_filter': {route: {} for route in r.ROUTES}}
    model = Recorder()
    result = r.predict(model, 'shared_filter', data, oracle, {'batch_size': 2}, counts, lambda: None,
                       shuffled=True, structural_work=work)
    assert set(result) == set(r.FIELDS) and model.training is True
    assert len(model.calls) == 6
    for index, source_ids, case_ids in ((2, [1, 2], [0, 1]), (5, [0], [2])):
        ids, lengths, actions = model.calls[index]
        assert ids.tolist() == source_ids
        assert lengths.tolist() == data['lengths'][source_ids].tolist()
        assert actions.tolist() == data['actions'][case_ids].tolist()
    assert counts['evaluation_case_views'] == 3
    assert counts['evaluation_shuffled_rollouts'] == 2
    for route in r.ROUTES[2:]:
        assert work['shared_filter'][route] == {'privileged_prefix_rows': 0, 'prefix_filter_rows': 6}
