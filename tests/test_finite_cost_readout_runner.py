"""Engineering-only qualification of paired schedules and unbiased batch scaling."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_finite_cost_readout as r

CONFIG = {'seed_namespace': 932001, 'fit_seeds': [932101], 'train_attempts': 8,
          'dev_attempts': 8, 'epochs': 1, 'batch_size': 3}


def test_paired_fits_finish_before_dev_and_never_receive_oracle(tmp_path, monkeypatch):
    folder = tmp_path / 'engineering'
    generate, factory = r.generate_attempt_split, r.learned_model
    generation, seen_oracle = [], []

    def checked_generation(split, attempts, horizon, **kwargs):
        assert kwargs == {'seed_namespace': 932001}
        generation.append(split)
        if split:
            barrier = json.loads((folder / 'checkpoint-barrier.json').read_text())
            assert barrier['fit_count'] == 3 and barrier['dev_generation_count'] == 0
            assert all((folder / item['path']).is_file() for item in barrier['checkpoints'])
        return generate(split, attempts, horizon, **kwargs)

    def checked_factory(arm, seed):
        assert generation == [0]
        assert (folder / 'oracle-train-check.json').is_file()
        model = factory(arm, seed)
        for method in ('blind_rollout', 'observed_rollout'):
            original = getattr(model, method)
            def hooked(*args, _original=original, **kwargs):
                assert not kwargs
                seen_oracle.append(False)
                return _original(*args, **kwargs)
            monkeypatch.setattr(model, method, hooked)
        return model

    monkeypatch.setattr(r, 'generate_attempt_split', checked_generation)
    monkeypatch.setattr(r, 'learned_model', checked_factory)
    result = r.run(folder, CONFIG, lambda: None)
    assert generation == [0, 1] and seen_oracle
    assert len(result['rows']) == 12 and len(result['prefix_rows']) == 3
    fits = result['fits']
    for key in ('operator_initial_sha256', 'reset_initial_sha256', 'case_order_sha256'):
        assert fits[0][key] == fits[1][key] == fits[2][key]
    assert [fit['parameter_metadata']['parameter_count'] for fit in fits] == [1088, 1088, 1120]
    np.testing.assert_allclose(fits[1]['readout_initial']['matrix'], fits[2]['readout_initial']['matrix'], atol=1e-12, rtol=0)
    assert fits[0]['readout_initial'] == fits[0]['readout_final']
    assert fits[1]['readout_initial'] == fits[1]['readout_final']
    assert all(fit['updates'] == 3 and fit['training_attempts'] == 8 for fit in fits)
    assert [fit['prefix_loss_weight'] for fit in fits] == [1, 1, 1]
    counts = result['counts']
    assert counts['optimizer_steps'] == counts['optimizer_attempts'] == 9
    assert counts['training_attempt_exposures'] == 24
    assert counts['training_case_exposures'] == 3 * result['dataset_counts']['train']['retained']
    assert counts['training_prefix_event_exposures'] == 3 * result['dataset_counts']['train']['valid_prefix_events']
    assert counts['training_prefix_rollouts'] == 9 and counts['evaluation_prefix_rollouts'] == 9
    assert all(result['structural_work'][arm]['training_prefix'] for arm in r.ARMS)
    assert counts['readout_snapshot_evaluations'] == 6 and counts['readout_snapshot_softmax_evaluations'] == 2
    assert all(counts[key] == 0 for key in ('checkpoint_decodes', 'array_decodes', 'external_model_calls', 'native_calls', 'teacher_calls'))
    for arm in r.ARMS:
        with np.load(folder / f'prefix-{arm}-932101-base.npz', allow_pickle=False) as saved:
            assert saved['probabilities'].shape == (8, 9, 5)
    with pytest.raises(FileExistsError):
        r.run(folder, CONFIG, lambda: None)


@pytest.mark.parametrize('arm', r.ARMS)
def test_global_denominators_partial_batch_and_empty_endpoint_step(tmp_path, monkeypatch, arm):
    # Three attempts, only one endpoint. Batch size one exercises empty batches;
    # batch size two below exercises the actual final partial-batch denominator.
    for batch_size in (1, 2):
        folder = tmp_path / str(batch_size)
        folder.mkdir()
        model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        model.weight.data.fill_(1.)
        model.readout_matrix = lambda: torch.zeros((4, 8), dtype=torch.float64)
        recorded = []

        class CaptureAdam:
            def __init__(self, parameters, **kwargs):
                self.parameters = list(parameters)
            def zero_grad(self, **kwargs):
                for p in self.parameters:
                    p.grad = None
            def step(self, _recorded=recorded):
                _recorded.append(float(self.parameters[0].grad))

        def call(m, a, batch, **kwargs):
            return {'value': m.weight.sum(), 'work': {}}

        def prefix(m, inputs, lengths):
            return {'nll': m.weight.sum() * inputs[:, :, 0], 'work': {}}

        monkeypatch.setattr(torch.optim, 'Adam', CaptureAdam)
        monkeypatch.setattr(r, 'call', call)
        monkeypatch.setattr(r, 'objective', lambda blind, observed, batch: blind['value'] * 7)
        monkeypatch.setattr(r, 'prefix_predictions', prefix)
        data = {'prefix': torch.zeros((1, 9, 31))}
        x = torch.zeros((3, 9, 31), dtype=torch.float64)
        x[:, 0, 0] = torch.tensor([2., 3., 5.])
        prefixes = {'prefix': x, 'lengths': torch.ones(3, dtype=torch.int64),
                    'event_mask': torch.tensor([[True] + [False] * 8] * 3),
                    'endpoint_rows': torch.tensor([-1, 0, -1])}
        config = {**r.DEFAULT_CONFIG, 'fit_seeds': [932101], 'epochs': 1,
                  'batch_size': batch_size, 'gradient_clip': 1e9}
        keys = ('training_blind_rollouts', 'training_observed_rollouts', 'zero_endpoint_batches',
                'training_prefix_rollouts', 'training_prefix_event_exposures', 'optimizer_attempts',
                'optimizer_steps', 'training_case_exposures', 'training_attempt_exposures',
                'checkpoint_writes', 'fit_count', 'readout_snapshot_evaluations', 'readout_snapshot_softmax_evaluations')
        counts = dict.fromkeys(keys, 0)
        work = {arm: {route: {} for route in r.ROUTES}}
        fit = r.train(model, arm, 932101, data, prefixes, config, folder, counts, work, lambda: None)
        order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([932101, 0, 818]))).permutation(3)
        expected = []
        for start in range(0, 3, batch_size):
            ids = order[start:start + batch_size]
            endpoint = 7. if 1 in ids else 0.
            likelihood = sum([2., 3., 5.][i] for i in ids) / 3
            expected.append(3 / len(ids) * (endpoint + likelihood))
        assert recorded == pytest.approx(expected)
        assert counts['optimizer_steps'] == len(expected)
        assert fit['zero_endpoint_batches'] >= 1
        assert counts['training_attempt_exposures'] == 3 and counts['training_case_exposures'] == 1
