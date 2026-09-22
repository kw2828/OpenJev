"""Fabricated saved-target, diagnostic and prefix-publication contracts only."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('budget_runner_fixture', ROOT / 'scripts/study_otto_training_budget.py')
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_final_diagnostics_center_only_allowed_actions_and_use_distinct_weight_precisions():
    masks = np.array([[True, True, False, False], [False, True, True, True]])
    target = np.array([[-1., 1., 0., 0.], [0., -1., 0., 1.]], dtype=np.float32)
    costs = np.array([[4., 6., np.inf, np.inf], [np.inf, 3., 3., 5.]], dtype=np.float64)
    identity = np.array([[2., 0., 99., -99.], [-100., 5e-11, 0., 2.]], dtype=np.float32)
    permutations = [(0, 1, 2, 3), (1, 0, 3, 2)] * 4
    scores = np.stack([identity[:, p] for p in permutations], axis=1)
    weights64 = np.array([2 / 3, 4 / 3], dtype=np.float64)
    weights32 = weights64.astype(np.float32)
    arrays, metrics = runner.diagnostic_metrics(scores, target, masks, costs, weights64, weights32, permutations)
    assert arrays['selected_action'].tolist() == [1, 1]  # Second row uses the deployed near-tie rule.
    assert arrays['label_regret'].tolist() == [2., 0.]
    assert arrays['optimal_set_hit'].tolist() == arrays['first_argmin_match'].tolist() == [False, True]
    assert arrays['single_view_mse'] == pytest.approx([4., 2 / 9], abs=1e-10)
    assert np.allclose(arrays['per_view_mse'], arrays['single_view_mse'][:, None], atol=1e-15, rtol=0)
    expected = (4 * float(weights32[0]) + (2 / 9) * float(weights32[1])) / 2
    assert metrics['single_view_weighted_mse'] == pytest.approx(expected, abs=1e-10)
    assert metrics['eight_view_weighted_mse'] == pytest.approx(expected, abs=1e-10)
    assert metrics['label_regret'] == pytest.approx(2 / 3)
    assert metrics['optimal_set_hit'] == metrics['first_argmin_match'] == pytest.approx(2 / 3)


def test_cache_copy_retains_every_original_target_byte_and_frozen_scale(tmp_path):
    run = runner.Run(SimpleNamespace(output=tmp_path))
    run.np = np
    run.budget = SimpleNamespace(check=lambda **_kwargs: None)
    run.rows = [{'episode_id': f'fabricated:{i % 144}'} for i in range(558)]
    run.features = np.zeros((558, 2836), dtype=np.float32)
    run.features[:, 0] = np.arange(558, dtype=np.float32)
    run.raw_costs = np.tile(np.array([2., 6., np.inf, np.inf]), (558, 1))
    run.allowed = np.isfinite(run.raw_costs)
    centered = np.tile(np.array([-2., 2., -0., 0.]), (558, 1))
    scale = 1.9131089760854865
    arrays = {'centered_float64': centered, 'scaled_float32': (centered / scale).astype(np.float32),
              'weights_float64': np.ones(558), 'weights_float32': np.ones(558, dtype=np.float32), 'allowed': run.allowed}
    run.reference = {'kind': 'r64', 'scale': scale, **arrays}
    run.expected_reference = {'scale': scale, 'arrays': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}}
    run.plan = {'inputs': {f'{runner.BASE}/run-01/training-data.npz': {'sha256': 'a' * 64, 'bytes': 1}}}
    features, bundle = run.training_data()
    assert features is run.features
    with np.load(tmp_path / 'training-data.npz', allow_pickle=False) as saved:
        assert saved['features'].tobytes() == features.tobytes()
        assert saved['r64_costs'].tobytes() == run.raw_costs.tobytes()
        for arm in ('short', 'long'):
            assert bundle[arm]['scale'].hex() == scale.hex()
            for name, value in arrays.items():
                assert saved[f'{arm}_{name}'].tobytes() == value.tobytes()
                assert bundle[arm][name].tobytes() == value.tobytes()
                assert not bundle[arm][name].flags.writeable
    info = json.loads((tmp_path / 'training-data.json').read_text())
    assert info['reference'] == run.expected_reference
    assert len(runner.payload_names()) == 34
    assert not any('panel-' in name for name in runner.payload_names())


def test_long_prefix_reload_requires_byte_identity_and_does_not_acknowledge_mismatch(tmp_path):
    run = runner.Run(SimpleNamespace(output=tmp_path))
    run.np = np
    run.budget = SimpleNamespace(check=lambda **_kwargs: None)
    exported = {'kind': 'fabricated', 'array': np.array([0., 1.], dtype=np.float32)}
    short = run.checkpoint('short', 30101, 'final', exported)
    matching = run.checkpoint('long', 30101, 'prefix', exported)
    assert run.prefix_pairs == [{'seed': 30101, 'short_final': short, 'long_prefix': matching, 'byte_identical': True}]
    run.checkpoint('short', 30102, 'final', exported)
    changed = {'kind': 'fabricated', 'array': np.array([-0., 1.], dtype=np.float32)}
    with pytest.raises(ValueError, match='long prefix checkpoint must exactly equal'):
        run.checkpoint('long', 30102, 'prefix', changed)
    assert len(run.prefix_pairs) == 1
    assert (tmp_path / 'long-30102-prefix.npz').exists()  # Preserve failed publication bytes.
