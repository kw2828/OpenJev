"""Independent deterministic witnesses; no scientific generator or learner."""
from __future__ import annotations

import copy
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_function_reuse_reference as a


def fixture():
    identity = np.eye(8)
    maps = np.array([identity, 2*identity, 3*identity])
    bx = np.repeat(np.tile(identity, (2, 1))[None], 3, axis=0)
    by = np.array([bx[i]*(i+1) for i in range(3)])
    fx = np.repeat(identity[:4][None], 2, axis=0)
    fy = np.array([3*identity[:4], identity[:4]])
    query = np.array([identity[7], identity[0]])
    target = np.array([3*identity[7], identity[0]])
    costs = np.array([identity[7], -identity[7], identity[0], -identity[0], identity[1], -identity[1]])
    return {
        'matrices': maps[None].copy(), 'basis_x': bx[None], 'basis_y': by[None],
        'fewshot_x': fx[None], 'fewshot_y': fy[None], 'query_x': query[None], 'target': target[None],
        'cost_vectors': np.repeat(costs[None, None], 2, axis=1),
        'true_index': np.array([[2, 0]], np.int64), 'maps': maps[None].copy(),
        'singular_values': np.full((1, 3, 8), np.sqrt(2)), 'ranks': np.full((1, 3), 8, np.int64),
        'selected_index': np.array([[2, 0]], np.int64),
        'block_residuals': np.array([[[.5, .125, 0.], [0., .125, .5]]]),
        'lookup_prediction': target[None].copy(),
        'fewshot_prediction': np.array([[np.zeros(8), identity[0]]]),
        'fewshot_ranks': np.full((1, 2), 4, np.int64), 'fewshot_singular_values': np.ones((1, 2, 4)),
        'lookup_choice': np.array([[1, 3]], np.int64), 'fewshot_choice': np.array([[0, 3]], np.int64),
        'lookup_fit_seconds': np.array([.125]), 'lookup_query_seconds': np.array([.25]),
        'fewshot_query_seconds': np.array([.5]),
    }


def test_independent_qr_svd_and_rational_full_evidence_witness():
    data = fixture()
    before = {key: value.copy() for key, value in data.items()}
    calls = []
    work = a.reconstruct(data, 3, np, check=lambda: calls.append(1))
    assert work == {'block_fits': 3, 'fewshot_fits': 2, 'queries': 2,
                    'methods': {'qr': 3, 'svd_minimum_norm': 2, 'svd_zero_rank': 0},
                    'exact_residual_tie_queries': 0}
    assert calls == [1]
    row = a.summarize(data, 0, 3, np)
    assert row['lookup_mse'] == row['lookup_regret'] == 0
    assert row['fewshot_mse'] == 9/16 and row['fewshot_regret'] == 3
    assert row['fewshot_action_accuracy'] == .5 and row['lookup_action_accuracy'] == 1
    assert row['retrieval_accuracy'] == 1 and row['min_relative_singular_value'] == 1
    assert row['map_array_bytes_per_context'] == 1536
    assert row['diagnostic_array_bytes_per_context'] == 219
    assert row['raw_basis_array_bytes_per_context'] == 6144
    assert all(row['conditions'].values())
    for name in data:
        np.testing.assert_array_equal(data[name], before[name])


def test_rank_deficient_and_zero_rank_are_retained_without_false_recovery():
    x = np.array([[1., 1.], [2., 2.]])
    y = np.array([[2., 4.], [4., 8.]])
    matrix, _, rank, method = a.independent_fit(x, y, np)
    np.testing.assert_allclose(matrix, [[1., 2.], [1., 2.]], atol=1e-14)
    assert rank == 1 and method == 'svd_minimum_norm'
    matrix, _, rank, method = a.independent_fit(np.zeros((2, 2)), y, np)
    np.testing.assert_array_equal(matrix, np.zeros((2, 2)))
    assert rank == 0 and method == 'svd_zero_rank'


def test_saved_map_exact_ties_choose_first_public_block_not_private_identity():
    maps = np.repeat(np.eye(2)[None], 3, axis=0)
    index, residual, prediction = a.public_query(maps, np.eye(2), np.eye(2), np.array([4., 7.]), np)
    assert index == 0
    np.testing.assert_array_equal(residual, [0., 0., 0.])
    np.testing.assert_array_equal(prediction, [4., 7.])


@pytest.mark.parametrize('key,index,value', [
    ('maps', (0, 0, 0, 0), 2.), ('singular_values', (0, 0, 0), 2.),
    ('ranks', (0, 0), 7), ('fewshot_ranks', (0, 0), 3),
    ('fewshot_singular_values', (0, 0, 0), 2.),
    ('selected_index', (0, 0), 1), ('block_residuals', (0, 0, 0), .6),
    ('lookup_prediction', (0, 0, 7), 2.), ('fewshot_prediction', (0, 0, 7), 1.),
    ('lookup_choice', (0, 0), 0), ('fewshot_choice', (0, 0), 1),
    ('true_index', (0, 0), 1), ('target', (0, 0, 7), 2.),
    ('basis_y', (0, 0, 0, 0), 2.), ('fewshot_y', (0, 0, 0, 0), 2.),
])
def test_rejects_independently_wrong_equations_rank_routes_predictions_and_choices(key, index, value):
    data = fixture(); data[key][index] = value
    with pytest.raises(ValueError):
        a.reconstruct(data, 3, np)


def test_rejects_wrong_shape_dtype_nonfinite_units_and_callback_failure():
    for name, replacement in [('maps', fixture()['maps'].astype(np.float32)),
                              ('target', np.full((1, 2, 8), np.nan)),
                              ('cost_vectors', 2*fixture()['cost_vectors']),
                              ('query_x', np.zeros((1, 1, 8)))]:
        data = fixture(); data[name] = replacement
        with pytest.raises(ValueError):
            a.reconstruct(data, 3, np)
    expected = RuntimeError('fabricated stop')
    def stop():
        raise expected
    with pytest.raises(RuntimeError) as caught:
        a.reconstruct(fixture(), 3, np, check=stop)
    assert caught.value is expected


def groups():
    return [{'cohort': cohort, 'blocks': blocks, 'lookup_mse': 0., 'retrieval_accuracy': 1.,
             'lookup_regret': 0., 'full_rank': True} for cohort in range(5) for blocks in (1, 3, 8, 16)]


@pytest.mark.parametrize('key,value', [('lookup_mse', 1.0001e-12), ('retrieval_accuracy', .999),
                                     ('lookup_regret', 1.0001e-10), ('full_rank', False)])
def test_each_of_twenty_groups_must_pass_no_average_rescue(key, value):
    rows = groups()
    assert a.classify(rows)['status'] == 'LINEAR_TASK_SOLVED_BY_CLASSICAL_REFERENCE'
    assert len(a.classify(rows)['conditions']) == 80
    rows[-1][key] = value
    result = a.classify(rows)
    assert result['status'] == 'REFERENCE_NOT_SOLVED' and sum(result['conditions'].values()) == 79


def test_full_group_roster_and_exact_typed_summary_are_required():
    with pytest.raises(ValueError, match='all20'):
        a.classify(groups()[:-1])
    rows = groups(); rows[-1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match='all20'):
        a.classify(rows)
    with pytest.raises(ValueError):
        a.compare({'ok': 1}, {'ok': True})
    with pytest.raises(ValueError):
        a.compare({'value': float('nan')}, {'value': 0.})


def test_source_and_payload_admission_fail_before_any_numerical_load(tmp_path, monkeypatch):
    repository, folder = tmp_path/'repo', tmp_path/'evidence'
    folder.mkdir(); (folder/'run').mkdir()
    monkeypatch.setattr(a, 'ROOT', repository)
    for key in a.THREADS:
        monkeypatch.setenv(key, '1')
    pins = {}
    for name in a.SOURCES:
        for base in (repository, folder/'sources'):
            path = base/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('fabricated source\n')
        pins[name] = a.descriptor(repository/name)
    registration = {'config': a.CONFIG, 'seconds_cap': 120, 'run_attempts': 1, 'data_selection': 'none',
                    'training_updates': 0, 'sources': pins,
                    'runtime': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                                'numpy': np.__version__, 'threads': {k: os.environ[k] for k in a.THREADS}}}
    (folder/'registration.json').write_text(json.dumps(registration))
    pin = a.descriptor(folder/'registration.json')
    for cohort in range(5):
        for blocks in a.GROUPS:
            (folder/f'run/cohort-{cohort:02d}-k-{blocks:02d}.npz').write_bytes(b'opaque, never decoded')
    (folder/'run/summary.json').write_text('{}')
    (folder/'run/started.json').write_text(json.dumps({'registration': pin}))
    files = {str(path.relative_to(folder)): a.descriptor(path) for path in (folder/'run').iterdir()}
    (folder/'run/receipt.json').write_text(json.dumps({'status': 'COMPLETE', 'registration': pin,
                                                     'elapsed_seconds': 1., 'files': files}))
    assert a.admit(folder)['registration'] == pin
    (folder/'run/cohort-00-k-01.npz').write_bytes(b'changed')
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('must authenticate before decode'))
    with pytest.raises(ValueError, match='payload bytes'):
        a.audit(folder)
