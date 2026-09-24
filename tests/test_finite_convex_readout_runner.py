"""Engineering-only end-to-end lineage, chronology and frozen-model controls."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_finite_convex_readout as r
from register_finite_convex_readout import CONFIG

ENGINEERING = {**CONFIG, 'dev_seed_namespace': 935001, 'dev_attempts': 8,
               'parent_seeds': [935101], 'train_namespace': 935001, 'batch_size': 3}


def upstream_fixture(tmp_path):
    folder = tmp_path / 'fabricated-parent'
    folder.mkdir()
    generated = r.generate_attempt_split(0, 8, 2, seed_namespace=935001)
    r._save(folder / 'train.npz', generated['data'])
    fits = []
    for parent in r.PARENTS:
        model = r.make_model(parent, 935101)
        values = r._state(model)
        name = f'{parent}-935101.npz'
        r._save(folder / name, values)
        with torch.no_grad():
            costs = model.readout_matrix().numpy().copy()
        fits.append({'arm': parent, 'seed': 935101, 'final_state_sha256': r._state_hash(values),
                     'checkpoint': {'path': name, **r._desc(folder / name)},
                     'readout_final': {'matrix': costs.tolist(), 'sha256': r.raw_hash(costs)}})
    r._write(folder / 'summary.json', {'fits': fits,
        'files': {'train.npz': r._desc(folder / 'train.npz')},
        'dataset_counts': {'train': generated['counts']}})
    return {'run': str(folder)}, generated


def test_all_heads_complete_before_fresh_dev_and_models_remain_unchanged(tmp_path, monkeypatch):
    upstream, generated = upstream_fixture(tmp_path)
    output = tmp_path / 'qualified-pipeline'
    generator, factory = r.generate_attempt_split, r.make_model
    seen = []

    def generation(split, attempts, horizon, **kwargs):
        assert split == 1 and attempts == 8 and horizon == 8
        assert kwargs == {'seed_namespace': 935001}
        barrier = json.loads((output / 'solve-barrier.json').read_text())
        assert barrier['completed_solves'] == 3 and barrier['dev_generation_count'] == 0
        assert all((output / row['file']).is_file() for row in barrier['solves'])
        seen.append('DEV')
        return generator(split, attempts, horizon, **kwargs)

    def make(parent, seed):
        assert seen == [] and seed == 935101
        model = factory(parent, seed)
        for method in ('blind_rollout', 'observed_rollout'):
            original = getattr(model, method)
            def guard(*args, _original=original, **kwargs):
                assert not kwargs, 'no privileged oracle argument'
                return _original(*args, **kwargs)
            monkeypatch.setattr(model, method, guard)
        return model

    monkeypatch.setattr(r, 'generate_attempt_split', generation)
    monkeypatch.setattr(r, 'make_model', make)
    result = r.run(output, ENGINEERING, lambda: None, upstream=upstream)
    assert seen == ['DEV'] and len(result['solves']) == 3 and len(result['rows']) == 24
    assert len(result['prediction_times']) == 6
    assert len(result['frozen_invariants']) == 3
    assert all(all(row['bitwise_equal'].values()) for row in result['frozen_invariants'])
    for row in result['solves']:
        assert row['source_state_sha256'] == row['model_state_after']
        assert row['solver_result']['complete'] and row['solver_result']['certificate']['fw_gap'] <= 1e-8
        assert row['solver_result']['certificate']['objective'] <= row['solver_result']['objective_initial'] + 1e-12
        assert row['train_original_head_max_error'] <= 1e-12
    count = result['counts']
    n = len(generated['data']['prefix'])
    assert count['train_array_decodes'] == 1
    assert count['checkpoint_decodes'] == count['model_constructions'] == count['solver_calls'] == count['completed_solves'] == 3
    assert count['train_readout_validation_rows'] == 3 * 2 * n * 2
    dev_n = result['dataset_counts']['base']['retained']
    assert count['head_matrix_rows'] == 6 * 3 * dev_n * 8
    assert count['evaluation_case_views'] == 6 * dev_n
    assert all(count[key] == 0 for key in ('checkpoint_writes', 'optimizer_steps', 'external_model_calls', 'teacher_calls', 'native_calls'))
    assert (output / 'train.npz').read_bytes() == (Path(upstream['run']) / 'train.npz').read_bytes()
    for parent in r.PARENTS:
        with np.load(output / f'states-base-{parent}-935101.npz', allow_pickle=False) as states:
            assert len(states.files) == 10
            for name in ('blind_states', 'observed_states', 'shuffled_states', 'prefix_states', 'shuffled_prefix_states'):
                assert r.same_bytes(states['original_' + name], states['solved_' + name])
    with pytest.raises(FileExistsError):
        r.run(output, ENGINEERING, lambda: None, upstream=upstream)


def test_a_failed_head_prevents_any_dev_generation_and_is_preserved(tmp_path, monkeypatch):
    upstream, _ = upstream_fixture(tmp_path)
    original = r.solve
    attempts = []

    def failed(problem, initial, **kwargs):
        result = original(problem, initial, **kwargs)
        result.update(complete=False, status='SOLVE_FAIL', failure_reasons=['fabricated_qualification_failure'])
        attempts.append(result)
        return result

    def no_dev(*args, **kwargs):
        raise AssertionError('DEV must remain unused')

    monkeypatch.setattr(r, 'solve', failed)
    monkeypatch.setattr(r, 'generate_attempt_split', no_dev)
    folder = tmp_path / 'failed-pipeline'
    with pytest.raises(ValueError, match='every TRAIN-only solve'):
        r.run(folder, ENGINEERING, lambda: None, upstream=upstream)
    assert len(attempts) == 3 and not (folder / 'solve-barrier.json').exists()
    failure = json.loads((folder / 'failure.json').read_text())
    assert failure['counts']['dev_generation_count'] == 0
    assert failure['counts']['solver_calls'] == 3 and failure['counts']['completed_solves'] == 0
    assert len((folder / 'solves.jsonl').read_text().splitlines()) == 3


def test_byte_parity_rejects_signed_zero_difference():
    first, second = np.array([0.], np.float64), np.array([-0.], np.float64)
    assert np.array_equal(first, second)
    assert not r.same_bytes(first, second)
