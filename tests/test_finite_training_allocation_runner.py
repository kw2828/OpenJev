"""Engineering-only allocation integration and retained-state contracts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_finite_training_allocation as runner


def test_three_allocations_complete_before_fresh_dev_and_restore_optimizer_state(tmp_path, monkeypatch):
    folder = tmp_path / 'integration'
    original_generate, original_factory = runner.generate_attempt_split, runner.learned_model
    generations, public_calls = [], []

    def generate(split, attempts, horizon, **kwargs):
        assert kwargs == {'seed_namespace': 939001}
        generations.append(split)
        if split:
            barrier = json.loads((folder / 'checkpoint-barrier.json').read_text())
            assert barrier['fit_count'] == 3 and barrier['dev_generation_count'] == 0
            assert all((folder / item['path']).is_file() for item in barrier['checkpoints'])
            assert len(list(folder.glob('boundary-*.npz'))) == 3
        return original_generate(split, attempts, horizon, **kwargs)

    def factory(arm, seed):
        assert arm == 'factorized' and seed == 939101 and generations == [0]
        model = original_factory(arm, seed)
        for name in ('blind_rollout', 'observed_rollout'):
            original = getattr(model, name)

            def hooked(*args, _original=original, **kwargs):
                assert not kwargs
                public_calls.append(True)
                return _original(*args, **kwargs)

            monkeypatch.setattr(model, name, hooked)
        return model

    monkeypatch.setattr(runner, 'generate_attempt_split', generate)
    monkeypatch.setattr(runner, 'learned_model', factory)
    config = {'seed_namespace': 939001, 'fit_seeds': [939101], 'train_attempts': 8,
              'dev_attempts': 8, 'batch_size': 3, 'stage1_seconds': 2., 'total_seconds': 4.}
    summary = runner.run(folder, config, lambda: None)
    assert generations == [0, 1] and public_calls
    assert len(summary['rows']) == 12 and len(summary['prefix_rows']) == 3
    fits, counts = summary['fits'], summary['counts']
    assert {fit['arm'] for fit in fits} == set(runner.ARMS)
    assert len({fit['initial_state_sha256'] for fit in fits}) == 1
    assert counts['checkpoint_writes'] == counts['optimizer_checkpoint_writes'] == 9
    assert counts['accepted_optimizer_steps'] <= counts['optimizer_steps'] == counts['optimizer_attempts']
    assert all(counts[k] == 0 for k in ('array_decodes', 'checkpoint_decodes', 'external_model_calls', 'native_calls', 'teacher_calls'))
    for fit in fits:
        assert fit['parameter_metadata']['parameter_count'] == 352
        allocation = json.loads((folder / fit['allocation']['path']).read_text())
        assert allocation['status'] == 'PASS' and len(allocation['stages']) == 2
        assert all(stage['accepted_updates'] > 0 for stage in allocation['stages'])
        assert [stage['deadline_seconds'] for stage in allocation['stages']] == [2., 4.]
        assert allocation['joint_cursor'] == allocation['accepted_joint_updates'] == fit['updates']
        assert allocation['boundary']['optimizer_reset'] == (fit['arm'] != 'joint_continuous')
        assert allocation['boundary']['joint_cursor'] == allocation['stages'][1]['joint_cursor_start']
        for item in allocation['checkpoints']:
            assert item['label'] in ('initial', 'boundary', 'final')
            assert (folder / item['metadata']['optimizer']['path']).exists()
        for row in allocation['trace']:
            assert row['accepted'] == (row['completed_elapsed'] <= row['deadline_seconds'])
            if row['rolled_back']:
                assert row['model_retained_sha256'] == row['model_before_sha256']
                assert row['optimizer_retained_sha256'] == row['optimizer_before_sha256']
                assert row['cursor_retained'] == row['cursor_before']
        if fit['arm'] == 'prefix_then_joint':
            assert allocation['boundary']['joint_cursor'] == 0
            with np.load(folder / 'initial-prefix_then_joint-939101.npz', allow_pickle=False) as initial, \
                    np.load(folder / 'boundary-prefix_then_joint-939101.npz', allow_pickle=False) as boundary:
                np.testing.assert_array_equal(initial['cost_logits'], boundary['cost_logits'])
        else:
            assert allocation['accepted_prefix_updates'] == 0
    # Audit these exact engineering artifacts, without renaming IDs or using
    # scientific seeds. Exercise the complete independent arithmetic/trace/
    # model-and-optimizer-boundary audit after the existing single smoke run.
    from audit_finite_training_allocation import audit

    generations_before_audit, model_calls_before_audit = list(generations), len(public_calls)
    audited = audit(folder, profile='engineering-939001')
    assert generations == generations_before_audit and len(public_calls) == model_calls_before_audit
    assert audited['agreement'] is True and audited['exact_oracle_agreement'] is True
    assert len(audited['rows']) == 12 and len(audited['prefix_rows']) == 3
    assert audited['counts']['array_decodes'] == 21
    assert audited['counts']['checkpoint_decodes'] == audited['counts']['optimizer_json_decodes'] == 9
    assert all(audited['counts'][name] == 0 for name in (
        'model_calls', 'optimizer_calls', 'world_or_generator_calls', 'native_calls'))
    assert set(audited['gates']) == set(runner.ARMS)
    for criteria in audited['gates'].values():
        for gate in criteria.values():
            assert gate['conditions']['minimum_train'] is False
            assert gate['conditions']['minimum_development'] is False
            assert gate['passed'] is False
    assert audited['advance']['passed'] is False
    # The science-default audit must reject an engineering run before loading
    # arrays, rather than silently infer a smaller profile from its metadata.
    with monkeypatch.context() as strict:
        def forbidden_load(*args, **kwargs):
            raise AssertionError('No scientific admission of engineering arrays')
        strict.setattr(np, 'load', forbidden_load)
        with pytest.raises(ValueError):
            audit(folder)
    with pytest.raises(FileExistsError):
        runner.run(folder, config, lambda: None)


@pytest.mark.parametrize('first,total', [(0., 1.), (1., 1.), (2., 1.), (float('nan'), 4.),
                                      (1., float('inf')), (True, 4.), (1, 4.), (1., 61.)])
def test_invalid_allocation_config_fails_before_data(tmp_path, monkeypatch, first, total):
    def forbidden(*args, **kwargs):
        raise AssertionError('No data before valid settings')
    monkeypatch.setattr(runner, 'generate_attempt_split', forbidden)
    with pytest.raises(ValueError):
        runner.run(tmp_path / 'absent', {'stage1_seconds': first, 'total_seconds': total}, lambda: None)
    assert not (tmp_path / 'absent').exists()


def test_optimizer_encoder_retains_dtype_shape_step_and_moments():
    source = {'state': {0: {'step': torch.tensor(2.), 'exp_avg': torch.tensor([.125, -.25], dtype=torch.float64)}},
              'param_groups': [{'params': [0], 'lr': .003, 'betas': (.9, .999), 'amsgrad': False, 'fused': None}]}
    payload = runner.optimizer_payload(source)
    assert payload['state']['0']['step'] == {'kind': 'tensor', 'dtype': 'torch.float32', 'shape': [], 'values': 2.}
    assert payload['state']['0']['exp_avg'] == {'kind': 'tensor', 'dtype': 'torch.float64', 'shape': [2], 'values': [.125, -.25]}
    assert runner.optimizer_payload(json.loads(json.dumps(payload))) == payload
    before = runner.optimizer_digest(source)
    source['state'][0]['exp_avg'][0] = .5
    assert runner.optimizer_digest(source) != before


def test_failed_allocation_trace_is_saved_before_fit_rejection(tmp_path, monkeypatch):
    result = {'status': 'FAILED_ZERO_ACCEPTED', 'trace': [{'accepted': False}], 'stages': []}
    monkeypatch.setattr(runner, 'run_allocation', lambda *a, **k: result)
    prefixes = {'prefix': torch.zeros((1, 9, 31)), 'event_mask': torch.ones((1, 9), dtype=torch.bool)}
    with pytest.raises(ValueError, match='both original allocation stages'):
        runner.train('prefix_then_joint', 939101, {'prefix': torch.zeros((1, 9, 31))}, prefixes,
                     {'batch_size': 1, 'stage1_seconds': 1., 'total_seconds': 2.}, tmp_path, {}, {}, lambda: None)
    assert json.loads((tmp_path / 'allocation-prefix_then_joint-939101.json').read_text()) == result
