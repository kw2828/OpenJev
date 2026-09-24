"""Engineering-only allocation integration and retained-state contracts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_finite_update_learning as runner


def test_three_transition_models_complete_before_fresh_dev_and_independent_audit(tmp_path, monkeypatch):
    folder = tmp_path / 'integration'
    original_generate, original_factory = runner.generate_attempt_split, runner.learned_model
    generations, public_calls = [], []

    def generate(split, attempts, horizon, **kwargs):
        assert kwargs == {'seed_namespace': 942001}
        generations.append(split)
        if split:
            barrier = json.loads((folder / 'checkpoint-barrier.json').read_text())
            assert barrier['fit_count'] == 3 and barrier['dev_generation_count'] == 0
            assert all((folder / item['path']).is_file() for item in barrier['checkpoints'])
            assert len(list(folder.glob('boundary-*.npz'))) == 3
        return original_generate(split, attempts, horizon, **kwargs)

    def factory(arm, seed, *, check):
        assert arm in runner.ARMS and seed == 942101 and generations == [0]
        model = original_factory(arm, seed, check=check)
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
    config = {'seed_namespace': 942001, 'fit_seeds': [942101], 'train_attempts': 8,
              'dev_attempts': 8, 'batch_size': 3, 'prefix_updates': 3, 'joint_updates': 4, 'fit_cap_seconds': 30.}
    summary = runner.run(folder, config, lambda: None)
    assert generations == [0, 1] and public_calls
    assert len(summary['rows']) == 12 and len(summary['prefix_rows']) == 3
    fits, counts = summary['fits'], summary['counts']
    assert {fit['arm'] for fit in fits} == set(runner.ARMS)
    hashes = {fit['arm']: fit['initial_state_sha256'] for fit in fits}
    assert hashes['original_free'] == hashes['rounded'] != hashes['matched_free']
    assert counts['checkpoint_writes'] == counts['optimizer_checkpoint_writes'] == 9
    assert counts['accepted_optimizer_steps'] == counts['optimizer_steps'] == counts['optimizer_attempts'] == 21
    assert counts['accepted_prefix_steps'] == 9 and counts['accepted_joint_steps'] == 12
    assert all(counts[k] == 0 for k in ('array_decodes', 'checkpoint_decodes', 'external_model_calls', 'native_calls', 'teacher_calls'))
    for fit in fits:
        assert fit['parameter_metadata']['parameter_count'] == 352
        allocation = json.loads((folder / fit['allocation']['path']).read_text())
        assert allocation['status'] == 'PASS' and len(allocation['stages']) == 2
        assert allocation['arm'] == 'prefix_then_joint'
        assert [stage['accepted_updates'] for stage in allocation['stages']] == [3, 4]
        assert [stage['target_updates'] for stage in allocation['stages']] == [3, 4]
        assert [stage['deadline_seconds'] for stage in allocation['stages']] == [30., 30.]
        assert all(stage['termination'] == 'completed_updates' for stage in allocation['stages'])
        assert allocation['prefix_updates'] == 3 and allocation['joint_updates'] == 4
        assert allocation['max_seconds'] == 30.
        assert allocation['joint_cursor'] == allocation['accepted_joint_updates'] == fit['updates']
        assert allocation['boundary']['optimizer_reset'] is True
        assert allocation['boundary']['joint_cursor'] == allocation['stages'][1]['joint_cursor_start']
        for item in allocation['checkpoints']:
            assert item['label'] in ('initial', 'boundary', 'final')
            assert (folder / item['metadata']['optimizer']['path']).exists()
        for row in allocation['trace']:
            assert row['accepted'] is True and row['rolled_back'] is False
            assert row['completed_elapsed'] <= row['deadline_seconds'] == 30.
            if row['rolled_back']:
                assert row['model_retained_sha256'] == row['model_before_sha256']
                assert row['optimizer_retained_sha256'] == row['optimizer_before_sha256']
                assert row['cursor_retained'] == row['cursor_before']
        assert allocation['boundary']['joint_cursor'] == 0
        with np.load(folder / f"initial-{fit['arm']}-942101.npz", allow_pickle=False) as initial, \
                np.load(folder / f"boundary-{fit['arm']}-942101.npz", allow_pickle=False) as boundary:
            np.testing.assert_array_equal(initial['cost_logits'], boundary['cost_logits'])
    paired = runner.verify_paired_exposure(fits, folder, runner._config({**runner.DEFAULT_CONFIG, **config}))
    assert set(paired) == {942101}
    # Audit these exact engineering artifacts, without renaming IDs or using
    # scientific seeds. Exercise the complete independent arithmetic/trace/
    # model-and-optimizer-boundary audit after the existing single smoke run.
    from audit_finite_update_learning import audit

    generations_before_audit, model_calls_before_audit = list(generations), len(public_calls)
    audited = audit(folder, profile='engineering-942001')
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


@pytest.mark.parametrize('key,value', [
    ('prefix_updates', 0), ('prefix_updates', -1), ('prefix_updates', True), ('prefix_updates', 3.),
    ('joint_updates', 0), ('joint_updates', False), ('joint_updates', 4.), ('joint_updates', 100000),
    ('fit_cap_seconds', 0.), ('fit_cap_seconds', float('nan')), ('fit_cap_seconds', float('inf')),
    ('fit_cap_seconds', True), ('fit_cap_seconds', 30), ('fit_cap_seconds', 120.01),
    ('stage1_seconds', 10.), ('total_seconds', 40.),
])
def test_invalid_allocation_config_fails_before_data(tmp_path, monkeypatch, key, value):
    def forbidden(*args, **kwargs):
        raise AssertionError('No data before valid settings')
    monkeypatch.setattr(runner, 'generate_attempt_split', forbidden)
    with pytest.raises(ValueError):
        runner.run(tmp_path / 'absent', {key: value}, lambda: None)
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
    monkeypatch.setattr(runner, 'run_update_allocation', lambda *a, **k: result)
    prefixes = {'prefix': torch.zeros((1, 9, 31)), 'event_mask': torch.ones((1, 9), dtype=torch.bool)}
    with pytest.raises(ValueError, match='both exact-update stages'):
        runner.train('rounded', 942101, {'prefix': torch.zeros((1, 9, 31))}, prefixes,
                     {'batch_size': 1, 'prefix_updates': 3, 'joint_updates': 4, 'fit_cap_seconds': 30.}, tmp_path, {}, {}, lambda: None)
    assert json.loads((tmp_path / 'allocation-rounded-942101.json').read_text()) == result


def test_failed_normalization_preserves_partial_work_and_allocation(tmp_path, monkeypatch):
    error = ValueError('failed fixed normalization')
    error.allocation_result = {'status': 'FAILED', 'trace': []}
    error.rounded_model_work = {'completed_sweeps': 2, 'rounded_transition_calls': 1}

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(runner, 'run_update_allocation', fail)
    with pytest.raises(ValueError, match='failed fixed normalization'):
        runner.train('rounded', 942101, {}, {},
                     {'prefix_updates': 3, 'joint_updates': 4, 'fit_cap_seconds': 30.}, tmp_path, {}, {}, lambda: None)
    assert json.loads((tmp_path / 'allocation-rounded-942101.json').read_text()) == error.allocation_result
    assert json.loads((tmp_path / 'failed-normalization-rounded-942101.json').read_text()) == error.rounded_model_work
    assert not (tmp_path / 'fits.jsonl').exists()


def paired_fixture(folder):
    config = {**runner.DEFAULT_CONFIG, 'seed_namespace': 942001, 'fit_seeds': [942101],
              'train_attempts': 8, 'dev_attempts': 8, 'batch_size': 3,
              'prefix_updates': 3, 'joint_updates': 4, 'fit_cap_seconds': 30.}
    fits = []
    for arm in runner.ARMS:
        trace = [{'kind': 'prefix', 'accepted': True, 'rolled_back': False} for _ in range(3)]
        trace += [{'kind': 'joint', 'accepted': True, 'rolled_back': False,
                   'result': {'diagnostics': {'indices': indices}}}
                  for indices in ([0, 1, 2], [3, 4, 5], [6, 7], [5, 0, 6])]
        value = {'status': 'PASS', 'accepted_prefix_updates': 3, 'accepted_joint_updates': 4,
                 'joint_cursor': 4, 'attempted_updates': 7, 'accepted_updates': 7, 'trace': trace}
        path = folder / f'allocation-{arm}-942101.json'
        path.write_text(json.dumps(value))
        fits.append({'arm': arm, 'seed': 942101, 'accepted_prefix_updates': 3,
                     'updates': 4, 'attempted_updates': 7,
                     'allocation': {'path': path.name, **runner._desc(path)}})
    return fits, config


def test_same_seed_exposure_requires_actual_ordered_indices(tmp_path):
    fits, config = paired_fixture(tmp_path)
    assert set(runner.verify_paired_exposure(fits, tmp_path, config)) == {942101}
    path = tmp_path / fits[-1]['allocation']['path']
    record = json.loads(path.read_text())
    # Same batch size and count, different examples/order: counts alone cannot pass.
    record['trace'][-1]['result']['diagnostics']['indices'] = [0, 5, 6]
    path.write_text(json.dumps(record))
    fits[-1]['allocation'].update(runner._desc(path))
    with pytest.raises(ValueError, match='identical ordered joint minibatches'):
        runner.verify_paired_exposure(fits, tmp_path, config)


@pytest.mark.parametrize('fault', ['short_count', 'discarded', 'stage_order', 'missing', 'stale_hash'])
def test_predev_barrier_rejects_partial_or_altered_exposure(tmp_path, fault):
    fits, config = paired_fixture(tmp_path)
    if fault == 'missing':
        fits.pop()
    else:
        path = tmp_path / fits[-1]['allocation']['path']
        record = json.loads(path.read_text())
        if fault == 'short_count':
            record['accepted_prefix_updates'] = 2
        elif fault == 'discarded':
            record['trace'][0]['accepted'] = False
            record['trace'][0]['rolled_back'] = True
        elif fault == 'stage_order':
            record['trace'][0], record['trace'][3] = record['trace'][3], record['trace'][0]
        else:
            record['extra_unbound_metadata'] = True
        path.write_text(json.dumps(record))
        if fault != 'stale_hash':
            fits[-1]['allocation'].update(runner._desc(path))
    with pytest.raises(ValueError):
        runner.verify_paired_exposure(fits, tmp_path, config)


def test_timeout_keeps_original_failed_controller_record(tmp_path, monkeypatch):
    error = TimeoutError('whole-fit safety cap')
    error.allocation_result = {'status': 'FAILED_TIMEOUT', 'accepted_prefix_updates': 2,
        'accepted_joint_updates': 0, 'trace': [{'accepted': False, 'rolled_back': True}]}

    def fail(*args, **kwargs):
        assert kwargs['prefix_updates'] == 3 and kwargs['joint_updates'] == 4
        assert kwargs['max_seconds'] == 30.
        raise error

    monkeypatch.setattr(runner, 'run_update_allocation', fail)
    with pytest.raises(TimeoutError, match='whole-fit safety cap') as caught:
        runner.train('rounded', 942101, {}, {},
                     {'prefix_updates': 3, 'joint_updates': 4, 'fit_cap_seconds': 30.},
                     tmp_path, {}, {}, lambda: None)
    assert caught.value is error
    assert json.loads((tmp_path / 'allocation-rounded-942101.json').read_text()) == error.allocation_result
    assert not (tmp_path / 'fits.jsonl').exists()
