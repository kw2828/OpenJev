"""Engineering-only integration of the frozen flow and qualified reuse fit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_finite_joint_reuse_training as qualified
import run_finite_reuse_replication as runner
import run_finite_update_learning as frozen

SMOKE = {'seed_namespace': 945001, 'fit_seeds': [945101], 'train_attempts': 8,
         'dev_attempts': 8, 'batch_size': 3, 'prefix_updates': 3,
         'joint_updates': 4, 'fit_cap_seconds': 30.}


def test_explicit_frozen_helpers_and_two_registered_engineering_geometries():
    assert runner.train is qualified.train
    assert runner._config is frozen._config
    assert runner.predict_prefix is frozen.predict_prefix
    assert runner.verify_paired_exposure is frozen.verify_paired_exposure
    assert runner.DEFAULT_CONFIG == {**frozen.DEFAULT_CONFIG, 'seed_namespace': 435260924,
                                    'dev_attempts': 512,
                                    'fit_seeds': [435261001, 435261002, 435261003, 435261004, 435261005]}
    assert runner._config({**runner.DEFAULT_CONFIG, **SMOKE}) == {**runner.DEFAULT_CONFIG, **SMOKE}
    exposure = {**runner.DEFAULT_CONFIG, 'seed_namespace': 945201, 'fit_seeds': [945301],
                'dev_attempts': 8, 'prefix_updates': 32, 'joint_updates': 64, 'fit_cap_seconds': 30.}
    assert runner._config(exposure) == exposure
    work = runner.new_structural_work()
    assert set(work) == set(runner.ARMS)
    assert all(set(routes) == {*qualified.ROUTES, *qualified.REUSE_ROUTES} for routes in work.values())
    assert all(not block for routes in work.values() for block in routes.values())


def test_all_reuse_fits_and_exact_reference_precede_dev_then_saved_output_audit(tmp_path, monkeypatch):
    folder = tmp_path / 'smoke'
    real_generate, real_train = runner.generate_attempt_split, runner.train
    generations, fits_called, references = [], [], []
    real_reference = runner.make_reference

    def generate(split, attempts, horizon, **kwargs):
        assert kwargs == {'seed_namespace': 945001}
        assert (attempts, horizon) == (8, 2 if split == 0 else 8)
        if split:
            assert fits_called == list(runner.ARMS)
            barrier = json.loads((folder / 'checkpoint-barrier.json').read_text())
            assert barrier['fit_count'] == 3 and barrier['dev_generation_count'] == 0
            assert barrier['oracle_train_verified'] is True
            assert len(barrier['checkpoints']) == 3
            assert all((folder / row['path']).is_file() for row in barrier['checkpoints'])
            assert len(list(folder.glob('initial-*.npz'))) == 3
            assert len(list(folder.glob('boundary-*.npz'))) == 3
            assert len(list(folder.glob('final-*.npz'))) == 3
            assert len(list(folder.glob('*-optimizer-*.json'))) == 9
        generations.append(split)
        return real_generate(split, attempts, horizon, **kwargs)

    def reference(epsilon):
        if epsilon == .30:
            assert fits_called == list(runner.ARMS) and generations == [0, 1]
            assert (folder / 'checkpoint-barrier.json').is_file()
        else:
            assert epsilon == .12 and not generations and not fits_called
        result = real_reference(epsilon)
        assert not list(result.parameters())
        references.append((epsilon, result))
        return result

    def train(arm, seed, data, prefixes, config, output, counts, work, check, *, implementation):
        assert generations == [0] and seed == 945101 and implementation == 'reuse'
        assert (output / 'oracle-train-check.json').is_file()
        assert not (output / 'base.npz').exists() and not (output / 'shift.npz').exists()
        assert 'oracle_prefix' not in data and 'oracle_prefix' not in prefixes
        model, fit = real_train(arm, seed, data, prefixes, config, output, counts, work, check,
                                implementation=implementation)
        fits_called.append(arm)
        return model, fit

    monkeypatch.setattr(runner, 'generate_attempt_split', generate)
    monkeypatch.setattr(runner, 'train', train)
    monkeypatch.setattr(runner, 'make_reference', reference)
    summary = runner.run(folder, SMOKE, lambda: None)
    assert generations == [0, 1, 2]
    assert summary['version'] == 'finite-reuse-replication-v1' and summary['implementation'] == 'reuse'
    assert len(summary['rows']) == 24 and len(summary['prefix_rows']) == 6
    assert len(summary['baseline_rows']) == 8 and len(summary['prediction_times']) == 6
    assert len(summary['files']) == 49
    assert {row['regime'] for row in summary['rows']} == {'base', 'shift'}
    assert [(row['regime'], row['arm'], row['seed']) for row in summary['prefix_rows']] == [
        (regime, arm, 945101) for regime in ('base', 'shift') for arm in runner.ARMS]
    assert [epsilon for epsilon, _model in references] == [.12, .30]
    assert set(summary['oracle_metadata']) == set(summary['oracle_state_sha256']) == {'base', 'shift'}
    assert summary['oracle_state_sha256']['base'] != summary['oracle_state_sha256']['shift']
    assert summary['regimes'] == {'train': {'split_id': 0, 'epsilon': .12},
                                 'base': {'split_id': 1, 'epsilon': .12},
                                 'shift': {'split_id': 2, 'epsilon': .30}}
    for regime, epsilon in [('base', .12), ('shift', .30)]:
        assert summary['oracle_metadata'][regime]['epsilon'] == epsilon
        assert summary['dataset_counts'][regime]['attempted'] == 8
        assert summary['dataset_counts'][regime]['retained'] + summary['dataset_counts'][regime]['discarded_found'] == 8
    fits, counts = summary['fits'], summary['counts']
    assert counts['checkpoint_writes'] == counts['optimizer_checkpoint_writes'] == 9
    assert counts['accepted_optimizer_steps'] == counts['optimizer_steps'] == counts['optimizer_attempts'] == 21
    assert counts['accepted_prefix_steps'] == 9 and counts['accepted_joint_steps'] == 12
    assert counts['training_attempt_exposures'] == 33  # Three batches 3/3/2, then first batch of epoch1.
    assert counts['train_generation_count'] == 1 and counts['dev_generation_count'] == 2
    assert counts['oracle_model_constructions'] == 2
    assert counts['evaluation_case_views'] == 3 * sum(summary['dataset_counts'][r]['retained'] for r in ('base', 'shift'))
    assert counts['evaluation_prefix_event_views'] == 3 * sum(summary['dataset_counts'][r]['valid_prefix_events'] for r in ('base', 'shift'))
    assert all(counts[k] == 0 for k in ('array_decodes', 'checkpoint_decodes', 'external_model_calls', 'native_calls', 'teacher_calls'))
    paired_batches = []
    for fit in fits:
        arm = fit['arm']
        assert fit['implementation'] == 'reuse' and fit['integration_version'] == qualified.VERSION
        assert fit['joint_structural_routes'] == list(qualified.REUSE_ROUTES)
        assert fit['parameter_metadata']['parameter_count'] == 352
        allocation = json.loads((folder / fit['allocation']['path']).read_text())
        assert allocation['status'] == 'PASS' and allocation['arm'] == 'prefix_then_joint'
        assert allocation['boundary']['optimizer_reset'] is True and allocation['joint_cursor'] == 4
        assert [stage['target_updates'] for stage in allocation['stages']] == [3, 4]
        assert all(row['accepted'] and not row['rolled_back'] for row in allocation['trace'])
        joint = [row['result']['diagnostics'] for row in allocation['trace'] if row['kind'] == 'joint']
        paired_batches.append([row['indices'] for row in joint])
        for cursor, row in enumerate(joint):
            epoch, offset = divmod(cursor, 3)
            order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([945101, epoch, 818]))).permutation(8)
            assert row['indices'] == order[offset * 3:(offset + 1) * 3].tolist()
        work = summary['structural_work'][arm]
        assert work['training_blind'] == work['training_observed'] == {}
        assert work['training_prefix']['probability_field_calls'] == 3
        assert work['joint_reuse_shared']['probability_field_calls'] == 4
        assert work['joint_reuse_shared']['rounded_transition_calls'] == (4 if arm == 'rounded' else 0)
        assert work['joint_reuse_prefix_nll']['prefix_nll_rows'] == sum(row['valid_events'] for row in joint)
        assert work['joint_reuse_endpoint_prefix']['reset_emission_rows'] == sum(row['eligible'] for row in joint)
        assert work['joint_reuse_blind']['cost_readout_rows'] == 2 * sum(row['eligible'] for row in joint)
        assert work['evaluation_blind'] and work['evaluation_observed'] and work['evaluation_prefix']
        with np.load(folder / f'initial-{arm}-945101.npz', allow_pickle=False) as initial, \
                np.load(folder / f'boundary-{arm}-945101.npz', allow_pickle=False) as boundary:
            assert initial['cost_logits'].tobytes() == boundary['cost_logits'].tobytes()
    assert paired_batches[0] == paired_batches[1] == paired_batches[2]
    final_hashes = {(row['arm'], row['seed']): row['final_state_sha256'] for row in fits}
    assert {(row['regime'], row['arm'], row['seed']) for row in summary['prediction_times']} == {
        (regime, arm, 945101) for regime in ('base', 'shift') for arm in runner.ARMS}
    assert all(row['model_state_before'] == row['model_state_after'] == final_hashes[row['arm'], row['seed']]
               and row['oracle_prefix_input'] is False for row in summary['prediction_times'])
    identifiers = []
    for split, regime in enumerate(('train', 'base', 'shift')):
        with np.load(folder / f'{regime}-prefix.npz', allow_pickle=False) as saved:
            ids = set(saved['case_ids'].tolist())
            assert len(ids) == 8 and ids == {f'ns945001-split{split}-case{i:010d}' for i in range(8)}
            assert int(saved['event_mask'].sum()) == summary['dataset_counts'][regime]['valid_prefix_events']
            assert int(saved['endpoint_eligible'].sum()) == summary['dataset_counts'][regime]['retained']
            identifiers.append(ids)
    assert not identifiers[0] & identifiers[1] and not identifiers[0] & identifiers[2] and not identifiers[1] & identifiers[2]
    assert set(summary['oracle_checks']) == {'train', 'base', 'shift'}
    assert all(max(errors.values()) <= 1e-12 for errors in summary['oracle_checks'].values())
    from audit_finite_reuse_replication import audit
    audited = audit(folder, profile='engineering-945001')
    assert generations == [0, 1, 2] and fits_called == list(runner.ARMS)
    assert audited['agreement'] is True and audited['exact_oracle_agreement'] is True
    assert len(audited['rows']) == 24 and len(audited['prefix_rows']) == 6
    assert audited['counts']['array_decodes'] == 29
    assert audited['counts']['checkpoint_decodes'] == audited['counts']['optimizer_json_decodes'] == 9
    assert all(audited['counts'][key] == 0 for key in ('model_calls', 'optimizer_calls', 'world_or_generator_calls', 'native_calls'))
    assert audited['data_cases']['train'] < 256
    assert all(audited['data_cases'][regime] < 64 for regime in ('base', 'shift'))
    assert set(audited['gates']) == set(audited['allocation_comparisons']) == {'base', 'shift'}
    assert audited['advance']['passed'] is False and len(audited['advance']['conditions']) == 22
    for regime in ('base', 'shift'):
        assert len(audited['allocation_comparisons'][regime]['advance']['conditions']) == 11
        for criteria in audited['gates'][regime].values():
            for gate in criteria.values():
                assert gate['conditions']['minimum_train'] is False
                assert gate['conditions']['minimum_development'] is False
                assert gate['passed'] is False
    with monkeypatch.context() as guard:
        def forbidden_load(*args, **kwargs):
            raise AssertionError('engineering data must not enter the science-default audit')
        guard.setattr(np, 'load', forbidden_load)
        with pytest.raises(ValueError):
            audit(folder)
    with pytest.raises(FileExistsError):
        runner.run(folder, SMOKE, lambda: None)


@pytest.mark.parametrize('key,value', [
    ('prefix_updates', 0), ('prefix_updates', True), ('joint_updates', 4.),
    ('joint_updates', 100000), ('fit_cap_seconds', 30), ('fit_cap_seconds', float('nan')),
    ('fit_cap_seconds', 120.01), ('stage1_seconds', 10.), ('total_seconds', 40.),
    ('implementation', 'separate'),
])
def test_invalid_or_alternative_settings_fail_before_generation(tmp_path, monkeypatch, key, value):
    def forbidden(*args, **kwargs):
        raise AssertionError('no generation before valid registered settings')
    monkeypatch.setattr(runner, 'generate_attempt_split', forbidden)
    with pytest.raises(ValueError):
        runner.run(tmp_path / 'absent', {**SMOKE, key: value}, lambda: None)
    assert not (tmp_path / 'absent').exists()


def test_original_allocation_failure_preserves_all_partial_work_and_never_generates_dev(tmp_path, monkeypatch):
    folder = tmp_path / 'failed'
    marker = TimeoutError('original fit safety cap')
    marker.allocation_result = {'status': 'FAILED_TIMEOUT', 'trace': [{'accepted': False, 'rolled_back': True}]}
    marker.joint_reuse_work = {'shared': {'probability_field_calls': 1}}
    marker.rounded_model_work = {'probability_field_calls': 1}
    real_generate = runner.generate_attempt_split
    generations = []

    def generate(split, *args, **kwargs):
        assert split == 0
        generations.append(split)
        return real_generate(split, *args, **kwargs)

    def fail(*args, **kwargs):
        assert kwargs['prefix_updates'] == 3 and kwargs['joint_updates'] == 4
        raise marker

    monkeypatch.setattr(runner, 'generate_attempt_split', generate)
    monkeypatch.setattr(qualified, 'run_update_allocation', fail)
    with pytest.raises(TimeoutError) as caught:
        runner.run(folder, SMOKE, lambda: None)
    assert caught.value is marker and generations == [0]
    failure = json.loads((folder / 'failure.json').read_text())
    assert failure['version'] == runner.VERSION and failure['completed_fits'] == 0
    assert failure['counts']['dev_generation_count'] == 0
    assert failure['failed_joint_reuse_work'] == marker.joint_reuse_work
    assert failure['failed_route_work'] == marker.rounded_model_work
    assert json.loads((folder / 'allocation-original_free-945101.json').read_text()) == marker.allocation_result
    assert json.loads((folder / 'failed-joint-reuse-original_free-945101.json').read_text()) == marker.joint_reuse_work
    assert json.loads((folder / 'failed-normalization-original_free-945101.json').read_text()) == marker.rounded_model_work
    assert not (folder / 'checkpoint-barrier.json').exists()
    assert not (folder / 'base.npz').exists() and not (folder / 'shift.npz').exists()
    assert not (folder / 'fits.jsonl').exists()
    with pytest.raises(FileExistsError):
        runner.run(folder, SMOKE, lambda: None)
