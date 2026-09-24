"""Engineering-only integration of the frozen flow and qualified reuse fit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_finite_head_learning as runner
import run_finite_head_training as qualified
import run_finite_update_learning as frozen

SMOKE = {'seed_namespace': 946001, 'fit_seeds': [946101], 'train_attempts': 8,
         'dev_attempts': 8, 'batch_size': 3, 'prefix_updates': 3,
         'joint_updates': 4, 'fit_cap_seconds': 30.}


def test_explicit_frozen_helpers_and_two_registered_engineering_geometries():
    assert runner.train is qualified.train
    assert runner._config is frozen._config
    assert runner.predict_prefix is frozen.predict_prefix
    assert runner.verify_paired_exposure is not frozen.verify_paired_exposure
    assert runner.DEFAULT_CONFIG == {**frozen.DEFAULT_CONFIG, 'seed_namespace': 436260924, 'dev_attempts': 512,
                                    'fit_seeds': [436261001, 436261002, 436261003, 436261004, 436261005]}
    assert runner._config({**runner.DEFAULT_CONFIG, **SMOKE}) == {**runner.DEFAULT_CONFIG, **SMOKE}
    exposure = {**runner.DEFAULT_CONFIG, 'seed_namespace': 946201, 'fit_seeds': [946301],
                'dev_attempts': 8, 'prefix_updates': 32, 'joint_updates': 64, 'fit_cap_seconds': 30.}
    assert runner._config(exposure) == exposure
    work = runner.new_structural_work()
    assert set(work) == set(runner.ARMS)
    assert all(set(routes) == {*qualified.ROUTES, *qualified.REUSE_ROUTES} for routes in work.values())
    assert all(not block for routes in work.values() for block in routes.values())
    assert runner.ARMS == ('rounded_anchor', 'rounded_random', 'matched_free_random')
    assert runner.TRANSPORT_ARMS == {'rounded_anchor': 'rounded', 'rounded_random': 'rounded',
                                     'matched_free_random': 'matched_free'}


def test_all_reuse_fits_and_exact_reference_precede_dev_then_saved_output_audit(tmp_path, monkeypatch):
    folder = tmp_path / 'smoke'
    real_generate, real_train = runner.generate_attempt_split, runner.train
    real_reference = runner.make_reference
    generations, fits_called, references = [], [], []

    def reference(epsilon):
        assert epsilon == .12 and not generations and not fits_called
        result = real_reference(epsilon)
        assert not list(result.parameters())
        references.append(epsilon)
        return result

    def generate(split, attempts, horizon, **kwargs):
        assert kwargs == {'seed_namespace': 946001}
        assert (attempts, horizon) == (8, 2 if split == 0 else 8)
        if split:
            assert fits_called == list(runner.ARMS)
            barrier = json.loads((folder / 'checkpoint-barrier.json').read_text())
            assert barrier['fit_count'] == 3 and barrier['dev_generation_count'] == 0
            assert barrier['oracle_train_verified'] is True
            assert len(barrier['prefix_pair_checks']) == 1
            assert barrier['prefix_pair_checks'][0]['same_initial_and_prefix_dynamics'] is True
            assert barrier['prefix_pair_checks'][0]['same_prefix_optimizer_states'] is True
            assert barrier['prefix_pair_checks'][0]['all_heads_unchanged_after_prefix'] is True
            assert set(barrier['prefix_pair_checks'][0]['head_initial_boundary_sha256']) == set(runner.ARMS)
            assert set(barrier['paired_batch_sha256']) == {'946101'}
            assert len(barrier['checkpoints']) == 3
            assert all((folder / row['path']).is_file() for row in barrier['checkpoints'])
            assert len(list(folder.glob('initial-*.npz'))) == 3
            assert len(list(folder.glob('boundary-*.npz'))) == 3
            assert len(list(folder.glob('final-*.npz'))) == 3
            assert len(list(folder.glob('*-optimizer-*.json'))) == 9
        generations.append(split)
        return real_generate(split, attempts, horizon, **kwargs)

    def train(arm, seed, data, prefixes, config, output, counts, work, check, *, implementation):
        assert generations == [0] and seed == 946101 and implementation == 'reuse'
        assert (output / 'oracle-train-check.json').is_file()
        assert not (output / 'base.npz').exists()
        assert 'oracle_prefix' not in data and 'oracle_prefix' not in prefixes
        model, fit = real_train(arm, seed, data, prefixes, config, output, counts, work, check,
                                implementation=implementation)
        fits_called.append(arm)
        return model, fit

    monkeypatch.setattr(runner, 'generate_attempt_split', generate)
    monkeypatch.setattr(runner, 'train', train)
    monkeypatch.setattr(runner, 'make_reference', reference)
    summary = runner.run(folder, SMOKE, lambda: None)
    assert generations == [0, 1]
    assert summary['version'] == 'finite-head-learning-v1' and summary['implementation'] == 'reuse'
    assert len(summary['rows']) == 12 and len(summary['prefix_rows']) == 3
    assert len(summary['files']) == 40 and len(summary['baseline_rows']) == 4
    assert references == [.12] and summary['oracle_metadata']['epsilon'] == .12
    assert summary['regimes'] == {'train': {'split_id': 0, 'epsilon': .12},
                                  'base': {'split_id': 1, 'epsilon': .12}}
    assert summary['transport_arms'] == runner.TRANSPORT_ARMS
    assert {r['regime'] for r in summary['rows']} == {'base'}
    assert {r['regime'] for r in summary['prefix_rows']} == {'base'}
    assert summary['prefix_pair_checks'] == summary['checkpoint_barrier']['prefix_pair_checks']
    assert summary['paired_batch_sha256'] == summary['checkpoint_barrier']['paired_batch_sha256']
    assert not (folder / 'shift.npz').exists() and not (folder / 'oracle-shift.npz').exists()
    fits, counts = summary['fits'], summary['counts']
    assert counts['checkpoint_writes'] == counts['optimizer_checkpoint_writes'] == 9
    assert counts['accepted_optimizer_steps'] == counts['optimizer_steps'] == counts['optimizer_attempts'] == 21
    assert counts['accepted_prefix_steps'] == 9 and counts['accepted_joint_steps'] == 12
    assert counts['training_attempt_exposures'] == 33  # Three batches 3/3/2, then first batch of epoch1.
    assert counts['train_generation_count'] == counts['dev_generation_count'] == 1
    assert counts['oracle_model_constructions'] == 1
    assert all(counts[k] == 0 for k in ('array_decodes', 'checkpoint_decodes', 'external_model_calls', 'native_calls', 'teacher_calls'))
    paired_batches = []
    for fit in fits:
        arm = fit['arm']
        assert fit['implementation'] == 'reuse' and fit['integration_version'] == qualified.VERSION
        assert fit['joint_structural_routes'] == list(qualified.REUSE_ROUTES)
        assert fit['parameter_metadata']['parameter_count'] == 352
        assert fit['model_metadata']['arm'] == arm
        assert fit['model_metadata']['transport_arm'] == runner.TRANSPORT_ARMS[arm]
        assert fit['model_metadata']['privileged_readout_initialization'] is (arm == 'rounded_anchor')
        assert fit['head_initialization_work']['standard_normal_entries'] == (0 if arm == 'rounded_anchor' else 32)
        assert fit['dynamics_boundary_hash_evaluations'] == 3
        assert set(fit['dynamics_boundary_sha256']) == {'initial', 'boundary', 'final'}
        assert fit['head_boundary_hash_evaluations'] == 3
        assert set(fit['head_boundary_sha256']) == {'initial', 'boundary', 'final'}
        assert fit['head_boundary_sha256']['initial'] == fit['head_boundary_sha256']['boundary']
        allocation = json.loads((folder / fit['allocation']['path']).read_text())
        assert allocation['status'] == 'PASS' and allocation['arm'] == 'prefix_then_joint'
        assert allocation['boundary']['optimizer_reset'] is True and allocation['joint_cursor'] == 4
        assert [stage['target_updates'] for stage in allocation['stages']] == [3, 4]
        assert all(row['accepted'] and not row['rolled_back'] for row in allocation['trace'])
        joint = [row['result']['diagnostics'] for row in allocation['trace'] if row['kind'] == 'joint']
        paired_batches.append([row['indices'] for row in joint])
        for cursor, row in enumerate(joint):
            epoch, offset = divmod(cursor, 3)
            order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([946101, epoch, 818]))).permutation(8)
            assert row['indices'] == order[offset * 3:(offset + 1) * 3].tolist()
        work = summary['structural_work'][arm]
        assert work['training_blind'] == work['training_observed'] == {}
        assert work['training_prefix']['probability_field_calls'] == 3
        assert work['joint_reuse_shared']['probability_field_calls'] == 4
        assert work['joint_reuse_shared']['rounded_transition_calls'] == (4 if arm.startswith('rounded_') else 0)
        assert work['joint_reuse_prefix_nll']['prefix_nll_rows'] == sum(row['valid_events'] for row in joint)
        assert work['joint_reuse_endpoint_prefix']['reset_emission_rows'] == sum(row['eligible'] for row in joint)
        assert work['joint_reuse_blind']['cost_readout_rows'] == 2 * sum(row['eligible'] for row in joint)
        assert work['evaluation_blind'] and work['evaluation_observed'] and work['evaluation_prefix']
        with np.load(folder / f'initial-{arm}-946101.npz', allow_pickle=False) as initial, \
                np.load(folder / f'boundary-{arm}-946101.npz', allow_pickle=False) as boundary:
            assert initial['cost_logits'].tobytes() == boundary['cost_logits'].tobytes()
    assert paired_batches[0] == paired_batches[1] == paired_batches[2]
    for label in ('initial', 'boundary'):
        with np.load(folder / f'{label}-rounded_anchor-946101.npz', allow_pickle=False) as anchor, \
                np.load(folder / f'{label}-rounded_random-946101.npz', allow_pickle=False) as random:
            for name in ('transition_logits', 'emission_logits', 'hazard_logits'):
                assert anchor[name].tobytes() == random[name].tobytes()
            assert anchor['cost_logits'].tobytes() != random['cost_logits'].tobytes()
        anchor_opt = json.loads((folder / f'{label}-optimizer-rounded_anchor-946101.json').read_text())
        random_opt = json.loads((folder / f'{label}-optimizer-rounded_random-946101.json').read_text())
        assert anchor_opt == random_opt
    with np.load(folder / 'initial-rounded_random-946101.npz', allow_pickle=False) as rounded, \
            np.load(folder / 'initial-matched_free_random-946101.npz', allow_pickle=False) as free:
        assert rounded['cost_logits'].tobytes() == free['cost_logits'].tobytes()
    for regime, split in (('train', 0), ('base', 1)):
        with np.load(folder / f'{regime}-prefix.npz', allow_pickle=False) as saved:
            assert set(saved['case_ids'].tolist()) == {f'ns946001-split{split}-case{i:010d}' for i in range(8)}
            assert int(saved['event_mask'].sum()) == summary['dataset_counts'][regime]['valid_prefix_events']
            assert int(saved['endpoint_eligible'].sum()) == summary['dataset_counts'][regime]['retained']
    final_hashes = {(r['arm'], r['seed']): r['final_state_sha256'] for r in fits}
    assert all(r['model_state_before'] == r['model_state_after'] == final_hashes[r['arm'], r['seed']]
               and r['oracle_prefix_input'] is False for r in summary['prediction_times'])
    assert all(max(errors.values()) <= 1e-12 for errors in summary['oracle_checks'].values())
    from audit_finite_head_learning import audit
    audited = audit(folder, profile='engineering-946001')
    assert generations == [0, 1] and fits_called == list(runner.ARMS)
    assert audited['agreement'] is True and audited['exact_oracle_agreement'] is True
    assert len(audited['rows']) == 12 and len(audited['prefix_rows']) == 3
    assert audited['counts']['array_decodes'] == 21
    assert audited['counts']['checkpoint_decodes'] == audited['counts']['optimizer_json_decodes'] == 9
    assert all(audited['counts'][key] == 0 for key in ('model_calls', 'optimizer_calls', 'world_or_generator_calls', 'native_calls'))
    assert set(audited['gates']) == set(runner.ARMS) and audited['advance']['passed'] is False
    for criteria in audited['gates'].values():
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
    assert json.loads((folder / 'allocation-rounded_anchor-946101.json').read_text()) == marker.allocation_result
    assert json.loads((folder / 'failed-joint-reuse-rounded_anchor-946101.json').read_text()) == marker.joint_reuse_work
    assert json.loads((folder / 'failed-normalization-rounded_anchor-946101.json').read_text()) == marker.rounded_model_work
    assert not (folder / 'checkpoint-barrier.json').exists()
    assert not (folder / 'base.npz').exists() and not (folder / 'fits.jsonl').exists()
    with pytest.raises(FileExistsError):
        runner.run(folder, SMOKE, lambda: None)


def _pair_fixture(folder):
    """Fabricated JSON only, with no numerical fit or checkpoint decoding."""
    fits = []
    for i, arm in enumerate(runner.ARMS):
        checkpoints = [{'label': label, 'metadata': {
            'dynamics_state_sha256': str(j + (i if label == 'final' else 0)) * 64,
            'head_state_sha256': str(i + (3 if label == 'final' else 0)) * 64,
            'optimizer_state_sha256': str(j + 4) * 64}}
            for j, label in enumerate(('initial', 'boundary', 'final'))]
        trace = [{'kind': 'prefix', 'accepted': True, 'rolled_back': False}] * 3
        trace += [{'kind': 'joint', 'accepted': True, 'rolled_back': False,
                   'result': {'diagnostics': {'indices': indices}}}
                  for indices in ([0, 1, 2], [3, 4, 5], [6, 7], [1, 3, 0])]
        value = {'status': 'PASS', 'accepted_prefix_updates': 3, 'accepted_joint_updates': 4,
            'joint_cursor': 4, 'attempted_updates': 7, 'accepted_updates': 7,
            'trace': trace, 'checkpoints': checkpoints}
        path = folder / f'allocation-{arm}-946101.json'
        path.write_text(json.dumps(value))
        fits.append({'arm': arm, 'seed': 946101, 'allocation': {'path': path.name, **runner._desc(path)},
            'accepted_prefix_updates': 3, 'updates': 4, 'attempted_updates': 7,
            'dynamics_boundary_sha256': {row['label']: row['metadata']['dynamics_state_sha256']
                                        for row in checkpoints}, 'dynamics_boundary_hash_evaluations': 3,
            'head_boundary_sha256': {row['label']: row['metadata']['head_state_sha256']
                                    for row in checkpoints}, 'head_boundary_hash_evaluations': 3})
    return fits


def test_new_arm_pair_checks_bind_saved_json_without_loading_arrays(tmp_path, monkeypatch):
    fits = _pair_fixture(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError('prefix pairing must not decode checkpoints')

    monkeypatch.setattr(np, 'load', forbidden)
    assert set(runner.verify_paired_exposure(fits, tmp_path, SMOKE)) == {946101}
    pair = runner.verify_prefix_pairing(fits, tmp_path, SMOKE)[0]
    assert pair['arms'] == ['rounded_anchor', 'rounded_random']
    assert pair['same_initial_and_prefix_dynamics'] is True
    assert pair['same_prefix_optimizer_states'] is True
    assert pair['all_heads_unchanged_after_prefix'] is True
    assert fits[0]['dynamics_boundary_sha256']['final'] != fits[1]['dynamics_boundary_sha256']['final']


@pytest.mark.parametrize('label,key', [
    ('initial', 'dynamics_state_sha256'), ('boundary', 'dynamics_state_sha256'),
    ('initial', 'optimizer_state_sha256'), ('boundary', 'optimizer_state_sha256')])
def test_mismatched_rounded_prefix_dynamics_or_adam_rejected(tmp_path, label, key):
    fits = _pair_fixture(tmp_path)
    fit = fits[1]
    path = tmp_path / fit['allocation']['path']
    value = json.loads(path.read_text())
    point = next(row for row in value['checkpoints'] if row['label'] == label)
    point['metadata'][key] = 'f' * 64
    if key == 'dynamics_state_sha256':
        fit['dynamics_boundary_sha256'][label] = 'f' * 64
    path.write_text(json.dumps(value))
    fit['allocation'].update(runner._desc(path))
    with pytest.raises(ValueError, match='rounded head arms'):
        runner.verify_prefix_pairing(fits, tmp_path, SMOKE)


def test_prefix_pairing_rejects_unbound_fit_hash_or_changed_allocation(tmp_path):
    fits = _pair_fixture(tmp_path)
    fits[0]['dynamics_boundary_sha256']['boundary'] = 'e' * 64
    with pytest.raises(ValueError, match='fit dynamics hashes'):
        runner.verify_prefix_pairing(fits, tmp_path, SMOKE)
    fits = _pair_fixture(tmp_path)
    path = tmp_path / fits[0]['allocation']['path']
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='unchanged allocation'):
        runner.verify_prefix_pairing(fits, tmp_path, SMOKE)


def test_local_pairing_checks_every_new_arm_joint_batch(tmp_path):
    fits = _pair_fixture(tmp_path)
    fit = fits[-1]
    path = tmp_path / fit['allocation']['path']
    value = json.loads(path.read_text())
    value['trace'][-1]['result']['diagnostics']['indices'] = [1, 3, 2]
    path.write_text(json.dumps(value))
    fit['allocation'].update(runner._desc(path))
    with pytest.raises(ValueError, match='identical ordered joint minibatches'):
        runner.verify_paired_exposure(fits, tmp_path, SMOKE)


@pytest.mark.parametrize('arm', runner.ARMS)
def test_prefix_stage_head_change_in_any_arm_rejected(tmp_path, arm):
    fits = _pair_fixture(tmp_path)
    fit = next(row for row in fits if row['arm'] == arm)
    path = tmp_path / fit['allocation']['path']
    value = json.loads(path.read_text())
    point = next(row for row in value['checkpoints'] if row['label'] == 'boundary')
    point['metadata']['head_state_sha256'] = 'f' * 64
    fit['head_boundary_sha256']['boundary'] = 'f' * 64
    path.write_text(json.dumps(value))
    fit['allocation'].update(runner._desc(path))
    with pytest.raises(ValueError, match='every head remains unchanged'):
        runner.verify_prefix_pairing(fits, tmp_path, SMOKE)


def test_unbound_fit_head_hash_rejected(tmp_path):
    fits = _pair_fixture(tmp_path)
    fits[0]['head_boundary_sha256']['final'] = 'f' * 64
    with pytest.raises(ValueError, match='fit head hashes'):
        runner.verify_prefix_pairing(fits, tmp_path, SMOKE)
