"""Bounded engineering runner checks, never scored data/model evaluation.

Checkpoint fixtures are the previously completed tiny engineering rehearsal.
Native checks use literal seed410 and tiny candidate horizons only. Preparation
and full lifecycle wiring are mocked without manufacturing a ready protocol.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import reacher_geometry_memory_study as study
import torch

from openjev.research import reacher_geometry_memory_protocol as protocol
from openjev.research.reacher_adaptive_search import SearchInputs
from openjev.research.robotics_reacher import native_replay


@pytest.fixture(autouse=True)
def isolated_engineering_rng():
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


def engineering_plan():
    plan = protocol.settings(engineering=True)
    plan.update(engineering=True, control_episodes=1, hidden_size=4, mlp_width=7,
                bootstrap_samples=8, particles=2, threads=1, cap_seconds=300, audit_cap_seconds=300)
    plan['random_stream_contract'] = {'namespace': plan['rng_namespace'], 'registry': protocol.registry(plan)}
    return plan


def inputs(h=2, block=1):
    rng = np.random.default_rng(410)
    chunks = (h + block - 1) // block
    arrays = [rng.normal(size=(1, k, chunks, 2)) for k in (64, 192, 64, 64, 63)]
    return SearchInputs(arrays[0], arrays[1], tuple(arrays[2:]))


def records(stem):
    meta = study.read(stem.with_suffix('.json'))
    with np.load(stem.with_suffix('.npz'), allow_pickle=False) as arrays:
        result = []
        for index, item in enumerate(meta):
            record = {'metadata': item, 'policy': {}, 'audit': {}}
            for key in arrays.files:
                group, field = key.split('__', 1)
                record[group][field] = arrays[key][index].copy()
            result.append(record)
        return result


@pytest.fixture
def inherited_engineering(tmp_path):
    cache_path = study.ROOT / 'output/reacher-cache-rehearsal-v1/attempt-01'
    geometry_path = study.ROOT / 'output/reacher-geometry-rehearsal-v1/attempt-02'
    if not (cache_path / 'audit/receipt.json').exists() or not (geometry_path / 'audit/receipt.json').exists():
        pytest.skip('Optional completed engineering evidence is absent')
    parent, source = study.authenticate_evidence(cache_path / 'plan.json', study.sha(cache_path / 'plan.json'),
        cache_path / 'audit/receipt.json', study.sha(cache_path / 'audit/receipt.json'), cache_path / 'execution',
        kind='cache', engineering=True)
    _, context = study.authenticate_evidence(geometry_path / 'plan.json', study.sha(geometry_path / 'plan.json'),
        geometry_path / 'audit/receipt.json', study.sha(geometry_path / 'audit/receipt.json'), geometry_path / 'execution',
        kind='geometry', engineering=True)
    plan = engineering_plan()
    plan.update(hidden_size=parent['hidden_size'], mlp_width=parent['mlp_width'], runtime=parent['runtime'],
                cache_source=source, geometry_source=context)
    out = tmp_path / 'execution'
    out.mkdir()
    study.copy_inherited(plan, out, float('inf'))
    return plan, out


def test_membership_includes_every_twelve_fit_and_no_exposed_control_histories():
    names = protocol.inherited_members(protocol.settings())
    assert len(names) == 74
    assert sum(name.endswith('checkpoint.pt') for name in names) == 12
    assert not any(name.startswith('control/') for name in names)
    assert len(study.SOURCES) == len(set(study.SOURCES)) == 90


def test_restore_actual_twelve_classes_without_optimizer_forward_or_rng_leak(inherited_engineering):
    plan, out = inherited_engineering
    before = torch.get_rng_state().clone()
    with patch.object(torch.optim, 'Adam', side_effect=AssertionError('No optimizer')), \
         patch.object(study.training, 'CacheTrainer', side_effect=AssertionError('No trainer')):
        models = study.restore_students(plan, out, float('inf'))
    assert torch.equal(before, torch.get_rng_state())
    assert list(models) == plan['fit_order'] and len(models) == 12
    receipt = study.read(out / 'all-models-restored.json')
    assert receipt['optimizer_constructed'] is False and receipt['new_updates'] == 0
    for row in protocol.fit_manifest(plan):
        model = models[row['name']]
        assert type(model).__name__ == row['model_class']
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        weights = torch.load(out / 'inherited' / 'fits' / row['name'] / 'weights.pt', weights_only=True)
        snapshot = torch.load(out / 'model-states' / f"{row['name']}-before.pt", weights_only=True)
        for key, value in model.state_dict().items():
            assert torch.equal(value, weights[key]) and torch.equal(value, snapshot[key])


@pytest.mark.parametrize('field,value', [('kind', 'cached_gru'), ('failed', True),
    ('cursor', {'epoch': 0, 'batch': 0}), ('settings', {'bad': True}), ('data_sha256', '0' * 64),
    ('runtime', {'bad': True}), ('source_sha256', {}), ('model_configuration', {'bad': True})])
def test_restore_rejects_resealed_mismatch_before_constructor(inherited_engineering, field, value):
    plan, out = inherited_engineering
    original_load = torch.load
    def tampered(path, *args, **kwargs):
        result = original_load(path, *args, **kwargs)
        if Path(path).name == 'checkpoint.pt':
            result = copy.deepcopy(result)
            result[field] = value
            result['integrity_sha256'] = study.training.canonical_state_hash({k: v for k, v in result.items() if k != 'integrity_sha256'})
        return result
    with patch.object(torch, 'load', side_effect=tampered), patch.object(study.training, '_construct', side_effect=AssertionError('No invalid model')), \
         pytest.raises(ValueError, match='checkpoint|schedule'):
        study.restore_students(plan, out, float('inf'))


def test_copied_checkpoint_mutation_fails_external_hash_before_loading(inherited_engineering):
    plan, out = inherited_engineering
    path = out / 'inherited/fits/cached_gru-pair0/weights.pt'
    path.write_bytes(path.read_bytes() + b'x')
    with patch.object(torch, 'load', side_effect=AssertionError('No load')), \
         pytest.raises(ValueError, match='identity mismatch'):
        study.restore_students(plan, out, float('inf'))


@pytest.mark.parametrize('scalar,value', [('noise_std', .07), ('dt', .01), ('hidden_size', 5), ('mlp_width', 8), ('residual_reward', False)])
def test_deployment_scalars_bound_to_parent(inherited_engineering, scalar, value):
    plan, out = inherited_engineering
    plan[scalar] = value
    with pytest.raises(ValueError, match='deployment'):
        study.restore_students(plan, out, float('inf'))


def test_prepare_only_writes_unrun_plan_without_models_or_draws(tmp_path):
    cfg = protocol.settings()
    parent = {key: cfg[key] for key in ('hidden_size', 'mlp_width', 'dt', 'noise_std', 'residual_reward')}
    with patch.object(study, 'authenticated_sources', return_value=(parent, {}, {'sources': {}}, {})), \
         patch.object(study, 'source_hashes', return_value={name: '0' * 64 for name in study.SOURCES}), \
         patch.object(study, 'stream_contract', return_value={'fixture': True}), \
         patch.object(study.training, '_construct', side_effect=AssertionError('No model')), \
         patch.object(protocol, 'draw_control_inputs', side_effect=AssertionError('No draw')):
        result = study.prepare(tmp_path / 'plan', cap_seconds=123, audit_cap_seconds=321)
    saved = study.read(tmp_path / 'plan/plan.json')
    assert result['ready_to_launch'] is False and saved['new_fits'] == 0
    assert saved['cap_seconds'] == 123 and saved['audit_cap_seconds'] == 321
    assert len(saved['execution_order']) == 51


def test_missing_independent_auditor_prevents_preparation(tmp_path):
    cfg = protocol.settings()
    with patch.object(study, 'authenticated_sources', return_value=(cfg, {}, {'sources': {}}, {})), \
         patch.object(study, 'source_hashes', side_effect=FileNotFoundError('independent auditor absent')), \
         patch.object(study, 'stream_contract', side_effect=AssertionError('No manifest after missing source')), \
         pytest.raises(FileNotFoundError):
        study.prepare(tmp_path / 'plan', cap_seconds=100, audit_cap_seconds=100)
    assert not (tmp_path / 'plan').exists()


@pytest.mark.parametrize('reference', protocol.REFERENCES)
def test_real_engineering_reference_rows_replay_and_preserve_privilege_boundary(reference, tmp_path, monkeypatch):
    cfg = engineering_plan()
    cfg.update(planning_horizon=2, action_block=1)
    schedule = np.ones(51, bool)
    schedule[8:14] = schedule[28:34] = False
    cases = [study.control.ControlCase(410, 410, schedule)]
    draws = inputs()
    audit_method = study.ReacherEpisode.audit_record
    privileged = 0
    def read_audit(self):
        nonlocal privileged
        privileged += 1
        if reference != 'known_state':
            raise AssertionError('Public reference read hidden state')
        return audit_method(self)
    monkeypatch.setattr(study.ReacherEpisode, 'audit_record', read_audit)
    out = tmp_path / reference
    timing = study.reference_control(cfg, 'ordinary', reference, [draws] * 50, out, cases=cases)
    record = records(out / 'episodes')[0]
    assert native_replay(record)['transitions'] == 50
    assert privileged == (50 if reference == 'known_state' else 0)
    assert timing['row_wall_seconds'] >= timing['setup_seconds'] + sum(timing['decision_seconds']) + sum(timing['native_step_seconds'])
    with np.load(out / 'planning.npz', allow_pickle=False) as planning:
        assert planning['candidate_scores'].shape == (1, 50, 256)
        if reference in protocol.PHYSICS_REFERENCES:
            np.testing.assert_array_equal(planning['public_packets'][0], record['policy']['packets'][:-1])
            np.testing.assert_array_equal(planning['previous_commands'][0, 1:], record['policy']['commands'][:-1])
            for step in (0, 8, 49):
                with np.load(out / 'physics' / f'{step:03d}.npz', allow_pickle=False) as saved, \
                     np.load(out / 'decisions' / f'{step:03d}.npz', allow_pickle=False) as trace:
                    roots = np.concatenate((saved['root__qpos'], saved['root__qvel']), axis=1)
                    np.testing.assert_array_equal(roots, planning['root_estimates'][:, step])
                    np.testing.assert_array_equal(saved['selected__qpos'][:, 0, 0], saved['root__qpos'])
                    np.testing.assert_array_equal(saved['selected__commands'][:, 0, 0], record['policy']['commands'][None, step])
                    raw = np.concatenate([saved[f'bank{i}__geometry_reward'] for i in range(4)], axis=1)
                    np.testing.assert_array_equal(raw, trace['raw_rewards'])
            final = study.read(out / 'physics-final.json')
            assert final['lifetime']['candidate_sequences_scored'] == 50 * 257
            assert final['lifetime']['native_transitions_completed'] == 256 * 99 + 50
        else:
            assert not planning['candidate_scores'].any() and not planning['planner_used']
            assert not (out / 'physics').exists() and not (out / 'decisions').exists()
            if reference == 'zero':
                assert not record['policy']['commands'].any()


def test_interrupted_physics_retains_original_exception_native_prefix_and_no_retry(tmp_path, monkeypatch):
    cfg = engineering_plan()
    cfg.update(planning_horizon=2, action_block=1)
    cases = [study.control.ControlCase(410, 410, np.ones(51, bool))]
    original = study.PhysicsGeometryCEM._bank
    sentinel = KeyboardInterrupt('engineering after paid callback')
    count = 0
    def interrupted(self, *args, **kwargs):
        nonlocal count
        original(self, *args, **kwargs)
        count += 1
        raise sentinel
    monkeypatch.setattr(study.PhysicsGeometryCEM, '_bank', interrupted)
    out = tmp_path / 'failed'
    with pytest.raises(KeyboardInterrupt) as error:
        study.reference_control(cfg, 'full', 'known_state', [inputs()] * 50, out, cases=cases)
    assert error.value is sentinel and count == 1
    assert study.read(out / 'failed.json')['completed_steps_by_case'] == [0]
    assert study.read(out / 'partial-physics.json')['status'] == 'failed'
    with np.load(out / 'partial-physics.npz', allow_pickle=False) as arrays:
        assert arrays['bank0__native_completed'].all()
        assert 'bank1__commands' not in arrays.files


def test_preflight_failure_preserved_in_exclusive_output(tmp_path):
    sentinel = RuntimeError('source authentication failed')
    out = tmp_path / 'attempt'
    with patch.object(study, 'validate', side_effect=sentinel), pytest.raises(RuntimeError) as error:
        study.run(tmp_path / 'missing-plan', '0' * 64, out)
    assert error.value is sentinel
    assert study.read(out / 'failed.json')['progress']['phase'] == 'validate-or-setup'
    with pytest.raises(FileExistsError):
        study.run(tmp_path / 'missing-plan', '0' * 64, out)


def test_full_lifecycle_wires_all51_rows_only_after_all12_restores(inherited_engineering, tmp_path, monkeypatch):
    plan, prepared = inherited_engineering
    plan['particles'] = 32  # Fixed public protocol value, regardless stub execution.
    order = []
    original_restore = study.restore_students
    def copy_inherited(cfg, out, deadline):
        import shutil
        shutil.copytree(prepared / 'inherited', out / 'inherited')
        study.write(out / 'inheritance.json', {'cache': cfg['cache_source'], 'geometry': cfg['geometry_source']})
    def restore(cfg, out, deadline):
        values = original_restore(cfg, out, deadline)
        order.append(('restore', len(values)))
        return values
    def draw(cfg, step):
        assert order and order[0] == ('restore', 12)
        order.append(('draw', step))
        return inputs(12, 3)
    def row(cfg, *args, cases, **kwargs):
        assert len([entry for entry in order if entry[0] == 'draw']) == 50
        order.append(('row', len(cases)))
        return {'setup_seconds': 0., 'decision_seconds': [0.] * 50,
                'native_step_seconds': [0.] * 50, 'row_wall_seconds': 0.}
    monkeypatch.setattr(study, 'copy_inherited', copy_inherited)
    monkeypatch.setattr(study, 'restore_students', restore)
    monkeypatch.setattr(protocol, 'draw_control_inputs', draw)
    monkeypatch.setattr(study.control, 'learned_control', row)
    monkeypatch.setattr(study, 'reference_control', row)
    out = tmp_path / 'whole-wiring'
    checks = []
    result = study.run_validated(plan, 'f' * 64, out, final_validate=lambda: checks.append(True))
    assert result['control_rows'] == 51 and checks == [True, True]
    assert len([entry for entry in order if entry[0] == 'row']) == 51
    done = study.read(out / 'completed.json')
    assert done['restored_models'] == 12 and done['diagnostic_roots'] == done['new_fits'] == 0
    assert not (out / 'diagnostic-completed.json').exists()
    initial = study.read(out / 'all-models-restored.json')
    final = study.read(out / 'final-models.json')
    assert final['student_tensor_sha256'] == {name: item['student_tensor_sha256'] for name, item in initial['models'].items()}
    assert done['cumulative_attempt_wall_seconds'] == plan['geometry_source']['prior_costs']['cumulative_attempt_wall_seconds'] + done['wall_seconds']
    assert done['files'] == {key: value for key, value in study.file_members(out).items() if key != 'completed.json'}


def test_cap_before_restore_never_draws_and_preserves_failure(tmp_path):
    plan = engineering_plan()
    plan['particles'] = 32
    with patch.object(protocol, 'draw_control_inputs', side_effect=AssertionError('No draw')), \
         pytest.raises(TimeoutError):
        study.run_validated(plan, '0' * 64, tmp_path / 'cap', begin=time.monotonic() - 400,
                            final_validate=lambda: None)
    assert study.read(tmp_path / 'cap/failed.json')['progress']['phase'] == 'initial-validation'


def test_ragged_native_failure_preserves_each_initialized_case(tmp_path, monkeypatch):
    cfg = engineering_plan()
    cfg['control_episodes'] = 2
    schedule = np.ones(51, bool)
    cases = [study.control.ControlCase(410, 410, schedule) for _ in range(2)]
    original = study.ReacherEpisode.step
    sentinel = KeyboardInterrupt('engineering second native case')
    calls = 0
    def interrupted(self, action):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise sentinel
        return original(self, action)
    monkeypatch.setattr(study.ReacherEpisode, 'step', interrupted)
    out = tmp_path / 'ragged'
    with pytest.raises(KeyboardInterrupt) as error:
        study.reference_control(cfg, 'full', 'zero', [None] * 50, out, cases=cases)
    assert error.value is sentinel
    assert study.read(out / 'failed.json')['completed_steps_by_case'] == [1, 0]
    assert study.read(out / 'partial-episodes/manifest.json')['completed_steps_by_case'] == [1, 0]
    assert len(records(out / 'partial-episodes/000')[0]['policy']['commands']) == 1
    assert len(records(out / 'partial-episodes/001')[0]['policy']['commands']) == 0


def test_failure_receipt_error_never_masks_original_baseexception(tmp_path):
    sentinel = KeyboardInterrupt('original preflight failure')
    with patch.object(study, 'validate', side_effect=sentinel), \
         patch.object(study, 'write', side_effect=OSError('storage unavailable')), \
         pytest.raises(KeyboardInterrupt) as error:
        study.run(tmp_path / 'plan', '0' * 64, tmp_path / 'attempt')
    assert error.value is sentinel and any('preservation failed' in note for note in sentinel.__notes__)
