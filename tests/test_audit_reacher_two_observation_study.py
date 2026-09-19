"""Handbuilt saved artifacts and mocked orchestration, never scientific work.

The fitting fixture imports a handbuilt saved-tensor builder using the already
excluded literal410 historical permutation generator. No model, optimizer or
native calls occur. Full real engineering integration is a separate prerequisite.
"""
import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_reacher_two_observation_study as audit

from openjev.research import reacher_two_observation_experiment as experiment
from openjev.research.reacher_adaptive_search import SearchInputs


def settings():
    value = audit.protocol.settings(engineering=True)
    value.update(control_episodes=1, train_episodes=3, epochs=2, batch_size=2, hidden_size=2, threads=1)
    return value


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + '\n')


def test_complete_membership_independently_matches_all42rows_and_nine_models():
    plan = settings()
    wanted = audit.expected_members(plan)
    assert wanted == set(experiment.expected_members(plan))
    assert len(wanted) == 8119
    assert len([p for p in wanted if p.startswith('inherited/') and '/lineage/' not in p]) == 44
    assert len([p for p in wanted if p.startswith('model-states/')]) == 24
    assert len([p for p in wanted if '/episodes.npz' in p]) == 42
    assert len([p for p in wanted if p.startswith('fits/')]) == 15
    assert len([p for p in wanted if '/controller-decisions/' in p]) == 450
    assert 'control/shift/two_observation_gru-pair2/completed.json' in wanted
    assert not any('prediction' in p and 'executed_predictions' not in p for p in wanted)


def test_numerical_kernel_view_does_not_mutate_bound_settings_or_legacy_registries():
    value = settings()
    before = copy.deepcopy(value)
    marker = object()
    kernel = audit.kernel_plan(value, marker)
    assert value == before and kernel['_settings'] is value and kernel['_streams'] is marker
    assert 'score_contract' not in value and kernel['score_contract']['geometry']['noise_std'] == .05
    assert audit.memory.protocol.__name__.endswith('reacher_geometry_memory_protocol')


def test_frozen_wrapper_seed_resolution_replaced_by_explicit_bound_handle(monkeypatch):
    value, handle = settings(), object()
    calls = []
    monkeypatch.setattr(audit.streams, 'seed', lambda p, r, *, contract: calls.append((p, r, contract)) or 17)
    monkeypatch.setattr(audit.streams, 'schedule', lambda p, i, panel, *, contract: calls.append((p, i, panel, contract)) or 'schedule')
    kernel = audit.kernel_plan(value, handle)
    assert audit.bound_seed(kernel, 'control/reset/0') == 17
    assert audit.bound_schedule(kernel, 0, 'shift') == 'schedule'
    assert calls[0] == (value, 'control/reset/0', handle)
    assert calls[1] == (value, 0, 'shift', handle)


def innovation_fixture(tmp_path, monkeypatch):
    value, marker = settings(), object()
    inputs = SearchInputs(np.zeros((1, 64, 4, 2)), np.zeros((1, 192, 4, 2)),
        tuple(np.full((1, n, 4, 2), i, np.float64) for i, n in enumerate((64, 64, 63), 1)))
    calls = []
    monkeypatch.setattr(audit.streams, 'draw_control_inputs', lambda p, step, *, contract: calls.append((p, step, contract)) or inputs)
    names, values = ('initial', 'random_extra', 'cem/1', 'cem/2', 'cem/3'), (inputs.initial, inputs.random_extra, *inputs.cem)
    stem = tmp_path / '049'
    np.savez(stem.with_suffix('.npz'), **{name.replace('/', '_'): a for name, a in zip(names, values, strict=True)})
    put(stem.with_suffix('.json'), {'prefix': 'planner/control/49', 'input_identities': dict(inputs.identities()),
        'shapes': {name: list(a.shape) for name, a in zip(names, values, strict=True)}, 'unused_anchor_draws': 7, 'full_horizon_innovations': True})
    return audit.kernel_plan(value, marker), stem, inputs, calls


def test_saved_innovations_include_all_unused_draws_at_short_terminal_horizon(tmp_path, monkeypatch):
    plan, stem, inputs, calls = innovation_fixture(tmp_path, monkeypatch)
    assert audit.audit_innovations(plan, stem, 49) is inputs
    assert calls == [(plan['_settings'], 49, plan['_streams'])]


@pytest.mark.parametrize('field', ['random_extra', 'cem_3', 'initial'])
def test_saved_innovation_corruption_rejected_even_unused(field, tmp_path, monkeypatch):
    plan, stem, _, _ = innovation_fixture(tmp_path, monkeypatch)
    values = dict(np.load(stem.with_suffix('.npz')))
    values[field][0, 0, 0, 0] += .25
    np.savez(stem.with_suffix('.npz'), **values)
    with pytest.raises(ValueError, match='innovations'):
        audit.audit_innovations(plan, stem, 49)


def handbuilt_fit(tmp_path):
    path = ROOT / 'tests/test_reacher_two_observation_training_audit.py'
    spec = importlib.util.spec_from_file_location('manual_saved_training', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = module.fixture()
    row = {'name': 'two_observation_gru-pair0', 'pair': 'pair0', 'arm': 'two_observation_gru',
        'initialization_path': 'initializations/pair0.pt', 'order_path': 'orders/pair0.pt'}
    plan = {'settings': case['settings'], 'sources': case['source_sha256'], 'runtime': case['runtime'],
        'lineage': {'cache': {'plan_sha256': 'b' * 64, 'audit_receipt_sha256': 'c' * 64,
            'members': dict.fromkeys((row['initialization_path'], row['order_path'], 'train.npz', 'train.json'), 'd' * 64)}}}
    expected = 'e' * 64
    case['provenance'] = audit.fit_provenance(plan, expected, row)
    case['checkpoint']['provenance'] = case['provenance']
    module.seal(case['checkpoint'])
    folder = tmp_path / 'fits' / row['name']
    folder.mkdir(parents=True)
    for name, payload in [('initial-weights.pt', case['initial_weights']), ('weights.pt', case['final_weights']), ('checkpoint.pt', case['checkpoint'])]:
        torch.save(payload, folder / name)
    (folder / 'training.jsonl').write_text(''.join(json.dumps(log) + '\n' for log in case['logs']))
    done = {'version': 'reacher-two-observation-fitting-v1', 'status': 'completed', **{key: row[key] for key in ('name', 'pair', 'arm')},
        'model_configuration': audit.training_audit.model_configuration(case['settings']), 'settings': case['settings'],
        'provenance': case['provenance'], 'source_sha256': plan['sources'], 'runtime': plan['runtime'],
        'updates': 4, 'optimizer_steps': 4, 'flushed_updates': 4, 'cursor': {'epoch': 2, 'batch': 0},
        'initialization_sha256': case['expected_initial_sha256'], 'orders_sha256': case['expected_orders_sha256'],
        'orders_hash_scope': 'canonical_state_hash of entire sealed payload including integrity field',
        'data_sha256': case['expected_data_sha256'], 'student_tensor_sha256': audit.tensor_hash(case['final_weights']),
        'checkpoint_integrity_sha256': case['checkpoint']['integrity_sha256'], 'log_chain_sha256': case['checkpoint']['log_chain_sha256'],
        'parameters': 163, 'constructor_seconds': .02, 'trainer_setup_seconds': .01, 'training_wall_seconds': .24,
        'update_call_seconds': .25, 'log_validation_flush_seconds': .01, 'payload_export_write_seconds': .01,
        'file_hash_seconds': .01, 'wall_seconds': .31,
        'files': {name: audit.sha(folder / name) for name in ('initial-weights.pt', 'weights.pt', 'checkpoint.pt', 'training.jsonl')},
        'timing_scope': 'Full fit through file hashes, excluding only completed.json write; nested timing scopes are not additive. Cap also checked after terminal write.',
        'scope': 'One externally authenticated fresh fit; not a whole-study audit or authorization.'}
    put(folder / 'completed.json', done)
    return plan, expected, row, case, folder


def run_fit_fixture(plan, expected, row, case, folder):
    return audit.audit_new_fit(plan, expected, folder.parents[1], row, case['public_data'],
        {'states': {'gru': case['initial_weights']}}, case['orders'])


def test_complete_serialized_manual_fit_checked_without_models_optimizers_or_native(tmp_path, monkeypatch):
    plan, expected, row, case, folder = handbuilt_fit(tmp_path)
    fail = lambda *a, **k: (_ for _ in ()).throw(AssertionError('No learned/optimizer/native call'))
    monkeypatch.setattr(torch.nn.GRUCell, '__init__', fail)
    monkeypatch.setattr(torch.optim.Adam, '__init__', fail)
    monkeypatch.setattr(audit.search_audit.native, 'make_env', fail)
    result = run_fit_fixture(plan, expected, row, case, folder)
    assert result['new_optimizer_updates'] == 4
    assert result['saved_training_audit']['work']['replayed_transition_samples'] == 264
    assert result['saved_training_audit']['new_model_calls'] == 0


@pytest.mark.parametrize('field,value', [('flushed_updates', 3), ('orders_sha256', 'a' * 64), ('parameters', 164),
    ('wall_seconds', .01), ('trainer_setup_seconds', .011), ('training_wall_seconds', .25),
    ('scope', 'verified numerical training'), ('data_sha256', 'a' * 64)])
def test_fitting_receipt_cannot_override_saved_arithmetic(field, value, tmp_path):
    plan, expected, row, case, folder = handbuilt_fit(tmp_path)
    done = audit.read(folder / 'completed.json')
    done[field] = value
    put(folder / 'completed.json', done)
    with pytest.raises(ValueError):
        run_fit_fixture(plan, expected, row, case, folder)


def test_actual_initial_payload_cannot_be_replaced_by_fitted_weights(tmp_path):
    plan, expected, row, case, folder = handbuilt_fit(tmp_path)
    torch.save(case['final_weights'], folder / 'initial-weights.pt')
    done = audit.read(folder / 'completed.json')
    done['files']['initial-weights.pt'] = audit.sha(folder / 'initial-weights.pt')
    put(folder / 'completed.json', done)
    with pytest.raises(ValueError, match='original pair'):
        run_fit_fixture(plan, expected, row, case, folder)


def test_checkpoint_neural_architecture_binding_cannot_change_with_matching_parameter_shape(tmp_path):
    plan, expected, row, case, folder = handbuilt_fit(tmp_path)
    checkpoint = case['checkpoint']
    checkpoint['model_configuration']['replay'] = 'only visible tokens'
    checkpoint['integrity_sha256'] = audit.state_hash({k: v for k, v in checkpoint.items() if k != 'integrity_sha256'})
    torch.save(checkpoint, folder / 'checkpoint.pt')
    done = audit.read(folder / 'completed.json')
    done['files']['checkpoint.pt'] = audit.sha(folder / 'checkpoint.pt')
    done['checkpoint_integrity_sha256'] = checkpoint['integrity_sha256']
    put(folder / 'completed.json', done)
    with pytest.raises(ValueError, match='configuration'):
        run_fit_fixture(plan, expected, row, case, folder)


def test_history_accounting_includes_all_padded_heads_residual_cost_and_selected_step():
    value = settings()
    work = audit.history_work(value, 2, 2 + 2 * 256 * 12)
    operations = work['operations']
    assert operations['replayed_observation_update_samples'] == 24
    assert operations['replayed_transition_samples'] == 22
    assert operations['linear_layer_sample_calls'] == 4 * (22 + 2 + 2 * 256 * 12)
    assert operations['analytic_reward_sample_calls'] == 22 + 2 + 2 * 256 * 12
    assert audit.neural_counts(2, 0)['reward_head.2']['completed_sample_calls'] == 22
    assert audit.neural_counts(0, 2)['observation_update']['completed_sample_calls'] == 0


def test_old_wrapper_registry_is_not_rebound_and_no_learned_entrypoint_called_in_source():
    tree = ast.parse(Path(audit.__file__).read_text())
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    forbidden = {'restore_checkpoint', '_construct', 'make_trainer', 'train_next', 'fit_one', 'Adam', 'GRUCell', 'forward', 'assimilate', 'advance'}
    assert not any(isinstance(call, ast.Attribute) and call.attr in forbidden for call in calls)
    assignments = [target for node in ast.walk(tree) if isinstance(node, ast.Assign) for target in node.targets]
    assert not any(isinstance(target, ast.Attribute) and target.attr in {'seed', 'registry', 'REGISTRY', 'protocol'} for target in assignments)


def test_wrong_plan_bytes_failure_preserved_before_any_numerical_audit(tmp_path, monkeypatch):
    path, out = tmp_path / 'plan.json', tmp_path / 'audit'
    put(path, {'synthetic': True})
    monkeypatch.setattr(audit, 'audit_saved', lambda *a, **k: pytest.fail('Must not enter numerical audit'))
    with pytest.raises(ValueError, match='SHA256'):
        audit.audit_plan(path, '0' * 64, tmp_path / 'execution', out, engineering=True)
    failed = audit.read(out / 'failed.json')
    assert failed['status'] == 'failed' and failed['phase'] == 'authenticate-plan'
    assert not (out / 'receipt.json').exists()


def test_existing_audit_is_never_clobbered(tmp_path):
    out = tmp_path / 'audit'
    out.mkdir()
    (out / 'sentinel').write_text('retain')
    with pytest.raises(ValueError, match='Exclusive'):
        audit.audit_plan(tmp_path / 'absent', 'a' * 64, tmp_path / 'absent-execution', out, engineering=True)
    assert (out / 'sentinel').read_text() == 'retain'
    assert not (out / 'failed.json').exists()


def test_exact_completion_rejects_partial_extra_missing_or_over_cap_before_payload_reads(tmp_path):
    plan = {'settings': settings(), 'engineering': True, 'cap_seconds': 1}
    done = {'status': 'completed', 'version': 'reacher-two-observation-runner-v1', 'study': audit.VERSION,
        'plan_sha256': 'a' * 64, 'engineering': True, 'new_fits': 3, 'inherited_fits': 6, 'restored_models': 9,
        'prefit_restored_models': 6, 'control_rows': 42, 'diagnostic_roots': 0, 'astra_calls': 0,
        'new_optimizer_steps': audit.protocol.coverage(plan['settings'])['new_optimizer_updates'],
        'wall_seconds': 2., 'evaluation_started_elapsed_seconds': .5, 'cumulative_attempt_wall_seconds': 2., 'files': {}}
    put(tmp_path / 'completed.json', done)
    with pytest.raises(ValueError, match='cap'):
        audit.validate_members(plan, 'a' * 64, tmp_path)
    done['wall_seconds'] = 1.
    put(tmp_path / 'completed.json', done)
    with pytest.raises(ValueError, match='membership'):
        audit.validate_members(plan, 'a' * 64, tmp_path)
    done['control_rows'] = 41
    put(tmp_path / 'completed.json', done)
    with pytest.raises(ValueError, match='control_rows'):
        audit.validate_members(plan, 'a' * 64, tmp_path)


def controller_fixture():
    plan = audit.kernel_plan(settings(), object())
    fit = {'student_tensor_sha256': 'a' * 64}
    root = {key: np.zeros((1, *shape), audit.history_state_dtype(key)) for key, shape in audit.history_state_shapes(plan).items()}
    carried = copy.deepcopy(root)
    transitions = 256 * 12
    selected = {'model_advance_seconds': .01, 'geometry_seconds': .005}
    scoring = {'work': {'imagined_transitions': transitions}, 'selected_advance': selected, 'search_seconds': .2}
    size = sum(x.nbytes for x in root.values())
    value = {'version': 'reacher-two-observation-control-v1', 'step': 0, 'model': audit.history_identity(plan, fit),
        'root_sha256': audit.np_state_hash(root), 'carried_sha256': audit.np_state_hash(carried),
        'reconstruction_neural_kernels': audit.neural_counts(1, 0), 'selected_neural_kernels': audit.neural_counts(0, 1),
        'reconstruction_attempted_samples': 1, 'reconstruction_completed_samples': 1,
        'model_work': audit.history_work(plan, 1, transitions + 1), 'search': scoring['work'], 'selected_advance': selected,
        'reconstruction_analytic_reward': {'completed_samples_lower_bound': 11, 'completed_samples_upper_bound': 11, 'exact': True},
        'root_snapshot_tensor_bytes': size, 'carried_snapshot_tensor_bytes': size, 'committed_state_clone_tensor_bytes': size,
        'real_boundary_schema_attempted_samples': 1,
        'selected_history_work': {'advance_state_schema_attempted_samples': 1,
            'completed_history_clone_tensor_bytes': sum(root[k].nbytes for k in audit.history_audit.REAL_KEYS),
            'completed_pending_action_clone_tensor_bytes': 8},
        'limits': audit.history_scoring_configuration(plan)['timing_scope'], 'reconstruction_seconds': .1,
        'search_seconds': .21, 'selected_advance_seconds': .02, 'controller_seconds': .4}
    return plan, value, scoring, root, carried, fit


def test_controller_boundary_preserves_padded_work_and_different_inner_outer_search_time():
    plan, value, scoring, root, carried, fit = controller_fixture()
    assert value['search_seconds'] > scoring['search_seconds']
    audit.audit_controller_metadata(plan, value, scoring, root, carried, fit, 0)


@pytest.mark.parametrize('mutation', [
    lambda x: x['reconstruction_neural_kernels']['observation_update'].update(completed_sample_calls=1),
    lambda x: x['reconstruction_neural_kernels']['reward_head.2'].update(completed_sample_calls=0),
    lambda x: x['selected_neural_kernels']['transition'].update(attempted_sample_calls=0),
    lambda x: x['reconstruction_analytic_reward'].update(completed_samples_lower_bound=0),
    lambda x: x['selected_history_work'].update(completed_pending_action_clone_tensor_bytes=0),
    lambda x: x.update(controller_seconds=.01),
    lambda x: x.update(root_snapshot_tensor_bytes=0),
    lambda x: x.update(step=1),
])
def test_controller_work_or_phase_cannot_be_reduced(mutation):
    plan, value, scoring, root, carried, fit = controller_fixture()
    mutation(value)
    with pytest.raises(ValueError):
        audit.audit_controller_metadata(plan, value, scoring, root, carried, fit, 0)


def mocked_enclosing_fixture(tmp_path, monkeypatch):
    from types import SimpleNamespace

    value = settings()
    plan = {'settings': value, 'sources': {}, 'runtime': {}, 'engineering': True, 'cap_seconds': 10, 'audit_cap_seconds': 10}
    context = SimpleNamespace(settings=value, streams=object(), plan_sha256='a' * 64, engineering=True, execution_authorized=True)
    execution = tmp_path / 'execution'
    execution.mkdir()
    put(execution / 'completed.json', {'synthetic': True})
    put(execution / 'random-streams.json', {'synthetic': True})
    records = [{'policy': {'commands': np.zeros((50, 2), np.float32), 'packets': np.zeros((51, 8), np.float32)},
        'audit': {'integration_state': np.zeros((51, 1)), 'actuator_noise': np.zeros((50, 2)), 'rewards': np.full(50, -.02)}}]
    reward_rows = [{'panel': row['panel'], 'label': row['label'], 'case_ids': ['control/0'],
        'native_rewards': np.stack([records[0]['audit']['rewards']])} for row in audit.protocol.execution_order(value)]
    arithmetic = audit.results.evaluate_rows(value, reward_rows)
    assert arithmetic['continuation_gate']['passed'] is False
    put(execution / 'results.json', arithmetic)
    calls = {'auth': 0, 'history': 0, 'inherited': 0, 'reference': 0, 'native': 0}

    def auth(*args, **kwargs):
        calls['auth'] += 1
        return context

    monkeypatch.setattr(experiment, 'validate_plan', auth)
    monkeypatch.setattr(experiment, 'runtime', dict)
    monkeypatch.setattr(audit, 'validate_members', lambda *a: {'files': {}, 'new_optimizer_steps': 12})
    monkeypatch.setattr(audit, 'audit_copied_lineage', lambda *a: None)
    fits = {row['name']: {} for row in audit.protocol.fit_manifest(value)}
    monkeypatch.setattr(audit, 'audit_fits_and_snapshots', lambda *a: (fits, {}, {}))
    monkeypatch.setattr(audit, 'audit_innovations', lambda *a: None)
    monkeypatch.setattr(audit.base, 'load_records', lambda *a: records)

    def cohort(*a):
        calls['native'] += 1
        return {'transitions': 50, 'max_abs_error': 0.}

    monkeypatch.setattr(audit, 'audit_cohort', cohort)

    def row_result(kind, row):
        calls[kind] += 1
        result = copy.deepcopy(arithmetic['controls'][row['panel']][row['label']])
        if kind == 'reference' and row['reference'] in audit.PHYSICS_REFERENCES:
            result['physics_work'] = {'candidate_native_transitions_replayed': 256 * 534,
                'selected_native_transitions_replayed': 50, 'max_abs_error': 0.}
        if kind == 'reference' and row['reference'] in ('particle', 'public_kinematic'):
            result['public_observer'] = {'observer_native_transitions_replayed': 49, 'max_abs_error': 0.}
        return result

    monkeypatch.setattr(audit, 'audit_history_control', lambda p, ex, f, row, *a: row_result('history', row))
    monkeypatch.setattr(audit.memory, 'audit_learned_control', lambda p, f, row, *a: row_result('inherited', row))
    monkeypatch.setattr(audit, 'audit_reference_control', lambda p, f, row, *a: row_result('reference', row))
    monkeypatch.setattr(audit, 'audit_boundaries', lambda *a: {})
    monkeypatch.setattr(audit, 'audit_costs', lambda *a: {})
    return plan, execution, calls, arithmetic


def test_complete_outer_dispatch_preserves_failed_gate_and_reauthenticates(tmp_path, monkeypatch):
    plan, execution, calls, arithmetic = mocked_enclosing_fixture(tmp_path, monkeypatch)
    value = audit.audit_saved(plan, 'a' * 64, execution, tmp_path / 'audit', engineering=True, plan_bytes=b'externally mocked')
    assert value['status'] == 'completed' and value['continuation_gate'] == arithmetic['continuation_gate']
    assert value['continuation_gate']['passed'] is False
    assert calls == {'auth': 2, 'history': 9, 'inherited': 18, 'reference': 15, 'native': 42}
    assert value['native_control_transitions_checked'] == 2100
    assert value['native_nominal_candidate_transitions_checked'] == 1230336
    assert value['native_nominal_selected_transitions_checked'] == 450
    assert (tmp_path / 'audit/receipt.json').is_file()


def test_final_authentication_failure_keeps_failed_receipt_not_success(tmp_path, monkeypatch):
    plan, execution, _, _ = mocked_enclosing_fixture(tmp_path, monkeypatch)
    real = experiment.validate_plan
    calls = []

    def auth(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError('source changed')
        return real(*a, **k)

    monkeypatch.setattr(experiment, 'validate_plan', auth)
    with pytest.raises(ValueError, match='source changed'):
        audit.audit_saved(plan, 'a' * 64, execution, tmp_path / 'audit', engineering=True, plan_bytes=b'mocked')
    assert (tmp_path / 'audit/failed.json').is_file()
    assert not (tmp_path / 'audit/receipt.json').exists()


def test_cap_after_receipt_write_demotes_completion_and_preserves_failure(tmp_path, monkeypatch):
    plan, execution, _, _ = mocked_enclosing_fixture(tmp_path, monkeypatch)
    out = tmp_path / 'audit'
    original = audit.check_budget

    def cap():
        if (out / 'receipt.json').exists():
            raise TimeoutError('after receipt cap')
        original()

    monkeypatch.setattr(audit, 'check_budget', cap)
    with pytest.raises(TimeoutError, match='after receipt'):
        audit.audit_saved(plan, 'a' * 64, execution, out, engineering=True, plan_bytes=b'mocked')
    assert not (out / 'receipt.json').exists()
    assert (out / 'invalid-receipt.json').is_file() and (out / 'failed.json').is_file()


def test_failure_preservation_error_never_replaces_original(tmp_path, monkeypatch):
    plan, execution, _, _ = mocked_enclosing_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, 'audit_copied_lineage', lambda *a: (_ for _ in ()).throw(ValueError('original lineage error')))
    monkeypatch.setattr(audit, 'write', lambda *a: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(ValueError, match='original lineage error') as error:
        audit.audit_saved(plan, 'a' * 64, execution, tmp_path / 'audit', engineering=True, plan_bytes=b'mocked')
    assert any('disk full' in note for note in error.value.__notes__)


def synthetic_scoring_trace(tmp_path, monkeypatch):
    from types import SimpleNamespace

    plan, _, _, root, carried, fit = controller_fixture()
    n, k, h = 1, 256, 1
    commands = np.zeros((n, k, h, 2), np.float32)
    angle = np.broadcast_to(np.array([1., 1., 0., 0.], np.float32), (n, k, h, 4)).copy()
    rewards = np.full((n, k, h), -1., np.float32)
    values = {'commands': commands, 'root_target': root['packet'][:, 4:6], 'predicted_angles': angle,
        'learned_rewards': rewards.copy(), 'selected_rewards': rewards.copy(), 'selected_ids': np.zeros(n, np.int64),
        'selected_actions': np.zeros((n, 2), np.float32), 'selected_angles': angle[:, 0, 0],
        'selected_learned_reward': rewards[:, 0, 0], 'selected_reward': rewards[:, 0, 0]}
    for key in audit.GEOMETRY_FIELDS:
        tail = (2,) if key in ('joint_angles', 'pair_norms', 'fingertip') else ()
        values['geometry_' + key] = np.zeros((n, k, h, *tail), np.float32)
        values['selected_geometry_' + key] = np.zeros((n, *tail), np.float32)
    selected = {'selected_angles', 'selected_learned_reward', 'selected_reward'} | {'selected_geometry_' + key for key in audit.GEOMETRY_FIELDS}
    size = sum(x.nbytes for x in root.values())
    real_size = sum(root[key].nbytes for key in audit.history_audit.REAL_KEYS)
    work = {'candidate_evaluations': k, 'imagined_transitions': k, 'root_tensor_bytes': size,
        'max_single_candidate_state_tensor_bytes': size * 64, 'model_work': audit.history_work(plan, 0, k),
        'neural_kernels': audit.neural_counts(0, k), 'history_work': {'advance_state_schema_attempted_samples': k,
            'completed_advance_history_clone_tensor_bytes': k * real_size, 'completed_pending_action_clone_tensor_bytes': k * 8,
            'candidate_root_repeat_scheduled_tensor_bytes': size * 256,
            'limits': 'Payload accounting only. Failed-tail copies, temporaries, tensor objects and validators are not fully enumerated; all remain in measured wall time.'},
        'geometry_samples': k, 'geometry_seconds': .004, 'model_advance_seconds': .008,
        'diagnostic_array_bytes': sum(v.nbytes for key, v in values.items() if key not in selected),
        'tensor_bytes_are_not_peak_process_memory': True}
    meta = {'version': 'reacher-two-observation-control-v1', 'model': audit.history_identity(plan, fit), 'step': 0,
        'score_mode': 'geometry', 'configuration': audit.history_scoring_configuration(plan),
        'root_sha256': audit.np_state_hash(root), 'callbacks': [{'candidate_start': 64 * i, 'candidate_stop': 64 * (i + 1),
            'bank_wall_seconds': .01} for i in range(4)], 'work': work, 'search_seconds': .2, 'selected_advance': {}}
    stem = tmp_path / '000'
    np.savez(stem.with_suffix('.npz'), **values)
    put(stem.with_suffix('.json'), meta)
    trace = {'result': SimpleNamespace(sequences=commands, selected_ids=values['selected_ids'],
        selected_actions=values['selected_actions'], scores=np.full((n, k), -1., np.float32)),
        'raw_rewards': rewards, 'search_seconds': .21}
    # Only the nested timer binding is under test. Independent geometry/native
    # correctness is deliberately stubbed and is not certified by this fixture.
    monkeypatch.setattr(audit.previous, 'audit_bank_arrays', lambda *a: {'candidate_evaluations': k, 'imagined_transitions': k, 'geometry_samples': k})
    monkeypatch.setattr(audit.previous, 'audit_bank_metadata', lambda *a: {'geometry_seconds': .001, 'model_advance_seconds': .002})
    monkeypatch.setattr(audit.previous, 'audit_selected', lambda *a: None)
    return plan, stem, root, carried, fit, trace, values


def test_inner_scorer_time_can_differ_from_outer_saved_trace_but_cannot_exceed_it(tmp_path, monkeypatch):
    plan, stem, root, carried, fit, trace, values = synthetic_scoring_trace(tmp_path, monkeypatch)
    _, meta, _ = audit.audit_history_scoring(plan, stem, root, carried, fit, trace, 0,
        values['selected_angles'], values['selected_learned_reward'])
    assert meta['search_seconds'] == .2 and trace['search_seconds'] == .21
    meta['search_seconds'] = .22
    put(stem.with_suffix('.json'), meta)
    with pytest.raises(ValueError, match='temporal identity'):
        audit.audit_history_scoring(plan, stem, root, carried, fit, trace, 0,
            values['selected_angles'], values['selected_learned_reward'])
