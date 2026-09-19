"""Synthetic arithmetic and seed410 engineering native fixtures, no fitted models."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from unittest.mock import patch

import audit_reacher_geometry_memory_study as audit
import numpy as np
import pytest
import reacher_geometry_memory_study as study
import torch

from openjev.research import reacher_geometry_memory_protocol as protocol
from openjev.research import reacher_search_protocol as artifacts
from openjev.research.reacher_adaptive_search import SearchInputs
from openjev.research.reacher_geometry_physics import PhysicsGeometryCEM
from openjev.research.robotics_reacher import make_env


def small_plan():
    plan = protocol.settings(engineering=True)
    plan.update(engineering=True, control_episodes=1, hidden_size=4, mlp_width=7, bootstrap_samples=8)
    return plan


def costs(plan):
    output = {}
    for panel in ('full', 'ordinary', 'shift'):
        output[panel] = {}
        for arm in ('residual_gru', 'encoded_current_gru', 'cached_gru', 'cached_mlp'):
            for pair in ('pair0', 'pair1', 'pair2'):
                value = 5. if arm == 'residual_gru' else 6.
                output[panel][f'{arm}-{pair}'] = {'mean_cost': value, 'episode_costs': [value] * plan['control_episodes']}
        for reference, value in (('known_state', 5.), ('particle', 5.), ('public_kinematic', 5.), ('zero', 10.), ('uniform', 30.)):
            output[panel][reference] = {'mean_cost': value, 'episode_costs': [value] * plan['control_episodes']}
    return output


def set_cost(data, panel, label, value):
    data[panel][label]['mean_cost'] = value
    data[panel][label]['episode_costs'] = [value]


def test_independent_gate_all25_and_secondary_cannot_rescue_any_failed_primary():
    plan = small_plan()
    data = costs(plan)
    gate = audit.qualification(plan, data)
    assert gate['passed'] and len(gate['checks']) == 25
    assert all(row['passed'] for row in gate['checks'])
    set_cost(data, 'shift', 'cached_gru-pair1', 4.9)
    set_cost(data, 'shift', 'cached_mlp-pair1', 100.)
    gate = audit.qualification(plan, data)
    assert not gate['passed']
    failed = [row['name'] for row in gate['checks'] if not row['passed']]
    assert failed == ['gap_pair/shift/cached_gru/pair1']
    assert gate['secondary_cannot_rescue_primary'] is True


def test_only_ordinary_known_state_reference_is_primary_not_particle_or_shift():
    plan, data = small_plan(), costs(small_plan())
    for panel in ('full', 'ordinary', 'shift'):
        set_cost(data, panel, 'particle', 100.)
        set_cost(data, panel, 'public_kinematic', 100.)
    set_cost(data, 'shift', 'known_state', 100.)
    assert audit.qualification(plan, data)['passed']
    set_cost(data, 'ordinary', 'known_state', 9.01)
    gate = audit.qualification(plan, data)
    assert [row['name'] for row in gate['checks'] if not row['passed']] == ['competence/ordinary/known_state']


def test_mean_margin_and_full_regression_are_required_even_if_fit_nonworse():
    plan, data = small_plan(), costs(small_plan())
    for pair in ('pair0', 'pair1', 'pair2'):
        set_cost(data, 'ordinary', f'encoded_current_gru-{pair}', 5.1)
        set_cost(data, 'full', f'cached_gru-{pair}', 4.89)
    gate = audit.qualification(plan, data)
    assert {row['name'] for row in gate['checks'] if not row['passed']} == {
        'gap_mean/ordinary/encoded_current_gru', 'full_mean/cached_gru'}


@pytest.mark.parametrize('corruption', ['omit', 'extra', 'wrong_mean', 'nonfinite', 'negative', 'wrong_count'])
def test_gate_rejects_incomplete_or_invalid_cost_evidence(corruption):
    plan, data = small_plan(), costs(small_plan())
    row = data['ordinary']['residual_gru-pair0']
    if corruption == 'omit':
        del data['ordinary']['cached_gru-pair2']
    elif corruption == 'extra':
        data['ordinary']['selected_best'] = row
    elif corruption == 'wrong_mean':
        row['mean_cost'] += 1
    elif corruption == 'nonfinite':
        row['episode_costs'][0] = np.nan
    elif corruption == 'negative':
        row.update(mean_cost=-1., episode_costs=[-1.])
    else:
        row['episode_costs'].append(5.)
    with pytest.raises(ValueError):
        audit.qualification(plan, data)


def test_expected_members_complete12_models_51rows_with_every_native_candidate_journal():
    plan = small_plan()
    members = audit.expected_members(plan)
    assert audit.inherited_members() == set(protocol.inherited_members(plan))
    assert sum(name.endswith('-before.pt') for name in members) == 12
    assert sum(name.endswith('-after.pt') for name in members) == 12
    assert sum(name.endswith('/episodes.npz') for name in members) == 51
    assert sum('/physics/' in name and name.endswith('.npz') for name in members) == 9 * 50
    assert sum('/scoring/' in name and name.endswith('.npz') for name in members) == 36 * 50
    assert sum(name.endswith('/physics-final.json') for name in members) == 9
    assert sum(name.endswith('/observer-final.json') for name in members) == 6
    assert not any('diagnostic/' in name or 'prediction' in name and 'executed_predictions' not in name for name in members)
    assert {f'inherited/{kind}-{suffix}.json' for kind in ('cache', 'geometry') for suffix in ('plan', 'audit', 'completed', 'summary')} <= members


@pytest.fixture
def nominal(tmp_path):
    plan = small_plan()
    env = make_env()
    env.reset(seed=410)
    model = env.unwrapped.model
    root = env.unwrapped.data
    qpos, qvel = root.qpos[None].copy(), root.qvel[None].copy()
    qvel[:, 2:] = 0.
    target = qpos[:, 2:].astype(np.float32)
    rng = np.random.default_rng(410)
    values = [rng.normal(size=(1, count, 4, 2)) for count in (64, 192, 64, 64, 63)]
    inputs = SearchInputs(values[0], values[1], tuple(values[2:]))
    physics = PhysicsGeometryCEM(model)
    with (patch.object(torch.nn.Linear, 'forward', side_effect=AssertionError('No neural inference')),
          patch.object(torch.nn.GRUCell, 'forward', side_effect=AssertionError('No neural inference')),
          patch.object(torch.optim.Adam, '__init__', side_effect=AssertionError('No optimizer'))):
        result, journal = physics.plan(qpos, qvel, target, inputs, step=49)
    stem = tmp_path / 'physics'
    study.save_physics(stem, journal)
    rewards = np.concatenate([bank['arrays']['geometry_reward'] for bank in journal['banks']], 1)
    artifacts.save_trace(tmp_path / 'decision', result, rewards, [64] * 4, journal['wall_seconds'])
    trace = audit.search_audit.audit_trace({**plan, 'planners': ['cem256']}, inputs, tmp_path / 'decision', step=49)
    yield {'plan': plan, 'model': model, 'roots': {'qpos': qpos, 'qvel': qvel, 'public_target': target},
           'journal': journal, 'stem': stem, 'trace': trace}
    env.close()


def test_actual_cem256_nominal_native_replay_including_separate_selected_root(nominal):
    case = nominal
    with (patch.object(PhysicsGeometryCEM, 'plan', side_effect=AssertionError('Audit called controller')),
          patch.object(PhysicsGeometryCEM, 'score_bank', side_effect=AssertionError('Audit called scorer')),
          patch.object(torch.nn.Linear, 'forward', side_effect=AssertionError('Audit called model')),
          patch.object(torch.nn.GRUCell, 'forward', side_effect=AssertionError('Audit called model')),
          patch.object(torch.optim.Adam, '__init__', side_effect=AssertionError('Audit optimizer'))):
        got = audit.audit_physics_trace(case['plan'], case['model'], case['stem'], case['trace'],
            np.concatenate((case['roots']['qpos'], case['roots']['qvel']), 1), case['roots']['public_target'], step=49)
    assert got['candidate_transitions'] == 256 and got['selected_transitions'] == 1
    assert got['max_abs_error'] == 0.
    assert got['counts']['native_substeps_completed'] == 514
    assert got['counts']['geometry_samples_completed'] == 257
    assert got['counts']['geometry_calls_completed'] == 5
    assert got['counts']['reset_calls_completed'] == 257


@pytest.mark.parametrize('field', ['native', 'angles', 'distance', 'double_action_cost', 'score', 'partial',
    'substep_mask', 'counter', 'root_hash', 'candidate_start', 'time'])
def test_nominal_bank_corruption_detected_from_saved_evidence(nominal, field):
    case = nominal
    bank = copy.deepcopy(case['journal']['banks'][0])
    arrays, metadata = bank['arrays'], bank['metadata']
    commands = arrays['commands'].copy()
    if field == 'native':
        arrays['qvel'][0, 0, 1, 0] += .01
    elif field == 'angles':
        arrays['predicted_angles'][0, 0, 0, 0] += .01
    elif field == 'distance':
        arrays['geometry_distance'][0, 0, 0] += .01
    elif field == 'double_action_cost':
        arrays['geometry_reward'] -= arrays['geometry_action_cost']
    elif field == 'score':
        arrays['scores'][0, 0] += .01
    elif field == 'partial':
        arrays['native_completed'][0, 0, 0] = False
    elif field == 'substep_mask':
        arrays['native_substeps_completed'][0, 0, 0] = 1
    elif field == 'counter':
        metadata['native_transitions_completed'] -= 1
    elif field == 'root_hash':
        metadata['root_sha256']['qpos'] = '0' * 64
    elif field == 'candidate_start':
        metadata['candidate_start'] = 64
    else:
        metadata['wall_seconds'] = metadata['native_seconds'] / 2
    with pytest.raises(ValueError):
        audit.audit_physics_bank(case['plan'], case['model'], case['roots'], arrays, metadata, commands,
                                 step=49, candidate_start=0)


def test_selected_native_root_reuse_not_candidate_terminal(nominal):
    case = nominal
    bank = copy.deepcopy(case['journal']['selected'])
    bank['arrays']['qpos'][0, 0, 0, 0] += .1
    with pytest.raises(ValueError, match='restored from root'):
        audit.audit_physics_bank(case['plan'], case['model'], case['roots'], bank['arrays'], bank['metadata'],
                                 bank['arrays']['commands'], step=49)


def test_trace_rejects_missing_selected_work_and_unbound_array(nominal):
    case = nominal
    metadata = json.loads(case['stem'].with_suffix('.json').read_text())
    metadata['selected'] = None
    case['stem'].with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='Complete paid nominal'):
        audit.audit_physics_trace(case['plan'], case['model'], case['stem'], case['trace'],
            np.concatenate((case['roots']['qpos'], case['roots']['qvel']), 1), case['roots']['public_target'], step=49)


def test_bad_external_plan_preserves_exclusive_failed_audit_receipt(tmp_path):
    plan = tmp_path / 'plan.json'
    plan.write_text('{}')
    out = tmp_path / 'audit'
    with pytest.raises(ValueError):
        audit.audit_plan(plan, '0' * 64, tmp_path / 'execution', out)
    receipt = json.loads((out / 'failed.json').read_text())
    assert receipt['saved_output_only'] and receipt['phase'] == 'authenticate-plan'
    before = (out / 'failed.json').read_bytes()
    with pytest.raises(ValueError, match='Exclusive'):
        audit.audit_plan(plan, '0' * 64, tmp_path / 'execution', out)
    assert (out / 'failed.json').read_bytes() == before


def test_audit_failure_preserves_receipt_and_restores_deadline_context(tmp_path):
    before = audit._DEADLINE.get()
    out = tmp_path / 'audit'
    plan = small_plan()
    plan.update(cap_seconds=10, audit_cap_seconds=10)
    with patch.object(audit, 'validate_runtime_sources', side_effect=ValueError('Synthetic runtime failure')), pytest.raises(ValueError):
        audit.audit_saved(plan, '0' * 64, tmp_path / 'missing', out, engineering=True)
    assert audit._DEADLINE.get() == before
    receipt = json.loads((out / 'failed.json').read_text())
    assert receipt['saved_output_only'] and 'Synthetic runtime failure' in receipt['error']
    assert not (out / 'receipt.json').exists()


def test_source_does_not_import_new_runner_or_neural_scorers():
    parsed = ast.parse(Path(audit.__file__).read_text())
    forbidden = {'reacher_geometry_memory_study', 'reacher_geometry_memory_control',
                 'reacher_geometry_physics', 'reacher_geometry_reward', 'reacher_cache_training'}
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            assert not forbidden & {alias.name.rsplit('.', 1)[-1] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or '').rsplit('.', 1)[-1] not in forbidden
            assert not forbidden & {alias.name for alias in node.names}

def cost_fixture(tmp_path):
    plan = small_plan()
    plan.update(cache_source={'prior_costs': {'cumulative_attempt_wall_seconds': 1.}},
                geometry_source={'prior_costs': {'cumulative_attempt_wall_seconds': 2.}})
    controls = {panel: {} for panel in ('full', 'ordinary', 'shift')}
    for row in protocol.execution_order(plan):
        value = {'setup_seconds': .001, 'decision_wall_seconds': .05,
                 'native_step_seconds': .01, 'row_wall_seconds': .07}
        if 'fit' in row:
            value['scoring_work'] = {'candidate_evaluations': 12800, 'imagined_transitions': 136704,
                'geometry_samples': 136704, 'geometry_seconds': .005, 'model_advance_seconds': .03,
                'selected_geometry_samples': 50, 'selected_geometry_seconds': .001,
                'selected_model_advance_seconds': .001}
        else:
            planned = row['reference'] in ('known_state', 'particle', 'public_kinematic')
            counts = {key: 0 for key in audit.PHYSICS_COUNTS}
            if planned:
                for key in counts:
                    if key.startswith(('candidate_sequences_', 'reset_calls_', 'forward_calls_')):
                        counts[key] = 12850
                    elif key.startswith('native_substeps_'):
                        counts[key] = 273508
                    elif key.startswith('geometry_calls_'):
                        counts[key] = 250
                    else:
                        counts[key] = 136754
            value['physics_work'] = {'candidate_native_transitions_replayed': 136704 if planned else 0,
                'selected_native_transitions_replayed': 50 if planned else 0,
                'native_seconds': .03 if planned else 0., 'geometry_seconds': .005 if planned else 0.,
                'operation_wall_seconds': .04 if planned else 0., 'counts': counts}
        controls[row['panel']][row['label']] = value
    saved = {'inheritance_copy_wall_seconds': .01, 'restore_wall_seconds': .01,
        'new_fits': 0, 'new_optimizer_steps': 0, 'new_prediction_episodes': 0, 'diagnostic_roots': 0,
        'innovation_generation_and_storage_seconds': .01,
        'control_row_wall_seconds': 51 * .07, 'control_setup_seconds': 51 * .001,
        'control_decision_seconds': 51 * .05, 'control_native_step_seconds': 51 * .01,
        'control_native_transitions': 2550, 'astra_calls': 0,
        'cache_parent_costs': plan['cache_source']['prior_costs'],
        'geometry_context_costs': plan['geometry_source']['prior_costs'],
        'compute_matched': False, 'accounting': 'Synthetic independent arithmetic fixture, not measured performance.'}
    (tmp_path / 'costs.json').write_text(json.dumps(saved))
    return plan, controls, saved, {'wall_seconds': 4., 'cumulative_attempt_wall_seconds': 6.}


def test_complete_cost_scope_counts_selected_work_and_prior_lineage_once(tmp_path):
    plan, controls, _, done = cost_fixture(tmp_path)
    got = audit.audit_costs(plan, tmp_path, done, {'wall_seconds': .01}, controls)
    assert got['physics_control_scoring']['candidate_native_transitions_replayed'] == 1230336
    assert got['physics_control_scoring']['selected_native_transitions_replayed'] == 450
    assert got['learned_control_scoring']['selected_geometry_samples'] == 1800
    assert got['cumulative_attempt_wall_seconds'] == 6.  # Cache ancestry already inside geometry prior2.


@pytest.mark.parametrize('kind', ['lost_selected_native', 'lost_selected_geometry', 'underpaid_substeps',
    'double_counted_prior', 'missing_row', 'omitted_copy_time', 'exceeds_whole_time'])
def test_cost_scope_drift_rejected(tmp_path, kind):
    plan, controls, saved, done = cost_fixture(tmp_path)
    if kind == 'lost_selected_native':
        controls['ordinary']['particle']['physics_work']['selected_native_transitions_replayed'] = 0
    elif kind == 'lost_selected_geometry':
        controls['ordinary']['cached_gru-pair1']['scoring_work']['selected_geometry_samples'] = 0
    elif kind == 'underpaid_substeps':
        controls['shift']['known_state']['physics_work']['counts']['native_substeps_completed'] -= 1
    elif kind == 'double_counted_prior':
        done['cumulative_attempt_wall_seconds'] += 1
    elif kind == 'missing_row':
        del controls['shift']['cached_mlp-pair2']
    elif kind == 'omitted_copy_time':
        del saved['inheritance_copy_wall_seconds']
    else:
        done['wall_seconds'] = 1.
    (tmp_path / 'costs.json').write_text(json.dumps(saved))
    with pytest.raises((ValueError, KeyError)):
        audit.audit_costs(plan, tmp_path, done, {'wall_seconds': .01}, controls)

def test_all_twelve_engineering_saved_restorations_and_last_snapshot_corruption(tmp_path):
    """No constructor: before/after evidence is copied from tiny published fits."""
    import shutil

    import reacher_geometry_memory_fixture as fixture

    if not (study.ROOT / fixture.CACHE_PARENT / 'execution' / 'completed.json').exists():
        pytest.skip('Optional retained completed tiny engineering artifacts are absent')
    parent, source, _, _ = fixture.parents()
    plan = small_plan()
    plan.update(hidden_size=parent['hidden_size'], mlp_width=parent['mlp_width'], runtime=parent['runtime'])
    execution = tmp_path / 'saved-restoration'
    for name in source['members']:
        target = execution / 'inherited' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(study.ROOT / source['execution_path'] / name, target)
    models, after = {}, {}
    updates = parent['epochs'] * (parent['train_episodes'] // parent['batch_size'])
    for row in protocol.fit_manifest(plan):
        folder = execution / 'inherited' / 'fits' / row['name']
        done = audit.read(folder / 'completed.json')
        for phase in ('before', 'after'):
            target = execution / 'model-states' / f"{row['name']}-{phase}.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(folder / 'weights.pt', target)
            if phase == 'after':
                after[target.relative_to(execution).as_posix()] = audit.sha(target)
        before = execution / 'model-states' / f"{row['name']}-before.pt"
        models[row['name']] = {'kind': row['arm'], 'model_class': protocol.MODEL_CLASSES[row['arm']],
            'checkpoint_sha256': audit.sha(folder / 'checkpoint.pt'), 'weights_sha256': audit.sha(folder / 'weights.pt'),
            'student_tensor_sha256': done['student_tensor_sha256'], 'successful_updates': updates,
            'configuration': audit.inherited.model_configuration(parent, row['arm']),
            'snapshot_path': before.relative_to(execution).as_posix(), 'snapshot_sha256': audit.sha(before)}
    audit.write(execution / 'all-models-restored.json', {'models': models, 'optimizer_constructed': False,
        'new_updates': 0, 'constructor_seed': 0, 'constructor_rng_isolated_and_weights_overwritten': True,
        'wall_seconds': .01, 'unix_time': 1.})
    final = {'student_tensor_sha256': {name: row['student_tensor_sha256'] for name, row in models.items()},
             'files': after, 'new_updates': 0}
    audit.write(execution / 'final-models.json', final)
    with (patch.object(study.training, '_construct', side_effect=AssertionError('No model construction')),
          patch.object(torch.nn.Linear, 'forward', side_effect=AssertionError('No inference')),
          patch.object(torch.nn.GRUCell, 'forward', side_effect=AssertionError('No inference')),
          patch.object(torch.optim.Adam, '__init__', side_effect=AssertionError('No optimizer'))):
        fits, _ = audit.audit_restoration(plan, parent, execution)
        assert len(fits) == 12 and all(row['observed_before_after_equal'] for row in fits.values())
        target = execution / 'model-states' / 'cached_mlp-pair2-after.pt'
        weights = torch.load(target, weights_only=True, map_location='cpu')
        weights[next(iter(weights))].reshape(-1)[0] += 1
        torch.save(weights, target)
        final['files'][target.relative_to(execution).as_posix()] = audit.sha(target)
        (execution / 'final-models.json').write_text(json.dumps(final))
        with pytest.raises(ValueError, match='tensors unchanged'):
            audit.audit_restoration(plan, parent, execution)
