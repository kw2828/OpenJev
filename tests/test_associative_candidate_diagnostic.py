import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip('minigrid', reason='Install minigrid==3.0.0 for the candidate diagnostic')

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location('_test_' + name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = load_script('associative_candidate_diagnostic')


def test_candidate_objective_ignores_unused_action_logits_and_gradients():
    logits = torch.tensor([[1., -1., 9., 8., 7., 6., 5.], [-2., 2., -9., 0., 7., 3., 2.]], requires_grad=True)
    labels = torch.tensor([0, 1])
    loss = study.objective(logits, labels)
    loss.backward()
    assert torch.equal(logits.grad[:, 2:], torch.zeros_like(logits.grad[:, 2:]))
    changed = logits.detach().clone()
    changed[:, 2:] = 10000
    torch.testing.assert_close(study.objective(changed, labels), loss.detach(), rtol=0, atol=0)
    with pytest.raises(ValueError, match='turn IDs'):
        study.objective(logits, torch.tensor([0, 2]))


def test_teacher_protocol_is_unchanged_and_fixed_training_settings_are_shared():
    assert study.teacher.PROTOCOL['version'] == 'associative-learnability-v1'
    assert study.teacher.PROTOCOL['objective'].startswith('Final-observation seven-action')
    assert 'continuation' not in study.teacher.PROTOCOL
    for key in ('modes', 'seeds', 'train_size', 'evaluation_sizes', 'updates', 'batch_size', 'optimizer',
                'learning_rate', 'betas', 'epsilon', 'weight_decay', 'gradient_clipping', 'fit_order_seed'):
        assert study.PROTOCOL[key] == study.teacher.PROTOCOL[key]


def test_gru_candidate_objective_reproduces_factorial_ce2_one_optimizer_step_exactly():
    factorial = load_script('memory_optimization_diagnostic')
    data, _ = study.teacher.training_packet()
    left = study.teacher.new_model('gru', 7)
    right = copy.deepcopy(left)
    labels = torch.from_numpy(data['labels'])
    left_logits = study.teacher.final_logits(left, data)
    right_logits = factorial.outputs(right, data, 'recursive')[0][-1]
    torch.testing.assert_close(left_logits, right_logits, rtol=0, atol=0)
    left_loss = study.objective(left_logits, labels)
    right_loss = factorial.objective(right_logits, labels, 'ce2')
    torch.testing.assert_close(left_loss, right_loss, rtol=0, atol=0)
    for model, loss in ((left, left_loss), (right, right_loss)):
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    for name, tensor in left.state_dict().items():
        torch.testing.assert_close(tensor, right.state_dict()[name], rtol=0, atol=0)


class FixedSevenActionPolicy(torch.nn.Module):
    def initial_state(self, count):
        return torch.zeros(count, 1)

    def sequence(self, observations, previous_actions, state, resets, **kwargs):
        logits = observations.new_tensor([4., 0., 9., 0., 0., 0., 0.])
        return logits.expand(*observations.shape[:2], 7), None, None


def test_evaluation_separates_candidate_decisions_from_seven_action_behavior():
    data, metadata = study.teacher.training_packet()
    result = study.evaluate(FixedSevenActionPolicy(), data, metadata, 'intact')
    assert result['accuracy7'] == 0 and result['accuracy2'] == .5
    for row in result['examples']:
        assert row['predicted_action7'] == 2 and row['predicted_candidate2'] == 0
        p7, p2 = np.array(row['probabilities7']), np.array(row['probabilities2'])
        np.testing.assert_allclose(p2, p7[:2] / p7[:2].sum(), rtol=1e-6)
        assert p7.sum() == pytest.approx(1., abs=1e-6)
        assert p2.sum() == pytest.approx(1., abs=1e-6)


def passing_evaluations():
    records = []
    for mode in study.PROTOCOL['modes']:
        for seed in study.PROTOCOL['seeds']:
            results = []
            for size in study.PROTOCOL['evaluation_sizes']:
                for condition in study.PROTOCOL['conditions']:
                    success = 1. if mode == 'fast_selective' and condition == 'intact' else .75
                    results.append({'size': size, 'condition': condition,
                                    'accuracy7': 0., 'accuracy2': success,
                                    'cross_entropy7': 2., 'conditional_binary_cross_entropy': .01,
                                    'target_probability7': .1, 'target_probability2': .99})
            records.append({'mode': mode, 'seed': seed, 'results': results})
    return records


def test_continuation_requires_each_fit_each_length_and_strict_confidence_gate():
    assert set(study.PROTOCOL['continuation']['controls']) == {'gru', 'feedforward', 'fast_global'}
    records = passing_evaluations()
    _, checks = study.continuation(records)
    assert all(checks.values())
    candidate = next(r for r in records if r['mode'] == 'fast_selective')
    result = next(r for r in candidate['results'] if r['size'] == 17 and r['condition'] == 'intact')
    result['accuracy2'] = .75
    assert not study.continuation(records)[1]['perfect_all_fits_size_17']
    result['accuracy2'] = 1.
    for record in records:
        if record['mode'] == 'fast_selective':
            for row in record['results']:
                if row['size'] == 17 and row['condition'] == 'intact':
                    row['conditional_binary_cross_entropy'] = .05
    assert not study.continuation(records)[1]['conditional_ce_size_17']


def test_one_bad_fit_cannot_hide_behind_a_passing_mean_cross_entropy():
    records = passing_evaluations()
    candidate = next(record for record in records if record['mode'] == 'fast_selective')
    result = next(row for row in candidate['results'] if row['size'] == 23 and row['condition'] == 'intact')
    result['conditional_binary_cross_entropy'] = .1
    means, checks = study.continuation(records)
    assert means['fast_selective']['23']['intact']['conditional_binary_cross_entropy'] < .05
    assert not checks['conditional_ce_size_23']
    assert checks['conditional_ce_size_11'] and checks['conditional_ce_size_17']


@pytest.mark.parametrize('condition_key', ['longest_store_reset_drop', 'longest_gain_over_fast_global',
                                         'longest_gain_over_feedforward', 'longest_gain_over_gru'])
def test_continuation_cannot_pass_without_store_dependence_and_each_control_gain(condition_key):
    records = passing_evaluations()
    target_mode = 'fast_selective' if condition_key == 'longest_store_reset_drop' else condition_key.removeprefix('longest_gain_over_')
    target_condition = 'reset_store' if target_mode == 'fast_selective' else 'intact'
    for record in records:
        if record['mode'] == target_mode:
            for row in record['results']:
                if row['size'] == 23 and row['condition'] == target_condition:
                    row['accuracy2'] = 1.
    assert not study.continuation(records)[1][condition_key]
    with pytest.raises(ValueError, match='all declared'):
        study.continuation(records[:-1])


@pytest.mark.parametrize('mode', study.PROTOCOL['modes'])
def test_postfit_telemetry_never_uses_targets_as_policy_inputs_or_changes_weights(mode):
    data, metadata = study.teacher.training_packet()
    model = study.teacher.new_model(mode, 7)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    original = study.telemetry(model, data, metadata)
    changed_data = {**data, 'labels': 1 - data['labels']}
    changed_metadata = copy.deepcopy(metadata)
    for context in changed_metadata['contexts']:
        context['target_turn'] = 1 - context['target_turn']
    assert study.telemetry(model, changed_data, changed_metadata) == original
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    assert len(original['per_step']) == 9
    for row in original['per_step']:
        if mode.startswith('fast_'):
            assert len(row['write_strengths']) == 4
            assert all(0 <= value <= 1 for value in row['write_strengths'])
            assert row['store_opposite_cue_frobenius'] >= 0
        else:
            assert row['write_strengths'] is None and row['store_opposite_cue_frobenius'] is None
