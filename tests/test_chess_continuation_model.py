import random

import chess
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from openjev.research.chess_anchor import AnchorChess
from openjev.research.chess_continuation_model import (
    CHECKPOINT_VERSION,
    OBJECTIVES,
    ContinuationChess,
    loss_components,
    training_objective,
)
from openjev.research.chess_spatial import encode_board, encode_candidates


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def batch():
    boards = [chess.Board(), chess.Board('7k/8/8/8/8/8/p7/7K b - - 0 1')]
    encoded = [encode_candidates(board) for board in boards]
    moves = max(len(ids) for ids, _ in encoded)
    candidates = torch.zeros(2, moves, 5, dtype=torch.long)
    mask = torch.zeros(2, moves, dtype=torch.bool)
    for index, (ids, features) in enumerate(encoded):
        candidates[index, :len(ids)] = torch.from_numpy(features)
        mask[index, :len(ids)] = True
    observations = torch.from_numpy(np.stack([encode_board(board) for board in boards]))
    return observations, candidates, mask


@pytest.mark.parametrize('seed', [17, 29])
def test_exact_actor_initialization_outputs_and_shared_critic_across_arms(seed):
    actor = AnchorChess('residual', seed=seed, width=32, depth=4)
    data = batch()
    expected = actor(*data)
    reference_critic = None
    for objective in OBJECTIVES:
        model = ContinuationChess(objective, seed=seed)
        assert model.forward.__func__ is AnchorChess.forward
        assert model.choose.__func__ is AnchorChess.choose
        for key, value in actor.state_dict().items():
            assert torch.equal(value, model.state_dict()[key]), key
        for observed, reference in zip(model(*data), expected, strict=True):
            assert torch.equal(observed, reference)
        critic = model.candidate_critic.state_dict()
        if reference_critic is None:
            reference_critic = critic
        assert all(torch.equal(value, reference_critic[key]) for key, value in critic.items())


def test_global_rng_unchanged_for_initialization_and_checkpoint_operations(tmp_path):
    torch.manual_seed(822)
    before = torch.random.get_rng_state().clone()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    device_type = 'mps' if torch.backends.mps.is_available() else 'cuda'
    device = torch.get_device_module(device_type)
    accelerator_states = [device.get_rng_state(i).clone() for i in range(device.device_count())]
    model = ContinuationChess('continuation', seed=29)
    path = tmp_path/'model.pt'; model.save(path, plan_sha256='a'*64)
    ContinuationChess.load(path, expected_plan_sha256='a'*64, expected_seed=29,
                           expected_objective='continuation')
    assert torch.equal(before, torch.random.get_rng_state())
    assert python_state == random.getstate()
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0]
    assert np.array_equal(numpy_state[1], after_numpy[1])
    assert numpy_state[2:] == after_numpy[2:]
    assert all(torch.equal(state, device.get_rng_state(i)) for i, state in enumerate(accelerator_states))


def test_critic_uses_exact_shared_policy_activation_and_propagates_into_trunk():
    model = ContinuationChess('continuation', seed=17)
    data = batch(); captured = []
    hook = model.policy_head[1].register_forward_hook(lambda _m, _i, value: captured.append(value))
    _, _, hidden = model(*data)
    hook.remove()
    values = model.candidate_values(hidden, data[1])
    assert torch.equal(values, torch.tanh(model.candidate_critic(captured[0])).squeeze(-1))
    targets = torch.zeros(2, dtype=torch.long)
    errors = (values[torch.arange(2), targets]-torch.tensor([.6, -.4])).square().mean()
    errors.backward()
    for module in (model.encoder[0], model.core.conv1, model.core.conv2,
                   model.policy_head[0], model.candidate_critic):
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all() and module.weight.grad.abs().sum() > 0
    assert model.policy_head[2].weight.grad is None
    assert all(p.grad is None for p in model.value_head.parameters())
    assert all(p.grad is None for p in model.aux_head.parameters())


@pytest.mark.parametrize('objective', OBJECTIVES)
def test_inference_is_actor_only_and_parameter_counts_are_explicit(objective, monkeypatch):
    model = ContinuationChess(objective, seed=17)
    actor = AnchorChess('residual', seed=17)
    def forbidden(*args, **kwargs):
        raise AssertionError('Inference evaluated the critic')
    monkeypatch.setattr(model, 'candidate_values', forbidden)
    monkeypatch.setattr(model.candidate_critic, 'forward', forbidden)
    expected, observed = actor.choose(chess.Board()), model.choose(chess.Board())
    for key in expected:
        if key != 'latency_ms':
            assert observed[key] == expected[key]
    counts = model.parameter_report()
    assert counts['stored_actor_parameters'] == actor.parameter_count() == 43726
    assert counts['stored_critic_parameters'] == 65
    assert counts['stored_parameters'] == 43791
    assert counts['inference_active_critic_parameters'] == 0
    assert counts['training_active_critic_parameters'] == (0 if objective == 'policy' else 65)
    assert counts['training_active_parameters'] == (
        counts['stored_parameters']-counts['unused_board_head_parameters']
        -(65 if objective == 'policy' else 0))


def test_padding_and_menu_permutation_preserve_actor_and_critic():
    model = ContinuationChess('continuation')
    data = batch(); before = [value.clone() for value in data]
    logits, root, hidden = model(*data)
    values = model.candidate_values(hidden, data[1])
    permutation = torch.arange(data[1].shape[1]-1, -1, -1)
    shuffled = model(data[0], data[1][:, permutation], data[2][:, permutation])
    torch.testing.assert_close(shuffled[0], logits[:, permutation])
    assert torch.equal(shuffled[1], root) and torch.equal(shuffled[2], hidden)
    torch.testing.assert_close(model.candidate_values(hidden, data[1][:, permutation]), values[:, permutation])
    assert torch.isneginf(logits[~data[2]]).all()
    assert (logits.softmax(-1)[~data[2]] == 0).all()
    assert all(torch.equal(value, old) for value, old in zip(data, before, strict=True))


def arithmetic():
    return {'logits': torch.tensor([[.2, -.4, .1], [.3, .1, -.2], [-.2, .7, .3]], requires_grad=True),
            'root_predictions': torch.tensor([.2, -.1, .3], requires_grad=True),
            'candidate_predictions': torch.tensor([[.1, -.2, .3], [.4, -.3, .2], [.1, .6, -.5]], requires_grad=True),
            'legal_mask': torch.ones(3, 3, dtype=torch.bool), 'targets': torch.tensor([0, 1, 2]),
            'teacher_values': torch.tensor([-.4, .2, .9]), 'behavior_indices': torch.tensor([1, 1, -1]),
            'continuation_values': torch.tensor([.9, -.9, torch.nan], requires_grad=True),
            'continuation_mask': torch.tensor([True, True, False])}


def test_unique_action_arithmetic_per_root_normalization_and_conflicting_labels():
    data = arithmetic()
    result = loss_components('continuation', **data)
    # Root 0 has two actions, root 1 has a duplicate, root 2 has no continuation.
    expected = torch.tensor([((.1+.4)**2+(-.2-.9)**2)/2, (-.3-.2)**2, (-.5-.9)**2]).mean()
    torch.testing.assert_close(result['candidate_value_mse'], expected)
    flat_mean = torch.tensor([(.1+.4)**2, (-.2-.9)**2, (-.3-.2)**2, (-.5-.9)**2]).mean()
    assert not torch.isclose(result['candidate_value_mse'], flat_mean)
    ce = F.cross_entropy(data['logits'], data['targets'])
    root_mse = F.mse_loss(data['root_predictions'], data['teacher_values'])
    torch.testing.assert_close(result['total'], ce+.5*root_mse+expected)
    assert result['teacher_targets_used'] == 3
    assert result['continuation_targets_used'] == 1
    assert result['supervised_targets_used'] == 4
    result['total'].backward()
    assert torch.isfinite(data['continuation_values'].grad).all()
    assert data['continuation_values'].grad[0] != 0
    assert (data['continuation_values'].grad[1:] == 0).all()
    assert data['candidate_predictions'].grad[1, 1] == pytest.approx(2*(-.3-.2)/3)


def test_all_missing_nan_labels_and_all_duplicate_actions_equal_best_value_exactly():
    data = arithmetic()
    best = loss_components('best_value', **data)
    for mask, indices, values in [
        (torch.zeros(3, dtype=torch.bool), torch.full((3,), -1), torch.full((3,), torch.nan)),
        (torch.ones(3, dtype=torch.bool), data['targets'], torch.tensor([1., -1., -.8])),
    ]:
        changed = {**data, 'continuation_mask': mask, 'behavior_indices': indices, 'continuation_values': values}
        result = loss_components('continuation', **changed)
        assert torch.equal(result['total'], best['total'])
        assert torch.equal(result['candidate_value_mse'], best['candidate_value_mse'])
        assert result['continuation_targets_used'] == 0 and result['supervised_targets_used'] == 3
    no_fields = {k: v for k, v in data.items() if k not in ('behavior_indices', 'continuation_values', 'continuation_mask')}
    assert torch.equal(loss_components('continuation', **no_fields)['total'], best['total'])


@pytest.mark.parametrize('objective', OBJECTIVES)
def test_training_gradients_and_policy_critic_skipping(objective, monkeypatch):
    model = ContinuationChess(objective)
    if objective == 'policy':
        def forbidden(*args):
            raise AssertionError('Policy training evaluated critic')
        monkeypatch.setattr(model, 'candidate_values', forbidden)
    result = training_objective(model, *batch(), torch.tensor([0, 1]), torch.tensor([.6, -.4]),
                                torch.tensor([1, -1]), torch.tensor([-.7, torch.nan]),
                                torch.tensor([True, False]))
    assert result['total'].requires_grad and torch.isfinite(result['total'])
    result['total'].backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert model.policy_head[0].weight.grad.abs().sum() > 0
    assert model.policy_head[2].weight.grad.abs().sum() > 0
    assert model.value_head[2].weight.grad.abs().sum() > 0
    critic_grad = model.candidate_critic.weight.grad
    assert (critic_grad is None) == (objective == 'policy')
    if critic_grad is not None:
        assert torch.isfinite(critic_grad).all() and critic_grad.abs().sum() > 0
    assert all(p.grad is None for p in model.aux_head.parameters())
    assert result['teacher_targets_used'] == (0 if objective == 'policy' else 2)
    assert result['continuation_targets_used'] == (1 if objective == 'continuation' else 0)


@pytest.mark.parametrize(('field', 'value'), [
    ('targets', torch.tensor([-1, 1, 2])), ('targets', torch.tensor([0, 3, 2])),
    ('targets', torch.tensor([0., 1., 2.])), ('targets', torch.tensor([0, 1])),
    ('teacher_values', torch.tensor([0., torch.nan, .1])),
    ('teacher_values', torch.tensor([0., 1.01, .1])),
    ('continuation_values', torch.tensor([torch.nan, 0., torch.nan])),
    ('continuation_values', torch.tensor([float('inf'), 0., torch.nan])),
    ('continuation_values', torch.tensor([1.1, 0., torch.nan])),
    ('continuation_mask', torch.ones(3)),
    ('continuation_mask', torch.ones(2, dtype=torch.bool)),
    ('behavior_indices', torch.tensor([-1, 1, -1])),
    ('behavior_indices', torch.tensor([1, 3, -1])),
    ('behavior_indices', torch.tensor([1, 1, -2])),
    ('root_predictions', torch.tensor([.1, 1.1, .2])),
    ('candidate_predictions', torch.zeros(3, 2)),
    ('teacher_values', torch.zeros(3, dtype=torch.float64)),
    ('root_predictions', torch.zeros(3, dtype=torch.float64)),
    ('candidate_predictions', torch.zeros(3, 3, dtype=torch.float64)),
    ('continuation_values', torch.zeros(3, dtype=torch.float64)),
])
def test_invalid_loss_inputs_rejected(field, value):
    data = arithmetic(); data[field] = value
    with pytest.raises(ValueError):
        loss_components('continuation', **data)


def test_illegal_padding_and_partial_continuation_fields_rejected():
    data = arithmetic(); data['legal_mask'][0, 1] = False
    with pytest.raises(ValueError, match='Legal mask/logits'):
        loss_components('continuation', **data)
    data['logits'] = data['logits'].detach().masked_fill(~data['legal_mask'], -torch.inf)
    with pytest.raises(ValueError, match='Behavior indices'):
        loss_components('continuation', **data)
    data['targets'][0] = 1
    with pytest.raises(ValueError, match='Teacher targets include'):
        loss_components('continuation', **data)
    data = arithmetic(); data['continuation_values'] = None
    with pytest.raises(ValueError, match='all three'):
        loss_components('continuation', **data)
    data = arithmetic()
    with pytest.raises(ValueError, match='must not compute'):
        loss_components('policy', **data)


def test_invalid_model_inputs_and_configuration_rejected():
    for objective in ('unknown', True, None):
        with pytest.raises(ValueError):
            ContinuationChess(objective)
    for seed in (True, -1, 2**63, 1.5):
        with pytest.raises(ValueError):
            ContinuationChess(seed=seed)
    model = ContinuationChess('best_value'); data = batch()
    hidden = model(*data)[2]
    with pytest.raises(ValueError):
        model.candidate_values(hidden[:, :1], data[1])
    with pytest.raises(ValueError):
        model.candidate_values(hidden, data[1].float())
    with pytest.raises(ValueError):
        model.candidate_values(hidden*torch.nan, data[1])
    data[0][0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match='Observations'):
        training_objective(model, *data, torch.tensor([0, 0]), torch.zeros(2))


@pytest.mark.parametrize('objective', OBJECTIVES)
def test_checkpoint_roundtrip_identity_and_no_overwrite(tmp_path, objective):
    model = ContinuationChess(objective, seed=29)
    path = tmp_path/'model.pt'; model.save(path, plan_sha256='a'*64)
    loaded = ContinuationChess.load(path, expected_plan_sha256='a'*64, expected_seed=29,
                                    expected_objective=objective)
    assert not loaded.training
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())
    for left, right in zip(model(*batch()), loaded(*batch()), strict=True):
        assert torch.equal(left, right)
    with pytest.raises(FileExistsError):
        model.save(path, plan_sha256='a'*64)
    for kwargs in ({'expected_plan_sha256': 'b'*64}, {'expected_seed': 17},
                   {'expected_objective': 'policy' if objective != 'policy' else 'continuation'},
                   {'expected_critic_seed': 9}):
        expected = {'expected_plan_sha256': 'a'*64, 'expected_seed': 29, 'expected_objective': objective, **kwargs}
        with pytest.raises(ValueError, match='identity mismatch'):
            ContinuationChess.load(path, **expected)
    with pytest.raises(ValueError):
        AnchorChess.load(path)


@pytest.mark.parametrize(('field', 'value'), [
    ('format_version', 'old'), ('encoding', 'other'), ('objective', 'policy'),
    ('recurrence', 'anchor'), ('seed', True), ('critic_seed', 17),
    ('width', 16), ('depth', 8), ('plan_sha256', 'b'*64), ('extra', 'field'),
])
def test_corrupted_checkpoint_metadata_rejected(tmp_path, field, value):
    path = tmp_path/'model.pt'; ContinuationChess('continuation', seed=17).save(path, plan_sha256='a'*64)
    payload = torch.load(path, weights_only=True)
    assert payload['format_version'] == CHECKPOINT_VERSION
    payload[field] = value; torch.save(payload, path)
    with pytest.raises(ValueError, match='metadata or requested identity'):
        ContinuationChess.load(path, expected_plan_sha256='a'*64, expected_seed=17, expected_objective='continuation')


@pytest.mark.parametrize('corruption', ['missing', 'extra', 'shape', 'dtype', 'nan', 'nontensor'])
def test_corrupted_checkpoint_tensors_rejected(tmp_path, corruption):
    path = tmp_path/'model.pt'; ContinuationChess().save(path, plan_sha256='a'*64)
    payload = torch.load(path, weights_only=True); state = payload['state_dict']
    key = 'candidate_critic.weight'
    if corruption == 'missing':
        del state[key]
    elif corruption == 'extra':
        state['extra'] = torch.tensor(0.)
    elif corruption == 'shape':
        state[key] = state[key][:, :1]
    elif corruption == 'dtype':
        state[key] = state[key].double()
    elif corruption == 'nan':
        state[key][0, 0] = torch.nan
    else:
        state[key] = 'not a tensor'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='checkpoint'):
        ContinuationChess.load(path, expected_plan_sha256='a'*64, expected_seed=17, expected_objective='policy')


def test_nonfinite_save_and_old_checkpoint_rejected(tmp_path):
    model = ContinuationChess('best_value')
    with torch.no_grad():
        model.candidate_critic.weight[0, 0] = torch.nan
    with pytest.raises(ValueError, match='checkpoint tensor'):
        model.save(tmp_path/'invalid.pt', plan_sha256='a'*64)
    assert not (tmp_path/'invalid.pt').exists()
    old = tmp_path/'old.pt'; AnchorChess().save(old, plan_sha256='a'*64)
    with pytest.raises(ValueError, match='metadata'):
        ContinuationChess.load(old, expected_plan_sha256='a'*64, expected_seed=17, expected_objective='policy')


def test_custom_private_seed_keeps_actor_identical_and_requires_checkpoint_binding(tmp_path):
    first = ContinuationChess('continuation', seed=17)
    custom = ContinuationChess('continuation', seed=17, critic_seed=123)
    for key, value in first.state_dict().items():
        if not key.startswith('candidate_critic.'):
            assert torch.equal(value, custom.state_dict()[key])
    assert not torch.equal(first.candidate_critic.weight, custom.candidate_critic.weight)
    path = tmp_path/'custom.pt'; custom.save(path, plan_sha256='a'*64)
    with pytest.raises(ValueError, match='identity'):
        ContinuationChess.load(path, expected_plan_sha256='a'*64, expected_seed=17, expected_objective='continuation')
    restored = ContinuationChess.load(path, expected_plan_sha256='a'*64, expected_seed=17,
                                     expected_objective='continuation', expected_critic_seed=123)
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in custom.state_dict().items())
