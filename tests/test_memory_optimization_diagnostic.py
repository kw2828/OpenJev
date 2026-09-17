import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip('minigrid', reason='Install minigrid==3.0.0 for the optimization diagnostic')

spec = importlib.util.spec_from_file_location(
    '_memory_optimization_test', Path(__file__).resolve().parents[1]/'scripts/memory_optimization_diagnostic.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def fixture():
    data, metadata = study.teacher.training_packet()
    model = study.teacher.new_model('gru', 101)
    obs = torch.from_numpy(data['observations'])
    previous = torch.from_numpy(data['previous'])
    resets = torch.from_numpy(data['resets'])
    return model, data, metadata, obs, previous, resets


def test_recursive_path_matches_original_model_exactly():
    model, _, _, obs, previous, resets = fixture()
    expected = model.sequence(obs, previous, model.initial_state(4), resets)
    actual = study.forward_sequence(model, obs, previous, resets, 'recursive')
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_both_paths_use_identical_model_initialization_and_first_observation_state():
    model, data, _, _, _, _ = fixture()
    second = study.teacher.new_model('gru', 101)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    recursive = study.outputs(model, data, 'recursive')
    cached = study.outputs(second, data, 'cache_h0')
    for a, b in zip(recursive, cached, strict=True):
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)


def test_cache_h0_retains_gradient_to_first_observation_but_not_intermediate_frames():
    model, data, _, obs, previous, resets = fixture()
    obs = obs.clone().requires_grad_()
    logits = study.forward_sequence(model, obs, previous, resets, 'cache_h0')[0][-1]
    study.objective(logits, torch.from_numpy(data['labels']), 'ce2').backward()
    assert obs.grad[0].abs().sum() > 0
    assert obs.grad[-1].abs().sum() > 0
    assert obs.grad[1:-1].count_nonzero() == 0
    assert torch.isfinite(obs.grad).all()
    assert model.memory.weight_hh.grad.abs().sum() > 0


def test_cached_final_output_equals_direct_first_then_fork_computation():
    model, _, _, obs, previous, resets = fixture()
    first = model.observe(obs[0], previous[0], model.initial_state(4), resets[0])[2]
    expected = model.observe(obs[-1], previous[-1], first, resets[-1])
    actual = study.forward_sequence(model, obs, previous, resets, 'cache_h0')
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a[-1], b, rtol=0, atol=0)


@pytest.mark.parametrize('path', study.PATHS)
def test_prefix_is_causal_and_labels_do_not_enter_model(path):
    model, data, _, obs, previous, resets = fixture()
    original = study.outputs(model, data, path)
    changed_data = {**data, 'labels': 1-data['labels']}
    for a, b in zip(study.outputs(model, changed_data, path), original, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    changed = obs.clone()
    changed[4:] = torch.randn_like(changed[4:])
    altered = study.forward_sequence(model, changed, previous, resets, path)
    for a, b in zip(original, altered, strict=True):
        torch.testing.assert_close(a[:4], b[:4], rtol=0, atol=0)


@pytest.mark.parametrize('path', study.PATHS)
def test_fork_ablation_erases_cue_and_matches_reactive_fork_without_resetting_action(path):
    model, _, _, obs, previous, resets = fixture()
    reference = model.observe(obs[-1], previous[-1], model.initial_state(4), resets[-1])
    actual = study.forward_sequence(model, obs, previous, resets, path, ablate_fork=True)
    for a, b in zip(actual, reference, strict=True):
        torch.testing.assert_close(a[-1], b, rtol=0, atol=0)
    torch.testing.assert_close(actual[0][-1, 0], actual[0][-1, 2], rtol=0, atol=0)
    torch.testing.assert_close(actual[0][-1, 1], actual[0][-1, 3], rtol=0, atol=0)


def test_binary_loss_excludes_five_unused_logits_without_changing_model_outputs():
    model, data, _, _, _, _ = fixture()
    logits = study.outputs(model, data, 'recursive')[0][-1]
    assert logits.shape == (4, 7)
    logits.retain_grad()
    study.objective(logits, torch.from_numpy(data['labels']), 'ce2').backward()
    assert logits.grad[:, :2].abs().sum() > 0
    assert logits.grad[:, 2:].count_nonzero() == 0
    assert model.actor.weight.grad[2:].count_nonzero() == 0


def test_partial_episode_resets_are_rejected_instead_of_leaking_cached_state():
    model, _, _, obs, previous, resets = fixture()
    resets = resets.clone()
    resets[3, 0] = True
    with pytest.raises(ValueError, match='single-episode'):
        study.forward_sequence(model, obs, previous, resets, 'cache_h0')
