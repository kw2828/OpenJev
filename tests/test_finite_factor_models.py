"""Fabricated factorial tests with an independently derived eight-state oracle."""
from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from openjev.research.finite_factor_models import ARMS, make_model


def inputs(horizon=8):
    prefix = torch.zeros((2, 3, 31), dtype=torch.float32)
    lengths = torch.tensor([2, 3], dtype=torch.int64)
    prefix[:, 0, 4] = prefix[:, 0, 9] = 1
    prefix[:, 1, 1] = prefix[:, 1, 5] = 1
    prefix[1, 2, 2] = prefix[1, 2, 6] = 1
    actions = torch.arange(2 * horizon, dtype=torch.int64).reshape(2, horizon) % 4
    observations = (actions + 1) % 4
    oracle = torch.stack((torch.arange(1, 9, dtype=torch.float64) / 36,
                          torch.arange(8, 0, -1, dtype=torch.float64) / 36))
    return prefix, lengths, actions, observations, oracle


def reference():
    branches = torch.empty((4, 4, 8, 8), dtype=torch.float64)
    found = torch.zeros((4, 8), dtype=torch.float64)
    costs = torch.empty((4, 8), dtype=torch.float64)
    for action in range(4):
        for source in range(8):
            destination = (source ^ 1, (source + 1) % 8, (2 * source) % 8 + source // 4, source ^ 4)[action]
            terminal = Fraction(0)
            for target in range(8):
                probability = Fraction(1, 400) + (Fraction(49, 50) if target == destination else 0)
                hazard = Fraction(1 + ((target // 4) ^ (action % 2)), 200)
                terminal += probability * hazard
                for odor in range(4):
                    likelihood = Fraction(22, 25) if odor == target % 4 else Fraction(1, 25)
                    branches[action, odor, target, source] = float(probability * (1 - hazard) * likelihood)
            found[action, source] = float(terminal)
            costs[action, source] = -.75 if action == ((source ^ (source >> 1)) % 4) else .25
    return branches, found, costs


def kwargs(arm, oracle):
    return {'oracle_prefix': oracle} if arm.startswith('exact_') else {}


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('horizon', (2, 8))
def test_actual_parameter_counts_and_shapes(arm, horizon):
    model = make_model(arm, 91)
    prefix, lengths, actions, observations, oracle = inputs(horizon)
    blind = model.blind_rollout(prefix, lengths, actions, **kwargs(arm, oracle))
    observed = model.observed_rollout(prefix, lengths, actions, observations, **kwargs(arm, oracle))
    assert 'probabilities' not in blind
    for result in (blind, observed):
        assert result['cost_contrasts'].shape == (2, horizon, 4)
        assert result['prior_states'].shape == (2, horizon, 8)
        assert result['cost_contrasts'].dtype == torch.float64
        assert result['survival_mass'].shape == (2, horizon)
        assert result['cost_contrasts'].sum(-1).abs().max() < 1e-14
    assert observed['probabilities'].shape == (2, horizon, 5)
    torch.testing.assert_close(observed['probabilities'].sum(-1), torch.ones((2, horizon), dtype=torch.float64),
                               rtol=0, atol=1e-13)
    count = {'exact_exact': 0, 'learned_exact': 5356, 'exact_learned': 1056, 'learned_learned': 6412}[arm]
    size = {'exact_exact': 0, 'learned_exact': 22352, 'exact_learned': 8448, 'learned_learned': 30800}[arm]
    meta = model.parameter_metadata()
    assert meta['count'] == meta['trainable_count'] == count
    assert sum(p.numel() for p in model.parameters()) == count
    assert meta['parameter_bytes'] == size
    assert meta['buffer_bytes'] == (8704 if arm.endswith('_exact') else 256)
    assert all('readout' not in name for name, _ in model.named_parameters())
    if arm.startswith('exact_'):
        assert all(not name.startswith(('assimilation.', 'projection.')) for name, _ in model.named_parameters())
        assert observed['work']['prefix_assimilation_rows'] == 0
        assert observed['work']['privileged_prefix_rows'] == 2


def test_exact_reference_matches_independent_blind_and_sequential_filter():
    model = make_model('exact_exact', 92)
    prefix, lengths, actions, observations, oracle = inputs()
    observations[0, 3:] = 4
    blind = model.blind_rollout(prefix, lengths, actions, oracle_prefix=oracle)
    observed = model.observed_rollout(prefix, lengths, actions, observations, oracle_prefix=oracle)
    b, found, costs = reference()
    for lane in range(2):
        bs, os = oracle[lane].clone(), oracle[lane].clone()
        absorbed = False
        for h, action in enumerate(actions[lane]):
            bs = b[action].sum(0) @ bs
            torch.testing.assert_close(blind['prior_states'][lane, h], bs, rtol=0, atol=2e-15)
            torch.testing.assert_close(blind['cost_contrasts'][lane, h], costs @ bs, rtol=0, atol=2e-15)
            if absorbed:
                torch.testing.assert_close(observed['probabilities'][lane, h],
                    torch.tensor([0., 0., 0., 0., 1.], dtype=torch.float64), rtol=0, atol=0)
                assert observed['cost_contrasts'][lane, h].count_nonzero() == 0
                continue
            branches = b[action] @ os
            p = torch.cat((branches.sum(-1), (found[action] @ os).reshape(1)))
            prior = branches.sum(0)
            torch.testing.assert_close(observed['probabilities'][lane, h], p, rtol=0, atol=2e-15)
            torch.testing.assert_close(observed['cost_contrasts'][lane, h], costs @ prior, rtol=0, atol=2e-15)
            label = observations[lane, h]
            if label == 4:
                absorbed = True
                os = torch.zeros_like(os)
            else:
                os = branches[label] / p[label]
            torch.testing.assert_close(observed['posterior_states'][lane, h], os, rtol=0, atol=2e-15)
    assert not list(model.parameters())
    assert not blind['cost_contrasts'].requires_grad


def test_pair_initialization_is_independent_of_other_factor_and_preserves_rng():
    before = torch.random.get_rng_state().clone()
    models = {arm: make_model(arm, 314) for arm in ARMS}
    assert torch.equal(before, torch.random.get_rng_state())
    a, b = models['learned_exact'].state_dict(), models['learned_learned'].state_dict()
    names = [name for name in a if name.startswith(('assimilation.', 'projection.'))]
    assert len(names) == 6
    assert all(torch.equal(a[name], b[name]) for name in names)
    assert torch.equal(models['exact_learned'].observed_logits, models['learned_learned'].observed_logits)
    generator = torch.Generator(device='cpu').manual_seed(314 ^ 0x9E3779B9)
    expected = .05 * torch.randn((4, 33, 8), generator=generator, dtype=torch.float64)
    assert torch.equal(models['exact_learned'].observed_logits, expected)
    assert models['learned_learned'].parameter_metadata()['prefix_initialization_seed'] == 314
    assert models['learned_learned'].parameter_metadata()['operator_initialization_seed'] == 314 ^ 0x9E3779B9
    for model in models.values():
        assert torch.equal(model.costs, reference()[2])


@pytest.mark.parametrize('arm', ARMS)
def test_oracle_boundary_prevents_privilege_leak_and_owns_storage(arm):
    model = make_model(arm, 88)
    prefix, lengths, actions, _, oracle = inputs(2)
    if arm.startswith('learned_'):
        with pytest.raises(ValueError, match='must not receive oracle'):
            model.blind_rollout(prefix, lengths, actions, oracle_prefix=oracle)
    else:
        with pytest.raises(ValueError, match='requires explicit privileged'):
            model.blind_rollout(prefix, lengths, actions)
        original = oracle.clone()
        encoded = model.encode_prefix(prefix, lengths, oracle_prefix=oracle)
        encoded.fill_(0)
        assert torch.equal(oracle, original)
        for bad in (torch.zeros_like(oracle), -oracle, oracle.float(), oracle[:, :7], oracle * torch.nan):
            with pytest.raises(ValueError):
                model.blind_rollout(prefix, lengths, actions, oracle_prefix=bad)
        # Exact boundary conditioning does not inspect token values as a learned feature.
        changed = prefix.clone()
        changed[:, :, 10] = 3
        one = model.blind_rollout(prefix, lengths, actions, oracle_prefix=oracle)
        two = model.blind_rollout(changed, lengths, actions, oracle_prefix=oracle)
        assert torch.equal(one['cost_contrasts'], two['cost_contrasts'])


@pytest.mark.parametrize('arm', ARMS)
def test_prelabel_causality_found_absorption_and_padding(arm):
    model = make_model(arm, 42)
    prefix, lengths, actions, observations, oracle = inputs()
    args = kwargs(arm, oracle)
    before = model.observed_rollout(prefix, lengths, actions, observations, **args)
    changed = observations.clone()
    changed[:, 2:] = 4
    after = model.observed_rollout(prefix, lengths, actions, changed, **args)
    for name in ('cost_contrasts', 'probabilities', 'survival_mass'):
        assert torch.equal(before[name][:, :3], after[name][:, :3])
    assert after['cost_contrasts'][:, 3:].count_nonzero() == 0
    assert after['survival_mass'][:, 3:].count_nonzero() == 0
    assert torch.equal(after['probabilities'][:, 3:, 4], torch.ones((2, 5), dtype=torch.float64))
    assert after['probabilities'][:, 3:, :4].count_nonzero() == 0
    padded = prefix.clone()
    padded[0, 2] = torch.nan
    poison = model.observed_rollout(padded, lengths, actions, observations, **args)
    assert torch.equal(before['cost_contrasts'], poison['cost_contrasts'])
    changed[0, 4] = 1
    with pytest.raises(ValueError):
        model.observed_rollout(prefix, lengths, actions, changed, **args)


@pytest.mark.parametrize('arm', ('learned_exact', 'exact_learned', 'learned_learned'))
def test_combined_loss_updates_only_declared_parameters_and_not_fixed_buffers(arm):
    model = make_model(arm, 124)
    prefix, lengths, actions, observations, oracle = inputs(2)
    args = kwargs(arm, oracle)
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    fixed = {name: b.detach().clone() for name, b in model.named_buffers()}
    blind = model.blind_rollout(prefix, lengths, actions, **args)
    observed = model.observed_rollout(prefix, lengths, actions, observations, **args)
    target = torch.tensor([.1, -.2, .3, -.2], dtype=torch.float64)
    events = torch.tensor([.1, .2, .3, .25, .15], dtype=torch.float64)
    loss = ((blind['cost_contrasts'] - target).square().mean()
            + (observed['cost_contrasts'] + target).square().mean()
            + .5 * ((blind['survival_mass'] - .9).square().mean()
                    + (observed['survival_mass'] - .95).square().mean())
            - (events * observed['probabilities'].log()).sum(-1).mean())
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    for name, parameter in model.named_parameters():
        assert not torch.equal(initial[name], parameter), name
    for name, buffer in model.named_buffers():
        assert torch.equal(fixed[name], buffer), name


@pytest.mark.parametrize('arm', ARMS)
def test_exported_operator_storage_cannot_mutate_parameters_or_buffers(arm):
    model = make_model(arm, 51)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    exported = model.operators()
    with torch.no_grad():
        exported['observed'].zero_()
        exported['found'].zero_()
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


@pytest.mark.parametrize('arm,seed', [('other', 0), ('exact_exact', True), ('learned_exact', -1)])
def test_invalid_configuration(arm, seed):
    with pytest.raises(ValueError):
        make_model(arm, seed)
