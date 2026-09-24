"""Fabricated exact-reference tests; no scientific namespaces or generation."""
from __future__ import annotations

import hashlib
from fractions import Fraction

import pytest
import torch

from openjev.research.finite_factor_models import make_model
from openjev.research.finite_regime_reference import VERSION, make_reference


def rational_laws(epsilon):
    eps = Fraction(str(epsilon))
    b = [[[[Fraction(0) for _ in range(8)] for _ in range(8)] for _ in range(4)] for _ in range(4)]
    found = [[Fraction(0) for _ in range(8)] for _ in range(4)]
    c = [[Fraction(-3, 4) if d == ((s ^ (s >> 1)) & 3) else Fraction(1, 4)
          for s in range(8)] for d in range(4)]
    for action in range(4):
        for current in range(8):
            nxt = (current ^ 1, (current + 1) % 8,
                   (2 * current) % 8 + current // 4, current ^ 4)[action]
            for destination in range(8):
                t = Fraction(1, 400) + (Fraction(49, 50) if destination == nxt else 0)
                h = Fraction(1 + ((destination // 4) ^ (action % 2)), 200)
                found[action][current] += t * h
                for odor in range(4):
                    emission = 1 - eps if odor == destination % 4 else eps / 3
                    b[action][odor][destination][current] = t * (1 - h) * emission
    a = [[[sum(b[action][odor][n][s] for odor in range(4)) for s in range(8)]
          for n in range(8)] for action in range(4)]
    return b, found, a, c


def vector(matrix, state):
    return [sum(x * y for x, y in zip(row, state, strict=True)) for row in matrix]


def tensor(values):
    if isinstance(values, list):
        return torch.tensor(convert(values), dtype=torch.float64)
    return torch.tensor(float(values), dtype=torch.float64)


def convert(values):
    return [convert(x) if isinstance(x, list) else float(x) for x in values]


def inputs(horizon=8):
    prefix = torch.zeros((2, 9, 31), dtype=torch.float32)
    prefix[:, 0, 4] = prefix[:, 0, 9] = 1
    for step in range(1, 9):
        prefix[:, step, step % 4] = prefix[:, step, 4 + (step + 1) % 4] = 1
    lengths = torch.tensor([9, 9], dtype=torch.int64)
    actions = torch.arange(2 * horizon, dtype=torch.int64).reshape(2, horizon) % 4
    observations = (actions + 1) % 4
    if horizon > 2:
        observations[0, 2:] = 4
    states = [[Fraction(i + 1, 36) for i in range(8)],
              [Fraction(8 - i, 36) for i in range(8)]]
    return prefix, lengths, actions, observations, tensor(states), states


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


@pytest.mark.parametrize('horizon', (2, 8))
def test_base_matches_frozen_reference_fields_and_work_bitwise(horizon):
    prefix, lengths, actions, observations, oracle, _ = inputs(horizon)
    old, new = make_model('exact_exact', 0), make_reference(.12)
    for route in ('blind_rollout', 'observed_rollout'):
        args = (prefix, lengths, actions) + ((observations,) if route.startswith('observed') else ())
        expected = getattr(old, route)(*args, oracle_prefix=oracle)
        actual = getattr(new, route)(*args, oracle_prefix=oracle)
        assert expected.keys() == actual.keys()
        for name, value in expected.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, actual[name]), name
            else:
                assert value == actual[name]


@pytest.mark.parametrize('epsilon', (.12, .30))
def test_regime_buffers_and_every_observed_step_match_rational_law(epsilon):
    b, found, a, c = rational_laws(epsilon)
    model = make_reference(epsilon)
    torch.testing.assert_close(model.known_observed, tensor(b), rtol=0, atol=2e-16)
    torch.testing.assert_close(model.known_found, tensor(found), rtol=0, atol=2e-17)
    assert torch.equal(model.costs, tensor(c))
    prefix, lengths, actions, observations, oracle, initial = inputs()
    blind = model.blind_rollout(prefix, lengths, actions, oracle_prefix=oracle)
    observed = model.observed_rollout(prefix, lengths, actions, observations, oracle_prefix=oracle)
    for case in range(2):
        bs, os, absorbed = initial[case], initial[case], False
        for step in range(8):
            action, odor = int(actions[case, step]), int(observations[case, step])
            bs = vector(a[action], bs)
            torch.testing.assert_close(blind['prior_states'][case, step], tensor(bs), rtol=0, atol=1e-12)
            torch.testing.assert_close(blind['cost_contrasts'][case, step], tensor(vector(c, bs)), rtol=0, atol=1e-12)
            assert float(blind['survival_mass'][case, step]) == pytest.approx(float(sum(bs)), abs=1e-12)
            if absorbed:
                prior, posterior, probs = [Fraction(0)] * 8, [Fraction(0)] * 8, [Fraction(0)] * 4 + [Fraction(1)]
            else:
                branches = [vector(b[action][o], os) for o in range(4)]
                probs = [sum(branch) for branch in branches] + [sum(x * y for x, y in zip(found[action], os, strict=True))]
                prior = [sum(branch[n] for branch in branches) for n in range(8)]
                posterior = [Fraction(0)] * 8 if odor == 4 else [x / probs[odor] for x in branches[odor]]
            torch.testing.assert_close(observed['probabilities'][case, step], tensor(probs), rtol=0, atol=1e-12)
            torch.testing.assert_close(observed['prior_states'][case, step], tensor(prior), rtol=0, atol=1e-12)
            torch.testing.assert_close(observed['posterior_states'][case, step], tensor(posterior), rtol=0, atol=1e-12)
            torch.testing.assert_close(observed['cost_contrasts'][case, step], tensor(vector(c, prior)), rtol=0, atol=1e-12)
            assert float(observed['survival_mass'][case, step]) == pytest.approx(float(sum(prior)), abs=1e-12)
            os, absorbed = posterior, absorbed or odor == 4


def test_shift_observed_law_must_change_even_when_blind_marginal_matches():
    prefix, lengths, actions, observations, oracle, _ = inputs()
    base, shift = make_reference(.12), make_reference(.30)
    base_blind = base.blind_rollout(prefix, lengths, actions, oracle_prefix=oracle)
    shift_blind = shift.blind_rollout(prefix, lengths, actions, oracle_prefix=oracle)
    torch.testing.assert_close(base_blind['cost_contrasts'], shift_blind['cost_contrasts'], rtol=0, atol=1e-12)
    base_observed = base.observed_rollout(prefix, lengths, actions, observations, oracle_prefix=oracle)
    shift_observed = shift.observed_rollout(prefix, lengths, actions, observations, oracle_prefix=oracle)
    assert not torch.allclose(base_observed['probabilities'][:, 0], shift_observed['probabilities'][:, 0], atol=1e-12, rtol=0)
    assert not torch.allclose(base_observed['posterior_states'][:, 0], shift_observed['posterior_states'][:, 0], atol=1e-12, rtol=0)
    assert state_hash(base) != state_hash(shift)


def test_forecasts_precede_current_label_and_absorb_found():
    prefix, lengths, actions, observations, oracle, _ = inputs()
    model = make_reference(.30)
    left = model.observed_rollout(prefix, lengths, actions, observations, oracle_prefix=oracle)
    changed = observations.clone()
    changed[1, 3] = (changed[1, 3] + 1) % 4
    right = model.observed_rollout(prefix, lengths, actions, changed, oracle_prefix=oracle)
    for name in ('probabilities', 'cost_contrasts', 'survival_mass', 'prior_states'):
        assert torch.equal(left[name][:, :4], right[name][:, :4])
    assert not torch.equal(left['posterior_states'][1, 3], right['posterior_states'][1, 3])
    assert not left['cost_contrasts'][0, 3:].any()
    assert not left['survival_mass'][0, 3:].any()
    assert torch.equal(left['probabilities'][0, 3:, 4], torch.ones(5, dtype=torch.float64))


@pytest.mark.parametrize('epsilon', (.12, .30))
def test_no_parameters_rng_changes_or_aliases_and_explicit_regime_metadata(epsilon):
    before = torch.random.get_rng_state().clone()
    model, other = make_reference(epsilon), make_reference(epsilon)
    assert torch.equal(before, torch.random.get_rng_state())
    assert not list(model.parameters())
    meta = model.parameter_metadata()
    assert meta['version'] == VERSION and meta['epsilon'] == epsilon
    assert meta['count'] == meta['trainable_count'] == meta['parameter_bytes'] == 0
    assert meta['buffer_bytes'] == 8712
    assert meta['privileged_prefix'] and meta['known_operators'] and meta['fixed_cost_readout']
    assert meta['regime_state_buffer'] == 'regime_epsilon'
    assert set(model.state_dict()) == {'costs', 'known_observed', 'known_found', 'regime_epsilon'}
    for name, value in model.state_dict().items():
        assert value.dtype == torch.float64 and value.device.type == 'cpu' and not value.requires_grad
        assert value.data_ptr() != other.state_dict()[name].data_ptr()
    original = state_hash(model)
    other.load_state_dict(model.state_dict())
    assert state_hash(other) == original
    args = inputs()
    model.observed_rollout(*args[:4], oracle_prefix=args[4])
    assert state_hash(model) == original


def test_explicit_boundary_required_and_cross_regime_state_is_rejected():
    model = make_reference(.30)
    prefix, lengths, actions, _, oracle, _ = inputs()
    with pytest.raises(ValueError, match='requires explicit'):
        model.blind_rollout(prefix, lengths, actions)
    copied = model.encode_prefix(prefix, lengths, oracle_prefix=oracle)
    assert copied.data_ptr() != oracle.data_ptr() and torch.equal(copied, oracle)
    model.load_state_dict(make_reference(.12).state_dict())
    with pytest.raises(ValueError, match='epsilon metadata'):
        model.parameter_metadata()


@pytest.mark.parametrize('epsilon', [True, 0, 1, .2, float('nan'), float('inf'), '.30'])
def test_only_explicit_registered_regimes_are_supported(epsilon):
    with pytest.raises(ValueError, match='explicit epsilon'):
        make_reference(epsilon)
